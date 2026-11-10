//! D-DT64-GEMM-SHAPE-3: the same virtual object over u8 lanes, consumed as
//! `u8 x i8 -> i32` (AMX `TDPBUSD`), next to the BF16 arms in [`crate::shape`].
//!
//! A is the source's lanes as **unsigned** u8; B is the synthetic operand as
//! **signed** i8. That is `TDPBUSD`'s native contract: A is never
//! reinterpreted as signed and no u8/i8 cast is introduced anywhere.
//!
//! The binding is the same [`VirtualObject`]: DT64 does not care what a lane
//! holds. Only the borrow and the arithmetic change.
//!
//! What ndarray's facade offers for INT8 today (audited 2026-11-10):
//! `backend::gemm_i8` -> `simd_int_ops::gemm_u8_i8` -> `int8_gemm_amx_tiled`
//! on 16/16/64-aligned shapes. A must be dense with row stride K, and B is
//! VNNI-packed inside every call. There is no strided-A entry and no prepacked
//! B for INT8 (the BF16 equivalents are `bf16_tile_gemm_16x16_packed_strided`
//! and `PackedBf16B`). So the only INT8 arm here is [`staged`]: a direct
//! (in-place) arm and prepared-B reuse are refused, not emulated.

use std::cell::Cell;

use lance_graph_contract::ontology::FieldRef;
use lance_graph_dt64::dr::{Frame, FrameRevisions, FrameStore, Scope};
use lance_graph_dt64::{CoordSpace, LaneAddr, LaneRefusal, LaneSource, SchemaRev};

use crate::bind::{World, CLASS, READER, SPACE_A};
use crate::fixture::Fixture;
use crate::shape::{BOperand, BValues, Selection, ShapeRefusal, VirtualObject, Where, MAX_ROWS};

/// INT8 tiles consume K in steps of 64.
pub const K_STEP: usize = 64;

/// `A[i][p]` as u8: the fixture value shifted into [0, 16]. Unsigned by
/// construction; never a reinterpreted i8.
pub fn a_u8(fx: &Fixture, i: usize, p: usize) -> u8 {
    u8::try_from(fx.value(i, p) + 8).expect("[-8, 8] + 8 is in u8")
}

/// `B[i][j]` as i8 (the same synthetic operand as the BF16 arms).
pub fn b_i8(b: &BOperand, i: usize, j: usize) -> i8 {
    i8::try_from(b.value(i, j)).expect("[-8, 8] is in i8")
}

/// One borrowed u8 lane: `n` values at `slab[row * pitch..][..n]`.
#[derive(Debug, Clone, Copy)]
pub struct U8Lane<'a> {
    pub slab: &'a [u8],
    pub row: usize,
    pub pitch: usize,
    pub n: usize,
}

impl<'a> U8Lane<'a> {
    /// The lane's values, in the lane's own row order.
    pub fn values(&self) -> &'a [u8] {
        &self.slab[self.row * self.pitch..self.row * self.pitch + self.n]
    }
}

/// The data owner: u8 lanes in one slab at a pitch padded to [`K_STEP`],
/// and the frame store that lives with them.
#[derive(Debug)]
pub struct U8Lanes {
    data: Box<[u8]>,
    pitch: usize,
    n: usize,
    frames: FrameStore,
}

impl U8Lanes {
    /// `fx.props` lanes over `fx.n` identities, field `f` at row `f`.
    pub fn slab_in_order(fx: &Fixture) -> U8Lanes {
        U8Lanes::from_fn(fx.n, fx.props, |i, p| a_u8(fx, i, p))
    }

    /// `props` lanes over `n` identities from encoded u8 values.
    pub fn from_fn(n: usize, props: usize, value: impl Fn(usize, usize) -> u8) -> U8Lanes {
        let pitch = n.div_ceil(K_STEP) * K_STEP;
        let mut data = vec![0u8; props * pitch].into_boxed_slice();
        for p in 0..props {
            for i in 0..n {
                data[p * pitch + i] = value(i, p);
            }
        }
        U8Lanes {
            data,
            pitch,
            n,
            frames: FrameStore::new(FrameRevisions {
                source: 1,
                relation_map: 1,
                coordinates: 1,
            }),
        }
    }

    /// Mutable frame owner, for advancing the source after a change.
    pub fn frames_mut(&mut self) -> &mut FrameStore {
        &mut self.frames
    }

    /// Physical row `r`, borrowed in place.
    pub fn row(&self, r: usize) -> Option<U8Lane<'_>> {
        (r * self.pitch + self.n <= self.data.len()).then_some(U8Lane {
            slab: &self.data,
            row: r,
            pitch: self.pitch,
            n: self.n,
        })
    }
}

/// The DT64 socket over a [`U8Lanes`]: field `f` lives in row `rows[f]`.
pub struct U8Socket<'a> {
    pub lanes: &'a U8Lanes,
    pub rows: Vec<usize>,
    pub space: CoordSpace,
    pub scope: Scope,
    pub borrows: Cell<u64>,
}

impl<'a> U8Socket<'a> {
    /// Field `f` at row `f`, in [`SPACE_A`].
    pub fn in_order(lanes: &'a U8Lanes, props: usize) -> U8Socket<'a> {
        U8Socket {
            lanes,
            rows: (0..props).collect(),
            space: SPACE_A,
            scope: READER,
            borrows: Cell::new(0),
        }
    }
}

impl LaneSource for U8Socket<'_> {
    type Lane<'b>
        = U8Lane<'b>
    where
        Self: 'b;

    fn lane_of(
        &self,
        class: u32,
        rev: SchemaRev,
        field: u8,
        _basis: &FieldRef,
        _frame: &Frame,
    ) -> Option<LaneAddr> {
        let f = usize::from(field);
        if class != CLASS || rev != SchemaRev(1) || f >= self.rows.len() {
            return None;
        }
        Some(LaneAddr {
            space: self.space,
            lane: u16::try_from(self.rows[f]).ok()?,
            scope: self.scope,
        })
    }

    fn borrow(&self, addr: &LaneAddr) -> Option<U8Lane<'_>> {
        self.borrows.set(self.borrows.get() + 1);
        self.lanes.row(usize::from(addr.lane))
    }

    fn frames(&self) -> &FrameStore {
        &self.lanes.frames
    }
}

/// The borrowed u8 lanes of an object, with the frame issue and lane
/// addresses they were resolved under.
#[derive(Debug, Clone)]
pub struct U8Lanes64<'s> {
    lanes: [U8Lane<'s>; MAX_ROWS],
    object: u64,
    frame: Frame,
    count: usize,
}

impl<'s> U8Lanes64<'s> {
    /// The borrowed lanes.
    pub fn lanes(&self) -> &[U8Lane<'s>] {
        &self.lanes[..self.count]
    }
}

/// Warm: [`lance_graph_dt64::LaneBinding::borrow_all`] checks the object's
/// binding once and hands out every bound u8 lane; they are placed in the
/// view's row order. Any refusal returns before a lane is placed. No
/// allocation.
pub fn resolve<'s>(
    obj: &VirtualObject,
    world: &World,
    socket: &'s U8Socket<'_>,
) -> Result<U8Lanes64<'s>, ShapeRefusal> {
    let binding = obj.binding();
    let mut by_field: [Option<U8Lane<'s>>; MAX_ROWS] = [None; MAX_ROWS];
    for item in binding.borrow_all(&world.store, &world.view, socket.frames(), socket)? {
        let (field, lane) = item?;
        by_field[usize::from(field)] = Some(lane);
    }
    let lane = |f: u8| by_field[usize::from(f)].ok_or(LaneRefusal::NotBound { field: f });
    let fields = obj.fields();
    let first = lane(fields[0])?;
    let mut lanes = [first; MAX_ROWS];
    for (slot, &f) in lanes.iter_mut().zip(fields).skip(1) {
        *slot = lane(f)?;
    }
    Ok(U8Lanes64 {
        lanes,
        object: obj.id(),
        frame: binding.frame().clone(),
        count: fields.len(),
    })
}

fn admit(obj: &VirtualObject, lanes: &U8Lanes64<'_>) -> Result<(), ShapeRefusal> {
    let same = lanes.object == obj.id()
        && lanes.count == obj.fields().len()
        && lanes.frame.same_issue(obj.binding().frame());
    if !same {
        return Err(ShapeRefusal::ForeignLanes);
    }
    let n = lanes.lanes().iter().map(|l| l.n).min().unwrap_or(0);
    if let Some(last) = obj.selection().last() {
        if last >= n {
            return Err(ShapeRefusal::OutOfRange { last, n });
        }
    }
    Ok(())
}

/// What the INT8 arm did.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Int8Report {
    /// K after padding to the 64-step.
    pub k_pad: usize,
    /// Bytes of A copied out of the source (u8, one byte per value).
    pub a_copied: usize,
    /// Bytes of B materialized row-major (i8).
    pub b_materialized: usize,
    /// 16 x 16 output tiles.
    pub tiles: usize,
    /// Logical `TDPBUSD` steps (`tiles x k_pad / 64`); B is re-packed per
    /// call inside ndarray and that pack is not counted here.
    pub dp_steps: usize,
}

/// Gather A (`rows x k_pad`, u8, zero-padded) into `a`.
pub fn stage_a(
    obj: &VirtualObject,
    lanes: &U8Lanes64<'_>,
    a: &mut Vec<u8>,
) -> Result<usize, ShapeRefusal> {
    admit(obj, lanes)?;
    let k_pad = obj.selection().len().div_ceil(K_STEP) * K_STEP;
    a.clear();
    a.resize(obj.fields().len() * k_pad, 0);
    for (r, lane) in lanes.lanes().iter().enumerate() {
        let values = lane.values();
        let row = &mut a[r * k_pad..(r + 1) * k_pad];
        match obj.selection().kind {
            // A region is contiguous in the lane: one copy, no per-value walk.
            Where::Region { start, len } => row[..len].copy_from_slice(&values[start..start + len]),
            Where::Mask { .. } => obj.selection().for_each(|k, i| row[k] = values[i]),
        }
    }
    Ok(obj.fields().len() * obj.selection().len())
}

/// [`stage_a`] with every selection walked value by value, regions too: the
/// baseline the region copy is measured against.
pub fn stage_a_per_value(
    obj: &VirtualObject,
    lanes: &U8Lanes64<'_>,
    a: &mut Vec<u8>,
) -> Result<usize, ShapeRefusal> {
    admit(obj, lanes)?;
    let k_pad = obj.selection().len().div_ceil(K_STEP) * K_STEP;
    a.clear();
    a.resize(obj.fields().len() * k_pad, 0);
    for (r, lane) in lanes.lanes().iter().enumerate() {
        let values = lane.values();
        let row = &mut a[r * k_pad..(r + 1) * k_pad];
        obj.selection().for_each(|k, i| row[k] = values[i]);
    }
    Ok(obj.fields().len() * obj.selection().len())
}

/// B for the GEMM: borrowed in place when the selection is a region whose
/// rows the source holds contiguously at this width and K needs no padding;
/// otherwise materialized into `bm`. Returns the slice and the bytes written.
pub fn b_operand<'b>(
    obj: &VirtualObject,
    b: &'b impl BValues,
    bm: &'b mut Vec<i8>,
) -> (&'b [i8], usize) {
    let k_pad = obj.selection().len().div_ceil(K_STEP) * K_STEP;
    if let Where::Region { start, len } = obj.selection().kind {
        if len == k_pad {
            if let Some(rows) = b.i8_rows(start, len, obj.cols()) {
                return (rows, 0);
            }
        }
    }
    let written = stage_b(obj, b, bm);
    (bm.as_slice(), written)
}

/// Materialize B (`k_pad x cols`, i8, zero rows past the selection) into `bm`.
pub fn stage_b(obj: &VirtualObject, b: &impl BValues, bm: &mut Vec<i8>) -> usize {
    let k_pad = obj.selection().len().div_ceil(K_STEP) * K_STEP;
    let n = obj.cols();
    bm.clear();
    bm.resize(k_pad * n, 0);
    obj.selection().for_each(|k, i| {
        for j in 0..n {
            bm[k * n + j] = b.i8_at(i, j);
        }
    });
    obj.selection().len() * n
}

/// The INT8 GEMM over already staged operands: `c = a x bm`, overwritten.
pub fn gemm(obj: &VirtualObject, a: &[u8], bm: &[i8], c: &mut [i32]) {
    let k_pad = obj.selection().len().div_ceil(K_STEP) * K_STEP;
    let (m, n) = (obj.fields().len(), obj.cols());
    assert_eq!(c.len(), m * n);
    c.fill(0);
    ndarray::backend::gemm_i8(a, bm, c, m, n, k_pad);
}

/// The INT8 arm end to end: stage A, materialize B, GEMM.
pub fn staged(
    obj: &VirtualObject,
    lanes: &U8Lanes64<'_>,
    b: &impl BValues,
    a: &mut Vec<u8>,
    bm: &mut Vec<i8>,
    c: &mut [i32],
) -> Result<Int8Report, ShapeRefusal> {
    let a_copied = stage_a(obj, lanes, a)?;
    let (b_slice, b_written) = b_operand(obj, b, bm);
    gemm(obj, a, b_slice, c);
    let k_pad = obj.selection().len().div_ceil(K_STEP) * K_STEP;
    Ok(Int8Report {
        k_pad,
        a_copied,
        b_materialized: b_written,
        tiles: obj.tiles(),
        dp_steps: obj.tiles() * k_pad / K_STEP,
    })
}

/// The exact answer in i64, from the value functions alone.
pub fn oracle(fx: &Fixture, sel: &Selection, fields: &[u8], cols: usize, b: &BOperand) -> Vec<i64> {
    let mut c = vec![0i64; fields.len() * cols];
    sel.for_each(|_, i| {
        for (p, &f) in fields.iter().enumerate() {
            let av = i64::from(a_u8(fx, i, usize::from(f)));
            for j in 0..cols {
                c[p * cols + j] += av * i64::from(b_i8(b, i, j));
            }
        }
    });
    c
}
