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
//! Two INT8 arms:
//! - [`staged`]: `backend::gemm_i8` (-> `int8_gemm_amx_tiled`). A is copied
//!   dense with row stride K, and B is gathered and VNNI-packed inside every
//!   call.
//! - [`direct`]: A read in place at the lanes' pitch through
//!   `ndarray::simd::int8_gemm_packed_strided`, and B prepared
//!   once ([`PreparedI8B`], one `PackedI8B` band per 16 output columns) and
//!   reused under the same identity rule as the BF16 [`crate::shape::PreparedB`]
//!   ([`ReuseKey`]). These are the INT8 twins of the BF16 entries ndarray
//!   added in #414.

use std::cell::Cell;

use lance_graph_contract::ontology::FieldRef;
use lance_graph_dt64::dr::{Frame, FrameRevisions, FrameStore, Scope};
use lance_graph_dt64::{CoordSpace, LaneAddr, LaneRefusal, LaneSource, SchemaRev};

use ndarray::simd::{int8_gemm_packed_strided, PackedI8B};

use crate::bind::{World, CLASS, READER, SPACE_A};
use crate::fixture::Fixture;
use crate::gemm::DirectRefusal;
use crate::shape::{
    admit_operand, BOperand, BValues, ReuseKey, Selection, ShapeRefusal, VirtualObject, Where,
    MAX_ROWS, TILE,
};

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

    /// The frame owner.
    pub fn frames(&self) -> &FrameStore {
        &self.frames
    }

    /// Mutable frame owner, for advancing the source after a change.
    pub fn frames_mut(&mut self) -> &mut FrameStore {
        &mut self.frames
    }

    /// Rewrite every value in place (`value(i, p)` for identity row `i` of
    /// lane `p`). The caller advances the frames afterwards.
    pub fn rewrite(&mut self, value: impl Fn(usize, usize) -> u8) {
        let (pitch, props) = (self.pitch, self.data.len() / self.pitch.max(1));
        for p in 0..props {
            for i in 0..self.n {
                self.data[p * pitch + i] = value(i, p);
            }
        }
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
    crate::shape::admit_operand(obj, b)?;
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

/// B as `cols / 16` VNNI-packed `K x 16` i8 bands, prepared once for one
/// object and one operand identity, reused by every output tile and every
/// repeated run. Rows are the selected identities in order; rows past the
/// selection are zero.
pub struct PreparedI8B {
    key: ReuseKey,
    bands: Vec<PackedI8B>,
}

impl PreparedI8B {
    /// Pack B for `obj`: `band t row k = B[i_k][16t..16t+16]` in VNNI order.
    /// Refuses an operand smaller than the object ([`admit_operand`]).
    pub fn prepare(obj: &VirtualObject, b: &impl BValues) -> Result<PreparedI8B, ShapeRefusal> {
        admit_operand(obj, b)?;
        let k_pad = obj.selection().len().div_ceil(K_STEP) * K_STEP;
        let mut bands: Vec<PackedI8B> = (0..obj.cols() / TILE)
            .map(|_| PackedI8B::zeroed(k_pad))
            .collect();
        obj.selection().for_each(|row, i| {
            for (t, band) in bands.iter_mut().enumerate() {
                let data = band.data_mut();
                for c in 0..TILE {
                    data[PackedI8B::vnni_index(row, c)] = b.i8_at(i, t * TILE + c);
                }
            }
        });
        Ok(PreparedI8B {
            key: ReuseKey::of(obj, b),
            bands,
        })
    }

    /// Bytes the bands occupy, K padding included.
    pub fn bytes(&self) -> usize {
        self.bands.iter().map(|b| b.data().len()).sum()
    }
}

/// Where 16 lanes sit when they can be read as one strided A operand.
struct Geometry<'s> {
    slab: &'s [u8],
    pitch: usize,
    first_row: usize,
}

/// The 16 lanes of a row tile as one strided operand: one allocation, one
/// pitch, consecutive rows. Anything else is refused, never copied.
fn geometry<'s>(fields: &[u8], lanes: &[U8Lane<'s>]) -> Result<Geometry<'s>, DirectRefusal> {
    if lanes.len() < TILE {
        return Err(DirectRefusal::PartialRowTile { lanes: lanes.len() });
    }
    let first = lanes[0];
    for (k, (lane, &field)) in lanes.iter().zip(fields).enumerate().skip(1) {
        if !std::ptr::eq(lane.slab, first.slab) {
            return Err(DirectRefusal::NotOneSlab { field });
        }
        if lane.pitch != first.pitch {
            return Err(DirectRefusal::PitchMismatch { field });
        }
        if lane.row != first.row + k {
            return Err(DirectRefusal::RowsNotConsecutive { field });
        }
    }
    Ok(Geometry {
        slab: first.slab,
        pitch: first.pitch,
        first_row: first.row,
    })
}

/// The INT8 direct arm: A read in place at the lanes' pitch, B from the
/// prepared bands. Needs a region selection; `c` is `rows x cols`, row-major,
/// overwritten. Rows of B past the selection are zero, so the read up to the
/// 64-step padding adds nothing; the read must still stay inside the lane's
/// own row of the slab.
pub fn direct(
    obj: &VirtualObject,
    lanes: &U8Lanes64<'_>,
    b: &PreparedI8B,
    operand: &impl BValues,
    c: &mut [i32],
) -> Result<Int8Report, ShapeRefusal> {
    b.key.admit(obj, operand).map_err(ShapeRefusal::Reuse)?;
    admit(obj, lanes)?;
    let Where::Region { start, len } = obj.selection().kind else {
        return Err(ShapeRefusal::NotARegion);
    };
    let (rows, cols) = (obj.fields().len(), obj.cols());
    assert_eq!(c.len(), rows * cols);
    let k_pad = len.div_ceil(K_STEP) * K_STEP;
    let mut report = Int8Report {
        k_pad,
        ..Int8Report::default()
    };
    // One call when every lane is one strided operand (the source's own
    // order); otherwise one call per 16-field group, each writing its rows.
    if geometry(obj.fields(), lanes.lanes()).is_ok() {
        run_rows(
            start,
            len,
            k_pad,
            0,
            obj.fields(),
            lanes.lanes(),
            b,
            c,
            cols,
            &mut report,
        )?;
    } else {
        for (g, (fs, ls)) in obj
            .fields()
            .chunks(TILE)
            .zip(lanes.lanes().chunks(TILE))
            .enumerate()
        {
            run_rows(start, len, k_pad, g * TILE, fs, ls, b, c, cols, &mut report)?;
        }
    }
    Ok(report)
}

/// One strided GEMM over the lanes `ls` (fields `fs`), writing C's rows from
/// `row0`.
#[allow(clippy::too_many_arguments)]
fn run_rows(
    start: usize,
    len: usize,
    k_pad: usize,
    row0: usize,
    fs: &[u8],
    ls: &[U8Lane<'_>],
    b: &PreparedI8B,
    c: &mut [i32],
    cols: usize,
    report: &mut Int8Report,
) -> Result<(), ShapeRefusal> {
    let geo = geometry(fs, ls).map_err(ShapeRefusal::Direct)?;
    if start + k_pad > geo.pitch {
        return Err(ShapeRefusal::OverRead {
            start,
            len,
            k_pad,
            n: ls[0].n,
            pitch: geo.pitch,
        });
    }
    let a = &geo.slab[geo.first_row * geo.pitch + start..];
    int8_gemm_packed_strided(a, geo.pitch, &b.bands, &mut c[row0 * cols..], fs.len());
    report.tiles += fs.len() / TILE * b.bands.len();
    report.dp_steps += fs.len() / TILE * b.bands.len() * (k_pad / K_STEP);
    Ok(())
}

/// The staged arm with a prepared B: A copied (one region copy per lane, or a
/// per-value walk for a mask) into `a`, B taken from the prepared bands. For
/// selections the direct arm cannot read in place, such as a scattered mask.
/// `c` is overwritten.
pub fn staged_prepared(
    obj: &VirtualObject,
    lanes: &U8Lanes64<'_>,
    b: &PreparedI8B,
    operand: &impl BValues,
    a: &mut Vec<u8>,
    c: &mut [i32],
) -> Result<Int8Report, ShapeRefusal> {
    b.key.admit(obj, operand).map_err(ShapeRefusal::Reuse)?;
    let a_copied = stage_a(obj, lanes, a)?;
    let k_pad = obj.selection().len().div_ceil(K_STEP) * K_STEP;
    let rows = obj.fields().len();
    assert_eq!(c.len(), rows * obj.cols());
    int8_gemm_packed_strided(a, k_pad, &b.bands, c, rows);
    Ok(Int8Report {
        k_pad,
        a_copied,
        b_materialized: 0,
        tiles: rows / TILE * b.bands.len(),
        dp_steps: rows / TILE * b.bands.len() * (k_pad / K_STEP),
    })
}
