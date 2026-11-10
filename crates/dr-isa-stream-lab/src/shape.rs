//! D-DT64-GEMM-SHAPE-1/2: virtual 64 x 64 and 256 x 256 BF16 objects over
//! source-owned lanes.
//!
//! Pipeline: universe -> WHERE -> DT64-shaped virtual object -> BF16 tiles ->
//! GEMM. DR supplies the coordinates, DT64 binds the fields, the source owns
//! the values, and the GEMM is one optional consumer of the shape.
//!
//! * The universe is a [`Bf16Lanes`](crate::source::Bf16Lanes) source: one
//!   lane per property over every identity, in one coordinate space.
//! * [`Selection`] is the WHERE result in that space: a contiguous identity
//!   **region** (for example one DR64 or DR256 identity tile) or a scattered
//!   **mask**.
//! * [`VirtualObject`] is the shaped object: a DT64 [`LaneBinding`] of the
//!   demanded fields plus the selection and the output width. It holds
//!   addresses and the selection, never values; it is not a matrix store and
//!   not a second copy of the universe.
//!
//! Axis semantics, fixed for every arm:
//! * **rows (M)** = demanded fields, in binding order (16 per row tile);
//! * **reduction (K)** = selected identities, ascending in the selection's
//!   row order, padded to the 32-step `TDPBF16PS` consumes;
//! * **columns (N)** = output columns of the synthetic operand B (16 per
//!   column tile).
//!
//! `C[p][j] = sum over selected i of A[p][i] * B[i][j]`, FP32. At 64 x 64 that
//! is 4 x 4 output tiles of 16 x 16 and K = 64 (two 32-steps); at 256 x 256 it
//! is 16 x 16 output tiles and K = 256 (eight 32-steps). The 256 x 256 here is
//! fields x columns over 256 identities: not an identity x identity adjacency
//! matrix, and not a claim about resident lanes.
//!
//! The logical tile geometry is not the physical operand. Physically, a
//! direct A tile is 16 rows of the source slab at the slab pitch, offset by
//! the region start; a staged A tile is a dense 16 x K buffer; B is N / 16
//! VNNI-packed K x 16 column panels.
//!
//! Four arms:
//! * [`direct`]: A read in place by `bf16_tile_gemm_16x16_packed_strided`.
//!   Refuses unless the selection is a region, every 16-field group is 16
//!   consecutive rows of one slab, and the K padding would read only the
//!   slab's zeroed padding, never live neighbouring identities.
//! * [`staged`]: A gathered into a dense 16 x K tile per field group, every
//!   copied byte counted. Accepts regions and masks.
//! * [`cblas`]: ndarray's `cblas_gemm_bf16bf16f32` over dense A and B, both
//!   materialized here and counted.
//! * [`oracle`]: a scalar `f64` sum from value functions, without lanes,
//!   bindings or tiles.

use lance_graph_dt64::dr::Frame;
use lance_graph_dt64::{CoordSpace, LaneBinding, LaneRefusal, LaneSource};
use ndarray::simd::{
    bf16_tile_gemm_16x16_packed, bf16_tile_gemm_16x16_packed_strided, PackedBf16B,
};

use crate::bind::World;
use crate::fixture::{bf16_exact, mix, BLOCK};
use crate::gemm::{geometry, DirectRefusal};
use crate::source::{SlabLane, Socket};

/// Edge of one output tile.
pub const TILE: usize = 16;
/// Most fields one object binds (DT64's field positions are `u8`).
pub const MAX_ROWS: usize = 256;

/// The WHERE result, in one coordinate space.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Where {
    /// Identities `start..start + len`.
    Region { start: usize, len: usize },
    /// The set bits of `words`, ascending, over `n` identities.
    Mask { n: usize, words: Box<[u64]> },
}

/// A WHERE result and the row order it is expressed in.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Selection {
    pub space: CoordSpace,
    pub kind: Where,
}

impl Selection {
    /// A contiguous identity region.
    pub fn region(space: CoordSpace, start: usize, len: usize) -> Selection {
        Selection {
            space,
            kind: Where::Region { start, len },
        }
    }

    /// The identities `ids` (any order; stored as a bitmap) over `n`.
    pub fn mask(space: CoordSpace, n: usize, ids: impl IntoIterator<Item = usize>) -> Selection {
        let mut words = vec![0u64; n.div_ceil(64)].into_boxed_slice();
        for i in ids {
            words[i / 64] |= 1 << (i % 64);
        }
        Selection {
            space,
            kind: Where::Mask { n, words },
        }
    }

    /// Selected identities: K before padding.
    pub fn len(&self) -> usize {
        match &self.kind {
            Where::Region { len, .. } => *len,
            Where::Mask { words, .. } => words.iter().map(|w| w.count_ones() as usize).sum(),
        }
    }

    /// Whether nothing is selected.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// The highest selected identity, if any.
    pub fn last(&self) -> Option<usize> {
        match &self.kind {
            Where::Region { start, len } => len.checked_sub(1).map(|l| start + l),
            Where::Mask { words, .. } => words
                .iter()
                .enumerate()
                .rev()
                .find(|(_, &w)| w != 0)
                .map(|(i, &w)| i * 64 + 63 - w.leading_zeros() as usize),
        }
    }

    /// K padded to the 32-identity step the tile GEMM consumes.
    pub fn k_pad(&self) -> usize {
        self.len().div_ceil(BLOCK) * BLOCK
    }

    /// Call `f(k, i)` for the `k`-th selected identity `i`, ascending.
    pub fn for_each(&self, mut f: impl FnMut(usize, usize)) {
        match &self.kind {
            Where::Region { start, len } => (0..*len).for_each(|k| f(k, start + k)),
            Where::Mask { words, .. } => {
                let mut k = 0;
                for (w, &word) in words.iter().enumerate() {
                    let mut bits = word;
                    while bits != 0 {
                        f(k, w * 64 + bits.trailing_zeros() as usize);
                        k += 1;
                        bits &= bits - 1;
                    }
                }
            }
        }
    }
}

/// Why the shape cannot be built or consumed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ShapeRefusal {
    /// The DT64 binding refused (frame, space, scope, route, mapping, basis).
    Lane(LaneRefusal),
    /// The field count is not a multiple of 16 in 16..=256.
    FieldCount { fields: usize },
    /// The output width is not a multiple of 16, or is zero.
    Columns { cols: usize },
    /// Nothing is selected.
    EmptySelection,
    /// The selection is in another row order than the binding.
    SelectionSpace {
        selection: CoordSpace,
        binding: CoordSpace,
    },
    /// Direct read: the lanes are not one tile load.
    Direct(DirectRefusal),
    /// Direct read: the WHERE result is a scattered mask, not a region.
    NotARegion,
    /// Direct read: padding K to the 32-step would read identities that are
    /// live data (not the slab's zeroed padding), or past the row.
    OverRead {
        start: usize,
        len: usize,
        k_pad: usize,
        n: usize,
        pitch: usize,
    },
    /// The prepared B does not belong to this object.
    Reuse(ReuseRefusal),
    /// The lanes were resolved by another object: another source or frame,
    /// another field count, or another lane mapping.
    ForeignLanes,
    /// The selection names identity `last`, but the lanes hold only `n`.
    OutOfRange { last: usize, n: usize },
}

impl From<LaneRefusal> for ShapeRefusal {
    fn from(r: LaneRefusal) -> Self {
        ShapeRefusal::Lane(r)
    }
}

/// The shaped object: a lane binding, the WHERE result and the output width.
/// Addresses and the selection only; no values.
#[derive(Debug, Clone)]
pub struct VirtualObject {
    binding: LaneBinding,
    fields: Vec<u8>,
    cols: usize,
    selection: Selection,
    /// Which bind made this object. Lanes carry it, so an arm can refuse
    /// lanes another object resolved with one compare. A clone keeps it: it
    /// has the same binding and the same lane addresses.
    id: u64,
}

/// Bind identities; only uniqueness matters.
static NEXT_OBJECT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);

impl VirtualObject {
    /// Cold: bind `fields` (a multiple of 16, at most 256) through DT64 under
    /// the selection's row order; the view has `cols` output columns.
    pub fn bind<S: LaneSource>(
        world: &World,
        socket: &S,
        fields: &[u8],
        cols: usize,
        selection: Selection,
    ) -> Result<VirtualObject, ShapeRefusal> {
        if fields.is_empty() || fields.len() > MAX_ROWS || !fields.len().is_multiple_of(TILE) {
            return Err(ShapeRefusal::FieldCount {
                fields: fields.len(),
            });
        }
        if cols == 0 || !cols.is_multiple_of(TILE) {
            return Err(ShapeRefusal::Columns { cols });
        }
        if selection.is_empty() {
            return Err(ShapeRefusal::EmptySelection);
        }
        let binding = world.bind(socket, fields, selection.space)?;
        Ok(VirtualObject {
            binding,
            fields: fields.to_vec(),
            cols,
            selection,
            id: NEXT_OBJECT.fetch_add(1, std::sync::atomic::Ordering::Relaxed),
        })
    }

    /// Which bind made this object (provenance for resolved lanes).
    pub fn id(&self) -> u64 {
        self.id
    }

    /// The lane binding (addresses only), for arms over other lane types.
    pub fn binding(&self) -> &LaneBinding {
        &self.binding
    }

    /// The demanded fields: the view's rows.
    pub fn fields(&self) -> &[u8] {
        &self.fields
    }

    /// Output columns.
    pub fn cols(&self) -> usize {
        self.cols
    }

    /// The WHERE result: the view's reduction axis.
    pub fn selection(&self) -> &Selection {
        &self.selection
    }

    /// The frame the binding was made at.
    pub fn frame(&self) -> &Frame {
        self.binding.frame()
    }

    /// Output tiles of the view.
    pub fn tiles(&self) -> usize {
        (self.fields.len() / TILE) * (self.cols / TILE)
    }

    /// FP32 result elements (`rows x cols`).
    pub fn c_len(&self) -> usize {
        self.fields.len() * self.cols
    }

    /// Warm: [`LaneBinding::borrow_all`] checks the binding once against the
    /// store, view and source, then hands out every bound lane in ascending
    /// field order; this places them in the view's row order. Any refusal,
    /// `Vanished` included, is returned before a lane is placed, so no arm
    /// ever runs on a partially resolved object. No allocation.
    pub fn resolve<'s>(
        &self,
        world: &World,
        socket: &'s Socket<'_>,
    ) -> Result<Lanes<'s>, ShapeRefusal> {
        if self.selection.space != self.binding.selection() {
            return Err(ShapeRefusal::SelectionSpace {
                selection: self.selection.space,
                binding: self.binding.selection(),
            });
        }
        let mut by_field: [Option<SlabLane<'s>>; MAX_ROWS] = [None; MAX_ROWS];
        for item in
            self.binding
                .borrow_all(&world.store, &world.view, socket.lanes.frames(), socket)?
        {
            let (field, lane) = item?;
            by_field[usize::from(field)] = Some(lane);
        }
        let lane = |f: u8| by_field[usize::from(f)].ok_or(LaneRefusal::NotBound { field: f });
        let first = lane(self.fields[0])?;
        let mut lanes = [first; MAX_ROWS];
        for (slot, &f) in lanes.iter_mut().zip(&self.fields).skip(1) {
            *slot = lane(f)?;
        }
        Ok(self.stamp(lanes))
    }

    /// Lanes with their provenance: this object's identity and frame issue.
    fn stamp<'s>(&self, lanes: [SlabLane<'s>; MAX_ROWS]) -> Lanes<'s> {
        Lanes {
            lanes,
            object: self.id,
            frame: self.binding.frame().clone(),
            count: self.fields.len(),
        }
    }

    /// Refuse lanes this object did not resolve: another frame issue (another
    /// source, or the source has moved), another field count, or another
    /// lane address in any row.
    pub fn admit_lanes(&self, lanes: &Lanes<'_>) -> Result<(), ShapeRefusal> {
        // The object identity implies the lane addresses (same binding) and
        // the field count; the frame is checked as well, at O(1).
        let same = lanes.object == self.id
            && lanes.count == self.fields.len()
            && lanes.frame.same_issue(self.binding.frame());
        if !same {
            return Err(ShapeRefusal::ForeignLanes);
        }
        // Every selected identity must exist in every lane: slab padding past
        // the lane's logical length is allowed as GEMM K padding (`direct`
        // checks that separately), never as a selected value.
        let n = lanes.lanes().iter().map(|l| l.n).min().unwrap_or(0);
        if let Some(last) = self.selection.last() {
            if last >= n {
                return Err(ShapeRefusal::OutOfRange { last, n });
            }
        }
        Ok(())
    }

    /// The same as [`Self::resolve`] the long way: `LaneBinding::lane` per
    /// field, which re-checks every bound lane on each call (quadratic in the
    /// field count). Kept to measure that cost.
    pub fn resolve_per_field<'s>(
        &self,
        world: &World,
        socket: &'s Socket<'_>,
    ) -> Result<Lanes<'s>, ShapeRefusal> {
        let lane = |f: u8| {
            self.binding
                .lane(&world.store, &world.view, socket.lanes.frames(), socket, f)
        };
        let first = lane(self.fields[0])?;
        let mut lanes = [first; MAX_ROWS];
        for (slot, &f) in lanes.iter_mut().zip(&self.fields).skip(1) {
            *slot = lane(f)?;
        }
        Ok(self.stamp(lanes))
    }
}

/// The borrowed lanes of a resolved object, in view row order, with the
/// identity of the object that resolved them and its frame issue, so an arm
/// can refuse lanes that belong to another object.
#[derive(Debug, Clone)]
pub struct Lanes<'s> {
    lanes: [SlabLane<'s>; MAX_ROWS],
    object: u64,
    frame: Frame,
    count: usize,
}

impl<'s> Lanes<'s> {
    /// The borrowed lanes.
    pub fn lanes(&self) -> &[SlabLane<'s>] {
        &self.lanes[..self.count]
    }
}

/// The synthetic B operand: `B[i][j]` for universe identity `i` and output
/// column `j`, an integer in [-8, 8]. `rev` names a revision of B.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BOperand {
    pub seed: u64,
    pub rev: u64,
}

impl BOperand {
    /// `B[i][j]`.
    pub fn value(&self, i: usize, j: usize) -> i32 {
        let h = mix(self.seed ^ self.rev.rotate_left(29) ^ ((j as u64) << 44) ^ i as u64);
        (h % 17) as i32 - 8
    }

    /// `B[i][j]` in bf16 (always exact).
    pub fn bf16(&self, i: usize, j: usize) -> u16 {
        bf16_exact(self.value(i, j)).expect("[-8, 8] is exact in bf16")
    }
}

/// Where B's values come from: computed per call ([`BOperand`], a hash) or
/// read from a table built once ([`BTable`]). Timing B preparation over a
/// `BOperand` measures the hash, not the data movement; benchmarks use a
/// `BTable`.
pub trait BValues {
    /// Which operand this is and at which revision: what prepared-B reuse is
    /// keyed on.
    fn identity(&self) -> OperandId;
    /// `B[i][j]` in bf16.
    fn bf16_at(&self, i: usize, j: usize) -> u16;
    /// `B[i][j]` in i8.
    fn i8_at(&self, i: usize, j: usize) -> i8;
    /// Rows `start..start + len` of B in i8, row-major with `cols` columns,
    /// borrowed in place if the source holds them that way.
    fn i8_rows(&self, _start: usize, _len: usize, _cols: usize) -> Option<&[i8]> {
        None
    }
}

/// Authoritative operand identity: the source that defines the values and
/// its revision. Not a digest of the values.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct OperandId {
    /// For a [`BOperand`], its seed (the seed defines the function); for a
    /// [`BTable`], the table's construction instance.
    pub source: OperandSource,
    pub rev: u64,
}

/// Where an operand's values are defined.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OperandSource {
    /// A deterministic function of `(seed, i, j)`.
    Function { seed: u64 },
    /// An immutable table, by the construction that filled it.
    Table { instance: u64 },
}

impl BValues for BOperand {
    fn identity(&self) -> OperandId {
        OperandId {
            source: OperandSource::Function { seed: self.seed },
            rev: self.rev,
        }
    }
    fn bf16_at(&self, i: usize, j: usize) -> u16 {
        self.bf16(i, j)
    }
    fn i8_at(&self, i: usize, j: usize) -> i8 {
        i8::try_from(self.value(i, j)).expect("[-8, 8] is in i8")
    }
}

/// B materialized once over the universe, row-major `n x cols`, in both
/// encodings. Reading it is a load, so preparing B from it times movement.
pub struct BTable {
    instance: u64,
    rev: u64,
    cols: usize,
    bf16: Vec<u16>,
    i8: Vec<i8>,
}

impl BTable {
    /// Every `B[i][j]` for `i < n`, `j < cols`.
    pub fn new(b: &BOperand, n: usize, cols: usize) -> BTable {
        let (mut bf16, mut i8s) = (Vec::with_capacity(n * cols), Vec::with_capacity(n * cols));
        for i in 0..n {
            for j in 0..cols {
                bf16.push(b.bf16_at(i, j));
                i8s.push(b.i8_at(i, j));
            }
        }
        BTable {
            instance: NEXT_TABLE.fetch_add(1, std::sync::atomic::Ordering::Relaxed),
            rev: b.rev,
            cols,
            bf16,
            i8: i8s,
        }
    }
}

/// Table instances; only uniqueness matters. A table never changes after
/// construction, so instance + revision identifies its values.
static NEXT_TABLE: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);

impl BValues for BTable {
    fn identity(&self) -> OperandId {
        OperandId {
            source: OperandSource::Table {
                instance: self.instance,
            },
            rev: self.rev,
        }
    }
    fn bf16_at(&self, i: usize, j: usize) -> u16 {
        self.bf16[i * self.cols + j]
    }
    fn i8_at(&self, i: usize, j: usize) -> i8 {
        self.i8[i * self.cols + j]
    }
    fn i8_rows(&self, start: usize, len: usize, cols: usize) -> Option<&[i8]> {
        (cols == self.cols)
            .then(|| self.i8.get(start * cols..(start + len) * cols))
            .flatten()
    }
}

/// Why a prepared B cannot serve an object.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReuseRefusal {
    /// Prepared for another selection (another population or row order).
    Selection,
    /// Prepared under another frame issue: the source has moved since.
    Frame,
    /// Prepared from another revision of B.
    Operand,
    /// Prepared for another output width.
    Columns,
}

/// B as `cols / 16` VNNI-packed K x 16 column panels, prepared once for one
/// object (selection, frame, width) and one operand identity and reused by every
/// output tile and every repeated run. Rows are the selected identities in
/// order; rows past the selection are zero.
pub struct PreparedB {
    /// The object it was packed for (fast path: one compare).
    object: u64,
    /// That object's selection, compared in full when the object differs:
    /// equality, never a digest.
    selection: Selection,
    frame: Frame,
    operand: OperandId,
    panels: Vec<PackedBf16B>,
}

/// What preparing B did.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct PrepReport {
    /// Panels packed (one per 16 output columns).
    pub panels: usize,
    /// bf16 entries written into the panels, in bytes.
    pub bytes_written: usize,
    /// Bytes the panels occupy, K padding included.
    pub bytes_allocated: usize,
}

impl PreparedB {
    /// Pack B for `obj`: `panel t row k = B[i_k][16t..16t+16]` in VNNI order.
    pub fn prepare(obj: &VirtualObject, b: &impl BValues) -> (PreparedB, PrepReport) {
        let k = obj.selection.k_pad();
        let mut panels: Vec<PackedBf16B> = (0..obj.cols / TILE)
            .map(|_| PackedBf16B::zeroed(k))
            .collect();
        let mut written = 0;
        obj.selection.for_each(|row, i| {
            for (t, panel) in panels.iter_mut().enumerate() {
                let data = panel.data_mut();
                for c in 0..TILE {
                    data[PackedBf16B::vnni_index(row, c)] = b.bf16_at(i, t * TILE + c);
                }
                written += TILE * 2;
            }
        });
        let allocated = panels.iter().map(|p| p.data().len() * 2).sum();
        (
            PreparedB {
                object: obj.id,
                selection: obj.selection.clone(),
                frame: obj.frame().clone(),
                operand: b.identity(),
                panels,
            },
            PrepReport {
                panels: obj.cols / TILE,
                bytes_written: written,
                bytes_allocated: allocated,
            },
        )
    }

    /// Panel `t` (output columns `16t..16t+16`), for inspection.
    pub fn panel(&self, t: usize) -> &PackedBf16B {
        &self.panels[t]
    }

    /// Whether this prepared B may serve `obj` with operand `b`.
    ///
    /// The operand is compared by identity (source and revision), always.
    /// For the object it was packed for, one identity compare covers the
    /// selection, frame and width. For any other object the selection is
    /// compared in full, then the frame issue and the width: a prepared B is
    /// shared across objects only when all three are equal.
    pub fn admit(&self, obj: &VirtualObject, b: &impl BValues) -> Result<(), ReuseRefusal> {
        if self.operand != b.identity() {
            return Err(ReuseRefusal::Operand);
        }
        if self.object == obj.id {
            return Ok(());
        }
        if self.selection != obj.selection {
            return Err(ReuseRefusal::Selection);
        }
        if !self.frame.same_issue(obj.frame()) {
            return Err(ReuseRefusal::Frame);
        }
        if self.panels.len() * TILE != obj.cols {
            return Err(ReuseRefusal::Columns);
        }
        Ok(())
    }
}

/// What one GEMM arm did.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ArmReport {
    /// 16 x 16 output tiles computed.
    pub tiles: usize,
    /// GEMM entry calls.
    pub calls: usize,
    /// Logical K=32 BF16 dot-product steps (`tiles x k_pad / 32`); excludes
    /// tile loads, configuration and packing.
    pub dp_steps: usize,
    /// Bytes of A copied out of the source.
    pub a_copied: usize,
    /// Bytes of B materialized row-major (cblas arm only).
    pub b_materialized: usize,
}

fn store_tile(c: &mut [f32], cols: usize, g: usize, t: usize, tile: &[f32; TILE * TILE]) {
    for r in 0..TILE {
        let row = (g * TILE + r) * cols + t * TILE;
        c[row..row + TILE].copy_from_slice(&tile[r * TILE..(r + 1) * TILE]);
    }
}

/// ARM A: A read in place from the slab, one strided tile GEMM per output
/// tile, B from the prepared panels. `c` is `rows x cols`, row-major.
pub fn direct(
    obj: &VirtualObject,
    lanes: &Lanes<'_>,
    b: &PreparedB,
    operand: &impl BValues,
    c: &mut [f32],
) -> Result<ArmReport, ShapeRefusal> {
    b.admit(obj, operand).map_err(ShapeRefusal::Reuse)?;
    obj.admit_lanes(lanes)?;
    let Where::Region { start, len } = obj.selection.kind else {
        return Err(ShapeRefusal::NotARegion);
    };
    assert_eq!(c.len(), obj.c_len());
    let k_pad = obj.selection.k_pad();
    let mut report = ArmReport::default();
    for (g, (fs, ls)) in obj
        .fields
        .chunks(TILE)
        .zip(lanes.lanes().chunks(TILE))
        .enumerate()
    {
        let geo = geometry(fs, ls).map_err(ShapeRefusal::Direct)?;
        let n = ls[0].n;
        // Rows k >= len of B are zero, but 0 * NaN is NaN: the padded read is
        // allowed only into the slab's zeroed padding past the last identity.
        let pads_into_live = len != k_pad && start + len != n;
        if pads_into_live || start + k_pad > geo.pitch {
            return Err(ShapeRefusal::OverRead {
                start,
                len,
                k_pad,
                n,
                pitch: geo.pitch,
            });
        }
        let a = &geo.slab[geo.first_row * geo.pitch + start..];
        for (t, panel) in b.panels.iter().enumerate() {
            let mut tile = [0f32; TILE * TILE];
            bf16_tile_gemm_16x16_packed_strided(a, geo.pitch, panel, &mut tile);
            store_tile(c, obj.cols, g, t, &tile);
            report.calls += 1;
            report.tiles += 1;
            report.dp_steps += k_pad / BLOCK;
        }
    }
    Ok(report)
}

/// ARM B: A gathered into a dense 16 x K tile per field group (in `scratch`,
/// reused across calls), then one packed tile GEMM per output tile.
pub fn staged(
    obj: &VirtualObject,
    lanes: &Lanes<'_>,
    b: &PreparedB,
    operand: &impl BValues,
    scratch: &mut Vec<u16>,
    c: &mut [f32],
) -> Result<ArmReport, ShapeRefusal> {
    b.admit(obj, operand).map_err(ShapeRefusal::Reuse)?;
    obj.admit_lanes(lanes)?;
    assert_eq!(c.len(), obj.c_len());
    let k_pad = obj.selection.k_pad();
    let len = obj.selection.len();
    let mut report = ArmReport::default();
    for (g, ls) in lanes.lanes().chunks(TILE).enumerate() {
        scratch.clear();
        scratch.resize(TILE * k_pad, 0);
        for (r, lane) in ls.iter().enumerate() {
            let values = lane.values();
            let row = &mut scratch[r * k_pad..(r + 1) * k_pad];
            match &obj.selection.kind {
                Where::Region { start, len } => {
                    row[..*len].copy_from_slice(&values[*start..start + len])
                }
                Where::Mask { .. } => obj.selection.for_each(|k, i| row[k] = values[i]),
            }
            report.a_copied += len * 2;
        }
        for (t, panel) in b.panels.iter().enumerate() {
            let mut tile = [0f32; TILE * TILE];
            bf16_tile_gemm_16x16_packed(scratch, panel, &mut tile);
            store_tile(c, obj.cols, g, t, &tile);
            report.calls += 1;
            report.tiles += 1;
            report.dp_steps += k_pad / BLOCK;
        }
    }
    Ok(report)
}

/// ARM C: ndarray's CBLAS-compatible BF16 GEMM over dense row-major A
/// (rows x K) and B (K x cols), both materialized here into `a` / `bm`.
pub fn cblas(
    obj: &VirtualObject,
    lanes: &Lanes<'_>,
    b: &impl BValues,
    a: &mut Vec<u16>,
    bm: &mut Vec<u16>,
    c: &mut [f32],
) -> Result<ArmReport, ShapeRefusal> {
    obj.admit_lanes(lanes)?;
    let m = obj.fields.len();
    let n = obj.cols;
    let k = obj.selection.len();
    assert_eq!(c.len(), m * n);
    a.clear();
    a.resize(m * k, 0);
    bm.clear();
    bm.resize(k * n, 0);
    for (r, lane) in lanes.lanes().iter().enumerate() {
        let values = lane.values();
        let row = &mut a[r * k..(r + 1) * k];
        obj.selection.for_each(|kk, i| row[kk] = values[i]);
    }
    obj.selection.for_each(|kk, i| {
        for j in 0..n {
            bm[kk * n + j] = b.bf16_at(i, j);
        }
    });
    ndarray::backend::cblas_gemm_bf16bf16f32(a, bm, c, m, n, k);
    Ok(ArmReport {
        tiles: obj.tiles(),
        calls: 1,
        dp_steps: obj.tiles() * obj.selection.k_pad() / BLOCK,
        a_copied: m * k * 2,
        b_materialized: k * n * 2,
    })
}

/// ARM D: `C[p][j] = sum over selected i of a(i, field_p) * b(i, j)` in f64,
/// from value functions alone. `rows x cols`, row-major.
pub fn oracle(
    selection: &Selection,
    fields: &[u8],
    cols: usize,
    a: impl Fn(usize, usize) -> f64,
    b: impl Fn(usize, usize) -> f64,
) -> Vec<f64> {
    let mut c = vec![0f64; fields.len() * cols];
    selection.for_each(|_, i| {
        for (p, &f) in fields.iter().enumerate() {
            let av = a(i, usize::from(f));
            for j in 0..cols {
                c[p * cols + j] += av * b(i, j);
            }
        }
    });
    c
}
