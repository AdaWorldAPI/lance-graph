//! **LAB.** A DR256 slab folded as three aligned mask planes.
//!
//! ```text
//! DT64   ClassView (the menu)            what a field means, in which order
//! DR     WideFieldMask (the selection)   which fields: storage, compute, display
//!        Plane (one per slab)            which rows: 256 selections of 256, borrowed
//!        RankMap (global -> local)       where a global id sits after compaction
//! exec   mask-RISC                       ternlog(a, b, c) -> popcount, per slab
//! ```
//!
//! A slab is 256 x 256 = 65,536 positions, so one plane is 1,024 words
//! (8 KiB). A population past one slab is stacked over an outer byte: slab
//! `s` holds rows `s * 65,536 ..`. No cube is allocated; a 256 x 256 x 256
//! domain is at most 256 slabs of three planes each.
//!
//! What is resolved once per source frame ([`Prepared`]) and what per query:
//!
//! | once per frame | per query |
//! |---|---|
//! | one [`LaneBinding`] per slab | `LaneBinding::borrow_all` (the check) |
//! | presence and scope planes | one mask-RISC program per slab |
//! | fk planes of an edge table (`lookups` counted) | nothing else |
//!
//! Planes carry their [`CoordSpace`]. A fold over planes of two spaces is
//! refused ([`Refused::Misaligned`]); nothing here translates between them.
//!
//! Not in this crate: the DR map table (`lance-graph-dt64-lance`), any
//! storage, any authorization decision. The scope plane is an input: who may
//! read is decided elsewhere (locality contract, boundary 5).

#![forbid(unsafe_code)]

use lance_graph_contract::class_view::{ClassId, ClassView, WideFieldMask};
use lance_graph_contract::ontology::{DisplayTemplate, FieldRef};
use lance_graph_dt64::dr::{Extent, Frame, FrameRevisions, FrameStore, Scope};
use lance_graph_dt64::{
    BindStore, ClassSchema, CoordSpace, LaneAddr, LaneBinding, LaneRefusal, LaneRequest,
    LaneSource, Refusal, RouteAt, SchemaRev,
};
use lance_graph_mask_risc::exec::{scratch_words_for, tile_words_for};
use lance_graph_mask_risc::{
    execute_into, words_for, ExecError, Foreign, LaneRef, MaskOp, Operand, Out, Planes, Pred,
    Program, Scratch, Terminal, Value,
};

/// Positions of one slab: 256 x 256.
pub const SLAB_ROWS: usize = 65_536;
/// Words of one slab plane.
pub const SLAB_WORDS: usize = SLAB_ROWS / 64;
/// Slabs a 256 x 256 x 256 domain holds.
pub const MAX_SLABS: usize = 256;
/// `a & b & c` in the VPTERNLOG index convention.
pub const AND3: u8 = 0x80;
/// Positions of one selection: one row of a slab.
pub const MASK_BITS: usize = 256;
/// Words of one selection, the chunk count of a 256-position `WideFieldMask`.
pub const MASK_WORDS: usize = MASK_BITS / 64;
/// Selections a full slab plane holds.
pub const SLAB_MASKS: usize = SLAB_ROWS / MASK_BITS;

// ------------------------------------------------------------------ DT64

/// The person class (lab-local id; no concept is minted for it).
pub const PERSON: u32 = 0xD256_0001;
/// The friendship edge class.
pub const FRIEND_OF: u32 = 0xD257_0001;
const PERSON_VIEW: ClassId = 0xD256;
const FRIEND_VIEW: ClassId = 0xD257;

pub mod order_store;

/// Field positions of [`PERSON`].
pub mod person {
    pub const ID: u8 = 0;
    pub const NAME: u8 = 1;
    pub const AGE: u8 = 2;
}
/// Field positions of [`FRIEND_OF`].
pub mod friend {
    pub const SRC: u8 = 0;
    pub const DST: u8 = 1;
}

/// The ClassView of both classes: the field basis, in order.
pub struct View {
    person: Vec<FieldRef>,
    friend: Vec<FieldRef>,
}

impl Default for View {
    fn default() -> View {
        let f = |iri: &str, label: &str| FieldRef::new(iri, label);
        View {
            person: vec![
                f("lab:personId", "person_id"),
                f("lab:name", "name"),
                f("lab:age", "age"),
            ],
            friend: vec![
                f("lab:person1", "person1_id"),
                f("lab:person2", "person2_id"),
            ],
        }
    }
}

impl ClassView for View {
    fn fields(&self, class: ClassId) -> &[FieldRef] {
        match class {
            PERSON_VIEW => &self.person,
            FRIEND_VIEW => &self.friend,
            _ => &[],
        }
    }
    fn template(&self, _class: ClassId) -> DisplayTemplate {
        DisplayTemplate::Detail
    }
    fn dolce_category_id(&self, _class: ClassId) -> u8 {
        0
    }
}

fn view_class(class: u32) -> ClassId {
    (class >> 16) as ClassId
}

/// The selection of `names` over `class`'s basis, or `None` if a name is not
/// a field of the class. This one mask is what storage projects, what
/// compute binds and what display shows.
pub fn select(view: &View, class: u32, names: &[&str]) -> Option<WideFieldMask> {
    let basis = view.fields(view_class(class));
    let mut positions = Vec::with_capacity(names.len());
    for name in names {
        let at = basis.iter().position(|f| f.label == *name)?;
        positions.push(u8::try_from(at).ok()?);
    }
    Some(WideFieldMask::from_positions(&positions))
}

/// The selected fields in basis order: `(position, column name)`. Storage
/// reads exactly these columns; display shows exactly these.
pub fn columns<'v>(view: &'v View, class: u32, mask: &WideFieldMask) -> Vec<(u8, &'v str)> {
    view.fields(view_class(class))
        .iter()
        .enumerate()
        .filter_map(|(i, f)| {
            let p = u8::try_from(i).ok()?;
            mask.has(p).then_some((p, f.label.as_str()))
        })
        .collect()
}

/// The index of `field` among the selected fields, which is the lane index a
/// program names. `None` if the field is not selected.
pub fn lane_index(mask: &WideFieldMask, field: u8) -> Option<u16> {
    if !mask.has(field) {
        return None;
    }
    Some((0..field).filter(|&p| mask.has(p)).count() as u16)
}

/// Bind store and routes of the two classes.
pub struct World {
    pub store: BindStore,
    pub view: View,
    person_route: RouteAt,
    friend_route: RouteAt,
}

impl Default for World {
    fn default() -> World {
        World::new()
    }
}

impl World {
    pub fn new() -> World {
        let view = View::default();
        let mut store = BindStore::new();
        let schema = |field_count| ClassSchema {
            schema_rev: SchemaRev(1),
            field_count,
            extent: Extent::DR256_GLOBAL_X64,
        };
        store
            .register(&view, PERSON, schema(3), Vec::new())
            .expect("the person class registers");
        store
            .register(&view, FRIEND_OF, schema(2), Vec::new())
            .expect("the edge class registers");
        let pv = store.route(1, PERSON).expect("person routes");
        let fv = store.route(2, FRIEND_OF).expect("edge routes");
        World {
            store,
            view,
            person_route: RouteAt {
                position: 1,
                version: pv,
            },
            friend_route: RouteAt {
                position: 2,
                version: fv,
            },
        }
    }

    fn route(&self, class: u32) -> RouteAt {
        if class == PERSON {
            self.person_route
        } else {
            self.friend_route
        }
    }
}

/// Whether DT64 admits one identity reservation of `tiles` DR256 tiles. One
/// slab is 256 tiles; a 257th is past the route position's 16 bits, so a
/// population past one slab cannot be one reservation. It is stacked.
pub fn admits_reservation(tiles: u32) -> Result<(), Refusal> {
    use lance_graph_dt64::dr::{Axis, Width};
    let view = View::default();
    let mut store = BindStore::new();
    store.register(
        &view,
        PERSON,
        ClassSchema {
            schema_rev: SchemaRev(1),
            field_count: 3,
            extent: Extent {
                identity: Axis::new(Width::W256, tiles),
                ..Extent::DR256_GLOBAL_X64
            },
        },
        Vec::new(),
    )
}

// ---------------------------------------------------------------- source

/// One table at one source frame: the lanes storage returned for the
/// selected fields, borrowed. A field storage did not read has no lane.
pub struct Table<'a> {
    class: u32,
    lanes: Vec<Option<LaneRef<'a>>>,
    n_rows: usize,
    frames: FrameStore,
    population: u64,
    scope: Scope,
}

impl<'a> Table<'a> {
    /// `lanes` are `(field position, lane)` for the fields storage read;
    /// every lane holds `n_rows` rows. `source_rev` is the storage version
    /// the lanes were read at.
    pub fn new(
        class: u32,
        population: u64,
        n_rows: usize,
        source_rev: u64,
        scope: Scope,
        lanes: &[(u8, LaneRef<'a>)],
    ) -> Table<'a> {
        let width = lanes.iter().map(|(p, _)| usize::from(*p) + 1).max();
        let mut by_position = vec![None; width.unwrap_or(0)];
        for (p, lane) in lanes {
            by_position[usize::from(*p)] = Some(*lane);
        }
        Table {
            class,
            lanes: by_position,
            n_rows,
            frames: FrameStore::new(FrameRevisions {
                source: source_rev,
                relation_map: 1,
                coordinates: 1,
            }),
            population,
            scope,
        }
    }

    pub fn n_rows(&self) -> usize {
        self.n_rows
    }

    pub fn slabs(&self) -> usize {
        self.n_rows.div_ceil(SLAB_ROWS).max(1)
    }

    /// Rows of slab `s`.
    pub fn slab_rows(&self, s: usize) -> usize {
        self.n_rows.saturating_sub(s * SLAB_ROWS).min(SLAB_ROWS)
    }

    /// The row order of slab `s`: one coordinate space per slab.
    pub fn space(&self, s: usize) -> CoordSpace {
        CoordSpace {
            population: (self.population << 8) | s as u64,
            coordinates: self.frames.revisions().coordinates,
        }
    }

    pub fn frames(&self) -> &FrameStore {
        &self.frames
    }

    /// The source moved to another storage version: every frame sealed
    /// before this is stale.
    pub fn advance_source(&mut self) {
        let mut r = self.frames.revisions();
        r.source += 1;
        self.frames.advance(r);
    }

    fn slab(&self, s: usize) -> SlabSource<'_, 'a> {
        SlabSource { table: self, s }
    }
}

/// One slab of a [`Table`], as the socket DT64 binds lanes through.
pub struct SlabSource<'t, 'a> {
    table: &'t Table<'a>,
    s: usize,
}

impl LaneSource for SlabSource<'_, '_> {
    type Lane<'l>
        = LaneRef<'l>
    where
        Self: 'l;

    fn lane_of(
        &self,
        class: u32,
        rev: SchemaRev,
        field: u8,
        _basis: &FieldRef,
        _frame: &Frame,
    ) -> Option<LaneAddr> {
        if class != self.table.class || rev != SchemaRev(1) {
            return None;
        }
        self.table.lanes.get(usize::from(field))?.as_ref()?;
        Some(LaneAddr {
            space: self.table.space(self.s),
            lane: u16::from(field),
            scope: self.table.scope,
        })
    }

    fn borrow(&self, addr: &LaneAddr) -> Option<LaneRef<'_>> {
        let lo = self.s * SLAB_ROWS;
        let hi = lo + self.table.slab_rows(self.s);
        Some(match (*self.table.lanes.get(usize::from(addr.lane))?)? {
            LaneRef::I32(v) => LaneRef::I32(v.get(lo..hi)?),
            LaneRef::U32(v) => LaneRef::U32(v.get(lo..hi)?),
            _ => return None,
        })
    }

    fn frames(&self) -> &FrameStore {
        &self.table.frames
    }
}

// ---------------------------------------------------------------- planes

/// One mask plane over one slab, with the row order it is in.
/// Read as 256 selections of 256 positions through [`Plane::mask`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Plane {
    pub space: CoordSpace,
    pub words: Vec<u64>,
}

impl Plane {
    /// A plane of `rows` rows from a per-row predicate.
    pub fn from_fn(space: CoordSpace, rows: usize, mut set: impl FnMut(usize) -> bool) -> Plane {
        let mut words = vec![0u64; words_for(rows)];
        for i in 0..rows {
            if set(i) {
                words[i / 64] |= 1 << (i % 64);
            }
        }
        Plane { space, words }
    }

    pub fn count(&self) -> u64 {
        self.words.iter().map(|w| u64::from(w.count_ones())).sum()
    }

    /// Selections this plane holds: one per 256 positions, the last one short
    /// when the slab is.
    pub fn masks(&self) -> usize {
        self.words.len().div_ceil(MASK_WORDS)
    }

    /// Selection `row` of this plane, borrowed in place. `None` past the plane
    /// or past row 255.
    pub fn mask(&self, row: usize) -> Option<Selection<'_>> {
        let at = u8::try_from(row).ok()?;
        let lo = row * MASK_WORDS;
        if lo >= self.words.len() {
            return None;
        }
        let hi = (lo + MASK_WORDS).min(self.words.len());
        Some(Selection {
            space: self.space,
            row: at,
            words: &self.words[lo..hi],
        })
    }

    /// Every selection, in row order.
    pub fn selections(&self) -> impl Iterator<Item = Selection<'_>> + '_ {
        (0..self.masks()).filter_map(|row| self.mask(row))
    }
}

/// One reusable selection of a slab plane: 256 positions, borrowed from the
/// plane's own words, with the row order it is in. The words are the chunks
/// of a `WideFieldMask` over the same positions (chunk `k` = positions
/// `64k..64k+63`); nothing is copied to read them.
///
/// A plane's last selection is short when the slab is: fewer than four
/// words, and every position past the slab's rows reads absent.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Selection<'p> {
    space: CoordSpace,
    row: u8,
    words: &'p [u64],
}

impl<'p> Selection<'p> {
    /// The row order the selection is in: its plane's.
    pub fn space(&self) -> CoordSpace {
        self.space
    }

    /// Which of the plane's selections this is.
    pub fn row(&self) -> u8 {
        self.row
    }

    /// The chunks, borrowed from the plane. At most [`MASK_WORDS`].
    pub fn words(&self) -> &'p [u64] {
        self.words
    }

    /// Is position `n` selected?
    pub fn has(&self, n: u8) -> bool {
        self.words
            .get(usize::from(n) / 64)
            .is_some_and(|w| w >> (n % 64) & 1 == 1)
    }

    /// Selected positions.
    pub fn count(&self) -> u32 {
        self.words.iter().map(|w| w.count_ones()).sum()
    }

    pub fn is_empty(&self) -> bool {
        self.words.iter().all(|&w| w == 0)
    }

    /// Whether `mask` selects exactly these positions, whatever width either
    /// side stores. Reads both in place.
    pub fn same_bits(&self, mask: &WideFieldMask) -> bool {
        let other = mask.words();
        (0..self.words.len().max(other.len()))
            .all(|i| self.words.get(i).copied().unwrap_or(0) == other.get(i).copied().unwrap_or(0))
    }
}

/// Global to local as a mask: a global id's local row is its rank among the
/// present ids. Stacked over the outer byte: `local = base[outer] + rank`.
///
/// Built once per frame. Rank preserves order, so rows compacted through it
/// stay in the source's order.
#[derive(Debug, Clone)]
pub struct RankMap {
    /// Presence over the global id space, 64 ids per word.
    words: Vec<u64>,
    /// Present ids before each word.
    prefix: Vec<u64>,
    present: u64,
}

impl RankMap {
    /// From presence over `globals` ids. Refuses a domain past 256 slabs.
    pub fn new(globals: usize, mut present: impl FnMut(usize) -> bool) -> Option<RankMap> {
        if globals > MAX_SLABS * SLAB_ROWS {
            return None;
        }
        let mut words = vec![0u64; words_for(globals)];
        for g in 0..globals {
            if present(g) {
                words[g / 64] |= 1 << (g % 64);
            }
        }
        let mut prefix = Vec::with_capacity(words.len());
        let mut run = 0u64;
        for w in &words {
            prefix.push(run);
            run += u64::from(w.count_ones());
        }
        Some(RankMap {
            words,
            prefix,
            present: run,
        })
    }

    /// Present ids, which is the local row count.
    pub fn len(&self) -> u64 {
        self.present
    }

    pub fn is_empty(&self) -> bool {
        self.present == 0
    }

    pub fn contains(&self, global: u64) -> bool {
        let w = (global / 64) as usize;
        self.words
            .get(w)
            .is_some_and(|word| word >> (global % 64) & 1 == 1)
    }

    /// The local row of `global`, or `None` if it is absent or out of range.
    pub fn local(&self, global: u64) -> Option<u64> {
        if !self.contains(global) {
            return None;
        }
        let w = (global / 64) as usize;
        let below = self.words[w] & ((1u64 << (global % 64)) - 1);
        Some(self.prefix[w] + u64::from(below.count_ones()))
    }

    /// Presence words, for a storage reader that turns them into its own
    /// row allow-list.
    pub fn words(&self) -> &[u64] {
        &self.words
    }
}

// -------------------------------------------------------------- prepared

/// Why a fold was refused.
#[derive(Debug, Clone, PartialEq)]
pub enum Refused {
    /// The binding, frame, route, scope or row order (DT64's refusal).
    Lane(LaneRefusal),
    /// A plane is in another row order than the slab's binding.
    Misaligned { slab: usize, plane: usize },
    /// The program names a field the selection does not hold.
    FieldNotSelected { field: u8 },
    /// The population is past 256 slabs.
    TooManySlabs { slabs: usize },
    /// mask-RISC refused the program.
    Exec(ExecError),
    /// An fk names no row of the node domain.
    ForeignKeyOutOfRange { edge: usize },
    /// The fold was compiled against another field selection, so its lane
    /// indices would name other columns of this preparation.
    SelectionChanged,
    /// The order index was built over another table, or over this table at
    /// another revision; its keys and row order are not this table's.
    IndexOfAnotherTable,
    /// `Fold::count` was called on a fold whose terminal is not a count.
    NotACount,
    /// The sum over the slab stack left `i64`.
    SumOverflow,
}

impl From<LaneRefusal> for Refused {
    fn from(r: LaneRefusal) -> Refused {
        Refused::Lane(r)
    }
}
impl From<ExecError> for Refused {
    fn from(e: ExecError) -> Refused {
        Refused::Exec(e)
    }
}

struct PreparedSlab {
    binding: LaneBinding,
    rows: usize,
    /// Three planes per slab, in the order the program names them.
    planes: [Plane; 3],
}

/// Everything resolved once per source frame for one table and one
/// selection: a binding and three planes per slab.
pub struct Prepared {
    class: u32,
    fields: WideFieldMask,
    slabs: Vec<PreparedSlab>,
    /// Foreign-key reads spent building fk planes. Zero for a node table.
    pub lookups: u64,
}

impl Prepared {
    pub fn slabs(&self) -> usize {
        self.slabs.len()
    }

    /// Bytes of planes held.
    pub fn plane_bytes(&self) -> usize {
        self.slabs
            .iter()
            .flat_map(|s| s.planes.iter())
            .map(|p| p.words.len() * 8)
            .sum()
    }

    /// Replace plane `plane` of slab `slab` (a test seam for misalignment).
    pub fn set_plane(&mut self, slab: usize, plane: usize, with: Plane) {
        self.slabs[slab].planes[plane] = with;
    }

    pub fn plane(&self, slab: usize, plane: usize) -> &Plane {
        &self.slabs[slab].planes[plane]
    }
}

fn bind_slabs(
    world: &World,
    table: &Table<'_>,
    fields: &WideFieldMask,
) -> Result<Vec<(LaneBinding, usize)>, Refused> {
    let slabs = table.slabs();
    if slabs > MAX_SLABS {
        return Err(Refused::TooManySlabs { slabs });
    }
    let frame = table.frames.seal();
    (0..slabs)
        .map(|s| {
            let binding = world.store.bind_lanes(
                &world.view,
                &table.frames,
                &table.slab(s),
                LaneRequest {
                    route: world.route(table.class),
                    rev: SchemaRev(1),
                    fields,
                    frame: &frame,
                    scope: table.scope,
                    selection: table.space(s),
                },
            )?;
            Ok((binding, table.slab_rows(s)))
        })
        .collect()
}

/// Prepare a node table: bind `fields` per slab and take the presence and
/// scope planes. `presence(row)` and `scope(row)` are per table row. The
/// third plane is all ones until a program writes its own selection.
pub fn prepare_nodes(
    world: &World,
    table: &Table<'_>,
    fields: &WideFieldMask,
    mut presence: impl FnMut(usize) -> bool,
    mut scope: impl FnMut(usize) -> bool,
) -> Result<Prepared, Refused> {
    let slabs = bind_slabs(world, table, fields)?
        .into_iter()
        .enumerate()
        .map(|(s, (binding, rows))| {
            let space = table.space(s);
            let base = s * SLAB_ROWS;
            PreparedSlab {
                binding,
                rows,
                planes: [
                    Plane::from_fn(space, rows, |i| presence(base + i)),
                    Plane::from_fn(space, rows, |i| scope(base + i)),
                    Plane::from_fn(space, rows, |_| true),
                ],
            }
        })
        .collect();
    Ok(Prepared {
        class: table.class,
        fields: fields.clone(),
        slabs,
        lookups: 0,
    })
}

/// Prepare an edge table against a node domain: per slab, the edge
/// presence plane and one plane per endpoint saying whether the row that
/// endpoint names is visible. The fk is read here, once per frame, and
/// counted; no query reads it again.
///
/// `node_visible(global id)` answers for the node the fk names; `None` means
/// the id is outside the node domain.
pub fn prepare_edges(
    world: &World,
    table: &Table<'_>,
    mut edge_present: impl FnMut(usize) -> bool,
    mut node_visible: impl FnMut(u32) -> Option<bool>,
) -> Result<Prepared, Refused> {
    let fields = WideFieldMask::from_positions(&[friend::SRC, friend::DST]);
    let mut lookups = 0u64;
    let mut slabs = Vec::new();
    for (s, (binding, rows)) in bind_slabs(world, table, &fields)?.into_iter().enumerate() {
        let space = table.space(s);
        let base = s * SLAB_ROWS;
        let source = table.slab(s);
        let mut fk: [&[u32]; 2] = [&[], &[]];
        for lane in binding.borrow_all(&world.store, &world.view, &table.frames, &source)? {
            match lane? {
                (friend::SRC, LaneRef::U32(v)) => fk[0] = v,
                (friend::DST, LaneRef::U32(v)) => fk[1] = v,
                (field, _) => return Err(Refused::FieldNotSelected { field }),
            }
        }
        let mut endpoint = |lane: &[u32]| -> Result<Plane, Refused> {
            let mut words = vec![0u64; words_for(rows)];
            for (i, &id) in lane.iter().enumerate() {
                lookups += 1;
                match node_visible(id) {
                    Some(true) => words[i / 64] |= 1 << (i % 64),
                    Some(false) => {}
                    None => return Err(Refused::ForeignKeyOutOfRange { edge: base + i }),
                }
            }
            Ok(Plane { space, words })
        };
        let src_ok = endpoint(fk[0])?;
        let dst_ok = endpoint(fk[1])?;
        slabs.push(PreparedSlab {
            binding,
            rows,
            planes: [
                Plane::from_fn(space, rows, |i| edge_present(base + i)),
                src_ok,
                dst_ok,
            ],
        });
    }
    Ok(Prepared {
        class: table.class,
        fields,
        slabs,
        lookups,
    })
}

// ---------------------------------------------------------------- kernel

/// What a fold answers over the selected rows. The field is an `I32` lane
/// of the prepared selection.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Agg {
    Count,
    Sum(u8),
    Min(u8),
    Max(u8),
}

/// The answer of one fold over the whole slab stack.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Answer {
    Count(u64),
    Sum(i64),
    /// `None` when no row is selected.
    Min(Option<i32>),
    Max(Option<i32>),
}

/// A query compiled once: a mask-RISC program over three planes, and its
/// scratch.
pub struct Fold {
    program: Program,
    scratch: Scratch<'static>,
    /// The field selection the program's lane indices were resolved in.
    /// `None` when the program reads no lane.
    compiled: Option<WideFieldMask>,
    agg: Agg,
}

/// Bytes of the scratch arena `Scratch::for_program` carves for `program`
/// over `n_rows` rows.
pub fn scratch_bytes_of(program: &Program, n_rows: usize) -> usize {
    if !program.requires_scratch() {
        return 0;
    }
    scratch_words_for(tile_words_for(n_rows), program.scratch_slots as usize).unwrap_or(0) * 8
}

fn terminal(prepared: &Prepared, agg: Agg, mask: Operand) -> Result<Terminal, Refused> {
    let lane =
        |field| lane_index(&prepared.fields, field).ok_or(Refused::FieldNotSelected { field });
    Ok(match agg {
        Agg::Count => Terminal::Count { mask },
        Agg::Sum(f) => Terminal::MaskedSumI32 {
            mask,
            lane: lane(f)?,
        },
        Agg::Min(f) => Terminal::MaskedMinI32 {
            mask,
            lane: lane(f)?,
        },
        Agg::Max(f) => Terminal::MaskedMaxI32 {
            mask,
            lane: lane(f)?,
        },
    })
}

impl Fold {
    /// `count(presence & (field > t) & scope)`: the selection is computed
    /// from the bound lane, the other two planes are the prepared ones.
    pub fn count_gt(prepared: &Prepared, field: u8, t: i32) -> Result<Fold, Refused> {
        let lane =
            lane_index(&prepared.fields, field).ok_or(Refused::FieldNotSelected { field })?;
        let program = Program::new(
            vec![
                MaskOp::Pred {
                    pred: Pred::GtI32 { lane, t },
                    under: None,
                    dst: 0,
                },
                MaskOp::Ternlog {
                    imm: AND3,
                    a: Operand::Plane(0),
                    b: Operand::Scratch(0),
                    c: Operand::Plane(1),
                    dst: 1,
                },
            ],
            Terminal::Count {
                mask: Operand::Scratch(1),
            },
        );
        Fold::new(program, Some(prepared.fields.clone()), Agg::Count)
    }

    /// `agg` over `a & b & c`, the three prepared planes. The selection is
    /// one ternlog; only the terminal reads a lane.
    pub fn and3(prepared: &Prepared, agg: Agg) -> Result<Fold, Refused> {
        let program = Program::new(
            vec![MaskOp::Ternlog {
                imm: AND3,
                a: Operand::Plane(0),
                b: Operand::Plane(1),
                c: Operand::Plane(2),
                dst: 0,
            }],
            terminal(prepared, agg, Operand::Scratch(0))?,
        );
        let compiled = (agg != Agg::Count).then(|| prepared.fields.clone());
        Fold::new(program, compiled, agg)
    }

    /// `agg` over `plane 0 & plane 1 & (lo < field < hi)`. The bounds are
    /// compared from the bound lane only in words where plane 0 has a row
    /// (the survivor skip), the upper one only where the lower one held.
    pub fn between(
        prepared: &Prepared,
        field: u8,
        lo: i32,
        hi: i32,
        agg: Agg,
    ) -> Result<Fold, Refused> {
        let lane =
            lane_index(&prepared.fields, field).ok_or(Refused::FieldNotSelected { field })?;
        let program = Program::new(
            vec![
                MaskOp::Pred {
                    pred: Pred::GtI32 { lane, t: lo },
                    under: Some(Operand::Plane(0)),
                    dst: 0,
                },
                MaskOp::Pred {
                    pred: Pred::LtI32 { lane, t: hi },
                    under: Some(Operand::Scratch(0)),
                    dst: 1,
                },
                MaskOp::Ternlog {
                    imm: AND3,
                    a: Operand::Plane(0),
                    b: Operand::Plane(1),
                    c: Operand::Scratch(1),
                    dst: 2,
                },
            ],
            terminal(prepared, agg, Operand::Scratch(2))?,
        );
        Fold::new(program, Some(prepared.fields.clone()), agg)
    }

    /// `count(a & b & c)` over the three prepared planes. Reads no lane.
    pub fn count_and3() -> Result<Fold, Refused> {
        let program = Program::new(
            vec![MaskOp::Ternlog {
                imm: AND3,
                a: Operand::Plane(0),
                b: Operand::Plane(1),
                c: Operand::Plane(2),
                dst: 0,
            }],
            Terminal::Count {
                mask: Operand::Scratch(0),
            },
        );
        Fold::new(program, None, Agg::Count)
    }

    fn new(program: Program, compiled: Option<WideFieldMask>, agg: Agg) -> Result<Fold, Refused> {
        let scratch = Scratch::for_program(&program, SLAB_ROWS)?;
        Ok(Fold {
            program,
            scratch,
            compiled,
            agg,
        })
    }

    /// Bytes of scratch the fold holds, as the executor sizes it: one tile
    /// per slot the program writes, plus the validation bitmap.
    pub fn scratch_bytes(&self) -> usize {
        scratch_bytes_of(&self.program, SLAB_ROWS)
    }

    /// [`Fold::answer`] of a count fold.
    pub fn count(
        &mut self,
        world: &World,
        table: &Table<'_>,
        prepared: &Prepared,
    ) -> Result<u64, Refused> {
        match self.answer(world, table, prepared)? {
            Answer::Count(c) => Ok(c),
            _ => Err(Refused::NotACount),
        }
    }

    /// Check every slab's binding, fold its three planes, combine the slab
    /// answers. Slabs are disjoint row ranges, so counts and sums add and
    /// extrema combine.
    pub fn answer(
        &mut self,
        world: &World,
        table: &Table<'_>,
        prepared: &Prepared,
    ) -> Result<Answer, Refused> {
        debug_assert_eq!(prepared.class, table.class);
        if self
            .compiled
            .as_ref()
            .is_some_and(|c| *c != prepared.fields)
        {
            return Err(Refused::SelectionChanged);
        }
        let mut total = match self.agg {
            Agg::Count => Answer::Count(0),
            Agg::Sum(_) => Answer::Sum(0),
            Agg::Min(_) => Answer::Min(None),
            Agg::Max(_) => Answer::Max(None),
        };
        for (s, slab) in prepared.slabs.iter().enumerate() {
            let source = table.slab(s);
            let mut lanes = [LaneRef::U32(&[]); 4];
            let mut n = 0;
            let bound =
                slab.binding
                    .borrow_all(&world.store, &world.view, &table.frames, &source)?;
            if self.compiled.is_some() {
                for lane in bound {
                    let (_, lane) = lane?;
                    lanes[n] = lane;
                    n += 1;
                }
            }
            let selection = slab.binding.selection();
            for (p, plane) in slab.planes.iter().enumerate() {
                if plane.space != selection {
                    return Err(Refused::Misaligned { slab: s, plane: p });
                }
            }
            let masks: [&[u64]; 3] = [
                &slab.planes[0].words,
                &slab.planes[1].words,
                &slab.planes[2].words,
            ];
            let planes = Planes {
                n_rows: slab.rows,
                masks: &masks,
                lanes: &lanes[..n],
            };
            let value = execute_into(
                &self.program,
                &planes,
                &Foreign {
                    planes: &[],
                    lanes: &[],
                },
                &mut self.scratch,
                Out::None,
            )?;
            total = match (total, value) {
                (Answer::Count(t), Value::Count(c)) => Answer::Count(t + c as u64),
                (Answer::Sum(t), Value::SumI64(v)) => {
                    Answer::Sum(t.checked_add(v).ok_or(Refused::SumOverflow)?)
                }
                (Answer::Min(t), Value::OptI32(v)) => {
                    Answer::Min([t, v].into_iter().flatten().min())
                }
                (Answer::Max(t), Value::OptI32(v)) => {
                    Answer::Max([t, v].into_iter().flatten().max())
                }
                (_, other) => unreachable!("{:?} returned {other:?}", self.agg),
            };
        }
        Ok(total)
    }
}

/// The mean of `count` values summing to `sum`, in the values' own unit,
/// rounded to the nearest integer with ties toward positive infinity (the
/// 1BRC reference's rounding of a tenths-of-a-degree mean). `None` when
/// `count` is zero.
pub fn mean_round_half_up(sum: i64, count: u64) -> Option<i64> {
    let n = i128::from(count);
    (n != 0).then(|| (2 * i128::from(sum) + n).div_euclid(2 * n) as i64)
}

// --------------------------------------------------------------- display

/// Row `row` of `table` shown through `mask`: `(column name, value)` for
/// exactly the selected fields, in basis order. Numeric lanes only.
pub fn display_row(
    view: &View,
    table: &Table<'_>,
    mask: &WideFieldMask,
    row: usize,
) -> Option<Vec<(String, i64)>> {
    columns(view, table.class, mask)
        .into_iter()
        .map(|(p, name)| {
            let v = match (*table.lanes.get(usize::from(p))?)? {
                LaneRef::I32(v) => i64::from(*v.get(row)?),
                LaneRef::U32(v) => i64::from(*v.get(row)?),
                _ => return None,
            };
            Some((name.to_string(), v))
        })
        .collect()
}

// ------------------------------------------------------------ order index

/// A selection that is never written as a mask: the rows of a table in the
/// order of one key, kept beside the table. The table's lanes are not
/// moved; only this index is in key order, the way a storage version lists
/// rows without rewriting them.
///
/// A predicate on the key is then a position range `[lo, hi)` of the index.
/// What a reader may see is one resident plane in index order, prepared once
/// per frame. `count` is that plane's popcount over the range: no selection
/// mask is produced, and only the words the range touches are read.
///
/// The index is its own row order, so its planes have their own
/// [`CoordSpace`] and cannot be folded with planes in table order.
pub struct OrderIndex {
    /// Index position -> table row.
    order: Vec<u32>,
    /// The key at each index position, ascending.
    keys: Vec<i32>,
    population: u64,
    /// The table the keys and the order were read from.
    origin: Origin,
}

/// What identifies the table an [`OrderIndex`] was built over.
#[derive(Clone, Copy, PartialEq, Eq)]
struct Origin {
    class: u32,
    population: u64,
    n_rows: usize,
    revisions: FrameRevisions,
}

impl Origin {
    fn of(table: &Table<'_>) -> Origin {
        Origin {
            class: table.class,
            population: table.population,
            n_rows: table.n_rows,
            revisions: table.frames.revisions(),
        }
    }
}

impl OrderIndex {
    /// Sort `table`'s rows by the `i32` field `key`. `population` names the
    /// index's row order and must differ from the table's.
    pub fn by_i32(table: &Table<'_>, key: u8, population: u64) -> Option<OrderIndex> {
        let Some(LaneRef::I32(lane)) = *table.lanes.get(usize::from(key))? else {
            return None;
        };
        if population == table.population || lane.len() > u32::MAX as usize {
            return None;
        }
        let mut order: Vec<u32> = (0..lane.len() as u32).collect();
        order.sort_by_key(|&r| (lane[r as usize], r));
        let keys = order.iter().map(|&r| lane[r as usize]).collect();
        Some(OrderIndex {
            order,
            keys,
            population,
            origin: Origin::of(table),
        })
    }

    /// Sort the table's rows by a computed key.
    ///
    /// Keyed by each edge's source position in a node index
    /// ([`OrderIndex::positions`]), the edges of any node range `[lo, hi)`
    /// of that index are one contiguous range of this one
    /// ([`OrderIndex::between`]`(lo, hi - 1)`). That is what a CSR layout
    /// gives, and CSR is only the mental model here: nothing but this
    /// ordinary index is built, no offsets array and no adjacency type.
    pub fn by_key(
        table: &Table<'_>,
        population: u64,
        mut key: impl FnMut(usize) -> i32,
    ) -> Option<OrderIndex> {
        let rows = table.n_rows;
        if population == table.population || rows > u32::MAX as usize {
            return None;
        }
        let row_keys: Vec<i32> = (0..rows).map(&mut key).collect();
        let mut order: Vec<u32> = (0..rows as u32).collect();
        order.sort_by_key(|&r| (row_keys[r as usize], r));
        let keys = order.iter().map(|&r| row_keys[r as usize]).collect();
        Some(OrderIndex {
            order,
            keys,
            population,
            origin: Origin::of(table),
        })
    }

    /// Table row -> index position: the inverse of the order.
    pub fn positions(&self) -> Vec<u32> {
        let mut at = vec![0u32; self.order.len()];
        for (i, &r) in self.order.iter().enumerate() {
            at[r as usize] = i as u32;
        }
        at
    }

    pub fn len(&self) -> usize {
        self.order.len()
    }

    pub fn is_empty(&self) -> bool {
        self.order.is_empty()
    }

    /// The table row at index position `at`.
    pub fn row(&self, at: usize) -> Option<u32> {
        self.order.get(at).copied()
    }

    /// The index positions whose key is greater than `t`.
    pub fn gt(&self, t: i32) -> core::ops::Range<usize> {
        self.keys.partition_point(|&k| k <= t)..self.keys.len()
    }

    /// The index positions whose key lies in `[lo, hi]`.
    pub fn between(&self, lo: i32, hi: i32) -> core::ops::Range<usize> {
        self.keys.partition_point(|&k| k < lo)..self.keys.partition_point(|&k| k <= hi)
    }

    /// Refuse a table other than the one this index was built over.
    fn check(&self, table: &Table<'_>) -> Result<(), Refused> {
        if self.origin == Origin::of(table) {
            Ok(())
        } else {
            Err(Refused::IndexOfAnotherTable)
        }
    }

    fn space(&self, s: usize, coordinates: u64) -> CoordSpace {
        CoordSpace {
            population: (self.population << 8) | s as u64,
            coordinates,
        }
    }
}

/// One resident visibility plane per slab of an [`OrderIndex`], and the
/// bindings that witness the table's frame.
pub struct PreparedOrder {
    bindings: Vec<LaneBinding>,
    planes: Vec<Plane>,
    rows: usize,
    scratch: Scratch<'static>,
}

/// Prepare `index`: bind the table's slabs (the frame witness) and write
/// `visible(table row)` once, in index order.
pub fn prepare_order(
    world: &World,
    table: &Table<'_>,
    fields: &WideFieldMask,
    index: &OrderIndex,
    mut visible: impl FnMut(usize) -> bool,
) -> Result<PreparedOrder, Refused> {
    index.check(table)?;
    let slabs = index.len().div_ceil(SLAB_ROWS).max(1);
    if slabs > MAX_SLABS {
        return Err(Refused::TooManySlabs { slabs });
    }
    let bindings = bind_slabs(world, table, fields)?
        .into_iter()
        .map(|(b, _)| b)
        .collect();
    let coordinates = table.frames.revisions().coordinates;
    let planes = (0..slabs)
        .map(|s| {
            let base = s * SLAB_ROWS;
            let rows = index.len().saturating_sub(base).min(SLAB_ROWS);
            Plane::from_fn(index.space(s, coordinates), rows, |i| {
                visible(index.order[base + i] as usize)
            })
        })
        .collect();
    let probe = range_program(0, 0);
    Ok(PreparedOrder {
        bindings,
        planes,
        rows: index.len(),
        scratch: Scratch::for_program(&probe, SLAB_ROWS)?,
    })
}

fn range_program(lo: u32, hi: u32) -> Program {
    Program::new(
        vec![MaskOp::Pred {
            pred: Pred::Range { lo, hi },
            under: Some(Operand::Plane(0)),
            dst: 0,
        }],
        Terminal::Count {
            mask: Operand::Scratch(0),
        },
    )
}

impl PreparedOrder {
    pub fn plane(&self, slab: usize) -> &Plane {
        &self.planes[slab]
    }

    pub fn set_plane(&mut self, slab: usize, with: Plane) {
        self.planes[slab] = with;
    }

    /// Visible rows in index positions `range`. Checks the table's frame,
    /// then folds one range per slab the range touches.
    pub fn count(
        &mut self,
        world: &World,
        table: &Table<'_>,
        index: &OrderIndex,
        range: core::ops::Range<usize>,
    ) -> Result<u64, Refused> {
        for (s, binding) in self.bindings.iter().enumerate() {
            binding.check(&world.store, &world.view, &table.frames, &table.slab(s))?;
        }
        index.check(table)?;
        let coordinates = table.frames.revisions().coordinates;
        let (lo, hi) = (range.start.min(self.rows), range.end.min(self.rows));
        let mut total = 0u64;
        if lo >= hi {
            return Ok(0);
        }
        for s in lo / SLAB_ROWS..=(hi - 1) / SLAB_ROWS {
            let plane = &self.planes[s];
            if plane.space != index.space(s, coordinates) {
                return Err(Refused::Misaligned { slab: s, plane: 0 });
            }
            let base = s * SLAB_ROWS;
            let rows = self.rows.saturating_sub(base).min(SLAB_ROWS);
            let a = lo.max(base) - base;
            let b = hi.min(base + rows) - base;
            let masks: [&[u64]; 1] = [&plane.words];
            let planes = Planes {
                n_rows: rows,
                masks: &masks,
                lanes: &[],
            };
            match execute_into(
                &range_program(a as u32, b as u32),
                &planes,
                &Foreign {
                    planes: &[],
                    lanes: &[],
                },
                &mut self.scratch,
                Out::None,
            )? {
                Value::Count(c) => total += c as u64,
                other => unreachable!("a Count terminal returned {other:?}"),
            }
        }
        Ok(total)
    }
}
