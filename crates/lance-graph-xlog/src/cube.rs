//! The resident hypercube: one execution context for every relation, and
//! views into it (D-XLOG-NATIVE-7, PR #1911; board entry `2026-11-01-xlog-resident-hypercube-context.md`).
//!
//! [`support`](crate::support) and [`bag_count`](crate::bag_count) build their
//! lanes per call, oriented for one join. A [`Cube`] builds them once, for
//! every relation and both columns, against one shared coordinate map: each
//! id in either column of any relation is looked up once, at construction,
//! and stored as its local position in the tile. Afterwards any relation can
//! join any other on either column without new lanes, new lookups or a new
//! index.
//!
//! The operations a caller composes ([`Cube::scan`], [`Cube::bind`],
//! [`Cube::join`]) return [`View`]s. A view is a `Copy` handle into the
//! cube's node list; making one touches no rows and runs nothing. Only the
//! terminals run:
//!
//! | terminal | answer | how |
//! |---|---|---|
//! | [`Cube::support`] | which join values have a partner on both sides | two `ScatterOrU32` programs, `and` |
//! | [`Cube::bag_count`] | derivations per join value | two `GroupReduce Count` programs, checked `i64` product |
//! | [`Cube::materialize_rows`] | the joined tuples | the one named place rows are built |
//!
//! Every terminal re-checks the cube's `MapTable` frame, so a view over a
//! stale coordinate assignment is refused. [`CubeStats`] counts programs run
//! and rows materialized, so a test can show that composing views runs
//! nothing.
//!
//! Scope: one tile (at most 256 ids across all relations), binary relations,
//! a join of two sides, at most one bound per side and only on the side's
//! free column. Union and difference are not here as views.
//!
//! [`Cube::closure`] is the one recursive terminal: `edge⁺` of one binary
//! relation, computed semi-naively over row masks built from the relation's
//! resident lanes (D-STAUNEN-5 P1). The delta difference is `mask_andnot`
//! inside the cube; nothing is added to the cube and no tuple is built.

mod body;
mod lend;
pub use body::{Atom, BodyAnswer, BodyRefusal, MAX_BODY_VARS};
pub use lend::CubeLanes;

use std::cell::{Cell, OnceCell};
use std::sync::atomic::{AtomicU64, Ordering};

use lance_graph_dt64::dr::{
    Axis, Frame, LocalMap, LocateRefusal, MapRead, MapRefusal, MapTable, PairDir, PairPlane,
    TileId, TileMap, Width,
};
use lance_graph_dt64::{relation_reach, ReachBuf, ReachRefusal, RelNo, RelationSource};
use lance_graph_mask_risc::{LaneRef, Out};
use lance_graph_quack::{Agg, Cmp, Col, Filter, GroupAddr, GroupAgg, Mask, Query};
use ndarray::simd::{checked_binary_i64, mask_and, mask_andnot, mask_any, mask_or_assign, I64Op};

use crate::wide::{self, DeltaLog, WideRefusal, WideRows, WideStats, BLOCK_POSITIONS};
use crate::{all_live, run_lanes, NativeRefusal};

/// A relation's index in the order given to [`Cube::new`].
pub type RelId = u32;

/// A stored one-column set's index, in the order [`Cube::add_set`] added it.
pub type SetId = u32;

/// An expression over stored sets, the input of [`Cube::reach`]
/// (D-XLOG-W5-B, decision D2). The leaves are stored sets and the only
/// operators are and, or and and-not: there are no constants and no
/// complement of the universe. It is plain data; building one runs nothing,
/// and [`Cube::reach`] evaluates it inside the terminal, so no mask computed
/// by one program is handed to another.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SetExpr {
    /// A set [`Cube::add_set`] stored.
    Set(SetId),
    /// Both.
    And(Box<SetExpr>, Box<SetExpr>),
    /// Either.
    Or(Box<SetExpr>, Box<SetExpr>),
    /// The first and not the second.
    AndNot(Box<SetExpr>, Box<SetExpr>),
    /// [`Cube::reach`] along relation `rel` from `seed`, every step landing
    /// in `target`, seeds included. As a node, a reachable set composes with
    /// the other operators inside one terminal (`reach ∧ blocked`), so a
    /// recursive head is never stored and handed back as a mask.
    Reach {
        /// The binary relation stepped along, column 0 to column 1.
        rel: RelId,
        /// Where the walk starts.
        seed: Box<SetExpr>,
        /// Where each step may land.
        target: Box<SetExpr>,
    },
}

impl SetExpr {
    /// `a ∧ b`.
    pub fn and(a: SetExpr, b: SetExpr) -> Self {
        Self::And(Box::new(a), Box::new(b))
    }
    /// `a ∨ b`.
    pub fn or(a: SetExpr, b: SetExpr) -> Self {
        Self::Or(Box::new(a), Box::new(b))
    }
    /// `a ∧ ¬b`.
    pub fn and_not(a: SetExpr, b: SetExpr) -> Self {
        Self::AndNot(Box::new(a), Box::new(b))
    }
    /// `seed ∪ reach(seed)` along `rel`, steps landing in `target`.
    pub fn reach(rel: RelId, seed: SetExpr, target: SetExpr) -> Self {
        Self::Reach {
            rel,
            seed: Box::new(seed),
            target: Box::new(target),
        }
    }
}

/// A handle into a [`Cube`]. Copying it copies no rows.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct View {
    cube: u64,
    node: u32,
}

/// One logged edge operation of a [`Cube::stage`] batch.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EdgeOp {
    /// [`Cube::add_edge`].
    Add,
    /// [`Cube::retire_edge`].
    Retire,
}

/// A batch of edge operations [`Cube::stage`] has checked, placed at their
/// coordinates and not yet applied. Holds no borrow, so the caller may
/// advance the table between staging and [`Cube::commit`].
#[derive(Debug)]
pub struct StagedEdges {
    cube: u64,
    /// The cube's frame when the batch was staged: every coordinate in the
    /// batch was placed under it. [`Cube::commit`] refuses the batch if the
    /// cube has adopted another frame since.
    staged_under: Frame,
    /// The grown tile and the exact staged map: its width and its local →
    /// global table. [`Cube::commit`] compares the frame's map against all of
    /// it, not only its length: two maps of equal length can assign
    /// different ids to the same positions.
    grown: Option<StagedMap>,
    n: usize,
    ops: Vec<(RelId, EdgeOp, u32, u32)>,
    rebind: Vec<(usize, u32)>,
}

/// The map a batch was staged under for its grown tile.
#[derive(Debug)]
struct StagedMap {
    tile: TileId,
    width: Width,
    globals: Vec<u64>,
}

impl StagedMap {
    /// Whether `view` is exactly this map: same width, same ids at the same
    /// positions, same length.
    fn is(&self, view: TileMap<'_>) -> bool {
        view.width() == self.width && view.globals() == self.globals.as_slice()
    }
}

/// The closure of one relation's BASE lanes (not its log), with the
/// provenance that makes it usable as the starting point of
/// [`Cube::refresh_closure`] and [`Cube::stage_closure`]: the cube, the
/// relation, the base generation and the frame it was computed under. Only
/// [`Cube::base_closure`] makes one, so a caller cannot present arbitrary
/// rows, a closure of the effective edges, or a closure of an older base as
/// a base closure.
#[derive(Clone, Debug)]
pub struct BaseClosure {
    cube: u64,
    rel: RelId,
    generation: u64,
    frame: Frame,
    closure: (WideRows, WideStats),
}

impl BaseClosure {
    /// The closure rows.
    pub fn rows(&self) -> &WideRows {
        &self.closure.0
    }

    /// The rows and the work that computed them.
    pub fn as_pair(&self) -> &(WideRows, WideStats) {
        &self.closure
    }

    /// The relation it is the base closure of.
    pub fn rel(&self) -> RelId {
        self.rel
    }

    /// The base generation it was computed at ([`Cube::base_generation`]).
    pub fn generation(&self) -> u64 {
        self.generation
    }
}

impl StagedEdges {
    /// Operations in the batch.
    pub fn len(&self) -> usize {
        self.ops.len()
    }

    /// Whether the batch holds no operation.
    pub fn is_empty(&self) -> bool {
        self.ops.is_empty()
    }
}

#[derive(Clone, Copy, Debug)]
enum Node {
    Scan(RelId),
    /// `col == global`. `value` is `global`'s coordinate under the cube's
    /// frame, `None` while no tile holds it (no row matches). The global id
    /// is kept so an extending rebase can place a value that was unmapped
    /// when the view was built: the view means "column equals this id", and
    /// keeps meaning it under every later frame.
    Bind {
        rel: RelId,
        col: u8,
        value: Option<u32>,
        global: u32,
    },
    /// `left.lcol == right.rcol`.
    Join {
        left: View,
        lcol: u8,
        right: View,
        rcol: u8,
    },
}

/// Why a view was not built or a terminal did not run.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum CubeRefusal {
    /// A view from another cube.
    ForeignView,
    /// No relation with this id.
    UnknownRel(RelId),
    /// No stored set with this id.
    UnknownSet(SetId),
    /// A column other than 0 or 1.
    NoSuchColumn(u8),
    /// `bind` takes a scan; `join` takes scans or binds.
    NotASide,
    /// The terminal takes a join.
    NotAJoin,
    /// A bound on the join column (not a free column).
    BoundOnJoinColumn,
    /// An id in a relation outside the shared map.
    Domain(MapRefusal),
    /// The map is not readable under the cube's frame.
    Map(MapRead),
    /// The computation refused.
    Native(NativeRefusal),
    /// [`Cube::closure`] holds one `u64` per row, so the map may have at most
    /// 64 positions; carries the map's size.
    ClosureWidth(usize),
    /// The fixpoint did not empty its delta within one round per position,
    /// which a correct semi-naive closure cannot do.
    FixpointBound,
    /// An id could not be placed in a tiled cube's tile list (not held by any
    /// tile, held by two, or a tile unreadable under the frame).
    Locate(LocateRefusal),
    /// The wide closure or refresh refused.
    Wide(WideRefusal),
    /// [`Cube::rebase`] was offered a frame under which this tile's map is not
    /// a pure extension of the one the cube was built on: some position would
    /// renumber, so every lane and logged edge would point at the wrong
    /// symbol. Carries the tile.
    Renumbered(TileId),
    /// The terminal reads one tile's map (`materialize_rows`) and the cube
    /// spans several; use [`Cube::closure_wide`] or read the lanes by
    /// coordinate.
    Tiled,
    /// A count total ([`Cube::join_size`], [`Cube::self_join_size`]) does
    /// not fit in `i128`.
    SumOverflow,
    /// A [`StagedEdges`] batch staged by another cube.
    ForeignBatch,
    /// A batch staged under a frame the cube has since left ([`Cube::commit`]
    /// or [`Cube::rebase`] adopted another one): its coordinates were placed
    /// under a layout that is no longer the cube's.
    StaleBatch,
    /// [`Cube::commit`] was given a frame that does not read the staged map
    /// for the grown tile, or that changes the map of a tile the batch did
    /// not grow. Carries the tile.
    FrameMismatch(TileId),
    /// The reach over the relation's lent plane refused.
    Reach(ReachRefusal),
    /// A [`BaseClosure`] made by another cube.
    ForeignClosure,
    /// A [`BaseClosure`] of another relation.
    ClosureRelation {
        /// The relation asked for.
        expected: RelId,
        /// The relation the closure is of.
        found: RelId,
    },
    /// A [`BaseClosure`] of an older base: [`Cube::refold`] replaced the
    /// relation's base lanes after it was computed.
    StaleBase {
        /// The generation the closure was computed at.
        minted: u64,
        /// The relation's current base generation.
        current: u64,
    },
    /// A [`BaseClosure`] computed under a frame the cube has since left.
    ClosureFrame,
}

/// What the cube has done since it was built.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct CubeStats {
    /// mask-risc programs run.
    pub programs: u64,
    /// Tuples built by [`Cube::materialize_rows`].
    pub rows_materialized: u64,
    /// Relations added by [`Cube::add_relation`].
    pub relations_added: u64,
}

/// `edge⁺` as one row mask per local position: bit `y` of row `x` is set
/// when `y` is reachable from `x` in one or more steps.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Closure {
    rows: Vec<u64>,
}

impl Closure {
    /// One word per local position of the cube's map.
    pub fn rows(&self) -> &[u64] {
        &self.rows
    }

    /// Whether local position `y` is reachable from `x`.
    pub fn contains(&self, x: u8, y: u8) -> bool {
        self.rows
            .get(usize::from(x))
            .is_some_and(|r| y < 64 && r >> y & 1 == 1)
    }
}

/// The work one [`Cube::closure`] did.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ClosureStats {
    /// Composition rounds, including the last, which found nothing new.
    pub rounds: u32,
    /// Rows of `edge` or'ed into a new frontier: one per bit of every delta
    /// composed. In semi-naive evaluation each closure pair enters a delta
    /// exactly once, so this equals `|edge⁺|`.
    pub rows_composed: u64,
}

/// One relation's lanes: both columns as local positions in the shared map.
struct Rel {
    n: usize,
    cols: [Vec<u32>; 2],
    live: Vec<u64>,
}

/// One side of a join, resolved from a view.
#[derive(Clone, Copy)]
struct Side {
    rel: RelId,
    /// `(free column, value)`; `Some(None)`: bound to an unmapped id.
    bound: Option<(u8, Option<u32>)>,
}

/// A side with the column it joins on.
type JoinSide = (Side, u8);

static NEXT_CUBE: AtomicU64 = AtomicU64::new(1);

/// The resident hypercube. See the module docs.
pub struct Cube {
    id: u64,
    frame: Frame,
    tile: TileId,
    /// Every tile the cube spans, in coordinate order: tile `i` owns wide
    /// coordinates `i * 256 .. (i + 1) * 256`. One entry for [`Cube::new`].
    tiles: Vec<TileId>,
    /// Built by [`Cube::new_tiled`]: lanes hold wide coordinates
    /// (`tile_ordinal * 256 + local`) instead of one tile's locals.
    tiled: bool,
    n: usize,
    rels: Vec<Rel>,
    /// One append-only delta log per relation, parallel to `rels`.
    logs: Vec<DeltaLog>,
    /// Per relation, how many times [`Cube::refold`] has folded a non-empty
    /// log into the base lanes, whether or not the lanes changed (an add then
    /// a retire of the same edge folds back to the same base and still
    /// counts: a still-valid closure is refused, never a stale one accepted).
    /// A [`BaseClosure`] records the value it was computed at. Parallel to
    /// `rels`.
    base_gen: Vec<u64>,
    /// Per relation, the base lanes folded with the relation's log, built on
    /// first read when the log is non-empty and dropped whenever the base or
    /// the log changes. A derived read: nothing persistent is written, and it
    /// is recomputed by folding. Parallel to `rels`.
    effective: Vec<OnceCell<Rel>>,
    /// Per relation, the presence of its effective pairs over the cube's
    /// coordinates, in both directions: the cross-reference, never rows. Built
    /// from [`Cube::lanes`] on first read under the cube's frame and dropped
    /// with `effective` (an edge logged, a refold, a commit that grew the
    /// map, a rebase), so it is never read past the state it was built from.
    /// Lent through [`RelationSource`]. Parallel to `rels`.
    planes: Vec<OnceCell<PairPlane>>,
    nodes: Vec<Node>,
    /// One-column sets ([`Cube::add_set`]): one bit per local position of
    /// the shared map, placed under the cube's frame like a relation's lanes.
    /// A set may be shorter than the current map after a commit or rebase
    /// grew it; the missing words read as zero.
    sets: Vec<Vec<u64>>,
    stats: Cell<CubeStats>,
}

impl Cube {
    /// Build the cube: every id of every relation looked up once in the
    /// tile's map, read under `frame`.
    pub fn new(
        table: &MapTable,
        frame: &Frame,
        tile: TileId,
        relations: &[&[(u32, u32)]],
    ) -> Result<Self, CubeRefusal> {
        let map = table.map(frame, tile).map_err(CubeRefusal::Map)?;
        let rels: Vec<Rel> = relations
            .iter()
            .map(|tuples| rel(tuples, map))
            .collect::<Result<_, _>>()?;
        Ok(Self {
            id: NEXT_CUBE.fetch_add(1, Ordering::Relaxed),
            frame: frame.clone(),
            tile,
            tiles: vec![tile],
            tiled: false,
            n: map.len(),
            logs: vec![DeltaLog::new(); rels.len()],
            base_gen: vec![0; rels.len()],
            effective: empty_effective(rels.len()),
            planes: empty_planes(rels.len()),
            rels,
            nodes: Vec::new(),
            sets: Vec::new(),
            stats: Cell::new(CubeStats::default()),
        })
    }

    /// Build a cube over several tiles. Every id is looked up once with
    /// [`MapTable::locate`]; its wide coordinate is `tile_ordinal * 256 +
    /// local`, where the ordinal is the tile's index in `tiles`. The 256-wide
    /// block is used whatever the tile's own width, so a DR64 tile simply
    /// occupies the first 64 positions of its block and coordinates stay
    /// stable if the tile later widens.
    ///
    /// `n` is `256 * tiles.len()`: those positions are reserved, and an
    /// unused one costs nothing in a wide row (an absent block reads as
    /// zero). An id held by no tile is refused as [`CubeRefusal::Locate`],
    /// so a selection that outgrows its tiles is a new frame with more
    /// tiles, never an in-place resize. A tiled cube is read by
    /// [`Cube::closure_wide`]; the single-tile terminals that decode a local
    /// through one map refuse it.
    pub fn new_tiled(
        table: &MapTable,
        frame: &Frame,
        tiles: &[TileId],
        relations: &[&[(u32, u32)]],
    ) -> Result<Self, CubeRefusal> {
        let place = |id: u32| -> Result<u32, CubeRefusal> {
            let (ordinal, local) = table
                .locate(frame, tiles, u64::from(id))
                .map_err(CubeRefusal::Locate)?;
            Ok(ordinal * BLOCK_POSITIONS as u32 + u32::from(local))
        };
        let rels: Vec<Rel> = relations
            .iter()
            .map(|tuples| rel_with(tuples, &place))
            .collect::<Result<_, _>>()?;
        for &t in tiles {
            table.map(frame, t).map_err(CubeRefusal::Map)?;
        }
        Ok(Self {
            id: NEXT_CUBE.fetch_add(1, Ordering::Relaxed),
            frame: frame.clone(),
            tile: tiles.first().copied().unwrap_or(0),
            tiles: tiles.to_vec(),
            tiled: true,
            n: BLOCK_POSITIONS.saturating_mul(tiles.len()),
            logs: vec![DeltaLog::new(); rels.len()],
            base_gen: vec![0; rels.len()],
            effective: empty_effective(rels.len()),
            planes: empty_planes(rels.len()),
            rels,
            nodes: Vec::new(),
            sets: Vec::new(),
            stats: Cell::new(CubeStats::default()),
        })
    }

    /// Add a relation under the cube's frame: a rule's output or a fixpoint
    /// delta. Each id of the new tuples is looked up once in the shared map;
    /// relations already in the cube are not touched. Its ids must be in the
    /// map, so a derived relation over the base relations' ids always fits.
    pub fn add_relation(
        &mut self,
        table: &MapTable,
        tuples: &[(u32, u32)],
    ) -> Result<RelId, CubeRefusal> {
        if self.tiled {
            return Err(CubeRefusal::Tiled);
        }
        let map = self.map(table)?;
        let r = rel(tuples, map)?;
        self.rels.push(r);
        self.logs.push(DeltaLog::new());
        self.base_gen.push(0);
        self.effective.push(OnceCell::new());
        self.planes.push(OnceCell::new());
        let mut s = self.stats.get();
        s.relations_added += 1;
        self.stats.set(s);
        Ok((self.rels.len() - 1) as RelId)
    }

    /// Store a one-column set under the cube's frame (D-XLOG-W5-B, decision
    /// D1). Each id is looked up once in the shared map, as for a relation,
    /// and becomes one bit of a position mask. Its ids must be in the map. On
    /// a tiled cube each id is placed at its wide coordinate, as
    /// [`Cube::new_tiled`] places relation ids (D-XLOG-W5-D), and an id no
    /// tile holds is refused as [`CubeRefusal::Locate`].
    pub fn add_set(&mut self, table: &MapTable, ids: &[u32]) -> Result<SetId, CubeRefusal> {
        let mut mask = vec![0u64; self.n.div_ceil(64)];
        for &id in ids {
            let l = self.coord(table, id)? as usize;
            mask[l / 64] |= 1 << (l % 64);
        }
        self.sets.push(mask);
        Ok((self.sets.len() - 1) as SetId)
    }

    /// The mask of `e` over `words` words. Every leaf must be a stored set.
    fn eval(&self, e: &SetExpr, words: usize) -> Result<Vec<u64>, CubeRefusal> {
        Ok(match e {
            SetExpr::Set(id) => {
                let set = self
                    .sets
                    .get(*id as usize)
                    .ok_or(CubeRefusal::UnknownSet(*id))?;
                let mut m = vec![0u64; words];
                m[..set.len()].copy_from_slice(set);
                m
            }
            SetExpr::And(a, b) | SetExpr::Or(a, b) | SetExpr::AndNot(a, b) => {
                let (a, b) = (self.eval(a, words)?, self.eval(b, words)?);
                let mut out = vec![0u64; words];
                match e {
                    SetExpr::And(..) => mask_and(&a, &b, &mut out),
                    SetExpr::AndNot(..) => mask_andnot(&a, &b, &mut out),
                    _ => {
                        out.copy_from_slice(&a);
                        mask_or_assign(&mut out, &b);
                    }
                }
                out
            }
            SetExpr::Reach { rel, seed, target } => {
                let seed = self.eval(seed, words)?;
                let target = self.eval(target, words)?;
                self.reach_from(*rel, seed, &target, words)?.0
            }
        })
    }

    /// The positions of a set expression: a read terminal, like
    /// [`Cube::support`]. The frame is re-checked. The mask is an answer to
    /// download, never an input: a later terminal is handed the expression
    /// again, not this mask, so no computed mask crosses a terminal. On a
    /// tiled cube the mask is indexed by wide coordinate.
    pub fn set_mask(&self, table: &MapTable, e: &SetExpr) -> Result<Vec<u64>, CubeRefusal> {
        self.check_frame(table)?;
        self.eval(e, self.n.div_ceil(64))
    }

    /// The positions reachable from `seed` along relation `rel`, every step
    /// landing in `target`, seeds included: `seed ∪ reach(seed)` (D-XLOG-W5-B,
    /// decision D3). Semi-naive over position masks: `next` is the or of the
    /// edge rows of the delta's positions, `delta' = (next ∧ target) andnot
    /// full`, `full |= delta'`, until a delta is empty. `seed` and `target`
    /// are evaluated here, from stored sets, so no computed mask crosses a
    /// terminal. The frame is re-checked; a delta still non-empty after one
    /// round per position is [`CubeRefusal::FixpointBound`], as in
    /// [`Cube::closure`]. One-column heads only; n-ary recursion is not here.
    /// A tiled cube's lanes, sets and answer are all in wide coordinates, so
    /// the same loop runs over `256 * tiles` positions (D-XLOG-W5-D).
    pub fn reach(
        &self,
        table: &MapTable,
        rel: RelId,
        seed: &SetExpr,
        target: &SetExpr,
    ) -> Result<(Vec<u64>, ClosureStats), CubeRefusal> {
        self.lanes(rel)?;
        self.check_frame(table)?;
        let words = self.n.div_ceil(64);
        let seed = self.eval(seed, words)?;
        let target = self.eval(target, words)?;
        self.reach_from(rel, seed, &target, words)
    }

    /// The semi-naive loop of [`Cube::reach`] over evaluated masks.
    fn reach_from(
        &self,
        rel: RelId,
        seed: Vec<u64>,
        target: &[u64],
        words: usize,
    ) -> Result<(Vec<u64>, ClosureStats), CubeRefusal> {
        // The relation's rows are the cube's own plane, lent; nothing is
        // rebuilt per call. The plane spans every reserved position, so the
        // cube's masks are widened to it and the answer narrowed back.
        let plane = self.plane(rel)?;
        let w = plane.words_per_row(PairDir::Forward);
        let widen = |m: &[u64]| {
            let mut out = vec![0u64; w];
            out[..m.len()].copy_from_slice(m);
            out
        };
        let (seed, target) = (widen(&seed), widen(target));
        let mut full = vec![0u64; w];
        let mut hop = vec![0u16; plane.rows_in(PairDir::Forward)];
        let mut scratch = vec![0u64; 2 * w];
        let done = relation_reach(
            self,
            rel,
            &self.frame,
            PairDir::Forward,
            &seed,
            Some(&target),
            &mut ReachBuf {
                reached: &mut full,
                hop: &mut hop,
                scratch: &mut scratch,
            },
        )
        .map_err(CubeRefusal::Reach)?;
        full.truncate(words);
        let stats = ClosureStats {
            rounds: done.rounds,
            rows_composed: done.rows_read,
        };
        Ok((full, stats))
    }

    pub fn stats(&self) -> CubeStats {
        self.stats.get()
    }

    fn push(&mut self, node: Node) -> View {
        self.nodes.push(node);
        View {
            cube: self.id,
            node: (self.nodes.len() - 1) as u32,
        }
    }

    fn node(&self, v: View) -> Result<Node, CubeRefusal> {
        if v.cube != self.id {
            return Err(CubeRefusal::ForeignView);
        }
        Ok(self.nodes[v.node as usize])
    }

    /// Every row of a relation.
    pub fn scan(&mut self, rel: RelId) -> Result<View, CubeRefusal> {
        if rel as usize >= self.rels.len() {
            return Err(CubeRefusal::UnknownRel(rel));
        }
        Ok(self.push(Node::Scan(rel)))
    }

    /// The rows of a scan whose column `col` holds `value`. One map lookup
    /// for `value`, under the cube's frame.
    pub fn bind(
        &mut self,
        table: &MapTable,
        scan: View,
        col: u8,
        value: u32,
    ) -> Result<View, CubeRefusal> {
        let Node::Scan(rel) = self.node(scan)? else {
            return Err(CubeRefusal::NotASide);
        };
        check_col(col)?;
        // A value no tile holds binds to nothing; any other refusal is real.
        let global = value;
        let value = match self.coord(table, value) {
            Ok(c) => Some(c),
            Err(CubeRefusal::Domain(_) | CubeRefusal::Locate(LocateRefusal::Unmapped)) => None,
            Err(e) => return Err(e),
        };
        Ok(self.push(Node::Bind {
            rel,
            col,
            value,
            global,
        }))
    }

    /// `left.lcol == right.rcol`, where each side is a scan or a bind.
    pub fn join(
        &mut self,
        left: View,
        lcol: u8,
        right: View,
        rcol: u8,
    ) -> Result<View, CubeRefusal> {
        check_col(lcol)?;
        check_col(rcol)?;
        self.side(left, lcol)?;
        self.side(right, rcol)?;
        Ok(self.push(Node::Join {
            left,
            lcol,
            right,
            rcol,
        }))
    }

    fn side(&self, v: View, join_col: u8) -> Result<Side, CubeRefusal> {
        match self.node(v)? {
            Node::Scan(rel) => Ok(Side { rel, bound: None }),
            Node::Bind {
                rel, col, value, ..
            } if col != join_col => Ok(Side {
                rel,
                bound: Some((col, value)),
            }),
            Node::Bind { .. } => Err(CubeRefusal::BoundOnJoinColumn),
            Node::Join { .. } => Err(CubeRefusal::NotASide),
        }
    }

    fn sides(&self, v: View) -> Result<(JoinSide, JoinSide), CubeRefusal> {
        let Node::Join {
            left,
            lcol,
            right,
            rcol,
        } = self.node(v)?
        else {
            return Err(CubeRefusal::NotAJoin);
        };
        Ok((
            (self.side(left, lcol)?, lcol),
            (self.side(right, rcol)?, rcol),
        ))
    }

    fn map<'t>(&self, table: &'t MapTable) -> Result<TileMap<'t>, CubeRefusal> {
        table.map(&self.frame, self.tile).map_err(CubeRefusal::Map)
    }

    /// Re-checks the cube's frame against every tile it spans. For a
    /// single-tile cube this is exactly [`Cube::map`]'s check.
    fn check_frame(&self, table: &MapTable) -> Result<(), CubeRefusal> {
        if !self.tiled {
            return self.map(table).map(|_| ());
        }
        for &t in &self.tiles {
            table.map(&self.frame, t).map_err(CubeRefusal::Map)?;
        }
        Ok(())
    }

    /// A global id's coordinate in this cube: the local position for a
    /// single-tile cube, the wide coordinate for a tiled one. Counts one map
    /// lookup per tile read.
    fn coord(&self, table: &MapTable, id: u32) -> Result<u32, CubeRefusal> {
        if self.tiled {
            let (ordinal, local) = table
                .locate(&self.frame, &self.tiles, u64::from(id))
                .map_err(CubeRefusal::Locate)?;
            return Ok(ordinal * BLOCK_POSITIONS as u32 + u32::from(local));
        }
        let l = self
            .map(table)?
            .local(u64::from(id))
            .map_err(CubeRefusal::Domain)?;
        Ok(u32::from(l))
    }

    /// Every tile the cube spans, read under `frame`, in coordinate order.
    fn views_under<'t>(
        &self,
        table: &'t MapTable,
        frame: &Frame,
    ) -> Result<Vec<TileMap<'t>>, CubeRefusal> {
        self.tiles
            .iter()
            .map(|&t| table.map(frame, t).map_err(CubeRefusal::Map))
            .collect()
    }

    /// [`Cube::coord`] over explicit tile views (in `self.tiles` order)
    /// instead of the table under the cube's frame: the same placement and
    /// the same refusals, so a staged layout places ids exactly as the
    /// committed one will.
    fn coord_in(&self, views: &[TileMap<'_>], id: u32) -> Result<u32, CubeRefusal> {
        let g = u64::from(id);
        if self.tiled {
            let mut found = None;
            for (ordinal, v) in views.iter().enumerate() {
                if let Ok(local) = v.local(g) {
                    if found.is_some() {
                        return Err(CubeRefusal::Locate(LocateRefusal::Ambiguous));
                    }
                    let ordinal = u32::try_from(ordinal)
                        .map_err(|_| CubeRefusal::Locate(LocateRefusal::Ambiguous))?;
                    found = Some(ordinal * BLOCK_POSITIONS as u32 + u32::from(local));
                }
            }
            return found.ok_or(CubeRefusal::Locate(LocateRefusal::Unmapped));
        }
        let view = views
            .first()
            .ok_or(CubeRefusal::Map(MapRead::NoSuchTile(self.tile)))?;
        view.local(g).map(u32::from).map_err(CubeRefusal::Domain)
    }

    /// The binds that were unmapped and are placed under `views`: `(node,
    /// coordinate)`. A bind that was mapped keeps its coordinate, because a
    /// layout is only adopted when it extends the old one (no renumbering).
    fn rebind(&self, views: &[TileMap<'_>]) -> Result<Vec<(usize, u32)>, CubeRefusal> {
        let mut out = Vec::new();
        for (i, node) in self.nodes.iter().enumerate() {
            if let Node::Bind {
                value: None,
                global,
                ..
            } = *node
            {
                match self.coord_in(views, global) {
                    Ok(c) => out.push((i, c)),
                    Err(CubeRefusal::Domain(_) | CubeRefusal::Locate(LocateRefusal::Unmapped)) => {}
                    Err(e) => return Err(e),
                }
            }
        }
        Ok(out)
    }

    fn apply_rebind(&mut self, rebind: &[(usize, u32)]) {
        for &(i, c) in rebind {
            if let Some(Node::Bind { value, .. }) = self.nodes.get_mut(i) {
                *value = Some(c);
            }
        }
    }

    /// The lanes every read goes through: relation `rel`'s base folded with
    /// its delta log ("sparse write, merge on read"). With an empty log these
    /// are the base lanes themselves, no copy. Otherwise they are the cached
    /// fold, built on first use. A read never modifies the base or the log;
    /// only [`Cube::refold`] replaces the base.
    fn lanes(&self, rel: RelId) -> Result<&Rel, CubeRefusal> {
        let i = rel as usize;
        let base = self.rels.get(i).ok_or(CubeRefusal::UnknownRel(rel))?;
        let log = self.log(rel)?;
        if log.is_empty() {
            return Ok(base);
        }
        let cell = self.effective.get(i).ok_or(CubeRefusal::UnknownRel(rel))?;
        Ok(cell.get_or_init(|| {
            let (src, dst) = wide::refold(&base.cols[0], &base.cols[1], log);
            Rel {
                n: src.len(),
                live: all_live(src.len()),
                cols: [src, dst],
            }
        }))
    }

    /// Drops relation `rel`'s cached effective lanes.
    fn invalidate(&mut self, rel: RelId) {
        if let Some(c) = self.effective.get_mut(rel as usize) {
            c.take();
        }
        if let Some(c) = self.planes.get_mut(rel as usize) {
            c.take();
        }
    }

    /// Relation `rel`'s pair plane: one bit per effective pair, both
    /// directions, over `256 * tiles` coordinates per axis. Built once per
    /// state of the relation; see the `planes` field.
    pub(crate) fn plane(&self, rel: RelId) -> Result<&PairPlane, CubeRefusal> {
        let lanes = self.lanes(rel)?;
        let cell = self
            .planes
            .get(rel as usize)
            .ok_or(CubeRefusal::UnknownRel(rel))?;
        Ok(cell.get_or_init(|| {
            let axis = Axis::new(Width::W256, self.tiles.len() as u32);
            let mut plane = PairPlane::new(&self.frame, axis, axis);
            for (&x, &y) in lanes.cols[0].iter().zip(&lanes.cols[1]) {
                // Lanes hold coordinates placed under this frame, all below
                // `256 * tiles`; a refusal here would be a cube bug.
                let placed = plane.insert(x as usize, y as usize);
                debug_assert!(placed.is_ok(), "lane coordinate past the cube's axis");
            }
            plane
        }))
    }

    fn run(&self, side: Side, query: &Query, out: Out<'_>) -> Result<(), CubeRefusal> {
        let r = self.lanes(side.rel)?;
        let lanes = [LaneRef::U32(&r.cols[0]), LaneRef::U32(&r.cols[1])];
        let mut s = self.stats.get();
        s.programs += 1;
        self.stats.set(s);
        run_lanes(r.n, &r.live, &lanes, query, out).map_err(CubeRefusal::Native)
    }

    /// The filter of a side, or `None` when it is bound to an unmapped id and
    /// no row can match (nothing runs).
    fn filter(side: Side) -> Option<Filter> {
        match side.bound {
            None => Some(Filter::plane(Mask(0))),
            Some((_, None)) => None,
            Some((col, Some(v))) => Some(Filter::cmp(Col(u16::from(col)), Cmp::EqU32(v))),
        }
    }

    fn reached(&self, side: Side, join_col: u8) -> Result<Vec<u64>, CubeRefusal> {
        let mut words = vec![0u64; self.n.div_ceil(64)];
        // An empty domain has nothing to reach; mask-risc refuses an empty sink.
        if self.n == 0 {
            return Ok(words);
        }
        if let Some(filter) = Self::filter(side) {
            let q = Query {
                filter,
                agg: Agg::ScatterOrU32 {
                    fk: Col(u16::from(join_col)),
                    out_rows: self.n as u32,
                },
            };
            self.run(side, &q, Out::Mask(&mut words))?;
        }
        Ok(words)
    }

    fn degrees(&self, side: Side, join_col: u8) -> Result<Vec<i64>, CubeRefusal> {
        let mut counts = vec![0i64; self.n];
        // An empty domain has nothing to count; mask-risc refuses an empty sink.
        if self.n == 0 {
            return Ok(counts);
        }
        if let Some(filter) = Self::filter(side) {
            let q = Query {
                filter,
                agg: Agg::GroupReduce {
                    key: GroupAddr::Local(Col(u16::from(join_col))),
                    agg: GroupAgg::Count,
                },
            };
            self.run(side, &q, Out::I64(&mut counts))?;
        }
        Ok(counts)
    }

    /// The join values with a partner on both sides: one bit per local
    /// position of the shared map.
    pub fn support(&self, table: &MapTable, join: View) -> Result<Vec<u64>, CubeRefusal> {
        let ((l, lc), (r, rc)) = self.sides(join)?;
        self.check_frame(table)?;
        let a = self.reached(l, lc)?;
        let b = self.reached(r, rc)?;
        let mut out = vec![0u64; a.len()];
        mask_and(&a, &b, &mut out);
        Ok(out)
    }

    /// Derivations per join value: one per pair of matching rows, as XLOG's
    /// `count` counts body bindings. A count past `i64` is refused.
    pub fn bag_count(&self, table: &MapTable, join: View) -> Result<Vec<i64>, CubeRefusal> {
        let ((l, lc), (r, rc)) = self.sides(join)?;
        self.check_frame(table)?;
        let a = self.degrees(l, lc)?;
        let b = self.degrees(r, rc)?;
        let mut out = vec![0i64; self.n];
        checked_binary_i64(I64Op::Mul, &a, &b, &mut out)
            .map_err(|e| CubeRefusal::Native(NativeRefusal::Count(e)))?;
        Ok(out)
    }

    /// Rows per coordinate of column `col` of one side: how often each
    /// position of the shared map occurs there. `side` is a scan, or a bind
    /// on the other column. One `GroupReduce Count` program; no row is built.
    ///
    /// Every relation and both columns are counted over the SAME positions,
    /// so two count vectors of this cube line up position by position: their
    /// product is [`Cube::bag_count`] and its total is [`Cube::join_size`].
    /// A position is a coordinate under the cube's frame, not a value: the
    /// order of positions carries no meaning.
    pub fn column_counts(
        &self,
        table: &MapTable,
        side: View,
        col: u8,
    ) -> Result<Vec<i64>, CubeRefusal> {
        check_col(col)?;
        let side = self.side(side, col)?;
        self.check_frame(table)?;
        self.degrees(side, col)
    }

    /// Total derivations of a join: the sum of [`Cube::bag_count`] over every
    /// position, i.e. `Σ_p left(p) · right(p)` of the two sides' column
    /// counts. This is the number of tuples [`Cube::materialize_rows`] would
    /// build, obtained without building one.
    ///
    /// Each product and the total are taken in `i128`, checked, so a position
    /// whose product passes `i64` (which [`Cube::bag_count`] refuses) is still
    /// answered. A total past `i128` is [`CubeRefusal::SumOverflow`].
    pub fn join_size(&self, table: &MapTable, join: View) -> Result<i128, CubeRefusal> {
        let ((l, lc), (r, rc)) = self.sides(join)?;
        self.check_frame(table)?;
        let a = self.degrees(l, lc)?;
        let b = self.degrees(r, rc)?;
        dot_counts(&a, &b).ok_or(CubeRefusal::SumOverflow)
    }

    /// Total derivations of one side joined with itself on `col`: `Σ_p c(p)²`
    /// over [`Cube::column_counts`]. Equal to [`Cube::join_size`] of that
    /// self-join, from one count program instead of two. Taken in `i128`,
    /// checked, as in [`Cube::join_size`].
    pub fn self_join_size(
        &self,
        table: &MapTable,
        side: View,
        col: u8,
    ) -> Result<i128, CubeRefusal> {
        let c = self.column_counts(table, side, col)?;
        dot_counts(&c, &c).ok_or(CubeRefusal::SumOverflow)
    }

    /// The coordinate with the most rows in column `col` of one side, and that
    /// count. Equal counts go to the lower coordinate. `None` when the side
    /// has no row.
    ///
    /// The coordinate is a position under the cube's frame, not an id:
    /// decode it through the map to name the key.
    pub fn heaviest(
        &self,
        table: &MapTable,
        side: View,
        col: u8,
    ) -> Result<Option<(u32, i64)>, CubeRefusal> {
        let counts = self.column_counts(table, side, col)?;
        let mut best: Option<(u32, i64)> = None;
        for (at, &c) in counts.iter().enumerate() {
            if c > best.map_or(0, |(_, b)| b) {
                best = Some((at as u32, c));
            }
        }
        Ok(best)
    }

    /// `edge⁺` of relation `rel`, semi-naively: `delta' = (delta ∘ edge) andnot
    /// full`, `full |= delta'`, until a delta is empty. Rows come from the
    /// relation's lanes, which already hold local positions, so nothing is
    /// looked up; the difference is a mask word, not a host-side set.
    pub fn closure(
        &self,
        table: &MapTable,
        rel: RelId,
    ) -> Result<(Closure, ClosureStats), CubeRefusal> {
        let r = self.lanes(rel)?;
        self.check_frame(table)?;
        if self.n > 64 {
            return Err(CubeRefusal::ClosureWidth(self.n));
        }
        let mut edge = vec![0u64; self.n];
        for (&a, &b) in r.cols[0].iter().zip(&r.cols[1]) {
            edge[a as usize] |= 1 << b;
        }
        let mut full = edge.clone();
        let mut delta = edge.clone();
        let mut next = vec![0u64; self.n];
        let mut stats = ClosureStats::default();
        // A path visits each position once, so the delta is empty by round n+1.
        while mask_any(&delta) {
            if stats.rounds as usize > self.n {
                return Err(CubeRefusal::FixpointBound);
            }
            stats.rounds += 1;
            for (x, &d) in delta.iter().enumerate() {
                let mut acc = 0u64;
                let mut bits = d;
                while bits != 0 {
                    acc |= edge[bits.trailing_zeros() as usize];
                    bits &= bits - 1;
                    stats.rows_composed += 1;
                }
                next[x] = acc;
            }
            mask_andnot(&next, &full, &mut delta);
            mask_or_assign(&mut full, &delta);
        }
        Ok((Closure { rows: full }, stats))
    }

    /// `edge⁺` of relation `rel` over wide coordinates, with no limit on the
    /// number of positions beyond [`wide::MAX_POSITIONS`]. Rows keep block
    /// presence, so a position range nothing reaches costs nothing.
    ///
    /// The edges used are the relation's *effective* edges: its base lanes
    /// folded with the relation's delta log ([`Cube::add_edge`],
    /// [`Cube::retire_edge`]), read through the same accessor as every other
    /// read, so the narrow [`Cube::closure`] and the terminals agree with it.
    /// The base and the log are not changed by a read; replacing the base is
    /// the explicit [`Cube::refold`].
    pub fn closure_wide(
        &self,
        table: &MapTable,
        rel: RelId,
    ) -> Result<(WideRows, WideStats), CubeRefusal> {
        let r = self.lanes(rel)?;
        self.check_frame(table)?;
        let n = u32::try_from(self.n).unwrap_or(u32::MAX);
        wide::closure_wide(n, &r.cols[0], &r.cols[1]).map_err(CubeRefusal::Wide)
    }

    fn log(&self, rel: RelId) -> Result<&DeltaLog, CubeRefusal> {
        self.logs
            .get(rel as usize)
            .ok_or(CubeRefusal::UnknownRel(rel))
    }

    /// Logs an ADD of edge `src -> dst` (global ids) on relation `rel`. The
    /// ids are placed exactly as at build, so an id outside the cube's map is
    /// refused: a new symbol is a rebase to a new frame ([`Cube::rebase`]),
    /// never an in-place resize, because positions must not renumber under
    /// edges already stored. The resident lanes are not touched; the log is
    /// merged on read.
    pub fn add_edge(
        &mut self,
        table: &MapTable,
        rel: RelId,
        src: u32,
        dst: u32,
    ) -> Result<(), CubeRefusal> {
        let (s, d) = self.log_coords(table, rel, src, dst)?;
        if let Some(l) = self.logs.get_mut(rel as usize) {
            l.add(s, d);
        }
        self.invalidate(rel);
        Ok(())
    }

    /// Logs a RETIRE of edge `src -> dst` (global ids) on relation `rel`; see
    /// [`Cube::add_edge`] for the placement rule. Retiring an edge that is
    /// not present is a no-op when folded. A closure refresh recomputes every
    /// row that could reach the retired source, because a closure row cannot
    /// be patched for a deletion: the removed edge may or may not have been
    /// the only path to a target.
    pub fn retire_edge(
        &mut self,
        table: &MapTable,
        rel: RelId,
        src: u32,
        dst: u32,
    ) -> Result<(), CubeRefusal> {
        let (s, d) = self.log_coords(table, rel, src, dst)?;
        if let Some(l) = self.logs.get_mut(rel as usize) {
            l.retire(s, d);
        }
        self.invalidate(rel);
        Ok(())
    }

    fn log_coords(
        &self,
        table: &MapTable,
        rel: RelId,
        src: u32,
        dst: u32,
    ) -> Result<(u32, u32), CubeRefusal> {
        self.log(rel)?;
        Ok((self.coord(table, src)?, self.coord(table, dst)?))
    }

    /// Relation `rel`'s delta log (read-only), `None` for an unknown relation.
    pub fn delta_log(&self, rel: RelId) -> Option<&DeltaLog> {
        self.logs.get(rel as usize)
    }

    /// The frame the cube reads its tile maps under.
    pub fn frame(&self) -> &Frame {
        &self.frame
    }

    /// Logged operations on relation `rel` not yet folded into its base.
    /// An unknown relation has none.
    pub fn delta_len(&self, rel: RelId) -> usize {
        self.logs.get(rel as usize).map_or(0, DeltaLog::len)
    }

    /// Relation `rel`'s base generation: how many times [`Cube::refold`] has
    /// replaced its base lanes with different ones. `None` for an unknown
    /// relation.
    pub fn base_generation(&self, rel: RelId) -> Option<u64> {
        self.base_gen.get(rel as usize).copied()
    }

    /// `edge⁺` of relation `rel`'s BASE lanes, ignoring its delta log, with
    /// the provenance [`Cube::refresh_closure`] and [`Cube::stage_closure`]
    /// check. With an empty log it equals [`Cube::closure_wide`]; with a
    /// non-empty one it does not, and pending edges are never certified as
    /// base.
    pub fn base_closure(&self, table: &MapTable, rel: RelId) -> Result<BaseClosure, CubeRefusal> {
        let r = self
            .rels
            .get(rel as usize)
            .ok_or(CubeRefusal::UnknownRel(rel))?;
        let generation = self
            .base_generation(rel)
            .ok_or(CubeRefusal::UnknownRel(rel))?;
        self.check_frame(table)?;
        let n = u32::try_from(self.n).unwrap_or(u32::MAX);
        let closure = wide::closure_wide(n, &r.cols[0], &r.cols[1]).map_err(CubeRefusal::Wide)?;
        Ok(BaseClosure {
            cube: self.id,
            rel,
            generation,
            frame: self.frame.clone(),
            closure,
        })
    }

    /// Refuses `old` unless it is this cube's closure of relation `rel`'s
    /// current base, computed under the cube's current frame.
    fn check_base(&self, rel: RelId, old: &BaseClosure) -> Result<(), CubeRefusal> {
        if old.cube != self.id {
            return Err(CubeRefusal::ForeignClosure);
        }
        if old.rel != rel {
            return Err(CubeRefusal::ClosureRelation {
                expected: rel,
                found: old.rel,
            });
        }
        let current = self
            .base_generation(rel)
            .ok_or(CubeRefusal::UnknownRel(rel))?;
        if old.generation != current {
            return Err(CubeRefusal::StaleBase {
                minted: old.generation,
                current,
            });
        }
        if !old.frame.same_issue(&self.frame) {
            return Err(CubeRefusal::ClosureFrame);
        }
        Ok(())
    }

    /// Brings `old` up to date with relation `rel`'s delta log without a full
    /// rebuild ([`wide::refresh`]). `old` must be this cube's
    /// [`BaseClosure`] of `rel` at the current base generation and frame;
    /// otherwise the refresh is refused before anything is computed
    /// ([`CubeRefusal::ForeignClosure`], [`CubeRefusal::ClosureRelation`],
    /// [`CubeRefusal::StaleBase`], [`CubeRefusal::ClosureFrame`]).
    pub fn refresh_closure(
        &self,
        table: &MapTable,
        rel: RelId,
        old: &BaseClosure,
    ) -> Result<(WideRows, WideStats), CubeRefusal> {
        let r = self
            .rels
            .get(rel as usize)
            .ok_or(CubeRefusal::UnknownRel(rel))?;
        self.check_base(rel, old)?;
        self.check_frame(table)?;
        let n = u32::try_from(self.n).unwrap_or(u32::MAX);
        wide::refresh(old.rows(), &r.cols[0], &r.cols[1], self.log(rel)?, n)
            .map_err(CubeRefusal::Wide)
    }

    /// Replaces relation `rel`'s base lanes with the base folded with its
    /// log, and empties the log. Explicit only: nothing in the cube calls
    /// this on a threshold, so the caller decides when a long log is worth
    /// the rewrite. With a non-empty log the base generation advances, so a
    /// [`BaseClosure`] of the old base is refused afterwards
    /// ([`CubeRefusal::StaleBase`]). With an empty log the base is unchanged
    /// and so is its generation.
    pub fn refold(&mut self, rel: RelId) -> Result<(), CubeRefusal> {
        let log = self.log(rel)?;
        if log.is_empty() {
            return Ok(());
        }
        let r = self
            .rels
            .get(rel as usize)
            .ok_or(CubeRefusal::UnknownRel(rel))?;
        let (src, dst) = wide::refold(&r.cols[0], &r.cols[1], log);
        let n = src.len();
        if let Some(r) = self.rels.get_mut(rel as usize) {
            r.n = n;
            r.live = all_live(n);
            r.cols = [src, dst];
        }
        if let Some(l) = self.logs.get_mut(rel as usize) {
            *l = DeltaLog::new();
        }
        if let Some(g) = self.base_gen.get_mut(rel as usize) {
            *g += 1;
        }
        self.invalidate(rel);
        Ok(())
    }

    /// Checks a batch of edge operations, optionally under one grown tile map,
    /// against the cube **without changing it or the table**: the grown map
    /// must extend the tile's current map (as [`Cube::rebase`] requires), and
    /// every id must be placed under the resulting layout (as
    /// [`Cube::add_edge`] requires). The map is not handed to the table, so
    /// no version is issued and no frame is sealed. The returned batch is
    /// applied by [`Cube::commit`], which re-checks that the cube is still
    /// under the frame the batch was staged under and that the frame it is
    /// given reads exactly the staged map.
    pub fn stage(
        &self,
        table: &MapTable,
        grown: Option<(TileId, &LocalMap)>,
        ops: &[(RelId, EdgeOp, u32, u32)],
    ) -> Result<StagedEdges, CubeRefusal> {
        let mut views = self.views_under(table, &self.frame)?;
        if let Some((t, map)) = grown {
            let i = self
                .tiles
                .iter()
                .position(|&x| x == t)
                .ok_or(CubeRefusal::Map(MapRead::NoSuchTile(t)))?;
            let candidate = map.view();
            if !candidate.extends(views[i]) {
                return Err(CubeRefusal::Renumbered(t));
            }
            views[i] = candidate;
        }
        let n = if self.tiled {
            self.n
        } else {
            views.first().map_or(self.n, TileMap::len)
        };
        let mut placed = Vec::with_capacity(ops.len());
        for &(rel, op, src, dst) in ops {
            self.log(rel)?;
            placed.push((
                rel,
                op,
                self.coord_in(&views, src)?,
                self.coord_in(&views, dst)?,
            ));
        }
        let rebind = if grown.is_some() {
            self.rebind(&views)?
        } else {
            Vec::new()
        };
        Ok(StagedEdges {
            cube: self.id,
            staged_under: self.frame.clone(),
            grown: grown.map(|(t, m)| {
                let v = m.view();
                StagedMap {
                    tile: t,
                    width: v.width(),
                    globals: v.globals().to_vec(),
                }
            }),
            n,
            ops: placed,
            rebind,
        })
    }

    /// The closure of relation `rel` as it will be once `staged` is
    /// committed: [`Cube::refresh_closure`] over the relation's log followed
    /// by the staged operations, at the staged size. Changes nothing, so a
    /// caller can compute it before committing. `old` is checked exactly as
    /// in [`Cube::refresh_closure`], and `staged` must be this cube's batch,
    /// staged under its current frame ([`CubeRefusal::ForeignBatch`],
    /// [`CubeRefusal::StaleBatch`]).
    pub fn stage_closure(
        &self,
        rel: RelId,
        old: &BaseClosure,
        staged: &StagedEdges,
    ) -> Result<(WideRows, WideStats), CubeRefusal> {
        let r = self
            .rels
            .get(rel as usize)
            .ok_or(CubeRefusal::UnknownRel(rel))?;
        self.check_base(rel, old)?;
        self.check_batch(staged)?;
        let mut log = self.log(rel)?.clone();
        for &(r2, op, s, d) in &staged.ops {
            if r2 == rel {
                match op {
                    EdgeOp::Add => log.add(s, d),
                    EdgeOp::Retire => log.retire(s, d),
                }
            }
        }
        let n = u32::try_from(staged.n).unwrap_or(u32::MAX);
        wide::refresh(old.rows(), &r.cols[0], &r.cols[1], &log, n).map_err(CubeRefusal::Wide)
    }

    /// Refuses a batch staged by another cube or under a frame the cube has
    /// since left.
    fn check_batch(&self, staged: &StagedEdges) -> Result<(), CubeRefusal> {
        if staged.cube != self.id {
            return Err(CubeRefusal::ForeignBatch);
        }
        if !staged.staged_under.same_issue(&self.frame) {
            return Err(CubeRefusal::StaleBatch);
        }
        Ok(())
    }

    /// The checks [`Cube::commit`] makes, for a caller that must know the
    /// outcome before it advances the [`MapTable`] the commit frame is sealed
    /// from. `grown` is the map the caller will install for the grown tile
    /// (`None` when nothing grows). If this returns `Ok`, the table installs
    /// exactly `grown` and no other tile changes, then `commit` under the
    /// sealed frame accepts. Changes nothing.
    pub fn check_commit(
        &self,
        staged: &StagedEdges,
        grown: Option<(TileId, &LocalMap)>,
    ) -> Result<(), CubeRefusal> {
        self.check_batch(staged)?;
        match (&staged.grown, grown) {
            (None, None) => Ok(()),
            (Some(g), Some((t, m))) if g.tile == t && g.is(m.view()) => {
                if self.tiles.contains(&t) {
                    Ok(())
                } else {
                    Err(CubeRefusal::FrameMismatch(t))
                }
            }
            (Some(g), _) => Err(CubeRefusal::FrameMismatch(g.tile)),
            (None, Some((t, _))) => Err(CubeRefusal::FrameMismatch(t)),
        }
    }

    /// Applies a batch [`Cube::stage`] accepted. Every check runs before
    /// anything changes, so a refusal leaves the cube exactly as it was:
    ///
    /// - the batch is this cube's ([`CubeRefusal::ForeignBatch`]) and was
    ///   staged under the cube's current frame ([`CubeRefusal::StaleBatch`]);
    /// - when a tile grew, `frame` reads exactly the staged map for that tile
    ///   (same width, same ids at the same positions, not only the same
    ///   length) and the cube's current map for every other tile;
    ///   [`CubeRefusal::FrameMismatch`] names the first tile that differs;
    /// - when nothing grew, `frame` reads the cube's current map for every
    ///   tile (the cube keeps its own frame).
    ///
    /// Then the operations are logged in the order given, the size and any
    /// newly placeable binds adopted, and the affected cached folds dropped.
    pub fn commit(
        &mut self,
        table: &MapTable,
        frame: &Frame,
        staged: StagedEdges,
    ) -> Result<(), CubeRefusal> {
        self.check_batch(&staged)?;
        let current = self.views_under(table, &self.frame)?;
        let offered = self.views_under(table, frame)?;
        for ((&t, cur), off) in self.tiles.iter().zip(&current).zip(&offered) {
            let ok = match &staged.grown {
                Some(g) if g.tile == t => g.is(*off),
                _ => off.width() == cur.width() && off.globals() == cur.globals(),
            };
            if !ok {
                return Err(CubeRefusal::FrameMismatch(t));
            }
        }
        if let Some(g) = &staged.grown {
            if !self.tiles.contains(&g.tile) {
                return Err(CubeRefusal::FrameMismatch(g.tile));
            }
        }
        // ---- no refusal below ----
        if staged.grown.is_some() {
            self.n = staged.n;
            self.apply_rebind(&staged.rebind);
            self.frame = frame.clone();
            for c in &mut self.effective {
                c.take();
            }
            for c in &mut self.planes {
                c.take();
            }
        }
        for &(rel, op, s, d) in &staged.ops {
            if let Some(l) = self.logs.get_mut(rel as usize) {
                match op {
                    EdgeOp::Add => l.add(s, d),
                    EdgeOp::Retire => l.retire(s, d),
                }
            }
            self.invalidate(rel);
        }
        Ok(())
    }

    /// Adopts `frame`, sealed after the table advanced to add symbols. Allowed
    /// only if, for every tile the cube holds, the new map is a pure
    /// extension of the one the cube was built (or last rebased) on
    /// ([`TileMap::extends`]): positions never renumber, so every stored
    /// coordinate and logged edge keeps naming the same symbol. Otherwise
    /// [`CubeRefusal::Renumbered`] names the tile and the cube is unchanged.
    ///
    /// A single-tile cube's `n` is its map's length, so it grows with the
    /// map; a tiled cube's positions were reserved at build and stay fixed.
    pub fn rebase(&mut self, table: &MapTable, frame: &Frame) -> Result<(), CubeRefusal> {
        // The table is append-only: the cube's own frame still reads the
        // layout it was built on, so the old side of the comparison is read
        // through the index and the cube keeps no copy of it.
        // Everything that can refuse is checked before anything changes.
        let old = self.views_under(table, &self.frame)?;
        let new = self.views_under(table, frame)?;
        for ((&t, o), m) in self.tiles.iter().zip(&old).zip(&new) {
            if !m.extends(*o) {
                return Err(CubeRefusal::Renumbered(t));
            }
        }
        let rebind = self.rebind(&new)?;
        if !self.tiled {
            if let Some(m) = new.first() {
                self.n = m.len();
            }
        }
        self.apply_rebind(&rebind);
        self.frame = frame.clone();
        // `n` may have grown; cached folds were built under the old one.
        for c in &mut self.effective {
            c.take();
        }
        for c in &mut self.planes {
            c.take();
        }
        Ok(())
    }

    /// The joined tuples `(left free, join value, right free)` as global ids:
    /// the one place this module builds rows, for a caller that downloads the
    /// relation. Restricted to the join's support first, so rows with no
    /// partner are never visited twice.
    pub fn materialize_rows(
        &self,
        table: &MapTable,
        join: View,
    ) -> Result<Vec<(u64, u64, u64)>, CubeRefusal> {
        let ((l, lc), (r, rc)) = self.sides(join)?;
        if self.tiled {
            return Err(CubeRefusal::Tiled);
        }
        let map = self.map(table)?;
        let keep = &self.support(table, join)?;
        let (rel_l, rel_r) = (self.lanes(l.rel)?, self.lanes(r.rel)?);
        let rows = side_rows;
        let mut out = Vec::new();
        for (jl, fl) in rows(keep, l, rel_l, lc) {
            for (jr, fr) in rows(keep, r, rel_r, rc) {
                if jl == jr {
                    let g = |x: u32| map.global(x as u8).expect("local position from intake");
                    out.push((g(fl), g(jl), g(fr)));
                }
            }
        }
        let mut s = self.stats.get();
        s.rows_materialized += out.len() as u64;
        self.stats.set(s);
        Ok(out)
    }
}

/// The `(join value, free value)` pairs of one side's effective lanes whose
/// join value is in `keep` and which satisfy the side's bound.
fn side_rows<'a>(
    keep: &'a [u64],
    side: Side,
    rel: &'a Rel,
    jc: u8,
) -> impl Iterator<Item = (u32, u32)> + 'a {
    let fc = 1 - jc as usize;
    let (j, f) = (&rel.cols[jc as usize], &rel.cols[fc]);
    let bound = side.bound.map(|(_, v)| v);
    (0..j.len())
        .filter(move |&i| keep[j[i] as usize / 64] >> (j[i] % 64) & 1 == 1)
        .filter(move |&i| bound.is_none_or(|v| v == Some(f[i])))
        .map(move |i| (j[i], f[i]))
}

fn empty_planes(n: usize) -> Vec<OnceCell<PairPlane>> {
    (0..n).map(|_| OnceCell::new()).collect()
}

/// The cube lends each relation's pair plane; it never copies or rebuilds one
/// for a reader.
impl RelationSource for Cube {
    fn borrow_relation(&self, rel: RelNo) -> Option<&PairPlane> {
        self.plane(rel).ok()
    }
}

fn empty_effective(n: usize) -> Vec<OnceCell<Rel>> {
    (0..n).map(|_| OnceCell::new()).collect()
}

fn check_col(col: u8) -> Result<(), CubeRefusal> {
    if col > 1 {
        Err(CubeRefusal::NoSuchColumn(col))
    } else {
        Ok(())
    }
}

fn rel(tuples: &[(u32, u32)], map: TileMap<'_>) -> Result<Rel, CubeRefusal> {
    rel_with(tuples, &|id| {
        let l = map.local(u64::from(id)).map_err(CubeRefusal::Domain)?;
        Ok(u32::from(l))
    })
}

/// Builds a relation's lanes, placing every id with `place`.
fn rel_with(
    tuples: &[(u32, u32)],
    place: &dyn Fn(u32) -> Result<u32, CubeRefusal>,
) -> Result<Rel, CubeRefusal> {
    let mut cols = [
        Vec::with_capacity(tuples.len()),
        Vec::with_capacity(tuples.len()),
    ];
    for &(a, b) in tuples {
        for (c, id) in [(0, a), (1, b)] {
            cols[c].push(place(id)?);
        }
    }
    Ok(Rel {
        n: tuples.len(),
        live: all_live(tuples.len()),
        cols,
    })
}

/// `Σ a[p] · b[p]` with every product and the running total in `i128`,
/// checked. Each product of two `i64` counts fits in `i128`; only the total
/// can overflow, and then the answer is `None`, never a wrapped value.
/// Both vectors are count vectors over the same positions, so their lengths
/// are equal; a mismatch is a broken invariant, not a short answer, so it
/// panics in every build instead of letting `zip` drop the tail.
fn dot_counts(a: &[i64], b: &[i64]) -> Option<i128> {
    assert_eq!(a.len(), b.len(), "count vectors over one map");
    a.iter().zip(b).try_fold(0i128, |acc, (&x, &y)| {
        acc.checked_add(i128::from(x).checked_mul(i128::from(y))?)
    })
}

#[cfg(test)]
mod dot_counts_tests {
    use super::dot_counts;

    /// A per-position product past `i64` is still answered: this is the
    /// case `bag_count`'s checked `i64` multiply refuses.
    #[test]
    fn a_product_past_i64_is_answered_exactly() {
        let big = i64::MAX;
        assert!(
            big.checked_mul(2).is_none(),
            "the i64 product must overflow"
        );
        assert_eq!(
            dot_counts(&[big, 3], &[2, 5]),
            Some(i128::from(big) * 2 + 15)
        );
    }

    /// A total past `i128` is refused, not wrapped.
    #[test]
    fn a_total_past_i128_is_refused() {
        let big = [i64::MAX; 3];
        // each product is about 2^126; three of them pass 2^127.
        assert_eq!(dot_counts(&big, &big), None);
        assert_eq!(
            dot_counts(&big[..1], &big[..1]),
            Some(i128::from(i64::MAX) * i128::from(i64::MAX))
        );
    }

    #[test]
    fn empty_vectors_sum_to_zero() {
        assert_eq!(dot_counts(&[], &[]), Some(0));
    }

    #[test]
    #[should_panic(expected = "count vectors over one map")]
    fn mismatched_lengths_panic_rather_than_truncate() {
        let _ = dot_counts(&[1, 2], &[3]);
    }
}
