//! The resident hypercube: one execution context for every relation, and
//! views into it (plan `d-xlog-native-v1.md` Amendment F).
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

use std::cell::Cell;
use std::sync::atomic::{AtomicU64, Ordering};

use lance_graph_dt64::dr::{Frame, LocalMap, LocateRefusal, MapRead, MapRefusal, MapTable, TileId};
use lance_graph_mask_risc::{LaneRef, Out};
use lance_graph_quack::{Agg, Cmp, Col, Filter, GroupAddr, GroupAgg, Mask, Query};
use ndarray::simd::{checked_binary_i64, mask_and, mask_andnot, mask_any, mask_or_assign, I64Op};

use crate::wide::{self, DeltaLog, WideRefusal, WideRows, WideStats, BLOCK_POSITIONS};
use crate::{all_live, run_lanes, NativeRefusal};

/// A relation's index in the order given to [`Cube::new`].
pub type RelId = u32;

/// A handle into a [`Cube`]. Copying it copies no rows.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct View {
    cube: u64,
    node: u32,
}

#[derive(Clone, Copy, Debug)]
enum Node {
    Scan(RelId),
    /// `col == value`. `None`: the value is not in the map, so no row
    /// matches.
    Bind {
        rel: RelId,
        col: u8,
        value: Option<u32>,
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
    /// The maps as they were when the cube was built or last rebased, one per
    /// entry of `tiles`. A [`MapTable`] drops a map when it advances, so the
    /// old assignment is gone by the time [`Cube::rebase`] must compare it;
    /// the cube keeps its own copy of the ids (a few bytes per position).
    snapshots: Vec<LocalMap>,
    n: usize,
    rels: Vec<Rel>,
    /// One append-only delta log per relation, parallel to `rels`.
    logs: Vec<DeltaLog>,
    nodes: Vec<Node>,
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
            snapshots: vec![snapshot(map)?],
            n: map.len(),
            logs: vec![DeltaLog::new(); rels.len()],
            rels,
            nodes: Vec::new(),
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
        let snapshots = tiles
            .iter()
            .map(|&t| {
                table
                    .map(frame, t)
                    .map_err(CubeRefusal::Map)
                    .and_then(snapshot)
            })
            .collect::<Result<_, _>>()?;
        Ok(Self {
            id: NEXT_CUBE.fetch_add(1, Ordering::Relaxed),
            frame: frame.clone(),
            tile: tiles.first().copied().unwrap_or(0),
            tiles: tiles.to_vec(),
            tiled: true,
            snapshots,
            n: BLOCK_POSITIONS.saturating_mul(tiles.len()),
            logs: vec![DeltaLog::new(); rels.len()],
            rels,
            nodes: Vec::new(),
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
        let mut s = self.stats.get();
        s.relations_added += 1;
        self.stats.set(s);
        Ok((self.rels.len() - 1) as RelId)
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
        let value = match self.coord(table, value) {
            Ok(c) => Some(c),
            Err(CubeRefusal::Domain(_) | CubeRefusal::Locate(LocateRefusal::Unmapped)) => None,
            Err(e) => return Err(e),
        };
        Ok(self.push(Node::Bind { rel, col, value }))
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
            Node::Bind { rel, col, value } if col != join_col => Ok(Side {
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

    fn map<'t>(&self, table: &'t MapTable) -> Result<&'t LocalMap, CubeRefusal> {
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

    fn run(&self, side: Side, query: &Query, out: Out<'_>) -> Result<(), CubeRefusal> {
        let r = &self.rels[side.rel as usize];
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

    /// `edge⁺` of relation `rel`, semi-naively: `delta' = (delta ∘ edge) andnot
    /// full`, `full |= delta'`, until a delta is empty. Rows come from the
    /// relation's lanes, which already hold local positions, so nothing is
    /// looked up; the difference is a mask word, not a host-side set.
    pub fn closure(
        &self,
        table: &MapTable,
        rel: RelId,
    ) -> Result<(Closure, ClosureStats), CubeRefusal> {
        let r = self
            .rels
            .get(rel as usize)
            .ok_or(CubeRefusal::UnknownRel(rel))?;
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
    /// [`Cube::retire_edge`]). The fold is done on the fly and the base is not
    /// changed; replacing it is the explicit [`Cube::refold`]. The narrow
    /// [`Cube::closure`] still reads the base lanes only.
    pub fn closure_wide(
        &self,
        table: &MapTable,
        rel: RelId,
    ) -> Result<(WideRows, WideStats), CubeRefusal> {
        let r = self
            .rels
            .get(rel as usize)
            .ok_or(CubeRefusal::UnknownRel(rel))?;
        self.check_frame(table)?;
        let n = u32::try_from(self.n).unwrap_or(u32::MAX);
        let log = self.log(rel)?;
        if log.is_empty() {
            return wide::closure_wide(n, &r.cols[0], &r.cols[1]).map_err(CubeRefusal::Wide);
        }
        let (src, dst) = wide::refold(&r.cols[0], &r.cols[1], log);
        wide::closure_wide(n, &src, &dst).map_err(CubeRefusal::Wide)
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

    /// Logged operations on relation `rel` not yet folded into its base.
    /// An unknown relation has none.
    pub fn delta_len(&self, rel: RelId) -> usize {
        self.logs.get(rel as usize).map_or(0, DeltaLog::len)
    }

    /// Brings `old` up to date with relation `rel`'s delta log without a full
    /// rebuild ([`wide::refresh`]). **Precondition:** `old` is the closure of
    /// the relation's *base* lanes (for example the result of
    /// [`Cube::closure_wide`] taken while the log was empty, or of the last
    /// [`Cube::refold`]); given any other closure the result is not defined.
    pub fn refresh_closure(
        &self,
        table: &MapTable,
        rel: RelId,
        old: &WideRows,
    ) -> Result<(WideRows, WideStats), CubeRefusal> {
        let r = self
            .rels
            .get(rel as usize)
            .ok_or(CubeRefusal::UnknownRel(rel))?;
        self.check_frame(table)?;
        let n = u32::try_from(self.n).unwrap_or(u32::MAX);
        wide::refresh(old, &r.cols[0], &r.cols[1], self.log(rel)?, n).map_err(CubeRefusal::Wide)
    }

    /// Replaces relation `rel`'s base lanes with the base folded with its
    /// log, and empties the log. Explicit only: nothing in the cube calls
    /// this on a threshold, so the caller decides when a long log is worth
    /// the rewrite. The old closure of the base is invalid afterwards.
    pub fn refold(&mut self, rel: RelId) -> Result<(), CubeRefusal> {
        let log = self.log(rel)?;
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
        Ok(())
    }

    /// Adopts `frame`, sealed after the table advanced to add symbols. Allowed
    /// only if, for every tile the cube holds, the new map is a pure
    /// extension of the one the cube was built (or last rebased) on
    /// ([`LocalMap::extends`]): positions never renumber, so every stored
    /// coordinate and logged edge keeps naming the same symbol. Otherwise
    /// [`CubeRefusal::Renumbered`] names the tile and the cube is unchanged.
    ///
    /// A single-tile cube's `n` is its map's length, so it grows with the
    /// map; a tiled cube's positions were reserved at build and stay fixed.
    pub fn rebase(&mut self, table: &MapTable, frame: &Frame) -> Result<(), CubeRefusal> {
        let mut next = Vec::with_capacity(self.tiles.len());
        for (&t, old) in self.tiles.iter().zip(&self.snapshots) {
            let map = table.map(frame, t).map_err(CubeRefusal::Map)?;
            if !map.extends(old) {
                return Err(CubeRefusal::Renumbered(t));
            }
            next.push(snapshot(map)?);
        }
        if !self.tiled {
            if let Some(m) = next.first() {
                self.n = m.len();
            }
        }
        self.snapshots = next;
        self.frame = frame.clone();
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
        let rows = |side: Side, jc: u8| {
            let rel = &self.rels[side.rel as usize];
            let fc = 1 - jc as usize;
            let (j, f) = (&rel.cols[jc as usize], &rel.cols[fc]);
            let bound = side.bound.map(|(_, v)| v);
            (0..j.len())
                .filter(move |&i| keep[j[i] as usize / 64] >> (j[i] % 64) & 1 == 1)
                .filter(move |&i| bound.is_none_or(|v| v == Some(f[i])))
                .map(move |i| (j[i], f[i]))
        };
        let mut out = Vec::new();
        for (jl, fl) in rows(l, lc) {
            for (jr, fr) in rows(r, rc) {
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

fn check_col(col: u8) -> Result<(), CubeRefusal> {
    if col > 1 {
        Err(CubeRefusal::NoSuchColumn(col))
    } else {
        Ok(())
    }
}

/// A copy of a tile's map for later comparison by [`Cube::rebase`].
fn snapshot(map: &LocalMap) -> Result<LocalMap, CubeRefusal> {
    let globals: Vec<u64> = (0..map.len())
        .filter_map(|l| u8::try_from(l).ok().and_then(|l| map.global(l)))
        .collect();
    LocalMap::prepare(map.width(), &globals).map_err(CubeRefusal::Domain)
}

fn rel(tuples: &[(u32, u32)], map: &LocalMap) -> Result<Rel, CubeRefusal> {
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
