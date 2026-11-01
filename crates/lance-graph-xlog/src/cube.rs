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
//! free column. Recursion, union and difference are not here.

use std::cell::Cell;
use std::sync::atomic::{AtomicU64, Ordering};

use lance_graph_dt64::dr::{Frame, LocalMap, MapRead, MapRefusal, MapTable, TileId};
use lance_graph_mask_risc::{LaneRef, Out};
use lance_graph_quack::{Agg, Cmp, Col, Filter, GroupAddr, GroupAgg, Mask, Query};
use ndarray::simd::{checked_binary_i64, mask_and, I64Op};

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
        value: Option<u8>,
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
}

/// What the cube has done since it was built.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct CubeStats {
    /// mask-risc programs run.
    pub programs: u64,
    /// Tuples built by [`Cube::materialize_rows`].
    pub rows_materialized: u64,
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
    bound: Option<(u8, Option<u8>)>,
}

/// A side with the column it joins on.
type JoinSide = (Side, u8);

static NEXT_CUBE: AtomicU64 = AtomicU64::new(1);

/// The resident hypercube. See the module docs.
pub struct Cube {
    id: u64,
    frame: Frame,
    tile: TileId,
    n: usize,
    rels: Vec<Rel>,
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
        let rels = relations
            .iter()
            .map(|tuples| rel(tuples, map))
            .collect::<Result<_, _>>()?;
        Ok(Self {
            id: NEXT_CUBE.fetch_add(1, Ordering::Relaxed),
            frame: frame.clone(),
            tile,
            n: map.len(),
            rels,
            nodes: Vec::new(),
            stats: Cell::new(CubeStats::default()),
        })
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
        let map = self.map(table)?;
        let value = map.local(u64::from(value)).ok();
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
            Some((col, Some(v))) => {
                Some(Filter::cmp(Col(u16::from(col)), Cmp::EqU32(u32::from(v))))
            }
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
        self.map(table)?;
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
        self.map(table)?;
        let a = self.degrees(l, lc)?;
        let b = self.degrees(r, rc)?;
        let mut out = vec![0i64; self.n];
        checked_binary_i64(I64Op::Mul, &a, &b, &mut out)
            .map_err(|e| CubeRefusal::Native(NativeRefusal::Count(e)))?;
        Ok(out)
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
        let map = self.map(table)?;
        let keep = &self.support(table, join)?;
        let rows = |side: Side, jc: u8| {
            let rel = &self.rels[side.rel as usize];
            let fc = 1 - jc as usize;
            let (j, f) = (&rel.cols[jc as usize], &rel.cols[fc]);
            let bound = side.bound.map(|(_, v)| v);
            (0..j.len())
                .filter(move |&i| keep[j[i] as usize / 64] >> (j[i] % 64) & 1 == 1)
                .filter(move |&i| bound.is_none_or(|v| v == Some(f[i] as u8)))
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

fn rel(tuples: &[(u32, u32)], map: &LocalMap) -> Result<Rel, CubeRefusal> {
    let mut cols = [
        Vec::with_capacity(tuples.len()),
        Vec::with_capacity(tuples.len()),
    ];
    for &(a, b) in tuples {
        for (c, id) in [(0, a), (1, b)] {
            let l = map.local(u64::from(id)).map_err(CubeRefusal::Domain)?;
            cols[c].push(u32::from(l));
        }
    }
    Ok(Rel {
        n: tuples.len(),
        live: all_live(tuples.len()),
        cols,
    })
}
