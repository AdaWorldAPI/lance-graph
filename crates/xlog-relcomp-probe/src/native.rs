//! The admitted chain query and its bag count, described in Quack and
//! evaluated by mask-risc (D-DYNBAG-0, plan `d-dynbag-holographic-bags-v1.md`
//! §17).
//!
//! This is the execution path of the probe. The DR64 register path
//! (`propagate`, over `dr-math-binding-probe::relcomp`) stays as a
//! differential cross-check: it answers support only, and it is limited to
//! one 64-slot tile.
//!
//! Each side of `q(Y) :- r(X, Y), s(Y, Z)` is one table with two `u32` lanes
//! (the free end, the Y end as a local position) and one resident plane (its
//! live rows). Each answer is one Quack `Query`, lowered to one mask-risc
//! `Program` and run by `execute_into`:
//!
//! | step | Quack | mask-risc terminal |
//! |---|---|---|
//! | a bound end (`X == x`, `Z == z`), or every row | `Filter::cmp(free, EqU32)` / `Filter::plane(live)` | the program's mask |
//! | Y with a partner on one side (support) | `Agg::ScatterOrU32 { fk: y }` | `ScatterOrU32`, the demanded result |
//! | rows per Y on one side (bag count) | `Agg::GroupReduce { Local(y), Count }` | `GroupReduce`, `Out::I64` sized to the Y map |
//!
//! Two calls remain on `ndarray::simd` directly, and neither is a fold:
//! - `mask_and` (forward ∧ backward) is an elementwise alias shared with
//!   `lance-graph-onnx` (Amendment C of `d-xlog-native-v1.md`).
//! - `checked_binary_i64(Mul)` (left count × right count per Y) is TRACKED:
//!   mask-risc has no elementwise combine of two `i64` sinks yet (§17.5
//!   MISSING-3). A product past `i64` is refused, never wrapped.
//!
//! The Y domain is DR's: a `lance_graph_dt64::dr::LocalMap` (global id →
//! local position), prepared by the caller and never derived here. A Y id the
//! map does not hold is refused.
//!
//! The only Rust loops here relabel inputs and outputs: they build the key
//! lanes from the tuples (one local lookup per row), the live plane, and decode
//! the answer bits back to ids. None is a computation over the population.

use lance_graph_contract::set_rules::Population;
use lance_graph_dt64::dr::{Frame, LocalMap, MapRead, MapRefusal, MapTable, TileId};
use lance_graph_dt64::{AlgebraRef, MethodBinding};
use lance_graph_mask_risc::{ExecError, Foreign, LaneRef, Out, Planes, Scratch, execute_into};
use lance_graph_quack::{
    Agg, Cmp, Col, Filter, GroupAddr, GroupAgg, LowerError, Mask, Query, lower,
};
use ndarray::simd::{I64ArithError, I64Op, checked_binary_i64, mask_and};
use xlog_core::AggOp;
use xlog_ir::{ProjectExpr, RirNode};

use crate::{ChainQuery, LowerRefusal, admit};

/// Why the native path did not answer.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum NativeRefusal {
    /// A Y id outside DR's local map.
    Domain(MapRefusal),
    /// A per-Y derivation count past `i64`.
    Count(I64ArithError),
    /// Quack refused to lower the description.
    Lower(LowerError),
    /// mask-risc refused to run the program.
    Exec(ExecError),
}

/// The masks over the Y domain, one bit per local position.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Support {
    /// Y with a left partner (under the left bound, if any).
    pub forward: Vec<u64>,
    /// Y with a right partner (under the right bound, if any).
    pub backward: Vec<u64>,
    /// `forward & backward`: the answer.
    pub answer: Vec<u64>,
}

/// The free end of a side's rows.
const FREE: Col = Col(0);
/// The Y end of a side's rows, as a local position in DR's map.
const Y: Col = Col(1);
/// The side's live rows.
const LIVE: Mask = Mask(0);

/// One relation's lanes: the free end and the Y end as local positions, and
/// its live-rows plane.
struct Side {
    free: Vec<u32>,
    y_local: Vec<u32>,
    live: Vec<u64>,
}

fn side(
    tuples: &[(u32, u32)],
    y_of: fn(&(u32, u32)) -> u32,
    free_of: fn(&(u32, u32)) -> u32,
    mid: &LocalMap,
) -> Result<Side, NativeRefusal> {
    let y_local = tuples
        .iter()
        .map(|t| mid.local(u64::from(y_of(t))).map(u32::from))
        .collect::<Result<_, _>>()
        .map_err(NativeRefusal::Domain)?;
    let n = tuples.len();
    let live = (0..n.div_ceil(64))
        .map(|i| {
            let rows = (n - i * 64).min(64);
            if rows == 64 {
                u64::MAX
            } else {
                (1u64 << rows) - 1
            }
        })
        .collect();
    Ok(Side {
        free: tuples.iter().map(free_of).collect(),
        y_local,
        live,
    })
}

fn sides(
    left: &[(u32, u32)],
    right: &[(u32, u32)],
    mid: &LocalMap,
) -> Result<(Side, Side), NativeRefusal> {
    // r(X, Y): Y is column 1, X free. s(Y, Z): Y is column 0, Z free.
    Ok((
        side(left, |t| t.1, |t| t.0, mid)?,
        side(right, |t| t.0, |t| t.1, mid)?,
    ))
}

/// The rows of one side the query reads: those whose free end equals the
/// bound, or every live row.
fn selected(bound: Option<u32>) -> Filter {
    match bound {
        Some(v) => Filter::cmp(FREE, Cmp::EqU32(v)),
        None => Filter::plane(LIVE),
    }
}

/// Describe, lower and run one query over one side.
fn run(side: &Side, query: &Query, out: Out<'_>) -> Result<(), NativeRefusal> {
    let n = side.free.len();
    let masks: [&[u64]; 1] = [&side.live];
    let lanes = [LaneRef::U32(&side.free), LaneRef::U32(&side.y_local)];
    let planes = Planes {
        n_rows: n,
        masks: &masks,
        lanes: &lanes,
    };
    let program = lower(query).map_err(NativeRefusal::Lower)?;
    let mut scratch = Scratch::for_program(&program, n).map_err(NativeRefusal::Exec)?;
    execute_into(&program, &planes, &Foreign::NONE, &mut scratch, out)
        .map_err(NativeRefusal::Exec)?;
    Ok(())
}

/// The Y a side reaches: `ScatterOrU32` over the Y lane, the demanded result.
fn reached(side: &Side, bound: Option<u32>, n_y: usize) -> Result<Vec<u64>, NativeRefusal> {
    let mut words = vec![0u64; n_y.div_ceil(64)];
    let query = Query {
        filter: selected(bound),
        agg: Agg::ScatterOrU32 {
            fk: Y,
            out_rows: n_y as u32,
        },
    };
    run(side, &query, Out::Mask(&mut words))?;
    Ok(words)
}

/// Rows per Y on one side: `GroupReduce` keyed by the Y lane, `Count`.
fn degrees(side: &Side, bound: Option<u32>, n_y: usize) -> Result<Vec<i64>, NativeRefusal> {
    let mut counts = vec![0i64; n_y];
    let query = Query {
        filter: selected(bound),
        agg: Agg::GroupReduce {
            key: GroupAddr::Local(Y),
            agg: GroupAgg::Count,
        },
    };
    run(side, &query, Out::I64(&mut counts))?;
    Ok(counts)
}

/// The support of `q(Y) :- r(X, Y), s(Y, Z)` (with optional bounds on X and
/// Z) over DR's Y map.
pub fn support(
    q: &ChainQuery,
    left: &[(u32, u32)],
    right: &[(u32, u32)],
    mid: &LocalMap,
) -> Result<Support, NativeRefusal> {
    let (l, r) = sides(left, right, mid)?;
    support_of(&l, &r, q, mid.len())
}

fn support_of(l: &Side, r: &Side, q: &ChainQuery, n: usize) -> Result<Support, NativeRefusal> {
    let forward = reached(l, q.left_bound, n)?;
    let backward = reached(r, q.right_bound, n)?;
    let mut answer = vec![0u64; forward.len()];
    mask_and(&forward, &backward, &mut answer);
    Ok(Support {
        forward,
        backward,
        answer,
    })
}

/// Per-Y derivation counts of the same body under bag semantics: one per
/// `(x, y, z)` witness, as XLOG's `count` counts body bindings. Returned for
/// every local position; a Y with no derivation has 0.
pub fn bag_count(
    q: &ChainQuery,
    left: &[(u32, u32)],
    right: &[(u32, u32)],
    mid: &LocalMap,
) -> Result<Vec<i64>, NativeRefusal> {
    let (l, r) = sides(left, right, mid)?;
    bag_count_of(&l, &r, q, mid.len())
}

fn bag_count_of(l: &Side, r: &Side, q: &ChainQuery, n: usize) -> Result<Vec<i64>, NativeRefusal> {
    let per_left = degrees(l, q.left_bound, n)?;
    let per_right = degrees(r, q.right_bound, n)?;
    let mut out = vec![0i64; n];
    checked_binary_i64(I64Op::Mul, &per_left, &per_right, &mut out)
        .map_err(NativeRefusal::Count)?;
    Ok(out)
}

/// Both sides' lanes, built once from the tuples under one `MapTable` frame
/// (plan §17.6 gates 2 and 3). Every query re-checks that frame and reads the
/// lanes; none looks a Y up again.
pub struct Intake {
    frame: Frame,
    tile: TileId,
    n_y: usize,
    left: Side,
    right: Side,
}

impl Intake {
    /// The one intake: each tuple's Y looked up once in the tile's map, read
    /// under `frame`.
    pub fn new(
        table: &MapTable,
        frame: &Frame,
        tile: TileId,
        left: &[(u32, u32)],
        right: &[(u32, u32)],
    ) -> Result<Self, BoundRefusal> {
        let mid = table.map(frame, tile).map_err(BoundRefusal::Map)?;
        let (l, r) = sides(left, right, mid).map_err(BoundRefusal::Native)?;
        Ok(Self {
            frame: frame.clone(),
            tile,
            n_y: mid.len(),
            left: l,
            right: r,
        })
    }

    /// The Y map the lanes were built against, if its frame is still current.
    pub fn map<'t>(&self, table: &'t MapTable) -> Result<&'t LocalMap, BoundRefusal> {
        table.map(&self.frame, self.tile).map_err(BoundRefusal::Map)
    }

    /// [`support`] over the intake's lanes.
    pub fn support(&self, table: &MapTable, q: &ChainQuery) -> Result<Support, BoundRefusal> {
        self.map(table)?;
        support_of(&self.left, &self.right, q, self.n_y).map_err(BoundRefusal::Native)
    }

    /// [`bag_count`] over the intake's lanes.
    pub fn bag_count(&self, table: &MapTable, q: &ChainQuery) -> Result<Vec<i64>, BoundRefusal> {
        self.map(table)?;
        bag_count_of(&self.left, &self.right, q, self.n_y).map_err(BoundRefusal::Native)
    }
}

/// The operation a polyfill word stands for, as this probe reads it. The
/// bind store names words by position only; which Quack description a word
/// lowers to is MISSING-1 (plan §17.5), so the caller supplies the reading
/// and any word without one is refused.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Word {
    /// `GroupReduce { Local(y), Count }` over one side.
    GroupCount,
    /// The checked i64 product of the two per-Y counts (TRACKED, MISSING-3).
    MulI64,
}

/// Why an admitted method was not run.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum BoundRefusal {
    /// The method is not a bag count: support is never offered as a count.
    NotBag,
    /// The method names a word the caller gave no reading for.
    UnknownWord(AlgebraRef),
    /// The method's steps are not the two-sided count this path computes.
    Shape,
    /// DR's Y map is not readable under the given frame.
    Map(MapRead),
    /// The computation itself refused.
    Native(NativeRefusal),
}

/// Run an admitted bag-count method over an [`Intake`]. Execution starts from
/// the `MethodBinding` the bind store admitted, and the intake re-checks its
/// `MapTable` frame: no admitted method, or a stale frame, and nothing runs.
///
/// The method must be `Population::Bag` and its steps must read, in order,
/// [`Word::GroupCount`], [`Word::GroupCount`], [`Word::MulI64`].
pub fn bag_count_admitted(
    method: &MethodBinding,
    words: impl Fn(AlgebraRef) -> Option<Word>,
    intake: &Intake,
    table: &MapTable,
    q: &ChainQuery,
) -> Result<Vec<i64>, BoundRefusal> {
    if method.population != Population::Bag {
        return Err(BoundRefusal::NotBag);
    }
    let steps = method
        .effects
        .iter()
        .map(|e| words(e.algebra).ok_or(BoundRefusal::UnknownWord(e.algebra)))
        .collect::<Result<Vec<_>, _>>()?;
    if steps != [Word::GroupCount, Word::GroupCount, Word::MulI64] {
        return Err(BoundRefusal::Shape);
    }
    intake.bag_count(table, q)
}

/// Global ids of the set bits of `mask`, in local order.
pub fn ids(mid: &LocalMap, mask: &[u64]) -> Vec<u64> {
    (0..mid.len())
        .filter(|&i| mask[i / 64] >> (i % 64) & 1 == 1)
        .filter_map(|i| mid.global(i as u8))
        .collect()
}

/// `(global id, count)` for every Y with at least one derivation, in local
/// order.
pub fn nonzero(mid: &LocalMap, counts: &[i64]) -> Vec<(u64, i64)> {
    counts
        .iter()
        .enumerate()
        .filter(|(_, c)| **c != 0)
        .filter_map(|(i, c)| mid.global(i as u8).map(|g| (g, *c)))
        .collect()
}

/// Admit XLOG's compiled `count` rule over the chain:
/// `qc(Y, count(W)) :- r(X, Y), s(Y, Z)`, which XLOG compiles to
/// `Project[0, 1](GroupBy{key 0, Count of 1}(Project[Y, W](Join)))`.
/// The join must be the one [`admit`] accepts. Any other aggregate, key or
/// projection is refused.
pub fn admit_count(body: &RirNode) -> Result<ChainQuery, LowerRefusal> {
    let RirNode::Project { input, columns } = body else {
        return Err(LowerRefusal::UnsupportedNode(name(body)));
    };
    if !matches!(
        columns.as_slice(),
        [ProjectExpr::Column(0), ProjectExpr::Column(1)]
    ) {
        return Err(LowerRefusal::UnsupportedProjection);
    }
    let RirNode::GroupBy {
        input,
        key_cols,
        aggs,
    } = input.as_ref()
    else {
        return Err(LowerRefusal::UnsupportedNode(name(input)));
    };
    if key_cols.as_slice() != [0] || aggs.as_slice() != [(1, AggOp::Count)] {
        return Err(LowerRefusal::UnsupportedAggregate);
    }
    let RirNode::Project { input, columns } = input.as_ref() else {
        return Err(LowerRefusal::UnsupportedNode(name(input)));
    };
    // `count(W)` over an inner join counts rows: no column is null, so which
    // column W is does not matter. Y must be a join column `admit` accepts.
    let [ProjectExpr::Column(y), ProjectExpr::Column(_)] = columns.as_slice() else {
        return Err(LowerRefusal::UnsupportedProjection);
    };
    admit(&RirNode::Project {
        input: input.clone(),
        columns: vec![ProjectExpr::Column(*y)],
    })
}

fn name(n: &RirNode) -> &'static str {
    match n {
        RirNode::Project { .. } => "Project",
        RirNode::GroupBy { .. } => "GroupBy",
        RirNode::Join { .. } => "Join",
        RirNode::ChainJoin { .. } => "ChainJoin",
        RirNode::Scan { .. } => "Scan",
        RirNode::Filter { .. } => "Filter",
        _ => "other",
    }
}
