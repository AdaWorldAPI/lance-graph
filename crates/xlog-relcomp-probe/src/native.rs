//! The admitted chain query and its bag count, computed on the
//! `ndarray::simd` polyfill.
//!
//! This is the execution path of the probe. The DR64 register path
//! (`propagate`, over `dr-math-binding-probe::relcomp`) stays as a
//! differential cross-check: it answers support only, and it is limited to
//! one 64-slot tile.
//!
//! Every step is one `ndarray::simd` function, with no loop over rows of our
//! own:
//!
//! | step | function |
//! |---|---|
//! | a bound end (`X == x`, `Z == z`) | `eq_u32_to_mask` |
//! | the other relation's ends projected onto Y | `mask_scatter_or_u32` (result only) |
//! | forward ∧ backward | `mask_and` |
//! | rows per Y, on each side | `masked_group_count_u32` |
//! | derivations per Y = left count × right count | `checked_binary_i64(Mul)` |
//!
//! The Y domain is DR's: a `lance_graph_dt64::dr::LocalMap` (global id →
//! local position), prepared by the caller and never derived here. A Y id the
//! map does not hold is refused. A product past `i64` is refused, never
//! wrapped.
//!
//! The only Rust loops here relabel inputs and outputs. They build the key
//! columns from the tuples (one local lookup per row) and decode the answer
//! bits back to ids. Neither is a computation over the population.

use lance_graph_dt64::dr::{LocalMap, MapRefusal};
use ndarray::simd::{
    I64ArithError, I64Op, checked_binary_i64, eq_u32_to_mask, mask_and, mask_scatter_or_u32,
    masked_group_count_u32,
};
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

/// One relation's columns: the free end and the Y end as local positions.
struct Side {
    free: Vec<u32>,
    y_local: Vec<u32>,
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
    Ok(Side {
        free: tuples.iter().map(free_of).collect(),
        y_local,
    })
}

/// The selected rows of one side: all of them, or those whose free end
/// equals the bound.
fn rows(side: &Side, bound: Option<u32>) -> Vec<u64> {
    let n = side.free.len();
    let mut words = vec![0u64; n.div_ceil(64)];
    match bound {
        Some(v) => eq_u32_to_mask(&side.free, v, &mut words),
        None => {
            for (i, w) in words.iter_mut().enumerate() {
                let live = (n - i * 64).min(64);
                *w = if live == 64 {
                    u64::MAX
                } else {
                    (1u64 << live) - 1
                };
            }
        }
    }
    words
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

/// The support of `q(Y) :- r(X, Y), s(Y, Z)` (with optional bounds on X and
/// Z) over DR's Y map.
pub fn support(
    q: &ChainQuery,
    left: &[(u32, u32)],
    right: &[(u32, u32)],
    mid: &LocalMap,
) -> Result<Support, NativeRefusal> {
    let (l, r) = sides(left, right, mid)?;
    let n = mid.len();
    let words = n.div_ceil(64);
    let mut forward = vec![0u64; words];
    let mut backward = vec![0u64; words];
    mask_scatter_or_u32(&rows(&l, q.left_bound), &l.y_local, &mut forward, n);
    mask_scatter_or_u32(&rows(&r, q.right_bound), &r.y_local, &mut backward, n);
    let mut answer = vec![0u64; words];
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
    let n = mid.len();
    let mut per_left = vec![0i64; n];
    let mut per_right = vec![0i64; n];
    masked_group_count_u32(&rows(&l, q.left_bound), &l.y_local, &mut per_left);
    masked_group_count_u32(&rows(&r, q.right_bound), &r.y_local, &mut per_right);
    let mut out = vec![0i64; n];
    checked_binary_i64(I64Op::Mul, &per_left, &per_right, &mut out)
        .map_err(NativeRefusal::Count)?;
    Ok(out)
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
