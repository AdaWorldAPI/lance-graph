//! The admitted chain query and its bag count. The computation lives in
//! `lance-graph-xlog` (Quack → mask-risc, gated by the bind store); this
//! module keeps what needs XLOG: admission of the compiled `count` rule, and
//! the `ChainQuery` → [`Bounds`] reading.
//!
//! The DR64 register path (`propagate`) stays as the differential.

use lance_graph_dt64::dr::LocalMap;
pub use lance_graph_xlog::{
    BoundRefusal, Bounds, Intake, NativeRefusal, Support, Word, bag_count_admitted, ids, nonzero,
};
use xlog_core::AggOp;
use xlog_ir::{ProjectExpr, RirNode};

use crate::{ChainQuery, LowerRefusal, admit};

/// [`lance_graph_xlog::support`] for an admitted chain.
pub fn support(
    q: &ChainQuery,
    left: &[(u32, u32)],
    right: &[(u32, u32)],
    mid: &LocalMap,
) -> Result<Support, NativeRefusal> {
    lance_graph_xlog::support(q.bounds(), left, right, mid)
}

/// [`lance_graph_xlog::bag_count`] for an admitted chain.
pub fn bag_count(
    q: &ChainQuery,
    left: &[(u32, u32)],
    right: &[(u32, u32)],
    mid: &LocalMap,
) -> Result<Vec<i64>, NativeRefusal> {
    lance_graph_xlog::bag_count(q.bounds(), left, right, mid)
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
