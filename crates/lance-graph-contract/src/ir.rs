//! Federated planner IR — operator vocabulary for cross-engine query routing.
//!
//! # Scope
//!
//! This module defines the **operator IR** that a *federated* planner uses
//! to decide which backend executes which part of a query:
//!
//! - PK lookup → KV (RocksDB / SurrealKV / TiKV)
//! - Range / aggregate → Lance projection (DataFusion)
//! - Cypher / graph traversal → lance-graph
//! - Vector ANN → lance-index
//! - Cognitive reasoning → cognitive-shader-actor
//!
//! It is intentionally **distinct from `crate::plan::PlannerContract`**,
//! which is the *lance-graph planner's* interface (the engine that
//! consumers like ladybug-rs talk to). The IR here is a level above:
//! it carries enough information for a cross-engine planner to choose
//! between engines per operator.
//!
//! It is also distinct from `crate::orchestration::OrchestrationBridge`,
//! which routes whole steps across systems (crewai-rust, n8n-rs, etc.).
//! The IR here routes operators within a single query.
//!
//! # Additive contract
//!
//! Added in 0.2.0. Pure addition — no existing surface in this crate is
//! touched. Consumers that don't import [`ir`] see no change.
//!
//! # Zero-dep
//!
//! Like the rest of this crate, this module pulls no dependencies. The
//! concrete planner that consumes the IR (in `surrealdb-core`) does the
//! cost-based decisions; here we just define the vocabulary.

use core::fmt;

/// Cardinality estimate for an operator's output, in rows.
///
/// `None` = unknown. Implementations should prefer to return `Some(0)`
/// rather than `None` when they can prove the result is empty
/// (e.g. a contradiction predicate), so the planner can prune the branch.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Cardinality(pub u64);

impl Cardinality {
    pub const UNKNOWN: Option<Self> = None;
    pub const EMPTY: Self = Self(0);

    #[inline]
    pub const fn rows(rows: u64) -> Self {
        Self(rows)
    }

    /// Saturating addition — for unioning two operator outputs.
    #[inline]
    pub const fn saturating_add(self, other: Self) -> Self {
        Self(self.0.saturating_add(other.0))
    }

    /// Saturating multiplication — for cross-products.
    #[inline]
    pub const fn saturating_mul(self, other: Self) -> Self {
        Self(self.0.saturating_mul(other.0))
    }
}

impl fmt::Display for Cardinality {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} rows", self.0)
    }
}

/// Which engine an operator targets. The planner uses this to decide
/// dispatch and to estimate per-engine cost.
///
/// Adding a new engine is an additive enum variant — existing consumers
/// continue to compile (they will get a `non_exhaustive`-style warning
/// if they `match` exhaustively; mitigation: use `_ =>`).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum EngineHint {
    /// Embedded LSM KV (RocksDB / SurrealKV).
    LocalKv,
    /// Distributed transactional KV with native MVCC.
    Tikv,
    /// Lance dataset (DataFusion analytic path).
    Lance,
    /// In-process Cypher engine over Arrow tables.
    LanceGraph,
    /// HNSW / IVF_PQ vector index.
    VectorIndex,
    /// Cognitive shader (lance-graph-cognitive crates).
    Cognitive,
    /// Engine selection is deferred to the planner's cost model.
    #[default]
    Auto,
}

/// Operator kind. The IR is intentionally coarse — it carries enough
/// signal for routing decisions, not enough to be a full physical plan.
///
/// Concrete physical operators live in each engine.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum OperatorKind {
    /// Read rows by primary key (point lookup).
    PointGet,
    /// Read rows by key range or secondary index range.
    RangeScan,
    /// Apply a predicate to a stream of rows.
    Filter,
    /// Project a subset of columns.
    Project,
    /// Aggregate (sum / count / avg / group-by).
    Aggregate,
    /// Inner / outer join across two operators.
    Join,
    /// Graph pattern match (Cypher MATCH).
    GraphMatch,
    /// Variable-length path traversal.
    GraphExpand,
    /// Vector ANN by distance.
    VectorAnn,
    /// Cognitive shader application (e.g. thinking-engine).
    CognitiveApply,
    /// Sink: write rows to a destination (typically transactional).
    Sink,
}

/// One operator in the IR tree.
///
/// This is a value type — `Operator` instances are cheap to construct
/// and inspect. The planner walks an [`OperatorTree`] and chooses an
/// [`EngineHint`] per node.
#[derive(Clone, Debug)]
pub struct Operator {
    /// What this operator does.
    pub kind: OperatorKind,
    /// Where it should run. `Auto` defers to the cost model.
    pub engine: EngineHint,
    /// Estimated output cardinality (None = unknown).
    pub estimated_cardinality: Option<Cardinality>,
    /// Tag for diagnostics / EXPLAIN output.
    pub tag: Option<&'static str>,
}

impl Operator {
    /// Build a new operator. The estimated_cardinality and tag default to None.
    pub const fn new(kind: OperatorKind) -> Self {
        Self {
            kind,
            engine: EngineHint::Auto,
            estimated_cardinality: None,
            tag: None,
        }
    }

    /// Builder: pin the engine.
    pub const fn with_engine(mut self, engine: EngineHint) -> Self {
        self.engine = engine;
        self
    }

    /// Builder: attach a cardinality estimate.
    pub const fn with_cardinality(mut self, c: Cardinality) -> Self {
        self.estimated_cardinality = Some(c);
        self
    }

    /// Builder: attach a diagnostic tag.
    pub const fn with_tag(mut self, tag: &'static str) -> Self {
        self.tag = Some(tag);
        self
    }
}

/// Composable operator tree. The IR is a tree, not a DAG — shared
/// subexpressions are inlined (the planner is free to CSE downstream).
///
/// Leaves have no children; internal nodes carry child operators.
#[derive(Clone, Debug)]
pub struct OperatorTree {
    pub op: Operator,
    pub children: Vec<OperatorTree>,
}

impl OperatorTree {
    /// Leaf node.
    pub fn leaf(op: Operator) -> Self {
        Self {
            op,
            children: Vec::new(),
        }
    }

    /// Internal node with the given children.
    pub fn node(op: Operator, children: Vec<OperatorTree>) -> Self {
        Self { op, children }
    }

    /// Walk the tree depth-first, calling `f` on every node.
    pub fn walk(&self, f: &mut impl FnMut(&Operator)) {
        f(&self.op);
        for c in &self.children {
            c.walk(f);
        }
    }

    /// Sum of estimated cardinalities of all nodes whose estimate is known.
    /// Useful as a rough total-work estimator for plan comparison.
    pub fn total_estimated_cardinality(&self) -> Cardinality {
        let mut total = Cardinality::EMPTY;
        self.walk(&mut |op| {
            if let Some(c) = op.estimated_cardinality {
                total = total.saturating_add(c);
            }
        });
        total
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cardinality_arithmetic_saturates() {
        let a = Cardinality::rows(u64::MAX - 1);
        let b = Cardinality::rows(10);
        assert_eq!(a.saturating_add(b), Cardinality::rows(u64::MAX));
        assert_eq!(a.saturating_mul(b), Cardinality::rows(u64::MAX));
    }

    #[test]
    fn operator_builder_chain() {
        let op = Operator::new(OperatorKind::RangeScan)
            .with_engine(EngineHint::Tikv)
            .with_cardinality(Cardinality::rows(1000))
            .with_tag("scan_users");
        assert_eq!(op.kind, OperatorKind::RangeScan);
        assert_eq!(op.engine, EngineHint::Tikv);
        assert_eq!(op.estimated_cardinality, Some(Cardinality::rows(1000)));
        assert_eq!(op.tag, Some("scan_users"));
    }

    #[test]
    fn tree_walks_in_dfs_order() {
        let tree = OperatorTree::node(
            Operator::new(OperatorKind::Join).with_tag("root"),
            vec![
                OperatorTree::leaf(
                    Operator::new(OperatorKind::RangeScan)
                        .with_engine(EngineHint::Tikv)
                        .with_tag("left"),
                ),
                OperatorTree::leaf(
                    Operator::new(OperatorKind::PointGet)
                        .with_engine(EngineHint::LocalKv)
                        .with_tag("right"),
                ),
            ],
        );
        let mut visited = Vec::new();
        tree.walk(&mut |op| visited.push(op.tag.unwrap_or("?")));
        assert_eq!(visited, vec!["root", "left", "right"]);
    }

    #[test]
    fn total_cardinality_sums_known_only() {
        let tree = OperatorTree::node(
            Operator::new(OperatorKind::Join), // no estimate
            vec![
                OperatorTree::leaf(
                    Operator::new(OperatorKind::RangeScan)
                        .with_cardinality(Cardinality::rows(100)),
                ),
                OperatorTree::leaf(
                    Operator::new(OperatorKind::RangeScan)
                        .with_cardinality(Cardinality::rows(200)),
                ),
                OperatorTree::leaf(Operator::new(OperatorKind::Filter)), // no estimate
            ],
        );
        assert_eq!(tree.total_estimated_cardinality(), Cardinality::rows(300));
    }

    #[test]
    fn engine_hint_default_is_auto() {
        assert_eq!(EngineHint::default(), EngineHint::Auto);
    }
}
