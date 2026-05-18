//! Tiny demo planner that exercises [`lance_graph_contract::ir`] types.
//!
//! Builds a 3-node operator tree:
//!
//! ```text
//! CognitiveApply (Cognitive engine, ~1 row out)
//! └── Filter        (LanceGraph engine, ~500 rows)
//!     └── RangeScan (TiKV engine,      ~10 000 rows)
//! ```
//!
//! The tree is then inspected to demonstrate:
//! 1. [`OperatorTree::walk`] visiting all three nodes.
//! 2. [`OperatorTree::total_estimated_cardinality`] summing the three estimates.
//!
//! Nothing here touches a real database — this is a pure-IR demo showing
//! that the contract crate's IR types compose correctly and that a consumer
//! can build and introspect a query plan without touching any engine.

use lance_graph_contract::ir::{Cardinality, EngineHint, Operator, OperatorKind, OperatorTree};

/// Estimated cardinalities used by the demo plan.
pub const SCAN_ROWS: u64 = 10_000;
pub const FILTER_ROWS: u64 = 500;
pub const APPLY_ROWS: u64 = 1;

/// Build the demo 3-node operator tree.
///
/// ```
/// use four_repo_demo::planner::build_demo_tree;
///
/// let tree = build_demo_tree();
/// assert_eq!(tree.children.len(), 1);            // CognitiveApply has one child
/// assert_eq!(tree.children[0].children.len(), 1); // Filter has one child (RangeScan)
/// ```
pub fn build_demo_tree() -> OperatorTree {
    // Leaf: RangeScan over TiKV.
    let scan = OperatorTree::leaf(
        Operator::new(OperatorKind::RangeScan)
            .with_engine(EngineHint::Tikv)
            .with_cardinality(Cardinality::rows(SCAN_ROWS))
            .with_tag("scan_events"),
    );

    // Middle: Filter executed by the lance-graph engine.
    let filter = OperatorTree::node(
        Operator::new(OperatorKind::Filter)
            .with_engine(EngineHint::LanceGraph)
            .with_cardinality(Cardinality::rows(FILTER_ROWS))
            .with_tag("filter_recent"),
        vec![scan],
    );

    // Root: CognitiveApply (runs the SumShader in the Cognitive engine).
    OperatorTree::node(
        Operator::new(OperatorKind::CognitiveApply)
            .with_engine(EngineHint::Cognitive)
            .with_cardinality(Cardinality::rows(APPLY_ROWS))
            .with_tag("sum_shader"),
        vec![filter],
    )
}

/// Returns the number of nodes in the tree (depth-first count).
pub fn node_count(tree: &OperatorTree) -> usize {
    let mut count = 0usize;
    tree.walk(&mut |_| count += 1);
    count
}

/// Print a human-readable summary of the tree and its total cardinality.
pub fn print_plan(tree: &OperatorTree) {
    println!("=== Demo Operator Tree ===");
    print_node(tree, 0);
    println!(
        "Total estimated cardinality: {}",
        tree.total_estimated_cardinality()
    );
}

fn print_node(tree: &OperatorTree, depth: usize) {
    let indent = "  ".repeat(depth);
    let tag = tree.op.tag.unwrap_or("(no tag)");
    let engine = format!("{:?}", tree.op.engine);
    let card = tree
        .op
        .estimated_cardinality
        .map(|c| format!("{}", c))
        .unwrap_or_else(|| "unknown".into());
    println!(
        "{}{:?} [{}] engine={} cardinality={}",
        indent,
        tree.op.kind,
        tag,
        engine,
        card
    );
    for child in &tree.children {
        print_node(child, depth + 1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tree_has_three_nodes() {
        let tree = build_demo_tree();
        assert_eq!(node_count(&tree), 3);
    }

    #[test]
    fn total_cardinality_is_sum_of_all_estimates() {
        let tree = build_demo_tree();
        let expected = SCAN_ROWS + FILTER_ROWS + APPLY_ROWS;
        assert_eq!(
            tree.total_estimated_cardinality(),
            Cardinality::rows(expected)
        );
    }

    #[test]
    fn tree_shape_is_correct() {
        let tree = build_demo_tree();
        // Root: CognitiveApply
        assert_eq!(tree.op.kind, OperatorKind::CognitiveApply);
        assert_eq!(tree.op.engine, EngineHint::Cognitive);
        // Child: Filter
        let filter = &tree.children[0];
        assert_eq!(filter.op.kind, OperatorKind::Filter);
        assert_eq!(filter.op.engine, EngineHint::LanceGraph);
        // Grandchild: RangeScan
        let scan = &filter.children[0];
        assert_eq!(scan.op.kind, OperatorKind::RangeScan);
        assert_eq!(scan.op.engine, EngineHint::Tikv);
    }
}
