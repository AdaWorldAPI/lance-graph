//! End-to-end integration tests for the four-repo-demo crate.
//!
//! Covers:
//! 1. SumShader via `CognitiveShaderActor`: values 5, then 7 → cumulative sums 5, 12.
//! 2. OperatorTree: 3 nodes, correct shape, correct total cardinality.

use std::sync::Arc;

use anyhow::Result;
use arrow_array::{Int64Array, RecordBatch};
use arrow_schema::{DataType, Field, Schema};
use cognitive_shader_actor::actor::CognitiveShaderActor;
use cognitive_shader_actor::messages::ShaderMessage;
use four_repo_demo::planner::{build_demo_tree, node_count, APPLY_ROWS, FILTER_ROWS, SCAN_ROWS};
use four_repo_demo::SumShader;
use lance_graph_contract::ir::{Cardinality, EngineHint, OperatorKind};
use ractor::{call, Actor};

// ---------------------------------------------------------------------------
// Helper
// ---------------------------------------------------------------------------

fn value_schema() -> Arc<arrow_schema::Schema> {
    Arc::new(arrow_schema::Schema::new(vec![Field::new(
        "value",
        DataType::Int64,
        false,
    )]))
}

fn single_row(value: i64, schema: &Arc<arrow_schema::Schema>) -> RecordBatch {
    RecordBatch::try_new(
        schema.clone(),
        vec![Arc::new(Int64Array::from(vec![value]))],
    )
    .unwrap()
}

fn extract_sum(batch: &RecordBatch) -> i64 {
    batch
        .column(0)
        .as_any()
        .downcast_ref::<Int64Array>()
        .unwrap()
        .value(0)
}

// ---------------------------------------------------------------------------
// Test 1 — CognitiveShaderActor + SumShader
// ---------------------------------------------------------------------------

/// Verify that the SumShader returns 5 after sending value=5, then 12 after
/// sending value=7, and that the actor drains cleanly.
#[tokio::test]
async fn sum_shader_actor_running_sum() -> Result<()> {
    let schema = value_schema();
    let shader = Arc::new(SumShader::new());

    let (actor_ref, handle) = Actor::spawn(
        Some("e2e-sum-shader".into()),
        CognitiveShaderActor::<SumShader>::new(),
        shader,
    )
    .await?;

    // Apply value=5 → expect running sum 5
    let r1 = call!(actor_ref, |reply| ShaderMessage::Apply {
        input: single_row(5, &schema),
        reply,
    })?
    .map_err(|e| anyhow::anyhow!("{}", e))?;
    assert_eq!(
        extract_sum(&r1),
        5,
        "after value=5 the running sum should be 5"
    );

    // Apply value=7 → expect running sum 12
    let r2 = call!(actor_ref, |reply| ShaderMessage::Apply {
        input: single_row(7, &schema),
        reply,
    })?
    .map_err(|e| anyhow::anyhow!("{}", e))?;
    assert_eq!(
        extract_sum(&r2),
        12,
        "after value=7 the running sum should be 12"
    );

    // Drain cleanly
    call!(actor_ref, |reply| ShaderMessage::Drain { reply })?;
    handle.await?;

    Ok(())
}

// ---------------------------------------------------------------------------
// Test 2 — OperatorTree shape and cardinality
// ---------------------------------------------------------------------------

/// Verify the demo operator tree has exactly 3 nodes.
#[test]
fn operator_tree_has_three_nodes() {
    let tree = build_demo_tree();
    assert_eq!(node_count(&tree), 3, "expected 3 nodes: RangeScan+Filter+CognitiveApply");
}

/// Verify the tree nodes are in the expected shape: CognitiveApply → Filter → RangeScan.
#[test]
fn operator_tree_shape_correct() {
    let tree = build_demo_tree();

    // Root
    assert_eq!(tree.op.kind, OperatorKind::CognitiveApply);
    assert_eq!(tree.op.engine, EngineHint::Cognitive);
    assert_eq!(tree.children.len(), 1);

    // Filter
    let filter = &tree.children[0];
    assert_eq!(filter.op.kind, OperatorKind::Filter);
    assert_eq!(filter.op.engine, EngineHint::LanceGraph);
    assert_eq!(filter.children.len(), 1);

    // RangeScan
    let scan = &filter.children[0];
    assert_eq!(scan.op.kind, OperatorKind::RangeScan);
    assert_eq!(scan.op.engine, EngineHint::Tikv);
    assert!(scan.children.is_empty());
}

/// Verify the total estimated cardinality is SCAN_ROWS + FILTER_ROWS + APPLY_ROWS.
#[test]
fn operator_tree_total_cardinality() {
    let tree = build_demo_tree();
    let expected = Cardinality::rows(SCAN_ROWS + FILTER_ROWS + APPLY_ROWS);
    assert_eq!(
        tree.total_estimated_cardinality(),
        expected,
        "total cardinality should be {SCAN_ROWS} + {FILTER_ROWS} + {APPLY_ROWS} = {}",
        SCAN_ROWS + FILTER_ROWS + APPLY_ROWS
    );
}
