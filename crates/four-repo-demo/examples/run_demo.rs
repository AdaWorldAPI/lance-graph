//! # run_demo — end-to-end integration demo
//!
//! Spawns a `CognitiveShaderActor::<SumShader>` via ractor, sends two Apply
//! messages with single-row batches (values 5, 7), then drains the actor and
//! prints the running sum (expected: 12).  Also prints the planner's operator
//! tree and its estimated total cardinality.
//!
//! Run with:
//! ```text
//! cargo run --manifest-path crates/four-repo-demo/Cargo.toml --example run_demo
//! ```

use std::sync::Arc;

use anyhow::Result;
use arrow_array::{Int64Array, RecordBatch};
use arrow_schema::{DataType, Field, Schema};
use cognitive_shader_actor::actor::CognitiveShaderActor;
use cognitive_shader_actor::messages::ShaderMessage;
use four_repo_demo::{build_demo_tree, print_plan, SumShader};
use ractor::{call, Actor};

#[tokio::main]
async fn main() -> Result<()> {
    // -----------------------------------------------------------------------
    // Part 1: Cognitive shader actor
    // -----------------------------------------------------------------------
    println!("=== Cognitive Shader Actor Demo ===");

    let shader = Arc::new(SumShader::new());
    let schema = Arc::new(Schema::new(vec![Field::new("value", DataType::Int64, false)]));

    // Spawn the actor with the SumShader instance as its arguments.
    let (actor_ref, handle) = Actor::spawn(
        Some("sum-shader-demo".into()),
        CognitiveShaderActor::<SumShader>::new(),
        shader,
    )
    .await?;

    // --- Apply batch 1: value = 5 ---
    let batch1 = RecordBatch::try_new(
        schema.clone(),
        vec![Arc::new(Int64Array::from(vec![5_i64]))],
    )?;
    let result1 = call!(actor_ref, |reply| ShaderMessage::Apply {
        input: batch1,
        reply,
    })?
    .map_err(|e| anyhow::anyhow!("apply 1 failed: {}", e))?;
    let sum_after_5 = result1
        .column(0)
        .as_any()
        .downcast_ref::<Int64Array>()
        .unwrap()
        .value(0);
    println!("After sending value=5  → running sum = {}", sum_after_5);

    // --- Apply batch 2: value = 7 ---
    let batch2 = RecordBatch::try_new(
        schema.clone(),
        vec![Arc::new(Int64Array::from(vec![7_i64]))],
    )?;
    let result2 = call!(actor_ref, |reply| ShaderMessage::Apply {
        input: batch2,
        reply,
    })?
    .map_err(|e| anyhow::anyhow!("apply 2 failed: {}", e))?;
    let sum_after_12 = result2
        .column(0)
        .as_any()
        .downcast_ref::<Int64Array>()
        .unwrap()
        .value(0);
    println!("After sending value=7  → running sum = {}", sum_after_12);

    // --- Drain ---
    call!(actor_ref, |reply| ShaderMessage::Drain { reply })?;
    println!("Actor drained cleanly.");
    handle.await?;

    println!();

    // -----------------------------------------------------------------------
    // Part 2: Planner IR demo
    // -----------------------------------------------------------------------
    let tree = build_demo_tree();
    print_plan(&tree);

    println!();
    println!("=== Summary ===");
    println!("Running sum after 5+7 = {}", sum_after_12);
    println!(
        "Operator tree node count = {}",
        four_repo_demo::node_count(&tree)
    );
    println!(
        "Total estimated cardinality = {}",
        tree.total_estimated_cardinality()
    );

    Ok(())
}
