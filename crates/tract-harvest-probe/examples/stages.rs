//! Per-stage provenance of a real model against its raw protobuf.
//!
//! `cargo run --release --example stages -- fixtures/jina/onnx_model.onnx`
//!
//! Weights are not needed: Tract's lazy external data keeps every external
//! initializer as a named weight whose bytes are never read.
use std::time::Instant;
use tract_harvest_probe::{provenance, source::SourceInventory};
use tract_onnx::prelude::*;

fn main() -> TractResult<()> {
    let path = std::env::args().nth(1).expect("model path");
    let bytes = std::fs::read(&path)?;
    let t = Instant::now();
    let src = SourceInventory::from_bytes(&bytes)?;
    println!(
        "source: sha256 {} ir {} opsets {:?} nodes {} initializers {} ({:?})",
        src.model_sha256,
        src.ir_version,
        src.opsets,
        src.nodes.len(),
        src.initializers.len(),
        t.elapsed()
    );
    println!(
        "source nodes with empty name: {}",
        src.nodes.iter().filter(|n| n.name.is_empty()).count()
    );

    let onnx = tract_onnx::onnx().with_lazy_external_data(true);
    let t = Instant::now();
    let mut inf = onnx.model_for_path(&path)?;
    // Tract's Flatten refuses a shape with more than one symbol
    // (hir/src/ops/array/flatten.rs:11), and the export flattens
    // [batch_size, sequence_length]. Pin the inputs to the slice's shape.
    if let Some(seq) = std::env::args()
        .nth(2)
        .and_then(|s| s.parse::<usize>().ok())
    {
        for i in 0..inf.inputs.len() {
            inf.set_input_fact(i, InferenceFact::dt_shape(i64::datum_type(), [1, seq]))?;
        }
        println!("inputs pinned to [1, {seq}] i64");
    }
    let t_inf = t.elapsed();
    let t = Instant::now();
    let typed = inf.clone().into_typed()?;
    let t_typed = t.elapsed();
    let t = Instant::now();
    let decl = typed.clone().into_decluttered()?;
    let t_decl = t.elapsed();
    println!("timings: import {t_inf:?}, into_typed {t_typed:?}, declutter {t_decl:?}");
    for p in [
        provenance::of_graph("inference", &src, &inf),
        provenance::of_graph("typed", &src, &typed),
        provenance::of_graph("decluttered", &src, &decl),
    ] {
        println!("{}: nodes {}, source names kept {}, lost {:?}, synthesized {}, source outputs labelled {}/{}",
            p.stage, p.nodes, p.source_names_kept, p.source_names_lost, p.synthesized_names,
            p.source_outputs_labelled, p.source_outputs_total);
    }
    let mut ops = std::collections::BTreeMap::<String, usize>::new();
    for n in decl.nodes() {
        *ops.entry(n.op.name().to_string()).or_default() += 1;
    }
    println!("decluttered op histogram: {ops:?}");
    Ok(())
}
