//! Dependency census of the shape-plumbing ops, and what folds.
//!
//! `cargo run --release --example shape_census -- fixtures/jina/onnx_model.onnx [seq]`
//!
//! Without `seq` the inputs keep their symbolic dims; with it they are pinned
//! to `[1, seq]`.
use std::time::Instant;
use tract_harvest_probe::{dr3, fold, registry, source::SourceInventory};
use tract_onnx::prelude::*;

const OPS: &[&str] = &["Unsqueeze", "Shape", "Gather", "Concat", "Reshape"];

fn main() {
    let path = std::env::args().nth(1).expect("model path");
    let src = SourceInventory::from_bytes(&std::fs::read(&path).unwrap()).unwrap();
    let pins: Vec<_> = std::env::args()
        .nth(2)
        .and_then(|s| s.parse::<usize>().ok())
        .map(|seq| {
            (0..src.inputs.len())
                .map(|i| (i, i64::datum_type(), vec![1, seq]))
                .collect()
        })
        .unwrap_or_default();
    let t = Instant::now();
    let fs = dr3::tract_fact_sources_path(std::path::Path::new(&path), &pins).unwrap();
    let t_facts = t.elapsed();
    let facts = fs.merged();
    let mut rj = std::collections::BTreeMap::<String, usize>::new();
    for r in &fs.rejected {
        *rj.entry(format!(
            "{}: {}",
            r.stage,
            r.reason
                .split(" (")
                .next()
                .unwrap_or(&r.reason)
                .chars()
                .take(40)
                .collect::<String>()
        ))
        .or_default() += 1;
    }
    println!(
        "admitted facts: inference {}, typed {}; rejected {:?}",
        fs.inference.len(),
        fs.typed.len(),
        rj
    );
    let (_, node_dep) = fold::classify(&src, &facts);
    println!(
        "pins {:?}; typed stage error: {:?}",
        pins.first().map(|p| &p.2),
        fs.typed_error.as_deref().map(|e| &e[..e.len().min(80)])
    );
    println!("census [Const, Shape, Symbolic, Data]:");
    for (op, c) in fold::census(&src, &node_dep, OPS) {
        println!("  {op:>10} {c:?}");
    }
    let all = fold::census(
        &src,
        &node_dep,
        &src.nodes
            .iter()
            .map(|n| n.op_type.as_str())
            .collect::<Vec<_>>(),
    );
    let tot = all.values().fold([0; 4], |a, c| {
        [a[0] + c[0], a[1] + c[1], a[2] + c[2], a[3] + c[3]]
    });
    println!("  all nodes  {tot:?}");
    let reg = registry::registry();
    let t = Instant::now();
    let f = fold::fold(&src, &facts, &reg);
    let t_fold = t.elapsed();
    let retained = f.retained(&src);
    let bytes: usize = retained
        .iter()
        .map(|(_, v)| v.shape().iter().product::<usize>() * 8)
        .sum();
    println!(
        "folded {} nodes in {t_fold:?} (facts {t_facts:?}); retained constants {} (~{bytes} B)",
        f.nodes.len(),
        retained.len()
    );
    let mut by = std::collections::BTreeMap::<String, usize>::new();
    for (_, why) in &f.blocked {
        *by.entry(why.clone()).or_default() += 1;
    }
    let cov = tract_harvest_probe::plan::coverage(&src, &facts, &reg);
    let tot = cov
        .values()
        .fold([0; 3], |a, c| [a[0] + c[0], a[1] + c[1], a[2] + c[2]]);
    println!(
        "registry v{} coverage [WHEN true, WHEN not true, unregistered] {tot:?}",
        reg.version
    );
    for op in OPS {
        println!("  {op:>10} {:?}", cov.get(*op));
    }
    println!("foldable by dependency but not folded:");
    for (why, c) in by {
        println!("  {c:>5}  {why}");
    }
}
