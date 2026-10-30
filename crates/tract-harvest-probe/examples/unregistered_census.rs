//! Every node the registry does not bind, with what a registration would key on.
//!
//! `cargo run --release --example unregistered_census -- fixtures/jina/onnx_model.onnx [seq]`
//!
//! One tab-separated line per unregistered node:
//! `node  op  signature  near_miss  inputs  output`, where `near_miss` names the
//! registered op types with the same `op_type` (so the gap is a dtype or
//! attribute, not a missing op), `inputs` is `dtype[shape]` per argument with
//! `=c` for a constant, and `output` is the first output's fact.
use std::collections::{BTreeMap, HashSet};
use tract_harvest_probe::registry::{registry, render_signature, signature_dtype, Dim, Fact};
use tract_harvest_probe::{dr3, fold, source::SourceInventory};
use tract_onnx::prelude::*;

#[path = "support/sources.rs"]
mod sources;

fn fact(f: Option<&Fact>) -> String {
    let Some(f) = f else { return "?".into() };
    let dt = f.dtype.map(|d| format!("{d:?}")).unwrap_or("?".into());
    let sh = match &f.shape {
        None => "?".into(),
        Some(s) => s
            .iter()
            .map(|d| match d {
                Dim::Known(k) => k.to_string(),
                Dim::Symbol(s) => s.clone(),
            })
            .collect::<Vec<_>>()
            .join(","),
    };
    format!("{dt}[{sh}]")
}

fn main() {
    let path = std::env::args().nth(1).expect("model path");
    let mut src = SourceInventory::from_bytes(&std::fs::read(&path).unwrap()).unwrap();
    sources::inline_matching_constants(&mut src, std::path::Path::new(&path));
    let pins: Vec<_> = std::env::args()
        .nth(2)
        .and_then(|s| s.parse::<usize>().ok())
        .map(|seq| {
            (0..src.inputs.len())
                .map(|i| (i, i64::datum_type(), vec![1, seq]))
                .collect()
        })
        .unwrap_or_default();
    let fs = dr3::tract_fact_sources_path(std::path::Path::new(&path), &pins).unwrap();
    let facts = fold::with_initializer_facts(&src, &fs.merged());
    let reg = registry();
    let folded = fold::fold(&src, &facts, &reg);
    let mut consts: HashSet<&str> = src
        .initializers
        .iter()
        .filter(|i| i.value.is_some())
        .map(|i| i.name.as_str())
        .collect();
    for n in &src.nodes {
        if n.op_type == "Constant" {
            consts.extend(n.outputs.iter().map(String::as_str));
        }
    }
    consts.extend(folded.values.keys().map(String::as_str));
    let mut per_sig: BTreeMap<String, usize> = BTreeMap::new();
    for node in &src.nodes {
        let args: Vec<_> = node.inputs.iter().map(|i| facts.get(i)).collect();
        if reg.lookup(node, &args).is_some() {
            continue;
        }
        let sig = render_signature(node, signature_dtype(node, &args, Some(0)));
        *per_sig.entry(sig.clone()).or_default() += 1;
        let near: Vec<&str> = reg
            .regs
            .iter()
            .filter(|r| r.op_type == node.op_type)
            .map(|r| reg.method(r).name)
            .collect();
        let inputs: Vec<String> = node
            .inputs
            .iter()
            .map(|i| {
                if i.is_empty() {
                    "-".into()
                } else {
                    format!(
                        "{}{}",
                        fact(facts.get(i)),
                        if consts.contains(i.as_str()) {
                            "=c"
                        } else {
                            ""
                        }
                    )
                }
            })
            .collect();
        println!(
            "{}\t{}\t{}\t{}\t{}\t{}",
            node.name,
            node.op_type,
            sig,
            if near.is_empty() {
                "-".into()
            } else {
                near.join("|")
            },
            inputs.join(" "),
            fact(node.outputs.first().and_then(|o| facts.get(o)))
        );
    }
    eprintln!("signatures:");
    for (s, c) in &per_sig {
        eprintln!("{c:5}  {s}");
    }
    eprintln!("total {}", per_sig.values().sum::<usize>());
}
