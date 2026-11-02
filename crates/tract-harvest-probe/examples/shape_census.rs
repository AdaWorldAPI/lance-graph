//! Dependency census of the shape-plumbing ops, and what folds.
//!
//! `cargo run --release --example shape_census -- fixtures/jina/onnx_model.onnx [seq]`
//!
//! Without `seq` the inputs keep their symbolic dims; with it they are pinned
//! to `[1, seq]`.
use std::time::Instant;
use tract_harvest_probe::{dr3, fold, registry, source::SourceInventory};
use tract_onnx::prelude::*;

const OPS: &[&str] = &[
    "Unsqueeze",
    "Shape",
    "Gather",
    "Concat",
    "Reshape",
    "Expand",
];

fn main() {
    let path = std::env::args().nth(1).expect("model path");
    let mut src = SourceInventory::from_bytes(&std::fs::read(&path).unwrap()).unwrap();
    // External-data Constants (the rotary inv_freq), when fetched by
    // tools/fetch_jina_constants.py.
    // The manifest sits next to the model's directory; a path without that
    // ancestor (a bare filename, a file directly under `/`) has none.
    let dir = std::path::Path::new(&path).parent();
    let manifest = dir
        .and_then(|d| d.parent())
        .map(|g| g.join("jina_constants.json"));
    if let (Some(dir), Some(manifest)) = (dir, manifest.filter(|m| m.exists())) {
        match src.inline_constants_from_manifest(&manifest, dir) {
            Ok(k) => println!("inlined {k} external constants"),
            Err(e) => println!("external constants not inlined: {e}"),
        }
    }
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
    let not_true: Vec<_> = cov
        .iter()
        .filter(|(_, c)| c[1] > 0)
        .map(|(k, c)| (k.as_str(), c[1]))
        .collect();
    println!("  WHEN not true by op: {not_true:?}");
    println!("foldable by dependency but not folded:");
    for (why, c) in by {
        println!("  {c:>5}  {why}");
    }

    // The Reshapes whose WHEN is not TRUE: which unregistered ops their
    // target depends on (the first unregistered or unfolded-unregistered
    // ancestors along the target input).
    let producer: std::collections::HashMap<&str, usize> = src
        .nodes
        .iter()
        .enumerate()
        .flat_map(|(i, n)| n.outputs.iter().map(move |o| (o.as_str(), i)))
        .collect();
    let mut consts: std::collections::HashMap<&str, &tract_harvest_probe::value::Value> = src
        .initializers
        .iter()
        .filter_map(|i| i.value.as_ref().map(|v| (i.name.as_str(), v)))
        .collect();
    for (k, v) in &f.values {
        consts.insert(k.as_str(), v);
    }
    let mut blocked_reshape = 0;
    let mut roots = std::collections::BTreeMap::<String, usize>::new();
    for n in src.nodes.iter().filter(|n| n.op_type == "Reshape") {
        let args: Vec<_> = n.inputs.iter().map(|i| facts.get(i)).collect();
        let Some(r) = reg.lookup(n, &args) else {
            continue;
        };
        let c = n
            .inputs
            .iter()
            .map(|i| consts.get(i.as_str()).copied())
            .collect();
        if (r.when)(&registry::NodeView {
            node: n,
            args,
            consts: c,
        }) == registry::K3::True
        {
            continue;
        }
        blocked_reshape += 1;
        let mut stack = vec![n.inputs[1].as_str()];
        let mut seen = std::collections::HashSet::new();
        let mut found = std::collections::BTreeSet::new();
        while let Some(v) = stack.pop() {
            if consts.contains_key(v) || !seen.insert(v) {
                continue;
            }
            let Some(&p) = producer.get(v) else { continue };
            let pn = &src.nodes[p];
            let pargs: Vec<_> = pn.inputs.iter().map(|i| facts.get(i)).collect();
            if reg.lookup(pn, &pargs).is_none() {
                found.insert(format!(
                    "{}({:?})",
                    pn.op_type,
                    registry::signature_dtype(pn, &pargs, Some(0)).unwrap()
                ));
            } else {
                stack.extend(pn.inputs.iter().map(String::as_str));
            }
        }
        *roots
            .entry(found.into_iter().collect::<Vec<_>>().join(" + "))
            .or_default() += 1;
    }
    println!("Reshape with WHEN not TRUE: {blocked_reshape}; unregistered ops their target depends on: {roots:?}");
    // Every shape-only blocked node, traced upstream through blocked
    // producers to the first node blocked for its own reason (unregistered or
    // WHEN not TRUE): the producer/guard that unblocks it.
    let why: std::collections::HashMap<usize, &str> =
        f.blocked.iter().map(|(i, w)| (*i, w.as_str())).collect();
    let mut first = std::collections::BTreeMap::<(String, String), usize>::new();
    for &i in why.keys() {
        let mut roots = std::collections::BTreeSet::new();
        let mut stack = vec![i];
        let mut seen = std::collections::HashSet::new();
        while let Some(j) = stack.pop() {
            if !seen.insert(j) {
                continue;
            }
            match why.get(&j) {
                Some(r) if !r.ends_with("an input is not folded upstream") => {
                    roots.insert(format!("{} [{}]", src.nodes[j].name, r));
                }
                Some(_) => {
                    for inp in &src.nodes[j].inputs {
                        if inp.is_empty() || consts.contains_key(inp.as_str()) {
                            continue;
                        }
                        match producer.get(inp.as_str()) {
                            Some(&p) if why.contains_key(&p) => stack.push(p),
                            Some(&p) => {
                                roots.insert(format!(
                                    "{} {} [not folded, {:?}]",
                                    src.nodes[p].op_type, src.nodes[p].name, node_dep[p]
                                ));
                            }
                            None => {
                                roots.insert(format!("{inp} [graph input]"));
                            }
                        }
                    }
                }
                None => {}
            }
        }
        *first
            .entry((
                src.nodes[i].op_type.clone(),
                roots.into_iter().collect::<Vec<_>>().join(" + "),
            ))
            .or_default() += 1;
    }
    println!("shape-only blocked, by op -> first unresolved producer/guard:");
    for ((op, r), c) in &first {
        println!("  {c:>3}  {op:<12} <- {r}");
    }
    // Not folded: runtime (reads an input value) vs shape-only (blocked).
    let mut runtime = std::collections::BTreeMap::<&str, usize>::new();
    let mut shape_only = std::collections::BTreeMap::<&str, usize>::new();
    for (i, n) in src.nodes.iter().enumerate() {
        if f.nodes.contains(&i) {
            continue;
        }
        let m = if node_dep[i] >= fold::Dep::Symbolic {
            &mut runtime
        } else {
            &mut shape_only
        };
        *m.entry(n.op_type.as_str()).or_default() += 1;
    }
    fn top<'a>(m: &std::collections::BTreeMap<&'a str, usize>) -> Vec<(&'a str, usize)> {
        let mut v: Vec<_> = m.iter().map(|(k, c)| (*k, *c)).collect();
        v.sort_by_key(|a| std::cmp::Reverse(a.1));
        v
    }
    println!(
        "not folded: runtime {} nodes {:?}",
        runtime.values().sum::<usize>(),
        top(&runtime)
    );
    println!(
        "not folded: shape-only (blocked) {} nodes {:?}",
        shape_only.values().sum::<usize>(),
        top(&shape_only)
    );
}
