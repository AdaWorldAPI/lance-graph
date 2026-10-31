//! Generate registry fields from a model's triage, the native-word table and
//! Tract's op index, for the templates that need no judgement.
//!
//! `cargo run --release --example registry_codegen -- <model.onnx> [seq] > src/generated.rs`
//!
//! Generated (one field per `(op, value dtype)` group):
//! - `unary_view`: an elementwise `fn(ArrayView, ArrayViewMut)` word, when
//!   Tract's parser reads no attribute;
//! - `mask_not`: the complement of a "not X" mask word, same condition;
//! - `dtype_row`: an existing field's signature, guard and DO at one more
//!   dtype (the DO must already handle it: a test proves it).
//!
//! Everything else is listed on stderr as manual, with the reason. The output
//! is deterministic for a given model, table, registry and Tract source.
use std::collections::BTreeSet;
use std::fmt::Write as _;
use std::path::Path;

use tract_harvest_probe::registry::{hand_written, AttrRule, Numeric};
use tract_harvest_probe::{dr3, fold, source::SourceInventory, tract_index, triage};
use tract_onnx::prelude::*;

#[path = "support/sources.rs"]
mod sources;

fn dt_ident(d: &str) -> &'static str {
    match d {
        "F32" => "F32",
        "F64" => "F64",
        "I32" => "I32",
        "I64" => "I64",
        "Bool" => "Bool",
        _ => "?",
    }
}

fn attrs_literal(rules: &[AttrRule]) -> String {
    if rules.is_empty() {
        return "NO_ATTRS".into();
    }
    let items: Vec<String> = rules
        .iter()
        .map(|r| {
            format!(
                "AttrRule {{ name: {:?}, value: {:?}, default: {:?} }}",
                r.name, r.value, r.default
            )
        })
        .collect();
    format!("&[{}]", items.join(", "))
}

fn numeric(n: &str, word: &str) -> String {
    match n {
        "exact" => "Numeric::Exact".into(),
        _ => format!("Numeric::Bounded(\"{word}; generated, ULP against ORT measured in tests\")"),
    }
}

fn main() {
    let path = std::env::args().nth(1).expect("model path");
    let seq = std::env::args().nth(2);
    let mut src = SourceInventory::from_bytes(&std::fs::read(&path).unwrap()).unwrap();
    sources::inline_matching_constants(&mut src, Path::new(&path));
    let pins: Vec<_> = seq
        .as_deref()
        .and_then(|s| s.parse::<usize>().ok())
        .map(|s| {
            (0..src.inputs.len())
                .map(|i| (i, i64::datum_type(), vec![1, s]))
                .collect()
        })
        .unwrap_or_default();
    let fs = dr3::tract_fact_sources_path(Path::new(&path), &pins).unwrap();
    let facts = fold::with_initializer_facts(&src, &fs.merged());
    let reg = hand_written();
    let folded = fold::fold(&src, &facts, &reg);
    let constants = triage::constant_names(&src, folded.values.keys().map(String::as_str));
    let crate_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
    let (tract_files, _nd) = sources::load(crate_dir);
    let tix = tract_index::index(&tract_files);
    let frontend: BTreeSet<String> = tix.keys().cloned().collect();
    let hints = triage::parse_hints(
        &std::fs::read_to_string(crate_dir.join("data/native_words.tsv")).unwrap(),
    )
    .unwrap();
    let findings = triage::triage(&src, &facts, &constants, &reg, &frontend, &hints);
    let groups = triage::groups(&findings);

    let mut drafts = String::new();
    let mut fns = String::new();
    for g in &groups {
        let manual = |why: &str| {
            eprintln!(
                "manual: {} {} ({} nodes): {why}",
                g.op_type, g.value_dtype, g.nodes
            )
        };
        let Some(&h) = g.hints.first() else {
            manual("no native word in data/native_words.tsv");
            continue;
        };
        let hint = &hints[h];
        let Some(entry) = tix.get(&g.op_type) else {
            manual("Tract does not register the op");
            continue;
        };
        let dt = dt_ident(&g.value_dtype);
        let lower = g.op_type.to_lowercase();
        match hint.template.as_str() {
            "unary_view" | "mask_not" if entry.opset_dependent || !entry.attrs.is_empty() => {
                manual("Tract's parser reads attributes or branches on the opset")
            }
            "unary_view" if dt == "F32" => {
                let f = format!("do_gen_{lower}_f32");
                writeln!(
                    fns,
                    "/// `{op}` f32: `{w}` elementwise.\nfn {f}(a: &[&Value], n: &SourceNode) -> Result<Vec<Value>, Refusal> {{\n    let x = f32_arg(a, 0, n)?;\n    let mut out = ArrayD::<f32>::zeros(x.raw_dim());\n    {w}(x.view(), out.view_mut());\n    Ok(vec![Value::F32(out)])\n}}\n",
                    op = g.op_type,
                    w = hint.word
                )
                .unwrap();
                writeln!(
                    drafts,
                    "        draft({w:?}, {op:?}, Some(0), DType::F32, NO_ATTRS, guard_true, \"guard_true\", {f}, {num}),",
                    w = hint.word,
                    op = g.op_type,
                    num = numeric(&hint.numeric, &hint.word)
                )
                .unwrap();
            }
            "mask_not" if dt == "F32" => {
                let f = format!("do_gen_{lower}_f32");
                writeln!(
                    fns,
                    "/// `{op}` f32: the complement of `{w}` over the valid bits.\nfn {f}(a: &[&Value], n: &SourceNode) -> Result<Vec<Value>, Refusal> {{\n    let x = f32_arg(a, 0, n)?;\n    let xs: Vec<f32> = x.iter().copied().collect();\n    let mut words = vec![0u64; lance_graph_onnx::mask_words(xs.len())];\n    {w}(&xs, &mut words);\n    for w in words.iter_mut() {{\n        *w = !*w;\n    }}\n    if !xs.len().is_multiple_of(64) {{\n        if let Some(last) = words.last_mut() {{\n            *last &= (1u64 << (xs.len() % 64)) - 1;\n        }}\n    }}\n    Ok(vec![Value::Bool {{ shape: x.shape().to_vec(), words }}])\n}}\n",
                    op = g.op_type,
                    w = hint.word
                )
                .unwrap();
                writeln!(
                    drafts,
                    "        draft({w:?}, {op:?}, Some(0), DType::F32, NO_ATTRS, guard_true, \"guard_true\", {f}, {num}),",
                    w = hint.word,
                    op = g.op_type,
                    num = numeric(&hint.numeric, &hint.word)
                )
                .unwrap();
            }
            "dtype_row" => {
                let triage::Gap::Dtype { .. } = g.gap else {
                    manual("dtype_row hint on a group that is not a dtype gap");
                    continue;
                };
                // The existing field this row extends: same op, its DO named
                // by the hint.
                let leaf = hint.word.rsplit("::").next().unwrap();
                let Some(base) = reg
                    .regs
                    .iter()
                    .find(|r| r.op_type == g.op_type && fn_name_of(&reg, r) == leaf)
                else {
                    manual("no existing field with the hinted DO");
                    continue;
                };
                writeln!(
                    drafts,
                    "        draft({name:?}, {op:?}, {arg:?}, DType::{dt}, {attrs}, {guard}, {guard:?}, {leaf}, {num}),",
                    name = reg.method(base).name,
                    op = g.op_type,
                    arg = base.dtype_arg,
                    attrs = attrs_literal(base.attrs),
                    guard = base.guard,
                    num = match reg.method(base).numeric {
                        Numeric::Exact => "Numeric::Exact".to_string(),
                        Numeric::Bounded(b) => format!("Numeric::Bounded({b:?})"),
                    }
                )
                .unwrap();
            }
            t => manual(&format!("template {t} is written by hand")),
        }
    }
    println!("//! Generated by `examples/registry_codegen.rs`; do not edit by hand.");
    println!("//!");
    println!(
        "//! Inputs: the triage of `{}` (seq {}), `data/native_words.tsv`, the",
        path,
        seq.as_deref().unwrap_or("symbolic")
    );
    println!("//! registry, and Tract's op index. Re-run the generator to change it.");
    println!();
    println!("use ndarray::ArrayD;");
    println!();
    println!("use crate::registry::*;");
    println!("use crate::source::SourceNode;");
    println!("use crate::value::{{DType, Value}};");
    println!();
    println!("#[allow(clippy::too_many_arguments)]");
    println!("fn draft(name: &'static str, op_type: &'static str, dtype_arg: Option<usize>, dtype: DType, attrs: &'static [AttrRule], when: Guard, guard: &'static str, do_: Polyfill, numeric: Numeric) -> (Method, Registration) {{");
    println!("    (Method {{ name, do_, numeric }}, Registration {{ domain: \"ai.onnx\", op_type, dtype_arg, dtype, attrs, opsets: OPSETS, when, guard, method: 0 }})");
    println!("}}");
    println!();
    println!("/// The generated fields, appended to the hand-written table.");
    println!("pub(crate) fn drafts() -> Vec<(Method, Registration)> {{");
    println!("    vec![");
    print!("{drafts}");
    println!("    ]");
    println!("}}");
    println!();
    print!("{fns}");
}

/// The DO's function name, read from the registration table's own source
/// (`Registry` keeps fn pointers, not names): the `reg!` block whose method
/// name, op type and dtype match.
fn fn_name_of(
    reg: &tract_harvest_probe::registry::Registry,
    r: &tract_harvest_probe::registry::Registration,
) -> String {
    let src = include_str!("../src/registry.rs");
    let name = reg.method(r).name;
    let needle = format!("{name:?},\n                {:?},", r.op_type);
    let dt = format!("{:?},", r.dtype);
    src.match_indices(&needle)
        .find_map(|(i, _)| {
            let block: Vec<&str> = src[i..].lines().take(9).map(str::trim).collect();
            (block.get(3) == Some(&dt.as_str())).then(|| {
                block
                    .get(6)
                    .unwrap_or(&"")
                    .trim_end_matches(',')
                    .to_string()
            })
        })
        .unwrap_or_default()
}
