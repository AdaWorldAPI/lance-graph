//! Triage report for any ONNX model: which nodes the registry does not bind,
//! why, where the frontend (Tract) handles each op, and which native word
//! could compute it.
//!
//! `cargo run --release --example onnx_triage -- <model.onnx> [seq]`
//!
//! Tract and ndarray sources are read from the cargo git checkouts named by
//! this crate's `Cargo.lock`, so the report describes the versions the probe
//! builds with. Native words come from `data/native_words.tsv`; a word whose
//! `fn` is not found in that source is marked `unverified`.
use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};

use tract_harvest_probe::registry::registry;
use tract_harvest_probe::{dr3, fold, source::SourceInventory, tract_index, triage};
use tract_onnx::prelude::*;

fn lock_rev(lock: &str, url_part: &str) -> Option<String> {
    lock.lines()
        .filter(|l| l.starts_with("source = \"git+") && l.contains(url_part))
        .find_map(|l| {
            l.rsplit('#')
                .next()
                .map(|r| r.trim_end_matches('"').to_string())
        })
}

fn checkout(prefix: &str, rev: &str) -> Option<PathBuf> {
    let home = std::env::var("CARGO_HOME")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from(std::env::var("HOME").unwrap()).join(".cargo"));
    let base = home.join("git/checkouts");
    std::fs::read_dir(&base).ok()?.flatten().find_map(|d| {
        let name = d.file_name().to_string_lossy().to_string();
        if !name.starts_with(prefix) {
            return None;
        }
        let p = d.path().join(&rev[..7]);
        p.exists().then_some(p)
    })
}

fn rs_files(dir: &Path, out: &mut Vec<(String, String)>) {
    let Ok(rd) = std::fs::read_dir(dir) else {
        return;
    };
    let mut entries: Vec<_> = rd.flatten().map(|e| e.path()).collect();
    entries.sort();
    for p in entries {
        if p.is_dir() {
            rs_files(&p, out);
        } else if p.extension().is_some_and(|e| e == "rs") {
            if let Ok(t) = std::fs::read_to_string(&p) {
                out.push((p.to_string_lossy().to_string(), t));
            }
        }
    }
}

fn main() {
    let path = std::env::args().nth(1).expect("model path");
    let mut src = SourceInventory::from_bytes(&std::fs::read(&path).unwrap()).unwrap();
    let dir = Path::new(&path).parent();
    let manifest = dir
        .and_then(|d| d.parent())
        .map(|g| g.join("jina_constants.json"));
    if let (Some(dir), Some(m)) = (dir, manifest.filter(|m| m.exists())) {
        src.inline_constants_from_manifest(&m, dir).unwrap();
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
    let fs = dr3::tract_fact_sources_path(Path::new(&path), &pins).unwrap();
    let facts = fold::with_initializer_facts(&src, &fs.merged());
    let reg = registry();
    let folded = fold::fold(&src, &facts, &reg);
    let constants = triage::constant_names(&src, folded.values.keys().map(String::as_str));

    let crate_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
    let lock = std::fs::read_to_string(crate_dir.join("Cargo.lock")).unwrap_or_default();
    let tract_dir = lock_rev(&lock, "/tract").and_then(|r| checkout("tract-", &r));
    let nd_dir = lock_rev(&lock, "/ndarray").and_then(|r| checkout("ndarray-", &r));
    let mut tract_files = Vec::new();
    if let Some(d) = &tract_dir {
        rs_files(&d.join("onnx/src"), &mut tract_files);
    }
    let tix = tract_index::index(&tract_files);
    let frontend: BTreeSet<String> = tix.keys().cloned().collect();

    let hints = triage::parse_hints(
        &std::fs::read_to_string(crate_dir.join("data/native_words.tsv")).unwrap(),
    )
    .unwrap();
    let mut nd_files = Vec::new();
    if let Some(d) = &nd_dir {
        rs_files(&d.join("src"), &mut nd_files);
    }
    let mut own = Vec::new();
    rs_files(&crate_dir.join("src"), &mut own);
    let verified: Vec<bool> = hints
        .iter()
        .map(|h| {
            let leaf = h.word.rsplit("::").next().unwrap();
            let pat = format!("fn {leaf}");
            let pool = if h.word.starts_with("ndarray::") {
                &nd_files
            } else {
                &own
            };
            pool.iter().any(|(_, t)| {
                t.match_indices(&pat)
                    .any(|(i, _)| t[i + pat.len()..].starts_with(['(', '<']))
            })
        })
        .collect();

    let findings = triage::triage(&src, &facts, &constants, &reg, &frontend, &hints);
    let groups = triage::groups(&findings);

    let ops_in_graph: BTreeSet<&str> = src.nodes.iter().map(|n| n.op_type.as_str()).collect();
    let unbound_ops: BTreeSet<&str> = findings.iter().map(|f| f.op_type.as_str()).collect();
    println!("# ONNX triage: {path}\n");
    println!(
        "- sources: tract `{}`, ndarray `{}`",
        tract_dir
            .as_ref()
            .map(|p| p.display().to_string())
            .unwrap_or("not found".into()),
        nd_dir
            .as_ref()
            .map(|p| p.display().to_string())
            .unwrap_or("not found".into())
    );
    println!(
        "- nodes: {} total, {} unbound",
        src.nodes.len(),
        findings.len()
    );
    println!(
        "- op types in graph: {}; with an unbound node: {}; of those registered by Tract: {}",
        ops_in_graph.len(),
        unbound_ops.len(),
        unbound_ops
            .iter()
            .filter(|o| frontend.contains(**o))
            .count()
    );
    let mut by_kind: BTreeMap<&str, usize> = BTreeMap::new();
    for f in &findings {
        *by_kind.entry(f.gap.kind()).or_default() += 1;
    }
    println!("- unbound nodes by gap: {by_kind:?}");
    let hinted = findings
        .iter()
        .filter(|f| f.hints.iter().any(|&h| verified[h]))
        .count();
    println!("- unbound nodes with a verified native word: {hinted}\n");

    println!("| nodes | op | value dtype | gap | Tract parser (attrs = default) | native words | roles |");
    println!("|---|---|---|---|---|---|---|");
    for g in &groups {
        let gap = match &g.gap {
            triage::Gap::Dtype { registered } => format!("dtype (registered: {registered:?})"),
            other => other.kind().to_string(),
        };
        let tract = match tix.get(&g.op_type) {
            None => "not registered".into(),
            Some(e) => {
                let file = e.file.rsplit("/onnx/src/").next().unwrap_or(&e.file);
                let attrs: Vec<String> = e
                    .attrs
                    .iter()
                    .map(|a| match &a.default {
                        Some(d) => format!("{}={d}", a.name),
                        None if a.required => format!("{} (required)", a.name),
                        None => a.name.clone(),
                    })
                    .collect();
                format!(
                    "{file}:{} {}{}{}",
                    e.line,
                    e.parser.as_deref().unwrap_or("closure"),
                    if attrs.is_empty() {
                        String::new()
                    } else {
                        format!(" [{}]", attrs.join(", "))
                    },
                    if e.opset_dependent {
                        " opset-dependent"
                    } else {
                        ""
                    }
                )
            }
        };
        let words: Vec<String> = g
            .hints
            .iter()
            .map(|&h| {
                format!(
                    "`{}` ({}, {}){}",
                    hints[h].word.rsplit("::").next().unwrap(),
                    hints[h].template,
                    hints[h].numeric,
                    if verified[h] { "" } else { " unverified" }
                )
            })
            .collect();
        let roles: Vec<String> = g.roles.iter().map(|(r, c)| format!("{r} ×{c}")).collect();
        println!(
            "| {} | {} | {} | {} | {} | {} | {} |",
            g.nodes,
            g.op_type,
            g.value_dtype,
            gap,
            tract,
            if words.is_empty() {
                "—".into()
            } else {
                words.join("; ")
            },
            roles.join("; ")
        );
    }
}
