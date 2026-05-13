//! D-PARITY-V2-10 — DTO classification CI gate.
//! classification: bare-metal
//!
//! Walks workspace member `src/**/*.rs`, parses with `syn`, finds every
//! `pub struct`/`pub enum` matching the eight DTO suffixes, asserts each
//! carries a `// classification: bare-metal | soa-glue | bridge-projection`
//! comment matching `soa-dto-dependency-ledger.md`'s 22-row table. Exit 0 if
//! all classified; exit 1 + error list otherwise.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use syn::Item;
use walkdir::WalkDir;

const SUFFIXES: &[&str] = &[
    "Dto", "Row", "Filter", "Step", "Slot", "Bridge", "Intent", "Event",
];

/// 22-row ledger map per soa-dto-dependency-ledger.md (2026-05-07).
const LEDGER: &[(&str, &str)] = &[
    ("ShaderEvent", "bare-metal"),
    ("UnifiedStep", "soa-glue"),
    ("WorldModelDto", "soa-glue"),
    ("WorldMapDto", "bridge-projection"),
    ("MetaFilter", "bare-metal"),
    ("CommitFilter", "bare-metal"),
    ("TekamoloSlot", "soa-glue"),
    ("SlotPrior", "soa-glue"),
    ("SlotPriorDelta", "soa-glue"),
    ("ExternalIntent", "bare-metal"),
    ("CognitiveEventRow", "bare-metal"),
    ("OntologyDto", "bridge-projection"),
    ("EntityTypeDto", "bridge-projection"),
    ("PropertyDto", "bridge-projection"),
    ("LinkTypeDto", "bridge-projection"),
    ("ActionTypeDto", "bridge-projection"),
    ("DriftEvent", "bridge-projection"),
    ("MappingRow", "bridge-projection"),
    ("OgitBridge", "bridge-projection"),
    ("MedcareBridge", "bridge-projection"),
    ("WoaBridge", "bridge-projection"),
];

fn matches_suffix(n: &str) -> bool {
    SUFFIXES.iter().any(|s| n.ends_with(s) && n.len() > s.len())
}

/// Parse `// classification: <v>` (or rustdoc form).
fn parse_class(line: &str) -> Option<String> {
    let rest = line
        .trim()
        .trim_start_matches('/')
        .trim()
        .to_ascii_lowercase();
    let v = rest.strip_prefix("classification:")?.trim().to_string();
    matches!(v.as_str(), "bare-metal" | "soa-glue" | "bridge-projection").then_some(v)
}

/// 1-based line of `struct <n>` / `enum <n>` in source.
fn decl_line(source: &str, n: &str) -> usize {
    let (s, e) = (format!("struct {n}"), format!("enum {n}"));
    source
        .lines()
        .enumerate()
        .find(|(_, l)| l.contains(&s) || l.contains(&e))
        .map(|(i, _)| i + 1)
        .unwrap_or(0)
}

struct Finding {
    name: String,
    file: PathBuf,
    line: usize,
    actual: Option<String>,
    expected: Option<String>,
}

fn scan_file(path: &Path, ledger: &HashMap<&str, &str>, out: &mut Vec<Finding>) {
    let Ok(src) = std::fs::read_to_string(path) else {
        return;
    };
    let Ok(file) = syn::parse_file(&src) else {
        return;
    };
    for item in &file.items {
        let name = match item {
            Item::Struct(s) => s.ident.to_string(),
            Item::Enum(e) => e.ident.to_string(),
            _ => continue,
        };
        if !matches_suffix(&name) {
            continue;
        }
        let line = decl_line(&src, &name);
        let lines: Vec<&str> = src.lines().collect();
        let start = line.saturating_sub(8);
        let actual = lines[start..line.min(lines.len())]
            .iter()
            .find_map(|l| parse_class(l));
        out.push(Finding {
            actual,
            expected: ledger.get(name.as_str()).map(|s| s.to_string()),
            name,
            file: path.to_path_buf(),
            line,
        });
    }
}

fn workspace_root() -> PathBuf {
    if let Ok(r) = std::env::var("DTO_CHECK_ROOT") {
        return PathBuf::from(r);
    }
    let m = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    m.parent()
        .and_then(|p| p.parent())
        .map(|p| p.to_path_buf())
        .unwrap_or(m)
}

/// Parse `members = [...]` from workspace Cargo.toml.
fn member_dirs(root: &Path) -> Vec<PathBuf> {
    let toml = std::fs::read_to_string(root.join("Cargo.toml")).unwrap_or_default();
    let (mut out, mut inside) = (Vec::new(), false);
    for line in toml.lines() {
        let t = line.trim();
        if t.starts_with("members") && t.contains('[') {
            inside = true;
            continue;
        }
        if inside {
            if t.starts_with(']') {
                break;
            }
            let q = t.trim_end_matches(',').trim();
            if q.len() > 2 && q.starts_with('"') && q.ends_with('"') {
                out.push(root.join(&q[1..q.len() - 1]));
            }
        }
    }
    out
}

fn main() {
    let root = workspace_root();
    let ledger: HashMap<&str, &str> = LEDGER.iter().copied().collect();
    let mut findings = Vec::new();
    for member in member_dirs(&root) {
        let src = member.join("src");
        if !src.exists() {
            continue;
        }
        for entry in WalkDir::new(&src).into_iter().filter_map(Result::ok) {
            let p = entry.path();
            if p.is_file() && p.extension().is_some_and(|e| e == "rs") {
                scan_file(p, &ledger, &mut findings);
            }
        }
    }

    let (mut ok, mut fail, mut errs) = (0usize, 0usize, Vec::<String>::new());
    for f in &findings {
        let loc = format!("{}:{}", f.file.display(), f.line);
        match (&f.actual, &f.expected) {
            (Some(a), Some(e)) if a == e => {
                println!("OK {} [{a}] {loc}", f.name);
                ok += 1;
            }
            (Some(a), Some(e)) => {
                errs.push(format!(
                    "FAIL: {} in {loc} classification {a} disagrees with ledger {e}",
                    f.name
                ));
                fail += 1;
            }
            (Some(a), None) => {
                println!("OK {} [{a}] {loc} (not in ledger)", f.name);
                ok += 1;
            }
            (None, _) => {
                errs.push(format!("FAIL: {} in {loc} missing classification", f.name));
                fail += 1;
            }
        }
    }
    println!("---");
    println!("scanned: {} types; ok: {ok}; fail: {fail}", findings.len());
    for e in &errs {
        eprintln!("{e}");
    }
    if fail > 0 {
        std::process::exit(1);
    }
}
