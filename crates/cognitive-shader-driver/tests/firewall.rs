//! CausalEdge64 firewall CI lint (C6) — the substrate-twin bar with teeth.
//!
//! `cognitive-shader-driver` must use exactly ONE `CausalEdge64`: the
//! `causal_edge::{edge::,}CausalEdge64` re-export. Two layout-incompatible
//! twins exist in the workspace and must NEVER be imported into this crate's
//! `src/`:
//!   * `ndarray::hpc::causal_diff::CausalEdge64`
//!   * `thinking_engine::layered::CausalEdge64`
//!
//! This test walks `src/` (std::fs, no external deps) and FAILS if either twin
//! path — or an aliased `... as` form of it — appears. The legitimate
//! `#[repr(transparent)]` reinterpret cast sites in `mailbox_soa.rs`
//! (`edges_raw` / `meta_raw`) reference only `causal_edge::CausalEdge64`, so
//! they are not matched; the test is scoped to the twin module paths, not the
//! word `CausalEdge64` itself.
//!
//! Source is CLEAN today (no twin imports). This gate keeps it that way as the
//! migration adds mailbox edge plumbing.

use std::fs;
use std::path::{Path, PathBuf};

/// The two forbidden twin module paths, tolerant of whitespace around `::`.
/// Matching is substring-based after whitespace normalization, so it also
/// catches `use ndarray::hpc::causal_diff::CausalEdge64 as Foo;` aliases.
const FORBIDDEN_TWINS: &[&str] = &[
    "ndarray::hpc::causal_diff::CausalEdge64",
    "thinking_engine::layered::CausalEdge64",
];

fn collect_rs_files(dir: &Path, out: &mut Vec<PathBuf>) {
    let entries = match fs::read_dir(dir) {
        Ok(e) => e,
        Err(_) => return,
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            collect_rs_files(&path, out);
        } else if path.extension().and_then(|e| e.to_str()) == Some("rs") {
            out.push(path);
        }
    }
}

/// Normalize whitespace around `::` so `causal_diff :: CausalEdge64` is caught.
fn normalize(src: &str) -> String {
    let mut s = src.to_string();
    // Collapse any spacing around `::` to the bare `::` token.
    while s.contains(" ::") {
        s = s.replace(" ::", "::");
    }
    while s.contains(":: ") {
        s = s.replace(":: ", "::");
    }
    s
}

#[test]
fn no_substrate_twin_causaledge64_in_src() {
    // The crate's own src/ directory (resolved relative to CARGO_MANIFEST_DIR
    // so the test is location-independent).
    let manifest_dir = env!("CARGO_MANIFEST_DIR");
    let src_dir = Path::new(manifest_dir).join("src");
    assert!(
        src_dir.is_dir(),
        "firewall: src dir not found at {}",
        src_dir.display()
    );

    let mut files = Vec::new();
    collect_rs_files(&src_dir, &mut files);
    assert!(
        !files.is_empty(),
        "firewall: walked src/ but found no .rs files — the walk is broken"
    );

    let mut violations: Vec<String> = Vec::new();
    for file in &files {
        let raw = match fs::read_to_string(file) {
            Ok(c) => c,
            Err(_) => continue,
        };
        let norm = normalize(&raw);
        for twin in FORBIDDEN_TWINS {
            if norm.contains(twin) {
                violations.push(format!(
                    "{} imports forbidden twin `{}`",
                    file.display(),
                    twin
                ));
            }
        }
    }

    assert!(
        violations.is_empty(),
        "CausalEdge64 firewall BLOCK — cognitive-shader-driver must use only \
         `causal_edge::CausalEdge64`, never a substrate twin:\n  {}",
        violations.join("\n  ")
    );
}

/// Meta-sanity: the firewall's own matcher must actually fire on a planted
/// twin string (guards against a normalize() bug silently disabling the gate).
#[test]
fn firewall_matcher_detects_planted_twin() {
    let planted = "use ndarray :: hpc :: causal_diff :: CausalEdge64;";
    let norm = normalize(planted);
    assert!(
        FORBIDDEN_TWINS.iter().any(|t| norm.contains(t)),
        "firewall matcher failed to detect a planted (whitespaced) twin import \
         — the lint would not catch a real violation"
    );
    // And the legitimate path must NOT trip the matcher.
    let ok = normalize("use causal_edge::edge::CausalEdge64;");
    assert!(
        !FORBIDDEN_TWINS.iter().any(|t| ok.contains(t)),
        "firewall matcher false-positives on the canonical causal_edge path"
    );
}
