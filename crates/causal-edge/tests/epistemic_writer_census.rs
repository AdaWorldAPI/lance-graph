//! The production admission boundary for bits 59..63, as a two-way ratchet.
//!
//! `CausalEdge64::with_epistemic_raw5` is the one physical writer of the
//! `EpistemicState5` field. Every call in a `crates/*/src` file must be
//! classified below, by file and enclosing function:
//!
//! - [`AUTHORIZED`]: a production writer, with what licenses it.
//! - [`TEST_FIXTURES`]: a call inside test code that builds an input.
//!
//! The test fails when a call appears that neither table names, and when a
//! table entry no longer matches any call. Adding a writer is therefore a
//! visible change to this file, never a silent second evidence authority.
//!
//! Scope: calls to `with_epistemic_raw5` in `src/`. Examples and `tests/`
//! directories are lab code and are not scanned. A write that bypasses the
//! method (raw `u64` arithmetic on the edge) is not caught here.

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

/// `(file relative to crates/, enclosing fn, why it may write)`.
const AUTHORIZED: &[(&str, &str, &str)] = &[
    (
        "lance-graph-planner/src/pearl.rs",
        "stamp",
        "behind pearl::revise (raise to what an operator earned), pearl::retract (lower to what is still earned) and pearl::hydrate (topology only)",
    ),
    (
        "causal-edge/src/edge_v3.rs",
        "rehydrate",
        "restores the existing code on the V3 -> CE64 lift; mints nothing",
    ),
    (
        "deepnsm-v2/src/wire.rs",
        "wire",
        "text reading: Associated or Related from independent verse support; never CausalCandidate or Causes",
    ),
];

/// `(file relative to crates/, enclosing fn)` of test-only calls.
const TEST_FIXTURES: &[(&str, &str)] = &[
    (
        "causal-edge/src/v2_layout_tests.rs",
        "every_code_round_trips_and_touches_only_59_63",
    ),
    (
        "causal-edge/src/v2_layout_tests.rs",
        "joint_write_replaces_both_halves",
    ),
    (
        "causal-edge/src/v2_layout_tests.rs",
        "the_field_survives_the_le_image",
    ),
    (
        "causal-edge/src/v2_layout_tests.rs",
        "the_topology_writer_is_a_factor_update_and_the_band_writer_is_not_canonical",
    ),
    (
        "causal-edge/src/edge_v3.rs",
        "the_lift_preserves_the_joint_epistemic_code",
    ),
    (
        "lance-graph-planner/src/chain_counterfactual.rs",
        "the_role_reports_the_cut_edges_own_coordinate_not_a_bare_boolean",
    ),
    ("lance-graph-planner/src/nars/scent_learn.rs", "edge"),
    ("lance-graph-planner/src/nars/scent.rs", "edge"),
    ("lance-graph-planner/src/nars/mediator.rs", "edge"),
    (
        "lance-graph-planner/src/nars/warrant.rs",
        "a_judgment_is_never_promoted_by_its_premises",
    ),
];

fn crates_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("crates/")
        .to_path_buf()
}

fn rust_files(dir: &Path, out: &mut Vec<PathBuf>) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    for e in entries.flatten() {
        let p = e.path();
        if p.is_dir() {
            if p.file_name().is_some_and(|n| n == "target") {
                continue;
            }
            rust_files(&p, out);
        } else if p.extension().is_some_and(|x| x == "rs") {
            out.push(p);
        }
    }
}

/// The name in the nearest `fn name` line at or above `line`.
fn enclosing_fn(lines: &[&str], line: usize) -> String {
    for l in lines[..=line].iter().rev() {
        let t = l.trim_start();
        if let Some(i) = t.find("fn ") {
            let before = &t[..i];
            if before.is_empty() || before.ends_with(' ') {
                let rest = &t[i + 3..];
                let name: String = rest
                    .chars()
                    .take_while(|c| c.is_alphanumeric() || *c == '_')
                    .collect();
                if !name.is_empty() {
                    return name;
                }
            }
        }
    }
    String::new()
}

/// Every `(file, enclosing fn)` that calls `with_epistemic_raw5` in a
/// `crates/*/src` file, excluding comments and the method's definitions.
fn call_sites() -> BTreeSet<(String, String)> {
    let root = crates_dir();
    let mut files = Vec::new();
    for krate in std::fs::read_dir(&root).expect("crates/").flatten() {
        rust_files(&krate.path().join("src"), &mut files);
    }
    let mut out = BTreeSet::new();
    for f in files {
        let text = std::fs::read_to_string(&f).expect("read source");
        let lines: Vec<&str> = text.lines().collect();
        for (i, l) in lines.iter().enumerate() {
            let t = l.trim_start();
            if t.starts_with("//") || !t.contains("with_epistemic_raw5(") {
                continue;
            }
            if t.contains("fn with_epistemic_raw5") {
                continue;
            }
            let rel = f
                .strip_prefix(&root)
                .expect("under crates/")
                .to_string_lossy()
                .replace('\\', "/");
            out.insert((rel, enclosing_fn(&lines, i)));
        }
    }
    out
}

fn classified() -> BTreeSet<(String, String)> {
    AUTHORIZED
        .iter()
        .map(|(f, n, _)| ((*f).to_string(), (*n).to_string()))
        .chain(
            TEST_FIXTURES
                .iter()
                .map(|(f, n)| ((*f).to_string(), (*n).to_string())),
        )
        .collect()
}

#[test]
fn every_epistemic_writer_is_classified() {
    let found = call_sites();
    let known = classified();
    let unknown: Vec<_> = found.difference(&known).collect();
    assert!(
        unknown.is_empty(),
        "with_epistemic_raw5 called outside the census; add it to AUTHORIZED \
         (with what licenses it) or TEST_FIXTURES: {unknown:?}"
    );
}

#[test]
fn every_census_entry_still_exists() {
    let found = call_sites();
    let known = classified();
    let stale: Vec<_> = known.difference(&found).collect();
    assert!(
        stale.is_empty(),
        "census entries with no matching call; remove them: {stale:?}"
    );
}

/// Anti-vacuity: the scan finds the authorized writers, so an empty scan
/// (wrong root, renamed method) cannot pass both directions at once.
#[test]
fn the_scan_finds_the_authorized_writers() {
    let found = call_sites();
    for (f, n, _) in AUTHORIZED {
        assert!(
            found.contains(&((*f).to_string(), (*n).to_string())),
            "{f}::{n} not found by the scan"
        );
    }
    assert!(found.len() >= AUTHORIZED.len() + TEST_FIXTURES.len());
}
