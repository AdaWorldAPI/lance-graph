//! A store written by ruff `crates/ruff_codegraph` from a Rails source tree
//! (`tests/data/ruff_codegraph`, copied from that crate's `expected/rev1`)
//! opens with the same reader as a `ruff_charon` store.

use std::path::{Path, PathBuf};

use lance_graph_codegraph::artifact::ResidualKind;
use lance_graph_codegraph::graph::{Codegraph, RunStatus};

fn data(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/data")
        .join(name)
}

fn count(g: &Codegraph, predicate: &str) -> usize {
    (0..g.rows())
        .filter(|&r| g.row_predicate(r) == predicate)
        .count()
}

#[test]
fn source_rows_open_and_keep_their_predicates() {
    let g = Codegraph::open(&data("ruff_codegraph")).expect("open");
    assert_eq!(
        g.crate_list(),
        vec![("shop".to_string(), RunStatus::Partial)]
    );
    assert_eq!(g.rows(), 73);
    for (predicate, n) in [
        ("DEFINES", 18),
        ("CALLBACK", 5),
        ("VALIDATES", 4),
        ("ASSOCIATION", 4),
        ("IMPORTS", 4),
        ("CALLS_NAMED", 1),
        ("UNRESOLVED_CALL", 2),
        ("CALLS_RESOLVED", 0),
    ] {
        assert_eq!(count(&g, predicate), n, "{predicate}");
    }
}

/// A name-matched call is a row, never an edge of the resolved call relation.
#[test]
fn a_name_matched_call_is_not_a_resolved_call() {
    let g = Codegraph::open(&data("ruff_codegraph")).expect("open");
    assert!(g.pos("shop::Ticket::touch").is_some());
    assert!(g.called().iter().all(|w| *w == 0));
    let row = (0..g.rows())
        .find(|&r| g.row_predicate(r) == "CALLS_NAMED")
        .expect("row");
    assert_eq!(g.row_subject(row), "shop::Ticket::close");
    assert_eq!(g.row_object(row), "shop::Ticket::touch");
}

#[test]
fn residuals_are_counted_by_reason() {
    let g = Codegraph::open(&data("ruff_codegraph")).expect("open");
    let counts = g
        .crate_residuals("shop")
        .expect("crate")
        .expect("residual file");
    assert_eq!(counts.of(ResidualKind::CalleeNotNamed), 2);
    assert_eq!(counts.of(ResidualKind::ReferenceNotNamed), 2);
    assert_eq!(counts.total(), 4);
}

/// A Rust store and a source store sit in one directory under two namespaces.
#[test]
fn opens_beside_a_charon_store() {
    let dir = Path::new(env!("CARGO_TARGET_TMPDIR")).join("ruff_codegraph_beside_charon");
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).expect("mkdir");
    for from in [data("ruff_codegraph"), data("fnref")] {
        for e in std::fs::read_dir(&from).expect("list") {
            let e = e.expect("entry");
            std::fs::copy(e.path(), dir.join(e.file_name())).expect("copy");
        }
    }
    let both = Codegraph::open(&dir).expect("open");
    let alone = Codegraph::open(&data("fnref")).expect("open");
    let names: Vec<String> = both.crate_list().into_iter().map(|(n, _)| n).collect();
    assert_eq!(names, ["fnref", "shop"]);
    assert_eq!(both.rows(), alone.rows() + 73);
}
