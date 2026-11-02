//! Coverage gates for closed-world diagnostics.
//!
//! "Dead function" reads the absence of CALLS / REFERENCES rows as "nothing
//! reaches it". That reading holds only when the harvest saw every body,
//! every callee and every taken address. The harvester records what it could
//! not see in `<crate>.residuals.tsv`. A residual, a missing residuals file,
//! or an unresolved (dynamic) call means the absence of a row proves nothing,
//! so the run must say `Partial`, never `Complete`.
//!
//! Each gate has a can-fire test and a can-stay-silent twin
//! (`a_fully_covered_crate_stays_complete`).

use std::path::{Path, PathBuf};

use lance_graph_codegraph::artifact::ResidualKind;
use lance_graph_codegraph::diag::{CrateKind, Kind, Run};
use lance_graph_codegraph::Codegraph;

const HEADER: &str = "#version 2\n#schema predicate\tsubject\tobject\tgrade\tfile\tstart\tend\n";
const RES_HEADER: &str = "#version 1\n#schema reason\tsubject\n";

fn row(pred: &str, s: &str, o: &str) -> String {
    format!("{pred}\t{s}\t{o}\tCOMPILER\tf.rs\t1\t2\n")
}

/// A library crate: `lib::entry` is public and calls nothing in the rows;
/// `lib::helper` is private and has no caller in the rows.
fn rows(extra: &[(&str, &str, &str)]) -> String {
    let mut t = String::from(HEADER);
    t += &row("DEFINES", "lib::entry", "lib::entry");
    t += &row("PUBLIC", "lib::entry", "lib::entry");
    t += &row("DEFINES", "lib::helper", "lib::helper");
    for (p, s, o) in extra {
        t += &row(p, s, o);
    }
    t += "CRATE\tlib\tCOMPLETE\tCOMPILER\t-\t-\t-\n";
    t
}

/// Write `store.rows.tsv`, and `store.residuals.tsv` when `residuals` is
/// `Some` (each entry one `reason\tsubject` line).
fn store(name: &str, rows: &str, residuals: Option<&[(&str, &str)]>) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("lgcg_cov_{name}_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(dir.join("store.rows.tsv"), rows).unwrap();
    if let Some(res) = residuals {
        write_residuals(&dir, res);
    }
    dir
}

fn write_residuals(dir: &Path, res: &[(&str, &str)]) {
    let mut t = String::from(RES_HEADER);
    for (reason, subject) in res {
        t += &format!("{reason}\t{subject}\n");
    }
    std::fs::write(dir.join("store.residuals.tsv"), t).unwrap();
}

fn dead_run(dir: &Path) -> Run {
    let g = Codegraph::open(dir).unwrap();
    let f = &g.diagnostics(CrateKind::Lib)[0];
    assert_eq!(f.kind, Kind::DeadFunction);
    f.run
}

/// The silence twin: every gate below must stay quiet on a crate whose
/// harvest saw everything. Also anti-vacuity: `helper` is reported dead, so
/// the diagnostic is live, not merely labelled.
#[test]
fn a_fully_covered_crate_stays_complete() {
    let d = store("clean", &rows(&[]), Some(&[]));
    let g = Codegraph::open(&d).unwrap();
    let f = &g.diagnostics(CrateKind::Lib)[0];
    assert_eq!(f.run, Run::Complete);
    assert_eq!(f.positions, g.mask_of(&["lib::helper"]));
    let _ = std::fs::remove_dir_all(&d);
}

/// Fails if a body the harvester could not structure is ignored: `entry`'s
/// calls are unknown, so `helper` may be called and the run is not complete.
#[test]
fn an_unstructured_body_degrades_the_dead_run() {
    let d = store(
        "body",
        &rows(&[]),
        Some(&[("BODY_NOT_STRUCTURED", "lib::entry")]),
    );
    assert_eq!(dead_run(&d), Run::Partial);
    let _ = std::fs::remove_dir_all(&d);
}

/// Fails if an unnamed callee or reference is ignored.
#[test]
fn an_unnamed_callee_or_reference_degrades_the_dead_run() {
    for reason in ["CALLEE_NOT_NAMED", "REFERENCE_NOT_NAMED"] {
        let d = store(reason, &rows(&[]), Some(&[(reason, "lib::entry")]));
        assert_eq!(dead_run(&d), Run::Partial, "{reason}");
        let _ = std::fs::remove_dir_all(&d);
    }
}

/// Fails if incomplete IMPLEMENTS coverage is ignored.
#[test]
fn a_missing_trait_method_degrades_the_dead_run() {
    for reason in ["TRAIT_METHOD_NOT_FOUND", "TRAIT_IMPL_METHOD_NOT_FOUND"] {
        let d = store(reason, &rows(&[]), Some(&[(reason, "lib::entry")]));
        assert_eq!(dead_run(&d), Run::Partial, "{reason}");
        let _ = std::fs::remove_dir_all(&d);
    }
}

/// Fails if a crate with no residuals file is treated as fully covered: the
/// absence of the file is not evidence that nothing was missed.
#[test]
fn a_missing_residuals_file_degrades_the_dead_run() {
    let d = store("nofile", &rows(&[]), None);
    assert_eq!(dead_run(&d), Run::Partial);
    let _ = std::fs::remove_dir_all(&d);
}

/// Fails if an unresolved (dynamic) call is ignored: a function reachable
/// only through a pointer is "dead" only if every taken address was seen,
/// which nothing here proves.
#[test]
fn a_dynamic_call_degrades_the_dead_run() {
    let d = store(
        "dyn",
        &rows(&[("UNRESOLVED_CALL", "lib::entry", "dynamic")]),
        Some(&[]),
    );
    assert_eq!(dead_run(&d), Run::Partial);
    let _ = std::fs::remove_dir_all(&d);
}

/// Fails if an unknown residual reason is accepted: a reason the reader does
/// not know cannot be classified, so intake refuses rather than guessing.
#[test]
fn an_unknown_residual_reason_is_refused() {
    let d = store(
        "unknown",
        &rows(&[]),
        Some(&[("SOMETHING_NEW", "lib::entry")]),
    );
    assert!(Codegraph::open(&d).is_err());
    let _ = std::fs::remove_dir_all(&d);
}

/// Fails if a refresh does not pick up the crate's new residuals: the rows
/// change (so the digest differs) and a residual appears.
#[test]
fn a_refresh_reads_the_new_residuals() {
    let d = store("refresh", &rows(&[]), Some(&[]));
    let mut g = Codegraph::open(&d).unwrap();
    assert_eq!(g.diagnostics(CrateKind::Lib)[0].run, Run::Complete);
    std::fs::write(
        d.join("store.rows.tsv"),
        rows(&[("DEFINES", "lib::other", "lib::other")]),
    )
    .unwrap();
    write_residuals(&d, &[("BODY_NOT_STRUCTURED", "lib::entry")]);
    g.refresh_crate(&d, "store").unwrap();
    assert_eq!(g.diagnostics(CrateKind::Lib)[0].run, Run::Partial);
    let _ = std::fs::remove_dir_all(&d);
}

/// Fails if the cause of a partial run is not recoverable as typed data: two
/// different gaps must report two different counts, each under its own kind.
#[test]
fn the_partial_cause_stays_typed() {
    let d = store(
        "typed",
        &rows(&[("UNRESOLVED_CALL", "lib::entry", "dynamic")]),
        Some(&[
            ("BODY_NOT_STRUCTURED", "lib::entry"),
            ("TRAIT_IMPL_METHOD_NOT_FOUND", "lib::entry"),
            ("TRAIT_IMPL_METHOD_NOT_FOUND", "lib::helper"),
        ]),
    );
    let c = Codegraph::open(&d).unwrap().coverage();
    assert_eq!(c.residuals.of(ResidualKind::BodyNotStructured), 1);
    assert_eq!(c.residuals.of(ResidualKind::TraitImplMethodNotFound), 2);
    assert_eq!(c.residuals.of(ResidualKind::CalleeNotNamed), 0);
    assert_eq!(c.dynamic_calls, 1);
    assert_eq!(c.crates_without_residuals, 0);
    let _ = std::fs::remove_dir_all(&d);
}
