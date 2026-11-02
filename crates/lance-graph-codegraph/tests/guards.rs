//! Refusal and robustness tests over stores written into a temp dir.

use std::fmt::Write as _;
use std::path::PathBuf;

use lance_graph_codegraph::diag::{CrateKind, Run};
use lance_graph_codegraph::graph::IntakeRefusal;
use lance_graph_codegraph::query::QueryRefusal;
use lance_graph_codegraph::{Codegraph, Pos};

const HEADER: &str = "#version 2\n#schema predicate\tsubject\tobject\tgrade\tfile\tstart\tend\n";

/// Write `content` as `store.rows.tsv` in a fresh per-test temp dir.
fn write(name: &str, content: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("lgcg_{name}_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(dir.join("store.rows.tsv"), content).unwrap();
    dir
}

fn row(pred: &str, s: &str, o: &str) -> String {
    format!("{pred}\t{s}\t{o}\tCOMPILER\tf.rs\t1\t2\n")
}

fn pop(mask: &[u64]) -> u32 {
    mask.iter().map(|w| w.count_ones()).sum()
}

/// Fails if a 70-symbol chain is refused at open, or if impact does not
/// refuse with ClosureWidth(70) (the answer must be UNKNOWN, not empty).
#[test]
fn closure_wider_than_64_is_refused() {
    let mut text = String::from(HEADER);
    for i in 0..70 {
        text += &row("DEFINES", &format!("chain::f{i}"), &format!("chain::f{i}"));
    }
    for i in 0..69 {
        text += &row(
            "CALLS_RESOLVED",
            &format!("chain::f{i}"),
            &format!("chain::f{}", i + 1),
        );
    }
    text += "CRATE\tchain\tCOMPLETE\tCOMPILER\t-\t-\t-\n";
    let dir = write("chain", &text);
    let g = Codegraph::open(&dir).expect("70 symbols fit a tile");
    let f0 = g.pos("chain::f0").unwrap();
    let r = g.impact(f0);
    assert!(matches!(r, Err(QueryRefusal::ClosureWidth(70))), "{r:?}");
    let _ = std::fs::remove_dir_all(&dir);
}

/// Fails if 300 symbols are accepted into one 256-position tile.
#[test]
fn too_many_symbols_is_refused() {
    let mut text = String::from(HEADER);
    for i in 0..300 {
        text += &row("DEFINES", &format!("big::s{i}"), &format!("big::s{i}"));
    }
    let dir = write("many", &text);
    let r = Codegraph::open(&dir);
    assert!(
        matches!(r, Err(IntakeRefusal::TooMany(300))),
        "{:?}",
        r.err()
    );
    let _ = std::fs::remove_dir_all(&dir);
}

/// Fails if a version-1 file is read instead of refused.
#[test]
fn old_version_is_refused() {
    let text = "#version 1\n#schema predicate\tsubject\tobject\tgrade\tfile\tstart\tend\n";
    let dir = write("v1", text);
    let r = Codegraph::open(&dir);
    assert!(
        matches!(r, Err(IntakeRefusal::Version { found: 1 })),
        "{:?}",
        r.err()
    );
    let _ = std::fs::remove_dir_all(&dir);
}

/// Fails if a data line with a missing field is accepted or misread.
#[test]
fn short_line_is_an_arity_refusal() {
    let mut text = String::from(HEADER);
    text += &row("DEFINES", "a::f", "a::f");
    text += "DEFINES\ta::g\ta::g\tCOMPILER\tf.rs\t1\n";
    let dir = write("arity", &text);
    let r = Codegraph::open(&dir);
    assert!(
        matches!(r, Err(IntakeRefusal::Arity { .. })),
        "{:?}",
        r.err()
    );
    let _ = std::fs::remove_dir_all(&dir);
}

/// Fails if column order matters: the same rows with permuted columns must
/// give identical symbols and masks at every position.
#[test]
fn permuted_columns_read_identically() {
    let rows = [
        ["DEFINES", "lib::a", "lib::a"],
        ["PUBLIC", "lib::a", "lib::a"],
        ["DEFINES", "lib::b", "lib::b"],
        ["CALLS_RESOLVED", "lib::a", "lib::b"],
        ["MAY_DISPATCH", "lib::a", "lib::T::m via Clause"],
        ["IMPLEMENTS", "lib::b", "lib::T::m"],
        ["CRATE", "lib", "COMPLETE"],
    ];
    let mut plain = String::from(HEADER);
    let mut perm =
        String::from("#version 2\n#schema object\tpredicate\tsubject\tgrade\tfile\tstart\tend\n");
    for [p, s, o] in rows {
        writeln!(plain, "{p}\t{s}\t{o}\tCOMPILER\tf.rs\t1\t2").unwrap();
        writeln!(perm, "{o}\t{p}\t{s}\tCOMPILER\tf.rs\t1\t2").unwrap();
    }
    let da = write("plain", &plain);
    let db = write("perm", &perm);
    let a = Codegraph::open(&da).unwrap();
    let b = Codegraph::open(&db).unwrap();
    assert_eq!(a.rows(), b.rows());
    assert_eq!(a.words(), b.words());
    assert!(pop(a.fns()) > 0 && pop(a.called()) > 0 && pop(a.may_targets()) > 0);
    for pos in 0..=u8::MAX {
        let pos: Pos = pos;
        assert_eq!(a.symbol(pos), b.symbol(pos));
    }
    assert_eq!(a.fns(), b.fns());
    assert_eq!(a.public(), b.public());
    assert_eq!(a.impl_subjects(), b.impl_subjects());
    assert_eq!(a.synthetic(), b.synthetic());
    assert_eq!(a.referenced(), b.referenced());
    assert_eq!(a.called(), b.called());
    assert_eq!(a.may_targets(), b.may_targets());
    assert_eq!(a.dynamic_callers(), b.dynamic_callers());
    assert_eq!(a.stored_owner(), b.stored_owner());
    let _ = std::fs::remove_dir_all(&da);
    let _ = std::fs::remove_dir_all(&db);
}

/// Fails if the CRATE row status does not decide Complete vs Partial for the
/// dead-code run.
#[test]
fn partial_crate_gives_partial_run() {
    let build = |status: &str| {
        let mut t = String::from(HEADER);
        t += &row("DEFINES", "lib::a", "lib::a");
        t += &row("DEFINES", "lib::b", "lib::b");
        t += &row("CALLS_RESOLVED", "lib::a", "lib::b");
        t += &format!("CRATE\tlib\t{status}\tCOMPILER\t-\t-\t-\n");
        t
    };
    let dp = write("partial", &build("PARTIAL"));
    let dc = write("complete", &build("COMPLETE"));
    let p = Codegraph::open(&dp).unwrap();
    let c = Codegraph::open(&dc).unwrap();
    assert_eq!(p.diagnostics(CrateKind::Lib)[0].run, Run::Partial);
    assert_eq!(c.diagnostics(CrateKind::Lib)[0].run, Run::Complete);
    let _ = std::fs::remove_dir_all(&dp);
    let _ = std::fs::remove_dir_all(&dc);
}
