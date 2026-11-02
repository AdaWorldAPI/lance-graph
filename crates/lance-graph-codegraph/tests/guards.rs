//! Refusal and robustness tests over stores written into a temp dir.

use std::fmt::Write as _;
use std::path::PathBuf;

use lance_graph_codegraph::diag::{CrateKind, Run};
use lance_graph_codegraph::graph::IntakeRefusal;
use lance_graph_codegraph::Codegraph;

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

/// Re-pinned: this used to assert `ClosureWidth(70)` because the cube's
/// closure held one `u64` per row. Closures are now wide, so a 70-symbol
/// chain (two words per row) must close across the word boundary. The chain
/// is f0 -> f1 -> ... -> f69, so the callers of f69 transitively are f0..f68.
/// Fails if any of those 69 bits is lost, including those in the second word.
#[test]
fn a_70_function_chain_closes_across_words() {
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
    let g = Codegraph::open(&dir).expect("70 symbols open");
    let f69 = g.pos("chain::f69").unwrap();
    let r = g.impact(f69).expect("wide closure never refuses");
    assert_eq!(pop(&r.confirmed), 69);
    let others: Vec<String> = (0..69).map(|i| format!("chain::f{i}")).collect();
    let refs: Vec<&str> = others.iter().map(String::as_str).collect();
    assert_eq!(r.confirmed, g.mask_of(&refs));
    let _ = std::fs::remove_dir_all(&dir);
}

/// Re-pinned: this used to assert `TooMany(300)`. A store of any size opens
/// now: 300 symbols take two 256-position tiles, and coordinates equal
/// positions. Fails if open is refused or a position past the first tile
/// does not round-trip.
#[test]
fn three_hundred_symbols_open_across_two_tiles() {
    let mut text = String::from(HEADER);
    for i in 0..300 {
        text += &row("DEFINES", &format!("big::s{i}"), &format!("big::s{i}"));
    }
    let dir = write("many", &text);
    let g = Codegraph::open(&dir).expect("300 symbols open");
    assert!(g.symbol(299).is_some());
    assert_eq!(g.pos("big::s299"), Some(299));
    assert_eq!(g.symbol(300), None);
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
    for pos in 0..=255_u32 {
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

/// Fails if a trait-impl method with no direct caller is reported dead.
/// Charon marks impl methods public whenever the trait is, so the fixtures
/// never reach this case; an impl of a private trait is not public and is
/// called only through dispatch.
#[test]
fn an_uncalled_private_impl_method_is_not_dead() {
    let mut t = String::from(HEADER);
    t += &row("DEFINES", "lib::impl_T_for_S::m", "lib::impl_T_for_S::m");
    t += &row("IMPLEMENTS", "lib::impl_T_for_S::m", "lib::T::m");
    t += &row("DEFINES", "lib::dead", "lib::dead");
    t += "CRATE\tlib\tCOMPLETE\tCOMPILER\t-\t-\t-\n";
    let d = write("private_impl", &t);
    let g = Codegraph::open(&d).unwrap();
    let dead = &g.diagnostics(CrateKind::Lib)[0].positions;
    assert_eq!(
        *dead,
        g.mask_of(&["lib::dead"]),
        "the impl method must stay silent"
    );
    let _ = std::fs::remove_dir_all(&d);
}
