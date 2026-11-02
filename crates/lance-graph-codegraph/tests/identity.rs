//! Identity gates: a fact whose symbol is not fully qualified is refused at
//! intake, so two distinct unresolved targets can never merge onto one
//! placeholder symbol (such as `?::?`). This is defence in depth; the
//! harvester must not emit such a symbol in the first place.

use std::path::PathBuf;

use lance_graph_codegraph::Codegraph;

const HEADER: &str = "#version 2\n#schema predicate\tsubject\tobject\tgrade\tfile\tstart\tend\n";
const RES: &str = "#version 1\n#schema reason\tsubject\n";

fn row(pred: &str, s: &str, o: &str) -> String {
    format!("{pred}\t{s}\t{o}\tCOMPILER\tf.rs\t1\t2\n")
}

fn store(name: &str, extra: &[(&str, &str, &str)]) -> PathBuf {
    let mut t = String::from(HEADER);
    for f in ["lib::f1", "lib::f2"] {
        t += &row("DEFINES", f, f);
        t += &row("PUBLIC", f, f);
    }
    for (p, s, o) in extra {
        t += &row(p, s, o);
    }
    t += "CRATE\tlib\tCOMPLETE\tCOMPILER\t-\t-\t-\n";
    let dir = std::env::temp_dir().join(format!("lgcg_id_{name}_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(dir.join("store.rows.tsv"), t).unwrap();
    std::fs::write(dir.join("store.residuals.tsv"), RES).unwrap();
    dir
}

/// Fails if two dispatches to distinct, unnamed trait methods are admitted:
/// both would land on the one symbol `?::?` and each caller would read as a
/// possible caller of the other's target.
#[test]
fn two_unnamed_dispatch_targets_are_refused() {
    let d = store(
        "unnamed",
        &[
            ("MAY_DISPATCH", "lib::f1", "?::? via Clause"),
            ("MAY_DISPATCH", "lib::f2", "?::? via Dyn"),
        ],
    );
    assert!(Codegraph::open(&d).is_err());
    let _ = std::fs::remove_dir_all(&d);
}

/// Fails if a partly qualified target (a named trait, an unnamed method) is
/// admitted: every unnamed method of that trait would share one symbol.
#[test]
fn an_unnamed_method_of_a_named_trait_is_refused() {
    let d = store("half", &[("MAY_DISPATCH", "lib::f1", "lib::T::? via Clause")]);
    assert!(Codegraph::open(&d).is_err());
    let _ = std::fs::remove_dir_all(&d);
}

/// Fails if a placeholder segment in a resolved call or a definition is
/// admitted.
#[test]
fn a_placeholder_in_any_symbol_is_refused() {
    for (p, s, o) in [
        ("CALLS_RESOLVED", "lib::f1", "lib::?"),
        ("DEFINES", "?::g", "?::g"),
        ("REFERENCES", "lib::f1", "?"),
    ] {
        let d = store(p, &[(p, s, o)]);
        assert!(Codegraph::open(&d).is_err(), "{p} {s} {o}");
        let _ = std::fs::remove_dir_all(&d);
    }
}

/// The silence twin: two dispatches to distinct, named targets are admitted
/// and stay distinct (each caller is a possible caller of its own target
/// only). Also: `?` inside a name segment (not a whole segment) is not a
/// placeholder.
#[test]
fn distinct_named_targets_are_admitted_and_stay_distinct() {
    let d = store(
        "named",
        &[
            ("MAY_DISPATCH", "lib::f1", "lib::A::m via Clause"),
            ("MAY_DISPATCH", "lib::f2", "lib::B::m via Dyn"),
            ("CALLS_RESOLVED", "lib::f1", "lib::r#try?x"),
            ("DEFINES", "lib::ia", "lib::ia"),
            ("IMPLEMENTS", "lib::ia", "lib::A::m"),
            ("DEFINES", "lib::ib", "lib::ib"),
            ("IMPLEMENTS", "lib::ib", "lib::B::m"),
        ],
    );
    let mut g = Codegraph::open(&d).expect("named targets must be admitted");
    // `callers` of an impl: the dispatches that may reach it, through the
    // IMPLEMENTS join (the join a shared `?::?` would have merged).
    let (pa, pb) = (g.pos("lib::ia").unwrap(), g.pos("lib::ib").unwrap());
    let (f1, f2) = (g.mask_of(&["lib::f1"]), g.mask_of(&["lib::f2"]));
    let a = g.callers(pa).unwrap();
    let b = g.callers(pb).unwrap();
    assert_eq!(a.graded.possible, f1);
    assert_eq!(b.graded.possible, f2);
    let _ = std::fs::remove_dir_all(&d);
}
