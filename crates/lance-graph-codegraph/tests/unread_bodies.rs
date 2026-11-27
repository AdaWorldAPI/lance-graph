//! Per-function UNKNOWN for bodies the harvest did not read.
//!
//! A `BODY_NOT_STRUCTURED` residual names a function whose body Charon did not
//! translate (for example an `async fn`, whose coroutine it does not support).
//! Such a function has no call rows at all, so an empty confirmed answer from it
//! means "not observed", never "calls nothing", and it may be a caller of any
//! function. Each gate has a can-fire half and a can-stay-silent half over the
//! same store.

use std::path::{Path, PathBuf};

use lance_graph_codegraph::impact_report::{Consumers, Gap, Scope};
use lance_graph_codegraph::Codegraph;

const HEADER: &str = "#version 2\n#schema predicate\tsubject\tobject\tgrade\tfile\tstart\tend\n";
const RES_HEADER: &str = "#version 1\n#schema reason\tsubject\n";

fn row(pred: &str, s: &str, o: &str) -> String {
    format!("{pred}\t{s}\t{o}\tCOMPILER\tf.rs\t1\t2\n")
}

/// `lib::handler` (unread body), `lib::read_fn` (read, calls `lib::leaf`),
/// `lib::leaf`, `lib::lonely` (read, calls nothing), `lib::outer` (read, calls
/// `lib::handler`). None of them is public, so no consumer gap muddies the
/// impact assertions.
fn rows() -> String {
    let mut t = String::from(HEADER);
    for f in ["handler", "read_fn", "leaf", "lonely", "outer"] {
        t += &row("DEFINES", &format!("lib::{f}"), &format!("lib::{f}"));
    }
    t += &row("CALLS_RESOLVED", "lib::read_fn", "lib::leaf");
    t += &row("CALLS_RESOLVED", "lib::outer", "lib::handler");
    t += "CRATE\tlib\tCOMPLETE\tCOMPILER\t-\t-\t-\n";
    t
}

fn write_residuals(dir: &Path, subjects: &[&str]) {
    let mut t = String::from(RES_HEADER);
    for s in subjects {
        t += &format!("BODY_NOT_STRUCTURED\t{s}\n");
    }
    std::fs::write(dir.join("store.residuals.tsv"), t).unwrap();
}

fn store(name: &str, unread: &[&str]) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("lgcg_unread_{name}_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(dir.join("store.rows.tsv"), rows()).unwrap();
    write_residuals(&dir, unread);
    dir
}

fn bits(mask: &[u64]) -> u32 {
    mask.iter().map(|w| w.count_ones()).sum()
}

/// G1. Fails if an unread body's callees come back as an empty answer.
/// Silence: a read body that calls nothing still answers empty, with no
/// UNKNOWN.
#[test]
fn an_unread_body_has_unknown_callees_and_a_read_one_does_not() {
    let d = store("callees", &["lib::handler"]);
    let mut g = Codegraph::open(&d).unwrap();
    assert_eq!(g.unread_bodies(), g.mask_of(&["lib::handler"]).as_slice());

    let h = g.callees(g.pos("lib::handler").unwrap()).unwrap();
    assert_eq!(
        bits(&h.graded.confirmed),
        0,
        "the fixture has no rows for it"
    );
    assert!(
        bits(&h.graded.unknown) > 0,
        "an unread body answered 'calls nothing'"
    );

    let lonely = g.callees(g.pos("lib::lonely").unwrap()).unwrap();
    assert_eq!(bits(&lonely.graded.confirmed), 0);
    assert_eq!(
        bits(&lonely.graded.unknown),
        0,
        "a read body that calls nothing must not be UNKNOWN"
    );
    let _ = std::fs::remove_dir_all(&d);
}

/// G2. Fails if a function called only from an unread body reports no
/// callers. Silence: with no residual the same function has no caller at all.
#[test]
fn an_unread_body_is_an_unknown_caller() {
    let d = store("callers", &["lib::handler"]);
    let mut g = Codegraph::open(&d).unwrap();
    let leaf = g.callers(g.pos("lib::leaf").unwrap()).unwrap();
    assert_eq!(leaf.graded.confirmed, g.mask_of(&["lib::read_fn"]));
    assert_eq!(
        leaf.graded.unknown,
        g.mask_of(&["lib::handler"]),
        "the unread body may call leaf and must be UNKNOWN, alone"
    );
    let _ = std::fs::remove_dir_all(&d);

    let d = store("callers_clean", &[]);
    let mut g = Codegraph::open(&d).unwrap();
    let lonely = g.callers(g.pos("lib::lonely").unwrap()).unwrap();
    assert_eq!(bits(&lonely.graded.unknown), 0, "no residual, no UNKNOWN");
    let _ = std::fs::remove_dir_all(&d);
}

/// An unread body may be recursive, so it is an UNKNOWN caller of itself, and
/// only of itself here: `outer` stays confirmed and is not repeated as
/// UNKNOWN. (`impact` excludes the changed function, which is the change, not
/// an impact of it; see the impact test.)
#[test]
fn an_unread_body_may_call_itself() {
    let d = store("self", &["lib::handler"]);
    let mut g = Codegraph::open(&d).unwrap();
    let h = g.callers(g.pos("lib::handler").unwrap()).unwrap();
    assert_eq!(h.graded.confirmed, g.mask_of(&["lib::outer"]));
    assert_eq!(h.graded.unknown, g.mask_of(&["lib::handler"]));
    let _ = std::fs::remove_dir_all(&d);
}

/// Impact: a change of `leaf` may reach the unread body and everything that
/// calls it (`outer`). The report names the cause as `UnreadBodies`, never as
/// `DynamicCalls` (the store has none). Silence: with no residual the impact
/// of `leaf` is `read_fn` alone and neither gap appears.
#[test]
fn impact_reaches_unread_bodies_and_their_callers() {
    let d = store("impact", &["lib::handler"]);
    let g = Codegraph::open(&d).unwrap();
    let leaf = g.pos("lib::leaf").unwrap();
    let r = g.impact(leaf).unwrap();
    assert_eq!(r.confirmed, g.mask_of(&["lib::read_fn"]));
    assert_eq!(r.unknown, g.mask_of(&["lib::handler", "lib::outer"]));

    let scope = Scope {
        consumers: Consumers::AllInStore,
        ..Scope::default()
    };
    let rep = g.impact_report("lib::leaf", &scope).unwrap();
    assert!(
        rep.gaps.contains(&Gap::UnreadBodies {
            unknown: 2,
            store_unread_bodies: 1
        }),
        "{:?}",
        rep.gaps
    );
    assert!(
        !rep.gaps
            .iter()
            .any(|x| matches!(x, Gap::DynamicCalls { .. })),
        "an unread body is not a dynamic call: {:?}",
        rep.gaps
    );
    let _ = std::fs::remove_dir_all(&d);

    let d = store("impact_clean", &[]);
    let g = Codegraph::open(&d).unwrap();
    let r = g.impact(g.pos("lib::leaf").unwrap()).unwrap();
    assert_eq!(r.confirmed, g.mask_of(&["lib::read_fn"]));
    assert_eq!(bits(&r.unknown), 0);
    let rep = g.impact_report("lib::leaf", &scope).unwrap();
    assert!(
        !rep.gaps
            .iter()
            .any(|x| matches!(x, Gap::UnreadBodies { .. })),
        "{:?}",
        rep.gaps
    );
    let _ = std::fs::remove_dir_all(&d);
}

/// Refresh: the rows and the residual counts stay identical, only WHICH body
/// is unread changes. Fails if the unchanged-file shortcut compares counts
/// alone and keeps the stale mask.
#[test]
fn a_refresh_picks_up_a_swapped_unread_body() {
    let d = store("swap", &["lib::handler"]);
    let mut g = Codegraph::open(&d).unwrap();
    assert_eq!(g.unread_bodies(), g.mask_of(&["lib::handler"]).as_slice());
    write_residuals(&d, &["lib::lonely"]);
    g.refresh_crate(&d, "store").unwrap();
    assert_eq!(
        g.unread_bodies(),
        g.mask_of(&["lib::lonely"]).as_slice(),
        "the refresh kept the old unread body"
    );
    let _ = std::fs::remove_dir_all(&d);
}

/// A residual for a function with no row has no position: it marks nothing
/// and changes no answer.
#[test]
fn a_residual_without_a_row_marks_nothing() {
    let d = store("norow", &["lib::nowhere"]);
    let mut g = Codegraph::open(&d).unwrap();
    assert_eq!(bits(g.unread_bodies()), 0);
    let leaf = g.callers(g.pos("lib::leaf").unwrap()).unwrap();
    assert_eq!(bits(&leaf.graded.unknown), 0);
    let _ = std::fs::remove_dir_all(&d);
}
