//! Oracle tests: exact answers over the committed fixture stores, checked by
//! hand against `tests/data/*/*.rows.tsv`.

use std::path::PathBuf;

use lance_graph_codegraph::diag::{CrateKind, FindingStatus, Kind, Run};
use lance_graph_codegraph::{Codegraph, Pos};

fn open(store: &str) -> Codegraph {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/data")
        .join(store);
    Codegraph::open(&dir).expect("fixture store opens")
}

fn m(g: &Codegraph, names: &[&str]) -> Vec<u64> {
    g.mask_of(names)
}

fn p(g: &Codegraph, name: &str) -> Pos {
    g.pos(name)
        .unwrap_or_else(|| panic!("symbol {name} not in store"))
}

fn pop(mask: &[u64]) -> u32 {
    mask.iter().map(|w| w.count_ones()).sum()
}

const SQ: &str = "cg_lib::shapes::impl_Shape_for_Square::area";
const RECT: &str = "cg_lib::shapes::impl_Shape_for_Rect::area";
const DOUBLE: &str = "cg_lib::util::double";
const TRIPLE: &str = "cg_lib::util::triple";
const TWICE: &str = "cg_lib::util::twice";
const AREA_OF: &str = "cg_lib::area_of";
const AREA_DYN: &str = "cg_lib::area_dyn";
const CHECKED: &str = "cg_lib::macro_use::checked";
const ASSERT_FAILED: &str = "core::panicking::assert_failed";
const RUN: &str = "cg_app::run";

/// Fails if `double`'s confirmed callers are not exactly {Square::area, run},
/// if it reports 2 sites wrongly, or if any possible/unknown caller appears.
#[test]
fn rev1_callers_of_double() {
    let mut g = open("rev1");
    let c = g.callers(p(&g, DOUBLE)).unwrap();
    assert_eq!(c.graded.confirmed, m(&g, &[SQ, RUN]));
    assert_eq!(c.sites, 2);
    assert_eq!(c.graded.possible, m(&g, &[]));
    assert_eq!(c.graded.unknown, m(&g, &[]));
}

/// Fails if `triple`'s only confirmed caller is not `checked` with 1 site.
#[test]
fn rev1_callers_of_triple() {
    let mut g = open("rev1");
    let c = g.callers(p(&g, TRIPLE)).unwrap();
    assert_eq!(c.graded.confirmed, m(&g, &[CHECKED]));
    assert_eq!(c.sites, 1);
}

/// Fails if the dispatch caller `area_of` is promoted to confirmed, or if the
/// dynamic caller `area_dyn` is not unknown.
#[test]
fn rev1_callers_of_square_area_grades_dispatch() {
    let mut g = open("rev1");
    let c = g.callers(p(&g, SQ)).unwrap();
    assert_eq!(c.graded.confirmed, m(&g, &[RUN]));
    assert_eq!(c.graded.possible, m(&g, &[AREA_OF]));
    assert_eq!(c.graded.unknown, m(&g, &[AREA_DYN]));
    let area_of_bit = m(&g, &[AREA_OF]);
    assert_ne!(c.graded.confirmed, area_of_bit);
    assert!(c
        .graded
        .confirmed
        .iter()
        .zip(&area_of_bit)
        .all(|(a, b)| a & b == 0));
}

/// Fails if Rect::area (never called directly) gains a confirmed caller.
#[test]
fn rev1_callers_of_rect_area() {
    let mut g = open("rev1");
    let c = g.callers(p(&g, RECT)).unwrap();
    assert_eq!(c.graded.confirmed, m(&g, &[]));
    assert_eq!(c.graded.possible, m(&g, &[AREA_OF]));
    assert_eq!(c.graded.unknown, m(&g, &[AREA_DYN]));
}

/// Fails if `run`'s four resolved callees or the 4 call sites are wrong.
#[test]
fn rev1_callees_of_run() {
    let mut g = open("rev1");
    let c = g.callees(p(&g, RUN)).unwrap();
    assert_eq!(c.graded.confirmed, m(&g, &[AREA_OF, AREA_DYN, SQ, DOUBLE]));
    assert_eq!(c.sites, 4);
}

/// Fails if `area_of`'s dispatch targets are confirmed or not both possible.
#[test]
fn rev1_callees_of_area_of() {
    let mut g = open("rev1");
    let c = g.callees(p(&g, AREA_OF)).unwrap();
    assert_eq!(c.graded.confirmed, m(&g, &[]));
    assert_eq!(c.graded.possible, m(&g, &[SQ, RECT]));
}

/// Fails if `checked`'s callees (including the external assert_failed) differ.
#[test]
fn rev1_callees_of_checked() {
    let mut g = open("rev1");
    let c = g.callees(p(&g, CHECKED)).unwrap();
    assert_eq!(c.graded.confirmed, m(&g, &[TRIPLE, ASSERT_FAILED]));
}

/// Fails if the reference sites to `double` are not 2 confirmed, 0 possible.
#[test]
fn rev1_references_of_double() {
    let g = open("rev1");
    let s = g.references(p(&g, DOUBLE));
    assert_eq!(s.confirmed, 2);
    assert_eq!(s.possible, 0);
    assert_eq!(pop(&s.rows), 2);
}

/// Fails if the transitive impact of `double` is mis-graded.
#[test]
fn rev1_impact_of_double() {
    let g = open("rev1");
    let i = g.impact(p(&g, DOUBLE)).unwrap();
    assert_eq!(i.confirmed, m(&g, &[SQ, RUN]));
    assert_eq!(i.possible, m(&g, &[AREA_OF]));
    assert_eq!(i.unknown, m(&g, &[AREA_DYN]));
}

/// Fails if area_of is called once instead of twice (sites 4, popcount 3).
#[test]
fn rev2_callees_of_run_counts_sites_not_positions() {
    let mut g = open("rev2");
    let c = g.callees(p(&g, RUN)).unwrap();
    assert_eq!(pop(&c.graded.confirmed), 3);
    assert_eq!(c.graded.confirmed, m(&g, &[AREA_OF, SQ, DOUBLE]));
    assert_eq!(c.sites, 4);
}

/// Fails if the two-hop chain checked -> twice -> double is not followed.
#[test]
fn rev2_impact_of_double_is_transitive() {
    let g = open("rev2");
    let i = g.impact(p(&g, DOUBLE)).unwrap();
    assert_eq!(i.confirmed, m(&g, &[SQ, RUN, TWICE, CHECKED]));
    assert_eq!(i.possible, m(&g, &[AREA_OF]));
    assert_eq!(i.unknown, m(&g, &[]));
}

/// Fails if dead code is not exactly `private_unused`, or the mask is vacuous
/// (nearly everything flagged).
///
/// Re-pinned from `Run::Complete`: rev1 has one dynamic call
/// (`cg_lib::area_dyn`). A function reachable only through a pointer is dead
/// only if every taken address was seen, which nothing proves, so the run is
/// `Partial`. The cause is asserted to be exactly that one call.
#[test]
fn rev1_dead_function() {
    let g = open("rev1");
    let f = &g.diagnostics(CrateKind::Lib)[0];
    assert_eq!(f.kind, Kind::DeadFunction);
    assert_eq!(f.positions, m(&g, &["cg_lib::util::private_unused"]));
    assert_eq!(f.run, Run::Partial);
    let c = g.coverage();
    assert_eq!(c.dynamic_calls, 1);
    assert_eq!(c.residuals.total(), 0);
    assert_eq!(c.crates_without_residuals, 0);
    assert_eq!(c.partial_crates, 0);
    assert!(pop(&f.positions) * 3 < pop(g.fns()));
}

/// Fails if the dynamic caller `area_dyn` is not the one UnresolvedCall row.
#[test]
fn rev1_unresolved_call() {
    let g = open("rev1");
    let f = &g.diagnostics(CrateKind::Lib)[1];
    assert_eq!(f.kind, Kind::UnresolvedCall);
    assert_eq!(f.positions, m(&g, &[AREA_DYN]));
    assert_eq!(pop(&f.rows), 1);
}

/// Fails if MAY_DISPATCH is mistaken for an unresolved call in rev2; the
/// anti-vacuity check proves a MAY_DISPATCH row exists.
#[test]
fn rev2_no_unresolved_call() {
    let g = open("rev2");
    assert!((0..g.rows()).any(|r| g.row_predicate(r) == "MAY_DISPATCH"));
    let f = &g.diagnostics(CrateKind::Lib)[1];
    assert_eq!(pop(&f.rows), 0);
    assert_eq!(pop(&f.positions), 0);
}

/// Fails if a callee that no stored crate defines (assert_failed, from
/// `core`) is reported dangling in a complete store.
#[test]
fn rev1_rev2_no_dangling_call() {
    for store in ["rev1", "rev2"] {
        let g = open(store);
        let callees_external = g.pos(ASSERT_FAILED).is_some();
        assert!(callees_external, "{store}: external callee must exist");
        let f = &g.diagnostics(CrateKind::Lib)[2];
        assert_eq!(f.kind, Kind::DanglingCall);
        assert_eq!(pop(&f.rows), 0, "{store}");
    }
}

/// Fails if the stale cross-crate call run -> area_dyn (area_dyn gone from
/// the lib) is not reported as exactly one Stale row.
#[test]
fn mixed_dangling_call_is_stale() {
    let g = open("mixed");
    let f = &g.diagnostics(CrateKind::Lib)[2];
    assert_eq!(f.kind, Kind::DanglingCall);
    assert_eq!(pop(&f.rows), 1);
    let row = f
        .rows
        .iter()
        .enumerate()
        .find_map(|(w, &bits)| (bits != 0).then(|| w * 64 + bits.trailing_zeros() as usize))
        .unwrap();
    assert_eq!(g.row_subject(row), RUN);
    assert_eq!(g.row_object(row), AREA_DYN);
    assert_eq!(f.status, FindingStatus::Stale);
}

/// Fails if UnusedImport claims to have run when no import facts exist.
#[test]
fn rev1_unused_import_not_run() {
    let g = open("rev1");
    let f = &g.diagnostics(CrateKind::Lib)[3];
    assert_eq!(f.kind, Kind::UnusedImport);
    assert_eq!(
        f.run,
        Run::NotRun {
            reason: "NO_IMPORT_FACTS"
        }
    );
}

/// Fails if a function used only as a function pointer (`helper`) is dead,
/// or `unused_private` is missed.
#[test]
fn fnref_dead_function() {
    let g = open("fnref");
    let f = &g.diagnostics(CrateKind::Lib)[0];
    assert_eq!(f.positions, m(&g, &["fnref::unused_private"]));
}

/// Fails if a binary crate (entry points unknown) is allowed to report dead code.
#[test]
fn rev1_bin_dead_function_refused() {
    let g = open("rev1");
    let f = &g.diagnostics(CrateKind::Bin)[0];
    assert_eq!(f.kind, Kind::DeadFunction);
    assert_eq!(
        f.run,
        Run::Refused {
            reason: "NO_ENTRY_FACTS"
        }
    );
}

/// Fails if any query or diagnostic materialises cube rows instead of folding.
#[test]
fn no_rows_materialized() {
    let mut g = open("rev1");
    let double = p(&g, DOUBLE);
    let run = p(&g, RUN);
    g.callers(double).unwrap();
    g.callees(run).unwrap();
    g.references(double);
    g.impact(double).unwrap();
    g.diagnostics(CrateKind::Lib);
    g.diagnostics(CrateKind::Bin);
    assert_eq!(g.cube().stats().rows_materialized, 0);
}
