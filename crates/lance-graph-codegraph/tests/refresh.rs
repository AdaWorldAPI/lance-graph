//! Refresh tests: `refresh_crate` / `refold` against clean opens of the
//! committed fixture stores (copied to a temp dir; `tests/data` is never
//! written).

use std::fs;
use std::path::{Path, PathBuf};

use lance_graph_codegraph::diag::CrateKind;
use lance_graph_codegraph::{Codegraph, Pos, RefreshRefusal};

fn data(store: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/data")
        .join(store)
}

/// A fresh temp dir holding a copy of `store`'s files.
fn copy_store(store: &str, tag: &str) -> PathBuf {
    // Tests run in parallel and several share a tag, so every call gets its
    // own directory: one test deleting another's files mid-read made a
    // `rows.tsv` look empty (`Version { found: 0 }`).
    static NEXT: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
    let n = NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let dir = std::env::temp_dir().join(format!("cg_refresh_{}_{tag}_{n}", std::process::id()));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).expect("temp dir");
    copy_files(&data(store), &dir);
    dir
}

fn copy_files(from: &Path, to: &Path) {
    for e in fs::read_dir(from).expect("read store") {
        let p = e.expect("entry").path();
        fs::copy(&p, to.join(p.file_name().expect("name"))).expect("copy");
    }
}

fn names(g: &Codegraph, mask: &[u64]) -> Vec<String> {
    let mut v: Vec<String> = (0..mask.len() * 64)
        .filter(|&p| mask[p / 64] >> (p % 64) & 1 == 1)
        .filter_map(|p| g.symbol(p as Pos).map(str::to_string))
        .collect();
    v.sort();
    v
}

fn all_names(g: &Codegraph) -> Vec<String> {
    (0..g.words() * 64)
        .filter_map(|p| g.symbol(p as Pos).map(str::to_string))
        .collect()
}

fn refreshed_rev2() -> Codegraph {
    let dir = copy_store("rev1", "eq");
    let mut g = Codegraph::open(&dir).expect("open rev1");
    let r2 = data("rev2");
    for k in ["cg_lib", "cg_app"] {
        fs::copy(
            r2.join(format!("{k}.rows.tsv")),
            dir.join(format!("{k}.rows.tsv")),
        )
        .expect("copy rev2");
        g.refresh_crate(&dir, k).expect("refresh");
    }
    g
}

/// Red if refresh leaves any callers/callees/impact answer, or the
/// DeadFunction set, different from a clean open of rev2 (disable: skip the
/// `add_edge`/`retire_edge` calls in `refresh_crate`).
#[test]
fn refresh_equals_clean_open() {
    let mut g = refreshed_rev2();
    let mut clean = Codegraph::open(&data("rev2")).expect("open rev2");
    for name in all_names(&clean) {
        let (pc, pg) = (
            clean.pos(&name).expect("clean"),
            g.pos(&name).expect("refreshed"),
        );
        let (cc, cg) = (clean.callers(pc).expect("q"), g.callers(pg).expect("q"));
        let (ec, eg) = (clean.callees(pc).expect("q"), g.callees(pg).expect("q"));
        for (a, b) in [
            (&cc.graded.confirmed, &cg.graded.confirmed),
            (&cc.graded.possible, &cg.graded.possible),
            (&cc.graded.unknown, &cg.graded.unknown),
        ] {
            assert_eq!(names(&clean, a), names(&g, b), "callers of {name}");
        }
        assert_eq!(cc.sites, cg.sites, "callers sites of {name}");
        for (a, b) in [
            (&ec.graded.confirmed, &eg.graded.confirmed),
            (&ec.graded.possible, &eg.graded.possible),
            (&ec.graded.unknown, &eg.graded.unknown),
        ] {
            assert_eq!(names(&clean, a), names(&g, b), "callees of {name}");
        }
        let (ic, ig) = (clean.impact(pc).expect("q"), g.impact(pg).expect("q"));
        for (a, b) in [
            (&ic.confirmed, &ig.confirmed),
            (&ic.possible, &ig.possible),
            (&ic.unknown, &ig.unknown),
        ] {
            assert_eq!(names(&clean, a), names(&g, b), "impact of {name}");
        }
    }
    let (dc, dg) = (
        &clean.diagnostics(CrateKind::Lib)[0],
        &g.diagnostics(CrateKind::Lib)[0],
    );
    assert_eq!(names(&clean, &dc.positions), names(&g, &dg.positions));
}

/// Red if the dependent crate's stale call is dropped or duplicated (disable:
/// retire edges of every crate, not only the refreshed one's difference).
#[test]
fn skipping_a_dependent_leaves_one_dangling_call() {
    let dir = copy_store("rev1", "dangle");
    let mut g = Codegraph::open(&dir).expect("open");
    fs::copy(
        data("rev2").join("cg_lib.rows.tsv"),
        dir.join("cg_lib.rows.tsv"),
    )
    .expect("copy");
    g.refresh_crate(&dir, "cg_lib").expect("refresh");
    let f = &g.diagnostics(CrateKind::Lib)[2];
    assert_eq!(f.rows.iter().map(|w| w.count_ones()).sum::<u32>(), 1);
    let row = f
        .rows
        .iter()
        .enumerate()
        .find_map(|(w, &b)| (b != 0).then(|| w * 64 + b.trailing_zeros() as usize))
        .expect("one row");
    assert_eq!(g.row_subject(row), "cg_app::run");
    assert_eq!(g.row_object(row), "cg_lib::area_dyn");
}

/// Red if refresh reassigns positions (disable: rebuild names from scratch in
/// `refresh_crate`).
#[test]
fn a_refresh_does_not_renumber() {
    let dir = copy_store("rev1", "pos");
    let mut g = Codegraph::open(&dir).expect("open");
    let before: Vec<(String, Pos)> = all_names(&g)
        .into_iter()
        .map(|n| {
            let p = g.pos(&n).expect("pos");
            (n, p)
        })
        .collect();
    fs::copy(
        data("rev2").join("cg_lib.rows.tsv"),
        dir.join("cg_lib.rows.tsv"),
    )
    .expect("copy");
    g.refresh_crate(&dir, "cg_lib").expect("refresh");
    for (n, p) in before {
        assert_eq!(g.pos(&n), Some(p), "{n}");
        assert_eq!(g.symbol(p), Some(n.as_str()));
    }
}

/// Red if the cached closure is rebuilt from scratch instead of refreshed
/// (disable: drop the cache unconditionally). Anti-vacuity: rev2 retires
/// edges, so some rows must be recomputed.
#[test]
fn refresh_recomputes_less_than_everything() {
    let dir = copy_store("rev1", "less");
    let mut g = Codegraph::open(&dir).expect("open");
    g.closure(lance_graph_codegraph::graph::RCONF)
        .expect("prime");
    g.closure(lance_graph_codegraph::graph::RPOSS)
        .expect("prime");
    fs::copy(
        data("rev2").join("cg_lib.rows.tsv"),
        dir.join("cg_lib.rows.tsv"),
    )
    .expect("copy");
    let st = g.refresh_crate(&dir, "cg_lib").expect("refresh");
    assert!(st.edges_retired > 0);
    assert!(st.rows_recomputed > 0, "{st:?}");
    assert!(st.rows_recomputed < all_names(&g).len() as u64, "{st:?}");
}

/// Red if refold changes any answer (disable: have refold drop the log
/// without folding it into the base).
#[test]
fn refold_keeps_answers() {
    let mut g = refreshed_rev2();
    let ps: Vec<Pos> = all_names(&g).iter().filter_map(|n| g.pos(n)).collect();
    let snap = |g: &mut Codegraph| -> Vec<(Vec<u64>, Vec<u64>, Vec<u64>)> {
        ps.iter()
            .map(|&p| {
                (
                    g.callers(p).expect("q").graded.confirmed,
                    g.callees(p).expect("q").graded.confirmed,
                    g.impact(p).expect("q").confirmed,
                )
            })
            .collect()
    };
    let before = snap(&mut g);
    g.refold().expect("refold");
    assert_eq!(before, snap(&mut g));
}

fn rows_file(count: usize) -> String {
    let mut s =
        String::from("#version 2\n#schema predicate\tsubject\tobject\tgrade\tfile\tstart\tend\n");
    for i in 0..count {
        s.push_str(&format!(
            "DEFINES\tbig::f{i}\tbig::f{i}\tCOMPILER\tbig/src/lib.rs\t{i}\t{}\n",
            i + 1
        ));
    }
    s
}

/// Red if growth past the reserved tile is accepted (disable: remove the
/// capacity check in `refresh_crate`). Also checks nothing changed.
#[test]
fn growth_past_the_reservation_needs_reopen() {
    let dir = std::env::temp_dir().join(format!("cg_refresh_{}_big", std::process::id()));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).expect("dir");
    fs::write(dir.join("big.rows.tsv"), rows_file(64)).expect("write");
    let mut g = Codegraph::open(&dir).expect("open");
    fs::write(dir.join("big.rows.tsv"), rows_file(65)).expect("write");
    let rows = g.rows();
    match g.refresh_crate(&dir, "big") {
        Err(RefreshRefusal::NeedsReopen { .. }) => {}
        other => panic!("expected NeedsReopen, got {other:?}"),
    }
    assert_eq!(g.rows(), rows);
    assert_eq!(g.pos("big::f64"), None);
}
