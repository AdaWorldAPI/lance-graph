//! Two cube safety findings, reproduced in isolation (D-XLOG-COV-0 §4).
//!
//! Nothing here repairs either defect; each test pins the current behaviour so
//! the follow-up fix turns it red on purpose. Both use only the public
//! `lance_graph_xlog::cube` API of this checkout.
//!
//! A. `Cube::refresh_closure` documents a precondition (`old` is the closure
//!    of the relation's *base* lanes) and does not check it. A closure of an
//!    older base is accepted and the answer is wrong, with no refusal.
//! B. `Cube::commit` checks that `frame` reads the staged map only with a
//!    `debug_assert`. In a release build a wrong frame is adopted, and a later
//!    terminal decodes a coordinate the adopted frame does not hold.

use lance_graph_dt64::dr::{Frame, LocalMap, MapTable, Width};
use lance_graph_xlog::cube::{Cube, EdgeOp};
use lance_graph_xlog::wide::WideRows;

/// Every `(x, y)` with `y` reachable from `x`, over coordinates `0..n`.
fn pairs(rows: &WideRows) -> Vec<(u32, u32)> {
    let n = rows.n();
    (0..n)
        .flat_map(|x| (0..n).map(move |y| (x, y)))
        .filter(|&(x, y)| rows.contains(x, y))
        .collect()
}

/// One W64 tile `0` holding `ids` in ascending order.
fn table_with(ids: &[u64]) -> (MapTable, Frame) {
    let mut t = MapTable::new(1, 1);
    t.advance(
        1,
        1,
        vec![(0, Some(LocalMap::prepare(Width::W64, ids).unwrap()))],
    );
    let f = t.seal();
    (t, f)
}

/// A. A closure of an older base is accepted and refreshed into a wrong
/// answer. The control (a closure of the current base) agrees with a fresh
/// `closure_wide`, so the difference is the stale input alone.
#[test]
fn a_refresh_from_a_stale_base_is_accepted_and_wrong() {
    let (table, frame) = table_with(&[1, 2, 3, 4, 5]);
    let edges = [(1u32, 2u32), (2, 3)];
    let mut cube = Cube::new_tiled(&table, &frame, &[0], &[edges.as_slice()]).unwrap();

    // Closure of base B0 = {1→2, 2→3}.
    let (stale, _) = cube.closure_wide(&table, 0).unwrap();

    // Base B1 = B0 + {3→4}, folded; then a logged add 4→5.
    cube.add_edge(&table, 0, 3, 4).unwrap();
    cube.refold(0).unwrap();
    let (current, _) = cube.closure_wide(&table, 0).unwrap();
    cube.add_edge(&table, 0, 4, 5).unwrap();

    let (truth, _) = cube.closure_wide(&table, 0).unwrap();
    let (control, _) = cube.refresh_closure(&table, 0, &current).unwrap();
    assert_eq!(pairs(&control), pairs(&truth), "precondition met: correct");

    // Precondition violated: `stale` is the closure of B0, not of B1.
    let wrong = cube.refresh_closure(&table, 0, &stale);
    let (wrong, _) = wrong.expect("DEFECT A: a stale base is not refused");
    assert_ne!(
        pairs(&wrong),
        pairs(&truth),
        "DEFECT A: the refreshed closure from a stale base differs from the truth"
    );
    // Anti-vacuity: the truth really contains a pair the stale refresh lost
    // (1 reaches 5 only through the folded 3→4).
    let lost: Vec<_> = pairs(&truth)
        .into_iter()
        .filter(|p| !pairs(&wrong).contains(p))
        .collect();
    assert!(!lost.is_empty(), "the stale refresh must lose a pair");
}

/// The cube of B, staged with a grown map `[1, 2, 3]` and one edge `3 → 2`,
/// with the table already advanced so `f2` reads the grown map. Returns the
/// stale frame too.
fn staged_case() -> (MapTable, Cube, Frame, Frame, lance_graph_xlog::cube::StagedEdges) {
    let (mut table, f1) = table_with(&[1, 2]);
    let edges = [(1u32, 2u32)];
    let cube = Cube::new(&table, &f1, 0, &[edges.as_slice()]).unwrap();
    let grown = LocalMap::prepare(Width::W64, &[1, 2, 3]).unwrap();
    let staged = cube
        .stage(&table, Some((0, &grown)), &[(0, EdgeOp::Add, 3, 2)])
        .unwrap();
    table.advance(2, 1, vec![(0, Some(grown))]);
    let f2 = table.seal();
    (table, cube, f1, f2, staged)
}

/// Control for B: committed under the frame that reads the staged map, the
/// new edge decodes to global ids.
#[test]
fn commit_under_the_right_frame_decodes_the_new_edge() {
    let (table, mut cube, _f1, f2, staged) = staged_case();
    cube.commit(&table, &f2, staged);
    assert_eq!(cube.frame().revisions(), f2.revisions());
    let l = cube.scan(0).unwrap();
    let r = cube.scan(0).unwrap();
    let j = cube.join(l, 1, r, 1).unwrap();
    let rows = cube.materialize_rows(&table, j).unwrap();
    assert!(rows.contains(&(3, 2, 3)), "the committed edge 3→2 is read back");
}

/// B in a debug build: the guard fires.
#[test]
#[cfg(debug_assertions)]
#[should_panic(expected = "the frame does not read the staged map")]
fn commit_under_a_wrong_frame_panics_in_debug() {
    let (table, mut cube, f1, _f2, staged) = staged_case();
    cube.commit(&table, &f1, staged);
}

/// B in a release build: the guard is compiled out, the wrong frame is
/// adopted, and a later terminal decodes a coordinate the adopted frame does
/// not hold. Run with `--release`.
#[test]
#[cfg(not(debug_assertions))]
fn commit_under_a_wrong_frame_is_adopted_in_release() {
    let (table, mut cube, f1, f2, staged) = staged_case();
    cube.commit(&table, &f1, staged);
    assert_eq!(
        cube.frame().revisions(),
        f1.revisions(),
        "DEFECT B: the stale frame was adopted without refusal"
    );
    assert_ne!(f1.revisions(), f2.revisions(), "fixture: two distinct frames");
    assert_eq!(cube.delta_len(0), 1, "the batch was applied");
    let l = cube.scan(0).unwrap();
    let r = cube.scan(0).unwrap();
    let j = cube.join(l, 1, r, 1).unwrap();
    let decoded = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        cube.materialize_rows(&table, j)
    }));
    assert!(
        decoded.is_err(),
        "DEFECT B: the adopted frame cannot decode the committed coordinate"
    );
}
