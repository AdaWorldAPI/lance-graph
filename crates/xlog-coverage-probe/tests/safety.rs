//! The two cube safety findings of D-XLOG-COV-0 §4, now refused.
//!
//! Both used to be pinned as defects; the repair turned them into typed
//! refusals in every build, and each test here was flipped on purpose.
//!
//! A. `Cube::refresh_closure` takes a `BaseClosure` that names the base
//!    generation it was computed over. A closure of an older base is refused
//!    with `CubeRefusal::StaleBase`.
//! B. `Cube::commit` returns `Result` and checks the frame in every build. A
//!    frame that does not read the staged map is refused with
//!    `CubeRefusal::FrameMismatch`, and the cube is unchanged.

use lance_graph_dt64::dr::{Frame, LocalMap, MapTable, Width};
use lance_graph_xlog::cube::{Cube, CubeRefusal, EdgeOp};
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

/// A. A closure of an older base is refused. The control (a closure of the
/// current base) agrees with a fresh `closure_wide`.
#[test]
fn a_refresh_from_a_stale_base_is_refused() {
    let (table, frame) = table_with(&[1, 2, 3, 4, 5]);
    let edges = [(1u32, 2u32), (2, 3)];
    let mut cube = Cube::new_tiled(&table, &frame, &[0], &[edges.as_slice()]).unwrap();

    // Closure of base B0 = {1→2, 2→3}.
    let stale = cube.base_closure(&table, 0).unwrap();

    // Base B1 = B0 + {3→4}, folded; then a logged add 4→5.
    cube.add_edge(&table, 0, 3, 4).unwrap();
    cube.refold(0).unwrap();
    let current = cube.base_closure(&table, 0).unwrap();
    cube.add_edge(&table, 0, 4, 5).unwrap();

    let (truth, _) = cube.closure_wide(&table, 0).unwrap();
    let (control, _) = cube.refresh_closure(&table, 0, &current).unwrap();
    assert_eq!(pairs(&control), pairs(&truth), "current base: correct");

    assert_eq!(
        cube.refresh_closure(&table, 0, &stale),
        Err(CubeRefusal::StaleBase {
            minted: 0,
            current: 1
        })
    );
}

/// The cube of B, staged with a grown map `[1, 2, 3]` and one edge `3 → 2`,
/// with the table already advanced so `f2` reads the grown map. Returns the
/// stale frame too.
fn staged_case() -> (
    MapTable,
    Cube,
    Frame,
    Frame,
    lance_graph_xlog::cube::StagedEdges,
) {
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
    cube.commit(&table, &f2, staged).unwrap();
    assert_eq!(cube.frame().revisions(), f2.revisions());
    let l = cube.scan(0).unwrap();
    let r = cube.scan(0).unwrap();
    let j = cube.join(l, 1, r, 1).unwrap();
    let rows = cube.materialize_rows(&table, j).unwrap();
    assert!(
        rows.contains(&(3, 2, 3)),
        "the committed edge 3→2 is read back"
    );
}

/// B, in debug and release alike: the wrong frame is refused and the cube
/// keeps its frame and an empty log.
#[test]
fn commit_under_a_wrong_frame_is_refused() {
    let (table, mut cube, f1, f2, staged) = staged_case();
    assert_ne!(
        f1.revisions(),
        f2.revisions(),
        "fixture: two distinct frames"
    );
    assert_eq!(
        cube.commit(&table, &f1, staged),
        Err(CubeRefusal::FrameMismatch(0))
    );
    assert_eq!(cube.frame().revisions(), f1.revisions());
    assert_eq!(cube.delta_len(0), 0, "nothing was applied");
}
