//! Cube safety: a stale or foreign closure snapshot, and a batch committed
//! under the wrong frame, are refused with a typed [`CubeRefusal`] in every
//! build, and a refusal leaves the cube exactly as it was.
//!
//! Each refusal test has a control that passes the same call the correct
//! input, so a guard that refused everything would fail the control, and a
//! guard that refused nothing would fail the refusal.

use lance_graph_dt64::dr::{Frame, LocalMap, MapTable, Width};
use lance_graph_xlog::cube::{Cube, CubeRefusal, EdgeOp, StagedEdges};
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

/// Everything a refusal must leave alone: the frame, every relation's log and
/// base generation, and every relation's effective closure.
fn state(cube: &Cube, table: &MapTable, rels: u32) -> String {
    let per_rel: Vec<String> = (0..rels)
        .map(|r| {
            let (rows, _) = cube.closure_wide(table, r).unwrap();
            format!(
                "{:?} {:?} {:?}",
                cube.delta_log(r),
                cube.base_generation(r),
                pairs(&rows)
            )
        })
        .collect();
    format!("{:?} {:?}", cube.frame().revisions(), per_rel)
}

/// Base `{1→2, 2→3}` over ids `1..=5`.
fn chain() -> (MapTable, Frame, Cube) {
    let (table, frame) = table_with(&[1, 2, 3, 4, 5]);
    let edges = [(1u32, 2u32), (2, 3)];
    let cube = Cube::new_tiled(&table, &frame, &[0], &[edges.as_slice()]).unwrap();
    (table, frame, cube)
}

#[test]
fn a_closure_of_a_folded_base_is_refused() {
    let (table, _frame, mut cube) = chain();
    let stale = cube.base_closure(&table, 0).unwrap();
    assert_eq!(stale.generation(), 0);

    cube.add_edge(&table, 0, 3, 4).unwrap();
    cube.refold(0).unwrap();
    assert_eq!(
        cube.base_generation(0),
        Some(1),
        "a non-empty refold advances"
    );
    cube.add_edge(&table, 0, 4, 5).unwrap();

    let before = state(&cube, &table, 1);
    assert_eq!(
        cube.refresh_closure(&table, 0, &stale),
        Err(CubeRefusal::StaleBase {
            minted: 0,
            current: 1
        })
    );
    let staged = cube.stage(&table, None, &[(0, EdgeOp::Add, 5, 1)]).unwrap();
    assert_eq!(
        cube.stage_closure(0, &stale, &staged).unwrap_err(),
        CubeRefusal::StaleBase {
            minted: 0,
            current: 1
        }
    );
    assert_eq!(state(&cube, &table, 1), before, "a refusal changes nothing");

    // Control: a snapshot of the current base refreshes to the truth.
    let current = cube.base_closure(&table, 0).unwrap();
    let (truth, _) = cube.closure_wide(&table, 0).unwrap();
    let (fresh, _) = cube.refresh_closure(&table, 0, &current).unwrap();
    assert_eq!(pairs(&fresh), pairs(&truth));
    // Anti-vacuity: the stale snapshot would have answered differently
    // (1 reaches 5 only through the folded 3→4).
    assert!(!pairs(stale.rows()).contains(&(0, 4)));
    assert!(pairs(&truth).contains(&(0, 4)));
}

#[test]
fn a_refold_of_an_empty_log_keeps_the_snapshot_valid() {
    let (table, _frame, mut cube) = chain();
    let snap = cube.base_closure(&table, 0).unwrap();
    cube.refold(0).unwrap();
    assert_eq!(cube.base_generation(0), Some(0), "nothing was folded");
    assert!(cube.refresh_closure(&table, 0, &snap).is_ok());
}

#[test]
fn pending_deltas_are_not_certified_as_base() {
    let (table, _frame, mut cube) = chain();
    cube.add_edge(&table, 0, 3, 4).unwrap();
    let base = cube.base_closure(&table, 0).unwrap();
    let (effective, _) = cube.closure_wide(&table, 0).unwrap();
    assert_ne!(
        pairs(base.rows()),
        pairs(&effective),
        "the snapshot certifies the base, not base plus log"
    );
    let (refreshed, _) = cube.refresh_closure(&table, 0, &base).unwrap();
    assert_eq!(pairs(&refreshed), pairs(&effective));
}

#[test]
fn a_closure_of_another_cube_is_refused() {
    let (table, frame, cube) = chain();
    let edges = [(1u32, 2u32), (2, 3)];
    let twin = Cube::new_tiled(&table, &frame, &[0], &[edges.as_slice()]).unwrap();
    let foreign = twin.base_closure(&table, 0).unwrap();
    assert_eq!(
        pairs(foreign.rows()),
        pairs(cube.base_closure(&table, 0).unwrap().rows()),
        "same edges, same closure: only the provenance differs"
    );
    let before = state(&cube, &table, 1);
    assert_eq!(
        cube.refresh_closure(&table, 0, &foreign),
        Err(CubeRefusal::ForeignClosure)
    );
    let staged = cube.stage(&table, None, &[(0, EdgeOp::Add, 3, 4)]).unwrap();
    assert_eq!(
        cube.stage_closure(0, &foreign, &staged).unwrap_err(),
        CubeRefusal::ForeignClosure
    );
    assert_eq!(state(&cube, &table, 1), before);
    let own = cube.base_closure(&table, 0).unwrap();
    assert!(cube.refresh_closure(&table, 0, &own).is_ok());
}

#[test]
fn a_closure_of_another_relation_is_refused() {
    let (table, frame) = table_with(&[1, 2, 3, 4, 5]);
    let r0 = [(1u32, 2u32)];
    let r1 = [(1u32, 2u32)];
    let cube = Cube::new_tiled(&table, &frame, &[0], &[r0.as_slice(), r1.as_slice()]).unwrap();
    let of_r1 = cube.base_closure(&table, 1).unwrap();
    assert_eq!(
        pairs(of_r1.rows()),
        pairs(cube.base_closure(&table, 0).unwrap().rows()),
        "same edges: only the relation differs"
    );
    assert_eq!(
        cube.refresh_closure(&table, 0, &of_r1),
        Err(CubeRefusal::ClosureRelation {
            expected: 0,
            found: 1
        })
    );
    assert!(cube.refresh_closure(&table, 1, &of_r1).is_ok());
}

#[test]
fn a_closure_under_a_left_frame_is_refused() {
    let (mut table, f1) = table_with(&[1, 2, 3]);
    let edges = [(1u32, 2u32), (2, 3)];
    let mut cube = Cube::new(&table, &f1, 0, &[edges.as_slice()]).unwrap();
    let snap = cube.base_closure(&table, 0).unwrap();
    table.advance(
        2,
        1,
        vec![(
            0,
            Some(LocalMap::prepare(Width::W64, &[1, 2, 3, 4]).unwrap()),
        )],
    );
    let f2 = table.seal();
    cube.rebase(&table, &f2).unwrap();
    assert_eq!(
        cube.base_generation(0),
        Some(0),
        "rebase keeps the base generation: only the frame check can refuse"
    );
    assert_eq!(
        cube.refresh_closure(&table, 0, &snap),
        Err(CubeRefusal::ClosureFrame)
    );
    let fresh = cube.base_closure(&table, 0).unwrap();
    assert!(cube.refresh_closure(&table, 0, &fresh).is_ok());
}

/// A single-tile cube over ids `[1, 2]` with edge `1→2`, and a batch staged
/// with the grown map `[1, 2, 3]` and the edge `3→2`. The table is not yet
/// advanced.
fn grown_batch() -> (MapTable, Frame, Cube, StagedEdges, LocalMap) {
    let (table, f1) = table_with(&[1, 2]);
    let edges = [(1u32, 2u32)];
    let cube = Cube::new(&table, &f1, 0, &[edges.as_slice()]).unwrap();
    let grown = LocalMap::prepare(Width::W64, &[1, 2, 3]).unwrap();
    let staged = cube
        .stage(&table, Some((0, &grown)), &[(0, EdgeOp::Add, 3, 2)])
        .unwrap();
    (table, f1, cube, staged, grown)
}

#[test]
fn a_commit_under_the_staged_map_is_accepted() {
    let (mut table, _f1, mut cube, staged, grown) = grown_batch();
    assert_eq!(cube.check_commit(&staged, Some((0, &grown))), Ok(()));
    table.advance(2, 1, vec![(0, Some(grown))]);
    let f2 = table.seal();
    assert_eq!(cube.commit(&table, &f2, staged), Ok(()));
    assert_eq!(cube.frame().revisions(), f2.revisions());
    let l = cube.scan(0).unwrap();
    let r = cube.scan(0).unwrap();
    let j = cube.join(l, 1, r, 1).unwrap();
    let rows = cube.materialize_rows(&table, j).unwrap();
    assert!(
        rows.contains(&(3, 2, 3)),
        "the committed edge 3→2 reads back"
    );
}

#[test]
fn a_commit_under_the_old_frame_is_refused() {
    let (mut table, f1, mut cube, staged, grown) = grown_batch();
    table.advance(2, 1, vec![(0, Some(grown))]);
    let before = state(&cube, &table, 1);
    assert_eq!(
        cube.commit(&table, &f1, staged),
        Err(CubeRefusal::FrameMismatch(0))
    );
    assert_eq!(state(&cube, &table, 1), before);
    assert_eq!(cube.frame().revisions(), f1.revisions());
    assert_eq!(cube.delta_len(0), 0, "nothing was logged");
}

#[test]
fn a_commit_under_a_same_length_different_map_is_refused() {
    let (mut table, f1, mut cube, staged, _grown) = grown_batch();
    let other = LocalMap::prepare(Width::W64, &[1, 2, 4]).unwrap();
    assert_eq!(other.view().len(), 3, "same length as the staged map");
    assert_eq!(
        cube.check_commit(&staged, Some((0, &other))),
        Err(CubeRefusal::FrameMismatch(0))
    );
    table.advance(2, 1, vec![(0, Some(other))]);
    let f2 = table.seal();
    let before = state(&cube, &table, 1);
    assert_eq!(
        cube.commit(&table, &f2, staged),
        Err(CubeRefusal::FrameMismatch(0))
    );
    assert_eq!(state(&cube, &table, 1), before);
    assert_eq!(cube.frame().revisions(), f1.revisions());
}

#[test]
fn a_batch_of_another_cube_is_refused() {
    let (mut table, _f1, cube, staged, grown) = grown_batch();
    let edges = [(1u32, 2u32)];
    let (t0, f0) = table_with(&[1, 2]);
    let mut other = Cube::new(&t0, &f0, 0, &[edges.as_slice()]).unwrap();
    assert_eq!(
        other.check_commit(&staged, Some((0, &grown))),
        Err(CubeRefusal::ForeignBatch)
    );
    table.advance(2, 1, vec![(0, Some(grown))]);
    let f2 = table.seal();
    let before = state(&other, &t0, 1);
    assert_eq!(
        other.commit(&table, &f2, staged),
        Err(CubeRefusal::ForeignBatch)
    );
    assert_eq!(state(&other, &t0, 1), before);
    drop(cube);
}

#[test]
fn a_batch_staged_before_a_frame_change_is_refused() {
    let (mut table, _f1, mut cube, staged, grown) = grown_batch();
    let late = cube.stage(&table, None, &[(0, EdgeOp::Add, 2, 1)]).unwrap();
    table.advance(2, 1, vec![(0, Some(grown))]);
    let f2 = table.seal();
    cube.commit(&table, &f2, staged).unwrap();
    assert_eq!(cube.check_commit(&late, None), Err(CubeRefusal::StaleBatch));
    let before = state(&cube, &table, 1);
    assert_eq!(cube.commit(&table, &f2, late), Err(CubeRefusal::StaleBatch));
    assert_eq!(state(&cube, &table, 1), before);
    // Control: the same edge staged under the cube's current frame commits.
    let now = cube.stage(&table, None, &[(0, EdgeOp::Add, 2, 1)]).unwrap();
    assert_eq!(cube.commit(&table, &f2, now), Ok(()));
}
