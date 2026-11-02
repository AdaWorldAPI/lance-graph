//! F7: a cube keeps answering after the map table advances.
//!
//! The table is a versioned append-only index, so a cube's own frame reads the
//! layout it was built under. Without `rebase` the cube answers exactly as
//! before; after a `rebase` onto an extending frame it answers over the
//! extended map.
//!
//! Disable it catches: restoring current-frame-only reads in dt64's
//! `MapTable::map` (a frame older than the table's state is refused as
//! stale) turns `an_old_frame_keeps_answering_without_rebase` red.

use lance_graph_dt64::dr::{LocalMap, MapTable, Width};
use lance_graph_xlog::cube::{Cube, CubeRefusal, EdgeOp};
use lance_graph_xlog::wide::WideRows;

const TILE: u32 = 7;
const EDGES: [(u32, u32); 4] = [(1, 2), (2, 3), (3, 4), (4, 1)];
const R: [(u32, u32); 3] = [(1, 2), (2, 3), (3, 4)];
const S: [(u32, u32); 3] = [(2, 3), (3, 4), (4, 5)];

fn base_table() -> MapTable {
    let mut t = MapTable::new(1, 1);
    t.advance(
        1,
        1,
        vec![(
            TILE,
            Some(LocalMap::prepare(Width::W256, &[1, 2, 3, 4, 5]).unwrap()),
        )],
    );
    t
}

fn extend(t: &mut MapTable) {
    t.advance(
        2,
        2,
        vec![(
            TILE,
            Some(LocalMap::prepare(Width::W256, &[1, 2, 3, 4, 5, 6, 7]).unwrap()),
        )],
    );
}

#[test]
fn an_old_frame_keeps_answering_without_rebase() {
    let mut table = base_table();
    let f1 = table.seal();
    let rels: [&[(u32, u32)]; 2] = [&R, &S];
    let mut cube = Cube::new(&table, &f1, TILE, &rels).unwrap();
    let l = cube.scan(0).unwrap();
    let r = cube.scan(1).unwrap();
    let j = cube.join(l, 1, r, 0).unwrap();
    let support = cube.support(&table, j).unwrap();
    let bag = cube.bag_count(&table, j).unwrap();
    assert!(support.iter().any(|w| *w != 0), "fixture must be non-empty");

    extend(&mut table);
    assert_eq!(cube.support(&table, j).unwrap(), support);
    assert_eq!(cube.bag_count(&table, j).unwrap(), bag);
}

fn dense(rows: &WideRows, x: u32) -> Vec<u64> {
    let mut v = vec![0u64; rows.blocks() as usize * 4];
    rows.row_into(x, &mut v);
    v
}

#[test]
fn an_old_frame_keeps_answering_the_wide_closure_without_rebase() {
    let mut table = base_table();
    let f1 = table.seal();
    let cube = Cube::new_tiled(&table, &f1, &[TILE], &[&EDGES]).unwrap();
    let (before, _) = cube.closure_wide(&table, 0).unwrap();
    let want: Vec<Vec<u64>> = (0..before.n()).map(|x| dense(&before, x)).collect();
    assert!(want.iter().any(|r| r.iter().any(|w| *w != 0)));

    extend(&mut table);
    let (after, _) = cube.closure_wide(&table, 0).unwrap();
    assert_eq!(after.n(), before.n());
    for x in 0..after.n() {
        assert_eq!(dense(&after, x), want[x as usize], "row {x}");
    }
}

#[test]
fn rebase_onto_an_extension_answers_over_the_extended_map() {
    let mut table = base_table();
    let f1 = table.seal();
    let rels: [&[(u32, u32)]; 2] = [&R, &S];
    let mut cube = Cube::new(&table, &f1, TILE, &rels).unwrap();
    let l = cube.scan(0).unwrap();
    let r = cube.scan(1).unwrap();
    let j = cube.join(l, 1, r, 0).unwrap();
    let before = cube.support(&table, j).unwrap();

    extend(&mut table);
    let f2 = table.seal();
    cube.rebase(&table, &f2).unwrap();
    let after = cube.support(&table, j).unwrap();
    // Same word count (both maps fit one word), the same positions set: the
    // extension appended ids 6 and 7 without renumbering.
    assert_eq!(before, after);
    // The new ids are addressable now: a derived relation over id 7 is accepted.
    assert!(cube.add_relation(&table, &[(6, 7)]).is_ok());
}

/// One relation `[(1, 2)]` over the single tile `0` mapping `[1, 2]`, and a
/// view `R ⋈ (R with column 0 bound to 3)` on column 1. Id 3 is unmapped, so
/// the bound side stores no coordinate.
fn bound_to_unmapped() -> (MapTable, Cube, lance_graph_xlog::cube::View) {
    let mut table = MapTable::new(1, 1);
    table.advance(
        1,
        1,
        vec![(0, Some(LocalMap::prepare(Width::W64, &[1, 2]).unwrap()))],
    );
    let frame = table.seal();
    let edges = [(1u32, 2u32)];
    let mut cube = Cube::new(&table, &frame, 0, &[edges.as_slice()]).unwrap();
    let scan = cube.scan(0).unwrap();
    let bound = cube.bind(&table, scan, 0, 3).unwrap();
    let scan2 = cube.scan(0).unwrap();
    let join = cube.join(bound, 1, scan2, 1).unwrap();
    (table, cube, join)
}

/// The map extended to `[1, 2, 3]` under a new frame.
fn extend_to_3(table: &mut MapTable) -> lance_graph_dt64::dr::Frame {
    table.advance(
        2,
        1,
        vec![(0, Some(LocalMap::prepare(Width::W64, &[1, 2, 3]).unwrap()))],
    );
    table.seal()
}

/// Codex P2 on #1923. Red if a view bound to an unmapped id keeps answering
/// "nothing" after an extending rebase places that id (disable: drop the
/// `apply_rebind` call in `Cube::rebase`). The old view must answer exactly
/// what a view built after the rebase answers; the bind means "column 0 is
/// id 3" under every frame.
#[test]
fn a_bind_to_an_unmapped_id_is_placed_by_an_extending_rebase() {
    let (mut table, mut cube, old_join) = bound_to_unmapped();
    // Before: id 3 is unmapped, so the bound side reaches nothing.
    assert!(cube
        .support(&table, old_join)
        .unwrap()
        .iter()
        .all(|&w| w == 0));
    let f2 = extend_to_3(&mut table);
    cube.rebase(&table, &f2).unwrap();
    cube.add_edge(&table, 0, 3, 2).unwrap();
    let scan = cube.scan(0).unwrap();
    let bound = cube.bind(&table, scan, 0, 3).unwrap();
    let scan2 = cube.scan(0).unwrap();
    let new_join = cube.join(bound, 1, scan2, 1).unwrap();
    let fresh = cube.support(&table, new_join).unwrap();
    // Anti-vacuity: the fresh view does see the new edge (id 2 is local 1).
    assert_eq!(fresh[0] >> 1 & 1, 1, "the fresh view reaches id 2");
    assert_eq!(cube.support(&table, old_join).unwrap(), fresh);
    assert_eq!(
        cube.bag_count(&table, old_join).unwrap(),
        cube.bag_count(&table, new_join).unwrap()
    );
}

/// The same through `Cube::stage` + `Cube::commit` (the refresh path). Red if
/// `commit` does not apply the staged rebinds (disable: drop the
/// `apply_rebind` call in `Cube::commit`).
#[test]
fn a_bind_to_an_unmapped_id_is_placed_by_a_committed_batch() {
    let (mut table, mut cube, old_join) = bound_to_unmapped();
    let grown = LocalMap::prepare(Width::W64, &[1, 2, 3]).unwrap();
    let staged = cube
        .stage(&table, Some((0, &grown)), &[(0, EdgeOp::Add, 3, 2)])
        .unwrap();
    let f2 = extend_to_3(&mut table);
    cube.commit(&table, &f2, staged);
    let s = cube.support(&table, old_join).unwrap();
    assert_eq!(s[0] >> 1 & 1, 1, "the old view reaches id 2 through (3, 2)");
}

/// A mapped bind keeps its meaning: binding to id 1 before the rebase and
/// after it answers the same (twin of the two tests above: re-resolution
/// touches only binds that were unmapped).
#[test]
fn a_bind_to_a_mapped_id_keeps_its_answer_across_a_rebase() {
    let (mut table, mut cube, _) = bound_to_unmapped();
    let scan = cube.scan(0).unwrap();
    let bound = cube.bind(&table, scan, 0, 1).unwrap();
    let scan2 = cube.scan(0).unwrap();
    let join = cube.join(bound, 1, scan2, 1).unwrap();
    let before = cube.support(&table, join).unwrap();
    assert_eq!(before[0] >> 1 & 1, 1, "bound to 1 reaches 2");
    let f2 = extend_to_3(&mut table);
    cube.rebase(&table, &f2).unwrap();
    assert_eq!(cube.support(&table, join).unwrap(), before);
}

/// Red if `Cube::stage` changes the cube or the table (disable: let `stage`
/// log the edges it places). Staging issues no version and logs nothing;
/// only `commit` does.
#[test]
fn staging_changes_nothing() {
    let (table, cube, _) = bound_to_unmapped();
    let v = table.current_version();
    let grown = LocalMap::prepare(Width::W64, &[1, 2, 3]).unwrap();
    let staged = cube
        .stage(&table, Some((0, &grown)), &[(0, EdgeOp::Add, 3, 2)])
        .unwrap();
    assert_eq!(staged.len(), 1);
    assert_eq!(table.current_version(), v);
    assert_eq!(cube.delta_len(0), 0);
    // A non-extending map is refused at staging, before anything moves.
    let swapped = LocalMap::prepare(Width::W64, &[2, 1, 3]).unwrap();
    assert!(matches!(
        cube.stage(&table, Some((0, &swapped)), &[]),
        Err(CubeRefusal::Renumbered(0))
    ));
    // An id outside the staged layout is refused at staging too.
    assert!(cube.stage(&table, None, &[(0, EdgeOp::Add, 9, 1)]).is_err());
}
