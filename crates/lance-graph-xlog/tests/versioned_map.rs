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
use lance_graph_xlog::cube::Cube;
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
