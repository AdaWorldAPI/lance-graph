//! Column counts and join sizes of the resident hypercube against plain
//! enumeration. The cube counts every relation and both columns over one set
//! of positions, so the count vectors of any two (relation, column) pairs
//! line up and give a join's size without building a tuple.

use std::collections::BTreeMap;

use lance_graph_dt64::dr::{LocalMap, MapRead, MapTable, Width};
use lance_graph_xlog::cube::{Cube, CubeRefusal};
use lance_graph_xlog::nonzero;

const TILE: u32 = 7;

/// r(X, Y), s(Y, Z), t(Z, X): repeated rows, a skewed key (1), an id with no
/// partner.
const R: [(u32, u32); 7] = [(1, 2), (5, 2), (1, 3), (9, 7), (4, 8), (1, 2), (1, 6)];
const S: [(u32, u32); 5] = [(2, 4), (2, 6), (3, 4), (7, 8), (9, 1)];
const T: [(u32, u32); 4] = [(4, 1), (6, 5), (8, 9), (4, 4)];
const RELS: [&[(u32, u32)]; 3] = [&R, &S, &T];

fn table_over(rels: &[&[(u32, u32)]], width: Width) -> MapTable {
    let mut all: Vec<u64> = rels
        .iter()
        .flat_map(|r| r.iter().flat_map(|&(a, b)| [u64::from(a), u64::from(b)]))
        .collect();
    all.sort_unstable();
    all.dedup();
    let mut t = MapTable::new(1, 1);
    t.advance(
        1,
        1,
        vec![(TILE, Some(LocalMap::prepare(width, &all).unwrap()))],
    );
    t
}

fn col(t: (u32, u32), c: u8) -> u32 {
    if c == 0 {
        t.0
    } else {
        t.1
    }
}

/// Occurrences of every id in one column, by enumeration.
fn count_ids(rel: &[(u32, u32)], c: u8) -> BTreeMap<u64, i64> {
    let mut m = BTreeMap::new();
    for &t in rel {
        *m.entry(u64::from(col(t, c))).or_insert(0) += 1;
    }
    m
}

/// Matching row pairs of `l.lc == r.rc`, by enumeration.
fn pairs(l: &[(u32, u32)], lc: u8, r: &[(u32, u32)], rc: u8) -> i128 {
    let mut n = 0;
    for &a in l {
        for &b in r {
            if col(a, lc) == col(b, rc) {
                n += 1;
            }
        }
    }
    n
}

/// Every (relation, column) count vector equals the enumeration, id by id.
#[test]
fn column_counts_equal_enumeration_for_every_relation_and_column() {
    let table = table_over(&RELS, Width::W256);
    let mut cube = Cube::new(&table, &table.seal(), TILE, &RELS).unwrap();
    let map = table.map(&table.seal(), TILE).unwrap();
    let mut compared = 0;
    for (rel, tuples) in RELS.iter().enumerate() {
        for c in 0..2u8 {
            let scan = cube.scan(rel as u32).unwrap();
            let got = cube.column_counts(&table, scan, c).unwrap();
            let want: Vec<(u64, i64)> = count_ids(tuples, c).into_iter().collect();
            let mut got_ids = nonzero(&map, &got);
            got_ids.sort_unstable();
            assert_eq!(got_ids, want, "relation {rel} column {c}");
            assert_eq!(
                got.iter().sum::<i64>(),
                tuples.len() as i64,
                "relation {rel} column {c}: the counts must add up to the rows"
            );
            compared += 1;
        }
    }
    assert_eq!(compared, 6, "three relations, two columns each");
}

/// The size of every join between any two (relation, column) pairs equals the
/// number of matching row pairs, and equals the product of the two count
/// vectors summed over positions. 36 joins from 6 count vectors.
#[test]
fn join_size_is_the_dot_product_of_two_count_vectors() {
    let table = table_over(&RELS, Width::W256);
    let mut cube = Cube::new(&table, &table.seal(), TILE, &RELS).unwrap();
    let mut nonzero_joins = 0;
    let mut zero_joins = 0;
    for (lr, l) in RELS.iter().enumerate() {
        for lc in 0..2u8 {
            for (rr, r) in RELS.iter().enumerate() {
                for rc in 0..2u8 {
                    let ls = cube.scan(lr as u32).unwrap();
                    let rs = cube.scan(rr as u32).unwrap();
                    let j = cube.join(ls, lc, rs, rc).unwrap();
                    let want = pairs(l, lc, r, rc);
                    let what = format!("{lr}.{lc} = {rr}.{rc}");
                    assert_eq!(cube.join_size(&table, j).unwrap(), want, "{what}");
                    let a = cube.column_counts(&table, ls, lc).unwrap();
                    let b = cube.column_counts(&table, rs, rc).unwrap();
                    let dot: i128 = a
                        .iter()
                        .zip(&b)
                        .map(|(&x, &y)| i128::from(x) * i128::from(y))
                        .sum();
                    assert_eq!(dot, want, "{what}: dot product of the count vectors");
                    if want == 0 {
                        zero_joins += 1;
                    } else {
                        nonzero_joins += 1;
                    }
                }
            }
        }
    }
    assert!(
        nonzero_joins >= 20 && zero_joins >= 1,
        "fixture must hold both joining and non-joining pairs: {nonzero_joins} / {zero_joins}"
    );
}

/// The size equals the tuples `materialize_rows` builds, and asking for the
/// size builds none.
#[test]
fn join_size_equals_materialized_rows_without_building_any() {
    let table = table_over(&RELS, Width::W256);
    let mut cube = Cube::new(&table, &table.seal(), TILE, &RELS).unwrap();
    let (r, s) = (cube.scan(0).unwrap(), cube.scan(1).unwrap());
    let j = cube.join(r, 1, s, 0).unwrap();
    let before = cube.stats();
    let size = cube.join_size(&table, j).unwrap();
    let after = cube.stats();
    assert_eq!(
        after.rows_materialized, before.rows_materialized,
        "no tuple may be built"
    );
    assert_eq!(
        after.programs - before.programs,
        2,
        "one count program per side"
    );
    let rows = cube.materialize_rows(&table, j).unwrap();
    assert_eq!(size, rows.len() as i128);
    assert!(size > 0, "fixture: r and s must join");
}

/// A bound on the other column restricts the counts to the matching rows.
#[test]
fn a_bound_side_counts_only_its_rows() {
    let table = table_over(&RELS, Width::W256);
    let mut cube = Cube::new(&table, &table.seal(), TILE, &RELS).unwrap();
    let map = table.map(&table.seal(), TILE).unwrap();
    let scan = cube.scan(0).unwrap();
    // r rows with X = 1, counted by Y.
    let bound = cube.bind(&table, scan, 0, 1).unwrap();
    let got = nonzero(&map, &cube.column_counts(&table, bound, 1).unwrap());
    let mut want = BTreeMap::new();
    for &(x, y) in &R {
        if x == 1 {
            *want.entry(u64::from(y)).or_insert(0i64) += 1;
        }
    }
    let mut got_sorted = got;
    got_sorted.sort_unstable();
    assert_eq!(got_sorted, want.into_iter().collect::<Vec<_>>());
    assert_eq!(
        cube.column_counts(&table, bound, 0),
        Err(CubeRefusal::BoundOnJoinColumn),
        "counting the bound column itself is refused"
    );
    // A value outside the map binds to nothing: every count is zero.
    let nothing = cube.bind(&table, scan, 0, 4242).unwrap();
    assert!(cube
        .column_counts(&table, nothing, 1)
        .unwrap()
        .iter()
        .all(|&c| c == 0));
}

/// The largest count names the heaviest coordinate of a column. Which
/// POSITION it sits at depends on the map; which ID it is does not.
#[test]
fn the_heaviest_id_is_the_same_under_two_different_maps() {
    let heaviest = |width: Width, extra: &[(u32, u32)]| -> (u64, i64) {
        // `extra` only adds ids to the map, so positions shift between runs.
        let with_extra: [&[(u32, u32)]; 4] = [&R, &S, &T, extra];
        let table = table_over(&with_extra, width);
        let mut cube = Cube::new(&table, &table.seal(), TILE, &with_extra).unwrap();
        let map = table.map(&table.seal(), TILE).unwrap();
        let scan = cube.scan(0).unwrap();
        let counts = cube.column_counts(&table, scan, 0).unwrap();
        nonzero(&map, &counts)
            .into_iter()
            .max_by_key(|&(id, c)| (c, std::cmp::Reverse(id)))
            .unwrap()
    };
    let a = heaviest(Width::W256, &[]);
    let b = heaviest(Width::W256, &[(0, 0), (100, 200)]);
    assert_eq!(a, (1, 4), "id 1 occurs four times in r.X");
    assert_eq!(
        a, b,
        "the heaviest id must not depend on where the map puts it"
    );
}

/// Refusals: a join passed as a side, a column past 1, a table that never
/// issued the cube's frame. Advancing the table does NOT refuse: the table is
/// a versioned index, so the cube's frame keeps reading the layout it was
/// sealed under and the counts stay identical.
#[test]
fn foreign_frames_and_wrong_views_are_refused() {
    let mut table = table_over(&RELS, Width::W256);
    let mut cube = Cube::new(&table, &table.seal(), TILE, &RELS).unwrap();
    let (r, s) = (cube.scan(0).unwrap(), cube.scan(1).unwrap());
    let j = cube.join(r, 1, s, 0).unwrap();
    assert_eq!(cube.column_counts(&table, j, 0), Err(CubeRefusal::NotASide));
    assert_eq!(
        cube.column_counts(&table, r, 2),
        Err(CubeRefusal::NoSuchColumn(2))
    );
    assert_eq!(cube.join_size(&table, r), Err(CubeRefusal::NotAJoin));

    let (counts, size) = (
        cube.column_counts(&table, r, 0).unwrap(),
        cube.join_size(&table, j).unwrap(),
    );
    let ids: Vec<u64> = (0..12).collect();
    table.advance(
        2,
        2,
        vec![(TILE, Some(LocalMap::prepare(Width::W256, &ids).unwrap()))],
    );
    assert_eq!(cube.column_counts(&table, r, 0), Ok(counts));
    assert_eq!(cube.join_size(&table, j), Ok(size));

    let alien = table_over(&RELS, Width::W256);
    assert!(matches!(
        cube.column_counts(&alien, r, 0),
        Err(CubeRefusal::Map(MapRead::Stale(_)))
    ));
    assert!(matches!(
        cube.join_size(&alien, j),
        Err(CubeRefusal::Map(MapRead::Stale(_)))
    ));
}
