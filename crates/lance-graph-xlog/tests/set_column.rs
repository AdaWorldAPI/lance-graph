//! EXO-394 slice N: `SetExpr::Column`, a relation's column as a set leaf.
//!
//! XLOG lowers `has_in(X) :- edge(_, X).` to `Project(Scan edge, [1])` and
//! `no_in(X) :- node(X), not has_in(X).` to a semi-join over a `Diff`. With a
//! column leaf, both are one set expression evaluated inside one terminal:
//! `node ∧ ¬col(edge, 1)`. The expected answers below are written out by
//! hand from the edge lists, not computed by the cube.

use lance_graph_dt64::dr::{LocalMap, MapRead, MapTable, Width};
use lance_graph_xlog::cube::{Cube, CubeRefusal, SetExpr};

const TILE: u32 = 5;

fn table(ids: &[u32]) -> MapTable {
    let mut all: Vec<u64> = ids.iter().map(|&i| u64::from(i)).collect();
    all.sort_unstable();
    all.dedup();
    let mut t = MapTable::new(1, 1);
    t.advance(
        1,
        1,
        vec![(TILE, Some(LocalMap::prepare(Width::W256, &all).unwrap()))],
    );
    t
}

/// The global ids of a position mask, sorted.
fn ids(t: &MapTable, mask: &[u64]) -> Vec<u64> {
    let map = t.map(&t.seal(), TILE).unwrap();
    let globals = map.globals();
    let mut out: Vec<u64> = (0..globals.len())
        .filter(|&p| mask.get(p / 64).is_some_and(|w| w >> (p % 64) & 1 == 1))
        .map(|p| globals[p])
        .collect();
    out.sort_unstable();
    out
}

/// edge 1->2, 2->3, 4->3; nodes 1..=5. Column 0 = {1,2,4}, column 1 = {2,3}.
fn fixture() -> (MapTable, Cube, SetExpr) {
    let edge: &[(u32, u32)] = &[(1, 2), (2, 3), (4, 3)];
    let t = table(&[1, 2, 3, 4, 5]);
    let mut cube = Cube::new(&t, &t.seal(), TILE, &[edge]).unwrap();
    let node = SetExpr::Set(cube.add_set(&t, &[1, 2, 3, 4, 5]).unwrap());
    (t, cube, node)
}

#[test]
fn a_column_leaf_is_the_columns_distinct_values() {
    let (t, cube, _) = fixture();
    let c0 = cube.set_mask(&t, &SetExpr::column(0, 0)).unwrap();
    let c1 = cube.set_mask(&t, &SetExpr::column(0, 1)).unwrap();
    assert_eq!(ids(&t, &c0), vec![1, 2, 4]);
    assert_eq!(ids(&t, &c1), vec![2, 3]);
}

/// `no_in(X) :- node(X), not has_in(X).` with `has_in(X) :- edge(_, X).`
#[test]
fn stratified_negation_over_a_column_leaf() {
    let (t, cube, node) = fixture();
    let no_in = SetExpr::and_not(node.clone(), SetExpr::column(0, 1));
    assert_eq!(ids(&t, &cube.set_mask(&t, &no_in).unwrap()), vec![1, 4, 5]);
    // Negative twin: negating column 0 instead gives a different answer.
    let no_out = SetExpr::and_not(node, SetExpr::column(0, 0));
    assert_eq!(ids(&t, &cube.set_mask(&t, &no_out).unwrap()), vec![3, 5]);
}

/// `isolated(X) :- node(X), not edge(X, _), not edge(_, X).`
#[test]
fn isolated_is_node_minus_both_columns() {
    let (t, cube, node) = fixture();
    let isolated = SetExpr::and_not(
        SetExpr::and_not(node, SetExpr::column(0, 0)),
        SetExpr::column(0, 1),
    );
    assert_eq!(ids(&t, &cube.set_mask(&t, &isolated).unwrap()), vec![5]);
}

/// The leaf reads the relation's effective edges: a retired edge leaves it.
#[test]
fn a_column_leaf_reads_the_delta_log() {
    let (t, mut cube, _) = fixture();
    cube.retire_edge(&t, 0, 4, 3).unwrap();
    let c0 = cube.set_mask(&t, &SetExpr::column(0, 0)).unwrap();
    assert_eq!(ids(&t, &c0), vec![1, 2]);
    cube.add_edge(&t, 0, 5, 1).unwrap();
    let c1 = cube.set_mask(&t, &SetExpr::column(0, 1)).unwrap();
    assert_eq!(ids(&t, &c1), vec![1, 2, 3]);
}

/// A column leaf composes inside `reach`, as a seed.
#[test]
fn a_column_leaf_seeds_reach() {
    let (t, cube, node) = fixture();
    // Sources (column 0 and not column 1) = {1, 4}; reach along edge = {1,2,3,4}.
    let seed = SetExpr::and_not(SetExpr::column(0, 0), SetExpr::column(0, 1));
    let (mask, _) = cube.reach(&t, 0, &seed, &node).unwrap();
    assert_eq!(ids(&t, &mask), vec![1, 2, 3, 4]);
}

#[test]
fn an_empty_relation_has_an_empty_column() {
    let empty: &[(u32, u32)] = &[];
    let t = table(&[1, 2]);
    let cube = Cube::new(&t, &t.seal(), TILE, &[empty]).unwrap();
    let c = cube.set_mask(&t, &SetExpr::column(0, 1)).unwrap();
    assert!(ids(&t, &c).is_empty());
}

#[test]
fn an_unknown_relation_or_column_is_refused() {
    let (t, cube, _) = fixture();
    assert_eq!(
        cube.set_mask(&t, &SetExpr::column(9, 0)),
        Err(CubeRefusal::UnknownRel(9))
    );
    assert_eq!(
        cube.set_mask(&t, &SetExpr::column(0, 2)),
        Err(CubeRefusal::NoSuchColumn(2))
    );
}

/// A table that never issued the cube's frame is refused.
#[test]
fn a_foreign_table_is_refused() {
    let (_, cube, _) = fixture();
    let alien = table(&[1, 2, 3, 4, 5]);
    assert!(matches!(
        cube.set_mask(&alien, &SetExpr::column(0, 0)),
        Err(CubeRefusal::Map(MapRead::Stale(_)))
    ));
}

/// Advancing the table is a lawful change of frame: the cube's frame stays
/// readable and the column's answer is the same.
#[test]
fn an_advanced_table_keeps_the_column() {
    let (mut t, cube, _) = fixture();
    let before = cube.set_mask(&t, &SetExpr::column(0, 1)).unwrap();
    t.advance(1, 1, vec![]);
    assert_eq!(cube.set_mask(&t, &SetExpr::column(0, 1)).unwrap(), before);
}
