//! The resident hypercube (plan `d-xlog-native-v1.md` Amendment F) against a
//! scalar oracle and against the per-call free functions, plus the gates that
//! make it a context rather than a faster operator: composing views runs
//! nothing, any pair of relations joins on any columns without a new lookup,
//! and a stale frame is refused.

use std::collections::BTreeMap;

use lance_graph_dt64::dr::{LocalMap, MapRead, MapTable, Width};
use lance_graph_xlog::cube::{Cube, CubeRefusal, View};
use lance_graph_xlog::{bag_count, ids, nonzero, support, Bounds};

const TILE: u32 = 7;

/// r(X, Y), s(Y, Z), t(Z, X): repeated rows, an id with no partner.
const R: [(u32, u32); 6] = [(1, 2), (5, 2), (1, 3), (9, 7), (4, 8), (1, 2)];
const S: [(u32, u32); 5] = [(2, 4), (2, 6), (3, 4), (7, 8), (9, 1)];
const T: [(u32, u32); 4] = [(4, 1), (6, 5), (8, 9), (4, 4)];

fn table_over(rels: &[&[(u32, u32)]]) -> MapTable {
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
        vec![(TILE, Some(LocalMap::prepare(Width::W256, &all).unwrap()))],
    );
    t
}

/// Every pair of matching rows, enumerated.
fn oracle(
    l: &[(u32, u32)],
    lc: u8,
    lb: Option<u32>,
    r: &[(u32, u32)],
    rc: u8,
    rb: Option<u32>,
) -> Vec<(u64, u64, u64)> {
    let col = |t: (u32, u32), c: u8| if c == 0 { t.0 } else { t.1 };
    let mut out = Vec::new();
    for &a in l {
        for &b in r {
            let (ja, fa, jb, fb) = (col(a, lc), col(a, 1 - lc), col(b, rc), col(b, 1 - rc));
            if ja == jb && lb.is_none_or(|v| v == fa) && rb.is_none_or(|v| v == fb) {
                out.push((u64::from(fa), u64::from(ja), u64::from(fb)));
            }
        }
    }
    out.sort_unstable();
    out
}

fn counts(rows: &[(u64, u64, u64)]) -> BTreeMap<u64, i64> {
    let mut m = BTreeMap::new();
    for &(_, y, _) in rows {
        *m.entry(y).or_insert(0) += 1;
    }
    m
}

/// A side of a join: a scan, or a bind on its free column.
fn side(cube: &mut Cube, table: &MapTable, rel: u32, join_col: u8, bound: Option<u32>) -> View {
    let scan = cube.scan(rel).unwrap();
    match bound {
        None => scan,
        Some(v) => cube.bind(table, scan, 1 - join_col, v).unwrap(),
    }
}

#[test]
fn the_cube_answers_what_the_free_functions_answer() {
    let rels: [&[(u32, u32)]; 2] = [&R, &S];
    let table = table_over(&rels);
    let mut cube = Cube::new(&table, &table.seal(), TILE, &rels).unwrap();
    let map = table.map(&table.seal(), TILE).unwrap();

    let mut ys: Vec<u64> = R
        .iter()
        .map(|t| u64::from(t.1))
        .chain(S.iter().map(|t| u64::from(t.0)))
        .collect();
    ys.sort_unstable();
    ys.dedup();
    let y_map = LocalMap::prepare(Width::W64, &ys).unwrap();

    for (left, right) in [
        (None, None),
        (Some(1), None),
        (None, Some(4)),
        (Some(1), Some(6)),
        (Some(99), None),
    ] {
        let l = side(&mut cube, &table, 0, 1, left);
        let r = side(&mut cube, &table, 1, 0, right);
        let j = cube.join(l, 1, r, 0).unwrap();
        let q = Bounds { left, right };

        let free = support(q, &R, &S, &y_map).unwrap();
        assert_eq!(
            ids(map, &cube.support(&table, j).unwrap()),
            ids(&y_map, &free.answer),
            "{q:?}"
        );
        let free = bag_count(q, &R, &S, &y_map).unwrap();
        assert_eq!(
            nonzero(map, &cube.bag_count(&table, j).unwrap()),
            nonzero(&y_map, &free),
            "{q:?}"
        );
    }
}

#[test]
fn any_pair_joins_on_any_columns_without_a_new_lookup() {
    let rels: [&[(u32, u32)]; 3] = [&R, &S, &T];
    let table = table_over(&rels);
    let mut cube = Cube::new(&table, &table.seal(), TILE, &rels).unwrap();
    let map = table.map(&table.seal(), TILE).unwrap();
    let after_intake = map.lookups();
    assert_eq!(after_intake, 2 * (R.len() + S.len() + T.len()) as u64);

    let mut checked = 0;
    for a in 0..3u32 {
        for b in 0..3u32 {
            for lc in 0..2u8 {
                for rc in 0..2u8 {
                    let l = cube.scan(a).unwrap();
                    let r = cube.scan(b).unwrap();
                    let j = cube.join(l, lc, r, rc).unwrap();
                    let want = oracle(rels[a as usize], lc, None, rels[b as usize], rc, None);
                    let got = cube.bag_count(&table, j).unwrap();
                    let got: BTreeMap<u64, i64> = nonzero(map, &got).into_iter().collect();
                    assert_eq!(got, counts(&want), "rel {a}.{lc} = rel {b}.{rc}");
                    let mut rows = cube.materialize_rows(&table, j).unwrap();
                    rows.sort_unstable();
                    assert_eq!(rows, want, "rel {a}.{lc} = rel {b}.{rc}");
                    checked += 1;
                }
            }
        }
    }
    assert_eq!(checked, 36);
    // 36 joins in every orientation: no id was looked up again.
    assert_eq!(map.lookups(), after_intake);
}

#[test]
fn composing_views_runs_nothing() {
    let rels: [&[(u32, u32)]; 3] = [&R, &S, &T];
    let table = table_over(&rels);
    let mut cube = Cube::new(&table, &table.seal(), TILE, &rels).unwrap();
    let mut last = None;
    for i in 0..50u32 {
        let l = side(&mut cube, &table, i % 3, 1, Some(1));
        let r = cube.scan((i + 1) % 3).unwrap();
        last = Some(cube.join(l, 1, r, 0).unwrap());
    }
    assert_eq!(cube.stats().programs, 0);
    assert_eq!(cube.stats().rows_materialized, 0);

    cube.support(&table, last.unwrap()).unwrap();
    assert_eq!(cube.stats().programs, 2, "a support is two programs");
    assert_eq!(
        cube.stats().rows_materialized,
        0,
        "a support builds no rows"
    );
    cube.bag_count(&table, last.unwrap()).unwrap();
    assert_eq!(cube.stats().programs, 4);
    assert_eq!(cube.stats().rows_materialized, 0);
}

#[test]
fn a_bound_outside_the_map_runs_nothing_on_its_side() {
    let rels: [&[(u32, u32)]; 2] = [&R, &S];
    let table = table_over(&rels);
    let mut cube = Cube::new(&table, &table.seal(), TILE, &rels).unwrap();
    let l = side(&mut cube, &table, 0, 1, Some(1000));
    let r = cube.scan(1).unwrap();
    let j = cube.join(l, 1, r, 0).unwrap();
    assert!(cube.support(&table, j).unwrap().iter().all(|w| *w == 0));
    assert_eq!(cube.stats().programs, 1, "only the unbound side ran");
}

/// Randomised: two relations over ids 0..12, every orientation, bound or not.
#[test]
fn differential_against_enumeration() {
    let mut seed = 0x9E37_79B9_7F4A_7C15u64;
    let mut next = |m: u32| {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        (seed % u64::from(m)) as u32
    };
    let mut nonempty = 0;
    for _ in 0..300 {
        let a: Vec<(u32, u32)> = (0..next(20) + 1).map(|_| (next(12), next(12))).collect();
        let b: Vec<(u32, u32)> = (0..next(20) + 1).map(|_| (next(12), next(12))).collect();
        let rels: [&[(u32, u32)]; 2] = [&a, &b];
        let table = table_over(&rels);
        let mut cube = Cube::new(&table, &table.seal(), TILE, &rels).unwrap();
        let map = table.map(&table.seal(), TILE).unwrap();
        let (lc, rc) = (next(2) as u8, next(2) as u8);
        let lb = (next(3) == 0).then(|| next(12));
        let rb = (next(3) == 0).then(|| next(12));
        let l = side(&mut cube, &table, 0, lc, lb);
        let r = side(&mut cube, &table, 1, rc, rb);
        let j = cube.join(l, lc, r, rc).unwrap();
        let want = oracle(&a, lc, lb, &b, rc, rb);
        nonempty += usize::from(!want.is_empty());
        let got: BTreeMap<u64, i64> = nonzero(map, &cube.bag_count(&table, j).unwrap())
            .into_iter()
            .collect();
        assert_eq!(got, counts(&want));
        let sup: Vec<u64> = counts(&want).into_keys().collect();
        assert_eq!(
            ids(map, &cube.support(&table, j).unwrap())
                .into_iter()
                .collect::<std::collections::BTreeSet<_>>(),
            sup.into_iter().collect()
        );
        let mut rows = cube.materialize_rows(&table, j).unwrap();
        rows.sort_unstable();
        assert_eq!(rows, want);
    }
    assert!(nonempty > 150, "differential mostly empty: {nonempty}");
}

#[test]
fn a_stale_frame_a_foreign_view_and_a_bound_join_column_are_refused() {
    let rels: [&[(u32, u32)]; 2] = [&R, &S];
    let mut table = table_over(&rels);
    let mut cube = Cube::new(&table, &table.seal(), TILE, &rels).unwrap();
    let mut other = Cube::new(&table, &table.seal(), TILE, &rels).unwrap();
    let l = cube.scan(0).unwrap();
    let r = cube.scan(1).unwrap();
    let j = cube.join(l, 1, r, 0).unwrap();

    let foreign = other.scan(0).unwrap();
    assert_eq!(cube.join(foreign, 1, r, 0), Err(CubeRefusal::ForeignView));
    let bound = cube.bind(&table, l, 1, 2).unwrap();
    assert_eq!(
        cube.join(bound, 1, r, 0),
        Err(CubeRefusal::BoundOnJoinColumn)
    );
    assert_eq!(cube.support(&table, l), Err(CubeRefusal::NotAJoin));

    table.advance(1, 1, vec![]);
    assert!(matches!(
        cube.support(&table, j),
        Err(CubeRefusal::Map(MapRead::Stale(_)))
    ));
    assert!(matches!(
        cube.bag_count(&table, j),
        Err(CubeRefusal::Map(MapRead::Stale(_)))
    ));
}
