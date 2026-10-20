//! The versioned relationship table (`join`) against nested-loop oracles.

use mork_pathmap_probe::join::{
    JoinExec, JoinRefusal, JoinTable, R_KEY, S_KEY, filter_plane, for_each_set,
};
use mork_pathmap_probe::{JoinInput, RelationView, Semantics, reference};

fn view(rows: &[[u32; 2]], generation: u64) -> RelationView<'_> {
    RelationView { rows, generation }
}

/// Deterministic rows with ids spread over the whole `u32` range: the two
/// datasets share a known fraction of `y` values and nothing else.
fn hetero(n: usize, seed: u64, ys: &[u32], fan: usize) -> Vec<[u32; 2]> {
    let mut s = seed;
    let mut next = || {
        s = s.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
        (s >> 33) as u32
    };
    (0..n)
        .flat_map(|_| {
            let y = ys[next() as usize % ys.len()];
            let a = next();
            (0..fan).map(move |k| [a.wrapping_add(k as u32), y])
        })
        .collect()
}

fn swap(rows: &[[u32; 2]]) -> Vec<[u32; 2]> {
    rows.iter().map(|&[a, b]| [b, a]).collect()
}

fn ys(n: u32, seed: u32) -> Vec<u32> {
    (0..n)
        .map(|i| i.wrapping_mul(2_654_435_761).wrapping_add(seed))
        .collect()
}

#[test]
fn heterogeneous_join_matches_the_oracle_in_both_semantics() {
    let all = ys(64, 7);
    // R uses the first 48 y values, S the last 48: 32 shared.
    let r = hetero(400, 1, &all[..48], 1);
    let s = swap(&hetero(300, 2, &all[16..], 2));
    let inp = JoinInput {
        r: view(&r, 3),
        s: view(&s, 11),
    };
    let table = JoinTable::prepare(inp.r, inp.s);
    let mut exec = JoinExec::new(&table);
    for sem in [Semantics::Set, Semantics::Bag] {
        let want = reference::answers(&inp, sem);
        assert!(!want.rows.is_empty());
        assert_eq!(exec.answers(&table, (3, 11), None, sem).unwrap(), want);
    }
    // Not degenerate: some y are in one relation only, and are excluded.
    assert!(table.domain.len() > 48);
    let bag = reference::answers(&inp, Semantics::Bag).total();
    assert_eq!(exec.bag_count(&table, &[3, 11], None).unwrap(), bag);
}

#[test]
fn a_star_of_three_relations_counts_like_the_oracle() {
    let all = ys(40, 3);
    let r = hetero(200, 4, &all[..30], 1);
    let s = swap(&hetero(150, 5, &all[10..], 1));
    let t = swap(&hetero(120, 6, &all[5..35], 3));
    let table = JoinTable::prepare_star(&[
        (view(&r, 1), R_KEY),
        (view(&s, 1), S_KEY),
        (view(&t, 1), S_KEY),
    ]);
    let mut exec = JoinExec::new(&table);
    let mut want = 0u64;
    for &[_, y] in &r {
        let a = s.iter().filter(|row| row[0] == y).count() as u64;
        let b = t.iter().filter(|row| row[0] == y).count() as u64;
        want += a * b;
    }
    assert!(want > 0);
    assert_eq!(exec.bag_count(&table, &[1, 1, 1], None).unwrap(), want);
    // The third relation narrows the joint: dropping it gives a different count.
    let two = JoinTable::prepare_star(&[(view(&r, 1), R_KEY), (view(&s, 1), S_KEY)]);
    let mut exec2 = JoinExec::new(&two);
    assert_ne!(exec2.bag_count(&two, &[1, 1], None).unwrap(), want);
}

#[test]
fn a_filter_plane_restricts_the_join_to_the_thoughts_slots() {
    let all = ys(32, 9);
    let r = hetero(200, 7, &all, 1);
    let s = swap(&hetero(200, 8, &all, 1));
    let inp = JoinInput {
        r: view(&r, 1),
        s: view(&s, 1),
    };
    let table = JoinTable::prepare(inp.r, inp.s);
    let mut exec = JoinExec::new(&table);
    let keep = |y: u32| y.is_multiple_of(3);
    let filter = filter_plane(&table, keep);
    let got = exec
        .answers(&table, (1, 1), Some(&filter), Semantics::Bag)
        .unwrap();
    let mut want = reference::answers(&inp, Semantics::Bag);
    want.rows.retain(|row, _| keep(row[1]));
    assert!(!want.rows.is_empty());
    assert_eq!(got, want);
    // Silence twin: the filter removes something.
    assert!(want.total() < reference::answers(&inp, Semantics::Bag).total());
    // A plane over another slot space is refused, not read past its end.
    assert!(matches!(
        exec.answers(&table, (1, 1), Some(&[0u64; 1000]), Semantics::Set),
        Err(JoinRefusal::FilterShape { .. })
    ));
}

#[test]
fn each_side_is_refused_against_its_own_generation() {
    let r = [[1, 10], [2, 11]];
    let s = [[10, 5], [11, 6]];
    let table = JoinTable::prepare(view(&r, 4), view(&s, 9));
    let mut exec = JoinExec::new(&table);
    assert!(exec.answers(&table, (4, 9), None, Semantics::Set).is_ok());
    assert_eq!(
        exec.answers(&table, (5, 9), None, Semantics::Set),
        Err(JoinRefusal::Stale {
            side: 0,
            bound: 4,
            current: 5
        })
    );
    assert_eq!(
        exec.answers(&table, (4, 8), None, Semantics::Set),
        Err(JoinRefusal::Stale {
            side: 1,
            bound: 9,
            current: 8
        })
    );
}

#[test]
fn replacing_one_side_equals_preparing_the_table_again() {
    let all = ys(48, 1);
    let r = hetero(300, 1, &all, 1);
    let s1 = swap(&hetero(300, 2, &all, 2));
    let s2 = swap(&hetero(250, 3, &all[..40], 1));
    let mut table = JoinTable::prepare(view(&r, 1), view(&s1, 1));
    table.replace_side(1, view(&s2, 2), S_KEY).unwrap();
    let mut exec = JoinExec::new(&table);
    let got = exec.answers(&table, (1, 2), None, Semantics::Bag).unwrap();
    let inp = JoinInput {
        r: view(&r, 1),
        s: view(&s2, 2),
    };
    assert_eq!(got, reference::answers(&inp, Semantics::Bag));
    // The old S answer differs, so the replacement did something.
    let old = JoinInput {
        r: view(&r, 1),
        s: view(&s1, 1),
    };
    assert_ne!(got, reference::answers(&old, Semantics::Bag));
}

#[test]
fn a_side_leaving_the_domain_is_refused() {
    let r = [[1, 10]];
    let s = [[10, 5]];
    let mut table = JoinTable::prepare(view(&r, 1), view(&s, 1));
    let s2 = [[10, 5], [77, 6]];
    assert_eq!(
        table.replace_side(1, view(&s2, 2), S_KEY),
        Err(JoinRefusal::OutsideDomain { side: 1, y: 77 })
    );
    // The table is unchanged: the old generation still answers.
    let mut exec = JoinExec::new(&table);
    assert!(exec.answers(&table, (1, 1), None, Semantics::Set).is_ok());
}

#[test]
fn the_joint_plane_has_exactly_the_shared_slots() {
    let r = [[0, 1], [0, 3], [0, 5]];
    let s = [[3, 0], [4, 0], [5, 0]];
    let table = JoinTable::prepare(view(&r, 1), view(&s, 1));
    let mut exec = JoinExec::new(&table);
    let mut shared = Vec::new();
    let n = table.domain.len();
    let joint = exec.joint(&table, None).unwrap().to_vec();
    for_each_set(&joint, n, |slot| shared.push(table.domain.id(slot)));
    assert_eq!(shared, vec![3, 5]);
}

/// The joint plane is itself an answer: the `y` values that every relation of
/// the star has (the existence half of the join). Enumeration and counts do not
/// need it to be exact, since an empty side emits nothing at its slot; this
/// test is what pins it.
#[test]
fn the_star_joint_is_the_existence_mask() {
    let r = [[0, 1], [0, 2], [0, 3], [0, 4]];
    let s = [[2, 0], [3, 0], [4, 0], [5, 0]];
    let t = [[3, 0], [4, 0], [6, 0]];
    let table = JoinTable::prepare_star(&[
        (view(&r, 1), R_KEY),
        (view(&s, 1), S_KEY),
        (view(&t, 1), S_KEY),
    ]);
    let mut exec = JoinExec::new(&table);
    let n = table.domain.len();
    let joint = exec.joint(&table, None).unwrap().to_vec();
    let mut shared = Vec::new();
    for_each_set(&joint, n, |slot| shared.push(table.domain.id(slot)));
    assert_eq!(shared, vec![3, 4]);
}
