//! The versioned relationship table (`join`) against nested-loop oracles.

use mork_pathmap_probe::join::{
    JoinExec, JoinRefusal, JoinTable, R_KEY, S_KEY, filter_plane, for_each_set,
};
use mork_pathmap_probe::{JoinInput, RelationView, reference};

fn view(rows: &[[u32; 2]], generation: u64) -> RelationView {
    RelationView::from_rows(rows, generation)
}

/// The native table agrees with the counting oracle: the bag count, the
/// derivations visited in place, and per-pattern counts that pin each R
/// row's `(x, y)` and each S row's `(y, z)`, so a misread lane cannot pass.
/// `keep` restricts both to the filter's `y` values.
fn agrees(
    exec: &mut JoinExec,
    table: &JoinTable,
    gens: (u64, u64),
    filter: Option<&[u64]>,
    inp: &JoinInput,
    keep: impl Fn(u32) -> bool,
) -> u64 {
    let oracle = |p: [Option<u32>; 3]| {
        if p[1].is_some_and(|y| !keep(y)) {
            return 0;
        }
        match p[1] {
            Some(_) => reference::matching(inp, p),
            None => (0..table.domain.len())
                .map(|slot| table.domain.id(slot))
                .filter(|&y| keep(y))
                .map(|y| reference::matching(inp, [p[0], Some(y), p[2]]))
                .sum(),
        }
    };
    let total = oracle([None; 3]);
    assert_eq!(
        exec.bag_count(table, &[gens.0, gens.1], filter).unwrap(),
        total
    );
    let (rx, ry) = (&inp.r.cols[0], &inp.r.cols[1]);
    for i in 0..rx.len() {
        let p = [Some(rx[i]), Some(ry[i]), None];
        assert_eq!(
            exec.matching(table, gens, filter, p).unwrap(),
            oracle(p),
            "{p:?}"
        );
    }
    let (sy, sz) = (&inp.s.cols[0], &inp.s.cols[1]);
    for j in 0..sy.len() {
        let p = [None, Some(sy[j]), Some(sz[j])];
        assert_eq!(
            exec.matching(table, gens, filter, p).unwrap(),
            oracle(p),
            "{p:?}"
        );
    }
    if filter.is_none() {
        // The rows read in place, folded: count and a checksum over x and z.
        let mix = |x: u32, z: u32| u64::from(x).wrapping_mul(31).wrapping_add(u64::from(z));
        let (mut visited, mut sum) = (0u64, 0u64);
        exec.for_each_answer(table, gens, |d| {
            visited += 1;
            sum = sum.wrapping_add(mix(d.x(), d.z()));
        })
        .unwrap();
        let mut want = 0u64;
        for i in 0..rx.len() {
            for j in 0..sy.len() {
                if ry[i] == sy[j] {
                    want = want.wrapping_add(mix(rx[i], sz[j]));
                }
            }
        }
        assert_eq!(visited, total);
        assert_eq!(sum, want, "rows read in place");
    }
    total
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
    let table = JoinTable::prepare(&inp.r, &inp.s);
    let mut exec = JoinExec::new(&table);
    let total = agrees(&mut exec, &table, (3, 11), None, &inp, |_| true);
    assert!(total > 0);
    // Not degenerate: some y are in one relation only, and are excluded.
    assert!(table.domain.len() > 48);
}

#[test]
fn a_star_of_three_relations_counts_like_the_oracle() {
    let all = ys(40, 3);
    let r = hetero(200, 4, &all[..30], 1);
    let s = swap(&hetero(150, 5, &all[10..], 1));
    let t = swap(&hetero(120, 6, &all[5..35], 3));
    let table = JoinTable::prepare_star(&[
        (&view(&r, 1), R_KEY),
        (&view(&s, 1), S_KEY),
        (&view(&t, 1), S_KEY),
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
    let two = JoinTable::prepare_star(&[(&view(&r, 1), R_KEY), (&view(&s, 1), S_KEY)]);
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
    let table = JoinTable::prepare(&inp.r, &inp.s);
    let mut exec = JoinExec::new(&table);
    let keep = |y: u32| y.is_multiple_of(3);
    let filter = filter_plane(&table, keep);
    let kept = agrees(&mut exec, &table, (1, 1), Some(&filter), &inp, keep);
    assert!(kept > 0);
    // Silence twin: the filter removes something.
    assert!(kept < reference::bag_count(&inp));
    // A plane over another slot space is refused, not read past its end.
    assert!(matches!(
        exec.bag_count(&table, &[1, 1], Some(&[0u64; 1000])),
        Err(JoinRefusal::FilterShape { .. })
    ));
}

#[test]
fn each_side_is_refused_against_its_own_generation() {
    let r = [[1, 10], [2, 11]];
    let s = [[10, 5], [11, 6]];
    let table = JoinTable::prepare(&view(&r, 4), &view(&s, 9));
    let mut exec = JoinExec::new(&table);
    assert!(exec.bag_count(&table, &[4, 9], None).is_ok());
    assert_eq!(
        exec.bag_count(&table, &[5, 9], None),
        Err(JoinRefusal::Stale {
            side: 0,
            bound: 4,
            current: 5
        })
    );
    assert_eq!(
        exec.bag_count(&table, &[4, 8], None),
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
    let mut table = JoinTable::prepare(&view(&r, 1), &view(&s1, 1));
    table.replace_side(1, &view(&s2, 2), S_KEY).unwrap();
    let mut exec = JoinExec::new(&table);
    let inp = JoinInput {
        r: view(&r, 1),
        s: view(&s2, 2),
    };
    let got = agrees(&mut exec, &table, (1, 2), None, &inp, |_| true);
    // The old S answer differs, so the replacement did something.
    let old = JoinInput {
        r: view(&r, 1),
        s: view(&s1, 1),
    };
    assert_ne!(got, reference::bag_count(&old));
}

#[test]
fn a_side_leaving_the_domain_is_refused() {
    let r = [[1, 10]];
    let s = [[10, 5]];
    let mut table = JoinTable::prepare(&view(&r, 1), &view(&s, 1));
    let s2 = [[10, 5], [77, 6]];
    assert_eq!(
        table.replace_side(1, &view(&s2, 2), S_KEY),
        Err(JoinRefusal::OutsideDomain { side: 1, y: 77 })
    );
    // The table is unchanged: the old generation still answers.
    let mut exec = JoinExec::new(&table);
    assert_eq!(exec.bag_count(&table, &[1, 1], None), Ok(1));
}

#[test]
fn the_joint_plane_has_exactly_the_shared_slots() {
    let r = [[0, 1], [0, 3], [0, 5]];
    let s = [[3, 0], [4, 0], [5, 0]];
    let table = JoinTable::prepare(&view(&r, 1), &view(&s, 1));
    let mut exec = JoinExec::new(&table);
    assert_eq!(joint_ids(&mut exec, &table), (2, true));
    assert!(in_joint(&mut exec, &table, &[3, 5]));
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
        (&view(&r, 1), R_KEY),
        (&view(&s, 1), S_KEY),
        (&view(&t, 1), S_KEY),
    ]);
    let mut exec = JoinExec::new(&table);
    assert_eq!(joint_ids(&mut exec, &table), (2, true));
    assert!(in_joint(&mut exec, &table, &[3, 4]));
}

/// The joint's population and whether its slots are in ascending `y` order.
fn joint_ids(exec: &mut JoinExec, table: &JoinTable) -> (usize, bool) {
    let n = table.domain.len();
    let joint = exec.joint(table, None).unwrap();
    let (mut count, mut last, mut ordered) = (0, None, true);
    for_each_set(joint, n, |slot| {
        let y = table.domain.id(slot);
        ordered &= last.is_none_or(|l| l < y);
        last = Some(y);
        count += 1;
    });
    (count, ordered)
}

/// Whether every `y` in `ys` is a joint slot.
fn in_joint(exec: &mut JoinExec, table: &JoinTable, ys: &[u32]) -> bool {
    let joint = exec.joint(table, None).unwrap();
    ys.iter().all(|&y| {
        table
            .domain
            .slot_of(y)
            .is_some_and(|s| joint[s >> 6] >> (s & 63) & 1 == 1)
    })
}

/// FAILS IF: a position selection is not the same table as preparing from
/// only the kept rows, or the selection does not exclude anything.
#[test]
fn a_position_selection_equals_preparing_the_kept_rows() {
    let all = ys(40, 5);
    let r = hetero(300, 11, &all, 1);
    let s = swap(&hetero(300, 12, &all, 2));
    let (rv, sv) = (view(&r, 1), view(&s, 1));
    // Keep R rows whose x is even, and S rows at even positions.
    let keep_r = |row: usize| rv.cols[0][row].is_multiple_of(2);
    let keep_s = |row: usize| row.is_multiple_of(2);
    let selected = JoinTable::prepare_star_where(&[(&rv, R_KEY, &keep_r), (&sv, S_KEY, &keep_s)]);
    let r2: Vec<[u32; 2]> = r
        .iter()
        .copied()
        .filter(|p| p[0].is_multiple_of(2))
        .collect();
    let s2: Vec<[u32; 2]> = s.iter().step_by(2).copied().collect();
    let inp = JoinInput {
        r: view(&r2, 1),
        s: view(&s2, 1),
    };
    let mut exec = JoinExec::new(&selected);
    let kept = agrees(&mut exec, &selected, (1, 1), None, &inp, |_| true);
    let full = JoinInput {
        r: rv.clone(),
        s: sv.clone(),
    };
    assert!(
        kept > 0 && kept < reference::bag_count(&full),
        "the selection excludes something"
    );
}
