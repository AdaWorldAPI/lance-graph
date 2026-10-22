//! Every admitted arm against the independent nested-loop oracle.

use mork_pathmap_probe::{
    Backend, JoinInput, Refusal, RelationView, Semantics, bind, boundary::mork_join, capability,
    reference,
};

fn input(r: &[[u32; 2]], s: &[[u32; 2]], gen_r: u64, gen_s: u64) -> JoinInput {
    JoinInput {
        r: RelationView::from_rows(r, gen_r),
        s: RelationView::from_rows(s, gen_s),
    }
}

const ARMS: [Backend; 3] = [Backend::Mork, Backend::PathMapZipper, Backend::Native];
const SET_ONLY: [Backend; 2] = [Backend::Mork, Backend::PathMapZipper];

fn pattern(row: &[u32; 3]) -> [Option<u32>; 3] {
    row.map(Some)
}

/// The native arm against the counting oracle, then each upstream engine's
/// set against both: every row it returns has a derivation, and every native
/// derivation (read in place) is in its set. Returns the set's size.
fn check_set(r: &[[u32; 2]], s: &[[u32; 2]]) -> usize {
    let inp = input(r, s, 1, 1);
    let mut nat = bind(Backend::Native, &inp, Semantics::Set).expect("native binds");
    assert_eq!(
        nat.answers(&inp),
        Err(Refusal::FoldOnly),
        "native builds no rows"
    );
    let (table, exec) = nat.native(&inp).unwrap().expect("native arm");
    assert_eq!(
        exec.bag_count(table, &[1, 1], None).unwrap(),
        reference::bag_count(&inp)
    );
    let mut size = None;
    for arm in SET_ONLY {
        let mut idx = bind(arm, &inp, Semantics::Set).expect("set semantics is admitted");
        let got = idx.answers(&inp).expect("fresh generations");
        for row in got.rows.keys() {
            let want = reference::matching(&inp, pattern(row));
            assert!(want > 0, "{arm:?} ({}) returned {row:?}", mork_join());
            assert_eq!(
                exec.matching(table, (1, 1), None, pattern(row)).unwrap(),
                want,
                "native vs oracle at {row:?}"
            );
        }
        exec.for_each_answer(table, (1, 1), |d| {
            assert!(
                got.rows.contains_key(&[d.x(), d.y(), d.z()]),
                "{arm:?} misses {:?}",
                (d.x(), d.y(), d.z())
            );
        })
        .unwrap();
        // Asking twice gives the same answer: the MORK arm removes its `Q` facts.
        assert_eq!(idx.answers(&inp).unwrap(), got, "{arm:?} second call");
        assert_eq!(*size.get_or_insert(got.rows.len()), got.rows.len());
    }
    size.unwrap()
}

#[test]
fn shared_variable_join_matches_the_oracle() {
    let r = [[1, 10], [1, 11], [2, 10], [3, 12]];
    let s = [[10, 100], [11, 101], [10, 102], [99, 103]];
    // (1,10,100) (1,10,102) (1,11,101) (2,10,100) (2,10,102); y = 12 and 99 match nothing.
    assert_eq!(check_set(&r, &s), 5);
}

#[test]
fn empty_and_disjoint_inputs_give_no_answers() {
    assert_eq!(check_set(&[], &[[1, 2]]), 0);
    assert_eq!(check_set(&[[1, 2]], &[]), 0);
    assert_eq!(check_set(&[[1, 2]], &[[3, 4]]), 0);
}

/// `x` and `z` occur in one atom each; only `y` is shared. A cross product
/// over `x` and `z` per `y` is the whole answer, no more.
#[test]
fn variable_scope_is_per_atom() {
    let r = [[1, 7], [2, 7], [3, 7]];
    let s = [[7, 4], [7, 5]];
    assert_eq!(check_set(&r, &s), 6);
}

/// Values that differ only in byte order or width must not alias.
#[test]
fn large_ids_do_not_alias() {
    let r = [[u32::MAX, 0x0100_0000], [0, 1]];
    let s = [[1, u32::MAX], [0x0100_0000, 2], [0x0000_0001, 3]];
    assert_eq!(check_set(&r, &s), 3);
}

/// Duplicate rows: the oracle counts 2 × 2 = 4 derivations; the upstream
/// indexes are sets and must refuse a bag request instead of answering 1.
#[test]
fn duplicates_are_refused_under_bag_semantics() {
    let r = [[1, 10], [1, 10]];
    let s = [[10, 5], [10, 5]];
    let inp = input(&r, &s, 1, 1);
    assert_eq!(reference::matching(&inp, pattern(&[1, 10, 5])), 4);
    for arm in SET_ONLY {
        assert!(matches!(bind(arm, &inp, Semantics::Bag), Err(Refusal::SetOnly(a)) if a == arm));
    }
    // The native table keeps multiplicities and counts the bag exactly.
    let mut native = bind(Backend::Native, &inp, Semantics::Bag).unwrap();
    let (table, exec) = native.native(&inp).unwrap().unwrap();
    assert_eq!(
        exec.matching(table, (1, 1), None, pattern(&[1, 10, 5]))
            .unwrap(),
        4
    );
    assert_eq!(exec.bag_count(table, &[1, 1], None).unwrap(), 4);
    // Under set semantics they agree with the oracle's set.
    assert_eq!(check_set(&r, &s), 1);
}

#[test]
fn a_rebound_generation_is_refused() {
    let r = [[1, 10]];
    let s = [[10, 5]];
    let bound = input(&r, &s, 4, 9);
    for arm in ARMS {
        let mut idx = bind(arm, &bound, Semantics::Set).unwrap();
        // Same rows, but the R view was resealed: the copy is stale.
        let current = input(&r, &s, 5, 9);
        assert_eq!(
            idx.answers(&current),
            Err(Refusal::Stale {
                bound: (4, 9),
                current: (5, 9)
            })
        );
        // Silence twin: the matching generations still answer.
        if arm == Backend::Native {
            assert!(idx.native(&bound).is_ok());
        } else {
            assert!(idx.answers(&bound).is_ok());
        }
    }
}

/// A generation that comes back after a reseal (ABA) is accepted: the witness
/// compares numbers, so the caller must never reuse one. Pinned, not hidden.
#[test]
fn a_reused_generation_number_is_not_detected() {
    let r1 = [[1, 10]];
    let r2 = [[2, 10]];
    let s = [[10, 5]];
    let mut idx = bind(
        Backend::PathMapZipper,
        &input(&r1, &s, 1, 1),
        Semantics::Set,
    )
    .unwrap();
    let got = idx.answers(&input(&r2, &s, 1, 1)).unwrap();
    assert!(
        got.rows.contains_key(&[1, 10, 5]),
        "answers come from the bound copy"
    );
    assert!(!got.rows.contains_key(&[2, 10, 5]));
}

/// The native arm answers only through the co-addressed table, for both
/// semantics, and its bag counts carry the oracle's multiplicities.
#[test]
fn native_join_answers_bags_from_the_coaddressed_table() {
    assert_eq!(capability(Backend::Native, Semantics::Set), Ok(()));
    assert_eq!(capability(Backend::Native, Semantics::Bag), Ok(()));
    let r = [[1, 10], [1, 10], [2, 10], [3, 11], [4, 99]];
    let s = [[10, 5], [10, 6], [10, 5], [11, 7], [12, 8]];
    let inp = input(&r, &s, 1, 1);
    assert!(
        reference::matching(&inp, pattern(&[1, 10, 5])) > 1,
        "fixture has multiplicities"
    );
    let mut idx = bind(Backend::Native, &inp, Semantics::Bag).unwrap();
    let (table, exec) = idx.native(&inp).unwrap().unwrap();
    assert_eq!(
        exec.bag_count(table, &[1, 1], None).unwrap(),
        reference::bag_count(&inp)
    );
    // Every derivation read in place is one the oracle has; visited once each.
    let mut visited = 0u64;
    exec.for_each_answer(table, (1, 1), |d| {
        assert!(reference::matching(&inp, [Some(d.x()), Some(d.y()), Some(d.z())]) > 0);
        visited += 1;
    })
    .unwrap();
    assert_eq!(visited, reference::bag_count(&inp));
    // Multiplicities per pattern, pinned and free columns alike.
    for p in [
        [Some(1), Some(10), Some(5)],
        [None, Some(10), None],
        [None, None, Some(5)],
        [Some(3), None, None],
        [None, Some(99), None],
    ] {
        assert_eq!(
            exec.matching(table, (1, 1), None, p).unwrap(),
            reference::matching(&inp, p),
            "{p:?}"
        );
    }
}

/// The general-rules oracle, given the fixed join as an `OracleRule`, answers
/// the same set as the fixed-query MORK arm on the same input.
#[test]
fn general_rules_agree_with_the_fixed_query() {
    use mork_pathmap_probe::boundary::{
        OracleAtom, OracleFacts, OracleRule, OracleTerm, mork_rules,
    };
    const R: u16 = 0;
    const S: u16 = 1;
    let r = [[1, 10], [1, 11], [2, 10], [3, 12], [1, 10]];
    let s = [[10, 100], [11, 101], [10, 102], [99, 103], [10, 100]];
    let inp = input(&r, &s, 1, 1);

    let mut fixed = bind(Backend::Mork, &inp, Semantics::Set).expect("set semantics is admitted");
    let want: std::collections::BTreeSet<Vec<u32>> = fixed
        .answers(&inp)
        .expect("fresh generations")
        .rows
        .keys()
        .map(|row| row.to_vec())
        .collect();
    assert!(!want.is_empty(), "fixture must produce answers");

    // Q(x, y, z) :- R(x, y), S(y, z).  x = 0, y = 1, z = 2.
    let (x, y, z) = (OracleTerm::Var(0), OracleTerm::Var(1), OracleTerm::Var(2));
    let rule = OracleRule {
        head: vec![x.clone(), y.clone(), z.clone()],
        body: vec![
            OracleAtom {
                rel: R,
                terms: vec![x, y.clone()],
            },
            OracleAtom {
                rel: S,
                terms: vec![y, z],
            },
        ],
    };
    let facts = [
        OracleFacts {
            rel: R,
            arity: 2,
            view: &inp.r,
        },
        OracleFacts {
            rel: S,
            arity: 2,
            view: &inp.s,
        },
    ];
    let (got, cost) = mork_rules(&facts, &[rule]).expect("general rules answer");
    assert_eq!(got.rows, want);
    // Over the distinct facts the body matches 5 times:
    // (1,10,100) (1,10,102) (1,11,101) (2,10,100) (2,10,102).
    assert_eq!(cost.matches, 5);
}
