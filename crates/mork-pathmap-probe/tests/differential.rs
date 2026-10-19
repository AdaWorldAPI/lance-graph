//! Every admitted arm against the independent nested-loop oracle.

use mork_pathmap_probe::{
    Backend, JoinInput, Refusal, RelationView, Semantics, bind, boundary::mork_join, capability,
    reference,
};

fn input<'a>(r: &'a [[u32; 2]], s: &'a [[u32; 2]], gen_r: u64, gen_s: u64) -> JoinInput<'a> {
    JoinInput {
        r: RelationView {
            rows: r,
            generation: gen_r,
        },
        s: RelationView {
            rows: s,
            generation: gen_s,
        },
    }
}

const ARMS: [Backend; 2] = [Backend::Mork, Backend::PathMapZipper];

fn check_set(r: &[[u32; 2]], s: &[[u32; 2]]) -> usize {
    let inp = input(r, s, 1, 1);
    let want = reference::answers(&inp, Semantics::Set);
    for arm in ARMS {
        let mut idx = bind(arm, &inp, Semantics::Set).expect("set semantics is admitted");
        let got = idx.answers(&inp).expect("fresh generations");
        assert_eq!(
            got,
            want,
            "{arm:?} ({}) disagrees with the oracle",
            mork_join()
        );
        // Asking twice gives the same answer: the MORK arm removes its `Q` facts.
        assert_eq!(idx.answers(&inp).unwrap(), want, "{arm:?} second call");
    }
    want.rows.len()
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
    assert_eq!(
        reference::answers(&inp, Semantics::Bag).rows[&[1, 10, 5]],
        4
    );
    for arm in ARMS {
        assert!(matches!(bind(arm, &inp, Semantics::Bag), Err(Refusal::SetOnly(a)) if a == arm));
    }
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
        assert!(idx.answers(&bound).is_ok());
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

#[test]
fn native_join_is_refused_not_emulated() {
    assert_eq!(
        capability(Backend::Native, Semantics::Set),
        Err(Refusal::NoAdmittedNativeJoin)
    );
    let r = [[1, 10]];
    let s = [[10, 5]];
    assert!(matches!(
        bind(Backend::Native, &input(&r, &s, 1, 1), Semantics::Set),
        Err(Refusal::NoAdmittedNativeJoin)
    ));
}
