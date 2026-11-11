//! G6: `CanCook` registered as a DT64 method and executed from its
//! `MethodRef`; its WHEN guard is lowered through relalg, never hand-written.

use foundry_parity_lab::bind::{World, CLASS, KNOWN0, STOCK0};
use foundry_parity_lab::fixture::{self as fx, STORES};
use foundry_parity_lab::method::{
    can_cook_method, execute_can_cook, method_id, MethodRefusal, PreparedMethod, QS,
};
use foundry_parity_lab::oracle::{Cook, Oracle};
use foundry_parity_lab::source::{Network, Order};
use lance_graph_contract::set_algebra::{Logic, SetExpr};
use lance_graph_dt64::{Guard, MethodRef, Refusal, SigRev};

/// Store rows the oracle says can make `q` of `m`.
fn oracle_set(m: usize, q: i32) -> u64 {
    (0..STORES)
        .filter(|&s| {
            matches!(
                Oracle::BASELINE.can_cook(s, m, q),
                Cook::Known { feasible: true, .. }
            )
        })
        .fold(0u64, |w, s| w | 1 << s)
}

fn mref(world: &World, m: usize, q: i32) -> MethodRef {
    world.store.method_ref(CLASS, method_id(m, q)).unwrap()
}

#[test]
fn every_registered_can_cook_method_matches_the_oracle() {
    let world = World::new();
    let net = Network::ingest(Order::Identity);
    for m in 0..fx::ITEMS {
        for q in QS {
            let got = execute_can_cook(&world, &net, &mref(&world, m, q), true).unwrap();
            let want = oracle_set(m, q);
            assert_eq!(got.stores, want, "item {m} q {q}");
            assert_eq!(got.count, u64::from(want.count_ones()), "item {m} q {q}");
        }
    }
    // Anti-vacuity: the guard selects some stores and rejects others.
    let s = oracle_set(0, 3);
    assert!(s != 0 && s.count_ones() < STORES as u32);
}

#[test]
fn another_row_order_answers_identically_by_identity() {
    let world = World::new();
    let a = Network::ingest(Order::Identity);
    let b = Network::ingest(Order::Reversed);
    for m in 0..fx::ITEMS {
        let x = execute_can_cook(&world, &a, &mref(&world, m, 1), true).unwrap();
        let y = execute_can_cook(&world, &b, &mref(&world, m, 1), true).unwrap();
        assert_eq!(x.stores, y.stores, "item {m}");
    }
}

#[test]
fn the_reader_scope_is_load_bearing_and_separate_from_the_method() {
    let world = World::new();
    let net = Network::ingest(Order::Identity);
    let row = fx::case::HIDDEN;
    let with = execute_can_cook(&world, &net, &mref(&world, 0, 1), true).unwrap();
    let without = execute_can_cook(&world, &net, &mref(&world, 0, 1), false).unwrap();
    assert_eq!(
        with.stores >> row & 1,
        0,
        "the denied store is never answered"
    );
    assert_eq!(
        without.stores >> row & 1,
        1,
        "its stock would satisfy the guard"
    );
    assert_eq!(with.stores, without.stores & !(1 << row));
}

#[test]
fn a_reference_with_another_guard_is_refused() {
    let world = World::new();
    let net = Network::ingest(Order::Identity);
    let forged = MethodRef {
        guard: Some(Guard::new(SetExpr::All, Logic::TwoValued).witness()),
        ..mref(&world, 0, 1)
    };
    assert!(matches!(
        execute_can_cook(&world, &net, &forged, true),
        Err(MethodRefusal::Admit(Refusal::GuardMismatch { .. }))
    ));
    let other_rev = MethodRef {
        sig_rev: SigRev(2),
        ..mref(&world, 0, 1)
    };
    assert!(matches!(
        execute_can_cook(&world, &net, &other_rev, true),
        Err(MethodRefusal::Admit(
            Refusal::SignatureRevisionMismatch { .. }
        ))
    ));
    // Silent half: the issued reference executes.
    assert!(execute_can_cook(&world, &net, &mref(&world, 0, 1), true).is_ok());
}

#[test]
fn a_reference_issued_by_another_store_is_refused() {
    let world = World::new();
    let other = World::new();
    let net = Network::ingest(Order::Identity);
    assert!(matches!(
        execute_can_cook(&world, &net, &mref(&other, 0, 1), true),
        Err(MethodRefusal::Admit(Refusal::ForeignStore { .. }))
    ));
}

#[test]
fn a_method_whose_reads_miss_a_guard_field_is_refused() {
    let k = fx::recipe(0).demand[0].0;
    let mut bad = can_cook_method(0, 1);
    bad.reads = lance_graph_contract::class_view::WideFieldMask::from_positions(
        &(0..=255u8)
            .filter(|&f| bad.reads.has(f) && usize::from(f) != STOCK0 + k)
            .collect::<Vec<_>>(),
    );
    bad.on = bad.reads.clone();
    let world = World::with_methods(vec![bad]);
    let net = Network::ingest(Order::Identity);
    match execute_can_cook(&world, &net, &mref(&world, 0, 1), true) {
        Err(MethodRefusal::NotBound { field }) => assert_eq!(usize::from(field), STOCK0 + k),
        other => panic!("expected NotBound, got {other:?}"),
    }
    // The known-mask field of the same category is still read: only the
    // dropped value lane is missing.
    assert!(can_cook_method(0, 1).reads.has((KNOWN0 + k) as u8));
}

/// The invariant `PreparedMethod` relies on to reuse relalg's program: a
/// `Keep` over any `SetExpr` shape lowers to exactly one fold. If relalg ever
/// lowers one into several, `prepare` refuses (`Folds`) and this test names
/// the shape.
#[test]
fn every_guard_shape_lowers_to_one_reusable_fold() {
    use lance_graph_contract::set_algebra::Cmp;
    use lance_graph_mask_risc::{LaneRef, Planes};
    use lance_graph_relalg::{rel_from_set, run_with_program, Measure, Query};
    let m = [0xF0F0u64];
    let l: Vec<i32> = (0..64).collect();
    let u: Vec<u32> = (0..64).collect();
    let masks: [&[u64]; 2] = [&m, &m];
    let lanes = [LaneRef::I32(&l), LaneRef::U32(&u)];
    let planes = Planes {
        n_rows: 64,
        masks: &masks,
        lanes: &lanes,
    };
    let shapes = [
        SetExpr::And(vec![SetExpr::Sym(0), SetExpr::i32(0, Cmp::Ge, 3)]),
        SetExpr::Or(vec![SetExpr::Sym(0), SetExpr::i32(0, Cmp::Ge, 30)]),
        SetExpr::negated(SetExpr::And(vec![SetExpr::Sym(1), SetExpr::ne_u32(1, 5)])),
        SetExpr::Or(vec![
            SetExpr::And(vec![SetExpr::Sym(0), SetExpr::negated(SetExpr::Sym(1))]),
            SetExpr::And(vec![SetExpr::i32(0, Cmp::Lt, 9), SetExpr::eq_u32(1, 4)]),
        ]),
        can_cook_guard_on_dense_planes(),
    ];
    for e in shapes {
        let mut keep = [0u64; 1];
        let (run, program) = run_with_program(
            &Query {
                rel: rel_from_set(&e),
                measure: Measure::Keep,
            },
            &planes,
            Some(&mut keep[..]),
        )
        .unwrap();
        assert_eq!(run.programs, 1, "{e:?}");
        assert!(program.is_some(), "{e:?}");
    }
}

/// A CanCook-shaped conjunction (masks AND thresholds) over the two dense
/// planes the shape test supplies.
fn can_cook_guard_on_dense_planes() -> SetExpr {
    use lance_graph_contract::set_algebra::Cmp;
    SetExpr::And(vec![
        SetExpr::Sym(0),
        SetExpr::Sym(1),
        SetExpr::i32(0, Cmp::Ge, 2),
        SetExpr::i32(0, Cmp::Ge, 5),
        SetExpr::ne_u32(1, 7),
    ])
}

/// The program is lowered once and its scratch reused: every later run of one
/// prepared method must equal the first and the oracle.
#[test]
fn a_prepared_method_answers_the_same_on_every_run() {
    let world = World::new();
    let net = Network::ingest(Order::Identity);
    for m in 0..fx::ITEMS {
        for q in QS {
            let p = PreparedMethod::prepare(&world, &net, &mref(&world, m, q), true).unwrap();
            let first = p.run().unwrap();
            assert_eq!(first.stores, oracle_set(m, q), "item {m} q {q}");
            for _ in 0..3 {
                assert_eq!(p.run().unwrap(), first, "item {m} q {q}");
            }
        }
    }
}
