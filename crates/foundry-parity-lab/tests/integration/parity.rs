//! Arm B (DT64 + mask-RISC) == oracle == arm A (materialized objects).

use foundry_parity_lab::arm_a::Objects;
use foundry_parity_lab::arm_b::{demanded_fields, Answer, Guards};
use foundry_parity_lab::bind::World;
use foundry_parity_lab::fixture::{case, CHEESE, FLOUR, ITEMS, STORES};
use foundry_parity_lab::oracle::{Cook, Limit, Oracle};
use foundry_parity_lab::source::{Network, Order};

use crate::common::{coarse, open};

#[test]
fn every_item_quantity_and_store_agrees_across_the_three_arms() {
    let world = World::new();
    let net = Network::ingest(Order::Identity);
    let objects = Objects::build(false);
    let oracle = Oracle::BASELINE;
    let (mut feasible, mut infeasible) = (0, 0);
    for m in 0..ITEMS {
        let query = open(&world, &net, m, false, false).expect("binds");
        for q in [1, 2, 5] {
            let b = query.answers(q, Guards::ALL).expect("answers");
            for s in 0..STORES {
                let o = oracle.can_cook(s, m, q);
                assert_eq!(
                    objects.can_cook(s, m, q),
                    o,
                    "arm A, store {s} item {m} q {q}"
                );
                assert_eq!(b[s], coarse(o), "arm B, store {s} item {m} q {q}");
                match b[s] {
                    Answer::Known { feasible: true, .. } => feasible += 1,
                    Answer::Known {
                        feasible: false, ..
                    } => infeasible += 1,
                    _ => {}
                }
            }
        }
    }
    // Anti-vacuity: both outcomes are common, so agreement is informative.
    let total = ITEMS * 3 * STORES;
    assert!(
        feasible * 5 > total && infeasible * 5 > total,
        "{feasible}/{infeasible} of {total}"
    );
}

#[test]
fn the_named_cases_answer_as_designed() {
    let world = World::new();
    let net = Network::ingest(Order::Identity);
    let o = Oracle::BASELINE;
    let b = open(&world, &net, 0, false, false)
        .expect("binds")
        .answers(1, Guards::ALL)
        .expect("answers");
    let known = |s, feasible, max, limit| {
        assert_eq!(
            o.can_cook(s, 0, 1),
            Cook::Known {
                feasible,
                max,
                limit
            },
            "store row {s}"
        );
        assert_eq!(b[s], Answer::Known { feasible, max }, "arm B store row {s}");
    };
    known(case::ALL_GOOD, true, 30, Limit::Staff(0));
    known(case::NO_CHEESE, false, 0, Limit::Ingredient(CHEESE));
    known(case::NO_COOK, false, 0, Limit::Staff(0));
    known(case::NO_OVEN, false, 0, Limit::Equipment(0));
    known(case::EXACTLY_ONE, true, 1, Limit::Ingredient(FLOUR));
    known(case::RESTOCKED, false, 0, Limit::Ingredient(CHEESE));
    assert_eq!(
        o.can_cook(case::EXACTLY_ONE, 0, 2),
        Cook::Known {
            feasible: false,
            max: 1,
            limit: Limit::Ingredient(FLOUR)
        }
    );
    assert_eq!(o.can_cook(case::HIDDEN, 0, 1), Cook::NotAuthorized);
    assert_eq!(b[case::HIDDEN], Answer::NotAuthorized);
    assert_eq!(
        o.can_cook(case::UNKNOWN_CHEESE, 0, 1),
        Cook::Unknown { category: CHEESE }
    );
    assert_eq!(b[case::UNKNOWN_CHEESE], Answer::Unknown);
}

#[test]
fn network_capacity_and_cost_agree() {
    let world = World::new();
    let net = Network::ingest(Order::Identity);
    let objects = Objects::build(false);
    let o = Oracle::BASELINE;
    for m in 0..ITEMS {
        let query = open(&world, &net, m, true, false).expect("binds");
        let cap = query.network_capacity().expect("capacity");
        assert_eq!(cap, o.network_capacity(m), "item {m}");
        assert_eq!(cap, objects.network_capacity(m), "item {m}");
        assert!(cap > 0, "item {m}: capacity must be informative");
        for q in [1, 3] {
            let cost = query.network_cost(q).expect("cost");
            assert_eq!(cost, o.network_cost(m, q), "item {m} q {q}");
        }
    }
}

#[test]
fn can_fulfill_agrees_and_discriminates() {
    let world = World::new();
    let net = Network::ingest(Order::Identity);
    let o = Oracle::BASELINE;
    let (mut yes, mut no) = (0, 0);
    for m in 0..ITEMS {
        let query = open(&world, &net, m, false, true).expect("binds");
        for deadline in [30, 45, 60] {
            let mask = query.fulfill_mask(2, deadline).expect("fulfill");
            for r in 0..STORES {
                let s = (query.store_at(r) - foundry_parity_lab::fixture::STORE_ID_BASE) as usize;
                let b = mask >> r & 1 == 1;
                // A store the oracle cannot judge (denied / unknown) is never
                // reported as able to fulfil.
                assert_eq!(
                    b,
                    o.can_fulfill(s, m, 2, deadline) == Some(true),
                    "{s} {m} {deadline}"
                );
                if b {
                    yes += 1
                } else {
                    no += 1
                }
            }
        }
    }
    assert!(yes > 20 && no > 20, "{yes} yes / {no} no");
}

#[test]
fn only_demanded_fields_are_bound_and_borrowed_once() {
    let world = World::new();
    let net = Network::ingest(Order::Identity);
    let fields = demanded_fields(0, false, false);
    let query = open(&world, &net, 0, false, false).expect("binds");
    // 3 ingredients x (stock, known) + staff + equipment + offered + storeId.
    assert_eq!(fields.len(), 10);
    assert_eq!(
        net.borrows.get(),
        10,
        "borrow_all: one borrow per bound field"
    );
    for q in 1..=5 {
        query.answers(q, Guards::ALL).expect("answers");
    }
    query.network_capacity().expect("capacity");
    assert_eq!(
        net.borrows.get(),
        10,
        "questions reuse the binding; no further lookups"
    );
}
