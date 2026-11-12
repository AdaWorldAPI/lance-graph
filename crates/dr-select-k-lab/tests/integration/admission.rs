//! Admission: what a view may answer, and what a stale selection may bind.

use dr_select_k_lab::bind::BindRefusal;
use dr_select_k_lab::counters::Counters;
use dr_select_k_lab::fixture::Pattern;
use dr_select_k_lab::ops::{self, Inadmissible, Op, PreparedFolds};
use dr_select_k_lab::{Bound, Order, Residual, Version};

use crate::common;

fn three_views(bound: &Bound, c: &Counters) -> Vec<Residual> {
    vec![
        Residual::All,
        ops::residual_a_ge(bound, 50, &Residual::Positions(Vec::new()), c),
        ops::residual_a_ge(bound, 50, &Residual::Mask(Vec::new()), c),
    ]
}

// Fails if top-by-key is admitted on a filter binding, for either requested
// order or any view, or if the refusal names the wrong order.
#[test]
fn top_by_key_is_refused_on_a_filter_binding_for_both_orders() {
    let src = common::source(Pattern::Random);
    let c = Counters::new();
    let bound = common::bind(&src, common::range_for_k(256), Order::Source, &c);
    assert_eq!(bound.order(), Order::Source);
    let mut folds = PreparedFolds::new(&bound);
    for want in common::ORDERS {
        let refusal = Err(Inadmissible::OrderNotWitnessed {
            want,
            have: Order::Source,
        });
        for res in three_views(&bound, &c) {
            assert_eq!(
                ops::admissible(
                    Op::TopByKey {
                        limit: 5,
                        order: want
                    },
                    &bound,
                    res.view()
                ),
                refusal,
                "admissible want={want:?}"
            );
            let mut out = Vec::new();
            assert_eq!(
                folds.top_by_key(&bound, res.view(), 5, want, &mut out, &c),
                refusal,
                "top_by_key want={want:?}"
            );
        }
    }
}

// Fails if a key-order binding admits a request for the other order, refuses
// a request for its own order, or names the wrong pair in the refusal. Covers
// both the admission check and the fold, for all three views.
#[test]
fn a_key_order_binding_admits_only_its_own_order() {
    let src = common::source(Pattern::Random);
    let c = Counters::new();
    for have in common::ORDERS {
        let bound = common::bind(&src, common::range_for_k(256), have, &c);
        assert_eq!(bound.order(), have);
        let mut folds = PreparedFolds::new(&bound);
        for want in common::ORDERS {
            let expected = if want == have {
                Ok(())
            } else {
                Err(Inadmissible::OrderNotWitnessed { want, have })
            };
            for res in three_views(&bound, &c) {
                assert_eq!(
                    ops::admissible(
                        Op::TopByKey {
                            limit: 5,
                            order: want
                        },
                        &bound,
                        res.view()
                    ),
                    expected,
                    "admissible have={have:?} want={want:?}"
                );
                let mut out = Vec::new();
                assert_eq!(
                    folds.top_by_key(&bound, res.view(), 5, want, &mut out, &c),
                    expected,
                    "top_by_key have={have:?} want={want:?}"
                );
            }
        }
    }
}

// Fails if a request for `Order::Source` is admitted on any binding: the
// source order declares no key order, so no binding can witness it.
#[test]
fn a_request_for_the_source_order_is_refused_on_every_binding() {
    let src = common::source(Pattern::Clustered);
    let c = Counters::new();
    for have in common::BINDINGS {
        let bound = common::bind(&src, common::range_for_k(256), have, &c);
        let mut folds = PreparedFolds::new(&bound);
        let refusal = Err(Inadmissible::OrderNotWitnessed {
            want: Order::Source,
            have,
        });
        for res in three_views(&bound, &c) {
            assert_eq!(
                ops::admissible(
                    Op::TopByKey {
                        limit: 5,
                        order: Order::Source
                    },
                    &bound,
                    res.view()
                ),
                refusal,
                "admissible have={have:?}"
            );
            let mut out = Vec::new();
            assert_eq!(
                folds.top_by_key(&bound, res.view(), 5, Order::Source, &mut out, &c),
                refusal,
                "top_by_key have={have:?}"
            );
        }
    }
}

// Fails if an order-free operation is refused for any view or any binding.
#[test]
fn order_free_operations_are_always_admitted() {
    let src = common::source(Pattern::Clustered);
    let c = Counters::new();
    for have in common::BINDINGS {
        let bound = common::bind(&src, common::range_for_k(256), have, &c);
        for res in three_views(&bound, &c) {
            for op in [Op::CountBGt { t: 500 }, Op::SumA] {
                assert_eq!(
                    ops::admissible(op, &bound, res.view()),
                    Ok(()),
                    "{op:?} {have:?}"
                );
            }
        }
    }
}

// Fails if a selection from before `advance` binds, or one made after does not.
#[test]
fn a_stale_selection_is_refused_and_a_fresh_one_binds() {
    let mut src = common::source(Pattern::Clustered);
    let c = Counters::new();
    let range = common::range_for_k(64);
    let stale = src.select_filter(range, &c);
    assert_eq!(stale.version, Version(1));
    src.advance();
    match Bound::bind(&src, stale, &c) {
        Err(BindRefusal::NotWitnessedForVersion { selection, source }) => {
            assert_eq!(selection, Version(1));
            assert_eq!(source, Version(2));
        }
        Ok(_) => panic!("a stale selection must not bind"),
    }
    // The refused bind touched nothing: only the selection call was counted.
    assert_eq!(c.snapshot()[0], 1);
    assert_eq!(c.snapshot()[1], 0);
    let fresh = src.select_filter(range, &c);
    let bound = Bound::bind(&src, fresh, &c).unwrap();
    assert_eq!(bound.version(), Version(2));
    assert_eq!(bound.k(), 64);
}
