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

// Fails if top-by-key is admitted on a binding that does not witness key order.
#[test]
fn top_by_key_is_refused_without_the_order_witness() {
    let src = common::source(Pattern::Random);
    let c = Counters::new();
    let bound = common::bind(&src, common::range_for_k(256), false, &c);
    assert_eq!(bound.order(), Order::Source);
    let mut folds = PreparedFolds::new(&bound);
    let want = Err(Inadmissible::OrderNotWitnessed {
        have: Order::Source,
    });
    for res in three_views(&bound, &c) {
        assert_eq!(
            ops::admissible(Op::TopByKey { limit: 5 }, &bound, res.view()),
            want
        );
        let mut out = Vec::new();
        assert_eq!(folds.top_by_key(&bound, res.view(), 5, &mut out, &c), want);
    }
}

// Fails if top-by-key is refused on a binding that witnesses key order.
#[test]
fn top_by_key_is_admitted_with_the_order_witness() {
    let src = common::source(Pattern::Random);
    let c = Counters::new();
    let bound = common::bind(&src, common::range_for_k(256), true, &c);
    assert_eq!(bound.order(), Order::KeyThenGuid);
    for res in three_views(&bound, &c) {
        assert_eq!(
            ops::admissible(Op::TopByKey { limit: 5 }, &bound, res.view()),
            Ok(())
        );
    }
}

// Fails if an order-free operation is refused for any view or either order.
#[test]
fn order_free_operations_are_always_admitted() {
    let src = common::source(Pattern::Clustered);
    let c = Counters::new();
    for ordered in [false, true] {
        let bound = common::bind(&src, common::range_for_k(256), ordered, &c);
        for res in three_views(&bound, &c) {
            for op in [Op::CountBGt { t: 500 }, Op::SumA] {
                assert_eq!(
                    ops::admissible(op, &bound, res.view()),
                    Ok(()),
                    "{op:?} {ordered}"
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
