//! Report's `FoldState` describes its merge through the contract's
//! `AlgebraLaw`. These are executor self-conformance tests: the description
//! must agree with what `FoldState::merge` / `identity` actually do. No other
//! executor is imported here; cross-crate agreement lives in lane-fold.

use lance_graph_contract::algebra_law::{AlgebraDescriptor, IdentityKind};
use lance_graph_report::ids::FieldId;
use lance_graph_report::plan::FoldState;

const F: FieldId = FieldId(7);

fn states() -> [FoldState; 4] {
    [
        FoldState::Count,
        FoldState::Sum(F),
        FoldState::Min(F),
        FoldState::Max(F),
    ]
}

/// Sample states including both extremes, so wrapping and saturation edges
/// are exercised, not just small positives.
const SAMPLES: [i64; 7] = [i64::MIN, -9, -1, 0, 1, 42, i64::MAX];

#[test]
fn min_and_max_are_idempotent_sum_and_count_are_not() {
    assert!(FoldState::Min(F).algebra_law().idempotent);
    assert!(FoldState::Max(F).algebra_law().idempotent);
    assert!(!FoldState::Sum(F).algebra_law().idempotent);
    assert!(!FoldState::Count.algebra_law().idempotent);
}

#[test]
fn nothing_claims_invertibility_or_order() {
    for s in states() {
        let law = s.algebra_law();
        assert!(
            !law.invertible,
            "{s:?}: no tested remove-a-contribution path exists"
        );
        assert!(!law.ordered, "{s:?} is order-insensitive");
    }
}

/// Two-sided: every flag report declares must hold on `merge`, and every flag
/// it withholds must have a witness that `merge` really lacks it. A wrong
/// descriptor fails here in either direction.
#[test]
fn report_descriptor_matches_its_executor() {
    for s in states() {
        let law = s.algebra_law();
        let m = |a, b| s.merge(a, b);

        let assoc = SAMPLES.iter().all(|&a| {
            SAMPLES
                .iter()
                .all(|&b| SAMPLES.iter().all(|&c| m(m(a, b), c) == m(a, m(b, c))))
        });
        let comm = SAMPLES
            .iter()
            .all(|&a| SAMPLES.iter().all(|&b| m(a, b) == m(b, a)));
        let idem = SAMPLES.iter().all(|&a| m(a, a) == a);
        assert_eq!(law.associative, assoc, "{s:?} associative");
        assert_eq!(law.commutative, comm, "{s:?} commutative");
        assert_eq!(law.idempotent, idem, "{s:?} idempotent");
    }
}

/// The concrete identity is the executor's; the semantic kind must name it.
/// The mapping from kind to value lives HERE, in the executor's own test,
/// never in the contract or a planner.
#[test]
fn concrete_identity_agrees_with_identity_kind() {
    for s in states() {
        let e = s.identity();
        assert!(
            SAMPLES
                .iter()
                .all(|&a| s.merge(e, a) == a && s.merge(a, e) == a),
            "{s:?}: identity() is not the merge identity"
        );
        let expected = match s.algebra_law().identity_kind {
            IdentityKind::Zero => 0,
            IdentityKind::Top => i64::MAX,
            IdentityKind::Bottom => i64::MIN,
        };
        assert_eq!(
            e, expected,
            "{s:?}: identity_kind names a different element"
        );
    }
}

/// Describing the law changed no execution: the executor's own values.
#[test]
fn report_execution_is_unchanged() {
    assert_eq!(FoldState::Count.identity(), 0);
    assert_eq!(FoldState::Sum(F).identity(), 0);
    assert_eq!(FoldState::Min(F).identity(), i64::MAX);
    assert_eq!(FoldState::Max(F).identity(), i64::MIN);
    assert_eq!(FoldState::Sum(F).merge(3, 4), 7);
    assert_eq!(FoldState::Sum(F).merge(i64::MAX, 1), i64::MIN);
    assert_eq!(FoldState::Min(F).merge(3, 4), 3);
    assert_eq!(FoldState::Max(F).merge(3, 4), 4);
}
