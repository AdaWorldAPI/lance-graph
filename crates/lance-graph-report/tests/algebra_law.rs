//! Report's `FoldState` and lane-fold's `Hom` describe their merges through
//! the contract's `AlgebraLaw`. The law is metadata: these tests check that
//! both crates describe the shared folds identically, and that report's
//! description agrees with what `FoldState::merge` / `identity` actually do.

use lance_graph_contract::algebra_law::{AlgebraDescriptor, AlgebraLaw, IdentityKind};
use lance_graph_lane_fold::Hom;
use lance_graph_report::ids::FieldId;
use lance_graph_report::plan::FoldState;

const F: FieldId = FieldId(7);

/// The conceptual twins. `Hom::Exists` has no report twin.
fn twins() -> [(FoldState, Hom); 4] {
    [
        (FoldState::Count, Hom::Count),
        (FoldState::Sum(F), Hom::Sum),
        (FoldState::Min(F), Hom::Min),
        (FoldState::Max(F), Hom::Max),
    ]
}

/// Sample states including both extremes, so wrapping and saturation edges
/// are exercised, not just small positives.
const SAMPLES: [i64; 7] = [i64::MIN, -9, -1, 0, 1, 42, i64::MAX];

#[test]
fn report_and_lane_fold_agree_on_shared_folds() {
    for (state, hom) in twins() {
        assert_eq!(
            state.algebra_law(),
            hom.algebra_law(),
            "{state:?} vs {hom:?}"
        );
    }
}

#[test]
fn min_and_max_are_idempotent_sum_and_count_are_not() {
    assert!(FoldState::Min(F).algebra_law().idempotent);
    assert!(FoldState::Max(F).algebra_law().idempotent);
    assert!(!FoldState::Sum(F).algebra_law().idempotent);
    assert!(!FoldState::Count.algebra_law().idempotent);
}

#[test]
fn exists_has_metadata_without_a_numeric_identity() {
    let law: AlgebraLaw = Hom::Exists.algebra_law();
    // A category, not a value: the contract holds no `i64` for `false`.
    assert_eq!(law.identity_kind, IdentityKind::Bottom);
    assert!(law.associative && law.commutative && law.idempotent);
}

#[test]
fn nothing_claims_invertibility_or_order() {
    let all = [
        FoldState::Count.algebra_law(),
        FoldState::Sum(F).algebra_law(),
        FoldState::Min(F).algebra_law(),
        FoldState::Max(F).algebra_law(),
        Hom::Exists.algebra_law(),
    ];
    for law in all {
        assert!(
            !law.invertible,
            "no tested remove-a-contribution path exists"
        );
        assert!(!law.ordered);
    }
}

/// Two-sided: every flag report declares must hold on `merge`, and every flag
/// it withholds must have a witness that `merge` really lacks it. A wrong
/// descriptor fails here in either direction.
#[test]
fn report_descriptor_matches_its_executor() {
    let states = [
        FoldState::Count,
        FoldState::Sum(F),
        FoldState::Min(F),
        FoldState::Max(F),
    ];
    for s in states {
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

        // The identity is the executor's; the kind must name it correctly.
        let e = s.identity();
        assert!(
            SAMPLES.iter().all(|&a| m(e, a) == a && m(a, e) == a),
            "{s:?} identity"
        );
        let expected = match law.identity_kind {
            IdentityKind::Zero => 0,
            IdentityKind::Top => i64::MAX,
            IdentityKind::Bottom => i64::MIN,
        };
        assert_eq!(e, expected, "{s:?} identity_kind names a different element");
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
