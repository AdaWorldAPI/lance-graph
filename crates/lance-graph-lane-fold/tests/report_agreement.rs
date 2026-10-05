//! Cross-executor conformance: lane-fold's `Hom` and report's `FoldState`
//! must describe the folds they share with identical `AlgebraLaw` metadata.
//!
//! Report is a DEV-dependency of lane-fold for this test only. In production
//! lane-fold depends on the contract and nothing else.

use lance_graph_contract::algebra_law::AlgebraDescriptor;
use lance_graph_lane_fold::Hom;
use lance_graph_report::ids::FieldId;
use lance_graph_report::plan::FoldState;

const F: FieldId = FieldId(7);

#[test]
fn hom_and_report_agree_on_shared_folds() {
    // `Hom::Exists` has no report twin and is checked in lane-fold's own tests.
    let twins = [
        (Hom::Count, FoldState::Count),
        (Hom::Sum, FoldState::Sum(F)),
        (Hom::Min, FoldState::Min(F)),
        (Hom::Max, FoldState::Max(F)),
    ];
    for (hom, state) in twins {
        assert_eq!(
            hom.algebra_law(),
            state.algebra_law(),
            "{hom:?} vs {state:?}"
        );
    }
}
