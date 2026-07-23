//! S8 — GraphBias → tactic selection (D-DIA-V2-B).
//!
//! The dialectic loop's SECOND axis. `advance_on_gate` moves the PHASE (shipped,
//! kanban); THIS selects WHICH of the five NARS tactics fires inside one
//! `CognitiveWork` step, driven by the graph's own health reading
//! ([`GraphBias`], `contract::sensorium`). The mapping is a pure LUT — no state,
//! no new signal type: `GraphBias` / `GraphSignals` / `suggested_bias` are
//! reused from the contract, and the five tactics from [`crate::nars::tactics`].
//!
//! The pairing is falsifiable, not decorative: each bias names a graph condition
//! that creates a specific tactic's PRECONDITION, so the bias-selected tactic
//! should FIRE on a graph exhibiting that bias while mismatched tactics return a
//! `ReasoningGap`. The confusion-matrix probe
//! (`examples/tactic_select_confusion.rs`) measures that the diagonal dominates —
//! the E-BASIN-WIDTH discipline applied to selection (beat a bias-blind null).

use crate::nars::tactics::{ASC_ID, CAS_ID, CR_ID, RCR_ID, TR_ID};
use lance_graph_contract::sensorium::GraphBias;

/// One of the five dialectic MOVES the S8 LUT can select — the recipe-level
/// choice, keyed to `contract::recipe_dispatch` ids {4, 6, 8, 7, 11}.
///
/// Distinct from two adjacent enums, deliberately:
/// - [`crate::nars::tactics::Tactic`] is the four-variant PROVENANCE tag stamped
///   on a produced `Candidate` — CAS splits into `CasUp`/`CasDown` there (the
///   two directions of a derivation). `TacticChoice` is the SELECTION currency:
///   one `Cas`, and the two revision moves (`Asc`/`Cr`) that produce no
///   `Candidate` and so are absent from `Tactic`.
/// - `contract::recipe_dispatch::RecipeInference` is the inference FAMILY
///   (Deduction/Induction/Abduction/Revision/Synthesis) — coarser: it collapses
///   ASC and CR both to `Revision`. `TacticChoice` keeps them apart because they
///   are different moves (self-critique vs thesis+antithesis synthesis).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TacticChoice {
    /// RCR — abduction over a shared predicate (recipe #4).
    Rcr,
    /// TR — analogical divergence across a similarity sibling (recipe #6).
    Tr,
    /// CAS — abstraction (recipe #8; up = induction, down = deduction).
    Cas,
    /// ASC — self-critique via independently-sourced counter-evidence (recipe #7).
    Asc,
    /// CR — dialectic synthesis of thesis + antithesis (recipe #11).
    Cr,
}

impl TacticChoice {
    /// The `contract::recipe_dispatch` id this move implements.
    #[must_use]
    pub fn recipe_id(self) -> u8 {
        match self {
            TacticChoice::Rcr => RCR_ID,
            TacticChoice::Tr => TR_ID,
            TacticChoice::Cas => CAS_ID,
            TacticChoice::Asc => ASC_ID,
            TacticChoice::Cr => CR_ID,
        }
    }
}

/// Select the tactic best matched to a graph condition (S8 — the second axis of
/// the dialectic loop, orthogonal to `advance_on_gate` PHASE movement).
///
/// Each arm pairs the bias's graph condition with the tactic whose PRECONDITION
/// that condition creates — the pairing the confusion-matrix falsifier
/// (`examples/tactic_select_confusion.rs`) measures:
///
/// | bias       | graph condition          | tactic  | why                                                     |
/// |------------|--------------------------|---------|---------------------------------------------------------|
/// | `Resolve`  | high contradiction       | **CR**  | thesis + antithesis on one statement → revision / CHOICE |
/// | `Explore`  | high entropy             | **RCR** | uncertainty → abduce explanatory hypotheses to seek     |
/// | `Exploit`  | rich, consistent graph   | **CAS** | a settled `is_a` hierarchy → deduction / induction over it |
/// | `Adapt`    | high plasticity          | **TR**  | flux → transfer a belief analogically to a sibling      |
/// | `Stagnant` | stuck, no revision       | **ASC** | inject independent counter-evidence to unstick          |
/// | `Balanced` | normal operation         | **CAS** | steady deductive progress — the workhorse default       |
#[must_use]
pub fn tactic_for_bias(bias: GraphBias) -> TacticChoice {
    match bias {
        GraphBias::Resolve => TacticChoice::Cr,
        GraphBias::Explore => TacticChoice::Rcr,
        GraphBias::Exploit => TacticChoice::Cas,
        GraphBias::Adapt => TacticChoice::Tr,
        GraphBias::Stagnant => TacticChoice::Asc,
        GraphBias::Balanced => TacticChoice::Cas,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use lance_graph_contract::recipe_dispatch::{inference, RecipeInference};
    use lance_graph_contract::sensorium::{suggested_bias, GraphSignals};

    /// Every bias maps to a move whose recipe id is one of the five tactics —
    /// the LUT is total and stays inside the pinned taxonomy.
    #[test]
    fn every_bias_maps_to_a_tactic_recipe_id() {
        let biases = [
            GraphBias::Resolve,
            GraphBias::Explore,
            GraphBias::Exploit,
            GraphBias::Adapt,
            GraphBias::Stagnant,
            GraphBias::Balanced,
        ];
        for b in biases {
            let id = tactic_for_bias(b).recipe_id();
            assert!(
                matches!(id, RCR_ID | TR_ID | CAS_ID | ASC_ID | CR_ID),
                "bias {b:?} → recipe id {id} is not a tactic id"
            );
        }
    }

    /// The mapping is the intended one (the falsifiable pairing, not an accident).
    #[test]
    fn mapping_is_the_intended_pairing() {
        assert_eq!(tactic_for_bias(GraphBias::Resolve), TacticChoice::Cr);
        assert_eq!(tactic_for_bias(GraphBias::Explore), TacticChoice::Rcr);
        assert_eq!(tactic_for_bias(GraphBias::Exploit), TacticChoice::Cas);
        assert_eq!(tactic_for_bias(GraphBias::Adapt), TacticChoice::Tr);
        assert_eq!(tactic_for_bias(GraphBias::Stagnant), TacticChoice::Asc);
        assert_eq!(tactic_for_bias(GraphBias::Balanced), TacticChoice::Cas);
    }

    /// The two revision moves stay DISTINCT here even though `RecipeInference`
    /// collapses them — the reason `TacticChoice` is not that coarser enum.
    #[test]
    fn asc_and_cr_are_distinct_but_share_the_revision_family() {
        assert_ne!(TacticChoice::Asc, TacticChoice::Cr);
        assert_eq!(
            inference(TacticChoice::Asc.recipe_id()),
            RecipeInference::Revision
        );
        assert_eq!(
            inference(TacticChoice::Cr.recipe_id()),
            RecipeInference::Revision
        );
    }

    /// End-to-end: a signal reading routes through `suggested_bias` to the
    /// matched tactic — the whole S8 path (`GraphSignals → GraphBias → tactic`).
    #[test]
    fn signals_route_to_the_matched_tactic() {
        // High contradiction → Resolve → CR.
        let s = GraphSignals {
            contradiction_rate: 0.4,
            ..Default::default()
        };
        assert_eq!(tactic_for_bias(suggested_bias(&s)), TacticChoice::Cr);

        // High entropy → Explore → RCR.
        let s = GraphSignals {
            truth_entropy: 0.8,
            ..Default::default()
        };
        assert_eq!(tactic_for_bias(suggested_bias(&s)), TacticChoice::Rcr);

        // Stuck (low revision, mid entropy) → Stagnant → ASC.
        let s = GraphSignals {
            revision_velocity: 0.02,
            truth_entropy: 0.6,
            ..Default::default()
        };
        assert_eq!(tactic_for_bias(suggested_bias(&s)), TacticChoice::Asc);
    }
}
