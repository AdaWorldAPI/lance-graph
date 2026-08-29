//! Revision policy: explicit horizon change under textual / grammatical resistance.
//!
//! # Status
//!
//! LANDED from the session draft (2026-08-29). This module models revision, not
//! truth persistence, and still has **no production write capability** — the
//! output types stop before actual-world mutation by design.
//!
//! `entropy-closure-causal-ground-v1` (PR #1057, merged) names revision as
//! *"the only write-back"* and *"the court of appeal"* in its
//! DETECT→BOUND→PROPOSE→FILTER→GATE→TEST→ACCEPT loop, while `counterfactual.rs`
//! shipped and this module did not. That gap is what this file closes. The intended placement is beside `temporal.rs` and
//! `counterfactual.rs`:
//!
//! - temporal remembers the awareness horizon and durable arrival;
//! - counterfactual explores sealed hypothetical timelines;
//! - revision records whether an encounter changed the interpretive horizon or
//!   merely returned inherited assumptions.
//!
//! `GadamerRevision` is a textual-hermeneutic policy, not a universal truth
//! algorithm. Echoes and closed cycles remain observable history but receive zero
//! additional evidential weight.

use std::array;
use std::fmt;

/// Minimal operations required from fixed-width masks.
///
/// The canonical mask type can implement this trait at integration time. The draft
/// includes implementations for `u64` and fixed arrays of `u64` words.
pub trait EvidenceMask: Clone + PartialEq + Eq + fmt::Debug {
    fn empty() -> Self;
    fn is_empty(&self) -> bool;
    fn union(&self, other: &Self) -> Self;
    fn intersection(&self, other: &Self) -> Self;
    fn difference(&self, other: &Self) -> Self;
    fn is_subset_of(&self, other: &Self) -> bool;

    fn intersects(&self, other: &Self) -> bool {
        !self.intersection(other).is_empty()
    }
}

impl EvidenceMask for u64 {
    fn empty() -> Self {
        0
    }

    fn is_empty(&self) -> bool {
        *self == 0
    }

    fn union(&self, other: &Self) -> Self {
        *self | *other
    }

    fn intersection(&self, other: &Self) -> Self {
        *self & *other
    }

    fn difference(&self, other: &Self) -> Self {
        *self & !*other
    }

    fn is_subset_of(&self, other: &Self) -> bool {
        (*self & !*other) == 0
    }
}

impl<const N: usize> EvidenceMask for [u64; N] {
    fn empty() -> Self {
        [0; N]
    }

    fn is_empty(&self) -> bool {
        self.iter().all(|word| *word == 0)
    }

    fn union(&self, other: &Self) -> Self {
        array::from_fn(|idx| self[idx] | other[idx])
    }

    fn intersection(&self, other: &Self) -> Self {
        array::from_fn(|idx| self[idx] & other[idx])
    }

    fn difference(&self, other: &Self) -> Self {
        array::from_fn(|idx| self[idx] & !other[idx])
    }

    fn is_subset_of(&self, other: &Self) -> bool {
        self.iter()
            .zip(other.iter())
            .all(|(left, right)| (*left & !*right) == 0)
    }
}

/// Opaque identifiers. Replace with canonical planner / OGAR IDs when wired.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct HorizonId(pub u64);
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct QuestionId(pub u64);
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct LanguageId(pub u16);
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct GrammarId(pub u32);
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct CodebookId(pub u32);
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct LensId(pub u32);

/// The reader's explicit interpretive horizon before or after an encounter.
///
/// `A` is intended to become the canonical `temporal::AwarenessRef`. Keeping it
/// generic prevents `revision.rs` from creating its own clock or temporal store.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct InterpretiveHorizon<A, M> {
    pub id: HorizonId,
    pub awareness: A,
    pub question: QuestionId,
    pub language: LanguageId,
    pub grammar: GrammarId,
    pub codebook: CodebookId,
    pub lens: LensId,
    /// Claims currently projected as the working whole.
    pub projected_claims: M,
    /// Independent roots accumulated across genuine encounters.
    pub independent_roots: M,
    /// Earlier interpretations consumed as interpretations, never as fresh evidence.
    pub inherited_roots: M,
    /// Tension deliberately preserved instead of forced into a false synthesis.
    pub unresolved_tension: M,
    pub revision_index: u16,
}

/// What the latest textual / grammatical / philosophical encounter produced.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EncounterEvidence<M> {
    /// Proposed working whole after reading this encounter.
    pub proposed_claims: M,
    /// Actual source / grammar / observation roots contacted independently here.
    pub independent_roots: M,
    /// Derived thoughts, summaries, or inherited interpretations consumed here.
    pub inherited_roots: M,
    /// Parts that resisted the prior projection.
    pub resistance: M,
    /// Contradictions that remain live after the encounter.
    pub contradictions: M,
    /// Parts whose reading changed because the projected whole changed.
    pub affected_parts: M,
}

/// Bounded ancestry summary supplied by temporal / counterfactual provenance.
///
/// No generic graph walk is required inside this module. The caller supplies the
/// fixed-width ancestry projection appropriate for the current SoA trajectory.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BasisView<M> {
    pub ancestry_independent_roots: M,
    pub ancestry_derived_roots: M,
    pub ancestor_claims: M,
    /// Set by the bounded provenance projection when the candidate depends on itself.
    pub closes_cycle: bool,
}

/// What an encounter did to the interpretive horizon.
///
/// # `HorizonFusion` — *Horizontverschmelzung* as thesis × antithesis × synthesis
///
/// This variant is the one the whole policy exists to make earnable rather
/// than assumable. Its three derivation conditions ARE the dialectical triad,
/// and none of them is decorative:
///
/// ```text
/// thesis      prior.projected_claims        the working whole brought in
/// antithesis  encounter.contradictions      what refused to be absorbed
/// synthesis   introduced ∪ preserved        what stands after the collision
/// ```
///
/// **It is Gadamer, not Hegel: the tension is PRESERVED, never dissolved.**
/// `unresolved_tension` accumulates by union and is never cleared, so a fusion
/// carries its contradiction forward as durable structure. A synthesis that
/// *resolved* its antithesis would be exactly the false synthesis this module
/// is built to refuse.
///
/// **Synthesis must be EARNED.** [`RevisionKind::ContradictionPreserved`] and
/// `HorizonFusion` see the same contradiction; the single thing separating
/// them is `has_new_root` — a genuinely new independent root, not present in
/// ancestry. Without it the outcome is `ContradictionPreserved` →
/// [`EvidentialEffect::Suspend`]: the tension is held open, and no evidential
/// weight is minted. With it, `HorizonFusion` →
/// [`EvidentialEffect::IncreaseEligible`].
///
/// That asymmetry is the anti-laundering invariant in executable form: no
/// amount of re-reading inherited material can produce a synthesis, because
/// re-reading yields no new independent root. Fusion of two horizons requires
/// that at least one of them actually touched the world.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RevisionKind {
    IndependentConfirmation,
    Reinterpretation,
    HorizonExpansion,
    HorizonFusion,
    AssumptionExposed,
    ContradictionPreserved,
    Suspended,
    Echo,
    ClosedCycle,
}

/// Evidential consequence is deliberately coarser than numerical confidence.
/// Downstream belief machinery may translate it, but echo/cycle can never inflate it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum EvidentialEffect {
    /// At least one genuinely new independent root was introduced.
    IncreaseEligible,
    /// Semantic movement occurred, but no new independent support was gained.
    NoIncrease,
    /// The interpretation should remain unresolved pending grounding.
    Suspend,
}

/// Explicit delta between the prior and resulting horizons.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RevisionDelta<A, M> {
    pub kind: RevisionKind,
    pub evidential_effect: EvidentialEffect,
    pub prior: InterpretiveHorizon<A, M>,
    pub resulting: InterpretiveHorizon<A, M>,
    pub preserved_claims: M,
    pub introduced_claims: M,
    pub withdrawn_claims: M,
    pub revised_claims: M,
    pub new_independent_roots: M,
    pub inherited_roots: M,
    pub resistance: M,
    pub contradictions: M,
    pub affected_parts: M,
}

/// Textual revision policy seam.
pub trait RevisionPolicy<A, M: EvidenceMask> {
    fn revise(
        &self,
        prior: &InterpretiveHorizon<A, M>,
        encounter: &EncounterEvidence<M>,
        ancestry: &BasisView<M>,
    ) -> RevisionDelta<A, M>
    where
        A: Clone;
}

/// Gadamer-shaped revision policy.
///
/// The policy makes the prior projection explicit, requires resistance or new
/// grounding for productive movement, preserves unresolved contradiction, and
/// prevents inherited derivations from counting as independent confirmation.
#[derive(Debug, Clone, Copy, Default)]
pub struct GadamerRevision;

impl<A, M: EvidenceMask> RevisionPolicy<A, M> for GadamerRevision {
    fn revise(
        &self,
        prior: &InterpretiveHorizon<A, M>,
        encounter: &EncounterEvidence<M>,
        ancestry: &BasisView<M>,
    ) -> RevisionDelta<A, M>
    where
        A: Clone,
    {
        let new_independent_roots = encounter
            .independent_roots
            .difference(&ancestry.ancestry_independent_roots);
        let preserved_claims = prior
            .projected_claims
            .intersection(&encounter.proposed_claims);
        let introduced_claims = encounter
            .proposed_claims
            .difference(&prior.projected_claims);
        let withdrawn_claims = prior
            .projected_claims
            .difference(&encounter.proposed_claims);
        let revised_claims = introduced_claims.union(&withdrawn_claims);

        let has_new_root = !new_independent_roots.is_empty();
        let has_resistance = !encounter.resistance.is_empty();
        let has_contradiction = !encounter.contradictions.is_empty();
        let same_projection = introduced_claims.is_empty() && withdrawn_claims.is_empty();
        let recycles_ancestor_claims = encounter
            .proposed_claims
            .is_subset_of(&ancestry.ancestor_claims);

        let kind = if ancestry.closes_cycle && !has_new_root && !has_resistance {
            RevisionKind::ClosedCycle
        } else if !has_new_root && !has_resistance && (same_projection || recycles_ancestor_claims)
        {
            RevisionKind::Echo
        } else if has_contradiction
            && (!introduced_claims.is_empty() || !preserved_claims.is_empty())
            && has_new_root
        {
            RevisionKind::HorizonFusion
        } else if has_contradiction {
            RevisionKind::ContradictionPreserved
        } else if has_resistance && !withdrawn_claims.is_empty() {
            RevisionKind::AssumptionExposed
        } else if has_new_root && same_projection {
            RevisionKind::IndependentConfirmation
        } else if has_new_root {
            RevisionKind::HorizonExpansion
        } else if has_resistance || !revised_claims.is_empty() {
            RevisionKind::Reinterpretation
        } else {
            RevisionKind::Suspended
        };

        // The three `IncreaseEligible` kinds are each derived ABOVE under a
        // `has_new_root` precondition, so a rootless confirmation / expansion /
        // fusion is unreachable BY CONSTRUCTION. The draft expressed this as a
        // match guard `if has_new_root` followed by a second listing of the same
        // three variants falling through to `NoIncrease`. That guard can never
        // be false, and the arms it shadowed are unreachable — a guard that
        // cannot fail is the vacuous-guard defect (`CLAUDE.md` falsifiability
        // rule), and the dead arms read as policy for a state that cannot occur.
        //
        // Asserted instead of re-guarded, so that relaxing a `kind` condition
        // above trips here in debug rather than silently minting evidential
        // weight for a rootless synthesis.
        debug_assert!(
            !matches!(
                kind,
                RevisionKind::IndependentConfirmation
                    | RevisionKind::HorizonExpansion
                    | RevisionKind::HorizonFusion
            ) || has_new_root,
            "an IncreaseEligible kind was derived without a new independent root"
        );

        let evidential_effect = match kind {
            RevisionKind::IndependentConfirmation
            | RevisionKind::HorizonExpansion
            | RevisionKind::HorizonFusion => EvidentialEffect::IncreaseEligible,
            RevisionKind::ContradictionPreserved | RevisionKind::Suspended => {
                EvidentialEffect::Suspend
            }
            RevisionKind::Reinterpretation
            | RevisionKind::AssumptionExposed
            | RevisionKind::Echo
            | RevisionKind::ClosedCycle => EvidentialEffect::NoIncrease,
        };

        let resulting = InterpretiveHorizon {
            id: HorizonId(prior.id.0.wrapping_add(1)),
            awareness: prior.awareness.clone(),
            question: prior.question,
            language: prior.language,
            grammar: prior.grammar,
            codebook: prior.codebook,
            lens: prior.lens,
            projected_claims: encounter.proposed_claims.clone(),
            independent_roots: prior.independent_roots.union(&new_independent_roots),
            inherited_roots: prior.inherited_roots.union(&encounter.inherited_roots),
            unresolved_tension: prior.unresolved_tension.union(&encounter.contradictions),
            revision_index: prior.revision_index.wrapping_add(1),
        };

        RevisionDelta {
            kind,
            evidential_effect,
            prior: prior.clone(),
            resulting,
            preserved_claims,
            introduced_claims,
            withdrawn_claims,
            revised_claims,
            new_independent_roots,
            inherited_roots: encounter.inherited_roots.clone(),
            resistance: encounter.resistance.clone(),
            contradictions: encounter.contradictions.clone(),
            affected_parts: encounter.affected_parts.clone(),
        }
    }
}

/// Output types deliberately stop before actual-world mutation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RevisionOutcome<A, M> {
    BranchUpdate(RevisionDelta<A, M>),
    HypothesisReport(HypothesisReport<M>),
    GroundingRequest(GroundingRequest<M>),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HypothesisReport<M> {
    pub claims: M,
    pub contradictions: M,
    pub kind: RevisionKind,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GroundingRequest<M> {
    pub claims_to_test: M,
    pub missing_independent_roots: M,
    pub resistant_parts: M,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn horizon(claims: u64, roots: u64) -> InterpretiveHorizon<u64, u64> {
        InterpretiveHorizon {
            id: HorizonId(1),
            awareness: 100,
            question: QuestionId(1),
            language: LanguageId(1),
            grammar: GrammarId(1),
            codebook: CodebookId(1),
            lens: LensId(1),
            projected_claims: claims,
            independent_roots: roots,
            inherited_roots: 0,
            unresolved_tension: 0,
            revision_index: 0,
        }
    }

    fn ancestry(claims: u64, roots: u64, cycle: bool) -> BasisView<u64> {
        BasisView {
            ancestry_independent_roots: roots,
            ancestry_derived_roots: claims,
            ancestor_claims: claims,
            closes_cycle: cycle,
        }
    }

    fn encounter(
        claims: u64,
        independent: u64,
        inherited: u64,
        resistance: u64,
        contradictions: u64,
    ) -> EncounterEvidence<u64> {
        EncounterEvidence {
            proposed_claims: claims,
            independent_roots: independent,
            inherited_roots: inherited,
            resistance,
            contradictions,
            affected_parts: resistance | contradictions,
        }
    }

    #[test]
    fn repeated_derived_claim_is_an_echo_and_adds_no_weight() {
        let prior = horizon(0b001, 0b001);
        let result = GadamerRevision.revise(
            &prior,
            &encounter(0b001, 0b001, 0b100, 0, 0),
            &ancestry(0b001, 0b001, false),
        );
        assert_eq!(result.kind, RevisionKind::Echo);
        assert_eq!(result.evidential_effect, EvidentialEffect::NoIncrease);
        assert!(result.new_independent_roots.is_empty());
    }

    #[test]
    fn self_supporting_ancestry_is_a_closed_cycle() {
        let prior = horizon(0b001, 0b001);
        let result = GadamerRevision.revise(
            &prior,
            &encounter(0b001, 0b001, 0b001, 0, 0),
            &ancestry(0b001, 0b001, true),
        );
        assert_eq!(result.kind, RevisionKind::ClosedCycle);
        assert_eq!(result.evidential_effect, EvidentialEffect::NoIncrease);
    }

    #[test]
    fn same_claim_from_a_new_root_is_independent_confirmation() {
        let prior = horizon(0b001, 0b001);
        let result = GadamerRevision.revise(
            &prior,
            &encounter(0b001, 0b011, 0, 0, 0),
            &ancestry(0b001, 0b001, false),
        );
        assert_eq!(result.kind, RevisionKind::IndependentConfirmation);
        assert_eq!(result.evidential_effect, EvidentialEffect::IncreaseEligible);
        assert_eq!(result.new_independent_roots, 0b010);
    }

    #[test]
    fn resistance_that_withdraws_a_claim_exposes_an_assumption() {
        let prior = horizon(0b011, 0b001);
        let result = GadamerRevision.revise(
            &prior,
            &encounter(0b001, 0b001, 0, 0b010, 0),
            &ancestry(0b011, 0b001, false),
        );
        assert_eq!(result.kind, RevisionKind::AssumptionExposed);
        assert_eq!(result.withdrawn_claims, 0b010);
        assert_eq!(result.evidential_effect, EvidentialEffect::NoIncrease);
    }

    #[test]
    fn independent_horizons_can_fuse_without_erasing_tension() {
        let prior = horizon(0b001, 0b001);
        let result = GadamerRevision.revise(
            &prior,
            &encounter(0b011, 0b011, 0, 0b010, 0b100),
            &ancestry(0b001, 0b001, false),
        );
        assert_eq!(result.kind, RevisionKind::HorizonFusion);
        assert_eq!(result.evidential_effect, EvidentialEffect::IncreaseEligible);
        assert_eq!(result.resulting.unresolved_tension, 0b100);
    }

    /// THE holy-grail falsifier: fusion and mere preservation see the SAME
    /// contradiction, and differ only by whether a new independent root exists.
    /// Without one, synthesis must NOT happen and no weight may be minted.
    ///
    /// Paired with `independent_horizons_can_fuse_without_erasing_tension`,
    /// which is the identical encounter WITH a new root. Two-sided by
    /// construction: neither half can pass for the other's reason.
    #[test]
    fn contradiction_without_a_new_root_suspends_instead_of_synthesising() {
        let prior = horizon(0b001, 0b001);
        // Identical to the fusion case EXCEPT independent_roots ⊆ ancestry,
        // so `new_independent_roots` is empty.
        let result = GadamerRevision.revise(
            &prior,
            &encounter(0b011, 0b001, 0, 0b010, 0b100),
            &ancestry(0b001, 0b001, false),
        );
        assert_eq!(
            result.kind,
            RevisionKind::ContradictionPreserved,
            "no new independent root ⇒ the tension is held, not synthesised"
        );
        assert_eq!(result.evidential_effect, EvidentialEffect::Suspend);
        assert!(
            result.new_independent_roots.is_empty(),
            "anti-vacuity: the discriminating quantity really is absent here"
        );
        // And the contradiction is still carried forward as durable structure.
        assert_eq!(result.resulting.unresolved_tension, 0b100);
    }

    /// Gadamer, not Hegel: fusion never CLEARS prior tension, it unions onto it.
    #[test]
    fn fusion_accumulates_tension_and_never_clears_prior_tension() {
        let mut prior = horizon(0b001, 0b001);
        prior.unresolved_tension = 0b1000; // tension carried in from before
        let result = GadamerRevision.revise(
            &prior,
            &encounter(0b011, 0b011, 0, 0b010, 0b100),
            &ancestry(0b001, 0b001, false),
        );
        assert_eq!(result.kind, RevisionKind::HorizonFusion);
        assert_eq!(
            result.resulting.unresolved_tension, 0b1100,
            "prior tension survives the synthesis; a fusion that resolved it \
             would be the false synthesis this policy refuses"
        );
    }

    /// Every `IncreaseEligible` kind is reachable ONLY with a new independent
    /// root — the invariant the removed match guard used to shadow.
    #[test]
    fn no_increase_eligible_outcome_is_reachable_without_a_new_root() {
        let prior = horizon(0b011, 0b001);
        // Sweep the rootless encounter space: no `independent_roots` beyond
        // ancestry, across every combination of resistance/contradiction/shape.
        for claims in [0b001_u64, 0b011, 0b111] {
            for resistance in [0_u64, 0b010] {
                for contradictions in [0_u64, 0b100] {
                    let result = GadamerRevision.revise(
                        &prior,
                        &encounter(claims, 0b001, 0, resistance, contradictions),
                        &ancestry(0b011, 0b001, false),
                    );
                    assert_ne!(
                        result.evidential_effect,
                        EvidentialEffect::IncreaseEligible,
                        "rootless encounter minted weight: claims={claims:b} \
                         resistance={resistance:b} contradictions={contradictions:b}"
                    );
                }
            }
        }
    }

    #[test]
    fn fixed_word_arrays_are_supported_without_heap_masks() {
        let left = [0b001_u64, 0b100];
        let right = [0b010_u64, 0b100];
        assert_eq!(left.union(&right), [0b011, 0b100]);
        assert_eq!(left.intersection(&right), [0, 0b100]);
    }
}
