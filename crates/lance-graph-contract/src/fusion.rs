//! Horizontverschmelzung: candidate-synthesis construction over two horizons.
//!
//! # The triptych — kept violently separate (operator, 2026-08-29)
//!
//! ```text
//! fusion.rs         GENERATES S   "what interpretation makes the relation
//!                                  between these positions intelligible?"
//! counterfactual.rs ATTACKS   S   "remove S — does explanatory structure
//!                                  collapse?"  (explanatory necessity)
//! revision.rs       LICENSES  S   "given provenance, independence,
//!                                  contradiction and the counterfactual
//!                                  result, what mutation is warranted?"
//! ```
//!
//! Three different questions. This module answers only the first, and its
//! output is a CANDIDATE — never a fact. Nothing here writes back; the write
//! path is [`crate::revision`], which PR #1057 ratifies as "the only
//! write-back" and "the court of appeal".
//!
//! # What this refuses to do
//!
//! The cheap outcomes are `thesis wins`, `antithesis wins`, and a 50/50
//! compromise. None is a synthesis. The question asked instead is:
//!
//! > **What assumption must change so that BOTH horizons become explicable,
//! > without laundering either one's evidence?**
//!
//! **A fake intelligence always synthesises.** A serious one can conclude *"I
//! understand both horizons better now, and they still disagree"* —
//! [`FusionOutcome::IrreducibleTension`]. That variant is load-bearing, and
//! `irreducible_tension_is_reachable_when_no_assumption_is_revisable` proves
//! it can actually occur.
//!
//! # An assumption IS an inherited root
//!
//! This is the mechanism that makes the refusal checkable rather than
//! stylistic. [`crate::revision::InterpretiveHorizon`] already separates
//! `independent_roots` (contact with the world) from `inherited_roots`
//! (interpretations consumed as interpretations). So:
//!
//! - an assumption held ONLY as an inherited root is **revisable** — dropping
//!   it withdraws no independent support;
//! - an assumption that is ALSO independently grounded is **not** freely
//!   revisable — dropping it would discard evidence.
//!
//! Synthesis requires a revisable assumption. With none, the contradiction is
//! irreducible and the honest answer is to say so.
//!
//! # The anti-alchemy law
//!
//! > **Understanding may increase without evidence increasing.**
//!
//! Two horizons that look like independent witnesses but trace to the same
//! root are ONE witness. [`FusionReceipt::shared_roots`] records that
//! collapse, and such a fusion is never [`EvidentialEffect::IncreaseEligible`]
//! however intelligible the synthesis becomes. Coherence is allowed to rise;
//! evidential weight is not.
//!
//! # No confidence scalar
//!
//! Deliberately absent from [`FusionReceipt`]: any `synthesis_confidence:
//! f64`. Scalar confidence stays in NARS `(f, c)`; the band in CE64 bits
//! 61-63 stays a permission level (PR #1057). A receipt carries structure and
//! provenance, never a number that invites averaging.

use crate::revision::{EvidenceMask, EvidentialEffect, HorizonId, InterpretiveHorizon};

/// What a fusion attempt concluded. Ordered coarse-to-fine by how much the
/// collision actually produced.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FusionOutcome<M> {
    /// A revisable assumption was found whose withdrawal makes both horizons
    /// explicable. The originals are NOT overwritten — see [`FusionReceipt`].
    Synthesis(SynthesizedClaim<M>),
    /// The antithesis had no independent grounding of its own.
    ThesisSurvives,
    /// The thesis had no independent grounding of its own.
    AntithesisSurvives,
    /// The horizons never actually contradicted — different questions, not
    /// rival answers.
    Complementary,
    /// Both horizons are independently grounded, they genuinely conflict, and
    /// NO assumption is revisable without discarding evidence. The honest
    /// terminal state, and the one a fake intelligence never reaches.
    IrreducibleTension,
    /// Neither side is independently grounded; the collision cannot be judged.
    Suspended,
    /// As `Suspended`, but the missing grounding is nameable.
    AskForMeans { missing_independent_roots: M },
}

/// The candidate produced by a successful fusion. A CANDIDATE — counterfactual
/// attacks it next, revision licenses it after that.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SynthesizedClaim<M> {
    /// What both horizons still assert together.
    pub preserved: M,
    /// Assumptions withdrawn to make the collision explicable. Revisable by
    /// construction: inherited, never independently grounded.
    pub revised_assumptions: M,
    /// Tension the synthesis does NOT dissolve. Gadamer, not Hegel.
    pub surviving_tension: M,
}

/// The audit record of a fusion attempt.
///
/// **The synthesis never overwrites thesis or antithesis.** Both horizons stay
/// recoverable, because tomorrow some evidence may falsify the synthesis and
/// the system must then be able to ask why it looked compelling, which support
/// was independent, which assumptions were fused, and which contradiction was
/// merely hidden. Without that, "revision" is history rewriting.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FusionReceipt<M> {
    pub thesis: HorizonId,
    pub antithesis: HorizonId,
    pub thesis_claims: M,
    pub antithesis_claims: M,
    /// Roots reached independently by BOTH sides — the common-ancestry
    /// collapse. Non-empty here means the two "witnesses" overlap.
    pub shared_roots: M,
    /// Roots held by exactly one side. This, not the union, is what makes a
    /// fusion evidentially eligible.
    pub disjoint_roots: M,
    pub inherited_roots: M,
    pub revisable_assumptions: M,
    pub surviving_tension: M,
    pub outcome: FusionOutcome<M>,
    pub evidential_effect: EvidentialEffect,
}

/// Fuse two interpretive horizons into a candidate synthesis.
///
/// `contradiction` is the mask of claims the caller has established as
/// genuinely conflicting between the two horizons — fusion does not infer
/// conflict, it is told where conflict is.
pub fn fuse<A, M: EvidenceMask>(
    thesis: &InterpretiveHorizon<A, M>,
    antithesis: &InterpretiveHorizon<A, M>,
    contradiction: &M,
) -> FusionReceipt<M> {
    let shared_roots = thesis
        .independent_roots
        .intersection(&antithesis.independent_roots);
    let thesis_only = thesis
        .independent_roots
        .difference(&antithesis.independent_roots);
    let antithesis_only = antithesis
        .independent_roots
        .difference(&thesis.independent_roots);
    let disjoint_roots = thesis_only.union(&antithesis_only);

    let inherited_roots = thesis.inherited_roots.union(&antithesis.inherited_roots);
    let all_independent = thesis
        .independent_roots
        .union(&antithesis.independent_roots);
    // An assumption is revisable iff it is inherited and NOT independently
    // grounded: withdrawing it discards interpretation, never evidence.
    let revisable_assumptions = inherited_roots.difference(&all_independent);

    let preserved = thesis
        .projected_claims
        .intersection(&antithesis.projected_claims);
    let surviving_tension = thesis
        .unresolved_tension
        .union(&antithesis.unresolved_tension)
        .union(contradiction);

    let thesis_grounded = !thesis_only.is_empty();
    let antithesis_grounded = !antithesis_only.is_empty();
    let has_contradiction = !contradiction.is_empty();

    let outcome = if !has_contradiction {
        FusionOutcome::Complementary
    } else if !thesis_grounded && !antithesis_grounded {
        // Includes the two-witnesses-one-source case: identical roots leave
        // both `*_only` masks empty however large the shared root set is.
        if revisable_assumptions.is_empty() {
            FusionOutcome::Suspended
        } else {
            FusionOutcome::AskForMeans {
                missing_independent_roots: revisable_assumptions.clone(),
            }
        }
    } else if thesis_grounded && !antithesis_grounded {
        FusionOutcome::ThesisSurvives
    } else if antithesis_grounded && !thesis_grounded {
        FusionOutcome::AntithesisSurvives
    } else if revisable_assumptions.is_empty() {
        // Both independently grounded, genuinely conflicting, and nothing may
        // be withdrawn without discarding evidence.
        FusionOutcome::IrreducibleTension
    } else {
        FusionOutcome::Synthesis(SynthesizedClaim {
            preserved: preserved.clone(),
            revised_assumptions: revisable_assumptions.clone(),
            surviving_tension: surviving_tension.clone(),
        })
    };

    // The anti-alchemy law. Only a genuinely two-witness fusion is eligible;
    // shared ancestry, however intelligible the result, is not.
    let evidential_effect = match &outcome {
        FusionOutcome::Synthesis(_) if thesis_grounded && antithesis_grounded => {
            EvidentialEffect::IncreaseEligible
        }
        FusionOutcome::Suspended
        | FusionOutcome::AskForMeans { .. }
        | FusionOutcome::IrreducibleTension => EvidentialEffect::Suspend,
        _ => EvidentialEffect::NoIncrease,
    };

    FusionReceipt {
        thesis: thesis.id,
        antithesis: antithesis.id,
        thesis_claims: thesis.projected_claims.clone(),
        antithesis_claims: antithesis.projected_claims.clone(),
        shared_roots,
        disjoint_roots,
        inherited_roots,
        revisable_assumptions,
        surviving_tension,
        outcome,
        evidential_effect,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::revision::{CodebookId, GrammarId, LanguageId, LensId, QuestionId};

    fn horizon(
        id: u64,
        claims: u64,
        independent: u64,
        inherited: u64,
    ) -> InterpretiveHorizon<u64, u64> {
        InterpretiveHorizon {
            id: HorizonId(id),
            awareness: 100,
            question: QuestionId(1),
            language: LanguageId(1),
            grammar: GrammarId(1),
            codebook: CodebookId(1),
            lens: LensId(1),
            projected_claims: claims,
            independent_roots: independent,
            inherited_roots: inherited,
            unresolved_tension: 0,
            revision_index: 0,
        }
    }

    /// THE anti-alchemy falsifier. Two horizons that look like independent
    /// witnesses but both trace to source X (`0b001`) are ONE witness. However
    /// intelligible the collision becomes, it may not mint evidential weight.
    #[test]
    fn two_witnesses_sharing_one_root_never_become_evidentially_eligible() {
        let t = horizon(1, 0b0001, 0b001, 0b0110); // T <- X, via A,B
        let a = horizon(2, 0b0010, 0b001, 0b1000); // A <- X, via D
        let r = fuse(&t, &a, &0b0100);

        assert_eq!(r.shared_roots, 0b001, "the common ancestor is recorded");
        assert!(
            r.disjoint_roots.is_empty(),
            "anti-vacuity: neither side has grounding the other lacks"
        );
        assert_ne!(
            r.evidential_effect,
            EvidentialEffect::IncreaseEligible,
            "shared ancestry must never mint weight"
        );
        assert!(matches!(
            r.outcome,
            FusionOutcome::Suspended | FusionOutcome::AskForMeans { .. }
        ));
    }

    /// The paired half: genuinely disjoint roots, and a revisable assumption,
    /// DO earn a synthesis. Without this the test above could pass by the
    /// engine simply never synthesising.
    #[test]
    fn disjoint_roots_plus_a_revisable_assumption_earn_a_synthesis() {
        let t = horizon(1, 0b0001, 0b0001, 0b0100); // T <- X
        let a = horizon(2, 0b0010, 0b0010, 0b0100); // A <- Y, X ⟂ Y
        let r = fuse(&t, &a, &0b1000);

        assert!(r.shared_roots.is_empty(), "genuinely two witnesses");
        assert_eq!(r.disjoint_roots, 0b0011);
        assert_eq!(r.revisable_assumptions, 0b0100, "inherited, not grounded");
        assert!(matches!(r.outcome, FusionOutcome::Synthesis(_)));
        assert_eq!(r.evidential_effect, EvidentialEffect::IncreaseEligible);
    }

    /// **A fake intelligence always synthesises.** Both horizons independently
    /// grounded, genuinely conflicting, and every assumption is ALSO
    /// independently grounded — so nothing may be withdrawn without discarding
    /// evidence. The honest answer is that they still disagree.
    #[test]
    fn irreducible_tension_is_reachable_when_no_assumption_is_revisable() {
        // inherited ⊆ independent ⇒ revisable_assumptions is empty.
        let t = horizon(1, 0b0001, 0b0101, 0b0100);
        let a = horizon(2, 0b0010, 0b1010, 0b1000);
        let r = fuse(&t, &a, &0b0001);

        assert!(
            r.revisable_assumptions.is_empty(),
            "anti-vacuity: the discriminating quantity really is absent"
        );
        assert_eq!(r.outcome, FusionOutcome::IrreducibleTension);
        assert_eq!(r.evidential_effect, EvidentialEffect::Suspend);
        assert!(
            !r.surviving_tension.is_empty(),
            "the disagreement is carried, not dissolved"
        );
    }

    /// No contradiction ⇒ the horizons were answering different questions.
    #[test]
    fn absent_contradiction_is_complementary_not_synthesis() {
        let t = horizon(1, 0b0001, 0b0001, 0b0100);
        let a = horizon(2, 0b0010, 0b0010, 0b1000);
        let r = fuse(&t, &a, &0);
        assert_eq!(r.outcome, FusionOutcome::Complementary);
        assert_eq!(r.evidential_effect, EvidentialEffect::NoIncrease);
    }

    /// An ungrounded antithesis does not survive contact with a grounded one.
    #[test]
    fn an_ungrounded_side_does_not_win() {
        let t = horizon(1, 0b0001, 0b0011, 0);
        let a = horizon(2, 0b0010, 0b0001, 0); // roots ⊂ thesis's
        let r = fuse(&t, &a, &0b1000);
        assert_eq!(r.outcome, FusionOutcome::ThesisSurvives);
        assert_eq!(r.evidential_effect, EvidentialEffect::NoIncrease);
    }

    /// The receipt must keep BOTH originals recoverable — otherwise revision
    /// is history rewriting.
    #[test]
    fn the_receipt_preserves_both_originals_and_carries_no_confidence_scalar() {
        let t = horizon(7, 0b0001, 0b0001, 0b0100);
        let a = horizon(9, 0b0010, 0b0010, 0b0100);
        let r = fuse(&t, &a, &0b1000);

        assert_eq!(r.thesis, HorizonId(7));
        assert_eq!(r.antithesis, HorizonId(9));
        assert_eq!(r.thesis_claims, 0b0001, "thesis recoverable after fusion");
        assert_eq!(
            r.antithesis_claims, 0b0010,
            "antithesis recoverable after fusion"
        );
        // The synthesis did NOT overwrite either horizon's claims.
        if let FusionOutcome::Synthesis(s) = &r.outcome {
            assert_ne!(s.preserved, r.thesis_claims);
            assert_ne!(s.preserved, r.antithesis_claims);
        } else {
            panic!("expected a synthesis for this fixture");
        }
    }
}
