// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! `recipe_dispatch` — the 34 recipes as a **rung-ordered, NaN-gated causal
//! ladder**, keyed by NARS inference type.
//!
//! # Thinking is an act of cause and effect (and a finder of it)
//!
//! Orchestrating the recipes is *itself* causal: firing a recipe is a **cause**
//! whose **effect** is the next [`ThoughtCtx`](crate::recipe_kernels::ThoughtCtx)
//! state, which triggers the next recipe. The rung order IS a causal chain. The
//! recipes also *find* cause and effect in the data (RCR/ICR read the witness
//! [`Kausal`](crate::causal_witness::Locus::Kausal) edge). And — the
//! **Versuchsleitereffekt** (experimenter effect) — the CHOICE of which recipe
//! fires at which rung biases what is found, so each dispatch records its own
//! **triggering cause** ([`RecipeStep::trigger`]): the grounded awareness that
//! licensed it. The orchestration's causal influence is logged, not hidden.
//!
//! # Deterministic collapse — the glass-box cat
//!
//! Before dispatch the awareness is in **superposition**: the witness register
//! holds many bundled loci; many recipes could fire. Dispatching a recipe (or
//! reading a locus by its role key) is a **measurement** that collapses it — but
//! *deterministically*: identity is recoverable by key, no measurement
//! randomness (Schrödinger's cat in a glass box). The [`ladder`] is the ordered
//! collapse. The **Versuchsleitereffekt** above IS the Heisenberg observer
//! effect: measuring disturbs, so the trigger is logged. And a [`nan_disqualifier`]
//! is a **conjugate variable not measurable in this basis** — an input the
//! current tenants cannot ground, so that recipe is skipped rather than read off
//! noise. The `N ≤ √d/4 ≈ 32` bundle bound (`I-VSA-IDENTITIES` Test 1) is the
//! substrate's uncertainty relation; the 24-locus witness sits safely under it.
//!
//! # What each recipe knows (operator ruling)
//!
//! 1. **inference type** — [`RecipeInference`] (deduction / induction=fanout /
//!    abduction / revision / counterfactual), the NARS inference it embodies.
//! 2. **rung level + order** — [`rung`] + [`dispatch_order`]: shallow forward
//!    deduction fires first, deep counterfactual/revision last, from the recipe
//!    [`Tier`](crate::recipes::Tier) plus the shipped
//!    [`InferenceType::rung_delta`](crate::nars::InferenceType::rung_delta).
//! 3. **NaN disqualifiers** — [`nan_disqualifier`]: a required input
//!    ([`Tactic::requires`](crate::recipe_kernels::Tactic)) that is NaN /
//!    undefined in the projected ctx DISQUALIFIES the recipe. This is
//!    `E-RELIABILITY-IS-CHECKLIST-COVERAGE` made a runtime gate — a recipe never
//!    fires on an input the tenants could not ground.

use crate::nars::InferenceType;
use crate::recipe_kernels::{kernel, ThoughtCtx, ThoughtField, ThoughtMask};
use crate::recipes::{recipe, Tier};

/// The inference a recipe embodies. Maps to [`InferenceType`] for the four NARS
/// primitives; `Counterfactual` is the causal-edge mantissa-`−6` extension
/// (`I-VSA-IDENTITIES` counterfactual world), which [`InferenceType`] represents
/// via mantissa, not a variant.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RecipeInference {
    /// Forward, exact (NARS Deduction).
    Deduction,
    /// Wide generate / fan-out (NARS Induction).
    Induction,
    /// Backward, effect→cause (NARS Abduction).
    Abduction,
    /// Merge / consensus / belief update (NARS Revision).
    Revision,
    /// Counterfactual world construction (causal-edge mantissa −6).
    Counterfactual,
}

impl RecipeInference {
    /// The shipped [`InferenceType`] this maps to (Counterfactual → Abduction's
    /// backward direction, its nearest NARS primitive; the causal-edge `−6`
    /// mantissa carries the counterfactual distinction downstream). The bridge
    /// is by VALUE — [`InferenceType::to_mantissa`] is the shared little-endian
    /// rule key every `CausalEdge64` consumer reads.
    #[must_use]
    pub fn nars(self) -> InferenceType {
        match self {
            RecipeInference::Deduction => InferenceType::Deduction,
            RecipeInference::Induction => InferenceType::Induction,
            RecipeInference::Abduction | RecipeInference::Counterfactual => {
                InferenceType::Abduction
            }
            RecipeInference::Revision => InferenceType::Revision,
        }
    }
    /// Escalation depth offset (the ORDER cost, not the mantissa *direction*):
    /// forward-exact deduction is cheapest, backward counterfactual deepest.
    /// Ded +1 → Ind +2 → Rev +3 → Abd +4 → Counterfactual +5.
    #[must_use]
    pub fn rung_delta(self) -> i16 {
        match self {
            RecipeInference::Deduction => 1,
            RecipeInference::Induction => 2,
            RecipeInference::Revision => 3,
            RecipeInference::Abduction => 4,
            RecipeInference::Counterfactual => 5,
        }
    }
    /// Short tag.
    #[must_use]
    pub fn tag(self) -> &'static str {
        match self {
            RecipeInference::Deduction => "deduction",
            RecipeInference::Induction => "induction/fanout",
            RecipeInference::Abduction => "abduction",
            RecipeInference::Revision => "revision",
            RecipeInference::Counterfactual => "counterfactual",
        }
    }
}

/// The inference each of the 34 recipes embodies (documented classification from
/// the recipe's characteristic operation — `recipes.rs` `substrate`/`mechanism`).
#[must_use]
pub fn inference(id: u8) -> RecipeInference {
    use RecipeInference::*;
    match id {
        1 => Induction,       // RTE recursive expansion (fan-out depth)
        2 => Induction,       // HTD hierarchical decompose (fan-out)
        3 => Revision,        // SMAD debate → consensus revise
        4 => Abduction,       // RCR reverse causality (backward S_O) — explicit
        5 => Deduction,       // TCP prune (forward select)
        6 => Induction,       // TR randomize (divergent generate)
        7 => Revision,        // ASC adversarial self-critique (belief revision)
        8 => Deduction,       // CAS abstraction scaling (level select)
        9 => Induction,       // IRS roleplay synthesis (generate variants)
        10 => Revision,       // MCP meta-cognition (calibration update)
        11 => Revision,       // CR contradiction resolution (revise on conflict)
        12 => Deduction,      // TCA temporal precedence (Granger forward)
        13 => Induction,      // CDT convergent/divergent (divergent generate)
        14 => Deduction,      // MCT multimodal fuse
        15 => Induction,      // LSI latent introspection (distribution read)
        16 => Deduction,      // PSO scaffold (order)
        17 => Revision,       // CDI dissonance induction (revise)
        18 => Revision,       // CWS context persistence (memory update)
        19 => Abduction,      // ARE reverse-engineer (recover component)
        20 => Revision,       // TCF cascade filter (agreement consensus)
        21 => Revision,       // SSR self-skepticism (down-weight)
        22 => Induction,      // ETD emergent decompose (fan-out)
        23 => Induction,      // AMP adaptive meta (explore/adapt)
        24 => Deduction,      // ZCF zero-shot fuse (compose)
        25 => Deduction,      // HPM pattern match (nearest)
        26 => Deduction,      // CUR uncertainty reduction (coarse→fine select)
        27 => Revision,       // MPC multi-perspective compress (majority)
        28 => Abduction,      // SSAM analogical mapping (analogy → hypothesis)
        29 => Deduction,      // IDR intent reframe (select dominant)
        30 => Revision,       // SPP shadow parallel (agreement verify)
        31 => Counterfactual, // ICR iterative counterfactual — explicit
        32 => Deduction,      // SDD distortion detect (validate)
        33 => Revision,       // DTMF meta-frame switch
        34 => Deduction,      // HKF cross-domain fuse (compose)
        _ => Deduction,
    }
}

/// The rung level (`1..=9`) a recipe fires at: a base from its difficulty
/// [`Tier`] plus its inference [`rung_delta`](RecipeInference::rung_delta).
/// Shallow forward deduction → low rung; deep counterfactual/revision → high.
#[must_use]
pub fn rung(id: u8) -> u8 {
    let tier = recipe(id).map(|r| r.tier).unwrap_or(Tier::CrossTier);
    let base: i16 = match tier {
        Tier::CrossTier => 1,     // infrastructure — fires everywhere, early
        Tier::Hard => 3,          // the ~65% plateau
        Tier::ExtremelyHard => 5, // convergent lock-in
    };
    (base + inference(id).rung_delta()).clamp(1, 9) as u8
}

/// The 34 recipe ids in **dispatch order** — ascending rung, then ascending id.
/// This IS the escalation ladder: the causal chain fires shallow → deep.
#[must_use]
pub fn dispatch_order() -> [u8; 34] {
    let mut ids: [u8; 34] = core::array::from_fn(|i| i as u8 + 1);
    ids.sort_by_key(|&id| (rung(id), id));
    ids
}

/// Map a [`ThoughtField`] to whether its ctx marker is **NaN / undefined**.
/// f32 markers use `is_nan`; the two collection markers use `is_empty` (an empty
/// input is as undefined as a NaN). `Rung` (a `u8`) is never NaN.
fn field_is_undefined(field: ThoughtField, ctx: &ThoughtCtx) -> bool {
    match field {
        ThoughtField::Sd => ctx.sd.is_nan(),
        ThoughtField::FreeEnergy => ctx.free_energy.is_nan(),
        ThoughtField::Dissonance => ctx.dissonance.is_nan(),
        ThoughtField::Temperature => ctx.temperature.is_nan(),
        ThoughtField::Confidence => ctx.confidence.is_nan(),
        ThoughtField::Rung => false,
        ThoughtField::Candidates => ctx.candidates.is_empty(),
        ThoughtField::Beliefs => ctx.beliefs.is_empty(),
    }
}

/// **The NaN disqualifier gate.** Returns the first required input that is NaN /
/// undefined in `ctx` (so the recipe is DISQUALIFIED from firing), or `None` if
/// the recipe's whole checklist is grounded. The required set is the kernel's
/// own [`requires`](crate::recipe_kernels::Tactic::requires) mask — reliability
/// as coverage, enforced at dispatch.
#[must_use]
pub fn nan_disqualifier(ctx: &ThoughtCtx, id: u8) -> Option<ThoughtField> {
    let req: ThoughtMask = kernel(id)?.requires();
    const FIELDS: [ThoughtField; 8] = [
        ThoughtField::Sd,
        ThoughtField::FreeEnergy,
        ThoughtField::Dissonance,
        ThoughtField::Temperature,
        ThoughtField::Confidence,
        ThoughtField::Rung,
        ThoughtField::Candidates,
        ThoughtField::Beliefs,
    ];
    FIELDS
        .into_iter()
        .find(|&f| req.has(f) && field_is_undefined(f, ctx))
}

/// One rung-ordered dispatch decision, recording its **triggering cause** (the
/// grounded required checklist that licensed it) so the orchestration's own
/// causal influence is auditable (Versuchsleitereffekt).
#[derive(Debug, Clone, Copy)]
pub struct RecipeStep {
    /// Recipe id (1..=34).
    pub id: u8,
    /// Rung this recipe fires at.
    pub rung: u8,
    /// Inference the recipe embodies.
    pub inference: RecipeInference,
    /// `None` = fired; `Some(field)` = disqualified because that input was NaN.
    pub disqualified_by: Option<ThoughtField>,
    /// The grounded required-field count that licensed firing (the cause the
    /// orchestration records for itself).
    pub trigger: u32,
}

/// Walk all 34 recipes in [`dispatch_order`], gating each on
/// [`nan_disqualifier`]. The returned steps are the **causal chain of the
/// orchestration** — each records whether it fired, at what rung, on which
/// inference, and its triggering-cause coverage.
#[must_use]
pub fn ladder(ctx: &ThoughtCtx) -> Vec<RecipeStep> {
    dispatch_order()
        .into_iter()
        .map(|id| {
            let disq = nan_disqualifier(ctx, id);
            let trigger = kernel(id).map(|k| k.requires().len()).unwrap_or(0);
            RecipeStep {
                id,
                rung: rung(id),
                inference: inference(id),
                disqualified_by: disq,
                trigger,
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::recipe_kernels::ThoughtCtx;

    #[test]
    fn every_recipe_has_an_inference_and_a_rung_in_range() {
        for id in 1..=34u8 {
            let r = rung(id);
            assert!((1..=9).contains(&r), "recipe {id} rung {r} out of 1..=9");
            let _ = inference(id); // total
        }
    }

    #[test]
    fn dispatch_order_is_rung_ascending() {
        let order = dispatch_order();
        assert_eq!(order.len(), 34);
        // a permutation of 1..=34
        let mut seen = order;
        seen.sort_unstable();
        assert_eq!(seen, core::array::from_fn::<u8, 34, _>(|i| i as u8 + 1));
        // non-decreasing rung
        let rungs: Vec<u8> = order.iter().map(|&id| rung(id)).collect();
        assert!(
            rungs.windows(2).all(|w| w[0] <= w[1]),
            "rungs ascend: {rungs:?}"
        );
    }

    #[test]
    fn counterfactual_fires_deepest() {
        // ICR (31) is the only Counterfactual; it should sit at a high rung.
        assert_eq!(inference(31), RecipeInference::Counterfactual);
        assert!(rung(31) >= 9, "counterfactual is the deepest rung");
        // RCR (4) abduction is explicit backward causality.
        assert_eq!(inference(4), RecipeInference::Abduction);
    }

    #[test]
    fn nan_input_disqualifies_the_requiring_recipe() {
        // Cr(11) requires Beliefs. An empty belief set → disqualified.
        let mut c = ThoughtCtx::new(vec![0.5]);
        c.beliefs.clear();
        assert_eq!(nan_disqualifier(&c, 11), Some(ThoughtField::Beliefs));
        // give it a belief → no longer disqualified on that field
        c.beliefs.push((7, 0.9, 0.8));
        assert_eq!(nan_disqualifier(&c, 11), None);
    }

    #[test]
    fn nan_free_energy_disqualifies_rte() {
        // Rte(1) requires FreeEnergy + Rung. NaN free_energy → disqualified.
        let mut c = ThoughtCtx::new(vec![0.5]);
        c.free_energy = f32::NAN;
        assert_eq!(nan_disqualifier(&c, 1), Some(ThoughtField::FreeEnergy));
    }

    #[test]
    fn ladder_records_the_causal_chain() {
        // A fully grounded ctx: everything fires.
        let mut c = ThoughtCtx::new(vec![0.9, 0.5, 0.1]);
        c.beliefs = vec![(7, 0.9, 0.8)];
        let steps = ladder(&c);
        assert_eq!(steps.len(), 34);
        assert!(
            steps.iter().all(|s| s.disqualified_by.is_none()),
            "all grounded → all fire"
        );
        // rung-ordered
        assert!(steps.windows(2).all(|w| w[0].rung <= w[1].rung));
        // an ungrounded ctx: candidate + belief recipes disqualified.
        let empty = ThoughtCtx {
            sd: f32::NAN,
            free_energy: f32::NAN,
            dissonance: f32::NAN,
            temperature: 0.5,
            confidence: f32::NAN,
            rung: 1,
            candidates: Vec::new(),
            beliefs: Vec::new(),
        };
        let steps2 = ladder(&empty);
        assert!(
            steps2.iter().any(|s| s.disqualified_by.is_some()),
            "ungrounded inputs disqualify some recipes"
        );
    }
}
