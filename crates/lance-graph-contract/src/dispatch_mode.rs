// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! `dispatch_mode` — the **pre-dispatch mode router** (E-DISORDER-GATE-1): the
//! rung-4 macro that CHOOSES how to think before the 34 fire. Cynefin reduced to
//! its honest mechanical core.
//!
//! # The defect it fixes
//!
//! `materialize::select_tactic` reads `free_energy >= 0.66 / >= 0.33 / else`.
//! **`NaN >= x` is always `false`**, so an *ungrounded* awareness state (a marker
//! a missing tenant could not ground — `recipe_substrate::SubstrateView::project`
//! emits `NaN`) falls into the `else` and is silently dispatched as the LOWEST-
//! surprise **"routine parallel work"** band. An undefined state is read as a calm
//! one — exactly backwards. This router runs FIRST and catches that groundedness
//! failure before the selector ever sees it.
//!
//! # Cynefin, the mechanical core (not the ceremony)
//!
//! Cynefin's own question is *"what is the relationship between cause and
//! effect?"* — which IS a read of the awareness state, no new vocabulary:
//!
//! | [`Domain`] | measured condition | cause↔effect | [`DispatchMode`] |
//! |---|---|---|---|
//! | `Confused` | any required marker NaN / no candidates | UNKNOWABLE (ungrounded) | `FieldGather` — ground fields cheaply first |
//! | `Chaotic` | gate BLOCK (`sd > SD_BLOCK`) + high surprise | NONE | `Stabilize` — prune + commit (Gate bucket) |
//! | `Complex` | a contradiction is bound (`dissonance > 0`) | only in retrospect | `Sweep` — the systematic ladder, contradiction PRESERVED |
//! | `Clear` | gate FLOW + low surprise + no contradiction | obvious | `Saccade` — one tactic (`select_tactic`) |
//! | `Complicated` | everything else | knowable by analysis | `Sweep` — the ladder |
//!
//! **The MUL tie-in (operator: the meta-uncertainty cluster IS MUL):** a
//! `DkPosition::MountStupid` (felt≫demonstrated competence — the Dunning-Kruger
//! novice) VETOES the `Clear` election. You feel it's obvious; you are not
//! competent to saccade — so analyze. This is circle-of-competence composing in
//! at the mode level, reading MUL's already-computed position, not a new layer.
//!
//! # What this is / is not
//!
//! It sequences the TWO shipped dispatch modes (`select_tactic` saccade /
//! `recipe_dispatch::ladder` sweep) plus two honest degenerate arms (field-gather
//! / stabilize). Zero new recipes (34-lock). The domain is computed per dispatch
//! and never stored (rung-by-view-election). It reads only LOGICAL markers +
//! optional MUL — never qualia (qualia is additive stakes, not logic).

use crate::mul::DkPosition;
use crate::recipe_kernels::{ThoughtCtx, SD_BLOCK, SD_FLOW};

/// The Cynefin domain a live awareness state falls into — its cause↔effect
/// relationship, read from measured observables (never stored).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Domain {
    /// Cause and effect are obvious — gate FLOW, low surprise, no contradiction.
    Clear,
    /// Cause and effect are knowable by analysis — the default working domain.
    Complicated,
    /// Cause and effect only in retrospect — a contradiction is bound.
    Complex,
    /// No cause and effect — gate BLOCK with high surprise.
    Chaotic,
    /// Ungrounded — a required marker is NaN / undefined (the defect zone).
    Confused,
}

/// The dispatch mode the domain elects. `Saccade`/`Sweep` are the two shipped
/// modes (`select_tactic` / `recipe_dispatch::ladder`); `FieldGather`/`Stabilize`
/// are the two honest degenerate arms for ungrounded / chaotic state.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DispatchMode {
    /// One tactic per surprise cycle — the foveated `select_tactic` (System-1).
    Saccade,
    /// The full rung-ordered NaN-gated ladder — `recipe_dispatch::ladder` (System-2).
    Sweep,
    /// Ground the fields first: fire only cheap CrossTier "infrastructure"
    /// recipes that populate markers, THEN re-classify. Never dispatch content
    /// reasoning on an undefined state.
    FieldGather,
    /// Prune + commit: Gate-bucket recipes only (the measured winners in BLOCK).
    Stabilize,
}

/// Is any marker a recipe could require NaN / undefined? (the groundedness census
/// — the exact fields `materialize::select_tactic` reads blind).
#[inline]
#[must_use]
pub fn is_ungrounded(ctx: &ThoughtCtx) -> bool {
    ctx.free_energy.is_nan()
        || ctx.sd.is_nan()
        || ctx.dissonance.is_nan()
        || ctx.confidence.is_nan()
        || ctx.candidates.is_empty()
}

/// Classify the awareness state into its Cynefin [`Domain`], optionally vetoing
/// the `Clear` election on a Dunning-Kruger novice (`MountStupid`). Reads LOGICAL
/// markers + optional MUL only — never qualia.
#[must_use]
pub fn classify(ctx: &ThoughtCtx, dk: Option<DkPosition>) -> Domain {
    // Confused FIRST — an ungrounded state is not routine, it is undefined.
    if is_ungrounded(ctx) {
        return Domain::Confused;
    }
    // Chaotic — no cause/effect: the gate is BLOCK and surprise is high.
    if ctx.sd > SD_BLOCK && ctx.free_energy >= 0.66 {
        return Domain::Chaotic;
    }
    // Complex — a bound contradiction: cause/effect only in retrospect.
    if ctx.dissonance > 0.0 {
        return Domain::Complex;
    }
    // Clear — obvious: gate FLOW, low surprise, no contradiction. VETOED for the
    // overconfident novice (circle-of-competence at the mode level).
    let clear = ctx.sd < SD_FLOW && ctx.free_energy < 0.33 && ctx.dissonance == 0.0;
    if clear && dk != Some(DkPosition::MountStupid) {
        return Domain::Clear;
    }
    // Everything else (incl. a MountStupid-vetoed Clear) is knowable by analysis.
    Domain::Complicated
}

/// Elect the dispatch mode for a domain.
#[inline]
#[must_use]
pub fn elect_mode(domain: Domain) -> DispatchMode {
    match domain {
        Domain::Clear => DispatchMode::Saccade,
        Domain::Complicated | Domain::Complex => DispatchMode::Sweep,
        Domain::Chaotic => DispatchMode::Stabilize,
        Domain::Confused => DispatchMode::FieldGather,
    }
}

/// One routing decision, recording its own triggering cause (Versuchsleitereffekt
/// discipline — the router's causal fingerprint on what was thought is logged).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Route {
    /// The classified domain.
    pub domain: Domain,
    /// The elected dispatch mode.
    pub mode: DispatchMode,
    /// Why: `true` if the `Clear` election was vetoed by a MountStupid DK novice.
    pub dk_vetoed_clear: bool,
    /// Why: `true` if the state was ungrounded (the defect the router guards).
    pub was_ungrounded: bool,
}

/// The full pre-dispatch route: classify → elect → record the trigger.
#[must_use]
pub fn route(ctx: &ThoughtCtx, dk: Option<DkPosition>) -> Route {
    let was_ungrounded = is_ungrounded(ctx);
    let domain = classify(ctx, dk);
    // A Clear that would have fired but for the DK veto lands in Complicated.
    let clear_but_for_dk =
        !was_ungrounded && ctx.sd < SD_FLOW && ctx.free_energy < 0.33 && ctx.dissonance == 0.0;
    Route {
        domain,
        mode: elect_mode(domain),
        dk_vetoed_clear: clear_but_for_dk && dk == Some(DkPosition::MountStupid),
        was_ungrounded,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn grounded(free_energy: f32, sd: f32, dissonance: f32) -> ThoughtCtx {
        let mut c = ThoughtCtx::new(vec![0.5, 0.4]);
        c.free_energy = free_energy;
        c.sd = sd;
        c.dissonance = dissonance;
        c.confidence = 0.6;
        c
    }

    #[test]
    fn ungrounded_nan_is_confused_not_routine() {
        // THE DEFECT: a NaN free_energy would read as the routine band in
        // select_tactic; the router catches it as Confused → FieldGather.
        let mut c = grounded(f32::NAN, 0.25, 0.0);
        assert_eq!(classify(&c, None), Domain::Confused);
        assert_eq!(elect_mode(Domain::Confused), DispatchMode::FieldGather);
        // empty candidates is also ungrounded
        c.free_energy = 0.5;
        c.candidates.clear();
        assert_eq!(classify(&c, None), Domain::Confused);
    }

    #[test]
    fn clear_state_saccades() {
        let c = grounded(0.1, 0.05, 0.0); // FLOW, low surprise, no contradiction
        assert_eq!(classify(&c, None), Domain::Clear);
        assert_eq!(elect_mode(Domain::Clear), DispatchMode::Saccade);
    }

    #[test]
    fn mount_stupid_vetoes_clear_into_complicated() {
        let c = grounded(0.1, 0.05, 0.0); // would be Clear
        assert_eq!(
            classify(&c, Some(DkPosition::MountStupid)),
            Domain::Complicated
        );
        // an expert on the same state still saccades
        assert_eq!(classify(&c, Some(DkPosition::Plateau)), Domain::Clear);
        let r = route(&c, Some(DkPosition::MountStupid));
        assert!(r.dk_vetoed_clear && r.mode == DispatchMode::Sweep);
    }

    #[test]
    fn contradiction_is_complex_sweep() {
        let c = grounded(0.2, 0.25, 0.4); // dissonance bound
        assert_eq!(classify(&c, None), Domain::Complex);
        assert_eq!(elect_mode(Domain::Complex), DispatchMode::Sweep);
    }

    #[test]
    fn block_high_surprise_is_chaotic_stabilize() {
        let c = grounded(0.8, SD_BLOCK + 0.1, 0.0); // BLOCK gate + high surprise
        assert_eq!(classify(&c, None), Domain::Chaotic);
        assert_eq!(elect_mode(Domain::Chaotic), DispatchMode::Stabilize);
    }

    #[test]
    fn classification_is_causal_perturbing_across_a_boundary_flips_the_mode() {
        // low surprise → Clear/Saccade; push surprise across the band → not Clear.
        let calm = grounded(0.1, 0.05, 0.0);
        let surprised = grounded(0.8, 0.05, 0.0);
        assert_ne!(classify(&calm, None), classify(&surprised, None));
        assert_ne!(route(&calm, None).mode, route(&surprised, None).mode);
    }
}
