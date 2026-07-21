// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! `dispatch_guard` — a recipe is grounded when its required loci are BOUND
//! (single-pass structural) **and** CAUSAL under the multipass Markov standing
//! wave (temporal persistence). Two real resolutions of the same `temporal.rs`
//! stream — not a coarse scalar shadow.
//!
//! # Why the standing wave, not a coarse scalar pre-filter (operator ruling)
//!
//! The text is read ~90% as the `temporal.rs` sorted Markov stream
//! (`E-MARKOV-TEMPORAL-STREAM-1`), and the workspace LITERALLY resolves the Markov
//! chain — the D-CSW-1 standing wave (per-rung persistence separates causal from
//! coincidental). Gating "is this recipe grounded?" on a coarse scalar proxy
//! (`recipe_dispatch::nan_disqualifier` over aggregate `ThoughtCtx` markers) when
//! the literal Markov resolver is right there is a degenerate approximation — and
//! it measured as exactly that: a *tautological subset* of the organ gate (0
//! independent catches). So the scalar path is dropped as a grounding gate.
//!
//! The honest redundant pair is two DIFFERENT resolutions of the loci organ:
//!
//! * **Single-pass structural binding** — [`loci_disqualifier`](crate::recipe_loci::loci_disqualifier):
//!   is every required locus BOUND (nonzero, placed in the `±8` window)?
//! * **Multipass Markov standing wave** — [`standing_wave_grounded`](crate::witness_fabric::standing_wave_grounded):
//!   does every required locus PERSIST across the standing wave (settle to a
//!   stable target as the hop budget grows) rather than being a coincidental
//!   co-occurrence?
//!
//! These are genuinely independent: a locus can be BOUND (passes the single-pass
//! gate) yet COINCIDENTAL (fails the wave) — a causally-hollow binding. The wave
//! catches those; the single-pass gate can't. That is real higher-confidence
//! redundancy (causal vs merely-present), the OPPOSITE of a coarse pre-filter.
//!
//! # The two gates are the two orthogonal axes of a self-solving tissue (Sudoku)
//!
//! Per the operator's framing (`E-SUDOKU-TISSUE-WEAVE-1`), grounding is a **Sudoku
//! puzzle solving itself** — a constraint-satisfaction fixpoint on TWO orthogonal
//! axes, not two redundant measures of one thing:
//!
//! * **Vertical / bottom-up / Maslow** — single-pass BINDING in rung-ascending
//!   order: a higher-rung locus can only be grounded once the lower-layer tissue
//!   it builds on is woven (`recipe_loci::carried_awareness`). The connective
//!   tissue must exist first.
//! * **Horizontal** — the multipass Markov STANDING WAVE: the multihop causality
//!   chain (`resolve_chain`, signed offsets → forward AND backward propagation)
//!   settling across hop budgets. Deeper budget = deeper thinking; settlement is
//!   the confidence.
//!
//! Their INTERSECTION is the unique grounding — which is *why* the composition is
//! higher-confidence: orthogonal constraints intersect to a solution, they do not
//! merely agree. The 34/34 `Fires`→`WaveCatch` flip measured in
//! `examples/dispatch_guard_redundancy` is a Sudoku row constraint being BLIND to
//! what a column constraint sees (single-pass binding cannot see horizontal
//! persistence). The 24-loci tissue is the grid; the 34 recipes weave it layer by
//! layer, rung by rung.
//!
//! The `nan_disqualifier` scalar check is retained ONLY as an optional
//! ctx-source-independent sanity flag ([`GuardVerdict::scalar_flag`]) for the case
//! a ctx arrives from a non-witness source; it never gates grounding here.

use crate::causal_witness::{CausalWitnessFacet, Locus};
use crate::recipe_dispatch::nan_disqualifier;
use crate::recipe_kernels::ThoughtCtx;
use crate::recipe_loci::{loci_disqualifier, required_loci};
use crate::witness_fabric::{standing_wave_grounded, WaveGrounding};

/// The relationship between the two independent grounding gates for one recipe.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GateOutcome {
    /// Every required locus is BOUND and CAUSAL — the recipe fires.
    Fires,
    /// A required locus is bound but COINCIDENTAL — the standing wave caught a
    /// causally-hollow binding the single-pass gate would have fired. The
    /// redundancy value (the wave's independent catch).
    WaveCatch,
    /// A required locus is UNBOUND — both gates block (a causal locus is bound by
    /// definition, so unbound fails the wave too). Agreement.
    Unbound,
}

impl GateOutcome {
    /// Does this cell exercise the wave's INDEPENDENT catch (bound-but-coincidental)?
    #[inline]
    #[must_use]
    pub const fn is_wave_catch(self) -> bool {
        matches!(self, GateOutcome::WaveCatch)
    }
}

/// The combined grounding verdict for one recipe against a window.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GuardVerdict {
    /// First required locus that is UNBOUND (single-pass structural gate), if any.
    pub unbound: Option<Locus>,
    /// First required locus that is bound but COINCIDENTAL under the standing
    /// wave, if any (the wave's independent catch).
    pub coincidental: Option<Locus>,
    /// Optional coarse scalar sanity flag — the field `nan_disqualifier` reports
    /// on the projected ctx. NOT a grounding gate here (documented degenerate);
    /// meaningful only when the ctx has a non-witness source.
    pub scalar_flag: Option<crate::recipe_kernels::ThoughtField>,
    /// The gate relationship.
    pub outcome: GateOutcome,
}

impl GuardVerdict {
    /// Both organ resolutions passed → the recipe is safe to fire.
    #[inline]
    #[must_use]
    pub const fn fires(&self) -> bool {
        matches!(self.outcome, GateOutcome::Fires)
    }
}

/// Ground recipe `id` against the row's window: single-pass BINDING ∧ multipass
/// Markov STANDING WAVE. `window` is `(stream_position, register)`; `focal_idx`
/// the focal row; `passes` the standing-wave budget. `ctx` (optional) supplies
/// only the degenerate scalar sanity flag.
#[must_use]
pub fn guard(
    ctx: Option<&ThoughtCtx>,
    window: &[(usize, CausalWitnessFacet)],
    focal_idx: usize,
    id: u8,
    passes: u8,
) -> GuardVerdict {
    let witness = window
        .get(focal_idx)
        .map(|&(_, w)| w)
        .unwrap_or(CausalWitnessFacet::ZERO);

    // Gate 1 — single-pass structural binding.
    let unbound = loci_disqualifier(&witness, id);

    // Gate 2 — multipass Markov standing wave: a required locus that is bound but
    // does not persist (coincidental) is caught here, not by gate 1.
    let coincidental = if unbound.is_some() {
        None // don't double-report; the binding gate already blocked
    } else {
        required_loci(id).iter().copied().find(|&l| {
            standing_wave_grounded(focal_idx, window, l, passes) == WaveGrounding::Coincidental
        })
    };

    let outcome = match (unbound.is_some(), coincidental.is_some()) {
        (true, _) => GateOutcome::Unbound,
        (false, true) => GateOutcome::WaveCatch,
        (false, false) => GateOutcome::Fires,
    };

    GuardVerdict {
        unbound,
        coincidental,
        scalar_flag: ctx.and_then(|c| nan_disqualifier(c, id)),
        outcome,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn wit(edges: &[(Locus, i8)]) -> CausalWitnessFacet {
        let mut w = CausalWitnessFacet::ZERO;
        for &(l, o) in edges {
            w = w.with(l, o);
        }
        w
    }

    #[test]
    fn unbound_locus_blocks_both_gates() {
        // #20 TCF needs Quorum; a window whose focal never binds it.
        let focal = wit(&[(Locus::Temporal, -1)]);
        let window = [(0usize, focal)];
        let v = guard(None, &window, 0, 20, 4);
        assert_eq!(v.outcome, GateOutcome::Unbound);
        assert_eq!(v.unbound, Some(Locus::Quorum));
        assert!(!v.fires());
    }

    #[test]
    fn bound_and_settled_chain_fires() {
        // #20 TCF needs Quorum. Bind it to a terminal (peer does NOT rebind) →
        // the standing wave settles immediately → Causal → fires.
        let focal = wit(&[(Locus::Quorum, 1)]);
        let peer = wit(&[(Locus::Temporal, 0)]); // no Quorum rebind → terminal
        let window = [(0usize, focal), (1, peer)];
        let v = guard(None, &window, 0, 20, 4);
        assert_eq!(v.outcome, GateOutcome::Fires);
        assert!(v.fires());
    }

    #[test]
    fn bound_but_coincidental_is_the_waves_independent_catch() {
        // #20 TCF needs Quorum, BOUND — but its chain keeps extending out of the
        // ±8 window (never settles) → the standing wave marks it Coincidental,
        // which the single-pass binding gate (bound == grounded) would have missed.
        let a = wit(&[(Locus::Quorum, 7)]);
        let b = wit(&[(Locus::Quorum, 7)]); // rebinds → chain leaves ±8 → escalates
        let window = [(0usize, a), (7, b)];
        let v = guard(None, &window, 0, 20, 8);
        assert_eq!(
            v.outcome,
            GateOutcome::WaveCatch,
            "wave catches the causally-hollow binding"
        );
        assert!(v.unbound.is_none() && v.coincidental == Some(Locus::Quorum));
        assert!(!v.fires(), "bound-but-coincidental does NOT fire");
    }

    #[test]
    fn wave_catch_is_independent_of_the_single_pass_gate() {
        // The single-pass gate (loci_disqualifier) sees Quorum BOUND → would fire.
        // The wave gate blocks it. That divergence is the higher-confidence
        // redundancy — not a coarse subset.
        let a = wit(&[(Locus::Quorum, 7)]);
        let b = wit(&[(Locus::Quorum, 7)]);
        let window = [(0usize, a), (7, b)];
        assert!(
            loci_disqualifier(&a, 20).is_none(),
            "single-pass: bound → grounded"
        );
        assert_eq!(
            guard(None, &window, 0, 20, 8).outcome,
            GateOutcome::WaveCatch
        );
    }
}
