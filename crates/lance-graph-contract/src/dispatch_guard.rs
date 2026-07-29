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
//!   does every required locus SETTLE within the `±8` reference horizon (stable
//!   target as the hop budget grows), or does its causal chain leave that horizon
//!   (a non-local cause) and need to [`Escalate`](GateOutcome::Escalate)?
//!
//! These are genuinely independent: a locus can be BOUND (passes the single-pass
//! gate) yet have a NON-LOCAL cause the wave sees leaving the reference horizon —
//! the wave escalates those; the single-pass gate can't see past the window. That
//! is real higher-confidence redundancy (local-causal vs beyond-the-horizon), the
//! OPPOSITE of a coarse pre-filter.
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
//!   the confidence. The `±8` window is only the **reference horizon** — a chain
//!   that leaves it is not coincidental, it [`Escalate`](GateOutcome::Escalate)s to
//!   a `temporal.rs` version-range read, and ultimately to the AriGraph
//!   `part_of:is_a` basins, which are ADDRESSED ABSOLUTELY (not temporal-window
//!   bounded) — `E-HORIZON-NOT-BOUND-1`.
//!
//! Their INTERSECTION is the unique grounding — which is *why* the composition is
//! higher-confidence: orthogonal constraints intersect to a solution, they do not
//! merely agree. The 34/34 `Fires`→`Escalate` flip measured in
//! `examples/dispatch_guard_redundancy` is a Sudoku row constraint being BLIND to
//! what a column constraint sees (single-pass binding cannot see that a chain's
//! causality lives beyond the reference horizon). The 24-loci tissue is the grid;
//! the 34 recipes weave it layer by layer, rung by rung.
//!
//! The `nan_disqualifier` scalar check is retained ONLY as an optional
//! ctx-source-independent sanity flag ([`GuardVerdict::scalar_flag`]) for the case
//! a ctx arrives from a non-witness source; it never gates grounding here.

use crate::causal_witness::{CausalWitnessFacet, Locus};
use crate::recipe_dispatch::nan_disqualifier;
use crate::recipe_kernels::ThoughtCtx;
use crate::recipe_loci::{loci_disqualifier, required_loci};
use crate::witness_fabric::{standing_wave_grounded_lens, WaveGrounding, WitnessLens};

/// The relationship between the two independent grounding gates for one recipe.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GateOutcome {
    /// Every required locus is BOUND and CAUSAL within the reference horizon — the
    /// recipe fires locally (cheap).
    Fires,
    /// A required locus is bound, but its causal chain LEAVES the `±8` reference
    /// horizon — the standing wave caught a NON-LOCAL cause the single-pass gate
    /// would have fired blind. Not coincidental (Romeo & Juliet's death is still
    /// caused by the distant feud): the recipe should ESCALATE to a `temporal.rs`
    /// read / the absolute AriGraph basin and search the causality over time. The
    /// redundancy value (the wave's independent, orthogonal catch).
    Escalate,
    /// A required locus is UNBOUND — both gates block (an unbound locus has no
    /// chain to stand a wave on). Agreement.
    Unbound,
}

impl GateOutcome {
    /// Does this cell exercise the wave's INDEPENDENT catch (bound-but-non-local →
    /// escalate over time / into the absolute basin)?
    #[inline]
    #[must_use]
    pub const fn is_escalate(self) -> bool {
        matches!(self, GateOutcome::Escalate)
    }
}

/// The combined grounding verdict for one recipe against a window.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GuardVerdict {
    /// First required locus that is UNBOUND (single-pass structural gate), if any.
    pub unbound: Option<Locus>,
    /// First required locus whose causal chain LEAVES the `±8` reference horizon
    /// under the standing wave, if any — the non-local cause to escalate for.
    pub escalate: Option<Locus>,
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
/// Markov STANDING WAVE.
///
/// **Zero-copy by construction** (operator ruling: "zero copy is a law without
/// escape hatches"). The focal register is CAST out of the row's own value
/// slab through [`WitnessLens`] — never gathered into a
/// `&[(usize, CausalWitnessFacet)]` slab first, which would store a second
/// projection of bytes the row array already holds. `focal_pos` is the
/// ABSOLUTE row position; `visible` stands in for "this position is part of
/// the window" exactly as it does in
/// [`resolve_chain_lens`](crate::witness_fabric::resolve_chain_lens), so
/// narrowing the window costs one predicate call per hop and never an
/// allocation. `passes` is the standing-wave budget; `ctx` (optional) supplies
/// only the degenerate scalar sanity flag.
///
/// A `focal_pos` the lens cannot address reads as the all-unbound register —
/// the same neutral the gathered form produced for a `focal_idx` past the end
/// of its window — so the verdict is [`Unbound`](GateOutcome::Unbound), never
/// a panic. Like the shipped lens resolvers, the FOCAL itself is read
/// unconditionally; `visible` gates the hop targets.
#[must_use]
pub fn guard(
    ctx: Option<&ThoughtCtx>,
    focal_pos: usize,
    lens: &WitnessLens<'_>,
    visible: impl Fn(usize) -> bool,
    id: u8,
    passes: u8,
) -> GuardVerdict {
    // A reference to a `const` item is 'static-promoted, so the absent-focal
    // fallback costs nothing and copies nothing.
    const ABSENT: CausalWitnessFacet = CausalWitnessFacet::ZERO;
    let witness: &CausalWitnessFacet = lens.at(focal_pos).unwrap_or(&ABSENT);

    // Gate 1 — single-pass structural binding.
    let unbound = loci_disqualifier(witness, id);

    // Gate 2 — multipass Markov standing wave: a required locus whose causal chain
    // leaves the ±8 reference horizon (a NON-LOCAL cause) is caught here, not by
    // gate 1. It is not coincidental — the recipe escalates to search over time.
    let escalate = if unbound.is_some() {
        None // don't double-report; the binding gate already blocked
    } else {
        required_loci(id).iter().copied().find(|&l| {
            standing_wave_grounded_lens(focal_pos, lens, &visible, l, passes)
                == WaveGrounding::Escalate
        })
    };

    let outcome = match (unbound.is_some(), escalate.is_some()) {
        (true, _) => GateOutcome::Unbound,
        (false, true) => GateOutcome::Escalate,
        (false, false) => GateOutcome::Fires,
    };

    GuardVerdict {
        unbound,
        escalate,
        scalar_flag: ctx.and_then(|c| nan_disqualifier(c, id)),
        outcome,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::canonical_node::{EdgeBlock, NodeGuid, NodeRow};
    use crate::witness_fabric::standing_wave_grounded;

    fn wit(edges: &[(Locus, i8)]) -> CausalWitnessFacet {
        let mut w = CausalWitnessFacet::ZERO;
        for &(l, o) in edges {
            w = w.with(l, o);
        }
        w
    }

    /// A row array indexed BY absolute stream position, each named position's
    /// CausalWitness register written through the lens's own producer (so the
    /// fixture cannot drift from the offsets the lens reads).
    fn rows_from(regs: &[(usize, CausalWitnessFacet)]) -> Vec<NodeRow> {
        let max_pos = regs.iter().map(|&(p, _)| p).max().unwrap_or(0);
        let mut rows: Vec<NodeRow> = (0..=max_pos)
            .map(|_| NodeRow {
                key: NodeGuid::local(1),
                edges: EdgeBlock::default(),
                value: [0u8; 480],
            })
            .collect();
        for &(pos, facet) in regs {
            WitnessLens::write_register(&mut rows[pos], &facet);
        }
        rows
    }

    /// **The pre-migration gathered body, verbatim** — kept ONLY as the test
    /// oracle for the equivalence argument below. It is `#[cfg(test)]`, so no
    /// production path can reach it; the materializing signature no longer
    /// exists in the crate's API.
    fn guard_gathered_oracle(
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
        let unbound = loci_disqualifier(&witness, id);
        let escalate = if unbound.is_some() {
            None
        } else {
            required_loci(id).iter().copied().find(|&l| {
                standing_wave_grounded(focal_idx, window, l, passes) == WaveGrounding::Escalate
            })
        };
        let outcome = match (unbound.is_some(), escalate.is_some()) {
            (true, _) => GateOutcome::Unbound,
            (false, true) => GateOutcome::Escalate,
            (false, false) => GateOutcome::Fires,
        };
        GuardVerdict {
            unbound,
            escalate,
            scalar_flag: ctx.and_then(|c| nan_disqualifier(c, id)),
            outcome,
        }
    }

    /// **The whole safety argument for the migration.** Across every recipe id
    /// (1..=34), several pass budgets, four real window shapes AND a
    /// peer-hidden variant of each, the lens `guard` must return EXACTLY the
    /// verdict the gathered body returned. Every field is compared, not just
    /// the outcome.
    #[test]
    fn lens_guard_matches_the_gathered_oracle_across_ids_passes_and_visibility() {
        let all = |off: i8| {
            let mut w = CausalWitnessFacet::ZERO;
            for &l in Locus::ALL.iter() {
                w = w.with(l, off);
            }
            w
        };
        let scenarios: [Vec<(usize, CausalWitnessFacet)>; 4] = [
            // settles inside ±8 (terminal peer)
            vec![(0usize, all(1)), (1, CausalWitnessFacet::ZERO)],
            // chain re-extends out of ±8 → escalates
            vec![(0usize, all(7)), (7, all(7))],
            // a focal that binds almost nothing → unbound-dominated
            vec![(0usize, wit(&[(Locus::Temporal, -1)]))],
            // multi-hop inside the horizon, with a gap
            vec![(0usize, all(2)), (2, all(2)), (4, CausalWitnessFacet::ZERO)],
        ];

        let (mut saw_fires, mut saw_escalate, mut saw_unbound) = (false, false, false);
        for scenario in &scenarios {
            let rows = rows_from(scenario);
            let lens = WitnessLens::new(&rows);
            // Full window, plus each one-peer-hidden subset.
            let mut hidden: Vec<Option<usize>> = vec![None];
            hidden.extend(scenario.iter().skip(1).map(|&(p, _)| Some(p)));
            for hide in hidden {
                let window: Vec<(usize, CausalWitnessFacet)> = scenario
                    .iter()
                    .copied()
                    .filter(|&(p, _)| Some(p) != hide)
                    .collect();
                let visible = |pos: usize| Some(pos) != hide;
                for id in 1..=34u8 {
                    for passes in [1u8, 2, 4, 8] {
                        let gathered = guard_gathered_oracle(None, &window, 0, id, passes);
                        let lensed = guard(None, 0, &lens, visible, id, passes);
                        assert_eq!(
                            gathered, lensed,
                            "guard diverged: id {id}, passes {passes}, hidden {hide:?}"
                        );
                        match lensed.outcome {
                            GateOutcome::Fires => saw_fires = true,
                            GateOutcome::Escalate => saw_escalate = true,
                            GateOutcome::Unbound => saw_unbound = true,
                        }
                    }
                }
            }
        }
        assert!(
            saw_fires && saw_escalate && saw_unbound,
            "the scenario set must exercise Fires, Escalate AND Unbound, or the \
             equivalence only covers one branch"
        );
    }

    /// A focal position the lens cannot address behaves like a `focal_idx`
    /// past the end of a gathered window: all-unbound, never a panic.
    #[test]
    fn out_of_range_focal_is_unbound_not_a_panic() {
        let rows = rows_from(&[(0usize, wit(&[(Locus::Quorum, 1)]))]);
        let lens = WitnessLens::new(&rows);
        let v = guard(None, 9_999, &lens, |_| true, 20, 4);
        assert_eq!(v.outcome, GateOutcome::Unbound);
        assert_eq!(
            v,
            guard_gathered_oracle(None, &[], 0, 20, 4),
            "must match the gathered form's empty-window verdict"
        );
    }

    #[test]
    fn unbound_locus_blocks_both_gates() {
        // #20 TCF needs Quorum; a row whose focal never binds it.
        let rows = rows_from(&[(0usize, wit(&[(Locus::Temporal, -1)]))]);
        let lens = WitnessLens::new(&rows);
        let v = guard(None, 0, &lens, |_| true, 20, 4);
        assert_eq!(v.outcome, GateOutcome::Unbound);
        assert_eq!(v.unbound, Some(Locus::Quorum));
        assert!(!v.fires());
    }

    #[test]
    fn bound_and_settled_chain_fires() {
        // #20 TCF needs Quorum. Bind it to a terminal (peer does NOT rebind) →
        // the standing wave settles immediately → Causal → fires.
        let rows = rows_from(&[
            (0usize, wit(&[(Locus::Quorum, 1)])),
            (1, wit(&[(Locus::Temporal, 0)])), // no Quorum rebind → terminal
        ]);
        let lens = WitnessLens::new(&rows);
        let v = guard(None, 0, &lens, |_| true, 20, 4);
        assert_eq!(v.outcome, GateOutcome::Fires);
        assert!(v.fires());
    }

    #[test]
    fn bound_but_non_local_cause_escalates_not_coincidental() {
        // #20 TCF needs Quorum, BOUND — but its chain keeps extending out of the
        // ±8 reference horizon → the standing wave marks it Escalate (a NON-LOCAL
        // cause, like Romeo & Juliet's death caused by the distant feud), NOT
        // coincidental. The single-pass binding gate (bound == grounded) misses it.
        let rows = rows_from(&[
            (0usize, wit(&[(Locus::Quorum, 7)])),
            (7, wit(&[(Locus::Quorum, 7)])), // rebinds → chain leaves ±8 → escalates
        ]);
        let lens = WitnessLens::new(&rows);
        let v = guard(None, 0, &lens, |_| true, 20, 8);
        assert_eq!(
            v.outcome,
            GateOutcome::Escalate,
            "wave catches the non-local cause → escalate over time, not coincidental"
        );
        assert!(v.unbound.is_none() && v.escalate == Some(Locus::Quorum));
        assert!(
            !v.fires(),
            "a non-local cause does NOT fire locally — it escalates"
        );
    }

    #[test]
    fn escalate_is_independent_of_the_single_pass_gate() {
        // The single-pass gate (loci_disqualifier) sees Quorum BOUND → would fire.
        // The wave gate escalates it. That divergence is the higher-confidence
        // redundancy — the horizontal axis the vertical binding is blind to.
        let a = wit(&[(Locus::Quorum, 7)]);
        let rows = rows_from(&[(0usize, a), (7, wit(&[(Locus::Quorum, 7)]))]);
        let lens = WitnessLens::new(&rows);
        assert!(
            loci_disqualifier(&a, 20).is_none(),
            "single-pass: bound → grounded"
        );
        assert_eq!(
            guard(None, 0, &lens, |_| true, 20, 8).outcome,
            GateOutcome::Escalate
        );
    }
}
