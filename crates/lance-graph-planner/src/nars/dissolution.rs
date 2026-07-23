//! `dissolution` — the S11 dissolution detector + Staunen↔Wisdom flow
//! accounting over a `BeliefArena` reasoning step
//! (`.claude/plans/dialectic-engine-v1.md` §1 S11, D-DIA-V3-A).
//!
//! This is the MIRROR of [`super::insight`]: where insight measures
//! crystallization WINNING (coherence closes), dissolution measures novelty
//! influx OUTRUNNING crystallization — "the rung tissue dissolves if the
//! thinking can't keep up" (operator pillar 4, plan §1 S11). It reuses
//! [`insight::Snapshot`](super::insight::Snapshot) as its sole signal carrier —
//! no new signal type is invented here.
//!
//! The RESPONSE to a dissolving step (the field-scale mass-induction sweep
//! minting parent concepts — V3-B, the second half of S11) is only TRIGGERED
//! from this module via [`should_elevate`]; it is NOT implemented here.
//!
//! **This is a registered convention, NOT a proven detector.** Per the
//! mandatory null-falsifier discipline (`E-BASIN-WIDTH`), "detects
//! dissolution" is only promotable once the discriminator BEATS a
//! size-preserving null — the flood-vs-crystallizing control in the
//! [`dissolution_beats_size_preserving_null`](tests) test. If a flooding
//! (disjoint, non-composing) influx does not out-score a size-matched
//! crystallizing (chain-extending) influx, the honest finding is that S11
//! measures size, not dissolution.

use super::insight::Snapshot;

/// Staunen — the novelty-influx pole (pillar 3): uncrystallized uncertainty
/// (confidence-spread entropy) + committed-contradiction tension (wonder). High
/// when new, unresolved material dominates the field.
#[must_use]
pub fn staunen(s: &Snapshot) -> f32 {
    0.5 * s.signals.truth_entropy + 0.5 * s.wonder
}

/// Wisdom — the crystallized-truth pole (pillar 3): closure density. High when
/// the graph has closed into strong transitive conclusions.
#[must_use]
pub fn wisdom(s: &Snapshot) -> f32 {
    s.coherence
}

/// The S11 flow/dissolution reading over one reasoning step (before→after).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Dissolution {
    /// ΔStaunen — novelty influx this step.
    pub d_staunen: f32,
    /// ΔWisdom — crystallization this step.
    pub d_wisdom: f32,
    /// Flow (pillar 3) — the crystallized FRACTION of the step's movement:
    /// `wisdom⁺ / (staunen⁺ + wisdom⁺)` (0 if neither moved). High when novelty
    /// influx was absorbed into closure; low when influx outran crystallization.
    pub flow: f32,
    /// Dissolution (S11) — fires ONLY when coherence failed to form this step
    /// (`Δwisdom ≤ 0`); then it is `clamp(Δstaunen − Δwisdom, 0, 1)` (the amount
    /// by which novelty influx outran crystallization). 0 when coherence rose
    /// (crystallization kept pace) — the rung tissue is NOT dissolving.
    pub dissolution: f32,
}

/// Score one reasoning step (S11). `before`/`after` bracket an INGESTION step
/// (new material observed + attempted closure), unlike `insight::detect` which
/// brackets a pure closure. `_yield_theta` reserved for symmetry with S10 (not
/// used by the delta-gated form).
#[must_use]
pub fn detect_dissolution(before: &Snapshot, after: &Snapshot, _yield_theta: f32) -> Dissolution {
    let d_staunen = staunen(after) - staunen(before);
    let d_wisdom = wisdom(after) - wisdom(before);
    let sp = d_staunen.max(0.0);
    let wp = d_wisdom.max(0.0);
    let flow = if sp + wp <= f32::EPSILON {
        0.0
    } else {
        wp / (sp + wp)
    };
    // "coherence unable to form" (plan S11) = wisdom did not rise this step.
    let dissolution = if d_wisdom <= f32::EPSILON {
        (d_staunen - d_wisdom).clamp(0.0, 1.0)
    } else {
        0.0
    };
    Dissolution {
        d_staunen,
        d_wisdom,
        flow,
        dissolution,
    }
}

/// The S11 field-elevation TRIGGER (not the response). When dissolution clears
/// `threshold`, the rung tissue is dissolving and the correct response is a
/// FIELD-scale mass-induction sweep minting parent concepts (V3-B — not here),
/// never per-thought churn (plan §1 S11).
#[must_use]
pub fn should_elevate(d: &Dissolution, threshold: f32) -> bool {
    d.dissolution > threshold
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::nars::{BeliefArena, CStmt, Copula, Stamp, TruthValue};

    fn inh(s: u16, p: u16) -> CStmt {
        CStmt {
            s,
            cop: Copula::Inh,
            p,
        }
    }

    /// Build a core chain, close it (snapshot BEFORE), then observe `new` edges and
    /// close again (snapshot AFTER), and score the ingestion step.
    fn score_ingest(core: &[(u16, u16)], new: &[(u16, u16)]) -> Dissolution {
        let mut arena = BeliefArena::new();
        for (i, &(s, p)) in core.iter().enumerate() {
            arena.observe(
                inh(s, p),
                TruthValue::new(0.95, 0.9),
                Stamp::source(i as u32),
            );
        }
        arena.close_transitive(64);
        let before = Snapshot::of(&arena, 0.0);
        for (j, &(s, p)) in new.iter().enumerate() {
            arena.observe(
                inh(s, p),
                TruthValue::new(0.95, 0.9),
                Stamp::source(1000 + j as u32),
            );
        }
        arena.close_transitive(64);
        let after = Snapshot::of(&arena, 0.0);
        detect_dissolution(&before, &after, 0.02)
    }

    /// The S11 MANDATORY gate (`E-BASIN-WIDTH`): a flooding influx (disjoint,
    /// non-composing edges) must score higher dissolution than a size-preserving
    /// crystallizing null (the SAME edge count, but extending the core chain so
    /// it composes into new transitive conclusions). If it does not, S11
    /// measures size, not crystallization-failure.
    #[test]
    fn dissolution_beats_size_preserving_null() {
        // Core: a chain 0→1→2→3→4→5→6 (6 edges).
        let core: Vec<(u16, u16)> = (0u16..6).map(|i| (i, i + 1)).collect();

        // Dissolving: 6 DISJOINT edges on fresh unconnected ids — compose
        // nothing, so closure density (wisdom) FALLS.
        let dissolving_new: Vec<(u16, u16)> = vec![
            (100, 101),
            (102, 103),
            (104, 105),
            (106, 107),
            (108, 109),
            (110, 111),
        ];
        let dissolving = score_ingest(&core, &dissolving_new);

        // Crystallizing null: SAME 6 edges but EXTENDING the core chain — they
        // close into many new transitive conclusions, so wisdom RISES.
        let crystallizing_new: Vec<(u16, u16)> = (6u16..12).map(|i| (i, i + 1)).collect();
        let crystallizing = score_ingest(&core, &crystallizing_new);

        assert!(
            dissolving.dissolution > crystallizing.dissolution,
            "S11 must beat the size-preserving null: dissolving {} vs crystallizing {} \
             (same edge count; if this fails, dissolution measures size, not \
             crystallization-failure)",
            dissolving.dissolution,
            crystallizing.dissolution
        );
        assert!(
            dissolving.dissolution > 0.0,
            "a flooding step registers dissolution"
        );
        assert_eq!(
            crystallizing.dissolution, 0.0,
            "a crystallizing step (wisdom rose) registers zero dissolution"
        );
    }

    /// A crystallizing step (novelty absorbed into closure) has high flow and
    /// zero dissolution.
    #[test]
    fn crystallizing_step_is_high_flow() {
        let core: Vec<(u16, u16)> = (0u16..6).map(|i| (i, i + 1)).collect();
        let crystallizing_new: Vec<(u16, u16)> = (6u16..12).map(|i| (i, i + 1)).collect();
        let crystallizing = score_ingest(&core, &crystallizing_new);

        assert!(
            crystallizing.flow > 0.5,
            "crystallizing step should have high flow, got {}",
            crystallizing.flow
        );
        assert_eq!(crystallizing.dissolution, 0.0);
    }

    /// A dissolving step (influx outran crystallization) has low flow — wisdom
    /// fell, so the crystallized-positive part is zero.
    #[test]
    fn dissolving_step_is_low_flow() {
        let core: Vec<(u16, u16)> = (0u16..6).map(|i| (i, i + 1)).collect();
        let dissolving_new: Vec<(u16, u16)> = vec![
            (100, 101),
            (102, 103),
            (104, 105),
            (106, 107),
            (108, 109),
            (110, 111),
        ];
        let dissolving = score_ingest(&core, &dissolving_new);

        assert!(
            dissolving.flow < 0.5,
            "dissolving step should have low flow, got {}",
            dissolving.flow
        );
    }

    /// `wisdom`/`staunen` read exactly from a `Snapshot`, and `should_elevate`
    /// gates correctly on `threshold`.
    #[test]
    fn poles_read_from_snapshot() {
        let mut arena = BeliefArena::new();
        arena.observe(inh(1, 2), TruthValue::new(0.9, 0.9), Stamp::source(0));
        arena.observe(inh(2, 3), TruthValue::new(0.9, 0.9), Stamp::source(1));
        arena.close_transitive(8);
        let snap = Snapshot::of(&arena, 0.0);

        assert_eq!(wisdom(&snap), snap.coherence);
        let expected_staunen = 0.5 * snap.signals.truth_entropy + 0.5 * snap.wonder;
        assert!((staunen(&snap) - expected_staunen).abs() < 1e-6);

        let d = Dissolution {
            d_staunen: 0.5,
            d_wisdom: -0.1,
            flow: 0.0,
            dissolution: 0.6,
        };
        assert!(should_elevate(&d, 0.5));
        assert!(!should_elevate(&d, 0.7));
    }
}
