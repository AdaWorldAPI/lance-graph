//! `reach_out` — the §3.6 felt-integration criterion for fetched/external
//! material (D-DIA-V5-A, `.claude/plans/dialectic-engine-v1.md` §3.6).
//!
//! Fetched/external material (spider/arXiv; the no-LLM constraint stays
//! intact — this is pure arena composition, not generation) is QUARANTINED
//! (its prior confidence capped) and integrates ONLY when the new concept
//! serves as the **middle term composing two disjoint-stamp pre-existing
//! beliefs** — i.e. adding it lands at least one NEW derivation via
//! transitive closure. Two felt forms of the same event:
//!
//! - **`DullShadow`** = nothing moves: no derivation lands (recognition
//!   without composition).
//! - **`NewInsight`** = the middle-term click: at least one new derivation
//!   lands and coherence (closure density) rises — the spark.
//!
//! This is the SAME event read two ways: the audit form (a derivation with
//! stamps, [`super::belief::BeliefArena`]) and the felt form (coherence /
//! expansion rising, [`super::insight::Snapshot`]). This module measures it
//! by composing the two shipped pieces — no new detector, no new engine.

use super::belief::{BeliefArena, CStmt, Stamp};
use super::insight::Snapshot;
use super::truth::TruthValue;

/// The felt form of a reach-out integration (plan §3.6).
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum FeltOutcome {
    /// The middle-term click — the quarantined bridge composed pre-existing
    /// beliefs into `new_derivations` (≥ 1) fresh derivations; closure
    /// density rose by `coherence_gain`. The spark: expansion and coherence
    /// rise together.
    NewInsight {
        /// Count of new derivations that landed as a result of the bridge.
        new_derivations: usize,
        /// Rise in closure density (`Snapshot::coherence`) across the step.
        coherence_gain: f32,
    },
    /// Recognition without composition — the bridge landed no new
    /// derivation. It stays quarantined; nothing in the field moved.
    DullShadow,
}

/// Configuration for a reach-out integration.
#[derive(Debug, Clone, Copy)]
pub struct ReachOutConfig {
    /// Confidence cap applied to fetched material (the quarantine prior,
    /// §3.6).
    pub quarantine_prior: f32,
    /// `close_transitive` pass budget.
    pub max_passes: u32,
}

impl Default for ReachOutConfig {
    fn default() -> Self {
        Self {
            quarantine_prior: 0.1,
            max_passes: 64,
        }
    }
}

/// Integrate one fetched `bridge` belief under the §3.6 felt criterion. The
/// bridge is observed at a quarantined confidence
/// (`min(truth.confidence, cfg.quarantine_prior)`), the arena is closed, and
/// the outcome is the felt form: [`FeltOutcome::NewInsight`] iff at least one
/// new derivation landed (the bridge served as a middle term composing
/// pre-existing beliefs), else [`FeltOutcome::DullShadow`].
///
/// The arena is closed to a FIXED POINT before the baseline is snapshotted, so
/// the reported result depends only on the fetched `bridge` — never on whatever
/// pending (un-closed) derivations the caller happened to pass in. Because the
/// pre-bridge arena is already at a fixed point, every derivation counted below
/// is provably bridge-caused (codex #830 P2).
#[must_use]
pub fn reach_out_integrate(
    arena: &mut BeliefArena,
    bridge: CStmt,
    truth: TruthValue,
    stamp: Stamp,
    cfg: &ReachOutConfig,
) -> FeltOutcome {
    // Establish the baseline at a fixed point FIRST — otherwise a pending
    // pre-existing chain (e.g. `A→B, B→C` not yet closed) would land during the
    // post-bridge close and be miscredited to the bridge (codex #830 P2).
    arena.close_transitive(cfg.max_passes);
    let before = Snapshot::of(arena, 0.0);
    let derived_before = arena.entries().iter().filter(|b| b.rung >= 1).count();

    let quarantined = TruthValue::new(truth.frequency, truth.confidence.min(cfg.quarantine_prior));
    arena.observe(bridge, quarantined, stamp);
    arena.close_transitive(cfg.max_passes);

    let after = Snapshot::of(arena, 0.0);
    let derived_after = arena.entries().iter().filter(|b| b.rung >= 1).count();
    let new_derivations = derived_after.saturating_sub(derived_before);

    if new_derivations > 0 {
        FeltOutcome::NewInsight {
            new_derivations,
            coherence_gain: after.coherence - before.coherence,
        }
    } else {
        FeltOutcome::DullShadow
    }
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

    /// A bridge that composes with a pre-existing, disjoint-stamp belief as a
    /// middle term is the middle-term click: `close_transitive` derives the
    /// new `A is_a B` conclusion, and the felt outcome is `NewInsight` with
    /// rising coherence.
    #[test]
    fn middle_term_click_is_new_insight() {
        let mut a = BeliefArena::new();
        // A is_a M (pre-existing, rung 0).
        a.observe(inh(1, 9), TruthValue::new(0.9, 0.9), Stamp::source(0));

        let out = reach_out_integrate(
            &mut a,
            inh(9, 2), // M is_a B — the fetched bridge.
            TruthValue::new(0.9, 0.9),
            Stamp::source(1), // disjoint from source 0.
            &ReachOutConfig::default(),
        );

        let FeltOutcome::NewInsight {
            new_derivations,
            coherence_gain,
        } = out
        else {
            panic!("middle-term composition must be NewInsight, got {out:?}");
        };
        assert!(new_derivations >= 1, "at least one derivation must land");
        assert!(
            a.get(inh(1, 2)).is_some(),
            "the composed A is_a B conclusion must be present"
        );
        assert!(
            coherence_gain > 0.0,
            "closure density must rise on the click: {coherence_gain}"
        );
    }

    /// A bridge that shares no term with anything already in the arena
    /// composes with nothing — no derivation lands, and the felt outcome is
    /// `DullShadow`: recognition without composition.
    #[test]
    fn lone_concept_is_dull_shadow() {
        let mut a = BeliefArena::new();
        a.observe(inh(1, 9), TruthValue::new(0.9, 0.9), Stamp::source(0));

        let out = reach_out_integrate(
            &mut a,
            inh(100, 101), // fresh, unconnected terms.
            TruthValue::new(0.9, 0.9),
            Stamp::source(1),
            &ReachOutConfig::default(),
        );

        assert_eq!(
            out,
            FeltOutcome::DullShadow,
            "an unconnected bridge composes with nothing"
        );
    }

    /// Both the insight and the shadow case ingest exactly ONE bridge belief
    /// — the distinction is composition (does it serve as a middle term?),
    /// never the count of beliefs added. This is the honest, size-preserving
    /// form of the §3.6 criterion.
    #[test]
    fn shadow_and_insight_are_size_matched() {
        let mut insight_arena = BeliefArena::new();
        insight_arena.observe(inh(1, 9), TruthValue::new(0.9, 0.9), Stamp::source(0));
        // Exactly one bridge belief added: composes as a middle term.
        let insight_out = reach_out_integrate(
            &mut insight_arena,
            inh(9, 2),
            TruthValue::new(0.9, 0.9),
            Stamp::source(1),
            &ReachOutConfig::default(),
        );

        let mut shadow_arena = BeliefArena::new();
        shadow_arena.observe(inh(1, 9), TruthValue::new(0.9, 0.9), Stamp::source(0));
        // Exactly one bridge belief added: composes with nothing.
        let shadow_out = reach_out_integrate(
            &mut shadow_arena,
            inh(100, 101),
            TruthValue::new(0.9, 0.9),
            Stamp::source(1),
            &ReachOutConfig::default(),
        );

        assert!(matches!(insight_out, FeltOutcome::NewInsight { .. }));
        assert_eq!(shadow_out, FeltOutcome::DullShadow);
    }

    /// Fetched material is quarantined regardless of the confidence offered:
    /// after integration the stored belief's confidence is capped at
    /// `quarantine_prior`, never trusted at face value.
    #[test]
    fn quarantine_caps_confidence() {
        let mut a = BeliefArena::new();
        a.observe(inh(1, 9), TruthValue::new(0.9, 0.9), Stamp::source(0));

        let bridge = inh(9, 2);
        let cfg = ReachOutConfig {
            quarantine_prior: 0.1,
            max_passes: 64,
        };
        let _ = reach_out_integrate(
            &mut a,
            bridge,
            TruthValue::new(0.9, 0.95), // high offered confidence.
            Stamp::source(1),
            &cfg,
        );

        let stored = a.get(bridge).expect("bridge belief must be present");
        assert!(
            stored.truth.confidence <= 0.1 + 1e-6,
            "fetched material must stay quarantined, got confidence {}",
            stored.truth.confidence
        );
    }

    /// The result must depend on the BRIDGE, not on the caller's prior closure
    /// state (codex #830 P2). An arena with a pending, un-closed chain
    /// (`A→B, B→C`) plus a totally unrelated fetched bridge must still be a
    /// `DullShadow`: the pending `A→C` closes in the BASELINE (fixed point
    /// first), so it is never miscredited to the bridge.
    #[test]
    fn pending_closure_is_not_miscredited_to_an_unrelated_bridge() {
        let mut a = BeliefArena::new();
        // An OPEN chain: A→B, B→C, deliberately NOT closed by the caller.
        a.observe(inh(1, 2), TruthValue::new(0.9, 0.9), Stamp::source(0));
        a.observe(inh(2, 3), TruthValue::new(0.9, 0.9), Stamp::source(1));
        // (no close_transitive here — the arena is passed in un-closed)

        let out = reach_out_integrate(
            &mut a,
            inh(100, 101), // fresh, unrelated to the pending chain.
            TruthValue::new(0.9, 0.9),
            Stamp::source(2),
            &ReachOutConfig::default(),
        );

        assert_eq!(
            out,
            FeltOutcome::DullShadow,
            "an unrelated bridge must not inherit the pending A→C closure as \
             its own insight"
        );
        // Sanity: the pending A→C DID get closed (in the baseline), it just
        // wasn't credited to the bridge.
        assert!(
            a.get(inh(1, 3)).is_some_and(|b| b.rung >= 1),
            "the pending chain still closes — into the baseline, not the bridge's credit"
        );
    }
}
