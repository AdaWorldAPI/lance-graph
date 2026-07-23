//! `regulate` — the S11 trigger→response regulation loop (D-DIA-V3-C, plan
//! `.claude/plans/dialectic-engine-v1.md` §1 S11).
//!
//! V3-A shipped the dissolution DETECTOR ([`super::dissolution::detect_dissolution`])
//! and the field-elevation TRIGGER ([`super::dissolution::should_elevate`]); V3-B
//! shipped `elevate_field` the RESPONSE ([`super::elevation::elevate_field`]). This
//! module CLOSES the loop: one regulation cycle that measures dissolution over a
//! step and, when the tissue is dissolving past a threshold, fires field-elevation
//! and re-closes. This is the concrete active-inference loop — the system doesn't
//! choose to elevate; dissolution *makes* it. Pure composition of shipped pieces —
//! no new detector, no new engine.
//!
//! The caller captures the `before` snapshot BEFORE ingesting the step's new
//! material, then calls [`regulate_cycle`], which closes, measures dissolution,
//! and elevates iff the tissue is dissolving. The loop is bounded (V3-B's
//! across-sweep idempotence prevents runaway minting).

use super::belief::BeliefArena;
use super::dissolution::{detect_dissolution, should_elevate, Dissolution};
use super::elevation::{elevate_field, Elevation};
use super::insight::Snapshot;

/// Configuration for one [`regulate_cycle`].
#[derive(Debug, Clone, Copy)]
pub struct CycleConfig {
    /// Passed to `detect_dissolution` (reserved; the delta-gated form ignores it).
    pub yield_theta: f32,
    /// Dissolution above this fires field-elevation (`should_elevate`).
    pub elevate_threshold: f32,
    /// Minimum shared-predicate cluster size to lift (`elevate_field`).
    pub min_cluster: usize,
    /// `close_transitive` pass budget.
    pub max_passes: u32,
}

impl Default for CycleConfig {
    fn default() -> Self {
        Self {
            yield_theta: 0.02,
            elevate_threshold: 0.01,
            min_cluster: 3,
            max_passes: 64,
        }
    }
}

/// What one regulation cycle did.
#[derive(Debug, Clone)]
pub struct CycleOutcome {
    /// The dissolution reading over the step (before → post-close).
    pub dissolution: Dissolution,
    /// `Some(elevation)` iff the tissue dissolved past the threshold and
    /// field-elevation fired; `None` if crystallization kept pace.
    pub elevated: Option<Elevation>,
    /// The arena snapshot after the cycle (post-elevation if it fired).
    pub after: Snapshot,
}

/// One S11 regulation cycle. `before` is the snapshot the caller captured
/// BEFORE ingesting this step's new material into `arena`. The cycle:
///
/// 1. `close_transitive` (crystallize what the new material composes);
/// 2. snapshot `after`, `detect_dissolution(before, after)`;
/// 3. if `should_elevate` → `elevate_field` + re-close (the response).
///
/// The elevation is TRIGGERED by the measurement, never chosen — the
/// active-inference loop.
#[must_use]
pub fn regulate_cycle(
    arena: &mut BeliefArena,
    before: &Snapshot,
    cfg: &CycleConfig,
) -> CycleOutcome {
    arena.close_transitive(cfg.max_passes);
    let after_pre = Snapshot::of(arena, 0.0);
    let dissolution = detect_dissolution(before, &after_pre, cfg.yield_theta);
    let elevated = if should_elevate(&dissolution, cfg.elevate_threshold) {
        let e = elevate_field(arena, cfg.min_cluster);
        arena.close_transitive(cfg.max_passes);
        Some(e)
    } else {
        None
    };
    let after = Snapshot::of(arena, 0.0);
    CycleOutcome {
        dissolution,
        elevated,
        after,
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

    /// Build a small coherent core (a chain 0→1→…→5), closed.
    fn core_arena() -> BeliefArena {
        let mut a = BeliefArena::new();
        for i in 0u16..6 {
            a.observe(
                inh(i, i + 1),
                TruthValue::new(0.95, 0.9),
                Stamp::source(i.into()),
            );
        }
        a.close_transitive(64);
        a
    }

    /// A dissolving step (a shared-predicate flood that does not compose with
    /// the core) triggers field-elevation, and the elevation lifts exactly the
    /// flooded cluster.
    #[test]
    fn dissolving_step_triggers_elevation() {
        let mut a = core_arena();
        let before = Snapshot::of(&a, 0.0);

        // 5 subjects share predicate 900 — disjoint from the core, non-composing.
        for s in 200u16..205 {
            a.observe(
                inh(s, 900),
                TruthValue::new(0.9, 0.9),
                Stamp::source(s.into()),
            );
        }

        let out = regulate_cycle(&mut a, &before, &CycleConfig::default());

        assert!(
            out.dissolution.dissolution > 0.0,
            "coherence density fell → dissolving"
        );
        assert!(
            out.elevated.is_some(),
            "the loop must fire elevation on a dissolving step"
        );
        let e = out.elevated.unwrap();
        assert_eq!(e.clusters_lifted, 1);
        assert_eq!(e.children_rehomed, 5);
    }

    /// A crystallizing step (chain-extending material that composes) does NOT
    /// trigger elevation.
    #[test]
    fn crystallizing_step_no_elevation() {
        let mut a = core_arena();
        let before = Snapshot::of(&a, 0.0);

        // Extend the core chain — composes into new transitive conclusions.
        for i in 6u16..12 {
            a.observe(
                inh(i, i + 1),
                TruthValue::new(0.95, 0.9),
                Stamp::source((100 + i).into()),
            );
        }

        let out = regulate_cycle(&mut a, &before, &CycleConfig::default());

        assert_eq!(
            out.dissolution.dissolution, 0.0,
            "coherence rose → not dissolving"
        );
        assert!(
            out.elevated.is_none(),
            "no elevation should fire when crystallization kept pace"
        );
    }

    /// The regulation loop is stable across cycles: once the field has been
    /// elevated and no new material is ingested, subsequent cycles neither
    /// elevate again nor grow the arena (bounded — no runaway minting).
    #[test]
    fn loop_is_stable_across_cycles() {
        let mut a = core_arena();
        let before = Snapshot::of(&a, 0.0);
        for s in 200u16..205 {
            a.observe(
                inh(s, 900),
                TruthValue::new(0.9, 0.9),
                Stamp::source(s.into()),
            );
        }
        let first = regulate_cycle(&mut a, &before, &CycleConfig::default());
        assert!(first.elevated.is_some(), "first cycle elevates");

        let entries_after_first = a.entries().len();

        // Two more cycles, no new material ingested between them.
        for _ in 0..2 {
            let before_extra = Snapshot::of(&a, 0.0);
            let out = regulate_cycle(&mut a, &before_extra, &CycleConfig::default());
            assert!(
                out.elevated.is_none(),
                "no new dissolution without new influx on an already-abstracted field"
            );
            assert_eq!(
                a.entries().len(),
                entries_after_first,
                "the arena must not grow across idle regulation cycles"
            );
        }
    }
}
