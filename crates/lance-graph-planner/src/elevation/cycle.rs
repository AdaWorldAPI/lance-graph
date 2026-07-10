//! Cycle Budget — the ONE per-cycle budget allocator (D-V3-W2d, milestone M12).
//!
//! Collapses the workspace's two budget concepts into one source:
//!
//! - the **per-cycle** −550 000 µs **Libet anchor** the contract's scheduler
//!   stamps on the `Planning → CognitiveWork` Σ-commit crossing
//!   (`lance_graph_contract::scheduler::NextPhaseScheduler`,
//!   `KanbanMove::libet_offset_us`) — the readiness window a thinking cycle
//!   has before its commit lands;
//! - the **per-strategy** [`PatienceBudget`] this module's sibling
//!   [`budget`](super::budget) derives from the thinking cluster.
//!
//! Before M12 the two drifted independently (a Convergent strategy could
//! claim 500 ms of patience inside a cycle that had 80 ms left). Now the
//! elevation layer **reads the Libet anchor** ([`CycleBudget::from_move`])
//! and every strategy slice is carved FROM the cycle's remainder
//! ([`CycleBudget::slice_for`]) — extend, don't shadow: the slice IS a
//! `budget_for_cluster` value with its latency capped, never a new type.
//!
//! **Advisory, never gating.** Per the standing V3 rule ("updates
//! reprioritize, never gate"), an exhausted budget must not deadlock a
//! cycle: [`CycleBudget::admits`] informs load-balancing/prioritization —
//! callers deprioritize or elevate, they do not refuse work. Write latency
//! may be treated as masked (Addendum-6 eager drain) so long as sink
//! throughput ≥ delta production rate — instrument both, gate neither.
//!
//! Cross-ref (M12 "both ways"): `contract::scheduler` (the anchor's write
//! side), `contract::kanban::KanbanMove::libet_offset_us` (the carrier),
//! `.claude/v3/ENTROPY-MILESTONES.md` M12, INTEGRATION-PLAN W2d.

use super::budget::{budget_for_cluster, PatienceBudget};
use crate::thinking::style::ThinkingCluster;
use lance_graph_contract::kanban::KanbanMove;
use std::time::Duration;

/// The per-cycle net thinking budget, in µs — the magnitude of the Libet
/// anchor (`-550_000 µs`) the contract scheduler stamps on the Σ-commit
/// crossing. A parity test below pins this against the REAL stamped move so
/// the two constants cannot drift apart silently.
pub const LIBET_CYCLE_BUDGET_US: u32 = 550_000;

/// Measured per-card kanban overhead (spawn + 3 Rubicon ticks + join):
/// **~66 µs** — onebrc-probe lane E (t2, 2026-07-02), fine granularity.
/// ~0.01 % of the cycle budget; the board is not a scheduling threat.
/// Measured, not hand-tuned — re-measure via `crates/onebrc-probe` lane E.
pub const KANBAN_CARD_OVERHEAD_US: u32 = 66;

/// Measured graph-flow per-step dispatch overhead: **~0.5 µs** (408–538 ns,
/// release, batch 4096 — Addendum-5 bench, 2026-07-02). Measured, not
/// hand-tuned — re-measure via the rs-graph-llm dispatch bench.
pub const GRAPH_FLOW_STEP_OVERHEAD_NS: u32 = 500;

/// The one per-cycle budget allocator (M12). Constructed from the Libet
/// anchor on the Σ-commit move (the read side of the M12 gate), charged as
/// work completes, and sliced per-strategy via [`Self::slice_for`].
///
/// Integer µs arithmetic throughout (saturating) — no f64, no NaN.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CycleBudget {
    total_us: u32,
    spent_us: u32,
}

impl CycleBudget {
    /// A fresh budget of `total_us` microseconds.
    pub const fn new(total_us: u32) -> Self {
        Self {
            total_us,
            spent_us: 0,
        }
    }

    /// The canonical per-cycle budget — the Libet window.
    pub const fn libet() -> Self {
        Self::new(LIBET_CYCLE_BUDGET_US)
    }

    /// **The M12 read side:** derive the cycle budget from a stamped
    /// [`KanbanMove`]. `Some` exactly when the move carries a Libet anchor
    /// (a negative `libet_offset_us` — the Σ-commit `Planning →
    /// CognitiveWork` crossing); the budget is the anchor's magnitude.
    /// Mid-cycle moves (offset `0`) carry no window → `None` (keep the
    /// current budget; a move never *shrinks* the cycle).
    pub fn from_move(mv: &KanbanMove) -> Option<Self> {
        if mv.libet_offset_us < 0 {
            Some(Self::new(mv.libet_offset_us.unsigned_abs()))
        } else {
            None
        }
    }

    /// Charge `us` microseconds of completed work (saturating — spending
    /// past the window records exhaustion, it never wraps or panics).
    pub fn charge(&mut self, us: u32) {
        self.spent_us = self.spent_us.saturating_add(us);
    }

    /// Microseconds remaining in the window (0 when exhausted).
    pub const fn remaining_us(&self) -> u32 {
        self.total_us.saturating_sub(self.spent_us)
    }

    /// Microseconds spent so far (may exceed `total_us` — see [`Self::charge`]).
    pub const fn spent_us(&self) -> u32 {
        self.spent_us
    }

    /// Has the window been fully spent?
    pub const fn exhausted(&self) -> bool {
        self.spent_us >= self.total_us
    }

    /// **Advisory** admission: does an estimated `estimated_us` of work fit
    /// the remainder? Load-balancers use this to *reprioritize* (defer the
    /// item, pick a cheaper elevation level, batch it into the next cycle)
    /// — NEVER to refuse or deadlock the cycle (standing rule: updates
    /// reprioritize, never gate).
    pub const fn admits(&self, estimated_us: u32) -> bool {
        estimated_us <= self.remaining_us()
    }

    /// How many kanban cards the remainder can carry at the measured
    /// ~66 µs/card overhead ([`KANBAN_CARD_OVERHEAD_US`]) — the
    /// load-balancing estimate for lane sizing over a 64k–256k SoA.
    pub const fn card_capacity(&self) -> u32 {
        self.remaining_us() / KANBAN_CARD_OVERHEAD_US
    }

    /// **The M12 collapse:** carve a per-strategy [`PatienceBudget`] slice
    /// from THIS cycle's remainder. Extend-don't-shadow: the slice is
    /// exactly [`budget_for_cluster`]'s value with `latency_budget` capped
    /// at the remaining window — fan-out/memory/ceiling personalities are
    /// untouched. A Convergent 500 ms patience inside an 80 ms remainder
    /// yields an 80 ms slice; the same call early in the cycle yields the
    /// full 500 ms.
    pub fn slice_for(&self, cluster: ThinkingCluster) -> PatienceBudget {
        let mut slice = budget_for_cluster(cluster);
        let cap = Duration::from_micros(self.remaining_us() as u64);
        slice.latency_budget = slice.latency_budget.min(cap);
        slice
    }
}

impl Default for CycleBudget {
    /// Defaults to the Libet window — the canonical cycle.
    fn default() -> Self {
        Self::libet()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use lance_graph_contract::collapse_gate::MailboxId;
    use lance_graph_contract::kanban::{ExecTarget, KanbanColumn};
    use lance_graph_contract::scheduler::{DatasetVersion, NextPhaseScheduler, VersionScheduler};
    use lance_graph_contract::soa_view::MailboxSoaView;

    /// Minimal view at a given phase (mirrors the supervisor's TestBoard idiom).
    struct PhaseView(KanbanColumn);
    impl MailboxSoaView for PhaseView {
        fn mailbox_id(&self) -> MailboxId {
            7
        }
        fn n_rows(&self) -> usize {
            0
        }
        fn w_slot(&self) -> u8 {
            7
        }
        fn current_cycle(&self) -> u32 {
            0
        }
        fn phase(&self) -> KanbanColumn {
            self.0
        }
        fn energy(&self) -> &[f32] {
            &[]
        }
        fn edges_raw(&self) -> &[u64] {
            &[]
        }
        fn meta_raw(&self) -> &[u32] {
            &[]
        }
        fn entity_type(&self) -> &[u16] {
            &[]
        }
    }

    #[test]
    fn libet_constant_pins_the_real_scheduler_stamp_no_silent_drift() {
        // The M12 parity gate: our budget constant equals the magnitude the
        // REAL contract scheduler stamps on the Σ-commit crossing. If the
        // contract anchor ever changes, this test fails loudly instead of
        // the two constants drifting apart.
        let mv = NextPhaseScheduler
            .on_version(
                &PhaseView(KanbanColumn::Planning),
                DatasetVersion(1),
                ExecTarget::Native,
            )
            .expect("Planning proposes the forward arc");
        assert_eq!(mv.to, KanbanColumn::CognitiveWork);
        assert!(mv.libet_offset_us < 0, "Σ-commit carries the anchor");
        assert_eq!(mv.libet_offset_us.unsigned_abs(), LIBET_CYCLE_BUDGET_US);

        // The read side: the budget derives FROM the stamped move.
        let budget = CycleBudget::from_move(&mv).expect("anchored move opens a window");
        assert_eq!(budget.remaining_us(), LIBET_CYCLE_BUDGET_US);
        assert_eq!(budget, CycleBudget::libet());
    }

    #[test]
    fn mid_cycle_moves_open_no_window() {
        // A mid-cycle advance (CognitiveWork → Evaluation) carries offset 0:
        // no new window — the current budget keeps running.
        let mv = NextPhaseScheduler
            .on_version(
                &PhaseView(KanbanColumn::CognitiveWork),
                DatasetVersion(2),
                ExecTarget::Native,
            )
            .expect("forward arc");
        assert_eq!(mv.libet_offset_us, 0);
        assert!(CycleBudget::from_move(&mv).is_none());
    }

    #[test]
    fn charge_saturates_and_admission_is_advisory_arithmetic() {
        let mut b = CycleBudget::new(100);
        assert!(b.admits(100));
        b.charge(60);
        assert_eq!(b.remaining_us(), 40);
        assert!(b.admits(40));
        assert!(!b.admits(41)); // advisory: caller reprioritizes, never refuses
        b.charge(u32::MAX); // overspend saturates, never wraps/panics
        assert!(b.exhausted());
        assert_eq!(b.remaining_us(), 0);
        assert!(!b.admits(1));
        assert_eq!(b.card_capacity(), 0);
    }

    #[test]
    fn strategy_slices_are_carved_from_the_cycle_remainder() {
        use std::time::Duration;

        // Early in the cycle a Convergent slice keeps its full 500 ms
        // personality (the cycle's 550 ms window doesn't bind).
        let fresh = CycleBudget::libet();
        let early = fresh.slice_for(ThinkingCluster::Convergent);
        assert_eq!(early.latency_budget, Duration::from_millis(500));

        // Late in the cycle (80 ms left) the SAME cluster is capped to the
        // remainder — the M12 collapse: one budget source, no drift.
        let mut late = CycleBudget::libet();
        late.charge(470_000);
        let slice = late.slice_for(ThinkingCluster::Convergent);
        assert_eq!(slice.latency_budget, Duration::from_millis(80));
        // Fan-out/memory/ceiling personalities are untouched (extend, don't
        // shadow) — identical to the uncapped cluster budget.
        let base = budget_for_cluster(ThinkingCluster::Convergent);
        assert_eq!(slice.result_threshold, base.result_threshold);
        assert_eq!(slice.memory_budget, base.memory_budget);
        assert_eq!(slice.ceiling, base.ceiling);

        // A Speed slice (10 ms) is never inflated by a large remainder.
        let speedy = fresh.slice_for(ThinkingCluster::Speed);
        assert_eq!(speedy.latency_budget, Duration::from_millis(10));
    }

    #[test]
    fn card_capacity_matches_the_lane_e_measurement_scale() {
        // 550_000 µs / 66 µs-per-card ≈ 8333 cards per cycle — the lane-E
        // finding ("~0.01 % of the budget per card") at load-balancer scale.
        let b = CycleBudget::libet();
        assert_eq!(
            b.card_capacity(),
            LIBET_CYCLE_BUDGET_US / KANBAN_CARD_OVERHEAD_US
        );
        assert!(b.card_capacity() > 8_000);
    }
}
