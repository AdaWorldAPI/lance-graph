//! # `kanban` — the 4-phase Rubicon kanban contract (zero-dep).
//!
//! The seam where three subsystems meet over the ONE per-mailbox SoA:
//! - **lance-graph-planner** emits a [`KanbanMove`] (the plan's output unit),
//! - **ractor** (the mailbox owner, `lance-graph-supervisor`) drives the
//!   transition — advancing a [`KanbanColumn`] *is* the mailbox lifecycle step,
//! - **surrealdb** (`surreal_container`) projects the columns as the kanban view
//!   over SoA-shaped Lance rows.
//!
//! Carried across the canonical [`crate::orchestration::OrchestrationBridge`] as a
//! `UnifiedStep { step_type: "kanban.*" }` ([`crate::orchestration::StepDomain::Kanban`]).
//!
//! Spec: `.claude/plans/unified-soa-convergence-v1.md` §5 + §8.4 (D-MBX-A6 Phase 1).
//!
//! ## Invariants honoured
//! - **R1 "one SoA never transformed":** a [`KanbanMove`] is a *transition record*,
//!   not SoA data — it carries only `Copy` scalars + a pointer, never the SoA.
//! - **R4 witness-as-pointer:** the witness is a `chain_position` index into the
//!   source mailbox's witness arc (mirrors
//!   [`crate::collapse_gate::CollapseGateEmission`]'s `chain_position`), never the
//!   witnessed data.

use crate::collapse_gate::MailboxId;

/// The four Rubicon phases (+ two terminal exits), Libet-anchored.
///
/// The mailbox lifecycle advances through these columns from spawn toward a
/// terminal column. The discriminants are stable (used as a kanban-column key and
/// for compact SoA storage) — **do not reorder**.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
#[repr(u8)]
pub enum KanbanColumn {
    /// `t < -550 ms` (Libet readiness-potential window): ractor owns the SoA;
    /// counterfactual pre-planning / expansion happens here. The spawn state.
    #[default]
    Planning = 0,
    /// `t >= -550 ms`: the SoA mutates under cognitive operations; the Σ-commit
    /// ratchet advances here.
    CognitiveWork = 1,
    /// `t > 0`: read back over the witness arc; residual free-energy assessed.
    Evaluation = 2,
    /// Terminal — calcify: commit to Lance SPO-G + AriGraph pointer.
    Commit = 3,
    /// Terminal — re-plan: re-enter [`Planning`](KanbanColumn::Planning) carrying
    /// the witness (the "act differently next time" exit).
    Plan = 4,
    /// Terminal — veto: drop the move (Libet "free won't", post-hoc inhibition).
    Prune = 5,
}

impl KanbanColumn {
    /// Is this a terminal column (no further in-cycle transition from here)?
    #[inline]
    pub fn is_terminal(self) -> bool {
        matches!(self, Self::Commit | Self::Plan | Self::Prune)
    }
}

/// One kanban transition: the planner's output unit and the ractor's lifecycle step.
///
/// `Copy` and small (≤ 16 B) so it rides the airgap as owned microcopy, never a
/// borrow into the SoA (R1). The witness is a *pointer* (R4).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct KanbanMove {
    /// The mailbox whose lifecycle is advancing.
    pub mailbox: MailboxId,
    /// Column the mailbox is leaving.
    pub from: KanbanColumn,
    /// Column the mailbox is entering.
    pub to: KanbanColumn,
    /// Witness pointer: position in the source mailbox's witness chain. Mirrors
    /// [`crate::collapse_gate::CollapseGateEmission`]'s `chain_position` —
    /// structural time, not a wall-clock stamp (R4).
    pub witness_chain_position: u32,
    /// Libet commit anchor: signed micros relative to the act. `-550_000` on the
    /// `Planning → CognitiveWork` Σ-commit; `0` otherwise. Structural offset only.
    pub libet_offset_us: i32,
}

// NOTE (follow-up, D-MBX-A6 Phase 2-3): the planner execution strategy
// { lance-graph-planner (native) | JIT | SurrealQL | elixir } is intentionally NOT carried
// on KanbanMove here — the planner-emit slice reuses the planner's existing strategy enum
// rather than duplicating it. Revisit whether KanbanMove needs an exec-target tag then.

// `KanbanMove` must stay a small owned microcopy (airgap discipline, I1):
// MailboxId(4) + 2×KanbanColumn(1) + u32(4) + i32(4) packs within 16 B.
const _: () = assert!(core::mem::size_of::<KanbanMove>() <= 16);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn kanban_column_discriminants_are_stable() {
        // Stable column key — do not reorder.
        assert_eq!(KanbanColumn::Planning as u8, 0);
        assert_eq!(KanbanColumn::CognitiveWork as u8, 1);
        assert_eq!(KanbanColumn::Evaluation as u8, 2);
        assert_eq!(KanbanColumn::Commit as u8, 3);
        assert_eq!(KanbanColumn::Plan as u8, 4);
        assert_eq!(KanbanColumn::Prune as u8, 5);
    }

    #[test]
    fn default_column_is_planning_the_spawn_state() {
        assert_eq!(KanbanColumn::default(), KanbanColumn::Planning);
    }

    #[test]
    fn terminal_columns_are_commit_plan_prune() {
        assert!(KanbanColumn::Commit.is_terminal());
        assert!(KanbanColumn::Plan.is_terminal());
        assert!(KanbanColumn::Prune.is_terminal());
        assert!(!KanbanColumn::Planning.is_terminal());
        assert!(!KanbanColumn::CognitiveWork.is_terminal());
        assert!(!KanbanColumn::Evaluation.is_terminal());
    }

    #[test]
    fn kanban_move_is_copy_and_small() {
        let m = KanbanMove {
            mailbox: 42,
            from: KanbanColumn::Planning,
            to: KanbanColumn::CognitiveWork,
            witness_chain_position: 7,
            libet_offset_us: -550_000,
        };
        let n = m; // Copy, not move
        assert_eq!(m, n);
        assert!(core::mem::size_of::<KanbanMove>() <= 16);
    }
}
