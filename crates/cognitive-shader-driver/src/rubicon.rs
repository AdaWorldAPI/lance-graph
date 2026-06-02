//! `rubicon` — map a resolver verdict onto the Rubicon kanban crossing.
//!
//! Seam 1 of `.claude/plans/le-domino-cognition-v1.md`: the one rung that connects
//! the two already-shipped halves — the SPO-2³ resolver
//! ([`causal_edge::CausalEdge64::route_against`] → [`DominoStep`], 6 tests green)
//! and the Libet-anchored Rubicon lifecycle ([`KanbanColumn`], DAG-tested) — into a
//! live, DAG-legal *resolve → Rubicon decision* mapping.
//!
//! `cognitive-shader-driver` is the only crate that can name **both** enums:
//! `DominoStep` lives in the zero-dep `causal-edge`, `KanbanColumn` in the zero-dep
//! `lance-graph-contract`, and neither can see the other — so this cannot be a
//! `From` impl (orphan rule); it is a free function here.
//!
//! Pure, total, zero-state, zero-`unsafe`. The mailbox-owner write (a
//! `MailboxSoaOwner` impl + the ractor `handle()` that calls this) is a later slice;
//! this is only the decision function.

use causal_edge::DominoStep;
use lance_graph_contract::kanban::KanbanColumn;

/// Map a resolver verdict ([`DominoStep`], from `route_against`) onto the legal
/// **Rubicon terminal decision at [`KanbanColumn::Evaluation`]** — the one column
/// where a resolved `(f,c)` verdict drives the kanban lifecycle.
///
/// The resolver runs while the mailbox is in [`CognitiveWork`](KanbanColumn::CognitiveWork);
/// at [`Evaluation`](KanbanColumn::Evaluation) (`t > 0`, residual free-energy
/// assessed) its verdict decides the 3-way terminal
/// (`Evaluation.next_phases() == [Commit, Plan, Prune]`):
///
/// | verdict | column | meaning |
/// |---|---|---|
/// | [`Settle`](DominoStep::Settle)     | [`Commit`](KanbanColumn::Commit) | converged → calcify the resolution as a Fact (the action lands) |
/// | [`Fork`](DominoStep::Fork)         | [`Plan`](KanbanColumn::Plan)     | confident contradiction → re-deliberate (caller also deposits the counterfactual lane) |
/// | [`Escalate`](DominoStep::Escalate) | [`Plan`](KanbanColumn::Plan)     | insufficient evidence → re-deliberate / elevate |
/// | [`Terminal`](DominoStep::Terminal) | [`Prune`](KanbanColumn::Prune)   | no bridging term → drop (Libet veto) |
///
/// Returns `None` when `current != Evaluation`: the verdict applies **only** at the
/// Rubicon decision point. The caller must not force a transition elsewhere — the
/// checked `MailboxSoaOwner::try_advance_phase` would (correctly) reject it.
///
/// `Fork` and `Escalate` map to the same *column* (`Plan`) but differ in
/// *side-effect* (Fork → deposit a counterfactual; Escalate → elevate a tier); the
/// caller keeps the originating [`DominoStep`] to dispatch that. This function
/// resolves only the column transition.
#[inline]
pub fn rubicon_transition(step: DominoStep, current: KanbanColumn) -> Option<KanbanColumn> {
    if current != KanbanColumn::Evaluation {
        return None;
    }
    Some(match step {
        DominoStep::Settle => KanbanColumn::Commit,
        DominoStep::Fork | DominoStep::Escalate => KanbanColumn::Plan,
        DominoStep::Terminal => KanbanColumn::Prune,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn evaluation_maps_every_verdict_and_is_dag_legal() {
        let eval = KanbanColumn::Evaluation;
        for (step, want) in [
            (DominoStep::Settle, KanbanColumn::Commit),
            (DominoStep::Fork, KanbanColumn::Plan),
            (DominoStep::Escalate, KanbanColumn::Plan),
            (DominoStep::Terminal, KanbanColumn::Prune),
        ] {
            let to = rubicon_transition(step, eval).expect("every verdict maps from Evaluation");
            assert_eq!(to, want);
            // The transition must be legal in the Rubicon DAG.
            assert!(
                eval.can_transition_to(to),
                "{to:?} must be a legal Evaluation successor",
            );
        }
    }

    #[test]
    fn only_settle_commits() {
        // Anti-wishful at the Rubicon: Settle is the ONLY verdict that crosses to
        // Commit (calcify). Fork/Escalate re-deliberate; Terminal drops.
        assert_eq!(
            rubicon_transition(DominoStep::Settle, KanbanColumn::Evaluation),
            Some(KanbanColumn::Commit),
        );
        for step in [DominoStep::Fork, DominoStep::Escalate, DominoStep::Terminal] {
            assert_ne!(
                rubicon_transition(step, KanbanColumn::Evaluation),
                Some(KanbanColumn::Commit),
                "{step:?} must not commit",
            );
        }
    }

    #[test]
    fn non_evaluation_never_transitions() {
        // The verdict applies only at the Evaluation decision point; from any other
        // column it yields None — the caller must not force an illegal transition
        // (try_advance_phase would reject it).
        for current in [
            KanbanColumn::Planning,
            KanbanColumn::CognitiveWork,
            KanbanColumn::Commit,
            KanbanColumn::Plan,
            KanbanColumn::Prune,
        ] {
            for step in [
                DominoStep::Settle,
                DominoStep::Fork,
                DominoStep::Escalate,
                DominoStep::Terminal,
            ] {
                assert_eq!(
                    rubicon_transition(step, current),
                    None,
                    "{step:?} @ {current:?} must not transition",
                );
            }
        }
    }
}
