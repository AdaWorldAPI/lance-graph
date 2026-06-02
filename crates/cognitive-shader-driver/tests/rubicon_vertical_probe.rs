//! Probe: the FSM → Rubicon → free-will/MUL vertical actually composes.
//!
//! Validates the claim written in
//! `.claude/knowledge/orchestration-boundary-v1.md` § "What the loop MEANS":
//! the object-level NARS verdict (`route_against → DominoStep`) plus the
//! meta-level MUL self-competence check (`MulAssessment::is_unskilled_overconfident`,
//! mul.rs:384 "used by the gate as a veto hint") compose into a legal
//! `KanbanColumn` (Rubicon) transition — AND the meta-veto OVERRIDES an
//! object-level `Settle` (confident-but-incompetent → NOT `Commit`; free will
//! = Libet's "free won't").
//!
//! This is a probe, not a feature: it composes three EXISTING primitives
//! (`causal_edge::route_against`, `lance_graph_contract::mul::MulAssessment`,
//! `lance_graph_contract::kanban::KanbanColumn`) with the resolver→Rubicon
//! adapter kept INLINE (no new production type). Offline — no surreal, no
//! `MailboxSoaOwner`. It is `route_against`'s first cross-crate caller and the
//! first place the documented vertical is exercised end to end.

use causal_edge::edge::InferenceType;
use causal_edge::{CausalEdge64, CausalMask, DominoStep, PlasticityState};
use lance_graph_contract::kanban::KanbanColumn;
use lance_graph_contract::mul::{MulAssessment, SituationInput};

/// Build an SPO edge with truth `(f, c)` as u8 — mirrors the causal-edge test
/// helper (`syllogism.rs::tests::edge`): Pearl mask SPO, neutral direction,
/// deduction stamp, all-hot.
fn edge(s: u8, p: u8, o: u8, f: u8, c: u8) -> CausalEdge64 {
    #[allow(deprecated)]
    CausalEdge64::pack(
        s,
        p,
        o,
        f,
        c,
        CausalMask::SPO,
        0,
        InferenceType::Deduction,
        PlasticityState::ALL_HOT,
        0,
    )
}

/// The composition under test: `(object-level DominoStep, meta-level veto)` →
/// the Rubicon transition out of `Evaluation`. This is exactly the mapping the
/// boundary doc claims (`Settle→Commit`, `Escalate|Fork→Plan`, `Terminal→Prune`,
/// with the meta-veto forcing `Prune`). Kept inline — a probe, not a minted type.
fn rubicon_target(step: DominoStep, meta_veto: bool) -> KanbanColumn {
    // Meta-veto (Libet "free won't") overrides the object-level push: a
    // confident-but-incompetent verdict never crosses the Rubicon to Commit.
    if meta_veto {
        return KanbanColumn::Prune;
    }
    match step {
        // converged + calibrated → cross the Rubicon, calcify
        DominoStep::Settle => KanbanColumn::Commit,
        // unsure / contradiction → re-deliberate
        DominoStep::Escalate | DominoStep::Fork => KanbanColumn::Plan,
        // no bridging term, dead chain → drop
        DominoStep::Terminal => KanbanColumn::Prune,
    }
}

/// A well-calibrated expert: felt ≈ demonstrated competence → no veto.
fn calibrated() -> SituationInput {
    SituationInput {
        felt_competence: 0.8,
        demonstrated_competence: 0.8,
        ..SituationInput::default()
    }
}

/// Mount-Stupid: feels highly competent, isn't (felt ≫ demonstrated) → the
/// veto fires (`DkPosition::MountStupid`).
fn overconfident() -> SituationInput {
    SituationInput {
        felt_competence: 0.9,
        demonstrated_competence: 0.2,
        ..SituationInput::default()
    }
}

#[test]
fn settle_plus_calibrated_crosses_to_commit() {
    // Object level: a consistent, confident chain (o1==s2, frequencies agree,
    // both confident) → Settle.
    let step = edge(10, 1, 20, 200, 200).route_against(edge(20, 2, 30, 200, 200));
    assert_eq!(step, DominoStep::Settle);

    // Meta level: a calibrated expert → no veto.
    let mul = MulAssessment::compute(&calibrated());
    assert!(!mul.is_unskilled_overconfident());

    // Vertical: Settle + no-veto → Commit, and it is a legal Rubicon edge.
    let to = rubicon_target(step, mul.is_unskilled_overconfident());
    assert_eq!(to, KanbanColumn::Commit);
    assert!(KanbanColumn::Evaluation.can_transition_to(to));
    assert!(to.is_absorbing()); // Commit calcifies to the cold path
}

#[test]
fn settle_but_overconfident_is_vetoed_away_from_commit() {
    // THE load-bearing claim of the vertical: the object level says "go", the
    // meta level says "you don't actually know" — and the meta-veto wins.
    let step = edge(10, 1, 20, 200, 200).route_against(edge(20, 2, 30, 200, 200));
    assert_eq!(step, DominoStep::Settle); // object level: commit-ward

    let mul = MulAssessment::compute(&overconfident());
    assert!(mul.is_unskilled_overconfident()); // meta level: unskilled-overconfident

    // Free will = the veto: an object-level Settle must NOT reach Commit when
    // the meta-layer flags overconfidence. This is the doctrine's whole point.
    let to = rubicon_target(step, mul.is_unskilled_overconfident());
    assert_ne!(
        to,
        KanbanColumn::Commit,
        "the MUL meta-veto must override an object-level Settle"
    );
    assert_eq!(to, KanbanColumn::Prune);
    assert!(KanbanColumn::Evaluation.can_transition_to(to));
}

#[test]
fn escalate_re_deliberates_to_plan() {
    // Object level: a contradiction at medium confidence (between UNCERTAIN and
    // CONFIDENT) → Escalate, never Settle (the anti-wishful guard).
    let step = edge(10, 1, 20, 230, 130).route_against(edge(20, 2, 30, 20, 140));
    assert_eq!(step, DominoStep::Escalate);

    let to = rubicon_target(step, /* meta_veto */ false);
    assert_eq!(to, KanbanColumn::Plan); // re-deliberate, not commit
    assert!(KanbanColumn::Evaluation.can_transition_to(to));
    assert!(!to.is_absorbing()); // Plan loops back to Planning, carrying the witness
}

#[test]
fn terminal_chain_is_pruned() {
    // Object level: no bridging middle term → Terminal → drop.
    let step = edge(10, 1, 20, 200, 200).route_against(edge(30, 2, 40, 200, 200));
    assert_eq!(step, DominoStep::Terminal);

    let to = rubicon_target(step, false);
    assert_eq!(to, KanbanColumn::Prune);
    assert!(KanbanColumn::Evaluation.can_transition_to(to));
}
