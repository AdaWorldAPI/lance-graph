//! The existing contract signals the plan's squash rule and write path rely
//! on, run on the D-TGB-0 fixtures. Each signal gets a can-fire case on an
//! anomaly fixture and a can-stay-silent case on an unremarkable one, so a
//! signal that fires on everything fails as surely as one that never fires.

use lance_graph_contract::alpha::{AlphaAllocation, AlphaMask, AlphaOverlay};
use lance_graph_contract::canonical_node::{EdgeBlock, NodeGuid, NodeRow, TailVariant};
use lance_graph_contract::kanban::KanbanColumn;
use lance_graph_contract::mul::{FlowState, GateDecision, TrustTexture};
use lance_graph_contract::revision::{
    BasisView, CodebookId, CounterfactualVerdict, EncounterEvidence, EvidentialEffect,
    GadamerRevision, GrammarId, HorizonId, InterpretiveHorizon, LanguageId, LensId, QuestionId,
    RevisionKind, RevisionPolicy, RevisionVerdict,
};
use thought_git_probe::fixtures::{
    Outcome, REGRESSION_SCANPATH, ROWS, SMOOTH_SCANPATH, SURPRISE_ATTENDED, SURPRISE_EXPECTED,
    rubicon_outcomes,
};

fn spine() -> Vec<NodeRow> {
    (0..ROWS)
        .map(|r| NodeRow {
            key: NodeGuid::mint_for(TailVariant::V3, 0x0B0B_0000, 1, 2, 3, 4, 0x11, r + 1),
            edges: EdgeBlock::default(),
            value: [0u8; 480],
        })
        .collect()
}

fn overlay_regressions(path: &[u32]) -> Vec<u16> {
    let base = spine();
    let alloc = AlphaAllocation::over(&base);
    let mut o = AlphaOverlay::new(alloc, 1);
    for &r in path {
        o.claim(base[r as usize].key, 0).expect("allocated");
    }
    o.regressions().into_iter().map(|(_, n)| n).collect()
}

/// Regression signal (`visits > 1`) fires where attention returns.
#[test]
fn regression_fires_on_a_return() {
    assert_eq!(overlay_regressions(&REGRESSION_SCANPATH), vec![2]);
}

/// ...and stays silent on a path without one.
#[test]
fn regression_is_silent_without_a_return() {
    assert!(overlay_regressions(&SMOOTH_SCANPATH).is_empty());
}

fn mask(base: &[NodeRow], rows: &[u32]) -> AlphaMask {
    AlphaAllocation::over(base)
        .mask_of(rows.iter().map(|&r| base[r as usize].key))
        .expect("allocated")
}

/// Cold/hot surprise (`expected.xor(attended)`) fires when they differ...
#[test]
fn surprise_fires_when_expected_and_attended_differ() {
    let base = spine();
    let s = mask(&base, &SURPRISE_EXPECTED).xor(&mask(&base, &SURPRISE_ATTENDED));
    assert_eq!(s.materialize_ordinals(), vec![3, 4]);
}

/// ...and is empty when the attended set is what was expected. The attended
/// mask here comes from a real overlay, not from the expected list.
#[test]
fn surprise_is_silent_when_attended_matches_expected() {
    let base = spine();
    let alloc = AlphaAllocation::over(&base);
    let mut o = AlphaOverlay::new(alloc, 1);
    for &r in &SURPRISE_EXPECTED {
        o.claim(base[r as usize].key, 0).expect("allocated");
    }
    assert!(
        mask(&base, &SURPRISE_EXPECTED)
            .xor(&o.attended_mask())
            .is_empty()
    );
}

/// The named route each outcome takes from `Planning`.
fn route(outcome: Outcome) -> Vec<KanbanColumn> {
    use KanbanColumn::*;
    let mut path = vec![Planning];
    let mut at = Planning;
    let step = |c: KanbanColumn, f: fn(KanbanColumn) -> Option<KanbanColumn>| f(c).expect("legal");
    if outcome == Outcome::Veto {
        path.push(step(at, KanbanColumn::veto));
        return path;
    }
    at = step(at, KanbanColumn::advance);
    path.push(at);
    at = step(at, KanbanColumn::advance);
    path.push(at);
    let last = match outcome {
        Outcome::Commit => step(at, KanbanColumn::advance),
        Outcome::Plan => step(at, KanbanColumn::revise),
        Outcome::Prune => step(at, KanbanColumn::veto),
        Outcome::Veto => unreachable!(),
    };
    path.push(last);
    path
}

/// Every fixture outcome is reached through the named kanban primitives, and
/// every step is a legal transition.
#[test]
fn every_outcome_is_a_legal_kanban_route() {
    use KanbanColumn::*;
    let ends: Vec<KanbanColumn> = rubicon_outcomes()
        .iter()
        .map(|t| {
            let p = route(t.outcome);
            for w in p.windows(2) {
                assert!(w[0].can_transition_to(w[1]), "{:?} -> {:?}", w[0], w[1]);
            }
            *p.last().expect("non-empty")
        })
        .collect();
    assert_eq!(ends, vec![Commit, Plan, Prune, Prune]);
}

/// The MUL gate: `Block` at `Evaluation` vetoes (nothing lands), `Hold` stays
/// put, `Flow` advances to `Commit`.
#[test]
fn the_mul_gate_decides_whether_the_write_crosses() {
    use KanbanColumn::*;
    let hold = GateDecision::Hold {
        texture: TrustTexture::Uncertain,
        flow: FlowState::Transition,
    };
    let block = GateDecision::Block {
        texture: TrustTexture::Overconfident,
        flow: FlowState::Anxiety,
    };
    assert_eq!(
        Evaluation.advance_on_gate(&GateDecision::Flow),
        Some(Commit)
    );
    assert_eq!(Evaluation.advance_on_gate(&hold), None);
    assert_eq!(Evaluation.advance_on_gate(&block), Some(Prune));
}

type Horizon = InterpretiveHorizon<(), u64>;

fn horizon(claims: u64, roots: u64) -> Horizon {
    InterpretiveHorizon {
        id: HorizonId(1),
        awareness: (),
        question: QuestionId(1),
        language: LanguageId(1),
        grammar: GrammarId(1),
        codebook: CodebookId(1),
        lens: LensId(1),
        projected_claims: claims,
        independent_roots: roots,
        inherited_roots: 0,
        unresolved_tension: 0,
        revision_index: 0,
    }
}

fn ancestry(h: &Horizon) -> BasisView<u64> {
    BasisView {
        ancestry_independent_roots: h.independent_roots,
        ancestry_derived_roots: h.inherited_roots,
        ancestor_claims: h.projected_claims,
        closes_cycle: false,
    }
}

fn encounter(claims: u64, roots: u64, inherited: u64, contra: u64) -> EncounterEvidence<u64> {
    EncounterEvidence {
        proposed_claims: claims,
        independent_roots: roots,
        inherited_roots: inherited,
        resistance: 0,
        contradictions: contra,
        affected_parts: 0,
    }
}

/// Replaying a thought's own material (no root outside its ancestry) earns
/// nothing and routes back to `Plan`, never to `Commit`.
#[test]
fn replayed_material_cannot_raise_evidence() {
    let prior = horizon(0b0011, 0b0001);
    let replay = encounter(0b0011, 0b0001, 0b0010, 0);
    let d = GadamerRevision.revise(&prior, &replay, &ancestry(&prior));
    assert_eq!(d.kind, RevisionKind::Echo);
    assert_eq!(d.evidential_effect, EvidentialEffect::NoIncrease);
    let verdict = RevisionVerdict {
        effect: d.evidential_effect,
        counterfactual: CounterfactualVerdict::Necessary,
    };
    assert_eq!(
        KanbanColumn::Evaluation.advance_on_revision(verdict),
        Some(KanbanColumn::Plan)
    );
}

/// Can-fire twin: a genuinely new root that survived the counterfactual
/// reaches `Commit`.
#[test]
fn a_new_root_that_survived_attack_reaches_commit() {
    let prior = horizon(0b0011, 0b0001);
    let fresh = encounter(0b0111, 0b0101, 0, 0);
    let d = GadamerRevision.revise(&prior, &fresh, &ancestry(&prior));
    assert_eq!(d.evidential_effect, EvidentialEffect::IncreaseEligible);
    let verdict = RevisionVerdict {
        effect: d.evidential_effect,
        counterfactual: CounterfactualVerdict::Necessary,
    };
    assert_eq!(
        KanbanColumn::Evaluation.advance_on_revision(verdict),
        Some(KanbanColumn::Commit)
    );
}

/// Landing the same thought twice: the second revision, against the horizon
/// the first one produced, has no new root.
#[test]
fn the_second_landing_of_the_same_thought_earns_nothing() {
    let prior = horizon(0b0011, 0b0001);
    let e = encounter(0b0111, 0b0101, 0, 0);
    let first = GadamerRevision.revise(&prior, &e, &ancestry(&prior));
    let second = GadamerRevision.revise(&first.resulting, &e, &ancestry(&first.resulting));
    assert_eq!(first.evidential_effect, EvidentialEffect::IncreaseEligible);
    assert_ne!(second.evidential_effect, EvidentialEffect::IncreaseEligible);
}

/// A contradiction without a new root is preserved and suspends: the cycle
/// holds instead of landing or pruning.
#[test]
fn a_preserved_contradiction_holds_the_cycle() {
    let prior = horizon(0b0011, 0b0001);
    let e = encounter(0b0111, 0b0001, 0, 0b1000);
    let d = GadamerRevision.revise(&prior, &e, &ancestry(&prior));
    assert_eq!(d.kind, RevisionKind::ContradictionPreserved);
    assert_eq!(d.evidential_effect, EvidentialEffect::Suspend);
    assert_eq!(
        KanbanColumn::Evaluation
            .advance_on_revision(RevisionVerdict::unadjudicated(d.evidential_effect)),
        None
    );
    assert_eq!(
        d.resulting.unresolved_tension & 0b1000,
        0b1000,
        "tension kept"
    );
}
