//! Gadamer decides whether revision is warranted; the CE64 register defines how
//! admitted evidence is numerically revised.
//!
//! `GadamerRevision` (lance-graph-contract) is a reasoning-level gate.
//! `CausalEdge64::revision` (causal-edge) is arithmetic. This probe is the
//! orchestration between them, kept here, above both: the register never sees
//! a horizon, and the gate never computes a truth value.
//!
//! The claim pinned: repeated echo or closed-cycle encounters cannot mint
//! confidence, because the gate never admits them to the register. The
//! register itself would happily pool the same evidence again; that is why
//! the gate has to sit above it.

use causal_edge::CausalEdge64;
use lance_graph_contract::revision::{
    BasisView, CodebookId, EncounterEvidence, EvidentialEffect, GadamerRevision, GrammarId,
    HorizonId, InterpretiveHorizon, LanguageId, LensId, QuestionId, RevisionKind, RevisionPolicy,
};

fn horizon(roots: u64) -> InterpretiveHorizon<(), u64> {
    InterpretiveHorizon {
        id: HorizonId(1),
        awareness: (),
        question: QuestionId(1),
        language: LanguageId(1),
        grammar: GrammarId(1),
        codebook: CodebookId(1),
        lens: LensId(1),
        projected_claims: 0b1,
        independent_roots: roots,
        inherited_roots: 0,
        unresolved_tension: 0,
        revision_index: 0,
    }
}

fn encounter(roots: u64) -> EncounterEvidence<u64> {
    EncounterEvidence {
        proposed_claims: 0b1, // the same working whole
        independent_roots: roots,
        inherited_roots: 0,
        resistance: 0,
        contradictions: 0,
        affected_parts: 0,
    }
}

fn edge(f: u8, c: u8) -> CausalEdge64 {
    let mut e = CausalEdge64(0);
    e.set_frequency_u8(f);
    e.set_confidence_u8(c);
    e
}

/// The orchestration: revise only what the gate admits.
fn orchestrate(
    belief: CausalEdge64,
    evidence: CausalEdge64,
    effect: EvidentialEffect,
) -> CausalEdge64 {
    match effect {
        EvidentialEffect::IncreaseEligible => belief.revision(evidence),
        EvidentialEffect::NoIncrease | EvidentialEffect::Suspend => belief,
    }
}

/// Run `rounds` encounters through gate + register; the encounter for round
/// `k` contacts `roots_of(k)` and the ancestry already holds `known`.
fn run(
    rounds: u32,
    cycle: bool,
    roots_of: impl Fn(u32) -> u64,
) -> (CausalEdge64, Vec<RevisionKind>) {
    let mut belief = edge(200, 100);
    let evidence = edge(200, 100);
    let mut known = 0b1u64;
    let mut kinds = Vec::new();
    for k in 0..rounds {
        let roots = roots_of(k);
        let ancestry = BasisView {
            ancestry_independent_roots: known,
            ancestry_derived_roots: 0,
            ancestor_claims: 0b1,
            closes_cycle: cycle,
        };
        let delta = GadamerRevision.revise(&horizon(known), &encounter(roots), &ancestry);
        kinds.push(delta.kind);
        belief = orchestrate(belief, evidence, delta.evidential_effect);
        known |= roots;
    }
    (belief, kinds)
}

/// Echo: the same root re-read ten times. The gate never admits it, so the
/// register never runs and confidence does not move.
#[test]
fn echo_cannot_mint_confidence() {
    let (belief, kinds) = run(10, false, |_| 0b1);
    assert!(kinds.iter().all(|k| *k == RevisionKind::Echo), "{kinds:?}");
    assert_eq!(belief.confidence_u8(), 100);
}

/// Closed cycle: the candidate depends on itself.
#[test]
fn a_closed_cycle_cannot_mint_confidence() {
    let (belief, kinds) = run(10, true, |_| 0b1);
    assert!(
        kinds.iter().all(|k| *k == RevisionKind::ClosedCycle),
        "{kinds:?}"
    );
    assert_eq!(belief.confidence_u8(), 100);
}

/// Can-fire arm: a genuinely new independent root each round is admitted and
/// revised, so confidence rises.
#[test]
fn new_independent_roots_are_revised() {
    let (belief, kinds) = run(10, false, |k| 1u64 << (k + 1));
    assert!(
        kinds
            .iter()
            .all(|k| *k == RevisionKind::IndependentConfirmation),
        "{kinds:?}"
    );
    assert!(belief.confidence_u8() > 200, "{}", belief.confidence_u8());
}

/// Why the gate must sit above the register: the arithmetic alone pools
/// whatever it is given, echo included.
#[test]
fn the_register_alone_would_inflate_an_echo() {
    let mut belief = edge(200, 100);
    for _ in 0..10 {
        belief = belief.revision(edge(200, 100));
    }
    assert!(belief.confidence_u8() > 200);
}
