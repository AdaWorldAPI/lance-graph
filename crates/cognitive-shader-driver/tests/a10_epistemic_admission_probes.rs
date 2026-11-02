//! A10 probe-first falsifiers F1, F3 and F6, run BEFORE any admission
//! writer exists. Each test pins what the shipped code does today; none of
//! them adds a guard to production code.
//!
//! - F1: is a stamped `Direct × Open` (`raw5 = 0`) distinguishable from an
//!   edge nobody wrote? Inside `CausalEdge64` it is not. Outside it, the alpha
//!   overlay's claim distinguishes them on the hot path.
//! - F3: overlapping vs independent premise roots. The production numeric
//!   revision (`Opcode::Revision.truth_code`) sees four scalars and cannot
//!   tell them apart. `GadamerRevision` can, but only through the
//!   `BasisView` its caller supplies.
//! - F6: maturation of bits 50..52 never alters bits 59..63.
//!
//! A test-local overlap guard (`causal-edge/tests/p_e2_stamp_gate_probe.rs`)
//! is NOT evidence that production revision enforces the rule; F3 measures
//! the production paths only.

use causal_edge::{CausalEdge64, Opcode};
use lance_graph_contract::alpha::{stamp_of, AlphaAllocation, AlphaOverlay};
use lance_graph_contract::canonical_node::{EdgeBlock, NodeGuid, NodeRow, TailVariant};
use lance_graph_contract::epistemic_state5::{
    Certification3, Epi5Gen, EpistemicState5, Topology2,
};
use lance_graph_contract::revision::{
    BasisView, CodebookId, EncounterEvidence, EvidentialEffect, GadamerRevision, GrammarId,
    HorizonId, InterpretiveHorizon, LanguageId, LensId, QuestionId, RevisionPolicy,
};

const EPI_MASK: u64 = 0b1_1111 << 59;

fn direct_open() -> EpistemicState5 {
    EpistemicState5::new(Epi5Gen::V1, Topology2::Direct, Certification3::Open)
}

fn rows(n: u32) -> Vec<NodeRow> {
    let mut rows: Vec<NodeRow> = (0..n)
        .map(|i| NodeRow {
            key: NodeGuid::mint_for(TailVariant::V3, 0x0A0A_0000, 4, 5, 6, 0x33, 1, i + 1),
            edges: EdgeBlock::default(),
            value: [0u8; 480],
        })
        .collect();
    rows.sort_by_key(|r| *r.key.as_bytes());
    rows
}

/// F1, CE64 half: `Direct × Open` is code 0 and W of a fresh edge is 0, so a
/// stamped edge and an unwritten one are the same 64 bits.
#[test]
fn f1_stamped_direct_open_is_bit_identical_to_an_unwritten_edge() {
    assert_eq!(direct_open().raw(), 0, "Direct × Open must be raw5 = 0");
    let unwritten = CausalEdge64(0);
    let stamped = unwritten.with_epistemic_raw5(direct_open().raw());
    assert_eq!(stamped.0, unwritten.0, "the CE64 word cannot carry the stamp");

    // Control: a non-zero code IS visible, so the comparison can fail.
    let supports = EpistemicState5::new(Epi5Gen::V1, Topology2::Direct, Certification3::Supports);
    assert_ne!(unwritten.with_epistemic_raw5(supports.raw()).0, unwritten.0);
}

/// F1, outside CE64: on the hot path the alpha overlay's claim is the
/// written-by fact. A claimed row reads `Some`, an unclaimed one `None`, even
/// though both carry the identical zero code.
#[test]
fn f1_alpha_claim_distinguishes_what_ce64_cannot() {
    let base = rows(8);
    let mut ov = AlphaOverlay::new(AlphaAllocation::over(&base), 1);
    let written = base[2].key;
    let never = base[5].key;
    ov.claim(written, 0).expect("allocated address");

    assert!(ov.get(written).is_some(), "stamped: claimed");
    assert!(ov.get(never).is_none(), "unwritten: not attended, no fallback");
    // The claim stamps the claimed row (`AlphaStamp`: cycle, saccade order,
    // visits); the base row stays all-zero. The stamp, not bits 59..63, is the
    // written-by record.
    let claimed = ov.get(written).expect("claimed");
    assert_ne!(claimed.value, base[2].value, "a claim stamps the row");
    assert_eq!(stamp_of(claimed).visits, 1);
    // Only the claimed row is materialised; a sealed image of this overlay
    // holds row 2 and not row 5. Row existence is the only carrier left once
    // the session-local claim is gone.
    assert_eq!(ov.claimed_len(), 1);
    assert_eq!(ov.rows()[0].key, written);
}

/// F3, production numeric revision: overlapping and independent premise roots
/// produce identical inputs, so they produce identical outputs. Confidence
/// rises in both cases; the overlap is counted twice.
#[test]
fn f3_production_truth_code_cannot_see_premise_overlap() {
    let (f, c) = (200u8, 128u8);
    // "Same root twice" and "two independent roots with the same reading" are
    // the same four scalars.
    let overlapping = Opcode::Revision.truth_code(f, c, f, c);
    let independent = Opcode::Revision.truth_code(f, c, f, c);
    assert_eq!(overlapping, independent);
    assert!(
        overlapping.1 > c,
        "production revision raises c on a duplicated root: c {c} -> {}",
        overlapping.1
    );
}

fn horizon() -> InterpretiveHorizon<(), u64> {
    InterpretiveHorizon {
        id: HorizonId(0),
        awareness: (),
        question: QuestionId(0),
        language: LanguageId(0),
        grammar: GrammarId(0),
        codebook: CodebookId(0),
        lens: LensId(0),
        projected_claims: 0b1,
        independent_roots: 0,
        inherited_roots: 0,
        unresolved_tension: 0,
        revision_index: 0,
    }
}

fn encounter(independent: u64, inherited: u64) -> EncounterEvidence<u64> {
    EncounterEvidence {
        proposed_claims: 0b1,
        independent_roots: independent,
        inherited_roots: inherited,
        resistance: 0,
        contradictions: 0,
        affected_parts: 0,
    }
}

fn ancestry(independent: u64) -> BasisView<u64> {
    BasisView {
        ancestry_independent_roots: independent,
        ancestry_derived_roots: 0,
        ancestor_claims: 0,
        closes_cycle: false,
    }
}

fn effect(enc: &EncounterEvidence<u64>, basis: &BasisView<u64>) -> EvidentialEffect {
    GadamerRevision.revise(&horizon(), enc, basis).evidential_effect
}

/// F3, contract revision: `GadamerRevision` refuses an overlapping root and
/// admits an independent one, given an ancestry that records the first root.
#[test]
fn f3_gadamer_revision_separates_overlap_from_independence() {
    let r1 = 1u64 << 1;
    let r2 = 1u64 << 2;
    assert_eq!(
        effect(&encounter(r1, 0), &ancestry(0)),
        EvidentialEffect::IncreaseEligible,
        "first contact with r1"
    );
    assert_ne!(
        effect(&encounter(r1, 0), &ancestry(r1)),
        EvidentialEffect::IncreaseEligible,
        "r1 again, already in ancestry"
    );
    assert_eq!(
        effect(&encounter(r2, 0), &ancestry(r1)),
        EvidentialEffect::IncreaseEligible,
        "r2 is a genuinely new root"
    );
    // An inherited root never counts as new evidence.
    assert_ne!(
        effect(&encounter(0, r1), &ancestry(0)),
        EvidentialEffect::IncreaseEligible
    );
}

/// F3, the gap: the overlap rule is enforced only through the caller's
/// `BasisView`. A caller that omits the ancestry gets the duplicate admitted.
#[test]
fn f3_overlap_enforcement_depends_on_the_supplied_ancestry() {
    let r1 = 1u64 << 1;
    assert_eq!(
        effect(&encounter(r1, 0), &ancestry(0)),
        EvidentialEffect::IncreaseEligible,
        "a repeated root with an empty ancestry is admitted again"
    );
    // The same encounter can also list a root as both independent and
    // inherited; the policy reads it as new because only ancestry is
    // subtracted.
    assert_eq!(
        effect(&encounter(r1, r1), &ancestry(0)),
        EvidentialEffect::IncreaseEligible
    );
}

/// F6: the maturation writers move bits 50..52 and never bits 59..63.
#[test]
fn f6_maturation_never_alters_epistemic_state() {
    let codes = [
        direct_open(),
        EpistemicState5::new(Epi5Gen::V1, Topology2::IndirectKnown, Certification3::Supports),
        EpistemicState5::new(Epi5Gen::V1, Topology2::Unknown, Certification3::Causes),
    ];
    for state in codes {
        let seed = CausalEdge64(0x0123_4567_89AB_CDEF & !EPI_MASK).with_epistemic_raw5(state.raw());

        for level in 0..=9u8 {
            let matured = seed.with_novelty_wisdom(level);
            assert_eq!(matured.epistemic_raw5(), state.raw(), "level {level}");
            assert_eq!(
                (matured.0 ^ seed.0) & !(0b111 << 50),
                0,
                "with_novelty_wisdom moved a bit outside 50..52"
            );
        }

        let mut learned = seed;
        for (i, c) in [10u8, 120, 200, 240, 250, 255].into_iter().enumerate() {
            let mut obs = CausalEdge64(seed.0 ^ ((i as u64 + 1) << 8));
            obs.set_frequency_u8(180);
            obs.set_confidence_u8(c);
            learned.learn(obs, i as u16);
            assert_eq!(
                learned.epistemic_raw5(),
                state.raw(),
                "learn() changed bits 59..63 at step {i}"
            );
        }
    }
}
