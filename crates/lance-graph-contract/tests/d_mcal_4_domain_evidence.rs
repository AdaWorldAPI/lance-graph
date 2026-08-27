//! D-MCAL-4 — a consent veto and an evidence contradiction, expressed as
//! domain evidence driving a domain execution gate, with **no MUL ground**.
//!
//! Falsifies F-MUL-1 and F-MUL-2 from
//! `.claude/plans/mul-calibration-not-verdict-v1.md` §5.
//!
//! # The red state this replaces
//!
//! Measured on `main` before this change
//! (`.claude/plans/mul-consumer-census-v1.md` §3), the only route into the
//! kanban phase DAG was `advance_on_gate(&GateDecision)`, whose `Block` and
//! `Hold` variants require a `TrustTexture` **and** a `FlowState`. Two external
//! producers reached it with neither axis in their data flow and invented both:
//!
//! - `ada-rs::contract_impls::gate_check` — a consent veto returning `Block`.
//! - `medcare-first-thought` (4 sites) — an evidence contradiction returning
//!   `Block { texture: Uncertain, flow: Anxiety }`, whose own source comment
//!   records that the payload is "descriptive, not behavior-affecting".
//!
//! # What makes this a real falsifier and not a restatement
//!
//! Each case asserts three things: the domain route is correct; **no
//! `GateDecision` is constructed anywhere in the domain path**; and the route
//! is *identical* to what the fabricating path produced. The third is the load-
//! bearing one — it shows behaviour is preserved, so removing the fabrication
//! costs nothing (F-MUL-5's premise).
//!
//! Each case also exercises all three outcomes (advance / stay / veto) from one
//! domain type, so a test that merely proved "the veto arm works" cannot pass
//! by accident.

use lance_graph_contract::kanban::KanbanColumn;
use lance_graph_contract::mul::{FlowState, GateDecision, TrustTexture};

// ═══════════════════════════════════════════════════════════════════════════
// F-MUL-1 — a consent veto is domain evidence
// ═══════════════════════════════════════════════════════════════════════════

/// The shape ada-rs actually has (`sovereignty::freedom::ConsentLevel`).
/// Consent is a *sovereignty* fact. It says nothing about how well calibrated
/// the system's confidence is, and nothing about challenge-versus-skill.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ConsentLevel {
    Full,
    Provisional,
    Veto,
}

impl ConsentLevel {
    fn allows_action(self) -> bool {
        matches!(self, ConsentLevel::Full)
    }
}

/// The domain's own execution gate: consent in, phase transition out.
/// Note what is absent — no `TrustTexture`, no `FlowState`, no `GateDecision`.
fn route_on_consent(phase: KanbanColumn, consent: ConsentLevel) -> Option<KanbanColumn> {
    match consent {
        ConsentLevel::Veto => phase.veto(),
        ConsentLevel::Provisional => None, // gather standing, stay put
        c if c.allows_action() => phase.advance(),
        _ => None,
    }
}

#[test]
fn f_mul_1_consent_veto_routes_without_constructing_mul_ground() {
    // The veto edge exists pre-Rubicon and post-actional …
    assert_eq!(
        route_on_consent(KanbanColumn::Planning, ConsentLevel::Veto),
        Some(KanbanColumn::Prune)
    );
    assert_eq!(
        route_on_consent(KanbanColumn::Evaluation, ConsentLevel::Veto),
        Some(KanbanColumn::Prune)
    );
    // … and not mid-CognitiveWork, where the DAG has no veto edge.
    assert_eq!(
        route_on_consent(KanbanColumn::CognitiveWork, ConsentLevel::Veto),
        None
    );
}

#[test]
fn f_mul_1_the_same_domain_type_also_advances_and_stays() {
    // Anti-vacuity: a consent gate that only ever vetoed would be a constant.
    assert_eq!(
        route_on_consent(KanbanColumn::Planning, ConsentLevel::Full),
        Some(KanbanColumn::CognitiveWork)
    );
    assert_eq!(
        route_on_consent(KanbanColumn::Planning, ConsentLevel::Provisional),
        None
    );
    // Three distinct outcomes from one axis, at one column.
    let outcomes = [
        route_on_consent(KanbanColumn::Planning, ConsentLevel::Full),
        route_on_consent(KanbanColumn::Planning, ConsentLevel::Provisional),
        route_on_consent(KanbanColumn::Planning, ConsentLevel::Veto),
    ];
    assert_eq!(outcomes[0], Some(KanbanColumn::CognitiveWork));
    assert_eq!(outcomes[1], None);
    assert_eq!(outcomes[2], Some(KanbanColumn::Prune));
}

#[test]
fn f_mul_1_domain_route_equals_the_fabricating_route_it_replaces() {
    // What ada-rs does today: invent a texture and a flow it never measured,
    // purely to satisfy the constructor. Both values below are arbitrary — the
    // point of the test is that they are, and that nothing depends on them.
    let fabricated = GateDecision::Block {
        texture: TrustTexture::Uncertain,
        flow: FlowState::Anxiety,
    };
    for phase in [
        KanbanColumn::Planning,
        KanbanColumn::CognitiveWork,
        KanbanColumn::Evaluation,
        KanbanColumn::Commit,
        KanbanColumn::Plan,
        KanbanColumn::Prune,
    ] {
        assert_eq!(
            route_on_consent(phase, ConsentLevel::Veto),
            phase.advance_on_gate(&fabricated),
            "domain veto diverged from the fabricating path at {phase:?}"
        );
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// F-MUL-2 — an evidence contradiction is domain evidence
// ═══════════════════════════════════════════════════════════════════════════

/// The shape `medcare-first-thought::patient_outcome` actually has: a NARS
/// truth expectation plus the cardinality of the supporting and contradicting
/// evidence sets. Nothing here is a calibration or a flow reading.
#[derive(Debug, Clone, Copy)]
struct EvidenceReading {
    expectation: f64,
    supporting: usize,
    contradicting: usize,
}

impl EvidenceReading {
    fn observed(&self) -> usize {
        self.supporting + self.contradicting
    }
    /// Real evidence exists and disagrees with the hypothesis.
    fn is_contradicted(&self) -> bool {
        self.observed() > 0 && self.expectation < 0.5
    }
    /// Real evidence exists and supports the hypothesis.
    fn is_supported(&self) -> bool {
        self.observed() > 0 && self.expectation > 0.5
    }
}

/// The domain's own execution gate: evidence in, phase transition out.
/// A non-finite expectation refuses to route at all, rather than silently
/// falling through a comparison that is false for NaN.
fn route_on_evidence(phase: KanbanColumn, ev: &EvidenceReading) -> Option<KanbanColumn> {
    if !ev.expectation.is_finite() {
        return None;
    }
    if ev.is_contradicted() {
        phase.veto()
    } else if ev.is_supported() {
        phase.advance()
    } else {
        None // unobserved or equivocal — gather more
    }
}

#[test]
fn f_mul_2_evidence_contradiction_routes_without_constructing_mul_ground() {
    let contradicted = EvidenceReading {
        expectation: 0.2,
        supporting: 1,
        contradicting: 4,
    };
    assert!(contradicted.is_contradicted());
    assert_eq!(
        route_on_evidence(KanbanColumn::Evaluation, &contradicted),
        Some(KanbanColumn::Prune)
    );
    assert_eq!(
        route_on_evidence(KanbanColumn::CognitiveWork, &contradicted),
        None
    );
}

#[test]
fn f_mul_2_the_same_domain_type_also_advances_and_stays() {
    // Anti-vacuity, and the three cases are genuinely distinct readings.
    let supported = EvidenceReading {
        expectation: 0.9,
        supporting: 6,
        contradicting: 0,
    };
    let unobserved = EvidenceReading {
        expectation: 0.5,
        supporting: 0,
        contradicting: 0,
    };
    let contradicted = EvidenceReading {
        expectation: 0.1,
        supporting: 0,
        contradicting: 3,
    };

    assert_eq!(
        route_on_evidence(KanbanColumn::Planning, &supported),
        Some(KanbanColumn::CognitiveWork)
    );
    assert_eq!(route_on_evidence(KanbanColumn::Planning, &unobserved), None);
    assert_eq!(
        route_on_evidence(KanbanColumn::Planning, &contradicted),
        Some(KanbanColumn::Prune)
    );
}

#[test]
fn f_mul_2_non_finite_expectation_refuses_to_route() {
    for bad in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        let ev = EvidenceReading {
            expectation: bad,
            supporting: 3,
            contradicting: 1,
        };
        assert_eq!(
            route_on_evidence(KanbanColumn::Planning, &ev),
            None,
            "non-finite expectation {bad} produced a transition"
        );
    }
}

#[test]
fn f_mul_2_domain_route_equals_the_fabricating_route_it_replaces() {
    // MedCare's four sites, reduced to their two fabricated pairs.
    let fab_block = GateDecision::Block {
        texture: TrustTexture::Uncertain,
        flow: FlowState::Anxiety,
    };
    let fab_hold = GateDecision::Hold {
        texture: TrustTexture::Calibrated,
        flow: FlowState::Boredom,
    };
    let contradicted = EvidenceReading {
        expectation: 0.2,
        supporting: 0,
        contradicting: 5,
    };
    let equivocal = EvidenceReading {
        expectation: 0.5,
        supporting: 0,
        contradicting: 0,
    };
    let supported = EvidenceReading {
        expectation: 0.8,
        supporting: 4,
        contradicting: 1,
    };

    for phase in [
        KanbanColumn::Planning,
        KanbanColumn::CognitiveWork,
        KanbanColumn::Evaluation,
        KanbanColumn::Commit,
        KanbanColumn::Plan,
        KanbanColumn::Prune,
    ] {
        assert_eq!(
            route_on_evidence(phase, &contradicted),
            phase.advance_on_gate(&fab_block),
            "contradiction route diverged at {phase:?}"
        );
        assert_eq!(
            route_on_evidence(phase, &equivocal),
            phase.advance_on_gate(&fab_hold),
            "equivocal route diverged at {phase:?}"
        );
        assert_eq!(
            route_on_evidence(phase, &supported),
            phase.advance_on_gate(&GateDecision::Flow),
            "support route diverged at {phase:?}"
        );
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// The primitives themselves
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn advance_and_veto_never_leave_the_dag() {
    for phase in [
        KanbanColumn::Planning,
        KanbanColumn::CognitiveWork,
        KanbanColumn::Evaluation,
        KanbanColumn::Commit,
        KanbanColumn::Plan,
        KanbanColumn::Prune,
    ] {
        if let Some(to) = phase.advance() {
            assert!(
                phase.can_transition_to(to),
                "advance() left the DAG: {phase:?} -> {to:?}"
            );
            assert_ne!(to, KanbanColumn::Prune, "advance() must never veto");
        }
        if let Some(to) = phase.veto() {
            assert!(
                phase.can_transition_to(to),
                "veto() left the DAG: {phase:?} -> {to:?}"
            );
            assert_eq!(to, KanbanColumn::Prune);
        }
    }
}

#[test]
fn advance_and_veto_are_distinct_where_both_are_legal() {
    // Anti-vacuity for the pair: at a column offering both edges they must not
    // collapse to the same answer, or one of them is decoration.
    for phase in [KanbanColumn::Planning, KanbanColumn::Evaluation] {
        let a = phase.advance();
        let v = phase.veto();
        assert!(a.is_some() && v.is_some(), "{phase:?} should offer both");
        assert_ne!(a, v, "advance() and veto() collapsed at {phase:?}");
    }
    // …and both are absent where the DAG offers neither.
    assert_eq!(KanbanColumn::Prune.advance(), None);
    assert_eq!(KanbanColumn::Prune.veto(), None);
}
