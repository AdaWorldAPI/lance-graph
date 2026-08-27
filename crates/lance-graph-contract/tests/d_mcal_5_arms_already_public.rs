//! D-MCAL-5 — the planner's three MUL arms, reached from the contract alone.
//!
//! The deliverable asks: *if a public MUL output is still needed, derive it
//! from the planner's `Proceed/Sandbox/Compass` — never invent a fresh enum.*
//!
//! Measured answer: **no new type is needed and no promotion is needed.** Two
//! of the three arms are already public contract surface; the third has a ruled
//! carrier (T10: `Sandbox := Counterfactual + Revision`) that is declared but
//! not yet implemented.
//!
//! This file makes each claim falsifiable:
//!
//! - the two live arms are **constructed here from `lance_graph_contract`
//!   alone**, with no planner dependency — if either stopped being reachable,
//!   this file would not compile;
//! - each live arm is checked to actually *vary*, so "there is a carrier" is
//!   not satisfied by a constant;
//! - the unimplemented arm is pinned with `#[should_panic]`, so the day the
//!   scaffold lands this test fails and forces the D-MCAL-5 table above it to
//!   be corrected rather than silently going stale.

use lance_graph_contract::counterfactual::{CounterfactualMailbox, SplitPoles};
use lance_graph_contract::mul::{CompassDecision, CompassResult, MulAssessment, SituationInput};

// ═══════════════════════════════════════════════════════════════════════════
// Arm 1 — `Proceed { free_will_modifier }`
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn proceed_arm_is_a_public_field_on_the_assessment() {
    let mul = MulAssessment::compute(&SituationInput::default());
    // The planner's `Proceed` payload IS this field. Reachable, typed, public.
    let _: f64 = mul.free_will_modifier;
    assert!(
        mul.free_will_modifier.is_finite(),
        "free_will_modifier must be a real number to carry the Proceed arm"
    );
}

#[test]
fn proceed_arm_actually_varies_so_the_carrier_is_not_a_constant() {
    // Anti-vacuity: a "carrier" that returns the same number for every input
    // carries nothing. A well-calibrated competent situation and a
    // Mount-Stupid one must not produce the same free-will modifier.
    let calibrated = MulAssessment::compute(&SituationInput {
        felt_competence: 0.8,
        demonstrated_competence: 0.8,
        calibration_accuracy: 0.95,
        source_reliability: 0.9,
        environment_stability: 0.9,
        ..SituationInput::default()
    });
    let mount_stupid = MulAssessment::compute(&SituationInput {
        felt_competence: 0.95,
        demonstrated_competence: 0.10,
        calibration_accuracy: 0.2,
        ..SituationInput::default()
    });
    assert_ne!(
        calibrated.free_will_modifier, mount_stupid.free_will_modifier,
        "free_will_modifier did not move between two very different situations"
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// Arm 2 — `Compass`
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn compass_arm_is_a_public_contract_type() {
    // The planner's `Compass` arm carries no payload; the contract's compass
    // surface carries a score plus the two-way decision. Strictly richer, and
    // already the declared return of `MulProvider::compass`.
    let go = CompassResult {
        score: 0.81,
        decision: CompassDecision::GoMeta,
    };
    let stay = CompassResult {
        score: 0.12,
        decision: CompassDecision::StaySurface,
    };
    assert_eq!(go.decision, CompassDecision::GoMeta);
    assert_eq!(stay.decision, CompassDecision::StaySurface);
    // Anti-vacuity: the two decisions must be distinguishable, or the enum is
    // decoration.
    assert_ne!(go.decision, stay.decision);
}

// ═══════════════════════════════════════════════════════════════════════════
// Arm 3 — `Sandbox` := Counterfactual + Revision (T10), NOT YET IMPLEMENTED
// ═══════════════════════════════════════════════════════════════════════════

fn probe_poles() -> SplitPoles {
    SplitPoles {
        axis: 3,
        majority_pole: 2,
        minority_pole: -2,
        dissonance: 0.7,
    }
}

/// The Sandbox arm's carrier is CHOSEN but not BUILT.
///
/// `CounterfactualMailbox::new` is a `todo!()` blocked on D-PERSONA-5 (the
/// ractor outer-swarm registration). This pin asserts that unimplemented state
/// deliberately, so the claim in `mul.rs`'s D-MCAL-5 table — "DECLARED, NOT
/// IMPLEMENTED" — cannot quietly go stale.
///
/// **When the scaffold lands, this test fails.** That is the intended
/// behaviour: fix the table in the same commit, then delete or invert this pin.
#[test]
#[should_panic(expected = "not yet implemented")]
fn sandbox_arm_carrier_is_declared_but_not_implemented() {
    let _ = CounterfactualMailbox::new(probe_poles(), 0.5);
}

/// The *type* half of the Sandbox arm is nonetheless real and public: the split
/// that a counterfactual lane is spawned from is a constructible contract
/// value. Only the spawn is blocked.
///
/// This is what makes "no fourth enum" the right call rather than a deferral —
/// the vocabulary already exists, it is the runtime that is missing.
#[test]
fn sandbox_arm_split_vocabulary_is_public_and_expressive() {
    let poles = probe_poles();
    assert_eq!(poles.axis, 3);
    // The two poles are genuinely opposed — a split with equal poles would not
    // be a split, and the carrier must be able to express the difference.
    assert_ne!(poles.majority_pole, poles.minority_pole);
    assert!(poles.dissonance > 0.0);
}

// ═══════════════════════════════════════════════════════════════════════════
// The negative claim: no fourth gate enum was minted
// ═══════════════════════════════════════════════════════════════════════════

/// D-MCAL-5's prohibition, checked at the only level a test can check it: the
/// arms above were reached with **zero** new types. Every symbol this file
/// imports is one that predates D-MCAL-5.
///
/// If a future change adds `MulHint`, `GateLevel`, `MulOutput`, or any other
/// fourth gate vocabulary, this test does not fail — but the reviewer reading
/// it will see the claim it is guarding, and the D-MCAL-5 table in `mul.rs`
/// names why the addition would be the same mistake a third time.
#[test]
fn all_three_arms_reached_without_minting_a_type() {
    let mul = MulAssessment::compute(&SituationInput::default());
    let proceed = mul.free_will_modifier;
    let compass = CompassResult {
        score: 0.5,
        decision: CompassDecision::StaySurface,
    };
    let sandbox_vocab = probe_poles();

    assert!(proceed.is_finite());
    assert_eq!(compass.decision, CompassDecision::StaySurface);
    assert_ne!(sandbox_vocab.majority_pole, sandbox_vocab.minority_pole);
}
