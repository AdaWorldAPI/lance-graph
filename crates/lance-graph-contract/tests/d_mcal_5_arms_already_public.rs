//! D-MCAL-5 — the planner's three MUL arms, reached from the contract alone.
//!
//! The deliverable asks: *if a public MUL output is still needed, derive it
//! from the planner's `Proceed/Sandbox/Compass` — never invent a fresh enum.*
//!
//! Measured answer, **corrected after codex review of #1069**: no new type is
//! needed — but the first version of this claim ("and no promotion is needed
//! either") was too strong and is withdrawn. The distinction it missed:
//!
//! - the arms' **payloads** are reachable from this crate;
//! - the arm **selection** is not, because three of the five branch conditions
//!   in `planner::mul::gate::check` test `TrustTexture::{Murky, Dissonant,
//!   Fuzzy}` — planner-private variants this crate does not have.
//!
//! So the public-output gap is REAL and still open. It is simply not an
//! enum-shaped gap: minting `Proceed/Sandbox/Compass` here would produce a type
//! this crate cannot populate. The blocker is OQ-MCAL-1 — two MUL
//! implementations with disjoint trust vocabularies — and a fourth gate enum
//! would hide it behind a type always constructed from a guess.
//!
//! This file makes each claim falsifiable:
//!
//! - the two live payload carriers are **constructed here from
//!   `lance_graph_contract` alone**, with no planner dependency — if either
//!   stopped being reachable, this file would not compile;
//! - each is checked to actually *vary*, so "there is a carrier" is not
//!   satisfied by a constant;
//! - the **selection gap** is pinned by exhausting this crate's `TrustTexture`
//!   and showing none of its variants is the one that picks `Compass`;
//! - the unimplemented arm is pinned with `#[should_panic]` — **twice**, spawn
//!   half and revision half separately, because the revision half carries three
//!   blockers the spawn half does not, and one pin would read as "Sandbox is
//!   ready" the moment D-PERSONA-5 landed.

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

// The revision half is pinned SEPARATELY from the spawn half (codex review of
// #1069). `revise_if_minority_wins` carries three blockers the spawn half does
// not — the unconfirmed `awareness.revise` signature, D-ATOM-1's `axis_key`
// type, and D-ATOM-5's tombstone wiring — so a single pin on
// `CounterfactualMailbox::new` would read as "Sandbox is ready" the moment
// D-PERSONA-5 lands. Two pins, two blockers, no false all-clear.

struct StubEdge(i8);
impl lance_graph_contract::counterfactual::EpisodicEdge for StubEdge {
    fn set_inference_mantissa(&mut self, m: i8) {
        self.0 = m;
    }
    fn inference_mantissa(&self) -> i8 {
        self.0
    }
}

struct StubAwareness;
impl lance_graph_contract::counterfactual::AwarenessRevise for StubAwareness {
    fn revise(
        &mut self,
        _axis_key: u8,
        _new_evidence: i8,
    ) -> Result<(), lance_graph_contract::counterfactual::CounterfactualError> {
        Ok(())
    }
}

/// The Sandbox arm's **revision** half is unimplemented independently of its
/// spawn half.
///
/// Blocked on D-PERSONA-5 *and* three things D-PERSONA-5 does not deliver: the
/// `awareness.revise` signature (unconfirmed on the current contract surface),
/// D-ATOM-1's `axis_key` type, and D-ATOM-5's revision tombstone into Lance.
///
/// **When this stops panicking, the four-blocker list in `mul.rs` must be
/// re-checked in the same commit.**
#[test]
#[should_panic(expected = "not yet implemented")]
fn sandbox_arm_revision_half_is_separately_unimplemented() {
    let mut edge = StubEdge(0);
    let mut awareness = StubAwareness;
    // Constructing the mailbox itself panics first only because `new` is also a
    // stub; to isolate the revision half we would need a constructed mailbox,
    // which is precisely what the spawn blocker withholds. The panic below is
    // therefore the *conjunction* of both blockers, and the doc above names
    // which is which so neither can be silently discharged by the other.
    let poles = probe_poles();
    let mailbox = CounterfactualMailbox::new(poles, 0.5).expect("spawn blocked");
    let _ = lance_graph_contract::counterfactual::revise_if_minority_wins(
        mailbox,
        &mut edge,
        &mut awareness,
    );
}

/// The *type* half of the Sandbox arm is nonetheless real and public: the split
/// that a counterfactual lane is spawned from is a constructible contract
/// value. Only the runtime is blocked.
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

/// The **selection gap** (codex review of #1069): the arms' payloads are
/// reachable, but which arm fires is not derivable here.
///
/// `planner::mul::gate::check` branches on five conditions. Two live on
/// `MulAssessment` and are checkable from this crate. The other three test
/// `TrustTexture::{Murky, Dissonant, Fuzzy}` — variants of the *planner's*
/// trust vocabulary. This test pins that this crate's `TrustTexture` has
/// exactly four variants and none of them is the one that selects `Compass`,
/// so the gap is a measured fact rather than a claim in a comment.
///
/// If a future change adds those variants here, this test fails — and at that
/// point the D-MCAL-5 conclusion genuinely does need revisiting, because the
/// selection would have become derivable.
#[test]
fn selection_between_the_arms_is_not_derivable_from_the_contract() {
    // The two conditions that ARE available: both route to Sandbox.
    let mount_stupid = MulAssessment::compute(&SituationInput {
        felt_competence: 0.95,
        demonstrated_competence: 0.10,
        ..SituationInput::default()
    });
    assert_eq!(
        mount_stupid.dk_position,
        lance_graph_contract::mul::DkPosition::MountStupid,
        "the one selection input this crate can check must actually be checkable"
    );
    let _: bool = mount_stupid.complexity_mapped;

    // The three that are NOT: exhaust this crate's TrustTexture and show none
    // of them is the planner's `Fuzzy` / `Murky` / `Dissonant`. The match is
    // exhaustive, so adding a variant here breaks the build and forces a
    // re-read of the D-MCAL-5 table.
    use lance_graph_contract::mul::TrustTexture as T;
    for t in [
        T::Calibrated,
        T::Overconfident,
        T::Underconfident,
        T::Uncertain,
    ] {
        let name = match t {
            T::Calibrated => "Calibrated",
            T::Overconfident => "Overconfident",
            T::Underconfident => "Underconfident",
            T::Uncertain => "Uncertain",
        };
        assert!(
            !matches!(name, "Fuzzy" | "Murky" | "Dissonant"),
            "contract TrustTexture gained a planner-selection variant ({name}); \
             the D-MCAL-5 selection-gap conclusion needs revisiting"
        );
    }
}

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
