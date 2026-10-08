//! PROBE-CF-MANTISSA and PROBE-W-PRESERVE
//! (`.claude/board/entries/2026-10-08-coresearch-ce64-moore-masking-wiring.md`).
//!
//! These pin what the CE64 operations do TODAY, two-sided, so a later fix
//! has to flip an assertion on purpose rather than drift past it:
//!
//! 1. `forward` selects its NARS rule from the WEIGHT operand's mantissa
//!    (bits 46..49). Counterfactual (−6) has no arm of its own, so it runs the
//!    Synthesis average and is then re-stamped −6.
//! 2. Which operations keep the W slot (53..58) and `EpistemicState5`
//!    (59..63), and which silently zero them.
//!
//! v2-layout only: under v1 those bits do not exist.
#![cfg(feature = "causal-edge-v2-layout")]

use causal_edge::edge::{CausalEdge64, InferenceType};
use causal_edge::pearl::CausalMask;
use causal_edge::plasticity::PlasticityState;

/// Compose table that keeps the left operand's palette index.
fn keep_left() -> Box<[u8; 256 * 256]> {
    let mut t = vec![0u8; 256 * 256].into_boxed_slice();
    for a in 0..256usize {
        for b in 0..256usize {
            t[a * 256 + b] = a as u8;
        }
    }
    t.try_into().expect("256*256")
}

#[allow(deprecated)] // v2 `pack` ignores temporal; that is not under test here
fn edge(f: u8, c: u8, rule: InferenceType) -> CausalEdge64 {
    CausalEdge64::pack(
        1,
        2,
        3,
        f,
        c,
        CausalMask::SPO,
        0,
        rule,
        PlasticityState::ALL_HOT,
        0,
    )
}

fn forward(running: CausalEdge64, weight: CausalEdge64) -> CausalEdge64 {
    let t = keep_left();
    running.forward(weight, &t, &t, &t)
}

fn fc(e: CausalEdge64) -> (u8, u8) {
    (e.frequency_u8(), e.confidence_u8())
}

const W: u8 = 0b10_1010;
const EPI5: u8 = 0b1_0101;

fn marked(e: CausalEdge64) -> CausalEdge64 {
    e.with_w_slot(W).with_epistemic_raw5(EPI5)
}

// ─── PROBE-CF-MANTISSA ──────────────────────────────────────────────────

/// Can-fire arm: the dispatch on the weight's mantissa is live. Deduction and
/// Synthesis give different truths on the same inputs, so the probe below
/// can tell the two apart.
#[test]
fn forward_dispatch_on_the_weight_mantissa_is_live() {
    let running = edge(255, 255, InferenceType::Deduction);
    let weight = edge(0, 0, InferenceType::Deduction);
    let synth = edge(0, 0, InferenceType::Synthesis);
    assert_ne!(fc(forward(running, weight)), fc(forward(running, synth)));
}

/// MEASURED DEFECT (pinned): a weight carrying Counterfactual (−6) computes
/// exactly the Synthesis average. The mantissa still reads −6 afterwards, so
/// the edge is labelled counterfactual but holds an average.
///
/// FAILS IF: `forward` gains a Counterfactual rule (then flip this test and
/// record the rule) — or stops re-stamping −6.
#[test]
fn a_counterfactual_weight_runs_the_synthesis_average() {
    let running = edge(255, 255, InferenceType::Deduction);
    let cf = edge(0, 0, InferenceType::Deduction)
        .with_inference_mantissa(InferenceType::Counterfactual.to_mantissa());
    let synth = edge(0, 0, InferenceType::Synthesis);
    assert_eq!(cf.inference_mantissa(), -6, "fixture must carry −6");

    let out_cf = forward(running, cf);
    let out_synth = forward(running, synth);
    assert_eq!(fc(out_cf), fc(out_synth), "−6 runs the Synthesis arm");
    assert_eq!(fc(out_cf), (128, 128));
    assert_eq!(out_cf.inference_mantissa(), -6, "and is re-stamped −6");
}

/// Silent arm: the same weight twice gives the same answer.
#[test]
fn forward_is_deterministic_on_identical_inputs() {
    let running = edge(255, 255, InferenceType::Deduction);
    let synth = edge(0, 0, InferenceType::Synthesis);
    assert_eq!(forward(running, synth).0, forward(running, synth).0);
}

// ─── PROBE-W-PRESERVE ───────────────────────────────────────────────────

#[test]
fn the_marked_fixture_really_carries_both_fields() {
    let e = marked(edge(200, 128, InferenceType::Deduction));
    assert_eq!(e.w_slot(), W);
    assert_eq!(e.epistemic_raw5(), EPI5);
}

/// Silent arm: single-field setters keep W and 59..63.
#[test]
fn single_field_setters_preserve_w_and_epi5() {
    let mut e = marked(edge(200, 128, InferenceType::Deduction));
    e.set_frequency(0.25);
    e.set_confidence(0.5);
    e.set_s_idx(9);
    let e = e.with_inference_mantissa(InferenceType::Induction.to_mantissa());
    assert_eq!((e.w_slot(), e.epistemic_raw5()), (W, EPI5));
}

/// Silent arm: `learn` revises in place and keeps both fields.
#[test]
fn learn_preserves_w_and_epi5() {
    let mut e = marked(edge(200, 128, InferenceType::Deduction));
    let before = e.confidence_u8();
    e.learn(edge(200, 128, InferenceType::Deduction), 7);
    assert!(e.confidence_u8() > before, "revision must have run");
    assert_eq!((e.w_slot(), e.epistemic_raw5()), (W, EPI5));
}

/// MEASURED DEFECT (pinned): `forward` rebuilds its result with `pack`, so
/// both fields come back zero even when BOTH operands carry them. Nothing
/// marks the loss.
///
/// FAILS IF: `forward` carries the fields through or flags the strip — flip
/// this test and say which.
#[test]
fn forward_silently_zeroes_w_and_epi5() {
    let running = marked(edge(255, 255, InferenceType::Deduction));
    let weight = marked(edge(200, 200, InferenceType::Deduction));
    let out = forward(running, weight);
    assert_eq!((out.w_slot(), out.epistemic_raw5()), (0, 0));
}

/// MEASURED (pinned): `syllogize` builds a fresh conclusion with `pack`, so it
/// carries neither field. For a NEW edge that may be correct; the point is that
/// it is unmarked either way.
#[test]
fn syllogize_conclusion_carries_neither_field() {
    // Chain figure: (1,2,3) then (3,4,5).
    #[allow(deprecated)]
    let second = CausalEdge64::pack(
        3,
        4,
        5,
        200,
        200,
        CausalMask::SPO,
        0,
        InferenceType::Deduction,
        PlasticityState::ALL_HOT,
        0,
    );
    let a = marked(edge(200, 200, InferenceType::Deduction));
    let b = marked(second);
    let s = a.syllogize(b).expect("chain figure");
    assert_eq!(
        (s.conclusion.w_slot(), s.conclusion.epistemic_raw5()),
        (0, 0)
    );
}
