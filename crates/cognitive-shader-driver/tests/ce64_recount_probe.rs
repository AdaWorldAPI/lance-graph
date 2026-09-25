//! Does `CausalEdge64::learn` know an observation it has already absorbed?
//!
//! CHARACTERISATION probe for the cross-cycle follow-up to #1293, not a
//! desired-behaviour contract. If emitted edges were written back and revised
//! each cycle with `learn`, the same SPOFC observation would arrive again on
//! the next dispatch over an unchanged store. These tests pin what `learn`
//! does with it today, so the change that adds evidence identity has to flip
//! them deliberately.

use causal_edge::edge::{CausalEdge64, InferenceType};
use causal_edge::pearl::CausalMask;
use causal_edge::plasticity::PlasticityState;

fn edge(s: u8, f: u8, c: u8, plasticity: PlasticityState) -> CausalEdge64 {
    CausalEdge64::pack(
        s,
        0,
        0,
        f,
        c,
        CausalMask::from_bits(0),
        0,
        InferenceType::Deduction,
        plasticity,
        0,
    )
}

/// Revising with one observation twice counts it twice: `learn` pools the
/// evidence weight unconditionally, with nothing to tell a repeat from a new
/// independent witness.
#[test]
fn learn_counts_a_repeated_observation_as_fresh_evidence() {
    let obs = edge(1, 200, 127, PlasticityState::ALL_HOT);
    let mut once = obs;
    once.learn(obs, 0);
    let mut twice = once;
    twice.learn(obs, 0);
    assert!(
        once.confidence_u8() > obs.confidence_u8(),
        "anti-vacuity: one revision must raise confidence"
    );
    assert!(
        twice.confidence_u8() > once.confidence_u8(),
        "same observation, second revision: {} -> {}",
        once.confidence_u8(),
        twice.confidence_u8()
    );
}

/// The driver emits with `PlasticityState::from_bits(0)`, which is
/// `ALL_FROZEN`: `learn` can then never move S/P/O, whatever the evidence.
#[test]
fn driver_emission_plasticity_freezes_every_plane() {
    assert_eq!(PlasticityState::from_bits(0), PlasticityState::ALL_FROZEN);
    let mut stored = edge(1, 200, 10, PlasticityState::from_bits(0));
    let stronger = edge(2, 200, 250, PlasticityState::ALL_HOT);
    stored.learn(stronger, 0);
    assert_eq!(stored.s_idx(), 1, "a frozen plane keeps its archetype");
    // With the planes hot, the same observation does move S.
    let mut hot = edge(1, 200, 10, PlasticityState::ALL_HOT);
    hot.learn(stronger, 0);
    assert_eq!(hot.s_idx(), 2);
}
