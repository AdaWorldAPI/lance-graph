//! T4 (CE64 coresearch, 2026-10-08): `learn` against `isa::contracts::LEARN`,
//! bit-exact over all eight plasticity patterns.
//!
//! The contract declares Pearl, Direction, Inference, Witness and Epistemic
//! as pass-through from A. Per plane, a FROZEN plane keeps A's archetype and
//! a HOT plane adopts the observation's when the observation is more
//! confident. The fixture keeps the revised confidence below 0.7, so the
//! plasticity field itself is not changed by the confidence transitions.
//!
//! The masks below are the v2 bit layout (plasticity 50..52, witness and
//! Epi5 53..63); v1 puts plasticity at 49..51 and temporal at 52..63, which
//! `learn` rewrites. Gated like the other layout-specific contract tests.
#![cfg(feature = "causal-edge-v2-layout")]

use causal_edge::edge::InferenceType;
use causal_edge::{CausalEdge64, CausalMask, PlasticityState};

/// Bits 40..49 (Pearl, Direction, Inference) and 53..63 (Witness, Epi5).
const PASS: u64 = (0x3FF << 40) | (0x7FF << 53);
const PLAST_SHIFT: u32 = 50;

#[allow(deprecated)] // v2 `pack` ignores temporal; not under test
fn edge(s: u8, p: u8, o: u8, c: u8, plast: PlasticityState) -> CausalEdge64 {
    let mut e = CausalEdge64::pack(
        s,
        p,
        o,
        200,
        c,
        CausalMask::SO,
        0b101,
        InferenceType::Abduction,
        plast,
        0,
    );
    // Non-zero witness and epistemic bits, so pass-through is observable.
    e.0 |= 0b101_1010_1101_u64 << 53;
    e
}

#[test]
fn learn_passes_frozen_planes_and_declared_fields_for_every_pattern() {
    for bits in 0u8..8 {
        let plast = PlasticityState::from_bits(bits);
        let a = edge(1, 2, 3, 51, plast); // c = 0.2
        let obs = edge(9, 8, 7, 77, PlasticityState::ALL_HOT); // c ~ 0.3
        let mut out = a;
        out.learn(obs, 0);

        assert_eq!(
            out.0 & PASS,
            a.0 & PASS,
            "pass-through, plasticity {bits:03b}"
        );
        assert_eq!(
            out.0 >> PLAST_SHIFT & 0b111,
            u64::from(bits),
            "plasticity {bits:03b}"
        );
        let want = |hot: bool, mine: u8, theirs: u8| if hot { theirs } else { mine };
        assert_eq!(
            out.s_idx(),
            want(plast.s_hot(), 1, 9),
            "S, plasticity {bits:03b}"
        );
        assert_eq!(
            out.p_idx(),
            want(plast.p_hot(), 2, 8),
            "P, plasticity {bits:03b}"
        );
        assert_eq!(
            out.o_idx(),
            want(plast.o_hot(), 3, 7),
            "O, plasticity {bits:03b}"
        );
        assert!(
            out.confidence_u8() > 51 && out.confidence_u8() < 179,
            "{}",
            out.confidence_u8()
        );
    }
}

/// The fixture can see a change: a hot plane really does move, so the
/// frozen-plane assertions above are not satisfied by `learn` doing nothing.
#[test]
fn learn_moves_a_hot_plane() {
    let mut out = edge(1, 2, 3, 51, PlasticityState::ALL_HOT);
    out.learn(edge(9, 8, 7, 77, PlasticityState::ALL_HOT), 0);
    assert_eq!((out.s_idx(), out.p_idx(), out.o_idx()), (9, 8, 7));
}
