//! ndarray reads CausalEdge64 the same way lance-graph writes it.
//!
//! Each case packs a word with `causal_edge::CausalEdge64::pack` and reads it
//! with ndarray's readers (`hpc::p64_bridge::causal_edge_compat`,
//! `hpc::jina::causal`) and with this crate's. The two must agree on every
//! field both of them expose.

use causal_edge::edge::InferenceType;
use causal_edge::{CausalEdge64, CausalMask, Orientation, PlasticityState};
use ndarray::hpc::jina::causal as jina;
use ndarray::hpc::p64_bridge::causal_edge_compat as compat;

const MASKS: [CausalMask; 8] = [
    CausalMask::None,
    CausalMask::O,
    CausalMask::P,
    CausalMask::PO,
    CausalMask::S,
    CausalMask::SO,
    CausalMask::SP,
    CausalMask::SPO,
];

const INFERENCES: [InferenceType; 7] = [
    InferenceType::Deduction,
    InferenceType::Induction,
    InferenceType::Abduction,
    InferenceType::Revision,
    InferenceType::Synthesis,
    InferenceType::Intervention,
    InferenceType::Counterfactual,
];

fn words() -> Vec<CausalEdge64> {
    let mut out = Vec::new();
    let codes = [(0u8, 0u8, 0u8), (3, 77, 4), (255, 128, 254), (61, 9, 200)];
    let truths = [(0u8, 0u8), (255, 252), (204, 191), (128, 115)];
    for (i, &(s, p, o)) in codes.iter().enumerate() {
        for &mask in &MASKS {
            for &inf in &INFERENCES {
                let (f, c) = truths[i];
                out.push(CausalEdge64::pack(
                    s,
                    p,
                    o,
                    f,
                    c,
                    mask,
                    Orientation::Core,
                    inf,
                    PlasticityState::from_bits(0b101),
                    0,
                ));
            }
        }
    }
    out
}

#[test]
fn ndarray_compat_reads_what_lance_graph_packs() {
    let words = words();
    assert_eq!(words.len(), 4 * 8 * 7);
    for e in words {
        let w = e.0;
        assert_eq!(compat::edge_to_palette_address(w), p64_bridge::edge_to_block(&e), "block of {w:#018x}");
        assert_eq!(compat::edge_nars(w), p64_bridge::edge_nars_f32(&e), "truth of {w:#018x}");
        assert_eq!(compat::edge_to_layer_mask(w), p64_bridge::edge_to_layer_mask(&e), "layer mask of {w:#018x}");
    }
}

#[test]
fn ndarray_jina_reads_what_lance_graph_packs() {
    for e in words() {
        let w = e.0;
        assert_eq!(jina::edge_s(w), e.s_idx());
        assert_eq!(jina::edge_p(w), e.p_idx());
        assert_eq!(jina::edge_o(w), e.o_idx());
        assert_eq!(jina::edge_freq(w), e.frequency_u8() as f32 / 255.0);
        assert_eq!(jina::edge_conf(w), e.confidence_u8() as f32 / 255.0);
        assert_eq!(jina::edge_pearl(w), e.causal_mask() as u8);
        assert_eq!(jina::edge_plasticity(w), 0b101);
    }
}
