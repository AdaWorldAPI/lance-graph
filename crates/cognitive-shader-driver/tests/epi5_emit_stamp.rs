//! Bits 59..63 on driver-emitted edges.
//!
//! The driver lowers resonance hits to `CausalEdge64`s. Before this test, no
//! emitted edge carried an `EpistemicState5`, so every one kept `raw5 = 0`,
//! which reads as `Direct × Open`: the strongest topology claim, on a
//! similarity hit. The stamp now comes from the cycle's MUL assessment
//! (`MulAssessment::resonance_epistemic_state`): never `Direct`, never
//! certified beyond `Open`.

use std::collections::BTreeSet;
use std::sync::Arc;

use bgz17::base17::Base17;
use bgz17::palette::Palette;
use bgz17::palette_semiring::PaletteSemiring;
use causal_edge::edge::CausalEdge64;
use cognitive_shader_driver::{
    BindSpaceBuilder, CognitiveShaderBuilder, CognitiveShaderDriver, ColumnWindow, MetaFilter,
    MetaWord, ShaderDispatch, StyleSelector,
};
use lance_graph_contract::qualia::QualiaI4_16D;
use p64_bridge::cognitive_shader::CognitiveShader;

fn equal_distance_semiring() -> PaletteSemiring {
    // Four distinct archetype INDICES with identical metric coordinates.
    // Therefore query 0 sees targets 0..3 at exactly equal distance 0.
    let entries = (0..4)
        .map(|_| Base17 { dims: [0i16; 17] })
        .collect::<Vec<_>>();
    PaletteSemiring::build(&Palette { entries })
}

fn one_source_row() -> cognitive_shader_driver::BindSpace {
    let content = [0u64; 256];
    BindSpaceBuilder::new(1)
        .push(
            &content,
            MetaWord::new(1, 1, 200, 200, 5),
            CausalEdge64::ZERO.0, // s_idx = 0 => query archetype 0
            QualiaI4_16D::ZERO,
            0,
            0,
        )
        .build()
}

fn one_active_block() -> [[u64; 64]; 8] {
    let mut planes = [[0u64; 64]; 8];
    // CAUSES plane, query block-row 0 -> block-column 0.
    // P64's 4×4 refinement expands this ONE topology bit to target
    // archetypes 0,1,2,3.
    planes[0][0] = 1;
    planes
}

#[test]
fn emitted_edges_carry_a_resonance_epistemic_state_not_the_zero_default() {
    use lance_graph_contract::epistemic_state5::{
        Certification3, Epi5Gen, EpistemicState5, Topology2,
    };

    let planes = one_active_block();
    let semiring = Arc::new(equal_distance_semiring());
    let driver = CognitiveShaderBuilder::new()
        .bindspace(Arc::new(one_source_row()))
        .semiring(Arc::clone(&semiring))
        .planes(planes)
        .build();
    let req = ShaderDispatch {
        rows: ColumnWindow::new(0, 1),
        meta_prefilter: MetaFilter::ALL,
        layer_mask: 0b0000_0001,
        radius: u16::MAX,
        style: StyleSelector::Ordinal(1),
        ..ShaderDispatch::default()
    };
    let crystal = driver.dispatch(&req);
    let n = crystal.bus.emitted_edge_count as usize;
    assert!(n > 0, "vacuous: the driver emitted no edge");

    for &raw in &crystal.bus.emitted_edges[..n] {
        let code = CausalEdge64(raw).epistemic_raw5();
        let st = EpistemicState5::decode(Epi5Gen::V1, code).expect("a meaningful V1 code");
        assert_ne!(
            st.topology(),
            Topology2::Direct,
            "edge {raw:#018x} claims Direct"
        );
        assert_eq!(st.certification(), Certification3::Open, "edge {raw:#018x}");
    }
}
