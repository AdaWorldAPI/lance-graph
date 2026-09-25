//! F-ARW-TARGET-1 — does a P64 target survive the live shader → CE64 seam?
//!
//! This is a CHARACTERISATION / FALSIFIER probe, not a desired-behaviour
//! contract. One BindSpace source row is made to produce four distinct P64
//! target archetypes at exactly equal distance under one predicate layer.
//!
//! Upstream anti-vacuity: `CognitiveShader::cascade` must expose >1 distinct
//! `CascadeHit.target` values.
//!
//! Downstream observation: `ShaderDriver` currently lowers each cascade hit to
//! `ShaderHit { row, distance, predicates, resonance, ... }`, retaining the
//! source BindSpace row but not `CascadeHit.target`. CE64 emission then derives
//! S/P/O from that source row. If all emitted CE64 words are identical, target
//! identity is proven lost at this seam.
//!
//! Re-pinned 2026-09-25 (#1293). The four targets used to become four
//! candidate slots and four byte-identical CE64s — repeated emissions of one
//! relation. Stage [3] now aggregates every supporting relationship per source
//! row (SPOFC): the four targets are ONE candidate with support 4, whose best
//! target is kept as its object (`driver::tests`,
//! `p64_targets_survive_as_spofc_support`). What stands is narrower than
//! "CE64 cannot carry the target": its 24-bit S/P/O can hold three palette256
//! indices, so a target has a place to go. The driver just never writes it
//! there — it packs S/O from the source row id and P = 0. So exactly one edge
//! now represents the four relations, with the target left in the SPOFC record.
//! Writing the aggregated support and target into CE64 is the follow-up.

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
fn distinct_p64_targets_collapse_to_one_emitted_ce64_identity() {
    let planes = one_active_block();
    let semiring = Arc::new(equal_distance_semiring());

    // Anti-vacuity: P64 itself really distinguishes multiple target identities.
    let shader = CognitiveShader::new(planes, semiring.as_ref());
    let raw = shader.cascade(0, u16::MAX, 0b0000_0001);
    let targets = raw.iter().map(|h| h.target).collect::<BTreeSet<_>>();
    assert!(
        targets.len() > 1,
        "probe is vacuous: P64 did not produce multiple target archetypes: {targets:?}"
    );
    assert!(
        raw.windows(2)
            .all(|pair| pair[0].distance == pair[1].distance),
        "probe requires equal metric distance so only identity can distinguish targets: {raw:?}"
    );

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
    // One source row with four relations is one candidate, so one CE64 — not
    // four identical copies of it.
    assert_eq!(
        n,
        1,
        "the four targets of one source row must be one candidate: {:?}",
        &crystal.bus.emitted_edges[..n]
    );

    let decoded = CausalEdge64(crystal.bus.emitted_edges[0]);
    assert_eq!(decoded.s_idx(), 0, "source row 0 is projected to S=0");
    assert_eq!(decoded.p_idx(), 0, "current emission writes P=0");
    assert_eq!(decoded.o_idx(), 0, "source row 0 is projected to O=0");
}
