//! End-to-end smoke test: ingest → dispatch → persist → read back.

use std::sync::Arc;

use bgz17::base17::Base17;
use bgz17::palette::Palette;
use bgz17::palette_semiring::PaletteSemiring;

use cognitive_shader_driver::bindspace::{BindSpace, WORDS_PER_FP};
use cognitive_shader_driver::driver::CognitiveShaderBuilder;
use cognitive_shader_driver::engine_bridge::{
    ingest_codebook_indices, persist_cycle, read_qualia_17d, write_qualia_17d,
    classification_distance, read_qualia_decomposed,
};
use cognitive_shader_driver::{
    CognitiveShaderDriver, ColumnWindow, MetaFilter, ShaderDispatch, StyleSelector,
};
use cognitive_shader_driver::auto_style;

fn palette_256() -> PaletteSemiring {
    let entries: Vec<Base17> = (0..256).map(|i| {
        let mut dims = [0i16; 17];
        dims[0] = (i * 100 % 3400) as i16;
        dims[1] = ((i * 37) % 200) as i16;
        Base17 { dims }
    }).collect();
    PaletteSemiring::build(&Palette { entries })
}

fn planes_chain() -> [[u64; 64]; 8] {
    let mut planes = [[0u64; 64]; 8];
    for i in 0..63 {
        planes[0][i] |= 1u64 << (i + 1); // CAUSES chain
    }
    for i in 0..64 {
        planes[2][i] |= 1u64 << i; // SUPPORTS self
    }
    planes
}

#[test]
fn full_pipeline_ingest_dispatch_persist_read() {
    // [1] Create BindSpace + driver.
    let mut bs = BindSpace::zeros(128);

    // [2] Ingest codebook indices (simulating sensor output).
    let indices: Vec<u16> = (0..32).collect();
    let (start, end) = ingest_codebook_indices(&mut bs, &indices, 1, 1000, 0);
    assert_eq!(start, 0);
    assert_eq!(end, 32);

    // Verify content fingerprints have bits set.
    for row in 0..32u32 {
        let words = bs.fingerprints.content_row(row as usize);
        let popcount: u32 = words.iter().map(|w| w.count_ones()).sum();
        assert_eq!(popcount, 1, "each ingested row should have exactly 1 bit set");
    }

    // [3] Write qualia for row 0 (fear-like) and row 1 (steelwind-like).
    let mut fear_q = [0.0f32; 17];
    fear_q[0] = 0.9; fear_q[1] = -0.8; fear_q[2] = 0.9;
    fear_q[3] = 0.1; fear_q[4] = 0.3; fear_q[5] = 0.8;
    write_qualia_17d(&mut bs, 0, &fear_q);

    let mut novel_q = [0.0f32; 17];
    novel_q[0] = 0.5; novel_q[1] = 0.5; novel_q[2] = 0.8;
    novel_q[4] = 0.9;
    write_qualia_17d(&mut bs, 1, &novel_q);

    // Verify CMYK→RGB decomposition.
    let (_, cd_fear) = read_qualia_decomposed(&bs, 0);
    let (_, cd_novel) = read_qualia_decomposed(&bs, 1);
    assert!(cd_fear < cd_novel, "fear should be closer to archetype than novel qualia");

    // [4] Build driver and dispatch.
    let sr = Arc::new(palette_256());
    let bs = Arc::new(bs);
    let driver = CognitiveShaderBuilder::new()
        .bindspace(bs.clone())
        .semiring(sr)
        .planes(planes_chain())
        .build();

    let req = ShaderDispatch {
        rows: ColumnWindow::new(0, 32),
        meta_prefilter: MetaFilter::ALL,
        layer_mask: 0xFF,
        radius: u16::MAX,
        style: StyleSelector::Auto,
        max_cycles: 10,
        ..Default::default()
    };

    let crystal = driver.dispatch(&req);

    // [5] Verify cycle_fingerprint is not all-zero (XOR fold of content rows).
    let fp_popcount: u32 = crystal.bus.cycle_fingerprint.iter()
        .map(|w| w.count_ones()).sum();
    assert!(fp_popcount > 0, "cycle_fingerprint should have bits from XOR fold");

    // [6] Verify resonance top_k has hits.
    let active_hits = crystal.bus.resonance.top_k.iter()
        .filter(|h| h.resonance > 0.0).count();
    assert!(active_hits > 0, "should have at least one resonance hit");

    // [7] Verify style was auto-detected.
    let style_ord = crystal.bus.resonance.style_ord;
    assert!(style_ord < 12, "style ordinal should be 0..11");

    // [8] Persist the cycle back into a fresh BindSpace.
    let mut persist_bs = BindSpace::zeros(4);
    persist_cycle(&mut persist_bs, 0, &crystal.bus, style_ord);

    // Verify cycle fingerprint was written.
    let persisted_fp = persist_bs.fingerprints.cycle_row(0);
    assert_eq!(persisted_fp, &crystal.bus.cycle_fingerprint[..]);

    // Verify meta was packed.
    let meta = persist_bs.meta.get(0);
    assert_eq!(meta.thinking(), style_ord);
    assert!(meta.awareness() > 0, "awareness should reflect gate state");

    eprintln!("Pipeline complete:");
    eprintln!("  Ingested: {} rows", end - start);
    eprintln!("  Cycle FP popcount: {}", fp_popcount);
    eprintln!("  Active hits: {}", active_hits);
    eprintln!("  Style: {} ({})", style_ord,
        cognitive_shader_driver::engine_bridge::unified_style(style_ord).name);
    eprintln!("  Gate: {}", if crystal.bus.gate.is_flow() { "Flow" }
        else if crystal.bus.gate.is_hold() { "Hold" } else { "Block" });
    eprintln!("  Fear CD: {:.3}, Novel CD: {:.3}", cd_fear, cd_novel);
}

#[test]
fn style_auto_detect_matches_qualia() {
    let mut bs = BindSpace::zeros(4);

    // High certainty + low urgency → should detect as analytical.
    let mut q = [0.0f32; 17];
    q[4] = 0.9; // certainty
    q[5] = 0.1; // urgency
    write_qualia_17d(&mut bs, 0, &q);

    let back = read_qualia_17d(&bs, 0);
    let detected = auto_style::style_from_qualia(&back);
    assert_eq!(detected, auto_style::ANALYTICAL,
        "high certainty + low urgency should auto-detect as analytical");
}
