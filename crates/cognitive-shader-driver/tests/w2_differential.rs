//! W2 differential harness — one `run()` body, two backing arms, bit-identical.
//!
//! The load-bearing proof of the W3+W4a read-shim: a `ShaderDriver` dispatched
//! against the singleton `BindSpace` and against a `MailboxSoA` that MIRRORS the
//! same window must produce a **byte-identical** `ShaderCrystal`. Every f32 is
//! compared via `to_bits()` (NOT a ULP tolerance): the two arms run the SAME
//! `run()` body and read identical bytes, so the arithmetic is bit-identical —
//! any ULP gap would mean the read path diverged, which is precisely the bug
//! this differential exists to catch (C4).
//!
//! Gated on `mailbox-thoughtspace` because the Mailbox arm of `BackingStore`
//! only exists under that feature.
//!
//! Construction detail: a `ShaderDriver` with NO registered mailbox routes its
//! `backing()` to the singleton fallback; a driver with the designated
//! `DEFAULT_MAILBOX` (id 0) registered routes to the mailbox arm. Both drivers
//! share the SAME `BindSpace`, planes, and semiring, so the only difference is
//! which substrate the six dispatch reads sweep.

#![cfg(feature = "mailbox-thoughtspace")]

use std::sync::Arc;

use bgz17::base17::Base17;
use bgz17::palette::Palette;
use bgz17::palette_semiring::PaletteSemiring;
use cognitive_shader_driver::bindspace::{BindSpace, BindSpaceBuilder, WORDS_PER_FP};
use cognitive_shader_driver::mailbox_soa::MailboxSoA;
use cognitive_shader_driver::{
    auto_style, CognitiveShaderBuilder, CognitiveShaderDriver, ColumnWindow, MetaFilter, MetaWord,
    ShaderCrystal, ShaderDispatch, ShaderHit, StyleSelector,
};
use lance_graph_contract::qualia::QualiaI4_16D;

const DEFAULT_MAILBOX: u32 = 0;

/// Deterministic per-row content fingerprint with a row-dependent bit pattern,
/// so distinct rows produce distinct Hamming distances (non-vacuous resonance).
fn content_for(row: usize) -> [u64; WORDS_PER_FP] {
    let mut w = [0u64; WORDS_PER_FP];
    // A dense, overlapping-but-row-shifted pattern: rows near each other are
    // similar (high resonance), distant rows differ (low) — exercises the
    // content Hamming pre-pass meaningfully.
    for bit in 0..2000usize {
        if (bit + row * 13) % 7 < 5 {
            w[bit / 64] |= 1u64 << (bit % 64);
        }
    }
    w[0] ^= row as u64; // guarantee per-row distinctness
    w
}

fn qualia_for(row: usize) -> QualiaI4_16D {
    QualiaI4_16D::ZERO
        .with(0, (row % 7) as i8)
        .with(3, -((row % 5) as i8))
        .with(9, (row % 4) as i8)
}

/// entity_type stays 0 (untyped) for every row so the ontology ctx_id path is
/// neutralized — the differential isolates the column reads, not ontology
/// resolution (re-homed at W4b).
fn meta_for(row: usize) -> MetaWord {
    MetaWord::new(
        (row % 12) as u8,
        2,
        200u8.saturating_sub((row * 3) as u8),
        200u8.saturating_sub((row * 5) as u8),
        (row % 6) as u8,
    )
}

fn edge_for(row: usize) -> u64 {
    // Vary s_idx (low byte) so the palette cascade query differs per row.
    0xAB00u64 | (row as u64 & 0xFF)
}

fn demo_semiring() -> PaletteSemiring {
    let entries: Vec<Base17> = (0..16)
        .map(|i| {
            let mut dims = [0i16; 17];
            dims[0] = (i as i16) * 100;
            dims[1] = ((i as i16) * 37) % 200;
            Base17 { dims }
        })
        .collect();
    let palette = Palette { entries };
    PaletteSemiring::build(&palette)
}

fn demo_planes() -> [[u64; 64]; 8] {
    let mut planes = [[0u64; 64]; 8];
    for (i, causes) in planes[0].iter_mut().take(16).enumerate() {
        if i + 1 < 64 {
            *causes |= 1u64 << (i + 1);
        }
    }
    for (i, supports) in planes[2].iter_mut().take(16).enumerate() {
        *supports |= 1u64 << i;
    }
    planes
}

/// Build a `BindSpace` of `len` rows with distinct per-row columns.
fn build_bindspace(len: usize) -> BindSpace {
    let mut b = BindSpaceBuilder::new(len);
    for row in 0..len {
        // entity_type = 0 (untyped) — neutralizes ontology (push, not push_typed).
        b = b.push(
            &content_for(row),
            meta_for(row),
            edge_for(row),
            qualia_for(row),
            0,
            0,
        );
    }
    b.build()
}

/// Build a `MailboxSoA<1024>` that mirrors a `len`-row `BindSpace` window,
/// row-for-row, across every migrated read column. `set_populated(len)` makes
/// it the `len`-row logical surrogate; `w_slot = 0` matches the edges' default.
fn mirror_mailbox(len: usize) -> MailboxSoA<1024> {
    let mut mb: MailboxSoA<1024> = MailboxSoA::new(DEFAULT_MAILBOX, 0, 1.0);
    for row in 0..len {
        mb.set_content(row, &content_for(row));
        mb.set_qualia(row, qualia_for(row));
        mb.set_meta(row, meta_for(row));
        mb.set_edge(row, causal_edge::CausalEdge64(edge_for(row)));
        mb.set_entity_type(row, 0);
    }
    mb.set_populated(len);
    mb
}

/// Assert two `ShaderCrystal`s are byte-identical across EVERY field (C4).
fn assert_crystals_bit_identical(a: &ShaderCrystal, b: &ShaderCrystal, ctx: &str) {
    // ── bus.resonance ──────────────────────────────────────────────────────
    let ra = &a.bus.resonance;
    let rb = &b.bus.resonance;
    assert_eq!(ra.hit_count, rb.hit_count, "{ctx}: hit_count");
    assert_eq!(ra.cycles_used, rb.cycles_used, "{ctx}: cycles_used");
    assert_eq!(ra.style_ord, rb.style_ord, "{ctx}: style_ord");
    assert_eq!(
        ra.entropy.to_bits(),
        rb.entropy.to_bits(),
        "{ctx}: entropy bits"
    );
    assert_eq!(
        ra.std_dev.to_bits(),
        rb.std_dev.to_bits(),
        "{ctx}: std_dev bits"
    );
    for k in 0..8 {
        let (ha, hb) = (&ra.top_k[k], &rb.top_k[k]);
        assert_hit_bit_identical(ha, hb, &format!("{ctx}: top_k[{k}]"));
    }

    // ── bus.cycle_fingerprint / emitted_edges / gate ───────────────────────
    assert_eq!(
        a.bus.cycle_fingerprint, b.bus.cycle_fingerprint,
        "{ctx}: cycle_fingerprint"
    );
    assert_eq!(
        a.bus.emitted_edge_count, b.bus.emitted_edge_count,
        "{ctx}: emitted_edge_count"
    );
    assert_eq!(
        a.bus.emitted_edges, b.bus.emitted_edges,
        "{ctx}: emitted_edges"
    );
    assert_eq!(a.bus.gate, b.bus.gate, "{ctx}: gate");

    // ── persisted_row ──────────────────────────────────────────────────────
    assert_eq!(a.persisted_row, b.persisted_row, "{ctx}: persisted_row");

    // ── meta (MetaSummary: 3×f32 + bool) ───────────────────────────────────
    assert_eq!(
        a.meta.confidence.to_bits(),
        b.meta.confidence.to_bits(),
        "{ctx}: meta.confidence bits"
    );
    assert_eq!(
        a.meta.meta_confidence.to_bits(),
        b.meta.meta_confidence.to_bits(),
        "{ctx}: meta.meta_confidence bits"
    );
    assert_eq!(
        a.meta.brier.to_bits(),
        b.meta.brier.to_bits(),
        "{ctx}: meta.brier bits"
    );
    assert_eq!(
        a.meta.should_admit_ignorance, b.meta.should_admit_ignorance,
        "{ctx}: meta.should_admit_ignorance"
    );

    // ── materialize (MaterializeProvenance) ────────────────────────────────
    assert_eq!(
        a.materialize.first_tactic, b.materialize.first_tactic,
        "{ctx}: materialize.first_tactic"
    );
    assert_eq!(
        a.materialize.steps, b.materialize.steps,
        "{ctx}: materialize.steps"
    );
    assert_eq!(
        a.materialize.rested, b.materialize.rested,
        "{ctx}: materialize.rested"
    );
    assert_eq!(
        a.materialize.final_free_energy.to_bits(),
        b.materialize.final_free_energy.to_bits(),
        "{ctx}: materialize.final_free_energy bits"
    );
    assert_eq!(
        a.materialize.fork, b.materialize.fork,
        "{ctx}: materialize.fork"
    );

    // ── alpha_composite (Option<AlphaComposite>) ───────────────────────────
    match (&a.alpha_composite, &b.alpha_composite) {
        (None, None) => {}
        (Some(ca), Some(cb)) => {
            assert_eq!(
                ca.alpha_acc.to_bits(),
                cb.alpha_acc.to_bits(),
                "{ctx}: alpha_acc bits"
            );
            assert_eq!(
                ca.hits_consumed, cb.hits_consumed,
                "{ctx}: alpha hits_consumed"
            );
            assert_eq!(ca.saturated, cb.saturated, "{ctx}: alpha saturated");
            for (i, (x, y)) in ca.color_acc.iter().zip(cb.color_acc.iter()).enumerate() {
                assert_eq!(x.to_bits(), y.to_bits(), "{ctx}: alpha color_acc[{i}] bits");
            }
        }
        _ => panic!("{ctx}: alpha_composite presence differs across arms"),
    }
}

fn assert_hit_bit_identical(a: &ShaderHit, b: &ShaderHit, ctx: &str) {
    assert_eq!(a.row, b.row, "{ctx}: row");
    assert_eq!(a.distance, b.distance, "{ctx}: distance");
    assert_eq!(a.predicates, b.predicates, "{ctx}: predicates");
    assert_eq!(a._pad, b._pad, "{ctx}: _pad");
    assert_eq!(
        a.resonance.to_bits(),
        b.resonance.to_bits(),
        "{ctx}: resonance bits"
    );
    assert_eq!(a.cycle_index, b.cycle_index, "{ctx}: cycle_index");
}

/// Run the SAME dispatch on a singleton-backed driver and a mailbox-backed
/// driver (both over the identical BindSpace/planes/semiring) and assert
/// bit-identity. Returns the singleton crystal so the caller can probe
/// non-vacuity.
fn diff_dispatch(len: usize, req: &ShaderDispatch) -> ShaderCrystal {
    let semiring = Arc::new(demo_semiring());
    let planes = demo_planes();

    // Arm A — singleton (no mailbox registered → backing() = Singleton fallback).
    let driver_singleton = CognitiveShaderBuilder::new()
        .bindspace(Arc::new(build_bindspace(len)))
        .semiring(semiring.clone())
        .planes(planes)
        .build();

    // Arm B — mailbox (DEFAULT_MAILBOX registered → backing() = Mailbox).
    let driver_mailbox = CognitiveShaderBuilder::new()
        .bindspace(Arc::new(build_bindspace(len)))
        .semiring(semiring)
        .planes(planes)
        .with_mailbox(DEFAULT_MAILBOX, mirror_mailbox(len))
        .build();

    let crystal_a = driver_singleton.dispatch(req);
    let crystal_b = driver_mailbox.dispatch(req);
    assert_crystals_bit_identical(&crystal_a, &crystal_b, "full-window vs mirror");
    crystal_a
}

#[test]
fn w2_differential_full_window_bit_identical() {
    let len = 12;
    let req = ShaderDispatch {
        rows: ColumnWindow::new(0, len as u32),
        meta_prefilter: MetaFilter::ALL,
        layer_mask: 0xFF,
        radius: u16::MAX,
        style: StyleSelector::Ordinal(auto_style::CREATIVE),
        ..Default::default()
    };
    let crystal = diff_dispatch(len, &req);
    // Non-vacuity: a creative style over similar rows must produce ≥1 hit.
    assert!(
        crystal.bus.resonance.hit_count > 0,
        "full-window dispatch must be non-vacuous (got 0 hits) — the \
         differential would pass trivially on an empty top_k"
    );
    assert!(
        crystal
            .bus
            .resonance
            .top_k
            .iter()
            .any(|h| h.resonance > 0.0),
        "expected at least one non-zero-resonance hit in top_k"
    );
}

#[test]
fn w2_differential_non_zero_window_bit_identical() {
    // C2 (P0): a window with start > 0. If the Mailbox prefilter iterated
    // `0..populated` instead of honouring `win.start`, this is the case that
    // diverges — same Vec shape, different rows. Bit-identity here is the
    // sentinel against that exact sentinel-lie.
    let len = 12;
    let req = ShaderDispatch {
        rows: ColumnWindow::new(1, (len - 1) as u32), // 1..11
        meta_prefilter: MetaFilter::ALL,
        layer_mask: 0xFF,
        radius: u16::MAX,
        style: StyleSelector::Ordinal(auto_style::CREATIVE),
        ..Default::default()
    };
    let crystal = diff_dispatch(len, &req);
    // Non-vacuity on the windowed case too: ≥1 hit must come from the [1,11)
    // window, and the headline hit's row must be inside the window.
    assert!(
        crystal.bus.resonance.hit_count > 0,
        "non-zero-window dispatch must be non-vacuous (got 0 hits)"
    );
    let headline_row = crystal.bus.resonance.top_k[0].row;
    assert!(
        (1..(len as u32 - 1)).contains(&headline_row),
        "headline hit row {headline_row} must lie inside the dispatched window [1, {})",
        len - 1
    );
}

#[test]
fn w2_differential_with_meta_prefilter_bit_identical() {
    // A restrictive prefilter (high nars_c_min) drops rows on BOTH arms; the
    // surviving-row order must match so the auto-style probe + cascade agree.
    let len = 12;
    let req = ShaderDispatch {
        rows: ColumnWindow::new(0, len as u32),
        meta_prefilter: MetaFilter {
            nars_c_min: 150,
            ..MetaFilter::ALL
        },
        layer_mask: 0xFF,
        radius: u16::MAX,
        style: StyleSelector::Ordinal(auto_style::ANALYTICAL),
        ..Default::default()
    };
    diff_dispatch(len, &req);
}

#[test]
fn w2_differential_alpha_merge_bit_identical() {
    // Exercise the α-front-to-back sink branch across both arms — its qualia
    // closure reads through the shim per hit, so this proves the qualia read
    // surface agrees under the Kerbl compositing path too.
    let len = 12;
    let req = ShaderDispatch {
        rows: ColumnWindow::new(0, len as u32),
        meta_prefilter: MetaFilter::ALL,
        layer_mask: 0xFF,
        radius: u16::MAX,
        style: StyleSelector::Ordinal(auto_style::CREATIVE),
        merge_override: Some(lance_graph_contract::collapse_gate::MergeMode::AlphaFrontToBack),
        ..Default::default()
    };
    let crystal = diff_dispatch(len, &req);
    assert!(
        crystal.alpha_composite.is_some(),
        "AlphaFrontToBack override must populate alpha_composite on both arms"
    );
}
