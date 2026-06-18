//! D-PARITY-V2-3 round-trip test — `BusDto → encode → BindSpace → unbind → BusDto`.
//!
//! Per `.claude/plans/palantir-parity-cascade-v2.md` D-PARITY-V2-3 + the
//! Tier 2 → Tier 3 transition documented in
//! `.claude/knowledge/soa-dto-dependency-ledger.md` (BusDto = Tier 2,
//! `thinking-engine::dto.rs:115`).
//!
//! Tolerance level: **bit-exact** for codebook_index, top_k indices with
//! positive energy at encode, energies (f32 in qualia), cycle_count
//! (full u16 fidelity in expert column), and converged. The `top_k`
//! entries with non-positive energy at encode (i.e. they did not set a
//! cycle bit) lose their idx but keep their energy through the qualia
//! store; their idx round-trips as 0.
//!
//! These tests gate on `--features with-engine` because `BusDto` lives
//! in `thinking-engine` (an optional dependency).

#![cfg(feature = "with-engine")]

use cognitive_shader_driver::bindspace::BindSpace;
use cognitive_shader_driver::engine_bridge::{dispatch_busdto, unbind_busdto};
use thinking_engine::dto::BusDto;

/// All-positive top_k: the strictest round-trip case (every supporter bit set).
fn make_dense_bus(seed: u16) -> BusDto {
    let mut top_k = [(0u16, 0.0f32); 8];
    for i in 0..8 {
        top_k[i] = (
            seed.wrapping_add((i as u16).wrapping_mul(37)),
            0.5 + (i as f32) * 0.05,
        );
    }
    BusDto {
        codebook_index: seed,
        energy: 0.85,
        top_k,
        cycle_count: 7,
        converged: true,
    }
}

// The full index-recovery round-trips (dense + sparse + zero-headline) depend
// on the `cycle` plane, which `mailbox-thoughtspace` drops (C5 / D-DIST-5).
// They stay LIVE on the singleton (default) build; a separate mailbox-arm test
// below pins the documented loss. Energy/cycle_count/converged/headline parity
// holds on BOTH builds and is asserted unconditionally where convenient.
#[cfg(not(feature = "mailbox-thoughtspace"))]
#[test]
fn busdto_round_trip_dense_top_k_is_bit_exact() {
    let mut bs = BindSpace::zeros(8);
    let bus = make_dense_bus(42);

    dispatch_busdto(&mut bs, 0, &bus, 1 /* analytical */);
    let recovered = unbind_busdto(&bs, 0);

    // Headline codebook_index: bit-exact.
    assert_eq!(
        recovered.codebook_index, bus.codebook_index,
        "codebook_index must round-trip bit-exact"
    );
    // Energy: bit-exact f32 (qualia store).
    assert_eq!(
        recovered.energy.to_bits(),
        bus.energy.to_bits(),
        "energy must round-trip bit-exact"
    );
    // cycle_count: bit-exact (expert column = u16, no saturation).
    assert_eq!(recovered.cycle_count, bus.cycle_count);
    // converged: bit-exact.
    assert_eq!(recovered.converged, bus.converged);

    // top_k: every entry has positive energy → idx must round-trip; energies are
    // f32 bit-exact stored in qualia. Indices may reorder for ties / collisions
    // because the encoder uses positional bits and the decoder walks ascending
    // bit positions. So we assert SET-equality on indices and bit-exact energies.
    let mut sent_idx: Vec<u16> = std::iter::once(bus.codebook_index)
        .chain(bus.top_k.iter().map(|&(i, _)| i))
        .collect();
    let mut got_idx: Vec<u16> = std::iter::once(recovered.codebook_index)
        .chain(recovered.top_k.iter().map(|&(i, _)| i))
        .collect();
    sent_idx.sort_unstable();
    got_idx.sort_unstable();
    sent_idx.dedup();
    got_idx.dedup();
    assert_eq!(
        sent_idx, got_idx,
        "top_k index SET must be bit-exact (positions in 0..16384)"
    );

    for i in 0..8 {
        assert_eq!(
            recovered.top_k[i].1.to_bits(),
            bus.top_k[i].1.to_bits(),
            "top_k[{i}].energy must round-trip bit-exact",
        );
    }
}

#[cfg(not(feature = "mailbox-thoughtspace"))]
#[test]
fn busdto_round_trip_sparse_top_k_preserves_positive_idx_set() {
    // Mix of positive + zero + negative energy supporters. The encoder only
    // sets bits for positive-energy entries; non-positive entries lose their
    // idx (round-trip as 0) but keep their f32 energy through qualia.
    let mut bs = BindSpace::zeros(4);
    let bus = BusDto {
        codebook_index: 1234,
        energy: 0.42,
        top_k: [
            (1234, 0.42), // positive — idx must round-trip
            (5000, 0.30), // positive — idx must round-trip
            (300, -0.10), // negative — idx LOST, energy kept
            (888, 0.0),   // zero — idx LOST, energy kept (zero)
            (777, 0.20),  // positive — idx must round-trip
            (0, 0.0),
            (0, 0.0),
            (0, 0.0),
        ],
        cycle_count: 3,
        converged: false,
    };

    dispatch_busdto(&mut bs, 1, &bus, 0);
    let recovered = unbind_busdto(&bs, 1);

    // Positive-energy indices must all be present in the recovered set.
    let positive_set: std::collections::BTreeSet<u16> = bus
        .top_k
        .iter()
        .filter(|(_, e)| *e > 0.0)
        .map(|&(i, _)| i)
        .chain(std::iter::once(bus.codebook_index))
        .collect();
    let recovered_set: std::collections::BTreeSet<u16> = recovered
        .top_k
        .iter()
        .filter(|(i, e)| *i != 0 && *e > 0.0)
        .map(|&(i, _)| i)
        .chain(std::iter::once(recovered.codebook_index))
        .collect();
    assert!(
        positive_set.is_subset(&recovered_set),
        "every positive-energy idx must appear in the recovered set: \
         positive={positive_set:?}, recovered={recovered_set:?}",
    );

    // Energies bit-exact regardless of sign.
    for i in 0..8 {
        assert_eq!(
            recovered.top_k[i].1.to_bits(),
            bus.top_k[i].1.to_bits(),
            "top_k[{i}].energy must round-trip bit-exact (sparse case)",
        );
    }
    // converged: bit-exact (false in this case).
    assert_eq!(recovered.converged, bus.converged);
    assert_eq!(recovered.cycle_count, bus.cycle_count);
}

#[test]
fn busdto_round_trip_high_cycle_count_is_lossless_via_expert_column() {
    // cycle_count > 63 saturates in MetaWord.free_e (6 bits) but we store
    // the full u16 in expert[row], so unbind must return the original.
    let mut bs = BindSpace::zeros(2);
    let bus = BusDto {
        codebook_index: 7,
        energy: 1.0,
        top_k: [(7, 1.0); 8],
        cycle_count: 9999, // way beyond MetaWord.free_e's 63-cap
        converged: false,
    };
    dispatch_busdto(&mut bs, 0, &bus, 11 /* metacognitive */);
    let recovered = unbind_busdto(&bs, 0);
    assert_eq!(
        recovered.cycle_count, 9999,
        "cycle_count must use the expert column for full u16 fidelity"
    );
    assert_eq!(recovered.converged, false);
}

#[test]
fn busdto_dispatch_writes_meta_thinking_style() {
    // The caller's style ordinal must land in MetaWord.thinking.
    let mut bs = BindSpace::zeros(3);
    let bus = make_dense_bus(99);
    dispatch_busdto(&mut bs, 2, &bus, 7 /* focused */);
    let m = bs.meta.get(2);
    assert_eq!(
        m.thinking(),
        7,
        "style ordinal must land in MetaWord.thinking"
    );
    assert_eq!(m.awareness(), 3, "converged=true → awareness=FLOW(3)");
    // free_e clamped to <=63, but cycle_count was 7 so no clamp.
    assert_eq!(m.free_e(), 7);
}

#[test]
fn busdto_round_trip_zero_codebook_index_is_handled() {
    // Edge case: codebook_index = 0 means bit 0 is the only definite set bit.
    // The decoder must still recover index 0 (the lowest-set-bit fallback path).
    let mut bs = BindSpace::zeros(1);
    let bus = BusDto {
        codebook_index: 0,
        energy: 0.1,
        top_k: [
            (0, 0.1),
            (0, 0.0),
            (0, 0.0),
            (0, 0.0),
            (0, 0.0),
            (0, 0.0),
            (0, 0.0),
            (0, 0.0),
        ],
        cycle_count: 0,
        converged: true,
    };
    dispatch_busdto(&mut bs, 0, &bus, 0);
    let recovered = unbind_busdto(&bs, 0);
    assert_eq!(recovered.codebook_index, 0);
    assert_eq!(recovered.cycle_count, 0);
    assert_eq!(recovered.converged, true);
    assert_eq!(recovered.energy.to_bits(), 0.1f32.to_bits());
}

// ── C5 / D-DIST-5: mailbox-arm downgraded contract (pin the documented loss) ──
//
// Under `mailbox-thoughtspace` the `cycle` plane is dropped, so `unbind_busdto`
// recovers ONLY the headline `codebook_index` (from qualia[9]); the non-headline
// `top_k[1..].idx` recover as `0`. This test asserts that loss EXPLICITLY rather
// than tolerating it (never relax the bit-exact singleton tests above — they
// stay live on the default build). Requires BOTH `with-engine` (BusDto) and
// `mailbox-thoughtspace` (the gated path).
#[cfg(feature = "mailbox-thoughtspace")]
#[test]
fn busdto_mailbox_arm_recovers_headline_only_nonheadline_idx_zero() {
    let mut bs = BindSpace::zeros(4);
    // Distinct non-zero indices so a non-zero recovery would be visible.
    let bus = BusDto {
        codebook_index: 4321,
        energy: 0.77,
        top_k: [
            (4321, 0.9), // headline-matching positive slot
            (1111, 0.8), // positive non-headline — idx WILL be lost (→ 0)
            (2222, 0.7), // positive non-headline — idx WILL be lost (→ 0)
            (3333, 0.6),
            (0, 0.0),
            (0, 0.0),
            (0, 0.0),
            (0, 0.0),
        ],
        cycle_count: 5,
        converged: true,
    };
    dispatch_busdto(&mut bs, 0, &bus, 1);
    let recovered = unbind_busdto(&bs, 0);

    // Headline survives — qualia[9] is lossless and not on the dropped plane.
    assert_eq!(
        recovered.codebook_index, 4321,
        "headline codebook_index must survive the cycle-plane drop (qualia[9])"
    );
    // top_k[0].idx echoes the headline because top_k[0] had positive energy.
    assert_eq!(
        recovered.top_k[0].0, 4321,
        "top_k[0].idx echoes the headline (positive energy slot)"
    );
    // Every NON-headline top_k idx is lost → recovers as 0 (the C5 loss).
    for i in 1..8 {
        assert_eq!(
            recovered.top_k[i].0, 0,
            "mailbox arm: non-headline top_k[{i}].idx must recover as 0 \
             (cycle plane dropped — C5 / D-DIST-5)"
        );
    }
    // Energies, cycle_count, converged still round-trip bit-exact (qualia/expert/meta).
    for i in 0..8 {
        assert_eq!(
            recovered.top_k[i].1.to_bits(),
            bus.top_k[i].1.to_bits(),
            "top_k[{i}].energy must round-trip bit-exact even on the mailbox arm",
        );
    }
    assert_eq!(recovered.energy.to_bits(), bus.energy.to_bits());
    assert_eq!(recovered.cycle_count, 5);
    assert_eq!(recovered.converged, true);
}

/// C8 (truth-architect addendum) — headline `codebook_index` round-trips
/// bit-exact across the full u16 corner corpus, on BOTH builds.
///
/// The fix stores `codebook_index` in the **f32 qualia tenant** (`q[9]`); f32
/// represents every integer in `[0, 2^24]` exactly, so every u16 survives. The
/// corner values pin the regression boundaries: `255→256` is where
/// `MetaWord.nars_f = codebook_index & 0xFF` aliases (a future refactor that
/// reconstructed the headline from `nars_f` would collapse 256/512/0), and
/// `65535` is the u16 max. The i4 column this tenant supersedes clamped every
/// value ≥ 1 to 1 — this test is the proof that the f32 tenant repairs it.
#[test]
fn busdto_codebook_index_corner_corpus_round_trips_bit_exact() {
    for &idx in &[0u16, 255, 256, 1234, 4095, 65535] {
        let mut bs = BindSpace::zeros(1);
        let bus = BusDto {
            codebook_index: idx,
            energy: 0.5,
            top_k: [(idx, 0.5); 8],
            cycle_count: 1,
            converged: true,
        };
        dispatch_busdto(&mut bs, 0, &bus, 0);
        let recovered = unbind_busdto(&bs, 0);
        assert_eq!(
            recovered.codebook_index, idx,
            "codebook_index {idx} must round-trip bit-exact via the f32 qualia tenant",
        );
    }
}
