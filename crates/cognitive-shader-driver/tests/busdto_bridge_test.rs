//! D-PARITY-V2-3 round-trip test — `BusDto → encode → BindSpace → unbind → BusDto`.
//!
//! Per `.claude/plans/palantir-parity-cascade-v2.md` D-PARITY-V2-3 + the
//! Tier 2 → Tier 3 transition documented in
//! `.claude/knowledge/soa-dto-dependency-ledger.md` (BusDto = Tier 2,
//! `thinking-engine::dto.rs:115`).
//!
//! Tolerance level:
//!  - **bit-exact** for `codebook_index` (rides the non-affective `temporal`
//!    lane as a lossless u16 IDENTITY pointer — I-VSA-IDENTITIES, 2026-06-17),
//!    `cycle_count` (full u16 fidelity in the `expert` column), and `converged`
//!    (1 bit in MetaWord).
//!  - **bit-exact** for the SET of top_k indices that had positive energy at
//!    encode (they set a cycle bit in 0..16384). Non-positive-energy entries
//!    set no bit, so their idx is lost (round-trips as 0).
//!  - **i4-quantized** for `energy` and the `top_k[*].energy` payload. These
//!    are AFFECTIVE continuous values in [0,1] that legitimately live in the
//!    `qualia` tenant. Post D-CSV-5b the qualia column is signed i4 (step 1/7
//!    on the positive side), so energy round-trips within ±1/7, NOT bit-exact.
//!    The pre-cutover header claimed bit-exact energy "f32 in qualia"; that was
//!    true only while the column was an f32 payload. Documented loss, asserted
//!    to the i4 tolerance below.
//!
//! These tests gate on `--features with-engine` because `BusDto` lives
//! in `thinking-engine` (an optional dependency). They had never run before
//! 2026-06-17 because the crate did not compile under `with-engine` (a missing
//! `QUALIA_DIMS` import); the codebook_index round-trip defect they assert was
//! latent until then.

#![cfg(feature = "with-engine")]

use cognitive_shader_driver::bindspace::BindSpace;
use cognitive_shader_driver::engine_bridge::{dispatch_busdto, unbind_busdto};
use thinking_engine::dto::BusDto;

/// i4 quantization tolerance for affective energy in [0,1]: positive step 1/7,
/// plus rounding slack. Any energy in [0,1] round-trips within this bound.
const ENERGY_I4_TOL: f32 = 1.0 / 7.0 + 1e-4;

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
    // Energy: i4-quantized in the affective qualia tenant (within ±1/7).
    assert!(
        (recovered.energy - bus.energy).abs() <= ENERGY_I4_TOL,
        "energy must round-trip within i4 tolerance: sent {}, got {} (tol {ENERGY_I4_TOL})",
        bus.energy,
        recovered.energy,
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
        assert!(
            (recovered.top_k[i].1 - bus.top_k[i].1).abs() <= ENERGY_I4_TOL,
            "top_k[{i}].energy must round-trip within i4 tolerance: sent {}, got {}",
            bus.top_k[i].1,
            recovered.top_k[i].1,
        );
    }
}

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

    // Energies i4-quantized regardless of sign (negative path step 1/8).
    for i in 0..8 {
        assert!(
            (recovered.top_k[i].1 - bus.top_k[i].1).abs() <= ENERGY_I4_TOL,
            "top_k[{i}].energy must round-trip within i4 tolerance (sparse case): \
             sent {}, got {}",
            bus.top_k[i].1,
            recovered.top_k[i].1,
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
    assert!(
        (recovered.energy - 0.1f32).abs() <= ENERGY_I4_TOL,
        "energy within i4 tolerance: got {}",
        recovered.energy,
    );
}
