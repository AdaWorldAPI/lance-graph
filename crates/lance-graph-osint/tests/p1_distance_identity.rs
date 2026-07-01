//! P1 · distance identity — THE convergence keystone.
//!
//! The V3 "one causal-distance format" claim: the 256×256 palette distance is a
//! single metric shared by every layer that reasons over centroid indices. This
//! probe pins that claim to *shipped code* by driving the three real distance
//! sources off ONE table and asserting they return identical values:
//!
//! ```text
//!   deepnsm::codebook::Codebook::subspace_distance_table(s)   f32 (source)
//!        │  quantize  (the single "f32 NARS edge" tolerance)
//!        ▼
//!   palette[a*256 + b]  : u16                                  the V3 palette
//!        ├───────────────────────────────┐
//!        ▼                                ▼
//!   SpoDistances.s_dist(a,b)         MatrixDistance::distance(a,b)
//!   (planner nars_engine, u16)       (arm-discovery oracle, u32)
//! ```
//!
//! Everything downstream (`causal_distance`, the 8 Pearl projections, ARM
//! discovery, style weighting) reads this same lookup. If the two integer
//! consumers disagree on identical bytes, or the engine table is not the
//! quantized deepnsm table, the "one format" claim breaks *here* — and this test
//! goes red before any convergence work built on it can be trusted.
//!
//! Integer-exact: the only float appears in the deepnsm source table, collapsed
//! by a fixed quantizer before either consumer sees it. No seed, no tolerance,
//! bit-identical on every target.

use deepnsm::codebook::{Codebook, NUM_CENTROIDS};
use lance_graph_arm_discovery::{rule::Item, CodebookDistance, FeatureSpec, MatrixDistance};
use lance_graph_planner::cache::nars_engine::{SpoDistances, SpoHead};

mod common;
use common::splitmix64;

/// Build a synthetic deepnsm `Codebook` via its `load_binary` path (its only
/// public constructor keeps `centroids` private). `[6][256][16]` f32 LE =
/// 24 576 floats = 98 304 bytes. Deterministic centroids in `[0, 1)`.
fn synth_codebook() -> Codebook {
    const N_F32: usize = 6 * NUM_CENTROIDS * 16; // CODEBOOK_SIZE
    let mut s = 0x0517_0700_0517_0701u64; // OSINT-flavoured constant, not entropy
    let mut bytes = Vec::with_capacity(N_F32 * 4);
    for _ in 0..N_F32 {
        // top 24 bits → [0,1) float, deterministic.
        let v = ((splitmix64(&mut s) >> 40) as f32) / (1u64 << 24) as f32;
        bytes.extend_from_slice(&v.to_le_bytes());
    }
    // Unique per call: both tests in this binary run concurrently under the
    // parallel harness and each removes its file after loading; a pid+len-only
    // path collides and one test can truncate the other's load mid-read.
    static CODEBOOK_SEQ: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let seq = CODEBOOK_SEQ.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let path = std::env::temp_dir().join(format!(
        "p1_osint_codebook_{}_{}_{}.bin",
        std::process::id(),
        seq,
        bytes.len()
    ));
    std::fs::write(&path, &bytes).expect("write synthetic codebook");
    let cb = Codebook::load_binary(&path).expect("load synthetic codebook");
    let _ = std::fs::remove_file(&path);
    cb
}

/// Quantize a 256×256 f32 squared-L2 table into the u16 palette. Fixed
/// max-normalize + round: deterministic function of the input bytes, the single
/// permitted "f32 NARS edge" collapse.
fn quantize(table_f32: &[f32]) -> Vec<u16> {
    let max = table_f32
        .iter()
        .copied()
        .fold(0.0f32, f32::max)
        .max(1e-6);
    table_f32
        .iter()
        .map(|&x| ((x / max) * 65_535.0).round().clamp(0.0, 65_535.0) as u16)
        .collect()
}

/// The three planes read from three distinct deepnsm subspaces — the OGAR
/// reading of the 6-byte CAM-PQ path as 3 SPO planes × 2 axes. We take the
/// first axis of each plane: S←subspace 0, P←subspace 2, O←subspace 4.
const S_SUBSPACE: usize = 0;
const P_SUBSPACE: usize = 2;
const O_SUBSPACE: usize = 4;

#[test]
fn p1_deepnsm_source_equals_both_integer_consumers() {
    let cb = synth_codebook();

    // One f32 source per plane → one quantized palette per plane.
    let s_pal = quantize(&cb.subspace_distance_table(S_SUBSPACE));
    let p_pal = quantize(&cb.subspace_distance_table(P_SUBSPACE));
    let o_pal = quantize(&cb.subspace_distance_table(O_SUBSPACE));
    assert_eq!(s_pal.len(), NUM_CENTROIDS * NUM_CENTROIDS);

    // Consumer 1 — planner nars_engine (u16 tables, direct lookup).
    let nars = SpoDistances {
        s_table: s_pal.clone(),
        p_table: p_pal.clone(),
        o_table: o_pal.clone(),
    };

    // Consumer 2 — arm-discovery oracle (u32 table, single 256-cardinality
    // feature so `code(Item{feature:0,category:c}) == c`). Feed the SAME S-plane
    // palette, widened to u32.
    let spec = FeatureSpec::new(vec![NUM_CENTROIDS as u32]);
    let s_pal_u32: Vec<u32> = s_pal.iter().map(|&v| v as u32).collect();
    let arm = MatrixDistance::new(&spec, s_pal_u32);

    // N deterministic index pairs.
    let mut s = 0xDEAD_BEEF_0000_0700u64;
    let n_pairs = 4096;
    for _ in 0..n_pairs {
        let a = (splitmix64(&mut s) % NUM_CENTROIDS as u64) as u8;
        let b = (splitmix64(&mut s) % NUM_CENTROIDS as u64) as u8;

        let nars_s = nars.s_dist(a, b);
        let arm_s = arm.distance(Item::new(0, a as u32), Item::new(0, b as u32));
        let src = s_pal[a as usize * NUM_CENTROIDS + b as usize];

        // (i) the two integer consumers agree byte-for-byte on one palette.
        assert_eq!(
            nars_s as u32, arm_s,
            "consumer divergence at ({a},{b}): nars_engine u16={nars_s} vs arm-discovery u32={arm_s}"
        );
        // (ii) both equal the quantized deepnsm source entry — the engine table
        //      IS the quantized deepnsm subspace table, not an independent one.
        assert_eq!(
            nars_s, src,
            "engine table is not the quantized deepnsm source at ({a},{b})"
        );
        // (iii) squared-L2 is symmetric; the palette inherits it.
        assert_eq!(
            nars.s_dist(a, b),
            nars.s_dist(b, a),
            "palette lost the source's symmetry at ({a},{b})"
        );
    }
}

/// `causal_distance` over the full SPO mask is exactly the sum of the three
/// per-plane palette lookups — the composition the 8 Pearl projections and the
/// downstream style weighting are built on. (P3 tests the masked *subsets*; this
/// only pins the all-planes sum so P1 covers the `SpoHead` entry point too.)
#[test]
fn p1_causal_distance_is_the_plane_sum() {
    let cb = synth_codebook();
    let nars = SpoDistances {
        s_table: quantize(&cb.subspace_distance_table(S_SUBSPACE)),
        p_table: quantize(&cb.subspace_distance_table(P_SUBSPACE)),
        o_table: quantize(&cb.subspace_distance_table(O_SUBSPACE)),
    };

    let mut s = 0x0517_0700_DEAD_0000u64;
    for _ in 0..1024 {
        let head = |seed: &mut u64| {
            let mut h = SpoHead::zero();
            h.s_idx = (splitmix64(seed) % 256) as u8;
            h.p_idx = (splitmix64(seed) % 256) as u8;
            h.o_idx = (splitmix64(seed) % 256) as u8;
            h
        };
        let a = head(&mut s);
        let b = head(&mut s);

        let expected = nars.s_dist(a.s_idx, b.s_idx) as u32
            + nars.p_dist(a.p_idx, b.p_idx) as u32
            + nars.o_dist(a.o_idx, b.o_idx) as u32;
        // mask 0b111 = all three planes active (S=0b100, P=0b010, O=0b001).
        assert_eq!(nars.causal_distance(&a, &b, 0b111), expected);
    }
}
