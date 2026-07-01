//! P2 · edge round-trip · and · P3 · Pearl ladder.
//!
//! P1 proved the 256×256 palette is one metric across three distance sources.
//! P2/P3 prove the *edge* carrier (`causal_edge::CausalEdge64`) and the
//! *distance* engine (`SpoDistances`) index that SAME palette, and that the
//! 3-bit Pearl mask means the same thing on both sides.
//!
//! **P2 — edge round-trip.** OSINT AIRO palette indices `(s,p,o)` → `pack_v2` →
//! the packed edge's `s_idx()/p_idx()/o_idx()` return them unchanged, and its
//! `causal_mask()` returns the Pearl level it was stamped with. The edge carries
//! the palette *coordinate* losslessly; `causal_distance` of the two edges'
//! heads equals the plain per-plane palette sum.
//!
//! **P3 — Pearl ladder.** `CausalMask` (causal-edge) and the `mask` byte
//! (`SpoDistances::causal_distance`) share ONE bit convention — S=0b100,
//! P=0b010, O=0b001 — verified by asserting the masked distance drops exactly
//! the excluded plane's term. `PO` (Level-2 Intervention) projects out the
//! Subject confounder: strictly less distance than `SPO` (Level-3
//! Counterfactual) whenever the Subject term is non-zero.
//!
//! Integer-exact, deterministic, no external input.

use causal_edge::{CausalEdge64, CausalMask, PlasticityState};
use lance_graph_planner::cache::nars_engine::{SpoDistances, SpoHead};

mod common;
use common::splitmix64;

/// A synthetic symmetric palette with a zero diagonal and strictly-positive
/// off-diagonal, so distinct indices always have a non-zero plane distance
/// (needed to exercise the `PO < SPO` confounder-projection assertion).
fn synth_palette(seed: u64) -> Vec<u16> {
    let mut t = vec![0u16; 256 * 256];
    let mut s = seed;
    for a in 0..256usize {
        for b in (a + 1)..256usize {
            let v = 1 + (splitmix64(&mut s) % 60_000) as u16; // 1..=60000
            t[a * 256 + b] = v;
            t[b * 256 + a] = v;
        }
    }
    t
}

/// Bridge a packed edge into the distance engine's head: the SAME palette
/// indices, nothing else. This is the join point P2 certifies.
fn head_of(e: CausalEdge64) -> SpoHead {
    let mut h = SpoHead::zero();
    h.s_idx = e.s_idx();
    h.p_idx = e.p_idx();
    h.o_idx = e.o_idx();
    h
}

fn nars() -> SpoDistances {
    SpoDistances {
        s_table: synth_palette(0x0700_0001),
        p_table: synth_palette(0x0700_0002),
        o_table: synth_palette(0x0700_0003),
    }
}

#[test]
fn p2_edge_round_trips_palette_indices_and_mask() {
    let mut s = 0x0517_0701_0000_0000u64; // osint_person flavour
    for _ in 0..2048 {
        let si = (splitmix64(&mut s) % 256) as u8;
        let pi = (splitmix64(&mut s) % 256) as u8;
        let oi = (splitmix64(&mut s) % 256) as u8;
        let freq = (splitmix64(&mut s) % 256) as u8;
        let conf = (splitmix64(&mut s) % 256) as u8;

        let edge = CausalEdge64::pack_v2(
            si,
            pi,
            oi,
            freq,
            conf,
            CausalMask::SPO,
            0,
            PlasticityState::ALL_FROZEN,
        );

        // The palette coordinate survives pack/unpack unchanged.
        assert_eq!(edge.s_idx(), si, "s_idx round-trip");
        assert_eq!(edge.p_idx(), pi, "p_idx round-trip");
        assert_eq!(edge.o_idx(), oi, "o_idx round-trip");
        // The Pearl level the edge was stamped with round-trips.
        assert_eq!(edge.causal_mask(), CausalMask::SPO, "causal_mask round-trip");
        // Frequency/confidence quantized bytes survive too.
        assert_eq!(edge.frequency_u8(), freq, "frequency_u8 round-trip");
        assert_eq!(edge.confidence_u8(), conf, "confidence_u8 round-trip");
    }
}

#[test]
fn p2_causal_distance_of_two_edges_is_the_plane_sum() {
    let d = nars();
    let mut s = 0x0517_0701_DEAD_0000u64;
    for _ in 0..2048 {
        let mk = |seed: &mut u64| {
            CausalEdge64::pack_v2(
                (splitmix64(seed) % 256) as u8,
                (splitmix64(seed) % 256) as u8,
                (splitmix64(seed) % 256) as u8,
                128,
                200,
                CausalMask::SPO,
                0,
                PlasticityState::ALL_FROZEN,
            )
        };
        let a = head_of(mk(&mut s));
        let b = head_of(mk(&mut s));

        let expected = d.s_dist(a.s_idx, b.s_idx) as u32
            + d.p_dist(a.p_idx, b.p_idx) as u32
            + d.o_dist(a.o_idx, b.o_idx) as u32;
        assert_eq!(d.causal_distance(&a, &b, CausalMask::SPO as u8), expected);
    }
}

#[test]
fn p3_pearl_masks_drop_exactly_the_excluded_plane() {
    let d = nars();
    let mut s = 0x0517_0701_BEEF_0000u64;
    let mut saw_strict_drop = 0u32;
    for _ in 0..4096 {
        let mk = |seed: &mut u64| {
            head_of(CausalEdge64::pack_v2(
                (splitmix64(seed) % 256) as u8,
                (splitmix64(seed) % 256) as u8,
                (splitmix64(seed) % 256) as u8,
                128,
                200,
                CausalMask::PO, // Level-2 Intervention on the osint_person pair
                0,
                PlasticityState::ALL_FROZEN,
            ))
        };
        let a = mk(&mut s);
        let b = mk(&mut s);

        let s_d = d.s_dist(a.s_idx, b.s_idx) as u32;
        let p_d = d.p_dist(a.p_idx, b.p_idx) as u32;
        let o_d = d.o_dist(a.o_idx, b.o_idx) as u32;

        // The bit convention is shared: each mask keeps exactly its planes.
        assert_eq!(d.causal_distance(&a, &b, CausalMask::S as u8), s_d);
        assert_eq!(d.causal_distance(&a, &b, CausalMask::PO as u8), p_d + o_d);
        assert_eq!(
            d.causal_distance(&a, &b, CausalMask::SPO as u8),
            s_d + p_d + o_d
        );
        assert_eq!(d.causal_distance(&a, &b, CausalMask::None as u8), 0);

        // Level-2 (PO) projects out the Subject confounder → strictly less than
        // Level-3 (SPO) exactly when the Subject term is non-zero.
        let po = d.causal_distance(&a, &b, CausalMask::PO as u8);
        let spo = d.causal_distance(&a, &b, CausalMask::SPO as u8);
        if s_d > 0 {
            assert!(po < spo, "PO must drop the non-zero Subject term");
            saw_strict_drop += 1;
        } else {
            assert_eq!(po, spo);
        }
    }
    // The synthetic palette has a zero diagonal only; distinct subjects give
    // s_d>0, so the strict-drop branch must fire on the vast majority of pairs.
    assert!(
        saw_strict_drop > 4000,
        "expected the confounder-projection branch to dominate, saw {saw_strict_drop}"
    );
}
