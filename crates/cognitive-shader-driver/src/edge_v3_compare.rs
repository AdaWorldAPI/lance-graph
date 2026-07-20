//! Compare-thinking harness: `CausalEdge64` (shipping) vs `CausalEdgeV3`
//! (staged parallel type), on the driver's REAL edge-emission recipe.
//!
//! Operator directive: *"migrate staged, keep causaledge64 in parallel and
//! compare thinking."* This module is the **parallel comparison**, additive
//! and read-only — it touches NO hot-loop code and stores NO new column. It
//! lifts each emitted `CausalEdge64` to `CausalEdgeV3` (via `from_v1`) and
//! proves two things on edges packed EXACTLY as [`crate::driver`] packs them
//! (`driver.rs` [5] "Emit one CausalEdge64 per strong hit"):
//!
//!   1. **Thinking is preserved.** For every premise pair, the V3 edges —
//!      SPO dropped, resolved from the target node's CAM-PQ facet — syllogize
//!      to the IDENTICAL conclusion as the v1 edges. The 24-bit in-edge SPO
//!      was a pure duplicate (`E-CAUSALEDGE-V3-96-STAGED-1`).
//!   2. **The v2 edge silently drops a temporal the driver computes; the V3
//!      edge can carry a per-edge one.** The driver passes `h.cycle_index` as
//!      `CausalEdge64::pack`'s `temporal` arg (`driver.rs`, [5]), but under
//!      the v2 layout that write is a no-op (`I-LEGACY-API-FEATURE-GATED` —
//!      the reclaimed bits are W-slot/lens). That drop is *why* `MailboxSoA`
//!      keeps its own cycle bookkeeping. `CausalEdgeV3`'s TE byte carries a
//!      **signed −128..+127 per-edge chain offset** in the edge itself.
//!
//! ### What TE is NOT (a width caveat that gates the cut-over)
//!
//! TE is an `i8` (this harness narrows further to 7 bits). It is a **relative
//! chain offset**, NOT a replacement for `MailboxSoA`'s cycle columns:
//! `current_cycle` is a mailbox-level **`u32`** clock and the per-row
//! `temporal` stamp is **`[u64; N]`** — neither folds losslessly into a byte.
//! So the eventual cut-over must NOT retire those columns into TE. If it wants
//! the absolute cycle *in* the V3 register, that is a `u32` in the reserved
//! `[8..12]` bytes — a real `ColumnDescriptor`/`ENVELOPE_LAYOUT_VERSION`
//! decision — while TE stays the signed relative offset it is here. (Corrected
//! per `v3-envelope-auditor`; the earlier "current_cycle becomes foldable"
//! wording was wrong on width and would have invited a lossy truncation.)
//!
//! This is the wedge for the eventual `MailboxSoA` cut-over (its own gated
//! PR): the harness proves the lift is thinking-preserving on real emission
//! BEFORE any stored `edges_v3` column is proposed.

use causal_edge::edge::CausalEdge64;
use causal_edge::CausalEdgeV3;

/// Lift one driver-emitted `CausalEdge64` to `CausalEdgeV3`, stamping the
/// emission `cycle_index` into the TE (temporal) byte — the value the v2 edge
/// dropped. `target` is the Lokal reference to the node whose 6×256² CAM-PQ
/// facet IS this edge's SPO (the edge itself carries NO SPO bytes).
///
/// `cycle_index` is narrowed to the low 7 bits (0..=127) to fit the signed i8
/// TE byte. This is deliberately LOSSY: TE is a per-edge chain offset, not the
/// global clock. The mailbox keeps the absolute counters (`current_cycle: u32`,
/// per-row `temporal: [u64; N]`); those are NOT foldable into this byte and the
/// cut-over must not try (see the module-level width caveat).
pub fn lift_emitted(edge: CausalEdge64, target: u16, cycle_index: u16) -> CausalEdgeV3 {
    let mut v3 = CausalEdgeV3::from_v1(edge, target);
    v3.set_temporal((cycle_index & 0x7F) as i8);
    v3
}

/// Compare-thinking predicate: does the V3 lift of two edges reason to the
/// same syllogistic conclusion as the v1 edges, when the SPO is resolved from
/// each edge's own (subject, predicate, object) — i.e. the node CAM-PQ facet
/// the Lokal target points at? Returns `true` iff the dedup is
/// thinking-preserving for this pair.
pub fn thinking_preserved(a: CausalEdge64, b: CausalEdge64) -> bool {
    let want = a.syllogize(b);
    let va = CausalEdgeV3::from_v1(a, 0xA000);
    let vb = CausalEdgeV3::from_v1(b, 0xB000);
    let ra = va.rehydrate(a.s_idx(), a.p_idx(), a.o_idx());
    let rb = vb.rehydrate(b.s_idx(), b.p_idx(), b.o_idx());
    let got = ra.syllogize(rb);
    want.map(|s| (s.conclusion, s.figure)) == got.map(|s| (s.conclusion, s.figure))
}

#[cfg(test)]
mod tests {
    use super::*;
    use causal_edge::edge::{CausalEdge64, InferenceType};
    use causal_edge::pearl::CausalMask;
    use causal_edge::plasticity::PlasticityState;

    /// Reproduce the driver's [5] emission recipe (`driver.rs` 479-494) for a
    /// small set of synthetic strong hits. Each tuple is `(row, resonance,
    /// predicates, cycle_index)` — the fields the driver reads off a
    /// `ShaderHit`. The pack mirrors the shipping code EXACTLY.
    fn emit_like_driver(hits: &[(usize, f32, u8, u16)]) -> Vec<(CausalEdge64, u16)> {
        hits.iter()
            .filter(|(_, res, _, _)| *res >= 0.2)
            .map(|&(row, res, predicates, cycle)| {
                let f = (res.clamp(0.0, 1.0) * 255.0) as u8;
                let c = (res.clamp(0.0, 1.0) * 255.0) as u8;
                let s_palette = (row % 256) as u8;
                let o_palette = ((row / 4) % 256) as u8;
                let edge = CausalEdge64::pack(
                    s_palette,
                    0,
                    o_palette,
                    f,
                    c,
                    CausalMask::from_bits(predicates & 0x07),
                    0,
                    InferenceType::Deduction,
                    PlasticityState::from_bits(0),
                    cycle & 0xFFF,
                );
                (edge, cycle)
            })
            .collect()
    }

    /// (1) On the driver's real emission recipe, the V3 lift reasons
    /// IDENTICALLY to the v1 edge for every premise pair — the SPO dedup is
    /// thinking-preserving on live-shaped edges, not just the unit fixtures.
    #[test]
    fn v3_thinking_preserved_on_driver_emission() {
        let emitted = emit_like_driver(&[
            (10, 0.95, 0b111, 3),
            (40, 0.80, 0b111, 900),
            (12, 0.62, 0b101, 7),
            (200, 0.71, 0b110, 41),
            (5, 0.10, 0b111, 1), // below 0.2 threshold: filtered, like the driver
        ]);
        // the threshold filter matched the driver (4 survive, not 5)
        assert_eq!(
            emitted.len(),
            4,
            "resonance<0.2 hit must be filtered like driver"
        );

        let mut compared = 0;
        for &(a, _) in &emitted {
            for &(b, _) in &emitted {
                assert!(
                    thinking_preserved(a, b),
                    "V3 lift diverged from v1 reasoning on a driver-shaped pair"
                );
                compared += 1;
            }
        }
        assert_eq!(compared, 16, "expected 4x4 pair comparisons");
    }

    /// (2) V3 recovers the temporal the v2 edge DROPS — proven WITHOUT calling
    /// the deprecated `temporal()` accessor: two edges packed differing ONLY
    /// in the `temporal` arg are byte-identical under v2 (the write is a
    /// no-op), yet their V3 lifts carry distinct TE bytes.
    #[test]
    fn v3_carries_temporal_that_v2_drops() {
        // identical except the temporal arg (cycle 5 vs 900)
        let e_early = CausalEdge64::pack(
            10,
            0,
            20,
            200,
            190,
            CausalMask::from_bits(0b111),
            0,
            InferenceType::Deduction,
            PlasticityState::from_bits(0),
            5,
        );
        let e_late = CausalEdge64::pack(
            10,
            0,
            20,
            200,
            190,
            CausalMask::from_bits(0b111),
            0,
            InferenceType::Deduction,
            PlasticityState::from_bits(0),
            900,
        );
        // v2 drops temporal: the two edges are byte-identical (no temporal bits).
        // This crate always builds the v2 layout — its `causal-edge` dep enables
        // the default `causal-edge-v2-layout` feature and Cargo unification only
        // ADDS features, so no build of `cognitive-shader-driver` can turn it off.
        // The byte-identity is therefore an always-active invariant here; gate it
        // only if this crate ever sets `default-features = false` on causal-edge.
        assert_eq!(
            e_early.0, e_late.0,
            "v2 CausalEdge64 must drop temporal (bits reclaimed); if this fails the layout changed"
        );

        // V3 lift stamps the cycle into TE: the two lifts now DIFFER, and the
        // cycle round-trips through the edge itself (no standalone column).
        let v_early = lift_emitted(e_early, 0xC000, 5);
        let v_late = lift_emitted(e_late, 0xC000, 900);
        assert_ne!(
            v_early, v_late,
            "V3 lifts must differ once TE carries the cycle"
        );
        assert_eq!(v_early.temporal(), 5, "V3 recovered the early cycle in TE");
        assert_eq!(
            v_late.temporal(),
            (900 & 0x7F) as i8,
            "V3 recovered the late cycle in TE"
        );

        // and reasoning is still preserved for the pair (temporal is orthogonal
        // to the syllogism, which depends only on SPO + truth + mask).
        assert!(thinking_preserved(e_early, e_late));
    }

    /// The Lokal target round-trips through the lift (the edge references the
    /// SPO-bearing node, never re-encodes SPO).
    #[test]
    fn v3_lift_preserves_target_reference() {
        let (edge, cycle) = emit_like_driver(&[(37, 0.9, 0b111, 12)])[0];
        let v3 = lift_emitted(edge, 0xBEEF, cycle);
        assert_eq!(v3.target(), 0xBEEF);
        assert_eq!(v3.temporal(), 12);
    }
}
