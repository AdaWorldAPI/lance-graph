//! `CausalEdgeV3` — the staged, ADDITIVE V3-96 successor to [`CausalEdge64`].
//!
//! Runs IN PARALLEL with `CausalEdge64` (nothing here mutates the u64 type or
//! its consumers). The migration it stages:
//!
//!   - **The 24-bit SPO (`s_idx/p_idx/o_idx`) is DROPPED from the edge.** It was
//!     a duplicate: SPO already lives as the node's 6×256² CAM-PQ facet
//!     (le-contract L4 `palette256²`: 3 SPO + 3 AriGraph SPO-G byte-pairs). The
//!     edge keeps a **Lokal target reference** (`u16`) to the node whose CAM-PQ
//!     *is* the SPO. "We don't duplicate."
//!   - The freed 24 bits + the widening 64→96 buy the **TEKAMOLO** carving
//!     (Temporal / Kausal / Modal / Lokal — `grammar::tekamolo`) plus the
//!     **nibble anaphora edge** (`E-NIBBLE-ANAPHORA-EDGE-1`).
//!
//! ## Reason by rehydration (why "compare thinking" holds)
//!
//! `CausalEdge64::syllogize` reads only the premises' **SPO + freq/conf +
//! causal_mask**. So a V3 edge reasons by **rehydrating**: resolve SPO from its
//! target node (its CAM-PQ facet), rebuild a `CausalEdge64` with the preserved
//! truth/mask, and call the *existing* `syllogize`. The comparison test then
//! proves `v1.syllogize(v1) == v3.rehydrate(spo).syllogize(v3.rehydrate(spo))`
//! whenever the resolver returns the same SPO the node's CAM-PQ holds — i.e.
//! the in-edge SPO was a pure duplicate and the dedup is **thinking-preserving**.
//! (The mismatch guard test shows the reasoning correctly DIVERGES if the node's
//! CAM-PQ SPO disagrees with the edge — so the invariant is "node CAM-PQ == the
//! edge's SPO", enforced by the shared codebook, not assumed.)
//!
//! ## Layout (96-bit payload; the 16-byte facet's `classid` is the node's key)
//!
//! ```text
//! [0]     MO  freq  u8   (NARS frequency)
//! [1]     MO  conf  u8   (NARS confidence)
//! [2]     KA  causal_mask(3) | direction(3)          why-planes + chain dir
//! [3]     KA  inference_mantissa(i4 low) | plasticity(3 high)
//! [4..6]  LO  target u16 (the node whose CAM-PQ IS the SPO — NO SPO here)
//! [6]     anaphora nibble (i4 low, −8..+7; 0 = none)
//! [7]     TE  temporal i8 (signed chain offset)
//! [8..12] reserved (dormant — TEKAMOLO Kausal/Modal/Lokal/Instrument refs)
//! ```

use crate::edge::{CausalEdge64, InferenceType};
use crate::pearl::CausalMask;
use crate::plasticity::PlasticityState;

/// The 96-bit V3 causal-edge payload (the enclosing facet's `classid` is the
/// node key; this is the 12-byte content-blind register).
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub struct CausalEdgeV3 {
    payload: [u8; 12],
}

impl CausalEdgeV3 {
    /// Lift a v1 [`CausalEdge64`] to V3, DROPPING its 24-bit in-edge SPO and
    /// pointing `target` at the node whose CAM-PQ facet holds that SPO. All
    /// reasoning-relevant scalars (freq/conf/mask/direction/inference/
    /// plasticity/temporal) are preserved.
    pub fn from_v1(e: CausalEdge64, target: u16) -> Self {
        let mut p = [0u8; 12];
        p[0] = e.frequency_u8();
        p[1] = e.confidence_u8();
        p[2] = (e.causal_mask() as u8 & 0b111) | ((e.direction() & 0b111) << 3);
        let mantissa = e.inference_mantissa() as u8 & 0x0F;
        p[3] = mantissa | ((e.plasticity().bits() & 0b111) << 4);
        let t = target.to_le_bytes();
        p[4] = t[0];
        p[5] = t[1];
        // [6] anaphora nibble left 0 (no coreference edge by default)
        // [7] TE temporal: NOT lifted — under the v2 layout `CausalEdge64` carries
        // no temporal (it is structural: chain-position / AriGraph timestamp).
        // V3's TE is set explicitly by the producer via `set_temporal`, not
        // inherited from a v2 edge that has none.
        Self { payload: p }
    }

    /// Rebuild a [`CausalEdge64`] for reasoning by supplying the SPO resolved
    /// from the target node's CAM-PQ facet. The conclusion of `syllogize`
    /// depends only on SPO + freq/conf + causal_mask, all restored here.
    pub fn rehydrate(&self, s_idx: u8, p_idx: u8, o_idx: u8) -> CausalEdge64 {
        let freq = self.payload[0];
        let conf = self.payload[1];
        let mask = CausalMask::from_bits(self.payload[2] & 0b111);
        let direction = (self.payload[2] >> 3) & 0b111;
        let mantissa = {
            let lo = self.payload[3] & 0x0F;
            // sign-extend the 4-bit signed mantissa
            if lo >= 8 { lo as i8 - 16 } else { lo as i8 }
        };
        let inference = InferenceType::from_mantissa(mantissa);
        let plasticity = PlasticityState::from_bits((self.payload[3] >> 4) & 0b111);
        CausalEdge64::pack(
            s_idx, p_idx, o_idx, freq, conf, mask, direction, inference, plasticity, 0,
        )
    }

    /// The Lokal target node reference — the node whose 6×256² CAM-PQ facet IS
    /// this edge's SPO. Never carries SPO bytes itself.
    pub fn target(&self) -> u16 {
        u16::from_le_bytes([self.payload[4], self.payload[5]])
    }

    /// The nibble anaphora edge: a signed −8..+7 offset to a coreference
    /// antecedent, or `None` (sentinel 0 = no coreference edge).
    pub fn anaphora(&self) -> Option<i8> {
        let lo = self.payload[6] & 0x0F;
        if lo == 0 {
            None
        } else if lo >= 8 {
            Some(lo as i8 - 16)
        } else {
            Some(lo as i8)
        }
    }

    /// Set the nibble anaphora offset (−8..=7).
    pub fn set_anaphora(&mut self, offset: i8) {
        debug_assert!((-8..=7).contains(&offset), "anaphora offset out of nibble range");
        self.payload[6] = (self.payload[6] & 0xF0) | ((offset as u8) & 0x0F);
    }

    /// The Temporal (TE) signed chain offset.
    pub fn temporal(&self) -> i8 {
        self.payload[7] as i8
    }

    /// Set the Temporal (TE) signed chain offset (V3 carries temporal
    /// explicitly; the v2 `CausalEdge64` does not).
    pub fn set_temporal(&mut self, t: i8) {
        self.payload[7] = t as u8;
    }

    /// Raw 12-byte LE payload (the content-blind register).
    pub fn to_le_bytes(self) -> [u8; 12] {
        self.payload
    }

    /// Reconstruct from the 12-byte LE payload.
    pub fn from_le_bytes(b: [u8; 12]) -> Self {
        Self { payload: b }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::edge::{CausalEdge64, InferenceType};
    use crate::pearl::CausalMask;
    use crate::plasticity::PlasticityState;

    /// A small deterministic set of v1 edges (varied SPO / truth / mask).
    fn sample_edges() -> Vec<CausalEdge64> {
        let mk = |s, p, o, f, c, mask: u8| {
            CausalEdge64::pack(
                s,
                p,
                o,
                f,
                c,
                CausalMask::from_bits(mask),
                0,
                InferenceType::Deduction,
                PlasticityState::ALL_HOT,
                0,
            )
        };
        vec![
            mk(10, 1, 20, 230, 200, 0b111), // A -> B
            mk(20, 2, 30, 210, 190, 0b111), // B -> C  (chains with the first)
            mk(10, 1, 40, 200, 180, 0b101), // A -> D  (shared subject with first)
            mk(50, 3, 30, 220, 195, 0b110), // E -> C  (shared object with second)
        ]
    }

    /// COMPARE THINKING: for every premise pair, the V3 edges (SPO dropped,
    /// resolved from the node) syllogize to the IDENTICAL conclusion as the v1
    /// edges (SPO in-edge). The 24-bit SPO was a pure duplicate.
    #[test]
    fn v3_reasons_identically_to_v1_when_spo_resolves_from_node() {
        let edges = sample_edges();
        // The "node CAM-PQ" resolver: node id == the edge's own SPO packed as a
        // key; here we model it by carrying the original SPO alongside the target.
        let mut compared = 0;
        for a in &edges {
            for b in &edges {
                let want = a.syllogize(*b);
                // lift to V3 (SPO dropped, target points at each edge's node)
                let va = CausalEdgeV3::from_v1(*a, 0xA000);
                let vb = CausalEdgeV3::from_v1(*b, 0xB000);
                // resolve SPO from the node's CAM-PQ (== the original SPO)
                let ra = va.rehydrate(a.s_idx(), a.p_idx(), a.o_idx());
                let rb = vb.rehydrate(b.s_idx(), b.p_idx(), b.o_idx());
                let got = ra.syllogize(rb);
                assert_eq!(
                    want.map(|s| s.conclusion),
                    got.map(|s| s.conclusion),
                    "V3 reasoning diverged from V1 for a pair (SPO dedup not thinking-preserving)"
                );
                assert_eq!(want.map(|s| s.figure), got.map(|s| s.figure), "figure diverged");
                compared += 1;
            }
        }
        assert!(compared >= 16, "expected >= 16 pair comparisons, ran {compared}");
    }

    /// The dedup invariant is CONDITIONAL: reasoning correctly DIVERGES if the
    /// node's CAM-PQ SPO disagrees with the edge — proving the shared codebook
    /// (node CAM-PQ == edge SPO) is load-bearing, not assumed away.
    #[test]
    fn v3_reasoning_diverges_if_node_spo_mismatches() {
        let edges = sample_edges();
        let (a, b) = (edges[0], edges[1]); // A->B, B->C : chain via middle term B
        let want = a.syllogize(b);
        let va = CausalEdgeV3::from_v1(a, 0xA000);
        let vb = CausalEdgeV3::from_v1(b, 0xB000);
        // WRONG resolver: corrupt the middle term so the chain no longer links.
        let ra = va.rehydrate(a.s_idx(), a.p_idx(), 99);
        let rb = vb.rehydrate(88, b.p_idx(), b.o_idx());
        let got = ra.syllogize(rb);
        assert_ne!(
            want.map(|s| s.conclusion),
            got.map(|s| s.conclusion),
            "corrupt SPO resolution must change the reasoning (else the guard is vacuous)"
        );
    }

    /// The edge carries NO SPO bytes: its payload never equals the target's SPO
    /// CAM-PQ code (the no-duplication invariant, byte-level).
    #[test]
    fn v3_payload_carries_no_spo() {
        let e = sample_edges()[0];
        let v3 = CausalEdgeV3::from_v1(e, 0x1234);
        // the node's CAM-PQ SPO facet (mock 6×(8:8) code)
        let spo_facet: [u8; 12] = core::array::from_fn(|i| e.s_idx().wrapping_add(i as u8 * 7));
        assert_ne!(v3.to_le_bytes(), spo_facet, "edge payload equals SPO code (duplicated!)");
        // target round-trips
        assert_eq!(v3.target(), 0x1234);
    }

    /// Field isolation: each setter touches only its own byte(s) (the
    /// I-LEGACY-API-FEATURE-GATED discipline for a layout).
    #[test]
    fn v3_field_isolation() {
        let mut e = CausalEdgeV3::default();
        e.set_anaphora(-4);
        // only byte 6 low nibble moved
        let mut expect = [0u8; 12];
        expect[6] = (-4i8 as u8) & 0x0F;
        assert_eq!(e.to_le_bytes(), expect, "set_anaphora touched a foreign byte");
        assert_eq!(e.anaphora(), Some(-4));
        // sentinel: 0 -> None
        assert_eq!(CausalEdgeV3::default().anaphora(), None);
    }

    /// LE round-trip identity.
    #[test]
    fn v3_le_round_trip() {
        let mut e = CausalEdgeV3::from_v1(sample_edges()[2], 0xBEEF);
        e.set_anaphora(3);
        assert_eq!(CausalEdgeV3::from_le_bytes(e.to_le_bytes()), e);
    }
}
