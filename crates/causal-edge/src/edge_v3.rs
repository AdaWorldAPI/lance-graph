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
//! ## Layout — a packed EDGE REGISTER, NOT a slot-pure §3 facet
//!
//! `CausalEdgeV3` is the V3-96 successor of the `MaterializedEdges` packed
//! edge register (tenants.md §2 #2, which stores `CausalEdge64` as a packed
//! register). Its field positions are hard-coded HERE, on purpose — it is an
//! *edge register*, deliberately NOT one of the le-contract §3 L1–L8
//! ClassView-selected facet payloads (`6×8:8` / `4×8:8:8` / …). Do not read
//! this as a content-blind facet: it is a typed edge register whose carving is
//! its own contract (the same way `CausalEdge64`'s u64 bit-layout is).
//!
//! Layout (96-bit register):
//!
//! ```text
//! [0]     MO  freq  u8   (NARS frequency)
//! [1]     MO  conf  u8   (NARS confidence)
//! [2]     KA  causal_mask(3) | direction(3)          why-planes + chain dir
//! [3]     KA  inference_mantissa(i4 low) | plasticity(3 high)
//! [4..6]  LO  target u16 (the node whose CAM-PQ IS the SPO — NO SPO here)
//! [6]     anaphora nibble (i4 low, −8..+7; 0 = none)
//! [7]     TE  temporal i8 (signed chain offset)
//! [8]     w_slot(6 low) | truth/topology RAW(2 high)     ← CE64-v2 preserve
//! [9]     spare/ReasoningBand RAW(3 low) | reserved(5 high)
//! [10..12] reserved (dormant — TEKAMOLO Kausal/Modal/Instrument refs)
//! ```
//!
//! ## CE64 → V3 → CE64 is LOSSLESS (the conversion contract)
//!
//! Every meaningful CE64-v2 field survives the round trip byte-exact, with
//! exactly TWO documented exclusions:
//!
//!   1. **The 24-bit in-edge SPO** — intentionally deduplicated (it lives in
//!      the target node's CAM-PQ facet). Supply the SAME resolved SPO to
//!      [`CausalEdgeV3::rehydrate`] and it returns bit-identical.
//!   2. **The deprecated v2 `temporal` field** — not valid CE64-v2 state at
//!      all (bits 52..63 were reclaimed for plasticity[2]/W/truth/spare). It is
//!      NOT mapped into V3's TE byte: V3 `temporal` is an INDEPENDENT signed
//!      relative chain offset the producer sets explicitly, never inherited.
//!
//! Under the v2 layout the 64 bits are fully partitioned — S/P/O, freq, conf,
//! causal_mask, direction, inference i4, plasticity, w_slot, truth, spare — so
//! "every field preserved" IS "every bit preserved" once the SPO is resupplied.
//!
//! ### The signed mantissa is carried RAW, never through `InferenceType`
//!
//! [`InferenceType`] is a **lossy compatibility projection** of the 4-bit
//! signed mantissa: `to_mantissa(from_mantissa(m)) != m` for **8 of the 16**
//! states (−8, −7, −5, −4, −3, −2, 0, +3 — e.g. `−2 → Abduction → −1`,
//! `+3 → Synthesis → +5`, and the `pack_v2` default `0 → Deduction → +1`).
//! Routing the mantissa through the enum on rehydration silently rewrote half
//! the state space, so [`CausalEdgeV3::rehydrate`] restores the RAW nibble with
//! [`CausalEdge64::set_inference_mantissa`] after construction; the
//! `InferenceType` argument to `pack` is a throwaway placeholder.
//!
//! ### RAW bits, not upgraded provenance
//!
//! `w_slot` / truth / spare are preserved as RAW ORDINALS. Copying a CE64
//! topology/truth ordinal `01` into V3 means "ordinal 01 preserved" — it is NOT
//! an assertion that `IndirectKnown` (or `Solid`) is now source-authoritative
//! for that row. Which lens the ordinal was written through is the producer's
//! knowledge, not the conversion's; see [`crate::layout::CausalTopology`].

use crate::edge::{CausalEdge64, InferenceType};
use crate::pearl::CausalMask;
use crate::plasticity::PlasticityState;

/// The 96-bit V3 causal-edge payload (the enclosing facet's `classid` is the
/// node key; this is the 12-byte content-blind register).
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub struct CausalEdgeV3 {
    payload: [u8; 12],
}

// The V3-96 register is EXACTLY 12 bytes (96 bits): `classid(4) | payload(12)`
// = the canonical 16-byte facet, the payload half. A width drift here silently
// corrupts every stored edge (I-LEGACY-API-FEATURE-GATED layout discipline).
const _: () = assert!(core::mem::size_of::<CausalEdgeV3>() == 12);

impl CausalEdgeV3 {
    /// Lift a [`CausalEdge64`] to V3, DROPPING its 24-bit in-edge SPO and
    /// pointing `target` at the node whose CAM-PQ facet holds that SPO.
    ///
    /// **Every other meaningful v2 field is preserved**: freq, conf,
    /// causal_mask, direction, the RAW signed inference mantissa, plasticity,
    /// w_slot, the truth/topology 2-bit ordinal, and the spare/band 3-bit
    /// ordinal. Paired with [`Self::rehydrate`] this is a bit-exact round trip
    /// once the same SPO is resupplied — see the module doc.
    ///
    /// **Not lifted:** the deprecated v2 `temporal` (not valid CE64-v2 state;
    /// V3's TE is an independent producer-set offset).
    ///
    /// **Provenance:** copying the v2 tail (`w_slot`/`truth`/`spare`) ASSERTS
    /// that its producer stamped it deliberately. When you do not know that,
    /// use [`Self::from_v1_tail_unstated`] — see its doc for why the signature
    /// alone cannot tell the two cases apart (D-ACR-7 BLOCK-1).
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
        // inherited from a v2 edge that has none. The DEPRECATED v2 `temporal()`
        // composite (bits 52..63) is not valid CE64-v2 state and is never mapped
        // here — doing so would alias plasticity[2]/W/truth/spare into TE.
        // [8]/[9]: the CE64-v2 tail — w_slot(6) + truth(2) + spare(3). RAW
        // ordinals, no lens interpretation (see the module doc). Under the v1
        // layout every one of these accessors is a documented zero stub, so this
        // writes zeros and the round trip stays exact there too.
        p[8] = (e.w_slot() & 0x3F) | ((e.truth_raw() & 0b11) << 6);
        p[9] = e.spare() & 0b111;
        Self { payload: p }
    }

    /// Lift a [`CausalEdge64`] whose v2 tail is **UNSTATED** — bytes `[8]`/`[9]`
    /// are zeroed instead of copied.
    ///
    /// # Why this exists (D-ACR-7 BLOCK-1)
    ///
    /// [`Self::from_v1`] copies `w_slot`/`truth`/`spare` raw. Under the **v1**
    /// layout that is provably safe: those accessors are documented zero stubs
    /// (`edge.rs`, `truth_raw` and `spare` return `0`), so the copy writes
    /// zeros anyway. Under **v2** they read real bits — and nothing in the
    /// signature says whether a producer deliberately stamped them or whether
    /// they are residue from a source that never meant anything by them.
    ///
    /// The reading contract that consumes these bits
    /// (`lance_graph_contract::band_reading`) resolves that gap by making
    /// provenance a **caller assertion**: unstated means refuse. This
    /// constructor is that assertion's honest half on the WRITE side — a lift
    /// that declines to claim the tail. A consumer then reads the zero-fallback
    /// (`Trust` / band `Absent`) rather than a plausible wrong ordinal.
    ///
    /// **Pick by what you know**, not by convenience:
    ///
    /// | you know | use |
    /// |---|---|
    /// | the producer stamped the tail | [`Self::from_v1`] |
    /// | you do not know, or the source predates the stamp | **this** |
    ///
    /// Every other field is lifted exactly as [`Self::from_v1`] lifts it.
    /// Note that [`Self::rehydrate`] on the result is therefore **not** a
    /// bit-exact round trip of the source: the tail was deliberately dropped.
    pub fn from_v1_tail_unstated(e: CausalEdge64, target: u16) -> Self {
        let mut v3 = Self::from_v1(e, target);
        v3.payload[8] = 0;
        v3.payload[9] = 0;
        v3
    }

    /// NARS frequency (u8, `f = val/255`) — byte 0.
    pub fn frequency(&self) -> u8 {
        self.payload[0]
    }

    /// NARS confidence (u8, `c = val/255`) — byte 1.
    pub fn confidence(&self) -> u8 {
        self.payload[1]
    }

    /// Pearl 2³ causal mask — byte 2, low 3 bits.
    pub fn causal_mask(&self) -> CausalMask {
        CausalMask::from_bits(self.payload[2] & 0b111)
    }

    /// Direction triad (3 bits, sign(dim0) per S,P,O) — byte 2, bits 3..6.
    pub fn direction(&self) -> u8 {
        (self.payload[2] >> 3) & 0b111
    }

    /// The RAW 4-bit signed inference mantissa (−8..=7) — byte 3, low nibble.
    ///
    /// This is the mantissa itself, NOT [`InferenceType`]. The enum is a lossy
    /// compatibility projection (8 of 16 states do not survive
    /// `to_mantissa(from_mantissa(m))`); the register is the ground truth.
    /// Sign is chain direction, magnitude is the NARS base rule index.
    pub fn inference_mantissa(&self) -> i8 {
        let lo = self.payload[3] & 0x0F;
        // sign-extend the 4-bit signed mantissa
        if lo >= 8 {
            lo as i8 - 16
        } else {
            lo as i8
        }
    }

    /// Plasticity flags (hot/cold per S,P,O) — byte 3, high 3 bits.
    pub fn plasticity(&self) -> PlasticityState {
        PlasticityState::from_bits((self.payload[3] >> 4) & 0b111)
    }

    /// Preserved CE64-v2 W-slot: witness corpus root handle (6-bit, 0..=63).
    /// Byte 8, low 6 bits. 0 = no corpus anchor.
    pub fn w_slot(&self) -> u8 {
        self.payload[8] & 0x3F
    }

    /// Preserved CE64-v2 truth/topology register: the RAW 2-bit ordinal
    /// (0..=3) — byte 8, high 2 bits.
    ///
    /// RAW on purpose. CE64 reads these same two bits through two lenses
    /// ([`crate::layout::TrustTexture`] epistemic, [`crate::layout::CausalTopology`]
    /// factual); which one the producer meant is not recoverable from the
    /// register, so V3 carries the ordinal and asserts nothing about it.
    /// Preserving ordinal `01` means "ordinal 01 preserved", never
    /// "`IndirectKnown` is now source-authoritative".
    pub fn truth_raw(&self) -> u8 {
        (self.payload[8] >> 6) & 0b11
    }

    /// Preserved CE64-v2 spare/`ReasoningBand` register: the RAW 3-bit ordinal
    /// (0..=7) — byte 9, low 3 bits. RAW for the same reason as
    /// [`Self::truth_raw`]: one register, two lenses, no provenance upgrade.
    pub fn spare_raw(&self) -> u8 {
        self.payload[9] & 0b111
    }

    /// Rebuild a [`CausalEdge64`] for reasoning by supplying the SPO resolved
    /// from the target node's CAM-PQ facet. The conclusion of `syllogize`
    /// depends only on SPO + freq/conf + causal_mask — but this restores the
    /// FULL v2 register, so the round trip
    /// `CausalEdge64 → from_v1 → rehydrate(same SPO)` is **bit-identical**
    /// (see the module doc's conversion contract for the two exclusions).
    ///
    /// The [`InferenceType`] handed to [`CausalEdge64::pack`] is a THROWAWAY
    /// placeholder: `pack` writes `inference.to_mantissa()`, and routing the
    /// stored mantissa through `InferenceType::from_mantissa` first would
    /// rewrite 8 of its 16 states. The raw nibble is restored immediately
    /// after with [`CausalEdge64::set_inference_mantissa`].
    pub fn rehydrate(&self, s_idx: u8, p_idx: u8, o_idx: u8) -> CausalEdge64 {
        let mantissa = self.inference_mantissa();
        let mut edge = CausalEdge64::pack(
            s_idx,
            p_idx,
            o_idx,
            self.frequency(),
            self.confidence(),
            self.causal_mask(),
            self.direction(),
            // placeholder ONLY — overwritten by set_inference_mantissa below.
            InferenceType::Deduction,
            self.plasticity(),
            0,
        );
        edge.set_inference_mantissa(mantissa);
        edge.set_w_slot(self.w_slot());
        // `set_truth` has no raw-u8 form, so the 2-bit ordinal goes through
        // `TrustTexture`. That is SAFE where `InferenceType` was not, and the
        // difference is the whole point: `from_bits_2`/`to_bits_2` is a total
        // BIJECTION on 0..=3 (four ordinals, four variants, discriminants
        // 0,1,2,3), whereas `InferenceType` maps 16 mantissa states onto 8
        // variants and cannot be injective. Pinned by
        // `trust_texture_bits_2_is_a_bijection_unlike_inference_type` — if a
        // variant is ever added or reordered, that test fails before this does.
        edge.set_truth(crate::layout::TrustTexture::from_bits_2(self.truth_raw()));
        edge.set_spare(self.spare_raw());
        edge
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
        debug_assert!(
            (-8..=7).contains(&offset),
            "anaphora offset out of nibble range"
        );
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
                assert_eq!(
                    want.map(|s| s.figure),
                    got.map(|s| s.figure),
                    "figure diverged"
                );
                compared += 1;
            }
        }
        assert!(
            compared >= 16,
            "expected >= 16 pair comparisons, ran {compared}"
        );
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
        assert_ne!(
            v3.to_le_bytes(),
            spo_facet,
            "edge payload equals SPO code (duplicated!)"
        );
        // target round-trips
        assert_eq!(v3.target(), 0x1234);
    }

    /// Field isolation matrix (the I-LEGACY-API-FEATURE-GATED discipline for a
    /// layout): each setter, exercised from a FULLY NON-ZERO baseline, must
    /// touch ONLY its own byte(s) and leave every other byte bit-identical.
    /// A zeroed baseline can hide a setter that ORs into a foreign byte's set
    /// bits, so the baseline is deliberately all-`0xAB` except where a field
    /// setter is expected to write.
    #[test]
    fn v3_field_isolation() {
        // ── set_anaphora: only byte 6's LOW nibble may move ──
        {
            let mut e = CausalEdgeV3::from_le_bytes([0xABu8; 12]);
            let before = e.to_le_bytes();
            e.set_anaphora(-4);
            let after = e.to_le_bytes();
            for (i, (&b, &a)) in before.iter().zip(after.iter()).enumerate() {
                if i == 6 {
                    // low nibble becomes -4; high nibble (0xA) preserved
                    assert_eq!(a, 0xA0 | ((-4i8 as u8) & 0x0F), "byte 6 wrong");
                } else {
                    assert_eq!(b, a, "set_anaphora corrupted byte {i}");
                }
            }
            assert_eq!(e.anaphora(), Some(-4));
        }

        // ── set_temporal: only byte 7 may move ──
        {
            let mut e = CausalEdgeV3::from_le_bytes([0xABu8; 12]);
            let before = e.to_le_bytes();
            e.set_temporal(-9);
            let after = e.to_le_bytes();
            for (i, (&b, &a)) in before.iter().zip(after.iter()).enumerate() {
                if i == 7 {
                    assert_eq!(a, -9i8 as u8, "byte 7 (temporal) wrong");
                } else {
                    assert_eq!(b, a, "set_temporal corrupted byte {i}");
                }
            }
            assert_eq!(e.temporal(), -9);
        }

        // ── the two setters compose without cross-contamination ──
        {
            let mut e = CausalEdgeV3::from_le_bytes([0xABu8; 12]);
            e.set_anaphora(5);
            e.set_temporal(-2);
            assert_eq!(e.anaphora(), Some(5), "temporal write clobbered anaphora");
            assert_eq!(e.temporal(), -2, "anaphora write clobbered temporal");
        }

        // ── from a zeroed base, set_anaphora writes exactly one nibble ──
        let mut z = CausalEdgeV3::default();
        z.set_anaphora(-4);
        let mut expect = [0u8; 12];
        expect[6] = (-4i8 as u8) & 0x0F;
        assert_eq!(
            z.to_le_bytes(),
            expect,
            "set_anaphora touched a foreign byte"
        );
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

    // ══════════════════════════════════════════════════════════════════════
    //  CE64 → V3 → CE64 LOSSLESSNESS
    //
    //  A LOW-LEVEL conversion-correctness suite, deliberately separate from
    //  the planner-level Stage-2.6 parity harness
    //  (`lance-graph-planner::cache::stage26_v3_parity`). The planner harness
    //  proves the ENGINE behaves identically through a V3 leg; this suite
    //  proves the CONVERSION preserves the register. Neither subsumes the
    //  other: the planner leg only ever carries `InferenceType::Deduction`
    //  (mantissa +1, one of the 8 states that happens to survive the old lossy
    //  path), so it was structurally blind to the mantissa defect below.
    // ══════════════════════════════════════════════════════════════════════

    /// `InferenceType` is a LOSSY projection of the 4-bit mantissa, and this
    /// is the measurement that says so — the premise the raw-carry fix rests
    /// on. If a future edit makes the enum injective this test fails and the
    /// module doc's "8 of 16" claim must be re-measured, not re-asserted.
    #[test]
    fn inference_type_is_a_lossy_projection_of_the_mantissa() {
        let lossy: Vec<i8> = (-8i8..=7)
            .filter(|&m| InferenceType::from_mantissa(m).to_mantissa() != m)
            .collect();
        assert_eq!(
            lossy,
            vec![-8, -7, -5, -4, -3, -2, 0, 3],
            "the InferenceType loss set moved; re-measure the module doc"
        );
        // anti-vacuity: the projection is not a blanket failure either — half
        // the states DO survive, which is exactly why the defect hid.
        assert_eq!(16 - lossy.len(), 8, "expected 8 surviving states");
    }

    /// Unlike `InferenceType`, `TrustTexture`'s 2-bit codec IS a total
    /// bijection — which is what licenses `rehydrate` routing the truth
    /// ordinal through it. Pinned so an added/reordered variant fails HERE,
    /// with a message naming the cause, rather than as a silent parity drift.
    #[test]
    fn trust_texture_bits_2_is_a_bijection_unlike_inference_type() {
        for raw in 0u8..=3 {
            assert_eq!(
                crate::layout::TrustTexture::from_bits_2(raw).to_bits_2(),
                raw,
                "TrustTexture is no longer a bijection on 2 bits; rehydrate's \
                 truth carry must switch to a raw setter"
            );
        }
    }

    /// REQUIREMENT 1 — the raw signed mantissa survives CE64 → V3 → CE64 for
    /// ALL 16 states.
    ///
    /// The old implementation routed the nibble through
    /// `InferenceType::from_mantissa(...) → pack(...) → to_mantissa()` and
    /// silently rewrote 8 of them (see the loss-set test above), including the
    /// `pack_v2` default `0 → +1`.
    ///
    /// DISABLE-RUN (verified red): restore the old body of `rehydrate` —
    /// pass `InferenceType::from_mantissa(mantissa)` to `pack` and drop the
    /// `set_inference_mantissa` call — and this fails on m = −8 with
    /// `rehydrated mantissa: left: 1, right: -8`.
    #[test]
    fn mantissa_round_trips_raw_for_all_16_states() {
        for m in -8i8..=7 {
            // Build the V3 register directly so the V3 half of the assertion
            // is layout-independent (it holds under v1 and v2 alike).
            let mut bytes = [0u8; 12];
            bytes[3] = (m as u8) & 0x0F;
            let v3 = CausalEdgeV3::from_le_bytes(bytes);
            assert_eq!(v3.inference_mantissa(), m, "V3 accessor lost m={m}");

            #[cfg(feature = "causal-edge-v2-layout")]
            {
                // and the full CE64 → V3 → CE64 leg
                let src = CausalEdge64::pack_v2(
                    7,
                    8,
                    9,
                    200,
                    180,
                    CausalMask::from_bits(0b101),
                    0b010,
                    PlasticityState::ALL_HOT,
                )
                .with_inference_mantissa(m);
                assert_eq!(src.inference_mantissa(), m, "source lost m={m}");
                let got = CausalEdgeV3::from_v1(src, 0x1234).rehydrate(7, 8, 9);
                assert_eq!(got.inference_mantissa(), m, "rehydrated mantissa");
            }
        }
    }

    /// The four states the brief named explicitly, each one a state the OLD
    /// implementation provably lost, asserted individually so a failure names
    /// the value rather than an index. Kept separate from the exhaustive sweep
    /// above: the sweep proves totality, this proves the regressions.
    #[cfg(feature = "causal-edge-v2-layout")]
    #[test]
    fn named_mantissa_regressions_minus2_plus3_minus4_minus5() {
        for (m, old_would_give) in [(-2i8, -1i8), (3, 5), (-4, 4), (-5, 5)] {
            // the defect, restated as a measurement rather than a memory
            assert_eq!(
                InferenceType::from_mantissa(m).to_mantissa(),
                old_would_give,
                "the old lossy path's output for m={m} moved"
            );
            let src = CausalEdge64::pack_v2(
                1,
                2,
                3,
                255,
                255,
                CausalMask::from_bits(0b111),
                0b111,
                PlasticityState::ALL_HOT,
            )
            .with_inference_mantissa(m);
            let got = CausalEdgeV3::from_v1(src, 0xFFFF).rehydrate(1, 2, 3);
            assert_eq!(got.inference_mantissa(), m, "m={m} not preserved");
            assert_ne!(
                got.inference_mantissa(),
                old_would_give,
                "m={m} still lands on the old lossy value"
            );
        }
    }

    /// One full-parity case:
    /// `(s, p, o, freq, conf, mask, dir, mantissa, w_slot, truth, spare)`.
    #[cfg(feature = "causal-edge-v2-layout")]
    type ParityCase = (u8, u8, u8, u8, u8, u8, u8, i8, u8, u8, u8);

    /// REQUIREMENT 3 — FULL FIELD PARITY over varied NON-ZERO edges.
    ///
    /// Every meaningful CE64-v2 field is asserted after CE64 → V3 → CE64 with
    /// the SAME resolved SPO resupplied. Under v2 the 64 bits are fully
    /// partitioned (S/P/O · freq · conf · mask · direction · mantissa ·
    /// plasticity · w_slot · truth · spare), so this is also a whole-register
    /// bit-identity check — asserted as such at the end.
    ///
    /// NOT compared: the deprecated v2 `temporal()`. It is not valid CE64-v2
    /// state (bits 52..63 are the reclaim zone) and V3's TE is an independent
    /// producer-set offset, never a lift of it.
    ///
    /// DISABLE-RUN (verified red, each independently): drop any one of the
    /// `set_w_slot` / `set_truth` / `set_spare` / `set_inference_mantissa`
    /// carries in `rehydrate`, or the `p[8]`/`p[9]` writes in `from_v1`.
    #[cfg(feature = "causal-edge-v2-layout")]
    #[test]
    fn full_v2_field_parity_across_ce64_v3_ce64() {
        // varied, deliberately non-zero, and spanning the ordinal ranges: all
        // 4 truth ordinals, w_slot at both ends of its 6 bits, spare across
        // its 3, mantissa on both signs.
        let cases: [ParityCase; 6] = [
            // s,  p,  o,  freq, conf, mask, dir, mantissa, w_slot, truth, spare
            (10, 1, 20, 230, 200, 0b111, 0b101, -2, 63, 3, 7),
            (20, 2, 30, 210, 190, 0b101, 0b010, 3, 1, 1, 1),
            (255, 254, 253, 255, 255, 0b111, 0b111, 7, 62, 2, 6),
            (1, 0, 255, 1, 254, 0b001, 0b100, -8, 33, 0, 5),
            (50, 3, 30, 220, 195, 0b110, 0b001, 0, 0, 2, 0),
            (99, 88, 77, 128, 127, 0b010, 0b011, -6, 21, 3, 4),
        ];
        for (s, p, o, freq, conf, mask, dir, mant, w, truth, spare) in cases {
            let src = CausalEdge64::pack_v2(
                s,
                p,
                o,
                freq,
                conf,
                CausalMask::from_bits(mask),
                dir,
                PlasticityState::from_bits(0b101),
            )
            .with_inference_mantissa(mant)
            .with_w_slot(w)
            .with_truth(crate::layout::TrustTexture::from_bits_2(truth))
            .with_spare(spare);

            // the source really carries what the case says (anti-vacuity: a
            // parity test over an all-zero source proves nothing)
            assert_eq!(src.w_slot(), w, "fixture w_slot");
            assert_eq!(src.truth_raw(), truth, "fixture truth");
            assert_eq!(src.spare(), spare, "fixture spare");
            assert_eq!(src.inference_mantissa(), mant, "fixture mantissa");

            let v3 = CausalEdgeV3::from_v1(src, 0xC0DE);
            let got = v3.rehydrate(s, p, o); // the SAME resolved SPO

            assert_eq!((got.s_idx(), got.p_idx(), got.o_idx()), (s, p, o), "SPO");
            assert_eq!(got.frequency_u8(), freq, "freq");
            assert_eq!(got.confidence_u8(), conf, "conf");
            assert_eq!(got.causal_mask() as u8, mask, "causal_mask");
            assert_eq!(got.direction(), dir, "direction");
            assert_eq!(got.inference_mantissa(), mant, "inference i4");
            assert_eq!(got.plasticity().bits(), 0b101, "plasticity");
            assert_eq!(got.w_slot(), w, "w_slot");
            assert_eq!(got.truth_raw(), truth, "truth/topology raw");
            assert_eq!(got.spare(), spare, "spare/band raw");

            // …and because v2 partitions all 64 bits, field parity IS bit
            // parity. This single line would catch a future field we forgot to
            // enumerate above — which the per-field asserts alone cannot.
            assert_eq!(
                got, src,
                "whole-register identity failed (a v2 field is unaccounted for)"
            );
        }
    }

    /// The V3 tail bytes are where the preserved CE64 fields live, and nothing
    /// else may leak into them. Complements `v3_field_isolation` above, which
    /// covers the two V3-native setters.
    #[cfg(feature = "causal-edge-v2-layout")]
    #[test]
    fn preserved_tail_occupies_exactly_bytes_8_and_9() {
        let src = CausalEdge64::pack_v2(
            1,
            2,
            3,
            4,
            5,
            CausalMask::from_bits(0),
            0,
            PlasticityState::from_bits(0),
        )
        .with_w_slot(0x2A) // 0b101010
        .with_truth(crate::layout::TrustTexture::from_bits_2(0b11))
        .with_spare(0b101);
        let bytes = CausalEdgeV3::from_v1(src, 0).to_le_bytes();
        assert_eq!(bytes[8], 0b1110_1010, "byte 8 = truth(2)<<6 | w_slot(6)");
        assert_eq!(bytes[9], 0b0000_0101, "byte 9 = spare(3), high 5 reserved");
        // bytes 10..12 stay dormant — the preserve must not creep into them
        assert_eq!(&bytes[10..12], &[0u8, 0], "reserved tail was written");
        // and TE was NOT inherited from the deprecated v2 temporal composite
        assert_eq!(bytes[7], 0, "TE must not be lifted from v2 temporal");
    }

    /// G10b (D-ACR-7 ratified gate, hosted HERE because both crates are
    /// zero-dep): the lift preserves the truth/spare ordinals — the module doc
    /// claims the round trip is byte-exact, and this is the test that asserts
    /// the truth/spare half of that claim through the ACCESSORS (not just the
    /// raw bytes), under BOTH lens vocabularies over the same unchanged bits.
    #[cfg(feature = "causal-edge-v2-layout")]
    #[test]
    fn g10b_lift_preserves_truth_and_spare_ordinals_under_both_lenses() {
        use crate::layout::{CausalTopology, ReasoningBand};

        // Firing half: nonzero ordinals in both registers, set via the
        // TOPOLOGY/BAND lens (the readings D-ACR-7 declares per class).
        let e = sample_edges()[0]
            .with_topology(CausalTopology::Unknown) // ordinal 0b11
            .with_reasoning_band(ReasoningBand::Transcendent); // ordinal 0b111
        let v3 = CausalEdgeV3::from_v1(e, 7);
        assert_eq!(
            v3.truth_raw(),
            e.truth_raw(),
            "truth ordinal dropped by lift"
        );
        assert_eq!(v3.spare_raw(), e.spare(), "spare ordinal dropped by lift");
        assert_eq!(v3.truth_raw(), 0b11);
        assert_eq!(v3.spare_raw(), 0b111);

        // Round trip: rehydrate with the same SPO restores the ordinals so the
        // reasoning carrier sees exactly what was stored.
        let back = v3.rehydrate(e.s_idx(), e.p_idx(), e.o_idx());
        assert_eq!(back.truth_raw(), e.truth_raw());
        assert_eq!(back.spare(), e.spare());
        assert_eq!(back.topology(), CausalTopology::Unknown);
        assert_eq!(back.reasoning_band(), ReasoningBand::Transcendent);

        // Stay-silent half: zero ordinals stay zero — the lift neither invents
        // a band nor upgrades provenance (the from_v1 trap D-ACR-7 fences is a
        // PROVENANCE gap, not a bit-copy defect; the bits themselves are exact).
        let z = sample_edges()[1];
        assert_eq!(z.truth_raw(), 0);
        assert_eq!(z.spare(), 0);
        let vz = CausalEdgeV3::from_v1(z, 7);
        assert_eq!(vz.truth_raw(), 0);
        assert_eq!(vz.spare_raw(), 0);
    }

    /// D-ACR-7 BLOCK-1, write side: the unstated lift DROPS the tail while the
    /// plain lift CLAIMS it. Two-sided by construction — a constructor that
    /// zeroed unconditionally, or one that never differed from `from_v1`,
    /// would carry exactly as much information as no constructor at all.
    #[cfg(feature = "causal-edge-v2-layout")]
    #[test]
    fn from_v1_tail_unstated_drops_what_from_v1_claims() {
        use crate::layout::{CausalTopology, ReasoningBand};

        // A source whose tail is NON-ZERO — without this the two lifts agree
        // trivially and the test proves nothing.
        let e = sample_edges()[0]
            .with_w_slot(0x2A)
            .with_topology(CausalTopology::Unknown)
            .with_reasoning_band(ReasoningBand::Transcendent);
        assert_ne!(
            (e.truth_raw(), e.spare()),
            (0, 0),
            "fixture must have a tail"
        );

        let claimed = CausalEdgeV3::from_v1(e, 7);
        let unstated = CausalEdgeV3::from_v1_tail_unstated(e, 7);

        // CLAIMS: the tail survives, ordinal for ordinal.
        assert_eq!(claimed.truth_raw(), e.truth_raw());
        assert_eq!(claimed.spare_raw(), e.spare());
        assert_eq!(claimed.w_slot(), e.w_slot() & 0x3F);

        // DECLINES: the tail is zero, so a consumer reads the zero-fallback
        // rather than an ordinal nobody vouched for.
        assert_eq!(
            unstated.truth_raw(),
            0,
            "unstated lift must not claim truth"
        );
        assert_eq!(
            unstated.spare_raw(),
            0,
            "unstated lift must not claim a band"
        );
        assert_eq!(unstated.w_slot(), 0, "the whole tail is dropped, not half");

        // And ONLY the tail differs — every other lifted field is identical.
        let (a, b) = (claimed.to_le_bytes(), unstated.to_le_bytes());
        assert_eq!(a[..8], b[..8], "bytes 0..8 must be untouched by the choice");
        assert_eq!(
            a[10..],
            b[10..],
            "bytes 10.. must be untouched by the choice"
        );
        assert_ne!(
            a[8..10],
            b[8..10],
            "and the tail bytes must be what differs"
        );
    }

    /// The can-stay-silent half: on a source whose tail is ALREADY zero the two
    /// lifts are byte-identical. The constructor is a declaration about
    /// provenance, never an unconditional mutation.
    #[cfg(feature = "causal-edge-v2-layout")]
    #[test]
    fn on_a_zero_tail_the_two_lifts_agree() {
        let z = sample_edges()[1];
        assert_eq!((z.truth_raw(), z.spare()), (0, 0));
        assert_eq!(
            CausalEdgeV3::from_v1(z, 7).to_le_bytes(),
            CausalEdgeV3::from_v1_tail_unstated(z, 7).to_le_bytes(),
        );
    }
}
