//! CausalEdge64: the atomic causal unit.
//!
//! One u64. One register. One read. Full causal edge with epistemic state.

use super::pearl::CausalMask;
use super::plasticity::PlasticityState;

/// NARS inference types encoded in 3 bits.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum InferenceType {
    /// A→B, B→C ⊢ A→C. Follow the chain.
    Deduction = 0,
    /// A→B, A→C ⊢ B→C. Generalize from shared cause.
    Induction = 1,
    /// A→B, C→B ⊢ A→C. Infer from shared effect.
    Abduction = 2,
    /// Merge two truth values for the same statement.
    Revision = 3,
    /// Combine complementary evidence across domains.
    Synthesis = 4,
    /// Do-calculus intervention: fix a variable by external action (Pearl's do-operator).
    /// Signals that the edge represents an interventional distribution P(Y | do(X=x)).
    Intervention = 5,
    /// Counterfactual: reason about what would have happened under a different world.
    /// Requires SPO mask + high confidence (see [`CausalEdge64::counterfactual_ready`]).
    Counterfactual = 6,
    /// Reserved for future inference types.
    Reserved7 = 7,
}

impl InferenceType {
    #[inline]
    fn from_bits(v: u8) -> Self {
        match v & 0b111 {
            0 => Self::Deduction,
            1 => Self::Induction,
            2 => Self::Abduction,
            3 => Self::Revision,
            4 => Self::Synthesis,
            5 => Self::Intervention,
            6 => Self::Counterfactual,
            _ => Self::Reserved7,
        }
    }

    /// Convert to the v2 signed inference mantissa (i4, range −8..+7).
    ///
    /// Mapping per pr-ce64-mb-2-causaledge64-v2.md §"Signed Mantissa Rationale":
    /// - Positive mantissa (+1..+7) = forward-chain direction.
    /// - Negative mantissa (−1..−7) = backward-chain direction.
    /// - Zero (0) = Identity / neutral — bare SPO assertions, no active NARS rule.
    ///
    /// Slot table (|mantissa| = base rule index):
    ///   0=Identity, 1=Deduction(+)/Abduction(-), 2=Induction(+)/Contraposition(-),
    ///   3=Exemplification(+)/Analogy-negative(-), 4=Revision+(+)/Revision-(-),
    ///   5=Synthesis(+)/Decomposition(-),
    ///   6=Intervention(+)/Counterfactual(-) [PR-LL-1 absorbed per L-9],
    ///   7=Extension(+)/Intension-negative(-) [future].
    #[inline]
    pub fn to_mantissa(self) -> i8 {
        match self {
            // Forward-chain (positive mantissa)
            Self::Deduction    => 1,
            Self::Induction    => 2,
            // NOTE: Abduction is backward in the signed encoding; v1 enum maps it to +3 slot
            // for forward semantics, but in v2 signed scheme Abduction is negative direction.
            // For v1 back-compat, Abduction here returns the forward slot (+3 = Exemplification
            // slot). Callers must use from_mantissa() to distinguish signed direction.
            Self::Abduction    => -1,   // backward: |1| = Abduction
            Self::Revision     => 4,    // forward: Revision-positive slot
            Self::Synthesis    => 5,    // forward: Synthesis slot
            Self::Intervention => 6,    // forward: PR-LL-1 Intervention (+6 per L-9)
            // Backward-chain (negative mantissa)
            Self::Counterfactual => -6, // backward: PR-LL-1 Counterfactual (−6 per L-9)
            Self::Reserved7    => 7,    // extension slot (positive, forward)
        }
    }

    /// Construct from the v2 signed inference mantissa (i4, range −8..+7).
    ///
    /// Maps magnitude + sign to the closest v1 InferenceType for back-compat.
    /// Positive values = forward-chain; negative values = backward-chain.
    /// Zero = Identity (returns Deduction as the neutral forward-chain rule).
    #[inline]
    pub fn from_mantissa(m: i8) -> Self {
        let mag = m.unsigned_abs() & 0x7; // magnitude 0..7
        let forward = m >= 0;
        match mag {
            0 => Self::Deduction,    // 0 = Identity/neutral → treat as Deduction
            1 => if forward { Self::Deduction } else { Self::Abduction },
            2 => if forward { Self::Induction } else { Self::Abduction }, // Contraposition → Abduction
            3 => if forward { Self::Synthesis } else { Self::Abduction }, // Exemplification / Analogy-neg
            4 => Self::Revision,     // Revision +/- (same enum, sign distinguishes)
            5 => if forward { Self::Synthesis } else { Self::Synthesis }, // Synthesis / Decomposition
            6 => if forward { Self::Intervention } else { Self::Counterfactual }, // PR-LL-1 (L-9)
            _ => Self::Reserved7,    // 7 = Extension / Intension-negative (future)
        }
    }
}

/// The 64-bit causal neuron.
///
/// Layout (LSB to MSB):
/// ```text
/// [0:7]   S palette index
/// [8:15]  P palette index
/// [16:23] O palette index
/// [24:31] NARS frequency (u8, f = val/255)
/// [32:39] NARS confidence (u8, c = val/255)
/// [40:42] Causal mask (3 bits, Pearl's 2³)
/// [43:45] Direction triad (3 bits, sign(dim0) per S,P,O)
/// [46:48] Inference type (3 bits)
/// [49:51] Plasticity flags (3 bits, hot/cold per S,P,O)
/// [52:63] Temporal index (12 bits, 4096 time slots)
/// ```
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct CausalEdge64(pub u64);

// Bit field positions and masks
const S_SHIFT: u32 = 0;
const P_SHIFT: u32 = 8;
const O_SHIFT: u32 = 16;
const FREQ_SHIFT: u32 = 24;
const CONF_SHIFT: u32 = 32;
const CAUSAL_SHIFT: u32 = 40;
const DIR_SHIFT: u32 = 43;
const INFER_SHIFT: u32 = 46;
const PLAST_SHIFT: u32 = 49;
const TEMPORAL_SHIFT: u32 = 52;

const BYTE_MASK: u64 = 0xFF;
const BITS3_MASK: u64 = 0b111;
const BITS12_MASK: u64 = 0xFFF;

impl CausalEdge64 {
    /// Zero edge: unknown, no evidence, no time.
    pub const ZERO: Self = Self(0);

    /// Pack all fields into a CausalEdge64 (v1 signature — retained for back-compat).
    ///
    /// When `causal-edge-v2-layout` is enabled:
    /// - The inference parameter is packed as a 3-bit value into bits 46-48 (lower
    ///   3 bits of the 4-bit mantissa field). This is compatible with v2 reads via
    ///   `inference_mantissa()` for non-negative v1 InferenceType values (0..7).
    /// - Plasticity is packed at v2 PLAST_SHIFT=50 (bits 50-52), not v1 PLAST_SHIFT=49.
    /// - **The `temporal` parameter is IGNORED.** Bits 52-63 are reclaimed by v2 for
    ///   W-slot (53-58), truth-band lens (59-60), and spare (61-63); bit 52 belongs
    ///   to plasticity. Writing temporal into those bits would corrupt the reclaim
    ///   zone (see plan §6 Option F, decision L-2). Use `pack_v2()` for new code.
    ///
    /// For new code, prefer `pack_v2()` (feature `causal-edge-v2-layout`).
    #[inline]
    pub fn pack(
        s_idx: u8,
        p_idx: u8,
        o_idx: u8,
        frequency: u8,
        confidence: u8,
        causal_mask: CausalMask,
        direction: u8,
        inference: InferenceType,
        plasticity: PlasticityState,
        temporal: u16,
    ) -> Self {
        let mut v: u64 = 0;
        v |= (s_idx as u64) << S_SHIFT;
        v |= (p_idx as u64) << P_SHIFT;
        v |= (o_idx as u64) << O_SHIFT;
        v |= (frequency as u64) << FREQ_SHIFT;
        v |= (confidence as u64) << CONF_SHIFT;
        v |= ((causal_mask as u64) & BITS3_MASK) << CAUSAL_SHIFT;
        v |= ((direction as u64) & BITS3_MASK) << DIR_SHIFT;
        // v2 layout: write the signed mantissa (4 bits, bits 46-49) via the
        // enum→mantissa mapping so that subsequent reads via inference_mantissa()
        // + InferenceType::from_mantissa() round-trip the *semantic* meaning.
        // Writing the raw enum discriminant into 3 bits would silently re-route
        // Abduction(2) as Induction(+2), Revision(3) as Synthesis(+3), etc.
        #[cfg(feature = "causal-edge-v2-layout")]
        {
            let mantissa_raw = (inference.to_mantissa() as u8) as u64 & 0xF;
            v |= mantissa_raw << INFER_SHIFT;
            v |= ((plasticity.bits() as u64) & BITS3_MASK) << crate::layout::PLAST_SHIFT;
            // v2: temporal is IGNORED. Bits 52-63 are reclaimed for plasticity[2]
            // + W-slot + truth + spare per plan §6 Option F / L-2. Writing the
            // v1 temporal here would corrupt the reclaim zone; silently drop it.
            let _ = temporal;
        }
        #[cfg(not(feature = "causal-edge-v2-layout"))]
        {
            v |= ((inference as u8 as u64) & BITS3_MASK) << INFER_SHIFT;
            v |= ((plasticity.bits() as u64) & BITS3_MASK) << PLAST_SHIFT;
            v |= ((temporal as u64) & BITS12_MASK) << TEMPORAL_SHIFT;
        }
        Self(v)
    }

    // ─── Palette Indices ────────────────────────────────────────────

    /// Subject palette index.
    #[inline(always)]
    pub fn s_idx(self) -> u8 {
        (self.0 >> S_SHIFT) as u8
    }

    /// Predicate palette index.
    #[inline(always)]
    pub fn p_idx(self) -> u8 {
        (self.0 >> P_SHIFT) as u8
    }

    /// Object palette index.
    #[inline(always)]
    pub fn o_idx(self) -> u8 {
        (self.0 >> O_SHIFT) as u8
    }

    /// Set subject palette index.
    #[inline]
    pub fn set_s_idx(&mut self, v: u8) {
        self.0 = (self.0 & !(BYTE_MASK << S_SHIFT)) | ((v as u64) << S_SHIFT);
    }

    /// Set predicate palette index.
    #[inline]
    pub fn set_p_idx(&mut self, v: u8) {
        self.0 = (self.0 & !(BYTE_MASK << P_SHIFT)) | ((v as u64) << P_SHIFT);
    }

    /// Set object palette index.
    #[inline]
    pub fn set_o_idx(&mut self, v: u8) {
        self.0 = (self.0 & !(BYTE_MASK << O_SHIFT)) | ((v as u64) << O_SHIFT);
    }

    // ─── NARS Truth ─────────────────────────────────────────────────

    /// NARS frequency as u8 (raw quantized value).
    #[inline(always)]
    pub fn frequency_u8(self) -> u8 {
        (self.0 >> FREQ_SHIFT) as u8
    }

    /// NARS confidence as u8 (raw quantized value).
    #[inline(always)]
    pub fn confidence_u8(self) -> u8 {
        (self.0 >> CONF_SHIFT) as u8
    }

    /// NARS frequency as f32 in [0, 1].
    #[inline]
    pub fn frequency(self) -> f32 {
        self.frequency_u8() as f32 / 255.0
    }

    /// NARS confidence as f32 in [0, 1].
    #[inline]
    pub fn confidence(self) -> f32 {
        self.confidence_u8() as f32 / 255.0
    }

    /// NARS expectation: c * (f - 0.5) + 0.5.
    #[inline]
    pub fn expectation(self) -> f32 {
        let f = self.frequency();
        let c = self.confidence();
        c * (f - 0.5) + 0.5
    }

    /// Evidence weight: c / (1 - c). Returns u16::MAX if c == 255.
    #[inline]
    pub fn evidence_weight(self) -> f32 {
        let c = self.confidence();
        if c >= 0.999 {
            f32::MAX
        } else {
            c / (1.0 - c)
        }
    }

    /// Set frequency (u8).
    #[inline]
    pub fn set_frequency_u8(&mut self, v: u8) {
        self.0 = (self.0 & !(BYTE_MASK << FREQ_SHIFT)) | ((v as u64) << FREQ_SHIFT);
    }

    /// Set confidence (u8).
    #[inline]
    pub fn set_confidence_u8(&mut self, v: u8) {
        self.0 = (self.0 & !(BYTE_MASK << CONF_SHIFT)) | ((v as u64) << CONF_SHIFT);
    }

    /// Set frequency from f32.
    #[inline]
    pub fn set_frequency(&mut self, f: f32) {
        self.set_frequency_u8((f.clamp(0.0, 1.0) * 255.0).round() as u8);
    }

    /// Set confidence from f32.
    #[inline]
    pub fn set_confidence(&mut self, c: f32) {
        self.set_confidence_u8((c.clamp(0.0, 1.0) * 255.0).round() as u8);
    }

    // ─── Causal Mask (Pearl's 2³) ───────────────────────────────────

    /// The 3-bit causal projection mask.
    #[inline(always)]
    pub fn causal_mask(self) -> CausalMask {
        CausalMask::from_bits(((self.0 >> CAUSAL_SHIFT) & BITS3_MASK) as u8)
    }

    /// Set the causal mask.
    #[inline]
    pub fn set_causal_mask(&mut self, m: CausalMask) {
        self.0 = (self.0 & !(BITS3_MASK << CAUSAL_SHIFT))
            | (((m as u8 as u64) & BITS3_MASK) << CAUSAL_SHIFT);
    }

    /// Match if this edge's causal mask contains AT LEAST the bits in `query_mask`.
    ///
    /// `query_mask` is interpreted as the low 3 bits of the Pearl 2³ packing
    /// (S=0b100, P=0b010, O=0b001 — see [`CausalMask`]). Higher bits are ignored.
    ///
    /// This is the query-side predicate used by graph WHERE clauses to filter
    /// edges by causal type. For example, `query_mask = CausalMask::PO as u8`
    /// (`0b011`) matches every edge whose causal mask has at least the P and O
    /// planes active — i.e. interventional edges (`PO`) and counterfactual
    /// edges (`SPO`), but not pure association (`SO`).
    ///
    /// Semantics:
    /// - `query_mask == edge_mask`: full match
    /// - `query_mask` is a subset of `edge_mask`: subset match
    /// - `query_mask` and `edge_mask` are disjoint (sharing no required bits):
    ///   no match
    /// - `query_mask == 0` (`CausalMask::None`): matches every edge — there
    ///   are no required bits, so the predicate is vacuously satisfied.
    #[inline(always)]
    pub const fn matches_causal(&self, query_mask: u8) -> bool {
        let q = query_mask & 0b111;
        let edge_mask = ((self.0 >> CAUSAL_SHIFT) & BITS3_MASK) as u8;
        (edge_mask & q) == q
    }

    /// Type-safe variant of [`Self::matches_causal`] taking a [`CausalMask`].
    #[inline(always)]
    pub fn matches_causal_mask(&self, query_mask: CausalMask) -> bool {
        self.matches_causal(query_mask as u8)
    }

    /// Is the S-plane active in the current causal projection?
    #[inline(always)]
    pub fn s_active(self) -> bool {
        (self.0 >> CAUSAL_SHIFT) & 0b100 != 0
    }

    /// Is the P-plane active in the current causal projection?
    #[inline(always)]
    pub fn p_active(self) -> bool {
        (self.0 >> CAUSAL_SHIFT) & 0b010 != 0
    }

    /// Is the O-plane active in the current causal projection?
    #[inline(always)]
    pub fn o_active(self) -> bool {
        (self.0 >> CAUSAL_SHIFT) & 0b001 != 0
    }

    // ─── Direction Triad ────────────────────────────────────────────

    /// 3-bit direction triad: sign(dim0) per S, P, O.
    /// bit 0 = S pathological, bit 1 = P pathological, bit 2 = O pathological.
    #[inline(always)]
    pub fn direction(self) -> u8 {
        ((self.0 >> DIR_SHIFT) & BITS3_MASK) as u8
    }

    /// Set direction triad.
    #[inline]
    pub fn set_direction(&mut self, d: u8) {
        self.0 = (self.0 & !(BITS3_MASK << DIR_SHIFT)) | (((d as u64) & BITS3_MASK) << DIR_SHIFT);
    }

    /// Is the subject plane pathological (dim0 negative)?
    #[inline(always)]
    pub fn s_pathological(self) -> bool {
        self.direction() & 0b001 != 0
    }

    /// Is the predicate plane pathological?
    #[inline(always)]
    pub fn p_pathological(self) -> bool {
        self.direction() & 0b010 != 0
    }

    /// Is the outcome plane pathological?
    #[inline(always)]
    pub fn o_pathological(self) -> bool {
        self.direction() & 0b100 != 0
    }

    // ─── Inference Type ─────────────────────────────────────────────

    /// NARS inference type for this edge (v1 3-bit accessor).
    ///
    /// DEPRECATED since 0.2.0: In v2 layout, bits 46-49 hold a 4-bit SIGNED mantissa
    /// (range −8..+7). This accessor reads only bits 46-48 and treats them as a 3-bit
    /// unsigned value, which is correct for v1-written edges but WRONG for v2-written
    /// edges where the mantissa sign bit lives in bit 49.
    ///
    /// Migration: use `inference_mantissa()` (feature `causal-edge-v2-layout`) and
    /// `InferenceType::from_mantissa()` to convert. The mapping is documented in
    /// pr-ce64-mb-2-causaledge64-v2.md §"Signed Mantissa Rationale".
    #[deprecated(
        since = "0.2.0",
        note = "bits 46-49 now hold a 4-bit signed mantissa in v2 layout; \
                use inference_mantissa() + InferenceType::from_mantissa() instead; \
                see pr-ce64-mb-2-causaledge64-v2.md §\"Signed Mantissa Rationale\"."
    )]
    #[inline(always)]
    pub fn inference_type(self) -> InferenceType {
        // v2: bits 46-49 hold a 4-bit signed mantissa. Route through from_mantissa
        // so the v1 API contract is preserved semantically — pack(X).inference_type()
        // round-trips to X for all v1 enum variants via the to_mantissa/from_mantissa
        // pair (e.g. Intervention → +6 → Intervention, Counterfactual → −6 →
        // Counterfactual, Abduction → −1 → Abduction). Reading the raw 3-bit field
        // (the v1 layout) would silently swap variants under v2 because the
        // discriminant numbering does not match the mantissa encoding.
        #[cfg(feature = "causal-edge-v2-layout")]
        {
            // inference_mantissa() reads the 4-bit signed field via arithmetic shift
            // sign-extension.
            let raw = ((self.0 >> INFER_SHIFT) & 0xF) as i8;
            let m = (raw << 4) >> 4; // sign-extend 4-bit → i8
            InferenceType::from_mantissa(m)
        }
        #[cfg(not(feature = "causal-edge-v2-layout"))]
        {
            InferenceType::from_bits(((self.0 >> INFER_SHIFT) & BITS3_MASK) as u8)
        }
    }

    /// Set inference type.
    ///
    /// **v2 behavior:** writes via `t.to_mantissa()` into the 4-bit signed field
    /// (bits 46-49) so that `inference_type()` round-trips. Writing the raw v1
    /// discriminant into 3 bits would silently corrupt the semantic (e.g.
    /// Counterfactual=6 stored as +6 mantissa → from_mantissa(+6)=Intervention).
    #[inline]
    pub fn set_inference_type(&mut self, t: InferenceType) {
        #[cfg(feature = "causal-edge-v2-layout")]
        {
            let raw = (t.to_mantissa() as u8) as u64 & 0xF;
            self.0 = (self.0 & !(0xF << INFER_SHIFT)) | (raw << INFER_SHIFT);
        }
        #[cfg(not(feature = "causal-edge-v2-layout"))]
        {
            self.0 = (self.0 & !(BITS3_MASK << INFER_SHIFT))
                | (((t as u8 as u64) & BITS3_MASK) << INFER_SHIFT);
        }
    }

    // ─── Plasticity ─────────────────────────────────────────────────

    /// Plasticity state for all three planes.
    ///
    /// In v1 layout, plasticity occupies bits 49-51 (PLAST_SHIFT=49).
    /// In v2 layout (`causal-edge-v2-layout` feature), plasticity occupies bits 50-52
    /// (PLAST_SHIFT=50) because the inference mantissa expanded from 3b to 4b (L-4).
    #[inline(always)]
    pub fn plasticity(self) -> PlasticityState {
        #[cfg(feature = "causal-edge-v2-layout")]
        { PlasticityState::from_bits(((self.0 >> crate::layout::PLAST_SHIFT) & BITS3_MASK) as u8) }
        #[cfg(not(feature = "causal-edge-v2-layout"))]
        { PlasticityState::from_bits(((self.0 >> PLAST_SHIFT) & BITS3_MASK) as u8) }
    }

    /// Set plasticity state.
    ///
    /// Uses v2 PLAST_SHIFT=50 when `causal-edge-v2-layout` feature is enabled,
    /// v1 PLAST_SHIFT=49 otherwise.
    #[inline]
    pub fn set_plasticity(&mut self, p: PlasticityState) {
        #[cfg(feature = "causal-edge-v2-layout")]
        {
            let shift = crate::layout::PLAST_SHIFT;
            self.0 = (self.0 & !(BITS3_MASK << shift))
                | (((p.bits() as u64) & BITS3_MASK) << shift);
        }
        #[cfg(not(feature = "causal-edge-v2-layout"))]
        {
            self.0 = (self.0 & !(BITS3_MASK << PLAST_SHIFT))
                | (((p.bits() as u64) & BITS3_MASK) << PLAST_SHIFT);
        }
    }

    // ─── Temporal Index ─────────────────────────────────────────────

    /// 12-bit temporal index (0..4095).
    ///
    /// DEPRECATED since 0.2.0: In v2 layout, bits 52-63 are reclaimed for
    /// W-slot (53-58), truth-band-lens (59-60), and spare (61-63). This accessor
    /// returns GARBAGE for v2-written edges. Temporal causality is structural:
    /// use chain-position in SpoWitnessChain or AriGraph Triplet.timestamp instead.
    /// See cognitive-substrate-convergence-v1.md L-2.
    #[deprecated(
        since = "0.2.0",
        note = "bits 52-63 reclaimed in v2 layout; temporal is structural \
                (SpoWitnessChain chain-position + AriGraph Triplet.timestamp); \
                see cognitive-substrate-convergence-v1.md L-2."
    )]
    #[inline(always)]
    pub fn temporal(self) -> u16 {
        ((self.0 >> TEMPORAL_SHIFT) & BITS12_MASK) as u16
    }

    /// Set temporal index.
    ///
    /// **v2 behavior:** NO-OP. Under `causal-edge-v2-layout` (default since
    /// 0.2.0), bits 52-63 are reclaimed for plasticity[2] + W-slot + lens +
    /// spare per plan §6 / L-2. Writing the v1 temporal here would corrupt
    /// the reclaim zone — same bug pattern as the v1-temporal-aliases-W3-
    /// reclaim-zone P1 codex caught in PR #381. Temporal causality is now
    /// structural (chain-position + AriGraph anchor); use those instead.
    ///
    /// Existing v1 callers (e.g. `CausalEdge64::learn`) continue to compile
    /// — the temporal arg becomes a silent drop under v2 with the rest of
    /// the reclaim-zone integrity preserved.
    #[inline]
    pub fn set_temporal(&mut self, t: u16) {
        #[cfg(feature = "causal-edge-v2-layout")]
        {
            let _ = t; // v2: bits 52-63 are W/lens/spare; do not overwrite
        }
        #[cfg(not(feature = "causal-edge-v2-layout"))]
        {
            self.0 = (self.0 & !(BITS12_MASK << TEMPORAL_SHIFT))
                | (((t as u64) & BITS12_MASK) << TEMPORAL_SHIFT);
        }
    }

    // ─── Causal Distance ────────────────────────────────────────────

    /// Distance between two edges, respecting causal mask.
    /// Only palette planes where the mask bit is set contribute to distance.
    /// Uses precomputed 256×256 distance matrices per plane.
    #[inline]
    pub fn causal_distance(
        self,
        other: Self,
        s_dm: &[u16; 256 * 256],
        p_dm: &[u16; 256 * 256],
        o_dm: &[u16; 256 * 256],
    ) -> u32 {
        let mask = self.causal_mask() as u8;
        let mut d = 0u32;
        if mask & 0b100 != 0 {
            d += s_dm[self.s_idx() as usize * 256 + other.s_idx() as usize] as u32;
        }
        if mask & 0b010 != 0 {
            d += p_dm[self.p_idx() as usize * 256 + other.p_idx() as usize] as u32;
        }
        if mask & 0b001 != 0 {
            d += o_dm[self.o_idx() as usize * 256 + other.o_idx() as usize] as u32;
        }
        d
    }

    // ─── Forward Pass (BNN-style) ───────────────────────────────────

    /// The forward pass: compose palettes + propagate truth + propagate causality.
    ///
    /// This IS the "neural network inference" — but every intermediate is
    /// a CausalEdge64 with full interpretability.
    #[inline]
    pub fn forward(
        self,
        weight: Self,
        compose_s: &[u8; 256 * 256],
        compose_p: &[u8; 256 * 256],
        compose_o: &[u8; 256 * 256],
    ) -> Self {
        // 1. Palette composition (the "multiply")
        let s_out = compose_s[self.s_idx() as usize * 256 + weight.s_idx() as usize];
        let p_out = compose_p[self.p_idx() as usize * 256 + weight.p_idx() as usize];
        let o_out = compose_o[self.o_idx() as usize * 256 + weight.o_idx() as usize];

        // 2. NARS truth propagation (the "activation function")
        // Under v2: decode the 4-bit signed mantissa (bits 46-49) and route
        // through the same InferenceType variants. Without this, v2 edges
        // built via `with_inference_mantissa()` route as 3-bit unsigned
        // (e.g. -1 = 0b1111 reads as Reserved7), bypassing Abduction/
        // Counterfactual semantics entirely.
        #[cfg(feature = "causal-edge-v2-layout")]
        #[allow(deprecated)] // weight.inference_type() is the v1 fallback below; v2 uses mantissa
        let resolved_infer = InferenceType::from_mantissa(weight.inference_mantissa());
        #[cfg(not(feature = "causal-edge-v2-layout"))]
        let resolved_infer = weight.inference_type();
        let (f_out, c_out) = match resolved_infer {
            InferenceType::Deduction => {
                let f = self.frequency() * weight.frequency();
                let c = f * self.confidence() * weight.confidence();
                (f, c)
            }
            InferenceType::Induction => {
                let f = weight.frequency();
                let w = self.frequency() * self.confidence() * weight.confidence();
                (f, w / (w + 1.0))
            }
            InferenceType::Abduction => {
                let f = self.frequency();
                let w = weight.frequency() * self.confidence() * weight.confidence();
                (f, w / (w + 1.0))
            }
            InferenceType::Revision => {
                let w1 = self.evidence_weight();
                let w2 = weight.evidence_weight();
                let ws = w1 + w2;
                if ws < f32::EPSILON {
                    (0.5, 0.0)
                } else {
                    let f = (self.frequency() * w1 + weight.frequency() * w2) / ws;
                    let c = ws / (ws + 1.0);
                    (f, c)
                }
            }
            InferenceType::Synthesis | _ => {
                let f = (self.frequency() + weight.frequency()) / 2.0;
                let c = (self.confidence() + weight.confidence()) / 2.0;
                (f, c)
            }
        };

        // 3. Causal mask: AND (only planes active in BOTH survive)
        let mask_out =
            CausalMask::from_bits((self.causal_mask() as u8) & (weight.causal_mask() as u8));

        // 4. Temporal: latest of the two
        let t_out = self.temporal().max(weight.temporal());

        // 5. Inherit plasticity from weight (the "learned" edge)
        //    and direction will be recomputed from composed palette entries
        #[allow(deprecated)] // pack() v2 path drops temporal; resolved_infer carries v2 mantissa
        let result = Self::pack(
            s_out,
            p_out,
            o_out,
            (f_out.clamp(0.0, 1.0) * 255.0).round() as u8,
            (c_out.clamp(0.0, 1.0) * 255.0).round() as u8,
            mask_out,
            weight.direction(), // TODO: recompute from composed palette dim0 signs
            resolved_infer,
            weight.plasticity(),
            t_out,
        );
        // Under v2: re-stamp the signed mantissa onto the result. pack() only
        // writes 3 bits (v1 enum discriminant) into bits 46-48; bit 49 (the
        // sign bit) stays 0, so negative mantissas (Abduction, Counterfactual)
        // would lose their sign. Override here with the resolved value.
        #[cfg(feature = "causal-edge-v2-layout")]
        let result = result.with_inference_mantissa(resolved_infer.to_mantissa());
        result
    }

    // ─── Learning (Evidence-Driven Plasticity) ──────────────────────

    /// Learn from an observation: NARS revision + plasticity update.
    ///
    /// This IS reinforcement learning — but the reward signal is
    /// evidence accumulation, not a scalar.
    #[inline]
    pub fn learn(&mut self, observation: Self, current_time: u16) {
        let plast = self.plasticity();

        // Only update planes that are HOT (plastic)
        if plast.s_hot() && observation.s_idx() != self.s_idx() {
            // Evidence disagrees with current archetype on S-plane
            // If confidence is high → archetype needs splitting (discovery)
            // For now: adopt observation's archetype if it has higher confidence
            if observation.confidence() > self.confidence() {
                self.set_s_idx(observation.s_idx());
            }
        }
        if plast.p_hot() && observation.p_idx() != self.p_idx() {
            if observation.confidence() > self.confidence() {
                self.set_p_idx(observation.p_idx());
            }
        }
        if plast.o_hot() && observation.o_idx() != self.o_idx() {
            if observation.confidence() > self.confidence() {
                self.set_o_idx(observation.o_idx());
            }
        }

        // NARS revision: merge evidence
        let w1 = self.evidence_weight();
        let w2 = observation.evidence_weight();
        let ws = w1 + w2;
        if ws > f32::EPSILON {
            let f_new = (self.frequency() * w1 + observation.frequency() * w2) / ws;
            let c_new = ws / (ws + 1.0);
            self.set_frequency(f_new);
            self.set_confidence(c_new);

            // Plasticity transitions based on confidence
            let mut new_plast = plast;
            if c_new > 0.9 {
                // High confidence: freeze all planes
                new_plast = PlasticityState::ALL_FROZEN;
            } else if c_new > 0.7 {
                // Moderate confidence: freeze planes with low error
                if observation.s_idx() == self.s_idx() {
                    new_plast = new_plast.freeze_s();
                }
                if observation.p_idx() == self.p_idx() {
                    new_plast = new_plast.freeze_p();
                }
                if observation.o_idx() == self.o_idx() {
                    new_plast = new_plast.freeze_o();
                }
            }
            self.set_plasticity(new_plast);
        }

        // Update temporal index
        self.set_temporal(current_time);
    }

    // ─── Diagnostics ────────────────────────────────────────────────

    /// Is this edge at the interventional level (do-calculus)?
    #[inline]
    pub fn is_interventional(self) -> bool {
        self.causal_mask() == CausalMask::PO
    }

    /// Is this edge at the counterfactual level?
    #[inline]
    pub fn is_counterfactual(self) -> bool {
        self.causal_mask() == CausalMask::SPO
    }

    /// Is there sufficient evidence for counterfactual reasoning?
    /// Requires SPO mask AND confidence > threshold.
    #[inline]
    pub fn counterfactual_ready(self, confidence_threshold: f32) -> bool {
        self.causal_mask() == CausalMask::SPO && self.confidence() >= confidence_threshold
    }

    /// Clinical concern level: count of pathological planes.
    #[inline]
    pub fn concern_level(self) -> u8 {
        let d = self.direction();
        (d & 1) + ((d >> 1) & 1) + ((d >> 2) & 1)
    }

    /// Is this edge fully frozen (all planes)?
    #[inline]
    pub fn is_frozen(self) -> bool {
        self.plasticity() == PlasticityState::ALL_FROZEN
    }
    // ─── V2 Pack ───────────────────────────────────────────────────────

    /// V2 pack: construct a CausalEdge64 without temporal (dropped per L-2).
    ///
    /// The 12 bits formerly used by temporal are reclaimed for W-slot (53-58),
    /// truth-band-lens (59-60), spare (61-63), and the mantissa expansion (46-49).
    /// W-slot, truth-band, spare, and inference mantissa default to zero on pack_v2.
    /// Set them after construction with `with_w_slot`, `with_truth`,
    /// `with_inference_mantissa`, `with_spare`, or `with_routing`.
    ///
    /// Note: the v1 `pack()` retains its 9-arg signature (with `temporal`) for
    /// back-compat and migration tests. Use `pack_v2()` for all new v2 code.
    #[cfg(feature = "causal-edge-v2-layout")]
    #[inline]
    pub fn pack_v2(
        s_idx: u8,
        p_idx: u8,
        o_idx: u8,
        frequency: u8,
        confidence: u8,
        causal_mask: CausalMask,
        direction: u8,
        plasticity: PlasticityState,
    ) -> Self {
        use crate::layout::{S_SHIFT as LS, P_SHIFT as LP, O_SHIFT as LO,
                            FREQ_SHIFT as LF, CONF_SHIFT as LC,
                            CAUSAL_SHIFT as LCA, DIR_SHIFT as LD,
                            PLAST_SHIFT as LPL, BITS3_MASK as B3};
        let mut v: u64 = 0;
        v |= (s_idx as u64) << LS;
        v |= (p_idx as u64) << LP;
        v |= (o_idx as u64) << LO;
        v |= (frequency as u64) << LF;
        v |= (confidence as u64) << LC;
        v |= ((causal_mask as u64) & B3) << LCA;
        v |= ((direction as u64) & B3) << LD;
        // inference mantissa defaults to 0 (Identity/neutral)
        v |= ((plasticity.bits() as u64) & B3) << LPL;
        // W-slot, truth-band, spare default to 0
        Self(v)
    }

}

// ─── V2 Accessors and Builders ─────────────────────────────────────────────
// All gated on `#[cfg(feature = "causal-edge-v2-layout")]`.
// G-slot is NOT present in v2 (L-3: redundant via palette family-prefix).
// See pr-ce64-mb-2-causaledge64-v2.md §4 for full method semantics.

#[cfg(feature = "causal-edge-v2-layout")]
impl CausalEdge64 {
    // ── Read Accessors ──────────────────────────────────────────────────────

    /// Witness corpus root handle (6-bit, 0..=63). 0 = no corpus anchor.
    ///
    /// WARNING: Returns GARBAGE for non-zero v1 edges — bits 53-58 were
    /// temporal MSBs in v1. Apply a version gate before calling on edges
    /// of unknown provenance. `CausalEdge64::ZERO` returns 0 (correct default).
    #[inline(always)]
    pub fn w_slot(self) -> u8 {
        use crate::layout::{W_SHIFT, BITS6_MASK};
        ((self.0 >> W_SHIFT) & BITS6_MASK) as u8
    }

    /// Inference mantissa: 4-bit signed i4, range −8..+7.
    ///
    /// sign = chain direction (+ = forward-chain, − = backward-chain).
    /// abs = base NARS rule index (0..7). See §"Signed Mantissa Rationale".
    /// `CausalEdge64::ZERO` returns 0 (neutral/identity — no NARS rule applied).
    ///
    /// Sign-extension: bits 46-49 extracted as 4-bit unsigned, then
    /// sign-extended to i8 via arithmetic shift: if bit 3 set, OR with 0xF0.
    #[inline(always)]
    pub fn inference_mantissa(self) -> i8 {
        use crate::layout::{INFER_SHIFT, BITS4_MASK};
        let raw = ((self.0 >> INFER_SHIFT) & BITS4_MASK) as u8;
        if raw & 0x8 != 0 { (raw | 0xF0) as i8 } else { raw as i8 }
    }

    /// Chain direction extracted from mantissa sign: 1=forward, −1=backward, 0=neutral.
    #[inline(always)]
    pub fn inference_direction(self) -> i8 {
        let m = self.inference_mantissa();
        if m > 0 { 1 } else if m < 0 { -1 } else { 0 }
    }

    /// Base NARS rule index (0..7) extracted from mantissa magnitude.
    #[inline(always)]
    pub fn inference_rule_index(self) -> u8 {
        self.inference_mantissa().unsigned_abs() & 0x7
    }

    /// Truth-band lens as `TrustTexture` (2-bit). Returns `Crystalline` for ZERO edges.
    ///
    /// WARNING: Bits 59-60 were temporal bits 7-8 in v1. v1 edges with temporal >= 128
    /// may read as Solid/Fuzzy/Murky. Apply a version gate on edges of unknown provenance.
    #[inline(always)]
    pub fn truth(self) -> crate::layout::TrustTexture {
        use crate::layout::{TRUTH_SHIFT, BITS2_MASK, TrustTexture};
        TrustTexture::from_bits_2(((self.0 >> TRUTH_SHIFT) & BITS2_MASK) as u8)
    }

    /// Raw 2-bit truth-band value (0..=3) without `TrustTexture` conversion.
    /// Useful for round-trip tests and direct comparisons.
    #[inline(always)]
    pub fn truth_raw(self) -> u8 {
        use crate::layout::{TRUTH_SHIFT, BITS2_MASK};
        ((self.0 >> TRUTH_SHIFT) & BITS2_MASK) as u8
    }

    /// Spare 3-bit field (bits 61-63). Reserved for sprint-12+ use.
    /// Returns 0 for ZERO edges and all v1-written edges (temporal MSBs were ≤ 0xFFF).
    #[inline(always)]
    pub fn spare(self) -> u8 {
        use crate::layout::{SPARE_SHIFT, BITS3_MASK};
        ((self.0 >> SPARE_SHIFT) & BITS3_MASK) as u8
    }

    // ── Builder-Shape Setters (functional update, returns Self) ─────────────

    /// Return new edge with W slot set to `w` (0..=63).
    #[inline]
    pub fn with_w_slot(self, w: u8) -> Self {
        use crate::layout::{W_SHIFT, BITS6_MASK, W_MASK};
        debug_assert!(w <= 63, "w_slot must fit 6 bits (0..=63), got {w}");
        Self((self.0 & !W_MASK) | (((w as u64) & BITS6_MASK) << W_SHIFT))
    }

    /// Return new edge with truth-band lens set.
    #[inline]
    pub fn with_truth(self, t: crate::layout::TrustTexture) -> Self {
        use crate::layout::{TRUTH_SHIFT, BITS2_MASK, TRUTH_MASK};
        Self((self.0 & !TRUTH_MASK) | ((t.to_bits_2() as u64 & BITS2_MASK) << TRUTH_SHIFT))
    }

    /// Return new edge with signed inference mantissa set (range −8..+7).
    ///
    /// Stored as 4-bit two's-complement in bits 46-49. Values outside −8..+7
    /// are naturally wrapped by the 4-bit mask (low nibble of `m as u8`).
    #[inline]
    pub fn with_inference_mantissa(self, m: i8) -> Self {
        use crate::layout::{INFER_SHIFT, BITS4_MASK, INFER_MASK};
        debug_assert!((-8..=7).contains(&m), "mantissa must be −8..+7, got {m}");
        let raw = (m as u8) & 0xF;
        Self((self.0 & !INFER_MASK) | ((raw as u64 & BITS4_MASK) << INFER_SHIFT))
    }

    /// Return new edge with spare bits set (0..=7, 3-bit field).
    #[inline]
    pub fn with_spare(self, s: u8) -> Self {
        use crate::layout::{SPARE_SHIFT, BITS3_MASK, SPARE_MASK};
        debug_assert!(s <= 7, "spare must fit 3 bits (0..=7), got {s}");
        Self((self.0 & !SPARE_MASK) | ((s as u64 & BITS3_MASK) << SPARE_SHIFT))
    }

    /// Set W-slot and truth-band in one mask-and-or operation (hot-path emit).
    ///
    /// Used by `MailboxSoA::dispatch_cycle()` when stamping routing onto emissions.
    /// NOTE: No `g` parameter — G-slot is absent in v2 layout (L-3: redundant via
    /// palette family-prefix + SoA partition + witness corpus root).
    /// Composable: `edge.with_routing(12, TrustTexture::Solid).with_inference_mantissa(-1)`.
    #[inline]
    pub fn with_routing(self, w: u8, t: crate::layout::TrustTexture) -> Self {
        use crate::layout::{W_SHIFT, TRUTH_SHIFT, BITS6_MASK, BITS2_MASK, W_MASK, TRUTH_MASK};
        debug_assert!(w <= 63, "w_slot ({w}) out of 6-bit range");
        let routing = ((w as u64 & BITS6_MASK) << W_SHIFT)
            | ((t.to_bits_2() as u64 & BITS2_MASK) << TRUTH_SHIFT);
        Self((self.0 & !(W_MASK | TRUTH_MASK)) | routing)
    }

    // ── Mutating Setters (&mut self, hot-path callers) ─────────────────────

    /// Set W slot in-place.
    #[inline]
    pub fn set_w_slot(&mut self, w: u8) {
        use crate::layout::{W_SHIFT, BITS6_MASK, W_MASK};
        debug_assert!(w <= 63);
        self.0 = (self.0 & !W_MASK) | (((w as u64) & BITS6_MASK) << W_SHIFT);
    }

    /// Set truth-band lens in-place.
    #[inline]
    pub fn set_truth(&mut self, t: crate::layout::TrustTexture) {
        use crate::layout::{TRUTH_SHIFT, BITS2_MASK, TRUTH_MASK};
        self.0 = (self.0 & !TRUTH_MASK) | ((t.to_bits_2() as u64 & BITS2_MASK) << TRUTH_SHIFT);
    }

    /// Set signed inference mantissa in-place (range −8..+7).
    #[inline]
    pub fn set_inference_mantissa(&mut self, m: i8) {
        use crate::layout::{INFER_SHIFT, BITS4_MASK, INFER_MASK};
        debug_assert!((-8..=7).contains(&m));
        let raw = (m as u8) & 0xF;
        self.0 = (self.0 & !INFER_MASK) | ((raw as u64 & BITS4_MASK) << INFER_SHIFT);
    }

    /// Set spare bits in-place (0..=7, 3-bit field).
    #[inline]
    pub fn set_spare(&mut self, s: u8) {
        use crate::layout::{SPARE_SHIFT, BITS3_MASK, SPARE_MASK};
        debug_assert!(s <= 7);
        self.0 = (self.0 & !SPARE_MASK) | ((s as u64 & BITS3_MASK) << SPARE_SHIFT);
    }
}

// ─── V1 Stub Accessors (feature off) ────────────────────────────────────────
// When the v2-layout feature is disabled, v2 accessors return safe defaults
// so callers compile without feature-gating at every call site.

#[cfg(not(feature = "causal-edge-v2-layout"))]
impl CausalEdge64 {
    #[inline(always)]
    pub fn w_slot(self) -> u8 { 0 }
    #[inline(always)]
    pub fn inference_mantissa(self) -> i8 { 0 }
    #[inline(always)]
    pub fn inference_direction(self) -> i8 { 0 }
    #[inline(always)]
    pub fn inference_rule_index(self) -> u8 { 0 }
    #[inline(always)]
    pub fn truth(self) -> crate::layout::TrustTexture { crate::layout::TrustTexture::Crystalline }
    #[inline(always)]
    pub fn truth_raw(self) -> u8 { 0 }
    #[inline(always)]
    pub fn spare(self) -> u8 { 0 }
    #[inline]
    pub fn with_w_slot(self, _w: u8) -> Self { self }
    #[inline]
    pub fn with_truth(self, _t: crate::layout::TrustTexture) -> Self { self }
    #[inline]
    pub fn with_inference_mantissa(self, _m: i8) -> Self { self }
    #[inline]
    pub fn with_spare(self, _s: u8) -> Self { self }
    #[inline]
    pub fn with_routing(self, _w: u8, _t: crate::layout::TrustTexture) -> Self { self }
    #[inline]
    pub fn set_w_slot(&mut self, _w: u8) {}
    #[inline]
    pub fn set_truth(&mut self, _t: crate::layout::TrustTexture) {}
    #[inline]
    pub fn set_inference_mantissa(&mut self, _m: i8) {}
    #[inline]
    pub fn set_spare(&mut self, _s: u8) {}
}


impl std::fmt::Debug for CausalEdge64 {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CausalEdge64")
            .field("spo", &(self.s_idx(), self.p_idx(), self.o_idx()))
            .field("f", &format!("{:.3}", self.frequency()))
            .field("c", &format!("{:.3}", self.confidence()))
            .field("pearl", &self.causal_mask())
            .field("dir", &format!("{:03b}", self.direction()))
            .field("infer", &self.inference_type())
            .field("plast", &self.plasticity())
            .field("t", &self.temporal())
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[cfg(not(feature = "causal-edge-v2-layout"))]
    fn test_roundtrip() {
        // v1-only: pack() carries temporal in bits 52-63. Under v2 (default),
        // those bits are reclaimed; pack() drops temporal and pack_v2() is
        // the canonical constructor.
        let edge = CausalEdge64::pack(
            143,
            7,
            201, // SPO palette indices
            209,
            181,            // f=0.82, c=0.71
            CausalMask::PO, // interventional level
            0b101,          // S and O pathological
            InferenceType::Deduction,
            PlasticityState::S_HOT,
            42, // temporal index
        );

        assert_eq!(edge.s_idx(), 143);
        assert_eq!(edge.p_idx(), 7);
        assert_eq!(edge.o_idx(), 201);
        assert_eq!(edge.frequency_u8(), 209);
        assert_eq!(edge.confidence_u8(), 181);
        assert_eq!(edge.causal_mask(), CausalMask::PO);
        assert_eq!(edge.direction(), 0b101);
        assert_eq!(edge.inference_type(), InferenceType::Deduction);
        assert_eq!(edge.plasticity(), PlasticityState::S_HOT);
        assert_eq!(edge.temporal(), 42);
    }

    #[test]
    fn test_forward_deduction_attenuates() {
        let input = CausalEdge64::pack(
            10,
            20,
            30,
            204,
            178, // f=0.80, c=0.70
            CausalMask::SPO,
            0,
            InferenceType::Deduction,
            PlasticityState::ALL_HOT,
            1,
        );
        let weight = CausalEdge64::pack(
            40,
            50,
            60,
            229,
            204, // f=0.90, c=0.80
            CausalMask::SPO,
            0,
            InferenceType::Deduction,
            PlasticityState::ALL_FROZEN,
            2,
        );

        // Dummy compose tables (identity for test)
        let compose = [0u8; 256 * 256];
        let result = input.forward(weight, &compose, &compose, &compose);

        // Deduction: f_out = f_in * f_w ≈ 0.80 * 0.90 = 0.72
        assert!(
            result.frequency() < input.frequency(),
            "Deduction should attenuate frequency: got {}",
            result.frequency()
        );
        // Confidence should also attenuate
        assert!(
            result.confidence() < input.confidence(),
            "Deduction should attenuate confidence: got {}",
            result.confidence()
        );
    }

    #[test]
    fn test_learn_increases_confidence() {
        let mut edge = CausalEdge64::pack(
            10,
            20,
            30,
            204,
            127, // f=0.80, c=0.50
            CausalMask::SPO,
            0,
            InferenceType::Revision,
            PlasticityState::ALL_HOT,
            1,
        );
        let observation = CausalEdge64::pack(
            10,
            20,
            30,
            204,
            127, // same frequency, same confidence
            CausalMask::SPO,
            0,
            InferenceType::Revision,
            PlasticityState::ALL_HOT,
            2,
        );

        let c_before = edge.confidence();
        edge.learn(observation, 3);
        assert!(
            edge.confidence() > c_before,
            "Learning from agreeing evidence should increase confidence"
        );
    }

    #[test]
    fn test_causal_mask_projection() {
        let edge = CausalEdge64::pack(
            10,
            20,
            30,
            200,
            200,
            CausalMask::PO,
            0,
            InferenceType::Deduction,
            PlasticityState::ALL_FROZEN,
            0,
        );

        // Interventional: P and O active, S inactive
        assert!(!edge.s_active());
        assert!(edge.p_active());
        assert!(edge.o_active());
        assert!(edge.is_interventional());
    }

    #[test]
    #[cfg(not(feature = "causal-edge-v2-layout"))]
    fn test_temporal_in_msb_gives_sort_order() {
        // v1-only: temporal bits 52-63 sort u64-naturally for the v1 layout.
        // Under v2 these bits are W/lens/spare per L-2; sort semantics are
        // intentionally different (chain-position carries time, not the edge).
        let early = CausalEdge64::pack(
            10,
            20,
            30,
            200,
            200,
            CausalMask::SPO,
            0,
            InferenceType::Revision,
            PlasticityState::ALL_HOT,
            100,
        );
        let late = CausalEdge64::pack(
            10,
            20,
            30,
            200,
            200,
            CausalMask::SPO,
            0,
            InferenceType::Revision,
            PlasticityState::ALL_HOT,
            200,
        );

        // Temporal is in MSBs → native u64 sort gives temporal ordering
        assert!(
            late.0 > early.0,
            "Later temporal index should produce larger u64"
        );
    }

    #[test]
    fn test_size() {
        assert_eq!(
            std::mem::size_of::<CausalEdge64>(),
            8,
            "CausalEdge64 must be exactly 8 bytes"
        );
    }

    // ─── matches_causal: query-side Pearl 2³ predicate (TD-INT-7) ────

    fn make_edge(mask: CausalMask) -> CausalEdge64 {
        CausalEdge64::pack(
            10,
            20,
            30,
            200,
            200,
            mask,
            0,
            InferenceType::Deduction,
            PlasticityState::ALL_FROZEN,
            0,
        )
    }

    #[test]
    fn test_matches_causal_full_match() {
        // query_mask == edge_mask: must match.
        let edge = make_edge(CausalMask::PO);
        assert!(edge.matches_causal(CausalMask::PO as u8));
        assert!(edge.matches_causal_mask(CausalMask::PO));

        let edge_spo = make_edge(CausalMask::SPO);
        assert!(edge_spo.matches_causal(CausalMask::SPO as u8));
    }

    #[test]
    fn test_matches_causal_subset_match() {
        // query_mask is a strict subset of edge_mask: must match.
        // SPO (0b111) contains PO (0b011), SO (0b101), SP (0b110), S, P, O.
        let edge = make_edge(CausalMask::SPO);
        assert!(
            edge.matches_causal(CausalMask::PO as u8),
            "SPO edge should match PO query (PO bits are subset of SPO)"
        );
        assert!(
            edge.matches_causal(CausalMask::SO as u8),
            "SPO edge should match SO query"
        );
        assert!(
            edge.matches_causal(CausalMask::P as u8),
            "SPO edge should match single-plane P query"
        );
        assert!(edge.matches_causal_mask(CausalMask::S));

        // PO (0b011) contains O (0b001) and P (0b010), but NOT S (0b100).
        let edge_po = make_edge(CausalMask::PO);
        assert!(edge_po.matches_causal(CausalMask::O as u8));
        assert!(edge_po.matches_causal(CausalMask::P as u8));
    }

    #[test]
    fn test_matches_causal_non_match() {
        // query_mask requires bits the edge does not have: must NOT match.
        // SO edge (0b101) does NOT have the P plane (0b010).
        let edge_so = make_edge(CausalMask::SO);
        assert!(!edge_so.matches_causal(CausalMask::P as u8));
        assert!(
            !edge_so.matches_causal(CausalMask::PO as u8),
            "SO edge must not match PO query — P bit is missing"
        );
        assert!(
            !edge_so.matches_causal_mask(CausalMask::SPO),
            "SO edge must not match SPO query — P bit is missing"
        );

        // P-only edge (0b010) does NOT match SO query (0b101).
        let edge_p = make_edge(CausalMask::P);
        assert!(!edge_p.matches_causal(CausalMask::SO as u8));
        assert!(!edge_p.matches_causal(CausalMask::S as u8));
        assert!(!edge_p.matches_causal(CausalMask::O as u8));
    }

    #[test]
    fn test_matches_causal_zero_mask_matches_anything() {
        // query_mask == 0 has no required bits → vacuously matches every edge.
        // This is the documented semantics: zero is the predicate-true element
        // of the bit lattice (no requirements means nothing to fail).
        for variant in [
            CausalMask::None,
            CausalMask::O,
            CausalMask::P,
            CausalMask::PO,
            CausalMask::S,
            CausalMask::SO,
            CausalMask::SP,
            CausalMask::SPO,
        ] {
            let edge = make_edge(variant);
            assert!(
                edge.matches_causal(0),
                "zero query_mask must match edge with mask {variant:?}"
            );
            assert!(
                edge.matches_causal_mask(CausalMask::None),
                "CausalMask::None query must match edge with mask {variant:?}"
            );
        }
    }

    #[test]
    fn test_matches_causal_high_bits_ignored() {
        // matches_causal must mask query down to the low 3 bits, so callers
        // passing a u8 with stray high bits get the same result as passing
        // the cleaned value.
        let edge = make_edge(CausalMask::PO);
        // 0b1111_0011 → low 3 bits = 0b011 = PO.
        assert!(edge.matches_causal(0b1111_0011));
        // 0b1111_0100 → low 3 bits = 0b100 = S — not present in PO edge.
        assert!(!edge.matches_causal(0b1111_0100));
    }
}
