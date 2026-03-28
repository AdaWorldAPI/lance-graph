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
    /// Reserved for future inference types.
    Reserved5 = 5,
    Reserved6 = 6,
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
            5 => Self::Reserved5,
            6 => Self::Reserved6,
            _ => Self::Reserved7,
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

    /// Pack all fields into a CausalEdge64.
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
        v |= ((inference as u8 as u64) & BITS3_MASK) << INFER_SHIFT;
        v |= ((plasticity.bits() as u64) & BITS3_MASK) << PLAST_SHIFT;
        v |= ((temporal as u64) & BITS12_MASK) << TEMPORAL_SHIFT;
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
        if c >= 0.999 { f32::MAX } else { c / (1.0 - c) }
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

    /// Is the S-plane active in the current causal projection?
    #[inline(always)]
    pub fn s_active(self) -> bool { (self.0 >> CAUSAL_SHIFT) & 0b100 != 0 }

    /// Is the P-plane active in the current causal projection?
    #[inline(always)]
    pub fn p_active(self) -> bool { (self.0 >> CAUSAL_SHIFT) & 0b010 != 0 }

    /// Is the O-plane active in the current causal projection?
    #[inline(always)]
    pub fn o_active(self) -> bool { (self.0 >> CAUSAL_SHIFT) & 0b001 != 0 }

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
        self.0 = (self.0 & !(BITS3_MASK << DIR_SHIFT))
            | (((d as u64) & BITS3_MASK) << DIR_SHIFT);
    }

    /// Is the subject plane pathological (dim0 negative)?
    #[inline(always)]
    pub fn s_pathological(self) -> bool { self.direction() & 0b001 != 0 }

    /// Is the predicate plane pathological?
    #[inline(always)]
    pub fn p_pathological(self) -> bool { self.direction() & 0b010 != 0 }

    /// Is the outcome plane pathological?
    #[inline(always)]
    pub fn o_pathological(self) -> bool { self.direction() & 0b100 != 0 }

    // ─── Inference Type ─────────────────────────────────────────────

    /// NARS inference type for this edge.
    #[inline(always)]
    pub fn inference_type(self) -> InferenceType {
        InferenceType::from_bits(((self.0 >> INFER_SHIFT) & BITS3_MASK) as u8)
    }

    /// Set inference type.
    #[inline]
    pub fn set_inference_type(&mut self, t: InferenceType) {
        self.0 = (self.0 & !(BITS3_MASK << INFER_SHIFT))
            | (((t as u8 as u64) & BITS3_MASK) << INFER_SHIFT);
    }

    // ─── Plasticity ─────────────────────────────────────────────────

    /// Plasticity state for all three planes.
    #[inline(always)]
    pub fn plasticity(self) -> PlasticityState {
        PlasticityState::from_bits(((self.0 >> PLAST_SHIFT) & BITS3_MASK) as u8)
    }

    /// Set plasticity state.
    #[inline]
    pub fn set_plasticity(&mut self, p: PlasticityState) {
        self.0 = (self.0 & !(BITS3_MASK << PLAST_SHIFT))
            | (((p.bits() as u64) & BITS3_MASK) << PLAST_SHIFT);
    }

    // ─── Temporal Index ─────────────────────────────────────────────

    /// 12-bit temporal index (0..4095).
    #[inline(always)]
    pub fn temporal(self) -> u16 {
        ((self.0 >> TEMPORAL_SHIFT) & BITS12_MASK) as u16
    }

    /// Set temporal index.
    #[inline]
    pub fn set_temporal(&mut self, t: u16) {
        self.0 = (self.0 & !(BITS12_MASK << TEMPORAL_SHIFT))
            | (((t as u64) & BITS12_MASK) << TEMPORAL_SHIFT);
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
        let (f_out, c_out) = match weight.inference_type() {
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
        let mask_out = CausalMask::from_bits(
            (self.causal_mask() as u8) & (weight.causal_mask() as u8),
        );

        // 4. Temporal: latest of the two
        let t_out = self.temporal().max(weight.temporal());

        // 5. Inherit plasticity from weight (the "learned" edge)
        //    and direction will be recomputed from composed palette entries
        Self::pack(
            s_out,
            p_out,
            o_out,
            (f_out.clamp(0.0, 1.0) * 255.0).round() as u8,
            (c_out.clamp(0.0, 1.0) * 255.0).round() as u8,
            mask_out,
            weight.direction(), // TODO: recompute from composed palette dim0 signs
            weight.inference_type(),
            weight.plasticity(),
            t_out,
        )
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
                if observation.s_idx() == self.s_idx() { new_plast = new_plast.freeze_s(); }
                if observation.p_idx() == self.p_idx() { new_plast = new_plast.freeze_p(); }
                if observation.o_idx() == self.o_idx() { new_plast = new_plast.freeze_o(); }
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
        self.causal_mask() == CausalMask::SPO
            && self.confidence() >= confidence_threshold
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
    fn test_roundtrip() {
        let edge = CausalEdge64::pack(
            143, 7, 201,           // SPO palette indices
            209, 181,              // f=0.82, c=0.71
            CausalMask::PO,        // interventional level
            0b101,                 // S and O pathological
            InferenceType::Deduction,
            PlasticityState::S_HOT,
            42,                    // temporal index
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
            10, 20, 30, 204, 178, // f=0.80, c=0.70
            CausalMask::SPO, 0, InferenceType::Deduction,
            PlasticityState::ALL_HOT, 1,
        );
        let weight = CausalEdge64::pack(
            40, 50, 60, 229, 204, // f=0.90, c=0.80
            CausalMask::SPO, 0, InferenceType::Deduction,
            PlasticityState::ALL_FROZEN, 2,
        );

        // Dummy compose tables (identity for test)
        let compose = [0u8; 256 * 256];
        let result = input.forward(weight, &compose, &compose, &compose);

        // Deduction: f_out = f_in * f_w ≈ 0.80 * 0.90 = 0.72
        assert!(result.frequency() < input.frequency(),
            "Deduction should attenuate frequency: got {}", result.frequency());
        // Confidence should also attenuate
        assert!(result.confidence() < input.confidence(),
            "Deduction should attenuate confidence: got {}", result.confidence());
    }

    #[test]
    fn test_learn_increases_confidence() {
        let mut edge = CausalEdge64::pack(
            10, 20, 30, 204, 127, // f=0.80, c=0.50
            CausalMask::SPO, 0, InferenceType::Revision,
            PlasticityState::ALL_HOT, 1,
        );
        let observation = CausalEdge64::pack(
            10, 20, 30, 204, 127, // same frequency, same confidence
            CausalMask::SPO, 0, InferenceType::Revision,
            PlasticityState::ALL_HOT, 2,
        );

        let c_before = edge.confidence();
        edge.learn(observation, 3);
        assert!(edge.confidence() > c_before,
            "Learning from agreeing evidence should increase confidence");
    }

    #[test]
    fn test_causal_mask_projection() {
        let edge = CausalEdge64::pack(
            10, 20, 30, 200, 200,
            CausalMask::PO, 0, InferenceType::Deduction,
            PlasticityState::ALL_FROZEN, 0,
        );

        // Interventional: P and O active, S inactive
        assert!(!edge.s_active());
        assert!(edge.p_active());
        assert!(edge.o_active());
        assert!(edge.is_interventional());
    }

    #[test]
    fn test_temporal_in_msb_gives_sort_order() {
        let early = CausalEdge64::pack(
            10, 20, 30, 200, 200,
            CausalMask::SPO, 0, InferenceType::Revision,
            PlasticityState::ALL_HOT, 100,
        );
        let late = CausalEdge64::pack(
            10, 20, 30, 200, 200,
            CausalMask::SPO, 0, InferenceType::Revision,
            PlasticityState::ALL_HOT, 200,
        );

        // Temporal is in MSBs → native u64 sort gives temporal ordering
        assert!(late.0 > early.0,
            "Later temporal index should produce larger u64");
    }

    #[test]
    fn test_size() {
        assert_eq!(std::mem::size_of::<CausalEdge64>(), 8,
            "CausalEdge64 must be exactly 8 bytes");
    }
}
