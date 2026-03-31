//! NARS Engine: causal inference on SPO-decomposed attention heads.
//!
//! Each attention head is NOT a HeadPrint (17 dims). It IS:
//!   - 3 palette indices: S(u8), P(u8), O(u8)
//!   - NARS truth: frequency(u8), confidence(u8)
//!   - Pearl mask: 3 bits selecting which SPO planes are active
//!   - 8 causal projections per head (one per Pearl mask value)
//!
//! Distance between two heads:
//!   d = Σ(active planes) distance_table[self.idx][other.idx]
//!   One table lookup per active plane. O(1) per plane. O(3) worst case.
//!
//! The 34 thinking styles are weight vectors over the 8 Pearl projections.

use super::triple_model::{Truth, truth_revision};

// ── SPO Head (mirrors CausalEdge64 layout) ──

/// One attention head as SPO palette indices + NARS truth + Pearl mask.
/// 8 bytes total — same size as CausalEdge64.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SpoHead {
    pub s_idx: u8,      // Subject palette index (0-255)
    pub p_idx: u8,      // Predicate palette index (0-255)
    pub o_idx: u8,      // Object palette index (0-255)
    pub freq: u8,       // NARS frequency (f = val/255)
    pub conf: u8,       // NARS confidence (c = val/255)
    pub pearl: u8,      // 3-bit Pearl mask (SPO)
    pub inference: u8,  // NARS inference type (0-7)
    pub temporal: u8,   // temporal index (0-255)
}

impl SpoHead {
    pub fn zero() -> Self {
        Self { s_idx: 0, p_idx: 0, o_idx: 0, freq: 128, conf: 0, pearl: 0, inference: 0, temporal: 0 }
    }

    pub fn frequency(&self) -> f32 { self.freq as f32 / 255.0 }
    pub fn confidence(&self) -> f32 { self.conf as f32 / 255.0 }
    pub fn truth(&self) -> Truth { Truth::new(self.frequency(), self.confidence()) }
    pub fn expectation(&self) -> f32 { self.confidence() * (self.frequency() - 0.5) + 0.5 }

    pub fn set_truth(&mut self, t: Truth) {
        self.freq = (t.frequency.clamp(0.0, 1.0) * 255.0).round() as u8;
        self.conf = (t.confidence.clamp(0.0, 0.99) * 255.0).round() as u8;
    }

    // Pearl mask accessors
    pub fn s_active(&self) -> bool { self.pearl & 0b100 != 0 }
    pub fn p_active(&self) -> bool { self.pearl & 0b010 != 0 }
    pub fn o_active(&self) -> bool { self.pearl & 0b001 != 0 }
}

// ── Pearl 2³ Causal Masks ──

pub const MASK_NONE: u8  = 0b000; // prior
pub const MASK_S: u8     = 0b100; // subject marginal
pub const MASK_P: u8     = 0b010; // predicate marginal
pub const MASK_O: u8     = 0b001; // object marginal
pub const MASK_SP: u8    = 0b110; // confounder detection
pub const MASK_SO: u8    = 0b101; // Level 1: Association P(Y|X)
pub const MASK_PO: u8    = 0b011; // Level 2: Intervention P(Y|do(X))
pub const MASK_SPO: u8   = 0b111; // Level 3: Counterfactual

/// All 8 Pearl projections in order.
pub const ALL_MASKS: [u8; 8] = [MASK_NONE, MASK_S, MASK_P, MASK_O, MASK_SP, MASK_SO, MASK_PO, MASK_SPO];

// ── Distance Tables (3 × 256×256 = 384 KB) ──

/// Three SPO distance tables — one per plane.
/// distance = table[a_idx * 256 + b_idx]
pub struct SpoDistances {
    pub s_table: Vec<u16>,  // 256×256 Subject distances
    pub p_table: Vec<u16>,  // 256×256 Predicate distances
    pub o_table: Vec<u16>,  // 256×256 Object distances
}

impl SpoDistances {
    pub fn new_zero() -> Self {
        Self {
            s_table: vec![0u16; 256 * 256],
            p_table: vec![0u16; 256 * 256],
            o_table: vec![0u16; 256 * 256],
        }
    }

    /// O(1) distance for one plane.
    #[inline]
    pub fn s_dist(&self, a: u8, b: u8) -> u16 {
        self.s_table[a as usize * 256 + b as usize]
    }
    #[inline]
    pub fn p_dist(&self, a: u8, b: u8) -> u16 {
        self.p_table[a as usize * 256 + b as usize]
    }
    #[inline]
    pub fn o_dist(&self, a: u8, b: u8) -> u16 {
        self.o_table[a as usize * 256 + b as usize]
    }

    /// Causal distance with Pearl mask: sum active planes. O(3) worst case.
    #[inline]
    pub fn causal_distance(&self, a: &SpoHead, b: &SpoHead, mask: u8) -> u32 {
        let mut d = 0u32;
        if mask & 0b100 != 0 { d += self.s_dist(a.s_idx, b.s_idx) as u32; }
        if mask & 0b010 != 0 { d += self.p_dist(a.p_idx, b.p_idx) as u32; }
        if mask & 0b001 != 0 { d += self.o_dist(a.o_idx, b.o_idx) as u32; }
        d
    }

    /// All 8 Pearl projections at once.
    pub fn all_projections(&self, a: &SpoHead, b: &SpoHead) -> [u32; 8] {
        let mut scores = [0u32; 8];
        for (i, &mask) in ALL_MASKS.iter().enumerate() {
            scores[i] = self.causal_distance(a, b, mask);
        }
        scores
    }
}

// ── NARS Inference Types ──

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum Inference {
    Deduction = 0,   // A→B, B→C ⊢ A→C
    Induction = 1,   // A→B, A→C ⊢ B→C
    Abduction = 2,   // A→B, C→B ⊢ A→C
    Revision = 3,    // merge evidence
    Analogy = 4,     // A→B, C≈A ⊢ C→B
    Resemblance = 5, // A≈B, A≈C ⊢ B≈C
    Synthesis = 6,   // complementary merge
}

// ── NARS Inference on SpoHeads ──

/// Apply NARS inference: produce new truth from two heads.
pub fn nars_infer(a: &SpoHead, b: &SpoHead, rule: Inference) -> Truth {
    let fa = a.frequency();
    let ca = a.confidence();
    let fb = b.frequency();
    let cb = b.confidence();
    match rule {
        Inference::Deduction => {
            let f = fa * fb;
            let c = fa * fb * ca * cb;
            Truth::new(f, c)
        }
        Inference::Induction => {
            let w = fa * ca * cb;
            Truth::new(fb, w / (w + 1.0))
        }
        Inference::Abduction => {
            let w = fb * ca * cb;
            Truth::new(fa, w / (w + 1.0))
        }
        Inference::Revision => {
            truth_revision(a.truth(), b.truth())
        }
        Inference::Analogy => {
            let f = fa * fb;
            let c = fb * ca * cb;
            Truth::new(f, c)
        }
        Inference::Resemblance => {
            let f = fa * fb;
            let c = fa * fb * ca * cb * 0.9; // slight discount
            Truth::new(f, c)
        }
        Inference::Synthesis => {
            let f = (fa + fb) / 2.0;
            let c = (ca + cb) / 2.0;
            Truth::new(f, c)
        }
    }
}

// ── Style Vectors over Pearl 2³ ──

/// A thinking style as weight vector over the 8 Pearl projections.
/// style_weight[i] × causal_distance(mask=ALL_MASKS[i]) → weighted score.
pub struct StyleVector {
    pub name: &'static str,
    pub weights: [f32; 8],  // one weight per Pearl mask
}

/// Predefined style vectors mapping thinking styles to causal emphasis.
pub fn analytical_style() -> StyleVector {
    StyleVector { name: "analytical", weights: [0.0, 0.05, 0.1, 0.1, 0.15, 0.1, 0.2, 0.3] }
    //                                          ___  S__   _P_  __O  SP_   S_O  _PO  SPO
    // Counterfactual (SPO) and Intervention (_PO) weighted highest
}

pub fn creative_style() -> StyleVector {
    StyleVector { name: "creative", weights: [0.15, 0.2, 0.1, 0.15, 0.1, 0.15, 0.1, 0.05] }
    // Prior and Subject marginal weighted highest — free association
}

pub fn empathetic_style() -> StyleVector {
    StyleVector { name: "empathetic", weights: [0.05, 0.1, 0.1, 0.2, 0.1, 0.1, 0.25, 0.1] }
    // Intervention (_PO) and Object (__O) — what effect do I have?
}

pub fn focused_style() -> StyleVector {
    StyleVector { name: "focused", weights: [0.0, 0.0, 0.05, 0.3, 0.05, 0.1, 0.2, 0.3] }
    // Object and Counterfactual — result-oriented
}

pub fn metacognitive_style() -> StyleVector {
    StyleVector { name: "metacognitive", weights: [0.1, 0.1, 0.15, 0.1, 0.2, 0.1, 0.1, 0.15] }
    // Confounder (SP_) weighted — am I confusing correlation with causation?
}

/// Score a candidate using a style vector.
pub fn style_score(
    candidate: &SpoHead,
    context: &SpoHead,
    distances: &SpoDistances,
    style: &StyleVector,
) -> f32 {
    let projections = distances.all_projections(candidate, context);
    let max_dist = 3.0 * 65535.0; // theoretical max across 3 planes
    let mut score = 0.0f32;
    for i in 0..8 {
        let normalized = 1.0 - (projections[i] as f32 / max_dist);
        score += style.weights[i] * normalized;
    }
    score
}

// ── NARS Engine ──

/// Closed-loop NARS feedback engine.
pub struct NarsEngine {
    pub distances: SpoDistances,
    /// Skepticism: grows with consecutive high-confidence outputs.
    pub consecutive_confident: u32,
    /// History of emitted heads with revised truth.
    pub history: Vec<(SpoHead, Truth)>,
}

impl NarsEngine {
    pub fn new(distances: SpoDistances) -> Self {
        Self { distances, consecutive_confident: 0, history: Vec::new() }
    }

    /// After emitting a candidate, record and update skepticism.
    pub fn on_emit(&mut self, head: &SpoHead) {
        if head.confidence() > 0.8 {
            self.consecutive_confident += 1;
        } else {
            self.consecutive_confident = 0;
        }
        self.history.push((*head, head.truth()));
    }

    /// After user responds, revise all history via NARS revision.
    pub fn on_response(&mut self, response: &SpoHead) {
        for (_, truth) in &mut self.history {
            *truth = truth_revision(*truth, response.truth());
        }
    }

    /// Detect contradiction: two heads with similar SPO but opposing truth.
    pub fn detect_contradiction(&self, a: &SpoHead, b: &SpoHead) -> Option<f32> {
        let structural_sim = 1.0 - self.distances.causal_distance(a, b, MASK_SPO) as f32 / (3.0 * 65535.0);
        let truth_conflict = (a.frequency() - b.frequency()).abs();
        if structural_sim > 0.7 && truth_conflict > 0.5 {
            Some(truth_conflict)
        } else {
            None
        }
    }

    /// Mutual entailment: if A→B then B→A (NARS symmetry).
    pub fn mutual_entailment(&self, a: &SpoHead, b: &SpoHead) -> SpoHead {
        let truth = nars_infer(a, b, Inference::Resemblance);
        let mut result = *b;
        result.s_idx = b.o_idx;  // swap S↔O
        result.o_idx = b.s_idx;
        result.set_truth(truth);
        result
    }

    /// Combinatorial entailment: A→B, B→C ⊢ A→C (NARS transitivity).
    pub fn combinatorial_entailment(&self, a: &SpoHead, b: &SpoHead) -> SpoHead {
        let truth = nars_infer(a, b, Inference::Deduction);
        let mut result = SpoHead::zero();
        result.s_idx = a.s_idx;  // A's subject
        result.p_idx = a.p_idx;  // A's predicate
        result.o_idx = b.o_idx;  // C's object (from B→C)
        result.set_truth(truth);
        result
    }

    /// Current skepticism level (log growth with consecutive confidence).
    pub fn skepticism(&self) -> f32 {
        0.1 + (self.consecutive_confident as f32 + 1.0).ln() * 0.1
    }

    /// Should we stop? All history high-confidence + low skepticism growth.
    pub fn should_stop(&self) -> bool {
        if self.history.len() < 3 { return false; }
        let recent: Vec<f32> = self.history.iter().rev().take(3)
            .map(|(_, t)| t.confidence).collect();
        recent.iter().all(|c| *c > 0.85)
    }

    /// Score a candidate with full Pearl 2³ + style vector.
    pub fn score(&self, candidate: &SpoHead, context: &SpoHead, style: &StyleVector) -> f32 {
        let base = style_score(candidate, context, &self.distances, style);
        // Apply skepticism penalty
        base * (1.0 - self.skepticism() * 0.1)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_spo_head_truth_roundtrip() {
        let mut head = SpoHead {
            s_idx: 10, p_idx: 20, o_idx: 30,
            freq: 204, conf: 178, pearl: MASK_SPO,
            inference: 0, temporal: 42,
        };
        let f = head.frequency();
        let c = head.confidence();
        // Roundtrip through Truth and back
        let t = head.truth();
        head.set_truth(t);
        assert!((head.frequency() - f).abs() < 0.01,
            "frequency roundtrip: {} vs {}", head.frequency(), f);
        assert!((head.confidence() - c).abs() < 0.01,
            "confidence roundtrip: {} vs {}", head.confidence(), c);
    }

    #[test]
    fn test_causal_distance_single_plane() {
        let mut dist = SpoDistances::new_zero();
        // Set S distance between idx 5 and idx 10 to 1000
        dist.s_table[5 * 256 + 10] = 1000;

        let a = SpoHead { s_idx: 5, p_idx: 0, o_idx: 0, freq: 128, conf: 128, pearl: 0, inference: 0, temporal: 0 };
        let b = SpoHead { s_idx: 10, p_idx: 0, o_idx: 0, freq: 128, conf: 128, pearl: 0, inference: 0, temporal: 0 };

        // Only S plane active
        assert_eq!(dist.causal_distance(&a, &b, MASK_S), 1000);
        // P plane only: should be 0 (no P distance set)
        assert_eq!(dist.causal_distance(&a, &b, MASK_P), 0);
        // No planes: always 0
        assert_eq!(dist.causal_distance(&a, &b, MASK_NONE), 0);
    }

    #[test]
    fn test_causal_distance_all_planes() {
        let mut dist = SpoDistances::new_zero();
        dist.s_table[1 * 256 + 2] = 100;
        dist.p_table[3 * 256 + 4] = 200;
        dist.o_table[5 * 256 + 6] = 300;

        let a = SpoHead { s_idx: 1, p_idx: 3, o_idx: 5, freq: 128, conf: 128, pearl: 0, inference: 0, temporal: 0 };
        let b = SpoHead { s_idx: 2, p_idx: 4, o_idx: 6, freq: 128, conf: 128, pearl: 0, inference: 0, temporal: 0 };

        assert_eq!(dist.causal_distance(&a, &b, MASK_SPO), 600); // 100 + 200 + 300
        assert_eq!(dist.causal_distance(&a, &b, MASK_SP), 300);  // 100 + 200
        assert_eq!(dist.causal_distance(&a, &b, MASK_PO), 500);  // 200 + 300
        assert_eq!(dist.causal_distance(&a, &b, MASK_SO), 400);  // 100 + 300
    }

    #[test]
    fn test_all_projections() {
        let mut dist = SpoDistances::new_zero();
        dist.s_table[1 * 256 + 2] = 100;
        dist.p_table[3 * 256 + 4] = 200;
        dist.o_table[5 * 256 + 6] = 300;

        let a = SpoHead { s_idx: 1, p_idx: 3, o_idx: 5, freq: 128, conf: 128, pearl: 0, inference: 0, temporal: 0 };
        let b = SpoHead { s_idx: 2, p_idx: 4, o_idx: 6, freq: 128, conf: 128, pearl: 0, inference: 0, temporal: 0 };

        let proj = dist.all_projections(&a, &b);
        assert_eq!(proj[0], 0);    // MASK_NONE
        assert_eq!(proj[1], 100);  // MASK_S
        assert_eq!(proj[2], 200);  // MASK_P
        assert_eq!(proj[3], 300);  // MASK_O
        assert_eq!(proj[4], 300);  // MASK_SP  = 100 + 200
        assert_eq!(proj[5], 400);  // MASK_SO  = 100 + 300
        assert_eq!(proj[6], 500);  // MASK_PO  = 200 + 300
        assert_eq!(proj[7], 600);  // MASK_SPO = 100 + 200 + 300
    }

    #[test]
    fn test_nars_deduction() {
        // High frequency, high confidence inputs
        let a = SpoHead { s_idx: 0, p_idx: 0, o_idx: 0, freq: 204, conf: 178, pearl: 0, inference: 0, temporal: 0 };
        let b = SpoHead { s_idx: 0, p_idx: 0, o_idx: 0, freq: 229, conf: 204, pearl: 0, inference: 0, temporal: 0 };

        let t = nars_infer(&a, &b, Inference::Deduction);
        // Deduction: f = fa * fb, c = fa * fb * ca * cb
        let fa = a.frequency();
        let fb = b.frequency();
        let expected_f = fa * fb;
        assert!((t.frequency - expected_f).abs() < 0.01, "deduction f: {} vs {}", t.frequency, expected_f);
        // Confidence should be less than both inputs (attenuation)
        assert!(t.confidence < a.confidence(), "deduction should attenuate confidence");
    }

    #[test]
    fn test_nars_revision() {
        let a = SpoHead { s_idx: 0, p_idx: 0, o_idx: 0, freq: 204, conf: 127, pearl: 0, inference: 0, temporal: 0 };
        let b = SpoHead { s_idx: 0, p_idx: 0, o_idx: 0, freq: 204, conf: 127, pearl: 0, inference: 0, temporal: 0 };

        let t = nars_infer(&a, &b, Inference::Revision);
        // Revision of two identical truths should increase confidence
        assert!(t.confidence > a.confidence(), "revision should increase confidence: {} vs {}", t.confidence, a.confidence());
        // Frequency should stay roughly the same
        assert!((t.frequency - a.frequency()).abs() < 0.02, "revision should preserve frequency");
    }

    #[test]
    fn test_nars_abduction() {
        let a = SpoHead { s_idx: 0, p_idx: 0, o_idx: 0, freq: 204, conf: 178, pearl: 0, inference: 0, temporal: 0 };
        let b = SpoHead { s_idx: 0, p_idx: 0, o_idx: 0, freq: 229, conf: 204, pearl: 0, inference: 0, temporal: 0 };

        let t = nars_infer(&a, &b, Inference::Abduction);
        // Abduction: f = fa, c = fb * ca * cb / (fb * ca * cb + 1)
        assert!((t.frequency - a.frequency()).abs() < 0.01, "abduction frequency should be fa");
        assert!(t.confidence < 1.0, "abduction confidence should be bounded");
        assert!(t.confidence > 0.0, "abduction confidence should be positive");
    }

    #[test]
    fn test_style_score_analytical_vs_creative() {
        let mut dist = SpoDistances::new_zero();
        // Set up distances so that SPO-level distance is high (counterfactual relevant)
        dist.s_table[1 * 256 + 2] = 10000;
        dist.p_table[3 * 256 + 4] = 10000;
        dist.o_table[5 * 256 + 6] = 10000;

        let candidate = SpoHead { s_idx: 1, p_idx: 3, o_idx: 5, freq: 200, conf: 200, pearl: MASK_SPO, inference: 0, temporal: 0 };
        let context = SpoHead { s_idx: 2, p_idx: 4, o_idx: 6, freq: 200, conf: 200, pearl: MASK_SPO, inference: 0, temporal: 0 };

        let analytical = analytical_style();
        let creative = creative_style();

        let a_score = style_score(&candidate, &context, &dist, &analytical);
        let c_score = style_score(&candidate, &context, &dist, &creative);

        // Both should produce valid scores
        assert!(a_score > 0.0, "analytical score should be positive: {}", a_score);
        assert!(c_score > 0.0, "creative score should be positive: {}", c_score);

        // With large distances, creative (which weights prior/marginals higher) should differ from analytical
        assert!((a_score - c_score).abs() > 0.001,
            "analytical and creative should produce different scores: {} vs {}", a_score, c_score);
    }

    #[test]
    fn test_detect_contradiction() {
        let dist = SpoDistances::new_zero(); // zero distances = structurally identical
        let engine = NarsEngine::new(dist);

        // Same SPO indices, opposing truth
        let a = SpoHead { s_idx: 5, p_idx: 10, o_idx: 15, freq: 230, conf: 200, pearl: MASK_SPO, inference: 0, temporal: 0 };
        let b = SpoHead { s_idx: 5, p_idx: 10, o_idx: 15, freq: 25, conf: 200, pearl: MASK_SPO, inference: 0, temporal: 0 };

        let contradiction = engine.detect_contradiction(&a, &b);
        assert!(contradiction.is_some(), "should detect contradiction");
        let conflict = contradiction.unwrap();
        assert!(conflict > 0.5, "conflict should be > 0.5: {}", conflict);

        // Same truth, same SPO: no contradiction
        let c = SpoHead { s_idx: 5, p_idx: 10, o_idx: 15, freq: 230, conf: 200, pearl: MASK_SPO, inference: 0, temporal: 0 };
        assert!(engine.detect_contradiction(&a, &c).is_none(), "same truth should not be a contradiction");
    }

    #[test]
    fn test_mutual_entailment() {
        let dist = SpoDistances::new_zero();
        let engine = NarsEngine::new(dist);

        let a = SpoHead { s_idx: 10, p_idx: 20, o_idx: 30, freq: 200, conf: 180, pearl: MASK_SPO, inference: 0, temporal: 0 };
        let b = SpoHead { s_idx: 40, p_idx: 50, o_idx: 60, freq: 210, conf: 190, pearl: MASK_SPO, inference: 0, temporal: 0 };

        let result = engine.mutual_entailment(&a, &b);
        // S and O should be swapped from b
        assert_eq!(result.s_idx, b.o_idx, "s_idx should be b's o_idx");
        assert_eq!(result.o_idx, b.s_idx, "o_idx should be b's s_idx");
        assert_eq!(result.p_idx, b.p_idx, "p_idx should remain b's p_idx");
    }

    #[test]
    fn test_combinatorial_entailment() {
        let dist = SpoDistances::new_zero();
        let engine = NarsEngine::new(dist);

        let a = SpoHead { s_idx: 10, p_idx: 20, o_idx: 30, freq: 200, conf: 180, pearl: MASK_SPO, inference: 0, temporal: 0 };
        let b = SpoHead { s_idx: 30, p_idx: 40, o_idx: 50, freq: 210, conf: 190, pearl: MASK_SPO, inference: 0, temporal: 0 };

        let result = engine.combinatorial_entailment(&a, &b);
        // A→B, B→C ⊢ A→C: result has A's subject, A's predicate, B's object
        assert_eq!(result.s_idx, a.s_idx, "should have A's subject");
        assert_eq!(result.p_idx, a.p_idx, "should have A's predicate");
        assert_eq!(result.o_idx, b.o_idx, "should have B's object");
        // Deduction truth should be attenuated
        assert!(result.confidence() < a.confidence(), "deduction should attenuate confidence");
    }

    #[test]
    fn test_skepticism_grows() {
        let dist = SpoDistances::new_zero();
        let mut engine = NarsEngine::new(dist);

        let s0 = engine.skepticism();

        // Emit high-confidence heads
        let confident_head = SpoHead { s_idx: 0, p_idx: 0, o_idx: 0, freq: 200, conf: 220, pearl: 0, inference: 0, temporal: 0 };
        engine.on_emit(&confident_head);
        let s1 = engine.skepticism();
        assert!(s1 > s0, "skepticism should grow after confident emit: {} vs {}", s1, s0);

        engine.on_emit(&confident_head);
        let s2 = engine.skepticism();
        assert!(s2 > s1, "skepticism should keep growing: {} vs {}", s2, s1);

        // Emit low-confidence head: resets consecutive counter
        let uncertain_head = SpoHead { s_idx: 0, p_idx: 0, o_idx: 0, freq: 128, conf: 50, pearl: 0, inference: 0, temporal: 0 };
        engine.on_emit(&uncertain_head);
        let s3 = engine.skepticism();
        assert!(s3 < s2, "skepticism should drop after uncertain emit: {} vs {}", s3, s2);
    }

    #[test]
    fn test_should_stop() {
        let dist = SpoDistances::new_zero();
        let mut engine = NarsEngine::new(dist);

        // Not enough history
        assert!(!engine.should_stop(), "should not stop with empty history");

        // Add 3 high-confidence entries
        let high_truth = Truth::new(0.9, 0.9);
        let head = SpoHead::zero();
        engine.history.push((head, high_truth));
        engine.history.push((head, high_truth));
        assert!(!engine.should_stop(), "should not stop with only 2 entries");

        engine.history.push((head, high_truth));
        assert!(engine.should_stop(), "should stop with 3 high-confidence entries");

        // Add one low-confidence entry
        let low_truth = Truth::new(0.5, 0.3);
        engine.history.push((head, low_truth));
        assert!(!engine.should_stop(), "should not stop after low-confidence entry");
    }
}
