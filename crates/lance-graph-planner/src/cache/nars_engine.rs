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

use super::triple_model::{truth_revision, Truth};

// Causal edge protocol: CausalEdge64 + NarsTables for hot path
#[allow(unused_imports)] // pack_truth intended for hot-path truth packing wiring
use causal_edge::tables::{pack_truth, unpack_c, unpack_f, NarsTables};
pub use causal_edge::CausalEdge64;
pub use causal_edge::CausalMask;
pub use causal_edge::PlasticityState;

// ── SPO Head (mirrors CausalEdge64 layout) ──

/// One attention head as SPO palette indices + NARS truth + Pearl mask.
/// 8 bytes total — same size as CausalEdge64.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SpoHead {
    pub s_idx: u8,     // Subject palette index (0-255)
    pub p_idx: u8,     // Predicate palette index (0-255)
    pub o_idx: u8,     // Object palette index (0-255)
    pub freq: u8,      // NARS frequency (f = val/255)
    pub conf: u8,      // NARS confidence (c = val/255)
    pub pearl: u8,     // 3-bit Pearl mask (SPO)
    pub inference: u8, // NARS inference type (0-7)
    pub temporal: u8,  // temporal index (0-255)
}

impl SpoHead {
    pub fn zero() -> Self {
        Self {
            s_idx: 0,
            p_idx: 0,
            o_idx: 0,
            freq: 128,
            conf: 0,
            pearl: 0,
            inference: 0,
            temporal: 0,
        }
    }

    pub fn frequency(&self) -> f32 {
        self.freq as f32 / 255.0
    }
    pub fn confidence(&self) -> f32 {
        self.conf as f32 / 255.0
    }
    pub fn truth(&self) -> Truth {
        Truth::new(self.frequency(), self.confidence())
    }
    pub fn expectation(&self) -> f32 {
        self.confidence() * (self.frequency() - 0.5) + 0.5
    }

    pub fn set_truth(&mut self, t: Truth) {
        self.freq = (t.frequency.clamp(0.0, 1.0) * 255.0).round() as u8;
        self.conf = (t.confidence.clamp(0.0, 0.99) * 255.0).round() as u8;
    }

    // Pearl mask accessors
    pub fn s_active(&self) -> bool {
        self.pearl & 0b100 != 0
    }
    pub fn p_active(&self) -> bool {
        self.pearl & 0b010 != 0
    }
    pub fn o_active(&self) -> bool {
        self.pearl & 0b001 != 0
    }
}

// ── Pearl 2³ Causal Masks ──

pub const MASK_NONE: u8 = 0b000; // prior
pub const MASK_S: u8 = 0b100; // subject marginal
pub const MASK_P: u8 = 0b010; // predicate marginal
pub const MASK_O: u8 = 0b001; // object marginal
pub const MASK_SP: u8 = 0b110; // confounder detection
pub const MASK_SO: u8 = 0b101; // Level 1: Association P(Y|X)
pub const MASK_PO: u8 = 0b011; // Level 2: Intervention P(Y|do(X))
pub const MASK_SPO: u8 = 0b111; // Level 3: Counterfactual

/// All 8 Pearl projections in order.
pub const ALL_MASKS: [u8; 8] = [
    MASK_NONE, MASK_S, MASK_P, MASK_O, MASK_SP, MASK_SO, MASK_PO, MASK_SPO,
];

// ── Distance Tables (3 × 256×256 = 384 KB) ──

/// Three SPO distance tables — one per plane.
/// distance = table[a_idx * 256 + b_idx]
pub struct SpoDistances {
    pub s_table: Vec<u16>, // 256×256 Subject distances
    pub p_table: Vec<u16>, // 256×256 Predicate distances
    pub o_table: Vec<u16>, // 256×256 Object distances
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
        if mask & 0b100 != 0 {
            d += self.s_dist(a.s_idx, b.s_idx) as u32;
        }
        if mask & 0b010 != 0 {
            d += self.p_dist(a.p_idx, b.p_idx) as u32;
        }
        if mask & 0b001 != 0 {
            d += self.o_dist(a.o_idx, b.o_idx) as u32;
        }
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
    Deduction = 0,      // A→B, B→C ⊢ A→C
    Induction = 1,      // A→B, A→C ⊢ B→C
    Abduction = 2,      // A→B, C→B ⊢ A→C
    Revision = 3,       // merge evidence
    Analogy = 4,        // A→B, C≈A ⊢ C→B
    Resemblance = 5,    // A≈B, A≈C ⊢ B≈C
    Synthesis = 6,      // complementary merge
    /// Pearl rung 2: do-calculus intervention.
    /// Surgically severs the causal mechanism and forces a variable to a value.
    /// Routes through MASK_PO (Predicate + Object planes) — the interventional
    /// projection P(Y | do(X)) excludes the Subject confounding plane.
    /// Confidence modifier: 0.85 (TUNED-LATER per PR-LL-4 GRPO data).
    Intervention = 7,
    /// Pearl rung 3: counterfactual reasoning via abduce→intervene→predict.
    /// Routes through MASK_SPO (all three planes) — full counterfactual
    /// P(Y_x = y | X = x', Y = y') requires subject, predicate, and object.
    /// Confidence modifier: 0.70 (TUNED-LATER; lower than Intervention due to
    /// compounded uncertainty across the 3-step chain).
    Counterfactual = 8,
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
        Inference::Revision => truth_revision(a.truth(), b.truth()),
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
        // Pearl rung 2: Intervention — do-calculus mechanism surgery.
        // Truth semantics: abduction form (infer cause from effect) with the
        // confidence modifier 0.85 applied to reflect mechanism-surgery uncertainty.
        // MASK_PO is the preferred Pearl mask (see `inference_to_pearl_mask`).
        // Implemented as Abduction ×0.85 confidence modifier.
        // TUNED-LATER: replace with dedicated do-calculus truth function once
        // PR-LL-4 GRPO training data provides empirical calibration.
        Inference::Intervention => {
            let w = fb * ca * cb;
            let c = (w / (w + 1.0)) * 0.85;
            Truth::new(fa, c)
        }
        // Pearl rung 3: Counterfactual — abduce→intervene→predict chain.
        // Truth semantics: deduction form with the confidence modifier 0.70 to
        // reflect compounded uncertainty across the 3-step chain.
        // MASK_SPO is the preferred Pearl mask (see `inference_to_pearl_mask`).
        // Implemented as Deduction ×0.70 confidence modifier.
        // TUNED-LATER: replace with dedicated 3-step chain truth function once
        // PR-LL-4 GRPO training data provides empirical calibration.
        Inference::Counterfactual => {
            let f = fa * fb;
            let c = (fa * fb * ca * cb) * 0.70;
            Truth::new(f, c)
        }
    }
}

// ── Style Vectors over Pearl 2³ ──

/// A thinking style as weight vector over the 8 Pearl projections.
/// style_weight[i] × causal_distance(mask=ALL_MASKS[i]) → weighted score.
pub struct StyleVector {
    pub name: &'static str,
    pub weights: [f32; 8], // one weight per Pearl mask
}

/// Predefined style vectors mapping thinking styles to causal emphasis.
pub fn analytical_style() -> StyleVector {
    StyleVector {
        name: "analytical",
        weights: [0.0, 0.05, 0.1, 0.1, 0.15, 0.1, 0.2, 0.3],
    }
    //                                          ___  S__   _P_  __O  SP_   S_O  _PO  SPO
    // Counterfactual (SPO) and Intervention (_PO) weighted highest
}

pub fn creative_style() -> StyleVector {
    StyleVector {
        name: "creative",
        weights: [0.15, 0.2, 0.1, 0.15, 0.1, 0.15, 0.1, 0.05],
    }
    // Prior and Subject marginal weighted highest — free association
}

pub fn empathetic_style() -> StyleVector {
    StyleVector {
        name: "empathetic",
        weights: [0.05, 0.1, 0.1, 0.2, 0.1, 0.1, 0.25, 0.1],
    }
    // Intervention (_PO) and Object (__O) — what effect do I have?
}

pub fn focused_style() -> StyleVector {
    StyleVector {
        name: "focused",
        weights: [0.0, 0.0, 0.05, 0.3, 0.05, 0.1, 0.2, 0.3],
    }
    // Object and Counterfactual — result-oriented
}

pub fn metacognitive_style() -> StyleVector {
    StyleVector {
        name: "metacognitive",
        weights: [0.1, 0.1, 0.15, 0.1, 0.2, 0.1, 0.1, 0.15],
    }
    // Confounder (SP_) weighted — am I confusing correlation with causation?
}

/// Intervention style: Pearl rung 2 — do-calculus traversal.
///
/// Weights the _PO mask (MASK_PO = 0b011, index 6) highest because
/// P(Y | do(X)) severs the subject confounding plane and reasons only
/// through the predicate-to-object causal mechanism. Secondary weight
/// on _P_ (MASK_P = 0b010, index 2) for predicate-marginal evidence.
///
/// Weight vector indices → Pearl masks:
///   [0] MASK_NONE, [1] MASK_S, [2] MASK_P, [3] MASK_O,
///   [4] MASK_SP,   [5] MASK_SO, [6] MASK_PO, [7] MASK_SPO
///
/// Starting calibration — TUNED-LATER once PR-LL-4 GRPO training data
/// provides empirical ground truth for do-calculus mask selection.
pub fn intervention_style() -> StyleVector {
    StyleVector {
        name: "intervention",
        weights: [0.0, 0.0, 0.15, 0.15, 0.0, 0.05, 0.50, 0.15],
    }
    //         ___   S__   _P_   __O   SP_   S_O  _PO   SPO
    // _PO (interventional) weighted highest: do(X) severs S confounding.
    // SPO kept at 0.15 as counterfactual residue; S_O at 0.05 for
    // association fallback when do-calculus degrades gracefully.
}

/// Counterfactual style: Pearl rung 3 — abduce→intervene→predict chain.
///
/// Weights MASK_SPO (index 7) highest because the full counterfactual
/// P(Y_x = y | X = x', Y = y') requires all three SPO planes to be
/// active — subject (background context abduction), predicate (mechanism
/// surgery), and object (outcome prediction). Secondary weight on _PO
/// (MASK_PO, index 6) for the intervene sub-step and __O (MASK_O, index 3)
/// for the predict sub-step.
///
/// Weight vector indices → Pearl masks: same ordering as above.
///
/// Starting calibration — TUNED-LATER once PR-LL-4 GRPO training data
/// provides empirical ground truth for 3-step chain mask selection.
pub fn counterfactual_style() -> StyleVector {
    StyleVector {
        name: "counterfactual",
        weights: [0.0, 0.05, 0.05, 0.10, 0.0, 0.05, 0.25, 0.50],
    }
    //         ___   S__   _P_   __O   SP_   S_O   _PO  SPO
    // SPO weighted at 0.50: all-planes counterfactual query.
    // _PO at 0.25: intervene sub-step.
    // __O at 0.10: predict sub-step (outcome plane).
    // S__ at 0.05: abduce sub-step (background context plane).
}

/// Map a local `Inference` type to the preferred Pearl mask for causal distance.
///
/// Returns the u8 mask value that selects which SPO planes to weight for
/// a given inference type. Consumers can use this mask directly with
/// `SpoDistances::causal_distance()` or route through the matching
/// `StyleVector` via `inference_to_style()`.
///
/// `Intervention` → MASK_PO (0b011): do(X) severs the subject plane.
/// `Counterfactual` → MASK_SPO (0b111): full rung-3 query uses all planes.
/// All other types → MASK_SPO by default (full evidence).
///
/// TODO(PR-LL-4): Tune mask overrides for Deduction (MASK_SO, association),
/// Induction (MASK_SP, confounder surface), Abduction (MASK_PO or SPO).
pub fn inference_to_pearl_mask(rule: Inference) -> u8 {
    match rule {
        // Pearl rung 2: intervention surgically removes subject confounding.
        Inference::Intervention => MASK_PO,
        // Pearl rung 3: counterfactual needs all three planes active.
        Inference::Counterfactual => MASK_SPO,
        // All other types default to full SPO mask (conservative).
        Inference::Deduction
        | Inference::Induction
        | Inference::Abduction
        | Inference::Revision
        | Inference::Analogy
        | Inference::Resemblance
        | Inference::Synthesis => MASK_SPO,
    }
}

/// Map a local `Inference` type to the matching `StyleVector`.
///
/// `Intervention` and `Counterfactual` route to their dedicated Pearl-mask-aware
/// style vectors. All other types fall through to analytical (full-coverage default).
pub fn inference_to_style(rule: Inference) -> StyleVector {
    match rule {
        Inference::Intervention => intervention_style(),
        Inference::Counterfactual => counterfactual_style(),
        // TODO(PR-LL-4): Add per-type style routing for Deduction, Induction,
        // Abduction, Revision once GRPO training data is available.
        _ => analytical_style(),
    }
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
    for (i, &proj) in projections.iter().enumerate().take(8) {
        let normalized = 1.0 - (proj as f32 / max_dist);
        score += style.weights[i] * normalized;
    }
    score
}

// ── NARS Engine ──

/// Closed-loop NARS feedback engine.
///
/// Dual-path: hot path uses NarsTables (u16 lookup, L1 cache),
/// cold path uses f32 Truth for configuration and analysis.
pub struct NarsEngine {
    pub distances: SpoDistances,
    /// Precomputed NARS lookup tables (128 KB, L1 cache resident).
    /// Hot path: revision/deduction as single memory read.
    pub tables: NarsTables,
    /// Skepticism: grows with consecutive high-confidence outputs.
    pub consecutive_confident: u32,
    /// History of emitted heads with revised truth.
    pub history: Vec<(SpoHead, Truth)>,
}

impl NarsEngine {
    pub fn new(distances: SpoDistances) -> Self {
        Self {
            distances,
            tables: NarsTables::build(1), // fast path: 1 c-level = 128 KB
            consecutive_confident: 0,
            history: Vec::new(),
        }
    }

    /// Hot path: NARS revision via lookup table. O(1), no float.
    #[inline]
    pub fn revise_fast(&self, f1: u8, _c1: u8, f2: u8, _c2: u8) -> (u8, u8) {
        let packed = self.tables.deduction[f1 as usize * 256 + f2 as usize];
        (unpack_f(packed), unpack_c(packed))
    }

    /// Hot path: SpoHead → CausalEdge64 for protocol transport.
    ///
    /// Maps the SpoHead's `inference` byte to the causal-edge `InferenceType`,
    /// including the new Pearl rung 2 (`Intervention = 7`) and rung 3
    /// (`Counterfactual = 8`) variants introduced in W1/W3.
    pub fn to_causal_edge(&self, head: &SpoHead) -> CausalEdge64 {
        // Map local `Inference` discriminant to the protocol `InferenceType`.
        // W1 added Intervention=5 and Counterfactual=6 to causal-edge's enum;
        // our local Inference has them as 7/8 respectively, so we must translate.
        let infer_type = match head.inference {
            0 => causal_edge::edge::InferenceType::Deduction,
            1 => causal_edge::edge::InferenceType::Induction,
            2 => causal_edge::edge::InferenceType::Abduction,
            3 => causal_edge::edge::InferenceType::Revision,
            4 => causal_edge::edge::InferenceType::Synthesis,
            // Pearl rung 2: Intervention (local=7 → protocol=5)
            7 => causal_edge::edge::InferenceType::Intervention,
            // Pearl rung 3: Counterfactual (local=8 → protocol=6)
            8 => causal_edge::edge::InferenceType::Counterfactual,
            // Analogy (4), Resemblance (5), Synthesis (6) — map to Synthesis as best fit.
            // 5, 6 in old local enum were Resemblance/Synthesis; handled here:
            5 | 6 => causal_edge::edge::InferenceType::Synthesis,
            _ => causal_edge::edge::InferenceType::Deduction,
        };
        CausalEdge64::pack(
            head.s_idx,
            head.p_idx,
            head.o_idx,
            head.freq,
            head.conf,
            CausalMask::from_bits(head.pearl),
            0, // direction
            infer_type,
            PlasticityState::from_bits(0b111), // all hot
            head.temporal as u16,
        )
    }

    /// Hot path: CausalEdge64 → SpoHead for local processing.
    ///
    /// v2 migration notes (per causal-edge 0.2.0 deprecations):
    /// - `inference` field reads via `inference_mantissa() + InferenceType::from_mantissa()`
    ///   to honor the v2 signed-mantissa encoding (bits 46-49). Direct
    ///   `inference_type() as u8` was correct for v1's 3-bit unsigned field but
    ///   silently misreads v2 negative mantissas (Abduction at −1, Counterfactual
    ///   at −6) as Reserved7/etc.
    /// - `temporal` is structural in v2 (cognitive-substrate-convergence-v1.md L-2);
    ///   bits 52-63 are reclaimed for W-slot/lens/spare. `SpoHead.temporal` is
    ///   set to 0 here. Callers needing chronological order must source it from
    ///   SpoWitnessChain chain-position or AriGraph Triplet.timestamp and write
    ///   it onto the SpoHead via a follow-up structural-temporal accessor (see
    ///   pr-ce64-mb-4-arigraph-spo-g.md for the AriGraph-side surface).
    pub fn from_causal_edge(&self, edge: CausalEdge64) -> SpoHead {
        SpoHead {
            s_idx: edge.s_idx(),
            p_idx: edge.p_idx(),
            o_idx: edge.o_idx(),
            freq: edge.frequency_u8(),
            conf: edge.confidence_u8(),
            pearl: edge.causal_mask() as u8,
            inference: causal_edge::edge::InferenceType::from_mantissa(edge.inference_mantissa()) as u8,
            // v2: temporal moved to structural source (chain-position / Triplet.timestamp);
            // caller responsible for populating chronological context.
            temporal: 0,
        }
    }

    /// Hot path: forward pass via CausalEdge64 compose tables.
    /// Input edge × weight edge → output edge. All O(1).
    pub fn forward_edge(
        &self,
        input: CausalEdge64,
        weight: CausalEdge64,
        compose_s: &[u8; 256 * 256],
        compose_p: &[u8; 256 * 256],
        compose_o: &[u8; 256 * 256],
    ) -> CausalEdge64 {
        input.forward(weight, compose_s, compose_p, compose_o)
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
        let structural_sim =
            1.0 - self.distances.causal_distance(a, b, MASK_SPO) as f32 / (3.0 * 65535.0);
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
        result.s_idx = b.o_idx; // swap S↔O
        result.o_idx = b.s_idx;
        result.set_truth(truth);
        result
    }

    /// Combinatorial entailment: A→B, B→C ⊢ A→C (NARS transitivity).
    pub fn combinatorial_entailment(&self, a: &SpoHead, b: &SpoHead) -> SpoHead {
        let truth = nars_infer(a, b, Inference::Deduction);
        let mut result = SpoHead::zero();
        result.s_idx = a.s_idx; // A's subject
        result.p_idx = a.p_idx; // A's predicate
        result.o_idx = b.o_idx; // C's object (from B→C)
        result.set_truth(truth);
        result
    }

    /// Current skepticism level (log growth with consecutive confidence).
    pub fn skepticism(&self) -> f32 {
        0.1 + (self.consecutive_confident as f32 + 1.0).ln() * 0.1
    }

    /// Should we stop? All history high-confidence + low skepticism growth.
    pub fn should_stop(&self) -> bool {
        if self.history.len() < 3 {
            return false;
        }
        let recent: Vec<f32> = self
            .history
            .iter()
            .rev()
            .take(3)
            .map(|(_, t)| t.confidence)
            .collect();
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
            s_idx: 10,
            p_idx: 20,
            o_idx: 30,
            freq: 204,
            conf: 178,
            pearl: MASK_SPO,
            inference: 0,
            temporal: 42,
        };
        let f = head.frequency();
        let c = head.confidence();
        // Roundtrip through Truth and back
        let t = head.truth();
        head.set_truth(t);
        assert!(
            (head.frequency() - f).abs() < 0.01,
            "frequency roundtrip: {} vs {}",
            head.frequency(),
            f
        );
        assert!(
            (head.confidence() - c).abs() < 0.01,
            "confidence roundtrip: {} vs {}",
            head.confidence(),
            c
        );
    }

    #[test]
    fn test_causal_distance_single_plane() {
        let mut dist = SpoDistances::new_zero();
        // Set S distance between idx 5 and idx 10 to 1000
        dist.s_table[5 * 256 + 10] = 1000;

        let a = SpoHead {
            s_idx: 5,
            p_idx: 0,
            o_idx: 0,
            freq: 128,
            conf: 128,
            pearl: 0,
            inference: 0,
            temporal: 0,
        };
        let b = SpoHead {
            s_idx: 10,
            p_idx: 0,
            o_idx: 0,
            freq: 128,
            conf: 128,
            pearl: 0,
            inference: 0,
            temporal: 0,
        };

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
        dist.s_table[256 + 2] = 100;
        dist.p_table[3 * 256 + 4] = 200;
        dist.o_table[5 * 256 + 6] = 300;

        let a = SpoHead {
            s_idx: 1,
            p_idx: 3,
            o_idx: 5,
            freq: 128,
            conf: 128,
            pearl: 0,
            inference: 0,
            temporal: 0,
        };
        let b = SpoHead {
            s_idx: 2,
            p_idx: 4,
            o_idx: 6,
            freq: 128,
            conf: 128,
            pearl: 0,
            inference: 0,
            temporal: 0,
        };

        assert_eq!(dist.causal_distance(&a, &b, MASK_SPO), 600); // 100 + 200 + 300
        assert_eq!(dist.causal_distance(&a, &b, MASK_SP), 300); // 100 + 200
        assert_eq!(dist.causal_distance(&a, &b, MASK_PO), 500); // 200 + 300
        assert_eq!(dist.causal_distance(&a, &b, MASK_SO), 400); // 100 + 300
    }

    #[test]
    fn test_all_projections() {
        let mut dist = SpoDistances::new_zero();
        dist.s_table[256 + 2] = 100;
        dist.p_table[3 * 256 + 4] = 200;
        dist.o_table[5 * 256 + 6] = 300;

        let a = SpoHead {
            s_idx: 1,
            p_idx: 3,
            o_idx: 5,
            freq: 128,
            conf: 128,
            pearl: 0,
            inference: 0,
            temporal: 0,
        };
        let b = SpoHead {
            s_idx: 2,
            p_idx: 4,
            o_idx: 6,
            freq: 128,
            conf: 128,
            pearl: 0,
            inference: 0,
            temporal: 0,
        };

        let proj = dist.all_projections(&a, &b);
        assert_eq!(proj[0], 0); // MASK_NONE
        assert_eq!(proj[1], 100); // MASK_S
        assert_eq!(proj[2], 200); // MASK_P
        assert_eq!(proj[3], 300); // MASK_O
        assert_eq!(proj[4], 300); // MASK_SP  = 100 + 200
        assert_eq!(proj[5], 400); // MASK_SO  = 100 + 300
        assert_eq!(proj[6], 500); // MASK_PO  = 200 + 300
        assert_eq!(proj[7], 600); // MASK_SPO = 100 + 200 + 300
    }

    #[test]
    fn test_nars_deduction() {
        // High frequency, high confidence inputs
        let a = SpoHead {
            s_idx: 0,
            p_idx: 0,
            o_idx: 0,
            freq: 204,
            conf: 178,
            pearl: 0,
            inference: 0,
            temporal: 0,
        };
        let b = SpoHead {
            s_idx: 0,
            p_idx: 0,
            o_idx: 0,
            freq: 229,
            conf: 204,
            pearl: 0,
            inference: 0,
            temporal: 0,
        };

        let t = nars_infer(&a, &b, Inference::Deduction);
        // Deduction: f = fa * fb, c = fa * fb * ca * cb
        let fa = a.frequency();
        let fb = b.frequency();
        let expected_f = fa * fb;
        assert!(
            (t.frequency - expected_f).abs() < 0.01,
            "deduction f: {} vs {}",
            t.frequency,
            expected_f
        );
        // Confidence should be less than both inputs (attenuation)
        assert!(
            t.confidence < a.confidence(),
            "deduction should attenuate confidence"
        );
    }

    #[test]
    fn test_nars_revision() {
        let a = SpoHead {
            s_idx: 0,
            p_idx: 0,
            o_idx: 0,
            freq: 204,
            conf: 127,
            pearl: 0,
            inference: 0,
            temporal: 0,
        };
        let b = SpoHead {
            s_idx: 0,
            p_idx: 0,
            o_idx: 0,
            freq: 204,
            conf: 127,
            pearl: 0,
            inference: 0,
            temporal: 0,
        };

        let t = nars_infer(&a, &b, Inference::Revision);
        // Revision of two identical truths should increase confidence
        assert!(
            t.confidence > a.confidence(),
            "revision should increase confidence: {} vs {}",
            t.confidence,
            a.confidence()
        );
        // Frequency should stay roughly the same
        assert!(
            (t.frequency - a.frequency()).abs() < 0.02,
            "revision should preserve frequency"
        );
    }

    #[test]
    fn test_nars_abduction() {
        let a = SpoHead {
            s_idx: 0,
            p_idx: 0,
            o_idx: 0,
            freq: 204,
            conf: 178,
            pearl: 0,
            inference: 0,
            temporal: 0,
        };
        let b = SpoHead {
            s_idx: 0,
            p_idx: 0,
            o_idx: 0,
            freq: 229,
            conf: 204,
            pearl: 0,
            inference: 0,
            temporal: 0,
        };

        let t = nars_infer(&a, &b, Inference::Abduction);
        // Abduction: f = fa, c = fb * ca * cb / (fb * ca * cb + 1)
        assert!(
            (t.frequency - a.frequency()).abs() < 0.01,
            "abduction frequency should be fa"
        );
        assert!(t.confidence < 1.0, "abduction confidence should be bounded");
        assert!(
            t.confidence > 0.0,
            "abduction confidence should be positive"
        );
    }

    #[test]
    fn test_nars_intervention_and_counterfactual() {
        let a = SpoHead {
            s_idx: 0,
            p_idx: 0,
            o_idx: 0,
            freq: 204,
            conf: 178,
            pearl: MASK_PO, // Intervention uses PO mask
            inference: 7,   // Inference::Intervention
            temporal: 0,
        };
        let b = SpoHead {
            s_idx: 0,
            p_idx: 0,
            o_idx: 0,
            freq: 229,
            conf: 204,
            pearl: MASK_SPO, // Counterfactual uses SPO mask
            inference: 8,    // Inference::Counterfactual
            temporal: 0,
        };

        // Intervention truth: confidence should be capped by 0.85 modifier
        let t_int = nars_infer(&a, &b, Inference::Intervention);
        assert!(
            (t_int.frequency - a.frequency()).abs() < 0.01,
            "Intervention frequency should be fa = {}, got {}",
            a.frequency(),
            t_int.frequency
        );
        assert!(
            t_int.confidence < a.confidence(),
            "Intervention confidence should be attenuated: got {}",
            t_int.confidence
        );

        // Counterfactual truth: confidence further attenuated by 0.70 modifier
        let t_cf = nars_infer(&a, &b, Inference::Counterfactual);
        assert!(
            t_cf.confidence < t_int.confidence,
            "Counterfactual confidence should be lower than Intervention: {} vs {}",
            t_cf.confidence,
            t_int.confidence
        );

        // Pearl mask routing
        assert_eq!(
            inference_to_pearl_mask(Inference::Intervention),
            MASK_PO,
            "Intervention should route to MASK_PO"
        );
        assert_eq!(
            inference_to_pearl_mask(Inference::Counterfactual),
            MASK_SPO,
            "Counterfactual should route to MASK_SPO"
        );
    }

    #[test]
    fn test_intervention_style_weights_po_highest() {
        let style = intervention_style();
        // MASK_PO is at index 6 in ALL_MASKS
        let po_weight = style.weights[6];
        for (i, &w) in style.weights.iter().enumerate() {
            if i != 6 {
                assert!(
                    po_weight >= w,
                    "intervention_style: PO weight ({}) should be >= weights[{}] ({})",
                    po_weight,
                    i,
                    w
                );
            }
        }
    }

    #[test]
    fn test_counterfactual_style_weights_spo_highest() {
        let style = counterfactual_style();
        // MASK_SPO is at index 7 in ALL_MASKS
        let spo_weight = style.weights[7];
        for (i, &w) in style.weights.iter().enumerate() {
            if i != 7 {
                assert!(
                    spo_weight >= w,
                    "counterfactual_style: SPO weight ({}) should be >= weights[{}] ({})",
                    spo_weight,
                    i,
                    w
                );
            }
        }
    }

    #[test]
    fn test_to_causal_edge_maps_intervention_and_counterfactual() {
        let dist = SpoDistances::new_zero();
        let engine = NarsEngine::new(dist);

        let int_head = SpoHead {
            s_idx: 5,
            p_idx: 10,
            o_idx: 15,
            freq: 200,
            conf: 180,
            pearl: MASK_PO,
            inference: 7, // Intervention
            temporal: 1,
        };
        let cf_head = SpoHead {
            s_idx: 5,
            p_idx: 10,
            o_idx: 15,
            freq: 200,
            conf: 180,
            pearl: MASK_SPO,
            inference: 8, // Counterfactual
            temporal: 2,
        };

        let int_edge = engine.to_causal_edge(&int_head);
        let cf_edge = engine.to_causal_edge(&cf_head);

        assert_eq!(
            int_edge.inference_type(),
            causal_edge::edge::InferenceType::Intervention,
            "SpoHead inference=7 should map to Intervention"
        );
        assert_eq!(
            cf_edge.inference_type(),
            causal_edge::edge::InferenceType::Counterfactual,
            "SpoHead inference=8 should map to Counterfactual"
        );

        // Verify causal masks are preserved
        assert!(
            int_edge.p_active() && int_edge.o_active() && !int_edge.s_active(),
            "Intervention edge should have PO mask"
        );
        assert!(
            cf_edge.s_active() && cf_edge.p_active() && cf_edge.o_active(),
            "Counterfactual edge should have SPO mask"
        );
    }

    #[test]
    fn test_style_score_analytical_vs_creative() {
        let mut dist = SpoDistances::new_zero();
        // Set up distances so that SPO-level distance is high (counterfactual relevant)
        dist.s_table[256 + 2] = 10000;
        dist.p_table[3 * 256 + 4] = 10000;
        dist.o_table[5 * 256 + 6] = 10000;

        let candidate = SpoHead {
            s_idx: 1,
            p_idx: 3,
            o_idx: 5,
            freq: 200,
            conf: 200,
            pearl: MASK_SPO,
            inference: 0,
            temporal: 0,
        };
        let context = SpoHead {
            s_idx: 2,
            p_idx: 4,
            o_idx: 6,
            freq: 200,
            conf: 200,
            pearl: MASK_SPO,
            inference: 0,
            temporal: 0,
        };

        let analytical = analytical_style();
        let creative = creative_style();

        let a_score = style_score(&candidate, &context, &dist, &analytical);
        let c_score = style_score(&candidate, &context, &dist, &creative);

        // Both should produce valid scores
        assert!(
            a_score > 0.0,
            "analytical score should be positive: {}",
            a_score
        );
        assert!(
            c_score > 0.0,
            "creative score should be positive: {}",
            c_score
        );

        // With large distances, creative (which weights prior/marginals higher) should differ from analytical
        assert!(
            (a_score - c_score).abs() > 0.001,
            "analytical and creative should produce different scores: {} vs {}",
            a_score,
            c_score
        );
    }

    #[test]
    fn test_detect_contradiction() {
        let dist = SpoDistances::new_zero(); // zero distances = structurally identical
        let engine = NarsEngine::new(dist);

        // Same SPO indices, opposing truth
        let a = SpoHead {
            s_idx: 5,
            p_idx: 10,
            o_idx: 15,
            freq: 230,
            conf: 200,
            pearl: MASK_SPO,
            inference: 0,
            temporal: 0,
        };
        let b = SpoHead {
            s_idx: 5,
            p_idx: 10,
            o_idx: 15,
            freq: 25,
            conf: 200,
            pearl: MASK_SPO,
            inference: 0,
            temporal: 0,
        };

        let contradiction = engine.detect_contradiction(&a, &b);
        assert!(contradiction.is_some(), "should detect contradiction");
        let conflict = contradiction.unwrap();
        assert!(conflict > 0.5, "conflict should be > 0.5: {}", conflict);

        // Same truth, same SPO: no contradiction
        let c = SpoHead {
            s_idx: 5,
            p_idx: 10,
            o_idx: 15,
            freq: 230,
            conf: 200,
            pearl: MASK_SPO,
            inference: 0,
            temporal: 0,
        };
        assert!(
            engine.detect_contradiction(&a, &c).is_none(),
            "same truth should not be a contradiction"
        );
    }

    #[test]
    fn test_mutual_entailment() {
        let dist = SpoDistances::new_zero();
        let engine = NarsEngine::new(dist);

        let a = SpoHead {
            s_idx: 10,
            p_idx: 20,
            o_idx: 30,
            freq: 200,
            conf: 180,
            pearl: MASK_SPO,
            inference: 0,
            temporal: 0,
        };
        let b = SpoHead {
            s_idx: 40,
            p_idx: 50,
            o_idx: 60,
            freq: 210,
            conf: 190,
            pearl: MASK_SPO,
            inference: 0,
            temporal: 0,
        };

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

        let a = SpoHead {
            s_idx: 10,
            p_idx: 20,
            o_idx: 30,
            freq: 200,
            conf: 180,
            pearl: MASK_SPO,
            inference: 0,
            temporal: 0,
        };
        let b = SpoHead {
            s_idx: 30,
            p_idx: 40,
            o_idx: 50,
            freq: 210,
            conf: 190,
            pearl: MASK_SPO,
            inference: 0,
            temporal: 0,
        };

        let result = engine.combinatorial_entailment(&a, &b);
        // A→B, B→C ⊢ A→C: result has A's subject, A's predicate, B's object
        assert_eq!(result.s_idx, a.s_idx, "should have A's subject");
        assert_eq!(result.p_idx, a.p_idx, "should have A's predicate");
        assert_eq!(result.o_idx, b.o_idx, "should have B's object");
        // Deduction truth should be attenuated
        assert!(
            result.confidence() < a.confidence(),
            "deduction should attenuate confidence"
        );
    }

    #[test]
    fn test_skepticism_grows() {
        let dist = SpoDistances::new_zero();
        let mut engine = NarsEngine::new(dist);

        let s0 = engine.skepticism();

        // Emit high-confidence heads
        let confident_head = SpoHead {
            s_idx: 0,
            p_idx: 0,
            o_idx: 0,
            freq: 200,
            conf: 220,
            pearl: 0,
            inference: 0,
            temporal: 0,
        };
        engine.on_emit(&confident_head);
        let s1 = engine.skepticism();
        assert!(
            s1 > s0,
            "skepticism should grow after confident emit: {} vs {}",
            s1,
            s0
        );

        engine.on_emit(&confident_head);
        let s2 = engine.skepticism();
        assert!(s2 > s1, "skepticism should keep growing: {} vs {}", s2, s1);

        // Emit low-confidence head: resets consecutive counter
        let uncertain_head = SpoHead {
            s_idx: 0,
            p_idx: 0,
            o_idx: 0,
            freq: 128,
            conf: 50,
            pearl: 0,
            inference: 0,
            temporal: 0,
        };
        engine.on_emit(&uncertain_head);
        let s3 = engine.skepticism();
        assert!(
            s3 < s2,
            "skepticism should drop after uncertain emit: {} vs {}",
            s3,
            s2
        );
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
        assert!(
            engine.should_stop(),
            "should stop with 3 high-confidence entries"
        );

        // Add one low-confidence entry
        let low_truth = Truth::new(0.5, 0.3);
        engine.history.push((head, low_truth));
        assert!(
            !engine.should_stop(),
            "should not stop after low-confidence entry"
        );
    }
}
