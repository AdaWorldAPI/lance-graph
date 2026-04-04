//! 10-Layer Cognitive Stack with 12 Thinking Styles.
//!
//! Migrated from ladybug-rs/src/cognitive/ + agi-chat/src/thinking/.
//!
//! ```text
//! L10 Crystallization  — what survives becomes system (→ KG + ghosts)
//! L9  Validation        — NARS + Brier + SPO sieve
//! L8  Integration       — evidence merge, superposition field
//! L7  Contingency       — novelty gate, could-be-otherwise
//! L6  Delegation        — multi-lens fanout
//! ─── single agent boundary ───
//! L5  Execution         — MatVec cycle (F32x16 SIMD)
//! L4  Routing           — AUTOPOIESIS: ghost bias + style selection
//! L3  Appraisal         — gestalt, superposition eval
//! L2  Resonance         — domino cascade (field binding)
//! L1  Recognition       — tokenize → codebook lookup
//! ```

/// 10 cognitive layers.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum LayerId {
    L1,  // Recognition
    L2,  // Resonance
    L3,  // Appraisal
    L4,  // Routing (autopoiesis)
    L5,  // Execution
    L6,  // Delegation
    L7,  // Contingency
    L8,  // Integration
    L9,  // Validation
    L10, // Crystallization
}

impl LayerId {
    pub fn as_u8(&self) -> u8 {
        match self {
            Self::L1 => 1, Self::L2 => 2, Self::L3 => 3, Self::L4 => 4, Self::L5 => 5,
            Self::L6 => 6, Self::L7 => 7, Self::L8 => 8, Self::L9 => 9, Self::L10 => 10,
        }
    }
    pub fn is_single_agent(&self) -> bool { self.as_u8() <= 5 }
    pub fn is_multi_agent(&self) -> bool { self.as_u8() > 5 }
}

/// 12 thinking styles with calibrated parameters.
/// From ladybug-rs/src/cognitive/unified_stack.rs.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
pub enum ThinkingStyle {
    #[default]
    Deliberate,     // balanced default
    Analytical,     // tight focus, few branches
    Convergent,     // narrowing toward answer
    Systematic,     // methodical coverage
    Creative,       // wide exploration, many branches
    Divergent,      // actively seeking alternatives
    Exploratory,    // maximum breadth
    Focused,        // single-point attention
    Diffuse,        // soft attention, peripheral
    Peripheral,     // edge detection, anomalies
    Intuitive,      // fast pattern match, trust experience
    Metacognitive,  // thinking about thinking
}

/// Parameters that modulate cascade behavior per style.
#[derive(Clone, Copy, Debug)]
pub struct StyleParams {
    /// Minimum similarity to count as resonant (higher = stricter).
    pub resonance_threshold: f32,
    /// Maximum branches to explore per stage.
    pub fan_out: usize,
    /// How much to favor novelty vs familiarity (0=exploit, 1=explore).
    pub exploration: f32,
    /// Processing speed priority (0=thorough, 1=fast).
    pub speed: f32,
    /// Bias toward collapsing (negative=hold longer, positive=commit faster).
    pub collapse_bias: f32,
}

impl ThinkingStyle {
    /// Get calibrated parameters for this style.
    pub fn params(&self) -> StyleParams {
        match self {
            Self::Analytical    => StyleParams { resonance_threshold: 0.85, fan_out: 3,  exploration: 0.05, speed: 0.1, collapse_bias: -0.10 },
            Self::Convergent    => StyleParams { resonance_threshold: 0.75, fan_out: 4,  exploration: 0.10, speed: 0.3, collapse_bias: -0.05 },
            Self::Systematic    => StyleParams { resonance_threshold: 0.70, fan_out: 5,  exploration: 0.10, speed: 0.2, collapse_bias:  0.00 },
            Self::Creative      => StyleParams { resonance_threshold: 0.35, fan_out: 12, exploration: 0.80, speed: 0.5, collapse_bias:  0.15 },
            Self::Divergent     => StyleParams { resonance_threshold: 0.40, fan_out: 10, exploration: 0.70, speed: 0.4, collapse_bias:  0.10 },
            Self::Exploratory   => StyleParams { resonance_threshold: 0.30, fan_out: 15, exploration: 0.90, speed: 0.6, collapse_bias:  0.20 },
            Self::Focused       => StyleParams { resonance_threshold: 0.90, fan_out: 1,  exploration: 0.00, speed: 0.2, collapse_bias: -0.15 },
            Self::Diffuse       => StyleParams { resonance_threshold: 0.45, fan_out: 8,  exploration: 0.40, speed: 0.5, collapse_bias:  0.05 },
            Self::Peripheral    => StyleParams { resonance_threshold: 0.20, fan_out: 20, exploration: 0.60, speed: 0.7, collapse_bias:  0.25 },
            Self::Intuitive     => StyleParams { resonance_threshold: 0.50, fan_out: 3,  exploration: 0.30, speed: 0.9, collapse_bias:  0.00 },
            Self::Deliberate    => StyleParams { resonance_threshold: 0.70, fan_out: 7,  exploration: 0.20, speed: 0.1, collapse_bias: -0.05 },
            Self::Metacognitive => StyleParams { resonance_threshold: 0.50, fan_out: 5,  exploration: 0.30, speed: 0.3, collapse_bias:  0.00 },
        }
    }

    /// Butterfly sensitivity: noise tolerance (low = sensitive to small changes).
    pub fn butterfly_sensitivity(&self) -> f32 {
        match self {
            Self::Peripheral    => 0.10,
            Self::Exploratory   => 0.15,
            Self::Creative      => 0.20,
            Self::Diffuse       => 0.25,
            Self::Intuitive     => 0.30,
            Self::Divergent     => 0.35,
            Self::Metacognitive => 0.40,
            Self::Deliberate    => 0.50,
            Self::Systematic    => 0.60,
            Self::Convergent    => 0.70,
            Self::Analytical    => 0.80,
            Self::Focused       => 0.90,
        }
    }

    pub fn all() -> &'static [ThinkingStyle] {
        &[
            Self::Deliberate, Self::Analytical, Self::Convergent, Self::Systematic,
            Self::Creative, Self::Divergent, Self::Exploratory, Self::Focused,
            Self::Diffuse, Self::Peripheral, Self::Intuitive, Self::Metacognitive,
        ]
    }
}

impl std::fmt::Display for ThinkingStyle {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "{:?}", self)
    }
}

/// Collapse gate: decides when a thought crystallizes.
/// SD = standard deviation of candidate scores.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GateState {
    /// SD < 0.15: consensus reached, commit the thought.
    Flow,
    /// 0.15 ≤ SD ≤ 0.35: superposition, keep exploring.
    Hold,
    /// SD > 0.35: high variance, switch style or clarify.
    Block,
}

pub const SD_FLOW_THRESHOLD: f32 = 0.15;
pub const SD_BLOCK_THRESHOLD: f32 = 0.35;

impl GateState {
    /// Evaluate gate from standard deviation of candidate scores.
    pub fn from_sd(sd: f32) -> Self {
        if sd < SD_FLOW_THRESHOLD { Self::Flow }
        else if sd > SD_BLOCK_THRESHOLD { Self::Block }
        else { Self::Hold }
    }

    /// Evaluate with style bias.
    pub fn from_sd_styled(sd: f32, style: &ThinkingStyle) -> Self {
        let biased_sd = sd + style.params().collapse_bias;
        Self::from_sd(biased_sd)
    }
}

/// Semantic depth levels (0-9). Rung elevation happens on:
/// - Sustained BLOCK state
/// - Predictive failure (free energy spike)
/// - Structural mismatch (no parse)
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
#[repr(u8)]
pub enum RungLevel {
    #[default]
    Surface = 0,        // literal, immediate
    Shallow = 1,        // simple inference
    Contextual = 2,     // situation-dependent
    Analogical = 3,     // metaphor, similarity
    Abstract = 4,       // generalized patterns
    Structural = 5,     // schema-level
    Counterfactual = 6, // what-if reasoning
    Meta = 7,           // reasoning about reasoning
    Recursive = 8,      // self-referential loops
    Transcendent = 9,   // beyond normal bounds
}

impl RungLevel {
    pub fn as_u8(&self) -> u8 { *self as u8 }

    pub fn from_u8(n: u8) -> Self {
        match n {
            0 => Self::Surface, 1 => Self::Shallow, 2 => Self::Contextual,
            3 => Self::Analogical, 4 => Self::Abstract, 5 => Self::Structural,
            6 => Self::Counterfactual, 7 => Self::Meta, 8 => Self::Recursive,
            _ => Self::Transcendent,
        }
    }

    /// Band for bucket addressing: Low(0-2), Mid(3-5), High(6-9).
    pub fn band(&self) -> &'static str {
        match self.as_u8() {
            0..=2 => "low",
            3..=5 => "mid",
            _ => "high",
        }
    }

    /// Should elevate rung based on cascade behavior?
    pub fn should_elevate(
        consecutive_blocks: usize,
        free_energy: f32,
        cascade_depth: usize,
    ) -> bool {
        consecutive_blocks >= 3 || free_energy > 0.15 || cascade_depth >= 4
    }
}

/// Meta-cognition: calibration of confidence estimates.
/// Brier score tracks how well-calibrated predictions are.
#[derive(Clone, Debug)]
pub struct MetaCognition {
    confidence_history: Vec<f32>,
    brier_sum: f32,
    prediction_count: u32,
    max_history: usize,
}

/// Meta-assessment of reasoning quality.
#[derive(Clone, Debug)]
pub struct MetaAssessment {
    pub confidence: f32,
    pub meta_confidence: f32,
    pub gate_state: GateState,
    pub should_admit_ignorance: bool,
    pub calibration_error: f32,
}

impl MetaCognition {
    pub fn new() -> Self {
        Self {
            confidence_history: Vec::new(),
            brier_sum: 0.0,
            prediction_count: 0,
            max_history: 100,
        }
    }

    /// Record a prediction outcome for calibration.
    pub fn record(&mut self, predicted_confidence: f32, was_correct: bool) {
        let outcome = if was_correct { 1.0 } else { 0.0 };
        let error = (predicted_confidence - outcome).powi(2);
        self.brier_sum += error;
        self.prediction_count += 1;
        self.confidence_history.push(predicted_confidence);
        if self.confidence_history.len() > self.max_history {
            self.confidence_history.remove(0);
        }
    }

    /// Current Brier score (calibration error). 0 = perfect, 1 = worst.
    pub fn brier_score(&self) -> f32 {
        if self.prediction_count == 0 { return 0.5; }
        self.brier_sum / self.prediction_count as f32
    }

    /// Is the system well-calibrated?
    pub fn is_well_calibrated(&self) -> bool {
        self.prediction_count > 10 && self.brier_score() < 0.15
    }

    /// Assess current meta-cognitive state.
    pub fn assess(&self, current_confidence: f32, gate: GateState) -> MetaAssessment {
        let variance = if self.confidence_history.len() > 1 {
            let mean: f32 = self.confidence_history.iter().sum::<f32>() / self.confidence_history.len() as f32;
            self.confidence_history.iter().map(|c| (c - mean).powi(2)).sum::<f32>() / self.confidence_history.len() as f32
        } else { 0.5 };

        let meta_confidence = (1.0 - variance).clamp(0.0, 1.0);
        let brier = self.brier_score();
        let should_admit = current_confidence < 0.3 && brier > 0.2;

        MetaAssessment {
            confidence: current_confidence,
            meta_confidence,
            gate_state: gate,
            should_admit_ignorance: should_admit,
            calibration_error: brier,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn all_12_styles() {
        assert_eq!(ThinkingStyle::all().len(), 12);
    }

    #[test]
    fn style_params_consistent() {
        for style in ThinkingStyle::all() {
            let p = style.params();
            assert!(p.resonance_threshold >= 0.0 && p.resonance_threshold <= 1.0);
            assert!(p.fan_out >= 1);
            assert!(p.exploration >= 0.0 && p.exploration <= 1.0);
        }
    }

    #[test]
    fn gate_thresholds() {
        assert_eq!(GateState::from_sd(0.10), GateState::Flow);
        assert_eq!(GateState::from_sd(0.25), GateState::Hold);
        assert_eq!(GateState::from_sd(0.40), GateState::Block);
    }

    #[test]
    fn gate_styled_creative_holds_longer() {
        // Creative has collapse_bias = 0.15 → SD appears higher → holds longer
        let gate = GateState::from_sd_styled(0.10, &ThinkingStyle::Creative);
        assert_eq!(gate, GateState::Hold); // 0.10 + 0.15 = 0.25 → Hold

        let gate = GateState::from_sd_styled(0.10, &ThinkingStyle::Focused);
        assert_eq!(gate, GateState::Flow); // 0.10 + (-0.15) = -0.05 → Flow
    }

    #[test]
    fn rung_elevation() {
        assert!(RungLevel::should_elevate(3, 0.0, 0));  // sustained block
        assert!(RungLevel::should_elevate(0, 0.2, 0));  // high free energy
        assert!(RungLevel::should_elevate(0, 0.0, 5));  // deep cascade
        assert!(!RungLevel::should_elevate(1, 0.05, 2)); // none triggered
    }

    #[test]
    fn metacog_brier() {
        let mut mc = MetaCognition::new();
        // Perfect calibration: predict 0.8, correct 80% of time
        for i in 0..20 {
            mc.record(0.8, i % 5 != 0); // correct 80%
        }
        assert!(mc.brier_score() < 0.2);
    }

    #[test]
    fn metacog_admit_ignorance() {
        let mut mc = MetaCognition::new();
        // Poor calibration
        for _ in 0..15 {
            mc.record(0.9, false); // overconfident, always wrong
        }
        let assessment = mc.assess(0.2, GateState::Block);
        assert!(assessment.should_admit_ignorance);
    }

    #[test]
    fn butterfly_ordering() {
        assert!(ThinkingStyle::Peripheral.butterfly_sensitivity() <
                ThinkingStyle::Focused.butterfly_sensitivity());
    }
}
