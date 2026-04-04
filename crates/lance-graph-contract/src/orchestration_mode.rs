//! OrchestrationMode — meta-conductor that sequences thinking styles via NARS RL.
//!
//! Not a thinking style itself — it's the conductor that decides which style
//! to play next, measures the result with Jina BF16 cross-validation, and
//! learns which combinations work via NARS truth value reinforcement.
//!
//! ```text
//! Observation → fan-out inference actions:
//!   Association    → "what relates to this?"
//!   Intuition      → "what feels right?"
//!   Abduction      → "what would explain this?"
//!   Deduction       → "what follows from this?"
//!   Induction       → "what pattern emerges?"
//!   Hypothesis      → "what if we assume X?"
//!   Synthesis       → "how do these combine?"
//!   Extrapolation   → "where does this trend lead?"
//!   Counterfactual  → "what if X hadn't happened?"
//!
//! Each action → selects a ThinkingStyle → executes → measures relevance
//! Relevance = Jina BF16 cross-model cosine (truth anchor)
//! NARS truth value RL: frequency = success rate, confidence = sample count
//! Next step: pick action with highest NARS expectation
//! ```
//!
//! Zero dependencies.

use crate::thinking::{ThinkingStyle, FieldModulation, StyleCluster};

/// Inference actions the orchestrator can take.
/// Each maps to a NARS inference type + a preferred thinking style.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum InferenceAction {
    /// "What relates to this?" — fan-out association search.
    Association = 0,
    /// "What feels right?" — fast System 1 pattern match.
    Intuition = 1,
    /// "What would explain this?" — abductive reasoning.
    Abduction = 2,
    /// "What follows from this?" — deductive chain.
    Deduction = 3,
    /// "What pattern emerges?" — inductive generalization.
    Induction = 4,
    /// "What if we assume X?" — hypothesis generation + test.
    Hypothesis = 5,
    /// "How do these combine?" — cross-domain synthesis.
    Synthesis = 6,
    /// "Where does this trend lead?" — forward projection.
    Extrapolation = 7,
    /// "What if X hadn't happened?" — Pearl Level 3.
    Counterfactual = 8,
}

impl InferenceAction {
    pub const ALL: [InferenceAction; 9] = [
        Self::Association, Self::Intuition, Self::Abduction,
        Self::Deduction, Self::Induction, Self::Hypothesis,
        Self::Synthesis, Self::Extrapolation, Self::Counterfactual,
    ];

    /// Preferred thinking style for this action.
    pub fn preferred_style(&self) -> ThinkingStyle {
        match self {
            Self::Association    => ThinkingStyle::Curious,
            Self::Intuition      => ThinkingStyle::Warm,        // fast, empathic
            Self::Abduction      => ThinkingStyle::Investigative,
            Self::Deduction      => ThinkingStyle::Logical,
            Self::Induction      => ThinkingStyle::Analytical,
            Self::Hypothesis     => ThinkingStyle::Speculative,
            Self::Synthesis      => ThinkingStyle::Creative,
            Self::Extrapolation  => ThinkingStyle::Philosophical,
            Self::Counterfactual => ThinkingStyle::Metacognitive,
        }
    }

    /// Fan-out: how many parallel paths this action spawns.
    pub fn default_fan_out(&self) -> usize {
        match self {
            Self::Association    => 8,  // wide net
            Self::Intuition      => 2,  // narrow, fast
            Self::Abduction      => 4,  // moderate
            Self::Deduction      => 2,  // precise chain
            Self::Induction      => 6,  // needs examples
            Self::Hypothesis     => 3,  // generate + test
            Self::Synthesis      => 4,  // cross-domain
            Self::Extrapolation  => 3,  // forward projection
            Self::Counterfactual => 2,  // single what-if
        }
    }

    /// Pearl causal level (0=none, 1=SEE, 2=DO, 3=IMAGINE).
    pub fn pearl_level(&self) -> u8 {
        match self {
            Self::Association | Self::Intuition | Self::Induction => 1, // SEE
            Self::Deduction | Self::Abduction | Self::Hypothesis => 2, // DO
            Self::Counterfactual | Self::Extrapolation | Self::Synthesis => 3, // IMAGINE
        }
    }
}

/// NARS truth value for RL (how well an action performed).
#[derive(Debug, Clone, Copy)]
pub struct ActionTruth {
    /// Frequency: success rate [0, 1].
    pub frequency: f32,
    /// Confidence: weight of evidence [0, 1).
    pub confidence: f32,
}

impl ActionTruth {
    pub fn prior() -> Self { Self { frequency: 0.5, confidence: 0.1 } }

    /// NARS expectation: E = c(f - 0.5) + 0.5
    pub fn expectation(&self) -> f32 {
        self.confidence * (self.frequency - 0.5) + 0.5
    }

    /// Revision: merge with new observation.
    pub fn revise(&self, outcome_positive: bool, jina_relevance: f32) -> Self {
        let new_f = if outcome_positive { jina_relevance } else { 1.0 - jina_relevance };
        let w1 = self.confidence / (1.0 - self.confidence + 1e-9);
        let w2 = jina_relevance.max(0.1); // relevance-weighted evidence
        let total = w1 + w2;
        let merged_f = (self.frequency * w1 + new_f * w2) / total;
        let merged_c = (total / (total + 1.0)).min(0.99);
        ActionTruth { frequency: merged_f, confidence: merged_c }
    }
}

/// Relevance measurement from Jina BF16 cross-model validation.
#[derive(Debug, Clone, Copy)]
pub struct RelevanceMeasurement {
    /// Jina cross-model cosine similarity [0, 1].
    pub jina_cosine: f32,
    /// BGE-M3 agreement (if available) [0, 1].
    pub bge_m3_cosine: f32,
    /// Combined relevance score (geometric mean of available lenses).
    pub combined: f32,
}

impl RelevanceMeasurement {
    pub fn from_jina(cosine: f32) -> Self {
        Self { jina_cosine: cosine, bge_m3_cosine: 0.0, combined: cosine }
    }

    pub fn from_multi_lens(jina: f32, bge: f32) -> Self {
        let combined = (jina * bge).sqrt(); // geometric mean
        Self { jina_cosine: jina, bge_m3_cosine: bge, combined }
    }
}

/// One step in the orchestration sequence.
#[derive(Debug, Clone)]
pub struct OrchestrationStep {
    /// Which action was taken.
    pub action: InferenceAction,
    /// Which thinking style was used.
    pub style: ThinkingStyle,
    /// Relevance of the result (measured by Jina BF16).
    pub relevance: Option<RelevanceMeasurement>,
    /// How many new edges were discovered.
    pub edges_discovered: usize,
    /// How many existing edges were confirmed.
    pub edges_confirmed: usize,
}

/// The orchestration mode — wraps and sequences thinking styles.
///
/// Maintains NARS truth values per (action, context) pair.
/// Picks the highest-expectation action at each step.
/// Measures results with Jina BF16 cross-validation.
/// Learns which combinations work via truth value RL.
pub struct OrchestrationMode {
    /// NARS truth per inference action (RL state).
    pub action_truths: [ActionTruth; 9],
    /// History of steps taken.
    pub history: Vec<OrchestrationStep>,
    /// Current fan-out multiplier (adjusted by RL).
    pub fan_out_multiplier: f32,
    /// Temperature: exploration vs exploitation (decays over time).
    pub temperature: f32,
    /// Steps executed.
    pub step_count: u64,
}

impl OrchestrationMode {
    pub fn new() -> Self {
        Self {
            action_truths: [ActionTruth::prior(); 9],
            history: Vec::new(),
            fan_out_multiplier: 1.0,
            temperature: 1.0, // start exploratory
            step_count: 0,
        }
    }

    /// Select the next inference action (epsilon-greedy with NARS expectation).
    pub fn select_action(&self) -> InferenceAction {
        // With probability `temperature`, explore uniformly
        // Otherwise, exploit highest expectation
        let explore = {
            // Deterministic pseudo-random based on step count
            let hash = self.step_count.wrapping_mul(0x9E3779B97F4A7C15);
            (hash % 100) as f32 / 100.0 < self.temperature * 0.2
        };

        if explore {
            // Pick least-explored action (lowest confidence)
            let mut min_conf = f32::MAX;
            let mut min_idx = 0;
            for (i, t) in self.action_truths.iter().enumerate() {
                if t.confidence < min_conf {
                    min_conf = t.confidence;
                    min_idx = i;
                }
            }
            InferenceAction::ALL[min_idx]
        } else {
            // Pick highest expectation
            let mut best_exp = f32::MIN;
            let mut best_idx = 0;
            for (i, t) in self.action_truths.iter().enumerate() {
                let exp = t.expectation();
                if exp > best_exp {
                    best_exp = exp;
                    best_idx = i;
                }
            }
            InferenceAction::ALL[best_idx]
        }
    }

    /// Get the thinking style + field modulation for the selected action.
    pub fn style_for_action(&self, action: InferenceAction) -> (ThinkingStyle, FieldModulation) {
        let style = action.preferred_style();
        let mut modulation = FieldModulation::default();
        modulation.fan_out = (action.default_fan_out() as f32 * self.fan_out_multiplier) as usize;
        modulation.exploration = self.temperature as f64;
        // Adjust depth/breadth based on action type
        match action {
            InferenceAction::Association | InferenceAction::Induction => {
                modulation.breadth_bias = 0.8;
                modulation.depth_bias = 0.2;
            }
            InferenceAction::Deduction | InferenceAction::Extrapolation => {
                modulation.depth_bias = 0.9;
                modulation.breadth_bias = 0.1;
            }
            InferenceAction::Counterfactual | InferenceAction::Hypothesis => {
                modulation.noise_tolerance = 0.7; // allow speculative results
            }
            _ => {}
        }
        (style, modulation)
    }

    /// Record the outcome of an action (RL update).
    ///
    /// `relevance` comes from Jina BF16 cross-model cosine.
    /// `positive` = did the action produce useful new knowledge?
    pub fn record_outcome(
        &mut self,
        action: InferenceAction,
        positive: bool,
        relevance: RelevanceMeasurement,
        edges_discovered: usize,
        edges_confirmed: usize,
    ) {
        let idx = action as usize;
        self.action_truths[idx] = self.action_truths[idx].revise(positive, relevance.combined);

        self.history.push(OrchestrationStep {
            action,
            style: action.preferred_style(),
            relevance: Some(relevance),
            edges_discovered,
            edges_confirmed,
        });

        self.step_count += 1;

        // Decay temperature (explore less over time)
        self.temperature = (self.temperature * 0.995).max(0.05);

        // Adjust fan-out based on recent success
        let recent_success = self.history.iter().rev().take(10)
            .filter(|s| s.edges_discovered > 0 || s.edges_confirmed > 0)
            .count();
        self.fan_out_multiplier = if recent_success > 7 {
            1.5 // expand: things are working
        } else if recent_success < 3 {
            0.5 // contract: wasting budget
        } else {
            1.0
        };
    }

    /// Get current action rankings (for monitoring/debugging).
    pub fn action_rankings(&self) -> Vec<(InferenceAction, f32, f32)> {
        let mut rankings: Vec<_> = InferenceAction::ALL.iter()
            .map(|&a| {
                let t = &self.action_truths[a as usize];
                (a, t.expectation(), t.confidence)
            })
            .collect();
        rankings.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        rankings
    }

    /// Summary stats for monitoring.
    pub fn stats(&self) -> OrchestrationStats {
        let total_discovered: usize = self.history.iter().map(|s| s.edges_discovered).sum();
        let total_confirmed: usize = self.history.iter().map(|s| s.edges_confirmed).sum();
        let avg_relevance = if self.history.is_empty() { 0.0 } else {
            self.history.iter()
                .filter_map(|s| s.relevance.map(|r| r.combined))
                .sum::<f32>() / self.history.len() as f32
        };

        OrchestrationStats {
            steps: self.step_count,
            temperature: self.temperature,
            fan_out_multiplier: self.fan_out_multiplier,
            total_discovered,
            total_confirmed,
            avg_relevance,
        }
    }
}

impl Default for OrchestrationMode {
    fn default() -> Self { Self::new() }
}

/// Monitoring stats.
#[derive(Debug, Clone)]
pub struct OrchestrationStats {
    pub steps: u64,
    pub temperature: f32,
    pub fan_out_multiplier: f32,
    pub total_discovered: usize,
    pub total_confirmed: usize,
    pub avg_relevance: f32,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_action_preferred_styles() {
        // Each action maps to a distinct style
        let styles: Vec<_> = InferenceAction::ALL.iter()
            .map(|a| a.preferred_style())
            .collect();
        assert_eq!(styles.len(), 9);
        // Deduction → Logical
        assert_eq!(InferenceAction::Deduction.preferred_style(), ThinkingStyle::Logical);
        // Counterfactual → Metacognitive
        assert_eq!(InferenceAction::Counterfactual.preferred_style(), ThinkingStyle::Metacognitive);
    }

    #[test]
    fn test_pearl_levels() {
        assert_eq!(InferenceAction::Association.pearl_level(), 1); // SEE
        assert_eq!(InferenceAction::Deduction.pearl_level(), 2);   // DO
        assert_eq!(InferenceAction::Counterfactual.pearl_level(), 3); // IMAGINE
    }

    #[test]
    fn test_nars_rl_learning() {
        let mut orch = OrchestrationMode::new();

        // All actions start at prior (expectation 0.5)
        for t in &orch.action_truths {
            assert!((t.expectation() - 0.5).abs() < 0.01);
        }

        // Reward Deduction repeatedly
        for _ in 0..10 {
            orch.record_outcome(
                InferenceAction::Deduction, true,
                RelevanceMeasurement::from_jina(0.9),
                3, 1,
            );
        }

        // Deduction should now have highest expectation
        let rankings = orch.action_rankings();
        assert_eq!(rankings[0].0, InferenceAction::Deduction);
        assert!(rankings[0].1 > 0.7, "deduction expectation should be high: {}", rankings[0].1);
    }

    #[test]
    fn test_exploration_decays() {
        let mut orch = OrchestrationMode::new();
        let t0 = orch.temperature;

        for _ in 0..100 {
            orch.record_outcome(
                InferenceAction::Association, true,
                RelevanceMeasurement::from_jina(0.5),
                1, 0,
            );
        }

        assert!(orch.temperature < t0, "temperature should decay");
        assert!(orch.temperature > 0.04, "temperature should have floor");
    }

    #[test]
    fn test_fan_out_adapts() {
        let mut orch = OrchestrationMode::new();

        // Many successes → fan-out increases
        for _ in 0..10 {
            orch.record_outcome(
                InferenceAction::Synthesis, true,
                RelevanceMeasurement::from_jina(0.8),
                5, 2,
            );
        }
        assert!(orch.fan_out_multiplier > 1.0, "should expand on success");

        // Many failures → fan-out decreases
        let mut orch2 = OrchestrationMode::new();
        for _ in 0..10 {
            orch2.record_outcome(
                InferenceAction::Synthesis, false,
                RelevanceMeasurement::from_jina(0.1),
                0, 0,
            );
        }
        assert!(orch2.fan_out_multiplier < 1.0, "should contract on failure");
    }

    #[test]
    fn test_multi_lens_relevance() {
        let r = RelevanceMeasurement::from_multi_lens(0.9, 0.8);
        assert!((r.combined - 0.849).abs() < 0.01); // sqrt(0.72) ≈ 0.849
    }

    #[test]
    fn test_select_explores_then_exploits() {
        let mut orch = OrchestrationMode::new();

        // First few selections should vary (exploration)
        let mut actions_seen = std::collections::HashSet::new();
        for _ in 0..20 {
            let action = orch.select_action();
            actions_seen.insert(action);
            orch.record_outcome(action, true, RelevanceMeasurement::from_jina(0.5), 1, 0);
        }
        // Should have explored multiple actions
        assert!(actions_seen.len() > 1, "should explore multiple actions");

        // Reward one action heavily
        for _ in 0..50 {
            orch.record_outcome(
                InferenceAction::Abduction, true,
                RelevanceMeasurement::from_jina(0.95),
                10, 5,
            );
        }

        // Now should mostly exploit Abduction
        let mut abduction_count = 0;
        for _ in 0..10 {
            if orch.select_action() == InferenceAction::Abduction {
                abduction_count += 1;
            }
            orch.step_count += 1; // advance for deterministic hash
        }
        assert!(abduction_count >= 5, "should exploit abduction after reward: {}/10", abduction_count);
    }
}
