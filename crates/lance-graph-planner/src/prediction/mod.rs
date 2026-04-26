//! Prediction Engine — temporal NARS simulation + news ingestion + scenario analysis.
//!
//! This module makes the planner predictive: given a graph state and a question
//! ("what happens if Iran's air defenses are destroyed?"), it:
//!
//! 1. Extracts relevant subgraph (the kill chain neighborhood)
//! 2. Runs NARS temporal inference (propagate truth values through time)
//! 3. Applies thinking-style-dependent exploration (Analytical = deep chain,
//!    Creative = lateral connections, Focused = single kill chain)
//! 4. Returns ranked scenarios with confidence scores
//!
//! ## Architecture
//!
//! ```text
//! News/Intel ──→ Ingestion ──→ Graph Expansion ──→ Prediction
//!                  │                │                    │
//!          entity extraction   new nodes/edges    NARS temporal
//!          (LLM or parser)    with truth values   propagation
//! ```

pub mod scenario;
pub mod ingestion;
pub mod temporal;

#[allow(unused_imports)] // ThinkingCluster intended for cluster-based prediction wiring
use crate::thinking::style::{ThinkingStyle, ThinkingCluster};
#[allow(unused_imports)] // PatienceBudget intended for budget-aware prediction wiring
use crate::elevation::budget::{PatienceBudget, budget_for_cluster};

/// A prediction scenario — one possible future outcome.
#[derive(Debug, Clone)]
pub struct Scenario {
    /// Human-readable name for this scenario.
    pub name: String,
    /// Description of what happens.
    pub description: String,
    /// Causal chain: sequence of (source, relationship, target) steps.
    pub chain: Vec<CausalStep>,
    /// Overall confidence (product of step confidences).
    pub confidence: f64,
    /// Which thinking style generated this scenario.
    pub style: ThinkingStyle,
    /// Estimated time horizon (in abstract "rounds").
    pub time_horizon: usize,
    /// Blind spots — what this scenario doesn't account for.
    pub blind_spots: Vec<String>,
}

/// One step in a causal chain.
#[derive(Debug, Clone)]
pub struct CausalStep {
    pub source: String,
    pub relationship: String,
    pub target: String,
    /// NARS truth value for this step.
    pub frequency: f64,
    pub confidence: f64,
    /// How this step was derived.
    pub derivation: Derivation,
}

/// How a causal step was derived.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Derivation {
    /// Directly observed in the graph.
    Observed,
    /// Inferred via NARS deduction (A→B, B→C ⟹ A→C).
    Deduction,
    /// Inferred via NARS abduction (A→B, C→B ⟹ A→C).
    Abduction,
    /// Inferred via NARS induction (A→B, A→C ⟹ B→C).
    Induction,
    /// Injected from external intelligence (news, LLM).
    Ingested,
    /// Hypothetical — scenario-specific assumption.
    Hypothetical,
}

/// A node that can be ingested into the graph from external sources.
#[derive(Debug, Clone)]
pub struct IngestNode {
    pub id: String,
    pub label: String,
    pub node_type: String,
    pub properties: Vec<(String, String)>,
    /// Source of this node (URL, document ID, etc.)
    pub source: String,
    /// Trust in the source (0..1).
    pub source_confidence: f64,
}

/// An edge that can be ingested into the graph from external sources.
#[derive(Debug, Clone)]
pub struct IngestEdge {
    pub source_id: String,
    pub target_id: String,
    pub relationship: String,
    pub frequency: f64,
    pub confidence: f64,
    pub source: String,
    pub derivation: Derivation,
}

/// Result of a prediction run.
#[derive(Debug, Clone)]
pub struct PredictionResult {
    /// Question that was asked.
    pub question: String,
    /// Generated scenarios, ranked by confidence.
    pub scenarios: Vec<Scenario>,
    /// Nodes that were ingested during prediction (if any).
    pub ingested_nodes: usize,
    /// Edges that were inferred during prediction.
    pub inferred_edges: usize,
    /// Total execution time in microseconds.
    pub elapsed_us: u64,
    /// Thinking styles that were used.
    pub styles_used: Vec<ThinkingStyle>,
}

/// Multi-style prediction: run the same question through multiple thinking styles
/// and compare/contrast the scenarios they produce.
pub fn predict_multi_style(
    question: &str,
    seed_edges: &[(String, String, String, f64, f64)], // (src, rel, tgt, freq, conf)
    styles: &[ThinkingStyle],
) -> PredictionResult {
    let t0 = std::time::Instant::now();
    let mut all_scenarios = Vec::new();

    for &style in styles {
        let budget = budget_for_cluster(style.cluster());
        let scenarios = scenario::generate_scenarios(
            question,
            seed_edges,
            style,
            &budget,
        );
        all_scenarios.extend(scenarios);
    }

    // Sort by confidence descending
    all_scenarios.sort_by(|a, b| b.confidence.partial_cmp(&a.confidence).unwrap_or(std::cmp::Ordering::Equal));

    // Count inferred edges
    let inferred_edges = all_scenarios.iter()
        .flat_map(|s| &s.chain)
        .filter(|step| step.derivation != Derivation::Observed)
        .count();

    PredictionResult {
        question: question.to_string(),
        scenarios: all_scenarios,
        ingested_nodes: 0,
        inferred_edges,
        elapsed_us: t0.elapsed().as_micros() as u64,
        styles_used: styles.to_vec(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_multi_style_prediction() {
        // Edges form:
        // Chain: Lavender → Gospel → Fire_Factory → Target (for Analytical)
        // Shared target: Iron_Dome → Target AND Fire_Factory → Target (for Creative/abduction)
        // Shared source: Iron_Dome → Target AND Iron_Dome → Military_Base (for Creative/induction)
        let edges = vec![
            ("Lavender".into(), "TARGETS".into(), "Gospel".into(), 0.95, 0.87),
            ("Gospel".into(), "CONFIRMS".into(), "Fire_Factory".into(), 0.90, 0.82),
            ("Fire_Factory".into(), "STRIKES".into(), "Target".into(), 0.88, 0.79),
            ("Iron_Dome".into(), "INTERCEPTS".into(), "Target".into(), 0.92, 0.85),
            ("Iron_Dome".into(), "DEFENDS".into(), "Military_Base".into(), 0.85, 0.78),
        ];

        let styles = vec![
            ThinkingStyle::Analytical,
            ThinkingStyle::Creative,
            ThinkingStyle::Focused,
        ];

        let result = predict_multi_style(
            "What happens if Iran strikes Israeli air bases?",
            &edges,
            &styles,
        );

        assert!(!result.scenarios.is_empty());
        assert_eq!(result.styles_used.len(), 3);
        // Each style should produce at least one scenario
        for style in &styles {
            assert!(result.scenarios.iter().any(|s| s.style == *style),
                "No scenario from {:?}", style);
        }
        // Scenarios should be sorted by confidence
        for w in result.scenarios.windows(2) {
            assert!(w[0].confidence >= w[1].confidence);
        }
    }
}
