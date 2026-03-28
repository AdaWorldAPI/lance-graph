//! Temporal NARS — propagate truth values through time.
//!
//! Standard NARS inference is atemporal: if A→B now, then A→C via deduction.
//! Temporal NARS adds a time dimension: if A→B at t=0, then at t=1 the confidence
//! decays or strengthens based on evidence accumulation.
//!
//! ## Application: War Scenario Simulation
//!
//! Given a starting condition (e.g., "Iran's air defenses destroyed"),
//! temporal NARS simulates the causal cascade through time:
//!
//! ```text
//! t=0: F35 → PENETRATES → Iran_AD           (f=0.75, c=0.65)
//! t=1: Iran_AD → RETALIATES → Israeli_Base   (f=0.60, c=0.45, DECAYED from t=0 evidence)
//! t=2: Iron_Dome → INTERCEPTS → Retaliation  (f=0.92, c=0.85, INDEPENDENT evidence)
//! t=3: Escalation → TRIGGERS → Regional_War  (f=0.30, c=0.20, SPECULATIVE)
//! ```
//!
//! Each timestep applies:
//! 1. Truth decay (confidence drops with time distance)
//! 2. Evidence revision (new evidence strengthens/weakens)
//! 3. Causal propagation (deduction through active edges)

use super::{CausalStep, Derivation};

/// A snapshot of the graph at a point in simulated time.
#[derive(Debug, Clone)]
pub struct TimeStep {
    /// Simulation round number.
    pub round: usize,
    /// Active edges at this timestep (with truth values).
    pub active_edges: Vec<CausalStep>,
    /// Edges that were active but decayed below threshold.
    pub decayed_edges: Vec<CausalStep>,
    /// New edges discovered through inference at this timestep.
    pub inferred_edges: Vec<CausalStep>,
    /// Narrative description of what happened.
    pub narrative: String,
}

/// Configuration for temporal simulation.
#[derive(Debug, Clone)]
pub struct TemporalConfig {
    /// How fast confidence decays per round (0..1). 0.9 = 10% decay/round.
    pub decay_rate: f64,
    /// Minimum confidence to keep an edge alive.
    pub alive_threshold: f64,
    /// Number of rounds to simulate.
    pub max_rounds: usize,
    /// Whether to apply NARS deduction between rounds.
    pub enable_inference: bool,
}

impl Default for TemporalConfig {
    fn default() -> Self {
        Self {
            decay_rate: 0.85,
            alive_threshold: 0.10,
            max_rounds: 10,
            enable_inference: true,
        }
    }
}

/// Run a temporal simulation: propagate causal chains through time.
pub fn simulate(
    initial_edges: &[CausalStep],
    trigger: &str, // The triggering event description
    config: &TemporalConfig,
) -> Vec<TimeStep> {
    let mut timesteps = Vec::new();
    let mut active: Vec<CausalStep> = initial_edges.to_vec();

    for round in 0..config.max_rounds {
        let mut decayed = Vec::new();
        let mut next_active = Vec::new();

        // Apply decay
        for mut edge in active {
            edge.confidence *= config.decay_rate;
            if edge.confidence >= config.alive_threshold {
                next_active.push(edge);
            } else {
                decayed.push(edge);
            }
        }

        // Apply inference (deduction at each timestep)
        let mut inferred = Vec::new();
        if config.enable_inference && next_active.len() >= 2 {
            // Deduction: A→B and B→C at this timestep ⟹ A→C
            for i in 0..next_active.len() {
                for j in 0..next_active.len() {
                    if i == j { continue; }
                    if next_active[i].target == next_active[j].source {
                        let freq = next_active[i].frequency * next_active[j].frequency;
                        let conf = next_active[i].confidence * next_active[j].confidence
                            * next_active[i].frequency * next_active[j].frequency;
                        if conf >= config.alive_threshold {
                            // Check we don't already have this edge
                            let src = &next_active[i].source;
                            let tgt = &next_active[j].target;
                            let already_exists = next_active.iter()
                                .chain(inferred.iter())
                                .any(|e| e.source == *src && e.target == *tgt);
                            if !already_exists && src != tgt {
                                inferred.push(CausalStep {
                                    source: src.clone(),
                                    relationship: format!("TEMPORAL_DEDUCTION_t{round}"),
                                    target: tgt.clone(),
                                    frequency: freq,
                                    confidence: conf,
                                    derivation: Derivation::Deduction,
                                });
                            }
                        }
                    }
                }
            }
        }

        // Build narrative
        let narrative = if round == 0 {
            format!("t={round}: {trigger}. {} active edges, {} inferred.",
                next_active.len(), inferred.len())
        } else if !inferred.is_empty() {
            let new_connections: Vec<String> = inferred.iter()
                .map(|e| format!("{} → {}", e.source, e.target))
                .collect();
            format!("t={round}: {} edges active, {} decayed. New connections: {}",
                next_active.len(), decayed.len(), new_connections.join(", "))
        } else if !decayed.is_empty() {
            let lost: Vec<String> = decayed.iter()
                .map(|e| format!("{} → {}", e.source, e.target))
                .collect();
            format!("t={round}: {} edges active. Lost confidence: {}",
                next_active.len(), lost.join(", "))
        } else {
            format!("t={round}: {} edges active, stable.", next_active.len())
        };

        // Add inferred edges to active set
        next_active.extend(inferred.clone());

        timesteps.push(TimeStep {
            round,
            active_edges: next_active.clone(),
            decayed_edges: decayed,
            inferred_edges: inferred,
            narrative,
        });

        active = next_active;

        // Stop if nothing is alive
        if active.is_empty() {
            break;
        }
    }

    timesteps
}

/// Compute the "cascade depth" — how many rounds until a specific target is reached.
pub fn cascade_depth(
    timesteps: &[TimeStep],
    target: &str,
) -> Option<usize> {
    for ts in timesteps {
        if ts.active_edges.iter().any(|e| e.target == target)
            || ts.inferred_edges.iter().any(|e| e.target == target)
        {
            return Some(ts.round);
        }
    }
    None
}

/// Summary statistics for a simulation run.
#[derive(Debug, Clone)]
pub struct SimulationSummary {
    pub total_rounds: usize,
    pub peak_active_edges: usize,
    pub total_inferred: usize,
    pub total_decayed: usize,
    pub cascade_reach: Vec<String>, // All nodes ever reached
    pub final_confidence_range: (f64, f64), // (min, max) of surviving edges
}

pub fn summarize(timesteps: &[TimeStep]) -> SimulationSummary {
    let mut all_reached = std::collections::HashSet::new();
    let mut peak = 0;
    let mut total_inferred = 0;
    let mut total_decayed = 0;

    for ts in timesteps {
        peak = peak.max(ts.active_edges.len());
        total_inferred += ts.inferred_edges.len();
        total_decayed += ts.decayed_edges.len();
        for e in &ts.active_edges {
            all_reached.insert(e.source.clone());
            all_reached.insert(e.target.clone());
        }
    }

    let final_range = if let Some(last) = timesteps.last() {
        let confs: Vec<f64> = last.active_edges.iter().map(|e| e.confidence).collect();
        if confs.is_empty() {
            (0.0, 0.0)
        } else {
            (*confs.iter().min_by(|a, b| a.partial_cmp(b).unwrap()).unwrap(),
             *confs.iter().max_by(|a, b| a.partial_cmp(b).unwrap()).unwrap())
        }
    } else {
        (0.0, 0.0)
    };

    SimulationSummary {
        total_rounds: timesteps.len(),
        peak_active_edges: peak,
        total_inferred,
        total_decayed,
        cascade_reach: all_reached.into_iter().collect(),
        final_confidence_range: final_range,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn iran_scenario_edges() -> Vec<CausalStep> {
        vec![
            CausalStep {
                source: "F35".into(), relationship: "PENETRATES".into(),
                target: "Iran_AD".into(), frequency: 0.75, confidence: 0.65,
                derivation: Derivation::Observed,
            },
            CausalStep {
                source: "Iran_AD".into(), relationship: "RETALIATES".into(),
                target: "Israeli_Base".into(), frequency: 0.60, confidence: 0.50,
                derivation: Derivation::Observed,
            },
            CausalStep {
                source: "Iron_Dome".into(), relationship: "INTERCEPTS".into(),
                target: "Retaliation".into(), frequency: 0.92, confidence: 0.85,
                derivation: Derivation::Observed,
            },
            CausalStep {
                source: "Israeli_Base".into(), relationship: "ESCALATES".into(),
                target: "Regional_War".into(), frequency: 0.30, confidence: 0.25,
                derivation: Derivation::Hypothetical,
            },
        ]
    }

    #[test]
    fn test_temporal_simulation_runs() {
        let edges = iran_scenario_edges();
        let config = TemporalConfig {
            max_rounds: 5,
            ..Default::default()
        };
        let result = simulate(&edges, "Iran air defense strike", &config);
        assert_eq!(result.len(), 5);
        // First round should have all edges
        assert!(result[0].active_edges.len() >= 4);
    }

    #[test]
    fn test_confidence_decays_over_time() {
        let edges = iran_scenario_edges();
        let config = TemporalConfig {
            decay_rate: 0.5, // Aggressive decay
            max_rounds: 5,
            enable_inference: false,
            ..Default::default()
        };
        let result = simulate(&edges, "test", &config);
        // Hypothetical edge (f=0.30, c=0.25) should decay and die quickly
        let hypothetical_alive_at_end = result.last()
            .map(|ts| ts.active_edges.iter()
                .any(|e| e.target == "Regional_War"))
            .unwrap_or(false);
        assert!(!hypothetical_alive_at_end, "Low-confidence edge should decay");
    }

    #[test]
    fn test_inference_discovers_new_edges() {
        let edges = iran_scenario_edges();
        let config = TemporalConfig {
            max_rounds: 3,
            enable_inference: true,
            ..Default::default()
        };
        let result = simulate(&edges, "test", &config);
        // F35 → Iran_AD and Iran_AD → Israeli_Base should produce F35 → Israeli_Base
        let has_deduction = result.iter().any(|ts|
            ts.inferred_edges.iter().any(|e|
                e.source == "F35" && e.target == "Israeli_Base"
            )
        );
        assert!(has_deduction, "Should infer F35 → Israeli_Base via deduction");
    }

    #[test]
    fn test_cascade_depth() {
        let edges = iran_scenario_edges();
        let config = TemporalConfig::default();
        let result = simulate(&edges, "test", &config);
        // Regional_War is directly connected at t=0
        let depth = cascade_depth(&result, "Regional_War");
        assert_eq!(depth, Some(0));
    }

    #[test]
    fn test_simulation_summary() {
        let edges = iran_scenario_edges();
        let config = TemporalConfig { max_rounds: 5, ..Default::default() };
        let result = simulate(&edges, "test", &config);
        let summary = summarize(&result);
        assert!(summary.peak_active_edges >= 4);
        assert!(summary.cascade_reach.len() >= 4);
    }

    #[test]
    fn test_empty_edges_terminates() {
        let result = simulate(&[], "nothing", &TemporalConfig::default());
        assert!(result.is_empty() || result[0].active_edges.is_empty());
    }
}
