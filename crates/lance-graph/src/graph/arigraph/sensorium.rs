// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Graph Sensorium — real-time signals from the knowledge graph for MUL self-regulation.
//!
//! The graph's own state is the primary sensory input to the meta-awareness layer.
//! These signals drive automatic style balancing: high contradiction rate triggers
//! more Explore/Reflex; low entropy triggers more Plan/Act.
//!
//! Lives in lance-graph (where the graph is) not q2 (where the orchestrator is).
//! The orchestrator in q2 imports and consumes these signals.

use serde::{Deserialize, Serialize};

use super::episodic::EpisodicMemory;
use super::triplet_graph::TripletGraph;

/// Real-time signals from the knowledge graph for MUL self-regulation.
///
/// Each field is a normalized [0, 1] signal that the orchestrator's MUL layer
/// consumes to determine Dunning-Kruger position, trust texture, and flow state.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphSensorium {
    /// Contradiction rate: contradictions / active_triplets. Range [0, 1].
    /// High = lots of conflicting evidence → Valley of Despair, increase exploration.
    /// Low = consistent knowledge → Slope or Plateau, increase exploitation.
    pub contradiction_rate: f32,

    /// Truth entropy: Shannon entropy of truth confidence distribution.
    /// Normalized to [0, 1] where 0 = all triplets have same confidence,
    /// 1 = uniform distribution across confidence bands.
    /// High = uncertain about everything → increase exploration.
    /// Low = confident knowledge → increase exploitation.
    pub truth_entropy: f32,

    /// Revision velocity: revisions_per_step over rolling window.
    /// Range [0, 1] where 1 = every step produces a revision.
    /// High = actively learning → keep current mode, the system is adapting.
    /// Low = stagnant → either mastery (if consistent) or stuck (if inconsistent).
    pub revision_velocity: f32,

    /// Plasticity flux: fraction of entities whose truth changed recently. Range [0, 1].
    /// High = environment is changing rapidly → lower trust, increase exploration.
    /// Low = stable environment → higher trust, increase exploitation.
    pub plasticity_flux: f32,

    /// Deduction yield: inferred_triplets / deduction_attempts. Range [0, 1].
    /// High = graph structure supports rich inference → Plan/Act more.
    /// Low = sparse graph, few chains → Explore more.
    pub deduction_yield: f32,

    /// Episodic saturation: episodes / capacity. Range [0, 1].
    /// High = memory full → start forgetting or compressing.
    pub episodic_saturation: f32,

    /// Active triplet count (not normalized, raw count for context).
    pub active_triplets: usize,

    /// Total entity count.
    pub total_entities: usize,

    /// Contradiction count (raw).
    pub contradictions: usize,
}

impl GraphSensorium {
    /// Compute sensorium directly from a TripletGraph + EpisodicMemory.
    ///
    /// This is the primary constructor — reads all signals from the live graph.
    /// Call this before every orchestrator step for self-regulated thinking.
    pub fn from_graph(
        graph: &TripletGraph,
        memory: &EpisodicMemory,
        revisions_in_window: usize,
        window_steps: usize,
        deduction_attempts: usize,
        deductions_produced: usize,
    ) -> Self {
        let active: Vec<_> = graph.triplets.iter().filter(|t| !t.is_deleted()).collect();
        let active_count = active.len();
        let total_entities = graph.entities().len();

        // Contradiction count from the graph's own detection.
        let contradictions = graph.detect_contradictions(0.3).len();

        // Truth confidence histogram: [certain, strong, moderate, weak, unknown]
        let mut histogram = [0usize; 5];
        for t in &graph.triplets {
            let c = t.truth.confidence;
            if c >= 0.9 {
                histogram[0] += 1; // certain
            } else if c >= 0.7 {
                histogram[1] += 1; // strong
            } else if c >= 0.4 {
                histogram[2] += 1; // moderate
            } else if c >= 0.1 {
                histogram[3] += 1; // weak
            } else {
                histogram[4] += 1; // unknown/deleted
            }
        }

        // Plasticity: how many entities have been revised recently?
        // Proxy: count entities with at least one triplet that has moderate confidence
        // (not at the extremes — those are settled or deleted).
        let plastic_entities = graph
            .entities()
            .iter()
            .filter(|e| {
                if let Some(indices) = graph.entity_index.get(e.as_str()) {
                    indices.iter().any(|&idx| {
                        let c = graph.triplets[idx].truth.confidence;
                        c > 0.2 && c < 0.8
                    })
                } else {
                    false
                }
            })
            .count();

        Self::compute(
            active_count,
            contradictions,
            &histogram,
            revisions_in_window,
            window_steps,
            plastic_entities,
            total_entities,
            deduction_attempts,
            deductions_produced,
            memory.len(),
            memory.capacity(),
        )
    }

    /// Compute from raw statistics (when you don't have direct graph access).
    pub fn compute(
        active_triplets: usize,
        contradictions: usize,
        confidence_histogram: &[usize; 5],
        revisions_in_window: usize,
        window_steps: usize,
        hot_entities: usize,
        total_entities: usize,
        deduction_attempts: usize,
        deductions_produced: usize,
        episodic_count: usize,
        episodic_capacity: usize,
    ) -> Self {
        let active = active_triplets.max(1) as f32;

        let contradiction_rate = (contradictions as f32 / active).clamp(0.0, 1.0);

        // Shannon entropy of confidence distribution
        let total: f32 = confidence_histogram.iter().sum::<usize>() as f32;
        let truth_entropy = if total > 0.0 {
            let mut h = 0.0f32;
            for &count in confidence_histogram {
                if count > 0 {
                    let p = count as f32 / total;
                    h -= p * p.ln();
                }
            }
            // Normalize by max entropy (ln(5) ≈ 1.609)
            (h / 1.609).clamp(0.0, 1.0)
        } else {
            0.0
        };

        let revision_velocity = if window_steps > 0 {
            (revisions_in_window as f32 / window_steps as f32).clamp(0.0, 1.0)
        } else {
            0.0
        };

        let plasticity_flux = if total_entities > 0 {
            (hot_entities as f32 / total_entities as f32).clamp(0.0, 1.0)
        } else {
            0.0
        };

        let deduction_yield = if deduction_attempts > 0 {
            (deductions_produced as f32 / deduction_attempts as f32).clamp(0.0, 1.0)
        } else {
            0.0
        };

        let episodic_saturation = if episodic_capacity > 0 {
            (episodic_count as f32 / episodic_capacity as f32).clamp(0.0, 1.0)
        } else {
            0.0
        };

        Self {
            contradiction_rate,
            truth_entropy,
            revision_velocity,
            plasticity_flux,
            deduction_yield,
            episodic_saturation,
            active_triplets,
            total_entities,
            contradictions,
        }
    }

    /// What the graph signals suggest: explore more, exploit more, or panic.
    pub fn suggested_bias(&self) -> GraphBias {
        if self.contradiction_rate > 0.3 {
            GraphBias::Resolve
        } else if self.truth_entropy > 0.7 {
            GraphBias::Explore
        } else if self.deduction_yield > 0.5 && self.truth_entropy < 0.3 {
            GraphBias::Exploit
        } else if self.plasticity_flux > 0.5 {
            GraphBias::Adapt
        } else if self.revision_velocity < 0.05 && self.truth_entropy > 0.4 {
            GraphBias::Stagnant
        } else {
            GraphBias::Balanced
        }
    }
}

/// Graph-suggested cognitive bias.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum GraphBias {
    /// High contradictions — focus on resolution (Reflex/Metacognitive).
    Resolve,
    /// High entropy — gather evidence (Explore/Divergent).
    Explore,
    /// Rich consistent graph — use the knowledge (Plan/Act/Analytical).
    Exploit,
    /// High plasticity — stay flexible (Creative/Exploratory).
    Adapt,
    /// Low revision + high entropy — stuck, need perturbation.
    Stagnant,
    /// Normal — let topology decide.
    Balanced,
}

// ============================================================================
// NARS Auto-Heal Contingency
// ============================================================================

/// Actions the auto-heal contingency can take to fix an unorganized graph.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealingAction {
    pub action: HealingType,
    pub reason: String,
    pub triplets_affected: usize,
}

/// Types of healing the graph can self-apply.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum HealingType {
    /// Run NARS revision on all triplets with very low confidence.
    BootstrapTruth,
    /// Run contradiction detection + resolution.
    ResolveContradictions,
    /// Run deduction to fill in missing links.
    InferMissingLinks,
    /// Compact soft-deleted triplets (garbage collection).
    CompactDeleted,
    /// Re-normalize truth values: scale all confidences so max = 0.95.
    NormalizeTruth,
}

/// Determine what healing actions the graph needs.
pub fn diagnose_healing(signals: &GraphSensorium) -> Vec<HealingAction> {
    let mut actions = Vec::new();

    if signals.contradiction_rate > 0.15 {
        actions.push(HealingAction {
            action: HealingType::ResolveContradictions,
            reason: format!(
                "Contradiction rate {:.1}% exceeds 15% threshold",
                signals.contradiction_rate * 100.0
            ),
            triplets_affected: signals.contradictions,
        });
    }

    if signals.truth_entropy > 0.6 && signals.revision_velocity < 0.1 {
        actions.push(HealingAction {
            action: HealingType::BootstrapTruth,
            reason: format!(
                "High entropy ({:.2}) + low revision velocity ({:.2}): truth values likely uninitialized",
                signals.truth_entropy, signals.revision_velocity
            ),
            triplets_affected: signals.active_triplets,
        });
    }

    if signals.deduction_yield < 0.1 && signals.truth_entropy < 0.5 {
        actions.push(HealingAction {
            action: HealingType::InferMissingLinks,
            reason: "Low deduction yield but consistent data — inference can fill gaps".into(),
            triplets_affected: 0,
        });
    }

    if signals.episodic_saturation > 0.85 {
        actions.push(HealingAction {
            action: HealingType::CompactDeleted,
            reason: format!(
                "Episodic saturation {:.0}% — compact to free space",
                signals.episodic_saturation * 100.0
            ),
            triplets_affected: 0,
        });
    }

    if signals.truth_entropy < 0.1 && signals.contradiction_rate < 0.05 {
        actions.push(HealingAction {
            action: HealingType::NormalizeTruth,
            reason: "Very low entropy with few contradictions — possible truth inflation".into(),
            triplets_affected: signals.active_triplets,
        });
    }

    actions
}

/// Apply healing actions to the graph.
///
/// Returns the number of triplets modified.
pub fn apply_healing(graph: &mut TripletGraph, actions: &[HealingAction]) -> usize {
    use crate::graph::spo::truth::TruthValue;
    let mut modified = 0;

    for action in actions {
        match action.action {
            HealingType::ResolveContradictions => {
                let contradictions = graph.detect_contradictions(0.3);
                for (i, j) in &contradictions {
                    // Reduce confidence of both contradicting triplets.
                    let ci = graph.triplets[*i].truth.confidence;
                    let cj = graph.triplets[*j].truth.confidence;
                    graph.triplets[*i].truth = TruthValue::new(
                        graph.triplets[*i].truth.frequency,
                        ci * 0.5,
                    );
                    graph.triplets[*j].truth = TruthValue::new(
                        graph.triplets[*j].truth.frequency,
                        cj * 0.5,
                    );
                    modified += 2;
                }
            }
            HealingType::BootstrapTruth => {
                for t in graph.triplets.iter_mut() {
                    if t.truth.confidence < 0.05 && t.truth.confidence > 0.0 {
                        // Set from weak evidence (1 positive, 0 negative).
                        t.truth = TruthValue::new(t.truth.frequency.max(0.5), 0.1);
                        modified += 1;
                    }
                }
            }
            HealingType::InferMissingLinks => {
                let inferred = graph.infer_deductions();
                let count = inferred.len();
                graph.add_triplets(&inferred);
                modified += count;
            }
            HealingType::CompactDeleted => {
                let before = graph.triplets.len();
                graph.triplets.retain(|t| !t.is_deleted());
                // Rebuild entity index after compaction.
                graph.entity_index.clear();
                for (idx, t) in graph.triplets.iter().enumerate() {
                    graph
                        .entity_index
                        .entry(t.subject.clone())
                        .or_default()
                        .push(idx);
                    graph
                        .entity_index
                        .entry(t.object.clone())
                        .or_default()
                        .push(idx);
                }
                modified += before - graph.triplets.len();
            }
            HealingType::NormalizeTruth => {
                let max_c = graph
                    .triplets
                    .iter()
                    .filter(|t| !t.is_deleted())
                    .map(|t| t.truth.confidence)
                    .fold(0.0f32, f32::max);
                if max_c > 0.95 {
                    let scale = 0.95 / max_c;
                    for t in graph.triplets.iter_mut() {
                        if !t.is_deleted() {
                            t.truth = TruthValue::new(
                                t.truth.frequency,
                                t.truth.confidence * scale,
                            );
                            modified += 1;
                        }
                    }
                }
            }
        }
    }

    modified
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::triplet_graph::Triplet;

    fn make_graph() -> TripletGraph {
        let mut g = TripletGraph::new();
        g.add_triplets(&[
            Triplet::new("alice", "bob", "knows", 1),
            Triplet::new("bob", "carol", "knows", 2),
            Triplet::new("carol", "dave", "knows", 3),
            Triplet::new("alice", "bob", "hates", 4), // contradicts "knows"
        ]);
        g
    }

    #[test]
    fn test_from_graph_computes_signals() {
        let graph = make_graph();
        let memory = EpisodicMemory::new(10);
        let signals = GraphSensorium::from_graph(&graph, &memory, 3, 10, 5, 2);
        assert_eq!(signals.active_triplets, 4);
        assert!(signals.contradiction_rate > 0.0); // alice→bob has two relations
        assert!(signals.truth_entropy >= 0.0);
        assert!((signals.revision_velocity - 0.3).abs() < 0.01); // 3/10
        assert!((signals.deduction_yield - 0.4).abs() < 0.01); // 2/5
    }

    #[test]
    fn test_suggested_bias_resolve() {
        let mut signals = GraphSensorium::compute(
            100, 40, &[10, 10, 30, 30, 20], 1, 10, 20, 50, 2, 20, 5, 20,
        );
        assert_eq!(signals.suggested_bias(), GraphBias::Resolve);
    }

    #[test]
    fn test_suggested_bias_stagnant() {
        let signals = GraphSensorium::compute(
            100, 5, &[20, 20, 20, 20, 20], 0, 20, 2, 100, 0, 10, 5, 20,
        );
        assert_eq!(signals.suggested_bias(), GraphBias::Stagnant);
    }

    #[test]
    fn test_suggested_bias_exploit() {
        let signals = GraphSensorium::compute(
            200, 2, &[180, 10, 5, 3, 2], 15, 20, 2, 100, 15, 20, 5, 50,
        );
        assert_eq!(signals.suggested_bias(), GraphBias::Exploit);
    }

    #[test]
    fn test_diagnose_contradictions() {
        let signals = GraphSensorium::compute(
            100, 20, &[10, 10, 30, 30, 20], 1, 10, 10, 50, 2, 20, 5, 20,
        );
        let actions = diagnose_healing(&signals);
        assert!(actions.iter().any(|a| a.action == HealingType::ResolveContradictions));
    }

    #[test]
    fn test_diagnose_bootstrap() {
        let signals = GraphSensorium::compute(
            100, 5, &[20, 20, 20, 20, 20], 0, 20, 2, 100, 0, 10, 5, 20,
        );
        let actions = diagnose_healing(&signals);
        assert!(actions.iter().any(|a| a.action == HealingType::BootstrapTruth));
    }

    #[test]
    fn test_apply_healing_resolve_contradictions() {
        let mut graph = make_graph();
        let signals = GraphSensorium::from_graph(&graph, &EpisodicMemory::new(10), 0, 1, 0, 0);
        let actions = diagnose_healing(&signals);

        if actions.iter().any(|a| a.action == HealingType::ResolveContradictions) {
            let modified = apply_healing(&mut graph, &actions);
            assert!(modified > 0);
            // Contradicting triplets should have reduced confidence.
        }
    }

    #[test]
    fn test_apply_healing_infer_links() {
        let mut graph = TripletGraph::new();
        graph.add_triplets(&[
            Triplet::new("alice", "bob", "knows", 1),
            Triplet::new("bob", "carol", "knows", 2),
        ]);
        let actions = vec![HealingAction {
            action: HealingType::InferMissingLinks,
            reason: "test".into(),
            triplets_affected: 0,
        }];
        let modified = apply_healing(&mut graph, &actions);
        assert!(modified > 0);
        // Should have added alice→carol deduction.
        assert!(graph.triplets.len() > 2);
    }

    #[test]
    fn test_apply_healing_compact() {
        let mut graph = TripletGraph::new();
        graph.add_triplets(&[
            Triplet::new("alice", "bob", "knows", 1),
            Triplet::new("bob", "carol", "knows", 2),
        ]);
        graph.delete_triplets(&[("alice".into(), "knows".into(), "bob".into())]);
        assert_eq!(graph.triplets.len(), 2); // still 2 (soft deleted)

        let actions = vec![HealingAction {
            action: HealingType::CompactDeleted,
            reason: "test".into(),
            triplets_affected: 0,
        }];
        apply_healing(&mut graph, &actions);
        assert_eq!(graph.triplets.len(), 1); // compacted to 1
    }

    #[test]
    fn test_empty_graph_sensorium() {
        let graph = TripletGraph::new();
        let memory = EpisodicMemory::new(10);
        let signals = GraphSensorium::from_graph(&graph, &memory, 0, 1, 0, 0);
        assert_eq!(signals.active_triplets, 0);
        assert_eq!(signals.suggested_bias(), GraphBias::Balanced);
    }
}
