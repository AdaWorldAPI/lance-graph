//! Graph Sensorium — shared contract types for real-time graph health signals.
//!
//! These types define the wire format between:
//! - **lance-graph arigraph** (produces `GraphSensorium` from live `TripletGraph`)
//! - **q2 orchestrator** (consumes signals for MUL self-regulation)
//! - **cockpit /mri endpoint** (visualizes brain activation + plasticity)
//!
//! The canonical computation (`from_graph()`, `apply_healing()`) lives in lance-graph.
//! This crate defines only the **data shapes** — no graph access, no computation.
//!
//! # Zero dependencies
//!
//! This module has no dependencies beyond `core`. Consumers re-derive
//! serde traits in their own crates if they need serialization.

/// Real-time signals from the knowledge graph for MUL self-regulation.
///
/// Each field is a normalized [0, 1] signal. lance-graph computes these from
/// the live `TripletGraph`; q2's orchestrator consumes them for style selection.
#[derive(Debug, Clone, Copy)]
pub struct GraphSensorium {
    /// Contradiction rate: contradictions / active_triplets.
    /// High = conflicting evidence → increase exploration.
    pub contradiction_rate: f32,
    /// Truth entropy: Shannon entropy of confidence distribution.
    /// High = uncertain about everything → increase exploration.
    pub truth_entropy: f32,
    /// Revision velocity: revisions per step.
    /// High = actively learning → maintain current mode.
    pub revision_velocity: f32,
    /// Plasticity flux: fraction of entities in transition.
    /// High = environment changing → lower trust, increase exploration.
    pub plasticity_flux: f32,
    /// Deduction yield: inferred / attempted.
    /// High = rich graph → plan/act more.
    pub deduction_yield: f32,
    /// Episodic saturation: count / capacity.
    /// High = memory full → consider forgetting.
    pub episodic_saturation: f32,
    /// Active triplet count (raw, not normalized).
    pub active_triplets: usize,
    /// Total entity count.
    pub total_entities: usize,
    /// Contradiction count (raw).
    pub contradictions: usize,
}

impl GraphSensorium {
    /// Derive suggested cognitive bias from the signals.
    ///
    /// This is pure computation on the signals — no graph access needed.
    /// Both lance-graph and q2 can call this.
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

    /// Zero sensorium (empty graph, no signals).
    pub const fn empty() -> Self {
        Self {
            contradiction_rate: 0.0,
            truth_entropy: 0.0,
            revision_velocity: 0.0,
            plasticity_flux: 0.0,
            deduction_yield: 0.0,
            episodic_saturation: 0.0,
            active_triplets: 0,
            total_entities: 0,
            contradictions: 0,
        }
    }
}

/// Graph-suggested cognitive bias.
///
/// Derived from `GraphSensorium::suggested_bias()`. Used by MUL to
/// modulate thinking style selection and temperature.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
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

/// Types of healing actions the graph immune system can apply.
///
/// lance-graph produces these via `diagnose_healing()` and applies them
/// via `apply_healing()`. q2 can display them and request healing via API.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HealingType {
    /// Set very-low-confidence triplets to weak prior (f=0.5, c=0.1).
    BootstrapTruth,
    /// Halve confidence of contradicting triplet pairs.
    ResolveContradictions,
    /// Run NARS deduction to fill missing A→C links.
    InferMissingLinks,
    /// Remove soft-deleted triplets and rebuild index.
    CompactDeleted,
    /// Scale all confidences so max = 0.95 (prevent inflation).
    NormalizeTruth,
}

/// A healing action with reason and scope.
#[derive(Debug, Clone)]
pub struct HealingAction {
    pub action: HealingType,
    pub reason: &'static str,
    pub triplets_affected: usize,
}

/// Diagnose healing from sensorium signals (no graph access needed).
///
/// Returns a prioritized list of healing actions. The caller (lance-graph)
/// applies them; q2 displays them as recommendations.
pub fn diagnose_healing(signals: &GraphSensorium) -> Vec<HealingAction> {
    let mut actions = Vec::new();

    if signals.contradiction_rate > 0.15 {
        actions.push(HealingAction {
            action: HealingType::ResolveContradictions,
            reason: "Contradiction rate exceeds 15% threshold",
            triplets_affected: signals.contradictions,
        });
    }

    if signals.truth_entropy > 0.6 && signals.revision_velocity < 0.1 {
        actions.push(HealingAction {
            action: HealingType::BootstrapTruth,
            reason: "High entropy + low revision: truth values likely uninitialized",
            triplets_affected: signals.active_triplets,
        });
    }

    if signals.deduction_yield < 0.1 && signals.truth_entropy < 0.5 {
        actions.push(HealingAction {
            action: HealingType::InferMissingLinks,
            reason: "Low deduction yield but consistent data",
            triplets_affected: 0,
        });
    }

    if signals.episodic_saturation > 0.85 {
        actions.push(HealingAction {
            action: HealingType::CompactDeleted,
            reason: "Episodic saturation above 85%",
            triplets_affected: 0,
        });
    }

    if signals.truth_entropy < 0.1 && signals.contradiction_rate < 0.05 {
        actions.push(HealingAction {
            action: HealingType::NormalizeTruth,
            reason: "Very low entropy with few contradictions — possible truth inflation",
            triplets_affected: signals.active_triplets,
        });
    }

    actions
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_empty_sensorium() {
        let s = GraphSensorium::empty();
        assert_eq!(s.suggested_bias(), GraphBias::Balanced);
    }

    #[test]
    fn test_high_contradiction_resolves() {
        let s = GraphSensorium {
            contradiction_rate: 0.4,
            truth_entropy: 0.5,
            revision_velocity: 0.3,
            plasticity_flux: 0.2,
            deduction_yield: 0.3,
            episodic_saturation: 0.3,
            active_triplets: 100,
            total_entities: 50,
            contradictions: 40,
        };
        assert_eq!(s.suggested_bias(), GraphBias::Resolve);
    }

    #[test]
    fn test_high_entropy_explores() {
        let s = GraphSensorium {
            contradiction_rate: 0.1,
            truth_entropy: 0.8,
            revision_velocity: 0.3,
            plasticity_flux: 0.2,
            deduction_yield: 0.1,
            episodic_saturation: 0.3,
            active_triplets: 100,
            total_entities: 50,
            contradictions: 10,
        };
        assert_eq!(s.suggested_bias(), GraphBias::Explore);
    }

    #[test]
    fn test_diagnose_contradictions() {
        let s = GraphSensorium {
            contradiction_rate: 0.25,
            truth_entropy: 0.3,
            revision_velocity: 0.5,
            plasticity_flux: 0.1,
            deduction_yield: 0.5,
            episodic_saturation: 0.2,
            active_triplets: 100,
            total_entities: 50,
            contradictions: 25,
        };
        let actions = diagnose_healing(&s);
        assert!(actions.iter().any(|a| a.action == HealingType::ResolveContradictions));
    }

    #[test]
    fn test_diagnose_bootstrap() {
        let s = GraphSensorium {
            contradiction_rate: 0.05,
            truth_entropy: 0.8,
            revision_velocity: 0.02,
            plasticity_flux: 0.1,
            deduction_yield: 0.0,
            episodic_saturation: 0.2,
            active_triplets: 100,
            total_entities: 50,
            contradictions: 5,
        };
        let actions = diagnose_healing(&s);
        assert!(actions.iter().any(|a| a.action == HealingType::BootstrapTruth));
    }

    #[test]
    fn test_healthy_graph_no_healing() {
        let s = GraphSensorium {
            contradiction_rate: 0.02,
            truth_entropy: 0.3,
            revision_velocity: 0.3,
            plasticity_flux: 0.1,
            deduction_yield: 0.5,
            episodic_saturation: 0.3,
            active_triplets: 100,
            total_entities: 50,
            contradictions: 2,
        };
        let actions = diagnose_healing(&s);
        assert!(actions.is_empty());
    }
}
