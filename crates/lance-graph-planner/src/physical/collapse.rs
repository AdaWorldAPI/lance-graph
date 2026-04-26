//! COLLAPSE: Apply GateState thresholds.
//!
//! From agi-chat's CollapseGate / Two-Stroke engine:
//! - FLOW (SD < 0.15): emit results — high coherence, proceed
//! - HOLD (0.15 ≤ SD < 0.35): persist to SPPM for later resolution
//! - BLOCK (SD ≥ 0.35): discard — too much dispersion
//!
//! The SD (standard deviation) is computed over the resonance scores
//! in each result group. Low dispersion = coherent result = FLOW.

#[allow(unused_imports)] // Morsel intended for collapse execution wiring
use super::{PhysicalOperator, Morsel};
use crate::ir::logical_op::CollapseGate;

/// COLLAPSE physical operator.
#[derive(Debug)]
pub struct CollapseOp {
    /// Gate thresholds.
    pub gate: CollapseGate,
    /// Child operator (typically AccumulateOp).
    pub child: Box<dyn PhysicalOperator>,
    /// Estimated output cardinality (after filtering).
    pub estimated_cardinality: f64,
}

/// Result of collapse gating.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GateState {
    /// Coherent result — emit to output.
    Flow,
    /// Uncertain result — persist for later resolution (SPPM).
    Hold,
    /// Incoherent result — discard.
    Block,
}

impl CollapseOp {
    /// Classify a set of resonance scores into a gate state.
    pub fn classify(&self, scores: &[f64]) -> GateState {
        if scores.is_empty() {
            return GateState::Block;
        }

        let sd = standard_deviation(scores);

        if sd < self.gate.flow_threshold {
            GateState::Flow
        } else if sd < self.gate.hold_threshold {
            GateState::Hold
        } else {
            GateState::Block
        }
    }

    /// Execute collapse: partition results into FLOW / HOLD / BLOCK.
    pub fn execute(
        &self,
        results: Vec<(Vec<f64>, Vec<u8>)>, // (scores, data) per result group
    ) -> CollapseResult {
        let mut flow = Vec::new();
        let mut hold = Vec::new();
        let mut blocked = 0usize;

        for (scores, data) in results {
            match self.classify(&scores) {
                GateState::Flow => flow.push(data),
                GateState::Hold => hold.push(data),
                GateState::Block => blocked += 1,
            }
        }

        CollapseResult { flow, hold, blocked }
    }
}

/// Result of collapse operation.
#[derive(Debug)]
pub struct CollapseResult {
    /// Results that passed FLOW gate — emit to client.
    pub flow: Vec<Vec<u8>>,
    /// Results in HOLD state — persist to SPPM for later.
    pub hold: Vec<Vec<u8>>,
    /// Count of blocked (discarded) results.
    pub blocked: usize,
}

impl PhysicalOperator for CollapseOp {
    fn name(&self) -> &str { "Collapse" }
    fn cardinality(&self) -> f64 { self.estimated_cardinality }
    fn is_pipeline_breaker(&self) -> bool { false }
    fn children(&self) -> Vec<&dyn PhysicalOperator> { vec![&*self.child] }
}

fn standard_deviation(values: &[f64]) -> f64 {
    if values.len() <= 1 {
        return 0.0;
    }
    let mean = values.iter().sum::<f64>() / values.len() as f64;
    let variance = values.iter()
        .map(|v| (v - mean).powi(2))
        .sum::<f64>() / values.len() as f64;
    variance.sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_collapse_flow() {
        let op = CollapseOp {
            gate: CollapseGate::default(),
            child: Box::new(DummyOp),
            estimated_cardinality: 10.0,
        };

        // Very coherent scores (low SD)
        let scores = vec![0.85, 0.86, 0.84, 0.85, 0.87];
        assert_eq!(op.classify(&scores), GateState::Flow);
    }

    #[test]
    fn test_collapse_hold() {
        let op = CollapseOp {
            gate: CollapseGate::default(),
            child: Box::new(DummyOp),
            estimated_cardinality: 10.0,
        };

        // Moderate dispersion (SD ≈ 0.24, between 0.15 and 0.35)
        let scores = vec![0.2, 0.6, 0.3, 0.7, 0.25];
        let state = op.classify(&scores);
        assert_eq!(state, GateState::Hold);
    }

    #[test]
    fn test_collapse_block() {
        let op = CollapseOp {
            gate: CollapseGate::default(),
            child: Box::new(DummyOp),
            estimated_cardinality: 10.0,
        };

        // High dispersion (SD ≈ 0.40, above 0.35)
        let scores = vec![0.0, 1.0, 0.0, 1.0, 0.0];
        assert_eq!(op.classify(&scores), GateState::Block);
    }

    // Dummy operator for testing
    #[derive(Debug)]
    struct DummyOp;
    impl PhysicalOperator for DummyOp {
        fn name(&self) -> &str { "Dummy" }
        fn cardinality(&self) -> f64 { 0.0 }
        fn is_pipeline_breaker(&self) -> bool { false }
        fn children(&self) -> Vec<&dyn PhysicalOperator> { vec![] }
    }
}
