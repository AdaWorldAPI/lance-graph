//! Strategy #10: CollapseGate — Resonance gating from agi-chat's FLOW/HOLD/BLOCK.

use crate::ir::{Arena, LogicalOp};
use crate::traits::*;
use crate::PlanError;

#[derive(Debug)]
pub struct CollapseGateStrategy;

impl PlanStrategy for CollapseGateStrategy {
    fn name(&self) -> &str { "collapse_gate" }
    fn capability(&self) -> PlanCapability { PlanCapability::ResonanceGating }

    fn affinity(&self, context: &PlanContext) -> f32 {
        // Relevant for resonance queries and truth-propagation queries
        if context.features.has_resonance {
            0.9
        } else if context.features.has_truth_values {
            0.6
        } else {
            0.1
        }
    }

    fn plan(&self, input: PlanInput, _arena: &mut Arena<LogicalOp>) -> Result<PlanInput, PlanError> {
        // Inserts COLLAPSE operators after ACCUMULATE operators.
        // Thresholds are derived from the thinking context:
        // - FLOW threshold = 0.15 (crystalline trust)
        // - HOLD threshold = 0.35 (fuzzy trust)
        // - BLOCK = anything above HOLD
        //
        // FLOW results go to client.
        // HOLD results persist to SPPM for later resolution.
        // BLOCK results are discarded.
        //
        // Implemented in physical::collapse::CollapseOp.
        Ok(input)
    }
}
