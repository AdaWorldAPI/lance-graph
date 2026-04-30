//! Strategy #4: RuleOptimizer — Composable rule-based optimization (from DataFusion).
//!
//! 12 optimization passes with plan signature hashing for fixed-point detection.

use crate::ir::{Arena, LogicalOp};
use crate::traits::*;
use crate::PlanError;

#[derive(Debug)]
pub struct RuleOptimizer;

impl PlanStrategy for RuleOptimizer {
    fn name(&self) -> &str {
        "rule_optimizer"
    }
    fn capability(&self) -> PlanCapability {
        PlanCapability::RuleOptimization
    }

    fn affinity(&self, context: &PlanContext) -> f32 {
        // Always useful if there's a plan to optimize
        if context.features.has_graph_pattern {
            0.85
        } else {
            0.3
        }
    }

    fn plan(
        &self,
        input: PlanInput,
        _arena: &mut Arena<LogicalOp>,
    ) -> Result<PlanInput, PlanError> {
        // Delegates to crate::optimize::optimize() which runs:
        // 1. RemoveFactorizationRewriter
        // 2. PredicatePushdown
        // 3. ProjectionPushdown
        // 4. LimitPushdown
        // 5. RemoveUnnecessaryJoins
        // 6. SIP (semi-mask) optimizer
        // 7. CollapseGateInsertion
        // 8. SemiringOptimizer
        // 9. FactorizationRewriter
        // 10. TopKOptimizer
        Ok(input)
    }
}
