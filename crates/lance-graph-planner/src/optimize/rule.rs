//! OptimizerRule trait (from DataFusion pattern).
//!
//! Each rule transforms a LogicalPlan → LogicalPlan.
//! Rules declare their traversal order (TopDown or BottomUp).

use super::RuleResult;
use crate::ir::LogicalPlan;
use crate::thinking::ThinkingContext;

/// Apply order for tree traversal.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ApplyOrder {
    /// Apply rule from root to leaves.
    TopDown,
    /// Apply rule from leaves to root.
    BottomUp,
}

/// An optimizer rule that transforms a logical plan.
pub trait OptimizerRule {
    /// Human-readable name for this rule.
    fn name(&self) -> &str;

    /// Apply the rule to the plan. Returns Changed/Unchanged/Error.
    fn apply(&self, plan: &LogicalPlan, ctx: &ThinkingContext) -> RuleResult;

    /// Traversal order (default: BottomUp).
    fn apply_order(&self) -> ApplyOrder {
        ApplyOrder::BottomUp
    }

    /// Whether this rule should be applied iteratively until fixed-point.
    fn iterative(&self) -> bool {
        false
    }

    /// Maximum iterations for iterative rules.
    fn max_iterations(&self) -> usize {
        10
    }
}
