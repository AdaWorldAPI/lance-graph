//! # Optimizer Framework
//!
//! Composable rule-based optimizer inspired by DataFusion's OptimizerRule trait
//! and Hyrise's validation-after-every-rule approach.
//!
//! 12 optimization passes (from Kuzudb), extended with resonance-specific rules.

mod rule;

pub use rule::{OptimizerRule, ApplyOrder};

use crate::ir::{Arena, LogicalOp, LogicalPlan, Node};
use crate::thinking::ThinkingContext;
use crate::PlanError;

/// Run all optimization passes on a logical plan.
pub fn optimize(
    mut plan: LogicalPlan,
    thinking: &ThinkingContext,
    _arena: &mut Arena<LogicalOp>,
) -> Result<LogicalPlan, PlanError> {
    let rules = build_rule_chain(thinking);

    for rule in &rules {
        let result = rule.apply(&plan, thinking);
        match result {
            RuleResult::Changed(new_root) => {
                plan.root = new_root;
            }
            RuleResult::Unchanged => {}
            RuleResult::Error(e) => {
                tracing::warn!("Optimizer rule {} failed: {}", rule.name(), e);
                // Continue with other rules (Hyrise pattern: don't fail on rule errors)
            }
        }
    }

    Ok(plan)
}

/// Result of applying an optimizer rule.
pub enum RuleResult {
    Changed(Node),
    Unchanged,
    Error(String),
}

/// Build the optimization rule chain based on thinking context.
fn build_rule_chain(thinking: &ThinkingContext) -> Vec<Box<dyn OptimizerRule>> {
    let mut rules: Vec<Box<dyn OptimizerRule>> = vec![
        // Phase 1: Remove factorization (prepare for optimization)
        Box::new(RemoveFactorizationRewriter),
        // Phase 2: Standard optimizations
        Box::new(PredicatePushdown),
        Box::new(ProjectionPushdown),
        Box::new(LimitPushdown),
        // Phase 3: Join optimization
        Box::new(RemoveUnnecessaryJoins),
    ];

    // Phase 4: SIP (if enabled and style is precise enough)
    if thinking.modulation.noise_tolerance < 0.5 {
        rules.push(Box::new(SipOptimizer));
    }

    // Phase 5: Resonance-specific
    rules.push(Box::new(CollapseGateInsertion));
    rules.push(Box::new(SemiringOptimizer));

    // Phase 6: Re-insert factorization
    rules.push(Box::new(FactorizationRewriter));

    // Phase 7: Final cleanup
    rules.push(Box::new(TopKOptimizer));

    rules
}

// === Built-in optimizer rules ===

struct RemoveFactorizationRewriter;
impl OptimizerRule for RemoveFactorizationRewriter {
    fn name(&self) -> &str { "remove_factorization" }
    fn apply(&self, _plan: &LogicalPlan, _ctx: &ThinkingContext) -> RuleResult {
        // Remove Flatten operators to enable optimization, then re-insert later.
        RuleResult::Unchanged // Placeholder
    }
}

struct PredicatePushdown;
impl OptimizerRule for PredicatePushdown {
    fn name(&self) -> &str { "predicate_pushdown" }
    fn apply(&self, _plan: &LogicalPlan, _ctx: &ThinkingContext) -> RuleResult {
        // Push Filter operators below joins/projections.
        // Convert cross-products with equality predicates into hash joins.
        // Convert primary-key equality predicates into index lookups.
        RuleResult::Unchanged // Placeholder — real impl walks the plan tree
    }
}

struct ProjectionPushdown;
impl OptimizerRule for ProjectionPushdown {
    fn name(&self) -> &str { "projection_pushdown" }
    fn apply(&self, _plan: &LogicalPlan, _ctx: &ThinkingContext) -> RuleResult {
        RuleResult::Unchanged
    }
}

struct LimitPushdown;
impl OptimizerRule for LimitPushdown {
    fn name(&self) -> &str { "limit_pushdown" }
    fn apply(&self, _plan: &LogicalPlan, _ctx: &ThinkingContext) -> RuleResult {
        RuleResult::Unchanged
    }
}

struct RemoveUnnecessaryJoins;
impl OptimizerRule for RemoveUnnecessaryJoins {
    fn name(&self) -> &str { "remove_unnecessary_joins" }
    fn apply(&self, _plan: &LogicalPlan, _ctx: &ThinkingContext) -> RuleResult {
        RuleResult::Unchanged
    }
}

/// Sideways Information Passing: after hash join build, create semi-masks
/// that filter probe-side scans to only matching rows.
struct SipOptimizer;
impl OptimizerRule for SipOptimizer {
    fn name(&self) -> &str { "sip_semi_mask" }
    fn apply(&self, _plan: &LogicalPlan, _ctx: &ThinkingContext) -> RuleResult {
        // Walk plan tree. For each HashJoin:
        // 1. After build side completes, create SemiMask
        // 2. Push SemiMask down to probe-side ScanNode
        // This turns hash joins into index-like probes.
        RuleResult::Unchanged
    }
}

/// Insert COLLAPSE gates based on thinking context and query shape.
struct CollapseGateInsertion;
impl OptimizerRule for CollapseGateInsertion {
    fn name(&self) -> &str { "collapse_gate_insertion" }
    fn apply(&self, _plan: &LogicalPlan, _ctx: &ThinkingContext) -> RuleResult {
        // Insert Collapse operators after Accumulate operators
        // when the thinking style requires gated output.
        // Thresholds come from the thinking context's free_will_modifier.
        RuleResult::Unchanged
    }
}

/// Optimize semiring usage: if a TruthPropagating semiring is used
/// but the query is simple enough for Boolean, downgrade.
struct SemiringOptimizer;
impl OptimizerRule for SemiringOptimizer {
    fn name(&self) -> &str { "semiring_optimizer" }
    fn apply(&self, _plan: &LogicalPlan, _ctx: &ThinkingContext) -> RuleResult {
        RuleResult::Unchanged
    }
}

struct FactorizationRewriter;
impl OptimizerRule for FactorizationRewriter {
    fn name(&self) -> &str { "factorization_rewriter" }
    fn apply(&self, _plan: &LogicalPlan, _ctx: &ThinkingContext) -> RuleResult {
        // Re-insert Flatten operators where required by downstream operators.
        // Walk the plan tree. For each operator that requires flat input,
        // check if the schema has an unflat group. If so, insert Flatten.
        RuleResult::Unchanged
    }
}

struct TopKOptimizer;
impl OptimizerRule for TopKOptimizer {
    fn name(&self) -> &str { "topk_optimizer" }
    fn apply(&self, _plan: &LogicalPlan, _ctx: &ThinkingContext) -> RuleResult {
        // Combine OrderBy + Limit into TopK operator.
        RuleResult::Unchanged
    }
}
