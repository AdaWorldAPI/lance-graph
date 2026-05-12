//! Strategy #2: ArenaIR — Build arena-allocated logical plan (Polars pattern).

use crate::ir::logical_op::*;
#[allow(unused_imports)] // intended for property propagation during plan building
use crate::ir::properties::PlanProperties;
#[allow(unused_imports)] // intended for schema-aware plan building
use crate::ir::schema::Schema;
use crate::ir::{Arena, LogicalOp, LogicalPlan, Node};
use crate::traits::*;
use crate::PlanError;

#[derive(Debug)]
pub struct ArenaIR;

impl PlanStrategy for ArenaIR {
    fn name(&self) -> &str {
        "arena_ir"
    }
    fn capability(&self) -> PlanCapability {
        PlanCapability::LogicalPlan
    }

    fn affinity(&self, context: &PlanContext) -> f32 {
        // High affinity if we have a graph pattern to plan
        if context.features.has_graph_pattern {
            0.9
        } else if context.features.has_mutation {
            0.7
        } else {
            0.3
        }
    }

    fn plan(
        &self,
        mut input: PlanInput,
        arena: &mut Arena<LogicalOp>,
    ) -> Result<PlanInput, PlanError> {
        // Build a basic logical plan from detected features.
        // Real implementation receives AST from CypherParse strategy.

        let root = if input.context.features.has_resonance
            || input.context.features.has_fingerprint_scan
        {
            // Resonance query: BROADCAST → SCAN → ACCUMULATE → COLLAPSE
            build_resonance_plan(arena, &input.context)
        } else if input.context.features.has_graph_pattern {
            // Standard graph query: ScanNode(s) → Join(s) → Filter → Return
            build_graph_plan(arena, &input.context)
        } else {
            arena.push(LogicalOp::EmptyResult)
        };

        let expr_arena = crate::ir::Arena::new();
        let plan = LogicalPlan::new(std::mem::take(arena), expr_arena, root);

        input.plan = Some(plan);
        Ok(input)
    }
}

fn build_resonance_plan(arena: &mut Arena<LogicalOp>, _context: &PlanContext) -> Node {
    let broadcast = arena.push(LogicalOp::Broadcast {
        fingerprint: crate::ir::expr::ExprNode(Node(0)),
        partitions: 4,
    });

    let scan = arena.push(LogicalOp::Scan {
        input: broadcast,
        strategy: ScanStrategy::Cascade,
        threshold: 1000,
        top_k: 10,
    });

    let accumulate = arena.push(LogicalOp::Accumulate {
        input: scan,
        semiring: SemiringType::XorBundle,
        traversal: scan,
    });

    arena.push(LogicalOp::Collapse {
        input: accumulate,
        gate: CollapseGate::default(),
    })
}

fn build_graph_plan(arena: &mut Arena<LogicalOp>, _context: &PlanContext) -> Node {
    // Single scan for now — DPJoinEnum will optimize multi-node patterns
    arena.push(LogicalOp::ScanNode {
        label: "Node".into(),
        alias: "n".into(),
        projections: None,
    })
}
