//! Strategy #12: WorkflowDAG — Workflow orchestration as graph execution (from LangGraph).
//!
//! A workflow `TaskA → TaskB → TaskC` is just a graph pattern
//! `(a:Task)-[:DEPENDS_ON]->(b:Task)-[:DEPENDS_ON]->(c:Task)`
//! with execution semantics.

use crate::ir::{Arena, LogicalOp};
use crate::traits::*;
use crate::PlanError;

#[derive(Debug)]
pub struct WorkflowDAG;

impl PlanStrategy for WorkflowDAG {
    fn name(&self) -> &str {
        "workflow_dag"
    }
    fn capability(&self) -> PlanCapability {
        PlanCapability::WorkflowOrchestration
    }

    fn affinity(&self, context: &PlanContext) -> f32 {
        if context.features.has_workflow {
            0.9
        } else {
            0.05 // Not relevant for pure queries
        }
    }

    fn plan(
        &self,
        input: PlanInput,
        _arena: &mut Arena<LogicalOp>,
    ) -> Result<PlanInput, PlanError> {
        // In full implementation:
        // 1. Parse workflow task declarations from query
        // 2. Build task dependency graph (just another graph pattern)
        // 3. Apply LangGraph-style channel+reducer semantics:
        //    - Each task has input/output channels
        //    - Reducers merge results from parallel tasks
        //    - Checkpoints between pipeline stages for replay
        // 4. Map to same morsel-driven pipeline executor
        //
        // The key insight: workflow tasks ARE graph traversal nodes.
        // The planner treats them identically.
        Ok(input)
    }
}
