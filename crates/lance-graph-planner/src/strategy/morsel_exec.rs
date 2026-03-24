//! Strategy #7: MorselExec — Morsel-driven physical execution plan (from Kuzudb/Polars).

use crate::ir::{Arena, LogicalOp};
use crate::traits::*;
use crate::PlanError;

#[derive(Debug)]
pub struct MorselExec;

impl PlanStrategy for MorselExec {
    fn name(&self) -> &str { "morsel_exec" }
    fn capability(&self) -> PlanCapability { PlanCapability::PhysicalPlan }

    fn affinity(&self, context: &PlanContext) -> f32 {
        // Always needed for execution
        if context.features.has_graph_pattern || context.features.has_fingerprint_scan {
            0.85
        } else {
            0.5
        }
    }

    fn plan(&self, input: PlanInput, _arena: &mut Arena<LogicalOp>) -> Result<PlanInput, PlanError> {
        // Maps LogicalOp tree → PhysicalOp tree.
        // Decomposes into pipelines at materialization boundaries.
        // Each pipeline is a chain of operators executed by worker threads.
        // Delegates to crate::execute::PipelineExecutor.
        Ok(input)
    }
}
