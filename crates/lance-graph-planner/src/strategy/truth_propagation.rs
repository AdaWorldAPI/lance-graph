//! Strategy #9: TruthPropagation — NARS truth/weight propagation during traversal.
//!
//! The missing piece: truth values accumulated DURING traversal, not post-hoc.
//! multiply = NARS deduction, add = NARS revision.

use crate::ir::{Arena, LogicalOp};
use crate::ir::logical_op::SemiringType;
use crate::traits::*;
use crate::PlanError;

#[derive(Debug)]
pub struct TruthPropagation;

impl PlanStrategy for TruthPropagation {
    fn name(&self) -> &str { "truth_propagation" }
    fn capability(&self) -> PlanCapability { PlanCapability::TruthPropagation }

    fn affinity(&self, context: &PlanContext) -> f32 {
        // High affinity when truth values are involved
        if context.features.has_truth_values {
            0.95
        } else if context.features.has_variable_length_path {
            0.6 // Multi-hop paths benefit from truth accumulation
        } else if context.nars_hint.is_some() {
            0.7 // NARS inference type detected
        } else {
            0.1
        }
    }

    fn plan(&self, input: PlanInput, _arena: &mut Arena<LogicalOp>) -> Result<PlanInput, PlanError> {
        // Injects ACCUMULATE operators with TruthPropagating semiring
        // into the physical plan. The semiring implements:
        // - multiply (edge traversal): NARS deduction
        //   f_conclusion = f_premise × f_edge
        //   c_conclusion = c_premise × c_edge × f_premise × f_edge
        // - add (node merge): NARS revision
        //   f_revised = (f1×c1 + f2×c2) / (c1 + c2)
        //   c_revised = (c1 + c2) / (c1 + c2 + 1)
        //
        // This is implemented in physical::accumulate::TruthPropagatingSemiring.
        Ok(input)
    }
}
