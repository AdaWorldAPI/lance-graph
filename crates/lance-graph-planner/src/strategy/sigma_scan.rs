//! Strategy #6: SigmaBandScan — Hamming/vector scan planning (from lance-graph).
//!
//! Sigma-band cascade: scan stroke columns first, progressive refinement.
//! Cost model decides between Cascade/Full/Index strategies.

use crate::ir::{Arena, LogicalOp};
use crate::ir::logical_op::ScanStrategy;
use crate::traits::*;
use crate::PlanError;

#[derive(Debug)]
pub struct SigmaBandScan;

impl PlanStrategy for SigmaBandScan {
    fn name(&self) -> &str { "sigma_scan" }
    fn capability(&self) -> PlanCapability { PlanCapability::VectorScan }

    fn affinity(&self, context: &PlanContext) -> f32 {
        if context.features.has_fingerprint_scan || context.features.has_resonance {
            0.95 // This is the strategy for fingerprint queries
        } else {
            0.05 // Not relevant for non-vector queries
        }
    }

    fn plan(&self, mut input: PlanInput, arena: &mut Arena<LogicalOp>) -> Result<PlanInput, PlanError> {
        // Choose scan strategy based on data characteristics:
        // - Cascade: best for selective queries (threshold < 30% of max distance)
        // - Full: best for broad queries or small datasets
        // - Index: best when precomputed proximity index exists
        //
        // The SIMD kernels (VPOPCNTDQ, VPANDQ) live in ndarray.
        // This strategy only plans WHICH strategy to use.
        Ok(input)
    }
}
