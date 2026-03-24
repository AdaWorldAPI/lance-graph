//! Strategy #13: ExtensionPlanner — Extension point for custom planning logic.
//!
//! From DataFusion's ExtensionPlanner + UserDefinedLogicalNode pattern.
//! First-class escape hatch for custom operators.

use crate::ir::{Arena, LogicalOp};
use crate::traits::*;
use crate::PlanError;

#[derive(Debug)]
pub struct ExtensionPlanner;

impl PlanStrategy for ExtensionPlanner {
    fn name(&self) -> &str { "extension" }
    fn capability(&self) -> PlanCapability { PlanCapability::Extension }

    fn affinity(&self, _context: &PlanContext) -> f32 {
        // Low default affinity — only activates when extensions are registered
        0.1
    }

    fn plan(&self, input: PlanInput, _arena: &mut Arena<LogicalOp>) -> Result<PlanInput, PlanError> {
        // Extension point for user-defined planning logic.
        // Users can register custom PlanStrategy implementations
        // via the strategy registry.
        Ok(input)
    }
}
