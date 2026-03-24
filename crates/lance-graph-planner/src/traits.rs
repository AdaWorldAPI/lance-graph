//! Core traits for the composable planning strategy system.

use crate::ir::{Arena, LogicalOp, LogicalPlan, Node};
use crate::PlanError;

/// What kind of planning problem a strategy solves.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PlanCapability {
    /// Parse query text into structured form.
    Parse,
    /// Build logical plan representation.
    LogicalPlan,
    /// Optimize join ordering (DP, greedy, etc.).
    JoinOrdering,
    /// Apply rule-based optimization passes.
    RuleOptimization,
    /// Estimate cardinality and cost.
    CostEstimation,
    /// Plan vector/fingerprint scans.
    VectorScan,
    /// Build physical execution plan.
    PhysicalPlan,
    /// Stream/pipeline execution.
    StreamExecution,
    /// Propagate truth/weight values during traversal.
    TruthPropagation,
    /// Apply resonance gating (FLOW/HOLD/BLOCK).
    ResonanceGating,
    /// JIT compile scan kernels.
    JitCompilation,
    /// Plan workflow/DAG execution.
    WorkflowOrchestration,
    /// Extension point for custom planning logic.
    Extension,
}

/// Pipeline phase ordering — strategies compose in this fixed order.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum PipelinePhase {
    Parse = 0,
    Plan = 1,
    Optimize = 2,
    Physicalize = 3,
    Execute = 4,
}

impl PlanCapability {
    /// Which pipeline phase this capability belongs to.
    pub fn phase(&self) -> PipelinePhase {
        match self {
            Self::Parse => PipelinePhase::Parse,
            Self::LogicalPlan | Self::JoinOrdering | Self::WorkflowOrchestration => PipelinePhase::Plan,
            Self::RuleOptimization | Self::CostEstimation => PipelinePhase::Optimize,
            Self::VectorScan | Self::PhysicalPlan | Self::TruthPropagation | Self::ResonanceGating => PipelinePhase::Physicalize,
            Self::StreamExecution | Self::JitCompilation => PipelinePhase::Execute,
            Self::Extension => PipelinePhase::Physicalize, // Extensions can slot in anywhere
        }
    }
}

/// Context passed to strategies for affinity scoring and planning.
#[derive(Debug, Clone)]
pub struct PlanContext {
    /// The raw query string.
    pub query: String,
    /// Detected query features (set by earlier strategies in the pipeline).
    pub features: QueryFeatures,
    /// MUL free will modifier (1.0 if no MUL).
    pub free_will_modifier: f64,
    /// Thinking style vector (23D sparse, None if no thinking orchestration).
    pub thinking_style: Option<Vec<f64>>,
    /// NARS inference type hint (None if not detected).
    pub nars_hint: Option<crate::thinking::NarsInferenceType>,
}

/// Detected query features — set incrementally as strategies analyze the query.
#[derive(Debug, Clone, Default)]
pub struct QueryFeatures {
    pub has_graph_pattern: bool,
    pub has_fingerprint_scan: bool,
    pub has_variable_length_path: bool,
    pub has_aggregation: bool,
    pub has_mutation: bool,
    pub has_workflow: bool,
    pub has_resonance: bool,
    pub has_truth_values: bool,
    pub num_match_clauses: usize,
    pub num_nodes: usize,
    pub num_edges: usize,
    pub estimated_complexity: f64,
}

/// Input/output for a strategy in the pipeline.
#[derive(Debug)]
pub struct PlanInput {
    /// Current plan state (None for the first strategy in pipeline).
    pub plan: Option<LogicalPlan>,
    /// Context (accumulated through pipeline).
    pub context: PlanContext,
}

/// A composable planning strategy.
pub trait PlanStrategy: Send + Sync + std::fmt::Debug {
    /// Human-readable name for the strategy.
    fn name(&self) -> &str;

    /// What kind of planning problem this strategy solves.
    fn capability(&self) -> PlanCapability;

    /// Can this strategy handle this query shape? Returns confidence 0.0..1.0.
    /// Higher = more suitable for this query.
    fn affinity(&self, context: &PlanContext) -> f32;

    /// Produce or refine a plan.
    /// Receives the accumulated plan from previous strategies.
    /// Returns the refined plan.
    fn plan(&self, input: PlanInput, arena: &mut Arena<LogicalOp>) -> Result<PlanInput, PlanError>;
}
