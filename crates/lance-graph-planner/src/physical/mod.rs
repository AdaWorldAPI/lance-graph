//! # Physical Operators
//!
//! The four fused resonance phases as physical plan operators:
//! BROADCAST(fingerprint) → SCAN(strategy) → ACCUMULATE(semiring) → COLLAPSE(gate)
//!
//! Plus standard graph operators mapped from logical plan.

pub mod accumulate;
pub mod broadcast;
pub mod cam_pq_scan;
pub mod collapse;
pub mod scan;

pub use accumulate::AccumulateOp;
pub use broadcast::BroadcastOp;
pub use cam_pq_scan::CamPqScanOp;
pub use collapse::CollapseOp;
pub use scan::ScanOp;

/// Physical operator trait.
/// Each operator processes morsels of data in a pipeline.
pub trait PhysicalOperator: std::fmt::Debug + Send + Sync {
    /// Human-readable name.
    fn name(&self) -> &str;

    /// Estimated output cardinality.
    fn cardinality(&self) -> f64;

    /// Whether this operator is a pipeline breaker (requires materialization).
    fn is_pipeline_breaker(&self) -> bool;

    /// Children of this operator.
    fn children(&self) -> Vec<&dyn PhysicalOperator>;
}

/// A morsel of data (batch of rows) flowing through the pipeline.
/// In the real implementation, this wraps Arrow RecordBatch.
#[derive(Debug, Clone)]
pub struct Morsel {
    /// Number of rows in this morsel.
    pub num_rows: usize,
    /// Column data (placeholder — real impl uses Arrow arrays).
    pub columns: Vec<ColumnData>,
}

/// Column data in a morsel (placeholder for Arrow integration).
#[derive(Debug, Clone)]
pub enum ColumnData {
    Int64(Vec<i64>),
    Float64(Vec<f64>),
    String(Vec<String>),
    /// Fingerprint column: Vec of u64 arrays.
    Fingerprint(Vec<Vec<u64>>),
    /// Truth value column: Vec of (frequency, confidence) pairs.
    TruthValue(Vec<(f64, f64)>),
}

/// Physical plan: tree of physical operators.
#[derive(Debug)]
pub struct PhysicalPlan {
    pub root: Box<dyn PhysicalOperator>,
    /// Pipeline decomposition (set by the executor).
    pub pipelines: Vec<Pipeline>,
}

/// A pipeline: chain of operators between pipeline breakers.
#[derive(Debug, Clone)]
pub struct Pipeline {
    pub id: usize,
    /// Operator indices in this pipeline (from source to sink).
    pub operators: Vec<usize>,
    /// Dependencies: pipelines that must complete before this one starts.
    pub dependencies: Vec<usize>,
}
