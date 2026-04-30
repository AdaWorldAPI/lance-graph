//! NARS inference types — shared across all consumers.
//!
//! Reconciles n8n-rs InferenceType with lance-graph-planner NarsInferenceType.

/// NARS inference type — determines the reasoning strategy.
///
/// Used by:
/// - n8n-rs ThinkingMode dispatch → QueryPlan
/// - lance-graph-planner → semiring selection
/// - crewai-rust NARS driver → truth value computation
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum InferenceType {
    /// Direct lookup: "I know X, find X" → exact CAM search.
    Deduction,
    /// Pattern matching: "Things like X" → wide CAM scan.
    Induction,
    /// Root cause: "Why did X happen?" → full DN-tree traversal.
    Abduction,
    /// Update belief: "X changed" → bundle_into with learning rate.
    Revision,
    /// Cross-domain: "Connect X and Y" → multi-path bundle.
    Synthesis,
}

/// Query strategy — how to execute a given inference type.
///
/// Maps 1:1 with n8n-rs QueryPlan variants but without the parameters.
/// Parameters come from ThinkingMode or FieldModulation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum QueryStrategy {
    /// Exact CAM search (Deduction).
    CamExact,
    /// Wide CAM scan (Induction).
    CamWide,
    /// Full DN-tree traversal (Abduction).
    DnTreeFull,
    /// Bundle into existing node (Revision).
    BundleInto,
    /// Bundle across paths (Synthesis).
    BundleAcross,
}

impl InferenceType {
    /// Map inference type to default query strategy.
    pub fn default_strategy(&self) -> QueryStrategy {
        match self {
            Self::Deduction => QueryStrategy::CamExact,
            Self::Induction => QueryStrategy::CamWide,
            Self::Abduction => QueryStrategy::DnTreeFull,
            Self::Revision => QueryStrategy::BundleInto,
            Self::Synthesis => QueryStrategy::BundleAcross,
        }
    }
}

/// Semiring choice — how to combine evidence across paths.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SemiringChoice {
    /// Boolean AND/OR (standard graph pattern matching).
    Boolean,
    /// Hamming distance minimum (nearest-neighbor search).
    HammingMin,
    /// NARS truth value conjunction (evidence fusion).
    NarsTruth,
    /// XOR bundle (creative association, multi-path binding).
    XorBundle,
    /// CAM-PQ ADC distance (compressed vector search).
    CamPqAdc,
}
