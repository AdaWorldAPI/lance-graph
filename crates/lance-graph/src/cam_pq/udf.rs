//! DataFusion UDF for CAM-PQ distance computation.
//!
//! Registers `cam_distance(query_vector, cam_column)` as a scalar UDF
//! that lance-graph's DataFusion planner can invoke in physical plans.
//!
//! Under the hood: precomputes distance tables from the query vector,
//! then calls ndarray's `DistanceTables::distance_batch()` (AVX-512)
//! on the CAM column.
//!
//! # Usage in Cypher/SQL
//!
//! ```sql
//! SELECT id, cam_distance(ARRAY[0.1, 0.2, ...], cam) AS dist
//! FROM vectors
//! ORDER BY dist ASC
//! LIMIT 10
//! ```

/// CAM-PQ UDF configuration.
///
/// Holds the codebook reference and provides the UDF factory.
/// In production, the codebook is loaded from a Lance table once
/// and shared across all queries via `Arc<CamCodebook>`.
#[derive(Debug, Clone)]
pub struct CamPqUdfConfig {
    /// Total vector dimension (must match codebook).
    pub total_dim: usize,
    /// Number of subspaces (always 6 for CAM-PQ).
    pub num_subspaces: usize,
    /// Codebook table name in the Lance dataset.
    pub codebook_table: String,
}

impl Default for CamPqUdfConfig {
    fn default() -> Self {
        Self {
            total_dim: 1024,
            num_subspaces: 6,
            codebook_table: "cam_codebook".into(),
        }
    }
}

/// Register the cam_distance UDF with a DataFusion SessionContext.
///
/// The UDF takes two arguments:
/// 1. `query_vector` — Float32 array (the full-precision query)
/// 2. `cam_column` — FixedSizeBinary(6) (the compressed CAM fingerprint)
///
/// Returns Float32 (squared L2 distance via ADC).
///
/// # Implementation Notes
///
/// The precompute step (distance tables from query) runs ONCE per query,
/// not per row. DataFusion's execution model evaluates the UDF per-batch,
/// so the tables are computed on the first batch and cached for the rest.
///
/// The actual distance computation calls into ndarray:
/// `ndarray::hpc::cam_pq::DistanceTables::distance_batch_avx512()`
pub fn register_cam_distance_udf(config: &CamPqUdfConfig) -> CamDistanceUdf {
    CamDistanceUdf {
        total_dim: config.total_dim,
        num_subspaces: config.num_subspaces,
    }
}

/// The CAM distance UDF handle.
///
/// This is a placeholder that will be registered with DataFusion's
/// `SessionContext::register_udf()` once ndarray is wired as a dependency.
/// The actual implementation calls:
///
/// ```text
/// 1. CamCodebook::precompute_distances(query)  → DistanceTables (6KB, L1 cache)
/// 2. DistanceTables::distance_batch(cam_column) → Float32Array of distances
/// ```
#[derive(Debug, Clone)]
pub struct CamDistanceUdf {
    pub total_dim: usize,
    pub num_subspaces: usize,
}
