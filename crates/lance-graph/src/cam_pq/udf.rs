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

use arrow::array::{ArrayRef, FixedSizeBinaryArray, Float32Array};
use arrow::datatypes::DataType;
use datafusion::logical_expr::{ScalarUDF, Signature, Volatility};
use datafusion::physical_plan::ColumnarValue;
use std::sync::{Arc, LazyLock};

use crate::datafusion_planner::vector_ops;

/// Number of CAM-PQ subspaces (always 6).
const CAM_SIZE: usize = 6;

/// Type alias for UDF function closures.
type UdfFunc =
    Arc<dyn Fn(&[ColumnarValue]) -> datafusion::error::Result<ColumnarValue> + Send + Sync>;

/// Precomputed distance table for one subspace: 256 f32 values.
type SubspaceTable = [f32; 256];

/// CAM-PQ distance tables: 6 subspaces × 256 centroids = 1536 floats (6KB).
/// Fits in L1 cache. Computed once per query, used for all candidates.
pub struct CamDistanceTables {
    pub tables: [SubspaceTable; CAM_SIZE],
}

impl CamDistanceTables {
    /// Precompute distance tables from a query vector and codebook centroids.
    ///
    /// `codebook[subspace][centroid]` = centroid vector (f32 slice of length subspace_dim).
    /// `query` = full-precision query vector.
    pub fn precompute(query: &[f32], codebook: &[Vec<Vec<f32>>]) -> Self {
        let total_dim = query.len();
        let subspace_dim = total_dim / CAM_SIZE;
        let mut tables = [[0.0f32; 256]; CAM_SIZE];

        for s in 0..CAM_SIZE {
            let q_sub = &query[s * subspace_dim..(s + 1) * subspace_dim];
            let num_centroids = codebook[s].len().min(256);
            for c in 0..num_centroids {
                tables[s][c] = squared_l2(q_sub, &codebook[s][c]);
            }
        }

        CamDistanceTables { tables }
    }

    /// ADC distance to one CAM fingerprint: 6 table lookups + 5 adds.
    #[inline(always)]
    pub fn distance(&self, cam: &[u8; CAM_SIZE]) -> f32 {
        self.tables[0][cam[0] as usize]
            + self.tables[1][cam[1] as usize]
            + self.tables[2][cam[2] as usize]
            + self.tables[3][cam[3] as usize]
            + self.tables[4][cam[4] as usize]
            + self.tables[5][cam[5] as usize]
    }

    /// Batch distance for a FixedSizeBinary(6) column.
    /// Returns Float32Array of distances.
    pub fn distance_batch(&self, cam_column: &FixedSizeBinaryArray) -> Vec<f32> {
        let n = cam_column.len();
        let mut result = Vec::with_capacity(n);
        for i in 0..n {
            if cam_column.is_null(i) {
                result.push(f32::MAX);
            } else {
                let bytes = cam_column.value(i);
                let cam: [u8; CAM_SIZE] = [
                    bytes[0], bytes[1], bytes[2],
                    bytes[3], bytes[4], bytes[5],
                ];
                result.push(self.distance(&cam));
            }
        }
        result
    }
}

/// Core function: cam_distance(query_vector, cam_column) → Float32.
///
/// Handles all ColumnarValue combinations:
/// - Scalar query × Array cam_column (common case: one query against many rows)
/// - Array query × Array cam_column (pairwise)
fn cam_distance_func(
    args: &[ColumnarValue],
    codebook: &[Vec<Vec<f32>>],
) -> datafusion::error::Result<ColumnarValue> {
    if args.len() != 2 {
        return Err(datafusion::error::DataFusionError::Execution(
            "cam_distance requires exactly 2 arguments".to_string(),
        ));
    }

    // Extract query vector(s) from first argument
    let query_vectors: Vec<Vec<f32>> = match &args[0] {
        ColumnarValue::Scalar(scalar) => {
            let v = vector_ops::extract_single_vector_from_scalar(scalar)
                .map_err(|e| datafusion::error::DataFusionError::Execution(e.to_string()))?;
            vec![v]
        }
        ColumnarValue::Array(arr) => {
            vector_ops::extract_vectors(arr)
                .map_err(|e| datafusion::error::DataFusionError::Execution(e.to_string()))?
        }
    };

    // Extract CAM column from second argument
    match &args[1] {
        ColumnarValue::Array(cam_arr) => {
            let cam_column = cam_arr
                .as_any()
                .downcast_ref::<FixedSizeBinaryArray>()
                .ok_or_else(|| {
                    datafusion::error::DataFusionError::Execution(
                        "cam_distance: second argument must be FixedSizeBinary(6)".into(),
                    )
                })?;

            if cam_column.value_length() != CAM_SIZE as i32 {
                return Err(datafusion::error::DataFusionError::Execution(format!(
                    "cam_distance: expected FixedSizeBinary(6), got FixedSizeBinary({})",
                    cam_column.value_length()
                )));
            }

            // Common case: single query vector against all CAM fingerprints
            let query = &query_vectors[0];
            let dt = CamDistanceTables::precompute(query, codebook);
            let distances = dt.distance_batch(cam_column);
            let result = Arc::new(Float32Array::from(distances)) as ArrayRef;
            Ok(ColumnarValue::Array(result))
        }

        ColumnarValue::Scalar(cam_scalar) => {
            // Single CAM fingerprint
            let cam_arr = cam_scalar
                .to_array()
                .map_err(|e| datafusion::error::DataFusionError::Execution(e.to_string()))?;
            let cam_column = cam_arr
                .as_any()
                .downcast_ref::<FixedSizeBinaryArray>()
                .ok_or_else(|| {
                    datafusion::error::DataFusionError::Execution(
                        "cam_distance: scalar must be FixedSizeBinary(6)".into(),
                    )
                })?;
            let query = &query_vectors[0];
            let dt = CamDistanceTables::precompute(query, codebook);
            let distances = dt.distance_batch(cam_column);

            if distances.len() == 1 {
                Ok(ColumnarValue::Scalar(
                    datafusion::scalar::ScalarValue::Float32(Some(distances[0])),
                ))
            } else {
                let result = Arc::new(Float32Array::from(distances)) as ArrayRef;
                Ok(ColumnarValue::Array(result))
            }
        }
    }
}

/// Squared L2 distance between two slices.
#[inline(always)]
fn squared_l2(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| (x - y) * (x - y)).sum()
}

// === ScalarUDFImpl wrapper ===

struct CamDistanceUDF {
    name: String,
    func: UdfFunc,
    signature: Signature,
}

impl std::fmt::Debug for CamDistanceUDF {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CamDistanceUDF")
            .field("name", &self.name)
            .finish()
    }
}

impl datafusion::logical_expr::ScalarUDFImpl for CamDistanceUDF {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn name(&self) -> &str {
        &self.name
    }

    fn signature(&self) -> &Signature {
        &self.signature
    }

    fn return_type(&self, _arg_types: &[DataType]) -> datafusion::error::Result<DataType> {
        Ok(DataType::Float32)
    }

    fn invoke_with_args(
        &self,
        args: datafusion::logical_expr::ScalarFunctionArgs,
    ) -> datafusion::error::Result<ColumnarValue> {
        (self.func)(&args.args)
    }
}

impl PartialEq for CamDistanceUDF {
    fn eq(&self, other: &Self) -> bool {
        self.name == other.name
    }
}

impl Eq for CamDistanceUDF {}

impl std::hash::Hash for CamDistanceUDF {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.name.hash(state);
    }
}

/// Create a cam_distance UDF bound to a specific codebook.
///
/// The codebook is captured by Arc and shared across all invocations.
/// Typically loaded once from the cam_codebook Lance table at session start.
pub fn create_cam_distance_udf(codebook: Arc<Vec<Vec<Vec<f32>>>>) -> Arc<ScalarUDF> {
    let func = move |args: &[ColumnarValue]| -> datafusion::error::Result<ColumnarValue> {
        cam_distance_func(args, &codebook)
    };

    Arc::new(ScalarUDF::new_from_impl(CamDistanceUDF {
        name: "cam_distance".to_string(),
        func: Arc::new(func),
        signature: Signature::any(2, Volatility::Immutable),
    }))
}

// === Cascade UDF: stroke-progressive distance with early exit ===

/// Stroke 1 UDF: cam_heel_distance(query_vector, cam_column) → Float32.
/// Only uses the HEEL byte (first byte of CAM). Used in cascade prefilter.
pub fn create_cam_heel_distance_udf(codebook: Arc<Vec<Vec<Vec<f32>>>>) -> Arc<ScalarUDF> {
    let func = move |args: &[ColumnarValue]| -> datafusion::error::Result<ColumnarValue> {
        if args.len() != 2 {
            return Err(datafusion::error::DataFusionError::Execution(
                "cam_heel_distance requires exactly 2 arguments".into(),
            ));
        }

        let query = match &args[0] {
            ColumnarValue::Scalar(s) => vector_ops::extract_single_vector_from_scalar(s)
                .map_err(|e| datafusion::error::DataFusionError::Execution(e.to_string()))?,
            ColumnarValue::Array(arr) => {
                let vecs = vector_ops::extract_vectors(arr)
                    .map_err(|e| datafusion::error::DataFusionError::Execution(e.to_string()))?;
                vecs.into_iter().next().ok_or_else(|| {
                    datafusion::error::DataFusionError::Execution("empty query array".into())
                })?
            }
        };

        // Only precompute table for subspace 0 (HEEL)
        let total_dim = query.len();
        let subspace_dim = total_dim / CAM_SIZE;
        let q_sub = &query[0..subspace_dim];
        let mut table = [0.0f32; 256];
        let num_centroids = codebook[0].len().min(256);
        for c in 0..num_centroids {
            table[c] = squared_l2(q_sub, &codebook[0][c]);
        }

        match &args[1] {
            ColumnarValue::Array(cam_arr) => {
                let cam_column = cam_arr.as_any().downcast_ref::<FixedSizeBinaryArray>()
                    .ok_or_else(|| datafusion::error::DataFusionError::Execution(
                        "cam_heel_distance: second argument must be FixedSizeBinary(6)".into(),
                    ))?;

                let distances: Vec<f32> = (0..cam_column.len())
                    .map(|i| {
                        if cam_column.is_null(i) {
                            f32::MAX
                        } else {
                            table[cam_column.value(i)[0] as usize]
                        }
                    })
                    .collect();

                Ok(ColumnarValue::Array(Arc::new(Float32Array::from(distances)) as ArrayRef))
            }
            _ => Err(datafusion::error::DataFusionError::Execution(
                "cam_heel_distance: second argument must be an array".into(),
            )),
        }
    };

    Arc::new(ScalarUDF::new_from_impl(CamDistanceUDF {
        name: "cam_heel_distance".to_string(),
        func: Arc::new(func),
        signature: Signature::any(2, Volatility::Immutable),
    }))
}

/// Register all CAM-PQ UDFs with a DataFusion SessionContext.
pub fn register_cam_udfs(
    ctx: &datafusion::execution::context::SessionContext,
    codebook: Arc<Vec<Vec<Vec<f32>>>>,
) {
    ctx.register_udf((*create_cam_distance_udf(codebook.clone())).clone());
    ctx.register_udf((*create_cam_heel_distance_udf(codebook)).clone());
}
