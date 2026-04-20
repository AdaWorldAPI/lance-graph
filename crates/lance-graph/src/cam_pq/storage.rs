//! Lance storage schema for CAM-PQ fingerprints and codebooks.
//!
//! # Tables
//!
//! ## `vectors` — Main data table
//! ```sql
//! CREATE TABLE vectors (
//!     id          BIGINT PRIMARY KEY,
//!     cam         FIXED_SIZE_BINARY(6),     -- 48-bit CAM fingerprint
//!     metadata    VARCHAR,
//!     timestamp   TIMESTAMP
//! );
//! ```
//!
//! ## `cam_codebook` — Codebook table (trained once, immutable)
//! ```sql
//! CREATE TABLE cam_codebook (
//!     subspace    TINYINT,                   -- 0-5 (HEEL through GAMMA)
//!     centroid_id TINYINT UNSIGNED,           -- 0-255
//!     vector      FIXED_SIZE_LIST(FLOAT, N),  -- centroid vector (D/6 dims)
//!     label       VARCHAR                     -- semantic label (CLAM mode)
//! );
//! ```

use arrow::datatypes::{DataType, Field, Schema};
use arrow_array::{
    builder::FixedSizeBinaryBuilder, Array, FixedSizeBinaryArray, Float32Array,
    Int64Array, RecordBatch, UInt8Array,
};
use arrow_array::builder::Float32Builder;
use std::sync::Arc;

/// CAM column name in Lance tables.
pub const CAM_COLUMN: &str = "cam";

/// CAM fingerprint size in bytes.
pub const CAM_SIZE: usize = 6;

/// Codebook table name.
pub const CODEBOOK_TABLE: &str = "cam_codebook";

/// Schema for the CAM vectors table.
pub fn cam_vectors_schema() -> Schema {
    Schema::new(vec![
        Field::new("id", DataType::Int64, false),
        Field::new("cam", DataType::FixedSizeBinary(CAM_SIZE as i32), false),
    ])
}

/// Schema for the codebook table.
pub fn cam_codebook_schema(subspace_dim: usize) -> Schema {
    Schema::new(vec![
        Field::new("subspace", DataType::UInt8, false),
        Field::new("centroid_id", DataType::UInt8, false),
        Field::new(
            "vector",
            DataType::FixedSizeList(
                Arc::new(Field::new("item", DataType::Float32, false)),
                subspace_dim as i32,
            ),
            false,
        ),
    ])
}

/// Build a RecordBatch of CAM fingerprints for writing to Lance.
///
/// `ids` and `cams` must have the same length.
/// Each `cams[i]` is a 6-byte CAM fingerprint.
pub fn build_cam_batch(ids: &[i64], cams: &[[u8; CAM_SIZE]]) -> Result<RecordBatch, arrow::error::ArrowError> {
    assert_eq!(ids.len(), cams.len());

    let schema = Arc::new(cam_vectors_schema());
    let n = ids.len();

    let id_array = Int64Array::from(ids.to_vec());
    let mut cam_builder = FixedSizeBinaryBuilder::with_capacity(n, CAM_SIZE as i32);
    for cam in cams {
        cam_builder.append_value(cam)?;
    }

    RecordBatch::try_new(
        schema,
        vec![
            Arc::new(id_array),
            Arc::new(cam_builder.finish()),
        ],
    )
}

/// Build a RecordBatch for the codebook table.
///
/// `codebook[subspace][centroid]` = centroid vector (f32 slice of length subspace_dim).
pub fn build_codebook_batch(
    codebook: &[Vec<Vec<f32>>],
    subspace_dim: usize,
) -> Result<RecordBatch, arrow::error::ArrowError> {
    let num_subspaces = codebook.len();
    let total_rows: usize = codebook.iter().map(|s| s.len()).sum();
    let schema = Arc::new(cam_codebook_schema(subspace_dim));

    let mut subspace_ids = Vec::with_capacity(total_rows);
    let mut centroid_ids = Vec::with_capacity(total_rows);
    let mut float_builder = Float32Builder::with_capacity(total_rows * subspace_dim);

    for s in 0..num_subspaces {
        for (c, centroid) in codebook[s].iter().enumerate() {
            subspace_ids.push(s as u8);
            centroid_ids.push(c as u8);
            for &val in centroid {
                float_builder.append_value(val);
            }
        }
    }

    let vector_array = arrow_array::builder::FixedSizeListBuilder::new(
        Float32Builder::with_capacity(total_rows * subspace_dim),
        subspace_dim as i32,
    );
    // Build the FixedSizeList manually via values + offsets
    let flat_values: Vec<f32> = codebook.iter()
        .flat_map(|s| s.iter().flat_map(|c| c.iter().copied()))
        .collect();
    let values_array = Float32Array::from(flat_values);
    let list_array = arrow_array::FixedSizeListArray::try_new(
        Arc::new(Field::new("item", DataType::Float32, false)),
        subspace_dim as i32,
        Arc::new(values_array),
        None,
    )?;

    // Drop the unused builder
    drop(vector_array);

    RecordBatch::try_new(
        schema,
        vec![
            Arc::new(UInt8Array::from(subspace_ids)),
            Arc::new(UInt8Array::from(centroid_ids)),
            Arc::new(list_array),
        ],
    )
}

/// Extract CAM fingerprints from a RecordBatch (read path).
///
/// Reads the "cam" column and returns Vec of 6-byte arrays.
pub fn extract_cam_fingerprints(batch: &RecordBatch) -> Vec<[u8; CAM_SIZE]> {
    let cam_col = batch
        .column_by_name(CAM_COLUMN)
        .expect("cam column not found")
        .as_any()
        .downcast_ref::<FixedSizeBinaryArray>()
        .expect("cam column must be FixedSizeBinary(6)");

    (0..cam_col.len())
        .map(|i| {
            let bytes = cam_col.value(i);
            [bytes[0], bytes[1], bytes[2], bytes[3], bytes[4], bytes[5]]
        })
        .collect()
}

/// Extract codebook from a codebook RecordBatch (read path).
///
/// Returns `codebook[subspace][centroid]` = Vec<f32>.
pub fn extract_codebook(batch: &RecordBatch, num_subspaces: usize) -> Vec<Vec<Vec<f32>>> {
    let subspace_col = batch
        .column_by_name("subspace")
        .expect("subspace column")
        .as_any()
        .downcast_ref::<UInt8Array>()
        .expect("subspace must be UInt8");
    let centroid_col = batch
        .column_by_name("centroid_id")
        .expect("centroid_id column")
        .as_any()
        .downcast_ref::<UInt8Array>()
        .expect("centroid_id must be UInt8");
    let vector_col = batch
        .column_by_name("vector")
        .expect("vector column")
        .as_any()
        .downcast_ref::<arrow_array::FixedSizeListArray>()
        .expect("vector must be FixedSizeList");

    let mut codebook = vec![vec![]; num_subspaces];

    for row in 0..batch.num_rows() {
        let s = subspace_col.value(row) as usize;
        let _c = centroid_col.value(row) as usize;
        let list_ref = vector_col.value(row);
        let values = list_ref
            .as_any()
            .downcast_ref::<Float32Array>()
            .expect("centroid values must be Float32");
        let centroid: Vec<f32> = (0..values.len()).map(|i| values.value(i)).collect();
        codebook[s].push(centroid);
    }

    codebook
}

/// Convert a trained codebook + encoded fingerprints into Arrow RecordBatches
/// ready for Lance persistence.
///
/// This is the canonical bridge from ndarray's `CamCodebook` training output
/// to the lance-graph storage layer. Use this instead of the raw CMPQ/CMFP
/// binary format (`bgz-tensor/src/bin/cam_pq_calibrate.rs`).
///
/// Returns `(vectors_batch, codebook_batch)`.
pub fn codebook_to_lance(
    codebook: &[Vec<Vec<f32>>],
    fingerprints: &[[u8; CAM_SIZE]],
) -> Result<(RecordBatch, RecordBatch), arrow::error::ArrowError> {
    let ids: Vec<i64> = (0..fingerprints.len() as i64).collect();
    let vectors = build_cam_batch(&ids, fingerprints)?;
    let subspace_dim = if codebook.is_empty() || codebook[0].is_empty() {
        0
    } else {
        codebook[0][0].len()
    };
    let cb = build_codebook_batch(codebook, subspace_dim)?;
    Ok((vectors, cb))
}

/// Storage statistics for a CAM-PQ dataset.
#[derive(Debug, Clone)]
pub struct CamStorageStats {
    pub num_vectors: u64,
    pub cam_bytes: u64,
    pub codebook_bytes: u64,
    pub compression_ratio: f64,
}

impl CamStorageStats {
    pub fn compute(num_vectors: u64, total_dim: usize) -> Self {
        let cam_bytes = num_vectors * CAM_SIZE as u64;
        let subspace_dim = total_dim / 6;
        let codebook_bytes = (6 * 256 * subspace_dim * 4) as u64;
        let raw_bytes = num_vectors * total_dim as u64 * 4;
        let compressed = cam_bytes + codebook_bytes;
        let compression_ratio = if compressed > 0 {
            raw_bytes as f64 / compressed as f64
        } else {
            0.0
        };
        CamStorageStats { num_vectors, cam_bytes, codebook_bytes, compression_ratio }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_build_cam_batch() {
        let ids: Vec<i64> = (0..100).collect();
        let cams: Vec<[u8; 6]> = (0..100u8).map(|i| [i, i + 1, i + 2, i + 3, i + 4, i + 5]).collect();

        let batch = build_cam_batch(&ids, &cams).unwrap();
        assert_eq!(batch.num_rows(), 100);
        assert_eq!(batch.num_columns(), 2);

        // Read back
        let extracted = extract_cam_fingerprints(&batch);
        assert_eq!(extracted.len(), 100);
        assert_eq!(extracted[0], [0, 1, 2, 3, 4, 5]);
        assert_eq!(extracted[99], [99, 100, 101, 102, 103, 104]);
    }

    #[test]
    fn test_build_codebook_batch() {
        // 6 subspaces × 4 centroids × dim 2
        let codebook: Vec<Vec<Vec<f32>>> = (0..6)
            .map(|s| {
                (0..4)
                    .map(|c| vec![s as f32 * 10.0 + c as f32, s as f32 * 10.0 + c as f32 + 0.5])
                    .collect()
            })
            .collect();

        let batch = build_codebook_batch(&codebook, 2).unwrap();
        assert_eq!(batch.num_rows(), 24); // 6 × 4

        let extracted = extract_codebook(&batch, 6);
        assert_eq!(extracted.len(), 6);
        assert_eq!(extracted[0].len(), 4);
        assert_eq!(extracted[0][0], vec![0.0, 0.5]);
        assert_eq!(extracted[5][3], vec![53.0, 53.5]);
    }

    #[test]
    fn test_storage_stats_1m_256d() {
        let stats = CamStorageStats::compute(1_000_000, 256);
        assert_eq!(stats.cam_bytes, 6_000_000);
        assert!(stats.compression_ratio > 100.0);
    }

    #[test]
    fn test_storage_stats_1b_1024d() {
        let stats = CamStorageStats::compute(1_000_000_000, 1024);
        assert_eq!(stats.cam_bytes, 6_000_000_000);
        assert!(stats.compression_ratio > 500.0);
    }

    #[test]
    fn test_schema_creation() {
        let vectors_schema = cam_vectors_schema();
        assert_eq!(vectors_schema.fields().len(), 2);
        assert_eq!(vectors_schema.field(1).data_type(), &DataType::FixedSizeBinary(6));

        let codebook_schema = cam_codebook_schema(170);
        assert_eq!(codebook_schema.fields().len(), 3);
    }
}
