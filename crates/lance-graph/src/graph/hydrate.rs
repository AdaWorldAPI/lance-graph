//! Hydrate bgz7 weight vectors into LanceDB.
//!
//! Base17 vectors (34 bytes, ρ=0.993 vs BF16) are stored as 17-dim f32
//! vector columns in Lance datasets. Lance handles indexing (IVF_PQ, RaBitQ)
//! and ANN search natively.
//!
//! Palette columns (palette_s/p/o) are kept for the SPO triple store path —
//! the bgz17 Palette→DistanceMatrix→SimilarityTable pipeline uses them for
//! O(1) precomputed distance lookups on millions of edges.

use arrow::array::{
    ArrayRef, FixedSizeListBuilder, Float32Builder, Int16Builder,
    StringArray, UInt32Array, UInt8Array,
};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use std::sync::Arc;

/// Schema for the hydrated weight table.
///
/// - `tensor_name`: which weight tensor (e.g. "model.layers.0.self_attn.q_proj")
/// - `row_idx`: row within tensor
/// - `vector`: 17-dim f32 for Lance vector search (i16→f32 is exact)
/// - `base17`: 17-dim i16 raw values (for direct L1, palette assignment)
/// - `palette_s/p/o`: SPO palette indices (populated later by palette pipeline)
pub fn weight_schema() -> Schema {
    Schema::new(vec![
        Field::new("tensor_name", DataType::Utf8, false),
        Field::new("row_idx", DataType::UInt32, false),
        Field::new(
            "vector",
            DataType::FixedSizeList(
                Arc::new(Field::new("item", DataType::Float32, false)),
                17,
            ),
            false,
        ),
        Field::new(
            "base17",
            DataType::FixedSizeList(
                Arc::new(Field::new("item", DataType::Int16, false)),
                17,
            ),
            false,
        ),
        Field::new("palette_s", DataType::UInt8, true),
        Field::new("palette_p", DataType::UInt8, true),
        Field::new("palette_o", DataType::UInt8, true),
    ])
}

/// Convert bgz7 compressed tensors to Arrow RecordBatch.
///
/// Stores both f32 (for Lance vector search) and i16 (for direct L1 / palette).
pub fn bgz7_to_batch(
    tensors: &[(String, Vec<ndarray::hpc::bgz17_bridge::Base17>)],
) -> RecordBatch {
    let mut names = Vec::new();
    let mut row_idxs = Vec::new();
    let mut vector_builder = FixedSizeListBuilder::new(Float32Builder::new(), 17);
    let mut base17_builder = FixedSizeListBuilder::new(Int16Builder::new(), 17);
    let mut total_rows = 0usize;

    for (name, rows) in tensors {
        for (r, fp) in rows.iter().enumerate() {
            names.push(name.clone());
            row_idxs.push(r as u32);
            for d in 0..17 {
                vector_builder.values().append_value(fp.dims[d] as f32);
                base17_builder.values().append_value(fp.dims[d]);
            }
            vector_builder.append(true);
            base17_builder.append(true);
            total_rows += 1;
        }
    }

    let name_array: ArrayRef = Arc::new(StringArray::from(names));
    let row_idx_array: ArrayRef = Arc::new(UInt32Array::from(row_idxs));
    let vector_array: ArrayRef = Arc::new(vector_builder.finish());
    let base17_array: ArrayRef = Arc::new(base17_builder.finish());
    let null_u8: ArrayRef = Arc::new(UInt8Array::from(vec![None::<u8>; total_rows]));

    RecordBatch::try_from_iter(vec![
        ("tensor_name", name_array),
        ("row_idx", row_idx_array),
        ("vector", vector_array),
        ("base17", base17_array),
        ("palette_s", null_u8.clone()),
        ("palette_p", null_u8.clone()),
        ("palette_o", null_u8),
    ])
    .expect("columns valid")
}

/// Load bgz7 file and convert to RecordBatch.
pub fn hydrate_bgz7(path: &str) -> Result<RecordBatch, String> {
    let compressed = ndarray::hpc::gguf_indexer::read_bgz7_file(path)?;
    let tensors: Vec<(String, Vec<ndarray::hpc::bgz17_bridge::Base17>)> = compressed
        .into_iter()
        .map(|ct| (ct.name, ct.rows))
        .collect();
    Ok(bgz7_to_batch(&tensors))
}

/// Write a RecordBatch to a Lance dataset.
pub async fn write_to_lance(
    batch: &RecordBatch,
    dataset_path: &str,
) -> Result<(), String> {
    use lance::dataset::{WriteMode, WriteParams};
    use lance::Dataset;

    let batches = vec![batch.clone()];
    let reader = arrow::record_batch::RecordBatchIterator::new(
        batches.into_iter().map(Ok),
        batch.schema(),
    );

    let params = WriteParams {
        mode: WriteMode::Append,
        ..Default::default()
    };

    Dataset::write(reader, dataset_path, Some(params))
        .await
        .map_err(|e| format!("Lance write error: {e}"))?;

    Ok(())
}

/// Hydrate bgz7 → Lance dataset in one call. Returns row count.
pub async fn hydrate_to_lance(bgz7_path: &str, dataset_path: &str) -> Result<usize, String> {
    let batch = hydrate_bgz7(bgz7_path)?;
    let n_rows = batch.num_rows();
    write_to_lance(&batch, dataset_path).await?;
    Ok(n_rows)
}

/// Compute HEEL: element-wise mean of all vectors (the gestalt).
pub fn compute_heel(batch: &RecordBatch) -> ndarray::hpc::bgz17_bridge::Base17 {
    let vector_col = batch.column_by_name("vector").expect("vector column");
    let list_array = vector_col
        .as_any()
        .downcast_ref::<arrow::array::FixedSizeListArray>()
        .expect("FixedSizeList");
    let values = list_array
        .values()
        .as_any()
        .downcast_ref::<arrow::array::Float32Array>()
        .expect("Float32");

    let n_rows = batch.num_rows();
    let mut sums = [0.0f64; 17];
    for row in 0..n_rows {
        let offset = row * 17;
        for d in 0..17 {
            sums[d] += values.value(offset + d) as f64;
        }
    }
    let mut dims = [0i16; 17];
    if n_rows > 0 {
        for d in 0..17 {
            dims[d] = (sums[d] / n_rows as f64).round() as i16;
        }
    }
    ndarray::hpc::bgz17_bridge::Base17 { dims }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::hpc::bgz17_bridge::Base17;

    #[test]
    fn test_weight_schema() {
        let schema = weight_schema();
        assert_eq!(schema.fields().len(), 7);
    }

    #[test]
    fn test_bgz7_to_batch() {
        let tensors = vec![
            ("layer.0.q_proj".into(), vec![Base17 { dims: [100; 17] }, Base17 { dims: [200; 17] }]),
            ("layer.0.k_proj".into(), vec![Base17 { dims: [-50; 17] }]),
        ];
        let batch = bgz7_to_batch(&tensors);
        assert_eq!(batch.num_rows(), 3);
        assert_eq!(batch.num_columns(), 7);
    }

    #[test]
    fn test_compute_heel() {
        let tensors = vec![("t".into(), vec![
            Base17 { dims: [10; 17] }, Base17 { dims: [20; 17] }, Base17 { dims: [30; 17] },
        ])];
        let batch = bgz7_to_batch(&tensors);
        let heel = compute_heel(&batch);
        assert_eq!(heel.dims[0], 20);
    }

    #[test]
    fn test_heel_asymmetric() {
        let tensors = vec![("t".into(), vec![
            Base17 { dims: [100, 200, -50, 300, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] },
            Base17 { dims: [150, 100, -30, 250, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] },
        ])];
        let batch = bgz7_to_batch(&tensors);
        let heel = compute_heel(&batch);
        assert_eq!(heel.dims[0], 125);
        assert_ne!(heel.dims[0], 0);
    }

    #[test]
    fn test_f32_preserves_i16() {
        let tensors = vec![("t".into(), vec![
            Base17 { dims: [-32768, 32767, 0, 1, -1, 12345, -12345, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] },
        ])];
        let batch = bgz7_to_batch(&tensors);
        let heel = compute_heel(&batch);
        assert_eq!(heel.dims[0], -32768);
        assert_eq!(heel.dims[1], 32767);
        assert_eq!(heel.dims[5], 12345);
    }

    #[test]
    fn test_both_columns_present() {
        let tensors = vec![("t".into(), vec![Base17 { dims: [42; 17] }])];
        let batch = bgz7_to_batch(&tensors);
        assert!(batch.column_by_name("vector").is_some());
        assert!(batch.column_by_name("base17").is_some());
        assert!(batch.column_by_name("palette_s").is_some());
    }

    #[test]
    #[ignore = "requires /tmp/qwen35_27b_v2_shard02.bgz7"]
    fn test_hydrate_real() {
        let batch = hydrate_bgz7("/tmp/qwen35_27b_v2_shard02.bgz7").unwrap();
        eprintln!("Hydrated: {} rows, {} cols", batch.num_rows(), batch.num_columns());
        let heel = compute_heel(&batch);
        eprintln!("HEEL: {:?}", heel.dims);
    }
}
