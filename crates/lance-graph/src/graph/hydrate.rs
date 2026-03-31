//! Hydrate bgz7 weight fingerprints into LanceDB for HHTL search.
//!
//! Reads bgz7 shards (Base17 fingerprints) and writes them as Arrow RecordBatches
//! for Lance Dataset storage with vector columns for HEEL/HIP/TWIG/LEAF cascade.

use arrow::array::{
    ArrayRef, FixedSizeListBuilder, Int16Builder, StringArray, UInt32Array, UInt8Array,
};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use std::sync::Arc;

/// Schema for the hydrated weight table.
pub fn weight_schema() -> Schema {
    Schema::new(vec![
        Field::new("tensor_name", DataType::Utf8, false),
        Field::new("row_idx", DataType::UInt32, false),
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
pub fn bgz7_to_batch(
    tensors: &[(String, Vec<ndarray::hpc::bgz17_bridge::Base17>)],
) -> RecordBatch {
    let schema = Arc::new(weight_schema());
    let mut names = Vec::new();
    let mut row_idxs = Vec::new();
    let mut base17_builder = FixedSizeListBuilder::new(Int16Builder::new(), 17);
    let mut total_rows = 0usize;

    for (name, rows) in tensors {
        for (r, fp) in rows.iter().enumerate() {
            names.push(name.clone());
            row_idxs.push(r as u32);
            for d in 0..17 {
                base17_builder.values().append_value(fp.dims[d]);
            }
            base17_builder.append(true);
            total_rows += 1;
        }
    }

    let name_array: ArrayRef = Arc::new(StringArray::from(names));
    let row_idx_array: ArrayRef = Arc::new(UInt32Array::from(row_idxs));
    let base17_array: ArrayRef = Arc::new(base17_builder.finish());
    let null_u8: ArrayRef = Arc::new(UInt8Array::from(vec![None::<u8>; total_rows]));

    // Let Arrow infer schema from columns instead of forcing it
    RecordBatch::try_from_iter(vec![
        ("tensor_name", name_array),
        ("row_idx", row_idx_array),
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

/// Compute HEEL vector: column-wise bundle of ALL BF16-hydrated rows.
pub fn compute_heel(batch: &RecordBatch) -> ndarray::hpc::bgz17_bridge::Base17 {
    let base17_col = batch.column_by_name("base17").expect("base17 column");
    let list_array = base17_col
        .as_any()
        .downcast_ref::<arrow::array::FixedSizeListArray>()
        .expect("FixedSizeList");
    let values = list_array
        .values()
        .as_any()
        .downcast_ref::<arrow::array::Int16Array>()
        .expect("Int16");

    let n_rows = batch.num_rows();
    let mut sums = [0i64; 17];
    for row in 0..n_rows {
        let offset = row * 17;
        for d in 0..17 {
            sums[d] += values.value(offset + d) as i64;
        }
    }
    let mut dims = [0i16; 17];
    if n_rows > 0 {
        for d in 0..17 { dims[d] = (sums[d] / n_rows as i64) as i16; }
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
        assert_eq!(schema.fields().len(), 6);
    }

    #[test]
    fn test_bgz7_to_batch() {
        let tensors = vec![
            ("layer.0.q_proj".into(), vec![Base17 { dims: [100; 17] }, Base17 { dims: [200; 17] }]),
            ("layer.0.k_proj".into(), vec![Base17 { dims: [-50; 17] }]),
        ];
        let batch = bgz7_to_batch(&tensors);
        assert_eq!(batch.num_rows(), 3);
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
    #[ignore = "requires /tmp/qwen35_27b_v2_shard02.bgz7"]
    fn test_hydrate_real() {
        let batch = hydrate_bgz7("/tmp/qwen35_27b_v2_shard02.bgz7").unwrap();
        eprintln!("Hydrated: {} rows", batch.num_rows());
        let heel = compute_heel(&batch);
        eprintln!("HEEL: {:?}", heel.dims);
    }
}
