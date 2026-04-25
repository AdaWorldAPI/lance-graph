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
    StringArray, UInt16Array, UInt32Array, UInt8Array,
};
use arrow::record_batch::RecordBatch;
use std::sync::Arc;

/// Functional partition of a weight tensor.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum TensorRole {
    QProj,     // "how this layer queries"
    KProj,     // "what this layer matches"
    VProj,     // "what this layer retrieves"
    OProj,     // "how this layer outputs attention"
    GateProj,  // "what this layer gates"
    UpProj,    // "what this layer amplifies"
    DownProj,  // "what this layer compresses"
    Embedding, // "vocabulary → hidden"
    Norm,      // "scale/bias"
    Other,     // unclassified
}

impl TensorRole {
    /// Parse tensor role from the full tensor name string.
    /// Works with both HuggingFace and GGUF naming conventions.
    pub fn from_name(name: &str) -> Self {
        let n = name.to_lowercase();
        if n.contains("q_proj") || n.contains("attn_q") || n.contains(".wq.") { TensorRole::QProj }
        else if n.contains("k_proj") || n.contains("attn_k") || n.contains(".wk.") { TensorRole::KProj }
        else if n.contains("v_proj") || n.contains("attn_v") || n.contains(".wv.") { TensorRole::VProj }
        else if n.contains("o_proj") || n.contains("attn_output") || n.contains(".wo.") { TensorRole::OProj }
        else if n.contains("gate_proj") || n.contains("ffn_gate") || n.contains(".w1.") { TensorRole::GateProj }
        else if n.contains("up_proj") || n.contains("ffn_up") || n.contains(".w3.") { TensorRole::UpProj }
        else if n.contains("down_proj") || n.contains("ffn_down") || n.contains(".w2.") { TensorRole::DownProj }
        else if n.contains("embed") || n.contains("token_embd") { TensorRole::Embedding }
        else if n.contains("norm") || n.contains("ln_") { TensorRole::Norm }
        else { TensorRole::Other }
    }

    /// Numeric ID for Arrow column storage.
    pub fn as_u8(&self) -> u8 {
        match self {
            TensorRole::QProj => 0,
            TensorRole::KProj => 1,
            TensorRole::VProj => 2,
            TensorRole::OProj => 3,
            TensorRole::GateProj => 4,
            TensorRole::UpProj => 5,
            TensorRole::DownProj => 6,
            TensorRole::Embedding => 7,
            TensorRole::Norm => 8,
            TensorRole::Other => 9,
        }
    }

    pub fn label(&self) -> &'static str {
        match self {
            TensorRole::QProj => "q_proj",
            TensorRole::KProj => "k_proj",
            TensorRole::VProj => "v_proj",
            TensorRole::OProj => "o_proj",
            TensorRole::GateProj => "gate_proj",
            TensorRole::UpProj => "up_proj",
            TensorRole::DownProj => "down_proj",
            TensorRole::Embedding => "embed",
            TensorRole::Norm => "norm",
            TensorRole::Other => "other",
        }
    }
}

/// Extract layer index from tensor name. Returns None for non-layer tensors.
pub fn parse_layer_idx(name: &str) -> Option<u16> {
    // Match "layers.N." or "blk.N."
    let n = name.to_lowercase();
    if let Some(pos) = n.find("layers.") {
        let rest = &n[pos + 7..];
        rest.split('.').next().and_then(|s| s.parse().ok())
    } else if let Some(pos) = n.find("blk.") {
        let rest = &n[pos + 4..];
        rest.split('.').next().and_then(|s| s.parse().ok())
    } else {
        None
    }
}

/// Convert bgz7 compressed tensors to Arrow RecordBatch with partition columns.
///
/// Each row gets `layer_idx` and `tensor_role` parsed from the tensor name.
/// This enables partitioned CAM indexing: per-role palettes, per-layer search.
pub fn bgz7_to_batch(
    tensors: &[(String, Vec<ndarray::hpc::bgz17_bridge::Base17>)],
) -> RecordBatch {
    let mut names = Vec::new();
    let mut row_idxs = Vec::new();
    let mut layer_idxs: Vec<Option<u16>> = Vec::new();
    let mut roles = Vec::new();
    let mut vector_builder = FixedSizeListBuilder::new(Float32Builder::new(), 17);
    let mut base17_builder = FixedSizeListBuilder::new(Int16Builder::new(), 17);
    let mut total_rows = 0usize;

    for (name, rows) in tensors {
        let role = TensorRole::from_name(name);
        let layer = parse_layer_idx(name);

        for (r, fp) in rows.iter().enumerate() {
            names.push(name.clone());
            row_idxs.push(r as u32);
            layer_idxs.push(layer);
            roles.push(role.as_u8());
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
    let layer_idx_array: ArrayRef = Arc::new(UInt16Array::from(layer_idxs));
    let role_array: ArrayRef = Arc::new(UInt8Array::from(roles));
    let vector_array: ArrayRef = Arc::new(vector_builder.finish());
    let base17_array: ArrayRef = Arc::new(base17_builder.finish());
    let null_u8: ArrayRef = Arc::new(UInt8Array::from(vec![None::<u8>; total_rows]));

    RecordBatch::try_from_iter(vec![
        ("tensor_name", name_array),
        ("row_idx", row_idx_array),
        ("layer_idx", layer_idx_array),
        ("tensor_role", role_array),
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
        for (d, sum) in sums.iter_mut().enumerate() {
            *sum += values.value(offset + d) as f64;
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
    fn test_bgz7_to_batch() {
        let tensors = vec![
            ("model.layers.0.self_attn.q_proj.weight".into(), vec![Base17 { dims: [100; 17] }, Base17 { dims: [200; 17] }]),
            ("model.layers.0.self_attn.k_proj.weight".into(), vec![Base17 { dims: [-50; 17] }]),
        ];
        let batch = bgz7_to_batch(&tensors);
        assert_eq!(batch.num_rows(), 3);
        assert_eq!(batch.num_columns(), 9);
    }

    #[test]
    fn test_tensor_role_parsing() {
        assert_eq!(TensorRole::from_name("model.layers.0.self_attn.q_proj.weight"), TensorRole::QProj);
        assert_eq!(TensorRole::from_name("model.layers.0.self_attn.k_proj.weight"), TensorRole::KProj);
        assert_eq!(TensorRole::from_name("model.layers.0.self_attn.v_proj.weight"), TensorRole::VProj);
        assert_eq!(TensorRole::from_name("model.layers.0.self_attn.o_proj.weight"), TensorRole::OProj);
        assert_eq!(TensorRole::from_name("model.layers.0.mlp.gate_proj.weight"), TensorRole::GateProj);
        assert_eq!(TensorRole::from_name("model.layers.0.mlp.up_proj.weight"), TensorRole::UpProj);
        assert_eq!(TensorRole::from_name("model.layers.0.mlp.down_proj.weight"), TensorRole::DownProj);
        assert_eq!(TensorRole::from_name("model.embed_tokens.weight"), TensorRole::Embedding);
        assert_eq!(TensorRole::from_name("model.layers.0.input_layernorm.weight"), TensorRole::Norm);
        // GGUF naming
        assert_eq!(TensorRole::from_name("blk.5.attn_q.weight"), TensorRole::QProj);
        assert_eq!(TensorRole::from_name("blk.5.ffn_gate.weight"), TensorRole::GateProj);
    }

    #[test]
    fn test_layer_idx_parsing() {
        assert_eq!(parse_layer_idx("model.layers.15.self_attn.q_proj.weight"), Some(15));
        assert_eq!(parse_layer_idx("blk.7.attn_q.weight"), Some(7));
        assert_eq!(parse_layer_idx("model.embed_tokens.weight"), None);
        assert_eq!(parse_layer_idx("model.layers.0.mlp.gate_proj.weight"), Some(0));
    }

    #[test]
    fn test_partition_columns_populated() {
        let tensors = vec![
            ("model.layers.5.self_attn.q_proj.weight".into(), vec![Base17 { dims: [100; 17] }]),
            ("model.layers.5.mlp.gate_proj.weight".into(), vec![Base17 { dims: [200; 17] }]),
            ("model.embed_tokens.weight".into(), vec![Base17 { dims: [50; 17] }]),
        ];
        let batch = bgz7_to_batch(&tensors);
        let roles = batch.column_by_name("tensor_role").unwrap();
        let role_arr = roles.as_any().downcast_ref::<UInt8Array>().unwrap();
        assert_eq!(role_arr.value(0), TensorRole::QProj.as_u8());
        assert_eq!(role_arr.value(1), TensorRole::GateProj.as_u8());
        assert_eq!(role_arr.value(2), TensorRole::Embedding.as_u8());

        let layers = batch.column_by_name("layer_idx").unwrap();
        let layer_arr = layers.as_any().downcast_ref::<UInt16Array>().unwrap();
        assert_eq!(layer_arr.value(0), 5);
        assert_eq!(layer_arr.value(1), 5);
        // First two have layer 5, third (embed) has no layer
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
    fn test_all_columns_present() {
        let tensors = vec![("model.layers.0.self_attn.q_proj.weight".into(), vec![Base17 { dims: [42; 17] }])];
        let batch = bgz7_to_batch(&tensors);
        assert!(batch.column_by_name("vector").is_some());
        assert!(batch.column_by_name("base17").is_some());
        assert!(batch.column_by_name("layer_idx").is_some());
        assert!(batch.column_by_name("tensor_role").is_some());
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
