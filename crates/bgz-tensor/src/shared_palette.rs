//! Shared palette strategy for BGZ-HHTL-D encoding.
//!
//! Instead of one palette per tensor (280 palettes for 1.7B model = 57 MB overhead),
//! group tensors by (shape, role) and share a single palette per group.
//!
//! Example: 28 talker gate_proj tensors (all [6144, 2048] BF16) share one 256-entry
//! palette. Same distribution, same role, same invariants.
//!
//! Result: 26 shared palettes instead of 280 → 5.4 MB overhead instead of 57 MB.
//!
//! Full compression: 3.86 GB → 11.2 MB (343:1, 0.29% of original).
//! Fits in 75 MB total RAM with 512-token KV cache.
//!
//! ```text
//! Qwen3-TTS-1.7B Shared Palette Groups:
//!
//! Group                           Tensors  Rows/each  Total Rows  1 Palette
//! ──────────────────────────────  ───────  ─────────  ──────────  ─────────
//! talker/gate [6144,2048]              28      6,144     172,032   ~206 KB
//! talker/up   [6144,2048]              28      6,144     172,032   ~206 KB
//! talker/down [2048,6144]              28      2,048      57,344   ~206 KB
//! talker/qko  [2048,2048]              56      2,048     114,688   ~206 KB
//! talker/qko  [1024,2048]              28      1,024      28,672   ~206 KB
//! talker/v    [1024,2048]              28      1,024      28,672   ~206 KB
//! talker/embed [151936,2048]            1    151,936     151,936   ~206 KB
//! cp/embed    [2048,2048]              15      2,048      30,720   ~206 KB
//! cp/lm_head  [2048,1024]              15      2,048      30,720   ~206 KB
//! ... (17 more groups)
//! ──────────────────────────────
//! Total: 26 groups, 869,760 rows, 26 palettes = 5.4 MB
//! Entries: 869,760 × 4 = 3.5 MB
//! Passthrough: 2.4 MB (norms, biases, small conv kernels)
//! TOTAL: 11.2 MB
//! ```

use crate::fisher_z::FisherZTable;
use crate::hhtl_d::{HhtlDTensor, HhtlDEntry, HeelBasin, build_hip_families};
use crate::hhtl_cache::HhtlCache;
use crate::palette::WeightPalette;
use crate::projection::Base17;
use std::collections::HashMap;

/// Key for grouping tensors that share a palette.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct PaletteGroupKey {
    /// Component (talker, code_predictor, speaker_encoder).
    pub component: String,
    /// Role (qko, v, gate, up, down, embed, lm_head, projection, other).
    pub role: String,
    /// Shape as (rows, cols). For 3D tensors with dim[2]=1, use (dim[0], dim[1]).
    pub shape: (usize, usize),
}

/// A shared palette group: one palette, many tensors.
#[derive(Clone, Debug)]
pub struct SharedPaletteGroup {
    pub key: PaletteGroupKey,
    /// Names of tensors in this group.
    pub tensor_names: Vec<String>,
    /// Shared HHTL cache (palette + distance + route tables).
    pub cache: HhtlCache,
    /// 16-way HIP family assignments (shared).
    pub hip_families: Vec<u8>,
    /// Per-tensor HHTL-D entries.
    pub tensor_entries: Vec<(String, Vec<HhtlDEntry>)>,
    /// Fisher z i8 pairwise cosine table for this group.
    /// k×k i8 values + 8 bytes family gamma. Shared across all tensors in group.
    pub fisher_z: Option<FisherZTable>,
}

/// Classify a tensor name into a palette group role.
pub fn classify_role(name: &str) -> &'static str {
    let n = name.to_lowercase();
    if n.contains("q_proj") || n.contains("k_proj") || n.contains("o_proj") { "qko" }
    else if n.contains("v_proj") { "v" }
    else if n.contains("gate_proj") { "gate" }
    else if n.contains("up_proj") { "up" }
    else if n.contains("down_proj") { "down" }
    else if n.contains("embed") { "embed" }
    else if n.contains("lm_head") { "lm_head" }
    else if n.contains("projection") || n.contains("codec_head") { "projection" }
    else { "other" }
}

/// Classify a tensor name into component.
pub fn classify_component(name: &str) -> &'static str {
    if name.contains("code_predictor") { "code_predictor" }
    else if name.contains("talker") { "talker" }
    else if name.contains("speaker_encoder") { "speaker_encoder" }
    else { "other" }
}

/// Determine if a tensor should be HHTL-D encoded (extended strategy).
///
/// Extended: any 2D weight tensor with ≥ 512 rows and > 100 KB.
/// Also handles 3D tensors with dim[2]=1 (conv with kernel_size=1).
pub fn is_encodable(shape: &[usize], size_bytes: usize) -> bool {
    if shape.len() == 2 && shape[0] >= 512 && size_bytes > 100_000 {
        return true;
    }
    if shape.len() == 3 && shape[2] == 1 && shape[0] >= 512 && size_bytes > 100_000 {
        return true;
    }
    false
}

/// Get effective 2D shape for palette grouping.
pub fn effective_shape(shape: &[usize]) -> (usize, usize) {
    match shape.len() {
        2 => (shape[0], shape[1]),
        3 if shape[2] == 1 => (shape[0], shape[1]),
        _ => (shape[0], shape.iter().skip(1).product()),
    }
}

/// Build a shared palette from Base17 rows sampled across multiple tensors.
///
/// Samples up to `max_sample` rows from each tensor's Base17 projection,
/// builds one CLAM palette for the entire group.
pub fn build_shared_palette(
    group_rows: &[Vec<Base17>],
    k: usize,
    max_sample_per_tensor: usize,
) -> WeightPalette {
    let mut combined: Vec<&Base17> = Vec::new();
    for rows in group_rows {
        let take = rows.len().min(max_sample_per_tensor);
        combined.extend(rows[..take].iter());
    }

    // Build from combined sample
    let owned: Vec<Base17> = combined.iter().map(|&&ref b| b.clone()).collect();
    let sample = if owned.len() > 4096 { &owned[..4096] } else { &owned[..] };
    WeightPalette::build(sample, k)
}

/// Encode a full group of tensors under a shared palette.
pub fn encode_group(
    key: &PaletteGroupKey,
    tensor_names: &[String],
    tensor_rows_f32: &[Vec<Vec<f32>>],
    cache: &HhtlCache,
    hip_families: &[u8],
) -> Vec<(String, Vec<HhtlDEntry>)> {
    let mut results = Vec::with_capacity(tensor_names.len());

    for (name, rows) in tensor_names.iter().zip(tensor_rows_f32.iter()) {
        let role_name = format!("{}_{}", key.component, key.role);
        let hhtld = HhtlDTensor::encode(&role_name, rows, cache, hip_families);
        results.push((name.clone(), hhtld.entries));
    }

    results
}

/// Build a complete shared palette group with Fisher z table.
///
/// This is the main entry point for encoding a group of tensors.
/// Collects representative f32 rows from the FIRST tensor's centroid
/// assignments and builds the Fisher z table from those representatives.
pub fn build_group_with_fisher_z(
    key: &PaletteGroupKey,
    tensor_names: &[String],
    tensor_rows_f32: &[Vec<Vec<f32>>],
    k: usize,
) -> SharedPaletteGroup {
    // Build Base17 projections from first tensor for palette
    let first_rows = &tensor_rows_f32[0];
    let base17_rows: Vec<Base17> = first_rows.iter()
        .map(|r| Base17::from_f32(r))
        .collect();
    let palette = WeightPalette::build(&base17_rows, k);
    let cache = HhtlCache::from_palette(palette.clone());
    let hip_families = build_hip_families(&palette.entries);

    // Collect representative f32 rows: one per centroid (nearest to centroid)
    let n_centroids = palette.entries.len();
    let mut reps: Vec<Vec<f32>> = vec![Vec::new(); n_centroids];
    let mut rep_dists: Vec<u32> = vec![u32::MAX; n_centroids];
    for (i, row) in first_rows.iter().enumerate() {
        if i >= base17_rows.len() { break; }
        let (ci, dist) = cache.nearest(&base17_rows[i]);
        let ci = ci as usize;
        if ci < n_centroids && dist < rep_dists[ci] {
            reps[ci] = row.clone();
            rep_dists[ci] = dist;
        }
    }
    for ci in 0..n_centroids {
        if reps[ci].is_empty() {
            reps[ci] = palette.entries[ci].to_f32(key.shape.1);
        }
    }

    let fisher_z = Some(FisherZTable::build(&reps, n_centroids));

    // Encode all tensors
    let entries = encode_group(key, tensor_names, tensor_rows_f32, &cache, &hip_families);

    SharedPaletteGroup {
        key: key.clone(),
        tensor_names: tensor_names.to_vec(),
        cache,
        hip_families,
        tensor_entries: entries,
        fisher_z,
    }
}

/// Compression statistics for a shared palette encoding.
#[derive(Clone, Debug)]
pub struct SharedCompressionStats {
    pub n_groups: usize,
    pub n_tensors: usize,
    pub total_rows: usize,
    pub original_bytes: usize,
    pub entries_bytes: usize,
    pub palette_bytes: usize,
    pub passthrough_bytes: usize,
    pub total_output_bytes: usize,
}

impl SharedCompressionStats {
    pub fn compression_ratio(&self) -> f64 {
        self.original_bytes as f64 / self.total_output_bytes.max(1) as f64
    }

    pub fn weight_ratio(&self) -> f64 {
        (self.original_bytes - self.passthrough_bytes) as f64 / self.entries_bytes.max(1) as f64
    }

    pub fn summary(&self) -> String {
        format!(
            "Groups: {}, Tensors: {}, Rows: {}, Original: {:.1} MB → Output: {:.1} MB ({:.0}:1)",
            self.n_groups,
            self.n_tensors,
            self.total_rows,
            self.original_bytes as f64 / 1e6,
            self.total_output_bytes as f64 / 1e6,
            self.compression_ratio()
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn classify_roles_correct() {
        assert_eq!(classify_role("talker.layers.0.self_attn.q_proj.weight"), "qko");
        assert_eq!(classify_role("talker.layers.0.self_attn.v_proj.weight"), "v");
        assert_eq!(classify_role("talker.layers.0.mlp.gate_proj.weight"), "gate");
        assert_eq!(classify_role("talker.model.text_embedding.weight"), "embed");
        assert_eq!(classify_role("talker.code_predictor.lm_head.0.weight"), "lm_head");
        assert_eq!(classify_role("talker.layers.0.input_layernorm.weight"), "other");
    }

    #[test]
    fn is_encodable_correct() {
        assert!(is_encodable(&[2048, 2048], 8_000_000));
        assert!(is_encodable(&[512, 128, 1], 200_000));
        assert!(!is_encodable(&[256, 2048], 1_000_000)); // too few rows
        assert!(!is_encodable(&[2048], 4_000)); // 1D
        assert!(!is_encodable(&[1024, 32], 50_000)); // too small bytes
    }

    #[test]
    fn effective_shape_correct() {
        assert_eq!(effective_shape(&[2048, 2048]), (2048, 2048));
        assert_eq!(effective_shape(&[512, 128, 1]), (512, 128));
        assert_eq!(effective_shape(&[64, 64, 3]), (64, 192));
    }

    #[test]
    fn palette_group_key_eq() {
        let a = PaletteGroupKey {
            component: "talker".into(),
            role: "gate".into(),
            shape: (6144, 2048),
        };
        let b = a.clone();
        assert_eq!(a, b);

        let mut map = HashMap::new();
        map.insert(a.clone(), 42);
        assert_eq!(*map.get(&b).unwrap(), 42);
    }
}
