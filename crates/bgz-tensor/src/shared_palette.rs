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
// HeelBasin reserved for future basin-aware group partitioning
use crate::hhtl_cache::HhtlCache;
#[allow(unused_imports)]
use crate::hhtl_d::{build_hip_families, HeelBasin, HhtlDEntry, HhtlDTensor};
use crate::matryoshka::SvdBasis;
use crate::palette::WeightPalette;
use crate::projection::Base17;
use crate::slot_l::{SlotL, SLOT_L_LANES};
// HashMap used by tests
#[allow(unused_imports)]
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
    /// Per-tensor Slot L entries: (name, [SlotL; n_rows], scale).
    /// Empty when the group role is argmax-regime and doesn't use Slot L.
    pub tensor_slot_l: Vec<(String, Vec<SlotL>, f32)>,
    /// Shared SVD basis across all tensors in the group.
    /// `None` for argmax-regime groups; `Some` for index-regime groups
    /// (embed / lm_head) to preserve per-row identity.
    pub svd_basis: Option<SvdBasis>,
}

/// Group-level dispatch: decide whether tensors with this role need the
/// Slot L leaf residual (index-regime) or the default Slot D + Slot V
/// only (argmax-regime).
///
/// Index-regime: vocab-sized tensors indexed directly by token ID (e.g.
/// `text_embedding.weight`, `lm_head.*.weight`, `codec_embedding.*.weight`).
/// The indexed row's identity cascades into all downstream state — there
/// is no argmax downstream to rescue reconstruction error, so per-row ρ
/// must be near 1. Slot L (8 × i8 on shared SVD basis) supplies that.
///
/// Argmax-regime: attention projections, MLP gate/up/down, output
/// projections, codec head. These feed into `hidden @ W.T → argmax`,
/// where small cos-noise per row leaves the top-1 choice unchanged. 4 B
/// per row (Slot D + Slot V) is sufficient.
pub fn should_use_leaf(role: &str) -> bool {
    matches!(role, "embed" | "lm_head")
}

/// Classify a tensor name into a palette group role.
pub fn classify_role(name: &str) -> &'static str {
    let n = name.to_lowercase();
    if n.contains("q_proj") || n.contains("k_proj") || n.contains("o_proj") {
        "qko"
    } else if n.contains("v_proj") {
        "v"
    } else if n.contains("gate_proj") {
        "gate"
    } else if n.contains("up_proj") {
        "up"
    } else if n.contains("down_proj") {
        "down"
    } else if n.contains("embed") {
        "embed"
    } else if n.contains("lm_head") {
        "lm_head"
    } else if n.contains("projection") || n.contains("codec_head") {
        "projection"
    } else {
        "other"
    }
}

/// Classify a tensor name into component.
pub fn classify_component(name: &str) -> &'static str {
    if name.contains("code_predictor") {
        "code_predictor"
    } else if name.contains("talker") {
        "talker"
    } else if name.contains("speaker_encoder") {
        "speaker_encoder"
    } else {
        "other"
    }
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
    let owned: Vec<Base17> = combined.iter().map(|&b| b.clone()).collect();
    let sample = if owned.len() > 4096 {
        &owned[..4096]
    } else {
        &owned[..]
    };
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
    // Empty-input guard: construct a safe empty group instead of panicking
    // on `tensor_rows_f32[0]`. Callers (including `build_group_with_leaf`'s
    // fallback path) may dispatch here with an empty slice after upstream
    // filtering — return the same shape they would get from a no-op build.
    if tensor_rows_f32.is_empty() {
        let empty_palette = WeightPalette::build(&[], k);
        let cache = HhtlCache::from_palette(empty_palette);
        return SharedPaletteGroup {
            key: key.clone(),
            tensor_names: tensor_names.to_vec(),
            cache,
            hip_families: Vec::new(),
            tensor_entries: Vec::new(),
            fisher_z: None,
            tensor_slot_l: Vec::new(),
            svd_basis: None,
        };
    }

    // Build Base17 projections from first tensor for palette
    let first_rows = &tensor_rows_f32[0];
    let base17_rows: Vec<Base17> = first_rows.iter().map(|r| Base17::from_f32(r)).collect();
    let palette = WeightPalette::build(&base17_rows, k);
    let cache = HhtlCache::from_palette(palette.clone());
    let hip_families = build_hip_families(&palette.entries);

    // Collect representative f32 rows: one per centroid (nearest to centroid)
    let n_centroids = palette.entries.len();
    let mut reps: Vec<Vec<f32>> = vec![Vec::new(); n_centroids];
    let mut rep_dists: Vec<u32> = vec![u32::MAX; n_centroids];
    for (i, row) in first_rows.iter().enumerate() {
        if i >= base17_rows.len() {
            break;
        }
        let (ci, dist) = cache.nearest(&base17_rows[i]);
        let ci = ci as usize;
        if ci < n_centroids && dist < rep_dists[ci] {
            reps[ci] = row.clone();
            rep_dists[ci] = dist;
        }
    }
    for (ci, rep) in reps.iter_mut().enumerate().take(n_centroids) {
        if rep.is_empty() {
            *rep = palette.entries[ci].to_f32(key.shape.1);
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
        tensor_slot_l: Vec::new(),
        svd_basis: None,
    }
}

/// Build a shared palette group with Slot L leaf residual on a shared SVD basis.
///
/// Use this entry point for index-regime groups (role == "embed" or "lm_head")
/// where per-row identity must be preserved. For argmax-regime groups
/// (everything else), prefer `build_group_with_fisher_z` — the extra Slot L
/// is wasted bytes when the downstream argmax is robust to small row noise.
///
/// The SVD basis is built **once per group** from a sample of the first
/// tensor's rows (up to 4096 sampled) and shared across all tensors in the
/// group, amortising the `n_components × n_cols × 2 B` basis overhead.
///
/// Wire cost per row: 4 B (Slot D + Slot V) + 8 B (Slot L) = **12 B/row**.
pub fn build_group_with_leaf(
    key: &PaletteGroupKey,
    tensor_names: &[String],
    tensor_rows_f32: &[Vec<Vec<f32>>],
    k: usize,
) -> SharedPaletteGroup {
    // Fall back to plain build if the group isn't index-regime.
    if !should_use_leaf(&key.role) {
        return build_group_with_fisher_z(key, tensor_names, tensor_rows_f32, k);
    }
    if tensor_rows_f32.is_empty() {
        return build_group_with_fisher_z(key, tensor_names, tensor_rows_f32, k);
    }

    // Shared palette from first tensor's Base17 projection (same as Fisher-z path).
    let first_rows = &tensor_rows_f32[0];
    let base17_rows: Vec<Base17> = first_rows.iter().map(|r| Base17::from_f32(r)).collect();
    let palette = WeightPalette::build(&base17_rows, k);
    let cache = HhtlCache::from_palette(palette.clone());
    let hip_families = build_hip_families(&palette.entries);

    // Shared SVD basis from a sample of the first tensor's f32 rows.
    // Cap at 4096 rows so SVD stays fast even on 151K-row vocab tensors.
    let sample_cap = first_rows.len().min(4096);
    let svd_sample: &[Vec<f32>] = &first_rows[..sample_cap];
    let basis = SvdBasis::build(&key.role, svd_sample, SLOT_L_LANES);

    // Collect representative f32 rows for Fisher z (same approach as the
    // plain build — one per centroid, nearest to it).
    let n_centroids = palette.entries.len();
    let mut reps: Vec<Vec<f32>> = vec![Vec::new(); n_centroids];
    let mut rep_dists: Vec<u32> = vec![u32::MAX; n_centroids];
    for (i, row) in first_rows.iter().enumerate() {
        if i >= base17_rows.len() {
            break;
        }
        let (ci, dist) = cache.nearest(&base17_rows[i]);
        let ci = ci as usize;
        if ci < n_centroids && dist < rep_dists[ci] {
            reps[ci] = row.clone();
            rep_dists[ci] = dist;
        }
    }
    for (ci, rep) in reps.iter_mut().enumerate().take(n_centroids) {
        if rep.is_empty() {
            *rep = palette.entries[ci].to_f32(key.shape.1);
        }
    }
    let fisher_z = Some(FisherZTable::build(&reps, n_centroids));

    // Encode each tensor with Slot L via encode_with_leaf.
    let role_name = format!("{}_{}", key.component, key.role);
    let mut tensor_entries: Vec<(String, Vec<HhtlDEntry>)> = Vec::with_capacity(tensor_names.len());
    let mut tensor_slot_l: Vec<(String, Vec<SlotL>, f32)> = Vec::with_capacity(tensor_names.len());
    for (name, rows) in tensor_names.iter().zip(tensor_rows_f32.iter()) {
        let hhtld = HhtlDTensor::encode_with_leaf(&role_name, rows, &cache, &hip_families, &basis);
        tensor_entries.push((name.clone(), hhtld.entries));
        if let (Some(slot_l), Some(scale)) = (hhtld.slot_l, hhtld.slot_l_scale) {
            tensor_slot_l.push((name.clone(), slot_l, scale));
        }
    }

    SharedPaletteGroup {
        key: key.clone(),
        tensor_names: tensor_names.to_vec(),
        cache,
        hip_families,
        tensor_entries,
        fisher_z,
        tensor_slot_l,
        svd_basis: Some(basis),
    }
}

impl SharedPaletteGroup {
    /// Total Slot L byte size across all tensors in this group (not counting
    /// the shared SVD basis, which is amortised per group).
    pub fn slot_l_byte_size(&self) -> usize {
        self.tensor_slot_l
            .iter()
            .map(|(_, entries, _)| entries.len() * SlotL::BYTE_SIZE)
            .sum()
    }

    /// Shared SVD basis byte size (0 if not present).
    pub fn svd_basis_byte_size(&self) -> usize {
        self.svd_basis.as_ref().map(|b| b.byte_size()).unwrap_or(0)
    }

    /// Find the SlotL vector for a given tensor name (and its shared scale).
    pub fn slot_l_for(&self, tensor_name: &str) -> Option<(&[SlotL], f32)> {
        self.tensor_slot_l
            .iter()
            .find(|(n, _, _)| n == tensor_name)
            .map(|(_, entries, scale)| (entries.as_slice(), *scale))
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
        assert_eq!(
            classify_role("talker.layers.0.self_attn.q_proj.weight"),
            "qko"
        );
        assert_eq!(
            classify_role("talker.layers.0.self_attn.v_proj.weight"),
            "v"
        );
        assert_eq!(
            classify_role("talker.layers.0.mlp.gate_proj.weight"),
            "gate"
        );
        assert_eq!(classify_role("talker.model.text_embedding.weight"), "embed");
        assert_eq!(
            classify_role("talker.code_predictor.lm_head.0.weight"),
            "lm_head"
        );
        assert_eq!(
            classify_role("talker.layers.0.input_layernorm.weight"),
            "other"
        );
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

    // ═════════════════════════════════════════════════════════════════
    // Slot L integration tests
    // ═════════════════════════════════════════════════════════════════

    #[test]
    fn should_use_leaf_classification() {
        assert!(should_use_leaf("embed"));
        assert!(should_use_leaf("lm_head"));
        // Argmax regime — should NOT use leaf.
        assert!(!should_use_leaf("qko"));
        assert!(!should_use_leaf("v"));
        assert!(!should_use_leaf("gate"));
        assert!(!should_use_leaf("up"));
        assert!(!should_use_leaf("down"));
        assert!(!should_use_leaf("projection"));
        assert!(!should_use_leaf("other"));
    }

    fn low_rank_rows(n: usize, cols: usize, seed: u32) -> Vec<Vec<f32>> {
        let n_atoms = 8usize;
        let mut atoms: Vec<Vec<f32>> = Vec::with_capacity(n_atoms);
        let mut s = seed;
        let mut next = || {
            s = s.wrapping_mul(1103515245).wrapping_add(12345);
            ((s >> 8) as i32 as f32) / 2_147_483_648.0
        };
        for _ in 0..n_atoms {
            let atom: Vec<f32> = (0..cols).map(|_| next()).collect();
            atoms.push(atom);
        }
        (0..n)
            .map(|_| {
                let mut row = vec![0.0f32; cols];
                for atom in &atoms {
                    let w = next() * 0.5;
                    for j in 0..cols {
                        row[j] += atom[j] * w;
                    }
                }
                row
            })
            .collect()
    }

    #[test]
    fn build_group_with_leaf_falls_back_for_argmax_regime() {
        // role="qko" -> should_use_leaf returns false -> fall back to the
        // plain Fisher z path -> svd_basis and tensor_slot_l are empty.
        let key = PaletteGroupKey {
            component: "talker".into(),
            role: "qko".into(),
            shape: (32, 64),
        };
        let names = vec!["talker.layers.0.self_attn.q_proj.weight".to_string()];
        let rows = vec![low_rank_rows(32, 64, 0x1111)];
        let group = build_group_with_leaf(&key, &names, &rows, 16);
        assert!(
            group.svd_basis.is_none(),
            "argmax-regime group should have no SVD basis"
        );
        assert!(
            group.tensor_slot_l.is_empty(),
            "argmax-regime group should have no Slot L entries"
        );
        assert_eq!(group.slot_l_byte_size(), 0);
        assert_eq!(group.svd_basis_byte_size(), 0);
    }

    #[test]
    fn build_group_with_leaf_populates_slot_l_for_index_regime() {
        // role="embed" -> should_use_leaf returns true -> SVD basis is built
        // and every tensor gets Slot L entries.
        let key = PaletteGroupKey {
            component: "talker".into(),
            role: "embed".into(),
            shape: (64, 128),
        };
        let names = vec![
            "talker.model.text_embedding.weight".to_string(),
            "talker.model.second_embedding.weight".to_string(),
        ];
        let rows = vec![
            low_rank_rows(64, 128, 0x2222),
            low_rank_rows(64, 128, 0x3333),
        ];
        let group = build_group_with_leaf(&key, &names, &rows, 16);

        assert!(group.svd_basis.is_some());
        assert_eq!(group.tensor_slot_l.len(), 2);
        assert_eq!(group.tensor_slot_l[0].1.len(), 64);
        assert_eq!(group.tensor_slot_l[1].1.len(), 64);

        // Slot L sizing
        assert_eq!(group.slot_l_byte_size(), 2 * 64 * SlotL::BYTE_SIZE);
        assert!(group.svd_basis_byte_size() > 0);

        // Lookup by name
        let (slot_l_a, scale_a) = group.slot_l_for(&names[0]).expect("first tensor");
        assert_eq!(slot_l_a.len(), 64);
        assert!(scale_a > 0.0);
        assert!(group.slot_l_for("nonexistent").is_none());
    }

    #[test]
    fn empty_input_returns_safe_empty_group_not_panic() {
        // Regression test for codex P2 on #182: empty-input guard must not
        // dispatch into a function that indexes tensor_rows_f32[0].
        let key = PaletteGroupKey {
            component: "talker".into(),
            role: "embed".into(), // index-regime path
            shape: (64, 128),
        };
        let empty_names: Vec<String> = vec![];
        let empty_rows: Vec<Vec<Vec<f32>>> = vec![];

        // Path 1: via build_group_with_leaf (index regime)
        let leaf_group = build_group_with_leaf(&key, &empty_names, &empty_rows, 16);
        assert!(leaf_group.tensor_entries.is_empty());
        assert!(leaf_group.tensor_slot_l.is_empty());
        assert!(leaf_group.svd_basis.is_none());
        assert!(leaf_group.fisher_z.is_none());

        // Path 2: direct call to build_group_with_fisher_z
        let fz_group = build_group_with_fisher_z(&key, &empty_names, &empty_rows, 16);
        assert!(fz_group.tensor_entries.is_empty());
        assert!(fz_group.fisher_z.is_none());

        // Path 3: argmax-regime key (build_group_with_leaf falls back to Fisher-z path)
        let key_argmax = PaletteGroupKey {
            component: "talker".into(),
            role: "qko".into(),
            shape: (64, 128),
        };
        let ax_group = build_group_with_leaf(&key_argmax, &empty_names, &empty_rows, 16);
        assert!(ax_group.tensor_entries.is_empty());
    }

    #[test]
    fn svd_basis_shared_across_group_not_per_tensor() {
        // Confirms amortisation: 2 tensors, 1 basis.
        let key = PaletteGroupKey {
            component: "talker".into(),
            role: "lm_head".into(),
            shape: (32, 64),
        };
        let names = vec![
            "talker.code_predictor.lm_head.0.weight".to_string(),
            "talker.code_predictor.lm_head.1.weight".to_string(),
        ];
        let rows = vec![low_rank_rows(32, 64, 0x4444), low_rank_rows(32, 64, 0x5555)];
        let group = build_group_with_leaf(&key, &names, &rows, 8);

        assert_eq!(group.tensor_slot_l.len(), 2);
        // One SVD basis, 2 tensors worth of entries:
        let basis_size = group.svd_basis_byte_size();
        let entries_size = group.slot_l_byte_size();
        assert!(basis_size > 0);
        assert_eq!(entries_size, 2 * 32 * SlotL::BYTE_SIZE);
        // Shared-basis amortisation: basis_size is constant regardless of
        // tensor count; entries_size scales linearly.
    }
}
