//! universal_hhtld_encode — model-generic HHTL-D encoder with SlotL dispatch.
//!
//! Implements the `universal_hhtld_encode` proposal from
//! `docs/COMPRESSION_MINDSET_SHIFTS.md` (PR #179). Consumes any BPE-vocab
//! safetensors model, buckets tensors by (component, role, shape), and
//! routes each bucket through `bgz_tensor::shared_palette::build_group_with_leaf`
//! which auto-dispatches on role:
//!
//! | Regime | `role` | Wire cost | Quality target |
//! |---|---|---|---|
//! | Argmax  | qko / v / gate / up / down / projection | 4 B/row  | ρ ≳ 0.95 |
//! | Index   | embed / lm_head                           | 12 B/row | ρ ≳ 0.98 |
//! | Passthrough | norms / bias / shapes < is_encodable | BF16     | exact    |
//!
//! ## Validation gates reported
//!
//! This example ships gates 1 + 3 of the four-gate plan:
//!
//!   1. Per-tensor ρ histogram, split by regime
//!   3. Storage ratio vs BF16 original
//!
//! Gates 2 (argmax-parity on held-out prompt) and 4 (WAV envelope match)
//! require integration with `tts_full_inference.rs` and land in a
//! follow-up PR.
//!
//! ## Usage
//!
//! ```sh
//! cargo run --release --example universal_hhtld_encode \
//!     --manifest-path crates/thinking-engine/Cargo.toml \
//!     -- /home/user/models/qwen3-tts-0.6b/model.safetensors
//! ```

use bgz_tensor::shared_palette::{
    classify_component, classify_role, effective_shape, is_encodable,
    should_use_leaf, build_group_with_leaf,
    PaletteGroupKey, SharedPaletteGroup,
};
use bgz_tensor::hhtl_d::HhtlDTensor;
use ndarray::hpc::safetensors::read_safetensors_header;
use ndarray::hpc::gguf::{GgmlType, GgufFile, f16_to_f32};
use ndarray::simd::bf16_to_f32_batch;

use std::collections::HashMap;
use std::fs::File;
use std::io::{BufReader, Read, Seek, SeekFrom};
use std::time::Instant;

const PALETTE_K: usize = 256;

// ═════════════════════════════════════════════════════════════════════
// Safetensors load — any BPE-vocab model, BF16 / F16 / F32 tensors
// ═════════════════════════════════════════════════════════════════════

/// Metadata about one tensor in the safetensors file.
#[derive(Clone, Debug)]
struct TensorMeta {
    name: String,
    shape: Vec<usize>,
    size_bytes: usize,
    f32_data: Vec<f32>,
}

fn load_all_tensors_f32(model_path: &str) -> (Vec<TensorMeta>, GgufFile) {
    let file = File::open(model_path).expect("open model");
    let mut reader = BufReader::new(file);
    let header = read_safetensors_header(&mut reader).expect("parse safetensors header");

    let mut tensors: Vec<TensorMeta> = Vec::with_capacity(header.tensors.len());
    for t in &header.tensors {
        let n: usize = t.dimensions.iter().map(|&d| d as usize).product();
        let shape: Vec<usize> = t.dimensions.iter().map(|&d| d as usize).collect();

        let elem_size = match t.dtype {
            GgmlType::BF16 | GgmlType::F16 => 2,
            GgmlType::F32 => 4,
            _ => continue,
        };

        reader.seek(SeekFrom::Start(header.tensor_data_offset + t.offset)).unwrap();
        let mut raw = vec![0u8; n * elem_size];
        if reader.read_exact(&mut raw).is_err() { continue; }

        let f32_data: Vec<f32> = match t.dtype {
            GgmlType::BF16 => {
                let u16s: Vec<u16> = raw.chunks_exact(2)
                    .map(|c| u16::from_le_bytes([c[0], c[1]]))
                    .collect();
                let mut out = vec![0.0f32; u16s.len()];
                bf16_to_f32_batch(&u16s, &mut out);
                out
            }
            GgmlType::F16 => raw.chunks_exact(2)
                .map(|c| f16_to_f32(u16::from_le_bytes([c[0], c[1]])))
                .collect(),
            GgmlType::F32 => raw.chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect(),
            _ => continue,
        };

        tensors.push(TensorMeta {
            name: t.name.clone(),
            shape,
            size_bytes: n * elem_size,
            f32_data,
        });
    }

    (tensors, header)
}

// ═════════════════════════════════════════════════════════════════════
// Tensor bucketing by (component, role, shape)
// ═════════════════════════════════════════════════════════════════════

struct Bucket {
    key: PaletteGroupKey,
    tensors: Vec<TensorMeta>,
}

fn bucket_tensors(tensors: Vec<TensorMeta>) -> (Vec<Bucket>, Vec<TensorMeta>) {
    // Encodable → bucketed groups, non-encodable → passthrough
    let mut by_key: HashMap<PaletteGroupKey, Vec<TensorMeta>> = HashMap::new();
    let mut passthrough: Vec<TensorMeta> = Vec::new();

    for t in tensors {
        if !is_encodable(&t.shape, t.size_bytes) {
            passthrough.push(t);
            continue;
        }
        let key = PaletteGroupKey {
            component: classify_component(&t.name).to_string(),
            role: classify_role(&t.name).to_string(),
            shape: effective_shape(&t.shape),
        };
        by_key.entry(key).or_insert_with(Vec::new).push(t);
    }

    let mut buckets: Vec<Bucket> = by_key.into_iter()
        .map(|(key, tensors)| Bucket { key, tensors })
        .collect();
    // Stable order for reporting
    buckets.sort_by(|a, b| (a.key.component.as_str(), a.key.role.as_str(), a.key.shape)
        .cmp(&(b.key.component.as_str(), b.key.role.as_str(), b.key.shape)));
    (buckets, passthrough)
}

// ═════════════════════════════════════════════════════════════════════
// Per-row cosine between originals and reconstructed rows
// ═════════════════════════════════════════════════════════════════════

fn cosine_f32(a: &[f32], b: &[f32]) -> f64 {
    let mut dot = 0.0f64;
    let mut na = 0.0f64;
    let mut nb = 0.0f64;
    let n = a.len().min(b.len());
    for i in 0..n {
        let x = a[i] as f64;
        let y = b[i] as f64;
        dot += x * y; na += x * x; nb += y * y;
    }
    let d = (na * nb).sqrt();
    if d < 1e-15 { 0.0 } else { dot / d }
}

/// Reconstruct a single row from a SharedPaletteGroup's state.
/// Uses `HhtlDTensor::reconstruct_row` via a temporary struct that borrows
/// the group's cache + entries + optional Slot L + SVD basis.
fn reconstruct_row_from_group(
    group: &SharedPaletteGroup,
    tensor_name: &str,
    row_idx: usize,
    n_cols: usize,
) -> Vec<f32> {
    let entries = group.tensor_entries.iter()
        .find(|(n, _)| n == tensor_name)
        .map(|(_, e)| e.clone())
        .unwrap_or_default();
    let (slot_l, scale) = group.slot_l_for(tensor_name)
        .map(|(s, sc)| (Some(s.to_vec()), Some(sc)))
        .unwrap_or((None, None));

    let hhtld = HhtlDTensor {
        role: group.key.role.clone(),
        basin: bgz_tensor::hhtl_d::HeelBasin::from_role(&group.key.role),
        cache: group.cache.clone(),
        entries,
        original_shape: [group.key.shape.0, group.key.shape.1],
        gamma_meta: group.cache.gamma_meta,
        fisher_z: None,
        slot_l,
        slot_l_scale: scale,
        svd_basis: group.svd_basis.clone(),
    };
    hhtld.reconstruct_row(row_idx, n_cols)
}

/// Row cosine statistics split by regime.
#[derive(Default, Debug)]
struct RhoStats {
    regime: String,
    n_rows: usize,
    sum: f64,
    min: f64,
    p5_candidates: Vec<f64>,  // collect all, sort at the end
}

impl RhoStats {
    fn new(regime: &str) -> Self {
        Self { regime: regime.to_string(), n_rows: 0, sum: 0.0, min: 1.0, p5_candidates: Vec::new() }
    }
    fn add(&mut self, rho: f64) {
        self.n_rows += 1;
        self.sum += rho;
        if rho < self.min { self.min = rho; }
        self.p5_candidates.push(rho);
    }
    fn mean(&self) -> f64 { if self.n_rows == 0 { 0.0 } else { self.sum / self.n_rows as f64 } }
    fn percentile(&mut self, p: f64) -> f64 {
        if self.p5_candidates.is_empty() { return 0.0; }
        self.p5_candidates.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let idx = ((p / 100.0) * (self.p5_candidates.len() as f64 - 1.0)).round() as usize;
        self.p5_candidates[idx.min(self.p5_candidates.len() - 1)]
    }
}

// ═════════════════════════════════════════════════════════════════════
// Main: encode + validate
// ═════════════════════════════════════════════════════════════════════

fn main() {
    let model_path = std::env::args().nth(1)
        .expect("usage: universal_hhtld_encode <model.safetensors>");

    println!("═══ universal_hhtld_encode — model-generic HHTL-D ═══");
    println!("  Model: {}", model_path);

    let t0 = Instant::now();
    let (tensors, _header) = load_all_tensors_f32(&model_path);
    println!("  Loaded {} tensors in {:.2}s", tensors.len(), t0.elapsed().as_secs_f32());

    let orig_bytes: usize = tensors.iter().map(|t| t.f32_data.len() * 2).sum(); // BF16 footprint
    let (buckets, passthrough) = bucket_tensors(tensors);

    println!("\n  {} encodable groups, {} passthrough tensors",
        buckets.len(), passthrough.len());

    // ─── Encode every bucket ────────────────────────────────────────
    let mut argmax_stats = RhoStats::new("argmax");
    let mut index_stats = RhoStats::new("index");
    let mut groups: Vec<SharedPaletteGroup> = Vec::new();

    let mut total_entries_bytes = 0usize;
    let mut total_slot_l_bytes = 0usize;
    let mut total_svd_basis_bytes = 0usize;
    let mut total_palette_bytes = 0usize;
    let mut total_fisher_z_bytes = 0usize;

    let n_buckets = buckets.len();
    for (i, bucket) in buckets.into_iter().enumerate() {
        let is_index = should_use_leaf(&bucket.key.role);
        let regime = if is_index { "INDEX" } else { "ARGMAX" };
        let names: Vec<String> = bucket.tensors.iter().map(|t| t.name.clone()).collect();
        let rows_f32: Vec<Vec<Vec<f32>>> = bucket.tensors.iter().map(|t| {
            let (r, c) = bucket.key.shape;
            (0..r).map(|ri| t.f32_data[ri * c..(ri + 1) * c].to_vec()).collect()
        }).collect();

        let tg = Instant::now();
        let group = build_group_with_leaf(&bucket.key, &names, &rows_f32, PALETTE_K);
        let el = tg.elapsed();

        // Per-tensor ρ vs original — spot-check first 64 rows per tensor to
        // keep time bounded on 151K-row vocab tensors. For smaller tensors
        // this covers every row.
        let mut group_rho_sum = 0.0f64;
        let mut group_rho_n = 0usize;
        for tensor in &bucket.tensors {
            let (r, c) = bucket.key.shape;
            let sample_n = r.min(64);
            for ri in 0..sample_n {
                let orig = &tensor.f32_data[ri * c..(ri + 1) * c];
                let recon = reconstruct_row_from_group(&group, &tensor.name, ri, c);
                let rho = cosine_f32(orig, &recon);
                if is_index { index_stats.add(rho); } else { argmax_stats.add(rho); }
                group_rho_sum += rho; group_rho_n += 1;
            }
        }
        let group_rho = if group_rho_n > 0 { group_rho_sum / group_rho_n as f64 } else { 0.0 };

        // Storage accounting
        let entries_b = group.tensor_entries.iter().map(|(_, e)| e.len() * 4).sum::<usize>();
        let slot_l_b = group.slot_l_byte_size();
        let svd_b = group.svd_basis_byte_size();
        let palette_b = group.cache.palette.entries.len() * 34;  // Base17 = 34 bytes
        let fisher_z_b = group.fisher_z.as_ref().map(|f| f.byte_size()).unwrap_or(0);
        total_entries_bytes += entries_b;
        total_slot_l_bytes += slot_l_b;
        total_svd_basis_bytes += svd_b;
        total_palette_bytes += palette_b;
        total_fisher_z_bytes += fisher_z_b;

        println!("  [{:>2}/{:<2}] {:<6} {}/{:<9} [{}×{}] × {:<2}  ρ̄={:.4}  {:>6.1}ms",
            i + 1, n_buckets, regime, bucket.key.component, bucket.key.role,
            bucket.key.shape.0, bucket.key.shape.1, bucket.tensors.len(),
            group_rho, el.as_secs_f32() * 1000.0);

        groups.push(group);
    }

    let passthrough_bytes: usize = passthrough.iter().map(|t| t.f32_data.len() * 2).sum();

    let total_output = total_entries_bytes + total_slot_l_bytes + total_svd_basis_bytes
                     + total_palette_bytes + total_fisher_z_bytes + passthrough_bytes;

    // ─── Report ─────────────────────────────────────────────────────
    println!("\n═══ GATE 1 — per-row ρ by regime ═══");
    for stats in [&mut argmax_stats, &mut index_stats] {
        if stats.n_rows == 0 { continue; }
        let p5 = stats.percentile(5.0);
        let p50 = stats.percentile(50.0);
        let p95 = stats.percentile(95.0);
        println!("  {:<6} regime: {} rows sampled  mean={:.4}  min={:.4}  p5={:.4}  p50={:.4}  p95={:.4}",
            stats.regime.to_uppercase(), stats.n_rows,
            stats.mean(), stats.min, p5, p50, p95);
    }

    println!("\n═══ GATE 3 — storage ratio ═══");
    println!("  Entries (Slot D + Slot V):  {:>10.2} MB", total_entries_bytes as f64 / 1e6);
    println!("  Slot L (8 × i8 per row):    {:>10.2} MB", total_slot_l_bytes as f64 / 1e6);
    println!("  SVD bases (shared):         {:>10.2} MB", total_svd_basis_bytes as f64 / 1e6);
    println!("  Palettes (Base17 × 256):    {:>10.2} MB", total_palette_bytes as f64 / 1e6);
    println!("  Fisher-Z tables:            {:>10.2} MB", total_fisher_z_bytes as f64 / 1e6);
    println!("  Passthrough (BF16):         {:>10.2} MB  ({} tensors)",
        passthrough_bytes as f64 / 1e6, passthrough.len());
    println!("  ─────────────────────────────────────");
    println!("  Total output:               {:>10.2} MB", total_output as f64 / 1e6);
    println!("  Original (BF16):            {:>10.2} MB", orig_bytes as f64 / 1e6);
    let ratio = orig_bytes as f64 / total_output.max(1) as f64;
    println!("  Ratio:                      {:.1} : 1", ratio);

    // Pass/fail gate
    println!("\n═══ DECISION ═══");
    let argmax_p5 = argmax_stats.percentile(5.0);
    let argmax_median = argmax_stats.percentile(50.0);
    let index_p5 = index_stats.percentile(5.0);
    let index_median = index_stats.percentile(50.0);

    let argmax_pass = argmax_median >= 0.95 && argmax_p5 >= 0.90;
    let index_pass = index_median >= 0.98 && index_p5 >= 0.95;
    let ratio_pass = ratio >= 2.0;

    println!("  ARGMAX regime (target median ≥ 0.95, p5 ≥ 0.90): {}",
        if argmax_pass { "PASS" } else { "FAIL" });
    println!("  INDEX  regime (target median ≥ 0.98, p5 ≥ 0.95): {}",
        if index_pass { "PASS" } else { "FAIL" });
    println!("  RATIO  (target ≥ 2:1):                            {}",
        if ratio_pass { "PASS" } else { "FAIL" });

    if argmax_pass && index_pass && ratio_pass {
        println!("\n  ★ STAGE-1 VALIDATION PASSED");
        println!("    Next: integrate with tts_full_inference for gates 2 + 4 (argmax-parity, WAV envelope)");
    } else {
        println!("\n  ✗ STAGE-1 VALIDATION FAILED — investigate failing gate(s) before integration");
    }

    println!("\n═══ DONE ═══");
}
