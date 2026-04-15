//! universal_hhtl_f32_encode — Path A: f32 centroid palette + SlotL.
//!
//! PR #183 showed that `HhtlDTensor`'s Base17 palette substrate cannot
//! reconstruct rows for f32 GEMM (per-row ρ ≈ 0.04 on real Qwen3). The
//! encoding is correct for HHTL cascade lookup inference but wrong for
//! reconstruction.
//!
//! This is the reconstruction-grade sibling: `HhtlF32Tensor` with CLAM
//! centroids stored as f32 vectors, same SlotL residual on top. Target:
//! median ρ ≥ 0.95 argmax regime, ≥ 0.98 index regime. Projected storage
//! ratio ~10–30:1 vs the 199:1 of the broken HhtlDTensor path.
//!
//! Usage:
//!   cargo run --release --example universal_hhtl_f32_encode \
//!     --manifest-path crates/thinking-engine/Cargo.toml \
//!     -- /path/to/model.safetensors

use bgz_tensor::shared_palette::{
    classify_component, classify_role, effective_shape, is_encodable, should_use_leaf,
    PaletteGroupKey,
};
use bgz_tensor::hhtl_f32::HhtlF32Tensor;
use bgz_tensor::matryoshka::SvdBasis;
use bgz_tensor::slot_l::SLOT_L_LANES;
use ndarray::hpc::safetensors::read_safetensors_header;
use ndarray::hpc::gguf::{GgmlType, GgufFile, f16_to_f32};
use ndarray::simd::bf16_to_f32_batch;

use std::collections::HashMap;
use std::fs::File;
use std::io::{BufReader, Read, Seek, SeekFrom};
use std::time::Instant;

const PALETTE_K: usize = 256;

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
                let u16s: Vec<u16> = raw.chunks_exact(2).map(|c| u16::from_le_bytes([c[0], c[1]])).collect();
                let mut out = vec![0.0f32; u16s.len()];
                bf16_to_f32_batch(&u16s, &mut out);
                out
            }
            GgmlType::F16 => raw.chunks_exact(2).map(|c| f16_to_f32(u16::from_le_bytes([c[0], c[1]]))).collect(),
            GgmlType::F32 => raw.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect(),
            _ => continue,
        };
        tensors.push(TensorMeta { name: t.name.clone(), shape, size_bytes: n * elem_size, f32_data });
    }
    (tensors, header)
}

struct Bucket { key: PaletteGroupKey, tensors: Vec<TensorMeta> }

fn bucket_tensors(tensors: Vec<TensorMeta>) -> (Vec<Bucket>, Vec<TensorMeta>) {
    let mut by_key: HashMap<PaletteGroupKey, Vec<TensorMeta>> = HashMap::new();
    let mut passthrough: Vec<TensorMeta> = Vec::new();
    for t in tensors {
        if !is_encodable(&t.shape, t.size_bytes) { passthrough.push(t); continue; }
        let key = PaletteGroupKey {
            component: classify_component(&t.name).to_string(),
            role: classify_role(&t.name).to_string(),
            shape: effective_shape(&t.shape),
        };
        by_key.entry(key).or_insert_with(Vec::new).push(t);
    }
    let mut buckets: Vec<Bucket> = by_key.into_iter().map(|(key, tensors)| Bucket { key, tensors }).collect();
    buckets.sort_by(|a, b| (a.key.component.as_str(), a.key.role.as_str(), a.key.shape)
        .cmp(&(b.key.component.as_str(), b.key.role.as_str(), b.key.shape)));
    (buckets, passthrough)
}

fn cosine_f32(a: &[f32], b: &[f32]) -> f64 {
    let mut dot = 0.0f64; let mut na = 0.0f64; let mut nb = 0.0f64;
    for i in 0..a.len().min(b.len()) {
        let x = a[i] as f64; let y = b[i] as f64;
        dot += x * y; na += x * x; nb += y * y;
    }
    let d = (na * nb).sqrt();
    if d < 1e-15 { 0.0 } else { dot / d }
}

#[derive(Default, Debug)]
struct RhoStats {
    regime: String,
    n_rows: usize,
    sum: f64,
    min: f64,
    p_candidates: Vec<f64>,
}

impl RhoStats {
    fn new(regime: &str) -> Self {
        Self { regime: regime.to_string(), n_rows: 0, sum: 0.0, min: 1.0, p_candidates: Vec::new() }
    }
    fn add(&mut self, rho: f64) {
        self.n_rows += 1; self.sum += rho;
        if rho < self.min { self.min = rho; }
        self.p_candidates.push(rho);
    }
    fn mean(&self) -> f64 { if self.n_rows == 0 { 0.0 } else { self.sum / self.n_rows as f64 } }
    fn percentile(&mut self, p: f64) -> f64 {
        if self.p_candidates.is_empty() { return 0.0; }
        self.p_candidates.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let idx = ((p / 100.0) * (self.p_candidates.len() as f64 - 1.0)).round() as usize;
        self.p_candidates[idx.min(self.p_candidates.len() - 1)]
    }
}

fn main() {
    let model_path = std::env::args().nth(1)
        .expect("usage: universal_hhtl_f32_encode <model.safetensors>");

    println!("═══ universal_hhtl_f32_encode — Path A (f32 centroid palette) ═══");
    println!("  Model: {}", model_path);

    let t0 = Instant::now();
    let (tensors, _header) = load_all_tensors_f32(&model_path);
    println!("  Loaded {} tensors in {:.2}s", tensors.len(), t0.elapsed().as_secs_f32());
    let orig_bytes: usize = tensors.iter().map(|t| t.f32_data.len() * 2).sum();
    let (buckets, passthrough) = bucket_tensors(tensors);
    println!("  {} encodable groups, {} passthrough tensors", buckets.len(), passthrough.len());

    let mut argmax_stats = RhoStats::new("argmax");
    let mut index_stats = RhoStats::new("index");

    let mut total_entries_bytes = 0usize;
    let mut total_slot_l_bytes = 0usize;
    let mut total_palette_bytes = 0usize;
    let mut total_svd_basis_bytes = 0usize;

    let n_buckets = buckets.len();
    for (i, bucket) in buckets.into_iter().enumerate() {
        let use_leaf = should_use_leaf(&bucket.key.role);
        let regime = if use_leaf { "INDEX " } else { "ARGMAX" };
        let (r, c) = bucket.key.shape;

        // Build SVD basis once per group (shared across all tensors in bucket) for index regime.
        let basis_opt: Option<SvdBasis> = if use_leaf && !bucket.tensors.is_empty() {
            // Sample up to 4096 rows from the first tensor.
            let first = &bucket.tensors[0];
            let sample_n = r.min(4096);
            let sample_rows: Vec<Vec<f32>> = (0..sample_n)
                .map(|ri| first.f32_data[ri * c..(ri + 1) * c].to_vec())
                .collect();
            Some(SvdBasis::build(&bucket.key.role, &sample_rows, SLOT_L_LANES))
        } else { None };

        // Encode each tensor in the bucket.
        let mut group_rho_sum = 0.0f64;
        let mut group_rho_n = 0usize;
        let tg = Instant::now();

        // Use one shared palette per bucket: build CLAM on the first tensor, apply to all.
        // (Matches the SharedPaletteGroup amortisation pattern.)
        let first_rows: Vec<Vec<f32>> = (0..r)
            .map(|ri| bucket.tensors[0].f32_data[ri * c..(ri + 1) * c].to_vec())
            .collect();
        // The palette comes from the first tensor; subsequent tensors encode against it.
        // For simplicity here we just encode each tensor independently to its own palette:
        // shared-palette variant is a follow-up optimisation.

        for tensor in &bucket.tensors {
            let rows: Vec<Vec<f32>> = (0..r)
                .map(|ri| tensor.f32_data[ri * c..(ri + 1) * c].to_vec())
                .collect();

            let hhtl = if let Some(ref basis) = basis_opt {
                HhtlF32Tensor::encode_with_leaf(&bucket.key.role, &rows, PALETTE_K, basis)
            } else {
                HhtlF32Tensor::encode(&bucket.key.role, &rows, PALETTE_K)
            };

            // ρ on first 64 rows per tensor (or all if fewer)
            let sample_n = r.min(64);
            for ri in 0..sample_n {
                let orig = &rows[ri];
                let recon = hhtl.reconstruct_row(ri, c);
                let rho = cosine_f32(orig, &recon);
                if use_leaf { index_stats.add(rho); } else { argmax_stats.add(rho); }
                group_rho_sum += rho; group_rho_n += 1;
            }

            total_entries_bytes += hhtl.entries_byte_size();
            total_slot_l_bytes += hhtl.slot_l_byte_size();
            total_palette_bytes += hhtl.palette_byte_size_bf16();
            total_svd_basis_bytes += hhtl.svd_basis_byte_size();
        }

        let group_rho = if group_rho_n > 0 { group_rho_sum / group_rho_n as f64 } else { 0.0 };
        let _ = first_rows;
        println!("  [{:>2}/{:<2}] {:<6} {}/{:<9} [{}×{}] × {:<2}  ρ̄={:.4}  {:>6.1}ms",
            i + 1, n_buckets, regime, bucket.key.component, bucket.key.role,
            bucket.key.shape.0, bucket.key.shape.1, bucket.tensors.len(),
            group_rho, tg.elapsed().as_secs_f32() * 1000.0);
    }

    let passthrough_bytes: usize = passthrough.iter().map(|t| t.f32_data.len() * 2).sum();
    let total_output = total_entries_bytes + total_slot_l_bytes + total_palette_bytes
                     + total_svd_basis_bytes + passthrough_bytes;

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
    println!("  Entries (1 byte/row):       {:>10.2} MB", total_entries_bytes as f64 / 1e6);
    println!("  Slot L (8 × i8 per row):    {:>10.2} MB", total_slot_l_bytes as f64 / 1e6);
    println!("  Palettes (f32 → BF16):      {:>10.2} MB", total_palette_bytes as f64 / 1e6);
    println!("  SVD bases:                  {:>10.2} MB", total_svd_basis_bytes as f64 / 1e6);
    println!("  Passthrough (BF16):         {:>10.2} MB  ({} tensors)",
        passthrough_bytes as f64 / 1e6, passthrough.len());
    println!("  ─────────────────────────────────────");
    println!("  Total output:               {:>10.2} MB", total_output as f64 / 1e6);
    println!("  Original (BF16):            {:>10.2} MB", orig_bytes as f64 / 1e6);
    let ratio = orig_bytes as f64 / total_output.max(1) as f64;
    println!("  Ratio:                      {:.1} : 1", ratio);

    println!("\n═══ DECISION ═══");
    let argmax_median = argmax_stats.percentile(50.0);
    let argmax_p5 = argmax_stats.percentile(5.0);
    let index_median = index_stats.percentile(50.0);
    let index_p5 = index_stats.percentile(5.0);
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
        println!("\n  ★ PATH A PASSED");
    } else {
        println!("\n  ✗ PATH A gate(s) failed — see per-tensor numbers above");
    }
    println!("\n═══ DONE ═══");
}
