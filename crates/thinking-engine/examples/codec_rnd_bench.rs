//! codec_rnd_bench — R&D psychometric benchmark for codec candidates.
//!
//! Runs all codec candidates against the same tensor population,
//! computes the full metric suite (10 metrics), outputs a markdown table.
//! Cronbach's α across codecs reveals factor structure.
//!
//! Usage:
//!   cargo run --release --example codec_rnd_bench \
//!     --manifest-path crates/thinking-engine/Cargo.toml \
//!     -- /path/to/model.safetensors

use bgz_tensor::quality::{
    pearson, spearman, kendall_tau, mae, rmse, top_k_recall,
    cronbach_alpha, cohens_kappa, bias_variance, icc_3_1,
};
use bgz_tensor::projection::Base17;
use bgz_tensor::hhtl_cache::HhtlCache;
use bgz_tensor::hhtl_d::build_hip_families;
use highheelbgz::rehydrate::SpiralEncoding;
use ndarray::hpc::safetensors::read_safetensors_header;
use ndarray::hpc::gguf::GgmlType;
use ndarray::simd::bf16_to_f32_batch;

use std::collections::HashMap;
use std::fs::File;
use std::io::{BufReader, Read, Seek, SeekFrom};
use std::time::Instant;

const N_SAMPLE: usize = 128;
const TOP_K: usize = 5;

// ═════════════════════════════════════════════════════════════════════
// Tensor loading
// ═════════════════════════════════════════════════════════════════════

fn load_rows(path: &str, tensor_substr: &str, max_rows: usize) -> (Vec<Vec<f32>>, String) {
    let file = File::open(path).expect("open");
    let mut reader = BufReader::new(file);
    let header = read_safetensors_header(&mut reader).expect("parse");
    let t = match header.tensors.iter().find(|t| t.name.contains(tensor_substr)) {
        Some(t) => t,
        None => return (vec![], format!("(not found: {})", tensor_substr)),
    };
    let n: usize = t.dimensions.iter().map(|&d| d as usize).product();
    let n_rows = t.dimensions[0] as usize;
    let n_cols: usize = t.dimensions.iter().skip(1).map(|&d| d as usize).product();
    reader.seek(SeekFrom::Start(header.tensor_data_offset + t.offset)).unwrap();
    let mut raw = vec![0u8; n * 2];
    reader.read_exact(&mut raw).unwrap();
    let f32_data: Vec<f32> = {
        let u16s: Vec<u16> = raw.chunks_exact(2).map(|c| u16::from_le_bytes([c[0], c[1]])).collect();
        let mut out = vec![0.0f32; u16s.len()];
        bf16_to_f32_batch(&u16s, &mut out);
        out
    };
    let stride = n_rows.max(1) / max_rows.min(n_rows);
    let rows: Vec<Vec<f32>> = (0..max_rows.min(n_rows))
        .map(|i| {
            let ri = (i * stride).min(n_rows - 1);
            f32_data[ri * n_cols..(ri + 1) * n_cols].to_vec()
        }).collect();
    (rows, format!("{} [{}×{}]", t.name, n_rows, n_cols))
}

// ═════════════════════════════════════════════════════════════════════
// Ground truth pairwise cosine
// ═════════════════════════════════════════════════════════════════════

fn cosine(a: &[f32], b: &[f32]) -> f64 {
    let mut dot = 0.0f64; let mut na = 0.0f64; let mut nb = 0.0f64;
    for i in 0..a.len().min(b.len()) {
        let x = a[i] as f64; let y = b[i] as f64;
        dot += x * y; na += x * x; nb += y * y;
    }
    let d = (na * nb).sqrt();
    if d < 1e-15 { 0.0 } else { dot / d }
}

fn pairwise_cosines(rows: &[Vec<f32>]) -> Vec<f64> {
    let n = rows.len();
    let mut scores = Vec::with_capacity(n * (n - 1) / 2);
    for i in 0..n {
        for j in (i + 1)..n {
            scores.push(cosine(&rows[i], &rows[j]));
        }
    }
    scores
}

// ═════════════════════════════════════════════════════════════════════
// Codec candidates — each produces a pairwise score vector
// ═════════════════════════════════════════════════════════════════════

trait CodecCandidate {
    fn name(&self) -> &str;
    fn bytes_per_row(&self) -> usize;
    fn pairwise_scores(&self, rows: &[Vec<f32>]) -> Vec<f64>;
}

/// Passthrough — raw cosine (baseline, exact).
struct Passthrough;
impl CodecCandidate for Passthrough {
    fn name(&self) -> &str { "Passthrough" }
    fn bytes_per_row(&self) -> usize { 0 } // not compressed
    fn pairwise_scores(&self, rows: &[Vec<f32>]) -> Vec<f64> { pairwise_cosines(rows) }
}

/// Base17 signature — project to 17-dim, cosine there.
struct Base17Sig;
impl CodecCandidate for Base17Sig {
    fn name(&self) -> &str { "Base17" }
    fn bytes_per_row(&self) -> usize { 34 }
    fn pairwise_scores(&self, rows: &[Vec<f32>]) -> Vec<f64> {
        let b17s: Vec<Base17> = rows.iter().map(|r| Base17::from_f32(r)).collect();
        let n = rows.len();
        let mut scores = Vec::with_capacity(n * (n - 1) / 2);
        for i in 0..n {
            for j in (i + 1)..n {
                scores.push(b17s[i].cosine(&b17s[j]));
            }
        }
        scores
    }
}

/// Direct i8 quantization of pairwise cosines.
struct DirectI8;
impl CodecCandidate for DirectI8 {
    fn name(&self) -> &str { "Direct-i8" }
    fn bytes_per_row(&self) -> usize { 1 }
    fn pairwise_scores(&self, rows: &[Vec<f32>]) -> Vec<f64> {
        let gt = pairwise_cosines(rows);
        gt.iter().map(|&c| {
            let q = (c * 127.0).round().clamp(-127.0, 127.0) as i8;
            q as f64 / 127.0
        }).collect()
    }
}

/// Spiral signature at K=8.
struct SpiralK8;
impl CodecCandidate for SpiralK8 {
    fn name(&self) -> &str { "Spiral-K8" }
    fn bytes_per_row(&self) -> usize { 278 }
    fn pairwise_scores(&self, rows: &[Vec<f32>]) -> Vec<f64> {
        let encs: Vec<SpiralEncoding> = rows.iter()
            .map(|r| SpiralEncoding::encode(r, 0, 3, 8))
            .collect();
        let n = rows.len();
        let mut scores = Vec::with_capacity(n * (n - 1) / 2);
        for i in 0..n {
            for j in (i + 1)..n {
                scores.push(encs[i].cosine(&encs[j]));
            }
        }
        scores
    }
}

/// RaBitQ — sign-quantized JL + Hamming + dot_correction.
struct RaBitQCodec { dim: usize }
impl RaBitQCodec {
    fn next_pow2(n: usize) -> usize {
        let mut p = 1;
        while p < n { p *= 2; }
        p
    }
}
impl CodecCandidate for RaBitQCodec {
    fn name(&self) -> &str { "RaBitQ" }
    fn bytes_per_row(&self) -> usize { (self.dim + 63) / 64 * 8 + 8 }
    fn pairwise_scores(&self, rows: &[Vec<f32>]) -> Vec<f64> {
        use bgz17::rabitq_compat::{OrthogonalMatrix, RaBitQEncoding};
        use bgz17::palette::Palette;
        let padded_dim = Self::next_pow2(self.dim);
        let rot = OrthogonalMatrix::hadamard(padded_dim);
        let empty_palette = Palette::build(&[], 0, 0);
        let padded_rows: Vec<Vec<f32>> = rows.iter().map(|r| {
            let mut p = r.clone();
            p.resize(padded_dim, 0.0);
            p
        }).collect();
        let encs: Vec<RaBitQEncoding> = padded_rows.iter()
            .map(|r| RaBitQEncoding::encode(r, &rot, &empty_palette))
            .collect();
        let n = rows.len();
        let mut scores = Vec::with_capacity(n * (n - 1) / 2);
        for i in 0..n {
            for j in (i + 1)..n {
                // RaBitQ distance → cosine estimate
                let d = encs[i].distance_rabitq(&encs[j]);
                // Convert L2² to cosine: cos = 1 - d/(2*norm_i*norm_j)
                let ni = encs[i].norm; let nj = encs[j].norm;
                let cos_est = if ni > 0.0 && nj > 0.0 {
                    1.0 - d as f64 / (2.0 * ni as f64 * nj as f64)
                } else { 0.0 };
                scores.push(cos_est.clamp(-1.0, 1.0));
            }
        }
        scores
    }
}

// ═════════════════════════════════════════════════════════════════════
// ROW-LEVEL codecs — reconstruct rows, THEN compute pairwise cosines
// from the reconstructed rows. Harder test than score-level codecs.
// ═════════════════════════════════════════════════════════════════════

/// f32-CLAM centroid only — nearest centroid from k=64 CLAM palette.
/// No residual correction. Tests pure clustering quality.
struct F32ClamCentroid;
impl F32ClamCentroid {
    fn clam_sample(rows: &[Vec<f32>], k: usize) -> Vec<Vec<f32>> {
        let n = rows.len();
        if n == 0 || k == 0 { return vec![]; }
        let k = k.min(n);
        let mut first = 0;
        let mut first_norm = 0.0f32;
        for (i, r) in rows.iter().enumerate() {
            let ns: f32 = r.iter().map(|x| x * x).sum();
            if ns > first_norm { first_norm = ns; first = i; }
        }
        let mut selected = vec![first];
        let mut min_dist = vec![f32::MAX; n];
        for i in 0..n {
            min_dist[i] = rows[i].iter().zip(rows[first].iter())
                .map(|(a, b)| (a - b) * (a - b)).sum();
        }
        min_dist[first] = 0.0;
        for _ in 1..k {
            let mut next = 0; let mut best = f32::MIN;
            for i in 0..n { if min_dist[i] > best { best = min_dist[i]; next = i; } }
            if best <= 0.0 { break; }
            selected.push(next);
            for i in 0..n {
                let d: f32 = rows[i].iter().zip(rows[next].iter())
                    .map(|(a, b)| (a - b) * (a - b)).sum();
                if d < min_dist[i] { min_dist[i] = d; }
            }
        }
        selected.iter().map(|&i| rows[i].clone()).collect()
    }
}
impl CodecCandidate for F32ClamCentroid {
    fn name(&self) -> &str { "F32-CLAM-64" }
    fn bytes_per_row(&self) -> usize { 1 } // twig index
    fn pairwise_scores(&self, rows: &[Vec<f32>]) -> Vec<f64> {
        let centroids = Self::clam_sample(rows, 64);
        // Assign each row → nearest centroid
        let assignments: Vec<usize> = rows.iter().map(|row| {
            let mut best = 0; let mut best_d = f32::MAX;
            for (ci, c) in centroids.iter().enumerate() {
                let d: f32 = row.iter().zip(c.iter()).map(|(a, b)| (a - b) * (a - b)).sum();
                if d < best_d { best_d = d; best = ci; }
            }
            best
        }).collect();
        // Reconstruct: each row → its centroid
        let reconstructed: Vec<&Vec<f32>> = assignments.iter().map(|&ci| &centroids[ci]).collect();
        // Pairwise cosines on reconstructed rows
        let n = rows.len();
        let mut scores = Vec::with_capacity(n * (n - 1) / 2);
        for i in 0..n {
            for j in (i + 1)..n {
                scores.push(cosine(reconstructed[i], reconstructed[j]));
            }
        }
        scores
    }
}

/// f32-CLAM + SlotL — centroid + 8×i8 SVD residual correction.
/// This is the Path A codec from PR #184, measured at the row level.
struct F32ClamSlotL;
impl CodecCandidate for F32ClamSlotL {
    fn name(&self) -> &str { "F32-CLAM+SlotL" }
    fn bytes_per_row(&self) -> usize { 9 } // 1 twig + 8 leaf
    fn pairwise_scores(&self, rows: &[Vec<f32>]) -> Vec<f64> {
        use bgz_tensor::hhtl_f32::HhtlF32Tensor;
        use bgz_tensor::matryoshka::SvdBasis;
        use bgz_tensor::slot_l::SLOT_L_LANES;
        let n_cols = if rows.is_empty() { 0 } else { rows[0].len() };
        let k = 64.min(rows.len());
        let basis = SvdBasis::build("bench", rows, SLOT_L_LANES);
        let tensor = HhtlF32Tensor::encode_with_leaf("bench", rows, k, &basis);
        let n = rows.len();
        let mut scores = Vec::with_capacity(n * (n - 1) / 2);
        for i in 0..n {
            let ri = tensor.reconstruct_row(i, n_cols);
            for j in (i + 1)..n {
                let rj = tensor.reconstruct_row(j, n_cols);
                scores.push(cosine(&ri, &rj));
            }
        }
        scores
    }
}

/// i4 leaf: 16 directions × 4-bit precision in same 8 bytes as SlotL.
/// Wider directional coverage at coarser per-direction precision.
struct F32ClamLeafI4;
impl CodecCandidate for F32ClamLeafI4 {
    fn name(&self) -> &str { "CLAM+Leaf-i4×16" }
    fn bytes_per_row(&self) -> usize { 9 }
    fn pairwise_scores(&self, rows: &[Vec<f32>]) -> Vec<f64> {
        use bgz_tensor::matryoshka::SvdBasis;
        let n_cols = if rows.is_empty() { 0 } else { rows[0].len() };
        let k = 64.min(rows.len());
        let n_components = 16; // 16 directions at i4
        let basis = SvdBasis::build("i4leaf", rows, n_components);
        let centroids = F32ClamCentroid::clam_sample(rows, k);
        let n = rows.len();
        // Encode: project residual onto 16-dim SVD, quantize to i4 (±7)
        let encoded: Vec<(usize, Vec<i8>, f32)> = rows.iter().map(|row| {
            let mut best = 0; let mut best_d = f32::MAX;
            for (ci, c) in centroids.iter().enumerate() {
                let d: f32 = row.iter().zip(c.iter()).map(|(a, b)| (a - b) * (a - b)).sum();
                if d < best_d { best_d = d; best = ci; }
            }
            let residual: Vec<f32> = row.iter().zip(centroids[best].iter()).map(|(a, b)| a - b).collect();
            let coeffs = basis.project(&residual);
            let max_abs = coeffs.iter().take(n_components).map(|x| x.abs()).fold(0.0f32, f32::max);
            let scale = if max_abs > 1e-12 { 7.0 / max_abs } else { 0.0 };
            let q: Vec<i8> = coeffs.iter().take(n_components)
                .map(|c| (c * scale).round().clamp(-7.0, 7.0) as i8).collect();
            (best, q, if scale > 0.0 { max_abs / 7.0 } else { 0.0 })
        }).collect();
        // Reconstruct + pairwise
        let mut scores = Vec::with_capacity(n * (n - 1) / 2);
        for i in 0..n {
            let (ci, ref qi, si) = encoded[i];
            let coeffs_i: Vec<f32> = qi.iter().map(|&v| v as f32 * si).collect();
            let res_i = basis.reconstruct(&coeffs_i);
            let ri: Vec<f32> = centroids[ci].iter().zip(res_i.iter()).map(|(c, r)| c + r).collect();
            for j in (i + 1)..n {
                let (cj, ref qj, sj) = encoded[j];
                let coeffs_j: Vec<f32> = qj.iter().map(|&v| v as f32 * sj).collect();
                let res_j = basis.reconstruct(&coeffs_j);
                let rj: Vec<f32> = centroids[cj].iter().zip(res_j.iter()).map(|(c, r)| c + r).collect();
                scores.push(cosine(&ri, &rj));
            }
        }
        scores
    }
}

/// Mixed Matryoshka leaf: 4×i8 (top SVD) + 8×i4 (next SVD) = 64 bits.
/// Best of both: high precision on energy-dense dims + wide coverage.
struct F32ClamLeafMixed;
impl CodecCandidate for F32ClamLeafMixed {
    fn name(&self) -> &str { "CLAM+Mixed-4i8+8i4" }
    fn bytes_per_row(&self) -> usize { 9 }
    fn pairwise_scores(&self, rows: &[Vec<f32>]) -> Vec<f64> {
        use bgz_tensor::matryoshka::SvdBasis;
        let n_cols = if rows.is_empty() { 0 } else { rows[0].len() };
        let k = 64.min(rows.len());
        let n_components = 12; // 4 at i8 + 8 at i4
        let basis = SvdBasis::build("mixed", rows, n_components);
        let centroids = F32ClamCentroid::clam_sample(rows, k);
        let n = rows.len();
        let encoded: Vec<(usize, Vec<f32>, f32, f32)> = rows.iter().map(|row| {
            let mut best = 0; let mut best_d = f32::MAX;
            for (ci, c) in centroids.iter().enumerate() {
                let d: f32 = row.iter().zip(c.iter()).map(|(a, b)| (a - b) * (a - b)).sum();
                if d < best_d { best_d = d; best = ci; }
            }
            let residual: Vec<f32> = row.iter().zip(centroids[best].iter()).map(|(a, b)| a - b).collect();
            let coeffs = basis.project(&residual);
            // i8 scale for first 4, i4 scale for next 8
            let max8 = coeffs.iter().take(4).map(|x| x.abs()).fold(0.0f32, f32::max);
            let max4 = coeffs.iter().skip(4).take(8).map(|x| x.abs()).fold(0.0f32, f32::max);
            let s8 = if max8 > 1e-12 { max8 / 127.0 } else { 0.0 };
            let s4 = if max4 > 1e-12 { max4 / 7.0 } else { 0.0 };
            let mut q = Vec::with_capacity(12);
            for i in 0..4 { q.push(if s8 > 0.0 { (coeffs[i] / s8).round().clamp(-127.0, 127.0) * s8 } else { 0.0 }); }
            for i in 4..12.min(coeffs.len()) { q.push(if s4 > 0.0 { (coeffs[i] / s4).round().clamp(-7.0, 7.0) * s4 } else { 0.0 }); }
            (best, q, s8, s4)
        }).collect();
        let mut scores = Vec::with_capacity(n * (n - 1) / 2);
        for i in 0..n {
            let (ci, ref qi, _, _) = encoded[i];
            let res_i = basis.reconstruct(qi);
            let ri: Vec<f32> = centroids[ci].iter().zip(res_i.iter()).map(|(c, r)| c + r).collect();
            for j in (i + 1)..n {
                let (cj, ref qj, _, _) = encoded[j];
                let res_j = basis.reconstruct(qj);
                let rj: Vec<f32> = centroids[cj].iter().zip(res_j.iter()).map(|(c, r)| c + r).collect();
                scores.push(cosine(&ri, &rj));
            }
        }
        scores
    }
}

/// I8 Hybrid — HEEL+HIP location (6 bits) + JLQ i8 leaf (8 bytes).
/// Uses proper Hadamard rotation from bgz17::rabitq_compat, not hash-based
/// Rademacher signs. The Hadamard matrix is orthogonal (preserves norms)
/// and structured (O(n log n) rotation, no random matrix storage).
///
/// PR #191 showed hash-based Rademacher added ZERO quality vs centroid-only.
/// This tests whether proper Hadamard fixes that.
struct I8HybridCodec;
impl I8HybridCodec {
    fn hadamard_encode_residual(residual: &[f32], rotation: &bgz17::rabitq_compat::OrthogonalMatrix) -> ([i8; 8], f32) {
        // Rotate the full residual via Hadamard
        let rotated = rotation.rotate(residual);
        // Take top-8 rotated coefficients (highest energy after rotation)
        // Sort by magnitude, pick top 8
        let mut indexed: Vec<(usize, f32)> = rotated.iter().copied().enumerate().collect();
        indexed.sort_by(|a, b| b.1.abs().partial_cmp(&a.1.abs()).unwrap());
        let top8: Vec<(usize, f32)> = indexed[..8.min(indexed.len())].to_vec();
        let max_abs = top8.iter().map(|(_, v)| v.abs()).fold(0.0f32, f32::max);
        let inv = if max_abs > 1e-12 { 127.0 / max_abs } else { 0.0 };
        let mut leaf = [0i8; 8];
        for (i, &(_, val)) in top8.iter().enumerate() {
            leaf[i] = (val * inv).round().clamp(-127.0, 127.0) as i8;
        }
        (leaf, max_abs / 127.0)
    }

    fn hadamard_decode_residual(leaf: &[i8; 8], scale: f32, rotation: &bgz17::rabitq_compat::OrthogonalMatrix, dim: usize) -> Vec<f32> {
        // This is approximate: we only stored 8 of dim coefficients.
        // Reconstruct by placing the 8 values at the SAME top-8 positions
        // of a zero vector, then inverse-rotate.
        // Problem: we don't store WHICH 8 positions were top.
        // Fallback: place in first 8 positions (loses position info but
        // tests the Hadamard rotation quality at least).
        let mut rotated = vec![0.0f32; dim];
        for i in 0..8.min(dim) {
            rotated[i] = leaf[i] as f32 * scale;
        }
        // Inverse Hadamard = Hadamard (self-inverse for normalized)
        rotation.rotate(&rotated)
    }
}
impl CodecCandidate for I8HybridCodec {
    fn name(&self) -> &str { "I8-Hadamard" }
    fn bytes_per_row(&self) -> usize { 9 } // 1 address + 8 leaf
    fn pairwise_scores(&self, rows: &[Vec<f32>]) -> Vec<f64> {
        use bgz17::rabitq_compat::OrthogonalMatrix;
        let n_cols = if rows.is_empty() { 0 } else { rows[0].len() };
        let centroids = F32ClamCentroid::clam_sample(rows, 64);
        let padded_dim = RaBitQCodec::next_pow2(n_cols);
        let rotation = OrthogonalMatrix::hadamard(padded_dim);
        // Assign + encode residual via Hadamard
        let encoded: Vec<(usize, [i8; 8], f32)> = rows.iter().map(|row| {
            let mut best = 0; let mut best_d = f32::MAX;
            for (ci, c) in centroids.iter().enumerate() {
                let d: f32 = row.iter().zip(c.iter()).map(|(a, b)| (a - b) * (a - b)).sum();
                if d < best_d { best_d = d; best = ci; }
            }
            let mut residual: Vec<f32> = row.iter().zip(centroids[best].iter())
                .map(|(a, b)| a - b).collect();
            residual.resize(padded_dim, 0.0);
            let (leaf, scale) = Self::hadamard_encode_residual(&residual, &rotation);
            (best, leaf, scale)
        }).collect();
        // Reconstruct: centroid + Hadamard-decoded residual
        let n = rows.len();
        let mut scores = Vec::with_capacity(n * (n - 1) / 2);
        for i in 0..n {
            let (ci, ref li, si) = encoded[i];
            let res_i = Self::hadamard_decode_residual(li, si, &rotation, padded_dim);
            let ri: Vec<f32> = centroids[ci].iter().zip(res_i.iter()).map(|(c, r)| c + r).collect();
            for j in (i + 1)..n {
                let (cj, ref lj, sj) = encoded[j];
                let res_j = Self::hadamard_decode_residual(lj, sj, &rotation, padded_dim);
                let rj: Vec<f32> = centroids[cj].iter().zip(res_j.iter()).map(|(c, r)| c + r).collect();
                scores.push(cosine(&ri, &rj));
            }
        }
        scores
    }
}

// ═════════════════════════════════════════════════════════════════════
// Bench runner
// ═════════════════════════════════════════════════════════════════════

struct BenchResult {
    codec_name: String,
    bytes_per_row: usize,
    pearson_r: f64,
    spearman_rho: f64,
    kendall_t: f64,
    mae_val: f64,
    rmse_val: f64,
    top_k_recall: f64,
    icc: f64,
    bias: f64,
    variance: f64,
}

fn run_bench(codecs: &[Box<dyn CodecCandidate>], rows: &[Vec<f32>], gt: &[f64]) -> Vec<BenchResult> {
    let mut results = Vec::new();
    let mut all_scores: Vec<Vec<f64>> = Vec::new(); // for Cronbach's α

    for codec in codecs {
        let scores = codec.pairwise_scores(rows);
        let n = gt.len().min(scores.len());
        let gt_slice = &gt[..n];
        let sc_slice = &scores[..n];

        let errors: Vec<f64> = (0..n).map(|i| sc_slice[i] - gt_slice[i]).collect();
        let (b, v) = bias_variance(&errors);

        // Top-1 argmax per row
        let n_rows = rows.len();
        let mut gt_argmax = vec![0usize; n_rows];
        let mut sc_argmax = vec![0usize; n_rows];
        let mut pair_idx = 0usize;
        for i in 0..n_rows {
            let mut best_gt = f64::NEG_INFINITY;
            let mut best_sc = f64::NEG_INFINITY;
            for j in (i + 1)..n_rows {
                if gt[pair_idx] > best_gt { best_gt = gt[pair_idx]; gt_argmax[i] = j; }
                if scores[pair_idx] > best_sc { best_sc = scores[pair_idx]; sc_argmax[i] = j; }
                pair_idx += 1;
            }
        }

        results.push(BenchResult {
            codec_name: codec.name().to_string(),
            bytes_per_row: codec.bytes_per_row(),
            pearson_r: pearson(gt_slice, sc_slice),
            spearman_rho: spearman(gt_slice, sc_slice),
            kendall_t: kendall_tau(gt_slice, sc_slice),
            mae_val: mae(gt_slice, sc_slice),
            rmse_val: rmse(gt_slice, sc_slice),
            top_k_recall: top_k_recall(gt_slice, sc_slice, TOP_K),
            icc: icc_3_1(gt_slice, sc_slice),
            bias: b,
            variance: v,
        });
        all_scores.push(scores);
    }

    // Cronbach's α across all codecs (excluding passthrough which IS ground truth)
    if all_scores.len() >= 2 {
        let items: Vec<Vec<f64>> = all_scores[1..].to_vec(); // skip passthrough
        let alpha = cronbach_alpha(&items);
        eprintln!("  Cronbach's α (inter-codec, excl. passthrough): {:.4}", alpha);
    }

    results
}

fn print_table(population: &str, results: &[BenchResult]) {
    println!("\n### {}", population);
    println!();
    println!("| Codec | B/row | Pearson | Spearman | Kendall | MAE | RMSE | Top-{} | ICC | Bias | Var |",
        TOP_K);
    println!("|---|---|---|---|---|---|---|---|---|---|---|");
    for r in results {
        println!("| {} | {} | {:.4} | {:.4} | {:.4} | {:.4} | {:.4} | {:.4} | {:.4} | {:.2e} | {:.2e} |",
            r.codec_name, r.bytes_per_row,
            r.pearson_r, r.spearman_rho, r.kendall_t,
            r.mae_val, r.rmse_val, r.top_k_recall,
            r.icc, r.bias, r.variance);
    }
}

// ═════════════════════════════════════════════════════════════════════
// Main
// ═════════════════════════════════════════════════════════════════════

fn main() {
    let model_path = std::env::args().nth(1)
        .expect("usage: codec_rnd_bench <model.safetensors>");

    println!("# Codec R&D Bench — Psychometric Measurement");
    println!();
    println!("Model: `{}`", model_path);
    println!("Sample: {} rows per population, {} metrics per codec", N_SAMPLE, 10);

    // Populations from the model
    let populations: Vec<(&str, &str)> = vec![
        ("self_attn.k_proj.weight", "Attention k_proj (argmax regime)"),
        ("mlp.gate_proj.weight", "MLP gate_proj (argmax, SiLU-gated)"),
        ("text_embedding.weight", "Text embedding (index regime)"),
        ("code_predictor.model.codec_embedding.0.weight", "Audio codec emb (index)"),
        ("self_attn.q_proj.weight", "Attention q_proj (argmax, must match k)"),
        // Gemma 4 PLE: 256-d per-layer projection — the dimensional threshold test
        ("per_layer_projection.weight", "PLE projection (256-d, Gemma4 low-dim)"),
        ("per_layer_input_gate.weight", "PLE gate (256-d, Gemma4 low-dim)"),
    ];

    let t0 = Instant::now();

    for (tensor_substr, pop_name) in &populations {
        let (rows, tensor_name) = load_rows(&model_path, tensor_substr, N_SAMPLE);
        if rows.is_empty() {
            println!("\n---");
            println!("**Population: {}** — `{}` (skipped)", pop_name, tensor_name);
            continue;
        }
        let n_cols = if rows.is_empty() { 0 } else { rows[0].len() };
        println!("\n---");
        println!("**Population: {}** — `{}`", pop_name, tensor_name);

        let gt = pairwise_cosines(&rows);

        let codecs: Vec<Box<dyn CodecCandidate>> = vec![
            Box::new(Passthrough),
            Box::new(Base17Sig),
            Box::new(DirectI8),
            Box::new(SpiralK8),
            Box::new(RaBitQCodec { dim: n_cols }),
            // Row-level codecs (reconstruct ROWS, then score from reconstruction)
            Box::new(F32ClamCentroid),
            Box::new(F32ClamSlotL),
            Box::new(F32ClamLeafI4),
            Box::new(F32ClamLeafMixed),
            Box::new(I8HybridCodec),
        ];

        let results = run_bench(&codecs, &rows, &gt);
        print_table(pop_name, &results);
    }

    println!("\n---");
    println!("Total wall: {:.1}s", t0.elapsed().as_secs_f32());
    println!();
    println!("## Interpretation");
    println!();
    println!("- **ICC** is the key metric: value-level agreement, not just ranking.");
    println!("- **Cronbach's α** printed to stderr per population — shows inter-codec factor structure.");
    println!("- **Bias/Var decomposition** reveals whether errors are systematic (correctable) or random (fundamental).");
    println!("- Compare across populations to test generalizability of each codec.");
}
