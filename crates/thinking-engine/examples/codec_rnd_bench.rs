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
    cronbach_alpha, bias_variance, icc_3_1,
};
use bgz_tensor::projection::Base17;
use highheelbgz::rehydrate::SpiralEncoding;
use ndarray::hpc::safetensors::read_safetensors_header;
use ndarray::simd::bf16_to_f32_batch;

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

// ── Lab / R&D: fractal descriptor codec ──

/// Fractal descriptor alone (7 B): encode each row by its MFDFA
/// parameters (D, w, σ, H) on the Hadamard-rotated coefficient
/// sequence. Pairwise "cosine" = similarity in descriptor space.
///
/// Expected to be weak: probe showed fractal magnitude statistics
/// are near-constant across Qwen3 rows (CoV(w) ≈ 0.19). This codec
/// measures that empirically via ICC_3_1.
#[cfg(feature = "lab")]
struct FractalDescOnly;

#[cfg(feature = "lab")]
impl CodecCandidate for FractalDescOnly {
    fn name(&self) -> &str { "Fractal-Desc(7B)" }
    fn bytes_per_row(&self) -> usize { 7 }
    fn pairwise_scores(&self, rows: &[Vec<f32>]) -> Vec<f64> {
        use bgz_tensor::fractal_descriptor::compute_mfdfa_descriptor;
        // Pad each row to a power-of-2, compute descriptor.
        let descs: Vec<[f32; 4]> = rows.iter().map(|r| {
            let n = r.len();
            let mut p = 1usize;
            while p < n { p <<= 1; }
            let mut buf = vec![0.0f32; p];
            buf[..n].copy_from_slice(r);
            let d = compute_mfdfa_descriptor(&buf);
            [d.d_local_f32(), d.w_mfs_f32(), d.sigma_energy_f32(), d.h_hurst_f32()]
        }).collect();
        let n = rows.len();
        let mut scores = Vec::with_capacity(n * (n - 1) / 2);
        for i in 0..n {
            for j in (i + 1)..n {
                // Normalized cosine between 4-D descriptors.
                let a = &descs[i];
                let b = &descs[j];
                let mut dot = 0.0f64;
                let mut na = 0.0f64;
                let mut nb = 0.0f64;
                for k in 0..4 {
                    dot += a[k] as f64 * b[k] as f64;
                    na += (a[k] as f64).powi(2);
                    nb += (b[k] as f64).powi(2);
                }
                let d = (na * nb).sqrt();
                scores.push(if d < 1e-15 { 0.0 } else { dot / d });
            }
        }
        scores
    }
}

/// Base17 (34 B anchors, phase signal) + FractalDescriptor (7 B shape).
/// = 41 B/row. Pairwise score blends Base17 cosine with descriptor
/// similarity weighted by complementary correlation. This is the
/// operational form of the "fractal leaf on golden-step anchors" concept.
#[cfg(feature = "lab")]
struct FractalPlusBase17;

#[cfg(feature = "lab")]
impl CodecCandidate for FractalPlusBase17 {
    fn name(&self) -> &str { "Fractal+Base17(41B)" }
    fn bytes_per_row(&self) -> usize { 41 }
    fn pairwise_scores(&self, rows: &[Vec<f32>]) -> Vec<f64> {
        use bgz_tensor::fractal_descriptor::compute_mfdfa_descriptor;
        let b17s: Vec<Base17> = rows.iter().map(|r| Base17::from_f32(r)).collect();
        let descs: Vec<[f32; 4]> = rows.iter().map(|r| {
            let n = r.len();
            let mut p = 1usize;
            while p < n { p <<= 1; }
            let mut buf = vec![0.0f32; p];
            buf[..n].copy_from_slice(r);
            let d = compute_mfdfa_descriptor(&buf);
            [d.d_local_f32(), d.w_mfs_f32(), d.sigma_energy_f32(), d.h_hurst_f32()]
        }).collect();
        let n = rows.len();
        let mut scores = Vec::with_capacity(n * (n - 1) / 2);
        for i in 0..n {
            for j in (i + 1)..n {
                let c_b17 = b17s[i].cosine(&b17s[j]);
                // Descriptor similarity (L2-normalized).
                let a = &descs[i]; let b = &descs[j];
                let mut dot = 0.0f64; let mut na = 0.0f64; let mut nb = 0.0f64;
                for k in 0..4 { dot += a[k] as f64 * b[k] as f64; na += (a[k] as f64).powi(2); nb += (b[k] as f64).powi(2); }
                let d_norm = (na * nb).sqrt();
                let c_desc = if d_norm < 1e-15 { 0.0 } else { dot / d_norm };
                // Simple blend: 0.75 anchors + 0.25 shape.
                scores.push(0.75 * c_b17 + 0.25 * c_desc);
            }
        }
        scores
    }
}

/// Phase-only (5 B): fractal statistics of the SIGN SEQUENCE
/// post-Hadamard. 5-D sign-flip density profile at scales 4/8/16/32/64.
/// Tests whether phase structure (not magnitude) distinguishes rows.
#[cfg(feature = "lab")]
struct FractalPhaseOnly;

#[cfg(feature = "lab")]
impl CodecCandidate for FractalPhaseOnly {
    fn name(&self) -> &str { "Fractal-Phase(5B)" }
    fn bytes_per_row(&self) -> usize { 5 }
    fn pairwise_scores(&self, rows: &[Vec<f32>]) -> Vec<f64> {
        use bgz_tensor::fractal_descriptor::PhaseDescriptor;
        let phases: Vec<PhaseDescriptor> = rows.iter().map(|r| {
            let n = r.len();
            let mut p = 1usize;
            while p < n { p <<= 1; }
            let mut buf = vec![0.0f32; p];
            buf[..n].copy_from_slice(r);
            PhaseDescriptor::from_row(&buf)
        }).collect();
        let n = rows.len();
        let mut scores = Vec::with_capacity(n * (n - 1) / 2);
        for i in 0..n {
            for j in (i + 1)..n {
                scores.push(phases[i].cosine(&phases[j]) as f64);
            }
        }
        scores
    }
}

/// Phase + Base17 (39 B): golden-step anchors + sign-sequence fractal.
/// Anchors carry partial phase (signs at 17 positions); fractal carries
/// multi-scale phase density. Tests whether combined beats Base17 alone.
#[cfg(feature = "lab")]
struct FractalPhasePlusBase17;

#[cfg(feature = "lab")]
impl CodecCandidate for FractalPhasePlusBase17 {
    fn name(&self) -> &str { "Phase+Base17(39B)" }
    fn bytes_per_row(&self) -> usize { 39 }
    fn pairwise_scores(&self, rows: &[Vec<f32>]) -> Vec<f64> {
        use bgz_tensor::fractal_descriptor::PhaseDescriptor;
        let b17s: Vec<Base17> = rows.iter().map(|r| Base17::from_f32(r)).collect();
        let phases: Vec<PhaseDescriptor> = rows.iter().map(|r| {
            let n = r.len();
            let mut p = 1usize;
            while p < n { p <<= 1; }
            let mut buf = vec![0.0f32; p];
            buf[..n].copy_from_slice(r);
            PhaseDescriptor::from_row(&buf)
        }).collect();
        let n = rows.len();
        let mut scores = Vec::with_capacity(n * (n - 1) / 2);
        for i in 0..n {
            for j in (i + 1)..n {
                let c_b17 = b17s[i].cosine(&b17s[j]);
                let c_phase = phases[i].cosine(&phases[j]) as f64;
                scores.push(0.75 * c_b17 + 0.25 * c_phase);
            }
        }
        scores
    }
}

/// Zipper codec — phase + magnitude φ-multiplexed in single container.
/// Phase stream: 64 sign bits at round(N/φ) stride.
/// Magnitude stream: 56 i8 samples at round(N/φ²) stride.
/// Total: 64 B. Matryoshka: phase-only (8 B level) + full (64 B level).
#[cfg(feature = "lab")]
struct ZipperPhaseOnly;

#[cfg(feature = "lab")]
impl CodecCandidate for ZipperPhaseOnly {
    fn name(&self) -> &str { "Zipper-Phase(8B)" }
    fn bytes_per_row(&self) -> usize { 8 }
    fn pairwise_scores(&self, rows: &[Vec<f32>]) -> Vec<f64> {
        use bgz_tensor::zipper::ZipperDescriptor;
        let zs: Vec<ZipperDescriptor> = rows.iter().map(|r| {
            let n = r.len();
            let mut p = 1usize;
            while p < n { p <<= 1; }
            let mut buf = vec![0.0f32; p];
            buf[..n].copy_from_slice(r);
            ZipperDescriptor::encode(&buf)
        }).collect();
        let n = rows.len();
        let mut scores = Vec::with_capacity(n * (n - 1) / 2);
        for i in 0..n {
            for j in (i + 1)..n {
                scores.push(zs[i].cosine_phase_only(&zs[j]) as f64);
            }
        }
        scores
    }
}

#[cfg(feature = "lab")]
struct ZipperFull;

#[cfg(feature = "lab")]
impl CodecCandidate for ZipperFull {
    fn name(&self) -> &str { "Zipper-Full(64B)" }
    fn bytes_per_row(&self) -> usize { 64 }
    fn pairwise_scores(&self, rows: &[Vec<f32>]) -> Vec<f64> {
        use bgz_tensor::zipper::ZipperDescriptor;
        let zs: Vec<ZipperDescriptor> = rows.iter().map(|r| {
            let n = r.len();
            let mut p = 1usize;
            while p < n { p <<= 1; }
            let mut buf = vec![0.0f32; p];
            buf[..n].copy_from_slice(r);
            ZipperDescriptor::encode(&buf)
        }).collect();
        let n = rows.len();
        let mut scores = Vec::with_capacity(n * (n - 1) / 2);
        for i in 0..n {
            for j in (i + 1)..n {
                scores.push(zs[i].cosine_zipper_full(&zs[j]) as f64);
            }
        }
        scores
    }
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

/// Base17Fz — Fisher z warped golden fold. Same 34 bytes, non-linear quantization.
struct Base17FzSig;
impl CodecCandidate for Base17FzSig {
    fn name(&self) -> &str { "Base17Fz" }
    fn bytes_per_row(&self) -> usize { 34 }
    fn pairwise_scores(&self, rows: &[Vec<f32>]) -> Vec<f64> {
        use bgz_tensor::projection::Base17Fz;
        let fzs: Vec<Base17Fz> = rows.iter().map(|r| Base17Fz::from_f32(r)).collect();
        let n = rows.len();
        let mut scores = Vec::with_capacity(n * (n - 1) / 2);
        for i in 0..n {
            for j in (i + 1)..n {
                scores.push(fzs[i].cosine(&fzs[j]));
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

/// Full-rank i4 leaf: n_components = min(16384, n_cols) at i4 precision.
/// At 256-d this means ALL 256 SVD directions at 4-bit = 128 bytes/row.
/// Tests whether full-rank i4 reaches product ICC (≥0.85).
struct F32ClamFullI4;
impl CodecCandidate for F32ClamFullI4 {
    fn name(&self) -> &str { "CLAM+i4×D" }
    fn bytes_per_row(&self) -> usize { 0 } // computed dynamically, shown in output
    fn pairwise_scores(&self, rows: &[Vec<f32>]) -> Vec<f64> {
        use bgz_tensor::matryoshka::SvdBasis;
        let n_cols = if rows.is_empty() { 0 } else { rows[0].len() };
        let k = 64.min(rows.len());
        let n_components = 16384.min(n_cols); // full rank at 256-d
        let basis = SvdBasis::build("fullI4", rows, n_components);
        let centroids = F32ClamCentroid::clam_sample(rows, k);
        let actual_d = basis.n_components;
        let bpr = 1 + (actual_d + 1) / 2 + 4; // twig + nibbles + scale
        eprintln!("    [CLAM+i4×D] n_components={}, bytes/row={}", actual_d, bpr);
        let n = rows.len();
        let encoded: Vec<(usize, Vec<i8>, f32)> = rows.iter().map(|row| {
            let mut best = 0; let mut best_d = f32::MAX;
            for (ci, c) in centroids.iter().enumerate() {
                let d: f32 = row.iter().zip(c.iter()).map(|(a, b)| (a - b) * (a - b)).sum();
                if d < best_d { best_d = d; best = ci; }
            }
            let residual: Vec<f32> = row.iter().zip(centroids[best].iter()).map(|(a, b)| a - b).collect();
            let coeffs = basis.project(&residual);
            let max_abs = coeffs.iter().take(actual_d).map(|x| x.abs()).fold(0.0f32, f32::max);
            let scale = if max_abs > 1e-12 { 7.0 / max_abs } else { 0.0 };
            let q: Vec<i8> = coeffs.iter().take(actual_d)
                .map(|c| (c * scale).round().clamp(-7.0, 7.0) as i8).collect();
            (best, q, if scale > 0.0 { max_abs / 7.0 } else { 0.0 })
        }).collect();
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

/// Bitpacked-as-CAM: full-rank i4 codes used directly as addresses.
/// Instead of reconstructing → cosine, compute Manhattan distance on
/// quantized i4 codes. The bitpacked representation IS the address —
/// same principle as HHTL-D Slot D but at full dimensionality.
struct BitpackedCamI4;
impl CodecCandidate for BitpackedCamI4 {
    fn name(&self) -> &str { "i4-CAM" }
    fn bytes_per_row(&self) -> usize { 0 } // dynamic
    fn pairwise_scores(&self, rows: &[Vec<f32>]) -> Vec<f64> {
        use bgz_tensor::matryoshka::SvdBasis;
        let n_cols = if rows.is_empty() { 0 } else { rows[0].len() };
        let n_components = 16384.min(n_cols);
        let basis = SvdBasis::build("cam4", rows, n_components);
        let actual_d = basis.n_components;
        eprintln!("    [i4-CAM] n_components={}, {} bits/address", actual_d, actual_d * 4);
        let n = rows.len();
        // Encode: project full row (not residual) onto SVD, quantize to i4
        let codes: Vec<Vec<i8>> = rows.iter().map(|row| {
            let coeffs = basis.project(row);
            let max_abs = coeffs.iter().take(actual_d).map(|x| x.abs()).fold(0.0f32, f32::max);
            let scale = if max_abs > 1e-12 { 7.0 / max_abs } else { 0.0 };
            coeffs.iter().take(actual_d)
                .map(|c| (c * scale).round().clamp(-7.0, 7.0) as i8).collect()
        }).collect();
        // Pairwise: 1 - normalized Manhattan distance → cosine-like score
        let mut scores = Vec::with_capacity(n * (n - 1) / 2);
        let max_manhattan = (actual_d * 14) as f64; // max Manhattan on ±7 codes
        for i in 0..n {
            for j in (i + 1)..n {
                let manhattan: i32 = codes[i].iter().zip(codes[j].iter())
                    .map(|(a, b)| (*a as i32 - *b as i32).abs())
                    .sum();
                // Convert Manhattan distance to cosine-like similarity
                scores.push(1.0 - manhattan as f64 / max_manhattan);
            }
        }
        scores
    }
}

/// Matryoshka variable-precision: i16/i8/i4/i2 bands from bgz-tensor.
/// Uses the actual production codec path (encode_row/decode_row).
struct MatryoshkaCodec;
impl CodecCandidate for MatryoshkaCodec {
    fn name(&self) -> &str { "Matryoshka-std" }
    fn bytes_per_row(&self) -> usize { 0 }
    fn pairwise_scores(&self, rows: &[Vec<f32>]) -> Vec<f64> {
        use bgz_tensor::matryoshka::{SvdBasis, BandProfile, encode_matrix, decode_matrix};
        let n_cols = if rows.is_empty() { 0 } else { rows[0].len() };
        let n_components = n_cols.min(512); // standard profile caps at 512
        let basis = SvdBasis::build("matryoshka", rows, n_components);
        let profile = BandProfile::standard(basis.n_components, n_cols);
        eprintln!("    [Matryoshka] n_components={}, bytes/row={}", basis.n_components, profile.bytes_per_row());
        let encoded = encode_matrix(rows, &basis, &profile);
        let decoded = decode_matrix(&encoded, &basis, &profile);
        let n = rows.len();
        let mut scores = Vec::with_capacity(n * (n - 1) / 2);
        for i in 0..n {
            for j in (i + 1)..n {
                scores.push(cosine(&decoded[i], &decoded[j]));
            }
        }
        scores
    }
}

// ═════════════════════════════════════════════════════════════════════
// EPIPHANY CODECS — derived from the 62-codec sweep findings
// ═════════════════════════════════════════════════════════════════════

/// E1: Gain-shape Hadamard i4 CAM.
/// Normalize to unit norm BEFORE Hadamard rotation, store gain separately.
/// All codes now live on the same unit sphere → Manhattan ≈ angular distance.
struct GainShapeHadCam;
impl CodecCandidate for GainShapeHadCam {
    fn name(&self) -> &str { "GS-Had-i4-CAM" }
    fn bytes_per_row(&self) -> usize { 0 }
    fn pairwise_scores(&self, rows: &[Vec<f32>]) -> Vec<f64> {
        use bgz17::rabitq_compat::OrthogonalMatrix;
        let n = rows.len();
        if n == 0 { return vec![]; }
        let n_cols = rows[0].len();
        let padded = RaBitQCodec::next_pow2(n_cols);
        let rot = OrthogonalMatrix::hadamard(padded);

        let encoded: Vec<Vec<i8>> = rows.iter().map(|row| {
            let norm: f64 = row.iter().map(|x| (*x as f64) * (*x as f64)).sum::<f64>().sqrt();
            let inv_norm = if norm > 1e-15 { 1.0 / norm } else { 0.0 };
            let mut unit: Vec<f32> = row.iter().map(|x| *x * inv_norm as f32).collect();
            unit.resize(padded, 0.0);
            let rotated = rot.rotate(&unit);
            let max_abs = rotated.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
            let scale = if max_abs > 1e-12 { 7.0 / max_abs } else { 0.0 };
            rotated.iter().take(n_cols)
                .map(|c| (c * scale).round().clamp(-7.0, 7.0) as i8).collect()
        }).collect();

        let max_manhattan = (n_cols * 14) as f64;
        let mut scores = Vec::with_capacity(n * (n - 1) / 2);
        for i in 0..n {
            for j in (i + 1)..n {
                let manhattan: i32 = encoded[i].iter().zip(encoded[j].iter())
                    .map(|(a, b)| (*a as i32 - *b as i32).abs()).sum();
                scores.push(1.0 - manhattan as f64 / max_manhattan);
            }
        }
        scores
    }
}

/// E1b: Gain-shape Hadamard i4 CAM with shared global scale.
/// One scale for all rows (the global max), not per-row.
/// Codes are directly comparable — same scale everywhere.
struct SharedScaleHadCam;
impl CodecCandidate for SharedScaleHadCam {
    fn name(&self) -> &str { "SS-Had-i4-CAM" }
    fn bytes_per_row(&self) -> usize { 0 }
    fn pairwise_scores(&self, rows: &[Vec<f32>]) -> Vec<f64> {
        use bgz17::rabitq_compat::OrthogonalMatrix;
        let n = rows.len();
        if n == 0 { return vec![]; }
        let n_cols = rows[0].len();
        let padded = RaBitQCodec::next_pow2(n_cols);
        let rot = OrthogonalMatrix::hadamard(padded);

        // First pass: rotate all, find global max
        let rotated_rows: Vec<Vec<f32>> = rows.iter().map(|row| {
            let mut padded_row = row.to_vec();
            padded_row.resize(padded, 0.0);
            let r = rot.rotate(&padded_row);
            r[..n_cols].to_vec()
        }).collect();

        let global_max = rotated_rows.iter()
            .flat_map(|r| r.iter())
            .map(|x| x.abs())
            .fold(0.0f32, f32::max);
        let scale = if global_max > 1e-12 { 7.0 / global_max } else { 0.0 };

        let codes: Vec<Vec<i8>> = rotated_rows.iter().map(|r| {
            r.iter().map(|c| (c * scale).round().clamp(-7.0, 7.0) as i8).collect()
        }).collect();

        let max_manhattan = (n_cols * 14) as f64;
        let mut scores = Vec::with_capacity(n * (n - 1) / 2);
        for i in 0..n {
            for j in (i + 1)..n {
                let manhattan: i32 = codes[i].iter().zip(codes[j].iter())
                    .map(|(a, b)| (*a as i32 - *b as i32).abs()).sum();
                scores.push(1.0 - manhattan as f64 / max_manhattan);
            }
        }
        scores
    }
}

/// E2: Hadamard i4 + i2 residue cascade.
/// First pass: Had-i4×D reconstruction (ICC 0.993).
/// Second pass: residual from i4 → Had rotate again → i2 full-rank.
/// Tests whether the last 0.7% gap is closeable.
struct HadI4PlusI2Residue;
impl CodecCandidate for HadI4PlusI2Residue {
    fn name(&self) -> &str { "Had-i4+i2res" }
    fn bytes_per_row(&self) -> usize { 0 }
    fn pairwise_scores(&self, rows: &[Vec<f32>]) -> Vec<f64> {
        use bgz17::rabitq_compat::OrthogonalMatrix;
        let n = rows.len();
        if n == 0 { return vec![]; }
        let n_cols = rows[0].len();
        let padded = RaBitQCodec::next_pow2(n_cols);
        let rot = OrthogonalMatrix::hadamard(padded);

        // CLAM centroids
        let centroids = F32ClamCentroid::clam_sample(rows, 64.min(n));

        let recon_rows: Vec<Vec<f32>> = rows.iter().map(|row| {
            // Centroid assignment
            let mut best = 0; let mut best_d = f32::MAX;
            for (ci, c) in centroids.iter().enumerate() {
                let d: f32 = row.iter().zip(c.iter()).map(|(a, b)| (a - b) * (a - b)).sum();
                if d < best_d { best_d = d; best = ci; }
            }

            // Pass 1: i4 on residual from centroid
            let residual: Vec<f32> = row.iter().zip(centroids[best].iter()).map(|(a, b)| a - b).collect();
            let mut pad_res = residual.clone();
            pad_res.resize(padded, 0.0);
            let rotated = rot.rotate(&pad_res);
            let max1 = rotated.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
            let s1 = if max1 > 1e-12 { 7.0 / max1 } else { 0.0 };
            let q1: Vec<i8> = rotated.iter().take(n_cols)
                .map(|c| (c * s1).round().clamp(-7.0, 7.0) as i8).collect();
            let ds1 = if s1 > 0.0 { max1 / 7.0 } else { 0.0 };

            // Reconstruct pass 1
            let mut full1 = vec![0.0f32; padded];
            for (k, &q) in q1.iter().enumerate() { full1[k] = q as f32 * ds1; }
            let recon1 = rot.rotate(&full1);

            // Pass 2: i2 on residual from pass 1
            let residual2: Vec<f32> = residual.iter().zip(recon1.iter().take(n_cols))
                .map(|(orig, r1)| orig - r1).collect();
            let mut pad_res2 = residual2;
            pad_res2.resize(padded, 0.0);
            let rotated2 = rot.rotate(&pad_res2);
            let max2 = rotated2.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
            let s2 = if max2 > 1e-12 { 1.0 / max2 } else { 0.0 };
            let q2: Vec<i8> = rotated2.iter().take(n_cols)
                .map(|c| (c * s2).round().clamp(-1.0, 1.0) as i8).collect();
            let ds2 = if s2 > 0.0 { max2 / 1.0 } else { 0.0 };

            // Reconstruct pass 2
            let mut full2 = vec![0.0f32; padded];
            for (k, &q) in q2.iter().enumerate() { full2[k] = q as f32 * ds2; }
            let recon2 = rot.rotate(&full2);

            // Final: centroid + pass1 + pass2
            centroids[best].iter()
                .zip(recon1.iter())
                .zip(recon2.iter())
                .map(|((c, r1), r2)| c + r1 + r2)
                .collect()
        }).collect();

        let mut scores = Vec::with_capacity(n * (n - 1) / 2);
        for i in 0..n {
            for j in (i + 1)..n {
                scores.push(cosine(&recon_rows[i], &recon_rows[j]));
            }
        }
        scores
    }
}

/// E3: Fisher z on DISTANCE, not coefficients.
/// Encode with Had-i4×D (linear). Reconstruct rows. Compute pairwise cosine.
/// Apply Fisher z: z = arctanh(cos) to the cosine scores.
/// This tests perceptually uniform distance scaling.
struct FisherZOnDistance;
impl CodecCandidate for FisherZOnDistance {
    fn name(&self) -> &str { "Had-i4+FzDist" }
    fn bytes_per_row(&self) -> usize { 0 }
    fn pairwise_scores(&self, rows: &[Vec<f32>]) -> Vec<f64> {
        use bgz17::rabitq_compat::OrthogonalMatrix;
        let n = rows.len();
        if n == 0 { return vec![]; }
        let n_cols = rows[0].len();
        let padded = RaBitQCodec::next_pow2(n_cols);
        let rot = OrthogonalMatrix::hadamard(padded);
        let centroids = F32ClamCentroid::clam_sample(rows, 64.min(n));

        // Had-i4 reconstruction (same as Had-i4×D-R)
        let recon_rows: Vec<Vec<f32>> = rows.iter().map(|row| {
            let mut best = 0; let mut best_d = f32::MAX;
            for (ci, c) in centroids.iter().enumerate() {
                let d: f32 = row.iter().zip(c.iter()).map(|(a, b)| (a - b) * (a - b)).sum();
                if d < best_d { best_d = d; best = ci; }
            }
            let residual: Vec<f32> = row.iter().zip(centroids[best].iter()).map(|(a, b)| a - b).collect();
            let mut pad_res = residual;
            pad_res.resize(padded, 0.0);
            let rotated = rot.rotate(&pad_res);
            let max_abs = rotated.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
            let scale = if max_abs > 1e-12 { 7.0 / max_abs } else { 0.0 };
            let codes: Vec<i8> = rotated.iter().take(n_cols)
                .map(|c| (c * scale).round().clamp(-7.0, 7.0) as i8).collect();
            let ds = if scale > 0.0 { max_abs / 7.0 } else { 0.0 };
            let mut full = vec![0.0f32; padded];
            for (k, &q) in codes.iter().enumerate() { full[k] = q as f32 * ds; }
            let recon = rot.rotate(&full);
            centroids[best].iter().zip(recon.iter()).map(|(c, r)| c + r).collect()
        }).collect();

        // Pairwise cosine → Fisher z transform
        let mut scores = Vec::with_capacity(n * (n - 1) / 2);
        for i in 0..n {
            for j in (i + 1)..n {
                let cos = cosine(&recon_rows[i], &recon_rows[j]);
                let clamped = cos.clamp(-0.999, 0.999);
                let z = 0.5 * ((1.0 + clamped) / (1.0 - clamped)).ln();
                scores.push(z);
            }
        }
        // Also transform ground truth the same way for fair comparison
        // (the bench compares against raw cosine GT, so we return z-scores
        // and the bench will compare z(codec) vs cosine(GT) — the ICC will
        // tell us if z-transform preserves the VALUE agreement)
        scores
    }
}

/// E5: Hadamard-sorted Matryoshka.
/// Hadamard rotate training rows, measure variance per coefficient,
/// sort by descending variance, allocate i16/i8/i4/i2 bands.
/// Matryoshka quality with Hadamard determinism.
struct HadSortedMatryoshka;
impl CodecCandidate for HadSortedMatryoshka {
    fn name(&self) -> &str { "Had-Matryoshka" }
    fn bytes_per_row(&self) -> usize { 0 }
    fn pairwise_scores(&self, rows: &[Vec<f32>]) -> Vec<f64> {
        use bgz17::rabitq_compat::OrthogonalMatrix;
        let n = rows.len();
        if n == 0 { return vec![]; }
        let n_cols = rows[0].len();
        let padded = RaBitQCodec::next_pow2(n_cols);
        let rot = OrthogonalMatrix::hadamard(padded);
        let centroids = F32ClamCentroid::clam_sample(rows, 64.min(n));

        // Rotate all residuals, measure variance per coefficient index
        let residuals: Vec<Vec<f32>> = rows.iter().map(|row| {
            let mut best = 0; let mut best_d = f32::MAX;
            for (ci, c) in centroids.iter().enumerate() {
                let d: f32 = row.iter().zip(c.iter()).map(|(a, b)| (a - b) * (a - b)).sum();
                if d < best_d { best_d = d; best = ci; }
            }
            let residual: Vec<f32> = row.iter().zip(centroids[best].iter()).map(|(a, b)| a - b).collect();
            let mut pad = residual;
            pad.resize(padded, 0.0);
            rot.rotate(&pad)
        }).collect();

        // Variance per coefficient (across all rows)
        let mut variances = vec![0.0f64; padded];
        for k in 0..padded {
            let mean: f64 = residuals.iter().map(|r| r[k] as f64).sum::<f64>() / n as f64;
            variances[k] = residuals.iter()
                .map(|r| { let d = r[k] as f64 - mean; d * d })
                .sum::<f64>() / n as f64;
        }

        // Sort indices by descending variance
        let mut sorted_idx: Vec<usize> = (0..padded).collect();
        sorted_idx.sort_by(|a, b| variances[*b].partial_cmp(&variances[*a]).unwrap());

        // Matryoshka bands on variance-sorted indices
        let d = n_cols.min(padded);
        let band_i16 = 64.min(d);
        let band_i8 = 192.min(d);
        let band_i4 = 384.min(d);

        let recon_rows: Vec<Vec<f32>> = rows.iter().enumerate().map(|(ri, row)| {
            let ref rotated = residuals[ri];
            let mut best = 0; let mut best_d = f32::MAX;
            for (ci, c) in centroids.iter().enumerate() {
                let d: f32 = row.iter().zip(c.iter()).map(|(a, b)| (a - b) * (a - b)).sum();
                if d < best_d { best_d = d; best = ci; }
            }

            // Quantize per band (on sorted indices)
            let mut recon_rotated = vec![0.0f32; padded];
            for (band_pos, &orig_idx) in sorted_idx.iter().take(d).enumerate() {
                let val = rotated[orig_idx];
                let (max_q, _bits): (f32, u8) = if band_pos < band_i16 {
                    (32767.0, 16)
                } else if band_pos < band_i8 {
                    (127.0, 8)
                } else if band_pos < band_i4 {
                    (7.0, 4)
                } else {
                    (1.0, 2)
                };
                // Per-coefficient quantize (simplified: use global scale per band)
                let q = val.clamp(-max_q, max_q).round();
                recon_rotated[orig_idx] = q;
            }

            // Need per-band scales for proper quantization
            // Compute band-wise max for scaling
            let mut band_maxes = [0.0f32; 4];
            for (band_pos, &orig_idx) in sorted_idx.iter().take(d).enumerate() {
                let abs_val = rotated[orig_idx].abs();
                let band = if band_pos < band_i16 { 0 }
                    else if band_pos < band_i8 { 1 }
                    else if band_pos < band_i4 { 2 }
                    else { 3 };
                if abs_val > band_maxes[band] { band_maxes[band] = abs_val; }
            }

            let mut recon_rot = vec![0.0f32; padded];
            for (band_pos, &orig_idx) in sorted_idx.iter().take(d).enumerate() {
                let val = rotated[orig_idx];
                let (max_q, band): (f32, usize) = if band_pos < band_i16 { (32767.0, 0) }
                    else if band_pos < band_i8 { (127.0, 1) }
                    else if band_pos < band_i4 { (7.0, 2) }
                    else { (1.0, 3) };
                let bmax = band_maxes[band];
                let scale = if bmax > 1e-12 { max_q / bmax } else { 0.0 };
                let q = (val * scale).round().clamp(-max_q, max_q);
                let dscale = if scale > 0.0 { bmax / max_q } else { 0.0 };
                recon_rot[orig_idx] = q * dscale;
            }

            let recon = rot.rotate(&recon_rot);
            centroids[best].iter().zip(recon.iter()).map(|(c, r)| c + r).collect()
        }).collect();

        let mut scores = Vec::with_capacity(n * (n - 1) / 2);
        for i in 0..n {
            for j in (i + 1)..n {
                scores.push(cosine(&recon_rows[i], &recon_rows[j]));
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
// Parametric codec grid — all combinations of basis × quant × mode × rank
// ═════════════════════════════════════════════════════════════════════

const EULER_GAMMA: f64 = 0.5772156649015329;
const _PHI: f64 = 1.618033988749895;

#[derive(Clone, Copy, PartialEq)]
enum PBasis { Svd, Had }
#[derive(Clone, Copy, PartialEq)]
enum PQuant { I2, I4, I8, GammaPhase, CircleOfFifths, FisherZ }
#[derive(Clone, Copy, PartialEq)]
enum PMode { Recon, Cam }
#[derive(Clone, Copy, PartialEq)]
enum PRank { N16, Full }

struct ParametricCodec {
    basis: PBasis,
    quant: PQuant,
    mode: PMode,
    rank: PRank,
    name_buf: String,
}

impl ParametricCodec {
    fn new(basis: PBasis, quant: PQuant, mode: PMode, rank: PRank) -> Self {
        let b = match basis { PBasis::Svd => "SVD", PBasis::Had => "Had" };
        let q = match quant {
            PQuant::I2 => "i2", PQuant::I4 => "i4", PQuant::I8 => "i8",
            PQuant::GammaPhase => "γφ", PQuant::CircleOfFifths => "Q5",
            PQuant::FisherZ => "Fz",
        };
        let m = match mode { PMode::Recon => "R", PMode::Cam => "C" };
        let r = match rank { PRank::N16 => "16", PRank::Full => "D" };
        ParametricCodec {
            basis, quant, mode, rank,
            name_buf: format!("{}-{}×{}-{}", b, q, r, m),
        }
    }

    fn max_val(&self) -> i32 {
        match self.quant {
            PQuant::I2 => 1,
            PQuant::I4 | PQuant::GammaPhase | PQuant::CircleOfFifths | PQuant::FisherZ => 7,
            PQuant::I8 => 127,
        }
    }

    fn all_variants() -> Vec<ParametricCodec> {
        let bases = [PBasis::Svd, PBasis::Had];
        let quants = [PQuant::I2, PQuant::I4, PQuant::I8, PQuant::GammaPhase, PQuant::CircleOfFifths, PQuant::FisherZ];
        let modes = [PMode::Recon, PMode::Cam];
        let ranks = [PRank::N16, PRank::Full];
        let mut out = Vec::new();
        for &b in &bases {
            for &q in &quants {
                for &m in &modes {
                    for &r in &ranks {
                        out.push(ParametricCodec::new(b, q, m, r));
                    }
                }
            }
        }
        out
    }
}

impl CodecCandidate for ParametricCodec {
    fn name(&self) -> &str { &self.name_buf }
    fn bytes_per_row(&self) -> usize { 0 }

    fn pairwise_scores(&self, rows: &[Vec<f32>]) -> Vec<f64> {
        use bgz_tensor::matryoshka::SvdBasis;
        use bgz17::rabitq_compat::OrthogonalMatrix;

        let n = rows.len();
        if n == 0 { return vec![]; }
        let n_cols = rows[0].len();
        let max_val = self.max_val();

        let n_comp = match self.rank {
            PRank::N16 => 16.min(n_cols),
            PRank::Full => n_cols,
        };

        let svd_basis = if self.basis == PBasis::Svd {
            Some(SvdBasis::build("p", rows, n_comp))
        } else { None };

        let padded_dim = RaBitQCodec::next_pow2(n_cols);
        let had_matrix = if self.basis == PBasis::Had {
            Some(OrthogonalMatrix::hadamard(padded_dim))
        } else { None };

        let actual_d = match &svd_basis {
            Some(b) => b.n_components,
            None => n_comp.min(padded_dim),
        };

        // γ-pressure table: pressure[k] = γ^(k/D), decays from 1.0 to γ
        let pressures: Vec<f32> = if self.quant == PQuant::GammaPhase {
            let d = actual_d.max(1) as f64;
            (0..actual_d).map(|k| EULER_GAMMA.powf(k as f64 / d) as f32).collect()
        } else { vec![] };

        // Circle-of-fifths: 12 semitone bins with φ-spaced phase progression.
        // Quintenzirkel maps coefficient index to a position on the circle of
        // fifths (interval of 7 semitones mod 12). The phase accumulates as
        // k*7 mod 12, creating a non-sequential but musically coherent ordering.
        // Magnitude is quantized to i4, phase bin encodes relative position.
        let q5_phases: Vec<f32> = if self.quant == PQuant::CircleOfFifths {
            (0..actual_d).map(|k| {
                let semitone = (k * 7) % 12;
                let phase = semitone as f64 * std::f64::consts::TAU / 12.0;
                phase.cos() as f32
            }).collect()
        } else { vec![] };


        let use_centroid = self.mode == PMode::Recon;
        let centroids = if use_centroid {
            F32ClamCentroid::clam_sample(rows, 64.min(n))
        } else { vec![] };

        // Encode: project + quantize
        // Tuple: (centroid_idx, codes, scale1, scale2)
        // scale2 only used by FisherZ (z_max)
        let encoded: Vec<(usize, Vec<i8>, f32, f32)> = rows.iter().map(|row| {
            let (ci, to_project) = if use_centroid {
                let mut best = 0; let mut best_d = f32::MAX;
                for (ci, c) in centroids.iter().enumerate() {
                    let d: f32 = row.iter().zip(c.iter()).map(|(a, b)| (a - b) * (a - b)).sum();
                    if d < best_d { best_d = d; best = ci; }
                }
                let res: Vec<f32> = row.iter().zip(centroids[best].iter()).map(|(a, b)| a - b).collect();
                (best, res)
            } else {
                (0, row.clone())
            };

            let coeffs: Vec<f32> = match self.basis {
                PBasis::Svd => {
                    let c = svd_basis.as_ref().unwrap().project(&to_project);
                    c[..actual_d.min(c.len())].to_vec()
                }
                PBasis::Had => {
                    let mut padded = to_project.clone();
                    padded.resize(padded_dim, 0.0);
                    let rotated = had_matrix.as_ref().unwrap().rotate(&padded);
                    rotated[..actual_d].to_vec()
                }
            };

            match self.quant {
                PQuant::GammaPhase => {
                    let scaled: Vec<f32> = coeffs.iter().zip(pressures.iter())
                        .map(|(c, p)| if *p > 1e-12 { c / p } else { 0.0 }).collect();
                    let max_abs = scaled.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
                    let inv = if max_abs > 1e-12 { 7.0 / max_abs } else { 0.0 };
                    let codes: Vec<i8> = scaled.iter()
                        .map(|c| (c * inv).round().clamp(-7.0, 7.0) as i8).collect();
                    (ci, codes, if inv > 0.0 { max_abs / 7.0 } else { 0.0 }, 0.0)
                }
                PQuant::CircleOfFifths => {
                    let modulated: Vec<f32> = coeffs.iter().enumerate().map(|(k, c)| {
                        if k < q5_phases.len() {
                            let phase_weight = (1.0 + q5_phases[k].abs()) * 0.5 + 0.5;
                            c / phase_weight
                        } else { *c }
                    }).collect();
                    let max_abs = modulated.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
                    let inv = if max_abs > 1e-12 { 7.0 / max_abs } else { 0.0 };
                    let codes: Vec<i8> = modulated.iter()
                        .map(|c| (c * inv).round().clamp(-7.0, 7.0) as i8).collect();
                    (ci, codes, if inv > 0.0 { max_abs / 7.0 } else { 0.0 }, 0.0)
                }
                PQuant::FisherZ => {
                    // Normalize to [-1,1], arctanh to z-space, quantize there.
                    // arctanh stretches tails → more i4 levels near ±1.
                    // tanh on decode guarantees valid range.
                    let max_abs = coeffs.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
                    let norm = if max_abs > 1e-12 { 1.0 / max_abs } else { 0.0 };
                    let z_vals: Vec<f64> = coeffs.iter().map(|c| {
                        let r = (*c * norm) as f64;
                        let clamped = r.clamp(-0.999, 0.999);
                        0.5 * ((1.0 + clamped) / (1.0 - clamped)).ln() // arctanh
                    }).collect();
                    let z_max = z_vals.iter().map(|z| z.abs()).fold(0.0f64, f64::max);
                    let z_inv = if z_max > 1e-12 { 7.0 / z_max } else { 0.0 };
                    let codes: Vec<i8> = z_vals.iter()
                        .map(|z| (z * z_inv).round().clamp(-7.0, 7.0) as i8).collect();
                    (ci, codes, max_abs, z_max as f32)
                }
                _ => {
                    let max_abs = coeffs.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
                    let mv = max_val as f32;
                    let inv = if max_abs > 1e-12 { mv / max_abs } else { 0.0 };
                    let codes: Vec<i8> = coeffs.iter()
                        .map(|c| (c * inv).round().clamp(-mv, mv) as i8).collect();
                    (ci, codes, if inv > 0.0 { max_abs / mv } else { 0.0 }, 0.0)
                }
            }
        }).collect();

        match self.mode {
            PMode::Recon => {
                let recon_rows: Vec<Vec<f32>> = (0..n).map(|i| {
                    let (ci, ref codes, scale, scale2) = encoded[i];
                    let dequant: Vec<f32> = match self.quant {
                        PQuant::GammaPhase => {
                            codes.iter().zip(pressures.iter())
                                .map(|(&q, &p)| q as f32 * scale * p).collect()
                        }
                        PQuant::CircleOfFifths => {
                            codes.iter().enumerate().map(|(k, &q)| {
                                let phase_weight = if k < q5_phases.len() {
                                    (1.0 + q5_phases[k].abs()) * 0.5 + 0.5
                                } else { 1.0 };
                                q as f32 * scale * phase_weight
                            }).collect()
                        }
                        PQuant::FisherZ => {
                            // Reverse: i4 → z-space → tanh → [-1,1] → denormalize
                            let z_max = scale2 as f64;
                            let z_scale = if z_max > 1e-12 { z_max / 7.0 } else { 0.0 };
                            codes.iter().map(|&q| {
                                let z = q as f64 * z_scale;
                                let r = z.tanh(); // back to [-1,1]
                                (r as f32) * scale // denormalize
                            }).collect()
                        }
                        _ => codes.iter().map(|&q| q as f32 * scale).collect(),
                    };
                    let res = match self.basis {
                        PBasis::Svd => svd_basis.as_ref().unwrap().reconstruct(&dequant),
                        PBasis::Had => {
                            let mut full = vec![0.0f32; padded_dim];
                            for (k, &v) in dequant.iter().enumerate() { full[k] = v; }
                            let row = had_matrix.as_ref().unwrap().rotate(&full);
                            row[..n_cols].to_vec()
                        }
                    };
                    if use_centroid {
                        centroids[ci].iter().zip(res.iter()).map(|(c, r)| c + r).collect()
                    } else { res }
                }).collect();

                let mut scores = Vec::with_capacity(n * (n - 1) / 2);
                for i in 0..n {
                    for j in (i + 1)..n {
                        scores.push(cosine(&recon_rows[i], &recon_rows[j]));
                    }
                }
                scores
            }
            PMode::Cam => {
                let max_dist: f64 = match self.quant {
                    PQuant::GammaPhase => {
                        pressures.iter().map(|p| 14.0 * *p as f64).sum()
                    }
                    PQuant::CircleOfFifths => {
                        (0..actual_d).map(|k| {
                            let pw = if k < q5_phases.len() {
                                (1.0 + q5_phases[k].abs() as f64) * 0.5 + 0.5
                            } else { 1.0 };
                            14.0 * pw
                        }).sum()
                    }
                    _ => (actual_d as f64) * (2 * max_val) as f64,
                };

                let mut scores = Vec::with_capacity(n * (n - 1) / 2);
                for i in 0..n {
                    for j in (i + 1)..n {
                        let dist: f64 = match self.quant {
                            PQuant::GammaPhase => {
                                encoded[i].1.iter().zip(encoded[j].1.iter())
                                    .zip(pressures.iter())
                                    .map(|((a, b), p)| (*a as f64 - *b as f64).abs() * *p as f64)
                                    .sum()
                            }
                            PQuant::CircleOfFifths => {
                                encoded[i].1.iter().zip(encoded[j].1.iter()).enumerate()
                                    .map(|(k, (a, b))| {
                                        let pw = if k < q5_phases.len() {
                                            (1.0 + q5_phases[k].abs() as f64) * 0.5 + 0.5
                                        } else { 1.0 };
                                        (*a as f64 - *b as f64).abs() * pw
                                    }).sum()
                            }
                            _ => {
                                encoded[i].1.iter().zip(encoded[j].1.iter())
                                    .map(|(a, b)| (*a as i32 - *b as i32).abs() as f64)
                                    .sum()
                            }
                        };
                        scores.push(1.0 - dist / max_dist.max(1e-12));
                    }
                }
                scores
            }
        }
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
    println!("Grid: 19 named + 48 parametric = 67 codecs");

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

        let mut codecs: Vec<Box<dyn CodecCandidate>> = vec![
            Box::new(Passthrough),
            Box::new(Base17Sig),
            Box::new(Base17FzSig),
            Box::new(DirectI8),
            Box::new(SpiralK8),
            Box::new(RaBitQCodec { dim: n_cols }),
            Box::new(F32ClamCentroid),
            Box::new(F32ClamSlotL),
            Box::new(F32ClamLeafI4),
            Box::new(F32ClamLeafMixed),
            Box::new(F32ClamFullI4),
            Box::new(BitpackedCamI4),
            Box::new(MatryoshkaCodec),
            Box::new(GainShapeHadCam),
            Box::new(SharedScaleHadCam),
            Box::new(HadI4PlusI2Residue),
            Box::new(FisherZOnDistance),
            Box::new(HadSortedMatryoshka),
            Box::new(I8HybridCodec),
        ];
        // Parametric grid: 2 bases × 5 quants × 2 modes × 2 ranks = 40 variants
        for p in ParametricCodec::all_variants() {
            codecs.push(Box::new(p));
        }

        // Lab / R&D candidates — fractal descriptor variants.
        #[cfg(feature = "lab")]
        {
            codecs.push(Box::new(FractalDescOnly));
            codecs.push(Box::new(FractalPlusBase17));
            codecs.push(Box::new(FractalPhaseOnly));
            codecs.push(Box::new(FractalPhasePlusBase17));
            codecs.push(Box::new(ZipperPhaseOnly));
            codecs.push(Box::new(ZipperFull));
        }

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
