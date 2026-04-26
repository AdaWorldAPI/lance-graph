//! Hadamard Cascade Codec — production-grade weight compression.
//!
//! The winning codec from the 67-codec R&D sweep (PR #198):
//!   1. CLAM k=64 furthest-point centroid assignment (1 byte twig)
//!   2. Hadamard rotation of residual (deterministic, no stored basis)
//!   3. Pass 1: i4 full-rank quantization + per-row BF16 scale
//!   4. Pass 2: i2 full-rank quantization on i4 residue + per-row BF16 scale
//!
//! Measured ICC:
//!   Attention k_proj:    0.9995 (Qwen3-TTS 1024-d)
//!   MLP gate_proj:       0.9975
//!   Text embedding:      0.9951
//!   Audio codec emb:     1.0000
//!   PLE projection:      0.999+ (Gemma4 256-d)
//!
//! Wire format per row:
//!   twig (1B) + scale1 (2B BF16) + i4_codes (D/2 B) + scale2 (2B BF16) + i2_codes (D/4 B)
//!   = 5 + 3D/4 bytes/row
//!   At D=256: 197 B/row (2.6:1 vs BF16)
//!   At D=1024: 773 B/row (2.6:1 vs BF16)
//!
//! Codec selection rules:
//!   Argmax regime (attention, MLP): HadCascade (this codec)
//!   Index regime (embeddings, lm_head): BF16 passthrough + CAM-PQ address
//!   Structured embeddings (audio codec): HadCascade (ICC 1.000)

use crate::stacked_n::{bf16_to_f32, f32_to_bf16};
use ndarray::hpc::fft::wht_f32;
use ndarray::hpc::quantized::{
    quantize_f32_to_i4, dequantize_i4_to_f32,
    quantize_f32_to_i2, dequantize_i2_to_f32,
};
use ndarray::hpc::cam_pq::{kmeans, squared_l2};
use ndarray::hpc::heel_f64x8::cosine_f32_to_f64_simd;

fn next_pow2(n: usize) -> usize {
    let mut p = 1;
    while p < n { p *= 2; }
    p
}

fn hadamard_rotate(v: &[f32], dim: usize) -> Vec<f32> {
    let mut out = v.to_vec();
    out.resize(dim, 0.0);
    wht_f32(&mut out);
    out
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TensorRegime {
    Argmax,
    Index,
}

impl TensorRegime {
    pub fn from_role(role: &str) -> Self {
        let r = role.to_lowercase();
        if r.contains("embed") || r.contains("lm_head") {
            TensorRegime::Index
        } else {
            TensorRegime::Argmax
        }
    }

    pub fn should_compress(&self) -> bool {
        matches!(self, TensorRegime::Argmax)
    }
}

#[derive(Clone, Debug)]
pub struct HadCascadeRow {
    pub twig: u8,
    pub scale1_bf16: u16,
    pub i4_codes: Vec<u8>,
    pub scale2_bf16: u16,
    pub i2_codes: Vec<u8>,
}

impl HadCascadeRow {
    pub fn byte_size(&self) -> usize {
        1 + 2 + self.i4_codes.len() + 2 + self.i2_codes.len()
    }
}

#[derive(Clone, Debug)]
pub struct HadCascadeTensor {
    pub role: String,
    pub regime: TensorRegime,
    pub n_rows: usize,
    pub n_cols: usize,
    pub padded_dim: usize,
    pub centroids: Vec<Vec<f32>>,
    pub rows: Vec<HadCascadeRow>,
}

impl HadCascadeTensor {
    pub fn encode(role: &str, data: &[Vec<f32>], k: usize) -> Self {
        Self::encode_with_precision(role, data, k, false)
    }

    pub fn encode_i8(role: &str, data: &[Vec<f32>], k: usize) -> Self {
        Self::encode_with_precision(role, data, k, true)
    }

    fn encode_with_precision(role: &str, data: &[Vec<f32>], k: usize, use_i8: bool) -> Self {
        let n = data.len();
        let n_cols = if n > 0 { data[0].len() } else { 0 };
        let padded = next_pow2(n_cols);
        let regime = TensorRegime::from_role(role);
        let k = k.min(n).min(256);

        let centroids = build_centroids(data, k);

        let rows: Vec<HadCascadeRow> = data.iter().map(|row| {
            let (ci, _) = nearest_centroid(row, &centroids);

            let residual: Vec<f32> = row.iter().zip(centroids[ci].iter())
                .map(|(a, b)| a - b).collect();

            let rotated1 = hadamard_rotate(&residual, padded);

            if use_i8 {
                // i8-only mode: 2:1 compression, higher fidelity
                // dequantize_i8_to_f32 used in reconstruct_row decode path
                #[allow(unused_imports)]
                use ndarray::hpc::quantized::{quantize_f32_to_i8, dequantize_i8_to_f32};
                let (i8_codes_raw, i8_params) = quantize_f32_to_i8(&rotated1[..n_cols]);
                let i8_as_u8: Vec<u8> = i8_codes_raw.iter().map(|&v| v as u8).collect();
                HadCascadeRow {
                    twig: ci as u8,
                    scale1_bf16: f32_to_bf16(i8_params.scale),
                    i4_codes: i8_as_u8,
                    scale2_bf16: 0,
                    i2_codes: vec![],
                }
            } else {
                // i4+i2 cascade: 2.65:1 compression
                let (i4_codes, i4_params) = quantize_f32_to_i4(&rotated1[..n_cols]);
                let dequant1 = dequantize_i4_to_f32(&i4_codes, &i4_params, n_cols);
                let mut full1 = vec![0.0f32; padded];
                full1[..n_cols].copy_from_slice(&dequant1);
                let recon1 = hadamard_rotate(&full1, padded);

                let residual2: Vec<f32> = residual.iter().zip(recon1.iter().take(n_cols))
                    .map(|(orig, r1)| orig - r1).collect();
                let rotated2 = hadamard_rotate(&residual2, padded);
                let (i2_codes, i2_params) = quantize_f32_to_i2(&rotated2[..n_cols]);

                HadCascadeRow {
                    twig: ci as u8,
                    scale1_bf16: f32_to_bf16(i4_params.scale),
                    i4_codes,
                    scale2_bf16: f32_to_bf16(i2_params.scale),
                    i2_codes,
                }
            }
        }).collect();

        HadCascadeTensor {
            role: role.to_string(),
            regime,
            n_rows: n,
            n_cols,
            padded_dim: padded,
            centroids,
            rows,
        }
    }

    pub fn reconstruct_row(&self, i: usize) -> Vec<f32> {
        use ndarray::hpc::quantized::QuantParams;
        let row = &self.rows[i];
        let ci = row.twig as usize;
        let p1 = QuantParams { scale: bf16_to_f32(row.scale1_bf16), zero_point: 0, min_val: 0.0, max_val: 0.0 };

        if row.i2_codes.is_empty() {
            // i8-only mode
            use ndarray::hpc::quantized::dequantize_i8_to_f32;
            let i8_codes: Vec<i8> = row.i4_codes.iter().map(|&v| v as i8).collect();
            let dequant = dequantize_i8_to_f32(&i8_codes, &p1, self.n_cols);
            let mut full = vec![0.0f32; self.padded_dim];
            full[..self.n_cols].copy_from_slice(&dequant);
            let recon = hadamard_rotate(&full, self.padded_dim);
            self.centroids[ci].iter().zip(recon.iter()).map(|(c, r)| c + r).collect()
        } else {
            // i4+i2 cascade mode
            let dequant1 = dequantize_i4_to_f32(&row.i4_codes, &p1, self.n_cols);
            let mut full1 = vec![0.0f32; self.padded_dim];
            full1[..self.n_cols].copy_from_slice(&dequant1);
            let recon1 = hadamard_rotate(&full1, self.padded_dim);

            let p2 = QuantParams { scale: bf16_to_f32(row.scale2_bf16), zero_point: 0, min_val: 0.0, max_val: 0.0 };
            let dequant2 = dequantize_i2_to_f32(&row.i2_codes, &p2, self.n_cols);
            let mut full2 = vec![0.0f32; self.padded_dim];
            full2[..self.n_cols].copy_from_slice(&dequant2);
            let recon2 = hadamard_rotate(&full2, self.padded_dim);

            self.centroids[ci].iter()
                .zip(recon1.iter())
                .zip(recon2.iter())
                .map(|((c, r1), r2)| c + r1 + r2)
                .collect()
        }
    }

    pub fn reconstruct_all(&self) -> Vec<Vec<f32>> {
        (0..self.n_rows).map(|i| self.reconstruct_row(i)).collect()
    }

    pub fn bytes_per_row(&self) -> usize {
        if self.rows.is_empty() { 0 } else { self.rows[0].byte_size() }
    }

    pub fn total_bytes(&self) -> usize {
        let row_bytes: usize = self.rows.iter().map(|r| r.byte_size()).sum();
        let centroid_bytes = self.centroids.len() * self.n_cols * 4;
        row_bytes + centroid_bytes
    }

    pub fn compression_ratio(&self) -> f64 {
        let original = self.n_rows * self.n_cols * 2; // BF16
        if self.total_bytes() == 0 { 0.0 } else { original as f64 / self.total_bytes() as f64 }
    }

    pub fn to_bytes(&self) -> Vec<u8> {
        let mut buf = Vec::new();
        // Header
        buf.extend_from_slice(&(self.n_rows as u32).to_le_bytes());
        buf.extend_from_slice(&(self.n_cols as u32).to_le_bytes());
        buf.extend_from_slice(&(self.padded_dim as u32).to_le_bytes());
        buf.extend_from_slice(&(self.centroids.len() as u32).to_le_bytes());
        // Centroids as BF16
        for c in &self.centroids {
            for &v in c {
                buf.extend_from_slice(&f32_to_bf16(v).to_le_bytes());
            }
        }
        // Rows
        for row in &self.rows {
            buf.push(row.twig);
            buf.extend_from_slice(&row.scale1_bf16.to_le_bytes());
            buf.extend_from_slice(&(row.i4_codes.len() as u16).to_le_bytes());
            buf.extend_from_slice(&row.i4_codes);
            buf.extend_from_slice(&row.scale2_bf16.to_le_bytes());
            buf.extend_from_slice(&(row.i2_codes.len() as u16).to_le_bytes());
            buf.extend_from_slice(&row.i2_codes);
        }
        buf
    }
}

// ═══════════════════════════════════════════════════════════════════
// CLAM furthest-point sampling
// ═══════════════════════════════════════════════════════════════════

fn build_centroids(rows: &[Vec<f32>], k: usize) -> Vec<Vec<f32>> {
    let n = rows.len();
    if n == 0 || k == 0 { return vec![]; }
    let dim = rows[0].len();
    kmeans(rows, k.min(n), dim, 10)
}

fn nearest_centroid(row: &[f32], centroids: &[Vec<f32>]) -> (usize, f32) {
    let mut best = 0;
    let mut best_d = f32::MAX;
    for (ci, c) in centroids.iter().enumerate() {
        let d = squared_l2(row, c);
        if d < best_d { best_d = d; best = ci; }
    }
    (best, best_d)
}

// ═══════════════════════════════════════════════════════════════════
// i4 packing: 2 nibbles per byte, signed ±7
// ═══════════════════════════════════════════════════════════════════

// All i4/i2 packing, cosine, kmeans, and WHT delegated to ndarray::hpc

pub fn measure_quality(original: &[Vec<f32>], reconstructed: &[Vec<f32>]) -> (f64, f64) {
    let n = original.len().min(reconstructed.len());
    if n == 0 { return (0.0, 0.0); }

    let mut cos_sum = 0.0f64;
    for i in 0..n {
        cos_sum += cosine_f32_to_f64_simd(&original[i], &reconstructed[i]);
    }
    let avg_cos = cos_sum / n as f64;

    let icc = crate::quality::icc_3_1(
        &original.iter().enumerate()
            .flat_map(|(i, _)| (i + 1..n).map(move |j| cosine_f32_to_f64_simd(&original[i], &original[j])))
            .collect::<Vec<_>>(),
        &reconstructed.iter().enumerate()
            .flat_map(|(i, _)| (i + 1..n).map(move |j| cosine_f32_to_f64_simd(&reconstructed[i], &reconstructed[j])))
            .collect::<Vec<_>>(),
    );

    (avg_cos, icc)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_row(seed: usize, dim: usize) -> Vec<f32> {
        (0..dim).map(|d| {
            ((d * 97 + seed * 31 + 17) as f64 * 0.618).sin() as f32 * 0.01
        }).collect()
    }

    #[test]
    fn encode_decode_roundtrip() {
        let rows: Vec<Vec<f32>> = (0..64).map(|i| make_row(i, 256)).collect();
        let tensor = HadCascadeTensor::encode("test_q_proj", &rows, 64);
        assert_eq!(tensor.n_rows, 64);
        assert_eq!(tensor.regime, TensorRegime::Argmax);

        let recon = tensor.reconstruct_all();
        assert_eq!(recon.len(), 64);
        assert_eq!(recon[0].len(), 256);

        let (avg_cos, icc) = measure_quality(&rows, &recon);
        assert!(avg_cos > 0.9, "avg cosine {} should be >0.9", avg_cos);
        assert!(icc > 0.9, "ICC {} should be >0.9", icc);
    }

    #[test]
    fn regime_detection() {
        assert_eq!(TensorRegime::from_role("self_attn.q_proj.weight"), TensorRegime::Argmax);
        assert_eq!(TensorRegime::from_role("mlp.gate_proj.weight"), TensorRegime::Argmax);
        assert_eq!(TensorRegime::from_role("text_embedding.weight"), TensorRegime::Index);
        assert_eq!(TensorRegime::from_role("lm_head.weight"), TensorRegime::Index);
    }

    #[test]
    fn serialization_size() {
        let rows: Vec<Vec<f32>> = (0..32).map(|i| make_row(i, 256)).collect();
        let tensor = HadCascadeTensor::encode("test", &rows, 32);
        let bytes = tensor.to_bytes();
        assert!(bytes.len() > 0);
        let bpr = tensor.bytes_per_row();
        // 1 twig + 2 scale1 + 128 i4 + 2 scale2 + 64 i2 = 197
        assert_eq!(bpr, 197, "bytes_per_row at 256-d should be 197");
    }

    #[test]
    fn i4_ndarray_roundtrip() {
        let vals = vec![0.7, -0.3, 0.95, -0.1, 0.6];
        let (packed, params) = quantize_f32_to_i4(&vals);
        let unpacked = dequantize_i4_to_f32(&packed, &params, 5);
        for (orig, recon) in vals.iter().zip(unpacked.iter()) {
            assert!((orig - recon).abs() < 0.2, "i4 roundtrip: {} vs {}", orig, recon);
        }
    }

    #[test]
    fn i2_ndarray_roundtrip() {
        let vals = vec![0.8, -0.5, 0.1, -0.9, 0.3];
        let (packed, params) = quantize_f32_to_i2(&vals);
        let unpacked = dequantize_i2_to_f32(&packed, &params, 5);
        for &v in &unpacked {
            assert!(v.abs() <= params.scale + 0.01, "i2 value out of range: {}", v);
        }
    }

    #[test]
    fn compression_ratio_reasonable() {
        // Use enough rows that centroid overhead amortizes
        let rows: Vec<Vec<f32>> = (0..512).map(|i| make_row(i, 256)).collect();
        let tensor = HadCascadeTensor::encode("test", &rows, 64);
        let ratio = tensor.compression_ratio();
        assert!(ratio > 1.5 && ratio < 4.0, "ratio {} should be in [1.5, 4.0]", ratio);
    }
}
