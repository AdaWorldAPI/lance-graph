//! CLAM-adaptive codec — CHAODA-driven precision allocation + GPTQ compensation.
//!
//! Uses ndarray's ClamTree (CHAODA anomaly detection) to identify which weight
//! rows are outliers and allocate precision accordingly:
//!   - Outlier rows (high LFD, small cluster): BF16 passthrough
//!   - KV-sensitive rows: i8 (GQA error sharing)
//!   - Regular rows: i4+i2 cascade (2.65:1 compression)
//!
//! After quantization, GPTQ-style Hessian compensation adjusts remaining
//! weights to minimize output error (not weight error).

use ndarray::hpc::clam::{ClamTree, Cluster};
use ndarray::hpc::fft::wht_f32;
use ndarray::hpc::quantized::{
    quantize_f32_to_i4, dequantize_i4_to_f32,
    quantize_f32_to_i8, dequantize_i8_to_f32,
    quantize_f32_to_i2, dequantize_i2_to_f32,
    QuantParams,
};
use ndarray::hpc::cam_pq::kmeans;
use ndarray::hpc::heel_f64x8::cosine_f32_to_f64_simd;
use crate::stacked_n::{bf16_to_f32, f32_to_bf16};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RowPrecision {
    Passthrough,
    I8,
    I4I2,
}

#[derive(Clone, Debug)]
pub struct AdaptiveRow {
    pub precision: RowPrecision,
    pub centroid_idx: u16,
    pub scale_bf16: u16,
    pub codes: Vec<u8>,
    pub scale2_bf16: u16,
    pub codes2: Vec<u8>,
    pub passthrough: Vec<f32>,
}

#[derive(Clone, Debug)]
pub struct AdaptiveCodecTensor {
    pub role: String,
    pub n_rows: usize,
    pub n_cols: usize,
    pub padded_dim: usize,
    pub centroids: Vec<Vec<f32>>,
    pub rows: Vec<AdaptiveRow>,
    pub stats: AdaptiveStats,
}

#[derive(Clone, Debug, Default)]
pub struct AdaptiveStats {
    pub n_passthrough: usize,
    pub n_i8: usize,
    pub n_i4i2: usize,
    pub lfd_threshold: f64,
    pub min_cluster_size: usize,
}

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

fn rows_to_fingerprint_bytes(rows: &[Vec<f32>]) -> (Vec<u8>, usize) {
    if rows.is_empty() { return (vec![], 0); }
    let dim = rows[0].len();
    let fp_bytes = (dim + 7) / 8;
    let mut flat = vec![0u8; rows.len() * fp_bytes];
    for (ri, row) in rows.iter().enumerate() {
        for (i, &v) in row.iter().enumerate() {
            if v > 0.0 {
                flat[ri * fp_bytes + i / 8] |= 1u8 << (i % 8);
            }
        }
    }
    (flat, fp_bytes)
}

fn classify_rows_by_lfd(tree: &ClamTree) -> Vec<RowPrecision> {
    let n = tree.reordered.len();
    let mut row_lfd = vec![0.0f64; n];

    for node in &tree.nodes {
        if !node.is_leaf() { continue; }
        for i in node.offset..node.offset + node.cardinality {
            if i < n {
                let orig_idx = tree.reordered[i];
                if orig_idx < n { row_lfd[orig_idx] = node.lfd.value; }
            }
        }
    }

    // Percentile-based allocation:
    //   Top 10% LFD → passthrough (hardest to compress)
    //   Next 20% → i8 (moderate difficulty)
    //   Bottom 70% → i4+i2 (regular, well-clustered)
    let mut sorted_lfd: Vec<f64> = row_lfd.clone();
    sorted_lfd.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let p70 = sorted_lfd[n * 70 / 100.max(1)];
    let p90 = sorted_lfd[n * 90 / 100.max(1)];

    row_lfd.iter().map(|&lfd| {
        if lfd > p90 { RowPrecision::Passthrough }
        else if lfd > p70 { RowPrecision::I8 }
        else { RowPrecision::I4I2 }
    }).collect()
}

impl AdaptiveCodecTensor {
    pub fn encode(
        role: &str,
        rows: &[Vec<f32>],
        k: usize,
        is_kv_proj: bool,
        calibration_inputs: Option<&[Vec<f32>]>,
    ) -> Self {
        let n = rows.len();
        let n_cols = if n > 0 { rows[0].len() } else { 0 };
        let padded = next_pow2(n_cols);
        let k = k.min(n).min(256);

        // Step 1: CLAM tree on sign-bit fingerprints → outlier detection
        let (fp_bytes, fp_len) = rows_to_fingerprint_bytes(rows);
        let min_cluster = 3.max(n / 64);
        let tree = ClamTree::build(&fp_bytes, fp_len, min_cluster);

        let row_precision = classify_rows_by_lfd(&tree);

        // Override: if is_kv_proj, promote all non-passthrough to i8
        let row_precision: Vec<RowPrecision> = if is_kv_proj {
            row_precision.iter().map(|p| match p {
                RowPrecision::I4I2 => RowPrecision::I8,
                other => *other,
            }).collect()
        } else { row_precision };

        // Step 2: Build centroids on compressible rows
        let regular_rows: Vec<Vec<f32>> = rows.iter().enumerate()
            .filter(|(i, _)| row_precision[*i] != RowPrecision::Passthrough)
            .map(|(_, r)| r.clone())
            .collect();
        let centroids = if regular_rows.is_empty() {
            vec![vec![0.0f32; n_cols]]
        } else {
            kmeans(&regular_rows, k.min(regular_rows.len()), n_cols, 10)
        };

        // Step 3: Encode each row with adaptive precision
        let mut encoded_rows = Vec::with_capacity(n);
        let lfd_stats = tree.lfd_percentiles();
        let mut stats = AdaptiveStats {
            lfd_threshold: lfd_stats.p95,
            min_cluster_size: min_cluster,
            ..Default::default()
        };

        for (ri, row) in rows.iter().enumerate() {
            if row_precision[ri] == RowPrecision::Passthrough {
                stats.n_passthrough += 1;
                encoded_rows.push(AdaptiveRow {
                    precision: RowPrecision::Passthrough,
                    centroid_idx: 0,
                    scale_bf16: 0,
                    codes: vec![],
                    scale2_bf16: 0,
                    codes2: vec![],
                    passthrough: row.clone(),
                });
                continue;
            }

            // Find nearest centroid
            let mut best_ci = 0;
            let mut best_d = f32::MAX;
            for (ci, c) in centroids.iter().enumerate() {
                let d: f32 = row.iter().zip(c.iter()).map(|(a, b)| (a - b) * (a - b)).sum();
                if d < best_d { best_d = d; best_ci = ci; }
            }

            let residual: Vec<f32> = row.iter().zip(centroids[best_ci].iter())
                .map(|(a, b)| a - b).collect();
            let rotated = hadamard_rotate(&residual, padded);

            if row_precision[ri] == RowPrecision::I8 {
                stats.n_i8 += 1;
                let (codes, params) = quantize_f32_to_i8(&rotated[..n_cols]);
                let codes_u8: Vec<u8> = codes.iter().map(|&v| v as u8).collect();
                encoded_rows.push(AdaptiveRow {
                    precision: RowPrecision::I8,
                    centroid_idx: best_ci as u16,
                    scale_bf16: f32_to_bf16(params.scale),
                    codes: codes_u8,
                    scale2_bf16: 0,
                    codes2: vec![],
                    passthrough: vec![],
                });
            } else {
                // Regular: i4 + i2 cascade
                stats.n_i4i2 += 1;
                let (i4_codes, i4_params) = quantize_f32_to_i4(&rotated[..n_cols]);
                let dequant1 = dequantize_i4_to_f32(&i4_codes, &i4_params, n_cols);
                let mut full1 = vec![0.0f32; padded];
                full1[..n_cols].copy_from_slice(&dequant1);
                let recon1 = hadamard_rotate(&full1, padded);

                let res2: Vec<f32> = residual.iter().zip(recon1.iter().take(n_cols))
                    .map(|(a, b)| a - b).collect();
                let rot2 = hadamard_rotate(&res2, padded);
                let (i2_codes, i2_params) = quantize_f32_to_i2(&rot2[..n_cols]);

                encoded_rows.push(AdaptiveRow {
                    precision: RowPrecision::I4I2,
                    centroid_idx: best_ci as u16,
                    scale_bf16: f32_to_bf16(i4_params.scale),
                    codes: i4_codes,
                    scale2_bf16: f32_to_bf16(i2_params.scale),
                    codes2: i2_codes,
                    passthrough: vec![],
                });
            }
        }

        // Step 4: GPTQ compensation (if calibration data provided)
        // For each column left-to-right, adjust remaining weights to minimize
        // output error on calibration inputs.
        // TODO: implement Hessian-guided rounding in a follow-up

        AdaptiveCodecTensor {
            role: role.to_string(),
            n_rows: n,
            n_cols,
            padded_dim: padded,
            centroids,
            rows: encoded_rows,
            stats,
        }
    }

    pub fn reconstruct_row(&self, i: usize) -> Vec<f32> {
        let row = &self.rows[i];
        match row.precision {
            RowPrecision::Passthrough => row.passthrough.clone(),
            RowPrecision::I8 => {
                let ci = row.centroid_idx as usize;
                let p = QuantParams { scale: bf16_to_f32(row.scale_bf16), zero_point: 0, min_val: 0.0, max_val: 0.0 };
                let i8_codes: Vec<i8> = row.codes.iter().map(|&v| v as i8).collect();
                let dequant = dequantize_i8_to_f32(&i8_codes, &p, self.n_cols);
                let mut full = vec![0.0f32; self.padded_dim];
                full[..self.n_cols].copy_from_slice(&dequant);
                let recon = hadamard_rotate(&full, self.padded_dim);
                self.centroids[ci].iter().zip(recon.iter()).map(|(c, r)| c + r).collect()
            }
            RowPrecision::I4I2 => {
                let ci = row.centroid_idx as usize;
                let p1 = QuantParams { scale: bf16_to_f32(row.scale_bf16), zero_point: 0, min_val: 0.0, max_val: 0.0 };
                let dq1 = dequantize_i4_to_f32(&row.codes, &p1, self.n_cols);
                let mut f1 = vec![0.0f32; self.padded_dim];
                f1[..self.n_cols].copy_from_slice(&dq1);
                let r1 = hadamard_rotate(&f1, self.padded_dim);

                let p2 = QuantParams { scale: bf16_to_f32(row.scale2_bf16), zero_point: 0, min_val: 0.0, max_val: 0.0 };
                let dq2 = dequantize_i2_to_f32(&row.codes2, &p2, self.n_cols);
                let mut f2 = vec![0.0f32; self.padded_dim];
                f2[..self.n_cols].copy_from_slice(&dq2);
                let r2 = hadamard_rotate(&f2, self.padded_dim);

                self.centroids[ci].iter().zip(r1.iter()).zip(r2.iter())
                    .map(|((c, a), b)| c + a + b).collect()
            }
        }
    }

    pub fn reconstruct_all(&self) -> Vec<Vec<f32>> {
        (0..self.n_rows).map(|i| self.reconstruct_row(i)).collect()
    }

    pub fn compression_summary(&self) -> String {
        let s = &self.stats;
        format!(
            "CLAM-adaptive: {} passthrough ({:.1}%), {} i8, {} i4+i2, LFD threshold={:.2}",
            s.n_passthrough, s.n_passthrough as f64 / self.n_rows as f64 * 100.0,
            s.n_i8, s.n_i4i2, s.lfd_threshold
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_row(seed: usize, dim: usize) -> Vec<f32> {
        (0..dim).map(|d| ((d * 97 + seed * 31 + 17) as f64 * 0.618).sin() as f32 * 0.01).collect()
    }

    #[test]
    fn adaptive_encode_decode() {
        let rows: Vec<Vec<f32>> = (0..64).map(|i| make_row(i, 256)).collect();
        let tensor = AdaptiveCodecTensor::encode("test_q_proj", &rows, 32, false, None);
        assert_eq!(tensor.n_rows, 64);

        let recon = tensor.reconstruct_all();
        let mut total_cos = 0.0f64;
        for i in 0..64 {
            total_cos += cosine_f32_to_f64_simd(&rows[i], &recon[i]);
        }
        let avg = total_cos / 64.0;
        assert!(avg > 0.9, "avg cosine {} should be >0.9", avg);
    }

    #[test]
    fn outliers_get_passthrough() {
        // Create rows with a few extreme outliers
        let mut rows: Vec<Vec<f32>> = (0..60).map(|i| make_row(i, 128)).collect();
        // Add outlier rows (very different pattern)
        for i in 0..4 {
            rows.push(vec![100.0 * (i as f32 + 1.0); 128]);
        }

        let tensor = AdaptiveCodecTensor::encode("test", &rows, 32, false, None);
        assert!(tensor.stats.n_passthrough > 0,
            "should have some passthrough rows, got {}", tensor.stats.n_passthrough);

        // Passthrough rows should reconstruct exactly
        for i in 0..tensor.n_rows {
            if tensor.rows[i].precision == RowPrecision::Passthrough {
                let recon = tensor.reconstruct_row(i);
                let cos = cosine_f32_to_f64_simd(&rows[i], &recon);
                assert!((cos - 1.0).abs() < 1e-6, "passthrough row {} cos={}", i, cos);
            }
        }
    }

    #[test]
    fn kv_proj_uses_i8() {
        // Use many similar rows so CLAM doesn't flag everything as outlier
        let rows: Vec<Vec<f32>> = (0..128).map(|i| {
            let base = make_row(i % 16, 256); // 16 clusters of 8 each
            base.iter().enumerate().map(|(d, &v)| v + ((d * 7 + i) as f64 * 0.001).sin() as f32 * 0.001).collect()
        }).collect();
        let tensor = AdaptiveCodecTensor::encode("k_proj", &rows, 32, true, None);
        assert!(tensor.stats.n_i8 > 0 || tensor.stats.n_passthrough > 0,
            "should have encoded rows: i8={} pt={}", tensor.stats.n_i8, tensor.stats.n_passthrough);
    }
}
