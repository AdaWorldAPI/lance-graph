//! XOR-adaptive codec — archetype XOR drives per-dimension precision.
//!
//! The XOR between a row's sign-bit fingerprint and its archetype's
//! fingerprint identifies exactly which dimensions flipped sign.
//! Sign-flipped dimensions get i8 precision (large residual).
//! Matching dimensions get i4 (small residual).
//!
//! No Hessian, no calibration data. The XOR IS the anomaly detector.

// wht_f32 reserved for future Hadamard-rotated XOR codec path
#[allow(unused_imports)]
use ndarray::hpc::fft::wht_f32;
use ndarray::hpc::cam_pq::kmeans;
// cosine_f32_to_f64_simd used by tests
#[allow(unused_imports)]
use ndarray::hpc::heel_f64x8::cosine_f32_to_f64_simd;
use crate::stacked_n::{bf16_to_f32, f32_to_bf16};

// Reserved for future Hadamard-rotation integration
#[allow(dead_code)]
fn next_pow2(n: usize) -> usize {
    let mut p = 1; while p < n { p *= 2; } p
}

fn sign_bits(v: &[f32]) -> Vec<u64> {
    let n_words = v.len().div_ceil(64);
    let mut bits = vec![0u64; n_words];
    for (i, &val) in v.iter().enumerate() {
        if val > 0.0 { bits[i / 64] |= 1u64 << (i % 64); }
    }
    bits
}

fn xor_delta(a: &[u64], b: &[u64]) -> Vec<u64> {
    a.iter().zip(b.iter()).map(|(x, y)| x ^ y).collect()
}

fn is_flipped(delta: &[u64], dim: usize) -> bool {
    (delta[dim / 64] >> (dim % 64)) & 1 == 1
}

fn popcount(bits: &[u64]) -> u32 {
    bits.iter().map(|w| w.count_ones()).sum()
}

#[derive(Clone, Debug)]
pub struct XorAdaptiveRow {
    pub centroid_idx: u16,
    pub n_flipped: u32,
    pub flipped_scale_bf16: u16,
    pub flipped_codes: Vec<i8>,
    pub flipped_indices: Vec<u16>,
    pub matched_scale_bf16: u16,
    pub matched_codes: Vec<u8>,
}

#[derive(Clone, Debug)]
pub struct XorAdaptiveTensor {
    pub role: String,
    pub n_rows: usize,
    pub n_cols: usize,
    pub centroids: Vec<Vec<f32>>,
    pub centroid_fps: Vec<Vec<u64>>,
    pub rows: Vec<XorAdaptiveRow>,
}

impl XorAdaptiveTensor {
    pub fn encode(role: &str, data: &[Vec<f32>], k: usize) -> Self {
        let n = data.len();
        let n_cols = if n > 0 { data[0].len() } else { 0 };
        let k = k.min(n).min(256);

        let centroids = kmeans(data, k, n_cols, 10);
        let centroid_fps: Vec<Vec<u64>> = centroids.iter().map(|c| sign_bits(c)).collect();

        let rows: Vec<XorAdaptiveRow> = data.iter().map(|row| {
            // Find nearest centroid
            let mut best_ci = 0;
            let mut best_d = f32::MAX;
            for (ci, c) in centroids.iter().enumerate() {
                let d: f32 = row.iter().zip(c.iter()).map(|(a, b)| (a - b) * (a - b)).sum();
                if d < best_d { best_d = d; best_ci = ci; }
            }

            let row_fp = sign_bits(row);
            let delta = xor_delta(&row_fp, &centroid_fps[best_ci]);
            let n_flipped = popcount(&delta);

            let residual: Vec<f32> = row.iter().zip(centroids[best_ci].iter())
                .map(|(a, b)| a - b).collect();

            // Split residual by XOR delta: flipped dims get i8, matched get i4
            let mut flipped_vals = Vec::new();
            let mut flipped_idx = Vec::new();
            let mut matched_vals = Vec::new();

            for (d, &res) in residual.iter().enumerate().take(n_cols) {
                if is_flipped(&delta, d) {
                    flipped_vals.push(res);
                    flipped_idx.push(d as u16);
                } else {
                    matched_vals.push(res);
                }
            }

            // Quantize flipped dims to i8 (high precision where sign differs)
            let flipped_max = flipped_vals.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
            let fs = if flipped_max > 1e-12 { 127.0 / flipped_max } else { 0.0 };
            let flipped_codes: Vec<i8> = flipped_vals.iter()
                .map(|v| (v * fs).round().clamp(-127.0, 127.0) as i8).collect();

            // Quantize matched dims to i4 (low precision where sign agrees)
            let matched_max = matched_vals.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
            let ms = if matched_max > 1e-12 { 7.0 / matched_max } else { 0.0 };
            let matched_packed: Vec<u8> = {
                let mut out = Vec::with_capacity(matched_vals.len().div_ceil(2));
                let mut i = 0;
                while i < matched_vals.len() {
                    let a = (matched_vals[i] * ms).round().clamp(-7.0, 7.0) as i8;
                    let b = if i + 1 < matched_vals.len() {
                        (matched_vals[i + 1] * ms).round().clamp(-7.0, 7.0) as i8
                    } else { 0 };
                    out.push(((a + 8) as u8) | (((b + 8) as u8) << 4));
                    i += 2;
                }
                out
            };

            XorAdaptiveRow {
                centroid_idx: best_ci as u16,
                n_flipped,
                flipped_scale_bf16: f32_to_bf16(if fs > 0.0 { flipped_max / 127.0 } else { 0.0 }),
                flipped_codes,
                flipped_indices: flipped_idx,
                matched_scale_bf16: f32_to_bf16(if ms > 0.0 { matched_max / 7.0 } else { 0.0 }),
                matched_codes: matched_packed,
            }
        }).collect();

        XorAdaptiveTensor { role: role.to_string(), n_rows: n, n_cols, centroids, centroid_fps, rows }
    }

    pub fn reconstruct_row(&self, i: usize) -> Vec<f32> {
        let row = &self.rows[i];
        let ci = row.centroid_idx as usize;
        let mut result = self.centroids[ci].clone();

        let fs = bf16_to_f32(row.flipped_scale_bf16);
        for (fi, &idx) in row.flipped_indices.iter().enumerate() {
            let d = idx as usize;
            if d < result.len() && fi < row.flipped_codes.len() {
                result[d] += row.flipped_codes[fi] as f32 * fs;
            }
        }

        let ms = bf16_to_f32(row.matched_scale_bf16);
        let mut mi = 0;
        for (d, res) in result.iter_mut().enumerate().take(self.n_cols) {
            if !row.flipped_indices.contains(&(d as u16)) {
                let byte_idx = mi / 2;
                if byte_idx < row.matched_codes.len() {
                    let nibble = if mi % 2 == 0 {
                        (row.matched_codes[byte_idx] & 0x0F) as i8 - 8
                    } else {
                        (row.matched_codes[byte_idx] >> 4) as i8 - 8
                    };
                    *res += nibble as f32 * ms;
                }
                mi += 1;
            }
        }

        result
    }

    pub fn reconstruct_all(&self) -> Vec<Vec<f32>> {
        (0..self.n_rows).map(|i| self.reconstruct_row(i)).collect()
    }

    pub fn avg_flipped_ratio(&self) -> f64 {
        if self.rows.is_empty() { return 0.0; }
        let total: f64 = self.rows.iter().map(|r| r.n_flipped as f64 / self.n_cols as f64).sum();
        total / self.rows.len() as f64
    }

    pub fn bytes_per_row_avg(&self) -> f64 {
        if self.rows.is_empty() { return 0.0; }
        let total: usize = self.rows.iter().map(|r| {
            2 + 2 + r.flipped_codes.len() + r.flipped_indices.len() * 2 + 2 + r.matched_codes.len()
        }).sum();
        total as f64 / self.rows.len() as f64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_row(seed: usize, dim: usize) -> Vec<f32> {
        (0..dim).map(|d| ((d * 97 + seed * 31 + 17) as f64 * 0.618).sin() as f32 * 0.01).collect()
    }

    #[test]
    fn xor_roundtrip_quality() {
        let rows: Vec<Vec<f32>> = (0..64).map(|i| make_row(i, 256)).collect();
        let tensor = XorAdaptiveTensor::encode("test", &rows, 32);
        let recon = tensor.reconstruct_all();
        let mut cos_sum = 0.0f64;
        for i in 0..64 {
            cos_sum += cosine_f32_to_f64_simd(&rows[i], &recon[i]);
        }
        assert!(cos_sum / 64.0 > 0.95, "avg cosine {} should be >0.95", cos_sum / 64.0);
    }

    #[test]
    fn flipped_ratio_reasonable() {
        let rows: Vec<Vec<f32>> = (0..32).map(|i| make_row(i, 128)).collect();
        let tensor = XorAdaptiveTensor::encode("test", &rows, 16);
        let ratio = tensor.avg_flipped_ratio();
        assert!(ratio > 0.0 && ratio < 1.0, "flip ratio {} should be in (0, 1)", ratio);
    }
}
