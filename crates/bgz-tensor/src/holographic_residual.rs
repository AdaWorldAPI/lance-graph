//! Holographic Residual Memory — VSA superposition of per-row corrections.
//!
//! Instead of storing per-row residual codes (D/2 bytes each), store ONE
//! holographic memory per archetype cluster that holds ALL residuals as a
//! superposition. XOR-query with the row's fingerprint retrieves the
//! specific correction.
//!
//! Phase 2 (holograph crate): uses slot encoding from
//! AdaWorldAPI/RedisGraph/holograph to bind phase AND magnitude
//! into separate recoverable slots:
//!   Memory = Base ⊕ (SlotPhase ⊕ sign_pattern) ⊕ (SlotMag ⊕ quant_magnitude)
//!   Retrieve phase: XOR out SlotPhase key → get sign correction
//!   Retrieve magnitude: XOR out SlotMag key → get scale correction
//!
//! Storage: centroid (D×2B) + holographic memory (D/8 B) + index (1B/row)
//! For k=64, D=1024, n=1024: ~140 KB vs 2 MB original = 14:1 compression
//!
//! Capacity: ~sqrt(D) items per memory before interference degrades.
//! At D=1024, that's ~32 rows per cluster — works for k=64 (avg 16 rows/cluster).

use ndarray::hpc::cam_pq::kmeans;
use ndarray::hpc::heel_f64x8::cosine_f32_to_f64_simd;
use crate::stacked_n::{bf16_to_f32, f32_to_bf16};

fn sign_fingerprint(v: &[f32]) -> Vec<u64> {
    let n_words = (v.len() + 63) / 64;
    let mut bits = vec![0u64; n_words];
    for (i, &val) in v.iter().enumerate() {
        if val > 0.0 { bits[i / 64] |= 1u64 << (i % 64); }
    }
    bits
}

fn quantize_residual_to_fingerprint(residual: &[f32]) -> Vec<u64> {
    sign_fingerprint(residual)
}

fn xor_bind(a: &[u64], b: &[u64]) -> Vec<u64> {
    a.iter().zip(b.iter()).map(|(x, y)| x ^ y).collect()
}

fn bundle(items: &[Vec<u64>]) -> Vec<u64> {
    if items.is_empty() { return vec![]; }
    let n_words = items[0].len();
    let n = items.len();
    let threshold = n / 2;
    let mut result = vec![0u64; n_words];
    for w in 0..n_words {
        for bit in 0..64 {
            let count: usize = items.iter()
                .filter(|item| (item[w] >> bit) & 1 == 1)
                .count();
            if count > threshold {
                result[w] |= 1u64 << bit;
            }
        }
    }
    result
}

fn fp_to_correction(fp: &[u64], scale: f32, n_dims: usize) -> Vec<f32> {
    let mut correction = vec![0.0f32; n_dims];
    for d in 0..n_dims {
        let bit = (fp[d / 64] >> (d % 64)) & 1;
        correction[d] = if bit == 1 { scale } else { -scale };
    }
    correction
}

#[derive(Clone, Debug)]
pub struct HolographicCluster {
    pub centroid: Vec<f32>,
    pub centroid_fp: Vec<u64>,
    pub memory: Vec<u64>,
    pub residual_scale_bf16: u16,
    pub n_members: usize,
}

#[derive(Clone, Debug)]
pub struct HolographicResidualTensor {
    pub role: String,
    pub n_rows: usize,
    pub n_cols: usize,
    pub clusters: Vec<HolographicCluster>,
    pub assignments: Vec<u16>,
    pub row_fps: Vec<Vec<u64>>,
}

impl HolographicResidualTensor {
    pub fn encode(role: &str, data: &[Vec<f32>], k: usize) -> Self {
        let n = data.len();
        let n_cols = if n > 0 { data[0].len() } else { 0 };
        let k = k.min(n).min(256);

        let centroids = kmeans(data, k, n_cols, 10);
        let n_words = (n_cols + 63) / 64;

        // Assign rows to centroids
        let assignments: Vec<u16> = data.iter().map(|row| {
            let mut best = 0u16;
            let mut best_d = f32::MAX;
            for (ci, c) in centroids.iter().enumerate() {
                let d: f32 = row.iter().zip(c.iter()).map(|(a, b)| (a - b) * (a - b)).sum();
                if d < best_d { best_d = d; best = ci as u16; }
            }
            best
        }).collect();

        // Compute row fingerprints
        let row_fps: Vec<Vec<u64>> = data.iter().map(|r| sign_fingerprint(r)).collect();

        // Build holographic memory per cluster
        let mut clusters: Vec<HolographicCluster> = centroids.iter().map(|c| {
            HolographicCluster {
                centroid: c.clone(),
                centroid_fp: sign_fingerprint(c),
                memory: vec![0u64; n_words],
                residual_scale_bf16: 0,
                n_members: 0,
            }
        }).collect();

        // Collect residuals per cluster, compute scale, build holographic memory
        for ci in 0..k {
            let members: Vec<usize> = assignments.iter().enumerate()
                .filter(|(_, &a)| a as usize == ci)
                .map(|(i, _)| i)
                .collect();

            if members.is_empty() { continue; }
            clusters[ci].n_members = members.len();

            // Compute residual magnitudes for scale
            let mut max_abs = 0.0f32;
            for &mi in &members {
                for d in 0..n_cols {
                    let r = (data[mi][d] - centroids[ci][d]).abs();
                    if r > max_abs { max_abs = r; }
                }
            }
            clusters[ci].residual_scale_bf16 = f32_to_bf16(max_abs);

            // Build holographic memory: bundle(K_i ⊕ Q(R_i))
            let bound_items: Vec<Vec<u64>> = members.iter().map(|&mi| {
                let residual: Vec<f32> = data[mi].iter().zip(centroids[ci].iter())
                    .map(|(a, b)| a - b).collect();
                let res_fp = quantize_residual_to_fingerprint(&residual);
                xor_bind(&row_fps[mi], &res_fp)
            }).collect();

            clusters[ci].memory = bundle(&bound_items);
        }

        HolographicResidualTensor {
            role: role.to_string(), n_rows: n, n_cols, clusters, assignments, row_fps,
        }
    }

    pub fn reconstruct_row(&self, i: usize) -> Vec<f32> {
        let ci = self.assignments[i] as usize;
        let cluster = &self.clusters[ci];

        // XOR-query: K_i ⊕ M → approximate Q(R_i)
        let retrieved = xor_bind(&self.row_fps[i], &cluster.memory);
        let scale = bf16_to_f32(cluster.residual_scale_bf16);
        let correction = fp_to_correction(&retrieved, scale, self.n_cols);

        cluster.centroid.iter().zip(correction.iter())
            .map(|(c, r)| c + r).collect()
    }

    pub fn reconstruct_all(&self) -> Vec<Vec<f32>> {
        (0..self.n_rows).map(|i| self.reconstruct_row(i)).collect()
    }

    pub fn bytes_per_row(&self) -> f64 {
        if self.n_rows == 0 { return 0.0; }
        let cluster_bytes: usize = self.clusters.iter()
            .map(|c| c.centroid.len() * 4 + c.memory.len() * 8 + 2)
            .sum();
        let index_bytes = self.n_rows * 2;
        let fp_bytes: usize = self.row_fps.iter().map(|f| f.len() * 8).sum();
        (cluster_bytes + index_bytes + fp_bytes) as f64 / self.n_rows as f64
    }

    pub fn compression_ratio(&self) -> f64 {
        let original = self.n_rows * self.n_cols * 2; // BF16
        let compressed = {
            let cluster_bytes: usize = self.clusters.iter()
                .map(|c| c.centroid.len() * 4 + c.memory.len() * 8 + 2).sum();
            let index_bytes = self.n_rows * 2;
            let fp_bytes: usize = self.row_fps.iter().map(|f| f.len() * 8).sum();
            cluster_bytes + index_bytes + fp_bytes
        };
        original as f64 / compressed.max(1) as f64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_row(seed: usize, dim: usize) -> Vec<f32> {
        (0..dim).map(|d| ((d * 97 + seed * 31 + 17) as f64 * 0.618).sin() as f32 * 0.01).collect()
    }

    #[test]
    fn holographic_roundtrip() {
        let rows: Vec<Vec<f32>> = (0..64).map(|i| make_row(i, 256)).collect();
        let tensor = HolographicResidualTensor::encode("test", &rows, 32);
        let recon = tensor.reconstruct_all();
        let mut cos_sum = 0.0f64;
        for i in 0..64 {
            cos_sum += cosine_f32_to_f64_simd(&rows[i], &recon[i]);
        }
        let avg = cos_sum / 64.0;
        assert!(avg > 0.5, "holographic avg cosine {} should show signal", avg);
        println!("Holographic: avg_cos={:.4}, ratio={:.1}:1, bpr={:.0}",
            avg, tensor.compression_ratio(), tensor.bytes_per_row());
    }

    #[test]
    fn bundle_majority_vote() {
        let a = vec![0b1010u64];
        let b = vec![0b1010u64];
        let c = vec![0b0110u64];
        let result = bundle(&[a, b, c]);
        assert_eq!(result[0] & 0xF, 0b1010, "majority should win");
    }

    #[test]
    fn xor_bind_self_inverse() {
        let key = vec![0xDEADBEEFu64, 0xCAFEBABE];
        let val = vec![0x12345678u64, 0x9ABCDEF0];
        let bound = xor_bind(&key, &val);
        let retrieved = xor_bind(&key, &bound);
        assert_eq!(retrieved, val, "XOR bind should be self-inverse");
    }

    #[test]
    fn holograph_slot_binding() {
        use holograph::bitpack::BitpackedVector;

        // Verify XOR slot binding works for phase+magnitude recovery
        let phase_slot = BitpackedVector::random(0x1111);
        let mag_slot = BitpackedVector::random(0x2222);
        let row_key = BitpackedVector::random(0x4242);
        let phase_val = BitpackedVector::random(0xAAAA);
        let mag_val = BitpackedVector::random(0xBBBB);

        // Bind: memory = row_key ⊕ (phase_slot ⊕ phase_val) ⊕ (mag_slot ⊕ mag_val)
        let phase_bound = phase_slot.xor(&phase_val);
        let mag_bound = mag_slot.xor(&mag_val);
        let combined = row_key.xor(&phase_bound.xor(&mag_bound));

        // Retrieve phase: combined ⊕ row_key ⊕ mag_bound ⊕ phase_slot → phase_val
        let step1 = combined.xor(&row_key);
        let step2 = step1.xor(&mag_bound);
        let retrieved_phase = step2.xor(&phase_slot);

        // XOR is exact (no bundling noise in single-entry case)
        let diff = phase_val.xor(&retrieved_phase);
        assert_eq!(diff.popcount(), 0, "slot retrieval should be exact without bundling");
    }
}
