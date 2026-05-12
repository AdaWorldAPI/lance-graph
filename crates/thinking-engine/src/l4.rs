//! # L4 Experience Vector — 16 KB personality.
//!
//! The PARTICLE side of wave-particle duality.
//!
//! L1-L3 are waves (cosine, interference, immutable tables).
//! L4 is particle (Hamming, XOR bind, mutable accumulator).
//!
//! L4 is OUTSIDE the L1→L2→L3 cascade. It fires as pre-perturbation
//! bias on sensor output and learns from L3 commits.
//!
//! Every thought-pair from L3 gets XOR-bound into 16,384 bits,
//! then accumulated into i8 counters with RL reward.

use std::io::{Read, Write};

/// Number of counters in the experience vector.
/// 16,384 = 16 KB of personality.
pub const ACCUM_LEN: usize = 16_384;

/// Number of bytes in a binarized vector.
/// 2048 bytes = 16,384 bits.
pub const BIN_BYTES: usize = 2048;

/// L4 Experience Vector — 16 KB personality.
/// Every thought-pair from L3 gets XOR-bound into 16,384 bits,
/// then accumulated into i8 counters with RL reward.
pub struct L4Experience {
    /// The personality. 16 KB of accumulated wave collapses.
    /// Each counter tracks reinforcement: positive = good, negative = bad.
    accum: [i8; ACCUM_LEN],
}

impl L4Experience {
    /// Create a blank personality — all zeros.
    pub fn new() -> Self {
        Self {
            accum: [0i8; ACCUM_LEN],
        }
    }

    /// Binarize a codebook centroid (f32 slice) into 16,384 bits.
    /// Sign-bit thresholding: bit = (value > 0.0) ? 1 : 0
    /// Pads or truncates to exactly 2048 bytes (16,384 bits).
    pub fn binarize(centroid: &[f32]) -> [u8; BIN_BYTES] {
        let mut out = [0u8; BIN_BYTES];
        let total_bits = BIN_BYTES * 8; // 16,384
        let n = centroid.len().min(total_bits);

        // Process 8 bits at a time for SIMD-friendly layout.
        let full_bytes = n / 8;
        for byte_idx in 0..full_bytes {
            let base = byte_idx * 8;
            let mut byte = 0u8;
            // MSB-first packing: bit 7 = first element in the group.
            byte |= ((centroid[base]     > 0.0) as u8) << 7;
            byte |= ((centroid[base + 1] > 0.0) as u8) << 6;
            byte |= ((centroid[base + 2] > 0.0) as u8) << 5;
            byte |= ((centroid[base + 3] > 0.0) as u8) << 4;
            byte |= ((centroid[base + 4] > 0.0) as u8) << 3;
            byte |= ((centroid[base + 5] > 0.0) as u8) << 2;
            byte |= ((centroid[base + 6] > 0.0) as u8) << 1;
            byte |= (centroid[base + 7] > 0.0) as u8;
            out[byte_idx] = byte;
        }

        // Handle remaining bits (< 8).
        let rem = n % 8;
        if rem > 0 {
            let base = full_bytes * 8;
            let mut byte = 0u8;
            for bit in 0..rem {
                byte |= ((centroid[base + bit] > 0.0) as u8) << (7 - bit);
            }
            out[full_bytes] = byte;
        }

        // Bytes beyond n/8 remain zero (padding).
        out
    }

    /// XOR bind two binary vectors = "this relationship".
    /// Commutative, self-inverse (a ⊕ a = 0).
    pub fn xor_bind(a: &[u8; BIN_BYTES], b: &[u8; BIN_BYTES]) -> [u8; BIN_BYTES] {
        let mut out = [0u8; BIN_BYTES];
        // Inner loop over 2048 bytes — auto-vectorizes well.
        for i in 0..BIN_BYTES {
            out[i] = a[i] ^ b[i];
        }
        out
    }

    /// Learn: accumulate reward into the experience vector.
    /// Where bound bit = 1: add reward. Where bit = 0: subtract reward.
    /// Saturating i8: -128 to +127.
    pub fn learn(&mut self, bound: &[u8; BIN_BYTES], reward: i8) {
        // Walk every bit of the bound vector.
        // SIMD-friendly: process byte-at-a-time, 8 counters per byte.
        for byte_idx in 0..BIN_BYTES {
            let byte = bound[byte_idx];
            let base = byte_idx * 8;

            // Unrolled 8 bits per byte for vectorization.
            for bit in 0..8u32 {
                let is_set = (byte >> (7 - bit)) & 1;
                let idx = base + bit as usize;
                if is_set == 1 {
                    self.accum[idx] = self.accum[idx].saturating_add(reward);
                } else {
                    self.accum[idx] = self.accum[idx].saturating_sub(reward);
                }
            }
        }
    }

    /// Recognize: dot product of bound vector with experience.
    /// High positive = "I've seen this before, it was good"
    /// Near zero = "new combination"
    /// Negative = "I've tried this, it was bad"
    pub fn recognize(&self, bound: &[u8; BIN_BYTES]) -> i32 {
        let mut score: i32 = 0;

        // Walk every bit. Where bit=1, add counter; where bit=0, subtract counter.
        for byte_idx in 0..BIN_BYTES {
            let byte = bound[byte_idx];
            let base = byte_idx * 8;

            for bit in 0..8u32 {
                let is_set = (byte >> (7 - bit)) & 1;
                let idx = base + bit as usize;
                let val = self.accum[idx] as i32;
                if is_set == 1 {
                    score += val;
                } else {
                    score -= val;
                }
            }
        }

        score
    }

    /// Bias sensor weights based on recognition.
    /// Returns a weight multiplier per codebook index:
    ///   >1.0 for familiar-good, <1.0 for familiar-bad, 1.0 for unknown.
    ///
    /// Pairs each consecutive pair of codebook indices, XOR-binds them,
    /// and maps recognition score to a sigmoid-like multiplier.
    pub fn bias_sensor(&self, codebook_indices: &[u16], centroids: &[Vec<f32>]) -> Vec<f64> {
        let n = codebook_indices.len();
        let mut weights = vec![1.0f64; n];

        if n < 2 || centroids.is_empty() {
            return weights;
        }

        // For each index, bind with its neighbor and check recognition.
        for i in 0..n {
            let j = if i + 1 < n { i + 1 } else { 0 };

            let idx_a = codebook_indices[i] as usize;
            let idx_b = codebook_indices[j] as usize;

            if idx_a >= centroids.len() || idx_b >= centroids.len() {
                continue;
            }

            let bin_a = Self::binarize(&centroids[idx_a]);
            let bin_b = Self::binarize(&centroids[idx_b]);
            let bound = Self::xor_bind(&bin_a, &bin_b);

            let score = self.recognize(&bound);

            // Map score to multiplier using a soft sigmoid.
            // score range is roughly [-16384*127, +16384*127].
            // Normalize to [-1, 1] then map to [0.5, 1.5].
            let norm = (score as f64) / (ACCUM_LEN as f64 * 32.0);
            let clamped = norm.clamp(-1.0, 1.0);
            weights[i] = 1.0 + 0.5 * clamped;
        }

        weights
    }

    /// Save personality to file (exactly 16,384 bytes).
    pub fn save(&self, path: &std::path::Path) -> std::io::Result<()> {
        let mut f = std::fs::File::create(path)?;
        // Cast &[i8; 16384] → &[u8; 16384] for writing.
        let bytes: &[u8] =
            unsafe { std::slice::from_raw_parts(self.accum.as_ptr() as *const u8, ACCUM_LEN) };
        f.write_all(bytes)?;
        f.flush()
    }

    /// Load personality from file.
    pub fn load(path: &std::path::Path) -> std::io::Result<Self> {
        let mut f = std::fs::File::open(path)?;
        let mut buf = [0u8; ACCUM_LEN];
        f.read_exact(&mut buf)?;
        // Reinterpret u8 → i8.
        let accum: [i8; ACCUM_LEN] =
            unsafe { std::mem::transmute(buf) };
        Ok(Self { accum })
    }

    /// Statistics: how many positive, negative, zero counters.
    pub fn stats(&self) -> (usize, usize, usize) {
        let mut pos = 0usize;
        let mut neg = 0usize;
        let mut zero = 0usize;
        for &v in self.accum.iter() {
            match v.cmp(&0) {
                std::cmp::Ordering::Greater => pos += 1,
                std::cmp::Ordering::Less => neg += 1,
                std::cmp::Ordering::Equal => zero += 1,
            }
        }
        (pos, neg, zero)
    }

    /// Entropy of the experience vector (how specialized).
    /// Uses the distribution of counter magnitudes.
    /// Low entropy = highly specialized. High entropy = diffuse experience.
    pub fn entropy(&self) -> f64 {
        // Build histogram of absolute values (0..=127 → 128 bins).
        let mut hist = [0u32; 128];
        for &v in self.accum.iter() {
            let mag = if v == i8::MIN { 127 } else { v.unsigned_abs() };
            hist[mag as usize] += 1;
        }

        let total = ACCUM_LEN as f64;
        let mut h = 0.0f64;
        for &count in hist.iter() {
            if count > 0 {
                let p = count as f64 / total;
                h -= p * p.ln();
            }
        }
        h
    }
}

impl Default for L4Experience {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_is_blank() {
        let l4 = L4Experience::new();
        assert!(l4.accum.iter().all(|&v| v == 0));
    }

    #[test]
    fn binarize_deterministic() {
        let data: Vec<f32> = (0..512).map(|i| (i as f32) - 256.0).collect();
        let a = L4Experience::binarize(&data);
        let b = L4Experience::binarize(&data);
        assert_eq!(a, b);
    }

    #[test]
    fn binarize_sign_threshold() {
        // 8 values: [pos, neg, pos, neg, pos, neg, pos, neg]
        let data = [1.0f32, -1.0, 2.0, -0.5, 0.1, -0.1, 3.0, -3.0];
        let bin = L4Experience::binarize(&data);
        // Expected bits: 1,0,1,0,1,0,1,0 = 0b10101010 = 0xAA
        assert_eq!(bin[0], 0xAA);

        // Zero should map to 0 (not > 0.0).
        let data2 = [0.0f32; 8];
        let bin2 = L4Experience::binarize(&data2);
        assert_eq!(bin2[0], 0x00);
    }

    #[test]
    fn xor_bind_self_is_zero() {
        let data: Vec<f32> = (0..512).map(|i| (i as f32) - 256.0).collect();
        let bin = L4Experience::binarize(&data);
        let xored = L4Experience::xor_bind(&bin, &bin);
        assert!(xored.iter().all(|&b| b == 0));
    }

    #[test]
    fn xor_bind_commutative() {
        let data_a: Vec<f32> = (0..1024).map(|i| (i as f32) - 512.0).collect();
        let data_b: Vec<f32> = (0..1024).map(|i| ((i * 7 + 3) as f32) - 500.0).collect();
        let a = L4Experience::binarize(&data_a);
        let b = L4Experience::binarize(&data_b);
        let ab = L4Experience::xor_bind(&a, &b);
        let ba = L4Experience::xor_bind(&b, &a);
        assert_eq!(ab, ba);
    }

    #[test]
    fn learn_positive_increases() {
        let mut l4 = L4Experience::new();
        let mut bound = [0u8; BIN_BYTES];
        bound[0] = 0xFF; // first 8 bits set

        l4.learn(&bound, 10);

        // Bits that were set should have +10.
        for i in 0..8 {
            assert_eq!(l4.accum[i], 10);
        }
        // Bits that were NOT set should have -10.
        for i in 8..16 {
            assert_eq!(l4.accum[i], -10);
        }
    }

    #[test]
    fn learn_negative_decreases() {
        let mut l4 = L4Experience::new();
        let mut bound = [0u8; BIN_BYTES];
        bound[0] = 0xFF;

        l4.learn(&bound, -5);

        // Bits set: add(-5) = -5.
        for i in 0..8 {
            assert_eq!(l4.accum[i], -5);
        }
        // Bits unset: sub(-5) = +5.
        for i in 8..16 {
            assert_eq!(l4.accum[i], 5);
        }
    }

    #[test]
    fn learn_saturates() {
        let mut l4 = L4Experience::new();
        let bound = [0xFF; BIN_BYTES]; // all bits set

        // Add 127 twice — should saturate at 127.
        l4.learn(&bound, 127);
        l4.learn(&bound, 127);
        assert_eq!(l4.accum[0], 127);

        // Reset and go negative.
        let mut l4 = L4Experience::new();
        let bound = [0x00; BIN_BYTES]; // all bits zero → subtract

        // Subtract 127 twice — should saturate at -128.
        l4.learn(&bound, 127);
        l4.learn(&bound, 127);
        assert_eq!(l4.accum[0], -128);
    }

    #[test]
    fn recognize_after_learn() {
        let mut l4 = L4Experience::new();

        let data_a: Vec<f32> = (0..512).map(|i| (i as f32) - 256.0).collect();
        let data_b: Vec<f32> = (0..512).map(|i| ((i * 3 + 1) as f32) - 200.0).collect();
        let a = L4Experience::binarize(&data_a);
        let b = L4Experience::binarize(&data_b);
        let bound = L4Experience::xor_bind(&a, &b);

        // Learn this pattern with positive reward.
        l4.learn(&bound, 10);
        l4.learn(&bound, 10);
        l4.learn(&bound, 10);

        let score = l4.recognize(&bound);
        // Score should be strongly positive.
        assert!(score > 0, "Expected positive score, got {}", score);
        // Each of 16384 counters contributes ±30, so max = 16384*30.
        // Score should be substantial.
        assert!(score > 1000, "Expected high score, got {}", score);
    }

    #[test]
    fn recognize_unknown_is_zero() {
        let l4 = L4Experience::new();

        let data: Vec<f32> = (0..512).map(|i| (i as f32) - 256.0).collect();
        let bin = L4Experience::binarize(&data);
        let bound = L4Experience::xor_bind(&bin, &bin); // self-bind = zero

        let score = l4.recognize(&bound);
        assert_eq!(score, 0);
    }

    #[test]
    fn save_load_roundtrip() {
        let mut l4 = L4Experience::new();
        let bound = [0xAA; BIN_BYTES];
        l4.learn(&bound, 42);

        let path = std::path::PathBuf::from("/tmp/l4_test_personality.bin");
        l4.save(&path).expect("save failed");

        let loaded = L4Experience::load(&path).expect("load failed");
        assert_eq!(l4.accum, loaded.accum);

        // Clean up.
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn bias_sensor_familiar() {
        let mut l4 = L4Experience::new();

        // Create two centroids.
        let c0: Vec<f32> = (0..256).map(|i| (i as f32) - 128.0).collect();
        let c1: Vec<f32> = (0..256).map(|i| ((i * 5 + 7) as f32) - 600.0).collect();
        let centroids = vec![c0.clone(), c1.clone()];

        // Learn the pair (0,1) with strong positive reward.
        let bin0 = L4Experience::binarize(&c0);
        let bin1 = L4Experience::binarize(&c1);
        let bound = L4Experience::xor_bind(&bin0, &bin1);
        for _ in 0..50 {
            l4.learn(&bound, 10);
        }

        let indices = vec![0u16, 1];
        let weights = l4.bias_sensor(&indices, &centroids);

        assert_eq!(weights.len(), 2);
        // Familiar-good patterns should be boosted above 1.0.
        assert!(weights[0] > 1.0, "Expected boost, got {}", weights[0]);
    }

    #[test]
    fn stats_after_learning() {
        let mut l4 = L4Experience::new();

        // Initially all zero.
        let (p, n, z) = l4.stats();
        assert_eq!(p, 0);
        assert_eq!(n, 0);
        assert_eq!(z, ACCUM_LEN);

        // Learn with a pattern that has some bits set.
        let mut bound = [0u8; BIN_BYTES];
        bound[0] = 0xFF; // 8 bits set
        l4.learn(&bound, 1);

        let (p, n, z) = l4.stats();
        // 8 bits set → +1, rest (16376) bits unset → -1.
        assert_eq!(p, 8);
        assert_eq!(n, ACCUM_LEN - 8);
        assert_eq!(z, 0);
        assert_eq!(p + n + z, ACCUM_LEN);
    }
}
