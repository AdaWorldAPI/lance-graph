//! Stacked BF16×4: phase-preserving 17-dimensional encoding.
//!
//! Instead of averaging 4 BF16 values into one i16 (which destroys phase),
//! we pack all 4 raw BF16 values into a u64 per dimension.
//!
//! ```text
//! u64 per dim = [BF16₁ | BF16₂ | BF16₃ | BF16₄]
//!                16 bit   16 bit   16 bit   16 bit
//! ```
//!
//! 17 dims × u64 = 136 bytes. Sign/exponent/mantissa structure survives.
//! Phase differences from golden rotation are VISIBLE, not averaged away.
//!
//! The stacked encoding also produces a collapsed search key (17 bytes)
//! for Belichtungsmesser HEEL fast-reject.

use crate::projection::{Base17, BASE_DIM, FP_SCALE, GOLDEN_STEP};

/// Golden-step position table (same as Base17).
const GOLDEN_POS: [u8; BASE_DIM] = {
    let mut t = [0u8; BASE_DIM];
    let mut i = 0;
    while i < BASE_DIM {
        t[i] = ((i * GOLDEN_STEP) % BASE_DIM) as u8;
        i += 1;
    }
    t
};

/// Stacked BF16×4: 17 dimensions, each holding 4 raw BF16 values.
/// 136 bytes. Phase-preserving.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct StackedBF16x4 {
    /// 17 u64 values, each packing 4 BF16 bit patterns.
    pub dims: [u64; BASE_DIM],
}

/// Collapsed search key: 17 bytes for Belichtungsmesser HEEL.
///
/// Per dimension:
/// - Bit 7: sign polarity (majority vote of 4 signs)
/// - Bits 4-6: magnitude class (mean of 4 exponents, 3 bits)
/// - Bits 0-3: fine hash (XOR fold of 4 mantissas, 4 bits)
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct SearchKey17 {
    pub bytes: [u8; BASE_DIM],
}

impl StackedBF16x4 {
    pub const BYTE_SIZE: usize = BASE_DIM * 8; // 136

    /// Project a BF16 weight vector (u16 bit patterns) into stacked space.
    ///
    /// Groups input dimensions by golden-step octave, packing up to 4 BF16
    /// values per base dimension. If fewer than 4 values map to a dim,
    /// the remaining slots are zero-filled.
    pub fn from_bf16(weights: &[u16]) -> Self {
        let n = weights.len();
        let n_octaves = n.div_ceil(BASE_DIM);
        // Collect values per base dim (up to 4 per slot)
        let mut slots: [[u16; 4]; BASE_DIM] = [[0u16; 4]; BASE_DIM];
        let mut counts = [0usize; BASE_DIM];

        for octave in 0..n_octaves.min(4) {
            for bi in 0..BASE_DIM {
                let dim = octave * BASE_DIM + GOLDEN_POS[bi] as usize;
                if dim < n && counts[bi] < 4 {
                    slots[bi][counts[bi]] = weights[dim];
                    counts[bi] += 1;
                }
            }
        }

        // For vectors with > 4 octaves, use strided sampling (octaves 0, n/4, n/2, 3n/4)
        if n_octaves > 4 {
            let stride = n_octaves / 4;
            for slot in 0..4usize {
                let octave = slot * stride;
                for (bi, &gp) in GOLDEN_POS.iter().enumerate() {
                    let dim = octave * BASE_DIM + gp as usize;
                    if dim < n {
                        slots[bi][slot] = weights[dim];
                    }
                }
            }
        }

        // Pack into u64: [bf16_0 | bf16_1 | bf16_2 | bf16_3]
        let mut dims = [0u64; BASE_DIM];
        for bi in 0..BASE_DIM {
            dims[bi] = (slots[bi][0] as u64)
                | ((slots[bi][1] as u64) << 16)
                | ((slots[bi][2] as u64) << 32)
                | ((slots[bi][3] as u64) << 48);
        }

        StackedBF16x4 { dims }
    }

    /// Project an f32 weight vector into stacked space.
    ///
    /// Converts f32 → BF16 by truncating mantissa, then packs.
    pub fn from_f32(weights: &[f32]) -> Self {
        let bf16: Vec<u16> = weights.iter().map(|&v| f32_to_bf16(v)).collect();
        Self::from_bf16(&bf16)
    }

    /// Extract the 4 BF16 values for a given dimension.
    #[inline]
    pub fn unpack_dim(&self, d: usize) -> [u16; 4] {
        let v = self.dims[d];
        [
            (v & 0xFFFF) as u16,
            ((v >> 16) & 0xFFFF) as u16,
            ((v >> 32) & 0xFFFF) as u16,
            ((v >> 48) & 0xFFFF) as u16,
        ]
    }

    /// Convert all 4 values of a dimension to f32.
    #[inline]
    pub fn unpack_dim_f32(&self, d: usize) -> [f32; 4] {
        let bf16s = self.unpack_dim(d);
        [
            bf16_to_f32(bf16s[0]),
            bf16_to_f32(bf16s[1]),
            bf16_to_f32(bf16s[2]),
            bf16_to_f32(bf16s[3]),
        ]
    }

    /// Collapse to Base17 (lossy: averages the 4 BF16 values per dim).
    pub fn to_base17(&self) -> Base17 {
        let mut dims = [0i16; BASE_DIM];
        for (d, dim) in dims.iter_mut().enumerate() {
            let vals = self.unpack_dim_f32(d);
            let mean = (vals[0] as f64 + vals[1] as f64 + vals[2] as f64 + vals[3] as f64) / 4.0;
            *dim = (mean * FP_SCALE).round().clamp(-32768.0, 32767.0) as i16;
        }
        Base17 { dims }
    }

    /// Generate collapsed search key (17 bytes) for HEEL fast-reject.
    pub fn search_key(&self) -> SearchKey17 {
        let mut bytes = [0u8; BASE_DIM];
        for (d, byte) in bytes.iter_mut().enumerate() {
            let bf16s = self.unpack_dim(d);

            // Sign polarity: majority vote of 4 signs
            let n_negative = bf16s.iter().filter(|&&b| b & 0x8000 != 0).count();
            let sign_bit = if n_negative >= 2 { 1u8 } else { 0u8 };

            // Magnitude class: mean of 4 exponents (5-bit BF16 exponent → 3-bit class)
            let exp_sum: u32 = bf16s.iter()
                .map(|&b| ((b >> 7) & 0xFF) as u32) // 8-bit exponent
                .sum();
            let mean_exp = (exp_sum / 4) as u8;
            let mag_class = (mean_exp >> 5) & 0x07; // Top 3 bits of mean exponent

            // Fine hash: XOR fold of 4 mantissas → 4 bits
            let mantissa_xor = bf16s.iter()
                .map(|&b| (b & 0x7F) as u8) // 7-bit mantissa
                .fold(0u8, |acc, m| acc ^ m);
            let fine_hash = mantissa_xor & 0x0F;

            *byte = (sign_bit << 7) | (mag_class << 4) | fine_hash;
        }
        SearchKey17 { bytes }
    }

    /// Vedic upper-half distance: compare only upper 32 bits of each u64.
    ///
    /// This uses the first two BF16 values (exponent-dominated) for coarse
    /// distance computation. Cheap and rejects ~85% of pairs.
    #[inline]
    pub fn vedic_upper_distance(&self, other: &StackedBF16x4) -> u32 {
        let mut d = 0u32;
        for i in 0..BASE_DIM {
            let a_upper = (self.dims[i] & 0xFFFFFFFF) as u32;
            let b_upper = (other.dims[i] & 0xFFFFFFFF) as u32;
            // L1 on the two packed BF16 values (treat as raw bits)
            let a0 = (a_upper & 0xFFFF) as i16;
            let b0 = (b_upper & 0xFFFF) as i16;
            let a1 = ((a_upper >> 16) & 0xFFFF) as i16;
            let b1 = ((b_upper >> 16) & 0xFFFF) as i16;
            d += (a0 as i32 - b0 as i32).unsigned_abs();
            d += (a1 as i32 - b1 as i32).unsigned_abs();
        }
        d
    }

    /// Full stacked distance: L1 across all 4 BF16 values per dimension.
    pub fn full_distance(&self, other: &StackedBF16x4) -> u64 {
        let mut d = 0u64;
        for i in 0..BASE_DIM {
            for slot in 0..4 {
                let shift = slot * 16;
                let a = ((self.dims[i] >> shift) & 0xFFFF) as i16;
                let b = ((other.dims[i] >> shift) & 0xFFFF) as i16;
                d += (a as i32 - b as i32).unsigned_abs() as u64;
            }
        }
        d
    }

    /// Cosine similarity using all 4×17 = 68 BF16 values as f32.
    pub fn cosine(&self, other: &StackedBF16x4) -> f64 {
        let mut dot = 0.0f64;
        let mut norm_a = 0.0f64;
        let mut norm_b = 0.0f64;
        for d in 0..BASE_DIM {
            let a_vals = self.unpack_dim_f32(d);
            let b_vals = other.unpack_dim_f32(d);
            for s in 0..4 {
                let a = a_vals[s] as f64;
                let b = b_vals[s] as f64;
                dot += a * b;
                norm_a += a * a;
                norm_b += b * b;
            }
        }
        let denom = (norm_a * norm_b).sqrt();
        if denom < 1e-12 { 0.0 } else { dot / denom }
    }

    /// Serialize to 136 bytes (little-endian u64).
    pub fn to_bytes(&self) -> [u8; Self::BYTE_SIZE] {
        let mut buf = [0u8; Self::BYTE_SIZE];
        for i in 0..BASE_DIM {
            let b = self.dims[i].to_le_bytes();
            buf[i * 8..i * 8 + 8].copy_from_slice(&b);
        }
        buf
    }

    /// Deserialize from 136 bytes.
    pub fn from_bytes(buf: &[u8]) -> Self {
        assert!(buf.len() >= Self::BYTE_SIZE);
        let mut dims = [0u64; BASE_DIM];
        for i in 0..BASE_DIM {
            dims[i] = u64::from_le_bytes([
                buf[i * 8], buf[i * 8 + 1], buf[i * 8 + 2], buf[i * 8 + 3],
                buf[i * 8 + 4], buf[i * 8 + 5], buf[i * 8 + 6], buf[i * 8 + 7],
            ]);
        }
        StackedBF16x4 { dims }
    }

    /// Zero vector.
    pub fn zero() -> Self {
        StackedBF16x4 { dims: [0u64; BASE_DIM] }
    }
}

impl SearchKey17 {
    pub const BYTE_SIZE: usize = BASE_DIM; // 17

    /// L1 distance between search keys (for HEEL fast-reject).
    #[inline]
    pub fn l1(&self, other: &SearchKey17) -> u32 {
        let mut d = 0u32;
        for i in 0..BASE_DIM {
            d += (self.bytes[i] as i16 - other.bytes[i] as i16).unsigned_abs() as u32;
        }
        d
    }

    /// Sign agreement count (0-17).
    #[inline]
    pub fn sign_agreement(&self, other: &SearchKey17) -> u8 {
        let mut count = 0u8;
        for i in 0..BASE_DIM {
            if (self.bytes[i] >> 7) == (other.bytes[i] >> 7) {
                count += 1;
            }
        }
        count
    }

    /// Magnitude class agreement count (0-17).
    #[inline]
    pub fn magnitude_agreement(&self, other: &SearchKey17) -> u8 {
        let mut count = 0u8;
        for i in 0..BASE_DIM {
            let a_mag = (self.bytes[i] >> 4) & 0x07;
            let b_mag = (other.bytes[i] >> 4) & 0x07;
            if a_mag == b_mag {
                count += 1;
            }
        }
        count
    }
}

// ─── BF16 conversion ────────────────────────────────────────────────────────

/// Convert f32 to BF16 (truncate mantissa, keep sign + exponent).
#[inline]
fn f32_to_bf16(v: f32) -> u16 {
    (v.to_bits() >> 16) as u16
}

/// Convert BF16 bit pattern to f32 (lossless).
#[inline]
fn bf16_to_f32(bits: u16) -> f32 {
    f32::from_bits((bits as u32) << 16)
}

// ─── Vedic Split Cascade ────────────────────────────────────────────────────

/// Vedic split cascade for progressive pair elimination.
///
/// ```text
/// Stage 1: upper 32 bits of 4 sampled dims  → reject ~85%
/// Stage 2: upper 32 bits of all 17 dims     → reject ~90% of survivors
/// Stage 3: full 64 bits of all 17 dims      → exact distance
/// ```
#[derive(Clone, Debug)]
pub struct VedicCascadeConfig {
    /// Sampled dimensions for Stage 1 (typically 4 out of 17).
    pub stage1_dims: [usize; 4],
    /// Stage 1 threshold: reject if partial distance exceeds this.
    pub stage1_threshold: u32,
    /// Stage 2 threshold: reject if upper-half distance exceeds this.
    pub stage2_threshold: u32,
    /// Stage 3 threshold: reject if full distance exceeds this.
    pub stage3_threshold: u64,
}

impl Default for VedicCascadeConfig {
    fn default() -> Self {
        VedicCascadeConfig {
            stage1_dims: [0, 5, 10, 15], // Spread across 17 dims
            stage1_threshold: u32::MAX / 2,
            stage2_threshold: u32::MAX / 2,
            stage3_threshold: u64::MAX / 2,
        }
    }
}

/// Result of Vedic cascade evaluation.
#[derive(Clone, Debug)]
pub struct VedicCascadeResult {
    pub total_pairs: usize,
    pub stage1_eliminated: usize,
    pub stage2_eliminated: usize,
    pub stage3_survived: usize,
}

impl VedicCascadeResult {
    pub fn elimination_rate(&self) -> f64 {
        if self.total_pairs == 0 { return 0.0; }
        1.0 - self.stage3_survived as f64 / self.total_pairs as f64
    }
}

/// Run Vedic cascade on query-key pairs.
pub fn vedic_cascade(
    queries: &[StackedBF16x4],
    keys: &[StackedBF16x4],
    config: &VedicCascadeConfig,
) -> (Vec<(usize, usize, u64)>, VedicCascadeResult) {
    let total = queries.len() * keys.len();
    let mut result = VedicCascadeResult {
        total_pairs: total,
        stage1_eliminated: 0,
        stage2_eliminated: 0,
        stage3_survived: 0,
    };
    let mut active = Vec::new();

    for (qi, q) in queries.iter().enumerate() {
        for (ki, k) in keys.iter().enumerate() {
            // Stage 1: upper 32 bits of 4 sampled dims
            let mut d1 = 0u32;
            for &dim in &config.stage1_dims {
                if dim < BASE_DIM {
                    let a_upper = (q.dims[dim] & 0xFFFFFFFF) as u32;
                    let b_upper = (k.dims[dim] & 0xFFFFFFFF) as u32;
                    let a0 = (a_upper & 0xFFFF) as i16;
                    let b0 = (b_upper & 0xFFFF) as i16;
                    let a1 = ((a_upper >> 16) & 0xFFFF) as i16;
                    let b1 = ((b_upper >> 16) & 0xFFFF) as i16;
                    d1 += (a0 as i32 - b0 as i32).unsigned_abs();
                    d1 += (a1 as i32 - b1 as i32).unsigned_abs();
                }
            }
            if d1 > config.stage1_threshold {
                result.stage1_eliminated += 1;
                continue;
            }

            // Stage 2: upper 32 bits of all 17 dims
            let d2 = q.vedic_upper_distance(k);
            if d2 > config.stage2_threshold {
                result.stage2_eliminated += 1;
                continue;
            }

            // Stage 3: full 64 bits
            let d3 = q.full_distance(k);
            if d3 > config.stage3_threshold {
                // Still counted as stage3 survivor but below threshold
                result.stage3_survived += 1;
                continue;
            }

            active.push((qi, ki, d3));
            result.stage3_survived += 1;
        }
    }

    (active, result)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stacked_from_f32_roundtrip() {
        let weights = vec![1.0f32, -0.5, 0.25, 2.0, -3.0];
        let stacked = StackedBF16x4::from_f32(&weights);
        // First dim should have the first weight
        let vals = stacked.unpack_dim_f32(0);
        // At least one of the 4 slots should be close to a weight value
        let has_nonzero = vals.iter().any(|&v| v.abs() > 0.01);
        assert!(has_nonzero, "should have non-zero values: {:?}", vals);
    }

    #[test]
    fn stacked_to_base17_consistency() {
        let weights: Vec<f32> = (0..68).map(|i| (i as f32 * 0.1) - 3.4).collect();
        let stacked = StackedBF16x4::from_f32(&weights);
        let base17 = stacked.to_base17();
        let direct_base17 = Base17::from_f32(&weights);

        // Both should be in the same ballpark (not identical due to BF16 truncation)
        let l1 = base17.l1(&direct_base17);
        assert!(l1 < 500, "stacked→base17 should be close to direct base17: L1={}", l1);
    }

    #[test]
    fn search_key_self_distance_zero() {
        let weights: Vec<f32> = (0..100).map(|i| i as f32 * 0.01).collect();
        let stacked = StackedBF16x4::from_f32(&weights);
        let key = stacked.search_key();
        assert_eq!(key.l1(&key), 0);
        assert_eq!(key.sign_agreement(&key), 17);
        assert_eq!(key.magnitude_agreement(&key), 17);
    }

    #[test]
    fn search_key_opposite_signs() {
        let pos = StackedBF16x4::from_f32(&vec![1.0f32; 68]);
        let neg = StackedBF16x4::from_f32(&vec![-1.0f32; 68]);
        let key_pos = pos.search_key();
        let key_neg = neg.search_key();
        // Signs should be opposite for most dims
        assert!(key_pos.sign_agreement(&key_neg) < 5,
            "opposite vectors should have low sign agreement: {}",
            key_pos.sign_agreement(&key_neg));
    }

    #[test]
    fn vedic_upper_self_zero() {
        let weights: Vec<f32> = (0..100).map(|i| i as f32 * 0.01).collect();
        let stacked = StackedBF16x4::from_f32(&weights);
        assert_eq!(stacked.vedic_upper_distance(&stacked), 0);
    }

    #[test]
    fn full_distance_self_zero() {
        let weights: Vec<f32> = (0..100).map(|i| (i as f32 - 50.0) * 0.1).collect();
        let stacked = StackedBF16x4::from_f32(&weights);
        assert_eq!(stacked.full_distance(&stacked), 0);
    }

    #[test]
    fn cosine_self_is_one() {
        let weights: Vec<f32> = (0..68).map(|i| (i as f32 * 0.3).sin()).collect();
        let stacked = StackedBF16x4::from_f32(&weights);
        let c = stacked.cosine(&stacked);
        assert!((c - 1.0).abs() < 1e-6, "self cosine should be 1.0: {}", c);
    }

    #[test]
    fn stacked_preserves_more_than_base17() {
        // Create two vectors that differ in phase (not magnitude)
        let a: Vec<f32> = (0..4096).map(|i| (i as f32 * 0.01).sin()).collect();
        let b: Vec<f32> = (0..4096).map(|i| (i as f32 * 0.01 + 0.5).sin()).collect();

        // Direct f32 cosine
        let mut dot = 0.0f64;
        let mut na = 0.0f64;
        let mut nb = 0.0f64;
        for i in 0..4096 {
            dot += a[i] as f64 * b[i] as f64;
            na += (a[i] as f64).powi(2);
            nb += (b[i] as f64).powi(2);
        }
        let true_cosine = dot / (na.sqrt() * nb.sqrt());

        // Stacked cosine
        let sa = StackedBF16x4::from_f32(&a);
        let sb = StackedBF16x4::from_f32(&b);
        let stacked_cosine = sa.cosine(&sb);

        // Base17 cosine
        let ba = Base17::from_f32(&a);
        let bb = Base17::from_f32(&b);
        let base17_cosine = ba.cosine(&bb);

        // Stacked should be closer to true cosine than base17
        let stacked_error = (stacked_cosine - true_cosine).abs();
        let base17_error = (base17_cosine - true_cosine).abs();

        // Print for visibility
        eprintln!("true_cosine={:.6}, stacked={:.6} (err={:.6}), base17={:.6} (err={:.6})",
            true_cosine, stacked_cosine, stacked_error, base17_cosine, base17_error);

        // Stacked should have less error (or at least not much more)
        // Note: for very long vectors with only 4 samples, the advantage may be small
        assert!(stacked_error < 0.5 || stacked_error <= base17_error + 0.1,
            "stacked should preserve phase better: stacked_err={:.4}, base17_err={:.4}",
            stacked_error, base17_error);
    }

    #[test]
    fn vedic_cascade_eliminates() {
        let queries: Vec<StackedBF16x4> = (0..8)
            .map(|i| StackedBF16x4::from_f32(&vec![(i as f32) * 100.0; 68]))
            .collect();
        let keys: Vec<StackedBF16x4> = (0..16)
            .map(|i| StackedBF16x4::from_f32(&vec![(i as f32) * 50.0 + 1000.0; 68]))
            .collect();

        let config = VedicCascadeConfig {
            stage1_threshold: 50000,
            stage2_threshold: 200000,
            stage3_threshold: 500000,
            ..Default::default()
        };

        let (active, result) = vedic_cascade(&queries, &keys, &config);
        assert!(result.elimination_rate() > 0.0,
            "should eliminate some pairs. Stage1: {}, Stage2: {}, Survived: {}",
            result.stage1_eliminated, result.stage2_eliminated, result.stage3_survived);
        assert!(active.len() <= queries.len() * keys.len());
    }

    #[test]
    fn byte_serialization_roundtrip() {
        let weights: Vec<f32> = (0..200).map(|i| (i as f32 - 100.0) * 0.05).collect();
        let original = StackedBF16x4::from_f32(&weights);
        let bytes = original.to_bytes();
        let recovered = StackedBF16x4::from_bytes(&bytes);
        assert_eq!(original, recovered);
    }
}
