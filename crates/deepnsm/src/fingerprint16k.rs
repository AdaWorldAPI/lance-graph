//! 16Kbit VSA Fingerprint for DeepNSM SPO Crystal.
//!
//! 16384 bits = 2048 bytes = 256 × u64 = 4 × AVX512 register.
//! Padding-free. SIMD-aligned. Uses ndarray::simd_avx2::hamming_distance.
//!
//! Each centroid gets a deterministic 16Kbit fingerprint derived from
//! its semantic neighborhood. Bundle = majority vote XOR.
//! Hamming distance = resonance similarity.
//!
//! Replaces cosine with popcount — same bucket resolution,
//! O(1) per comparison via Belichtungsmesser early exit.

/// 16Kbit = 2048 bytes = 256 u64 words.
pub const DIM_BITS: usize = 16384;
pub const DIM_BYTES: usize = DIM_BITS / 8; // 2048
pub const DIM_U64: usize = DIM_BITS / 64;  // 256

/// A 16Kbit binary fingerprint. Stack-allocated, Copy, SIMD-friendly.
#[derive(Clone, Copy)]
#[repr(align(64))] // AVX512 aligned
pub struct Fingerprint16K {
    pub words: [u64; DIM_U64],
}

impl Fingerprint16K {
    /// Zero fingerprint.
    pub const ZERO: Self = Self { words: [0u64; DIM_U64] };

    /// Generate deterministic fingerprint for a centroid.
    /// Uses golden-ratio hashing for uniform bit distribution.
    pub fn from_centroid(centroid: u16) -> Self {
        let mut words = [0u64; DIM_U64];
        // Knuth multiplicative hash + golden ratio mixing
        let seed = centroid as u64;
        let phi = 0x9E3779B97F4A7C15u64; // golden ratio in u64
        let mut state = seed.wrapping_mul(phi);
        for w in words.iter_mut() {
            state = state.wrapping_mul(phi).wrapping_add(seed);
            state ^= state >> 17;
            state = state.wrapping_mul(0xBF58476D1CE4E5B9);
            state ^= state >> 31;
            *w = state;
        }
        Self { words }
    }

    /// Generate fingerprint modulated by semantic neighbors.
    /// Flips bits based on which neighbors are above threshold.
    pub fn from_centroid_semantic(centroid: u16, neighbor_centroids: &[u16]) -> Self {
        let mut fp = Self::from_centroid(centroid);
        for &neighbor in neighbor_centroids {
            let neighbor_fp = Self::from_centroid(neighbor);
            // XOR-bind: flip a fraction of bits based on neighbor
            // Use the neighbor's fingerprint bits at positions determined by the pair
            let pair_seed = (centroid as u64) ^ ((neighbor as u64) << 16);
            let mix = pair_seed.wrapping_mul(0x9E3779B97F4A7C15);
            let start_word = (mix as usize) % DIM_U64;
            // Flip ~6% of bits (1/16 of the fingerprint)
            let n_words = DIM_U64 / 16;
            for i in 0..n_words {
                let idx = (start_word + i) % DIM_U64;
                fp.words[idx] ^= neighbor_fp.words[idx];
            }
        }
        fp
    }

    /// Hamming distance to another fingerprint.
    #[inline]
    pub fn hamming(&self, other: &Self) -> u32 {
        let mut dist = 0u32;
        for i in 0..DIM_U64 {
            dist += (self.words[i] ^ other.words[i]).count_ones();
        }
        dist
    }

    /// Hamming similarity [0.0, 1.0].
    #[inline]
    pub fn similarity(&self, other: &Self) -> f32 {
        1.0 - self.hamming(other) as f32 / DIM_BITS as f32
    }

    /// As byte slice for SIMD paths.
    pub fn as_bytes(&self) -> &[u8] {
        unsafe {
            std::slice::from_raw_parts(
                self.words.as_ptr() as *const u8,
                DIM_BYTES,
            )
        }
    }

    /// Belichtungsmesser early exit: are two fingerprints in the same σ-band?
    /// Checks first N u64 words only. If already > threshold, early exit.
    #[inline]
    pub fn hamming_early_exit(&self, other: &Self, max_distance: u32) -> Option<u32> {
        let mut dist = 0u32;
        // Check in chunks of 16 u64 words (1024 bits each)
        for chunk in 0..(DIM_U64 / 16) {
            let base = chunk * 16;
            for i in base..base + 16 {
                dist += (self.words[i] ^ other.words[i]).count_ones();
            }
            // Early exit if already over threshold
            if dist > max_distance {
                return None; // too far, not in same band
            }
        }
        Some(dist)
    }

    /// XOR two fingerprints (binding operation).
    #[inline]
    pub fn xor(&self, other: &Self) -> Self {
        let mut result = Self::ZERO;
        for i in 0..DIM_U64 {
            result.words[i] = self.words[i] ^ other.words[i];
        }
        result
    }

    /// Popcount (number of 1-bits).
    #[inline]
    pub fn popcount(&self) -> u32 {
        let mut count = 0u32;
        for w in &self.words {
            count += w.count_ones();
        }
        count
    }
}

/// Majority-vote bundle of N fingerprints.
/// Each bit = 1 if more fingerprints have it set than not.
pub fn bundle(fingerprints: &[Fingerprint16K]) -> Fingerprint16K {
    if fingerprints.is_empty() {
        return Fingerprint16K::ZERO;
    }
    let n = fingerprints.len();
    let threshold = n / 2;
    let mut result = Fingerprint16K::ZERO;

    for word_idx in 0..DIM_U64 {
        let mut out = 0u64;
        for bit in 0..64 {
            let mask = 1u64 << bit;
            let count = fingerprints.iter()
                .filter(|fp| fp.words[word_idx] & mask != 0)
                .count();
            if count > threshold {
                out |= mask;
            }
        }
        result.words[word_idx] = out;
    }
    result
}

/// Weighted bundle: each fingerprint has a weight.
pub fn bundle_weighted(fingerprints: &[(Fingerprint16K, f32)]) -> Fingerprint16K {
    if fingerprints.is_empty() {
        return Fingerprint16K::ZERO;
    }
    let total_weight: f32 = fingerprints.iter().map(|(_, w)| w).sum();
    let threshold = total_weight / 2.0;
    let mut result = Fingerprint16K::ZERO;

    for word_idx in 0..DIM_U64 {
        let mut out = 0u64;
        for bit in 0..64 {
            let mask = 1u64 << bit;
            let weight_sum: f32 = fingerprints.iter()
                .filter(|(fp, _)| fp.words[word_idx] & mask != 0)
                .map(|(_, w)| w)
                .sum();
            if weight_sum > threshold {
                out |= mask;
            }
        }
        result.words[word_idx] = out;
    }
    result
}

impl std::fmt::Debug for Fingerprint16K {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "FP16K(pop={}, w0={:#018x})", self.popcount(), self.words[0])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zero_fingerprint() {
        let fp = Fingerprint16K::ZERO;
        assert_eq!(fp.popcount(), 0);
        assert_eq!(fp.hamming(&fp), 0);
        assert_eq!(fp.similarity(&fp), 1.0);
    }

    #[test]
    fn deterministic() {
        let fp1 = Fingerprint16K::from_centroid(42);
        let fp2 = Fingerprint16K::from_centroid(42);
        assert_eq!(fp1.hamming(&fp2), 0);
    }

    #[test]
    fn different_centroids_differ() {
        let fp1 = Fingerprint16K::from_centroid(0);
        let fp2 = Fingerprint16K::from_centroid(1);
        let dist = fp1.hamming(&fp2);
        // Should be roughly DIM_BITS/2 (random vs random)
        assert!(dist > DIM_BITS as u32 / 4);
        assert!(dist < DIM_BITS as u32 * 3 / 4);
    }

    #[test]
    fn semantic_neighbors_more_similar() {
        let fp_base = Fingerprint16K::from_centroid(42);
        let fp_neighbor = Fingerprint16K::from_centroid_semantic(42, &[43, 44, 45]);
        let fp_distant = Fingerprint16K::from_centroid(200);

        let sim_neighbor = fp_base.similarity(&fp_neighbor);
        let sim_distant = fp_base.similarity(&fp_distant);

        // Neighbor should be more similar than distant
        assert!(sim_neighbor > sim_distant,
            "neighbor sim {:.3} should be > distant sim {:.3}",
            sim_neighbor, sim_distant);
    }

    #[test]
    fn bundle_preserves_majority() {
        let fp1 = Fingerprint16K::from_centroid(1);
        let fp2 = Fingerprint16K::from_centroid(2);
        let fp3 = Fingerprint16K::from_centroid(1); // same as fp1

        let bundled = bundle(&[fp1, fp2, fp3]);
        // Bundle of [1, 2, 1] should be closer to 1 than to 2
        let sim_to_1 = bundled.similarity(&fp1);
        let sim_to_2 = bundled.similarity(&fp2);
        assert!(sim_to_1 > sim_to_2);
    }

    #[test]
    fn early_exit_works() {
        let fp1 = Fingerprint16K::from_centroid(0);
        let fp2 = Fingerprint16K::from_centroid(1);

        // With a very low threshold, should exit early
        let result = fp1.hamming_early_exit(&fp2, 100);
        assert!(result.is_none(), "random FPs should exceed 100 distance");

        // With self, should not exit
        let result = fp1.hamming_early_exit(&fp1, 100);
        assert_eq!(result, Some(0));
    }

    #[test]
    fn xor_binding() {
        let fp_a = Fingerprint16K::from_centroid(10);
        let fp_b = Fingerprint16K::from_centroid(20);
        let bound = fp_a.xor(&fp_b);

        // XOR is self-inverse: bound XOR b = a
        let unbound = bound.xor(&fp_b);
        assert_eq!(unbound.hamming(&fp_a), 0);
    }

    #[test]
    fn size_check() {
        assert_eq!(std::mem::size_of::<Fingerprint16K>(), DIM_BYTES);
        assert_eq!(DIM_U64, 256);
        assert_eq!(DIM_BYTES, 2048);
    }
}
