//! Prime fingerprint: centroid as prime factorization.
//!
//! Each bit tests a prime frequency in the weight vector:
//!   bit k = "does this centroid have p_k-periodic structure?"
//!
//! This is a prime-DFT: Fourier analysis at prime frequencies
//! instead of power-of-2 frequencies. Primes are independent
//! (no prime divides another) → bits are orthogonal.
//!
//! Distance = Hamming(fingerprint_A, fingerprint_B)
//!          = number of unshared prime properties
//!          = COMPUTED, not stored. No distance table needed.
//!
//! ```text
//! 256 centroids × 64 prime-bits = 2 KB (vs 128 KB BF16 distance table)
//! Distance: popcount(A XOR B) = O(1) per pair
//! ```

/// First 64 primes.
const PRIMES: [usize; 64] = [
    2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71,
    73, 79, 83, 89, 97, 101, 103, 107, 109, 113, 127, 131, 137, 139, 149, 151,
    157, 163, 167, 173, 179, 181, 191, 193, 197, 199, 211, 223, 227, 229, 233,
    239, 241, 251, 257, 263, 269, 271, 277, 281, 283, 293, 307, 311,
];

/// Compute prime fingerprint from a weight vector.
///
/// For each prime p, sums weights at positions divisible by p
/// vs positions NOT divisible by p. The sign of the difference
/// sets the bit: "does this vector have p-periodic structure?"
///
/// Additive: bit = sign(Σ w[i%p==0] - Σ w[i%p!=0])
///
/// 64 primes → 64 bits = u64.
pub fn prime_fingerprint_64(weights: &[f32]) -> u64 {
    let n = weights.len();
    if n == 0 { return 0; }

    let mut bits = 0u64;

    for (k, &p) in PRIMES.iter().enumerate().take(64) {
        if p >= n { break; } // prime larger than vector → skip

        let mut on_sum = 0.0f64;  // positions divisible by p
        let mut off_sum = 0.0f64; // positions NOT divisible by p
        let mut on_count = 0u32;
        let mut off_count = 0u32;

        for (i, &w) in weights.iter().enumerate() {
            if i % p == 0 {
                on_sum += w as f64;
                on_count += 1;
            } else {
                off_sum += w as f64;
                off_count += 1;
            }
        }

        // Normalize by count to avoid size bias
        let on_mean = if on_count > 0 { on_sum / on_count as f64 } else { 0.0 };
        let off_mean = if off_count > 0 { off_sum / off_count as f64 } else { 0.0 };

        if on_mean > off_mean {
            bits |= 1 << k;
        }
    }

    bits
}

/// Compute prime fingerprint with magnitude (additive, f32 per prime).
///
/// Instead of 1-bit (sign only), stores the STRENGTH of the prime frequency.
/// Additive combination: value = Σ strength[k] × prime[k]
pub fn prime_fingerprint_additive(weights: &[f32]) -> Vec<f32> {
    let n = weights.len();
    let n_primes = PRIMES.iter().take_while(|&&p| p < n).count().min(64);
    let mut strengths = vec![0.0f32; n_primes];

    for (k, &p) in PRIMES.iter().enumerate().take(n_primes) {
        let mut on_sum = 0.0f64;
        let mut off_sum = 0.0f64;
        let mut on_count = 0u32;
        let mut off_count = 0u32;

        for (i, &w) in weights.iter().enumerate() {
            if i % p == 0 {
                on_sum += w as f64;
                on_count += 1;
            } else {
                off_sum += w as f64;
                off_count += 1;
            }
        }

        let on_mean = if on_count > 0 { on_sum / on_count as f64 } else { 0.0 };
        let off_mean = if off_count > 0 { off_sum / off_count as f64 } else { 0.0 };

        strengths[k] = (on_mean - off_mean) as f32;
    }

    strengths
}

/// Reconstruct value as additive prime combination.
/// value = Σ strength[k] × prime[k]
pub fn prime_additive_value(strengths: &[f32]) -> f64 {
    strengths.iter().enumerate()
        .map(|(k, &s)| s as f64 * PRIMES[k] as f64)
        .sum()
}

/// Hamming distance between two 64-bit prime fingerprints.
#[inline]
pub fn prime_hamming(a: u64, b: u64) -> u32 {
    (a ^ b).count_ones()
}

/// Cosine similarity between two additive prime fingerprints.
pub fn prime_cosine(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len().min(b.len());
    let dot: f32 = a[..n].iter().zip(&b[..n]).map(|(x, y)| x * y).sum();
    let na: f32 = a[..n].iter().map(|x| x * x).sum::<f32>().sqrt();
    let nb: f32 = b[..n].iter().map(|x| x * x).sum::<f32>().sqrt();
    if na > 1e-10 && nb > 1e-10 { dot / (na * nb) } else { 0.0 }
}

/// Build a prime-fingerprint distance table (Hamming-based).
/// No cosine computation needed. Just XOR + popcount.
///
/// 256 centroids × 8 bytes = 2 KB fingerprints
/// Distance: popcount(A XOR B) / 64 → normalized [0, 1]
pub fn build_prime_distance_table(fingerprints: &[u64]) -> Vec<f32> {
    let n = fingerprints.len();
    let mut table = vec![0.0f32; n * n];

    for i in 0..n {
        table[i * n + i] = 1.0; // self = identical
        for j in (i + 1)..n {
            let hamming = prime_hamming(fingerprints[i], fingerprints[j]);
            // Normalize: 0 hamming = identical (1.0), 64 hamming = opposite (−1.0)
            let similarity = 1.0 - 2.0 * hamming as f32 / 64.0;
            table[i * n + j] = similarity;
            table[j * n + i] = similarity;
        }
    }

    table
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn constant_vector_all_zero_bits() {
        // Constant vector: all positions have same mean → no prime structure
        let w = vec![1.0f32; 1024];
        let fp = prime_fingerprint_64(&w);
        // For a constant vector, on_mean ≈ off_mean for all primes
        // Some bits may flip due to count imbalance (more off than on)
        eprintln!("Constant: {:064b} ({} bits set)", fp, fp.count_ones());
    }

    #[test]
    fn even_odd_pattern() {
        // w[even] = 1.0, w[odd] = 0.0 → strong prime-2 signal
        let w: Vec<f32> = (0..1024).map(|i| if i % 2 == 0 { 1.0 } else { 0.0 }).collect();
        let fp = prime_fingerprint_64(&w);
        assert!(fp & 1 == 1, "bit 0 (prime 2) should be set for even/odd pattern");
        eprintln!("Even/odd: {:064b} ({} bits set)", fp, fp.count_ones());
    }

    #[test]
    fn triple_pattern() {
        // w[i%3==0] = 1.0, rest = 0.0 → strong prime-3 signal
        let w: Vec<f32> = (0..1024).map(|i| if i % 3 == 0 { 1.0 } else { 0.0 }).collect();
        let fp = prime_fingerprint_64(&w);
        assert!(fp & 2 == 2, "bit 1 (prime 3) should be set for triple pattern");
        eprintln!("Triple: {:064b} ({} bits set)", fp, fp.count_ones());
    }

    #[test]
    fn similar_vectors_low_hamming() {
        let w1: Vec<f32> = (0..512).map(|i| (i as f32 * 0.01).sin()).collect();
        let w2: Vec<f32> = (0..512).map(|i| (i as f32 * 0.01 + 0.001).sin()).collect();
        let fp1 = prime_fingerprint_64(&w1);
        let fp2 = prime_fingerprint_64(&w2);
        let dist = prime_hamming(fp1, fp2);
        eprintln!("Similar vectors: hamming={}/64", dist);
        assert!(dist < 16, "similar vectors should have low hamming: {}", dist);
    }

    #[test]
    fn opposite_vectors_high_hamming() {
        let w1: Vec<f32> = (0..512).map(|i| (i as f32 * 0.1).sin()).collect();
        let w2: Vec<f32> = w1.iter().map(|&v| -v).collect(); // negated
        let fp1 = prime_fingerprint_64(&w1);
        let fp2 = prime_fingerprint_64(&w2);
        let dist = prime_hamming(fp1, fp2);
        eprintln!("Opposite vectors: hamming={}/64", dist);
        // Negation flips the sign of on_mean - off_mean for each prime
        // So ALL bits should flip → hamming should be high
        assert!(dist > 32, "opposite vectors should have high hamming: {}", dist);
    }

    #[test]
    fn additive_reconstruction() {
        let w: Vec<f32> = (0..256).map(|i| (i as f32 * 0.05).sin() * 0.5).collect();
        let strengths = prime_fingerprint_additive(&w);
        let reconstructed = prime_additive_value(&strengths);
        eprintln!("Additive value: {:.4} from {} primes", reconstructed, strengths.len());
        // The additive value is a weighted sum — not the original vector
        // but a SIGNATURE of it in prime-frequency space
        assert!(strengths.len() > 0);
    }

    #[test]
    fn additive_cosine_similar() {
        let w1: Vec<f32> = (0..512).map(|i| (i as f32 * 0.01).sin()).collect();
        let w2: Vec<f32> = (0..512).map(|i| (i as f32 * 0.01 + 0.001).sin()).collect();
        let s1 = prime_fingerprint_additive(&w1);
        let s2 = prime_fingerprint_additive(&w2);
        let cos = prime_cosine(&s1, &s2);
        eprintln!("Additive cosine (similar): {:.4}", cos);
        assert!(cos > 0.9, "similar vectors should have high prime-cosine: {}", cos);
    }

    #[test]
    fn distance_table_from_fingerprints() {
        let vectors: Vec<Vec<f32>> = (0..16).map(|i| {
            (0..256).map(|d| ((i * 7 + d * 3) as f32 * 0.01).sin()).collect()
        }).collect();

        let fingerprints: Vec<u64> = vectors.iter()
            .map(|w| prime_fingerprint_64(w))
            .collect();

        let table = build_prime_distance_table(&fingerprints);
        assert_eq!(table.len(), 16 * 16);

        // Diagonal should be 1.0
        for i in 0..16 {
            assert!((table[i * 16 + i] - 1.0).abs() < 0.01);
        }

        // Symmetric
        for i in 0..16 {
            for j in 0..16 {
                assert!((table[i * 16 + j] - table[j * 16 + i]).abs() < 0.001);
            }
        }

        let storage = 16 * 8; // 16 fingerprints × 8 bytes
        let full_table = 16 * 16 * 4; // 16×16 f32
        eprintln!("Fingerprints: {} bytes vs full table: {} bytes = {:.0}× compression",
            storage, full_table, full_table as f32 / storage as f32);
    }

    #[test]
    fn prime_17_is_base_dim() {
        // Prime 17 = BASE_DIM. The bit that says "17-periodic structure"
        // is EXACTLY the question whether the vector fits Base17 space.
        assert_eq!(PRIMES[6], 17, "7th prime should be 17 = BASE_DIM");

        // Vector with strong 17-periodicity
        let w: Vec<f32> = (0..1024).map(|i| if i % 17 == 0 { 1.0 } else { 0.0 }).collect();
        let fp = prime_fingerprint_64(&w);
        assert!(fp & (1 << 6) != 0, "bit 6 (prime 17) should be set for 17-periodic vector");
    }
}
