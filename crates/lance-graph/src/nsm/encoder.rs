// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Encoder bridging CAM-PQ codebook distances to semantic triple comparison.
//!
//! The core data structure is `WordDistanceMatrix`: a 4096x4096 upper-triangle
//! packed matrix of u8 palette-quantized distances. O(1) per lookup.

use super::parser::SpoTriple;

/// Maximum vocabulary size (12-bit ranks).
pub const MAX_VOCAB: usize = 4096;

/// Number of NSM semantic primes (Wierzbicka).
pub const NUM_PRIMES: usize = 63;

/// Number of semantic role vectors.
pub const NUM_ROLES: usize = 6;

/// Size of each role vector in bytes.
pub const ROLE_VECTOR_BYTES: usize = 1250;

/// Word distance matrix: palette-quantized (u8) upper-triangle packed.
///
/// For N=4096 words, the upper triangle has N*(N-1)/2 = ~8.4M entries.
/// Each entry is a u8 distance (0=identical, 255=maximally different).
#[derive(Clone, Debug)]
pub struct WordDistanceMatrix {
    /// Upper triangle storage. Index = row*(2*N-row-1)/2 + (col-row-1).
    data: Vec<u8>,
    /// Number of words (rows/cols).
    size: usize,
}

impl WordDistanceMatrix {
    /// Create a zero-initialized matrix for `n` words.
    pub fn new(n: usize) -> Self {
        let entries = n * (n.saturating_sub(1)) / 2;
        WordDistanceMatrix {
            data: vec![0u8; entries],
            size: n,
        }
    }

    /// Build from a set of f32 vectors, one per word.
    ///
    /// Computes all pairwise L2 distances, quantizes to u8 (0..255).
    pub fn build(vectors: &[Vec<f32>]) -> Self {
        let n = vectors.len().min(MAX_VOCAB);
        let mut mat = Self::new(n);

        if n == 0 {
            return mat;
        }

        // Find max distance for normalization
        let mut max_dist: f32 = 0.0;
        for i in 0..n {
            for j in (i + 1)..n {
                let d = l2_distance(&vectors[i], &vectors[j]);
                if d > max_dist {
                    max_dist = d;
                }
            }
        }

        if max_dist < f32::EPSILON {
            return mat;
        }

        // Quantize and store
        for i in 0..n {
            for j in (i + 1)..n {
                let d = l2_distance(&vectors[i], &vectors[j]);
                let q = ((d / max_dist) * 255.0).round().min(255.0) as u8;
                mat.set(i, j, q);
            }
        }

        mat
    }

    /// Get quantized distance between words at ranks `a` and `b`. O(1).
    ///
    /// Returns 0 if a == b, and is symmetric: get(a,b) == get(b,a).
    pub fn get(&self, a: usize, b: usize) -> u32 {
        if a == b {
            return 0;
        }
        let (lo, hi) = if a < b { (a, b) } else { (b, a) };
        let idx = lo * (2 * self.size - lo - 1) / 2 + (hi - lo - 1);
        if idx < self.data.len() {
            self.data[idx] as u32
        } else {
            255
        }
    }

    /// Set distance for pair (a, b). a must be < b.
    fn set(&mut self, a: usize, b: usize, val: u8) {
        let (lo, hi) = if a < b { (a, b) } else { (b, a) };
        let idx = lo * (2 * self.size - lo - 1) / 2 + (hi - lo - 1);
        if idx < self.data.len() {
            self.data[idx] = val;
        }
    }

    /// Number of words in the matrix.
    pub fn size(&self) -> usize {
        self.size
    }
}

/// L2 distance between two f32 slices.
fn l2_distance(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len().min(b.len());
    let mut sum = 0.0f32;
    for i in 0..n {
        let d = a[i] - b[i];
        sum += d * d;
    }
    sum.sqrt()
}

/// Role vectors for XOR binding (6 roles x 1250 bytes).
///
/// Generated deterministically from a seed using XorShift PRNG.
#[derive(Clone, Debug)]
pub struct RoleVectors {
    /// Role vectors: [role_index][byte_index].
    pub vectors: [[u8; ROLE_VECTOR_BYTES]; NUM_ROLES],
}

impl RoleVectors {
    /// Generate role vectors from a 64-bit seed using XorShift64.
    pub fn from_seed(seed: u64) -> Self {
        let mut state = if seed == 0 { 0xDEAD_BEEF_CAFE_BABE } else { seed };
        let mut vectors = [[0u8; ROLE_VECTOR_BYTES]; NUM_ROLES];

        for role in 0..NUM_ROLES {
            for byte in 0..ROLE_VECTOR_BYTES {
                state = xorshift64(state);
                vectors[role][byte] = (state & 0xFF) as u8;
            }
        }

        RoleVectors { vectors }
    }

    /// XOR-bind a rank into a role vector position.
    ///
    /// Embeds the 12-bit rank at a position determined by the role.
    pub fn bind(&self, role: usize, rank: u16) -> [u8; ROLE_VECTOR_BYTES] {
        let mut result = self.vectors[role.min(NUM_ROLES - 1)];
        // XOR the rank bytes into the first two bytes
        let rank_bytes = rank.to_le_bytes();
        result[0] ^= rank_bytes[0];
        result[1] ^= rank_bytes[1];
        result
    }
}

/// XorShift64 PRNG step.
fn xorshift64(mut state: u64) -> u64 {
    state ^= state << 13;
    state ^= state >> 7;
    state ^= state << 17;
    state
}

/// An encoded triple ready for distance comparison.
#[derive(Debug, Clone, PartialEq)]
pub struct EncodedTriple {
    /// Subject rank (12-bit).
    pub subject: u16,
    /// Predicate rank (12-bit).
    pub predicate: u16,
    /// Object rank (12-bit).
    pub object: u16,
    /// Whether the predicate is negated.
    pub negated: bool,
}

/// An encoded sentence: a collection of encoded triples.
#[derive(Debug, Clone)]
pub struct EncodedSentence {
    /// The encoded triples.
    pub triples: Vec<EncodedTriple>,
}

/// Result of comparing two triples.
#[derive(Debug, Clone)]
pub struct TripleSimilarity {
    /// Total distance (sum of S, P, O distances).
    pub distance: u32,
    /// Calibrated similarity score in [0.0, 1.0].
    pub similarity: f32,
    /// Subject distance component.
    pub subject_distance: u32,
    /// Predicate distance component.
    pub predicate_distance: u32,
    /// Object distance component.
    pub object_distance: u32,
}

/// The NSM encoder: bridges word distances to triple/sentence comparison.
#[derive(Clone, Debug)]
pub struct NsmEncoder {
    /// Word-to-word distance matrix.
    pub matrix: WordDistanceMatrix,
    /// Role vectors for XOR binding.
    pub roles: RoleVectors,
    /// Ranks of the 63 NSM semantic primes (subset of vocabulary).
    pub prime_ranks: Vec<u16>,
}

impl NsmEncoder {
    /// Create an encoder with a distance matrix, roles, and prime ranks.
    pub fn new(
        matrix: WordDistanceMatrix,
        roles: RoleVectors,
        prime_ranks: Vec<u16>,
    ) -> Self {
        NsmEncoder {
            matrix,
            roles,
            prime_ranks,
        }
    }

    /// Distance between two triples: sum of S, P, O word distances.
    ///
    /// O(1): three matrix lookups.
    pub fn triple_distance(&self, a: &SpoTriple, b: &SpoTriple) -> u32 {
        let sd = self.matrix.get(a.subject() as usize, b.subject() as usize);
        let pd = self.matrix.get(a.predicate() as usize, b.predicate() as usize);
        let od = self.matrix.get(a.object() as usize, b.object() as usize);
        sd + pd + od
    }

    /// Calibrated similarity between two triples.
    ///
    /// Maps total distance to [0.0, 1.0] using linear scaling.
    /// Max possible distance = 255 * 3 = 765.
    pub fn triple_similarity(&self, a: &SpoTriple, b: &SpoTriple) -> f32 {
        let dist = self.triple_distance(a, b);
        let max_dist = 255u32 * 3;
        1.0 - (dist as f32 / max_dist as f32)
    }

    /// Detailed similarity breakdown between two triples.
    pub fn triple_similarity_detail(
        &self,
        a: &SpoTriple,
        b: &SpoTriple,
    ) -> TripleSimilarity {
        let sd = self.matrix.get(a.subject() as usize, b.subject() as usize);
        let pd = self.matrix.get(a.predicate() as usize, b.predicate() as usize);
        let od = self.matrix.get(a.object() as usize, b.object() as usize);
        let dist = sd + pd + od;
        let max_dist = 255u32 * 3;
        TripleSimilarity {
            distance: dist,
            similarity: 1.0 - (dist as f32 / max_dist as f32),
            subject_distance: sd,
            predicate_distance: pd,
            object_distance: od,
        }
    }

    /// Find the nearest NSM prime to a given word rank.
    ///
    /// Scans all 63 primes in the distance matrix. O(63) = O(1).
    pub fn nearest_prime(&self, rank: u16) -> (u16, f32) {
        let mut best_rank: u16 = 0;
        let mut best_dist: u32 = u32::MAX;

        for &prime_rank in &self.prime_ranks {
            let d = self.matrix.get(rank as usize, prime_rank as usize);
            if d < best_dist {
                best_dist = d;
                best_rank = prime_rank;
            }
        }

        let similarity = 1.0 - (best_dist as f32 / 255.0);
        (best_rank, similarity)
    }

    /// Decompose a word into all NSM primes, sorted by distance (nearest first).
    pub fn decompose(&self, rank: u16) -> Vec<(u16, f32)> {
        let mut pairs: Vec<(u16, f32)> = self
            .prime_ranks
            .iter()
            .map(|&pr| {
                let d = self.matrix.get(rank as usize, pr as usize);
                let sim = 1.0 - (d as f32 / 255.0);
                (pr, sim)
            })
            .collect();

        pairs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        pairs
    }

    /// Sentence-level similarity: average of best-matching triple pairs.
    pub fn sentence_similarity(
        &self,
        a: &EncodedSentence,
        b: &EncodedSentence,
    ) -> f32 {
        if a.triples.is_empty() || b.triples.is_empty() {
            return 0.0;
        }

        let mut total_sim = 0.0f32;
        let mut count = 0;

        for at in &a.triples {
            let at_spo = SpoTriple::new(at.subject, at.predicate, at.object);
            let mut best_sim = 0.0f32;
            for bt in &b.triples {
                let bt_spo = SpoTriple::new(bt.subject, bt.predicate, bt.object);
                let sim = self.triple_similarity(&at_spo, &bt_spo);
                // Penalize negation mismatch
                let sim = if at.negated != bt.negated {
                    sim * 0.5
                } else {
                    sim
                };
                if sim > best_sim {
                    best_sim = sim;
                }
            }
            total_sim += best_sim;
            count += 1;
        }

        total_sim / count as f32
    }
}

/// Build a small test encoder with synthetic distances.
pub fn test_encoder() -> NsmEncoder {
    // Create a small matrix for ranks 0..71 (test vocabulary range)
    let n = 71;
    let mut mat = WordDistanceMatrix::new(n);

    // Fill with synthetic distances based on rank difference
    // (simple heuristic: closer ranks = smaller distance)
    let mut state = 0x1234_5678_9ABC_DEF0u64;
    for i in 0..n {
        for j in (i + 1)..n {
            let rank_diff = (j - i) as f32;
            state = xorshift64(state);
            let noise = ((state & 0xFF) as f32) / 255.0 * 30.0;
            let base = (rank_diff * 3.0).min(200.0);
            let d = (base + noise).min(255.0) as u8;
            mat.set(i, j, d);
        }
    }

    // NSM primes are ranks 0..63 in our test vocabulary
    let prime_ranks: Vec<u16> = (0..63u16).collect();

    NsmEncoder::new(
        mat,
        RoleVectors::from_seed(42),
        prime_ranks,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_distance_matrix_self() {
        let mat = WordDistanceMatrix::new(100);
        assert_eq!(mat.get(0, 0), 0);
        assert_eq!(mat.get(50, 50), 0);
    }

    #[test]
    fn test_distance_matrix_symmetric() {
        let vecs: Vec<Vec<f32>> = (0..10)
            .map(|i| vec![i as f32 * 0.1, i as f32 * 0.2, i as f32 * 0.3])
            .collect();
        let mat = WordDistanceMatrix::build(&vecs);
        assert_eq!(mat.get(2, 5), mat.get(5, 2));
        assert_eq!(mat.get(0, 9), mat.get(9, 0));
    }

    #[test]
    fn test_distance_matrix_build() {
        let vecs: Vec<Vec<f32>> = vec![
            vec![0.0, 0.0],
            vec![1.0, 0.0],
            vec![0.0, 1.0],
        ];
        let mat = WordDistanceMatrix::build(&vecs);
        assert_eq!(mat.size(), 3);
        assert_eq!(mat.get(0, 0), 0);
        // Distance from (0,0) to (1,0) should equal distance from (0,0) to (0,1)
        assert_eq!(mat.get(0, 1), mat.get(0, 2));
    }

    #[test]
    fn test_triple_distance() {
        let enc = test_encoder();
        let a = SpoTriple::new(50, 67, 51); // cat chases dog
        let b = SpoTriple::new(50, 67, 51); // same
        assert_eq!(enc.triple_distance(&a, &b), 0);

        let c = SpoTriple::new(50, 60, 51); // cat runs dog
        assert!(enc.triple_distance(&a, &c) > 0);
    }

    #[test]
    fn test_triple_similarity() {
        let enc = test_encoder();
        let a = SpoTriple::new(50, 67, 51);
        let b = SpoTriple::new(50, 67, 51);
        let sim = enc.triple_similarity(&a, &b);
        assert!((sim - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_nearest_prime() {
        let enc = test_encoder();
        // Rank 50 (cat) should find some nearest prime
        let (prime_rank, sim) = enc.nearest_prime(50);
        assert!(prime_rank < 63);
        assert!(sim >= 0.0 && sim <= 1.0);
    }

    #[test]
    fn test_decompose() {
        let enc = test_encoder();
        let decomp = enc.decompose(50);
        assert_eq!(decomp.len(), 63);
        // Should be sorted by similarity descending
        for w in decomp.windows(2) {
            assert!(w[0].1 >= w[1].1);
        }
    }

    #[test]
    fn test_role_vectors_deterministic() {
        let r1 = RoleVectors::from_seed(42);
        let r2 = RoleVectors::from_seed(42);
        assert_eq!(r1.vectors, r2.vectors);
    }

    #[test]
    fn test_role_bind() {
        let roles = RoleVectors::from_seed(42);
        let bound = roles.bind(0, 50);
        // Should differ from the base role vector at least in first 2 bytes
        assert_ne!(bound[0], roles.vectors[0][0]);
    }

    #[test]
    fn test_sentence_similarity_identical() {
        let enc = test_encoder();
        let t = EncodedTriple {
            subject: 50,
            predicate: 67,
            object: 51,
            negated: false,
        };
        let s1 = EncodedSentence {
            triples: vec![t.clone()],
        };
        let s2 = EncodedSentence {
            triples: vec![t],
        };
        let sim = enc.sentence_similarity(&s1, &s2);
        assert!((sim - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_sentence_similarity_negation_penalty() {
        let enc = test_encoder();
        let t1 = EncodedTriple {
            subject: 50,
            predicate: 67,
            object: 51,
            negated: false,
        };
        let t2 = EncodedTriple {
            subject: 50,
            predicate: 67,
            object: 51,
            negated: true,
        };
        let s1 = EncodedSentence {
            triples: vec![t1],
        };
        let s2 = EncodedSentence {
            triples: vec![t2],
        };
        let sim = enc.sentence_similarity(&s1, &s2);
        // Should be penalized for negation mismatch: 1.0 * 0.5 = 0.5
        assert!((sim - 0.5).abs() < f32::EPSILON);
    }
}
