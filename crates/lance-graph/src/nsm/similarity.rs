// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Calibrated similarity table for mapping palette-quantized distances to
//! meaningful similarity scores.
//!
//! Built from the exact distance distribution of a word distance matrix.
//! Provides O(1) lookup from u8 distance to f32 similarity.

use super::encoder::WordDistanceMatrix;

/// Number of entries in the similarity lookup table (one per u8 distance value).
const TABLE_SIZE: usize = 256;

/// Similarity table: maps u8 distance -> f32 similarity.
///
/// Built by computing the CDF of all pairwise distances, then inverting
/// so that similarity reflects percentile rank rather than raw distance.
#[derive(Clone, Debug)]
pub struct SimilarityTable {
    /// Lookup table: index = u8 distance, value = f32 similarity in [0.0, 1.0].
    table: [f32; TABLE_SIZE],
}

impl SimilarityTable {
    /// Build a similarity table from a distance matrix.
    ///
    /// Process:
    /// 1. Collect all pairwise distances from the upper triangle.
    /// 2. Sort to form the empirical CDF.
    /// 3. For each distance d, similarity = 1.0 - CDF(d).
    ///    (Distance 0 -> similarity 1.0, max distance -> similarity ~0.0)
    pub fn from_distance_matrix(matrix: &WordDistanceMatrix) -> Self {
        let n = matrix.size();
        let num_pairs = n * (n.saturating_sub(1)) / 2;

        if num_pairs == 0 {
            return Self::linear();
        }

        // Collect all pairwise distances
        let mut distances = Vec::with_capacity(num_pairs);
        for i in 0..n {
            for j in (i + 1)..n {
                distances.push(matrix.get(i, j) as u8);
            }
        }

        // Count histogram
        let mut histogram = [0u64; TABLE_SIZE];
        for &d in &distances {
            histogram[d as usize] += 1;
        }

        // Build CDF
        let total = distances.len() as f64;
        let mut cdf = [0.0f64; TABLE_SIZE];
        let mut cumulative = 0u64;
        for i in 0..TABLE_SIZE {
            cumulative += histogram[i];
            cdf[i] = cumulative as f64 / total;
        }

        // Invert: similarity = 1.0 - CDF(d)
        let mut table = [0.0f32; TABLE_SIZE];
        for i in 0..TABLE_SIZE {
            table[i] = (1.0 - cdf[i]) as f32;
        }
        // Distance 0 should always be 1.0
        table[0] = 1.0;

        SimilarityTable { table }
    }

    /// Create a simple linear similarity table (fallback when no matrix is available).
    ///
    /// similarity(d) = 1.0 - d/255.0
    pub fn linear() -> Self {
        let mut table = [0.0f32; TABLE_SIZE];
        for (i, slot) in table.iter_mut().enumerate() {
            *slot = 1.0 - (i as f32 / 255.0);
        }
        SimilarityTable { table }
    }

    /// Look up similarity for a u8 distance. O(1).
    pub fn similarity(&self, distance: u8) -> f32 {
        self.table[distance as usize]
    }

    /// Look up similarity for a u32 distance (clamped to 0..255).
    pub fn similarity_u32(&self, distance: u32) -> f32 {
        let d = distance.min(255) as usize;
        self.table[d]
    }

    /// Classify a similarity score into a named band.
    ///
    /// - "foveal": >= 0.85 (very close, near-synonym)
    /// - "near":   >= 0.65 (semantically related)
    /// - "good":   >= 0.40 (useful association)
    /// - "miss":   < 0.40  (too distant)
    pub fn band(&self, similarity: f32) -> &'static str {
        if similarity >= 0.85 {
            "foveal"
        } else if similarity >= 0.65 {
            "near"
        } else if similarity >= 0.40 {
            "good"
        } else {
            "miss"
        }
    }

    /// Convert a calibrated similarity threshold to approximate cosine equivalent.
    ///
    /// Uses the empirical mapping: cosine ~ 0.5 + 0.5 * calibrated_similarity.
    /// This is a rough approximation valid for typical word embedding spaces.
    pub fn cosine_equivalent(&self, threshold: f32) -> f32 {
        (0.5 + 0.5 * threshold).clamp(0.0, 1.0)
    }

    /// Return the raw table for inspection.
    pub fn raw_table(&self) -> &[f32; TABLE_SIZE] {
        &self.table
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::nsm::encoder::test_encoder;

    #[test]
    fn test_linear_table() {
        let table = SimilarityTable::linear();
        assert!((table.similarity(0) - 1.0).abs() < f32::EPSILON);
        assert!((table.similarity(255) - 0.0).abs() < 0.005);
        assert!((table.similarity(128) - 0.498).abs() < 0.01);
    }

    #[test]
    fn test_from_distance_matrix() {
        let enc = test_encoder();
        let table = SimilarityTable::from_distance_matrix(&enc.matrix);
        // Distance 0 should map to similarity 1.0
        assert!((table.similarity(0) - 1.0).abs() < f32::EPSILON);
        // Similarity should be monotonically non-increasing
        for i in 1..255 {
            assert!(table.similarity(i as u8) >= table.similarity((i + 1) as u8) - f32::EPSILON);
        }
    }

    #[test]
    fn test_band_classification() {
        let table = SimilarityTable::linear();
        assert_eq!(table.band(0.90), "foveal");
        assert_eq!(table.band(0.85), "foveal");
        assert_eq!(table.band(0.70), "near");
        assert_eq!(table.band(0.50), "good");
        assert_eq!(table.band(0.20), "miss");
        assert_eq!(table.band(0.0), "miss");
    }

    #[test]
    fn test_cosine_equivalent() {
        let table = SimilarityTable::linear();
        // threshold=1.0 -> cosine=1.0
        assert!((table.cosine_equivalent(1.0) - 1.0).abs() < f32::EPSILON);
        // threshold=0.0 -> cosine=0.5
        assert!((table.cosine_equivalent(0.0) - 0.5).abs() < f32::EPSILON);
        // threshold=0.5 -> cosine=0.75
        assert!((table.cosine_equivalent(0.5) - 0.75).abs() < f32::EPSILON);
    }

    #[test]
    fn test_similarity_u32_clamp() {
        let table = SimilarityTable::linear();
        // Values > 255 should clamp
        assert!((table.similarity_u32(300) - table.similarity_u32(255)).abs() < f32::EPSILON);
    }

    #[test]
    fn test_empty_matrix_fallback() {
        let mat = WordDistanceMatrix::new(0);
        let table = SimilarityTable::from_distance_matrix(&mat);
        // Should fall back to linear
        assert!((table.similarity(0) - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_raw_table_access() {
        let table = SimilarityTable::linear();
        let raw = table.raw_table();
        assert_eq!(raw.len(), 256);
        assert!((raw[0] - 1.0).abs() < f32::EPSILON);
    }
}
