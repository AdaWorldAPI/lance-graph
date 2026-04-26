//! Distribution-free similarity score via inverted empirical CDF.
//!
//! Maps raw palette/Hamming distances to f32 similarity in [0.0, 1.0],
//! calibrated from corpus statistics. Drop-in replacement for cosine similarity.

/// 256-entry lookup table mapping distance → similarity.
///
/// Built from corpus statistics (mean μ, standard deviation σ).
/// Resolution: 256 buckets across [0, 2×μ].
/// Distances beyond 2×μ map to 0.0 (noise).
/// Distances at 0 map to 1.0 (identical).
pub struct SimilarityTable {
    /// 256 similarity values, indexed by distance bucket.
    table: [f32; 256],
    /// Distance range per bucket.
    bucket_width: u32,
    /// Maximum mapped distance.
    max_distance: u32,
}

impl SimilarityTable {
    /// Build from mean and standard deviation (parametric sigmoid).
    pub fn from_stats(mu: u32, sigma: u32) -> Self {
        let max_distance = 2 * mu;
        let bucket_width = (max_distance / 256).max(1);
        let mut table = [0.0f32; 256];
        let sigma_f = (sigma.max(1)) as f32;
        let mu_f = mu as f32;
        for (i, entry) in table.iter_mut().enumerate() {
            let distance = (i as u32 * bucket_width) + bucket_width / 2;
            let z = (mu_f - distance as f32) / sigma_f;
            *entry = 1.0 / (1.0 + (-z).exp());
        }
        Self { table, bucket_width, max_distance }
    }

    /// Build from empirical CDF (reservoir samples).
    pub fn from_reservoir(samples: &mut [u32]) -> Self {
        if samples.is_empty() {
            return Self::from_stats(1000, 100);
        }
        samples.sort_unstable();
        let n = samples.len();
        let mu = samples[n / 2]; // median as mu
        let mean = samples.iter().map(|&s| s as f64).sum::<f64>() / n as f64;
        let var = samples.iter().map(|&s| { let d = s as f64 - mean; d * d }).sum::<f64>() / n as f64;
        let _sigma = var.sqrt() as u32;

        let max_distance = 2 * mu;
        let bucket_width = (max_distance / 256).max(1);
        let mut table = [0.0f32; 256];

        for (i, entry) in table.iter_mut().enumerate() {
            let bucket_center = (i as u32 * bucket_width) + bucket_width / 2;
            // CDF: fraction of samples <= bucket_center
            let count = samples.partition_point(|&s| s <= bucket_center);
            let cdf = count as f32 / n as f32;
            *entry = 1.0 - cdf; // similarity = 1 - CDF
        }

        Self { table, bucket_width, max_distance }
    }

    /// Lookup similarity for a raw distance. O(1).
    #[inline(always)]
    pub fn similarity(&self, distance: u32) -> f32 {
        if distance >= self.max_distance { return 0.0; }
        let bucket = (distance / self.bucket_width).min(255) as usize;
        self.table[bucket]
    }

    /// Batch similarity lookup.
    pub fn similarity_batch(&self, distances: &[u32], out: &mut [f32]) {
        assert_eq!(distances.len(), out.len());
        for (i, &d) in distances.iter().enumerate() {
            out[i] = self.similarity(d);
        }
    }

    pub fn bucket_width(&self) -> u32 { self.bucket_width }
    pub fn max_distance(&self) -> u32 { self.max_distance }
    pub fn table(&self) -> &[f32; 256] { &self.table }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_from_stats_basic() {
        let table = SimilarityTable::from_stats(1000, 200);
        // Distance 0 → high similarity
        assert!(table.similarity(0) > 0.9);
        // Distance at mu → ~0.5
        assert!((table.similarity(1000) - 0.5).abs() < 0.1);
        // Distance 2*mu → 0.0
        assert_eq!(table.similarity(2000), 0.0);
    }

    #[test]
    fn test_monotonicity() {
        let table = SimilarityTable::from_stats(1000, 200);
        let mut prev = 1.1f32;
        for d in (0..2000).step_by(10) {
            let s = table.similarity(d);
            assert!(s <= prev + 0.01, "not monotone at d={}: {} > {}", d, s, prev);
            prev = s;
        }
    }

    #[test]
    fn test_self_similarity() {
        let table = SimilarityTable::from_stats(1000, 200);
        assert!(table.similarity(0) > 0.95);
    }

    #[test]
    fn test_from_reservoir() {
        let mut samples: Vec<u32> = (0..1000).map(|i| (i * 2 + 100) as u32).collect();
        let table = SimilarityTable::from_reservoir(&mut samples);
        assert!(table.similarity(0) > 0.9);
        assert!(table.similarity(table.max_distance()) < 0.1);
    }

    #[test]
    fn test_batch() {
        let table = SimilarityTable::from_stats(1000, 200);
        let distances = vec![0, 500, 1000, 1500, 2000];
        let mut out = vec![0.0; 5];
        table.similarity_batch(&distances, &mut out);
        for (i, &d) in distances.iter().enumerate() {
            assert_eq!(out[i], table.similarity(d));
        }
    }
}
