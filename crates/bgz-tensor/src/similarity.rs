//! SimilarityTable: 256-entry CDF lookup for converting L1 distance to similarity.
//!
//! Maps Base17 L1 distance → [0.0, 1.0] similarity score calibrated from
//! real embedding cosine similarity (e.g., Jina API ground truth).
//!
//! The table is built by collecting (L1_distance, cosine_similarity) pairs
//! from real data, then fitting a monotone decreasing mapping via CDF binning.
//!
//! 256 entries × 2 bytes (f16) = 512 bytes. Fits in a single cache line pair.

use crate::projection::Base17;

/// Number of CDF bins.
pub const N_BINS: usize = 256;

/// SimilarityTable: maps L1 distance quantile → similarity score.
///
/// Usage: `table.similarity(a.l1(&b))` → f32 in [0, 1].
#[derive(Clone, Debug)]
pub struct SimilarityTable {
    /// Maximum L1 distance observed during calibration.
    pub max_l1: u32,
    /// 256 similarity values (f32), indexed by quantized L1 distance.
    /// Index 0 = L1 distance 0 → similarity 1.0.
    /// Index 255 = max L1 → similarity ≈ 0.0.
    pub bins: [f32; N_BINS],
}

impl SimilarityTable {
    /// Look up similarity for an L1 distance.
    #[inline]
    pub fn similarity(&self, l1: u32) -> f32 {
        if self.max_l1 == 0 {
            return 1.0;
        }
        let idx = ((l1 as u64 * 255) / self.max_l1 as u64).min(255) as usize;
        self.bins[idx]
    }

    /// Build from paired observations: (L1 distance, ground-truth cosine similarity).
    ///
    /// Bins observations by L1 quantile, takes mean cosine per bin.
    /// Ensures monotone decreasing (higher L1 → lower similarity).
    pub fn calibrate(pairs: &[(u32, f64)]) -> Self {
        if pairs.is_empty() {
            return Self::linear_fallback(10000);
        }

        let mut sorted: Vec<(u32, f64)> = pairs.to_vec();
        sorted.sort_by_key(|&(l1, _)| l1);

        let max_l1 = sorted.last().map(|p| p.0).unwrap_or(1).max(1);
        let mut bins = [0.0f32; N_BINS];

        // Bin by L1 quantile
        let mut bin_sums = [0.0f64; N_BINS];
        let mut bin_counts = [0u32; N_BINS];

        for &(l1, cos) in &sorted {
            let idx = ((l1 as u64 * 255) / max_l1 as u64).min(255) as usize;
            bin_sums[idx] += cos;
            bin_counts[idx] += 1;
        }

        // Mean cosine per bin
        for (i, bin) in bins.iter_mut().enumerate().take(N_BINS) {
            if bin_counts[i] > 0 {
                *bin = (bin_sums[i] / bin_counts[i] as f64) as f32;
            } else {
                *bin = f32::NAN; // will be interpolated
            }
        }

        // Interpolate empty bins
        Self::interpolate_nans(&mut bins);

        // Enforce monotone decreasing
        Self::enforce_monotone(&mut bins);

        SimilarityTable { max_l1, bins }
    }

    /// Linear fallback when no real data is available.
    /// similarity = 1.0 - l1/max_l1.
    pub fn linear_fallback(max_l1: u32) -> Self {
        let mut bins = [0.0f32; N_BINS];
        for (i, bin) in bins.iter_mut().enumerate().take(N_BINS) {
            *bin = 1.0 - (i as f32 / 255.0);
        }
        SimilarityTable { max_l1, bins }
    }

    /// Fill NaN bins via linear interpolation from neighbors.
    fn interpolate_nans(bins: &mut [f32; N_BINS]) {
        // Forward pass: find first valid
        let mut last_valid = None;
        for i in 0..N_BINS {
            if !bins[i].is_nan() {
                if let Some(prev) = last_valid {
                    // Interpolate gap
                    let gap = i - prev;
                    if gap > 1 {
                        let start_val = bins[prev];
                        let end_val = bins[i];
                        for (offset, bin) in bins[(prev + 1)..i].iter_mut().enumerate() {
                            let t = (offset + 1) as f32 / gap as f32;
                            *bin = start_val + t * (end_val - start_val);
                        }
                    }
                }
                last_valid = Some(i);
            }
        }
        // Fill leading NaNs
        if let Some(first) = (0..N_BINS).find(|&i| !bins[i].is_nan()) {
            for i in 0..first {
                bins[i] = bins[first];
            }
        }
        // Fill trailing NaNs
        if let Some(last) = (0..N_BINS).rev().find(|&i| !bins[i].is_nan()) {
            for i in (last + 1)..N_BINS {
                bins[i] = bins[last];
            }
        }
        // If all NaN, default to linear
        if bins[0].is_nan() {
            for (i, bin) in bins.iter_mut().enumerate().take(N_BINS) {
                *bin = 1.0 - i as f32 / 255.0;
            }
        }
    }

    /// Ensure bins are monotone decreasing (higher index = higher L1 = lower similarity).
    fn enforce_monotone(bins: &mut [f32; N_BINS]) {
        for i in 1..N_BINS {
            if bins[i] > bins[i - 1] {
                bins[i] = bins[i - 1];
            }
        }
    }

    /// Spearman rank correlation between L1 distances and cosine similarities.
    ///
    /// Uses the same ranking algorithm as quality.rs but specialized for
    /// paired (L1, cosine) data. Expected: ρ > 0.85 for well-calibrated table.
    pub fn spearman_l1_vs_cosine(pairs: &[(u32, f64)]) -> f64 {
        if pairs.len() < 2 {
            return 0.0;
        }
        let n = pairs.len();

        // L1 ranks (ascending)
        let mut l1_indexed: Vec<(usize, u32)> = pairs
            .iter()
            .enumerate()
            .map(|(i, &(l1, _))| (i, l1))
            .collect();
        l1_indexed.sort_by_key(|&(_, l1)| l1);
        let mut l1_ranks = vec![0.0f64; n];
        assign_ranks(&l1_indexed, &mut l1_ranks);

        // Cosine ranks (descending — higher cosine = lower rank number)
        let mut cos_indexed: Vec<(usize, f64)> = pairs
            .iter()
            .enumerate()
            .map(|(i, &(_, c))| (i, -c))
            .collect();
        cos_indexed.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        let mut cos_ranks = vec![0.0f64; n];
        // Direct rank assignment for cosine (already sorted)
        for (i, &(_, c)) in cos_indexed.iter().enumerate() {
            let _ = c;
            cos_ranks[cos_indexed[i].0] = (i + 1) as f64;
        }

        crate::quality::pearson(&l1_ranks, &cos_ranks)
    }

    /// Byte size of the table.
    pub const fn byte_size() -> usize {
        N_BINS * 4 + 4 // bins (f32) + max_l1
    }
}

/// Assign average ranks to sorted (original_index, value) pairs.
fn assign_ranks(sorted: &[(usize, u32)], ranks: &mut [f64]) {
    let n = sorted.len();
    let mut i = 0;
    while i < n {
        let mut j = i + 1;
        while j < n && sorted[j].1 == sorted[i].1 {
            j += 1;
        }
        let avg_rank = (i + j) as f64 / 2.0 + 0.5;
        for k in i..j {
            ranks[sorted[k].0] = avg_rank;
        }
        i = j;
    }
}

/// Collect L1-vs-cosine calibration pairs from two sets of Base17 vectors
/// and their known cosine similarities.
pub fn collect_calibration_pairs(
    embeddings_a: &[Base17],
    embeddings_b: &[Base17],
    cosine_ground_truth: &[f64],
) -> Vec<(u32, f64)> {
    let n = embeddings_a
        .len()
        .min(embeddings_b.len())
        .min(cosine_ground_truth.len());
    (0..n)
        .map(|i| (embeddings_a[i].l1(&embeddings_b[i]), cosine_ground_truth[i]))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn linear_fallback_endpoints() {
        let table = SimilarityTable::linear_fallback(10000);
        assert!((table.similarity(0) - 1.0).abs() < 0.01);
        assert!(table.similarity(10000) < 0.01);
    }

    #[test]
    fn linear_fallback_monotone() {
        let table = SimilarityTable::linear_fallback(10000);
        for l1 in (0..10000).step_by(100) {
            let next = l1 + 100;
            assert!(
                table.similarity(l1) >= table.similarity(next),
                "similarity should decrease with L1: s({})={} > s({})={}",
                l1,
                table.similarity(l1),
                next,
                table.similarity(next)
            );
        }
    }

    #[test]
    fn calibrate_from_synthetic() {
        // Generate synthetic (L1, cosine) pairs: cosine ≈ 1 - l1/max
        let pairs: Vec<(u32, f64)> = (0..1000)
            .map(|i| {
                let l1 = i * 10;
                let cos = 1.0 - (l1 as f64 / 10000.0);
                (l1 as u32, cos)
            })
            .collect();

        let table = SimilarityTable::calibrate(&pairs);
        // Should be monotone
        for i in 1..N_BINS {
            assert!(
                table.bins[i] <= table.bins[i - 1],
                "bin {} ({}) > bin {} ({})",
                i,
                table.bins[i],
                i - 1,
                table.bins[i - 1]
            );
        }
        // Near-zero L1 should give high similarity
        assert!(table.similarity(0) > 0.9);
        // High L1 should give low similarity
        assert!(table.similarity(9000) < 0.2);
    }

    #[test]
    fn calibrate_nonlinear() {
        // Exponential decay: cosine drops fast then flattens
        let pairs: Vec<(u32, f64)> = (0..500)
            .map(|i| {
                let l1 = i * 20;
                let cos = (-l1 as f64 / 2000.0).exp();
                (l1 as u32, cos)
            })
            .collect();

        let table = SimilarityTable::calibrate(&pairs);
        // Should capture nonlinearity: mid-range has steeper drop
        let low = table.similarity(100);
        let mid = table.similarity(5000);
        let high = table.similarity(9000);
        assert!(low > mid);
        assert!(mid >= high);
    }

    #[test]
    fn spearman_perfect_inverse() {
        let pairs: Vec<(u32, f64)> = (0..100)
            .map(|i| (i as u32, 1.0 - i as f64 / 100.0))
            .collect();
        let rho = SimilarityTable::spearman_l1_vs_cosine(&pairs);
        assert!(
            rho > 0.99,
            "perfect inverse should give ρ ≈ 1.0, got {}",
            rho
        );
    }

    #[test]
    fn collect_pairs_basic() {
        let a = vec![Base17 { dims: [100; 17] }, Base17 { dims: [200; 17] }];
        let b = vec![Base17 { dims: [110; 17] }, Base17 { dims: [-200; 17] }];
        let cosines = vec![0.95, -0.90];
        let pairs = collect_calibration_pairs(&a, &b, &cosines);
        assert_eq!(pairs.len(), 2);
        assert!(pairs[0].0 < pairs[1].0, "first pair should have smaller L1");
    }
}
