// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Cost estimation for Hamming-predicate queries using cascade band statistics.
//!
//! When the DataFusion planner encounters a `WHERE hamming(a, b) < threshold`
//! predicate, we use the cascade's sigma-band classification to estimate
//! selectivity and decide whether to use a cascade-accelerated scan or a
//! full sequential scan.
//!
//! # Band-Based Selectivity Estimates
//!
//! For 16384-bit binary vectors with binomial distribution (mu=8192, sigma=64):
//!
//! | Band    | Threshold Range       | Estimated Selectivity |
//! |---------|-----------------------|-----------------------|
//! | Foveal  | < mu - 3σ  (< 8000)  | 0.1%                  |
//! | Near    | < mu - 2σ  (< 8064)  | 1.0%                  |
//! | Good    | < mu - σ   (< 8128)  | 5.0%                  |
//! | Weak    | < mu       (< 8192)  | 20.0%                 |
//! | Reject  | >= mu                 | 100.0%                |

use crate::graph::blasgraph::hdr::Band;

/// Estimated selectivity for each cascade band.
///
/// These are conservative estimates based on the normal approximation
/// to the binomial distribution for 16384-bit random binary vectors.
pub fn band_selectivity(band: &Band) -> f64 {
    match band {
        Band::Foveal => 0.001,  // 0.1% — extremely selective
        Band::Near => 0.01,     // 1.0% — very selective
        Band::Good => 0.05,     // 5.0% — moderately selective
        Band::Weak => 0.20,     // 20.0% — weakly selective
        Band::Reject => 1.00,   // 100.0% — no filtering
    }
}

/// Decide whether a cascade-accelerated scan is worthwhile.
///
/// Returns `true` if the band selectivity is below the threshold (default 5%),
/// meaning the cascade will eliminate enough candidates to justify the overhead
/// of the two-phase scan.
pub fn should_use_cascade(band: &Band) -> bool {
    band_selectivity(band) < CASCADE_SELECTIVITY_THRESHOLD
}

/// Selectivity threshold below which a cascade scan is preferred.
///
/// If the estimated selectivity is below this value, the cascade's overhead
/// (reading stroke columns, computing sampled distances) is paid back by
/// the reduction in full-distance computations.
const CASCADE_SELECTIVITY_THRESHOLD: f64 = 0.05;

/// Estimate the cost of a full sequential scan in arbitrary cost units.
///
/// `num_rows` is the total number of rows in the dataset.
/// Each row requires a full 16384-bit Hamming distance computation.
pub fn full_scan_cost(num_rows: usize) -> f64 {
    // Cost = number of rows × cost per full distance computation.
    // A 16384-bit Hamming distance is ~256 u64 popcount operations.
    num_rows as f64 * 256.0
}

/// Estimate the cost of a cascade-accelerated scan in arbitrary cost units.
///
/// The cascade first reads stroke columns (cheap), then computes full distances
/// only for survivors. The cost model is:
///
///   cost = num_rows × stroke_cost + survivors × full_distance_cost
///
/// where `survivors = num_rows × selectivity`.
pub fn cascade_scan_cost(num_rows: usize, band: &Band) -> f64 {
    let selectivity = band_selectivity(band);
    let stroke_cost = 8.0; // Reading and comparing a few sampled bytes per row
    let full_cost = 256.0; // Full 16384-bit Hamming distance

    let survivors = (num_rows as f64 * selectivity).ceil();

    num_rows as f64 * stroke_cost + survivors * full_cost
}

/// Choose the optimal scan strategy for a Hamming predicate.
///
/// Returns `ScanStrategy::Cascade` if the cascade scan is cheaper,
/// `ScanStrategy::FullScan` otherwise.
pub fn choose_scan_strategy(num_rows: usize, band: &Band) -> ScanStrategy {
    if should_use_cascade(band) && num_rows > MIN_ROWS_FOR_CASCADE {
        ScanStrategy::Cascade
    } else {
        ScanStrategy::FullScan
    }
}

/// Minimum number of rows for a cascade scan to be worthwhile.
/// Below this, the overhead of the cascade setup is not justified.
const MIN_ROWS_FOR_CASCADE: usize = 1000;

/// Scan strategy for Hamming-predicate queries.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScanStrategy {
    /// Full sequential scan computing Hamming distance for every row.
    FullScan,
    /// Cascade-accelerated scan: stroke-based pre-filtering + full distance
    /// only for survivors.
    Cascade,
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_band_selectivity_ordering() {
        // Selectivity should increase from Foveal to Reject.
        assert!(band_selectivity(&Band::Foveal) < band_selectivity(&Band::Near));
        assert!(band_selectivity(&Band::Near) < band_selectivity(&Band::Good));
        assert!(band_selectivity(&Band::Good) < band_selectivity(&Band::Weak));
        assert!(band_selectivity(&Band::Weak) < band_selectivity(&Band::Reject));
    }

    #[test]
    fn test_band_selectivity_bounds() {
        for band in &[Band::Foveal, Band::Near, Band::Good, Band::Weak, Band::Reject] {
            let sel = band_selectivity(band);
            assert!(sel >= 0.0 && sel <= 1.0, "selectivity out of range: {sel}");
        }
    }

    #[test]
    fn test_should_use_cascade_foveal() {
        assert!(should_use_cascade(&Band::Foveal));
    }

    #[test]
    fn test_should_use_cascade_near() {
        assert!(should_use_cascade(&Band::Near));
    }

    #[test]
    fn test_should_not_use_cascade_good() {
        // Good is at 5% — at the threshold boundary, >= threshold means false.
        assert!(!should_use_cascade(&Band::Good));
    }

    #[test]
    fn test_should_not_use_cascade_weak() {
        assert!(!should_use_cascade(&Band::Weak));
    }

    #[test]
    fn test_should_not_use_cascade_reject() {
        assert!(!should_use_cascade(&Band::Reject));
    }

    #[test]
    fn test_cascade_cheaper_than_full_for_foveal() {
        let rows = 1_000_000;
        let cascade = cascade_scan_cost(rows, &Band::Foveal);
        let full = full_scan_cost(rows);
        assert!(
            cascade < full,
            "cascade ({cascade}) should be cheaper than full ({full}) for Foveal"
        );
    }

    #[test]
    fn test_cascade_more_expensive_for_reject() {
        let rows = 1_000_000;
        let cascade = cascade_scan_cost(rows, &Band::Reject);
        let full = full_scan_cost(rows);
        // When selectivity is 100%, cascade adds stroke overhead on top of full scan.
        assert!(
            cascade > full,
            "cascade ({cascade}) should be more expensive than full ({full}) for Reject"
        );
    }

    #[test]
    fn test_choose_strategy_foveal_large() {
        assert_eq!(
            choose_scan_strategy(1_000_000, &Band::Foveal),
            ScanStrategy::Cascade
        );
    }

    #[test]
    fn test_choose_strategy_reject() {
        assert_eq!(
            choose_scan_strategy(1_000_000, &Band::Reject),
            ScanStrategy::FullScan
        );
    }

    #[test]
    fn test_choose_strategy_small_dataset() {
        // Even for Foveal, if the dataset is tiny, full scan is preferred.
        assert_eq!(
            choose_scan_strategy(100, &Band::Foveal),
            ScanStrategy::FullScan
        );
    }

    #[test]
    fn test_full_scan_cost_proportional() {
        let cost_1k = full_scan_cost(1000);
        let cost_2k = full_scan_cost(2000);
        assert!((cost_2k - 2.0 * cost_1k).abs() < 1e-6);
    }
}
