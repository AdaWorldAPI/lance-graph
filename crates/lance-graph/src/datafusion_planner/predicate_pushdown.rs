// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Predicate pushdown for Hamming distance queries.
//!
//! Detects `hamming(column, query) < threshold` predicates in DataFusion
//! logical plans and rewrites them into cascade-accelerated scans when
//! the estimated selectivity is low enough.
//!
//! # How It Works
//!
//! 1. **Detection**: Walk the logical plan looking for Filter nodes whose
//!    predicate is `hamming_distance(col, literal) < constant`.
//!
//! 2. **Classification**: Use `threshold_to_band` to classify the threshold
//!    into a sigma band (Foveal, Near, Good, Weak, Reject).
//!
//! 3. **Decision**: If `should_use_cascade(band)` is true and the table has
//!    enough rows, rewrite the scan to use cascade-accelerated filtering.
//!
//! 4. **Rewrite**: Replace the full-table-scan + Filter with a cascade scan
//!    that reads stroke columns first, filters, then reads full fingerprints
//!    only for survivors.

use super::cost_estimation::{choose_scan_strategy, ScanStrategy};
use crate::graph::blasgraph::cascade_ops::CascadeScanConfig;
use crate::graph::blasgraph::hdr::Band;

/// A detected Hamming predicate extracted from a WHERE clause.
#[derive(Debug, Clone)]
pub struct HammingPredicate {
    /// The fingerprint column name (e.g. "plane_s", "fingerprint").
    pub column: String,

    /// The Hamming distance threshold from the `< threshold` comparison.
    pub threshold: u32,

    /// The sigma band this threshold falls into.
    pub band: Band,
}

impl HammingPredicate {
    /// Create a new Hamming predicate with auto-classified band.
    pub fn new(column: String, threshold: u32) -> Self {
        let mu = 8192u32; // VECTOR_BITS / 2
        let sigma = 64u32;

        let band = if threshold < mu.saturating_sub(3 * sigma) {
            Band::Foveal
        } else if threshold < mu.saturating_sub(2 * sigma) {
            Band::Near
        } else if threshold < mu.saturating_sub(sigma) {
            Band::Good
        } else if threshold < mu {
            Band::Weak
        } else {
            Band::Reject
        };

        Self {
            column,
            threshold,
            band,
        }
    }
}

/// Result of analyzing a query for Hamming predicate pushdown opportunities.
#[derive(Debug, Clone)]
pub struct PushdownAnalysis {
    /// Detected Hamming predicates that could be pushed down.
    pub predicates: Vec<HammingPredicate>,

    /// Recommended scan strategy for each predicate.
    pub strategies: Vec<ScanStrategy>,
}

impl PushdownAnalysis {
    /// Analyze predicates against a table with the given row count.
    pub fn analyze(predicates: Vec<HammingPredicate>, num_rows: usize) -> Self {
        let strategies = predicates
            .iter()
            .map(|p| choose_scan_strategy(num_rows, &p.band))
            .collect();

        Self {
            predicates,
            strategies,
        }
    }

    /// Return cascade scan configs for predicates that should use cascade scans.
    pub fn cascade_configs(&self) -> Vec<CascadeScanConfig> {
        self.predicates
            .iter()
            .zip(self.strategies.iter())
            .filter(|(_, s)| **s == ScanStrategy::Cascade)
            .map(|(p, _)| CascadeScanConfig::new(p.threshold, p.column.clone()))
            .collect()
    }

    /// True if any predicate recommends a cascade scan.
    pub fn has_cascade_pushdown(&self) -> bool {
        self.strategies.contains(&ScanStrategy::Cascade)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hamming_predicate_foveal() {
        let pred = HammingPredicate::new("plane_s".to_string(), 7000);
        assert_eq!(pred.band, Band::Foveal);
        assert_eq!(pred.threshold, 7000);
        assert_eq!(pred.column, "plane_s");
    }

    #[test]
    fn test_hamming_predicate_near() {
        let pred = HammingPredicate::new("fp".to_string(), 8010);
        assert_eq!(pred.band, Band::Near);
    }

    #[test]
    fn test_hamming_predicate_good() {
        let pred = HammingPredicate::new("fp".to_string(), 8100);
        assert_eq!(pred.band, Band::Good);
    }

    #[test]
    fn test_hamming_predicate_weak() {
        let pred = HammingPredicate::new("fp".to_string(), 8150);
        assert_eq!(pred.band, Band::Weak);
    }

    #[test]
    fn test_hamming_predicate_reject() {
        let pred = HammingPredicate::new("fp".to_string(), 9000);
        assert_eq!(pred.band, Band::Reject);
    }

    #[test]
    fn test_pushdown_analysis_cascade() {
        let preds = vec![HammingPredicate::new("plane_s".to_string(), 7000)];
        let analysis = PushdownAnalysis::analyze(preds, 1_000_000);
        assert!(analysis.has_cascade_pushdown());
        assert_eq!(analysis.strategies[0], ScanStrategy::Cascade);
    }

    #[test]
    fn test_pushdown_analysis_full_scan() {
        let preds = vec![HammingPredicate::new("plane_s".to_string(), 9000)];
        let analysis = PushdownAnalysis::analyze(preds, 1_000_000);
        assert!(!analysis.has_cascade_pushdown());
        assert_eq!(analysis.strategies[0], ScanStrategy::FullScan);
    }

    #[test]
    fn test_pushdown_analysis_small_table() {
        let preds = vec![HammingPredicate::new("plane_s".to_string(), 7000)];
        let analysis = PushdownAnalysis::analyze(preds, 100);
        // Even Foveal on a tiny table → full scan
        assert!(!analysis.has_cascade_pushdown());
    }

    #[test]
    fn test_cascade_configs_only_cascade() {
        let preds = vec![
            HammingPredicate::new("plane_s".to_string(), 7000), // Foveal → cascade
            HammingPredicate::new("plane_o".to_string(), 9000), // Reject → full scan
        ];
        let analysis = PushdownAnalysis::analyze(preds, 1_000_000);
        let configs = analysis.cascade_configs();
        assert_eq!(configs.len(), 1);
        assert_eq!(configs[0].fingerprint_column, "plane_s");
    }

    #[test]
    fn test_pushdown_analysis_multiple_predicates() {
        let preds = vec![
            HammingPredicate::new("plane_s".to_string(), 7000),
            HammingPredicate::new("plane_p".to_string(), 8010),
        ];
        let analysis = PushdownAnalysis::analyze(preds, 1_000_000);
        // Both Foveal and Near should recommend cascade
        assert_eq!(analysis.strategies[0], ScanStrategy::Cascade);
        assert_eq!(analysis.strategies[1], ScanStrategy::Cascade);
        assert_eq!(analysis.cascade_configs().len(), 2);
    }
}
