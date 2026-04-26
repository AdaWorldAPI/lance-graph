//! SCAN: Vectorized Hamming distance computation.
//!
//! Three strategies (chosen by cost model):
//! - Cascade: stroke columns first, progressive refinement (lance-graph sigma-band)
//! - Full: brute force SIMD over all rows (ndarray kernels)
//! - Index: precomputed proximity index lookup
//!
//! The SIMD kernels live in ndarray; this operator orchestrates them.

#[allow(unused_imports)] // Morsel, ColumnData intended for scan execution wiring
use super::{PhysicalOperator, Morsel, ColumnData};
use crate::ir::logical_op::ScanStrategy;

/// SCAN physical operator.
#[derive(Debug)]
pub struct ScanOp {
    /// Which strategy to use.
    pub strategy: ScanStrategy,
    /// Hamming distance threshold.
    pub threshold: u32,
    /// Top-K results per partition.
    pub top_k: usize,
    /// Estimated output cardinality.
    pub estimated_cardinality: f64,
    /// Child operator (typically BroadcastOp).
    pub child: Box<dyn PhysicalOperator>,
}

impl ScanOp {
    /// Execute scan against a partition's data.
    ///
    /// In the real implementation, this calls ndarray SIMD kernels:
    /// - Cascade: scan stroke columns (high-order bits) first, then refine
    /// - Full: VPOPCNTDQ over all rows
    /// - Index: lookup in precomputed proximity index
    pub fn execute_partition(
        &self,
        query_fp: &[u64],
        data: &[Vec<u64>],
    ) -> Vec<(usize, u32)> {
        match self.strategy {
            ScanStrategy::Cascade => self.cascade_scan(query_fp, data),
            ScanStrategy::Full => self.full_scan(query_fp, data),
            ScanStrategy::Index => self.index_scan(query_fp, data),
            ScanStrategy::CamPq => {
                // CAM-PQ uses its own physical operator (CamPqScanOp).
                // If we reach here, fall back to full scan on raw fingerprints.
                self.full_scan(query_fp, data)
            }
        }
    }

    /// Cascade scan: progressive refinement on stroke columns.
    /// Scan high-order u64 words first. If partial distance already exceeds
    /// threshold, skip remaining words (sigma-band pruning).
    fn cascade_scan(&self, query_fp: &[u64], data: &[Vec<u64>]) -> Vec<(usize, u32)> {
        let mut results = Vec::new();

        for (idx, row_fp) in data.iter().enumerate() {
            let mut distance = 0u32;
            let mut pruned = false;

            // Scan words from most significant (stroke columns) to least
            for (q_word, r_word) in query_fp.iter().zip(row_fp.iter()) {
                distance += (q_word ^ r_word).count_ones();
                // Sigma-band: early exit if partial distance already too high
                if distance > self.threshold {
                    pruned = true;
                    break;
                }
            }

            if !pruned && distance <= self.threshold {
                results.push((idx, distance));
            }
        }

        // Sort by distance, keep top-K
        results.sort_by_key(|&(_, d)| d);
        results.truncate(self.top_k);
        results
    }

    /// Full scan: brute force Hamming distance over all rows.
    /// In production, this calls ndarray's AVX-512 VPOPCNTDQ kernel.
    fn full_scan(&self, query_fp: &[u64], data: &[Vec<u64>]) -> Vec<(usize, u32)> {
        let mut results: Vec<(usize, u32)> = data.iter().enumerate()
            .map(|(idx, row_fp)| {
                let distance: u32 = query_fp.iter()
                    .zip(row_fp.iter())
                    .map(|(q, r)| (q ^ r).count_ones())
                    .sum();
                (idx, distance)
            })
            .filter(|&(_, d)| d <= self.threshold)
            .collect();

        results.sort_by_key(|&(_, d)| d);
        results.truncate(self.top_k);
        results
    }

    /// Index scan: precomputed proximity index lookup.
    /// Placeholder — would use lance index or custom proximity structure.
    fn index_scan(&self, query_fp: &[u64], data: &[Vec<u64>]) -> Vec<(usize, u32)> {
        // Fall back to full scan for now
        self.full_scan(query_fp, data)
    }
}

impl PhysicalOperator for ScanOp {
    fn name(&self) -> &str { "Scan" }
    fn cardinality(&self) -> f64 { self.estimated_cardinality }
    fn is_pipeline_breaker(&self) -> bool { false }
    fn children(&self) -> Vec<&dyn PhysicalOperator> { vec![&*self.child] }
}
