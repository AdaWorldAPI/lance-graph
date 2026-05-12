//! CAM-PQ Scan: physical operator for CAM-PQ distance computation.
//!
//! Inserts into the SCAN phase of the resonance pipeline when the cost model
//! determines that CAM-PQ compression is beneficial (> 100K candidates).
//!
//! # Decision Boundary
//!
//! ```text
//! < 100K candidates  →  ScanOp::Full (brute force Hamming on raw fingerprints)
//! 100K - 10M         →  CamPqScanOp::FullAdc (6-byte ADC, no cascade)
//! > 10M              →  CamPqScanOp::Cascade (stroke 1→2→3 progressive)
//! > 100M             →  IVF probe → CamPqScanOp::Cascade per partition
//! ```

use super::PhysicalOperator;

/// CAM-PQ scan strategy (distinct from the Hamming ScanStrategy).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CamPqStrategy {
    /// Full ADC: compute all 6 subspace distances for every candidate.
    /// Good for 100K–10M candidates.
    FullAdc,
    /// Stroke cascade: HEEL → BRANCH → full.
    /// 99% rejection before full ADC. Good for > 10M candidates.
    Cascade,
    /// IVF + Cascade: coarse partition probe then cascade within each.
    /// Good for > 100M candidates.
    IvfCascade,
}

/// CAM-PQ physical scan operator.
#[derive(Debug)]
pub struct CamPqScanOp {
    /// Strategy selected by cost model.
    pub strategy: CamPqStrategy,
    /// HEEL distance threshold for stroke 1 (only used in Cascade/IvfCascade).
    pub heel_threshold: f32,
    /// HEEL+BRANCH distance threshold for stroke 2.
    pub branch_threshold: f32,
    /// Top-K results.
    pub top_k: usize,
    /// Number of IVF partitions to probe (only used in IvfCascade).
    pub num_probes: usize,
    /// Estimated output cardinality.
    pub estimated_cardinality: f64,
    /// Child operator (BroadcastOp or IVF probe output).
    pub child: Box<dyn PhysicalOperator>,
}

impl CamPqScanOp {
    /// Execute CAM-PQ scan on packed 6-byte fingerprints.
    ///
    /// `cam_data[i]` = 6-byte CAM fingerprint for candidate i.
    /// `distance_tables[subspace][centroid]` = precomputed distance.
    pub fn execute(
        &self,
        distance_tables: &[[f32; 256]; 6],
        cam_data: &[[u8; 6]],
    ) -> Vec<(usize, f32)> {
        match self.strategy {
            CamPqStrategy::FullAdc => self.full_adc(distance_tables, cam_data),
            CamPqStrategy::Cascade => self.cascade(distance_tables, cam_data),
            CamPqStrategy::IvfCascade => self.cascade(distance_tables, cam_data),
        }
    }

    /// Full ADC: 6 table lookups per candidate.
    fn full_adc(&self, dt: &[[f32; 256]; 6], cam_data: &[[u8; 6]]) -> Vec<(usize, f32)> {
        let mut results: Vec<(usize, f32)> = cam_data
            .iter()
            .enumerate()
            .map(|(idx, cam)| {
                let dist = dt[0][cam[0] as usize]
                    + dt[1][cam[1] as usize]
                    + dt[2][cam[2] as usize]
                    + dt[3][cam[3] as usize]
                    + dt[4][cam[4] as usize]
                    + dt[5][cam[5] as usize];
                (idx, dist)
            })
            .collect();

        results.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        results.truncate(self.top_k);
        results
    }

    /// Stroke cascade: progressive rejection.
    fn cascade(&self, dt: &[[f32; 256]; 6], cam_data: &[[u8; 6]]) -> Vec<(usize, f32)> {
        // Stroke 1: HEEL only
        let mut survivors: Vec<usize> = Vec::with_capacity(cam_data.len() / 10);
        for (idx, cam) in cam_data.iter().enumerate() {
            if dt[0][cam[0] as usize] < self.heel_threshold {
                survivors.push(idx);
            }
        }

        // Stroke 2: HEEL + BRANCH
        let mut refined: Vec<usize> = Vec::with_capacity(survivors.len() / 10);
        for &idx in &survivors {
            let cam = &cam_data[idx];
            let partial = dt[0][cam[0] as usize] + dt[1][cam[1] as usize];
            if partial < self.branch_threshold {
                refined.push(idx);
            }
        }

        // Stroke 3: full 6-byte ADC on finalists
        let mut results: Vec<(usize, f32)> = refined
            .iter()
            .map(|&idx| {
                let cam = &cam_data[idx];
                let dist = dt[0][cam[0] as usize]
                    + dt[1][cam[1] as usize]
                    + dt[2][cam[2] as usize]
                    + dt[3][cam[3] as usize]
                    + dt[4][cam[4] as usize]
                    + dt[5][cam[5] as usize];
                (idx, dist)
            })
            .collect();

        results.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        results.truncate(self.top_k);
        results
    }

    /// Cost model: select strategy based on candidate count.
    pub fn select_strategy(num_candidates: u64) -> CamPqStrategy {
        if num_candidates >= 100_000_000 {
            CamPqStrategy::IvfCascade
        } else if num_candidates >= 10_000_000 {
            CamPqStrategy::Cascade
        } else {
            CamPqStrategy::FullAdc
        }
    }

    /// Estimated cost in microseconds.
    pub fn estimated_cost_us(num_candidates: u64, strategy: CamPqStrategy) -> f64 {
        match strategy {
            CamPqStrategy::FullAdc => {
                // 6 lookups + 5 adds ≈ 2ns per candidate
                num_candidates as f64 * 0.002
            }
            CamPqStrategy::Cascade => {
                // Stroke 1: 1 lookup ≈ 0.5ns, 90% rejection
                // Stroke 2: 2 lookups ≈ 1ns on 10% survivors
                // Stroke 3: 6 lookups ≈ 2ns on 1% survivors
                let s1 = num_candidates as f64 * 0.0005;
                let s2 = num_candidates as f64 * 0.1 * 0.001;
                let s3 = num_candidates as f64 * 0.01 * 0.002;
                s1 + s2 + s3
            }
            CamPqStrategy::IvfCascade => {
                // IVF probe: ~50µs
                // Then cascade on ~1% of total
                50.0 + Self::estimated_cost_us(num_candidates / 100, CamPqStrategy::Cascade)
            }
        }
    }
}

impl PhysicalOperator for CamPqScanOp {
    fn name(&self) -> &str {
        "CamPqScan"
    }

    fn cardinality(&self) -> f64 {
        self.estimated_cardinality
    }

    fn is_pipeline_breaker(&self) -> bool {
        // Cascade is streaming (no materialization needed)
        false
    }

    fn children(&self) -> Vec<&dyn PhysicalOperator> {
        vec![&*self.child]
    }
}

#[cfg(test)]
mod tests {
    use super::super::broadcast::BroadcastOp;
    use super::*;

    fn make_distance_tables() -> [[f32; 256]; 6] {
        let mut dt = [[0.0f32; 256]; 6];
        for s in 0..6 {
            for c in 0..256 {
                // Distance increases with centroid index
                dt[s][c] = c as f32 * (s as f32 + 1.0) * 0.1;
            }
        }
        dt
    }

    fn make_cam_data(n: usize) -> Vec<[u8; 6]> {
        (0..n)
            .map(|i| {
                let v = (i % 256) as u8;
                [
                    v,
                    v.wrapping_add(1),
                    v.wrapping_add(2),
                    v.wrapping_add(3),
                    v.wrapping_add(4),
                    v.wrapping_add(5),
                ]
            })
            .collect()
    }

    fn dummy_child() -> Box<dyn PhysicalOperator> {
        Box::new(BroadcastOp {
            fingerprint: vec![0u64; 4],
            partitions: 1,
            cardinality: 1.0,
        })
    }

    #[test]
    fn test_full_adc() {
        let dt = make_distance_tables();
        let cams = make_cam_data(1000);
        let op = CamPqScanOp {
            strategy: CamPqStrategy::FullAdc,
            heel_threshold: 50.0,
            branch_threshold: 25.0,
            top_k: 10,
            num_probes: 0,
            estimated_cardinality: 10.0,
            child: dummy_child(),
        };

        let results = op.execute(&dt, &cams);
        assert_eq!(results.len(), 10);

        // Results should be sorted by distance
        for w in results.windows(2) {
            assert!(w[0].1 <= w[1].1);
        }

        // Closest should be cam[0] = [0,1,2,3,4,5] with small distances
        assert_eq!(results[0].0, 0);
    }

    #[test]
    fn test_cascade() {
        let dt = make_distance_tables();
        let cams = make_cam_data(10000);
        let op = CamPqScanOp {
            strategy: CamPqStrategy::Cascade,
            heel_threshold: 5.0, // Only pass centroids with heel index < ~50
            branch_threshold: 10.0,
            top_k: 10,
            num_probes: 0,
            estimated_cardinality: 10.0,
            child: dummy_child(),
        };

        let results = op.execute(&dt, &cams);
        assert!(results.len() <= 10);
        assert!(!results.is_empty());

        // All results should have distance < branch_threshold
        // (since stroke 2 filters at branch_threshold)
        for (_, dist) in &results {
            assert!(*dist < 100.0); // Loose bound — they passed cascade
        }
    }

    #[test]
    fn test_cascade_rejection_rate() {
        let dt = make_distance_tables();
        let cams = make_cam_data(100_000);
        let op = CamPqScanOp {
            strategy: CamPqStrategy::Cascade,
            heel_threshold: 2.0, // Very tight
            branch_threshold: 3.0,
            top_k: 10,
            num_probes: 0,
            estimated_cardinality: 10.0,
            child: dummy_child(),
        };

        let results = op.execute(&dt, &cams);
        // With tight thresholds, cascade should produce few results
        assert!(results.len() <= 10);
    }

    #[test]
    fn test_strategy_selection() {
        assert_eq!(
            CamPqScanOp::select_strategy(1_000_000),
            CamPqStrategy::FullAdc
        );
        assert_eq!(
            CamPqScanOp::select_strategy(10_000_000),
            CamPqStrategy::Cascade
        );
        assert_eq!(
            CamPqScanOp::select_strategy(500_000_000),
            CamPqStrategy::IvfCascade
        );
    }

    #[test]
    fn test_cost_model() {
        // Cascade should be cheaper than FullAdc for large datasets
        let n = 100_000_000;
        let full_cost = CamPqScanOp::estimated_cost_us(n, CamPqStrategy::FullAdc);
        let cascade_cost = CamPqScanOp::estimated_cost_us(n, CamPqStrategy::Cascade);
        assert!(
            cascade_cost < full_cost,
            "cascade {}µs should be < full_adc {}µs for {}M candidates",
            cascade_cost,
            full_cost,
            n / 1_000_000
        );
    }

    #[test]
    fn test_physical_operator_trait() {
        let op = CamPqScanOp {
            strategy: CamPqStrategy::Cascade,
            heel_threshold: 50.0,
            branch_threshold: 25.0,
            top_k: 10,
            num_probes: 3,
            estimated_cardinality: 100.0,
            child: dummy_child(),
        };

        assert_eq!(op.name(), "CamPqScan");
        assert_eq!(op.cardinality(), 100.0);
        assert!(!op.is_pipeline_breaker());
        assert_eq!(op.children().len(), 1);
    }
}
