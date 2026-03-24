//! IVF (Inverted File) coarse partitioning for billion-scale CAM-PQ search.
//!
//! Clusters CAM fingerprints into N lists using coarse centroids.
//! Query probes P lists, then runs CAM-PQ cascade within each.
//!
//! For billion-scale: IVF coarse filter reduces search from 1B to ~10M candidates,
//! then CAM-PQ cascade reduces to ~100 finalists.

/// IVF configuration.
#[derive(Debug, Clone)]
pub struct IvfConfig {
    /// Number of coarse partitions (IVF lists).
    pub num_partitions: usize,
    /// Number of partitions to probe per query.
    pub num_probes: usize,
    /// HEEL distance threshold for stroke 1 (within each partition).
    pub heel_threshold: f32,
    /// HEEL+BRANCH distance threshold for stroke 2.
    pub branch_threshold: f32,
}

impl Default for IvfConfig {
    fn default() -> Self {
        Self {
            num_partitions: 1024,
            num_probes: 10,
            heel_threshold: 50.0,
            branch_threshold: 25.0,
        }
    }
}

/// IVF partition assignment for a single vector.
#[derive(Debug, Clone, Copy)]
pub struct IvfAssignment {
    /// Partition index.
    pub partition: u32,
    /// Distance to partition centroid.
    pub distance: f32,
}

/// IVF index metadata stored alongside the Lance dataset.
///
/// The actual coarse centroids and partition assignments are stored
/// as Lance tables. This struct holds the configuration and statistics
/// for the index.
#[derive(Debug, Clone)]
pub struct IvfIndexMeta {
    pub config: IvfConfig,
    /// Total vectors indexed.
    pub num_vectors: u64,
    /// Average partition size.
    pub avg_partition_size: u64,
    /// Max partition size (for load balancing).
    pub max_partition_size: u64,
}

impl IvfIndexMeta {
    pub fn new(config: IvfConfig, num_vectors: u64) -> Self {
        let avg = num_vectors / config.num_partitions as u64;
        Self {
            config,
            num_vectors,
            avg_partition_size: avg,
            max_partition_size: avg * 3, // Estimate: 3× average for skewed distributions
        }
    }

    /// Expected candidates after IVF probe: num_probes × avg_partition_size.
    pub fn expected_candidates(&self) -> u64 {
        self.config.num_probes as u64 * self.avg_partition_size
    }

    /// Estimated query time (microseconds) based on expected candidates.
    pub fn estimated_query_us(&self) -> f64 {
        // Rough model: 2ns per ADC distance + 50µs overhead
        let candidates = self.expected_candidates() as f64;
        candidates * 0.002 + 50.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ivf_config_default() {
        let config = IvfConfig::default();
        assert_eq!(config.num_partitions, 1024);
        assert_eq!(config.num_probes, 10);
    }

    #[test]
    fn test_ivf_meta_billion_scale() {
        let config = IvfConfig {
            num_partitions: 4096,
            num_probes: 10,
            ..Default::default()
        };
        let meta = IvfIndexMeta::new(config, 1_000_000_000);

        // Expected candidates: 10 × (1B / 4096) ≈ 2.4M
        let candidates = meta.expected_candidates();
        assert!(candidates > 2_000_000);
        assert!(candidates < 3_000_000);

        // Should be fast: < 10ms
        let us = meta.estimated_query_us();
        assert!(us < 10_000.0, "query should be < 10ms, got {}µs", us);
    }
}
