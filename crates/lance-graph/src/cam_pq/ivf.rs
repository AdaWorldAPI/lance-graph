//! IVF (Inverted File) coarse partitioning for billion-scale CAM-PQ search.
//!
//! Clusters vectors into N lists using coarse centroids.
//! Query probes P lists, then runs CAM-PQ cascade within each.
//!
//! For billion-scale: IVF coarse filter reduces search from 1B to ~10M candidates,
//! then CAM-PQ cascade reduces to ~100 finalists.
//!
//! # Layout
//!
//! ```text
//! IVF Index (on disk as Lance tables):
//!   coarse_centroids.lance  — N centroid vectors (one per partition)
//!   partition_{i}.lance     — CAM fingerprints in partition i
//!
//! Query flow:
//!   1. Compute distance to all N coarse centroids
//!   2. Pick top-P closest partitions
//!   3. Within each partition: CAM-PQ stroke cascade
//!   4. Merge top-K across all P partitions
//! ```

use std::cmp::Ordering;

/// IVF configuration.
#[derive(Debug, Clone)]
pub struct IvfConfig {
    /// Number of coarse partitions (IVF lists).
    pub num_partitions: usize,
    /// Number of partitions to probe per query.
    pub num_probes: usize,
    /// HEEL distance threshold for stroke 1.
    pub heel_threshold: f32,
    /// HEEL+BRANCH distance threshold for stroke 2.
    pub branch_threshold: f32,
    /// Top-K per partition before merge.
    pub top_k_per_partition: usize,
}

impl Default for IvfConfig {
    fn default() -> Self {
        Self {
            num_partitions: 1024,
            num_probes: 10,
            heel_threshold: 50.0,
            branch_threshold: 25.0,
            top_k_per_partition: 100,
        }
    }
}

/// A coarse centroid for one IVF partition.
#[derive(Debug, Clone)]
pub struct CoarseCentroid {
    /// Partition index.
    pub partition_id: u32,
    /// Full-precision centroid vector.
    pub vector: Vec<f32>,
}

/// IVF partition assignment for a single vector.
#[derive(Debug, Clone, Copy)]
pub struct IvfAssignment {
    /// Partition index.
    pub partition: u32,
    /// Distance to partition centroid.
    pub distance: f32,
}

/// IVF index: coarse centroids + partition metadata.
#[derive(Debug, Clone)]
pub struct IvfIndex {
    /// Configuration.
    pub config: IvfConfig,
    /// Coarse centroids (one per partition).
    pub centroids: Vec<CoarseCentroid>,
    /// Number of vectors in each partition.
    pub partition_sizes: Vec<usize>,
    /// Total vectors indexed.
    pub total_vectors: u64,
}

impl IvfIndex {
    /// Train IVF coarse centroids from a sample of vectors.
    ///
    /// Uses k-means clustering to find `num_partitions` centroids.
    pub fn train(vectors: &[Vec<f32>], config: IvfConfig) -> Self {
        let dim = vectors[0].len();
        let k = config.num_partitions.min(vectors.len());

        // K-means for coarse centroids
        let centroids_raw = kmeans(vectors, k, dim, 20);

        let centroids: Vec<CoarseCentroid> = centroids_raw
            .into_iter()
            .enumerate()
            .map(|(i, v)| CoarseCentroid {
                partition_id: i as u32,
                vector: v,
            })
            .collect();

        // Assign vectors to partitions to get sizes
        let mut partition_sizes = vec![0usize; k];
        for v in vectors {
            let assignment = Self::assign_to_partition_inner(v, &centroids);
            partition_sizes[assignment.partition as usize] += 1;
        }

        IvfIndex {
            config,
            centroids,
            partition_sizes,
            total_vectors: vectors.len() as u64,
        }
    }

    /// Assign a single vector to its nearest partition.
    pub fn assign(&self, vector: &[f32]) -> IvfAssignment {
        Self::assign_to_partition_inner(vector, &self.centroids)
    }

    /// Assign multiple vectors to partitions.
    pub fn assign_batch(&self, vectors: &[Vec<f32>]) -> Vec<IvfAssignment> {
        vectors.iter().map(|v| self.assign(v)).collect()
    }

    /// Find the top-P closest partitions for a query (probe list).
    pub fn probe(&self, query: &[f32]) -> Vec<IvfAssignment> {
        let mut distances: Vec<IvfAssignment> = self.centroids.iter()
            .map(|c| IvfAssignment {
                partition: c.partition_id,
                distance: squared_l2(query, &c.vector),
            })
            .collect();

        distances.sort_by(|a, b| a.distance.partial_cmp(&b.distance).unwrap_or(Ordering::Equal));
        distances.truncate(self.config.num_probes);
        distances
    }

    /// Expected candidates after IVF probe.
    pub fn expected_candidates(&self) -> u64 {
        if self.centroids.is_empty() {
            return 0;
        }
        let avg_size = self.total_vectors / self.centroids.len() as u64;
        self.config.num_probes as u64 * avg_size
    }

    fn assign_to_partition_inner(vector: &[f32], centroids: &[CoarseCentroid]) -> IvfAssignment {
        let mut best_id = 0u32;
        let mut best_dist = f32::MAX;
        for c in centroids {
            let d = squared_l2(vector, &c.vector);
            if d < best_dist {
                best_dist = d;
                best_id = c.partition_id;
            }
        }
        IvfAssignment { partition: best_id, distance: best_dist }
    }
}

/// IVF + CAM-PQ query result.
#[derive(Debug, Clone)]
pub struct IvfCamResult {
    /// Global vector ID.
    pub id: u64,
    /// Partition it came from.
    pub partition: u32,
    /// ADC distance (from CAM-PQ).
    pub distance: f32,
}

/// Merge top-K results from multiple partitions.
pub fn merge_partition_results(
    partition_results: &[Vec<IvfCamResult>],
    global_top_k: usize,
) -> Vec<IvfCamResult> {
    let mut merged: Vec<IvfCamResult> = partition_results.iter()
        .flat_map(|r| r.iter().cloned())
        .collect();

    merged.sort_by(|a, b| a.distance.partial_cmp(&b.distance).unwrap_or(Ordering::Equal));
    merged.truncate(global_top_k);
    merged
}

/// Squared L2 distance.
#[inline(always)]
fn squared_l2(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| (x - y) * (x - y)).sum()
}

/// Simple k-means clustering (same algorithm as ndarray cam_pq, standalone copy).
fn kmeans(data: &[Vec<f32>], k: usize, dim: usize, iterations: usize) -> Vec<Vec<f32>> {
    let n = data.len();
    if n == 0 || k == 0 {
        return vec![vec![0.0; dim]; k];
    }
    let k = k.min(n);

    // Farthest-first init
    let mut centroids: Vec<Vec<f32>> = Vec::with_capacity(k);
    centroids.push(data[0].clone());
    let mut min_dists = vec![f32::MAX; n];

    for _ in 1..k {
        let last = centroids.last().unwrap();
        for (i, v) in data.iter().enumerate() {
            let d = squared_l2(v, last);
            if d < min_dists[i] {
                min_dists[i] = d;
            }
        }
        let best = min_dists.iter().enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(Ordering::Equal))
            .map(|(i, _)| i)
            .unwrap_or(0);
        centroids.push(data[best].clone());
    }

    // Lloyd's iterations
    let mut assignments = vec![0usize; n];
    for _ in 0..iterations {
        for (i, v) in data.iter().enumerate() {
            let mut best_c = 0;
            let mut best_d = f32::MAX;
            for (c, centroid) in centroids.iter().enumerate() {
                let d = squared_l2(v, centroid);
                if d < best_d {
                    best_d = d;
                    best_c = c;
                }
            }
            assignments[i] = best_c;
        }

        let mut sums = vec![vec![0.0f32; dim]; k];
        let mut counts = vec![0usize; k];
        for (i, v) in data.iter().enumerate() {
            let c = assignments[i];
            counts[c] += 1;
            for (d, val) in v.iter().enumerate() {
                sums[c][d] += val;
            }
        }
        for c in 0..k {
            if counts[c] > 0 {
                for d in 0..dim {
                    centroids[c][d] = sums[c][d] / counts[c] as f32;
                }
            }
        }
    }

    centroids
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_clustered_vectors(clusters: usize, per_cluster: usize, dim: usize) -> Vec<Vec<f32>> {
        let mut vecs = Vec::new();
        for c in 0..clusters {
            let center = c as f32 * 100.0;
            for i in 0..per_cluster {
                let mut v = vec![center; dim];
                // Small perturbation
                v[0] += i as f32 * 0.01;
                vecs.push(v);
            }
        }
        vecs
    }

    #[test]
    fn test_ivf_train() {
        let vecs = make_clustered_vectors(8, 100, 24);
        let config = IvfConfig {
            num_partitions: 8,
            num_probes: 2,
            ..Default::default()
        };
        let index = IvfIndex::train(&vecs, config);

        assert_eq!(index.centroids.len(), 8);
        assert_eq!(index.total_vectors, 800);
        assert_eq!(index.partition_sizes.iter().sum::<usize>(), 800);
    }

    #[test]
    fn test_ivf_assign() {
        let vecs = make_clustered_vectors(4, 50, 12);
        let config = IvfConfig {
            num_partitions: 4,
            num_probes: 2,
            ..Default::default()
        };
        let index = IvfIndex::train(&vecs, config);

        // A vector near cluster 0 (center ~0.0) should go to a nearby partition
        let query = vec![0.5; 12];
        let assignment = index.assign(&query);
        assert!(assignment.distance < 100.0); // Should be close to a centroid
    }

    #[test]
    fn test_ivf_probe() {
        let vecs = make_clustered_vectors(8, 100, 24);
        let config = IvfConfig {
            num_partitions: 8,
            num_probes: 3,
            ..Default::default()
        };
        let index = IvfIndex::train(&vecs, config);

        let query = vec![50.0; 24]; // Near cluster with center=0 or 100
        let probes = index.probe(&query);
        assert_eq!(probes.len(), 3);

        // Probes should be sorted by distance
        for w in probes.windows(2) {
            assert!(w[0].distance <= w[1].distance);
        }
    }

    #[test]
    fn test_ivf_expected_candidates() {
        let vecs = make_clustered_vectors(10, 100, 12);
        let config = IvfConfig {
            num_partitions: 10,
            num_probes: 3,
            ..Default::default()
        };
        let index = IvfIndex::train(&vecs, config);

        let expected = index.expected_candidates();
        // 3 probes × (1000/10) = 300
        assert_eq!(expected, 300);
    }

    #[test]
    fn test_merge_partition_results() {
        let p1 = vec![
            IvfCamResult { id: 1, partition: 0, distance: 0.5 },
            IvfCamResult { id: 2, partition: 0, distance: 1.0 },
        ];
        let p2 = vec![
            IvfCamResult { id: 3, partition: 1, distance: 0.3 },
            IvfCamResult { id: 4, partition: 1, distance: 0.8 },
        ];

        let merged = merge_partition_results(&[p1, p2], 3);
        assert_eq!(merged.len(), 3);
        assert_eq!(merged[0].id, 3); // distance 0.3
        assert_eq!(merged[1].id, 1); // distance 0.5
        assert_eq!(merged[2].id, 4); // distance 0.8
    }

    #[test]
    fn test_ivf_billion_scale_estimates() {
        let config = IvfConfig {
            num_partitions: 4096,
            num_probes: 10,
            ..Default::default()
        };
        // Simulate billion-scale without actually training on 1B vectors
        let index = IvfIndex {
            config,
            centroids: (0..4096).map(|i| CoarseCentroid {
                partition_id: i as u32,
                vector: vec![0.0; 256],
            }).collect(),
            partition_sizes: vec![244_140; 4096], // ~1B / 4096
            total_vectors: 1_000_000_000,
        };

        let expected = index.expected_candidates();
        // 10 × (1B / 4096) ≈ 2.4M
        assert!(expected > 2_000_000);
        assert!(expected < 3_000_000);
    }
}
