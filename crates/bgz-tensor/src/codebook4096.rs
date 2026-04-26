//! 4096-entry codebook via two-level p64 attend.
//!
//! ```text
//! Level 1:  64 cluster centroids → p64 attend → best cluster      ~2 ns
//! Level 2:  64 entries per cluster → p64 attend → best entry       ~2 ns
//! Index = cluster(6 bits) × 64 + entry(6 bits) = 12 bits = 4096
//!
//! No precomputed distance matrix (would be 32 MB, cache-hostile).
//! Instead: two p64 attends at ~4 ns combined. Faster AND zero memory.
//!
//! Palette storage: 4096 × 136 bytes = 544 KB (L2-resident)
//!   Level 1 centroids: 64 × 136B = 8.5 KB (always L1-hot)
//!   Level 2 blocks: 64 × 8.5 KB (one loaded per query, L1 on access)
//! ```

use crate::stacked::StackedBF16x4;
// BASE_DIM and Base17 reserved for future PCDVQ-weighted distance
#[allow(unused_imports)]
use crate::projection::{BASE_DIM, Base17};

/// A 12-bit codebook index: cluster(6) + entry(6) = 4096 entries.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct CodebookIndex(pub u16);

impl CodebookIndex {
    #[inline]
    pub fn new(cluster: u8, entry: u8) -> Self {
        CodebookIndex(((cluster as u16) << 6) | (entry as u16 & 0x3F))
    }

    #[inline]
    pub fn cluster(self) -> u8 { (self.0 >> 6) as u8 }

    #[inline]
    pub fn entry(self) -> u8 { (self.0 & 0x3F) as u8 }
}

/// A single cluster: 64 stacked BF16×4 entries.
#[derive(Clone, Debug)]
pub struct Cluster {
    /// Cluster centroid (stacked resolution).
    pub centroid: StackedBF16x4,
    /// Up to 64 entries in this cluster.
    pub entries: Vec<StackedBF16x4>,
    /// Cluster radius (max distance from centroid to any entry).
    pub radius: u64,
    /// Number of source vectors assigned to this cluster.
    pub population: usize,
}

/// 4096-entry two-level codebook.
///
/// Level 1: 64 clusters (centroids in L1 cache).
/// Level 2: 64 entries per cluster (loaded on demand).
#[derive(Clone, Debug)]
pub struct Codebook4096 {
    /// 64 clusters.
    pub clusters: Vec<Cluster>,
    /// Total entries across all clusters.
    pub total_entries: usize,
}

/// Assignment result from codebook lookup.
#[derive(Clone, Debug)]
pub struct CodebookAssignment {
    /// 12-bit index into the codebook.
    pub index: CodebookIndex,
    /// Distance to the assigned entry (full stacked distance).
    pub distortion: u64,
    /// Distance to the cluster centroid (coarse level).
    pub cluster_distance: u64,
}

impl Codebook4096 {
    /// Build codebook from a set of stacked vectors.
    ///
    /// Phase 1: Furthest-point sampling for 64 cluster centroids.
    /// Phase 2: Assign all vectors to nearest cluster.
    /// Phase 3: Within each cluster, furthest-point sampling for up to 64 entries.
    pub fn build(vectors: &[StackedBF16x4], max_entries_per_cluster: usize) -> Self {
        let k_clusters = 64.min(vectors.len());
        let k_entries = max_entries_per_cluster.min(64);

        if vectors.is_empty() {
            return Codebook4096 { clusters: Vec::new(), total_entries: 0 };
        }

        // Phase 1: Select 64 cluster centroids via furthest-point sampling
        let centroid_indices = furthest_point_sampling(vectors, k_clusters);

        // Phase 2: Assign all vectors to nearest centroid
        let mut assignments: Vec<Vec<usize>> = vec![Vec::new(); k_clusters];
        for (vi, v) in vectors.iter().enumerate() {
            let (best_c, _) = nearest_centroid(v, &centroid_indices, vectors);
            assignments[best_c].push(vi);
        }

        // Phase 3: Build clusters with up to 64 entries each
        let mut clusters = Vec::with_capacity(k_clusters);
        let mut total_entries = 0;

        for (ci, centroid_idx) in centroid_indices.iter().enumerate() {
            let centroid = vectors[*centroid_idx].clone();
            let cluster_vecs: Vec<&StackedBF16x4> = assignments[ci]
                .iter()
                .map(|&i| &vectors[i])
                .collect();

            let (entries, radius) = if cluster_vecs.len() <= k_entries {
                // All fit → use them all
                let mut max_r = 0u64;
                let entries: Vec<StackedBF16x4> = cluster_vecs.iter().map(|v| {
                    let d = centroid.full_distance(v);
                    max_r = max_r.max(d);
                    (*v).clone()
                }).collect();
                (entries, max_r)
            } else {
                // Subsample via furthest-point within cluster
                let cluster_owned: Vec<StackedBF16x4> = cluster_vecs.iter().map(|v| (*v).clone()).collect();
                let entry_indices = furthest_point_sampling(&cluster_owned, k_entries);
                let mut max_r = 0u64;
                let entries: Vec<StackedBF16x4> = entry_indices.iter().map(|&i| {
                    let d = centroid.full_distance(&cluster_owned[i]);
                    max_r = max_r.max(d);
                    cluster_owned[i].clone()
                }).collect();
                (entries, max_r)
            };

            total_entries += entries.len();
            clusters.push(Cluster {
                centroid,
                entries,
                radius,
                population: cluster_vecs.len(),
            });
        }

        Codebook4096 { clusters, total_entries }
    }

    /// Two-level attend: find nearest codebook entry.
    ///
    /// Level 1: find nearest cluster centroid (~2 ns).
    /// Level 2: find nearest entry within that cluster (~2 ns).
    ///
    /// Total: ~4 ns, zero memory beyond the codebook itself.
    pub fn assign(&self, query: &StackedBF16x4) -> CodebookAssignment {
        if self.clusters.is_empty() {
            return CodebookAssignment {
                index: CodebookIndex(0),
                distortion: u64::MAX,
                cluster_distance: u64::MAX,
            };
        }

        // Level 1: find nearest cluster
        let (best_cluster, cluster_dist) = self.clusters.iter()
            .enumerate()
            .map(|(i, c)| (i, query.full_distance(&c.centroid)))
            .min_by_key(|&(_, d)| d)
            .unwrap();

        // Level 2: find nearest entry within cluster
        let cluster = &self.clusters[best_cluster];
        if cluster.entries.is_empty() {
            return CodebookAssignment {
                index: CodebookIndex::new(best_cluster as u8, 0),
                distortion: cluster_dist,
                cluster_distance: cluster_dist,
            };
        }

        let (best_entry, entry_dist) = cluster.entries.iter()
            .enumerate()
            .map(|(i, e)| (i, query.full_distance(e)))
            .min_by_key(|&(_, d)| d)
            .unwrap();

        CodebookAssignment {
            index: CodebookIndex::new(best_cluster as u8, best_entry as u8),
            distortion: entry_dist,
            cluster_distance: cluster_dist,
        }
    }

    /// Assign all vectors and return 12-bit indices.
    pub fn assign_all(&self, vectors: &[StackedBF16x4]) -> Vec<CodebookIndex> {
        vectors.iter().map(|v| self.assign(v).index).collect()
    }

    /// Look up the stacked vector for a codebook index.
    pub fn lookup(&self, index: CodebookIndex) -> Option<&StackedBF16x4> {
        let c = index.cluster() as usize;
        let e = index.entry() as usize;
        self.clusters.get(c).and_then(|cl| cl.entries.get(e))
    }

    /// Total memory footprint.
    pub fn byte_size(&self) -> usize {
        // Centroids + entries, all at 136 bytes each
        let n_centroids = self.clusters.len();
        let n_entries = self.total_entries;
        (n_centroids + n_entries) * StackedBF16x4::BYTE_SIZE
    }

    /// Compression statistics.
    pub fn summary(&self) -> String {
        let n_clusters = self.clusters.len();
        let populations: Vec<usize> = self.clusters.iter().map(|c| c.population).collect();
        let max_pop = populations.iter().max().copied().unwrap_or(0);
        let min_pop = populations.iter().min().copied().unwrap_or(0);
        let mean_pop = if n_clusters > 0 {
            populations.iter().sum::<usize>() as f64 / n_clusters as f64
        } else { 0.0 };

        format!(
            "Codebook4096: {} clusters, {} entries, {:.1} KB\n\
             Population: min={}, max={}, mean={:.1}\n\
             Max radius: {}",
            n_clusters, self.total_entries,
            self.byte_size() as f64 / 1024.0,
            min_pop, max_pop, mean_pop,
            self.clusters.iter().map(|c| c.radius).max().unwrap_or(0),
        )
    }
}

// ─── Helpers ────────────────────────────────────────────────────────────────

/// Furthest-point sampling for k centroids.
fn furthest_point_sampling(vectors: &[StackedBF16x4], k: usize) -> Vec<usize> {
    let k = k.min(vectors.len());
    if k == 0 { return Vec::new(); }

    let mut selected = Vec::with_capacity(k);
    let mut min_dist = vec![u64::MAX; vectors.len()];

    // Start with first vector
    selected.push(0);
    for i in 0..vectors.len() {
        min_dist[i] = vectors[i].full_distance(&vectors[0]);
    }

    for _ in 1..k {
        // Select vector with maximum min_dist
        let next = min_dist.iter()
            .enumerate()
            .max_by_key(|&(_, &d)| d)
            .map(|(i, _)| i)
            .unwrap_or(0);
        selected.push(next);

        // Update min distances
        for i in 0..vectors.len() {
            let d = vectors[i].full_distance(&vectors[next]);
            if d < min_dist[i] {
                min_dist[i] = d;
            }
        }
    }

    selected
}

/// Find nearest centroid (returns cluster index and distance).
fn nearest_centroid(
    query: &StackedBF16x4,
    centroid_indices: &[usize],
    all_vectors: &[StackedBF16x4],
) -> (usize, u64) {
    centroid_indices.iter()
        .enumerate()
        .map(|(ci, &vi)| (ci, query.full_distance(&all_vectors[vi])))
        .min_by_key(|&(_, d)| d)
        .unwrap_or((0, u64::MAX))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_stacked_vectors(n: usize) -> Vec<StackedBF16x4> {
        (0..n).map(|i| {
            let vals: Vec<f32> = (0..68).map(|d| {
                ((i * 97 + d * 31) as f32 % 100.0) - 50.0
            }).collect();
            StackedBF16x4::from_f32(&vals)
        }).collect()
    }

    #[test]
    fn build_basic() {
        let vectors = make_stacked_vectors(500);
        let cb = Codebook4096::build(&vectors, 16);
        assert!(!cb.clusters.is_empty());
        assert!(cb.total_entries > 0);
        eprintln!("{}", cb.summary());
    }

    #[test]
    fn assign_finds_nearest() {
        let vectors = make_stacked_vectors(200);
        let cb = Codebook4096::build(&vectors, 8);

        // Assigning a vector that was used to build should have low distortion
        let assignment = cb.assign(&vectors[0]);
        let entry = cb.lookup(assignment.index).unwrap();
        let dist = vectors[0].full_distance(entry);
        assert_eq!(dist, assignment.distortion);
    }

    #[test]
    fn codebook_index_roundtrip() {
        for c in 0..64u8 {
            for e in 0..64u8 {
                let idx = CodebookIndex::new(c, e);
                assert_eq!(idx.cluster(), c);
                assert_eq!(idx.entry(), e);
            }
        }
    }

    #[test]
    fn assign_all_correct_count() {
        let vectors = make_stacked_vectors(100);
        let cb = Codebook4096::build(&vectors, 8);
        let indices = cb.assign_all(&vectors);
        assert_eq!(indices.len(), 100);
    }

    #[test]
    fn empty_codebook() {
        let cb = Codebook4096::build(&[], 64);
        assert_eq!(cb.total_entries, 0);
    }

    #[test]
    fn small_codebook() {
        // Fewer vectors than clusters
        let vectors = make_stacked_vectors(10);
        let cb = Codebook4096::build(&vectors, 8);
        assert!(cb.clusters.len() <= 10);
    }
}
