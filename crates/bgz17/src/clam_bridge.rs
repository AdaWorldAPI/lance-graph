//! Bridge between bgz17 LayeredScope and CLAM/CAKES search algorithms.
//!
//! Provides a `Bgz17Metric` that uses the layered distance codec
//! (scent → palette → base17) instead of raw Hamming distance.
//! The CLAM tree is built over palette-encoded edges, and search
//! algorithms benefit from the layered pruning cascade.
//!
//! ## Design
//!
//! This module replicates the minimal CLAM tree interface locally
//! (no ndarray dependency) so that the bridge can be tested standalone.
//! The trait signatures match ndarray's `ClamTree` so the types can be
//! used as a drop-in when wired into production.
//!
//! ## Layered distance protocol
//!
//! ```text
//! 1. Scent check (1 byte XOR popcount) — prunes obviously distant pairs
//! 2. Palette matrix lookup (3 cache loads) — O(1) approximate distance
//! 3. Base17 L1 (102 bytes) — precise distance for tight candidates
//! ```
//!
//! The `Bgz17Metric::distance()` method cascades through layers,
//! stopping as soon as a layer provides sufficient resolution.
//! Layer utilization stats are tracked for diagnostics.

use crate::base17::SpoBase17;
use crate::distance_matrix::SpoDistanceMatrices;
use crate::palette::PaletteEdge;
use crate::scope::Bgz17Scope;

use std::cell::Cell;

// ─── Layer utilization tracking ─────────────────────────────────

/// Tracks how many distance calls were resolved at each layer.
#[derive(Debug, Clone, Copy, Default)]
pub struct LayerStats {
    /// Calls resolved at Layer 0 (scent pruning).
    pub scent_resolved: u64,
    /// Calls resolved at Layer 1 (palette lookup).
    pub palette_resolved: u64,
    /// Calls resolved at Layer 2 (base17 L1).
    pub base_resolved: u64,
    /// Total distance calls.
    pub total_calls: u64,
}

impl LayerStats {
    /// Reset counters.
    pub fn reset(&mut self) {
        *self = Self::default();
    }

    /// Print utilization percentages.
    pub fn report(&self) -> String {
        if self.total_calls == 0 {
            return "LayerStats: no calls".to_string();
        }
        let pct = |n: u64| 100.0 * n as f64 / self.total_calls as f64;
        format!(
            "LayerStats: {} total calls — scent {:.1}%, palette {:.1}%, base {:.1}%",
            self.total_calls,
            pct(self.scent_resolved),
            pct(self.palette_resolved),
            pct(self.base_resolved),
        )
    }
}

// ─── Bgz17Metric ────────────────────────────────────────────────

/// A distance metric using the bgz17 layered codec.
///
/// Wraps a `Bgz17Scope` and provides a distance function compatible
/// with CLAM tree construction and search.
///
/// Points are identified by index into the scope's edge arrays.
/// The distance function cascades through layers:
///   scent → palette → base17
///
/// The `scent_threshold` controls when scent alone is sufficient.
/// If the scent distance exceeds the threshold, the pair is "obviously
/// distant" and we return a large sentinel without consulting palette.
pub struct Bgz17Metric {
    /// Precomputed distance matrices for O(1) palette lookups.
    pub matrices: SpoDistanceMatrices,
    /// Per-edge palette indices.
    pub palette_edges: Vec<PaletteEdge>,
    /// Per-edge scent bytes.
    pub scent: Vec<u8>,
    /// Per-edge base17 patterns (for Layer 2 refinement).
    pub base_patterns: Vec<SpoBase17>,
    /// Edge count.
    pub edge_count: usize,
    /// Scent popcount threshold: if XOR popcount > threshold, skip palette.
    /// Default: 5 (out of 8 bits — only 1-2-3 bit agreement passes).
    pub scent_threshold: u32,
    /// Layer utilization stats (interior mutability for use in `distance`).
    stats: Cell<LayerStats>,
}

impl Bgz17Metric {
    /// Build from a `Bgz17Scope`.
    pub fn from_scope(scope: &Bgz17Scope) -> Self {
        Bgz17Metric {
            matrices: scope.matrices.clone(),
            palette_edges: scope.palette_indices.clone(),
            scent: scope.scent.clone(),
            base_patterns: scope.base_patterns.clone(),
            edge_count: scope.edge_count,
            scent_threshold: 5,
            stats: Cell::new(LayerStats::default()),
        }
    }

    /// Layered distance between two edges by index.
    ///
    /// Returns a `u64` distance suitable for CLAM tree operations.
    /// The value is the palette distance (Layer 1) for most pairs,
    /// upgraded to base17 L1 (Layer 2) for tight candidates.
    pub fn distance(&self, a: usize, b: usize) -> u64 {
        let mut stats = self.stats.get();
        stats.total_calls += 1;

        // Layer 0: scent check
        let scent_xor = self.scent[a] ^ self.scent[b];
        let scent_dist = scent_xor.count_ones();

        if scent_dist > self.scent_threshold {
            // Obviously distant — return large sentinel based on scent
            // Scale: scent_dist is 0-8, map to distance space
            // Use scent_dist * 10000 as a coarse upper bound
            stats.scent_resolved += 1;
            self.stats.set(stats);
            return scent_dist as u64 * 10000;
        }

        // Layer 1: palette matrix lookup (3 cache loads)
        let pe_a = &self.palette_edges[a];
        let pe_b = &self.palette_edges[b];
        let palette_dist = self.matrices.spo_distance(
            pe_a.s_idx, pe_a.p_idx, pe_a.o_idx,
            pe_b.s_idx, pe_b.p_idx, pe_b.o_idx,
        ) as u64;

        // For CLAM tree construction we need consistent distances.
        // Palette resolution is sufficient for most clustering decisions.
        // Only refine to base17 if palette distance is very small
        // (decision boundary where palette quantization error matters).
        if palette_dist > 100 {
            stats.palette_resolved += 1;
            self.stats.set(stats);
            return palette_dist;
        }

        // Layer 2: base17 L1 (precise)
        let base_dist = self.base_patterns[a].l1(&self.base_patterns[b]) as u64;
        stats.base_resolved += 1;
        self.stats.set(stats);
        base_dist
    }

    /// Get current layer utilization stats.
    pub fn stats(&self) -> LayerStats {
        self.stats.get()
    }

    /// Reset layer stats.
    pub fn reset_stats(&self) {
        self.stats.set(LayerStats::default());
    }
}

// ─── Minimal CLAM tree replica ──────────────────────────────────
//
// These types replicate ndarray's ClamTree interface just enough to
// build and search a tree over bgz17 edges. The signatures match
// ndarray so this can be swapped to a real dependency later.

/// Simple SplitMix64 RNG for deterministic seed selection.
struct SplitMix64(u64);

impl SplitMix64 {
    fn new(seed: u64) -> Self { Self(seed) }
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^ (z >> 31)
    }
}

/// Local Fractal Dimension.
#[derive(Debug, Clone, Copy)]
pub struct Lfd {
    pub value: f64,
}

impl Lfd {
    fn compute(count_r: usize, count_half_r: usize) -> Self {
        let value = if count_half_r == 0 || count_r <= count_half_r {
            0.0
        } else {
            (count_r as f64 / count_half_r as f64).log2()
        };
        Lfd { value }
    }
}

/// A cluster node in the bgz17 CLAM tree.
#[derive(Debug, Clone)]
pub struct Bgz17Cluster {
    pub center_idx: usize,
    pub radius: u64,
    pub cardinality: usize,
    pub offset: usize,
    pub depth: usize,
    pub lfd: Lfd,
    pub left: Option<usize>,
    pub right: Option<usize>,
}

impl Bgz17Cluster {
    #[inline]
    pub fn is_leaf(&self) -> bool { self.left.is_none() }

    #[inline]
    pub fn delta_plus(&self, dist_to_center: u64) -> u64 {
        dist_to_center.saturating_add(self.radius)
    }

    #[inline]
    pub fn delta_minus(&self, dist_to_center: u64) -> u64 {
        dist_to_center.saturating_sub(self.radius)
    }
}

/// CLAM tree built over bgz17 edges using layered distance.
pub struct Bgz17ClamTree {
    pub nodes: Vec<Bgz17Cluster>,
    pub reordered: Vec<usize>,
    pub num_leaves: usize,
    pub metric: Bgz17Metric,
}

impl Bgz17ClamTree {
    /// Build a CLAM tree from a `Bgz17Scope` using layered distance.
    pub fn build(scope: &Bgz17Scope, min_cluster_size: usize) -> Self {
        let metric = Bgz17Metric::from_scope(scope);
        let n = metric.edge_count;

        let mut indices: Vec<usize> = (0..n).collect();
        let mut nodes = Vec::with_capacity(2 * n);
        let mut rng = SplitMix64::new(0xDEAD_BEEF_CAFE_BABE);

        if n > 0 {
            Self::partition(
                &metric,
                &mut indices,
                0, n, 0,
                min_cluster_size.max(1),
                &mut nodes,
                &mut rng,
            );
        }

        let num_leaves = nodes.iter().filter(|c| c.is_leaf()).count();

        Bgz17ClamTree {
            nodes,
            reordered: indices,
            num_leaves,
            metric,
        }
    }

    /// Recursive partition (CAKES Algorithm 1).
    #[allow(clippy::too_many_arguments)]
    fn partition(
        metric: &Bgz17Metric,
        indices: &mut [usize],
        start: usize,
        end: usize,
        depth: usize,
        min_card: usize,
        nodes: &mut Vec<Bgz17Cluster>,
        rng: &mut SplitMix64,
    ) -> usize {
        let n = end - start;
        let node_idx = nodes.len();
        let working = &mut indices[start..end];

        // Find center: geometric median of sqrt(n) seeds
        let num_seeds = (n as f64).sqrt().ceil() as usize;
        let num_seeds = num_seeds.max(1).min(n);

        for i in 0..num_seeds.min(working.len()) {
            let j = i + (rng.next_u64() as usize % (working.len() - i));
            working.swap(i, j);
        }

        let center_local = if num_seeds <= 1 {
            0
        } else {
            let mut best_idx = 0;
            let mut best_sum = u64::MAX;
            for s in 0..num_seeds {
                let si = working[s];
                let mut sum = 0u64;
                for (t, &wt) in working.iter().enumerate().take(num_seeds) {
                    if s != t {
                        sum += metric.distance(si, wt);
                    }
                }
                if sum < best_sum {
                    best_sum = sum;
                    best_idx = s;
                }
            }
            best_idx
        };

        working.swap(0, center_local);
        let center_idx = working[0];

        // Compute radius + find left pole
        let mut radius = 0u64;
        let mut left_pole_local = 0;
        let mut left_pole_dist = 0u64;

        let mut distances: Vec<u64> = Vec::with_capacity(n);
        for (i, &wi) in working.iter().enumerate() {
            let d = metric.distance(center_idx, wi);
            distances.push(d);
            if d > radius { radius = d; }
            if d > left_pole_dist {
                left_pole_dist = d;
                left_pole_local = i;
            }
        }

        // LFD
        let half_radius = radius / 2;
        let count_r = distances.iter().filter(|&&d| d <= radius).count();
        let count_half_r = distances.iter().filter(|&&d| d <= half_radius).count();
        let lfd = Lfd::compute(count_r, count_half_r);

        // Find right pole
        let left_pole_idx = working[left_pole_local];
        let mut right_pole_local = 0;
        let mut right_pole_dist = 0u64;
        for (i, &wi) in working.iter().enumerate() {
            let d = metric.distance(left_pole_idx, wi);
            if d > right_pole_dist {
                right_pole_dist = d;
                right_pole_local = i;
            }
        }
        let right_pole_idx = working[right_pole_local];

        // Partition into L and R
        let mut side: Vec<bool> = Vec::with_capacity(n);
        for &wi in working.iter() {
            let dl = metric.distance(left_pole_idx, wi);
            let dr = metric.distance(right_pole_idx, wi);
            side.push(dl <= dr);
        }

        let mut cursor = 0;
        for i in 0..n {
            if side[i] {
                working.swap(cursor, i);
                side.swap(cursor, i);
                cursor += 1;
            }
        }
        let split = cursor;

        nodes.push(Bgz17Cluster {
            center_idx,
            radius,
            cardinality: n,
            offset: start,
            depth,
            lfd,
            left: None,
            right: None,
        });

        let should_split = n > min_card
            && depth < 256
            && radius > 0
            && split > 0
            && split < n;

        if should_split {
            let left_idx = Self::partition(
                metric, indices, start, start + split, depth + 1, min_card, nodes, rng,
            );
            nodes[node_idx].left = Some(left_idx);

            let right_idx = Self::partition(
                metric, indices, start + split, end, depth + 1, min_card, nodes, rng,
            );
            nodes[node_idx].right = Some(right_idx);
        }

        node_idx
    }

    // ─── Search algorithms (CAKES) ──────────────────────────────

    /// rho-NN search: find all edges within radius rho of query edge.
    pub fn rho_nn(&self, query_idx: usize, rho: u64) -> Vec<(usize, u64)> {
        let mut hits = Vec::new();
        let mut stack = vec![0usize];

        while let Some(node_idx) = stack.pop() {
            let cluster = &self.nodes[node_idx];
            let delta = self.metric.distance(query_idx, cluster.center_idx);
            let d_minus = cluster.delta_minus(delta);

            if d_minus > rho {
                continue;
            }

            if cluster.is_leaf() {
                let start = cluster.offset;
                let end = start + cluster.cardinality;
                for &edge_idx in &self.reordered[start..end] {
                    let d = self.metric.distance(query_idx, edge_idx);
                    if d <= rho {
                        hits.push((edge_idx, d));
                    }
                }
            } else {
                if let Some(left) = cluster.left { stack.push(left); }
                if let Some(right) = cluster.right { stack.push(right); }
            }
        }

        hits.sort_by_key(|&(_, d)| d);
        hits
    }

    /// k-NN via Repeated rho-NN (CAKES Algorithm 4).
    pub fn knn_repeated_rho(&self, query_idx: usize, k: usize) -> Vec<(usize, u64)> {
        if self.nodes.is_empty() {
            return Vec::new();
        }
        let root = &self.nodes[0];
        let mut rho = root.radius / root.cardinality.max(1) as u64;
        if rho == 0 { rho = 1; }

        loop {
            let hits = self.rho_nn(query_idx, rho);
            if hits.len() >= k {
                return hits.into_iter().take(k).collect();
            }
            if hits.is_empty() {
                rho *= 2;
            } else {
                let ratio = k as f64 / hits.len() as f64;
                let factor = ratio.clamp(1.1, 2.0);
                rho = ((rho as f64) * factor).ceil() as u64;
            }
            if rho > root.radius {
                rho = root.radius;
                let hits = self.rho_nn(query_idx, rho);
                return hits.into_iter().take(k).collect();
            }
        }
    }

    /// k-NN via Depth-First Sieve (CAKES Algorithm 6).
    pub fn knn_dfs_sieve(&self, query_idx: usize, k: usize) -> Vec<(usize, u64)> {
        use std::cmp::Reverse;
        use std::collections::BinaryHeap;

        if self.nodes.is_empty() {
            return Vec::new();
        }

        // Q: min-heap of (delta_minus, node_idx)
        let mut queue: BinaryHeap<Reverse<(u64, usize)>> = BinaryHeap::new();
        // H: max-heap of (distance, edge_idx) — worst hit on top
        let mut hits: BinaryHeap<(u64, usize)> = BinaryHeap::new();

        let root = &self.nodes[0];
        let root_delta = self.metric.distance(query_idx, root.center_idx);
        queue.push(Reverse((root.delta_minus(root_delta), 0)));

        while let Some(&Reverse((d_minus, node_idx))) = queue.peek() {
            if hits.len() >= k {
                if let Some(&(worst, _)) = hits.peek() {
                    if worst <= d_minus { break; }
                }
            }

            queue.pop();
            let cluster = &self.nodes[node_idx];

            if cluster.is_leaf() {
                let start = cluster.offset;
                let end = start + cluster.cardinality;
                for &edge_idx in &self.reordered[start..end] {
                    let d = self.metric.distance(query_idx, edge_idx);
                    if hits.len() < k {
                        hits.push((d, edge_idx));
                    } else if let Some(&(worst, _)) = hits.peek() {
                        if d < worst {
                            hits.pop();
                            hits.push((d, edge_idx));
                        }
                    }
                }
            } else {
                for child_idx in [cluster.left, cluster.right].iter().flatten() {
                    let child = &self.nodes[*child_idx];
                    let child_delta = self.metric.distance(query_idx, child.center_idx);
                    let child_d_minus = child.delta_minus(child_delta);

                    if hits.len() >= k {
                        if let Some(&(worst, _)) = hits.peek() {
                            if child_d_minus > worst { continue; }
                        }
                    }
                    queue.push(Reverse((child_d_minus, *child_idx)));
                }
            }
        }

        let mut result: Vec<(usize, u64)> = hits.into_iter().map(|(d, idx)| (idx, d)).collect();
        result.sort_by_key(|&(_, d)| d);
        result
    }

    /// Brute-force k-NN for ground-truth comparison.
    pub fn brute_force_knn(&self, query_idx: usize, k: usize) -> Vec<(usize, u64)> {
        let mut dists: Vec<(usize, u64)> = (0..self.metric.edge_count)
            .map(|i| {
                // Use base17 L1 for ground truth (most precise layer available)
                let d = self.metric.base_patterns[query_idx].l1(&self.metric.base_patterns[i]) as u64;
                (i, d)
            })
            .collect();
        dists.sort_by_key(|&(_, d)| d);
        dists.truncate(k);
        dists
    }
}

/// Build a `Bgz17ClamTree` from a scope.
///
/// This is the main entry point for the bridge. Takes a bgz17 scope
/// (built from raw accumulator planes) and returns a CLAM tree that
/// uses layered distance for all operations.
pub fn build_clam_tree(scope: &Bgz17Scope, min_cluster_size: usize) -> Bgz17ClamTree {
    Bgz17ClamTree::build(scope, min_cluster_size)
}

// ─── Tests ──────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scope::Bgz17Scope;

    fn random_plane(seed: u64) -> Vec<i8> {
        let mut v = vec![0i8; 16384];
        let mut s = seed;
        for x in v.iter_mut() {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            *x = (s >> 33) as i8;
        }
        v
    }

    fn make_test_scope(n: usize) -> Bgz17Scope {
        let planes: Vec<(Vec<i8>, Vec<i8>, Vec<i8>)> = (0..n)
            .map(|i| {
                let seed = i as u64;
                (
                    random_plane(seed * 3),
                    random_plane(seed * 3 + 1),
                    random_plane(seed * 3 + 2),
                )
            })
            .collect();
        Bgz17Scope::build(1, &planes, 32)
    }

    #[test]
    fn test_bgz17_metric_self_distance() {
        let scope = make_test_scope(20);
        let metric = Bgz17Metric::from_scope(&scope);

        // Self-distance should be 0
        for i in 0..20 {
            assert_eq!(metric.distance(i, i), 0, "self-distance of edge {} should be 0", i);
        }
    }

    #[test]
    fn test_bgz17_metric_symmetric() {
        let scope = make_test_scope(20);
        let metric = Bgz17Metric::from_scope(&scope);

        for i in 0..20 {
            for j in 0..20 {
                assert_eq!(
                    metric.distance(i, j),
                    metric.distance(j, i),
                    "distance({}, {}) should be symmetric",
                    i, j
                );
            }
        }
    }

    #[test]
    fn test_build_clam_tree() {
        let scope = make_test_scope(50);
        let tree = build_clam_tree(&scope, 3);

        assert!(!tree.nodes.is_empty(), "tree should have nodes");
        assert_eq!(tree.reordered.len(), 50);
        assert!(tree.num_leaves > 0, "tree should have leaves");

        println!("Bgz17ClamTree: {} nodes, {} leaves", tree.nodes.len(), tree.num_leaves);
    }

    #[test]
    fn test_rho_nn_self() {
        let scope = make_test_scope(50);
        let tree = build_clam_tree(&scope, 3);

        // Query edge 0 at rho=0 should find itself
        let hits = tree.rho_nn(0, 0);
        assert!(!hits.is_empty(), "rho_nn(0, 0) should find at least the query");
        assert_eq!(hits[0].0, 0, "first hit should be edge 0");
        assert_eq!(hits[0].1, 0, "self-distance should be 0");
    }

    #[test]
    fn test_knn_matches_brute_force() {
        let scope = make_test_scope(100);
        let tree = build_clam_tree(&scope, 3);

        let k = 5;
        let query_idx = 7;

        // Brute-force using palette distance (same metric the tree uses)
        let mut bf: Vec<(usize, u64)> = (0..100)
            .map(|i| (i, tree.metric.distance(query_idx, i)))
            .collect();
        bf.sort_by_key(|&(_, d)| d);
        bf.truncate(k);

        // DFS Sieve
        tree.metric.reset_stats();
        let dfs_result = tree.knn_dfs_sieve(query_idx, k);

        assert_eq!(
            dfs_result.len(), k,
            "DFS sieve should return {} hits, got {}",
            k, dfs_result.len()
        );

        // The k-th distance should match brute-force k-th distance
        let dfs_max = dfs_result.last().unwrap().1;
        let bf_max = bf.last().unwrap().1;
        assert_eq!(
            dfs_max, bf_max,
            "DFS sieve k-th distance ({}) should match brute-force k-th distance ({})",
            dfs_max, bf_max
        );

        let stats = tree.metric.stats();
        println!("DFS Sieve: {}", stats.report());
    }

    #[test]
    fn test_knn_repeated_rho_matches_brute_force() {
        let scope = make_test_scope(100);
        let tree = build_clam_tree(&scope, 3);

        let k = 5;
        let query_idx = 3;

        // Brute-force
        let mut bf: Vec<(usize, u64)> = (0..100)
            .map(|i| (i, tree.metric.distance(query_idx, i)))
            .collect();
        bf.sort_by_key(|&(_, d)| d);
        bf.truncate(k);

        tree.metric.reset_stats();
        let rr_result = tree.knn_repeated_rho(query_idx, k);

        assert_eq!(rr_result.len(), k);
        let rr_max = rr_result.last().unwrap().1;
        let bf_max = bf.last().unwrap().1;
        assert_eq!(
            rr_max, bf_max,
            "Repeated rho-NN k-th distance ({}) should match brute-force ({})",
            rr_max, bf_max
        );

        let stats = tree.metric.stats();
        println!("Repeated rho-NN: {}", stats.report());
    }

    #[test]
    fn test_layer_utilization_stats() {
        let scope = make_test_scope(200);
        let tree = build_clam_tree(&scope, 5);

        tree.metric.reset_stats();

        // Run several queries
        for q in 0..10 {
            let _ = tree.knn_dfs_sieve(q, 5);
        }

        let stats = tree.metric.stats();
        println!("\n=== Layer Utilization (10 queries, 200 edges) ===");
        println!("{}", stats.report());
        println!(
            "  Scent resolved:   {:>6} ({:.1}%)",
            stats.scent_resolved,
            if stats.total_calls > 0 {
                100.0 * stats.scent_resolved as f64 / stats.total_calls as f64
            } else {
                0.0
            }
        );
        println!(
            "  Palette resolved: {:>6} ({:.1}%)",
            stats.palette_resolved,
            if stats.total_calls > 0 {
                100.0 * stats.palette_resolved as f64 / stats.total_calls as f64
            } else {
                0.0
            }
        );
        println!(
            "  Base resolved:    {:>6} ({:.1}%)",
            stats.base_resolved,
            if stats.total_calls > 0 {
                100.0 * stats.base_resolved as f64 / stats.total_calls as f64
            } else {
                0.0
            }
        );

        assert!(stats.total_calls > 0, "should have made some distance calls");
    }

    #[test]
    fn test_brute_force_knn_consistency() {
        let scope = make_test_scope(50);
        let tree = build_clam_tree(&scope, 3);

        let bf = tree.brute_force_knn(0, 5);
        assert_eq!(bf.len(), 5);
        assert_eq!(bf[0].0, 0, "self should be closest");
        assert_eq!(bf[0].1, 0, "self-distance should be 0");

        // Check sorted
        for w in bf.windows(2) {
            assert!(w[0].1 <= w[1].1, "brute-force results should be sorted");
        }
    }
}
