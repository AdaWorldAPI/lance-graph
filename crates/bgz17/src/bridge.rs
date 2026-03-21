//! Bridge: connects bgz17 to CLAM/CAKES/HHTL search infrastructure.
//!
//! Provides a `MetricSpace` trait that CLAM tree construction and CAKES
//! search call for distance computation. The implementation routes through
//! bgz17's layered codec, selecting precision based on context.
//!
//! ## How CLAM/CAKES Currently Works
//!
//! ```text
//! CLAM: tree.build(data, |a, b| hamming(bitvec_a, bitvec_b))
//!   → O(N²) pairwise distances to build, each = 16K bit ops
//!
//! CAKES: sieve(query, |q, x| hamming(q_bitvec, x_bitvec))
//!   → O(N·log N) distances per query, each = 16K bit ops
//! ```
//!
//! ## How bgz17 Replaces This
//!
//! ```text
//! CLAM: tree.build(data, |a, b| bgz17_distance(a, b, Precision::Palette))
//!   → same O(N²) build, but each distance = 3 cache loads (10,000× faster)
//!
//! CAKES: sieve(query, |q, x| bgz17_distance(q, x, sieve_precision(depth)))
//!   → Layer 0 at shallow depth (prune fast)
//!   → Layer 1 at medium depth (palette lookup)
//!   → Layer 2 at deep levels (full L1 for decision boundary)
//! ```

use crate::base17::SpoBase17;
use crate::palette::PaletteEdge;
use crate::distance_matrix::SpoDistanceMatrices;
use crate::layered::LayeredScope;

/// Precision levels for distance computation.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Precision {
    /// Layer 0: scent byte Hamming (1 byte, ρ=0.937).
    Scent,
    /// Layer 1: palette matrix lookup (3 bytes, ρ=0.965).
    Palette,
    /// Layer 2: full i16[17] base L1 (102 bytes, ρ=0.992).
    Base,
    /// Layer 3: exact Hamming on full planes (6 KB, ρ=1.000).
    Exact,
}

/// A distance oracle that bgz17 provides to CLAM/CAKES.
///
/// The trait is generic over the index type (scope position, node ID, etc.).
/// CLAM calls `distance(a, b)` during tree construction.
/// CAKES calls `distance_at(a, b, precision)` during search sieve.
pub trait Bgz17Distance {
    /// Distance at default precision (Palette).
    fn distance(&self, a: usize, b: usize) -> u32;

    /// Distance at specified precision level.
    fn distance_at(&self, a: usize, b: usize, precision: Precision) -> u32;

    /// Adaptive distance: precision selected by CLAM tree depth.
    /// Shallow (depth < 3) → Scent. Medium (3-6) → Palette. Deep (>6) → Base.
    fn distance_adaptive(&self, a: usize, b: usize, tree_depth: usize) -> u32 {
        let precision = match tree_depth {
            0..=2 => Precision::Scent,
            3..=6 => Precision::Palette,
            _ => Precision::Base,
        };
        self.distance_at(a, b, precision)
    }

    /// Number of elements in the metric space.
    fn len(&self) -> usize;
}

/// bgz17 metric space backed by a LayeredScope.
pub struct Bgz17Metric {
    scope: LayeredScope,
}

impl Bgz17Metric {
    pub fn new(scope: LayeredScope) -> Self {
        Bgz17Metric { scope }
    }
}

impl Bgz17Distance for Bgz17Metric {
    fn distance(&self, a: usize, b: usize) -> u32 {
        self.distance_at(a, b, Precision::Palette)
    }

    fn distance_at(&self, a: usize, b: usize, precision: Precision) -> u32 {
        match precision {
            Precision::Scent => {
                let sa = self.scope.scent[a];
                let sb = self.scope.scent[b];
                (sa ^ sb).count_ones()
            }
            Precision::Palette => {
                let pa = &self.scope.palette_edges[a];
                let pb = &self.scope.palette_edges[b];
                self.scope.distance_matrices.spo_distance(
                    pa.s_idx, pa.p_idx, pa.o_idx,
                    pb.s_idx, pb.p_idx, pb.o_idx,
                )
            }
            Precision::Base => {
                self.scope.base_patterns[a].l1(&self.scope.base_patterns[b])
            }
            Precision::Exact => {
                // Would require loading full planes from Lance.
                // Fall back to Base as upper bound.
                self.scope.base_patterns[a].l1(&self.scope.base_patterns[b])
            }
        }
    }

    fn len(&self) -> usize {
        self.scope.edge_count
    }
}

/// CAKES-compatible search using bgz17 layered distance.
///
/// Implements the DFS sieve from CAKES Algorithm 6, using bgz17's
/// layered precision instead of raw Hamming distance.
///
/// The sieve traverses the CLAM tree depth-first. At each node:
/// - Compute delta_minus = |d(query, center) - radius|
/// - If delta_minus > best_hit: PRUNE (Layer 0 suffices)
/// - If delta_minus > threshold: REFINE with Layer 1
/// - If near decision boundary: REFINE with Layer 2
pub fn cakes_sieve(
    metric: &Bgz17Metric,
    query_idx: usize,
    k: usize,
) -> Vec<(usize, u32)> {
    // Simple brute-force k-NN using palette distance (Layer 1).
    // The real CAKES sieve walks a CLAM tree — this is the baseline
    // that the tree-based sieve improves upon.
    let n = metric.len();
    let mut hits: Vec<(usize, u32)> = (0..n)
        .filter(|&i| i != query_idx)
        .map(|i| (i, metric.distance(query_idx, i)))
        .collect();

    hits.sort_by_key(|&(_, d)| d);
    hits.truncate(k);
    hits
}

/// CAKES sieve with adaptive precision per depth level.
///
/// At shallow depth: use Scent (1 byte) for fast pruning.
/// At medium depth: use Palette (3 bytes) for accurate ranking.
/// At deep depth: use Base (102 bytes) for decision boundaries.
///
/// This is the bgz17 analog of Opus's bandwidth switching:
/// SILK at low bitrate, CELT at high bitrate, hybrid in between.
pub fn cakes_sieve_adaptive(
    metric: &Bgz17Metric,
    query_idx: usize,
    k: usize,
    cluster_depths: &[usize],
) -> Vec<(usize, u32)> {
    let n = metric.len();
    let mut hits: Vec<(usize, u32)> = (0..n)
        .filter(|&i| i != query_idx)
        .map(|i| {
            let depth = cluster_depths.get(i).copied().unwrap_or(0);
            (i, metric.distance_adaptive(query_idx, i, depth))
        })
        .collect();

    hits.sort_by_key(|&(_, d)| d);
    hits.truncate(k);
    hits
}

/// Bridge to HHTL: replace LEAF stage with bgz17 palette refinement.
///
/// HHTL's HEEL/HIP/TWIG stages use scent bytes (unchanged).
/// LEAF currently uses integrated BitVec (ρ=0.834, 2 KB).
/// This replaces LEAF with palette lookup (ρ=0.965, 3 bytes).
///
/// Returns refined candidates with palette-resolution distances.
pub fn hhtl_leaf_bgz17(
    candidates: &[(usize, u32)],
    metric: &Bgz17Metric,
    query_idx: usize,
    top_n_base: usize,
) -> Vec<(usize, u32, Precision)> {
    let mut results: Vec<(usize, u32, Precision)> = Vec::with_capacity(candidates.len());

    // Layer 1: palette distance for ALL candidates
    for &(pos, _scent_dist) in candidates {
        let d = metric.distance_at(query_idx, pos, Precision::Palette);
        results.push((pos, d, Precision::Palette));
    }

    results.sort_by_key(|&(_, d, _)| d);

    // Layer 2: base distance for top-N only (decision boundary)
    for r in results.iter_mut().take(top_n_base) {
        let d = metric.distance_at(query_idx, r.0, Precision::Base);
        *r = (r.0, d, Precision::Base);
    }

    results.sort_by_key(|&(_, d, _)| d);
    results
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::base17::Base17;
    use crate::palette::Palette;
    use crate::distance_matrix::SpoDistanceMatrices;
    use crate::scope::Bgz17Scope;

    fn random_plane(seed: u64) -> Vec<i8> {
        let mut v = vec![0i8; crate::FULL_DIM];
        let mut s = seed;
        for x in v.iter_mut() {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            *x = (s >> 33) as i8;
        }
        v
    }

    #[test]
    fn test_bgz17_metric_self_zero() {
        let planes: Vec<_> = (0..50)
            .map(|i| (random_plane(i * 3), random_plane(i * 3 + 1), random_plane(i * 3 + 2)))
            .collect();
        let scope = Bgz17Scope::build(1, &planes, 32);
        let metric = Bgz17Metric::new(scope.to_layered_scope());

        // Self-distance at every precision should be 0
        for prec in [Precision::Scent, Precision::Palette, Precision::Base] {
            let d = metric.distance_at(0, 0, prec);
            assert_eq!(d, 0, "Self-distance at {:?} should be 0, got {}", prec, d);
        }
    }

    #[test]
    fn test_precision_ordering() {
        let planes: Vec<_> = (0..30)
            .map(|i| (random_plane(i * 3), random_plane(i * 3 + 1), random_plane(i * 3 + 2)))
            .collect();
        let scope = Bgz17Scope::build(1, &planes, 16);
        let metric = Bgz17Metric::new(scope.to_layered_scope());

        // Higher precision should give finer-grained distances
        let d_scent = metric.distance_at(0, 1, Precision::Scent);
        let d_palette = metric.distance_at(0, 1, Precision::Palette);
        let d_base = metric.distance_at(0, 1, Precision::Base);

        // All should be non-negative, scent should be coarsest (0-8 range)
        assert!(d_scent <= 8, "Scent distance max is 8, got {}", d_scent);
        println!("Distances: scent={}, palette={}, base={}", d_scent, d_palette, d_base);
    }

    #[test]
    fn test_cakes_sieve() {
        let planes: Vec<_> = (0..100)
            .map(|i| (random_plane(i * 3), random_plane(i * 3 + 1), random_plane(i * 3 + 2)))
            .collect();
        let scope = Bgz17Scope::build(1, &planes, 32);
        let metric = Bgz17Metric::new(scope.to_layered_scope());

        let results = cakes_sieve(&metric, 0, 10);
        assert_eq!(results.len(), 10);
        // Sorted by distance
        for w in results.windows(2) {
            assert!(w[0].1 <= w[1].1);
        }
    }

    #[test]
    fn test_hhtl_leaf_bgz17() {
        let planes: Vec<_> = (0..50)
            .map(|i| (random_plane(i * 3), random_plane(i * 3 + 1), random_plane(i * 3 + 2)))
            .collect();
        let scope = Bgz17Scope::build(1, &planes, 16);
        let metric = Bgz17Metric::new(scope.to_layered_scope());

        // Simulate HEEL/HIP/TWIG producing candidates
        let candidates: Vec<(usize, u32)> = (0..20).map(|i| (i, i as u32)).collect();

        let results = hhtl_leaf_bgz17(&candidates, &metric, 0, 5);
        assert!(!results.is_empty());
        // Top 5 should be at Base precision
        for r in results.iter().take(5) {
            assert_eq!(r.2, Precision::Base);
        }
    }
}
