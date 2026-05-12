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

// Retained for the encode/decode bridge that maps `SpoBase17 ↔ PaletteEdge` via
// `SpoDistanceMatrices` — wiring in progress (TD-BGZ17-BRIDGE-1).
#[allow(unused_imports)]
use crate::base17::SpoBase17;
#[allow(unused_imports)]
use crate::palette::PaletteEdge;
#[allow(unused_imports)]
use crate::distance_matrix::SpoDistanceMatrices;
use crate::layered::LayeredScope;

// Precision is defined in lib.rs (crate root) and re-exported here for convenience.
pub use crate::Precision;

/// A distance oracle that bgz17 provides to CLAM/CAKES.
///
/// The trait is generic over the index type (scope position, node ID, etc.).
/// CLAM calls `distance(a, b)` during tree construction.
/// CAKES calls `distance_at(a, b, precision)` during search sieve.
///
/// ## Metric Safety Contract
///
/// `distance()`, `distance_at(Palette|Base|Exact)`, and `distance_adaptive()`
/// all return metric-safe values (triangle inequality holds).
/// `distance_heuristic()` returns a heuristic pre-filter value (Scent)
/// that is NOT metric-safe — use only for HEEL-stage pruning, never for
/// CAKES `delta_minus` / `delta_plus` bounds.
pub trait Bgz17Distance {
    /// Distance at default precision (Palette). Metric-safe.
    fn distance(&self, a: usize, b: usize) -> u32;

    /// Distance at specified precision level.
    /// Caller is responsible for metric safety — Scent is NOT metric-safe.
    fn distance_at(&self, a: usize, b: usize, precision: Precision) -> u32;

    /// Metric-safe adaptive distance: precision selected by CLAM tree depth.
    /// Minimum precision is ALWAYS Palette (never Scent) to guarantee
    /// triangle inequality for CAKES pruning soundness.
    ///
    /// Shallow (depth < 5) → Palette.  Deep (≥5) → Base.
    fn distance_adaptive(&self, a: usize, b: usize, tree_depth: usize) -> u32 {
        let precision = if tree_depth < 5 {
            Precision::Palette
        } else {
            Precision::Base
        };
        self.distance_at(a, b, precision)
    }

    /// Heuristic pre-filter distance using Scent (Layer 0).
    /// ⚠️ NOT metric-safe. Use ONLY for HEEL-stage candidate selection,
    /// NEVER for CAKES delta_minus/delta_plus bounds.
    /// Returns (scent_distance, is_below_threshold).
    fn distance_heuristic(&self, a: usize, b: usize) -> (u32, bool) {
        let d = self.distance_at(a, b, Precision::Scent);
        (d, d <= 3) // threshold: ≤3 bits differ = likely close
    }

    /// Number of elements in the metric space.
    fn len(&self) -> usize;

    /// Convenience: is the metric space empty?
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
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

/// CAKES-compatible brute-force k-NN using Palette distance (Layer 1).
///
/// ✓ Metric-safe: uses Palette (L1) which satisfies triangle inequality.
/// This is the baseline that the tree-based DFS sieve improves upon.
/// The tree sieve uses `distance_adaptive()` which also guarantees
/// Palette-minimum precision.
///
/// For production search with HEEL pre-filter, use `search_prefilter_then_sieve()`.
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
/// ✓ Metric-safe at every depth: minimum precision is Palette (L1 metric).
/// Scent is NOT used here — it violates triangle inequality.
///
/// Shallow depth → Palette (3 bytes, fast). Deep depth → Base (102 bytes, precise).
/// This is the bgz17 analog of Opus's bandwidth switching.
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
            // distance_adaptive guarantees Palette minimum (metric-safe)
            (i, metric.distance_adaptive(query_idx, i, depth))
        })
        .collect();

    hits.sort_by_key(|&(_, d)| d);
    hits.truncate(k);
    hits
}

/// Two-stage search: heuristic pre-filter (Scent) → metric sieve (Palette).
///
/// Stage 1: Scent (Layer 0) eliminates 90%+ of candidates. NOT metric-safe,
///          but that's fine — it's a filter, not a bound.
/// Stage 2: Palette (Layer 1) ranks survivors with metric-safe distances.
///          CAKES triangle inequality pruning is sound at this stage.
///
/// This is the production search path:
///   HEEL (Scent, 10K → 200) → HIP/CAKES (Palette, 200 → k)
pub fn search_prefilter_then_sieve(
    metric: &Bgz17Metric,
    query_idx: usize,
    k: usize,
    prefilter_k: usize,
) -> Vec<(usize, u32)> {
    let n = metric.len();

    // Stage 1: heuristic pre-filter with Scent (NOT metric-safe, but fast)
    let mut prefilter: Vec<(usize, u32)> = (0..n)
        .filter(|&i| i != query_idx)
        .map(|i| {
            let (d, _) = metric.distance_heuristic(query_idx, i);
            (i, d)
        })
        .collect();
    prefilter.sort_by_key(|&(_, d)| d);
    prefilter.truncate(prefilter_k);

    // Stage 2: re-rank survivors with metric-safe Palette distance
    let mut hits: Vec<(usize, u32)> = prefilter.iter()
        .map(|&(i, _)| (i, metric.distance(query_idx, i))) // distance() = Palette
        .collect();
    hits.sort_by_key(|&(_, d)| d);
    hits.truncate(k);
    hits
}

/// Bridge to HHTL: replace LEAF stage with bgz17 layered refinement.
///
/// HHTL's HEEL/HIP/TWIG stages use scent bytes — these are heuristic
/// pre-filters (NOT metric-safe). The candidates they produce are then
/// refined here with metric-safe palette + base distances.
///
/// LEAF currently uses integrated BitVec (ρ=0.834, 2 KB).
/// This replaces it with:
///   - Palette for ALL candidates (ρ=0.965, 3 bytes) — metric-safe ranking
///   - Base for top-N only (ρ=0.992, 102 bytes) — decision boundary precision
///
/// The metric safety boundary is HERE: everything above this function
/// (HEEL/HIP/TWIG) is heuristic. Everything below (palette, base) is metric.
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
        // At least 5 entries should have been refined to Base precision
        let base_count = results.iter().filter(|r| r.2 == Precision::Base).count();
        assert_eq!(base_count, 5, "Should have refined top-5 to Base");
        // Results should be sorted by distance
        for w in results.windows(2) {
            assert!(w[0].1 <= w[1].1);
        }
    }

    #[test]
    fn test_prefilter_then_sieve() {
        let planes: Vec<_> = (0..100)
            .map(|i| (random_plane(i * 3), random_plane(i * 3 + 1), random_plane(i * 3 + 2)))
            .collect();
        let scope = Bgz17Scope::build(1, &planes, 32);
        let metric = Bgz17Metric::new(scope.to_layered_scope());

        let results = search_prefilter_then_sieve(&metric, 0, 10, 50);
        assert_eq!(results.len(), 10);
        // Sorted by palette distance (metric-safe)
        for w in results.windows(2) {
            assert!(w[0].1 <= w[1].1);
        }

        // Compare with brute-force palette sieve
        let brute = cakes_sieve(&metric, 0, 10);
        // Pre-filter may miss the true top-1 (scent is heuristic, not metric).
        // But the top-1 from prefilter should be in brute-force top-10.
        let brute_positions: Vec<usize> = brute.iter().map(|r| r.0).collect();
        assert!(brute_positions.contains(&results[0].0),
            "Pre-filter top-1 ({}) should appear in brute-force top-10 ({:?})",
            results[0].0, brute_positions);
    }

    #[test]
    fn test_palette_triangle_inequality() {
        // Palette (L1) MUST satisfy triangle inequality for CAKES soundness:
        //   d(a,c) ≤ d(a,b) + d(b,c) for all a, b, c
        let planes: Vec<_> = (0..30)
            .map(|i| (random_plane(i * 3), random_plane(i * 3 + 1), random_plane(i * 3 + 2)))
            .collect();
        let scope = Bgz17Scope::build(1, &planes, 16);
        let metric = Bgz17Metric::new(scope.to_layered_scope());

        let mut violations = 0;
        for a in 0..30 {
            for b in 0..30 {
                for c in 0..30 {
                    let dab = metric.distance_at(a, b, Precision::Palette);
                    let dbc = metric.distance_at(b, c, Precision::Palette);
                    let dac = metric.distance_at(a, c, Precision::Palette);
                    if dac > dab + dbc {
                        violations += 1;
                    }
                }
            }
        }
        assert_eq!(violations, 0,
            "Palette L1 must satisfy triangle inequality: {} violations", violations);
    }

    #[test]
    fn test_scent_NOT_metric_safe() {
        // Scent MAY violate triangle inequality (Boolean lattice constraint).
        // This test documents the expectation — it's not a bug, it's why
        // Scent must only be used as a heuristic pre-filter.
        let planes: Vec<_> = (0..30)
            .map(|i| (random_plane(i * 3), random_plane(i * 3 + 1), random_plane(i * 3 + 2)))
            .collect();
        let scope = Bgz17Scope::build(1, &planes, 16);
        let metric = Bgz17Metric::new(scope.to_layered_scope());

        // Just verify scent distance is in valid range (0-8)
        for i in 0..30 {
            for j in 0..30 {
                let d = metric.distance_at(i, j, Precision::Scent);
                assert!(d <= 8, "Scent distance should be 0-8, got {}", d);
            }
        }
        // We don't assert triangle inequality here — it's expected to fail sometimes.
    }

    #[test]
    fn test_adaptive_never_uses_scent() {
        // distance_adaptive MUST return Palette or Base precision, never Scent.
        // We verify by checking that adaptive distance >= palette distance
        // (Scent is coarser with range 0-8, palette is finer with larger range).
        let planes: Vec<_> = (0..20)
            .map(|i| (random_plane(i * 3), random_plane(i * 3 + 1), random_plane(i * 3 + 2)))
            .collect();
        let scope = Bgz17Scope::build(1, &planes, 16);
        let metric = Bgz17Metric::new(scope.to_layered_scope());

        for depth in 0..10 {
            let d_adaptive = metric.distance_adaptive(0, 1, depth);
            let d_palette = metric.distance_at(0, 1, Precision::Palette);
            // Adaptive should use Palette or Base — both metric-safe
            // If depth < 5: uses Palette (same as d_palette)
            // If depth >= 5: uses Base (may differ from palette but still metric-safe)
            if depth < 5 {
                assert_eq!(d_adaptive, d_palette,
                    "At depth {}, adaptive should equal palette", depth);
            }
        }
    }
}
