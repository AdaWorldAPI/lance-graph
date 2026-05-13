//! Query Router: decides whether bgz17 handles a query or falls back.
//!
//! bgz17 handles single-hop similarity search (KNN, range).
//! blasgraph handles multi-hop reasoning (BFS, SSSP, PageRank).
//! ndarray cascade handles bootstrap (< 32 edges, no palette yet).
//!
//! ## The Synergy Map
//!
//! ```text
//! HEEL/HIP/TWIG (scent, Layer 0) ─── heuristic pre-filter ───┐
//!                                                              │
//! LEAF L1: bgz17 palette (3 bytes, ρ=0.965) ← REPLACES       │
//!          integrated BitVec (2KB, ρ=0.834)    KILL THIS       │
//!                                                              │
//! LEAF L2: bgz17 Base17 (102 bytes, ρ=0.992) ← top 10        │
//!                                                              │
//! LEAF L3: exact S+P+O planes (6KB, ρ=1.000) ← top 3/fallback│
//!                                                              │
//! FALLBACK: blasgraph grb_mxm (multi-hop, semiring algebra)   │
//! FALLBACK: ndarray cascade (bootstrap, < 32 edges)           │
//! ```

/// What kind of search the caller wants.
#[derive(Clone, Debug)]
pub enum SearchType {
    /// Single-hop k-nearest neighbors.
    Knn { k: usize },
    /// Single-hop range search (all within distance threshold).
    Range { threshold: u32 },
    /// Multi-hop breadth-first search.
    Bfs { max_depth: usize },
    /// Multi-hop shortest path.
    Sssp,
    /// Anomaly detection (needs CLAM LFD).
    AnomalyDetect,
    /// Semiring algebra (XOR_BUNDLE, RESONANCE, etc).
    SemiringOp,
}

/// Where to route the query.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Route {
    /// bgz17 layered search: scent pre-filter → palette → base → exact.
    /// Handles 95%+ of queries. Single-hop similarity.
    Bgz17Layered,
    /// blasgraph semiring algebra: grb_mxm, BFS, SSSP, PageRank.
    /// Multi-hop paths, vector composition (XOR bind), resonance.
    /// bgz17 can't compose paths — needs BitVec XOR as multiply.
    BlasGraph,
    /// ndarray cascade: Stroke 1 (prefix) → 2 (full Hamming) → 3 (precision).
    /// Used for bootstrap (< 32 edges, no palette built yet) or when
    /// CLAM tree is too shallow (LFD < 1.0, near-uniform distribution).
    NdarrayCascade,
    /// CLAM + CHAODA: anomaly detection using local fractal dimension.
    /// Needs full CLAM tree with LFD scores. Falls back to cascade if
    /// tree not yet built.
    ClamChaoda,
}

/// Route a query to the correct subsystem.
///
/// bgz17 handles single-hop similarity when palette exists (≥ 32 edges).
/// blasgraph handles multi-hop / semiring operations.
/// ndarray cascade handles bootstrap before palette is built.
pub fn route_query(
    edge_count: usize,
    has_clam_tree: bool,
    search_type: &SearchType,
) -> Route {
    match search_type {
        // Single-hop similarity: bgz17 if palette exists
        SearchType::Knn { .. } | SearchType::Range { .. } => {
            if edge_count < 32 {
                Route::NdarrayCascade // Not enough edges for palette
            } else {
                Route::Bgz17Layered  // Palette + optional base refinement
            }
        }

        // Multi-hop: must use blasgraph semirings
        // Palette can't compose paths — needs BitVec XOR as multiply
        SearchType::Bfs { .. } | SearchType::Sssp => {
            Route::BlasGraph
        }

        // Semiring algebra: requires vector operations
        SearchType::SemiringOp => {
            Route::BlasGraph
        }

        // Anomaly detection: CHAODA needs full LFD from CLAM tree
        SearchType::AnomalyDetect => {
            if has_clam_tree {
                Route::ClamChaoda
            } else {
                Route::NdarrayCascade // Build tree first
            }
        }
    }
}

/// Fallback detector: when bgz17 should escalate to a different system.
#[derive(Clone, Debug)]
pub struct FallbackSignals {
    /// Scent prune rate: fraction of edges eliminated at Layer 0.
    /// If < 0.30, scent has low discrimination — consider full HHTL expansion.
    pub scent_prune_rate: f32,
    /// Palette collision rate: fraction of candidate pairs sharing same palette index.
    /// If > 0.10, palette is too coarse — escalate to Layer 2 (Base17).
    pub palette_collision_rate: f32,
    /// CLAM tree depth: shallow trees can't prune effectively.
    /// If < 3, consider ndarray cascade instead.
    pub clam_tree_depth: usize,
    /// Maximum CHAODA anomaly score among candidates.
    /// If > 0.75, load full planes (Layer 3) for ground truth.
    pub max_anomaly_score: f32,
}

impl FallbackSignals {
    /// Determine if bgz17 should escalate from its current layer.
    pub fn should_escalate(&self) -> Option<Route> {
        if self.max_anomaly_score > 0.75 {
            // Anomalous region — can't trust palette
            return Some(Route::NdarrayCascade);
        }
        if self.clam_tree_depth < 3 {
            // Tree too shallow for effective pruning
            return Some(Route::NdarrayCascade);
        }
        if self.scent_prune_rate < 0.30 {
            // Scent not discriminating — still use bgz17 but skip scent pre-filter
            // (go straight to palette brute-force)
            return None; // bgz17 handles it, just skip HEEL
        }
        None // bgz17 handles it fine
    }

    /// Determine minimum precision needed based on signals.
    pub fn minimum_precision(&self) -> crate::Precision {
        if self.max_anomaly_score > 0.75 || self.palette_collision_rate > 0.10 {
            crate::Precision::Base
        } else {
            crate::Precision::Palette
        }
    }
}

/// SIMD integration guidance for the search path.
///
/// Documents which hot paths bgz17 replaces and which remain unchanged.
///
/// ```text
/// ndarray Hot Path              With bgz17                 Speedup
/// ─────────────────────────────────────────────────────────────────
/// cascade Stroke 1 (128B)      → scent byte XOR (1B)      ~128×
/// cascade Stroke 2 (2KB)       → palette lookup (3B)       ~10,000×
/// clam_search::rho_nn          → palette on 3B edges       ~10,000×
/// clam_compress::hamming_to_c  → XOR-diff on 102B Base17   ~20×
/// hamming_distance_raw (2KB)   → NOT REPLACED (Layer 3)    1× (same)
/// grb_mxm inner product        → NOT REPLACED (semiring)   N/A
/// ```
pub struct SimdGuidance;

impl SimdGuidance {
    /// Which ndarray hot paths bgz17 replaces.
    pub fn replaced_paths() -> &'static [(&'static str, &'static str, &'static str)] {
        &[
            ("cascade::query Stroke 1", "scent byte XOR (1B)", "~128×"),
            ("cascade::query Stroke 2", "palette lookup (3B)", "~10,000×"),
            ("clam_search::rho_nn", "palette on 3B edges", "~10,000×"),
            ("clam_compress::hamming_to_compressed", "XOR-diff on 102B Base17", "~20×"),
        ]
    }

    /// Which ndarray hot paths remain unchanged (fallback).
    pub fn unchanged_paths() -> &'static [(&'static str, &'static str)] {
        &[
            ("bitwise::hamming_distance_raw", "Still needed for Layer 3 exact fallback"),
            ("grb_mxm inner product", "Needs BitVec XOR — bgz17 has no vector to XOR"),
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_route_knn_with_palette() {
        let route = route_query(1000, false, &SearchType::Knn { k: 10 });
        assert_eq!(route, Route::Bgz17Layered);
    }

    #[test]
    fn test_route_knn_bootstrap() {
        let route = route_query(20, false, &SearchType::Knn { k: 10 });
        assert_eq!(route, Route::NdarrayCascade);
    }

    #[test]
    fn test_route_bfs_always_blasgraph() {
        let route = route_query(10000, true, &SearchType::Bfs { max_depth: 3 });
        assert_eq!(route, Route::BlasGraph);
    }

    #[test]
    fn test_route_sssp_always_blasgraph() {
        let route = route_query(10000, true, &SearchType::Sssp);
        assert_eq!(route, Route::BlasGraph);
    }

    #[test]
    fn test_route_semiring_always_blasgraph() {
        let route = route_query(10000, true, &SearchType::SemiringOp);
        assert_eq!(route, Route::BlasGraph);
    }

    #[test]
    fn test_route_anomaly_with_tree() {
        let route = route_query(10000, true, &SearchType::AnomalyDetect);
        assert_eq!(route, Route::ClamChaoda);
    }

    #[test]
    fn test_route_anomaly_without_tree() {
        let route = route_query(10000, false, &SearchType::AnomalyDetect);
        assert_eq!(route, Route::NdarrayCascade);
    }

    #[test]
    fn test_fallback_high_anomaly() {
        let signals = FallbackSignals {
            scent_prune_rate: 0.90,
            palette_collision_rate: 0.02,
            clam_tree_depth: 8,
            max_anomaly_score: 0.85,
        };
        assert_eq!(signals.should_escalate(), Some(Route::NdarrayCascade));
        assert_eq!(signals.minimum_precision(), crate::Precision::Base);
    }

    #[test]
    fn test_fallback_shallow_tree() {
        let signals = FallbackSignals {
            scent_prune_rate: 0.90,
            palette_collision_rate: 0.02,
            clam_tree_depth: 2,
            max_anomaly_score: 0.10,
        };
        assert_eq!(signals.should_escalate(), Some(Route::NdarrayCascade));
    }

    #[test]
    fn test_fallback_normal() {
        let signals = FallbackSignals {
            scent_prune_rate: 0.85,
            palette_collision_rate: 0.03,
            clam_tree_depth: 7,
            max_anomaly_score: 0.20,
        };
        assert_eq!(signals.should_escalate(), None);
        assert_eq!(signals.minimum_precision(), crate::Precision::Palette);
    }

    #[test]
    fn test_fallback_palette_collision() {
        let signals = FallbackSignals {
            scent_prune_rate: 0.85,
            palette_collision_rate: 0.15,
            clam_tree_depth: 7,
            max_anomaly_score: 0.20,
        };
        // Shouldn't escalate system — just use higher precision
        assert_eq!(signals.should_escalate(), None);
        assert_eq!(signals.minimum_precision(), crate::Precision::Base);
    }
}
