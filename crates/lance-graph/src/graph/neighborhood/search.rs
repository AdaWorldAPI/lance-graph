// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Heel / Hip / Twig / Leaf — progressive neighborhood vector search.
//!
//! ## Cascade stages
//!
//! | Stage | Name | Operation                                    | Budget    |
//! |-------|------|----------------------------------------------|-----------|
//! | 1     | HEEL | L1 on MY scent vector (byte 0 only)          | 1 vector  |
//! | 2     | HIP  | L1 on survivors' scent vectors (2nd hop)     | ~50 vecs  |
//! | 3     | TWIG | L1 on 2nd-hop survivors' scent (3rd hop)     | ~50 vecs  |
//! | 4     | LEAF | Load cold planes, exact verification         | ~50 cold  |
//!
//! Total: ~1.2 MB loaded, ~1.1 ms, ~200K nodes explored across 3 hops.

use std::collections::HashSet;

use super::scope::NeighborhoodVector;
use super::zeckf64::zeckf64_scent_distance;

/// The "ideal" ZeckF64 edge: all 7 close bits set, all quantiles = 0.
/// This represents "identical triples". We rank by L1 distance to this ideal.
const IDEAL_EDGE: u64 = 0x7F; // byte 0 = 0111_1111, bytes 1-7 = 0

/// Rank a ZeckF64 edge by how close it is to ideal (identical).
/// Lower score = more similar.
/// - Scent: penalizes missing close bits (ideal = 0x7F, all set)
/// - Resolution: penalizes high quantile values (ideal = 0)
#[inline]
fn rank_edge(edge: u64) -> u32 {
    let mut dist = 0u32;
    for i in 0..8 {
        let actual = ((edge >> (i * 8)) & 0xFF) as i16;
        let ideal = ((IDEAL_EDGE >> (i * 8)) & 0xFF) as i16;
        dist += (actual - ideal).unsigned_abs() as u32;
    }
    dist
}

/// Rank using scent byte only: distance from ideal scent (0x7F).
#[inline]
fn rank_scent(edge: u64) -> u32 {
    let s = (edge & 0x7F) as u32; // mask out sign bit
    127u32.saturating_sub(s)
}

/// A search hit at any stage of the cascade.
#[derive(Debug, Clone)]
pub struct HeelResult {
    /// Scope position of the hit.
    pub position: usize,
    /// Distance metric used for ranking.
    pub distance: u32,
    /// Which hop discovered this node (0 = HEEL, 1 = HIP, 2 = TWIG).
    pub hop: u8,
}

/// Configuration for the search cascade.
#[derive(Debug, Clone)]
pub struct SearchConfig {
    /// Number of survivors to keep at each stage.
    pub k: usize,
    /// Use scent-only distance (byte 0) for HEEL/HIP/TWIG stages.
    /// If false, uses full ZeckF64 L1 distance (all 8 bytes).
    pub scent_only: bool,
}

impl Default for SearchConfig {
    fn default() -> Self {
        Self {
            k: 50,
            scent_only: true,
        }
    }
}

/// The search cascade: HEEL → HIP → TWIG → (LEAF is external).
///
/// Operates entirely on in-memory neighborhood vectors. LEAF stage
/// requires loading cold data from Lance and is handled externally.
pub struct SearchCascade;

impl SearchCascade {
    /// **HEEL** — First stage: find top-K from my own neighborhood vector.
    ///
    /// Scans the query node's neighborhood vector and returns the `k`
    /// closest non-zero entries by scent distance.
    ///
    /// Cost: 1 vector loaded, N comparisons (N = scope size).
    pub fn heel(query: &NeighborhoodVector, config: &SearchConfig) -> Vec<HeelResult> {
        let mut hits: Vec<HeelResult> = query
            .entries
            .iter()
            .enumerate()
            .filter(|(_, &e)| e != 0)
            .map(|(i, &e)| HeelResult {
                position: i,
                distance: if config.scent_only {
                    rank_scent(e)
                } else {
                    rank_edge(e)
                },
                hop: 0,
            })
            .collect();

        hits.sort_by_key(|h| h.distance);
        hits.truncate(config.k);
        hits
    }

    /// **HIP** — Second stage: expand from HEEL survivors into their neighborhoods.
    ///
    /// For each survivor from HEEL, loads their neighborhood vector and finds
    /// new nodes not yet seen. Each survivor opens a 90-degree window into
    /// different parts of the graph.
    ///
    /// Cost: `survivors.len()` vectors loaded, up to `survivors.len() × scope_size`
    /// comparisons.
    pub fn hip(
        heel_survivors: &[HeelResult],
        neighborhoods: &[NeighborhoodVector],
        config: &SearchConfig,
    ) -> Vec<HeelResult> {
        let mut seen: HashSet<usize> = heel_survivors.iter().map(|h| h.position).collect();
        let mut hits = Vec::new();

        for survivor in heel_survivors {
            let nv = &neighborhoods[survivor.position];
            for (j, &edge) in nv.entries.iter().enumerate() {
                if edge == 0 || seen.contains(&j) {
                    continue;
                }
                seen.insert(j);
                hits.push(HeelResult {
                    position: j,
                    distance: if config.scent_only {
                        rank_scent(edge)
                    } else {
                        rank_edge(edge)
                    },
                    hop: 1,
                });
            }
        }

        hits.sort_by_key(|h| h.distance);
        hits.truncate(config.k);
        hits
    }

    /// **TWIG** — Third stage: expand from HIP survivors into their neighborhoods.
    ///
    /// Same operation as HIP, one more hop out. The `already_seen` set
    /// includes positions from both HEEL and HIP stages.
    ///
    /// Cost: same as HIP. Total explored after TWIG: ~200K unique nodes.
    pub fn twig(
        hip_survivors: &[HeelResult],
        already_seen: &HashSet<usize>,
        neighborhoods: &[NeighborhoodVector],
        config: &SearchConfig,
    ) -> Vec<HeelResult> {
        let mut seen = already_seen.clone();
        let mut hits = Vec::new();

        for survivor in hip_survivors {
            let nv = &neighborhoods[survivor.position];
            for (j, &edge) in nv.entries.iter().enumerate() {
                if edge == 0 || seen.contains(&j) {
                    continue;
                }
                seen.insert(j);
                hits.push(HeelResult {
                    position: j,
                    distance: if config.scent_only {
                        rank_scent(edge)
                    } else {
                        rank_edge(edge)
                    },
                    hop: 2,
                });
            }
        }

        hits.sort_by_key(|h| h.distance);
        hits.truncate(config.k);
        hits
    }

    /// Run the full 3-hop cascade: HEEL → HIP → TWIG.
    ///
    /// Returns the union of survivors from all three stages, deduplicated,
    /// sorted by distance. The caller then runs LEAF (cold verification)
    /// on these candidates.
    ///
    /// # Arguments
    /// * `query_position` — scope position of the query node
    /// * `neighborhoods` — all neighborhood vectors in the scope
    /// * `config` — search parameters
    pub fn search(
        query_position: usize,
        neighborhoods: &[NeighborhoodVector],
        config: &SearchConfig,
    ) -> Vec<HeelResult> {
        let query = &neighborhoods[query_position];

        // HEEL
        let heel_results = Self::heel(query, config);

        // HIP
        let hip_results = Self::hip(&heel_results, neighborhoods, config);

        // Collect all seen positions for TWIG dedup
        let mut seen: HashSet<usize> = HashSet::new();
        seen.insert(query_position);
        for h in &heel_results {
            seen.insert(h.position);
        }
        for h in &hip_results {
            seen.insert(h.position);
        }

        // TWIG
        let twig_results = Self::twig(&hip_results, &seen, neighborhoods, config);

        // Merge all survivors, deduplicate by position, sort by distance
        let mut all: Vec<HeelResult> = Vec::new();
        let mut final_seen: HashSet<usize> = HashSet::new();
        final_seen.insert(query_position);

        for result_set in [&heel_results, &hip_results, &twig_results] {
            for hit in result_set {
                if final_seen.insert(hit.position) {
                    all.push(hit.clone());
                }
            }
        }

        all.sort_by_key(|h| h.distance);
        all.truncate(config.k);
        all
    }

    /// **LEAF** — Verification stage (scent-level re-ranking).
    ///
    /// Given candidate positions from HEEL/HIP/TWIG and a query node's
    /// neighborhood vector, re-rank using full ZeckF64 L1 distance
    /// (all 8 bytes instead of just byte 0).
    ///
    /// This is the in-memory portion of LEAF. Full LEAF also loads cold
    /// data from `cognitive_nodes.lance` — that step is handled by the
    /// Lance integration layer.
    pub fn leaf_rerank(
        candidates: &[HeelResult],
        query_vector: &NeighborhoodVector,
        top_k: usize,
    ) -> Vec<HeelResult> {
        let mut reranked: Vec<HeelResult> = candidates
            .iter()
            .filter(|c| c.position < query_vector.entries.len())
            .map(|c| {
                let edge = query_vector.entries[c.position];
                HeelResult {
                    position: c.position,
                    distance: rank_edge(edge),
                    hop: c.hop,
                }
            })
            .collect();

        reranked.sort_by_key(|h| h.distance);
        reranked.truncate(top_k);
        reranked
    }

    /// Compute the scent-only L1 distance between two neighborhood vectors.
    ///
    /// Sums `|scent(a[i]) - scent(b[i])|` over all positions.
    /// This is the metric used for ANN search on the `scent` column.
    pub fn neighborhood_scent_l1(a: &NeighborhoodVector, b: &NeighborhoodVector) -> u64 {
        let len = a.entries.len().min(b.entries.len());
        let mut dist = 0u64;
        for i in 0..len {
            dist += zeckf64_scent_distance(a.entries[i], b.entries[i]) as u64;
        }
        dist
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::blasgraph::types::BitVec;
    use crate::graph::neighborhood::scope::ScopeBuilder;

    fn random_triple(seed: u64) -> (BitVec, BitVec, BitVec) {
        (
            BitVec::random(seed * 3),
            BitVec::random(seed * 3 + 1),
            BitVec::random(seed * 3 + 2),
        )
    }

    fn build_test_scope(n: usize) -> Vec<NeighborhoodVector> {
        let node_ids: Vec<u64> = (0..n as u64).collect();
        let planes: Vec<_> = (0..n).map(|i| random_triple(i as u64 + 1000)).collect();
        let (_, neighborhoods) = ScopeBuilder::build(1, &node_ids, &planes);
        neighborhoods
    }

    #[test]
    fn test_heel_returns_results() {
        let neighborhoods = build_test_scope(100);
        let config = SearchConfig {
            k: 10,
            scent_only: true,
        };

        let results = SearchCascade::heel(&neighborhoods[0], &config);
        assert!(!results.is_empty(), "HEEL should find neighbors");
        assert!(results.len() <= 10, "Should respect k=10");

        // Results should be sorted by distance
        for window in results.windows(2) {
            assert!(window[0].distance <= window[1].distance);
        }

        // All results should be hop 0
        for r in &results {
            assert_eq!(r.hop, 0);
        }
    }

    #[test]
    fn test_hip_expands_beyond_heel() {
        let neighborhoods = build_test_scope(100);
        let config = SearchConfig {
            k: 20,
            scent_only: true,
        };

        let heel = SearchCascade::heel(&neighborhoods[0], &config);
        let hip = SearchCascade::hip(&heel, &neighborhoods, &config);

        // HIP should find nodes NOT in HEEL results
        let heel_positions: HashSet<usize> = heel.iter().map(|h| h.position).collect();
        for h in &hip {
            assert!(
                !heel_positions.contains(&h.position),
                "HIP should not duplicate HEEL positions"
            );
            assert_eq!(h.hop, 1);
        }
    }

    #[test]
    fn test_full_cascade_explores_more_than_heel() {
        let neighborhoods = build_test_scope(200);
        let config = SearchConfig {
            k: 30,
            scent_only: true,
        };

        let heel_only = SearchCascade::heel(&neighborhoods[0], &config);
        let full = SearchCascade::search(0, &neighborhoods, &config);

        // Full cascade should explore at least as many unique nodes as HEEL
        let heel_positions: HashSet<usize> = heel_only.iter().map(|h| h.position).collect();
        let full_positions: HashSet<usize> = full.iter().map(|h| h.position).collect();
        assert!(
            full_positions.len() >= heel_positions.len(),
            "Full cascade should explore >= HEEL nodes"
        );
    }

    #[test]
    fn test_cascade_no_self_reference() {
        let neighborhoods = build_test_scope(50);
        let config = SearchConfig::default();

        let results = SearchCascade::search(0, &neighborhoods, &config);
        for r in &results {
            assert_ne!(r.position, 0, "Should not include query node itself");
        }
    }

    #[test]
    fn test_leaf_rerank_with_full_distance() {
        let neighborhoods = build_test_scope(50);
        let config = SearchConfig {
            k: 20,
            scent_only: true,
        };

        let candidates = SearchCascade::search(0, &neighborhoods, &config);
        let reranked = SearchCascade::leaf_rerank(&candidates, &neighborhoods[0], 10);

        assert!(reranked.len() <= 10);
        // Reranked should use full L1, so ordering may differ from scent-only
        for window in reranked.windows(2) {
            assert!(window[0].distance <= window[1].distance);
        }
    }

    #[test]
    fn test_neighborhood_scent_l1_self_is_zero() {
        let neighborhoods = build_test_scope(10);
        let d = SearchCascade::neighborhood_scent_l1(&neighborhoods[0], &neighborhoods[0]);
        assert_eq!(d, 0, "Self-distance should be 0");
    }

    #[test]
    fn test_neighborhood_scent_l1_symmetry() {
        let neighborhoods = build_test_scope(10);
        let d_ab = SearchCascade::neighborhood_scent_l1(&neighborhoods[0], &neighborhoods[1]);
        let d_ba = SearchCascade::neighborhood_scent_l1(&neighborhoods[1], &neighborhoods[0]);
        assert_eq!(d_ab, d_ba, "L1 distance should be symmetric");
    }
}
