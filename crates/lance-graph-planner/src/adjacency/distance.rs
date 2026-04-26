//! Batch fingerprint distance ON adjacent pairs.
//!
//! Not "scan all nodes" — scan only what's ADJACENT.
//! This is the key insight: resonance scanning happens on the adjacency graph,
//! not on a flat table.

use super::csr::AdjacencyStore;
#[allow(unused_imports)] // intended for batch distance computation wiring
use super::batch::AdjacencyBatch;

/// Result of adjacent fingerprint distance scan.
#[derive(Debug, Clone)]
pub struct AdjacentDistanceResult {
    /// (source_node, target_node, hamming_distance)
    pub matches: Vec<(u64, u64, u32)>,
}

/// Compute Hamming distance between a query fingerprint and all adjacent nodes' fingerprints.
///
/// For each source in the batch, look up its adjacent nodes' fingerprints
/// from the node property store, compute Hamming distance, filter by threshold.
///
/// In production, this calls ndarray's AVX-512 VPOPCNTDQ kernel.
pub fn adjacent_fingerprint_distance(
    store: &AdjacencyStore,
    source_ids: &[u64],
    query_fp: &[u64],
    node_fingerprints: &[Vec<u64>], // node_id → fingerprint
    threshold: u32,
) -> AdjacentDistanceResult {
    let batch = store.batch_adjacent(source_ids);
    let mut matches = Vec::new();

    for i in 0..batch.num_sources() {
        let source = batch.source_ids[i];
        let targets = batch.targets_for(i);

        for &target in targets {
            if let Some(target_fp) = node_fingerprints.get(target as usize) {
                let distance = hamming_distance(query_fp, target_fp);
                if distance <= threshold {
                    matches.push((source, target, distance));
                }
            }
        }
    }

    // Sort by distance
    matches.sort_by_key(|&(_, _, d)| d);

    AdjacentDistanceResult { matches }
}

/// Hamming distance between two fingerprints.
/// In production, this is an ndarray SIMD kernel.
#[inline]
fn hamming_distance(a: &[u64], b: &[u64]) -> u32 {
    a.iter().zip(b.iter())
        .map(|(x, y)| (x ^ y).count_ones())
        .sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_adjacent_fingerprint_distance() {
        let store = AdjacencyStore::from_edges("KNOWS".into(), 4, &[
            (0, 1), (0, 2), (0, 3),
        ]);

        // Node fingerprints (simple 2-word fingerprints for testing)
        let query_fp = vec![0xFF00FF00u64, 0x00FF00FFu64];
        let node_fps = vec![
            vec![0u64, 0u64],                     // node 0
            vec![0xFF00FF00u64, 0x00FF00FFu64],   // node 1: exact match
            vec![0xFF00FF01u64, 0x00FF00FFu64],   // node 2: 1 bit off
            vec![0x00000000u64, 0xFFFFFFFFu64],   // node 3: very different
        ];

        let result = adjacent_fingerprint_distance(
            &store, &[0], &query_fp, &node_fps, 5,
        );

        // Node 1: distance 0 (exact match) — should be included
        // Node 2: distance 1 — should be included
        // Node 3: very different — should be excluded
        assert_eq!(result.matches.len(), 2);
        assert_eq!(result.matches[0], (0, 1, 0)); // Exact match first
        assert_eq!(result.matches[1], (0, 2, 1)); // 1 bit off second
    }
}
