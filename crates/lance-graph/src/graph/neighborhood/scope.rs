// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Scope builder: construct neighborhood vectors for a scope of up to 10K nodes.
//!
//! A **scope** is a working set of up to 10,000 nodes. Each node's
//! **neighborhood vector** records its ZeckF64 edge to every other node
//! in the scope. Position in the vector IS the address — no separate ID column.
//!
//! Storage options (per node):
//!   - Scent only:      `[u8; N]`  — 10 KB/node, ρ ≈ 0.94
//!   - Scent + fine:    `[u16; N]` — 20 KB/node, ρ ≈ 0.96
//!   - Full progressive: `[u64; N]` — 80 KB/node, ρ ≈ 0.98

use crate::graph::blasgraph::types::BitVec;

use super::zeckf64;

/// Maximum number of nodes in a single scope.
pub const MAX_SCOPE_SIZE: usize = 10_000;

/// A node's neighborhood vector: one ZeckF64 per scope neighbor.
///
/// `entries[i]` = ZeckF64 edge from this node to scope node `i`.
/// `0x0000000000000000` = no edge (self-edge or unpopulated slot).
#[derive(Clone)]
pub struct NeighborhoodVector {
    /// The global node ID for this node.
    pub node_id: u64,
    /// ZeckF64 entries, one per scope position.
    pub entries: Vec<u64>,
}

impl NeighborhoodVector {
    /// Create a new empty neighborhood vector.
    pub fn new(node_id: u64, scope_size: usize) -> Self {
        Self {
            node_id,
            entries: vec![0u64; scope_size],
        }
    }

    /// Number of non-zero (populated) edges.
    pub fn edge_count(&self) -> usize {
        self.entries.iter().filter(|&&e| e != 0).count()
    }

    /// Extract the scent column: byte 0 of each entry.
    pub fn scent_vector(&self) -> Vec<u8> {
        self.entries.iter().map(|&e| e as u8).collect()
    }

    /// Extract the resolution column: byte 1 of each entry.
    pub fn resolution_vector(&self) -> Vec<u8> {
        self.entries.iter().map(|&e| (e >> 8) as u8).collect()
    }
}

/// Maps scope positions to global node IDs.
#[derive(Clone)]
pub struct ScopeMap {
    /// Scope identifier.
    pub scope_id: u64,
    /// `node_ids[i]` = global node ID at scope position `i`.
    pub node_ids: Vec<u64>,
}

impl ScopeMap {
    /// Create a new scope map.
    pub fn new(scope_id: u64, node_ids: Vec<u64>) -> Self {
        assert!(
            node_ids.len() <= MAX_SCOPE_SIZE,
            "Scope size {} exceeds maximum {}",
            node_ids.len(),
            MAX_SCOPE_SIZE
        );
        Self { scope_id, node_ids }
    }

    /// Number of nodes in the scope.
    pub fn len(&self) -> usize {
        self.node_ids.len()
    }

    /// Whether the scope is empty.
    pub fn is_empty(&self) -> bool {
        self.node_ids.is_empty()
    }

    /// Look up the scope position of a global node ID.
    pub fn position_of(&self, global_id: u64) -> Option<usize> {
        self.node_ids.iter().position(|&id| id == global_id)
    }
}

/// Builds neighborhood vectors for all nodes in a scope.
pub struct ScopeBuilder;

impl ScopeBuilder {
    /// Build neighborhood vectors for a scope of nodes.
    ///
    /// Takes parallel slices: `node_ids[i]` has SPO planes `planes[i]`.
    /// Each plane triple is `(subject, predicate, object)` as `BitVec`.
    ///
    /// Returns `(scope_map, neighborhoods)` where `neighborhoods[i]`
    /// is the neighborhood vector for `node_ids[i]`.
    ///
    /// Cost: O(N²) pairwise comparisons where N = node count.
    /// For N = 10K this is 100M comparisons — takes ~1 second.
    pub fn build(
        scope_id: u64,
        node_ids: &[u64],
        planes: &[(BitVec, BitVec, BitVec)],
    ) -> (ScopeMap, Vec<NeighborhoodVector>) {
        assert_eq!(
            node_ids.len(),
            planes.len(),
            "node_ids and planes must have same length"
        );
        assert!(
            node_ids.len() <= MAX_SCOPE_SIZE,
            "Scope size {} exceeds maximum {}",
            node_ids.len(),
            MAX_SCOPE_SIZE
        );

        let n = node_ids.len();
        let scope_map = ScopeMap::new(scope_id, node_ids.to_vec());

        let mut neighborhoods: Vec<NeighborhoodVector> = node_ids
            .iter()
            .map(|&id| NeighborhoodVector::new(id, n))
            .collect();

        // Compute pairwise ZeckF64 edges. Symmetric: compute once, store both.
        for i in 0..n {
            for j in (i + 1)..n {
                let edge = zeckf64(
                    (&planes[i].0, &planes[i].1, &planes[i].2),
                    (&planes[j].0, &planes[j].1, &planes[j].2),
                );
                neighborhoods[i].entries[j] = edge;
                neighborhoods[j].entries[i] = edge;
            }
        }

        (scope_map, neighborhoods)
    }

    /// Build neighborhood vectors using pre-computed Hamming distances.
    ///
    /// `distances[i][j]` = `(ds, dp, d_o)` for node pair `(i, j)`.
    /// Only the upper triangle (`j > i`) needs to be populated.
    pub fn build_from_distances(
        scope_id: u64,
        node_ids: &[u64],
        distances: &[Vec<(u32, u32, u32)>],
    ) -> (ScopeMap, Vec<NeighborhoodVector>) {
        let n = node_ids.len();
        let scope_map = ScopeMap::new(scope_id, node_ids.to_vec());

        let mut neighborhoods: Vec<NeighborhoodVector> = node_ids
            .iter()
            .map(|&id| NeighborhoodVector::new(id, n))
            .collect();

        for i in 0..n {
            for j in (i + 1)..n {
                let (ds, dp, d_o) = distances[i][j];
                let edge = super::zeckf64::zeckf64_from_distances(ds, dp, d_o);
                neighborhoods[i].entries[j] = edge;
                neighborhoods[j].entries[i] = edge;
            }
        }

        (scope_map, neighborhoods)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::blasgraph::types::BitVec;

    fn random_triple(seed: u64) -> (BitVec, BitVec, BitVec) {
        (
            BitVec::random(seed * 3),
            BitVec::random(seed * 3 + 1),
            BitVec::random(seed * 3 + 2),
        )
    }

    #[test]
    fn test_scope_builder_basic() {
        let node_ids: Vec<u64> = (0..5).collect();
        let planes: Vec<_> = (0..5).map(|i| random_triple(i + 100)).collect();

        let (scope, neighborhoods) = ScopeBuilder::build(1, &node_ids, &planes);

        assert_eq!(scope.len(), 5);
        assert_eq!(neighborhoods.len(), 5);

        // Self-edges should be zero
        for (i, nv) in neighborhoods.iter().enumerate() {
            assert_eq!(nv.entries[i], 0, "Self-edge for node {} should be 0", i);
        }

        // Non-self edges should be non-zero (random triples are different)
        for nv in &neighborhoods {
            assert!(nv.edge_count() > 0, "Should have at least one edge");
        }
    }

    #[test]
    fn test_scope_symmetry() {
        let node_ids: Vec<u64> = (0..10).collect();
        let planes: Vec<_> = (0..10).map(|i| random_triple(i + 200)).collect();

        let (_, neighborhoods) = ScopeBuilder::build(1, &node_ids, &planes);

        // edge(i→j) == edge(j→i)
        for i in 0..10 {
            for j in (i + 1)..10 {
                assert_eq!(
                    neighborhoods[i].entries[j], neighborhoods[j].entries[i],
                    "Asymmetric edge between {} and {}",
                    i, j
                );
            }
        }
    }

    #[test]
    fn test_scent_vector_extraction() {
        let node_ids: Vec<u64> = (0..3).collect();
        let planes: Vec<_> = (0..3).map(|i| random_triple(i + 300)).collect();

        let (_, neighborhoods) = ScopeBuilder::build(1, &node_ids, &planes);

        let scent = neighborhoods[0].scent_vector();
        assert_eq!(scent.len(), 3);
        // scent[0] should be 0 (self-edge)
        assert_eq!(scent[0], 0);
    }

    #[test]
    fn test_scope_map_lookup() {
        let node_ids = vec![100, 200, 300, 400, 500];
        let scope = ScopeMap::new(1, node_ids);

        assert_eq!(scope.position_of(300), Some(2));
        assert_eq!(scope.position_of(999), None);
        assert_eq!(scope.len(), 5);
    }
}
