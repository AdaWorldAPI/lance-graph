//! CSR/CSC compressed adjacency — the core storage.
//!
//! Column-major, vectorized, morsel-compatible.
//! Supports pluggable encodings (raw CSR today, VSA tomorrow).

use super::batch::AdjacencyBatch;
use super::properties::EdgeProperties;
use std::collections::HashMap;

/// The adjacency substrate. Everything else operates ON this.
#[derive(Debug, Clone)]
pub struct AdjacencyStore {
    /// Number of nodes in this store.
    pub num_nodes: u64,
    /// Number of edges.
    pub num_edges: u64,

    // === CSR (Compressed Sparse Row) for outgoing edges ===
    /// node_id → start offset into csr_targets.
    /// Length = num_nodes + 1 (sentinel at end).
    pub csr_offsets: Vec<u64>,
    /// Packed target node_ids (sorted per source for intersection).
    pub csr_targets: Vec<u64>,
    /// Edge ID for each adjacency entry (indexes into edge_properties).
    pub csr_edge_ids: Vec<u64>,

    // === CSC (Compressed Sparse Column) for incoming edges ===
    /// Same data, transposed — enables backward traversal.
    pub csc_offsets: Vec<u64>,
    pub csc_sources: Vec<u64>,
    pub csc_edge_ids: Vec<u64>,

    /// Edge properties stored columnar (not row-major!).
    /// NARS truth values are columns here: "truth_f", "truth_c", "truth_t".
    pub edge_properties: EdgeProperties,

    /// Relationship type for this adjacency store.
    /// One AdjacencyStore per relationship type (Kuzu pattern).
    pub rel_type: String,
}

impl AdjacencyStore {
    /// Create an empty adjacency store.
    pub fn new(rel_type: String, num_nodes: u64) -> Self {
        Self {
            num_nodes,
            num_edges: 0,
            csr_offsets: vec![0; num_nodes as usize + 1],
            csr_targets: Vec::new(),
            csr_edge_ids: Vec::new(),
            csc_offsets: vec![0; num_nodes as usize + 1],
            csc_sources: Vec::new(),
            csc_edge_ids: Vec::new(),
            edge_properties: EdgeProperties::new(),
            rel_type,
        }
    }

    /// Core primitive: get adjacent node_ids for a single source node (outgoing).
    #[inline]
    pub fn adjacent(&self, source: u64) -> &[u64] {
        let start = self.csr_offsets[source as usize] as usize;
        let end = self.csr_offsets[source as usize + 1] as usize;
        &self.csr_targets[start..end]
    }

    /// Core primitive: get adjacent node_ids for a single source node (incoming).
    #[inline]
    pub fn adjacent_incoming(&self, target: u64) -> &[u64] {
        let start = self.csc_offsets[target as usize] as usize;
        let end = self.csc_offsets[target as usize + 1] as usize;
        &self.csc_sources[start..end]
    }

    /// Edge IDs for outgoing edges from a source node.
    #[inline]
    pub fn edge_ids(&self, source: u64) -> &[u64] {
        let start = self.csr_offsets[source as usize] as usize;
        let end = self.csr_offsets[source as usize + 1] as usize;
        &self.csr_edge_ids[start..end]
    }

    /// Degree of a node (outgoing).
    #[inline]
    pub fn out_degree(&self, node: u64) -> u64 {
        self.csr_offsets[node as usize + 1] - self.csr_offsets[node as usize]
    }

    /// Degree of a node (incoming).
    #[inline]
    pub fn in_degree(&self, node: u64) -> u64 {
        self.csc_offsets[node as usize + 1] - self.csc_offsets[node as usize]
    }

    /// Kuzu's core primitive: batch-get all adjacent node_ids for a batch of sources.
    /// Returns a flat vector + offsets (like Arrow ListArray).
    /// This is WHERE vectorized traversal happens.
    pub fn batch_adjacent(&self, source_ids: &[u64]) -> AdjacencyBatch {
        let mut offsets = Vec::with_capacity(source_ids.len() + 1);
        let mut targets = Vec::new();
        let mut edge_ids = Vec::new();

        offsets.push(0u64);

        for &src in source_ids {
            let adj = self.adjacent(src);
            let eids = self.edge_ids(src);
            targets.extend_from_slice(adj);
            edge_ids.extend_from_slice(eids);
            offsets.push(targets.len() as u64);
        }

        AdjacencyBatch {
            source_ids: source_ids.to_vec(),
            offsets,
            targets,
            edge_ids,
        }
    }

    /// Build from edge list (src, dst) pairs.
    pub fn from_edges(rel_type: String, num_nodes: u64, edges: &[(u64, u64)]) -> Self {
        let num_edges = edges.len() as u64;

        // Build CSR
        let mut csr_offsets = vec![0u64; num_nodes as usize + 1];
        for &(src, _) in edges {
            csr_offsets[src as usize + 1] += 1;
        }
        // Prefix sum
        for i in 1..=num_nodes as usize {
            csr_offsets[i] += csr_offsets[i - 1];
        }

        let mut csr_targets = vec![0u64; edges.len()];
        let mut csr_edge_ids = vec![0u64; edges.len()];
        let mut positions = csr_offsets[..num_nodes as usize].to_vec();

        for (edge_id, &(src, dst)) in edges.iter().enumerate() {
            let pos = positions[src as usize] as usize;
            csr_targets[pos] = dst;
            csr_edge_ids[pos] = edge_id as u64;
            positions[src as usize] += 1;
        }

        // Sort each adjacency list by target (required for intersection)
        for node in 0..num_nodes as usize {
            let start = csr_offsets[node] as usize;
            let end = csr_offsets[node + 1] as usize;
            // Sort targets and edge_ids together
            let mut pairs: Vec<(u64, u64)> = (start..end)
                .map(|i| (csr_targets[i], csr_edge_ids[i]))
                .collect();
            pairs.sort_by_key(|p| p.0);
            for (i, (t, e)) in pairs.into_iter().enumerate() {
                csr_targets[start + i] = t;
                csr_edge_ids[start + i] = e;
            }
        }

        // Build CSC (transpose)
        let mut csc_offsets = vec![0u64; num_nodes as usize + 1];
        for &(_, dst) in edges {
            csc_offsets[dst as usize + 1] += 1;
        }
        for i in 1..=num_nodes as usize {
            csc_offsets[i] += csc_offsets[i - 1];
        }

        let mut csc_sources = vec![0u64; edges.len()];
        let mut csc_edge_ids = vec![0u64; edges.len()];
        let mut positions = csc_offsets[..num_nodes as usize].to_vec();

        for (edge_id, &(src, dst)) in edges.iter().enumerate() {
            let pos = positions[dst as usize] as usize;
            csc_sources[pos] = src;
            csc_edge_ids[pos] = edge_id as u64;
            positions[dst as usize] += 1;
        }

        // Sort CSC adjacency lists
        for node in 0..num_nodes as usize {
            let start = csc_offsets[node] as usize;
            let end = csc_offsets[node + 1] as usize;
            let mut pairs: Vec<(u64, u64)> = (start..end)
                .map(|i| (csc_sources[i], csc_edge_ids[i]))
                .collect();
            pairs.sort_by_key(|p| p.0);
            for (i, (s, e)) in pairs.into_iter().enumerate() {
                csc_sources[start + i] = s;
                csc_edge_ids[start + i] = e;
            }
        }

        Self {
            num_nodes,
            num_edges,
            csr_offsets,
            csr_targets,
            csr_edge_ids,
            csc_offsets,
            csc_sources,
            csc_edge_ids,
            edge_properties: EdgeProperties::with_capacity(edges.len()),
            rel_type,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_adjacency_store() {
        // Triangle: 0→1, 1→2, 2→0
        let store = AdjacencyStore::from_edges(
            "KNOWS".into(), 3,
            &[(0, 1), (1, 2), (2, 0)],
        );

        assert_eq!(store.adjacent(0), &[1]);
        assert_eq!(store.adjacent(1), &[2]);
        assert_eq!(store.adjacent(2), &[0]);
        assert_eq!(store.out_degree(0), 1);
    }

    #[test]
    fn test_batch_adjacent() {
        // Star: 0→1, 0→2, 0→3, 1→3
        let store = AdjacencyStore::from_edges(
            "LINKS".into(), 4,
            &[(0, 1), (0, 2), (0, 3), (1, 3)],
        );

        let batch = store.batch_adjacent(&[0, 1]);
        assert_eq!(batch.targets_for(0), &[1, 2, 3]); // 0's adjacency
        assert_eq!(batch.targets_for(1), &[3]);         // 1's adjacency
    }

    #[test]
    fn test_incoming_adjacency() {
        let store = AdjacencyStore::from_edges(
            "KNOWS".into(), 3,
            &[(0, 2), (1, 2)],
        );

        // Node 2 has incoming edges from 0 and 1
        let incoming = store.adjacent_incoming(2);
        assert_eq!(incoming.len(), 2);
        assert!(incoming.contains(&0));
        assert!(incoming.contains(&1));
    }
}
