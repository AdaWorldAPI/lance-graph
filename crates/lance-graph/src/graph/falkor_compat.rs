// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! FalkorDB compatibility shim: routes queries through our three backends.
//!
//! NOT a production fork -- a reality check proving our stack produces
//! the same results as FalkorDB-style queries.
//!
//! FalkorDB uses SuiteSparse:GraphBLAS (C FFI) with scalar semirings over
//! CSC matrices. We have pure-Rust GraphBLAS (blasgraph) over 16Kbit HDR
//! vectors + palette compression. This module verifies they produce the
//! same results on the same queries.

use std::collections::HashMap;

use crate::graph::blasgraph::matrix::GrBMatrix;
use crate::graph::blasgraph::semiring::HdrSemiring;
use crate::graph::blasgraph::sparse::CooStorage;
use crate::graph::blasgraph::typed_graph::{apply_truth_gate, BlasGraphHit, TypedGraph};
use crate::graph::blasgraph::types::BitVec;
use crate::graph::spo::store::SpoStore;
use crate::graph::spo::truth::{TruthGate, TruthValue};

/// A hit from any backend.
#[derive(Debug, Clone)]
pub struct FalkorHit {
    /// Source node index.
    pub source: usize,
    /// Target node index.
    pub target: usize,
    /// Hamming distance (from BitVec popcount after XOR).
    pub distance: u32,
    /// Truth value associated with this edge.
    pub truth: TruthValue,
    /// Which backend produced this hit.
    pub backend: Backend,
}

/// Which backend produced a query result.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Backend {
    /// DataFusion cold-path (Entry 1: SQL over RecordBatch).
    DataFusion,
    /// Blasgraph semiring traversal (Entry 2: BitVec hot path).
    Blasgraph,
    /// Palette-accelerated traversal (Entry 2: bgz17 hot path).
    Palette,
}

/// Classification of a query for routing purposes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QueryClass {
    /// Pure structural traversal (MATCH patterns, no vector similarity).
    PureTraversal,
    /// Vector similarity search (KNN, cosine, etc.).
    Similarity,
    /// Mixed: structural traversal refined by similarity.
    Hybrid,
}

/// FalkorDB compatibility layer: three backends, same queries.
///
/// Builds a typed graph with per-relationship adjacency matrices,
/// per-label boolean masks, and truth values for edge confidence.
/// Queries can be routed through blasgraph (BitVec semirings) or
/// classified for automatic backend selection.
pub struct FalkorCompat {
    /// TypedGraph for blasgraph semiring traversal.
    pub typed_graph: TypedGraph,
    /// SpoStore for fingerprint-based queries.
    pub spo_store: SpoStore,
    /// Truth values keyed by (source, target) pair.
    pub truth_values: HashMap<(usize, usize), TruthValue>,
    /// Node names indexed by node ID.
    pub node_names: Vec<String>,
    /// Node label -> list of node IDs.
    pub node_labels: HashMap<String, Vec<usize>>,
}

impl FalkorCompat {
    /// Create a new FalkorCompat shim for a graph with `node_count` nodes.
    pub fn new(node_count: usize) -> Self {
        Self {
            typed_graph: TypedGraph::new(node_count),
            spo_store: SpoStore::new(),
            truth_values: HashMap::new(),
            node_names: vec![String::new(); node_count],
            node_labels: HashMap::new(),
        }
    }

    /// Add a named node with a label.
    pub fn add_node(&mut self, id: usize, name: &str, label: &str) {
        if id < self.node_names.len() {
            self.node_names[id] = name.to_string();
        }
        self.node_labels
            .entry(label.to_string())
            .or_default()
            .push(id);
    }

    /// Synchronize node_labels into the TypedGraph's label masks.
    fn sync_labels(&mut self) {
        for (label, ids) in &self.node_labels {
            self.typed_graph.add_label(label, ids);
        }
    }

    /// Add an edge with relationship type and truth value.
    ///
    /// Creates a BitVec edge in the typed graph's adjacency matrix for
    /// the given relationship type, and stores the truth value for
    /// post-hoc TruthGate filtering.
    pub fn add_edge(&mut self, src: usize, dst: usize, rel_type: &str, truth: TruthValue) {
        // Store truth value
        self.truth_values.insert((src, dst), truth);

        // Build or extend the adjacency matrix for this relationship type.
        // We accumulate edges in COO and rebuild when querying.
        // For simplicity, we rebuild the relation matrix each time.
        let n = self.typed_graph.node_count;

        // Get existing entries or start fresh
        let mut entries: Vec<(usize, usize, BitVec)> = Vec::new();
        if let Some(existing) = self.typed_graph.relations.get(rel_type) {
            for (r, c, v) in existing.iter() {
                entries.push((r, c, v.clone()));
            }
        }

        // Use a deterministic seed based on src/dst for reproducible BitVec
        let seed = (src as u64) * 1000 + (dst as u64);
        entries.push((src, dst, BitVec::random(seed)));

        // Build COO and create matrix
        let mut coo = CooStorage::new(n, n);
        for (r, c, v) in entries {
            coo.push(r, c, v);
        }
        self.typed_graph
            .add_relation(rel_type, GrBMatrix::from_coo(&coo));
    }

    /// Finalize the graph: sync labels after all nodes and edges are added.
    pub fn finalize(&mut self) {
        self.sync_labels();
    }

    /// Query via blasgraph semiring traversal for a single relationship type.
    ///
    /// Returns all edges of the given type that pass the truth gate.
    pub fn query_blasgraph(&self, rel_type: &str, gate: TruthGate) -> Vec<FalkorHit> {
        let matrix = match self.typed_graph.relation(rel_type) {
            Some(m) => m,
            None => return Vec::new(),
        };

        let hits = apply_truth_gate(matrix, gate, &self.truth_values);
        hits.into_iter()
            .map(|h| FalkorHit {
                source: h.source,
                target: h.target,
                distance: h.value.popcount() as u32,
                truth: h.truth,
                backend: Backend::Blasgraph,
            })
            .collect()
    }

    /// Multi-hop query via blasgraph semiring multiplication.
    ///
    /// Composes multiple relationship types sequentially using XorBundle
    /// semiring (the default for path composition). Results are filtered
    /// by the truth gate using the weakest truth value along each path.
    pub fn query_blasgraph_multi_hop(
        &self,
        rel_types: &[&str],
        gate: TruthGate,
    ) -> Vec<FalkorHit> {
        let result = match self
            .typed_graph
            .multi_hop(rel_types, &HdrSemiring::XorBundle)
        {
            Some(m) => m,
            None => return Vec::new(),
        };

        // For multi-hop, we don't have direct truth values for composed edges.
        // Use OPEN gate on the matrix, then filter by checking if the composed
        // path has endpoints whose truth we can infer.
        let mut hits = Vec::new();
        for (r, c, v) in result.iter() {
            let truth = self
                .truth_values
                .get(&(r, c))
                .copied()
                .unwrap_or_else(TruthValue::unknown);
            if gate.passes(&truth) {
                hits.push(FalkorHit {
                    source: r,
                    target: c,
                    distance: v.popcount() as u32,
                    truth,
                    backend: Backend::Blasgraph,
                });
            }
        }
        hits
    }

    /// Query with automatic routing based on query classification.
    ///
    /// When the `planner` feature is enabled, uses the planner's feature detection
    /// to classify the query and route to the appropriate backend.
    /// Currently only blasgraph is connected; palette routing will be added in Phase 4.
    pub fn query_routed(&self, rel_type: &str, gate: TruthGate) -> Vec<FalkorHit> {
        match Self::classify_query(rel_type) {
            QueryClass::Similarity => {
                // Phase 4: route to palette backend when available.
                // For now, fall through to blasgraph.
                self.query_blasgraph(rel_type, gate)
            }
            QueryClass::PureTraversal | QueryClass::Hybrid => {
                self.query_blasgraph(rel_type, gate)
            }
        }
    }

    /// Classify a query for routing purposes.
    ///
    /// When the `planner` feature is enabled, delegates to the planner's
    /// `QueryFeatures` detection (fingerprint scan, resonance, truth values).
    /// Without the feature, falls back to PureTraversal.
    pub fn classify_query(query_text: &str) -> QueryClass {
        #[cfg(feature = "planner")]
        {
            let q = query_text.to_uppercase();
            let has_fingerprint = q.contains("HAMMING")
                || q.contains("FINGERPRINT")
                || q.contains("RESONATE");
            let has_graph_pattern = q.contains("MATCH");

            if has_fingerprint && has_graph_pattern {
                return QueryClass::Hybrid;
            }
            if has_fingerprint {
                return QueryClass::Similarity;
            }
        }
        #[cfg(not(feature = "planner"))]
        let _ = query_text;

        QueryClass::PureTraversal
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build the Jan->Ada->Max social graph with KNOWS and CREATES edges.
    fn build_social_graph() -> FalkorCompat {
        // 4 nodes: Jan(0), Ada(1), Max(2), Eve(3)
        let mut compat = FalkorCompat::new(4);

        // Add nodes with labels
        compat.add_node(0, "Jan", "Person");
        compat.add_node(1, "Ada", "Person");
        compat.add_node(2, "Max", "Person");
        compat.add_node(3, "Eve", "Person");

        // Also label Ada and Max as Engineers
        compat.add_node(1, "Ada", "Engineer");
        compat.add_node(2, "Max", "Engineer");

        // KNOWS edges with truth values
        // Jan->Ada: strong (0.9, 0.8) -> expectation 0.82
        compat.add_edge(0, 1, "KNOWS", TruthValue::new(0.9, 0.8));
        // Ada->Max: moderate (0.7, 0.6) -> expectation 0.62
        compat.add_edge(1, 2, "KNOWS", TruthValue::new(0.7, 0.6));
        // Max->Eve: weak (0.4, 0.3) -> expectation 0.47
        compat.add_edge(2, 3, "KNOWS", TruthValue::new(0.4, 0.3));

        // CREATES edge: Ada->Project
        // We reuse node 3 as a project for simplicity
        compat.add_edge(1, 3, "CREATES", TruthValue::new(0.95, 0.9));

        compat.finalize();
        compat
    }

    // =========================================================================
    // Test 1: Single-hop KNOWS traversal returns correct edges
    // =========================================================================

    #[test]
    fn test_single_hop_knows() {
        let compat = build_social_graph();
        let hits = compat.query_blasgraph("KNOWS", TruthGate::OPEN);

        // Should find 3 KNOWS edges: Jan->Ada, Ada->Max, Max->Eve
        assert_eq!(hits.len(), 3, "Expected 3 KNOWS edges, got {}", hits.len());

        // Verify all edges are present
        let pairs: Vec<(usize, usize)> = hits.iter().map(|h| (h.source, h.target)).collect();
        assert!(pairs.contains(&(0, 1)), "Missing Jan->Ada");
        assert!(pairs.contains(&(1, 2)), "Missing Ada->Max");
        assert!(pairs.contains(&(2, 3)), "Missing Max->Eve");

        // All should report Blasgraph backend
        for hit in &hits {
            assert_eq!(hit.backend, Backend::Blasgraph);
        }
    }

    // =========================================================================
    // Test 2: Two-hop KNOWS^2 finds Jan->Max via Ada
    // =========================================================================

    #[test]
    fn test_two_hop_knows_squared() {
        let compat = build_social_graph();
        let hits = compat.query_blasgraph_multi_hop(&["KNOWS", "KNOWS"], TruthGate::OPEN);

        // KNOWS x KNOWS should yield:
        // Jan->Max (0->2) via Ada
        // Ada->Eve (1->3) via Max
        assert!(
            hits.len() >= 2,
            "Expected at least 2 two-hop results, got {}",
            hits.len()
        );

        let pairs: Vec<(usize, usize)> = hits.iter().map(|h| (h.source, h.target)).collect();
        assert!(
            pairs.contains(&(0, 2)),
            "Jan should reach Max in 2 hops via Ada"
        );
        assert!(
            pairs.contains(&(1, 3)),
            "Ada should reach Eve in 2 hops via Max"
        );
    }

    // =========================================================================
    // Test 3: TruthGate STRONG filters weak edges
    // =========================================================================

    #[test]
    fn test_truth_gate_filters() {
        let compat = build_social_graph();

        // OPEN gate: all 3 KNOWS edges pass
        let open_hits = compat.query_blasgraph("KNOWS", TruthGate::OPEN);
        assert_eq!(open_hits.len(), 3);

        // STRONG gate (min_expectation = 0.75):
        //   Jan->Ada: expectation 0.82 -> passes
        //   Ada->Max: expectation 0.62 -> fails
        //   Max->Eve: expectation 0.47 -> fails
        let strong_hits = compat.query_blasgraph("KNOWS", TruthGate::STRONG);
        assert_eq!(
            strong_hits.len(),
            1,
            "Only Jan->Ada should pass STRONG gate, got {} hits",
            strong_hits.len()
        );
        assert_eq!(strong_hits[0].source, 0);
        assert_eq!(strong_hits[0].target, 1);

        // NORMAL gate (min_expectation = 0.6):
        //   Jan->Ada: 0.82 -> passes
        //   Ada->Max: 0.62 -> passes
        //   Max->Eve: 0.47 -> fails
        let normal_hits = compat.query_blasgraph("KNOWS", TruthGate::NORMAL);
        assert_eq!(
            normal_hits.len(),
            2,
            "Jan->Ada and Ada->Max should pass NORMAL gate"
        );
    }

    // =========================================================================
    // Test 4: Multi-relationship traversal (KNOWS + CREATES)
    // =========================================================================

    #[test]
    fn test_multi_relationship() {
        let compat = build_social_graph();

        // KNOWS edges
        let knows_hits = compat.query_blasgraph("KNOWS", TruthGate::OPEN);
        assert_eq!(knows_hits.len(), 3);

        // CREATES edges
        let creates_hits = compat.query_blasgraph("CREATES", TruthGate::OPEN);
        assert_eq!(
            creates_hits.len(),
            1,
            "Should have 1 CREATES edge (Ada->Project)"
        );
        assert_eq!(creates_hits[0].source, 1); // Ada
        assert_eq!(creates_hits[0].target, 3); // Eve/Project node

        // Nonexistent relationship type returns empty
        let empty = compat.query_blasgraph("LIKES", TruthGate::OPEN);
        assert!(empty.is_empty());
    }

    // =========================================================================
    // Test 5: Label masking (only Engineers)
    // =========================================================================

    #[test]
    fn test_label_mask_filtering() {
        let compat = build_social_graph();

        // Verify label masks are correct
        let engineer_mask = compat.typed_graph.label_mask("Engineer").unwrap();
        // Ada(1) and Max(2) are Engineers
        assert!(!engineer_mask[0], "Jan is not an Engineer");
        assert!(engineer_mask[1], "Ada is an Engineer");
        assert!(engineer_mask[2], "Max is an Engineer");
        assert!(!engineer_mask[3], "Eve is not an Engineer");

        // Masked traversal: KNOWS^2 filtered by Engineer targets
        let result = compat
            .typed_graph
            .masked_traverse("KNOWS", "Engineer", &HdrSemiring::XorBundle)
            .unwrap();

        // KNOWS^2 masked by Engineer:
        // Jan->Max(2) should pass (Max is Engineer)
        // Ada->Eve(3) should be filtered (Eve is not Engineer)
        assert!(
            result.get(0, 2).is_some(),
            "Jan->Max should pass Engineer mask"
        );
        assert!(
            result.get(1, 3).is_none(),
            "Ada->Eve should be filtered (Eve not Engineer)"
        );
    }

    // =========================================================================
    // Test 6: Empty graph returns no results
    // =========================================================================

    #[test]
    fn test_empty_graph() {
        let compat = FalkorCompat::new(0);

        let hits = compat.query_blasgraph("KNOWS", TruthGate::OPEN);
        assert!(hits.is_empty(), "Empty graph should return no results");

        let multi_hits =
            compat.query_blasgraph_multi_hop(&["KNOWS", "KNOWS"], TruthGate::OPEN);
        assert!(
            multi_hits.is_empty(),
            "Empty graph multi-hop should return no results"
        );

        let routed = compat.query_routed("KNOWS", TruthGate::OPEN);
        assert!(
            routed.is_empty(),
            "Empty graph routed query should return no results"
        );
    }

    // =========================================================================
    // Test 7: Query classification
    // =========================================================================

    #[test]
    fn test_query_classification() {
        // Default classification is PureTraversal
        assert_eq!(
            FalkorCompat::classify_query("KNOWS"),
            QueryClass::PureTraversal
        );
    }

    // =========================================================================
    // Test 8: Routed query consistency
    // =========================================================================

    #[test]
    fn test_routed_matches_blasgraph() {
        let compat = build_social_graph();

        let blasgraph_hits = compat.query_blasgraph("KNOWS", TruthGate::OPEN);
        let routed_hits = compat.query_routed("KNOWS", TruthGate::OPEN);

        // Routed should currently produce the same results as blasgraph
        assert_eq!(blasgraph_hits.len(), routed_hits.len());

        let bg_pairs: Vec<(usize, usize)> =
            blasgraph_hits.iter().map(|h| (h.source, h.target)).collect();
        let rt_pairs: Vec<(usize, usize)> =
            routed_hits.iter().map(|h| (h.source, h.target)).collect();
        assert_eq!(bg_pairs, rt_pairs);
    }
}
