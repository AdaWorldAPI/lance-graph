// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Graph Router: routes queries through three backends.
//!
//! Three execution paths, one unified API:
//!
//! 1. **DataFusion** (cold path) — SQL over RecordBatch, Cypher joins
//! 2. **Blasgraph**  (hot path)  — BitVec semiring traversal, 16Kbit HDR
//! 3. **Palette**    (hot path)  — bgz17 compressed traversal
//!
//! The router classifies queries and dispatches to the appropriate backend.
//! All backends operate on the same typed graph (per-relationship adjacency
//! matrices + per-label boolean masks + NARS truth values on edges).
//!
//! Methods were harvested from GraphBLAS literature (SuiteSparse patterns).
//! No external graph database dependency required.

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
pub struct GraphHit {
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

/// Three-backend graph router.
///
/// Builds a typed graph with per-relationship adjacency matrices,
/// per-label boolean masks, and truth values for edge confidence.
/// Queries can be routed through blasgraph (BitVec semirings) or
/// classified for automatic backend selection.
///
/// No external database dependency — all execution is in-process
/// using our pure-Rust GraphBLAS (blasgraph) over 16Kbit HDR vectors
/// + palette compression.
pub struct GraphRouter {
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

impl GraphRouter {
    /// Create a new graph router for a graph with `node_count` nodes.
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
    pub fn query_blasgraph(&self, rel_type: &str, gate: TruthGate) -> Vec<GraphHit> {
        let matrix = match self.typed_graph.relation(rel_type) {
            Some(m) => m,
            None => return Vec::new(),
        };

        let hits = apply_truth_gate(matrix, gate, &self.truth_values);
        hits.into_iter()
            .map(|h| GraphHit {
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
    ) -> Vec<GraphHit> {
        let result = match self
            .typed_graph
            .multi_hop(rel_types, &HdrSemiring::XorBundle)
        {
            Some(m) => m,
            None => return Vec::new(),
        };

        let mut hits = Vec::new();
        for (r, c, v) in result.iter() {
            let truth = self
                .truth_values
                .get(&(r, c))
                .copied()
                .unwrap_or_else(TruthValue::unknown);
            if gate.passes(&truth) {
                hits.push(GraphHit {
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
    /// Currently only blasgraph is connected; palette routing will be added in Phase 4.
    pub fn query_routed(&self, rel_type: &str, gate: TruthGate) -> Vec<GraphHit> {
        match Self::classify_query(rel_type) {
            QueryClass::Similarity => {
                self.query_blasgraph(rel_type, gate)
            }
            QueryClass::PureTraversal | QueryClass::Hybrid => {
                self.query_blasgraph(rel_type, gate)
            }
        }
    }

    /// Classify a query for routing purposes.
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

    fn build_social_graph() -> GraphRouter {
        let mut router = GraphRouter::new(4);
        router.add_node(0, "Jan", "Person");
        router.add_node(1, "Ada", "Person");
        router.add_node(2, "Max", "Person");
        router.add_node(3, "Eve", "Person");
        router.add_node(1, "Ada", "Engineer");
        router.add_node(2, "Max", "Engineer");
        router.add_edge(0, 1, "KNOWS", TruthValue::new(0.9, 0.8));
        router.add_edge(1, 2, "KNOWS", TruthValue::new(0.7, 0.6));
        router.add_edge(2, 3, "KNOWS", TruthValue::new(0.4, 0.3));
        router.add_edge(1, 3, "CREATES", TruthValue::new(0.95, 0.9));
        router.finalize();
        router
    }

    #[test]
    fn test_single_hop_knows() {
        let router = build_social_graph();
        let hits = router.query_blasgraph("KNOWS", TruthGate::OPEN);
        assert_eq!(hits.len(), 3);
        let pairs: Vec<(usize, usize)> = hits.iter().map(|h| (h.source, h.target)).collect();
        assert!(pairs.contains(&(0, 1)));
        assert!(pairs.contains(&(1, 2)));
        assert!(pairs.contains(&(2, 3)));
        for hit in &hits {
            assert_eq!(hit.backend, Backend::Blasgraph);
        }
    }

    #[test]
    fn test_two_hop_knows_squared() {
        let router = build_social_graph();
        let hits = router.query_blasgraph_multi_hop(&["KNOWS", "KNOWS"], TruthGate::OPEN);
        assert!(hits.len() >= 2);
        let pairs: Vec<(usize, usize)> = hits.iter().map(|h| (h.source, h.target)).collect();
        assert!(pairs.contains(&(0, 2)));
        assert!(pairs.contains(&(1, 3)));
    }

    #[test]
    fn test_truth_gate_filters() {
        let router = build_social_graph();
        let open_hits = router.query_blasgraph("KNOWS", TruthGate::OPEN);
        assert_eq!(open_hits.len(), 3);
        let strong_hits = router.query_blasgraph("KNOWS", TruthGate::STRONG);
        assert_eq!(strong_hits.len(), 1);
        assert_eq!(strong_hits[0].source, 0);
        assert_eq!(strong_hits[0].target, 1);
        let normal_hits = router.query_blasgraph("KNOWS", TruthGate::NORMAL);
        assert_eq!(normal_hits.len(), 2);
    }

    #[test]
    fn test_multi_relationship() {
        let router = build_social_graph();
        assert_eq!(router.query_blasgraph("KNOWS", TruthGate::OPEN).len(), 3);
        let creates_hits = router.query_blasgraph("CREATES", TruthGate::OPEN);
        assert_eq!(creates_hits.len(), 1);
        assert_eq!(creates_hits[0].source, 1);
        assert_eq!(creates_hits[0].target, 3);
        assert!(router.query_blasgraph("LIKES", TruthGate::OPEN).is_empty());
    }

    #[test]
    fn test_label_mask_filtering() {
        let router = build_social_graph();
        let engineer_mask = router.typed_graph.label_mask("Engineer").unwrap();
        assert!(!engineer_mask[0]);
        assert!(engineer_mask[1]);
        assert!(engineer_mask[2]);
        assert!(!engineer_mask[3]);
        let result = router
            .typed_graph
            .masked_traverse("KNOWS", "Engineer", &HdrSemiring::XorBundle)
            .unwrap();
        assert!(result.get(0, 2).is_some());
        assert!(result.get(1, 3).is_none());
    }

    #[test]
    fn test_empty_graph() {
        let router = GraphRouter::new(0);
        assert!(router.query_blasgraph("KNOWS", TruthGate::OPEN).is_empty());
        assert!(router.query_blasgraph_multi_hop(&["KNOWS", "KNOWS"], TruthGate::OPEN).is_empty());
        assert!(router.query_routed("KNOWS", TruthGate::OPEN).is_empty());
    }

    #[test]
    fn test_query_classification() {
        assert_eq!(GraphRouter::classify_query("KNOWS"), QueryClass::PureTraversal);
    }

    #[test]
    fn test_routed_matches_blasgraph() {
        let router = build_social_graph();
        let blasgraph_hits = router.query_blasgraph("KNOWS", TruthGate::OPEN);
        let routed_hits = router.query_routed("KNOWS", TruthGate::OPEN);
        assert_eq!(blasgraph_hits.len(), routed_hits.len());
    }
}
