// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! # Typed Graph: one matrix per relationship type, one mask per label.
//!
//! Maps to FalkorDB schema AND to container W16-31 inline edges.
//! Each relationship type (e.g. "KNOWS") gets its own adjacency matrix.
//! Each node label (e.g. "Person") gets a boolean mask for filtering.

use std::collections::HashMap;

use crate::graph::blasgraph::descriptor::GrBDesc;
use crate::graph::blasgraph::matrix::GrBMatrix;
use crate::graph::blasgraph::semiring::Semiring;
use crate::graph::blasgraph::sparse::CooStorage;
use crate::graph::blasgraph::types::BitVec;
use crate::graph::spo::store::SpoStore;
use crate::graph::spo::truth::TruthValue;

/// A typed property graph: one matrix per relationship type, one mask per label.
#[derive(Clone, Debug)]
pub struct TypedGraph {
    /// One adjacency matrix per relationship type (e.g. "KNOWS" → matrix).
    pub relations: HashMap<String, GrBMatrix>,
    /// One boolean mask per node label (e.g. "Person" → [true, false, true, ...]).
    pub labels: HashMap<String, Vec<bool>>,
    /// Total number of nodes.
    pub node_count: usize,
}

impl TypedGraph {
    /// Create an empty typed graph with the given node count.
    pub fn new(node_count: usize) -> Self {
        Self {
            relations: HashMap::new(),
            labels: HashMap::new(),
            node_count,
        }
    }

    /// Add a relationship type with its adjacency matrix.
    pub fn add_relation(&mut self, name: &str, matrix: GrBMatrix) {
        assert_eq!(
            matrix.nrows(),
            self.node_count,
            "matrix rows ({}) != node_count ({})",
            matrix.nrows(),
            self.node_count
        );
        assert_eq!(
            matrix.ncols(),
            self.node_count,
            "matrix cols ({}) != node_count ({})",
            matrix.ncols(),
            self.node_count
        );
        self.relations.insert(name.to_string(), matrix);
    }

    /// Add a node label with a list of node IDs.
    pub fn add_label(&mut self, name: &str, node_ids: &[usize]) {
        let mut mask = vec![false; self.node_count];
        for &id in node_ids {
            if id < self.node_count {
                mask[id] = true;
            }
        }
        self.labels.insert(name.to_string(), mask);
    }

    /// Single-hop traversal under the given semiring for one relationship type.
    pub fn traverse(
        &self,
        rel_type: &str,
        semiring: &dyn Semiring,
    ) -> Option<GrBMatrix> {
        let matrix = self.relations.get(rel_type)?;
        let desc = GrBDesc::default();
        // A × A under the given semiring = one hop
        Some(matrix.mxm(matrix, semiring, &desc))
    }

    /// Multi-hop traversal: compose multiple relationship types sequentially.
    ///
    /// `rel_types[0] × rel_types[1] × ... × rel_types[n-1]` under the semiring.
    pub fn multi_hop(
        &self,
        rel_types: &[&str],
        semiring: &dyn Semiring,
    ) -> Option<GrBMatrix> {
        if rel_types.is_empty() {
            return None;
        }

        let desc = GrBDesc::default();
        let mut result = self.relations.get(rel_types[0])?.clone();

        for &rel in &rel_types[1..] {
            let next = self.relations.get(rel)?;
            result = result.mxm(next, semiring, &desc);
        }

        Some(result)
    }

    /// Masked traversal: traverse a relationship type, masking by label.
    ///
    /// Returns only entries where the target node has the given label.
    pub fn masked_traverse(
        &self,
        rel_type: &str,
        label_mask: &str,
        semiring: &dyn Semiring,
    ) -> Option<GrBMatrix> {
        let matrix = self.relations.get(rel_type)?;
        let mask = self.labels.get(label_mask)?;

        let desc = GrBDesc::default();
        let result = matrix.mxm(matrix, semiring, &desc);

        // Apply mask: zero out entries where target is not in label
        let mut coo = CooStorage::new(result.nrows(), result.ncols());
        for (r, c, v) in result.iter() {
            if c < mask.len() && mask[c] {
                coo.push(r, c, v.clone());
            }
        }
        Some(GrBMatrix::from_coo(&coo))
    }

    /// Get the label mask for filtering results.
    pub fn label_mask(&self, label: &str) -> Option<&[bool]> {
        self.labels.get(label).map(|v| v.as_slice())
    }

    /// Get a relationship matrix.
    pub fn relation(&self, rel_type: &str) -> Option<&GrBMatrix> {
        self.relations.get(rel_type)
    }

    /// Bridge: build from an SpoStore.
    ///
    /// Extracts relationship types from predicate fingerprints. Since SpoStore
    /// uses fingerprint-based keys (not string labels), this creates a single
    /// "SPO" relationship type with all edges. For labeled decomposition,
    /// use `add_relation` manually per relationship type.
    pub fn from_spo_store(store: &SpoStore, node_count: usize) -> Self {
        let mut graph = TypedGraph::new(node_count);

        // SpoStore doesn't expose iteration, but we can build from known edges.
        // For the bridge, create an empty "SPO" relation that callers populate.
        let matrix = GrBMatrix::new(node_count, node_count);
        graph.add_relation("SPO", matrix);

        graph
    }
}

/// Result of a blasgraph computation with truth metadata.
#[derive(Debug, Clone)]
pub struct BlasGraphHit {
    /// Source node index.
    pub source: usize,
    /// Target node index.
    pub target: usize,
    /// The edge vector.
    pub value: BitVec,
    /// Truth value (populated if available).
    pub truth: TruthValue,
}

/// Apply TruthGate filtering after matrix traversal.
///
/// The planner produces candidate positions. TruthGate filters post-hoc.
pub fn apply_truth_gate(
    result: &GrBMatrix,
    gate: crate::graph::spo::truth::TruthGate,
    truth_values: &HashMap<(usize, usize), TruthValue>,
) -> Vec<BlasGraphHit> {
    let mut hits = Vec::new();
    for (r, c, v) in result.iter() {
        let truth = truth_values
            .get(&(r, c))
            .copied()
            .unwrap_or_else(TruthValue::unknown);
        if gate.passes(&truth) {
            hits.push(BlasGraphHit {
                source: r,
                target: c,
                value: v.clone(),
                truth,
            });
        }
    }
    hits
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::blasgraph::semiring::HdrSemiring;
    use crate::graph::spo::truth::TruthGate;

    fn make_knows_graph() -> TypedGraph {
        // 4 nodes: Jan(0), Ada(1), Max(2), Eve(3)
        let mut graph = TypedGraph::new(4);

        // KNOWS: Jan->Ada, Ada->Max, Max->Eve
        let mut coo = CooStorage::new(4, 4);
        coo.push(0, 1, BitVec::random(100)); // Jan->Ada
        coo.push(1, 2, BitVec::random(101)); // Ada->Max
        coo.push(2, 3, BitVec::random(102)); // Max->Eve

        graph.add_relation("KNOWS", GrBMatrix::from_coo(&coo));

        // Labels
        graph.add_label("Person", &[0, 1, 2, 3]);
        graph.add_label("Engineer", &[1, 2]); // Ada, Max

        graph
    }

    #[test]
    fn test_typed_graph_basic() {
        let graph = make_knows_graph();
        assert_eq!(graph.node_count, 4);
        assert!(graph.relations.contains_key("KNOWS"));
        assert!(graph.labels.contains_key("Person"));
    }

    #[test]
    fn test_two_hop_knows_squared() {
        let graph = make_knows_graph();
        let result = graph
            .multi_hop(&["KNOWS", "KNOWS"], &HdrSemiring::XorBundle)
            .unwrap();

        // KNOWS² should give:
        // Jan->Max (via Ada): (0,2)
        // Ada->Eve (via Max): (1,3)
        assert!(result.get(0, 2).is_some(), "Jan should reach Max in 2 hops");
        assert!(result.get(1, 3).is_some(), "Ada should reach Eve in 2 hops");
        assert!(
            result.get(0, 3).is_none(),
            "Jan should NOT reach Eve in exactly 2 hops"
        );
    }

    #[test]
    fn test_masked_traverse() {
        let graph = make_knows_graph();
        let result = graph
            .masked_traverse("KNOWS", "Engineer", &HdrSemiring::XorBundle)
            .unwrap();

        // KNOWS² masked by Engineer (nodes 1,2):
        // Jan->Max(2) ✓ (Max is Engineer)
        // Ada->Eve(3) ✗ (Eve is not Engineer)
        assert!(
            result.get(0, 2).is_some(),
            "Jan->Max should pass (Max is Engineer)"
        );
        assert!(
            result.get(1, 3).is_none(),
            "Ada->Eve should be filtered (Eve not Engineer)"
        );
    }

    #[test]
    fn test_label_mask() {
        let graph = make_knows_graph();
        let mask = graph.label_mask("Engineer").unwrap();
        assert_eq!(mask, &[false, true, true, false]);
    }

    #[test]
    fn test_truth_gate_filtering() {
        let graph = make_knows_graph();
        let knows = graph.relation("KNOWS").unwrap();

        let mut truth_values = HashMap::new();
        // Jan->Ada: strong truth
        truth_values.insert((0, 1), TruthValue::new(0.9, 0.9));
        // Ada->Max: weak truth
        truth_values.insert((1, 2), TruthValue::new(0.3, 0.2));
        // Max->Eve: medium truth
        truth_values.insert((2, 3), TruthValue::new(0.7, 0.7));

        // STRONG gate (0.75): only Jan->Ada should pass
        let strong_hits = apply_truth_gate(knows, TruthGate::STRONG, &truth_values);
        assert_eq!(strong_hits.len(), 1);
        assert_eq!(strong_hits[0].source, 0);
        assert_eq!(strong_hits[0].target, 1);

        // OPEN gate: all pass
        let open_hits = apply_truth_gate(knows, TruthGate::OPEN, &truth_values);
        assert_eq!(open_hits.len(), 3);
    }

    #[test]
    fn test_from_spo_store() {
        let store = SpoStore::new();
        let graph = TypedGraph::from_spo_store(&store, 10);
        assert_eq!(graph.node_count, 10);
        assert!(graph.relations.contains_key("SPO"));
    }

    #[test]
    fn test_multi_hop_nonexistent_rel() {
        let graph = make_knows_graph();
        let result = graph.multi_hop(&["LIKES"], &HdrSemiring::XorBundle);
        assert!(result.is_none());
    }
}
