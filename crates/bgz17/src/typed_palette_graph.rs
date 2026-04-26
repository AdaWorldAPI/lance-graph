//! TypedPaletteGraph: typed property graph in palette space.
//!
//! Mirror of lance-graph's TypedGraph but with PaletteMatrix instead of GrBMatrix.
//! Each relation type is a PaletteMatrix (sparse CSR with PaletteEdge values).
//! Supports single-hop and multi-hop traversal via palette semiring mxm.

use crate::distance_matrix::SpoDistanceMatrices;
// PaletteEdge retained for the encode→store path that converts SpoBase17 edges
// into palette-indexed form before inserting into the typed graph (TD-BGZ17-TPG-1).
#[allow(unused_imports)]
use crate::palette::PaletteEdge;
use crate::palette_matrix::PaletteMatrix;
use crate::palette_semiring::SpoPaletteSemiring;
use crate::scalar_sparse::ScalarCsr;
use std::collections::HashMap;

/// A typed property graph in palette space.
///
/// Each relation type maps to a PaletteMatrix (sparse adjacency with PaletteEdge values).
/// Labels are boolean vectors indicating node membership.
pub struct TypedPaletteGraph {
    /// Relation type -> PaletteMatrix adjacency.
    pub relations: HashMap<String, PaletteMatrix>,
    /// Label name -> boolean vector (true if node has label).
    pub labels: HashMap<String, Vec<bool>>,
    /// Number of nodes in the graph.
    pub node_count: usize,
    /// Per-plane palette semirings (compose tables + distance matrices).
    pub semirings: SpoPaletteSemiring,
    /// Per-plane precomputed distance matrices.
    pub distances: SpoDistanceMatrices,
}

impl TypedPaletteGraph {
    /// Create a new empty typed palette graph.
    pub fn new(
        node_count: usize,
        semirings: SpoPaletteSemiring,
        distances: SpoDistanceMatrices,
    ) -> Self {
        TypedPaletteGraph {
            relations: HashMap::new(),
            labels: HashMap::new(),
            node_count,
            semirings,
            distances,
        }
    }

    /// Add a relation type with its PaletteMatrix.
    pub fn add_relation(&mut self, name: &str, matrix: PaletteMatrix) {
        self.relations.insert(name.to_string(), matrix);
    }

    /// Add a label, marking the given node IDs as having this label.
    pub fn add_label(&mut self, name: &str, node_ids: &[usize]) {
        let mut mask = vec![false; self.node_count];
        for &id in node_ids {
            if id < self.node_count {
                mask[id] = true;
            }
        }
        self.labels.insert(name.to_string(), mask);
    }

    /// Single-hop traversal: mxm of the relation matrix with itself.
    ///
    /// Returns the PaletteMatrix for 2-hop paths through `rel_type`.
    /// Returns None if the relation type is not found.
    pub fn traverse(&self, rel_type: &str) -> Option<PaletteMatrix> {
        let mat = self.relations.get(rel_type)?;
        let k_s = self.semirings.subject.k;
        // p/o codebook sizes retained for the multi-codebook mxm path
        // where each S/P/O plane uses its own distance table (TD-BGZ17-MULTI-K).
        let _k_p = self.semirings.predicate.k;
        let _k_o = self.semirings.object.k;
        // For mxm we need a single k; use subject k and corresponding compose tables.
        // The mxm takes separate compose tables per plane.
        Some(PaletteMatrix::mxm(
            mat,
            mat,
            &self.semirings.subject.compose_table,
            &self.semirings.predicate.compose_table,
            &self.semirings.object.compose_table,
            k_s, // k_pal for compose table indexing
            &self.distances,
        ))
    }

    /// Multi-hop traversal: chain mxm across multiple relation types.
    ///
    /// `rel_types`: sequence of relation type names to traverse.
    /// Returns the composed PaletteMatrix, or None if any relation type is missing.
    pub fn multi_hop(&self, rel_types: &[&str]) -> Option<PaletteMatrix> {
        if rel_types.is_empty() {
            return None;
        }

        let mut result = self.relations.get(rel_types[0])?.clone();
        let k_s = self.semirings.subject.k;

        for &rel in &rel_types[1..] {
            let next = self.relations.get(rel)?;
            result = PaletteMatrix::mxm(
                &result,
                next,
                &self.semirings.subject.compose_table,
                &self.semirings.predicate.compose_table,
                &self.semirings.object.compose_table,
                k_s,
                &self.distances,
            );
        }

        Some(result)
    }

    /// Convert a relation to a ScalarCsr with SPO distances from origin.
    ///
    /// Each PaletteEdge is replaced by its combined S+P+O distance from (0,0,0).
    /// Returns None if the relation type is not found.
    pub fn to_distance_csr(&self, rel_type: &str) -> Option<ScalarCsr> {
        let mat = self.relations.get(rel_type)?;
        Some(mat.to_distance_csr(&self.distances))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::base17::Base17;
    use crate::palette::Palette;
    use crate::palette_semiring::SpoPaletteSemiring;
    use crate::distance_matrix::SpoDistanceMatrices;
    use crate::BASE_DIM;

    fn make_palette(k: usize) -> Palette {
        let entries = (0..k).map(|i| {
            let mut dims = [0i16; BASE_DIM];
            for d in 0..BASE_DIM {
                dims[d] = ((i * 97 + d * 31) % 512) as i16 - 256;
            }
            Base17 { dims }
        }).collect();
        Palette { entries }
    }

    fn make_graph(k: usize) -> TypedPaletteGraph {
        let pal = make_palette(k);
        let semirings = SpoPaletteSemiring::build(&pal, &pal, &pal);
        let distances = SpoDistanceMatrices::build(&pal, &pal, &pal);
        TypedPaletteGraph::new(4, semirings, distances)
    }

    #[test]
    fn test_basic_build() {
        let g = make_graph(16);
        assert_eq!(g.node_count, 4);
        assert!(g.relations.is_empty());
        assert!(g.labels.is_empty());
    }

    #[test]
    fn test_add_relation_and_label() {
        let mut g = make_graph(16);
        let pe = PaletteEdge { s_idx: 1, p_idx: 2, o_idx: 3 };
        let mat = PaletteMatrix::from_triples(4, 4, &[
            (0, 1, pe),
            (1, 2, pe),
            (2, 3, pe),
        ]);
        g.add_relation("knows", mat);
        g.add_label("person", &[0, 1, 2, 3]);

        assert!(g.relations.contains_key("knows"));
        assert_eq!(g.labels["person"], vec![true, true, true, true]);
    }

    #[test]
    fn test_traverse_produces_mxm() {
        let mut g = make_graph(16);
        let pe = PaletteEdge { s_idx: 1, p_idx: 2, o_idx: 3 };
        let mat = PaletteMatrix::from_triples(4, 4, &[
            (0, 1, pe),
            (1, 2, pe),
            (2, 3, pe),
        ]);
        g.add_relation("knows", mat);

        let result = g.traverse("knows").expect("traverse should return Some");
        // 2-hop: 0→1→2, 1→2→3 should exist
        assert!(result.get(0, 2).is_some(), "2-hop path 0→1→2 should exist");
        assert!(result.get(1, 3).is_some(), "2-hop path 1→2→3 should exist");
        // No 2-hop from 2→? beyond 3, and 3 has no outgoing
        assert!(result.get(0, 1).is_none(), "direct 1-hop should not be in 2-hop result");
    }

    #[test]
    fn test_multi_hop_chains() {
        let mut g = make_graph(16);
        let pe_a = PaletteEdge { s_idx: 1, p_idx: 2, o_idx: 3 };
        let pe_b = PaletteEdge { s_idx: 4, p_idx: 5, o_idx: 6 };

        let mat_a = PaletteMatrix::from_triples(4, 4, &[
            (0, 1, pe_a),
            (1, 2, pe_a),
        ]);
        let mat_b = PaletteMatrix::from_triples(4, 4, &[
            (2, 3, pe_b),
        ]);
        g.add_relation("knows", mat_a);
        g.add_relation("likes", mat_b);

        // multi_hop: knows then likes = 0→1→? (knows∘likes at 1→? needs likes from 1, none)
        // Actually: knows gives 0→1, 1→2. Then likes gives 2→3.
        // knows∘likes: for each (i,k) in knows and (k,j) in likes: i→j
        // (0,1) in knows, (1,?) in likes: none. (1,2) in knows, (2,3) in likes: 1→3.
        let result = g.multi_hop(&["knows", "likes"]).expect("multi_hop should return Some");
        assert!(result.get(1, 3).is_some(), "path 1→2→3 via knows∘likes should exist");
    }

    #[test]
    fn test_multi_hop_missing_relation() {
        let g = make_graph(16);
        assert!(g.multi_hop(&["nonexistent"]).is_none());
    }

    #[test]
    fn test_to_distance_csr() {
        let mut g = make_graph(16);
        let pe = PaletteEdge { s_idx: 1, p_idx: 2, o_idx: 3 };
        let mat = PaletteMatrix::from_triples(4, 4, &[
            (0, 1, pe),
            (1, 2, pe),
        ]);
        g.add_relation("knows", mat);

        let csr = g.to_distance_csr("knows").expect("should return Some");
        assert_eq!(csr.nnz(), 2);
        // Distances should be non-negative
        for &v in &csr.vals {
            assert!(v >= 0.0);
        }
    }

    #[test]
    fn test_to_distance_csr_missing() {
        let g = make_graph(16);
        assert!(g.to_distance_csr("nonexistent").is_none());
    }
}
