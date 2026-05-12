//! CausalNetwork: a graph of CausalEdge64 neurons.
//!
//! Not a neural network in the backpropagation sense.
//! A causal graph where every edge IS a 64-bit neuron,
//! every forward pass IS a compose-table lookup,
//! every learning step IS NARS evidence revision,
//! and every Pearl level IS a 3-bit mask.

use crate::edge::{CausalEdge64, InferenceType};
use crate::pearl::CausalMask;
use crate::plasticity::PlasticityState;
use crate::tables::NarsTables;

/// A causal network: edges stored as u64 array, adjacency as CSR.
pub struct CausalNetwork {
    /// All edges (the "weights" / "neurons").
    pub edges: Vec<CausalEdge64>,

    /// CSR row pointers: outgoing edges from node i start at row_ptr[i].
    pub row_ptr: Vec<usize>,

    /// CSR column indices: edges[col_idx[j]] connects to node col_idx[j].
    pub col_idx: Vec<usize>,

    /// Edge indices: which edge in `edges` corresponds to CSR position j.
    pub edge_idx: Vec<usize>,

    /// Compose tables per plane (256×256 u8, precomputed from palette).
    pub compose_s: Box<[u8; 256 * 256]>,
    pub compose_p: Box<[u8; 256 * 256]>,
    pub compose_o: Box<[u8; 256 * 256]>,

    /// Distance matrices per plane (256×256 u16, precomputed from palette).
    pub dist_s: Box<[u16; 256 * 256]>,
    pub dist_p: Box<[u16; 256 * 256]>,
    pub dist_o: Box<[u16; 256 * 256]>,

    /// Precomputed NARS lookup tables.
    pub nars_tables: NarsTables,

    /// Current time slot (monotonically increasing).
    pub current_time: u16,
}

/// Result of a multi-hop causal query.
#[derive(Debug, Clone)]
pub struct CausalPath {
    /// Sequence of edges traversed.
    pub hops: Vec<CausalEdge64>,
    /// Final composed edge (cumulative forward pass).
    pub conclusion: CausalEdge64,
    /// Pearl level achieved (minimum mask across path).
    pub causal_level: CausalMask,
}

impl CausalNetwork {
    /// Forward pass through a sequence of edges (causal chain).
    ///
    /// Input edge is composed with each weight edge in sequence.
    /// Each intermediate IS a CausalEdge64 with full interpretability.
    pub fn forward_chain(&self, input: CausalEdge64, path: &[usize]) -> CausalPath {
        let mut current = input;
        let mut hops = Vec::with_capacity(path.len() + 1);
        hops.push(current);

        for &edge_id in path {
            let weight = self.edges[edge_id];
            current = current.forward(
                weight,
                &self.compose_s,
                &self.compose_p,
                &self.compose_o,
            );
            hops.push(current);
        }

        let causal_level = current.causal_mask();
        CausalPath {
            hops,
            conclusion: current,
            causal_level,
        }
    }

    /// Learn from observation: update all edges on the path with NARS revision.
    pub fn learn_path(&mut self, path: &[usize], observation: CausalEdge64) {
        self.current_time = self.current_time.wrapping_add(1);
        for &edge_id in path {
            self.edges[edge_id].learn(observation, self.current_time);
        }
    }

    /// Causal query: find edges matching a palette pattern at a specific Pearl level.
    ///
    /// Returns edges sorted by causal distance (closest first).
    pub fn causal_query(
        &self,
        query: CausalEdge64,
        pearl_level: CausalMask,
        max_results: usize,
    ) -> Vec<(usize, u32)> {
        // Set the query's causal mask to the requested Pearl level
        let mut q = query;
        q.set_causal_mask(pearl_level);

        let mut results: Vec<(usize, u32)> = self.edges
            .iter()
            .enumerate()
            .map(|(i, &edge)| {
                let d = q.causal_distance(
                    edge,
                    &self.dist_s,
                    &self.dist_p,
                    &self.dist_o,
                );
                (i, d)
            })
            .collect();

        results.sort_by_key(|&(_, d)| d);
        results.truncate(max_results);
        results
    }

    /// Detect Simpson's Paradox: find edges where S_O and _PO disagree.
    ///
    /// Returns edge indices with potential Simpson's Paradox,
    /// along with the associational and interventional directions.
    pub fn detect_simpsons_paradox(&self) -> Vec<(usize, u8, u8)> {
        let mut paradoxes = Vec::new();

        for (i, &edge) in self.edges.iter().enumerate() {
            // Check if this edge has enough evidence
            if edge.confidence() < 0.5 { continue; }

            // Compare direction at S_O (association) vs _PO (intervention)
            // We need to query neighbors at both levels to get directions.
            // Simplified: check if the edge's own direction bits suggest
            // conflicting signals across planes.
            let dir = edge.direction();
            let s_path = dir & 0b001 != 0; // S pathological
            let o_path = dir & 0b100 != 0; // O pathological

            // If subject is pathological AND outcome is pathological,
            // but the predicate (treatment) is NOT pathological,
            // there may be confounding: sick patients get treated,
            // treatment works, but S_O still shows negative association.
            if s_path && o_path && !(dir & 0b010 != 0) {
                paradoxes.push((i, dir, edge.causal_mask() as u8));
            }
        }

        paradoxes
    }

    /// Evidence audit: trace how a specific edge's truth value evolved.
    ///
    /// Returns edges with the same SPO indices sorted by temporal index.
    pub fn evidence_trail(&self, s: u8, p: u8, o: u8) -> Vec<&CausalEdge64> {
        let mut trail: Vec<&CausalEdge64> = self.edges
            .iter()
            .filter(|e| e.s_idx() == s && e.p_idx() == p && e.o_idx() == o)
            .collect();

        trail.sort_by_key(|e| e.temporal());
        trail
    }

    /// Counterfactual query: "what if THIS patient had THAT treatment?"
    ///
    /// Takes a patient edge (S known) and an alternative treatment archetype,
    /// composes them through the compose table, returns predicted outcome.
    pub fn counterfactual(
        &self,
        patient_edge: CausalEdge64,
        alternative_p: u8,
    ) -> CausalEdge64 {
        // Create a hypothetical edge with the alternative predicate
        let mut hypothetical = patient_edge;
        hypothetical.set_p_idx(alternative_p);
        hypothetical.set_causal_mask(CausalMask::SPO); // counterfactual level

        // Find most similar edges in the network at SPO level
        let similar = self.causal_query(hypothetical, CausalMask::SPO, 10);

        if similar.is_empty() {
            return hypothetical;
        }

        // Bundle the top-k similar edges' outcomes via majority vote
        let mut f_sum = 0.0f32;
        let mut c_sum = 0.0f32;
        let mut o_votes = [0u32; 256];
        let mut count = 0u32;

        for &(idx, _dist) in &similar {
            let edge = self.edges[idx];
            f_sum += edge.frequency();
            c_sum += edge.confidence();
            o_votes[edge.o_idx() as usize] += 1;
            count += 1;
        }

        // Majority vote for outcome archetype
        let best_o = o_votes.iter().enumerate()
            .max_by_key(|&(_, &v)| v)
            .map(|(i, _)| i as u8)
            .unwrap_or(hypothetical.o_idx());

        let mut result = hypothetical;
        result.set_o_idx(best_o);
        result.set_frequency((f_sum / count as f32).min(1.0));
        // Counterfactual confidence is attenuated by how hypothetical it is
        result.set_confidence((c_sum / count as f32 * 0.8).min(1.0));
        result.set_inference_type(InferenceType::Abduction); // counterfactual = abduction
        result
    }

    /// Network statistics.
    pub fn stats(&self) -> NetworkStats {
        let total = self.edges.len();
        let frozen = self.edges.iter().filter(|e| e.is_frozen()).count();
        let hot = self.edges.iter().filter(|e| e.plasticity() == PlasticityState::ALL_HOT).count();
        let high_conf = self.edges.iter().filter(|e| e.confidence() > 0.7).count();
        let interventional = self.edges.iter().filter(|e| e.is_interventional()).count();
        let counterfactual = self.edges.iter().filter(|e| e.is_counterfactual()).count();

        let avg_f = self.edges.iter().map(|e| e.frequency()).sum::<f32>() / total.max(1) as f32;
        let avg_c = self.edges.iter().map(|e| e.confidence()).sum::<f32>() / total.max(1) as f32;

        NetworkStats {
            total_edges: total,
            frozen_edges: frozen,
            hot_edges: hot,
            high_confidence: high_conf,
            interventional_edges: interventional,
            counterfactual_edges: counterfactual,
            avg_frequency: avg_f,
            avg_confidence: avg_c,
            memory_bytes: total * 8, // 8 bytes per edge
        }
    }
}

/// Network-level statistics.
#[derive(Debug)]
pub struct NetworkStats {
    pub total_edges: usize,
    pub frozen_edges: usize,
    pub hot_edges: usize,
    pub high_confidence: usize,
    pub interventional_edges: usize,
    pub counterfactual_edges: usize,
    pub avg_frequency: f32,
    pub avg_confidence: f32,
    pub memory_bytes: usize,
}
