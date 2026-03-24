//! Cost model for query planning.
//!
//! Thinking-aware: the modulation parameters from the thinking context
//! influence cost estimates (e.g., exploratory styles tolerate higher costs).

use crate::thinking::ThinkingContext;
use super::QueryGraphNode;

/// Cost model influenced by thinking context.
pub struct CostModel {
    /// Base cardinality for unknown tables.
    base_cardinality: f64,
    /// Cost penalty for flattening factorized groups (from Kuzudb).
    flatten_penalty: f64,
    /// Exploration discount: exploratory styles accept costlier plans.
    exploration_discount: f64,
}

impl CostModel {
    pub fn new(thinking: &ThinkingContext) -> Self {
        // Exploration discount: exploratory styles tolerate 2x cost
        let exploration_discount = 1.0 + thinking.modulation.exploration;

        Self {
            base_cardinality: 1000.0,
            flatten_penalty: 2.0,
            exploration_discount,
        }
    }

    /// Cost of scanning a single node table.
    pub fn scan_cost(&self, node: &QueryGraphNode) -> f64 {
        // In a real implementation, this would use table statistics.
        self.base_cardinality / self.exploration_discount
    }

    /// Cost of a hash join.
    /// cost = probe_cost + build_cost + probe_cardinality + penalty * flat_build_cardinality
    pub fn hash_join_cost(&self, probe_cost: f64, build_cost: f64) -> f64 {
        let probe_card = probe_cost; // Simplified: cost ≈ cardinality
        let build_card = build_cost;
        (probe_cost + build_cost + probe_card + self.flatten_penalty * build_card)
            / self.exploration_discount
    }

    /// Cost of an index nested loop join (extend through adjacency list).
    /// Much cheaper than hash join when bound node has index.
    pub fn inl_join_cost(&self, left_cost: f64) -> f64 {
        // INL cost = left cost + left cardinality (one lookup per left row)
        (left_cost + left_cost * 0.5) / self.exploration_discount
    }

    /// Cost of a WCO (worst-case optimal) join.
    pub fn wco_join_cost(&self, child_costs_sum: f64, num_children: usize) -> f64 {
        // WCO is cheaper than pairwise hash joins for star patterns
        let intersection_factor = 0.5f64.powi(num_children as i32 - 1);
        (child_costs_sum * intersection_factor) / self.exploration_discount
    }

    /// Cost of a resonance scan (BROADCAST → SCAN).
    pub fn resonance_scan_cost(&self, partitions: usize, threshold: u32) -> f64 {
        let selectivity = threshold as f64 / 2000.0; // Higher threshold = more selective
        self.base_cardinality * selectivity * partitions as f64 / self.exploration_discount
    }

    /// Cost of an ACCUMULATE operation.
    pub fn accumulate_cost(&self, input_cost: f64, hops: usize) -> f64 {
        input_cost * hops as f64 * 1.5 / self.exploration_discount
    }
}
