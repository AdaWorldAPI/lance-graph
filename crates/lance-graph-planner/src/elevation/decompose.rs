//! Fan-out detection and query decomposition.
//!
//! When the planner detects a query that fans out (e.g., `MATCH (a)-[:SIMILAR*2]->(b)`),
//! it doesn't just "go faster" — it DECOMPOSES the query into independent sub-queries
//! at the elevation boundary. Each sub-query runs in its own morsel.

use crate::physical::Morsel;

/// Result of fan-out analysis.
#[derive(Debug, Clone)]
pub struct FanOutAnalysis {
    /// Estimated fan-out factor per hop.
    pub fan_out_per_hop: f64,
    /// Number of hops in the path expansion.
    pub hops: usize,
    /// Estimated total result count (fan_out_per_hop ^ hops * input_size).
    pub estimated_total: f64,
    /// Whether decomposition is recommended.
    pub should_decompose: bool,
}

/// Analyze fan-out from query features.
///
/// Variable-length path queries can explode quadratically. This function
/// estimates the explosion factor and recommends decomposition.
pub fn analyze_fanout(
    has_variable_length_path: bool,
    num_match_clauses: usize,
    estimated_complexity: f64,
    input_cardinality: f64,
) -> FanOutAnalysis {
    if !has_variable_length_path {
        return FanOutAnalysis {
            fan_out_per_hop: 1.0,
            hops: 1,
            estimated_total: input_cardinality * num_match_clauses as f64,
            should_decompose: false,
        };
    }

    // Conservative fan-out estimate: typical graph has avg degree 5-10
    let fan_out_per_hop = 5.0 + estimated_complexity * 10.0;
    let hops = num_match_clauses.max(2); // at least 2 for VLP
    let estimated_total = input_cardinality * fan_out_per_hop.powi(hops as i32);

    FanOutAnalysis {
        fan_out_per_hop,
        hops,
        estimated_total,
        // Decompose if estimated results exceed 100K
        should_decompose: estimated_total > 100_000.0,
    }
}

/// Decompose a fan-out result set into independent sub-queries.
///
/// Takes the intermediate results from a lower elevation level and splits
/// them into independent anchor points. Each anchor becomes its own
/// morsel at the next elevation level.
///
/// Returns the number of sub-queries created.
pub fn decompose_fanout(results_so_far: &Morsel, max_partitions: usize) -> Vec<Morsel> {
    if results_so_far.num_rows == 0 {
        return vec![];
    }

    let partition_size = (results_so_far.num_rows / max_partitions).max(1);
    let num_partitions = (results_so_far.num_rows + partition_size - 1) / partition_size;

    (0..num_partitions).map(|i| {
        let start = i * partition_size;
        let end = ((i + 1) * partition_size).min(results_so_far.num_rows);
        Morsel {
            num_rows: end - start,
            columns: vec![], // Real impl: slice Arrow columns
        }
    }).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_no_fanout_for_simple_query() {
        let analysis = analyze_fanout(false, 1, 0.2, 1000.0);
        assert!(!analysis.should_decompose);
        assert_eq!(analysis.fan_out_per_hop, 1.0);
    }

    #[test]
    fn test_fanout_detected_for_vlp() {
        let analysis = analyze_fanout(true, 3, 0.5, 1000.0);
        assert!(analysis.should_decompose);
        assert!(analysis.estimated_total > 100_000.0);
    }

    #[test]
    fn test_decompose_splits_morsel() {
        let morsel = Morsel { num_rows: 1000, columns: vec![] };
        let parts = decompose_fanout(&morsel, 4);
        assert_eq!(parts.len(), 4);
        let total: usize = parts.iter().map(|m| m.num_rows).sum();
        assert_eq!(total, 1000);
    }

    #[test]
    fn test_decompose_empty_morsel() {
        let morsel = Morsel { num_rows: 0, columns: vec![] };
        let parts = decompose_fanout(&morsel, 4);
        assert!(parts.is_empty());
    }
}
