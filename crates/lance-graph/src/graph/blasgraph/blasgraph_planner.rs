// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! # Blasgraph Physical Planner
//!
//! Second execution backend: Cypher → `grb_mxm` instead of Cypher → SQL.
//! Maps `LogicalOperator::Expand` → matrix multiply, `ScanByLabel` → label mask,
//! `Filter` → predicate mask.
//!
//! **CRITICAL:** TruthGate filtering happens AFTER matrix traversal, not during.
//! The planner produces candidate positions. Then `apply_truth_gate` filters.

use std::collections::HashMap;

use crate::graph::blasgraph::descriptor::GrBDesc;
use crate::graph::blasgraph::matrix::GrBMatrix;
use crate::graph::blasgraph::semiring::Semiring;
use crate::graph::blasgraph::sparse::CooStorage;
use crate::graph::blasgraph::typed_graph::TypedGraph;
use crate::graph::blasgraph::types::BitVec;
use crate::logical_plan::LogicalOperator;

/// Compile a logical plan to a blasgraph matrix result.
///
/// Returns a result matrix where entry `(i, j)` means node `i` connects to node `j`
/// via the traversal pattern described by the plan.
///
/// Only handles the subset of logical operators that map to matrix algebra:
/// - `ScanByLabel` → label mask (diagonal)
/// - `Expand` → matrix multiply
/// - `VariableLengthExpand` → iterated matrix multiply
/// - `Filter` → mask application (structural only)
/// - Other operators → pass through or error
pub fn compile_to_blasgraph(
    plan: &LogicalOperator,
    graph: &TypedGraph,
    semiring: &dyn Semiring,
) -> Result<GrBMatrix, BlasGraphPlanError> {
    let desc = GrBDesc::default();

    match plan {
        LogicalOperator::ScanByLabel { label, .. } => {
            // Produce a diagonal matrix for the label mask
            let mask = graph
                .label_mask(label)
                .ok_or_else(|| BlasGraphPlanError::UnknownLabel(label.clone()))?;

            let mut coo = CooStorage::new(graph.node_count, graph.node_count);
            for (i, &has_label) in mask.iter().enumerate() {
                if has_label {
                    coo.push(i, i, BitVec::random(i as u64 + 1));
                }
            }
            Ok(GrBMatrix::from_coo(&coo))
        }

        LogicalOperator::Expand {
            input,
            relationship_types,
            ..
        } => {
            // Compile the input first
            let input_matrix = compile_to_blasgraph(input, graph, semiring)?;

            // Get the relationship matrix (first matching type)
            let rel_matrix = find_relation_matrix(graph, relationship_types)?;

            // input × relationship = one-hop expansion
            Ok(input_matrix.mxm(rel_matrix, semiring, &desc))
        }

        LogicalOperator::VariableLengthExpand {
            input,
            relationship_types,
            min_length,
            max_length,
            ..
        } => {
            let input_matrix = compile_to_blasgraph(input, graph, semiring)?;
            let rel_matrix = find_relation_matrix(graph, relationship_types)?;

            let min = min_length.unwrap_or(1) as usize;
            let max = max_length.unwrap_or(3) as usize;

            // Iterated multiply: accumulate hops from min to max
            let mut power = rel_matrix.clone();
            let mut accumulated = if min <= 1 {
                input_matrix.mxm(rel_matrix, semiring, &desc)
            } else {
                // Compute rel^min
                for _ in 1..min {
                    power = power.mxm(rel_matrix, semiring, &desc);
                }
                input_matrix.mxm(&power, semiring, &desc)
            };

            // Add higher powers
            for hop in (min + 1)..=max {
                if hop <= 1 {
                    continue;
                }
                // power = rel^hop
                power = power.mxm(rel_matrix, semiring, &desc);
                let hop_result = input_matrix.mxm(&power, semiring, &desc);
                // Union with accumulated
                accumulated = accumulated.ewise_add(
                    &hop_result,
                    crate::graph::blasgraph::types::BinaryOp::First,
                    &desc,
                );
            }

            Ok(accumulated)
        }

        LogicalOperator::Filter { input, .. } => {
            // For now, compile the input and return it.
            // Structural filters (label masks) are applied during ScanByLabel.
            // Property filters require post-processing outside the matrix algebra.
            compile_to_blasgraph(input, graph, semiring)
        }

        LogicalOperator::Project { input, .. } => {
            // Pass through — projection is handled after matrix computation
            compile_to_blasgraph(input, graph, semiring)
        }

        _ => Err(BlasGraphPlanError::UnsupportedOperator(format!(
            "{:?}",
            std::mem::discriminant(plan)
        ))),
    }
}

/// Find the first matching relationship matrix.
fn find_relation_matrix<'a>(
    graph: &'a TypedGraph,
    relationship_types: &[String],
) -> Result<&'a GrBMatrix, BlasGraphPlanError> {
    for rel_type in relationship_types {
        if let Some(matrix) = graph.relation(rel_type) {
            return Ok(matrix);
        }
    }
    // If no specific type requested, try the "SPO" catch-all
    if relationship_types.is_empty() {
        if let Some(matrix) = graph.relation("SPO") {
            return Ok(matrix);
        }
    }
    Err(BlasGraphPlanError::UnknownRelationType(
        relationship_types.first().cloned().unwrap_or_default(),
    ))
}

/// Errors from blasgraph plan compilation.
#[derive(Debug, Clone)]
pub enum BlasGraphPlanError {
    /// Referenced label not found in TypedGraph.
    UnknownLabel(String),
    /// Referenced relationship type not found in TypedGraph.
    UnknownRelationType(String),
    /// Logical operator not supported by blasgraph backend.
    UnsupportedOperator(String),
}

impl std::fmt::Display for BlasGraphPlanError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            BlasGraphPlanError::UnknownLabel(l) => write!(f, "Unknown label: {}", l),
            BlasGraphPlanError::UnknownRelationType(r) => {
                write!(f, "Unknown relationship type: {}", r)
            }
            BlasGraphPlanError::UnsupportedOperator(o) => {
                write!(f, "Unsupported operator for blasgraph: {}", o)
            }
        }
    }
}

impl std::error::Error for BlasGraphPlanError {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::blasgraph::semiring::HdrSemiring;
    use crate::graph::blasgraph::typed_graph::{apply_truth_gate, TypedGraph};
    use crate::graph::spo::truth::{TruthGate, TruthValue};

    fn make_test_graph() -> TypedGraph {
        // 4 nodes: 0=Jan(Person), 1=Ada(Person,Engineer), 2=Max(Person,Engineer), 3=Eve(Person)
        let mut graph = TypedGraph::new(4);

        let mut coo = CooStorage::new(4, 4);
        coo.push(0, 1, BitVec::random(100)); // Jan->Ada
        coo.push(1, 2, BitVec::random(101)); // Ada->Max
        coo.push(2, 3, BitVec::random(102)); // Max->Eve
        graph.add_relation("KNOWS", GrBMatrix::from_coo(&coo));

        graph.add_label("Person", &[0, 1, 2, 3]);
        graph.add_label("Engineer", &[1, 2]);

        graph
    }

    #[test]
    fn test_scan_by_label() {
        let graph = make_test_graph();
        let plan = LogicalOperator::ScanByLabel {
            variable: "a".to_string(),
            label: "Person".to_string(),
            properties: HashMap::new(),
        };
        let result = compile_to_blasgraph(&plan, &graph, &HdrSemiring::Boolean).unwrap();
        // Diagonal matrix with 4 entries (all persons)
        assert_eq!(result.nnz(), 4);
        for i in 0..4 {
            assert!(result.get(i, i).is_some());
        }
    }

    #[test]
    fn test_expand_single_hop() {
        let graph = make_test_graph();
        // MATCH (a:Person)-[:KNOWS]->(b)
        let plan = LogicalOperator::Expand {
            input: Box::new(LogicalOperator::ScanByLabel {
                variable: "a".to_string(),
                label: "Person".to_string(),
                properties: HashMap::new(),
            }),
            source_variable: "a".to_string(),
            target_variable: "b".to_string(),
            target_label: "Person".to_string(),
            relationship_types: vec!["KNOWS".to_string()],
            direction: crate::ast::RelationshipDirection::Outgoing,
            relationship_variable: None,
            properties: HashMap::new(),
            target_properties: HashMap::new(),
        };

        let result = compile_to_blasgraph(&plan, &graph, &HdrSemiring::XorBundle).unwrap();
        // Person diagonal × KNOWS should produce edges
        assert!(result.nnz() > 0);
    }

    #[test]
    fn test_planner_plus_truth_gate() {
        let graph = make_test_graph();
        let plan = LogicalOperator::ScanByLabel {
            variable: "a".to_string(),
            label: "Person".to_string(),
            properties: HashMap::new(),
        };
        let result = compile_to_blasgraph(&plan, &graph, &HdrSemiring::Boolean).unwrap();

        let mut truth_values = HashMap::new();
        truth_values.insert((0, 0), TruthValue::new(0.95, 0.95)); // Jan: strong
        truth_values.insert((1, 1), TruthValue::new(0.9, 0.9)); // Ada: strong
        truth_values.insert((2, 2), TruthValue::new(0.3, 0.2)); // Max: weak
        truth_values.insert((3, 3), TruthValue::new(0.7, 0.7)); // Eve: medium

        // STRONG gate filters weak edges
        let hits = apply_truth_gate(&result, TruthGate::STRONG, &truth_values);
        // Only Jan and Ada pass STRONG (expectation > 0.75)
        let passing_nodes: Vec<usize> = hits.iter().map(|h| h.source).collect();
        assert!(passing_nodes.contains(&0), "Jan should pass STRONG gate");
        assert!(passing_nodes.contains(&1), "Ada should pass STRONG gate");
        assert!(
            !passing_nodes.contains(&2),
            "Max should NOT pass STRONG gate"
        );
    }

    #[test]
    fn test_unknown_label() {
        let graph = make_test_graph();
        let plan = LogicalOperator::ScanByLabel {
            variable: "a".to_string(),
            label: "Robot".to_string(),
            properties: HashMap::new(),
        };
        let result = compile_to_blasgraph(&plan, &graph, &HdrSemiring::Boolean);
        assert!(result.is_err());
    }

    #[test]
    fn test_unknown_relation() {
        let graph = make_test_graph();
        let plan = LogicalOperator::Expand {
            input: Box::new(LogicalOperator::ScanByLabel {
                variable: "a".to_string(),
                label: "Person".to_string(),
                properties: HashMap::new(),
            }),
            source_variable: "a".to_string(),
            target_variable: "b".to_string(),
            target_label: "Person".to_string(),
            relationship_types: vec!["LIKES".to_string()],
            direction: crate::ast::RelationshipDirection::Outgoing,
            relationship_variable: None,
            properties: HashMap::new(),
            target_properties: HashMap::new(),
        };
        let result = compile_to_blasgraph(&plan, &graph, &HdrSemiring::XorBundle);
        assert!(result.is_err());
    }
}
