//! Strategy #1: CypherParse — Intent parsing via lance-graph's nom parser.

use crate::ir::{Arena, LogicalOp};
use crate::traits::*;
use crate::PlanError;

#[derive(Debug)]
pub struct CypherParse;

impl PlanStrategy for CypherParse {
    fn name(&self) -> &str {
        "cypher_parse"
    }
    fn capability(&self) -> PlanCapability {
        PlanCapability::Parse
    }

    fn affinity(&self, context: &PlanContext) -> f32 {
        // Always high affinity — every query needs parsing
        if context.query.to_uppercase().contains("MATCH")
            || context.query.to_uppercase().contains("CREATE")
            || context.query.to_uppercase().contains("RETURN")
        {
            0.95
        } else {
            0.5
        }
    }

    fn plan(
        &self,
        mut input: PlanInput,
        _arena: &mut Arena<LogicalOp>,
    ) -> Result<PlanInput, PlanError> {
        let q = input.context.query.to_uppercase();

        // Detect query features from syntax
        input.context.features.has_graph_pattern = q.contains("MATCH");
        input.context.features.has_fingerprint_scan =
            q.contains("HAMMING") || q.contains("FINGERPRINT") || q.contains("RESONATE");
        input.context.features.has_variable_length_path =
            q.contains("*..") || q.contains("*1..") || q.contains("*2..");
        input.context.features.has_aggregation =
            q.contains("COUNT") || q.contains("SUM") || q.contains("AVG") || q.contains("COLLECT");
        input.context.features.has_mutation = q.contains("CREATE")
            || q.contains("SET")
            || q.contains("DELETE")
            || q.contains("MERGE");
        input.context.features.has_resonance = q.contains("RESONATE");
        input.context.features.has_truth_values = q.contains("TRUTH") || q.contains("CONFIDENCE");
        input.context.features.has_workflow = q.contains("WORKFLOW") || q.contains("TASK");
        input.context.features.num_match_clauses = q.matches("MATCH").count();

        // Estimate complexity from detected features
        let mut complexity = input.context.features.num_match_clauses as f64 * 0.2;
        if input.context.features.has_variable_length_path {
            complexity += 0.3;
        }
        if input.context.features.has_fingerprint_scan {
            complexity += 0.2;
        }
        if input.context.features.has_aggregation {
            complexity += 0.1;
        }
        input.context.features.estimated_complexity = complexity.min(1.0);

        // Real implementation: call lance-graph's parser::parse_cypher_query()
        // to produce a full AST. For now, feature detection is the output.

        Ok(input)
    }
}

/// Additive helper — single-source-of-truth lexical feature extraction over
/// a Cypher-shaped query string. Mirrors the inline block in
/// `CypherParse::plan` and `PlannerAwareness::plan_auto` so future callers
/// have one canonical entry point. Existing call sites are intentionally
/// left untouched until a dedup refactor is approved.
pub fn extract_features(query: &str) -> QueryFeatures {
    let q = query.to_uppercase();

    let has_graph_pattern = q.contains("MATCH");
    let has_fingerprint_scan =
        q.contains("HAMMING") || q.contains("FINGERPRINT") || q.contains("RESONATE");
    let has_variable_length_path =
        q.contains("*..") || q.contains("*1..") || q.contains("*2..");
    let has_aggregation = q.contains("COUNT")
        || q.contains("SUM")
        || q.contains("AVG")
        || q.contains("COLLECT");
    let has_mutation =
        q.contains("CREATE") || q.contains("SET") || q.contains("DELETE") || q.contains("MERGE");
    let has_resonance = q.contains("RESONATE");
    let has_truth_values = q.contains("TRUTH") || q.contains("CONFIDENCE");
    let has_workflow = q.contains("WORKFLOW") || q.contains("TASK");
    let num_match_clauses = q.matches("MATCH").count();

    let mut complexity = num_match_clauses as f64 * 0.2;
    if has_variable_length_path {
        complexity += 0.3;
    }
    if has_fingerprint_scan {
        complexity += 0.2;
    }
    if has_aggregation {
        complexity += 0.1;
    }

    QueryFeatures {
        has_graph_pattern,
        has_fingerprint_scan,
        has_variable_length_path,
        has_aggregation,
        has_mutation,
        has_workflow,
        has_resonance,
        has_truth_values,
        num_match_clauses,
        num_nodes: 0,
        num_edges: 0,
        estimated_complexity: complexity.min(1.0),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extract_features_detects_match_and_aggregation() {
        let f = extract_features("MATCH (n) RETURN count(n)");
        assert!(f.has_graph_pattern);
        assert!(f.has_aggregation);
        assert_eq!(f.num_match_clauses, 1);
    }

    #[test]
    fn extract_features_counts_match_clauses() {
        let f = extract_features("MATCH (a) MATCH (b) MATCH (c) RETURN a, b, c");
        assert_eq!(f.num_match_clauses, 3);
    }

    #[test]
    fn extract_features_detects_resonance_and_fingerprint() {
        let f = extract_features("MATCH (n) WHERE RESONATE(n.fp, $q, 0.7) RETURN n");
        assert!(f.has_resonance);
        assert!(f.has_fingerprint_scan);
    }

    #[test]
    fn extract_features_complexity_caps_at_one() {
        let f = extract_features(
            "MATCH (a)-[*..5]->(b) MATCH (c)-[*..5]->(d) MATCH (e)-[*..5]->(f) \
             WHERE RESONATE(a.fp, $q, 0.5) RETURN count(*)",
        );
        assert!(f.estimated_complexity <= 1.0);
        assert!(f.estimated_complexity > 0.5);
    }

    #[test]
    fn extract_features_matches_cypher_parse_strategy_output() {
        // Equivalence check: the additive helper must produce the same
        // QueryFeatures the existing CypherParse::plan would write.
        let strategy = CypherParse;
        let context = PlanContext {
            query: "MATCH (n) WHERE RESONATE(n.fp, $q, 0.5) RETURN count(n)".into(),
            features: QueryFeatures::default(),
            free_will_modifier: 1.0,
            thinking_style: None,
            nars_hint: None,
        };
        let input = PlanInput {
            plan: None,
            context: context.clone(),
        };
        let mut arena = Arena::<LogicalOp>::new();
        let out = strategy.plan(input, &mut arena).unwrap();
        let inline = out.context.features;
        let helper = extract_features(&context.query);

        assert_eq!(inline.has_graph_pattern, helper.has_graph_pattern);
        assert_eq!(inline.has_fingerprint_scan, helper.has_fingerprint_scan);
        assert_eq!(
            inline.has_variable_length_path,
            helper.has_variable_length_path
        );
        assert_eq!(inline.has_aggregation, helper.has_aggregation);
        assert_eq!(inline.has_mutation, helper.has_mutation);
        assert_eq!(inline.has_workflow, helper.has_workflow);
        assert_eq!(inline.has_resonance, helper.has_resonance);
        assert_eq!(inline.has_truth_values, helper.has_truth_values);
        assert_eq!(inline.num_match_clauses, helper.num_match_clauses);
        assert!(
            (inline.estimated_complexity - helper.estimated_complexity).abs() < 1e-9
        );
    }
}
