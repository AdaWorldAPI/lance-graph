//! Strategy #1: CypherParse — Lexical feature detection over the raw query string.
//!
//! ## Architectural note
//!
//! Earlier drafts of this strategy promised to "call lance-graph's
//! `parser::parse_cypher_query()` to produce a full AST". That path is
//! **blocked by the dependency graph**: `lance-graph` already depends on
//! `lance-graph-planner` (optional, behind the `planner` feature), so adding
//! `lance-graph` as a dependency of `lance-graph-planner` would create a
//! Cargo cycle.
//!
//! Two real unblock paths exist for future work (tracked as F-tasks):
//!
//! 1. **Parser extraction**: lift `crates/lance-graph/src/parser.rs` into a
//!    zero-dep crate (e.g. `lance-graph-cypher`) that both `lance-graph` and
//!    `lance-graph-planner` depend on. Highest leverage; unblocks all
//!    parser-touching strategies (CypherParse, GqlParse, future Sparql).
//!
//! 2. **AST handoff via context**: define a trait/AST type in the existing
//!    zero-dep `lance-graph-contract` crate, have `lance-graph` parse and
//!    attach the parsed AST to `PlanContext` before invoking the planner.
//!    `CypherParse::plan` would then transcode AST → `LogicalOp` arena
//!    instead of re-parsing from text.
//!
//! Until one of those lands, this strategy does **lexical** feature detection
//! only (keyword scanning on the uppercased query string). That is enough to
//! drive strategy affinity scoring downstream — it is **not** a real parser.

use crate::ir::{Arena, LogicalOp};
use crate::traits::*;
use crate::PlanError;

#[derive(Debug)]
pub struct CypherParse;

/// Extract `QueryFeatures` from the raw query text by lexical scanning.
///
/// Single source of truth for the feature-detection used by both
/// `CypherParse::plan` and `PlannerAwareness::plan_auto`. Keyword-based,
/// case-insensitive on the uppercased query string.
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

impl PlanStrategy for CypherParse {
    fn name(&self) -> &str {
        "cypher_parse"
    }
    fn capability(&self) -> PlanCapability {
        PlanCapability::Parse
    }

    fn affinity(&self, context: &PlanContext) -> f32 {
        let q = context.query.to_uppercase();
        if q.contains("MATCH") || q.contains("CREATE") || q.contains("RETURN") {
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
        input.context.features = extract_features(&input.context.query);
        Ok(input)
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
    fn cypher_parse_strategy_populates_features_on_plan() {
        let strategy = CypherParse;
        let context = PlanContext {
            query: "MATCH (n) RETURN n".into(),
            features: QueryFeatures::default(),
            free_will_modifier: 1.0,
            thinking_style: None,
            nars_hint: None,
        };
        let input = PlanInput {
            plan: None,
            context,
        };
        let mut arena = Arena::<LogicalOp>::new();
        let out = strategy.plan(input, &mut arena).unwrap();
        assert!(out.context.features.has_graph_pattern);
        assert_eq!(out.context.features.num_match_clauses, 1);
    }
}
