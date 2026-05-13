//! Strategy #16: GqlParse — ISO GQL (ISO/IEC 39075) → IR transpiler.
//!
//! GQL is the ISO standard for graph query languages. It's structurally similar
//! to Cypher (neo4j donated the syntax) but adds:
//! - Graph types and schema constraints
//! - Multi-graph queries (USE graph_name)
//! - OPTIONAL MATCH → LEFT MATCH
//! - Explicit path modes (WALK, TRAIL, ACYCLIC, SIMPLE)
//! - Composable graph patterns
//!
//! Since GQL and Cypher share 90% of syntax, this strategy reuses most of
//! CypherParse's feature detection and adds GQL-specific extensions.
//!
//! ## GQL-specific syntax → IR mapping
//!
//! | GQL syntax                        | IR node                                  |
//! |-----------------------------------|------------------------------------------|
//! | `MATCH (n:Person)`                | ScanNode { label: "Person" }             |
//! | `MATCH (a)-[:KNOWS]->(b)`         | IndexNestedLoopJoin                      |
//! | `LEFT MATCH`                      | HashJoin { join_type: Left }             |
//! | `MATCH ANY SHORTEST`              | ShortestPath                             |
//! | `MATCH ALL TRAIL`                 | RecursiveExtend (no repeated edges)      |
//! | `RETURN`                          | Return/Projection                        |
//! | `LET x = (...)`                   | (subquery binding)                       |
//! | `FILTER`                          | Filter                                   |
//! | `FOR .. IN .. RETURN`             | Aggregate + Projection                   |
//! | `ORDER BY .. LIMIT`               | OrderBy + Limit                          |
//! | `INSERT (n:Person {name: "Jan"})` | CreateNode                               |

use crate::ir::{Arena, LogicalOp};
use crate::traits::*;
use crate::PlanError;

#[derive(Debug)]
pub struct GqlParse;

impl PlanStrategy for GqlParse {
    fn name(&self) -> &str {
        "gql_parse"
    }
    fn capability(&self) -> PlanCapability {
        PlanCapability::Parse
    }

    fn affinity(&self, context: &PlanContext) -> f32 {
        let q = context.query.to_uppercase();
        // GQL-specific keywords that distinguish from plain Cypher
        let gql_signals = [
            "LEFT MATCH",
            "MANDATORY MATCH",
            "ANY SHORTEST",
            "ALL SHORTEST",
            "ANY CHEAPEST",
            "WALK",
            "TRAIL",
            "ACYCLIC",
            "SIMPLE",
            "GRAPH TYPE",
            "USE GRAPH",
            "LET ",
            "FOR ",
            "INSERT (",
            "DETACH DELETE",
        ];

        let gql_score: f32 = gql_signals.iter().filter(|kw| q.contains(*kw)).count() as f32 * 0.2;

        if gql_score > 0.0 {
            (0.7 + gql_score).min(0.98)
        } else if q.contains("MATCH") && q.contains("RETURN") {
            // Could be Cypher or GQL — we score slightly below CypherParse
            // so Cypher wins for plain queries. If GQL-specific syntax is
            // present above, we score higher.
            0.4
        } else {
            0.0
        }
    }

    fn plan(
        &self,
        mut input: PlanInput,
        _arena: &mut Arena<LogicalOp>,
    ) -> Result<PlanInput, PlanError> {
        let q = input.context.query.to_uppercase();

        // ── Core pattern detection (shared with Cypher) ──
        input.context.features.has_graph_pattern = q.contains("MATCH");
        input.context.features.has_fingerprint_scan =
            q.contains("HAMMING") || q.contains("FINGERPRINT") || q.contains("RESONATE");
        input.context.features.has_variable_length_path = q.contains("*..")
            || q.contains("*1..")
            || q.contains("*2..")
            || q.contains("TRAIL")
            || q.contains("WALK")
            || q.contains("ACYCLIC")
            || q.contains("SIMPLE");
        input.context.features.has_aggregation = q.contains("COUNT")
            || q.contains("SUM")
            || q.contains("AVG")
            || q.contains("COLLECT")
            || q.contains("FOR ");
        input.context.features.has_mutation = q.contains("INSERT")
            || q.contains("SET")
            || q.contains("DELETE")
            || q.contains("MERGE")
            || q.contains("REMOVE");
        input.context.features.has_resonance = q.contains("RESONATE");
        input.context.features.has_truth_values = q.contains("TRUTH") || q.contains("CONFIDENCE");
        input.context.features.has_workflow = q.contains("WORKFLOW") || q.contains("TASK");
        input.context.features.num_match_clauses = q.matches("MATCH").count();

        // ── GQL-specific features ──

        // Path modes (GQL 7.16)
        let has_path_mode = q.contains("ANY SHORTEST")
            || q.contains("ALL SHORTEST")
            || q.contains("ANY CHEAPEST")
            || q.contains("ALL TRAIL")
            || q.contains("ANY SIMPLE")
            || q.contains("ALL ACYCLIC");
        if has_path_mode {
            input.context.features.has_variable_length_path = true;
        }

        // LEFT MATCH (GQL's OPTIONAL MATCH)
        let has_left_match = q.contains("LEFT MATCH") || q.contains("MANDATORY MATCH");

        // LET bindings (GQL subquery composition)
        let has_let = q.contains("LET ");

        // Multi-graph (USE GRAPH)
        let _has_multi_graph = q.contains("USE GRAPH");

        // ── Complexity estimation ──
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
        if has_path_mode {
            complexity += 0.2;
        }
        if has_left_match {
            complexity += 0.1;
        }
        if has_let {
            complexity += 0.15;
        }
        input.context.features.estimated_complexity = complexity.min(1.0);

        // GQL AST production: delegate to CypherParse for shared syntax,
        // then layer GQL-specific rewrites. For now, feature detection is
        // the output — ArenaIR will build the plan from detected features.

        Ok(input)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gql_affinity_for_gql_query() {
        let parser = GqlParse;

        // GQL-specific query
        let ctx = PlanContext {
            query: "MATCH ANY SHORTEST (a:Person)-[:KNOWS]->{1,5}(b:Person) RETURN a, b".into(),
            features: QueryFeatures::default(),
            free_will_modifier: 1.0,
            thinking_style: None,
            nars_hint: None,
        };
        assert!(parser.affinity(&ctx) > 0.7);
    }

    #[test]
    fn test_gql_affinity_for_plain_cypher() {
        let parser = GqlParse;

        // Plain Cypher — GqlParse should score lower than CypherParse
        let ctx = PlanContext {
            query: "MATCH (n:Person) RETURN n".into(),
            features: QueryFeatures::default(),
            free_will_modifier: 1.0,
            thinking_style: None,
            nars_hint: None,
        };
        assert!(parser.affinity(&ctx) < 0.5);
    }

    #[test]
    fn test_gql_affinity_for_left_match() {
        let parser = GqlParse;
        let ctx = PlanContext {
            query: "MATCH (a:Person) LEFT MATCH (a)-[:KNOWS]->(b) RETURN a, b".into(),
            features: QueryFeatures::default(),
            free_will_modifier: 1.0,
            thinking_style: None,
            nars_hint: None,
        };
        assert!(parser.affinity(&ctx) > 0.7);
    }

    #[test]
    fn test_gql_detects_path_modes() {
        let parser = GqlParse;
        let mut arena = Arena::new();
        let input = PlanInput {
            plan: None,
            context: PlanContext {
                query: "MATCH ALL TRAIL (a)-[:KNOWS]->{1,10}(b) RETURN a, b".into(),
                features: QueryFeatures::default(),
                free_will_modifier: 1.0,
                thinking_style: None,
                nars_hint: None,
            },
        };

        let result = parser.plan(input, &mut arena).unwrap();
        assert!(result.context.features.has_variable_length_path);
        assert!(result.context.features.estimated_complexity > 0.4);
    }

    #[test]
    fn test_gql_detects_let_bindings() {
        let parser = GqlParse;
        let mut arena = Arena::new();
        let input = PlanInput {
            plan: None,
            context: PlanContext {
                query:
                    "LET friends = (MATCH (a)-[:KNOWS]->(b) RETURN b) MATCH (f) IN friends RETURN f"
                        .into(),
                features: QueryFeatures::default(),
                free_will_modifier: 1.0,
                thinking_style: None,
                nars_hint: None,
            },
        };

        let result = parser.plan(input, &mut arena).unwrap();
        assert!(result.context.features.has_graph_pattern);
    }

    #[test]
    fn test_gql_zero_affinity_for_gremlin() {
        let parser = GqlParse;
        let ctx = PlanContext {
            query: "g.V().hasLabel(\"Person\").out(\"KNOWS\")".into(),
            features: QueryFeatures::default(),
            free_will_modifier: 1.0,
            thinking_style: None,
            nars_hint: None,
        };
        assert!(parser.affinity(&ctx) < 0.1);
    }
}
