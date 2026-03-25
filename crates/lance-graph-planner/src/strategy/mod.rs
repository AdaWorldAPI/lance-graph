//! Strategy registry: all 16 composable planning strategies.
//!
//! | # | Strategy          | Capability           | Source               |
//! |---|-------------------|---------------------|----------------------|
//! | 1 | CypherParse       | Parse               | lance-graph nom      |
//! | 2 | ArenaIR           | LogicalPlan         | Polars arena pattern |
//! | 3 | DPJoinEnum        | JoinOrdering        | Kuzudb DP            |
//! | 4 | RuleOptimizer     | RuleOptimization    | DataFusion           |
//! | 5 | HistogramCost     | CostEstimation      | Hyrise               |
//! | 6 | SigmaBandScan     | VectorScan          | lance-graph          |
//! | 7 | MorselExec        | PhysicalPlan        | Kuzudb/Polars        |
//! | 8 | StreamPipeline    | StreamExecution      | Polars               |
//! | 9 | TruthPropagation  | TruthPropagation    | lance-graph semiring |
//! |10 | CollapseGate      | ResonanceGating     | agi-chat             |
//! |11 | JitCompile        | JitCompilation      | ndarray              |
//! |12 | WorkflowDAG       | WorkflowOrchestration | LangGraph          |
//! |13 | ExtensionPlanner  | Extension           | DataFusion           |
//! |14 | GremlinParse      | Parse               | TinkerPop Gremlin    |
//! |15 | SparqlParse       | Parse               | W3C SPARQL           |
//! |16 | GqlParse          | Parse               | ISO GQL (39075)      |

pub mod cypher_parse;
pub mod gremlin_parse;
pub mod sparql_parse;
pub mod gql_parse;
pub mod arena_ir;
pub mod dp_join;
pub mod rule_optimizer;
pub mod histogram_cost;
pub mod sigma_scan;
pub mod morsel_exec;
pub mod stream_pipeline;
pub mod truth_propagation;
pub mod collapse_gate;
pub mod jit_compile;
pub mod workflow_dag;
pub mod extension;

use crate::traits::PlanStrategy;

/// Create the default strategy registry with all 16 strategies.
///
/// Parse strategies are all registered — the affinity system ensures only
/// the correct parser fires for each query language. For example:
/// - `g.V().hasLabel(...)` → GremlinParse scores 0.95, others score 0.0
/// - `MATCH (n) RETURN n` → CypherParse scores 0.95, GqlParse scores 0.4
/// - `SELECT ?s WHERE { ... }` → SparqlParse scores 0.95, others score 0.0
/// - `MATCH ANY SHORTEST ...` → GqlParse scores 0.9, CypherParse scores 0.95 (both run)
pub fn default_strategies() -> Vec<Box<dyn PlanStrategy>> {
    vec![
        // Parse phase: polyglot parsers (affinity selects the right one)
        Box::new(cypher_parse::CypherParse),
        Box::new(gremlin_parse::GremlinParse),
        Box::new(sparql_parse::SparqlParse),
        Box::new(gql_parse::GqlParse),
        // Plan phase
        Box::new(arena_ir::ArenaIR),
        Box::new(dp_join::DPJoinEnum),
        // Optimize phase
        Box::new(rule_optimizer::RuleOptimizer),
        Box::new(histogram_cost::HistogramCost),
        // Physicalize phase
        Box::new(sigma_scan::SigmaBandScan),
        Box::new(morsel_exec::MorselExec),
        Box::new(truth_propagation::TruthPropagation),
        Box::new(collapse_gate::CollapseGateStrategy),
        // Execute phase
        Box::new(stream_pipeline::StreamPipeline),
        Box::new(jit_compile::JitCompile),
        // Cross-cutting
        Box::new(workflow_dag::WorkflowDAG),
        Box::new(extension::ExtensionPlanner),
    ]
}
