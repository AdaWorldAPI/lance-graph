//! Strategy registry: all 13 composable planning strategies.
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

pub mod cypher_parse;
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

/// Create the default strategy registry with all 13 strategies.
pub fn default_strategies() -> Vec<Box<dyn PlanStrategy>> {
    vec![
        Box::new(cypher_parse::CypherParse),
        Box::new(arena_ir::ArenaIR),
        Box::new(dp_join::DPJoinEnum),
        Box::new(rule_optimizer::RuleOptimizer),
        Box::new(histogram_cost::HistogramCost),
        Box::new(sigma_scan::SigmaBandScan),
        Box::new(morsel_exec::MorselExec),
        Box::new(stream_pipeline::StreamPipeline),
        Box::new(truth_propagation::TruthPropagation),
        Box::new(collapse_gate::CollapseGateStrategy),
        Box::new(jit_compile::JitCompile),
        Box::new(workflow_dag::WorkflowDAG),
        Box::new(extension::ExtensionPlanner),
    ]
}
