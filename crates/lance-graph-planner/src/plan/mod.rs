//! # Query Planner (Inner Loop)
//!
//! DP-based join order enumeration with factorization-aware plan space (from Kuzudb).
//! The thinking context controls planner behavior: fan-out, depth, thresholds.

mod dp_enumerator;
mod cost;

pub use dp_enumerator::DpEnumerator;
pub use cost::CostModel;

use crate::ir::{Arena, LogicalOp, LogicalPlan, Node, SubPlansTable, SubqueryGraph};
use crate::ir::expr::{AExpr, ExprNode};
use crate::ir::schema::Schema;
use crate::ir::properties::PlanProperties;
use crate::thinking::ThinkingContext;
use crate::PlanError;

/// Planner configuration.
#[derive(Debug, Clone)]
pub struct PlannerConfig {
    /// Max subgraph nodes to enumerate exactly (Kuzudb: 7).
    pub max_level_exact: usize,
    /// Max plans per subgraph (Kuzudb: 10).
    pub max_plans_per_subgraph: usize,
    /// Max total subgraphs in DP table (Kuzudb: 50).
    pub max_subgraphs: usize,
    /// Whether to consider WCO joins.
    pub enable_wco_joins: bool,
    /// Whether to consider SIP (sideways information passing).
    pub enable_sip: bool,
    /// Default scan strategy for resonance queries.
    pub default_scan_strategy: crate::ir::logical_op::ScanStrategy,
}

impl Default for PlannerConfig {
    fn default() -> Self {
        Self {
            max_level_exact: 7,
            max_plans_per_subgraph: 10,
            max_subgraphs: 50,
            enable_wco_joins: true,
            enable_sip: true,
            default_scan_strategy: crate::ir::logical_op::ScanStrategy::Cascade,
        }
    }
}

/// A node in the query graph (extracted from parsed Cypher).
#[derive(Debug, Clone)]
pub struct QueryGraphNode {
    pub id: usize,
    pub label: String,
    pub alias: String,
    pub predicates: Vec<ExprNode>,
}

/// An edge in the query graph.
#[derive(Debug, Clone)]
pub struct QueryGraphEdge {
    pub id: usize,
    pub rel_type: String,
    pub alias: String,
    pub src_id: usize,
    pub dst_id: usize,
    pub direction: crate::ir::logical_op::Direction,
    pub predicates: Vec<ExprNode>,
}

/// The query graph: nodes + edges + predicates.
#[derive(Debug, Clone)]
pub struct QueryGraph {
    pub nodes: Vec<QueryGraphNode>,
    pub edges: Vec<QueryGraphEdge>,
    /// Predicates that span multiple nodes (join predicates).
    pub join_predicates: Vec<ExprNode>,
}

/// Plan a query using the thinking context.
///
/// This is the main entry point for the inner loop.
/// 1. Parse query into QueryGraph (simplified — real parser is in lance-graph core)
/// 2. Run DP enumeration with thinking-informed parameters
/// 3. Return the best logical plan
pub fn plan_query(
    query: &str,
    thinking: &ThinkingContext,
    arena: &mut Arena<LogicalOp>,
    config: &PlannerConfig,
) -> Result<LogicalPlan, PlanError> {
    // Parse into query graph (simplified for now — delegates to lance-graph parser)
    let query_graph = parse_to_query_graph(query, arena)?;

    if query_graph.nodes.is_empty() {
        // Empty query → EmptyResult
        let expr_arena = crate::ir::Arena::new();
        let root = arena.push(LogicalOp::EmptyResult);
        return Ok(LogicalPlan::new(
            std::mem::replace(arena, Arena::new()),
            expr_arena,
            root,
        ));
    }

    // Check if this is a resonance query (BROADCAST → SCAN → ACCUMULATE → COLLAPSE)
    let is_resonance = query.to_uppercase().contains("RESONATE")
        || query.to_uppercase().contains("HAMMING");

    if is_resonance {
        return plan_resonance_query(query, thinking, arena, config);
    }

    // Standard graph query: DP join enumeration
    let enumerator = DpEnumerator::new(config, thinking);
    let expr_arena = crate::ir::Arena::new();
    let root = enumerator.enumerate(&query_graph, arena)?;

    Ok(LogicalPlan::new(
        std::mem::replace(arena, Arena::new()),
        expr_arena,
        root,
    ))
}

/// Plan a resonance query: BROADCAST → SCAN → ACCUMULATE → COLLAPSE.
fn plan_resonance_query(
    query: &str,
    thinking: &ThinkingContext,
    arena: &mut Arena<LogicalOp>,
    config: &PlannerConfig,
) -> Result<LogicalPlan, PlanError> {
    let expr_arena = crate::ir::Arena::new();
    let scan_params = thinking.modulation.to_scan_params();

    // BROADCAST: distribute query fingerprint
    let broadcast = arena.push(LogicalOp::Broadcast {
        fingerprint: ExprNode(crate::ir::Node(0)), // Placeholder
        partitions: thinking.modulation.fan_out.max(1),
    });

    // SCAN: vectorized Hamming distance
    let scan = arena.push(LogicalOp::Scan {
        input: broadcast,
        strategy: config.default_scan_strategy,
        threshold: scan_params.threshold,
        top_k: scan_params.top_k as usize,
    });

    // ACCUMULATE: propagate with selected semiring
    let accumulate = arena.push(LogicalOp::Accumulate {
        input: scan,
        semiring: thinking.semiring.semiring,
        traversal: scan, // Self-reference for leaf accumulation
    });

    // COLLAPSE: apply gate thresholds
    let collapse = arena.push(LogicalOp::Collapse {
        input: accumulate,
        gate: crate::ir::logical_op::CollapseGate::default(),
    });

    Ok(LogicalPlan::new(
        std::mem::replace(arena, Arena::new()),
        expr_arena,
        collapse,
    ))
}

/// Simplified query graph extraction (real parser is in lance-graph core crate).
fn parse_to_query_graph(
    query: &str,
    arena: &mut Arena<LogicalOp>,
) -> Result<QueryGraph, PlanError> {
    // This is a simplified parser for the planner crate.
    // The real parser (66KB nom-based) lives in lance-graph core.
    // Here we extract enough structure for DP enumeration.

    let q = query.to_uppercase();

    // Count MATCH patterns to estimate query graph size
    let match_count = q.matches("MATCH").count();

    if match_count == 0 {
        return Ok(QueryGraph {
            nodes: Vec::new(),
            edges: Vec::new(),
            join_predicates: Vec::new(),
        });
    }

    // For now, produce a minimal single-node query graph.
    // The real integration will receive a QueryGraph from lance-graph's parser.
    Ok(QueryGraph {
        nodes: vec![QueryGraphNode {
            id: 0,
            label: "Unknown".into(),
            alias: "n".into(),
            predicates: Vec::new(),
        }],
        edges: Vec::new(),
        join_predicates: Vec::new(),
    })
}
