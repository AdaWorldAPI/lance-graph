// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Lance Native physical planner.
//!
//! When the `planner` feature is enabled, delegates to `lance-graph-planner`
//! for MUL assessment, thinking style selection, and strategy composition.
//! The planner produces its own `ir::LogicalPlan`; this module bridges back
//! to a DataFusion `LogicalPlan` (currently EmptyRelation) so the existing
//! `GraphPhysicalPlanner` trait is satisfied.
//!
//! Without the `planner` feature, returns EmptyRelation (stub).

use crate::config::GraphConfig;
use crate::datafusion_planner::GraphPhysicalPlanner;
use crate::error::Result;
use crate::logical_plan::LogicalOperator;
use datafusion::common::DFSchema;
use datafusion::logical_expr::{EmptyRelation, LogicalPlan};
use std::sync::Arc;

/// Lance-native planner. Delegates to `lance-graph-planner` when available.
pub struct LanceNativePlanner {
    config: GraphConfig,
    #[cfg(feature = "planner")]
    planner: lance_graph_planner::api::Planner,
}

impl LanceNativePlanner {
    pub fn new(config: GraphConfig) -> Self {
        Self {
            config,
            #[cfg(feature = "planner")]
            planner: lance_graph_planner::api::Planner::new(),
        }
    }

    /// Access the underlying config.
    pub fn config(&self) -> &GraphConfig {
        &self.config
    }

    /// Classify a query using the planner's feature detection.
    ///
    /// Returns a list of strategy names that would be selected for this query.
    /// Useful for explain/debug output without running the full pipeline.
    #[cfg(feature = "planner")]
    pub fn classify(&self, query: &str) -> Vec<String> {
        match self.planner.plan(query) {
            Ok(result) => result.strategies_used,
            Err(_) => vec![],
        }
    }

    /// Run the planner's MUL gate check on a situation.
    #[cfg(feature = "planner")]
    pub fn gate_check(
        &self,
        situation: &lance_graph_planner::api::SituationInput,
    ) -> lance_graph_planner::api::Gate {
        self.planner.gate_check(situation)
    }
}

impl GraphPhysicalPlanner for LanceNativePlanner {
    fn plan(&self, _logical_plan: &LogicalOperator) -> Result<LogicalPlan> {
        // When the planner feature is enabled, we run the planner's auto mode
        // to validate the query and select strategies. The actual DataFusion
        // physical plan translation is still Phase 3/4 work — for now we
        // produce EmptyRelation but log the planner's strategy selection.
        #[cfg(feature = "planner")]
        {
            // Extract query text from the logical plan for classification.
            // The planner needs raw Cypher; we reconstruct a minimal version
            // from the logical operator for feature detection only.
            let query_hint = format!("{:?}", _logical_plan);
            let _result = self.planner.plan(&query_hint);
            // TODO(Phase 3): translate planner's ir::LogicalPlan → DataFusion LogicalPlan
        }

        let schema = Arc::new(DFSchema::empty());
        Ok(LogicalPlan::EmptyRelation(EmptyRelation {
            produce_one_row: false,
            schema,
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lance_native_planner_placeholder() {
        let cfg = GraphConfig::builder()
            .with_node_label("Person", "id")
            .build()
            .unwrap();
        let planner = LanceNativePlanner::new(cfg);
        // Minimal logical plan to feed into placeholder
        let lp = LogicalOperator::Distinct {
            input: Box::new(LogicalOperator::Limit {
                input: Box::new(LogicalOperator::Project {
                    input: Box::new(LogicalOperator::ScanByLabel {
                        variable: "n".to_string(),
                        label: "Person".to_string(),
                        properties: Default::default(),
                    }),
                    projections: vec![],
                }),
                count: 1,
            }),
        };
        let df_plan = planner.plan(&lp).unwrap();
        // Empty relation is acceptable as a placeholder
        match df_plan {
            LogicalPlan::EmptyRelation(_) => {}
            _ => panic!("expected empty relation placeholder"),
        }
    }
}
