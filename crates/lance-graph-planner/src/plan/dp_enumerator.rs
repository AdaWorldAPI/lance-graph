//! DP-based join order enumeration (from Kuzudb).
//!
//! Enumerates subgraph plans level by level (edge-at-a-time).
//! At each level considers:
//! 1. INL joins (extend through adjacency list)
//! 2. Hash joins (probe/build with flipped sides)
//! 3. WCO joins (multi-way intersection)
//!
//! Plans above max_level_exact fall back to greedy left-deep enumeration.
//! Keeps up to max_plans_per_subgraph plans per subgraph,
//! differentiated by factorization encoding.

#[allow(unused_imports)] // Direction intended for edge traversal wiring
use crate::ir::logical_op::{Direction, JoinType};
#[allow(unused_imports)] // intended for factorization schema wiring
use crate::ir::schema::Schema;
use crate::ir::{Arena, LogicalOp, LogicalPlanRef, Node, SubPlansTable, SubqueryGraph};
use crate::plan::{CostModel, PlannerConfig, QueryGraph, QueryGraphEdge, QueryGraphNode};
use crate::thinking::ThinkingContext;
use crate::PlanError;

pub struct DpEnumerator<'a> {
    config: &'a PlannerConfig,
    #[allow(dead_code)] // future wiring for thinking-aware join enumeration
    thinking: &'a ThinkingContext,
    cost_model: CostModel,
}

impl<'a> DpEnumerator<'a> {
    pub fn new(config: &'a PlannerConfig, thinking: &'a ThinkingContext) -> Self {
        Self {
            config,
            thinking,
            cost_model: CostModel::new(thinking),
        }
    }

    /// Enumerate join orders for the query graph. Returns the root node in the arena.
    pub fn enumerate(
        &self,
        graph: &QueryGraph,
        arena: &mut Arena<LogicalOp>,
    ) -> Result<Node, PlanError> {
        let num_nodes = graph.nodes.len();

        if num_nodes == 0 {
            return Ok(arena.push(LogicalOp::EmptyResult));
        }

        // Single node: just scan
        if num_nodes == 1 {
            return Ok(self.plan_single_node(&graph.nodes[0], arena));
        }

        // Initialize DP table
        let mut dp = SubPlansTable::new();

        // Level 1: single-node scans
        for node in &graph.nodes {
            let sg = SubqueryGraph::singleton(node.id);
            let scan_node = self.plan_single_node(node, arena);
            let cost = self.cost_model.scan_cost(node);

            dp.add_plan(
                sg,
                LogicalPlanRef {
                    root: scan_node,
                    cost,
                    factorization_encoding: 0, // Single node, no factorization choice
                },
            );
        }

        // Level 2+: enumerate edge-at-a-time
        let max_level = if num_nodes <= self.config.max_level_exact {
            num_nodes
        } else {
            self.config.max_level_exact
        };

        for level in 2..=max_level {
            self.enumerate_level(level, graph, arena, &mut dp)?;
        }

        // If query is larger than max_level_exact, use greedy extension
        if num_nodes > self.config.max_level_exact {
            self.greedy_extend(graph, arena, &mut dp)?;
        }

        // Find the best plan for the full query graph
        let full_sg = SubqueryGraph((1u64 << num_nodes) - 1);
        dp.best_plan(full_sg)
            .map(|p| p.root)
            .ok_or_else(|| PlanError::Plan("No valid plan found for full query graph".into()))
    }

    /// Enumerate plans at a specific DP level.
    fn enumerate_level(
        &self,
        level: usize,
        graph: &QueryGraph,
        arena: &mut Arena<LogicalOp>,
        dp: &mut SubPlansTable,
    ) -> Result<(), PlanError> {
        let num_nodes = graph.nodes.len();
        let full_mask = (1u64 << num_nodes) - 1;

        // For each subset of size `level`
        for mask in 1..=full_mask {
            let sg = SubqueryGraph(mask);
            if sg.count() as usize != level {
                continue;
            }

            // Try all ways to split this subgraph into two connected parts
            for left_mask in sg.subsets() {
                let left = SubqueryGraph(left_mask.0);
                let right = SubqueryGraph(sg.0 & !left.0);

                if right.0 == 0 || left.count() >= sg.count() {
                    continue;
                }

                // Check that both sides have plans
                let left_plans: Vec<_> = dp
                    .get(left)
                    .map(|sp| sp.plans().cloned().collect())
                    .unwrap_or_default();
                let right_plans: Vec<_> = dp
                    .get(right)
                    .map(|sp| sp.plans().cloned().collect())
                    .unwrap_or_default();

                if left_plans.is_empty() || right_plans.is_empty() {
                    continue;
                }

                // Check if there's a connecting edge
                let connecting_edge = self.find_connecting_edge(graph, left, right);

                for lp in &left_plans {
                    for rp in &right_plans {
                        // Option 1: Hash join (if there's a connecting edge with equality)
                        if let Some(edge) = &connecting_edge {
                            let join_node = arena.push(LogicalOp::HashJoin {
                                left: lp.root,
                                right: rp.root,
                                join_keys: vec![(
                                    graph.nodes[edge.src_id].alias.clone(),
                                    graph.nodes[edge.dst_id].alias.clone(),
                                )],
                                join_type: JoinType::Inner,
                            });

                            let cost = self.cost_model.hash_join_cost(lp.cost, rp.cost);
                            let encoding = lp.factorization_encoding | rp.factorization_encoding;

                            dp.add_plan(
                                sg,
                                LogicalPlanRef {
                                    root: join_node,
                                    cost,
                                    factorization_encoding: encoding,
                                },
                            );

                            // Also try flipped sides
                            let flipped = arena.push(LogicalOp::HashJoin {
                                left: rp.root,
                                right: lp.root,
                                join_keys: vec![(
                                    graph.nodes[edge.dst_id].alias.clone(),
                                    graph.nodes[edge.src_id].alias.clone(),
                                )],
                                join_type: JoinType::Inner,
                            });

                            let flipped_cost = self.cost_model.hash_join_cost(rp.cost, lp.cost);

                            dp.add_plan(
                                sg,
                                LogicalPlanRef {
                                    root: flipped,
                                    cost: flipped_cost,
                                    factorization_encoding: encoding,
                                },
                            );
                        }

                        // Option 2: INL join (if one side is single edge scan)
                        if connecting_edge.is_some() && right.count() == 1 {
                            if let Some(edge) = &connecting_edge {
                                let inl = arena.push(LogicalOp::IndexNestedLoopJoin {
                                    left: lp.root,
                                    rel_type: edge.rel_type.clone(),
                                    direction: edge.direction,
                                    dst_alias: graph.nodes[edge.dst_id].alias.clone(),
                                });

                                let cost = self.cost_model.inl_join_cost(lp.cost);

                                dp.add_plan(
                                    sg,
                                    LogicalPlanRef {
                                        root: inl,
                                        cost,
                                        factorization_encoding: lp.factorization_encoding,
                                    },
                                );
                            }
                        }
                    }
                }

                // Option 3: WCO join (multi-way intersection)
                if self.config.enable_wco_joins && level >= 3 {
                    self.try_wco_join(sg, graph, arena, dp);
                }
            }
        }

        Ok(())
    }

    /// Try to create a WCO (worst-case optimal) join.
    /// Detects star patterns where multiple relationships converge on one node.
    fn try_wco_join(
        &self,
        sg: SubqueryGraph,
        graph: &QueryGraph,
        arena: &mut Arena<LogicalOp>,
        dp: &mut SubPlansTable,
    ) {
        // For each node in the subgraph, check if it's a convergence point
        for node in &graph.nodes {
            if sg.contains(SubqueryGraph::singleton(node.id)) {
                // Count edges in the subgraph that touch this node
                let touching_edges: Vec<&QueryGraphEdge> = graph
                    .edges
                    .iter()
                    .filter(|e| {
                        (e.src_id == node.id || e.dst_id == node.id)
                            && sg.contains(SubqueryGraph::singleton(e.src_id))
                            && sg.contains(SubqueryGraph::singleton(e.dst_id))
                    })
                    .collect();

                // WCO requires at least 2 edges converging on one node
                if touching_edges.len() >= 2 {
                    // Build separate scan plans for each edge
                    let mut children = Vec::new();
                    let mut total_cost = 0.0;

                    for edge in &touching_edges {
                        let other_id = if edge.src_id == node.id {
                            edge.dst_id
                        } else {
                            edge.src_id
                        };
                        let other_sg = SubqueryGraph::singleton(other_id);

                        if let Some(plan) = dp.best_plan(other_sg) {
                            children.push(plan.root);
                            total_cost += plan.cost;
                        }
                    }

                    if children.len() >= 2 {
                        let wco = arena.push(LogicalOp::WcoJoin {
                            intersect_alias: node.alias.clone(),
                            children,
                        });

                        let cost = self
                            .cost_model
                            .wco_join_cost(total_cost, touching_edges.len());

                        dp.add_plan(
                            sg,
                            LogicalPlanRef {
                                root: wco,
                                cost,
                                factorization_encoding: 0,
                            },
                        );
                    }
                }
            }
        }
    }

    /// Greedy left-deep extension for large queries (> max_level_exact nodes).
    fn greedy_extend(
        &self,
        graph: &QueryGraph,
        arena: &mut Arena<LogicalOp>,
        dp: &mut SubPlansTable,
    ) -> Result<(), PlanError> {
        let num_nodes = graph.nodes.len();
        let full_sg = SubqueryGraph((1u64 << num_nodes) - 1);

        // Start from the best plan at the max exact level
        let max_exact = self.config.max_level_exact;
        let mut best_mask = 0u64;
        let mut best_cost = f64::MAX;

        // Find the best subgraph of max_exact size
        for mask in 1..=(1u64 << num_nodes) - 1 {
            let sg = SubqueryGraph(mask);
            if sg.count() as usize == max_exact {
                if let Some(plan) = dp.best_plan(sg) {
                    if plan.cost < best_cost {
                        best_cost = plan.cost;
                        best_mask = mask;
                    }
                }
            }
        }

        if best_mask == 0 {
            return Err(PlanError::Plan(
                "No base plan found for greedy extension".into(),
            ));
        }

        let mut current_sg = SubqueryGraph(best_mask);

        // Greedily add one node at a time
        while current_sg != full_sg {
            let mut best_next: Option<(usize, Node, f64)> = None;

            for node in &graph.nodes {
                let node_sg = SubqueryGraph::singleton(node.id);
                if current_sg.contains(node_sg) {
                    continue; // Already included
                }

                // Check if there's an edge connecting this node to the current subgraph
                let edge = self.find_connecting_edge_to_node(graph, current_sg, node.id);
                if edge.is_none() {
                    continue;
                }

                let edge = edge.unwrap();
                let current_root = dp.best_plan(current_sg).unwrap().root;

                let join_node = arena.push(LogicalOp::IndexNestedLoopJoin {
                    left: current_root,
                    rel_type: edge.rel_type.clone(),
                    direction: edge.direction,
                    dst_alias: node.alias.clone(),
                });

                let cost = self.cost_model.inl_join_cost(best_cost);

                match &best_next {
                    Some((_, _, existing_cost)) if cost >= *existing_cost => {}
                    _ => {
                        best_next = Some((node.id, join_node, cost));
                    }
                }
            }

            if let Some((node_id, join_node, cost)) = best_next {
                let new_sg = current_sg.union(SubqueryGraph::singleton(node_id));
                dp.add_plan(
                    new_sg,
                    LogicalPlanRef {
                        root: join_node,
                        cost,
                        factorization_encoding: 0,
                    },
                );
                current_sg = new_sg;
                best_cost = cost;
            } else {
                break; // No more connected nodes to add
            }
        }

        Ok(())
    }

    fn plan_single_node(&self, node: &QueryGraphNode, arena: &mut Arena<LogicalOp>) -> Node {
        arena.push(LogicalOp::ScanNode {
            label: node.label.clone(),
            alias: node.alias.clone(),
            projections: None,
        })
    }

    fn find_connecting_edge<'b>(
        &self,
        graph: &'b QueryGraph,
        left: SubqueryGraph,
        right: SubqueryGraph,
    ) -> Option<&'b QueryGraphEdge> {
        graph.edges.iter().find(|e| {
            let src_in_left = left.contains(SubqueryGraph::singleton(e.src_id));
            let dst_in_right = right.contains(SubqueryGraph::singleton(e.dst_id));
            let src_in_right = right.contains(SubqueryGraph::singleton(e.src_id));
            let dst_in_left = left.contains(SubqueryGraph::singleton(e.dst_id));
            (src_in_left && dst_in_right) || (src_in_right && dst_in_left)
        })
    }

    fn find_connecting_edge_to_node<'b>(
        &self,
        graph: &'b QueryGraph,
        sg: SubqueryGraph,
        node_id: usize,
    ) -> Option<&'b QueryGraphEdge> {
        graph.edges.iter().find(|e| {
            (e.src_id == node_id && sg.contains(SubqueryGraph::singleton(e.dst_id)))
                || (e.dst_id == node_id && sg.contains(SubqueryGraph::singleton(e.src_id)))
        })
    }
}
