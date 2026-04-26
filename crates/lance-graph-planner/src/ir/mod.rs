//! Arena-allocated plan IR.
//!
//! Inspired by Polars: plans are stored in arenas with `Node` handles.
//! O(1) node replacement, cache-friendly, serializable.
//! Extended with Kuzudb's factorization schema.

pub mod logical_op;
pub mod expr;
pub mod schema;
pub mod properties;

pub use logical_op::LogicalOp;
pub use expr::{AExpr, ExprNode};
pub use schema::{FactorizationGroup, Schema};
pub use properties::PlanProperties;

use std::collections::HashMap;

/// Opaque handle into an arena. Copy + Eq + Hash.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Node(pub u32);

impl Node {
    pub fn index(self) -> usize {
        self.0 as usize
    }
}

/// Type-safe arena. Stores plan nodes or expressions contiguously.
#[derive(Debug)]
pub struct Arena<T> {
    items: Vec<T>,
}

impl<T> Arena<T> {
    pub fn new() -> Self {
        Self { items: Vec::new() }
    }

    pub fn with_capacity(cap: usize) -> Self {
        Self { items: Vec::with_capacity(cap) }
    }

    /// Insert a new item, returning its handle.
    pub fn push(&mut self, item: T) -> Node {
        let idx = self.items.len();
        self.items.push(item);
        Node(idx as u32)
    }

    /// Get a reference by handle.
    pub fn get(&self, node: Node) -> &T {
        &self.items[node.index()]
    }

    /// Get a mutable reference by handle. O(1) in-place replacement.
    pub fn get_mut(&mut self, node: Node) -> &mut T {
        &mut self.items[node.index()]
    }

    /// Replace a node in-place. Returns the old value.
    pub fn replace(&mut self, node: Node, new_val: T) -> T {
        std::mem::replace(&mut self.items[node.index()], new_val)
    }

    pub fn len(&self) -> usize {
        self.items.len()
    }

    pub fn is_empty(&self) -> bool {
        self.items.is_empty()
    }

    pub fn iter(&self) -> impl Iterator<Item = (Node, &T)> {
        self.items.iter().enumerate().map(|(i, t)| (Node(i as u32), t))
    }
}

impl<T> Default for Arena<T> {
    fn default() -> Self {
        Self::new()
    }
}

/// A logical plan: arena of operators + arena of expressions + root node.
#[derive(Debug)]
pub struct LogicalPlan {
    /// The operator arena
    pub ops: Arena<LogicalOp>,
    /// The expression arena
    pub exprs: Arena<AExpr>,
    /// Root of the plan tree
    pub root: Node,
    /// Schema at the root
    pub schema: Schema,
    /// Estimated cost
    pub cost: f64,
    /// Plan properties (UCCs, FDs, ordering)
    pub properties: PlanProperties,
}

impl LogicalPlan {
    pub fn new(ops: Arena<LogicalOp>, exprs: Arena<AExpr>, root: Node) -> Self {
        Self {
            ops,
            exprs,
            root,
            schema: Schema::default(),
            cost: f64::MAX,
            properties: PlanProperties::default(),
        }
    }
}

/// Subquery graph bitmask for DP join enumeration (from Kuzudb).
/// Each bit represents a node/edge in the query graph.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SubqueryGraph(pub u64);

impl SubqueryGraph {
    pub fn empty() -> Self {
        Self(0)
    }

    pub fn singleton(idx: usize) -> Self {
        Self(1u64 << idx)
    }

    pub fn union(self, other: Self) -> Self {
        Self(self.0 | other.0)
    }

    pub fn intersect(self, other: Self) -> Self {
        Self(self.0 & other.0)
    }

    pub fn contains(self, other: Self) -> bool {
        (self.0 & other.0) == other.0
    }

    pub fn is_subset_of(self, other: Self) -> bool {
        other.contains(self)
    }

    pub fn count(self) -> u32 {
        self.0.count_ones()
    }

    /// Iterate over all proper non-empty subsets.
    pub fn subsets(self) -> SubsetIter {
        SubsetIter { full: self.0, current: self.0 }
    }
}

pub struct SubsetIter {
    full: u64,
    current: u64,
}

impl Iterator for SubsetIter {
    type Item = SubqueryGraph;

    fn next(&mut self) -> Option<Self::Item> {
        if self.current == 0 {
            return None;
        }
        // Enumerate subsets via bit tricks (Kuzudb pattern)
        self.current = (self.current.wrapping_sub(1)) & self.full;
        if self.current == 0 {
            None
        } else {
            Some(SubqueryGraph(self.current))
        }
    }
}

/// DP table for join enumeration (from Kuzudb).
/// Maps SubqueryGraph → up to MAX_PLANS plans, differentiated by factorization encoding.
pub struct SubPlansTable {
    table: HashMap<SubqueryGraph, SubgraphPlans>,
}

/// Max plans per subgraph (Kuzudb uses 10).
const MAX_PLANS_PER_SUBGRAPH: usize = 10;

pub struct SubgraphPlans {
    plans: Vec<(u64, LogicalPlanRef)>,
}

/// Lightweight reference to a plan (root node + cost + factorization encoding).
#[derive(Debug, Clone)]
pub struct LogicalPlanRef {
    pub root: Node,
    pub cost: f64,
    pub factorization_encoding: u64,
}

impl Default for SubgraphPlans {
    fn default() -> Self {
        Self::new()
    }
}

impl SubgraphPlans {
    pub fn new() -> Self {
        Self { plans: Vec::new() }
    }

    /// Add a plan if it's better than existing plans with the same factorization encoding,
    /// or if we have room for a new encoding.
    pub fn add_plan(&mut self, plan: LogicalPlanRef) {
        let encoding = plan.factorization_encoding;

        // Check if we already have a plan with this encoding
        for (enc, existing) in &mut self.plans {
            if *enc == encoding {
                if plan.cost < existing.cost {
                    *existing = plan;
                }
                return;
            }
        }

        // New encoding — add if room
        if self.plans.len() < MAX_PLANS_PER_SUBGRAPH {
            self.plans.push((encoding, plan));
        } else {
            // Replace the most expensive plan if this one is cheaper
            if let Some((idx, _)) = self.plans.iter().enumerate()
                .max_by(|a, b| a.1 .1.cost.partial_cmp(&b.1 .1.cost).unwrap())
            {
                if plan.cost < self.plans[idx].1.cost {
                    self.plans[idx] = (encoding, plan);
                }
            }
        }
    }

    pub fn best(&self) -> Option<&LogicalPlanRef> {
        self.plans.iter()
            .map(|(_, p)| p)
            .min_by(|a, b| a.cost.partial_cmp(&b.cost).unwrap())
    }

    pub fn plans(&self) -> impl Iterator<Item = &LogicalPlanRef> {
        self.plans.iter().map(|(_, p)| p)
    }
}

impl Default for SubPlansTable {
    fn default() -> Self {
        Self::new()
    }
}

impl SubPlansTable {
    pub fn new() -> Self {
        Self { table: HashMap::new() }
    }

    pub fn get(&self, sg: SubqueryGraph) -> Option<&SubgraphPlans> {
        self.table.get(&sg)
    }

    pub fn get_or_insert(&mut self, sg: SubqueryGraph) -> &mut SubgraphPlans {
        self.table.entry(sg).or_default()
    }

    pub fn add_plan(&mut self, sg: SubqueryGraph, plan: LogicalPlanRef) {
        self.get_or_insert(sg).add_plan(plan);
    }

    pub fn best_plan(&self, sg: SubqueryGraph) -> Option<&LogicalPlanRef> {
        self.table.get(&sg).and_then(|sp| sp.best())
    }
}
