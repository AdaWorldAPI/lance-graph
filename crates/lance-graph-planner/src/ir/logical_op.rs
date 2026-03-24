//! Logical operators — the vocabulary of graph query plans.
//!
//! Merged from lance-graph (12 variants) + kuzudb (47 variants),
//! extended with resonance-specific operators.

use super::Node;

/// Logical operator enum. Each variant stores child node handles into the arena.
#[derive(Debug, Clone)]
pub enum LogicalOp {
    // === Graph Scan ===
    /// Scan nodes by label. Leaf operator.
    ScanNode {
        label: String,
        /// Alias for the scanned node variable
        alias: String,
        /// Column projections (None = all)
        projections: Option<Vec<String>>,
    },

    /// Scan relationships by type.
    ScanRelationship {
        rel_type: String,
        alias: String,
        direction: Direction,
        /// Bound node (already scanned)
        bound_node: Node,
    },

    // === Joins (from Kuzudb) ===
    /// Hash join: probe side (left) joined with build side (right).
    HashJoin {
        left: Node,
        right: Node,
        join_keys: Vec<(String, String)>,
        join_type: JoinType,
    },

    /// Index nested loop join: extend through adjacency list.
    /// Used when bound node has sequential scan guarantee.
    IndexNestedLoopJoin {
        left: Node,
        rel_type: String,
        direction: Direction,
        /// The node variable being extended to
        dst_alias: String,
    },

    /// Worst-case optimal join: multi-way intersection on adjacency lists.
    /// Activates when multiple relationships converge on a single node.
    WcoJoin {
        /// The intersect node (where all rels converge)
        intersect_alias: String,
        /// Each child is a separate rel-scan plan
        children: Vec<Node>,
    },

    /// Cross product (no join condition). Rare, usually rewritten to hash join.
    CrossProduct {
        left: Node,
        right: Node,
    },

    // === Filter / Project ===
    /// Filter rows by predicate expression.
    Filter {
        input: Node,
        predicate: super::ExprNode,
    },

    /// Project specific columns/expressions.
    Projection {
        input: Node,
        expressions: Vec<super::ExprNode>,
    },

    // === Aggregation ===
    /// Group-by aggregation.
    Aggregate {
        input: Node,
        group_by: Vec<super::ExprNode>,
        aggregates: Vec<AggregateExpr>,
    },

    // === Order / Limit ===
    OrderBy {
        input: Node,
        sort_keys: Vec<SortKey>,
    },

    Limit {
        input: Node,
        count: usize,
        offset: usize,
    },

    TopK {
        input: Node,
        sort_keys: Vec<SortKey>,
        k: usize,
    },

    // === Graph-specific ===
    /// Variable-length path expansion (recursive).
    RecursiveExtend {
        input: Node,
        rel_type: String,
        direction: Direction,
        min_hops: usize,
        max_hops: usize,
        dst_alias: String,
    },

    /// Shortest path computation.
    ShortestPath {
        input: Node,
        rel_type: String,
        direction: Direction,
        dst_alias: String,
    },

    // === Factorization (from Kuzudb) ===
    /// Flatten a factorized group. Inserted by FactorizationRewriter.
    Flatten {
        input: Node,
        /// The expression group to flatten
        group_pos: usize,
    },

    // === Resonance-specific (BROADCAST → SCAN → ACCUMULATE → COLLAPSE) ===

    /// BROADCAST: Distribute a query fingerprint to all scan partitions.
    Broadcast {
        /// The fingerprint source (expression producing a Container)
        fingerprint: super::ExprNode,
        /// Number of partitions
        partitions: usize,
    },

    /// SCAN: Vectorized Hamming distance computation.
    Scan {
        input: Node,
        strategy: ScanStrategy,
        /// Sigma-band threshold
        threshold: u32,
        /// Top-K results per partition
        top_k: usize,
    },

    /// ACCUMULATE: Propagate values along graph edges using a semiring.
    /// This is where truth values get accumulated DURING traversal.
    Accumulate {
        input: Node,
        /// Which semiring algebra to use
        semiring: SemiringType,
        /// The edge traversal pattern
        traversal: Node,
    },

    /// COLLAPSE: Apply GateState thresholds.
    /// FLOW (SD < 0.15) → emit. HOLD (0.15-0.35) → persist. BLOCK (> 0.35) → discard.
    Collapse {
        input: Node,
        gate: CollapseGate,
    },

    // === DML ===
    CreateNode {
        label: String,
        properties: Vec<(String, super::ExprNode)>,
    },

    CreateRelationship {
        rel_type: String,
        src: Node,
        dst: Node,
        properties: Vec<(String, super::ExprNode)>,
    },

    SetProperty {
        input: Node,
        property: String,
        value: super::ExprNode,
    },

    Delete {
        input: Node,
    },

    // === Utility ===
    /// Return results to client.
    Return {
        input: Node,
        columns: Vec<super::ExprNode>,
    },

    /// Union of multiple plan branches.
    Union {
        children: Vec<Node>,
        all: bool,
    },

    /// Semi-mask for sideways information passing (SIP).
    /// After hash join build, filter probe-side scan to matching rows only.
    SemiMask {
        input: Node,
        /// The hash table source (build side of a join)
        mask_source: Node,
        /// Column to filter on
        column: String,
    },

    /// Empty result set.
    EmptyResult,
}

/// Edge direction in graph traversal.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Direction {
    Outgoing,
    Incoming,
    Both,
}

/// Join type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum JoinType {
    Inner,
    Left,
    Right,
    Full,
    Semi,
    Anti,
}

/// Aggregate expression.
#[derive(Debug, Clone)]
pub struct AggregateExpr {
    pub function: AggFunction,
    pub input: super::ExprNode,
    pub distinct: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AggFunction {
    Count,
    Sum,
    Avg,
    Min,
    Max,
    Collect,
}

/// Sort key for ORDER BY.
#[derive(Debug, Clone)]
pub struct SortKey {
    pub expr: super::ExprNode,
    pub ascending: bool,
    pub nulls_first: bool,
}

/// Scan strategy for SCAN operator (chosen by cost model).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScanStrategy {
    /// Stroke columns first, progressive refinement (from lance-graph sigma-band).
    Cascade,
    /// Brute force SIMD over all rows.
    Full,
    /// Precomputed proximity index.
    Index,
}

/// Semiring type for ACCUMULATE operator.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SemiringType {
    /// Standard boolean AND/OR.
    Boolean,
    /// Hamming distance minimum.
    HammingMin,
    /// Tropical semiring (min, +).
    Tropical,
    /// XOR bundle for superposition.
    XorBundle,
    /// NARS truth value propagation: multiply = deduction, add = revision.
    TruthPropagating,
    /// Palette semiring (from bgz17).
    Palette,
    /// Custom user-defined semiring.
    Custom(u16),
}

/// Collapse gate thresholds (from agi-chat's CollapseGate).
#[derive(Debug, Clone)]
pub struct CollapseGate {
    /// Standard deviation threshold for FLOW (emit results).
    pub flow_threshold: f64,
    /// SD threshold for HOLD (persist to SPPM for later).
    pub hold_threshold: f64,
    // Above hold_threshold → BLOCK (discard).
}

impl Default for CollapseGate {
    fn default() -> Self {
        Self {
            flow_threshold: 0.15,
            hold_threshold: 0.35,
        }
    }
}
