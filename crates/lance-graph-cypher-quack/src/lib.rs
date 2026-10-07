//! **Experimental.** Cypher → Quack over TABULAR inputs, beside DataFusion,
//! routed nowhere.
//!
//! ```text
//! Cypher text
//!   │ lance_graph::parser::parse_cypher_query           (upstream, read-only)
//!   │ lance_graph::semantic::SemanticAnalyzer::analyze  ($params substituted)
//!   │ lance_graph::logical_plan::LogicalPlanner::plan   → bound LogicalOperator
//!   │ demand::classify                                  → which carrier
//!   │ lower_plan                    (this crate)        → quack::Query   (names → Col here, once)
//!   │ lance_graph_quack::lower                          → mask_risc::Program
//!   ▼
//! Compiled::execute  (mask_risc::execute_into over borrowed lanes)  → Answer
//! ```
//!
//! # The seam, and why it is this one
//!
//! The input is the BOUND `LogicalOperator`, not the AST:
//! - `SemanticAnalyzer` already validates variables and substitutes
//!   parameters, and `LogicalPlanner` already normalises patterns into
//!   `ScanByLabel` / `Filter` / `Expand` / `Project`. Lowering from the AST
//!   would repeat that binding; this crate binds only what the upstream
//!   stages do not: names to lanes ([`Binding`]).
//! - It edits no upstream-owned file (`cypher-mask-lowering-v2` §2).
//!
//! # What lowers today (every other shape is a typed [`Refusal`])
//!
//! | shape | demand | carrier |
//! |---|---|---|
//! | `MATCH (n:L) [WHERE p] RETURN count(*) / count(n) / sum / min / max` | `TerminalSet` | one program over the node table |
//! | `MATCH (a:L)-[:R]->(b:L) RETURN count(*)` | `TerminalCount` | count over the edge rows, both endpoints semijoined |
//! | `… RETURN count(DISTINCT b)` / `count(DISTINCT b.<id>)` | `TerminalSet` | `CountDistinctOrderedU32` over `dst`, only if the edge table is stored in `dst` order |
//!
//! Semantics are WALK, as in DataFusion: one hop counts edge rows, so
//! parallel edges and self-loops each count. A hop with a `WHERE`, a second
//! hop, variable length and every other shape refuse with the named reason
//! (the gaps are listed on [`Refusal`]).
//!
//! # What this is not
//!
//! Not a second evaluator: it builds a `quack::Query` and calls
//! `lance_graph_quack::lower`; execution is mask-risc's. Not a fallback
//! router: nothing here calls DataFusion, and a refusal is never retried
//! elsewhere.

#![forbid(unsafe_code)]

pub mod demand;

use std::collections::HashMap;

use lance_graph::ast::{
    BooleanExpression, ComparisonOperator, PropertyValue, RelationshipDirection, ValueExpression,
};
use lance_graph::config::GraphConfig;
use lance_graph::logical_plan::{LogicalOperator, LogicalPlanner, ProjectionItem};
use lance_graph::parser::parse_cypher_query;
use lance_graph::semantic::SemanticAnalyzer;
use lance_graph_mask_risc::{
    execute_into, words_for, ExecError, Foreign, ForeignPlane, LaneRef, Out, Planes, Program,
    Scratch, Value,
};
use lance_graph_quack::{lower, Agg, Cmp, Col, Filter, ForeignPlane as FP, Mask, Query};

pub use demand::{classify, Demand};

// ---------------------------------------------------------------- binding

/// What a property lane holds.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Kind {
    /// A signed `i32` lane: every comparison and `sum`/`min`/`max`.
    I32,
    /// A `u32` lane: `=` and `<>` only (an ordered `u32` compare is a gap).
    U32,
}

/// One node label's table: its row space, its properties' lanes.
///
/// The label's id property must hold the ROW ORDINAL (`0..n_rows`), because
/// the edge table's endpoint columns are row-ordinal foreign keys into it.
#[derive(Debug, Clone)]
pub struct NodeTable {
    /// The Cypher label.
    pub label: String,
    /// The property the label is keyed by (`GraphConfig`'s id field).
    pub id_property: String,
    /// `(property, lane, kind)`; the lane index is the position in the
    /// `lanes` slice handed to [`Compiled::execute`].
    pub properties: Vec<(String, Col, Kind)>,
}

/// One relationship type's edge table: one row per edge.
#[derive(Debug, Clone)]
pub struct EdgeTable {
    /// The Cypher relationship type.
    pub rel_type: String,
    /// The node label both endpoints index (one population per mask).
    pub label: String,
    /// The `u32` lane of source row ordinals.
    pub src: Col,
    /// The `u32` lane of target row ordinals.
    pub dst: Col,
    /// Whether the rows are stored in non-decreasing `dst` order — the
    /// physical precondition of an exact `count(DISTINCT dst)`. The executor
    /// checks it again and refuses a lane out of order.
    pub dst_ordered: bool,
}

/// The consumer's resolution of names to lanes. Passed beside the query,
/// never inside `GraphConfig` (an upstream type).
#[derive(Debug, Clone, Default)]
pub struct Binding {
    /// Node tables, one per label.
    pub nodes: Vec<NodeTable>,
    /// Edge tables, one per relationship type.
    pub edges: Vec<EdgeTable>,
}

impl Binding {
    /// The `GraphConfig` the upstream planner needs: one node label per node
    /// table and one relationship per edge table. Its source/target field
    /// names are placeholders the lowering never reads.
    pub fn graph_config(&self) -> GraphConfig {
        let mut b = GraphConfig::builder();
        for n in &self.nodes {
            b = b.with_node_label(&n.label, &n.id_property);
        }
        for e in &self.edges {
            b = b.with_relationship(e.rel_type.as_str(), "src", "dst");
        }
        b.build()
            .expect("a binding's labels form a valid GraphConfig")
    }

    fn node(&self, label: &str) -> Option<&NodeTable> {
        self.nodes
            .iter()
            .find(|n| n.label.eq_ignore_ascii_case(label))
    }

    fn edge(&self, rel: &str) -> Option<&EdgeTable> {
        self.edges
            .iter()
            .find(|e| e.rel_type.eq_ignore_ascii_case(rel))
    }
}

impl NodeTable {
    fn property(&self, name: &str) -> Option<(Col, Kind)> {
        self.properties
            .iter()
            .find(|(p, _, _)| p.eq_ignore_ascii_case(name))
            .map(|&(_, c, k)| (c, k))
    }
}

// ---------------------------------------------------------------- refusals

/// Why a query does not lower. Each variant carries the offending construct.
/// A refusal is the answer, not a cue to fall back.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum Refusal {
    /// The upstream parser rejected the text.
    Unparsed(String),
    /// Semantic analysis or logical planning failed.
    Unplanned(String),
    /// A label, relationship type or property has no [`Binding`] entry.
    Unbound(String),
    /// The consumer needs the identity of two or more variables.
    Bindings,
    /// A shape this lowering does not (yet) recognise.
    Shape(String),
    /// A known gap. The current ones:
    /// - `WHERE` with a hop: ordered compare through an fk (only
    ///   `EqU32Via` exists), and a node-table `Keep` mask fed to the edge
    ///   table's `Semijoin` is the forbidden two-program shape;
    /// - two or more hops: the walk count needs a foreign-value sum;
    /// - variable length: a hop chain over a computed frontier.
    Gap(String),
    /// `count(DISTINCT b)` over an edge table not stored in `dst` order.
    Layout(String),
    /// A value the lanes cannot hold (`avg`, a non-integer literal, an
    /// integer out of `i32` range, a value-`DISTINCT`, a string).
    Value(String),
}

// ---------------------------------------------------------------- compile

/// What a compiled query returns.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ResultShape {
    /// `count` or `sum`: an integer.
    Int,
    /// `min` / `max`: an integer, or `None` over no rows (Cypher `null`).
    OptInt,
}

/// The table a compiled program runs over.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum On {
    /// The node table of this label.
    Nodes(String),
    /// The edge table of this relationship type, whose endpoints index the
    /// node table of `label` (supplied as foreign plane 0).
    Edges { rel_type: String, label: String },
}

/// Lowered once; executed any number of times.
#[derive(Debug, Clone)]
pub struct Compiled {
    /// The mask-risc program.
    pub program: Program,
    /// Which table it runs over.
    pub on: On,
    /// What it returns.
    pub result: ResultShape,
    /// The carrier the demand selected.
    pub demand: Demand,
}

/// A scalar answer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Answer {
    /// `count` / `sum`.
    Int(i64),
    /// `min` / `max`; `None` = no rows.
    OptInt(Option<i64>),
}

/// Parse, bind, plan (upstream, read-only), classify and lower.
pub fn compile(
    text: &str,
    params: &HashMap<String, serde_json::Value>,
    bind: &Binding,
) -> Result<Compiled, Refusal> {
    let ast = parse_cypher_query(text).map_err(|e| Refusal::Unparsed(e.to_string()))?;
    let cfg = bind.graph_config();
    let sem = SemanticAnalyzer::new(cfg.clone())
        .analyze(&ast, params)
        .map_err(|e| Refusal::Unplanned(e.to_string()))?;
    if !sem.errors.is_empty() {
        return Err(Refusal::Unplanned(sem.errors.join("; ")));
    }
    let plan = LogicalPlanner::new(&cfg)
        .plan(&sem.ast)
        .map_err(|e| Refusal::Unplanned(e.to_string()))?;
    lower_plan(&plan, bind)
}

/// Lower a bound plan. Exposed so a caller holding a plan need not re-plan.
pub fn lower_plan(plan: &LogicalOperator, bind: &Binding) -> Result<Compiled, Refusal> {
    let demand = classify(plan);
    if demand == Demand::Bindings {
        return Err(Refusal::Bindings);
    }
    let (input, projections) = peel(plan)?;
    let [item] = projections else {
        return Err(Refusal::Shape(format!(
            "{} RETURN items; one is supported",
            projections.len()
        )));
    };
    let agg = aggregate(item)?;

    match input {
        LogicalOperator::ScanByLabel { .. } | LogicalOperator::Filter { .. } => {
            let (var, label, pred) = single_population(input)?;
            let table = bind
                .node(label)
                .ok_or_else(|| Refusal::Unbound(format!("label {label}")))?;
            let mut parts = vec![Filter::plane(LIVE)];
            if let Some(p) = pred {
                parts.push(filter(p, var, table)?);
            }
            let (agg, result) = node_agg(&agg, var, table)?;
            finish(
                Query {
                    filter: Filter::and(parts),
                    agg,
                },
                On::Nodes(table.label.clone()),
                result,
                demand,
            )
        }
        LogicalOperator::Expand { .. } => one_hop(input, &agg, bind, demand),
        LogicalOperator::VariableLengthExpand { .. } => Err(Refusal::Gap(
            "variable length: a hop chain over a computed frontier".into(),
        )),
        other => Err(Refusal::Shape(format!("{} under RETURN", op_name(other)))),
    }
}

const WHERE_WITH_HOP: &str =
    "WHERE with a hop: ordered compare through an fk / Keep→Semijoin is forbidden";
const LIVE: Mask = Mask(0);
const NODES: FP = FP(0);

fn finish(query: Query, on: On, result: ResultShape, demand: Demand) -> Result<Compiled, Refusal> {
    let program = lower(&query).map_err(|e| Refusal::Shape(format!("quack lowering: {e}")))?;
    Ok(Compiled {
        program,
        on,
        result,
        demand,
    })
}

/// Strip `Distinct` / `Sort` / `Limit ≥ 1` above a one-row aggregate (no-ops
/// on a single row) and return the projection. `Offset` and `LIMIT 0` change
/// a one-row result and are refused.
fn peel(plan: &LogicalOperator) -> Result<(&LogicalOperator, &[ProjectionItem]), Refusal> {
    let mut op = plan;
    loop {
        op = match op {
            LogicalOperator::Distinct { input } | LogicalOperator::Sort { input, .. } => input,
            LogicalOperator::Limit { input, count } if *count >= 1 => input,
            LogicalOperator::Limit { .. } | LogicalOperator::Offset { .. } => {
                return Err(Refusal::Shape(
                    "SKIP / LIMIT 0 over a one-row result".into(),
                ))
            }
            LogicalOperator::Project { input, projections } => return Ok((input, projections)),
            other => return Err(Refusal::Shape(format!("{} at the top", op_name(other)))),
        };
    }
}

/// A RETURN aggregate, as this crate understands it.
enum AggItem<'a> {
    /// `count(*)`, `count(n)`, `count(n.p)`: one per binding (no NULLs here:
    /// every lane is dense and every live row has a value).
    Count,
    /// `count(DISTINCT n)` (`None`) or `count(DISTINCT n.p)` (`Some(p)`).
    /// It counts distinct NODES only when `p` is the label's id property;
    /// [`distinct_node`] refuses any other `p` as a value-`DISTINCT`.
    CountDistinctNode(&'a str, Option<&'a str>),
    Sum(&'a str, &'a str),
    Min(&'a str, &'a str),
    Max(&'a str, &'a str),
}

fn aggregate(item: &ProjectionItem) -> Result<AggItem<'_>, Refusal> {
    let ValueExpression::AggregateFunction {
        name,
        args,
        distinct,
    } = &item.expression
    else {
        return Err(Refusal::Shape(
            "a non-aggregate RETURN (rows, not a scalar)".into(),
        ));
    };
    let name = name.to_ascii_lowercase();
    use ValueExpression::{Property as P, Variable as V};
    match (name.as_str(), args.as_slice(), *distinct) {
        ("count", [V(_) | P(_)], false) => Ok(AggItem::Count),
        ("count", [V(v)], true) if v != "*" => Ok(AggItem::CountDistinctNode(v, None)),
        ("count", [P(p)], true) => Ok(AggItem::CountDistinctNode(&p.variable, Some(&p.property))),
        ("sum", [P(p)], false) => Ok(AggItem::Sum(&p.variable, &p.property)),
        // min/max see only the support, so DISTINCT changes nothing
        ("min", [P(p)], _) => Ok(AggItem::Min(&p.variable, &p.property)),
        ("max", [P(p)], _) => Ok(AggItem::Max(&p.variable, &p.property)),
        _ => Err(Refusal::Value(format!(
            "aggregate {name}{} not lowered",
            if *distinct { " DISTINCT" } else { "" }
        ))),
    }
}

/// `ScanByLabel` with at most one `Filter` above it.
fn single_population(
    op: &LogicalOperator,
) -> Result<(&str, &str, Option<&BooleanExpression>), Refusal> {
    match op {
        LogicalOperator::ScanByLabel {
            variable,
            label,
            properties,
        } => {
            if !properties.is_empty() {
                return Err(Refusal::Shape("inline pattern properties".into()));
            }
            Ok((variable, label, None))
        }
        // The planner puts a WHERE above the hops it filters.
        LogicalOperator::Filter { input, .. }
            if matches!(
                input.as_ref(),
                LogicalOperator::Expand { .. } | LogicalOperator::VariableLengthExpand { .. }
            ) =>
        {
            Err(Refusal::Gap(WHERE_WITH_HOP.into()))
        }
        LogicalOperator::Filter { input, predicate } => match single_population(input)? {
            (v, l, None) => Ok((v, l, Some(predicate))),
            _ => Err(Refusal::Shape("stacked filters".into())),
        },
        other => Err(Refusal::Shape(format!(
            "{} in a single population",
            op_name(other)
        ))),
    }
}

fn node_agg(agg: &AggItem<'_>, var: &str, t: &NodeTable) -> Result<(Agg, ResultShape), Refusal> {
    let col = |v: &str, p: &str| -> Result<Col, Refusal> {
        if v != var {
            return Err(Refusal::Shape(format!(
                "aggregate over {v}, pattern binds {var}"
            )));
        }
        match t.property(p) {
            Some((c, Kind::I32)) => Ok(c),
            Some((_, Kind::U32)) => {
                Err(Refusal::Value(format!("{p} is u32; sum/min/max need i32")))
            }
            None => Err(Refusal::Unbound(format!("property {}.{p}", t.label))),
        }
    };
    Ok(match *agg {
        AggItem::Count => (Agg::Count, ResultShape::Int),
        // one row per node: a distinct-node count is the row count
        AggItem::CountDistinctNode(v, p) => {
            distinct_node(v, p, var, t)?;
            (Agg::Count, ResultShape::Int)
        }
        AggItem::Sum(v, p) => (Agg::SumI32(col(v, p)?), ResultShape::Int),
        AggItem::Min(v, p) => (Agg::MinI32(col(v, p)?), ResultShape::OptInt),
        AggItem::Max(v, p) => (Agg::MaxI32(col(v, p)?), ResultShape::OptInt),
    })
}

/// `count(DISTINCT v[.p])` counts distinct nodes of `t` bound to `want`:
/// `v` must be that variable, and `p`, if given, its id property.
fn distinct_node(v: &str, p: Option<&str>, want: &str, t: &NodeTable) -> Result<(), Refusal> {
    if v != want {
        return Err(Refusal::Shape(format!(
            "count(DISTINCT {v}) where the carrier is over {want}"
        )));
    }
    match p {
        Some(p) if !p.eq_ignore_ascii_case(&t.id_property) => Err(Refusal::Value(format!(
            "count(DISTINCT {v}.{p}) is a value DISTINCT"
        ))),
        _ => Ok(()),
    }
}

/// `WHERE` over one variable's integer properties.
fn filter(e: &BooleanExpression, var: &str, t: &NodeTable) -> Result<Filter, Refusal> {
    Ok(match e {
        BooleanExpression::And(a, b) => Filter::and([filter(a, var, t)?, filter(b, var, t)?]),
        BooleanExpression::Or(a, b) => Filter::or([filter(a, var, t)?, filter(b, var, t)?]),
        BooleanExpression::Not(a) => Filter::negate(filter(a, var, t)?),
        BooleanExpression::Comparison {
            left,
            operator,
            right,
        } => {
            // property OP literal, or literal OP property (operator mirrored)
            let (p, op, lit) = match (left, right) {
                (ValueExpression::Property(p), ValueExpression::Literal(l)) => {
                    (p, operator.clone(), l)
                }
                (ValueExpression::Literal(l), ValueExpression::Property(p)) => {
                    (p, mirror(operator.clone()), l)
                }
                _ => {
                    return Err(Refusal::Shape(
                        "a comparison that is not property-vs-literal".into(),
                    ))
                }
            };
            if p.variable != var {
                return Err(Refusal::Shape(format!(
                    "predicate on {}, pattern binds {var}",
                    p.variable
                )));
            }
            let (col, kind) = t
                .property(&p.property)
                .ok_or_else(|| Refusal::Unbound(format!("property {}.{}", t.label, p.property)))?;
            let PropertyValue::Integer(v) = lit else {
                return Err(Refusal::Value(format!("literal {lit:?}")));
            };
            use ComparisonOperator as C;
            match kind {
                Kind::I32 => {
                    let v =
                        i32::try_from(*v).map_err(|_| Refusal::Value(format!("{v} out of i32")))?;
                    Filter::cmp(
                        col,
                        match op {
                            C::Equal => Cmp::EqI32(v),
                            C::NotEqual => Cmp::NeI32(v),
                            C::LessThan => Cmp::LtI32(v),
                            C::LessThanOrEqual => Cmp::LeI32(v),
                            C::GreaterThan => Cmp::GtI32(v),
                            C::GreaterThanOrEqual => Cmp::GeI32(v),
                        },
                    )
                }
                Kind::U32 => {
                    let v =
                        u32::try_from(*v).map_err(|_| Refusal::Value(format!("{v} out of u32")))?;
                    match op {
                        C::Equal => Filter::cmp(col, Cmp::EqU32(v)),
                        C::NotEqual => Filter::cmp(col, Cmp::NeU32(v)),
                        _ => return Err(Refusal::Gap("ordered compare on a u32 lane".into())),
                    }
                }
            }
        }
        other => return Err(Refusal::Shape(format!("predicate {other:?}"))),
    })
}

fn mirror(op: ComparisonOperator) -> ComparisonOperator {
    use ComparisonOperator as C;
    match op {
        C::LessThan => C::GreaterThan,
        C::LessThanOrEqual => C::GreaterThanOrEqual,
        C::GreaterThan => C::LessThan,
        C::GreaterThanOrEqual => C::LessThanOrEqual,
        same => same,
    }
}

/// `(a:L)-[:R]->(b:L)` with no filter, re-anchored on the edge table.
fn one_hop(
    op: &LogicalOperator,
    agg: &AggItem<'_>,
    bind: &Binding,
    demand: Demand,
) -> Result<Compiled, Refusal> {
    let LogicalOperator::Expand {
        input,
        source_variable: _,
        target_variable,
        target_label,
        relationship_types,
        direction,
        properties,
        target_properties,
        ..
    } = op
    else {
        unreachable!("one_hop is called on an Expand")
    };
    let (src_var, src_label) = match input.as_ref() {
        LogicalOperator::ScanByLabel {
            variable,
            label,
            properties,
        } if properties.is_empty() => (variable.as_str(), label.as_str()),
        LogicalOperator::ScanByLabel { .. } => {
            return Err(Refusal::Shape("inline pattern properties".into()))
        }
        LogicalOperator::Expand { .. } | LogicalOperator::VariableLengthExpand { .. } => {
            return Err(Refusal::Gap(
                "two or more hops: the walk count needs a foreign-value sum".into(),
            ))
        }
        LogicalOperator::Filter { .. } => return Err(Refusal::Gap(WHERE_WITH_HOP.into())),
        other => return Err(Refusal::Shape(format!("{} under a hop", op_name(other)))),
    };
    if !properties.is_empty() || !target_properties.is_empty() {
        return Err(Refusal::Shape("inline pattern properties".into()));
    }
    let [rel] = relationship_types.as_slice() else {
        return Err(Refusal::Shape(
            "a hop over zero or several relationship types".into(),
        ));
    };
    let e = bind
        .edge(rel)
        .ok_or_else(|| Refusal::Unbound(format!("relationship {rel}")))?;
    for l in [src_label, target_label.as_str()] {
        if !e.label.eq_ignore_ascii_case(l) {
            return Err(Refusal::Shape(format!(
                "{rel} indexes {}, pattern names {l}",
                e.label
            )));
        }
    }
    let nodes = bind
        .node(&e.label)
        .ok_or_else(|| Refusal::Unbound(format!("label {}", e.label)))?;
    // Which edge column holds each pattern variable's row ordinal.
    let (src_col, dst_col) = match direction {
        RelationshipDirection::Outgoing => (e.src, e.dst),
        RelationshipDirection::Incoming => (e.dst, e.src),
        RelationshipDirection::Undirected => {
            return Err(Refusal::Shape(
                "an undirected hop (each edge would count twice)".into(),
            ))
        }
    };
    let filter = Filter::and([
        Filter::plane(LIVE),
        Filter::semijoin(src_col, NODES),
        Filter::semijoin(dst_col, NODES),
    ]);
    let agg = match (demand, agg) {
        // one edge row is one walk: the bag count is exact (#1311)
        (Demand::TerminalCount, AggItem::Count) | (Demand::EarlierCount, AggItem::Count) => {
            Agg::Count
        }
        (Demand::TerminalSet, AggItem::CountDistinctNode(v, p))
            if *v == target_variable.as_str() =>
        {
            distinct_node(v, *p, target_variable, nodes)?;
            if dst_col != e.dst || !e.dst_ordered {
                return Err(Refusal::Layout(format!(
                    "count(DISTINCT {v}) needs {rel} stored in target-column order"
                )));
            }
            Agg::CountDistinctOrderedU32 { key: dst_col }
        }
        (Demand::EarlierSet, AggItem::CountDistinctNode(v, _)) if *v == src_var => {
            return Err(Refusal::Layout(format!(
                "count(DISTINCT {v}) needs {rel} stored in source-column order"
            )))
        }
        _ => {
            return Err(Refusal::Shape(format!(
                "after a hop, demand {demand:?} with this aggregate is not lowered"
            )))
        }
    };
    finish(
        Query { filter, agg },
        On::Edges {
            rel_type: e.rel_type.clone(),
            label: nodes.label.clone(),
        },
        ResultShape::Int,
        demand,
    )
}

fn op_name(op: &LogicalOperator) -> &'static str {
    match op {
        LogicalOperator::ScanByLabel { .. } => "ScanByLabel",
        LogicalOperator::Unwind { .. } => "Unwind",
        LogicalOperator::Filter { .. } => "Filter",
        LogicalOperator::Expand { .. } => "Expand",
        LogicalOperator::VariableLengthExpand { .. } => "VariableLengthExpand",
        LogicalOperator::Project { .. } => "Project",
        LogicalOperator::Join { .. } => "Join",
        LogicalOperator::Distinct { .. } => "Distinct",
        LogicalOperator::Sort { .. } => "Sort",
        LogicalOperator::Offset { .. } => "Offset",
        LogicalOperator::Limit { .. } => "Limit",
    }
}

// ---------------------------------------------------------------- execute

/// The borrowed lanes of one table, in [`Col`] order, and its row count.
/// Every row is live (the live plane is all-ones over `n_rows`).
#[derive(Clone, Copy)]
pub struct TableLanes<'a> {
    /// Row count.
    pub n_rows: usize,
    /// Lanes, indexed by `Col`.
    pub lanes: &'a [LaneRef<'a>],
}

impl Compiled {
    /// Execute over `on`'s lanes. For an edge program, `nodes` is the node
    /// table its endpoints index (its row count bounds the semijoins).
    pub fn execute(
        &self,
        on: TableLanes<'_>,
        nodes: Option<TableLanes<'_>>,
    ) -> Result<Answer, ExecError> {
        let live = ones(on.n_rows);
        let node_live = nodes.map(|n| ones(n.n_rows)).unwrap_or_default();
        let masks: [&[u64]; 1] = [&live];
        let planes = Planes {
            n_rows: on.n_rows,
            masks: &masks,
            lanes: on.lanes,
        };
        let fplanes = nodes.map(|n| {
            [ForeignPlane {
                words: &node_live,
                rows: n.n_rows,
            }]
        });
        let foreign = Foreign {
            planes: fplanes.as_ref().map_or(&[][..], |p| &p[..]),
            lanes: &[],
        };
        let mut scratch = Scratch::for_program(&self.program, on.n_rows)?;
        let v = execute_into(&self.program, &planes, &foreign, &mut scratch, Out::None)?;
        Ok(match (self.result, v) {
            (ResultShape::Int, Value::Count(c)) => Answer::Int(c as i64),
            (ResultShape::Int, Value::SumI64(s)) => Answer::Int(s),
            (ResultShape::OptInt, Value::OptI32(m)) => Answer::OptInt(m.map(i64::from)),
            (shape, v) => unreachable!("{shape:?} program returned {v:?}"),
        })
    }
}

fn ones(n: usize) -> Vec<u64> {
    let mut w = vec![u64::MAX; words_for(n)];
    if !n.is_multiple_of(64) {
        if let Some(last) = w.last_mut() {
            *last = (1u64 << (n % 64)) - 1;
        }
    }
    w
}
