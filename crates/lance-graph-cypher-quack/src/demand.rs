//! The demand classifier: which carrier a query's RETURN consumer needs from
//! the frontier its MATCH pattern produces.
//!
//! **Provenance.** This is `LogicalOperator::consumer_semantics()` from the
//! closed-unmerged #1305 (branch `ccr-2fcc2bd3-8o7m2l` @ `67abd29`,
//! including its final review fix `a37b63b`: hops that do not chain, and a
//! variable bound twice, fail closed). #1305 put it inside the upstream-owned
//! `logical_plan.rs`; here it is a free function over the public
//! `LogicalOperator`, so no upstream file changes. The logic is unchanged.
//!
//! **Why it is revived.** `cypher-mask-lowering-v2` §12 harvested only
//! #1305's measurements, because v2 refuses every path count after a hop.
//! Over a TABULAR edge table that refusal is broader than the semantics
//! require: one hop re-anchors on the edge population, where one edge row is
//! one walk, so a bag count is exact (#1311). Choosing between that carrier
//! and a node mask is exactly this classifier's question.
//!
//! The five kinds, and the existing Quack carrier each selects (see
//! `.claude/tools/carrier_sufficiency.py --min` for the exhaustive check):
//!
//! | demand | needs | Quack carrier (tabular, one hop) |
//! |---|---|---|
//! | `TerminalSet` | support of the last variable | node mask; after a hop `CountDistinctOrderedU32` / `ScatterOrU32` |
//! | `EarlierSet` | support of an earlier variable | the same over the reverse key |
//! | `TerminalCount` | walks, keyed on the last variable | count over the edge rows (re-anchor) |
//! | `EarlierCount` | walks, keyed on an earlier variable | the same; grouped by the earlier key |
//! | `Bindings` | identity of two or more variables | none: refused |
//!
//! It says what the consumer NEEDS, never that the query lowers.

use lance_graph::ast::{BooleanExpression, ValueExpression};
use lance_graph::logical_plan::LogicalOperator;

/// Which carrier a query's consumer needs from the frontier the pattern
/// produces — the multiplicity contract of
/// `.claude/plans/cypher-mask-multiplicity-contract-v1.md` §3.2.
///
/// A Boolean mask is the SUPPORT of a frontier: which distinct nodes a
/// variable can take. After a hop, Cypher's bag semantics count one row per
/// BINDING (path), so `count(*)` over `(a)->(b)->(c)` is the path count, not
/// the popcount of `c`'s mask, and a forward hop chain gives the exact support
/// of the TERMINAL variable only — an earlier variable needs a backward pass.
///
/// This says what the consumer NEEDS. It does not say the query lowers:
/// ordering, `SKIP`/`LIMIT`, value-`DISTINCT` and strings are judged
/// separately (lowering plan §4).
///
/// Deliberately not `Serialize`: it is consumed once, in process, to choose a
/// carrier; it is never persisted or sent.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Demand {
    /// The support of the terminal variable suffices. Every single-population
    /// query is this: one row is one node.
    TerminalSet,
    /// The support of ONE earlier variable — Boolean, but it needs a backward
    /// (semi-join) pass; the forward chain over-approximates it.
    EarlierSet,
    /// Support plus a per-node path count keyed on the terminal variable.
    TerminalCount,
    /// Support plus a path count keyed on ONE earlier variable.
    EarlierCount,
    /// The identity of two or more variables: rows must be enumerated.
    Bindings,
}

/// Classify the carrier this plan's consumer needs. Pure; fails closed to
/// [`Demand::Bindings`] on any shape it does not recognise.
pub fn classify(plan: &LogicalOperator) -> Demand {
    // Peel the wrappers that sit above the RETURN projection.
    let mut op = plan;
    let mut distinct = false;
    loop {
        match op {
            LogicalOperator::Sort { input, .. }
            | LogicalOperator::Offset { input, .. }
            | LogicalOperator::Limit { input, .. } => op = input,
            LogicalOperator::Distinct { input } => {
                distinct = true;
                op = input;
            }
            _ => break,
        }
    }
    let (input, projections) = match op {
        LogicalOperator::Project { input, projections } => (input, projections),
        _ => return Demand::Bindings,
    };

    let pattern = match PatternShape::of(input) {
        Some(p) => p,
        None => return Demand::Bindings,
    };
    if pattern.hops == 0 {
        return Demand::TerminalSet;
    }
    if pattern.cross_variable_filter {
        return Demand::Bindings;
    }

    let mut vars: Vec<&str> = Vec::new();
    let mut has_aggregate = false;
    let mut sensitive_aggregate = false;
    for p in projections {
        collect_value_vars(&p.expression, &mut vars);
        aggregate_flags(&p.expression, &mut has_aggregate, &mut sensitive_aggregate);
    }
    vars.sort_unstable();
    vars.dedup();

    let focus = match vars.as_slice() {
        [] => pattern.terminal.as_str(),
        [one] => one,
        _ => return Demand::Bindings,
    };
    if pattern.relationship_vars.iter().any(|r| r == focus) {
        return Demand::Bindings;
    }
    let terminal = focus == pattern.terminal;
    let sensitive = sensitive_aggregate || (!has_aggregate && !distinct);
    match (terminal, sensitive) {
        (true, false) => Demand::TerminalSet,
        (false, false) => Demand::EarlierSet,
        (true, true) => Demand::TerminalCount,
        (false, true) => Demand::EarlierCount,
    }
}

/// The pattern under a RETURN projection, as the classifier needs it.
struct PatternShape {
    hops: usize,
    terminal: String,
    relationship_vars: Vec<String>,
    cross_variable_filter: bool,
}

impl PatternShape {
    /// `None` for anything that is not one linear pattern: a `Join`, an
    /// `Unwind`, a nested projection (`WITH`), an unknown operator, hops that
    /// do not chain (each hop's source must be the next inner hop's target,
    /// or the scanned variable), or a variable bound twice. The last two are
    /// what the planner emits for `MATCH (a)->(b), (a)->(c)` and
    /// `(a)->(b)->(a)`: both need binding identity a forward mask chain lacks.
    fn of(op: &LogicalOperator) -> Option<Self> {
        let mut shape = PatternShape {
            hops: 0,
            terminal: String::new(),
            relationship_vars: Vec::new(),
            cross_variable_filter: false,
        };
        // The source the next inner hop (or the scan) must produce.
        let mut expect: Option<&str> = None;
        let mut bound: Vec<&str> = Vec::new();
        let mut op = op;
        loop {
            match op {
                LogicalOperator::Filter { input, predicate } => {
                    let mut vars = Vec::new();
                    collect_bool_vars(predicate, &mut vars);
                    vars.sort_unstable();
                    vars.dedup();
                    if vars.len() >= 2 {
                        shape.cross_variable_filter = true;
                    }
                    op = input;
                }
                LogicalOperator::Expand {
                    input,
                    source_variable,
                    target_variable,
                    relationship_variable,
                    ..
                }
                | LogicalOperator::VariableLengthExpand {
                    input,
                    source_variable,
                    target_variable,
                    relationship_variable,
                    ..
                } => {
                    if expect.is_some_and(|e| e != target_variable)
                        || bound.contains(&target_variable.as_str())
                    {
                        return None;
                    }
                    bound.push(target_variable);
                    expect = Some(source_variable);
                    // The outermost hop is the last one: its target is terminal.
                    if shape.hops == 0 {
                        shape.terminal = target_variable.clone();
                    }
                    shape.hops += 1;
                    if let Some(r) = relationship_variable {
                        shape.relationship_vars.push(r.clone());
                    }
                    op = input;
                }
                LogicalOperator::ScanByLabel { variable, .. } => {
                    if expect.is_some_and(|e| e != variable) || bound.contains(&variable.as_str()) {
                        return None;
                    }
                    if shape.hops == 0 {
                        shape.terminal = variable.clone();
                    }
                    return Some(shape);
                }
                _ => return None,
            }
        }
    }
}

fn collect_value_vars<'a>(v: &'a ValueExpression, out: &mut Vec<&'a str>) {
    match v {
        ValueExpression::Variable(name) => {
            if name != "*" {
                out.push(name);
            }
        }
        ValueExpression::Property(p) => out.push(&p.variable),
        ValueExpression::Literal(_)
        | ValueExpression::Parameter(_)
        | ValueExpression::VectorLiteral(_) => {}
        ValueExpression::ScalarFunction { args, .. }
        | ValueExpression::AggregateFunction { args, .. } => {
            for a in args {
                collect_value_vars(a, out);
            }
        }
        ValueExpression::Arithmetic { left, right, .. }
        | ValueExpression::VectorDistance { left, right, .. }
        | ValueExpression::VectorSimilarity { left, right, .. } => {
            collect_value_vars(left, out);
            collect_value_vars(right, out);
        }
    }
}

fn collect_bool_vars<'a>(e: &'a BooleanExpression, out: &mut Vec<&'a str>) {
    match e {
        BooleanExpression::Comparison { left, right, .. } => {
            collect_value_vars(left, out);
            collect_value_vars(right, out);
        }
        BooleanExpression::And(l, r) | BooleanExpression::Or(l, r) => {
            collect_bool_vars(l, out);
            collect_bool_vars(r, out);
        }
        BooleanExpression::Not(inner) => collect_bool_vars(inner, out),
        BooleanExpression::Exists(p) => out.push(&p.variable),
        BooleanExpression::In { expression, list } => {
            collect_value_vars(expression, out);
            for v in list {
                collect_value_vars(v, out);
            }
        }
        BooleanExpression::Like { expression, .. }
        | BooleanExpression::ILike { expression, .. }
        | BooleanExpression::Contains { expression, .. }
        | BooleanExpression::StartsWith { expression, .. }
        | BooleanExpression::EndsWith { expression, .. }
        | BooleanExpression::IsNull(expression)
        | BooleanExpression::IsNotNull(expression) => collect_value_vars(expression, out),
    }
}

/// `has` — any aggregate at all. `sensitive` — an aggregate whose answer
/// moves with the path count: a non-DISTINCT `count`/`sum`/`avg`/`collect`.
/// `min`/`max` and every DISTINCT aggregate see only the support.
fn aggregate_flags(v: &ValueExpression, has: &mut bool, sensitive: &mut bool) {
    match v {
        ValueExpression::AggregateFunction {
            name,
            args,
            distinct,
        } => {
            *has = true;
            let lower = name.to_lowercase();
            let bag = matches!(lower.as_str(), "count" | "sum" | "avg" | "collect");
            if bag && !*distinct {
                *sensitive = true;
            }
            for a in args {
                aggregate_flags(a, has, sensitive);
            }
        }
        ValueExpression::ScalarFunction { args, .. } => {
            for a in args {
                aggregate_flags(a, has, sensitive);
            }
        }
        ValueExpression::Arithmetic { left, right, .. }
        | ValueExpression::VectorDistance { left, right, .. }
        | ValueExpression::VectorSimilarity { left, right, .. } => {
            aggregate_flags(left, has, sensitive);
            aggregate_flags(right, has, sensitive);
        }
        _ => {}
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use lance_graph::config::GraphConfig;
    use lance_graph::logical_plan::LogicalPlanner;
    use lance_graph::parser::parse_cypher_query;
    use serde::Serialize;

    // --- #1305's tests, unchanged except for the call site ---

    fn semantics(query: &str) -> Demand {
        let ast = parse_cypher_query(query).unwrap();
        let config = GraphConfig::default();
        let mut planner = LogicalPlanner::new(&config);
        classify(&planner.plan(&ast).unwrap())
    }

    const TWO_HOP: &str = "MATCH (a:Person)-[:KNOWS]->(b:Person)-[:KNOWS]->(c:Person)";

    fn two_hop(tail: &str) -> Demand {
        semantics(&format!("{TWO_HOP} {tail}"))
    }

    #[test]
    fn terminal_set_consumers_need_only_the_terminal_support() {
        assert_eq!(two_hop("RETURN DISTINCT c.name"), Demand::TerminalSet);
        assert_eq!(
            two_hop("RETURN count(DISTINCT c) AS n"),
            Demand::TerminalSet
        );
        assert_eq!(two_hop("RETURN min(c.age) AS m"), Demand::TerminalSet);
    }

    #[test]
    fn an_earlier_variable_is_not_the_forward_frontier() {
        // count(DISTINCT b) = 3 on the §0 fixture while the forward dst₁ is 4.
        assert_eq!(two_hop("RETURN count(DISTINCT b) AS n"), Demand::EarlierSet);
        assert_eq!(two_hop("RETURN min(a.age) AS m"), Demand::EarlierSet);
    }

    #[test]
    fn path_counting_consumers_need_a_count_lane() {
        assert_eq!(two_hop("RETURN count(*) AS n"), Demand::TerminalCount);
        assert_eq!(two_hop("RETURN sum(c.age) AS s"), Demand::TerminalCount);
        // The pair that keeps T-11 honest: the same variable, with and without DISTINCT.
        assert_eq!(two_hop("RETURN c.name"), Demand::TerminalCount);
        assert_eq!(two_hop("RETURN DISTINCT c.name"), Demand::TerminalSet);
        assert_eq!(
            two_hop("RETURN c.name, count(*) AS n"),
            Demand::TerminalCount
        );
        assert_eq!(
            two_hop("RETURN a.name, count(*) AS n"),
            Demand::EarlierCount
        );
        assert_eq!(two_hop("RETURN sum(a.age) AS s"), Demand::EarlierCount);
    }

    #[test]
    fn two_variables_need_the_bindings() {
        assert_eq!(two_hop("RETURN a.name, c.name"), Demand::Bindings);
        assert_eq!(
            two_hop("WHERE a.age = c.age RETURN count(DISTINCT c) AS n"),
            Demand::Bindings
        );
        assert_eq!(
            semantics("MATCH (a:Person)-[:KNOWS]->(b:Person) WITH b RETURN count(*) AS n"),
            Demand::Bindings
        );
    }

    #[test]
    fn a_single_population_is_one_row_per_node() {
        // G3 — the can-stay-silent half: no hop, so a popcount is exact.
        assert_eq!(
            semantics("MATCH (n:Person) WHERE n.age > 30 RETURN count(*) AS n"),
            Demand::TerminalSet
        );
        assert_eq!(
            semantics("MATCH (n:Person) RETURN n.name"),
            Demand::TerminalSet
        );
        // A single-variable WHERE after a hop is not a cross-variable filter.
        assert_eq!(
            two_hop("WHERE c.age > 30 RETURN count(DISTINCT c) AS n"),
            Demand::TerminalSet
        );
    }

    #[test]
    fn hops_that_do_not_chain_are_not_a_forward_mask_chain() {
        // Two arms sharing `a`: the planner nests the Expands, but the outer
        // hop's source is `a`, not the inner hop's target `b`.
        assert_eq!(
            semantics(
                "MATCH (a:Person)-[:KNOWS]->(b:Person), (a)-[:KNOWS]->(c:Person) \
                 RETURN count(DISTINCT c) AS n"
            ),
            Demand::Bindings
        );
        // A variable bound twice closes a cycle: binding identity again.
        assert_eq!(
            semantics(
                "MATCH (a:Person)-[:KNOWS]->(b:Person)-[:KNOWS]->(a) \
                 RETURN count(DISTINCT b) AS n"
            ),
            Demand::Bindings
        );
        // Rebinding mid-chain: only the per-hop check sees it (the scan's
        // variable `a` is fresh here).
        assert_eq!(
            semantics(
                "MATCH (a:Person)-[:KNOWS]->(b:Person)-[:KNOWS]->(c:Person)-[:KNOWS]->(b) \
                 RETURN count(DISTINCT c) AS n"
            ),
            Demand::Bindings
        );
        // Silence twin: a genuine chain still classifies.
        assert_eq!(
            two_hop("RETURN count(DISTINCT c) AS n"),
            Demand::TerminalSet
        );
    }

    #[test]
    fn two_disconnected_patterns_are_not_a_single_population() {
        // G3's paired arm: no hop, but a Join — a cross product, not a popcount.
        assert_eq!(
            semantics("MATCH (a:Person), (b:Person) RETURN count(*) AS n"),
            Demand::Bindings
        );
    }

    /// `Demand` must never become persistable (§5.2 mint trip-wire).
    /// Inherent methods whose impl bound fails are skipped, so the call falls
    /// back to the trait method only when the type is NOT `Serialize`.
    #[test]
    fn consumer_semantics_is_not_serialize() {
        struct Probe<T>(std::marker::PhantomData<T>);
        trait NotSerialize {
            fn is_serialize(&self) -> bool {
                false
            }
        }
        impl<T> NotSerialize for Probe<T> {}
        impl<T: Serialize> Probe<T> {
            fn is_serialize(&self) -> bool {
                true
            }
        }
        assert!(!Probe::<Demand>(std::marker::PhantomData).is_serialize());
        // Can-fire: the probe does detect a Serialize type.
        assert!(Probe::<LogicalOperator>(std::marker::PhantomData).is_serialize());
    }
}
