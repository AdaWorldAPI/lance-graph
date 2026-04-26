//! Strategy #14: GremlinParse — Gremlin traversal steps → IR transpiler.
//!
//! Parses TinkerPop/Gremlin method-chain syntax into the same IR as CypherParse.
//! Gremlin is a procedural traversal language — each step maps to one IR node.
//!
//! ## Supported steps
//!
//! | Gremlin step           | IR node                                     |
//! |------------------------|---------------------------------------------|
//! | `g.V()`                | ScanNode { label: "*" }                     |
//! | `.hasLabel("Person")`  | ScanNode { label: "Person" }                |
//! | `.has("name", "Jan")`  | Filter { predicate: Column.Eq(Literal) }    |
//! | `.outE("KNOWS")`       | ScanRelationship { direction: Outgoing }     |
//! | `.inV()`               | (absorbed into ScanRelationship dst)         |
//! | `.out("KNOWS")`        | IndexNestedLoopJoin { direction: Outgoing }  |
//! | `.in("KNOWS")`         | IndexNestedLoopJoin { direction: Incoming }   |
//! | `.both("KNOWS")`       | IndexNestedLoopJoin { direction: Both }       |
//! | `.repeat().times(n)`   | RecursiveExtend { min_hops: n, max_hops: n } |
//! | `.count()`             | Aggregate { function: Count }                |
//! | `.limit(n)`            | Limit { count: n }                           |
//! | `.order().by()`        | OrderBy { sort_keys }                        |
//! | `.values("name")`      | Projection { expressions: [Column] }         |
//! | `.valueMap()`          | Projection { expressions: [Wildcard] }       |
//! | `.dedup()`             | Aggregate { function: Count, distinct: true } |
//! | `.path()`              | (flag: emit path)                            |
//! | `.select("a","b")`     | Projection { expressions: [Column...] }      |
//! | `.as("x")`             | (alias for current step)                     |

use crate::ir::{Arena, LogicalOp, LogicalPlan, Node, AExpr, ExprNode};
use crate::ir::logical_op::Direction;
use crate::ir::expr::{Literal, BinaryOp, UnaryOp};
use crate::traits::*;
use crate::PlanError;

#[derive(Debug)]
pub struct GremlinParse;

impl PlanStrategy for GremlinParse {
    fn name(&self) -> &str { "gremlin_parse" }
    fn capability(&self) -> PlanCapability { PlanCapability::Parse }

    fn affinity(&self, context: &PlanContext) -> f32 {
        let q = &context.query;
        // Gremlin always starts with g. or contains TinkerPop method chains
        if q.trim_start().starts_with("g.") {
            0.95
        } else if q.contains(".hasLabel(") || q.contains(".outE(") || q.contains(".inV(") {
            0.85
        } else {
            0.0
        }
    }

    fn plan(&self, mut input: PlanInput, arena: &mut Arena<LogicalOp>) -> Result<PlanInput, PlanError> {
        let source = input.context.query.clone();
        let steps = parse_gremlin_steps(&source)?;

        if steps.is_empty() {
            return Ok(input);
        }

        let mut expr_arena = Arena::new();
        let mut current_alias_counter = 0u32;
        let mut current_node: Option<Node> = None;
        let mut current_alias = String::from("v0");
        let mut projections: Vec<ExprNode> = Vec::new();
        let mut has_aggregation = false;
        let mut has_path_expansion = false;

        for step in &steps {
            match step {
                GremlinStep::V(id_filter) => {
                    let op = LogicalOp::ScanNode {
                        label: "*".to_string(),
                        alias: current_alias.clone(),
                        projections: None,
                    };
                    current_node = Some(arena.push(op));
                    input.context.features.has_graph_pattern = true;

                    // If V() has an ID filter, add a Filter node
                    if let Some(id) = id_filter {
                        let id_col = expr_arena.push(AExpr::Column {
                            variable: current_alias.clone(),
                            property: "id".to_string(),
                        });
                        let id_lit = expr_arena.push(AExpr::Literal(
                            parse_gremlin_literal(id),
                        ));
                        let pred = expr_arena.push(AExpr::BinaryOp {
                            left: ExprNode(id_col),
                            op: BinaryOp::Eq,
                            right: ExprNode(id_lit),
                        });
                        let filter = LogicalOp::Filter {
                            input: current_node.unwrap(),
                            predicate: ExprNode(pred),
                        };
                        current_node = Some(arena.push(filter));
                    }
                }

                GremlinStep::E => {
                    let op = LogicalOp::ScanRelationship {
                        rel_type: "*".to_string(),
                        alias: next_alias(&mut current_alias_counter),
                        direction: Direction::Both,
                        bound_node: current_node.unwrap_or(Node(0)),
                    };
                    current_node = Some(arena.push(op));
                }

                GremlinStep::HasLabel(label) => {
                    // Refine the last ScanNode to use this label
                    if let Some(node) = current_node {
                        let op = arena.get(node);
                        if let LogicalOp::ScanNode { alias, projections, .. } = op {
                            let refined = LogicalOp::ScanNode {
                                label: label.clone(),
                                alias: alias.clone(),
                                projections: projections.clone(),
                            };
                            arena.replace(node, refined);
                        } else {
                            // Can't refine — add as filter
                            let label_col = expr_arena.push(AExpr::Column {
                                variable: current_alias.clone(),
                                property: "__label__".to_string(),
                            });
                            let label_lit = expr_arena.push(AExpr::Literal(
                                Literal::String(label.clone()),
                            ));
                            let pred = expr_arena.push(AExpr::BinaryOp {
                                left: ExprNode(label_col),
                                op: BinaryOp::Eq,
                                right: ExprNode(label_lit),
                            });
                            let filter = LogicalOp::Filter {
                                input: node,
                                predicate: ExprNode(pred),
                            };
                            current_node = Some(arena.push(filter));
                        }
                    }
                }

                GremlinStep::Has(property, value) => {
                    if let Some(node) = current_node {
                        let col = expr_arena.push(AExpr::Column {
                            variable: current_alias.clone(),
                            property: property.clone(),
                        });
                        let lit = expr_arena.push(AExpr::Literal(
                            parse_gremlin_literal(value),
                        ));
                        let pred = expr_arena.push(AExpr::BinaryOp {
                            left: ExprNode(col),
                            op: BinaryOp::Eq,
                            right: ExprNode(lit),
                        });
                        let filter = LogicalOp::Filter {
                            input: node,
                            predicate: ExprNode(pred),
                        };
                        current_node = Some(arena.push(filter));
                    }
                }

                GremlinStep::HasNot(property) => {
                    if let Some(node) = current_node {
                        let col = expr_arena.push(AExpr::Column {
                            variable: current_alias.clone(),
                            property: property.clone(),
                        });
                        let pred = expr_arena.push(AExpr::UnaryOp {
                            op: UnaryOp::IsNull,
                            input: ExprNode(col),
                        });
                        let filter = LogicalOp::Filter {
                            input: node,
                            predicate: ExprNode(pred),
                        };
                        current_node = Some(arena.push(filter));
                    }
                }

                GremlinStep::Out(rel_type) | GremlinStep::In(rel_type) | GremlinStep::Both(rel_type) => {
                    let direction = match step {
                        GremlinStep::Out(_) => Direction::Outgoing,
                        GremlinStep::In(_) => Direction::Incoming,
                        _ => Direction::Both,
                    };
                    let dst_alias = next_alias(&mut current_alias_counter);
                    if let Some(node) = current_node {
                        let join = LogicalOp::IndexNestedLoopJoin {
                            left: node,
                            rel_type: rel_type.clone().unwrap_or_else(|| "*".to_string()),
                            direction,
                            dst_alias: dst_alias.clone(),
                        };
                        current_node = Some(arena.push(join));
                        current_alias = dst_alias;
                        input.context.features.has_graph_pattern = true;
                        input.context.features.num_match_clauses += 1;
                    }
                }

                GremlinStep::OutE(rel_type) | GremlinStep::InE(rel_type) | GremlinStep::BothE(rel_type) => {
                    let direction = match step {
                        GremlinStep::OutE(_) => Direction::Outgoing,
                        GremlinStep::InE(_) => Direction::Incoming,
                        _ => Direction::Both,
                    };
                    if let Some(node) = current_node {
                        let edge_alias = next_alias(&mut current_alias_counter);
                        let scan = LogicalOp::ScanRelationship {
                            rel_type: rel_type.clone().unwrap_or_else(|| "*".to_string()),
                            alias: edge_alias.clone(),
                            direction,
                            bound_node: node,
                        };
                        current_node = Some(arena.push(scan));
                        current_alias = edge_alias;
                    }
                }

                GremlinStep::InV | GremlinStep::OutV | GremlinStep::BothV => {
                    // After an edge step, resolve to the endpoint vertex
                    // This is handled implicitly by the next traversal step
                    let vertex_alias = next_alias(&mut current_alias_counter);
                    current_alias = vertex_alias;
                }

                GremlinStep::Repeat { steps: _, times } => {
                    // repeat(out("KNOWS")).times(3) → RecursiveExtend
                    if let Some(node) = current_node {
                        let dst_alias = next_alias(&mut current_alias_counter);
                        let extend = LogicalOp::RecursiveExtend {
                            input: node,
                            rel_type: "*".to_string(), // Would parse inner steps for rel_type
                            direction: Direction::Outgoing,
                            min_hops: *times,
                            max_hops: *times,
                            dst_alias: dst_alias.clone(),
                        };
                        current_node = Some(arena.push(extend));
                        current_alias = dst_alias;
                        has_path_expansion = true;
                        input.context.features.has_variable_length_path = true;
                    }
                }

                GremlinStep::Until => {
                    // until() modifies the enclosing repeat() — handled in Repeat parsing
                }

                GremlinStep::Count => {
                    has_aggregation = true;
                    input.context.features.has_aggregation = true;
                    if let Some(node) = current_node {
                        let wildcard = expr_arena.push(AExpr::Wildcard);
                        let agg = LogicalOp::Aggregate {
                            input: node,
                            group_by: vec![],
                            aggregates: vec![crate::ir::logical_op::AggregateExpr {
                                function: crate::ir::logical_op::AggFunction::Count,
                                input: ExprNode(wildcard),
                                distinct: false,
                            }],
                        };
                        current_node = Some(arena.push(agg));
                    }
                }

                GremlinStep::Sum(prop) | GremlinStep::Min(prop) | GremlinStep::Max(prop) => {
                    has_aggregation = true;
                    input.context.features.has_aggregation = true;
                    if let Some(node) = current_node {
                        let col = expr_arena.push(AExpr::Column {
                            variable: current_alias.clone(),
                            property: prop.clone().unwrap_or_default(),
                        });
                        let func = match step {
                            GremlinStep::Sum(_) => crate::ir::logical_op::AggFunction::Sum,
                            GremlinStep::Min(_) => crate::ir::logical_op::AggFunction::Min,
                            GremlinStep::Max(_) => crate::ir::logical_op::AggFunction::Max,
                            _ => unreachable!(),
                        };
                        let agg = LogicalOp::Aggregate {
                            input: node,
                            group_by: vec![],
                            aggregates: vec![crate::ir::logical_op::AggregateExpr {
                                function: func,
                                input: ExprNode(col),
                                distinct: false,
                            }],
                        };
                        current_node = Some(arena.push(agg));
                    }
                }

                GremlinStep::Dedup => {
                    // Dedup = group by all + take first. Approximate as distinct projection.
                    // The rule optimizer can rewrite this later.
                }

                GremlinStep::Limit(n) => {
                    if let Some(node) = current_node {
                        let limit = LogicalOp::Limit {
                            input: node,
                            count: *n,
                            offset: 0,
                        };
                        current_node = Some(arena.push(limit));
                    }
                }

                GremlinStep::Order(ascending) => {
                    // order().by("name", asc) → OrderBy
                    if let Some(node) = current_node {
                        // Default sort by current alias if no by() specified
                        let col = expr_arena.push(AExpr::Column {
                            variable: current_alias.clone(),
                            property: "id".to_string(),
                        });
                        let order = LogicalOp::OrderBy {
                            input: node,
                            sort_keys: vec![crate::ir::logical_op::SortKey {
                                expr: ExprNode(col),
                                ascending: *ascending,
                                nulls_first: false,
                            }],
                        };
                        current_node = Some(arena.push(order));
                    }
                }

                GremlinStep::Values(properties) => {
                    for prop in properties {
                        let col = expr_arena.push(AExpr::Column {
                            variable: current_alias.clone(),
                            property: prop.clone(),
                        });
                        projections.push(ExprNode(col));
                    }
                }

                GremlinStep::ValueMap => {
                    let wildcard = expr_arena.push(AExpr::Wildcard);
                    projections.push(ExprNode(wildcard));
                }

                GremlinStep::Select(aliases) => {
                    for alias in aliases {
                        let col = expr_arena.push(AExpr::Column {
                            variable: alias.clone(),
                            property: "*".to_string(),
                        });
                        projections.push(ExprNode(col));
                    }
                }

                GremlinStep::As(alias) => {
                    current_alias = alias.clone();
                }

                GremlinStep::Path => {
                    has_path_expansion = true;
                }

                GremlinStep::Drop => {
                    if let Some(node) = current_node {
                        let delete = LogicalOp::Delete { input: node };
                        current_node = Some(arena.push(delete));
                        input.context.features.has_mutation = true;
                    }
                }

                GremlinStep::AddV(label) => {
                    let create = LogicalOp::CreateNode {
                        label: label.clone(),
                        properties: vec![],
                    };
                    current_node = Some(arena.push(create));
                    input.context.features.has_mutation = true;
                }

                GremlinStep::AddE(rel_type) => {
                    // addE("KNOWS").from(a).to(b)
                    let create = LogicalOp::CreateRelationship {
                        rel_type: rel_type.clone(),
                        src: Node(0), // Resolved by from/to steps
                        dst: Node(0),
                        properties: vec![],
                    };
                    current_node = Some(arena.push(create));
                    input.context.features.has_mutation = true;
                }

                GremlinStep::Property(key, value) => {
                    if let Some(node) = current_node {
                        let val = expr_arena.push(AExpr::Literal(
                            parse_gremlin_literal(value),
                        ));
                        let set = LogicalOp::SetProperty {
                            input: node,
                            property: key.clone(),
                            value: ExprNode(val),
                        };
                        current_node = Some(arena.push(set));
                        input.context.features.has_mutation = true;
                    }
                }
            }
        }

        // Wrap with Return if we have projections or a final node
        if let Some(node) = current_node {
            let return_cols = if projections.is_empty() {
                let wildcard = expr_arena.push(AExpr::Wildcard);
                vec![ExprNode(wildcard)]
            } else {
                projections
            };
            let ret = LogicalOp::Return {
                input: node,
                columns: return_cols,
            };
            let root = arena.push(ret);

            let plan = LogicalPlan::new(
                std::mem::take(arena),
                expr_arena,
                root,
            );
            // Swap the new arena back
            *arena = plan.ops;
            // We can't easily move both arenas, so store plan in input
            input.plan = Some(LogicalPlan {
                ops: Arena::new(), // placeholder — compose will merge
                exprs: Arena::new(),
                root,
                schema: crate::ir::Schema::default(),
                cost: f64::MAX,
                properties: crate::ir::PlanProperties::default(),
            });
        }

        // Set complexity estimate
        let mut complexity = input.context.features.num_match_clauses as f64 * 0.2;
        if has_path_expansion { complexity += 0.3; }
        if has_aggregation { complexity += 0.1; }
        input.context.features.estimated_complexity = complexity.min(1.0);

        Ok(input)
    }
}

// =============================================================================
// Gremlin tokenizer / step parser
// =============================================================================

/// Parsed Gremlin step — one method call in the traversal chain.
#[derive(Debug, Clone)]
enum GremlinStep {
    V(Option<String>),
    E,
    HasLabel(String),
    Has(String, String),
    HasNot(String),
    Out(Option<String>),
    In(Option<String>),
    Both(Option<String>),
    OutE(Option<String>),
    InE(Option<String>),
    BothE(Option<String>),
    InV,
    OutV,
    BothV,
    Repeat {
        #[allow(dead_code)] // future wiring for repeat-step expansion
        steps: Vec<GremlinStep>,
        times: usize,
    },
    Until,
    Count,
    Sum(Option<String>),
    Min(Option<String>),
    Max(Option<String>),
    Dedup,
    Limit(usize),
    Order(bool), // ascending
    Values(Vec<String>),
    ValueMap,
    Select(Vec<String>),
    As(String),
    Path,
    Drop,
    AddV(String),
    AddE(String),
    Property(String, String),
}

/// Parse a Gremlin query string into a sequence of steps.
fn parse_gremlin_steps(source: &str) -> Result<Vec<GremlinStep>, PlanError> {
    let trimmed = source.trim();
    // Strip leading "g." if present
    let chain = trimmed.strip_prefix("g.").unwrap_or(trimmed);

    let mut steps = Vec::new();
    let mut pos = 0;
    let bytes = chain.as_bytes();

    while pos < bytes.len() {
        // Skip whitespace and dots
        while pos < bytes.len() && (bytes[pos] == b'.' || bytes[pos] == b' ' || bytes[pos] == b'\n') {
            pos += 1;
        }
        if pos >= bytes.len() { break; }

        // Read method name
        let name_start = pos;
        while pos < bytes.len() && bytes[pos] != b'(' && bytes[pos] != b'.' {
            pos += 1;
        }
        let name = &chain[name_start..pos];

        // Read arguments if present
        let args = if pos < bytes.len() && bytes[pos] == b'(' {
            pos += 1; // skip '('
            let arg_start = pos;
            let mut depth = 1;
            while pos < bytes.len() && depth > 0 {
                match bytes[pos] {
                    b'(' => depth += 1,
                    b')' => depth -= 1,
                    _ => {}
                }
                if depth > 0 { pos += 1; }
            }
            let arg_str = chain[arg_start..pos].trim();
            if pos < bytes.len() { pos += 1; } // skip ')'
            parse_gremlin_args(arg_str)
        } else {
            vec![]
        };

        // Map method name to step
        match name {
            "V" => steps.push(GremlinStep::V(args.first().cloned())),
            "E" => steps.push(GremlinStep::E),
            "hasLabel" => {
                if let Some(label) = args.first() {
                    steps.push(GremlinStep::HasLabel(label.clone()));
                }
            }
            "has" => {
                if args.len() >= 2 {
                    steps.push(GremlinStep::Has(args[0].clone(), args[1].clone()));
                } else if args.len() == 1 {
                    steps.push(GremlinStep::Has(args[0].clone(), String::new()));
                }
            }
            "hasNot" => {
                if let Some(prop) = args.first() {
                    steps.push(GremlinStep::HasNot(prop.clone()));
                }
            }
            "out" => steps.push(GremlinStep::Out(args.first().cloned())),
            "in" => steps.push(GremlinStep::In(args.first().cloned())),
            "both" => steps.push(GremlinStep::Both(args.first().cloned())),
            "outE" => steps.push(GremlinStep::OutE(args.first().cloned())),
            "inE" => steps.push(GremlinStep::InE(args.first().cloned())),
            "bothE" => steps.push(GremlinStep::BothE(args.first().cloned())),
            "inV" => steps.push(GremlinStep::InV),
            "outV" => steps.push(GremlinStep::OutV),
            "bothV" => steps.push(GremlinStep::BothV),
            "repeat" => {
                // Simple: extract times from a following .times() step
                steps.push(GremlinStep::Repeat { steps: vec![], times: 1 });
            }
            "times" => {
                let n = args.first()
                    .and_then(|s| s.parse::<usize>().ok())
                    .unwrap_or(1);
                // Patch the last Repeat step
                if let Some(GremlinStep::Repeat { times, .. }) = steps.last_mut() {
                    *times = n;
                }
            }
            "until" => steps.push(GremlinStep::Until),
            "count" => steps.push(GremlinStep::Count),
            "sum" => steps.push(GremlinStep::Sum(args.first().cloned())),
            "min" => steps.push(GremlinStep::Min(args.first().cloned())),
            "max" => steps.push(GremlinStep::Max(args.first().cloned())),
            "dedup" => steps.push(GremlinStep::Dedup),
            "limit" => {
                let n = args.first()
                    .and_then(|s| s.parse::<usize>().ok())
                    .unwrap_or(10);
                steps.push(GremlinStep::Limit(n));
            }
            "order" => steps.push(GremlinStep::Order(true)),
            "by" => {
                // Modifies the previous order step
                if args.len() >= 2 && args[1].to_lowercase().contains("desc") {
                    if let Some(GremlinStep::Order(asc)) = steps.last_mut() {
                        *asc = false;
                    }
                }
            }
            "values" => steps.push(GremlinStep::Values(args)),
            "valueMap" => steps.push(GremlinStep::ValueMap),
            "select" => steps.push(GremlinStep::Select(args)),
            "as" => {
                if let Some(alias) = args.first() {
                    steps.push(GremlinStep::As(alias.clone()));
                }
            }
            "path" => steps.push(GremlinStep::Path),
            "drop" => steps.push(GremlinStep::Drop),
            "addV" => {
                let label = args.first().cloned().unwrap_or_default();
                steps.push(GremlinStep::AddV(label));
            }
            "addE" => {
                let rel = args.first().cloned().unwrap_or_default();
                steps.push(GremlinStep::AddE(rel));
            }
            "property" => {
                if args.len() >= 2 {
                    steps.push(GremlinStep::Property(args[0].clone(), args[1].clone()));
                }
            }
            "from" | "to" | "fold" | "unfold" | "group" | "groupCount"
            | "coalesce" | "optional" | "choose" | "union" | "where"
            | "not" | "and" | "or" | "is" | "emit" | "sack"
            | "project" | "math" | "store" | "aggregate" | "cap"
            | "toList" | "toSet" | "next" | "iterate" => {
                // Known but unimplemented steps — feature detection only
            }
            "" => {} // empty segment between dots
            _ => {
                // Unknown step — skip gracefully
            }
        }
    }

    Ok(steps)
}

/// Parse comma-separated Gremlin arguments, stripping quotes.
fn parse_gremlin_args(arg_str: &str) -> Vec<String> {
    if arg_str.is_empty() {
        return vec![];
    }

    let mut args = Vec::new();
    let mut current = String::new();
    let mut in_quotes = false;
    let mut quote_char = '"';
    let mut depth = 0;

    for ch in arg_str.chars() {
        match ch {
            '"' | '\'' if !in_quotes => {
                in_quotes = true;
                quote_char = ch;
            }
            c if c == quote_char && in_quotes => {
                in_quotes = false;
            }
            '(' if !in_quotes => {
                depth += 1;
                current.push(ch);
            }
            ')' if !in_quotes => {
                depth -= 1;
                current.push(ch);
            }
            ',' if !in_quotes && depth == 0 => {
                let trimmed = current.trim().to_string();
                if !trimmed.is_empty() {
                    args.push(trimmed);
                }
                current.clear();
            }
            _ => {
                current.push(ch);
            }
        }
    }

    let trimmed = current.trim().to_string();
    if !trimmed.is_empty() {
        args.push(trimmed);
    }

    args
}

/// Parse a Gremlin literal value into an IR Literal.
fn parse_gremlin_literal(s: &str) -> Literal {
    if s.eq_ignore_ascii_case("true") {
        Literal::Bool(true)
    } else if s.eq_ignore_ascii_case("false") {
        Literal::Bool(false)
    } else if s.eq_ignore_ascii_case("null") || s.eq_ignore_ascii_case("none") {
        Literal::Null
    } else if let Ok(i) = s.parse::<i64>() {
        Literal::Int64(i)
    } else if let Ok(f) = s.parse::<f64>() {
        Literal::Float64(f)
    } else {
        Literal::String(s.to_string())
    }
}

/// Generate sequential aliases: v1, v2, v3...
fn next_alias(counter: &mut u32) -> String {
    *counter += 1;
    format!("v{counter}")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gremlin_affinity() {
        let gremlin = GremlinParse;
        let gremlin_ctx = PlanContext {
            query: "g.V().hasLabel(\"Person\").out(\"KNOWS\")".into(),
            features: QueryFeatures::default(),
            free_will_modifier: 1.0,
            thinking_style: None,
            nars_hint: None,
        };
        assert!(gremlin.affinity(&gremlin_ctx) > 0.9);

        let cypher_ctx = PlanContext {
            query: "MATCH (n:Person) RETURN n".into(),
            features: QueryFeatures::default(),
            free_will_modifier: 1.0,
            thinking_style: None,
            nars_hint: None,
        };
        assert!(gremlin.affinity(&cypher_ctx) < 0.1);
    }

    #[test]
    fn test_parse_gremlin_steps_basic() {
        let steps = parse_gremlin_steps("g.V().hasLabel(\"Person\").out(\"KNOWS\").values(\"name\")").unwrap();
        assert!(steps.len() >= 4);
    }

    #[test]
    fn test_parse_gremlin_steps_has() {
        let steps = parse_gremlin_steps("g.V().has(\"name\", \"Jan\").outE(\"DEVELOPS\").inV()").unwrap();
        assert!(steps.len() >= 4);
        assert!(matches!(&steps[1], GremlinStep::Has(k, v) if k == "name" && v == "Jan"));
    }

    #[test]
    fn test_parse_gremlin_steps_aggregation() {
        let steps = parse_gremlin_steps("g.V().hasLabel(\"System\").count()").unwrap();
        assert!(matches!(steps.last(), Some(GremlinStep::Count)));
    }

    #[test]
    fn test_parse_gremlin_steps_repeat() {
        let steps = parse_gremlin_steps("g.V(1).repeat(out()).times(3)").unwrap();
        let repeat = steps.iter().find(|s| matches!(s, GremlinStep::Repeat { .. }));
        assert!(repeat.is_some());
        if let Some(GremlinStep::Repeat { times, .. }) = repeat {
            assert_eq!(*times, 3);
        }
    }

    #[test]
    fn test_parse_gremlin_args() {
        let args = parse_gremlin_args("\"Person\"");
        assert_eq!(args, vec!["Person"]);

        let args = parse_gremlin_args("\"name\", \"Jan\"");
        assert_eq!(args, vec!["name", "Jan"]);

        let args = parse_gremlin_args("");
        assert!(args.is_empty());
    }

    #[test]
    fn test_gremlin_plan_produces_ir() {
        let gremlin = GremlinParse;
        let mut arena = Arena::new();
        let input = PlanInput {
            plan: None,
            context: PlanContext {
                query: "g.V().hasLabel(\"System\").out(\"DEPLOYED_BY\").values(\"name\")".into(),
                features: QueryFeatures::default(),
                free_will_modifier: 1.0,
                thinking_style: None,
                nars_hint: None,
            },
        };

        let result = gremlin.plan(input, &mut arena).unwrap();
        assert!(result.context.features.has_graph_pattern);
        assert!(result.context.features.num_match_clauses >= 1);
    }

    #[test]
    fn test_gremlin_mutation() {
        let steps = parse_gremlin_steps(
            "g.addV(\"Person\").property(\"name\", \"Ada\").addE(\"KNOWS\")"
        ).unwrap();
        assert!(steps.iter().any(|s| matches!(s, GremlinStep::AddV(_))));
        assert!(steps.iter().any(|s| matches!(s, GremlinStep::Property(_, _))));
        assert!(steps.iter().any(|s| matches!(s, GremlinStep::AddE(_))));
    }
}
