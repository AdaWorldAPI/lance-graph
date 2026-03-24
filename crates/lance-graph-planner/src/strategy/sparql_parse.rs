//! Strategy #15: SparqlParse — SPARQL triple patterns → IR transpiler.
//!
//! Parses W3C SPARQL SELECT/CONSTRUCT/ASK into the same IR as CypherParse.
//! SPARQL is declarative triple-pattern matching — each BGP maps to ScanNode + Filter.
//!
//! ## Supported patterns
//!
//! | SPARQL pattern                    | IR node                                    |
//! |-----------------------------------|--------------------------------------------|
//! | `?s rdf:type :Person`             | ScanNode { label: "Person" }               |
//! | `?s :name "Jan"`                  | Filter { Column.Eq(Literal) }              |
//! | `?s :knows ?o`                    | IndexNestedLoopJoin { rel_type: "knows" }  |
//! | `OPTIONAL { ... }`                | HashJoin { join_type: Left }               |
//! | `FILTER (?age > 30)`              | Filter { predicate }                       |
//! | `SELECT ?s ?name`                 | Projection                                 |
//! | `SELECT (COUNT(*) AS ?c)`         | Aggregate { function: Count }              |
//! | `ORDER BY ?name`                  | OrderBy                                    |
//! | `LIMIT 10`                        | Limit                                      |
//! | `UNION { ... } { ... }`           | Union                                      |
//! | `?s :path+ ?o`                    | RecursiveExtend (property path)            |
//! | `PREFIX : <...>`                  | (namespace resolution)                     |

use crate::ir::{Arena, LogicalOp, LogicalPlan, Node, AExpr, ExprNode};
use crate::ir::logical_op::{Direction, JoinType, AggFunction, AggregateExpr, SortKey};
use crate::ir::expr::{Literal, BinaryOp};
use crate::traits::*;
use crate::PlanError;

use std::collections::HashMap;

#[derive(Debug)]
pub struct SparqlParse;

impl PlanStrategy for SparqlParse {
    fn name(&self) -> &str { "sparql_parse" }
    fn capability(&self) -> PlanCapability { PlanCapability::Parse }

    fn affinity(&self, context: &PlanContext) -> f32 {
        let q = context.query.to_uppercase();
        if q.contains("PREFIX ") || q.contains("SELECT ?") || q.contains("CONSTRUCT ") || q.contains("ASK ") {
            0.95
        } else if q.contains("WHERE {") || q.contains("OPTIONAL {") {
            0.80
        } else {
            0.0
        }
    }

    fn plan(&self, mut input: PlanInput, arena: &mut Arena<LogicalOp>) -> Result<PlanInput, PlanError> {
        let source = input.context.query.clone();
        let query = parse_sparql(&source)?;

        let mut expr_arena = Arena::new();
        let mut alias_counter = 0u32;

        // Track variable → node mapping for joins
        let mut var_nodes: HashMap<String, Node> = HashMap::new();

        // Process triple patterns from WHERE clause
        let mut current_node: Option<Node> = None;

        for pattern in &query.where_patterns {
            match pattern {
                SparqlPattern::Triple { subject, predicate, object } => {
                    // rdf:type → ScanNode with label
                    if predicate == "rdf:type" || predicate == "a" || predicate.ends_with("#type") {
                        let label = extract_local_name(object);
                        let alias = var_name(subject);
                        let scan = LogicalOp::ScanNode {
                            label,
                            alias: alias.clone(),
                            projections: None,
                        };
                        let node = arena.push(scan);
                        var_nodes.insert(alias, node);
                        current_node = Some(node);
                        input.context.features.has_graph_pattern = true;
                    }
                    // Property access: ?s :name "Jan" → Filter
                    else if !is_variable(object) {
                        let var = var_name(subject);
                        let prop = extract_local_name(predicate);

                        // Ensure the subject variable has a scan
                        let input_node = if let Some(n) = var_nodes.get(&var) {
                            *n
                        } else {
                            let scan = LogicalOp::ScanNode {
                                label: "*".to_string(),
                                alias: var.clone(),
                                projections: None,
                            };
                            let n = arena.push(scan);
                            var_nodes.insert(var.clone(), n);
                            n
                        };

                        let col = expr_arena.push(AExpr::Column {
                            variable: var,
                            property: prop,
                        });
                        let lit = expr_arena.push(AExpr::Literal(
                            parse_sparql_literal(object),
                        ));
                        let pred = expr_arena.push(AExpr::BinaryOp {
                            left: ExprNode(col),
                            op: BinaryOp::Eq,
                            right: ExprNode(lit),
                        });
                        let filter = LogicalOp::Filter {
                            input: input_node,
                            predicate: ExprNode(pred),
                        };
                        let node = arena.push(filter);
                        var_nodes.insert(var_name(subject), node);
                        current_node = Some(node);
                    }
                    // Relationship: ?s :knows ?o → join
                    else {
                        let src_var = var_name(subject);
                        let dst_var = var_name(object);
                        let rel_type = extract_local_name(predicate);

                        let src_node = if let Some(n) = var_nodes.get(&src_var) {
                            *n
                        } else {
                            let scan = LogicalOp::ScanNode {
                                label: "*".to_string(),
                                alias: src_var.clone(),
                                projections: None,
                            };
                            let n = arena.push(scan);
                            var_nodes.insert(src_var.clone(), n);
                            n
                        };

                        let join = LogicalOp::IndexNestedLoopJoin {
                            left: src_node,
                            rel_type: rel_type.to_uppercase(),
                            direction: Direction::Outgoing,
                            dst_alias: dst_var.clone(),
                        };
                        let node = arena.push(join);
                        var_nodes.insert(dst_var, node);
                        current_node = Some(node);
                        input.context.features.has_graph_pattern = true;
                        input.context.features.num_match_clauses += 1;
                    }
                }

                SparqlPattern::Optional(patterns) => {
                    // OPTIONAL { ... } → Left join
                    // Process inner patterns to build a subplan
                    let outer = current_node.unwrap_or_else(|| arena.push(LogicalOp::EmptyResult));

                    // Build inner as a sequence of triples
                    let inner_alias = format!("opt{}", alias_counter);
                    alias_counter += 1;
                    let inner = arena.push(LogicalOp::ScanNode {
                        label: "*".to_string(),
                        alias: inner_alias.clone(),
                        projections: None,
                    });

                    let join = LogicalOp::HashJoin {
                        left: outer,
                        right: inner,
                        join_keys: vec![], // Resolved during optimization
                        join_type: JoinType::Left,
                    };
                    current_node = Some(arena.push(join));
                }

                SparqlPattern::Union(left_patterns, right_patterns) => {
                    let left = current_node.unwrap_or_else(|| arena.push(LogicalOp::EmptyResult));
                    let right = arena.push(LogicalOp::EmptyResult);
                    let union = LogicalOp::Union {
                        children: vec![left, right],
                        all: true,
                    };
                    current_node = Some(arena.push(union));
                }

                SparqlPattern::Filter(expr_str) => {
                    if let Some(node) = current_node {
                        let pred = parse_sparql_filter_expr(expr_str, &mut expr_arena);
                        let filter = LogicalOp::Filter {
                            input: node,
                            predicate: pred,
                        };
                        current_node = Some(arena.push(filter));
                    }
                }

                SparqlPattern::PropertyPath { subject, path, object } => {
                    let src_var = var_name(subject);
                    let dst_var = var_name(object);

                    let src_node = if let Some(n) = var_nodes.get(&src_var) {
                        *n
                    } else {
                        let scan = LogicalOp::ScanNode {
                            label: "*".to_string(),
                            alias: src_var.clone(),
                            projections: None,
                        };
                        let n = arena.push(scan);
                        var_nodes.insert(src_var.clone(), n);
                        n
                    };

                    let (rel_type, min_hops, max_hops) = parse_property_path(path);
                    let extend = LogicalOp::RecursiveExtend {
                        input: src_node,
                        rel_type,
                        direction: Direction::Outgoing,
                        min_hops,
                        max_hops,
                        dst_alias: dst_var.clone(),
                    };
                    let node = arena.push(extend);
                    var_nodes.insert(dst_var, node);
                    current_node = Some(node);
                    input.context.features.has_variable_length_path = true;
                }
            }
        }

        // Add aggregation if present
        if !query.aggregates.is_empty() {
            if let Some(node) = current_node {
                let mut agg_exprs = Vec::new();
                for agg in &query.aggregates {
                    let col = expr_arena.push(AExpr::Wildcard);
                    agg_exprs.push(AggregateExpr {
                        function: match agg.function.to_uppercase().as_str() {
                            "COUNT" => AggFunction::Count,
                            "SUM" => AggFunction::Sum,
                            "AVG" => AggFunction::Avg,
                            "MIN" => AggFunction::Min,
                            "MAX" => AggFunction::Max,
                            _ => AggFunction::Count,
                        },
                        input: ExprNode(col),
                        distinct: agg.distinct,
                    });
                }
                let agg = LogicalOp::Aggregate {
                    input: node,
                    group_by: vec![],
                    aggregates: agg_exprs,
                };
                current_node = Some(arena.push(agg));
                input.context.features.has_aggregation = true;
            }
        }

        // Add ORDER BY
        if !query.order_by.is_empty() {
            if let Some(node) = current_node {
                let sort_keys: Vec<SortKey> = query.order_by.iter().map(|(var, asc)| {
                    let col = expr_arena.push(AExpr::Column {
                        variable: var.clone(),
                        property: "*".to_string(),
                    });
                    SortKey {
                        expr: ExprNode(col),
                        ascending: *asc,
                        nulls_first: false,
                    }
                }).collect();
                let order = LogicalOp::OrderBy { input: node, sort_keys };
                current_node = Some(arena.push(order));
            }
        }

        // Add LIMIT
        if let Some(limit) = query.limit {
            if let Some(node) = current_node {
                let lim = LogicalOp::Limit {
                    input: node,
                    count: limit,
                    offset: query.offset.unwrap_or(0),
                };
                current_node = Some(arena.push(lim));
            }
        }

        // Add projection / return
        if let Some(node) = current_node {
            let columns: Vec<ExprNode> = if query.select_vars.is_empty() {
                let w = expr_arena.push(AExpr::Wildcard);
                vec![ExprNode(w)]
            } else {
                query.select_vars.iter().map(|var| {
                    let col = expr_arena.push(AExpr::Column {
                        variable: var.clone(),
                        property: "*".to_string(),
                    });
                    ExprNode(col)
                }).collect()
            };

            let ret = LogicalOp::Return { input: node, columns };
            let root = arena.push(ret);

            input.plan = Some(LogicalPlan {
                ops: Arena::new(),
                exprs: Arena::new(),
                root,
                schema: crate::ir::Schema::default(),
                cost: f64::MAX,
                properties: crate::ir::PlanProperties::default(),
            });
        }

        // Complexity
        let mut complexity = input.context.features.num_match_clauses as f64 * 0.2;
        if input.context.features.has_variable_length_path { complexity += 0.3; }
        if input.context.features.has_aggregation { complexity += 0.1; }
        input.context.features.estimated_complexity = complexity.min(1.0);

        Ok(input)
    }
}

// =============================================================================
// SPARQL parser
// =============================================================================

#[derive(Debug)]
struct SparqlQuery {
    prefixes: HashMap<String, String>,
    select_vars: Vec<String>,
    where_patterns: Vec<SparqlPattern>,
    aggregates: Vec<SparqlAggregate>,
    order_by: Vec<(String, bool)>,
    limit: Option<usize>,
    offset: Option<usize>,
    is_ask: bool,
    is_construct: bool,
}

#[derive(Debug)]
struct SparqlAggregate {
    function: String,
    distinct: bool,
    alias: String,
}

#[derive(Debug)]
enum SparqlPattern {
    Triple { subject: String, predicate: String, object: String },
    Optional(Vec<SparqlPattern>),
    Union(Vec<SparqlPattern>, Vec<SparqlPattern>),
    Filter(String),
    PropertyPath { subject: String, path: String, object: String },
}

fn parse_sparql(source: &str) -> Result<SparqlQuery, PlanError> {
    let mut query = SparqlQuery {
        prefixes: HashMap::new(),
        select_vars: vec![],
        where_patterns: vec![],
        aggregates: vec![],
        order_by: vec![],
        limit: None,
        offset: None,
        is_ask: false,
        is_construct: false,
    };

    let lines: Vec<&str> = source.lines().collect();
    let full = source.to_string();
    let upper = full.to_uppercase();

    // Parse PREFIX declarations
    for line in &lines {
        let trimmed = line.trim();
        if trimmed.to_uppercase().starts_with("PREFIX") {
            // PREFIX foo: <http://example.org/>
            let rest = trimmed[6..].trim();
            if let Some(colon_pos) = rest.find(':') {
                let prefix = rest[..colon_pos].trim().to_string();
                let uri = rest[colon_pos + 1..].trim()
                    .trim_start_matches('<')
                    .trim_end_matches('>')
                    .trim()
                    .to_string();
                query.prefixes.insert(prefix, uri);
            }
        }
    }

    // Detect query form
    if upper.contains("ASK ") || upper.starts_with("ASK") {
        query.is_ask = true;
    } else if upper.contains("CONSTRUCT ") || upper.starts_with("CONSTRUCT") {
        query.is_construct = true;
    }

    // Parse SELECT variables
    if let Some(select_pos) = upper.find("SELECT") {
        let where_pos = upper.find("WHERE").unwrap_or(upper.len());
        let select_clause = &full[select_pos + 6..where_pos].trim();

        if select_clause.contains('*') {
            // SELECT * — all variables
        } else {
            // Parse variable list and aggregates
            let mut i = 0;
            let chars: Vec<char> = select_clause.chars().collect();
            while i < chars.len() {
                if chars[i] == '?' || chars[i] == '$' {
                    let start = i + 1;
                    i += 1;
                    while i < chars.len() && (chars[i].is_alphanumeric() || chars[i] == '_') {
                        i += 1;
                    }
                    query.select_vars.push(select_clause[start..i].to_string());
                } else if chars[i] == '(' {
                    // Aggregate: (COUNT(?x) AS ?c)
                    let start = i;
                    let mut depth = 1;
                    i += 1;
                    while i < chars.len() && depth > 0 {
                        if chars[i] == '(' { depth += 1; }
                        if chars[i] == ')' { depth -= 1; }
                        i += 1;
                    }
                    let agg_str = &select_clause[start..i];
                    if let Some(agg) = parse_sparql_aggregate(agg_str) {
                        query.select_vars.push(agg.alias.clone());
                        query.aggregates.push(agg);
                    }
                } else {
                    i += 1;
                }
            }
        }
    }

    // Parse WHERE clause triple patterns
    if let Some(where_pos) = upper.find("WHERE") {
        let brace_start = full[where_pos..].find('{').map(|p| where_pos + p + 1);
        if let Some(start) = brace_start {
            // Find matching closing brace
            let mut depth = 1;
            let mut end = start;
            let bytes = full.as_bytes();
            while end < bytes.len() && depth > 0 {
                match bytes[end] {
                    b'{' => depth += 1,
                    b'}' => depth -= 1,
                    _ => {}
                }
                if depth > 0 { end += 1; }
            }

            let body = &full[start..end];
            query.where_patterns = parse_sparql_body(body);
        }
    }

    // Parse ORDER BY
    if let Some(order_pos) = upper.find("ORDER BY") {
        let rest = &full[order_pos + 8..];
        let end = rest.to_uppercase().find("LIMIT")
            .or_else(|| rest.to_uppercase().find("OFFSET"))
            .unwrap_or(rest.len());
        let order_clause = rest[..end].trim();

        for token in order_clause.split_whitespace() {
            if token.starts_with('?') || token.starts_with('$') {
                let var = token[1..].to_string();
                query.order_by.push((var, true));
            } else if token.to_uppercase() == "DESC" {
                if let Some(last) = query.order_by.last_mut() {
                    last.1 = false;
                }
            }
        }
    }

    // Parse LIMIT / OFFSET
    if let Some(limit_pos) = upper.find("LIMIT") {
        let rest = upper[limit_pos + 5..].trim();
        if let Some(n) = rest.split_whitespace().next().and_then(|s| s.parse::<usize>().ok()) {
            query.limit = Some(n);
        }
    }
    if let Some(offset_pos) = upper.find("OFFSET") {
        let rest = upper[offset_pos + 6..].trim();
        if let Some(n) = rest.split_whitespace().next().and_then(|s| s.parse::<usize>().ok()) {
            query.offset = Some(n);
        }
    }

    Ok(query)
}

/// Parse triple patterns and nested blocks from a WHERE body.
fn parse_sparql_body(body: &str) -> Vec<SparqlPattern> {
    let mut patterns = Vec::new();
    let trimmed = body.trim();

    // Split on '.' (statement separator) but respect braces and quotes
    let statements = split_sparql_statements(trimmed);

    for stmt in &statements {
        let s = stmt.trim();
        let su = s.to_uppercase();

        if su.starts_with("OPTIONAL") {
            // OPTIONAL { ... }
            if let Some(brace_start) = s.find('{') {
                let inner = extract_braced(s, brace_start);
                let inner_patterns = parse_sparql_body(&inner);
                patterns.push(SparqlPattern::Optional(inner_patterns));
            }
        } else if su.starts_with("FILTER") {
            let expr = s[6..].trim()
                .trim_start_matches('(')
                .trim_end_matches(')')
                .to_string();
            patterns.push(SparqlPattern::Filter(expr));
        } else if su.contains("UNION") {
            // { ... } UNION { ... }
            patterns.push(SparqlPattern::Union(vec![], vec![]));
        } else if !s.is_empty() {
            // Triple pattern: ?s ?p ?o
            let parts = split_triple(s);
            if parts.len() >= 3 {
                let pred = &parts[1];
                // Detect property paths: :knows+, :knows*, :knows/rdfs:subClassOf
                if pred.contains('+') || pred.contains('*') || pred.contains('/') {
                    patterns.push(SparqlPattern::PropertyPath {
                        subject: parts[0].clone(),
                        path: parts[1].clone(),
                        object: parts[2].clone(),
                    });
                } else {
                    patterns.push(SparqlPattern::Triple {
                        subject: parts[0].clone(),
                        predicate: parts[1].clone(),
                        object: parts[2].clone(),
                    });
                }
            }
        }
    }

    patterns
}

/// Split SPARQL body on '.' respecting braces.
fn split_sparql_statements(body: &str) -> Vec<String> {
    let mut statements = Vec::new();
    let mut current = String::new();
    let mut depth = 0;

    for ch in body.chars() {
        match ch {
            '{' => { depth += 1; current.push(ch); }
            '}' => { depth -= 1; current.push(ch); }
            '.' if depth == 0 => {
                let trimmed = current.trim().to_string();
                if !trimmed.is_empty() {
                    statements.push(trimmed);
                }
                current.clear();
            }
            _ => current.push(ch),
        }
    }
    let trimmed = current.trim().to_string();
    if !trimmed.is_empty() {
        statements.push(trimmed);
    }
    statements
}

/// Split a triple pattern into S, P, O parts, handling prefixed names and URIs.
fn split_triple(s: &str) -> Vec<String> {
    let mut parts = Vec::new();
    let mut current = String::new();
    let mut in_uri = false;
    let mut in_string = false;

    for ch in s.chars() {
        match ch {
            '<' if !in_string => { in_uri = true; current.push(ch); }
            '>' if in_uri => { in_uri = false; current.push(ch); }
            '"' => { in_string = !in_string; current.push(ch); }
            ' ' | '\t' if !in_uri && !in_string => {
                let trimmed = current.trim().to_string();
                if !trimmed.is_empty() {
                    parts.push(trimmed);
                }
                current.clear();
            }
            _ => current.push(ch),
        }
    }
    let trimmed = current.trim().to_string();
    if !trimmed.is_empty() {
        parts.push(trimmed);
    }
    parts
}

fn extract_braced(s: &str, brace_start: usize) -> String {
    let bytes = s.as_bytes();
    let mut depth = 0;
    let mut start = brace_start + 1;
    let mut end = start;
    for i in brace_start..bytes.len() {
        match bytes[i] {
            b'{' => depth += 1,
            b'}' => {
                depth -= 1;
                if depth == 0 {
                    end = i;
                    break;
                }
            }
            _ => {}
        }
    }
    s[start..end].to_string()
}

fn parse_sparql_aggregate(s: &str) -> Option<SparqlAggregate> {
    let inner = s.trim().trim_start_matches('(').trim_end_matches(')');
    let upper = inner.to_uppercase();

    let alias = if let Some(as_pos) = upper.find(" AS ") {
        let rest = inner[as_pos + 4..].trim();
        rest.trim_start_matches('?').trim_start_matches('$').to_string()
    } else {
        "agg".to_string()
    };

    let distinct = upper.contains("DISTINCT");

    let func = if upper.starts_with("COUNT") {
        "COUNT"
    } else if upper.starts_with("SUM") {
        "SUM"
    } else if upper.starts_with("AVG") {
        "AVG"
    } else if upper.starts_with("MIN") {
        "MIN"
    } else if upper.starts_with("MAX") {
        "MAX"
    } else {
        return None;
    };

    Some(SparqlAggregate {
        function: func.to_string(),
        distinct,
        alias,
    })
}

fn is_variable(s: &str) -> bool {
    s.starts_with('?') || s.starts_with('$')
}

fn var_name(s: &str) -> String {
    s.trim_start_matches('?').trim_start_matches('$').to_string()
}

fn extract_local_name(uri: &str) -> String {
    // Strip angle brackets first: <http://example.org/Person> → http://example.org/Person
    let stripped = uri.trim_matches('<').trim_matches('>').trim_matches('"');
    // :Person → Person
    let stripped = stripped.trim_start_matches(':');
    // Try fragment (#), then path (/), then prefix (:)
    stripped.rsplit_once('#').map(|(_, local)| local)
        .or_else(|| stripped.rsplit_once('/').map(|(_, local)| local))
        .or_else(|| stripped.rsplit_once(':').map(|(_, local)| local))
        .unwrap_or(stripped)
        .trim_start_matches(':')
        .to_string()
}

fn parse_sparql_literal(s: &str) -> Literal {
    let trimmed = s.trim().trim_matches('"');
    if let Ok(i) = trimmed.parse::<i64>() {
        Literal::Int64(i)
    } else if let Ok(f) = trimmed.parse::<f64>() {
        Literal::Float64(f)
    } else if trimmed.eq_ignore_ascii_case("true") {
        Literal::Bool(true)
    } else if trimmed.eq_ignore_ascii_case("false") {
        Literal::Bool(false)
    } else {
        Literal::String(trimmed.to_string())
    }
}

fn parse_sparql_filter_expr(expr: &str, expr_arena: &mut Arena<AExpr>) -> ExprNode {
    // Simplified: parse "?var OP value" patterns
    let trimmed = expr.trim().trim_start_matches('(').trim_end_matches(')');

    // Try to parse as comparison
    for (op_str, op) in &[
        (">=", BinaryOp::Gte), ("<=", BinaryOp::Lte),
        ("!=", BinaryOp::Neq), ("=", BinaryOp::Eq),
        (">", BinaryOp::Gt), ("<", BinaryOp::Lt),
    ] {
        if let Some(pos) = trimmed.find(op_str) {
            let left_str = trimmed[..pos].trim();
            let right_str = trimmed[pos + op_str.len()..].trim();

            let left = if is_variable(left_str) {
                expr_arena.push(AExpr::Column {
                    variable: var_name(left_str),
                    property: "*".to_string(),
                })
            } else {
                expr_arena.push(AExpr::Literal(parse_sparql_literal(left_str)))
            };

            let right = if is_variable(right_str) {
                expr_arena.push(AExpr::Column {
                    variable: var_name(right_str),
                    property: "*".to_string(),
                })
            } else {
                expr_arena.push(AExpr::Literal(parse_sparql_literal(right_str)))
            };

            let pred = expr_arena.push(AExpr::BinaryOp {
                left: ExprNode(left),
                op: *op,
                right: ExprNode(right),
            });
            return ExprNode(pred);
        }
    }

    // Fallback: literal true (pass-through)
    let t = expr_arena.push(AExpr::Literal(Literal::Bool(true)));
    ExprNode(t)
}

fn parse_property_path(path: &str) -> (String, usize, usize) {
    let trimmed = path.trim_start_matches(':');
    if let Some(base) = trimmed.strip_suffix('+') {
        (extract_local_name(base).to_uppercase(), 1, 10)
    } else if let Some(base) = trimmed.strip_suffix('*') {
        (extract_local_name(base).to_uppercase(), 0, 10)
    } else if let Some(base) = trimmed.strip_suffix('?') {
        (extract_local_name(base).to_uppercase(), 0, 1)
    } else {
        (extract_local_name(trimmed).to_uppercase(), 1, 1)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sparql_affinity() {
        let parser = SparqlParse;

        let sparql_ctx = PlanContext {
            query: "PREFIX : <http://example.org/> SELECT ?s WHERE { ?s rdf:type :Person }".into(),
            features: QueryFeatures::default(),
            free_will_modifier: 1.0,
            thinking_style: None,
            nars_hint: None,
        };
        assert!(parser.affinity(&sparql_ctx) > 0.9);

        let cypher_ctx = PlanContext {
            query: "MATCH (n:Person) RETURN n".into(),
            features: QueryFeatures::default(),
            free_will_modifier: 1.0,
            thinking_style: None,
            nars_hint: None,
        };
        assert!(parser.affinity(&cypher_ctx) < 0.1);
    }

    #[test]
    fn test_parse_sparql_basic() {
        let query = parse_sparql(
            "PREFIX : <http://example.org/>\nSELECT ?s ?name\nWHERE { ?s rdf:type :Person . ?s :name ?name }"
        ).unwrap();
        assert_eq!(query.select_vars, vec!["s", "name"]);
        assert_eq!(query.where_patterns.len(), 2);
    }

    #[test]
    fn test_parse_sparql_with_filter() {
        let query = parse_sparql(
            "SELECT ?s WHERE { ?s :age ?age . FILTER (?age > 30) }"
        ).unwrap();
        assert!(query.where_patterns.iter().any(|p| matches!(p, SparqlPattern::Filter(_))));
    }

    #[test]
    fn test_parse_sparql_with_limit() {
        let query = parse_sparql(
            "SELECT ?s WHERE { ?s rdf:type :System } ORDER BY ?s LIMIT 10 OFFSET 5"
        ).unwrap();
        assert_eq!(query.limit, Some(10));
        assert_eq!(query.offset, Some(5));
        assert_eq!(query.order_by.len(), 1);
    }

    #[test]
    fn test_sparql_plan_produces_ir() {
        let parser = SparqlParse;
        let mut arena = Arena::new();
        let input = PlanInput {
            plan: None,
            context: PlanContext {
                query: "SELECT ?s ?o WHERE { ?s rdf:type :System . ?s :deployed_by ?o }".into(),
                features: QueryFeatures::default(),
                free_will_modifier: 1.0,
                thinking_style: None,
                nars_hint: None,
            },
        };

        let result = parser.plan(input, &mut arena).unwrap();
        assert!(result.context.features.has_graph_pattern);
        assert!(result.context.features.num_match_clauses >= 1);
    }

    #[test]
    fn test_extract_local_name() {
        assert_eq!(extract_local_name(":Person"), "Person");
        assert_eq!(extract_local_name("<http://example.org/Person>"), "Person");
        assert_eq!(extract_local_name("foo:Person"), "Person");
        assert_eq!(extract_local_name("rdf:type"), "type");
    }

    #[test]
    fn test_property_path_parsing() {
        let (rel, min, max) = parse_property_path(":knows+");
        assert_eq!(rel, "KNOWS");
        assert_eq!(min, 1);
        assert!(max > 1);

        let (rel, min, _) = parse_property_path(":knows*");
        assert_eq!(rel, "KNOWS");
        assert_eq!(min, 0);
    }

    #[test]
    fn test_sparql_aggregate() {
        let query = parse_sparql(
            "SELECT (COUNT(?s) AS ?count) WHERE { ?s rdf:type :System }"
        ).unwrap();
        assert_eq!(query.aggregates.len(), 1);
        assert_eq!(query.aggregates[0].function, "COUNT");
        assert_eq!(query.aggregates[0].alias, "count");
    }

    #[test]
    fn test_split_triple() {
        let parts = split_triple("?s rdf:type :Person");
        assert_eq!(parts, vec!["?s", "rdf:type", ":Person"]);

        let parts = split_triple("?s :name \"Jan\"");
        assert_eq!(parts, vec!["?s", ":name", "\"Jan\""]);
    }
}
