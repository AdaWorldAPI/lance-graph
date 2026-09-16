//! **W0-b — the corpus census.** `cypher-mask-lowering-v1.md` §7.0's second
//! measurement, and half of the plan's own STOP gate:
//!
//! > *"Take a real Cypher corpus (the parser's own 44 tests are the floor …).
//! > Classify every query by §3/§4: **what fraction lowers fully, what fraction
//! > splits, what fraction is pure grace?** (OQ-5.) If the full-lowering
//! > fraction is negligible, this plan's premise is wrong and Wave 1 does not
//! > start."*
//!
//! This is MEASUREMENT, not production code: it builds no mask program, mints
//! no byte, and changes no behaviour. It reports a number so that Wave 1 can be
//! started or abandoned on evidence.
//!
//! # The corpus is the committed corpus, by construction
//!
//! The query strings are not transcribed here — they are `include_str!`-ed out
//! of the three committed sources that hold them and extracted at runtime. A
//! transcribed corpus drifts silently from the tests it claims to mirror; an
//! extracted one cannot. Strings that are not queries (the planner's own error
//! messages, which also contain the word `MATCH`) self-eliminate: they fail to
//! parse and are reported in their own bucket rather than counted.
//!
//! # The classification is §3/§4's, not an opinion
//!
//! Every disposition below cites the plan row that rules it. The three outcomes
//! are §2's:
//!
//! - **Full** — every operator and every expression in the plan lowers.
//! - **Split** — at least one lowers and at least one does not (§2 property 3:
//!   *"the realistic query is not 'all mask' or 'all SQL'"*).
//! - **Grace** — nothing lowers.
//!
//! Run: `cargo run -p lance-graph --example w0b_corpus_census`

use std::collections::BTreeMap;

use lance_graph::ast::{BooleanExpression, PropertyValue, ValueExpression};
use lance_graph::logical_plan::{LogicalOperator, LogicalPlanner};
use lance_graph::parser::parse_cypher_query;
use lance_graph::GraphConfig;

/// Why a construct cannot lower. The string is the plan row that rules it, so a
/// census line can be checked against the plan without re-deriving anything.
type GraceReason = &'static str;

/// Every committed source that carries Cypher query literals. Adding a fourth
/// is a one-line change here; the extractor never needs to know.
const SOURCES: &[(&str, &str)] = &[
    ("parser.rs", include_str!("../src/parser.rs")),
    ("logical_plan.rs", include_str!("../src/logical_plan.rs")),
    ("semantic.rs", include_str!("../src/semantic.rs")),
];

/// Pull every string literal out of Rust source: raw (`r#"…"#`) first, then
/// ordinary (`"…"`, honouring `\"`). Raw strings are taken first because an
/// ordinary-string scan would otherwise cut a raw query at its first inner
/// quote — which is exactly how a naive grep reports `MATCH (n:Person {name: `
/// as a whole query.
///
/// **Comments and CHAR literals are skipped, and the char case is not
/// hypothetical.** `parser.rs:931-933` contains `char('"')` three times — a
/// double quote inside a `'…'` literal. A scanner that does not know about char
/// literals opens a string there, goes one quote out of phase for the rest of
/// the file, and then reports raw Rust source as a query. The first run of this
/// census did exactly that: three "queries" came back reading
/// `); // Verify the AST structure let ast = result.unwrap(); …`. They were
/// harmless (they fail to parse, so they never reached the census) but they
/// were also proof the extractor was wrong, and a quieter version of the same
/// bug could have swallowed a real query instead of manufacturing a fake one.
fn string_literals(src: &str) -> Vec<String> {
    let b = src.as_bytes();
    let mut out = Vec::new();
    let mut i = 0usize;
    while i < b.len() {
        // line comment
        if b[i] == b'/' && i + 1 < b.len() && b[i + 1] == b'/' {
            while i < b.len() && b[i] != b'\n' {
                i += 1;
            }
            continue;
        }
        // block comment
        if b[i] == b'/' && i + 1 < b.len() && b[i + 1] == b'*' {
            i += 2;
            while i + 1 < b.len() && !(b[i] == b'*' && b[i + 1] == b'/') {
                i += 1;
            }
            i = (i + 2).min(b.len());
            continue;
        }
        // char literal — but NOT a lifetime (`&'a str`). A `'` opens a char
        // literal only when it is escaped (`'\''`) or closes two bytes later
        // (`'x'`); `'a ` is a lifetime and must be walked past as ordinary text.
        if b[i] == b'\'' && i + 1 < b.len() {
            if b[i + 1] == b'\\' {
                let mut j = i + 2;
                while j < b.len() && b[j] != b'\'' {
                    j += 1;
                }
                i = j + 1;
                continue;
            }
            if i + 2 < b.len() && b[i + 2] == b'\'' {
                i += 3;
                continue;
            }
        }
        // raw string: r, then one or more '#', then '"'
        if b[i] == b'r' && i + 1 < b.len() {
            let mut h = i + 1;
            while h < b.len() && b[h] == b'#' {
                h += 1;
            }
            if h > i + 1 && h < b.len() && b[h] == b'"' {
                let hashes = h - (i + 1);
                let close = format!("\"{}", "#".repeat(hashes));
                if let Some(end) = src[h + 1..].find(&close) {
                    out.push(src[h + 1..h + 1 + end].to_string());
                    i = h + 1 + end + close.len();
                    continue;
                }
            }
        }
        if b[i] == b'"' {
            let mut j = i + 1;
            let mut lit = String::new();
            while j < b.len() {
                if b[j] == b'\\' && j + 1 < b.len() {
                    // keep the escaped character; a `\"` must not end the literal
                    lit.push(b[j + 1] as char);
                    j += 2;
                    continue;
                }
                if b[j] == b'"' {
                    break;
                }
                lit.push(b[j] as char);
                j += 1;
            }
            out.push(lit);
            i = j + 1;
            continue;
        }
        i += 1;
    }
    out
}

/// A literal is a census candidate if it reads like a query rather than a
/// message. The parser is the real filter — this only avoids feeding it every
/// string in 66 KB of source.
fn looks_like_a_query(s: &str) -> bool {
    let u = s.to_uppercase();
    (u.contains("MATCH ") || u.starts_with("UNWIND ")) && u.contains("RETURN")
}

/// §3.5 / §4.1 — a projection's disposition.
fn classify_value(v: &ValueExpression, grace: &mut Vec<GraceReason>) {
    match v {
        // T-3 `RETURN n` (the mask itself) · T-4 `RETURN n.prop` (mask + lane).
        ValueExpression::Variable(_) | ValueExpression::Property(_) => {}
        ValueExpression::Literal(PropertyValue::Integer(_))
        | ValueExpression::Literal(PropertyValue::Boolean(_))
        | ValueExpression::Parameter(_) => {}
        // P-9 — a float or string constant is not a mask predicate's operand.
        ValueExpression::Literal(_) => grace.push("§4.2 P-9 non-integer literal"),
        ValueExpression::AggregateFunction {
            name,
            args,
            distinct,
        } => {
            // T-12: DISTINCT over VALUES needs value identity a mask has not.
            if *distinct {
                grace.push("§4.1 T-12 count(DISTINCT …)");
            }
            match name.to_lowercase().as_str() {
                // T-1/T-2 count · T-5 sum · T-6 min/max · T-7 avg (`[H]`, OQ-7).
                "count" | "sum" | "min" | "max" | "avg" => {}
                // T-12 — collect is multiplicity, which a population has not.
                "collect" => grace.push("§4.1 T-12 collect()"),
                _ => grace.push("§4.2 G-4 unknown aggregate"),
            }
            for a in args {
                classify_value(a, grace);
            }
        }
        // G-4 — every scalar string function.
        ValueExpression::ScalarFunction { .. } => grace.push("§4.2 G-4 scalar function"),
        // No §3.5 terminal computes a value expression; arithmetic is a value,
        // not a population.
        ValueExpression::Arithmetic { .. } => grace.push("§4.2 arithmetic projection"),
        // G-7 — §4.4, and the plan calls this one *the interesting one*.
        ValueExpression::VectorDistance { .. } | ValueExpression::VectorSimilarity { .. } => {
            grace.push("§4.4 G-7 vector distance/similarity")
        }
        ValueExpression::VectorLiteral(_) => grace.push("§4.4 G-7 vector literal"),
    }
}

/// §3.2 / §3.3 / §4.2 — a `WHERE` predicate's disposition.
fn classify_bool(e: &BooleanExpression, grace: &mut Vec<GraceReason>) {
    match e {
        // P-1..P-5 — compare a lane to a constant.
        BooleanExpression::Comparison { left, right, .. } => {
            classify_value(left, grace);
            classify_value(right, grace);
        }
        // §3.3 — three leaves are one TERNLOG immediate.
        BooleanExpression::And(a, b) | BooleanExpression::Or(a, b) => {
            classify_bool(a, grace);
            classify_bool(b, grace);
        }
        BooleanExpression::Not(a) => classify_bool(a, grace),
        // T-9 `mask_any`; and absence is a zero-fallback, never a validity bit.
        BooleanExpression::Exists(_) => {}
        BooleanExpression::IsNull(_) | BooleanExpression::IsNotNull(_) => {}
        // P-7 — N compare→mask sweeps, `mask_or`-accumulated.
        BooleanExpression::In { expression, list } => {
            classify_value(expression, grace);
            for v in list {
                classify_value(v, grace);
            }
        }
        // G-3 — §4.2, variable-width values.
        BooleanExpression::Like { .. }
        | BooleanExpression::ILike { .. }
        | BooleanExpression::Contains { .. }
        | BooleanExpression::StartsWith { .. }
        | BooleanExpression::EndsWith { .. } => grace.push("§4.2 G-3 string predicate"),
    }
}

/// Walk the plan. `lowered` counts nodes that DO lower, so the Full / Split /
/// Grace verdict is a comparison of two counts rather than a guess.
fn classify_plan(op: &LogicalOperator, grace: &mut Vec<GraceReason>, lowered: &mut usize) {
    match op {
        // §3.1 — N-2's `label → classid` binding is stated ABSENT by §1.4; the
        // census counts the SHAPE as lowering and reports OQ-1 separately,
        // because conflating "no route yet" with "cannot lower" would make the
        // premise unfalsifiable.
        LogicalOperator::ScanByLabel { .. } => *lowered += 1,
        LogicalOperator::Filter { input, predicate } => {
            *lowered += 1;
            classify_bool(predicate, grace);
            classify_plan(input, grace, lowered);
        }
        // §3.4 R-1 — the hop (Wave 2).
        LogicalOperator::Expand { input, .. } => {
            *lowered += 1;
            classify_plan(input, grace, lowered);
        }
        // §3.4 R-7/R-8 — the fixpoint (Wave 3).
        LogicalOperator::VariableLengthExpand { input, .. } => {
            *lowered += 1;
            classify_plan(input, grace, lowered);
        }
        LogicalOperator::Project { input, projections } => {
            *lowered += 1;
            for p in projections {
                classify_value(&p.expression, grace);
            }
            classify_plan(input, grace, lowered);
        }
        // T-11 — `DISTINCT` over a node variable is the identity and lowers to
        // nothing at all; over a value (T-12) it is grace. The distinction is
        // made by what the projection under it holds, so it is decided there:
        // this node itself is free either way.
        LogicalOperator::Distinct { input } => classify_plan(input, grace, lowered),
        // §4.1 — order and position.
        LogicalOperator::Sort { input, .. } => {
            grace.push("§4.1 G-1 ORDER BY");
            classify_plan(input, grace, lowered);
        }
        LogicalOperator::Offset { input, .. } => {
            grace.push("§4.1 G-2 SKIP");
            classify_plan(input, grace, lowered);
        }
        LogicalOperator::Limit { input, .. } => {
            grace.push("§4.1 G-2 LIMIT");
            classify_plan(input, grace, lowered);
        }
        // §4.3 — a new relation, not a population.
        LogicalOperator::Unwind { input, .. } => {
            grace.push("§4.3 G-5 UNWIND");
            if let Some(i) = input {
                classify_plan(i, grace, lowered);
            }
        }
        // §4.5 — cross-address-space.
        LogicalOperator::Join { left, right, .. } => {
            grace.push("§4.5 Join");
            classify_plan(left, grace, lowered);
            classify_plan(right, grace, lowered);
        }
    }
}

fn main() {
    let config = GraphConfig::default();

    let mut seen: BTreeMap<String, &'static str> = BTreeMap::new();
    for (file, src) in SOURCES {
        for lit in string_literals(src) {
            if looks_like_a_query(&lit) {
                seen.entry(lit).or_insert(file);
            }
        }
    }

    let mut parse_fail: Vec<&String> = Vec::new();
    let mut plan_fail: Vec<(&String, bool)> = Vec::new();
    let mut full: Vec<&String> = Vec::new();
    let mut split: Vec<(&String, Vec<GraceReason>)> = Vec::new();
    let mut pure_grace: Vec<(&String, Vec<GraceReason>)> = Vec::new();
    let mut reason_hist: BTreeMap<GraceReason, usize> = BTreeMap::new();

    // Provenance per source, so a reader can see WHICH committed file supplied
    // the corpus rather than taking "62 literals" on trust.
    let mut per_source: BTreeMap<&'static str, usize> = BTreeMap::new();
    for file in seen.values() {
        *per_source.entry(file).or_insert(0) += 1;
    }

    for q in seen.keys() {
        let Ok(ast) = parse_cypher_query(q) else {
            parse_fail.push(q);
            continue;
        };
        let mut planner = LogicalPlanner::new(&config);
        let Ok(plan) = planner.plan(&ast) else {
            // A refusal is classified by the AST it refused, not by eye: does
            // every RETURN item name a bare node variable? That shape is §3.5
            // T-3 — `Terminal::Keep`, "the mask itself" — the single most
            // mask-native return there is.
            let bare_node = !ast.return_clause.items.is_empty()
                && ast
                    .return_clause
                    .items
                    .iter()
                    .all(|it| matches!(it.expression, ValueExpression::Variable(_)));
            plan_fail.push((q, bare_node));
            continue;
        };
        let mut grace = Vec::new();
        let mut lowered = 0usize;
        classify_plan(&plan, &mut grace, &mut lowered);
        for r in &grace {
            *reason_hist.entry(r).or_insert(0) += 1;
        }
        if grace.is_empty() {
            full.push(q);
        } else if lowered > 0 {
            split.push((q, grace));
        } else {
            pure_grace.push((q, grace));
        }
    }

    let classified = full.len() + split.len() + pure_grace.len();
    println!("W0-b — CORPUS CENSUS (cypher-mask-lowering-v1 §7.0)");
    println!("=================================================\n");
    println!("candidate literals extracted : {}", seen.len());
    for (file, n) in &per_source {
        println!("    from {file:<16}       : {n}");
    }
    println!("  did not parse (not queries): {}", parse_fail.len());
    println!("  parsed but did not plan    : {}", plan_fail.len());
    println!("  CLASSIFIED                 : {classified}\n");

    let pct = |n: usize| {
        if classified == 0 {
            0.0
        } else {
            100.0 * n as f64 / classified as f64
        }
    };
    println!(
        "  Full  (everything lowers)  : {:>3}  ({:5.1} %)",
        full.len(),
        pct(full.len())
    );
    println!(
        "  Split (mask prefix + DF)   : {:>3}  ({:5.1} %)",
        split.len(),
        pct(split.len())
    );
    println!(
        "  Grace (nothing lowers)     : {:>3}  ({:5.1} %)",
        pure_grace.len(),
        pct(pure_grace.len())
    );

    println!("\nGRACE REASONS (why a query is not Full) — count, plan row");
    println!("---------------------------------------------------------");
    let mut rows: Vec<_> = reason_hist.iter().collect();
    rows.sort_by(|a, b| b.1.cmp(a.1).then(a.0.cmp(b.0)));
    for (reason, n) in rows {
        println!("  {n:>3}  {reason}");
    }

    println!("\nFULL-LOWERING QUERIES (the premise, enumerated)");
    println!("----------------------------------------------");
    for q in &full {
        println!("  {}", q.replace('\n', " "));
    }

    // Both exclusion buckets are ENUMERATED, not merely counted. A census that
    // reports only "3 excluded" cannot be checked: the reader cannot tell a
    // planner's error message from a real query the planner refused, and those
    // two have opposite meanings for the premise.
    println!("\nEXCLUDED — did not parse (the parser's own NEGATIVE tests live here)");
    println!("----------------------------------------------------------------------");
    for q in &parse_fail {
        println!("  {}", q.replace('\n', " "));
    }

    // The refusals are the most informative bucket in this census, and the
    // reason is counted rather than asserted: the incumbent planner REFUSES
    // `RETURN <node variable>` — §3.5 T-3, `Terminal::Keep`, the query the mask
    // path answers with no work at all. Those queries are excluded from the
    // percentages above, so the Full fraction UNDERSTATES the premise by
    // exactly the shape that most favours it.
    let bare_node_refusals = plan_fail.iter().filter(|(_, b)| *b).count();
    println!("\nEXCLUDED — parsed but did not plan (a REFUSAL, read each one)");
    println!("-------------------------------------------------------------");
    for (q, bare) in &plan_fail {
        let tag = if *bare {
            "[T-3 RETURN <node>]"
        } else {
            "[other]"
        };
        println!("  {tag} {}", q.replace('\n', " "));
    }
    println!(
        "\n  {} of {} refusals return a BARE NODE VARIABLE (§3.5 T-3).",
        bare_node_refusals,
        plan_fail.len()
    );

    println!("\nVERDICT");
    println!("-------");
    if classified == 0 {
        println!("  NO CORPUS — the extractor found nothing. Fix the extractor,");
        println!("  do not read this as a finding about the plan.");
    } else if full.is_empty() {
        println!("  STOP. No query lowers fully. §7.0's STOP condition fires:");
        println!("  the premise is wrong as stated and Wave 1 does not start.");
    } else {
        println!(
            "  {} of {} queries ({:.1} %) lower FULLY; {} more ({:.1} %) lower",
            full.len(),
            classified,
            pct(full.len()),
            split.len(),
            pct(split.len())
        );
        println!("  as a mask PREFIX under a DataFusion residue (§2's `Split`).");
        println!();
        println!("  THREE THINGS THIS NUMBER IS NOT:");
        println!("  1. It is CONDITIONAL on OQ-1. Every Full query below scans");
        println!("     by LABEL, and §1.4 states the `label -> classid` binding");
        println!("     is ABSENT. Counting the SHAPE as lowering is what keeps");
        println!("     the premise falsifiable; it is not a claim the route exists.");
        println!("  2. It measures the TEST corpus, not a workload. These queries");
        println!("     were written to exercise the parser and the planner, so the");
        println!("     shape distribution is theirs, not a user's.");
        println!("  3. It says nothing about SPEED. §7.0 is a correctness gate;");
        println!("     no row of this output is a performance claim.");
        println!("  Whether that is 'negligible' is §7.0's word and the");
        println!("  operator's call; the number is the number.");
    }
}
