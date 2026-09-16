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
use std::fs;
use std::path::{Path, PathBuf};

use lance_graph::ast::{BooleanExpression, PropertyValue, ValueExpression};
use lance_graph::logical_plan::{LogicalOperator, LogicalPlanner};
use lance_graph::parser::parse_cypher_query;
use lance_graph::GraphConfig;

/// Why a construct cannot lower. The string is the plan row that rules it, so a
/// census line can be checked against the plan without re-deriving anything.
type GraceReason = &'static str;

/// The corpus is EVERY `.rs` file under the workspace's `crates/`, walked at
/// run time — not a hand-listed set.
///
/// ⊘ The first version of this census listed three files by hand
/// (`parser.rs`, `logical_plan.rs`, `semantic.rs`) and reported **46.0 %** over
/// 50 classified queries. Codex flagged it on the PR and was right: a walk of
/// the same tree with the same extractor finds Cypher query literals in **33
/// files** carrying **342 distinct queries** — the whole of
/// `crates/lance-graph/tests/`, `src/query.rs`, the planner's strategy modules,
/// the Python bindings. The hand list was 15 % of the corpus, and a STOP gate
/// cleared on 15 % of a corpus is cleared on a subset nobody chose
/// deliberately.
///
/// The defect before the defect: the three files were picked after a `grep` for
/// `"…MATCH …"` reported zero hits in the DataFusion builder modules. That grep
/// cannot see a raw string or a literal spanning lines, which is the same blind
/// spot that made the extractor itself wrong (see [`string_literals`]). Hence a
/// WALK: a file added to the tree enters the corpus with no edit here, so this
/// list cannot silently go stale the way the last one did.
fn rust_sources(dir: &Path, out: &mut Vec<PathBuf>) {
    let Ok(entries) = fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            // `target/` is build output, not committed source.
            if path.file_name().is_some_and(|n| n == "target") {
                continue;
            }
            rust_sources(&path, out);
        } else if path.extension().is_some_and(|e| e == "rs") {
            out.push(path);
        }
    }
}

/// The workspace's `crates/` directory, derived from this crate's manifest dir
/// so the walk is anchored to the checkout rather than to a working directory.
fn crates_root() -> PathBuf {
    let mut p = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    p.pop(); // crates/lance-graph -> crates
    p
}

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

/// §3.2 / §4.2 — an INLINE PATTERN PROPERTY's disposition (`MATCH (p:Person
/// {name: "Alice"})`). These are `PropertyValue`, not `ValueExpression`, and
/// they are a different surface from `WHERE`: the planner stores them on the
/// operator (`ScanByLabel.properties`, `Expand.properties` /
/// `.target_properties`) rather than in a `Filter` node.
///
/// ⊘ The first version of this census ignored all four maps and counted every
/// scan as lowered, so `MATCH (p:Person {name: "Alice"})-[:KNOWS]->…` came back
/// **Full** when its string equality is P-9 grace. Codex flagged it on the PR.
/// An inline property is a predicate wherever the planner chose to file it.
fn classify_property_value(v: &PropertyValue, grace: &mut Vec<GraceReason>) {
    match v {
        // P-1..P-5 — an integer or boolean equality against a lane.
        PropertyValue::Integer(_) | PropertyValue::Boolean(_) => {}
        // A parameter is substituted before planning; its VALUE decides, and
        // this census cannot see it, so it is counted maskable and said so.
        PropertyValue::Parameter(_) => {}
        // `IS NULL` has no NULL to test against — absence is a zero-fallback.
        PropertyValue::Null => {}
        // A property-to-property equality is two lanes, still Boolean.
        PropertyValue::Property(_) => {}
        // P-9 — §4.2. A string or float constant is not a mask operand.
        PropertyValue::String(_) => grace.push("§4.2 P-9 inline string property"),
        PropertyValue::Float(_) => grace.push("§4.2 P-9 inline float property"),
    }
}

/// Every inline property map on one operator.
fn classify_property_map(
    props: &std::collections::HashMap<String, PropertyValue>,
    grace: &mut Vec<GraceReason>,
) {
    for v in props.values() {
        classify_property_value(v, grace);
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
        LogicalOperator::ScanByLabel { properties, .. } => {
            *lowered += 1;
            classify_property_map(properties, grace);
        }
        LogicalOperator::Filter { input, predicate } => {
            *lowered += 1;
            classify_bool(predicate, grace);
            classify_plan(input, grace, lowered);
        }
        // §3.4 R-1 — the hop (Wave 2). Both property maps count: an inline
        // property on the relationship and one on the target node are each a
        // predicate the hop has to carry.
        LogicalOperator::Expand {
            input,
            properties,
            target_properties,
            ..
        } => {
            *lowered += 1;
            classify_property_map(properties, grace);
            classify_property_map(target_properties, grace);
            classify_plan(input, grace, lowered);
        }
        // §3.4 R-7/R-8 — the fixpoint (Wave 3).
        LogicalOperator::VariableLengthExpand {
            input,
            target_properties,
            ..
        } => {
            *lowered += 1;
            classify_property_map(target_properties, grace);
            classify_plan(input, grace, lowered);
        }
        LogicalOperator::Project { input, projections } => {
            *lowered += 1;
            for p in projections {
                classify_value(&p.expression, grace);
            }
            classify_plan(input, grace, lowered);
        }
        // T-11 vs T-12 — and the distinction has to be made HERE.
        //
        // `DISTINCT n` (a node variable) is FREE: a mask IS a set, so there is
        // no multiplicity to collapse. `DISTINCT n.p` (a value) is GRACE:
        // distinct over VALUES needs value identity, which a population mask
        // does not carry.
        //
        // ⊘ The first version of this census recursed and left the decision to
        // `classify_value`, whose comment even said so — but that function
        // accepts a bare `Property` unconditionally (T-4, `RETURN n.prop`, is
        // legitimately `[G]`), so T-12 never fired at all and the reason
        // histogram carried no T-12 row. Codex flagged it on the PR. What
        // distinguishes the two is not the projection alone but the projection
        // UNDER A `Distinct`, so the `Distinct` node is the only place that can
        // tell them apart.
        LogicalOperator::Distinct { input } => {
            // A guarded `matches!` rather than a let chain: this crate is
            // edition 2021, where let chains are not available.
            if matches!(
                input.as_ref(),
                LogicalOperator::Project { projections, .. }
                    if !projections
                        .iter()
                        .all(|p| matches!(p.expression, ValueExpression::Variable(_)))
            ) {
                grace.push("§4.1 T-12 DISTINCT over a value");
            }
            classify_plan(input, grace, lowered);
        }
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

    let root = crates_root();
    let mut files = Vec::new();
    rust_sources(&root, &mut files);
    files.sort();

    let mut seen: BTreeMap<String, String> = BTreeMap::new();
    for path in &files {
        let Ok(src) = fs::read_to_string(path) else {
            continue;
        };
        let label = path
            .strip_prefix(&root)
            .unwrap_or(path)
            .to_string_lossy()
            .into_owned();
        for lit in string_literals(&src) {
            if looks_like_a_query(&lit) {
                seen.entry(lit).or_insert_with(|| label.clone());
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
    let mut per_source: BTreeMap<&str, usize> = BTreeMap::new();
    for file in seen.values() {
        *per_source.entry(file.as_str()).or_insert(0) += 1;
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
    println!("    rust files walked          : {}", files.len());
    println!("    files carrying a query     : {}", per_source.len());
    let mut by_count: Vec<_> = per_source.iter().collect();
    by_count.sort_by(|a, b| b.1.cmp(a.1).then(a.0.cmp(b.0)));
    for (file, n) in by_count {
        println!("      {n:>4}  {file}");
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
