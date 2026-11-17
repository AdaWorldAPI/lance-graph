//! W7 census (D-XLOG-W7-P0 step 1): classify the refused rules whose head has
//! 3 or more columns, by body shape and by consumer.
//!
//! Usage: w7_census <census.tsv> <root>
//!
//! `<root>` holds the source files the census names, at the census's own
//! revision (`xlog@007bb19:<path>` is read as `<root>/<path>`). Reads only; it
//! runs no cube and no evaluator.
//!
//! Shape of a head, over all its non-fact rules:
//! - `one-atom`: every rule has exactly one positive body atom, and every head
//!   variable occurs in it. The head is a filter or projection of one
//!   relation's rows (case A: lanes plus one shared position table).
//! - `join`: some rule has two or more positive body atoms (cases B and C).
//! - `other`: neither, for example an epistemic or aggregate body.
//!
//! Consumers of a head: the rules that read it (`fold` when the reading rule's
//! head has an aggregate, `rule` otherwise), and the queries on it (`query`).
//! A head nothing reads is `none`: it is the program's answer.
use std::collections::{BTreeMap, BTreeSet};

use xlog_coverage_probe::literals;
use xlog_logic::{parse_program, Atom, BodyLiteral, Program, Rule, Term};

/// The query rule XLOG's compiler generates for query `index`: head over the
/// query atom's distinct variables in order of first appearance (or `1`),
/// body the query atom. A copy of the probe's private `query_rule`.
fn query_rule(index: usize, atom: &Atom) -> Rule {
    let mut seen: BTreeSet<&str> = BTreeSet::new();
    let mut terms: Vec<Term> = Vec::new();
    for t in &atom.terms {
        for name in t.variables() {
            if seen.insert(name) {
                terms.push(Term::Variable(name.to_string()));
            }
        }
    }
    if terms.is_empty() {
        terms.push(Term::Integer(1));
    }
    Rule {
        head: Atom {
            predicate: format!("__xlog_query_{index}"),
            terms,
        },
        body: vec![BodyLiteral::Positive(atom.clone())],
    }
}

fn read_atoms(r: &Rule) -> Vec<&str> {
    r.body
        .iter()
        .filter_map(BodyLiteral::atom)
        .map(|a| a.predicate.as_str())
        .collect()
}

fn shape(rules: &[&Rule]) -> &'static str {
    let defining: Vec<&&Rule> = rules.iter().filter(|r| !r.is_fact()).collect();
    if defining.is_empty() {
        return "facts";
    }
    let mut one_atom = true;
    for r in &defining {
        let pos: Vec<_> = r
            .body
            .iter()
            .filter_map(|l| match l {
                BodyLiteral::Positive(a) => Some(a),
                _ => None,
            })
            .collect();
        if pos.len() >= 2 {
            return "join";
        }
        let head_vars: BTreeSet<&str> = r.head.variables().into_iter().collect();
        let covered = pos.first().is_some_and(|a| {
            let vs: BTreeSet<&str> = a.variables().into_iter().collect();
            head_vars.is_subset(&vs)
        });
        if !covered {
            one_atom = false;
        }
    }
    if one_atom {
        "one-atom"
    } else {
        "other"
    }
}

fn consumers(p: &Program, head: &str) -> String {
    let mut kinds: BTreeSet<&str> = BTreeSet::new();
    for r in &p.rules {
        if r.head.predicate != head && read_atoms(r).contains(&head) {
            kinds.insert(if r.has_aggregation() { "fold" } else { "rule" });
        }
    }
    if p.queries.iter().any(|q| q.atom.predicate == head) {
        kinds.insert("query");
    }
    if kinds.is_empty() {
        "none".into()
    } else {
        kinds.into_iter().collect::<Vec<_>>().join("+")
    }
}

fn main() {
    let mut args = std::env::args().skip(1);
    let (Some(tsv), Some(root)) = (args.next(), args.next()) else {
        eprintln!("usage: w7_census <census.tsv> <root>");
        std::process::exit(2);
    };
    let text = std::fs::read_to_string(&tsv).expect("census readable");
    // (file, literal, head) of every refused rule, once.
    let mut refused: BTreeSet<(String, usize, String)> = BTreeSet::new();
    for line in text.lines().skip(1) {
        let c: Vec<&str> = line.split('\t').collect();
        if c.len() < 4 || !matches!(c[3], "refused" | "refused-on-data") {
            continue;
        }
        let literal: usize = c[1].parse().expect("literal index");
        refused.insert((c[0].to_string(), literal, c[2].to_string()));
    }
    let mut by_shape: BTreeMap<(&str, String), usize> = BTreeMap::new();
    let (mut total, mut wide, mut unread, mut wide_rules) = (0, 0, 0, 0);
    println!("file\tliteral\thead\tarity\tshape\tconsumers");
    for (file, literal, head) in &refused {
        total += 1;
        let path = format!(
            "{root}/{}",
            file.split_once(':').map_or(file.as_str(), |(_, p)| p)
        );
        let src = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{path}: {e}"));
        let Some(lit) = literals(&src).get(*literal).copied() else {
            unread += 1;
            continue;
        };
        let Ok(p) = parse_program(lit) else {
            unread += 1;
            continue;
        };
        let query: Option<Rule> = head
            .strip_prefix("__xlog_query_")
            .and_then(|i| i.parse::<usize>().ok())
            .and_then(|i| p.queries.get(i).map(|q| query_rule(i, &q.atom)));
        let rules: Vec<&Rule> = match &query {
            Some(q) => vec![q],
            None => p
                .rules
                .iter()
                .filter(|r| r.head.predicate == *head)
                .collect(),
        };
        let Some(arity) = rules.first().map(|r| r.head.arity()) else {
            unread += 1;
            eprintln!("unread: {file} #{literal} {head}");
            continue;
        };
        if arity < 3 {
            continue;
        }
        wide += 1;
        wide_rules += rules.iter().filter(|r| !r.is_fact()).count();
        let s = shape(&rules);
        let c = consumers(&p, head);
        *by_shape.entry((s, c.clone())).or_default() += 1;
        println!("{file}\t{literal}\t{head}\t{arity}\t{s}\t{c}");
    }
    eprintln!(
        "refused heads: {total}; heads of 3+ columns: {wide} ({wide_rules} rules); unread: {unread}"
    );
    for ((s, c), n) in by_shape {
        eprintln!("  {n:4}  {s:9}  {c}");
    }
}
