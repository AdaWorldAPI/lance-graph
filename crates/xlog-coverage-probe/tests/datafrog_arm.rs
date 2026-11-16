//! The Datafrog arm against the naive evaluator, component by component.

use std::collections::BTreeMap;

use xlog_coverage_probe::datafrog_arm::{self, NOT_APPLICABLE};
use xlog_coverage_probe::naive::{self, Relation};
use xlog_logic::{parse_program, Program, Rule};

type Store = BTreeMap<String, Relation>;
type Eval = fn(&[&Rule], &mut Store) -> Result<(), String>;

fn program(src: &str) -> Program {
    parse_program(src).expect("parse")
}

fn pairs(p: &[(i128, i128)]) -> Relation {
    p.iter().map(|&(a, b)| vec![a, b]).collect()
}

fn unary(v: &[i128]) -> Relation {
    v.iter().map(|&a| vec![a]).collect()
}

fn store(entries: &[(&str, Relation)]) -> Store {
    entries
        .iter()
        .map(|(n, r)| ((*n).to_string(), r.clone()))
        .collect()
}

fn by_heads<'a>(p: &'a Program, heads: &[&str]) -> Vec<&'a Rule> {
    p.proper_rules()
        .filter(|r| heads.contains(&r.head.predicate.as_str()))
        .collect()
}

fn run(eval: Eval, p: &Program, heads: &[&str], mut base: Store) -> Result<Store, String> {
    let rules = by_heads(p, heads);
    eval(&rules, &mut base)?;
    Ok(base)
}

fn datafrog_run(p: &Program, heads: &[&str], base: &Store) -> Result<Store, String> {
    run(datafrog_arm::evaluate_component, p, heads, base.clone())
}

fn naive_run(p: &Program, heads: &[&str], base: &Store) -> Result<Store, String> {
    run(naive::evaluate_component, p, heads, base.clone())
}

const TC: &str = "path(X,Y) :- edge(X,Y). path(X,Z) :- path(X,Y), edge(Y,Z).";

#[test]
fn transitive_closure_agrees_with_naive() {
    let cycle: Vec<(i128, i128)> = (0..6).map(|i| (i, (i + 1) % 6)).collect();
    let base = store(&[("edge", pairs(&cycle))]);
    let p = program(TC);
    let got = datafrog_run(&p, &["path"], &base).expect("datafrog");
    let want = naive_run(&p, &["path"], &base).expect("naive");
    assert_eq!(got["path"], want["path"]);
    assert_eq!(got["path"].len(), 36, "a 6-cycle closes to all 36 pairs");
}

#[test]
fn stratified_negation_agrees_with_naive() {
    let src = "reach(X,Y) :- edge(X,Y). reach(X,Z) :- reach(X,Y), edge(Y,Z). \
               unreached(X,Y) :- node(X), node(Y), not reach(X,Y).";
    let p = program(src);
    let base = store(&[
        ("edge", pairs(&[(0, 1), (1, 2), (2, 3)])),
        ("node", unary(&[0, 1, 2, 3, 4])),
    ]);
    let run_both = |eval: Eval| -> Store {
        let s = run(eval, &p, &["reach"], base.clone()).expect("reach");
        run(eval, &p, &["unreached"], s).expect("unreached")
    };
    let got = run_both(datafrog_arm::evaluate_component);
    let want = run_both(naive::evaluate_component);
    assert_eq!(got["reach"], want["reach"]);
    assert_eq!(got["unreached"], want["unreached"]);
    assert_eq!(got["reach"].len(), 6);
    let unreached = got["unreached"].len();
    assert!(
        unreached > 0 && unreached < 25,
        "a strict, non-empty subset of node x node: {unreached}"
    );
}

#[test]
fn out_of_subset_is_not_applicable() {
    let r = pairs(&[(1, 2), (2, 2), (3, 3), (4, 5)]);
    let base = store(&[("r", r)]);
    let cases: [(&str, &str, bool); 5] = [
        ("q(X,Y) :- r(X,Y), X < Y.", "comparison", true),
        ("p(X) :- r(X,Y), not p(Y).", "own component", false),
        ("q(X) :- r(X, X).", "repeated", true),
        (
            "q(X) :- r(X,Y), not r(Y, 3).",
            "integer constant in negated atom r",
            true,
        ),
        ("q(X) :- r(X, 5000000000).", "does not fit u32", true),
    ];
    for (src, construct, naive_answers) in cases {
        let p = program(src);
        let head = p
            .proper_rules()
            .next()
            .expect("rule")
            .head
            .predicate
            .clone();
        let err = datafrog_run(&p, &[head.as_str()], &base).expect_err(src);
        assert!(err.starts_with(NOT_APPLICABLE), "{src}: {err}");
        assert!(err.contains(construct), "{src}: {err}");
        if naive_answers {
            assert!(naive_run(&p, &[head.as_str()], &base).is_ok(), "{src}");
        }
    }
    // A constant in a positive atom is now inside the subset.
    let p = program("q(X) :- r(X, 3).");
    let got = datafrog_run(&p, &["q"], &base).expect("constant is supported");
    let want = naive_run(&p, &["q"], &base).expect("naive");
    assert_eq!(got["q"], want["q"]);
    assert!(!got["q"].is_empty());
    // Silence twin: a rule inside the subset is answered and agrees.
    let p = program("q(X,Z) :- r(X,Y), r(Y,Z).");
    let got = datafrog_run(&p, &["q"], &base).expect("in subset");
    let want = naive_run(&p, &["q"], &base).expect("naive");
    assert_eq!(got["q"], want["q"]);
    assert!(!got["q"].is_empty());
}

/// Node 3 has two successors, the other nodes have some too.
fn fan() -> Store {
    store(&[(
        "edge",
        pairs(&[(3, 4), (3, 5), (1, 3), (2, 4), (4, 5), (5, 6)]),
    )])
}

#[test]
fn body_constant_filter_agrees_with_naive() {
    let base = fan();
    let p = program("q(Y) :- edge(3, Y).");
    let got = datafrog_run(&p, &["q"], &base).expect("datafrog");
    let want = naive_run(&p, &["q"], &base).expect("naive");
    assert_eq!(got["q"], want["q"]);
    assert!(!got["q"].is_empty());
    let unfiltered = program("q(Y) :- edge(X, Y).");
    let all = naive_run(&unfiltered, &["q"], &base).expect("naive");
    assert!(
        got["q"].len() < all["q"].len(),
        "the filter must remove rows: {} vs {}",
        got["q"].len(),
        all["q"].len()
    );
}

#[test]
fn constant_in_later_atom_agrees() {
    let base = fan();
    let p = program("q(X) :- edge(X, Y), edge(Y, 4).");
    let got = datafrog_run(&p, &["q"], &base).expect("datafrog");
    let want = naive_run(&p, &["q"], &base).expect("naive");
    assert_eq!(got["q"], want["q"]);
    assert!(!got["q"].is_empty());
}

#[test]
fn head_constant_agrees() {
    let base = fan();
    let p = program("q(X, 1) :- edge(X, _).");
    let got = datafrog_run(&p, &["q"], &base).expect("datafrog");
    let want = naive_run(&p, &["q"], &base).expect("naive");
    assert_eq!(got["q"], want["q"]);
    assert!(!got["q"].is_empty());
    assert!(got["q"].iter().all(|row| row[1] == 1));
}

fn chain() -> Store {
    store(&[("edge", pairs(&[(0, 1), (1, 2), (2, 3)]))])
}

#[test]
fn corrupt_rule_body_is_detected() {
    let good = program(TC);
    let bad = program("path(X,Y) :- edge(X,Y). path(X,Z) :- path(Y,X), edge(Y,Z).");
    let base = chain();
    let want = naive_run(&good, &["path"], &base).expect("naive");
    let clean = datafrog_run(&good, &["path"], &base).expect("datafrog");
    assert_eq!(clean["path"], want["path"]);
    assert!(!want["path"].is_empty());
    let corrupt = datafrog_run(&bad, &["path"], &base).expect("datafrog, corrupt rule");
    assert_ne!(corrupt["path"], want["path"]);
}

#[test]
fn swapped_base_columns_are_detected() {
    let p = program(TC);
    let base = chain();
    let want = naive_run(&p, &["path"], &base).expect("naive");
    let clean = datafrog_run(&p, &["path"], &base).expect("datafrog");
    assert_eq!(clean["path"], want["path"]);
    assert!(!want["path"].is_empty());
    let swapped: Relation = base["edge"].iter().map(|r| vec![r[1], r[0]]).collect();
    let got = datafrog_run(&p, &["path"], &store(&[("edge", swapped)])).expect("datafrog");
    assert_ne!(got["path"], want["path"]);
}

#[test]
fn dropped_seed_fact_is_detected() {
    let p = program(TC);
    let base = chain();
    let want = naive_run(&p, &["path"], &base).expect("naive");
    let clean = datafrog_run(&p, &["path"], &base).expect("datafrog");
    assert_eq!(clean["path"], want["path"]);
    assert!(!want["path"].is_empty());
    let mut edges = base["edge"].clone();
    assert!(edges.remove(&vec![1, 2]));
    let got = datafrog_run(&p, &["path"], &store(&[("edge", edges)])).expect("datafrog");
    assert_ne!(got["path"], want["path"]);
}

#[test]
fn cross_product_and_three_atom_join() {
    let p = program("pair(X,Y) :- a(X), b(Y).");
    let base = store(&[("a", unary(&[1, 2])), ("b", unary(&[3, 4, 5]))]);
    let got = datafrog_run(&p, &["pair"], &base).expect("datafrog");
    let want = naive_run(&p, &["pair"], &base).expect("naive");
    assert_eq!(got["pair"], want["pair"]);
    assert_eq!(got["pair"].len(), 6, "no shared variable: a cross product");

    let p = program("tri(X,Y,Z) :- e(X,Y), e(Y,Z), e(Z,X).");
    let base = store(&[("e", pairs(&[(0, 1), (1, 2), (2, 0), (2, 3)]))]);
    let got = datafrog_run(&p, &["tri"], &base).expect("datafrog");
    let want = naive_run(&p, &["tri"], &base).expect("naive");
    assert_eq!(got["tri"], want["tri"]);
    assert_eq!(got["tri"].len(), 3, "the triangle read from each corner");
}

/// The census wiring, end to end through `reconcile`: an in-subset rule is
/// labelled `agree`, a rule with an integer constant in a positive atom also
/// `agree`, a rule with a comparison `not-applicable` with its reason, and no
/// label moves the rung.
#[test]
fn reconcile_labels_datafrog_without_moving_the_rung() {
    use xlog_coverage_probe::{reconcile, Rung, DEFAULT_SEEDS};
    let chain = reconcile("q(X,Z) :- r(X,Y), s(Y,Z).", &DEFAULT_SEEDS);
    let q = chain.rules.iter().find(|r| r.head == "q").expect("q");
    assert_eq!(q.rung, Rung::OracleAgrees, "{q:?}");
    assert_eq!(q.datafrog, "agree", "{q:?}");
    assert!(q.datafrog_reason.is_empty(), "{q:?}");

    let constant = reconcile("q(X) :- r(X,3).", &DEFAULT_SEEDS);
    let q = constant.rules.iter().find(|r| r.head == "q").expect("q");
    assert_eq!(q.rung, Rung::OracleAgrees, "{q:?}");
    assert_eq!(q.datafrog, "agree", "{q:?}");
    assert!(q.datafrog_reason.is_empty(), "{q:?}");

    // A comparison is refused by the cube itself, so Datafrog never runs on
    // it; a repeated variable is a shape the cube runs and Datafrog does not.
    let diagonal = reconcile("q(X) :- r(X,X).", &DEFAULT_SEEDS);
    let q = diagonal.rules.iter().find(|r| r.head == "q").expect("q");
    assert_eq!(q.rung, Rung::OracleAgrees, "{q:?}");
    assert_eq!(q.datafrog, "not-applicable", "{q:?}");
    assert!(q.datafrog_reason.contains("repeated"), "{q:?}");
}

/// A cross product past the row budget is refused by the join itself, which
/// stops producing rows at the budget instead of materialising the product
/// (1,500 x 1,500 = 2,250,000 > 2,000,000).
#[test]
fn an_oversized_cross_product_stops_inside_the_join() {
    let program = xlog_logic::parse_program("q(X,Y) :- a(X), b(Y).").expect("parse");
    let rules: Vec<&xlog_logic::Rule> = program.proper_rules().collect();
    let mut store: std::collections::BTreeMap<String, xlog_coverage_probe::naive::Relation> =
        std::collections::BTreeMap::new();
    let col: xlog_coverage_probe::naive::Relation = (0..1500i128).map(|v| vec![v]).collect();
    store.insert("a".to_string(), col.clone());
    store.insert("b".to_string(), col);
    let err = xlog_coverage_probe::datafrog_arm::evaluate_component(&rules, &mut store)
        .expect_err("over budget");
    assert!(err.contains("a join stopped at the budget"), "{err}");
    assert!(
        !store.contains_key("q"),
        "no partial head relation is stored"
    );
}
