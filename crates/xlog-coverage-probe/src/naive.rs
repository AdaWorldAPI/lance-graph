//! An independent naive bottom-up Datalog evaluator over the parsed AST.
//!
//! It shares nothing with the lowering under test (no RIR, no plan nodes, no
//! cube) and nothing with XLOG's reference evaluator. It exists because that
//! evaluator refuses rules with fewer than two positive atoms, which left most
//! of the census without an oracle.
//!
//! Semantics, stated so a reader can check them: set semantics (Datalog); a
//! rule fires for every assignment of its variables that matches all positive
//! atoms, satisfies every comparison, and finds no row for any negated atom;
//! each component is iterated to its least fixpoint. Components are evaluated
//! in the order the caller gives (the compiler's stratum order). Negation of a
//! predicate of the component being evaluated is refused, not approximated.
//!
//! Supported terms: named variables, `_`, non-negative or negative integers.
//! Supported literals: positive atoms, negated atoms, comparisons between
//! variables and integers. Anything else is refused with its name, so a rule
//! the evaluator cannot judge stays unverified rather than looking judged.

use std::collections::{BTreeMap, BTreeSet};

use xlog_logic::ast::CompOp;
use xlog_logic::{BodyLiteral, Rule, Term};

/// A relation: a set of integer rows.
pub type Relation = BTreeSet<Vec<i128>>;

/// Rows above this in any relation abort the component. A policy pin.
const MAX_ROWS: usize = 200_000;
/// Fixpoint rounds above this abort the component. A policy pin.
const MAX_ROUNDS: usize = 10_000;
/// Partial bindings above this in one rule firing abort the component. A
/// policy pin that keeps a census over dense seeds bounded.
pub(crate) const MAX_BINDINGS: usize = 2_000_000;

/// Why a rule cannot be evaluated here.
fn term_refusal(t: &Term) -> Option<String> {
    match t {
        Term::Variable(_) | Term::Anonymous | Term::Integer(_) => None,
        other => Some(format!(
            "naive oracle: unsupported term {}",
            term_kind(other)
        )),
    }
}

fn term_kind(t: &Term) -> &'static str {
    match t {
        Term::Variable(_) => "variable",
        Term::Anonymous => "anonymous",
        Term::Integer(_) => "integer",
        Term::Float(_) => "float",
        Term::String(_) => "string",
        Term::Symbol(_) => "symbol",
        Term::List(_) => "list",
        Term::Cons { .. } => "cons",
        Term::Compound { .. } => "compound",
        Term::PredRef(_) => "predicate reference",
        Term::Aggregate(_) => "aggregate",
    }
}

/// Check one rule is inside the supported fragment and safe: every head
/// variable and every variable of a negated atom or comparison is bound by a
/// positive atom.
pub fn check_rule(rule: &Rule) -> Result<(), String> {
    for t in &rule.head.terms {
        if matches!(t, Term::Anonymous) {
            return Err("naive oracle: `_` in a head".to_string());
        }
        if let Some(why) = term_refusal(t) {
            return Err(why);
        }
    }
    let mut bound: BTreeSet<&str> = BTreeSet::new();
    for lit in &rule.body {
        match lit {
            BodyLiteral::Positive(a) => {
                for t in &a.terms {
                    if let Some(why) = term_refusal(t) {
                        return Err(why);
                    }
                    if let Term::Variable(v) = t {
                        bound.insert(v);
                    }
                }
            }
            BodyLiteral::Negated(a) => {
                for t in &a.terms {
                    if let Some(why) = term_refusal(t) {
                        return Err(why);
                    }
                }
            }
            BodyLiteral::Comparison(c) => {
                for t in [&c.left, &c.right] {
                    if matches!(t, Term::Anonymous) {
                        return Err("naive oracle: `_` in a comparison".to_string());
                    }
                    if let Some(why) = term_refusal(t) {
                        return Err(why);
                    }
                }
            }
            BodyLiteral::Epistemic(_) => {
                return Err("naive oracle: unsupported literal epistemic".to_string())
            }
            BodyLiteral::IsExpr(_) => {
                return Err("naive oracle: unsupported literal is-expression".to_string())
            }
            BodyLiteral::Univ(_) => {
                return Err("naive oracle: unsupported literal univ".to_string())
            }
        }
    }
    let unbound = |t: &Term| matches!(t, Term::Variable(v) if !bound.contains(v.as_str()));
    if rule.head.terms.iter().any(unbound) {
        return Err("naive oracle: unsafe rule (head variable not bound)".to_string());
    }
    for lit in &rule.body {
        let vars_ok = match lit {
            BodyLiteral::Negated(a) => !a.terms.iter().any(unbound),
            BodyLiteral::Comparison(c) => !unbound(&c.left) && !unbound(&c.right),
            _ => true,
        };
        if !vars_ok {
            return Err(
                "naive oracle: unsafe rule (variable bound only under negation or comparison)"
                    .to_string(),
            );
        }
    }
    Ok(())
}

type Binding = BTreeMap<String, i128>;

fn value(t: &Term, b: &Binding) -> Option<i128> {
    match t {
        Term::Integer(n) => Some(i128::from(*n)),
        Term::Variable(v) => b.get(v).copied(),
        _ => None,
    }
}

/// Extend `b` so `terms` matches `row`, or `None`.
fn unify(terms: &[Term], row: &[i128], b: &Binding) -> Option<Binding> {
    if terms.len() != row.len() {
        return None;
    }
    let mut out = b.clone();
    for (t, &v) in terms.iter().zip(row) {
        match t {
            Term::Anonymous => {}
            Term::Integer(n) => {
                if i128::from(*n) != v {
                    return None;
                }
            }
            Term::Variable(name) => match out.get(name) {
                Some(&old) if old != v => return None,
                Some(_) => {}
                None => {
                    out.insert(name.clone(), v);
                }
            },
            _ => return None,
        }
    }
    Some(out)
}

fn compare(op: CompOp, l: i128, r: i128) -> bool {
    match op {
        CompOp::Eq => l == r,
        CompOp::Ne => l != r,
        CompOp::Lt => l < r,
        CompOp::Le => l <= r,
        CompOp::Gt => l > r,
        CompOp::Ge => l >= r,
    }
}

/// Every head row the rule derives over `store` (one round, no iteration).
fn fire(rule: &Rule, store: &BTreeMap<String, Relation>) -> Result<Relation, String> {
    let empty = Relation::new();
    let positives: Vec<_> = rule
        .body
        .iter()
        .filter_map(|l| match l {
            BodyLiteral::Positive(a) => Some(a),
            _ => None,
        })
        .collect();
    let mut bindings: Vec<Binding> = vec![Binding::new()];
    for atom in positives {
        let rel = store.get(&atom.predicate).unwrap_or(&empty);
        let mut next = Vec::new();
        for b in &bindings {
            for row in rel {
                if let Some(nb) = unify(&atom.terms, row, b) {
                    // Checked while building, so the bound caps the allocation
                    // and not only the result.
                    if next.len() == MAX_BINDINGS {
                        return Err(format!(
                            "naive oracle: more than {MAX_BINDINGS} partial bindings"
                        ));
                    }
                    next.push(nb);
                }
            }
        }
        bindings = next;
        if bindings.is_empty() {
            break;
        }
    }
    let mut out = Relation::new();
    'binding: for b in bindings {
        for lit in &rule.body {
            match lit {
                BodyLiteral::Comparison(c) => {
                    let (Some(l), Some(r)) = (value(&c.left, &b), value(&c.right, &b)) else {
                        continue 'binding;
                    };
                    if !compare(c.op, l, r) {
                        continue 'binding;
                    }
                }
                BodyLiteral::Negated(a) => {
                    let rel = store.get(&a.predicate).unwrap_or(&empty);
                    if rel.iter().any(|row| unify(&a.terms, row, &b).is_some()) {
                        continue 'binding;
                    }
                }
                _ => {}
            }
        }
        let row: Option<Vec<i128>> = rule.head.terms.iter().map(|t| value(t, &b)).collect();
        if let Some(row) = row {
            out.insert(row);
        }
    }
    Ok(out)
}

/// Evaluate one component (the rules whose heads are in `heads`) to its least
/// fixpoint over `store`, then add the result to `store`.
pub fn evaluate_component(
    rules: &[&Rule],
    store: &mut BTreeMap<String, Relation>,
) -> Result<(), String> {
    let heads: BTreeSet<&str> = rules.iter().map(|r| r.head.predicate.as_str()).collect();
    for rule in rules {
        check_rule(rule)?;
        for lit in &rule.body {
            if let BodyLiteral::Negated(a) = lit {
                if heads.contains(a.predicate.as_str()) {
                    return Err(format!(
                        "naive oracle: negation of {} inside its own component",
                        a.predicate
                    ));
                }
            }
        }
    }
    for h in &heads {
        store.entry((*h).to_string()).or_default();
    }
    for _ in 0..MAX_ROUNDS {
        let mut changed = false;
        for rule in rules {
            let derived = fire(rule, store)?;
            let rel = store.entry(rule.head.predicate.clone()).or_default();
            for row in derived {
                changed |= rel.insert(row);
            }
            if rel.len() > MAX_ROWS {
                return Err(format!("naive oracle: more than {MAX_ROWS} rows"));
            }
        }
        if !changed {
            return Ok(());
        }
    }
    Err(format!(
        "naive oracle: no fixpoint after {MAX_ROUNDS} rounds"
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use xlog_logic::parse_program;

    fn rel(rows: &[&[i128]]) -> Relation {
        rows.iter().map(|r| r.to_vec()).collect()
    }

    fn run(src: &str, base: &[(&str, Relation)]) -> Result<BTreeMap<String, Relation>, String> {
        let p = parse_program(src).map_err(|e| e.to_string())?;
        let rules: Vec<&Rule> = p.proper_rules().collect();
        let mut store: BTreeMap<String, Relation> = base
            .iter()
            .map(|(n, r)| ((*n).to_string(), r.clone()))
            .collect();
        evaluate_component(&rules, &mut store)?;
        Ok(store)
    }

    #[test]
    fn a_one_atom_rule_copies_and_projects() {
        let s = run(
            "q(Y) :- r(X, Y).",
            &[("r", rel(&[&[1, 2], &[3, 2], &[4, 5]]))],
        )
        .unwrap();
        assert_eq!(s["q"], rel(&[&[2], &[5]]));
    }

    #[test]
    fn a_repeated_variable_selects_the_diagonal() {
        let s = run(
            "q(X) :- r(X, X).",
            &[("r", rel(&[&[1, 1], &[1, 2], &[3, 3]]))],
        )
        .unwrap();
        assert_eq!(s["q"], rel(&[&[1], &[3]]));
    }

    #[test]
    fn a_constant_selects_and_a_head_constant_constructs() {
        let s = run(
            "q(Y, 7) :- r(1, Y).",
            &[("r", rel(&[&[1, 2], &[2, 3], &[1, 4]]))],
        )
        .unwrap();
        assert_eq!(s["q"], rel(&[&[2, 7], &[4, 7]]));
    }

    #[test]
    fn transitive_closure_reaches_its_fixpoint() {
        let s = run(
            "tc(X, Y) :- e(X, Y). tc(X, Z) :- tc(X, Y), e(Y, Z).",
            &[("e", rel(&[&[1, 2], &[2, 3], &[3, 1]]))],
        )
        .unwrap();
        assert_eq!(s["tc"].len(), 9, "a 3-cycle closes to all 9 pairs");
    }

    #[test]
    fn a_comparison_filters_and_negation_subtracts() {
        let base = [
            ("r", rel(&[&[1, 2], &[3, 1], &[2, 5]])),
            ("s", rel(&[&[2, 5]])),
        ];
        let s = run("q(X, Y) :- r(X, Y), X < Y, not s(X, Y).", &base).unwrap();
        assert_eq!(s["q"], rel(&[&[1, 2]]));
    }

    #[test]
    fn unsafe_and_unsupported_rules_are_refused_not_evaluated() {
        let base = [("r", rel(&[&[1, 2]]))];
        assert!(run("q(X, Z) :- r(X, Y).", &base)
            .unwrap_err()
            .contains("unsafe"));
        assert!(run("q(X) :- r(X, Y), not s(Z).", &base)
            .unwrap_err()
            .contains("unsafe"));
        assert!(run("p(X) :- r(X, Y), not p(Y).", &base)
            .unwrap_err()
            .contains("own component"));
    }

    /// Five independent atoms over 36 rows would build 36^5 bindings; the
    /// refusal must come at the bound, not after the full intermediate.
    #[test]
    fn the_binding_bound_refuses_before_the_intermediate_is_built() {
        let dense: Relation = (0..6)
            .flat_map(|a| (0..6).map(move |b| vec![a, b]))
            .collect();
        let err = run(
            "q(A) :- r(A, B), r(C, D), r(E, F), r(G, H), r(I, J).",
            &[("r", dense)],
        )
        .unwrap_err();
        assert!(err.contains("partial bindings"), "{err}");
    }

    #[test]
    fn an_empty_base_gives_an_empty_answer() {
        let s = run("q(X) :- r(X, Y).", &[("r", Relation::new())]).unwrap();
        assert!(s["q"].is_empty());
    }
}
