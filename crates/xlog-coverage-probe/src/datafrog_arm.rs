//! A third, independent evaluator: the Datafrog engine (AdaWorldAPI fork).
//!
//! It answers the same question as [`crate::naive`] over the same parsed AST
//! and the same seed rows, but through a different mechanism: a semi-naive
//! fixpoint over Datafrog `Variable`s, with every body join lowered to keyed
//! `from_join` / `from_antijoin` statements. It shares no code with the naive
//! evaluator, with XLOG's reference evaluator, or with the cube. It never
//! decides a rung; the census only records whether it agrees.
//!
//! Supported subset (anything else is `datafrog: not applicable: <why>`):
//! rules whose body is positive and negated atoms over named variables and
//! `_` (anonymous in positive atoms only), with at least one positive atom;
//! no constants, comparisons, aggregates or other literals; no variable
//! repeated inside one atom; head terms that are variables bound by a positive
//! atom; negation only of predicates outside the component being evaluated,
//! over variables a positive atom binds. Every value read must fit `u32`.
//!
//! Same contract as [`crate::naive::evaluate_component`]: the rules of one
//! component go in, other predicates are read from `store`, and on `Ok` each
//! head's full relation replaces its entry in `store`.

use std::collections::{BTreeMap, BTreeSet};

use datafrog::{Iteration, Relation, Variable};
use xlog_logic::{Atom, BodyLiteral, Rule, Term};

use crate::naive;

/// Prefix of every "outside the supported subset" error.
pub const NOT_APPLICABLE: &str = "datafrog: not applicable: ";

/// A relation or intermediate result above this many rows aborts the
/// component. A policy pin.
const MAX_ROWS: usize = 2_000_000;

type Row = Vec<u32>;
/// A join input: the join key and the whole row.
type Keyed = (Row, Row);

fn not_applicable<T>(why: impl AsRef<str>) -> Result<T, String> {
    Err(format!("{NOT_APPLICABLE}{}", why.as_ref()))
}

fn term_kind(t: &Term) -> &'static str {
    match t {
        Term::Variable(_) => "variable",
        Term::Anonymous => "anonymous variable",
        Term::Integer(_) => "integer constant",
        Term::Float(_) => "float constant",
        Term::String(_) => "string constant",
        Term::Symbol(_) => "symbol constant",
        Term::List(_) => "list",
        Term::Cons { .. } => "cons",
        Term::Compound { .. } => "compound term",
        Term::PredRef(_) => "predicate reference",
        Term::Aggregate(_) => "aggregate",
    }
}

/// An atom as a predicate and one column name per term. Anonymous terms of a
/// positive atom get a fresh name no source variable can have.
struct AtomCols {
    pred: String,
    cols: Vec<String>,
}

fn atom_cols(atom: &Atom, positive: bool, fresh: &mut usize) -> Result<AtomCols, String> {
    let mut cols: Vec<String> = Vec::with_capacity(atom.terms.len());
    let mut seen: BTreeSet<&str> = BTreeSet::new();
    for t in &atom.terms {
        match t {
            Term::Variable(v) => {
                if !seen.insert(v.as_str()) {
                    return not_applicable(format!(
                        "variable {v} repeated inside atom {}",
                        atom.predicate
                    ));
                }
                cols.push(v.clone());
            }
            Term::Anonymous if positive => {
                *fresh += 1;
                cols.push(format!("_#{}", *fresh));
            }
            Term::Anonymous => {
                return not_applicable(format!(
                    "anonymous variable in negated atom {}",
                    atom.predicate
                ));
            }
            other => {
                return not_applicable(format!("{} in atom {}", term_kind(other), atom.predicate));
            }
        }
    }
    Ok(AtomCols {
        pred: atom.predicate.clone(),
        cols,
    })
}

struct RulePlan {
    positives: Vec<AtomCols>,
    negatives: Vec<AtomCols>,
    /// The head predicate and the column name of each head term.
    head: AtomCols,
}

fn plan_rule(rule: &Rule, heads: &BTreeSet<&str>, fresh: &mut usize) -> Result<RulePlan, String> {
    let mut positives: Vec<AtomCols> = Vec::new();
    let mut negatives: Vec<AtomCols> = Vec::new();
    for lit in &rule.body {
        match lit {
            BodyLiteral::Positive(a) => positives.push(atom_cols(a, true, fresh)?),
            BodyLiteral::Negated(a) => negatives.push(atom_cols(a, false, fresh)?),
            BodyLiteral::Comparison(_) => return not_applicable("comparison literal"),
            BodyLiteral::Epistemic(_) => return not_applicable("epistemic literal"),
            BodyLiteral::IsExpr(_) => return not_applicable("is-expression"),
            BodyLiteral::Univ(_) => return not_applicable("univ literal"),
        }
    }
    if positives.is_empty() {
        return not_applicable(format!(
            "rule for {} has no positive atom",
            rule.head.predicate
        ));
    }
    let bound: BTreeSet<&str> = positives
        .iter()
        .flat_map(|a| a.cols.iter().map(String::as_str))
        .collect();
    let mut head_cols: Vec<String> = Vec::with_capacity(rule.head.terms.len());
    for t in &rule.head.terms {
        match t {
            Term::Variable(v) if bound.contains(v.as_str()) => head_cols.push(v.clone()),
            Term::Variable(v) => {
                return not_applicable(format!(
                    "head variable {v} of {} is not bound by a positive atom",
                    rule.head.predicate
                ));
            }
            other => {
                return not_applicable(format!(
                    "{} in the head of {}",
                    term_kind(other),
                    rule.head.predicate
                ));
            }
        }
    }
    for neg in &negatives {
        if heads.contains(neg.pred.as_str()) {
            return not_applicable(format!("negation of {} inside its own component", neg.pred));
        }
        if let Some(v) = neg.cols.iter().find(|c| !bound.contains(c.as_str())) {
            return not_applicable(format!(
                "variable {v} of negated atom {} is not bound by a positive atom",
                neg.pred
            ));
        }
    }
    Ok(RulePlan {
        positives,
        negatives,
        head: AtomCols {
            pred: rule.head.predicate.clone(),
            cols: head_cols,
        },
    })
}

fn note_arity<'a>(
    arities: &mut BTreeMap<&'a str, usize>,
    pred: &'a str,
    arity: usize,
) -> Result<(), String> {
    match arities.insert(pred, arity) {
        Some(old) if old != arity => not_applicable(format!(
            "predicate {pred} is used with arities {old} and {arity}"
        )),
        _ => Ok(()),
    }
}

/// The rows `store` holds for `pred`, as `u32` rows of exactly `arity`.
fn load(
    pred: &str,
    arity: usize,
    store: &BTreeMap<String, naive::Relation>,
) -> Result<Vec<Row>, String> {
    let Some(rel) = store.get(pred) else {
        return Ok(Vec::new());
    };
    rel.iter()
        .map(|row| -> Result<Row, String> {
            if row.len() != arity {
                return not_applicable(format!(
                    "relation {pred} has a row of arity {} but is used with arity {arity}",
                    row.len()
                ));
            }
            row.iter()
                .map(|&v| {
                    u32::try_from(v).or_else(|_| {
                        not_applicable(format!("value {v} of {pred} does not fit u32"))
                    })
                })
                .collect()
        })
        .collect()
}

fn project(row: &Row, idx: &[usize]) -> Row {
    idx.iter().map(|&i| row[i]).collect()
}

fn size<T>(v: &Variable<T>) -> usize {
    v.num_stable() + v.recent.borrow().len()
}

/// The statements of one iteration step, and the intermediate variables whose
/// size is watched.
#[derive(Default)]
struct Lowered {
    stmts: Vec<Box<dyn FnMut()>>,
    rows: Vec<Variable<Row>>,
    keyed: Vec<Variable<Keyed>>,
}

impl Lowered {
    fn over_bound(&self, heads: &[Variable<Row>]) -> bool {
        self.rows.iter().chain(heads).any(|v| size(v) > MAX_ROWS)
            || self.keyed.iter().any(|v| size(v) > MAX_ROWS)
    }
}

/// `keyed = input` re-keyed on the columns at `idx`.
fn keyed_stage(
    it: &mut Iteration,
    out: &mut Lowered,
    input: &Variable<Row>,
    idx: Vec<usize>,
) -> Variable<Keyed> {
    let keyed = it.variable::<Keyed>("keyed");
    let (src, dst) = (input.clone(), keyed.clone());
    out.stmts.push(Box::new(move || {
        dst.from_map(&src, |row| (project(row, &idx), row.clone()))
    }));
    out.keyed.push(keyed.clone());
    keyed
}

/// Lower one rule to statements. The accumulator starts as the first positive
/// atom's relation and is joined with each further positive atom on the
/// variables they share (an empty key is a cross product), then each negated
/// atom is anti-joined, then the head columns are projected out.
fn lower_rule(
    it: &mut Iteration,
    vars: &BTreeMap<String, Variable<Row>>,
    store: &BTreeMap<String, naive::Relation>,
    plan: &RulePlan,
    out: &mut Lowered,
) -> Result<(), String> {
    let var_of = |pred: &str| -> Result<Variable<Row>, String> {
        vars.get(pred)
            .cloned()
            .ok_or_else(|| format!("{NOT_APPLICABLE}no relation variable for {pred}"))
    };
    let Some((first, rest)) = plan.positives.split_first() else {
        return not_applicable("rule has no positive atom");
    };
    let mut acc = var_of(&first.pred)?;
    let mut acc_cols: Vec<String> = first.cols.clone();

    for atom in rest {
        let (acc_idx, atom_idx): (Vec<usize>, Vec<usize>) = acc_cols
            .iter()
            .enumerate()
            .filter_map(|(i, c)| atom.cols.iter().position(|a| a == c).map(|j| (i, j)))
            .unzip();
        let extra: Vec<usize> = (0..atom.cols.len())
            .filter(|&j| !acc_cols.contains(&atom.cols[j]))
            .collect();
        let new_cols: Vec<String> = extra.iter().map(|&j| atom.cols[j].clone()).collect();

        let left = keyed_stage(it, out, &acc, acc_idx);
        let right = keyed_stage(it, out, &var_of(&atom.pred)?, atom_idx);
        let next = it.variable::<Row>("join");
        out.rows.push(next.clone());
        let (l, r, n) = (left.clone(), right.clone(), next.clone());
        out.stmts.push(Box::new(move || {
            n.from_join(&l, &r, |_key, lrow, rrow| {
                let mut row = lrow.clone();
                row.extend(extra.iter().map(|&j| rrow[j]));
                row
            })
        }));
        acc_cols.extend(new_cols);
        acc = next;
    }

    for atom in &plan.negatives {
        let idx: Option<Vec<usize>> = atom
            .cols
            .iter()
            .map(|c| acc_cols.iter().position(|a| a == c))
            .collect();
        let Some(idx) = idx else {
            return not_applicable(format!(
                "a variable of negated atom {} is not bound",
                atom.pred
            ));
        };
        let neg: Relation<Row> = Relation::from_vec(load(&atom.pred, atom.cols.len(), store)?);
        let keyed = keyed_stage(it, out, &acc, idx);
        let next = it.variable::<Row>("anti");
        out.rows.push(next.clone());
        let n = next.clone();
        out.stmts.push(Box::new(move || {
            n.from_antijoin(&keyed, &neg, |_key, row| row.clone())
        }));
        acc = next;
    }

    let head_idx: Option<Vec<usize>> = plan
        .head
        .cols
        .iter()
        .map(|c| acc_cols.iter().position(|a| a == c))
        .collect();
    let Some(head_idx) = head_idx else {
        return not_applicable("a head variable is not bound");
    };
    let head = var_of(&plan.head.pred)?;
    out.stmts.push(Box::new(move || {
        head.from_map(&acc, |row| project(row, &head_idx))
    }));
    Ok(())
}

/// Evaluate one component to its least fixpoint over `store` with Datafrog,
/// then put each head's full relation into `store`.
pub fn evaluate_component(
    rules: &[&Rule],
    store: &mut BTreeMap<String, naive::Relation>,
) -> Result<(), String> {
    let heads: BTreeSet<&str> = rules.iter().map(|r| r.head.predicate.as_str()).collect();
    let mut fresh = 0usize;
    let plans: Vec<RulePlan> = rules
        .iter()
        .map(|r| plan_rule(r, &heads, &mut fresh))
        .collect::<Result<_, _>>()?;

    let mut arities: BTreeMap<&str, usize> = BTreeMap::new();
    for plan in &plans {
        let atoms = plan
            .positives
            .iter()
            .chain(&plan.negatives)
            .chain(std::iter::once(&plan.head));
        for a in atoms {
            note_arity(&mut arities, &a.pred, a.cols.len())?;
        }
    }

    // One variable per predicate read positively or derived. Derived
    // predicates are seeded too: the store may hold stated facts for them.
    let var_preds: BTreeSet<&str> = plans
        .iter()
        .flat_map(|p| p.positives.iter().map(|a| a.pred.as_str()))
        .chain(heads.iter().copied())
        .collect();
    let mut it = Iteration::new();
    let mut vars: BTreeMap<String, Variable<Row>> = BTreeMap::new();
    for pred in var_preds {
        let var = it.variable::<Row>(pred);
        var.extend(load(pred, arities[pred], store)?);
        vars.insert(pred.to_string(), var);
    }

    let mut lowered = Lowered::default();
    for plan in &plans {
        lower_rule(&mut it, &vars, store, plan, &mut lowered)?;
    }
    let head_vars: Vec<Variable<Row>> =
        heads.iter().filter_map(|h| vars.get(*h).cloned()).collect();

    while it.changed() {
        for stmt in lowered.stmts.iter_mut() {
            stmt();
        }
        if lowered.over_bound(&head_vars) {
            return Err(format!("datafrog: more than {MAX_ROWS} rows"));
        }
    }

    let mut derived: Vec<(String, naive::Relation)> = Vec::with_capacity(heads.len());
    for h in &heads {
        let Some(var) = vars.remove(*h) else { continue };
        let rel = var.complete();
        if rel.len() > MAX_ROWS {
            return Err(format!("datafrog: more than {MAX_ROWS} rows"));
        }
        let rows: naive::Relation = rel
            .iter()
            .map(|row| row.iter().map(|&c| i128::from(c)).collect())
            .collect();
        derived.push(((*h).to_string(), rows));
    }
    store.extend(derived);
    Ok(())
}
