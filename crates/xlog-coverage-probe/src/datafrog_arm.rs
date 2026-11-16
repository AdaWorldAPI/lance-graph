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
//! `_`; integer constants in positive atoms (each becomes a hidden column plus
//! a semijoin against a one-row relation); no constants in negated atoms;
//! comparisons (`=`, `!=`, `<`, `<=`, `>`, `>=`) between variables a positive
//! atom binds and integer constants, applied as a filtered join against a
//! one-row relation; no aggregates or other literals; no variable repeated
//! inside one atom; head terms that are variables bound by a positive atom or
//! integer constants; negation only of predicates outside the component being
//! evaluated, over variables a positive atom binds (an anonymous variable in a
//! negated atom is "for some value": the anti-join reads the atom's relation
//! projected onto its named columns). A rule with no positive atom starts from
//! one empty row, so only a ground body can make it hold. Every value read,
//! and every integer constant, must fit `u32`.
//!
//! Same contract as [`crate::naive::evaluate_component`]: the rules of one
//! component go in, other predicates are read from `store`, and on `Ok` each
//! head's full relation replaces its entry in `store`.

use std::cell::Cell;
use std::collections::{BTreeMap, BTreeSet};
use std::rc::Rc;

use datafrog::{Iteration, Relation, Variable};
use xlog_logic::ast::CompOp;
use xlog_logic::ast::Comparison;
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

/// An atom as a predicate and one column name per term. Anonymous terms and
/// integer constants of a positive atom get a fresh name no source variable can
/// have; each constant also records a `(column, value)` filter.
struct AtomCols {
    pred: String,
    cols: Vec<String>,
    filters: Vec<(String, u32)>,
}

/// What one head term emits: the value of a body column, or a constant.
enum HeadSlot {
    Column(String),
    Const(u32),
}

struct HeadPlan {
    pred: String,
    slots: Vec<HeadSlot>,
}

/// A head slot resolved against the accumulator's columns.
enum Projected {
    Column(usize),
    Const(u32),
}

fn fit_u32(n: impl Copy + std::fmt::Display + TryInto<u32>) -> Result<u32, String> {
    n.try_into()
        .or_else(|_| not_applicable(format!("integer constant {n} does not fit u32")))
}

fn atom_cols(atom: &Atom, positive: bool, fresh: &mut usize) -> Result<AtomCols, String> {
    let mut cols: Vec<String> = Vec::with_capacity(atom.terms.len());
    let mut filters: Vec<(String, u32)> = Vec::new();
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
            Term::Anonymous => {
                *fresh += 1;
                cols.push(format!("_#{}", *fresh));
            }
            Term::Integer(n) if positive => {
                let value = fit_u32(*n)?;
                *fresh += 1;
                let col = format!("_c#{}", *fresh);
                cols.push(col.clone());
                filters.push((col, value));
            }
            Term::Integer(_) => {
                return not_applicable(format!(
                    "integer constant in negated atom {}",
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
        filters,
    })
}

/// One side of a comparison: a bound variable's column or a constant.
enum Operand {
    Column(String),
    Const(u32),
}

struct CompPlan {
    left: Operand,
    op: CompOp,
    right: Operand,
}

fn is_anonymous_col(c: &str) -> bool {
    c.starts_with("_#")
}

fn operand(t: &Term, bound: &BTreeSet<&str>) -> Result<Operand, String> {
    match t {
        Term::Variable(v) if bound.contains(v.as_str()) => Ok(Operand::Column(v.clone())),
        Term::Variable(v) => not_applicable(format!(
            "comparison variable {v} is not bound by a positive atom"
        )),
        Term::Integer(n) => Ok(Operand::Const(fit_u32(*n)?)),
        other => not_applicable(format!("{} in a comparison", term_kind(other))),
    }
}

fn holds(op: CompOp, l: u32, r: u32) -> bool {
    match op {
        CompOp::Eq => l == r,
        CompOp::Ne => l != r,
        CompOp::Lt => l < r,
        CompOp::Le => l <= r,
        CompOp::Gt => l > r,
        CompOp::Ge => l >= r,
    }
}

struct RulePlan {
    positives: Vec<AtomCols>,
    negatives: Vec<AtomCols>,
    comparisons: Vec<CompPlan>,
    /// The head predicate and what each head term emits.
    head: HeadPlan,
}

fn plan_rule(rule: &Rule, heads: &BTreeSet<&str>, fresh: &mut usize) -> Result<RulePlan, String> {
    let mut positives: Vec<AtomCols> = Vec::new();
    let mut negatives: Vec<AtomCols> = Vec::new();
    let mut comparisons: Vec<&Comparison> = Vec::new();
    for lit in &rule.body {
        match lit {
            BodyLiteral::Positive(a) => positives.push(atom_cols(a, true, fresh)?),
            BodyLiteral::Negated(a) => negatives.push(atom_cols(a, false, fresh)?),
            BodyLiteral::Comparison(c) => comparisons.push(c),
            BodyLiteral::Epistemic(_) => return not_applicable("epistemic literal"),
            BodyLiteral::IsExpr(_) => return not_applicable("is-expression"),
            BodyLiteral::Univ(_) => return not_applicable("univ literal"),
        }
    }
    let bound: BTreeSet<&str> = positives
        .iter()
        .flat_map(|a| a.cols.iter().map(String::as_str))
        .collect();
    let mut head_slots: Vec<HeadSlot> = Vec::with_capacity(rule.head.terms.len());
    for t in &rule.head.terms {
        match t {
            Term::Variable(v) if bound.contains(v.as_str()) => {
                head_slots.push(HeadSlot::Column(v.clone()))
            }
            Term::Integer(n) => head_slots.push(HeadSlot::Const(fit_u32(*n)?)),
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
        if let Some(v) = neg
            .cols
            .iter()
            .find(|c| !is_anonymous_col(c) && !bound.contains(c.as_str()))
        {
            return not_applicable(format!(
                "variable {v} of negated atom {} is not bound by a positive atom",
                neg.pred
            ));
        }
    }
    let comparisons = comparisons
        .into_iter()
        .map(|c| {
            Ok(CompPlan {
                left: operand(&c.left, &bound)?,
                op: c.op,
                right: operand(&c.right, &bound)?,
            })
        })
        .collect::<Result<Vec<_>, String>>()?;
    Ok(RulePlan {
        positives,
        negatives,
        comparisons,
        head: HeadPlan {
            pred: rule.head.predicate.clone(),
            slots: head_slots,
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
    /// Set by a join that reached [`MAX_ROWS`] results in one call. The join
    /// stops producing rows then, so its result never holds more than the
    /// budget before the bound is checked.
    overflow: Rc<Cell<bool>>,
}

impl Lowered {
    fn over_bound(&self, heads: &[Variable<Row>]) -> Option<String> {
        if self.overflow.get() {
            return Some(format!(
                "datafrog: more than {MAX_ROWS} rows (a join stopped at the budget)"
            ));
        }
        let over = self.rows.iter().chain(heads).any(|v| size(v) > MAX_ROWS)
            || self.keyed.iter().any(|v| size(v) > MAX_ROWS);
        over.then(|| format!("datafrog: more than {MAX_ROWS} rows"))
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

/// Keep only the rows of `acc` whose constant columns hold their constants: one
/// semijoin against a one-row relation per filter, inside datafrog.
fn apply_filters(
    it: &mut Iteration,
    out: &mut Lowered,
    mut acc: Variable<Row>,
    acc_cols: &[String],
    filters: &[(String, u32)],
) -> Result<Variable<Row>, String> {
    for (col, value) in filters {
        let Some(idx) = acc_cols.iter().position(|c| c == col) else {
            return not_applicable(format!("constant column {col} is not bound"));
        };
        let keyed = keyed_stage(it, out, &acc, vec![idx]);
        let allowed: Relation<(Row, ())> = Relation::from_vec(vec![(vec![*value], ())]);
        let next = it.variable::<Row>("filter");
        out.rows.push(next.clone());
        let n = next.clone();
        out.stmts.push(Box::new(move || {
            n.from_join(&keyed, &allowed, |_key, row, _| row.clone())
        }));
        acc = next;
    }
    Ok(acc)
}

/// Lower one rule to statements. The accumulator starts as the first positive
/// atom's relation and is joined with each further positive atom on the
/// variables they share (an empty key is a cross product); each atom's
/// constant filters are applied as soon as its columns are in; then each
/// negated atom is anti-joined, then the head slots are projected out.
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
    let (mut acc, mut acc_cols, rest): (Variable<Row>, Vec<String>, &[AtomCols]) =
        match plan.positives.split_first() {
            Some((first, rest)) => {
                let acc = var_of(&first.pred)?;
                let cols = first.cols.clone();
                (
                    apply_filters(it, out, acc, &cols, &first.filters)?,
                    cols,
                    rest,
                )
            }
            None => {
                // No positive atom: one empty row, so only a ground body holds.
                let unit = it.variable::<Row>("unit");
                unit.extend(vec![Vec::new()]);
                out.rows.push(unit.clone());
                (unit, Vec::new(), &[])
            }
        };

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
        let overflow = Rc::clone(&out.overflow);
        out.stmts.push(Box::new(move || {
            let mut made = 0usize;
            n.from_join_filtered(&l, &r, |_key, lrow, rrow| {
                if made == MAX_ROWS {
                    overflow.set(true);
                    return None;
                }
                made += 1;
                let mut row = lrow.clone();
                row.extend(extra.iter().map(|&j| rrow[j]));
                Some(row)
            })
        }));
        acc_cols.extend(new_cols);
        acc = apply_filters(it, out, next, &acc_cols, &atom.filters)?;
    }

    for comp in &plan.comparisons {
        let side = |o: &Operand| -> Result<Projected, String> {
            match o {
                Operand::Column(c) => acc_cols
                    .iter()
                    .position(|a| a == c)
                    .map(Projected::Column)
                    .ok_or_else(|| format!("{NOT_APPLICABLE}comparison column {c} is not bound")),
                Operand::Const(v) => Ok(Projected::Const(*v)),
            }
        };
        let (l, r, op) = (side(&comp.left)?, side(&comp.right)?, comp.op);
        let keyed = keyed_stage(it, out, &acc, Vec::new());
        let one: Relation<(Row, ())> = Relation::from_vec(vec![(Vec::new(), ())]);
        let next = it.variable::<Row>("compare");
        out.rows.push(next.clone());
        let n = next.clone();
        out.stmts.push(Box::new(move || {
            n.from_join_filtered(&keyed, &one, |_key, row, _| {
                let value = |p: &Projected| match p {
                    Projected::Column(i) => row[*i],
                    Projected::Const(v) => *v,
                };
                holds(op, value(&l), value(&r)).then(|| row.clone())
            })
        }));
        acc = next;
    }

    for atom in &plan.negatives {
        // Named columns only: an anonymous one is "for some value", read by
        // projecting the negated relation onto the named columns.
        let named: Vec<usize> = (0..atom.cols.len())
            .filter(|&j| !is_anonymous_col(&atom.cols[j]))
            .collect();
        let idx: Option<Vec<usize>> = named
            .iter()
            .map(|&j| acc_cols.iter().position(|a| *a == atom.cols[j]))
            .collect();
        let Some(idx) = idx else {
            return not_applicable(format!(
                "a variable of negated atom {} is not bound",
                atom.pred
            ));
        };
        let rows = load(&atom.pred, atom.cols.len(), store)?;
        let neg: Relation<Row> =
            Relation::from_vec(rows.iter().map(|row| project(row, &named)).collect());
        let keyed = keyed_stage(it, out, &acc, idx);
        let next = it.variable::<Row>("anti");
        out.rows.push(next.clone());
        let n = next.clone();
        out.stmts.push(Box::new(move || {
            n.from_antijoin(&keyed, &neg, |_key, row| row.clone())
        }));
        acc = next;
    }

    let projected: Option<Vec<Projected>> = plan
        .head
        .slots
        .iter()
        .map(|s| match s {
            HeadSlot::Column(c) => acc_cols.iter().position(|a| a == c).map(Projected::Column),
            HeadSlot::Const(v) => Some(Projected::Const(*v)),
        })
        .collect();
    let Some(projected) = projected else {
        return not_applicable("a head variable is not bound");
    };
    let head = var_of(&plan.head.pred)?;
    out.stmts.push(Box::new(move || {
        head.from_map(&acc, |row| {
            projected
                .iter()
                .map(|p| match p {
                    Projected::Column(i) => row[*i],
                    Projected::Const(v) => *v,
                })
                .collect()
        })
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
        for a in plan.positives.iter().chain(&plan.negatives) {
            note_arity(&mut arities, &a.pred, a.cols.len())?;
        }
        note_arity(&mut arities, &plan.head.pred, plan.head.slots.len())?;
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
        if let Some(why) = lowered.over_bound(&head_vars) {
            return Err(why);
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
