//! Guarded rewrite rules over measures of one population: the rule catalogue
//! and fixpoint driver of D-CMR-6
//! (`.claude/plans/d-cmr-concept-chain-normalization-v1.md` §3.5, §4, §7).
//!
//! A [`Measure`] is an integer-valued expression over subsets of ONE
//! population ([`SetExpr`] from [`crate::set_algebra`]): counts, lane sums,
//! row-range counts, and their sums and differences. [`rewrite`] applies the
//! rules of [`CATALOGUE`] until none fires or the step budget is spent.
//!
//! A rule is an identity, not a stored result. Each one names its pattern,
//! the semantic conditions it needs ([`Semantics`]), and the reason it
//! refuses when a condition fails ([`Refusal`]). A refused match keeps the
//! original expression and is recorded in the [`Trace`].
//!
//! # Size
//!
//! Every applied rewrite strictly decreases [`Measure::size`] in the order
//! `(set nodes, all nodes)`. The driver checks this on each step and does not
//! apply a rewrite that fails it, so no rule can grow an expression, and the
//! fixpoint terminates even before the budget.
//!
//! # What is not here
//!
//! Lowering, evaluation, storage, joins. The rules read no row. A consumer
//! (relalg, Quack, crepe) evaluates the rewritten measure; it keeps no second
//! copy of these rules.
//!
//! Zero dependencies.

use crate::set_algebra::{normalize, SetExpr};

/// How the arithmetic of a measure is evaluated.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Arith {
    /// Exact integers: no overflow can occur (mask-risc today: `i32` values
    /// into an `i64` sum within the row bound).
    Exact,
    /// Two's-complement wrapping: a group, so cancellation still holds.
    Wrapping,
    /// Checked: an overflow is an error the caller sees. A rewrite that
    /// removes an overflow the original would report changes the answer.
    Checked,
    /// IEEE floating point with stepwise rounding.
    Ieee,
}

/// Whether a set may be UNKNOWN on some rows (SQL NULL).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Truth {
    /// Every row is in a set or not.
    TwoValued,
    /// A row may be UNKNOWN; a measure counts only rows where the set is TRUE.
    ThreeValued,
}

/// What a lane sum returns for no rows.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SumFamily {
    /// `0` for no rows: an ordinary sum.
    Plain,
    /// The `_sym` family: a sentinel marks "no row" and is not a value.
    Sym,
}

/// Whether the population is a set of distinct rows or a bag after fan-out.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Population {
    Distinct,
    Bag,
}

/// The semantic conditions every rule's guard reads.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Semantics {
    /// The population size `N`, when it is fixed and known.
    pub universe: Option<u64>,
    pub arith: Arith,
    pub truth: Truth,
    pub sums: SumFamily,
    pub population: Population,
}

impl Semantics {
    /// Two-valued, exact, plain sums, distinct rows over a universe of `n`.
    pub fn exact(n: u64) -> Semantics {
        Semantics {
            universe: Some(n),
            arith: Arith::Exact,
            truth: Truth::TwoValued,
            sums: SumFamily::Plain,
            population: Population::Distinct,
        }
    }
}

/// A half-open interval `[lo, hi)` of row positions in the order named by
/// `basis`. Two ranges compose only on the same basis.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct RowRange {
    pub basis: u32,
    pub lo: u64,
    pub hi: u64,
}

/// An integer-valued expression over subsets of one population.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Measure {
    Const(i128),
    /// The number of rows in the set.
    Count(SetExpr),
    /// The sum of an `i32` lane over the rows in the set.
    Sum {
        lane: u16,
        set: SetExpr,
    },
    /// The number of rows in the intersection of the ranges.
    CountRows(Vec<RowRange>),
    /// An opaque operand. `pure: false` means two evaluations may differ, so
    /// it may be neither duplicated nor dropped.
    Opaque {
        id: u32,
        pure: bool,
    },
    Add(Box<Measure>, Box<Measure>),
    Sub(Box<Measure>, Box<Measure>),
}

fn set_nodes(e: &SetExpr) -> usize {
    match e {
        SetExpr::And(cs) | SetExpr::Or(cs) => 1 + cs.iter().map(set_nodes).sum::<usize>(),
        SetExpr::Not(c) => 1 + set_nodes(c),
        _ => 1,
    }
}

impl Measure {
    pub fn plus(a: Measure, b: Measure) -> Measure {
        Measure::Add(Box::new(a), Box::new(b))
    }

    pub fn minus(a: Measure, b: Measure) -> Measure {
        Measure::Sub(Box::new(a), Box::new(b))
    }

    /// `(set nodes, all nodes)`, compared lexicographically. Ranges count as
    /// set nodes.
    pub fn size(&self) -> (usize, usize) {
        match self {
            Measure::Const(_) | Measure::Opaque { .. } => (0, 1),
            Measure::Count(s) | Measure::Sum { set: s, .. } => {
                let n = set_nodes(s);
                (n, n + 1)
            }
            Measure::CountRows(rs) => (rs.len(), rs.len() + 1),
            Measure::Add(a, b) | Measure::Sub(a, b) => {
                let (sa, ta) = a.size();
                let (sb, tb) = b.size();
                (sa + sb, ta + tb + 1)
            }
        }
    }

    /// True when the measure contains a lane sum.
    pub fn has_sum(&self) -> bool {
        match self {
            Measure::Sum { .. } => true,
            Measure::Add(a, b) | Measure::Sub(a, b) => a.has_sum() || b.has_sum(),
            _ => false,
        }
    }

    /// True when the measure contains no impure operand.
    pub fn is_pure(&self) -> bool {
        match self {
            Measure::Opaque { pure, .. } => *pure,
            Measure::Add(a, b) | Measure::Sub(a, b) => a.is_pure() && b.is_pure(),
            _ => true,
        }
    }
}

/// A rule of the catalogue.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Rule {
    /// Normalize the set inside a measure ([`normalize`]).
    SetNormalize,
    /// `Count(¬A) → N − Count(A)`.
    Complement,
    /// `Sum(A) − Sum(A∧B) → Sum(A∧¬B)`, and the same for `Count`.
    RelationalDifference,
    /// `(a + b) − a → b` and `(b + a) − a → b`.
    AdditiveCancellation,
    /// `⋂ [loᵢ, hiᵢ) → [max lo, min hi)` on one basis.
    RangeIntersect,
    /// `Count([lo, hi)) → max(0, hi − lo)`.
    RangeCount,
}

/// Why a rule whose pattern matched did not fire.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Refusal {
    /// A set may be UNKNOWN on some rows.
    NotTwoValued,
    /// The universe `N` is not fixed.
    NoUniverse,
    /// Checked arithmetic: the rewrite could remove a reported overflow.
    Checked,
    /// IEEE stepwise rounding: the rewrite changes the rounding.
    Ieee,
    /// The `_sym` sum family: its empty sentinel is not a value.
    SymFamily,
    /// The operand is impure: it may be neither dropped nor duplicated.
    Impure,
    /// The ranges are on different row orders.
    BasisMismatch,
    /// A range bound lies past the universe.
    PastUniverse,
    /// The population is a bag, so a range does not count distinct rows.
    Bag,
}

/// One catalogue entry, as data.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RuleSpec {
    pub rule: Rule,
    pub pattern: &'static str,
    pub requires: &'static str,
    pub refuses: &'static [Refusal],
}

/// The catalogue, in the order the driver tries the rules at each node.
pub const CATALOGUE: &[RuleSpec] = &[
    RuleSpec {
        rule: Rule::SetNormalize,
        pattern: "Count(S) / Sum(S) with S not normalized",
        requires: "two-valued sets; the result is strictly smaller",
        refuses: &[Refusal::NotTwoValued],
    },
    RuleSpec {
        rule: Rule::Complement,
        pattern: "Count(¬A) → N − Count(A)",
        requires: "two-valued A; a fixed universe N of distinct rows; exact or wrapping arithmetic",
        refuses: &[
            Refusal::NotTwoValued,
            Refusal::NoUniverse,
            Refusal::Bag,
            Refusal::Checked,
            Refusal::Ieee,
        ],
    },
    RuleSpec {
        rule: Rule::RelationalDifference,
        pattern: "Sum(A) − Sum(A∧B) → Sum(A∧¬B); Count likewise",
        requires: "two-valued B; exact or wrapping arithmetic; plain sums",
        refuses: &[
            Refusal::NotTwoValued,
            Refusal::Checked,
            Refusal::Ieee,
            Refusal::SymFamily,
        ],
    },
    RuleSpec {
        rule: Rule::AdditiveCancellation,
        pattern: "(a + b) − a → b",
        requires: "exact or wrapping arithmetic; a pure; no `_sym` sum in a",
        refuses: &[
            Refusal::Checked,
            Refusal::Ieee,
            Refusal::Impure,
            Refusal::SymFamily,
        ],
    },
    RuleSpec {
        rule: Rule::RangeIntersect,
        pattern: "Count(⋂[loᵢ, hiᵢ)) → Count([max lo, min hi))",
        requires: "one basis; every bound within a fixed universe",
        refuses: &[
            Refusal::BasisMismatch,
            Refusal::NoUniverse,
            Refusal::PastUniverse,
        ],
    },
    RuleSpec {
        rule: Rule::RangeCount,
        pattern: "Count([lo, hi)) → max(0, hi − lo)",
        requires: "distinct rows; the bound within a fixed universe; not IEEE",
        refuses: &[
            Refusal::Bag,
            Refusal::Ieee,
            Refusal::NoUniverse,
            Refusal::PastUniverse,
        ],
    },
];

/// What [`rewrite`] did.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Trace {
    /// Applied rules, in order.
    pub steps: Vec<Rule>,
    /// Matched-but-refused rules, each `(rule, reason)` once, in first-seen
    /// order.
    pub refusals: Vec<(Rule, Refusal)>,
    /// True when the step budget ran out before a fixpoint.
    pub budget_exhausted: bool,
}

impl Trace {
    fn refuse(&mut self, rule: Rule, why: Refusal) {
        if !self.refusals.contains(&(rule, why)) {
            self.refusals.push((rule, why));
        }
    }
}

/// Every rule in [`CATALOGUE`].
pub fn all_rules() -> Vec<Rule> {
    CATALOGUE.iter().map(|s| s.rule).collect()
}

/// Apply every catalogue rule to a fixpoint, at most `budget` steps.
pub fn rewrite(m: &Measure, sem: &Semantics, budget: usize) -> (Measure, Trace) {
    rewrite_with(m, sem, budget, &all_rules())
}

/// [`rewrite`] restricted to `enabled` rules (the others never fire).
pub fn rewrite_with(
    m: &Measure,
    sem: &Semantics,
    budget: usize,
    enabled: &[Rule],
) -> (Measure, Trace) {
    let mut trace = Trace::default();
    let mut cur = m.clone();
    for _ in 0..budget {
        match step(&cur, sem, enabled, &mut trace) {
            Some((next, rule)) => {
                trace.steps.push(rule);
                cur = next;
            }
            None => return (cur, trace),
        }
    }
    // The budget is exhausted only if another step would still fire.
    let mut probe = Trace::default();
    trace.budget_exhausted = step(&cur, sem, enabled, &mut probe).is_some();
    (cur, trace)
}

/// One rewrite somewhere in `m`: the node first (so a pattern that spans
/// children is matched before the children are rewritten), then the children.
fn step(
    m: &Measure,
    sem: &Semantics,
    enabled: &[Rule],
    trace: &mut Trace,
) -> Option<(Measure, Rule)> {
    for spec in CATALOGUE {
        if !enabled.contains(&spec.rule) {
            continue;
        }
        match apply(spec.rule, m, sem) {
            Some(Ok(next)) if next.size() < m.size() => return Some((next, spec.rule)),
            Some(Ok(_)) | None => {}
            Some(Err(why)) => trace.refuse(spec.rule, why),
        }
    }
    if let Measure::Add(a, b) | Measure::Sub(a, b) = m {
        let rebuild = |a: Measure, b: Measure| match m {
            Measure::Add(..) => Measure::plus(a, b),
            _ => Measure::minus(a, b),
        };
        if let Some((na, r)) = step(a, sem, enabled, trace) {
            return Some((rebuild(na, (**b).clone()), r));
        }
        if let Some((nb, r)) = step(b, sem, enabled, trace) {
            return Some((rebuild((**a).clone(), nb), r));
        }
    }
    None
}

fn conjuncts(e: &SetExpr) -> Vec<SetExpr> {
    match e {
        SetExpr::And(cs) => cs.clone(),
        _ => vec![e.clone()],
    }
}

/// `inner = outer ∧ B`: the conjuncts of `inner` minus those of `outer`
/// (as a multiset), when every conjunct of `outer` occurs and some remain.
fn extra_conjunct(outer: &SetExpr, inner: &SetExpr) -> Option<SetExpr> {
    let mut rest = conjuncts(inner);
    for c in conjuncts(outer) {
        let at = rest.iter().position(|x| *x == c)?;
        rest.remove(at);
    }
    match rest.len() {
        0 => None,
        1 => rest.pop(),
        _ => Some(SetExpr::And(rest)),
    }
}

fn range_bounds_ok(rs: &[RowRange], sem: &Semantics) -> Result<(), Refusal> {
    let n = sem.universe.ok_or(Refusal::NoUniverse)?;
    if rs.iter().any(|r| r.hi > n || r.lo > n) {
        return Err(Refusal::PastUniverse);
    }
    Ok(())
}

/// `None`: the pattern does not match. `Some(Err)`: it matches and a guard
/// refuses. `Some(Ok)`: the rewritten node.
fn apply(rule: Rule, m: &Measure, sem: &Semantics) -> Option<Result<Measure, Refusal>> {
    match rule {
        Rule::SetNormalize => {
            let (set, rebuild): (&SetExpr, Box<dyn Fn(SetExpr) -> Measure>) = match m {
                Measure::Count(s) => (s, Box::new(Measure::Count)),
                Measure::Sum { lane, set } => {
                    let lane = *lane;
                    (set, Box::new(move |s| Measure::Sum { lane, set: s }))
                }
                _ => return None,
            };
            let n = normalize(set);
            if n == *set {
                return None;
            }
            if sem.truth != Truth::TwoValued {
                return Some(Err(Refusal::NotTwoValued));
            }
            Some(Ok(rebuild(n)))
        }
        Rule::Complement => {
            let Measure::Count(SetExpr::Not(a)) = m else {
                return None;
            };
            if sem.truth != Truth::TwoValued {
                return Some(Err(Refusal::NotTwoValued));
            }
            let Some(n) = sem.universe else {
                return Some(Err(Refusal::NoUniverse));
            };
            // N counts row positions; in a bag a row carries a multiplicity.
            if sem.population == Population::Bag {
                return Some(Err(Refusal::Bag));
            }
            // `Const(N)` may overflow, or round, where the complement count
            // does not.
            match sem.arith {
                Arith::Checked => return Some(Err(Refusal::Checked)),
                Arith::Ieee => return Some(Err(Refusal::Ieee)),
                Arith::Exact | Arith::Wrapping => {}
            }
            Some(Ok(Measure::minus(
                Measure::Const(i128::from(n)),
                Measure::Count((**a).clone()),
            )))
        }
        Rule::RelationalDifference => {
            let Measure::Sub(x, y) = m else {
                return None;
            };
            let (a, c, rebuild): (&SetExpr, &SetExpr, Box<dyn Fn(SetExpr) -> Measure>) =
                match (&**x, &**y) {
                    (Measure::Count(a), Measure::Count(c)) => (a, c, Box::new(Measure::Count)),
                    (Measure::Sum { lane: l1, set: a }, Measure::Sum { lane: l2, set: c })
                        if l1 == l2 =>
                    {
                        let lane = *l1;
                        (a, c, Box::new(move |s| Measure::Sum { lane, set: s }))
                    }
                    _ => return None,
                };
            let b = extra_conjunct(a, c)?;
            let is_sum = matches!(&**x, Measure::Sum { .. });
            if sem.truth != Truth::TwoValued {
                return Some(Err(Refusal::NotTwoValued));
            }
            match sem.arith {
                Arith::Checked => return Some(Err(Refusal::Checked)),
                Arith::Ieee => return Some(Err(Refusal::Ieee)),
                Arith::Exact | Arith::Wrapping => {}
            }
            if is_sum && sem.sums == SumFamily::Sym {
                return Some(Err(Refusal::SymFamily));
            }
            let mut cs = conjuncts(a);
            cs.push(SetExpr::negated(b));
            Some(Ok(rebuild(SetExpr::And(cs))))
        }
        Rule::AdditiveCancellation => {
            let Measure::Sub(x, a) = m else {
                return None;
            };
            let Measure::Add(p, q) = &**x else {
                return None;
            };
            let kept = if **p == **a {
                (**q).clone()
            } else if **q == **a {
                (**p).clone()
            } else {
                return None;
            };
            match sem.arith {
                Arith::Checked => return Some(Err(Refusal::Checked)),
                Arith::Ieee => return Some(Err(Refusal::Ieee)),
                Arith::Exact | Arith::Wrapping => {}
            }
            if !a.is_pure() {
                return Some(Err(Refusal::Impure));
            }
            // A `_sym` sum over no row is a sentinel, not an additive value.
            if sem.sums == SumFamily::Sym && a.has_sum() {
                return Some(Err(Refusal::SymFamily));
            }
            Some(Ok(kept))
        }
        Rule::RangeIntersect => {
            let Measure::CountRows(rs) = m else {
                return None;
            };
            if rs.len() < 2 {
                return None;
            }
            if rs.iter().any(|r| r.basis != rs[0].basis) {
                return Some(Err(Refusal::BasisMismatch));
            }
            if let Err(why) = range_bounds_ok(rs, sem) {
                return Some(Err(why));
            }
            let lo = rs.iter().map(|r| r.lo).max().unwrap_or(0);
            let hi = rs.iter().map(|r| r.hi).min().unwrap_or(0);
            Some(Ok(Measure::CountRows(vec![RowRange {
                basis: rs[0].basis,
                lo,
                hi: hi.max(lo),
            }])))
        }
        Rule::RangeCount => {
            let Measure::CountRows(rs) = m else {
                return None;
            };
            let [r] = rs.as_slice() else {
                return None;
            };
            if sem.population == Population::Bag {
                return Some(Err(Refusal::Bag));
            }
            // Stepwise rounding of the count differs from one conversion.
            if sem.arith == Arith::Ieee {
                return Some(Err(Refusal::Ieee));
            }
            if let Err(why) = range_bounds_ok(rs, sem) {
                return Some(Err(why));
            }
            Some(Ok(Measure::Const(i128::from(r.hi.saturating_sub(r.lo)))))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::set_algebra::{Cmp, Res};

    // ── the independent oracle: a materialized table ─────────────────────

    /// Rows of `i32` lanes (`None` = NULL) and two-valued symbol columns.
    struct Table {
        lanes: Vec<Vec<Option<i32>>>,
        syms: Vec<Vec<bool>>,
        /// The multiplicity of each row (1 for a distinct population).
        mult: Vec<u64>,
    }

    impl Table {
        fn n(&self) -> usize {
            self.mult.len()
        }
    }

    /// Kleene evaluation: `None` is UNKNOWN.
    fn set_at(t: &Table, e: &SetExpr, row: usize) -> Option<bool> {
        match e {
            SetExpr::All => Some(true),
            SetExpr::Empty => Some(false),
            SetExpr::Sym(k) => Some(t.syms[*k as usize][row]),
            SetExpr::Res(Res::I32 { lane, op, t: c }) => {
                t.lanes[*lane as usize][row].map(|v| match op {
                    Cmp::Lt => v < *c,
                    Cmp::Le => v <= *c,
                    Cmp::Gt => v > *c,
                    Cmp::Ge => v >= *c,
                    Cmp::Eq => v == *c,
                    Cmp::Ne => v != *c,
                })
            }
            SetExpr::Res(Res::U32 { lane, eq, v }) => {
                t.lanes[*lane as usize][row].map(|x| (x as u32 == *v) == *eq)
            }
            SetExpr::Not(c) => set_at(t, c, row).map(|b| !b),
            SetExpr::And(cs) => {
                let mut acc = Some(true);
                for c in cs {
                    acc = match (acc, set_at(t, c, row)) {
                        (Some(false), _) | (_, Some(false)) => Some(false),
                        (Some(true), Some(true)) => Some(true),
                        _ => None,
                    };
                }
                acc
            }
            SetExpr::Or(cs) => {
                let mut acc = Some(false);
                for c in cs {
                    acc = match (acc, set_at(t, c, row)) {
                        (Some(true), _) | (_, Some(true)) => Some(true),
                        (Some(false), Some(false)) => Some(false),
                        _ => None,
                    };
                }
                acc
            }
        }
    }

    /// How the oracle accumulates.
    #[derive(Clone, Copy, PartialEq, Debug)]
    enum Acc {
        Exact,
        Wrap64,
        Checked64,
        /// An eager accumulator into `i32`: the counterexample, not a model
        /// of today's fold.
        Checked32,
        F32,
    }

    #[derive(Debug, PartialEq, Clone, Copy)]
    enum Val {
        I(i128),
        F(f32),
        Overflow,
    }

    fn fit(acc: Acc, v: i128) -> Val {
        match acc {
            Acc::Exact => Val::I(v),
            Acc::Wrap64 => Val::I(i128::from(v as i64)),
            Acc::Checked64 => i64::try_from(v).map_or(Val::Overflow, |x| Val::I(x.into())),
            Acc::Checked32 => i32::try_from(v).map_or(Val::Overflow, |x| Val::I(x.into())),
            Acc::F32 => Val::F(v as f32),
        }
    }

    fn binop(acc: Acc, a: Val, b: Val, sub: bool) -> Val {
        match (a, b) {
            (Val::Overflow, _) | (_, Val::Overflow) => Val::Overflow,
            (Val::F(x), Val::F(y)) => Val::F(if sub { x - y } else { x + y }),
            (Val::I(x), Val::I(y)) => fit(acc, if sub { x - y } else { x + y }),
            _ => unreachable!(),
        }
    }

    /// Evaluate a measure. `calls` counts impure-operand evaluations.
    fn eval(t: &Table, m: &Measure, acc: Acc, calls: &mut i128) -> Val {
        match m {
            Measure::Const(c) => fit(acc, *c),
            Measure::Opaque { id, pure } => {
                if *pure {
                    fit(acc, i128::from(*id))
                } else {
                    *calls += 1;
                    fit(acc, *calls)
                }
            }
            Measure::Count(s) => {
                let mut v = fit(acc, 0);
                for r in 0..t.n() {
                    if set_at(t, s, r) == Some(true) {
                        v = binop(acc, v, fit(acc, t.mult[r].into()), false);
                    }
                }
                v
            }
            Measure::Sum { lane, set } => {
                let mut v = fit(acc, 0);
                for r in 0..t.n() {
                    if set_at(t, set, r) == Some(true) {
                        if let Some(x) = t.lanes[*lane as usize][r] {
                            for _ in 0..t.mult[r] {
                                v = binop(acc, v, fit(acc, x.into()), false);
                            }
                        }
                    }
                }
                v
            }
            Measure::CountRows(rs) => {
                // Rows are positions; every range here is on basis 0's order.
                let mut v = fit(acc, 0);
                for r in 0..t.n() {
                    if rs.iter().all(|x| (x.lo..x.hi).contains(&(r as u64))) {
                        v = binop(acc, v, fit(acc, t.mult[r].into()), false);
                    }
                }
                v
            }
            Measure::Add(a, b) => {
                let x = eval(t, a, acc, calls);
                binop(acc, x, eval(t, b, acc, calls), false)
            }
            Measure::Sub(a, b) => {
                let x = eval(t, a, acc, calls);
                binop(acc, x, eval(t, b, acc, calls), true)
            }
        }
    }

    fn ev(t: &Table, m: &Measure, acc: Acc) -> Val {
        eval(t, m, acc, &mut 0)
    }

    // ── random inputs ────────────────────────────────────────────────────

    struct Rng(u64);
    impl Rng {
        fn next(&mut self) -> u64 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            self.0
        }
        fn below(&mut self, n: u64) -> u64 {
            self.next() % n
        }
    }

    const LANES: u16 = 3;
    const SYMS: u32 = 3;

    fn table(rng: &mut Rng, n: usize, nulls: bool, wide: bool) -> Table {
        let lanes = (0..LANES)
            .map(|_| {
                (0..n)
                    .map(|_| {
                        if nulls && rng.below(5) == 0 {
                            None
                        } else if wide {
                            Some(rng.next() as i32)
                        } else {
                            Some(rng.below(9) as i32 - 4)
                        }
                    })
                    .collect()
            })
            .collect();
        let syms = (0..SYMS)
            .map(|_| (0..n).map(|_| rng.below(2) == 0).collect())
            .collect();
        Table {
            lanes,
            syms,
            mult: vec![1; n],
        }
    }

    fn leaf(rng: &mut Rng) -> SetExpr {
        match rng.below(3) {
            0 => SetExpr::Sym(rng.below(u64::from(SYMS)) as u32),
            1 => SetExpr::i32(
                rng.below(u64::from(LANES)) as u16,
                [Cmp::Lt, Cmp::Le, Cmp::Gt, Cmp::Ge, Cmp::Eq, Cmp::Ne][rng.below(6) as usize],
                rng.below(9) as i32 - 4,
            ),
            _ => SetExpr::negated(SetExpr::Sym(rng.below(u64::from(SYMS)) as u32)),
        }
    }

    fn set(rng: &mut Rng, depth: u32) -> SetExpr {
        if depth == 0 || rng.below(3) == 0 {
            return leaf(rng);
        }
        let k = 2 + rng.below(2) as usize;
        let cs = (0..k).map(|_| set(rng, depth - 1)).collect();
        match rng.below(3) {
            0 => SetExpr::And(cs),
            1 => SetExpr::Or(cs),
            _ => SetExpr::negated(SetExpr::And(cs)),
        }
    }

    /// A measure built so that the catalogue's patterns occur.
    fn measure(rng: &mut Rng, depth: u32) -> Measure {
        let lane = rng.below(u64::from(LANES)) as u16;
        match rng.below(if depth == 0 { 4 } else { 7 }) {
            0 => Measure::Count(set(rng, 2)),
            1 => Measure::Count(SetExpr::negated(set(rng, 2))),
            2 => Measure::Sum {
                lane,
                set: set(rng, 2),
            },
            3 => Measure::Const(rng.below(20) as i128 - 10),
            4 => {
                let a = set(rng, 1);
                let b = set(rng, 1);
                let mut cs = conjuncts(&a);
                cs.push(b);
                if rng.below(2) == 0 {
                    Measure::minus(
                        Measure::Sum { lane, set: a },
                        Measure::Sum {
                            lane,
                            set: SetExpr::And(cs),
                        },
                    )
                } else {
                    Measure::minus(Measure::Count(a), Measure::Count(SetExpr::And(cs)))
                }
            }
            5 => {
                let a = measure(rng, depth - 1);
                let b = measure(rng, depth - 1);
                Measure::minus(Measure::plus(a.clone(), b), a)
            }
            _ => Measure::plus(measure(rng, depth - 1), measure(rng, depth - 1)),
        }
    }

    const BUDGET: usize = 256;

    // ── the differential property ────────────────────────────────────────

    /// 600 random measures over random two-valued tables, every subset of
    /// rules: the answer never changes, the size never grows, and the rules
    /// actually fire (anti-vacuity).
    #[test]
    fn rewrites_keep_the_answer_under_exact_and_wrapping() {
        let mut rng = Rng(0x9E37_79B9_7F4A_7C15);
        let rules = all_rules();
        let (mut fired, mut changed) = (0usize, 0usize);
        for i in 0..600 {
            let n = 1 + rng.below(40) as usize;
            let t = table(&mut rng, n, false, i % 3 == 0);
            let m = measure(&mut rng, 2);
            let mask = rng.next();
            let enabled: Vec<Rule> = rules
                .iter()
                .enumerate()
                .filter(|(k, _)| i % 2 == 0 || mask >> k & 1 == 1)
                .map(|(_, r)| *r)
                .collect();
            for (arith, acc) in [(Arith::Exact, Acc::Exact), (Arith::Wrapping, Acc::Wrap64)] {
                let sem = Semantics {
                    arith,
                    ..Semantics::exact(n as u64)
                };
                let (r, tr) = rewrite_with(&m, &sem, BUDGET, &enabled);
                assert!(!tr.budget_exhausted, "no fixpoint for {m:?}");
                assert!(r.size() <= m.size(), "size grew: {m:?} -> {r:?}");
                assert!(
                    tr.refusals.is_empty(),
                    "refused on valid input: {:?}",
                    tr.refusals
                );
                assert_eq!(ev(&t, &m, acc), ev(&t, &r, acc), "{m:?}\n-> {r:?}");
                fired += tr.steps.len();
                changed += usize::from(r != m);
            }
        }
        // Anti-vacuity: the catalogue rewrote most inputs, not a few.
        assert!(changed * 2 > 1200, "only {changed} of 1200 rewritten");
        assert!(fired > 1200, "only {fired} steps fired");
    }

    /// Each rule fires on its own at least once in the random run above's
    /// distribution (so a dead rule is visible).
    #[test]
    fn every_rule_fires_on_random_input() {
        let mut rng = Rng(7);
        let mut seen = std::collections::BTreeSet::new();
        for _ in 0..300 {
            let m = measure(&mut rng, 2);
            let (_, tr) = rewrite(&m, &Semantics::exact(30), BUDGET);
            seen.extend(tr.steps);
        }
        // Range rules are exercised by experiment D, not by this generator.
        for r in [
            Rule::SetNormalize,
            Rule::Complement,
            Rule::RelationalDifference,
            Rule::AdditiveCancellation,
        ] {
            assert!(seen.contains(&r), "{r:?} never fired");
        }
    }

    // ── experiment A: cancellation ───────────────────────────────────────

    fn diff_ab() -> (Measure, SetExpr, SetExpr) {
        let a = SetExpr::Sym(0);
        let b = SetExpr::i32(1, Cmp::Gt, 0);
        let m = Measure::minus(
            Measure::Sum {
                lane: 0,
                set: a.clone(),
            },
            Measure::Sum {
                lane: 0,
                set: SetExpr::And(vec![a.clone(), b.clone()]),
            },
        );
        (m, a, b)
    }

    #[test]
    fn experiment_a_relational_difference_matches_the_i128_oracle() {
        let (m, _, _) = diff_ab();
        let (r, tr) = rewrite(&m, &Semantics::exact(64), BUDGET);
        assert!(tr.steps.contains(&Rule::RelationalDifference), "{tr:?}");
        assert!(matches!(r, Measure::Sum { .. }), "not one sum: {r:?}");
        let mut rng = Rng(11);
        for _ in 0..200 {
            let t = table(&mut rng, 64, false, true);
            assert_eq!(ev(&t, &m, Acc::Exact), ev(&t, &r, Acc::Exact));
        }
    }

    /// The eager `i32` accumulator is the counterexample: the original sum
    /// overflows while the rewritten one fits, so under checked arithmetic
    /// the rewrite would hide an error. The rule must refuse there.
    #[test]
    fn experiment_a_checked_arithmetic_refuses_and_the_i32_counterexample_shows_why() {
        let (m, _, _) = diff_ab();
        // Three rows in A; the two big ones are also in B.
        let t = Table {
            lanes: vec![
                vec![Some(i32::MAX), Some(i32::MAX), Some(5)],
                vec![Some(1), Some(1), Some(-1)],
                vec![Some(0); 3],
            ],
            syms: vec![vec![true; 3], vec![false; 3], vec![false; 3]],
            mult: vec![1; 3],
        };
        let rewritten = Measure::Sum {
            lane: 0,
            set: SetExpr::And(vec![
                SetExpr::Sym(0),
                SetExpr::negated(SetExpr::i32(1, Cmp::Gt, 0)),
            ]),
        };
        assert_eq!(ev(&t, &m, Acc::Checked32), Val::Overflow);
        assert_eq!(ev(&t, &rewritten, Acc::Checked32), Val::I(5));
        assert_eq!(ev(&t, &m, Acc::Exact), Val::I(5));

        let sem = Semantics {
            arith: Arith::Checked,
            ..Semantics::exact(3)
        };
        let (r, tr) = rewrite(&m, &sem, BUDGET);
        assert_eq!(r, m, "checked arithmetic must keep the original");
        assert!(tr
            .refusals
            .contains(&(Rule::RelationalDifference, Refusal::Checked)));
    }

    // ── experiment D: composition ────────────────────────────────────────

    #[test]
    fn experiment_d_intersection_enables_the_range_count() {
        let rs = vec![
            RowRange {
                basis: 0,
                lo: 3,
                hi: 40,
            },
            RowRange {
                basis: 0,
                lo: 10,
                hi: 55,
            },
            RowRange {
                basis: 0,
                lo: 0,
                hi: 30,
            },
        ];
        let m = Measure::CountRows(rs);
        let (r, tr) = rewrite(&m, &Semantics::exact(64), BUDGET);
        assert_eq!(tr.steps, vec![Rule::RangeIntersect, Rule::RangeCount]);
        assert_eq!(r, Measure::Const(20));
        let t = table(&mut Rng(3), 64, false, false);
        assert_eq!(ev(&t, &m, Acc::Exact), Val::I(20));

        // Without the intersection the count cannot fire: composition is real.
        let (r2, tr2) = rewrite_with(&m, &Semantics::exact(64), BUDGET, &[Rule::RangeCount]);
        assert_eq!(r2, m);
        assert!(tr2.steps.is_empty());

        // Disjoint ranges count zero.
        let disjoint = Measure::CountRows(vec![
            RowRange {
                basis: 0,
                lo: 0,
                hi: 5,
            },
            RowRange {
                basis: 0,
                lo: 9,
                hi: 12,
            },
        ]);
        assert_eq!(
            rewrite(&disjoint, &Semantics::exact(64), BUDGET).0,
            Measure::Const(0)
        );
        assert_eq!(ev(&t, &disjoint, Acc::Exact), Val::I(0));
    }

    // ── experiment E: invalid rewrites refuse ────────────────────────────

    /// NULL: with a three-valued set the complement miscounts, so the rule
    /// refuses; the unguarded form is shown wrong on the same table.
    #[test]
    fn experiment_e_null_refuses_the_complement() {
        let a = SetExpr::i32(0, Cmp::Gt, 0);
        let m = Measure::Count(SetExpr::negated(a.clone()));
        let t = Table {
            lanes: vec![
                vec![Some(1), None, Some(-1), None],
                vec![Some(0); 4],
                vec![Some(0); 4],
            ],
            syms: vec![vec![true; 4]; 3],
            mult: vec![1; 4],
        };
        let unguarded = Measure::minus(Measure::Const(4), Measure::Count(a));
        assert_ne!(ev(&t, &m, Acc::Exact), ev(&t, &unguarded, Acc::Exact));

        let sem = Semantics {
            truth: Truth::ThreeValued,
            ..Semantics::exact(4)
        };
        let (r, tr) = rewrite(&m, &sem, BUDGET);
        assert_eq!(r, m);
        assert!(tr
            .refusals
            .contains(&(Rule::Complement, Refusal::NotTwoValued)));
        assert!(tr
            .refusals
            .contains(&(Rule::SetNormalize, Refusal::NotTwoValued)));
    }

    /// NULL: two-valued normalization turns `A ∨ ¬A` into `All`, which
    /// counts the UNKNOWN rows too.
    #[test]
    fn experiment_e_null_refuses_set_normalization() {
        let a = SetExpr::i32(0, Cmp::Gt, 0);
        let m = Measure::Count(SetExpr::Or(vec![a.clone(), SetExpr::negated(a)]));
        let t = Table {
            lanes: vec![
                vec![Some(1), None, Some(-1)],
                vec![Some(0); 3],
                vec![Some(0); 3],
            ],
            syms: vec![vec![true; 3]; 3],
            mult: vec![1; 3],
        };
        assert_eq!(ev(&t, &m, Acc::Exact), Val::I(2));
        assert_eq!(ev(&t, &Measure::Count(SetExpr::All), Acc::Exact), Val::I(3));
        let sem = Semantics {
            truth: Truth::ThreeValued,
            ..Semantics::exact(3)
        };
        let (r, tr) = rewrite(&m, &sem, BUDGET);
        assert_eq!(ev(&t, &r, Acc::Exact), Val::I(2), "rewritten to {r:?}");
        assert!(tr
            .refusals
            .contains(&(Rule::SetNormalize, Refusal::NotTwoValued)));
        // Two-valued, the same expression is the whole universe.
        let (r2, _) = rewrite(&m, &Semantics::exact(3), BUDGET);
        assert_eq!(r2, Measure::Count(SetExpr::All));
    }

    #[test]
    fn experiment_e_null_refuses_the_relational_difference() {
        let (m, a, b) = diff_ab();
        let t = Table {
            lanes: vec![
                vec![Some(7), Some(9)],
                vec![Some(1), None],
                vec![Some(0); 2],
            ],
            syms: vec![vec![true; 2], vec![false; 2], vec![false; 2]],
            mult: vec![1; 2],
        };
        let mut cs = conjuncts(&a);
        cs.push(SetExpr::negated(b));
        let unguarded = Measure::Sum {
            lane: 0,
            set: SetExpr::And(cs),
        };
        assert_ne!(ev(&t, &m, Acc::Exact), ev(&t, &unguarded, Acc::Exact));

        let sem = Semantics {
            truth: Truth::ThreeValued,
            ..Semantics::exact(2)
        };
        let (r, tr) = rewrite(&m, &sem, BUDGET);
        assert_eq!(r, m);
        assert!(tr
            .refusals
            .contains(&(Rule::RelationalDifference, Refusal::NotTwoValued)));
    }

    #[test]
    fn experiment_e_no_universe_refuses_the_complement() {
        let m = Measure::Count(SetExpr::negated(SetExpr::Sym(0)));
        let sem = Semantics {
            universe: None,
            ..Semantics::exact(0)
        };
        let (r, tr) = rewrite(&m, &sem, BUDGET);
        assert_eq!(r, m);
        assert_eq!(tr.refusals, vec![(Rule::Complement, Refusal::NoUniverse)]);
    }

    /// IEEE: in `f32`, `Sum(A) − Sum(A∧B)` and `Sum(A∧¬B)` round differently.
    #[test]
    fn experiment_e_ieee_refuses_cancellation_rules() {
        let (m, _, _) = diff_ab();
        let t = Table {
            lanes: vec![
                vec![Some(1 << 24), Some(1), Some(1)],
                vec![Some(1), Some(-1), Some(-1)],
                vec![Some(0); 3],
            ],
            syms: vec![vec![true; 3], vec![false; 3], vec![false; 3]],
            mult: vec![1; 3],
        };
        let unguarded = Measure::Sum {
            lane: 0,
            set: SetExpr::And(vec![
                SetExpr::Sym(0),
                SetExpr::negated(SetExpr::i32(1, Cmp::Gt, 0)),
            ]),
        };
        assert_ne!(ev(&t, &m, Acc::F32), ev(&t, &unguarded, Acc::F32));

        let sem = Semantics {
            arith: Arith::Ieee,
            ..Semantics::exact(3)
        };
        let (r, tr) = rewrite(&m, &sem, BUDGET);
        assert_eq!(r, m);
        assert!(tr
            .refusals
            .contains(&(Rule::RelationalDifference, Refusal::Ieee)));

        let big = Measure::Const(1 << 24);
        let c = Measure::minus(Measure::plus(big.clone(), Measure::Const(1)), big.clone());
        assert_ne!(ev(&t, &c, Acc::F32), Val::F(1.0));
        let (r, tr) = rewrite(&c, &sem, BUDGET);
        assert_eq!(r, c);
        assert!(tr
            .refusals
            .contains(&(Rule::AdditiveCancellation, Refusal::Ieee)));
    }

    #[test]
    fn experiment_e_checked_refuses_additive_cancellation() {
        let max = Measure::Const(i128::from(i64::MAX));
        let c = Measure::minus(Measure::plus(max.clone(), Measure::Const(1)), max);
        let t = table(&mut Rng(5), 1, false, false);
        assert_eq!(ev(&t, &c, Acc::Checked64), Val::Overflow);
        assert_eq!(ev(&t, &Measure::Const(1), Acc::Checked64), Val::I(1));
        let sem = Semantics {
            arith: Arith::Checked,
            ..Semantics::exact(1)
        };
        let (r, tr) = rewrite(&c, &sem, BUDGET);
        assert_eq!(r, c);
        assert!(tr
            .refusals
            .contains(&(Rule::AdditiveCancellation, Refusal::Checked)));
    }

    /// Impure: dropping an operand that is evaluated twice changes the answer.
    #[test]
    fn experiment_e_impure_refuses_additive_cancellation() {
        let call = Measure::Opaque { id: 0, pure: false };
        let c = Measure::minus(Measure::plus(call.clone(), Measure::Const(10)), call);
        let t = table(&mut Rng(5), 1, false, false);
        assert_ne!(ev(&t, &c, Acc::Exact), Val::I(10));
        let (r, tr) = rewrite(&c, &Semantics::exact(1), BUDGET);
        assert_eq!(r, c);
        assert_eq!(
            tr.refusals,
            vec![(Rule::AdditiveCancellation, Refusal::Impure)]
        );

        // The pure twin cancels.
        let pure = Measure::Opaque { id: 3, pure: true };
        let p = Measure::minus(Measure::plus(Measure::Const(10), pure.clone()), pure);
        assert_eq!(
            rewrite(&p, &Semantics::exact(1), BUDGET).0,
            Measure::Const(10)
        );
    }

    #[test]
    fn experiment_e_sym_family_refuses_the_sum_difference() {
        let (m, _, _) = diff_ab();
        let sem = Semantics {
            sums: SumFamily::Sym,
            ..Semantics::exact(8)
        };
        let (r, tr) = rewrite(&m, &sem, BUDGET);
        assert_eq!(r, m);
        assert!(tr
            .refusals
            .contains(&(Rule::RelationalDifference, Refusal::SymFamily)));
        // The Count difference has no sentinel and still fires.
        let cm = Measure::minus(
            Measure::Count(SetExpr::Sym(0)),
            Measure::Count(SetExpr::And(vec![SetExpr::Sym(0), SetExpr::Sym(1)])),
        );
        let (r, tr) = rewrite(&cm, &sem, BUDGET);
        assert!(tr.steps.contains(&Rule::RelationalDifference), "{r:?}");
    }

    /// Incompatible coordinates: ranges on two row orders never intersect,
    /// and a bound past the universe refuses.
    #[test]
    fn experiment_e_incompatible_coordinates_refuse() {
        let m = Measure::CountRows(vec![
            RowRange {
                basis: 0,
                lo: 0,
                hi: 10,
            },
            RowRange {
                basis: 1,
                lo: 5,
                hi: 20,
            },
        ]);
        let (r, tr) = rewrite(&m, &Semantics::exact(64), BUDGET);
        assert_eq!(r, m);
        assert_eq!(
            tr.refusals,
            vec![(Rule::RangeIntersect, Refusal::BasisMismatch)]
        );

        let past = Measure::CountRows(vec![RowRange {
            basis: 0,
            lo: 0,
            hi: 70,
        }]);
        let (r, tr) = rewrite(&past, &Semantics::exact(64), BUDGET);
        assert_eq!(r, past);
        assert_eq!(tr.refusals, vec![(Rule::RangeCount, Refusal::PastUniverse)]);
    }

    /// Bag: after fan-out a range holds rows with multiplicity, so
    /// `hi − lo` undercounts.
    #[test]
    fn experiment_e_bag_refuses_the_range_count() {
        let m = Measure::CountRows(vec![RowRange {
            basis: 0,
            lo: 0,
            hi: 4,
        }]);
        let mut t = table(&mut Rng(9), 4, false, false);
        t.mult = vec![1, 3, 1, 2];
        assert_ne!(ev(&t, &m, Acc::Exact), Val::I(4));
        let sem = Semantics {
            population: Population::Bag,
            ..Semantics::exact(4)
        };
        let (r, tr) = rewrite(&m, &sem, BUDGET);
        assert_eq!(r, m);
        assert_eq!(tr.refusals, vec![(Rule::RangeCount, Refusal::Bag)]);
    }

    /// Bag: `N` counts row positions, but a row carries a multiplicity, so
    /// `N − Count(A)` miscounts (Codex review of #1651).
    #[test]
    fn experiment_e_bag_refuses_the_complement() {
        let m = Measure::Count(SetExpr::negated(SetExpr::Sym(0)));
        let mut t = table(&mut Rng(13), 2, false, false);
        t.syms[0] = vec![true, false];
        t.mult = vec![2, 1];
        let unguarded = Measure::minus(Measure::Const(2), Measure::Count(SetExpr::Sym(0)));
        assert_eq!(ev(&t, &m, Acc::Exact), Val::I(1));
        assert_eq!(ev(&t, &unguarded, Acc::Exact), Val::I(0));
        let sem = Semantics {
            population: Population::Bag,
            ..Semantics::exact(2)
        };
        let (r, tr) = rewrite(&m, &sem, BUDGET);
        assert_eq!(r, m);
        assert_eq!(tr.refusals, vec![(Rule::Complement, Refusal::Bag)]);
    }

    /// Checked / IEEE: `Const(N)` overflows (or rounds) on its own while the
    /// complement count is small.
    #[test]
    fn experiment_e_checked_and_ieee_refuse_the_complement() {
        let m = Measure::Count(SetExpr::negated(SetExpr::Sym(0)));
        let t = table(&mut Rng(17), 4, false, false);
        let n = 1u64 << 63;
        let unguarded = Measure::minus(
            Measure::Const(i128::from(n)),
            Measure::Count(SetExpr::Sym(0)),
        );
        assert_eq!(ev(&t, &unguarded, Acc::Checked64), Val::Overflow);
        assert!(matches!(ev(&t, &m, Acc::Checked64), Val::I(_)));
        for (arith, why) in [
            (Arith::Checked, Refusal::Checked),
            (Arith::Ieee, Refusal::Ieee),
        ] {
            let sem = Semantics {
                arith,
                ..Semantics::exact(n)
            };
            let (r, tr) = rewrite(&m, &sem, BUDGET);
            assert_eq!(r, m);
            assert_eq!(tr.refusals, vec![(Rule::Complement, why)]);
        }
        // Wrapping is a group: the complement still fires.
        let sem = Semantics {
            arith: Arith::Wrapping,
            ..Semantics::exact(4)
        };
        assert!(rewrite(&m, &sem, BUDGET)
            .1
            .steps
            .contains(&Rule::Complement));
    }

    /// `_sym` sums: the dropped operand may be the empty sentinel, which does
    /// not cancel (Codex review of #1651).
    #[test]
    fn experiment_e_sym_family_refuses_cancelling_a_sum() {
        let s = Measure::Sum {
            lane: 0,
            set: SetExpr::Empty,
        };
        let c = Measure::minus(Measure::plus(s.clone(), Measure::Const(3)), s);
        let sem = Semantics {
            sums: SumFamily::Sym,
            ..Semantics::exact(4)
        };
        let (r, tr) = rewrite(&c, &sem, BUDGET);
        assert_eq!(r, c);
        assert_eq!(
            tr.refusals,
            vec![(Rule::AdditiveCancellation, Refusal::SymFamily)]
        );
        // A dropped operand with no sum still cancels under `_sym`.
        let k = Measure::Count(SetExpr::Sym(0));
        let d = Measure::minus(Measure::plus(k.clone(), Measure::Const(3)), k);
        assert_eq!(rewrite(&d, &sem, BUDGET).0, Measure::Const(3));
    }

    /// IEEE: counting a range step by step in `f32` rounds; `hi − lo`
    /// converted once does not.
    #[test]
    fn experiment_e_ieee_refuses_the_range_count() {
        let m = Measure::CountRows(vec![RowRange {
            basis: 0,
            lo: 0,
            hi: 1 << 25,
        }]);
        let sem = Semantics {
            arith: Arith::Ieee,
            ..Semantics::exact(1 << 25)
        };
        let (r, tr) = rewrite(&m, &sem, BUDGET);
        assert_eq!(r, m);
        assert_eq!(tr.refusals, vec![(Rule::RangeCount, Refusal::Ieee)]);
        // Stepwise f32 increments stop at 2^24.
        let mut acc = 0f32;
        for _ in 0..(1u32 << 25) {
            acc += 1.0;
        }
        assert_ne!(acc, (1u32 << 25) as f32);
    }

    // ── driver ───────────────────────────────────────────────────────────

    #[test]
    fn the_budget_stops_the_driver_and_says_so() {
        let mut m = Measure::Count(SetExpr::negated(SetExpr::Sym(0)));
        for k in 0..10 {
            let a = Measure::Count(SetExpr::negated(SetExpr::Sym(k % SYMS)));
            m = Measure::plus(m, a);
        }
        let (r, tr) = rewrite(&m, &Semantics::exact(9), 3);
        assert_eq!(tr.steps.len(), 3);
        assert!(tr.budget_exhausted);
        let (done, tr2) = rewrite(&r, &Semantics::exact(9), BUDGET);
        assert!(!tr2.budget_exhausted);
        assert_eq!(done, rewrite(&m, &Semantics::exact(9), BUDGET).0);
    }

    #[test]
    fn the_catalogue_lists_every_rule_once() {
        let rules = all_rules();
        let mut sorted = rules.clone();
        sorted.sort();
        sorted.dedup();
        assert_eq!(sorted.len(), rules.len());
        assert_eq!(rules.len(), 6);
        assert!(CATALOGUE.iter().all(|s| !s.refuses.is_empty()));
    }
}
