//! Rewrites: each takes a description to an equivalent, cheaper one, before
//! anything runs. None of them reads a row.
//!
//! The rewrites themselves live in
//! [`lance_graph_contract::set_algebra::normalize`] (D-CMR-1 / D-RA-2,
//! `ADR-2026-10-20-SIX` §2): the contract owns what a selection means, this
//! crate owns how it executes. Here a [`Rel`] is converted to the contract's
//! [`SetExpr`] (`Plane(i)` is the opaque set `Sym(i)`; an [`Atom`] is the
//! residual [`Res`] with the same fields), normalized there, and converted
//! back. The rule list is documented at the contract module.
//!
//! [`symbolic`] likewise asks the contract's rule catalogue
//! ([`lance_graph_contract::set_rules`], D-CMR-6) whether the measure folds
//! to a constant, instead of matching `Empty` / `All` here.

use lance_graph_contract::set_algebra::{self as sa, Res, SetExpr};
use lance_graph_contract::set_rules::{self as sr, Semantics};

use crate::ir::{Atom, Cmp, Measure, Rel};
use crate::lower::Answer;

/// Rewrite `rel` to an equivalent description, to a fixpoint.
pub fn normalize(rel: &Rel) -> Rel {
    from_set(&sa::normalize(&to_set(rel)))
}

/// Step budget for [`symbolic`]'s rewrite; every step shrinks the measure, so
/// a description this crate lowers reaches its fixpoint well inside it.
const SYMBOLIC_BUDGET: usize = 256;

/// The answer, if the description alone determines it: no row is read and
/// no program runs.
///
/// The decision is the contract's: the measure is rewritten by
/// [`set_rules::rewrite`](sr::rewrite) under this crate's semantics (planes
/// and lanes are two-valued, sums are exact `i64` over `i32`, rows are
/// distinct, the universe is `n_rows`), and answered only when it folds to a
/// constant. `Min` / `Max` are answered (`None`) only when the rows are
/// provably empty, i.e. when their `Count` folds to `0`.
pub fn symbolic(rel: &Rel, m: Measure, n_rows: usize) -> Option<Answer> {
    let set = to_set(rel);
    let probe = match m {
        Measure::Count | Measure::Min(_) | Measure::Max(_) => sr::Measure::Count(set),
        Measure::Sum(lane) => sr::Measure::Sum { lane, set },
        Measure::Keep => return None,
    };
    let sem = Semantics::exact(n_rows as u64);
    let sr::Measure::Const(c) = sr::rewrite(&probe, &sem, SYMBOLIC_BUDGET).0 else {
        return None;
    };
    match m {
        Measure::Count => u64::try_from(c).ok().map(Answer::Count),
        Measure::Sum(_) => i64::try_from(c).ok().map(Answer::Sum),
        Measure::Min(_) => (c == 0).then_some(Answer::Min(None)),
        Measure::Max(_) => (c == 0).then_some(Answer::Max(None)),
        Measure::Keep => None,
    }
}

fn cmp_to(c: Cmp) -> sa::Cmp {
    match c {
        Cmp::Lt => sa::Cmp::Lt,
        Cmp::Le => sa::Cmp::Le,
        Cmp::Gt => sa::Cmp::Gt,
        Cmp::Ge => sa::Cmp::Ge,
        Cmp::Eq => sa::Cmp::Eq,
        Cmp::Ne => sa::Cmp::Ne,
    }
}

fn cmp_from(c: sa::Cmp) -> Cmp {
    match c {
        sa::Cmp::Lt => Cmp::Lt,
        sa::Cmp::Le => Cmp::Le,
        sa::Cmp::Gt => Cmp::Gt,
        sa::Cmp::Ge => Cmp::Ge,
        sa::Cmp::Eq => Cmp::Eq,
        sa::Cmp::Ne => Cmp::Ne,
    }
}

pub(crate) fn to_set(rel: &Rel) -> SetExpr {
    match rel {
        Rel::All => SetExpr::All,
        Rel::Empty => SetExpr::Empty,
        Rel::Plane(i) => SetExpr::Sym(u32::from(*i)),
        Rel::Atom(Atom::I32 { lane, op, t }) => SetExpr::Res(Res::I32 {
            lane: *lane,
            op: cmp_to(*op),
            t: *t,
        }),
        Rel::Atom(Atom::U32 { lane, eq, v }) => SetExpr::Res(Res::U32 {
            lane: *lane,
            eq: *eq,
            v: *v,
        }),
        Rel::And(cs) => SetExpr::And(cs.iter().map(to_set).collect()),
        Rel::Or(cs) => SetExpr::Or(cs.iter().map(to_set).collect()),
        Rel::Not(c) => SetExpr::Not(Box::new(to_set(c))),
    }
}

/// Inverse of [`to_set`]. Every `Sym` the contract returns came from a
/// `Plane`, so the narrowing never fails on a converted input.
pub(crate) fn from_set(e: &SetExpr) -> Rel {
    match e {
        SetExpr::All => Rel::All,
        SetExpr::Empty => Rel::Empty,
        SetExpr::Sym(i) => {
            Rel::Plane(u16::try_from(*i).expect("Sym ids originate from u16 planes"))
        }
        SetExpr::Res(Res::I32 { lane, op, t }) => Rel::i32(*lane, cmp_from(*op), *t),
        SetExpr::Res(Res::U32 { lane, eq, v }) => Rel::Atom(Atom::U32 {
            lane: *lane,
            eq: *eq,
            v: *v,
        }),
        SetExpr::And(cs) => Rel::And(cs.iter().map(from_set).collect()),
        SetExpr::Or(cs) => Rel::Or(cs.iter().map(from_set).collect()),
        SetExpr::Not(c) => Rel::Not(Box::new(from_set(c))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// FAILS IF: the conversion is lossy. Order of variants and fields is
    /// shared with the contract, so sorting agrees on both sides.
    #[test]
    fn conversion_round_trips() {
        let rel = Rel::Or(vec![
            Rel::All,
            Rel::Empty,
            Rel::Plane(u16::MAX),
            Rel::i32(3, Cmp::Le, -7),
            Rel::ne_u32(1, 9),
            Rel::And(vec![Rel::eq_u32(0, 2), Rel::Not(Box::new(Rel::Plane(4)))]),
        ]);
        assert_eq!(from_set(&to_set(&rel)), rel);
    }
}
