//! Rewrites: each takes a description to an equivalent, cheaper one, before
//! anything runs. None of them reads a row.
//!
//! 1. Negation is pushed to the leaves (De Morgan); a negated comparison
//!    becomes the opposite comparison.
//! 2. `And` / `Or` are flattened; `All` and `Empty` are absorbed or
//!    dominate; duplicates are removed; `p ∧ ¬p` is `Empty`, `p ∨ ¬p` is
//!    `All`.
//! 3. In an `And`, every comparison on one `I32` lane becomes one interval
//!    (plus the excluded values inside it); an empty interval makes the whole
//!    conjunction `Empty`. Equalities on one `U32` lane either agree or
//!    contradict.
//! 4. In an `Or`, intervals on one `I32` lane that overlap or touch are
//!    merged; a union covering the whole lane is `All`.
//! 5. Absorption: `x ∧ (x ∨ y) = x`, `x ∨ (x ∧ y) = x`.
//!
//! The rewrites run to a fixpoint. An interval is always emitted as `<` / `>`
//! bounds or one `==`, the forms `IntSLess` / `IntEqual` lower without a
//! negation pass.

use std::collections::{BTreeMap, BTreeSet};

use crate::ir::{Atom, Cmp, Measure, Rel};
use crate::lower::Answer;

const LO: i64 = i32::MIN as i64;
const HI: i64 = i32::MAX as i64;

/// Rewrite `rel` to an equivalent description, to a fixpoint.
pub fn normalize(rel: &Rel) -> Rel {
    let mut cur = nnf(rel, false);
    for _ in 0..64 {
        let next = simplify(&cur);
        if next == cur {
            return next;
        }
        cur = next;
    }
    cur
}

/// The answer, if the description alone determines it: no row is read and
/// no program runs.
pub fn symbolic(rel: &Rel, m: Measure, n_rows: usize) -> Option<Answer> {
    match (rel, m) {
        (Rel::Empty, Measure::Count) => Some(Answer::Count(0)),
        (Rel::Empty, Measure::Sum(_)) => Some(Answer::Sum(0)),
        (Rel::Empty, Measure::Min(_)) => Some(Answer::Min(None)),
        (Rel::Empty, Measure::Max(_)) => Some(Answer::Max(None)),
        (Rel::All, Measure::Count) => Some(Answer::Count(n_rows as u64)),
        _ => None,
    }
}

fn negate(a: Atom) -> Atom {
    match a {
        Atom::I32 { lane, op, t } => Atom::I32 {
            lane,
            t,
            op: match op {
                Cmp::Lt => Cmp::Ge,
                Cmp::Ge => Cmp::Lt,
                Cmp::Le => Cmp::Gt,
                Cmp::Gt => Cmp::Le,
                Cmp::Eq => Cmp::Ne,
                Cmp::Ne => Cmp::Eq,
            },
        },
        Atom::U32 { lane, eq, v } => Atom::U32 { lane, eq: !eq, v },
    }
}

/// Negation normal form: `Not` survives only around a plane.
fn nnf(rel: &Rel, neg: bool) -> Rel {
    match rel {
        Rel::All if neg => Rel::Empty,
        Rel::Empty if neg => Rel::All,
        Rel::Plane(_) if neg => Rel::Not(Box::new(rel.clone())),
        Rel::Atom(a) if neg => Rel::Atom(negate(*a)),
        Rel::All | Rel::Empty | Rel::Plane(_) | Rel::Atom(_) => rel.clone(),
        Rel::Not(c) => nnf(c, !neg),
        Rel::And(cs) | Rel::Or(cs) => {
            let cs = cs.iter().map(|c| nnf(c, neg)).collect();
            match (rel, neg) {
                (Rel::And(_), false) | (Rel::Or(_), true) => Rel::And(cs),
                _ => Rel::Or(cs),
            }
        }
    }
}

fn simplify(rel: &Rel) -> Rel {
    match rel {
        Rel::All | Rel::Empty | Rel::Plane(_) => rel.clone(),
        Rel::Atom(_) => conj(vec![rel.clone()]),
        Rel::Not(c) => match simplify(c) {
            Rel::All => Rel::Empty,
            Rel::Empty => Rel::All,
            Rel::Not(x) => *x,
            s => Rel::Not(Box::new(s)),
        },
        Rel::And(cs) => conj(cs.iter().map(simplify).collect()),
        Rel::Or(cs) => disj(cs.iter().map(simplify).collect()),
    }
}

fn complement(r: &Rel) -> Option<Rel> {
    match r {
        Rel::Plane(_) => Some(Rel::Not(Box::new(r.clone()))),
        Rel::Not(x) => Some((**x).clone()),
        _ => None,
    }
}

/// `[lo, hi]` as `<` / `>` bounds or one `==`; empty for the whole lane.
fn emit_interval(lane: u16, lo: i64, hi: i64) -> Vec<Rel> {
    if lo == hi {
        return vec![Rel::i32(lane, Cmp::Eq, lo as i32)];
    }
    let mut v = Vec::new();
    if hi < HI {
        v.push(Rel::i32(lane, Cmp::Lt, (hi + 1) as i32));
    }
    if lo > LO {
        v.push(Rel::i32(lane, Cmp::Gt, (lo - 1) as i32));
    }
    v
}

/// The interval an ordered comparison or equality selects; `None` for `!=`.
fn interval(op: Cmp, t: i32) -> Option<(i64, i64)> {
    let t = i64::from(t);
    Some(match op {
        Cmp::Lt => (LO, t - 1),
        Cmp::Le => (LO, t),
        Cmp::Gt => (t + 1, HI),
        Cmp::Ge => (t, HI),
        Cmp::Eq => (t, t),
        Cmp::Ne => return None,
    })
}

/// A conjunction of simplified children.
fn conj(children: Vec<Rel>) -> Rel {
    let mut flat = Vec::new();
    for c in children {
        match c {
            Rel::And(cs) => flat.extend(cs),
            Rel::Empty => return Rel::Empty,
            Rel::All => {}
            c => flat.push(c),
        }
    }
    // Fuse comparisons per lane.
    let mut ranges: BTreeMap<u16, (i64, i64, BTreeSet<i64>)> = BTreeMap::new();
    let mut eqs: BTreeMap<u16, (BTreeSet<u32>, BTreeSet<u32>)> = BTreeMap::new();
    let mut rest = Vec::new();
    for c in flat {
        match c {
            Rel::Atom(Atom::I32 { lane, op, t }) => {
                let e = ranges.entry(lane).or_insert((LO, HI, BTreeSet::new()));
                match interval(op, t) {
                    Some((lo, hi)) => {
                        e.0 = e.0.max(lo);
                        e.1 = e.1.min(hi);
                    }
                    None => {
                        e.2.insert(i64::from(t));
                    }
                }
            }
            Rel::Atom(Atom::U32 { lane, eq, v }) => {
                let e = eqs.entry(lane).or_default();
                if eq { &mut e.0 } else { &mut e.1 }.insert(v);
            }
            c => rest.push(c),
        }
    }
    let mut out = Vec::new();
    for (lane, (mut lo, mut hi, excl)) in ranges {
        while lo <= hi && excl.contains(&lo) {
            lo += 1;
        }
        while lo <= hi && excl.contains(&hi) {
            hi -= 1;
        }
        if lo > hi {
            return Rel::Empty;
        }
        out.extend(emit_interval(lane, lo, hi));
        if lo + 1 < hi {
            for &v in excl.range(lo + 1..hi) {
                out.push(Rel::i32(lane, Cmp::Ne, v as i32));
            }
        }
    }
    for (lane, (is, nots)) in eqs {
        let mut is = is.into_iter();
        match (is.next(), is.next()) {
            (Some(_), Some(_)) => return Rel::Empty,
            (Some(v), None) if nots.contains(&v) => return Rel::Empty,
            (Some(v), None) => out.push(Rel::eq_u32(lane, v)),
            _ => out.extend(nots.into_iter().map(|v| Rel::ne_u32(lane, v))),
        }
    }
    out.extend(rest);
    finish(out, true)
}

/// A disjunction of simplified children.
fn disj(children: Vec<Rel>) -> Rel {
    let mut flat = Vec::new();
    for c in children {
        match c {
            Rel::Or(cs) => flat.extend(cs),
            Rel::All => return Rel::All,
            Rel::Empty => {}
            c => flat.push(c),
        }
    }
    // Intervals per lane: a comparison atom, or an `And` that is exactly the
    // bounds of one interval on one lane.
    let mut ranges: BTreeMap<u16, Vec<(i64, i64)>> = BTreeMap::new();
    let mut eqs: BTreeMap<u16, (BTreeSet<u32>, BTreeSet<u32>)> = BTreeMap::new();
    let mut rest = Vec::new();
    for c in flat {
        if let Some((lane, lo, hi)) = as_interval(&c) {
            ranges.entry(lane).or_default().push((lo, hi));
            continue;
        }
        match c {
            Rel::Atom(Atom::U32 { lane, eq, v }) => {
                let e = eqs.entry(lane).or_default();
                if eq { &mut e.0 } else { &mut e.1 }.insert(v);
            }
            c => rest.push(c),
        }
    }
    let mut out = Vec::new();
    for (lane, mut iv) in ranges {
        iv.sort_unstable();
        let mut merged: Vec<(i64, i64)> = Vec::new();
        for (lo, hi) in iv {
            match merged.last_mut() {
                Some(last) if lo <= last.1 + 1 => last.1 = last.1.max(hi),
                _ => merged.push((lo, hi)),
            }
        }
        for (lo, hi) in merged {
            let atoms = emit_interval(lane, lo, hi);
            match atoms.len() {
                0 => return Rel::All,
                1 => out.extend(atoms),
                _ => out.push(finish(atoms, true)),
            }
        }
    }
    for (lane, (is, nots)) in eqs {
        let mut ns = nots.iter();
        match (ns.next(), ns.next()) {
            // `v ≠ a ∨ v ≠ b` with `a ≠ b` holds everywhere.
            (Some(_), Some(_)) => return Rel::All,
            (Some(a), None) if is.contains(a) => return Rel::All,
            // `v ≠ a ∨ v = b` with `b ≠ a` is `v ≠ a`.
            (Some(&a), None) => out.push(Rel::ne_u32(lane, a)),
            _ => out.extend(is.into_iter().map(|v| Rel::eq_u32(lane, v))),
        }
    }
    out.extend(rest);
    finish(out, false)
}

/// The interval a simplified child selects on one lane, if it is exactly
/// one.
fn as_interval(c: &Rel) -> Option<(u16, i64, i64)> {
    let one = |r: &Rel| match r {
        Rel::Atom(Atom::I32 { lane, op, t }) => interval(*op, *t).map(|(lo, hi)| (*lane, lo, hi)),
        _ => None,
    };
    match c {
        Rel::Atom(_) => one(c),
        Rel::And(cs) if !cs.is_empty() => {
            let mut acc: Option<(u16, i64, i64)> = None;
            for x in cs {
                let (lane, lo, hi) = one(x)?;
                acc = Some(match acc {
                    None => (lane, lo, hi),
                    Some((l, a, b)) if l == lane => (l, a.max(lo), b.min(hi)),
                    Some(_) => return None,
                });
            }
            acc.filter(|&(_, lo, hi)| lo <= hi)
        }
        _ => None,
    }
}

/// Sort, deduplicate, apply complement and absorption, and collapse.
fn finish(mut out: Vec<Rel>, and: bool) -> Rel {
    out.sort();
    out.dedup();
    let (unit, zero) = if and {
        (Rel::All, Rel::Empty)
    } else {
        (Rel::Empty, Rel::All)
    };
    if out
        .iter()
        .any(|c| complement(c).is_some_and(|n| out.binary_search(&n).is_ok()))
    {
        return zero;
    }
    // Absorption: drop a dual child that shares a member with its siblings.
    let set: BTreeSet<Rel> = out.iter().cloned().collect();
    out.retain(|c| match (c, and) {
        (Rel::Or(ms), true) | (Rel::And(ms), false) => !ms.iter().any(|m| set.contains(m)),
        _ => true,
    });
    match out.len() {
        0 => unit,
        1 => out.pop().unwrap_or(unit),
        _ if and => Rel::And(out),
        _ => Rel::Or(out),
    }
}
