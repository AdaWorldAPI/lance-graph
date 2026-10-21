//! Single-table set algebra: what a selection MEANS, independent of how any
//! executor stores or evaluates it (D-CMR-1 / D-RA-2, `ADR-2026-10-20-SIX` §2).
//!
//! ```text
//!                    lance-graph-contract::set_algebra
//!            (SetExpr, Res, Runs, normalize: the one implementation)
//!                 ▲                     ▲                    ▲
//!            relalg (lowers         Quack (later)        crepe / R2IL (later)
//!            to mask-risc)
//! ```
//!
//! # What lives here, and what does not
//!
//! - [`SetExpr`]: a description of one population's subset. A [`SetExpr::Sym`]
//!   is an opaque two-valued set (an executor may bind it to a plane, a
//!   bitmap, anything); a [`Res`] is a residual predicate on one lane. Neither
//!   names a storage layout. Bitsets are consumers of this meaning, not part
//!   of it.
//! - [`normalize`]: the rewrite to an equivalent, cheaper description. It
//!   reads no row and builds nothing.
//! - [`Runs`]: sorted, disjoint, half-open row intervals, closed under `∩`,
//!   `∪`, `∖`, and complement against an EXPLICIT universe.
//!
//! Not here: joins, gathers, positional addressing across tables, recursion,
//! lowering, execution. A consumer owns those.
//!
//! # Rewrites (all run to a fixpoint)
//!
//! 1. Negation is pushed to the leaves (De Morgan); a negated comparison
//!    becomes the opposite comparison.
//! 2. `And` / `Or` are flattened (associativity), so the depth a consumer
//!    admits is measured on normalized structure, never parser nesting;
//!    `All` / `Empty` are absorbed or dominate; duplicates are removed
//!    (idempotence); `p ∧ ¬p` is `Empty`, `p ∨ ¬p` is `All`.
//! 3. In an `And`, every comparison on one `I32` lane becomes one interval
//!    (plus the excluded values inside it); an empty interval makes the whole
//!    conjunction `Empty`. Equalities on one `U32` lane agree or contradict.
//! 4. In an `Or`, intervals on one `I32` lane that overlap or touch merge; a
//!    union covering the whole lane is `All`.
//! 5. Absorption, matched against FLATTENED sibling sets: in an `Or`, a dual
//!    child `And(ms)` is dropped when some member `m` is already covered by
//!    the siblings (`m` is a sibling, or `m` is an `Or` whose members are all
//!    siblings), or when a sibling `And(ks)` has every `k` among `ms`
//!    (`x ∨ (x ∧ y)` with `x` itself a conjunction); dually in an `And`.
//!    This closes
//!    ISS-RELALG-ABSORPTION-AFTER-DE-MORGAN: `¬(x ∧ (x ∨ y))` with
//!    `x = P1 ∧ P4` is `¬P1 ∨ ¬P4`, not the wider form De Morgan emits.
//!
//! No rewrite distributes, so normalization never grows an expression
//! exponentially to remove nesting.
//!
//! An interval is always emitted as `<` / `>` bounds or one `==`.
//!
//! Zero dependencies.

use std::collections::{BTreeMap, BTreeSet};

/// An ordered comparison against a constant.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Cmp {
    Lt,
    Le,
    Gt,
    Ge,
    Eq,
    Ne,
}

/// A residual predicate on one lane of the table: opaque to interval
/// reasoning except for what its own lane type licenses.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Res {
    /// `lane op t` over signed 32-bit values.
    I32 { lane: u16, op: Cmp, t: i32 },
    /// `lane == v` (`eq`) or `lane != v` (`!eq`) over unsigned 32-bit codes.
    U32 { lane: u16, eq: bool, v: u32 },
}

impl Res {
    /// The complement predicate (two-valued: a row either satisfies it or
    /// not).
    pub fn negate(self) -> Res {
        match self {
            Res::I32 { lane, op, t } => Res::I32 {
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
            Res::U32 { lane, eq, v } => Res::U32 { lane, eq: !eq, v },
        }
    }
}

/// A description of a subset of one population. The variant order is part
/// of the contract: [`normalize`] sorts siblings by it.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum SetExpr {
    /// The whole universe.
    All,
    /// The empty set.
    Empty,
    /// An opaque two-valued set, identified by number. What carries it is the
    /// consumer's business.
    Sym(u32),
    /// A residual predicate on one lane.
    Res(Res),
    And(Vec<SetExpr>),
    Or(Vec<SetExpr>),
    Not(Box<SetExpr>),
}

impl SetExpr {
    pub fn i32(lane: u16, op: Cmp, t: i32) -> SetExpr {
        SetExpr::Res(Res::I32 { lane, op, t })
    }

    pub fn eq_u32(lane: u16, v: u32) -> SetExpr {
        SetExpr::Res(Res::U32 { lane, eq: true, v })
    }

    pub fn ne_u32(lane: u16, v: u32) -> SetExpr {
        SetExpr::Res(Res::U32 { lane, eq: false, v })
    }

    pub fn negated(e: SetExpr) -> SetExpr {
        SetExpr::Not(Box::new(e))
    }

    /// Nesting depth of junctions (`And` / `Or` / `Not`); a leaf is 0. After
    /// [`normalize`] this is the depth of the flattened structure.
    pub fn depth(&self) -> usize {
        match self {
            SetExpr::And(cs) | SetExpr::Or(cs) => {
                1 + cs.iter().map(SetExpr::depth).max().unwrap_or(0)
            }
            SetExpr::Not(c) => 1 + c.depth(),
            _ => 0,
        }
    }

    /// Number of leaves (`Sym` and `Res`).
    pub fn leaves(&self) -> usize {
        match self {
            SetExpr::Sym(_) | SetExpr::Res(_) => 1,
            SetExpr::And(cs) | SetExpr::Or(cs) => cs.iter().map(SetExpr::leaves).sum(),
            SetExpr::Not(c) => c.leaves(),
            _ => 0,
        }
    }
}

const LO: i64 = i32::MIN as i64;
const HI: i64 = i32::MAX as i64;

/// Rewrite `e` to an equivalent description, to a fixpoint (at most 64
/// rounds; every rewrite shrinks or keeps the expression).
pub fn normalize(e: &SetExpr) -> SetExpr {
    let mut cur = nnf(e, false);
    for _ in 0..64 {
        let next = simplify(&cur);
        if next == cur {
            return next;
        }
        cur = next;
    }
    cur
}

/// Negation normal form: `Not` survives only around a `Sym`.
fn nnf(e: &SetExpr, neg: bool) -> SetExpr {
    match e {
        SetExpr::All if neg => SetExpr::Empty,
        SetExpr::Empty if neg => SetExpr::All,
        SetExpr::Sym(_) if neg => SetExpr::negated(e.clone()),
        SetExpr::Res(r) if neg => SetExpr::Res(r.negate()),
        SetExpr::All | SetExpr::Empty | SetExpr::Sym(_) | SetExpr::Res(_) => e.clone(),
        SetExpr::Not(c) => nnf(c, !neg),
        SetExpr::And(cs) | SetExpr::Or(cs) => {
            let cs = cs.iter().map(|c| nnf(c, neg)).collect();
            match (e, neg) {
                (SetExpr::And(_), false) | (SetExpr::Or(_), true) => SetExpr::And(cs),
                _ => SetExpr::Or(cs),
            }
        }
    }
}

fn simplify(e: &SetExpr) -> SetExpr {
    match e {
        SetExpr::All | SetExpr::Empty | SetExpr::Sym(_) => e.clone(),
        SetExpr::Res(_) => conj(vec![e.clone()]),
        SetExpr::Not(c) => match simplify(c) {
            SetExpr::All => SetExpr::Empty,
            SetExpr::Empty => SetExpr::All,
            SetExpr::Not(x) => *x,
            s => SetExpr::negated(s),
        },
        SetExpr::And(cs) => conj(cs.iter().map(simplify).collect()),
        SetExpr::Or(cs) => disj(cs.iter().map(simplify).collect()),
    }
}

fn complement(e: &SetExpr) -> Option<SetExpr> {
    match e {
        SetExpr::Sym(_) => Some(SetExpr::negated(e.clone())),
        SetExpr::Not(x) => Some((**x).clone()),
        _ => None,
    }
}

/// `[lo, hi]` as `<` / `>` bounds or one `==`; empty for the whole lane.
fn emit_interval(lane: u16, lo: i64, hi: i64) -> Vec<SetExpr> {
    if lo == hi {
        return vec![SetExpr::i32(lane, Cmp::Eq, lo as i32)];
    }
    let mut v = Vec::new();
    if hi < HI {
        v.push(SetExpr::i32(lane, Cmp::Lt, (hi + 1) as i32));
    }
    if lo > LO {
        v.push(SetExpr::i32(lane, Cmp::Gt, (lo - 1) as i32));
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
fn conj(children: Vec<SetExpr>) -> SetExpr {
    let mut flat = Vec::new();
    for c in children {
        match c {
            SetExpr::And(cs) => flat.extend(cs),
            SetExpr::Empty => return SetExpr::Empty,
            SetExpr::All => {}
            c => flat.push(c),
        }
    }
    let mut ranges: BTreeMap<u16, (i64, i64, BTreeSet<i64>)> = BTreeMap::new();
    let mut eqs: BTreeMap<u16, (BTreeSet<u32>, BTreeSet<u32>)> = BTreeMap::new();
    let mut rest = Vec::new();
    for c in flat {
        match c {
            SetExpr::Res(Res::I32 { lane, op, t }) => {
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
            SetExpr::Res(Res::U32 { lane, eq, v }) => {
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
            return SetExpr::Empty;
        }
        out.extend(emit_interval(lane, lo, hi));
        if lo + 1 < hi {
            for &v in excl.range(lo + 1..hi) {
                out.push(SetExpr::i32(lane, Cmp::Ne, v as i32));
            }
        }
    }
    for (lane, (is, nots)) in eqs {
        let mut is = is.into_iter();
        match (is.next(), is.next()) {
            (Some(_), Some(_)) => return SetExpr::Empty,
            (Some(v), None) if nots.contains(&v) => return SetExpr::Empty,
            (Some(v), None) => out.push(SetExpr::eq_u32(lane, v)),
            _ => out.extend(nots.into_iter().map(|v| SetExpr::ne_u32(lane, v))),
        }
    }
    out.extend(rest);
    finish(out, true)
}

/// A disjunction of simplified children.
fn disj(children: Vec<SetExpr>) -> SetExpr {
    let mut flat = Vec::new();
    for c in children {
        match c {
            SetExpr::Or(cs) => flat.extend(cs),
            SetExpr::All => return SetExpr::All,
            SetExpr::Empty => {}
            c => flat.push(c),
        }
    }
    let mut ranges: BTreeMap<u16, Vec<(i64, i64)>> = BTreeMap::new();
    let mut excluded: BTreeMap<u16, BTreeSet<i64>> = BTreeMap::new();
    let mut eqs: BTreeMap<u16, (BTreeSet<u32>, BTreeSet<u32>)> = BTreeMap::new();
    let mut rest = Vec::new();
    for c in flat {
        if let Some((lane, lo, hi)) = as_interval(&c) {
            ranges.entry(lane).or_default().push((lo, hi));
            continue;
        }
        match c {
            SetExpr::Res(Res::I32 {
                lane,
                op: Cmp::Ne,
                t,
            }) => {
                excluded.entry(lane).or_default().insert(i64::from(t));
            }
            SetExpr::Res(Res::U32 { lane, eq, v }) => {
                let e = eqs.entry(lane).or_default();
                if eq { &mut e.0 } else { &mut e.1 }.insert(v);
            }
            c => rest.push(c),
        }
    }
    let mut out = Vec::new();
    // `x ≠ a` on a lane: a second `x ≠ b`, or any interval holding `a`,
    // covers the whole lane; otherwise every interval on the lane lies
    // inside `x ≠ a` and is absorbed by it.
    for (lane, ne) in excluded {
        let mut ne = ne.into_iter();
        let (Some(a), None) = (ne.next(), ne.next()) else {
            return SetExpr::All;
        };
        let iv = ranges.remove(&lane).unwrap_or_default();
        if iv.iter().any(|&(lo, hi)| lo <= a && a <= hi) {
            return SetExpr::All;
        }
        out.push(SetExpr::i32(lane, Cmp::Ne, a as i32));
    }
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
                0 => return SetExpr::All,
                1 => out.extend(atoms),
                _ => out.push(finish(atoms, true)),
            }
        }
    }
    for (lane, (is, nots)) in eqs {
        let mut ns = nots.iter();
        match (ns.next(), ns.next()) {
            (Some(_), Some(_)) => return SetExpr::All,
            (Some(a), None) if is.contains(a) => return SetExpr::All,
            (Some(&a), None) => out.push(SetExpr::ne_u32(lane, a)),
            _ => out.extend(is.into_iter().map(|v| SetExpr::eq_u32(lane, v))),
        }
    }
    out.extend(rest);
    finish(out, false)
}

/// The interval a simplified child selects on one lane, if it is exactly one.
fn as_interval(c: &SetExpr) -> Option<(u16, i64, i64)> {
    let one = |r: &SetExpr| match r {
        SetExpr::Res(Res::I32 { lane, op, t }) => interval(*op, *t).map(|(lo, hi)| (*lane, lo, hi)),
        _ => None,
    };
    match c {
        SetExpr::Res(_) => one(c),
        SetExpr::And(cs) if !cs.is_empty() => {
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

/// Whether `m` is covered by `sibs` in a junction of kind `and`: inside an
/// `Or` (`and == false`), `m ⊆ ⋃ sibs`; inside an `And`, `m ⊇ ⋂ sibs`. True
/// when `m` is a sibling, or `m` is the parent's own kind (an `Or` inside an
/// `Or`'s dual child, an `And` inside an `And`'s) with every member a sibling.
fn covered(m: &SetExpr, sibs: &[&SetExpr], and: bool) -> bool {
    if sibs.contains(&m) {
        return true;
    }
    match (m, and) {
        (SetExpr::Or(ks), false) | (SetExpr::And(ks), true) => {
            !ks.is_empty() && ks.iter().all(|k| sibs.contains(&k))
        }
        _ => false,
    }
}

/// Whether sibling `s` absorbs a dual child with members `ms`: inside an `Or`
/// (`and == false`), `s = And(ks)` with every `k` in `ms`, so
/// `And(ms) ⊆ s`; inside an `And`, `s = Or(ks)` with every `k` in `ms`, so
/// `Or(ms) ⊇ s`. This is `x ∨ (x ∧ y) = x` when `x` is itself a junction and
/// flattening has spread it into `ms`.
fn sub_junction(s: &SetExpr, ms: &[SetExpr], and: bool) -> bool {
    match (s, and) {
        (SetExpr::And(ks), false) | (SetExpr::Or(ks), true) => {
            !ks.is_empty() && ks.iter().all(|k| ms.contains(k))
        }
        _ => false,
    }
}

/// Sort, deduplicate, apply complement and absorption, and collapse.
fn finish(mut out: Vec<SetExpr>, and: bool) -> SetExpr {
    out.sort();
    out.dedup();
    let (unit, zero) = if and {
        (SetExpr::All, SetExpr::Empty)
    } else {
        (SetExpr::Empty, SetExpr::All)
    };
    if out
        .iter()
        .any(|c| complement(c).is_some_and(|n| out.binary_search(&n).is_ok()))
    {
        return zero;
    }
    // Absorption against the flattened siblings. Children are dropped one at
    // a time and each test uses only the children still kept, so two dual
    // children can never justify dropping each other.
    let mut i = 0;
    while i < out.len() {
        let drop = match (&out[i], and) {
            (SetExpr::Or(ms), true) | (SetExpr::And(ms), false) => {
                let sibs: Vec<&SetExpr> = out
                    .iter()
                    .enumerate()
                    .filter(|&(j, _)| j != i)
                    .map(|(_, s)| s)
                    .collect();
                ms.iter().any(|m| covered(m, &sibs, and))
                    || sibs.iter().any(|s| sub_junction(s, ms, and))
            }
            _ => false,
        };
        if drop {
            out.remove(i);
        } else {
            i += 1;
        }
    }
    match out.len() {
        0 => unit,
        1 => out.pop().unwrap_or(unit),
        _ if and => SetExpr::And(out),
        _ => SetExpr::Or(out),
    }
}

/// A universe was violated: an interval reached past the population size the
/// caller declared.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct OutOfUniverse {
    pub universe: u64,
    pub end: u64,
}

/// Sorted, disjoint, non-adjacent, non-empty half-open row intervals
/// `[lo, hi)`. Closed under `∩`, `∪`, `∖` at a cost linear in the number of
/// intervals, with no row read.
#[derive(Debug, Clone, Default, PartialEq, Eq, Hash)]
pub struct Runs {
    runs: Vec<(u64, u64)>,
}

impl Runs {
    /// The empty set.
    pub fn empty() -> Runs {
        Runs::default()
    }

    /// `[0, universe)`.
    pub fn full(universe: u64) -> Runs {
        Runs::from_intervals([(0, universe)])
    }

    /// Canonicalize arbitrary intervals: empty ones dropped, overlapping or
    /// touching ones merged.
    pub fn from_intervals(iv: impl IntoIterator<Item = (u64, u64)>) -> Runs {
        let mut v: Vec<(u64, u64)> = iv.into_iter().filter(|&(lo, hi)| lo < hi).collect();
        v.sort_unstable();
        let mut runs: Vec<(u64, u64)> = Vec::with_capacity(v.len());
        for (lo, hi) in v {
            match runs.last_mut() {
                Some(last) if lo <= last.1 => last.1 = last.1.max(hi),
                _ => runs.push((lo, hi)),
            }
        }
        Runs { runs }
    }

    /// The canonical intervals.
    pub fn intervals(&self) -> &[(u64, u64)] {
        &self.runs
    }

    pub fn is_empty(&self) -> bool {
        self.runs.is_empty()
    }

    /// Number of rows.
    pub fn count(&self) -> u64 {
        self.runs.iter().map(|&(lo, hi)| hi - lo).sum()
    }

    pub fn contains(&self, row: u64) -> bool {
        let i = self.runs.partition_point(|&(_, hi)| hi <= row);
        self.runs.get(i).is_some_and(|&(lo, _)| lo <= row)
    }

    /// The exclusive upper end of the last interval, or 0.
    pub fn end(&self) -> u64 {
        self.runs.last().map_or(0, |&(_, hi)| hi)
    }

    /// `self ∪ o`: one merge pass over the two canonical lists, O(n + m).
    pub fn union(&self, o: &Runs) -> Runs {
        let (a, b) = (&self.runs, &o.runs);
        let (mut i, mut j) = (0, 0);
        let mut out: Vec<(u64, u64)> = Vec::with_capacity(a.len() + b.len());
        while i < a.len() || j < b.len() {
            let next = if j >= b.len() || (i < a.len() && a[i].0 <= b[j].0) {
                i += 1;
                a[i - 1]
            } else {
                j += 1;
                b[j - 1]
            };
            match out.last_mut() {
                Some(last) if next.0 <= last.1 => last.1 = last.1.max(next.1),
                _ => out.push(next),
            }
        }
        Runs { runs: out }
    }

    pub fn intersect(&self, o: &Runs) -> Runs {
        let (mut i, mut j) = (0, 0);
        let mut out = Vec::new();
        while i < self.runs.len() && j < o.runs.len() {
            let (a, b) = (self.runs[i], o.runs[j]);
            let (lo, hi) = (a.0.max(b.0), a.1.min(b.1));
            if lo < hi {
                out.push((lo, hi));
            }
            if a.1 < b.1 {
                i += 1;
            } else {
                j += 1;
            }
        }
        Runs { runs: out }
    }

    /// `self ∖ o`. Needs no universe: it is `self ∩ ¬o` restricted to `self`.
    pub fn difference(&self, o: &Runs) -> Runs {
        let mut out = Vec::new();
        let mut j = 0;
        for &(lo, hi) in &self.runs {
            let mut cur = lo;
            while j < o.runs.len() && o.runs[j].1 <= cur {
                j += 1;
            }
            let mut k = j;
            while k < o.runs.len() && o.runs[k].0 < hi {
                let (blo, bhi) = o.runs[k];
                if blo > cur {
                    out.push((cur, blo));
                }
                cur = cur.max(bhi);
                if bhi >= hi {
                    break;
                }
                k += 1;
            }
            if cur < hi {
                out.push((cur, hi));
            }
        }
        Runs { runs: out }
    }

    /// `[0, universe) ∖ self`. Refused when `self` reaches past `universe`:
    /// complement only exists against an explicitly declared population.
    pub fn complement(&self, universe: u64) -> Result<Runs, OutOfUniverse> {
        let end = self.end();
        if end > universe {
            return Err(OutOfUniverse { universe, end });
        }
        Ok(Runs::full(universe).difference(self))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── Independent oracle: materialized bitsets over an explicit universe.
    //    It knows nothing of `normalize`; it evaluates by definition.

    const N: usize = 97; // not a word multiple: exercises the tail
    const SYMS: u32 = 4;

    struct World {
        syms: Vec<Vec<bool>>,
        i32_lanes: [Vec<i32>; 2],
        u32_lanes: [Vec<u32>; 2],
    }

    fn world(seed: u64) -> World {
        let mut s = seed;
        let mut next = move || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            s
        };
        let syms = (0..SYMS)
            .map(|_| (0..N).map(|_| next() % 3 == 0).collect())
            .collect();
        let i32_lanes = [
            (0..N).map(|_| (next() % 11) as i32 - 5).collect(),
            (0..N).map(|_| (next() % 7) as i32).collect(),
        ];
        let u32_lanes = [
            (0..N).map(|_| (next() % 4) as u32).collect(),
            (0..N).map(|_| (next() % 3) as u32).collect(),
        ];
        World {
            syms,
            i32_lanes,
            u32_lanes,
        }
    }

    fn res_holds(w: &World, r: Res, row: usize) -> bool {
        match r {
            Res::I32 { lane, op, t } => {
                let x = w.i32_lanes[lane as usize][row];
                match op {
                    Cmp::Lt => x < t,
                    Cmp::Le => x <= t,
                    Cmp::Gt => x > t,
                    Cmp::Ge => x >= t,
                    Cmp::Eq => x == t,
                    Cmp::Ne => x != t,
                }
            }
            Res::U32 { lane, eq, v } => (w.u32_lanes[lane as usize][row] == v) == eq,
        }
    }

    fn eval(w: &World, e: &SetExpr) -> Vec<bool> {
        match e {
            SetExpr::All => vec![true; N],
            SetExpr::Empty => vec![false; N],
            SetExpr::Sym(i) => w.syms[*i as usize].clone(),
            SetExpr::Res(r) => (0..N).map(|row| res_holds(w, *r, row)).collect(),
            SetExpr::Not(c) => eval(w, c).into_iter().map(|b| !b).collect(),
            SetExpr::And(cs) => cs.iter().fold(vec![true; N], |acc, c| {
                acc.iter().zip(eval(w, c)).map(|(a, b)| *a && b).collect()
            }),
            SetExpr::Or(cs) => cs.iter().fold(vec![false; N], |acc, c| {
                acc.iter().zip(eval(w, c)).map(|(a, b)| *a || b).collect()
            }),
        }
    }

    struct Gen(u64);
    impl Gen {
        fn next(&mut self) -> u64 {
            self.0 = self
                .0
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            self.0 >> 33
        }
        fn leaf(&mut self) -> SetExpr {
            const OPS: [Cmp; 6] = [Cmp::Lt, Cmp::Le, Cmp::Gt, Cmp::Ge, Cmp::Eq, Cmp::Ne];
            match self.next() % 5 {
                0 | 1 => SetExpr::Sym((self.next() % SYMS as u64) as u32),
                2 | 3 => SetExpr::i32(
                    (self.next() % 2) as u16,
                    OPS[(self.next() % 6) as usize],
                    (self.next() % 13) as i32 - 6,
                ),
                _ => SetExpr::Res(Res::U32 {
                    lane: (self.next() % 2) as u16,
                    eq: self.next().is_multiple_of(2),
                    v: (self.next() % 4) as u32,
                }),
            }
        }
        fn expr(&mut self, depth: u32) -> SetExpr {
            if depth == 0 || self.next().is_multiple_of(4) {
                return match self.next() % 12 {
                    0 => SetExpr::All,
                    1 => SetExpr::Empty,
                    _ => self.leaf(),
                };
            }
            match self.next() % 3 {
                0 => SetExpr::negated(self.expr(depth - 1)),
                k => {
                    let n = 1 + self.next() % 4;
                    let cs = (0..n).map(|_| self.expr(depth - 1)).collect();
                    if k == 1 {
                        SetExpr::And(cs)
                    } else {
                        SetExpr::Or(cs)
                    }
                }
            }
        }
    }

    /// FAILS IF: any rewrite changes the selected set. Differential against
    /// materialized bitsets over random nested expressions, incl. `All`,
    /// `Empty` and negations, in several worlds.
    #[test]
    fn normalize_preserves_meaning_against_materialized_bitsets() {
        let mut g = Gen(0x5EED);
        let mut changed = 0;
        for k in 0..3000 {
            let w = world(1 + k % 7);
            let e = g.expr(5);
            let n = normalize(&e);
            assert_eq!(eval(&w, &n), eval(&w, &e), "{e:?}\n => {n:?}");
            assert_eq!(normalize(&n), n, "not idempotent: {e:?}");
            if n != e {
                changed += 1;
            }
        }
        // Anti-vacuity: the rewrites actually fired on most inputs.
        assert!(changed * 2 > 3000, "only {changed} rewritten");
    }

    /// FAILS IF: normalization ever grows an expression (no distribution).
    #[test]
    fn normalize_never_grows_leaf_count() {
        let mut g = Gen(0xB0B);
        for _ in 0..3000 {
            let e = g.expr(5);
            assert!(normalize(&e).leaves() <= e.leaves(), "{e:?}");
        }
    }

    fn x() -> SetExpr {
        SetExpr::And(vec![SetExpr::Sym(1), SetExpr::Sym(4)])
    }

    /// N17, ISS-RELALG-ABSORPTION-AFTER-DE-MORGAN. FAILS IF: absorption is
    /// matched only against unflattened members, leaving
    /// `¬P1 ∨ ¬P4 ∨ ((¬P1 ∨ ¬P4) ∧ ¬P5 ∧ ¬P7)`.
    #[test]
    fn absorption_after_de_morgan_matches_flattened_siblings() {
        let y = SetExpr::And(vec![SetExpr::Sym(5), SetExpr::Sym(7)]);
        let e = SetExpr::negated(SetExpr::And(vec![x(), SetExpr::Or(vec![x(), y])]));
        let want = SetExpr::Or(vec![
            SetExpr::negated(SetExpr::Sym(1)),
            SetExpr::negated(SetExpr::Sym(4)),
        ]);
        assert_eq!(normalize(&e), want);
    }

    /// Dual of N17. FAILS IF: an `Or` whose members are all conjunction
    /// siblings is not absorbed: `P1 ∧ P4 ∧ ((P1 ∧ P4) ∨ P5)` is `P1 ∧ P4`.
    #[test]
    fn dual_absorption_inside_a_conjunction() {
        let e = SetExpr::And(vec![x(), SetExpr::Or(vec![x(), SetExpr::Sym(5)])]);
        assert_eq!(normalize(&e), x());
    }

    /// FAILS IF: absorption misses `x ∨ (x ∧ y)` when `x` is itself a
    /// conjunction, i.e. one dual child is a sub-conjunction of another:
    /// `¬(x ∧ (x ∨ P1))` with `x = P0 ∨ P2` is `¬P0 ∧ ¬P2`.
    #[test]
    fn absorption_of_a_superset_conjunction() {
        let x = SetExpr::Or(vec![SetExpr::Sym(0), SetExpr::Sym(2)]);
        let e = SetExpr::negated(SetExpr::And(vec![
            x.clone(),
            SetExpr::Or(vec![x.clone(), SetExpr::Sym(1)]),
        ]));
        assert_eq!(normalize(&e), normalize(&SetExpr::negated(x)));
    }

    /// Silence twin. FAILS IF: absorption fires when only SOME members of the
    /// nested junction are siblings: `¬P1 ∨ ((¬P1 ∨ ¬P4) ∧ ¬P5)` stays.
    #[test]
    fn absorption_stays_silent_on_a_partial_cover() {
        let e = SetExpr::Or(vec![
            SetExpr::negated(SetExpr::Sym(1)),
            SetExpr::And(vec![
                SetExpr::Or(vec![
                    SetExpr::negated(SetExpr::Sym(1)),
                    SetExpr::negated(SetExpr::Sym(4)),
                ]),
                SetExpr::negated(SetExpr::Sym(5)),
            ]),
        ]);
        // Same set of children, in sort order (`And` sorts before `Not`).
        let SetExpr::Or(cs) = &e else { unreachable!() };
        assert_eq!(
            normalize(&e),
            SetExpr::Or(vec![cs[1].clone(), cs[0].clone()])
        );
    }

    /// FAILS IF: nesting from the parser survives normalization: the depth a
    /// consumer admits is that of the flattened form.
    #[test]
    fn associative_nesting_flattens_before_depth_is_measured() {
        let mut nested = SetExpr::Sym(0);
        for i in 1..8u32 {
            nested = SetExpr::Or(vec![nested, SetExpr::Sym(i % SYMS + 10 * i)]);
        }
        assert_eq!(nested.depth(), 7);
        let n = normalize(&nested);
        assert_eq!(n.depth(), 1, "{n:?}");
        assert_eq!(n.leaves(), 8);
    }

    #[test]
    fn de_morgan_and_complement_pairs() {
        let a = SetExpr::Sym(0);
        let b = SetExpr::Sym(1);
        let e = SetExpr::negated(SetExpr::Or(vec![a.clone(), b.clone()]));
        assert_eq!(
            normalize(&e),
            SetExpr::And(vec![SetExpr::negated(a.clone()), SetExpr::negated(b)])
        );
        assert_eq!(
            normalize(&SetExpr::And(vec![a.clone(), SetExpr::negated(a.clone())])),
            SetExpr::Empty
        );
        assert_eq!(
            normalize(&SetExpr::Or(vec![a.clone(), SetExpr::negated(a)])),
            SetExpr::All
        );
    }

    // ── Runs: laws against materialized bitsets, with an explicit universe.

    fn bits(r: &Runs, u: u64) -> Vec<bool> {
        (0..u).map(|i| r.contains(i)).collect()
    }

    fn rand_runs(g: &mut Gen, u: u64) -> Runs {
        let n = g.next() % 5;
        Runs::from_intervals((0..n).map(|_| {
            let lo = g.next() % (u + 1);
            let hi = lo + g.next() % (u + 1 - lo + 1);
            (lo, hi.min(u))
        }))
    }

    fn canonical(r: &Runs) -> bool {
        r.intervals().iter().all(|&(lo, hi)| lo < hi)
            && r.intervals().windows(2).all(|w| w[0].1 < w[1].0)
    }

    /// FAILS IF: `∩ ∪ ∖ ¬` disagree with the bitset definition, lose
    /// canonical form, or break De Morgan / distributivity, incl. empty and
    /// full sets and the universe boundary.
    #[test]
    fn runs_obey_the_set_laws_against_bitsets() {
        let mut g = Gen(77);
        for u in [0u64, 1, 2, 63, 64, 65, 100] {
            for _ in 0..300 {
                let (a, b, c) = (
                    rand_runs(&mut g, u),
                    rand_runs(&mut g, u),
                    rand_runs(&mut g, u),
                );
                let (ba, bb) = (bits(&a, u), bits(&b, u));
                let and: Vec<bool> = ba.iter().zip(&bb).map(|(x, y)| *x && *y).collect();
                let or: Vec<bool> = ba.iter().zip(&bb).map(|(x, y)| *x || *y).collect();
                let diff: Vec<bool> = ba.iter().zip(&bb).map(|(x, y)| *x && !*y).collect();
                let not_a: Vec<bool> = ba.iter().map(|x| !x).collect();
                let i = a.intersect(&b);
                let un = a.union(&b);
                let d = a.difference(&b);
                let na = a.complement(u).unwrap();
                for r in [&i, &un, &d, &na] {
                    assert!(canonical(r), "{r:?}");
                }
                assert_eq!(bits(&i, u), and);
                assert_eq!(bits(&un, u), or);
                assert_eq!(bits(&d, u), diff);
                assert_eq!(bits(&na, u), not_a);
                assert_eq!(i.count(), and.iter().filter(|x| **x).count() as u64);
                // De Morgan.
                assert_eq!(
                    un.complement(u).unwrap(),
                    na.intersect(&b.complement(u).unwrap())
                );
                // Distributivity.
                assert_eq!(a.intersect(&b.union(&c)), i.union(&a.intersect(&c)));
                // Identities.
                assert_eq!(a.intersect(&Runs::full(u)), a);
                assert_eq!(a.union(&Runs::empty()), a);
                assert!(a.intersect(&Runs::empty()).is_empty());
                assert_eq!(a.complement(u).unwrap().complement(u).unwrap(), a);
                assert_eq!(a.difference(&b), a.intersect(&b.complement(u).unwrap()));
            }
        }
    }

    /// FAILS IF: complement silently assumes a universe smaller than the set
    /// it complements.
    #[test]
    fn complement_requires_an_explicit_universe_that_covers_the_set() {
        let r = Runs::from_intervals([(3, 10)]);
        assert_eq!(
            r.complement(9),
            Err(OutOfUniverse {
                universe: 9,
                end: 10
            })
        );
        assert_eq!(r.complement(10).unwrap(), Runs::from_intervals([(0, 3)]));
        assert_eq!(Runs::empty().complement(0).unwrap(), Runs::empty());
        assert_eq!(Runs::empty().complement(5).unwrap(), Runs::full(5));
    }
}
