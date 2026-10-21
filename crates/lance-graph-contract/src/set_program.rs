//! A batch of selections over one population with its shared parts named
//! once (D-RU-2 / U2, `.claude/plans/d-relalg-unified-v1.md` §5.1).
//!
//! A [`SetProgram`] holds `lets` and `queries`, each a [`SetExpr`]. A let is
//! the definition of one [`SetExpr::Sym`] id: `Sym(resident + k)` means
//! `lets[k]`. Ids below `resident` name sets the caller already holds (an
//! executor's resident planes). The meaning of every query is the query with
//! its let references replaced by their definitions ([`SetProgram::inline`]).
//!
//! # What this buys an executor
//!
//! An executor may compute each let once, keep the result, and let every
//! query read it. The alternative, sharing a node inside one expression,
//! would turn a fold's tree into a DAG; a fold that releases consumed
//! operands (relalg's `call_owned`) needs every operand used once, so the
//! shared value is kept outside the fold instead. Each fold stays a tree.
//!
//! # Rules
//!
//! - **Order.** A let may refer only to resident ids and to earlier lets;
//!   a query may refer to any let. So no let depends on itself and an
//!   executor can evaluate the lets in order ([`SetProgram::validate`]).
//! - **Immutability.** A let's value is fixed for the program. Nothing
//!   assigns to an id twice.
//! - **Version.** A program describes one table at one version
//!   ([`SetProgram::version`]). An executor must refuse to evaluate it
//!   against another version; reusing a let's value across versions would
//!   answer from stale rows.
//!
//! # Finding the shared parts
//!
//! [`share`] normalizes every query ([`normalize`]) and lifts into a let any
//! set of at least two members that two or more junctions of one kind hold
//! (`a ∧ b` in `a ∧ b ∧ c` and `a ∧ b ∧ d`; an identical junction is the
//! case where the set is all its members). Normalization flattens nested
//! conjunctions, so a shared `x ∧ y` written inside two rules is found only
//! as such a common subset, never as a node. The part with the most leaves
//! goes first, so a shared expression is not split into smaller lets that
//! only it uses.
//!
//! Zero dependencies.

use std::collections::{BTreeMap, BTreeSet};

use crate::set_algebra::{normalize, SetExpr};

/// Selections over one table at one version, with shared parts as lets.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SetProgram {
    /// The table version every let and query is described against.
    pub version: u64,
    /// Ids below this are resident sets the caller supplies.
    pub resident: u32,
    /// `lets[k]` defines `Sym(resident + k)`.
    pub lets: Vec<SetExpr>,
    /// The selections the program answers.
    pub queries: Vec<SetExpr>,
}

/// Why a program is malformed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProgramError {
    /// `lets[at]` names `sym`, which is itself or a later let.
    ForwardReference { at: usize, sym: u32 },
    /// A let or query names an id at or past `resident + lets.len()`.
    UnknownSym { sym: u32 },
}

impl SetProgram {
    /// The id `lets[k]` defines.
    pub fn let_sym(&self, k: usize) -> u32 {
        self.resident + k as u32
    }

    /// Checks that every reference is defined and that lets refer only
    /// backwards.
    pub fn validate(&self) -> Result<(), ProgramError> {
        let bound = self.resident + self.lets.len() as u32;
        for (k, l) in self.lets.iter().enumerate() {
            let own = self.let_sym(k);
            for sym in syms(l) {
                if sym >= bound {
                    return Err(ProgramError::UnknownSym { sym });
                }
                if sym >= own {
                    return Err(ProgramError::ForwardReference { at: k, sym });
                }
            }
        }
        for q in &self.queries {
            if let Some(sym) = syms(q).into_iter().find(|&s| s >= bound) {
                return Err(ProgramError::UnknownSym { sym });
            }
        }
        Ok(())
    }

    /// `e` with every let reference replaced by its definition, recursively:
    /// what `e` means without the lets. Resident ids stay.
    pub fn inline(&self, e: &SetExpr) -> SetExpr {
        match e {
            SetExpr::Sym(s) if *s >= self.resident => {
                let k = (*s - self.resident) as usize;
                match self.lets.get(k) {
                    Some(def) => self.inline(def),
                    None => e.clone(),
                }
            }
            SetExpr::All | SetExpr::Empty | SetExpr::Sym(_) | SetExpr::Res(_) => e.clone(),
            SetExpr::And(cs) => SetExpr::And(cs.iter().map(|c| self.inline(c)).collect()),
            SetExpr::Or(cs) => SetExpr::Or(cs.iter().map(|c| self.inline(c)).collect()),
            SetExpr::Not(c) => SetExpr::negated(self.inline(c)),
        }
    }
}

/// Every `Sym` id `e` names, in visiting order.
fn syms(e: &SetExpr) -> Vec<u32> {
    let mut out = Vec::new();
    fn walk(e: &SetExpr, out: &mut Vec<u32>) {
        match e {
            SetExpr::Sym(s) => out.push(*s),
            SetExpr::And(cs) | SetExpr::Or(cs) => cs.iter().for_each(|c| walk(c, out)),
            SetExpr::Not(c) => walk(c, out),
            _ => {}
        }
    }
    walk(e, &mut out);
    out
}

/// Which junction a node is: `true` for `And`, `false` for `Or`.
fn kind(e: &SetExpr) -> Option<(bool, &[SetExpr])> {
    match e {
        SetExpr::And(cs) => Some((true, cs)),
        SetExpr::Or(cs) => Some((false, cs)),
        _ => None,
    }
}

fn junction(and: bool, members: Vec<SetExpr>) -> SetExpr {
    if and {
        SetExpr::And(members)
    } else {
        SetExpr::Or(members)
    }
}

/// Every junction node in `e`, nested ones included, as its kind and member
/// set.
fn junctions(e: &SetExpr, out: &mut Vec<(bool, BTreeSet<SetExpr>)>) {
    if let Some((and, cs)) = kind(e) {
        out.push((and, cs.iter().cloned().collect()));
    }
    match e {
        SetExpr::And(cs) | SetExpr::Or(cs) => cs.iter().for_each(|c| junctions(c, out)),
        SetExpr::Not(c) => junctions(c, out),
        _ => {}
    }
}

/// `e` with the members `part` taken out of every junction of the same kind
/// that holds all of them, and `sym` put in their place. A junction whose
/// members are exactly `part` becomes `sym`.
fn replace(e: &SetExpr, and: bool, part: &BTreeSet<SetExpr>, sym: &SetExpr) -> SetExpr {
    let rec = |cs: &[SetExpr]| -> Vec<SetExpr> {
        cs.iter().map(|c| replace(c, and, part, sym)).collect()
    };
    match e {
        SetExpr::And(cs) | SetExpr::Or(cs) => {
            let this_and = matches!(e, SetExpr::And(_));
            if this_and == and && part.iter().all(|m| cs.contains(m)) {
                let rest: Vec<SetExpr> = cs.iter().filter(|c| !part.contains(c)).cloned().collect();
                if rest.is_empty() {
                    return sym.clone();
                }
                let mut members = rec(&rest);
                members.push(sym.clone());
                return junction(and, members);
            }
            junction(this_and, rec(cs))
        }
        SetExpr::Not(c) => SetExpr::negated(replace(c, and, part, sym)),
        _ => e.clone(),
    }
}

/// `e` with each id renamed through `rename` (ids not in it stay).
fn rename_syms(e: &SetExpr, rename: &BTreeMap<u32, u32>) -> SetExpr {
    match e {
        SetExpr::Sym(s) => SetExpr::Sym(*rename.get(s).unwrap_or(s)),
        SetExpr::And(cs) => SetExpr::And(cs.iter().map(|c| rename_syms(c, rename)).collect()),
        SetExpr::Or(cs) => SetExpr::Or(cs.iter().map(|c| rename_syms(c, rename)).collect()),
        SetExpr::Not(c) => SetExpr::negated(rename_syms(c, rename)),
        _ => e.clone(),
    }
}

/// The next part to lift: among the member sets two junctions of one kind
/// have in common (at least two members), the one with the most leaves that
/// at least two junctions hold; ties go to more holders, then to the
/// expression order.
fn next_part(nodes: &[(bool, BTreeSet<SetExpr>)]) -> Option<(bool, BTreeSet<SetExpr>)> {
    let mut parts: BTreeSet<(bool, BTreeSet<SetExpr>)> = BTreeSet::new();
    for (i, (ka, ma)) in nodes.iter().enumerate() {
        for (kb, mb) in &nodes[i + 1..] {
            if ka != kb {
                continue;
            }
            let common: BTreeSet<SetExpr> = ma.intersection(mb).cloned().collect();
            if common.len() >= 2 {
                parts.insert((*ka, common));
            }
        }
    }
    parts
        .into_iter()
        .map(|(and, part)| {
            let holders = nodes
                .iter()
                .filter(|(k, m)| *k == and && part.is_subset(m))
                .count();
            let leaves: usize = part.iter().map(SetExpr::leaves).sum();
            (leaves, holders, and, part)
        })
        .max_by(|a, b| {
            (a.0, a.1)
                .cmp(&(b.0, b.1))
                .then_with(|| (&b.2, &b.3).cmp(&(&a.2, &a.3)))
        })
        .map(|(_, _, and, part)| (and, part))
}

/// A program for `queries` (over `resident` caller-held sets, at `version`)
/// in which every part two junctions share after normalization is a let.
///
/// The returned queries mean what the given ones mean
/// ([`SetProgram::inline`] of each, normalized, equals the given query
/// normalized), and the lets satisfy [`SetProgram::validate`].
pub fn share(version: u64, resident: u32, queries: &[SetExpr]) -> SetProgram {
    let mut qs: Vec<SetExpr> = queries.iter().map(normalize).collect();
    // Lets in extraction order, each with the id it was given then.
    let mut lets: Vec<(u32, SetExpr)> = Vec::new();
    // A cap, not a count: a round's part is held by two or more junctions
    // and afterwards co-occurs only inside its let. If the cap stops the
    // loop early the program is still correct; it just shares less.
    let budget: usize = qs.iter().map(SetExpr::leaves).sum();
    for _ in 0..budget {
        let mut nodes = Vec::new();
        for e in qs.iter().chain(lets.iter().map(|(_, l)| l)) {
            junctions(e, &mut nodes);
        }
        let Some((and, part)) = next_part(&nodes) else {
            break;
        };
        let id = resident + lets.len() as u32;
        let sym = SetExpr::Sym(id);
        for q in &mut qs {
            *q = replace(q, and, &part, &sym);
        }
        for (_, l) in &mut lets {
            *l = replace(l, and, &part, &sym);
        }
        lets.push((id, junction(and, part.into_iter().collect())));
    }
    // A let extracted later can occur inside one extracted earlier, so order
    // the lets by dependency and renumber them densely.
    let order = dependency_order(&lets);
    let rename: BTreeMap<u32, u32> = order
        .iter()
        .enumerate()
        .map(|(new, &old)| (lets[old].0, resident + new as u32))
        .collect();
    SetProgram {
        version,
        resident,
        lets: order
            .iter()
            .map(|&i| normalize(&rename_syms(&lets[i].1, &rename)))
            .collect(),
        queries: qs
            .iter()
            .map(|q| normalize(&rename_syms(q, &rename)))
            .collect(),
    }
}

/// Indices into `lets` such that every let comes after the lets it names.
/// The references form a DAG: a let names only lets replaced inside it, and
/// replacement never puts an id inside its own definition.
fn dependency_order(lets: &[(u32, SetExpr)]) -> Vec<usize> {
    let index: BTreeMap<u32, usize> = lets
        .iter()
        .enumerate()
        .map(|(i, (id, _))| (*id, i))
        .collect();
    let mut placed = vec![false; lets.len()];
    let mut order = Vec::with_capacity(lets.len());
    fn visit(
        i: usize,
        lets: &[(u32, SetExpr)],
        index: &BTreeMap<u32, usize>,
        placed: &mut [bool],
        order: &mut Vec<usize>,
    ) {
        if placed[i] {
            return;
        }
        placed[i] = true;
        for s in syms(&lets[i].1) {
            if let Some(&j) = index.get(&s) {
                visit(j, lets, index, placed, order);
            }
        }
        order.push(i);
    }
    for i in 0..lets.len() {
        visit(i, lets, &index, &mut placed, &mut order);
    }
    order
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::set_algebra::Cmp;

    fn a(t: i32) -> SetExpr {
        SetExpr::i32(0, Cmp::Lt, t)
    }
    fn b(v: u32) -> SetExpr {
        SetExpr::eq_u32(1, v)
    }
    fn c(t: i32) -> SetExpr {
        SetExpr::i32(2, Cmp::Gt, t)
    }

    /// The shared junction of the tests: `(a<10 ∨ b=3) ∧ c>5`.
    fn shared() -> SetExpr {
        SetExpr::And(vec![SetExpr::Or(vec![a(10), b(3)]), c(5)])
    }

    fn meaning(p: &SetProgram, q: &SetExpr) -> SetExpr {
        normalize(&p.inline(q))
    }

    /// FAILS IF: a repeated junction is not lifted, or a lifted one changes
    /// what a query means.
    #[test]
    fn a_repeated_junction_becomes_one_let_and_meanings_are_kept() {
        let qs = vec![
            SetExpr::And(vec![shared(), b(7)]),
            SetExpr::Or(vec![shared(), a(-4)]),
            SetExpr::And(vec![a(0), c(1)]),
        ];
        let p = share(9, 2, &qs);
        assert_eq!(p.validate(), Ok(()));
        assert_eq!(p.lets, vec![normalize(&shared())]);
        for (q, orig) in p.queries.iter().zip(&qs) {
            assert_eq!(meaning(&p, q), normalize(orig));
        }
        // The two queries that share it name the let instead of repeating it.
        assert!(syms(&p.queries[0]).contains(&2));
        assert!(syms(&p.queries[1]).contains(&2));
        assert!(syms(&p.queries[2]).is_empty());
    }

    /// FAILS IF: the share pass lifts a junction that occurs once.
    /// Anti-vacuity: the queries have junctions; none repeats.
    #[test]
    fn nothing_is_lifted_when_nothing_repeats() {
        let qs = vec![
            SetExpr::And(vec![a(1), b(2)]),
            SetExpr::Or(vec![a(3), c(4)]),
        ];
        let p = share(1, 0, &qs);
        assert!(p.lets.is_empty());
        assert_eq!(p.queries, qs.iter().map(normalize).collect::<Vec<_>>());
    }

    /// FAILS IF: the largest repeated junction is split into smaller lets.
    /// Its inner `Or` repeats too, but only inside the shared junction, so
    /// one let suffices.
    #[test]
    fn the_largest_repeated_junction_is_lifted_whole() {
        let qs = vec![
            SetExpr::And(vec![shared(), b(7)]),
            SetExpr::And(vec![shared(), b(8)]),
        ];
        let p = share(1, 0, &qs);
        assert_eq!(p.lets.len(), 1, "{:?}", p.lets);
        assert_eq!(p.lets[0], normalize(&shared()));
    }

    /// FAILS IF: a let that a larger let uses is placed after it, or ids are
    /// not renumbered. The inner `Or` also occurs alone in a third query, so
    /// both are lets and the larger one names the smaller.
    #[test]
    fn nested_lets_are_ordered_by_dependency() {
        let inner = SetExpr::Or(vec![a(10), b(3)]);
        let qs = vec![
            SetExpr::And(vec![shared(), b(7)]),
            SetExpr::And(vec![shared(), b(8)]),
            SetExpr::And(vec![inner.clone(), c(99)]),
            SetExpr::And(vec![inner.clone(), b(5)]),
        ];
        let p = share(1, 0, &qs);
        assert_eq!(p.validate(), Ok(()));
        assert_eq!(p.lets.len(), 2, "{:?}", p.lets);
        assert_eq!(p.lets[0], normalize(&inner));
        assert!(syms(&p.lets[1]).contains(&0), "{:?}", p.lets[1]);
        for (q, orig) in p.queries.iter().zip(&qs) {
            assert_eq!(meaning(&p, q), normalize(orig));
        }
    }

    /// FAILS IF: two queries that differ only in how they were written are
    /// not recognised as one shared junction.
    #[test]
    fn sharing_is_judged_after_normalization() {
        let x = SetExpr::And(vec![SetExpr::Or(vec![a(10), b(3)]), c(5)]);
        let y = SetExpr::And(vec![c(5), SetExpr::Or(vec![b(3), a(10)])]);
        let p = share(1, 0, &[x, y]);
        assert_eq!(p.lets.len(), 1);
        assert_eq!(p.queries, vec![SetExpr::Sym(0), SetExpr::Sym(0)]);
    }

    /// FAILS IF: a forward or unknown reference passes validation.
    #[test]
    fn validation_refuses_forward_and_unknown_references() {
        let fwd = SetProgram {
            version: 1,
            resident: 1,
            lets: vec![SetExpr::And(vec![SetExpr::Sym(2), a(1)]), b(2)],
            queries: vec![SetExpr::Sym(1)],
        };
        assert_eq!(
            fwd.validate(),
            Err(ProgramError::ForwardReference { at: 0, sym: 2 })
        );
        let own = SetProgram {
            lets: vec![SetExpr::Or(vec![SetExpr::Sym(1), a(1)])],
            queries: vec![],
            ..fwd.clone()
        };
        assert_eq!(
            own.validate(),
            Err(ProgramError::ForwardReference { at: 0, sym: 1 })
        );
        let unknown = SetProgram {
            lets: vec![],
            queries: vec![SetExpr::Sym(1)],
            ..fwd
        };
        assert_eq!(unknown.validate(), Err(ProgramError::UnknownSym { sym: 1 }));
    }

    /// FAILS IF: a set of conditions two conjunctions have in common is not
    /// shared because neither conjunction holds it as a node.
    #[test]
    fn a_common_subset_of_two_conjunctions_is_shared() {
        let qs = vec![
            SetExpr::And(vec![a(1), b(2), c(3)]),
            SetExpr::And(vec![a(1), b(2), c(4)]),
        ];
        let p = share(1, 0, &qs);
        assert_eq!(p.lets, vec![normalize(&SetExpr::And(vec![a(1), b(2)]))]);
        for (q, orig) in p.queries.iter().zip(&qs) {
            assert_eq!(meaning(&p, q), normalize(orig));
            assert_eq!(q.leaves(), 2, "{q:?}");
        }
    }

    /// FAILS IF: an `And` part is taken out of an `Or` that holds the same
    /// members. `a ∨ b` and `a ∧ b` share members but not meaning.
    #[test]
    fn parts_are_shared_only_between_junctions_of_one_kind() {
        let qs = vec![
            SetExpr::And(vec![a(1), b(2), c(3)]),
            SetExpr::Or(vec![a(1), b(2), c(4)]),
        ];
        let p = share(1, 0, &qs);
        assert!(p.lets.is_empty(), "{:?}", p.lets);
    }

    /// FAILS IF: resident ids are treated as lets. `Sym(0)` is resident here.
    #[test]
    fn resident_ids_are_not_lets() {
        let r = SetExpr::Sym(0);
        let qs = vec![
            SetExpr::And(vec![r.clone(), a(1), b(2)]),
            SetExpr::And(vec![r.clone(), a(1), b(2)]),
        ];
        let p = share(1, 1, &qs);
        assert_eq!(p.lets.len(), 1);
        assert_eq!(p.queries, vec![SetExpr::Sym(1), SetExpr::Sym(1)]);
        assert_eq!(p.inline(&SetExpr::Sym(0)), SetExpr::Sym(0));
        assert_eq!(p.validate(), Ok(()));
    }
}
