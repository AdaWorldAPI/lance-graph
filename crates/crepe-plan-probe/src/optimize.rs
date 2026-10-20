//! The cold rewrites that run before anything reaches relalg. Each one is a
//! proven equivalence under the semantics it is gated on; none reads a row.
//!
//! 1. **Binding propagation** (the adornment step of Soufflé's magic-set
//!    transformation, without the magic predicates: the spike has no
//!    recursion, so the demand never has to travel through a derived
//!    relation). A query that binds head positions is unified with each rule
//!    head. A head constant that differs from the bound value makes the rule
//!    unreachable; a head variable is replaced by the bound value everywhere
//!    in the rule, so the constant reaches the body atoms.
//! 2. **Duplicate atom** (`A ∩ A = A`). Only where the answer is a set or an
//!    existence: under bag semantics a repeated atom multiplies every
//!    derivation, so removing it changes the counts.
//! 3. **Duplicate rule** (`A ∪ A = A`). Same gate, same reason: two equal
//!    rules derive every answer twice.
//!
//! relalg's own rewrites (flattening, `All`/`Empty`, duplicate planes,
//! absorption, intervals) run later on the plane expression; nothing here
//! repeats them.

use mork_pathmap_probe::Semantics;

use crate::logic::{Atom, RelId, Rule, Term, VarId};

/// What the caller asks for.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ResultKind {
    /// Whether any answer exists.
    Exists,
    /// How many answers, counted as a set or as a bag.
    Count(Semantics),
    /// The answers themselves, as a set or as a bag.
    Bindings(Semantics),
}

impl ResultKind {
    /// True when duplicate derivations cannot change the answer.
    pub fn duplicate_insensitive(self) -> bool {
        matches!(
            self,
            ResultKind::Exists
                | ResultKind::Count(Semantics::Set)
                | ResultKind::Bindings(Semantics::Set)
        )
    }
}

/// A query: the goal relation and, per head position, the value it is bound
/// to (or `None`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Demand {
    pub goal: RelId,
    pub bound: Vec<Option<u64>>,
}

/// Which rewrites run. Turning one off may change the plan shape and the cost,
/// never the answer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Options {
    pub propagate: bool,
    pub dedup: bool,
}

impl Options {
    pub const ALL: Options = Options {
        propagate: true,
        dedup: true,
    };
    pub const NONE: Options = Options {
        propagate: false,
        dedup: false,
    };
}

/// One rewrite that happened, for the plan report.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Rewrite {
    /// The rule's head constant at `pos` contradicts the bound value.
    DroppedRule { rule: usize, pos: usize },
    /// The head variable at `pos` was replaced by the bound value.
    BoundHeadVar { rule: usize, pos: usize, value: u64 },
    /// Body atom `atom` repeated an earlier one.
    DuplicateAtom { rule: usize, atom: usize },
    /// The rule repeated an earlier rule.
    DuplicateRule { rule: usize },
}

/// The rules to evaluate, plus what is still to be applied to the answers.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Optimized {
    /// The rules for the goal, rewritten. Each keeps its source index.
    pub rules: Vec<(usize, Rule)>,
    /// Head positions still to be filtered after evaluation (what propagation
    /// did not push into the rules).
    pub post_filter: Vec<Option<u64>>,
    pub rewrites: Vec<Rewrite>,
}

/// Rewrites the rules for `demand`.
pub fn optimize(rules: &[Rule], demand: &Demand, kind: ResultKind, opts: Options) -> Optimized {
    let mut rewrites = Vec::new();
    let mut out: Vec<(usize, Rule)> = Vec::new();
    for (i, rule) in rules.iter().enumerate() {
        if rule.head.rel != demand.goal {
            continue;
        }
        let mut rule = rule.clone();
        if opts.propagate {
            match specialize(&rule, &demand.bound) {
                None => {
                    let pos = conflict(&rule, &demand.bound).expect("a conflict");
                    rewrites.push(Rewrite::DroppedRule { rule: i, pos });
                    continue;
                }
                Some((r, bound)) => {
                    for (pos, value) in bound {
                        rewrites.push(Rewrite::BoundHeadVar {
                            rule: i,
                            pos,
                            value,
                        });
                    }
                    rule = r;
                }
            }
        }
        if opts.dedup && kind.duplicate_insensitive() {
            let mut kept: Vec<Atom> = Vec::new();
            for (a, atom) in rule.body.iter().enumerate() {
                if kept.contains(atom) {
                    rewrites.push(Rewrite::DuplicateAtom { rule: i, atom: a });
                } else {
                    kept.push(atom.clone());
                }
            }
            rule.body = kept;
        }
        out.push((i, rule));
    }
    if opts.dedup && kind.duplicate_insensitive() {
        let mut seen: Vec<Rule> = Vec::new();
        out.retain(|(i, r)| {
            let c = r.canonical();
            if seen.contains(&c) {
                rewrites.push(Rewrite::DuplicateRule { rule: *i });
                false
            } else {
                seen.push(c);
                true
            }
        });
    }
    let post_filter = if opts.propagate {
        vec![None; demand.bound.len()]
    } else {
        demand.bound.clone()
    };
    Optimized {
        rules: out,
        post_filter,
        rewrites,
    }
}

/// `None` when a head constant contradicts the demand; otherwise the rule with
/// every bound head variable replaced, and the replacements made.
fn specialize(rule: &Rule, bound: &[Option<u64>]) -> Option<(Rule, Vec<(usize, u64)>)> {
    let mut subst: Vec<(VarId, u64)> = Vec::new();
    let mut made = Vec::new();
    for (pos, (t, b)) in rule.head.terms.iter().zip(bound).enumerate() {
        let Some(d) = *b else { continue };
        match *t {
            Term::Const(c) if c != d => return None,
            Term::Const(_) => {}
            Term::Var(v) => match subst.iter().find(|(w, _)| *w == v) {
                // `Answer(x, x)` asked as `(1, 2)`: the head cannot match.
                Some(&(_, prev)) if prev != d => return None,
                Some(_) => {}
                None => {
                    subst.push((v, d));
                    made.push((pos, d));
                }
            },
        }
    }
    let apply = |a: &Atom| Atom {
        rel: a.rel,
        terms: a
            .terms
            .iter()
            .map(|t| match *t {
                Term::Var(v) => subst
                    .iter()
                    .find(|(w, _)| *w == v)
                    .map_or(*t, |&(_, d)| Term::Const(d)),
                c => c,
            })
            .collect(),
    };
    Some((
        Rule {
            head: apply(&rule.head),
            body: rule.body.iter().map(apply).collect(),
        },
        made,
    ))
}

fn conflict(rule: &Rule, bound: &[Option<u64>]) -> Option<usize> {
    let mut seen: Vec<(VarId, u64)> = Vec::new();
    for (pos, (t, b)) in rule.head.terms.iter().zip(bound).enumerate() {
        let Some(d) = *b else { continue };
        match *t {
            Term::Const(c) if c != d => return Some(pos),
            Term::Var(v) => match seen.iter().find(|(w, _)| *w == v) {
                Some(&(_, prev)) if prev != d => return Some(pos),
                Some(_) => {}
                None => seen.push((v, d)),
            },
            _ => {}
        }
    }
    None
}
