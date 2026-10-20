//! Independent oracle: the resolved, UNOPTIMIZED rules evaluated by nested
//! loops over the raw rows, bag semantics. Shares no code with
//! [`crate::optimize`] or [`crate::native`].

use std::collections::BTreeMap;

use crate::logic::{Atom, RelId, Rule, Term};

/// Bag of head tuples for every rule whose head is `goal`, then filtered by the
/// demand.
pub fn eval(
    rules: &[Rule],
    goal: RelId,
    demand: &[Option<u64>],
    rows: &dyn Fn(RelId) -> Vec<Vec<u64>>,
) -> BTreeMap<Vec<u64>, u64> {
    let mut out = BTreeMap::new();
    for rule in rules.iter().filter(|r| r.head.rel == goal) {
        let mut env: Vec<Option<u64>> = vec![None; 64];
        walk(rule, 0, &mut env, rows, &mut out);
    }
    out.retain(|t, _| t.iter().zip(demand).all(|(v, d)| d.is_none_or(|d| d == *v)));
    out
}

fn walk(
    rule: &Rule,
    i: usize,
    env: &mut Vec<Option<u64>>,
    rows: &dyn Fn(RelId) -> Vec<Vec<u64>>,
    out: &mut BTreeMap<Vec<u64>, u64>,
) {
    if i == rule.body.len() {
        let t = rule
            .head
            .terms
            .iter()
            .map(|t| match *t {
                Term::Const(c) => c,
                Term::Var(v) => env[v.0 as usize].expect("head var bound"),
            })
            .collect();
        *out.entry(t).or_insert(0) += 1;
        return;
    }
    let Atom { rel, terms } = &rule.body[i];
    for row in rows(*rel) {
        let saved = env.clone();
        let ok = terms.iter().zip(&row).all(|(t, &v)| match *t {
            Term::Const(c) => c == v,
            Term::Var(x) => match env[x.0 as usize] {
                Some(b) => b == v,
                None => {
                    env[x.0 as usize] = Some(v);
                    true
                }
            },
        });
        if ok {
            walk(rule, i + 1, env, rows, out);
        }
        *env = saved;
    }
}
