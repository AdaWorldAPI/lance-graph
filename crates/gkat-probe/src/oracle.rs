//! The independent oracle: a direct operational interpreter of [`Exp`],
//! sharing no code with [`crate::automaton`].
//!
//! It explores every run of an expression in which the environment picks an
//! arbitrary declared atom at the start and after every action, up to a
//! **stated bound** on the number of actions. Within that bound the
//! exploration is exhaustive: every run with at most `max_actions` actions is
//! visited. Runs that would perform more actions are cut and recorded as cut,
//! not as accepted or rejected.
//!
//! **What this oracle can and cannot show.** Equal accepted sets up to `n`
//! actions is necessary for equivalence, not sufficient: two flows may first
//! differ after `n` actions. A difference it finds, though, is a real
//! difference. So the tests use it in one direction only: whenever the
//! automaton says "equivalent", the oracle must agree at every bound tried;
//! whenever the oracle finds a difference, the automaton must say
//! "different".

use std::collections::{BTreeMap, BTreeSet, HashSet};

use crate::expr::{Action, Atom, Atoms, BExp, Exp};

// Divergence is detected exactly, not by a step budget. Between two actions
// the atom cannot change, so a silent run is determined by its continuation
// stack. Any silent run that never ends re-enters some loop head with a
// stack it has had before, and the set of stacks at loop heads is finite
// (each is a path through the expression). A repeated loop-head stack is
// therefore divergence, and a stretch with no repeat terminates, however
// long it is. (A fixed budget misclassified long terminating stretches;
// Codex review on #1676.)

/// One accepted run: the atoms observed (one more than the actions) and the
/// actions performed.
pub type GuardedString = (Vec<Atom>, Vec<Action>);

/// Everything the bounded exploration saw.
#[derive(Debug, Default)]
pub struct Exploration {
    /// Accepted runs with at most `max_actions` actions.
    pub accepted: BTreeSet<GuardedString>,
    /// Every emitted action with the atom observed immediately before it.
    pub emitted: Vec<(Vec<Atom>, Vec<Action>, Atom, Action)>,
    /// Runs cut at the action bound.
    pub cut: usize,
    /// Runs that loop forever without acting.
    pub diverged: usize,
    /// Runs that reject.
    pub rejected: usize,
}

/// Explores `e` over `atoms` up to `max_actions` actions.
pub fn explore(e: &Exp, atoms: &Atoms, max_actions: usize) -> Exploration {
    let mut out = Exploration::default();
    for a in atoms.iter() {
        run(
            vec![e],
            a,
            &mut vec![a],
            &mut Vec::new(),
            atoms,
            max_actions,
            &mut out,
        );
    }
    out
}

fn run(
    mut stack: Vec<&Exp>,
    atom: Atom,
    seen_atoms: &mut Vec<Atom>,
    actions: &mut Vec<Action>,
    atoms: &Atoms,
    max_actions: usize,
    out: &mut Exploration,
) {
    let mut loop_heads: HashSet<Vec<*const Exp>> = HashSet::new();
    loop {
        let Some(e) = stack.pop() else {
            out.accepted.insert((seen_atoms.clone(), actions.clone()));
            return;
        };
        match e {
            Exp::Fail => {
                out.rejected += 1;
                return;
            }
            Exp::Skip => {}
            Exp::Assert(b) => {
                if !b.eval(atom) {
                    out.rejected += 1;
                    return;
                }
            }
            Exp::Seq(f, g) => {
                stack.push(g);
                stack.push(f);
            }
            Exp::If(b, f, g) => stack.push(if b.eval(atom) { f } else { g }),
            Exp::While(b, body) => {
                if b.eval(atom) {
                    let mut key: Vec<*const Exp> = stack.iter().map(|x| *x as *const Exp).collect();
                    key.push(e as *const Exp);
                    if !loop_heads.insert(key) {
                        out.diverged += 1;
                        return;
                    }
                    stack.push(e);
                    stack.push(body);
                }
            }
            Exp::Do(p) => {
                if actions.len() == max_actions {
                    out.cut += 1;
                    return;
                }
                out.emitted
                    .push((seen_atoms.clone(), actions.clone(), atom, *p));
                actions.push(*p);
                for next in atoms.iter() {
                    seen_atoms.push(next);
                    run(
                        stack.clone(),
                        next,
                        seen_atoms,
                        actions,
                        atoms,
                        max_actions,
                        out,
                    );
                    seen_atoms.pop();
                }
                actions.pop();
                return;
            }
        }
    }
}

/// Whether the oracle sees the same accepted runs for both expressions.
pub fn same_up_to(e: &Exp, f: &Exp, atoms: &Atoms, max_actions: usize) -> bool {
    explore(e, atoms, max_actions).accepted == explore(f, atoms, max_actions).accepted
}

/// Whether some emitted `action` within the bound follows an atom that does
/// not satisfy `authorized`.
pub fn unsafe_within(
    e: &Exp,
    atoms: &Atoms,
    max_actions: usize,
    action: Action,
    authorized: &BExp,
) -> bool {
    explore(e, atoms, max_actions)
        .emitted
        .iter()
        .any(|(_, _, atom, p)| *p == action && !authorized.eval(*atom))
}

/// The GKAT determinacy property (paper §2.2) of a set of guarded strings:
/// two strings that observe the same atoms up to a position must take the
/// same step there: the same action, or both accept.
pub fn deterministic(strings: &BTreeSet<GuardedString>) -> bool {
    // Map each observed atom prefix to what follows it: an action, or `None`
    // for acceptance. A prefix that both accepts and continues is as
    // nondeterministic as one followed by two different actions (Codex
    // review on #1676).
    let mut next: BTreeMap<Vec<Atom>, Option<Action>> = BTreeMap::new();
    for (atoms, actions) in strings {
        let steps = actions
            .iter()
            .map(|a| Some(*a))
            .chain(std::iter::once(None));
        for (i, step) in steps.enumerate() {
            let key = atoms[..=i].to_vec();
            match next.get(&key) {
                Some(seen) if *seen != step => return false,
                Some(_) => {}
                None => {
                    next.insert(key, step);
                }
            }
        }
    }
    true
}
