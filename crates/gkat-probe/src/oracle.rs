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

use std::collections::{BTreeMap, BTreeSet};

use crate::expr::{Action, Atom, Atoms, BExp, Exp};

/// Internal steps allowed between two actions before a run counts as
/// diverging. Between actions the atom cannot change, so a run that takes
/// more steps than the expression has nodes without acting is in a loop that
/// never acts; the bound only needs to exceed any terminating stretch.
const FUEL: usize = 1 << 14;

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
    let mut fuel = FUEL;
    loop {
        let Some(e) = stack.pop() else {
            out.accepted.insert((seen_atoms.clone(), actions.clone()));
            return;
        };
        if fuel == 0 {
            out.diverged += 1;
            return;
        }
        fuel -= 1;
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
/// two strings that observe the same atoms up to a position must have taken
/// the same actions up to that position.
pub fn deterministic(strings: &BTreeSet<GuardedString>) -> bool {
    // Map each observed atom prefix to the action taken after it.
    let mut next: BTreeMap<Vec<Atom>, Action> = BTreeMap::new();
    for (atoms, actions) in strings {
        for (i, action) in actions.iter().enumerate() {
            let key = atoms[..=i].to_vec();
            match next.get(&key) {
                Some(seen) if seen != action => return false,
                Some(_) => {}
                None => {
                    next.insert(key, *action);
                }
            }
        }
    }
    true
}
