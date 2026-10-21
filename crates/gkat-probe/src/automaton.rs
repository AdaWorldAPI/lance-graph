//! Guarded automata, the Thompson construction, normalization, the
//! bisimulation equivalence check and the separate commit-safety check.
//!
//! Follows Smolka et al. (arXiv:1907.05920): guarded automata (§4.1), the
//! Thompson construction with uniform continuations (§4.2, Proposition 4.2),
//! normalization by removing transitions into states that can never accept
//! (§5.2), and equivalence of normal automata by bisimulation with a
//! union-find, after Hopcroft and Karp (§5.3, Algorithm 1).
//!
//! **Scope.** For expressions in the admitted fragment over a fixed, declared
//! atom set, the paper proves that bisimilarity of the normalized Thompson
//! automata coincides with equivalence of the guarded-string semantics. This
//! module is an implementation of that procedure; its correctness is tested
//! against the independent oracle in [`crate::oracle`], not proven.

use std::collections::{BTreeSet, VecDeque};

use crate::expr::{Action, Atom, Atoms, BExp, Exp, Refusal};

/// What a state does on one atom.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Out {
    Accept,
    Reject,
    /// Perform the action, then continue in the state.
    Step(Action, usize),
}

/// A guarded automaton over a declared atom set. `start[a]` and
/// `states[s][a]` are indexed by the atom's position in [`Atoms`].
#[derive(Clone, Debug)]
pub struct Automaton {
    pub start: Vec<Out>,
    pub states: Vec<Vec<Out>>,
}

/// Builds the guarded Thompson automaton of `e` over `atoms`.
pub fn thompson(e: &Exp, atoms: &Atoms) -> Result<Automaton, Refusal> {
    atoms.admit(e)?;
    let mut states = Vec::new();
    let start = build(e, atoms, &mut states);
    Ok(Automaton { start, states })
}

fn uniform(atoms: &Atoms, out: Out) -> Vec<Out> {
    vec![out; atoms.len()]
}

fn sat(b: &BExp, atoms: &Atoms, a: usize) -> bool {
    b.eval(atoms.get(a))
}

/// Returns the initial pseudostate of `e`; appends its states to `states`.
fn build(e: &Exp, atoms: &Atoms, states: &mut Vec<Vec<Out>>) -> Vec<Out> {
    match e {
        Exp::Fail => uniform(atoms, Out::Reject),
        Exp::Skip => uniform(atoms, Out::Accept),
        Exp::Assert(b) => (0..atoms.len())
            .map(|a| {
                if sat(b, atoms, a) {
                    Out::Accept
                } else {
                    Out::Reject
                }
            })
            .collect(),
        Exp::Do(p) => {
            let s = states.len();
            states.push(uniform(atoms, Out::Accept));
            uniform(atoms, Out::Step(*p, s))
        }
        Exp::If(b, e1, e2) => {
            let i1 = build(e1, atoms, states);
            let i2 = build(e2, atoms, states);
            (0..atoms.len())
                .map(|a| if sat(b, atoms, a) { i1[a] } else { i2[a] })
                .collect()
        }
        Exp::Seq(e1, e2) => {
            let lo = states.len();
            let i1 = build(e1, atoms, states);
            let hi = states.len();
            let i2 = build(e2, atoms, states);
            // Uniform continuation: wherever e1 would accept, continue as e2.
            for s in &mut states[lo..hi] {
                for (a, out) in s.iter_mut().enumerate() {
                    if *out == Out::Accept {
                        *out = i2[a];
                    }
                }
            }
            (0..atoms.len())
                .map(|a| if i1[a] == Out::Accept { i2[a] } else { i1[a] })
                .collect()
        }
        Exp::While(b, body) => {
            let lo = states.len();
            let ib = build(body, atoms, states);
            let hi = states.len();
            // On ¬b the loop exits. On b it enters the body; a body that
            // would accept without acting would repeat on the same atom
            // forever, so it rejects (the paper's `e^(b)` construction).
            let head: Vec<Out> = (0..atoms.len())
                .map(|a| {
                    if !sat(b, atoms, a) {
                        Out::Accept
                    } else {
                        match ib[a] {
                            Out::Step(p, s) => Out::Step(p, s),
                            Out::Accept | Out::Reject => Out::Reject,
                        }
                    }
                })
                .collect();
            for s in &mut states[lo..hi] {
                for (a, out) in s.iter_mut().enumerate() {
                    if *out == Out::Accept {
                        *out = head[a];
                    }
                }
            }
            head
        }
    }
}

impl Automaton {
    /// States from which some run accepts.
    pub fn live(&self) -> Vec<bool> {
        let mut live: Vec<bool> = self
            .states
            .iter()
            .map(|s| s.contains(&Out::Accept))
            .collect();
        loop {
            let mut changed = false;
            for (i, s) in self.states.iter().enumerate() {
                if !live[i] && s.iter().any(|o| matches!(o, Out::Step(_, t) if live[*t])) {
                    live[i] = true;
                    changed = true;
                }
            }
            if !changed {
                return live;
            }
        }
    }

    /// Normalization: a step into a state that can never accept is a reject.
    pub fn normalize(&self) -> Automaton {
        let live = self.live();
        let fix = |row: &Vec<Out>| -> Vec<Out> {
            row.iter()
                .map(|o| match o {
                    Out::Step(_, t) if !live[*t] => Out::Reject,
                    other => *other,
                })
                .collect()
        };
        Automaton {
            start: fix(&self.start),
            states: self.states.iter().map(fix).collect(),
        }
    }

    /// Indices of the states reachable from the start pseudostate.
    pub fn reachable(&self) -> BTreeSet<usize> {
        let mut seen = BTreeSet::new();
        let mut work: Vec<usize> = Vec::new();
        let push = |row: &Vec<Out>, seen: &mut BTreeSet<usize>, work: &mut Vec<usize>| {
            for o in row {
                if let Out::Step(_, t) = o
                    && seen.insert(*t)
                {
                    work.push(*t);
                }
            }
        };
        push(&self.start, &mut seen, &mut work);
        while let Some(s) = work.pop() {
            push(&self.states[s], &mut seen, &mut work);
        }
        seen
    }
}

/// A guarded-string prefix: the atom observed before each action, and the
/// action taken.
pub type Witness = Vec<(Atom, Action)>;

/// The result of an equivalence check.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Equivalence {
    Equivalent {
        /// Pairs merged by the union-find (the proof's size).
        pairs: usize,
    },
    /// The two flows differ on this prefix followed by `atom`.
    Different {
        prefix: Witness,
        atom: Atom,
        left: Out,
        right: Out,
    },
}

impl Equivalence {
    pub fn holds(&self) -> bool {
        matches!(self, Equivalence::Equivalent { .. })
    }
}

struct UnionFind(Vec<usize>);

impl UnionFind {
    fn find(&mut self, x: usize) -> usize {
        let mut r = x;
        while self.0[r] != r {
            r = self.0[r];
        }
        let mut y = x;
        while self.0[y] != r {
            let next = self.0[y];
            self.0[y] = r;
            y = next;
        }
        r
    }
}

/// Decides GKAT equivalence of `e` and `f` over `atoms`.
pub fn equivalent(e: &Exp, f: &Exp, atoms: &Atoms) -> Result<Equivalence, Refusal> {
    let x = thompson(e, atoms)?.normalize();
    let y = thompson(f, atoms)?.normalize();
    // Node ids: 0 = x.start, 1..=nx = x.states, nx+1 = y.start, then y.states.
    let nx = x.states.len();
    let y0 = nx + 1;
    let row = |n: usize| -> &Vec<Out> {
        if n == 0 {
            &x.start
        } else if n <= nx {
            &x.states[n - 1]
        } else if n == y0 {
            &y.start
        } else {
            &y.states[n - y0 - 1]
        }
    };
    let node_x = |s: usize| s + 1;
    let node_y = |s: usize| s + y0 + 1;
    let mut uf = UnionFind((0..y0 + 1 + y.states.len()).collect());
    let mut queue: VecDeque<(usize, usize, Witness)> = VecDeque::new();
    queue.push_back((0, y0, Vec::new()));
    let mut pairs = 0;
    while let Some((p, q, prefix)) = queue.pop_front() {
        let (rp, rq) = (uf.find(p), uf.find(q));
        if rp == rq {
            continue;
        }
        uf.0[rp] = rq;
        pairs += 1;
        let (rowp, rowq) = (row(p), row(q));
        for a in 0..atoms.len() {
            match (rowp[a], rowq[a]) {
                (Out::Accept, Out::Accept) | (Out::Reject, Out::Reject) => {}
                (Out::Step(pa, s), Out::Step(qa, t)) if pa == qa => {
                    let mut next = prefix.clone();
                    next.push((atoms.get(a), pa));
                    queue.push_back((node_x(s), node_y(t), next));
                }
                (l, r) => {
                    return Ok(Equivalence::Different {
                        prefix,
                        atom: atoms.get(a),
                        left: l,
                        right: r,
                    });
                }
            }
        }
    }
    Ok(Equivalence::Equivalent { pairs })
}

/// The result of the commit-safety check.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Safety {
    Safe,
    /// A reachable step performs the guarded action on an atom that does not
    /// satisfy the authorization.
    Unsafe {
        prefix: Witness,
        atom: Atom,
    },
}

impl Safety {
    pub fn holds(&self) -> bool {
        self == &Safety::Safe
    }
}

/// Checks that every reachable step performing `action` happens on an atom
/// satisfying `authorized`.
///
/// The check runs on the unnormalized automaton: an action that is emitted on
/// a run which later rejects was still emitted, so a run's eventual rejection
/// does not excuse an unauthorized intent. The atom checked is the one
/// observed **immediately** before the action: after any intervening action
/// the environment may have changed every test, so an authorization observed
/// earlier does not carry over.
pub fn commit_safe(
    e: &Exp,
    atoms: &Atoms,
    action: Action,
    authorized: &BExp,
) -> Result<Safety, Refusal> {
    let m = thompson(e, atoms)?;
    // Breadth-first over (row, prefix) so the counterexample is shortest.
    let mut seen = BTreeSet::new();
    let mut queue: VecDeque<(Option<usize>, Witness)> = VecDeque::new();
    queue.push_back((None, Vec::new()));
    while let Some((node, prefix)) = queue.pop_front() {
        let row = match node {
            None => &m.start,
            Some(s) => &m.states[s],
        };
        for (a, out) in row.iter().enumerate() {
            if let Out::Step(p, t) = out {
                let atom = atoms.get(a);
                if *p == action && !authorized.eval(atom) {
                    return Ok(Safety::Unsafe { prefix, atom });
                }
                if seen.insert(*t) {
                    let mut next = prefix.clone();
                    next.push((atom, *p));
                    queue.push_back((Some(*t), next));
                }
            }
        }
    }
    Ok(Safety::Safe)
}
