//! The admitted GKAT fragment: Boolean tests, atoms and guarded expressions.
//!
//! Syntax follows Smolka et al., *Guarded Kleene Algebra with Tests*
//! (POPL 2020, arXiv:1907.05920), §2.1:
//!
//! ```text
//! b ::= 0 | 1 | t | ¬b | b ∧ b | b ∨ b          (t a primitive test)
//! e ::= 0 | 1 | b | p | e · e | e +_b e | e^(b)   (p an uninterpreted action)
//! ```
//!
//! `e +_b f` is `if b then e else f` and `e^(b)` is `while b do e`. There is
//! no unguarded union and no Kleene star: they are what this fragment leaves
//! out, and constructs that need them are refused one layer up in
//! [`crate::flow`], never encoded here.

use std::collections::BTreeSet;
use std::fmt;

/// The largest number of primitive tests a probe may declare.
///
/// Atoms are explicit truth assignments, so `k` tests give up to `2^k`
/// atoms. Eight tests bound the atom set at 256. The paper's
/// `O(n α(n))` bound (Theorem 5.10) assumes a fixed atom set; equivalence
/// with an unbounded number of tests is co-NP-hard (Proposition 5.11), so
/// the probe refuses rather than enumerate past this bound.
pub const MAX_TESTS: u8 = 8;

/// An uninterpreted action, identified only by its label.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Action(pub &'static str);

impl fmt::Debug for Action {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.0)
    }
}

/// One truth assignment to the declared primitive tests: bit `t` is test `t`.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub struct Atom(pub u16);

impl Atom {
    pub fn holds(self, test: u8) -> bool {
        self.0 >> test & 1 == 1
    }
}

/// A Boolean test expression over primitive tests.
#[derive(Clone, PartialEq, Eq, Debug)]
pub enum BExp {
    False,
    True,
    Test(u8),
    Not(Box<BExp>),
    And(Box<BExp>, Box<BExp>),
    Or(Box<BExp>, Box<BExp>),
}

impl BExp {
    pub fn eval(&self, atom: Atom) -> bool {
        match self {
            BExp::False => false,
            BExp::True => true,
            BExp::Test(t) => atom.holds(*t),
            BExp::Not(b) => !b.eval(atom),
            BExp::And(a, b) => a.eval(atom) && b.eval(atom),
            BExp::Or(a, b) => a.eval(atom) || b.eval(atom),
        }
    }

    pub fn negated(self) -> BExp {
        BExp::Not(Box::new(self))
    }

    pub fn and(self, other: BExp) -> BExp {
        BExp::And(Box::new(self), Box::new(other))
    }

    pub fn or(self, other: BExp) -> BExp {
        BExp::Or(Box::new(self), Box::new(other))
    }

    fn max_test(&self) -> Option<u8> {
        match self {
            BExp::False | BExp::True => None,
            BExp::Test(t) => Some(*t),
            BExp::Not(b) => b.max_test(),
            BExp::And(a, b) | BExp::Or(a, b) => a.max_test().max(b.max_test()),
        }
    }
}

/// A GKAT expression in the admitted fragment.
#[derive(Clone, PartialEq, Eq, Debug)]
pub enum Exp {
    /// `0`: reject.
    Fail,
    /// `1`: accept without acting.
    Skip,
    /// `b`: accept if the current atom satisfies `b`, else reject.
    Assert(BExp),
    /// `p`: perform an action. The next atom is chosen by the environment.
    Do(Action),
    /// `e · f`.
    Seq(Box<Exp>, Box<Exp>),
    /// `e +_b f`, i.e. `if b then e else f`.
    If(BExp, Box<Exp>, Box<Exp>),
    /// `e^(b)`, i.e. `while b do e`.
    While(BExp, Box<Exp>),
}

impl Exp {
    pub fn seq(self, other: Exp) -> Exp {
        Exp::Seq(Box::new(self), Box::new(other))
    }

    pub fn if_(b: BExp, then: Exp, otherwise: Exp) -> Exp {
        Exp::If(b, Box::new(then), Box::new(otherwise))
    }

    pub fn while_(b: BExp, body: Exp) -> Exp {
        Exp::While(b, Box::new(body))
    }

    /// The actions the expression can name.
    pub fn actions(&self) -> BTreeSet<Action> {
        let mut out = BTreeSet::new();
        self.collect_actions(&mut out);
        out
    }

    fn collect_actions(&self, out: &mut BTreeSet<Action>) {
        match self {
            Exp::Fail | Exp::Skip | Exp::Assert(_) => {}
            Exp::Do(p) => {
                out.insert(*p);
            }
            Exp::Seq(a, b) | Exp::If(_, a, b) => {
                a.collect_actions(out);
                b.collect_actions(out);
            }
            Exp::While(_, e) => e.collect_actions(out),
        }
    }

    fn max_test(&self) -> Option<u8> {
        match self {
            Exp::Fail | Exp::Skip | Exp::Do(_) => None,
            Exp::Assert(b) => b.max_test(),
            Exp::Seq(a, b) => a.max_test().max(b.max_test()),
            Exp::If(c, a, b) => c.max_test().max(a.max_test()).max(b.max_test()),
            Exp::While(c, e) => c.max_test().max(e.max_test()),
        }
    }
}

/// Why the probe refuses an input instead of answering.
#[derive(Clone, PartialEq, Eq, Debug)]
pub enum Refusal {
    /// More primitive tests were declared than [`MAX_TESTS`].
    AtomBound { tests: usize, max: u8 },
    /// The declared constraint leaves no atom.
    NoAtoms,
    /// An expression names a test the atom set does not declare.
    UndeclaredTest { test: u8, declared: u8 },
    /// An expression names an action outside the declared alphabet.
    UndeclaredAction(Action),
    /// Two flows were compared under different declarations.
    DeclarationMismatch(&'static str),
    /// Unguarded choice between branches: outside GKAT (no `+` in the fragment).
    UnguardedChoice,
    /// `GoTo` / `GoBack`: unstructured control flow, outside GKAT.
    UnstructuredJump(&'static str),
    /// A guard used a projection its observable does not declare.
    UndeclaredProjection {
        observable: &'static str,
        projection: &'static str,
    },
    /// An observable name that was not declared.
    UnknownObservable(&'static str),
}

/// The declared atom set: every truth assignment to `tests` primitive tests
/// that satisfies `constraint`.
#[derive(Clone, Debug)]
pub struct Atoms {
    tests: u8,
    list: Vec<Atom>,
}

impl Atoms {
    /// Enumerates the atoms of `tests` primitive tests under `constraint`.
    /// Refuses more than [`MAX_TESTS`] tests and an empty atom set.
    pub fn new(tests: usize, constraint: &BExp) -> Result<Atoms, Refusal> {
        if tests > MAX_TESTS as usize {
            return Err(Refusal::AtomBound {
                tests,
                max: MAX_TESTS,
            });
        }
        if let Some(t) = constraint.max_test()
            && t as usize >= tests
        {
            return Err(Refusal::UndeclaredTest {
                test: t,
                declared: tests as u8,
            });
        }
        let list: Vec<Atom> = (0..1u32 << tests)
            .map(|a| Atom(a as u16))
            .filter(|a| constraint.eval(*a))
            .collect();
        if list.is_empty() {
            return Err(Refusal::NoAtoms);
        }
        Ok(Atoms {
            tests: tests as u8,
            list,
        })
    }

    pub fn tests(&self) -> u8 {
        self.tests
    }

    pub fn len(&self) -> usize {
        self.list.len()
    }

    pub fn is_empty(&self) -> bool {
        self.list.is_empty()
    }

    pub fn iter(&self) -> impl Iterator<Item = Atom> + '_ {
        self.list.iter().copied()
    }

    pub fn get(&self, index: usize) -> Atom {
        self.list[index]
    }

    /// Refuses an expression that names a test outside this atom set.
    pub fn admit(&self, e: &Exp) -> Result<(), Refusal> {
        match e.max_test() {
            Some(t) if t >= self.tests => Err(Refusal::UndeclaredTest {
                test: t,
                declared: self.tests,
            }),
            _ => Ok(()),
        }
    }
}
