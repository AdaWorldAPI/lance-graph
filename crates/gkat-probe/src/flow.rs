//! A small graph-flow-shaped source language and its lowering to GKAT.
//!
//! The source language is deliberately larger than GKAT: it has the
//! constructs a workflow author can write, including the ones GKAT cannot
//! express. Lowering admits the structured fragment and **refuses** the rest
//! with a typed [`Refusal`]; it never invents a predicate to make an
//! unguarded choice look guarded, and never translates a jump.
//!
//! Correspondence with `graph_flow::NextAction` (rs-graph-llm
//! `graph-flow/src/task.rs`):
//!
//! | `NextAction` | here |
//! |---|---|
//! | `Continue`, `ContinueAndExecute` | sequence (the two differ in scheduling, not in the control-flow language) |
//! | `End` | end of the flow: accept |
//! | `WaitForInput` | [`Step::WaitForInput`]: a guarded loop whose test is observed at resume |
//! | `GoTo(id)` | [`Step::GoTo`]: refused |
//! | `GoBack` | [`Step::GoBack`]: refused |
//!
//! Conditional edges correspond to [`Step::If`]. graph-flow has no edge type
//! that picks a branch without a condition; [`Step::Choice`] models one so the
//! refusal can be tested.
//!
//! ## Guards and projections
//!
//! An observable is declared as Boolean or as three-valued (K3: T, F, U). A
//! K3 observable is a leaf over a nullable column (`crates/lance-graph-quack`,
//! `Filter::sql_where`): the column's stored value plus its validity plane.
//! The leaf is the Boolean read off the stored value, and what that means
//! depends on the column's type. For a `BOOLEAN` column the leaf is the stored
//! bit itself (in Quack, a mask plane). For a numeric or text column it is a
//! comparison such as `rev = 5`, and 0 and 1 in the lane are ordinary
//! numbers. Either way the observable occupies two tests: `leaf` and `valid`.
//! Under `valid = 0` no value is stored (PostgreSQL) or the payload is
//! arbitrary (Quack), so the leaf there reads nothing meaningful. There is no
//! constraint between the two tests: all four pairs are atoms, and NULL is the
//! two with `valid = 0`. A K3 observable reaches a Boolean guard only through
//! a declared projection, and neither projection reads the leaf under a NULL:
//!
//! * [`Cond::Where`]: designated `{T}`, i.e. `T(φ) = valid ∧ leaf`, what
//!   Quack's `WHERE` keeps;
//! * [`Cond::Check`]: designated `{T, U}`, i.e. `¬F(φ) = ¬valid ∨ leaf`.
//!
//! A Boolean observable is read with [`Cond::Is`]. Using a K3 projection on a
//! Boolean observable, or `Is` on a K3 one, is refused.

use crate::expr::{Action, Atoms, BExp, Exp, Refusal};

/// The kind of an observable.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Kind {
    Bool,
    K3,
}

/// A guard condition in the source language.
#[derive(Clone, PartialEq, Eq, Debug)]
pub enum Cond {
    /// A Boolean observable is true.
    Is(&'static str),
    /// WHERE projection of a K3 observable: only `T` passes.
    Where(&'static str),
    /// CHECK projection of a K3 observable: `T` and `U` pass.
    Check(&'static str),
    Not(Box<Cond>),
    And(Box<Cond>, Box<Cond>),
    Or(Box<Cond>, Box<Cond>),
}

impl Cond {
    pub fn negated(self) -> Cond {
        Cond::Not(Box::new(self))
    }
    pub fn and(self, other: Cond) -> Cond {
        Cond::And(Box::new(self), Box::new(other))
    }
    pub fn or(self, other: Cond) -> Cond {
        Cond::Or(Box::new(self), Box::new(other))
    }
}

/// A step of a source flow.
#[derive(Clone, PartialEq, Eq, Debug)]
pub enum Step {
    Skip,
    /// Abort the flow: reject.
    Abort,
    Act(Action),
    Seq(Vec<Step>),
    If(Cond, Box<Step>, Box<Step>),
    While(Cond, Box<Step>),
    /// Suspend until `ready` holds. Lowers to `while ¬ready do suspend`: the
    /// flow performs `suspend`, and `ready` is a test of the external state
    /// observed at resume. Snapshot and codebook staleness are not modelled;
    /// they are a separate runtime check.
    WaitForInput {
        ready: Cond,
        suspend: Action,
    },
    /// Pick a branch without a condition. Outside GKAT: refused.
    Choice(Vec<Step>),
    /// Jump to a named task. Outside GKAT: refused.
    GoTo(&'static str),
    /// Return to the previous task. Outside GKAT: refused.
    GoBack,
}

impl Step {
    pub fn if_(c: Cond, then: Step, otherwise: Step) -> Step {
        Step::If(c, Box::new(then), Box::new(otherwise))
    }
    pub fn while_(c: Cond, body: Step) -> Step {
        Step::While(c, Box::new(body))
    }
}

/// The declarations a flow is checked under: its observables (which fix the
/// atom set) and its action alphabet.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct Signature {
    pub observables: Vec<(&'static str, Kind)>,
    pub actions: Vec<Action>,
}

impl Signature {
    /// The first primitive test of an observable: the leaf for K3
    /// (followed by its validity bit), or the only test of a Boolean.
    fn tests_of(&self, name: &'static str) -> Result<(Kind, u8), Refusal> {
        let mut next = 0u8;
        for (n, k) in &self.observables {
            if *n == name {
                return Ok((*k, next));
            }
            next += match k {
                Kind::Bool => 1,
                Kind::K3 => 2,
            };
        }
        Err(Refusal::UnknownObservable(name))
    }

    /// The number of primitive tests the observables occupy.
    pub fn test_count(&self) -> usize {
        self.observables
            .iter()
            .map(|(_, k)| match k {
                Kind::Bool => 1,
                Kind::K3 => 2,
            })
            .sum()
    }

    /// The declared atom set. Refuses more than [`crate::expr::MAX_TESTS`]
    /// primitive tests.
    pub fn atoms(&self) -> Result<Atoms, Refusal> {
        let tests = self.test_count();
        if tests > crate::expr::MAX_TESTS as usize {
            return Err(Refusal::AtomBound {
                tests,
                max: crate::expr::MAX_TESTS,
            });
        }
        // No constraint: under a NULL the stored payload, and so the leaf, is
        // arbitrary. The guards, not the atom set, make NULL one value.
        Atoms::new(tests, &BExp::True)
    }

    /// Lowers a condition to a Boolean test.
    pub fn guard(&self, c: &Cond) -> Result<BExp, Refusal> {
        Ok(match c {
            Cond::Is(n) => match self.tests_of(n)? {
                (Kind::Bool, t) => BExp::Test(t),
                (Kind::K3, _) => {
                    return Err(Refusal::UndeclaredProjection {
                        observable: n,
                        projection: "Is",
                    });
                }
            },
            Cond::Where(n) => match self.tests_of(n)? {
                (Kind::K3, t) => BExp::Test(t + 1).and(BExp::Test(t)),
                (Kind::Bool, _) => {
                    return Err(Refusal::UndeclaredProjection {
                        observable: n,
                        projection: "Where",
                    });
                }
            },
            Cond::Check(n) => match self.tests_of(n)? {
                (Kind::K3, t) => BExp::Test(t + 1).negated().or(BExp::Test(t)),
                (Kind::Bool, _) => {
                    return Err(Refusal::UndeclaredProjection {
                        observable: n,
                        projection: "Check",
                    });
                }
            },
            Cond::Not(a) => self.guard(a)?.negated(),
            Cond::And(a, b) => self.guard(a)?.and(self.guard(b)?),
            Cond::Or(a, b) => self.guard(a)?.or(self.guard(b)?),
        })
    }

    /// Lowers a flow to GKAT, or refuses it.
    pub fn lower(&self, s: &Step) -> Result<Exp, Refusal> {
        Ok(match s {
            Step::Skip => Exp::Skip,
            Step::Abort => Exp::Fail,
            Step::Act(p) => {
                if !self.actions.contains(p) {
                    return Err(Refusal::UndeclaredAction(*p));
                }
                Exp::Do(*p)
            }
            Step::Seq(steps) => {
                let mut out = Exp::Skip;
                for (i, st) in steps.iter().enumerate() {
                    let e = self.lower(st)?;
                    out = if i == 0 { e } else { out.seq(e) };
                }
                out
            }
            Step::If(c, a, b) => Exp::if_(self.guard(c)?, self.lower(a)?, self.lower(b)?),
            Step::While(c, body) => Exp::while_(self.guard(c)?, self.lower(body)?),
            Step::WaitForInput { ready, suspend } => {
                if !self.actions.contains(suspend) {
                    return Err(Refusal::UndeclaredAction(*suspend));
                }
                Exp::while_(self.guard(ready)?.negated(), Exp::Do(*suspend))
            }
            Step::Choice(_) => return Err(Refusal::UnguardedChoice),
            Step::GoTo(_) => return Err(Refusal::UnstructuredJump("GoTo")),
            Step::GoBack => return Err(Refusal::UnstructuredJump("GoBack")),
        })
    }
}

/// A flow under its signature.
#[derive(Clone, Debug)]
pub struct Flow {
    pub signature: Signature,
    pub body: Step,
}

impl Flow {
    /// Lowers the flow and builds its atom set.
    pub fn lower(&self) -> Result<(Exp, Atoms), Refusal> {
        let atoms = self.signature.atoms()?;
        let e = self.signature.lower(&self.body)?;
        atoms.admit(&e)?;
        Ok((e, atoms))
    }
}

/// Decides equivalence of two flows. Refuses flows declared under different
/// signatures: equivalence is only meaningful over the same action alphabet,
/// the same observables and the same projections.
pub fn flows_equivalent(a: &Flow, b: &Flow) -> Result<crate::automaton::Equivalence, Refusal> {
    if a.signature != b.signature {
        return Err(Refusal::DeclarationMismatch("signature"));
    }
    let (ea, atoms) = a.lower()?;
    let (eb, _) = b.lower()?;
    crate::automaton::equivalent(&ea, &eb, &atoms)
}
