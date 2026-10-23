//! Stage B, E1 end to end, probe-local.
//!
//! E1 is exact-integer SYMBOLIC SUBSTITUTION: the premises `a = 2*b`,
//! `b = 2*c`, `z = 2*a` imply `z = 8*c`. It is not a population measure: no
//! SUM fold, no lanes, no masks.
//!
//! The pipeline, each hop named by the traceroute ([`trace`]):
//!
//! 1. [`laws`] admits or refuses each of the probe's two laws for the chosen
//!    [`Arith`] domain (the contract's own vocabulary).
//! 2. [`encode`] + `session` hand egglog (cold, bounded, public API only,
//!    proofs mode) the premises and the admitted rewrites, step it up to
//!    `fuel` times and report whether it saturated.
//! 3. The extraction is egglog's own: `prove-variants` on the goal, cheapest
//!    first under its default AST-size cost, with every non-target variable a
//!    `Def` priced at 1000 ([`encode::HEADER`]). The answer is the cheapest
//!    variant that is not the goal's own name. egglog checks every proof it
//!    returns, with its own checker (not an independent check).
//! 4. [`lower`] turns the term into calls on the EXISTING R2IL scalar
//!    arithmetic (`INT_MULT` through `FoldDialect`); no new opcode.
//! 5. [`oracle`] recomputes the answer in `i128` from the original premises,
//!    sharing no code with steps 2-4.

pub mod artifact;
pub mod catalogue;
pub mod encode;
pub mod heritage;
pub mod laws;
pub mod lower;
pub mod oracle;
mod session;
pub mod term;
pub mod trace;

use std::time::{Duration, Instant};

pub use lance_graph_contract::set_rules::Arith;
pub use laws::{LawId, LawRefusal, RuleId};
pub use lower::{Step, lower_and_run, lower_steps};
pub use term::Tm;

use session::{Session, VARIANTS};

/// One premise: `lhs = coeff * rhs`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Eq {
    /// The defined variable.
    pub lhs: &'static str,
    /// The integer coefficient.
    pub coeff: i64,
    /// The variable it is a multiple of.
    pub rhs: &'static str,
}

/// A substitution problem: from the premises, express `goal` in `target_var`.
#[derive(Debug, Clone, PartialEq)]
pub struct Problem {
    /// The premises.
    pub premises: Vec<Eq>,
    /// The variable to express.
    pub goal: &'static str,
    /// The variable the answer should be in.
    pub target_var: &'static str,
}

/// E1: `a = 2b`, `b = 2c`, `z = 2a`; goal `z` in terms of `c`.
pub fn e1_problem() -> Problem {
    Problem {
        premises: vec![
            Eq {
                lhs: "a",
                coeff: 2,
                rhs: "b",
            },
            Eq {
                lhs: "b",
                coeff: 2,
                rhs: "c",
            },
            Eq {
                lhs: "z",
                coeff: 2,
                rhs: "a",
            },
        ],
        goal: "z",
        target_var: "c",
    }
}

/// A longer chain `p0 = 2 p1, ..., p5 = 2 p6` (goal `p0` in `p6`), which
/// needs more saturation steps than E1.
pub fn chain_problem() -> Problem {
    Problem {
        premises: vec![
            Eq {
                lhs: "p0",
                coeff: 2,
                rhs: "p1",
            },
            Eq {
                lhs: "p1",
                coeff: 2,
                rhs: "p2",
            },
            Eq {
                lhs: "p2",
                coeff: 2,
                rhs: "p3",
            },
            Eq {
                lhs: "p3",
                coeff: 2,
                rhs: "p4",
            },
            Eq {
                lhs: "p4",
                coeff: 2,
                rhs: "p5",
            },
            Eq {
                lhs: "p5",
                coeff: 2,
                rhs: "p6",
            },
        ],
        goal: "p0",
        target_var: "p6",
    }
}

/// Whether egglog reached a fixpoint within the fuel.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Saturation {
    /// A step changed nothing. `iterations` counts the steps run, INCLUDING
    /// that last, quiet one. Zero means no rule was admitted, so no step ran.
    Saturated {
        /// Steps run, the final quiet step included.
        iterations: usize,
    },
    /// Every one of `fuel` steps still changed the e-graph: nothing is
    /// certified about completeness.
    Incomplete {
        /// The step budget that was exhausted.
        fuel: usize,
    },
}

/// Wall-clock cost of the cold compile, split. Reported, never gated.
#[derive(Debug, Clone, Copy, Default)]
pub struct ColdCost {
    /// Building the egglog program text and the law table.
    pub encode: Duration,
    /// egglog: setup, saturation, `prove-variants`.
    pub egglog: Duration,
}

/// The result of the cold compile.
#[derive(Debug, Clone)]
pub struct Compiled {
    /// The arithmetic domain the laws were admitted under.
    pub arith: Arith,
    /// The variable the answer is expressed in.
    pub target_var: &'static str,
    /// Laws admitted, with their domain.
    pub admitted: Vec<RuleId>,
    /// Laws refused, with the reason.
    pub refused: Vec<(LawId, LawRefusal)>,
    /// Fixpoint or fuel exhaustion.
    pub saturation: Saturation,
    /// The chosen term, rendered `(Mul (Num 8) (Var "c"))`: egglog's cheapest
    /// variant of the goal other than the goal's own `(Def "z")`, which is a
    /// self-identity, not an answer. A defined variable left in it is
    /// rendered `(Def "b")`.
    pub extracted: String,
    /// Every variant `prove-variants` returned for the goal, in egglog's
    /// order (cheapest first).
    pub variants: Vec<String>,
    /// egglog's checked proof text that `goal = extracted`; `None` when no
    /// non-trivial variant exists.
    pub proof: Option<String>,
    /// The extracted term mentions no variable other than `target_var`.
    pub closed: bool,
    /// Cold-compile cost.
    pub timings: ColdCost,
    /// Non-fatal observations.
    pub notes: Vec<String>,
    /// A fatal egglog error; everything else is then at its default.
    pub error: Option<String>,
}

/// Why a term could not be run on R2IL scalar arithmetic.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LowerRefusal {
    /// A variable with no supplied value.
    UnboundVar(String),
    /// A checked multiply left `i64`.
    Overflow,
    /// The term cannot be lowered at all.
    NotLowerable(&'static str),
}

/// Everything `prepare` leaves behind.
struct Prepared {
    session: Session,
    admitted: Vec<RuleId>,
    refused: Vec<(LawId, LawRefusal)>,
    saturation: Saturation,
}

fn prepare(problem: &Problem, arith: Arith, fuel: usize) -> Result<Prepared, String> {
    let (admitted, refused) = laws::admit_all(arith);
    let program = encode::setup(problem, &admitted);
    let mut session = Session::new();
    session.run(&program)?;
    let saturation = session.saturate(fuel, !admitted.is_empty())?;
    Ok(Prepared {
        session,
        admitted,
        refused,
        saturation,
    })
}

/// Cold-compiles `problem` under `arith`, with at most `fuel` saturation
/// steps. Never panics on an egglog failure: it is returned in
/// [`Compiled::error`].
pub fn compile(problem: &Problem, arith: Arith, fuel: usize) -> Compiled {
    let t_encode = Instant::now();
    let (admitted, refused) = laws::admit_all(arith);
    let _ = encode::setup(problem, &admitted); // priced here; rebuilt inside `prepare`
    let encode_cost = t_encode.elapsed();

    let mut out = Compiled {
        arith,
        target_var: problem.target_var,
        admitted: admitted.clone(),
        refused: refused.clone(),
        saturation: Saturation::Incomplete { fuel },
        extracted: String::new(),
        variants: Vec::new(),
        proof: None,
        closed: false,
        timings: ColdCost {
            encode: encode_cost,
            egglog: Duration::ZERO,
        },
        notes: Vec::new(),
        error: None,
    };

    let t_egg = Instant::now();
    if let Err(e) = compile_inner(problem, arith, fuel, &mut out) {
        out.error = Some(e);
    }
    out.timings.egglog = t_egg.elapsed();
    out
}

fn compile_inner(
    problem: &Problem,
    arith: Arith,
    fuel: usize,
    out: &mut Compiled,
) -> Result<(), String> {
    let Prepared {
        mut session,
        admitted,
        refused,
        saturation,
    } = prepare(problem, arith, fuel)?;
    out.admitted = admitted;
    out.refused = refused;
    out.saturation = saturation;

    // egglog's own extraction: variants cheapest first, each with its proof.
    let proved = session.prove_variants(problem, VARIANTS)?;
    out.variants = proved.iter().map(|(t, _)| t.clone()).collect();
    let goal_name = encode::sym(problem.goal, problem.target_var);
    let pick = proved.iter().find_map(|(t, proof)| {
        let tm = Tm::parse(t)?;
        (tm.to_string() != goal_name).then(|| (tm, proof.clone()))
    });
    match pick {
        Some((tm, proof)) => {
            out.closed = tm.foreign_vars(problem.target_var).is_empty();
            out.extracted = tm.to_string();
            out.proof = Some(proof);
        }
        None => {
            out.extracted = goal_name;
            out.notes
                .push("no variant other than the goal itself".to_string());
        }
    }
    Ok(())
}

/// Runs the same setup and saturation as [`compile`], then asks egglog to
/// `(prove (= (Var goal) <rhs_term>))`. `Ok` is the checked proof text; `Err`
/// is egglog's refusal (or a setup failure).
pub fn try_prove(
    problem: &Problem,
    arith: Arith,
    fuel: usize,
    rhs_term: &str,
) -> Result<String, String> {
    let mut p = prepare(problem, arith, fuel)?;
    p.session.prove_eq(problem, rhs_term)
}
