//! Extracted term -> existing R2IL scalar arithmetic.
//!
//! The lowering target is the opcode that already exists: push
//! [`Val::Scalar`] literals and call
//! `ogar_loco::Dialect::call(&mut FoldDialect, INT_MULT, [0; 3], &mut stack)`,
//! the exact pattern of `lance-graph-r2il/tests/procs.rs:953-962`. No new
//! opcode, no mask, no lane: `FoldDialect` is built over empty planes.
//!
//! `INT_MULT` is CHECKED (`checked_mul`, `lance-graph-r2il/src/lib.rs:1064`):
//! overflow is `FoldError::Overflow`, never a wrap. A term compiled under
//! `Arith::Wrapping` is therefore refused up front rather than run on
//! checked arithmetic and silently answered.

use lance_graph_contract::set_rules::Arith;
use lance_graph_mask_risc::{Foreign, Planes};
use lance_graph_r2il::{FoldDialect, FoldError, INT_MULT, Val};

use crate::term::Tm;
use crate::{Compiled, LowerRefusal};

/// One step of the lowered program, symbolic (no input value yet), so the
/// sequence is comparable across runs and independent of the inputs.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Step {
    /// `push k`: a literal scalar.
    PushConst(i64),
    /// Push the supplied value of the named variable.
    PushVar(String),
    /// `call INT_MULT` on the top two scalars.
    IntMult,
}

impl Step {
    /// Mnemonic for the traceroute.
    pub fn mnemonic(&self) -> String {
        match self {
            Step::PushConst(k) => format!("push {k}"),
            Step::PushVar(v) => format!("push ${v}"),
            Step::IntMult => "INT_MULT".to_string(),
        }
    }
}

/// Post-order walk: `Num k` pushes, `Var v` pushes, `Mul l r` is `l; r; INT_MULT`.
pub fn steps_of(tm: &Tm) -> Vec<Step> {
    let mut out = Vec::new();
    walk(tm, &mut out);
    out
}

fn walk(tm: &Tm, out: &mut Vec<Step>) {
    match tm {
        Tm::Num(k) => out.push(Step::PushConst(*k)),
        Tm::Var(v) | Tm::Def(v) => out.push(Step::PushVar(v.clone())),
        Tm::Mul(l, r) => {
            walk(l, out);
            walk(r, out);
            out.push(Step::IntMult);
        }
    }
}

/// The call sequence for the compiled problem's extracted term. Refuses a
/// `Wrapping` term and an unparseable one.
pub fn lower_steps(c: &Compiled) -> Result<Vec<Step>, LowerRefusal> {
    if c.arith == Arith::Wrapping {
        return Err(LowerRefusal::NotLowerable(
            "R2IL scalar arithmetic is checked, not wrapping",
        ));
    }
    let tm = Tm::parse(&c.extracted).ok_or(LowerRefusal::NotLowerable(
        "extracted term is not parseable",
    ))?;
    Ok(steps_of(&tm))
}

/// Lowers and executes. `inputs` supplies values by variable name; only
/// `c.target_var` may be read, any other variable is `UnboundVar`.
pub fn lower_and_run(c: &Compiled, inputs: &[(&str, i64)]) -> Result<i64, LowerRefusal> {
    let steps = lower_steps(c)?;
    run_steps(&steps, c.target_var, inputs)
}

/// Executes a step list on a scalar-only `FoldDialect`.
pub fn run_steps(
    steps: &[Step],
    target_var: &str,
    inputs: &[(&str, i64)],
) -> Result<i64, LowerRefusal> {
    let planes = Planes {
        n_rows: 0,
        masks: &[],
        lanes: &[],
    };
    let mut d = FoldDialect::new(&planes, &Foreign::NONE);
    let mut stack: Vec<Val> = Vec::new();
    for step in steps {
        match step {
            Step::PushConst(k) => stack.push(Val::Scalar(*k)),
            Step::PushVar(v) => {
                let value = (v == target_var)
                    .then(|| {
                        inputs
                            .iter()
                            .find(|(n, _)| *n == v.as_str())
                            .map(|(_, x)| *x)
                    })
                    .flatten();
                match value {
                    Some(x) => stack.push(Val::Scalar(x)),
                    None => return Err(LowerRefusal::UnboundVar(v.clone())),
                }
            }
            Step::IntMult => {
                // Same call shape as lance-graph-r2il/tests/procs.rs:958-960.
                d.call(INT_MULT, [0; 3], &mut stack).map_err(|e| match e {
                    FoldError::Overflow(_) => LowerRefusal::Overflow,
                    _ => LowerRefusal::NotLowerable("fold dialect refused INT_MULT"),
                })?;
            }
        }
    }
    match stack.as_slice() {
        [Val::Scalar(r)] => Ok(*r),
        _ => Err(LowerRefusal::NotLowerable(
            "lowering did not leave exactly one scalar",
        )),
    }
}
