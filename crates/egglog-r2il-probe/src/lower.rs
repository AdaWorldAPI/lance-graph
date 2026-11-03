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
//!
//! No name reaches execution. Only the target variable may be read, so
//! lowering resolves every variable once: the target becomes
//! [`Step::PushInput`], and any other variable is refused there, at lowering,
//! as [`LowerRefusal::UnboundVar`]. [`run_steps`] takes the input value and
//! reads no string. The variable's name lives only in the cold traceroute
//! ([`Step::mnemonic_with`]).

use lance_graph_contract::set_rules::Arith;
use lance_graph_mask_risc::{Foreign, Planes};
use lance_graph_r2il::{FoldDialect, FoldError, INT_MULT, Val};

use crate::term::Tm;
use crate::{Compiled, LowerRefusal};

/// One step of the lowered program, symbolic (no input value yet), so the
/// sequence is comparable across runs and independent of the inputs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Step {
    /// `push k`: a literal scalar.
    PushConst(i64),
    /// Push the input: the target variable's value. Resolved at lowering;
    /// carries no name.
    PushInput,
    /// `call INT_MULT` on the top two scalars.
    IntMult,
}

impl Step {
    /// Mnemonic for the traceroute, with the input shown as `$input`.
    pub fn mnemonic(&self) -> String {
        self.mnemonic_with("input")
    }

    /// Mnemonic with the input shown under `label` (the target's name). Cold:
    /// only the traceroute reads it.
    pub fn mnemonic_with(&self, label: &str) -> String {
        match self {
            Step::PushConst(k) => format!("push {k}"),
            Step::PushInput => format!("push ${label}"),
            Step::IntMult => "INT_MULT".to_string(),
        }
    }
}

/// Post-order walk: `Num k` pushes, the target pushes the input, `Mul l r`
/// is `l; r; INT_MULT`. Any other variable is refused.
pub fn steps_of(tm: &Tm, target_var: &str) -> Result<Vec<Step>, LowerRefusal> {
    let mut out = Vec::new();
    walk(tm, target_var, &mut out)?;
    Ok(out)
}

fn walk(tm: &Tm, target_var: &str, out: &mut Vec<Step>) -> Result<(), LowerRefusal> {
    match tm {
        Tm::Num(k) => out.push(Step::PushConst(*k)),
        Tm::Var(v) | Tm::Def(v) if v == target_var => out.push(Step::PushInput),
        Tm::Var(v) | Tm::Def(v) => return Err(LowerRefusal::UnboundVar(v.clone())),
        Tm::Mul(l, r) => {
            walk(l, target_var, out)?;
            walk(r, target_var, out)?;
            out.push(Step::IntMult);
        }
    }
    Ok(())
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
    steps_of(&tm, c.target_var)
}

/// Lowers and executes. `inputs` supplies values by variable name; only
/// `c.target_var` may be read, any other variable is `UnboundVar` (refused at
/// lowering). The name lookup happens here, once, before execution.
pub fn lower_and_run(c: &Compiled, inputs: &[(&str, i64)]) -> Result<i64, LowerRefusal> {
    let steps = lower_steps(c)?;
    let input = inputs
        .iter()
        .find(|(n, _)| *n == c.target_var)
        .map(|(_, x)| *x);
    match (steps.contains(&Step::PushInput), input) {
        (true, None) => Err(LowerRefusal::UnboundVar(c.target_var.to_string())),
        (_, x) => run_steps(&steps, x.unwrap_or(0)),
    }
}

/// Executes a step list on a scalar-only `FoldDialect`. `input` is the value
/// [`Step::PushInput`] pushes. No name is read.
pub fn run_steps(steps: &[Step], input: i64) -> Result<i64, LowerRefusal> {
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
            Step::PushInput => stack.push(Val::Scalar(input)),
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
