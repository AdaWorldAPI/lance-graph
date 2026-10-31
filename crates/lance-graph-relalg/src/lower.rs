//! Lowering: a [`Query`] becomes R2IL calls over [`FoldDialect`], which
//! builds and runs one mask-risc program at the measure. Nothing here loops
//! over rows.

use lance_graph_mask_risc::{Foreign, Planes, Program};
use lance_graph_r2il::{
    FoldDialect, FoldError, Val, INT_AND, INT_EQUAL, INT_NOT, INT_OR, INT_S_LESS, LOAD, POP_COUNT,
};
use ogar_loco::FnIndex;
use ogar_r2il::{KEEP, MAX, MIN, RANGE, SUM};

use crate::ir::{Atom, Cmp, Measure, Query, Rel};
use crate::normalize::{normalize, symbolic};

/// `LOAD` spaces, as `FoldDialect` reads them.
const SPACE_U32: u8 = 0;
const SPACE_I32: u8 = 1;
const SPACE_PLANE: u8 = 3;

/// What a query produced.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Answer {
    /// Row count.
    Count(u64),
    /// Σ of the lane over the rows.
    Sum(i64),
    /// Minimum, `None` for no rows.
    Min(Option<i32>),
    /// Maximum, `None` for no rows.
    Max(Option<i32>),
    /// The rows were written into the caller's mask.
    Kept,
}

/// An answer and what it cost.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Run {
    /// The answer.
    pub answer: Answer,
    /// mask-risc ops executed (the facade passes).
    pub passes: usize,
    /// mask-risc programs executed; `0` when the answer was known from the
    /// description alone.
    pub programs: usize,
}

/// Why a query was refused.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RelError {
    /// The dialect or mask-risc refused the lowering.
    Fold(FoldError),
    /// `Measure::Keep` without a mask to write into.
    NoKeepSink,
}

impl From<FoldError> for RelError {
    fn from(e: FoldError) -> Self {
        RelError::Fold(e)
    }
}

/// Rewrite the query, then lower and run it.
pub fn run(q: &Query, planes: &Planes<'_>, keep: Option<&mut [u64]>) -> Result<Run, RelError> {
    run_with_program(q, planes, keep).map(|(run, _)| run)
}

/// [`run`], also returning the mask-risc program the fold ran, so a caller can
/// inspect its ops (for example, that no `Gather` or `Scatter` was emitted).
/// `None` when the description alone answered and no program ran.
pub fn run_with_program(
    q: &Query,
    planes: &Planes<'_>,
    keep: Option<&mut [u64]>,
) -> Result<(Run, Option<Program>), RelError> {
    let rel = normalize(&q.rel);
    if let Some(answer) = symbolic(&rel, q.measure, planes.n_rows) {
        let run = Run {
            answer,
            passes: 0,
            programs: 0,
        };
        return Ok((run, None));
    }
    lower_and_run(
        &Query {
            rel,
            measure: q.measure,
        },
        planes,
        keep,
    )
}

/// Lower and run the query exactly as written, without any rewrite. The
/// reference the rewrites are measured against.
pub fn run_as_written(
    q: &Query,
    planes: &Planes<'_>,
    keep: Option<&mut [u64]>,
) -> Result<Run, RelError> {
    lower_and_run(q, planes, keep).map(|(run, _)| run)
}

fn lower_and_run(
    q: &Query,
    planes: &Planes<'_>,
    keep: Option<&mut [u64]>,
) -> Result<(Run, Option<Program>), RelError> {
    let mut d = match (q.measure, keep) {
        (Measure::Keep, Some(sink)) => FoldDialect::with_keep(planes, &Foreign::NONE, sink),
        (Measure::Keep, None) => return Err(RelError::NoKeepSink),
        _ => FoldDialect::new(planes, &Foreign::NONE),
    };
    let mut stack = Vec::new();
    lower_rel(&mut d, &mut stack, &q.rel)?;
    let answer = match q.measure {
        Measure::Count => {
            call(&mut d, POP_COUNT, [0; 3], &mut stack)?;
            Answer::Count(scalar(&mut stack)? as u64)
        }
        Measure::Sum(lane) => {
            load(&mut d, &mut stack, SPACE_I32, lane)?;
            call(&mut d, SUM, [0; 3], &mut stack)?;
            Answer::Sum(scalar(&mut stack)?)
        }
        Measure::Min(lane) => {
            load(&mut d, &mut stack, SPACE_I32, lane)?;
            call(&mut d, MIN, [0; 3], &mut stack)?;
            let v = scalar(&mut stack)?;
            Answer::Min((v != i64::MAX).then_some(v as i32))
        }
        Measure::Max(lane) => {
            load(&mut d, &mut stack, SPACE_I32, lane)?;
            call(&mut d, MAX, [0; 3], &mut stack)?;
            let v = scalar(&mut stack)?;
            Answer::Max((v != i64::MIN).then_some(v as i32))
        }
        Measure::Keep => {
            call(&mut d, KEEP, [0; 3], &mut stack)?;
            Answer::Kept
        }
    };
    let run = Run {
        answer,
        passes: d.facade_passes(),
        programs: d.programs_run(),
    };
    Ok((run, d.last_program().cloned()))
}

fn call(
    d: &mut FoldDialect<'_>,
    f: FnIndex,
    v: [u8; 3],
    s: &mut Vec<Val>,
) -> Result<(), FoldError> {
    d.call(f, v, s)
}

fn load(d: &mut FoldDialect<'_>, s: &mut Vec<Val>, space: u8, idx: u16) -> Result<(), FoldError> {
    s.push(Val::Scalar(i64::from(idx)));
    call(d, LOAD, [space, 0, 0], s)
}

fn scalar(s: &mut Vec<Val>) -> Result<i64, FoldError> {
    match s.pop() {
        Some(Val::Scalar(v)) => Ok(v),
        _ => Err(FoldError::WrongOperandKind(POP_COUNT)),
    }
}

/// The empty population: `RANGE 0 0`, which reads no lane.
fn empty(d: &mut FoldDialect<'_>, s: &mut Vec<Val>) -> Result<(), FoldError> {
    call(d, RANGE, [0, 0, 0], s)
}

/// Push the mask of `rel`.
fn lower_rel(d: &mut FoldDialect<'_>, s: &mut Vec<Val>, rel: &Rel) -> Result<(), FoldError> {
    match rel {
        Rel::Empty => empty(d, s),
        Rel::All => {
            empty(d, s)?;
            owned(d, INT_NOT, s)
        }
        Rel::Plane(p) => load(d, s, SPACE_PLANE, *p),
        Rel::Atom(a) => lower_atom(d, s, *a),
        Rel::Not(c) => {
            lower_rel(d, s, c)?;
            owned(d, INT_NOT, s)
        }
        Rel::And(cs) => fold(d, s, cs, INT_AND, &Rel::All),
        Rel::Or(cs) => fold(d, s, cs, INT_OR, &Rel::Empty),
    }
}

fn fold(
    d: &mut FoldDialect<'_>,
    s: &mut Vec<Val>,
    cs: &[Rel],
    op: FnIndex,
    unit: &Rel,
) -> Result<(), FoldError> {
    let Some((first, rest)) = cs.split_first() else {
        return lower_rel(d, s, unit);
    };
    lower_rel(d, s, first)?;
    for c in rest {
        lower_rel(d, s, c)?;
        owned(d, op, s)?;
    }
    Ok(())
}

/// A mask op whose operands this lowering owns: every subexpression's value
/// is consumed exactly once, so the dialect may release the consumed slots
/// (the contract of `FoldDialect::call_owned`). Width is then free and only
/// depth costs scratch slots.
fn owned(d: &mut FoldDialect<'_>, f: FnIndex, s: &mut Vec<Val>) -> Result<(), FoldError> {
    d.call_owned(f, [0; 3], s)
}

fn lower_atom(d: &mut FoldDialect<'_>, s: &mut Vec<Val>, a: Atom) -> Result<(), FoldError> {
    match a {
        Atom::U32 { lane, eq, v } => {
            load(d, s, SPACE_U32, lane)?;
            s.push(Val::Scalar(i64::from(v)));
            call(d, INT_EQUAL, [0; 3], s)?;
            if !eq {
                owned(d, INT_NOT, s)?;
            }
            Ok(())
        }
        Atom::I32 { lane, op, t } => {
            let t = Val::Scalar(i64::from(t));
            // `IntSLess` is the only ordered compare: `<` is `lane < t`, `>`
            // is `t < lane`, and `<=` / `>=` are the negations of `>` / `<`.
            match op {
                Cmp::Lt | Cmp::Ge => {
                    load(d, s, SPACE_I32, lane)?;
                    s.push(t);
                    call(d, INT_S_LESS, [0; 3], s)?;
                }
                Cmp::Gt | Cmp::Le => {
                    s.push(t);
                    load(d, s, SPACE_I32, lane)?;
                    call(d, INT_S_LESS, [0; 3], s)?;
                }
                Cmp::Eq | Cmp::Ne => {
                    load(d, s, SPACE_I32, lane)?;
                    s.push(t);
                    call(d, INT_EQUAL, [0; 3], s)?;
                }
            }
            if matches!(op, Cmp::Ge | Cmp::Le | Cmp::Ne) {
                owned(d, INT_NOT, s)?;
            }
            Ok(())
        }
    }
}
