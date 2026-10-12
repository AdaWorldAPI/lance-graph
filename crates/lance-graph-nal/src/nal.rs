//! The typed NAL surface: `nal::{Operation}::bind(…)` (plan §4 G).
//!
//! Each operation is a zero-sized type. [`NalOp::lower`] is its procedure: the
//! R2IL calls it emits through [`FoldDialect`], reading its bound coordinates
//! as scalars off the loco stack. `bind` resolves the coordinates on the
//! witnessed order, runs the procedure ONCE and keeps the mask-risc program it
//! produced as a [`Prepared`]; every call after that runs the kept program.
//!
//! None of the operations here earns a byte (plan §4 A): `FN` is `None` for
//! all of them.

use lance_graph_mask_risc::{Foreign, LaneRef, Lowering, Planes, Pred, Program, Scratch, Value};
use lance_graph_r2il::{procs, FoldDialect, FoldError, Val, INT_S_LESS, LOAD};
use ogar_loco::{Dialect, FnIndex};

use crate::{Belief, Bound, NalError, Statement, LANE_C};

/// One NAL operation: its byte, if it earned one, and its procedure.
pub trait NalOp {
    /// The `ogar-nal` byte when the operation earned one, else `None`: the
    /// operation is a procedure over existing R2IL calls.
    const FN: Option<FnIndex>;
    /// Scalars popped, scalars pushed.
    const ARITY: (u8, u8);
    /// Emit the operation's calls. Never loops over rows.
    fn lower(d: &mut FoldDialect<'_>, stack: &mut Vec<Val>) -> Result<(), FoldError>;
}

fn pop_u32(stack: &mut Vec<Val>, f: FnIndex) -> Result<u32, FoldError> {
    match stack.pop() {
        Some(Val::Scalar(v)) => u32::try_from(v).map_err(|_| FoldError::WrongOperandKind(f)),
        Some(_) => Err(FoldError::WrongOperandKind(f)),
        None => Err(FoldError::Underflow(f)),
    }
}

/// `lo hi -- n`: the number of rows in the coordinate range `[lo, hi)`.
///
/// Calls: the range predicate, `PopCount`.
fn range_count(d: &mut FoldDialect<'_>, stack: &mut Vec<Val>) -> Result<(), FoldError> {
    let hi = pop_u32(stack, procs_fn())?;
    let lo = pop_u32(stack, procs_fn())?;
    d.emit_pred(Pred::Range { lo, hi }, stack);
    procs::count(d, stack)
}

/// The function index named in a refusal raised by these procedures.
fn procs_fn() -> FnIndex {
    lance_graph_r2il::POP_COUNT
}

/// A query bound to its coordinates, with the program its procedure produced.
///
/// `extent` is the coordinate range the program runs over. A query whose rows
/// are one range (a statement, a subject) is restricted to it before any lane
/// is read, so its cost follows the range, not the population.
#[derive(Debug, Clone)]
pub struct Prepared {
    program: Program,
    extent: std::ops::Range<usize>,
}

impl Prepared {
    fn of<O: NalOp>(b: &Bound<'_>, args: &[i64]) -> Result<Self, NalError> {
        Self::over::<O>(b, args, 0..b.n_rows())
    }

    fn over<O: NalOp>(
        b: &Bound<'_>,
        args: &[i64],
        extent: std::ops::Range<usize>,
    ) -> Result<Self, NalError> {
        let lanes = [LaneRef::I32(&b.beliefs.c)];
        let planes = Planes {
            n_rows: b.n_rows(),
            masks: &[],
            lanes: &lanes,
        };
        let foreign = Foreign {
            planes: &[],
            lanes: &[],
        };
        let mut d = FoldDialect::new(&planes, &foreign);
        let mut stack: Vec<Val> = args.iter().map(|&v| Val::Scalar(v)).collect();
        O::lower(&mut d, &mut stack)?;
        if let Some(e) = d.poison() {
            return Err(e.into());
        }
        match (stack.as_slice(), d.programs_run(), d.last_program()) {
            ([Val::Scalar(_)], 1, Some(p)) => Ok(Prepared {
                program: p.clone(),
                extent,
            }),
            _ => Err(NalError::NoAnswer),
        }
    }

    /// The scratch this query needs: none for a range fold, one tile per slot
    /// otherwise. Allocate it once and reuse it across calls.
    ///
    /// # Errors
    ///
    /// [`NalError::Exec`] if the program declares more slots than can exist.
    pub fn scratch(&self, b: &Bound<'_>) -> Result<Scratch<'static>, NalError> {
        Ok(Scratch::for_program(&self.program, b.n_rows())?)
    }

    /// How mask-risc folds this query.
    pub fn lowering(&self) -> Lowering {
        self.program.compile().lowering()
    }

    /// The kept program.
    pub fn program(&self) -> &Program {
        &self.program
    }

    /// Run the kept program and read its count.
    ///
    /// # Errors
    ///
    /// [`NalError::Exec`] if `b` is not the population the query was bound on
    /// (a range past its rows), or [`NalError::NoAnswer`].
    pub fn count(&self, b: &Bound<'_>, scratch: &mut Scratch<'_>) -> Result<usize, NalError> {
        match b.run(&self.program, scratch, self.extent.clone())? {
            Value::Count(n) => Ok(n),
            _ => Err(NalError::NoAnswer),
        }
    }
}

/// `<S copula P>?` — how many beliefs the population holds about a statement.
pub struct Question;

impl NalOp for Question {
    const FN: Option<FnIndex> = None;
    const ARITY: (u8, u8) = (2, 1);
    fn lower(d: &mut FoldDialect<'_>, stack: &mut Vec<Val>) -> Result<(), FoldError> {
        range_count(d, stack)
    }
}

impl Question {
    /// Bind the statement's coordinate range.
    ///
    /// # Errors
    ///
    /// [`NalError`] if the procedure is refused.
    pub fn bind(b: &Bound<'_>, st: Statement) -> Result<Prepared, NalError> {
        let (lo, hi) = b.statement(st);
        Prepared::of::<Self>(b, &[i64::from(lo), i64::from(hi)])
    }
}

/// The beliefs of a statement with confidence code `>= t`.
pub struct TruthAtLeast;

impl NalOp for TruthAtLeast {
    const FN: Option<FnIndex> = None;
    const ARITY: (u8, u8) = (2, 1);
    fn lower(d: &mut FoldDialect<'_>, stack: &mut Vec<Val>) -> Result<(), FoldError> {
        range_count(d, stack)
    }
}

impl TruthAtLeast {
    /// Bind the statement's range cut at `t`: confidence is stored inverted,
    /// so the threshold is the end of a prefix of the statement's range.
    ///
    /// # Errors
    ///
    /// [`NalError`] if the procedure is refused.
    pub fn bind(b: &Bound<'_>, st: Statement, t: u8) -> Result<Prepared, NalError> {
        let (lo, hi) = b.statement_at_least(st, t);
        Prepared::of::<Self>(b, &[i64::from(lo), i64::from(hi)])
    }
}

/// Every belief about a subject: the support of `S` in the population.
pub struct Support;

impl NalOp for Support {
    const FN: Option<FnIndex> = None;
    const ARITY: (u8, u8) = (2, 1);
    fn lower(d: &mut FoldDialect<'_>, stack: &mut Vec<Val>) -> Result<(), FoldError> {
        range_count(d, stack)
    }
}

impl Support {
    /// Bind the subject's range.
    ///
    /// # Errors
    ///
    /// [`NalError`] if the procedure is refused.
    pub fn bind(b: &Bound<'_>, s: u16) -> Result<Prepared, NalError> {
        let (lo, hi) = b.subject(s);
        Prepared::of::<Self>(b, &[i64::from(lo), i64::from(hi)])
    }
}

/// Beliefs about a subject with confidence `>= t`, over every statement of
/// that subject. The subject is one range, the threshold is not (each
/// statement restarts the confidence order), so the program reads the `c`
/// lane, and only over the subject's range: the range is the execution
/// extent, never a mask the lane predicate is ANDed with.
pub struct TruthAtLeastAcross;

impl NalOp for TruthAtLeastAcross {
    const FN: Option<FnIndex> = None;
    const ARITY: (u8, u8) = (1, 1);
    /// `t -- n`, counting `t − 1 < c`. Calls: `LOAD`, `IntSLess`, `PopCount`.
    fn lower(d: &mut FoldDialect<'_>, stack: &mut Vec<Val>) -> Result<(), FoldError> {
        let t = pop_u32(stack, procs_fn())?;
        stack.push(Val::Scalar(i64::from(t) - 1));
        stack.push(Val::Scalar(i64::from(LANE_C)));
        d.call(LOAD, [1, 0, 0], stack)?;
        d.call(INT_S_LESS, [0; 3], stack)?;
        procs::count(d, stack)
    }
}

impl TruthAtLeastAcross {
    /// Bind the subject's range and the threshold.
    ///
    /// # Errors
    ///
    /// [`NalError`] if the procedure is refused.
    pub fn bind(b: &Bound<'_>, s: u16, t: u8) -> Result<Prepared, NalError> {
        let (lo, hi) = b.subject(s);
        Prepared::over::<Self>(b, &[i64::from(t)], lo as usize..hi as usize)
    }
}

/// NAL choice: the highest-confidence belief about a statement.
pub struct Choice;

impl Choice {
    /// The first row of the statement's range. Confidence is stored inverted,
    /// so that row has the highest confidence; among equal confidence, the
    /// highest frequency. `None` when the statement has no belief.
    pub fn of(b: &Bound<'_>, st: Statement) -> Option<Belief> {
        let (lo, hi) = b.statement(st);
        (lo < hi).then(|| b.belief(lo as usize)).flatten()
    }
}
