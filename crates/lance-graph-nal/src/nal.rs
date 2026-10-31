//! The typed NAL surface: `nal::{Operation}::bind(…)` (plan §4 G).
//!
//! Each operation is a zero-sized type. [`NalOp::lower`] is its procedure: the
//! R2IL calls it emits through [`FoldDialect`], reading its scalars off the
//! loco stack. `bind` resolves the [`Aperture`] to its range on the witnessed
//! order, runs the procedure ONCE and keeps the mask-risc program it produced
//! as a [`Prepared`]; every call after that runs the kept program over the
//! range.
//!
//! None of the operations here earns a byte (plan §4 A): `FN` is `None` for
//! all of them. [`Revision`] folds through a dialect method rather than a
//! byte ([`FoldDialect::fold_cross_power_sums`]).

use std::ops::Range;

use lance_graph_contract::ordered_lane::OrderedLaneWitness;
use lance_graph_mask_risc::{
    CompiledProgram, CrossPowerSums, Foreign, Lowering, Out, Planes, Pred, Program, Scratch, Value,
};
use lance_graph_r2il::{procs, FoldDialect, FoldError, Val, INT_S_LESS, LOAD, POP_COUNT};
use ogar_loco::FnIndex;

use crate::{
    evidence, Aperture, Bound, NalError, Statement, HORIZON, LANE_C, LANE_EVIDENCE, LANE_F, LANE_W,
    MAX_CONFIDENCE_CODE, WEIGHT_SCALE,
};

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

fn pop_i64(stack: &mut Vec<Val>) -> Result<i64, FoldError> {
    match stack.pop() {
        Some(Val::Scalar(v)) => Ok(v),
        Some(_) => Err(FoldError::WrongOperandKind(POP_COUNT)),
        None => Err(FoldError::Underflow(POP_COUNT)),
    }
}

fn pop_u32(stack: &mut Vec<Val>) -> Result<u32, FoldError> {
    u32::try_from(pop_i64(stack)?).map_err(|_| FoldError::WrongOperandKind(POP_COUNT))
}

/// `lo hi --`, pushing the mask of rows `[lo, hi)`.
fn range(d: &mut FoldDialect<'_>, stack: &mut Vec<Val>) -> Result<(), FoldError> {
    let hi = pop_u32(stack)?;
    let lo = pop_u32(stack)?;
    d.emit_pred(Pred::Range { lo, hi }, stack);
    Ok(())
}

/// A query bound to its range, with the program its procedure produced.
#[derive(Debug, Clone)]
pub struct Prepared {
    program: CompiledProgram,
    extent: Range<usize>,
    /// The population the range belongs to; a run on any other is refused.
    witness: OrderedLaneWitness,
}

impl Prepared {
    /// Run `O`'s procedure once over `extent` with `args` on the stack and keep
    /// the program.
    fn of<O: NalOp>(b: &Bound<'_>, args: &[i64], extent: Range<usize>) -> Result<Self, NalError> {
        let lanes = b.lanes();
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
                program: CompiledProgram::new(p.clone()),
                extent,
                witness: b.witness(),
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
        Ok(Scratch::for_program(self.program.program(), b.n_rows())?)
    }

    /// How mask-risc folds this query.
    pub fn lowering(&self) -> Lowering {
        self.program.lowering()
    }

    /// The kept program.
    pub fn program(&self) -> &Program {
        self.program.program()
    }

    /// The rows the program runs over.
    pub fn extent(&self) -> Range<usize> {
        self.extent.clone()
    }

    fn value(&self, b: &Bound<'_>, scratch: &mut Scratch<'_>) -> Result<Value, NalError> {
        if b.witness() != self.witness {
            return Err(NalError::WrongPopulation);
        }
        Ok(b.run(&self.program, scratch, self.extent.clone(), Out::None)?)
    }

    /// Run a counting query.
    ///
    /// # Errors
    ///
    /// [`NalError::WrongPopulation`] if `b` is not the population the query was
    /// bound on, [`NalError::Exec`] if mask-risc refuses the program, or
    /// [`NalError::NoAnswer`] if the query does not count.
    pub fn count(&self, b: &Bound<'_>, scratch: &mut Scratch<'_>) -> Result<usize, NalError> {
        match self.value(b, scratch)? {
            Value::Count(n) => Ok(n),
            _ => Err(NalError::NoAnswer),
        }
    }

    /// Run a maximum query; `None` for an empty selection.
    ///
    /// # Errors
    ///
    /// As [`Prepared::count`].
    pub fn max(&self, b: &Bound<'_>, scratch: &mut Scratch<'_>) -> Result<Option<i32>, NalError> {
        match self.value(b, scratch)? {
            Value::OptI32(v) => Ok(v),
            _ => Err(NalError::NoAnswer),
        }
    }
}

/// How many beliefs an aperture selects: `<S copula P>?` with a statement
/// aperture, the support of `S` with a subject aperture.
pub struct Question;

impl NalOp for Question {
    const FN: Option<FnIndex> = None;
    const ARITY: (u8, u8) = (2, 1);
    /// `lo hi -- n`. Calls: the range predicate, `PopCount`.
    fn lower(d: &mut FoldDialect<'_>, stack: &mut Vec<Val>) -> Result<(), FoldError> {
        range(d, stack)?;
        procs::count(d, stack)
    }
}

impl Question {
    /// Bind the aperture's range.
    ///
    /// # Errors
    ///
    /// [`NalError`] if the witness or the procedure is refused.
    pub fn bind(b: &Bound<'_>, a: Aperture) -> Result<Prepared, NalError> {
        let (lo, hi) = b.range(a)?;
        Prepared::of::<Self>(b, &[i64::from(lo), i64::from(hi)], 0..b.n_rows())
    }
}

/// The beliefs an aperture selects with confidence code `>= t`. Confidence is
/// a value lane, not part of the name, so this is a lane predicate; it reads
/// only the aperture's range, which is the execution extent.
pub struct TruthAtLeast;

impl NalOp for TruthAtLeast {
    const FN: Option<FnIndex> = None;
    const ARITY: (u8, u8) = (1, 1);
    /// `t -- n`, counting `t − 1 < c`. Calls: `LOAD`, `IntSLess`, `PopCount`.
    fn lower(d: &mut FoldDialect<'_>, stack: &mut Vec<Val>) -> Result<(), FoldError> {
        let t = pop_i64(stack)?;
        stack.push(Val::Scalar(t - 1));
        stack.push(Val::Scalar(i64::from(LANE_C)));
        d.call(LOAD, [1, 0, 0], stack)?;
        d.call(INT_S_LESS, [0; 3], stack)?;
        procs::count(d, stack)
    }
}

impl TruthAtLeast {
    /// Bind the aperture's range and the threshold.
    ///
    /// # Errors
    ///
    /// [`NalError`] if the witness or the procedure is refused.
    pub fn bind(b: &Bound<'_>, a: Aperture, t: u8) -> Result<Prepared, NalError> {
        let (lo, hi) = b.range(a)?;
        Prepared::of::<Self>(b, &[i64::from(t)], lo as usize..hi as usize)
    }
}

/// NAL choice: the highest confidence among the beliefs an aperture selects.
/// Which belief holds it (an argmax) is not a fold mask-risc has yet.
pub struct Choice;

impl NalOp for Choice {
    const FN: Option<FnIndex> = None;
    const ARITY: (u8, u8) = (2, 1);
    /// `lo hi -- max c`. Calls: the range predicate, `LOAD`, `MAX`.
    fn lower(d: &mut FoldDialect<'_>, stack: &mut Vec<Val>) -> Result<(), FoldError> {
        range(d, stack)?;
        procs::max(d, stack, LANE_C)
    }
}

impl Choice {
    /// Bind the aperture's range.
    ///
    /// # Errors
    ///
    /// [`NalError`] if the witness or the procedure is refused.
    pub fn bind(b: &Bound<'_>, a: Aperture) -> Result<Prepared, NalError> {
        let (lo, hi) = b.range(a)?;
        Prepared::of::<Self>(b, &[i64::from(lo), i64::from(hi)], lo as usize..hi as usize)
    }
}

/// NAL revision of one statement: its independent evidence pooled, `w = Σwᵢ`,
/// `f = Σwᵢ·fᵢ / Σwᵢ`, `c = w / (w + k)`. One grouped cross-moment fold over
/// the statement's range, keyed by the evidence class: group 0 is the
/// independent evidence, the others are checked empty (overlap, no
/// provenance) or reported (duplicates, counted once).
pub struct Revision;

impl NalOp for Revision {
    const FN: Option<FnIndex> = None;
    const ARITY: (u8, u8) = (2, 0);
    /// `lo hi --`, the sink holding the moments. Calls: the range predicate,
    /// then [`FoldDialect::fold_cross_power_sums`] with key = evidence class,
    /// `x = w`, `y = f`.
    fn lower(d: &mut FoldDialect<'_>, stack: &mut Vec<Val>) -> Result<(), FoldError> {
        range(d, stack)?;
        d.fold_cross_power_sums(LANE_EVIDENCE, LANE_W, LANE_F, stack)
    }
}

/// A revised truth, with the exact sums it was rounded from once.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Revised {
    /// Frequency code, `round(Σw·f / Σw)`.
    pub f: u8,
    /// Confidence code, `round(255 · Σw / (Σw + k·WEIGHT_SCALE))`, capped at
    /// [`MAX_CONFIDENCE_CODE`].
    pub c: u8,
    /// `Σw` of the independent evidence, at [`WEIGHT_SCALE`].
    pub w: i64,
    /// `Σw·f`, `f` in codes.
    pub wf: i128,
    /// Independent pieces pooled.
    pub rows: u64,
    /// Exact duplicates seen and counted once.
    pub duplicates: u64,
}

/// `round(a / b)` for `a ≥ 0`, `b > 0`.
fn round_div(a: i128, b: i128) -> i128 {
    (2 * a + b) / (2 * b)
}

/// The commit law over the four evidence groups. Rounds once.
///
/// This is the exact NAL revision `w = Σwᵢ`, capped only at the end. It
/// matches ONA's pairwise `Truth_Revision` until ONA's running confidence hits
/// its 0.99 cap; past that ONA turns the capped confidence back into a bounded
/// weight, so its answer depends on arrival order (ten `(255, 230)` then ten
/// `(0, 230)` give f = 104, the reverse order 151). A fold over a set cannot
/// have an order, so it keeps the exact sum (f = 128) — a documented
/// divergence, pinned in `tests/w3.rs`.
fn commit(g: &[CrossPowerSums; evidence::CLASSES]) -> Result<Option<Revised>, NalError> {
    let overlap = g[evidence::OVERLAP as usize].n;
    if overlap > 0 {
        return Err(NalError::DependentEvidence { rows: overlap });
    }
    let unprovenanced = g[evidence::NO_PROVENANCE as usize].n;
    if unprovenanced > 0 {
        return Err(NalError::MissingProvenance {
            rows: unprovenanced,
        });
    }
    let ind = &g[evidence::INDEPENDENT as usize];
    if ind.sum_x <= 0 {
        return Ok(None);
    }
    let w = i128::from(ind.sum_x);
    let f = round_div(ind.sum_xy, w);
    let c = round_div(255 * w, w + i128::from(HORIZON) * i128::from(WEIGHT_SCALE));
    Ok(Some(Revised {
        f: u8::try_from(f).unwrap_or(u8::MAX),
        c: u8::try_from(c).unwrap_or(u8::MAX).min(MAX_CONFIDENCE_CODE),
        w: ind.sum_x,
        wf: ind.sum_xy,
        rows: ind.n,
        duplicates: g[evidence::DUPLICATE as usize].n,
    }))
}

/// A revision bound to its statement's range, with the program its
/// procedure produced.
#[derive(Debug, Clone)]
pub struct PreparedRevision {
    program: CompiledProgram,
    extent: Range<usize>,
    witness: OrderedLaneWitness,
}

impl Revision {
    /// Bind the statement's range and keep the fold.
    ///
    /// # Errors
    ///
    /// [`NalError`] if the witness or the procedure is refused.
    pub fn bind(b: &Bound<'_>, st: Statement) -> Result<PreparedRevision, NalError> {
        let (lo, hi) = b.range(Aperture::statement(b.class(), st))?;
        let lanes = b.lanes();
        let planes = Planes {
            n_rows: b.n_rows(),
            masks: &[],
            lanes: &lanes,
        };
        let foreign = Foreign {
            planes: &[],
            lanes: &[],
        };
        let mut sink = [CrossPowerSums::default(); evidence::CLASSES];
        let mut d = FoldDialect::with_cross_sums(&planes, &foreign, &mut sink);
        let mut stack = vec![Val::Scalar(i64::from(lo)), Val::Scalar(i64::from(hi))];
        Self::lower(&mut d, &mut stack)?;
        if let Some(e) = d.poison() {
            return Err(e.into());
        }
        match (stack.is_empty(), d.programs_run(), d.last_program()) {
            (true, 1, Some(p)) => Ok(PreparedRevision {
                program: CompiledProgram::new(p.clone()),
                extent: lo as usize..hi as usize,
                witness: b.witness(),
            }),
            _ => Err(NalError::NoAnswer),
        }
    }
}

impl PreparedRevision {
    /// The scratch this revision needs. Allocate it once and reuse it.
    ///
    /// # Errors
    ///
    /// [`NalError::Exec`] if the program declares more slots than can exist.
    pub fn scratch(&self, b: &Bound<'_>) -> Result<Scratch<'static>, NalError> {
        Ok(Scratch::for_program(self.program.program(), b.n_rows())?)
    }

    /// The kept program.
    pub fn program(&self) -> &Program {
        self.program.program()
    }

    /// Run the fold and commit: the revised truth, `None` without evidence.
    ///
    /// # Errors
    ///
    /// [`NalError::WrongPopulation`] on another population;
    /// [`NalError::DependentEvidence`] / [`NalError::MissingProvenance`] when
    /// the statement holds evidence that cannot be revised exactly;
    /// [`NalError::Exec`] if mask-risc refuses the program.
    pub fn revise(
        &self,
        b: &Bound<'_>,
        scratch: &mut Scratch<'_>,
    ) -> Result<Option<Revised>, NalError> {
        if b.witness() != self.witness {
            return Err(NalError::WrongPopulation);
        }
        let mut sink = [CrossPowerSums::default(); evidence::CLASSES];
        match b.run(
            &self.program,
            scratch,
            self.extent.clone(),
            Out::CrossPowerSums(&mut sink),
        )? {
            Value::GroupCrossPowerSums => commit(&sink),
            _ => Err(NalError::NoAnswer),
        }
    }
}
