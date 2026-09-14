//! The result and error vocabulary shared by the executor and the oracle.
//!
//! Both `exec` and `reference` produce exactly these — the differential suite
//! compares them with `==`, so an executor cannot "succeed" where the oracle
//! rejects, or reject with a different reason.

use crate::ir::Operand;

/// What a [`crate::Program`] produced.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Value {
    /// [`crate::Terminal::Keep`]: the final mask stays in this operand
    /// (a scratch slot the caller reads back, or the input plane itself).
    Mask(Operand),
    /// [`crate::Terminal::Count`].
    Count(usize),
    /// [`crate::Terminal::Any`] / [`crate::Terminal::All`].
    Bool(bool),
    /// [`crate::Terminal::MaskedSumI32`].
    SumI64(i64),
    /// [`crate::Terminal::MaskedMinI32`] / [`crate::Terminal::MaskedMaxI32`].
    OptI32(Option<i32>),
    /// [`crate::Terminal::BlendI32`]: the caller's `out` slice was written.
    Blended,
}

/// The lane width a predicate or terminal expects, for [`ExecError::LaneKind`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LaneKind {
    /// `LaneRef::I32`.
    I32,
    /// `LaneRef::U32`.
    U32,
    /// `LaneRef::U64`.
    U64,
}

/// Why a program could not run. Validation is total and happens BEFORE any
/// op writes: a rejected program leaves scratch and `out` untouched.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExecError {
    /// The caller's scratch has fewer slots than [`crate::Program::scratch_slots`].
    ScratchTooSmall { need: u32, have: usize },
    /// The caller's scratch words do not match `words_for(planes.n_rows)`.
    ScratchWords { expected: usize, found: usize },
    /// `Operand::Plane(i)` with `i >= planes.masks.len()`.
    PlaneOutOfRange(u16),
    /// A predicate or terminal names `lane >= planes.lanes.len()`.
    LaneOutOfRange(u16),
    /// The lane exists but has the wrong width.
    LaneKind {
        lane: u16,
        expected: LaneKind,
        found: LaneKind,
    },
    /// A plane, lane, or `out` slice is not `n_rows` (or `words_for(n_rows)`) long.
    LenMismatch {
        what: &'static str,
        expected: usize,
        found: usize,
    },
    /// [`crate::Terminal::BlendI32`] without a caller `out` slice.
    BlendNeedsOut,
    /// [`crate::Terminal::MaskedSumI32`] over more rows than
    /// [`crate::MASKED_SUM_I32_MAX_ROWS`] — the executor refuses rather than wraps.
    SumRowBound { n_rows: usize },
    /// Input plane `i` carries set bits past `n_rows`. The executor works on
    /// whole words and would read them; the oracle reads rows and would not —
    /// so a dirty tail is refused rather than let the two diverge.
    PlaneTail(u16),
    /// A gated predicate whose gate IS its destination: the facade cannot
    /// read the gate while overwriting it, so the program is refused.
    GateAliasesDst { dst: u16 },
}
