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
    /// [`crate::Terminal::ScatterOrU32`]: the caller's `Out::Mask` buffer was
    /// written.
    Scattered,
    /// [`crate::Terminal::GroupSumI32`]: the caller's `Out::I64` buffer was
    /// written, one slot per group.
    GroupSummed,
}

/// The caller's terminal-result destination — one variant per shape a
/// [`crate::Terminal`] may write into, plus `None` for a terminal that writes
/// nothing (every reducer that returns its answer as a [`Value`] instead).
#[derive(Debug, PartialEq, Eq)]
pub enum Out<'a> {
    /// No destination buffer is needed for this terminal.
    None,
    /// [`crate::Terminal::BlendI32`]'s destination, `n_rows` long.
    I32(&'a mut [i32]),
    /// [`crate::Terminal::GroupSumI32`]'s destination — one `i64` per group,
    /// its length IS the group universe `K`.
    I64(&'a mut [i64]),
    /// [`crate::Terminal::ScatterOrU32`]'s destination, `words_for(out_rows)`
    /// long.
    Mask(&'a mut [u64]),
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
    /// The CALLER's scratch has fewer slots than
    /// [`crate::Program::scratch_slots`]. `have` is the caller's slot count.
    ScratchTooSmall { need: u32, have: usize },
    /// The buffer handed to [`crate::Scratch::over`] is shorter than the
    /// layout needs, reported in WORDS.
    ///
    /// Distinct from [`Self::ScratchTooSmall`], which counts SLOTS. A buffer
    /// can be long enough in words for the wrong number of slots, or hold
    /// enough slots at the wrong width, and reporting one number for both
    /// would hand the caller a figure it cannot size from. The needed value
    /// is exactly [`crate::scratch_words_for`].
    ScratchBufferTooSmall {
        need_words: usize,
        have_words: usize,
    },
    /// The PROGRAM names a scratch slot beyond its own declared
    /// `scratch_slots` — a hand-built program whose count does not cover its
    /// operands (`Program::new` computes a covering count, so this reports a
    /// program that was assembled by hand and lied). Distinct from
    /// [`Self::ScratchTooSmall`], which is about the caller's buffer.
    ScratchSlotUndeclared { slot: u16, declared: u32 },
    /// The PROGRAM declares more scratch slots than `Operand::Scratch` can
    /// address ([`crate::MAX_SCRATCH_SLOTS`] = 65,536). Nothing in the
    /// program could reach the surplus, so the count is a lie — and sizing an
    /// arena from it would allocate unboundedly on the strength of one
    /// public field. Refused before any allocation, by the validator BOTH
    /// paths share, so the executor and the oracle refuse identically.
    ScratchSlotsUnaddressable { declared: u32 },
    /// An op or the terminal READS a scratch slot no earlier op wrote. The
    /// executor would see whatever the caller's reused buffer holds; the
    /// oracle models a fresh arena. Rather than let the two diverge, the
    /// program is refused — pre-filled scratch is a named PR5 gap, not a
    /// supported input.
    ScratchReadBeforeWrite { slot: u16 },
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
    /// [`crate::Pred::Range`] with `lo > hi` or `hi > n_rows`. Refused by the
    /// validator both paths share, so the executor never hands
    /// `mask_set_range` a bound past the scratch words (its own assert would
    /// panic) and the oracle never indexes a row that does not exist.
    RangeOutOfBounds { lo: u32, hi: u32, n_rows: usize },
    /// [`crate::Terminal::CountKeyRunsU32`] met a key smaller than the open
    /// run's key: the lane is not in key order, so a run is not a key and
    /// the count would be wrong. Refused at that element, with O(1) state —
    /// non-decreasing order is the clustering certificate the fold can
    /// check in its own pass. The logical query is fine; THIS lowering
    /// needs a lane stored in key order (a T0 address projection), and no
    /// such projection is resident for this lane.
    LaneNotOrdered { lane: u16 },
    /// [`crate::MaskOp::Gather`]'s `foreign` names no entry of the caller's
    /// [`crate::Foreign::planes`].
    ForeignOutOfRange(u16),
    /// [`crate::Terminal::GroupSumViaI32`]'s `key` names no entry of the
    /// caller's [`crate::Foreign::lanes`]. Distinct from [`Self::LaneOutOfRange`]
    /// — that one bounds a program's OWN `Planes::lanes`, this one bounds
    /// the SEPARATE foreign address space, and the two must never be
    /// checked against the same length.
    ForeignLaneOutOfRange(u16),
    /// The foreign lane [`crate::Terminal::GroupSumViaI32::key`] names
    /// exists, but at the wrong width — mirrors [`Self::LaneKind`] over
    /// [`crate::Foreign::lanes`] rather than `Planes::lanes`.
    ForeignLaneKind {
        lane: u16,
        expected: LaneKind,
        found: LaneKind,
    },
    /// A terminal's [`crate::Out`] is missing, or is the wrong SHAPE
    /// (`Out::I32` where an `Out::I64` was needed, and so on) — or, for
    /// [`crate::Terminal::ScatterOrU32`], sized to the wrong `out_rows`, or
    /// for [`crate::Terminal::GroupSumI32`], the empty `Out::I64([])`
    /// (`K == 0` names no group at all). `what` names the terminal.
    /// [`crate::Terminal::BlendI32`] keeps its own [`Self::BlendNeedsOut`]
    /// rather than routing through here — two spellings of "needs an `out`",
    /// kept apart because `BlendI32` predates `Out` and every existing
    /// caller already matches on the old variant.
    TerminalNeedsOut { what: &'static str },
}
