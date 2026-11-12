//! Selection folds: two folds over a borrowed list of row positions.
//!
//! Decision D-DR-SELECT-K-SV (plan `dr-select-k-v1` §13) admits one operand
//! the [`crate::Program`] vocabulary cannot express: a selection in its
//! position form. This module is that admission and nothing more. It is a
//! separate entry beside [`crate::execute_into`]; [`crate::Program`],
//! [`crate::Planes`] and every existing call site are unchanged.
//!
//! # What the operand is
//!
//! `positions: &[u32]` is a list of rows **of the frame the planes describe**
//! (`0..planes.n_rows`), borrowed from whoever holds the bound selection. It
//! is a bag, in list order: a row listed twice is read twice, which a mask
//! cannot say. The executor never converts it to a mask and never builds one.
//!
//! # What it is not
//!
//! - **Not a second row universe.** A position is a row of this frame or the
//!   fold is refused ([`ExecError::SelOutOfRange`]). There is no foreign
//!   plane here and no key; nothing is read from another table.
//! - **Not a position read out of a value lane.** The list is handed in by
//!   the caller's binding. No fold here takes a position from the values of
//!   another lane, which is the shape `graph_hop` is quarantined for.
//! - **Not a second executor.** Each fold is one delegation to one
//!   `ndarray::simd` function (law L3), allocates nothing (L1), names no ISA
//!   (L2), and has an independent oracle with shared validation (L4).
//! - **Not version-aware.** Like [`crate::Planes`], this layer knows a frame
//!   only as `n_rows`. Whether the list was witnessed for the version the
//!   lanes come from is the binding's check, made before the call.
//!
//! # Overflow
//!
//! [`SelFold::SumI32`] follows [`crate::Terminal::MaskedSumI32`]: the result
//! is exact or the fold is refused. A bag may be longer than the frame, so
//! the bound is on the list ([`SEL_SUM_I32_MAX_POSITIONS`]), not on `n_rows`.
//! Within it the facade's wrapping accumulation cannot wrap.

use ndarray::simd::{count_gt_i32_at, sum_i32_at};

use crate::ir::{LaneRef, Planes, MASKED_SUM_I32_MAX_ROWS};
use crate::reference::{first_out_of_range, validate_sel_frame};
use crate::value::{ExecError, Value};

/// The most positions [`SelFold::SumI32`] accepts: `2^32`, the same count
/// [`MASKED_SUM_I32_MAX_ROWS`] allows a masked sum, for the same reason — an
/// `i64` holds that many copies of `i32::MIN` or `i32::MAX` exactly.
pub const SEL_SUM_I32_MAX_POSITIONS: usize = MASKED_SUM_I32_MAX_ROWS;

/// The folds a position selection admits. Each reads one `i32` lane of
/// [`Planes::lanes`] at the listed rows.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SelFold {
    /// How many listed positions hold a value greater than `t` (signed,
    /// strict). Yields [`Value::Count`].
    CountGtI32 { lane: u16, t: i32 },
    /// Σ `lane[p]` over the listed positions, widened to `i64`. Yields
    /// [`Value::SumI64`].
    SumI32 { lane: u16 },
}

/// Runs one selection fold over `positions`, a borrowed bag of rows of the
/// frame `planes` describes.
///
/// Refusals, checked before anything is computed and identical to
/// [`crate::reference_execute_sel`]'s: the lane is missing
/// ([`ExecError::LaneOutOfRange`]), is not `i32` ([`ExecError::LaneKind`]) or
/// does not span the frame ([`ExecError::LenMismatch`]); a sum's list is over
/// its bound ([`ExecError::SelSumBound`]); a position is not a row of the
/// frame ([`ExecError::SelOutOfRange`], naming the first).
///
/// Never allocates. Mask planes of `planes` are not read.
pub fn execute_sel(
    fold: SelFold,
    planes: &Planes<'_>,
    positions: &[u32],
) -> Result<Value, ExecError> {
    validate_sel_frame(fold, planes, positions)?;
    let lane = match fold {
        SelFold::CountGtI32 { lane, .. } | SelFold::SumI32 { lane } => lane,
    };
    // `validate_sel_frame` established that the lane exists, is `i32` and
    // spans the frame, so "below the lane's length" is "a row of the frame".
    let LaneRef::I32(values) = planes.lanes[usize::from(lane)] else {
        unreachable!("validate_sel_frame admits only an i32 lane")
    };
    let value = match fold {
        SelFold::CountGtI32 { t, .. } => {
            count_gt_i32_at(values, positions, t).map(|c| Value::Count(c as usize))
        }
        SelFold::SumI32 { .. } => sum_i32_at(values, positions).map(Value::SumI64),
    };
    // The facade reports an out-of-range position as `None`; the walk that
    // names the offender runs only on that refusing path.
    value.ok_or_else(|| {
        let (at, position) = first_out_of_range(planes.n_rows, positions)
            .expect("the facade refused, so a position is out of range");
        ExecError::SelOutOfRange { at, position }
    })
}
