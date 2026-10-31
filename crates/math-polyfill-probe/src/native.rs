//! One native polyfill operation, with no ONNX type anywhere in it.
//!
//! [`SumWhere`]: `Σ { x[r] : x present at r ∧ x[r] ⋈ t }`, exact, in `i64`.
//! Inputs are a lane index and an optional presence plane into the caller's
//! [`Planes`](lance_graph_mask_risc::Planes); the operation is lowered onto a
//! lent [`FoldDialect`] with existing R2IL calls only (`LOAD`, `IntSLess`,
//! `IntNot`, `IntAnd`, `SUM`) and executed by mask-risc.
//!
//! Binding (building `Planes` and the dialect) is the caller's, so a hot loop
//! reuses one dialect and pays only for the fold.

use lance_graph_r2il::procs::{is_not_null, sum};
use lance_graph_r2il::{FoldDialect, FoldError, Val, INT_AND, INT_NOT, INT_S_LESS, LOAD};

/// `LOAD` space of an `I32` lane, as `FoldDialect` reads it.
const SPACE_I32: u8 = 1;

/// The comparison of each element against the threshold.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Pred {
    /// `x > t`
    Gt(i32),
    /// `x >= t`
    Ge(i32),
    /// `x < t`
    Lt(i32),
    /// `x <= t`
    Le(i32),
}

impl Pred {
    /// The predicate on one value: the reference the oracle uses.
    pub fn holds(self, x: i32) -> bool {
        match self {
            Pred::Gt(t) => x > t,
            Pred::Ge(t) => x >= t,
            Pred::Lt(t) => x < t,
            Pred::Le(t) => x <= t,
        }
    }
}

/// `Σ { lane[r] : present(r) ∧ pred(lane[r]) }`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SumWhere {
    /// `I32` lane index.
    pub lane: u16,
    /// Presence plane of the lane, if it has NULLs.
    pub present: Option<u16>,
    /// Which elements count.
    pub pred: Pred,
}

impl SumWhere {
    /// Lower and run. `stack` is caller-owned so a hot loop allocates nothing.
    pub fn eval(self, d: &mut FoldDialect<'_>, stack: &mut Vec<Val>) -> Result<i64, FoldError> {
        stack.clear();
        // `IntSLess` is the only ordered compare: `x < t` and `t < x`, and the
        // inclusive forms are their negations.
        let (t, lane_first, negate) = match self.pred {
            Pred::Lt(t) => (t, true, false),
            Pred::Ge(t) => (t, true, true),
            Pred::Gt(t) => (t, false, false),
            Pred::Le(t) => (t, false, true),
        };
        let t = Val::Scalar(i64::from(t));
        let lane = Val::Scalar(i64::from(self.lane));
        if lane_first {
            stack.push(lane);
            d.call(LOAD, [SPACE_I32, 0, 0], stack)?;
            stack.push(t);
        } else {
            stack.push(t);
            stack.push(lane);
            d.call(LOAD, [SPACE_I32, 0, 0], stack)?;
        }
        d.call(INT_S_LESS, [0; 3], stack)?;
        if negate {
            d.call(INT_NOT, [0; 3], stack)?;
        }
        if let Some(p) = self.present {
            is_not_null(d, stack, p)?;
            d.call(INT_AND, [0; 3], stack)?;
        }
        sum(d, stack, self.lane)?;
        match stack.pop() {
            Some(Val::Scalar(v)) if stack.is_empty() => Ok(v),
            _ => Err(FoldError::WrongOperandKind(INT_AND)),
        }
    }
}

/// The reference: a row loop in `i128`, sharing no code with [`SumWhere`].
pub fn sum_where_oracle(x: &[i32], present: Option<&[u64]>, pred: Pred) -> i128 {
    (0..x.len())
        .filter(|&r| present.is_none_or(|p| p[r / 64] >> (r % 64) & 1 == 1))
        .filter(|&r| pred.holds(x[r]))
        .map(|r| i128::from(x[r]))
        .sum()
}
