//! The three operations over a bound selection, per admissible view.
//!
//! SKELETON: signatures and contracts are fixed; bodies are `todo!()`.

use crate::bind::{Bound, Order, Residual, View};
use crate::counters::Counters;

/// The operations the lab compares.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Op {
    /// `count(residual AND b > t)`.
    CountBGt { t: i32 },
    /// `sum(a)` over the residual, as `i64`.
    SumA,
    /// The first `limit` guids of the residual in key order, ties by guid.
    TopByKey { limit: usize },
}

/// Why a view may not answer an operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Inadmissible {
    /// The operation needs key order and the binding does not witness it.
    /// A mask or ascending positions keep the local order only.
    OrderNotWitnessed { have: Order },
}

/// Whether `view` over `bound` can lawfully answer `op`. `CountBGt` and
/// `SumA` are order-free and always admissible. `TopByKey` is admissible
/// only when `bound.order() == Order::KeyThenGuid`, for every view; nothing
/// in this crate sorts at query time.
pub fn admissible(op: Op, bound: &Bound, view: View<'_>) -> Result<(), Inadmissible> {
    let _ = (op, bound, view);
    todo!()
}

/// Build the residual `a >= t` over the K local rows in the form of `like`:
/// `Residual::All` is returned unchanged (it ignores `t`), `Positions`
/// yields ascending local positions, `Mask` yields `ceil(K/64)` words.
/// Bumps `conversions` by one and `bytes_copied` by the bytes written, for
/// the two stored forms.
pub fn residual_a_ge(bound: &Bound, t: i32, like: &Residual, counters: &Counters) -> Residual {
    let _ = (bound, t, like, counters);
    todo!()
}

/// Compiled mask-RISC programs and scratch for one binding: built once,
/// reused by every fold over it.
pub struct PreparedFolds {
    _private: (),
}

impl PreparedFolds {
    /// Prepare the programs for `bound` (sized for its K rows).
    pub fn new(bound: &Bound) -> PreparedFolds {
        let _ = bound;
        todo!()
    }

    /// `count(residual AND b > t)`. Bumps `operand_rows` by the rows handed
    /// to the fold: K for `All` and `Mask`, the position count for
    /// `Positions`.
    pub fn count_b_gt(
        &mut self,
        bound: &Bound,
        view: View<'_>,
        t: i32,
        counters: &Counters,
    ) -> u64 {
        let _ = (bound, view, t, counters);
        todo!()
    }

    /// `sum(a)` over the residual. Same counting as [`Self::count_b_gt`].
    pub fn sum_a(&mut self, bound: &Bound, view: View<'_>, counters: &Counters) -> i64 {
        let _ = (bound, view, counters);
        todo!()
    }

    /// The first `limit` guids of the residual in key order. Appends to
    /// `out` (cleared first). Returns `Err` exactly when [`admissible`] does.
    pub fn top_by_key(
        &mut self,
        bound: &Bound,
        view: View<'_>,
        limit: usize,
        out: &mut Vec<u128>,
        counters: &Counters,
    ) -> Result<(), Inadmissible> {
        let _ = (bound, view, limit, out, counters);
        todo!()
    }
}

/// The resident baseline: the same answers folded over all N source rows
/// with an N-bit selection plane, no binding. `selected` is one bit per
/// source row (`key in range AND a >= t_a`), built by the caller once.
pub mod baseline {
    use crate::counters::Counters;
    use crate::source::Source;

    /// `count(selected AND b > t)` over N rows.
    pub fn count_b_gt(source: &Source, selected: &[u64], t: i32, counters: &Counters) -> u64 {
        let _ = (source, selected, t, counters);
        todo!()
    }

    /// `sum(a)` over the selected source rows.
    pub fn sum_a(source: &Source, selected: &[u64], counters: &Counters) -> i64 {
        let _ = (source, selected, counters);
        todo!()
    }
}
