//! D-MPF-1: the mathematical polyfill, as it already exists.
//!
//! The Stage 1 polyfill API is not new code. It is the typed procedure layer
//! over R2IL that already ships in [`lance_graph_r2il::procs`] (selection,
//! comparison, NULL presence, `COUNT` / `SUM` / `MIN` / `MAX`, Boolean set
//! operations) plus the checked scalar arithmetic of
//! [`FoldDialect`] (`IntAdd`, `IntSub`, `IntMult`, `IntSDiv`). A procedure
//! emits R2IL calls only; mask-risc executes them. This crate adds no opcode,
//! no IR, no executor.
//!
//! What this crate holds:
//!
//! - [`Expr`]: one small expression, written ONLY with those procedures:
//!   `SUM(v | sel) − k·COUNT(sel)` with
//!   `sel = valid(v) ∧ x ≥ lo ∧ y < hi`.
//! - [`eval`]: lowers it through the procedures onto a lent [`FoldDialect`].
//! - [`oracle`]: an independent row loop over the same lanes, in `i128`, that
//!   never touches the dialect, mask-risc or the procedures.
//!
//! Semantics preserved, and where each is decided:
//!
//! - Domain: `i32` lanes, `i64` results. Integer only; there is no float lane
//!   in the fold path, so no IEEE claim is made.
//! - NULL: `v`'s presence plane. A row whose `v` is NULL is outside `sel`
//!   (SQL `WHERE` drops UNKNOWN); it is not counted and not summed.
//! - Multiplicity: rows are a set of row positions; `COUNT` is the popcount
//!   of the selection mask.
//! - Overflow: refused, never wrapped (`FoldError::Overflow`).
//! - Exactness: the answer is exact, and must equal [`oracle`] bit for bit.

pub mod native;
pub mod onnx_compat;

use lance_graph_r2il::procs::{and_cmp_i32, count, is_not_null, sum, Order};
use lance_graph_r2il::{FoldDialect, FoldError, Val, INT_MULT, INT_SUB};

/// Lane and plane indexes the expression reads.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Layout {
    /// `I32` lane `x`.
    pub x: u16,
    /// `I32` lane `y`.
    pub y: u16,
    /// `I32` lane `v`, the summed value.
    pub v: u16,
    /// Resident mask plane: rows where `v` is present (not NULL).
    pub valid: u16,
}

/// `SUM(v | sel) − k·COUNT(sel)`, `sel = valid(v) ∧ x ≥ lo ∧ y < hi`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Expr {
    /// Lower bound on `x`, inclusive.
    pub lo: i32,
    /// Upper bound on `y`, exclusive.
    pub hi: i32,
    /// The scalar multiplier on the count.
    pub k: i64,
}

/// Push the selection mask. Emitted once per fold: a fold ends the epoch of
/// every slot built before it (see `procs::margin`).
fn select(d: &mut FoldDialect<'_>, s: &mut Vec<Val>, l: Layout, e: Expr) -> Result<(), FoldError> {
    is_not_null(d, s, l.valid)?;
    and_cmp_i32(d, s, l.x, Order::Ge, e.lo)?;
    and_cmp_i32(d, s, l.y, Order::Lt, e.hi)
}

/// Evaluate `e` on the dialect's planes. Every step is an existing procedure
/// or an existing scalar R2IL call.
pub fn eval(d: &mut FoldDialect<'_>, l: Layout, e: Expr) -> Result<i64, FoldError> {
    let mut s = Vec::with_capacity(8);
    select(d, &mut s, l, e)?;
    sum(d, &mut s, l.v)?;
    select(d, &mut s, l, e)?;
    count(d, &mut s)?;
    s.push(Val::Scalar(e.k));
    d.call(INT_MULT, [0; 3], &mut s)?;
    d.call(INT_SUB, [0; 3], &mut s)?;
    match s.pop() {
        Some(Val::Scalar(v)) if s.is_empty() => Ok(v),
        _ => Err(FoldError::WrongOperandKind(INT_SUB)),
    }
}

/// The rows the selection keeps, and the answer, computed row by row.
/// `None` when the exact answer does not fit `i64` (the dialect must refuse).
pub fn oracle(x: &[i32], y: &[i32], v: &[i32], valid: &[u64], e: Expr) -> (u64, Option<i64>) {
    let mut sum: i128 = 0;
    let mut n: u64 = 0;
    for r in 0..v.len() {
        let present = valid[r / 64] >> (r % 64) & 1 == 1;
        if present && x[r] >= e.lo && y[r] < e.hi {
            sum += i128::from(v[r]);
            n += 1;
        }
    }
    let ans = sum - i128::from(e.k) * i128::from(n);
    (n, i64::try_from(ans).ok())
}
