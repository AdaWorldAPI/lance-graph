//! Procedures: named R2IL call sequences over [`FoldDialect`].
//!
//! A procedure is a Rust function that emits R2IL calls through
//! [`Dialect::call`] and pushes literals onto the loco stack. It adds no
//! opcode and no lowering: what reaches mask-risc is exactly what the same
//! calls written out by hand would produce. This is the inlined form of
//! D-SQLP-2. An executable `PROC_DEF`/`PROC_CALL` contract (per-call arity,
//! call frames, `PROC_ARG`, returns) is an OPEN item in OGAR; these functions
//! are what such a body would contain.
//!
//! Stack conventions: a "mask" is a [`Val::Slot`] or a resident plane
//! address on top of the stack. Lane arguments are indexes into
//! [`Planes::lanes`](lance_graph_mask_risc::Planes::lanes).

use ogar_loco::{Dialect, FnIndex};
use ogar_r2il::{GROUP_SUM, KEEP, SUM};

use crate::{FoldDialect, FoldError, Val, INT_SUB, INT_S_LESS, LOAD};

/// A selection that a procedure re-emits once per fold (see [`margin`]).
pub type Select<'s> = dyn Fn(&mut FoldDialect<'_>, &mut Vec<Val>) -> Result<(), FoldError> + 's;

/// `LOAD` space for a `U32` lane.
const SPACE_U32: u8 = 0;
/// `LOAD` space for an `I32` lane.
const SPACE_I32: u8 = 1;

fn call(
    d: &mut FoldDialect<'_>,
    f: FnIndex,
    v: [u8; 3],
    stack: &mut Vec<Val>,
) -> Result<(), FoldError> {
    d.call(f, v, stack)
}

/// Push the address of lane `idx` in `space`.
fn load(
    d: &mut FoldDialect<'_>,
    stack: &mut Vec<Val>,
    space: u8,
    idx: u16,
) -> Result<(), FoldError> {
    stack.push(Val::Scalar(i64::from(idx)));
    call(d, LOAD, [space, 0, 0], stack)
}

/// `lane < t` on an `I32` lane. Pushes the mask.
///
/// Calls: `LOAD`, `IntSLess`.
pub fn select_lt(
    d: &mut FoldDialect<'_>,
    stack: &mut Vec<Val>,
    lane: u16,
    t: i32,
) -> Result<(), FoldError> {
    load(d, stack, SPACE_I32, lane)?;
    stack.push(Val::Scalar(i64::from(t)));
    call(d, INT_S_LESS, [0; 3], stack)
}

/// The rows where `lane < t`, written into the dialect's `KEEP` sink.
/// Pushes nothing.
///
/// Calls: `LOAD`, `IntSLess`, `KEEP`.
pub fn filter_lt(
    d: &mut FoldDialect<'_>,
    stack: &mut Vec<Val>,
    lane: u16,
    t: i32,
) -> Result<(), FoldError> {
    select_lt(d, stack, lane, t)?;
    call(d, KEEP, [0; 3], stack)
}

/// `SUM(val_lane | mask)` for the mask on top of the stack. Pushes the sum.
///
/// Calls: `LOAD`, `SUM`.
pub fn sum(d: &mut FoldDialect<'_>, stack: &mut Vec<Val>, val_lane: u16) -> Result<(), FoldError> {
    load(d, stack, SPACE_I32, val_lane)?;
    call(d, SUM, [0; 3], stack)
}

/// Per-group `SUM(val_lane | mask)` keyed by the `U32` lane `key_lane`, for
/// the mask on top of the stack, into the dialect's group sink. Pushes
/// nothing.
///
/// Calls: `LOAD`, `LOAD`, `GROUP_SUM`.
pub fn group_sum(
    d: &mut FoldDialect<'_>,
    stack: &mut Vec<Val>,
    key_lane: u16,
    val_lane: u16,
) -> Result<(), FoldError> {
    load(d, stack, SPACE_U32, key_lane)?;
    load(d, stack, SPACE_I32, val_lane)?;
    call(d, GROUP_SUM, [0; 3], stack)
}

/// `SUM(revenue | sel) - SUM(cost | sel)`. Pushes the difference.
///
/// Each `SUM` is its own fold, and a fold ends the epoch of every slot built
/// before it, so `select` is emitted once per `SUM` rather than held across
/// the first one.
///
/// Calls: `select`, `LOAD`, `SUM`, `select`, `LOAD`, `SUM`, `IntSub`.
pub fn margin(
    d: &mut FoldDialect<'_>,
    stack: &mut Vec<Val>,
    select: &Select<'_>,
    revenue: u16,
    cost: u16,
) -> Result<(), FoldError> {
    select(d, stack)?;
    sum(d, stack, revenue)?;
    select(d, stack)?;
    sum(d, stack, cost)?;
    call(d, INT_SUB, [0; 3], stack)
}
