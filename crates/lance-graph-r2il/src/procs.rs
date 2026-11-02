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

use ogar_loco::FnIndex;
use ogar_r2il::{ALL, ANY, GROUP_SUM, KEEP, KEY_RUNS, MAX, MIN, SUM};

use crate::{
    FoldDialect, FoldError, Val, INT_ADD, INT_AND, INT_EQUAL, INT_NOT, INT_OR, INT_SUB, INT_S_LESS,
    INT_XOR, LOAD, POP_COUNT,
};

/// A selection that a procedure re-emits once per fold (see [`margin`]).
pub type Select<'s> = dyn Fn(&mut FoldDialect<'_>, &mut Vec<Val>) -> Result<(), FoldError> + 's;

/// `LOAD` space for a `U32` lane.
const SPACE_U32: u8 = 0;
/// `LOAD` space for an `I32` lane.
const SPACE_I32: u8 = 1;
/// `LOAD` space for a resident mask plane.
const SPACE_PLANE: u8 = 3;

fn call(
    d: &mut FoldDialect<'_>,
    f: FnIndex,
    v: [u8; 3],
    stack: &mut Vec<Val>,
) -> Result<(), FoldError> {
    d.call(f, v, stack)
}

/// [`FoldDialect::call_owned`]: for a chain whose intermediate result this
/// procedure created and consumes exactly once, so its slot can be reused.
/// Only the chains inside one procedure use it; a value handed in on the
/// caller's stack goes through [`call`].
fn call_chain(d: &mut FoldDialect<'_>, f: FnIndex, stack: &mut Vec<Val>) -> Result<(), FoldError> {
    d.call_owned(f, [0; 3], stack)
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

/// The rows of the resident plane `live` that pass every condition: each
/// `(lane, v, negate)` is `U32 lane = v`, or `!=` when `negate`. Pushes the
/// mask.
///
/// Calls: `LOAD` (plane), then per condition `LOAD`, `IntEqual`, [`IntNot`,]
/// `IntAnd`.
pub fn select_u32(
    d: &mut FoldDialect<'_>,
    stack: &mut Vec<Val>,
    live: u16,
    conds: &[(u16, u32, bool)],
) -> Result<(), FoldError> {
    load(d, stack, SPACE_PLANE, live)?;
    for &(lane, v, negate) in conds {
        load(d, stack, SPACE_U32, lane)?;
        stack.push(Val::Scalar(i64::from(v)));
        call(d, INT_EQUAL, [0; 3], stack)?;
        if negate {
            call(d, INT_NOT, [0; 3], stack)?;
        }
        call(d, INT_AND, [0; 3], stack)?;
    }
    Ok(())
}

/// How many rows the mask on top of the stack holds. Pushes the count.
///
/// Calls: `PopCount`.
pub fn count(d: &mut FoldDialect<'_>, stack: &mut Vec<Val>) -> Result<(), FoldError> {
    call(d, POP_COUNT, [0; 3], stack)
}

/// Whether the mask on top of the stack holds any row. Pushes 1 or 0.
///
/// Calls: `ANY`.
pub fn exists(d: &mut FoldDialect<'_>, stack: &mut Vec<Val>) -> Result<(), FoldError> {
    call(d, ANY, [0; 3], stack)
}

/// Whether the mask on top of the stack holds all `n_rows` rows. Pushes 1
/// or 0.
///
/// Calls: `ALL`.
pub fn all(d: &mut FoldDialect<'_>, stack: &mut Vec<Val>) -> Result<(), FoldError> {
    call(d, ALL, [0; 3], stack)
}

/// Whether the resident planes `a` and `b` differ in any row. Pushes 1 or 0.
///
/// This is mask equality as a fold: the planes are equal exactly when their
/// `Xor` is empty. Nothing is decoded into rows.
///
/// Calls: `LOAD`, `LOAD`, `IntXor`, `ANY`.
pub fn differ(
    d: &mut FoldDialect<'_>,
    stack: &mut Vec<Val>,
    a: u16,
    b: u16,
) -> Result<(), FoldError> {
    load(d, stack, SPACE_PLANE, a)?;
    load(d, stack, SPACE_PLANE, b)?;
    call(d, INT_XOR, [0; 3], stack)?;
    exists(d, stack)
}

/// Push the conjunction of the resident planes `planes` as a mask. An
/// empty list pushes nothing; the next fold then refuses with
/// [`FoldError::Underflow`].
///
/// Calls: `LOAD`, then per further plane `LOAD`, `IntAnd`.
pub fn intersect(
    d: &mut FoldDialect<'_>,
    stack: &mut Vec<Val>,
    planes: &[u16],
) -> Result<(), FoldError> {
    for (k, &p) in planes.iter().enumerate() {
        load(d, stack, SPACE_PLANE, p)?;
        if k > 0 {
            call_chain(d, INT_AND, stack)?;
        }
    }
    Ok(())
}

/// Push the union of the resident planes `planes` as a mask, as
/// [`intersect`] with `IntOr`.
///
/// Calls: `LOAD`, then per further plane `LOAD`, `IntOr`.
pub fn union(
    d: &mut FoldDialect<'_>,
    stack: &mut Vec<Val>,
    planes: &[u16],
) -> Result<(), FoldError> {
    for (k, &p) in planes.iter().enumerate() {
        load(d, stack, SPACE_PLANE, p)?;
        if k > 0 {
            call_chain(d, INT_OR, stack)?;
        }
    }
    Ok(())
}

/// Write the mask on top of the stack into the dialect's `KEEP` sink.
/// Pushes nothing.
///
/// Calls: `KEEP`.
pub fn keep(d: &mut FoldDialect<'_>, stack: &mut Vec<Val>) -> Result<(), FoldError> {
    call(d, KEEP, [0; 3], stack)
}

/// How many rows the conjunction of the resident planes `planes` holds.
/// Pushes the count.
///
/// This is the count of a composed path: with planes `[member, valid,
/// carriers]` it is the number of join keys present on both sides.
///
/// Calls: [`intersect`], then `PopCount`.
pub fn count_and(
    d: &mut FoldDialect<'_>,
    stack: &mut Vec<Val>,
    planes: &[u16],
) -> Result<(), FoldError> {
    intersect(d, stack, planes)?;
    count(d, stack)
}

/// `SUM(val_lane | mask)` for the mask on top of the stack. Pushes the sum.
///
/// Calls: `LOAD`, `SUM`.
pub fn sum(d: &mut FoldDialect<'_>, stack: &mut Vec<Val>, val_lane: u16) -> Result<(), FoldError> {
    load(d, stack, SPACE_I32, val_lane)?;
    call(d, SUM, [0; 3], stack)
}

/// `MIN(val_lane | mask)` for the mask on top of the stack. Pushes the
/// minimum, or `i64::MAX` when the mask is empty (SQL NULL; no `I32` value
/// equals it).
///
/// Calls: `LOAD`, `MIN`.
pub fn min(d: &mut FoldDialect<'_>, stack: &mut Vec<Val>, val_lane: u16) -> Result<(), FoldError> {
    load(d, stack, SPACE_I32, val_lane)?;
    call(d, MIN, [0; 3], stack)
}

/// `MAX(val_lane | mask)` for the mask on top of the stack. Pushes the
/// maximum, or `i64::MIN` when the mask is empty.
///
/// Calls: `LOAD`, `MAX`.
pub fn max(d: &mut FoldDialect<'_>, stack: &mut Vec<Val>, val_lane: u16) -> Result<(), FoldError> {
    load(d, stack, SPACE_I32, val_lane)?;
    call(d, MAX, [0; 3], stack)
}

/// An ordered comparison of an `I32` lane against a threshold.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Order {
    /// `lane < t`.
    Lt,
    /// `lane <= t`.
    Le,
    /// `lane > t`.
    Gt,
    /// `lane >= t`.
    Ge,
}

/// AND `lane op t` (an `I32` lane) onto the mask on top of the stack.
///
/// `IntSLess` is the only ordered compare, so `>` is `t < lane` and `<=` /
/// `>=` are the negations of `>` / `<`. The negation covers every row; the
/// AND with the mask below confines it.
///
/// Calls: `LOAD`, `IntSLess`, [`IntNot`,] `IntAnd`.
pub fn and_cmp_i32(
    d: &mut FoldDialect<'_>,
    stack: &mut Vec<Val>,
    lane: u16,
    op: Order,
    t: i32,
) -> Result<(), FoldError> {
    let t = Val::Scalar(i64::from(t));
    match op {
        Order::Lt | Order::Ge => {
            load(d, stack, SPACE_I32, lane)?;
            stack.push(t);
        }
        Order::Gt | Order::Le => {
            stack.push(t);
            load(d, stack, SPACE_I32, lane)?;
        }
    }
    call(d, INT_S_LESS, [0; 3], stack)?;
    if matches!(op, Order::Le | Order::Ge) {
        call(d, INT_NOT, [0; 3], stack)?;
    }
    call(d, INT_AND, [0; 3], stack)
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

// ── PostgreSQL primitives as call sequences ──────────────────────────────
//
// A column's NULLs are its presence plane: a resident mask with a bit set
// where the value is present (Quack's model). Every procedure below either
// leaves a mask on the stack (a WHERE predicate: KEEP it into a sink, and a
// kept mask is what a Lance 13 masked read takes as its row-address
// prefilter) or a scalar (an aggregate).

/// `x IS NULL`: the complement of `x`'s presence plane `valid`. Pushes the
/// mask.
///
/// Calls: `LOAD`, `IntNot`.
pub fn is_null(d: &mut FoldDialect<'_>, stack: &mut Vec<Val>, valid: u16) -> Result<(), FoldError> {
    load(d, stack, SPACE_PLANE, valid)?;
    call(d, INT_NOT, [0; 3], stack)
}

/// `x IS NOT NULL`: `x`'s presence plane `valid`. Pushes the mask.
///
/// Calls: `LOAD`.
pub fn is_not_null(
    d: &mut FoldDialect<'_>,
    stack: &mut Vec<Val>,
    valid: u16,
) -> Result<(), FoldError> {
    load(d, stack, SPACE_PLANE, valid)
}

/// `x BETWEEN lo AND hi` on an `I32` lane, under the mask on top of the stack
/// (pass the presence plane there for a nullable `x`: a NULL `x` is UNKNOWN,
/// which `WHERE` drops). An empty range (`lo > hi`) keeps nothing, as in SQL.
///
/// Calls: [`and_cmp_i32`] twice.
pub fn and_between_i32(
    d: &mut FoldDialect<'_>,
    stack: &mut Vec<Val>,
    lane: u16,
    lo: i32,
    hi: i32,
) -> Result<(), FoldError> {
    and_cmp_i32(d, stack, lane, Order::Ge, lo)?;
    and_cmp_i32(d, stack, lane, Order::Le, hi)
}

/// `x IN (v1, …, vn)` on a `U32` lane. Pushes the mask. An empty list keeps
/// nothing (`x IN ()` is FALSE). A NULL in the list cannot be expressed here;
/// that is K3 territory (Quack `sql_where`).
///
/// Calls: per value `LOAD`, `IntEqual`, `IntOr` (the first without `IntOr`);
/// for an empty list `LOAD`, `IntEqual`, `IntNot`, `IntAnd` (x = 0 ∧ x ≠ 0).
pub fn in_u32(
    d: &mut FoldDialect<'_>,
    stack: &mut Vec<Val>,
    lane: u16,
    values: &[u32],
) -> Result<(), FoldError> {
    if values.is_empty() {
        for negate in [false, true] {
            load(d, stack, SPACE_U32, lane)?;
            stack.push(Val::Scalar(0));
            call(d, INT_EQUAL, [0; 3], stack)?;
            if negate {
                call(d, INT_NOT, [0; 3], stack)?;
            }
        }
        return call(d, INT_AND, [0; 3], stack);
    }
    for (k, &v) in values.iter().enumerate() {
        load(d, stack, SPACE_U32, lane)?;
        stack.push(Val::Scalar(i64::from(v)));
        call(d, INT_EQUAL, [0; 3], stack)?;
        if k > 0 {
            call_chain(d, INT_OR, stack)?;
        }
    }
    Ok(())
}

/// `COUNT(x)` for the mask on top of the stack: rows selected AND present.
/// `COUNT(*)` is [`count`]. Pushes the count.
///
/// Calls: `LOAD`, `IntAnd`, `PopCount`.
pub fn count_valid(
    d: &mut FoldDialect<'_>,
    stack: &mut Vec<Val>,
    valid: u16,
) -> Result<(), FoldError> {
    load(d, stack, SPACE_PLANE, valid)?;
    call(d, INT_AND, [0; 3], stack)?;
    count(d, stack)
}

/// `AVG(x)` over an `I32` lane: pushes `SUM(x)` then `COUNT(x)`, both over
/// `select ∧ valid`. The quotient is left to the caller because PostgreSQL's
/// `avg(int)` is `numeric` (fractional) and R2IL has no exact rational; a
/// count of 0 is SQL NULL. `select` is re-emitted per fold, as in [`margin`].
///
/// Calls: `select`, `LOAD`, `IntAnd`, `LOAD`, `SUM`; `select`, `LOAD`,
/// `IntAnd`, `PopCount`.
pub fn avg_parts(
    d: &mut FoldDialect<'_>,
    stack: &mut Vec<Val>,
    select: &Select<'_>,
    val_lane: u16,
    valid: u16,
) -> Result<(), FoldError> {
    select(d, stack)?;
    load(d, stack, SPACE_PLANE, valid)?;
    call(d, INT_AND, [0; 3], stack)?;
    sum(d, stack, val_lane)?;
    select(d, stack)?;
    count_valid(d, stack, valid)
}

/// `bool_or(b)` over a nullable boolean `b`, under the selection `select`.
/// `valid` is `b`'s presence plane and `flag` the plane of rows where `b` is
/// TRUE (its bits under a NULL are ignored).
///
/// Pushes two scalars: the value, then `present` on top. `present == 0` is SQL
/// NULL (no selected row has a non-NULL `b`, including an empty selection);
/// the value is then 0 and must not be read as FALSE. Otherwise the value is 1
/// when some contributing row is TRUE. The contributing rows are
/// `select ∧ valid`; `select` is re-emitted per fold, as in [`avg_parts`].
///
/// Calls: `select`, `LOAD`, `IntAnd`, `LOAD`, `IntAnd`, `ANY`; `select`,
/// `LOAD`, `IntAnd`, `ANY`.
pub fn bool_or(
    d: &mut FoldDialect<'_>,
    stack: &mut Vec<Val>,
    select: &Select<'_>,
    valid: u16,
    flag: u16,
) -> Result<(), FoldError> {
    select(d, stack)?;
    load(d, stack, SPACE_PLANE, valid)?;
    call(d, INT_AND, [0; 3], stack)?;
    load(d, stack, SPACE_PLANE, flag)?;
    call(d, INT_AND, [0; 3], stack)?;
    exists(d, stack)?;
    present(d, stack, select, valid)
}

/// `bool_and(b)` (`every`) over a nullable boolean `b`, under `select`. Same
/// arguments and result shape as [`bool_or`]: the value, then `present`. When
/// present, the value is 1 unless some contributing row is FALSE; NULL rows
/// do not count as FALSE. When not present the value is 0 (SQL NULL).
///
/// The value is `present − has_false`: with no contributing row both are 0.
///
/// Calls: `select`, `LOAD`, `IntAnd`, `ANY` (present); `select`, `LOAD`,
/// `IntAnd`, `LOAD`, `IntNot`, `IntAnd`, `ANY` (has_false); `IntSub`.
pub fn bool_and(
    d: &mut FoldDialect<'_>,
    stack: &mut Vec<Val>,
    select: &Select<'_>,
    valid: u16,
    flag: u16,
) -> Result<(), FoldError> {
    present(d, stack, select, valid)?;
    let has_values = *stack.last().ok_or(FoldError::Underflow(ANY))?;
    select(d, stack)?;
    load(d, stack, SPACE_PLANE, valid)?;
    call(d, INT_AND, [0; 3], stack)?;
    load(d, stack, SPACE_PLANE, flag)?;
    call(d, INT_NOT, [0; 3], stack)?;
    call(d, INT_AND, [0; 3], stack)?;
    exists(d, stack)?;
    call(d, INT_SUB, [0; 3], stack)?;
    stack.push(has_values);
    Ok(())
}

/// Whether any row of `select ∧ valid` exists: the non-NULL population of a
/// nullable aggregate. Pushes 1 or 0.
fn present(
    d: &mut FoldDialect<'_>,
    stack: &mut Vec<Val>,
    select: &Select<'_>,
    valid: u16,
) -> Result<(), FoldError> {
    select(d, stack)?;
    load(d, stack, SPACE_PLANE, valid)?;
    call(d, INT_AND, [0; 3], stack)?;
    exists(d, stack)
}

/// `NOT EXISTS (…)` for the mask on top of the stack. Pushes 1 or 0.
///
/// Calls: `ANY`, then 1 − that (`IntSub`).
pub fn not_exists(d: &mut FoldDialect<'_>, stack: &mut Vec<Val>) -> Result<(), FoldError> {
    exists(d, stack)?;
    let any = stack.pop().ok_or(FoldError::Underflow(ANY))?;
    stack.push(Val::Scalar(1));
    stack.push(any);
    call(d, INT_SUB, [0; 3], stack)
}

/// `a EXCEPT b` over two resident planes of the same rows (set semantics on
/// row identity): `a ∧ ¬b`. Pushes the mask. `INTERSECT` is [`intersect`],
/// `UNION` is [`union`].
///
/// Calls: `LOAD`, `LOAD`, `IntNot`, `IntAnd`.
pub fn except(
    d: &mut FoldDialect<'_>,
    stack: &mut Vec<Val>,
    a: u16,
    b: u16,
) -> Result<(), FoldError> {
    load(d, stack, SPACE_PLANE, a)?;
    load(d, stack, SPACE_PLANE, b)?;
    call(d, INT_NOT, [0; 3], stack)?;
    call(d, INT_AND, [0; 3], stack)
}

/// `COUNT(DISTINCT key)` for the mask on top of the stack, on a KEY-ORDERED
/// `U32` lane (the T0 key-order projection): one per run of equal keys that
/// holds a selected row whose key is not NULL. `valid` is the key's presence
/// plane; a NULL key is excluded whatever its payload, so a payload equal to
/// a real key adds nothing and an all-NULL selection counts 0 (SQL
/// `count(DISTINCT x)`; `GROUP BY` treats NULL keys differently). For a key
/// that cannot be NULL pass a plane of every row. An unordered lane is refused
/// by mask-risc (`LaneNotOrdered`), never answered; nothing is sorted or
/// hashed. Pushes the count.
///
/// Calls: `LOAD`, `IntAnd`, `LOAD`, `KEY_RUNS`.
pub fn count_distinct_sorted(
    d: &mut FoldDialect<'_>,
    stack: &mut Vec<Val>,
    key_lane: u16,
    valid: u16,
) -> Result<(), FoldError> {
    load(d, stack, SPACE_PLANE, valid)?;
    call(d, INT_AND, [0; 3], stack)?;
    load(d, stack, SPACE_U32, key_lane)?;
    call(d, KEY_RUNS, [0; 3], stack)
}

/// `COUNT(*)` of two selections added (`UNION ALL` of two counts), checked.
/// Pushes the sum.
///
/// Calls: `select_a`, `PopCount`, `select_b`, `PopCount`, `IntAdd`.
pub fn count_union_all(
    d: &mut FoldDialect<'_>,
    stack: &mut Vec<Val>,
    select_a: &Select<'_>,
    select_b: &Select<'_>,
) -> Result<(), FoldError> {
    select_a(d, stack)?;
    count(d, stack)?;
    select_b(d, stack)?;
    count(d, stack)?;
    call(d, INT_ADD, [0; 3], stack)
}
