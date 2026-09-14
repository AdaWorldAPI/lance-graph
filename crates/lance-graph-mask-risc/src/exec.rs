//! The borrowing executor — every op is ONE delegation to an `ndarray::simd`
//! facade word over caller-owned memory.
//!
//! Laws. L1, L2, L4 and L5 are tests (`tests/no_alloc.rs`,
//! `the_crate_names_no_isa`, `the_oracle_has_no_facade_token`,
//! `exactly_one_materialiser`); **L3 is a reading of `execute`**, not a test —
//! no instrument counts facade calls, so it is `[claimed, unverified]`:
//!
//! - **L1** `execute` never allocates: [`Scratch::new`] is the only allocation
//!   and it is the caller's.
//! - **L2** no ISA: this file carries no feature gate and no architecture
//!   intrinsic; which backend runs a word is `ndarray`'s business.
//! - **L3** one delegation per op, with three named second shapes: the
//!   aliasing form (when `dst` is also an input the op routes to the facade's
//!   `_assign` member, or — for [`MaskOp::AndNot`] / [`MaskOp::Ternlog`] with
//!   `dst` on the right — to the same in-place ternlog with a permuted
//!   immediate); the tail clear an ODD ternlog immediate owes, spelled once in
//!   [`clear_tail`]; and `ternlog_self`'s fills for `x = f(x, x, x)`, which no
//!   facade word can express (it would need one buffer borrowed mutably and
//!   shared at once).
//! - **L4** the oracle in [`crate::reference`] is independent; the validation
//!   rules are shared with it (`validate`) so both sides refuse identically.
//! - **L5** exactly one materialiser: [`materialize_rows`].

use ndarray::simd::{
    blend_i32, eq_i32_to_mask, eq_i32_to_mask_under, eq_u32_to_mask, eq_u32_to_mask_under,
    ge_i32_to_mask, ge_i32_to_mask_under, gt_i32_to_mask, gt_i32_to_mask_under, le_i32_to_mask,
    le_i32_to_mask_under, lt_i32_to_mask, lt_i32_to_mask_under, mask_all, mask_and,
    mask_and_assign, mask_andnot, mask_andnot_assign, mask_any, mask_not, mask_not_assign, mask_or,
    mask_or_assign, mask_xor, mask_xor_assign, masked_max_i32, masked_min_i32, masked_sum_i32,
    ne_i32_to_mask, ne_i32_to_mask_under, ne_u32_to_mask, ne_u32_to_mask_under, popcount_batch_u64,
    ternary_match_u32_to_mask, ternary_match_u32_to_mask_under, ternary_match_u64_to_mask,
    ternary_match_u64_to_mask_under,
};

use crate::ir::{LaneRef, MaskOp, Operand, Planes, Pred, Program, Terminal, MAX_SCRATCH_SLOTS};
use crate::reference::validate;
use crate::ternlog_dispatch::{ternlog_dispatch, ternlog_dispatch_assign};
use crate::value::{ExecError, Value};
use crate::words_for;

/// Caller-owned scratch: `slots` buffers of `words` u64 each. The ONLY
/// allocation in this crate's execution path, made once by the caller and
/// reused across every `execute`.
#[derive(Debug, Clone)]
pub struct Scratch {
    words: usize,
    slots: Vec<Box<[u64]>>,
}

impl Scratch {
    /// Allocate `slots` zeroed buffers of `words` u64 each.
    pub fn new(words: usize, slots: usize) -> Self {
        Self {
            words,
            slots: (0..slots)
                .map(|_| vec![0u64; words].into_boxed_slice())
                .collect(),
        }
    }

    /// Allocate exactly what `program` needs over `n_rows` rows.
    ///
    /// Fallible, and deliberately so. `Operand::Scratch` is a `u16`, so no
    /// program can ADDRESS more than [`MAX_SCRATCH_SLOTS`] slots; a larger
    /// count means a hand-built program lied about a PUBLIC field. This used
    /// to be a `debug_assert`, which release builds drop — leaving the lie to
    /// reach `Scratch::new` and allocate unboundedly. The same bound is in
    /// [`validate`], so a program refused here is refused identically by
    /// `execute` and by the oracle; the check is repeated rather than
    /// delegated because a caller sizes its arena BEFORE `execute` runs, and
    /// a validation that happens afterwards cannot prevent this allocation.
    pub fn for_program(program: &Program, n_rows: usize) -> Result<Self, ExecError> {
        if program.scratch_slots > MAX_SCRATCH_SLOTS {
            return Err(ExecError::ScratchSlotsUnaddressable {
                declared: program.scratch_slots,
            });
        }
        // `scratch_slots` is a `u32` count; a program naming slot `u16::MAX`
        // needs 65,536 buffers, which fits `usize` on every supported target.
        Ok(Self::new(words_for(n_rows), program.scratch_slots as usize))
    }

    /// Words per slot.
    pub fn words(&self) -> usize {
        self.words
    }

    /// Number of slots.
    pub fn slots(&self) -> usize {
        self.slots.len()
    }

    /// Borrow slot `i` (the way a caller reads back a [`Terminal::Keep`] result).
    pub fn slot(&self, i: u16) -> Option<&[u64]> {
        self.slots.get(usize::from(i)).map(|b| &**b)
    }

    /// Take slot `i` out of the arena so the remaining slots can be read while
    /// it is written. `Box<[u64]>::default()` is a dangling empty slice — no
    /// allocation — and [`Self::restore`] puts the buffer back.
    fn take(&mut self, i: u16) -> Box<[u64]> {
        core::mem::take(&mut self.slots[usize::from(i)])
    }

    fn restore(&mut self, i: u16, buf: Box<[u64]>) {
        self.slots[usize::from(i)] = buf;
    }
}

/// The ONE materialiser: row indices of every set bit of `mask`, in
/// ascending order. **O(n_rows)** and it allocates — this is the boundary a
/// consumer crosses deliberately, by this name, never as normal execution
/// state (the mask-native invariant).
pub fn materialize_rows(mask: &[u64], n_rows: usize) -> Vec<usize> {
    let mut rows = Vec::new();
    for (w, &word) in mask.iter().enumerate() {
        let mut bits = word;
        while bits != 0 {
            let row = w * 64 + bits.trailing_zeros() as usize;
            if row >= n_rows {
                // Rows ascend within a word and words ascend, so the FIRST
                // out-of-range row ends the whole walk — a `break` would only
                // end this word's bits and leave the bound stated per word.
                return rows;
            }
            rows.push(row);
            bits &= bits - 1;
        }
    }
    rows
}

/// Clear every bit at or past `n_rows`, in the tail WORD — the contract's own
/// `clear_tail` spelling, and the obligation `mask_ternlog`'s doc places on
/// the caller for an odd immediate (`f(0,0,0) = 1` sets the whole tail).
///
/// Narrower than `ndarray`'s private `clear_mask_tail`, which also zeroes
/// every word PAST the tail word. The two agree only on exactly-sized
/// buffers — which is what the exact `ScratchWords` / `LenMismatch` checks
/// guarantee, and why this one-word form is sound here.
fn clear_tail(dst: &mut [u64], n_rows: usize) {
    let live = n_rows % 64;
    if !n_rows.is_multiple_of(64) {
        if let Some(w) = dst.get_mut(n_rows / 64) {
            *w &= (1u64 << live) - 1;
        }
    }
}

/// Re-index a VPTERNLOG immediate for the in-place form. `map[i]` is the NEW
/// position (`0 = x`, the buffer being written; `1 = y`; `2 = z`) of ORIGINAL
/// operand `i` (`0 = a`, `1 = b`, `2 = c`), so `new(x, y, z) = old(a, b, c)`
/// for every assignment. The identity map returns `imm` unchanged; several
/// originals may share one new position (an aliased operand).
fn remap_imm(imm: u8, map: [u8; 3]) -> u8 {
    let mut out = 0u8;
    for idx in 0..8u8 {
        let new_bits = [(idx >> 2) & 1, (idx >> 1) & 1, idx & 1];
        let bit = |orig: usize| new_bits[usize::from(map[orig])];
        let old_idx = (bit(0) << 2) | (bit(1) << 1) | bit(2);
        out |= ((imm >> old_idx) & 1) << idx;
    }
    out
}

/// The two-input facade words as ternlog tables (`c` ignored, index
/// `(a<<2)|(b<<1)|c`): the executor never dispatches on these — they exist
/// so the aliasing arms below can re-index ONE table instead of carrying a
/// private special case per op.
const AND_IMM: u8 = 0xC0;
const OR_IMM: u8 = 0xFC;
const XOR_IMM: u8 = 0x3C;
const ANDNOT_IMM: u8 = 0x30;

/// One two-input pass into `dst`. `dst == a` routes to the facade's in-place
/// member; `dst == b` does the same when the table is symmetric, and
/// otherwise goes through the in-place ternlog with the table re-indexed
/// (`x = y & !x` is table `0x0C` over `(x, y, y)`). Whether the op commutes
/// is READ from the table (`remap_imm(imm, [1, 0, 2]) == imm`), not declared.
#[allow(clippy::too_many_arguments)]
fn two_input(
    planes: &Planes<'_>,
    s: &mut Scratch,
    a: Operand,
    b: Operand,
    dst: u16,
    op: fn(&[u64], &[u64], &mut [u64]),
    assign: fn(&mut [u64], &[u64]),
    imm: u8,
) {
    let d = Operand::Scratch(dst);
    // Every two-input table is EVEN (`f(0,0,0) = 0`), which is why this
    // function owes no tail clear — unlike the `Ternlog` arm. Stated here
    // because it is a precondition on a private fn, not a property of the
    // caller's input.
    debug_assert!(imm & 1 == 0, "two-input table {imm:#04x} is odd");
    let commutative = remap_imm(imm, [1, 0, 2]) == imm;
    let mut x = s.take(dst);
    if a == d && b == d {
        ternlog_self(remap_imm(imm, [0, 0, 0]), &mut x, planes.n_rows);
    } else if a == d {
        assign(&mut x, read(planes, s, b));
    } else if b == d {
        let aa = read(planes, s, a);
        if commutative {
            assign(&mut x, aa);
        } else {
            ternlog_dispatch_assign(remap_imm(imm, [1, 0, 0]), &mut x, aa, aa);
        }
    } else {
        op(read(planes, s, a), read(planes, s, b), &mut x);
    }
    s.restore(dst, x);
}

/// `x = f(x, x, x)` — a table over ONE buffer collapses to two bits:
/// `f(0,0,0)` (bit 0) and `f(1,1,1)` (bit 7). No facade word takes a single
/// operand in place except the complement, so the four outcomes are spelled
/// here: identity, complement (via `mask_not_assign`, which owns the tail
/// law), all-zero, all-one-then-tail-cleared.
fn ternlog_self(imm: u8, x: &mut [u64], n_rows: usize) {
    match (imm & 1, (imm >> 7) & 1) {
        (0, 1) => {}
        (1, 0) => mask_not_assign(x, n_rows),
        (0, 0) => x.fill(0),
        _ => {
            x.fill(u64::MAX);
            clear_tail(x, n_rows);
        }
    }
}

/// Borrow an operand for reading (the scratch arena with `dst` already taken
/// out, so a read of `dst`'s own slot here is a bug the aliasing arms prevent).
fn read<'a>(planes: &Planes<'a>, s: &'a Scratch, o: Operand) -> &'a [u64] {
    match o {
        Operand::Plane(i) => planes.masks[usize::from(i)],
        Operand::Scratch(i) => {
            // A slot taken out of the arena reads as EMPTY, and the facade
            // words no-op on an empty slice — a wrong answer, not a crash.
            // Every aliasing arm routes `operand == dst` to an `_assign` form
            // before reaching here, so this can only fire if a future arm
            // forgets to; it is cheaper to assert than to debug.
            debug_assert!(
                !s.slots[usize::from(i)].is_empty() || s.words == 0,
                "slot {i} was read while taken out of the arena"
            );
            &s.slots[usize::from(i)]
        }
    }
}

fn lane_i32<'a>(planes: &Planes<'a>, lane: u16) -> &'a [i32] {
    match planes.lanes[usize::from(lane)] {
        LaneRef::I32(v) => v,
        // unreachable after `validate`, but the type system does not know
        // that: an empty lane makes every facade call a no-op of the right
        // shape rather than a panic.
        _ => &[],
    }
}

fn lane_u32<'a>(planes: &Planes<'a>, lane: u16) -> &'a [u32] {
    match planes.lanes[usize::from(lane)] {
        LaneRef::U32(v) => v,
        _ => &[],
    }
}

fn lane_u64<'a>(planes: &Planes<'a>, lane: u16) -> &'a [u64] {
    match planes.lanes[usize::from(lane)] {
        LaneRef::U64(v) => v,
        _ => &[],
    }
}

/// One predicate pass into `dst`: the ungated facade member, or the `_under`
/// member when a gate is present (cost then follows the gate's live words).
fn run_pred(planes: &Planes<'_>, s: &Scratch, pred: Pred, under: Option<Operand>, dst: &mut [u64]) {
    match (pred, under) {
        (Pred::GtI32 { lane, t }, None) => gt_i32_to_mask(lane_i32(planes, lane), t, dst),
        (Pred::GtI32 { lane, t }, Some(u)) => {
            gt_i32_to_mask_under(lane_i32(planes, lane), t, read(planes, s, u), dst)
        }
        (Pred::LtI32 { lane, t }, None) => lt_i32_to_mask(lane_i32(planes, lane), t, dst),
        (Pred::LtI32 { lane, t }, Some(u)) => {
            lt_i32_to_mask_under(lane_i32(planes, lane), t, read(planes, s, u), dst)
        }
        (Pred::GeI32 { lane, t }, None) => ge_i32_to_mask(lane_i32(planes, lane), t, dst),
        (Pred::GeI32 { lane, t }, Some(u)) => {
            ge_i32_to_mask_under(lane_i32(planes, lane), t, read(planes, s, u), dst)
        }
        (Pred::LeI32 { lane, t }, None) => le_i32_to_mask(lane_i32(planes, lane), t, dst),
        (Pred::LeI32 { lane, t }, Some(u)) => {
            le_i32_to_mask_under(lane_i32(planes, lane), t, read(planes, s, u), dst)
        }
        (Pred::EqI32 { lane, v }, None) => eq_i32_to_mask(lane_i32(planes, lane), v, dst),
        (Pred::EqI32 { lane, v }, Some(u)) => {
            eq_i32_to_mask_under(lane_i32(planes, lane), v, read(planes, s, u), dst)
        }
        (Pred::NeI32 { lane, v }, None) => ne_i32_to_mask(lane_i32(planes, lane), v, dst),
        (Pred::NeI32 { lane, v }, Some(u)) => {
            ne_i32_to_mask_under(lane_i32(planes, lane), v, read(planes, s, u), dst)
        }
        (Pred::EqU32 { lane, v }, None) => eq_u32_to_mask(lane_u32(planes, lane), v, dst),
        (Pred::EqU32 { lane, v }, Some(u)) => {
            eq_u32_to_mask_under(lane_u32(planes, lane), v, read(planes, s, u), dst)
        }
        (Pred::NeU32 { lane, v }, None) => ne_u32_to_mask(lane_u32(planes, lane), v, dst),
        (Pred::NeU32 { lane, v }, Some(u)) => {
            ne_u32_to_mask_under(lane_u32(planes, lane), v, read(planes, s, u), dst)
        }
        (
            Pred::MatchU32 {
                lane,
                pattern,
                care,
            },
            None,
        ) => ternary_match_u32_to_mask(lane_u32(planes, lane), pattern, care, dst),
        (
            Pred::MatchU32 {
                lane,
                pattern,
                care,
            },
            Some(u),
        ) => ternary_match_u32_to_mask_under(
            lane_u32(planes, lane),
            pattern,
            care,
            read(planes, s, u),
            dst,
        ),
        (
            Pred::MatchU64 {
                lane,
                pattern,
                care,
            },
            None,
        ) => ternary_match_u64_to_mask(lane_u64(planes, lane), pattern, care, dst),
        (
            Pred::MatchU64 {
                lane,
                pattern,
                care,
            },
            Some(u),
        ) => ternary_match_u64_to_mask_under(
            lane_u64(planes, lane),
            pattern,
            care,
            read(planes, s, u),
            dst,
        ),
    }
}

/// Run `program` over `planes` with the caller's `scratch`; `out` is the
/// destination a [`Terminal::BlendI32`] writes. Validation is total and
/// happens before any write, so an `Err` leaves `scratch` and `out` untouched.
pub fn execute(
    program: &Program,
    planes: &Planes<'_>,
    scratch: &mut Scratch,
    out: Option<&mut [i32]>,
) -> Result<Value, ExecError> {
    // BEFORE the capacity check, not after: an over-declared count is a lie
    // about the PROGRAM, and the caller's buffer is irrelevant to it. Checked
    // second, every such program reports `ScratchTooSmall` instead — which the
    // oracle and `Scratch::for_program` never say, so the two paths would
    // refuse the same program with different errors, and a caller that grows
    // its arena on `ScratchTooSmall` would allocate from exactly the count
    // this bound exists to reject.
    if program.scratch_slots > MAX_SCRATCH_SLOTS {
        return Err(ExecError::ScratchSlotsUnaddressable {
            declared: program.scratch_slots,
        });
    }
    if scratch.slots.len() < program.scratch_slots as usize {
        return Err(ExecError::ScratchTooSmall {
            need: program.scratch_slots,
            have: scratch.slots.len(),
        });
    }
    let words = words_for(planes.n_rows);
    if scratch.words != words {
        return Err(ExecError::ScratchWords {
            expected: words,
            found: scratch.words,
        });
    }
    validate(program, planes, out.as_deref().map(<[i32]>::len))?;
    let n_rows = planes.n_rows;

    for op in &program.ops {
        match *op {
            MaskOp::Pred { pred, under, dst } => {
                let mut d = scratch.take(dst);
                run_pred(planes, scratch, pred, under, &mut d);
                scratch.restore(dst, d);
            }
            MaskOp::And { a, b, dst } => two_input(
                planes,
                scratch,
                a,
                b,
                dst,
                mask_and,
                mask_and_assign,
                AND_IMM,
            ),
            MaskOp::Or { a, b, dst } => {
                two_input(planes, scratch, a, b, dst, mask_or, mask_or_assign, OR_IMM)
            }
            MaskOp::Xor { a, b, dst } => two_input(
                planes,
                scratch,
                a,
                b,
                dst,
                mask_xor,
                mask_xor_assign,
                XOR_IMM,
            ),
            MaskOp::AndNot { a, b, dst } => two_input(
                planes,
                scratch,
                a,
                b,
                dst,
                mask_andnot,
                mask_andnot_assign,
                ANDNOT_IMM,
            ),
            MaskOp::Not { a, dst } => {
                if a == Operand::Scratch(dst) {
                    mask_not_assign(&mut scratch.slots[usize::from(dst)], n_rows);
                } else {
                    let mut d = scratch.take(dst);
                    mask_not(read(planes, scratch, a), n_rows, &mut d);
                    scratch.restore(dst, d);
                }
            }
            MaskOp::Ternlog { imm, a, b, c, dst } => {
                let d = Operand::Scratch(dst);
                let mut x = scratch.take(dst);
                if a != d && b != d && c != d {
                    ternlog_dispatch(
                        imm,
                        read(planes, scratch, a),
                        read(planes, scratch, b),
                        read(planes, scratch, c),
                        &mut x,
                    );
                } else {
                    // `dst` is an input: the in-place form takes it as `x`,
                    // the remaining distinct operands become `y` (and `z`),
                    // and the table is re-indexed to match.
                    let mut map = [0u8; 3];
                    let mut others: [Option<Operand>; 2] = [None, None];
                    let mut n = 0usize;
                    for (i, o) in [a, b, c].into_iter().enumerate() {
                        if o == d {
                            map[i] = 0;
                        } else if let Some(pos) = others[..n].iter().position(|&p| p == Some(o)) {
                            map[i] = pos as u8 + 1;
                        } else {
                            others[n] = Some(o);
                            map[i] = n as u8 + 1;
                            n += 1;
                        }
                    }
                    let imm2 = remap_imm(imm, map);
                    match (others[0], others[1]) {
                        (Some(y), Some(z)) => ternlog_dispatch_assign(
                            imm2,
                            &mut x,
                            read(planes, scratch, y),
                            read(planes, scratch, z),
                        ),
                        (Some(y), None) => {
                            let yy = read(planes, scratch, y);
                            ternlog_dispatch_assign(imm2, &mut x, yy, yy)
                        }
                        _ => ternlog_self(imm2, &mut x, n_rows),
                    }
                }
                if imm & 1 == 1 {
                    clear_tail(&mut x, n_rows);
                }
                scratch.restore(dst, x);
            }
        }
    }

    Ok(match program.terminal {
        Terminal::Count { mask } => {
            Value::Count(popcount_batch_u64(read(planes, scratch, mask)) as usize)
        }
        Terminal::Any { mask } => Value::Bool(mask_any(read(planes, scratch, mask))),
        Terminal::All { mask } => Value::Bool(mask_all(read(planes, scratch, mask), n_rows)),
        Terminal::MaskedSumI32 { mask, lane } => Value::SumI64(masked_sum_i32(
            lane_i32(planes, lane),
            read(planes, scratch, mask),
        )),
        Terminal::MaskedMinI32 { mask, lane } => Value::OptI32(masked_min_i32(
            lane_i32(planes, lane),
            read(planes, scratch, mask),
        )),
        Terminal::MaskedMaxI32 { mask, lane } => Value::OptI32(masked_max_i32(
            lane_i32(planes, lane),
            read(planes, scratch, mask),
        )),
        Terminal::BlendI32 { mask, then, els } => {
            // `validate` already refused a missing or mis-sized `out`.
            if let Some(o) = out {
                blend_i32(
                    read(planes, scratch, mask),
                    lane_i32(planes, then),
                    lane_i32(planes, els),
                    o,
                );
            }
            Value::Blended
        }
        Terminal::Keep { mask } => Value::Mask(mask),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every production source of the crate. The two structural laws (L2 no
    /// ISA, L5 one materialiser) read the SAME list, so a new module cannot
    /// be covered by one and invisible to the other.
    const CRATE_SOURCES: [&str; 7] = [
        include_str!("exec.rs"),
        include_str!("fuse.rs"),
        include_str!("ir.rs"),
        include_str!("lib.rs"),
        include_str!("reference.rs"),
        include_str!("ternlog_dispatch.rs"),
        include_str!("value.rs"),
    ];

    /// FAILS IF: the identity permutation changes a table — the remap would
    /// then be wrong for every aliasing shape at once.
    #[test]
    fn remap_identity_is_the_identity() {
        for imm in 0..=255u8 {
            assert_eq!(remap_imm(imm, [0, 1, 2]), imm);
        }
    }

    /// FAILS IF: the bit TRANSFER is inverted (`out[old] |= imm[new]` gives
    /// `0x08`). It does NOT distinguish a map from its inverse — for
    /// `[1, 0, 0]` both readings coincide — so the map's DIRECTION is pinned
    /// by `ternlog_every_aliasing_map`'s 3-cycle shapes instead. The in-place
    /// `x = y & !x` table over `(x, y, y)` is true exactly at `(x=0, y=1)`,
    /// indices `0b010` and `0b011`, derived by hand.
    #[test]
    fn andnot_with_dst_on_the_right_is_table_0x0c() {
        assert_eq!(remap_imm(ANDNOT_IMM, [1, 0, 0]), 0x0C);
    }

    /// FAILS IF: a symmetric table is read as non-commutative or vice versa —
    /// `AndNot` would then route `dst == b` to `mask_andnot_assign(b, a)`,
    /// computing `b & !a` in place of `a & !b`.
    #[test]
    fn commutativity_is_read_from_the_table() {
        for imm in [AND_IMM, OR_IMM, XOR_IMM] {
            assert_eq!(remap_imm(imm, [1, 0, 2]), imm, "{imm:#x} is symmetric");
        }
        assert_ne!(remap_imm(ANDNOT_IMM, [1, 0, 2]), ANDNOT_IMM);
    }

    /// FAILS IF: ANY production module grows an ISA branch — law L2, which
    /// the crate claims for itself and so must check for itself. Which
    /// instruction runs a word is `ndarray`'s business.
    #[test]
    fn the_crate_names_no_isa() {
        // Split so this test's own text cannot match itself; `std::arch` and
        // the runtime-detection macro are named too, because the law is "no
        // ISA", not "not these three spellings".
        let n = [
            "target_",
            "feature",
            "core::",
            "std::",
            "arch",
            "cfg(target_",
            "is_x86_",
            "detected!",
        ];
        let banned = [
            [n[0], n[1]].concat(),
            [n[2], n[4]].concat(),
            [n[3], n[4]].concat(),
            [n[5], n[4]].concat(),
            [n[6], n[7]].concat(),
        ];
        for src in CRATE_SOURCES {
            let production = src.split("#[cfg(test)]").next().unwrap_or("");
            // CODE only: `lib.rs` states this very law in prose and so names
            // the tokens it forbids. A doc line is the opposite of a breach.
            for line in production
                .lines()
                .filter(|l| !l.trim_start().starts_with("//"))
            {
                for needle in &banned {
                    assert!(
                        !line.contains(needle.as_str()),
                        "ISA token {needle:?} in: {line}"
                    );
                }
            }
        }
    }

    /// FAILS IF: `execute` reports a DIFFERENT error than the oracle for an
    /// over-declared program — which it did until the ceiling check moved
    /// ahead of the capacity check. The capacity check fired first and said
    /// `ScratchTooSmall`, an error neither the oracle nor `for_program` ever
    /// produces for this input, so the two paths refused the same program
    /// with two different reasons.
    ///
    /// The assertion is the EQUALITY, not the constant. Pinning
    /// `ScratchSlotsUnaddressable` on both sides would pass if someone later
    /// changed both to the same wrong thing; asserting they agree is the law
    /// ("executor and oracle refuse identically") stated directly.
    ///
    /// The caller's scratch is deliberately REAL and small — 1 slot — which
    /// is the only way this input occurs in practice: nobody can hold an
    /// arena for a count that cannot be addressed, so the capacity check was
    /// always the one that fired.
    #[test]
    fn the_executor_and_the_oracle_refuse_an_over_declared_program_identically() {
        let zero = vec![0u64; 2];
        let masks: [&[u64]; 1] = [&zero];
        let planes = Planes {
            n_rows: 70,
            masks: &masks,
            lanes: &[],
        };
        let lying = Program {
            ops: vec![],
            terminal: Terminal::Count {
                mask: Operand::Plane(0),
            },
            scratch_slots: MAX_SCRATCH_SLOTS + 1,
        };
        let mut small = Scratch::new(2, 1);
        let from_executor = execute(&lying, &planes, &mut small, None).unwrap_err();
        let from_oracle = crate::reference::reference_execute(&lying, &planes, None).unwrap_err();
        assert_eq!(
            from_executor, from_oracle,
            "executor and oracle must refuse the same program with the same error"
        );
        // ...and it is the ceiling error, not the capacity one the pre-fix
        // ordering produced. Without this half the equality could be
        // satisfied by both paths saying `ScratchTooSmall`.
        assert_eq!(
            from_executor,
            ExecError::ScratchSlotsUnaddressable {
                declared: MAX_SCRATCH_SLOTS + 1
            }
        );
    }

    /// FAILS IF: `for_program` guards the addressable ceiling with a
    /// `debug_assert` (which release builds drop) instead of returning an
    /// error. `Program::scratch_slots` is PUBLIC, so a struct literal can
    /// declare four billion slots while naming no scratch operand; sizing an
    /// arena from that allocates unboundedly before `execute` ever validates.
    ///
    /// The bound is deliberately repeated here rather than delegated to
    /// `validate`: a caller builds its arena BEFORE `execute` runs, so a
    /// refusal that happens inside `execute` arrives after the allocation it
    /// was supposed to prevent.
    #[test]
    fn for_program_refuses_an_unaddressable_slot_count_in_every_build() {
        let lying = Program {
            ops: vec![],
            terminal: Terminal::Count {
                mask: Operand::Plane(0),
            },
            scratch_slots: MAX_SCRATCH_SLOTS + 1,
        };
        assert_eq!(
            Scratch::for_program(&lying, 70).err(),
            Some(ExecError::ScratchSlotsUnaddressable {
                declared: MAX_SCRATCH_SLOTS + 1
            })
        );
        // CAN-STAY-SILENT: an ordinary program still gets its arena, sized
        // exactly. A guard that refused everything would pass the half above.
        let ok = Program::new(
            vec![MaskOp::Not {
                a: Operand::Plane(0),
                dst: 2,
            }],
            Terminal::Count {
                mask: Operand::Scratch(2),
            },
        );
        let s = Scratch::for_program(&ok, 70).expect("addressable");
        assert_eq!(s.slots(), 3);
        assert_eq!(s.words(), 2);
    }

    /// FAILS IF: a materialiser appears that is not one of the two the crate
    /// admits — law L5. Every production `pub fn` RETURNING an owning
    /// collection must be `materialize_rows` (the consumer boundary) or
    /// `reference_scratch` (the oracle's independent reading).
    #[test]
    fn exactly_one_materialiser() {
        let mut found = Vec::new();
        for src in CRATE_SOURCES {
            let production = src.split("#[cfg(test)]").next().unwrap_or("");
            for line in production.lines() {
                let t = line.trim_start();
                // ANY owning collection RETURN, not just `Vec<usize>`: the
                // oracle's arena copy is a `Vec<Vec<u64>>`, and a guard that
                // greps the one shape it knows cannot see the next one. Split
                // on the arrow so a `Vec` PARAMETER is not read as a return.
                let returns = t.split("->").nth(1).unwrap_or("");
                if t.starts_with("pub fn ") && returns.contains("Vec<") {
                    let name = t
                        .strip_prefix("pub fn ")
                        .and_then(|r| r.split('(').next())
                        .unwrap_or(t);
                    found.push(name.to_string());
                }
            }
        }
        // `reference_scratch` is the ONE written-down exemption: the oracle
        // must hold its own unpacked reading of the arena, because an oracle
        // sharing the executor's bit packing could not falsify a packing bug
        // (law L4). Named here so the guard SEES it.
        assert_eq!(found, ["materialize_rows", "reference_scratch"]);
    }

    /// FAILS IF: `materialize_rows` reads phantom bits past `n_rows` or
    /// mis-orders rows.
    #[test]
    fn materialize_rows_is_ordered_and_bounded() {
        let mask = [0b1011u64, u64::MAX];
        assert_eq!(materialize_rows(&mask, 66), vec![0, 1, 3, 64, 65]);
        assert_eq!(materialize_rows(&mask, 2), vec![0, 1]);
        assert_eq!(materialize_rows(&[], 0), Vec::<usize>::new());
    }

    /// FAILS IF: an odd immediate's tail survives to the terminal — F-X3.
    /// `0xFF` is `true` everywhere, so over 70 rows the count must be 70, not
    /// the 128 a whole-word read of two saturated words would report. Disable:
    /// delete the `clear_tail` call in the `Ternlog` arm.
    #[test]
    fn odd_ternlog_immediate_does_not_inflate_a_count() {
        let n = 70;
        let zero = vec![0u64; 2];
        let masks: [&[u64]; 1] = [&zero];
        let planes = Planes {
            n_rows: n,
            masks: &masks,
            lanes: &[],
        };
        let p = Program::new(
            vec![MaskOp::Ternlog {
                imm: 0xFF,
                a: Operand::Plane(0),
                b: Operand::Plane(0),
                c: Operand::Plane(0),
                dst: 0,
            }],
            Terminal::Count {
                mask: Operand::Scratch(0),
            },
        );
        let mut s = Scratch::for_program(&p, n).expect("addressable");
        assert_eq!(execute(&p, &planes, &mut s, None), Ok(Value::Count(70)));
        // the all-alias shape: dst is read three times and written once
        let p2 = Program::new(
            vec![
                MaskOp::Ternlog {
                    imm: 0xFF,
                    a: Operand::Plane(0),
                    b: Operand::Plane(0),
                    c: Operand::Plane(0),
                    dst: 0,
                },
                MaskOp::Ternlog {
                    imm: 0x01,
                    a: Operand::Scratch(0),
                    b: Operand::Scratch(0),
                    c: Operand::Scratch(0),
                    dst: 0,
                },
            ],
            Terminal::Count {
                mask: Operand::Scratch(0),
            },
        );
        let mut s2 = Scratch::for_program(&p2, n).expect("addressable");
        // 0x01 is `!(x|x|x)` = `!x`: the complement of all-ones is empty
        assert_eq!(execute(&p2, &planes, &mut s2, None), Ok(Value::Count(0)));
    }

    /// FAILS IF: a program is run before its scratch is checked — the two
    /// scratch errors are the executor's own (the oracle owns no scratch).
    #[test]
    fn scratch_is_checked_before_anything_runs() {
        let zero = vec![0u64; 2];
        let masks: [&[u64]; 1] = [&zero];
        let planes = Planes {
            n_rows: 70,
            masks: &masks,
            lanes: &[],
        };
        let p = Program::new(
            vec![MaskOp::Not {
                a: Operand::Plane(0),
                dst: 3,
            }],
            Terminal::Any {
                mask: Operand::Scratch(3),
            },
        );
        let mut small = Scratch::new(2, 2);
        assert_eq!(
            execute(&p, &planes, &mut small, None),
            Err(ExecError::ScratchTooSmall { need: 4, have: 2 })
        );
        let mut wrong = Scratch::new(1, 4);
        assert_eq!(
            execute(&p, &planes, &mut wrong, None),
            Err(ExecError::ScratchWords {
                expected: 2,
                found: 1
            })
        );
        assert!(
            wrong.slot(3).is_some_and(|w| w.iter().all(|&x| x == 0)),
            "nothing was written"
        );
    }
}
