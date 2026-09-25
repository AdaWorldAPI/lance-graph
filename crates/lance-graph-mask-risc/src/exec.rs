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
    eq_u32_via_to_mask, ge_i32_to_mask, ge_i32_to_mask_under, gt_i32_to_mask, gt_i32_to_mask_under,
    le_i32_to_mask, le_i32_to_mask_under, lt_i32_to_mask, lt_i32_to_mask_under, mask_all, mask_and,
    mask_and_assign, mask_andnot, mask_andnot_assign, mask_any, mask_gather_u32, mask_not,
    mask_not_assign, mask_or, mask_or_assign, mask_scatter_or_u32, mask_set_range, mask_xor,
    mask_xor_assign, masked_group_count_u32, masked_group_count_u32_pair,
    masked_group_count_u32_via, masked_group_max_i32, masked_group_max_i32_pair,
    masked_group_max_i32_via, masked_group_min_i32, masked_group_min_i32_pair,
    masked_group_min_i32_via, masked_group_sum_i32, masked_group_sum_i32_via,
    masked_group_sum_sym_i32, masked_group_sum_sym_i32_pair, masked_group_sum_sym_i32_via,
    masked_key_run_count_u32, masked_max_i32, masked_min_i32, masked_sum_i32, ne_i32_to_mask,
    ne_i32_to_mask_under, ne_u32_to_mask, ne_u32_to_mask_under, popcount_batch_u64,
    ternary_match_u32_to_mask, ternary_match_u32_to_mask_under, ternary_match_u64_to_mask,
    ternary_match_u64_to_mask_under, KeyRunCarry,
};

use crate::ir::{
    span_words, touched_words, Compiled, Foreign, FusedFold, FusedTerminal, FusedTernlog,
    GroupFold, GroupKey, LaneRef, Lowering, MaskOp, Operand, Planes, Pred, Program, Terminal,
    FUSED_SLOT_CAP, MAX_SCRATCH_SLOTS,
};
use crate::reference::{out_shape, validate};
use crate::ternlog_dispatch::{
    ternlog_any_dispatch, ternlog_dispatch, ternlog_dispatch_assign, ternlog_popcount_dispatch,
};
use crate::value::{ExecError, Out, Value};
use crate::words_for;
use core::ops::Range;

/// Where a [`Scratch`]'s words live: owned by the arena, or borrowed from a
/// buffer the caller grows and keeps.
///
/// The borrowed form is what lets a consumer with a stable population — one
/// `n_rows` per resource, reused call after call — hold ONE growing buffer and
/// allocate nothing after the largest shape it will ever see. Allocation then
/// depends on the MAXIMUM, never on the history: a smaller program carves less
/// of the same buffer instead of asking for a new one.
#[derive(Debug)]
enum Store<'a> {
    Owned(Box<[u64]>),
    Borrowed(&'a mut [u64]),
}

impl Store<'_> {
    fn as_slice(&self) -> &[u64] {
        match self {
            Store::Owned(b) => b,
            Store::Borrowed(b) => b,
        }
    }

    fn as_mut_slice(&mut self) -> &mut [u64] {
        match self {
            Store::Owned(b) => b,
            Store::Borrowed(b) => b,
        }
    }
}

/// Total `u64` a [`Scratch`] occupies for `slots` slots of `words` each: the
/// slot arena first, then the read-before-write bitmap, in ONE region.
///
/// Public because it is the CALLER's sizing function — a consumer growing a
/// buffer for [`Scratch::over`] asks this rather than reimplementing the
/// layout, which is the only way the two can never disagree. `None` on
/// overflow, never a wrapped answer that would silently under-allocate.
/// Words per scratch slot the default constructors carve. The executor runs a
/// program one tile at a time (see [`execute_into`]), so execution state is
/// `slots × TILE_WORDS` words however many rows the planes hold — 8 words =
/// 512 rows = one full-width vector per facade call.
pub const TILE_WORDS: usize = 256;

/// The slot width [`Scratch::for_program`] / [`Scratch::over_for_program`]
/// carve for `n_rows`: one tile, or the whole (shorter) population.
pub fn tile_words_for(n_rows: usize) -> usize {
    words_for(n_rows).min(TILE_WORDS)
}

pub fn scratch_words_for(words: usize, slots: usize) -> Option<usize> {
    slots.checked_mul(words)?.checked_add(slots.div_ceil(64))
}

/// The slot arena as one op sees it: every slot EXCEPT the one being written.
///
/// This replaces a take/restore dance over `Vec<Box<[u64]>>`. With a flat
/// arena the disjointness is `split_at_mut`, so the borrow checker proves what
/// used to rest on a convention — that every `take` was paired with exactly
/// one `restore`, and that no arm read the slot it had taken.
struct Slots<'s> {
    words: usize,
    /// Slots `[0, hole)`.
    left: &'s [u64],
    /// Slots `(hole, slots)`.
    right: &'s [u64],
    hole: usize,
}

impl<'s> Slots<'s> {
    /// Every slot, no hole — for the terminal, which reads and writes nothing.
    fn whole(arena: &'s [u64], words: usize) -> Self {
        Self {
            words,
            left: arena,
            right: &[],
            hole: usize::MAX,
        }
    }

    fn get(&self, i: usize) -> &'s [u64] {
        // Reading the hole is a bug in an aliasing arm, not a possible input:
        // every `operand == dst` case is routed to an `_assign` form before it
        // reaches here. Without this the index arithmetic below underflows,
        // which panics either way — but says nothing about why.
        debug_assert!(
            i != self.hole,
            "slot {i} was read while it is this op's write target"
        );
        if self.words == 0 {
            return &[];
        }
        if i < self.hole {
            &self.left[i * self.words..][..self.words]
        } else {
            &self.right[(i - self.hole - 1) * self.words..][..self.words]
        }
    }
}

/// Caller-owned scratch: `slots` buffers of `words` u64 each, laid out in one
/// flat region so a borrowed backing store is possible at all.
///
/// The ONLY allocation in this crate's execution path, and with
/// [`Scratch::over`] not even that — the caller supplies the words.
#[derive(Debug)]
pub struct Scratch<'a> {
    words: usize,
    slots: usize,
    /// `slots * words` of slot arena, then `slots.div_ceil(64)` of
    /// read-before-write bitmap ([`crate::reference::validate`] clears and
    /// reuses the prefix it needs). One region, so ONE caller buffer serves
    /// both and there is a single sizing question, not two.
    store: Store<'a>,
}

impl Scratch<'static> {
    /// Allocate `slots` zeroed buffers of `words` u64 each.
    pub fn new(words: usize, slots: usize) -> Self {
        let total = scratch_words_for(words, slots)
            .expect("scratch size overflows usize; use `Scratch::over` with a checked buffer");
        Self {
            words,
            slots,
            store: Store::Owned(vec![0u64; total].into_boxed_slice()),
        }
    }

    /// Allocate exactly what `program` needs over `n_rows` rows: `scratch_slots`
    /// slots of ONE TILE each ([`tile_words_for`]), never the population.
    ///
    /// Fallible, and deliberately so. `Operand::Scratch` is a `u16`, so no
    /// program can ADDRESS more than [`MAX_SCRATCH_SLOTS`] slots; a larger
    /// count means a hand-built program lied about a PUBLIC field. This used
    /// to be a `debug_assert`, which release builds drop — leaving the lie to
    /// reach the allocator unbounded. The same bound is in [`validate`], so a
    /// program refused here is refused identically by `execute` and by the
    /// oracle; the check is repeated rather than delegated because a caller
    /// sizes its arena BEFORE `execute` runs, and a validation that happens
    /// afterwards cannot prevent this allocation.
    pub fn for_program(program: &Program, n_rows: usize) -> Result<Self, ExecError> {
        if program.scratch_slots > MAX_SCRATCH_SLOTS {
            return Err(ExecError::ScratchSlotsUnaddressable {
                declared: program.scratch_slots,
            });
        }
        // `scratch_slots` is a `u32` count; a program naming slot `u16::MAX`
        // needs 65,536 buffers, which fits `usize` on every supported target.
        Ok(Self::new(tile_words_for(n_rows), slots_needed(program)))
    }
}

impl<'a> Scratch<'a> {
    /// Carve a scratch out of `buf`, allocating nothing.
    ///
    /// `buf` must hold at least [`scratch_words_for`]`(words, slots)`; it may
    /// be LONGER, and everything past that prefix is never read and never
    /// written — which is what lets a caller keep one buffer grown to the
    /// largest shape it has seen and carve a smaller one out of its front.
    ///
    /// The excess is untouched rather than merely unused: `clear_tail` clears
    /// only the tail WORD while the facade's `mask_not` zeroes every word past
    /// it, and the two agree only on an exactly-sized slot. Slots are carved
    /// exact here for that reason, not for tidiness.
    pub fn over(buf: &'a mut [u64], words: usize, slots: usize) -> Result<Self, ExecError> {
        if slots as u64 > u64::from(MAX_SCRATCH_SLOTS) {
            return Err(ExecError::ScratchSlotsUnaddressable {
                declared: u32::try_from(slots).unwrap_or(u32::MAX),
            });
        }
        // Overflow is reported as a buffer that cannot be long enough,
        // because that is what it is: no allocation of any size satisfies a
        // layout whose extent does not fit `usize`.
        let need = scratch_words_for(words, slots).ok_or(ExecError::ScratchBufferTooSmall {
            need_words: usize::MAX,
            have_words: buf.len(),
        })?;
        if buf.len() < need {
            return Err(ExecError::ScratchBufferTooSmall {
                need_words: need,
                have_words: buf.len(),
            });
        }
        let region = &mut buf[..need];
        region.fill(0);
        Ok(Self {
            words,
            slots,
            store: Store::Borrowed(region),
        })
    }

    /// [`Scratch::over`] sized for `program` over `n_rows` rows — the shape a
    /// consumer calls once per evaluation against its own growing buffer.
    pub fn over_for_program(
        buf: &'a mut [u64],
        program: &Program,
        n_rows: usize,
    ) -> Result<Self, ExecError> {
        if program.scratch_slots > MAX_SCRATCH_SLOTS {
            return Err(ExecError::ScratchSlotsUnaddressable {
                declared: program.scratch_slots,
            });
        }
        Self::over(buf, tile_words_for(n_rows), slots_needed(program))
    }

    /// Words per slot.
    pub fn words(&self) -> usize {
        self.words
    }

    /// Number of slots.
    pub fn slots(&self) -> usize {
        self.slots
    }

    /// Borrow slot `i` — ONE tile wide; after a tiled run it holds the LAST
    /// tile only. A [`Terminal::Keep`] result is read from its [`Out::Mask`],
    /// or from this slot when the scratch is a single population-wide tile.
    pub fn slot(&self, i: u16) -> Option<&[u64]> {
        let i = usize::from(i);
        if i >= self.slots {
            return None;
        }
        Some(&self.store.as_slice()[i * self.words..][..self.words])
    }

    /// Length of the slot arena; the bitmap starts here.
    fn arena_len(&self) -> usize {
        self.slots * self.words
    }

    /// The read-before-write bitmap [`validate`] owns.
    fn written_mut(&mut self) -> &mut [u64] {
        let arena = self.arena_len();
        &mut self.store.as_mut_slice()[arena..]
    }

    /// Split off slot `dst` for writing, leaving every other slot readable.
    fn split(&mut self, dst: u16) -> (&mut [u64], Slots<'_>) {
        let words = self.words;
        let hole = usize::from(dst);
        let arena = self.arena_len();
        let region = &mut self.store.as_mut_slice()[..arena];
        let (left, rest) = region.split_at_mut(hole * words);
        let (mid, right) = rest.split_at_mut(words);
        (
            mid,
            Slots {
                words,
                left,
                right,
                hole,
            },
        )
    }

    /// Every slot, read-only.
    fn all(&self) -> Slots<'_> {
        Slots::whole(&self.store.as_slice()[..self.arena_len()], self.words)
    }
}

/// The ONE materialiser: row indices below `n_rows` whose bits are set in
/// `mask`, in ascending order. **O(n_rows)** and it allocates — this is the
/// boundary a consumer crosses deliberately, by this name, never as normal
/// execution state (the mask-native invariant).
///
/// The bound is the point of the sentence, not decoration: every other
/// operation in this crate is O(words) over a borrowed buffer, and the whole
/// reason this function is named rather than implicit is that it is the one
/// place a caller pays per ROW. A set bit at or past `n_rows` ends the walk
/// rather than being returned — that is the half a reader cannot get from the
/// signature.
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
    s: &mut Scratch<'_>,
    a: Operand,
    b: Operand,
    dst: u16,
    op: fn(&[u64], &[u64], &mut [u64]),
    assign: fn(&mut [u64], &[u64]),
    imm: u8,
    t: Tile,
) {
    let d = Operand::Scratch(dst);
    // Every two-input table is EVEN (`f(0,0,0) = 0`), which is why this
    // function owes no tail clear — unlike the `Ternlog` arm. Stated here
    // because it is a precondition on a private fn, not a property of the
    // caller's input.
    debug_assert!(imm & 1 == 0, "two-input table {imm:#04x} is odd");
    let commutative = remap_imm(imm, [1, 0, 2]) == imm;
    let (x, rest) = s.split(dst);
    let x = &mut x[..t.words];
    if a == d && b == d {
        ternlog_self(remap_imm(imm, [0, 0, 0]), x, t.rows);
    } else if a == d {
        assign(x, read(planes, &rest, b, t));
    } else if b == d {
        let aa = read(planes, &rest, a, t);
        if commutative {
            assign(x, aa);
        } else {
            ternlog_dispatch_assign(remap_imm(imm, [1, 0, 0]), x, aa, aa);
        }
    } else {
        op(read(planes, &rest, a, t), read(planes, &rest, b, t), x);
    }
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

/// One tile of the population: words `[w0, w0 + words)` of every mask and
/// scratch slot, rows `[r0, r0 + rows)` of every lane. The executor walks a
/// program tile by tile, so nothing it writes is ever population-sized: the
/// scratch is `slots × tile` words whatever `n_rows` is, and the only
/// population-sized writes are the DEMANDED sinks (`Out`). The last tile is
/// short when `words` is not a multiple of the slot width; `rows` is then
/// exactly the rows those words carry, so every tail law holds per tile.
#[derive(Debug, Clone, Copy)]
struct Tile {
    w0: usize,
    words: usize,
    r0: usize,
    rows: usize,
}

/// Borrow an operand for reading (the scratch arena with `dst` already taken
/// out, so a read of `dst`'s own slot here is a bug the aliasing arms prevent),
/// restricted to tile `t`.
fn read<'a>(planes: &Planes<'a>, s: &Slots<'a>, o: Operand, t: Tile) -> &'a [u64] {
    match o {
        Operand::Plane(i) => &planes.masks[usize::from(i)][t.w0..t.w0 + t.words],
        Operand::Scratch(i) => &s.get(usize::from(i))[..t.words],
    }
}

/// Rows `[r0, r0 + rows)` of `v`, or the empty slice when the lane is short —
/// unreachable after `validate`, but a slice out of range would panic where
/// an empty lane makes every facade call a no-op of the right shape.
fn rows_of<T>(v: &[T], t: Tile) -> &[T] {
    v.get(t.r0..t.r0 + t.rows).unwrap_or(&[])
}

fn lane_i32<'a>(planes: &Planes<'a>, lane: u16, t: Tile) -> &'a [i32] {
    match planes.lanes[usize::from(lane)] {
        LaneRef::I32(v) => rows_of(v, t),
        _ => &[],
    }
}

fn lane_u32<'a>(planes: &Planes<'a>, lane: u16, t: Tile) -> &'a [u32] {
    match planes.lanes[usize::from(lane)] {
        LaneRef::U32(v) => rows_of(v, t),
        _ => &[],
    }
}

fn lane_u64<'a>(planes: &Planes<'a>, lane: u16, t: Tile) -> &'a [u64] {
    match planes.lanes[usize::from(lane)] {
        LaneRef::U64(v) => rows_of(v, t),
        _ => &[],
    }
}

/// Borrow a foreign `U32` lane for reading — [`lane_u32`]'s twin over
/// [`Foreign::lanes`], a SEPARATE address space from `planes.lanes` (see
/// [`Foreign`]'s own doc). WHOLE, never tiled: it is addressed through a
/// foreign key, not by this population's row. Unreachable after `validate`
/// on a well-typed program, same fallback discipline as the lane helpers.
fn foreign_lane_u32<'a>(foreign: &Foreign<'a>, key: u16) -> &'a [u32] {
    match foreign.lanes[usize::from(key)] {
        LaneRef::U32(v) => v,
        _ => &[],
    }
}

/// One predicate pass into `dst` (one tile of it): the ungated facade member,
/// or the `_under` member when a gate is present (cost then follows the
/// gate's live words).
fn run_pred<'a>(
    planes: &Planes<'a>,
    foreign: &Foreign<'a>,
    s: &Slots<'a>,
    pred: Pred,
    under: Option<Operand>,
    dst: &mut [u64],
    t: Tile,
) {
    match (pred, under) {
        // The join filter in factored form: the foreign lane is read WHOLE
        // (addressed by the key), this table's fk one tile at a time. The
        // gate is applied after, as for `Range` — there is no `_under` twin
        // and one AND over a tile costs less than a second facade word.
        (Pred::EqU32Via { fk, key, v }, under) => {
            eq_u32_via_to_mask(
                lane_u32(planes, fk, t),
                foreign_lane_u32(foreign, key),
                v,
                dst,
            );
            if let Some(u) = under {
                mask_and_assign(dst, read(planes, s, u, t));
            }
        }
        (Pred::GtI32 { lane, t: v }, None) => gt_i32_to_mask(lane_i32(planes, lane, t), v, dst),
        (Pred::GtI32 { lane, t: v }, Some(u)) => {
            gt_i32_to_mask_under(lane_i32(planes, lane, t), v, read(planes, s, u, t), dst)
        }
        (Pred::LtI32 { lane, t: v }, None) => lt_i32_to_mask(lane_i32(planes, lane, t), v, dst),
        (Pred::LtI32 { lane, t: v }, Some(u)) => {
            lt_i32_to_mask_under(lane_i32(planes, lane, t), v, read(planes, s, u, t), dst)
        }
        (Pred::GeI32 { lane, t: v }, None) => ge_i32_to_mask(lane_i32(planes, lane, t), v, dst),
        (Pred::GeI32 { lane, t: v }, Some(u)) => {
            ge_i32_to_mask_under(lane_i32(planes, lane, t), v, read(planes, s, u, t), dst)
        }
        (Pred::LeI32 { lane, t: v }, None) => le_i32_to_mask(lane_i32(planes, lane, t), v, dst),
        (Pred::LeI32 { lane, t: v }, Some(u)) => {
            le_i32_to_mask_under(lane_i32(planes, lane, t), v, read(planes, s, u, t), dst)
        }
        (Pred::EqI32 { lane, v }, None) => eq_i32_to_mask(lane_i32(planes, lane, t), v, dst),
        (Pred::EqI32 { lane, v }, Some(u)) => {
            eq_i32_to_mask_under(lane_i32(planes, lane, t), v, read(planes, s, u, t), dst)
        }
        (Pred::NeI32 { lane, v }, None) => ne_i32_to_mask(lane_i32(planes, lane, t), v, dst),
        (Pred::NeI32 { lane, v }, Some(u)) => {
            ne_i32_to_mask_under(lane_i32(planes, lane, t), v, read(planes, s, u, t), dst)
        }
        (Pred::EqU32 { lane, v }, None) => eq_u32_to_mask(lane_u32(planes, lane, t), v, dst),
        (Pred::EqU32 { lane, v }, Some(u)) => {
            eq_u32_to_mask_under(lane_u32(planes, lane, t), v, read(planes, s, u, t), dst)
        }
        (Pred::NeU32 { lane, v }, None) => ne_u32_to_mask(lane_u32(planes, lane, t), v, dst),
        (Pred::NeU32 { lane, v }, Some(u)) => {
            ne_u32_to_mask_under(lane_u32(planes, lane, t), v, read(planes, s, u, t), dst)
        }
        (
            Pred::MatchU32 {
                lane,
                pattern,
                care,
            },
            None,
        ) => ternary_match_u32_to_mask(lane_u32(planes, lane, t), pattern, care, dst),
        (
            Pred::MatchU32 {
                lane,
                pattern,
                care,
            },
            Some(u),
        ) => ternary_match_u32_to_mask_under(
            lane_u32(planes, lane, t),
            pattern,
            care,
            read(planes, s, u, t),
            dst,
        ),
        (
            Pred::MatchU64 {
                lane,
                pattern,
                care,
            },
            None,
        ) => ternary_match_u64_to_mask(lane_u64(planes, lane, t), pattern, care, dst),
        (
            Pred::MatchU64 {
                lane,
                pattern,
                care,
            },
            Some(u),
        ) => ternary_match_u64_to_mask_under(
            lane_u64(planes, lane, t),
            pattern,
            care,
            read(planes, s, u, t),
            dst,
        ),
        // `lo <= hi <= n_rows` was validated; clipped to this tile the range
        // is in-range for the tile's words and `mask_set_range`'s own asserts
        // cannot fire. A range that misses the tile entirely clips to an empty
        // one, which clears the tile — the correct answer for it.
        (Pred::Range { lo, hi }, under) => {
            let clip = |r: u32| (r as usize).clamp(t.r0, t.r0 + t.rows) - t.r0;
            mask_set_range(dst, clip(lo), clip(hi));
            if let Some(u) = under {
                mask_and_assign(dst, read(planes, s, u, t));
            }
        }
    }
}

/// Run `program` over `planes` with the caller's `scratch`; `out` is the
/// destination a [`Terminal::BlendI32`] writes. Validation is total and
/// happens before any result write, so an `Err` leaves the scratch slots and
/// `out` untouched (the one run-time refusal, and what it may leave in
/// scratch, is documented on [`execute_into`]).
///
/// A thin wrapper over [`execute_into`]: no foreign planes, and `out` widened
/// to [`Out::I32`] (or [`Out::None`] for `out: None`) — the shape every
/// caller of this crate already had before [`MaskOp::Gather`] existed.
pub fn execute(
    program: &Program,
    planes: &Planes<'_>,
    scratch: &mut Scratch<'_>,
    out: Option<&mut [i32]>,
) -> Result<Value, ExecError> {
    execute_into(
        program,
        planes,
        &Foreign::NONE,
        scratch,
        out.map_or(Out::None, Out::I32),
    )
}

/// Slots [`Scratch::for_program`] / [`Scratch::over_for_program`] carve: none
/// for a program [`Program::requires_scratch`] says needs none.
fn slots_needed(program: &Program) -> usize {
    if program.requires_scratch() {
        program.scratch_slots as usize
    } else {
        0
    }
}

/// The bits of word `w` that fall inside `[lo, hi)` (absolute rows; `w` must
/// be a word the range touches, so `hi > w * 64`).
fn edge_mask(w: usize, lo: usize, hi: usize) -> u64 {
    let base = w * 64;
    let from = lo.saturating_sub(base).min(64);
    let to = (hi - base).min(64);
    let upper = if to == 64 { u64::MAX } else { (1u64 << to) - 1 };
    upper & (u64::MAX << from)
}

/// Evaluate a [`FusedTerminal`] over the resident plane's touched words.
///
/// Nothing is written: the interior words are read from the borrowed plane
/// as they are (a range's interior mask is all ones), and the two edge words
/// are masked in a register. `validate` has already proven `lo <= hi <=
/// n_rows` and the plane index in range.
fn run_fused(f: FusedTerminal, planes: &Planes<'_>) -> Value {
    let span = touched_words(f.lo, f.hi);
    let Some(p) = f.plane else {
        return match f.fold {
            FusedFold::Count => Value::Count((f.hi - f.lo) as usize),
            FusedFold::Any => Value::Bool(f.lo < f.hi),
        };
    };
    if span.is_empty() {
        return match f.fold {
            FusedFold::Count => Value::Count(0),
            FusedFold::Any => Value::Bool(false),
        };
    }
    let plane = planes.masks[usize::from(p)];
    let (first, last) = (span.start, span.end - 1);
    let (lo, hi) = (f.lo as usize, f.hi as usize);
    let head = [plane[first] & edge_mask(first, lo, hi)];
    let tail = [plane[last] & edge_mask(last, lo, hi)];
    let interior = if last > first + 1 {
        &plane[first + 1..last]
    } else {
        &[][..]
    };
    match f.fold {
        FusedFold::Count => {
            let mut n = popcount_batch_u64(&head) + popcount_batch_u64(interior);
            if last != first {
                n += popcount_batch_u64(&tail);
            }
            Value::Count(n as usize)
        }
        FusedFold::Any => {
            Value::Bool(mask_any(&head) || mask_any(interior) || (last != first && mask_any(&tail)))
        }
    }
}

/// Evaluate a [`FusedTernlog`] over the absolute extent `[lo, hi)`.
///
/// Nothing is written. Whole words inside the extent go straight to
/// `ndarray::simd::mask_ternlog_{popcount,any}` over the borrowed resident
/// planes. A word the extent cuts — an unaligned `lo`, or an unaligned `hi`,
/// which includes the population's own last word when `n_rows % 64 != 0` — is
/// combined one word at a time into a one-word register and restricted by
/// [`edge_mask`] before it is counted. That last case is load-bearing for an
/// ODD table: `f(0,0,0) = 1` sets the dead tail bits of the last word, and
/// only the edge mask keeps them out of the count (the tiled path clears the
/// same tail against `n_rows`).
fn run_fused_ternlog(f: FusedTernlog, planes: &Planes<'_>, lo: usize, hi: usize) -> Value {
    let span = span_words(lo, hi);
    let empty = match f.fold {
        FusedFold::Count => Value::Count(0),
        FusedFold::Any => Value::Bool(false),
    };
    if span.is_empty() {
        return empty;
    }
    let (a, b, c) = (
        planes.masks[usize::from(f.a)],
        planes.masks[usize::from(f.b)],
        planes.masks[usize::from(f.c)],
    );
    let (first, last) = (span.start, span.end - 1);
    let head_cut = !lo.is_multiple_of(64);
    let tail_cut = !hi.is_multiple_of(64);
    // One cut word, combined in a register and restricted to the extent.
    let edge = |w: usize| -> u64 {
        let mut reg = [0u64; 1];
        ternlog_dispatch(f.imm, &a[w..=w], &b[w..=w], &c[w..=w], &mut reg);
        reg[0] & edge_mask(w, lo, hi)
    };
    let mut edges = [None, None];
    if head_cut {
        edges[0] = Some(edge(first));
    }
    if tail_cut && (last != first || !head_cut) {
        edges[1] = Some(edge(last));
    }
    let from = first + usize::from(head_cut);
    let to = (last + 1).saturating_sub(usize::from(tail_cut)).max(from);
    let interior = from..to;
    match f.fold {
        FusedFold::Count => {
            let e: u64 = edges
                .iter()
                .flatten()
                .map(|w| u64::from(w.count_ones()))
                .sum();
            let n = ternlog_popcount_dispatch(
                f.imm,
                &a[interior.clone()],
                &b[interior.clone()],
                &c[interior],
            );
            Value::Count((e + n) as usize)
        }
        FusedFold::Any => Value::Bool(
            edges.iter().flatten().any(|&w| w != 0)
                || ternlog_any_dispatch(
                    f.imm,
                    &a[interior.clone()],
                    &b[interior.clone()],
                    &c[interior],
                ),
        ),
    }
}

/// The tiles an execution over the ABSOLUTE row extent `[lo, hi)` visits,
/// each as `(word range, edge)`.
///
/// The extent is an outer restriction in the same row coordinates as
/// [`Planes`]: tile word `w` is word `w` of every resident mask, and its rows
/// are `w * 64 ..` of every lane — nothing is rebased, copied, or renumbered.
/// Only the words [`span_words`] says the extent touches are visited. A word
/// the extent cuts (an unaligned `lo`, or an unaligned `hi` short of
/// `n_rows`) is its own one-word tile carrying `Some(edge)`, the in-extent
/// bits the terminal is restricted to; every other tile is at most
/// `tile_words` whole words and carries `None`. For the whole population
/// `[0, n_rows)` there is no edge, and the tiles are exactly the ones
/// whole-population execution has always walked.
///
/// Crate-private: tile width and edge representation are executor
/// implementation detail, not API. The structural claim — work scales with
/// the extent, not with `n_rows` — is pinned by the in-crate test
/// `extent_tile_tests` against this exact plan.
pub(crate) fn extent_tiles(n_rows: usize, tile_words: usize, extent: Range<usize>) -> ExtentTiles {
    let span = span_words(extent.start, extent.end);
    ExtentTiles {
        first: span.start,
        cur: span.start,
        end: span.end,
        lo: extent.start,
        hi: extent.end,
        head_edge: !extent.start.is_multiple_of(64),
        tail_edge: !extent.end.is_multiple_of(64) && extent.end < n_rows,
        tile_words: tile_words.max(1),
    }
}

/// Iterator returned by [`extent_tiles`].
#[derive(Debug, Clone)]
pub(crate) struct ExtentTiles {
    first: usize,
    cur: usize,
    end: usize,
    lo: usize,
    hi: usize,
    head_edge: bool,
    tail_edge: bool,
    tile_words: usize,
}

impl Iterator for ExtentTiles {
    type Item = (Range<usize>, Option<u64>);

    fn next(&mut self) -> Option<Self::Item> {
        if self.cur >= self.end {
            return None;
        }
        let w = self.cur;
        if (w == self.first && self.head_edge) || (w + 1 == self.end && self.tail_edge) {
            self.cur += 1;
            return Some((w..w + 1, Some(edge_mask(w, self.lo, self.hi))));
        }
        let stop = if self.tail_edge {
            self.end - 1
        } else {
            self.end
        };
        let n = self.tile_words.min(stop - w);
        self.cur += n;
        Some((w..w + n, None))
    }
}

/// A terminal's mask on an edge tile, restricted to the extent. `fill`
/// reads the out-of-extent rows as SET — the identity of `All` — every other
/// terminal reads them as CLEAR. A one-word register temporary; nothing
/// resident is written.
fn clip<'a>(m: &'a [u64], edge: Option<u64>, buf: &'a mut [u64; 1], fill: bool) -> &'a [u64] {
    match edge {
        None => m,
        Some(e) => {
            buf[0] = if fill { m[0] | !e } else { m[0] & e };
            &buf[..]
        }
    }
}

/// Fold one tile's `Option` reduction into the running one.
fn fold_opt(acc: Option<i32>, tile: Option<i32>, f: fn(i32, i32) -> i32) -> Option<i32> {
    match (acc, tile) {
        (Some(a), Some(b)) => Some(f(a, b)),
        (a, None) => a,
        (None, b) => b,
    }
}

/// Run `program` over `planes` with the caller's `scratch`, resolving any
/// [`MaskOp::Gather`] / [`Terminal::GroupSumViaI32`] against `foreign`;
/// `out` is the DEMANDED sink — the destination a [`Terminal::BlendI32`] /
/// [`Terminal::ScatterOrU32`] / [`Terminal::GroupSumI32`] /
/// [`Terminal::GroupSumViaI32`] / [`Terminal::Keep`] writes (every other
/// terminal ignores it — an `Out` of the wrong shape for the terminal that IS
/// present is refused by [`validate`], never silently accepted and silently
/// not written). Validation is total and happens before any result write, so
/// a validation `Err` leaves the scratch slots and `out` untouched. ONE
/// refusal is raised at run time instead: [`ExecError::LaneNotOrdered`] from
/// [`Terminal::CountKeyRunsU32`], at the first key descent, mid-walk. That
/// terminal has no sink, so `out` is still untouched; the scratch slots may
/// hold the tiles walked before the descent — scratch is ALU state, never a
/// result, and nothing reads it back as one.
///
/// **Execution is tiled.** The program runs once per `scratch.words()`-word
/// tile of the population; every op writes at most one tile of its slot, and
/// the terminal folds each tile into O(1) accumulators or into `out`. Nothing
/// population-sized is written except `out` itself — a mask exists as a
/// whole only where a caller demanded it ([`Out::Mask`]). A caller who carves
/// a scratch `words_for(n_rows)` wide gets one tile, i.e. the old
/// whole-population behaviour, and may then read a [`Terminal::Keep`] result
/// from its slot with `out: Out::None`; under a narrower scratch `Keep`
/// requires `Out::Mask` (`TerminalNeedsOut`).
pub fn execute_into(
    program: &Program,
    planes: &Planes<'_>,
    foreign: &Foreign<'_>,
    scratch: &mut Scratch<'_>,
    out: Out<'_>,
) -> Result<Value, ExecError> {
    execute_extent(program, planes, foreign, scratch, out, 0..planes.n_rows)
}

/// [`execute_into`] restricted to the ABSOLUTE row extent `[lo, hi)`.
///
/// **The extent is an outer restriction, never a rebasing.** It lives in the
/// same row coordinates as `planes`: a `Pred::Range { lo: 1000, hi: 2000 }`
/// executed over the extent `1500..1700` means `[1000, 2000) ∩ [1500, 1700)`,
/// and lane element `r` is row `r` whatever the extent. The program's
/// meaning is unchanged; only the population it is evaluated over shrinks.
///
/// Work is proportional to the extent: only the tiles [`extent_tiles`]
/// yields are visited, and a word the extent cuts is restricted in a
/// register at the terminal. No lane or mask is copied or rebased.
///
/// `execute_into` is this call with `0..n_rows`, which accepts every
/// terminal. A partial extent accepts the terminals whose per-extent results
/// merge by a shipped law — `Count` (sum), `Any` (or), `All` (and),
/// `MaskedSumI32` (sum), `MaskedMinI32` / `MaskedMaxI32` (min / max) — plus
/// `Keep`, which writes only the in-extent bits of its population-addressed
/// [`Out::Mask`] and leaves every other bit as the caller holds it (so
/// disjoint extents compose into one buffer in any SEQUENTIAL order). Anything else is
/// [`ExecError::ExtentUnsupported`]; `lo > hi` or `hi > n_rows` is
/// [`ExecError::ExtentOutOfRange`]. Both are refused before execution.
///
/// **Sequential, not concurrent.** An unaligned boundary puts two extents in
/// one physical `u64` of the `Keep` sink, and the edge merge is a
/// read-modify-write. Partial `Keep` sinks compose in any sequential order;
/// concurrent execution requires word-disjoint sink ownership (boundaries on
/// multiples of 64), separate partial sinks plus a merge, or another
/// explicitly synchronized strategy. Nothing here licenses two writers on one
/// `Out::Mask` at once.
///
/// Foreign planes and lanes ([`MaskOp::Gather`], `GroupKey::Via`) are
/// addressed by KEY, not by this table's rows, so the extent never slices
/// them.
pub fn execute_extent(
    program: &Program,
    planes: &Planes<'_>,
    foreign: &Foreign<'_>,
    scratch: &mut Scratch<'_>,
    out: Out<'_>,
    extent: Range<usize>,
) -> Result<Value, ExecError> {
    execute_compiled(&program.compile(), planes, foreign, scratch, out, extent)
}

/// [`execute_extent`] over a program whose lowering was recognised once
/// ([`Program::compile`]). Identical results; the only difference is that
/// this call does not re-derive the fold from the program text. Validation
/// still runs, against this call's planes, foreign tables and `out`.
pub fn execute_compiled(
    compiled: &Compiled<'_>,
    planes: &Planes<'_>,
    foreign: &Foreign<'_>,
    scratch: &mut Scratch<'_>,
    mut out: Out<'_>,
    extent: Range<usize>,
) -> Result<Value, ExecError> {
    let program = compiled.program();
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
    let (elo, ehi) = (extent.start, extent.end);
    if elo > ehi || ehi > planes.n_rows {
        return Err(ExecError::ExtentOutOfRange {
            lo: elo,
            hi: ehi,
            n_rows: planes.n_rows,
        });
    }
    let whole = elo == 0 && ehi == planes.n_rows;
    if !whole {
        let refused = match program.terminal {
            Terminal::Count { .. }
            | Terminal::Any { .. }
            | Terminal::All { .. }
            | Terminal::MaskedSumI32 { .. }
            | Terminal::MaskedMinI32 { .. }
            | Terminal::MaskedMaxI32 { .. }
            | Terminal::Keep { .. } => None,
            Terminal::BlendI32 { .. } => Some("BlendI32"),
            Terminal::ScatterOrU32 { .. } => Some("ScatterOrU32"),
            Terminal::ScatterCountU32 { .. } => Some("ScatterCountU32"),
            Terminal::CountKeyRunsU32 { .. } => Some("CountKeyRunsU32"),
            Terminal::GroupSumI32 { .. } => Some("GroupSumI32"),
            Terminal::GroupSumViaI32 { .. } => Some("GroupSumViaI32"),
            Terminal::GroupReduce { .. } => Some("GroupReduce"),
        };
        if let Some(what) = refused {
            return Err(ExecError::ExtentUnsupported { what });
        }
    }
    // A fused program folds from its operands: it reads no slot and writes no
    // membership bit, so the scratch capacity checks below do not apply to it.
    // Validation stays total — the one declared slot is tracked in a local
    // word of read-before-write bookkeeping, never in the caller's arena.
    if let Lowering::Range(f) = compiled.lowering() {
        let mut written = [0u64; FUSED_SLOT_CAP.div_ceil(64)];
        validate(program, planes, foreign, out_shape(&out), &mut written)?;
        // The extent composes with the program's own range by intersection,
        // in absolute rows — the #1268 fold, over a narrower span.
        let (a, b) = ((f.lo as usize).max(elo), (f.hi as usize).min(ehi));
        let f = if a < b {
            FusedTerminal {
                lo: a as u32,
                hi: b as u32,
                ..f
            }
        } else {
            FusedTerminal { hi: f.lo, ..f }
        };
        return Ok(run_fused(f, planes));
    }
    // The Boolean-membership fold: a chain of Boolean ops over at most three
    // resident planes, collapsed symbolically to one ternlog table (#1272) and
    // folded by Count/Any — also no slot, no membership bit written.
    if let Lowering::Ternlog(f) = compiled.lowering() {
        let mut written = [0u64; FUSED_SLOT_CAP.div_ceil(64)];
        validate(program, planes, foreign, out_shape(&out), &mut written)?;
        return Ok(run_fused_ternlog(f, planes, elo, ehi));
    }
    if scratch.slots() < program.scratch_slots as usize {
        return Err(ExecError::ScratchTooSmall {
            need: program.scratch_slots,
            have: scratch.slots(),
        });
    }
    let words = words_for(planes.n_rows);
    let tw = scratch.words;
    if words > 0 && tw == 0 {
        return Err(ExecError::ScratchWords {
            expected: 1,
            found: 0,
        });
    }
    validate(
        program,
        planes,
        foreign,
        out_shape(&out),
        scratch.written_mut(),
    )?;
    // Under tiling a `Keep` result exists as a whole ONLY in a demanded
    // `Out::Mask`: the scratch slot holds the last tile alone, so any other
    // `out` shape (`None`, or the `I32` the legacy `execute` wrapper passes
    // for every terminal) would return `Value::Mask` over a partial result.
    if matches!(program.terminal, Terminal::Keep { .. })
        && !matches!(out, Out::Mask(_))
        && (tw < words || !whole)
    {
        return Err(ExecError::TerminalNeedsOut { what: "Keep" });
    }
    // The sinks that ACCUMULATE across tiles start from zero here, once. The
    // sink is the requested result, so this is a write the law permits; the
    // facade kernels themselves never zero a destination they only add to.
    match (&program.terminal, &mut out) {
        (Terminal::ScatterOrU32 { .. } | Terminal::ScatterCountU32 { .. }, Out::Mask(o)) => {
            o.fill(0)
        }
        (Terminal::GroupSumI32 { .. } | Terminal::GroupSumViaI32 { .. }, Out::I64(o)) => o.fill(0),
        (Terminal::GroupReduce { fold, .. }, Out::I64(o)) => o.fill(fold.seed()),
        _ => {}
    }
    let n_rows = planes.n_rows;
    let mut count = 0usize;
    let mut any = false;
    let mut all = true;
    let mut sum = 0i64;
    let mut min: Option<i32> = None;
    let mut max: Option<i32> = None;
    // The whole state of a key-ORDERED distinct count: the open run's key
    // and whether it was hit. Two words, however many rows.
    let mut run_carry = KeyRunCarry::default();
    let mut runs = 0usize;

    for (span, edge) in extent_tiles(n_rows, tw, elo..ehi) {
        let (w0, tws) = (span.start, span.len());
        let r0 = w0 * 64;
        let t = Tile {
            w0,
            words: tws,
            r0,
            rows: (n_rows - r0).min(tws * 64),
        };
        // The register an edge tile's terminal mask is restricted in.
        let mut eb = [0u64; 1];
        for op in &program.ops {
            match *op {
                MaskOp::Pred { pred, under, dst } => {
                    let (d, rest) = scratch.split(dst);
                    run_pred(planes, foreign, &rest, pred, under, &mut d[..t.words], t);
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
                    t,
                ),
                MaskOp::Or { a, b, dst } => two_input(
                    planes,
                    scratch,
                    a,
                    b,
                    dst,
                    mask_or,
                    mask_or_assign,
                    OR_IMM,
                    t,
                ),
                MaskOp::Xor { a, b, dst } => two_input(
                    planes,
                    scratch,
                    a,
                    b,
                    dst,
                    mask_xor,
                    mask_xor_assign,
                    XOR_IMM,
                    t,
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
                    t,
                ),
                MaskOp::Not { a, dst } => {
                    let (d, rest) = scratch.split(dst);
                    let d = &mut d[..t.words];
                    if a == Operand::Scratch(dst) {
                        mask_not_assign(d, t.rows);
                    } else {
                        mask_not(read(planes, &rest, a, t), t.rows, d);
                    }
                }
                MaskOp::Ternlog { imm, a, b, c, dst } => {
                    let d = Operand::Scratch(dst);
                    let (x, rest) = scratch.split(dst);
                    let x = &mut x[..t.words];
                    if a != d && b != d && c != d {
                        ternlog_dispatch(
                            imm,
                            read(planes, &rest, a, t),
                            read(planes, &rest, b, t),
                            read(planes, &rest, c, t),
                            x,
                        );
                    } else {
                        // `dst` is an input: the in-place form takes it as
                        // `x`, the remaining distinct operands become `y`
                        // (and `z`), and the table is re-indexed to match.
                        let mut map = [0u8; 3];
                        let mut others: [Option<Operand>; 2] = [None, None];
                        let mut n = 0usize;
                        for (i, o) in [a, b, c].into_iter().enumerate() {
                            if o == d {
                                map[i] = 0;
                            } else if let Some(pos) = others[..n].iter().position(|&p| p == Some(o))
                            {
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
                                x,
                                read(planes, &rest, y, t),
                                read(planes, &rest, z, t),
                            ),
                            (Some(y), None) => {
                                let yy = read(planes, &rest, y, t);
                                ternlog_dispatch_assign(imm2, x, yy, yy)
                            }
                            _ => ternlog_self(imm2, x, t.rows),
                        }
                    }
                    if imm & 1 == 1 {
                        clear_tail(x, t.rows);
                    }
                }
                MaskOp::Gather {
                    lane,
                    foreign: fidx,
                    dst,
                } => {
                    // No aliasing shape to consider: unlike every other op,
                    // `Gather` never reads `dst` as an input — it only writes
                    // it — so there is no `dst == a` case to route to an
                    // in-place facade form. The foreign plane is read WHOLE
                    // (it is addressed by the key, not by this tile's rows).
                    let (d, _rest) = scratch.split(dst);
                    let fp = &foreign.planes[usize::from(fidx)];
                    mask_gather_u32(
                        fp.words,
                        fp.rows,
                        lane_u32(planes, lane, t),
                        &mut d[..t.words],
                    );
                }
            }
        }

        let slots = scratch.all();
        match program.terminal {
            Terminal::Count { mask } => {
                count +=
                    popcount_batch_u64(clip(read(planes, &slots, mask, t), edge, &mut eb, false))
                        as usize;
            }
            Terminal::Any { mask } => {
                any |= mask_any(clip(read(planes, &slots, mask, t), edge, &mut eb, false))
            }
            Terminal::All { mask } => {
                all &= mask_all(
                    clip(read(planes, &slots, mask, t), edge, &mut eb, true),
                    t.rows,
                )
            }
            Terminal::MaskedSumI32 { mask, lane } => {
                sum += masked_sum_i32(
                    lane_i32(planes, lane, t),
                    clip(read(planes, &slots, mask, t), edge, &mut eb, false),
                );
            }
            Terminal::MaskedMinI32 { mask, lane } => {
                min = fold_opt(
                    min,
                    masked_min_i32(
                        lane_i32(planes, lane, t),
                        clip(read(planes, &slots, mask, t), edge, &mut eb, false),
                    ),
                    i32::min,
                );
            }
            Terminal::MaskedMaxI32 { mask, lane } => {
                max = fold_opt(
                    max,
                    masked_max_i32(
                        lane_i32(planes, lane, t),
                        clip(read(planes, &slots, mask, t), edge, &mut eb, false),
                    ),
                    i32::max,
                );
            }
            Terminal::BlendI32 { mask, then, els } => {
                // `validate` already refused a missing or mis-shaped `out`.
                if let Out::I32(o) = &mut out {
                    blend_i32(
                        read(planes, &slots, mask, t),
                        lane_i32(planes, then, t),
                        lane_i32(planes, els, t),
                        &mut o[t.r0..t.r0 + t.rows],
                    );
                }
            }
            Terminal::ScatterOrU32 {
                mask,
                lane,
                out_rows,
            }
            | Terminal::ScatterCountU32 {
                mask,
                lane,
                out_rows,
            } => {
                // `validate` already refused a missing or mis-sized `out`;
                // the kernel ORs into it, tile after tile. For the count
                // form the buffer is the fold's accumulator and only its
                // popcount leaves.
                if let Out::Mask(o) = &mut out {
                    mask_scatter_or_u32(
                        read(planes, &slots, mask, t),
                        lane_u32(planes, lane, t),
                        o,
                        out_rows as usize,
                    );
                }
            }
            Terminal::CountKeyRunsU32 { mask, lane } => {
                // A key below the open run's key means the lane is not in
                // key order: refuse, never over-count. Sinks are untouched
                // (this terminal has none); scratch is ALU state and may
                // hold the tiles walked so far (see `execute_into`'s doc).
                match masked_key_run_count_u32(
                    lane_u32(planes, lane, t),
                    read(planes, &slots, mask, t),
                    &mut run_carry,
                ) {
                    Some(closed) => runs += closed,
                    None => return Err(ExecError::LaneNotOrdered { lane }),
                }
            }
            Terminal::GroupSumI32 { mask, key, val } => {
                // `validate` already refused a missing or too-small `out`;
                // the kernel adds into it, tile after tile.
                if let Out::I64(o) = &mut out {
                    masked_group_sum_i32(
                        read(planes, &slots, mask, t),
                        lane_u32(planes, key, t),
                        lane_i32(planes, val, t),
                        o,
                    );
                }
            }
            Terminal::GroupSumViaI32 { mask, fk, key, val } => {
                // `validate` already refused a missing/too-small `out`, a
                // wrong-width `fk`/`val`, and an out-of-range or wrong-width
                // foreign `key` — one delegation per tile (law L3).
                if let Out::I64(o) = &mut out {
                    masked_group_sum_i32_via(
                        read(planes, &slots, mask, t),
                        lane_u32(planes, fk, t),
                        foreign_lane_u32(foreign, key),
                        lane_i32(planes, val, t),
                        o,
                    );
                }
            }
            Terminal::GroupReduce { mask, key, fold } => {
                // `validate` already refused a missing/too-small `out` and
                // every wrong-width lane; one delegation per tile (law L3),
                // the sink seeded above with the fold's identity.
                if let Out::I64(o) = &mut out {
                    let m = read(planes, &slots, mask, t);
                    match (key, fold) {
                        (GroupKey::Lane(k), GroupFold::Count) => {
                            masked_group_count_u32(m, lane_u32(planes, k, t), o)
                        }
                        (GroupKey::Via { fk, key }, GroupFold::Count) => {
                            masked_group_count_u32_via(
                                m,
                                lane_u32(planes, fk, t),
                                foreign_lane_u32(foreign, key),
                                o,
                            )
                        }
                        (GroupKey::Lane(k), GroupFold::MinI32(v)) => masked_group_min_i32(
                            m,
                            lane_u32(planes, k, t),
                            lane_i32(planes, v, t),
                            o,
                        ),
                        (GroupKey::Via { fk, key }, GroupFold::MinI32(v)) => {
                            masked_group_min_i32_via(
                                m,
                                lane_u32(planes, fk, t),
                                foreign_lane_u32(foreign, key),
                                lane_i32(planes, v, t),
                                o,
                            )
                        }
                        (GroupKey::Lane(k), GroupFold::MaxI32(v)) => masked_group_max_i32(
                            m,
                            lane_u32(planes, k, t),
                            lane_i32(planes, v, t),
                            o,
                        ),
                        (GroupKey::Via { fk, key }, GroupFold::MaxI32(v)) => {
                            masked_group_max_i32_via(
                                m,
                                lane_u32(planes, fk, t),
                                foreign_lane_u32(foreign, key),
                                lane_i32(planes, v, t),
                                o,
                            )
                        }
                        (GroupKey::Lane(k), GroupFold::SumSymI32(v)) => masked_group_sum_sym_i32(
                            m,
                            lane_u32(planes, k, t),
                            lane_i32(planes, v, t),
                            o,
                        ),
                        (GroupKey::Via { fk, key }, GroupFold::SumSymI32(v)) => {
                            masked_group_sum_sym_i32_via(
                                m,
                                lane_u32(planes, fk, t),
                                foreign_lane_u32(foreign, key),
                                lane_i32(planes, v, t),
                                o,
                            )
                        }
                        (GroupKey::Pair { hi, lo, stride }, GroupFold::Count) => {
                            masked_group_count_u32_pair(
                                m,
                                lane_u32(planes, hi, t),
                                lane_u32(planes, lo, t),
                                stride,
                                o,
                            )
                        }
                        (GroupKey::Pair { hi, lo, stride }, GroupFold::MinI32(v)) => {
                            masked_group_min_i32_pair(
                                m,
                                lane_u32(planes, hi, t),
                                lane_u32(planes, lo, t),
                                stride,
                                lane_i32(planes, v, t),
                                o,
                            )
                        }
                        (GroupKey::Pair { hi, lo, stride }, GroupFold::MaxI32(v)) => {
                            masked_group_max_i32_pair(
                                m,
                                lane_u32(planes, hi, t),
                                lane_u32(planes, lo, t),
                                stride,
                                lane_i32(planes, v, t),
                                o,
                            )
                        }
                        (GroupKey::Pair { hi, lo, stride }, GroupFold::SumSymI32(v)) => {
                            masked_group_sum_sym_i32_pair(
                                m,
                                lane_u32(planes, hi, t),
                                lane_u32(planes, lo, t),
                                stride,
                                lane_i32(planes, v, t),
                                o,
                            )
                        }
                    }
                }
            }
            Terminal::Keep { mask } => {
                // The demanded mask, one tile at a time. With `Out::None` the
                // scratch is single-tile (checked above) and the slot IS the
                // result.
                if let Out::Mask(o) = &mut out {
                    let m = read(planes, &slots, mask, t);
                    match edge {
                        None => o[t.w0..t.w0 + t.words].copy_from_slice(m),
                        // An edge word writes only its in-extent bits: the
                        // neighbour extent's bits in the same word survive.
                        Some(e) => o[t.w0] = (o[t.w0] & !e) | (m[0] & e),
                    }
                }
            }
        }
    }

    Ok(match program.terminal {
        Terminal::Count { .. } => Value::Count(count),
        Terminal::Any { .. } => Value::Bool(any),
        Terminal::All { .. } => Value::Bool(all),
        Terminal::MaskedSumI32 { .. } => Value::SumI64(sum),
        Terminal::MaskedMinI32 { .. } => Value::OptI32(min),
        Terminal::MaskedMaxI32 { .. } => Value::OptI32(max),
        Terminal::BlendI32 { .. } => Value::Blended,
        Terminal::ScatterOrU32 { .. } => Value::Scattered,
        Terminal::ScatterCountU32 { .. } => Value::Count(match &out {
            Out::Mask(o) => popcount_batch_u64(o) as usize,
            _ => 0,
        }),
        Terminal::CountKeyRunsU32 { .. } => Value::Count(runs + run_carry.finish()),
        Terminal::GroupSumI32 { .. } | Terminal::GroupSumViaI32 { .. } => Value::GroupSummed,
        Terminal::GroupReduce { .. } => Value::GroupReduced,
        Terminal::Keep { mask } => Value::Mask(mask),
    })
}

#[cfg(test)]
mod extent_tile_tests {
    use super::*;

    /// The plan the executor iterates is proportional to the extent. A tiny
    /// extent in a million rows is one one-word tile; the whole population is
    /// exactly the `TILE_WORDS` chunking whole-population execution always used.
    #[test]
    fn the_tiles_visited_scale_with_the_extent_not_the_population() {
        let n = 1 << 20;
        let tile_count = |lo: usize, hi: usize| extent_tiles(n, TILE_WORDS, lo..hi).count();
        let words_visited = |lo: usize, hi: usize| {
            extent_tiles(n, TILE_WORDS, lo..hi)
                .map(|(w, _)| w.len())
                .sum::<usize>()
        };
        assert_eq!(tile_count(500_001, 500_002), 1);
        assert_eq!(words_visited(500_001, 500_002), 1);
        // 64 rows across a word seam: two one-word edge tiles; aligned: one.
        assert_eq!(tile_count(500_001, 500_065), 2);
        assert_eq!(tile_count(500_032, 500_096), 1); // 500_032 = 64 * 7813
        assert_eq!(words_visited(512_000, 512_512), 8);
        assert_eq!(tile_count(512_000, 512_512), 1);
        assert_eq!(tile_count(0, 0), 0);
        let whole: Vec<_> = extent_tiles(n, TILE_WORDS, 0..n).collect();
        assert_eq!(whole.len(), n / 64 / TILE_WORDS);
        for (i, (w, edge)) in whole.iter().enumerate() {
            assert_eq!(*w, i * TILE_WORDS..(i + 1) * TILE_WORDS);
            assert!(edge.is_none());
        }
        // Every extent: tiles are contiguous, disjoint, cover exactly the touched
        // words, and only a word the extent cuts carries an edge.
        let n = 1317;
        for lo in [0usize, 1, 63, 64, 65, 500, 1300] {
            for hi in [lo, lo + 1, lo + 63, lo + 64, lo + 700, n] {
                let hi = hi.min(n);
                if hi < lo {
                    continue;
                }
                let tiles: Vec<_> = extent_tiles(n, TILE_WORDS, lo..hi).collect();
                let covered: Vec<usize> = tiles.iter().flat_map(|(w, _)| w.clone()).collect();
                let want: Vec<usize> = touched_words(lo as u32, hi as u32).collect();
                assert_eq!(covered, want, "extent {lo}..{hi}");
                for (w, edge) in &tiles {
                    if edge.is_some() {
                        assert_eq!(w.len(), 1, "an edge tile is one word");
                        let cut_lo = w.start == lo / 64 && lo % 64 != 0;
                        let cut_hi = w.start == (hi - 1) / 64 && hi % 64 != 0 && hi < n;
                        assert!(cut_lo || cut_hi, "edge on a word the extent does not cut");
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod borrowed_scratch_tests {
    use super::*;
    use crate::ir::{LaneRef, MaskOp, Operand, Planes, Pred, Program, Terminal};

    /// A three-op program that touches three slots, every aliasing shape
    /// exercised by the differential suite already — this only needs the
    /// arena to be non-trivial.
    fn fixture(n: usize) -> (Vec<u64>, Vec<i32>, Program) {
        let mut seed = 0x5EEDu64;
        let mut lcg = || {
            seed = seed
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            seed >> 11
        };
        let mut mask: Vec<u64> = (0..words_for(n)).map(|_| lcg() & lcg()).collect();
        if !n.is_multiple_of(64) && !mask.is_empty() {
            let last = mask.len() - 1;
            mask[last] &= (1u64 << (n % 64)) - 1;
        }
        let lane: Vec<i32> = (0..n).map(|_| (lcg() % 2000) as i32 - 1000).collect();
        let p = Program::new(
            vec![
                MaskOp::Pred {
                    pred: Pred::GtI32 { lane: 0, t: 100 },
                    under: None,
                    dst: 0,
                },
                MaskOp::Pred {
                    pred: Pred::LtI32 { lane: 0, t: 800 },
                    under: Some(Operand::Scratch(0)),
                    dst: 1,
                },
                MaskOp::Ternlog {
                    imm: 0xE8,
                    a: Operand::Plane(0),
                    b: Operand::Scratch(0),
                    c: Operand::Scratch(1),
                    dst: 2,
                },
            ],
            Terminal::Count {
                mask: Operand::Scratch(2),
            },
        );
        (mask, lane, p)
    }

    fn run_owned(n: usize) -> (Value, Vec<Vec<u64>>) {
        let (mask, lane, p) = fixture(n);
        let masks: [&[u64]; 1] = [&mask];
        let lanes = [LaneRef::I32(&lane)];
        let planes = Planes {
            n_rows: n,
            masks: &masks,
            lanes: &lanes,
        };
        let mut s = Scratch::for_program(&p, n).expect("addressable");
        let v = execute(&p, &planes, &mut s, None).expect("runs");
        let slots = (0..s.slots())
            .map(|i| s.slot(i as u16).unwrap().to_vec())
            .collect();
        (v, slots)
    }

    /// FAILS IF: the borrowed arena computes anything different from the owned
    /// one, OR `Scratch::over` reads or writes a single word past the layout it
    /// was asked for — the poison is `u64::MAX`, the value most likely to
    /// corrupt a mask if it leaked into one.
    #[test]
    fn a_borrowed_arena_matches_an_owned_one_and_never_touches_the_excess() {
        for n in [0usize, 1, 63, 64, 65, 999, 4097] {
            let (mask, lane, p) = fixture(n);
            let masks: [&[u64]; 1] = [&mask];
            let lanes = [LaneRef::I32(&lane)];
            let planes = Planes {
                n_rows: n,
                masks: &masks,
                lanes: &lanes,
            };

            let need = scratch_words_for(tile_words_for(n), p.scratch_slots as usize).unwrap();
            // twice the room, every excess word poisoned
            let mut buf = vec![u64::MAX; need * 2 + 16];
            let excess_start = need;
            let mut s = Scratch::over(&mut buf, tile_words_for(n), p.scratch_slots as usize)
                .expect("buffer is long enough");
            let got = execute(&p, &planes, &mut s, None).expect("runs");
            let got_slots: Vec<Vec<u64>> = (0..s.slots())
                .map(|i| s.slot(i as u16).unwrap().to_vec())
                .collect();
            drop(s);

            let (want, want_slots) = run_owned(n);
            assert_eq!(got, want, "n={n}: borrowed result differs from owned");
            assert_eq!(got_slots, want_slots, "n={n}: borrowed slots differ");
            assert!(
                buf[excess_start..].iter().all(|&w| w == u64::MAX),
                "n={n}: Scratch::over touched {} words past its layout",
                buf[excess_start..]
                    .iter()
                    .filter(|&&w| w != u64::MAX)
                    .count()
            );
        }
    }

    /// FAILS IF: a slot the program never writes reads back as whatever the
    /// caller's buffer happened to hold, instead of as zero.
    ///
    /// This is the one thing `Scratch::over`'s zero-fill actually buys, and
    /// nothing else in the suite could see it: every other fixture declares
    /// exactly the slots its ops write, so the arena is fully overwritten
    /// before anything reads it and the fill is inert. A program may
    /// OVER-declare — `validate` rejects only under-declaring — and then
    /// `slot()` is a public read of a slot no op touched. Owned and borrowed
    /// must be interchangeable there too, or a consumer that swaps one for
    /// the other gets different answers out of the same program.
    ///
    /// Found by a disable run: removing the fill failed no test at all.
    #[test]
    fn a_slot_the_program_never_writes_reads_as_zero_from_either_arena() {
        let n = 200usize;
        let (mask, lane, mut p) = fixture(n);
        // Over-declare: the ops touch slots 0..=2, the program claims five.
        assert_eq!(p.scratch_slots, 3, "fixture writes exactly three slots");
        p.scratch_slots = 5;

        let masks: [&[u64]; 1] = [&mask];
        let lanes = [LaneRef::I32(&lane)];
        let planes = Planes {
            n_rows: n,
            masks: &masks,
            lanes: &lanes,
        };

        let mut owned = Scratch::for_program(&p, n).expect("addressable");
        execute(&p, &planes, &mut owned, None).expect("runs");

        let need = scratch_words_for(words_for(n), 5).unwrap();
        let mut buf = vec![u64::MAX; need];
        let mut borrowed = Scratch::over(&mut buf, words_for(n), 5).expect("fits");
        execute(&p, &planes, &mut borrowed, None).expect("runs");

        for i in 3..5u16 {
            let o = owned.slot(i).expect("declared");
            let b = borrowed.slot(i).expect("declared");
            assert!(
                o.iter().all(|&w| w == 0),
                "slot {i} of an OWNED arena must be zero"
            );
            assert_eq!(
                o, b,
                "slot {i}: owned and borrowed disagree on an unwritten slot"
            );
        }
    }

    /// FAILS IF: a short buffer is carved anyway — which would hand `execute` an
    /// arena whose last slot overlaps the read-before-write bitmap.
    #[test]
    fn a_buffer_one_word_short_is_refused_and_says_both_numbers() {
        let need = scratch_words_for(4, 3).unwrap();
        assert_eq!(need, 3 * 4 + 1, "layout is slots*words then the bitmap");
        let mut buf = vec![0u64; need - 1];
        assert_eq!(
            Scratch::over(&mut buf, 4, 3).err(),
            Some(ExecError::ScratchBufferTooSmall {
                need_words: need,
                have_words: need - 1,
            })
        );
        // can-it-stay-silent: exactly enough is accepted
        let mut exact = vec![0u64; need];
        assert!(Scratch::over(&mut exact, 4, 3).is_ok());
    }

    /// FAILS IF: one buffer cannot serve a SMALLER shape after a larger one —
    /// the property that makes allocation a function of the maximum rather than
    /// of the population's history, which is the entire reason this constructor
    /// exists. A cache keyed by exact size passes every same-size test and
    /// fails this one.
    #[test]
    fn one_buffer_grown_once_serves_every_smaller_shape() {
        let big = 4097usize;
        let (_, _, p_big) = fixture(big);
        let cap = scratch_words_for(words_for(big), p_big.scratch_slots as usize).unwrap();
        let mut buf = vec![0u64; cap];

        // descending, so every call after the first carves a strict prefix
        for n in [4097usize, 999, 65, 64, 1, 0] {
            let (mask, lane, p) = fixture(n);
            let masks: [&[u64]; 1] = [&mask];
            let lanes = [LaneRef::I32(&lane)];
            let planes = Planes {
                n_rows: n,
                masks: &masks,
                lanes: &lanes,
            };
            let mut s = Scratch::over_for_program(&mut buf, &p, n).expect("prefix fits");
            let got = execute(&p, &planes, &mut s, None).expect("runs");
            drop(s);
            let (want, _) = run_owned(n);
            assert_eq!(got, want, "n={n} against a buffer sized for {big}");
        }
    }

    /// FAILS IF: `scratch_words_for` wraps instead of reporting that no buffer
    /// can be long enough. A wrapped product would make `over` accept a tiny
    /// buffer for an enormous layout.
    #[test]
    fn an_overflowing_layout_has_no_size_rather_than_a_wrapped_one() {
        assert_eq!(scratch_words_for(usize::MAX, 2), None);
        assert_eq!(scratch_words_for(2, usize::MAX), None);
        // and the constructor surfaces it as a buffer that cannot suffice
        let mut buf = [0u64; 8];
        assert!(matches!(
            Scratch::over(&mut buf, usize::MAX, 2),
            Err(ExecError::ScratchBufferTooSmall { .. })
        ));
    }
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
        // `All` is never fused, so this program genuinely needs its slots.
        let ok = Program::new(
            vec![MaskOp::Not {
                a: Operand::Plane(0),
                dst: 2,
            }],
            Terminal::All {
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
            // Read each signature WHOLE, up to its body: a `pub fn` whose
            // parameters wrap onto several lines carries its `->` on a later
            // line, and a guard that reads the first line alone cannot see
            // it (`reference_scratch_with_foreign` escaped exactly so).
            for decl in production.split("pub fn ").skip(1) {
                let sig = decl.split('{').next().unwrap_or("");
                // ANY owning collection RETURN, not just `Vec<usize>`: the
                // oracle's arena copy is a `Vec<Vec<u64>>`, and a guard that
                // greps the one shape it knows cannot see the next one. Split
                // on the arrow so a `Vec` PARAMETER is not read as a return.
                let returns = sig.split("->").nth(1).unwrap_or("");
                if returns.contains("Vec<") {
                    let name = sig.split('(').next().unwrap_or(sig).trim();
                    found.push(name.to_string());
                }
            }
        }
        // `reference_scratch` / `reference_scratch_with_foreign` are the ONE
        // written-down exemption in its two entry shapes: the oracle must
        // hold its own unpacked reading of the arena, because an oracle
        // sharing the executor's bit packing could not falsify a packing bug
        // (law L4). Named here so the guard SEES them.
        assert_eq!(
            found,
            [
                "materialize_rows",
                "reference_scratch",
                "reference_scratch_with_foreign"
            ]
        );
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
        // `All` is never fused (a `Count`/`Any` over this op would fold and
        // need no scratch at all), so the scratch checks below are reached.
        let p = Program::new(
            vec![MaskOp::Not {
                a: Operand::Plane(0),
                dst: 3,
            }],
            Terminal::All {
                mask: Operand::Scratch(3),
            },
        );
        let mut small = Scratch::new(2, 2);
        assert_eq!(
            execute(&p, &planes, &mut small, None),
            Err(ExecError::ScratchTooSmall { need: 4, have: 2 })
        );
        let mut wrong = Scratch::new(0, 4);
        assert_eq!(
            execute(&p, &planes, &mut wrong, None),
            Err(ExecError::ScratchWords {
                expected: 1,
                found: 0
            })
        );
        // A NARROW scratch is not wrong — it is a tile width. One word over
        // 70 rows runs as two tiles and answers exactly what two words do.
        let mut narrow = Scratch::new(1, 4);
        let mut wide = Scratch::new(2, 4);
        assert_eq!(
            execute(&p, &planes, &mut narrow, None),
            execute(&p, &planes, &mut wide, None)
        );
        assert_eq!(
            execute(&p, &planes, &mut narrow, None),
            Ok(Value::Bool(true))
        );
    }
}
