//! The scalar oracle — the same [`Program`] evaluated ONE ROW AT A TIME in
//! plain Rust, with no SIMD facade anywhere in this file (law L4: a test
//! greps the source). Every executor is diffed against it on every backend.
//!
//! It also owns [`validate`], the ONE spelling of the rules a program must
//! satisfy before it runs; the executor calls the same function, so both
//! sides refuse the same bad program with the same [`ExecError`].
//!
//! # This file materialises, deliberately
//!
//! The oracle holds one `bool` per row per slot where the executor holds one
//! bit, and [`reference_scratch`] packs a whole second copy of the arena. That
//! is the point: an oracle sharing the executor's bit packing could not
//! falsify a bit-packing bug. It is the crate's one exemption from the
//! single-materialiser law, named in `exec.rs`'s `exactly_one_materialiser`
//! so the guard sees it rather than missing it by accident — never a read
//! path a consumer should reach for.

use crate::ir::{
    LaneRef, MaskOp, Operand, Planes, Pred, Program, Terminal, MASKED_SUM_I32_MAX_ROWS,
    MAX_SCRATCH_SLOTS,
};
use crate::value::{ExecError, LaneKind, Value};
use crate::words_for;

fn kind_of(lane: &LaneRef<'_>) -> LaneKind {
    match lane {
        LaneRef::I32(_) => LaneKind::I32,
        LaneRef::U32(_) => LaneKind::U32,
        LaneRef::U64(_) => LaneKind::U64,
    }
}

/// The lane a predicate reads, and the width that lane must have.
///
/// One place where `Pred`'s variants map to lane widths, so the executor and
/// the oracle cannot disagree about which column a predicate touches. Adding a
/// `Pred` variant without extending this match is a compile error, which is
/// the point of listing the variants explicitly rather than matching a field.
fn pred_lane_and_kind(pred: Pred) -> (u16, LaneKind) {
    match pred {
        Pred::GtI32 { lane, .. }
        | Pred::LtI32 { lane, .. }
        | Pred::GeI32 { lane, .. }
        | Pred::LeI32 { lane, .. }
        | Pred::EqI32 { lane, .. }
        | Pred::NeI32 { lane, .. } => (lane, LaneKind::I32),
        Pred::EqU32 { lane, .. } | Pred::NeU32 { lane, .. } | Pred::MatchU32 { lane, .. } => {
            (lane, LaneKind::U32)
        }
        Pred::MatchU64 { lane, .. } => (lane, LaneKind::U64),
    }
}

/// One operand's index, against the two address spaces it could name.
///
/// A plane index is bounded by what the CALLER supplied (`planes.masks`); a
/// scratch index by what the PROGRAM declared (`p.scratch_slots`). The two
/// errors stay distinct for that reason — `PlaneOutOfRange` is a caller
/// mismatch, `ScratchSlotUndeclared` a hand-built program whose count does not
/// cover its own operands, and `Program::new` computes a covering count so the
/// latter can only come from a struct literal.
///
/// Note this is per-OPERAND and therefore structurally blind to a declared
/// count that no operand reaches; that bound lives in `validate` itself.
fn check_operand(p: &Program, planes: &Planes<'_>, o: Operand) -> Result<(), ExecError> {
    match o {
        Operand::Plane(i) if usize::from(i) >= planes.masks.len() => {
            Err(ExecError::PlaneOutOfRange(i))
        }
        Operand::Scratch(i) if u32::from(i) >= p.scratch_slots => {
            Err(ExecError::ScratchSlotUndeclared {
                slot: i,
                declared: p.scratch_slots,
            })
        }
        _ => Ok(()),
    }
}

/// A lane index and its WIDTH, in that order.
///
/// Both halves are needed and neither implies the other: a lane can exist at
/// the wrong width, and the wrong width is not a bounds error. Reporting them
/// as one status would collapse two different caller mistakes — a mis-sized
/// `Planes` and a predicate naming the wrong column — into one message.
///
/// This is what licenses `eval_pred`'s `_ => 0` fallback: by the time any row
/// is read, every lane a predicate names has been proven present and correctly
/// typed.
fn check_lane(planes: &Planes<'_>, lane: u16, expected: LaneKind) -> Result<(), ExecError> {
    match planes.lanes.get(usize::from(lane)) {
        None => Err(ExecError::LaneOutOfRange(lane)),
        Some(l) if kind_of(l) != expected => Err(ExecError::LaneKind {
            lane,
            expected,
            found: kind_of(l),
        }),
        Some(_) => Ok(()),
    }
}

/// A scratch slot is readable only after some EARLIER op has written it.
///
/// Without this rule the executor would read whatever the caller's REUSED
/// [`crate::exec::Scratch`] still holds while the oracle reads a fresh zeroed
/// arena — a divergence no fixture built from `Scratch::for_program` can
/// express, because every such fixture starts zeroed. Pre-filled scratch is a
/// named PR5 gap, not a supported input.
///
/// The check is a bitmap marked FORWARD as the op walk proceeds, not a
/// backward scan of earlier ops. Both are allocation-free — law L1 forbids
/// allocating anywhere `execute` reaches, and `execute` calls `validate` on
/// every run — but the scan was **quadratic in op count**, and the doc that
/// used to sit here claimed that count "is single digits for every program
/// this IR can express". That was false: `Program::ops` is a public `Vec`, so
/// a hand-built chain of `Not`s each reading the previous slot is admissible.
/// Measured on exactly that chain, at `n_rows = 0` so validation is the only
/// work: 0.62 ms at 1024 ops, 11 ms at 4096, 179 ms at 16_384, **3.06 s at
/// 65_536** — a clean quadratic, and a caller-controlled stall before a single
/// row is touched.
///
/// The bitmap is CALLER-OWNED, borrowed — the same shape [`crate::Scratch`]
/// already is, and for the same reason. A fixed `[u64; 1024]` on the stack
/// also allocates nothing, and was tried first; it costs an 8 KiB zero on
/// every `execute`, and that is not free on this hot path. Measured over three
/// runs of `count_probe`, with `handwritten` as the control because it builds
/// no `Program` and so never validates:
///
/// | arm | borrowed bitmap | 8 KiB stack array |
/// |---|---|---|
/// | handwritten (control) | 1027-1055 | 1027-1055 |
/// | interpreted | 1039-1048 | 1197-1363 |
/// | fused | 924-926 | 1129-1281 |
///
/// The control not moving while both validating arms did is what makes that
/// attributable to the zero rather than to machine noise.
///
/// Sized by the caller's slot count, so a three-slot program zeroes ONE word
/// per call. `slot_words` is the only part read or written.
struct WrittenSlots<'a> {
    bits: &'a mut [u64],
}

impl WrittenSlots<'_> {
    /// Clear exactly the words `declared` slots occupy, leaving the rest of
    /// the caller's buffer alone — that prefix is the whole cost per call.
    fn reset(&mut self, declared: u32) {
        let words = (declared as usize).div_ceil(64).min(self.bits.len());
        self.bits[..words].fill(0);
    }

    /// Record that `slot` has been written by an op already validated.
    fn mark(&mut self, slot: u16) {
        if let Some(w) = self.bits.get_mut(usize::from(slot) / 64) {
            *w |= 1u64 << (usize::from(slot) % 64);
        }
    }

    /// `Ok` if `o` is a plane, or a scratch slot some earlier op wrote.
    ///
    /// A slot outside the buffer reads as NOT written rather than panicking.
    /// It is unreachable — `check_operand` refuses `slot >= scratch_slots`
    /// before any operand reaches here, on every arm and on every terminal —
    /// but this crate forbids unsafe and prefers not to rely on a panic for a
    /// bound another function owns.
    fn readable(&self, o: Operand) -> Result<(), ExecError> {
        let Operand::Scratch(slot) = o else {
            return Ok(());
        };
        let set = self
            .bits
            .get(usize::from(slot) / 64)
            .is_some_and(|w| w >> (usize::from(slot) % 64) & 1 == 1);
        if set {
            Ok(())
        } else {
            Err(ExecError::ScratchReadBeforeWrite { slot })
        }
    }
}

/// The destination every op writes — one place, so a new `MaskOp` variant is a
/// compile error here rather than a slot that silently never gets marked.
fn dst_of(op: &MaskOp) -> u16 {
    match *op {
        MaskOp::Pred { dst, .. }
        | MaskOp::And { dst, .. }
        | MaskOp::Or { dst, .. }
        | MaskOp::Xor { dst, .. }
        | MaskOp::AndNot { dst, .. }
        | MaskOp::Not { dst, .. }
        | MaskOp::Ternlog { dst, .. } => dst,
    }
}

/// Apply the program validation shared by the executor and the oracle.
///
/// An unaddressable scratch count is rejected first, followed by plane and
/// lane shapes, a dirty plane tail, each op in program order, and the
/// terminal's requirements. `out_len` is the caller's `out` slice length, if
/// any. `written_bits` is reusable workspace for the read-before-write check;
/// the prefix needed for the program's declared slots is cleared and updated.
pub(crate) fn validate(
    p: &Program,
    planes: &Planes<'_>,
    out_len: Option<usize>,
    written_bits: &mut [u64],
) -> Result<(), ExecError> {
    // FIRST, before anything walks a plane or an op: a declared slot count
    // above the addressable ceiling is refused. The count is a PUBLIC field,
    // so a hand-built program can declare four billion slots while naming no
    // `Operand::Scratch` at all — which every per-operand check below is
    // structurally blind to, since there is no operand to check. Both
    // arena-sizing paths (`Scratch::for_program` and the oracle's own `run`)
    // allocate proportionally to this field, so the refusal has to happen
    // here, ahead of them, rather than in a `debug_assert` that release
    // builds drop.
    if p.scratch_slots > MAX_SCRATCH_SLOTS {
        return Err(ExecError::ScratchSlotsUnaddressable {
            declared: p.scratch_slots,
        });
    }
    let n = planes.n_rows;
    let words = words_for(n);
    // `Operand::Plane` is a `u16`: planes past 65,536 cannot be named by any
    // program, and refusing them here is what makes the index in `PlaneTail`
    // below exact rather than saturated.
    if planes.masks.len() > usize::from(u16::MAX) + 1 {
        return Err(ExecError::LenMismatch {
            what: "masks",
            expected: usize::from(u16::MAX) + 1,
            found: planes.masks.len(),
        });
    }
    for m in planes.masks {
        if m.len() != words {
            return Err(ExecError::LenMismatch {
                what: "plane",
                expected: words,
                found: m.len(),
            });
        }
    }
    for l in planes.lanes {
        if l.len() != n {
            return Err(ExecError::LenMismatch {
                what: "lane",
                expected: n,
                found: l.len(),
            });
        }
    }
    if !n.is_multiple_of(64) {
        for (i, m) in planes.masks.iter().enumerate() {
            if m[n / 64] >> (n % 64) != 0 {
                // Exact, not saturated: the count check above already refused
                // anything past the plane index type's range.
                return Err(ExecError::PlaneTail(u16::try_from(i).unwrap_or(u16::MAX)));
            }
        }
    }
    // Marked forward, op by op. `mark` happens AFTER the op's own operands are
    // checked, which is what keeps `MaskOp::Not { a: Scratch(0), dst: 0 }` — an
    // op reading its own destination before anything wrote it — a refusal
    // rather than a self-satisfying read.
    let mut written_slots = WrittenSlots { bits: written_bits };
    written_slots.reset(p.scratch_slots);
    for op in &p.ops {
        let written = |o: Operand| written_slots.readable(o);
        match *op {
            MaskOp::Pred { pred, under, dst } => {
                if let Some(u) = under {
                    check_operand(p, planes, u)?;
                    if u == Operand::Scratch(dst) {
                        return Err(ExecError::GateAliasesDst { dst });
                    }
                    written(u)?;
                }
                let (lane, kind) = pred_lane_and_kind(pred);
                check_lane(planes, lane, kind)?;
                check_operand(p, planes, Operand::Scratch(dst))?;
            }
            MaskOp::And { a, b, dst }
            | MaskOp::Or { a, b, dst }
            | MaskOp::Xor { a, b, dst }
            | MaskOp::AndNot { a, b, dst } => {
                check_operand(p, planes, a)?;
                check_operand(p, planes, b)?;
                check_operand(p, planes, Operand::Scratch(dst))?;
                written(a)?;
                written(b)?;
            }
            MaskOp::Not { a, dst } => {
                check_operand(p, planes, a)?;
                check_operand(p, planes, Operand::Scratch(dst))?;
                written(a)?;
            }
            MaskOp::Ternlog { a, b, c, dst, .. } => {
                check_operand(p, planes, a)?;
                check_operand(p, planes, b)?;
                check_operand(p, planes, c)?;
                check_operand(p, planes, Operand::Scratch(dst))?;
                written(a)?;
                written(b)?;
                written(c)?;
            }
        }
        written_slots.mark(dst_of(op));
    }
    match p.terminal {
        Terminal::Count { mask }
        | Terminal::Any { mask }
        | Terminal::All { mask }
        | Terminal::Keep { mask } => {
            check_operand(p, planes, mask)?;
            written_slots.readable(mask)
        }
        Terminal::MaskedSumI32 { mask, lane } => {
            check_operand(p, planes, mask)?;
            written_slots.readable(mask)?;
            check_lane(planes, lane, LaneKind::I32)?;
            if n > MASKED_SUM_I32_MAX_ROWS {
                return Err(ExecError::SumRowBound { n_rows: n });
            }
            Ok(())
        }
        Terminal::MaskedMinI32 { mask, lane } | Terminal::MaskedMaxI32 { mask, lane } => {
            check_operand(p, planes, mask)?;
            written_slots.readable(mask)?;
            check_lane(planes, lane, LaneKind::I32)
        }
        Terminal::BlendI32 { mask, then, els } => {
            check_operand(p, planes, mask)?;
            written_slots.readable(mask)?;
            check_lane(planes, then, LaneKind::I32)?;
            check_lane(planes, els, LaneKind::I32)?;
            match out_len {
                None => Err(ExecError::BlendNeedsOut),
                Some(len) if len != n => Err(ExecError::LenMismatch {
                    what: "out",
                    expected: n,
                    found: len,
                }),
                Some(_) => Ok(()),
            }
        }
    }
}

fn plane_bit(planes: &Planes<'_>, i: u16, row: usize) -> bool {
    (planes.masks[usize::from(i)][row / 64] >> (row % 64)) & 1 == 1
}

fn i32_at(planes: &Planes<'_>, lane: u16, row: usize) -> i32 {
    match planes.lanes[usize::from(lane)] {
        LaneRef::I32(v) => v[row],
        _ => 0,
    }
}

/// One predicate, one row — the scalar SPEC of what each `Pred` means.
///
/// This is the definition the executor's `ndarray::simd` delegation is diffed
/// against, so it is written to be obviously right rather than fast: plain
/// Rust comparison operators, one row at a time, no facade token anywhere
/// (law L4). `i32` comparisons are signed and `u32`/`u64` exact-bitwise,
/// matching the DuckDB semantics `lib.rs` enumerates.
///
/// The two TCAM arms are the only non-obvious ones: `(value ^ pattern) & care
/// == 0` means *every bit `care` selects must match `pattern`*, and bits
/// outside `care` are ignored — so `care == 0` matches every row and
/// `care == !0` is plain equality.
///
/// A lane of the wrong width reads as `0` here rather than panicking, which is
/// safe only because `validate` has already refused a mismatched lane kind
/// (`ExecError::LaneKind`) before this is ever called. Both the executor and
/// this oracle call that same `validate`, so neither can reach the fallback.
fn eval_pred(planes: &Planes<'_>, pred: Pred, row: usize) -> bool {
    let u32_at = |lane: u16| match planes.lanes[usize::from(lane)] {
        LaneRef::U32(v) => v[row],
        _ => 0,
    };
    let u64_at = |lane: u16| match planes.lanes[usize::from(lane)] {
        LaneRef::U64(v) => v[row],
        _ => 0,
    };
    match pred {
        Pred::GtI32 { lane, t } => i32_at(planes, lane, row) > t,
        Pred::LtI32 { lane, t } => i32_at(planes, lane, row) < t,
        Pred::GeI32 { lane, t } => i32_at(planes, lane, row) >= t,
        Pred::LeI32 { lane, t } => i32_at(planes, lane, row) <= t,
        Pred::EqI32 { lane, v } => i32_at(planes, lane, row) == v,
        Pred::NeI32 { lane, v } => i32_at(planes, lane, row) != v,
        Pred::EqU32 { lane, v } => u32_at(lane) == v,
        Pred::NeU32 { lane, v } => u32_at(lane) != v,
        Pred::MatchU32 {
            lane,
            pattern,
            care,
        } => (u32_at(lane) ^ pattern) & care == 0,
        Pred::MatchU64 {
            lane,
            pattern,
            care,
        } => (u64_at(lane) ^ pattern) & care == 0,
    }
}

/// The oracle's state: one `Vec<bool>` per scratch slot, one entry per row.
struct Rows {
    slots: Vec<Vec<bool>>,
}

impl Rows {
    fn bit(&self, planes: &Planes<'_>, o: Operand, row: usize) -> bool {
        match o {
            Operand::Plane(i) => plane_bit(planes, i, row),
            Operand::Scratch(i) => self.slots[usize::from(i)][row],
        }
    }

    fn set(&mut self, dst: u16, row: usize, v: bool) {
        self.slots[usize::from(dst)][row] = v;
    }
}

/// Evaluate every op of `p`, row by row, into a fresh `Rows` arena.
///
/// The op loop is the outer one and rows the inner, which mirrors the
/// executor's word-at-a-time pass and is what makes the two comparable: an op
/// sees every earlier op's completed result and none of its own. `Ternlog`
/// indexes its immediate as `(a << 2) | (b << 1) | c`, the same bit order the
/// executor's dispatch uses — one convention, stated in `lib.rs` and spelled
/// identically on both sides.
///
/// The arena starts all-false and is never seeded from a caller, which is the
/// oracle's fresh-arena assumption. That assumption is true by construction
/// rather than by fixture: `validate` refuses any program that READS a scratch
/// slot no earlier op wrote (`ExecError::ScratchReadBeforeWrite`), and it
/// refuses a `scratch_slots` count past `MAX_SCRATCH_SLOTS` before this
/// allocates anything proportional to it.
fn run(p: &Program, planes: &Planes<'_>) -> Rows {
    let n = planes.n_rows;
    let mut rows = Rows {
        slots: (0..p.scratch_slots).map(|_| vec![false; n]).collect(),
    };
    for op in &p.ops {
        for row in 0..n {
            let v = match *op {
                MaskOp::Pred {
                    pred,
                    under,
                    dst: _,
                } => under.is_none_or(|u| rows.bit(planes, u, row)) && eval_pred(planes, pred, row),
                MaskOp::And { a, b, .. } => rows.bit(planes, a, row) & rows.bit(planes, b, row),
                MaskOp::Or { a, b, .. } => rows.bit(planes, a, row) | rows.bit(planes, b, row),
                MaskOp::Xor { a, b, .. } => rows.bit(planes, a, row) ^ rows.bit(planes, b, row),
                MaskOp::AndNot { a, b, .. } => rows.bit(planes, a, row) & !rows.bit(planes, b, row),
                MaskOp::Not { a, .. } => !rows.bit(planes, a, row),
                MaskOp::Ternlog { imm, a, b, c, .. } => {
                    let idx = (u8::from(rows.bit(planes, a, row)) << 2)
                        | (u8::from(rows.bit(planes, b, row)) << 1)
                        | u8::from(rows.bit(planes, c, row));
                    (imm >> idx) & 1 == 1
                }
            };
            let dst = match *op {
                MaskOp::Pred { dst, .. }
                | MaskOp::And { dst, .. }
                | MaskOp::Or { dst, .. }
                | MaskOp::Xor { dst, .. }
                | MaskOp::AndNot { dst, .. }
                | MaskOp::Not { dst, .. }
                | MaskOp::Ternlog { dst, .. } => dst,
            };
            rows.set(dst, row, v);
        }
    }
    rows
}

/// The read-before-write bitmap [`validate`] needs, allocated only once the
/// declared slot count is known to be addressable.
///
/// The ORDER is the whole content of this function. `Program::scratch_slots`
/// is a public `u32`, so a hand-built program can declare `u32::MAX` slots;
/// sizing the bitmap from it first asks the allocator for 512 MiB and only
/// then calls the check that was going to reject the program anyway. On a
/// machine that cannot serve it, a refusal this crate owes as an
/// [`ExecError`] arrives as an abort instead. Both oracle entry points
/// therefore size through here, never from the field directly.
///
/// [`validate`] repeats the same check — deliberately, since the executor
/// supplies its own caller-owned bitmap and never passes through here.
fn written_bitmap(p: &Program) -> Result<Vec<u64>, ExecError> {
    if p.scratch_slots > MAX_SCRATCH_SLOTS {
        return Err(ExecError::ScratchSlotsUnaddressable {
            declared: p.scratch_slots,
        });
    }
    Ok(vec![0u64; (p.scratch_slots as usize).div_ceil(64)])
}

/// Evaluate `p` over `planes` row by row; `out` is the destination a
/// [`Terminal::BlendI32`] writes. Same validation, same [`Value`], same
/// [`ExecError`] as the executor.
pub fn reference_execute(
    p: &Program,
    planes: &Planes<'_>,
    out: Option<&mut [i32]>,
) -> Result<Value, ExecError> {
    // The oracle allocates by design (its `Rows` arena is this crate's one
    // named L5 exemption), so a local bitmap costs it nothing it was not
    // already paying — but it is sized through the guarded helper, not from
    // the declared count.
    let mut bits = written_bitmap(p)?;
    validate(p, planes, out.as_deref().map(<[i32]>::len), &mut bits)?;
    let n = planes.n_rows;
    let rows = run(p, planes);
    let rows = &rows;
    let survivors = |mask: Operand| (0..n).filter(move |&r| rows.bit(planes, mask, r));
    Ok(match p.terminal {
        Terminal::Count { mask } => Value::Count(survivors(mask).count()),
        Terminal::Any { mask } => Value::Bool(survivors(mask).next().is_some()),
        Terminal::All { mask } => Value::Bool(survivors(mask).count() == n),
        Terminal::MaskedSumI32 { mask, lane } => Value::SumI64(
            survivors(mask)
                .map(|r| i64::from(i32_at(planes, lane, r)))
                .sum(),
        ),
        Terminal::MaskedMinI32 { mask, lane } => {
            Value::OptI32(survivors(mask).map(|r| i32_at(planes, lane, r)).min())
        }
        Terminal::MaskedMaxI32 { mask, lane } => {
            Value::OptI32(survivors(mask).map(|r| i32_at(planes, lane, r)).max())
        }
        Terminal::BlendI32 { mask, then, els } => {
            if let Some(o) = out {
                for (r, slot) in o.iter_mut().enumerate() {
                    let pick = if rows.bit(planes, mask, r) { then } else { els };
                    *slot = i32_at(planes, pick, r);
                }
            }
            Value::Blended
        }
        Terminal::Keep { mask } => Value::Mask(mask),
    })
}

/// Every scratch slot's FINAL contents after `p` runs, packed LSB-first with
/// tail bits zero — the shape an executor's `Scratch` holds, built here by
/// hand so a `Keep` result and every intermediate can be diffed word by word.
pub fn reference_scratch(p: &Program, planes: &Planes<'_>) -> Result<Vec<Vec<u64>>, ExecError> {
    // `Some(n_rows)`, not `None`: slot contents are a function of the OPS
    // alone, so a `BlendI32` terminal's missing `out` must not make this
    // report `BlendNeedsOut` — that silently skipped the differential's
    // whole scratch comparison for every blend program.
    let mut bits = written_bitmap(p)?;
    validate(p, planes, Some(planes.n_rows), &mut bits)?;
    let n = planes.n_rows;
    let rows = run(p, planes);
    Ok(rows
        .slots
        .iter()
        .map(|slot| {
            let mut words = vec![0u64; words_for(n)];
            for (r, &b) in slot.iter().enumerate() {
                if b {
                    words[r / 64] |= 1u64 << (r % 64);
                }
            }
            words
        })
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    const P0: Operand = Operand::Plane(0);
    const S0: Operand = Operand::Scratch(0);

    fn lcg(seed: &mut u64) -> u64 {
        *seed = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        *seed >> 11
    }

    struct Fx {
        n: usize,
        mask: Vec<u64>,
        i32s: Vec<i32>,
        u32s: Vec<u32>,
        /// A SECOND, distinct `i32` lane. Without it a `BlendI32 { then: 0,
        /// els: 0 }` fixture is vacuous: `out` equals lane 0 for every mask,
        /// so swapping the branches changes nothing.
        i32s_b: Vec<i32>,
    }

    impl Fx {
        fn new(n: usize) -> Self {
            let mut s = 99u64 ^ n as u64;
            let mut mask: Vec<u64> = (0..words_for(n))
                .map(|_| lcg(&mut s) & lcg(&mut s))
                .collect();
            if !n.is_multiple_of(64) {
                mask[words_for(n) - 1] &= (1u64 << (n % 64)) - 1;
            }
            let i32s = (0..n).map(|_| (lcg(&mut s) % 2000) as i32 - 1000).collect();
            let u32s = (0..n).map(|_| (lcg(&mut s) % 64) as u32).collect();
            let i32s_b = (0..n).map(|_| -1 - (lcg(&mut s) % 500) as i32).collect();
            Self {
                n,
                mask,
                i32s,
                u32s,
                i32s_b,
            }
        }

        /// Run `f` against a `Planes` borrowing this fixture.
        ///
        /// A closure rather than a getter because `Planes` holds `&[&[u64]]`:
        /// the slice OF references is itself a temporary, so a function
        /// returning `Planes` would return a borrow of a local.
        fn with<R>(&self, f: impl FnOnce(&Planes<'_>) -> R) -> R {
            let masks: [&[u64]; 1] = [&self.mask];
            let lanes = [
                LaneRef::I32(&self.i32s),
                LaneRef::U32(&self.u32s),
                LaneRef::I32(&self.i32s_b),
            ];
            f(&Planes {
                n_rows: self.n,
                masks: &masks,
                lanes: &lanes,
            })
        }
    }

    /// FAILS IF: the oracle names the SIMD facade OR a sibling module that
    /// would import it transitively — law L4. Needles are concatenated so
    /// this test's own text cannot match itself.
    ///
    /// Deliberately NOT split on `#[cfg(test)]`, unlike the ISA guard: the
    /// oracle's independence is the whole claim, so a test module that reached
    /// for the facade to check the oracle would undermine it just as much as
    /// production code would.
    /// Blank out comment spans, keeping line structure so a failure still
    /// reports a recognisable line. Handles `/* */` and both leading and
    /// trailing `//`.
    fn strip_comments(src: &str) -> String {
        let mut out = String::with_capacity(src.len());
        let mut in_block = false;
        for line in src.lines() {
            let b = line.as_bytes();
            let mut i = 0;
            while i < b.len() {
                if in_block {
                    if b[i] == b'*' && b.get(i + 1) == Some(&b'/') {
                        in_block = false;
                        i += 2;
                    } else {
                        i += 1;
                    }
                } else if b[i] == b'/' && b.get(i + 1) == Some(&b'*') {
                    in_block = true;
                    i += 2;
                } else if b[i] == b'/' && b.get(i + 1) == Some(&b'/') {
                    break;
                } else {
                    out.push(b[i] as char);
                    i += 1;
                }
            }
            out.push('\n');
        }
        out
    }

    /// FAILS IF: law L4 is breached — any line of CODE in this file names the
    /// SIMD facade, which would make the oracle share an implementation with
    /// the thing it exists to falsify. Also fails if `strip_comments` stops
    /// stripping, since the needle then matches this file's own prose.
    #[test]
    fn the_oracle_has_no_facade_token() {
        let needle = ["nd", "array"].concat();
        // CODE only, matching `the_crate_names_no_isa`: this file's own doc
        // comments explain the law and therefore name what it forbids, and a
        // doc line is the opposite of a breach. Before this filter the guard
        // was stricter than its sibling for no reason and a comment could
        // break it — which is exactly what happened when `eval_pred` gained a
        // doc explaining why it carries no facade token.
        //
        // A leading-`//` filter was the first attempt and was not enough: an
        // INLINE trailing comment, or a `/* */` span, would still break the
        // guard while being just as much prose. `strip_comments` removes both.
        //
        // Boundary, stated rather than discovered later: a `//` inside a
        // string literal truncates that line early, and nested block comments
        // are not modelled. Both make the guard slightly WEAKER (it could miss
        // a needle in such a line), never stricter — and neither shape is a
        // facade call, which is what the law is actually about.
        for line in strip_comments(include_str!("reference.rs")).lines() {
            assert!(
                !line.contains(&needle),
                "the oracle must not name the SIMD facade (law L4): {line}"
            );
        }
    }

    /// FAILS IF: the packing is MSB-first (word 1 would read `0xFC00…`), or
    /// `Not` is wrong. Note the oracle iterates `0..n` rows and so cannot set
    /// a tail bit at all — the tail law is structural here, and this test
    /// pins bit ORDER, which is what the executor is diffed against.
    #[test]
    fn packed_scratch_obeys_the_tail_law() {
        let zero = vec![0u64; 2];
        let masks: [&[u64]; 1] = [&zero];
        let planes = Planes {
            n_rows: 70,
            masks: &masks,
            lanes: &[],
        };
        let p = Program::new(
            vec![MaskOp::Not { a: P0, dst: 0 }],
            Terminal::Keep { mask: S0 },
        );
        let slots = reference_scratch(&p, &planes).unwrap_or_default();
        assert_eq!(slots, vec![vec![u64::MAX, 0b11_1111]]);
    }

    /// FAILS IF: a gated predicate is not `pred & gate` row for row — and the
    /// fixture is vacuous (both the gate and the predicate must be selective).
    #[test]
    fn a_gated_predicate_equals_the_ungated_one_under_the_gate() {
        let fx = Fx::new(130);
        fx.with(|planes| {
            let pred = Pred::GtI32 { lane: 0, t: 600 };
            let gated = Program::new(
                vec![MaskOp::Pred { pred, under: Some(P0), dst: 0 }],
                Terminal::Count { mask: S0 },
            );
            let manual = Program::new(
                vec![MaskOp::Pred { pred, under: None, dst: 0 }, MaskOp::And { a: S0, b: P0, dst: 1 }],
                Terminal::Count { mask: Operand::Scratch(1) },
            );
            let g = reference_execute(&gated, planes, None);
            assert_eq!(g, reference_execute(&manual, planes, None));
            let ungated = Program::new(vec![MaskOp::Pred { pred, under: None, dst: 0 }], Terminal::Count { mask: S0 });
            let u = reference_execute(&ungated, planes, None);
            assert!(matches!((g, u), (Ok(Value::Count(a)), Ok(Value::Count(b))) if a > 0 && a < b && b * 3 < 130));
        });
    }

    /// FAILS IF: any terminal disagrees with an independent spelling over the
    /// raw fixture data (`filter().count()`, `iter().sum()`, …).
    #[test]
    fn every_terminal_matches_an_independent_spelling() {
        let fx = Fx::new(130);
        fx.with(|planes| {
            let pred = MaskOp::Pred {
                pred: Pred::LtI32 { lane: 0, t: -600 },
                under: None,
                dst: 0,
            };
            let sel: Vec<usize> = (0..fx.n).filter(|&r| fx.i32s[r] < -600).collect();
            assert!(sel.len() * 3 < fx.n && !sel.is_empty());
            let run = |t: Terminal, out: Option<&mut [i32]>| {
                reference_execute(&Program::new(vec![pred], t), planes, out)
            };
            assert_eq!(
                run(Terminal::Count { mask: S0 }, None),
                Ok(Value::Count(sel.len()))
            );
            assert_eq!(run(Terminal::Any { mask: S0 }, None), Ok(Value::Bool(true)));
            assert_eq!(
                run(Terminal::All { mask: S0 }, None),
                Ok(Value::Bool(false))
            );
            let sum: i64 = sel.iter().map(|&r| i64::from(fx.i32s[r])).sum();
            assert_eq!(
                run(Terminal::MaskedSumI32 { mask: S0, lane: 0 }, None),
                Ok(Value::SumI64(sum))
            );
            let min = sel.iter().map(|&r| fx.i32s[r]).min();
            let max = sel.iter().map(|&r| fx.i32s[r]).max();
            assert_eq!(
                run(Terminal::MaskedMinI32 { mask: S0, lane: 0 }, None),
                Ok(Value::OptI32(min))
            );
            assert_eq!(
                run(Terminal::MaskedMaxI32 { mask: S0, lane: 0 }, None),
                Ok(Value::OptI32(max))
            );
            let mut out = vec![0i32; fx.n];
            assert_eq!(
                run(
                    Terminal::BlendI32 {
                        mask: S0,
                        then: 0,
                        els: 2
                    },
                    Some(&mut out)
                ),
                Ok(Value::Blended)
            );
            // two DISTINCT lanes, so swapping `then`/`els` is visible; the
            // expectation is spelled from the raw fixture, not from the blend
            let want: Vec<i32> = (0..fx.n)
                .map(|r| {
                    if sel.contains(&r) {
                        fx.i32s[r]
                    } else {
                        fx.i32s_b[r]
                    }
                })
                .collect();
            assert_eq!(out, want);
            assert_ne!(
                out, fx.i32s,
                "the two lanes must differ where the mask is clear"
            );
            assert_eq!(run(Terminal::Keep { mask: P0 }, None), Ok(Value::Mask(P0)));
            let empty = Program::new(
                vec![MaskOp::AndNot {
                    a: P0,
                    b: P0,
                    dst: 0,
                }],
                Terminal::MaskedMinI32 { mask: S0, lane: 0 },
            );
            assert_eq!(
                reference_execute(&empty, planes, None),
                Ok(Value::OptI32(None))
            );
            let all = Program::new(
                vec![
                    MaskOp::Or {
                        a: P0,
                        b: P0,
                        dst: 0,
                    },
                    MaskOp::Not { a: S0, dst: 0 },
                    MaskOp::Or {
                        a: S0,
                        b: P0,
                        dst: 0,
                    },
                ],
                Terminal::All { mask: S0 },
            );
            assert_eq!(reference_execute(&all, planes, None), Ok(Value::Bool(true)));
        });
    }

    /// FAILS IF: `All` over zero rows is not vacuously true or `Any` not false.
    #[test]
    fn empty_planes_reduce_to_the_identities() {
        let planes = Planes {
            n_rows: 0,
            masks: &[&[][..]],
            lanes: &[],
        };
        let all = Program::new(vec![], Terminal::All { mask: P0 });
        let any = Program::new(vec![], Terminal::Any { mask: P0 });
        assert_eq!(
            reference_execute(&all, &planes, None),
            Ok(Value::Bool(true))
        );
        assert_eq!(
            reference_execute(&any, &planes, None),
            Ok(Value::Bool(false))
        );
    }

    /// FAILS IF: any refusal is missing or carries the wrong payload — one
    /// program per `ExecError` arm the oracle can produce, EXCEPT
    /// `SumRowBound`, which needs a lane of 2^32 rows (~16 GiB) and is
    /// therefore verified by reading `validate` only: `[claimed, unverified]`.
    #[test]
    fn every_error_arm_is_reachable_with_its_exact_payload() {
        let fx = Fx::new(70);
        fx.with(|planes| {
            let e = |ops: Vec<MaskOp>, t: Terminal| {
                reference_execute(&Program::new(ops, t), planes, None)
            };
            assert_eq!(
                e(
                    vec![MaskOp::Not {
                        a: Operand::Plane(7),
                        dst: 0
                    }],
                    Terminal::Any { mask: S0 }
                ),
                Err(ExecError::PlaneOutOfRange(7))
            );
            assert_eq!(
                e(
                    vec![MaskOp::Pred {
                        pred: Pred::GtI32 { lane: 9, t: 0 },
                        under: None,
                        dst: 0
                    }],
                    Terminal::Any { mask: S0 }
                ),
                Err(ExecError::LaneOutOfRange(9))
            );
            assert_eq!(
                e(
                    vec![MaskOp::Pred {
                        pred: Pred::EqU32 { lane: 0, v: 0 },
                        under: None,
                        dst: 0
                    }],
                    Terminal::Any { mask: S0 }
                ),
                Err(ExecError::LaneKind {
                    lane: 0,
                    expected: LaneKind::U32,
                    found: LaneKind::I32
                })
            );
            assert_eq!(
                e(
                    vec![MaskOp::Pred {
                        pred: Pred::GtI32 { lane: 0, t: 0 },
                        under: Some(S0),
                        dst: 0
                    }],
                    Terminal::Any { mask: S0 }
                ),
                Err(ExecError::GateAliasesDst { dst: 0 })
            );
            assert_eq!(
                e(
                    vec![],
                    Terminal::BlendI32 {
                        mask: P0,
                        then: 0,
                        els: 0
                    }
                ),
                Err(ExecError::BlendNeedsOut)
            );
            let mut short = vec![0i32; 3];
            assert_eq!(
                reference_execute(
                    &Program::new(
                        vec![],
                        Terminal::BlendI32 {
                            mask: P0,
                            then: 0,
                            els: 0
                        }
                    ),
                    planes,
                    Some(&mut short)
                ),
                Err(ExecError::LenMismatch {
                    what: "out",
                    expected: 70,
                    found: 3
                })
            );
            // reading a slot no op wrote: the executor would see the
            // caller's reused buffer, the oracle a fresh arena
            assert_eq!(
                e(vec![], Terminal::Any { mask: S0 }),
                Err(ExecError::ScratchReadBeforeWrite { slot: 0 })
            );
            assert_eq!(
                e(
                    vec![MaskOp::And {
                        a: S0,
                        b: P0,
                        dst: 0
                    }],
                    Terminal::Any { mask: S0 }
                ),
                Err(ExecError::ScratchReadBeforeWrite { slot: 0 }),
                "an op reading its own destination before anything wrote it"
            );
            assert!(e(
                vec![
                    MaskOp::Not { a: P0, dst: 0 },
                    MaskOp::And {
                        a: S0,
                        b: P0,
                        dst: 0
                    }
                ],
                Terminal::Any { mask: S0 }
            )
            .is_ok());
            let lying = Program {
                ops: vec![],
                terminal: Terminal::Any {
                    mask: Operand::Scratch(4),
                },
                scratch_slots: 1,
            };
            assert_eq!(
                reference_execute(&lying, planes, None),
                Err(ExecError::ScratchSlotUndeclared {
                    slot: 4,
                    declared: 1
                })
            );
        });
        let short_plane = vec![0u64; 1];
        let masks: [&[u64]; 1] = [&short_plane];
        let planes = Planes {
            n_rows: 70,
            masks: &masks,
            lanes: &[],
        };
        assert_eq!(
            reference_execute(
                &Program::new(vec![], Terminal::Any { mask: P0 }),
                &planes,
                None
            ),
            Err(ExecError::LenMismatch {
                what: "plane",
                expected: 2,
                found: 1
            })
        );
        let dirty = vec![0u64, 1u64 << 10];
        let masks: [&[u64]; 1] = [&dirty];
        let planes = Planes {
            n_rows: 70,
            masks: &masks,
            lanes: &[],
        };
        assert_eq!(
            reference_execute(
                &Program::new(vec![], Terminal::Any { mask: P0 }),
                &planes,
                None
            ),
            Err(ExecError::PlaneTail(0))
        );
        let short_lane = vec![0i32; 3];
        let lanes = [LaneRef::I32(&short_lane)];
        let planes = Planes {
            n_rows: 70,
            masks: &[],
            lanes: &lanes,
        };
        assert_eq!(
            reference_execute(
                &Program::new(vec![], Terminal::MaskedSumI32 { mask: S0, lane: 0 }),
                &planes,
                None
            ),
            Err(ExecError::LenMismatch {
                what: "lane",
                expected: 70,
                found: 3
            })
        );
    }

    /// FAILS IF: the read-before-write check is quadratic in OP COUNT again.
    ///
    /// A hand-built chain of `Not`s, each reading the slot the previous wrote.
    /// `Program::ops` is a public `Vec`, so this is admissible input, and the
    /// backward scan this replaced took 3.06 s on 65_536 ops at `n_rows = 0` —
    /// a caller-controlled stall before a single row is touched. The doc that
    /// licensed the scan claimed the op count "is single digits for every
    /// program this IR can express", which was never true of a hand-built one.
    ///
    /// The assertion is a WALL-CLOCK bound, which is a blunt instrument, so it
    /// is set two orders of magnitude above the linear form's measured cost
    /// (1.5 ms at 65_536 ops) and two below the quadratic's (3.06 s). Anything
    /// in between is a machine slower than any this runs on; anything above it
    /// is the quadratic returning. The correctness half is asserted too —
    /// being fast is not the claim, being fast AND still accepting a valid
    /// chain is.
    #[test]
    fn the_read_before_write_check_is_linear_in_op_count() {
        const N: usize = 16_384;
        let mut ops = Vec::with_capacity(N);
        ops.push(MaskOp::Not { a: P0, dst: 0 });
        for i in 1..N {
            ops.push(MaskOp::Not {
                a: Operand::Scratch((i - 1) as u16),
                dst: i as u16,
            });
        }
        let p = Program::new(
            ops,
            Terminal::Count {
                mask: Operand::Scratch((N - 1) as u16),
            },
        );
        assert_eq!(
            p.scratch_slots, N as u32,
            "every op must claim its own slot"
        );
        // `n_rows = 0`, so validation is the ONLY work and the measurement
        // cannot be diluted by row evaluation.
        let empty: Vec<u64> = Vec::new();
        let masks: [&[u64]; 1] = [&empty];
        let planes = Planes {
            n_rows: 0,
            masks: &masks,
            lanes: &[],
        };
        let t = std::time::Instant::now();
        let got = reference_execute(&p, &planes, None);
        let elapsed = t.elapsed();
        assert_eq!(got, Ok(Value::Count(0)), "the chain is valid and must run");
        assert!(
            elapsed < std::time::Duration::from_millis(500),
            "validation of {N} ops took {elapsed:?} — the quadratic scan is back"
        );
    }

    /// FAILS IF: `validate` bounds only the INDEX of an `Operand::Scratch`
    /// and never the declared COUNT. Both fixtures name NO scratch operand at
    /// all, which is the whole point — `check_operand` is structurally blind
    /// to a lie it has no operand to catch, so a per-operand check cannot be
    /// what makes this pass.
    ///
    /// Two-sided: the surplus is refused, and the exact ceiling is ACCEPTED.
    /// Without the second half a validator that refused every non-zero count
    /// would look correct.
    #[test]
    fn a_declared_slot_count_past_the_addressable_ceiling_is_refused() {
        let fx = Fx::new(70);
        fx.with(|planes| {
            let lying = Program {
                ops: vec![],
                terminal: Terminal::Any { mask: P0 },
                scratch_slots: MAX_SCRATCH_SLOTS + 1,
            };
            // Anti-vacuity: nothing here NAMES a scratch slot, so the only
            // rule that can reject it is the count bound itself.
            assert!(
                !lying.ops.iter().any(|op| matches!(
                    op,
                    MaskOp::Pred {
                        under: Some(Operand::Scratch(_)),
                        ..
                    }
                )) && matches!(
                    lying.terminal,
                    Terminal::Any {
                        mask: Operand::Plane(_)
                    }
                ),
                "fixture must name no scratch operand, or it proves nothing"
            );
            assert_eq!(
                reference_execute(&lying, planes, None),
                Err(ExecError::ScratchSlotsUnaddressable {
                    declared: MAX_SCRATCH_SLOTS + 1
                })
            );
            // ...and the oracle's OTHER entry point refuses identically,
            // because its arena is sized from the same field.
            assert_eq!(
                reference_scratch(&lying, planes),
                Err(ExecError::ScratchSlotsUnaddressable {
                    declared: MAX_SCRATCH_SLOTS + 1
                })
            );
            // CAN-STAY-SILENT: exactly 65,536 slots is the largest count
            // every index can still address (slot `u16::MAX` is the
            // 65,536th), so it must pass validation untouched.
            let exact = Program {
                ops: vec![],
                terminal: Terminal::Any { mask: P0 },
                scratch_slots: MAX_SCRATCH_SLOTS,
            };
            assert!(reference_execute(&exact, planes, None).is_ok());
        });
    }

    /// FAILS IF: validation is not in program order — the FIRST bad op is
    /// the one reported, even when a later op is wrong in a "worse" way.
    #[test]
    fn the_first_bad_op_in_program_order_is_reported() {
        let fx = Fx::new(70);
        fx.with(|planes| {
            let p = Program::new(
                vec![
                    MaskOp::Not {
                        a: Operand::Plane(3),
                        dst: 0,
                    },
                    MaskOp::Pred {
                        pred: Pred::GtI32 { lane: 9, t: 0 },
                        under: None,
                        dst: 1,
                    },
                ],
                Terminal::Any { mask: S0 },
            );
            assert_eq!(
                reference_execute(&p, planes, None),
                Err(ExecError::PlaneOutOfRange(3))
            );
            let q = Program::new(
                vec![
                    MaskOp::Pred {
                        pred: Pred::GtI32 { lane: 9, t: 0 },
                        under: None,
                        dst: 1,
                    },
                    MaskOp::Not {
                        a: Operand::Plane(3),
                        dst: 0,
                    },
                ],
                Terminal::Any { mask: S0 },
            );
            assert_eq!(
                reference_execute(&q, planes, None),
                Err(ExecError::LaneOutOfRange(9))
            );
        });
    }
}
