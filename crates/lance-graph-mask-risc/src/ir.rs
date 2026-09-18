//! The op vocabulary — deliberately small. Every op names its operands by
//! *slot* (a borrowed input plane or a caller-owned scratch buffer); nothing
//! in the IR owns bytes.

/// Where a mask operand lives.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Operand {
    /// Input mask plane `i` of [`Planes::masks`] — borrowed, read-only.
    Plane(u16),
    /// Scratch buffer `i` of the caller's [`crate::Scratch`] — read/write.
    Scratch(u16),
}

/// A borrowed typed value plane.
#[derive(Debug, Clone, Copy)]
pub enum LaneRef<'a> {
    /// Signed 32-bit lane (ordered compares are signed).
    I32(&'a [i32]),
    /// Unsigned 32-bit lane (equality is exact bitwise; classids live here).
    U32(&'a [u32]),
    /// 64-bit lane (edge targets, ids). NOT a reading of the 12-byte V3
    /// register: which carving that register is read under is the
    /// ClassView's choice, never a lane width's, and a contiguous `&[u64]`
    /// cannot alias a 12-in-16-byte stride anyway. A strided `Operand` is a
    /// gap in THIS IR — `ndarray::simd` already ships
    /// `ternary_match_strided_to_mask`'s `(base, stride, group)` shape — and
    /// closing it is PR4/PR5 work, not this variant.
    U64(&'a [u64]),
}

impl LaneRef<'_> {
    /// Element count of the lane.
    pub fn len(&self) -> usize {
        match self {
            LaneRef::I32(v) => v.len(),
            LaneRef::U32(v) => v.len(),
            LaneRef::U64(v) => v.len(),
        }
    }

    /// Whether the lane is empty.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// Everything an execution borrows: mask planes and value lanes, all owned by
/// the caller (a mailbox, an overlay, a row store), all `n_rows` long.
#[derive(Debug, Clone, Copy)]
pub struct Planes<'a> {
    /// Row count every plane spans; tail bits past it are zero.
    pub n_rows: usize,
    /// Resident mask planes (alpha, focus, class masks, …).
    pub masks: &'a [&'a [u64]],
    /// Resident value lanes.
    pub lanes: &'a [LaneRef<'a>],
}

/// A value-lane predicate that produces a mask — the vector half of a
/// columnar filter. `lane` indexes [`Planes::lanes`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Pred {
    /// `lane[i] > t` (signed `i32`).
    GtI32 { lane: u16, t: i32 },
    /// `lane[i] < t`.
    LtI32 { lane: u16, t: i32 },
    /// `lane[i] >= t`.
    GeI32 { lane: u16, t: i32 },
    /// `lane[i] <= t`.
    LeI32 { lane: u16, t: i32 },
    /// `lane[i] == v` (signed lane, exact).
    EqI32 { lane: u16, v: i32 },
    /// `lane[i] != v`.
    NeI32 { lane: u16, v: i32 },
    /// `lane[i] == v` (`u32` lane, exact bitwise).
    EqU32 { lane: u16, v: u32 },
    /// `lane[i] != v`.
    NeU32 { lane: u16, v: u32 },
    /// `((lane[i] ^ pattern) & care) == 0` over a `u32` lane.
    MatchU32 { lane: u16, pattern: u32, care: u32 },
    /// `((lane[i] ^ pattern) & care) == 0` over a `u64` lane.
    MatchU64 { lane: u16, pattern: u64, care: u64 },
    /// `lo <= i < hi` — a predicate on the ROW INDEX, reading no lane.
    ///
    /// The contiguous-range write. On an address-ordered lane an address
    /// prefix names a contiguous subtree, so the prefix's mask is a range,
    /// not a sweep: `ndarray::simd::mask_set_range` fills it in three passes
    /// over the words (zero before, ones inside, zero after, two computed
    /// edge words) with no per-row compare at all. This is the op
    /// `lance-graph-quack`'s `Filter::prefix_u64` doc named as *"NOT yet a
    /// range WRITE … waits on the primitive"* — the primitive is in ndarray;
    /// this is the IR name for it (`ISS-MASK-RISC-HAD-NO-RANGE-OP`).
    ///
    /// The IR does NOT decide whether a lane is address-ordered — that is the
    /// planner's knowledge (a V3 table's row address is its rail). A caller
    /// that lowers a prefix to `Range` on an unordered lane gets a wrong
    /// answer, not an error; the oracle agrees with the executor on the range
    /// itself, which is all either can check.
    ///
    /// Validated: `lo <= hi` and `hi <= n_rows`
    /// ([`crate::ExecError::RangeOutOfBounds`]), so the write can never reach
    /// the tail. `lo == hi` is a legal empty range (an all-zero write, not a
    /// no-op). With `under`, the result is `range & gate`.
    Range { lo: u32, hi: u32 },
}

/// One instruction. Destinations are always [`Operand::Scratch`]; input planes
/// are read-only by construction.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MaskOp {
    /// `dst = pred(lane)`. With `under = Some(m)`, the predicate is evaluated
    /// only where `m` has a survivor and the rest is written zero — the
    /// survivor skip. Its granularity is the facade's: 64-row WORDS (an
    /// executor is free to skip coarser chunks; the result is identical by
    /// construction, since a skipped chunk is an all-zero gate). Compare cost
    /// follows the gate's live words; the per-word gate test is still ∝ rows/64.
    ///
    /// On a V3 table the coarser chunk has a natural size: a 256-row block —
    /// four words, one 256-bit vector, the 2-nibble prefix cell of the OGAR
    /// tier tile — and a 64k table's mask is exactly 1 024 words or 256 such
    /// blocks, no remainder in either unit. The rail (`u8:u8`) is not a unit
    /// of the mask at all: it is the exact row ADDRESS of a 64k table, 256 ×
    /// 256 = every row and nothing else (operator, 2026-09-15). Which unit an
    /// executor skips in is its own choice; the result is identical.
    Pred {
        pred: Pred,
        under: Option<Operand>,
        dst: u16,
    },
    /// `dst = a & b`.
    And { a: Operand, b: Operand, dst: u16 },
    /// `dst = a | b`.
    Or { a: Operand, b: Operand, dst: u16 },
    /// `dst = a ^ b`.
    Xor { a: Operand, b: Operand, dst: u16 },
    /// `dst = a & !b`.
    AndNot { a: Operand, b: Operand, dst: u16 },
    /// `dst = !a` (tail cleared).
    Not { a: Operand, dst: u16 },
    /// `dst = table[imm](a, b, c)` — any 3-input Boolean function, Intel
    /// VPTERNLOG index convention `(a << 2) | (b << 1) | c`. Semantics only;
    /// the realization per backend is `ndarray`'s.
    ///
    /// Tail obligation: an EVEN immediate (`imm & 1 == 0`, i.e. `f(0,0,0) =
    /// 0`) leaves the tail zero for conforming inputs; an ODD one — every
    /// table whose root is a negation, which a fuser mints routinely — sets
    /// every tail bit, and the executor clears the tail against `n_rows`
    /// before any `Any`/`Count` terminal reads `dst`. Same rule as
    /// `ndarray::simd::mask_ternlog`'s own doc.
    Ternlog {
        imm: u8,
        a: Operand,
        b: Operand,
        c: Operand,
        dst: u16,
    },
}

/// What the program produces. Exactly one per program; the mask it reads is
/// the program's final result.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Terminal {
    /// Population count of `mask`.
    Count { mask: Operand },
    /// Whether any bit of `mask` is set.
    Any { mask: Operand },
    /// Whether every row is set in `mask`.
    All { mask: Operand },
    /// Σ `lane[i]` over set bits, widened to `i64`. Carry-safe only while
    /// `n_rows <= `[`MASKED_SUM_I32_MAX_ROWS`]: an `i64` holds `2^32` copies
    /// of `i32::MIN` or `i32::MAX` exactly, and one more can wrap. An
    /// executor rejects this terminal on a wider plane rather than wrap.
    MaskedSumI32 { mask: Operand, lane: u16 },
    /// min `lane[i]` over set bits (`None` if empty).
    MaskedMinI32 { mask: Operand, lane: u16 },
    /// max `lane[i]` over set bits.
    MaskedMaxI32 { mask: Operand, lane: u16 },
    /// `out[i] = mask[i] ? then[i] : else[i]` into a caller buffer — the
    /// `CASE WHEN` shape with no compaction. The executor writes it into the
    /// caller's `out` slice passed alongside the program.
    BlendI32 { mask: Operand, then: u16, els: u16 },
    /// The final mask itself stays in `mask` (a scratch slot the caller
    /// reads back); nothing is reduced.
    Keep { mask: Operand },
}

/// The widest plane [`Terminal::MaskedSumI32`] is defined on: `2^32` rows.
/// The binding side is the NEGATIVE one: `2^32 · i32::MIN = −2^63 = i64::MIN`
/// exactly, and one more row of `i32::MIN` wraps. The positive side has two
/// rows of slack (`2^32 + 2` copies of `i32::MAX` still fit), so `2^32` is the
/// tight bound, not a round-number convenience. The IR states the bound; the
/// executor enforces it (`n_rows` is a plain `usize` on [`Planes`]).
pub const MASKED_SUM_I32_MAX_ROWS: usize = 1 << 32;

/// A straight-line program: ops in order, then one terminal.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Program {
    /// Instructions, executed in order.
    pub ops: Vec<MaskOp>,
    /// The single result.
    pub terminal: Terminal,
    /// How many scratch slots the program touches: `max slot + 1` over EVERY
    /// `Operand::Scratch` the program names — destinations, sources, a
    /// `Pred`'s gate, and the terminal's operands alike — sizing from
    /// destinations alone under-reports. Note a program that READS a slot no
    /// earlier op wrote is refused at validation
    /// (`ExecError::ScratchReadBeforeWrite`): pre-filled scratch is a named
    /// PR5 gap, so such a count is an upper bound on a program that will not
    /// run, never a licence to pre-fill. `u32`, not `u16`: slot `u16::MAX` is the 65,536th, and
    /// 65,536 does not fit the index type — an executor sizing its arena
    /// from this field must be able to read the count the widest slot
    /// implies.
    pub scratch_slots: u32,
}

/// The addressable scratch-slot ceiling: 65,536.
///
/// [`Operand::Scratch`] is a `u16`, so a program can NAME slots `0..=65_535`.
/// A [`Program::scratch_slots`] count above this is unreachable by
/// construction — no op or terminal could ever address the surplus — so it
/// can only come from a hand-built program that lied, and
/// `reference::validate` refuses it rather than let an executor size an
/// arena from it. One spelling, read by both the validator and
/// [`crate::Scratch::for_program`].
pub const MAX_SCRATCH_SLOTS: u32 = u16::MAX as u32 + 1;

// The value, pinned independently of the derivation above — two spellings of
// one fact, so a change to either has to be deliberate.
//
// This exists because the guard's own falsifiers are written as
// `MAX_SCRATCH_SLOTS + 1`, which TRACKS the constant: raising the ceiling
// cannot make them fail, so they cannot be what catches a change to it.
// Widening `Operand::Scratch` past `u16` would move the derivation and break
// this line, which is the intent — it forces a re-pin rather than a silent
// drift. Disable-verified: changing the derivation to any other value fails
// the build here.
const _: () = assert!(MAX_SCRATCH_SLOTS == 65_536);

impl Program {
    /// Assemble a program, computing its scratch requirement from every
    /// operand named by the ops and terminal.
    pub fn new(ops: Vec<MaskOp>, terminal: Terminal) -> Self {
        let mut slots = 0u32;
        // widened, never wrapped or saturated: slot `u16::MAX` is the
        // 65,536th and needs 65,536 buffers — a `u16` count cannot say so
        // (wrapping reported 0; saturating reported 65,535, one short)
        let mut touch = |o: Operand| {
            if let Operand::Scratch(i) = o {
                slots = slots.max(u32::from(i) + 1);
            }
        };
        for op in &ops {
            match *op {
                MaskOp::Pred { under, dst, .. } => {
                    if let Some(u) = under {
                        touch(u);
                    }
                    touch(Operand::Scratch(dst));
                }
                MaskOp::And { a, b, dst }
                | MaskOp::Or { a, b, dst }
                | MaskOp::Xor { a, b, dst }
                | MaskOp::AndNot { a, b, dst } => {
                    touch(a);
                    touch(b);
                    touch(Operand::Scratch(dst));
                }
                MaskOp::Not { a, dst } => {
                    touch(a);
                    touch(Operand::Scratch(dst));
                }
                MaskOp::Ternlog { a, b, c, dst, .. } => {
                    touch(a);
                    touch(b);
                    touch(c);
                    touch(Operand::Scratch(dst));
                }
            }
        }
        match terminal {
            Terminal::Count { mask }
            | Terminal::Any { mask }
            | Terminal::All { mask }
            | Terminal::MaskedSumI32 { mask, .. }
            | Terminal::MaskedMinI32 { mask, .. }
            | Terminal::MaskedMaxI32 { mask, .. }
            | Terminal::BlendI32 { mask, .. }
            | Terminal::Keep { mask } => touch(mask),
        }
        Self {
            ops,
            terminal,
            scratch_slots: slots,
        }
    }

    /// Count of ops of each physical kind — the "logical ops vs physical
    /// passes" bookkeeping the benchmark reports.
    pub fn op_histogram(&self) -> OpHistogram {
        let mut h = OpHistogram::default();
        for op in &self.ops {
            match op {
                // `Pred::Range` reads NO value lane (it is a row-index
                // predicate), so it is not a value-lane predicate pass; it
                // spends one linear write over the destination mask instead.
                MaskOp::Pred {
                    pred: Pred::Range { .. },
                    ..
                } => h.ranges += 1,
                MaskOp::Pred { .. } => h.predicates += 1,
                MaskOp::And { .. }
                | MaskOp::Or { .. }
                | MaskOp::Xor { .. }
                | MaskOp::AndNot { .. } => h.two_input += 1,
                MaskOp::Not { .. } => h.not += 1,
                MaskOp::Ternlog { .. } => h.ternlog += 1,
            }
        }
        h
    }
}

/// Per-kind op counts of a program.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct OpHistogram {
    /// Value-lane predicates (each one pass over a value lane).
    pub predicates: usize,
    /// Two-input mask passes (`and`/`or`/`xor`/`andnot`).
    pub two_input: usize,
    /// Complements.
    pub not: usize,
    /// Three-input passes.
    pub ternlog: usize,
    /// Row-index range writes (`Pred::Range`) — counted apart from
    /// [`predicates`](Self::predicates) because they read no value lane.
    /// Each is ONE pass over the destination mask: `mask_set_range` writes
    /// every word of `out_words` exactly once across disjoint segments (the
    /// two zero-fills below/above the range, then the head/body/tail of the
    /// range itself), never re-walking the whole mask per segment.
    pub ranges: usize,
}

impl OpHistogram {
    /// Total mask-word passes the program spends after its predicates.
    pub fn mask_passes(&self) -> usize {
        self.two_input + self.not + self.ternlog + self.ranges
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// FAILS IF: `Program::new` under-reports the arena the widest `dst`
    /// needs — `dst == u16::MAX` is slot 65,535, so 65,536 buffers; a `u16`
    /// count wrapped that to 0 in release and a saturating `u16` add reported
    /// 65,535 (one short — the codex P2 on PR #1225). Both builds must agree
    /// on 65,536.
    #[test]
    fn scratch_slots_count_the_65_536th_slot() {
        let p = Program::new(
            vec![MaskOp::Not {
                a: Operand::Plane(0),
                dst: u16::MAX,
            }],
            Terminal::Count {
                mask: Operand::Scratch(u16::MAX),
            },
        );
        assert_eq!(p.scratch_slots, 65_536);
        let q = Program::new(
            vec![],
            Terminal::Count {
                mask: Operand::Plane(0),
            },
        );
        assert_eq!(q.scratch_slots, 0);
    }

    /// FAILS IF: `scratch_slots` counts destinations only. A terminal that
    /// reads slot 7 with no ops, a source operand above every `dst`, and a
    /// `Pred` gate above every `dst` each name a buffer the arena must hold.
    #[test]
    fn scratch_slots_count_read_only_slots_too() {
        let terminal_only = Program::new(
            vec![],
            Terminal::Keep {
                mask: Operand::Scratch(7),
            },
        );
        assert_eq!(terminal_only.scratch_slots, 8);
        let source_above_dst = Program::new(
            vec![MaskOp::And {
                a: Operand::Scratch(9),
                b: Operand::Plane(0),
                dst: 1,
            }],
            Terminal::Count {
                mask: Operand::Scratch(1),
            },
        );
        assert_eq!(source_above_dst.scratch_slots, 10);
        let gate_above_dst = Program::new(
            vec![MaskOp::Pred {
                pred: Pred::GtI32 { lane: 0, t: 0 },
                under: Some(Operand::Scratch(11)),
                dst: 0,
            }],
            Terminal::Any {
                mask: Operand::Scratch(0),
            },
        );
        assert_eq!(gate_above_dst.scratch_slots, 12);
        // and a program that only ever names PLANES needs no scratch at all
        let planes_only = Program::new(
            vec![],
            Terminal::Count {
                mask: Operand::Plane(3),
            },
        );
        assert_eq!(planes_only.scratch_slots, 0);
    }

    /// FAILS IF: the stated bound is not the real one. At the bound both
    /// extremes fit an `i64`; one row past it a lane of `i32::MIN` wraps
    /// (the negative side binds — the positive side still fits for two more
    /// rows, which is why the first draft of this test, written against
    /// `i32::MAX`, was red: the bound it asserted was not the tight one).
    #[test]
    fn masked_sum_bound_is_exactly_where_i64_stops_fitting() {
        let n = MASKED_SUM_I32_MAX_ROWS as i128;
        assert!(n * i128::from(i32::MAX) <= i128::from(i64::MAX));
        assert!(n * i128::from(i32::MIN) >= i128::from(i64::MIN));
        assert!((n + 1) * i128::from(i32::MIN) < i128::from(i64::MIN));
        assert!((n + 2) * i128::from(i32::MAX) <= i128::from(i64::MAX));
        assert!((n + 3) * i128::from(i32::MAX) > i128::from(i64::MAX));
    }

    /// FAILS IF: the histogram miscounts a kind, or `mask_passes` counts a
    /// predicate as a mask pass (predicates sweep VALUE lanes and are
    /// reported separately). Fixture: one of each kind, so every counter is
    /// exactly 1 and the pass total is 4.
    /// `Pred::Range` is a row-index predicate: it reads no value lane, so it
    /// must NOT land in `predicates`, and its destination write must be
    /// counted in `mask_passes()`. Before this split a range-only program
    /// reported one predicate and ZERO mask passes, which is the one shape a
    /// cost model would read as free. (CodeRabbit, PR #1246.)
    #[test]
    fn range_is_counted_apart_from_value_lane_predicates() {
        let range_only = Program::new(
            vec![MaskOp::Pred {
                pred: Pred::Range { lo: 3, hi: 9 },
                under: None,
                dst: 0,
            }],
            Terminal::Count {
                mask: Operand::Scratch(0),
            },
        );
        let h = range_only.op_histogram();
        assert_eq!(h.ranges, 1, "the range must be counted");
        assert_eq!(h.predicates, 0, "a range reads no value lane");
        assert_eq!(h.mask_passes(), 1, "and its write is not free");

        // A lane-reading predicate still counts as one, and not as a range.
        let lane_only = Program::new(
            vec![MaskOp::Pred {
                pred: Pred::GtI32 { lane: 0, t: 3 },
                under: None,
                dst: 0,
            }],
            Terminal::Count {
                mask: Operand::Scratch(0),
            },
        );
        let h2 = lane_only.op_histogram();
        assert_eq!((h2.predicates, h2.ranges), (1, 0));
        assert_eq!(h2.mask_passes(), 0, "a bare lane predicate spends none");
    }

    #[test]
    fn op_histogram_counts_each_physical_kind_once() {
        let p = Program::new(
            vec![
                MaskOp::Pred {
                    pred: Pred::GtI32 { lane: 0, t: 3 },
                    under: None,
                    dst: 0,
                },
                MaskOp::And {
                    a: Operand::Scratch(0),
                    b: Operand::Plane(0),
                    dst: 1,
                },
                MaskOp::Not {
                    a: Operand::Scratch(1),
                    dst: 2,
                },
                MaskOp::Ternlog {
                    imm: 0x80,
                    a: Operand::Scratch(0),
                    b: Operand::Scratch(1),
                    c: Operand::Scratch(2),
                    dst: 3,
                },
            ],
            Terminal::Count {
                mask: Operand::Scratch(3),
            },
        );
        let h = p.op_histogram();
        assert_eq!(
            h,
            OpHistogram {
                predicates: 1,
                two_input: 1,
                not: 1,
                ternlog: 1,
                ranges: 0,
            }
        );
        assert_eq!(h.mask_passes(), 3);
        assert_eq!(p.scratch_slots, 4);
    }
}
