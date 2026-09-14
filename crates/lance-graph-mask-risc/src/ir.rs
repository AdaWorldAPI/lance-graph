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
    /// cannot alias a 12-in-16-byte stride anyway — the strided operand
    /// family (`ternary_match_strided_to_mask`'s `(base, stride, group)`
    /// shape) is a named PR3 gap, not this variant.
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
}

/// One instruction. Destinations are always [`Operand::Scratch`]; input planes
/// are read-only by construction.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MaskOp {
    /// `dst = pred(lane)`. With `under = Some(m)`, only 1024-row chunks in
    /// which `m` has a survivor are evaluated; the others are written zero —
    /// the survivor-word skip (cost ∝ survivors, never ∝ rows).
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
    /// Σ `lane[i]` over set bits, widened to `i64`.
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

/// A straight-line program: ops in order, then one terminal.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Program {
    /// Instructions, executed in order.
    pub ops: Vec<MaskOp>,
    /// The single result.
    pub terminal: Terminal,
    /// How many scratch slots the program touches (`max dst + 1`).
    pub scratch_slots: u16,
}

impl Program {
    /// Assemble a program, computing its scratch requirement from the ops.
    pub fn new(ops: Vec<MaskOp>, terminal: Terminal) -> Self {
        let mut slots = 0u16;
        for op in &ops {
            let d = match *op {
                MaskOp::Pred { dst, .. }
                | MaskOp::And { dst, .. }
                | MaskOp::Or { dst, .. }
                | MaskOp::Xor { dst, .. }
                | MaskOp::AndNot { dst, .. }
                | MaskOp::Not { dst, .. }
                | MaskOp::Ternlog { dst, .. } => dst,
            };
            // saturating: `dst == u16::MAX` must not wrap `scratch_slots` to 0
            // in release (the build where the damage would be silent)
            slots = slots.max(d.saturating_add(1));
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
}

impl OpHistogram {
    /// Total mask-word passes the program spends after its predicates.
    pub fn mask_passes(&self) -> usize {
        self.two_input + self.not + self.ternlog
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// FAILS IF: `Program::new` wraps its slot count in release — `dst ==
    /// u16::MAX` would then report `scratch_slots == 0` for a program that
    /// writes slot 65535 (debug would panic instead; the saturating add
    /// makes both builds agree).
    #[test]
    fn scratch_slots_saturate_at_the_widest_dst() {
        let p = Program::new(
            vec![MaskOp::Not {
                a: Operand::Plane(0),
                dst: u16::MAX,
            }],
            Terminal::Count {
                mask: Operand::Scratch(u16::MAX),
            },
        );
        assert_eq!(p.scratch_slots, u16::MAX);
        let q = Program::new(
            vec![],
            Terminal::Count {
                mask: Operand::Plane(0),
            },
        );
        assert_eq!(q.scratch_slots, 0);
    }

    /// FAILS IF: the histogram miscounts a kind, or `mask_passes` counts a
    /// predicate as a mask pass (predicates sweep VALUE lanes and are
    /// reported separately). Fixture: one of each kind, so every counter is
    /// exactly 1 and the pass total is 4.
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
                ternlog: 1
            }
        );
        assert_eq!(h.mask_passes(), 3);
        assert_eq!(p.scratch_slots, 4);
    }
}
