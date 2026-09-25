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
    /// cannot alias a 12-in-16-byte stride anyway — that is what
    /// [`LaneRef::Strided`] is for.
    U64(&'a [u64]),
    /// A field VIEW over unchanged record bytes: record `i`'s field starts at
    /// `bytes[first_offset + i * stride]`. Nothing is extracted — a
    /// `NodeRow` column stays inside its 512-byte rows, and any number of
    /// views (the classid at `+0`, a facet at `+4`, …) can borrow the same
    /// buffer at once. Read only by the strided predicates and terminal
    /// ([`Pred::EqU32Strided`], [`Pred::NeU32Strided`],
    /// [`Pred::MatchFacetStrided`], [`Terminal::MaskedStridedGroupSum`]),
    /// which realise through `ndarray::simd`'s `*_strided_*` kernels.
    Strided(StridedRef<'a>),
}

/// The coordinate descriptor of a [`LaneRef::Strided`] view: where record
/// `0`'s field starts, how far apart records are, and how many there are.
/// Four words describing the map; the bytes stay the caller's.
#[derive(Debug, Clone, Copy)]
pub struct StridedRef<'a> {
    /// The record bytes, borrowed (a row store, a mailbox's SoA slab, …).
    pub bytes: &'a [u8],
    /// Byte offset of record `0`'s field.
    pub first_offset: usize,
    /// Bytes between consecutive records' fields (a `NodeRow` is 512).
    pub stride: usize,
    /// How many records the view spans; must equal `Planes::n_rows`.
    pub records: usize,
}

impl LaneRef<'_> {
    /// Element count of the lane.
    pub fn len(&self) -> usize {
        match self {
            LaneRef::I32(v) => v.len(),
            LaneRef::U32(v) => v.len(),
            LaneRef::U64(v) => v.len(),
            LaneRef::Strided(v) => v.records,
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

/// One foreign table's resident validity, as bits over ITS OWN row space —
/// never `n_rows`-checked against the program's own [`Planes`], because it
/// isn't the same population.
#[derive(Debug, Clone, Copy)]
pub struct ForeignPlane<'a> {
    /// The plane's packed bits, LSB-first, `words_for(rows)` long.
    pub words: &'a [u64],
    /// The row count this plane spans — the FOREIGN table's row count, not
    /// the executing program's `n_rows`.
    pub rows: usize,
}

/// The foreign-table masks a program's [`MaskOp::Gather`] may address —
/// every entry a mask over ANOTHER table's row space. A foreign plane is
/// never a member of [`Planes::masks`] and its length is never checked
/// against the executing program's `n_rows`: that would be checking a
/// population against a population it is not.
///
/// `lanes` is the value-lane twin: foreign VALUE lanes over the SAME other
/// table's rows, addressed by [`Terminal::GroupSumViaI32::key`]. Its length
/// is the foreign table's row count — never `n_rows` either, and never
/// [`ForeignPlane::rows`]-checked against a particular plane, since a
/// program may name planes and lanes belonging to different foreign tables
/// in the same [`Foreign`] (there is no assumption that plane `i` and lane
/// `i` share a row space).
#[derive(Debug, Clone, Copy)]
pub struct Foreign<'a> {
    /// Foreign planes, indexed by [`MaskOp::Gather::foreign`].
    pub planes: &'a [ForeignPlane<'a>],
    /// Foreign value lanes, indexed by [`Terminal::GroupSumViaI32::key`].
    pub lanes: &'a [LaneRef<'a>],
}

impl Foreign<'_> {
    /// The empty foreign set — every program that names no `Gather` or
    /// `GroupSumViaI32` runs against this.
    pub const NONE: Foreign<'static> = Foreign {
        planes: &[],
        lanes: &[],
    };
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
    /// `foreign.lanes[key][fk[i]] == v` — an equality predicate on the OTHER
    /// table, evaluated THROUGH this table's foreign key, row by row, in one
    /// facade pass (`ndarray::simd::eq_u32_via_to_mask`). This is the join
    /// filter in factored form: `line WHERE partner.country = 3` reads
    /// `country[partner_id[i]]` directly, so neither a predicate plane over
    /// `partner` nor a gathered mask over `line` ever exists — the same
    /// address indirection [`Terminal::GroupSumViaI32`] uses for its key,
    /// applied to a predicate. Zero fallback: an `fk` that names no foreign
    /// row does not match. `fk` is a `U32` lane of THIS table; `key` indexes
    /// [`Foreign::lanes`] and must be `U32`.
    EqU32Via { fk: u16, key: u16, v: u32 },
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
    /// `u32_le(field_i) == v` over a [`LaneRef::Strided`] view — read in
    /// place at the view's offset, no extracted column
    /// (`ndarray::simd::eq_u32_strided_to_mask`).
    EqU32Strided { lane: u16, v: u32 },
    /// `u32_le(field_i) != v` over a [`LaneRef::Strided`] view.
    NeU32Strided { lane: u16, v: u32 },
    /// `((field_i[k] ^ pattern[k]) & care[k]) == 0` for every `k < 12` over a
    /// [`LaneRef::Strided`] view: the ternary match of one 12-byte V3 facet
    /// payload, in place (`ndarray::simd::ternary_match_strided_to_mask`).
    MatchFacetStrided {
        lane: u16,
        pattern: [u8; 12],
        care: [u8; 12],
    },
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
    /// `dst[i] = foreign.planes[foreign].words[lane_u32[i]]` — the fk
    /// SEMIJOIN: `dst` is set for row `i` exactly when `lane`'s value names a
    /// row that survives on the foreign table. Out-of-range is the
    /// zero-fallback ([`MaskOp::Gather`]'s underlying facade primitive
    /// contract), never an error: `line WHERE EXISTS partner p ON p.rid =
    /// line.partner_id AND <pred on p>` is `Gather{lane: partner_id, foreign:
    /// <mask of p rows satisfying pred>}` — the fk gather.
    ///
    /// No `under` field: the facade primitive it lowers to
    /// (`ndarray::simd::mask_gather_u32`) takes no gate. Compose a gate with
    /// [`MaskOp::And`] instead — the same shape a gated `Ternlog` or a gated
    /// `Range` write already uses when the fused kernel doesn't exist.
    Gather { lane: u16, foreign: u16, dst: u16 },
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
    /// Σ over set bits of every record's `groups` little-endian unsigned
    /// fields of `group_bytes` (`1..=4`) each, starting at the
    /// [`LaneRef::Strided`] view's offset — a whole register summed in place
    /// (`ndarray::simd::masked_strided_group_sum`), yielding
    /// [`crate::Value::StridedSum`]: `None` when the sum leaves `i64`.
    MaskedStridedGroupSum {
        mask: Operand,
        lane: u16,
        groups: u8,
        group_bytes: u8,
    },
    /// `out[i] = mask[i] ? then[i] : else[i]` into a caller buffer — the
    /// `CASE WHEN` shape with no compaction. The executor writes it into the
    /// caller's `out` slice passed alongside the program.
    BlendI32 { mask: Operand, then: u16, els: u16 },
    /// The final mask itself stays in `mask` (a scratch slot the caller
    /// reads back); nothing is reduced.
    Keep { mask: Operand },
    /// The one-to-many hop: for every row `i` where `mask` holds, sets bit
    /// `lane[i]` of the caller's `Out::Mask` buffer — provided it is `<
    /// out_rows`. Writes `out[i] |= lane[i]`'s target, union across
    /// repeats — [`ndarray::simd::mask_scatter_or_u32`]'s contract, out of
    /// range silently dropped. The caller's buffer must be exactly
    /// `words_for(out_rows)` long.
    ///
    /// **Survival condition:** the scattered mask is legal only when it IS
    /// the externally demanded result (a `hop` whose answer is the target
    /// population's mask). It is never an intermediate: a follow-on program
    /// or fold that consumes it is the forbidden
    /// projection → population → projection shape. A count over the
    /// targets is [`Terminal::CountKeyRunsU32`] (a key-ORDERED lane; there
    /// is no lowering for an unordered one); a filter through
    /// the targets is [`Pred::EqU32Via`] / [`MaskOp::Gather`] over a
    /// RESIDENT plane.
    ScatterOrU32 {
        mask: Operand,
        lane: u16,
        out_rows: u32,
    },
    /// `COUNT(DISTINCT lane[i])` over the rows where `mask` holds, keys `<
    /// out_rows` (out of range silently dropped) — the one-to-many hop
    /// FOLDED to its count. The caller's `Out::Mask` (exactly
    /// `words_for(out_rows)` long) is the fold's accumulator, one bit per
    /// distinct key; the terminal zeroes it, scatters into it tile by tile
    /// and answers its popcount as [`Value::Count`]. No second program ever
    /// reads it: `docs WHERE EXISTS line … ` counted, without the doc mask
    /// becoming an intermediate. (When the mask itself is the demanded
    /// result — a `hop` — use [`Terminal::ScatterOrU32`].)
    ///
    /// **HELD, not a lowering target.** The accumulator is population-sized
    /// (one bit per key of the universe). `tests/distinct.rs`'s pigeonhole
    /// falsifier shows that is the minimum for an exact distinct count
    /// under ARBITRARY row order — but an arbitrary row order is not a
    /// licence to carry it: the law is that an equivalent T0 projection
    /// (a lane stored in key order) serves the fold with O(1) state
    /// ([`Terminal::CountKeyRunsU32`]), and a lane that lacks one is a
    /// lowering limitation, not permission to materialise a seen-set. This
    /// terminal survives as the falsifier's instrument and for a caller
    /// whose requested computation explicitly asks for a seen-set; no
    /// lowering emits it as the automatic fallback for exact DISTINCT.
    ScatterCountU32 {
        mask: Operand,
        lane: u16,
        out_rows: u32,
    },
    /// `COUNT(DISTINCT lane[i])` over the rows where `mask` holds, on a
    /// KEY-ORDERED lane — every run of equal consecutive `lane` values is
    /// one key, so the count is the number of runs containing a selected
    /// row, folded tile by tile with a two-word carry
    /// ([`ndarray::simd::masked_key_run_count_u32`] + `KeyRunCarry`): no
    /// population-sized set, no `Out` buffer, `Out::None`. The answer is
    /// [`Value::Count`].
    ///
    /// The precondition is a lane stored in KEY ORDER (the T0 address
    /// projection that puts a child population under its parent), and it is
    /// ENFORCED, not trusted: the first key smaller than the open run's key
    /// refuses the program with [`ExecError::LaneNotOrdered`]. Non-decreasing
    /// order is the one contiguity certificate checkable with O(1) state in
    /// the same pass (proving "each key occurs in one run" would need the seen-set
    /// this terminal exists to avoid); a contiguous-but-unsorted lane is
    /// refused too, deliberately, and the check inspects EVERY key, selected
    /// or not (`1 2 1` under `1 0 1` is two runs of `1`, not one). Nothing is
    /// ever over-counted. A lane with
    /// no key-ordered projection resident is a lowering limitation, and the
    /// refusal is the answer — not [`Terminal::ScatterCountU32`].
    CountKeyRunsU32 { mask: Operand, lane: u16 },
    /// The one-terminal `GROUP BY … SUM`: for every row `i` where `mask`
    /// holds, adds `val[i]` into the caller's `Out::I64` buffer at index
    /// `key[i]` — provided `key[i] < out.len()`
    /// ([`ndarray::simd::masked_group_sum_i32`]'s contract; a key past the
    /// group universe is dropped, not an error). `out.len()` IS the group
    /// universe `K`. Same carry bound as [`Terminal::MaskedSumI32`]
    /// ([`MASKED_SUM_I32_MAX_ROWS`]) — a per-key sum can never overflow
    /// past it either, since it is strictly less work than the one-group
    /// sum the bound was derived against.
    GroupSumI32 { mask: Operand, key: u16, val: u16 },
    /// The FK-KEYED `GROUP BY … SUM`: `SUM(line.amount) GROUP BY
    /// partner.country` — `fk` is a `U32` lane of THIS table (the foreign
    /// key), `key` indexes [`Foreign::lanes`] and must be a `U32` lane on
    /// the foreign table (the group key there), and `val` is an `I32` lane
    /// of this table. For every row `i` where `mask` holds, resolves
    /// `foreign.lanes[key][fk[i]]` and adds `val[i]` at that index into the
    /// caller's `Out::I64` buffer
    /// ([`ndarray::simd::masked_group_sum_i32_via`]'s contract). Zero
    /// fallback at BOTH hops — `fk[i] >= foreign.lanes[key].len()` drops the
    /// row (the fk names no foreign row), and a resolved key `>= out.len()`
    /// drops it too (the resolved key names no group); neither is an error.
    /// The indirection is FUSED: no remapped key lane is ever materialised
    /// between the two hops. Same carry bound as [`Terminal::GroupSumI32`]
    /// ([`MASKED_SUM_I32_MAX_ROWS`]).
    GroupSumViaI32 {
        mask: Operand,
        fk: u16,
        key: u16,
        val: u16,
    },
    /// The rest of the keyed-reduction family — `COUNT(*)`, `MIN(v)`,
    /// `MAX(v)` … `GROUP BY` — as ONE terminal parameterised by where each
    /// row's group lives ([`GroupKey`]) and what is folded into it
    /// ([`GroupFold`]). For every row `i` where `mask` holds, resolves the
    /// row's group and folds it into the caller's `Out::I64` buffer, whose
    /// length IS the group universe `K`. Delegates tile by tile to the
    /// `ndarray::simd::masked_group_{count,min,max}` family; a key past the
    /// universe, and (for [`GroupKey::Via`]) an fk naming no foreign row,
    /// drop the row rather than erroring.
    ///
    /// The executor seeds the sink before the first tile with the fold's
    /// identity ([`GroupFold::seed`]): `0` for a count, `i64::MAX` for a
    /// minimum, `i64::MIN` for a maximum. A MIN/MAX slot still holding its
    /// seed afterwards is a group no selected row named — the SQL `NULL` of
    /// an empty group. Only [`GroupFold::SumSymI32`] carries a row bound
    /// ([`GROUP_SUM_SYM_MAX_ROWS`]).
    ///
    /// The coalescing `SUM` (empty group reads `0`) keeps its own terminals
    /// ([`Terminal::GroupSumI32`] / [`Terminal::GroupSumViaI32`]). The
    /// NULL-preserving `SUM` lives here as [`GroupFold::SumSymI32`], because the
    /// empty-group rule is the same for every seeded fold: a slot still
    /// holding [`GroupFold::seed`] is empty ([`GroupFold::is_empty_slot`]).
    GroupReduce {
        mask: Operand,
        key: GroupKey,
        fold: GroupFold,
    },
}

/// Where a [`Terminal::GroupReduce`] reads each row's group.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GroupKey {
    /// The group of row `i` is `lanes[lane][i]`, a `U32` lane of this table.
    Lane(u16),
    /// The group of row `i` is `foreign.lanes[key][lanes[fk][i]]` — `fk` a
    /// `U32` lane of this table, `key` a `U32` lane of the foreign table.
    /// The two hops are fused; no remapped key lane is materialised.
    Via { fk: u16, key: u16 },
    /// The group of row `i` is `lanes[hi][i] * stride + lanes[lo][i]` — the
    /// composite address of a two-column `GROUP BY`. `hi` and `lo` are both
    /// `U32` lanes of this table. Fused: no composite key lane is ever
    /// materialised. Zero fallback: a minor key `lanes[lo][i] >= stride`
    /// names no group and drops the row, exactly like a resolved key past
    /// the group universe — same contract as
    /// [`ndarray::simd::masked_group_count_u32_pair`] and its
    /// min/max/sum siblings.
    Pair { hi: u16, lo: u16, stride: u32 },
}

/// What a [`Terminal::GroupReduce`] folds into each group's slot.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GroupFold {
    /// `COUNT(*)`: each selected row adds 1.
    Count,
    /// `MIN(lanes[val])` over an `I32` lane.
    MinI32(u16),
    /// `MAX(lanes[val])` over an `I32` lane.
    MaxI32(u16),
    /// `SUM(lanes[val])` over an `I32` lane in the SYMMETRIC range — the
    /// `_sym` reading of `ndarray::simd`: real sums live in `±(2^63 − 1)`
    /// and `ndarray::simd::SYM_EMPTY_I64` (`i64::MIN`) is reserved for "no
    /// selected row reached this group". The first row a group sees REPLACES
    /// the marker; later rows add. That keeps a group whose values cancel to
    /// `0` distinct from an empty one, which the full-range
    /// [`Terminal::GroupSumI32`] cannot tell apart. The reservation is
    /// named, never implied: every other sum here is full range. Carries the
    /// tighter [`GROUP_SUM_SYM_MAX_ROWS`]. The raw sink is internal encoding;
    /// a consumer maps the marker away before treating slots as integers.
    SumSymI32(u16),
}

impl GroupFold {
    /// The fold's identity — what every slot holds before the first row.
    /// For MIN/MAX it lies outside the `i32` range, so it doubles as the
    /// empty-group marker.
    pub const fn seed(self) -> i64 {
        match self {
            GroupFold::Count => 0,
            GroupFold::MinI32(_) => i64::MAX,
            GroupFold::MaxI32(_) => i64::MIN,
            GroupFold::SumSymI32(_) => ndarray::simd::SYM_EMPTY_I64,
        }
    }

    /// Whether a slot holding `v` after the fold is a group no selected row
    /// reached — the SQL `NULL` of an empty group. The rule is uniform:
    /// empty ⇔ the slot still holds [`GroupFold::seed`]. `COUNT` has no
    /// empty groups: a zero count is a real answer, not a `NULL`.
    pub const fn is_empty_slot(self, v: i64) -> bool {
        match self {
            GroupFold::Count => false,
            _ => v == self.seed(),
        }
    }
}

/// The widest plane a NULL-preserving [`GroupFold::SumSymI32`] is defined on:
/// `2^32 − 1` rows. One less than [`MASKED_SUM_I32_MAX_ROWS`] because the
/// seed doubles as the empty marker: exactly `2^32` rows of `i32::MIN` sum
/// to `i64::MIN`, a real value that would read back as `NULL`. Below that
/// row count every real sum lies strictly above `i64::MIN`, leaving the
/// symmetric range `±(2^63 − 1)` — closed under negation.
pub const GROUP_SUM_SYM_MAX_ROWS: usize = (1 << 32) - 1;

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
                // `foreign` is not a `Scratch`/`Plane` operand — it indexes
                // `Foreign::planes`, a wholly separate address space this
                // touch-based accounting has no business over.
                MaskOp::Gather { dst, .. } => {
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
            | Terminal::MaskedStridedGroupSum { mask, .. }
            | Terminal::BlendI32 { mask, .. }
            | Terminal::ScatterOrU32 { mask, .. }
            | Terminal::ScatterCountU32 { mask, .. }
            | Terminal::CountKeyRunsU32 { mask, .. }
            | Terminal::GroupSumI32 { mask, .. }
            | Terminal::GroupSumViaI32 { mask, .. }
            | Terminal::GroupReduce { mask, .. }
            | Terminal::Keep { mask } => touch(mask),
        }
        Self {
            ops,
            terminal,
            scratch_slots: slots,
        }
    }

    /// The fused lowering of this program, if its terminal can fold straight
    /// from its operands without any derived membership bits being written.
    ///
    /// A mask expression denotes membership; whether that membership ever
    /// becomes a bitmap is the TERMINAL's decision, not the op's. The ops of
    /// this IR are assignments (`dst = …`), so read literally every program
    /// writes its intermediate membership into a scratch slot before the
    /// terminal reads it back. For a demanded result that is a scalar —
    /// `Count`, `Any` — that write is a materialization nobody asked for.
    ///
    /// The shape recognised here is `Range[lo, hi)` optionally gated by a
    /// RESIDENT plane, folded by `Count` or `Any`: the range's interior words
    /// have an all-ones mask, so the whole relation is the resident plane's own
    /// words over the touched span plus two edge masks. No slot is needed.
    ///
    /// Everything else returns `None` and runs the ordinary tiled path.
    /// `Keep` in particular is NEVER fused: it is the explicit election of a
    /// bitmap, and the materialization is its whole point.
    pub fn fused_terminal(&self) -> Option<FusedTerminal> {
        let [MaskOp::Pred {
            pred: Pred::Range { lo, hi },
            under,
            dst,
        }] = self.ops.as_slice()
        else {
            return None;
        };
        let plane = match under {
            None => None,
            Some(Operand::Plane(p)) => Some(*p),
            // A scratch gate is itself derived membership; nothing here holds
            // it, so the shape is not fusable.
            Some(Operand::Scratch(_)) => return None,
        };
        if usize::from(*dst) >= FUSED_SLOT_CAP {
            return None;
        }
        let fold = match self.terminal {
            Terminal::Count { mask } if mask == Operand::Scratch(*dst) => FusedFold::Count,
            Terminal::Any { mask } if mask == Operand::Scratch(*dst) => FusedFold::Any,
            _ => return None,
        };
        Some(FusedTerminal {
            lo: *lo,
            hi: *hi,
            plane,
            fold,
        })
    }

    /// The no-mask lowering of a Boolean membership over RESIDENT planes, if
    /// this program is one: a sequence of `And` / `Or` / `Xor` / `AndNot` /
    /// `Not` / `Ternlog` ops that, taken together, read at most THREE distinct
    /// [`Operand::Plane`]s, folded by `Count` or `Any` of a slot the sequence
    /// wrote.
    ///
    /// The program representation collapses before execution: each op is
    /// interpreted SYMBOLICALLY as an 8-bit truth table over the (at most
    /// three) plane leaves, in the VPTERNLOG input convention (leaf 0 reads
    /// `0xF0`, leaf 1 `0xCC`, leaf 2 `0xAA`). `And`/`Or`/`Xor`/`AndNot`/`Not`
    /// combine their inputs' tables bitwise, and `Ternlog` applies its own
    /// immediate to its three inputs' tables bit by bit. The table left in the
    /// terminal's slot IS the immediate of one ternlog over the leaves, so
    /// `ndarray::simd::mask_ternlog_popcount` / `mask_ternlog_any` fold the
    /// whole sequence straight from the planes' words to a scalar. No
    /// intermediate slot is ever written: the scratch ops exist only in the
    /// program text, never in memory.
    ///
    /// Slots are tracked in program order, so a slot overwritten mid-sequence
    /// reads its latest value, exactly as the tiled path executes it. `Not`'s
    /// tail clearing needs no special case: the fold restricts the
    /// population's own last word to its live rows, which is the same result
    /// on every live row.
    ///
    /// Declines (returns `None`, ordinary path) on: a fourth distinct plane
    /// anywhere in the sequence, a `Pred` or `Gather`, a read of a slot the
    /// sequence has not yet written, a slot at or above
    /// [`FUSED_SLOT_CAP`] (the interpreter keeps its tables in a fixed
    /// on-stack array so that recognition never allocates), or any terminal
    /// other than `Count`/`Any`. `Keep` is not a scalar fold; its bitmap
    /// lowering is [`Program::fused_keep`].
    pub fn fused_ternlog(&self) -> Option<FusedTernlog> {
        let (fold, slot) = match self.terminal {
            Terminal::Count {
                mask: Operand::Scratch(s),
            } => (FusedFold::Count, s),
            Terminal::Any {
                mask: Operand::Scratch(s),
            } => (FusedFold::Any, s),
            _ => return None,
        };
        let (imm, a, b, c) = self.ternlog_table(slot)?;
        Some(FusedTernlog { imm, a, b, c, fold })
    }

    /// The bitmap-producing twin of [`Program::fused_ternlog`]: the same
    /// Boolean chain over at most three resident planes, but consumed by
    /// [`Terminal::Keep`] of the slot it wrote. The chain collapses to ONE
    /// ternlog table exactly as for `Count`/`Any`; the executor then writes
    /// that table's result straight into the caller's [`crate::Out::Mask`] in
    /// a single pass, instead of writing every intermediate op into scratch
    /// and copying the last slot out.
    ///
    /// `Keep` stays the explicit election of a bitmap: this lowering still
    /// writes one — the demanded one, and only it. It applies only when the
    /// caller passes `Out::Mask`; a whole-width scratch caller that reads the
    /// result from its slot (`Out::None`) keeps the tiled path, so
    /// [`Program::requires_scratch`] stays `true` for these programs.
    pub fn fused_keep(&self) -> Option<FusedKeep> {
        let Terminal::Keep {
            mask: Operand::Scratch(slot),
        } = self.terminal
        else {
            return None;
        };
        let (imm, a, b, c) = self.ternlog_table(slot)?;
        Some(FusedKeep { imm, a, b, c, slot })
    }

    /// Interpret the op sequence symbolically as ONE 8-bit ternlog table over
    /// at most three distinct resident planes, read from `slot` once the
    /// sequence has run. The shared core of [`Program::fused_ternlog`] and
    /// [`Program::fused_keep`]; see the former for the declines.
    fn ternlog_table(&self, slot: u16) -> Option<(u8, u16, u16, u16)> {
        if self.ops.is_empty() {
            return None;
        }
        // Distinct plane leaves in first-read order, and the table each one
        // contributes in the VPTERNLOG input convention.
        const LEAF_TABLES: [u8; 3] = [0xF0, 0xCC, 0xAA];
        let mut leaves: [u16; 3] = [0; 3];
        let mut n_leaves = 0usize;
        let mut slots: [Option<u8>; FUSED_SLOT_CAP] = [None; FUSED_SLOT_CAP];
        let read = |o: &Operand,
                    leaves: &mut [u16; 3],
                    n_leaves: &mut usize,
                    slots: &[Option<u8>; FUSED_SLOT_CAP]|
         -> Option<u8> {
            match *o {
                Operand::Plane(p) => {
                    let i = match leaves[..*n_leaves].iter().position(|&q| q == p) {
                        Some(i) => i,
                        None if *n_leaves < 3 => {
                            leaves[*n_leaves] = p;
                            *n_leaves += 1;
                            *n_leaves - 1
                        }
                        None => return None,
                    };
                    Some(LEAF_TABLES[i])
                }
                Operand::Scratch(s) => *slots.get(usize::from(s))?,
            }
        };
        for op in &self.ops {
            let (t, dst) = match op {
                MaskOp::And { a, b, dst } => (
                    read(a, &mut leaves, &mut n_leaves, &slots)?
                        & read(b, &mut leaves, &mut n_leaves, &slots)?,
                    *dst,
                ),
                MaskOp::Or { a, b, dst } => (
                    read(a, &mut leaves, &mut n_leaves, &slots)?
                        | read(b, &mut leaves, &mut n_leaves, &slots)?,
                    *dst,
                ),
                MaskOp::Xor { a, b, dst } => (
                    read(a, &mut leaves, &mut n_leaves, &slots)?
                        ^ read(b, &mut leaves, &mut n_leaves, &slots)?,
                    *dst,
                ),
                MaskOp::AndNot { a, b, dst } => (
                    read(a, &mut leaves, &mut n_leaves, &slots)?
                        & !read(b, &mut leaves, &mut n_leaves, &slots)?,
                    *dst,
                ),
                MaskOp::Not { a, dst } => (!read(a, &mut leaves, &mut n_leaves, &slots)?, *dst),
                MaskOp::Ternlog { imm, a, b, c, dst } => {
                    let (ta, tb, tc) = (
                        read(a, &mut leaves, &mut n_leaves, &slots)?,
                        read(b, &mut leaves, &mut n_leaves, &slots)?,
                        read(c, &mut leaves, &mut n_leaves, &slots)?,
                    );
                    (apply_table(*imm, ta, tb, tc), *dst)
                }
                MaskOp::Pred { .. } | MaskOp::Gather { .. } => return None,
            };
            *slots.get_mut(usize::from(dst))? = Some(t);
        }
        let imm = (*slots.get(usize::from(slot))?)?;
        if n_leaves == 0 {
            return None;
        }
        // Unused leaf positions are don't-cares of `imm` (it was computed
        // without them); bind them to leaf 0 so every operand is a real plane.
        let a = leaves[0];
        let b = if n_leaves > 1 { leaves[1] } else { a };
        let c = if n_leaves > 2 { leaves[2] } else { a };
        Some((imm, a, b, c))
    }

    /// The two-level twin of [`Program::fused_ternlog`] / [`Program::fused_keep`]
    /// for chains over FOUR or FIVE distinct resident planes.
    ///
    /// The chain is interpreted symbolically as one 32-bit truth table over
    /// its leaves, then split as a simple disjoint decomposition
    /// `f = h(g(x, y, z), u, v)` (Ashenhurst): for some choice of three inner
    /// leaves, every one of the four `(u, v)` restrictions of `f` is one of
    /// `{0, g, !g, 1}` for a single common `g`. When such a split exists the
    /// executor runs the chain as two ternlog passes per chunk — `t = g(x,y,z)`
    /// into an on-stack chunk, then `h(t, u, v)` folded by `Count`/`Any` or
    /// written into the demanded `Out::Mask` — instead of one tile pass per op.
    ///
    /// Declines (returns `None`) on everything [`Program::fused_ternlog`]
    /// declines except the three-plane limit, on a sixth distinct plane, and
    /// on a chain with no such decomposition, which keeps the tiled path. A
    /// chain over three or fewer planes is claimed by the one-level folds
    /// first ([`Program::lowering`] tries them in that order).
    pub fn fused_tern2(&self) -> Option<FusedTern2> {
        let (fold, slot) = match self.terminal {
            Terminal::Count {
                mask: Operand::Scratch(s),
            } => (Tern2Fold::Count, s),
            Terminal::Any {
                mask: Operand::Scratch(s),
            } => (Tern2Fold::Any, s),
            Terminal::Keep {
                mask: Operand::Scratch(s),
            } => (Tern2Fold::Keep { slot: s }, s),
            _ => return None,
        };
        let (f, leaves, n_leaves) = self.chain_table5(slot)?;
        let (inner, imm1, outer, imm2) = decompose5(f)?;
        // Leaf positions the chain never read are don't-cares of `f`, hence of
        // both tables; bind them to a real plane so every operand is one.
        let plane = |i: usize| if i < n_leaves { leaves[i] } else { leaves[0] };
        Some(FusedTern2 {
            imm1,
            x: plane(inner[0]),
            y: plane(inner[1]),
            z: plane(inner[2]),
            imm2,
            u: plane(outer[0]),
            v: plane(outer[1]),
            fold,
        })
    }

    /// Interpret the op sequence symbolically as ONE 32-bit truth table over
    /// at most five distinct resident planes, read from `slot`. Bit `k` of a
    /// table is the function's value when leaf `i` equals `(k >> i) & 1`.
    fn chain_table5(&self, slot: u16) -> Option<(u32, [u16; 5], usize)> {
        if self.ops.is_empty() {
            return None;
        }
        let mut leaves: [u16; 5] = [0; 5];
        let mut n_leaves = 0usize;
        let mut slots: [Option<u32>; FUSED_SLOT_CAP] = [None; FUSED_SLOT_CAP];
        let read = |o: &Operand,
                    leaves: &mut [u16; 5],
                    n_leaves: &mut usize,
                    slots: &[Option<u32>; FUSED_SLOT_CAP]|
         -> Option<u32> {
            match *o {
                Operand::Plane(p) => {
                    let i = match leaves[..*n_leaves].iter().position(|&q| q == p) {
                        Some(i) => i,
                        None if *n_leaves < 5 => {
                            leaves[*n_leaves] = p;
                            *n_leaves += 1;
                            *n_leaves - 1
                        }
                        None => return None,
                    };
                    Some(LEAF_TABLES5[i])
                }
                Operand::Scratch(s) => *slots.get(usize::from(s))?,
            }
        };
        for op in &self.ops {
            let (t, dst) = match op {
                MaskOp::And { a, b, dst } => (
                    read(a, &mut leaves, &mut n_leaves, &slots)?
                        & read(b, &mut leaves, &mut n_leaves, &slots)?,
                    *dst,
                ),
                MaskOp::Or { a, b, dst } => (
                    read(a, &mut leaves, &mut n_leaves, &slots)?
                        | read(b, &mut leaves, &mut n_leaves, &slots)?,
                    *dst,
                ),
                MaskOp::Xor { a, b, dst } => (
                    read(a, &mut leaves, &mut n_leaves, &slots)?
                        ^ read(b, &mut leaves, &mut n_leaves, &slots)?,
                    *dst,
                ),
                MaskOp::AndNot { a, b, dst } => (
                    read(a, &mut leaves, &mut n_leaves, &slots)?
                        & !read(b, &mut leaves, &mut n_leaves, &slots)?,
                    *dst,
                ),
                MaskOp::Not { a, dst } => (!read(a, &mut leaves, &mut n_leaves, &slots)?, *dst),
                MaskOp::Ternlog { imm, a, b, c, dst } => {
                    let (ta, tb, tc) = (
                        read(a, &mut leaves, &mut n_leaves, &slots)?,
                        read(b, &mut leaves, &mut n_leaves, &slots)?,
                        read(c, &mut leaves, &mut n_leaves, &slots)?,
                    );
                    (apply_table32(*imm, ta, tb, tc), *dst)
                }
                MaskOp::Pred { .. } | MaskOp::Gather { .. } => return None,
            };
            *slots.get_mut(usize::from(dst))? = Some(t);
        }
        let f = (*slots.get(usize::from(slot))?)?;
        if n_leaves == 0 {
            return None;
        }
        Some((f, leaves, n_leaves))
    }

    /// How this program executes, decided from its text alone: the range
    /// fold ([`Program::fused_terminal`]), the Boolean-membership fold
    /// ([`Program::fused_ternlog`]), or the tiled path. Tried in that order,
    /// the same order the executor has always tried them.
    pub fn lowering(&self) -> Lowering {
        if let Some(f) = self.fused_terminal() {
            Lowering::Range(f)
        } else if let Some(f) = self.fused_ternlog() {
            Lowering::Ternlog(f)
        } else if let Some(f) = self.fused_keep() {
            Lowering::TernlogKeep(f)
        } else if let Some(f) = self.fused_tern2() {
            Lowering::Tern2(f)
        } else {
            Lowering::Tiled
        }
    }

    /// Recognise this program's lowering ONCE, for a caller that executes it
    /// repeatedly. See [`Compiled`].
    pub fn compile(&self) -> Compiled<'_> {
        Compiled {
            program: self,
            lowering: self.lowering(),
        }
    }

    /// Whether executing this program needs any scratch slot at all.
    ///
    /// DERIVED, not declared: a flag is a claim, a derived predicate is a
    /// proof. `false` exactly when [`Program::fused_terminal`] or
    /// [`Program::fused_ternlog`] lowers the program (or it names no slot),
    /// and [`crate::Scratch::for_program`] carves zero slots for such a
    /// program.
    pub fn requires_scratch(&self) -> bool {
        self.compile().requires_scratch()
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
                //
                // A GATED range spends a SECOND pass, and that is not true of
                // any other predicate: every lane predicate has a fused
                // `*_to_mask_under` kernel, so gating it costs nothing extra.
                // There is no `mask_set_range_under`, so `exec` runs the gated
                // range as `mask_set_range` followed by `mask_and_assign` —
                // literally an `and`, which is what `two_input` already counts.
                // Charged there rather than to a new field, so the asymmetry is
                // visible in the histogram instead of hidden behind a name.
                // (CodeRabbit, PR #1246. Closing this would need the fused
                // primitive upstream, tracked as `mask_set_range_under`.)
                MaskOp::Pred {
                    pred: Pred::Range { .. },
                    under,
                    ..
                } => {
                    h.ranges += 1;
                    if under.is_some() {
                        h.two_input += 1;
                    }
                }
                // The strided predicates have no `*_under` kernel either, so a
                // gated one is the strided kernel followed by `mask_and_assign`
                // — one extra `and`, charged to `two_input` like the gated
                // range above. `NeU32Strided` is also `!(==)`: the eq kernel
                // and then `mask_not_assign`, a `not` pass on every call,
                // gated or not. (Codex P2, PR #1284.)
                MaskOp::Pred {
                    pred:
                        Pred::EqU32Strided { .. }
                        | Pred::NeU32Strided { .. }
                        | Pred::MatchFacetStrided { .. },
                    under,
                    ..
                } => {
                    h.predicates += 1;
                    if matches!(
                        op,
                        MaskOp::Pred {
                            pred: Pred::NeU32Strided { .. },
                            ..
                        }
                    ) {
                        h.not += 1;
                    }
                    if under.is_some() {
                        h.two_input += 1;
                    }
                }
                MaskOp::Pred { .. } => h.predicates += 1,
                MaskOp::And { .. }
                | MaskOp::Or { .. }
                | MaskOp::Xor { .. }
                | MaskOp::AndNot { .. } => h.two_input += 1,
                MaskOp::Not { .. } => h.not += 1,
                MaskOp::Ternlog { .. } => h.ternlog += 1,
                MaskOp::Gather { .. } => h.gather += 1,
            }
        }
        h
    }
}

/// How a [`Program`] executes, as [`Program::lowering`] decides it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Lowering {
    /// A range over at most one resident plane, folded straight to
    /// `Count`/`Any` ([`Program::fused_terminal`]).
    Range(FusedTerminal),
    /// A Boolean chain over at most three resident planes, collapsed to one
    /// ternlog table and folded straight to `Count`/`Any`
    /// ([`Program::fused_ternlog`]).
    Ternlog(FusedTernlog),
    /// The same collapsed chain, consumed by `Keep`: written in one pass into
    /// the demanded `Out::Mask` ([`Program::fused_keep`]). Taken only when the
    /// caller passes `Out::Mask`; otherwise the executor runs it tiled.
    TernlogKeep(FusedKeep),
    /// A chain over four or five resident planes, split into two ternlog
    /// tables ([`Program::fused_tern2`]). A `Keep` terminal takes this path
    /// only when the caller passes `Out::Mask`; otherwise it runs tiled.
    Tern2(FusedTern2),
    /// Neither fold applies: the ops run tile by tile through scratch.
    Tiled,
}

/// A [`Program`] whose lowering was recognised once, at construction.
///
/// Recognising a fold is compilation, not execution: the answer depends only
/// on the program's text, so it is the same on every call. `execute_extent`
/// recognises on each call, which is right for a one-shot program; a caller
/// that runs the same program many times (per request, per extent, per tile
/// of a larger scan) builds a `Compiled` once and hands it to
/// [`crate::exec::execute_compiled`], which skips recognition entirely.
///
/// It BORROWS the program, so the program cannot change while the recognised
/// lowering is held: the cache cannot go stale by construction. What stays
/// per call is validation, because it checks the program against the
/// `Planes`, `Foreign` and `Out` of that call.
#[derive(Debug, Clone, Copy)]
pub struct Compiled<'p> {
    program: &'p Program,
    lowering: Lowering,
}

impl<'p> Compiled<'p> {
    /// The program this lowering was recognised from.
    pub fn program(&self) -> &'p Program {
        self.program
    }

    /// The recognised lowering.
    pub fn lowering(&self) -> Lowering {
        self.lowering
    }

    /// [`Program::requires_scratch`], answered from the cached lowering.
    pub fn requires_scratch(&self) -> bool {
        // `TernlogKeep` needs no scratch when the caller demands `Out::Mask`,
        // but a caller reading the result from its slot (`Out::None`) runs
        // tiled, so the program still requires scratch in general.
        self.program.scratch_slots > 0
            && match self.lowering {
                Lowering::Tiled | Lowering::TernlogKeep(_) => true,
                Lowering::Tern2(f) => matches!(f.fold, Tern2Fold::Keep { .. }),
                Lowering::Range(_) | Lowering::Ternlog(_) => false,
            }
    }
}

/// A program the executor folds without writing membership bits: see
/// [`Program::fused_terminal`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FusedTerminal {
    /// First row of the range.
    pub lo: u32,
    /// One past the last row of the range.
    pub hi: u32,
    /// The resident plane gating the range, or `None` for a bare range.
    pub plane: Option<u16>,
    /// The scalar the terminal demands.
    pub fold: FusedFold,
}

/// A Boolean membership over resident planes that the executor folds
/// without writing membership bits: see [`Program::fused_ternlog`].
///
/// The table is in the VPTERNLOG index convention `(a << 2) | (b << 1) | c`
/// that [`MaskOp::Ternlog`] already uses. It is the WHOLE op sequence's
/// collapsed function; a sequence over fewer than three distinct planes gets a
/// table that ignores the unused positions, which are bound to `a`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FusedTernlog {
    /// The 8-bit truth table.
    pub imm: u8,
    /// Resident plane read as the table's `a`.
    pub a: u16,
    /// Resident plane read as the table's `b`.
    pub b: u16,
    /// Resident plane read as the table's `c`.
    pub c: u16,
    /// The scalar the terminal demands.
    pub fold: FusedFold,
}

/// A [`Program::fused_keep`] lowering: one ternlog table over three resident
/// planes, written straight into the demanded `Out::Mask`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FusedKeep {
    /// The 8-bit truth table.
    pub imm: u8,
    /// Resident plane read as the table's `a`.
    pub a: u16,
    /// Resident plane read as the table's `b`.
    pub b: u16,
    /// Resident plane read as the table's `c`.
    pub c: u16,
    /// The scratch slot the program's `Keep` names — what `Value::Mask`
    /// reports, exactly as the tiled path reports it.
    pub slot: u16,
}

/// A [`Program::fused_tern2`] lowering: `h(g(x, y, z), u, v)` with `g` =
/// `imm1` and `h` = `imm2`, both in the VPTERNLOG index convention.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FusedTern2 {
    /// The inner table `g`.
    pub imm1: u8,
    /// Resident plane read as `g`'s `a`.
    pub x: u16,
    /// Resident plane read as `g`'s `b`.
    pub y: u16,
    /// Resident plane read as `g`'s `c`.
    pub z: u16,
    /// The outer table `h`, read as `h(g, u, v)`.
    pub imm2: u8,
    /// Resident plane read as `h`'s `b`.
    pub u: u16,
    /// Resident plane read as `h`'s `c`.
    pub v: u16,
    /// What the terminal demands.
    pub fold: Tern2Fold,
}

/// The terminal a [`FusedTern2`] feeds.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Tern2Fold {
    /// Population of the relation.
    Count,
    /// Whether the relation is non-empty.
    Any,
    /// The bitmap itself, written into the demanded `Out::Mask`; `slot` is
    /// what `Value::Mask` reports, exactly as the tiled path reports it.
    Keep { slot: u16 },
}

/// The five leaf truth tables: bit `k` is set exactly when `(k >> i) & 1`.
const LEAF_TABLES5: [u32; 5] = [
    0xAAAA_AAAA,
    0xCCCC_CCCC,
    0xF0F0_F0F0,
    0xFF00_FF00,
    0xFFFF_0000,
];

/// [`apply_table`] over 32-bit tables: output bit `i` is
/// `imm[(ta_i << 2) | (tb_i << 1) | tc_i]`.
fn apply_table32(imm: u8, ta: u32, tb: u32, tc: u32) -> u32 {
    let mut out = 0u32;
    for idx in 0..8u8 {
        if imm >> idx & 1 == 1 {
            let pick = |bit: u8, t: u32| if idx & bit != 0 { t } else { !t };
            out |= pick(4, ta) & pick(2, tb) & pick(1, tc);
        }
    }
    out
}

/// Find a simple disjoint decomposition `f = h(g(x, y, z), u, v)` of a
/// five-leaf table. Returns `([x, y, z], g, [u, v], h)` with `g` in the
/// ternlog convention over `(x, y, z)` and `h` over `(g, u, v)`.
///
/// Tries the ten inner triples in lexicographic order and takes the first
/// that works, so the answer is a function of `f` alone.
fn decompose5(f: u32) -> Option<([usize; 3], u8, [usize; 2], u8)> {
    for x in 0..5 {
        for y in x + 1..5 {
            for z in y + 1..5 {
                let mut rest = (0..5).filter(|&i| i != x && i != y && i != z);
                let (u, v) = (rest.next()?, rest.next()?);
                // The four restrictions r[(U << 1) | V] as 8-bit ternlog tables
                // over (x, y, z).
                let mut r = [0u8; 4];
                for (uv, ru) in r.iter_mut().enumerate() {
                    let (bu, bv) = ((uv >> 1) & 1, uv & 1);
                    for j in 0..8usize {
                        let k = ((j >> 2) & 1) << x
                            | ((j >> 1) & 1) << y
                            | (j & 1) << z
                            | bu << u
                            | bv << v;
                        *ru |= (((f >> k) & 1) as u8) << j;
                    }
                }
                let g = r
                    .iter()
                    .copied()
                    .find(|&t| t != 0 && t != 0xFF)
                    .unwrap_or(0xF0);
                let mut h = 0u8;
                let ok = r.iter().enumerate().all(|(uv, &t)| {
                    // h index = (G << 2) | uv; set h for G = 0 and G = 1.
                    let (h0, h1) = match t {
                        0 => (0, 0),
                        0xFF => (1, 1),
                        t if t == g => (0, 1),
                        t if t == !g => (1, 0),
                        _ => return false,
                    };
                    h |= h0 << uv | h1 << (4 | uv);
                    true
                });
                if ok {
                    return Some(([x, y, z], g, [u, v], h));
                }
            }
        }
    }
    None
}

/// Apply the VPTERNLOG table `imm` bitwise to three input tables: output bit
/// `i` is `imm[(ta_i << 2) | (tb_i << 1) | tc_i]`.
fn apply_table(imm: u8, ta: u8, tb: u8, tc: u8) -> u8 {
    let mut out = 0u8;
    for i in 0..8 {
        let idx = ((ta >> i) & 1) << 2 | ((tb >> i) & 1) << 1 | ((tc >> i) & 1);
        out |= ((imm >> idx) & 1) << i;
    }
    out
}

/// The highest scratch slot (exclusive) a fused shape may name.
///
/// The fused paths validate a program with an on-stack slot bitmap of
/// `FUSED_SLOT_CAP.div_ceil(64)` words and never touch real scratch, so a
/// slot the bitmap cannot mark would be rejected as a read before a write.
/// [`Program::fused_ternlog`]'s symbolic interpreter keeps its per-slot
/// tables in a fixed on-stack array of the same length, because recognition
/// runs on the execute path and must not allocate. A shape naming a higher
/// slot is simply not fused and runs on the tiled path. A bound of the
/// recogniser, never of the semantics.
pub const FUSED_SLOT_CAP: usize = 32;

/// The scalar folds that may consume membership without materializing it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FusedFold {
    /// Population of the relation.
    Count,
    /// Whether the relation is non-empty.
    Any,
}

/// The mask words a row range `[lo, hi)` touches: none when `lo == hi`,
/// otherwise `floor(lo / 64) ..= floor((hi - 1) / 64)`. One spelling, used by
/// the fused executor and by the tests that pin the touched-word law.
pub fn touched_words(lo: u32, hi: u32) -> core::ops::Range<usize> {
    span_words(lo as usize, hi as usize)
}

/// [`touched_words`] over `usize` rows — the SAME law, for an execution
/// extent, whose bounds are `Planes::n_rows`-typed rather than `Pred::Range`-
/// typed. `touched_words` delegates here, so there is one spelling.
pub(crate) fn span_words(lo: usize, hi: usize) -> core::ops::Range<usize> {
    if lo >= hi {
        return 0..0;
    }
    (lo / 64)..((hi - 1) / 64 + 1)
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
    ///
    /// A range under a gate spends a second pass, counted in
    /// [`two_input`](Self::two_input) because it IS one: `exec` follows the
    /// write with `mask_and_assign`. Unlike every lane predicate, which has a
    /// fused `*_to_mask_under` kernel, no `mask_set_range_under` exists.
    pub ranges: usize,
    /// `Gather` (the fk semijoin). One pass over the destination mask, the
    /// same accounting as a range write — it reads no value LANE (it reads
    /// a foreign PLANE at a data-dependent row), so it is counted apart from
    /// [`predicates`](Self::predicates) for the same reason `ranges` is.
    pub gather: usize,
}

impl OpHistogram {
    /// Total mask-word passes the program spends after its predicates.
    pub fn mask_passes(&self) -> usize {
        self.two_input + self.not + self.ternlog + self.ranges + self.gather
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

    /// A GATED range costs two passes, and a gated LANE predicate costs none:
    /// every lane predicate has a fused `*_to_mask_under` kernel, while the
    /// range has no `mask_set_range_under`, so `exec` runs write-then-`and`.
    /// Both halves matter — asserting only the range would not show that the
    /// extra pass is specific to it. (CodeRabbit, PR #1246.)
    #[test]
    fn a_gated_range_costs_the_intersection_but_a_gated_lane_predicate_does_not() {
        let gated_range = Program::new(
            vec![MaskOp::Pred {
                pred: Pred::Range { lo: 3, hi: 9 },
                under: Some(Operand::Plane(0)),
                dst: 0,
            }],
            Terminal::Count {
                mask: Operand::Scratch(0),
            },
        );
        let h = gated_range.op_histogram();
        assert_eq!((h.ranges, h.two_input), (1, 1), "write + intersection");
        assert_eq!(h.mask_passes(), 2, "a gated range spends BOTH");

        let gated_lane = Program::new(
            vec![MaskOp::Pred {
                pred: Pred::GtI32 { lane: 0, t: 3 },
                under: Some(Operand::Plane(0)),
                dst: 0,
            }],
            Terminal::Count {
                mask: Operand::Scratch(0),
            },
        );
        let h2 = gated_lane.op_histogram();
        assert_eq!(
            (h2.predicates, h2.two_input),
            (1, 0),
            "the fused `*_to_mask_under` kernel adds no pass"
        );
        assert_eq!(h2.mask_passes(), 0, "so gating a lane predicate is free");
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
                gather: 0,
            }
        );
        assert_eq!(h.mask_passes(), 3);
        assert_eq!(p.scratch_slots, 4);
    }

    /// FAILS IF: the histogram forgets the passes the executor really spends
    /// on a strided predicate — the `and` a gate costs (no `*_under` strided
    /// kernel exists) and the `not` that turns `NeU32Strided`'s eq kernel into
    /// `!=`. Silence twin: an UNGATED Eq or Match is one predicate and nothing
    /// else, exactly like a lane predicate.
    #[test]
    fn op_histogram_charges_the_strided_predicates_extra_passes() {
        let pred = |pred, under| {
            Program::new(
                vec![MaskOp::Pred {
                    pred,
                    under,
                    dst: 0,
                }],
                Terminal::Count {
                    mask: Operand::Scratch(0),
                },
            )
            .op_histogram()
        };
        let gate = Some(Operand::Plane(0));
        let eq = Pred::EqU32Strided { lane: 0, v: 7 };
        let ne = Pred::NeU32Strided { lane: 0, v: 7 };
        let facet = Pred::MatchFacetStrided {
            lane: 0,
            pattern: [0; 12],
            care: [0xFF; 12],
        };
        // Silence twin: ungated Eq / Match spend no mask pass.
        assert_eq!(pred(eq, None).mask_passes(), 0);
        assert_eq!(pred(facet, None).mask_passes(), 0);
        // A gate is one `and`.
        assert_eq!(pred(eq, gate).two_input, 1);
        assert_eq!(pred(facet, gate).mask_passes(), 1);
        // `!=` is always one `not`, plus the gate's `and` when gated.
        assert_eq!(pred(ne, None).not, 1);
        assert_eq!(pred(ne, None).mask_passes(), 1);
        assert_eq!(pred(ne, gate).mask_passes(), 2);
        for p in [eq, ne, facet] {
            assert_eq!(pred(p, gate).predicates, 1, "{p:?} is still one predicate");
        }
    }
}
