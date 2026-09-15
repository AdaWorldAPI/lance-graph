//! A columnar query surface whose **operators are masking ops**.
//!
//! # What this is, and the shape it deliberately does NOT have
//!
//! DuckDB's operator set — scan, filter, project, aggregate, group — expressed
//! so that every operator LOWERS to a [`Program`] and is executed by the one
//! evaluator in `lance-graph-mask-risc`, which sits on `ndarray::simd`'s
//! masking algebra.
//!
//! What a columnar engine normally grows, and what is absent here on purpose:
//!
//! | the usual shape | why it is absent |
//! |---|---|
//! | an expression interpreter (`Expr` tree walked per batch) | a filter IS a `Pred`; walking a tree per batch is the second evaluator |
//! | a row iterator / `next()` volcano loop | the unit is a mask over `n_rows`, never a row |
//! | a per-operator kernel library | every operator is a `MaskOp` composition; a new operator is a new LOWERING, never a new kernel |
//! | a physical-plan `dyn Operator` chain | a plan is a `Program` — one flat op list, one terminal |
//! | a validity bitmap beside the data | the table's validity IS a resident mask plane ([`Filter::Plane`]); there is no separate NULL |
//! | a hash table for GROUP BY | a group is a mask; K groups are K gated equalities over the kept filter ([`lower_group_by`]) |
//!
//! The rule that keeps it honest: **this crate may build a [`Program`] and
//! must never evaluate one.** `execute` is called by the consumer, on a
//! scratch the consumer owns. A single `match` over operators here that
//! computed anything would be the duplicate evaluator the whole arc exists to
//! avoid.
//!
//! # The survivor skip — where a scan under a validity plane gets its cost model
//!
//! An `AND` whose children include a resident plane `g` (alpha, focus, a class
//! mask) is lowered so that every comparison beneath it is evaluated `under
//! g`: the predicate runs only over the 64-row words where `g` has a survivor
//! ([`MaskOp::Pred`]'s `under`). The rewrite is sound for ANY Boolean
//! remainder — `g ∧ rest(X₁..Xₙ) = g ∧ rest(g∧X₁, .., g∧Xₙ)`, since a row with
//! `g = 0` gives 0 on both sides and a row with `g = 1` leaves every leaf
//! unchanged — so gating passes through `NOT` and `OR` alike as long as `g`
//! itself stays a leaf. The plane leaf is DROPPED only when the gated
//! remainder is identically zero wherever `g` is zero: a gated comparison is;
//! an `AND` is if any child is; an `OR` is if every child is; a `NOT` never
//! is. `alpha & ((A & B) | C)` therefore costs three gated predicates and one
//! Boolean pass, with alpha never read as an operand at all.
//!
//! # Two lowerings, one meaning
//!
//! [`lower`] evaluates predicates IN PLACE and folds a junction's children
//! into its first child's slot — a filter of depth `d` costs `d + 1` slots,
//! width is free. [`lower_fused`] gives every predicate its own slot and hands
//! the Boolean skeleton to the fuser, which turns any subtree over three
//! leaves into one [`MaskOp::Ternlog`] — fewer mask passes, more slots. The
//! differential suite runs both against the same per-row oracle; a consumer
//! picks by whether it is scratch-bound or pass-bound.
//!
//! # Provenance — this is harvest-driven, not remembered
//!
//! Every operator below answers to a row of
//! `.claude/plans/duckdb-to-v3-translation-matrix-v1.md`, which reads DuckDB's
//! own source with `file:line` and rules each concept KEEP / ADAPT /
//! ELIMINATE / V3 BETTER / NEEDS FALSIFIER. The matrix is the specification;
//! this crate is one reading of it.
//!
//! | operator here | matrix row | verdict there |
//! |---|---|---|
//! | no `SelectionVector`, anywhere | R1 | ELIMINATE — an index list is the materialisation the mask-native invariant forbids |
//! | [`Col`] over a borrowed lane | R2 | KEEP — a flat vector IS an SoA lane |
//! | [`Cmp`] taking a scalar | R3 | ELIMINATE `ConstantVector` — a constant never acquires a representation |
//! | [`Filter::Plane`] | R6 | KEEP the representation, ELIMINATE the role — same packed `u64`, no NULL plane |
//! | [`Filter::prefix_u32`] / [`Filter::prefix_u64`] | R5 | ADAPT — *"the closest DuckDB comes to the V3 address"* |
//! | [`Query`] → [`Program`] | E1 | ADAPT — a state tree with per-node scratch becomes straight-line code over numbered slots |
//! | [`Agg`] as one terminal | E2 | ADAPT — Select-vs-Execute's two carriers collapse to one mask + one terminal tag |
//! | the six `i32` comparisons + two `u32` | E4 | ADAPT — DuckDB's 14-way physical-type switch narrows to what the lanes actually hold |
//! | `And` / `Or` | E5, E6 | ADAPT / **V3 BETTER** — DuckDB must SORT after OR to restore row order (`execute_conjunction.cpp:139`); a mask never lost it |
//! | [`Agg::BlendI32`] | E7 | ADAPT — CASE's narrowing false-set becomes a blend |
//! | [`Agg::Any`] / [`Agg::All`] | C7 | V3 BETTER — `HasNull`/`HasNotNull` are literally `mask_any`/`mask_all` |
//! | [`lower_group_by`] | A3 | V3 BETTER for the addressed case — a mask plane where the hash table would be |
//!
//! The harvest itself was re-run on 2026-09-14 and its first pass was
//! **repaired**, which is why the table above can cite what it cites. The
//! matrix's own §6 recorded the failure honestly: the `ruff_cpp_spo` harvest
//! had been pointed at 22 `.cpp` translation units, **seven of which came back
//! 100 % `Empty`**, because DuckDB's execution is template-dispatched and lives
//! in headers — so *"no row in this matrix cites a harvest TSV as evidence"*.
//! Pointing the same harvester at the headers (`scalar_executor.hpp`,
//! `comparison_operators.hpp`, `validity_mask.hpp`, `selection_vector.hpp`,
//! `ht_entry.hpp`, `vector.hpp`) yields **123 methods and 1,622 events** where
//! the `.cpp` pass yielded none.
//!
//! # What the header harvest changed here
//!
//! Two things, and both are in the code rather than only in this comment.
//!
//! **`Pred::MatchU64` was reachable from nothing.** It had been in the IR since
//! PR3, and this crate had no spelling for it, so a borrowed `LaneRef::U64` —
//! edge targets, ids, addresses — was queryable by no query. [`Cmp::MatchU64`]
//! closes that, and it is what the prefix operator over a 64-bit address needs.
//!
//! **The range primitive is real, and DuckDB's own bit-plane has it.** The
//! matrix files `mask_set_range` as T1 gap G6 on the strength of V3's
//! trie-reveal measurement. The header harvest shows the same operation on the
//! other side: `TemplatedValidityMask::SetRangeInvalid`, on the same packed-`u64`
//! carrier V3 uses. So [`Filter::prefix_u32`] lowers to a ternary-match SWEEP
//! today and says so plainly; the range WRITE waits on the primitive, per the
//! missing-capability STOP rule, rather than being hand-rolled one layer up.
//!
//! # What DuckDB has that this does not: the measurement loop
//!
//! `AdaptiveFilter` (matrix row A1) permutes a conjunction's terms at RUNTIME,
//! seeded from a static selectivity heuristic and then adapted by measured
//! RUNTIME. This crate cannot do that, and the reason is structural rather
//! than unfinished: it never executes, so there is nothing for it to measure.
//!
//! Order is still a real cost lever here — under the survivor skip a conjunct
//! whose survivors die in whole WORDS shrinks every later predicate's live
//! count — so [`Filter::and_by_skip`] takes the ordering decision as an INPUT.
//! The measurement that licensed even that much is
//! `examples/adaptive_order_probe.rs`; read [`Filter::and_by_skip`] for what it
//! does and does not establish.
//!
//! ⊘ This section previously called A1 "its only NEEDS-FALSIFIER of this kind"
//! and said "the caller currently owns it with no help". Both were true when
//! written and stopped being true in the same branch: A1 left that column and
//! `and_by_skip` is the help.
//!
//! # Status
//!
//! Filter (`=`/`<>`/`<`/`<=`/`>`/`>=`/ternary match/`IN`, `AND`/`OR`/`NOT`,
//! resident planes), the aggregates `COUNT`/`EXISTS`/`ALL`/`SUM`/`MIN`/`MAX`,
//! projection ([`Agg::Rows`] keeps the mask, [`Agg::BlendI32`] is the `CASE`
//! shape), and a two-phase `GROUP BY` over a categorical key. Absent, named:
//! the join (`src_mask → hop → dst_mask` is `lance-graph-mask-risc`'s PR5 —
//! there is no `hop` op to lower to yet); a one-terminal `GROUP BY SUM`
//! (`ndarray::simd` ships `masked_strided_group_sum`, but the IR names no
//! strided operand or group terminal, so K programs is the honest spelling
//! today); and everything the IR itself excludes — strings, `ORDER BY`,
//! three-valued NULL.

#![forbid(unsafe_code)]

use std::cmp::Reverse;

use lance_graph_mask_risc::{
    fuse, BoolExpr, FuseError, MaskOp, Operand, Pred, Program, Terminal, MAX_SCRATCH_SLOTS,
};

/// A column reference — an index into [`Planes::lanes`](lance_graph_mask_risc::Planes).
///
/// Not a name: name resolution is the catalogue's job and this crate has no
/// catalogue. A consumer that has names resolves them before it gets here.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Col(pub u16);

/// A resident mask plane — an index into
/// [`Planes::masks`](lance_graph_mask_risc::Planes): alpha, focus, a class
/// mask, a kept filter. The table's validity lives here, not in a NULL bit.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Mask(pub u16);

/// A predicate over one column — the filter's whole vocabulary.
///
/// One variant per masking op that produces a mask from a value lane. That is
/// not a coincidence and not a coding convenience: **the query language's
/// predicate set IS the masking algebra's predicate set**, so a predicate this
/// crate cannot spell is one the substrate cannot run, and adding one here
/// without adding it below would be the first crack.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Cmp {
    /// `col == v` over a signed lane.
    EqI32(i32),
    /// `col != v`.
    NeI32(i32),
    /// `col < v`.
    LtI32(i32),
    /// `col <= v`.
    LeI32(i32),
    /// `col > v`.
    GtI32(i32),
    /// `col >= v`.
    GeI32(i32),
    /// `col == v` over an unsigned lane, exact bitwise.
    EqU32(u32),
    /// `col != v`.
    NeU32(u32),
    /// `(col ^ pattern) & care == 0` — the ternary match, which SQL has no
    /// spelling for and the substrate has had all along.
    MatchU32 {
        /// The bits to compare.
        pattern: u32,
        /// Which bits participate; zero means "don't care".
        care: u32,
    },
    /// The ternary match over a 64-bit lane — edge targets, ids, addresses.
    ///
    /// `Pred::MatchU64` has been in the IR since PR3 and was unreachable from
    /// this crate, so a `LaneRef::U64` lane could be borrowed and never
    /// queried. That is the whole of the gap this closes; there is no new
    /// primitive underneath.
    MatchU64 {
        /// The bits to compare.
        pattern: u64,
        /// Which bits participate; zero means "don't care".
        care: u64,
    },
}

/// A filter expression: predicates over columns and resident planes, composed
/// with AND/OR/NOT.
///
/// Deliberately a tree HERE and never at run time — it is lowered once into a
/// flat [`Program`] and the tree is gone before anything executes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Filter {
    /// A leaf comparison on one column.
    Cmp(Col, Cmp),
    /// A resident mask plane read as a predicate — the scan's validity, a
    /// focus, a class mask. `Filter::Plane(alpha)` alone is `SELECT … FROM t`
    /// with no `WHERE`: the table is its validity plane, and a query over
    /// every row of it lowers to zero ops.
    Plane(Mask),
    /// Every child must hold.
    And(Vec<Filter>),
    /// Some child must hold.
    Or(Vec<Filter>),
    /// The child must not hold.
    Not(Box<Filter>),
}

impl Filter {
    /// `col <cmp>` — the leaf builder, so call sites read as the query does.
    pub fn cmp(col: Col, cmp: Cmp) -> Self {
        Filter::Cmp(col, cmp)
    }

    /// A resident plane as a predicate.
    pub fn plane(mask: Mask) -> Self {
        Filter::Plane(mask)
    }

    /// Conjunction.
    pub fn and(parts: impl IntoIterator<Item = Filter>) -> Self {
        Filter::And(parts.into_iter().collect())
    }

    /// Disjunction.
    pub fn or(parts: impl IntoIterator<Item = Filter>) -> Self {
        Filter::Or(parts.into_iter().collect())
    }

    /// Negation.
    ///
    /// Named `negate` rather than `not`: an inherent `not` shadows
    /// `std::ops::Not` and reads ambiguously at a call site that also uses the
    /// operator. Clippy's `should_implement_trait` is right here.
    pub fn negate(inner: Filter) -> Self {
        Filter::Not(Box::new(inner))
    }

    /// `col IN (set)` over an unsigned lane — a disjunction of equalities,
    /// which is exactly what it is; there is no IN-list kernel because none
    /// is needed. An empty set is `x IN ()`, refused at lowering for the same
    /// reason an empty `OR` is: it is that `OR`.
    pub fn in_u32(col: Col, set: impl IntoIterator<Item = u32>) -> Self {
        Filter::Or(
            set.into_iter()
                .map(|v| Filter::Cmp(col, Cmp::EqU32(v)))
                .collect(),
        )
    }

    /// `col IN (set)` over a signed lane. See [`Filter::in_u32`].
    pub fn in_i32(col: Col, set: impl IntoIterator<Item = i32>) -> Self {
        Filter::Or(
            set.into_iter()
                .map(|v| Filter::Cmp(col, Cmp::EqI32(v)))
                .collect(),
        )
    }

    /// An `AND` whose children are ordered by how much of the gate each one
    /// KILLS — the V3 answer to DuckDB's `AdaptiveFilter`, and deliberately
    /// not DuckDB's algorithm.
    ///
    /// `parts` is `(skip_score, child)`; children sort by DESCENDING score,
    /// stably, so equal scores keep the caller's order.
    ///
    /// # Why the score is dead WORDS and not selectivity
    ///
    /// DuckDB orders conjunction terms by a static selectivity heuristic and
    /// then adapts by measured RUNTIME (`adaptive_filter.cpp`: swap two
    /// neighbours, measure 10 iterations, keep if mean runtime dropped). It
    /// can rank that way because there term `k` runs only on the survivors of
    /// `1..k-1` — a selective term
    /// first literally shrinks the input. In V3 a predicate sweep costs the
    /// full column wherever it sits, so ordering can only pay by AVOIDANCE:
    /// the survivor skip drops a 64-row WORD when the gate has no survivor in
    /// it. The matrix (row A1) said so and required the measurement before any
    /// port. `examples/adaptive_order_probe.rs` is that measurement, over
    /// 65,536 rows, five conjuncts, all 120 orderings, four regimes:
    ///
    /// | regime | survivors | skipped, worst → best order |
    /// |---|---|---|
    /// | permissive | 94.3 % | **0.00 % → 0.00 %** |
    /// | moderate | 21.8 % | **0.00 % → 0.00 %** |
    /// | selective | 0.055 % | 5.66 % → **80.66 %** (14.2×) |
    /// | clustered (an address prefix) | 0.047 % | 0.00 % → **99.90 %** |
    ///
    /// Two findings, and a third that is a correction rather than a result.
    ///
    /// **Order does move the skip fraction, so A1 is not ELIMINATE** — but
    /// only where there is anything to skip. At 21.8 % survival with
    /// survivors SCATTERED, a 64-row word is all-dead with probability
    /// `0.78163^64 ≈ 1.4e-7`, so NO ordering skips anything and the whole
    /// question is moot. Word granularity needs the accumulator to die in
    /// whole words, not merely to be small — scattering is the condition,
    /// not density, and a dense-but-contiguous population has plenty of dead
    /// words.
    ///
    /// **Which means selectivity cannot tell you WHETHER reordering is worth
    /// anything.** The selective and the clustered regimes have almost
    /// identical survivor counts — 36 and 31 rows — and differ by 19
    /// percentage points of achievable skip (best against best: 80.66 % vs
    /// 99.90 %), because one conjunct's survivors are contiguous and the
    /// other's are scattered. A selectivity-only cost model cannot separate
    /// those two cases. That is also why V3 has this lever at all: an address
    /// prefix selects a contiguous subtree ([`Filter::prefix_u64`]), which is
    /// the clustered row of that table.
    ///
    /// Note what that does NOT say. An earlier version of this paragraph read
    /// *"rank by selectivity and those two look the same; rank by DEAD WORDS
    /// and they do not"* — a claim about the right SORT KEY. The probe does
    /// not support it: it enumerates all 120 permutations and reports
    /// min/max, never computing a selectivity-ranked order, never computing a
    /// dead-word-ranked order, never comparing two ranking rules, and never
    /// calling this function. Whether dead words is the better key to sort by
    /// is **untested**; what is measured is the between-regime diagnostic
    /// above.
    ///
    /// # Why the score is the CALLER's, and why there is no hill-climb
    ///
    /// The shipped surface of this crate builds programs and never evaluates
    /// one, so there is no point at which it could measure a score; it comes
    /// from a previous execution the caller ran.
    ///
    /// DuckDB's adjacent-transposition hill-climb is not ported — **and the
    /// reason first given here was measured FALSE, so it is worth stating
    /// correctly.** The claim was that "the quantity being optimised is a step
    /// function of clustering … so a local search over adjacent swaps is
    /// exploring the wrong landscape". Instrumenting the probe's own
    /// `skipped_words` model with the clustered regime's prefix term at each
    /// index gives `[4092, 3069, 2046, 1023, 0]` — adjacent deltas all
    /// exactly `-1023`, a monotone linear ramp, because the prefix term's
    /// mask is one live word of 1024 and each gated position past it skips
    /// the other 1023. Every forward adjacent swap improves it by the same
    /// amount. That is the friendliest possible hill-climb landscape, not the
    /// wrong one.
    ///
    /// The step-like behaviour is BETWEEN regimes (does this conjunction
    /// contain a clustered term at all); the search space is WITHIN one
    /// permutation set. The original argument reasoned from the first to the
    /// second.
    ///
    /// The real reason is narrower: this crate never executes, so there is no
    /// runtime for a hill-climb to measure. Whether a consumer that DOES
    /// execute should run one is unanswered here — and DuckDB's loop adapts
    /// on measured RUNTIME, seeded from a static selectivity heuristic, not
    /// on measured selectivity, so the comparison would have to be against
    /// that.
    ///
    /// # This lever is INERT on a conjunction that carries a resident plane
    ///
    /// Measured, and it is the price of the `emit_gated` fix in `b7e6cef`:
    /// once an `AND` carries a plane, every `Pred` inside it gates on that
    /// same FIXED plane regardless of position, so the skipped-word count is
    /// order-independent and this ordering buys exactly zero. Plane-free
    /// conjunctions are unaffected — their preds still chain on the running
    /// accumulator, which is where `adaptive_order_probe` measured the lever
    /// and the only place it claims one. Since the crate's own headline shape
    /// (`Filter::Plane(alpha)` = `SELECT … FROM t`) IS planed, that is most
    /// real queries. Recorded as `ISS-QUACK-AND-BY-SKIP-IS-INERT-UNDER-A-PLANE`
    /// with the measurement and the way out.
    ///
    /// # One caveat, because it is a real override
    ///
    /// When this `AND` also carries a resident plane that the gate walk
    /// DROPS as implied, a child whose result is a subset of that plane is
    /// rotated to the front regardless of score, so the ordering here applies
    /// among the children the rotation leaves alone.
    ///
    /// ⊘ That rotation was called "a correctness requirement, not a
    /// preference" until 2026-09-15. It WAS one before the `emit_gated` fix;
    /// afterwards the plane gates every child wherever it sits, so what the
    /// rotation still buys is SLOT ECONOMY (a plane first makes the
    /// accumulator a plane operand, costing no scratch slot: 1 vs 2 on
    /// `alpha AND focus AND v < 50`). Disable-verified — rotation removed,
    /// 120,000 differential cases, 65,919 of them planed, zero divergences, on
    /// a harness proven able to see this class of regression by restoring the
    /// pre-fix gate line and watching it fail 2,668 of the same cases.
    pub fn and_by_skip(parts: impl IntoIterator<Item = (u32, Filter)>) -> Self {
        let mut scored: Vec<(u32, Filter)> = parts.into_iter().collect();
        // `sort_by_key` is stable, so equal scores keep the caller's order
        // rather than being permuted by an implementation detail. `Reverse`
        // rather than a reversed comparator: a stable sort over a reversed KEY
        // keeps ties in caller order, while reversing the COMPARISON of a
        // stable sort would too — but clippy rejects the latter spelling, and
        // the two are only equivalent because the key is `Copy`.
        scored.sort_by_key(|&(score, _)| Reverse(score));
        Filter::And(scored.into_iter().map(|(_, f)| f).collect())
    }

    /// Rows whose 32-bit address lane starts with the top `bits` of `prefix` —
    /// the operator SQL has no name for and the V3 address was built around.
    ///
    /// # Why this is not just another equality
    ///
    /// On an address-ordered lane a prefix names a CONTIGUOUS RANGE: every row
    /// under one trie node is `2^(32 - bits)` consecutive addresses. DuckDB
    /// arrives at the same shape and then discards it — `SequenceVector`
    /// compresses a range to three scalars (`vector.cpp:498-500`) and
    /// `ToUnifiedFormat` flattens it to N materialised values before any kernel
    /// runs (`vector.cpp:461-465`), after which `DataChunk::Slice` re-manufactures
    /// the range as a per-row index loop (`data_chunk.cpp:394-397`). That round
    /// trip is the matrix's R5/R8, and it is what "better than faithful" means
    /// here: the range is kept, not rebuilt.
    ///
    /// # What this actually lowers to today, stated exactly
    ///
    /// A ternary match — a full sweep of the lane, one pass, no allocation. It
    /// is NOT yet a range WRITE. The range write is `mask_set_range`, the
    /// matrix's T1 gap G6, and it is absent from `ndarray::simd`; per the
    /// missing-capability STOP rule a consumer does not hand-roll it one layer
    /// up, so this crate spells the PREDICATE and waits for the primitive.
    ///
    /// That the primitive is real rather than wished for is the harvest's
    /// evidence, not this crate's opinion: DuckDB's own bit-plane carries
    /// `TemplatedValidityMask::SetRangeInvalid` (`validity_mask.hpp`, harvested
    /// 2026-09-14 from the headers) — the same packed-`u64` representation V3
    /// uses, with the range operation already on it.
    ///
    /// `bits` is clamped to 32; `bits == 0` matches every row (care is empty),
    /// which is the honest reading of "no significant bits" rather than an
    /// error, and `bits == 32` matches exactly one address.
    pub fn prefix_u32(col: Col, prefix: u32, bits: u32) -> Self {
        let care = match bits.min(32) {
            0 => 0,
            b => u32::MAX << (32 - b),
        };
        Filter::Cmp(
            col,
            Cmp::MatchU32 {
                pattern: prefix & care,
                care,
            },
        )
    }

    /// [`Filter::prefix_u32`] over a 64-bit address lane.
    ///
    /// This is the one that reaches the canonical GUID's own prefix: classid,
    /// then HEEL/HIP/TWIG, each a nibble-addressed tier of the cascade. A
    /// `bits` that lands on a tier boundary selects exactly that subtree.
    pub fn prefix_u64(col: Col, prefix: u64, bits: u32) -> Self {
        let care = match bits.min(64) {
            0 => 0,
            b => u64::MAX << (64 - b),
        };
        Filter::Cmp(
            col,
            Cmp::MatchU64 {
                pattern: prefix & care,
                care,
            },
        )
    }
}

/// Why a query could not be lowered.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum LowerError {
    /// An `And`/`Or` with no children. Refused rather than folded to a
    /// constant: an empty conjunction is `true` and an empty disjunction is
    /// `false`, and a caller that built one by accident wants to hear about it
    /// rather than receive whichever identity this crate happened to pick.
    EmptyJunction,
    /// The program would need more scratch slots than a `u16` can name.
    TooManySlots {
        /// The count that overflowed.
        needed: usize,
    },
    /// A `GROUP BY` asked for [`Agg::BlendI32`]. Every group program writes
    /// the WHOLE `out` slice, so K groups would leave the last group's blend
    /// and silently discard K − 1 — refused rather than answered wrongly.
    GroupedBlend,
}

impl core::fmt::Display for LowerError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            LowerError::EmptyJunction => {
                write!(f, "an AND/OR with no children has no non-arbitrary meaning")
            }
            LowerError::TooManySlots { needed } => {
                write!(f, "needs {needed} scratch slots; the address space is u16")
            }
            LowerError::GroupedBlend => {
                write!(f, "a blend writes the whole output; it cannot be grouped")
            }
        }
    }
}

/// What a query asks for.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Agg {
    /// How many rows survive the filter.
    Count,
    /// Whether any row survives.
    Any,
    /// Whether every row survives.
    All,
    /// Σ over a signed lane, restricted to surviving rows.
    SumI32(Col),
    /// min over surviving rows.
    MinI32(Col),
    /// max over surviving rows.
    MaxI32(Col),
    /// The surviving rows themselves — projection. Nothing is reduced and
    /// nothing is copied: the result is the final mask
    /// ([`Terminal::Keep`]), and the projected columns are the resident
    /// lanes the caller already holds. The one materialiser is
    /// `lance_graph_mask_risc::materialize_rows`, and it is the caller's
    /// to invoke.
    Rows,
    /// `CASE WHEN filter THEN then ELSE els END` — written per row into a
    /// caller-supplied `out` slice, no compaction ([`Terminal::BlendI32`]).
    BlendI32 {
        /// The lane read where the filter holds.
        then: Col,
        /// The lane read where it does not.
        els: Col,
    },
}

/// One query: a filter and what to ask of the rows that pass it.
///
/// The filter is required. There is no unfiltered table in the substrate —
/// every table IS its validity plane — so "every row" is spelled
/// [`Filter::Plane`]`(alpha)`, which lowers to zero ops and reads the plane
/// straight from the terminal. (A first draft of this crate tried to spell it
/// without a plane, as a predicate over lane 0 followed by a constant
/// ternlog; that typechecked only when lane 0 happened to be `U32` — a latent
/// lane-kind bug. The plane leaf is the spelling that is correct by
/// construction.)
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Query {
    /// The filter.
    pub filter: Filter,
    /// The aggregate.
    pub agg: Agg,
}

/// `GROUP BY key` over a low-cardinality unsigned key lane whose values are
/// `0..groups` — the dictionary-encoded / categorical shape.
///
/// The filter is required for the same reason [`Query`]'s is; the whole
/// table is `Filter::Plane(alpha)`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GroupBy {
    /// Rows admitted to any group.
    pub filter: Filter,
    /// The key lane (`u32`).
    pub key: Col,
    /// Number of key values; group `g` is the rows whose key equals `g`.
    pub groups: u32,
    /// The aggregate computed per group. [`Agg::BlendI32`] is refused.
    pub agg: Agg,
}

/// The two-phase plan a [`GroupBy`] lowers to — DuckDB's pipeline break,
/// with a mask plane where the hash table would be.
///
/// The caller runs `filter` (a [`Terminal::Keep`]), reads the mask its
/// `Value::Mask(op)` names — a scratch slot, or for a bare-plane filter the
/// plane itself — presents it as `planes.masks[filter_plane]`, and runs each
/// program in `groups` over the widened planes. Each group program is ONE
/// gated equality and a terminal: the key lane is compared once per group,
/// but only over the live words of the kept filter (the survivor skip), which
/// is the bitmap-index cost model rather than the hash-table one.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GroupPlan {
    /// Phase 1: the filter, kept.
    pub filter: Program,
    /// The plane index phase 2 reads the kept filter at.
    pub filter_plane: u16,
    /// Phase 2: one program per key value, in key order.
    pub groups: Vec<Program>,
}

/// Lower a query to a [`Program`], evaluating predicates in place.
///
/// # The allocation rule
///
/// Slots are assigned by a strict post-order walk, and a junction's children
/// are folded left-to-right into the FIRST child's slot. So a filter of depth
/// `d` costs `d + 1` slots, never one per leaf — which is what keeps a wide
/// conjunction from asking for a scratch arena proportional to its width. A
/// resident plane costs nothing: it is read as an operand where it stands.
///
/// # Errors
///
/// [`LowerError::EmptyJunction`] for an `And`/`Or` with no children;
/// [`LowerError::TooManySlots`] if the walk needs more than `u16::MAX + 1`.
pub fn lower(q: &Query) -> Result<Program, LowerError> {
    lower_with(&q.filter, q.agg, |node, ops| emit_inplace(node, 0, ops))
}

/// Lower a query to a [`Program`] whose Boolean skeleton is fused into
/// ternlogs.
///
/// Every comparison gets its own slot, then the fuser reduces the skeleton —
/// any subtree over three leaves is one [`MaskOp::Ternlog`]. Fewer mask passes
/// than [`lower`]; more slots, proportional to the number of comparisons.
///
/// # Errors
///
/// As [`lower`].
pub fn lower_fused(q: &Query) -> Result<Program, LowerError> {
    lower_with(&q.filter, q.agg, emit_fused)
}

/// Lower a `GROUP BY` to its two-phase [`GroupPlan`]; `filter_plane` is the
/// plane index the caller will bind the kept filter at.
///
/// # Errors
///
/// As [`lower`], plus [`LowerError::GroupedBlend`].
pub fn lower_group_by(g: &GroupBy, filter_plane: u16) -> Result<GroupPlan, LowerError> {
    if matches!(g.agg, Agg::BlendI32 { .. }) {
        return Err(LowerError::GroupedBlend);
    }
    let filter = lower_with(&g.filter, Agg::Rows, |node, ops| emit_inplace(node, 0, ops))?;
    let groups = (0..g.groups)
        .map(|v| {
            Program::new(
                vec![MaskOp::Pred {
                    pred: Pred::EqU32 { lane: g.key.0, v },
                    under: Some(Operand::Plane(filter_plane)),
                    dst: 0,
                }],
                terminal_of(g.agg, Operand::Scratch(0)),
            )
        })
        .collect();
    Ok(GroupPlan {
        filter,
        filter_plane,
        groups,
    })
}

/// The shared front half of every lowering: gate the tree, emit it, read the
/// aggregate off wherever the result landed.
fn lower_with(
    filter: &Filter,
    agg: Agg,
    emit: impl FnOnce(&Node, &mut Vec<MaskOp>) -> Result<Operand, LowerError>,
) -> Result<Program, LowerError> {
    let (node, _) = gate_walk(filter, None)?;
    let mut ops = Vec::new();
    let mask = emit(&node, &mut ops)?;
    Ok(Program::new(ops, terminal_of(agg, mask)))
}

/// The filter after the survivor-skip pass, before any slot is assigned.
enum Node {
    /// A comparison, evaluated under `under` when there is a gate.
    Pred {
        pred: Pred,
        under: Option<Mask>,
    },
    /// A resident plane read as an operand.
    Plane(Mask),
    And(Vec<Node>),
    Or(Vec<Node>),
    Not(Box<Node>),
}

/// Rotate a child whose result is a SUBSET of the gate to the front, and say
/// whether one was found.
///
/// `flags[i]` is `gate_walk`'s "vanishes with the gate", which is exactly
/// "this child's result is a subset of the gate". Putting such a child first
/// makes the running accumulator a subset of the gate from the first fold
/// onward, which keeps the accumulator a plane operand and costs no scratch
/// slot.
///
/// ⊘ This read "that is the precondition [`emit_gated`] needs before it may
/// narrow later comparisons onto the accumulator instead of onto the gate".
/// Since `b7e6cef` `emit_gated` does not narrow onto the accumulator at all
/// while a plane gate is live — the plane always wins — so the rotation is an
/// optimisation, not a precondition. See the disable run recorded on
/// [`Query::and_by_skip`].
///
/// Shared by BOTH `AND` arms of [`gate_walk`] — the already-gated one and the
/// one that establishes a gate — because a first version rotated only in the
/// second, and a gated `AND` nested inside a gated `AND` reproduced the same
/// wrong answer one level down. Two spellings of one rule is how that happens
/// twice.
fn hoist_gate_subset(nodes: &mut [Node], flags: &mut [bool]) -> bool {
    match flags.iter().position(|&v| v) {
        Some(i) => {
            nodes.swap(0, i);
            flags.swap(0, i);
            true
        }
        None => false,
    }
}

/// Walk `f` under `gate`, establishing at most one gate per `AND` path.
///
/// Returns the node and whether it is identically zero wherever the gate is
/// zero — the condition under which an `AND` may drop its gate plane as a
/// leaf (see the crate doc's soundness argument).
fn gate_walk(f: &Filter, gate: Option<Mask>) -> Result<(Node, bool), LowerError> {
    Ok(match f {
        Filter::Cmp(col, cmp) => (
            Node::Pred {
                pred: pred_of(*col, *cmp),
                under: gate,
            },
            gate.is_some(),
        ),
        Filter::Plane(m) => (Node::Plane(*m), gate == Some(*m)),
        // Gating passes THROUGH a negation (the rewrite is sound for any
        // remainder while the gate stays a leaf), but `!x` is 1 where the
        // gate is 0, so a negation never lets the gate be dropped.
        Filter::Not(inner) => (Node::Not(Box::new(gate_walk(inner, gate)?.0)), false),
        Filter::Or(parts) => {
            let (nodes, flags) = walk_all(parts, gate)?;
            (Node::Or(nodes), flags.iter().all(|&v| v))
        }
        Filter::And(parts) => {
            if let Some(g) = gate {
                // Already gated from above: this AND's own planes are plain
                // leaves; one gate per comparison is all `under` can carry.
                let (mut nodes, mut flags) = walk_all(parts, Some(g))?;
                let subset = hoist_gate_subset(&mut nodes, &mut flags);
                (Node::And(nodes), subset)
            } else {
                let found = parts.iter().find_map(|p| match p {
                    Filter::Plane(m) => Some(*m),
                    _ => None,
                });
                let Some(g) = found else {
                    let (nodes, _) = walk_all(parts, None)?;
                    return Ok((Node::And(nodes), false));
                };
                let mut nodes = Vec::with_capacity(parts.len());
                let mut flags = Vec::with_capacity(parts.len());
                let mut gate_taken = false;
                for p in parts {
                    if !gate_taken && matches!(p, Filter::Plane(m) if *m == g) {
                        gate_taken = true;
                        continue;
                    }
                    let (n, v) = gate_walk(p, Some(g))?;
                    nodes.push(n);
                    flags.push(v);
                }
                // The gate is implied by any child that vanishes with it;
                // otherwise it must be read as a leaf of its own.
                //
                // When it IS implied, the vanishing child is rotated to the
                // FRONT, and that is load-bearing rather than tidy. `vanishes`
                // means exactly "this child's result is a subset of the gate",
                // so putting one first makes the running accumulator a subset
                // of the gate from the first fold onward — which is what lets
                // [`emit_gated`] narrow later comparisons onto the accumulator
                // without losing the gate.
                //
                // ⊘ This said "Without the rotation this is a silent WRONG
                // ANSWER, not a missed optimisation". TRUE BEFORE `b7e6cef`,
                // FALSE AFTER: the plane now gates every child wherever it
                // sits, so the `AND` emits `⋂ children` where each flagged
                // child is `plane & pred ⊆ plane`, and `⋂ children ⊆ plane`
                // holds for ANY ordering. What survives is slot economy.
                // The original measurement, kept because it is what the
                // rotation was built from:
                // `alpha AND focus AND v < 50`, the foreign plane `focus` sat
                // first, the accumulator therefore started as `focus` (which is
                // NOT a subset of `alpha`), the comparison was gated on that
                // instead of on `alpha`, and `alpha` — already dropped as
                // "implied" — was nowhere in the program. The per-row oracle
                // caught it immediately.
                //
                // AND is commutative, so the rotation costs nothing
                // semantically. It is also the first place in this crate where
                // term ORDER changes the emitted program, which is the
                // mechanism row A1 is about.
                if !hoist_gate_subset(&mut nodes, &mut flags) {
                    nodes.insert(0, Node::Plane(g));
                }
                (Node::And(nodes), false)
            }
        }
    })
}

/// [`gate_walk`] over every child of a junction, refusing an empty one.
///
/// One of THREE spellings of that refusal — the others are the
/// `acc.ok_or(EmptyJunction)` in [`emit_inplace`] and in [`assign_slots`].
/// Measured: disabling any ONE leaves the suite green, because the other two
/// still catch it; disabling all three turns
/// `an_empty_junction_is_refused_rather_than_folded_to_an_identity` red. So
/// the refusal is load-bearing and the redundancy is deliberate — each
/// spelling guards a different stage (the gate walk, the in-place emitter,
/// the fused emitter) and none may be removed as "obviously dead" on the
/// strength of its own disable run coming back green.
///
/// The early one is not merely belt-and-braces: without it an empty `Or`
/// reaches `flags.iter().all(..)` over an EMPTY vector, which is vacuously
/// `true`, and reports that it vanishes with the gate — letting a parent
/// `AND` drop a gate plane it should have kept. The program is refused
/// downstream either way, so nothing observable changes today; it is a
/// vacuous-truth corner not worth leaving open.
fn walk_all(parts: &[Filter], gate: Option<Mask>) -> Result<(Vec<Node>, Vec<bool>), LowerError> {
    if parts.is_empty() {
        return Err(LowerError::EmptyJunction);
    }
    let mut nodes = Vec::with_capacity(parts.len());
    let mut flags = Vec::with_capacity(parts.len());
    for p in parts {
        let (n, v) = gate_walk(p, gate)?;
        nodes.push(n);
        flags.push(v);
    }
    Ok((nodes, flags))
}

/// Emit `n` with its result in `dst` (or, for a plane, where it already is),
/// folding junction children into the first child's slot.
fn emit_inplace(n: &Node, dst: u16, ops: &mut Vec<MaskOp>) -> Result<Operand, LowerError> {
    emit_gated(n, dst, None, ops)
}

/// [`emit_inplace`] with a RUNNING gate: an operand every comparison beneath
/// `n` is evaluated `under`.
///
/// # The accumulator gate, and why it is the one that makes order matter
///
/// The plane gate ([`gate_walk`]) skips words where a RESIDENT mask is empty.
/// This is the other one: inside an `AND`, once the first `k` conjuncts have
/// been folded into `dst`, conjunct `k + 1` is only consulted where that
/// partial result still has a survivor. So the gate NARROWS as the conjunction
/// proceeds, and a selective term early shrinks the live-word count of every
/// term after it.
///
/// Soundness is the same identity the plane gate uses — `g ∧ rest(X) =
/// g ∧ rest(g ∧ X)` — so it passes through `OR` and `NOT` alike. What it may
/// NOT do is let anything be dropped: the accumulator is a real operand of the
/// `AND`, never elided, so there is no analog here of the plane's
/// vanishes-with-the-gate rule.
///
/// It applies to `AND` only. Under an `OR`, `acc | p` depends on `p` exactly
/// where `acc` is ZERO — gating there would discard the bits that matter and
/// quietly answer `acc`. That asymmetry is the same one `lgj-abi`'s
/// `plan_lower` documents, and it is the correctness question in both.
fn emit_gated(
    n: &Node,
    dst: u16,
    acc_gate: Option<Operand>,
    ops: &mut Vec<MaskOp>,
) -> Result<Operand, LowerError> {
    match n {
        Node::Pred { pred, under } => {
            // A `Pred` carries exactly ONE `under`, so when both a plane gate
            // and an accumulator are available this is a choice, not a union
            // — and the choice is the PLANE, always.
            //
            // An earlier version preferred the accumulator on the reasoning
            // that it is strictly narrower, "because the plane was the AND's
            // first conjunct, so it is already folded into the accumulator".
            // That holds only inside the AND that ESTABLISHED the plane,
            // where [`hoist_gate_subset`] arranges it. It is false the moment
            // an OUTER conjunction with no plane of its own wraps an inner one
            // that has:
            //
            // ```text
            // P1 AND (Plane(focus) AND P2)
            // ```
            //
            // The outer AND finds no plane among its own parts, so it
            // establishes no gate and simply folds; the inner AND establishes
            // `focus`, gates P2 under it, sees P2 vanish and DROPS the plane
            // as implied. Then the outer emitter hands its accumulator `P1`
            // down, this line preferred it over `focus`, and `focus` — already
            // elided — appeared nowhere in the program. Measured on a 512-row
            // fixture: the oracle selects 29 rows, the emitted program
            // selected 204, and the op list was `GtI32 AND NeU32` with no
            // trace of the plane. A silent wrong answer, not a slow one.
            // Pinned by `a_nested_plane_survives_an_outer_accumulator`.
            //
            // Preferring the plane is sound unconditionally: the emitted mask
            // is `plane & pred`, and the enclosing junction intersects the
            // accumulator afterwards anyway, so `acc & (plane & pred)` is
            // exactly the wanted value. What it costs is the EXTRA narrowing
            // the accumulator would have given inside a plane's own AND — the
            // plane still gates there, just less tightly than it could. That
            // cost is named in `ISSUES.md` rather than traded against a
            // correctness hole, and A1's lever is untouched because a
            // conjunction of plain comparisons carries no plane at all: those
            // preds take the `None` arm below and gate on the accumulator.
            let gate = under.map(|m| Operand::Plane(m.0)).or(acc_gate);
            ops.push(MaskOp::Pred {
                pred: *pred,
                under: gate,
                dst,
            });
            Ok(Operand::Scratch(dst))
        }
        Node::Plane(m) => Ok(Operand::Plane(m.0)),
        Node::Not(inner) => {
            let a = emit_gated(inner, dst, acc_gate, ops)?;
            ops.push(MaskOp::Not { a, dst });
            Ok(Operand::Scratch(dst))
        }
        Node::And(parts) | Node::Or(parts) => {
            let is_and = matches!(n, Node::And(_));
            let mut acc: Option<Operand> = None;
            for part in parts {
                // The first child lands in `dst`; every sibling after it
                // borrows the next slot up in turn, so width costs one slot,
                // not one per child.
                let slot = if acc.is_none() {
                    dst
                } else {
                    dst.checked_add(1).ok_or(LowerError::TooManySlots {
                        needed: usize::from(dst) + 2,
                    })?
                };
                // Inside an AND, every child after the first is gated on the
                // partial result. Inside an OR the inherited gate is passed
                // through unchanged — an OR may not gate on its own
                // accumulator, but it is still inside whatever AND encloses
                // it, and that gate remains sound.
                let child_gate = match (is_and, acc) {
                    (true, Some(a)) => Some(a),
                    _ => acc_gate,
                };
                let b = emit_gated(part, slot, child_gate, ops)?;
                acc = Some(match acc {
                    None => b,
                    Some(a) => {
                        ops.push(if is_and {
                            MaskOp::And { a, b, dst }
                        } else {
                            MaskOp::Or { a, b, dst }
                        });
                        Operand::Scratch(dst)
                    }
                });
            }
            acc.ok_or(LowerError::EmptyJunction)
        }
    }
}

/// Emit `n` with every comparison in its own slot and the skeleton fused.
fn emit_fused(n: &Node, ops: &mut Vec<MaskOp>) -> Result<Operand, LowerError> {
    let expr = assign_slots(n, ops)?;
    let first_free = u16::try_from(ops.len()).map_err(|_| LowerError::TooManySlots {
        needed: ops.len() + 1,
    })?;
    match fuse(&expr, first_free) {
        Ok(fused) => {
            ops.extend(fused.ops);
            Ok(fused.result)
        }
        Err(FuseError::SlotOverflow) => Err(LowerError::TooManySlots {
            needed: MAX_SCRATCH_SLOTS as usize + 1,
        }),
    }
}

/// Give each comparison the next slot and mirror the skeleton as a
/// [`BoolExpr`] over the resulting operands (n-ary junctions left-folded).
fn assign_slots(n: &Node, ops: &mut Vec<MaskOp>) -> Result<BoolExpr, LowerError> {
    match n {
        Node::Pred { pred, under } => {
            let dst = u16::try_from(ops.len()).map_err(|_| LowerError::TooManySlots {
                needed: ops.len() + 1,
            })?;
            ops.push(MaskOp::Pred {
                pred: *pred,
                under: under.map(|m| Operand::Plane(m.0)),
                dst,
            });
            Ok(BoolExpr::Leaf(Operand::Scratch(dst)))
        }
        Node::Plane(m) => Ok(BoolExpr::Leaf(Operand::Plane(m.0))),
        Node::Not(inner) => Ok(BoolExpr::Not(Box::new(assign_slots(inner, ops)?))),
        Node::And(parts) | Node::Or(parts) => {
            let is_and = matches!(n, Node::And(_));
            let mut acc: Option<BoolExpr> = None;
            for part in parts {
                let e = assign_slots(part, ops)?;
                acc = Some(match acc {
                    None => e,
                    Some(a) => {
                        if is_and {
                            BoolExpr::And(Box::new(a), Box::new(e))
                        } else {
                            BoolExpr::Or(Box::new(a), Box::new(e))
                        }
                    }
                });
            }
            acc.ok_or(LowerError::EmptyJunction)
        }
    }
}

/// `agg` read over `mask`.
fn terminal_of(agg: Agg, mask: Operand) -> Terminal {
    match agg {
        Agg::Count => Terminal::Count { mask },
        Agg::Any => Terminal::Any { mask },
        Agg::All => Terminal::All { mask },
        Agg::SumI32(c) => Terminal::MaskedSumI32 { mask, lane: c.0 },
        Agg::MinI32(c) => Terminal::MaskedMinI32 { mask, lane: c.0 },
        Agg::MaxI32(c) => Terminal::MaskedMaxI32 { mask, lane: c.0 },
        Agg::Rows => Terminal::Keep { mask },
        Agg::BlendI32 { then, els } => Terminal::BlendI32 {
            mask,
            then: then.0,
            els: els.0,
        },
    }
}

/// One comparison → one `Pred`. Exhaustive, and the compiler keeps it so.
fn pred_of(col: Col, cmp: Cmp) -> Pred {
    let lane = col.0;
    match cmp {
        Cmp::EqI32(v) => Pred::EqI32 { lane, v },
        Cmp::NeI32(v) => Pred::NeI32 { lane, v },
        Cmp::LtI32(t) => Pred::LtI32 { lane, t },
        Cmp::LeI32(t) => Pred::LeI32 { lane, t },
        Cmp::GtI32(t) => Pred::GtI32 { lane, t },
        Cmp::GeI32(t) => Pred::GeI32 { lane, t },
        Cmp::EqU32(v) => Pred::EqU32 { lane, v },
        Cmp::NeU32(v) => Pred::NeU32 { lane, v },
        Cmp::MatchU32 { pattern, care } => Pred::MatchU32 {
            lane,
            pattern,
            care,
        },
        Cmp::MatchU64 { pattern, care } => Pred::MatchU64 {
            lane,
            pattern,
            care,
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use lance_graph_mask_risc::{
        execute, materialize_rows, reference_execute, scratch_words_for, words_for, LaneRef,
        Planes, Scratch, Value,
    };

    const N: usize = 1000;

    const VALS: Col = Col(0);
    const CLASS: Col = Col(1);
    const ALT: Col = Col(2);
    const ADDR: Col = Col(3);
    const ALPHA: Mask = Mask(0);
    const FOCUS: Mask = Mask(1);

    /// Three lanes and two resident planes, every tail bit clean.
    struct Fx {
        vals: Vec<i32>,
        classes: Vec<u32>,
        alt: Vec<i32>,
        /// An ADDRESS-ORDERED lane: row `i` holds address `i << 8`, so the
        /// top bits are a trie prefix and a prefix predicate must select a
        /// contiguous row range. That ordering is the fixture's whole point —
        /// on an unordered lane a prefix is just an equality with holes, and
        /// the range claim would be untestable.
        addr: Vec<u64>,
        masks: Vec<Vec<u64>>,
    }

    fn plane(n: usize, set: impl Fn(usize) -> bool) -> Vec<u64> {
        let mut words = vec![0u64; words_for(n)];
        for r in (0..n).filter(|&r| set(r)) {
            words[r / 64] |= 1u64 << (r % 64);
        }
        words
    }

    impl Fx {
        fn new(n: usize) -> Self {
            let vals = (0..n)
                .map(|i| ((i as i64 * 7) % 401 - 200) as i32)
                .collect();
            let classes = (0..n).map(|i| (i % 5) as u32).collect();
            let alt = (0..n).map(|i| ((i as i64 * 13) % 89 - 44) as i32).collect();
            let addr = (0..n).map(|i| (i as u64) << 8).collect();
            let masks = vec![plane(n, |r| r % 3 != 0), plane(n, |r| r % 7 == 0)];
            Fx {
                vals,
                classes,
                alt,
                addr,
                masks,
            }
        }

        fn n(&self) -> usize {
            self.vals.len()
        }

        fn i32_at(&self, col: Col, row: usize) -> i32 {
            match col {
                VALS => self.vals[row],
                ALT => self.alt[row],
                other => panic!("{other:?} is not a signed lane of the fixture"),
            }
        }

        fn u32_at(&self, col: Col, row: usize) -> u32 {
            match col {
                CLASS => self.classes[row],
                other => panic!("{other:?} is not an unsigned lane of the fixture"),
            }
        }

        fn u64_at(&self, col: Col, row: usize) -> u64 {
            match col {
                ADDR => self.addr[row],
                other => panic!("{other:?} is not a 64-bit lane of the fixture"),
            }
        }

        fn bit(&self, m: Mask, row: usize) -> bool {
            (self.masks[usize::from(m.0)][row / 64] >> (row % 64)) & 1 == 1
        }

        /// The INDEPENDENT reading — a plain row loop that shares no code
        /// with the lowering. This is the oracle: a filter tree walked per
        /// row, which is exactly the shape this crate refuses to ship,
        /// written here because a test oracle is the one place it is
        /// licensed.
        fn oracle(&self, f: &Filter, row: usize) -> bool {
            match f {
                Filter::Cmp(col, cmp) => match *cmp {
                    Cmp::EqI32(x) => self.i32_at(*col, row) == x,
                    Cmp::NeI32(x) => self.i32_at(*col, row) != x,
                    Cmp::LtI32(x) => self.i32_at(*col, row) < x,
                    Cmp::LeI32(x) => self.i32_at(*col, row) <= x,
                    Cmp::GtI32(x) => self.i32_at(*col, row) > x,
                    Cmp::GeI32(x) => self.i32_at(*col, row) >= x,
                    Cmp::EqU32(x) => self.u32_at(*col, row) == x,
                    Cmp::NeU32(x) => self.u32_at(*col, row) != x,
                    Cmp::MatchU32 { pattern, care } => {
                        (self.u32_at(*col, row) ^ pattern) & care == 0
                    }
                    Cmp::MatchU64 { pattern, care } => {
                        (self.u64_at(*col, row) ^ pattern) & care == 0
                    }
                },
                Filter::Plane(m) => self.bit(*m, row),
                Filter::And(ps) => ps.iter().all(|p| self.oracle(p, row)),
                Filter::Or(ps) => ps.iter().any(|p| self.oracle(p, row)),
                Filter::Not(p) => !self.oracle(p, row),
            }
        }

        fn rows(&self, f: &Filter) -> Vec<usize> {
            (0..self.n()).filter(|&r| self.oracle(f, r)).collect()
        }

        fn with_planes<R>(&self, extra: &[Vec<u64>], f: impl FnOnce(&Planes<'_>) -> R) -> R {
            let lanes = [
                LaneRef::I32(&self.vals),
                LaneRef::U32(&self.classes),
                LaneRef::I32(&self.alt),
                LaneRef::U64(&self.addr),
            ];
            let masks: Vec<&[u64]> = self.masks.iter().chain(extra).map(Vec::as_slice).collect();
            f(&Planes {
                n_rows: self.n(),
                masks: &masks,
                lanes: &lanes,
            })
        }

        /// The executor, on a scratch sized exactly from the program.
        fn exec(&self, program: &Program, extra: &[Vec<u64>], out: Option<&mut [i32]>) -> Value {
            self.with_planes(extra, |planes| {
                let words = words_for(self.n());
                let slots = program.scratch_slots as usize;
                let mut buf = vec![0u64; scratch_words_for(words, slots).expect("sized")];
                let mut scratch = Scratch::over(&mut buf, words, slots).expect("carves");
                execute(program, planes, &mut scratch, out).expect("runs")
            })
        }

        /// The executor on a `Keep` program, copying the kept mask out of
        /// wherever `Value::Mask` says it landed.
        fn exec_mask(&self, program: &Program, extra: &[Vec<u64>]) -> Vec<u64> {
            self.with_planes(extra, |planes| {
                let words = words_for(self.n());
                let slots = program.scratch_slots as usize;
                let mut buf = vec![0u64; scratch_words_for(words, slots).expect("sized")];
                let mut scratch = Scratch::over(&mut buf, words, slots).expect("carves");
                match execute(program, planes, &mut scratch, None).expect("runs") {
                    Value::Mask(Operand::Scratch(i)) => scratch.slot(i).expect("written").to_vec(),
                    Value::Mask(Operand::Plane(p)) => planes.masks[usize::from(p)].to_vec(),
                    other => panic!("not a kept mask: {other:?}"),
                }
            })
        }

        /// mask-risc's own row-at-a-time oracle — a second independent arm.
        fn reference(&self, program: &Program, out: Option<&mut [i32]>) -> Value {
            self.with_planes(&[], |planes| {
                reference_execute(program, planes, out).expect("runs")
            })
        }

        fn count(&self, f: &Filter) -> usize {
            let q = Query {
                filter: f.clone(),
                agg: Agg::Count,
            };
            match self.exec(&lower(&q).expect("lowers"), &[], None) {
                Value::Count(c) => c,
                other => panic!("not a count: {other:?}"),
            }
        }
    }

    /// FAILS IF: an outer conjunction's accumulator replaces a plane gate
    /// established by an INNER conjunction, so the plane — already elided as
    /// implied — vanishes from the program entirely.
    ///
    /// The shape is `P1 AND (Plane(focus) AND P2)`. It is the one the survivor
    /// skip's own machinery cannot reach by the route the other gate tests
    /// take: the OUTER `AND` finds no plane among its parts, so it establishes
    /// no gate, while the INNER one establishes `focus`, sees `P2` vanish
    /// under it and drops it. Nothing in the outer scope then carries `focus`.
    ///
    /// Reported as a P1 by codex on PR #1235 and reproduced before it was
    /// believed: the oracle selected **29** rows of 512 and the emitted
    /// program selected **204**, its op list `GtI32 AND NeU32` with no trace
    /// of the plane. That is the failure mode this asserts against, and the
    /// op-list half is what distinguishes "the answer happened to match" from
    /// "the gate is actually there".
    #[test]
    fn a_nested_plane_survives_an_outer_accumulator() {
        let fx = Fx::new(512);
        let f = Filter::and([
            Filter::cmp(VALS, Cmp::GtI32(0)),
            Filter::and([Filter::plane(FOCUS), Filter::cmp(CLASS, Cmp::NeU32(0))]),
        ]);

        // Anti-vacuity, two-sided: the plane must actually exclude rows the
        // rest admits, or a lost gate would be invisible in the count.
        let without_plane = fx
            .rows(&Filter::and([
                Filter::cmp(VALS, Cmp::GtI32(0)),
                Filter::cmp(CLASS, Cmp::NeU32(0)),
            ]))
            .len();
        let expected = fx.rows(&f).len();
        assert!(
            expected > 0 && expected * 2 < without_plane,
            "the fixture must make the plane load-bearing: {expected} with it, \
             {without_plane} without"
        );

        assert_eq!(
            fx.count(&f),
            expected,
            "the plane was lost from the program"
        );

        // ...and structurally: SOME operand must still read `FOCUS`. A count
        // that happens to agree is not evidence the gate survived.
        let prog = lower(&Query {
            filter: f,
            agg: Agg::Count,
        })
        .expect("lowers");
        let reads_focus = prog.ops.iter().any(
            |op| matches!(op, MaskOp::Pred { under: Some(Operand::Plane(p)), .. } if *p == FOCUS.0),
        );
        assert!(
            reads_focus,
            "no op reads FOCUS — the nested plane is gone: {:?}",
            prog.ops
        );
    }

    /// `alpha & ((A & B) | C)` — the vertical slice.
    fn slice_filter() -> Filter {
        Filter::and([
            Filter::plane(ALPHA),
            Filter::or([
                Filter::and([
                    Filter::cmp(VALS, Cmp::GtI32(0)),
                    Filter::cmp(CLASS, Cmp::NeU32(0)),
                ]),
                Filter::cmp(VALS, Cmp::LtI32(-190)),
            ]),
        ])
    }

    /// FAILS IF: either lowering and an independent per-row reading of the
    /// same filter disagree.
    ///
    /// This is the whole claim of the crate — that a query expressed as
    /// masking ops answers what the query means — so it is checked against an
    /// oracle that never sees a `Program`, for the in-place AND the fused
    /// lowering, over shapes that exercise every gate rule: a gate over a
    /// nested `OR`, a gate that must survive a negation, a gate that must
    /// survive a foreign plane, two planes, a plane alone, an IN-list.
    #[test]
    fn every_lowered_filter_agrees_with_an_independent_per_row_reading() {
        let fx = Fx::new(N);
        let cases: Vec<(&str, Filter)> = vec![
            ("one leaf", Filter::cmp(VALS, Cmp::GtI32(0))),
            ("u32 leaf", Filter::cmp(CLASS, Cmp::EqU32(2))),
            ("a plane alone", Filter::plane(ALPHA)),
            (
                "and of two",
                Filter::and([
                    Filter::cmp(VALS, Cmp::GtI32(-50)),
                    Filter::cmp(CLASS, Cmp::NeU32(0)),
                ]),
            ),
            (
                "or of two",
                Filter::or([
                    Filter::cmp(VALS, Cmp::LtI32(-150)),
                    Filter::cmp(CLASS, Cmp::EqU32(4)),
                ]),
            ),
            ("not", Filter::negate(Filter::cmp(VALS, Cmp::GeI32(0)))),
            (
                "and of four — the width case",
                Filter::and([
                    Filter::cmp(VALS, Cmp::GtI32(-180)),
                    Filter::cmp(VALS, Cmp::LtI32(180)),
                    Filter::cmp(CLASS, Cmp::NeU32(3)),
                    Filter::cmp(CLASS, Cmp::NeU32(1)),
                ]),
            ),
            (
                "nested — or inside and, with a not",
                Filter::and([
                    Filter::cmp(VALS, Cmp::GeI32(-100)),
                    Filter::or([
                        Filter::cmp(CLASS, Cmp::EqU32(1)),
                        Filter::negate(Filter::cmp(VALS, Cmp::GtI32(100))),
                    ]),
                ]),
            ),
            (
                "ternary match — no SQL spelling, the substrate had it all along",
                Filter::cmp(
                    CLASS,
                    Cmp::MatchU32 {
                        pattern: 0b100,
                        care: 0b110,
                    },
                ),
            ),
            ("gate over a comparison", {
                Filter::and([Filter::plane(ALPHA), Filter::cmp(VALS, Cmp::GtI32(0))])
            }),
            ("gate over a nested or — the slice", slice_filter()),
            (
                "gate over a negation — the gate must stay a leaf",
                Filter::and([
                    Filter::plane(ALPHA),
                    Filter::negate(Filter::cmp(VALS, Cmp::GtI32(0))),
                ]),
            ),
            (
                "gate over an or with a negated arm",
                Filter::and([
                    Filter::plane(ALPHA),
                    Filter::or([
                        Filter::cmp(CLASS, Cmp::EqU32(3)),
                        Filter::negate(Filter::cmp(VALS, Cmp::GtI32(-100))),
                    ]),
                ]),
            ),
            (
                "gate beside a foreign plane — the foreign plane is a leaf",
                Filter::and([
                    Filter::plane(ALPHA),
                    Filter::plane(FOCUS),
                    Filter::cmp(VALS, Cmp::LtI32(50)),
                ]),
            ),
            (
                "two planes or'd, then a comparison",
                Filter::and([
                    Filter::or([Filter::plane(ALPHA), Filter::plane(FOCUS)]),
                    Filter::cmp(VALS, Cmp::NeI32(0)),
                ]),
            ),
            ("plane after the comparisons", {
                Filter::and([
                    Filter::cmp(VALS, Cmp::GtI32(-100)),
                    Filter::cmp(VALS, Cmp::LtI32(100)),
                    Filter::plane(FOCUS),
                ])
            }),
            (
                "a gated and nested under a gated and",
                Filter::and([
                    Filter::plane(ALPHA),
                    Filter::and([Filter::plane(FOCUS), Filter::cmp(VALS, Cmp::GtI32(-100))]),
                ]),
            ),
            ("not of a plane", Filter::negate(Filter::plane(ALPHA))),
            (
                "in-list over the class lane",
                Filter::in_u32(CLASS, [1, 3, 3]),
            ),
            (
                "in-list over the value lane",
                Filter::in_i32(VALS, [-200, 5, 100, 12]),
            ),
        ];

        for (label, f) in cases {
            let expected = fx.rows(&f).len();
            let q = Query {
                filter: f,
                agg: Agg::Count,
            };
            let inplace = lower(&q).expect("lowers in place");
            let fused = lower_fused(&q).expect("lowers fused");
            assert_eq!(
                fx.exec(&inplace, &[], None),
                Value::Count(expected),
                "{label}: the in-place program and the per-row oracle disagree"
            );
            assert_eq!(
                fx.exec(&fused, &[], None),
                Value::Count(expected),
                "{label}: the fused program and the per-row oracle disagree"
            );
            // Anti-vacuity: agreement on "nothing" or "everything" would hold
            // for a lowering that ignored the filter entirely.
            assert!(
                expected > 0 && expected < N,
                "{label} selects {expected}/{N} — a degenerate case proves nothing"
            );
        }
    }

    /// How `p` gates its comparisons, and whether `g` survives as an operand.
    ///
    /// Returns `(every comparison gated, some comparison gated on the
    /// ACCUMULATOR, g is never read as a Boolean operand)`.
    ///
    /// The middle field did not exist before accumulator gating: a gate that is
    /// a scratch slot is the partial result of the conjunction so far, so it
    /// NARROWS as the conjunction proceeds, where a plane gate is fixed. That
    /// narrowing is the mechanism row A1 is about.
    fn gate_shape(p: &Program, g: Mask) -> (bool, bool, bool) {
        let gate = Operand::Plane(g.0);
        let mut all_gated = true;
        let mut on_accumulator = false;
        let mut read_as_leaf = false;
        for op in &p.ops {
            match *op {
                MaskOp::Pred { under, .. } => match under {
                    Some(Operand::Scratch(_)) => on_accumulator = true,
                    Some(_) => {}
                    None => all_gated = false,
                },
                MaskOp::And { a, b, .. }
                | MaskOp::Or { a, b, .. }
                | MaskOp::Xor { a, b, .. }
                | MaskOp::AndNot { a, b, .. } => read_as_leaf |= a == gate || b == gate,
                MaskOp::Not { a, .. } => read_as_leaf |= a == gate,
                MaskOp::Ternlog { a, b, c, .. } => {
                    read_as_leaf |= a == gate || b == gate || c == gate;
                }
            }
        }
        let terminal_reads_gate = matches!(
            p.terminal,
            Terminal::Count { mask }
                | Terminal::Any { mask }
                | Terminal::All { mask }
                | Terminal::MaskedSumI32 { mask, .. }
                | Terminal::MaskedMinI32 { mask, .. }
                | Terminal::MaskedMaxI32 { mask, .. }
                | Terminal::BlendI32 { mask, .. }
                | Terminal::Keep { mask } if mask == gate
        );
        (
            all_gated,
            on_accumulator,
            !read_as_leaf && !terminal_reads_gate,
        )
    }

    /// FAILS IF: the survivor skip gates the wrong leaves, drops the gate where
    /// the drop is unsound, or stops narrowing onto the accumulator.
    ///
    /// - **can-fire.** Under `alpha & ((A & B) | C)` every comparison is gated
    ///   and alpha is never read as a Boolean operand; under the in-place
    ///   lowering at least one comparison is gated on the ACCUMULATOR.
    /// - **can-stay-silent.** Under `alpha & !X` the comparison is still gated,
    ///   but alpha IS read as a leaf — `!(alpha & X)` is 1 exactly where alpha
    ///   is 0, so dropping it would be unsound.
    /// - **the foreign plane.** A second plane beside the gate stays an
    ///   ordinary leaf; it does not become a second gate.
    ///
    /// The differential holds either way; this pins the SHAPE, so a walk that
    /// dropped the gate under a negation — or one that quietly stopped gating
    /// on the accumulator, which no count would ever notice — fails here.
    #[test]
    fn the_gate_reaches_every_comparison_and_is_dropped_only_where_it_vanishes() {
        for (is_inplace, lowering) in [(true, lower as fn(&Query) -> _), (false, lower_fused)] {
            let slice = lowering(&Query {
                filter: slice_filter(),
                agg: Agg::Count,
            })
            .expect("lowers");
            let (gated, on_acc, dropped) = gate_shape(&slice, ALPHA);
            assert!(
                gated && dropped,
                "the slice: every predicate gated, alpha implied: {:?}",
                slice.ops
            );
            // ⊘ RE-PINNED. This asserted `on_acc` for the in-place lowering:
            // inside a plane's own AND, later comparisons narrowed onto the
            // running accumulator rather than onto the plane. That override
            // is what the codex P1 on PR #1235 showed to be UNSOUND once an
            // outer conjunction supplies an unrelated accumulator, so
            // [`emit_gated`] now always prefers the plane and the assertion
            // became false. It is inverted rather than deleted, because the
            // property is load-bearing in the other direction: a plane gate
            // must never be silently replaced.
            //
            // A1's lever is untouched and is pinned separately by
            // `skip_ordering_moves_the_work_and_never_the_answer`, whose
            // conjunction carries no plane — those preds still gate on the
            // accumulator.
            //
            // ⊘ AND IT IS A REAL FALSIFIER ON ONE ARM ONLY. `assign_slots`
            // never emits a `Scratch` gate at all, so on the FUSED arm no
            // input can make this fail — it passes structurally, not because
            // the property holds. The in-place arm is where it bites, and
            // where restoring the pre-fix gate line turns it red. Asserting
            // the two separately is the point: a single `assert!` run over
            // both arms reads as twice the evidence and is once.
            if is_inplace {
                assert!(
                    !on_acc,
                    "a plane-gated comparison narrowed onto the accumulator \
                     instead of the plane — the codex-P1 override is back: {:?}",
                    slice.ops
                );
            } else {
                assert!(
                    !slice.ops.iter().any(|op| matches!(
                        op,
                        MaskOp::Pred {
                            under: Some(Operand::Scratch(_)),
                            ..
                        }
                    )),
                    "the fused arm is expected to carry no scratch gate at \
                     all; if that ever changes, the in-place assertion above \
                     stops being the only place this is tested: {:?}",
                    slice.ops
                );
            }

            let negated = lowering(&Query {
                filter: Filter::and([
                    Filter::plane(ALPHA),
                    Filter::negate(Filter::cmp(VALS, Cmp::GtI32(0))),
                ]),
                agg: Agg::Count,
            })
            .expect("lowers");
            let (gated, _, dropped) = gate_shape(&negated, ALPHA);
            assert!(
                gated && !dropped,
                "under a negation: still gated, but alpha stays a leaf: {:?}",
                negated.ops
            );

            let foreign = lowering(&Query {
                filter: Filter::and([
                    Filter::plane(ALPHA),
                    Filter::plane(FOCUS),
                    Filter::cmp(VALS, Cmp::LtI32(50)),
                ]),
                agg: Agg::Count,
            })
            .expect("lowers");
            let (gated, _, dropped) = gate_shape(&foreign, ALPHA);
            assert!(gated && dropped, "the first plane gates: {:?}", foreign.ops);
            assert!(
                !gate_shape(&foreign, FOCUS).2,
                "the foreign plane is read as a leaf, not gated away: {:?}",
                foreign.ops
            );
        }
    }

    /// FAILS IF: a resident plane costs an op or a slot.
    ///
    /// `SELECT count(*) FROM t` is the plane's population; the program is
    /// empty, the terminal reads the plane, and projection returns the plane
    /// itself rather than a copy of it.
    #[test]
    fn a_resident_plane_alone_is_a_zero_op_program() {
        let fx = Fx::new(N);
        let expected = fx.rows(&Filter::plane(ALPHA)).len();
        assert!(expected > 0 && expected < N);
        for lowering in [lower, lower_fused] {
            let count = lowering(&Query {
                filter: Filter::plane(ALPHA),
                agg: Agg::Count,
            })
            .expect("lowers");
            assert!(count.ops.is_empty(), "{:?}", count.ops);
            assert_eq!(count.scratch_slots, 0);
            assert_eq!(
                count.terminal,
                Terminal::Count {
                    mask: Operand::Plane(0)
                }
            );
            assert_eq!(fx.exec(&count, &[], None), Value::Count(expected));

            let rows = lowering(&Query {
                filter: Filter::plane(ALPHA),
                agg: Agg::Rows,
            })
            .expect("lowers");
            assert_eq!(
                fx.exec(&rows, &[], None),
                Value::Mask(Operand::Plane(0)),
                "projection of a plane is the plane, not a copy"
            );
        }
    }

    /// FAILS IF: the fused lowering does not buy passes with slots — or buys
    /// nothing.
    ///
    /// On the slice, in place is two Boolean passes (an AND, then an OR) in
    /// two slots; fused is one ternlog in four. Both are pinned, so a fuser
    /// that stopped fusing OR an in-place emitter that started allocating per
    /// leaf would each fail their own line.
    #[test]
    fn the_fused_lowering_trades_slots_for_passes() {
        let q = Query {
            filter: slice_filter(),
            agg: Agg::Count,
        };
        let inplace = lower(&q).expect("lowers");
        let fused = lower_fused(&q).expect("lowers");

        assert_eq!(inplace.op_histogram().predicates, 3);
        assert_eq!(fused.op_histogram().predicates, 3);
        assert_eq!(inplace.op_histogram().mask_passes(), 2);
        assert_eq!(fused.op_histogram().mask_passes(), 1);
        assert_eq!(fused.op_histogram().ternlog, 1);
        assert_eq!(inplace.scratch_slots, 2);
        assert_eq!(fused.scratch_slots, 4);
    }

    /// FAILS IF: a junction allocates a slot per child in the in-place form.
    ///
    /// Children fold left-to-right into the first child's slot, so width is
    /// free and only DEPTH costs. Without this a 64-wide conjunction would ask
    /// for a 64-slot arena — a scratch allocation proportional to the query's
    /// text rather than to its shape. The fused form DOES pay width, and that
    /// is pinned too: it is the trade, not a defect.
    #[test]
    fn a_wide_conjunction_costs_one_extra_slot_not_one_per_child() {
        let wide = Query {
            filter: Filter::and((0..32).map(|i| Filter::cmp(VALS, Cmp::NeI32(i)))),
            agg: Agg::Count,
        };
        let p = lower(&wide).expect("lowers");
        assert_eq!(
            p.scratch_slots, 2,
            "32 children must cost depth-2, not 32 slots"
        );
        let pf = lower_fused(&wide).expect("lowers");
        assert!(
            pf.scratch_slots >= 32,
            "fused pays one slot per comparison: {}",
            pf.scratch_slots
        );

        // Paired: DEPTH does cost, or the constant above would be vacuous.
        let deep = Query {
            filter: Filter::and([
                Filter::cmp(VALS, Cmp::GtI32(0)),
                Filter::or([
                    Filter::cmp(CLASS, Cmp::EqU32(1)),
                    Filter::and([
                        Filter::cmp(VALS, Cmp::LtI32(50)),
                        Filter::cmp(CLASS, Cmp::NeU32(2)),
                    ]),
                ]),
            ]),
            agg: Agg::Count,
        };
        let pd = lower(&deep).expect("lowers");
        assert!(
            pd.scratch_slots > p.scratch_slots,
            "depth must cost more than width: deep={} wide={}",
            pd.scratch_slots,
            p.scratch_slots
        );
    }

    /// FAILS IF: an empty AND/OR — or the `IN ()` that is one — is folded to
    /// a constant instead of refused.
    ///
    /// An empty conjunction is `true` and an empty disjunction is `false`, so
    /// whichever identity the code picked would silently be the answer to a
    /// query the caller built by accident.
    ///
    /// Disable note, because a single-site run misreads: the refusal is
    /// spelled at THREE sites ([`walk_all`], [`emit_inplace`],
    /// [`assign_slots`]) and disabling any one leaves this green — not
    /// because the test is vacuous but because the other two still catch it.
    /// Disabling all three together is what turns it red. A future session
    /// measuring one site and concluding "not load-bearing" would be reading
    /// the redundancy, not the guard.
    #[test]
    fn an_empty_junction_is_refused_rather_than_folded_to_an_identity() {
        for f in [
            Filter::And(vec![]),
            Filter::Or(vec![]),
            Filter::in_u32(CLASS, []),
            Filter::in_i32(VALS, []),
            Filter::and([Filter::plane(ALPHA), Filter::Or(vec![])]),
        ] {
            let q = Query {
                filter: f.clone(),
                agg: Agg::Count,
            };
            assert_eq!(lower(&q), Err(LowerError::EmptyJunction), "{f:?}");
            assert_eq!(lower_fused(&q), Err(LowerError::EmptyJunction), "{f:?}");
        }
    }

    /// FAILS IF: an IN-list is anything other than the disjunction it claims
    /// to be.
    ///
    /// Structural equality with the hand-built OR, and agreement with the
    /// oracle; duplicates in the set are harmless because they are harmless
    /// in an OR.
    #[test]
    fn an_in_list_is_a_disjunction_of_equalities() {
        let fx = Fx::new(N);
        let by_hand = Filter::or([
            Filter::cmp(CLASS, Cmp::EqU32(1)),
            Filter::cmp(CLASS, Cmp::EqU32(3)),
            Filter::cmp(CLASS, Cmp::EqU32(3)),
        ]);
        let in_list = Filter::in_u32(CLASS, [1, 3, 3]);
        assert_eq!(in_list, by_hand);
        let expected = (0..N).filter(|&r| [1, 3].contains(&fx.classes[r])).count();
        assert_eq!(fx.count(&in_list), expected);
        assert!(expected > 0 && expected < N);
    }

    /// FAILS IF: the aggregates and the projections do not read the filter's
    /// mask.
    ///
    /// `Count`/`Any`/`All`/`Sum`/`Min`/`Max`/`Rows`/`Blend` over the SAME
    /// filter must be mutually consistent with the per-row oracle — an
    /// aggregate wired to the wrong operand would still return a plausible
    /// number, and a blend wired to the wrong lane a plausible column.
    #[test]
    fn every_aggregate_and_projection_reads_the_same_filter_mask() {
        let fx = Fx::new(N);
        let f = Filter::and([
            Filter::plane(ALPHA),
            Filter::cmp(VALS, Cmp::GtI32(-100)),
            Filter::cmp(CLASS, Cmp::EqU32(3)),
        ]);
        let rows = fx.rows(&f);
        assert!(
            !rows.is_empty() && rows.len() < N,
            "fixture must be a proper subset"
        );

        let sum: i64 = rows.iter().map(|&r| i64::from(fx.vals[r])).sum();
        let min = rows.iter().map(|&r| fx.vals[r]).min();
        let max = rows.iter().map(|&r| fx.vals[r]).max();

        let q = |agg| Query {
            filter: f.clone(),
            agg,
        };
        let run = |agg| fx.exec(&lower(&q(agg)).expect("lowers"), &[], None);
        assert_eq!(run(Agg::Count), Value::Count(rows.len()));
        assert_eq!(run(Agg::Any), Value::Bool(true));
        assert_eq!(run(Agg::All), Value::Bool(false));
        assert_eq!(run(Agg::SumI32(VALS)), Value::SumI64(sum));
        assert_eq!(run(Agg::MinI32(VALS)), Value::OptI32(min));
        assert_eq!(run(Agg::MaxI32(VALS)), Value::OptI32(max));

        // Projection: the kept mask materialises to exactly the oracle's rows.
        let kept = fx.exec_mask(&lower(&q(Agg::Rows)).expect("lowers"), &[]);
        assert_eq!(materialize_rows(&kept, N), rows);

        // CASE: every row reads `then` where the filter holds, `els` where not.
        let mut out = vec![0i32; N];
        assert_eq!(
            fx.exec(
                &lower(&q(Agg::BlendI32 {
                    then: VALS,
                    els: ALT
                }))
                .expect("lowers"),
                &[],
                Some(&mut out),
            ),
            Value::Blended
        );
        let expected: Vec<i32> = (0..N)
            .map(|r| {
                if fx.oracle(&f, r) {
                    fx.vals[r]
                } else {
                    fx.alt[r]
                }
            })
            .collect();
        assert_eq!(out, expected);
        assert!(
            out.iter().zip(&fx.vals).any(|(o, v)| o != v),
            "the blend must actually pick from `els` somewhere"
        );
    }

    /// FAILS IF: the two-phase GROUP BY does not partition the filtered rows
    /// by key.
    ///
    /// Per-group counts and sums equal the oracle's, and the counts sum to
    /// the ungrouped filtered count: a group program that ignored the key
    /// would report the total K times, one that ignored the kept filter
    /// would report the unfiltered class sizes, and either breaks the
    /// partition identity.
    #[test]
    fn a_group_by_partitions_the_filtered_rows_by_key() {
        let fx = Fx::new(N);
        let filter = Filter::and([Filter::plane(ALPHA), Filter::cmp(VALS, Cmp::GtI32(0))]);
        let rows = fx.rows(&filter);
        let groups = 5u32;
        let filter_plane = u16::try_from(fx.masks.len()).expect("fits");

        for agg in [Agg::Count, Agg::SumI32(VALS)] {
            let plan = lower_group_by(
                &GroupBy {
                    filter: filter.clone(),
                    key: CLASS,
                    groups,
                    agg,
                },
                filter_plane,
            )
            .expect("lowers");
            assert_eq!(plan.groups.len(), groups as usize);
            assert_eq!(plan.filter_plane, filter_plane);

            let kept = fx.exec_mask(&plan.filter, &[]);
            let extra = vec![kept];
            let results: Vec<Value> = plan
                .groups
                .iter()
                .map(|p| fx.exec(p, &extra, None))
                .collect();

            for (g, value) in results.iter().enumerate() {
                let members: Vec<usize> = rows
                    .iter()
                    .copied()
                    .filter(|&r| fx.classes[r] == g as u32)
                    .collect();
                let expected = match agg {
                    Agg::Count => Value::Count(members.len()),
                    Agg::SumI32(_) => {
                        Value::SumI64(members.iter().map(|&r| i64::from(fx.vals[r])).sum())
                    }
                    other => unreachable!("{other:?}"),
                };
                assert_eq!(*value, expected, "group {g}");
                assert!(!members.is_empty(), "group {g} must be non-empty to count");
            }
            if agg == Agg::Count {
                let total: usize = results
                    .iter()
                    .map(|v| match v {
                        Value::Count(c) => *c,
                        other => panic!("{other:?}"),
                    })
                    .sum();
                assert_eq!(total, rows.len(), "the groups partition the filter");
                assert!(rows.len() < N);
            }
        }

        assert_eq!(
            lower_group_by(
                &GroupBy {
                    filter,
                    key: CLASS,
                    groups,
                    agg: Agg::BlendI32 {
                        then: VALS,
                        els: ALT
                    },
                },
                filter_plane,
            ),
            Err(LowerError::GroupedBlend)
        );
    }

    /// FAILS IF: an address prefix does not select a CONTIGUOUS row range, or
    /// the range is not the size the prefix arithmetic says it is.
    ///
    /// This is the matrix's §5.1 claim — *the address IS the trie, so a prefix
    /// predicate is a range rather than a sweep* — made checkable instead of
    /// asserted. It matters because the whole "better than faithful" argument
    /// rests on it: DuckDB has the compressed range in `SequenceVector` and
    /// throws it away at `ToUnifiedFormat`, then rebuilds it as a per-row index
    /// loop in `DataChunk::Slice`. If V3's prefix did not actually select a
    /// range, keeping the range would be keeping nothing.
    ///
    /// Two-sided, because "selects a contiguous run" alone would hold for a
    /// predicate that selected everything: each additional significant bit must
    /// HALVE the run, and the widest prefix must select exactly one row.
    #[test]
    fn an_address_prefix_selects_exactly_a_contiguous_trie_subtree() {
        // A POWER-OF-TWO population, deliberately: a subtree can only halve
        // cleanly while it still fits inside the population. At N = 1000 the
        // widest prefix here selects 1000 rather than 1024, and the halving
        // claim then reads as a defect when it is a fixture artifact — which
        // is exactly what the first version of this test measured.
        const POW2: usize = 1024;
        let fx = Fx::new(POW2);
        // The lane is `row << 8`, so bit 8 + k of the address is bit k of the
        // row index: a prefix of `24 + b` significant bits pins the top `b`
        // bits of the row index and leaves `24 - 8 = 16`... stated the way the
        // arithmetic actually runs, `care = !0 << (64 - bits)`, and a row
        // survives when its address agrees on those bits.
        let base_row = 384usize;
        let base = fx.addr[base_row];
        // The arithmetic, stated so the expected sizes are DERIVED and not
        // fitted to what the code happened to return: row `r` sits at address
        // `r << 8`, so row bit `k` is address bit `8 + k`. A prefix of `bits`
        // significant bits pins address bits `[64 - bits, 63]`, hence row bits
        // `k >= 56 - bits`, leaving `clamp(56 - bits, 0, 10)` of them free — a
        // subtree of `2^free` consecutive rows.
        let subtree_of = |bits: u32| 1usize << (56u32.saturating_sub(bits).min(10));

        let mut previous: Option<usize> = None;
        for bits in [46u32, 47, 48, 49, 50] {
            let f = Filter::prefix_u64(ADDR, base, bits);
            let rows = fx.rows(&f);
            assert!(!rows.is_empty(), "bits={bits} selected nothing");

            // Contiguous: the selected rows are consecutive, with no holes.
            // A sweep over an unordered lane could not satisfy this, which is
            // exactly the property being claimed.
            let first = rows[0];
            assert!(
                rows.iter().enumerate().all(|(i, &r)| r == first + i),
                "bits={bits} selected a non-contiguous set: {:?}..",
                &rows[..rows.len().min(8)]
            );
            assert!(
                rows.contains(&base_row),
                "bits={bits} must contain the row the prefix was taken from"
            );

            // Each extra significant bit halves the subtree. That is the
            // arithmetic the prefix IS; a care mask built wrongly would still
            // select a contiguous run, just the wrong one.
            assert_eq!(
                rows.len(),
                subtree_of(bits),
                "bits={bits} selected {} rows; the prefix arithmetic says {}",
                rows.len(),
                subtree_of(bits)
            );
            if let Some(prev) = previous {
                assert_eq!(
                    rows.len() * 2,
                    prev,
                    "bits={bits} selected {} rows against {prev} for one bit fewer \
                     — a trie level must halve",
                    rows.len()
                );
            }
            previous = Some(rows.len());
        }
        assert_eq!(
            previous,
            Some(64),
            "the 50-bit prefix pins the row index down to a 64-row subtree"
        );

        // The widest prefix is a single address; the empty one is every row.
        assert_eq!(fx.count(&Filter::prefix_u64(ADDR, base, 64)), 1);
        assert_eq!(fx.count(&Filter::prefix_u64(ADDR, base, 0)), POW2);
    }

    /// FAILS IF: the 64-bit ternary match disagrees with an independent per-row
    /// reading, or the lowered program does not reach the `U64` lane at all.
    ///
    /// `Pred::MatchU64` shipped with the IR in PR3 and this crate could not
    /// spell it, so a borrowed `LaneRef::U64` — edge targets, ids, addresses —
    /// was queryable by nothing. The anti-vacuity half matters more than usual
    /// here: a care mask of zero matches every row, which is a real answer and
    /// a useless test.
    #[test]
    fn the_64_bit_ternary_match_agrees_with_a_per_row_reading() {
        let fx = Fx::new(N);
        let cases: [(&str, u64, u64); 3] = [
            ("one low nibble of the address", 0x300, 0xF00),
            ("a sparse care mask", 0x2_0100, 0x3_0100),
            ("every bit — exact equality", fx.addr[7], u64::MAX),
        ];
        for (label, pattern, care) in cases {
            let f = Filter::cmp(ADDR, Cmp::MatchU64 { pattern, care });
            let expected = fx.rows(&f).len();
            assert_eq!(fx.count(&f), expected, "{label}");
            assert!(
                expected > 0 && expected < N,
                "{label} selects {expected}/{N} — a degenerate case proves nothing"
            );
        }
        // Composed with the rest of the vocabulary, so the U64 lane is not a
        // second world that only works alone.
        // 48 significant bits, not 56: at 56 the subtree is a SINGLE row, and
        // whether the conjunction is non-empty then turns on whether that one
        // row happens to satisfy the other two conjuncts — which it does not,
        // so the first version measured 0/1000 and looked like a defect. A
        // 256-row subtree is what makes this a composition test rather than a
        // coin flip.
        let mixed = Filter::and([
            Filter::plane(ALPHA),
            Filter::prefix_u64(ADDR, fx.addr[500], 48),
            Filter::cmp(CLASS, Cmp::NeU32(0)),
        ]);
        let expected = fx.rows(&mixed).len();
        assert!(expected > 0 && expected < N, "{expected}/{N}");
        assert_eq!(fx.count(&mixed), expected);
    }

    /// FAILS IF: skip-ordering changes the ANSWER, or does not change the
    /// ORDER, or stops being stable for equal scores.
    ///
    /// The safety property is the one that matters most: `AND` is commutative,
    /// so reordering may move work and must never move the result. A lowering
    /// that reordered children but got the gate chain wrong would produce a
    /// different count, and the per-row oracle would catch it — which is why
    /// the answer is checked against the oracle and not merely against the
    /// unordered lowering.
    ///
    /// Then the can-fire half: the emitted program must actually differ. A
    /// builder that sorted and then lost the order somewhere in `gate_walk`
    /// would pass the safety half perfectly.
    #[test]
    fn skip_ordering_moves_the_work_and_never_the_answer() {
        let fx = Fx::new(N);
        // Written worst-first on purpose: the permissive term leads, which is
        // the shape DuckDB's AdaptiveFilter exists to fix.
        let permissive = Filter::cmp(VALS, Cmp::GtI32(-900));
        let moderate = Filter::cmp(CLASS, Cmp::NeU32(0));
        let selective = Filter::cmp(CLASS, Cmp::EqU32(3));

        let written = Filter::and([permissive.clone(), moderate.clone(), selective.clone()]);
        // Scores are the caller's evidence — here, dead words each term would
        // leave behind, highest first.
        let ordered = Filter::and_by_skip([
            (1, permissive.clone()),
            (7, moderate.clone()),
            (900, selective.clone()),
        ]);

        // Same answer, and the oracle — not the other lowering — is the judge.
        let expected = fx.rows(&written).len();
        assert!(
            expected > 0 && expected < N,
            "{expected}/{N} — a degenerate conjunction proves nothing"
        );
        assert_eq!(fx.rows(&ordered).len(), expected);
        assert_eq!(fx.count(&written), expected);
        assert_eq!(fx.count(&ordered), expected);

        // The order really moved: the most selective term now leads.
        assert_eq!(
            ordered,
            Filter::and([selective.clone(), moderate.clone(), permissive.clone()]),
            "and_by_skip must sort descending by score"
        );
        assert_ne!(ordered, written, "the fixture must not already be ordered");

        // ...and the emitted program reflects it: the FIRST predicate is the
        // one that was scored highest. Without this the sort could be undone
        // between the builder and the lowering and nothing would notice.
        let program = lower(&Query {
            filter: ordered.clone(),
            agg: Agg::Count,
        })
        .expect("lowers");
        assert_eq!(
            program.ops.first(),
            Some(&MaskOp::Pred {
                pred: Pred::EqU32 {
                    lane: CLASS.0,
                    v: 3
                },
                under: None,
                dst: 0,
            }),
            "the highest-scored term must be the ungated seed: {:?}",
            program.ops
        );

        // Stability: equal scores keep the caller's order, so a scorer with no
        // information cannot silently permute a hand-tuned conjunction.
        assert_eq!(
            Filter::and_by_skip([
                (5, permissive.clone()),
                (5, moderate.clone()),
                (5, selective.clone()),
            ]),
            written,
            "equal scores must be stable"
        );
    }

    /// FAILS IF: any arm of the vertical slice disagrees on a 64k slab.
    ///
    /// `COUNT(alpha & ((A & B) | C))` over 65,536 rows through the per-row
    /// oracle, the in-place program on the executor, the fused program on the
    /// executor, and both programs on mask-risc's reference evaluator — five
    /// readings, one number. Anti-vacuity: the gate binds (the count is
    /// strictly below the ungated remainder's) and the answer is a proper
    /// subset of alpha.
    #[test]
    fn the_vertical_slice_agrees_across_every_arm_on_a_64k_slab() {
        let fx = Fx::new(1 << 16);
        let f = slice_filter();
        let expected = fx.rows(&f).len();
        let alpha = fx.rows(&Filter::plane(ALPHA)).len();
        let ungated = fx
            .rows(&Filter::or([
                Filter::and([
                    Filter::cmp(VALS, Cmp::GtI32(0)),
                    Filter::cmp(CLASS, Cmp::NeU32(0)),
                ]),
                Filter::cmp(VALS, Cmp::LtI32(-190)),
            ]))
            .len();
        assert!(expected > 0 && expected < alpha, "{expected} of {alpha}");
        assert!(
            expected < ungated,
            "the gate must bind: {expected} vs {ungated}"
        );

        let q = Query {
            filter: f,
            agg: Agg::Count,
        };
        let inplace = lower(&q).expect("lowers");
        let fused = lower_fused(&q).expect("lowers");
        assert_eq!(fx.exec(&inplace, &[], None), Value::Count(expected));
        assert_eq!(fx.exec(&fused, &[], None), Value::Count(expected));
        assert_eq!(fx.reference(&inplace, None), Value::Count(expected));
        assert_eq!(fx.reference(&fused, None), Value::Count(expected));
    }
}
