//! `lance-graph-r2il` — the fold reading of R2IL, executed over mask-risc.
//!
//! [`FoldDialect`] is an [`ogar_loco::Dialect`] that interprets R2IL calls
//! (`ogar_r2il`'s opcode table, R2IL band `0x90..=0xE1` plus the fold band
//! `0xE2..=0xED`) over borrowed column planes. Predicate calls build
//! [`MaskOp`]s; at every scalar-producing fold (`SUM`, `POP_COUNT`,
//! `GROUP_SUM`, `KEEP`) the accumulated ops are finalized into one
//! mask-risc [`Program`] and run with [`execute_into`]. mask-risc is the only
//! bulk engine; this crate holds no opcode loop over rows.
//!
//! Extracted unchanged from `r2il-mask-abi-probe/tests/row_bridge.rs`
//! (D-SQLP-1, `.claude/plans/sql-polyfill-r2il-v1.md`). That file keeps the
//! reference tests, disable falsifiers and allocation measurements; the
//! counting global allocator stays there, in the test binary.
//!
//! # Execution model
//!
//! A mask-risc [`Program`] has exactly one terminal, so a body such as
//! `SUM(revenue | m) - SUM(cost | m)` cannot be one `Program`. Instead each
//! scalar-producing fold finalizes the ops accumulated so far, runs them, and
//! pushes the result as [`Val::Scalar`]; later scalar calls combine those
//! results. Nothing row-sized reaches the loco stack: only [`Val::Scalar`],
//! [`Val::Slot`] (a scratch-slot name plus its epoch) and [`Val::Address`].
//!
//! The dialect borrows its planes. Its scratch buffer is allocated once at
//! construction, sized by the tile (`scratch_words_for(tile_words_for(n_rows),
//! MAX_SLOTS)`), and carved fresh at every fold, so interpreter allocation
//! does not grow with the row count. `GROUP_SUM` and `KEEP` write into
//! caller-lent sinks ([`FoldDialect::with_groups`], [`FoldDialect::with_keep`]).
//!
//! Slots carry the epoch they were minted in. Every fold restarts slot
//! numbering, so a slot from an earlier fold may alias a fresh one; consuming
//! arms refuse it with [`FoldError::StaleSlot`].
//!
//! # Implemented, and refused by name
//!
//! Fold band: `VIA`, `SUM`, `MIN`, `MAX`, `GROUP_SUM`, `KEY_RUNS`, `RANGE`,
//! `ANY`, `ALL` and `KEEP` are wired. `SCATTER_OR` and `BLEND` are refused as
//! [`FoldError::Unimplemented`]. `ANY`/`ALL` push 1 or 0; `KEY_RUNS` pushes
//! `COUNT(DISTINCT key)` over a key-ordered lane.
//!
//! `MIN`/`MAX` of an empty selection push the fold's identity, `i64::MAX` /
//! `i64::MIN`: the same seed mask-risc's grouped `MIN`/`MAX` leave in an
//! empty group. No `I32` value equals it, so a caller reads it as SQL NULL.
//!
//! R2IL band: `Load` (spaces 0 = U32, 1 = I32, 2 = U64, 3 = resident mask
//! plane), `IntAdd`, `IntSub`, `IntMult`, `IntSDiv` (scalar/scalar, checked), `IntAnd` (with survivor gating),
//! `IntOr`, `IntXor`, `IntNot`, `IntEqual`, `IntSLess` and `PopCount` are wired. `IntLess` and
//! `IntLessEqual` are refused as [`FoldError::NoUnsignedLaneCompare`]
//! (mask-risc has no unsigned lane compare); `IntSub` of two addresses as
//! [`FoldError::LaneVersusLane`]; an unknown `Load` space as
//! [`FoldError::UnknownLoadSpace`]; every other byte as
//! [`FoldError::Unimplemented`].
//!
//! # Boolean modes
//!
//! [`BoolMode`] decides how `IntAnd`, `IntOr` and `IntNot` on masks lower,
//! mirroring quack's `lower` / `lower_fused`:
//!
//! - [`BoolMode::InPlace`] (default): one mask-risc op per call. `IntAnd`
//!   gates a just-emitted predicate under its left operand (survivor gating)
//!   instead of spending an `And`.
//! - [`BoolMode::Fused`]: the calls build one [`BoolExpr`]; the first consumer
//!   (`PopCount`, `ANY`, `ALL`, `SUM`, `GROUP_SUM`, `KEEP`) lowers it through mask-risc
//!   [`fuse()`] into ternlogs. Fewer passes, no survivor gating.
//!
//! Both modes keep the same rows. Predicates are emitted the same way in
//! both; only the Boolean combination differs.
//!
//! # Procedures
//!
//! [`procs`] holds named call sequences (`filter_lt`, `select_u32`, `count`,
//! `exists`, `all`, `differ`, `intersect`, `union`, `keep`, `count_and`,
//! the PostgreSQL primitives `is_null`, `is_not_null`, `and_between_i32`,
//! `in_u32`, `count_valid`, `avg_parts`, `bool_or`, `bool_and`, `not_exists`,
//! `except`, `count_distinct_sorted`, `count_union_all`,
//! `sum`, `group_sum`, `margin`). They are Rust functions over [`Dialect::call`], not opcodes.
//!
//! # Slot limit
//!
//! The scratch buffer is sized once, for [`MAX_SLOTS`] slots. One fold may
//! therefore use at most [`MAX_SLOTS`] scratch slots between two fold
//! boundaries; a wider body is refused at that fold with
//! [`FoldError::Exec`]`(ScratchBufferTooSmall)`, never truncated. The limit
//! keeps allocation independent of program size as well as row count. A
//! caller that needs wider folds needs a larger arena, which is a constructor
//! change, not a silent grow.
//!
//! # Branching on a population
//!
//! `Dialect::truthy` returns a bare `bool` through `&self`, so the dialect
//! cannot refuse an `IF`/`CBRANCH` on a mask. Asked to branch on a
//! non-scalar, it records [`FoldError::BranchOnPopulation`] in a poison flag
//! and answers `false`; the caller must read [`FoldDialect::poison`] after
//! `Interpreter::run()` returns. Detectable, not abortable.
//!
//! # Opcode authority
//!
//! `ogar_r2il` names only its fold-band opcodes; R2IL-band opcodes are
//! addressed by ordinal. The ordinals below are the single local spelling and
//! are pinned against `ogar_r2il::R2ILFn::MNEMONICS` by this crate's tests,
//! so the arity table stays in `ogar_r2il` alone.

pub mod binding;
pub mod binding_native;
pub mod procs;

use std::cell::Cell;

use lance_graph_mask_risc::{
    execute_into, fuse, scratch_words_for, tile_words_for, BoolExpr, CrossPowerSums, Foreign,
    FuseError, GroupKey, LaneKind, MaskOp, Operand, Out, Planes, Pred, Program, Scratch, Terminal,
    Value,
};
use ogar_loco::{Dialect, FnIndex};
use ogar_r2il::{ALL, ANY, GROUP_SUM, KEEP, KEY_RUNS, MAX, MIN, R2IL_BASE, RANGE, SUM, VIA};

// ── R2IL-band opcodes this dialect wires, by their VERIFIED ordinals ───────
//
// `ogar_r2il` does not name each of its 82 R2IL-band opcodes individually
// (only the fold band gets named `pub const`s) — the newtype's whole point
// is that "the ordinal IS the identity". These offsets were confirmed
// directly against `R2ILFn::MNEMONICS`'s declaration order before use, not
// assumed from the spec: `Load`=1, `IntAdd`=9, `IntSub`=10, `IntAnd`=20,
// `IntEqual`=27, `IntLess`=29, `IntSLess`=30, `IntLessEqual`=31,
// `PopCount`=41.
pub const LOAD: FnIndex = FnIndex(R2IL_BASE + 1);
/// `IntAdd` = ordinal 9. Scalar + scalar only (two fold results); checked.
pub const INT_ADD: FnIndex = FnIndex(R2IL_BASE + 9);
pub const INT_SUB: FnIndex = FnIndex(R2IL_BASE + 10);
/// `IntMult` = ordinal 11. Scalar × scalar only; checked.
pub const INT_MULT: FnIndex = FnIndex(R2IL_BASE + 11);
/// `IntSDiv` = ordinal 13. Scalar / scalar only, truncating toward zero (SQL
/// integer division); division by zero is refused.
pub const INT_S_DIV: FnIndex = FnIndex(R2IL_BASE + 13);
pub const INT_AND: FnIndex = FnIndex(R2IL_BASE + 20);
pub const INT_OR: FnIndex = FnIndex(R2IL_BASE + 21);
/// `IntXor` = ordinal 22, arity 2 (confirmed against `R2ILFn::MNEMONICS`).
/// Over two populations it lowers to `MaskOp::Xor`: the rows where exactly
/// one side holds. `ANY` over it answers "do these masks differ".
pub const INT_XOR: FnIndex = FnIndex(R2IL_BASE + 22);
/// `IntNot` = ordinal 23, arity 1 (confirmed against `R2ILFn::MNEMONICS`).
/// Over a population it lowers to `MaskOp::Not`.
pub const INT_NOT: FnIndex = FnIndex(R2IL_BASE + 23);
pub const INT_EQUAL: FnIndex = FnIndex(R2IL_BASE + 27);
pub const INT_LESS: FnIndex = FnIndex(R2IL_BASE + 29);
pub const INT_S_LESS: FnIndex = FnIndex(R2IL_BASE + 30);
pub const INT_LESS_EQUAL: FnIndex = FnIndex(R2IL_BASE + 31);
pub const POP_COUNT: FnIndex = FnIndex(R2IL_BASE + 41);

/// Ceiling on scratch slots any ONE finalized fold program may touch (see
/// the crate doc's § Slot limit; a wider fold is refused). The probe's `gating_only_folds_the_last_ungated_pred` (3 preds + 1 real `And`)
/// is the widest measured body, at 4. Sized with headroom, not tightly, since
/// the whole point of `Scratch::over` is that a shared buffer can be carved
/// smaller than its own capacity for free.
pub const MAX_SLOTS: usize = 8;

// ── the dialect's stack value: never a population ──────────────────────────

/// Where a fold-band `LOAD` (space 0/1/2) or `VIA` points — never row-sized
/// data, only the ADDRESS of a lane or a join.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Addr {
    /// A resident value lane, at `idx` into [`Planes::lanes`].
    Lane { idx: u16, kind: LaneKind },
    /// A join address: `fk` is a `U32` lane of THIS table, `key` indexes
    /// [`Foreign::lanes`] on the OTHER table — the same two fields
    /// [`Pred::EqU32Via`] carries.
    Via { fk: u16, key: u16 },
    /// A RESIDENT mask plane, [`Planes::masks`]`[idx]` — read as an operand
    /// where it stands (`Operand::Plane`), never copied into a slot.
    /// `LOAD` space 3.
    Plane(u16),
}

/// The loco stack's value type. Every variant is either a scalar (a runtime
/// number the loco body computed) or an ADDRESS (a compile-time-shaped
/// description of where a lane or a join lives) — never a mask, never a row.
/// A predicate-producing call pushes [`Val::Slot`], a scratch slot NAME, and
/// nothing else in this crate ever reaches for the population the slot
/// addresses.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Val {
    /// A number computed by the loco body — a literal, a comparison result,
    /// or a fold's finalized-and-executed answer.
    Scalar(i64),
    /// A scratch slot holding a mask, produced by a predicate or a mask op —
    /// together with the EXECUTION EPOCH it was minted in
    /// (the dialect's current epoch at the moment of the push). `finalize_and_run`
    /// discards the `ops` graph a slot number referred to and restarts slot
    /// numbering from 0 on every fold, so a bare `u16` alone could alias a
    /// slot freshly minted in a LATER epoch. The epoch is what lets a
    /// consuming arm tell the two apart and refuse the stale one
    /// ([`FoldError::StaleSlot`]) instead of silently folding it.
    Slot { slot: u16, epoch: u32 },
    /// A resolved address — a lane or a join — not yet combined with a
    /// value into a predicate.
    Address(Addr),
}

/// Why a fold-band call was refused.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FoldError {
    /// A control-flow or non-branching call wanted an operand and the
    /// dialect's own stack was empty.
    Underflow(FnIndex),
    /// The call is covered by [`ogar_r2il::R2ILVocabulary`]'s arity table but the
    /// VALUES on the stack are the wrong shape for it (e.g. a `Slot` where
    /// only `Scalar`/`Address` combinations are meaningful).
    WrongOperandKind(FnIndex),
    /// A fold-band or R2IL-band byte this dialect deliberately does not
    /// implement yet — named individually, never silently coerced onto a
    /// byte that IS implemented.
    Unimplemented(FnIndex),
    /// `IntLess`/`IntLessEqual` (the UNSIGNED compares): mask-risc has no
    /// unsigned lane compare, only the signed `IntSLess`/`IntSLessEqual`
    /// pair — refused for every input, not merely a wrong-shape one.
    NoUnsignedLaneCompare(FnIndex),
    /// `IntAdd`/`IntSub` with BOTH operands addresses: arithmetic between
    /// two lane addresses is not a fold-band operation this dialect knows
    /// how to interpret.
    LaneVersusLane(FnIndex),
    /// `LOAD`'s own immediate named a space outside
    /// `{0=U32, 1=I32, 2=U64, 3=resident mask plane}`.
    UnknownLoadSpace(u8),
    /// `KEEP` ran on a dialect that was not lent a mask sink.
    NoKeepSink,
    /// A [`Val::Slot`] whose epoch does not match the dialect's current epoch was
    /// fed to a consuming arm (`INT_AND`/`POP_COUNT`/`SUM`/`GROUP_SUM`). The slot names a
    /// scratch position in an `ops` graph a PRIOR fold's `finalize_and_run`
    /// already discarded — its number may numerically alias a slot minted
    /// fresh in the current epoch, so it is refused rather than folded.
    StaleSlot,
    /// [`Dialect::truthy`] was asked to branch on a non-`Scalar` value — see
    /// the crate doc's § Branching on a population. Set via [`FoldDialect::poison`],
    /// never returned directly (the trait cannot return an error here).
    BranchOnPopulation,
    /// A finalized fold's `execute_into` call itself refused the program.
    Exec(lance_graph_mask_risc::ExecError),
    /// The fused mode's Boolean expression could not be lowered by mask-risc
    /// `fuse` (it needed a scratch slot past `u16::MAX`).
    Fuse(FuseError),
    /// Scalar `IntSDiv` by zero. PostgreSQL raises `division_by_zero`; the
    /// dialect refuses rather than answer.
    DivisionByZero(FnIndex),
    /// Scalar `IntAdd`/`IntMult` left `i64`. PostgreSQL raises
    /// `numeric_value_out_of_range` for `bigint`; the dialect refuses rather
    /// than wrap.
    Overflow(FnIndex),
}

/// How a body's Boolean calls (`IntAnd`, `IntOr`, `IntNot` on masks) are
/// lowered. See the crate doc's § Boolean modes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum BoolMode {
    /// One mask-risc op per call, written in place. `IntAnd` gates a
    /// just-emitted predicate under its left operand instead of spending an
    /// `And` (survivor gating). The default.
    #[default]
    InPlace,
    /// The calls build one [`BoolExpr`]; at the fold it is lowered through
    /// mask-risc [`fuse()`] into ternlogs. Fewer passes, no survivor gating.
    Fused,
}

/// Slot numbers at or above this name a pending fused expression, not a
/// scratch slot. Real slots never reach it: a fold is limited to
/// [`MAX_SLOTS`].
const VIRTUAL: u16 = 0x8000;

/// Which caller buffer a finalized fold writes into.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum OutKind {
    None,
    Group,
    Keep,
    CrossSums,
}

/// Interprets one loco body, building [`MaskOp`]s and RUNNING them at every
/// scalar-producing fold boundary (see the crate doc's § Execution model).
/// Borrows the planes rather than owning them — legal because
/// `Interpreter::new` takes its dialect by value with no lifetime bound
/// tying it to the interpreter's own `'a`.
pub struct FoldDialect<'p> {
    planes: &'p Planes<'p>,
    foreign: &'p Foreign<'p>,
    /// The `MaskOp`s accumulated since the last fold boundary — NEVER
    /// row-sized; its length is bounded by the number of predicate/mask
    /// calls in the loco body between two folds.
    ops: Vec<MaskOp>,
    /// The next unused scratch slot WITHIN the current `ops` accumulation.
    /// Reset to 0 every time a fold finalizes and runs.
    next_slot: u16,
    /// Slots below [`Self::next_slot`] whose value is dead, released by
    /// [`Self::call_owned`] and handed out again by [`Self::fresh`] before
    /// a new slot is minted. Cleared with `next_slot` at every fold. Only
    /// `call_owned` ever releases a slot, so a caller that never uses it sees
    /// exactly the old numbering.
    free: Vec<u16>,
    /// The current execution epoch — incremented every time a fold
    /// finalizes and runs, at the same point [`Self::next_slot`] resets to
    /// 0. Every [`Val::Slot`] minted carries this value at mint time; a
    /// consuming arm refuses a slot whose epoch does not match the CURRENT
    /// value ([`FoldError::StaleSlot`]) — see [`Val::Slot`]'s doc comment.
    epoch: u32,
    /// Caller-owned scratch for `Scratch::over`, allocated ONCE at
    /// construction to `scratch_words_for(tile_words_for(n_rows), MAX_SLOTS)`
    /// and carved fresh (never re-allocated) at every fold boundary — this
    /// is what keeps allocation row-independent ACROSS folds, and it is the
    /// property `the_dialect_side_allocates_nothing_proportional_to_rows`
    /// pins.
    scratch: Vec<u64>,
    /// Caller-owned `GROUP_SUM` result sink, BORROWED — the same ownership
    /// shape as `mask-risc`'s `Out::I64(&mut [i64])`. Empty unless the dialect
    /// was built by [`Self::with_groups`]; its width is the group universe K,
    /// never the row population N. The caller allocates it once and reads it
    /// back after the run, so the dialect never copies it. `mask-risc`
    /// zero-fills it before each grouped fold, so it always holds the LAST
    /// grouped fold's answer.
    group_sink: &'p mut [i64],
    /// Caller-owned `KEEP` destination, `words_for(n_rows)` long — the
    /// demanded result when a body asks for the mask itself. `None` unless
    /// the dialect was built by [`Self::with_keep`].
    keep_sink: Option<&'p mut [u64]>,
    /// Caller-owned sink for [`FoldDialect::fold_cross_power_sums`], one
    /// [`CrossPowerSums`] per group; empty unless the dialect was built by
    /// [`Self::with_cross_sums`]. Its length is the group universe K.
    cross_sink: &'p mut [CrossPowerSums],
    /// The last fold program this dialect finalized and ran, kept so a test
    /// can compare it physically against `quack::lower`'s. Moved in after
    /// execution, never cloned.
    last_program: Option<Program>,
    /// Total `MaskOp`s across every finalized program this dialect has run —
    /// the "physical facade passes" measurement.
    facade_passes: usize,
    /// How many folds actually finalized-and-ran.
    programs_run: usize,
    /// Set by [`Dialect::truthy`] when asked to branch on a non-`Scalar`
    /// value. `Cell`, not a plain field, because `truthy` takes `&self`.
    poison: Cell<Option<FoldError>>,
    /// [`BoolMode`]; fixed at construction.
    mode: BoolMode,
    /// Fused mode only: the pending Boolean expressions of the current
    /// epoch, named by `VIRTUAL + index`. Cleared at every fold.
    exprs: Vec<BoolExpr>,
    /// Fused mode only: where each pending expression was lowered to, once a
    /// consumer needed it, so a second consumer reuses the same ops.
    lowered: Vec<Option<Operand>>,
    /// Set only for the duration of [`Self::call_shared`]: the operands of
    /// the call stay readable afterwards, so `IntAnd` must not gate the last
    /// predicate in place.
    keep_operands: bool,
}

impl<'p> FoldDialect<'p> {
    /// A fresh dialect over `planes`/`foreign`, with its scratch buffer
    /// allocated ONCE, sized `scratch_words_for(tile_words_for(n_rows),
    /// MAX_SLOTS)` — the allocation
    /// `the_dialect_side_allocates_nothing_proportional_to_rows` pins as
    /// row-independent.
    pub fn new(planes: &'p Planes<'p>, foreign: &'p Foreign<'p>) -> Self {
        let words = tile_words_for(planes.n_rows);
        let cap = scratch_words_for(words, MAX_SLOTS)
            .expect("MAX_SLOTS is a small constant; no overflow");
        Self {
            planes,
            foreign,
            ops: Vec::new(),
            next_slot: 0,
            free: Vec::new(),
            epoch: 0,
            scratch: vec![0u64; cap],
            group_sink: &mut [],
            keep_sink: None,
            cross_sink: &mut [],
            last_program: None,
            facade_passes: 0,
            programs_run: 0,
            poison: Cell::new(None),
            mode: BoolMode::InPlace,
            exprs: Vec::new(),
            lowered: Vec::new(),
            keep_operands: false,
        }
    }

    /// [`Self::new`], writing `GROUP_SUM` results into the caller's `sink`.
    /// Its length is the group universe K; the dialect allocates nothing for
    /// it.
    pub fn with_groups(
        planes: &'p Planes<'p>,
        foreign: &'p Foreign<'p>,
        sink: &'p mut [i64],
    ) -> Self {
        let mut d = Self::new(planes, foreign);
        d.group_sink = sink;
        d
    }

    /// [`Self::new`], writing `KEEP`'s mask into the caller's `sink`
    /// (`words_for(n_rows)` long). The dialect allocates nothing for it.
    pub fn with_keep(
        planes: &'p Planes<'p>,
        foreign: &'p Foreign<'p>,
        sink: &'p mut [u64],
    ) -> Self {
        let mut d = Self::new(planes, foreign);
        d.keep_sink = Some(sink);
        d
    }

    /// [`Self::new`], writing [`Self::fold_cross_power_sums`] results into the
    /// caller's `sink`. Its length is the group universe K; the dialect
    /// allocates nothing for it.
    pub fn with_cross_sums(
        planes: &'p Planes<'p>,
        foreign: &'p Foreign<'p>,
        sink: &'p mut [CrossPowerSums],
    ) -> Self {
        let mut d = Self::new(planes, foreign);
        d.cross_sink = sink;
        d
    }

    /// Grouped cross moments of two `I32` lanes over the mask on top of the
    /// stack: pops the mask, folds `(lanes[x][i], lanes[y][i])` into group
    /// `lanes[key][i]` of the lent sink as `n, Σx, Σy, Σx², Σy², Σxy`
    /// (mask-risc `Terminal::GroupCrossPowerSumsI32`, keyed by a `U32` lane).
    /// Pushes nothing; the sink is the result, like `GROUP_SUM`.
    ///
    /// Not an R2IL byte: the fold band names no cross-moment call. Like
    /// [`Self::emit_pred`] it is a dialect method a procedure calls directly.
    ///
    /// # Errors
    ///
    /// [`FoldError::Underflow`] / [`FoldError::WrongOperandKind`] if the top
    /// of the stack is not a mask, [`FoldError::StaleSlot`] for a slot of an
    /// earlier fold, [`FoldError::Exec`] if mask-risc refuses the program (a
    /// wrong lane kind, an empty sink).
    pub fn fold_cross_power_sums(
        &mut self,
        key: u16,
        x: u16,
        y: u16,
        stack: &mut Vec<Val>,
    ) -> Result<(), FoldError> {
        let m = fold_mask(stack.pop().ok_or(FoldError::Underflow(GROUP_SUM))?)
            .ok_or(FoldError::WrongOperandKind(GROUP_SUM))?;
        let mask = self.resolve_mask(m)?;
        let terminal = Terminal::GroupCrossPowerSumsI32 {
            mask,
            key: GroupKey::Lane(key),
            x,
            y,
        };
        match self.finalize_and_run(terminal, OutKind::CrossSums)? {
            Value::GroupCrossPowerSums => Ok(()),
            other => {
                panic!("cross-moment terminal shape guarantees GroupCrossPowerSums, got {other:?}")
            }
        }
    }

    /// The population-branch poison flag, if [`Dialect::truthy`] ever set
    /// one during the run — read AFTER `Interpreter::run()` returns.
    pub fn poison(&self) -> Option<FoldError> {
        self.poison.get()
    }

    /// Total `MaskOp`s across every finalized fold this dialect has run —
    /// the "physical facade passes" measurement.
    pub fn facade_passes(&self) -> usize {
        self.facade_passes
    }

    /// How many folds finalized and ran.
    pub fn programs_run(&self) -> usize {
        self.programs_run
    }

    /// The last fold program this dialect finalized and ran, for physical
    /// comparison against another lowering. Borrowed, never cloned here.
    pub fn last_program(&self) -> Option<&Program> {
        self.last_program.as_ref()
    }

    /// Length in words of the scratch buffer allocated once at construction
    /// (`scratch_words_for(tile_words_for(n_rows), MAX_SLOTS)`).
    pub fn scratch_len(&self) -> usize {
        self.scratch.len()
    }

    /// This dialect with its Boolean calls lowered in `mode`.
    #[must_use]
    pub fn with_mode(mut self, mode: BoolMode) -> Self {
        self.mode = mode;
        self
    }

    /// The Boolean mode this dialect was built with.
    pub fn mode(&self) -> BoolMode {
        self.mode
    }

    /// [`Dialect::call`] for a caller that reads the operands again after
    /// the call. `IntAnd` normally gates the last emitted predicate in place
    /// under the other operand (survivor gating), which changes what that
    /// predicate's slot holds; here it emits a separate `And` instead. A
    /// stack program that consumes its operands keeps using `call`.
    pub fn call_shared(
        &mut self,
        f: FnIndex,
        v: [u8; 3],
        stack: &mut Vec<Val>,
    ) -> Result<(), FoldError> {
        self.keep_operands = true;
        let r = self.call(f, v, stack);
        self.keep_operands = false;
        r
    }

    /// Drop the predicates and expressions accumulated since the last fold
    /// and advance the epoch, so a slot minted before this call is refused
    /// ([`FoldError::StaleSlot`]). A caller that runs independent programs on
    /// one dialect calls this at each boundary, so one program's unconsumed
    /// predicates neither grow the next program nor take its scratch slots.
    pub fn discard_pending(&mut self) {
        if self.ops.is_empty() && self.exprs.is_empty() && self.next_slot == 0 {
            return;
        }
        self.ops.clear();
        self.next_slot = 0;
        self.free.clear();
        self.epoch += 1;
        self.exprs.clear();
        self.lowered.clear();
    }

    /// The expression a mask operand stands for, for the fused mode.
    fn expr_of(&self, v: Val) -> Result<BoolExpr, FoldError> {
        match v {
            Val::Address(Addr::Plane(p)) => Ok(BoolExpr::Leaf(Operand::Plane(p))),
            Val::Slot { slot, epoch } => {
                if epoch != self.epoch {
                    return Err(FoldError::StaleSlot);
                }
                if slot >= VIRTUAL {
                    self.exprs
                        .get(usize::from(slot - VIRTUAL))
                        .cloned()
                        .ok_or(FoldError::StaleSlot)
                } else {
                    Ok(BoolExpr::Leaf(Operand::Scratch(slot)))
                }
            }
            _ => Err(FoldError::WrongOperandKind(INT_AND)),
        }
    }

    /// Push a pending fused expression as a virtual slot.
    fn push_expr(&mut self, e: BoolExpr, stack: &mut Vec<Val>) {
        let slot = VIRTUAL + u16::try_from(self.exprs.len()).unwrap_or(u16::MAX - VIRTUAL);
        self.exprs.push(e);
        self.lowered.push(None);
        stack.push(Val::Slot {
            slot,
            epoch: self.epoch,
        });
    }

    /// The operand a consumer reads for `slot`. A real slot is itself; a
    /// virtual one is lowered through mask-risc `fuse` the first time it is
    /// consumed, its ternlogs appended to this fold's ops.
    /// Resolve a fold's mask: a computed slot of the current epoch, or a
    /// resident plane read where it stands. Shape is judged by
    /// [`fold_mask`] first, so a wrong-kind operand is reported as such
    /// before the epoch is.
    fn resolve_mask(&mut self, m: FoldMask) -> Result<Operand, FoldError> {
        match m {
            FoldMask::Slot { slot, epoch } => {
                if epoch != self.epoch {
                    return Err(FoldError::StaleSlot);
                }
                self.operand_of(slot)
            }
            FoldMask::Plane(p) => Ok(Operand::Plane(p)),
        }
    }

    fn operand_of(&mut self, slot: u16) -> Result<Operand, FoldError> {
        if slot < VIRTUAL {
            return Ok(Operand::Scratch(slot));
        }
        let i = usize::from(slot - VIRTUAL);
        if let Some(Some(op)) = self.lowered.get(i) {
            return Ok(*op);
        }
        let expr = self.exprs.get(i).ok_or(FoldError::StaleSlot)?;
        let f = fuse(expr, self.next_slot).map_err(FoldError::Fuse)?;
        self.next_slot = f
            .next_slot
            .ok_or(FoldError::Fuse(FuseError::SlotOverflow))?;
        self.ops.extend(f.ops);
        self.lowered[i] = Some(f.result);
        Ok(f.result)
    }

    /// Fused-mode `IntAnd`/`IntOr`: combine two mask operands into one
    /// pending expression.
    fn fuse_binary(
        &mut self,
        a: Val,
        b: Val,
        stack: &mut Vec<Val>,
        build: fn(Box<BoolExpr>, Box<BoolExpr>) -> BoolExpr,
    ) -> Result<(), FoldError> {
        let e = build(Box::new(self.expr_of(a)?), Box::new(self.expr_of(b)?));
        self.push_expr(e, stack);
        Ok(())
    }

    /// The next unused scratch slot within the current `ops` accumulation,
    /// advancing [`Self::next_slot`] by one.
    fn fresh(&mut self) -> u16 {
        if let Some(s) = self.free.pop() {
            return s;
        }
        let s = self.next_slot;
        self.next_slot += 1;
        s
    }

    /// [`Dialect::call`] for a caller that OWNS the masks it combines.
    ///
    /// For `IntOr`, `IntAnd` and `IntNot` in [`BoolMode::InPlace`], the slots
    /// of the consumed operands are released once the op is emitted (unless
    /// one of them is the result), and [`Self::fresh`] hands them out again.
    /// An `Or` or `And` chain over `k` atoms then needs three slots, not
    /// `2k - 1`. Every other call, and every call in [`BoolMode::Fused`], is
    /// plain [`Dialect::call`].
    ///
    /// # Contract
    ///
    /// The caller guarantees that no other [`Val`] anywhere (on this stack, on
    /// another stack, or held aside) names the consumed slots. [`Val`] is
    /// `Copy`, so the dialect cannot check this: a copy that survives the call
    /// would read whatever a later op writes into the released slot. A
    /// lowering that consumes each subexpression's value exactly once, as
    /// `lance-graph-relalg` does, meets the contract by construction. A
    /// stack program that can duplicate values must use `call`.
    pub fn call_owned(
        &mut self,
        f: FnIndex,
        v: [u8; 3],
        stack: &mut Vec<Val>,
    ) -> Result<(), FoldError> {
        let arity = match f {
            INT_OR | INT_AND => 2,
            INT_NOT => 1,
            _ => return self.call(f, v, stack),
        };
        if self.mode != BoolMode::InPlace || stack.len() < arity {
            return self.call(f, v, stack);
        }
        let epoch = self.epoch;
        let consumed: Vec<u16> = stack[stack.len() - arity..]
            .iter()
            .filter_map(|v| match *v {
                Val::Slot { slot, epoch: e } if e == epoch => Some(slot),
                _ => None,
            })
            .collect();
        self.call(f, v, stack)?;
        let result = match stack.last() {
            Some(&Val::Slot { slot, epoch: e }) if e == self.epoch => Some(slot),
            _ => None,
        };
        for s in consumed {
            if Some(s) != result && !self.free.contains(&s) {
                self.free.push(s);
            }
        }
        Ok(())
    }

    /// Survivor gating (verbatim from W0C, restated over `Operand`): if `b`
    /// was produced by the op emitted LAST and that op is an ungated `Pred`,
    /// gate it under `a` instead of spending a facade pass on `And`.
    fn and_peephole(&mut self, a: Operand, b: Operand) -> u16 {
        if self.keep_operands {
            let dst = self.fresh();
            self.ops.push(MaskOp::And { a, b, dst });
            return dst;
        }
        if let (
            Operand::Scratch(bs),
            Some(MaskOp::Pred {
                under: under @ None,
                dst,
                ..
            }),
        ) = (b, self.ops.last_mut())
        {
            if *dst == bs {
                *under = Some(a);
                return bs;
            }
        }
        let dst = self.fresh();
        self.ops.push(MaskOp::And { a, b, dst });
        dst
    }

    /// Emit a predicate, push its scratch slot — carrying the CURRENT
    /// epoch, per [`Val::Slot`]'s contract.
    ///
    /// Public so a caller can mint a slot from a mask-risc [`Pred`] without
    /// going through an R2IL byte; the probe's epoch tests use it to hold a
    /// slot across a fold boundary.
    pub fn emit_pred(&mut self, pred: Pred, stack: &mut Vec<Val>) {
        let dst = self.fresh();
        self.ops.push(MaskOp::Pred {
            pred,
            under: None,
            dst,
        });
        stack.push(Val::Slot {
            slot: dst,
            epoch: self.epoch,
        });
    }

    /// Finalize the ops accumulated so far into a `Program` with `terminal`,
    /// RUN it over this dialect's borrowed planes, and reset for the next
    /// fold. This is the crate doc's § Execution model: a
    /// scalar-producing fold executes here, not merely lowers.
    ///
    /// `grouped` selects the output: `false` → `Out::None` (scalar
    /// terminals), `true` → `Out::I64` over [`Self::group_sink`]. It is a
    /// parameter rather than a second finalize function so the slot-epoch
    /// boundary below exists in exactly one place for every terminal.
    fn finalize_and_run(&mut self, terminal: Terminal, out: OutKind) -> Result<Value, FoldError> {
        let ops = std::mem::take(&mut self.ops);
        self.next_slot = 0;
        self.free.clear();
        self.epoch += 1;
        self.exprs.clear();
        self.lowered.clear();
        self.facade_passes += ops.len();
        self.programs_run += 1;
        let program = Program::new(ops, terminal);
        let words = tile_words_for(self.planes.n_rows);
        let mut scratch = Scratch::over(&mut self.scratch, words, program.scratch_slots as usize)
            .map_err(FoldError::Exec)?;
        let out = match out {
            OutKind::None => Out::None,
            OutKind::Group => Out::I64(self.group_sink),
            OutKind::Keep => Out::Mask(self.keep_sink.as_deref_mut().ok_or(FoldError::NoKeepSink)?),
            OutKind::CrossSums => Out::CrossPowerSums(self.cross_sink),
        };
        let value = execute_into(&program, self.planes, self.foreign, &mut scratch, out)
            .map_err(FoldError::Exec);
        self.last_program = Some(program);
        value
    }

    /// [`Self::finalize_and_run`], asserting the terminal's known result
    /// shape (`Terminal::Count` → `Value::Count`, `Terminal::MaskedSumI32` →
    /// `Value::SumI64`, `Terminal::Any`/`All` → `Value::Bool` as 0 or 1) — a caller-side invariant, not a new fallible path:
    /// every call site below passes a terminal whose `Value` shape is fixed
    /// by construction.
    fn finalize_and_run_scalar(&mut self, terminal: Terminal) -> Result<i64, FoldError> {
        match self.finalize_and_run(terminal, OutKind::None)? {
            Value::Count(n) => Ok(i64::try_from(n).unwrap_or(i64::MAX)),
            Value::SumI64(n) => Ok(n),
            Value::Bool(b) => Ok(i64::from(b)),
            other => panic!("terminal shape guarantees Count, SumI64 or Bool, got {other:?}"),
        }
    }
}

/// A value that names a mask: a scratch slot or a resident plane.
fn is_mask(v: Val) -> bool {
    matches!(v, Val::Slot { .. } | Val::Address(Addr::Plane(_)))
}

impl Dialect for FoldDialect<'_> {
    type Value = Val;
    type Error = FoldError;

    /// A `Slot` (or any non-`Scalar`) poisons rather than answering — see
    /// the crate doc's § Branching on a population. Bodies that legitimately
    /// branch always branch on a `Val::Scalar`.
    fn truthy(&self, v: &Val) -> bool {
        match v {
            Val::Scalar(s) => *s != 0,
            _ => {
                self.poison.set(Some(FoldError::BranchOnPopulation));
                false
            }
        }
    }

    /// Always `0` — no fold body uses `REPEAT`; the loop bound
    /// this hook would answer for a mask's "count" is never exercised here.
    fn repeat_count(&self, _: &Val) -> u32 {
        0
    }

    /// Execute one non-branching call: pop its operands off the loco stack,
    /// push its result. The whole dispatch table this crate wires — see the
    /// crate doc's § "Implemented, and refused by name"
    /// for the exhaustive list of which of the twelve fold-band opcodes and
    /// which R2IL-band opcodes each arm below covers.
    fn call(&mut self, f: FnIndex, v: [u8; 3], stack: &mut Vec<Val>) -> Result<(), FoldError> {
        match f {
            FnIndex::NUMBER => {
                stack.push(Val::Scalar(i64::from(v[0])));
                Ok(())
            }
            LOAD => {
                let idx = stack.pop().ok_or(FoldError::Underflow(f))?;
                let Val::Scalar(idx) = idx else {
                    return Err(FoldError::WrongOperandKind(f));
                };
                let kind = match v[0] {
                    0 => LaneKind::U32,
                    1 => LaneKind::I32,
                    2 => LaneKind::U64,
                    3 => {
                        stack.push(Val::Address(Addr::Plane(
                            u16::try_from(idx).unwrap_or(u16::MAX),
                        )));
                        return Ok(());
                    }
                    other => return Err(FoldError::UnknownLoadSpace(other)),
                };
                stack.push(Val::Address(Addr::Lane {
                    idx: u16::try_from(idx).unwrap_or(u16::MAX),
                    kind,
                }));
                Ok(())
            }
            VIA => {
                // `(Scalar key, Scalar fk)`: `key` was pushed first (LHS,
                // popped second); `fk` was pushed second (RHS, popped
                // first) — the ogar_r2il fold-band doc's documented order.
                let fk = stack.pop().ok_or(FoldError::Underflow(f))?;
                let key = stack.pop().ok_or(FoldError::Underflow(f))?;
                match (key, fk) {
                    (Val::Scalar(key), Val::Scalar(fk)) => {
                        stack.push(Val::Address(Addr::Via {
                            fk: u16::try_from(fk).unwrap_or(u16::MAX),
                            key: u16::try_from(key).unwrap_or(u16::MAX),
                        }));
                        Ok(())
                    }
                    _ => Err(FoldError::WrongOperandKind(f)),
                }
            }
            INT_EQUAL => {
                let b = stack.pop().ok_or(FoldError::Underflow(f))?;
                let a = stack.pop().ok_or(FoldError::Underflow(f))?;
                // Symmetric: the address may have been pushed first or
                // second, and equality does not care which.
                match (a, b) {
                    (
                        Val::Address(Addr::Lane {
                            idx,
                            kind: LaneKind::U32,
                        }),
                        Val::Scalar(x),
                    )
                    | (
                        Val::Scalar(x),
                        Val::Address(Addr::Lane {
                            idx,
                            kind: LaneKind::U32,
                        }),
                    ) => {
                        self.emit_pred(
                            Pred::EqU32 {
                                lane: idx,
                                v: u32::try_from(x).unwrap_or(u32::MAX),
                            },
                            stack,
                        );
                        Ok(())
                    }
                    (
                        Val::Address(Addr::Lane {
                            idx,
                            kind: LaneKind::I32,
                        }),
                        Val::Scalar(x),
                    )
                    | (
                        Val::Scalar(x),
                        Val::Address(Addr::Lane {
                            idx,
                            kind: LaneKind::I32,
                        }),
                    ) => {
                        self.emit_pred(
                            Pred::EqI32 {
                                lane: idx,
                                v: i32::try_from(x).unwrap_or(i32::MAX),
                            },
                            stack,
                        );
                        Ok(())
                    }
                    (Val::Address(Addr::Via { fk, key }), Val::Scalar(x))
                    | (Val::Scalar(x), Val::Address(Addr::Via { fk, key })) => {
                        self.emit_pred(
                            Pred::EqU32Via {
                                fk,
                                key,
                                v: u32::try_from(x).unwrap_or(u32::MAX),
                            },
                            stack,
                        );
                        Ok(())
                    }
                    (Val::Scalar(x), Val::Scalar(y)) => {
                        stack.push(Val::Scalar(i64::from(x == y)));
                        Ok(())
                    }
                    _ => Err(FoldError::WrongOperandKind(f)),
                }
            }
            INT_LESS | INT_LESS_EQUAL => Err(FoldError::NoUnsignedLaneCompare(f)),
            INT_S_LESS => {
                // Asymmetric: `a` is the LHS (pushed first, popped second),
                // `b` the RHS (pushed second, popped first). `(Addr, Scalar)`
                // reads "lane < t"; `(Scalar, Addr)` reverses to "t < lane",
                // stored as `GtI32` since `Pred` has no `t < lane` shape.
                let b = stack.pop().ok_or(FoldError::Underflow(f))?;
                let a = stack.pop().ok_or(FoldError::Underflow(f))?;
                match (a, b) {
                    (
                        Val::Address(Addr::Lane {
                            idx,
                            kind: LaneKind::I32,
                        }),
                        Val::Scalar(t),
                    ) => {
                        self.emit_pred(
                            Pred::LtI32 {
                                lane: idx,
                                t: i32::try_from(t).unwrap_or(i32::MAX),
                            },
                            stack,
                        );
                        Ok(())
                    }
                    (
                        Val::Scalar(t),
                        Val::Address(Addr::Lane {
                            idx,
                            kind: LaneKind::I32,
                        }),
                    ) => {
                        self.emit_pred(
                            Pred::GtI32 {
                                lane: idx,
                                t: i32::try_from(t).unwrap_or(i32::MAX),
                            },
                            stack,
                        );
                        Ok(())
                    }
                    _ => Err(FoldError::WrongOperandKind(f)),
                }
            }
            INT_AND => {
                let b = stack.pop().ok_or(FoldError::Underflow(f))?;
                let a = stack.pop().ok_or(FoldError::Underflow(f))?;
                if self.mode == BoolMode::Fused && is_mask(a) && is_mask(b) {
                    return self.fuse_binary(a, b, stack, BoolExpr::And);
                }
                match (a, b) {
                    (
                        Val::Slot {
                            slot: sa,
                            epoch: ea,
                        },
                        Val::Slot {
                            slot: sb,
                            epoch: eb,
                        },
                    ) => {
                        if ea != self.epoch || eb != self.epoch {
                            return Err(FoldError::StaleSlot);
                        }
                        let dst = self.and_peephole(Operand::Scratch(sa), Operand::Scratch(sb));
                        stack.push(Val::Slot {
                            slot: dst,
                            epoch: self.epoch,
                        });
                        Ok(())
                    }
                    // A resident plane ANDed with a slot: the plane is the
                    // peephole's left operand, so a last ungated pred is gated
                    // UNDER the plane (the survivor skip) instead of spending
                    // an `And` pass.
                    (Val::Address(Addr::Plane(p)), Val::Slot { slot, epoch })
                    | (Val::Slot { slot, epoch }, Val::Address(Addr::Plane(p))) => {
                        if epoch != self.epoch {
                            return Err(FoldError::StaleSlot);
                        }
                        let dst = self.and_peephole(Operand::Plane(p), Operand::Scratch(slot));
                        stack.push(Val::Slot {
                            slot: dst,
                            epoch: self.epoch,
                        });
                        Ok(())
                    }
                    // Two resident planes: nothing to gate, one `And` pass.
                    (Val::Address(Addr::Plane(pa)), Val::Address(Addr::Plane(pb))) => {
                        let dst = self.fresh();
                        self.ops.push(MaskOp::And {
                            a: Operand::Plane(pa),
                            b: Operand::Plane(pb),
                            dst,
                        });
                        stack.push(Val::Slot {
                            slot: dst,
                            epoch: self.epoch,
                        });
                        Ok(())
                    }
                    (Val::Scalar(x), Val::Scalar(y)) => {
                        stack.push(Val::Scalar(x & y));
                        Ok(())
                    }
                    _ => Err(FoldError::WrongOperandKind(f)),
                }
            }
            // Like `IntAnd` without the survivor gate: an `Or` cannot gate its
            // right operand under its left, so every in-place `IntOr` on masks
            // is one `Or` pass.
            INT_OR => {
                let b = stack.pop().ok_or(FoldError::Underflow(f))?;
                let a = stack.pop().ok_or(FoldError::Underflow(f))?;
                if let (Val::Scalar(x), Val::Scalar(y)) = (a, b) {
                    stack.push(Val::Scalar(x | y));
                    return Ok(());
                }
                if !(is_mask(a) && is_mask(b)) {
                    return Err(FoldError::WrongOperandKind(f));
                }
                if self.mode == BoolMode::Fused {
                    return self.fuse_binary(a, b, stack, BoolExpr::Or);
                }
                let operand = |v: Val, epoch_now: u32| match v {
                    Val::Address(Addr::Plane(p)) => Ok(Operand::Plane(p)),
                    Val::Slot { slot, epoch } if epoch == epoch_now => Ok(Operand::Scratch(slot)),
                    _ => Err(FoldError::StaleSlot),
                };
                let (a, b) = (operand(a, self.epoch)?, operand(b, self.epoch)?);
                let dst = self.fresh();
                self.ops.push(MaskOp::Or { a, b, dst });
                stack.push(Val::Slot {
                    slot: dst,
                    epoch: self.epoch,
                });
                Ok(())
            }
            // Like `IntOr`: one `Xor` pass in place, a `BoolExpr::Xor` when fused.
            INT_XOR => {
                let b = stack.pop().ok_or(FoldError::Underflow(f))?;
                let a = stack.pop().ok_or(FoldError::Underflow(f))?;
                if let (Val::Scalar(x), Val::Scalar(y)) = (a, b) {
                    stack.push(Val::Scalar(x ^ y));
                    return Ok(());
                }
                if !(is_mask(a) && is_mask(b)) {
                    return Err(FoldError::WrongOperandKind(f));
                }
                if self.mode == BoolMode::Fused {
                    return self.fuse_binary(a, b, stack, BoolExpr::Xor);
                }
                let operand = |v: Val, epoch_now: u32| match v {
                    Val::Address(Addr::Plane(p)) => Ok(Operand::Plane(p)),
                    Val::Slot { slot, epoch } if epoch == epoch_now => Ok(Operand::Scratch(slot)),
                    _ => Err(FoldError::StaleSlot),
                };
                let (a, b) = (operand(a, self.epoch)?, operand(b, self.epoch)?);
                let dst = self.fresh();
                self.ops.push(MaskOp::Xor { a, b, dst });
                stack.push(Val::Slot {
                    slot: dst,
                    epoch: self.epoch,
                });
                Ok(())
            }
            // Scalar arithmetic between fold results (`AVG` = `SUM` /
            // `COUNT`, a bag product |L|·|R|). A lane operand has no scalar
            // reading here: two lanes are `LaneVersusLane`, anything else
            // `WrongOperandKind`.
            INT_ADD | INT_MULT | INT_S_DIV => {
                let b = stack.pop().ok_or(FoldError::Underflow(f))?;
                let a = stack.pop().ok_or(FoldError::Underflow(f))?;
                match (a, b) {
                    (Val::Scalar(x), Val::Scalar(y)) => {
                        let r = match f {
                            INT_ADD => x.checked_add(y).ok_or(FoldError::Overflow(f))?,
                            INT_MULT => x.checked_mul(y).ok_or(FoldError::Overflow(f))?,
                            _ if y == 0 => return Err(FoldError::DivisionByZero(f)),
                            _ => x.checked_div(y).ok_or(FoldError::Overflow(f))?,
                        };
                        stack.push(Val::Scalar(r));
                        Ok(())
                    }
                    (Val::Address(_), Val::Address(_)) => Err(FoldError::LaneVersusLane(f)),
                    _ => Err(FoldError::WrongOperandKind(f)),
                }
            }
            INT_SUB => {
                let b = stack.pop().ok_or(FoldError::Underflow(f))?;
                let a = stack.pop().ok_or(FoldError::Underflow(f))?;
                match (a, b) {
                    (Val::Scalar(x), Val::Scalar(y)) => {
                        let r = x.checked_sub(y).ok_or(FoldError::Overflow(f))?;
                        stack.push(Val::Scalar(r));
                        Ok(())
                    }
                    (Val::Address(_), Val::Address(_)) => Err(FoldError::LaneVersusLane(f)),
                    _ => Err(FoldError::WrongOperandKind(f)),
                }
            }
            POP_COUNT => {
                let m = fold_mask(stack.pop().ok_or(FoldError::Underflow(f))?)
                    .ok_or(FoldError::WrongOperandKind(f))?;
                let mask = self.resolve_mask(m)?;
                let n = self.finalize_and_run_scalar(Terminal::Count { mask })?;
                stack.push(Val::Scalar(n));
                Ok(())
            }
            // `COUNT(DISTINCT key)` under a mask, on a KEY-ORDERED `U32` lane:
            // one per run of equal keys holding a selected row
            // (`Terminal::CountKeyRunsU32`, O(1) state, no seen-set). An
            // unordered lane is refused by mask-risc (`LaneNotOrdered`),
            // never answered. Operands: mask, then the key lane.
            KEY_RUNS => {
                let key = stack.pop().ok_or(FoldError::Underflow(f))?;
                let m = fold_mask(stack.pop().ok_or(FoldError::Underflow(f))?)
                    .ok_or(FoldError::WrongOperandKind(f))?;
                let Val::Address(Addr::Lane {
                    idx: lane,
                    kind: LaneKind::U32,
                }) = key
                else {
                    return Err(FoldError::WrongOperandKind(f));
                };
                let mask = self.resolve_mask(m)?;
                let n = self.finalize_and_run_scalar(Terminal::CountKeyRunsU32 { mask, lane })?;
                stack.push(Val::Scalar(n));
                Ok(())
            }
            // The Boolean reduction terminals: is any row selected, are all
            // `n_rows` selected. mask-risc's `Any`/`All` stop at the first
            // deciding word; the answer is pushed as 1 or 0.
            ANY | ALL => {
                let m = fold_mask(stack.pop().ok_or(FoldError::Underflow(f))?)
                    .ok_or(FoldError::WrongOperandKind(f))?;
                let mask = self.resolve_mask(m)?;
                let terminal = if f == ANY {
                    Terminal::Any { mask }
                } else {
                    Terminal::All { mask }
                };
                let b = self.finalize_and_run_scalar(terminal)?;
                stack.push(Val::Scalar(b));
                Ok(())
            }
            SUM => {
                let b = stack.pop().ok_or(FoldError::Underflow(f))?;
                let a = stack.pop().ok_or(FoldError::Underflow(f))?;
                let i32_lane = |v: Val| match v {
                    Val::Address(Addr::Lane {
                        idx,
                        kind: LaneKind::I32,
                    }) => Some(idx),
                    _ => None,
                };
                // Either order: mask then lane, or lane then mask.
                let (m, lane) = match (fold_mask(a), i32_lane(b), i32_lane(a), fold_mask(b)) {
                    (Some(m), Some(lane), _, _) | (_, _, Some(lane), Some(m)) => (m, lane),
                    _ => return Err(FoldError::WrongOperandKind(f)),
                };
                let mask = self.resolve_mask(m)?;
                let n = self.finalize_and_run_scalar(Terminal::MaskedSumI32 { mask, lane })?;
                stack.push(Val::Scalar(n));
                Ok(())
            }
            MIN | MAX => {
                let b = stack.pop().ok_or(FoldError::Underflow(f))?;
                let a = stack.pop().ok_or(FoldError::Underflow(f))?;
                let i32_lane = |v: Val| match v {
                    Val::Address(Addr::Lane {
                        idx,
                        kind: LaneKind::I32,
                    }) => Some(idx),
                    _ => None,
                };
                // Either order, as `SUM`.
                let (m, lane) = match (fold_mask(a), i32_lane(b), i32_lane(a), fold_mask(b)) {
                    (Some(m), Some(lane), _, _) | (_, _, Some(lane), Some(m)) => (m, lane),
                    _ => return Err(FoldError::WrongOperandKind(f)),
                };
                let mask = self.resolve_mask(m)?;
                let (terminal, empty) = if f == MIN {
                    (Terminal::MaskedMinI32 { mask, lane }, i64::MAX)
                } else {
                    (Terminal::MaskedMaxI32 { mask, lane }, i64::MIN)
                };
                let n = match self.finalize_and_run(terminal, OutKind::None)? {
                    Value::OptI32(v) => v.map_or(empty, i64::from),
                    other => panic!("terminal shape guarantees OptI32, got {other:?}"),
                };
                stack.push(Val::Scalar(n));
                Ok(())
            }
            GROUP_SUM => {
                // Stack spelling: mask, key-address, value-address. The key
                // ADDRESS picks the physical terminal: a resident U32 lane is
                // `GroupSumI32`, a `VIA` address is `GroupSumViaI32`. Any
                // other key shape is refused, never coerced onto one of them.
                let val = stack.pop().ok_or(FoldError::Underflow(f))?;
                let key = stack.pop().ok_or(FoldError::Underflow(f))?;
                let mask = stack.pop().ok_or(FoldError::Underflow(f))?;
                let m = fold_mask(mask).ok_or(FoldError::WrongOperandKind(f))?;
                let Val::Address(Addr::Lane {
                    idx: val,
                    kind: LaneKind::I32,
                }) = val
                else {
                    return Err(FoldError::WrongOperandKind(f));
                };
                // The key shape is checked first, so a wrong-kind body is
                // reported as wrong-kind.
                let via = match key {
                    Val::Address(Addr::Lane {
                        idx,
                        kind: LaneKind::U32,
                    }) => (idx, None),
                    Val::Address(Addr::Via { fk, key }) => (key, Some(fk)),
                    _ => return Err(FoldError::WrongOperandKind(f)),
                };
                // Then the epoch, BEFORE a fused expression is lowered and
                // before the fold, like SUM's arm.
                let mask = self.resolve_mask(m)?;
                let terminal = match via {
                    (key, None) => Terminal::GroupSumI32 { mask, key, val },
                    (key, Some(fk)) => Terminal::GroupSumViaI32 { mask, fk, key, val },
                };
                match self.finalize_and_run(terminal, OutKind::Group)? {
                    Value::GroupSummed => Ok(()),
                    other => {
                        panic!("GROUP_SUM terminal shape guarantees GroupSummed, got {other:?}")
                    }
                }
            }
            INT_NOT => {
                let a = stack.pop().ok_or(FoldError::Underflow(f))?;
                if self.mode == BoolMode::Fused && is_mask(a) {
                    let e = BoolExpr::Not(Box::new(self.expr_of(a)?));
                    self.push_expr(e, stack);
                    return Ok(());
                }
                let operand = match a {
                    Val::Scalar(x) => {
                        stack.push(Val::Scalar(!x));
                        return Ok(());
                    }
                    Val::Address(Addr::Plane(p)) => Operand::Plane(p),
                    Val::Slot { slot, epoch } => {
                        if epoch != self.epoch {
                            return Err(FoldError::StaleSlot);
                        }
                        Operand::Scratch(slot)
                    }
                    Val::Address(_) => return Err(FoldError::WrongOperandKind(f)),
                };
                let dst = self.fresh();
                self.ops.push(MaskOp::Not { a: operand, dst });
                stack.push(Val::Slot {
                    slot: dst,
                    epoch: self.epoch,
                });
                Ok(())
            }
            // Arity 0: the population constructor takes its bounds from the
            // call's immediates (`lo = v[0]`, `hi = v[1]`, so 0..=255 each)
            // and lowers to `Pred::Range`, which reads no lane. mask-risc
            // validates `lo <= hi <= n_rows` before executing.
            RANGE => {
                self.emit_pred(
                    Pred::Range {
                        lo: u32::from(v[0]),
                        hi: u32::from(v[1]),
                    },
                    stack,
                );
                Ok(())
            }
            // Arity 1, pushes nothing: the mask itself is the demanded result,
            // written into the caller's `keep_sink` by `Terminal::Keep`.
            KEEP => {
                let m = fold_mask(stack.pop().ok_or(FoldError::Underflow(f))?)
                    .ok_or(FoldError::WrongOperandKind(f))?;
                let mask = self.resolve_mask(m)?;
                match self.finalize_and_run(Terminal::Keep { mask }, OutKind::Keep)? {
                    Value::Mask(_) => Ok(()),
                    other => panic!("KEEP terminal shape guarantees Mask, got {other:?}"),
                }
            }
            other => Err(FoldError::Unimplemented(other)),
        }
    }
}

/// A fold's mask operand, by shape only: a computed slot or a resident plane.
#[derive(Clone, Copy)]
enum FoldMask {
    Slot { slot: u16, epoch: u32 },
    Plane(u16),
}

fn fold_mask(v: Val) -> Option<FoldMask> {
    match v {
        Val::Slot { slot, epoch } => Some(FoldMask::Slot { slot, epoch }),
        Val::Address(Addr::Plane(p)) => Some(FoldMask::Plane(p)),
        _ => None,
    }
}
