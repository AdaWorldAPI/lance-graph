//! **W1b — the fold `Dialect`: loco programs that EXECUTE at fold boundaries,
//! over OGAR's real `0xE2..=0xED` vocabulary.**
//!
//! This file supersedes `tests/row_bridge.rs`'s prior content (the W0C probe,
//! whose own doc comment is quoted below for provenance). That version
//! declared `MaskFoldVocabulary` — a probe-local arity table with packed
//! `EQ_VIA fk,key,v` immediates — which is exactly the invent-a-local-
//! vocabulary shape W1 exists to remove: a second arity table in the tree,
//! next to `ogar_r2il::R2ILVocabulary`'s real one. This file drops it and
//! addresses the SAME operations through the codebook `ogar_r2il` now ships
//! (`FOLD_BASE = 0xE2`, `FOLD_OPS = 12`, twelve named `FnIndex` constants,
//! plus the pre-existing R2IL band `0x90..=0xE1` for `Load`/`IntEqual`/
//! `IntSLess`/`IntAnd`/`IntSub`/`PopCount`, …). One arity table in the tree,
//! not two — that is the entire point of this rewrite.
//!
//! W0C's own measurements carry forward unchanged (see § Measurements
//! below): the survivor-gating peephole still folds a join predicate and a
//! local compare to **2** physical facade passes against quack's **3**, and
//! the loco side still allocates nothing proportional to row count. What is
//! NEW here is the refinement a single-terminal `Program` cannot express.
//!
//! # THE REFINEMENT: a scalar-producing fold must EXECUTE, not just lower
//!
//! A mask-risc [`Program`] has exactly ONE terminal. Mathcad's
//! `Margin := SUM(revenue | m) - SUM(cost | m)` needs TWO folds and a scalar
//! subtract, so it cannot be one `Program`. The multi-body route is closed
//! too: every R2IL op has `body_refs = 0` (§ "The fold extension band" in
//! `ogar_r2il`), so no op under this vocabulary branches to another body.
//!
//! So: when a scalar-producing fold op fires (`SUM`/`POP_COUNT`/…), the
//! dialect FINALIZES the [`MaskOp`]s accumulated so far into a `Program`,
//! RUNS it via [`execute_into`], and pushes the real result as a
//! `Val::Scalar`. A later scalar op (`INT_SUB`) then combines two such
//! results. This does not weaken "loco composes, mask-risc executes" — the
//! dialect still *calls* `execute_into`, mask-risc still executes and the
//! fold is still tile-fused. What changes is only WHEN: at each fold
//! boundary rather than once at the end of the whole body. A worksheet is a
//! SEQUENCE of folds joined by scalar arithmetic, and that is exactly what
//! this shape expresses. Nothing row-sized ever reaches the loco stack: only
//! [`Val::Scalar`], [`Val::Slot`] (a scratch slot NAME) and [`Val::Address`]
//! do.
//!
//! Consequence for the type: the dialect BORROWS the planes.
//! `Interpreter::new` takes `dialect: D` by VALUE and carries no lifetime
//! bound tying `D` to the interpreter's own `'a`, so [`FoldDialect`] holding
//! `&'p Planes<'p>` / `&'p Foreign<'p>` is legal — confirmed against the real
//! `struct Interpreter<'a, V: Vocabulary, D: Dialect> { .. dialect: D .. }`
//! before writing a line of this file, not assumed from the spec.
//!
//! # What is implemented, and what is refused BY NAME
//!
//! Of the twelve `0xE2..=0xED` fold-band opcodes: **`VIA`** (the join-key
//! address constructor) and **`SUM`** (the reduction this file's three
//! frontends actually need) are wired. The other ten — `RANGE`, `MIN`,
//! `MAX`, `GROUP_SUM`, `KEY_RUNS`, `ANY`, `ALL`, `KEEP`, `SCATTER_OR`,
//! `BLEND` — are refused as [`FoldError::Unimplemented`], named individually
//! in the refusal, never silently coerced into one of the two that exist.
//! Landing any of them without its own falsifier would be the same enum-
//! explosion `ogar_r2il`'s own `ARITY` doc warns against ("a first draft…
//! invented nine variants from memory").
//!
//! Of the R2IL band: `Load`, `IntSub` (scalar/scalar only — `Addr`/`Addr` is
//! a NAMED refusal, not an omission), `IntAnd` (with the survivor-gating
//! peephole), `IntEqual`, `IntSLess`, and `PopCount` are wired. `IntAdd` is
//! refused as `Unimplemented` (nothing in the three frontends below needs
//! it — the Mathcad case only ever SUBTRACTS two folds). `IntLess` and
//! `IntSLessEqual`'s unsigned sibling `IntLessEqual` are refused by NAME as
//! [`FoldError::NoUnsignedLaneCompare`], because mask-risc — this dialect's
//! target machine — has no unsigned lane compare (`ogar_r2il`'s own module
//! doc states this explicitly; it is the mechanical REASON the refusal
//! exists, not a gap this file left open). `IntSLessEqual`, `IntNotEqual`,
//! `IntOr`, `IntXor`, `IntNot` are untested and fall to the generic
//! `Unimplemented` catch-all — none of the three frontends needs them, and
//! landing them ahead of a falsifier would be exactly the anti-pattern this
//! doc already names twice.
//!
//! # Immediate ranges — one decode, so one range, stated exactly
//!
//! `FnIndex::NUMBER`'s call value is decoded exactly once in this file
//! (`Val::Scalar(i64::from(v[0]))`), and it is UNCONDITIONALLY UNSIGNED —
//! `v[0]` is a plain `u8`, so the range is **0..=255** for every `NUMBER`
//! literal below, whether it later feeds an unsigned path (`INT_EQUAL`'s
//! `v`, a lane index for `LOAD`/`VIA`) or a SIGNED one (`INT_S_LESS`'s `t`,
//! cast `i32::try_from(t)` after the fact — a value already in 0..=255
//! decodes as the same non-negative `i32`). This file never decodes a
//! signed inline byte (no `as i8`, no two's-complement reinterpretation
//! anywhere in [`FoldDialect::call`]), so every literal used by the three
//! frontends and every test below (`V = 3`, `T = 17`, `POSTED = 2`, every
//! lane index, every `LOAD` space) is chosen to stay non-negative and under
//! 256 — a real constraint this file ran into directly: seeing
//! `gating_only_folds_the_last_ungated_pred`'s own doc comment for the
//! `NUMBER:(T - 40)` byte that silently became 233 instead of -23.
//! `FnIndex::CONSTANT` (the pool load, for a literal that needs to be
//! negative or ≥ 256) is NOT wired here — every literal this file needs
//! fits in `NUMBER`'s 0..=255, so `CONSTANT` is left as one more
//! `Unimplemented` byte, not a gap load-bearing for anything below.
//!
//! **Deviation from the W1b spec's literal `FoldDialect` shape:** the spec
//! lists a `group_sink: Vec<i64>` field "sized by `groups` at construction".
//! It is intentionally OMITTED here: `GROUP_SUM` is one of the ten refused
//! ops, so nothing ever writes to it, and an unused field fails
//! `cargo clippy -- -D warnings` (`dead_code`) in this crate's gate. It is
//! not a functional gap — reintroducing it is exactly the work of landing
//! `GROUP_SUM` for real, with its own falsifier, per the rule above.
//!
//! # MEASURED GAP: a branch on a population is DETECTABLE, not ABORTABLE
//!
//! `Dialect::truthy(&self, value: &Self::Value) -> bool` is a bare `bool`
//! through `&self`. It cannot return an error and it cannot take
//! `&mut self`. `interpret.rs` calls it from `IF` (`run_branching`, the
//! `FnIndex::IF` arm), `IF_ELSE`, and the `WHILE`/`REPEAT_UNTIL` loop — all
//! inside `run_branching`, which never routes those bytes to `Dialect::call`
//! (confirmed by reading `interpret.rs` directly: `run_body` special-cases
//! `is_engine_control(f)` and dispatches straight to `run_branching`, never
//! to `self.dialect.call`). So **the fold dialect cannot refuse a
//! `CBRANCH`/`IF` on a `Slot`** — whatever `truthy` answers, the interpreter
//! proceeds.
//!
//! The design here, and it is the strongest available through this trait:
//! - `truthy` on a `Val::Slot` (or any non-`Scalar` value) sets a
//!   `Cell<Option<FoldError>>` poison flag (legal through `&self` via
//!   interior mutability) and returns `false` — so an `IF` on a mask never
//!   branches, which at least keeps a poisoned run from taking a
//!   population-shaped decision.
//! - `truthy` on a `Val::Scalar` returns the real bool.
//! - [`FoldDialect::poison`] exposes the flag; a caller reads it through
//!   `Interpreter::dialect()` AFTER `run()` returns.
//!
//! The falsifier is real and reachable (`cbranch_on_a_slot_poisons_but_does_not_abort`
//! asserts the poison is set), but the honest statement is "detectable, not
//! abortable": a body whose LAST call is the bad branch is caught only by
//! the post-run check, never by the interpreter itself refusing mid-run.
//!
//! **Named upstream follow-up, NOT a change made in this file:**
//! `Dialect::truthy` returning `Result<bool, Self::Error>` would make this a
//! true refusal. That is an `ogar-loco` API change with its own blast radius
//! (every existing `Dialect` impl, every call site in `interpret.rs`) and it
//! belongs in its own PR with its own falsifier. `ogar-loco` is NOT touched
//! by this file.
//!
//! # The survivor-gating peephole (carried forward verbatim from W0C)
//!
//! If the right operand is `Val::Slot(b)` AND the LAST emitted `MaskOp` is a
//! `Pred` with `under: None` and `dst == b`, gate it under the left operand
//! instead of spending a facade pass on a real `And`. Only the LAST op, and
//! only if UNGATED — `gating_only_folds_the_last_ungated_pred` pins that a
//! mask consumed by `INT_AND` that is NOT the last emitted op keeps its own
//! pass.
//!
//! **A finding from actually disabling "LAST", not merely from the spec:**
//! under this vocabulary's stack discipline (no `DUP`/`SWAP` primitive), the
//! op that produced a value SITTING ON TOP of the stack is *always*
//! `self.ops.last()` at the moment something pops it — every predicate/mask
//! call pushes exactly one `MaskOp` and one stack value, 1:1, and nothing
//! can pop past an unconsumed value beneath it. Confirmed by disabling it
//! directly: replacing `self.ops.last_mut()` with a scan of the WHOLE `ops`
//! vec for a `dst == bs` entry left `under: None` changed **nothing** — all
//! 12 tests stayed green, because the one op with `dst == bs` (dst values
//! are never reused) is provably the last one whenever it is still
//! consultable at all. So "only the LAST op" is not an independent
//! guarantee THIS dialect's tests can falsify — "only if UNGATED" is the
//! real, falsifiable half (below), and the disable table's row for it is
//! written against that guard, not against "last-ness".
//!
//! # Disable table (each row run red-then-green against the committed file)
//!
//! | assertion | disable | observed |
//! |---|---|---|
//! | 2 facade passes (A and B) | peephole condition forced `false` (`&& false` on `*dst == bs`) | both count tests red — `logical_calls_fold_to_the_same_physical_pass_count_as_quack` reports 3 ops, not 2 |
//! | only if UNGATED (not "already gated") | the `under: None` binding dropped, so an already-gated op is re-gated (overwriting its `under`) | `gating_only_folds_the_last_ungated_pred` red — 3 ops instead of 4, the outer `AND` wrongly re-gates the already-gated inner pred instead of spending a real `And` |
//! | join semantics, not shape | `INT_EQUAL` on `Addr::Via` emits `EqU32` on the `fk` lane instead of `EqU32Via` | `a_join_count_carried_as_loco_bytes_matches_quack_and_the_oracle` red at every row count |
//! | signedness | `INT_S_LESS`'s reversed-operand arm removed (always emits `LtI32`) | `a_join_count_carried_as_loco_bytes_matches_quack_and_the_oracle` AND `gating_only_folds_the_last_ungated_pred` both red — frontend A/B undercounts against the oracle |
//! | Mathcad two folds | `SUM` lowers without running (pushes `Val::Scalar(0)` instead of finalizing+executing) | `the_mathcad_case_runs_two_folds_and_subtracts_them` red on both the value AND the `programs_run == 2` assertion (observed `programs_run=0`) |
//! | row-independent alloc | `finalize_and_run` allocates a fresh, population-sized (`words_for(n_rows)`, not tile-capped) buffer per fold instead of reusing `self.scratch` | `the_dialect_side_allocates_nothing_proportional_to_rows` red — 992 bytes at `n=1,000` vs 17,120 at `n=65,536` |
//! | each refusal (×6: `IntLess`, `IntSub(Addr,Addr)`, `LOAD` unknown space, `VIA` wrong kind, plus the poison flag) | its own refusal/poison arm deleted and replaced with a silent coercion (or, for the poison flag, simply not set) | the matching test red, individually, one arm at a time |
//!
//! Committed BEFORE every disable run; each restored via a kept `/tmp` copy
//! (never `git checkout`, per this session's own scope). Every patch anchor
//! was confirmed to occur exactly once before trusting a green run as
//! evidence — `cargo fmt` has moved anchors in this arc before.
//!
//! # Prior doc (W0C, superseded, kept for provenance)
//!
//! > **W0C — the row bridge: a merged operation carried as loco PROGRAM
//! > DATA, executed by the ONE tile-fused evaluator, with no population
//! > crossing.** The capstone thesis under test: *"we have already built
//! > the instruction set; stop re-implementing programs as Rust enum
//! > variants."* Measured then: logical calls (4) → 2 physical facade
//! > passes via loco, 3 via `quack::lower`; 0 population-sized bytes on the
//! > loco side; allocation identical at 1,000 and 65,536 rows. That evidence
//! > is what this file's own measurements reproduce and extend.

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;

use lance_graph_mask_risc::{
    execute_into, scratch_words_for, tile_words_for, Foreign, LaneKind, LaneRef, MaskOp, Operand,
    Out, Planes, Pred, Program, Scratch, Terminal, Value, TILE_WORDS,
};
use lance_graph_quack::{Agg, Cmp, Col, Filter, ForeignLane, Query};
use ogar_loco::vocabulary::conformance::validate;
use ogar_loco::{
    Call, Dialect, FnIndex, FunctionBody, Interpreter, LaneShape, Program as LocoProgram,
};
use ogar_r2il::{R2ILVocabulary, R2IL_BASE, SUM, VIA};

// ── allocation counter — THREAD-LOCAL, not global ──────────────────────────
//
// W0C's own instrument used a process-global `AtomicUsize`, mitigated with a
// min-over-8-repetitions loop. That mitigation is insufficient, not merely
// imprecise: `cargo test` runs test binaries' `#[test]` functions on a
// thread pool, so a global counter is shared with every OTHER test running
// concurrently. If every one of the 8 samples at `n=1,000` happens to land
// during a burst of unrelated allocation from a sibling test, the MINIMUM
// over those samples is contaminated too — and a DIFFERENT burst pattern at
// `n=65,536` can contaminate that measurement by a different amount, making
// `the_dialect_side_allocates_nothing_proportional_to_rows`'s equality
// assertion fail NONDETERMINISTICALLY depending on scheduling, not on this
// file's own behaviour. A thread-local counter sidesteps this structurally:
// [`Counting::alloc`] runs on whichever thread performs the allocation, so
// [`BYTES`] accumulates ONLY the measuring thread's own allocations —
// sibling tests on other threads cannot contribute a single byte to it,
// with no sampling or minimum-taking needed to filter them out.
//
// `const { Cell::new(0) }` is the initializer, not a plain `Cell::new(0)`:
// a non-`const` `thread_local!` initializer uses a lazily-boxed slow path
// that itself allocates on first access — inside a `GlobalAlloc::alloc`
// implementation, that first access would recurse back into `alloc`. The
// `const` initializer compiles to genuine `#[thread_local]` storage with no
// such lazy path, which is what makes it usable as an allocator's own
// counter without either recursing or needing a reentrancy guard.
//
// `dealloc` never subtracts — [`BYTES`] measures bytes ALLOCATED over the
// measured window, not a live-set delta, matching the W0C instrument this
// one supersedes.

std::thread_local! {
    static BYTES: Cell<usize> = const { Cell::new(0) };
}

struct Counting;

// SAFETY: pass-through to `System`; the counter is the only addition.
unsafe impl GlobalAlloc for Counting {
    /// Tallies `layout.size()` into the CALLING THREAD's [`BYTES`] cell,
    /// then delegates to [`System`].
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        BYTES.with(|b| b.set(b.get() + layout.size()));
        // SAFETY: same layout, same contract as the caller's.
        unsafe { System.alloc(layout) }
    }
    /// Delegates to [`System`] untouched — [`BYTES`] is cumulative-alloc
    /// only, never decremented.
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        // SAFETY: `ptr` came from `alloc` above with this `layout`.
        unsafe { System.dealloc(ptr, layout) }
    }
}

#[global_allocator]
static A: Counting = Counting;

/// The cumulative byte count [`Counting::alloc`] has tallied on THIS
/// thread so far — never contaminated by another thread's allocations.
fn bytes_now() -> usize {
    BYTES.with(Cell::get)
}

// ── R2IL-band opcodes this dialect wires, by their VERIFIED ordinals ───────
//
// `ogar_r2il` does not name each of its 82 R2IL-band opcodes individually
// (only the fold band gets named `pub const`s) — the newtype's whole point
// is that "the ordinal IS the identity". These offsets were confirmed
// directly against `R2ILFn::MNEMONICS`'s declaration order before use, not
// assumed from the spec: `Load`=1, `IntAdd`=9, `IntSub`=10, `IntAnd`=20,
// `IntEqual`=27, `IntLess`=29, `IntSLess`=30, `IntLessEqual`=31,
// `PopCount`=41.
const LOAD: FnIndex = FnIndex(R2IL_BASE + 1);
const INT_SUB: FnIndex = FnIndex(R2IL_BASE + 10);
const INT_AND: FnIndex = FnIndex(R2IL_BASE + 20);
const INT_EQUAL: FnIndex = FnIndex(R2IL_BASE + 27);
const INT_LESS: FnIndex = FnIndex(R2IL_BASE + 29);
const INT_S_LESS: FnIndex = FnIndex(R2IL_BASE + 30);
const INT_LESS_EQUAL: FnIndex = FnIndex(R2IL_BASE + 31);
const POP_COUNT: FnIndex = FnIndex(R2IL_BASE + 41);

/// Ceiling on scratch slots any ONE finalized fold program may touch in this
/// file's fixtures — `gating_only_folds_the_last_ungated_pred`'s 3 preds + 1
/// real `And` is the widest, at 4. Sized with headroom, not tightly, since
/// the whole point of `Scratch::over` is that a shared buffer can be carved
/// smaller than its own capacity for free.
const MAX_SLOTS: usize = 8;

// ── the dialect's stack value: never a population ──────────────────────────

/// Where a fold-band `LOAD` (space 0/1/2) or `VIA` points — never row-sized
/// data, only the ADDRESS of a lane or a join.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Addr {
    /// A resident value lane, at `idx` into [`Planes::lanes`].
    Lane { idx: u16, kind: LaneKind },
    /// A join address: `fk` is a `U32` lane of THIS table, `key` indexes
    /// [`Foreign::lanes`] on the OTHER table — the same two fields
    /// [`Pred::EqU32Via`] carries.
    Via { fk: u16, key: u16 },
}

/// The loco stack's value type. Every variant is either a scalar (a runtime
/// number the loco body computed) or an ADDRESS (a compile-time-shaped
/// description of where a lane or a join lives) — never a mask, never a row.
/// A predicate-producing call pushes [`Val::Slot`], a scratch slot NAME, and
/// nothing else in this file ever reaches for the population the slot
/// addresses.
/// A scratch-slot capability is valid only inside the fold epoch that minted
/// it. Slot NUMBERS are deliberately reused from zero after every finalized
/// fold; the epoch is what prevents an old logical name from becoming an ABA
/// alias of a new slot with the same number.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct SlotRef {
    slot: u16,
    epoch: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Val {
    /// A number computed by the loco body — a literal, a comparison result,
    /// or a fold's finalized-and-executed answer.
    Scalar(i64),
    /// A scratch slot holding a mask, produced by a predicate or a mask op.
    Slot(SlotRef),
    /// A resolved address — a lane or a join — not yet combined with a
    /// value into a predicate.
    Address(Addr),
}

/// Why a fold-band call was refused.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum FoldError {
    /// A control-flow or non-branching call wanted an operand and the
    /// dialect's own stack was empty.
    Underflow(FnIndex),
    /// The call is covered by [`R2ILVocabulary`]'s arity table but the
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
    /// `LOAD`'s own immediate named a space outside `{0=U32, 1=I32, 2=U64}`.
    UnknownLoadSpace(u8),
    /// A scratch-slot NAME crossed a fold boundary. Numeric slot ids are
    /// reused after every finalized fold, so accepting this would let an old
    /// value silently alias a live slot in the next program.
    StaleSlot {
        slot: u16,
        produced_epoch: u64,
        current_epoch: u64,
    },
    /// [`Dialect::truthy`] was asked to branch on a non-`Scalar` value — see
    /// the module doc's § MEASURED GAP. Set via [`FoldDialect::poison`],
    /// never returned directly (the trait cannot return an error here).
    BranchOnPopulation,
    /// A finalized fold's `execute_into` call itself refused the program.
    Exec(lance_graph_mask_risc::ExecError),
}

/// Interprets one loco body, building [`MaskOp`]s and RUNNING them at every
/// scalar-producing fold boundary (see the module doc's § THE REFINEMENT).
/// Borrows the planes rather than owning them — legal because
/// `Interpreter::new` takes its dialect by value with no lifetime bound
/// tying it to the interpreter's own `'a`.
struct FoldDialect<'p> {
    planes: &'p Planes<'p>,
    foreign: &'p Foreign<'p>,
    /// The `MaskOp`s accumulated since the last fold boundary — NEVER
    /// row-sized; its length is bounded by the number of predicate/mask
    /// calls in the loco body between two folds.
    ops: Vec<MaskOp>,
    /// The next unused scratch slot WITHIN the current `ops` accumulation.
    /// Reset to 0 every time a fold finalizes and runs.
    next_slot: u16,
    /// Fold generation that makes a `Val::Slot` a capability rather than a
    /// bare reusable integer. Incremented at every finalized fold boundary,
    /// exactly when `next_slot` restarts at zero, even if execution then
    /// refuses the program. Namespace reuse and execution success are separate.
    epoch: u64,
    /// Caller-owned scratch for `Scratch::over`, allocated ONCE at
    /// construction to `scratch_words_for(tile_words_for(n_rows), MAX_SLOTS)`
    /// and carved fresh (never re-allocated) at every fold boundary — this
    /// is what keeps allocation row-independent ACROSS folds, and it is the
    /// property `the_dialect_side_allocates_nothing_proportional_to_rows`
    /// pins.
    scratch: Vec<u64>,
    /// Total `MaskOp`s across every finalized program this dialect has run —
    /// the "physical facade passes" measurement.
    facade_passes: usize,
    /// How many folds actually finalized-and-ran.
    programs_run: usize,
    /// Set by [`Dialect::truthy`] when asked to branch on a non-`Scalar`
    /// value. `Cell`, not a plain field, because `truthy` takes `&self`.
    poison: Cell<Option<FoldError>>,
}

impl<'p> FoldDialect<'p> {
    /// A fresh dialect over `planes`/`foreign`, with its scratch buffer
    /// allocated ONCE, sized `scratch_words_for(tile_words_for(n_rows),
    /// MAX_SLOTS)` — the allocation
    /// `the_dialect_side_allocates_nothing_proportional_to_rows` pins as
    /// row-independent.
    fn new(planes: &'p Planes<'p>, foreign: &'p Foreign<'p>) -> Self {
        let words = tile_words_for(planes.n_rows);
        let cap = scratch_words_for(words, MAX_SLOTS)
            .expect("MAX_SLOTS is a small constant; no overflow");
        Self {
            planes,
            foreign,
            ops: Vec::new(),
            next_slot: 0,
            epoch: 0,
            scratch: vec![0u64; cap],
            facade_passes: 0,
            programs_run: 0,
            poison: Cell::new(None),
        }
    }

    /// The population-branch poison flag, if [`Dialect::truthy`] ever set
    /// one during the run — read AFTER `Interpreter::run()` returns.
    fn poison(&self) -> Option<FoldError> {
        self.poison.get()
    }

    /// The next unused scratch slot within the current `ops` accumulation,
    /// advancing [`Self::next_slot`] by one and tagging the name with the
    /// current fold epoch.
    fn fresh(&mut self) -> SlotRef {
        let slot = self.next_slot;
        self.next_slot += 1;
        SlotRef {
            slot,
            epoch: self.epoch,
        }
    }

    /// Lower a logical slot name to mask-risc's physical scratch operand.
    /// The executor intentionally knows only numeric slots inside ONE
    /// `Program`; the dialect is therefore the boundary that must reject a
    /// name carried across fold programs before numeric reuse can alias it.
    fn scratch_operand(&self, s: SlotRef) -> Result<Operand, FoldError> {
        if s.epoch != self.epoch {
            return Err(FoldError::StaleSlot {
                slot: s.slot,
                produced_epoch: s.epoch,
                current_epoch: self.epoch,
            });
        }
        Ok(Operand::Scratch(s.slot))
    }

    /// Survivor gating (verbatim from W0C, restated over `Operand`): if `b`
    /// was produced by the op emitted LAST and that op is an ungated `Pred`,
    /// gate it under `a` instead of spending a facade pass on `And`.
    fn and_peephole(&mut self, a: SlotRef, b: SlotRef) -> Result<SlotRef, FoldError> {
        let a = self.scratch_operand(a)?;
        let b = self.scratch_operand(b)?;
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
                return Ok(SlotRef {
                    slot: bs,
                    epoch: self.epoch,
                });
            }
        }
        let dst = self.fresh();
        self.ops.push(MaskOp::And {
            a,
            b,
            dst: dst.slot,
        });
        Ok(dst)
    }

    /// Emit a predicate, push its scratch slot.
    fn emit_pred(&mut self, pred: Pred, stack: &mut Vec<Val>) {
        let dst = self.fresh();
        self.ops.push(MaskOp::Pred {
            pred,
            under: None,
            dst: dst.slot,
        });
        stack.push(Val::Slot(dst));
    }

    /// Finalize the ops accumulated so far into a `Program` with `terminal`,
    /// RUN it over this dialect's borrowed planes, and reset for the next
    /// fold. This is the THE REFINEMENT the module doc describes: a
    /// scalar-producing fold executes here, not merely lowers.
    fn finalize_and_run(&mut self, terminal: Terminal) -> Result<Value, FoldError> {
        let ops = std::mem::take(&mut self.ops);
        self.next_slot = 0;
        // The logical namespace ends HERE, not after successful execution.
        // Once numeric slot allocation may restart at zero, every SlotRef
        // minted for the previous program is stale. Advancing only on success
        // would reopen the ABA hole after a validation/runtime refusal if the
        // dialect were reused.
        self.epoch = self
            .epoch
            .checked_add(1)
            .expect("one interpreter run cannot exhaust u64 fold epochs");
        self.facade_passes += ops.len();
        self.programs_run += 1;
        let program = Program::new(ops, terminal);
        let words = tile_words_for(self.planes.n_rows);
        let mut scratch = Scratch::over(&mut self.scratch, words, program.scratch_slots as usize)
            .map_err(FoldError::Exec)?;
        execute_into(&program, self.planes, self.foreign, &mut scratch, Out::None)
            .map_err(FoldError::Exec)
    }

    /// [`Self::finalize_and_run`], asserting the terminal's known result
    /// shape (`Terminal::Count` → `Value::Count`, `Terminal::MaskedSumI32` →
    /// `Value::SumI64`) — a caller-side invariant, not a new fallible path:
    /// every call site below passes a terminal whose `Value` shape is fixed
    /// by construction.
    fn finalize_and_run_scalar(&mut self, terminal: Terminal) -> Result<i64, FoldError> {
        match self.finalize_and_run(terminal)? {
            Value::Count(n) => Ok(i64::try_from(n).unwrap_or(i64::MAX)),
            Value::SumI64(n) => Ok(n),
            other => panic!("terminal shape guarantees Count or SumI64, got {other:?}"),
        }
    }
}

impl Dialect for FoldDialect<'_> {
    type Value = Val;
    type Error = FoldError;

    /// A `Slot` (or any non-`Scalar`) poisons rather than answering — see
    /// the module doc's § MEASURED GAP. This probe's bodies that legitimately
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

    /// Always `0` — none of this file's bodies use `REPEAT`; the loop bound
    /// this hook would answer for a mask's "count" is never exercised here.
    fn repeat_count(&self, _: &Val) -> u32 {
        0
    }

    /// Execute one non-branching call: pop its operands off the loco stack,
    /// push its result. The whole dispatch table this file wires — see the
    /// module doc's § "What is implemented, and what is refused BY NAME"
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
                match (a, b) {
                    (Val::Slot(sa), Val::Slot(sb)) => {
                        let dst = self.and_peephole(sa, sb)?;
                        stack.push(Val::Slot(dst));
                        Ok(())
                    }
                    (Val::Scalar(x), Val::Scalar(y)) => {
                        stack.push(Val::Scalar(x & y));
                        Ok(())
                    }
                    _ => Err(FoldError::WrongOperandKind(f)),
                }
            }
            INT_SUB => {
                let b = stack.pop().ok_or(FoldError::Underflow(f))?;
                let a = stack.pop().ok_or(FoldError::Underflow(f))?;
                match (a, b) {
                    (Val::Scalar(x), Val::Scalar(y)) => {
                        stack.push(Val::Scalar(x - y));
                        Ok(())
                    }
                    (Val::Address(_), Val::Address(_)) => Err(FoldError::LaneVersusLane(f)),
                    _ => Err(FoldError::WrongOperandKind(f)),
                }
            }
            POP_COUNT => {
                let mask = stack.pop().ok_or(FoldError::Underflow(f))?;
                let Val::Slot(s) = mask else {
                    return Err(FoldError::WrongOperandKind(f));
                };
                let mask = self.scratch_operand(s)?;
                let n = self.finalize_and_run_scalar(Terminal::Count { mask })?;
                stack.push(Val::Scalar(n));
                Ok(())
            }
            SUM => {
                let b = stack.pop().ok_or(FoldError::Underflow(f))?;
                let a = stack.pop().ok_or(FoldError::Underflow(f))?;
                let (slot, lane) = match (a, b) {
                    (
                        Val::Slot(s),
                        Val::Address(Addr::Lane {
                            idx,
                            kind: LaneKind::I32,
                        }),
                    ) => (s, idx),
                    (
                        Val::Address(Addr::Lane {
                            idx,
                            kind: LaneKind::I32,
                        }),
                        Val::Slot(s),
                    ) => (s, idx),
                    _ => return Err(FoldError::WrongOperandKind(f)),
                };
                let mask = self.scratch_operand(slot)?;
                let n = self.finalize_and_run_scalar(Terminal::MaskedSumI32 { mask, lane })?;
                stack.push(Val::Scalar(n));
                Ok(())
            }
            other => Err(FoldError::Unimplemented(other)),
        }
    }
}

// ── fixtures ─────────────────────────────────────────────────────────────

/// A tiny deterministic PRNG (PCG-shaped LCG step) — the fixtures below are
/// seeded, reproducible test data, never anything cryptographic.
fn lcg(seed: &mut u64) -> u64 {
    *seed = seed
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    *seed >> 11
}

/// The join+local-compare fixture, shared by frontends A and B — the same
/// shape W0C used (`fk` THROUGH `country`, local `amount` compare).
struct Tables {
    fk: Vec<u32>,
    amount: Vec<i32>,
    country: Vec<u32>,
}

impl Tables {
    /// `fk` deliberately overshoots `foreign_rows` on some rows (the
    /// `+ 3` headroom): `Pred::EqU32Via`'s underlying kernel
    /// (`ndarray::simd::eq_u32_via_to_mask`) is `addr < table.len() &&
    /// table[addr] == v` — an out-of-range `fk` DROPS the row (never
    /// matches), it does NOT fall back to comparing against `0`. This is a
    /// different contract from `MaskOp::Gather`'s kernel
    /// (`mask_gather_u32`), which genuinely IS zero-fallback — the two must
    /// not be conflated, and this fixture exercises only the former.
    fn seeded(n: usize, foreign_rows: usize, seed: u64) -> Self {
        let mut s = seed;
        let fk = (0..n)
            .map(|_| (lcg(&mut s) % (foreign_rows as u64 + 3)) as u32)
            .collect();
        let amount = (0..n).map(|_| (lcg(&mut s) % 200) as i32 - 100).collect();
        let country = (0..foreign_rows)
            .map(|_| (lcg(&mut s) % 6) as u32)
            .collect();
        Self {
            fk,
            amount,
            country,
        }
    }
    /// This table's own resident lanes, `Planes::lanes` order: `fk`(idx0,
    /// U32), `amount`(idx1, I32).
    fn lanes(&self) -> [LaneRef<'_>; 2] {
        [LaneRef::U32(&self.fk), LaneRef::I32(&self.amount)]
    }
    /// The foreign table's lanes, `Foreign::lanes` order: `country`(idx0,
    /// U32).
    fn foreign_lanes(&self) -> [LaneRef<'_>; 1] {
        [LaneRef::U32(&self.country)]
    }
    /// The scalar truth, written without the substrate. `.get(..) ==
    /// Some(&v)` is the DROP reading, not a zero-fallback one: `None` (an
    /// out-of-range `fk`) never equals `Some(&v)`, so the row is excluded —
    /// exactly `eq_u32_via_to_mask`'s own contract, including at `v == 0`.
    fn oracle(&self, v: u32, t: i32) -> usize {
        self.fk
            .iter()
            .zip(&self.amount)
            .filter(|(&fk, &a)| self.country.get(fk as usize) == Some(&v) && a > t)
            .count()
    }
}

/// The Mathcad fixture: `status`, `amount`, `cost` — three RESIDENT lanes,
/// no foreign table. `Margin = SUM(amount | status==POSTED) -
/// SUM(cost | status==POSTED)`.
struct LedgerTables {
    status: Vec<u32>,
    amount: Vec<i32>,
    cost: Vec<i32>,
}

impl LedgerTables {
    /// `n` rows of `status` (0..=3), `amount` (-100..=399) and `cost`
    /// (-50..=249) — the row VALUES may be negative (they live in resident
    /// `i32` lanes and never cross the loco byte encoding), only the
    /// `NUMBER` LITERALS the byte program itself carries are constrained to
    /// 0..=255 (see the module doc's § "Immediate ranges").
    fn seeded(n: usize, seed: u64) -> Self {
        let mut s = seed;
        let status = (0..n).map(|_| (lcg(&mut s) % 4) as u32).collect();
        let amount = (0..n).map(|_| (lcg(&mut s) % 500) as i32 - 100).collect();
        let cost = (0..n).map(|_| (lcg(&mut s) % 300) as i32 - 50).collect();
        Self {
            status,
            amount,
            cost,
        }
    }
    /// This table's resident lanes, `Planes::lanes` order: `status`(idx0,
    /// U32), `amount`(idx1, I32), `cost`(idx2, I32).
    fn lanes(&self) -> [LaneRef<'_>; 3] {
        [
            LaneRef::U32(&self.status),
            LaneRef::I32(&self.amount),
            LaneRef::I32(&self.cost),
        ]
    }
    /// Written without the substrate: the two sums, separately.
    fn oracle(&self, posted: u32) -> (i64, i64) {
        let sum_amount: i64 = self
            .status
            .iter()
            .zip(&self.amount)
            .filter(|(&s, _)| s == posted)
            .map(|(_, &a)| i64::from(a))
            .sum();
        let sum_cost: i64 = self
            .status
            .iter()
            .zip(&self.cost)
            .filter(|(&s, _)| s == posted)
            .map(|(_, &c)| i64::from(c))
            .sum();
        (sum_amount, sum_cost)
    }
}

/// Runs a native `quack::lower`-produced mask-risc [`Program`] directly
/// (never through loco) — the "native path" comparator this file's
/// correctness tests check the loco-carried result against.
fn run_quack(program: &Program, tables: &Tables) -> Value {
    let lanes = tables.lanes();
    let planes = Planes {
        n_rows: tables.fk.len(),
        masks: &[],
        lanes: &lanes,
    };
    let flanes = tables.foreign_lanes();
    let foreign = Foreign {
        planes: &[],
        lanes: &flanes,
    };
    let mut scratch = Scratch::for_program(program, planes.n_rows).expect("scratch");
    execute_into(program, &planes, &foreign, &mut scratch, Out::None).expect("executes")
}

// ── frontend byte programs ──────────────────────────────────────────────

const V: u8 = 3;
const T: i8 = 17;
const ROWS: [usize; 6] = [1, 63, 64, 130, 1000, 65_536];

/// **A — quack-shaped.** `COUNT(*) WHERE country THROUGH partner_id = V AND
/// amount > T`, join leaf first (quack orders conjuncts for selectivity).
fn frontend_a() -> LocoProgram {
    let calls = [
        Call::with_value(FnIndex::NUMBER, 0), // key: Foreign::lanes idx0 (country)
        Call::with_value(FnIndex::NUMBER, 0), // fk: Planes::lanes idx0 (partner_id)
        Call::new(VIA),
        Call::with_value(FnIndex::NUMBER, V),
        Call::new(INT_EQUAL),
        Call::with_value(FnIndex::NUMBER, T as u8),
        Call::with_value(FnIndex::NUMBER, 1), // lane idx1 (amount)
        Call::with_value(LOAD, 1),            // space=I32
        Call::new(INT_S_LESS),
        Call::new(INT_AND),
        Call::new(POP_COUNT),
    ];
    let body = FunctionBody::from_calls(LaneShape::Quads, &calls).expect("11 calls fit Quads");
    LocoProgram {
        functions: vec![body],
    }
}

/// **B — Blockly-shaped.** Same query, LOCAL filter first (a person draws
/// the simple block first).
fn frontend_b() -> LocoProgram {
    let calls = [
        Call::with_value(FnIndex::NUMBER, T as u8),
        Call::with_value(FnIndex::NUMBER, 1),
        Call::with_value(LOAD, 1),
        Call::new(INT_S_LESS),
        Call::with_value(FnIndex::NUMBER, 0),
        Call::with_value(FnIndex::NUMBER, 0),
        Call::new(VIA),
        Call::with_value(FnIndex::NUMBER, V),
        Call::new(INT_EQUAL),
        Call::new(INT_AND),
        Call::new(POP_COUNT),
    ];
    let body = FunctionBody::from_calls(LaneShape::Quads, &calls).expect("11 calls fit Quads");
    LocoProgram {
        functions: vec![body],
    }
}

const POSTED: u8 = 2;

/// **C — Mathcad-shaped.** `SUM(amount | posted) - SUM(cost | posted)`. Uses
/// lane idx2 for `cost` (a 3-lane `LedgerTables` fixture) where the spec's
/// illustrative byte listing wrote `NUMBER:3` against a 4-lane sketch — a
/// 3-lane fixture with no gap is simpler and asserts the same two things
/// (two folds, then a subtract), so this is a deliberate, reported
/// deviation from the spec's exact numeral, not a functional change.
fn frontend_c() -> LocoProgram {
    let calls = [
        Call::with_value(FnIndex::NUMBER, POSTED),
        Call::with_value(FnIndex::NUMBER, 0), // status lane idx0 (U32)
        Call::with_value(LOAD, 0),            // space=U32
        Call::new(INT_EQUAL),
        Call::with_value(FnIndex::NUMBER, 1), // amount lane idx1 (I32)
        Call::with_value(LOAD, 1),
        Call::new(SUM),
        Call::with_value(FnIndex::NUMBER, POSTED),
        Call::with_value(FnIndex::NUMBER, 0),
        Call::with_value(LOAD, 0),
        Call::new(INT_EQUAL),
        Call::with_value(FnIndex::NUMBER, 2), // cost lane idx2 (I32)
        Call::with_value(LOAD, 1),
        Call::new(SUM),
        Call::new(INT_SUB),
    ];
    let body = FunctionBody::from_calls(LaneShape::Quads, &calls).expect("15 calls fit Quads");
    LocoProgram {
        functions: vec![body],
    }
}

/// Runs one loco body to completion through a fresh [`FoldDialect`] and
/// returns the final loco stack — for tests that only need the RESULT, not
/// the measurement fields [`RunStats`] carries.
fn run_frontend(body: &LocoProgram, planes: &Planes<'_>, foreign: &Foreign<'_>) -> Vec<Val> {
    let vocab = validate(R2ILVocabulary).expect("R2ILVocabulary conforms");
    let dialect = FoldDialect::new(planes, foreign);
    let mut it = Interpreter::new(&vocab, body, dialect);
    it.run().expect("body runs to completion");
    it.stack().to_vec()
}

/// What a frontend run measured, alongside its result — `facade_passes`
/// (total `MaskOp`s across every finalized fold) and `programs_run` (how
/// many folds actually finalized-and-ran).
struct RunStats {
    /// The loco stack after `run()` returns.
    stack: Vec<Val>,
    /// Total `MaskOp`s across every finalized fold this run performed.
    facade_passes: usize,
    /// How many folds actually finalized-and-ran.
    programs_run: usize,
    /// [`FoldDialect::poison`], read after the run completed.
    poison: Option<FoldError>,
}

/// [`run_frontend`], but also returning the dialect's own measurements
/// ([`RunStats`]) and surfacing a run error instead of `expect`-panicking —
/// the shape every refusal test and every measurement test below needs.
fn run_frontend_with_stats(
    body: &LocoProgram,
    planes: &Planes<'_>,
    foreign: &Foreign<'_>,
) -> Result<RunStats, ogar_loco::RunError<FoldError>> {
    let vocab = validate(R2ILVocabulary).expect("R2ILVocabulary conforms");
    let dialect = FoldDialect::new(planes, foreign);
    let mut it = Interpreter::new(&vocab, body, dialect);
    it.run()?;
    let d = it.dialect();
    Ok(RunStats {
        stack: it.stack().to_vec(),
        facade_passes: d.facade_passes,
        programs_run: d.programs_run,
        poison: d.poison(),
    })
}

// ── tests ──────────────────────────────────────────────────────────────────

/// FAILS IF: frontend A, frontend B, or `quack::lower`'s own program
/// disagree on the count at any row boundary — the acid test's requirement
/// #1 ("same result") over both loco-carried orderings and the native path.
#[test]
fn a_join_count_carried_as_loco_bytes_matches_quack_and_the_oracle() {
    let via_quack = lance_graph_quack::lower(&Query {
        filter: Filter::and([
            Filter::eq_u32_via(Col(0), ForeignLane(0), u32::from(V)),
            Filter::cmp(Col(1), Cmp::GtI32(i32::from(T))),
        ]),
        agg: Agg::Count,
    })
    .expect("quack lowers");
    let a = frontend_a();
    let b = frontend_b();
    for (i, &n) in ROWS.iter().enumerate() {
        let t = Tables::seeded(n, 40, 11 + i as u64);
        let expect = t.oracle(u32::from(V), i32::from(T)) as i64;

        let lanes = t.lanes();
        let flanes = t.foreign_lanes();
        let planes = Planes {
            n_rows: n,
            masks: &[],
            lanes: &lanes,
        };
        let foreign = Foreign {
            planes: &[],
            lanes: &flanes,
        };

        let stack_a = run_frontend(&a, &planes, &foreign);
        assert_eq!(stack_a, vec![Val::Scalar(expect)], "frontend A, n={n}");
        let stack_b = run_frontend(&b, &planes, &foreign);
        assert_eq!(stack_b, vec![Val::Scalar(expect)], "frontend B, n={n}");

        assert_eq!(
            run_quack(&via_quack, &t),
            Value::Count(expect as usize),
            "quack path, n={n}"
        );
    }
}

/// FAILS IF: either frontend spends MORE facade passes than quack's native
/// lowering, the two frontends disagree with each other (order-
/// independence), or the pinned counts drift silently.
///
/// MEASURED: 4 logical calls (`VIA`, `INT_EQUAL`, `INT_S_LESS`, `INT_AND`)
/// fold to **2** physical `MaskOp`s via this dialect on EITHER ordering,
/// against **3** via `quack::lower` (its `emit_gated` gates the second
/// conjunct and then still spends a redundant trailing `And` — a finding
/// about the native path, pinned here so it cannot drift unnoticed; fixing
/// quack is its own wave, not a drive-by here — carried forward unchanged
/// from W0C).
#[test]
fn logical_calls_fold_to_the_same_physical_pass_count_as_quack() {
    let via_quack = lance_graph_quack::lower(&Query {
        filter: Filter::and([
            Filter::eq_u32_via(Col(0), ForeignLane(0), u32::from(V)),
            Filter::cmp(Col(1), Cmp::GtI32(i32::from(T))),
        ]),
        agg: Agg::Count,
    })
    .expect("quack lowers");

    let t = Tables::seeded(1000, 40, 5);
    let lanes = t.lanes();
    let flanes = t.foreign_lanes();
    let planes = Planes {
        n_rows: 1000,
        masks: &[],
        lanes: &lanes,
    };
    let foreign = Foreign {
        planes: &[],
        lanes: &flanes,
    };

    let a = frontend_a();
    let stats_a = run_frontend_with_stats(&a, &planes, &foreign).expect("frontend A runs");
    let b = frontend_b();
    let stats_b = run_frontend_with_stats(&b, &planes, &foreign).expect("frontend B runs");

    println!(
        "W1b: A facade_passes={} B facade_passes={} quack ops={}",
        stats_a.facade_passes,
        stats_b.facade_passes,
        via_quack.ops.len()
    );

    assert_eq!(
        stats_a.facade_passes, 2,
        "frontend A: join pred + gated compare"
    );
    assert_eq!(
        stats_b.facade_passes, 2,
        "frontend B: same count, opposite order"
    );
    assert_eq!(
        stats_a.programs_run, 1,
        "one fold — POP_COUNT is the only terminal call"
    );
    assert_eq!(stats_b.programs_run, 1);
    assert!(
        stats_a.facade_passes <= via_quack.ops.len(),
        "never more passes than the native lowering"
    );
    // Pinned two-sided so a quack improvement forces a deliberate re-pin here
    // rather than leaving this comment describing a gap that closed.
    assert_eq!(
        via_quack.ops.len(),
        3,
        "quack::lower's redundant trailing AND (see module doc)"
    );
}

/// FAILS IF: anything the dialect side allocates scales with rows. The
/// bytes spent building AND RUNNING the fold program must be IDENTICAL at
/// 1,000 and 65,536 rows — `tile_words_for` saturates at `TILE_WORDS` for
/// any `n_rows` past 512, so the dialect's own scratch buffer, and every
/// `MaskOp` it builds, is sized by the BODY, never by the row count.
#[test]
fn the_dialect_side_allocates_nothing_proportional_to_rows() {
    // The counter is THREAD-LOCAL (see the module-level note above `BYTES`),
    // so a sibling test running on another thread cannot contribute a
    // single byte here — the minimum-over-repetitions below is no longer
    // filtering cross-test contamination, only the small residual noise a
    // single thread's own allocator bookkeeping can introduce run to run.
    let a = frontend_a();
    let mut per_n = Vec::new();
    for &n in &[1_000usize, 65_536] {
        let t = Tables::seeded(n, 40, 5);
        let lanes = t.lanes();
        let flanes = t.foreign_lanes();
        let planes = Planes {
            n_rows: n,
            masks: &[],
            lanes: &lanes,
        };
        let foreign = Foreign {
            planes: &[],
            lanes: &flanes,
        };
        let mut bytes = usize::MAX;
        let mut last = Vec::new();
        for _ in 0..8 {
            let before = bytes_now();
            let stats = run_frontend_with_stats(&a, &planes, &foreign).expect("runs");
            bytes = bytes.min(bytes_now() - before);
            last = stats.stack;
        }
        println!("W1b: n={n} dialect-side bytes={bytes} -> {last:?}");
        per_n.push(bytes);
    }
    assert_eq!(
        per_n[0], per_n[1],
        "dialect-side allocation is independent of row count"
    );
    assert!(per_n[0] > 0, "the counter is live");
    assert!(
        per_n[1] < 65_536 / 8,
        "not a population-sized mask: {} bytes",
        per_n[1]
    );
}

/// FAILS IF: the executor's own scratch grows past one tile per slot — the
/// acid test's requirement #4, checked structurally (the buffer's exact
/// WORD length) rather than via allocator bytes, since
/// [`FoldDialect::new`] sizes it with a closed formula.
#[test]
fn the_executor_scratch_stays_tile_local_at_a_large_row_count() {
    let t = Tables::seeded(65_536, 40, 9);
    let lanes = t.lanes();
    let flanes = t.foreign_lanes();
    let planes = Planes {
        n_rows: 65_536,
        masks: &[],
        lanes: &lanes,
    };
    let foreign = Foreign {
        planes: &[],
        lanes: &flanes,
    };
    let dialect = FoldDialect::new(&planes, &foreign);
    let tile_cap_words = TILE_WORDS * MAX_SLOTS;
    println!(
        "W1b: scratch words={} tile_cap_words={}",
        dialect.scratch.len(),
        tile_cap_words
    );
    assert!(
        dialect.scratch.len() <= tile_cap_words + MAX_SLOTS.div_ceil(64) + 1,
        "scratch stays tile-local: {} <= {tile_cap_words}",
        dialect.scratch.len()
    );
    // A population mask at 65,536 rows would need `65_536 / 64 = 1024`
    // words for ONE slot alone — comfortably more than the whole tile-local
    // buffer this dialect ever allocates, across every slot.
    assert!(dialect.scratch.len() < 65_536 / 64);
}

/// FAILS IF: a body whose stack discipline is broken is silently accepted.
/// The vocabulary owns arity (refused as `UncoveredArity` by the engine
/// itself for anything outside `R2ILVocabulary`); the dialect owns
/// underflow and wrong-shape operands. Both refusals must be reachable.
#[test]
fn malformed_bodies_are_refused_not_approximated() {
    let t = Tables::seeded(64, 8, 3);
    let lanes = t.lanes();
    let flanes = t.foreign_lanes();
    let planes = Planes {
        n_rows: 64,
        masks: &[],
        lanes: &lanes,
    };
    let foreign = Foreign {
        planes: &[],
        lanes: &flanes,
    };

    // INT_AND with one operand: the dialect sees an underflow on the second
    // pop.
    let calls = [Call::with_value(FnIndex::NUMBER, 1), Call::new(INT_AND)];
    let body = FunctionBody::from_calls(LaneShape::Quads, &calls).unwrap();
    let program = LocoProgram {
        functions: vec![body],
    };
    let err = run_frontend_with_stats(&program, &planes, &foreign)
        .err()
        .expect("underflow is refused");
    assert!(matches!(
        err,
        ogar_loco::RunError::Dialect(FoldError::Underflow(f)) if f == INT_AND
    ));
}

/// FAILS IF: the survivor-gating peephole rewrites an op it must not — a
/// mask consumed by `INT_AND` that is NOT the last emitted op keeps its own
/// pass, and the result is still exact.
/// `(join) (compare1) (compare2) (AND) (AND)`: the inner `AND` gates the
/// third op under the second; the outer `AND` finds the last op already
/// gated and must spend a real `And`.
#[test]
fn gating_only_folds_the_last_ungated_pred() {
    let t = Tables::seeded(1000, 40, 3);
    let lanes = t.lanes();
    let flanes = t.foreign_lanes();
    let planes = Planes {
        n_rows: 1000,
        masks: &[],
        lanes: &lanes,
    };
    let foreign = Foreign {
        planes: &[],
        lanes: &flanes,
    };

    let calls = [
        Call::with_value(FnIndex::NUMBER, 0),
        Call::with_value(FnIndex::NUMBER, 0),
        Call::new(VIA),
        Call::with_value(FnIndex::NUMBER, V),
        Call::new(INT_EQUAL),
        Call::with_value(FnIndex::NUMBER, T as u8),
        Call::with_value(FnIndex::NUMBER, 1),
        Call::with_value(LOAD, 1),
        Call::new(INT_S_LESS),
        // A WEAKER, redundant threshold — not `T - 40`: `NUMBER`'s immediate
        // is read back as `i64::from(v[0])`, an UNSIGNED byte, so a negative
        // literal round-trips as its two's-complement positive value instead
        // (found by running this test: `(T - 40) as u8` silently became 233,
        // making the second predicate `amount > 233` — always false, and the
        // count came back 0). `0 < T` keeps the predicate genuinely
        // redundant without needing a signed immediate.
        Call::with_value(FnIndex::NUMBER, 0),
        Call::with_value(FnIndex::NUMBER, 1),
        Call::with_value(LOAD, 1),
        Call::new(INT_S_LESS),
        Call::new(INT_AND),
        Call::new(INT_AND),
        Call::new(POP_COUNT),
    ];
    let body = FunctionBody::from_calls(LaneShape::Quads, &calls).unwrap();
    let program = LocoProgram {
        functions: vec![body],
    };
    let stats = run_frontend_with_stats(&program, &planes, &foreign).expect("runs");
    assert_eq!(stats.facade_passes, 4, "3 preds + 1 real AND");
    // amount > T && amount > T-40 == amount > T
    let expect = t.oracle(u32::from(V), i32::from(T)) as i64;
    assert_eq!(stats.stack, vec![Val::Scalar(expect)]);
}

/// FAILS IF: the Mathcad case does not actually run TWO folds — the acid
/// test's requirement #5, proving `programs_run == 2` so a fused single
/// program cannot accidentally answer for the wrong reason.
#[test]
fn the_mathcad_case_runs_two_folds_and_subtracts_them() {
    let t = LedgerTables::seeded(500, 21);
    let lanes = t.lanes();
    let planes = Planes {
        n_rows: 500,
        masks: &[],
        lanes: &lanes,
    };
    let foreign = Foreign {
        planes: &[],
        lanes: &[],
    };
    let c = frontend_c();
    let stats = run_frontend_with_stats(&c, &planes, &foreign).expect("runs");

    let (sum_amount, sum_cost) = t.oracle(u32::from(POSTED));
    let expect = sum_amount - sum_cost;
    println!(
        "W1b: mathcad sum_amount={sum_amount} sum_cost={sum_cost} margin={expect} programs_run={}",
        stats.programs_run
    );
    assert_eq!(stats.stack, vec![Val::Scalar(expect)]);
    assert_eq!(
        stats.programs_run, 2,
        "two folds actually ran, not one fused program"
    );
}

/// FAILS IF: a logical slot name can survive one finalized fold and then
/// alias a newly-written slot with the same numeric id in the next fold.
///
/// The positive control matters: after the first fold resets `next_slot`,
/// epoch 1 deliberately writes numeric slot 0 before using the stale epoch-0
/// slot 0 as a gate for live slot 1. If the epoch check is removed, this is
/// NOT a read-before-write shape: mask-risc sees slot 0 as genuinely written
/// in the current program and accepts the numeric alias. The dialect must
/// reject it before lowering to `Operand::Scratch(0)`.
#[test]
fn a_stale_slot_cannot_alias_a_live_slot_in_the_next_fold_epoch() {
    let t = Tables::seeded(64, 8, 17);
    let lanes = t.lanes();
    let flanes = t.foreign_lanes();
    let planes = Planes {
        n_rows: 64,
        masks: &[],
        lanes: &lanes,
    };
    let foreign = Foreign {
        planes: &[],
        lanes: &flanes,
    };
    let mut d = FoldDialect::new(&planes, &foreign);

    // Epoch 0: mint slot 0, run a fold through it, but retain the logical
    // name as the adversarial stale value.
    let mut first = Vec::new();
    d.emit_pred(Pred::GtI32 { lane: 1, t: 0 }, &mut first);
    let Val::Slot(stale) = first.pop().expect("predicate pushes a slot") else {
        unreachable!("emit_pred only pushes Val::Slot");
    };
    let first_mask = d.scratch_operand(stale).expect("fresh slot is current");
    d.finalize_and_run_scalar(Terminal::Count { mask: first_mask })
        .expect("first fold runs");
    assert_eq!(
        d.epoch, 1,
        "finalizing a fold advances the generation at the namespace boundary"
    );

    // Epoch 1 writes numeric slot 0, then slot 1. This is the decisive ABA
    // shape: the stale name's NUMBER is live in the current program.
    let mut current = Vec::new();
    d.emit_pred(Pred::GtI32 { lane: 1, t: 3 }, &mut current);
    let Val::Slot(live0) = current.pop().expect("slot 0") else {
        unreachable!();
    };
    d.emit_pred(Pred::LtI32 { lane: 1, t: 120 }, &mut current);
    let Val::Slot(live1) = current.pop().expect("slot 1") else {
        unreachable!();
    };
    assert_eq!(stale.slot, live0.slot, "numeric slot 0 is intentionally reused");
    assert_ne!(
        stale.epoch, live0.epoch,
        "only the generation distinguishes the two logical names"
    );

    let mut combine = vec![Val::Slot(stale), Val::Slot(live1)];
    let err = d
        .call(INT_AND, [0; 3], &mut combine)
        .expect_err("a stale logical name must be refused before lowering");
    assert_eq!(
        err,
        FoldError::StaleSlot {
            slot: stale.slot,
            produced_epoch: stale.epoch,
            current_epoch: live1.epoch,
        }
    );

    // Positive control: model exactly what deleting the epoch check would
    // lower. Because current epoch slot 0 has already been written, the
    // executor's own read-before-write validator ACCEPTS the alias. That is
    // why this guard belongs above mask-risc rather than inside Scratch.
    let mut aliased_ops = d.ops.clone();
    match aliased_ops.last_mut().expect("two current-epoch predicates") {
        MaskOp::Pred { under, dst, .. } => {
            assert_eq!(*dst, live1.slot);
            *under = Some(Operand::Scratch(stale.slot));
        }
        other => panic!("expected trailing predicate, got {other:?}"),
    }
    let aliased = Program::new(
        aliased_ops,
        Terminal::Count {
            mask: Operand::Scratch(live1.slot),
        },
    );
    let mut scratch = Scratch::for_program(&aliased, planes.n_rows).expect("addressable");
    assert!(
        execute_into(&aliased, &planes, &foreign, &mut scratch, Out::None).is_ok(),
        "numeric ABA is validator-clean once the reused slot was written this epoch"
    );
}

/// FAILS IF: a fold that resets numeric slot allocation but then errors
/// leaves the old generation live.
///
/// Execution success is irrelevant to slot identity. Once `next_slot` can
/// restart at zero, a pre-boundary SlotRef must be stale, otherwise a caller
/// that inspects/reuses the dialect after a refusal can recreate the same ABA
/// alias as the successful-fold case.
#[test]
fn a_failed_finalized_fold_still_ends_the_slot_epoch() {
    let t = Tables::seeded(64, 8, 29);
    let lanes = t.lanes();
    let flanes = t.foreign_lanes();
    let planes = Planes {
        n_rows: 64,
        masks: &[],
        lanes: &lanes,
    };
    let foreign = Foreign {
        planes: &[],
        lanes: &flanes,
    };
    let mut d = FoldDialect::new(&planes, &foreign);

    let mut stack = Vec::new();
    d.emit_pred(Pred::GtI32 { lane: 1, t: 0 }, &mut stack);
    let Val::Slot(stale) = stack.pop().expect("predicate pushes slot") else {
        unreachable!();
    };

    // Replace the valid pending op with a deliberately invalid current
    // program: it reads scratch 0 before any op writes it.
    d.ops.clear();
    d.ops.push(MaskOp::And {
        a: Operand::Scratch(0),
        b: Operand::Scratch(0),
        dst: 0,
    });
    assert!(
        d.finalize_and_run_scalar(Terminal::Count {
            mask: Operand::Scratch(0),
        })
        .is_err(),
        "the adversarial program must refuse"
    );
    assert_eq!(
        d.epoch, 1,
        "namespace generation advances even though execution failed"
    );

    let mut current = Vec::new();
    d.emit_pred(Pred::GtI32 { lane: 1, t: 3 }, &mut current);
    let Val::Slot(live) = current.pop().expect("fresh predicate slot") else {
        unreachable!();
    };
    assert_eq!(stale.slot, live.slot, "numeric slot zero is reused");
    assert_ne!(stale.epoch, live.epoch, "logical names must not alias");
    assert_eq!(
        d.scratch_operand(stale),
        Err(FoldError::StaleSlot {
            slot: stale.slot,
            produced_epoch: stale.epoch,
            current_epoch: live.epoch,
        })
    );
}

// ── refusal tests (acid test requirement #6) ────────────────────────────

/// FAILS IF: an `IF` on a `Val::Slot` either aborts the run (it cannot —
/// see the module doc's § MEASURED GAP) or leaves no trace at all. The run
/// must complete `Ok`, and the poison flag must be set afterward.
#[test]
fn cbranch_on_a_slot_poisons_but_does_not_abort() {
    let t = Tables::seeded(64, 8, 4);
    let lanes = t.lanes();
    let flanes = t.foreign_lanes();
    let planes = Planes {
        n_rows: 64,
        masks: &[],
        lanes: &lanes,
    };
    let foreign = Foreign {
        planes: &[],
        lanes: &flanes,
    };
    // Build a Slot (a real mask), then branch on it. The branch target is
    // never resolved because `truthy` on a Slot always answers `false`.
    let calls = [
        Call::with_value(FnIndex::NUMBER, 0),
        Call::with_value(FnIndex::NUMBER, 1),
        Call::with_value(LOAD, 1),
        Call::new(INT_S_LESS),
        Call::with_value(FnIndex::IF, 1),
    ];
    let body = FunctionBody::from_calls(LaneShape::Quads, &calls).unwrap();
    let program = LocoProgram {
        functions: vec![body],
    };
    let stats = run_frontend_with_stats(&program, &planes, &foreign)
        .expect("truthy(Slot) answers false, so the run completes");
    assert_eq!(
        stats.poison,
        Some(FoldError::BranchOnPopulation),
        "the poison flag records the population-branch, even though the run succeeded"
    );
}

/// FAILS IF: `IntLess` (the UNSIGNED compare) is ever silently accepted —
/// mask-risc has no unsigned lane compare, only the signed pair.
#[test]
fn int_less_unsigned_is_refused() {
    let t = Tables::seeded(8, 2, 1);
    let lanes = t.lanes();
    let flanes = t.foreign_lanes();
    let planes = Planes {
        n_rows: 8,
        masks: &[],
        lanes: &lanes,
    };
    let foreign = Foreign {
        planes: &[],
        lanes: &flanes,
    };
    let calls = [
        Call::with_value(FnIndex::NUMBER, 1),
        Call::with_value(FnIndex::NUMBER, 2),
        Call::new(INT_LESS),
    ];
    let body = FunctionBody::from_calls(LaneShape::Quads, &calls).unwrap();
    let program = LocoProgram {
        functions: vec![body],
    };
    let err = run_frontend_with_stats(&program, &planes, &foreign)
        .err()
        .expect("unsigned compare is refused");
    assert!(matches!(
        err,
        ogar_loco::RunError::Dialect(FoldError::NoUnsignedLaneCompare(f)) if f == INT_LESS
    ));
}

/// FAILS IF: `IntSub` between two lane ADDRESSES is ever silently accepted
/// as if it were scalar arithmetic.
#[test]
fn int_sub_of_two_lanes_is_refused() {
    let t = Tables::seeded(8, 2, 2);
    let lanes = t.lanes();
    let flanes = t.foreign_lanes();
    let planes = Planes {
        n_rows: 8,
        masks: &[],
        lanes: &lanes,
    };
    let foreign = Foreign {
        planes: &[],
        lanes: &flanes,
    };
    let calls = [
        Call::with_value(FnIndex::NUMBER, 0),
        Call::with_value(LOAD, 0),
        Call::with_value(FnIndex::NUMBER, 1),
        Call::with_value(LOAD, 1),
        Call::new(INT_SUB),
    ];
    let body = FunctionBody::from_calls(LaneShape::Quads, &calls).unwrap();
    let program = LocoProgram {
        functions: vec![body],
    };
    let err = run_frontend_with_stats(&program, &planes, &foreign)
        .err()
        .expect("lane-versus-lane subtraction is refused");
    assert!(matches!(
        err,
        ogar_loco::RunError::Dialect(FoldError::LaneVersusLane(f)) if f == INT_SUB
    ));
}

/// FAILS IF: `LOAD` with a space byte outside `{0,1,2}` is ever silently
/// accepted as one of the three known lane kinds.
#[test]
fn load_with_an_unknown_space_is_refused() {
    let t = Tables::seeded(8, 2, 3);
    let lanes = t.lanes();
    let flanes = t.foreign_lanes();
    let planes = Planes {
        n_rows: 8,
        masks: &[],
        lanes: &lanes,
    };
    let foreign = Foreign {
        planes: &[],
        lanes: &flanes,
    };
    let calls = [
        Call::with_value(FnIndex::NUMBER, 0),
        Call::with_value(LOAD, 9),
    ];
    let body = FunctionBody::from_calls(LaneShape::Quads, &calls).unwrap();
    let program = LocoProgram {
        functions: vec![body],
    };
    let err = run_frontend_with_stats(&program, &planes, &foreign)
        .err()
        .expect("unknown load space is refused");
    assert!(matches!(
        err,
        ogar_loco::RunError::Dialect(FoldError::UnknownLoadSpace(9))
    ));
}

/// FAILS IF: a fold-band op (`0xE2..`) is ever silently coerced when the
/// stack holds the wrong VALUE kinds for it — here `VIA` handed a `Slot`
/// where it wants two `Scalar`s.
#[test]
fn wrong_operand_kind_on_a_fold_op_is_refused() {
    let t = Tables::seeded(8, 2, 5);
    let lanes = t.lanes();
    let flanes = t.foreign_lanes();
    let planes = Planes {
        n_rows: 8,
        masks: &[],
        lanes: &lanes,
    };
    let foreign = Foreign {
        planes: &[],
        lanes: &flanes,
    };
    let calls = [
        Call::with_value(FnIndex::NUMBER, 0),
        Call::with_value(FnIndex::NUMBER, 0),
        Call::new(VIA),
        Call::with_value(FnIndex::NUMBER, V),
        Call::new(INT_EQUAL), // -> Val::Slot
        Call::with_value(FnIndex::NUMBER, 5),
        Call::new(VIA), // pops (Slot, Scalar) — wrong kind
    ];
    let body = FunctionBody::from_calls(LaneShape::Quads, &calls).unwrap();
    let program = LocoProgram {
        functions: vec![body],
    };
    let err = run_frontend_with_stats(&program, &planes, &foreign)
        .err()
        .expect("wrong operand kind is refused");
    assert!(matches!(
        err,
        ogar_loco::RunError::Dialect(FoldError::WrongOperandKind(f)) if f == VIA
    ));
}
