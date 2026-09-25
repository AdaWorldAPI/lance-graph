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
//! address constructor), **`SUM`** (the scalar reduction this file's three
//! frontends need) and **`GROUP_SUM`** (the keyed reduction, one byte whose
//! physical terminal is chosen by the KEY operand's address kind) are wired.
//! The other nine — `RANGE`, `MIN`, `MAX`, `KEY_RUNS`, `ANY`, `ALL`, `KEEP`,
//! `SCATTER_OR`, `BLEND` — are refused as [`FoldError::Unimplemented`], named
//! individually in the refusal, never silently coerced into one of the three
//! that exist.
//! Landing any of them without its own falsifier would be the same enum-
//! explosion `ogar_r2il`'s own `ARITY` doc warns against ("a first draft…
//! invented nine variants from memory").
//!
//! Of the R2IL band: `Load`, `IntSub` (scalar/scalar only — `Addr`/`Addr` is
//! a NAMED refusal, not an omission), `IntAnd` (with the survivor-gating
//! peephole), `IntEqual`, `IntSLess`, and `PopCount` are wired. `IntAdd` is
//! refused as `Unimplemented` (nothing in the three frontend-SHAPED programs
//! below needs
//! it — the Mathcad case only ever SUBTRACTS two folds). `IntLess` and
//! `IntSLessEqual`'s unsigned sibling `IntLessEqual` are refused by NAME as
//! [`FoldError::NoUnsignedLaneCompare`], because mask-risc — this dialect's
//! target machine — has no unsigned lane compare (`ogar_r2il`'s own module
//! doc states this explicitly; it is the mechanical REASON the refusal
//! exists, not a gap this file left open). `IntSLessEqual`, `IntNotEqual`,
//! `IntOr`, `IntXor`, `IntNot` are untested and fall to the generic
//! `Unimplemented` catch-all — none of the three frontend-shaped programs
//! needs them, and
//! landing them ahead of a falsifier would be exactly the anti-pattern this
//! doc already names twice.
//!
//! # What this file establishes, and two things it does NOT
//!
//! Stated because both were claimed more strongly than the code supports,
//! and review caught them rather than a test (added 2026-09-22, post-#1258).
//!
//! **Established.** ONE checked R2IL vocabulary and ONE typed `Dialect`
//! execute ordinary scalar R2IL operations and population folds in the SAME
//! body, on one stack. `the_mathcad_case_runs_two_folds_and_subtracts_them`
//! is the proof: two folds run (`programs_run == 2`), each returns a scalar
//! at its fold boundary, and R2IL arithmetic continues over the results.
//!
//! **NOT established: that two classids are needed for that.** This file
//! never touches `VocabularyRegistry`, `CONCEPT_R2IL_MACHINE` or
//! `CONCEPT_R2IL_FOLD`; it calls `validate(R2ILVocabulary)` and nothing
//! else. OGAR #306 mints those ids, and they may well be right as an
//! ENTRY-POINT discriminator between the machine and folded readings of the
//! same table — but scalar R2IL and folds coexisting inside one folded body
//! demonstrably needs no cross-vocabulary call, because that is exactly what
//! this file does without one. A body switching vocabularies mid-stream is
//! unexercised here and may not be a seam this workload ever needs.
//!
//! **NOT established: that three FRONTENDS agree.** There are three
//! hand-written byte programs in three frontend SHAPES. Only the
//! quack-shaped one is checked against an independently executed oracle
//! (`lance_graph_quack::lower` in the same test). `blockly_abi::
//! lower_program_with_pool` is never invoked, and Mathcad is not a producer
//! at all. The honest claim is that three program shapes execute through one
//! dialect and one of them matches a native path bit-for-bit. Feeding real
//! blockly-rs output through this dialect is the wave that would upgrade it,
//! and it is blocked only on that repo not being present here.
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
//! frontend-shaped programs and every test below (`V = 3`, `T = 17`, `POSTED = 2`, every
//! lane index, every `LOAD` space) is chosen to stay non-negative and under
//! 256 — a real constraint this file ran into directly: seeing
//! `gating_only_folds_the_last_ungated_pred`'s own doc comment for the
//! `NUMBER:(T - 40)` byte that silently became 233 instead of -23.
//! `FnIndex::CONSTANT` (the pool load, for a literal that needs to be
//! negative or ≥ 256) is NOT wired here — every literal this file needs
//! fits in `NUMBER`'s 0..=255, so `CONSTANT` is left as one more
//! `Unimplemented` byte, not a gap load-bearing for anything below.
//!
//! **`GROUP_SUM` and its sink.** The W1b spec's `group_sink: Vec<i64>` field
//! was left out while `GROUP_SUM` was a refused byte (an unwritten field fails
//! `clippy -D warnings`). It is back now, as [`FoldDialect::group_sink`], sized
//! ONCE by [`FoldDialect::with_groups`] to the group universe K — the demanded
//! `GROUP BY` result, never the row population N. The stack spelling is
//! `mask, key-address, value-address`: a resident `U32` key lane lowers to
//! [`Terminal::GroupSumI32`], a `VIA` key address lowers to
//! [`Terminal::GroupSumViaI32`]. The loco byte is the same in both cases; the
//! ADDRESS decides the physical terminal, which is the point of carrying
//! addresses on the stack instead of pre-resolved terminals.
//!
//! `GROUP_SUM` is a WRITE terminal: it pops three and pushes nothing, which
//! makes it the first dialect-side op that can shrink the stack past a value
//! sitting beneath it. See
//! `group_sum_exposes_what_lay_beneath_it_and_the_epoch_guard_refuses_it`.
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
use ogar_r2il::{R2ILVocabulary, GROUP_SUM, R2IL_BASE, SUM, VIA};

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
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Val {
    /// A number computed by the loco body — a literal, a comparison result,
    /// or a fold's finalized-and-executed answer.
    Scalar(i64),
    /// A scratch slot holding a mask, produced by a predicate or a mask op —
    /// together with the EXECUTION EPOCH it was minted in
    /// ([`FoldDialect::epoch`] at the moment of the push). `finalize_and_run`
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
    /// A [`Val::Slot`] whose epoch does not match [`FoldDialect::epoch`] was
    /// fed to a consuming arm (`INT_AND`/`POP_COUNT`/`SUM`/`GROUP_SUM`). The slot names a
    /// scratch position in an `ops` graph a PRIOR fold's `finalize_and_run`
    /// already discarded — its number may numerically alias a slot minted
    /// fresh in the current epoch, so it is refused rather than folded.
    StaleSlot,
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
            group_sink: &mut [],
            facade_passes: 0,
            programs_run: 0,
            poison: Cell::new(None),
        }
    }

    /// [`Self::new`], writing `GROUP_SUM` results into the caller's `sink`.
    /// Its length is the group universe K; the dialect allocates nothing for
    /// it.
    fn with_groups(planes: &'p Planes<'p>, foreign: &'p Foreign<'p>, sink: &'p mut [i64]) -> Self {
        let mut d = Self::new(planes, foreign);
        d.group_sink = sink;
        d
    }

    /// The population-branch poison flag, if [`Dialect::truthy`] ever set
    /// one during the run — read AFTER `Interpreter::run()` returns.
    fn poison(&self) -> Option<FoldError> {
        self.poison.get()
    }

    /// The next unused scratch slot within the current `ops` accumulation,
    /// advancing [`Self::next_slot`] by one.
    fn fresh(&mut self) -> u16 {
        let s = self.next_slot;
        self.next_slot += 1;
        s
    }

    /// Survivor gating (verbatim from W0C, restated over `Operand`): if `b`
    /// was produced by the op emitted LAST and that op is an ungated `Pred`,
    /// gate it under `a` instead of spending a facade pass on `And`.
    fn and_peephole(&mut self, a: Operand, b: Operand) -> u16 {
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
    fn emit_pred(&mut self, pred: Pred, stack: &mut Vec<Val>) {
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
    /// fold. This is the THE REFINEMENT the module doc describes: a
    /// scalar-producing fold executes here, not merely lowers.
    ///
    /// `grouped` selects the output: `false` → `Out::None` (scalar
    /// terminals), `true` → `Out::I64` over [`Self::group_sink`]. It is a
    /// parameter rather than a second finalize function so the slot-epoch
    /// boundary below exists in exactly one place for every terminal.
    fn finalize_and_run(&mut self, terminal: Terminal, grouped: bool) -> Result<Value, FoldError> {
        let ops = std::mem::take(&mut self.ops);
        self.next_slot = 0;
        self.epoch += 1;
        self.facade_passes += ops.len();
        self.programs_run += 1;
        let program = Program::new(ops, terminal);
        let words = tile_words_for(self.planes.n_rows);
        let mut scratch = Scratch::over(&mut self.scratch, words, program.scratch_slots as usize)
            .map_err(FoldError::Exec)?;
        let out = if grouped {
            Out::I64(self.group_sink)
        } else {
            Out::None
        };
        execute_into(&program, self.planes, self.foreign, &mut scratch, out)
            .map_err(FoldError::Exec)
    }

    /// [`Self::finalize_and_run`], asserting the terminal's known result
    /// shape (`Terminal::Count` → `Value::Count`, `Terminal::MaskedSumI32` →
    /// `Value::SumI64`) — a caller-side invariant, not a new fallible path:
    /// every call site below passes a terminal whose `Value` shape is fixed
    /// by construction.
    fn finalize_and_run_scalar(&mut self, terminal: Terminal) -> Result<i64, FoldError> {
        match self.finalize_and_run(terminal, false)? {
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
                let Val::Slot { slot: s, epoch } = mask else {
                    return Err(FoldError::WrongOperandKind(f));
                };
                if epoch != self.epoch {
                    return Err(FoldError::StaleSlot);
                }
                let n = self.finalize_and_run_scalar(Terminal::Count {
                    mask: Operand::Scratch(s),
                })?;
                stack.push(Val::Scalar(n));
                Ok(())
            }
            SUM => {
                let b = stack.pop().ok_or(FoldError::Underflow(f))?;
                let a = stack.pop().ok_or(FoldError::Underflow(f))?;
                let (mask, lane) = match (a, b) {
                    (
                        Val::Slot { slot: s, epoch },
                        Val::Address(Addr::Lane {
                            idx,
                            kind: LaneKind::I32,
                        }),
                    ) => {
                        if epoch != self.epoch {
                            return Err(FoldError::StaleSlot);
                        }
                        (Operand::Scratch(s), idx)
                    }
                    (
                        Val::Address(Addr::Lane {
                            idx,
                            kind: LaneKind::I32,
                        }),
                        Val::Slot { slot: s, epoch },
                    ) => {
                        if epoch != self.epoch {
                            return Err(FoldError::StaleSlot);
                        }
                        (Operand::Scratch(s), idx)
                    }
                    _ => return Err(FoldError::WrongOperandKind(f)),
                };
                let n = self.finalize_and_run_scalar(Terminal::MaskedSumI32 { mask, lane })?;
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
                let Val::Slot { slot: s, epoch } = mask else {
                    return Err(FoldError::WrongOperandKind(f));
                };
                let Val::Address(Addr::Lane {
                    idx: val,
                    kind: LaneKind::I32,
                }) = val
                else {
                    return Err(FoldError::WrongOperandKind(f));
                };
                let mask = Operand::Scratch(s);
                let terminal = match key {
                    Val::Address(Addr::Lane {
                        idx: key,
                        kind: LaneKind::U32,
                    }) => Terminal::GroupSumI32 { mask, key, val },
                    Val::Address(Addr::Via { fk, key }) => {
                        Terminal::GroupSumViaI32 { mask, fk, key, val }
                    }
                    _ => return Err(FoldError::WrongOperandKind(f)),
                };
                // Epoch checked AFTER the shape checks so a wrong-kind body is
                // reported as wrong-kind, and BEFORE the fold, like SUM's arm.
                if epoch != self.epoch {
                    return Err(FoldError::StaleSlot);
                }
                match self.finalize_and_run(terminal, true)? {
                    Value::GroupSummed => Ok(()),
                    other => {
                        panic!("GROUP_SUM terminal shape guarantees GroupSummed, got {other:?}")
                    }
                }
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

/// Group-universe width K for the `GROUP_SUM` fixture.
const GROUPS: usize = 16;

/// The `GROUP_SUM` fixture: one `active` flag, one resident `direct_key`, and
/// a `VIA` spelling of the SAME key (`foreign_key[fk[i]]`), plus a signed
/// `value`. `direct_key` is BUILT from the VIA path, including both of its
/// drops, so resident and VIA addressing share one oracle while exercising
/// two different terminals.
struct GroupTables {
    active: Vec<u32>,
    direct_key: Vec<u32>,
    fk: Vec<u32>,
    value: Vec<i32>,
    foreign_key: Vec<u32>,
}

impl GroupTables {
    fn seeded(n: usize, foreign_rows: usize, seed: u64) -> Self {
        let mut s = seed;
        // Every 11th foreign row resolves OUTSIDE the group universe: the
        // second-hop drop.
        let foreign_key: Vec<u32> = (0..foreign_rows)
            .map(|i| {
                if i % 11 == 0 {
                    GROUPS as u32 + 3
                } else {
                    (lcg(&mut s) % GROUPS as u64) as u32
                }
            })
            .collect();
        // Every 13th row addresses NO foreign row: the first-hop drop.
        let fk: Vec<u32> = (0..n)
            .map(|i| {
                if i % 13 == 0 {
                    foreign_rows as u32 + 5
                } else {
                    (lcg(&mut s) % foreign_rows as u64) as u32
                }
            })
            .collect();
        // The resident key is the VIA key resolved ahead of time, with both
        // drops folded into one out-of-universe value.
        let direct_key = fk
            .iter()
            .map(|&a| {
                foreign_key
                    .get(a as usize)
                    .copied()
                    .filter(|&k| (k as usize) < GROUPS)
                    .unwrap_or(GROUPS as u32 + 7)
            })
            .collect();
        let active = (0..n).map(|_| (lcg(&mut s) & 1) as u32).collect();
        let value = (0..n).map(|_| (lcg(&mut s) % 401) as i32 - 200).collect();
        Self {
            active,
            direct_key,
            fk,
            value,
            foreign_key,
        }
    }

    fn lanes(&self) -> [LaneRef<'_>; 4] {
        [
            LaneRef::U32(&self.active),     // idx0
            LaneRef::U32(&self.direct_key), // idx1
            LaneRef::U32(&self.fk),         // idx2
            LaneRef::I32(&self.value),      // idx3
        ]
    }

    fn foreign_lanes(&self) -> [LaneRef<'_>; 1] {
        [LaneRef::U32(&self.foreign_key)]
    }

    /// `SUM(value WHERE active = 1) GROUP BY key`, over whichever key lane
    /// is passed — `direct_key` for the real oracle, `fk` for the wrong-
    /// terminal control. Out-of-universe keys drop.
    fn grouped_sum_by(&self, key: &[u32]) -> Vec<i64> {
        let mut out = vec![0i64; GROUPS];
        for ((&a, &k), &v) in self.active.iter().zip(key).zip(&self.value) {
            if a == 1 {
                if let Some(slot) = out.get_mut(k as usize) {
                    *slot += i64::from(v);
                }
            }
        }
        out
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

/// `SUM(value WHERE active = 1) GROUP BY direct_key` — the key is a resident
/// `U32` lane address, so `GROUP_SUM` lowers to `GroupSumI32`.
fn frontend_group_sum_direct() -> LocoProgram {
    let calls = [
        Call::with_value(FnIndex::NUMBER, 1),
        Call::with_value(FnIndex::NUMBER, 0), // active idx0
        Call::with_value(LOAD, 0),            // space=U32
        Call::new(INT_EQUAL),                 // -> mask
        Call::with_value(FnIndex::NUMBER, 1), // direct_key idx1
        Call::with_value(LOAD, 0),            // -> key address (U32 lane)
        Call::with_value(FnIndex::NUMBER, 3), // value idx3
        Call::with_value(LOAD, 1),            // -> value address (I32 lane)
        Call::new(GROUP_SUM),
    ];
    let body = FunctionBody::from_calls(LaneShape::Quads, &calls).expect("9 calls fit Quads");
    LocoProgram {
        functions: vec![body],
    }
}

/// The same query with the key spelled `foreign_key THROUGH fk` — a `VIA`
/// address, so the SAME `GROUP_SUM` byte lowers to `GroupSumViaI32`.
fn frontend_group_sum_via() -> LocoProgram {
    let calls = [
        Call::with_value(FnIndex::NUMBER, 1),
        Call::with_value(FnIndex::NUMBER, 0), // active idx0
        Call::with_value(LOAD, 0),
        Call::new(INT_EQUAL),                 // -> mask
        Call::with_value(FnIndex::NUMBER, 0), // key: Foreign::lanes idx0
        Call::with_value(FnIndex::NUMBER, 2), // fk: Planes::lanes idx2
        Call::new(VIA),                       // -> key address (VIA)
        Call::with_value(FnIndex::NUMBER, 3), // value idx3
        Call::with_value(LOAD, 1),
        Call::new(GROUP_SUM),
    ];
    let body = FunctionBody::from_calls(LaneShape::Quads, &calls).expect("10 calls fit Quads");
    LocoProgram {
        functions: vec![body],
    }
}

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
/// the measurement fields [`RunStats`] carries. Panics if the run completed
/// but was POISONED (branched on a population — see the module doc's
/// § MEASURED GAP): `Interpreter::run()` alone cannot see this, since a
/// poisoned branch still returns `Ok`, and a caller that only wants the
/// happy-path stack must never mistake a poisoned run for a clean one. A
/// caller that needs to OBSERVE poison without panicking uses
/// [`run_frontend_with_stats`] instead.
fn run_frontend(body: &LocoProgram, planes: &Planes<'_>, foreign: &Foreign<'_>) -> Vec<Val> {
    let vocab = validate(R2ILVocabulary).expect("R2ILVocabulary conforms");
    let dialect = FoldDialect::new(planes, foreign);
    let mut it = Interpreter::new(&vocab, body, dialect);
    it.run().expect("body runs to completion");
    if let Some(poison) = it.dialect().poison() {
        panic!(
            "run_frontend: the run completed Ok but was poisoned ({poison:?}) — \
             refusing to report it as success"
        );
    }
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
    /// The caller-owned `GROUP_SUM` sink, moved out after the run — empty
    /// unless the run was given one. Never a copy of the dialect's.
    group_sink: Vec<i64>,
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
    run_frontend_with_groups(body, planes, foreign, 0)
}

/// [`run_frontend_with_stats`] over a dialect writing `GROUP_SUM` results into
/// a sink of width `groups` (0 = no sink, the scalar-only frontends). The sink
/// is allocated once HERE, lent to the dialect, and moved into [`RunStats`]
/// after the interpreter is dropped — never cloned.
fn run_frontend_with_groups(
    body: &LocoProgram,
    planes: &Planes<'_>,
    foreign: &Foreign<'_>,
    groups: usize,
) -> Result<RunStats, ogar_loco::RunError<FoldError>> {
    let vocab = validate(R2ILVocabulary).expect("R2ILVocabulary conforms");
    let mut group_sink = vec![0i64; groups];
    let (stack, facade_passes, programs_run, poison) = {
        let dialect = FoldDialect::with_groups(planes, foreign, &mut group_sink);
        let mut it = Interpreter::new(&vocab, body, dialect);
        it.run()?;
        let d = it.dialect();
        (
            it.stack().to_vec(),
            d.facade_passes,
            d.programs_run,
            d.poison(),
        )
    };
    Ok(RunStats {
        stack,
        facade_passes,
        programs_run,
        group_sink,
        poison,
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
/// one full tile (`64 × TILE_WORDS` rows) and 64 tiles — `tile_words_for`
/// saturates at `TILE_WORDS` for any `n_rows` past one tile, so the
/// dialect's own scratch buffer, and every `MaskOp` it builds, is sized by
/// the BODY, never by the row count. Both populations are stated against
/// the constant: below one tile the scratch legitimately grows with rows.
#[test]
fn the_dialect_side_allocates_nothing_proportional_to_rows() {
    // The counter is THREAD-LOCAL (see the module-level note above `BYTES`),
    // so a sibling test running on another thread cannot contribute a
    // single byte here — the minimum-over-repetitions below is no longer
    // filtering cross-test contamination, only the small residual noise a
    // single thread's own allocator bookkeeping can introduce run to run.
    let a = frontend_a();
    let mut per_n = Vec::new();
    let one_tile = 64 * TILE_WORDS;
    for &n in &[one_tile, 64 * one_tile] {
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
        per_n[1] < 64 * one_tile / 8,
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
    // 64 tiles: large enough that one population mask (`n / 64` words)
    // outweighs the whole tile-local buffer across every slot.
    let n = 64 * 64 * TILE_WORDS;
    let t = Tables::seeded(n, 40, 9);
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
    // A population mask at `n` rows needs `n / 64` words for ONE slot
    // alone — more than the whole tile-local buffer this dialect ever
    // allocates, across every slot.
    assert!(dialect.scratch.len() < n / 64);
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

// ── epoch / stale-slot tests ────────────────────────────────────────────

/// FAILS IF: a [`Val::Slot`] minted before a fold executed — whose `ops`
/// graph that fold's `finalize_and_run` has since DISCARDED — is ever
/// accepted by a later consuming call as though it still named live
/// scratch. `finalize_and_run` resets `next_slot` to 0 on every fold, so a
/// stale slot's NUMBER can numerically alias a slot minted fresh in the new
/// epoch; only the epoch distinguishes them. Constructed directly at the
/// dialect level, which isolates the guard from the engine — for the same
/// state reached by a REAL loco body through `Interpreter::run`, see
/// `engine_control_flow_reaches_a_stale_slot_so_the_guard_is_load_bearing`.
/// ⊘ This comment previously said no reachable body could produce this
/// shape. That was wrong; see that test for the engine op that does.
#[test]
fn a_stale_slot_is_refused_not_silently_folded() {
    let t = Tables::seeded(16, 4, 9);
    let lanes = t.lanes();
    let flanes = t.foreign_lanes();
    let planes = Planes {
        n_rows: 16,
        masks: &[],
        lanes: &lanes,
    };
    let foreign = Foreign {
        planes: &[],
        lanes: &flanes,
    };
    let mut dialect = FoldDialect::new(&planes, &foreign);

    // Mint a slot in epoch 0, then hold onto it OFF the stack — as if a
    // (hypothetical, currently unreachable) body kept a reference across a
    // fold boundary.
    let mut throwaway: Vec<Val> = Vec::new();
    dialect.emit_pred(Pred::EqI32 { lane: 1, v: 0 }, &mut throwaway);
    let stale = throwaway.pop().expect("emit_pred pushed a slot");

    // Run a fold to completion: this is what discards the ops graph `stale`
    // referred to, advances the epoch, and resets `next_slot` back to 0.
    let mut fold_stack: Vec<Val> = Vec::new();
    dialect.emit_pred(Pred::EqI32 { lane: 1, v: 1 }, &mut fold_stack);
    dialect
        .call(POP_COUNT, [0, 0, 0], &mut fold_stack)
        .expect("the fold runs to completion");

    // A slot minted in the NEW (current) epoch, after the fold — numerically
    // slot 0 again, since `next_slot` was reset. POSITIVE CONTROL: without
    // this half, an implementation that refused EVERY slot, stale or not,
    // would also pass.
    let mut fresh_stack: Vec<Val> = Vec::new();
    dialect.emit_pred(Pred::EqI32 { lane: 1, v: 2 }, &mut fresh_stack);
    let lane_addr = Val::Address(Addr::Lane {
        idx: 1,
        kind: LaneKind::I32,
    });
    fresh_stack.push(lane_addr);
    dialect
        .call(SUM, [0, 0, 0], &mut fresh_stack)
        .expect("a slot from the CURRENT epoch is accepted by SUM");

    // NEGATIVE: `stale` (minted before the fold, in the now-dead epoch) fed
    // to SUM must be refused, not silently folded — folding it would run
    // SUM against whatever the CURRENT epoch's slot 0 actually holds, which
    // has nothing to do with the predicate `stale` originally named.
    let mut stale_call_stack = vec![stale, lane_addr];
    let err = dialect
        .call(SUM, [0, 0, 0], &mut stale_call_stack)
        .expect_err("a stale slot must be refused");
    assert_eq!(err, FoldError::StaleSlot);
}

/// FAILS IF: any implemented op ever pushes anything other than EXACTLY ONE
/// value per call. Every value-producing arm in `Dialect::call` pops its
/// arity's worth of operands and pushes exactly one result, so THE DIALECT
/// never shrinks the stack past a value sitting beneath a fold's own result.
/// Covers the implemented set — `NUMBER`, `LOAD`, `VIA`, `INT_EQUAL`,
/// `INT_S_LESS`, `INT_AND`, `INT_SUB`, `POP_COUNT`, `SUM` — driving the
/// refusal arms too would add nothing: a refusal never reaches a push.
///
/// `GROUP_SUM` is the one implemented op that does NOT push exactly one: it
/// is a write terminal, pops three and pushes nothing. That exception is
/// pinned at the bottom of this test (so a later "push a status scalar"
/// change cannot slip in unnoticed) and its consequence — the stack shrinks
/// past whatever lay beneath the grouped fold — is pinned by
/// `group_sum_exposes_what_lay_beneath_it_and_the_epoch_guard_refuses_it`.
///
/// ⊘ WHAT THIS DOES NOT PROVE, corrected after review. This comment used to
/// conclude that the push-exactly-one property makes a stale slot
/// UNREACHABLE from any real loco body, with `Store` named as the op whose
/// arrival would change that. The property is true and worth pinning; the
/// conclusion was false. `IF` is never dispatched to a `Dialect` at all —
/// `ogar_loco`'s engine handles it in `run_branching`, popping the condition
/// and pushing nothing, as `IF_ELSE` and `REPEAT` also do. So a census of
/// this trait's arms structurally cannot see the consumer that removes the
/// barrier, and the `FoldError::StaleSlot` guard is load-bearing TODAY. The
/// body that reaches it is
/// `engine_control_flow_reaches_a_stale_slot_so_the_guard_is_load_bearing`.
/// The lesson generalises past this file: an exhaustive census of one
/// dispatch surface says nothing about a second dispatch surface above it.
#[test]
fn no_implemented_op_can_expose_a_stale_slot() {
    let t = Tables::seeded(16, 4, 13);
    let lanes = t.lanes();
    let flanes = t.foreign_lanes();
    let planes = Planes {
        n_rows: 16,
        masks: &[],
        lanes: &lanes,
    };
    let foreign = Foreign {
        planes: &[],
        lanes: &flanes,
    };

    /// Asserts `after == before - arity + 1` — exactly one value pushed,
    /// regardless of arity — and prints the op so a failure names which one.
    fn assert_pushes_exactly_one(f: FnIndex, arity: usize, before: usize, after: usize) {
        assert_eq!(
            after,
            before - arity + 1,
            "{f:?}: expected exactly one value pushed (arity {arity}), \
             stack went {before} -> {after}"
        );
    }

    // NUMBER: arity 0.
    {
        let mut dialect = FoldDialect::new(&planes, &foreign);
        let mut stack: Vec<Val> = Vec::new();
        let before = stack.len();
        dialect
            .call(FnIndex::NUMBER, [7, 0, 0], &mut stack)
            .expect("NUMBER always succeeds");
        assert_pushes_exactly_one(FnIndex::NUMBER, 0, before, stack.len());
    }
    // LOAD: arity 1 (pops an index scalar), space=1 (I32).
    {
        let mut dialect = FoldDialect::new(&planes, &foreign);
        let mut stack = vec![Val::Scalar(1)];
        let before = stack.len();
        dialect
            .call(LOAD, [1, 0, 0], &mut stack)
            .expect("LOAD of a valid space/index succeeds");
        assert_pushes_exactly_one(LOAD, 1, before, stack.len());
    }
    // VIA: arity 2 (pops key then fk, both scalars).
    {
        let mut dialect = FoldDialect::new(&planes, &foreign);
        let mut stack = vec![Val::Scalar(0), Val::Scalar(0)];
        let before = stack.len();
        dialect
            .call(VIA, [0, 0, 0], &mut stack)
            .expect("VIA of two scalars succeeds");
        assert_pushes_exactly_one(VIA, 2, before, stack.len());
    }
    // INT_EQUAL: arity 2, the scalar/scalar (plain-arithmetic) shape.
    {
        let mut dialect = FoldDialect::new(&planes, &foreign);
        let mut stack = vec![Val::Scalar(3), Val::Scalar(3)];
        let before = stack.len();
        dialect
            .call(INT_EQUAL, [0, 0, 0], &mut stack)
            .expect("INT_EQUAL of two scalars succeeds");
        assert_pushes_exactly_one(INT_EQUAL, 2, before, stack.len());
    }
    // INT_S_LESS: arity 2, the (Address::Lane I32, Scalar) shape.
    {
        let mut dialect = FoldDialect::new(&planes, &foreign);
        let mut stack = vec![
            Val::Address(Addr::Lane {
                idx: 1,
                kind: LaneKind::I32,
            }),
            Val::Scalar(0),
        ];
        let before = stack.len();
        dialect
            .call(INT_S_LESS, [0, 0, 0], &mut stack)
            .expect("INT_S_LESS of a lane and a scalar succeeds");
        assert_pushes_exactly_one(INT_S_LESS, 2, before, stack.len());
    }
    // INT_AND: arity 2, the (Slot, Slot) shape — drives the peephole.
    {
        let mut dialect = FoldDialect::new(&planes, &foreign);
        let mut stack: Vec<Val> = Vec::new();
        dialect.emit_pred(Pred::EqI32 { lane: 1, v: 0 }, &mut stack);
        dialect.emit_pred(Pred::EqI32 { lane: 1, v: 1 }, &mut stack);
        let before = stack.len();
        dialect
            .call(INT_AND, [0, 0, 0], &mut stack)
            .expect("INT_AND of two current-epoch slots succeeds");
        assert_pushes_exactly_one(INT_AND, 2, before, stack.len());
    }
    // INT_SUB: arity 2, the scalar/scalar shape.
    {
        let mut dialect = FoldDialect::new(&planes, &foreign);
        let mut stack = vec![Val::Scalar(5), Val::Scalar(2)];
        let before = stack.len();
        dialect
            .call(INT_SUB, [0, 0, 0], &mut stack)
            .expect("INT_SUB of two scalars succeeds");
        assert_pushes_exactly_one(INT_SUB, 2, before, stack.len());
    }
    // POP_COUNT: arity 1 — finalizes and RUNS a fold, still pushes exactly
    // one `Val::Scalar`.
    {
        let mut dialect = FoldDialect::new(&planes, &foreign);
        let mut stack: Vec<Val> = Vec::new();
        dialect.emit_pred(Pred::EqI32 { lane: 1, v: 0 }, &mut stack);
        let before = stack.len();
        dialect
            .call(POP_COUNT, [0, 0, 0], &mut stack)
            .expect("POP_COUNT of a current-epoch slot succeeds");
        assert_pushes_exactly_one(POP_COUNT, 1, before, stack.len());
    }
    // SUM: arity 2 — also finalizes and runs a fold.
    {
        let mut dialect = FoldDialect::new(&planes, &foreign);
        let mut stack: Vec<Val> = Vec::new();
        dialect.emit_pred(Pred::EqI32 { lane: 1, v: 0 }, &mut stack);
        stack.push(Val::Address(Addr::Lane {
            idx: 1,
            kind: LaneKind::I32,
        }));
        let before = stack.len();
        dialect
            .call(SUM, [0, 0, 0], &mut stack)
            .expect("SUM of a current-epoch slot and a lane address succeeds");
        assert_pushes_exactly_one(SUM, 2, before, stack.len());
    }
    // GROUP_SUM: arity 3, pushes ZERO — the documented exception. Built with
    // a sink, over the `Tables` fixture's own U32 (`fk`, idx0) and I32
    // (`amount`, idx1) lanes.
    {
        let mut sink = [0i64; 4];
        let mut dialect = FoldDialect::with_groups(&planes, &foreign, &mut sink);
        let mut stack: Vec<Val> = Vec::new();
        dialect.emit_pred(Pred::EqI32 { lane: 1, v: 0 }, &mut stack);
        stack.push(Val::Address(Addr::Lane {
            idx: 0,
            kind: LaneKind::U32,
        }));
        stack.push(Val::Address(Addr::Lane {
            idx: 1,
            kind: LaneKind::I32,
        }));
        let before = stack.len();
        dialect
            .call(GROUP_SUM, [0, 0, 0], &mut stack)
            .expect("GROUP_SUM of a slot, a U32 key and an I32 value succeeds");
        assert_eq!(
            stack.len(),
            before - 3,
            "GROUP_SUM is a write terminal: pops three, pushes nothing"
        );
    }
}

/// The case that decides how bad a stale slot actually is: a stale slot
/// number that is **live in the current epoch**.
///
/// `a_stale_slot_is_refused_not_silently_folded` proves the guard fires, but
/// its disable run can only ever show `mask-risc`'s own
/// `ScratchReadBeforeWrite`, because its positive control folds and so clears
/// `ops` again -- by the negative case nothing has written slot 0. That makes
/// the downstream net look like it already covers this. It does not.
///
/// Here the current epoch HAS written slot 0, by minting a predicate and
/// deliberately NOT folding it, so with the epoch check removed there is no
/// read-before-write to catch. The program is well-formed and answers under
/// the WRONG predicate: a silent wrong number, not an error.
///
/// Three anti-vacuity guards, because each one is a way this fixture could
/// measure nothing: the two slots must ALIAS by number, their two answers
/// must DIFFER (otherwise the aliasing is unobservable even when it happens),
/// and the live slot's accepted answer is checked against an independent
/// oracle so the positive half cannot pass on a wrong-but-successful result.
#[test]
fn a_stale_slot_that_aliases_a_live_slot_is_refused_not_answered_wrongly() {
    // `amount` (lane 1) is in [-100, 99], so these two thresholds select very
    // different populations and therefore very different sums.
    const T_STALE: i32 = -100;
    const T_LIVE: i32 = 50;

    let t = Tables::seeded(64, 8, 11);
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

    let sum_over = |thr: i32| -> i64 {
        t.amount
            .iter()
            .filter(|&&a| a > thr)
            .map(|&a| i64::from(a))
            .sum()
    };
    let expect_live = sum_over(T_LIVE);
    assert_ne!(
        sum_over(T_STALE),
        expect_live,
        "the two predicates must disagree on this fixture, or aliasing one for \
         the other could never be observed"
    );

    let mut dialect = FoldDialect::new(&planes, &foreign);

    // Epoch 0: mint a slot and hold it off-stack, as a body keeping a
    // reference across a fold boundary would.
    let mut held: Vec<Val> = Vec::new();
    dialect.emit_pred(
        Pred::GtI32 {
            lane: 1,
            t: T_STALE,
        },
        &mut held,
    );
    let stale = held.pop().expect("emit_pred pushed a slot");

    // Advance the epoch by folding something unrelated.
    let mut warm: Vec<Val> = Vec::new();
    dialect.emit_pred(Pred::GtI32 { lane: 1, t: 0 }, &mut warm);
    dialect
        .call(POP_COUNT, [0, 0, 0], &mut warm)
        .expect("the warm-up fold runs");

    // Epoch 1: mint the live predicate. It takes the same slot NUMBER, since
    // next_slot was reset. Do NOT fold it -- `ops` must still carry its write
    // when the stale slot is used below, or the downstream read-before-write
    // guard would mask the aliasing this test exists to expose.
    let mut live: Vec<Val> = Vec::new();
    dialect.emit_pred(Pred::GtI32 { lane: 1, t: T_LIVE }, &mut live);
    let fresh = *live.last().expect("a live slot");
    assert_eq!(
        slot_number_of(stale),
        slot_number_of(fresh),
        "the fixture is only meaningful if the two slots ALIAS: same number, \
         different epoch. If next_slot stops resetting, this measures nothing"
    );

    let lane_addr = Val::Address(Addr::Lane {
        idx: 1,
        kind: LaneKind::I32,
    });

    // NEGATIVE. Without the epoch check this returns Ok, carrying the sum
    // taken under T_LIVE while the caller believes it asked for T_STALE.
    let mut bad: Vec<Val> = vec![stale, lane_addr];
    assert_eq!(
        dialect.call(SUM, [0, 0, 0], &mut bad),
        Err(FoldError::StaleSlot),
        "a stale slot aliasing a live one must be refused, never answered"
    );

    // POSITIVE. The fresh slot of the same number is accepted and answers
    // under its own predicate.
    let mut good: Vec<Val> = vec![fresh, lane_addr];
    dialect
        .call(SUM, [0, 0, 0], &mut good)
        .expect("a current-epoch slot is accepted by SUM");
    match good.as_slice() {
        [Val::Scalar(n)] => assert_eq!(
            *n, expect_live,
            "the live slot must answer under ITS OWN predicate"
        ),
        other => panic!("SUM should leave exactly one scalar, got {other:?}"),
    }
}

/// The body that proves the stale-slot guard is LOAD-BEARING TODAY, not
/// defensive against a future `Store`.
///
/// ⊘ This test exists because the reachability argument shipped in #1259's
/// first draft was WRONG, and wrong in an instructive way: it enumerated the
/// arms of [`FoldDialect::call`], found every one pushes exactly one result,
/// and concluded the stack can never shrink past a value beneath a fold's
/// result. The enumeration was correct. The conclusion did not follow,
/// because `IF` is never dispatched to a `Dialect` at all — `ogar_loco`'s
/// engine handles it in `run_branching`, where it does `self.pop(f)` for the
/// condition and pushes NOTHING. `IF_ELSE` and `REPEAT` do the same. So the
/// engine is a consumer-without-push that a census of the dialect's own
/// match arms structurally cannot see. Found by review, not by the suite.
///
/// The sequence, which is a valid body and never poisons:
///
/// 1. a predicate mints `S0`
/// 2. a predicate that matches NOTHING mints `S1`
/// 3. `POP_COUNT(S1)` folds, so the epoch advances and `next_slot` resets,
///    leaving `[S0, Scalar(0)]`
/// 4. `IF` consumes the `Scalar(0)`. `truthy` on a `Scalar` is an ordinary
///    answer, not the poison path, and 0 is false so nothing branches. The
///    stack is now `[S0]` — the barrier is GONE
/// 5. a predicate mints `S2` in the current epoch
/// 6. `INT_AND` consumes both, and `S0` is a reachable stale slot
///
/// Asserted: the run fails with `RunError::Dialect(FoldError::StaleSlot)`
/// AND the poison is unset, because a poisoned run would mean the body took
/// the branch-on-population path and this sequence would be proving
/// something else.
///
/// WHY THAT EXPECTED VALUE IS DISCRIMINATING, not merely correct — the
/// argument is the reviewer's, added here because a future reader will
/// otherwise raise the same doubt the author did. `INT_AND` accepts
/// `(Slot, Slot)` or `(Scalar, Scalar)` and answers anything mixed with
/// `WrongOperandKind`. So each way this fixture could be passing for the
/// wrong reason produces a DIFFERENT error:
///
/// - if `POP_COUNT` on the empty mask yielded something truthy, `IF` would
///   branch into `never_taken`, which pushes a scalar, and step 6 would see
///   `(Scalar, Slot)` → `WrongOperandKind`
/// - if `IF` did NOT pop its condition, step 6 would see `(Scalar, Slot)`
///   for the same reason → `WrongOperandKind`
/// - `poison() == None` independently witnesses that `IF` received a
///   `Scalar` and not a `Slot`
///
/// `StaleSlot` is therefore reachable only through the intended path, which
/// is what makes the assertion evidence rather than a coincidence. The
/// disable run agrees from the other side: with all four epoch checks
/// removed the body returns `Ok(())`.
#[test]
fn engine_control_flow_reaches_a_stale_slot_so_the_guard_is_load_bearing() {
    // `amount` (lane 1, I32) is in [-100, 99], so 200 matches no row and the
    // fold at step 3 yields exactly 0 — which is what makes the IF fall
    // through rather than branch.
    const NO_MATCH: u8 = 200;
    const EMPTY_BODY_TARGET: u8 = 1;

    let calls = [
        // 1. S0
        Call::with_value(FnIndex::NUMBER, 0),
        Call::with_value(FnIndex::NUMBER, 1),
        Call::with_value(LOAD, 1),
        Call::new(INT_EQUAL),
        // 2. S1, an empty mask
        Call::with_value(FnIndex::NUMBER, NO_MATCH),
        Call::with_value(FnIndex::NUMBER, 1),
        Call::with_value(LOAD, 1),
        Call::new(INT_EQUAL),
        // 3. fold it: epoch advances, next_slot resets, pushes Scalar(0)
        Call::new(POP_COUNT),
        // 4. the engine eats the scalar and pushes nothing
        Call::with_value(FnIndex::IF, EMPTY_BODY_TARGET),
        // 5. S2, current epoch, same slot NUMBER as the stale S0
        Call::with_value(FnIndex::NUMBER, 1),
        Call::with_value(FnIndex::NUMBER, 1),
        Call::with_value(LOAD, 1),
        Call::new(INT_EQUAL),
        // 6. S0 is now reachable, and stale
        Call::new(INT_AND),
    ];
    let main = FunctionBody::from_calls(LaneShape::Quads, &calls).expect("15 calls fit Quads");
    // Target 1 resolves to functions[1] (`body_at` is 0-based and rejects 0),
    // so the program is well formed even though step 4 never branches.
    let never_taken =
        FunctionBody::from_calls(LaneShape::Pairs, &[Call::with_value(FnIndex::NUMBER, 0)])
            .expect("one call fits Pairs");
    let body = LocoProgram {
        functions: vec![main, never_taken],
    };

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

    let vocab = validate(R2ILVocabulary).expect("R2ILVocabulary conforms");
    let dialect = FoldDialect::new(&planes, &foreign);
    let mut it = Interpreter::new(&vocab, &body, dialect);
    let outcome = it.run();

    assert_eq!(
        outcome,
        Err(ogar_loco::RunError::Dialect(FoldError::StaleSlot)),
        "engine control flow removes the barrier, so INT_AND reaches a stale \
         slot and must be refused BY THE EPOCH GUARD"
    );
    assert_eq!(
        it.dialect().poison(),
        None,
        "this body branches on a SCALAR, so it must not poison -- a poisoned \
         run would mean the fixture proved the branch-on-population path \
         instead of the stale-slot one"
    );
}

/// The slot number inside a [`Val::Slot`], for fixtures that must prove two
/// slots alias. Panics on any other variant: a caller asking this of a
/// non-slot has a broken fixture, not a runtime condition to handle.
fn slot_number_of(v: Val) -> u16 {
    match v {
        Val::Slot { slot, .. } => slot,
        other => panic!("expected a slot, got {other:?}"),
    }
}

/// FAILS IF: `run_frontend` reports a poisoned run (one that branched on a
/// population — see `cbranch_on_a_slot_poisons_but_does_not_abort`, whose
/// body this test reuses verbatim) as though it were a clean success.
/// `Interpreter::run()` itself returns `Ok` for this body (a poisoned branch
/// is not a run error — see the module doc's § MEASURED GAP), so
/// `run_frontend` must check [`FoldDialect::poison`] itself before handing
/// the stack back; a silent `Ok` here would be indistinguishable from a body
/// that never branched on a population at all.
#[test]
#[should_panic(expected = "poisoned")]
fn a_poisoned_run_is_not_reported_as_success() {
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
    // Same body as `cbranch_on_a_slot_poisons_but_does_not_abort`: a Slot
    // reaches `IF`, `truthy` answers `false` and sets the poison flag, and
    // the run completes `Ok` regardless.
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
    let _ = run_frontend(&program, &planes, &foreign);
}

// ── GROUP_SUM ────────────────────────────────────────────────────────────

/// FAILS IF: resident and `VIA` addressing through the ONE `GROUP_SUM` byte
/// disagree with each other or with an independent grouped oracle, at any
/// row boundary — or if the key address stops deciding the terminal.
///
/// Both frontends are the same query. The resident one names `direct_key`,
/// the `VIA` one names `foreign_key THROUGH fk`; `direct_key` is built from
/// that very path, drops included, so one oracle serves both and a
/// disagreement can only come from the lowering.
///
/// Anti-vacuity, because each is a way this could measure nothing:
/// - many groups are non-zero (an all-zero sink would equal an all-zero
///   oracle whatever the terminal did);
/// - both VIA drops actually fire on active rows (otherwise the drop paths
///   are untested and a terminal that ignored them would still pass);
/// - the wrong-terminal answer (resident `GroupSumI32` over the `fk` lane,
///   i.e. a lowering that saw `VIA` and forgot the second hop) DIFFERS from
///   the oracle, so a selector that confused the two address kinds fails.
#[test]
fn group_sum_direct_and_via_both_equal_the_grouped_oracle() {
    for (n, seed) in [
        (63usize, 3u64),
        (64, 5),
        (130, 7),
        (1000, 11),
        (4096, 0x1261),
    ] {
        let t = GroupTables::seeded(n, 47, seed);
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
        let expect = t.grouped_sum_by(&t.direct_key);

        let direct =
            run_frontend_with_groups(&frontend_group_sum_direct(), &planes, &foreign, GROUPS)
                .expect("resident GROUP_SUM runs");
        let via = run_frontend_with_groups(&frontend_group_sum_via(), &planes, &foreign, GROUPS)
            .expect("VIA GROUP_SUM runs");

        for (name, r) in [("resident", &direct), ("VIA", &via)] {
            assert_eq!(r.group_sink, expect, "n={n}: {name} grouped result");
            assert!(
                r.stack.is_empty(),
                "n={n}: {name}: GROUP_SUM pushes nothing"
            );
            assert_eq!(r.programs_run, 1, "n={n}: {name}: one grouped fold");
            assert_eq!(r.facade_passes, 1, "n={n}: {name}: one shared predicate");
            assert_eq!(r.poison, None, "n={n}: {name}: no population branch");
        }
        assert_eq!(direct.group_sink, via.group_sink, "n={n}: resident == VIA");

        if n == 4096 {
            let nonzero = expect.iter().filter(|&&x| x != 0).count();
            assert!(
                nonzero >= GROUPS / 2,
                "anti-vacuity: {nonzero} non-zero groups"
            );
            let first_hop_drops = (0..n)
                .filter(|&i| t.active[i] == 1 && t.fk[i] as usize >= t.foreign_key.len())
                .count();
            let second_hop_drops = (0..n)
                .filter(|&i| {
                    t.active[i] == 1
                        && t.foreign_key
                            .get(t.fk[i] as usize)
                            .is_some_and(|&k| k as usize >= GROUPS)
                })
                .count();
            assert!(
                first_hop_drops > 0,
                "anti-vacuity: first VIA hop never drops"
            );
            assert!(
                second_hop_drops > 0,
                "anti-vacuity: second VIA hop never drops"
            );
            assert_ne!(
                t.grouped_sum_by(&t.fk),
                expect,
                "anti-vacuity: a VIA lowered as a resident fk-keyed sum must be wrong"
            );
        }
    }
}

/// FAILS IF: `GROUP_SUM` accepts any operand shape other than
/// `(current-epoch slot, U32-lane-or-VIA key, I32-lane value)` — or refuses
/// the one it must accept. The positive control is the first case: without
/// it, an arm that refused everything would pass the rest.
#[test]
fn group_sum_refuses_every_other_operand_shape() {
    let t = GroupTables::seeded(64, 7, 17);
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
    let u32_key = Val::Address(Addr::Lane {
        idx: 1,
        kind: LaneKind::U32,
    });
    let i32_val = Val::Address(Addr::Lane {
        idx: 3,
        kind: LaneKind::I32,
    });
    let i32_as_key = Val::Address(Addr::Lane {
        idx: 3,
        kind: LaneKind::I32,
    });
    let u32_as_val = Val::Address(Addr::Lane {
        idx: 1,
        kind: LaneKind::U32,
    });
    let via_as_val = Val::Address(Addr::Via { fk: 2, key: 0 });

    // Each case: (label, key, value, mask-is-a-slot, expected).
    type ShapeCase = (&'static str, Val, Val, bool, Result<(), FoldError>);
    let cases: [ShapeCase; 6] = [
        ("positive control", u32_key, i32_val, true, Ok(())),
        (
            "I32 lane as key",
            i32_as_key,
            i32_val,
            true,
            Err(FoldError::WrongOperandKind(GROUP_SUM)),
        ),
        (
            "scalar as key",
            Val::Scalar(1),
            i32_val,
            true,
            Err(FoldError::WrongOperandKind(GROUP_SUM)),
        ),
        (
            "U32 lane as value",
            u32_key,
            u32_as_val,
            true,
            Err(FoldError::WrongOperandKind(GROUP_SUM)),
        ),
        (
            "VIA as value",
            u32_key,
            via_as_val,
            true,
            Err(FoldError::WrongOperandKind(GROUP_SUM)),
        ),
        (
            "scalar as mask",
            u32_key,
            i32_val,
            false,
            Err(FoldError::WrongOperandKind(GROUP_SUM)),
        ),
    ];
    for (label, key, val, mask_is_slot, expected) in cases {
        let mut sink = [0i64; GROUPS];
        let mut d = FoldDialect::with_groups(&planes, &foreign, &mut sink);
        let mut stack: Vec<Val> = Vec::new();
        if mask_is_slot {
            d.emit_pred(Pred::EqU32 { lane: 0, v: 1 }, &mut stack);
        } else {
            stack.push(Val::Scalar(1));
        }
        stack.push(key);
        stack.push(val);
        assert_eq!(
            d.call(GROUP_SUM, [0, 0, 0], &mut stack),
            expected,
            "{label}"
        );
    }

    // Stale mask: minted in epoch 0, then a POP_COUNT fold ends that epoch.
    // Every other operand is well-shaped, so only the epoch can refuse it.
    let mut sink = [0i64; GROUPS];
    let mut d = FoldDialect::with_groups(&planes, &foreign, &mut sink);
    let mut held: Vec<Val> = Vec::new();
    d.emit_pred(Pred::EqU32 { lane: 0, v: 1 }, &mut held);
    let stale = held.pop().expect("emit_pred pushed a slot");
    let mut fold: Vec<Val> = Vec::new();
    d.emit_pred(Pred::EqU32 { lane: 0, v: 0 }, &mut fold);
    d.call(POP_COUNT, [0, 0, 0], &mut fold)
        .expect("the intervening fold runs");
    let mut stack = vec![stale, u32_key, i32_val];
    assert_eq!(
        d.call(GROUP_SUM, [0, 0, 0], &mut stack),
        Err(FoldError::StaleSlot),
        "a mask from a finished epoch"
    );

    // Underflow: two operands where three are needed.
    let mut sink = [0i64; GROUPS];
    let mut d = FoldDialect::with_groups(&planes, &foreign, &mut sink);
    let mut stack = vec![u32_key, i32_val];
    assert_eq!(
        d.call(GROUP_SUM, [0, 0, 0], &mut stack),
        Err(FoldError::Underflow(GROUP_SUM))
    );
}

/// FAILS IF: a slot minted BEFORE a `GROUP_SUM` fold, and left beneath it on
/// the stack, can be consumed AFTER the fold as though it were still live.
///
/// This is new reachability, not a restatement of the IF case. Until now every
/// dialect-dispatched op pushed exactly one value, so only the engine's
/// `IF`/`IF_ELSE`/`REPEAT` could shrink the stack past a fold's operands.
/// `GROUP_SUM` pushes nothing, so it does the same from INSIDE the dialect:
/// the body below leaves slot A under the grouped fold, and after the fold ends
/// the epoch, A is back on top for `POP_COUNT` with a dead epoch.
///
/// Positive control: the same body WITHOUT the leading slot A runs cleanly to
/// the same grouped answer, so the refusal is the exposed stale slot and not
/// a defect in `GROUP_SUM` itself.
#[test]
fn group_sum_exposes_what_lay_beneath_it_and_the_epoch_guard_refuses_it() {
    let t = GroupTables::seeded(256, 23, 41);
    let lanes = t.lanes();
    let flanes = t.foreign_lanes();
    let planes = Planes {
        n_rows: 256,
        masks: &[],
        lanes: &lanes,
    };
    let foreign = Foreign {
        planes: &[],
        lanes: &flanes,
    };
    let mask = [
        Call::with_value(FnIndex::NUMBER, 1),
        Call::with_value(FnIndex::NUMBER, 0), // active idx0
        Call::with_value(LOAD, 0),
        Call::new(INT_EQUAL),
    ];
    let group_sum = [
        Call::with_value(FnIndex::NUMBER, 1), // direct_key idx1
        Call::with_value(LOAD, 0),
        Call::with_value(FnIndex::NUMBER, 3), // value idx3
        Call::with_value(LOAD, 1),
        Call::new(GROUP_SUM),
    ];
    let program = |calls: Vec<Call>| LocoProgram {
        functions: vec![FunctionBody::from_calls(LaneShape::Quads, &calls).expect("fits Quads")],
    };

    // Positive control: mask + GROUP_SUM, nothing beneath.
    let clean: Vec<Call> = mask.iter().chain(&group_sum).copied().collect();
    let ok = run_frontend_with_groups(&program(clean), &planes, &foreign, GROUPS)
        .expect("a clean grouped fold runs");
    assert_eq!(ok.group_sink, t.grouped_sum_by(&t.direct_key));
    assert!(ok.stack.is_empty());

    // Slot A beneath, then the grouped fold, then consume whatever is on top.
    let exposed: Vec<Call> = mask
        .iter()
        .chain(&mask)
        .chain(&group_sum)
        .copied()
        .chain([Call::new(POP_COUNT)])
        .collect();
    let err = run_frontend_with_groups(&program(exposed), &planes, &foreign, GROUPS)
        .err()
        .expect("a stale slot exposed by GROUP_SUM must be refused");
    assert!(
        matches!(err, ogar_loco::RunError::Dialect(FoldError::StaleSlot)),
        "expected StaleSlot, got {err:?}"
    );
}
