# 2026-09-23 — CubeCL / LLVM boundary, and the regrade of the CubeCL scheduler audit

**Status:** DECISION (operator) · OPEN (shader-driver delta dispatch; the future scheduler)
**Supersedes:** the chat-only "CubeCL fold-before-steal audit" (never on the board). Its
proposed next step, a CubeCL dispatcher PR, is dropped.

## DECISION
- **Ours, never refactored toward CubeCL or LLVM:** the T0 fold-before-dispatch law; the
  masking substrate (bit planes, ternlog, the keyed-reduction family, `ndarray::simd` as the
  only polyfill); R2IL as the IR. A representation we invented is never reshaped to match an
  external one, however familiar the external one is.
- **CubeCL is lab-only:** a comparison point for `cognitive-shader-driver`, to inspire shader
  and scheduling ideas. The CubeCL lab fork PR stays lab hygiene only.
- **The scheduler may borrow ideas** (rayon vs CubeCL, bare-metal folding, Mississippi Queen
  board economics, hexagon) and rebuild them as OUR scheduler, which sits ABOVE the fold.
- **LLVM only as a PARALLEL compiler arm**, fed the same folded program. JIT codegen (compile
  per query at run time) and folding into an explicit precompiled polyfill are different
  philosophies; neither reshapes the other.

**SCOPE:** every lowering, scheduler, shader-driver and backend decision in lance-graph and
ndarray. **BASIS:** operator direction, 2026-09-23. **REVISIT WHEN:** a measured result from
the parallel arm exists — and even then any argument for changing the IR, the fold law or the
polyfill goes to the operator, never straight into code.

## Regrade of the old audit

| Old section | Verdict |
|---|---|
| §1 #1260 residue (failed fold retires its slots; unobservable today) | KEEP |
| §2 #1261 residue; `GroupLowering::{Folded, Forest}` as the scheduler seam | KEEP. Still only quack's own tests call `lower_group_by_auto` (`crates/lance-graph-quack/src/lib.rs`). |
| §3 CubeCL CPU scheduler mechanics | KEEP as lab notes; reframed as lessons below |
| §4 8→64 overflow workers, suspected cross-stream deadlock | REGRADE to anti-patterns; fixing a lab reference is not our work |
| §5 tier split drawing "T0.5 CubeCL IR → LLVM O3" beside T0, with CubeCL's kernel types as the seam | **REWRITE** (below) — this was the boundary violation |
| §6 invariant ownership | lance-graph half KEEP; the CubeCL half becomes laws for OUR scheduler |
| §7 four falsifiers written in CubeCL | RE-TARGET at our scheduler, when it exists |
| "Next PR: CubeCL" | DROP |

## Corrected tier picture

```
T1 semantics ─fold (T2)─► folded R2IL program ─► OUR dispatch unit ─► OUR scheduler
                                  │              (program + requirements + extent + affinity)
                                  └─► [parallel lab arm] LLVM/CubeCL compile of the SAME
                                       folded program → result + timing, compared only
```
T0 realization is `mask-risc` / `ndarray::simd` — one path, not a fork. The parallel arm
answers only: do results agree (an oracle); what are JIT compile+run vs fold+dispatch
latencies (never measured here); where does fused codegen beat the fold (a finding for us
to answer our own way).

## Lessons for OUR scheduler
From reading CubeCL's CPU runtime (not built or run here; its CPU backend needs native LLVM):
1. **Extent is a first-class field of the dispatch unit.** CubeCL buries its stealable unit
   (a cube range) inside compiled code with no ABI slot for it. mask-risc already exposes it:
   the unit is `(program, tile range)`.
2. **A waiting task must not stall its worker.** CubeCL's 4-slot parked queue does exactly
   that, and is the likely route to the suspected cross-stream deadlock.
3. **Special pools stay isolated.** CubeCL's barrier-overflow workers leak into ordinary work
   permanently.
4. **Split by extent, merge by the fold's own algebra** — see
   `TD-SYM-SUM-MERGE-IS-NOT-ADDITION-1` for the `_sym` SUM merge law.

rayon splits recursively over a range (extent-native); CubeCL launches a fixed geometry. Our
fold turns most work into one small program over K slots, so the default is **inline**;
splitting is for the `Forest` residue or very large tile ranges only.

Mississippi Queen / hexagon stays **[H]**: `.claude/plans/spog-alpha-channel-v1.md` §4 already
says *topology chooses neighborhood, masks choose admissibility, BLAS chooses magnitude*. For
scheduling that suggests an MQ-style budget pricing which tiles to dispatch next. Not
buildable: the MQ source docs are still missing (`.claude/board/LATEST_STATE.md`, "no MQ docs").

**Laws the future scheduler must carry:** exactly-once; barrier affinity; pool isolation;
blocked-doesn't-block; extent-merge-by-fold-algebra. The scheduler itself remains gated on
OQ-5 (rayon vendor vs `std::thread::scope`, `.claude/plans/causaledge64-mailbox-rename-soa-v1.md`,
`STATUS_BOARD.md` row D-CE64-MB-1-impl).

## The residue is already native (operator framing)
The HAVING work shows the architecture already doing this "T0.5-ish" job in its own
vocabulary. The residue is not a task object or a generic IR. It is **group slot + reserved
empty code + exact fold kernel + presence normalization at exit**: tiny, typed, algebraic,
with materialization delayed to the boundary. The twin path (`_sym` SUM vs full-range SUM +
COUNT) is a built-in semantic witness, not debt: it catches `_sym` degrading to an ordinary
sum, and a genuine `i64::MIN` sum being mistaken for emptiness.

## `cognitive-shader-driver` and the NNUE reference — OPEN
- The driver has no scheduler: one synchronous `run` per cycle
  (`crates/cognitive-shader-driver/src/driver.rs`), recomputing its pre-pass and cascade on
  every call.
- The NNUE lesson is the delta. stockfish-rs `HalfKaAccumulator::apply_move` updates 2
  columns on a quiet move vs 32 on a refresh, byte-exact by wrapping
  (`stockfish-rs/src/eval/incremental.rs`, tests
  `incremental_halfka_matches_full_refresh_every_ply` and
  `incremental_column_count_is_bounded`).
- Caveat, measured: stockfish-rs's own search does not use it yet — `evaluate()` calls
  `Accumulator::refresh` every time (`src/eval/network_eval.rs`), and `search.rs` calls that
  `evaluate`. The reference is proven in isolation, not wired.
- Direction: **delta-driven dispatch** — touch only rows whose inputs changed; the fold law
  applied along time. CubeCL contributes nothing here. No code change now.
