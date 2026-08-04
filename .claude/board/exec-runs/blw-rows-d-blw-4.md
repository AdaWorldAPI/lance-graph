# D-BLW-4 — `examples/blw_rows.rs` (Opus filigree agent, 2026-08-04)

**Branch:** `claude/x265-x266-plans-review-h9osnl`. **Status: EDIT-ONLY —
NOT COMPILED, NOT LINTED, NOT RUN, NOT MEASURED.** No cargo command of any
kind was issued. Nothing below is a measurement. Every "asserts", "gates",
"detects" refers to *code that was written*, never to an observed run.

## Files touched

| path | change |
|---|---|
| `crates/lance-graph-planner/examples/blw_rows.rs` | **new**, ~1,200 lines |

Nothing else. No `Cargo.toml` edit was needed (`cognitive-shader-driver` is
already a dev-dep from D-BLW-1; examples are auto-discovered).

## The axis, held

ONE `MailboxSoA<2048>`, constructed once, never a second owner at any row
count. The scaled unit is **rows inside that one tenant**. Owner count never
appears as a variable. No fabricated tenants; no new SoA; no lens-owned
mailbox; no stance node type; no owner/mailbox/tenant field on the DTO.

## Preflight grep (the §7 gate) — **23**, not 0

`batch_writer|BatchWriter|KanbanStep|KanbanMove|kanban|owner_adapter|MailboxSoA|SoaEnvelope`
→ **23** matching lines.

## Substrate symbols consumed, with `file:line`

| surface | definition |
|---|---|
| `MailboxSoA<N>` — the tenant | `crates/cognitive-shader-driver/src/mailbox_soa.rs:58` |
| `MailboxSoA::new` (asserts `w_slot < 64`) | `mailbox_soa.rs:292` |
| `MailboxSoA::write_row` — the ONE cycle-aware mutator | `mailbox_soa.rs:417` |
| `MailboxSoA::{populated,set_populated,cycle,tick}` | `mailbox_soa.rs:486`, `:495`, `:558`, `:399` |
| `MailboxSoA::{content_row,topic_row,angle_row}` | `mailbox_soa.rs:680`, `:697`, `:714` |
| `impl MailboxSoaView for MailboxSoA<N>` | `mailbox_soa.rs:852` |
| `identity_plane_at` override (real planes, `populated`-guarded) | `mailbox_soa.rs:886` |
| `MailboxSoaView::{energy,edges_raw,meta_raw,entity_type,n_rows,phase}` | `crates/lance-graph-contract/src/soa_view.rs:82`,`:85`,`:87`,`:89`,`:71`,`:77` |
| `WriteCell` / `WriteOutcome` | `mailbox_soa.rs:262`, `:241` |
| `owner_adapter::emit_bootstrap_intent` → `rebind_bootstrap` | `crates/lance-graph-planner/src/owner_adapter.rs:92`, `:68` |
| `BatchWriter::{cast,on_behalf_of,intent_moves,drain_pending_payloads}` | `crates/lance-graph-planner/src/batch_writer.rs:104`, `:128`, `:120`, `:151` |
| `StrategyOutcome` (the D-MBX-A6 carrier) | `crates/lance-graph-planner/src/traits.rs:182` |
| `KanbanMove` / `KanbanColumn` / `ExecTarget` | `contract/src/kanban.rs` |

## Pre-registered thresholds (constants, declared above the measurement code)

| gate | constant | value | decides |
|---|---|---|---|
| G-A | `BODY_FLOOR_US` | 100.0 | per-row body cost floor (W2's "≥100 µs bodies") |
| G-A | `MIN_SEQ_WALL_MS` | 50.0 | sweep long enough to beat scheduler noise |
| G-A | `MIN_THREADS_TO_EVALUATE` | 4 | a 2× gate must be reachable at all |
| G-B | `THROUGHPUT_FLATNESS` | 0.25 | sequential rows/s flat across row counts ⇒ rows/s is a meaningful unit |
| G-C | `SPEEDUP_GATE` | 2.0 | **the claim** — inherited from W2 verbatim, only the unit re-pinned |
| protocol | `RUNS` / `WARMUPS` | 5 / 1 | median of ≥5 after one discarded warm-up |
| instrument | `VERDICT_DISTINCT_FRACTION` | 0.5 | the verdict vector can detect anything at all |

Also pinned: `ROW_COUNTS = [256, 1024, 2000]`, `DEFAULT_BODY_REPS = 48`,
`MAX_THREADS = 8`, `SPARSENESS_FLOOR_ROWS = 512`.

**Can the design even test G-C?** Only if G-A is met, and G-A is *measured,
not assumed*: the harness computes the real per-row body cost and the real
sequential wall and prints `MET` / `NOT MET` with the failing reasons. When
G-A is not met, **G-C prints `NOT EVALUATED (INCONCLUSIVE)`** rather than a
number. That is the honest half — `DEFAULT_BODY_REPS = 48` was pinned to land
near the 100 µs floor in an *unoptimised* build; in a release build the body
will very likely fall below the floor and the run will correctly refuse to
claim a speedup.

**Assert vs report split:** correctness falsifiers `panic!`; measurement gates
print. A missed threshold regrades the claim (§12.3a′ kill condition) and does
not fail the build.

## The falsifiers, as written (NOT as measured)

- **PROBE-VERDICT** — distinct verdict values ≥ 50 % of rows, asserted *before*
  any equality check uses the vector as evidence. A near-constant vector cannot
  detect a reordering, so every downstream equality would be vacuous.
- **PROBE-DETECT** — three halves on the real (non-trivial) vector:
  *can-stay-silent* (an identical copy compares equal), *lost update* (one slot
  zeroed → located exactly, with a prior `assert_ne!(verdict, 0)` so the probe
  itself is non-vacuous), *reordering* (`rotate_left(1)` → located, with a prior
  assert that adjacent verdicts differ so a rotation is observable).
- **PROBE-IRON** — the iron rule over the CONCURRENT path: full 12 MiB byte
  image before/after; `IMAGE_LEN` asserted so a dropped column cannot pass as
  "byte-identical"; a scale-free non-zero-coverage gate (`nonzero > seated*8`)
  so an all-zero image cannot make it trivially true. Plus concurrent ==
  sequential element-by-element, plus two concurrent runs bit-identical.
- **PROBE-IRON+** — the same image compared again after the ENTIRE timed
  workload (the row grid + the thread sweep, tens of thousands of bodies on
  1..T threads), so "evaluation mutates nothing" covers what was actually
  timed, not a separate demonstration sweep.
- **Determinism per run** — `measure` compares the concurrent vector against
  the sequential vector on **every** run including warm-ups, panicking with the
  offending row index; it is not checked once at the end.
- **Gated write-back, both halves** — the dirty set is data-dependent
  (`verdict % 64 == 0`), asserted non-empty AND a sparse minority above the
  stated `SPARSENESS_FLOOR_ROWS` (the floor is stated rather than tuned);
  row-image comparison asserts `differs == is_dirty` for every capacity row,
  plus `changed == dirty.len()` and `changed > 0`.
- **PROBE-MUT-a/b** — the byte comparator's can-bark twins, ported from
  D-BLW-1: a gated one-column write (must locate "fixed columns") and a
  **one-bit** write into the all-zero ANGLE plane (must locate "ANGLE plane").

## Which half is parallelised

**READ only.** `sweep_concurrent` takes `&V: MailboxSoaView + Sync` and hands
each scoped thread the SAME shared `&V` plus a disjoint `&mut [u64]` output
chunk. No lock, no atomic, no `unsafe`. The `V: Sync` bound, instantiated at
`MailboxSoA<2048>`, is the compile-time soundness proof — if the SoA ever
gains interior mutability this example stops compiling.

**WRITE is NOT parallelised** and no speedup is claimed for it: `write_row`
takes `&mut self` and is the single mutator by construction.

## The seam I stopped at

**No seal, no applied lifecycle step.** The harness casts the write intent
(`emit_bootstrap_intent`, ahead of the write, as designed), asserts the
write-on-behalf pairing and the rebind anti-vacuity, drains the payload and
asserts the descriptor round-trips unchanged — then **stops**. It never calls
`persist_cycle` / `recover_and_apply` / `try_advance_phase`: *no successful
write ⇒ no applied step* (`owner_adapter.rs` module doc), and D-BLW-1 already
proved that loop. Re-running it here would add wall time and no new evidence.
The dangling cast is deliberate and is stated in the output.

## What I did NOT do

- Did not run cargo in any form.
- Did not run `rustfmt` either — "fmt" is named in the prohibition, and the
  formatting is therefore UNVERIFIED (see below).
- Did not edit `blw_tenant.rs`, `blw_binding.rs`, or any `src/` file.
- Did not touch `persist_sink.rs`, `temporal.rs`, `batch_writer.rs`,
  `owner_adapter.rs`, `soa_view.rs`, `mailbox_soa.rs`, or `crates/jc` (§12.5).
- Did not build a stance instrument (§12.3c / §12.7 killed it).
- Did not resolve `ISS-MAILBOXSOA-ROW-COST-VS-512B-CANON` — every byte figure
  printed is labelled a figure of `MailboxSoA<2048>`, never of the 512 B canon
  `NodeRow`, and the two are never averaged.
- Did not write `AGENT_LOG.md` or any shared board file.

## Open questions I could not close without a compiler

1. **`Tenant: Sync`.** Reasoned from the field list (`mailbox_soa.rs:58-230`):
   plain `Copy` arrays + `Box<[u64]>` + scalars, no `Cell`/`RefCell`/`Rc`, so
   auto-`Sync` should hold. NOT verified. The `V: Sync` bound makes a wrong
   answer a compile error rather than a silent one.
2. **`thread::scope` lifetimes.** The chunk borrows are taken *before* entering
   the scope closure on purpose (a reborrow created inside the closure does not
   satisfy the `'scope` bound `spawn` requires). Believed correct; unverified.
3. **Formatting.** `cargo fmt --check` is the likeliest gate to fail. I
   hand-matched rustfmt's `max_width=100` / `fn_call_width=60` /
   `chain_width=60` heuristics against shapes that are known-green in
   `blw_tenant.rs`, and rewrote three constructs (a filter chain → an explicit
   loop, a `map_or` closure block → a `match`, a 4-line chain → one line) to
   remove guesswork. Residual risk is real; the orchestrator's `cargo fmt` will
   settle it.
4. **Clippy under `-D warnings`.** Deliberate avoidances: no unused generic
   type parameter (the `assert_shareable::<T>()` helper was removed —
   `clippy::extra_unused_type_parameters` is warn-by-default — and replaced by
   the `V: Sync` bound, which proves the same thing); `div_ceil` instead of
   manual ceiling; no slice indexed by a loop variable; every
   `RowSpanDescriptor` field explicitly read so `dead_code` cannot fire on a
   field only touched by a derive. Other default lints unverified.
5. **Wall time.** UNVERIFIED and the main runtime risk. Expected order: ~75k
   row bodies across the grid + thread sweep, plus 7 snapshots of ~12 MiB each.
   `blw_tenant` did 12 snapshots in ~3 s, so the snapshots are ~2 s; the bodies
   dominate. If the run is too slow, the knobs in order are the corpus bound
   (`-- 512`), then `BLW_BODY_REPS` — but note that lowering reps will push the
   body under `BODY_FLOOR_US` and correctly turn G-C `INCONCLUSIVE`.
6. **Whether G-C can pass at all in a debug build.** Unknown. A debug build
   makes the body expensive (good for G-A) but also makes thread overhead
   relatively small (good for G-C), so a pass is plausible — but a KILL or an
   INCONCLUSIVE is an equally valid, reportable outcome and the harness is
   written to say so rather than to produce a number.
