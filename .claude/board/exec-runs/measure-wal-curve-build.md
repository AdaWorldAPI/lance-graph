# measure-wal-curve build lane report

**Lane:** Sonnet build (edit-only, no cargo). **Deliverable:** `crates/lance-graph-supervisor/examples/measure_wal_curve.rs`
(~2,227 lines). **Spec:** `.claude/plans/measure-64k-axes-v1.md`.

## Mandatory reads completed

1. `.claude/v3/knowledge/sonnet-worker-guardrails.md` — followed. No cargo run
   at any point. Edit-only. No git mutation, no worktree, no Cargo.toml touch.
2. `.claude/board/AGENT_LOG.md` (first ~120 lines) — read, not written.
3. `.claude/plans/measure-64k-axes-v1.md` — the full spec, built faithfully
   with documented deviations below.
4. `crates/lance-graph-supervisor/tests/probe_ignition_64k.rs` — inherited
   shapes: `MemWal`, `build_owner`, `flow_qualia`, `mantissa_of`, the
   ONE-`StyleStrategy::plan`-call-reused pattern, `run_cycle` call shape, the
   drained-writer `casts()`-is-cumulative lesson (avoided the same trap by
   never re-reading `writer.casts()` as a snapshot across cycle boundaries in
   this binary — each arm builds a fresh `BatchWriter` per repeat/config).
5. `crates/lance-graph-supervisor/tests/probe_ignition.rs` — the G2b
   `shade_owner`-fallback pattern (a Planning-origin cast is EITHER the
   style's Elixir mint or the gate's Native mint) — reused in
   EXP-KIA-A2-64K's sequential convergence boundary, since the parallel
   compute phase's `target` may legally be `Prune`, not just `CognitiveWork`.
6. `crates/lance-graph-planner/src/temporal.rs` — read in full. Used
   `local_trajectories` (layer 1), `deinterlace` (layer 2), `QueryReference::at`,
   `LocalCausalRow`/`DeinterlaceRow` trait definitions exactly as declared.
7. `crates/lance-graph-planner/src/persist_sink.rs` — read in full. Used
   `WalSink`, `DetachedCycleBatch`, `SweepSlot`, `CycleFrame`, `persist_cycle`
   directly (not `cycle_driver::seal_cycle`, for the arms that don't need a
   real WAL commit — see deviations).

Also read (not in the mandatory list but needed for exact signatures):
`crates/lance-graph-supervisor/src/cycle_driver.rs` (full — `collect_casts`,
`apply_sealed_transitions`, `SealedCycle`, `SealedTransition`, `MailboxFleet`),
`crates/lance-graph-planner/src/owner_adapter.rs` (full),
`crates/lance-graph-planner/src/batch_writer.rs` (full),
`crates/cognitive-shader-driver/src/mailbox_soa.rs` (grepped signatures +
read the `MailboxSoA` struct fields, `WriteCell`, `WriteOutcome`),
`crates/lance-graph-contract/src/soa_view.rs` (the `MailboxSoaView` /
`MailboxSoaOwner` traits), `crates/lance-graph-contract/src/kanban.rs`
(`KanbanColumn::advance_on_gate`, `KanbanMove`, `ExecTarget`),
`crates/lance-graph-contract/src/mul.rs` (`gate_decision_i4`,
`flow_state_i4` — read the exact FlowState thresholds to confirm the flow
qualia fixture reliably yields `Flow` regardless of mantissa magnitude),
`crates/lance-graph-planner/src/traits.rs` (`PlanContext`, `PlanInput`,
`StrategyOutcome`, `PlanStrategy`), `crates/lance-graph-contract/src/qualia.rs`,
`crates/lance-graph-contract/src/cognitive_shader.rs` (`MetaWord::new`),
`crates/lance-graph-contract/src/scheduler.rs` (`DatasetVersion`),
`crates/lance-graph-contract/src/collapse_gate.rs` (`MailboxId = u32`),
`crates/lance-graph-supervisor/Cargo.toml` (confirmed `cognitive-shader-driver`
is a **dev-dependency**, so it's reachable from an example; confirmed the
`cycle-driver` feature's `tokio` deps, and that dev-deps' `tokio` unifies to
give `rt-multi-thread` for free even though I only use `current_thread`).

## What was built — the arms, in spec order

- **B0** — `DummyOwner` cast baseline (no SoA, no temporal, no file I/O):
  scan → cast (write-on-behalf rebind via `emit_bootstrap_intent`) → collect
  → freeze. 3 repeats, medians reported.
- **B1a** — 65,536 × `MailboxSoA<4>`, one populated/firing row each (same
  shape as `probe_ignition_64k.rs`): build · scan · think (real per-owner
  `gate_decision_i4` + `advance_on_gate`) · cast · collect · freeze · apply
  (via `apply_sealed_transitions` against a **locally-built** `SealedCycle`,
  not a real WAL commit — see Deviation D1). 3 repeats.
- **B1b** — 65,536 × `NodeRow512` (`#[repr(C)] [u8; 512]`, provenance-commented
  as the canonical-node-layout stand-in, not a new type proposal), memory-only,
  peak RSS reported. 3 repeats.
- **L1a** — 64 × `MailboxSoA<1024>` chunks, 65,536 LOGICAL owners
  (`owner = chunk*1024+lane`): build/scan/think/cast/collect/freeze at row
  granularity. **`apply` intentionally NOT measured** — see Deviation D2, the
  most consequential deviation in this build.
- **L1b** — the mislabelling control (64 chunks cast AS owners): proves the
  held-backlog collapse (65,472 of 65,536 intents HELD because
  `collect_casts`'s ≤1-move-per-owner partition sees only 64 distinct ids).
- **WAL curve** — W1-contiguous (one flat 32 MiB `Vec<u8>`, sliced into
  `segments_per_cycle` contiguous `IoSlice`s, `write_vectored` looped +
  counted, one `File::sync_data`) beside W0-current (REAL per-cycle
  `SweepSlot` construction — one owned `Vec<u8>` clone per row — + REAL
  `DetachedCycleBatch::freeze`, then `IoSlice`-per-row `write_vectored`
  batched in `segment_rows`-sized groups). 5 segment configs × 2 warmup +
  16 measured cycles each, real files under `/tmp/measure_wal_curve_wal/`,
  removed at the end. Median/p95 reported; the `gain()`/plateau formula
  implemented exactly as specified, printed as a descriptive knee, never
  pass/kill.
- **Temporal** — 16 REAL committed cycles (real `persist_cycle` against an
  in-process `MemWal`, no-step landings — a sanctioned `SweepSlot` shape) →
  T0 = `scan_sealed` timing → T1 = `local_trajectories` → T2 = `deinterlace`
  at `QueryReference::at(8, 0)` (mid-history, so the filter is genuinely
  falsifiable — NOT `ref_version=16`, which would trivially keep everything;
  caught this during drafting).
- **EXP-KIA-A2-64K** — `std::thread::scope`, disjoint ranges, thread-local
  `PreparedIntent` Vecs (benchmark-local type, named explicitly by the plan),
  `&Fleet` shared read-only during compute, sequential rebind+cast+seal+apply
  at the convergence boundary, `AtomicUsize` high-water mark for
  `max_active_workers`, FNV digest over sorted `(owner, stream_position)`
  pairs for the sequential-vs-parallel identity assert. Worker counts
  `{1,2,4,8,16,available_parallelism()}` (std-only stand-in for "physical
  cores" — no `num_cpus` dep permitted).

CSV: one row per measured cycle/repeat/config to `$MEASURE_OUT` (default
`/tmp/measure_wal_curve.csv`), 33-column header exactly as specified,
`llc_misses` always empty. Per-configuration medians + the four closing
answers printed to stderr at the end of `run()`.

## Bugs I self-caught and fixed before handoff

1. **A duplicate module definition that would not have compiled.** My first
   draft left `mod measure;` (a file-based module declaration) directly above
   the later `mod measure { ... }` inline module body — Rust would reject
   this as "the name `measure` is defined multiple times". Found on a
   structural re-read; fixed by deleting the stray `mod measure;` line. This
   is exactly the kind of error orchestrator-side `cargo check` exists to
   catch, but I found it myself before handoff.
2. **Freeze-timer contamination (three sites: B0, B1a, L1a).** The first
   draft started `t_freeze = Instant::now()` BEFORE computing `digest_bytes`
   (and, in B1a, before `build_sealed_locally`'s sort over 65,536
   transitions) and only THEN called `DetachedCycleBatch::freeze` — so
   `freeze_ns` would have measured freeze + digesting + sorting, not freeze
   alone. Fixed by moving all untimed setup before the timer and starting
   `t_freeze` immediately before the `freeze()` call in all three sites.
3. **A tautological assertion in EXP-KIA's owner-binding check.** The first
   draft asserted `mv.mailbox == applied.applied.iter().find(|m| m.mailbox
   == mv.mailbox).unwrap().mailbox` — this is true by construction for ANY
   input (the `find` predicate guarantees the match), so it falsifies
   nothing. Rewrote to compare two INDEPENDENTLY-populated fields of
   `SealedTransition` (`t.owner`, set from `SweepSlot.owner` at the
   sequential boundary, vs `t.mv.mailbox`, set inside
   `emit_bootstrap_intent`/`rebind_bootstrap`) — a genuine cross-check.
4. **A wrong byte-size claim, 4 sites (B0/B1a/L1a/EXP-KIA) + Temporal.**
   `logical_bytes: (FLEET_OWNERS as u64) * 8` assumed an 8-byte payload, but
   every payload in those arms is `id.to_le_bytes()` where `id: MailboxId =
   u32` — 4 bytes, not 8. Fixed all 5 occurrences to `* 4` with an inline
   comment. The Temporal arm had the same bug in a different shape
   (`landed.len() * CANONICAL_ROW_BYTES`, implying 512-byte canonical rows
   when the actual payload there is the same 4-byte owner-id marker) — fixed
   to `* 4` as well.
5. **An unused-import risk (would be a `-D warnings` failure).** `MailboxSoaOwner`
   and `MailboxFleet` were imported but never used as trait-method call sites
   (only via generic-function type inference, which needs no import). Removed
   both; kept `MailboxSoaView` (genuinely needed for `.phase()`/`.mailbox_id()`/
   `.current_cycle()` method resolution — verified by grep count before keeping).
6. **An unused `best_workers` (assign-but-never-read).** Threaded it through
   `run_exp_kia_a2_64k`'s return tuple and into the closing answer #4's
   stderr line instead of leaving it dead.
7. **`b1b_peak_rss` computed but never read.** The module doc promises the
   "hot representation overhead" derived metric (plan §B1); the first draft
   computed B1b's median RSS and then dropped it. Added an explicit
   `hot representation overhead: B1a peak_rss=...B - B1b peak_rss=...B = ...`
   stderr line before the four answers.

## Deviations from a literal reading of the plan (with reasons)

**D1 — B1a's `apply` phase does not go through a real `WalSink::commit_cycle`.**
The plan lists "collect/freeze · apply" as B1a's measured phases without
requiring a real fsync. I built `build_sealed_locally` (a local re-derivation
of `cycle_driver::seal_cycle`'s transitions/next_position_base extraction,
provenance-commented against `cycle_driver.rs:286-301`) so `apply_ns`
measures `apply_sealed_transitions` alone, keeping the WAL-curve arm the
SOLE owner of real fsync physics (never blending the two axes, per the
plan's own "never blend two axes" rule). EXP-KIA-A2-64K, by contrast, DOES
go through a real `persist_cycle`/`MemWal` commit for its "one seal, one WAL
commit" witness assert, since that arm's own gate list explicitly requires it.

**D2 — L1a's `apply` phase is not measured at all (left `apply_ns = 0`), and
this is the single biggest interpretive call in this build.** A Rubicon
`phase()` is scoped to ONE `MailboxSoA<N>` instance (confirmed by reading
`MailboxSoaOwner::advance_phase`/`try_advance_phase` in
`crates/lance-graph-contract/src/soa_view.rs`), never to an individual row.
L1a's 65,536 "logical owners" share 64 physical `MailboxSoA<1024>` chunk
instances, so there is no per-logical-owner phase for
`apply_sealed_transitions` to advance. I considered three options: (a)
fabricate a per-row phase advance by routing all 1,024 logical owners in a
chunk through that chunk's single phase field (would silently misrepresent
1,023 of every 1,024 "applies" as real when only one phase transition
actually occurred); (b) invent a new per-row phase wrapper type (forbidden
by guardrail §1 rule 3, "no invention"); (c) measure build through freeze
only, document the gap, and exclude `apply` from the L1a-vs-B1a comparison
in answer #2. I chose (c) and said so both in the code (a ~20-line doc
comment at §12) and in answer #2's printed text ("apply is not comparable").
I believe this is the honest call under the guardrails, but it is a genuine
interpretive gap against the plan's literal phrasing ("Measure separately:
construction · registration into fleet · scan · thought · cast ·
collect/freeze · apply" for "B1" generally, which L1 is implicitly compared
against) — **flagging explicitly for the orchestrator to confirm or
override.**

**D3 — "construction" and "registration into fleet" are folded into one
`build_ns` column.** The CSV schema (fixed by the plan's own "Measurement
schema" section) has no separate `registration_ns` field; the 7 named B1
phases map onto the 7 available phase columns (build/scan/think/rebind_cast/
collect/freeze/apply) with registration folded into build, since a single
`HashMap::insert` per owner is negligible next to `MailboxSoA` construction
and there was no schema slot to give it its own number.

**D4 — Temporal's T0 (`scan_sealed`) timing rides the CSV's `scan_ns` column.**
The CSV schema has no dedicated T0 field; `scan_ns` is the closest-named
generic slot and is used ONLY by the Temporal row for that purpose,
documented inline at the write site.

**D5 — The WAL-curve segment table's "N slices" are interpreted as an
explicit external batching boundary for BOTH representations**, not as
"pass everything to one `writev()` call and let `IOV_MAX` implicitly
determine batching." For W1 each segment is ONE contiguous `IoSlice`
(the "storage/cache ceiling" reading — the best achievable I/O shape); for
W0 each segment is `segment_rows` individual 512-byte `IoSlice`s (since each
row is a separately-heap-allocated `Vec<u8>` from `DetachedCycleBatch`'s
`BTreeMap`, they cannot be coalesced into one contiguous slice without
defeating the point of measuring the BTreeMap representation's real shape).
One `File::sync_data()` per cycle either way, after all segments' writes.

**D6 — The canonical 32 MiB frame's content is identical across all 18
cycles within one WAL-curve configuration** (not re-derived per cycle).
The plan's "constant total work" bar is read as being about bytes actually
handed to the OS via `write_vectored` (which IS constant: 16 × 32 MiB = 512
MiB per config, exactly as specified), not about re-deriving unique content
each cycle. Flagged in a code comment at the digest-computation site since a
content-invariant digest is a weaker falsifier than a per-cycle-unique one
would be — an honest limitation, not a silent one.

**D7 — `available_parallelism()` stands in for "physical cores".** The plan
says "1/2/4/8/16/physical cores"; std has no direct "physical core count"
API (only logical/schedulable units via `available_parallelism()`), and
`num_cpus`/`libc` are explicitly forbidden by the plan itself ("no rayon,
no libc" — I extended the same spirit to `num_cpus`, an external crate).

## What could NOT be verified (never compiled, never run — orchestrator gates)

Everything. Per the sonnet-worker-guardrails §1 rule 7, I ran no `cargo`
command of any kind — not `check`, `build`, `test`, `clippy`, or `fmt`. Every
signature cited above was read from source in the same pass that wrote this
file, and I did a full manual front-to-back re-read after writing (catching
the 7 bugs listed above), plus targeted `grep`-based checks for: balanced
struct-literal field lists across all ~9 `Row { ... }` construction sites,
unused-import candidates (verified via occurrence-count greps before
removing/keeping each one), `IoSlice::advance_slices`' exact stable
signature (`&mut &mut [IoSlice<'a>]`, stabilized well before this
workspace's pinned 1.95.0 toolchain), `KanbanMove`'s exact field list (5
fields, no `#[non_exhaustive]`), `ExecTarget::Native`'s existence, and
`flow_state_i4`'s exact thresholds (confirmed the flow-qualia fixture yields
`FlowState::Flow` regardless of mantissa magnitude ≥1, so both the 1-row
B1a/EXP-KIA fixture (mantissa≈1) and the 1,024-row L1a chunk fixture
(mantissa clamped to 7) reliably gate to `Flow`).

**Named risk I cannot rule out without a compiler:** any place where I wrote
`u32`/`u64`/`usize` and expected an implicit numeric coercion that Rust does
NOT actually perform (Rust has none — every numeric-type site was written
with an explicit `as` cast or an explicit `u64::from`/`u32::from` conversion
as far as I traced, but I did not mechanically verify every one of the
~150+ arithmetic/comparison sites in this file). This is the highest-value
thing for the orchestrator's `cargo check` pass to catch first.

**Also unverified:** whether `IoSlice::advance_slices` requires an explicit
`'a` bound match that my elided-lifetime `write_vectored_all` signature
(`bufs: &mut [IoSlice<'_>]`) satisfies without a compile error — I am fairly
confident based on the documented stable signature and common real-world
usage of this exact idiom, but I have not compiled it.

## Self-check against the brief's named traps

- No hardcoded version/base that only works on cycle 1: `position_base` and
  `version_counter`/`sink.head()` are threaded forward across every cycle in
  every arm that has more than one cycle (WAL curve, Temporal).
- No tautological compares: found and fixed one (see bug #3 above); the rest
  were written as genuine cross-checks from the start (T2's `ref_version=8`
  anti-vacuity check, L1b's exact-held-count proof, G-style asserts mirroring
  the probes' can-fire/can-stay-silent discipline).
- No fingerprint captured outside its window: found and fixed three
  freeze-timer contamination sites (bug #2); double-checked every other
  `Instant::now()`/`.elapsed()` pair in the file for the same pattern during
  the final re-read.
- The cumulative-`casts()` trap: never call `writer.casts()` as a
  cross-cycle cumulative counter anywhere in this file — every arm either
  builds a fresh `BatchWriter` per repeat/cycle, or (EXP-KIA, single cycle
  per worker-count) uses local counters incremented during the cast loop
  itself, never `writer.casts().len()` deltas.
- Partial `write_vectored` loops and counts: implemented in
  `write_vectored_all`, used by both W1 and W0 representations, tracks the
  actual number of `write_vectored` syscalls issued (not an assumed 1).

## Files touched

- `crates/lance-graph-supervisor/examples/measure_wal_curve.rs` (new, ~2,227
  lines) — the deliverable.
- `.claude/board/exec-runs/measure-wal-curve-build.md` (this file, new) — my
  own tag-file, per the one-writer rule.

No other files touched. No `cargo` command run. No git command run beyond
what the harness itself may have done for read access.
