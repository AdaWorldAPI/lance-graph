# M-arm / O-arm build — Sonnet lane (edit-only, not compiled)

Branch: `claude/x265-x266-plans-review-h9osnl` (already checked out; not switched).
File touched: `crates/lance-graph-supervisor/examples/measure_wal_curve.rs` ONLY.
No `cargo` command was run at any point (guardrail §1 rule 7). No worktree
created. No git mutation performed. No `Cargo.toml` or `.github/` touched.

Spec followed: `.claude/plans/measure-64k-axes-v3.md` (M-arm + O-arm sections),
cross-read with `.claude/plans/measure-64k-axes-v1.md` (ground truth + A0
measured numbers) and `.claude/plans/measure-64k-axes-v2.md` (D1 `WriteOrderKey`
field list, D2 the ordered-chunk fast path). A-arm was NOT built (deferred per
the plan; explicitly out of scope for this brief).

## What was built

### CSV schema (shared, both arms)
- Added `morton_reorder_ns` as the **34th column, appended at the end** —
  existing 33 columns untouched, not renumbered. `Row` struct gained the field;
  `header()` and `to_csv()` updated; all **8 pre-existing** `Row{}` literals
  (B0, B1a, B1b, WAL-curve, Temporal, L1a, L1b, EXP-KIA) now set
  `morton_reorder_ns: 0` (mechanical, done via a scripted regex insertion after
  each site's `result_digest:` line, then hand-verified — 8/8 matched, no site
  missed or double-patched).

### §15 — M-arm (Morton reorder inserted before the seal)
- `WriteOrderKey { morton_chunk: u32, lane: u16, cycle_position: u64 }` per v2
  D1 — identity stays on `MailboxId` (the key is derived FROM an owner id,
  never stored back onto it).
- `morton_key_for(owner)`: splits the 16-bit owner id into two 8-bit
  coordinates, Morton-(Z-order-)interleaves them via the standard libmorton
  bit-spread trick (verified by hand on `0xFF -> 0x5555`), reads the top 6
  bits as `morton_chunk` (0..64) and bottom 10 as `lane` (0..1,024) — a
  **bijection** over the full 65,536-owner space (not a linear
  `chunk = owner/1024` split like L1a's).
- Two pipelines (`run_m_arm_pipeline(morton: bool, wal_path)`, `async fn`),
  `WARMUP_CYCLES=2 + MEASURED_CYCLES=16` real cycles each, identical cast
  content/order between the two configs:
  - cast (`emit_bootstrap_intent`, 512 B `NodeRow512` payload keyed on
    `(cycle, owner)` — deliberately NOT constant, so a trajectory digest
    actually depends on cycle order) → collect (`collect_casts`) →
    **[Morton only] `order_cycle_stably` sort by `morton_key_for`, then
    relabel `stream_position` to the sorted rank** (timed as
    `morton_reorder_ns`) → seal (the REAL `DetachedCycleBatch::freeze`, unmodified)
    → a REAL chunked `write_vectored` byte write of the frozen landings'
    512 B payloads (4,096-row/2 MiB groups, the house pattern from
    `run_wal_curve`'s W0-current path) + one `fsync` → commit into an
    in-process `MemWal` via the REAL `persist_cycle` (for the later T1 read).
  - Both configs assert `bytes_written == CANONICAL_FRAME_BYTES` every cycle
    (the A0 comparability assert, reused).
- **Digest identity (mandatory, an `assert_eq!`, never a print):**
  `semantic_digest` — `(owner, row, payload)` sorted BY OWNER (order-
  independent by construction) over the LAST measured cycle's frozen
  landings — natural vs Morton. This is a genuine cross-check because the two
  pipelines' `landings` orders differ (natural = stream_position-arrival
  order; Morton = the reorder's rank order) while the coalesced content must
  match.
- **T1 ordered-vs-unordered:** after both 16-cycle histories are sealed,
  `local_trajectories` (the real `temporal.rs` function, unmodified) is timed
  on both 1,048,576-row histories.
- **v2 D2 ordered-chunk fast path:** `local_trajectories_ordered_chunk_fastpath`
  — validates `stream_position` strictly increasing across the WHOLE scanned
  history (the collapse of "version × chunk × lane monotonic" this benchmark's
  relabeling scheme produces onto one counter), then **appends directly**
  (`BTreeMap::entry(...).or_default().push(...)`, **no `sort_by_key` anywhere in
  the function** — verified by inspection) instead of `local_trajectories`'s
  group-then-sort. Run against the Morton-ordered history; digest-compared
  against the generic path's output on the SAME history (`assert!`, mandatory).
  **Can-it-fire proof** (CLAUDE.md falsifiability rule): a hand-built 2-row
  `Vec<LandedSlot>` with a deliberately regressed `stream_position` is asserted
  to return `Err(..)` — the guard is proven non-decorative, not just asserted
  to exist.
- **SUM verdict** (`delta_total = reorder_cost − (seal+write+sync+T1 savings)`,
  pre-registered direction) printed with an explicit WINS/does-not-win line —
  never the gain alone.
- Two CSV rows: `m_arm_natural` / `m_arm_morton`, `morton_reorder_ns` populated
  only on the Morton row (never fabricated on the row that didn't pay it).

### §16 — O-arm (ordering source)
- `PreSealRow` — a `LocalCausalRow` view over an in-flight `SweepSlot`,
  letting O-B call `local_trajectories` on **cast-time** data before any
  seal/WAL exists.
- `derive_order_from_temporal_replay` — sandwiched between
  `// FIREWALL-START: derive_order_from_temporal_replay` /
  `// FIREWALL-END: ...` sentinel comments. Groups cast-time rows via
  `local_trajectories`, flattens by `BTreeMap` iteration order (owner-
  ascending) — the temporal-sourced physical order, independent of arrival
  order.
- **Firewall enforcement:** a compile-time self-scan in `run_o_arm`
  (`include_str!("measure_wal_curve.rs")`, matching `probe_ignition.rs`'s G2a
  pattern) slices the source between the two sentinel markers and asserts that
  **scoped region** contains neither a `scan_sealed` call nor a `sink.sealed`
  read (both needles built via `format!("{}_{}" / "{}.{}" , ...)`
  concatenation so the check string itself never appears contiguously in
  source), plus a can-stay-silent check that the region DOES contain
  `local_trajectories`. Scoped (not whole-file) deliberately — the file has
  legitimate `scan_sealed` calls elsewhere (`run_temporal`, `run_m_arm`) that
  would false-positive a whole-file negative scan.
- **Non-vacuity fixture:** `scrambled_cast_order()` — a bit-reversal
  permutation of the owner id, used as the CAST loop order for BOTH O-A and
  O-B. Documented reasoning: every other arm in this file casts owners
  0..65,535 in ascending order, which would make O-A's arrival-order and
  O-B's temporal-sourced (owner-ascending) order trivially coincide
  regardless of whether O-B's derivation does real work — the exact vacuous-
  assertion shape CLAUDE.md's falsifiability rule forbids. `assert_ne!` against
  the identity ordering proves the fixture is actually scrambled.
- O-A: cast → collect → seal (`DetachedCycleBatch::freeze` on the natural,
  arrival-ordered slots) → WAL (`MemWal::commit_cycle`) → [after 16 cycles]
  temporal replay (`local_trajectories`).
- O-B: cast (same scrambled order) → collect → **`derive_order_from_temporal_replay`
  first** (timed as `order_derive_ns`), relabel `stream_position` to the
  derived rank → seal → WAL → temporal replay.
- **PRIMARY OBSERVABLE, computed and printed BEFORE any timing** (both in
  source-code order and in the eprintln sequence): `trajectory_digest` —
  the RECOVERED trajectory (owner-ascending, each chain in `cast_seq` order,
  read back via `scan_sealed` + `local_trajectories`) for O-A vs O-B,
  `assert!`-checked equal.
- **Kill-condition disposition, reported honestly, not glossed:** CONSTRUCTIBLE.
  O-B's derivation uses `local_trajectories`'s `BTreeMap`-grouping (distinct
  code path from O-A's seal-side `order_cycle_stably` `Vec::sort_by_key`) — not
  literally the same code, so not a disguised O-A; both are `O(n log n)`-class
  under this harness's one-row-per-owner-per-cycle shape, which the report
  states explicitly rather than hiding.
- Two CSV rows: `o_a_today_pipeline` / `o_b_ordering_sourced_first`.

## Deviations from the spec, with reasons

1. **Both arms reuse ONE relabeling trick for `stream_position`** (Morton rank
   for M-arm, temporal-derived rank for O-B) rather than adding a new,
   separate physical-order field to `SweepSlot`/`DetachedCycleBatch`. Reason:
   `DetachedCycleBatch::freeze` (and `persist_cycle`, which calls it
   internally) always sorts landings by `stream_position` — there is no other
   hook to influence physical write order without either (a) modifying
   `persist_sink.rs` (out of scope — edit-only on this ONE file) or (b) locally
   reimplementing `freeze` (rejected — it would stop using the REAL seal
   function, weakening the fidelity of "same seal path as A0"). Relabeling
   `stream_position` to the desired rank lets the REAL, unmodified
   `DetachedCycleBatch::freeze`/`persist_cycle` do the ordering, at the cost of
   `stream_position` no longer literally meaning "arrival order" inside these
   two benchmark arms. This is flagged here explicitly since production code
   depends on `stream_position`'s cross-cycle-monotonic-per-owner contract for
   crash recovery (`persist_sink.rs`'s own doc comment) — verified the
   relabeling PRESERVES that contract (owner X's rank is fixed across cycles,
   so `position_base(N) + rank(X)` is strictly increasing in `N`), but this is
   a benchmark-local repurposing, not a new production pattern to imitate.
2. **O-arm bypasses `persist_cycle`'s validation** (`CycleMismatch` /
   `OwnerMismatch` checks) by calling `DetachedCycleBatch::freeze` +
   `MemWal::commit_cycle` directly instead of `persist_cycle`. M-arm, by
   contrast, DOES go through `persist_cycle` (after its own separate real-byte
   write). This is an asymmetry between the two arms' MemWal-commit paths —
   harmless here because the synthetic input is guaranteed self-consistent
   (single owner-cycle pairing, no cross-owner moves), but noted rather than
   silently normalized away.
3. **M-arm's real byte write uses 4,096-row (2 MiB) `write_vectored` chunks**,
   not the full 5-point `SEGMENT_TABLE` sweep A0's WAL-curve arm already
   covers. The plan's M-arm section does not re-ask for a segment-size sweep;
   segment size is already A0's own axis. Chose one representative chunk size
   (matching one of A0's five configs) so the M-arm's own axis (ordering)
   isn't confounded with the segment-size axis A0 already measured.
4. **`wal_syscalls` in the M-arm CSV rows is the REAL median syscall count**
   (self-caught during review — an earlier draft hardcoded `0` while
   discarding a real measured value; fixed before finishing, see self-check
   list below).
5. Did not build a THIRD binary or a second `#[test]` harness — everything
   lives in the ONE release binary per the plan's explicit "one release binary
   is a standing rule."

## Signature/type facts verified by reading source (not compiled)

- `MailboxId = u32` (`lance-graph-contract/src/collapse_gate.rs:121`).
- `SweepSlot`, `LandedSlot`, `CycleFrame`, `CycleId`, `DatasetVersion` are all
  `pub`-fielded tuple/record structs, cross-crate-constructible (needed for the
  fast path's synthetic negative-test fixture).
- `DetachedCycleBatch::freeze(frame, casts)` internally calls
  `order_cycle_stably(&mut casts, |s| s.stream_position)` — confirmed by
  reading `persist_sink.rs:262-278`, which is WHY the relabeling trick (point 1
  above) works and why a naive "reorder then freeze" without relabeling would
  have been silently undone.
- `persist_cycle` also calls `DetachedCycleBatch::freeze` internally
  (`persist_sink.rs:356`) — confirmed BEFORE designing the M-arm's dual-commit
  shape (real write via local `freeze()` call + separate `persist_cycle` call
  for the MemWal, rather than trying to reuse one `freeze()` result for both,
  which would have required bypassing `persist_cycle`).
- `order_cycle_stably<T, K: Ord>(rows: &mut [T], key: impl FnMut(&T) -> K)` is
  `pub` in `persist_sink.rs` — added to this file's existing `use
  lance_graph_planner::persist_sink::{...}` import list (not a new dependency,
  an existing exported fn from an already-depended-on module).
- `local_trajectories<R: LocalCausalRow + Clone>(global: &[R]) ->
  BTreeMap<MailboxId, Vec<R>>` — added `std::collections::BTreeMap` to this
  file's own `use std::collections::{...}` import (was `HashMap` only).
- `BenchRow` (already defined in this file at §11) derives `Clone` and impls
  `LocalCausalRow` — reused verbatim for both new arms rather than minting a
  parallel row type.

## Self-check performed (could not run a compiler, so these substitute)

- **Brace/paren/bracket balance**, string-literal- and line-comment-stripped,
  on JUST the new §15/§16 region: **68/68 braces, 384/384 parens, 13/13
  brackets** — perfectly balanced.
- Same check on the WHOLE file after the edit: braces and brackets balanced;
  parens off by exactly 1, but **the identical −1 offset already exists in the
  pre-edit file** (verified via `git show HEAD:...` and diffing the same
  cleaned count) — attributable to a pre-existing artifact (likely a char/byte
  literal my crude string-stripper doesn't special-case), not anything
  introduced here.
- Grepped every new identifier (`WriteOrderKey`, `morton_key_for`,
  `MArmPhaseMedians`, `run_m_arm`/`run_m_arm_pipeline`, `PreSealRow`,
  `derive_order_from_temporal_replay`, `OArmPhaseMedians`, `run_o_arm`/
  `run_o_arm_pipeline`, `scrambled_cast_order`, `trajectory_digest`,
  `local_trajectories_ordered_chunk_fastpath`, `semantic_digest`) for
  accidental collisions with existing names in the file — none found.
- Verified the `include_str!` firewall markers (`FIREWALL-START: ...` /
  `FIREWALL-END: ...`) appear FIRST as the literal comment sentinels (line
  ~2886/2912) and only SECOND as the scan's own string-literal copies (line
  ~3099/3100) — `.find()` returns the leftmost match, so the scoped region
  slices exactly the intended function body, not a self-referential mismatch.
- Verified all 12 `csv.write(&Row { ... })` literals (8 pre-existing + 4 new)
  supply `morton_reorder_ns`; grep count of the field name = 13 (1 struct decl
  + 12 literals) — matches exactly, no site missed.
- Re-derived by hand that `morton_spread_u8(0xFF) == 0x5555` per the standard
  libmorton bit-spread algebra (each of 8 set bits lands on an even position of
  the 16-bit result) — the one arithmetic claim in the new code worth hand-
  verifying since it can't be unit-tested here.
- Traced the async/`.await` capture shape carefully after **catching and fixing
  a real bug in my own first draft**: an earlier version of `run_m_arm_pipeline`
  built a FRESH `tokio::runtime` INSIDE the per-cycle loop and used
  `rt.block_on(async { persist_cycle(&sink, ...) })` — a non-`move` async block
  trying to move an owned `Vec<SweepSlot>` out through an implicit reference
  capture, which does not compile, PLUS constructing 18 throwaway runtimes.
  Fixed by making `run_m_arm_pipeline` (and its caller `run_m_arm`) `async fn`,
  matching the O-arm's already-correct shape, sharing the ONE
  `tokio::runtime` `run()` already builds for `run_temporal`/
  `run_exp_kia_a2_64k`.

## What could NOT be verified without a compiler

- Exact `rustc` type-inference outcomes at generic call sites (e.g.
  `order_cycle_stably(&mut slots, |s| morton_key_for(s.owner))`'s closure
  return type, `collect::<Vec<_>>()` targets) — checked by hand against the
  real signatures read from source, but not compiler-confirmed.
- `clippy -D warnings` cleanliness (e.g. whether the new `Row` literals trip
  `clippy::too_many_arguments`-adjacent lints — the file already carries a
  module-level `#![allow(...)]` covering the cast lints used throughout, which
  the new code also relies on, but a fresh clippy pass has not run).
- Actual measured numbers, obviously — this lane produced NO run output; the
  orchestrator's central `cargo build --release` + one release run is the next
  gate.
- Whether the 512 B-per-row real-write path for 16 cycles × 2 pipelines
  (≈1 GiB total transient WAL scratch, reclaimed per-pipeline via
  `fs::remove_dir_all` at the end of `run_m_arm`, matching the house
  discipline that caused ENOSPC before) fits comfortably in this host's ~90%-full
  `/tmp` — sized deliberately smaller than A0's own 5.8 GiB peak, but not
  measured here.

## Files touched

- `crates/lance-graph-supervisor/examples/measure_wal_curve.rs` (only file
  edited; header docstring updated to point at this report; CSV schema +
  8 pre-existing `Row{}` literals updated; §15 M-arm + §16 O-arm added; `run()`
  wired to call both under the existing shared `tokio` runtime).
- `.claude/board/exec-runs/m-arm-o-arm-build.md` (this file — my own tag-file,
  per the one-writer board-hygiene rule; `AGENT_LOG.md` itself was read, not
  written).
