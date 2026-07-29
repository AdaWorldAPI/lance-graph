# lens-migration-engineer — run ZC2 (2026-07-29)

Branch `claude/x265-x266-plans-review-h9osnl`. Warden items 1, 2, 3 in order.
No commits, no pushes — the orchestrator commits centrally.

## Verdicts

### MIGRATED — `crates/lance-graph-contract/src/dispatch_guard.rs:130` (`guard`)

- BEFORE: `guard(ctx, window: &[(usize, CausalWitnessFacet)], focal_idx, id, passes)`;
  body destructured `window.get(focal_idx).map(|&(_, w)| w)` (a 12-byte copy out
  of a caller-gathered slab) and called the gathered `standing_wave_grounded`.
- AFTER: `guard(ctx, focal_pos, lens: &WitnessLens<'_>, visible: impl Fn(usize) -> bool, id, passes)`;
  `lens.at(focal_pos)` is a cast into the row's own value slab, the fallback is
  `&const ABSENT` (`'static`-promoted, no copy), and gate 2 runs
  `standing_wave_grounded_lens`. **The gathered signature no longer exists in
  the crate API.**
- EQUIVALENCE: `lens_guard_matches_the_gathered_oracle_across_ids_passes_and_visibility`
  — the pre-migration body kept verbatim as a `#[cfg(test)]` oracle, compared
  field-by-field over 4 window shapes × {full, each-one-peer-hidden} × ids 1..=34
  × passes {1,2,4,8}; an anti-vacuity assert requires all three of
  Fires/Escalate/Unbound to occur. Plus `out_of_range_focal_is_unbound_not_a_panic`
  (bounds), and the 4 original behaviour tests re-pointed at the lens.
- Call sites migrated: 4 in-module tests + `examples/dispatch_guard_redundancy.rs`.
  That example re-runs **34/34 Fires→Escalate, all four gates green** — bit-identical
  to the figure pinned in the module doc, so the migration is end-to-end confirmed
  against a published measurement, not only against unit fixtures.

### MIGRATED — `crates/deepnsm-v2/src/introspect.rs:117` (`confidence_delta_self`)

- BEFORE: `stream.window_at(v).into_iter().filter(..).count()` — two full-window
  `Vec<Spo>` allocations per call, and `TemporalStream::{window_at, window_range}`
  returned `Vec<Spo>` (a gathered projection of `entries`, the primary store).
- AFTER: both accessors return `impl Iterator<Item = &Spo> + '_` — a borrowing
  projection of the store; the call site is `stream.window_at(v).filter(..).count()`.
  **No Vec-returning window accessor remains on `TemporalStream`.** The store
  (`entries: Vec<(u64, Spo)>`) is untouched, per the brief.
- EQUIVALENCE: `window_projection_matches_an_independent_recount` (lib.rs) —
  every reference version 0..=20 against a hand-written raw filter, plus the
  explicit-range arm, with an assert that the sweep covered the empty, partial
  AND all-visible windows; and
  `confidence_self_matches_recount_across_every_version_pair` (introspect.rs) —
  every (v1,v2) pair 0..=11 × 3 beliefs (present / absent / other) against the
  UNTOUCHED `confidence_delta_recount` oracle, with an anti-vacuity assert that
  the fixture moves the confidence.
- `confidence_delta_recount` and `most_frequent_belief` were NOT touched (the
  G-SRS4-2 independent oracle; converting them would destroy the check).
- Collateral call sites updated: 2 asserts in `lib.rs` tests, 1 whole-book
  window count in `examples/bible_wave.rs` (`.len()` → `.count()`; the KILL gate
  still fires on the same condition).

### PARTIAL — `crates/lance-graph-contract/src/witness_fabric.rs` (the `window:` family)

Twins built + proven; gathered forms retained (doc-marked) because callers
outside this crate still exist. Per the card this is step 1 of 3 — **not done**.

- ADDED: `elect_peers_lens`, `quorum_mantissa_lens`, `trajectory_of_lens`,
  `standing_wave_stratified_lens`, `standing_wave_diagnosed_lens`. Every one
  takes `(focal_pos, &WitnessLens, visible: impl Fn(usize) -> bool, ..)`, reads
  registers via `WitnessLens::at`, filters by PREDICATE, bounds-checks, and
  returns the neutral value for an unaddressable focal.
- ADDED: `WitnessLens::write_register(&mut NodeRow, &CausalWitnessFacet)` — the
  PRODUCER side, so no caller outside the module needs the raw offsets. This is
  what let the example and the `dispatch_guard` tests build row sources without
  exporting `WITNESS_REGISTER_START` (an exported literal offset is the drift
  bug the `const _` asserts exist to prevent). Offsets stay private and derived.
- Each of the 5 gathered originals now carries the "materializes a SECOND
  projection … prefer the lens twin" doc mark, matching `resolve_chain`'s.
- EQUIVALENCE:
  - `lens_peer_fabric_matches_gathered_across_visibility` — a 5-row fabric with
    real agreement, a genuine Kausal dissent, an all-unbound peer and a TIE at
    the maximum (so the first-maximum tie-break is exercised); compared over the
    full window and every one-peer-hidden subset, plus all-invisible and
    out-of-range focal. Asserts that hiding a peer actually changes an election
    (otherwise the predicate is untested).
  - `peer_fabric_is_non_trivial` — pins agreement 2, quorum offset +2 (not the
    tying +3), contradiction offset +1, mantissa > 0.
  - `lens_stratified_diagnosed_trajectory_match_gathered` — 4 chain scenarios ×
    passes {1,2,3,5,8}, comparing verdict AND settle pass AND `EscalateReason`;
    asserts all three verdicts occur.
  - `lens_visibility_changes_stratified_and_diagnosed_like_a_shrunk_window` —
    hiding the hop target must change the diagnosis exactly as dropping it from
    the gathered window does, reason field included.
  - `write_register_round_trips_through_the_lens` — write/read agree, and the
    0xEE canary outside the register survives (the write touches only its lane).
- Documented semantic scoping (honest, not silent): the lens peer domain is
  `{ pos ∈ 0..lens.len() | pos != focal_pos && visible(pos) }` visited ascending;
  `elect_peers` keeps the FIRST maximum, so the two forms agree exactly when the
  gathered window is position-ascending — which is how every producer builds one.
  A hand-built descending/duplicated window is outside the equivalence and says so.

### BLOCKED — `crates/lance-graph-planner/src/traits.rs:91` (`WitnessWindow::rows`)

- BLOCKER: `WitnessWindow { rows: Vec<(usize, CausalWitnessFacet)>, .. }` is a
  STORED gathered window, held as a field of the shipped `PlanContext`. Making
  it a lens requires a lifetime parameter on `WitnessWindow` and therefore on
  `PlanContext` — a public-shape redesign, i.e. a deliberate API decision, not a
  cleanup pass. Reported, not landed. (`style_strategy.rs:901`'s test helper
  hangs off the same type.)

### NOT-A-VIOLATION (cleared, untouched)

- `introspect.rs:135` `confidence_delta_recount`, `:159` `most_frequent_belief`
  — the G-SRS4-2 INDEPENDENT oracle; their falsifying value IS being a second
  path over a caller-held slice.
- `deepnsm-v2/src/wave.rs` `WitnessStream` — already adjudicated an owned
  parallel container in commit 2f0f62d and documented in the module header;
  out of this run's scope.

## Still outstanding

**14 gathered `window:` parameters remain**, all now unblocked by the twins
except the stored one:

| where | count | note |
|---|---|---|
| `contract/witness_fabric.rs` gathered originals | 7 | retained: twin exists for all 7; delete after callers migrate |
| `planner/nars/meta_basin.rs` | 7 (`grade_rows`, `stable_under_perturbation`, `stability_sweep`, `stability_around`, `outlier_suggestions`, `coarse_flags`, `ranked_outlier_suggestions`) | migratable NOW — they call `quorum_mantissa` / `trajectory_of`, whose twins landed this run |
| `planner/traits.rs` `WitnessWindow.rows` | 1 (stored) | **BLOCKED** — needs the `PlanContext` lifetime decision |
| `contract/examples/probe_dcsw2_basin_rung.rs` | 1 call site | builds a gathered window for `standing_wave_grounded` |
| `deepnsm-v2/src/wave.rs` `WitnessStream::{window_at, window_range, ground_at, resolve_at}` | 4 | owned parallel container, separately adjudicated |

## Inventory — the `revisions:` version-axis family (NOT started, per the fence)

All **8** live in `crates/lance-graph-contract/src/witness_fabric.rs`; all take
`revisions: &[CausalWitnessFacet]` (an owned facet series, oldest→newest):

| line | fn |
|---|---|
| 1215 | `opinion_strength` |
| 1225 | `is_opinion` |
| 1316 | `revision_trajectory` |
| 1441 | `belief_runs` |
| 1476 | `superseded_runs` |
| 1499 | `superseded_spread_sample` |
| 1592 | `suggest_reopening` |
| 1678 | `foresight_sample` |

**No external callers** — every call site is inside `witness_fabric.rs` or its
tests, so this sub-wave is self-contained in one file.

Scoping note for whoever takes it: this axis is NOT the same shape as the
`window:` family. A window indexes ROW POSITIONS in one row array, so
`WitnessLens<'a>{ rows }` + `at(pos)` covers it. A revision series is the SAME
logical row at successive Lance VERSIONS — the source is a version-range read
(`QueryReference::at(v, rung)` / `deinterlace`), not a contiguous row slice.
A `RevisionLens` therefore needs a source that can address `(row, version)`,
and that source has to be named before the twin can be written. Do not assume
the existing `WitnessLens` generalizes.

## Gates run (scoped, per the cargo-hygiene rule — no `--all`)

- `cargo test -p lance-graph-contract` → 1123 + 7 + 8 + 7 + 4 + 21 pass, 0 fail
- `cargo clippy -p lance-graph-contract --all-targets -- -D warnings` → clean
- `cargo clippy -p lance-graph-planner --all-targets -- -D warnings` → clean
  (downstream consumer of the changed contract surface)
- `cargo test --manifest-path crates/deepnsm-v2/Cargo.toml` → 98 pass, 0 fail
- `cargo clippy --manifest-path crates/deepnsm-v2/Cargo.toml --all-targets -- -D warnings` → clean
- `cargo fmt` clean on both crates
- `cargo run -p lance-graph-contract --example dispatch_guard_redundancy` →
  34/34 flip, ALL GATES GREEN (unchanged from pre-migration)

## ⊘ Correction (2026-07-29, CodeRabbit mechanical-fixes pass)

The "Still outstanding" section above says **14** gathered `window:`
parameters remain, but its own table sums to **20** (7 + 7 + 1 + 1 + 4). The
arithmetic: table total 20; minus the 4 `WitnessStream` rows (separately
adjudicated NOT-A-VIOLATION, not part of this migration's scope) = 16
genuinely outstanding, of which 1 is BLOCKED (`WitnessWindow.rows`) and 1 is
an example call site (`probe_dcsw2_basin_rung.rs`) — leaving **14 live
migratable parameters** (`contract/witness_fabric.rs` gathered originals: 7;
`planner/nars/meta_basin.rs`: 7), which is what the "14" was counting. Unit
made explicit: parameters, not call sites.
