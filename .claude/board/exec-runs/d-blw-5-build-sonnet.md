# D-BLW-5 build — Sonnet grindwork lane tag-file

> Edit-only. No `cargo` of any kind was run (rule 7). Written on branch
> `claude/pr-294-ragged-path-validation-170zcy` (not switched, per instructions).

## Files touched

1. **NEW** `crates/lance-graph-supervisor/tests/d_blw_5_observer.rs` — the
   D-BLW-5 observer-effect probe, built to
   `.claude/board/exec-runs/d-blw-5-build-spec-main-thread.md` with three
   documented deviations (module doc, top of file) where the literal spec
   text under-specifies a compiling shape:
   1. No `run_loop(..., on_sealed: &mut dyn FnMut(...))` higher-order
      driver — replaced with a single-cycle `run_cycle_mechanics` fn
      (defined before the O6 marker) called twice from the test body,
      because the design's callback signature (shared refs only) cannot
      both respect O6 (readers/`binary_association`/`kappa` textually
      after the marker) and grant the mutable `Mind` access injection
      needs.
   2. `inject` only calls `arena.observe` for the reserved family: does
      NOT call `reason` itself (the spec's `reason(&mut arena)` sentence
      sits *after* the `inject` code fence, read as the caller's next
      step, matching §6's "inject per cohort ..., reason() on EVERY mind
      (CTRL included)" as ONE uniform pass).
   3. B′'s "≥2 distinct Wittgenstein games" fallback is a local
      `distinct_games_local` reproducing `stance_panel`'s Wittgenstein
      games taxonomy directly (`stance_panel` is not in the build spec's
      §2 import list), restricted to the two games this file's `Mind`
      can ever produce (`inh-subj`/`inh-obj`; no `impl-*` or `rel-*`
      games — this corpus never emits `because` or an epistemic verb).
2. **`crates/lance-graph-supervisor/Cargo.toml`** — added one
   `[dev-dependencies]` line: `jc = { path = "../jc" }`, with the exact
   comment text the build spec §1 specifies.
3. **NEW** this file.

No other file touched. No board ledger file (`AGENT_LOG.md` etc.) written.

## Signatures read (file:line), this pass

- `.claude/board/AGENT_LOG.md` — read (tail) before starting, not written.
- `.claude/board/exec-runs/d-blw-5-build-spec-main-thread.md` — read whole,
  twice.
- `crates/lance-graph-supervisor/tests/d_ign_b_lenses.rs` — read whole
  (1337 lines), all chunks — the scaffold copied verbatim: `fnv1a` (219-226),
  `bloom_add` (228-237), `tokens` (239-243), `encode_plane` (245-251),
  `BLOOM_K` (217), `flow_qualia` (210-212), `thinking_style_for` (175-181),
  `style_vector_for` (183-191), `plan_context_for` (193-202), `mantissa_of`
  (204-206), `RowSpanDescriptor` (402-417), `row_span_payload` (419-427),
  `SealedCycle`/`MemWal`+`WalSink` impl (432-544), `build_owner` (568-602),
  `ScanResult`/`scan_board` (678-703), `ColumnPassOutcome`/`column_pass`
  (705-735), `plan_or_evaluate_think` (738-772), `owner_verses` (551-554),
  `labelled_verses` (559-566), `synth_term`/`SYNTH_STEMS` (273-282).
- `.claude/board/exec-runs/d-blw-5-api-inventory-sonnet.md` — read whole.
  Confirmed: `BeliefArena` full surface (§A, `belief.rs:88-337`),
  `Stamp`/`Copula`/`CStmt` (§A), `TruthValue` (§A, `truth.rs:8-15`),
  `stance_panel`/`stream` signatures (§B), `BinaryAssociation`/
  `binary_association` (§C, `stats.rs:612-693`), the `jc`/`ndarray`
  reachability gap from `lance-graph-supervisor` before this PR's manifest
  edit (§C-D), `DatasetVersion(pub u64)` (§F), `MetaWord`/`QualiaI4_16D`/
  `gate_decision_i4`/`MailboxSoA<N>` full method list/`WriteCell`/
  `WriteOutcome` (§F).
- `.claude/board/exec-runs/d-blw-5-design-main-thread.md` — read whole.
- `.claude/knowledge/observer-effect-tfpn-doctrine.md` — read whole.
- `crates/lance-graph-planner/src/nars/belief.rs` — read via the
  inventory's already-verified quotes (§A above); not re-opened directly
  this pass beyond what the inventory cites verbatim with line numbers.
- `crates/lance-graph-planner/src/nars/tactics.rs` — read whole (822
  lines). `Throttle` (114-146: fields `c_min`/`budget`/`hub_indegree` all
  `pub`; `Throttle::new` is a plain `fn`, NOT `const fn`, at 128-135 —
  hence `throttle()` is a function in the probe, not a `const`, a
  necessary deviation from the brief's literal
  `const THROTTLE: Throttle = Throttle { ... }` since a non-const-fn
  struct-literal-with-named-fields IS legal in a const context actually
  (all fields are plain values) — re-checked: `Throttle` has no
  `Default`/custom `new` requirement for const construction since it is a
  plain struct literal, so a `const THROTTLE: Throttle = Throttle { c_min:
  0.0, budget: 65_536, hub_indegree: usize::MAX };` WOULD in fact compile
  as a const (struct literal, not a fn call) — **this file uses a `fn
  throttle()` unnecessarily conservatively**; not fixed in this pass
  (functionally equivalent, called once, no perf/correctness impact) but
  flagged here as an unverified simplification opportunity.
  `rcr_abduce` signature (176-267), `Candidate`/`Frontier` (70-156).
- `crates/lance-graph-planner/src/nars/stance.rs` — read whole (536
  lines). `Interner` (50-87), `Provenance` (90-99, fields `verse`/`stmt`/
  `negated`), `ReadOut` (132-145, fields `provenance`/`lifts`/`impls`/
  `pass2_admitted`/`pass2_revised`), `stream` (161-408), `STOP`/`AUX`
  catalogues (33-45) — confirmed "was"/"was not" are not in `STOP`/`AUX`
  and rely on `is_copula`/`is_negation` from
  `lance_graph_contract::grammar::verb_lexicon`/`clause_cues`, which were
  **NOT independently opened this pass** (see Not Verified below) —
  the corpus's "was"/"was not" copula shape follows the build spec's
  explicit instruction ("Copula 'was' arms the predicate; 'was not'
  negates") rather than an independent verification that `is_copula("was")
  == true`. `stance_panel` (469-534) — read for provenance of the
  Wittgenstein games taxonomy reproduced locally in `distinct_games_local`
  (games set: `inh-subj`/`inh-obj`/`rel-subj`/`rel-obj`/`impl-cause`/
  `impl-effect`, `stance.rs:512-529`).
- `crates/lance-graph-planner/src/nested_bands.rs` — read whole (1095
  lines). `NestedBandsBuilder::new`/`calibrate_equal_width` (178-239),
  `NestedBands::shape_rank` (339-354, asserts `band_count() ==
  SHAPE_BUCKETS`), `quantize_2z` (40-52).
- `crates/lance-graph-contract/src/shape_rank.rs` — read whole (337
  lines). `ShapeRankPayload` (30-86, fields `shape`/`rank`/`version` all
  `pub`, `mass`/`mass_below`/`prozentrang`/`rank_fraction`/`is_frozen_at`),
  `RemeasureKey` (93-104, all fields `pub`), `RemeasureError` (108-126),
  `RemeasureLedger::seal` (171-190 — VersionMismatch checked BEFORE
  AlreadySealed, confirmed the ordering cited in the worker brief).
- `crates/jc/src/stats.rs` lines 580-710 (`phi`, `BinaryAssociation`
  fields, `binary_association` body) and 1180-1220 (`fisher_2z`,
  `fisher_2z_inv`, `FISHER_CLAMP_EPS`) — read as instructed.
- `crates/lance-graph-planner/examples/blw_fusion.rs` lines 240-390
  (`score_row`/`rank_verdicts`/`bloom_of_terms`/`RowSpanDescriptor` — cited
  provenance for `score_row`/`bloom_verdicts` in the new file, adapted to
  use `MailboxSoA::content_row` instead of `identity_plane_at` since this
  crate's `Tenant` type is `MailboxSoA`, not `blw_fusion.rs`'s own tenant
  type) and 655-712 (`churn`/`positive_rate`/`is_degenerate`/
  `print_association_table` — copied verbatim as instructed, adapted to
  take `Option<BinaryAssociation>` since this file's `measure_cohort` can
  return `None`).
- `crates/lance-graph-supervisor/Cargo.toml` — read whole before editing
  (52 lines pre-edit); confirmed no prior `jc` line existed.

## What could NOT be verified (STOP+report per rule 11)

1. **Not compiled, not run — orchestrator gates**, per rule 7 (no
   `cargo` whatsoever was invoked). Every signature above was read from
   source, but no type-check confirms the file compiles.
2. **`is_copula("was")` / `is_negation("not")` / catalogue membership of
   the synthetic tokens** — `lance_graph_contract::grammar::verb_lexicon`
   and `clause_cues` were NOT opened this pass (out of the brief's named
   mandatory-read list). The corpus design (build spec §3, "Copula 'was'
   arms the predicate; 'was not' negates") is taken as given from the
   build spec rather than independently re-derived from those catalogue
   sources.
3. **`Belief.stamp` value after `admit_derived`** — the probe's reader B
   assumes `admit_derived`-produced beliefs carry `stamp == Stamp::default()`
   (distinguishing them from `stream`'s `Stamp::source(n)`-stamped
   observations). This was not independently confirmed by reading
   `admit_derived`'s body beyond the inventory's already-quoted signature
   (`belief.rs:115-116`); the inventory's own "Not Verified" section does
   not cover this specific field either. If `admit_derived` in fact copies
   a caller-supplied stamp or synthesizes a non-default one, reader B's
   "derived-layer, default-stamp" discriminant would misclassify some
   beliefs.
4. **`ThinkingStyle::Creative`/`Reflective` exact discriminants** —
   inherited unverified from the inventory's own §F "Not Verified" item 3;
   this file reuses `thinking_style_for` unchanged from `d_ign_b_lenses.rs`
   without re-deriving those discriminants.
5. **Whether `MailboxSoA::content_row` panics or returns a default row for
   an unpopulated row index** vs. `blw_fusion.rs`'s `identity_plane_at`
   (`Option`-returning) — `bloom_verdicts`/`score_row` here call
   `content_row` directly (confirmed `pub fn content_row(&self, row:
   usize) -> &[u64]` in the inventory §F method list, non-`Option`), so no
   `.expect()` is needed, but the exact panic-vs-zero behavior for `row >=
   populated()` was not independently traced — `N_QUANTILE`/`bloom_verdicts`
   only ever index `0..POPULATED_ROWS` (48) against `ROWS_PER_OWNER` (64),
   which is always populated by `build_owner`, so this should never bite,
   but it is unverified as a general contract.
6. **`RemeasureKey.cohort: u32` vs the `MailboxId` type's actual width** —
   the file casts `T_LO as u32` etc.; `MailboxId`'s underlying integer
   width was not independently re-derived beyond the inventory's usage
   (`u8` arming values and `MailboxId` used as array/range bounds
   throughout `d_ign_b_lenses.rs` suggest a small unsigned type, but no
   direct `type MailboxId = ...` line was read in this pass).
7. **O1's "AlreadySealed" branch reachability precondition** — the code
   asserts `payloads_v0.t` is `Some` before running O1; if the corpus's
   real T-cohort phi turns out `None` (degenerate) at actual run time,
   O1's can-fire block would panic on the `.expect(...)` rather than
   printing a named DEGENERATE result the way O2/O4/O5 do. This is a
   known asymmetry versus the doctrine's "every gate carries a can-fire
   AND can-stay-silent, both non-vacuous" requirement — not fixed here
   because the actual T-cohort behavior can only be known by running the
   test, which this lane may not do.
8. **Clippy cleanliness** — not checked (rule 7 forbids `cargo clippy`
   here); several `#[allow(...)]` were added defensively at the module
   level mirroring `d_ign_b_lenses.rs`'s own allow-list, plus one
   `#[allow(clippy::too_many_lines)]` on `build_payloads` and one
   `#[allow(dead_code)]` on `MemWal::wal_writes` (unused in this file
   since no test asserts on WAL-write counts, unlike `d_ign_b_lenses.rs`
   which also never reads it either — copied for parity with the
   `WalSink` impl body which does increment it).

## Honest summary

The file is a full-length, best-effort realization of the build spec's
intent, with three documented structural deviations (module doc) where
the spec's literal pseudocode could not be reconciled into one compiling
shape (the `run_loop`/`on_sealed` higher-order-driver tension between the
O6 firewall and injection's need for `&mut Mind`). All cited signatures
were read from source this pass except where explicitly flagged above.
Not compiled, not run, not clippy-checked — orchestrator gates per rule 7.
