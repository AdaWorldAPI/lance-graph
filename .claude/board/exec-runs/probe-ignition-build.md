# PROBE-IGNITION — build record (Sonnet build lane)

**Deliverable:** `crates/lance-graph-supervisor/tests/probe_ignition.rs` — written,
not compiled, not run (edit-only per the guardrails; no `cargo` of any kind was
executed by this lane). Two tests: `probe_ignition_scan_and_cast_no_messaging`
(the main 6-cycle run, G1-G8, G10, G11, self-scans, not-claimed block) and
`probe_ignition_g9_drained_writer_retry_footgun` (the G9 side fixture, its own
`MemWal`-derived `FlakyWal`).

## What was built

- Whole file wrapped in `#[cfg(feature = "cycle-driver")] mod probe_ignition { ... }`
  (the w2b pattern). **No Cargo.toml change** — verified both manifests before
  writing: `lance-graph-supervisor`'s dev-dep on `cognitive-shader-driver` and
  its `cycle-driver` feature (`dep:lance-graph-planner`, `dep:tokio`) already
  existed exactly as the brief assumed.
- The pinned run shape verbatim: `FLEET_OWNERS=64`, `ROWS_PER_OWNER=64`,
  `POPULATED_ROWS=48`, `CORPUS_VERSES=3072`, `SCOPE=0..32`, `CYCLES=6`,
  `WAKE_CYCLE=4`, the 7 cohorts at their exact id ranges (IGNITE_A 0..6,
  IGNITE_C 6..12, REST 12..20, CONTRA 20..24, UNARMED 24..31, ORPHAN 31,
  OUTSIDE 32..64).
- All 11 gates (G1-G11, with G2/G3 carrying their a/b/c sub-parts), each with
  both can-fire and can-stay-silent halves, printed as
  `eprintln!("probe.ignition.G<n> ...")` plus asserts.
- The G2a/G3b/G11 compile-time self-scans via `include_str!("probe_ignition.rs")`.
  Needles are built by string CONCATENATION (never written as one contiguous
  literal anywhere in the file, including inside the scan code itself) so the
  scan cannot self-match — documented in a comment at the scan site.
- Qualia: `flow_qualia()`/`block_qualia()` re-derived locally with a doc
  comment citing `cycle_driver.rs:1669`/`:1675` as provenance (those fixtures
  are `#[cfg(test)]`-private, not importable).
- Corpus: `BLW_KJV_TSV` env (default `/tmp/kjv_verses.tsv`), deterministic
  synthetic fallback with printed provenance, pairwise-distinctness guard on a
  5-owner content-plane sample.
- Write-back pass strictly AFTER apply (never during compute); the REST
  branch records a rest and skips `run_cycle` entirely on zero staged casts.
- G9 side fixture: 2 owners, own `MemWal`-wrapping `FlakyWal` with a
  `fail_next` `AtomicBool`, injected WAL failure, retry via `seal_cycle`, then
  the drained-writer zero-slot observation via a fresh `collect_casts` call.
- G10: probe-local `column_pass` (counts `missing`) vs the shipped
  `run_cognitive_work_gated_over` (silently drops the same id) on a
  single-element `[ORPHAN_ID]` list; asserts the two totals differ by exactly 1.
- §5 Not-claimed block, all 12 items, printed at the end of the main test.

## Deviations from the design note (both stated in-file, with reasons)

1. **Energize uses a direct `owner.energy[row]` write, not `apply_edges(&[(row,
   CausalEdge64)])`.** `causal_edge::CausalEdge64` is unreachable from this
   crate: `cognitive-shader-driver` depends on `causal-edge` privately (no
   `pub use` anywhere in its `lib.rs` or `mailbox_soa.rs` — grepped), and
   `lance-graph-supervisor`'s own `Cargo.toml` has no dependency edge to
   `causal-edge` at all. Adding one is a Cargo.toml change, which the brief
   forbids outright. `owner.energy` is the exact public field `apply_edges`
   itself mutates (`mailbox_soa.rs:66,362`), and is the field
   `examples/blw_fusion.rs:515` already writes directly for its own
   energizing — precedent in the same crate family, not an invented mechanism.
2. **The scheduled wake runs at the TOP of cycle 4's loop body (before
   `scan_board`), not as step 14 after the write-back as the design's own
   numbering literally lists it.** The design's cohort table says "wake at
   c4 → Evaluation" — cycle 4's own gate read must already see the post-wake
   mantissa. Read literally (wake after write-back), the effect would only be
   visible from cycle 5 onward, contradicting the table. Documented at the top
   of the file as a placement correction, not a behavioural addition (still
   exactly one write, still gated on `c == WAKE_CYCLE`, still REST-only).

No other deviations. The mid-flight G2b correction (Planning→Prune casts are
`Native`/the gate's mint, not `Elixir`/the style's, when the gate says Prune)
was folded in as instructed, with a one-line comment citing the design note's
§2 step 8 as the authority, at both the G1 c1-decomposition site (20 Flow + 4
Block, not 24 uniform) and the G2b per-cycle exec-conditioned-on-`(from,to)`
check.

## Bugs I found and fixed during my own re-read (before declaring done)

Per the guardrails' "read the exact signature of everything you call" rule, I
re-read the whole file after the first draft and caught three real defects
that would either fail to compile or silently prove nothing:

1. **`run_cycle`'s `CycleFrame` base version was hardcoded wrong.** First
   draft used a stub `DatasetVersion(0)` for every cycle; `MemWal::commit_cycle`
   rejects any `base != head`, so cycle 2 onward would have failed with
   `WriteFailed("stale base...")`. Fixed to `sink.head()`, captured fresh each
   cycle.
2. **A tautological assertion** (`assert_eq!(sink.wal_writes(), sink.wal_writes(),
   ...)`) in the REST branch — compared a value to itself, which is exactly
   the vacuous-assertion pattern this workspace's own falsifiability rule
   forbids. Fixed by capturing `wal_writes_at_top_of_cycle` before the
   cast-staging passes and comparing against that.
3. **`end_of_c5` was never actually captured during cycle 5** — the first
   draft set it in a fallback block AFTER the whole loop ended, by which point
   the fleet was already in its end-of-c6 state, making the G6 "byte-identical
   across a rest cycle" comparison compare c6 to itself. Traced through the
   cohort arcs by hand and confirmed **both c5 and c6 are zero-cast rest
   cycles** (REST reaches Evaluation with mantissa 0 at c4's write-back and
   never re-fires); fixed by capturing the fingerprint inside the loop's own
   `c == 5` rest branch, and changed the c6 comparison to `panic!` (rather
   than silently skip) if `end_of_c5` is somehow still `None` — an honest
   failure instead of a mask.
4. **A `Option<&T>` vs `Option<T>` type mismatch** in the G3b "changed set"
   computation, caused by relying on default-binding-mode ergonomics through
   `.filter(|(id, p)| ...)` over a `HashMap::iter()` (whose `Item` is `(&K,&V)`
   and whose `.filter()` predicate receives `&Item`, i.e. one more reference
   layer than the naive reading suggests). Rewrote as an explicit
   `for (&id, &prev) in &snap_after_cast { if snap_post_apply.get(&id) !=
   Some(&prev) { ... } }` loop, which is unambiguous.

I did not have a compiler to confirm these were the ONLY defects; see below.

## Signature mismatches found between the design note and source

- **§2 step 8's literal G2b claim was already flagged and corrected by the
  orchestrator's mid-flight message** (see above) before I built the G2b
  assertion at all — I did not need to independently re-discover this.
- Everything else in the design note's API references matched the Sonnet
  inventory and my own source reads exactly: `run_cycle`/`collect_casts`/
  `seal_cycle`/`apply_sealed_transitions`/`shade_owner`/
  `run_cognitive_work_gated_over` signatures, `StyleStrategy::plan`/
  `::reliability_for`/`::intended_move` (private, reached only via `.plan()`),
  `MailboxSoA` constructor/`write_row`/`apply_edges`/`consume_firing`/
  `pending_count`/`qualia_at`/`meta_at`/`energy` field, `KanbanColumn::
  advance_on_gate`/`next_phases`/`can_transition_to`, `gate_decision_i4`/
  `trust_texture_i4`/`flow_state_i4` (the last not directly called, only its
  logic re-derived for the G4 anti-rig comment), `emit_bootstrap_intent`/
  `rebind_bootstrap`, `BatchWriter::{cast,on_behalf_of,intent_moves,
  drain_pending_payloads}`.
- One inventory gap I closed myself: the API inventory listed
  `TrustTexture` only as a return type of `trust_texture_i4`, not its own
  definition. I independently grepped and confirmed
  `lance_graph_contract::mul::TrustTexture` (`pub enum`, `#[derive(Debug,
  Clone, Copy, PartialEq, Eq)]`, `Calibrated` variant) before using it in the
  G4 assert.

## What I could NOT verify (no compiler; report honestly)

- **Not compiled, not run.** Every signature was read from source in this
  same pass; the file's overall correctness beyond that manual trace is
  unverified.
- I did not independently re-derive `flow_state_i4`'s exact match arms from
  source in this pass (I read them earlier in the session and used the logic
  in prose/comments only — I never call `flow_state_i4` directly, only
  `gate_decision_i4` which composes it internally).
- I did not check whether `clippy -D warnings` would flag anything beyond the
  casts I pre-emptively allowed (`cast_possible_truncation`,
  `cast_possible_wrap`, `cast_sign_loss`) and the pairwise-distinctness loop I
  rewrote to avoid `needless_range_loop`. There may be other nits (e.g.
  `too_many_lines` on the main test function, which is long by construction
  given the 6-cycle loop plus 5 epilogue blocks).
- I did not verify `MailboxId`'s underlying type by opening
  `collapse_gate.rs` in a prior turn of THIS session — I did open it in this
  build pass specifically and confirmed `pub type MailboxId = u32;`.
- The exact wire-format assumption that `[900, 901]` and other bare integer
  literals infer as `MailboxId` (`u32`) from context is standard Rust type
  inference and should hold, but is unverified by a compiler.

## Board-adjacent files touched

Only this tag-file (`.claude/board/exec-runs/probe-ignition-build.md`) and the
deliverable itself. `AGENT_LOG.md` was read, not written (one-writer rule).
No `cargo`, no `git commit`, no branch change.
