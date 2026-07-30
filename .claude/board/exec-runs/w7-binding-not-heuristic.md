# W7 — binding-not-heuristic falsifier — execution record

## Files touched
- NEW: `crates/lance-graph-planner/examples/probe_binding_not_heuristic.rs`
  (only code file created; no core/contract change).
- Board: `EPIPHANIES.md` prepend (`E-A-CHIP-BEARS-LOAD-OR-IT-IS-A-JOKE-1`),
  this file. Same commit, per the board-hygiene rule.

## What W7 closes that W6 left open
W6's binder took ALREADY-RESOLVED `(pronoun_pos, antecedent_pos)` fixture pairs
— it proved the write/escalate path, never the resolution. W7 supplies the two
RESOLVERS (cheap recency baseline vs. structural binding) over shared
tagger-level features, with gold annotations read ONLY by assert arms, and
makes them disagree on real KJV text. The operator's 2026-07-30 entropy-work
ruling is the spec: a store must be effort on behalf of the thought after you;
bookkeeping a better algorithm makes redundant is a joke pretending to be
thinking. Mechanical form: the pull-test.

## Fixture honesty
- Features are what a POS tagger + chunker emits (POS, number, animacy,
  relative spans, main-clause conjunctions, complementizers). No annotation
  encodes an antecedent.
- The structural resolver COMPUTES from those features (R1 complement-subject
  → matrix subject; R2 subject continuity with relative-span skip + animacy
  repair). Two rules only — deliberately coarse; this falsifies the CHIP, it
  is not a coreference system.
- Two load-bearing-annotation tests guard against fixture-supplies-the-answer:
  `animacy_check_is_load_bearing` (bare R2 picks `eyes`@2, which is NOT gold)
  and `relative_span_skip_is_load_bearing` (without the span, `god`@15 would
  be clause-0 subject material).

## Gates (all falsifiable, all green)
- **B1 divergence** — Gen 3:1: heuristic → `god`@15 (d=−4), binding →
  `serpent`@2 (gold). Falsifier: resolvers agreeing, or recency being right.
- **B2 escalate-not-clamp** — heuristic's wrong answer FITS ±8 (the temptation
  is real and asserted); binding's right answer is d=−17 → binder escalates,
  nibble stays 0. Falsifier: a stored nibble (either clamped or cheap-wrong).
- **B3 stay-silent** — Gen 3:7: both resolvers == gold on both `they`s; chips
  bound at −5 and −3 (distinct, nonzero — anti-vacuity).
- **B4 pull-test** — chip chain `they`@12 → `they`@9 → `them`@4 via
  `resolves_to` (two nibble follows, no far verdict cached); heuristic-only
  reconstruction fails on 3:1. Falsifier: chain not composing, or the cheap
  resolver sufficing everywhere.

## Verification
- `cargo run -p lance-graph-planner --example probe_binding_not_heuristic`
  → ALL GATES GREEN (re-run after fmt: still green).
- `cargo test -p lance-graph-planner --example probe_binding_not_heuristic`
  → 4/4.
- `cargo clippy -p lance-graph-planner --example probe_binding_not_heuristic
  -- -D warnings` → clean (pre-existing unrelated `cognitive-shader-driver`
  duplicate-bin-target warning only).
- `cargo fmt -p lance-graph-planner` → ran; gates re-verified green after.

### Workspace-wide checks — MEASURED, then deferred (not skipped)

Review asked for `cargo fmt --all` + `cargo clippy --all-targets
--all-features`. Both were investigated rather than argued from practice:

- **`cargo fmt --all -- --check` → 1,094 hunks / 20,307 diff lines across 64
  files in 9 crates** (55 in `lance-graph-ontology`; also `bgz-tensor`,
  `causal-edge`, `ogar-emitter`, `ogar-encryption`, `ogar-from-ruff`,
  `ogar-render-askama`, `sigma-tier-router`, `surreal_container`).
  **Zero hunks in this PR's file** — `probe_binding_not_heuristic.rs` is
  fmt-clean under the workspace-wide config, which is the property the
  guideline protects. Running the sweep here would attach a 20k-line reformat
  of nine untouched crates to a 752-line additive probe; the diff stops being
  reviewable and pre-existing drift lands silently under an unrelated title.
  Deferred to its own PR — logged as `TD-WORKSPACE-FMT-DRIFT`.
- **`clippy --all-targets --all-features`** — blocked, not declined by
  preference: all-features is the known `TD-LANCE-GRAPH-ALL-FEATURES-DELTA-BREAK`
  surface (the `delta` feature does not build), so the invocation cannot pass
  regardless of this diff. The scoped `-D warnings` run above covers every line
  the PR adds.

All nine review threads (codex ×2, CodeRabbit ×7) addressed and resolved.

## Honest boundaries
- Escalation is side-band (the binder's return), not row state — a 0 nibble
  alone is indistinguishable from never-attempted. Same boundary W6 recorded.
- The resolvers are two rules over annotated features; real-corpus coreference
  lives in `probe_eyes_opened.rs` / deepnsm. W7's claim is exactly: the chip's
  placement algorithm is structural, and its refusal on 3:1 is a choice
  (the wrong answer was storable), not a range limitation.

## Follow-ons surfaced (not done here)
- Quorum binder: `Locus::Quorum`/`Contradiction` slots are never
  production-written while grading rescans Θ(N·k)
  (`TD-LENS-QUORUM-SCANS-THE-WHOLE-LENS`) — one defect, two views; W6/W7's
  binder shape retires both.
- `PROBE-RUNG-ELIGIBILITY` (CONJECTURE row in `zero-copy-lens-law.md`): the
  pull-test is now its defined pass/fail; run per locus.

## Post-review hardening (codex 2 findings + CodeRabbit 7 comments, same PR)

- **Gold type-separated (codex #1, CodeRabbit ×3):** `Tok` no longer carries a
  `gold` field; fixtures return `(Vec<Tok>, Gold)` where `Gold` is a separate
  `(pronoun_pos, antecedent_pos)` list read only by assert arms. A future
  `toks[p].gold` shortcut is now a COMPILE ERROR — closed by type, not
  convention (strictly stronger than the suggested invariance test).
- **B2 exercises the binder (codex #2, CodeRabbit ×2):** the heuristic's wrong
  target is bound into scratch rows; the gate asserts `Ok(-4)` AND stored
  nibble `-4` — "storable-but-wrong" proven at the binder, not by re-deriving
  its range predicate. New unit test `binder_accepts_the_tempting_wrong_target`.
- **Relative-span test was VACUOUS (codex P2, CodeRabbit Major):** confirmed —
  `subject_of_clause` scans forward and English is head-first, so `serpent`@2
  wins with or without the filter; the old "load-bearing" claim was false for
  this rule. Replaced with the honest pair:
  `clause_segmentation_is_load_bearing` (erase the `and` boundary →
  resolution FAILS — the true 3:1 counterfactual) and
  `relative_span_filter_can_fire` (synthetic clause where the first candidate
  is in-relative — filter changes the outcome, asserted against the
  unfiltered pick). The earlier claim in this file's "Fixture honesty" section
  is superseded by this addendum.
- **Pull-test had no divergent STORED chip (codex P1):** confirmed — 3:1
  escalates (no chip) and 3:7's chips match the heuristic, so stored state was
  entirely heuristic-reconstructible. Added the constructed in-range
  interposition fixture ("the man which the boy saw slept, and he smiled",
  labelled built-English, not KJV): recency → boy (d=−4), binding → man
  (d=−7, gold, BINDS). New gate B4 asserts the STORED nibble (−7) differs
  from the heuristic reconstruction (−4); old composition gate renumbered B5.
  New unit test `divergent_chip_differs_from_heuristic_reconstruction`.
- **EPIPHANIES wording (CodeRabbit Minor):** "chip stores the verified answer"
  → "stores the POINTER the verification selected; the answer is recomputed
  through it" — pointer-not-answer kept consistent.
- **Declined, with reasons:** workspace-wide `cargo fmt --all` / `clippy
  --all-targets --all-features` (repo practice is scoped `-p` runs — a
  workspace-wide sweep would touch unrelated crates/files and the
  all-features build is the known `TD-LANCE-GRAPH-ALL-FEATURES-DELTA-BREAK`
  surface); methods-on-carrier nitpick (the carrier litmus targets cognitive
  state carriers, not probe fixtures — two flat resolver functions keep the
  falsifier legible; noted, not a doctrine violation).
- **Operator ruling landed mid-review** (see
  `E-64K-THOUGHTS-DONT-DO-QUORUM-PLASTICITY-BREATHES-FROM-TENSION-1`): the
  quorum-binder follow-on this file proposed is WITHDRAWN — 64k thoughts
  don't do quorum; plasticity does that work; contradictions are fuel.
  `TD-LENS-QUORUM` survives as scan-cost debt only.

Re-verified after all changes: 5 gates green (B1–B5), 7 unit tests green,
clippy `-D warnings` clean, fmt clean.
