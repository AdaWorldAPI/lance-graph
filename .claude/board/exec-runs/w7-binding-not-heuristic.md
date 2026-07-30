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
