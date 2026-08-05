# oracle-funnel-probe-v1 — PROBE-ORACLE-FUNNEL: the generate→validate→segment funnel, staged

> **Status:** Stage 0 PRE-REGISTERED 2026-08-05 (this section written BEFORE the
> probe ran; results appended below after). Stage 1 (LLM arm) GATED.
> **Scope:** measurement only. No new OGAR mints, no NARS-34 ids, no new
> lance-graph code. Stage 0's harness is an `ogar-loco` example (OGAR-side),
> because every funnel gate lives there and the wishlist's dependency ruling
> ("no cross-repo dependency in either direction — the consumer wants the
> TABLE AS DATA", OGAR `.claude/handovers/2026-08-05-1430-…-to-ogar-loco.md`)
> forbids importing `ogar-loco` here. Candidates and legends cross as data.
> **Consumes (all shipped, OGAR #241 + #244):** `CheckedVocabulary` /
> `validate` (W-1 compose-then-check), `FnSpec.name` in the composed table
> (W-2, OQ-1 answered "in the spec"), `statement_bounds` (R5),
> `telemetry::FunnelTally` + `RefusalGate` (W-4).
> **Feeds:** the eventual PROBE-GADAMER-BAG (LLM emits NARS-tactic bodies →
> funnel → Gadamer-projection falsifiers), which waits on the operator's W-3
> NARS-34 byte assignments + `ResultBehavior` decision. Stage 0 deliberately
> uses ONLY the existing blockly shared core so nothing waits on that gate.

## 1. The question Stage 0 answers

Before any LLM is wired: **does the funnel discriminate?** A refuse-don't-guess
firewall is only a firewall if structured input survives it and garbage does
not. Stage 0 measures the funnel's transfer function on three deterministic
generator arms that bracket what an LLM could emit — a floor (uniform bytes),
a mid-model (knows the legend, not the grammar), and a ceiling (knows the
stack discipline). An LLM arm (Stage 1) is only worth API cost if the gap
between floor and ceiling is wide enough to place a generator inside it.

## 2. Design (deterministic, seeded — SplitMix64, seed 0x9E3779B97F4A7C15)

N = 1000 bodies per arm, `LaneShape::Pairs`, body length L uniform in 1..=16
calls (well under the 180-call budget, so `BodyError::Overflow` /
`ValueBeyondShape` are structurally excluded — the body-ENTRY gate is
therefore NOT exercised by Stage 0; recorded as an honest boundary, not a
finding). Vocabulary: the empty-domain blockly core (`EmptyVocab` pattern from
`statements.rs` tests) through `validate()` — so the W-1 compose-then-check
path is on the measured path.

- **Arm R (floor — uniform random):** every call = uniform random byte as
  `FnIndex` + uniform random value byte. Models an LLM that knows nothing.
- **Arm G (mid — legend-constrained):** calls drawn uniformly from the slots
  the composed table declares BOTH `stack_arity` and `pushes_result` for
  (the segmentable set). Models an LLM that read the legend but has not
  internalized the stack discipline.
- **Arm W (ceiling — stack-aware walk):** tracks depth; picks uniformly among
  covered calls whose arity ≤ current depth; ends only at depth 0 or 1.
  Models an LLM that has internalized the discipline. By construction every
  body should segment.

Per candidate: `FunctionBody::from_calls` (infallible under the constructions
above) → `statement_bounds(&checked, &body)` → tally into `FunnelTally` via
the shipped `From<&StatementError>` mapping. Report per arm: survivor rate,
`nonzero_gates()` distribution, and for survivors the statement-count
mean/max (does the trailing-expression rule fire on real output?).

**Legend cost (the Stage-1 anchor):** render the legend a Stage-1 prompt
would carry — one line per named+covered slot: `NAME arity=N pushes=B` from
the composed table (W-2's "legend = serialization of the validated table").
Report bytes, lines, and estimated tokens (bytes/4, stated as an estimate).
Prompt-cache measurement is DEFERRED to Stage 1 — it needs a real API call
and is not fakeable honestly.

## 3. Pre-registered expectations (written before running)

- **E1 (floor):** Arm R survival < 5%, refusals dominated by
  `StatementUncovered` (most of the 256-byte space is uncovered).
- **E2 (mid):** Arm G survival strictly between R and W, refusals dominated
  by `StatementUnderflow` + `StatementDangling` (covered calls, wrong order).
- **E3 (ceiling, two-sided):** Arm W survival = 100%. A shortfall is a REAL
  finding either way — a generator bug or a gate over-fire — and must be
  diagnosed, not averaged away.
- **E4 (discrimination — the KILL):** spread between Arm R and Arm W
  survival ≥ 50 points. Below that the funnel does not discriminate and the
  Stage-1 LLM arm is NOT justified on this vocabulary.

## 4. Stage ladder

- **Stage 0 (this probe):** deterministic arms, no LLM, no API. Runs now.
- **Stage 1 (GATED on operator word + API):** the rig `CompletionModel`
  adapter (D-RLG-1) emits N candidates against the rendered legend; same
  funnel, same tally; adds measured legend token cost under prompt caching.
  Success bar: LLM survival must beat Arm G (else the model adds nothing a
  legend-constrained sampler doesn't), with validity feedback (W-4's safe
  half) looped raw and NO fitness feedback (D-BLW-5 payload law).
- **Stage 2 (GATED on W-3 mint):** PROBE-GADAMER-BAG — NARS-34 vocabulary,
  survivors executed and scored by the shipped Gadamer-projection falsifiers.

## 5. Results (appended after the run — see §3 for what was predicted)

**Run 2026-08-05** — OGAR `crates/ogar-loco/examples/funnel_probe.rs`
(OGAR PR #245), deterministic, bit-exact reproducible. Gates green
(`test`/`clippy --examples -D warnings`/`fmt`, all `-p ogar-loco`).

| arm | survived | dominant refusal | survivor statements |
|---|---|---|---|
| R (floor) | 5/1000 (0.5%) | `StatementUncovered` 831, `StatementUnderflow` 164 | mean 1.00, max 1 |
| G (mid) | 25/1000 (2.5%) | `StatementUnderflow` 970, `StatementDangling` 5 | mean 1.20, max 2 |
| W (ceiling) | 1000/1000 (100%) | — | mean 2.32, max 8 |

Segmentable set: 50 slots. Legend: 50 lines, 1531 bytes, ~382 tokens
(bytes/4 estimate). Gate-identity spot check passed (underflow →
`StatementUnderflow`).

**Verdict against §3:**

- **E1 MET** — 0.5% < 5%, `Uncovered` dominant.
- **E2 MET** — 2.5% strictly between, `Underflow` dominant.
- **E3 MET** — 100%, and the mean-2.32/max-8 statement counts show the
  trailing-expression rule fires on real output, not just fixtures.
- **E4 MET (the KILL passes wide)** — spread 99.5 points vs the 50-point
  bar. The funnel discriminates; **Stage 1 is justified** on this
  vocabulary (still gated on operator word + API).

**Finding beyond the pre-registration — THE LEGEND IS NOT THE GRAMMAR.**
Arm G was designed as the mid model and landed at the floor: knowing the
full vocabulary table lifts survival only 0.5% → 2.5%, while internalizing
the stack discipline lifts it to 100%. Nearly all of the funnel's
constraint is the discipline, not the vocabulary. Consequence for Stage 1
prompt design: serializing the legend into the prompt is necessary but
nearly worthless alone — the prompt must teach the postfix/stack
discipline (or the generator must be constrained decoder-side), and the
cheap ~382-token legend leaves ample budget for that. Board entry:
`EPIPHANIES.md` `E-THE-LEGEND-IS-NOT-THE-GRAMMAR-1`.
