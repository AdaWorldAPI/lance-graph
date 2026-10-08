# 2026-10-08 — CE64 ISA: strict decoder, register methods, field contracts

**Status:** TEST-PINNED. Reasoning chooses the operation; `CausalEdge64`
defines what that operation means.

Code: `crates/causal-edge/src/isa.rs` (new), `edge.rs`, `syllogism.rs`,
`network.rs`. Tests: `crates/causal-edge/tests/ce64_isa_contract.rs`,
`ce64_isa_golden.rs` (+ `ce64_legacy_capture.txt`), `ce64_op_contract.rs`;
`crates/lance-graph-planner/tests/gadamer_isa_revision.rs`.

## What changed

- **Strict decoder.** Of the 16 inference codes, exactly `+1` Deduction,
  `+2` Induction, `-1` Abduction, `+4` Revision, `+5` Synthesis execute
  (`Opcode::decode`). The other 11 (`0`, `-8`, `-2`, `±3`, `-4`, `-5`,
  `±6`, `±7`) return `IsaFault::Unsupported { mantissa }`. Before, `±6` and
  `±7` ran the Synthesis mean and came back stamped `±6` and `7`; `0`, `-8`,
  `-2`, `±3`, `-4`, `-5` ran another instruction and came back stamped with
  that instruction's code.
- **`forward` only dispatches**: decode, then `execute(op, ..)`. It now
  returns `Result`; so do `forward_chain`, `NarsEngine::forward_edge` and
  `replay_step` (`ReplayError::Isa`).
- **Register methods**: `deduction`, `induction`, `abduction`, `synthesis`
  take an explicit `Compose` (S/P/O payload algebra) and return `Self`;
  `revision(rhs)` is truth-only (writes F, C; keeps every other field of
  `self`; no `Compose`); `counterfactual` and `intervention` are partial and
  always refuse.
- **One truth function per instruction** (`isa::truth`); `forward`, `learn`
  and `syllogize` delegate. The three previous copies are gone.
- **Field contracts** (`isa::contracts`): reads / computes / passes /
  constants, payload algebra, failure condition, measured by perturbation in
  `ce64_isa_contract.rs`.

## Field-liveness matrix

| instruction | computes | passes through | constant | Compose | fails |
|---|---|---|---|---|---|
| forward / `execute` | S/P/O, F, C, Pearl (AND) | inference ← B, Direction ← B, Plasticity ← B | W = 0, Epi5 = 0 | yes | B's code has no implementation |
| `revision` | F, C | everything else ← A | — | no | never |
| `learn` | S/P/O, F, C, Plasticity | Pearl, Direction, inference, W, Epi5 ← A | — | no | never |
| `syllogize` | S/P/O, F, C, Pearl, inference | — | Direction 0, Plasticity 0b111, W 0, Epi5 0 | no | `None` without a figure |

The W/Epi5 zeroing in `forward` and `syllogize` is legacy behaviour, now
DECLARED, not changed.

## Golden vectors

- Legacy capture: 126 raw-word vectors recorded from the pre-ISA code. 71
  reproduce bit for bit; the 55 whose weight carries an unsupported code
  now refuse with that code. Nothing else moved.
- Normative: 23 forward vectors through the named methods, checked against
  an independent f64 reference (±1 on F/C) and the declared contract.

## The max-confidence revision defect (not normative)

`revision(c=255, c=255)` returns `f = c = 0`, in `forward`, `revision` and
`learn`. Path: `c = 1.0 >= 0.999` → `evidence_weight = f32::MAX` for both →
`ws = MAX + MAX = +inf` → `f = inf/inf = NaN`, `c = NaN` → the float-to-u8
cast saturates NaN to 0. Not a u8 overflow, LUT bin, or zero-denominator
fallback. Also: ONE operand at 255 is finite but dominates (its weight is
`f32::MAX`), where the capped reference gives a weighted mean. Both kept
out of the normative set and pinned.

## Gadamer stays above

`GadamerRevision` decides whether evidence is admitted; `revision` only
computes. Ten echo or closed-cycle encounters leave confidence unchanged
because the gate never calls the register; ten new roots raise it. The
register alone would inflate an echo (pinned).

## Disable runs (anchor asserted, each red)

CF decoded as Synthesis (6 tests) · forward ignores the decoded op (4) ·
learn clears W (4) · revision clears Epi5 (1) · diverging revision formula
(4) · known-bad promoted to normative (2) · a policy type named in `isa.rs`
(1) · `counterfactual` computes (1) · Gadamer gate bypassed (2).

## OPEN

- `forward`'s revision step (+4) composes payload; `revision(rhs)` does
  not. Whether a payload-composing revision step should exist is a
  decision, not made here.
- The c = 255 saturation rule (collapse and dominance). Smallest fix: cap
  `c` before `evidence_weight` as the f64 reference does.
- `forward` writes the executed code into the result's bits 46..49. The
  transition-loop reading ("do not whisper the operation back into the
  result edge") says it should not. Pre-existing; not changed here.
- `learn` mixes arithmetic (revision) with policy (archetype reassignment,
  freeze thresholds 0.9 / 0.7). Named split, not done.
- `syllogize` chooses the rule from the figure (a choice) and computes it.
- Planner truth copies outside the CE64 register (`TruthValue`,
  `nars_infer`, `NarsTables`) still differ; out of scope.
- **#1406 merge**: its probe calls `forward` as infallible and needs a
  `.unwrap()` once both land.
