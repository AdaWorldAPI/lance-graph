# D-RPF-G2 — the law for decision-bearing wiring; P8 audit closed (2026-10-10)

Follows `entries/2026-10-09-d-rpf-table-1-fast-path-cost.md` (#1426), which
already holds the P8 semantic audit. This entry re-checks it against code,
records the G2 boundary, and closes two of its four open items.

## P8 — re-checked in code

| question | answer | where |
|---|---|---|
| Which law does replay run? | Whatever `NarsTables` the caller passes; every caller passes `build(1)` (B1) except `tests/chain_confidence.rs` (16) | `chain_replay.rs::replay_step` |
| Who calls it outside tests? | Nobody. `replay_chain`, `counterfactual_replay` and `pearl::reason`'s `CutContext` are built only under `#[cfg(test)]` (`chain_admission.rs:165`, `pearl.rs:427`, `chain_counterfactual.rs:272`), in `tests/` and in examples | read 2026-10-10 |
| `NarsEngine::tables` consumed? | No production reader. `NarsEngine::new` (production instance `strategy/chat_bundle.rs:42`) allocates `build(1)`; `revise_fast`/`deduce_fast` have test callers only | `cache/nars_engine.rs` |
| Does replay overwrite forward's truth? | **Yes, confirmed.** `CausalEdge64::execute` computes `op.truth(..)` and packs it with `op.inference_type()`; `replay_step` then writes revision's F/C over it. The emitted edge names the weight's opcode (e.g. Deduction) and carries a revision truth | `causal-edge/src/edge.rs` `execute`; `chain_replay.rs` `replay_step` step 3 |

No production behaviour depends on any of this today.

## G2 — recommendation (DECISION, scoped)

- DECISION: future decision-bearing consumers of replay/revision use exact
  revision (law A). A table resolution is admissible only for a hot,
  dependent replay whose measured decision-flip rate on that workload's own
  corpus is below a bar stated before the run.
- SCOPE: new wiring only. `NarsEngine::new`'s `build(1)` default and every
  existing public default stay as they are (operator-held).
- BASIS: #1426 — no Bn is decision-stable (B16: 4.7 % verdict flips); A beats
  every table cold and beats B8/B16 on random access.
- REVISIT WHEN: a real recorded chain corpus exists (both #1426 corpora are
  synthetic) or a consumer needs replay throughput A cannot meet.

G1 (the (255,255) cell and finite vs dogmatic evidence) is untouched and stays
open; it is a separate decision.

## Closed here

1. **Load-bearing fixture re-derived under law A.**
   `tests/d_rpf_table_1.rs::a_load_bearing_fixture_under_law_a`: seed 200/200,
   support 250/250, refutation 10/230, cut 0. Law A: `LoadBearing {210 → 64}`.
   Tables `[B1, B2, B4, B8, B16]` → load-bearing `[false, false, false, false,
   true]`. The mirror image of the old fixture (load-bearing under B1 only).
   Disable run: law A replaced by B1 → red, B1 reads `TruthOnly {118 → 105}`;
   with B1 even the supported factual arm falls below the bar.
2. **`DEFAULT_FREQUENCY_BAR` rationale** no longer cites B1's constant 170 as
   evidence; the semantic reason (consistency is strength, not amount of
   evidence) stands alone. `the_bar_is_not_inert` keeps pinning 170, labelled
   as B1's constant.

## Still open

- forward's discarded truth (compute waste) and the opcode/truth label
  mismatch on replayed edges — recorded, unchanged.
- G1.

```
PR (this) | STATUS: done | OUTCOME: P8 re-checked in code (no production
consumer; replay overwrites forward's truth); G2 recorded as law A for new
decision-bearing wiring, defaults untouched; law-A load-bearing fixture
pinned; frequency-bar rationale corrected | OPEN: G1; forward truth discard
and opcode label mismatch
```
