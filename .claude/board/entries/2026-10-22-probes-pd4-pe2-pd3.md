# 2026-10-22: probes P-D4, P-E2, P-D3 — measured

**Status:** MEASURED (lab probes; no production code changed). These are the
top three probes pre-registered in `2026-10-22-coresearch-pal256-evidence-probes.md`.

## P-E2: count each source once (`crates/causal-edge/tests/p_e2_stamp_gate_probe.rs`)

Production revision (`Opcode::Revision.truth_code`) double counts a
duplicated source:

| duplicated k | c (start 128) | posterior sd ratio |
|---|---|---|
| 2 | 170 | 0.837 |
| 4 | 204 | 0.658 |
| 8 | 227 | 0.491 |

A test-local gate in the ONA shape fixes this. It checks for stamp overlap before revising, and on overlap it keeps the stronger belief. Results:
- Duplicates give Δc = 0 and an sd ratio of 1.000 for k = 1..8 (pass band 1.00 ± 0.05).
- k disjoint sources give exactly the ungated codes (c 170/204/227).
- A shared premise reached by two paths is not pooled.

Disable runs:
- Without the overlap check, the duplicate and shared-premise tests fail.
- With the gate refusing every revision, the disjoint and shared-premise tests fail.

Still open: where the stamp lives (G5).

## P-D3: out of support by distance, not readout (`crates/bgz-tensor/tests/p_d3_ood_support_probe.rs`)

Synthetic setup: 16 clusters in Base17 space, a production `WeightPalette` with 16 entries, and the production `FamilyGamma` readout.

| far-field set | distance AUROC | readout-magnitude AUROC |
|---|---|---|
| ×4 scaled | 1.000 | **0.500** |
| random direction | 1.000 | 0.885 |

The readout is blind to scaling, as Hein et al. Thm 3.1 predicts. The radius flag `distortion > radii[index]` fires on 2.0% of the in-support hold-out and on 100% of the ×4 set. Fisher-z encode clamps silently: the clamp condition counts 0/120 in-range cosines and 403/403 saturating ones, and `encode(1.0) == encode(0.99995)`.

Disable runs:
- Using magnitude in place of distance fails the AUROC test.
- A flag that is always on fails the false-positive rate (FPR) bound.
- A flag at 100× the radius fails recall.

## P-D4: the answer/abstain gate (`crates/causal-edge/tests/p_d4_decision_gate_probe.rs`)

The world is synthetic and was fixed before the first run:
- 40 blocks × 25 questions × 4 candidates.
- Each block shares its sources: 26 reliable, 10 weak, 4 out of support.
- 30% of questions are echoed.
- All intervals come from a block bootstrap.

Results:

1. **The pre-registered margin (Jeffreys mean ± 2 sd) is vacuous: it answers 0/1000.** A no-evidence candidate has sd 0.35, so the rule cannot clear with 3–9 observations. Killed. Pinned by `preregistered_interval_margin_is_vacuous`.
2. **Post-hoc gate** (declared after run 1, before any other run). It answers when three conditions all hold:
   - the top candidate has ≥ 2 more distinct sources than the runner-up;
   - the top candidate has ≥ 2 sources;
   - the support flag is clear.

   Brier(always-answer) − Brier(gate) = **0.0242 [0.0079, 0.0409]** at 44% coverage. This clears the 0.02 bar, but it is post-hoc, so it is weaker evidence than a pre-registered pass.
3. **Label-blind control.** The pre-registered shuffled-label Brier control cannot stay silent: replacing p ≠ π with π lowers the Brier score on any labels. It is replaced by within-block selective accuracy:

   | labels | answered − abstained accuracy |
   |---|---|
   | real | +0.120 [0.058, 0.185] |
   | shuffled within blocks | −0.032 [−0.090, 0.034] |

   The pooled version of this gap is confounded, because whole out-of-support blocks are abstained (Simpson's paradox: shuffled 0.249 [0.159, 0.333]). It is reported only.
4. **Ablations.** Only the margin carries the gain in this world:

   | removed | Brier gain | pass? |
   |---|---|---|
   | margin | 0.0182 | fails |
   | stamp gate | 0.0225 | passes |
   | support flag | 0.0234 | passes |

   The margin already counts distinct sources, and it already abstains on chance-level blocks, so this world cannot separate the three conditions. A world that does would need echoes that create a false margin, and out-of-support inputs that carry a confident margin.

## Open

- Re-run P-D4 in a world where echoes and out-of-support inputs produce confident false margins. That world must be fixed before it is run.
- G5: the stamp's home.
- The calibration claim needs real held-out labelled outcomes, not a synthetic world.
- CI runs `bgz-tensor` with `--lib` only, so P-D3 runs locally, not in CI.

Closeout: `STATUS: measured | OUTCOME: P-E2 and P-D3 pass as registered; P-D4's registered margin is vacuous and its post-hoc gate passes on the margin alone | OPEN: separating world for P-D4, G5, real labels`
