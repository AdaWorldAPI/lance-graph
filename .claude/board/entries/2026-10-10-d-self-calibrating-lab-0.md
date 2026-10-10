# D-SELF-CALIBRATING-LAB-0 — the hypothesis lab, v0: shared statistics + Gomoku and crossword arms (2026-10-10)

**Status:** MEASURED (one task family). Probe only; no production code.
**Code:** `crates/cognitive-shader-driver/examples/shared/lab.rs` (task-agnostic
statistics) and `moore_nars_gomoku_probe.rs --lab` (the Gomoku arm). Tests run
in CI (`test = true`, rust-test.yml shader-driver step).
**Predecessors:** D-MOORE-NARS-0 (the learner), D-LAB-P1 (certified minimax),
D-RPF-P5 (what the expectation means).

## What v0 is

A task probe runs each strategy arm on matched blocks (seeds) and hands
`lab.rs` one outcome per (arm, block). `lab.rs` provides:

| piece | certificate kind | note |
|---|---|---|
| paired comparison: per-block difference, exact sign-flip p (exhaustive ≤ 20 blocks), 95 % t-interval | statistical | t-interval assumes roughly normal block differences; the p-value assumes nothing beyond exchangeability of signs |
| Holm over a pre-registered family | statistical | α = 0.05 |
| verdict per pre-registration | statistical | SUPPORTED needs Holm rejection, the declared direction and ≥ the declared minimum effect; FALSIFIED needs the interval to rule the minimum out; otherwise INCONCLUSIVE; a wrong block count is never SUPPORTED |
| reliability bins, ECE, Brier | empirical | |
| deterministic checks (e.g. identical games) | deterministic | reported separately, never folded into a p-value |

Experiment record = (task, context, strategy arm, hypothesis, block seed,
outcome, cost). Provenance is the seed set, the arm's `Cfg`, and the commit.
Not yet a SoA record; v0 keeps it in the probe (see OPEN).

Falsifiers: Holm's step-down stop and the min-effect gate are each
disable-verified red (`holm_matches_the_textbook_step_down`,
`a_planted_effect_is_found_and_a_null_is_not`). The first version of the
min-effect test was vacuous (the gate's removal stayed green); a significant
tiny effect was added and the disable then failed.

## Pilot → amendments → confirmation

The pilot (seeds 1001–1012, 100 games, whole-run score) was exploratory.
It exposed three design errors, each recorded in code before the
confirmatory run:

1. **H4's identical-games check failed** — not the certificate: the chooser
   has a fold-cost term, so a cheaper ASC changes which recipe it picks.
   H4 is now measured at cost weight 0, and the interaction is pinned
   (`the_certificate_changes_no_game_for_a_cost_blind_learner`: identical at
   cost 0, not identical at 0.02).
2. **100 games measured the learning phase.** Endpoint is now the last
   100-game window of 300.
3. **Calibration needs the right population** (see below).

Confirmation: 16 fresh blocks (seeds 2001–2016), 9×9 five vs Threat, oracle
off, 28.6 s wall.

## Hypothesis ledger (Gomoku, confirmatory)

| id | hypothesis | effect (treatment − baseline, + = better) | 95 % CI | p | Holm | verdict |
|---|---|---|---|---|---|---|
| H2 | learned chooser beats best fixed recipe (HPM), score | +0.0238 | [+0.0060, +0.0415] | 0.0156 | reject | **SUPPORTED** |
| H4 | certified minimax cuts folds/step (cost 0) | +28.9 | [+20.3, +37.6] | <0.0001 | reject | **SUPPORTED** (+ deterministic: identical games on all 16 blocks) |
| H7 | fold-cost term cuts folds/step | +142.7 | [+118.3, +167.1] | <0.0001 | reject | **SUPPORTED** |
| H3 | delayed (one-reply) credit beats immediate, score | +0.0225 | [+0.0036, +0.0414] | 0.0267 | keep | INCONCLUSIVE |
| H5 | decay 0.95 adapts faster after Threat → Aggressor, post-switch score | +0.0072 | [−0.0051, +0.0195] | 0.2322 | keep | **FALSIFIED** (CI rules out the 0.02 minimum) |

H2 replicates D-MOORE-NARS-0's central claim on fresh seeds. H3 points the
same way as D-MOORE-NARS-0 but does not survive the family correction at
16 blocks.

## Pareto (mean over blocks; empirical)

| arm | score | folds/step |
|---|---|---|
| cost 0 + certified | 0.5331 | 368.6 |
| cost 0 | 0.5331 | 397.6 |
| decay 0.95 | 0.5159 | 231.9 |
| base (cost 0.02) | 0.5069 | 254.8 |
| immediate credit | 0.4844 | 0.0 |
| fixed HPM | 0.4831 | 0.0 |

`cost 0` is dominated by `cost 0 + certified` (same score, fewer folds).
The cost term is a trade-off, not a free win: −0.026 score for −143
folds/step. Immediate credit collapses onto HPM (zero-fold recipes).

## Calibration — what the NARS expectation predicts

The chooser's expectation e of the chosen recipe vs whether the step was
productive (retained activation > 0), base arm:

- over **all** outcomes: ECE 0.317 (e overstates);
- over **non-neutral** outcomes (activation ≠ 0): ECE 0.050.

A zero activation adds no evidence, so e estimates P(positive | non-neutral),
and it is well calibrated for that. Reading it as P(productive step) is the
error. Same lesson as D-RPF-P5: e is a posterior mean of the evidence the
substrate actually records, nothing more.

## Crossword arm (`crossword_crossing_care_probe lab`)

Same `lab.rs`, second task family, separate Holm family. Blocks = 16 fresh
puzzles (seed 0x1AB5EED; the original used the 0xC0FFEE sets), side 5, open
workload (half givens, cap 50). Endpoints in natural logs (min effect 0.1 ≈
10 %); ops is the deterministic cost counter, not wall time. 3.4 s wall.

Deterministic half: every arm sound on every puzzle; same fill count wherever
two arms both solved; 16/16 solved by each arm.

| id | hypothesis | effect (ln, + = treatment better) | 95 % CI | Holm | verdict |
|---|---|---|---|---|---|
| C1 | Support cuts search nodes vs Baseline | +3.59 (≈36×) | [+3.00, +4.17] | reject | **SUPPORTED** |
| C2 | Support cuts total ops vs Baseline | −2.57 (≈13× more ops) | [−2.90, −2.24] | reject | **FALSIFIED** |
| C3 | Single counterfactuals cut ops vs Baseline | −1.07 (≈2.9× more) | [−1.71, −0.43] | reject | **FALSIFIED** |
| C4 | Pair nogoods cut ops beyond Single | −0.78 (≈2.2× more) | [−1.13, −0.43] | reject | **FALSIFIED** |

Geometric means: Baseline 2,427 nodes / 4,718 ops; Support 67 / 61,700;
Single 471 / 13,717; Pair 434 / 29,895. Replicates D-SCF-CARE-PAIR-0 on
fresh puzzles. One difference worth noting: here Single needs 5× fewer nodes
than Baseline, while the original's open set showed similar node counts;
the original attributed that to enumeration order under the cap.

## Across the two tasks (empirical, not yet a tested meta-hypothesis)

On both tasks the strategy that wins on a work-avoidance proxy (nodes;
learned value) is not automatically the cheapest end to end, and the lab
separates the two only because cost is a declared endpoint. On Gomoku the
deterministic certificate is the one change that cuts cost with zero quality
change; on crossword no tested arm beats Baseline on ops. A cost-aware
strategy selector is therefore the next lab question, not a conclusion.

## OPEN

- Held-out evaluation (Moore #1428) and the cross-task meta-experiment need
  both tasks in one process: the record should move to a SoA-shaped carrier
  then, not before.
- Strategy arms not yet present: Thompson sampling, eligibility traces over W,
  context-conditioned selection, cost-aware experiment selection.
- H3 needs more blocks to decide; that is a pre-registered follow-up, not a
  re-reading of this run.

```
PR (this) | STATUS: measured | OUTCOME: lab v0 (paired sign-flip + Holm +
verdicts + reliability) with a Gomoku arm; pilot exposed 3 design errors,
fixed before a fresh-seed confirmation: H2 H4 H7 supported, H3 inconclusive,
H5 falsified; e calibrated (ECE 0.05) only over non-neutral outcomes |
OPEN: held-out domain, SoA record, Thompson/eligibility arms, cost-aware
selector; crossword arm: C1 supported, C2-C4 falsified (fresh puzzles)
```

MIRROR
- BIAS CHECK: the H4 pilot "failure" looked like a certificate bug; the
  disable-verified P1 test already proved the certificate, so the divergence
  had to come from elsewhere, and the cost term was it.
- BLIND SPOT: one opponent family (Threat, Aggressor), one board size.
