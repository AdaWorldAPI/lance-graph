# D-SELF-CALIBRATING-LAB-1 — Thompson sampling and context, a second pre-registered family (2026-10-10)

**Status:** MEASURED. Probe only (`moore_nars_gomoku_probe --lab-v1`).
**Builds on:** `entries/2026-10-10-d-self-calibrating-lab-0.md` (same
statistics, same horizon: 300 games, endpoint = last 100-game window).
Fresh seeds 4001–4016; a separate Holm family (α = 0.05); 16.4 s wall.

## New arms

- `Chooser::Thompson`: per recipe, a draw from Beta(pos + ½, neg + ½), the
  Jeffreys posterior D-RPF-P5 identified, minus the same fold-cost term as
  the argmax chooser. Sampler: `lab::beta_sample` (Gamma via
  Marsaglia–Tsang, boost for shape < 1), moment-tested on Beta(½, ½),
  Beta(2.5, 7.5), Beta(30.5, 10.5).
- `Memory::Global`: every situation shares one key (context-free baseline).

## Ledger (statistical)

| id | hypothesis | effect (+ = treatment better) | 95 % CI | p | Holm | verdict |
|---|---|---|---|---|---|---|
| T1 | Thompson beats expectation argmax, score | −0.0500 | [−0.0620, −0.0380] | <0.0001 | reject | **FALSIFIED** |
| T2 | Thompson adapts faster after Threat → Aggressor | −0.0153 | [−0.0244, −0.0062] | 0.0040 | reject | **FALSIFIED** |
| T3 | structural context beats one global key, score | +0.0247 | [+0.0050, +0.0444] | 0.0186 | reject | **SUPPORTED** |
| T4 | Thompson reduces folds/step | −18.5 (Thompson costs more) | [−38.4, +1.4] | 0.0596 | keep | **FALSIFIED** (the interval cannot reach the 5-fold minimum) |

Pareto (empirical): argmax e 0.5062 score / 244.8 folds per step; Thompson
0.4562 / 263.3; global 0.4816 / 32.3.

## Reading

- Thompson is worse here in the learned window and after a switch. A likely
  mechanism, not tested: evidence arrives in fractional units (activation/7 ≤ 1
  per update) spread over many structural keys, so posteriors stay wide and
  sampling keeps exploring. Testing it needs an evidence-scale arm, not a
  re-reading of this run.
- Context matters (T3). The global key collapses onto cheap recipes (32
  folds/step) and loses 0.025 score: the structural signature buys quality
  at about 8× the folds.
- Expectation argmax stays the chooser.

Falsifiers: `beta_sample_has_the_beta_moments`,
`thompson_is_seeded_and_differs_from_argmax`; replacing the sampler with the
posterior mean turns both red.

```
PR (this) | STATUS: measured | OUTCOME: Thompson falsified on score (-0.050)
and adaptation (-0.015); structural context supported (+0.025 over a global
key) | OPEN: evidence-scale arm for Thompson; eligibility traces;
cost-aware selector; held-out domain
```
