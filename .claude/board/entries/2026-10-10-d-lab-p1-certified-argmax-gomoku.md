# D-LAB-P1 — certified argmax on Gomoku: a deterministic certificate (2026-10-10)

**Status:** MEASURED. Probe only
(`crates/cognitive-shader-driver/examples/moore_nars_gomoku_probe.rs`,
`minimax_certified` + `certified_minimax_picks_what_minimax_picks_with_fewer_folds`).
Stockfish O1 untouched. `Op::Asc` and `Opp::Lookahead` still call `minimax`.

## Certificate (kind: deterministic)

A candidate's value is the minimum over its replies, so the running minimum
is an upper bound U that only falls. The best finished candidate has an exact
value L. When U ≤ L, the candidate cannot be chosen (the choice needs a strict
`>`), and its remaining replies are skipped. No value exceeds 7, so a finished
7 ends the search. This is the alpha-beta cutoff at the root, stated as a
certificate: `L_best ≥ U_j` ⇒ j is excluded. It holds by construction, not by
measurement; the corpus test checks the implementation.

#1428's residual bound was not reused: it is proven for the linear flow model
only.

## Measured (400 reasoning positions; propagation-decided positions excluded)

| k1/k2 | folds full | folds certified | saved | positions with a cut | pick changed |
|---|---|---|---|---|---|
| 4/4 | 442,720 | 282,480 | 36.2 % | 400/400 | 0 |
| 5/5 | 665,076 | 367,344 | 44.8 % | 400/400 | 0 |
| 8/8 | 1,601,258 | 624,386 | 61.0 % | 400/400 | 0 |

Savings grow with branching, as expected for cutoffs. Overhead: one
comparison per evaluated reply. Abstentions: none (an exact search never
abstains; it only skips).

CI: the probe is `test = true` and runs under rust-test.yml's shader-driver
step. Disable run: cutting at `U ≤ L + 1` (unsound by one) → the test fails on
k = 4/4 with a changed pick. Pinned fold counts are exact (deterministic
corpus).

## For the lab

This is strategy S7 (certified early stopping) on one task, with a
decision-identical guarantee. It answers H4 on Gomoku minimax: computation
drops 36–61 % with zero changed decisions. It does not touch D-MOORE-NARS-0's
"less work not achieved" by itself: that probe's recipes would need to call
`minimax_certified`, which would change its reported folds but not its games.
That swap is the first lab arm to measure, not an edit made here.

```
PR (this) | STATUS: measured | OUTCOME: deterministic certified argmax on
Gomoku minimax: identical picks on 400 positions, 36-61% fewer folds; unsound
cut changes a pick | OPEN: recipes still call minimax by default; no
certificate yet for statistical (sampled) values
```
