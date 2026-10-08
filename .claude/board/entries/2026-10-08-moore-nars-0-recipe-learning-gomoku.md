# 2026-10-08 — D-MOORE-NARS-0: on Gomoku, reasoning consequences teach recipe choice, through structure, not boards

**Status:** TEST-PINNED (`crates/cognitive-shader-driver/examples/moore_nars_gomoku_probe.rs`, `test = true`, 7 tests: 6 structural + 1 learning falsifier, ~35 s debug). MEASURED (`cargo run --release -p cognitive-shader-driver --example moore_nars_gomoku_probe`, 427 s; 9×9 five-in-a-row, seeded openings, 3 seeds pooled). Re-measured after the review fixes (horizon in learner turns, frozen runs leave fold costs alone, immediate credit keeps its own reading at game end): only the h=3 arm moved, 0.495 → 0.496.

## SCOPE

The world is a coupled Moore field: Gomoku on 9×9. The loop for one learner move:

1. Hydrate the empty cells that have a stone among their 8 neighbours.
2. Read every line in the 4 Moore directions, for both sides.
3. Propagate mechanically: own win, else forced block.
4. Only if 2 or more candidates remain, the chooser picks a recipe from a structural signature.
5. The recipe produces a move.
6. W names the pending credit slot.
7. At the learner's next turn, the board yields a signed activation.
8. That activation credits `(signature, recipe)` as NARS evidence.

What is learned is `situation → recipe`, never `position → move`. The signature is relative (mine and theirs) and coordinate-free: the best line class on each side, the opponent's open-three count, whether attack and block conflict, and the population size.

The opponents are fixed and deterministic:
- **Threat**: a line-pattern heuristic;
- **Lookahead**: 2-ply search;
- **Aggressor**: ignores blocking unless forced.

## Census of the 34 catalogue recipes (read from `contract::recipes::RECIPES`)

| standing | count | recipes |
|---|---|---|
| realized here | 9 | RCR, TCP, TR, ASC, CR, TCF, HPM, CUR, ICR |
| chooser-level (is the learner, not an operator) | 6 | MCP, CDT, PSO, CWS, AMP, DTMF |
| redundant with a realized operator | 6 | RTE, SSR (ASC), SMAD, MPC, SPP (TCF), CAS (CUR) |
| unreachable (substrate absent: VSA, clusters, personas) | 8 | IRS, MCT, LSI, ARE, ZCF, IDR, SDD, HKF |
| meaningful, not wired | 4 | HTD, TCA, ETD; SSAM deliberately (it would learn moves) |
| unsafe | 1 | CDI (HOLD = pass; Gomoku has no pass) |

The 9 realized recipes take their catalogue IDEA onto Moore lines. The catalogue's own substrate strings (CLAM, InnerCouncil, VSA) are not present here.

## Measured

Each figure is a score: win = 1, draw = 0.5. Unless a window or seeds are named, it is from the last window of 400 games, pooled over 3 seeds.

**Learning versus its controls, against Threat:**

| arm | first → last window | notes |
|---|---|---|
| best fixed recipe (CUR / HPM) | 0.498 / 0.493 | 0 extra folds |
| uniform recipe | ~0.24 | |
| **persistent structural learner** | **0.500 → 0.535** | per seed, last window: 0.509 / 0.539 / 0.559 |
| reset every game | ~0.47 | flat |
| literal board memory | ~0.49 | memory hits 0.3–3.5% |
| immediate credit (before the opponent answers) | ~0.50 | stays on HPM 98–99% |
| credit 3 turns later (W h=3) | 0.496 | horizon counted in learner turns (review fix) |
| contradiction only | ~0.47 | |
| final win/loss only | 0.46 → 0.49 | |
| cost-blind | 0.534 | 448 folds/step vs 290 |

**Transfer, frozen learned state:**
- unseen openings: 0.526 vs 0.493 untrained;
- 11×11: 0.507 vs 0.493.

**No collapse onto one recipe.** Over 169 signatures, the favourite recipes are HPM 57, TCF 29, CR 19, RCR 18, ICR 17, ASC 16, TCP 11, CUR 2.

**Opponent switch, Threat → Lookahead:**
- **Pretrained learner:** the first window starts at 0.626, above a fresh learner's 0.591; its later windows reach 0.65–0.69.
- **Both beat the fixed recipes:** fixed HPM scores 0.570 and fixed ICR 0.455.
- **Decay 0.98 sheds ICR** (0.22 → 0.10) and cuts folds, but its score drifts down from 0.677 to 0.593.
- **The Aggressor switch is uninformative:** Aggressor is weak, and every arm scores about 0.93–0.96.

## What this answers, in the CC's terms

- **Persistent learned state helps.** The learner plays better and chooses recipes better than reset-every-game. The CC's critical falsifier does not fire.
- **Structure, not coordinates, carries the gain.** Literal-board memory never recurs, so it learns nothing.
- **The credit spine is the W-delayed consequence after the opponent's reply.** Immediate credit learns nothing (it stays on the cheapest recipe). So does credit stretched over 3 turns: the intervening moves dilute it.
- **Contradiction-only and final-result-only signals are too sparse** at this scale.
- **"Less work" is NOT achieved.** The learner plays better by spending more:
  - about 280 folds per reasoning step against 0 for fixed HPM, and the figure rises over training;
  - the cost term only trims it (448 → 290 at the same score).
- **Unlearning after a switch is not demonstrated.** No preference trapped the learner: pretrained ≥ fresh throughout. But no arm shows a clean recover-after-drop curve, and decay is not better.

## Caveats

- **The regret/optimality column is not independent evidence.** It scores each step by the one-reply consequence against the deterministic opponent, which is also the training signal. The game score is the external yardstick.
- **ICR is privileged against Threat.** It simulates exactly the reply Threat will make. Against Lookahead it is worse than HPM, yet the learner keeps a ~27% ICR share. The situation mix changes after the switch, so a changed recipe distribution is not by itself unlearning.
- **Most games are draws** between near-equal heuristics on 9×9. The score difference is carried by the decisive minority.

## Gates

- 5 structural falsifiers, each red when its guarded code is disabled:
  - the census covers the catalogue exactly;
  - propagation takes the win before the block;
  - the register never names the recipe;
  - W follows the pending slots;
  - replay is deterministic.
- 1 learning falsifier with 4 disable runs, each red at its own assertion:
  - credit off;
  - reset off;
  - literal memory given the structural key;
  - immediate credit given the delayed consequence.

## OPEN

- **Spending less work.** Make the cost term or the signal reward cheap sufficiency, without giving back the score.
- **A real unlearning test.** It needs an opponent switch under which a previously dominant recipe becomes actively bad in the same situations.
- **Per-step attribution over longer horizons.** For example, eligibility traces through the W chain, instead of the undivided h=3 credit.
- **Stronger opponents.** Lookahead is not stronger than HPM here.
- **Larger boards and Renju constraints**, then Go local fights.
