# 2026-10-07 — D-CE64-CYCLE-0: what a CE64 register must carry across cycles, and what self-orchestration buys at scale

**Status:** TEST-PINNED (`crates/lance-graph-planner/examples/staunen_scheduler_probe.rs`, `test = true`, 23 tests run by `member-tests`; comparative tests pool 3 seeds). MEASURED (`cargo run --release -p lance-graph-planner --example staunen_scheduler_probe`, 4-core Xeon 2.8 GHz). No production code changed.

## DECISION

- **SCOPE:** the D-CE64-STAUNEN-0 executor extended rather than duplicated: `run_with(opts)`, a lazy-stamp heap scheduler, a FIFO worklist baseline (reacts to new evidence, reads no entropy or activation), a wall-clock budget per window with continuation, `Forget` arms at window boundaries, the #1390 Simpson population plus its positive trial, end-attractor and fold-quadrant readings.
- **The register is treated as the cross-cycle carrier; which state it must carry is measured by forgetting each piece at every window boundary** (single seed, 2000 windows, budget 3):

| forgotten | resolved | same run? | verdict |
|---|---|---|---|
| nothing | 0.955 | — | — |
| posterior (H) | 0.955 | identical | derived from the evidence table: persisting H would be an echo |
| bits 40..42 (question) | 0.955 | identical | an echo of the opcode in this loop |
| register (Epi5, mask) + what ran at which code | 0.024 | no | earned Epi5 is re-derived every window and starves the real work |
| evidence table (latest outcome per channel) | 0.000 | no | needed |
| attention state (habit, progress, coherence, surprise) | 0.378 | no | needed; this is how earlier windows direct later ones |

- **W:** belief moved on 4648 folds while Epi5 moved on 72; `pearl::revise` changed F/C on 0. A W that advances only on Epi5 changes would reference about 1.5% of belief updates. Nothing here mints a W.
- **BASIS:** all weights, likelihoods and thresholds are policy pins.

## Measured: self-orchestration against baselines (pooled 3 seeds, 600 windows)

| policy | folds | resolved | wake | folds on settled basins | resolved basin-windows per fold |
|---|---|---|---|---|---|
| full | 4806 | 0.865 | 7.9 | 0.047 | 10.36 |
| FIFO worklist | 5400 | 0.909 | 8.3 | 0.094 | 9.69 |
| round-robin | 5400 | **0.911** | **7.5** | 0.093 | 9.72 |
| random | 5400 | 0.838 | 12.4 | 0.050 | 8.94 |
| no entropy and no activation | 5400 | 0.486 | 11.4 | 0.018 | 5.18 |

- **Correction to D-CE64-STAUNEN-0 / #1395:** its "wake 4.2 vs round-robin 5.1" was one seed. Pooled, the full scheduler has **no wake advantage** (7.9 vs 7.5). Exact tie-breaking alone had moved a single-seed wake latency by half.
- What survives pooling:
  - ablations each matter (no entropy 0.515, no surprise wake 43.1, no habituation 0.712, no progress 0.822, naive 0.629);
  - activation drives sleeping (without it, 5400 folds instead of 4806);
  - the full scheduler spends 11% fewer folds and wastes half as many on settled basins, at a lower resolved fraction than the two simple baselines.

## Measured: dense windows, 350 ms cap

| folds per window | policy | ns/fold | folds that fit in 350 ms | productive share |
|---|---|---|---|---|
| 1M (327,680 basins) | FIFO / round-robin | ~220 | all 1M (~252 ms per window) | 0.72 |
| 1M (327,680 basins) | full (heap) | ~800 | ~440k | 0.89 |
| 1k–100k | FIFO / round-robin | ~210 | — | ~1.00 |
| 1k–100k | full (heap) | 490–710 | — | 0.88 |

- At 1M, FIFO yields about twice as many productive folds per second as the heap-driven scheduler.
- **Within a window, later folds are less productive** (FIFO: first tenth 0.953, last tenth 0.483). Informative work is front-loaded and runs out.
- **"Does fold 1,000,000 get better directed by folds 1..999,999?"** Inside a window, no. Across windows, the carried attention state is what directs later windows (forgetting it: 0.955 → 0.378 resolved).

## Other readings

- Fold quadrants (activation sign × ΔH, full): falsify 1794, anomaly 1935, leave 419, dig 233, other 758.
- Polarity: hydration reacts negative / zero / positive (2233 / 561 / 2168). The SP check is negative only on Simpson basins (pinned with a silence twin).
- End attractors (learned, wrong-settled, exhausted, waiting, sterile, active):
  - full: [31, 0, 0, 0, 7, 2]
  - naive: [13, 13, 0, 4, 7, 3]. Naive scheduling ends **wrongly settled** as often as learned: an attractor is not understanding, and the reading distinguishes them.
- **The sterile class is weak:** on the 600-window shape only 3 of 8 noise basins meet it.

## Gates

19 disable runs, each red on its named falsifier: 13 from D-CE64-STAUNEN-0, re-run after the refactor, and 6 new:
- the forgotten belief not rebuilt
- forgetting the register as a no-op
- the W counter dropped
- Simpson replaced by a plain trial
- a deadline that never binds
- no sterile class

## OPEN

- Within-window direction does not improve; front-loading is the dominant dynamic.
- The heap scheduler costs ~3.5× FIFO per fold. A cheaper priority structure (bucketed priorities, Moore-local queues) is untested.
- Moore, EWA, Sudoku and Quack folds are not in the loop (they live outside the planner). Substrate-agnosticism is therefore claimed for nothing beyond Pearl folds here.
- The register itself carries none of the needed cross-cycle state except Epi5:
  - The evidence table, which drives this loop, lives beside the register; it is 4 channels × 2 bits.
  - Whether that state, attention, and a reference to belief updates should have a home in the register remains open; bits 50..52 conflict with `learn` (D-CE64-STAUNEN-0).
- No counterfactual (−6) fold in this loop; isolation is pinned in D-CE64-LOOP-0 and the planner's Pearl tests.
