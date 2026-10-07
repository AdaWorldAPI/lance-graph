# 2026-10-07 — D-CE64-STAUNEN-0: raw surprise is captured by noise; habituated surprise wakes settled basins

**Status:** TEST-PINNED (`crates/lance-graph-planner/examples/staunen_scheduler_probe.rs`, `test = true`, 18 tests, run by `member-tests`). MEASURED for the numbers below (`cargo run --release -p lance-graph-planner --example staunen_scheduler_probe`; 32 real + 8 noise basins, 3 folds/window, 2000 windows, a mediator switch in each real basin about 1/300 windows).

## DECISION

- **SCOPE:** a probe. Many basins, each one `CausalEdge64` register plus a declared hypothesis set, carried across decision windows with a fold budget per window. Operators are production `pearl::hydrate` and `pearl::reason` + `revise`.
- **Surprisal** `-log2 p(outcome)` (under the basin's predictive) is distinct from **Shannon H** over its posterior; `log2(popcount)` is only H's uniform case (pinned).
- The world's truth sits in an `Oracle` the scheduler never receives. Regimes are classified afterwards, from per-window aggregates.
- **Bits 50..52 are NOT written.** They are plasticity, and `CausalEdge64::learn` gates on them: a 3-bit surprisal there would decide whether `learn` adopts an archetype (pinned). The carried surprisal is a transient field.
- **BASIS:** likelihood tables, habit and progress EWMA rates, `PROGRESS_PRIOR`, `PROGRESS_RELAX` and the sleep threshold 0.05 are **policy pins**.

## Measured

| policy | folds | resolved | wake (windows) | noise share |
|---|---|---|---|---|
| **naive** (raw H + raw surprisal) | 6000 | **0.649** | 36.3 | 0.918 |
| full (habituated surprise + learning progress + activation inhibition + sleep 0.05) | 5170 | 0.952 | 4.2 | 0.832 |
| no habituation | 6000 | 0.792 | 28.4 | 0.890 |
| no learning progress | 6000 | 0.803 | 23.9 | 0.904 |
| no surprise | 2644 | 0.818 | 39.9 | 0.747 |
| no entropy | 2413 | 0.834 | 2.4 | 0.652 |
| no activation | 5998 | 0.958 | 3.0 | 0.855 |
| no entropy, no activation | 6000 | 0.826 | 4.9 | 0.867 |
| round-robin | 6000 | **0.964** | 5.1 | 0.880 |
| seeded random | 6000 | 0.942 | 7.1 | 0.874 |
| full, 3-bit binary carrier | 6000 | 0.929 | 10.3 | 0.866 |
| full, 3-bit thermometer carrier | 2575 | 0.807 | 39.4 | 0.747 |

- **The naive reading of "surprise creates work" is captured by noise** (the noisy-TV failure of curiosity-driven search). Before the two fixes the full scheduler lost to round-robin under scarcity (0.649 vs 0.964), and its noise share grew to 98% by the last epoch.
- **Habituation** (only surprisal above the basin's own habitual level interrupts) and **learning progress** (expected gain weighted by entropy reduction that persisted a window later) each carry the fix.
- **Surprise is what wakes a settled basin:** without it, re-resolution after a switch takes 39.9 windows instead of 4.2.
- **Activation drives inhibition:** without it the scheduler never sleeps.
- **Without entropy and activation** it is worse than seeded random.
- **Round-robin still resolves slightly more** (0.964 vs 0.952). The full scheduler wins on wake latency, folds spent, folds on settled basins (0.014 vs 0.028) and resolved basin-windows per fold (11.8 vs 10.2).
- **Energy vs quality:** sleep 0.1 spends 1995 folds for 0.829 resolved; 0.2 spends 732 for 0.656. Big savings cost resolution here.
- **Does fold N get better directed because of folds 1..N−1?** **Not measured beyond the first epoch.** The useful-fold fraction is flat (0.43–0.45) from epoch 1 to 9. History helps through per-basin habit and progress, not through a trend over the run.
- **Long run** (62,500 windows, 157,658 folds): resolved 0.978, stable; ~4.4 µs wall time per fold, including all window events.
- A **3-bit carrier** costs wake latency (binary ×2.5). The thermometer, the reading that would keep plasticity's hot-count monotone, nearly loses the interrupt.
- **Two defects fixed during the build:**
  - Re-running an unchanged hydration after an unrelated seal bump counted the same evidence again (pseudo-replication), underflowing posteriors so basins could not follow a switch. The belief is now built from the latest outcome per channel.
  - Without relaxation of unsampled progress the learned gate was absorbing (resolution decayed over long runs).

## Gates

13 disable runs, each red on its named falsifier:
- habituation off
- progress gate off
- surprise term off
- activation inhibition off
- entropy replaced by a constant
- a contradiction given H = 0
- surprisal replaced by log2(popcount)
- the witness overwritten by a counter
- multiply instead of replace
- no cancellation
- a wall-clock seed
- the Q3 carrier at full precision
- no sleep

## OPEN

- One world model, one seed. The ranking against round-robin is close and may not survive another shape.
- Moore, EWA and Sudoku folds are not used: they live outside the planner (`cognitive-shader-driver`, `jc`, `perturbation-sim`).
- `planner::nars::basin_resonance::staunen` (`2·|f−0.5|·c`, truth stakes) is a different quantity from surprisal. The name collision is unresolved.
- No home for a cross-cycle surprisal in the register is established. Bits 50..52 conflict with `learn`, and 3 bits measurably lose information.
