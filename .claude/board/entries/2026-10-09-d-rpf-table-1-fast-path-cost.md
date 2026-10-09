# 2026-10-09 — D-RPF-TABLE-1: what did the confidence-inert fast path buy?

**Status:** MEASURED. Tests, one bench example, comment corrections and two
isolated guards (T2, T4). No production semantics changed; no G1 or G2
policy chosen.
**Harness:** `crates/lance-graph-planner/tests/d_rpf_table_1.rs` (numeric,
replay drift, decisions; 15 tests) and
`crates/lance-graph-planner/examples/d_rpf_table_1_bench.rs` (throughput).
**Parent:** `2026-10-09-d-rpf-conf-0-confidence-surface.md` (#1424). Its
T1/T3/R1/G4/R3 results are taken as given and not re-measured.

Laws: **A** = `CausalEdge64::revision` (confidence-aware). **Bn** =
`NarsTables::build(n).revise`, n ∈ {1, 2, 4, 8, 16}. All through the real
implementations. A's replay uses `step_with` (= `replay_step` with the law as
a parameter) and A's decisions use `decide` (= the `counterfactual_replay` +
`Reaction::classify` rule); both are pinned bit-exact against the real
functions for every table, so A-vs-Bn differences come only from the law.

## Where B1 actually runs (correcting #1424)

#1424 said "the replay/counterfactual path emits B1". Read in code:

- `replay_chain` / `counterfactual_replay` / `pearl::reason` take the tables
  from the caller. **No production code constructs a `CutContext`.** Every
  caller is a test, an example or a probe, and all of them pass `build(1)`
  except `tests/chain_confidence.rs` (16).
- `NarsEngine::new` allocates `build(1)` (256 KiB per engine; production
  instance in `strategy/chat_bundle.rs:42`), but **no production code reads
  `NarsEngine::tables`**: `revise_fast` / `deduce_fast` have only test
  callers, and `on_response` revises with `truth_revision` (law C).
- `cognitive-shader-driver` does a `tables.revise` per hit, but **discards
  the result** (`_revised_truth`, "observed only"), and nothing calls
  `with_nars_tables` outside tests.

So today B1 changes **no production decision**. It is the default of every
replay fixture and probe and of `NarsEngine`. G2 is therefore a question about
what future wiring should use, and about what the existing fixtures pin.

## History (classification)

- `NarsTables` and the "16 for full precision" comment: `de61aa5d`
  (2026-03-28, "CausalEdge64 — 64-bit causal neuron crate").
- `NarsEngine::new → build(1)`: `1d355e16` (2026-03-31, "wire causal-edge
  protocol into NarsEngine hot path"), commented "fast path: 1 c-level =
  128 KB" and "L1-resident". That hot path indexed the **deduction** table
  until `6e5e674a` (2026-09-10). No benchmark accompanies either commit.
- `replay_step` with the table: `badd0894` (2026-08-31, D-DCR-1). Its message
  cites "the kernel W0 measured at ~35 ns"; `dcr_w0_replay_budget` timed the
  table+forward kernel at `build(1)` only and never compared it with direct
  revision.

Classification: **a deliberate production default chosen on an unmeasured
speed/size assumption**, not a benchmark shortcut. It is not an accident, and
it was never measured against the law it replaces.

## Axis 1 — numeric surface vs A (R3 domain, 1,638,400 cells)

| law | tables | bytes | cells ≠ A | broad ≠ A | broad max dF | broad max dC | mean dF | mean dC | p50/p95/p99 dF | p50/p95/p99 dC | distinct c |
|---|---|---|---|---|---|---|---|---|---|---|---|
| B1 | 1 | 262,144 | 1,636,550 | 1,598,385 | 128 | 168 | 32.72 | 47.65 | 24/99/123 | 46/98/136 | 1 |
| B2 | 4 | 655,360 | 1,634,496 | 1,596,341 | 127 | 100 | 16.76 | 25.25 | 9/65/99 | 23/56/68 | 3 |
| B4 | 16 | 2,228,224 | 1,629,527 | 1,591,392 | 125 | 55 | 8.93 | 12.92 | 4/35/66 | 12/28/35 | 10 |
| B8 | 64 | 8,519,680 | 1,616,783 | 1,578,704 | 121 | 28 | 4.68 | 6.51 | 2/18/37 | 6/14/17 | 32 |
| B16 | 256 | 33,685,504 | 1,585,273 | 1,547,346 | 113 | 14 | 2.41 | 3.27 | 1/9/19 | 3/7/9 | 92 |

Mean error roughly halves per doubling. No level approaches A: even B16
differs on 96.8% of cells, and **broad max dF stays above 100 at every level**,
because a bucket midpoint misstates the relative weight of two inputs whose
confidences sit far apart inside their buckets.

## Axis 2 — replay drift (256 chains × 64 steps, two corpora)

`Uniform` spans the grid; `W0` is the distribution `dcr_w0_replay_budget`
sized the kernel on (f 128..=255, c 128..=227).

- **Every table diverges from A at step 1** on 245–256 of 256 chains. No chain
  under any table stays equal to A for 64 steps.
- Mean |dF| / |dC| at step 64 (hundredths in the pin):

| law | Uniform dF | Uniform dC | W0 dF | W0 dC |
|---|---|---|---|---|
| B1 | 43.86 | 83.25 | 16.34 | 81.00 |
| B2 | 42.75 | 45.57 | 16.33 | 32.00 |
| B4 | 39.53 | 24.14 | 11.86 | 18.91 |
| B8 | 33.77 | 12.52 | 6.56 | 9.74 |
| B16 | 28.05 | 6.20 | 3.63 | 4.69 |

- Frequency error **grows** with chain length on the uniform corpus at every
  resolution (B16: 2.2 at step 1, 28.1 at step 64). Error is not monotone in
  steps on W0 (B16 peaks at step 16).
- Only F/C ever differ: palettes, mask, direction, plasticity, witness and
  Epi5 agree with A at every step (pinned). So the table cannot move Epi5 or
  Pearl operator selection.
- A itself saturates on the uniform corpus: 43/256 chains are at c = 255 by
  step 64 (frequency frozen from then on), and 13 steps hit the G1 cell. W0
  never does.

## Axis 3 — decisions (every cut of 256 chains)

Flips against A. "verdict" = the chain's own frequency-bar verdict;
"LB" = load-bearing; "class" = `Reaction` variant; "c-only" = A said TruthOnly
with equal frequencies and the law said Inert.

| corpus, len | law | cuts | verdict | LB | class | c-only |
|---|---|---|---|---|---|---|
| Uniform, 8 | B1 | 2048 | 776 (37.9%) | 244 (11.9%) | 905 (44.2%) | 16 |
| Uniform, 8 | B2 | 2048 | 552 | 205 | 751 | 13 |
| Uniform, 8 | B4 | 2048 | 368 | 183 | 646 | 13 |
| Uniform, 8 | B8 | 2048 | 168 | 105 | 502 | 13 |
| Uniform, 8 | B16 | 2048 | 96 (4.7%) | 77 (3.8%) | 426 (20.8%) | 15 |
| Uniform, 2 | B1 | 512 | 104 | 125 | 156 | 1 |
| Uniform, 2 | B16 | 512 | 6 | 9 | 35 | 2 |
| W0, 8 | B1 | 2048 | 0 | 0 | 632 (30.9%) | 54 |
| W0, 8 | B16 | 2048 | 0 | 0 | 238 (11.6%) | 45 |

(All 20 rows are pinned in `AXIS3`.)

- **The W0 zeros are structural, not stability.** Every W0 frequency is
  ≥ 128, so every weighted mean is too, and no chain can fall below the 128
  bar under any law (pinned in `w0_corpus_cannot_flip_the_frequency_bar`).
- **The repository's own load-bearing fixture is load-bearing under B1 only.**
  `chain_counterfactual::cutting_a_load_bearing_edge_flips_the_verdict`
  (seed 200/200, steps 250/250 then 40/30, cut 0) gives B1 133 → 120, across
  the bar. A gives 246 → 194, and B2…B16 give 207–244 → 184–198: both arms
  clear the bar, so the cut is **not** load-bearing. The fixture was found by
  sweeping under B1 and pins a property of B1, not of revision.
- `DEFAULT_FREQUENCY_BAR`'s doc justifies reading frequency because
  confidence "saturates" at 170 in every chain. 170 is not saturation: it is
  B1 discarding both confidences and returning one constant. Under A,
  confidence varies along every chain. The frequency-over-confidence choice may
  still be right; its stated evidence is B1's constant.
- Chain admission reads no truth, so the law cannot affect it.

## Axis 4 — throughput

Machine: 4 vCPU Xeon @ 2.1 GHz; per core 48 KiB L1d, 2 MiB L2; 260 MiB shared
L3. Median of 7 runs per cell; the table shows the median of 3 such runs (the
spread across runs is in the transcript and stays within the noted ranges).

| law | memory | micro unif ns | micro w0 ns | replay dep ns | replay batch ns | cold ns |
|---|---|---|---|---|---|---|
| A | 0 | 12.1 | 12.1 | 32.7 | 15.7 | 11.5 |
| B1 | 256 KiB | 3.4 | 8.5 | 7.4 | 7.0 | 19.0 |
| B2 | 640 KiB | 3.8 | 8.6 | 7.4 | 7.4 | 28.2 |
| B4 | 2.1 MiB | 13.4 | 3.6 | 7.7 | 7.2 | 36.8 |
| B8 | 8.1 MiB | 17.5 | 6.0 | 8.4 | 7.3 | 51.7 |
| B16 | 32.1 MiB | 29.2 | 12.4 | 8.9 | 9.8 | 60.3 |
| A-slow (anti-vacuity) | 0 | ~480 | ~470 | ~500 | ~475 | ~455 |

Reading:

- **Replay, hot table:** every resolution is 3.7–4.4× faster than A on the
  dependent replay and 1.6–2.2× faster on the batch replay. The
  dependent replay is latency-bound and A's float divisions sit on the chain;
  the table is one load. B16 is within ~1.2× of B1 here.
- **Random access (micro, uniform):** only B1 and B2 beat A (3.5×). B4 ties A;
  **B8 and B16 are slower than A** (17.5 and 29.2 ns vs 12.1), because random
  indices miss L2.
- **Cold:** **A beats every table**, B1 included (11.5 vs 19.0 ns).
- The replay rows use 64 cyclic weights (the W0 probe's shape), so the touched
  table entries stay hot. A replay whose weights are spread over the grid
  behaves like the micro-uniform row, not the replay row.
- Unexplained, recorded: B1/B2 micro are slower on `w0` inputs than on
  `uniform` (8.5 vs 3.4 ns) and B4 the reverse, consistently across runs.

## Pareto table

| law | memory | replay dep / batch ns | micro unif ns | cold ns | broad max dF / dC | mean dF / dC | Uniform len 8 flips: verdict / LB / class |
|---|---|---|---|---|---|---|---|
| A | 0 | 32.7 / 15.7 | 12.1 | 11.5 | 0 / 0 | 0 / 0 | 0 / 0 / 0 |
| B1 | 256 KiB | 7.4 / 7.0 | 3.4 | 19.0 | 128 / 168 | 32.7 / 47.6 | 776 / 244 / 905 |
| B2 | 640 KiB | 7.4 / 7.4 | 3.8 | 28.2 | 127 / 100 | 16.8 / 25.3 | 552 / 205 / 751 |
| B4 | 2.1 MiB | 7.7 / 7.2 | 13.4 | 36.8 | 125 / 55 | 8.9 / 12.9 | 368 / 183 / 646 |
| B8 | 8.1 MiB | 8.4 / 7.3 | 17.5 | 51.7 | 121 / 28 | 4.7 / 6.5 | 168 / 105 / 502 |
| B16 | 32.1 MiB | 8.9 / 9.8 | 29.2 | 60.3 | 113 / 14 | 2.4 / 3.3 | 96 / 77 / 426 |

## The pre-registered boundaries, evaluated

| boundary | result |
|---|---|
| B1 survives if ≥ 1.5× faster in replay **and** planner outcomes stable | **Fails.** The speed half holds (2.2–4.4× replay, 3.6× random micro). The stability half does not: 37.9% verdict flips, 11.9% load-bearing flips and 44.2% Reaction-class flips on uniform 8-step chains, and the repository's own load-bearing fixture flips. |
| Some Bn faster while decision-stable | **Fails at every n.** B16 is the closest: 4.7% verdict, 3.8% LB, 20.8% class flips on uniform 8-step chains. |
| Table approach wounded if A replay ≥ best table / 1.1 | **Not wounded on hot replay** (A is 2–4× slower there). **Wounded off the hot path:** on random access B8/B16 are slower than A, B4 ties, and on a cold cache A beats every table. |
| B16 "full precision" falsified if material divergence remains | **Falsified.** 96.8% of cells differ, broad max dF 113, decision flips at every length. Comments corrected to "maximum table resolution". |

So the answer to "what did we buy by throwing confidence away": **4.4×
throughput on a hot, dependent replay (2.2× batched), and 3.6× on random lookups — for a law
that flips about 4 in 10 verdicts on spread inputs.** Paying for resolution
does not rescue it: B16 keeps the hot-replay speed, loses the random-access
and cold speed, and still flips decisions.

## The replay contract (item 11)

`replay_step` runs `forward` (opcode = the weight's inference field: palettes
compose, mask intersects, direction and plasticity from the weight, F/C from
that opcode's truth function), then overwrites F/C with the table's revision.

- They are different operations: forward computes the **rule** the weight
  declares (deduction, induction, …); the table computes **revision** of the
  running truth with the weight. The overwrite is deliberate (`badd0894`:
  "no second truth register").
- forward's own F/C computation is discarded every step. Its other outputs
  are kept, and its opcode decode still matters (it faults on Counterfactual
  and Intervention weights). Possible compute waste, not removed here.
- The output edge's inference field names the weight's opcode (e.g.
  Deduction) while its F/C carry a revision. Recorded as a label/value
  mismatch, not changed.

## Comment corrections landed

- `tables.rs`: "Use 16 for full precision" → "maximum table resolution … NOT
  numerical full precision", with the measured gap; "Fits L1 cache" and
  "32 MB — fits L2" replaced by the measured cache picture.
- `lib.rs`: tables are 128 KiB (u16 entries), not 64 KB, and do not fit L1.
- `nars_engine.rs`: "(128 KB, L1 cache resident)" and "(full precision)"
  corrected. No behaviour change.

## Independent guards (#1423)

- **T2** (`causal-edge/tests/t2_pearl_subset_mask.rs`): PO (0b011) < SO
  (0b101) numerically while PO ranks above SO; PO and SO are incomparable
  subsets; no `field >= k` threshold selects "Intervention or above"; a paired
  test shows the check passes for a numeric ranking.
- **T4** (`causal-edge/tests/t4_learn_plasticity_patterns.rs`): `learn` over
  all eight plasticity patterns keeps frozen planes, moves hot planes, and
  passes Pearl/Direction/Inference/Witness/Epi5 bit-exactly.

## Disable runs (all red as required)

- B1 replaced by A → 8 tests fail (surface, drift, flips, fixture, both
  mirror tests).
- All `c_levels` forced to 1 → surface, drift, flips and fixture pins fail.
- Real `replay_step` stops writing the table result → both mirror-faithfulness
  tests fail. (The drift/decision pins measure the laws through the mirror, so
  they do not move; the mirror tests are what tie them to production.)
- Flip counter made blind to confidence → the confidence-only fixture, the
  flips pin and the decision-mirror test fail.
- Bench: A-slow measures ~40× slower than A, so the harness separates kernels.
- T4: frozen-S check removed → fails; `learn` clearing the Pearl bits → fails.
- T2: rung read from the numeric field → 2 of 4 fail.

## Kill conditions for what this leaves open

- **B1 as a default** is dead unless a workload is shown whose inputs keep
  frequencies on one side of every decision bar, *and* whose decisions do not
  read `Reaction`'s class. W0 shows the first half can hold; the second did not
  (30.9% class flips).
- **A table at any resolution** survives only for hot, dependent replay where
  the decisions it feeds are frequency-bar verdicts on inputs that cannot
  straddle the bar. Otherwise A is as fast or faster (random, cold) and exact.
- **The load-bearing fixture** must be re-derived under whichever law
  production adopts. Under A it is not load-bearing.
- **`DEFAULT_FREQUENCY_BAR`'s rationale** must be re-stated without B1's
  constant before any change to the replay law.

```
STATUS: measured | OUTCOME: B1 buys 2-4.4x hot-replay and 3.6x random-lookup
speed and loses ~38% of verdicts on spread inputs; no table resolution is
decision-stable; A wins cold and beats B8/B16 on random access; no production
code reads any NarsTables revision today; B16 "full precision" corrected;
T2/T4 guards landed | OPEN: G2 (which law future wiring uses), G1,
load-bearing fixture and frequency-bar rationale under the chosen law,
forward's discarded truth
```

MIRROR
- BLIND SPOT: two synthetic corpora; no real recorded chain set exists in the
  tree to replay. The replay bench uses 64 cyclic weights, which flatters every
  table.
- BIAS CHECK: the question was framed against B1, which made "B16 fixes it"
  the easy counter-story; the micro-uniform and cold rows are what refute it.
- STILL OPEN: why B1/B2 micro lookups are slower on narrower inputs.
