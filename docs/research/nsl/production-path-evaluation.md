# NSL on lance-graph: production path evaluation

**Status:** MEASURED on synthetic data and frozen NSL oracles. Code:
`crates/lance-graph-nsl`. Plan: `AdaWorldAPI/neural-structured-learning`
`research/2026-10-25/` (H1 `01a`, H5 `05a`, H7 `07a`, `00a` W0–W2).

Three paths were tried against the frozen NSL oracles:

- **A. Label codebook:** node IDs interned once, sorted; edges are rank codes.
- **B1. Palette512 (9-bit sign-magnitude, SM9) edge weights.**
- **B2. SM9 neighbor features:** the graph-loss distance computed in code
  space, and input features stored as SM9 behind a model.

## Recommendation

| path | verdict | why |
|---|---|---|
| A. label codebook | **ADOPT** | Exact NSL parity (18 of 19 fixtures; NaN refused). Cold build halved, query unchanged. |
| B1. SM9 weights for **selection** | **REJECT** | Changes the top neighbor for 7–21% of seeds, and the packed order for 45–60%. |
| B1'. exact f64 order, SM9 weight **value** for the loss | **PROBE** | Selection stays exact; the stored weight error is ≤ 0.25%. Breaks bit parity of `NL_nbr_*_weight`, so it needs its own tolerance contract. |
| B2. SM9 distance in the **training loss** | **REJECT** | Logits are recomputed every step and never stored, so there is nothing to save. Quantizing them destroys close pairs, which is where regularization drives them. |
| B2'. SM9 **stored neighbor features** | **PROBE** | 3.6× less storage than f32, about 1% loss error for pairs differing by 0.05 or more, failure below about 0.01. Needs a real model and a training run (B01, Cora) before adoption. |

The production path is:

1. A, plus the frozen-join CSR, as the neighbor store.
2. The loss in f32 (`loss::graph_loss`, within 3 ULP of TensorFlow).
3. SM9 only as a storage format for neighbor input features, behind a
   gate that falls back to f32 for near-duplicate pairs.

## A. Label codebook

Every node ID is stored once and sorted. A node's code is its rank, so NSL's
tie-break (larger ID first) is a `u32` compare. Edges are `(src, dst, f64)`
collapsed by a stable sort, which keeps NSL's rule that the first of two
equal weights wins. Python treats `0.0 == -0.0`, and the bits of whichever
was read first are emitted (new fixtures G02b and G02c). An unstable sort
would get this wrong, and its disable fails.

| | String maps | codebook |
|---|---|---|
| cold parse, 20,001 edges | ≈ 7,600 µs | ≈ 3,700 µs |
| cold freeze | ≈ 2,000 µs | ≈ 1,000 µs |
| `neighbors(seed, 16)` | 1–2 ns | 1 ns |

Cold builds are reported, not gated. 14 disables, each red on a named test.

## B1. SM9 edge weights

The quantizer is monotone, so the quantized order equals NSL's exactly when
no two neighbors of one seed share a code. That is pinned two-sided in
`sm9_weights_keep_nsl_order_on_grid_and_can_break_it_off_grid`: on-grid
weights keep the order, and 0.8000 / 0.8015 (both code 204) flip it.

On weights shaped like NSL `build_graph` output (similarity ≥ threshold,
written with `%f`, power-law degree 2–65, 2,000 seeds):

| threshold | max weight rel err | top-1 changed | top-4 set | top-4 order | full order |
|---|---|---|---|---|---|
| 0.80 | 2.4e-3 | 7.2% | 8.8% | 24.7% | 45.1% |
| 0.95 | 2.1e-3 | 20.6% | 23.6% | 50.2% | 60.5% |

A threshold of 0.8 leaves only 52 of the 255 positive codes in use, and 0.95
leaves 14. A signed format with zero point 0 cannot shift its range to
[threshold, 1]. That rules out quantized weights as sort keys. The sort runs
once, at freeze, so keeping it exact costs nothing at query time.

## B2. SM9 features and the graph loss

### What NSL computes

`GraphRegularization.call` runs the base model on the copied neighbor
features, and `PairwiseDistance` compares the **logits** of seed and
neighbors. The default is L2 = `mean_squared_error` under
`SUM_BY_NONZERO_WEIGHTS`, i.e. divided by the count of nonzero
weight-broadcast elements. Logits exist for one step and are never stored.

### Oracle

`oracle/nsl_loss_oracle.py` runs NSL's unmodified `pairwise_distance_wrapper`
(TF 2.21) on 10 seeded cases and freezes inputs plus result:

- L2 at d = 4/32/128, L2 summed over the axis, L1, cosine;
- near-duplicates, outliers, zero weights;
- the `00a` toy, which matches the hand value 0.18851.

The f32 reference in `loss.rs` agrees within 0–3 ULP on all 10.

### Quantized distance (400 seeds per row, vs the f32 reference)

Each row of the sweep has 16 pairs at d = 32. The noise is the per-component
difference between seed and neighbor, with features ~ N(0, 1). Shown:
relative loss error as median / p95, and the 5th percentile of the gradient
cosine.

| noise | SM9 shared | int8 shared | SM9 per-row | int8 per-row | grad cos SM9 / int8 |
|---|---|---|---|---|---|
| 0.5 | 7.6e-4 / 2.1e-3 | 1.4e-3 / 4.5e-3 | 4.8e-4 / 1.4e-3 | 9.5e-4 / 2.7e-3 | 0.99990 / 0.99958 |
| 0.14 | 2.5e-3 / 7.6e-3 | 6.6e-3 / 1.9e-2 | 1.6e-3 / 5.1e-3 | 4.2e-3 / 1.2e-2 | 0.99894 / 0.99571 |
| 0.05 | 1.2e-2 / 2.9e-2 | 4.5e-2 / 8.5e-2 | 6.3e-3 / 1.7e-2 | 2.3e-2 / 4.9e-2 | 0.99190 / 0.96839 |
| 0.01 | 2.6e-1 / 4.1e-1 | 1.0 / 1.5 | 1.4e-1 / 2.1e-1 | 5.7e-1 / 7.7e-1 | 0.84588 / 0.62692 |
| 0.001 | 8.9 / 13 | 19 / 28 | 6.4 / 8.4 | 14 / 20 | 0.24269 / 0.15380 |

Findings:

- **9-bit beats int8 at every noise level.** For L2 and cosine the error
  ratio approaches 4× at small differences, as expected when the dominant
  error is the quantization bias (≈ step²/6 per element) and the step halves.
- **Per-row scales beat one shared scale,** but are no longer integer-only.
- **Every encoding fails once the difference falls below about 0.01** (a
  similarity of about 0.9999). L1 degrades earlier: the gradient is
  sign(a − b), and signs flip (gradient cosine 0.86 at 0.05).
- **Subtracting the bias** (`l2_bias_correction_probe`) cuts the error about
  3× at 0.05–0.01, overshoots at 0.001 (equal codes correlate the errors), and
  leaves the gradient unchanged.

Graph regularization pulls neighbor logits together, so the failure regime
is the converged state. A quantized loss would stall training at its own
noise floor.

### Stored input features (fixed random tanh layer, 64 → 16)

| noise | SM9 per-row | int8 per-row |
|---|---|---|
| 0.5 | 1.1e-3 / 2.5e-3 | 1.9e-3 / 5.9e-3 |
| 0.14 | 3.3e-3 / 1.0e-2 | 6.9e-3 / 2.0e-2 |
| 0.05 | 1.1e-2 / 3.2e-2 | 2.8e-2 / 7.8e-2 |
| 0.01 | 1.8e-1 / 3.1e-1 | 7.0e-1 / 1.0 |

The pattern is the same. The model is fixed and random, so this does not
show what happens as training converges.

## Literature (checked against the papers where cited)

- **Neural Graph Machines** (Bui, Ravi, Ramavajjala, arXiv 1703.04818): the
  loss is Σ w_uv d(h_u, h_v), with L1 or L2 on the last layer. Edge weights
  come from cosine similarity, up to 5 neighbors. No quantization is studied.
- **ScaNN** (Guo et al., arXiv 1908.10396): for inner-product search,
  quantization should be judged by the score error on high-scoring pairs, not
  by reconstruction error. That supports the finding here: the error that
  matters is on the closest pairs, and uniform quantization is worst there.
- No published ablation of quantized features or edge weights for an
  NGM/NSL-style graph loss was found. These measurements are the evidence.

## What exists, and what production would need

From ndarray `origin/master` `2cbe131` and lance-graph `origin/main` `e731a49`:

- **Shipped:** `hpc::codec9::{Codec9, EpiLane9}` (scalar encode/decode, scale
  fixed at compile time) and `simd::lut9`.
- **Not shipped:** an SM9 dot / L2 / cosine kernel or a u8×u8 dot. These exist
  only as example probes, `palette512` marked EXPERIMENT / NOT ADMITTED.
- **lance-graph:** no 9-bit or int8 embedding store, and no graph-loss code
  before this crate.

So B2' needs, in order:

1. A runtime per-row (or per-tensor) scale for Codec9.
2. A batch encode/decode.
3. A Cora-scale training comparison (B01).
4. Only then, a dot or L2 kernel admitted in ndarray.

## Not covered

- Real training runs: accuracy or convergence under each storage format.
- Cora or MNIST scale (B01, B02).
- SIMD timing of an SM9 kernel.
- The plain HashMap + SoA control from H1.
