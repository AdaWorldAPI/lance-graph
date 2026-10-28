# NSL on lance-graph: production path evaluation

**Status:** MEASURED on synthetic data and frozen NSL oracles. Code:
`crates/lance-graph-nsl`. Plan: `AdaWorldAPI/neural-structured-learning`
`research/2026-10-25/` (H1 `01a`, H5 `05a`, H7 `07a`, `00a` W0–W2).

Three paths were tried against the frozen NSL oracles:

- **A. Label codebook:** node IDs interned once, sorted; edges are rank codes.
- **B1. Palette512 (9-bit sign-magnitude, SM9) edge weights.**
- **B2. SM9 neighbor features:** the graph-loss distance computed in code
  space, and input features stored as SM9 behind a model.
- **C. The palette mechanism proper,** not uniform quantization: a palette256
  index as the needle (which row) with the Fisher-z i8 pair LUT for the
  distance between two needles (FamilyGamma). Plus the TFLM-shaped softmax
  table keyed by `q - q_max` (the max reads exp(0) = 1) for NSL's KL / JS
  losses, and residual coding (source codes plus a separately scaled
  residual).

A and B were measured first, with uniform quantization only. Section C adds
the palette and softmax mechanisms. It overturns the B1 conclusion for
top-k selection and replaces the B2 production answer.

## Recommendation

| path | verdict | why |
|---|---|---|
| A. label codebook | **ADOPT** | Exact NSL parity (18 of 19 fixtures; NaN refused). Cold build halved, query unchanged. |
| B1. SM9 weights (fixed 1/255) for **selection** | **REJECT** | Changes the top neighbor for 7–21% of seeds, and the packed order for 45–60%. |
| C1. **Fisher-z i8** weights (FamilyGamma fitted to the observed range) | **PROBE** | Top-1 changes for 0.6–0.8% of seeds and the top-4 order for about 5%: 9–34× better than SM9, 3× better than a fitted affine u8. Not exact, so the compat path keeps f64 order. A compact store or a quantized top-k behind a tolerance contract. |
| C5. **z-linear at 9 bits** (SM9 reach, 511 codes, same 8-byte envelope) | **PROBE (best for weights)** | Top-1 changes for 0.3–0.4% of seeds, top-4 for 2.9–4.3%. About half the 8-bit error, and 4× below FamilyGamma i8 on the palette512 LUT. |
| C5'. **Euler-gamma Gumbel deskew** in the 8-byte envelope | **REJECT for selection** | It does flatten the codes (entropy 6.6 → 7.7 bits) and cuts mean error 1.4–3×. But it moves codes out of the tail, so top-1 order gets 2–3× worse and the LUT's max error grows 20×. A perfect quantile deskew does the same. |
| B1'. exact f64 order, SM9 weight **value** for the loss | **PROBE** | Selection stays exact; the stored weight error is ≤ 0.25%. Breaks bit parity of `NL_nbr_*_weight`, so it needs its own tolerance contract. |
| B2. both endpoints quantized, distance in code space | **REJECT** | Fails once seed and neighbor differ by less than about 0.01, the converged state. Holds for L2, L1, cosine, and KL / JS through the softmax table. |
| C2. palette256 needle + pair LUT for the **distance** | **REJECT** | 47–99.8% of close pairs land on one centroid and read distance 0. Loss error 50–100%, LUT reading irrelevant. Right for identity, wrong for a residual. |
| C3. TFLM-shaped softmax table, int8 **output** grid (1/256) | **REJECT** | 17% median KL error even for distant pairs. Probabilities below 1/512 round to 0. An int16 output grid adds nothing over input error. |
| C4. **residual coding**: source codes + int8 residual on its own scale | **PROBE (strongest)** | Median loss error flat across noise 0.5 → 0.001: 0.02–0.07% for L2, L1 and cosine; 0.2–0.5% for KL and JS. At small KL that is inside TF's own f32 error. Needs a real stored-feature use and a training run. |
| B2'. SM9 **stored neighbor features**, endpoints coded independently | **SUPERSEDED by C4** | About 1% error for pairs differing by 0.05 or more, failure below about 0.01. |

The production path is:

1. A, plus the frozen-join CSR, as the neighbor store; f64 weights for the
   compat path (exact NSL order).
2. The loss in f32 (`loss::graph_loss`: within 3 ULP of TensorFlow for L2 / L1
   / cosine, inside TF's own cancellation budget for KL / JS).
3. If stored features or weights must shrink: neighbors coded as a residual
   against their seed (C4), and weights as Fisher-z i8 (C1). Both are PROBE
   until a training run on real data.

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

## C. The palette mechanism

The palette doctrine (`E-PALETTE256-IS-A-NEEDLE-THE-COLON-IS-THE-DISTRIBUTION-1`)
splits a value into two parts:

- **the needle:** a palette256 index, exact. It says which entry a value is.
- **the distribution:** read from a pair LUT. Here that is the Fisher-z i8
  table of `bgz-tensor::fisher_z`. It stores atanh(cosine) on a range fitted
  to the data, so resolution is spent near ±1, where the tails are.

Same index reads 1.0 from the address, never from the table. TFLite's palette
polyfill (D-TFLP) applies the same idea to softmax: a table keyed by
`x − max_row`, so the maximum reads table[0] = 1 exactly, plus one row
reduction.

The mirrors in `eval.rs` (test-only) are:

- `FamilyGamma`: atanh clamped to ±0.9999, i8 truncation. It matches
  bgz-tensor's own encode/decode test, and its truncation is pinned.
- `softmax_table_row`, keyed by `q − q_max`.
- `Palette`: spherical k-means, k = 256. Self-pairs read 1 from the address;
  the table's own diagonal decodes below 1.

All three are pinned by named tests.

### C1. Fisher-z weights: the MP3 effect is real, for selection

Same `build_graph`-shaped weights and seed as B1:

| threshold | encoding | codes used | top-1 changed | top-4 order | full order |
|---|---|---|---|---|---|
| 0.80 | SM9 at 1/255 | 52 | 7.2% | 24.7% | 45.1% |
| 0.80 | affine u8, fitted range | 256 | 2.4% | 7.2% | 26.9% |
| 0.80 | Fisher-z i8, fitted | 242 | **0.8%** | **5.0%** | 42.6% |
| 0.95 | SM9 at 1/255 | 14 | 20.6% | 50.2% | 60.5% |
| 0.95 | affine u8, fitted range | 256 | 1.6% | 6.4% | 27.7% |
| 0.95 | Fisher-z i8, fitted | 252 | **0.6%** | **4.8%** | 40.4% |

There are two effects.

**Fitting the range** is most of the gain over SM9. SM9 wastes 80–95% of its
codes below the threshold.

**The atanh stretch** buys another 3× at top-1. It spends codes near 1, where
the strongest neighbors live and where `max_nbrs` cuts. It pays for that at
the low end: the full order is worse than affine. That trade is right for
NSL, which keeps only the top `max_nbrs`.

On the palette LUT itself (below), the same shape shows up. Fisher-z has a
larger maximum error overall (1.5e-2 vs affine 3.2e-3), and a smaller one
for pairs with cosine above 0.9 (2.2e-3 vs 3.2e-3).

### C2. Palette needle for the loss distance

Spherical k-means with k = 256 is trained on 8,192 rows of 64 clustered
prototypes (d = 32). Then 200 × 16 fresh pairs are drawn. The loss is
NSL cosine, with each row's cosine read as pair(index(seed), index(neighbor)).

| spread | noise | same needle | rel err, f32 LUT | Fisher-z LUT | affine LUT |
|---|---|---|---|---|---|
| 0.05 | 0.14 | 47.1% | 0.96 | 0.96 | 0.96 |
| 0.05 | 0.01 | 91.2% | 0.55 | 0.54 | 0.53 |
| 0.05 | 0.001 | 99.2% | 1.0 | 1.0 | 1.0 |
| 0.15 | 0.05 | 84.6% | 0.46 | 0.48 | 0.48 |
| 0.15 | 0.001 | 99.8% | 1.0 | 1.0 | 1.0 |

Values are median relative error.

The LUT reading does not matter: the error is the codebook's. A needle answers
which row a value is, exactly. It cannot carry the gap between a seed and its
near-duplicate neighbor, and that gap is all the graph loss measures. The
doctrine says the same thing from the other side: the distribution is read
from the colon pair, and two equal needles have none.

### C3. NSL's KL / JS through the softmax table

The oracle was extended with six cases: KL and JS with `transform_fn=SOFTMAX`
on 10-class logits, at three noise levels. The earlier cases are
byte-identical. The f64 reference lands inside TF's own f32 cancellation
budget (≤ 0.10 of it). Measured in ULP, TF's value moves by up to 22,000 at
the smallest divergence, because the KL is a difference of near-equal f32
terms.

Logits are quantized under one shared scale. The softmax uses the table keyed
by `q − q_max`, and the divergence is NSL's. The sweep uses 400 seeds,
16 pairs × 10 classes, logits ~ 2·N(0,1), and reports median relative error:

| dist | noise | in9 | in8 | in8 + int8 output (1/256) | in8 + int16 output | **in8 + int8 residual** |
|---|---|---|---|---|---|---|
| KL | 0.5 | 2.2e-3 | 4.4e-3 | 1.7e-1 | 4.3e-3 | **1.7e-3** |
| KL | 0.05 | 2.0e-2 | 4.8e-2 | 6.4e-1 | 4.7e-2 | **1.7e-3** |
| KL | 0.01 | 2.1e-1 | 8.1e-1 | 7.1 | 8.1e-1 | **1.8e-3** |
| KL | 0.001 | 7.3 | 16 | 220 | 16 | **5.3e-3** |
| JS | 0.05 | 1.8e-2 | 4.2e-2 | 2.9e-1 | 4.2e-2 | **1.9e-3** |
| JS | 0.001 | 7.8 | 15 | 33 | 15 | **1.8e-3** |

Findings:

- **The needle is exact, and it does not help here.** The max reads exactly 1,
  but the divergence between two near-equal distributions lives in the
  logit differences below the input step.
- **The int8 output grid is the dominant error.** TFLM int8 softmax writes
  probabilities at 1/256. Any class below 1/512 becomes 0, and NSL's
  `ln(p + 1e-7)` turns that into a large term. An int16 output grid matches
  input-only error. "TFLM-shaped" here means the table and the reduction; the
  table is f64, not TFLM's fixed-point one, so this is not byte parity.
- **Residual coding fixes all of it.** Code the seed's logits, then code
  `neighbor − seed` on its own scale. The residual's step is set by the gap
  itself, so the gap survives at every noise level.

### C4. Residual coding across every NSL distance

Same 400-seed sweep as B2. Seed int8 plus residual int8, median / p95:

| dist | d | noise 0.5 | 0.05 | 0.01 | 0.001 |
|---|---|---|---|---|---|
| L2 | 32 | 4.5e-4 / 1.3e-3 | 4.5e-4 / 1.2e-3 | 4.3e-4 / 1.3e-3 | 4.7e-4 / 1.3e-3 |
| L1 | 32 | 2.5e-4 / 7.9e-4 | 3.1e-4 / 7.8e-4 | 2.9e-4 / 8.2e-4 | 2.5e-4 / 8.7e-4 |
| cosine | 32 | 6.8e-4 / 1.7e-3 | 6.9e-4 / 2.0e-3 | 6.7e-4 / 2.0e-3 | 7.2e-4 / 2.0e-3 |
| L2 | 128 | 2.3e-4 / 7.0e-4 | 2.7e-4 / 6.8e-4 | 2.4e-4 / 7.8e-4 | 2.5e-4 / 7.4e-4 |

Compare B2 at noise 0.001: an error of 9–19 with both endpoints coded
independently. Here it is flat, and 2× better at d = 128. The falsifier
`residual_coding_keeps_a_near_duplicate_loss_that_endpoint_coding_loses` is
two-sided on the same near-duplicate case: endpoint coding has a relative
error above 1, residual coding below 1%.

This is the palette doctrine's own answer, made concrete:

- the needle (the seed's codes) says where the pair is;
- the residual carries the distribution (how far apart the two are).

It is `EdgeCodecFlavor::CoarseResidue` in miniature.

### C5. Euler-gamma deskew in the 8-byte envelope, and 9 bits

The question: can `std::f64::consts::EULER_GAMMA` deskew the value
distribution, still inside FamilyGamma's 8 bytes of metadata (two f32)?

The 8-byte envelope can hold two shapes over z = atanh(c):

- **z-linear:** `(z_min, z_range)`, codes spread evenly over z. This is
  FamilyGamma, but with rounding instead of truncation.
- **z-Gumbel:** the Gumbel probability-integral transform, with
  `beta = sd·√6/π` (its sign set by the skew) and
  `mu = mean − EULER_GAMMA·beta`. Codes are equal-probability bins.

EULER_GAMMA is the mean of the standard Gumbel distribution, so it is what
places the location; `euler_gamma_puts_the_gumbel_location_where_the_sample_has_it`
pins this. The z-Gumbel shape is the deskew: on Gumbel-shaped z it turns the
code histogram flat.

The helix "constant γ shove" (`z + EULER_GAMMA·const`) is the wrong place for
γ. A constant shift before a range fit cancels exactly, so the codes do not
change.

Each shape is run at 255 levels (i8 reach) and 511 levels (the numeric reach
of a 9-bit SM9 lane, i.e. 2 × u9: a palette512 index pair keys a 512 × 512
LUT of 9-bit values). The equal-count quantile is the ceiling for any deskew.
It needs 255 / 511 floats, not 8 bytes.

| data | encoding | entropy (bits) | mean abs err | top-10% err | max err | top-1 order | top-4 order |
|---|---|---|---|---|---|---|---|
| uniform w ≥ 0.8 | FamilyGamma i8 | 6.56 | 1.4e-3 | 1.5e-4 | 5.4e-3 | 0.6% | 5.7% |
| | z-linear 8 | 6.57 | 7.1e-4 | 7.5e-5 | 2.7e-3 | 0.9% | 5.2% |
| | z-Gumbel 8 | 7.72 | 2.6e-4 | 2.1e-4 | 1.1e-3 | 2.1% | 6.8% |
| | quantile 8 (ceiling) | 7.99 | 2.0e-4 | 1.9e-4 | 5.5e-4 | 1.6% | 6.5% |
| | **z-linear 9** | 7.57 | 3.5e-4 | 3.7e-5 | 1.4e-3 | **0.4%** | **2.9%** |
| | z-Gumbel 9 | 8.71 | 1.3e-4 | 1.0e-4 | 5.5e-4 | 0.8% | 3.0% |
| embedding cosines ≥ 0.8 | FamilyGamma i8 | 7.10 | 4.0e-4 | 2.7e-4 | 1.0e-3 | 0.7% | 8.7% |
| | z-linear 8 | 7.11 | 2.0e-4 | 1.4e-4 | 5.2e-4 | 0.5% | 7.7% |
| | z-Gumbel 8 | 7.92 | 1.5e-4 | 5.2e-4 | 9.5e-3 | 1.4% | 7.0% |
| | **z-linear 9** | 8.11 | 1.0e-4 | 6.7e-5 | 2.6e-4 | **0.3%** | **4.3%** |
| | z-Gumbel 9 | 8.92 | 7.4e-5 | 2.6e-4 | 6.4e-3 | 0.8% | 4.5% |
| palette512 pair cosines | FamilyGamma i8 | 5.80 | 7.0e-3 | 5.6e-3 | 1.5e-2 | – | – |
| | z-linear 8 | 5.80 | 3.5e-3 | 2.7e-3 | 7.3e-3 | – | – |
| | z-Gumbel 8 | 7.70 | 2.0e-3 | 1.2e-2 | 1.6e-1 | – | – |
| | **z-linear 9** | 6.80 | 1.7e-3 | 1.4e-3 | 3.6e-3 | – | – |
| | z-Gumbel 9 | 8.69 | 1.1e-3 | 7.1e-3 | 1.4e-1 | – | – |

The embedding cosines come from 200 clusters × 20 neighbors, d = 32, the same
shape as `build_graph`. The z skew is +0.8 to +4.6 on all three sets.

Findings:

- **The deskew works, and it is the wrong goal here.** The Gumbel envelope
  raises entropy by 1–2 bits and lowers the mean error. It gets there by
  taking codes away from the long right tail in z, which is where the
  strongest neighbors and the closest pairs sit. Top-1 order gets 2–3× worse,
  and the LUT's maximum error grows 20×. The quantile ceiling shows the same
  pattern, so this is a property of deskewing, not of the Gumbel fit.
  Selection and "which pairs are close" need resolution in the tail; that is
  the MP3 effect the z-stretch already gives. A flat code histogram optimizes
  the average, and the average is not what NSL reads.
- **Rounding beats FamilyGamma's truncation.** At the same 255 levels and the
  same range, z-linear has half FamilyGamma's mean and max error.
  bgz-tensor's `encode` truncates (`as i8`), which biases every value toward
  zero by up to one step.
- **9 bits (2 × u9) is the real gain.** z-linear at 511 levels halves the error
  of 8 bits and halves top-4 order changes, at the same 8 bytes of metadata.
  On the palette512 LUT it is 4× below FamilyGamma i8 (mean 1.7e-3, max
  3.6e-3).
- **palette512 needles still collide** (`palette512_needle_probe`, f32 LUT).
  The share of close pairs on one centroid drops from 47% to 25% at noise 0.14
  and from 91% to 87% at noise 0.01, but the median loss error stays at
  0.59–0.93. The needle says which entry; the residual has to carry the gap
  (C4).

**Limit.** The residual is per pair. A neighbor shared by many seeds needs one
residual per seed it serves. That is the copy NSL makes and this crate
removed, unless neighbors are coded against a shared anchor.

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
- C1 and C4 on real `build_graph` output and real stored features.
- A shared-anchor residual for neighbors used by many seeds.
- TFLM byte parity of the softmax table. The TFLM reference is not built
  here; the probe is `.claude/probes/fisherz-softmax-v1/softmax_oracle.cc`.
