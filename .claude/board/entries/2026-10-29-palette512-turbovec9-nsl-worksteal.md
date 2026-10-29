# 2026-10-29 — Palette512 × TurboVec9 × NSL: work-steal / consolidation map

**Status: RESEARCH / FALSIFIER PLAN, not an implementation or an admission.**
**Single dated entry; do not copy this text into ndarray or TurboVec boards.**
Repo boundaries: ndarray owns numerical math and DR512, TurboVec owns the
TurboQuant index, lance-graph owns graph identity/edge evidence and NSL
integration. Prior reports in `crates/lance-graph-turbovec/KNOWLEDGE.md`
are historical measurements, **not current API compatibility evidence**.

## Thesis

**Two 9-bit spaces with deliberately different meaning:**

| Area | `ndarray::hpc::codec9` DR512 | `turbovec` 9-bit lab |
|---|---|---|
| Codes | 512 words = 511 signed numeric values + 1 NEVER_TOUCHED | 512 Lloyd-Max centroid IDs, all numeric |
| Word | P×M (one presence/polarity bit + u8 magnitude); touched zero is +0 | unsigned 0..511 index, packed into nine bit planes |
| Source truth | `Codec9::encode_touched`, `EpiLane9` | `codebook::lab_codebook9`, current upstream `Rotation` |
| Calibration | Compile-time `Calibration` + nearest rounding | Scalar lab Beta Lloyd-Max + norm correction; **no TQ+ calibration** in lab |
| Utility | Preserve negative/positive/zero/never-seen **epistemic state** | Retain the ordering of continuous embedding similarities |
| Learner role | Encode selected values and absence; preserve sparse update integrity | Generate/retrieve plausible near-neighbor / hard-negative candidate IDs |
| Forbidden inference | `P` cannot be reused as arbitrary numeric centroid-high-bit | 9-bit codes do **not** carry DR512 NaN, signedness or permissions |

Never reinterpret one codeword as the other, even though the widths match.
A mapping needs a decoded numeric interpretation and a selected encoding
policy. Never manufacture evidence by converting NEVER_TOUCHED into +0.

**Storage math (payload only)**: 9b/dim = 1.125 B/dim vs i8 1 B,
i4 = 0.5 B and f32 = 4 B. Therefore 9b costs **12.5% over i8**
and **2.25× i4**, and saves **71.875% vs f32** before metadata,
padding, blocked layouts and LUT/codebook caches. 511 signed values
in DR512 do not represent 512 distinct numerical centroid values.

## Work stealing from existing substrate (no third codec)

| Work unit | Existing canonical source | STEAL / COMPOSE | Do not reproduce |
|---|---|---|---|
| Presence / sparse lane | ndarray `src/hpc/codec9.rs`; LG `lane_codec.rs` | Codec9 encode & 512-codeword parity | Alternative `-0`/Asym/sentinel law |
| u8 / i8 distance lookup | ndarray `simd_stride::{lut_u8,lut9}` | Facade LUTs if matching exact operation | Second nibble-LUT implementation in LG |
| Coarse + signed i4 residue | ndarray `src/hpc/edge_codec.rs` | `CoarseResidueCodec`, `ProductQuantizer` | Claiming it is TurboQuant's Beta Lloyd-Max |
| Coarse search cascade | ndarray `src/hpc/cam_pq.rs` | PQ-ADC / six 256-entry per-query tables | Duplicate trained `CamCodebook` |
| Search index | AdaWorldAPI/turbovec `turbovec/src/search.rs`, `codebook.rs`, `rotation.rs` | Vector-major, grouped VNNI dot, TQ+ if available, index/ID/filter | New clone of upstream index |
| Learning input | LG `lance-graph-nsl/src/lib.rs` | `FeatureTable` + `NeighborBatch`, `SourceRef`/seal; Google `pack_nbrs` oracle | Copying features or silently reordering edges |
| Learning loss | LG `lance-graph-nsl/src/eval.rs` | Graph loss fixtures and paired DR512 loss experiments | Another TensorFlow-loss semantics |
| KV/cache precedent | LG `bgz-tensor/src/turboquant_kv.rs` | Gain/shape i4 and Hamming as an independent **control** | Calling this byte-compatible with TurboVec or certified top-k |
| AVX/VNNI matmul baseline | ndarray `simd::matmul_i8_to_i32` | Measure on GEMM-shaped rerank | Replace LUT-ADC scan with full GEMM by default |

The old `lance-graph-turbovec` bridge references `turbovec`'s removed
`ndarray-simd` feature and removed `search_polyfill_with_db` /
`reconstruct_db_i8_transposed` methods after the 2026-10-29 upstream
refresh. **Unverified and likely non-building with current sibling
main**. First run its standalone cargo gate and repair as a separate
version-aware compatibility PR; do not let this block the independent
NSL and TurboVec lab probes.

## Why U9 may offer complementary learning benefits

Google NSL regularizes a model with labeled/unlabeled neighbor
relationships. Retrieval and loss are separate operations. TurboQuant's
random/structured rotation plus Beta Lloyd-Max quantization is
**data-oblivious in its codebook**, enabling candidate search without
per-corpus centroid training; online index maintenance is its potential
advantage. This does not confer privacy guarantees or automatically
preserve training quality.

1. **Construct better candidate graphs**: measure held-out *labeled
   graph-edge* precision/recall as well as f32-neighbor recall when
   replacing i4 with U9. False neighbors can produce incorrect regularizer
   targets. Existing controlled probe: `AdaWorldAPI/turbovec#7`.
2. **Hard-negative mining**: U9 might separate close negatives that i4
   aliases. Falsify with class/labeled graph identities, class-balanced
   precision@k, training loss/gradient comparison. Do not infer
   improvement from synthetic random-vector recall alone.
3. **Online / changing embeddings**: measure codebook stability and rebuild
   cost. The *fixed* Lloyd-Max codebook may avoid retraining that PQ needs,
   but TQ+ calibration and index caches still have cost.
4. **Progressive search**: coarse Palette256/HHTL or CAM-PQ candidates
   followed by U9 rerank only if pruning carries an admissible bound or a
   measured miss rate; otherwise no claim of exact top-k.
5. **QJL residual debias**: Google's TurboQuant paper adds a **separate
   1-bit residual stage** to correct inner-product bias. Compare
   i4 + 1-bit QJL against i9 and i8 under fixed memory and retrieval
   budget before concluding that more centroids always win. Our
   TurboVec9 scalar lab **does not** implement QJL.

DR512's complementary contribution is the **meaning of an update**:
presence/absence with one sentinel, touched-zero canonicalization,
signed value encoding and sparse read/write hygiene. It can transport
selected graph evidence, weights or states where the declared class
meaning fits, but **it does not by itself implement NSL learning,
gradient updates, or an unbiased dot-product estimator**.

## Measured receipts vs untested hypotheses

**Measured** on 2026-10-29 TurboVec9 scalar CI, deterministically
generated random unit vectors:

- n=512, d=64, q=32, k=10: scalar i4 Recall@10=0.78750,
  scalar i9=0.98438; payload 32 vs 72 B/vector; scan 149.8 vs
  268.8 us/query on AMD EPYC 7763 (Rust 1.99).
- n=2048, d=128, q=64, k=10: scalar i4=0.73750,
  scalar i9=0.97969; payload 64 vs 144 B/vector;
  scan 1.103 vs 2.102 ms/query.
- The shipped SIMD i4 is a **separate** baseline with calibration,
  so its speed cannot be interpreted as the overhead of adding bits.
- The first run says nothing about true graph-edge recall, gradients,
  convergence or production SIMD i9 performance.

**Historical** 2026-06-13 LG bridge: NativeLut 76,073 ns/query vs
PolyfillGemm 866,899 ns/query on n=20,000/d=512 (11.4×).
That compares *algorithm shapes*, but it predates current TurboVec
block-Hadamard, vector-major VNNI and current API; remeasure before
projecting ratios onto TurboVec9.

## Ordered work packages and stop conditions

| ID / destination | Concrete deliverable | Admission gate |
|---|---|---|
| **W1 / TurboVec** | #7: i4/i9 graph labels independent of quantized retrieval; compare f32, scalar i4, scalar i9, shipped i4 | 4-edge-any/coverage and f32 top-k vs same seeds; CI output checked |
| **W2 / LG NSL** | Bind *existing* `FeatureTable` + `NeighborBatch` to selected neighbor IDs; preserve the Google weight/order/dedup oracle | 0 changed `pack_nbrs` fixtures; no feature cloning; stale `SourceRef` refusal |
| **W3 / LG NSL** | Compare f32/DR512/i4/U9 graph loss and *gradient direction* on a fixed labeled graph, with held-out examples | Gradient cosine/sign, loss error, label accuracy across epochs; f32/TF reference |
| **W4 / ndarray** | Import `dot4_u8_i8_add` + nibble ops ONLY if needed by a proven consumer; parity all backends and saturating-AVX2 counterexample | Same results vs scalar at boundary values; VNNI/GFNI tiers gated |
| **W5 / LG bridge** | Compatibility audit against current TurboVec API, then version-aware thin adapter | Standalone `cargo test --manifest-path crates/lance-graph-turbovec/Cargo.toml` green |
| **W6 / cross** | Compare fixed-budget i4, i4+QJL, i8, i9 and DR512 on **different** respective use-cases | No apples-to-oranges byte count, direct accuracy, p95 latency and L1 behavior |

Do not open duplicate issues in ndarray/OGAR/Linear for the same
W-number without checking existing boards. **No existing production
semantics are relaxed by this research note.**

## Source references

- Google TurboQuant research: https://arxiv.org/abs/2504.19874 and
  https://research.google/blog/turboquant-redefining-ai-efficiency-with-extreme-compression/
- Google NSL:
  https://github.com/tensorflow/neural-structured-learning and
  https://www.tensorflow.org/neural_structured_learning/framework
- https://github.com/AdaWorldAPI/turbovec/pull/3 (scalar i9 codebook and baseline)
- https://github.com/AdaWorldAPI/turbovec/pull/4 (AVX2 facade route)
- https://github.com/AdaWorldAPI/turbovec/pull/7 (graph-edge falsifier)
- https://github.com/AdaWorldAPI/ndarray/pull/383 (canonical DR512)
- https://github.com/AdaWorldAPI/ndarray/pull/384 (synergy map)
- https://github.com/AdaWorldAPI/lance-graph/pull/1846 (NSL oracle)
- `crates/lance-graph-turbovec/KNOWLEDGE.md` (older measured counterexample)
- `crates/bgz-tensor/src/turboquant_kv.rs` (non-equivalent KV control)
