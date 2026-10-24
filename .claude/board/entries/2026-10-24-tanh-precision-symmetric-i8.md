# 2026-10-24: f32 tanh precision and symmetric signed-i8 quantization

**Status:** MEASURED (synthetic) for A, B and C. D (real model data) is
**NOT RUN**. Nothing here changes a wire format or an existing decoder.

## A — `vstanh` (ndarray PR #369, branch `fix/vstanh-small-arg`, head `4f6c132`)

The old kernel computed `2·sigmoid(2x) − 1`. That cancels: every |x| below
about 1e-7 came back as 0, −0.0 came back as +0.0, and the error reached
868,220,004 ULP. Measured over all 2³² f32 inputs on AVX-512:
1,728,053,246 nonzero inputs zeroed and 975,101,106 odd-symmetry violations.

The new kernel works on |x|:

- |x| < 0.625: `a + a·z·P(z)`, z = a², Cephes tanhf coefficients;
- otherwise `1 − 2/(exp(2a) + 1)`;
- the sign is restored by a float select, and x == 0 returns x, so −0.0 is kept.

`simd_exp_f32` was 39 ULP off (a degree-5 Taylor series). It is now a Cephes
expf minimax with unfused `*` and `+`, so it stays bit-equal across backends,
and measures 1 ULP over 4 M points on [−87, 88.5].

Exhaustive result over 2³² inputs (AVX-512):

- max 1 ULP in every magnitude band against f64 `tanh` rounded once;
- 0 zeroed, 0 odd-symmetry violations, 0 values outside [−1, 1], 0 monotonicity violations;
- NaN preserved; 1 flip-risk input, which agrees with mpmath;
- the oracle matches mpmath correct rounding on 200,000/200,000 sampled inputs.

Measured in the same run: `vstanh` 3.06 ns/elem, the old kernel 2.46,
scalar `f32::tanh` 11.66.

The probe gives the same numbers on v4, v3, the x86-64 baseline, and NEON
under qemu. wasm parity is checked for exp only. The ndarray gates pass (2590
lib tests, clippy, fmt). Disable runs D1 to D5 each turn the gate red.

Not run: tanh under the wasm harness, the nightly-simd arm, and the
exhaustive sweep on the non-AVX-512 tiers.

## B — symmetric signed i8 (`crates/bgz-tensor/tests/symmetric_i8_quantization_probe.rs`)

The proposed code space has −127..127 as values and −128 as an invalid or
NaN sentinel. Values are encoded with round-half-to-even on `x/scale`,
saturating at ±127. The probe is pure math and does not touch `FamilyGamma`.

| hypothesis | result |
|---|---|
| H1: zero is the exact centre of the valid range | holds for the proposal (127 values each side). Fails for conventional i8: 128 vs 127, and −128 has no negation (`checked_neg` is None, `saturating_neg` gives 127, `unsigned_abs` gives 128) |
| H2: `encode(−x) == −encode(x)` | holds for the proposal over ±200 codes, the half-codes and random x. Fails for `floor(x + 0.5)` and for any affine zero point |
| H3: data is median-zero | is a property of the data, not the codec. Falsified for every non-centred family below |
| H4: a symmetric code removes the half-bin offset, not calibration offset | holds. On shifted or skewed data affine beats symmetric by about 2× RMSE (gaussian +4σ: 1.06e-2 vs 2.12e-2; exponential: 1.34e-2 vs 2.68e-2), and symmetric uses fewer than 128 of its bins there |

Over all 256×256 code pairs, the sentinel-propagating saturating add is
commutative and odd, and never produces −128 from valid inputs.
Conventional `saturating_add` breaks oddness on 16,256 of the 255×255
negatable pairs.

Distribution table (13 seeded distributions × 4 schemes; metrics are bias,
median, MAE, RMSE, max error, zero preservation, saturation, bins, entropy,
Spearman and near-zero error): see the test's `--nocapture` output.
Highlights:

- **Symmetric** keeps exact zeros on every sparse set (90 / 99 / 99.9 % zero).
  On centred data it is within 1 % of affine RMSE (9.32e-3 vs 9.23e-3).
- **Affine** wins on shifted, skewed and bimodal data. It also keeps zero
  exactly when zero lies inside the range, because the zero point is integer.
- **Unsigned 7-bit** and **legacy truncation** both lose every exact zero.
  Truncation adds a signed bias of up to +8.7e-2 on heavy tails, and doubles
  MAE on uniform data.

Disable runs, all red: round half away from zero in place of ties-to-even;
no sentinel propagation in add; add clamped to −128.

### B4 — the FamilyGamma wire contract

- `encode` = `(atanh(clamp(c, ±0.9999)) − z_min)/z_range·254 − 127`,
  clamped to −128..127, then `as i8`, which truncates toward zero.
  `decode` = `tanh((c + 127)/254·z_range + z_min)`.
- **−128 is a valid saturated value code.** `FisherZTable::build` never
  stores it, because the fit covers the table's own off-diagonal and the
  diagonal clamps to +127 (checked over 300 random tables). But `encode`
  emits it for every input below the fitted range: cosine 0 in an
  all-positive family, or a held-out pair. It decodes to the finite value
  `tanh(z_min − z_range/254)`.
- Code 0 is the family's mid-z, not cosine 0. In an all-positive family it
  decodes to 0.656. `lookup_i8` returns 0 for an unknown centroid, which is
  therefore not neutral in such a family.
- `as i8` truncation moves codes half a bin toward code 0 on both sides
  (measured −0.501 above zero and +0.501 below).
- The 8-byte gamma plus k² raw i8 carries no version byte. It is written
  durably: thinking-engine `tts_17b_hhtld_encode.rs:676-680` stores
  `{role}.fisher_z` and `{role}.fisher_z_gamma` in safetensors, and
  `tts_17b_hhtld_decode.rs:298-300` reads them back.
  `palette256_reference_contract.rs` pins `to_bytes` byte for byte.

**Conclusion:** a −128 sentinel cannot be introduced in place. Doing so would
change the meaning of a byte that existing artifacts may contain (codes
encoded below the range). Any symmetric codec needs a new, versioned format
and leaves `FamilyGamma` and its decoder untouched. Not proposed for merge
here; this is a decision for the format's owner.

Other i8 conventions seen in this workspace (not changed):

- thinking-engine `signed_engine` and `tensor_bridge` round and clamp to −128..127;
- `l4.rs` maps `i8::MIN` to magnitude 127;
- bgz-tensor `zipper`, `slot_l`, `matryoshka` and `xor_adaptive` clamp to −127..127.

The workspace has no single i8 convention.

## C — error decomposition (`crates/bgz-tensor/examples/fisher_z_error_decomposition.rs`)

Each pair's round-trip error is split into stages that sum exactly to the
codec total; this is asserted per pair. Mean |e|, k = 256, d = 64:

| family | calibration | rounding | truncation (extra) | f32 decode | tanh kernel (vstanh − f32::tanh) | total |
|---|---|---|---|---|---|---|
| centred | 2.2e-9 | 1.0e-3 | 2.0e-3 | 1.5e-8 | 3.2e-9 (≤2 ULP) | 2.0e-3 |
| all-positive | 1.5e-8 | 4.7e-4 | 9.3e-4 | 3.4e-8 | 3.1e-8 (≤2 ULP) | 9.3e-4 |
| held-out gamma | 2.6e-5 (max **3.6e-2**) | 3.2e-4 | 6.5e-4 | 3.1e-8 | 3.0e-8 | 6.7e-4 (max 3.6e-2) |
| high similarity | 1.5e-8 | 1.7e-5 | 3.4e-5 | 1.5e-8 | 0 | 3.4e-5 |
| near-duplicate | 7.3e-5 | 0 | 0 | 0 | 0 | 7.3e-5 |

The ranking is: calibration on out-of-range inputs, then truncation, then
quantization, then f32 arithmetic and the tanh kernel. The tanh kernel sits
five orders of magnitude below quantization. `vstanh` is not on the codec's
path today, and batching the decode through it would cost nothing measurable.

The held-out case is the realistic risk: 80 of 32,640 pairs fall outside the
fitted range, 27 of them land on −128, and none is flagged. When every
cosine of a family exceeds 0.9999, the family collapses to one z.

## D — real-data validation manifest: **NOT RUN**

| # | input | command | measures | status |
|---|---|---|---|---|
| D1 | jina-v3 Q8_0 GGUF (`gaianet/jina-embeddings-v3-GGUF`, snapshot `d7d998aab1a7f2ea0aa256d8b3f035cbd0af682a`, the default path in `bgz-tensor/examples/gguf_families.rs`) | build `FisherZTable` per role group from real representatives, then the C decomposition | stage errors on real families | NOT RUN — file not on disk; record its sha256 when fetched |
| D2 | the same GGUF | held-out split per group (fit on half the rows, evaluate on the rest) | how often real data reaches −128 or the clamp | NOT RUN |
| D3 | existing thinking-engine `*.fisher_z` safetensors artifacts | histogram of stored codes | whether any durable artifact holds −128 | NOT RUN — no artifact on disk |
| D4 | Jina v5 safetensors (the workspace ground truth) | same as D1 | the same on a second model | NOT RUN |
| D5 | ONNX Runtime, offline | `Tanh` and `Exp` operator tests against `vstanh` and `vsexp` | operator conformance | NOT RUN — blocked on ndarray #369 merging |

Each run must record: the file sha256, the git SHA of lance-graph and
ndarray, the target-cpu the arm printed, and the exact command.

## Proposed handoff (not posted to EXO-50)

1. Merge ndarray #369 (Workstream A).
2. Then the lance-graph-onnx `Tanh` → `vstanh` and `Exp` → `vsexp` alias,
   removing the lib.rs:35-36 sentence that calls them approximate, and
   adding operator tests (Stage 4).
3. The decision to make: whether a versioned symmetric Fisher-Z format is
   wanted. If yes, it gets a new format tag and its own reader;
   `FamilyGamma` and its decoder stay as they are.
4. Separately: report out-of-range codes instead of clamping silently, and
   run D1–D3.
