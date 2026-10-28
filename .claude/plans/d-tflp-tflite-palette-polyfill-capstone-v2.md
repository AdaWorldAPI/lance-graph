# D-TFLP — TFLite models on the native polyfill (capstone v2: synergy revision)

**Date:** 2026-10-26. **Status:** PROPOSAL / docs only. No crate, kernel or ABI change.
**Supersedes:** v1 (`d-tflp-tflite-palette-polyfill-capstone-v1.md`, #1785) where this file says so.
v1 §1–§2 (the key fact, the asset map) and §5 (two code domains) stand unchanged and are not
restated. Every other v1 section is listed below as KEPT, MERGED, CLOSED, EXCLUDED or EXPANDED.
**Inputs reconciled:** GPT review of #1785 (six corrections); #1786–#1803 on `main` `13e370261`;
ndarray `master` `af4910a` (#376, #377).

## 0. What changed since v1, in one table

| source | what it adds to D-TFLP | effect |
|---|---|---|
| GPT review of #1785 | 6 corrections (exactness, fusion, conv padding, softmax algorithm, table cost, kernel/build parity) | folded into §2 below |
| #1800 softmax certification (lab) | certified preparation gate; TFLM-free replay crate; 14+8 calibrations decided | D-TFLP-4 rescoped (§3.3) |
| #1801 8-bit admission (lab) | `Receipt` / `Expect` / `bind`, 6×6 codec matrix, only the diagonal binds, 17 tests | prepared-object identity (§4) |
| #1803 | evidence compaction; run record regenerable, not kept | none on decisions |
| #1802 D-ONNX-INTAKE-0 | cold-frontend bake-off pattern, adversarial fixture list, ORT as oracle | same shape for the TFLite importer (§4, D-TFLP-5a) |
| #1794–#1799 DT64 | DT64-5 maps "TFLite int8 nonlinear → prepared code LUT, source-compatible"; DT64-5D = one ONNX/TFLite leaf matching D-NAM | shared leaf; keeps predicate bits and Palette256 apart (§6) |
| #1787–#1792 DR256 / DQM | DuckDB machine-room reuse plan cites v1 open point 5 (pair table vs fixed-point SIMD) and keeps "Softmax is not a unary table" | confirms §3.2, §3.3 |
| ndarray #376 / #377 | `masked_sum_i32` (dense branch-free / sparse walk), `mask_scatter_or_u16` | candidates for masked SUM/MEAN and u16 index maps (§3.5) |

## 1. Exactness claim (corrects v1 §1 items 1–3)

v1 said a table is "bit-exact by construction". Corrected: **a table is exact relative to its
generator.** It is bit-exact once its generator has been exhaustively compared against an
independently compiled TFLM reference, at the pinned revision, for that operator, parameter set and
build mode. A Rust port is not self-certifying. Every TABLE verdict therefore names (operator,
parameter set, TFLM revision, compiler/build mode) and the comparison that admitted it.

Chains stay closed only where TFLite requantizes to int8 between ops; a `DEQUANTIZE` is a float
boundary, not a closed table.

## 2. The six GPT corrections, where each lands

| # | correction | lands in |
|---|---|---|
| G1 | table exact only after exhaustive comparison to pinned TFLM, per op + params + build mode | §1; every D-TFLP gate |
| G2 | fusion needs matching intermediate quantization, wiring, no side effects, observable outputs; Mish fuses only if the imported subgraph is deterministic, single-input, elementwise | §3.1 Mish |
| G3 | conv padding: TFLM skips out-of-image positions (`conv.h:87-92,199-204`); dense GEMM + bias fold needs padded cells equal to the input zero point; test groups, dilation, stride, edges; refuse per-tensor `weights_offset ≠ 0` | §3.4, D-TFLP-3 gate |
| G4 | softmax is a table plus a per-row algorithm (`softmax.h:75-133`: `diff_min`, `kScaledDiffIntegerBits=5`, `kAccumulationIntegerBits=12`, `GetReciprocal`, `RoundingDivideByPOT`) | §3.3 |
| G5 | TABLE is a binding-time strategy; a 64 KiB pair table may lose to SIMD arithmetic or exceed MCU SRAM; measure | §3.2, §7 |
| G6 | parity is a pinned kernel/build claim; `simd_fixedpoint` double-rounding is the default, its shift/range contracts go into the oracle matrix | §5 |

## 3. Operator classes (v1 §3, revised)

Verdict vocabulary KEPT: ALIAS / INLINE / TABLE / NATIVE-TODO / REFUSE.

### 3.1 Unary → TABLE (KEPT, one correction)

`TANH`, `LOGISTIC` read. Still candidates (kernels not read): `HARD_SWISH`, `ELU`, the `RELU`
family, `QUANTIZE`/`DEQUANTIZE`. **Mish (corrected per G2):** a converted Mish is fused into one
256-entry table only after the importer shows the actual subgraph is deterministic, single-input and
elementwise with matching intermediate quantization; otherwise each builtin stays its own table, and
a fused custom op stays REFUSE.

### 3.2 Pair → TABLE or fixed-point SIMD (MERGED with v1 open point 5)

`ADD`, `MUL` read. The choice between a 65,536-entry table and direct `simd_fixedpoint` arithmetic is
made at binding time per target and measured (G5). Both are admissible only after the G1 comparison.
On MCU targets the 64 KiB table is refused when it does not fit the stated SRAM budget. The DQM
reuse plan cites this same open point; D-TFLP-2 answers it for both.

### 3.3 Softmax (REWRITTEN from #1800 evidence and G4)

v1 said "exp table + one reduction". Corrected:

- **Prepared once:** the `(admitted, A = exp_on_negative_values, B = Rescale<12>)` table over
  `d = −255..0`, from `(mult, shift)` of `PreprocessSoftmaxScaling(β, scale)`.
- **Per row:** `diff_min` skip, row sum `S` (Q12.19), reciprocal, `RoundingDivideByPOT`, output
  offset and clamp — the per-row algorithm, not a table.
- **Reciprocal:** default is the **source** reciprocal (`GetReciprocal`). The exact reciprocal is
  admitted only for a calibration the certified gate decided `PROVED_OUTPUT_EQUIVALENT`. Measured on
  main (lab): of 14 known calibrations, 1 proved (calibration 9) and 13 require the source reciprocal;
  all 8 held-out require it. The saving is about 26 ns per row on the reciprocal alone and is not
  resolvable for the whole recipe at n ≥ 256. **D-TFLP-4 therefore ports the source reciprocal; the
  exact path is an optional, receipt-gated specialization, not the plan.**
- **Domain:** int8 input, output scale 1/256, zero point −128, `S ∈ [2^19, 2^28)`; outside it, REFUSE.
- **Oracle:** the #1800 replay fixtures (529 full / 27 reduced witnesses, 7 equivalence rows, 10
  unreachable) become D-TFLP-4's CI oracle, so CI needs no TFLM build.

### 3.4 FC / conv (KEPT + G3)

Bias fold KEPT. Added (G3): padded positions must equal the input zero point before a dense GEMM is
equivalent; else the binding uses TFLM's skip semantics. Gate adds groups, dilation, stride, image
edges, and REFUSE for per-tensor `weights_offset ≠ 0`. Accumulator range (v1 open point 1) is MERGED
with #1801 OPEN item "accumulator wrapping": biased `VPDPBUSD`/`TDPBUSD` wrap by design; the gate
proves the model's `k` stays in range, and the `wrapping_*` proposal stays on its research branch.

### 3.5 Reductions and data movement (EXPANDED)

Masked `SUM`/`MEAN` over admitted rows can use ndarray `masked_sum_i32` (#376); `u16` index maps for
small-index `GATHER` can use `mask_scatter_or_u16` (#377). Both are **candidates**, exact only after
the G1 comparison; neither is asserted to match TFLM rounding.

### 3.6 Refused (KEPT)

Custom ops, Flex ops, control flow, dynamic shapes, int16 (16×8), sparse weights. EXPANDED with the
#1802 adversarial list where it applies to TFLite: unknown op/version, unknown options, mismatched
shapes, truncated buffers, int8/u8 signed zero, bool tensors.

## 4. Identity of prepared objects (REWRITTEN from #1800 + #1801)

v1 said "keyed by model digest, subgraph, index and parameter set". Corrected into two keys:

- **Numerical cache key:** what the table is a function of (softmax: `(mult, shift)`; unary: the
  requant parameters). Equal keys mean identical bytes, and a table may be shared. Two calibrations
  can prepare the same table (measured: known calibrations 1 and 2).
- **Semantic receipt:** what the result is issued for — codec signature, scale/zero point, operator,
  source revision and digest, codebook generation. Never derived from the cache key. #1801's
  `Receipt`/`bind` is the lab shape: of a 6×6 producer/consumer codec matrix, only the diagonal binds
  (rounding, code space, storage and NaN-policy mismatches all refuse).

A TABLE binding needs both: cache key for sharing, receipt for refusal. This replaces the v1 "never by
shape alone" rule (which still holds, Burn precedent).

**Not decided here:** who issues `codebook_generation` (EXO-152 / PZL-4); where a bound receipt lives
(D-NFR S5 binding store, MISSING); whether `Receipt` enters a contract type.

## 5. Reference truth (v1 §6, EXPANDED by G6)

TFLM reference kernels at fork `8f6596d`, compiled offline. Every parity claim names kernel, revision
and build mode. ndarray `simd_fixedpoint` double rounding is the default arm; its shift and range
contracts are copied into the oracle matrix. Full-runtime parity stays UNVERIFIED (v1 open point 2,
KEPT).

## 6. Boundaries with neighbouring plans (NEW)

- **D-ONNX-INTAKE-0 (#1802):** same cold-frontend rules apply to the TFLite importer: no native crate
  depends on it; unknown nodes refuse; importer IDs are not stable source identities; no automatic
  conversion to Palette256.
- **DT64-5 (#1798):** TFLite int8 nonlinear is a "prepared code LUT, source-compatible" method leaf.
  Execution metadata (code space, −128 policy, rounding) never enters the 64 DT64 predicate slots;
  Palette256 (256 codes) and the predicate catalogue (64) are distinct. DT64-5D ("one ONNX method leaf
  matching D-NAM") and D-TFLP-5 share the leaf contract.
- **DR256 / DQM (#1787–#1792):** RelAlg is not extended with Tanh, MatMul or Conv; nothing in the hot
  path matches a TFLite operator name. Unchanged.
- **D-NAM-PZL (#1784):** domain B, unchanged (v1 §5).

## 7. Stages (v1 §7, revised)

| D-id | change vs v1 | gate |
|---|---|---|
| D-TFLP-0 | shipped (#1785); this v2 is its revision | docs |
| D-TFLP-1 | KEPT; first experiment | G1 exhaustive vs TFLM `tanh.h`/`logistic.h`, ≥5 param sets, disable run red |
| D-TFLP-2 | MERGED with open point 5: table **and** fixed-point arm, keep the faster per target | G1 over all 65,536 pairs; measured ns per arm; SRAM budget refusal |
| D-TFLP-3 | + G3 padding/zero point, groups/dilation/stride/edges, `weights_offset` refusal, wrap range | G1 + G3 cases |
| D-TFLP-4 | REWRITTEN: source reciprocal + per-row algorithm; exact reciprocal only behind a PROVED receipt | #1800 replay fixtures pass; `--reciprocal exact` negative control fails 556 witnesses |
| D-TFLP-5a | NEW: TFLite intake bake-off (FlatBuffer reader options), #1802 pattern | coverage report, adversarial fixtures |
| D-TFLP-5 | facade; prepared objects carry cache key + receipt (§4) | identity tests incl. same-table/different-receipt |
| D-TFLP-6..8 | KEPT | as v1 |

Order: 1 → 2 → 4 in parallel with 3; 5a before 5; 6 once 5a reads op lists.

## 8. Costs (v1 §8, KEPT + G5)

Add per pair op: table bytes vs SRAM budget, and table vs fixed-point ns per element per arm. Softmax:
reciprocal ns source vs exact, per-row cost by `n`, gate cost per calibration (lab: median 291 ms) and
certificate cost per source revision (lab: 5.4 s).

## 9. Open points (v1 §9, revised)

| v1 # | now |
|---|---|
| 1 accumulator range | MERGED into §3.4 with #1801 "accumulator wrapping"; still OPEN |
| 2 full-runtime parity | OPEN, unchanged |
| 3 target models | OPEN, unchanged |
| 4 int16 | EXCLUDED from first scope, unchanged |
| 5 pair table worth it | MERGED into D-TFLP-2; also cited by the DQM reuse plan |
| — | NEW OPEN: generation identity owner (EXO-152); binding store (D-NFR S5); `Receipt` contract placement |
| — | NEW OPEN: Mish subgraph validation from a real converted model (G2) |
| — | NEW OPEN: reading the five unread TFLM kernels (`hard_swish`, `elu`, `squared_difference`, `maximum_minimum`, `comparisons`) |

## 10. Closed and excluded (for the record)

CLOSED (by evidence on main, lab scope): whether the exact reciprocal can replace `GetReciprocal`
(answered per calibration; mostly no); whether a codec-space mismatch is caught at binding (yes, 6×6
diagonal); whether a table can silently belong to another calibration (refused,
`table_identity_mismatch`).

EXCLUDED: a TFLite runtime dependency; a tensor AST; an ONNX/TFLite → RelAlg graph; automatic domain
A → B conversion; a Fisher-Z round trip in domain A.
