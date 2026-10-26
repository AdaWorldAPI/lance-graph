# source-behavior-v1

Amendment 6.2 lab: SourceParity vs CertifiedMath tables for TFLite Micro int8 TANH,
a stage trace of their 17 differences, and source-behavior-preserving fusion of
TANH → LOGISTIC. Lab only. No production API; the receipts here are lab keys, not an ABI.

| file | role |
|---|---|
| `source_trace.cc` | TFLM harness. `trace` re-executes the int8 TANH loop body stage by stage with TFLM's own fixed-point functions and aborts unless every byte equals the real kernel's; `logistic` runs `Register_LOGISTIC`; `chain` runs TANH then LOGISTIC as two kernel calls on one int8 tensor. |
| `prepare.py` | Both TANH tables per calibration (2,224, from `../tanh-sweep-v1/sweep.json`); the independent reference; discrepancy trace; receipts; fingerprints; negative controls → `results.json`, plus `artifacts.bin` (NOT committed, 1.4 MB, sha256 `db55e763…6be2`, regenerates identically). |
| `compose.py` | The fused table vs the two-kernel sequence vs certified math with and without the intermediate quantization; missing-stage and wrong-calibration controls → `compose.json`. |
| `parity/` | Rust probe: reads `artifacts.bin`; checks digests; reconstructs from a sparse patch; checks the index mapping; runs admission refusals; times lookups. Links no TFLM and no exp/tanh. `DISABLE=patch` and `DISABLE=xor` are its falsifiers. |
| `comparison.md` | Findings, counterexamples and costs. |

The source oracle is reused from `../tanh-sweep-v1/tanh_int8_sweep.cc`.

**Pins.**
- TFLM checkout `5f64e4ad`; `tensorflow/` tree identical to `8f6596d` (`git diff 8f6596d HEAD -- tensorflow/` is empty).
- Reference kernels: x86_64 gcc 13.3.0, flags as in `../tanh-sweep-v1/README.md`.
- Python 3.11.17, mpmath 1.4.1, numpy 2.4.6.
- rustc 1.99.0, sha2 0.10.9.
- Calibration seed 61 (inherited from the sweep).

```sh
# in the TFLM checkout, the same g++ line as ../tanh-sweep-v1/README.md with source_trace.cc -> source_trace
uv run prepare.py ./tanh_sweep ./source_trace results.json artifacts.bin
uv run compose.py ./tanh_sweep ./source_trace compose.json
(cd parity && cargo build --release && ./target/release/source-behavior-parity ../artifacts.bin)
DISABLE=patch ./parity/target/release/source-behavior-parity artifacts.bin   # must report reconstruct failures
DISABLE=xor   ./parity/target/release/source-behavior-parity artifacts.bin   # must report index failures
nm parity/target/release/source-behavior-parity | grep -ciE 'tflite|tanh|gemmlowp'   # must be 0
```

## Amendment 6.3

| file | role |
|---|---|
| `trace63.cc` | TFLM harness. `ltrace`: LOGISTIC int8 stage trace, aborts unless byte-equal to the real kernel on all 256 codes. `softexp`: softmax prepare (`PreprocessSoftmaxScaling`, `CalculateInputRadius`) and the per-difference source weights `exp_on_negative_values` and `Rescale<12>` for d = −255..0. `recip`: TFLM `GetReciprocal`. Only TFLM's own primitives are called. |
| `attrib63.py` | A: LOGISTIC ±1/128 attribution against independent ideals. B: four-arm (SS/SC/CS/CC) attribution of the 569,344 TANH→LOGISTIC cells from the 6.2 `artifacts.bin` (sha256 checked) → `attrib63.json`. |
| `softmax63.py` | C: the TFLM int8 SOFTMAX oracle (reused `../fisherz-softmax-v1/softmax_oracle.cc`), a source-compatible prepared recipe (source weights + integer reduction + the reciprocal, re-implemented and checked against TFLM's), the F24 exact-first/exact-zero arm, and an independent reference → `softmax63.json`. |
| `recip_search63.py` | Searches rows where the source reciprocal, not an exact one, decides a byte; confirms each hit with the kernel → `recip63.json`. |
| `softdiff63.py` | Stage attribution of the softmax cells where TFLM differs from the math contract: one source stage at a time replaced by exact arithmetic → `softdiff63.json`. |

Build `trace63.cc` with the same g++ line as `source_trace.cc`. Build `softmax_oracle.cc` with that line plus
`-Wno-error=unused-result`, because the committed oracle ignores a `scanf` result.

```sh
uv run attrib63.py ./trace63 artifacts.bin attrib63.json
uv run softmax63.py ./softmax_oracle ./trace63 softmax63.json
uv run recip_search63.py ./softmax_oracle ./trace63 recip63.json
uv run softdiff63.py ./trace63 softdiff63.json
```

## Amendment 6.3 addendum (backend / observability)

| file | role |
|---|---|
| `backend63.py` | Links the three unchanged oracle sources against each TFLM library variant (`gen_<variant>/`, same checkout, built with `make … microlite BASE_GENDIR=gen_<variant>` and the flags in `backend63.json`) and compares output bytes: 2,224 TANH and 2,224 LOGISTIC calibrations, the 840 softmax rows, and 14 overflow rows (ties, n = 4,095 … 16,385) → `backend63.json`. |
| `prop63.py` | Error propagation TANH → LOGISTIC: internal vs observable error, fan-in onto intermediate codes → `prop63.json`. |
| `recipwit63.cc` + `recipwit63.py` | Reciprocal observability on reachable rows `[max]*k + [max+d]*m`; witnesses re-run through the real kernel; perturbed-reciprocal negative control → `recipwit63.json`. |
| `recipall63.cc` + `recipall63.py` | Exhaustive superset check: every S in [B[0], 2^28) × every admitted weight → `recipall63.json`. |
| `reach63.cc` + `reach63.py` | Turns each superset candidate into a real int8 row (min-coin DP over admitted weights) and runs it through the TFLM kernel → `reach63.json`. `recipall63_control_delta256.json`: the same superset check with the reciprocal perturbed by +256 (negative control). |
| `reachcheck63.py` | Independent check of every witness with the `softmax63.py` Python recipe. |
| `perf63.cc` | Kernel Invoke vs prepared lookup / source-compatible softmax recipe, with output parity on every timed buffer. |
| `lutperf/` | Scalar 256-byte lookup vs the same lookup composed from ndarray `U8x64::{permute_bytes, cmpgt_mask, mask_blend}`. |

Variant flags (appended after TFLM's own; kernels are `KERNEL_OPTIMIZATION_LEVEL`, everything else
`CORE_OPTIMIZATION_LEVEL`): `gccO0` `-O0`/`-O0`; `gccO2novec` kernels `-O2 -fno-tree-vectorize
-fno-tree-slp-vectorize`; `gccO3native` `-O3 -march=native`/both; `clang` `TOOLCHAIN=clang
CXX_TOOL=clang++ CC_TOOL=clang` with `-Wno-error=double-promotion` appended (clang rejects a float→double
promotion in `micro_utils.h` under `-Werror`); `ubsan` `-O1 -fsanitize=undefined
-fno-sanitize-recover=undefined`; `ubsanshift` the same with `-fno-sanitize=signed-integer-overflow`.

## Amendment 6.4 follow-up (source domain gate, reciprocal levels, pruning)

| file | role |
|---|---|
| `ubsanctl64.cc` | Can-fire / can-stay-silent control for the UBSan flags: `add` (4,096 maxima, `fixedpoint.h:94`) and `shift` (`GetReciprocal(0)`, `common.h:976`) must be reported; `none` (511 maxima) must exit 0 → `ubsanctl64.txt`. |
| `recipall64_ubsan.json` | `recipall63.cc` and `trace63.cc` rebuilt with `-O1 -fsanitize=undefined -fno-sanitize-recover=undefined` against `gen_ubsan/`, run over the full admitted superset of all 14 calibrations (`recipall63.py`, unchanged driver). |
| `reach64.cc` + `reach64.py` | Exact reachability of every superset candidate: residue rule mod B[0] (Dijkstra over the admitted weights), so each pair is reachable or PROVED unreachable; rows are re-checked (positive counts, admitted codes, sum = S) before the kernel runs → `reach64.json`. |
| `reach64_xcheck.py` | The residue rule against a plain unbounded-coin DP on [0, 2^21], per calibration → `reach64_xcheck.txt` (mismatches must be 0). |
| `recipprune64.cc` + `recipprune64.py` | The superset search pruned to S intervals around the visible quantization thresholds, using a measured reciprocal-error bound per exponent band; the candidate set must equal the exhaustive one → `recipprune64.json` (exact arm), `recipprune64_control_delta256.json` (negative control), `recipprune64_window_sweep.json` (window disable runs). `recipall64_control_delta256_full.json`: the exhaustive δ = 256 control re-run listing every candidate (the 6.3 file kept 10 samples). |

`recipprune64.cc` and `ubsanctl64.cc` build with the `trace63.cc` g++ line (`ubsanctl64` with the UBSan flags
and `-Igen_ubsan/…/genfiles/`); `reach64.cc` is plain C++17.

```sh
uv run reach64.py ./reach64 ./trace63 ./softmax_oracle recipall63.json reach64.json
uv run reach64_xcheck.py ./reach64 ./trace63 reach64_xcheck.txt
uv run recipprune64.py ./recipprune64 ./trace63 recipall63.json recipprune64.json
uv run recipprune64.py ./recipprune64 ./trace63 recipall64_control_delta256_full.json recipprune64_control_delta256.json 256
uv run recipprune64.py ./recipprune64 ./trace63 recipall63.json /tmp/w.json 0 <window>     # disable runs
```

## Amendment 6.5 (admission gate, fixtures, economics)

| file | role |
|---|---|
| `softcert65.cc` | `global`: source-global certificate of the TFLM int8 softmax reciprocal (domain, monotone exact reference, E_h per band) → `softcert65_global.txt`. `gate`: per-calibration decision PROVED_OUTPUT_EQUIVALENT / REQUIRES_SOURCE_RECIPROCAL / REFUSE_UNKNOWN from that certificate and one calibration's prepared weights; boundary-pruned search plus exact reachability; built-in mutation switches. |
| `softcert65.py` | Runs the gate on the 14 known and 8 held-out calibrations and validates every decision without the gate's code; `--mutations` runs 10 certifier and 3 input mutations → `softcert65_run.json`. |
| `heldout65.py` | 8 held-out calibrations and their exhaustive superset reference (unchanged `recipall63`) → `recipall65_heldout.json`. Two of the eight have the same β·scale as a known calibration and therefore the identical weight table; six are genuinely new. |
| `fixtures65.py` → `fixtures65/` | Durable witness fixtures and their SHA-256 manifest; replayed without TFLM by `../pal256-admission-v1/replay`. |
| `perf65.cc` → `perf65.txt`, `perf65_runs23.txt` | Kernel vs source recipe vs exact-reciprocal recipe vs the reciprocal alone, calibrations 9 and 10, n = 8 … 511, three runs, parity beside every arm. |

The certifier and the benchmark build with the `trace63.cc` g++ line (`perf65` links the TFLM library).

```sh
./softcert65 global 5f64e4ad $(cat <tflm>/…/{common.h,fixedpoint.h,reference/softmax.h} | sha256sum | cut -c1-64) > softcert65_global.txt
uv run heldout65.py ./recipall63 ./trace63 recipall65_heldout.json 3
uv run softcert65.py ./softcert65 ./trace63 ./softmax_oracle softcert65_run.json --mutations
uv run fixtures65.py softcert65_run.json ./trace63 ./softmax_oracle fixtures65
```
