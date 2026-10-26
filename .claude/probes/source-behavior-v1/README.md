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
