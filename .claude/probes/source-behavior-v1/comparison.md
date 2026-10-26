# Findings (Amendment 6.2)

Raw results are in `results.json` and `compose.json`; this file only explains them.

## The 17 TANH differences have one cause

| stage | effect on the 17 cells |
|---|---|
| input rescale (`MultiplyByQuantizedMultiplier`) | exact on the 15 power-of-two cells (Q4.27 value 524288 = 2^19); −0.42 on the random calibration, moving the value AWAY from the boundary |
| `exp_on_negative_values` (gemmlowp fixed point) | returns e^(−2/256) low by **355.7** Q0.31 units; an exact division of this value already gives 8,388,744.6 > 2^23 = 8,388,608, the rounding boundary |
| `one_minus_x_over_one_plus_x_for_x_in_0_1` (Newton-Raphson) | −0.6 units, does not decide anything |
| `RoundingDivideByPOT(raw, 24)` | correct rounding of the value it receives |
| saturation | not involved (inside the radius) |
| float decode/encode | not on the path: the int8 path is integer throughout |

The ideal value is 8,388,565.3, 42.7 units **below** the boundary, so the correctly rounded code is 0;
the kernel gives ±1. Earliest divergence is the fixed-point exp, in all 17 cells.

## Two tables, two receipts

- **SourceParity** = TFLM bytes. It matches the kernel on 569,344 / 569,344 cells and reproduces
  the 7 committed fixture tables.
- **CertifiedMath** = interval-certified. It matches the independent mp.tanh reference on
  569,344 / 569,344 cells; f32 decode vs exact decode differ on 0 cells.
- **Overlap.** The two agree on 2,215 calibrations and differ on 9 (17 cells, Δ ∈ {−1: 9, +1: 8}).
  The receipts differ on every calibration, so a byte-equal table is still not interchangeable.

## Fingerprint economics (2,224 calibrations)

| representation | bytes | runtime |
|---|---|---|
| both complete tables + digests | 1,281,024 | direct lookup 0.28-0.52 ns/elem |
| SourceParity only | 640,512 | direct |
| math table + sparse source patch (count + (idx, byte) + digest) | 716,162 | overlay lookup 3.6-3.7 ns/elem (7-13× slower), or reconstruct once (9-13 ns per table) then direct |
| shared math table + keyed exceptions | 712,020 | same as above |

- Validation: sha256 of one 256-byte table costs 245-327 ns; all 4,448 tables take 1.2 ms cold.
- Python preparation: all source tables 0.10 s (kernel execution); all certified tables, with the
  independent reference and the f32 control, 123.5 s.
- **Verdict:** keep the 256-byte table at runtime. A patch saves 44% of cold bytes only when both
  contracts are stored, and must be reconstructed and digest-checked at load. It is never a
  per-element stage.

## Fusion keeps intermediate rounding

TANH (out 1/128, 0) → LOGISTIC (in 1/128, 0; out 1/256, −128), 569,344 cells:

| comparison | differing cells |
|---|---|
| fused table (from independently prepared stage tables) vs two-kernel sequence | **0** |
| stage-1 table vs sequence's intermediate tensor | 0 |
| certified math with the intermediate quantization (`math_iq`) vs sequence | 48,806 = 48,789 (TFLM LOGISTIC differs from correctly rounded at inputs ±1/128) + 17 (the TANH cells) |
| certified math without the intermediate quantization (`math_cont`) vs sequence | 69,652 |
| `math_cont` vs `math_iq` | 57,376 (the intermediate quantization itself) |
| control: TANH stage missing | 562,407 |
| control: intermediate declared 1/64 instead of 1/128 | 433,646 |

**Counterexample**, scale 1/128, zp 0, code −128 (x = −1):
- Source: 128·tanh(−1) = −97.48 → −97; then σ(−97/128) × 256 = 81.69 → 82 − 128 = **−46**.
- Continuous math: σ(tanh(−1)) × 256 = 81.49 → 81 − 128 = **−47**.
