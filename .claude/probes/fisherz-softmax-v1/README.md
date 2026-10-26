# fisherz-softmax-v1

Probe for audit Amendment 4 (`.claude/board/entries/2026-10-26-d-tflp-parity-audit.md`):
softmax over FamilyGamma (Fisher-z) i8 codes, computed four ways.

- `probe.py`: reference (mpmath, 50 digits, on the real decoded representatives
  of `bgz-tensor/src/fisher_z.rs` `FamilyGamma::decode`), the f32 chain, a
  256-entry difference table with integer accumulation (F = 16 and 24
  fractional bits), and TFLite Micro's int8 SOFTMAX.
- `softmax_oracle.cc`: runs TFLite Micro's real `Register_SOFTMAX()` through
  `KernelRunner`; build it like ndarray `tools/tflm_oracle/README.md`
  (TFLM 8f6596d), replacing the source file.
- `results.json`: the run recorded in the amendment.

Run: `uv run probe.py <path-to-softmax_oracle> results.json` (about 75 s).
The reference never sees a difference table; the tables are built from f64
`exp(k*d)` and checked against it.
