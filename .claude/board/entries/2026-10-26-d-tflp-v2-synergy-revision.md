# 2026-10-26 — D-TFLP v2: synergy revision after #1786–#1803

**Status:** PROPOSAL, docs only. Plan: `.claude/plans/d-tflp-tflite-palette-polyfill-capstone-v2.md`.

Reconciles v1 (#1785) with GPT's six review corrections, the 6.5 lab evidence (#1800 softmax
certification, #1801 8-bit admission, #1803 compaction), D-ONNX-INTAKE-0 (#1802), DT64 (#1794–#1799),
DR256/DQM (#1787–#1792) and ndarray #376/#377.

Changes to stage rows: D-TFLP-0 shipped; D-TFLP-2 measures table and fixed-point arms; D-TFLP-3 adds
padding = input zero point and `weights_offset` refusal; D-TFLP-4 ports the source reciprocal (13/14
known and 8/8 held-out calibrations require it, lab), exact reciprocal only behind a PROVED receipt;
D-TFLP-5a added; D-TFLP-5 carries cache key + receipt.

OPEN: generation identity owner (EXO-152); binding store (D-NFR S5); `Receipt` contract placement;
accumulator wrapping; full-runtime parity; target models; Mish subgraph validation; five unread TFLM
kernels.
