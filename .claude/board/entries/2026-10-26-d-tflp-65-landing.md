# 2026-10-26 — D-TFLP 6.5 landing: softmax evidence + 8-bit admission

**Status:** MEASURED (lab). Docs and probes only; no crate, ABI or kernel change. Does not move any
`D-TFLP-n` status row: D-TFLP-4 (softmax in ndarray) stays Queued.

The Amendment 6.5 lab work moves from research branch `claude/bpe-tokenization-architecture-3xd4eh`
(not rebased) to `main` through two **merged lab PRs** (2026-10-26):

- **[#1800](https://github.com/AdaWorldAPI/lance-graph/pull/1800), merged**: softmax fixture replay (runs under `cargo test`, no TFLM) and the certified preparation
  gate. While landing, the gate gained a table↔calibration check: it recomputes the table from the
  claimed (scale, β) and refuses `table_identity_mismatch`. It also gained separate outputs for the
  numerical cache key `(mult, shift)` and the semantic receipt. Re-run results: 601/529/72 known and
  514/446/68 held-out, identical to the research run; fixtures regenerate byte-identical.
- **[#1801](https://github.com/AdaWorldAPI/lance-graph/pull/1801), merged**: `pal256-admission-v1/admit` (17 tests). The identity claim is narrowed to
  TESTED_COMPATIBLE_FOR_PINNED_CODEC_SET: 21 instantiations on ndarray `af4910a1`, with pairwise-distinct
  signatures. `HANDOFF.md` maps it onto D-NAM §2 / PZL-4, the D-NFR S5 binding store (MISSING) and
  the **canonical D-DT64-Q0** source-attested borrowed seam ([#1799](https://github.com/AdaWorldAPI/lance-graph/pull/1799)); DT64-QB is only a compatibility pointer.

OPEN: generation identity owner (EXO-152); binding store (D-NFR S5); contract placement of `Receipt`;
accumulator-wrapping semantics (proposal kept on the research branch, in no landing PR).


**TANH evidence is separate from D-TFLP-1 implementation status.** [Research branch](https://github.com/AdaWorldAPI/lance-graph/tree/claude/bpe-tokenization-architecture-3xd4eh) [commit `4082dbb79`](https://github.com/AdaWorldAPI/lance-graph/commit/4082dbb79) records 569,344-cell TANH source-parity and independently certified mathematical tables, the 17 traced differences, and the tested TANH→LOGISTIC fusion. See [dedicated evidence/learning/Tract-harvest entry](2026-10-26-d-tflp-tanh-proof-learning-tract-harvest.md). D-TFLP-1 native ndarray integration remains QUEUED, while existing lab certification is DONE within its stated domain. Adaptive-learning generations, 8-byte FamilyGamma source receipts and Tract cold-intake tests remain OPEN.
