# 2026-10-26 — D-TFLP 6.5 landing: softmax evidence + 8-bit admission

**Status:** MEASURED (lab). Docs and probes only; no crate, ABI or kernel change. Does not move any
`D-TFLP-n` status row: D-TFLP-4 (softmax in ndarray) stays Queued.

The Amendment 6.5 lab work moves from research branch `claude/bpe-tokenization-architecture-3xd4eh`
(not rebased) to `main` as two draft PRs:

- **#1800**: softmax fixture replay (runs under `cargo test`, no TFLM) and the certified preparation
  gate. While landing, the gate gained a table↔calibration check: it recomputes the table from the
  claimed (scale, β) and refuses `table_identity_mismatch`. It also gained separate outputs for the
  numerical cache key `(mult, shift)` and the semantic receipt. Re-run results: 601/529/72 known and
  514/446/68 held-out, identical to the research run; fixtures regenerate byte-identical.
- **this PR**: `pal256-admission-v1/admit` (17 tests). The identity claim is narrowed to
  TESTED_COMPATIBLE_FOR_PINNED_CODEC_SET: 21 instantiations on ndarray `af4910a1`, with pairwise-distinct
  signatures. `HANDOFF.md` maps it onto D-NAM §2 / PZL-4, the D-NFR S5 binding store (MISSING) and
  DT64-QB (#1799).

OPEN: generation identity owner (EXO-152); binding store (D-NFR S5); contract placement of `Receipt`;
accumulator-wrapping semantics (proposal kept on the research branch, in no landing PR).
