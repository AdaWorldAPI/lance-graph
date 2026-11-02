# D-XLOG-NATIVE-5 — XLOG bag count bound through the production bind store; Epi5 reading of XLOG results

**Date 2026-10-31 · Status IN PR · Plan `.claude/plans/d-xlog-native-v1.md` Amendment A**

- `crates/xlog-relcomp-probe/tests/dto_binding.rs` routes, seals and admits through `lance-graph-dt64` (`BindStore`, `dr::FrameStore`, `MethodRef`). The count is three steps over two polyfill words, named by position in one lab polyfill lens: mask-risc's keyed count twice, then `checked_binary_i64(Mul)`. Refusals: stale frame (`MapChanged`), retracted position (`Unrouted`). Execution stays in the lab register, which refuses the admitted Bag population (`BagNeedsMultiplicity`).
- `tests/epi5_reading.rs` covers XLOG results written as `CausalEdge64` bits 59..63:
  - a base fact is `Direct × Open`; a composed edge with Y retained is `IndirectKnown × Open`;
  - the reading is refused when undeclared or under `V1Legacy` provenance;
  - an unstamped composed edge reads as `Direct`.
- Plan Amendment A: a binding names mask-risc's op vocabulary (addenda 8 and 9); the `ndarray::simd` column is what an executor would call.

**OPEN:** method identity (`MethodSignature` / `MethodRef` / `MethodKey`, EXO-50); frame catalogue extent (D8); production classids and the XLOG Epi5 declaration (OGAR); many-to-many join home (D-XLOG-NATIVE-3).
