# D-DYNBAG-X: cross-frontend projection crosswalk

**Date 2026-10-31 · D-id D-DYNBAG-0 (scope) · Status OPEN (map, no code)**

Read-only harvest of XLOG, ONNX/Tract, SQL/R2IL, egglog-MLIR, Codegraph (rust-analyzer + Charon) and Scallop against DT64, ClassView, Quack, relalg, R2IL and mask-risc. The full table and the gaps are in `.claude/plans/d-dynbag-holographic-bags-v1.md` §18.

- **MEASURED (read):** no component executes a DT64 `MethodBinding`; `BindStore::admit`/`resolve` have no production caller (`lance-graph-dt64/src/lib.rs:796`; only dt64 tests and `xlog-relcomp-probe/tests/dto_binding.rs`).
- **MEASURED (read):** the Tract probe is the one place where the resolved binding is the one dispatched (`tract-harvest-probe/src/plan.rs:116-128,171,204`), but through its own `Registry`, not `BindStore`.
- **MEASURED (read):** `admit` checks method id, `SigRev` and guard witness only (`lib.rs:796-822`); it cannot refuse on an effect or on POSSIBLE/UNKNOWN evidence.
- **MEASURED (read):** "project" means columns (XLOG RIR), rows kept as a mask (relalg `Keep`, Quack `Rows`) and field labels (`ClassView::project`, 64-bit `FieldMask` only; `Resolved::field_view`, `WideFieldMask`). Three owners' words, not one operation.
- **OPEN:** the executor seam that turns an admitted binding into the Quack/R2IL description that runs (§18.4 X1); without it no two consumers share a binding contract.
- **Address idea `[DD:CC][F###]`:** the concept half is domain:concept, not domain:app (`ogar_codebook.rs:179`); the low half is the app lens (`0x0000`–`0x000C`, V3 marker `0x1000`). A per-method slot in the classid would be a second spelling of `MethodRef { class, MethodId }`, which DT64 rules out ("a codebook item never gets a classid", `lib.rs:51-65`). PROPOSED only; OGAR decision.

**Fruit:** X2 (effect/evidence refusal in `admit`) is small, owned by DT64, and unblocks the codegraph fixture.
