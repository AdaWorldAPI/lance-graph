# 2026-10-23 — D-ONNX-RELALG-CAP — S0 result: ONNX as import membrane, DR64³ plan not found

**Status:** S0 documentation; no code. Plan: `.claude/plans/d-onnx-relalg-cap-v1.md`. Read at `origin/main` `59538865`; ONNX facts at upstream commit `0c8af065` (ONNX 1.23.0 = IR 14, ai.onnx opset 28; default domain 203 operators, 155 ops + 48 functions; `ai.onnx.ml` 19).

## What was searched

- The repo for `DR64³`, `DR256³`, `64×64×64`, `64x64x64` (`grep -rIl`).
- Branches and PRs: branch `ccr-e97e5d58-1bkumg-cartesian12`; GitHub PR search "cartesian".
- Rust crates for any ONNX dependency (onnx/ort/tract/burn/rten/onnxruntime/wonnx/inkwell/llvm/cranelift) and for `ModelProto`/`GraphProto`/opset in `*.rs`.
- relalg, R2IL, mask-risc, contract set algebra, `dr-math-binding-probe`, `dir-rmap-probe` (read with file:line).

## What was found

- DR64³ / DR256³ original plan: NOT FOUND. 0 files contain any of the strings. The branch above is CartesianAddress12/Quad8 shader work. 23 "cartesian" PRs, none is a DR64³ plan. The 12-stage PR sequence is not found. The one 18-bit figure in the repo is the "18-bit hot envelope = 262,144" mailbox working set (`.claude/knowledge/delta-card-addressing-integration-map.md`), a working-set size, not an identity. Other 262144 hits are byte counts, CSV values and index spaces.
- The plan's DR64³ section is a NEW, labelled proposal (R1 chain contraction, R2 triangle), not recovered history. Sizes 32 KiB / 2 MiB are arithmetic only.
- No ONNX dependency or parser exists in the Rust tree. onnxruntime 1.29.0 is installed in the container's Python; protoc 3.21.12 is installed; `onnx.reference` cannot run without the C++ extension.
- relalg is single-table Boolean selection plus four I32 aggregates (`crates/lance-graph-relalg/src/ir.rs`). The "mathematical polyfill landing zone" proposal holds only for the Boolean/compare/reduce fragment.

## BLOCKING corrections adopted (opusC)

1. B1: ONNX is an import membrane only; no new AST, no ONNX rewrite pass or interpreter. "Mathematical polyfill landing zone" is dropped.
2. B2: per-op admission table, default REFUSED; value-derived addressing ops (`Gather*`, `Scatter*`, `NonZero`, `Unique`, `Compress`) refused under ADMISSION B.
3. B3: no float law admitted; float ops lower as written or are refused.
4. B4: every stage has a proof-engine column; "certified" only for egglog-proved Boolean-Distinct composition associativity (#1707, reuse #1740); the "256³ Validate" stage is a differential test.

## Open points

- OP-ONNX-1: home for value maths (operator).
- OP-ONNX-2: importer dependency options A-E (operator).
- OP-ONNX-3: DR64³/DR256³ original plan missing.
- OP-ONNX-4: bit-position scan versus D-RU-6 / ADMISSION B.
- OP-ONNX-5: no independent proof checker.
- OP-ONNX-6: certificate rebinding beyond #1740 (focus/scope/ABI/geometry).
- OP-ONNX-7: prover revision not recorded (egglog `branch = "main"`): a certificate-identity gap, not a dependency defect.
- OP-ONNX-8: onnxruntime int32 `ReduceSum` saturation unexplained.
- OP-ONNX-9: `dr-mathematical-compiler-v1.md` calls egglog's self-check "independent".
- OP-ONNX-10: `dr-mathematical-compiler-v1.md` versus D-RA-2.
- OP-ONNX-11..13: R2 closing-view check placement; `observe_focus` scope omission; refusal granularity and strings in the contract.
- OP-ONNX-14: which proof object datafrog, Ascent or MeTTa/MORK/PathMap would produce; the operator rule names them, but only egglog produces one here today.

D-ONNX-RELALG-CAP | STATUS: open | OUTCOME: S0 plan and this entry; 4 BLOCKING corrections adopted; DR64³ original plan not found | OPEN: OP-ONNX-1..14
