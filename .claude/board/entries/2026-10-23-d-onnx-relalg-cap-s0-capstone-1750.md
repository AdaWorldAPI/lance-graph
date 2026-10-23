# 2026-10-23 — E-D-ONNX-RELALG-CAP-S0-1: the ONNX → RelAlg → DR64³ pipeline is false as stated; integer island only

**Status:** OPEN — PROPOSAL, S0 docs-only. Nothing implemented. Read at `origin/main` `59538865`; ndarray `1ecbccb`; ONNX tags v1.23.2 (opset 28, IR 14) and v1.17.0 (opset 22, IR 10).

Plan: `.claude/plans/d-onnx-relalg-cap-capstone-1750-v1.md` (#1750's S0; #1744's S0 is at `d-onnx-relalg-cap-v1.md`).

## Central finding

1. **The pipeline "ONNX graph → RelAlg polyfill → Cartesian projection → native execution" is false as stated.** `lance-graph-relalg` is a single-table Boolean selection over `i32`/`u32` lanes with Count/Sum/Min/Max/Keep (`crates/lance-graph-relalg/src/ir.rs:34-112`). It has no tensors, floats, lane-to-lane arithmetic or nonlinear functions. The contract refuses IEEE and Bag by name (`set_rules.rs:44,465-472,581-586`).
2. **Corrected routing:** the ONNX `ModelProto` stays the only graph. An intake classifier sends each node to one of: a relalg integer selection/aggregate island, an R2IL/mask-risc terminal, an existing ndarray kernel, or a typed refusal. Gather/Scatter, control flow and `ai.onnx.ml` are refusals. D-RA-2 = A is respected: no tensor node, no second AST, no new executor.
3. **Zero new mathematical primitives.** Of 226 opset-28 names: directly 1, compositionally 93, kernel 91, external 4, unsupported/unknown 37, polyfill 0. About 16 can reach relalg.
4. **DR64³ plan not found.** `git grep` over `main` and 833 remote branches returned 0 files. Recorded as an OPEN documentation dependency; no stages reconstructed. What exists: DR64 `BoundDr64`, DR256 paging in `dir-rmap-probe`, and the R∘S chain. "64³ Explore → 256³ Validate → Certified Method → 64³ Reuse" is invalid as one track.
5. **Corrections to the fact base:** `mish_f32` exists (`ndarray/src/hpc/linalg/activations_ext.rs:168`); the "no shared relational AST" line is at `INTEGRATION_PLANS.md:16`, not `ISSUES.md:16`.
6. **Certification scope:** only egglog-proved associativity of Boolean-Distinct composition over DR64 (`relcomp.rs:421`, #1707), reuse proven same-shape only (#1740), no checker independent of egglog.

## Not claimed

No "ONNX-compatible" label; no "semantic preservation" at S1; no float certification; no DR64³ existence or speed; no Gather on the substrate path; no LLVM AOT path. Full list: plan section 11.

## Next gate

Before S1 the operator must provide: (1) **one target model** to drive the op list; (2) **AdaWorldAPI protobuf fork coordinates** (or confirmation that none exists). Without the second, the session stops and asks; there is no crates.io fallback.

## Open points

19 registered in plan section 9 (OP-1 … OP-19): BLOCKED OP-3, OP-6, OP-7; the rest OPEN, OP-13 CLOSED (runtimes as oracle arms only).
