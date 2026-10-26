# Claude Code handoff: D-NAM native algebra DTO normalization

**Date 2026-10-25. Status: READ-ONLY SCOUT / IMPLEMENTATION GATED.** Source of truth: `.claude/plans/d-nam-native-algebra-dto-capstone-v1.md`; [EXO-152](https://linear.app/exo-red/issue/EXO-152) plan, [EXO-50](https://linear.app/exo-red/issue/EXO-50) append-only handoff. This is **not** approval to merge or refactor.

## Task
Map ONNX, TFLite, MLIR and SQL/RelAlg operators as **versioned DTO semantics over existing native methods and their polyfills**. Never create a new tensor graph, interpreter, general AST, runtime dispatcher, library of duplicate kernels or independent method registry.

## Before writing code
1. Fetch main and list exact source refs + merged status of #1644, #1682, #1752, #1753, #1754, #1755, #1758 and the D-ONNX S0 arc #1744/#1750; read current `.claude/board/STATUS_BOARD.md`, `INTEGRATION_PLANS.md`, `LATEST_STATE.md` and relevant owner decisions.
2. Read `contract::set_algebra` and `set_rules`, `relalg::ir/normalize/lower`, `quack::bind`, R2IL procedures / mask-risc, `lance-graph-onnx/src/lib.rs`, ndarray kernels; check actual ClassView/method signature API before proposing new types.
3. Read EXO-86, EXO-102, EXO-115, EXO-126, EXO-127, EXO-132 and newest EXO-50 comments. Do not overwrite others' sessions or claim their code.
4. Pin official ONNX and TFLite builtin operator schema versions; quote exact MLIR dialect documentation/version and SQL oracle semantics. Gemini ONNX IR overview is an **unverified external research overview**, not authority for this project's representation.

## First deliverable (no mutation)
Output a table for four narrow cases: i32 Greater scalar; i32 Select/Where; guarded masked sum; f32 Tanh. Each row includes external name+version, concrete native symbol/composition, all shape/axis/dtype/mask/NULL/overflow/precision constraints, a source anchor, negative test, independent reference and refusal. Distinguish import semantics, normalization, execution, proof and export as separate columns.

## Future implementation PRs (only when approved)
- D-NAM-1: cold crosswalk and typed admission, reuse method identity already available, no per-call interpretation.
- D-NAM-2: ONNX/TFLite **leaves**, TFLite bool/quantization boundary; no native dependency on the facades, no arbitrary model conversion claim.
- D-NAM-3: MLIR indexing and SQL/RelAlg legal normalization: contract owns set rules, relalg planning; no `Rel::MatMul`.
- D-NAM-4: same native method routes through Quack/SoA/DR and R2IL/ndarray; reuse bindings, zero redundant guards, no cross-table gathers.
- D-NAM-5: conditional reverse DTO and oracle/cost closure; no automatic inference of exportability.

## Kill tests
Wrong opset, empty/invalid axes, shape mismatch, signed zero/NaN, i32 overflow, SQL K3 and Bag, TFLite BOOL packing and zero-point, synthetic-only oracle, wrong model weights/codebook, stale focus/frame, 0*Inf numeric contraction and effects. Counterexample → refuse, do not repair via a new runtime.

## Response format
Propose the *smallest* code diff per owner crate, actual test command, expected failure of a deliberate mutation, end-to-end cost accounting and PR gate; clearly mark EXISTS/LAB/PROPOSED/MISSING/REFUSED. Link output to EXO-152 via its Linear issue and send **one new envelope comment** to EXO-50 with explicit sender/session ID; GitHub remains authoritative. Never infer operator approval from an agent's proposal.
