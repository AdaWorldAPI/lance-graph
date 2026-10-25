# D-NAM-1 — Typed native method normalization and versioned DTO crosswalk

**2026-10-25 • ENGINEERING SPEC + STATIC CANDIDATE CROSSWALK VALIDATOR / PROPOSED**. Parent: [D-NAM capstone](d-nam-native-algebra-dto-capstone-v1.md), [EXO-152](https://linear.app/exo-red/issue/EXO-152). Independent implementation later; no contract ABI edit in this PR.

## Artifacts actually included in this draft PR (offline-only)

- `.claude/specs/d-nam-method-crosswalk-v1.json`: machine-readable **CANDIDATE** rows for four existing native anchors and ONNX/TFLite/MLIR/RelAlg readings. No external operator is tagged `admitted`; unspecified external versions remain `null`.
- `tools/d_nam_validate_crosswalk.py`: **static manifest linter**, no runtime import/dispatch. It refuses duplicate native IDs, missing dialect readings/negative cases, unreasoned refusals, and any `admitted` row missing a pinned version, independent oracle and evidence.
- Review command: `python3 tools/d_nam_validate_crosswalk.py`. This validates manifest shape only, **not** operator equivalence, Rust integration, ONNX/TFLite compatibility or CI coverage. A code owner must verify and run it separately before promotion.

**This is the first small tooling implementation; no production Rust method binding or DTO executor has been added.**

## Purpose
Construct the *smallest declarative mapping* that says a dialect operator (or finite operator pattern) means an existing native method/composition under explicit guards. Do not build new `Method` enum until auditing existing ClassView/method signature/ogar-r2il opcode identity; do not create another generic AST. Method identity is source-owned; DTO names are aliases with evidence.

## Existing owners to reuse
- `lance-graph-onnx/src/lib.rs`: root exact ALIAS/INLINE, `bounded::tanh_f32/exp_f32`; `ReduceSum` and `sum_where_greater_i32`.
- `lance-graph-relalg/src/ir.rs`: `Rel::Atom`, `Measure`, single-table relations; `lower.rs` executes R2IL FoldDialect.
- `lance-graph-quack/src/bind.rs`: compile names/values to numeric references once.
- `lance-graph-contract` D-RA-2 set algebra rules, `lance-graph-r2il` procedure call/readings, `lance-graph-mask-risc` numeric/mask terminals, `ndarray::simd` actual kernels.
- Linear [EXO-86](https://linear.app/exo-red/issue/EXO-86), [EXO-115](https://linear.app/exo-red/issue/EXO-115), [EXO-102](https://linear.app/exo-red/issue/EXO-102).

## Minimal schema: versioned crosswalk row (not a new runtime type yet)
| Field | Semantic obligation |
|---|---|
| native anchor | symbol / admitted composition, repo commit + owner |
| dialect selector | ONNX domain+opset+op_type; TFLite builtin opcode+schema/model version; MLIR dialect+op/traits; SQL dialect/null/multiplicity |
| operand contract | dtype, rank, logical shape, symbolic dims, strides, axis, broadcast, scalar-vs-lane |
| result demand | scalar/value lane/packed mask/materialized values, keepdims and output shape |
| carrier & law | Bool/Distinct, Bag, SQL K3, i32 sum, FP32, quantized; semiring; legal reordering |
| effects / authority | read/write alias, scope, same-table seek vs forbidden cross-table gather |
| route | DIRECT native symbol / COMPOSE existing polyfill / REFUSE; selected at cold bind |
| evidence | pinned upstream spec, direct-native test, independent oracle, negative controls and measured cost |
| reverse export | explicitly supported, conditional, or refused; never inferred from import success |

External node and model identity are *provenance*, not the native object's identity. Prepared route identity must reflect native frame/codebook/operand validity under D-NFR; do not use shape alone for compiled palette caches.

## Worked normalization groups (candidate / narrow acceptance only)
1. `X > t`: `ONNX Greater`, `TFLite GREATER`, MLIR `arith.cmpi sgt`, `Rel::i32(Gt)`; only rank-1 i32 vs scalar threshold, same element count and signed comparison. Native `gt_i32_to_mask`. Result packed mask is **internal**, TFLite/ONNX external BOOL formats need boundary packing/unpacking.
2. `select(mask,X,0)`: ONNX `Where`, TFLite `SELECT_V2`, MLIR `arith.select`; native `blend_i32` where shapes/dtypes/order agree. This is a **value** operation, not `Rel::Keep`.
3. `Σ_i select(X_i>t,X_i,0)`: composition `gt_i32_to_mask + masked_sum_i32`; ONNX `Greater/Where/ReduceSum`, TFLite candidate `GREATER/SELECT_V2/SUM`, and single-table RelAlg predicate + `Measure::Sum`. Guard integer range and shape; direct native i64 fold must not silently become ONNX/TFLite i32 result.
4. `tanh(x)`: ONNX bounded alias `ndarray::hpc::vml::vstanh` (#1758); TFLite `TANH` as separate dialect numeric policy and quantization; MLIR `math.tanh` candidate. **Outside** Boolean RelAlg; approximate bound is not an algebraic rewrite law.

**Counterexample inputs before admission:** empty n, SIMD tails n=1,15,16,17,63,64,65, nonzero threshold, signed MIN/MAX, sum beyond i32, false-vs-unknown SQL NULL, mismatched broadcast, f32 signed zero and NaN, TFLite bool layout, different opcode versions. Prove tests are non-vacuous and failures are informative.

## Implementation ownership and acceptance
- Proposed owner of crosswalk **source data**: one static declarative file generated/reviewed at cold time and consumed only by external DTO adapters; choose location after inspecting existing method registry, do not duplicate it.
- Never dispatch by ONNX/TFLite name in the hot loop; native methods are selected once. No dynamic string matching in SoA/DR/R2IL execution.
- D-RA-2 stays fixed. Normalize relational subset via contract's set rules, as scoped by EXO-115; value math stays ndarray/native numeric methods, not `Rel`.
- Gate D-NAM-1: owner-reviewed table of at least four rows with native anchors, exact version keys, typed positive+negative fixture, tests run by future code PR, plus a no-new-IR dependency graph. Unknown external version means REFUSE. An unexecuted spec does not satisfy the test gate.

## Kill criteria
If the plan requires a general AST or per-call registry; if unknown ops fall through; if method identity is guessed by matching shape; if K3/Bag/FP semantics are merged; or if one polyfill cannot reproduce the independent oracle, stop and split the candidate rather than weakening its contract.
