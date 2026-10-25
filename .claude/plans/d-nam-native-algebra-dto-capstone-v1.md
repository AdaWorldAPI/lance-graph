# D-NAM — Native Algebra Methods × external DTO dialects (capstone v1)

**Date:** 2026-10-25. **Status:** PROPOSED / docs only; this PR is NOT an implementation, executable proof or operator approval. **Linear:** [EXO-152](https://linear.app/exo-red/issue/EXO-152) (plan), [EXO-50](https://linear.app/exo-red/issue/EXO-50) (append-only session handoff).

## Thesis
A native mathematical method and its admitted composed polyfill are primary. ONNX, TFLite, MLIR and SQL/RelAlg expressions are **versioned data-transfer-object (DTO) readings** of that mathematics, not separate execution engines. Normalization is *partial and typed*: an external expression is recognized only when its operation, carrier, address domain, shape, reduction, semantics and effects agree with the admitted native recipe. Export reverses this crosswalk only where a compatible external representation is witnessed.

```text
ONNX / TFLite / MLIR / SQL           native Quack / RelAlg / ndarray callers
             \                         /
                 semantic DTO mapping
                         |
             normalized native method identity
          + admitted polyfill + semantic guards
                         |
            Quack binding / R2IL / mask-risc
                         |
           SoA values + DR mask/focus + ndarray
```

The center is a *method identity/crosswalk*, not a universal tensor, graph or AST. ONNX ModelProto / TFLite FlatBuffer / MLIR remain external artifacts; any importer or exporter belongs outside the native execution hot path.

## Binding decisions inherited, not reopened
- **D-RA-2=A** [EXO-86](https://linear.app/exo-red/issue/EXO-86), [#1644](https://github.com/AdaWorldAPI/lance-graph/pull/1644): `lance-graph-contract` owns single-table set algebra and rewrites. `lance-graph-relalg` owns planning/lowering. Do not grow `Rel` into `Rel::MatMul/Tanh/Conv` or create a second full graph IR.
- **D-NFC** [#1753](https://github.com/AdaWorldAPI/lance-graph/pull/1753), `d-native-first-dependency-correction-v1.md`: ONNX is external leaf facade. Native R2IL/ndarray work never depends on importing ONNX. `lance-graph-onnx` [#1754](https://github.com/AdaWorldAPI/lance-graph/pull/1754), [#1758](https://github.com/AdaWorldAPI/lance-graph/pull/1758) merged aliases and bounded Tanh/Exp.
- **Existing narrow reality:** `lance-graph-relalg/src/ir.rs` is one-table relational selection `Rel {All,Empty,Plane,Atom,And,Or,Not}` + `Measure {Count,Sum,Min,Max,Keep}`; `relalg/lower.rs` uses R2IL FoldDialect. `quack/bind.rs` resolves names, IDs and codebook values before execution; it does not currently consume relalg's whole IR. [EXO-115](https://linear.app/exo-red/issue/EXO-115) governs any normalization bridge.
- **SoA / DR:** [#1755](https://github.com/AdaWorldAPI/lance-graph/pull/1755) validates borrowed SoA kernel access in a lab. D-NFR contract describes DR64² lab, DR64³ proposed, codebook identity and DR→Palette link missing. Never claim DR64³ or Palette substitution shipped merely because a plan names them.
- **ADMISSION B** [EXO-102](https://linear.app/exo-red/issue/EXO-102): bounded, version-witnessed same-table preparation seeks; cross-table value-derived positional gathers forbidden. Preserve SQL K3 NULL and Bag/Distinct separate from Boolean sets ([EXO-126](https://linear.app/exo-red/issue/EXO-126)).
- **Proof:** Egglog/DialEgg may certify an admitted, typed rewrite offline; [EXO-127](https://linear.app/exo-red/issue/EXO-127) does not authorize FP reassociation, runtime egraph dispatch or a new theorem checker. Palette composition requires carrier closure evidence ([EXO-132](https://linear.app/exo-red/issue/EXO-132)).
- **Scope fence:** D-DTM-1 [#1777](https://github.com/AdaWorldAPI/lance-graph/pull/1777) can later transport methods/receipts, but transport, distributed state, Kafka and Flight are not dependencies of this arc.

## The four dialect readings, one native example
`Σ{x_i | x_i > t}` with rank-1 i32 input:
- ONNX candidate: `Greater(X,t) -> Where(cond,X,0) -> ReduceSum`.
- TFLite candidate: `GREATER -> SELECT_V2 -> SUM`.
- MLIR candidate: `linalg.generic` predicate and reduction with explicit indexing/iterator maps.
- RelAlg candidate: `Rel::i32(lane,Cmp::Gt,t), Measure::Sum(lane)` on same table.
- Direct native precedent: `ndarray::simd::gt_i32_to_mask` + `masked_sum_i32`; ONNX facade already has `ReduceSum::sum_where_greater_i32`.

**Important:** These are NOT universally equivalent schemas. They form one admitted *restricted fixture* only when scalar threshold, broadcasting, output shape, reduction axes, value carrier, zero filling, integer range and null policy coincide. The TFLite runtime bool byte format must not be mistaken for native packed mask layout. The `Rel` query selects rows; `Where` returns values. The zero-filled composition is equivalent to masked sum only under the agreed numerical domain and overflow behavior. Refuse mismatches.

## Method contract, not another IR
A cold, versioned crosswalk row consists of:
`{native_method, composition?, dialect, domain/op/version, type, logical_shape, index_map, mask/validity, reduction_axes, carrier/semiring, overflow+FP contract, state/effects, proof/evidence, output_shape, export_policy, refusal}`.
This is an **engineering schema sketch**, not a proposed `struct` merged into contract. Owner of actual storage is decided in D-NAM-1 after locating existing ClassId/method registry. Native operation function pointers are bound once, not switched on dialect per call.

## PR trajectory: independent branches from main
| Stage | Deliverable (separate PR) | Gate / owner |
|---|---|---|
| **D-NAM-0** | This alignment and read-only agent handoff; board index | docs-only, no ABI |
| **D-NAM-1** | Versioned crosswalk schema, method normalization, concrete four-dialect examples, typed refusals | contract/Quack owners; no duplicative method registry |
| **D-NAM-2** | ONNX + TFLite thin *external leaf* DTO mappings | ONNX facade/ndarray owners; official schema pins and differential oracle |
| **D-NAM-3** | MLIR indexing semantics and SQL/RelAlg normalizer bridge | EXO-86/115/126; no floats in set rewrite |
| **D-NAM-4** | Binding to SoA/DR/Quack/R2IL native prepared recipes | EXO-102/127/151; zero hot-path dispatch, focus and identity guards |
| **D-NAM-5** | Conditional native→DTO export, independent conformance and full-cost closure | no claim of full model conversion; reversibility only for admitted subset |

**Dependency DAG:** S0→S1; S1→{S2,S3}; S2+S3→S4; S4→S5. All draft branches start at main; **dependency is approval/test dependency**, not a stacked commit dependency. After merges rebase each surviving branch and verify ancestry per [EXO-136](https://linear.app/exo-red/issue/EXO-136).

## Acceptance and stop conditions
1. No native execution crate imports ONNX/TFLite/MLIR, no new graph AST, runtime interpreter, foreign evaluation fallback, or new SIMD implementation.
2. Type/shape/axis/quantization/NaN/±0/overflow/empty/K3/Bag/effects/version are retained or typed-refused; unknown domain/version never guessed.
3. Prove an admitted alias vs a direct native call; polyfill vs *independent* scalar oracle; compare external runtime as separate oracle where available.
4. An admitted normalization cannot broaden support: ONNX `MatMul` ≠ Boolean relation compose; SQL NULL ≠ false; `Gather` under a tensor backend ≠ permitted cross-table positional gather.
5. Count cold parse/map, preparation, bind, conversions, allocations, native calls and full output; zero hot-loop registry lookup. Kill a specialization if conversion erases the benefit.
6. Gated PR stages remain **PROPOSED** until code, tests, receipts and owner decisions exist. Draft green docs are not an implementation.

## Canonical source breadcrumbs
- `.claude/plans/d-onnx-relalg-cap-v1.md`, `d-onnx-relalg-cap-capstone-1750-v1.md`, `d-onnx-relalg-s1-s2-semantic-intake-v1.md`, `d-onnx-relalg-s3d-physical-shootout-v1.md`, `d-onnx-relalg-s4-s5-palette-export-v1.md`
- `.claude/plans/d-native-repr-contract-v1.md`, `d-native-first-dependency-correction-v1.md`, `d-relalg-unified-v1.md`, `d-onnx-native-v1.md`
- [EXO-152](https://linear.app/exo-red/issue/EXO-152), [EXO-50](https://linear.app/exo-red/issue/EXO-50) (handoff only).

## Opened draft PR receipts (2026-10-25)

- D-NAM-0: [#1778](https://github.com/AdaWorldAPI/lance-graph/pull/1778) — capstone and board; this stage only.
- D-NAM-1: [#1779](https://github.com/AdaWorldAPI/lance-graph/pull/1779) — static crosswalk spec, candidate JSON manifest and fail-closed schema validator; runtime mapping still NOT implemented.
- D-NAM-2: [#1780](https://github.com/AdaWorldAPI/lance-graph/pull/1780) — ONNX/TFLite engineering spec.
- D-NAM-3: [#1781](https://github.com/AdaWorldAPI/lance-graph/pull/1781) — MLIR/SQL/RelAlg engineering spec.
- D-NAM-4: [#1782](https://github.com/AdaWorldAPI/lance-graph/pull/1782) — prepared Quack/SoA/DR recipes engineering spec.
- D-NAM-5: [#1783](https://github.com/AdaWorldAPI/lance-graph/pull/1783) — reverse DTO and conformance closure spec.

All six are **drafts**, independent branches from main. Only the D-NAM-1 static manifest/validator are code tooling; no native Rust runtime/ABI integration exists in this arc yet. Every proposed implementation gate needs a separate code review/approval.
