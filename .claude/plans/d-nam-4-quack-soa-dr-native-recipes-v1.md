# D-NAM-4 — Cold DTO→native method binding onto Quack, SoA, DR and R2IL

**2026-10-25 • ENGINEERING SPEC / PROPOSED; NO CODE IN THIS PR.** Parent [EXO-152](https://linear.app/exo-red/issue/EXO-152), stages D-NAM-1/2/3. See native representation contract `d-native-repr-contract-v1.md`, [#1755](https://github.com/AdaWorldAPI/lance-graph/pull/1755), [EXO-102](https://linear.app/exo-red/issue/EXO-102), [EXO-127](https://linear.app/exo-red/issue/EXO-127), [EXO-151](https://linear.app/exo-red/issue/EXO-151).

## Responsibility split
- DTO frontends: select and validate a **native method signature** once; no foreign strings or schemas past cold binding.
- Native Method / Quack binder: resolve permitted ClassId, lane, value carrier and prepared route; reuse `bind.rs` numeric names/codebooks; **do not create a second table catalog**.
- Contract/RelAlg: normalize Boolean set predicates and provide relational plan where applicable. The `Rel` enum remains narrow.
- R2IL/mask-risc: native Boolean masks / integer reduction and composed polyfills; `FoldDialect` is one R2IL reading, not all of R2IL.
- ndarray: numerical arithmetic, transcendental, contraction and SIMD; do not recreate kernels.
- SoA / DR: borrow a versioned value plane and mask/focus under existing source owner. DR coordinate is not an arbitrary element index or palette code.

## Prepared recipe (conceptual only)
```text
validated dialect spec → existing source-issued method identity
    → resolve native aliases/compositions and native numeric carrier
    → bind inputs: sealed SoA borrow, domain, mask, codebook generation,
                   logical index map, external provenance (cold)
    → choose direct native route / native polyfill / explicit REFUSE
    → call native kernels over borrowed data (hot: NO dialect lookup)
```
Do not add `NativeRecipe` public type until checking existing `CompiledProgram`, Quack Query, ClassView method signature and R2IL route APIs. Borrow exclusivity may rule out a per-op frame check; [#1755](https://github.com/AdaWorldAPI/lance-graph/pull/1755) explicitly withdrew the redundant `FrameStore` guard. External authorization/provenance remain distinct from Rust lifetime safety.

## Admission gates
1. Static signatures: dtype/rank/axes/broadcast/layout, logical coordinate domain, output shape, null/Bag/Distinct, precision, effects.
2. Binder validity: source owner, ClassId/method, frame+focus scope, codebook generation/digest where relevant; refuse stale, wrong tenant or wrong model/layer. Never identify a compiled weight palette by shape alone.
3. ADMISSION B: no cross-table value-derived positional reads. Version-witnessed same-table preparation seek permitted only with counted `attest_cost`, `projection_build_cost`, `query_read_cost`.
4. Execution: exact native fallback (where eligible) or typed refusal; bounded approximation requires named error policy and per-layer identity, never hidden fallback.
5. Mask-vs-value: `Rel::Keep` writes set mask, `Where/SELECT_V2` writes values. Don't conflate DR physical focus with mathematical filter.
6. Quantization: TFLite affine i8/u8 ≠ Palette256. A Palette256³ kernel is an *optional separately calibrated recipe* only after codebook identity, closure, error and fair end-to-end cost gates.

## Targeted implementation slice
Reuse #1752's native `Greater+masked_sum_i32` composition. Bind the SAME resident lane once, run as (A) direct `ndarray` composite, (B) R2IL fold, (C) existing relalg Query, (D) Quack route where eligible. All results must equal independent scalar oracle or typed-refuse if semantics mismatch. Report bind/prep/execute/materialize and bytes for 0/1/17/64/65/4096 rows; include wrong-frame, changed lane/weights, empty and overflow. Negative-control removal of mask or wrong threshold **must turn test red**.

## Performance falsifier
Compare direct baseline to mapped recipe at N=1,8,64,256 reuses: `cold/N + bind + prepare + convert + execute + materialize + transfer` and memory. A proposed mapping that adds an allocation, per-call dynamic registry or unbounded materialization loses its claimed zero-overhead story. Report microseconds/nanoseconds with environment and controls, never infer a benchmark result from design.

**Blockers:** No production DR64³ third-axis type, no universal Quack frame binding store, codebook runtime ID/DR→palette link missing in D-NFR. Those are *not* prerequisites for the tiny rank-1 native masked sum fixture. Do not expand its scope to solve them.
