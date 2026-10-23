# D-NFC — canonical dependency correction: the native polyfill is the root

**Status:** DECISION (operator, 2026-10-23: approved the architecture corrections to #1746–#1751 and #1750). Documentation only. Merges nothing, adds no ABI, changes no execution contract.
**Representation contract:** `.claude/plans/d-native-repr-contract-v1.md` (D-NFR). This document cites it for every shipped / lab / missing claim.
**S0 dispute:** #1744, #1745 and #1750 remain competing S0 drafts. This document does not pick one; §6 is a proposal for their owners.

## 1. The two P0 invariants

1. **ONNX is an external mathematical compatibility specification.** The existing R2IL mathematical polyfill is the native starting point. No ONNX graph, Tract runtime or Optimum dependency is in the internal execution path.
2. **Quack binding store + DR64³ + Palette256³ is the intended native representation architecture.** DR64³ owns the proposed local relational / focus geometry; Palette256³ owns the proposed associated numerical representation. Existing R2IL, mask-risc, BGZ, Helix and ndarray primitives are reused. Implemented components and unimplemented integration contracts are kept apart (D-NFR §2).

> We implement ONNX mathematics using our native algebra. We do not implement our native algebra inside ONNX.

## 2. The canonical dependency diagram

Every affected PR references this diagram instead of restating its own.

```text
  ARCHITECTURAL DEPENDENCY (each layer needs only the ones above it)

  [1] R2IL mathematical router               SHIPPED  route per ClassId, once: DIRECT / COMPOSE
        │   (ClassId → LazyLock → prebound              onto ndarray::simd (via mask-risc where a
        │    native callable → kernel)                  program is needed), LOWER to r2sleigh only
        │                                              for machine semantics, else REFUSE (D-MPF-4).
        │                                              The ClassId-keyed table itself: LAB (#1752).
  [2] Quack binding                           SHIPPED  bind once → numeric Query (S1–S3)
        │   binding store / frame validity    MISSING  (S5; smallest seam = D-NFR-1)
  [3] DR64³ relational / focus geometry       LAB      frame, focus, scope, 64×64 register,
        │                                              256-slot focus (S6–S12); 3rd axis PROPOSED
  [4] Palette256³ numerical representation    SHIPPED  bgz17 S/P/O codebooks (S13–S14)
        │   bound to DR coordinates           MISSING  (S15 codebook identity, S16 binding)
  [5] Native execution                        SHIPPED  mask-risc + ndarray::simd; bgz17 / Helix
        │
  [6] egglog transformation + certification   LAB      Boolean / Distinct laws only (S12)

  External leaf facades (nothing in [1]–[6] depends on them):

  [7] ONNX vocabulary facade                  DRAFT    lance-graph-onnx (#1754): ONNX names
        ┄┄> [5] native functions                       re-exported (ALIAS) or behind a minimal
                                                       adapter (INLINE); typed refusals; no
                                                       dispatch, no init; depends on ndarray only
  [8] Tract / ORT reference comparison,       OPTIONAL differential oracles, offline tooling
      Optimum model export                             never in the execution path

  These are architectural dependencies, not a serial schedule: [7] conformance
  tests can start now. They test the native layers against ONNX's definitions;
  they never define the native representation.
```

> ⊘ **Correction to layer [7] (2026-10-23, D-ONNX-NATIVE-0, `.claude/plans/d-onnx-native-v1.md`).** The diagram first drew [7] "ONNX mathematical conformance adapters — PROPOSED, one operator's semantics → native call" inside the vertical chain, between [6] and [8]. That placement read as if the chain passed through ONNX.
>
> - [7] is a **leaf**: an ONNX name maps straight onto an existing native function, and no native layer depends on it.
> - `lance-graph-onnx` (#1754) implements this. Its only normal dependency is `ndarray`; `cargo tree -i` shows nothing depends on it; it has no `LazyLock`, ClassId routing or per-call choice.
> - It does not import R2IL's router [1]. The router and the facade are independent.
> - Which S0 draft documents [7] (§6) is still the owners' decision. #1754 does not decide it.

> ⊘ **Correction to layer [1] (2026-10-23, D-MPF-4, `.claude/plans/d-mpf-math-polyfill-v1.md` §10).** Layer [1] first read "R2IL mathematical polyfill — `lance-graph-r2il::procs` + FoldDialect". That equated R2IL with one of its strategies, and the replaced wording stays visible here for that reason.
>
> - **R2IL** is an opcode table (`ogar-r2il`) with two readings: machine (`0xC400`: r2sleigh's `r2conc` / `r2sym`) and fold (`0xC401`: `FoldDialect`).
> - **Layer [1]** is the router. It resolves a route once per ClassId and stores the callable in the binding. `FoldDialect`, mask-risc programs and direct `ndarray::simd` calls are strategies beneath it.
> - **R2Sleigh** is required only for machine semantics: carry/borrow flags, address-space load/store, branches, width-exact integers. No population operation in the current set needs it.
> - **Hard invariant:** ClassId → LazyLock → prepared native dispatch table → direct kernel execution. No per-call registry or hash lookup, backend `match`, capability probe, R2Sleigh decision, dialect reconstruction, ONNX dispatch or CPUID. Quack binding, DR frame/focus/scope and Palette codebook revisions are enforced by rebinding or by a versioned identity check, never by re-routing.

## 3. Root assumptions behind the contradictions

The two harvests found 218 tagged passages across #1744–#1751 (119 + 99). They are not 218 contradictions; they follow from seven root assumptions. Each is corrected once, at the PR that owns it. The other PRs point here.

| id | root assumption (as written in the PRs) | corrected statement | owning PR | where it recurs |
|---|---|---|---|---|
| A1 | the ONNX `ModelProto` / `GraphProto` is the internal graph and interchange representation | native typed expressions and compiled R2IL procedures are the internal contract; an ONNX file is external input, read only by a conformance adapter [7] | S0 (disputed: #1744 / #1745 / #1750) | #1746, #1749, #1751 |
| A2 | native work waits for ONNX intake: an importer, an AdaWorldAPI protobuf fork, one target model | native Stage 1 has no ONNX prerequisite. The importer, protobuf fork and target model gate only an optional external-file adapter | #1746 (S1 intake) | #1744, #1745, #1748, #1749, #1750, #1751 |
| A3 | a benchmark or gate is defined by "one genuine ONNX operator" | benchmarks measure native operations; ONNX conformance is a separate test dimension on the same operations | #1748 | #1749, #1751 |
| A4 | the fallback is the original ONNX graph, or a whole-model delegated ONNX backend | the native fallback is the exact native method (e.g. the tensor arm, or the uncompressed lane). An ONNX model may remain an external validation artifact; it is never the canonical internal fallback | #1746, #1751 | #1750 |
| A5 | DR and Palette are optional physical strategies beneath an ONNX graph | DR64³ is the intended native relational representation and Palette256³ its numerical companion (D-NFR §2). Tensor / DR / hybrid benchmarks compare physical implementations of one native contract; the architectural target is not the benchmark winner | #1747, #1748 | #1745, #1751 |
| A6 | Optimum / ORT tooling sits in the plan of record | Optimum is downstream and optional [8]. Nothing native depends on it | #1751 | #1746, #1748, #1749 |
| A7 | a compiled artifact is identified by ONNX model / layer / node identity | identity is the native binding: Quack resolution under a sealed frame, plus the codebook generation (D-NFR S5, S6, S15). ONNX node identity, where it exists, is external provenance only | #1749 | #1746, #1747, #1751 |

What the PRs already got right, and is kept unchanged: no new AST or graph format; typed refusals with whole-model `STRICT_NATIVE` refusal; approximate is never silently exact; `ORTQuantizer` is not a BGZ codebook constructor; Burn's dimension-keyed `CompiledLinear` / `CompiledAttention` registration is unsafe without identity binding.

## 4. Native fallback and exact versus approximate

- **Exact native operations** (selection, Boolean combination, integer folds on mask-risc) have no fallback: they are the reference. Overflow and division by zero are refused, never wrapped.
- **Approximate native operations** (palette lookup, compiled linear / attention) run only when a binding opts in, and always name their error contract: a CLAM radius is a bound; a rank correlation (ρ) is a quality measurement, not a bound. Their fallback is the exact native method on the uncompressed data, never an ONNX graph.
- **Floating-point operations** run in existing ndarray / bgz / Helix kernels with explicit numerical validation. They carry no algebraic certificate, and none is invented for them.
- **An ONNX model** may be kept as an external validation artifact, compared offline through a reference runtime [8].

## 5. Measured baseline that the corrections cite (#1752, draft, pending independent review)

| path at 64 rows (one mask word) | ns per call |
|---|---:|
| fair scalar `i64` loop, runtime threshold | 12 |
| ndarray compare + masked sum | 39 |
| mask-risc, compiled program | 94–99 |
| R2IL procedure composition | 171–172 |
| ONNX-shaped wrapper (rebinds every call) | 217–234 |
| ClassId-prebound route, guard included (D-MPF-4) | 61–67 |
| ClassId-prebound route, callable alone (D-MPF-4) | 36–39 |

- The 50 ns one-word target is not met. The ndarray step alone is 39 ns.
- The earlier "1.6× faster than the scalar loop" is SUPERSEDED: it compared against an `i128` filter-chain oracle. A fair `i64` loop is ~3× faster than the fold at 64 K rows.
- At 64 K rows all paths converge on `ndarray::simd::masked_sum_i32`, a scalar set-bit walk.
- The ONNX-shaped wrapper is a diagnostic compatibility boundary, not an architectural layer.
- ⊘ (D-MPF-4) The mask-risc and R2IL-composition rows are what those paths cost when chosen per call. They are not a floor for R2IL routing. A route resolved once per ClassId equals the ndarray row: the callable alone costs 36–39 ns. The frame/focus/scope guard adds 27–28 ns. "R2IL prepared" (FoldDialect's `last_program` kept as a `CompiledProgram`) equals the mask-risc row.

## 6. S0 reconciliation proposal (owner-facing, not a decision)

The three S0 drafts agree on more than they disagree: no new AST, typed refusals, relalg stays narrow, numerics stay in ndarray. They disagree on who owns the capstone, and all three take A1 and A2 as given. Proposal:

1. Each owner adopts §1 and §2 by reference, through a marked supersession note on their own branch, and leaves the rest of their draft unchanged.
2. The surviving S0 then documents layer [7] only: the conformance-adapter boundary and its refusal rules. Native stages are owned by D-MPF (#1752) and D-NFR.
3. The owners choose which draft carries [7], or merge the three, on EXO-50. Nothing here decides it.

## 7. PR cross-reference

| PR | correction applied (marked supersession, its own plan + description) |
|---|---|
| #1746 | A1, A2, A4: native polyfill API first; ONNX semantics are external mappings; intake prerequisites move to the optional adapter |
| #1747 | A5, A7: DR64³ is the intended native relational representation, not an ONNX optimization |
| #1748 | A3, A5: benchmarks compare physical implementations of native contracts; native methods first |
| #1749 | A1, A7: GraphProto does not own internal representation; Quack + DR64³ + Palette256³ is the internal integration |
| #1750 | A1, A2, A4 (owned): `ModelProto` is not the internal graph; OP-6 / OP-7 gate only the optional adapter |
| #1751 | A4, A6: Optimum downstream and optional; ONNX graph fallback is external interoperability |
| #1752 | none: the shipped polyfill is the root; kept in draft for independent review |
| #1744, #1745 | not edited; §6 proposal |
