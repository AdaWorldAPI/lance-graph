# D-CFA-5 — Compiler-style counterfactual analysis; guarded law admission

**Status:** cold design / preregistered falsification; NOT IMPLEMENTED, NOT RUN. Stacked after [CFA-4](cfa-4-nsl-onnx.md) for review; the neural arm is logically OPTIONAL. The required evidence parent is [CFA-3](cfa-3-arm-logic.md), not an ONNX prediction.

## Question
Can source-typed and replayed behavioral dependency candidates act as a **compiler-style analysis surface**, exposing may-def/use, contextual prevention, interference, redundant branches, constrained commuting and potential equivalent compositions, while preserving language-specific effects and independently refusing unproven transformations?

### Compiler analogy is analysis, not a second compiler
- LLVM MemorySSA: useful **may-def/use and versioning comparison**. It is not exact source aliasing, does not alone allow volatile/atomic reordering and does not imply observations.
- MLIR dataflow: dependency propagation and fixed-point comparison; use for conceptual pass composition, not an inserted mandatory MLIR AST in the Quack hot path.
- Ruff SPO / OGAR: read-only source structure and type/provenance seed. Rust HIR/MIR source authority remains rust-analyzer/rustc where explicitly pinned and supported; Ruff is not a Rust semantic equivalence oracle.
- DR/Quack: existing typed relation queries and execution; Datafrog/Ascent/MORK are *cold comparison/oracle views*, not a replacement hot engine.
- ARM + Thought-Git: observed/model-conditioned evidence and counterfactual differences. They can propose **which optimization to inspect**, not which law is mathematically true.
- Egglog/DialEgg: optional cold equality saturation and MLIR translation of supported expressions; only admitted rewrite laws with effect/order obligations. Equality saturation under unverified axioms is not a proof of language semantics.
- Native compiled recipe remains D-DRMC/R2IL/ndarray/mask-risc; no new instruction set, method registry, general graph or source write path.

## Candidate analysis questions (not new opcodes)
- `may_def/use(A,B)`: static/sourced potential memory and field dependence.
- `observed_interferes(A,B | ctx)`: intervention alters B trace/outcome, with pinned model and scope.
- `conditionally_prevents(A,B | ctx)`: guard- or ordering-dependent suppression.
- `jointly_required({A,B},C | ctx)` and `redundant({A,B},output | ctx)`: factorized hyperedges; don't erase multiplicity.
- `commutes_candidate(A,B | ctx)`: compare A∘B and B∘A, including visible effects/exceptions, not just final bitsets.
- `equivalence_candidate(original,rewrite | laws, effect_scope)`: MUST require source-compatible typing, source/law witnesses, independently checked proof or bounded-domain exhaustive oracle (as advertised), target/ABI and execution parity.
- `proposal`: inert ProposedMethod or existing recipe reference under SAG authority; never direct code edit.

## Claim and warrant levels
1. Source fact: which source method/guard/effect is actually declared.
2. Static possibility: may-read, may-write, unresolved dispatch, alias uncertainty.
3. Observation: input-state trace or model-relative intervention result.
4. Statistical learned association: ARM/NSL candidate only.
5. Logical derivation from explicitly warranted premises: inference receipt, not automatically a law.
6. Domain-scoped theorem or independently checked transformation: only after proof obligations AND target/source-parity checks.
7. Authorized application: separate business/system owner effect gate; a compiler proof cannot grant permission.

## Falsification register
| ID | Counterexample | Mandatory result |
|---|---|---|
| C01 | static read/write overlap yet no output change in one fixture | may-dependency persists; no global independence assertion |
| C02 | distinct methods with same final result but different exception/trace | equivalence refuses |
| C03 | A and B commute on fixture, fail with other guard | no global commutation law |
| C04 | float reassociation alters intermediate rounding | refuses bit-exact rewrite absent target numeric proof |
| C05 | integer overflow or panic differs on a boundary | refuses optimization |
| C06 | Rust drop/borrow/unsafe/volatile/atomic ordering differs | refuses side-effect rewrite |
| C07 | generic/trait dynamic dispatch unresolved | possible target remains unresolved; no false proof |
| C08 | MLIR pattern matched but source method provenance is missing | no extraction into a source patch |
| C09 | egglog saturates an unproven user-discovered law | remains candidate; no admitted-law certificate |
| C10 | detached Keep reused after data source mutation | stale input must recompute/refuse (#1792) |
| C11 | output version is equal but codebook/scope source changed | reject inappropriate cache reuse |
| C12 | learned NSL rank is confident but oracle contradicts | preserve witness/counterexample, no supersession |
| C13 | same rewritten operation has different input aliasing across source frames | no universal transform |
| C14 | proof checker unavailable for claimed general theorem | explicit `NoCertifiedCompiler` / hypothesis, never `proven` |
| C15 | correct pure optimization but no performance gain after cold+hot accounting | retain baseline, not automatic promotion |

## Future probe topology (separately approved)
Cold input: pinned source AST/HIR/MIR when available + Ruff crosswalk + S1 bindings + S2/3 replay/evidence. Existing RelAlg normalized queries construct dependency proposals. Optionally feed a *supported effect-free subset* to egglog/DialEgg MLIR import and extract a candidate expression. Check against an independent scalar/source reference, pinned numerical model, source effect checker and law checker. Record the complete Tracert witness per D-SAG-6. Then either `REFUSE`, `PARK`, or issue an *inert candidate recipe* for operator approval; no source edits or actual external actions.

**Quantitative comparison (deferred):** baseline static compiler analysis vs ARM-only vs Thought-Git counterfactual + higher-order vs optional NSL heuristic. Report falsifier coverage, candidate reduction, false positive transformation rate, cold µs, warm per-fold ns, peak scratch and total cost separately. Specify supported semantic subset and independent oracle in next review.

## Deliberate OPEN questions for second round
Where exactly should the behavioral observation projection bind existing `ProposedMethod` without introducing a duplicate type? How much of the source effect model can the current harvest attest? Can a real equivalent transform be found under supported zero-side-effect arithmetic before attempting Odoo/OGAR side effects? Does NFR-OP-2 want a third DR axis or only a factorized logical view? These are decisions, not settled architecture.

## Sources
- [D-SAG-6](../../../.claude/plans/d-sag-6-rust-codegraph-math-logic-tracert-v1.md), [DT64 Q0](../../../.claude/plans/d-dt64-q0-borrowed-quack-binding-seam-v1.md), [D-DRMC](../../../.claude/plans/dr-mathematical-compiler-v1.md)
- [#1760](https://github.com/AdaWorldAPI/lance-graph/pull/1760), [#1747](https://github.com/AdaWorldAPI/lance-graph/pull/1747), [#1792](https://github.com/AdaWorldAPI/lance-graph/pull/1792)
- [LLVM MemorySSA](https://llvm.org/docs/MemorySSA.html), [MLIR Dataflow Analysis](https://mlir.llvm.org/docs/Tutorials/DataFlowAnalysis/), [Alive2](https://github.com/AliveToolkit/alive2)
- [DialEgg CGO 2025](https://2025.cgo.org/details/cgo-2025-papers/44/DialEgg-Dialect-Agnostic-MLIR-Optimizer-using-Equality-Saturation-with-Egglog)
