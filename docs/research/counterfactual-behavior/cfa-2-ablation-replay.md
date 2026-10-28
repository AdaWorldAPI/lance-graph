# D-CFA-2 — 64-thought intervention/replay and higher-order outcomes

**Status: design / falsifiers only; no evaluation run.** Parents [CFA-0](README.md) → [CFA-1](cfa-1-typed-incidence.md). Nothing here adds a Thought-Git engine, new output tensor, real-world intervention rights or changed source behavior.

## Thesis
Evaluate an S1 method-incidence view over pinned reference B[v]. T0 is the immutable sealed baseline; T1..=63 are independent speculative thought slots with generation-aware reuse. **64³ candidate intervention addresses are not 64³ concurrently executed programs**, and Thought-Git slots are independent of ON/WHEN/DO ordinals.

An intervention must be explicitly typed and its effects bounded: suppress one registered MethodId; suppress all methods at a cell; remove an ON event; change a WHEN predicate (requires a legal input model); or suppress a DO proposal. Different interventions yield different interpretations.

## What a comparison must preserve (conceptual receipt; NOT a new ABI)
- Exact immutable base ref, source/config/recipe versions and seed if used, thought slot-generation, program identity, selected intervention and frame/focus/scope.
- Baseline and altered **observable outputs**, exception/effect proposals, method firing/guard trace, read/write sets, derivation roots, and typed difference masks.
- Intermediate contradictions, order/multiplicity, negative dependencies and retractions when needed by protected readers. #1739 proves final-only squash loses some of these.
- Full covered observation boundary: final output equality does **not** show trace/effect equivalence, nor infer causality outside the deterministic model.
- Explicit three-valued guards, unknown source semantics, permitted scheduling phase, and independent authorization before touching any externally effectful action.

## Higher-order comparisons
Single ablation of A, single B, and joint {A,B} on the **same pinned baseline** should expose redundant alternatives or conjunctions that pairwise correlation misses. For deterministic pure programs also test A∘B and B∘A; equal outputs for a tested fixture are not a universal commutativity law. Candidate classifications: context-dependent requires, prevents, enables, interferes, redundant, synergizes, commutes-with, explicit supersedes. Do not confuse method co-occurrence or a Git DAG parent with dependency or cause.

## Kill tests
| ID | Counterexample | Required |
|---|---|---|
| A01 | T0 is used for an intervention | refuse: immutable givens |
| A02 | 63 slots live; request another; release/reuse with stale token | capacity refuse and ABA-safe generation checks |
| A03 | A and B independently supply the same final value | single ablations silent, joint ablation differs (redundancy) |
| A04 | C fires only when both A and B apply | {A,B} hyperedge; pairwise edges cannot imply single sufficiency |
| A05 | A suppresses B under G, not under NOT G | guarded prevention, no unconditional ban |
| A06 | final result identical but an intermediate exception/trace differs | not strong equivalence; protected squash refuses or retains witness |
| A07 | duplicate roots across thoughts are replayed | no independence inflation or new certification from replay |
| A08 | negative or stale premise removed | DRed/retraction and recompute/refuse; not stale silent merge |
| A09 | two live methods share cell, one ablated | only targeted method removed; no accidental sibling deletion |
| A10 | effectful operation (network, money, external write) inside thought | inert proposal or typed refusal; never performed 63× |
| A11 | source changed but local coordinate, cycle, or old Keep unchanged | rebinding/recompute/refusal required; #1792 authority survives |
| A12 | two sequences share final output but differ in I/O or panic | not certified reorderable |

## Executable benchmark arms (future)
1. Independent manually specified transition-system oracle with complete source facts.
2. Existing Thought-Git lab receipt/squash and Quack relational projection where legal.
3. Original source execution sandbox only when a source-owner supplies authorization and deterministic isolation.

Compare per-method intervention and presence-only cell ablations; count thought batches, output/trace diff completeness, false dependencies, memory and per-ablation cost. Pre-register fixture coverage and replay horizons before measurements. Never claim causality from a single ablation in a synthetic model.

## Follow-up
[CFA-3](cfa-3-arm-logic.md) treats these as evidence-bearing observations, not as laws. Disjoint source roots and a held-out evaluation split are required for any generalization.

Existing anchors: [Thought Git](../../../.claude/plans/thought-git-api-v1.md), [protected squash #1739](https://github.com/AdaWorldAPI/lance-graph/pull/1739), [replay #1684](https://github.com/AdaWorldAPI/lance-graph/pull/1684), [DT64 edge hypothesis #1797](https://github.com/AdaWorldAPI/lance-graph/pull/1797).
