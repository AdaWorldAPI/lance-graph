# D-DRMC-0 — DR Mathematical Compiler: observed structure to reusable executable law

**Status:** PROPOSAL, staged draft. **Scope:** a new contract/probe arc alongside Quack binding, not a second executor or a replacement for existing D-RFT, D-QSP, D-CMR or D-RMAP owners. **Date:** 2026-10-22.

## Thesis

> DR is the structural ABI between ClassView-declared capabilities and mathematical compilation. The compiler discovers candidates from witnessed observed populations, checks explicit assumptions and counterexamples, normalizes eligible mathematics, proves *transformations* where possible, and retains compiled recipes that may be rebound to later populations.

Names, CUI/TUI labels, tokens and embedding similarity are not proof of mathematical structure. An observational statement about a finite population, an algebraic theorem conditional on axioms, and an empirical/causal generalization are **three different outputs**. None licenses the next without its own evidence.

The same logical DR contract can bind `Identity × Identity`, `Identity × Property`, `Community × Community` or `Class × Capability`. DR64 is one physical shape (64 × 64 Boolean bits = 512 B); a 256 × 256 presence bitmap is 8 KiB. A physical bit encodes participation, **not** a 16-bit foreign ID, rank, evidence, bag multiplicity, SQL NULL, or a proof. These remain separately bound, witnessed carriers.

HHTL cascade `(part_of:is_a)` coordinate reading and `hhtl::NiblePath` taxonomy routing are *different* objects. Morton8x8's u16 maps 256 × 256 exact Cartesian positions; HHTL masks refine admissible address ranges. HHTL prefix containment does **not** establish arbitrary ontology membership or causality.

## Three contracts, independent owners

1. **Semantic/structural (zero-dep contract):** ClassView declares per-method identity, typed operand domains, arity, type/semiring, applicable law, effects, and result demand. `compute_dag` declares dependency order but does not itself implement a method; the ClassView method registry cannot mint facts or authorize writes.
2. **Binding (Quack-adjacent adapter):** resolve class/method and relation-map identities once, then bind borrowed canonical SoA/ordered relation/DR operands with exact row/column coordinates, source+map+palette/index seals, focus, snapshot, validity, authorization scope and method capability. No hot labels, DTO hydration, implicit global→local gather, or storage ownership. A bound 64×64 register may be consumed directly by existing SIMD kernels; R2IL is the general SIMD polyfill, **not** mandatory re-dispatch over prepared bits.
3. **Mathematical compilation (cold and replayable):** RelAlg/typed AST expresses admissible mathematics; source-independent oracles (scalar, MORK, datafrog, PostgreSQL as appropriate) test hypotheses; Ruler/Enumo-style guarded law candidates enter egglog; eggcc-inspired extraction respects effects/order; independently checkable proof artifacts and cost model select existing R2IL/Quack/mask-risc procedures. No egglog/MORK interpreter in the runtime hot fold, no new opcodes by default.

`MeTTa` is a declaration frontend, not the numeric executor. `MORK` / `PathMap` are independent differential/reverse-engineering arms, not a canonical relation store. `datafrog` checks fixed-point recursion, not unearned production support. `Thought-Git` may record immutable law/proof/recipe/binding/replay/commit references, but retains its independent write/version authority. ACT-R / Rubicon / Kanban / cognitive-shader-driver consume results and create *proposals*, never become mathematical-ISA owners.

## Formal claim families

Given a finite, sealed population `P@v`, a stated constraint `Gamma` and a relation `R`:

`violations = count({x in P@v | Gamma(x) = TRUE and R(x) = FALSE})`.

A finite-population claim requires an independently attested **complete** checked universe, nonzero eligible membership, no UNKNOWN where a universal conclusion is asserted, zero violations and immutable snapshot/scope proof. Merely counting zero failures in an arbitrary sample is not a theorem. Under SQL K3, UNKNOWN ≠ FALSE and WHERE projects TRUE only; under graph paths, DISTINCT support ≠ BAG derivation counts.

A mathematical-law claim `forall x in domain D: f(x)=g(x)` requires separately admitted domain axioms, rule-checker and independently reproducible proof; source provenance and executable recipe identity are separate. A broader observational or causal claim requires new data, held-out populations or interventions/assumptions, not only proof of compiler rewrites.

## Cold vs hot

```text
real populations / named methods / schema
          ↓ ClassView / Quack-style BIND
typed DR/SoA relation + scope, versions, capability, admissible carriers
          ↓ normalized RelAlg / algebraic AST (no row execution)
candidate rules + independent differential oracles
          ↓ cold Ruler/Enumo → egglog → effect-aware extraction
proof-checked law + compiled R2IL/Quack recipe + target/ABI identity
          ↓ cache of mathematical recipe (not source-specific fact)
runtime: re-BIND current witnesses → choose admitted carrier
          ↓ direct resident SIMD OR R2IL/Quack → mask-risc → ndarray::simd
result + read receipt  ──────→ consumer ACT-R/Rubicon/MedCare/IAM/etc.
                                  effect proposal → independent commit gate
```

No new evaluator: physical-specialized folds may consume already-bound DR register directly (`dir-rmap-probe::compose` precedent from #1685); other population programs remain the existing mask-risc machinery. A compiled mathematical identity does not bypass snapshot, scope, permission, SQL validity, bag semantics, or commit authorization.

## Existing anchors and ownership

- `lance-graph-contract::{class_view, set_rules, set_algebra, hhtl, morton8x8}`: contract grammar, no app-specific behavior.
- `lance-graph-quack/src/bind.rs`: numeric resolution membrane; `lance-graph-cypher-quack`: bounded, explicit refusals, not full graph execution.
- `lance-graph-relalg` and `lance-graph-r2il`: prototype normalizer/procedural polyfill; multi-relation `mu` and arbitrary preimage not yet shipped.
- `mork-pathmap-probe`, `lance-graph-metta`, `dir-rmap-probe`: differential/front-end/DR lab evidence, not general production semantics.
- `egglog-r2il-probe`: E1 synthetic exact-integer evidence (#1690–#1696); **not** a theorem-proving platform or production optimizer. #1698 is actively resolving contradictory premises and intermediate checked overflow. Do not edit its owned files in this stack.
- `lance-graph-act-r`, `gkat-probe`, `rubicon_witness`, `kanban`: proposal/decision/lifecycle boundaries.
- D-RMAP-1A controls separately owned RelationshipMaps/MapPalettes/Thought bindings; D-QSP and D-RFT (#1667) control Quack SQL parity and thought frontier. D-CMR controls `set_rules` law catalogue. EXO-50 is the shared coordination inbox.

## Staged falsifiers and review gates

| Stage | Deliverable | Minimal red counterexample |
| --- | --- | --- |
| **0** | This ownership/contract charter | Incorrectly treating physical DR presence as proof, bag count or CUI |
| **1** | Standalone, typed, borrowed DR64 binding *probe* beside Quack boundary | Swap `Identity²` for `Identity×Property`; stale coordinate, source, map or ACL; foreign row-slot confusion |
| **2** | Finite-population claim ledger, isolated from theorem proof | Empty sample; incomplete coverage; UNKNOWN; nonzero counterexample; changed snapshot; forged counters are **not** independent attestations |
| **3** | Cold law/recipe descriptor and guarded runtime admission, anchored to Math Foundry | Changed method semantics / checker / target / ABI; exact→checked overflow; unverified law or aliasing stale binding |
| **4** | Differential suite and downstream consumption handoff | MORK set vs bag; Datafrog recursion; K3; scope revocation; unsupported association/causality; ACT-R decision must not commit |

Each stage is independently reviewable and STACKED upon the prior draft branch. Only after source attestation, oracle checks, independent proof checker and full cost tests may a specific pattern become a production capability. Each must refuse unsupported shapes, not silently degrade semantics.

## Experimental success criterion

One ClassView-declared method over a witnessed `Identity×Property` DR64, and one `Identity²` relation, normalize to the same **correctly typed** mathematical primitive *only when warranted by guards*; a independently evaluated finite population falsifies wrong assumptions; a checked law allows compiling a reusable target recipe; the same recipe binds to two fresh, different snapshots without sharing their source data. Report bind, indexing, prefetch, execution, extraction, storage, proof-check, cold/warm amortization and negative controls. No performance claim in this document.

**This is an architecture and experiment charter, not implementation, permission to merge or an achieved scientific discovery.**
