# cognitive-stack — Integration Plan

How the **new stack** (Elixir templates) composes with the **old stack** (the
AdaWorldAPI forks) into the Cognitive Compilation loop, and how this golden-image
crate fits the larger plan at `../../.claude/plans/cognitive-compilation-v1.md`.

## 1. The loop, mapped to crates/repos

```text
new task
  │
  ▼
[ lance-graph ]  basin match: have we compiled this shape before?      ← OLD
  │
  ├─ known ─────────────────────────────────────────────────────────┐
  │                                                                   ▼
  │                                         [ template-runtime ]  run the      ← NEW
  │                                         compiled elixir-template as the
  │                                         deterministic reflex (OGAR-action
  │                                         dispatch). NO LLM.
  │                                                   │
  │                                                   ▼
  │                                         [ OGAR ] validate against the      ← OLD
  │                                         semantic type system
  │                                                   │
  │                                                   ▼
  │                                         [ surrealdb kv-lance ] store the   ← OLD
  │                                         result in the provenance/timeline
  │
  └─ unknown / low-confidence ──────────────────────────────────────┐
                                                                      ▼
                                          [ rig ]  LLM teacher solves once     ← SEPARATE
                                          (NOT linked into this binary)
                                                   │ records ExecutionTrace
                                                   ▼
                                          [ cognitive-compiler ] trace →       ← NEW
                                          elixir-template candidate
                                                   │
                                                   ▼
                                          [ template-equivalence ] replay vs   ← NEW
                                          trace; grade Exact/RankOrder/…
                                                   │ (review + repair)
                                                   ▼
                                          promote → [ lance-graph ] index the  ← OLD
                                          new template basin; next match is a
                                          reflex with no LLM.
```

- **ractor** is the control-plane ownership fence around the SoA/mailbox state
  the loop mutates (compile-time guarantee, not in the nanosecond hot path).
- **ndarray** is underneath everything (SIMD/Fingerprint/CAM-PQ) via lance-graph.

## 2. Why this binary has no LLM

The architecture's core invariant (§18 of the plan): *no LLM in the hot path once
a template passes.* The reflex is `template-runtime` over OGAR actions, persisted
through surrealdb-kv-lance — none of which need rig. Linking rig here would be a
category error. rig lives in its own repo, wired to the **same** surrealdb-kv-lance
fork (scoped to `protocol-ws + kv-mem + kv-lance`, no rocksdb/tikv), and is invoked
only at learning/escalation time to produce the traces `cognitive-compiler`
consumes.

## 3. Fork wiring (identical to `crates/symbiont`)

| Dependency | Source | Notes |
|---|---|---|
| lance-graph, lance-graph-contract, lance-graph-ogar | path (in-repo) | spine + AR bridge |
| elixir-template, template-runtime, template-equivalence, cognitive-compiler | path (in-repo) | the new stack |
| ndarray | path `../../../ndarray` (sibling) | Dockerfile clones it |
| ractor | git, `claude/jirak-…` branch | MessagingErr fix |
| surrealdb-core | git `main`, `default-features=false, features=["kv-lance"]` | no rocksdb/tikv |
| ogar-vocab / ogar-ontology / ogar-adapter-surrealql | git `main` | OGAR semantic types |

A single `[patch]` folds the git `lance-graph-contract` (pulled transitively by
`ogar-class-view`) onto the in-repo path copy, so exactly one `ClassView` exists.
All surrealdb sources resolve to `AdaWorldAPI/surrealdb#main` (one source, features
unioned to enable `kv-lance`).

## 4. Build / verification

- **Toolchain:** Rust 1.95 (fork MSRV).
- **System deps:** `protobuf-compiler` (lance-encoding build script), `cmake`,
  `clang`, `pkg-config`, `libssl-dev`.
- **Validation model:** the Dockerfile is the canonical "does it all link"
  validator (Railway/CI), exactly like `symbiont`. The new-stack crates are unit-
  tested standalone (17 tests across the four); this crate's delta over the proven
  `symbiont` graph is those four zero-dep crates + `main`.

## 5. What is intentionally out of scope here

- `lance-template-index`, `review-gates`, `github-promoter` — deferred; their
  homes are the existing planner / agent ensemble (see the plan §8).
- rig / any LLM — separate repo, learning-time only (§2 above).
- No engine changes to surrealdb (kv-lance is consumed as-is), ractor, or ndarray.
