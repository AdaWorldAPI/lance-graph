# lance-graph Session Handover — 2026-03-12

## The Boring Version

This repo is being restructured from a monolith (19K lines in one crate)
into 8 focused crates with clean separation of concerns.

**Plan:** `CRATE_SEPARATION_PLAN.md` in repo root

**Canonical architecture docs (in ladybug-rs):**
- Prompt 15: RISC Brain Vision
- Prompt 19: Hot/Cold One-Way Mirror
- Prompt 20: Four Invariants (this repo = "The Face")
- Prompt 21: This plan

## Role in the Four-Repo Architecture

```
rustynum       → The Muscle (SIMD substrate)
ladybug-rs     → The Brain (BindSpace, SPO Crystal, server)
lance-graph    → The Face ← THIS REPO (Cypher/SQL query surface)
staunen        → The Bet (6 instructions, no GPU)
```

lance-graph owns: parser, planner, execution engine, BlasGraph algebra, server binary.
lance-graph DOES NOT own: BindSpace, SpineCache, qualia, awareness loop (those are ladybug-rs).

## Key Imports

- holograph/src/graphblas/ → lance-graph-blasgraph (7 semiring algebras)
- holograph/src/{bitpack,hdr_cascade,epiphany,resonance}.rs → lance-graph-spo
- ladybug-rs/src/graph/spo/{sparse,scent}.rs → lance-graph-spo

## Current State

- Monolith crate compiles and passes tests
- Tests: 12 test files, ~9300 lines
- graph/spo/ is a stale copy from ladybug-rs (WILL BE REPLACED during separation)
- DataFusion planner is the most valuable code (5633 lines)
- Parser is identical to ladybug-rs lance_parser (which is being deleted)
