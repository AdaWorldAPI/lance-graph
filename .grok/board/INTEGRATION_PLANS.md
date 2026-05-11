# INTEGRATION_PLANS.md — Current High-Priority Integration Plans (Updated Mirror)

**Date**: 2026-05-08  
**Focus**: Only active, high-signal plans. Historical plans pruned or summarized.

## 1. Hot-Path Cypher Completion (Current Highest Priority)

**Goal**: Move from the current stub in `cypher_bridge.rs` to real Cypher (and future polyglot) support in the hot path.

**Key Deliverables** (detailed plan in `HOT_PATH_CYPHER_COMPLETION.md`):
- Define lightweight `CypherParseResult` DTO in `lance-graph-contract`.
- Add pluggable `CypherParser` hook in the bridge.
- Map parse results into canonical SoA surface (`BindSpace` + `MetaWord` + `CausalEdge64`).
- Support causal hints (`CausalMask` / `InferenceType`).

**Status**: Detailed plan created. Ready for incremental implementation once SoA surface is declared canonical.

---

## 2. Evolutionary Unification of the SoA DTO Surface (Foundation)

**Goal**: Explicitly own and strengthen `BindSpace` + `MetaWord` + `cycle_fingerprint` as the single canonical SoA surface (see `UNIFIED_SOA_SURFACE_PLAN.md`).

This is the required foundation for #1 and all future hot-path work.

## 2. 8-Mask Superposition over CausalEdge64

**Goal**: Treat Pearl’s 8 `CausalMask` states as parallel representational dimensions.

**Proposed Layers**:
- L1: 64×64 operations over `CausalEdge64`
- L2: 256×256 palette distance matrices as attention headers
- L4: Projection of the 8 mask-views into 4096 codebook space to fill NARS gaps

**Status**: Strong conceptual direction from user. Needs concrete data layout and performance analysis while preserving hot-path characteristics.

## 3. Thinking Styles ↔ CausalEdge64 Modulation

**Goal**: Make `AgentStyle` (Plan/Act/Explore/Reflex) influence the processing of `CausalEdge64` instances.

**Open Questions**:
- Should certain styles prefer specific `CausalMask` levels?
- How should style selection affect plasticity decisions?
- Can the `MetaOrchestrator` use `CausalEdge64` statistics as input to MUL?

**Status**: Clear gap. High potential.

## 4. JC Mathematical Foundation → Hot Path Design

**Goal**: Use the proofs in the `jc` crate (especially Pearl 2³ and weak-dependence results) as guardrails when designing the compact polyglot tag and superposition mechanisms.

**Status**: The foundation exists and runs in CI. Needs explicit connection to new hot-path components.

---

## Deferred / Lower Priority

- Full DataFusion cold path enhancements (keep for analytical use cases only).
- Historical grammar / crystal / DeepNSM plans (largely superseded by current `CausalEdge64` + shader architecture).

---

**Rule**: Before starting new integration work, check this file + `LATEST_STATE.md` + `TECH_DEBT.md`.