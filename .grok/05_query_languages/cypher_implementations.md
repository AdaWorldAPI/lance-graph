# Cypher Implementations — Landscape & Technical Debt

**Date**: 2026-05-08  
**Status**: High-signal synthesis. Multiple implementations exist across the codebase history and current structure.

## Executive Summary

There are **multiple (estimated 4–6) distinct implementations or approaches** to Cypher (and polyglot query support) in this ecosystem. They fall into two broad categories:

- **Cold Path**: Traditional, heavy, DataFusion-based relational planning.
- **Hot Path**: Compact, SoA-oriented, `CausalEdge64`-native, designed for the resonance/shader loop.

The `cognitive-shader-driver` version (currently a stub in `cypher_bridge.rs`) represents one of the **later / hot-path oriented attempts** (likely #4 or #6 depending on how historical variants are counted).

## Known / Inferred Implementations

| # | Implementation | Location | Path Type | Maturity | Notes / Technical Debt |
|---|----------------|----------|-----------|----------|------------------------|
| 1 | DataFusion Planner (full relational) | `crates/lance-graph/src/datafusion_planner/` | Cold | High | Mature but heavy. Classic columnar joins, scans, expressions. High build/runtime cost. |
| 2 | Historical / legacy parsers | Various (possibly older `lang-graph` or planner modules) | Mixed | Unknown | User references suggest earlier attempts existed. |
| 3 | Navigator / AriGraph Cypher exposure | `crates/lance-graph/src/graph/navigator.rs` (historical references) | Cold/Mixed | Medium | Cypher procedures exposed via graph API. |
| **4** | `cognitive-shader-driver` CypherBridge (stub) | `crates/cognitive-shader-driver/src/cypher_bridge.rs` | **Hot** (intended) | Low (Phase 1 stub) | Keyword-only classifier. Designed for `OrchestrationBridge` + `BindSpace`. **This is likely the one you refer to as #4 or #6**. |
| 5 | Compact Polyglot Vision (proposed) | Not yet implemented | Hot | Planned | 3-byte (or small) tag system for Cypher / Gremlin / GQL / SQL / NARS etc. "OGIT of query languages". |
| 6 | Full hot-path parser (Phase 2 target) | Not yet wired | Hot | Planned | Real `parse_cypher_query` integration into `cognitive-shader-driver` canonical path. |

> **Note**: Exact historical count (4 vs 6) is fuzzy because some implementations were in separate crates or refactored. The important signal is the **recurring pattern of multiple parallel efforts**.

## Technical Debt Analysis

### High Debt Areas

| Area | Debt Level | Description | Risk | Recommended Action |
|------|------------|-------------|------|--------------------|
| **Multiple parallel Cypher implementations** | High | At least 2–3 active or recent approaches (DataFusion cold path + shader stub + planned compact). Fragmented ownership. | Confusion, duplicated effort, inconsistent semantics | Consolidate around the hot-path canonical bridge (`OrchestrationBridge` + `BindSpace`) |
| **cypher_bridge.rs is only a stub** | High | Currently does naive `starts_with("CREATE")` / `starts_with("MATCH")`. Explicitly marked "Phase 1". | Hot path has almost no real Cypher support | Wire real parser in Phase 2 while staying inside canonical API |
| **Cold path (DataFusion) carries full engine weight** | Medium-High | Excellent for complex analytics, but overkill for most cognitive/resonance use cases. | Performance tax on common path | Keep for analytical workloads; make hot path the default for cognitive cycles |
| **No unified compact polyglot encoding yet** | Medium | The "3-byte OGIT of query languages" vision is powerful but not yet designed/implemented. | Missed opportunity for extreme hot-path efficiency | Design the small tag system (language + op family + Pearl/NARS hints) |
| **LAB-ONLY vs Canonical boundary** | Medium | `cypher_bridge` is correctly placed under LAB-ONLY for now, but real functionality must move to canonical surface. | Risk of leaking test scaffolding into production API | Enforce the rule documented in `cognitive-shader-driver/src/lib.rs` |
| **Missing connection to CausalEdge64** | Medium | `BindSpace` already stores `CausalEdge64` in `edges` column, but Cypher intent is not yet translated into `CausalMask` / `InferenceType` / plasticity. | Lost semantic richness in hot path | Map parsed Cypher constructs → `CausalEdge64` fields |

### Positive Signals (Low Debt)

- Strong architectural boundary between **Canonical** and **LAB-ONLY** in `cognitive-shader-driver`.
- `BindSpace` + `MetaColumn` prefilter + `CausalEdge64` storage is an excellent foundation for a compact hot path.
- `OrchestrationBridge` trait provides a clean extension point.
- `jc` crate mathematical proofs (including Pearl) can underwrite the compact representation.

## Epiphanies

1. **The fragmentation is not accidental** — It reflects the tension between "we need full Cypher for compatibility" (cold path) and "we need extreme speed + causal semantics for cognition" (hot path). Both are valid; the debt comes from not having a clear primary path + fallback strategy.

2. **The hot path is winning architecturally** — `cognitive-shader-driver` + `BindSpace` + native `CausalEdge64` storage is the direction aligned with the overall Resonance + L1–4 loop vision. The DataFusion path is legacy for specialized analytical use cases.

3. **3-byte polyglot tag is the missing primitive** — A compact, O(1) identifier for (language family + operation class + causal level) would allow the shader to dispatch with almost zero parsing cost — true "OGIT for query languages".

4. **Technical debt is mostly organizational + prioritization**, not fundamental design flaws. The pieces (`BindSpace`, `CausalEdge64`, `OrchestrationBridge`, `jc` proofs) are already in place.

## Recommended Documentation & Implementation Priorities

1. **Document the strategy** clearly: Hot path (compact + `CausalEdge64`-native) is primary for cognitive workloads; DataFusion cold path is for heavy analytical queries.
2. **Design the compact polyglot tag** (3-byte or small opcode) as the unifying primitive.
3. **Move real Cypher support** into the canonical hot path (Phase 2 of `cypher_bridge`), keeping it inside `OrchestrationBridge`.
4. **Explicitly map** parsed query constructs → `CausalEdge64` fields (`CausalMask`, `InferenceType`, plasticity hints).
5. **Update `boot.md`** and this file whenever a new implementation or major refactor occurs.

---

**Next Session Starting Point**: Design the 3-byte polyglot query language tag + sketch how it flows into `CausalEdge64` creation inside the shader driver.