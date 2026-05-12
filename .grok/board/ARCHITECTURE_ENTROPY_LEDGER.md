# ARCHITECTURE_ENTROPY_LEDGER.md — Current Entropy Sources (Pruned & Updated)

**Date**: 2026-05-08  
**Purpose**: Track sources of architectural entropy (complexity, ambiguity, duplication) vs resolved items. Pruned version of original.

## Current High Entropy Sources

| Source | Description | Impact | Status |
|--------|-------------|--------|--------|
| **Multiple Cypher / Query Language Implementations** | 4–6 approaches (DataFusion cold path + shader stub + proposed compact). No single clear hot-path strategy. | High | Active Debt |
| **Incomplete Hot Path Cypher Support** | `cypher_bridge.rs` is still a minimal stub. | High | Active Debt |
| **Missing Compact Polyglot Tag Design** | The "3-byte OGIT of query languages" vision is not yet specified. | Medium-High | Open Opportunity |
| **Weak Mapping: Query Intent → CausalEdge64** | Parsed queries not yet translated into `CausalMask` / `InferenceType` / plasticity. | Medium-High | Active Debt |
| **Cold vs Hot Path Strategy** | Unclear when to use DataFusion vs shader driver path. | Medium | Needs Decision |
| **Thinking Styles ↔ CausalEdge64 Interaction** | How `AgentStyle` should influence `CausalEdge64` processing is undefined. | Medium | Open |

## Recently Reduced Entropy (Positive)

- `cognitive-shader-driver` + `BindSpace` now clearly positioned as the SoA driver.
- `CausalEdge64` established as the atomic causal register with native storage in `BindSpace.edges`.
- `jc` mathematical proofs running in CI provide strong guardrails.
- Clear Canonical vs LAB-ONLY boundary documented in `cognitive-shader-driver/lib.rs`.
- `cycle_fingerprint` recognized as the unit of thought.

## Resolved / Archived (from original ledger)

Many older entropy items related to early grammar, crystal bundling details, and contract surface debates have been resolved or superseded by the convergence on `CausalEdge64` + `cognitive-shader-driver` as the core.

---

**Maintenance**: Update this file whenever significant new entropy is introduced or major debt is resolved. Link to `TECH_DEBT.md` for actionable items.