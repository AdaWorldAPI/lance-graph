# TECH_DEBT.md — Technical Debt Register (Grok Mirror, Pruned & Updated)

**Date**: 2026-05-08  
**Source**: Mirrored + heavily pruned/updated from `.claude/board/TECH_DEBT.md`. Focused on current high-impact debt.

## Top Priority Debt (Current)

| ID | Area | Debt | Impact | Owner / Next Step |
|----|------|------|--------|-------------------|
| **TD-NARS-01** | NARS + Thinking fragmentation | Logic duplicated across `causal-edge`, `nars_engine.rs` (planner), `cognitive_frameworks.rs` (learning), `orchestrator.rs`, `auto_style.rs`, DeepNSM/Holograph. No canonical hot-path surface or unified style modulation. 12 vs 36 styles split. | High duplication (40%+ session restart cost), drift risk, inconsistent Pearl/NARS application | Use new canonical inventory `.grok/03_cognitive_layers/NARS_THINKING_IMPLEMENTATIONS_INVENTORY_MIGRATION.md` as single source. Execute phased migration to `CausalEdge64` + shader-driver `NarsOp`. |
| **TD-QUERY-01** | Multiple Cypher implementations | 4–6 parallel approaches (DataFusion cold + shader stub + proposed compact). No clear primary hot path. | High confusion + duplicated effort | Consolidate on `cognitive-shader-driver` canonical bridge. Design 3-byte polyglot tag. |
| **TD-QUERY-02** | `cypher_bridge.rs` is stub only | Keyword classifier (`starts_with CREATE/MATCH`). Explicitly Phase 1. | Hot path has almost no real Cypher | Wire real parser in Phase 2 while staying inside `OrchestrationBridge`. |
| **TD-HOT-01** | Missing compact polyglot encoding | "OGIT of query languages in 3 bytes" vision exists but not designed. | Missed extreme hot-path efficiency | Design small tag (language family + op class + Pearl/NARS hints). |
| **TD-INTEGRATION-01** | Weak `CausalEdge64` ↔ query intent mapping | Parsed Cypher not yet translated into `CausalMask` / `InferenceType` / plasticity. | Lost semantic power in hot path | Define mapping rules inside shader driver / `BindSpace`. |
| **TD-ARCH-01** | Cold vs Hot path strategy unclear | DataFusion path is mature but heavy for cognitive workloads. Hot path is directionally correct but incomplete. | Performance tax on common case | Declare hot path primary for cognition; cold path for heavy analytics. |
| **TD-CANONICAL-01** | Risk of LAB-ONLY leakage | Many useful things still behind `serve`/`grpc` features. | Test scaffolding polluting canonical API | Enforce "extend canonical bridge, not Wire DTOs" rule. |

## Lower Priority / Historical Debt (Pruned)

Many older items from the original `.claude/board/TECH_DEBT.md` (pre-2026-05) related to early grammar, crystal, and contract work have been deprioritized or resolved as the architecture has converged on `CausalEdge64` + `cognitive-shader-driver` + `jc` proofs as the core.

## Positive (Debt Being Reduced)

- Strong Canonical vs LAB-ONLY boundary in `cognitive-shader-driver/lib.rs`.
- `BindSpace` + native `CausalEdge64` storage is excellent substrate.
- `jc` proofs running in CI provide mathematical guardrails.
- `MetaColumn` cheap prefilter + SoA design is high-signal optimization.

---

**Maintenance Rule**: When significant new debt is discovered or old debt is resolved, update this file + `LATEST_STATE.md`.