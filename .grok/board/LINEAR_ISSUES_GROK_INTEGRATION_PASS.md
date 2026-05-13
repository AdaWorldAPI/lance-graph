# LINEAR_ISSUES_GROK_INTEGRATION_PASS.md — Ready-to-Paste Issues for grok-integration-pass-2026-05

**Project**: grok-integration-pass-2026-05  
**Date**: 2026-05-09  
**Purpose**: Structured issues you can copy directly into Linear.

---

## Recommended Labels (Create these first)

- `SoA`
- `Hot-Path-Cypher`
- `Grammar-VSA-TEKAMOLO`
- `Multi-Zone-OGIT`
- `Foundry-Integration`
- `Technical-Debt`
- `Documentation`
- `High`
- `Medium`
- `Low`

---

## Suggested Epics / Groups

You can create these as **Epics** or use **Labels** + **Projects** in Linear:

1. SoA DTO Unification
2. Hot-Path Cypher
3. Grammar + VSA + TEKAMOLO
4. Multi-Zone Ontology + OGIT
5. Palantir Foundry Integration
6. Knowledge Base & Documentation

---

## Issues (Copy-Paste Ready)

### 1. SoA DTO Unification

**Title**: Unify SoA DTO surface (BindSpace + MetaWord + cycle_fingerprint)

**Description**:
Evolve `cognitive-shader-driver` + `lance-graph-contract` so that `BindSpace` + `MetaWord` + `cycle_fingerprint` (Vsa16kF32) become the single canonical SoA surface.

- Declare canonical surface clearly
- Add thin adapters for existing Wire DTO consumers
- No breaking changes, evolutionary approach
- Update documentation in `.grok/`

**Labels**: `SoA`, `High`  
**Priority**: High

---

**Title**: Create migration/adapter guidance for existing consumers

**Description**:
Document clear patterns for how existing code using Wire DTOs or historical types can migrate to the new canonical `BindSpace` surface.

**Labels**: `SoA`, `Documentation`  
**Priority**: Medium

---

### 2. Hot-Path Cypher Completion

**Title**: Complete real Cypher support in the hot path (Phase 2)

**Description**:
Move `cypher_bridge.rs` from keyword stub to real parsing routed through `OrchestrationBridge` → `BindSpace` → `CausalEdge64`.

- Define lightweight `CypherParseResult` DTO in contract
- Add pluggable parser hook
- Map parsed constructs to `CausalEdge64` fields (`CausalMask`, `InferenceType`, plasticity)
- Keep dependency footprint low in `cognitive-shader-driver`

**Labels**: `Hot-Path-Cypher`, `High`  
**Priority**: High

---

**Title**: Design compact 3-byte polyglot query language tag

**Description**:
Design a small (3-byte or less) polyglot tag that can represent Cypher / Gremlin / GQL / SQL / NARS operations. This should live on top of the unified SoA surface and feed into `CausalEdge64`.

**Labels**: `Hot-Path-Cypher`, `Medium`  
**Priority**: Medium

---

### 3. Grammar + VSA + TEKAMOLO

**Title**: Evolve DeepNSM + Holograph for SPO + TEKAMOLO role bundling

**Description**:
Enable grammatical role encoding (SPO + TEKAMOLO) into Vsa16kF32 vectors using Holograph VSA binding + DeepNSM grammar heuristics.

- Extend slot/positional encoding for TEKAMOLO dimensions
- Implement clean bind/unbind for grammatical roles
- Connect epiphany detection to structured fact promotion

**Labels**: `Grammar-VSA-TEKAMOLO`, `High`  
**Priority**: High

---

**Title**: Epiphany-driven promotion to CausalEdge64 + AriGraph

**Description**:
When Holograph detects an epiphany on bundled grammatical representations, automatically unbind and promote into `CausalEdge64` entries with NARS truth values and wisdom/"Staunen" markers.

**Labels**: `Grammar-VSA-TEKAMOLO`, `Medium`  
**Priority**: Medium

---

### 4. Multi-Zone Ontology + OGIT

**Title**: Clarify and own Zone 2 surface via spear

**Description**:
Formally define `spear` as the owned Zone 2 surface (HubSpot-like API over `BindSpace`). Make `lance-graph-callcenter` the realtime transport layer. Enforce zero-copy + no serialization rules in Zone 2.

**Labels**: `Multi-Zone-OGIT`, `High`  
**Priority**: High

---

**Title**: Achieve uniform OGIT semantic hydration across domain repos

**Description**:
Ensure all domain repositories (spear, SharePoint, MedCare-rs, Smb-office-rs, q2, Woa-rs, Stalwart) provide consistent inner (Zone 1/2) and outer (Zone 3) ontology views through OGIT.

**Labels**: `Multi-Zone-OGIT`, `High`  
**Priority**: High

---

**Title**: Port SharePoint C# logic into spear crate

**Description**:
Migrate the 3-layer cache + soft-pull QoS pipeline logic from C# into the Rust `spear` crate while preserving behavior and using the same DTO family as emails.

**Labels**: `Multi-Zone-OGIT`, `Medium`  
**Priority**: Medium

---

### 5. Palantir Foundry Integration

**Title**: Position OGIT as Foundry ontology spine

**Description**:
Strengthen OGIT to serve as the single source of truth for object types, link types, and properties. Define clear inner vs outer ontology views per schema.

**Labels**: `Foundry-Integration`, `High`  
**Priority**: High

---

**Title**: Evolve q2 into Gotham-style intelligence surface

**Description**:
Develop `q2` toward Palantir Gotham-like investigation and analysis capabilities, leveraging cross-domain OGIT entities + `CausalEdge64` causality.

**Labels**: `Foundry-Integration`, `Medium`  
**Priority**: Medium

---

**Title**: Cross-domain semantic + causal reasoning (MedCare + SMB + OSINT)

**Description**:
Enable powerful cross-domain graph reasoning and diagnostics by wiring MedCare-rs, Smb-office-rs, and q2 through OGIT + `CausalEdge64`.

**Labels**: `Foundry-Integration`, `Medium`  
**Priority**: Medium

---

### 6. Knowledge Base & Documentation

**Title**: Add missing 01_overview.md

**Description**:
Create `01_overview.md` as a high-level system map and key invariants document, aligned with current architecture understanding.

**Labels**: `Documentation`, `Low`  
**Priority**: Low

---

**Title**: Maintain and evolve .grok/ board

**Description**:
Keep `LATEST_STATE.md`, `TECH_DEBT.md`, `EPIPHANIES.md`, and integration plans up to date as work progresses.

**Labels**: `Documentation`, `Low`  
**Priority**: Low

---

## Suggested Kanban Columns (if not using default)

- **Backlog**
- **Ready**
- **In Progress**
- **Review**
- **Done**

---

You can copy the issues above directly into Linear. Let me know if you want me to adjust priorities, split any issues further, or add more detail to specific ones.