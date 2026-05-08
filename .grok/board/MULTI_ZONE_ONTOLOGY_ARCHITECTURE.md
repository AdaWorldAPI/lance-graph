# MULTI_ZONE_ONTOLOGY_ARCHITECTURE.md — Zone 1 / Zone 2 / Zone 3 + OGIT Hydration Vision

**Date**: 2026-05-08  
**Scope**: Inner hot ontology (Zone 1), mid-tier zero-copy layer (Zone 2 via spear + callcenter), outer serialized ontology (Zone 3). All domains unified via OGIT semantic hydration. Palantir Foundry + AGI aspirations.

---

## Architecture Overview (Current Vision)

The system is intentionally split into three performance and serialization zones:

### Zone 1 — Inner Ontology (Hot Path)
- **Latency target**: 20–200 ns
- **Characteristics**: Zero serialization, pure in-memory, register-level + SoA.
- **Core components**:
  - `BindSpace` (Struct-of-Arrays with Vsa16kF32 cycle fingerprints + native `CausalEdge64` storage)
  - `CollapseGate` (mutation control, entropy budget, promotion decisions)
  - `cognitive-shader-driver` as the SoA wiring layer
  - `CausalEdge64` as the atomic causal + epistemic unit
- **Ontology access**: OGIT-mapped labels + CAM codebook schema via **3–5 byte pointers** into OGIT (O(1) schema + label lookup).
- **Goal**: Maximum speed for continuous resonance, thinking cycles, and L1–L4 closed loop.

### Zone 2 — Mid-Tier / BindSpace Surface (Currently Being Defined)
- **Latency target**: 2–200 ms
- **Characteristics**: Still zero-copy to `BindSpace`, no serialization inside the zone. Realtime-capable.
- **Current implementation**: `lance-graph-callcenter` (Supabase realtime transcode layer).
- **Emerging component**: **spear** repository
  - Intended to become the **HubSpot-like surface** for outer consumers.
  - Plugs into `lance-graph-callcenter`.
  - Should provide a clean, domain-agnostic API surface over the inner ontology while remaining in Zone 2 performance characteristics.
- **Ontology**: Same OGIT + CAM codebook inheritance as Zone 1 (3–5 byte pointers).

### Zone 3 — Outer Ontology (Cold / Serialized Path)
- **Latency target**: Higher (ms to seconds acceptable)
- **Characteristics**: Full serialization (JSON, gRPC, MySQL Parallelbetrieb, etc.).
- **Purpose**: External integration, long-term persistence, cross-system interoperability.
- **Ontology**: Still aims to be **Palantir Foundry aspiring** per domain, with OGIT as the semantic spine.

---

## Cross-Domain Repositories (All via OGIT Hydration)

The vision is that every domain repository becomes a first-class citizen by hydrating through **OGIT** (inner and outer ontology per domain = Palantir Foundry aspiring).

| Repository          | Domain Focus                          | Current State / Debt | OGIT Role | Notes |
|---------------------|---------------------------------------|----------------------|-----------|-------|
| **spear**           | HubSpot-like surface for Zone 2       | New / In design     | Primary consumer of Zone 2 BindSpace | Should become the clean external-facing surface while staying zero-copy to inner ontology |
| **Stalwart Mailserver** | Email / communication               | External            | Hydrate mail events into OGIT      | Emails as first-class entities |
| **SharePoint**      | Documents, invoices, workflows        | C# (ShareGate-like 3-layer cache + QoS pipelines) | Needs Rust transcoding into spear crate | Invoices should use same DTOs as emails |
| **MedCare-rs**      | Clinical studies, patients, diagnostics | Active              | Strong cross-domain contributor    | ViT + FMA + SNOMED → graph reasoning + diagnostics |
| **Smb-office-rs**   | Small business logic                  | Active              | Business entities + workflows      | Core SMB domain |
| **q2**              | OSINT, intelligence                   | Aspiring            | Palantir Gotham / Neo4j style      | High-value external integration target |
| **Woa-rs**          | Work orders                           | Active              | Operational workflows              | Strong OGIT candidate |
| **lance-graph-callcenter** | Realtime transcode / Zone 2 surface | Existing            | Bridge between Zone 1 and spear    | Supabase realtime |

**Key Principle**: All of the above should eventually flow through **OGIT semantic hydration** so that inner ontology (Zone 1) and outer ontology (Zone 3) per domain remain consistent and Palantir Foundry-like.

---

## Technical Debt & Gaps (Current)

1. **Zone 2 is still fuzzy**
   - `lance-graph-callcenter` exists but `spear` is not yet a clear, owned surface.
   - Risk of leaking serialization concerns into Zone 2 or duplicating BindSpace access logic.

2. **SharePoint C# logic needs transcoding**
   - 3-layer cache + soft-pull QoS pipelines in C#.
   - Invoices should use the **same DTO family** as emails.
   - Needs clean migration path into a Rust `spear` crate without losing QoS characteristics.

3. **OGIT integration is still early**
   - While OGIT is being built as the canonical ontology spine, many domain repositories are not yet fully wired through it for both inner and outer ontology views.
   - 3–5 byte pointer + CAM codebook inheritance is powerful but not yet uniformly applied across all repositories.

4. **Zone boundaries and serialization contracts are not crisp enough**
   - Clear rules for what can cross from Zone 1 → Zone 2 → Zone 3 without introducing hidden serialization or copy costs.
   - `CollapseGate` decisions need to be visible/auditable across zones.

5. **Cross-domain reasoning is powerful but under-connected**
   - MedCare-rs (clinical) + Smb-office-rs + q2 (OSINT) have high potential for cross-domain graph reasoning, but the wiring through OGIT + `CausalEdge64` is incomplete.

6. **"AGI aspiring" thinking layer vs domain surfaces**
   - The inner thinking engine wants to be AGI-aspiring.
   - The outer surfaces (spear, q2, etc.) need to remain practical and HubSpot/Foundry-like without forcing AGI complexity on external consumers.

---

## Recommended Evolutionary Path (Debt-Free)

### Phase 1 — Clarify & Own Zone 2 Surface
- Formally define `spear` as the **owned Zone 2 surface** (HubSpot-like API over BindSpace).
- Make `lance-graph-callcenter` the realtime transport into spear.
- Document strict rules: Zone 2 = zero-copy to BindSpace, no serialization, OGIT + CAM codebook only.

### Phase 2 — OGIT Semantic Hydration Standard
- Create a clear contract: every domain repository must provide:
  - Inner ontology view (Zone 1 compatible, 3–5 byte OGIT pointers + CAM)
  - Outer ontology view (Zone 3, Foundry-style, serializable)
- Use OGIT as the single source of truth for schema + label mapping across all domains.

### Phase 3 — SharePoint Transcoding into spear
- Port the C# 3-layer cache + QoS pipeline logic into a Rust module inside the spear crate.
- Ensure invoices use the same DTO family as emails (unified transcode surface).
- Preserve soft-pull QoS characteristics.

### Phase 4 — Cross-Domain Graph Reasoning
- Wire MedCare-rs, Smb-office-rs, q2, Woa-rs, etc. through OGIT hydration.
- Enable `CausalEdge64` + Pearl/NARS reasoning to operate across domains (clinical + business + OSINT).
- Make causality trajectories and diagnostics first-class outputs.

### Phase 5 — Zone Boundary Enforcement + CollapseGate Visibility
- Make `CollapseGate` decisions visible and auditable from Zone 2 and Zone 3.
- Define clear serialization boundaries and zero-copy guarantees.

---

## Connection to Previous Work

- Builds directly on **unified SoA DTO surface** (`BindSpace` + `MetaWord` + `cycle_fingerprint`).
- Strong synergy with **Hot-Path Cypher Completion** (Zone 2 can expose clean Cypher-like surfaces via spear).
- Natural extension of **Grammar + VSA + TEKAMOLO** work (grammatical roles from text can hydrate into OGIT entities in any zone).
- `CausalEdge64` + Pearl masks become the common causal currency across all zones.
- `jc` mathematical proofs provide soundness for aggressive zero-copy and binding decisions.

---

## Palantir Foundry + AGI Aspirations (Strategic Framing)

- **Inner (Zone 1)**: AGI-aspiring thinking (resonance, L1–L4 loop, continuous causal reasoning via `CausalEdge64`).
- **Mid (Zone 2 via spear)**: Foundry/HubSpot-aspiring operational surface — clean, queryable, realtime, but not forcing full AGI complexity on consumers.
- **Outer (Zone 3)**: Foundry-style interoperability with the outside world (serialization, long-term storage, external systems).

OGIT acts as the **semantic bridge** that keeps inner and outer ontology consistent per domain while allowing different performance and serialization characteristics.

---

**This is a major, cross-cutting architectural vision.** It has excellent coherence with everything we have explored so far (SoA unification, `CausalEdge64`, hot-path Cypher, grammar/VSA/TEKAMOLO, multi-repo OGIT hydration).

The main debt right now is **lack of crisp Zone 2 ownership** (`spear`) and **incomplete uniform OGIT hydration** across all domain repositories.

Ready for the next area whenever you are. 🌸