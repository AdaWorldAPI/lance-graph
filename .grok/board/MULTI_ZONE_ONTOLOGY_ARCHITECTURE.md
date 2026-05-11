# MULTI_ZONE_ONTOLOGY_ARCHITECTURE.md — Zone 1 / Zone 2 / Zone 3 + OGIT Hydration Vision

**Date**: 2026-05-10 (updated with spear Zone 2 ↔ Zone 3 bridge clarification)  
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

### Zone 2 — Mid-Tier / BindSpace Surface (Cold Path Consumer Zone)
- **Latency target**: 2–200 ms (cold path)
- **Characteristics**: Zero-copy access to `BindSpace` inside the same binary, no serialization *inside the zone itself*. Pure in-process transcode and access layer to the hot ontology.
- **Current implementation**: `lance-graph-callcenter` — the **pure transcode layer** (in same binary as Zone 1). No Supabase Realtime. Provides zero-copy, high-performance bridge from outer surfaces into `BindSpace` / cognitive shaders.
- **Spear's role (bridges Zone 2 ↔ Zone 3)**: **spear** repository (AdaWorldAPI/Spear) **clicks into both Zone 2 and Zone 3**.
  - It interfaces with **Zone 2** via `lance-graph-callcenter` for zero-copy, high-performance access to the inner ontology (`BindSpace`, OGIT-hydrated views, CausalEdge64, etc.).
  - It owns **Zone 3** responsibilities for all serialized outbound/inbound integrations: JSON, gRPC, MySQL Parallelbetrieb, mail (Stalwart), SharePoint, external APIs, etc.
  - Harbors the **future HubSpot reverse-engineered operational surface** + ticket orchestration (Hiro/datagroup/almato).
  - Already contains stubs to Stalwart Mailserver.
  - Intended to absorb transcodes of AdaWorldAPI/SharePoint (ShareGate-like soft-pull QoS + OGIT messaging ontology).
  - **Why spear must span both**: HubSpot-like CRM, ticket systems, workorder/billing workflows, and external integrations inherently require serialization and I/O with the outside world. We deliberately keep this business logic **out of lance-graph / core engine** (and even out of pure Zone 2 transcode) so the hot path and callcenter stay lean, focused on cognition and zero-copy ontology access.
- **Ontology**: Same OGIT + CAM codebook inheritance as Zone 1 (3–5 byte `g` pointers for O(1) schema + label lookup) when talking to Zone 2. When crossing into Zone 3, spear produces rich serializable Foundry-style representations while preserving semantic consistency via OGIT.

### Zone 3 — Outer Ontology (Cold / Serialized Path)
- **Latency target**: Higher (ms to seconds acceptable)
- **Characteristics**: Full serialization (JSON, gRPC, MySQL Parallelbetrieb, etc.).
- **Purpose**: External integration, long-term persistence, cross-system interoperability.
- **Ontology**: Still aims to be **Palantir Foundry aspiring** per domain, with OGIT as the semantic spine.

---

## L1–L4 Cognitive Shader Dimensions + SPO-G / SPO-W Extension

The inner thinking engine (Zone 1) is built around **square, power-of-two cognitive shader surfaces** for efficient resonance, attention, and semantic lookup:

| Layer | Size          | Purpose                                      | Connection to Broader Architecture |
|-------|---------------|----------------------------------------------|------------------------------------|
| **L1** | 64×64        | `CausalEdge64` — atomic causal + epistemic register (S/P/O palettes + NARS truth + Pearl masks + plasticity) | Foundation of all L1–L4 loops. Native in `BindSpace`. |
| **L2** | 256×256      | Palette ranking / **attention mask**         | Modulates which CausalEdge64 planes are prioritized. |
| **L3** | 4096×4096    | **CAM semantic codebook** (Coca — calibrated native English speaker vocabulary) | High-dimensional semantic surface for concept binding and similarity. |
| **L4** | (Planned) +4096 | **3-byte OGIT ontology** (O(1) schema + label switch) | Explicit OGIT layer for content-addressable memory. Can be hot-switched without affecting L3. |

**SPO-G and SPO-W (Witness) Extension**:
- **SPO-G**: Cross-domain / Meta-awareness layer. Serves as a reference / bundling surface for semantic trajectories.
- **SPO-W (Witness)**: Meta-awareness / reference surface drawn from Markov chain role bundling (grammar heuristics + TEKAMOLO trajectories).
- These act as **semantic trajectory witnesses** that can reference or cross-link domain-specific SPO structures across MedCare-rs, Smb-office-rs, Woa-rs, q2, etc.
- They live naturally in the L3/L4 space and are promoted via `CollapseGate` when entropy or cross-domain resonance thresholds are met.

This design keeps the core L1–L4 loop extremely regular (square matrices) while allowing OGIT and cross-domain witness surfaces to be added as clean, swappable layers.

---

## Cross-Domain Repositories (All via OGIT Hydration)

All consumer repositories are compiled into the **same binary** as the core engine for tight integration while maintaining separation of concerns. The vision is that every domain repository becomes a first-class citizen by hydrating through **OGIT** (inner and outer ontology per domain = Palantir Foundry / Gotham aspiring).

| Repository          | Domain Focus                                      | Current State                          | OGIT Role / Integration                          | Notes |
|---------------------|---------------------------------------------------|----------------------------------------|--------------------------------------------------|-------|
| **spear** (AdaWorldAPI/Spear) | HubSpot-like + ticket orchestration (bridges Zone 2 ↔ Zone 3) | In design / Stubs present             | Bridges Zone 2 (zero-copy via callcenter + OGIT) and Zone 3 (serialized external I/O) | **Must span both zones** because HubSpot CRM, tickets (Hiro), mail, SharePoint, MySQL etc. require serialization. Keeps hubspot/ticket logic **out of lance-graph core**. Stubs to Stalwart; will absorb SharePoint + Hiro transcodes. |
| **Stalwart Mailserver** | Email / communication                            | External (stubbed in spear)           | Hydrate mail events into OGIT                   | Emails as first-class entities in spear |
| **SharePoint**      | Documents, invoices, workflows                    | C# (ShareGate-like 3-layer cache + soft-pull QoS) | Needs Rust transcode into spear crate           | Invoices use same DTO family as emails. OGIT messaging ontology. |
| **Hiro**            | Ticket orchestration (datagroup/almato)           | External                              | Transcode into spear                            | Workflow / ticket domain |
| **MedCare-rs**      | Clinical studies, patients, diagnostics           | Active                                | Strong cross-domain contributor                 | ViT + FMA + SNOMED → graph reasoning + diagnostics. High cross-domain potential. |
| **Smb-office-rs**   | Small business logic                              | Active (from C#)                      | Business entities + workflows                   | Overlap with Woa-rs. Core SMB domain. |
| **Woa-rs**          | Work orders / billing                             | Active (WIP — transcode from Python website) | Operational workflows + billing                 | Strong OGIT candidate. Overlap with Smb-office-rs. |
| **q2**              | OSINT, intelligence                               | Aspiring                              | Palantir Gotham / Neo4j style                   | High-value external integration target |
| **lance-graph-callcenter** | Transcode / Bridge layer (Zone 2)            | Existing (same binary)                | Bridge between Zone 1 (shader-driver) and spear | **No Supabase Realtime**. Pure transcode layer in same binary. |

**Key Principle**: All of the above should eventually flow through **OGIT semantic hydration** so that inner ontology (Zone 1) and outer ontology (Zone 3) per domain remain consistent and Palantir Foundry-like.

---

## Technical Debt & Gaps (Current — Updated with Fresh Context)

1. **spear bridging Zone 2 ↔ Zone 3 needs crisp contracts**
   - `lance-graph-callcenter` is confirmed as the **pure transcode layer** (zero-copy, no serialization, in same binary).
   - `spear` now clearly owns the bridge: Zone 2 side (zero-copy callcenter access + OGIT) **and** Zone 3 side (all serialized external I/O, HubSpot features, tickets, mail, SharePoint).
   - Still needs: crisp API contracts between spear's Zone 2-facing and Zone 3-facing parts, full absorption of SharePoint QoS pipelines + Stalwart stubs into spear, and clear separation so HubSpot/ticket logic never leaks into lance-graph or pure callcenter.

2. **SharePoint + Hiro transcoding into spear remains high priority**
   - C# 3-layer cache + soft-pull QoS logic needs clean Rust port.
   - Must preserve invoice/email DTO unification and OGIT messaging ontology.
   - Hiro ticket orchestration (datagroup/almato) should follow the same pattern.

3. **L1–L4 shader surfaces + SPO-G/W need explicit implementation plan**
   - 64×64 CausalEdge64, 256×256 attention, 4096×4096 Coca CAM, + optional 4096 OGIT layer are well-defined conceptually.
   - `SPO-G` (cross-domain/meta) and `SPO-W` (witness from Markov/grammar bundling) need concrete residency in L3/L4 and promotion rules via CollapseGate.
   - Integration with grammar heuristics (DeepNSM) and TEKAMOLO trajectories is still loose.

4. **OGIT hydration still needs uniform application across all repos**
   - woa-rs (Python transcode WIP), Smb-office-rs (C# origin), MedCare-rs, and spear consumers must all expose both inner (3–5 byte pointer + CAM) and outer (Foundry-style) views.
   - One-time LanceDB DataFusion codebook generation at startup is the target mechanism.

5. **Inner AGI vs Outer Palantir + spear bridge enforcement**
   - Zone 1 = pure AGI-aspiring hot path.
   - Zone 2 (callcenter) = lean zero-copy transcode.
   - spear = deliberate bridge owning both Zone 2 access and Zone 3 serialized business logic (HubSpot, tickets, external I/O). This split exists specifically to keep hubspot/ticket systems out of lance-graph.
   - Clear boundaries, consumer-friendly APIs on spear's Zone 3 surface, and no AGI/hot-path leakage into Zone 3 code paths are still emerging.

6. **Cross-domain reasoning wiring is powerful but incomplete**
   - MedCare-rs + Smb-office-rs + q2 + Woa-rs have excellent potential via OGIT + CausalEdge64.
   - SPO-G/SPO-W should become the explicit cross-domain witness layer.

---

## Recommended Evolutionary Path (Debt-Free — Updated)

### Phase 1 — Own Zone 2 Surface + Confirm No Supabase Dependency
- Formally own `spear` as the **Zone 2 HubSpot-like surface** (in same binary).
- Confirm `lance-graph-callcenter` as pure transcode layer (no Supabase Realtime).
- Document strict Zone 2 contract: zero-copy to BindSpace, OGIT + CAM only, consumer-friendly APIs.

### Phase 2 — L1–L4 Shader Surfaces + SPO-G / SPO-W Implementation
- Implement the square shader stack explicitly:
  - 64×64 `CausalEdge64`
  - 256×256 attention / palette ranking
  - 4096×4096 Coca CAM codebook
  - +4096 OGIT ontology layer (O(1) switch)
- Define residency and promotion rules for **SPO-G** (cross-domain/meta) and **SPO-W** (witness from Markov/grammar/TEKAMOLO bundling) as semantic trajectory surfaces.
- Wire grammar heuristics into L3/L4 promotion paths.

### Phase 3 — OGIT Hydration Standard + One-Time Codebook Generation
- Mandate that every domain repo (spear, MedCare-rs, Smb-office-rs, Woa-rs, SharePoint transcode, etc.) exposes:
  - Inner view: 3–5 byte `g` pointers + CAM codebook (Zone 1/2 compatible)
  - Outer view: rich Foundry-style serializable representation (Zone 3)
- Implement **one-time LanceDB + DataFusion codebook generation** at startup (~200 µs target).
- Make OGIT the single source of truth for schema + label + codebook inheritance.

### Phase 4 — spear Absorbs SharePoint + Hiro + Stalwart Stubs
- Port C# ShareGate-like 3-layer cache + soft-pull QoS pipelines into spear (Rust).
- Ensure unified DTO family (invoices = emails).
- Bring Hiro ticket orchestration and Stalwart Mailserver stubs under spear.
- Preserve OGIT messaging ontology.

### Phase 5 — Cross-Domain Reasoning + Inner AGI / Outer Palantir Enforcement
- Wire MedCare-rs + Smb-office-rs + q2 + Woa-rs through OGIT + SPO-G/SPO-W.
- Enable full `CausalEdge64` + Pearl/NARS + cross-domain diagnostics.
- Enforce clean separation: Zone 1 + shader-driver = AGI-aspiring (full L1–L4 + SPO-G/W). spear + domain surfaces = practical Foundry/HubSpot-like (consumer-friendly, no AGI leakage).

### Phase 6 — Zone Boundary Enforcement + CollapseGate Auditability
- Make `CollapseGate` decisions visible and policy-enforceable from spear (Zone 2) and external consumers (Zone 3).
- Define crisp serialization boundaries while preserving zero-copy guarantees inside the binary.

---

## Connection to Previous Work

- Builds directly on **unified SoA DTO surface** (`BindSpace` + `MetaWord` + `cycle_fingerprint`).
- Strong synergy with **Hot-Path Cypher Completion** (Zone 2 via spear can expose clean Cypher-like surfaces).
- Natural extension of **Grammar + VSA + TEKAMOLO** work — now explicitly connected via **SPO-G / SPO-W** semantic trajectory witnesses (Markov role bundling → L3/L4 promotion).
- `CausalEdge64` + Pearl masks + L1–L4 square shaders become the common causal + attention currency across all zones.
- `jc` mathematical proofs provide soundness for aggressive zero-copy, binding, and square-matrix shader designs.
- New explicit L1–L4 dimensions (64×64 → 256×256 → 4096×4096 Coca + optional OGIT layer) and SPO-G/W surfaces are now part of the canonical multi-zone vision.

---

## Palantir Foundry + AGI Aspirations (Strategic Framing)

- **Inner (Zone 1 — AGI aspiration)**: 
  - Continuous resonance + L1–L4 closed-loop thinking.
  - Square power-of-two cognitive shaders: **64×64 CausalEdge64** → **256×256 attention masks** → **4096×4096 CAM semantic codebook** (Coca native English vocabulary) → optional **+4096 3-byte OGIT ontology layer** (O(1) hot-switch).
  - `SPO-G` (cross-domain/meta) + `SPO-W` (witness from Markov/grammar role bundling) as semantic trajectory surfaces.
  - Full `CausalEdge64` + Pearl + NARS + plasticity engine.

- **spear (bridges Zone 2 ↔ Zone 3)**: 
  - **Palantir Foundry / HubSpot-aspiring** operational surface that deliberately spans the serialization boundary.
  - Zone 2 side: Clean, typed, high-performance zero-copy access to BindSpace / cognitive state / OGIT via `lance-graph-callcenter`.
  - Zone 3 side: Serialized external integrations, HubSpot CRM features, ticket systems (Hiro), mail, SharePoint, MySQL, etc. Owns the business logic that inherently requires I/O and serialization.
  - All consumer DTOs and workflows (MedCare-rs, Smb-office-rs, Woa-rs, SharePoint transcode, Hiro, Stalwart) hydrated through OGIT for semantic consistency.
  - Does **not** force full AGI complexity or hot-path details on external consumers or even on spear's own Zone 3 code paths.

- **Outer (Zone 3)**: 
  - **Palantir Foundry + Gotham Neo4j aspiring** interoperability.
  - Serialized (JSON/gRPC/MySQL), long-term storage, external systems.
  - Uses LanceDB + DataFusion tables with **one-time hydrated OGIT/OWL codebook generation** at startup.
  - Full provenance and cross-domain graph capabilities.

**OGIT** (with future OWL) is the **semantic spine** that makes inner (AGI) and outer (Foundry) ontology consistent per domain while allowing radically different performance characteristics. The 3–5 byte `g` pointer + CAM codebook enables O(1) schema/label switching across all zones and repositories.

---

## Inner AGI vs Outer Palantir — Clean Separation of Concerns

The architecture deliberately separates concerns while allowing controlled bridging:

- **Zone 1 (Hot Path)**: Pure AGI-aspiring continuous thinking engine (`cognitive-shader-driver` L1–L4 + `CausalEdge64` + `SPO-G`/`SPO-W` + `BindSpace`). Zero serialization, register-level speed.
- **Zone 2 (Mid-Tier Transcode)**: `lance-graph-callcenter` — zero-copy, in-process bridge to the hot ontology. No serialization. Lean and focused.
- **spear (bridges Zone 2 ↔ Zone 3)**: The practical operational surface. 
  - Zone 2 side: High-performance zero-copy access to BindSpace / OGIT / cognitive state via callcenter.
  - Zone 3 side: All serialized external integrations (JSON/gRPC/MySQL/mail/SharePoint/HubSpot APIs/ticket systems). Owns the HubSpot reverse-engineered CRM, ticket orchestration (Hiro), workorder/billing workflows, etc.
  - This split exists **precisely because we cannot (and should not) put HubSpot-like business logic or ticket systems inside lance-graph or pure Zone 2**.
- **Zone 3 (Outbound Serialized)**: External world — long-term persistence, cross-system I/O, consumer APIs. Uses rich serializable representations while OGIT keeps semantics consistent.

**Consumer-friendly surface**: spear + domain repos expose clean, typed APIs. External consumers (and even internal Zone 3 code) do **not** need to understand the full AGI machinery or hot-path details. They get Foundry/HubSpot-like ergonomics.

This is the core of the "inner AGI aspiration — outer Palantir aspiration" strategy. Spear is the deliberate bridge that makes the split workable.

---

**This is a major, cross-cutting architectural vision.** It has excellent coherence with everything we have explored so far (SoA unification, `CausalEdge64`, hot-path Cypher, grammar/VSA/TEKAMOLO, multi-repo OGIT hydration, and the deliberate spear bridge across Zone 2/3).

The main debt right now is **implementing the crisp spear Zone 2↔Zone 3 bridge contracts** (so HubSpot/ticket logic stays cleanly in spear) and **incomplete uniform OGIT hydration** across all domain repositories.

Ready for the next area whenever you are. 🌸