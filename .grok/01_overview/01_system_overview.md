# 01_overview — High-Level System Map & Key Invariants

**Purpose**: Single source of truth for the overall architecture of the Resonance-Based Cognitive System in `lance-graph`. Low-entropy entry point and high-level map. Aligned with current multi-zone, L1–L4 shader, spear bridge, and OGIT understanding (as of 2026-05-10).

**Status**: Active / Living  
**Last Updated**: 2026-05-10

---

## 1. System Vision & Aspirations

**lance-graph** implements a **hybrid vector-symbolic, resonance-driven cognitive architecture** designed for **continuous thinking**.

Key inspirations and differentiators:
- Resonance-based field computation (holograph + cam_pq + HDR cascade) producing stable attractors
- Compact causal registers (`CausalEdge64`) as universal atoms carrying Pearl + NARS + plasticity
- Explicit Pearl causal hierarchy (2³ masks) treated as parallel representational dimensions
- NARS-style reasoning embedded at both **atomic** (`CausalEdge64`) and **meta** (`MetaOrchestrator` / `StyleTopology`) levels
- Self-regulating meta-orchestration via Thinking Styles + MUL (Meta-Uncertainty Layer)
- Mathematical verification & soundness guarantees via the `jc` (Jirak-Cartan) crate running in CI
- Deliberate **multi-zone** performance + serialization split while preserving semantic unity through **OGIT** ontology spine

**Dual Aspiration** (Inner vs Outer):
- **Inside the binary (Zone 1 + Zone 2 transcode)**: AGI-grade continuous reasoning — never-stopping L1–L4 closed loop, meta-cognition, self-modifying strategies, cross-domain causal trajectories
- **Outside (Zone 3 via spear)**: Practical, consumer-friendly surfaces inspired by **Palantir Foundry** (ontology-driven apps, intelligence surfaces) and **HubSpot** (operational CRM/workflows). `spear` reverse-engineers HubSpot-like ergonomics while absorbing transcodes (SharePoint with OGIT messaging + ShareGate-like QoS, Hiro ticket orchestration, Stalwart Mailserver stubs, domain repos like MedCare-rs, Woa-rs, Smb-office-rs)

The architecture avoids forcing AGI complexity on external consumers or even on spear's Zone 3 business logic.

---

## 2. Multi-Zone Ontology Architecture (Core Structural Invariant)

The system is intentionally split into three zones with different latency, serialization, and complexity profiles. **OGIT** (3–5 byte `g` pointers + CAM codebook + content-addressable schema) acts as the semantic bridge enabling O(1) schema/label switching and consistent inner/outer ontology views per domain.

| Zone | Target Latency | Serialization | Key Components | Primary Purpose | Ontology Characteristics |
|------|----------------|---------------|----------------|-----------------|--------------------------|
| **Zone 1 — Hot Inner** | 20–200 ns | None (pure in-memory, register-level, zero-copy) | `cognitive-shader-driver` (SoA wiring), `BindSpace` (Vsa16kF32 fingerprints + native `CausalEdge64` storage), `CollapseGate`, L1–L4 square shaders, `CausalEdge64` | Continuous resonance, collapse, thinking cycles, learning, meta-orchestration | Rich inner AGI view: full causal + epistemic state, Pearl masks, NARS truth, plasticity, cross-domain witnesses (SPO-G/SPO-W) |
| **Zone 2 — Mid-Tier / Cold Consumer Path** | 2–200 ms | None *inside* the binary (pure transcode layer) | `lance-graph-callcenter` (transcode), `spear` (Zone 2 side) | High-performance, zero-copy access to inner ontology for operational surfaces | OGIT + CAM codebook inheritance (O(1) `g` pointers). Clean, typed, consumer-friendly APIs over BindSpace |
| **Zone 3 — Outer Serialized** | ms – seconds (acceptable) | Full (JSON, gRPC, MySQL Parallelbetrieb, external APIs, mail, SharePoint) | `spear` (Zone 3 side + business logic), domain repositories (MedCare-rs, Woa-rs, Smb-office-rs, transcoded SharePoint/Hiro) | External integrations, long-term persistence, consumer-facing apps & workflows, intelligence surfaces | Palantir Foundry / Gotham Neo4j aspiring (rich serializable objects, links, provenance) + OGIT semantic spine for consistency |

**Spear Bridge Rule (Critical)**: `spear` (AdaWorldAPI/Spear) **deliberately clicks into both Zone 2 and Zone 3**. 
- Zone 2 side: zero-copy `BindSpace` / `cognitive-shader-driver` access via `lance-graph-callcenter`
- Zone 3 side: owns all serialized I/O and business logic (HubSpot reverse-engineering, ticket systems, mail, SharePoint QoS pipelines, invoices/emails unified DTOs)
- **Why**: HubSpot-like CRM, ticket orchestration (Hiro/datagroup/almato), and external system integrations inherently require serialization and I/O. This complexity is kept **out of `lance-graph` core and pure Zone 2** so the hot path and transcode layer stay lean and focused on cognition + zero-copy ontology access.

All domain repos are compiled into the same binary for tight integration while `spear` provides separation of concerns.

---

## 3. L1–L4 Cognitive Shader Model + CausalEdge64 (Inner Engine)

The hot inner thinking runs on **regular, square, power-of-two cognitive shader surfaces** for efficient resonance, attention, and semantic lookup:

| Layer | Matrix Size | Primary Role | Connection to Architecture |
|-------|-------------|--------------|----------------------------|
| **L1** | 64×64 | `CausalEdge64` — atomic causal + epistemic register | Foundation of all loops. Packs S/P/O palettes, NARS frequency/confidence/truth, Pearl 2³ masks (8 parallel causal perspectives), plasticity state, temporal index. Lives natively in `BindSpace.edges` |
| **L2** | 256×256 | Palette ranking / **attention mask** | Modulates priority of L1 planes. Part of resonance → collapse flow |
| **L3** | 4096×4096 | **CAM semantic codebook** (Coca — calibrated native English speaker vocabulary) | High-dimensional semantic surface for concept binding, similarity, grounding |
| **L4** | (Planned +4096) | **3-byte OGIT ontology** (O(1) schema + label switch) | Explicit, hot-switchable OGIT layer. Content-addressable memory index + labels. Can be swapped without affecting L3 |

**SPO-G and SPO-W (Witness) Extension** (cross-domain / meta layer):
- **SPO-G**: Cross-domain / meta-awareness reference and bundling surface for semantic trajectories
- **SPO-W (Witness)**: Meta-awareness / reference surface drawn from Markov chain role bundling (grammar heuristics + TEKAMOLO trajectories). Enables promotion of high-signal structures across domains (clinical + business + OSINT)

These live in L3/L4 space and are promoted via `CollapseGate` when entropy or resonance thresholds are met. Enable powerful cross-domain graph reasoning and diagnostics.

**CausalEdge64 as Universal Register**:
- One `u64` self-describing atom sufficient for hot-path causal reasoning, NARS inference steps, Pearl do-calculus masks, plasticity decisions, and temporal tracking.
- **Pearl 2³ Masks**: 8 explicit causal perspectives treated as dimensions → true multi-perspective reasoning. Accuracy + weak-dependence proofs live in `jc` crate and run in CI.
- **PlasticityState**: Per-plane hot/cold/frozen flags (clinically and architecturally superior to global learning rates).

**cycle_fingerprint** (Vsa16kF32): Unit of thought emitted per cognitive cycle. First-class citizen in `BindSpace`.

---

## 4. Continuous Thinking Loop (L1–L4 Closed Loop + Promotion Membrane)

The architecture is deliberately **never-stopping**:

1. **Resonance** — holograph + cam_pq + HDR cascade (L0–L3 field) produces interference patterns and stable peaks
2. **Collapse / Decision** — `CollapseGate` uses `CausalEdge64` fields + plasticity + entropy budget to decide mutation, promotion, freeze
3. **Shader Processing** — L1–L4 square matrices perform attention, semantic lookup, causal inference, NARS steps
4. **Thought Emission** — `cycle_fingerprint` + updated `CausalEdge64` planes
5. **Meta-Orchestration** — `MetaOrchestrator` + Thinking Styles (Plan/Act/Explore/Reflex clusters) + MUL adapt strategy at meta level (NARS-style)
6. **Learning & Plasticity** — Selective per-plane updates; feedback into resonance field
7. **Promotion Membrane** — Elevates high-value causal structures, cross-domain witnesses (SPO-G/W), and ontology elements across zones/domains while respecting entropy budgets

**Core Invariants**:
- Order preservation where semantically required (e.g., DB change ordering)
- Strong mathematical guarantees on dependence, concentration, and causal semantics (jc crate)
- Selective plasticity > global rates
- OGIT as single source of truth for schema + labels (inner rich view + outer practical view)
- Clean Zone 2 ↔ Zone 3 bridge via spear (no HubSpot/ticket leakage into core)
- Continuous operation (no "stop and think" discrete steps)

---

## 5. Key Components & Layer Map

| Component | Location | Role | Documentation |
|-----------|----------|------|---------------|
| `CausalEdge64` + Pearl + Plasticity | `crates/causal-edge/` (`edge.rs`, `pearl.rs`, `plasticity.rs`) | Atomic causal/epistemic register | `02_core_primitives/causal_edge64.md` |
| `BindSpace` + `cycle_fingerprint` | `cognitive-shader-driver` (SoA layer) | Canonical high-performance substrate for hot path | `03_cognitive_layers/cognitive_shader_driver.md` |
| `cognitive-shader-driver` | `crates/cognitive-shader-driver/` | SoA wiring, dispatch, L1–L4 orchestration, hot-path Cypher bridge | `03_cognitive_layers/cognitive_shader_driver.md` + `05_query_languages/cypher_implementations.md` |
| `MetaOrchestrator` + Thinking Styles + MUL | `crates/lance-graph/src/graph/arigraph/orchestrator.rs` + related | Meta-cognition, NARS at meta level, adaptive strategy switching | `03_cognitive_layers/meta_orchestrator.md` |
| `jc` (Jirak-Cartan) crate | `crates/jc/` + `.github/workflows/jc-proof.yml` | Executable proofs (Pearl 2³, Jirak weak-dependence, concentration bounds, etc.) running in CI | `04_mathematical_foundation/jc_jirak_cartan.md` |
| Resonance Field | `crates/holograph/`, `cam_pq/` | L0–L3 field computation, interference, stable peaks, HHTL subspaces | `03_cognitive_layers/` (partial) |
| OGIT Ontology | Across crates + `lanceDB` CAM | 3–5B O(1) pointers, CAM codebook, schema hot-switch, inner/outer views | `MULTI_ZONE_ONTOLOGY_ARCHITECTURE.md`, `PALANTIR_FOUNDRY_INTEGRATION.md` |
| `spear` | `AdaWorldAPI/Spear` (external but compiled in) | Zone 2/3 bridge, HubSpot-like surface, transcodes (SharePoint, Hiro, Stalwart) | `MULTI_ZONE_ONTOLOGY_ARCHITECTURE.md`, `PALANTIR_FOUNDRY_INTEGRATION.md` |
| Domain Repos | `MedCare-rs`, `Woa-rs`, `Smb-office-rs`, etc. | Clinical, workorder/billing, SMB logic — all OGIT-hydrated | Same as above |
| NARS Full Integration | Dual (atomic + meta) | Full inventory, migration plan, before/after, DTO/SoA ownership | `03_cognitive_layers/NARS_THINKING_IMPLEMENTATIONS_INVENTORY_MIGRATION.md` |

---

## 6. Current Technical Debt Snapshot (High-Signal Only)

From `board/TECH_DEBT.md` and `board/LATEST_STATE.md` (2026-05-10):

**High Priority / High Entropy**:
- Hot-path Cypher completion & unification (multiple historical implementations 4–6 range; fragmentation; unify on canonical `OrchestrationBridge` + 3-byte polyglot tag vision)
- Full Promotion Membrane implementation and CausalEdge64 ↔ resonance integration
- Thinking Styles modulation of `CausalEdge64` fields / `CausalMask` selection

**Medium**:
- Uniform OGIT hydration + one-time LanceDB/DataFusion codebook generation across all repos (woa-rs Python WIP, SharePoint C# → spear Rust)
- Explicit L1–L4 shader surfaces + SPO-G/SPO-W residency and CollapseGate promotion rules

**Low / Conceptual Clear**:
- L1–L4 + SPO-G/W implementation details (square matrices, O(1) OGIT switch)

**Positive (Debt Being Actively Reduced)**:
- Strong Canonical vs LAB-ONLY boundary in `cognitive-shader-driver`
- `BindSpace` + native `CausalEdge64` as excellent, clean substrate
- `jc` proofs in CI provide formal license for aggressive VSA/tensor designs
- Clear multi-zone model + spear as deliberate Zone 2 ↔ Zone 3 bridge
- NARS inventory as single source of truth (prevents 40%+ duplication)

**Entropy Reduction Rule**: When overwhelmed, ask "Which `CausalEdge64` fields or which Thinking Style is active here?"

---

## 7. Documentation & Workflow Principles

- **Low Entropy**: One primary concept per file; heavy use of tables; explicit source paths; distinguish Epiphany vs Observation
- **Living Skeleton**: Update relevant `.grok/` file + `board/LATEST_STATE.md` + `epiphanies.md` after significant progress
- **Bootstrap Procedure** (every session):
  1. Read `boot.md`
  2. Check latest in `board/LATEST_STATE.md` + `board/EPIPHANIES.md`
  3. Use this `01_overview/` as high-level map
  4. Drill into specific section / crate
  5. End with "Next Session Starting Point" note

---

## 8. Quick Links & File Map

- **High-Level Map** (this file): `01_overview/01_system_overview.md`
- **Core Primitives**: `02_core_primitives/causal_edge64.md`
- **Cognitive Layers & Full NARS Inventory**: `03_cognitive_layers/`
- **Mathematical Foundation & CI Proofs**: `04_mathematical_foundation/`
- **Query Languages & Hot-Path Cypher**: `05_query_languages/cypher_implementations.md`
- **Living Board (Plans, Debt, Integration)**: `board/` — especially `LATEST_STATE.md`, `TECH_DEBT.md`, `MULTI_ZONE_ONTOLOGY_ARCHITECTURE.md`, `PALANTIR_FOUNDRY_INTEGRATION.md`, `EPIPHANIES.md`, `UNIFIED_SOA_SURFACE_PLAN.md`
- **Session Bootstrap**: `boot.md`
- **High-Signal Insights**: `epiphanies.md`
- **README & Navigation**: `README.md`

**Recommended Starting Point for Any Question**:
`boot.md` → `board/LATEST_STATE.md` → relevant part of this overview → specific deep file.

---

**This is the canonical high-level system map.** It will evolve as architecture solidifies. Keep it aligned with `board/MULTI_ZONE_ONTOLOGY_ARCHITECTURE.md` and `board/PALANTIR_FOUNDRY_INTEGRATION.md`.

*Feel free to expand with more focused sub-files inside `01_overview/` (e.g., `multi_zone_model.md`, `l1_l4_shader_details.md`) as patterns stabilize.*