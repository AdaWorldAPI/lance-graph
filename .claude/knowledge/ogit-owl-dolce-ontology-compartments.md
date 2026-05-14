# OGIT, OWL, DOLCE — Ontology Compartments and Global Addressing

> **READ BY:** integration-lead, truth-architect, certification-officer, anyone wiring ontology semantics into AriGraph / CausalEdge64 / SPO storage
>
> **PREREQUISITES:** `causal-edge-64-spo-variant.md`, `spo-schema-and-mailbox-sidecar.md`, `spo-ontology-format-stack.md`
>
> **Status:** CONJECTURE (architectural proposal; OWL DOLCE adoption + label inheritance are sprint-12+ scope)

---

## 1. Three Ontology Layers, Three Roles

The workspace eventually needs to support three ontology layers, each with a distinct role:

| Layer | What it provides | Where it lives | Role |
|---|---|---|---|
| **OGIT** | Domain-specific ontologies (Healthcare/MedCare, SMB, Q2-Cockpit, FMA, HubSpot, etc.) — named entity types + relations + roles | `lance-graph-contract::manifest` (codegen from YAML), `data/family_registry.ttl` | **Domain content** — what types of entities exist in each tenant's world |
| **OWL** | RDF semantics (subClassOf, sameAs, owl:Restriction, etc.) — formal type hierarchy + cross-type axioms | Oxigraph RDF triple store + AriGraph SPO-G | **Formal axiomatic structure** — how types relate via subsumption, equivalence, restriction |
| **DOLCE** | Upper ontology (Endurant/Perdurant, Quality, AbstractEntity, etc.) — the universal categorical scaffold | OWL ontology, imported alongside domain ontologies | **Categorical scaffold** — what KIND of thing this is in the broadest sense |

OGIT is **domain content** (5 specific tenants × their types). OWL is **formal structure** (subsumption + axioms). DOLCE is **categorical scaffold** (Endurant vs Perdurant vs Quality vs Abstract).

---

## 2. The CAM Codebook Connection (OGIT + CAM)

Per PR #366 (Sprint-7 W1 family-hydration) + CLAUDE.md "Model Registry":

OGIT entities map to **CAM codebook entries** via the Healthcare/SMB/Q2/etc. family registry:

```text
data/family_registry.ttl
  ↓ parsed by parse_family_registry()
  ↓ produces FAMILY_TABLE OnceLock
  ↓ Healthcare basins 0x10..=0x19 (FMA/SNOMED/ICD10/RxNorm/LOINC/MONDO/HPO/DRON/CHEBI/RadLex)
  ↓
CAM codebook layout:
  0x00..=0x0F  → universal / DOLCE / common
  0x10..=0x19  → Healthcare family ↑ (10 slots)
  0x20..=0x2F  → SMB family
  0x30..=0x3F  → Q2 family
  ...
  0x80..=0x82  → SMB Foundry slots
  0xA0..=0xAD  → SMB BSON slots (per OGIT PR #3 + S7-W7)
```

Each CAM codebook entry = one canonical entity archetype, addressable by its 8-bit family-prefixed index. **The CAM codebook IS the OGIT manifestation** in the quantized hot-path. Slot inheritance happens via prefix-matching: a Healthcare slot inherits the universal slots (0x00..=0x0F) + the Healthcare-family-specific slots.

---

## 3. Label Inheritance (OWL subClassOf in CAM)

When `MedicalDoctor subClassOf Physician subClassOf Person`, the CAM codebook can implement this via **slot inheritance**:

```text
slot 0x10 = Person       (universal scaffold)
slot 0x11 = Physician    (Healthcare-specific, inherits slot 0x10's attributes)
slot 0x12 = MedicalDoctor (refines slot 0x11)
```

Querying for "all Persons in this corpus":
- Without inheritance: scan all slots with explicit Person tag (only slot 0x10)
- With inheritance: bit-mask scan over slots 0x10, 0x11, 0x12 — all three carry the Person inheritance bit

**Implementation alternatives:**

| Alternative | Detail | Trade |
|---|---|---|
| **Bit-mask inheritance** | Each slot has an inheritance bitmap; "is this slot a Person?" = bitmap[Person] | Fast lookup; precomputed at codebook load |
| **Path-based inheritance** | Slot ID encodes a tree path (0x10 → 0x11 → 0x12 via prefix); ancestor queries walk path bits | Variable-length encoding; simpler inheritance check |
| **External hierarchy table** | Separate table maps slot_id → ancestors[]; lookup at query time | Most flexible; one extra cache miss per query |

The recommendation (CONJECTURE — needs profiling): **bit-mask inheritance** for the CAM codebook, because Healthcare typically has 10-30 type levels and bitmask-OR is one register op.

---

## 4. OWL DOLCE as Orthogonal Compartment

Per the user's framing: "OWL DOLCE as orthogonal OGIT ontology semantic compartment vs global addressing."

The key insight: **DOLCE is not a domain — it's a categorical scaffold that runs orthogonal to OGIT domains.**

```text
Global address space (8-bit CAM slot):
  ┌──────────────────────────────────────────────────────────────┐
  │ 0x00..=0x0F  Universal / DOLCE scaffolds                     │
  │   0x01 = Endurant                                             │
  │   0x02 = Perdurant                                            │
  │   0x03 = Quality                                              │
  │   0x04 = AbstractEntity                                       │
  ├──────────────────────────────────────────────────────────────┤
  │ 0x10..=0x1F  Healthcare (OGIT MedCare family)                │
  │   0x10 = Person                                               │
  │   0x11 = Physician   (Endurant × Person × Healthcare)        │
  │   0x12 = Diagnosis   (Perdurant × Diagnosis-event × HC)      │
  ├──────────────────────────────────────────────────────────────┤
  │ 0x20..=0x2F  SMB (OGIT smb-office family)                    │
  │ 0x30..=0x3F  Q2 (OGIT q2-cockpit family)                     │
  │ ...                                                            │
  └──────────────────────────────────────────────────────────────┘
```

A slot's **identity** has two coordinates:
- **Domain axis (OGIT family):** Healthcare / SMB / Q2 / ...
- **Categorical axis (DOLCE scaffold):** Endurant / Perdurant / Quality / Abstract

A `Physician` slot is simultaneously:
- In the Healthcare family (OGIT)
- An Endurant (DOLCE)
- A subClassOf Person (OWL)

**The 8-bit CAM slot encodes the domain × OGIT identity; DOLCE category is carried as a parallel inheritance attribute.** Queries can filter on either axis:

- "All Endurants in Healthcare" = `DOLCE_endurants ∩ Healthcare_slots`
- "All Perdurants across all domains" = `DOLCE_perdurants` (slot-prefix-agnostic)
- "All Physicians" = `Healthcare slots inheriting Person` (OGIT-specific)

---

## 5. Semantic Compartment vs Global Addressing

The user's question: how does the compartment vs global addressing play out with SPO-G, SPO-W, and CausalEdge64?

### 5.1 Compartment addressing (G-bound)

```text
Triple address: (G, S, P, O)
  G is a partition key — the slot 0x11 "Physician" means different things 
  in G_med vs G_legal (in legal, it might be a forensic-witness type).

  Query: "find all Physicians in G_med"
  Path:  filter by G = G_med, then by slot = 0x11
```

This is the **canonical RDF SPO-G pattern.** G is the security/tenant/belief boundary. Inside a G, slot addresses are namespaced.

### 5.2 Global addressing (DOLCE-bound)

```text
Triple address: (DolceCategory, S, P, O)
  DolceCategory = Endurant / Perdurant / Quality / Abstract — UNIVERSAL.
  
  Query: "find all Endurants (across all G)"
  Path:  filter by DolceCategory = Endurant, no G filter.
```

This is **cross-domain reasoning** via the DOLCE scaffold. A Physician (Healthcare) and an Attorney (Legal) are both Endurants × Person; cross-domain queries that match on Endurant traverse both.

### 5.3 How it plays out with CausalEdge64

Per the corrected sprint-10 v2 layout (`causal-edge-64-synergies-and-pr-trajectory.md` §4):

```text
CausalEdge64 fields with G/DOLCE awareness:
  S palette index (8b)  — points into compartment-prefixed CAM codebook entry
  P palette index (8b)  — same
  O palette index (8b)  — same
  (G implicit via per-tenant SoA partition)
  W slot (6-8b)        — points to witness corpus root
                          (corpus root anchor carries: tenant, G, belief, DolceCategory bitmap)
  Truth-band lens (2b)  — 4 lens states (incl. cross-compartment lens)
```

Each S/P/O index is **already compartment-scoped** by virtue of the 8-bit family-prefix layout (slot 0x11 IS a Healthcare Physician; not a generic Physician). The G partition is implicit in the SoA (per-tenant supervision); the W corpus carries the full (G, DolceCategory) tuple at its root for cross-compartment joins.

### 5.4 How mailboxes should handle it

Per `spo-schema-and-mailbox-sidecar.md` §4.2 + W7 SigmaTierRouter spec:

- **Σ1-Σ8 mailboxes** (cycle-speed, ms to 200 ns):
  - All edges share the supervisor's tenant ⇒ same compartment ⇒ no G filtering needed at mailbox level
  - DolceCategory is queryable via CAM slot prefix bits — no separate filter needed
  - Routing happens purely by SPO palette indices + Σ-tier

- **Σ9-Σ10 escalation mailboxes** (EPIPHANY-tier):
  - May carry cross-compartment witness chains
  - Witness corpus root anchor explicitly names the (G, DolceCategory) of origin
  - Receiving compartment can refuse via `CognitiveBridgeGate` (per PR #366 wire) — Chinese-wall fires before reasoning
  - DolceCategory-matched cross-compartment queries become **brokered**: supervisor coordinates, no direct mailbox-to-mailbox traffic

---

## 6. Unifying Thinking-Engine Richness with Ontology Schema

The thinking-engine 8-channel CausalEdge64 has rich cognitive operators (BECOMES / CAUSES / SUPPORTS / REFINES / GROUNDS / ABSTRACTS / RELATES / CONTRADICTS). The OWL DOLCE / OGIT ontology has rich categorical structure. How do they compose?

### 6.1 Channel-to-axiom mapping

| 8-channel operator | OWL / DOLCE meaning | OGIT consequence |
|---|---|---|
| **BECOMES** | `dolce:overlaps` between two Perdurants | Time-evolution: this event becomes that state |
| **CAUSES** | `dolce:participates(cause, effect)` between Perdurants | Causal-chain inference (Pearl Level 2) |
| **SUPPORTS** | `owl:sameAs` (with confidence c < 1) | NARS revision: same-statement merging |
| **REFINES** | `rdfs:subClassOf` (downward) | Taxonomic specialization |
| **GROUNDS** | `dolce:characterizes(quality, endurant)` | Abductive justification (the Quality grounds the Endurant) |
| **ABSTRACTS** | `rdfs:subClassOf` (upward) | Taxonomic generalization |
| **RELATES** | `dolce:participates / dolce:hasPart / etc.` (generic relation) | Lateral semantic neighborhood |
| **CONTRADICTS** | `owl:disjointWith` | Negation, refutation, mutual exclusion |

This mapping makes the 8-channel cascade variant **a runtime interpretation of OWL axioms**. The cascade's energy perturbations ARE the operationalization of subsumption / equivalence / disjointness in the cycle-speed loop.

### 6.2 Ontology-aware splat search

Per `tetrahedral-epiphany-splat-integration-v1.md`: the 4096×4096 surface is "a deterministic question surface" not a distance ledger:

```text
[A{4096}, B{4096}]::[projection_mask]() -> C{4096}
```

With ontology awareness:
- `projection_mask` is an **OWL filter** — e.g., "only project to slots that are subClassOf Physician"
- The splat respects DOLCE category boundaries (Endurants don't splat into Perdurant slots)
- The 8-channel cascade routes attention through ontology-compatible neighbors only

This is how **the thinking-engine operational richness gets unified with ontology schema synergies**: each channel's effect is filtered by the ontology axioms; cross-channel composition is guided by DOLCE category-compatibility; the splat doesn't disperse uniformly — it follows ontology pathways.

### 6.3 Operationalization

For sprint-12+ (per `cognitive-shader-driver-thinking-engine-reunification.md`):

```rust
// thinking-engine::layered::TierEngine gains ontology filter
impl TierEngine {
    pub fn emit_causal_edges_filtered(
        &self,
        k: usize,
        ontology_filter: &OntologyFilter,
    ) -> Vec<(u16, CausalEdge64)> {
        // ... existing top-k search ...
        // ... existing neighbor selection ...
        for neighbor in neighbors.iter().take(4) {
            if !ontology_filter.allows_channel(peak, neighbor, channel) {
                continue; // ontology says this transition isn't allowed
            }
            // ... emit edge with the channel's strength
        }
    }
}
```

The `OntologyFilter` reads OWL axioms + DOLCE categories + OGIT family registry and returns per-(channel, source, target) allow/deny. This is the **bridge between OWL formal semantics and 8-channel energy dynamics**.

---

## 7. Recommendations

For sprint-10 + sprint-12+:

1. **Sprint-10 v2 CausalEdge64 SPO variant** keeps G implicit in per-tenant SoA; W-slot carries corpus root which names the (G, DolceCategory) tuple.

2. **OGIT family registry stays the codebook source** (per PR #366). Slot layout: `0x00..=0x0F` universal/DOLCE, `0x10+` domain-prefixed.

3. **DOLCE category bitmap is parallel attribute to CAM slot**. Each slot in the codebook gets a 4-8 bit DOLCE category mask (Endurant / Perdurant / Quality / Abstract + AnimateEndurant, NonAnimate, Event, Process, etc.).

4. **OWL inheritance via bit-mask**. Precompute at codebook load; ancestors[slot] is a bitmap. `is_a(slot, target_type)` = `(ancestors[slot] & (1 << target_type)) != 0`.

5. **Channel-to-axiom mapping (per §6.1)** becomes the canonical interpretation of thinking-engine 8-channel CausalEdge64 in ontology-aware reasoning.

6. **OntologyFilter trait (sprint-12+)** wires OWL/DOLCE/OGIT semantics into the 8-channel cascade via `emit_causal_edges_filtered`.

7. **Cross-compartment Σ9-Σ10 escalation** goes through `CognitiveBridgeGate` (PR #366) for Chinese-wall enforcement before any cross-G reasoning fires.

---

## 8. Cross-references

- `causal-edge-64-spo-variant.md`, `causal-edge-64-thinking-engine-variant.md`, `causal-edge-64-synergies-and-pr-trajectory.md` — the dual-variant analysis
- `spo-schema-and-mailbox-sidecar.md` — SPO-G + SPO-W schema design (G vs W roles)
- `spo-ontology-format-stack.md` — CAM-PQ codebook structure
- `cognitive-shader-driver-thinking-engine-reunification.md` — how thinking-engine and SoA reunite
- `lab-vs-canonical-surface.md` — UnifiedStep / OrchestrationBridge canonical surface
- `.claude/plans/lance-graph-ontology-v5.md` — workspace ontology plan
- `.claude/plans/oxigraph-arigraph-cognitive-shader-soa-merge-v1.md` — Oxigraph RDF integration
- `data/family_registry.ttl` — OGIT family seed data (PR #366 S7-W1)
- `lance-graph-contract::manifest` — codegen target consuming family_registry.ttl

---

*Authored 2026-05-14. Recommendations CONJECTURE pending implementation; OGIT family slot layout is FINDING (verified against PR #366 S7-W1).*
