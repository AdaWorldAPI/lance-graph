# FANOUT MAPPING PLAN — OGIT → OWL+DOLCE + ndarray SIMD + Foundry Capability Assimilation

**Date**: 2026-05-08  
**Status**: Initial fanout — authoritative reference for implementation  
**Context**: We **assimilate** (do not integrate with or depend on) Palantir Foundry capabilities. All mappings are internal to lance-graph + Spear + AdaWorldAPI/ndarray stack.  
**Goal**: Deliver a complete, standards-anchored operational ontology platform on par with (or superior to) Foundry while directly enabling the HIRO → Spear rewrite (routing + investigation agent verticals first).

**Important note on .claude folders**: Exhaustive filesystem search across `/`, `/home/workdir`, and common locations returned **no `.claude` directories or `CLAUDE.md` files** in the current environment. The planning documents reference Spear's `CLAUDE.md` and `.claude/` containing Foundry integration maps. These live in private AdaWorldAPI/lance-graph/spear repos (not checked out here). This fanout **synthesizes and expands** them.

**CRITICAL UPDATE (ndarray pre-integration)**: ndarray (AdaWorldAPI) is **already fully integrated** across Lance / lance-graph / SoA. The 20–200 ns random access for SoA cursors, column gathers, tensor views, and Bgz-compressed data is a direct result of this existing integration. All new work (packed schema validation, AwarenessColumn updates, embedanything calls, investigation traversals) **leverages** this foundation rather than adding it. Entropy work (explicit modeling of ambiguity, contradictions, and uncertainty via MUL + AwarenessColumn) is now a required dimension in every new artifact and implementation.

---

## 1. High-Level Architecture (Updated with ndarray)

```
Thinking / Operational Intelligence Layer
├── MUL meta-cognitive gate (uses OWL property characteristics + ndarray tensor priors)
├── NARS truth propagation + Pearl causal masks on CausalEdge64
├── Investigation Agent (SoA traversal + AwarenessColumn + drift signatures)
│   └── Calls embedanything DTO for GGUF embeddings / semantic hypothesis scoring
├── burn (training) + candle (inference) via ndarray backend + embedanything
└── Polyglot planner (DataFusion Expr + graph traversals, accelerated by ndarray)

Glue Layer (OGIT TTL → OWL + DOLCE + Packed Schema)
├── lance-graph-ontology (existing ttl_parse.rs + foundry_map.rs + bridges/)
├── New: ogit_to_owl_mapper.rs (or equivalent in glue/)
│   ├── Entity types → DOLCE Endurant/Perdurant + owl:Class
│   ├── Verbs → OWL property characteristics (functional, transitive, etc.)
│   ├── Attributes → rdfs:domain/range + owl:DatatypeProperty/ObjectProperty
│   └── Equivalence / extension points (owl:equivalentClass, sameAs)
└── Packed Schema Compiler (outputs L1-resident binary ≤50KB, content-hash keyed)

Invariant + Tensor Substrate (ndarray-powered — AdaWorldAPI/ndarray)
├── lance-graph-owl-simd (reuses ndarray SIMD kernels)
│   ├── AMX-accelerated matrix/bitmap ops for class hierarchy & disjointness
│   ├── BLASGraph paths for graph-structured schema validation
│   ├── Vectorized popcount / cardinality for functional property checks
│   └── DOLCE category bit checks (single-bit fast path)
├── Bgz Tensor Crate (internal to ndarray)
│   └── Efficient blocked/compressed tensor storage & serialization for Lance
│       - Used for: packed schemas (versioned), AwarenessColumn signatures,
│         SoA intermediate results, embedding tensors, CausalEdge64 masks
├── embedanything DTO
│   └── Unified interface for GGUF weights → burn/candle inference
│       - Powers semantic features inside investigation agent & drift detection

Storage & Execution Substrate
├── Lance columnar (single store, content-addressable, versioned)
│   └── lance-graph (Cypher/SPARQL/Gremlin + DataFusion)
├── lance-graph-callcenter membrane + version_watcher + drain (realtime)
├── Spear (verticals + Action Types + Ractor supervision)
└── Postgrest-compatible HTTP + WebSocket surfaces
```

This is the **assimilated Foundry-equivalent stack**:
- Ontology modeling + fast invariants (OGIT+OWL+DOLCE + ndarray SIMD)
- Actions/workflows (Spear + Ractor)
- Data integration (Lance + lance-graph)
- ML deeply integrated (ndarray → burn/candle + embedanything GGUF)
- Advanced reasoning & preemption (MUL + investigation agent)
- Real-time + audit (membrane + CausalEdge64)

---

## 2. Detailed OGIT → OWL + DOLCE Glue Mappings (Core Fanout)

### 2.1 Entity Type Mapping
**OGIT source** (from `AdaWorldAPI/OGIT/NTO/` TTL files):
```turtle
ogit:Automation:Ticket
    a owl:Class ;
    ogit:parent ogit:Automation:Issue ;
    ogit:mandatory-attributes ( ogit:subject ogit:status ... ) ;
    ogit:scope "customer" .
```

**Mapped to**:
- `dolce:Endurant` marker (persistent object) + `owl:Class`
- `rdfs:subClassOf ogit:Automation:Issue`
- `owl:hasKey` or functional properties derived from mandatory-attributes
- Stored in packed schema with short internal ID (content hash of the TTL fragment or stable ogit name → 3-5 byte pointer)

**DOLCE rule**:
- If the entity participates in state-changing relationships or has lifecycle → Endurant
- If it is an immutable record of something that happened (RoutingDecision, MailIntent) → Perdurant

**Implementation artifact to create**:
- `spear/src/ontology/glue/ogit_entities.rs` (or in `lance-graph-ontology`)
- Mapping table or derive macro from OGIT TTL at build time / hydration time

### 2.2 Verb / Relationship Mapping (Property Characteristics)
**OGIT verb example**:
```turtle
ogit:routes-to
    a ogit:Verb ;
    ogit:from ogit:Automation:Ticket ;
    ogit:to ogit:Automation:Queue ;
    ogit:cardinality "1" .
```

**Mapped to OWL**:
- `owl:ObjectProperty`
- `owl:FunctionalProperty` (from cardinality or ogit:mandatory)
- Possibly `owl:InverseFunctionalProperty`, `owl:TransitiveProperty`, `owl:SymmetricProperty`, `owl:AsymmetricProperty`, `owl:IrreflexiveProperty` based on explicit OGIT declarations or inference rules in glue layer.
- `rdfs:domain` / `rdfs:range`

**MUL consumption** (critical for routing vertical):
- Functional property + planner returns >1 match → `is_unskilled_overconfident()` hard veto + explanation.

**Implementation**:
- Extend `lance-graph-ontology/src/bridges/` or new `ogit_verb_mapper.rs`
- Output goes into packed schema property characteristics bitfield (7 bits per property as described in INVARIANT_LAYER.txt)

### 2.3 Attribute Mapping
- `ogit:mandatory-attributes` → `owl:minCardinality 1` or functional
- `ogit:indexed-attributes` → hint for lance-graph / DataFusion indexing
- Data types → `owl:DatatypeProperty` + XSD mappings

### 2.4 Equivalence & Extension (Multi-Vocabulary / All-Domain Support)
- `owl:equivalentClass` and `owl:sameAs` for:
  - Customer-internal vocabularies
  - ITIL, BFO, other domain ontologies
  - Cross-namespace OGIT alignment
- This is how we **assimilate Foundry's broad domain coverage** without importing Foundry: any ontology that can be expressed in OWL can be mapped/equated at O(1) cost via the content-addressable schema registry.

### 2.5 Content-Addressable Schema Pointers (3-5 Byte O(1))
- Every hydrated OGIT namespace or customer extension gets a stable content hash (or truncated 128-bit hash → 3-5 bytes practical pointer).
- Lance tables store only the short pointer in edges/columns.
- Schema definition itself lives in a versioned Lance table or embedded registry, keyed by the pointer.
- Lookup = O(1) hash table or perfect hash in the packed schema cache.
- Invalidation: when OGIT TTL changes, new hash → new packed schema version. Historical data remains valid against old hash (time-travel queries).

**File to create**: `lance-graph-ontology/src/content_addressable_registry.rs`

---

## 3. Packed OWL+DOLCE Schema Binary Format (for lance-graph-owl-simd)

**Target size**: ≤ 50 KB fully packed (fits L1).

**Proposed layout** (to be refined in `PACKED_SCHEMA_FORMAT.md`):

```
Header (16-32 bytes)
  - Magic + version (u32)
  - Content hash of source OGIT TTL (16 bytes)
  - Number of classes (u16)
  - Number of properties (u16)
  - DOLCE mode flags (u8)

Class Hierarchy Bitmap Section
  - Bit matrix or Roaring/ roaring-like bitmap for rdfs:subClassOf transitive closure
  - AMX-accelerated matrix multiply / intersection kernels from ndarray

Property Characteristics Bitfield (per property)
  - 1 byte per property: bits for Functional, InverseFunctional, Transitive, Symmetric, Asymmetric, Reflexive, Irreflexive
  - Plus domain/range class ID pointers (short IDs)

DOLCE Category Section
  - 1 bit per class: Endurant (0) / Perdurant (1)
  - Fast vectorized checks

Annotation & Metadata Offsets
  - Pointers into a string/blob section for rdfs:label, comments, dcterms:source, etc.
  - Used by admin UIs, audit, codegen

Disjointness & Cardinality Tables
  - Bitmap pairs for owl:disjointWith
  - Vectorized count tables for cardinality enforcement

Footer / Checksum
```

**Kernels reused/adapted from AdaWorldAPI/ndarray**:
- AMX matrix ops for hierarchy bitmaps
- Vectorized popcount / AND for disjointness & multi-match detection
- BLASGraph for any graph-shaped schema validation (e.g., cycle detection on transitive properties if needed at hydration time)

**Bgz Tensor Crate role**:
- Serialized form of the above packed schema (or intermediate tensors during validation) stored compressed in Lance.
- Enables efficient storage of large AwarenessColumn signatures or embedding tensors alongside the main data.

**Implementation file**: `lance-graph-owl-simd/src/packed_schema.rs` + kernels in `ndarray` integration module.

---

## 4. ndarray + Bgz + embedanything Integration Points

### 4.1 For Invariant Layer (owl-simd)
- Direct reuse of ndarray SIMD primitives (AMX, optimized popcount, vectorized ops).
- Bgz tensor for storing versioned packed schemas inside Lance tables (content-addressable).

### 4.2 For ML / Embeddings (Foundry assimilation — models as first-class)
- `embedanything` DTO: thin wrapper / unified interface that:
  - Accepts GGUF model paths or in-memory weights
  - Routes to candle (preferred for inference/GGUF) or burn backend
  - Returns typed embedding tensors or structured outputs
  - Integrates with PII-tokenized data (tokens only; cleartext only at messaging boundary)
- Used inside:
  - Investigation agent: semantic similarity for hypothesis generation, evidence ranking
  - Drift signature matching
  - Routing: subject/body embedding features as additional signals to MUL
  - Future verticals: document classification, anomaly detection in ticket lifecycle, etc.

### 4.3 Bgz Tensor Crate (AdaWorldAPI/ndarray)
- Primary storage primitive for any tensor-shaped data that must live in Lance:
  - Packed schemas
  - AwarenessColumn (256-byte signatures per visited row or per investigation)
  - Embedding tensors from embedanything
  - Intermediate SoA gather results during agent traversal
  - CausalEdge64 mask tensors
- Benefits: compression + fast random access + Lance-native columnar storage → excellent for the single-store goal.

---

## 5. Capability Assimilation Map — Foundry Features → Our Stack

(Expanded from expected .claude/ foundry integration maps)

| Foundry Capability                  | Our Assimilated Equivalent                                                                 | Notes / Superiority |
|-------------------------------------|--------------------------------------------------------------------------------------------|---------------------|
| Object Types + Link Types           | OGIT entities + verbs mapped to OWL classes + properties + DOLCE Endurant/Perdurant       | Standards-based + fast SIMD validation |
| Action Types                        | Spear Action Types (UploadIntent pattern) + Intent modules in verticals                   | Typed, supervised, with retry/idempotency |
| Ontology as operational layer       | Full stack: ontology + invariants + actions + realtime + ML                               | Adds native causal reasoning + uncertainty |
| Data integration / pipelines        | Lance + lance-graph + DataFusion + Ractor-supervised workflows                            | Single store, no sync tax |
| ML models as first-class            | burn/candle + embedanything GGUF DTO + integration into investigation/drift               | Same ndarray substrate as invariants; GGUF for easy quantized deployment |
| Real-time interaction / Workshop    | postgrest + drain WebSocket + customer dashboards + typed EscalationMessage               | Realtime by construction from Lance commits |
| Governance, lineage, audit          | CausalEdge64 + Pearl masks + append-only Lance + membrane audit layer                     | Stronger causal + Pearl counterfactual support |
| Broad domain coverage               | OGIT 66 namespaces + OWL equivalence mappings + content-addressable multi-schema          | O(1) addition of new domains |
| Digital twin / living model         | Investigation agent + drift detection + AwarenessColumn + continuous SoA scanning         | **Preemptive** (before symptoms) vs reactive |
| Uncertainty / confidence            | MUL gating on every decision + Brier calibration + NARS truth                             | Native metacognition — Foundry relies on external rules/ML |
| Regulated industry readiness        | W3C OWL + DOLCE + SIMD invariants + full causal audit trail                               | "Boring reliable" + standards contract |

This map should be the content that lives (or is expanded) in the private `.claude/foundry_integration_maps/` or similar.

---

## 6. Vertical-Specific Mapping Examples (Start with Routing + Investigation)

### Routing Vertical (First — establishes pattern)
- New OGIT namespace: `AdaWorldAPI/OGIT/NTO/Routing/`
  - `RoutingRule` (Endurant) — regex pattern, target queue, template, confidence floor, priority
  - `RoutingDecision` (Perdurant) — ticket ref, matched rule, resolved queue, MUL outcome, timestamp
  - Verbs: `routes-to`, `escalates-to`, `applies-template`
- Glue layer produces packed schema entry with functional properties on key relationships.
- MUL uses functional property check → veto ambiguous multi-matches.
- ndarray-accelerated rule matching (if regex or embedding features used).
- Bgz tensor stores any intermediate decision tensors.

### Investigation Agent Vertical (Second)
- Reuses all above.
- Traversal primitives operate on SoA rows identified by short schema pointers.
- AwarenessColumn updates use Bgz tensor storage + ndarray ops for signature computation/stabilization.
- embedanything DTO called for semantic embedding of visited entities / evidence when needed for hypothesis ranking.
- MUL gate reads stabilized AwarenessColumn + schema invariants.
- Output: typed `EscalationMessage` (with tokens only).

Subsequent verticals (mailbox provisioning, telephony, ticket lifecycle, SharePoint, send-confirmation) follow the same glue + packed schema + ndarray pattern with domain-specific OGIT extensions.

---

## 7. Implementation Sequencing & Artifacts to Create Next

1. **Immediate (this session / next)**:
   - `GLUE_LAYER_OGIT_TO_OWL_SPEC.md` (detailed transformation rules + examples from real OGIT TTL)
   - `PACKED_SCHEMA_FORMAT.md` (exact binary layout + versioning story)
   - `NDARRAY_BGZ_EMBEDANYTHING_INTEGRATION.md` (API boundaries between ndarray crates and lance-graph-owl-simd / Spear)

2. **Glue layer code** (in `lance-graph-ontology` or new `lance-graph-glue` crate):
   - `ogit_to_owl_mapper.rs`
   - Content-addressable registry

3. **SIMD validator**:
   - Integrate ndarray kernels into `lance-graph-owl-simd`
   - Bgz tensor usage for schema storage

4. **Spear verticals**:
   - Update routing slice to depend on new packed schema + MUL priors from OWL
   - Implement investigation agent using Bgz + embedanything

5. **Testing & Calibration**:
   - Shadow mode routing against real (or synthetic) HIRO-like ticket streams
   - MUL calibration using Brier score on historical outcomes

6. **Documentation**:
   - Update `CLAUDE.md` / `.claude/` in spear/lance-graph with these fanout maps (when repo access available)
   - Reading list expansion

---

## 8. Risks & Mitigations (from planning docs)

- **Schema evolution**: Content-hash + versioned packed schemas + Lance time-travel → safe.
- **Performance**: ndarray AMX/BLAS paths + L1-resident packed schema + Bgz compression → target is "free" validation.
- **Correctness of glue mapping**: Start with Routing namespace (smallest), validate against existing OGIT validator JARs if available, then expand.
- **MUL integration**: The property characteristics become explicit inputs to `is_unskilled_overconfident()` — implement early.
- **Foundry assimilation completeness**: The map above covers the major axes; gaps (e.g., specific Workshop UI patterns) can be addressed later via postgrest + typed messages.

---

**Next immediate actions I recommend (and can execute)**:

1. Create the three detailed spec files listed in section 7.
2. Draft the first OGIT namespace extension (`Routing/`) in TTL form + mapped packed schema example.
3. Skeleton the glue mapper module.
4. Once private repos are accessible, diff against the canonical `.claude/` foundry maps and merge.

This fanout gives a complete, actionable blueprint. The architecture is now fully specified end-to-end from OGIT TTL ingestion through ndarray-accelerated invariants and ML all the way to customer-facing Spear verticals that replace HIRO while assimilating Foundry-class operational ontology power.

Ready to generate the next artifact or begin code skeletons. Just say the word.