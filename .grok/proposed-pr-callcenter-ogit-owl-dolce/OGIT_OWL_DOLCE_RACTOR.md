# OGIT + OWL + DOLCE + Ractor in lance-graph-callcenter

**Status**: Design Phase  
**Date**: 2026-05-11  
**Goal**: Turn `lance-graph-callcenter` into the Rosetta Stone + Ractor orchestrator between spear and the high-performance lance-graph substrate (DnPath + CAM-PQ).

## 1. Architecture Vision

```
spear (verticals)
   ↓ (light contracts + need-to-have facades)
lance-graph-callcenter
   ├── Rosetta Stone Layer
   │    ├── DnPath (context-addressable) + enriched semantics
   │    ├── CAM-PQ + Schema Labels + Codebook mapping
   │    └── ogit + owl_dolce (semantic translation)
   ├── Ractor Orchestration Layer
   │    ├── Membrane supervisor
   │    ├── Intent dispatch actors
   │    └── ExternalMembrane impl (LanceDB + DataFusion)
   └── Contracts (lance_graph_contract)
```

**Key Rule**: Heavy ontology and semantic work stays behind the membrane. Spear consumers only see clean facades.

## 2. Phased Implementation Plan

### Phase 0 – Foundation (Contracts + Ontology Layer)
- Stabilize `lance_graph_contract` with core OGIT primitives (`Label`, `OntologyDto`, `Schema`).
- Define minimal OWL + DOLCE concepts that are useful for this domain (e.g., `Endurant`, `Perdurant`, `Quality`, `Role`, `Dependency`).
- Create `ogit` and `owl_dolce` modules (can start inside `lance-graph-callcenter`).

### Phase 1 – Semantic Substrate (Current Focus)
- Extend `DnPath` (or add `SemanticDnPath` / `DnPathWithOntology`) to carry:
  - OGIT schema label(s)
  - OWL class / DOLCE category
  - Optional CAM centroid reference
- Map CAM Codebook `label` + archetypes to OGIT/OWL concepts.
- Add helper functions for context-addressable semantic lookup.

### Phase 2 – Ractor Orchestration
- Introduce Ractor inside `lance-graph-callcenter`.
- Create membrane supervisor + actor pool for `ExternalMembrane` operations.
- Expose Ractor-friendly APIs so spear can supervise or dispatch.

### Phase 3 – Consumer Integration (Spear side)
- Spear verticals depend only on contracts + callcenter facade.
- Heavy lifting (ontology, CAM-PQ, Lance) stays inside the membrane.
- Consumers pull specific OGIT schemas or DOLCE categories on demand.

### Phase 4 – Advanced Capabilities
- Ontology-guided CAM training / semantic loss.
- Hybrid vector + OWL reasoning paths.
- Full Ractor supervision tree across the membrane.

## 3. What Should "Click" into lance-graph-callcenter from Spear

From spear’s perspective, `lance-graph-callcenter` should expose:

- `ExternalMembrane` trait (already exists)
- Enriched `DnPath` / semantic navigation
- OGIT schema label lookup + OWL/DOLCE classification helpers
- Ractor-friendly async dispatch (supervisable)
- Clean `Intent` types that carry semantic context

Spear should **not** need deep knowledge of CAM-PQ internals or Arrow layouts.

## 4. Immediate Next Technical Steps (Starting Now)

1. Extend `dn_path.rs` with optional semantic annotations (non-breaking).
2. Create `ogit_owl_dolce.rs` module inside `lance-graph-callcenter`.
3. Define mapping between CAM archetypes / codebook labels and OGIT/OWL/DOLCE.
4. Update `lib.rs` re-exports and documentation.

## 5. Design Principles

- **Contracts first**: Ontology and `lance_graph_contract` must be stable before heavy consumer work.
- **Membrane as Rosetta Stone**: `lance-graph-callcenter` translates between semantic world and high-performance substrate.
- **Ractor as orchestrator**: Use Ractor for supervision and dispatch inside the membrane.
- **Need-to-have in consumers**: Spear verticals only import what they actually use.
- **Performance preserved**: All extensions must keep the zero-copy + O(1) characteristics of DnPath + CAM-PQ.

