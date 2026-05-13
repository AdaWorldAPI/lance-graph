# Lance-Graph Backend Package (Ontology + Cognitive Fabric)

**Generated**: 2026-05-11  
**Context**: Analysis of AdaWorldAPI/lance-graph repo, focusing on `lance-graph-ontology` and `lance-graph-cognitive/src/fabric`

## What is included in this package

This zip contains:

1. **firefly_frame.rs** - The actual 16384-bit Firefly Frame microinstruction format from the repo (the "cognitive CPU" frame spec that supports NARS, Cypher, Lance, Causal, Quantum, Memory, Control languages).

2. **Current backend status** (from repo exploration):
   - `lance-graph-ontology` crate has **OGIT-canonical ontology spine** scaffolding started.
   - No DOLCE or OWL support yet.
   - Bridges for external namespaces (SharePoint, Email).
   - Bilingual DTO surface mentioned in commits.

3. **Proposed extensions** (new files in `proposed/`):
   - `ogit_dolce_spine.rs` — Rust foundation for OGIT spine aligned with DOLCE categories (highly recommended for NARS/cognitive compatibility).
   - `owl_support.rs` — Basic OWL import/export sketch.
   - `fabric_ontology_bridge.rs` — Glue to connect the ontology spine to the cognitive fabric / Firefly Frame / shader driver.

## Current State Summary (Double-check)

| Component              | Status in `lance-graph-ontology`          | Recommendation |
|------------------------|-------------------------------------------|----------------|
| OGIT spine             | Scaffolding exists (commit: "scaffold the OGIT-canonical ontology spine") | Keep as project canonical layer |
| DOLCE                  | **Absent**                                | Add alignment (especially Endurant/Perdurant/Quality for cognitive grounding) |
| OWL                    | **Absent**                                | Add serialization layer for interoperability |
| Connection to cognitive fabric (`firefly_frame`, shader-driver) | Not connected yet | Use the proposed bridge |

**OGIT** here appears to be used as the internal canonical ontology (possibly "Ontology for General Intelligence Tasks" or geospatial variant — the spine is the core).

## Why DOLCE + OGIT + OWL matters for your architecture

- **DOLCE** provides excellent foundational categories for cognitive/NARS-style reasoning (perdurants for events/processes that map to NARS temporal inference and Firefly "CONTEXT" qualia/truth values).
- **OGIT spine** = your project-specific canonical model.
- **OWL** = standard way to import/export ontologies and reason externally.
- **Firefly Frame** already has language prefix for **NARS (0x3)** and **Causal (0x4)** — perfect place to ground in a DOLCE-aligned ontology.

## Python Scripts Rule (Important)

**All Python scripts in this project are to be treated as an internal convenience layer** around the already existing **MCP tool calls** (core Rust engine, ontology spine, cognitive fabric / Firefly Frame encoder, graph query engines, etc.).

- Never duplicate core logic in Python.
- Python only adds ergonomics, high-level orchestration, validation, and developer-friendly APIs.
- The real work (OGIT+DOLCE spine, Firefly Frame encoding, NARS inference, Cypher/Lance execution) always happens in Rust via PyO3.

The new `python/convenience/lance_graph_convenience.py` demonstrates this pattern.

## How to use this package

1. Copy `firefly_frame.rs` into your `crates/lance-graph-cognitive/src/fabric/`
2. Create `crates/lance-graph-ontology/src/ontology/` and add the proposed `ogit_dolce_spine.rs`
3. Implement the bridge so Firefly Frames can carry ontology-typed payloads.
4. Use `python/convenience/lance_graph_convenience.py` as the recommended Python entry point (it already follows the "convenience layer around MCP" rule).

## Next steps I can help with

- Full implementation of the OGIT+DOLCE spine (Rust)
- Mapping specific DOLCE classes to NARS terms
- Integration code between ontology and the 16384-bit Firefly Frame
- OWL <-> Rust struct bidirectional mapping
- More convenience methods in the Python layer (following the internal convenience rule)

Just say the word and I'll iterate on any of the proposed files.

---
*This package was created because full `git clone` is not possible in the current execution environment. All original code is fetched from the public GitHub repo.*
