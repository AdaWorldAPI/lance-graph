# .grok/ — Architecture Research & Documentation

> **Purpose**: A living, low-entropy knowledge base for the Resonance-Based Cognitive System in `lance-graph`.
> This folder exists to make the overwhelmingly complex codebase navigable and to capture high-signal architectural insights.

---

## Repository at a Glance

This is a **hybrid vector-symbolic cognitive architecture** that combines:

- **Resonance-based field computation** (holograph + cam_pq + HDR cascade)
- **Compact causal registers** (`CausalEdge64`)
- **Explicit Pearl causal hierarchy** (2³ masks)
- **NARS-style reasoning** embedded at atomic + meta levels
- **Self-regulating meta-orchestration** (Thinking Styles + MUL)
- **Mathematical verification** via the `jc` (Jirak-Cartan) crate running in CI

The system is designed for **continuous thinking** (L1–4 closed loop) with strong guarantees on dependence, concentration, and causal semantics.

---

## Documentation Skeleton

This is intentionally modular. Start here and expand as needed.

### Core Sections

| Section | File | Status | Description |
|---------|------|--------|-------------|
| **Overview** | `01_overview/01_system_overview.md` | Active | High-level system map + key invariants (multi-zone, L1–L4 shaders, spear bridge, OGIT) |
| **Core Primitives** | `02_core_primitives/` | Active | `CausalEdge64`, Pearl masks, Plasticity |
| **Cognitive Layers** | `03_cognitive_layers/` | Active | Resonance, MetaOrchestrator, Thinking Styles |
| **Mathematical Foundation** | `04_mathematical_foundation/` | Active | `jc` crate, proofs, CI verification |
| **Query Languages & Cypher** | `05_query_languages/cypher_implementations.md` | Active | Multiple implementations, cold vs hot path, technical debt |
| **Integration Architecture** | `05_integration_architecture.md` | Planned | How layers compose (SoA bus, L1–L4 loop, Promotion Membrane) |
| **CI & Verification** | `06_ci_and_verification.md` | Planned | `jc-proof.yml`, mathematical guarantees in CI |
| **Epiphanies & Research Notes** | `epiphanies.md` | Living | High-signal insights, potential vs drift |

### Quick Navigation (Current Focus Areas)

- **High-Level System Map & Multi-Zone Overview** → `01_overview/01_system_overview.md`
- **CausalEdge64** → `02_core_primitives/causal_edge64.md`
- **Pearl 2³ Masks** → `02_core_primitives/pearl_masks.md`
- **Thinking Styles + MetaOrchestrator** → `03_cognitive_layers/meta_orchestrator.md`
- **NARS + Thinking Full Inventory & Migration** (single source — paths, uniqueness, DTO/SoA, before/after, inner/outer ontology + OGIT) → `03_cognitive_layers/NARS_THINKING_IMPLEMENTATIONS_INVENTORY_MIGRATION.md`
- **cognitive-shader-driver + Cypher** → `03_cognitive_layers/cognitive_shader_driver.md` + `05_query_languages/cypher_implementations.md`
- **Jirak-Cartan Mathematical Proofs** → `04_mathematical_foundation/jc_jirak_cartan.md`
- **CI Proof Pipeline** → `06_ci_and_verification.md`
- **Grok Session Tooling & GitHub Sync** (meta) → `github_mcp_wrapper.py` (use this PyGithub-like wrapper for all .grok GitHub operations; bulk push, ls, cat, etc. via MCP connected tools)

---

## How to Use This Documentation

1. Start with the section most relevant to your current question.
2. Each file should contain:
   - Key structs / enums with links to source
   - Core invariants / guarantees
   - Open questions / research directions
   - Connections to other sections
3. Keep entropy low — prefer tables, short paragraphs, and explicit links.

---

**Status**: This is a living skeleton. It will grow as we explore.

Last updated: 2026-05-08

---

*Feel free to add new files or sections as new high-signal patterns emerge.*