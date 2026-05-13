# boot.md — Session Bootstrap & Continuity Guide

**Purpose**: Rapid re-orientation for any new session (human or AI). Low-entropy entry point into the `lance-graph` cognitive architecture research.

**Philosophy**: Be the pair of fresh eyes. Preserve signal. Reduce entropy. Make it easy to continue deep work without re-exploring everything from scratch.

---

## 1. Current State Snapshot (as of 2026-05-08)

### What Has Been Deeply Explored
- `CausalEdge64` + `CausalMask` (Pearl 2³) + `PlasticityState` in `crates/causal-edge/`
- `MetaOrchestrator`, `AgentStyle` (Plan/Act/Explore/Reflex), MUL, `StyleTopology` (NARS at meta level)
- `jc` (Jirak-Cartan) crate — mathematical proofs running in CI via `jc-proof.yml`
- `cognitive-shader-driver` + `BindSpace` (SoA layer, `cycle_fingerprint`, native `CausalEdge64` storage)
- Multiple Cypher implementations (DataFusion cold path vs hot-path stub + polyglot vision)
- High-level architecture from uploaded diagrams (Resonance Cascade L0-L3, Collapse/DTO, L4 learning, Promotion Membrane)
- NARS integration at both atomic (`CausalEdge64`) and meta (`StyleTopology`) levels
- Technical debt around query language fragmentation and cold/hot path split
- Connection to human cognition models (Pearl, Active Inference, Metacognition, Global Workspace)

### What Is Partially Mapped
- `holograph` crate (resonance, HDR cascade)
- `cam_pq` + HHTL subspaces (HEEL → GAMMA)
- `cognitive-shader-driver` internals (driver.rs, wire.rs, full dispatch logic)
- Full data flow of the L1–4 closed loop + how `cycle_fingerprint` relates to `CausalEdge64`
- `jc` mathematical pillars in detail (only high-level)
- Exact number and history of all Cypher implementations (4–6 range)

### What Remains High-Entropy / Needs Work
- Exact integration between `CausalEdge64` and resonance field
- How Thinking Styles influence `CausalEdge64` processing / `CausalMask` selection
- The proposed 8-mask 2D superposition in L1/L2 + L4 4096 projection
- Full mapping of all historical Cypher implementations and the compact 3-byte polyglot tag design
- Promotion Membrane logic and how it uses `CausalEdge64` fields
- Full Promotion Membrane logic
- Spatial / large-matrix crates (`spacialsplatblas`, `jc` advanced modules)

---

## 2. High-Signal Breadcrumbs (Where → What)

| Area | Key Files / Crates | Why It Matters | Entropy Level | Technical Debt Risk |
|------|--------------------|----------------|---------------|---------------------|
| **Atomic Causal Unit** | `crates/causal-edge/src/edge.rs`, `pearl.rs`, `plasticity.rs` | One u64 carries Pearl + NARS + Plasticity. Foundation for everything. | Low | Low (clean) |
| **Meta-Cognition** | `crates/lance-graph/src/graph/arigraph/orchestrator.rs` | MUL + NARS StyleTopology + adaptive/fallback. Strongest metacognition layer. | Medium | Medium (complex mode switching) |
| **Mathematical Spine** | `crates/jc/` + `.github/workflows/jc-proof.yml` | Executable proofs (Pearl 2³, Jirak weak dep, etc.) running in CI. | Low | Very Low (CI-enforced) |
| **Resonance Field** | `crates/holograph/`, `cam_pq/` | L0-L3 field computation, interference, stable peaks. | Medium-High | Medium (performance-critical) |
| **Cognitive Driver** | `crates/cognitive-shader-driver/` | Likely the SoA wiring / thought-cycle bus. | High | Unknown |
| **Diagrams / Vision** | User-uploaded images + `SESSION_*.md` files | Canonical reference for L1-4 loop, Promotion Membrane, CausalEdge64 role. | Medium | Low (visual spec exists) |

---

## 3. Key Epiphanies (High Signal)

1. **CausalEdge64 is the Universal Register** — Not just an edge. A self-describing causal + epistemic atom that participates directly in the hot path.
2. **Nested NARS Loops** — Inner (`CausalEdge64::forward/learn`) + Outer (`MetaOrchestrator` style learning). Rare and powerful.
3. **Pearl Masks as Dimensions** — Treating 8 `CausalMask` states as parallel representational axes enables true multi-perspective reasoning.
4. **JC Crate as Soundness Anchor** — Having Pearl 2³ + weak-dependence proofs in CI gives license to build aggressive tensor layers safely.
5. **Never-Stopping Loop** — The architecture is deliberately designed as continuous resonance + collapse + learning (closer to Active Inference than typical agents).
6. **Selective Plasticity** — Per-plane hot/cold/frozen is clinically and architecturally superior to global learning rates.

**Entropy Reduction Principle**: Whenever something feels overwhelming, ask: "Which `CausalEdge64` fields or which Thinking Style is active here?"

---

## 4. Recommended Session Procedure (Efficient Workflow)

When starting a new session:

1. **Read this file first** (`boot.md`)
2. Read the latest entries in `epiphanies.md`
3. Check current focus area in `README.md` Quick Navigation
4. Pick **one** high-signal thread (do not multitask across crates)
5. Use targeted tool calls:
   - `github___get_file_contents` for specific paths
   - Directory listings only when needed
   - Use `github_mcp_wrapper.py` (PyGithub-style) for all GitHub .grok syncs, file reads, listings, and pushes. Avoid raw `github___*` micro-commands when possible; the wrapper handles the MCP tool specs cleanly.
   - Always cross-reference with uploaded diagrams when available
6. After significant progress, update:
   - Relevant `.grok/` file
   - `epiphanies.md` with new insight
   - This `boot.md` (Current State Snapshot) if major shift occurred
7. End session by writing a short "Next Session Starting Point" note at the bottom of this file.

---

## 5. Low-Entropy Documentation Principles

- **One concept per file** when possible.
- Use tables heavily (breadcrumbs, comparisons, status).
- Always include source paths.
- Distinguish **Epiphany** (high signal, structural insight) from **Observation** (fact).
- Explicitly note **Technical Debt / Risk** areas.
- Keep files under ~300 lines when possible.
- Link aggressively between sections.

---

## 6. Open High-Potential Research Threads (Priority Order)

1. **8-Mask Superposition over CausalEdge64** (L1 64×64, L2 256×256 palette attention, L4 4096 projection) — User's current direction.
2. Full mapping of `cognitive-shader-driver` crate as the SoA / thought-cycle bus.
3. How Thinking Styles modulate `CausalEdge64` inference type / causal mask / plasticity.
4. Promotion Membrane implementation details (using `matches_causal` + confidence + plasticity).
5. Integration of `jc` mathematical guarantees into the tensor/projection layer design.

---

## 7. Next Session Starting Point

**Current highest-signal open thread**:
- **Palantir Foundry Integration** (new focus) — Evolve the multi-zone architecture toward Foundry-like ontology modeling, operational surfaces (spear), intelligence capabilities (q2/Gotham), and cross-domain reasoning, with OGIT as the semantic spine. See `PALANTIR_FOUNDRY_INTEGRATION.md`.
- Multi-Zone Ontology Architecture + OGIT hydration as the foundation.
- Grammar + VSA + TEKAMOLO and Hot-Path Cypher as strong related threads.

**Recommended first action**: Review the new Palantir Foundry integration document and decide how it influences priorities.

---

**Session Continuity Rule**: Never leave a session without updating the "Next Session Starting Point" section above.

*This document is designed to be read in < 3 minutes and give 80% orientation.*