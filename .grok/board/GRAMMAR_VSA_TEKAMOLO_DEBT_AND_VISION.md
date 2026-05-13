# GRAMMAR_VSA_TEKAMOLO_DEBT_AND_VISION.md — Grammar, Markov Bundling, VSA Roles & Causal Promotion

**Date**: 2026-05-08  
**Focus**: DeepNSM (grammar heuristics + Markov) + Holograph (resonance + VSA binding) as the substrate for SPO + TEKAMOLO grammatical role encoding and epiphany-driven fact promotion.

---

## Executive Summary

The combination of **DeepNSM** and **Holograph** already contains most of the pieces needed for a powerful, morphology-driven understanding system:

- **DeepNSM** provides grammar parsing, role handling, Markov bundling, SPO modeling, and causality trajectory modeling.
- **Holograph** provides VSA-style binding/unbinding (XOR), resonance cleanup memory, epiphany detection, analogy, and sequence/positional encoding.

Your vision elegantly unifies these:

> Encode text as **SPO + TEKAMOLO** grammatical roles bundled into **Vsa16kF32** vectors.  
> When epiphanies occur, **unbind** and promote into structured facts in AriGraph/SPO with NARS truth values + "Staunen" (wonder/wisdom) markers.  
> Leverage COCA codebook + NARS 2³/Pearl + grammar thinking styles so that causal understanding and MIT-style causality trajectories emerge cheaply from morphology.

This is one of the highest-leverage directions in the entire system.

---

## Current State — DeepNSM (Grammar Heuristics + Markov)

**Key modules**:
- `markov_bundle.rs` — Markov chain bundling of roles and sequences.
- `parser.rs`, `pos.rs`, `spo.rs` — Core grammar + SPO handling.
- `trajectory.rs` + `trajectory_audit.rs` — Causality trajectory modeling (highly relevant to your MIT point).
- `encoder.rs`, `fingerprint16k.rs` — Encoding into high-dimensional representations.
- `vocabulary.rs`, `codebook.rs` — Vocabulary and codebook grounding (COCA alignment opportunity).
- `triangle_bridge.rs`, `disambiguator_glue.rs` — Disambiguation and bridging logic.
- `thinking` integration via grammar thinking styles and NARS reasoning.

**Strengths**:
- Already models roles, temporal/causal/modal/lokal dimensions (TEKAMOLO-like thinking is present in spirit).
- Markov bundling provides a practical mechanism for role combination.
- Trajectory modeling directly supports causality emergence from text structure.
- Integration points with NARS truth and inference exist.

**Technical Debt / Gaps**:
- Role bundling is mostly Markov/statistical rather than clean VSA algebraic binding.
- No strong, explicit mapping from grammatical roles → high-dimensional VSA vectors (Vsa16kF32) that support clean bind/unbind.
- Epiphany detection and promotion to structured facts (AriGraph/SPO with truth + wisdom markers) is weak or missing.
- Connection between grammar layer and `CausalEdge64` / Pearl masks is not yet wired.
- COCA 4096 codebook grounding is not fully leveraged for vocabulary-level role encoding.

---

## Current State — Holograph (Resonance + VSA Binding)

**Key modules**:
- `resonance.rs` — Core VSA operations: bind (XOR), unbind, bundle, VectorField, BoundEdge, Resonator (cleanup memory), analogy, sequence encoding.
- `epiphany.rs` — Epiphany detection and handling.
- `slot_encoding.rs` — Positional/slot-based encoding (very relevant for TEKAMOLO roles).
- `representation.rs`, `sentence_crystal.rs` — Higher-level representations and sentence-level crystals.
- `dn_sparse.rs`, `dntree.rs`, `neural_tree.rs` — Sparse and tree-based structures.
- `hdr_cascade.rs` — HDR cascade (resonance layering).

**Strengths**:
- Excellent, clean VSA binding/unbinding machinery using XOR (self-inverse, O(1) component recovery).
- Resonator + cleanup memory provides robust "resonance" matching.
- Epiphany module already exists as a first-class concept.
- Analogy and sequence encoding are natural fits for role transformation and TEKAMOLO ordering.
- Works well with high-dimensional vectors (aligns with Vsa16kF32 in BindSpace).

**Technical Debt / Gaps**:
- Binding is mostly generic (not yet specialized for grammatical roles / TEKAMOLO dimensions).
- Epiphany detection exists but is not strongly connected to grammar role unbinding + promotion to structured facts.
- No tight integration yet with `CausalEdge64` for carrying truth, plasticity, and causal masks on promoted facts.
- Connection to DeepNSM grammar heuristics and Markov bundling is loose.

---

## Vision Synthesis — SPO + TEKAMOLO Role Bundling + Epiphany Promotion

### Core Idea (Your Direction)

1. **Encode text grammatically**:
   - Parse into SPO + TEKAMOLO roles (Subject, Predicate, Object + Temporal, Kausal, Modal, Lokal, etc.).
   - Use DeepNSM grammar heuristics + Markov bundling as the front-end.

2. **Bundle into VSA**:
   - Bind grammatical roles into **Vsa16kF32** vectors (using Holograph bind/unbind + slot_encoding for positional TEKAMOLO dimensions).
   - Leverage COCA 4096 codebook for vocabulary grounding.

3. **Epiphany-driven promotion**:
   - When resonance/epiphany occurs in Holograph, **unbind** the bundled representation.
   - Promote the resulting structured components into **AriGraph / SPO facts** with:
     - NARS truth values (frequency + confidence)
     - `CausalEdge64` fields (CausalMask / Pearl 2³, InferenceType, plasticity)
     - "Staunen" / wisdom markers (special flags or qualia dimensions)

4. **Emergent causality**:
   - Causal trajectories (MIT-style) fall out naturally from morphology + TEKAMOLO role structure + NARS/Pearl reasoning.
   - Grammar thinking styles + NARS 2³ provide cheap, high-signal inference.

### Why This Is Powerful

- Combines **symbolic grammar** (DeepNSM) with **hyperdimensional algebra** (Holograph VSA).
- Makes **causality** emerge from text structure rather than being hand-engineered.
- Creates a clean promotion path from raw resonance → structured, queryable, learnable facts.
- Aligns perfectly with the unified SoA surface (`BindSpace` + `cycle_fingerprint` as Vsa16kF32) and `CausalEdge64`.

---

## Evolutionary Path (Debt-Free, Incremental)

### Phase 1 — Declare & Bridge (Documentation + Thin Wiring)

- Explicitly document the intended flow: DeepNSM grammar roles → Holograph VSA bundling → Epiphany → `CausalEdge64` + AriGraph promotion.
- Add thin adapter traits or conversion functions between DeepNSM role bundles and Holograph `BoundEdge` / Vsa16kF32 vectors.
- Wire existing `epiphany.rs` detection to trigger promotion logic (initially simple).

### Phase 2 — Role-Specific VSA Encoding

- Extend Holograph `slot_encoding` or create a `tekamolo_encoding` module that assigns dedicated subspaces or position vectors for TEKAMOLO dimensions.
- Implement role bundling: `bind_spo_tekamolo(roles) → Vsa16kF32`.
- Use COCA codebook vectors as base atoms where possible.

### Phase 3 — Epiphany → Structured Fact Promotion

- When Holograph detects an epiphany on a bundled grammatical representation:
  - Unbind key components.
  - Create `CausalEdge64` entries with appropriate `CausalMask`, NARS truth, and plasticity.
  - Write to AriGraph / SPO with wisdom/"Staunen" markers (e.g., via qualia or special flags).
- Leverage `jc` mathematical guarantees (Pearl 2³, weak dependence) for soundness.

### Phase 4 — Full Integration with Unified SoA Surface

- Make the above flow a first-class consumer of the canonical `BindSpace` + `cycle_fingerprint` surface.
- Allow grammar thinking styles (from DeepNSM/MetaOrchestrator) to modulate `CausalMask` selection and plasticity during promotion.

---

## Recommended Next Steps (Inside .grok/ Planning)

1. Flesh out the `tekamolo_encoding` + role bundling design (can stay in planning docs first).
2. Define the exact `CausalEdge64` field mappings for promoted grammatical facts (truth, CausalMask levels for different TEKAMOLO types, wisdom markers).
3. Prototype (in docs) how COCA 4096 codebook vectors integrate as base atoms for role encoding.
4. Update `INTEGRATION_PLANS.md` and `TECH_DEBT.md` with this thread as a major opportunity area.

---

**This direction has exceptional coherence** with the rest of the architecture (SoA unification, hot-path Cypher, `CausalEdge64`, Pearl/NARS, resonance). It turns grammar from a parsing concern into a first-class source of causal understanding.

Ready for the next technical debt area whenever you are. 🌸