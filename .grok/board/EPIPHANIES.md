# EPIPHANIES.md — High-Signal Insights (Pruned & Updated Mirror)

**Date**: 2026-05-08  
**Approach**: Pruned from original `.claude/board/EPIPHANIES.md`. Kept only structural, high-signal realizations. Added new ones from current exploration. Removed repetitive or outdated entries.

---

## Core Architectural Epiphanies (Current)

### 1. CausalEdge64 is the Universal Atomic Register
One `u64` can carry full causal semantics (Pearl 2³), epistemic state (NARS frequency/confidence), plasticity control, and temporal ordering. This is not "just an edge" — it is the fundamental unit that makes the entire hot path feasible at register speed.

### 2. Nested NARS Loops (Atomic + Meta)
- Inner loop: `CausalEdge64::forward()` + `learn()` (register-level inference & revision)
- Outer loop: `MetaOrchestrator` + `StyleTopology` (learning which thinking styles work)
This double nesting is rare and powerful.

### 3. Pearl 2³ Masks as First-Class Representational Dimensions
Treating the 8 `CausalMask` states not just as filters but as parallel dimensions enables true multi-perspective reasoning via superposition. This is one of the cleanest encodings of Pearl’s ladder seen in a production system.

### 4. cognitive-shader-driver + BindSpace = The Real SoA Driver
The shader **is** the driver. `BindSpace` with its column-oriented design, cheap `MetaColumn` prefilter, and native `CausalEdge64` storage is the practical implementation of the thought-cycle bus shown in the diagrams.

### 5. cycle_fingerprint as the Unit of Thought
Every cognitive cycle emits one `Fingerprint<256>` (now Vsa16kF32). This single value serves as cache key, retrieval key, replay key, and next-cycle seed. This is a major convergence point in the architecture.

### 6. Hot Path vs Cold Path is the Fundamental Split
- Hot path (`cognitive-shader-driver` + `BindSpace` + `CausalEdge64`) is optimized for continuous resonance + causal semantics.
- Cold path (DataFusion planner) is the inherited analytical engine.
The system needs a clear strategy: hot path as default for cognition, cold path for heavy analytics.

### 7. JC Crate Provides Mathematical Guardrails
Having executable proofs (Pearl 2³ accuracy, Jirak weak-dependence, Hadamard concentration, etc.) running in CI is extremely powerful. It gives us license to design aggressive compact representations (like the proposed 3-byte polyglot tag) with confidence.

### 8. Technical Debt is Mostly Fragmentation + Incomplete Hot Path
The biggest current debt is the existence of 4–6 Cypher implementations and the fact that the hot-path version is still a stub. The foundation (`BindSpace`, `CausalEdge64`, `OrchestrationBridge`) is already excellent — the debt is prioritization and completion.

---

## Open High-Potential Research Threads

- 8-mask superposition over `CausalEdge64` (L1/L2 tensor operations + L4 4096 projection)
- Compact 3-byte polyglot query language tag ("OGIT of query languages")
- Mapping parsed query intent → `CausalEdge64` fields
- Promotion Membrane implementation using `CausalEdge64` + plasticity + confidence
- How Thinking Styles should modulate `CausalMask` / `InferenceType` / plasticity
- **Unifying the SoA DTO surface evolutionarily** (declare `BindSpace` + `MetaWord` + `cycle_fingerprint` as the canonical core with thin adapters only). This is currently the highest-leverage move to reduce fragmentation debt without introducing new abstractions or breaking changes.

---

*This is a living, pruned document. New high-signal epiphanies should be added here rather than scattered across multiple files.*