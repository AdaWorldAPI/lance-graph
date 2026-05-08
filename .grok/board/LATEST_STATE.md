# LATEST_STATE — Current High-Signal Snapshot (Grok Mirror)

**Purpose**: Rapid orientation for any session. Pruned and updated from `.claude/board/` version. Focus on current architecture, not historical PR noise.

**Last Major Update**: 2026-05-08 (Grok synthesis pass — incorporated CausalEdge64, cognitive-shader-driver SoA, hot/cold paths, JC proofs, technical debt analysis).

---

## Core Architecture (Current Understanding)

### Primary Primitives
- **`CausalEdge64`** (`crates/causal-edge/`): The atomic causal + epistemic register. One `u64` carrying S/P/O palettes, NARS truth, `CausalMask` (Pearl 2³), `PlasticityState`, `InferenceType`, temporal index. Lives natively in `BindSpace.edges`.
- **`BindSpace`** (`cognitive-shader-driver`): High-performance Struct-of-Arrays substrate. Multiple named fingerprint planes + `EdgeColumn` (CausalEdge64) + packed `MetaColumn` (cheap prefilter) + Qualia. `cycle_fingerprint` (Vsa16kF32) is the unit of thought emitted per cognitive cycle.
- **`cognitive-shader-driver`**: The SoA wiring layer and thought-cycle bus. "The shader IS the driver." Canonical surface via `OrchestrationBridge` + `UnifiedStep`. Strong Canonical vs LAB-ONLY separation.

### Hot Path vs Cold Path
- **Hot Path** (Primary for cognition): `cognitive-shader-driver` + `BindSpace` + `CausalEdge64` + compact representation. Register-level + SoA sweeps. `cycle_fingerprint` as first-class unit.
- **Cold Path** (Analytical workloads): `datafusion_planner` in `lance-graph`. Traditional columnar joins/scans. More mature for complex queries but heavier.

### Mathematical Foundation
- **`jc` (Jirak-Cartan) crate**: Executable mathematical proofs running in CI (`jc-proof.yml`). Includes Pearl 2³ mask accuracy, Jirak weak-dependence Berry-Esseen, Hadamard concentration, etc. Provides soundness anchor for aggressive hot-path designs.

### Query Language Situation (High Technical Debt)
Multiple (4–6) Cypher implementations exist:
- DataFusion cold path (mature, heavy).
- `cognitive-shader-driver/cypher_bridge.rs` (hot-path intent, currently minimal stub — keyword classifier only). This is one of the later attempts.
- Proposed compact polyglot tag system ("OGIT of query languages" — 3-byte or small opcode for Cypher/Gremlin/GQL/SQL/NARS/etc.).

**Technical Debt**: Fragmentation is the main issue. Hot path needs real parser wiring while staying inside the canonical `OrchestrationBridge`. Missing mapping from parsed intent → `CausalEdge64` fields.

---

## Key Open High-Potential Directions

1. **Evolutionary Unification of the SoA DTO Surface** (new highest-signal thread) — Strengthen `BindSpace` + `MetaWord` + `cycle_fingerprint` as the canonical core with thin adapters only. See `.grok/board/UNIFIED_SOA_SURFACE_PLAN.md`. No flattening or deletion without superior replacement.
2. **8-Mask Superposition over CausalEdge64** (L1 64×64, L2 256×256 palette attention, L4 4096 projection).
3. Compact 3-byte polyglot query language tag ("OGIT of query languages") built on the unified SoA surface.
4. Full hot-path Cypher (and polyglot) support via the canonical `OrchestrationBridge`.
5. How Thinking Styles should modulate `CausalEdge64` fields.
6. Promotion Membrane implementation using `CausalEdge64` + plasticity + confidence.

---

## What to Read Next (Efficient Order)

1. `.grok/boot.md` — Session bootstrap + continuity procedure.
2. `.grok/board/UNIFIED_SOA_SURFACE_PLAN.md` — Highest-signal current direction (SoA DTO unification).
3. `.grok/05_query_languages/cypher_implementations.md` — Current Cypher landscape + debt.
4. `.grok/03_cognitive_layers/cognitive_shader_driver.md` — SoA layer deep dive.
5. `.grok/02_core_primitives/causal_edge64.md` — Atomic register.
6. `.grok/epiphanies.md` — High-signal insights.

---

**Session Rule**: Always update this file + `boot.md` "Next Session Starting Point" before ending deep work.

*This is a pruned, modernized mirror. Outdated historical PR noise and low-signal inventory have been summarized or removed.*