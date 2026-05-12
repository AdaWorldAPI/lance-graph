# Cross-source pattern recognition matrix

> **READ FIRST** alongside `.claude/knowledge/tier-0-pattern-recognition.md`.
>
> This workspace has 4 parallel architectural taxonomies that don't currently cross-reference each other. Future sessions reading one and missing the parallels is the recurring failure mode behind the "Designing What's Already Built" anti-pattern. This doc is the canonical mapping.

## The 4 taxonomies

1. **A-O Patterns** (sprint-2 PR #358 + PR #359 + sprint-3 PR #360) — `.claude/plans/unified-ogit-architecture-v1.md`, `.claude/patterns.md`, `.claude/knowledge/tier-0-pattern-recognition.md`
2. **Pillars 0-4** (palantir-parity-cascade) — `.claude/plans/palantir-parity-cascade-v2.md`, `.claude/plans/ogit-cascade-supabase-callcenter-v1.md`
3. **`.grok/` epiphanies + open threads** — `.grok/01_overview/` through `.grok/05_query_languages/`, `FANOUT_MAPPING_PLAN.md`, `GLUE_LAYER_OGIT_TO_OWL_SPEC.md`, etc.
4. **Shipped substrate** (~28 crates) — what actually exists in `crates/`, anchored to TD-X rows in `.claude/board/TECH_DEBT.md`

## Master mapping table

| A-O | Pillar | .grok/ doc | Shipped substrate (file:symbol) | TD ref |
|---|---|---|---|---|
| **A** SPO-G u32 slot | Pillar 0 (OGIT u32 namespace) | `04_mathematical_foundation/` | `lance-graph-ontology/src/namespace.rs::SchemaPtr.ontology_context_id` (PARTIAL) | TD-OGIT-G-SLOT-1 |
| **B** Context Bundle | Pillar 0 (OWL overlay) | `GLUE_LAYER_OGIT_TO_OWL_SPEC.md` | (typed surface missing) | TD-CONTEXT-BUNDLE-2 |
| **C** Generic Bridge | Pillar 0 (BridgeFromRegistry) | `FANOUT_MAPPING_PLAN.md` § bridge dispatch | `lance-graph-ontology/src/bridges/mod.rs::BridgeFromRegistry` + 3 impls (PARTIAL) | TD-GENERIC-BRIDGE-3 |
| **D** Meta-Structure Hydration | Pillar 0 (TTL → bundle) | `04_mathematical_foundation/` § hydration | `lance-graph-ontology/src/ttl_parse.rs::parse_ttl_directory_with_provenance` (PARTIAL) | (anatomy demo PR-ANATOMY-1) |
| **E** Compile-Time Consumer Binding | Pillar 1 (manifest format) | `MCP/` | (manifest format missing) | TD-MANIFEST-MODULES-4 |
| **F** ractor Supervisor | Pillar 1 (supervised mesh) | `INVESTIGATION_AGENT_CODE_SKETCH.rs` | `cognitive-shader-driver/src/grpc.rs` (shape proven, supervisor missing) | TD-RACTOR-SUPERVISOR-5 |
| **G** Best-Practice Thinking Inheritance | Pillar 2 (per-G styles) | `03_cognitive_layers/` | `p64-bridge::STYLES` (12-base codebook) | (no TD; Pattern G deferred) |
| **H** Switchable Cognitive Vessel | Pillar 2 (cognitive-shader-driver) | `01_overview/` | `p64-bridge/src/lib.rs::cognitive_shader::CognitiveShader` ✅ SHIPPED | (no TD) |
| **I** Implicit Cognition | Pillar 2 (CycleAccumulator) | (parallel to FANOUT) | `lance-graph-contract/src/cycle_accumulator.rs` ✅ SHIPPED | (no TD) |
| **J** INT4-32D Atoms | Pillar 3 (proximity reranker) | `03_cognitive_layers/` § proximity | `thinking-engine/src/reranker_lens.rs` (lens shape only) | TD-INT4-32D-ATOMS-6 |
| **K** Circular Compilation | Pillar 3 (JIT actor) | `02_core_primitives/` § JITson | `lance-graph/src/cam_pq/jitson_kernel.rs` (kernel scope) | TD-CIRCULAR-COMPILATION-7 |
| **L** SPO-Chain Narrative | Pillar 4 (NARS chains) | `05_query_languages/` | `lance-graph-planner::nars::truth::TruthValue` + AriGraph (PARTIAL) | (no TD; deferred) |
| **M** Wave-Particle Bimodal | Pillar 2 (qualia + SPO) | `03_cognitive_layers/` § wave-particle | bgz17 + qualia (wave) + AriGraph + SPO (particle) ◐ PRIMITIVES SHIPPED | (no TD) |
| **N** Fingerprint-as-Codebook | Pillar 0 (substrate operation) | `02_core_primitives/` § codebook | `thinking-engine::prime_fingerprint`, `qualia::FAMILY_CENTROIDS`, `p64-bridge::STYLES` ✅ SHIPPED | (no TD) |
| **O** Phenomenological Memory | Pillar 2 (Qualia17D) | `03_cognitive_layers/` § qualia | `thinking-engine/src/qualia.rs` (39 KB) ✅ SHIPPED | (no TD) |

## Shipped consumer scaffolds (Pattern C precedents)

The Pattern C "consumer template" worked example doesn't need to be hypothetical (hubspo-rs). Two real precedents already validate the LOC reduction claim:

| Consumer | Repo / PR | Scaffold size | Status |
|---|---|---|---|
| **woa-rs** | `AdaWorldAPI/woa-rs` PR #2 | ~150 LOC | merged |
| **medcare-rs medcare-bridge** | `AdaWorldAPI/medcare-rs` PR #110 | ~250 LOC | merged |
| **smb-office-rs SmbMembraneGate** | `AdaWorldAPI/smb-office-rs` PR #29 | ~30 LOC newtype | merged |
| **medcare-rs MedCareMembraneGate** | `AdaWorldAPI/medcare-rs` PR #98 | ~30 LOC newtype | merged |
| (hypothetical) **hubspo-rs** | (W8 spec proposed scaffolding from scratch) | <150 LOC target | PROPOSED, NOT NECESSARY |

The architectural validation gate originally scoped to "build hubspo-rs in <150 LOC" should re-target to "verify woa-rs (~150 LOC) + medcare-bridge (~250 LOC) hit the LOC budget claim" — those are merged precedents.

## Maintenance protocol

When a new architectural insight surfaces in any one of the 4 taxonomies:
1. **Locate it in this matrix** — find the closest existing row
2. **Update all 4 columns** — A-O letter, Pillar number, .grok/ section, shipped-substrate file
3. **Cite the substrate file** with `file:symbol` precision (so future sessions can `grep` to verify before designing)
4. **If the insight has no existing row** — add a new row only after confirming it's not a re-discovery (use the substrate-grep checklist in `.claude/knowledge/cca2a-sprint-prompt-template.md`)

## Cross-references

- `.claude/knowledge/tier-0-pattern-recognition.md` — canonical A-O letter assignments + status
- `.claude/plans/palantir-parity-cascade-v2.md` — Pillars 0-4 architecture
- `.claude/plans/ogit-cascade-supabase-callcenter-v1.md` — Pillars 0-4 cascade execution
- `.grok/01_overview/` through `.grok/05_query_languages/` — parallel-session epiphanies
- `.claude/board/TECH_DEBT.md` — TD-X execution backlog
- `.claude/board/ARCHITECTURE_ENTROPY_LEDGER.md` — per-component scoring + clusters
- `.claude/knowledge/cca2a-sprint-prompt-template.md` — substrate-grep checklist (next deliverable)

## Provenance

This doc was created post-#360 (sprint-3 specs PR) review pass. Reviewer flagged:
> "Four parallel architectural taxonomies now exist in this workspace... None of them cross-reference the other three. Future sessions read one and miss the parallels."

This matrix is that cross-reference.
