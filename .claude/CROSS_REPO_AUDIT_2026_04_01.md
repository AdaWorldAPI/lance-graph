# Cross-Repository Integrated Audit

> **Date**: 2026-04-01
> **Scope**: lance-graph, ndarray, q2, AriGraph
> **Auditor**: Claude Opus 4.6 (8 parallel agents, 67+42+20+27+32+28+25+16 = 257 tool calls)
> **Branch**: `claude/review-neuronprint-handover-KEFcb`

---

## I. EXECUTIVE SUMMARY

| Repo | PRs (last 25) | `.claude/` files | Tests passing | Activity window |
|------|--------------|-----------------|---------------|-----------------|
| **lance-graph** | 25 (24 merged, 1 open) | 63+ (29 root, 34 prompts, 4 agents) | ~300+ | Mar 13 -- Mar 31 |
| **ndarray** | 25 (all merged) | 51 (7 root, 25 prompts, 9 agents, 8 knowledge) | ~200+ | Mar 30 -- Mar 31 (36h burst) |
| **q2** | 10 (all merged) | 30 (11 prompts, 4 rules, 2 skills, hooks) | unknown | Mar 25 -- Mar 29 |
| **AriGraph** | 0 | none | N/A | dormant (upstream reference) |

**Critical finding**: Session B (bgz17) is marked PARTIAL in integration-lead but is actually **DONE** (121 tests, all modules on main). The agent file is stale by ~10 days.

**Biggest blocker**: `bgz17` is not a dependency of `lance-graph` Cargo.toml. Until the `bgz17-codec` feature flag is wired, Sessions C and D cannot proceed.

**Biggest gap**: `LanguageBackend` trait has **zero implementations**. Neither server generates natural language. The entire NL generation layer is missing.

---

## II. ITEM-BY-ITEM CATEGORIZATION

### Legend

| Category | Meaning |
|----------|---------|
| **DONE** | Merged, tested, on main. No action needed. |
| **OUTDATED** | Prompt/doc describes work that was already completed but doc wasn't updated. |
| **TECH DEBT** | Implemented but has known bugs, stubs, or missing wiring. Should fix. |
| **OPEN TASK** | Planned, has clear benefit, not yet started. Should do. |
| **DEPRECATED** | Superseded by later equivalent features. Can archive. |

---

### A. Integration Sessions (v3)

| Item | Category | Evidence | Action |
|------|----------|----------|--------|
| **Session A**: blasgraph CSC + Cypher planner | **DONE** | PR #29 merged 2026-03-22, 193 tests, Phase 1 all `[x]` | None |
| **Session B**: bgz17 container annex | **DONE** | PR #28 merged 2026-03-21, container.rs shipped | None |
| **Session B**: palette semiring + SIMD (items 2-7) | **DONE** | `palette_semiring`, `palette_matrix`, `palette_csr`, `simd`, `typed_palette_graph` all on main, 121 tests | **Update integration-lead agent to mark DONE** |
| **Session C**: ndarray bgz17 dual-path | **OPEN TASK** | No PR, no branch, bgz17 not in lance-graph Cargo.toml, Phase 3 all `[ ]` | **Critical path** -- see Integrationsplan |
| **Session D**: FalkorDB retrofit | **OPEN TASK** | No PR, orphan branch exists (`claude/lance-graph-falkordb-integration-TIkjC`), Phase 4 all `[ ]` | Blocked on Session C |
| `integration-lead.md` agent file | **OUTDATED** | Still says Session B is "PARTIAL" | Update status markers |

### B. NeuronPrint / 6D SPO (Session 6d)

| Item | Category | Evidence | Action |
|------|----------|----------|--------|
| `neuron.rs` NeuronPrint/Query/Trace | **DONE** | On main, 9 tests | None |
| `hydrate.rs` TensorRole + partition columns | **DONE** | On main, 9 tests | None |
| `serve.rs` SPO extraction + NARS reasoning | **DONE** | On main, functional | None |
| Rosetta docs (NEURONPRINT_ROSETTA.md, NEURON_QUERY_LANGUAGE.md) | **DONE** | On main | None |
| `message_to_base17()` still uses byte hash | **TECH DEBT** | Handover item 1 "Must Fix" | Low priority (embedding endpoint secondary) |
| `AutocompleteCache.palette_indices` unused field | **TECH DEBT** | Handover item 2 "Must Fix" | Remove or repurpose |
| DataFusion UDFs (l1, magnitude, xor_bind, bundle, etc.) | **OPEN TASK** | Handover item 3, not started | Quick win -- pure scalar functions |
| Hydrate real model with partition columns | **OPEN TASK** | Handover item 4, not started | Needs bgz7 files on disk |
| Per-role palettes (6 palettes, one per tensor role) | **OPEN TASK** | Handover item 5, not started | Research exploration |
| NeuronPrint construction from partitioned Lance | **OPEN TASK** | Handover item 6, not started | Depends on item 4 |
| Q*K alignment per layer analysis | **OPEN TASK** | Handover item 7, research | Depends on UDFs |
| Gate magnitude distribution | **OPEN TASK** | Handover item 8, research | Depends on UDFs |
| Up/Down polysemanticity detector | **OPEN TASK** | Handover item 9, research | Depends on UDFs |
| Cross-model NeuronPrint diff | **OPEN TASK** | Handover item 10, research | Needs 2+ indexed models |
| AriGraph episodic with NeuronPrint | **OPEN TASK** | Handover item 11, research | See Integrationsplan |
| Cypher extension Phase 2 (parser) | **OPEN TASK** | Handover item 12, not started | Low priority |

### C. OSINT Pipeline

| Item | Category | Evidence | Action |
|------|----------|----------|--------|
| `lance-graph-osint` pipeline skeleton | **DONE** | URL fetch, rule-based SPO extract, NARS revision, palette export | None |
| `reader-lm` crate (Qwen2 forward pass) | **TECH DEBT** | Forward pass written, **bgz7-to-f32 bridge missing**, tokenizer stub | Critical blocker for real inference |
| `bge-m3` crate (XLM-RoBERTa forward pass) | **TECH DEBT** | Forward pass written, falls back to hash, tokenizer stub | Critical blocker for semantic embedding |
| OSINT pipeline wiring to reader-lm | **OPEN TASK** | `reader.rs` does hand-rolled HTML strip, not Reader LM | Wire after bgz7-to-f32 bridge |
| OSINT pipeline wiring to bge-m3 | **OPEN TASK** | `reader.rs` uses hash embedding, not BGE-M3 | Wire after bgz7-to-f32 bridge |
| ndarray safetensors indexing (reader-lm, bge-m3) | **DONE** | `#[ignore]` integration tests in safetensors.rs | Run to produce bgz7 files |
| ndarray serve.rs (axum API) | **DEPRECATED** | Deleted in PR #72 (same session as reader-lm addition) | Replaced by lance-graph serve.rs |

### D. API / Serve Layer

| Item | Category | Evidence | Action |
|------|----------|----------|--------|
| lance-graph serve.rs (port 3000) | **DONE** | SPO extract + NARS reasoning, 4 endpoints | Functional but outputs diagnostic strings |
| q2 cockpit-server (port 2718) | **TECH DEBT** | Correct OpenAI wire format, **all inference stubs** | Wire to ndarray ModelRouter |
| q2 cockpit React UI | **DONE** | Neural debugger, MRI, analyst, strategy diagnostics | None |
| `LanguageBackend` trait | **TECH DEBT** | Defined in `language.rs`, **ZERO implementations** | **Critical gap** -- see Integrationsplan |
| `ContextBlackboard` BFS retrieval | **DONE** | Implemented with graph loading, 10 tests | Not yet called by any server |
| `select_backend()` DK-gated routing | **DONE** | Implemented | Not yet called by anything |
| ndarray `ModelRouter` (GPT-2 + OpenChat) | **DONE** | Library ready | Not wired to any server |
| P18 Internal LLM Language Surface | **OPEN TASK** | Design prompt exists, trait defined, no implementation | High-impact -- enables NL generation |

### E. Encoding / Codec Stack

| Item | Category | Evidence | Action |
|------|----------|----------|--------|
| Base17 i16[17] codec | **DONE** | ndarray `bgz17_bridge.rs`, SIMD multi-versioned | None |
| 3x16kbit Plane accumulator | **DONE** | ndarray `plane.rs`, alpha-aware Hamming, RL support | None |
| Palette compression (256 archetypes) | **DONE** | lance-graph `bgz17/palette.rs`, distance matrix, compose table | None |
| HHTL cascade (HEEL/HIP/TWIG/LEAF) | **DONE** | ndarray `cascade.rs` + p64 `hhtl_cascade_search` | None |
| bgz-tensor weight decompression | **TECH DEBT** | Crate exists, **bgz7-to-f32 path not wired** to reader-lm/bge-m3 | Critical for model inference |
| Compose table (semiring materialization) | **DONE** | `compose[a][b] = palette_idx of xor_bind(a,b)`, 64KB | None |
| ZeckBF17 (codec-research) | **TECH DEBT** | 7 known bugs (critical: cyclic shift mismatch), research sandbox | Fix if pursuing octave encoding |

### F. p64 / Bridge / Contract

| Item | Category | Evidence | Action |
|------|----------|----------|--------|
| p64 Palette64 + SIMD kernels | **DONE** | ndarray crate, AVX-512/AVX2/scalar, attend/nearest/moe_gate | None |
| p64-bridge (CausalEdge64 -> Palette) | **DONE** | lance-graph crate, CognitiveShader cascade+deduce | None |
| lance-graph-contract | **DONE** | Zero-dep, 15 tests, ThinkingStyle/PlannerContract/etc. | None |
| Convergence (episodes -> palette layers) | **DONE** | lance-graph-planner, triplet_to_headprint, classify_relation | None |
| Contract adoption by ladybug/crewai/n8n | **OPEN TASK** | Phase 3, not started | See Integrationsplan |
| SESSION_WIRE_SYNAPSES bridges 1-3 | **TECH DEBT** | All `todo!()` stubs in ndarray | Wire Jina Palette, attention metrics, crystal encoder |

### G. AriGraph Transcode

| Item | Category | Evidence | Action |
|------|----------|----------|--------|
| TripletGraph (BFS, entity index, NARS truth) | **DONE** | lance-graph `arigraph/triplet_graph.rs` | None |
| EpisodicMemory (fingerprint retrieval) | **DONE** | lance-graph `arigraph/episodic.rs` | None |
| Sensorium (temperature, plasticity, NARS auto-heal) | **DONE** | lance-graph `arigraph/sensorium.rs` | None |
| Orchestrator (meta-reasoning) | **DONE** | lance-graph `arigraph/orchestrator.rs` | None |
| xAI client (external LLM) | **DONE** | lance-graph `arigraph/xai_client.rs` | None |
| Python AriGraph (upstream) | **DEPRECATED** | 0 PRs, no .claude/, superseded by Rust transcode | Reference only |
| AriGraph with NeuronPrint encoding | **OPEN TASK** | Handover item 11 | See Integrationsplan |

### H. q2 Application Layer

| Item | Category | Evidence | Action |
|------|----------|----------|--------|
| neural-debug (static scanner) | **DONE** | lance-graph crate, CLI `neural-scan` | None |
| neural-debug (runtime instrument + MCP) | **DONE** | q2 crate, strategy_check, cockpit UI hook | None |
| Cockpit OpenAI endpoints | **TECH DEBT** | Wire format correct, inference stubbed | Wire to ModelRouter |
| Cockpit MRI/analyst/orchestrator | **DONE** | Real NARS inference on graph data | None |
| OpenClaw integration | **OPEN TASK** | Design docs only (FINAL_STACK.md, VISION.md), zero code | See Integrationsplan |

### I. `.claude/` Root Files -- Staleness Audit

| File | Category | Reason |
|------|----------|--------|
| `FIX_BLASGRAPH_SPO.md` | **OUTDATED** | blasgraph SPO fixed in Session A (PR #29) |
| `SESSION_B_HDR_RENAME.md` | **DEPRECATED** | HDR rename done, `hdr.rs` exists as Cascade |
| `UNIFIED_HDR_RENAME_AND_CROSSPOLINATE.md` | **DEPRECATED** | Work completed, absorbed into Session B |
| `SESSION_FALKORDB_CROSSCHECK.md` | **DEPRECATED** | Replaced by `session_D_v3_falkordb_retrofit.md` |
| `FALKORDB_ANALYSIS.md` | **DEPRECATED** | Findings absorbed into Session D v3 |
| `SESSION_D_LENS_CORRECTION.md` | **DONE** | Independent track, still valid |
| `SESSION_J_PACKED_DATABASE.md` | **DONE** | PackedDatabase optimization, independent |
| `SESSION_LANCE_ECOSYSTEM_INVENTORY.md` | **DONE** | Reference material, still relevant |
| `SESSION_LANGGRAPH_ORCHESTRATION.md` | **DONE** | Separate track, still valid |
| `BELICHTUNGSMESSER.md` | **DONE** | Reference for Cascade architecture |
| `BF16_SEMIRING_EPIPHANIES.md` | **DONE** | Foundational knowledge, validated by implementation |
| `DEEP_ADJACENT_EXPLORATION.md` | **DONE** | Research reference |
| `FINAL_STACK.md` | **DONE** | Architecture vision, still valid |
| `GPU_CPU_SPLIT_ARCHITECTURE.md` | **DONE** | Reference |
| `INTEGRATION_SESSIONS.md` | **OUTDATED** | Sessions G-L target rustynum, ndarray blackboard shows core types ported |
| `INVENTORY_MAP.md` | **OUTDATED** | Pre-dates v3 session plan, many items now done |
| `OVERLOOKED_THREADS.md` | **OUTDATED** | Several threads now addressed by NeuronPrint/p64 work |
| `RESEARCH_REFERENCE.md` | **DONE** | Still valid reference |
| `RESEARCH_THREADS.md` | **OUTDATED** | Several threads now resolved |
| `SCOPE_E_FINDINGS.md` | **DONE** | Polyglot notebook findings, valid |
| `VISION_ORCHESTRATED_THINKING.md` | **DONE** | Still valid architecture vision |
| `DEEPNSM_CAM_REFERENCE.md` | **DONE** | Reference for CAM pipeline |
| `SESSION_2026_03_25_CROSS_REFERENCE.md` | **DONE** | Cross-reference, still valid |
| `blackboard.md` | **OUTDATED** | Not updated since Mar 23, pre-dates NeuronPrint/reader-lm work |
| `LANGGRAPH_*.md` (5 files) | **DONE** | Separate LangGraph transcode track, valid |

### J. `.claude/prompts/` -- Staleness Audit

| File | Category | Reason |
|------|----------|--------|
| `session_A_v3_blasgraph_csc_planner.md` | **DONE** | Completed marker in file |
| `session_B_v3_bgz17_container_semiring.md` | **OUTDATED** | All deliverables done but prompt not updated |
| `session_C_v3_ndarray_bgz17_dualpath.md` | **OPEN TASK** | Prerequisites now satisfied (A+B done) |
| `session_D_v3_falkordb_retrofit.md` | **OPEN TASK** | Blocked on C |
| `session_MASTER_map_v3.md` | **OUTDATED** | Status markers stale |
| `session_6d_neuronprint_handover.md` | **DONE** (core) / **OPEN TASK** (items 3-12) | Latest prompt, highest precedence |
| `SESSION_BGZ_TENSOR_HYDRATE.md` | **TECH DEBT** | Empty file (0 bytes), placeholder never populated |
| `SESSION_CAPSTONE.md` | **DONE** | Architecture map complete |
| `FINAL_MAP.md` | **DONE** | 27 epiphanies documented |
| `session_ndarray_migration_inventory.md` | **OUTDATED** | Pre-dates 25-PR ndarray burst, many items now done |
| `session_integration_plan.md` | **OUTDATED** | Pre-dates NeuronPrint work |
| `session_epiphany_integration.md` | **OUTDATED** | Pre-dates later epiphany consolidation |
| `session_master_integration.md` | **OUTDATED** | Superseded by FINAL_MAP and CAPSTONE |
| `session_unified_26_epiphanies.md` | **OUTDATED** | Superseded by FINAL_MAP |
| `audio_codec_*.md` (4 files) | **OPEN TASK** | Audio research track, not started |
| `P18_INTERNAL_LLM_LANGUAGE_SURFACE.md` | **OPEN TASK** | LanguageBackend design, zero implementations |
| `VERIFY_COMPRESSION_REVOLUTION.md` | **OPEN TASK** | Verification protocol, not yet run end-to-end |
| `research_quantized_graph_algebra.md` | **DONE** | Research reference |
| All other session prompts | **DONE** | Completed work |

---

## III. CROSS-REPO DEPENDENCY GRAPH

```
ndarray (HPC foundation)
  ├── Base17, Plane, Fingerprint, Cascade, BF16Truth
  ├── p64 (Palette64, HeelPlanes, SIMD kernels)
  ├── ModelRouter (GPT-2, OpenChat) ──────────────────┐
  └── safetensors indexer (reader-lm, bge-m3 → bgz7)  │
                                                        │
lance-graph (graph engine)                              │
  ├── bgz17 crate (palette, distance, compose) ←── NOT │ WIRED as dep
  ├── AriGraph transcode (TripletGraph, EpisodicMemory) │
  ├── NeuronPrint (6D, neuron.rs)                       │
  ├── serve.rs (SPO + NARS, port 3000)                  │
  ├── lance-graph-contract (zero-dep types)             │
  ├── p64-bridge (CausalEdge64 → Palette)               │
  ├── reader-lm (forward pass, NO weight bridge) ←── BLOCKED
  ├── bge-m3 (forward pass, NO weight bridge) ←──── BLOCKED
  ├── lance-graph-osint (rule-based, NOT using models)  │
  └── LanguageBackend trait (0 implementations) ←── BLOCKED
                                                        │
q2 (application layer)                                  │
  ├── cockpit-server (port 2718, stubs) ←───── needs ───┘
  ├── neural-debug (runtime + MCP)
  ├── cockpit React UI (neural debugger, MRI, analyst)
  └── WIRE_LANCE_GRAPH.md prompt

AriGraph (Python, dormant)
  └── Reference only, fully transcoded to lance-graph
```

**Three critical blocking edges:**
1. `bgz17` not in lance-graph `Cargo.toml` → Session C cannot start
2. `bgz7-to-f32` bridge missing → reader-lm/bge-m3 cannot infer
3. `LanguageBackend` has 0 implementations → no natural language generation anywhere

---

## IV. STATISTICS

| Metric | Value |
|--------|-------|
| Total PRs audited | 60 (25+25+10+0) |
| Total `.claude/` files | 144 (63+51+30+0) |
| Prompt files audited | 70 (34+25+11) |
| Agent definitions | 13 (4 lance-graph + 9 ndarray) |
| Items categorized DONE | 42 |
| Items categorized OUTDATED | 14 |
| Items categorized TECH DEBT | 10 |
| Items categorized OPEN TASK | 18 |
| Items categorized DEPRECATED | 5 |
| Agent tool calls total | 257 |
| Audit duration | ~25 min (8 parallel agents) |
