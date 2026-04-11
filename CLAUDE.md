# CLAUDE.md — lance-graph

> **Updated**: 2026-03-28
> **Role**: The obligatory spine — query engine, codec stack, semantic transformer, and orchestration contract
> **Status**: 11 crates, 5 in workspace, 4 excluded (standalone), Phases 1-2 DONE, Phase 3 IN PROGRESS

---

## What This Is

Graph query engine AND cognitive codec stack for the Ada architecture. lance-graph is no longer
just "The Face" — it has become the **obligatory spine** together with ndarray. Everything flows
through lance-graph: Cypher/GQL/Gremlin/SPARQL parsing, semiring algebra, SPO triples,
CAM-PQ compressed search, distributional semantics (DeepNSM), attention-as-table-lookup
(bgz-tensor), thinking orchestration, and the contract crate that unifies all consumers.

```
Architecture (post-expansion):
  ndarray            = The Foundation  (SIMD, GEMM, HPC, Fingerprint<256>, CAM-PQ codec)
  lance-graph        = The Spine       (query + codec + semantics + contracts)  <-- THIS REPO
  ladybug-rs         = The Brain       (BindSpace, SPO server, 4096 surface)
  crewai-rust        = The Agents      (agent orchestration, thinking styles)
  n8n-rs             = The Orchestrator (workflow DAG, step routing)

Dependency chain:
  ladybug-rs ──► ndarray (path dep, direct)
  ladybug-rs ──► lance-graph-contract (traits)
  n8n-rs     ──► lance-graph-contract (traits)
  crewai-rust──► lance-graph-contract (traits)
  lance-graph──► ndarray (default dep, with fallback)
```

---

## Workspace Structure

```toml
[workspace]
members = [
    "crates/lance-graph",          # Core: Cypher parser, DataFusion planner, graph algebra, NSM
    "crates/lance-graph-catalog",  # Catalog providers (Unity Catalog, connectors)
    "crates/lance-graph-python",   # Python bindings (PyO3/maturin)
    "crates/lance-graph-benches",  # Benchmarks
    "crates/lance-graph-planner",  # Unified query planner (16 strategies, MUL, thinking, elevation)
    "crates/lance-graph-contract", # Zero-dep trait crate (THE single source of truth for types)
    "crates/neural-debug",         # Static scanner + runtime registry
]
exclude = [
    "crates/lance-graph-codec-research",  # ZeckBF17 research codec
    "crates/bgz17",                       # Palette semiring codec (0 deps, 121 tests)
    "crates/deepnsm",                     # Distributional semantic engine (0 deps, 4096 COCA)
    "crates/bgz-tensor",                  # Metric-algebraic tensor codec (attention as lookup)
]
```

### Crate Details

**lance-graph** (core, ~18K LOC) — `crates/lance-graph/`
- `parser.rs` + `ast.rs` — Cypher parser (nom, 44 tests)
- `semantic.rs` — semantic analysis
- `logical_plan.rs` — logical plan
- `datafusion_planner/` — Cypher→DataFusion SQL (~6K LOC: scan, join, predicate pushdown, vector, UDF, cost)
- `graph/spo/` — SPO triple store (truth, merkle, semiring, builder, 30 tests)
- `graph/blasgraph/` — GraphBLAS sparse matrix (~5K LOC: 7 semirings, CSR/CSC/COO/HyperCSR, HHTL, cascade)
- `graph/neighborhood/` — neighborhood search (CLAM, scope, zeckf64)
- `graph/metadata.rs` — MetadataStore (Arrow RecordBatch CRUD)
- `graph/graph_router.rs` — Three-backend graph router (blasgraph/DataFusion/palette)
- `nsm/` — DeepNSM DataFusion wiring (tokenizer, parser, encoder, similarity, UDF scaffold)
- `nsm_bridge.rs` — NSM→SPO mapping with NARS truth values + Arrow export (13 tests)

**lance-graph-planner** (10,326 LOC) — `crates/lance-graph-planner/`
- 16 composable strategies: CypherParse, GqlParse, GremlinParse, SparqlParse, ArenaIR, DPJoinEnum, RuleOptimizer, HistogramCost, SigmaBandScan, MorselExec, TruthPropagation, CollapseGate, StreamPipeline, JitCompile, WorkflowDAG, ExtensionPlanner
- `thinking/` — 12 styles, NARS dispatch, sigma chain (Ω→Δ→Φ→Θ→Λ), semiring auto-selection
- `mul/` — Meta-Uncertainty Layer (Dunning-Kruger, trust qualia, compass, homeostasis, gate)
- `elevation/` — dynamic elevation L0:Point→L5:Async (cost model that smells resistance)
- `adjacency/` — Kuzu-style CSR/CSC substrate with batch intersection
- `physical/` — CamPqScanOp, CollapseOp, TruthPropagatingSemiring
- `api.rs` — Planner + CamSearch + Polyglot detection

**lance-graph-contract** (1,076 LOC, ZERO DEPS) — `crates/lance-graph-contract/`
- `thinking.rs` — 36 ThinkingStyles, 6 clusters, τ addresses, FieldModulation, ScanParams
- `mul.rs` — SituationInput, MulAssessment, DkPosition, TrustTexture, FlowState, GateDecision
- `plan.rs` — PlannerContract trait, PlanResult, QueryFeatures, StrategySelector
- `orchestration.rs` — OrchestrationBridge trait, StepDomain, UnifiedStep, BridgeSlot
- `cam.rs` — CamCodecContract, DistanceTableProvider, IvfContract
- `jit.rs` — JitCompiler, StyleRegistry, KernelHandle
- `nars.rs` — InferenceType(5), QueryStrategy(5), SemiringChoice(5)

**deepnsm** (standalone, ~2,200 LOC, 0 deps) — `crates/deepnsm/`
- Replaces transformer inference: 680GB model → 16.5MB, 50ms/token → <10μs/sentence
- 4,096-word COCA vocabulary (98.4% English coverage)
- 4096² u8 distance matrix from CAM-PQ codebook
- 512-bit VSA encoder: XOR bind + majority bundle (word order sensitive)
- 6-state PoS FSM → SPO triples (36-bit packed)

**bgz-tensor** (standalone, ~1,300 LOC, 0 deps) — `crates/bgz-tensor/`
- Attention via table lookup: Q·K^T/√d → table[q_idx][k_idx] in O(1)
- Weight matrix 64MB → Base17 136KB → 256 archetypes 8.5KB → distance table 128KB
- AttentionSemiring: distance table (u16) + compose table (u8) = multi-hop in O(1)
- HHTL cascade: 95% of pairs skipped

**bgz17** (standalone, ~3,500 LOC, 0 deps, 121 tests) — `crates/bgz17/`
- Palette semiring codec, PaletteMatrix mxm, PaletteCsr, Base17 VSA
- SIMD batch_palette_distance, TypedPaletteGraph, container pack/unpack

---

## ndarray Integration Policy

**ndarray is the default dependency.** lance-graph should always use ndarray for:
- `Fingerprint<256>` (canonical type, replaces standalone `NdarrayFingerprint` mirror)
- SIMD dispatch via `simd_caps()` singleton
- CAM-PQ codec (ndarray implements `CamCodecContract`)
- CLAM tree (ndarray has 46 tests, full build+search+rho_nn)
- BLAS L1/L2/L3 via MKL/OpenBLAS/native backend
- ZeckF64 (ndarray is canonical, 3 copies need dedup)
- HDR cascade search
- JIT compilation via jitson/Cranelift

**Fallback without ndarray**: When the `ndarray-hpc` feature is disabled, lance-graph falls
back to its standalone implementations in `blasgraph/ndarray_bridge.rs`. This is for:
- Minimal builds (CI, wasm, embedded)
- Downstream consumers who don't need HPC compute
- Compilation without the full ndarray dependency tree

```toml
# In crates/lance-graph/Cargo.toml:
[dependencies]
ndarray = { path = "../../../ndarray", optional = true, default-features = false }

[features]
default = ["unity-catalog", "delta", "ndarray-hpc"]
ndarray-hpc = ["dep:ndarray"]
```

---

## Build Commands

```bash
# Check workspace (default: with ndarray)
cargo check

# Check without ndarray (fallback mode)
cargo check -p lance-graph --no-default-features

# Run core tests
cargo test -p lance-graph

# Run planner tests
cargo test -p lance-graph-planner

# Run contract tests
cargo test -p lance-graph-contract

# Run standalone codec tests (fast, no network)
cargo test --manifest-path crates/bgz17/Cargo.toml
cargo test --manifest-path crates/deepnsm/Cargo.toml
cargo test --manifest-path crates/bgz-tensor/Cargo.toml

# Run all workspace tests
cargo test

# Python bindings
cd crates/lance-graph-python && maturin develop
```

---

## Current Status (2026-03-28)

### What's DONE
- **Phase 1** (blasgraph CSC/Planner): DONE — CscStorage, HyperCsrStorage, TypedGraph, 7 semirings, SIMD Hamming
- **Phase 2** (bgz17 container/semiring): DONE — 121 tests, PaletteSemiring, PaletteMatrix, PaletteCsr, Base17 VSA
- **Unified planner**: DONE — 16 strategies, MUL assessment, thinking orchestration, dynamic elevation, CAM-PQ operator
- **Contract crate**: DONE — zero-dep trait crate with canonical types for all consumers
- **Polyglot parsing**: DONE — Cypher, GQL (ISO 39075), Gremlin, SPARQL → same IR
- **DeepNSM**: DONE — distributional semantic engine, 4096 COCA vocabulary, 512-bit VSA
- **bgz-tensor**: DONE — attention as table lookup, AttentionSemiring, HHTL cascade
- **NSM bridge**: DONE — NSM→SPO mapping, NARS truth values, Arrow 57 RecordBatch, 13 tests
- **Deep audit**: DONE — 8 inventory documents in docs/ (see docs/UNIFIED_INVENTORY.md)

### What's IN PROGRESS (Phase 3)
- [ ] Wire ndarray as default dep (Cargo.toml change + `ndarray-hpc` feature flag)
- [ ] Replace `NdarrayFingerprint` with `ndarray::hpc::fingerprint::Fingerprint<256>`
- [ ] Dedup ZeckF64 (3 copies → 1 canonical in ndarray)
- [ ] Wire CAM-PQ: ndarray codec → lance-graph UDF → planner operator
- [ ] Contract adoption: planner + n8n-rs + crewai-rust depend on contract crate
- [ ] Move bgz17 from `exclude` to `members` with `bgz17-codec` feature flag
- [ ] Consolidate nsm/ module to thin wrappers over crates/deepnsm/

### What's OPEN (Phase 4+)
- [ ] Wire planner strategies to lance-graph core (actual parser, not regex)
- [ ] Wire elevation to execution with timing feedback
- [ ] GraphRouter 3-backend routing (DataFusion + palette + blasgraph)
- [ ] n8n-rs OrchestrationBridge implementation
- [ ] End-to-end integration test (query → thinking → plan → execute → result)

### Test Summary
- bgz17: **121 passing** (standalone)
- lance-graph core: **~100+ passing** (SPO + Cypher + semirings + NSM bridge)
- lance-graph-planner: **~15 passing** (strategy selection, truth propagation, collapse gate)
- lance-graph-contract: **~15 passing** (thinking styles, τ addresses, modulation)
- Total: **250+ passing**

---

## Key Dependencies

```toml
arrow = "57"
datafusion = "51"
lance = "2"
lance-linalg = "2"
ndarray = { path = "../../../ndarray" }  # AdaWorldAPI fork, default, optional fallback
nom = "7.1"
snafu = "0.8"
deltalake = "0.30"  # optional
```

---

## Architecture Notes

### The Codec Stack (see docs/CODEC_COMPRESSION_ATLAS.md)
```
Full planes (16Kbit, ρ=1.000) → ZeckBF17 (48B, ρ=0.982) → Base17 (34B, ρ=0.965)
  → PaletteEdge (3B, ρ=0.937) / CAM-PQ (6B, varies) → Scent (1B, ρ=0.937)
```

### The Thinking Pipeline (see docs/THINKING_ORCHESTRATION_WIRING.md)
```
YAML card → 23D vector → ThinkingStyle(36) → FieldModulation(7D) → ScanParams
  → MUL assessment → NARS type → semiring selection → 16 strategies → elevation
```

### The Semiring Inventory (see docs/SEMIRING_ALGEBRA_SURFACE.md)
- 7 HDR semirings (blasgraph): XorBundle, BindFirst, HammingMin, SimilarityMax, Resonance, Boolean, XorField
- 1 SPO truth semiring (spo/): HammingMin with frequency/confidence
- 1 palette semiring (bgz17): PaletteCompose with 256×256 distance table
- 1 planner semiring (planner): TruthPropagating with NARS deduction/revision
- 1 attention semiring (bgz-tensor): AttentionTable + ComposeTable
- 5 contract semiring choices: Boolean, HammingMin, NarsTruth, XorBundle, CamPqAdc

### Type Duplication (see docs/TYPE_DUPLICATION_MAP.md)
- Fingerprint/BitVec: 4 copies (ndarray canonical)
- ZeckF64: 3 copies (ndarray canonical)
- ThinkingStyle: 4 copies (contract canonical, not yet adopted)
- Base17: 3 copies (bgz17 canonical)
- NARS InferenceType: 3 copies (contract canonical)
- CSR adjacency: 5 implementations (different purposes)

---

## Cross-Repo Dependencies

```
WHO DEPENDS ON lance-graph-contract:
  ladybug-rs     — PlannerContract, OrchestrationBridge
  crewai-rust    — ThinkingStyleProvider, MulProvider
  n8n-rs         — JitCompiler, StyleRegistry, OrchestrationBridge

WHO DEPENDS ON lance-graph:
  ladybug-rs     — "stolen" parser copy (to be replaced with dep)

WHO WE DEPEND ON:
  ndarray        — Fingerprint, CAM-PQ, CLAM, BLAS, ZeckF64, HDR cascade, JIT
  lance          — columnar storage, versioning, vector search
  datafusion     — SQL query engine
  arrow          — columnar memory format

SIBLING REPOS:
  /home/user/ndarray/       — The Foundation (BLAS, Fingerprint, CAM-PQ, CLAM, jitson)
  /home/user/ladybug-rs/    — The Brain (BindSpace, 4096 surface, SPO server)
  /home/user/crewai-rust/   — The Agents (agent cards, thinking styles, YAML templates)
  /home/user/n8n-rs/        — The Orchestrator (workflow DAG, step routing, compiled styles)
  /home/user/kuzudb/        — Reference (column-grouped CSR adjacency model)
```

---

## Knowledge Base (agents read these before working)

```
.claude/knowledge/signed-session-findings.md     — BF16 tables, gate modulation, quality checks
.claude/knowledge/phi-spiral-reconstruction.md   — φ-spiral, family zipper, stride/offset, Zeckendorf, VSA
.claude/knowledge/primzahl-encoding-research.md  — prime fingerprint, Zeckendorf vs BF16 vs prime encoding
.claude/knowledge/bf16-hhtl-terrain.md           — BF16-HHTL correction chain, 5 hard constraints, probe queue
.claude/knowledge/zeckendorf-spiral-proof.md     — φ-spiral proof (scope-limited, see header before citing)
.claude/knowledge/two-basin-routing.md           — Two-basin doctrine, representation routing, pairwise rule, attribution
.claude/knowledge/encoding-ecosystem.md          — MANDATORY: full encoding map, synergies, read-before-write checklist
.claude/knowledge/frankenstein-checklist.md       — Composition failure modes (VibeTensor §7), boundary test matrix
.claude/CALIBRATION_STATUS_GROUND_TRUTH.md       — OVERRIDE: read BEFORE any SESSION_*.md
.claude/PLAN_BF16_DISTANCE_TABLES.md             — 5-phase plan for BF16 distance tables
.claude/TECHNICAL_DEBT_SIGNED_SESSION.md          — 56% useful, 44% bypass (honest review)
.claude/CODING_PRACTICES.md                       — 6 patterns from EmbedAnything + quality checks
```

## Knowledge Activation (MANDATORY)

**P0 Rule: `.claude/knowledge/encoding-ecosystem.md` must be read BEFORE any
codec, encoding, distance, compression, or representation work.** This is the
map of all 8+ encoding representations, their crate locations, their invariants,
their synergies, and their FINDING/CONJECTURE status. Never guess architecture.

Every `.claude/knowledge/` document has a `READ BY:` header listing which agents
MUST load it before producing output in that domain. When a knowledge trigger fires
(see `.claude/agents/README.md § Knowledge Activation Protocol`), the relevant
knowledge docs are loaded BEFORE the agent responds.

**Critical process rule:** `.claude/knowledge/bf16-hhtl-terrain.md` contains a
probe queue with CONJECTURE/FINDING status for each architectural claim. Any agent
proposing changes to HHTL cascade, γ+φ placement, Slot D/V layout, or bucketing
strategy MUST check the probe queue first. If the relevant probe is NOT RUN, the
next deliverable is the probe, not more synthesis.

**Insight update cycle:**
```
Claim → Probe defined (pass/fail criteria) → Probe written (example file)
→ Probe run → Result recorded in knowledge doc → CONJECTURE promoted to FINDING or corrected
```
No knowledge doc should contain unmarked conjectures. Label everything.

## Model Registry (Jina v5 is ground truth anchor)

```
Model            Base       Tokenizer      Vocab    Hidden  Act   Status
─────            ────       ─────────      ─────    ──────  ───   ──────
Jina v5 small    Qwen3      Qwen3 BPE      151K    1024    silu  GROUND TRUTH (safetensors + ONNX on disk)
Reranker v3      Qwen3      Qwen3 BPE      151K    1024    silu  SAME BASE as v5 (listwise, cross-encoder)
ModernBERT-large OLMo       OLMo BPE       50K     1024    gelu  ONNX on disk (GeGLU, code-friendly)
BGE-M3           XLM-R      SentencePiece  250K    1024    gelu  baked u8 lens (multilingual)
Jina v3          XLM-R      SentencePiece  250K    1024    gelu  baked u8 lens (LEGACY, not ground truth)
Qwopus 27B       Qwen2      Qwen2 BPE      151K    5120    silu  305 tables in Release v0.1.2
Reader-LM 1.5B   Qwen2      Qwen2 BPE      151K    —       —     in Release, not baked

CRITICAL: Reranker v3 = Qwen3 base (NOT v2 XLM-RoBERTa).
  Same tokenizer as Jina v5. Same architecture. Same silu gate.
  Baked reranker lens was built from Qwen2 GGUF → needs rebuild with Qwen3 tokens.

Tokenizer sharing:
  Qwen3 BPE (151K): Jina v5, Reranker v3
  Qwen2 BPE (151K): Qwopus, Reader-LM (DIFFERENT from Qwen3!)
  XLM-RoBERTa (250K): Jina v3, BGE-M3 (LEGACY)
  OLMo (50K): ModernBERT
```

## Thinking Engine (crates/thinking-engine/)

```
Core:        engine.rs, bf16_engine.rs, signed_engine.rs, role_tables.rs
Composition: composite_engine.rs, dual_engine.rs, layered.rs, domino.rs
Calibration: cronbach.rs, ground_truth.rs, reencode_safety.rs (x256 proven)
Encoding:    spiral_segment.rs, prime_fingerprint.rs (VSA bundle perturbation)
Patterns:    pooling.rs, builder.rs, auto_detect.rs, tokenizer_registry.rs
Sensors:     jina_lens.rs, bge_m3_lens.rs, reranker_lens.rs, sensor.rs
Cognition:   cognitive_stack.rs, ghosts.rs, persona.rs, qualia.rs, world_model.rs
Bridge:      l4_bridge.rs, bridge.rs, semantic_chunker.rs, tensor_bridge.rs

Examples:
  jina_v5_ground_truth.rs    — end-to-end pipeline (tokenize → ground truth)
  end_to_end_signed.rs       — BF16/i8 smoke test (CDF collapse confirmed)
  dual_signed_experiment.rs  — u8 vs BF16 comparison across 3 lenses
  calibrate_lenses.rs        — Spearman ρ + ICC (real Qwen3 tokenizer)
  stream_signed_lens.rs      — 5-lane encoder from GGUF (BF16 stream)
  stream_hdr_lens.rs         — u8 CDF HDR lens from GGUF

Data on disk (gitignored, download from HF or Releases):
  jina-v5-onnx/              model.safetensors (1.2 GB) + model.onnx (2.3 GB) + tokenizer
  modernbert-onnx/           model.onnx (1.5 GB) + tokenizer
  jina-reranker-v3-BF16-5lane/ 5-lane encoding (u8/i8/γ+φ, 1.1 MB)
```

## Documentation Index

```
docs/UNIFIED_INVENTORY.md              — Master index, LOC census, crate map, dep graph
docs/TYPE_DUPLICATION_MAP.md           — Every duplicated type with file:line precision
docs/CODEC_COMPRESSION_ATLAS.md        — Full→ZeckBF17→BGZ17→CAM-PQ→Scent chain
docs/THINKING_ORCHESTRATION_WIRING.md  — End-to-end thinking pipeline (9 layers)
docs/SEMIRING_ALGEBRA_SURFACE.md       — All 15 semirings across all repos
docs/ADJACENCY_SYNERGY_MAP.md          — 6 CSR models, BLAS×PQ×BGZ17, KuzuDB
docs/METADATA_SCHEMA_INVENTORY.md      — 4096 surface, 16Kbit, Lance persistence, Arrow
docs/INTEGRATION_DEBT_AND_PATHS.md     — Strengths, weaknesses, epiphanies, 7 paths
docs/ORCHESTRATION_IS_GRAPH.md         — Capstone: orchestration AS graph traversal
docs/CONSUMER_WIRING_INSTRUCTIONS.md   — How to consume lance-graph-contract
```

## Session: AutocompleteCache + p64 Convergence (2026-03-31)

### New in lance-graph-planner
- `src/cache/` — 7 modules, 39 tests:
  - `kv_bundle.rs`: HeadPrint=Base17 (from ndarray), AttentionMatrix 64×64/256×256
  - `candidate_pool.rs`: ranked candidates, Phase (Exposition→Coda)
  - `triple_model.rs`: self/user/impact × 4096 heads, DK, Plasticity, Truth=NarsTruth
  - `lane_eval.rs`: Euler-gamma tension, DK-adaptive, 4096-head evaluation
  - `nars_engine.rs`: SpoHead, Pearl 2³, NarsTables (causal-edge hot path), StyleVectors
  - `convergence.rs`: AriGraph triplets → p64 Palette layers → Blumenstrauss
  - `kv_bundle.rs`: VSA superposition store
- `src/strategy/chat_bundle.rs`: AutocompleteCacheStrategy (Strategy #17)
- `src/serve.rs`: Axum REST server, OpenAI-compatible /v1/chat/completions
- `AUTOCOMPLETE_CACHE_PLAN.md`: full implementation plan with 6 agent scopes

### New in bgz-tensor
- `src/hhtl_cache.rs`: HHTL cache with RouteAction (Skip/Attend/Compose/Escalate), HipCache k=64
- `src/hydrate.rs`: --download/--reindex/--verify with feature flags (qwen35-9b/27b-v1/v2)
- `data/manifest.json`: SHA256 for all 41 shards
- GitHub Release v0.1.0-bgz-data: 41 bgz7 assets, 685 MB

### Dependencies
- lance-graph-planner → ndarray (hardware: Base17, read_bgz7_file)
- lance-graph-planner → causal-edge (protocol: CausalEdge64, NarsTables)
- lance-graph-planner → p64 + p64-bridge + bgz17 (convergence highway)

### Architecture Rules
- ndarray = hardware acceleration (SIMD, no thinking logic)
- causal-edge = protocol (CausalEdge64, NarsTables = precomputed NARS as lookup tables)
- lance-graph-planner = thinking (NarsEngine, AutocompleteCache, Styles)
- p64 = convergence point (both repos meet, no circular deps)
- AriGraph (lance-graph core) cannot be planner dep (circular) — use p64 convergence instead

### 18 Papers Synthesized
EMPA, InstCache, Semantic Caching, C2C, ContextCache, Krites, Thinking Intervention,
ThinkPatterns, Thinkless, Holographic Resonance, DapQ, Tensor Networks, PMC Attention Heads,
LFRU, Illusion of Causality, NARS Same/Opposite, KVTC, CacheSlide.
All findings in `.claude/knowledge/session_autocomplete_cache.md`.

### Benchmark
611M SPO lookups/sec. 17K tokens/sec. 388 KB RAM. 100% information preservation.
