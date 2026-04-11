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
.claude/agents/workspace-primer.md               — ONBOARDING FIRST: 20 canonical rules + mandatory reading list for any new session touching lance-graph / ndarray / reader-lm. Wake this agent BEFORE truth-architect or adk-coordinator so the session is oriented before specialist work begins.
.claude/knowledge/signed-session-findings.md     — BF16 tables, gate modulation, quality checks
.claude/knowledge/phi-spiral-reconstruction.md   — φ-spiral, family zipper, stride/offset, Zeckendorf, VSA
.claude/knowledge/primzahl-encoding-research.md  — prime fingerprint, Zeckendorf vs BF16 vs prime encoding
.claude/knowledge/bf16-hhtl-terrain.md           — BF16-HHTL correction chain, 5 hard constraints (§ C3 three-regime γ+φ rule), probe queue
.claude/knowledge/zeckendorf-spiral-proof.md     — φ-spiral proof (scope-limited, see header before citing)
.claude/knowledge/two-basin-routing.md           — Two-basin doctrine, representation routing, pairwise rule, attribution
.claude/knowledge/encoding-ecosystem.md          — MANDATORY: full encoding map, synergies, read-before-write checklist
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

### Production models (use these)

```
Model            Base       Tokenizer       Vocab   Hidden  Act   Status
─────            ────       ─────────       ─────   ──────  ───   ──────
Jina v5 small    Qwen3      Qwen 3.x BPE    151K    1024    silu  GROUND TRUTH (safetensors + ONNX on disk)
Reader-LM v3     = Jina v5  (same model, alternate name — BERT 3.x architecture lineage; NOT the older
                             Qwen2-based Reader-LM 1.5B/v1/v2. Use the Jina v5 entry above.)
Reranker v3      Qwen3      Qwen 3.x BPE    151K    1024    silu  SAME BASE as Jina v5 (listwise, cross-encoder)
Qwopus 27B       Qwen 3.5   Qwen 3.x BPE    151K    5120    silu  305 tables in Release v0.1.2
ModernBERT-large OLMo       OLMo BPE        50K     1024    gelu  ONNX on disk (GeGLU, code-friendly)

Tokenizer sharing (production):
  Qwen 3.x BPE (151K): Jina v5, Reader-LM v3, Reranker v3, Qwopus
  OLMo        (50K):   ModernBERT
```

**CRITICAL**:
- Reader-LM v3 and Jina v5 are the **same model** (Jina v5 BERT 3.x).
  Use either name; expect the same weights and tokenizer.
  Reader-LM 1.5B / v1 / v2 are DIFFERENT older models — see Research-only section.
- Reranker v3 = Qwen3 base (NOT XLM-RoBERTa). Same tokenizer as Jina v5.
  Baked reranker lens was built from an older Qwen2 GGUF → needs rebuild with Qwen 3.x tokens.
- Qwopus is Qwen 3.5 (NOT Qwen 2). Confirmed by `bgz-tensor` feature flags `qwen35-*`.

### Research-only / diagnostic fallback

These models are kept for v5-vs-older **behavioral diffing**. Do NOT use for
production work. Do reach for them when a Jina v5 result is surprising and
you need to isolate what architectural change between the pre-v5 era and now
produced the current behavior.

```
Model            Base       Tokenizer       Vocab   Hidden  Act   Status
─────            ────       ─────────       ─────   ──────  ───   ──────
Jina v3 small    XLM-R      SentencePiece   250K    1024    gelu  RESEARCH-ONLY — pre-v5 reference lens
BGE-M3           XLM-R      SentencePiece   250K    1024    gelu  RESEARCH-ONLY — multilingual reference
Reader-LM 1.5B   Qwen2      Qwen2 BPE       151K    —       —     RESEARCH-ONLY — v1/v2 Qwen2 lineage
                                                                  (NOT v3 = Jina v5)

Tokenizer sharing (research-only):
  XLM-RoBERTa (250K): Jina v3, BGE-M3
  Qwen2 BPE   (151K): Reader-LM 1.5B (v1/v2 only — v3 uses Qwen 3.x BPE)
```

### Precision hierarchy (workspace-wide rule)

```
Ground truth:    The source file on disk, byte-exact, SHA-pinned. For
                 Jina v5 this is `data/jina-v5-onnx/model.safetensors`
                 (verified BF16 by `examples/probe_jina_v5_safetensors.rs`,
                 NOT F16 as earlier notes claimed). For other models:
                 the GGUF / safetensors / ONNX bytes as downloaded.
                 Never duplicated, never re-baked, never "upscaled to
                 F32 and stored" — the F32 view is a method, not a buffer.

Compute:         BF16 with fused `mul_add` (hardware FMA: AVX-512
                 VDPBF16PS, ARM SVE BFMMLA, Apple AMX). Use
                 `ndarray::hpc::quantized::bf16_gemm_f32` (or
                 `mixed_precision_gemm`) and the BF16 lane conversions in
                 the same module. F32-precision accumulation happens in
                 hardware registers, invisible to the caller. BF16 memory
                 bandwidth.

Method F32:      F32 is a *method*, not a storage format. It may appear in
                 a stack window, a SIMD register, or the F32-accumulate
                 pipe of a hardware FMA instruction. It never persists as
                 a `Vec<f32>` buffer in the certification or bake pipelines.
                 If you find yourself allocating `Vec<f32>` for upscaled
                 temp data, stop — the source bytes plus the upcast method
                 already give you every F32 value you need on demand.

Source F16:      F16 → F32 is a deterministic bit-expansion with zero
                 information loss (F32 is a strict superset). The method
                 `ndarray::hpc::gguf::f16_to_f32` is proven lossless over
                 all 65,536 F16 bit patterns (±0, subnormals, normals, ±∞,
                 IEEE-correct QNaN payload preservation) by
                 `crates/thinking-engine/examples/probe_jina_v5_safetensors.rs`.
                 The Jina v5 *safetensors* on disk is BF16 not F16, but
                 other code paths (GGUF, reranker, some exports) still read
                 F16 so the primitive must be atomic-clock correct.

F16 → BF16:      Mantissa truncation 10 → 7 bits, NOT an exponent-range
                 issue. BF16 has MORE exponent bits than F16 (8 vs 5) and
                 covers ~33 orders of magnitude more range than F16's
                 ~65 504 maximum — every F16 value fits inside BF16 range
                 with room to spare. Earlier notes that said "F16 max
                 ~65 504 overflows before reaching BF16 range" were
                 backwards. The 3 lost mantissa bits round-to-nearest-even
                 (RNE) under hardware `_mm512_cvtneps_pbh`. The scalar
                 fallback `f32_to_bf16_scalar` in `src/simd_avx512.rs`
                 is plain truncation (not RNE) and therefore drifts by
                 ~1 ULP on ~50% of values from the hardware path —
                 pending replacement with a SIMD RNE routine for
                 certification reproducibility.

F64 constants:   π, e, φ, Euler-γ live as `std::f64::consts` 52-bit
                 mantissas. Used for calibration math (GammaProfile log /
                 exp, Dupain-Sós pre-rank selection). Applied to BF16
                 tensors by splatting a BF16-converted constant into a
                 lane and running the FMA — never by upscaling the tensor
                 to F32 and round-tripping.

Storage:         BF16 on disk for full-resolution weights; Base17 i16
                 fixed-point (×256 via GammaProfile-calibrated
                 quantization) for palette planes; u8 palette index for
                 HHTL HEEL/HIP; u8 distance matrix for DeepNSM 4096².

Discouraged:     8-bit quantization as a COMPUTE precision (Q4/Q5/Q8/INT8).
                 Fine only as a calibrated STORAGE format after the above
                 normalization chain. Never the precision of forward
                 passes.

Certification:   Every derived format (lab BF16, Base17, palette, bgz-hhtl-d)
                 is verified against the ground-truth source by Pearson,
                 Spearman, and Cronbach α reported to 4 decimal places.
                 Target: lab BF16 at ≥ 0.9999 (effectively lossless), bgz-
                 hhtl-d at ≥ 0.9980 after the inherent cascade entry tax.
                 The harness uses the deterministic SplitMix64 pair sampler
                 seeded 0x9E3779B97F4A7C15 so any two runs produce
                 bit-identical metrics.
```

## Certification Process

Every derived format in the workspace (lab BF16, Base17 i16 fixed-point,
γ+φ-calibrated `i8`/`u8`, Codebook4096 palette, bgz-hhtl-d cascade) must
be numerically certified against its source file before the derivation
is trusted as a runtime format. Certification answers the single
question **"does format X preserve the semantic properties of format Y
to target T?"** with a reproducible 4-decimal Pearson / Spearman /
Cronbach α report.

### Canonical artifacts

- **Process doctrine**: `.claude/knowledge/certification-harness.md`
- **Scoped agent**: `.claude/agents/certification-officer.md`
  (wake with "certify X against Y" for any lane × source pair)
- **Lane derivation**: `crates/thinking-engine/examples/seven_lane_encoder.rs`
  — the existing 7-lane encoder is the canonical bake tool. Do not
  rewrite it. The certification layer reuses its lane-derivation logic
  and adds an independent validation pass on top.
- **Metric primitives**:
  - `bgz_tensor::quality::pearson(x, y) -> f64` at `crates/bgz-tensor/src/quality.rs:13`
  - `bgz_tensor::quality::spearman(x, y) -> f64` at `crates/bgz-tensor/src/quality.rs:47`
  - `thinking_engine::cronbach::cronbach_alpha(items) -> f32` at `crates/thinking-engine/src/cronbach.rs:27`
- **Upcast primitives** (atomic-clock lossless):
  - `ndarray::hpc::gguf::f16_to_f32` (proven over all 65,536 F16 patterns)
  - `ndarray::hpc::quantized::BF16::to_f32` (trivial shift, lossless)
  - `ndarray::simd::f32_to_bf16_batch_rne` at `ndarray/src/simd_avx512.rs:1913`
    (pure AVX-512-F RNE, byte-exact vs hardware `_mm512_cvtneps_pbh` on 1M inputs)
- **Source access**:
  - Local mmap for files ≤ ~12 GB (Jina v5, BGE-M3, Reader-LM 1.5B)
  - `ndarray::hpc::safetensors::stream_index_safetensors_bf16` +
    `HttpRangeReader::with_chunk_size(url, size, 256 * 1024 * 1024)`
    for remote sources (Qwen 3.5 9B/27B/..., up to ~800 GB for 397B variants)
- **Pair sampler**: SplitMix64 with pinned seed `0x9E3779B97F4A7C15`
  (Knuth φ-fraction multiplicative hash constant), range
  `[0, min(tokenizer_vocab, embed_rows))`
- **Real-life corpus**: tier-1/2/3/4 calibration sentences from
  `crates/thinking-engine/examples/jina_v5_ground_truth.rs`

### The 7 channels and their targets

| Lane | Format          | Primary metric             | Target     |
|------|-----------------|----------------------------|------------|
| 1    | `u8` CDF        | Spearman ρ                 | ≥ 0.9990   |
| 2    | `i8` direct     | Pearson r                  | ≥ 0.9980   |
| 3    | `u8` γ+φ        | Spearman ρ                 | ≥ 0.9990   |
| 4    | `i8` γ+φ signed | Pearson r                  | ≥ 0.9980   |
| 5    | `f32` SiLU delta| ‖delta‖ reported           | no threshold |
| 6    | `bf16` RNE      | **Pearson + Spearman + Cronbach α** | **≥ 0.9999** (atomic clock) |
| 7    | `u8` drift      | mean / max drift reported  | no threshold |

Lane 6 is the atomic-clock lab BF16 lane — the only one certified
against all three metrics simultaneously at 0.9999 or better. Lanes
1-4 are compressed derivations with the single most-appropriate
metric; Lanes 5 and 7 are informational signals.

### Non-negotiable rules

1. **F32 is a method, not a buffer.** Never allocate `Vec<f32>` for
   upcast temp data. Stream from mmap or HTTP range, upcast in a
   stack window, discard. See workspace-primer Rules 6-7.
2. **Real-life corpus only.** No synthetic test inputs. Every
   certification samples via the deterministic SplitMix64 pair
   sampler + tier-1..4 text corpus. Rule 23.
3. **NaN scan at every stage.** Source → upcast → cosine → metrics.
   A single NaN halts the run with a diagnostic JSON. The Apr 11
   2026 `f16_to_f32` quiet-bit glitch (fixed in ndarray `17bfde3`)
   is the precedent. Rule 22.
4. **Reference is ALWAYS re-derived from source.** Never trust an
   on-disk `cosine_matrix_*.f32` file as the reference. The reference
   column is computed fresh on every run via the proven-lossless
   upcast method.
5. **Lab BF16 derivation uses RNE, not truncation.** Call
   `f32_to_bf16_batch_rne` or `f32_to_bf16_scalar_rne` — not the
   legacy `f32_to_bf16` truncation path, which drifts ~1 ULP from
   hardware RNE on ~50 % of values and cannot hit 0.9999.
6. **Two-universes firewall.** Basin 1 (continuous neural embeddings)
   and Basin 2 (discrete distributional codebooks / DeepNSM CAM-PQ)
   cannot be certified across each other as if they were the same
   format. Rule 21. Cross-basin measurements are valid only as
   explicit multi-lens Cronbach α runs, not as "does X preserve Y".

### Statistical confidence intervals (v2.5)

The harness produces two confidence intervals per lane, side by side, so
the certification report carries its own internal consistency check:

| Tool | Kind | Role | Citation |
|------|------|------|----------|
| **Fisher z** | closed-form parametric (arctanh, SE = 1/√(n−3)) | **3σ authority** — published 4-decimal CI. Zero sampling noise, unlimited z-space resolution, but assumes bivariate normality. | Fisher (1915, 1921) |
| **BCa bootstrap** | distribution-free (bias z₀ + accel a) | **2σ cross-check** — nonparametric consistency test. At B=2000 resolves 2σ tails to ~45 samples per endpoint. | Efron (1987) JASA 82(397); Efron & Tibshirani (1994) Ch. 14 |
| **Jackknife SE** | grouped leave-one-centroid-out | provides the acceleration `a` for BCa and a separate 3σ envelope estimator. Delete-255 group jackknife on 256 balanced groups. | Efron & Tibshirani (1994) §11.5 |

BCa is the literature standard for correlation metrics on embedding
distributions per Deutsch 2021 (arXiv:2104.00054), because the normality
assumption Fisher z requires does **not** hold for cosine-similarity
distributions on text embeddings (Mu & Viswanath 2018 "All-but-the-Top";
Ethayarajh 2019 "How Contextual are Contextualized Word Representations?").
The harness therefore publishes Fisher z as the 3σ authority but cross-
checks it against BCa at 2σ. Agreement within the BCa/Fisher width ratio
~1.0 confirms both methods; material disagreement is a doctrine-level
failure because it means the Fisher z assumption has broken down and the
distribution-free BCa is the correct published CI.

3σ BCa requires B ≥ 5000 to resolve the 0.135% tail to ≥ 6 samples per
endpoint; the default B=2000 is a 2σ configuration and the reported 3σ
BCa bounds are informational only.

### CHAODA outlier protection (v2.5)

The harness runs a CHAODA sanity pass using `ndarray::hpc::clam::ClamTree::
anomaly_scores` (Ishaq et al. 2021 "Clustered Hierarchical Anomaly and
Outlier Detection Algorithms", arXiv:2103.11774; Phase 4 of
`ndarray/.claude/prompts/01_clam_qualiacam.md`). Top-10% most-anomalous
centroids by LFD-normalized score are removed and the primary metric is
recomputed on the filtered pair set. Three verdicts:

- `clean` (|Δ| < 1e-4): distribution is CHAODA-clean, certification
  stands as published.
- `filtering_helps` (Δ > 1e-4): outliers were depressing the metric —
  flag for operator review, filtered metric is the more honest number.
- `filter_removed_easy_pairs` (Δ < −1e-4): the flagged points were
  actually the high-correlation ones — LFD-based CHAODA is not
  capturing true outliers in this lane and the verdict is advisory.

The point of CHAODA is to distinguish BGZ-class compression error
(structural) from contamination error (contingent on specific outlier
pairs). If a certification shows `clean` across all lanes, the error
the lanes hit is pure compression cost and that defines the floor the
endgame `bgz-hhtl-d` format must approach.

### BGZ distribution floor (v2.5)

The harness reports a naive uniform-u8 quantizer baseline over
`[min, max]` of the reference cosine distribution. This is the
simplest compressed representation that fits in one byte per pair
(max |err| = (max−min)/512) and defines the information-theoretic
floor that any real u8 lane must beat to justify its complexity.
γ+φ+CDF (Lane 3) should deliver Δρ > 0 over this floor or the
projection adds no value and should be dropped.

**Endgame framing.** `bgz-hhtl-d` (the obligatory runtime cascade
format) is gated at ≥ 0.9980 Pearson after its inherent cascade entry
tax. The "cascade entry tax" is the delta between bgz-hhtl-d's measured
Pearson and this naive u8 floor; the delta between γ+φ+CDF and the
naive floor is the budget the HHTL cascade has left to spend before
hitting the 0.9980 target. Every future certification of a compressed
lane ends by quoting (Pearson − naive_u8_pearson) so the cascade-tax
budget is visible at a glance.

### How to certify a new model

1. Wake `certification-officer` with the source path, the tensor
   name (e.g. `embed_tokens.weight`), and the target lane(s).
2. The agent reads this section, the knowledge doc, and the source
   header. Identifies dtype, vocab, and whether the tokenizer vocab
   matches the embed rows.
3. The agent runs `seven_lane_encoder.rs` to produce (or re-use) the
   7-lane tables. Applies the Lane 6 RNE swap if not yet applied to
   the encoder.
4. The agent samples 1000 random pairs via SplitMix64 + corpus pairs,
   runs them through the lane-derivation logic as an independent
   validation set, computes per-lane × per-sample-set metrics.
5. The agent writes JSON to
   `.claude/knowledge/certification/{model_slug}_{derivation_slug}.json`
   and returns a one-line summary.
6. If `PASS`, the derivation is certified and can be used as a
   runtime format. If `FAIL`, the agent returns the top-5 worst
   pairs by error magnitude for diagnosis.

### Retest policy

Any certification result where the source or derivation touched
`ndarray::hpc::gguf::f16_to_f32` **before Apr 11 2026** is a
retest candidate per workspace-primer Rule 22 (the NaN quiet-bit
glitch fix in `17bfde3`). Known-stale artifacts include
`jina-v5-7lane/`, `jina-v5-codebook/`, `jina-reranker-v3-BF16-5lane/`,
`jina-reranker-v3-BF16-7lane/`. Retest by re-running the encoder
under the fixed primitives and comparing new metrics against old
byte-for-byte; zero-or-within-f32-rounding delta exonerates, material
delta invalidates.

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
