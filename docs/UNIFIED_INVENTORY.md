# Unified Inventory: The Complete Map

> *Generated 2026-03-27. Covers lance-graph (HEAD 366b914), ndarray (HEAD 51018efc),
> ladybug-rs, n8n-rs, crewai-rust, kuzudb after full rebase onto latest upstream.*

## 1. Repository Census

| Repo | Branch | HEAD | LOC (Rust) | Crates | Tests | Build Status |
|------|--------|------|-----------|--------|-------|-------------|
| **lance-graph** | `claude/inventory-deep-audit-VrJae` = `origin/main` | `366b914` | ~30K | 8 (5 in workspace, 2 excluded, 1 neural-debug) | 84+ lib, 121 bgz17 | Compiles |
| **ndarray** | `claude/inventory-deep-audit-VrJae` = `origin/master` | `51018ef` | ~60K | 6 (ndarray core + 5 sub) | 880 lib, 300 doctest | exit 101 (broken) |
| **ladybug-rs** | `claude/inventory-deep-audit-VrJae` | — | ~40K | 1 binary | — | Compiles with ndarray path dep |
| **n8n-rs** | `claude/inventory-deep-audit-VrJae` | — | ~5K Rust (n8n-rust/) | 3 (server, contract, grpc) | — | — |
| **crewai-rust** | `claude/inventory-deep-audit-VrJae` | — | ~15K | 1 | — | — |
| **kuzudb** | `claude/inventory-deep-audit-VrJae` | empty | 0 | — | — | Bare .git only |

## 2. Lance-Graph Crate Map (Post-Rebase)

```
lance-graph/                              # Workspace root
├── Cargo.toml                            # resolver = "2"
│
├── crates/lance-graph/                   # CORE: Cypher parser + DataFusion planner + graph algebra
│   ├── src/parser.rs                     #   1,932L  nom Cypher parser (MATCH/CREATE/SET/DELETE/RETURN)
│   ├── src/ast.rs                        #     544L  AST nodes
│   ├── src/semantic.rs                   #   1,719L  Semantic validation
│   ├── src/logical_plan.rs               #   1,417L  Logical planning
│   ├── src/query.rs                      #   1,999L  CypherQuery public API
│   ├── src/datafusion_planner/           #   ~6,000L DataFusion backend (scan, join, UDF, vector, cost, pushdown)
│   ├── src/graph/metadata.rs             #     533L  MetadataStore: Arrow RecordBatch CRUD
│   ├── src/graph/spo/                    #   ~2,500L SPO triple store (truth, merkle, semiring, builder)
│   ├── src/graph/blasgraph/              #   ~5,000L GraphBLAS algebra (7 semirings, CSR/CSC/COO/HyperCSR)
│   ├── src/graph/neighborhood/           #   ~2,000L Neighborhood vectors, scent, zeckf64, CLAM
│   ├── src/graph/falkor_compat.rs        #     476L  FalkorDB 3-backend router
│   └── tests/ (14 files)                 #   ~5,000L Integration tests
│
├── crates/lance-graph-planner/           # NEW: Unified query planner (16 strategies, MUL, thinking)
│   ├── src/lib.rs                        #     468L  PlannerAwareness entry point
│   ├── src/api.rs                        #     669L  Planner + CamSearch + Polyglot API
│   ├── src/thinking/ (4 modules)         #     636L  Style selection, NARS dispatch, sigma chain, semiring
│   ├── src/mul/ (6 modules)              #     704L  Meta-Uncertainty Layer (DK, trust, compass, gate, flow)
│   ├── src/strategy/ (16 modules)        #   ~3,200L 16 composable strategies (parse→plan→optimize→execute)
│   ├── src/elevation/ (5 modules)        #     800L  Dynamic elevation (L0:Point→L5:Async)
│   ├── src/adjacency/ (5 modules)        #     800L  Kuzu-style CSR/CSC substrate
│   ├── src/physical/ (5 modules)         #   ~1,200L CamPqScanOp, Collapse, Broadcast, Accumulate
│   ├── src/nars/ (3 modules)             #     230L  TruthValue + Inference rules
│   ├── src/ir/ (5 modules)               #     800L  Arena-allocated LogicalOp IR
│   └── src/plan/ + optimize/ + execute/  #     800L  Cost, DP enumeration, rule optimizer
│   TOTAL: 10,326L
│
├── crates/lance-graph-contract/          # NEW: Zero-dep trait crate (THE SINGLE SOURCE OF TRUTH)
│   ├── src/thinking.rs                   #     325L  36 styles, 6 clusters, τ addresses, FieldModulation
│   ├── src/mul.rs                        #     162L  SituationInput, MulAssessment, DK, Trust, Flow, Gate
│   ├── src/plan.rs                       #     154L  PlannerContract trait, PlanResult, QueryFeatures
│   ├── src/orchestration.rs              #     157L  OrchestrationBridge trait, StepDomain, UnifiedStep
│   ├── src/cam.rs                        #      79L  CamCodecContract, DistanceTableProvider, IvfContract
│   ├── src/jit.rs                        #      94L  JitCompiler, StyleRegistry, KernelHandle
│   └── src/nars.rs                       #      70L  InferenceType (5), QueryStrategy (5), SemiringChoice (5)
│   TOTAL: 1,076L — ZERO DEPENDENCIES
│
├── crates/neural-debug/                  # NEW: Static scanner + runtime registry
│   TOTAL: 850L
│
├── crates/bgz17/                         # EXCLUDED: Palette semiring codec (0 deps, 121 tests)
│   ├── src/palette_semiring.rs           #   Algebraic compose + distance
│   ├── src/palette_csr.rs                #   ArchetypeTree for O(k²) search
│   ├── src/container.rs                  #   728L  256-word BitVec pack/unpack
│   ├── src/base17.rs                     #   Golden-step VSA (xor_bind, bundle) on i16[17]
│   ├── src/rabitq_compat.rs              #   RaBitQ bridge
│   ├── src/clam_bridge.rs                #   777L  CLAM tree integration
│   └── src/simd.rs                       #   SIMD batch_palette_distance
│   TOTAL: ~3,500L
│
├── crates/lance-graph-codec-research/    # EXCLUDED: Research (FFT, bands, diamond, zeckbf17)
│   TOTAL: ~2,000L
│
├── crates/lance-graph-catalog/           # Unity Catalog + namespace + type mapping
├── crates/lance-graph-python/            # PyO3 bindings
└── crates/lance-graph-benches/           # Benchmarks
```

## 3. ndarray Module Map (Post-Rebase)

```
ndarray/src/hpc/                          # 55 HPC modules, 46,918 LOC, 880 tests
│
├── BLAS STACK:
│   ├── blas_level1.rs                    #   11 tests: dot, axpy, scal, nrm2, asum
│   ├── blas_level2.rs                    #   10 tests: gemv, ger, symv, trmv, trsv
│   ├── blas_level3.rs                    #    5 tests: gemm, syrk, trsm, symm
│   ├── quantized.rs                      #   BF16 GEMM, Int8 GEMM
│   └── lapack.rs                         #    4 tests: LU, Cholesky, QR
│
├── BACKENDS (mutually exclusive features):
│   ├── backend/native.rs                 #   Pure Rust SIMD microkernels (default)
│   ├── backend/mkl.rs                    #   Intel MKL FFI (-lmkl_rt): cblas_sgemm, cblas_dgemm, VML, DFTI
│   ├── backend/openblas.rs               #   OpenBLAS FFI
│   └── backend/kernels_avx512.rs         #   AVX-512 specialized kernels (11 types)
│
├── SIMD DISPATCH:
│   └── simd_caps.rs                      #   NEW: LazyLock<SimdCaps> singleton (detect once, ~1ns dispatch)
│
├── COGNITIVE CORE:
│   ├── fingerprint.rs                    #   12 tests: Fingerprint<N> const-generic [u64;N], XOR group
│   ├── plane.rs                          #   16 tests: i8[16384] accumulator, L1-resident
│   ├── node.rs                           #    9 tests: SPO cognitive atom (3 planes)
│   ├── seal.rs                           #    4 tests: Blake3 merkle integrity
│   ├── cascade.rs                        #   12 tests: 3-stroke HDR cascade (PackedDatabase, Welford drift)
│   ├── bf16_truth.rs                     #   23 tests: BF16 weights, awareness, PackedQualia
│   ├── causality.rs                      #   17 tests: CausalityDirection, NarsTruthValue
│   ├── blackboard.rs                     #   36 tests: Zero-copy typed-slot arena
│   ├── nars.rs                           #   NARS truth value operations
│   └── qualia.rs + qualia_gate.rs        #   16-channel phenomenal coloring + gate
│
├── CODEC / ENCODING:
│   ├── cam_pq.rs                         #   CAM-PQ: 6-byte FAISS PQ (HEEL/BRANCH/TWIG_A/TWIG_B/LEAF/GAMMA)
│   ├── bgz17_bridge.rs                   #   Base17 bridge: Fingerprint<256> → i16[17] (standalone port)
│   ├── zeck.rs                           #   ZeckF64: 8-byte progressive SPO edge encoding
│   ├── clam.rs                           #   46 tests: CLAM tree (divisive hierarchical, LFD)
│   ├── clam_search.rs + clam_compress.rs #   Search + compression on CLAM trees
│   ├── crystal_encoder.rs                #   Crystal encoding for embeddings
│   └── palette_codec.rs + palette_distance.rs  # Variable-width palette codec
│
├── GRAPH / ADJACENCY:
│   ├── graph.rs                          #   VerbCodebook: XOR binding edges (base_dim=4096)
│   ├── binding_matrix.rs                 #   3D XYZ binding popcount matrix (256³ = 16.7M points)
│   ├── spo_bundle.rs                     #   SPO bundle operations
│   ├── hdc.rs                            #   Hyperdimensional computing (rotate, bind, bundle)
│   ├── bnn.rs + bnn_causal_trajectory.rs #   26 tests: Binary neural networks
│   └── spatial_hash.rs                   #   KNN ring expansion
│
├── JITSON:
│   ├── jitson/ (8 modules)               #   JSON parser + validator + template + scan pipeline
│   └── jitson_cranelift/ (5 modules)     #   Cranelift JIT: engine, IR, detect, noise_jit (NEW: wired cache)
│
└── MISC (27+ modules):
    ├── arrow_bridge.rs                   #   26 tests: ThreePlaneFingerprintBuffer, GateState
    ├── fft.rs, vml.rs, statistics.rs     #   FFT, vectorized math, stats
    ├── activations.rs, kernels.rs        #   Neural activations, SIMD kernel dispatch
    ├── distance.rs                       #   L2, cosine, Hamming, Jaccard
    ├── deepnsm.rs, organic.rs, holo.rs   #   Deep NSM, organic, holographic
    ├── tekamolo.rs, vsa.rs               #   VSA operations
    └── surround_metadata.rs              #   Surround metadata encoding
```

## 4. Dependency Graph (As-Built)

```
┌─────────────────────────────────────────────────────────────────────┐
│                        Single Binary                                │
│                                                                     │
│  ┌──────────────┐     ┌─────────────────────┐     ┌──────────────┐ │
│  │  ladybug-rs   │────▶│      ndarray         │     │  crewai-rust │ │
│  │  (The Brain)  │     │  (The Foundation)    │     │ (The Agents) │ │
│  │  path dep ────┘     │  55 HPC modules      │     └──────┬───────┘ │
│  │  "stolen" parser    │  BLAS/MKL/OpenBLAS   │            │         │
│  └──────┬───────┘     │  Fingerprint<256>    │            │         │
│         │              └──────────────────────┘            │         │
│         │                                                   │         │
│         │  uses ndarray::hpc::clam::ClamTree               │         │
│         │  uses ndarray::hpc::kernels::SigmaGate           │         │
│         │                                                   │         │
│  ┌──────┴───────────────────────────────────────────────────┴───┐   │
│  │              lance-graph-contract (ZERO DEPS)                │   │
│  │  PlannerContract, OrchestrationBridge, CamCodecContract,     │   │
│  │  JitCompiler, ThinkingStyle(36), MulProvider                 │   │
│  └──────────────────────────┬──────────────────────────────────┘   │
│                              │ impl                                 │
│  ┌──────────────────────────┴──────────────────────────────────┐   │
│  │              lance-graph-planner (10,326L)                   │   │
│  │  MUL + Thinking + 16 Strategies + Elevation + CAM-PQ        │   │
│  └──────────────────────────┬──────────────────────────────────┘   │
│                              │ uses                                 │
│  ┌──────────────────────────┴──────────────────────────────────┐   │
│  │              lance-graph (CORE)                               │   │
│  │  Cypher parser, DataFusion, BlasGraph semirings, SPO, FalkorDB│  │
│  └──────────────────────────────────────────────────────────────┘   │
│                                                                     │
│  ┌──────────────┐     ┌──────────────┐     ┌──────────────────┐   │
│  │   n8n-rs      │     │   bgz17       │     │ codec-research   │   │
│  │ (Orchestrator)│     │ (EXCLUDED)    │     │ (EXCLUDED)       │   │
│  │ HTTP routers  │     │ 0 deps        │     │ FFT research     │   │
│  └──────────────┘     │ 121 tests     │     └──────────────────┘   │
│                        └──────────────┘                             │
└─────────────────────────────────────────────────────────────────────┘
```

### 4A. Missing Edges (Integration Debt)

| From | To | Status | What's needed |
|------|----|--------|--------------|
| lance-graph | ndarray | **NOT WIRED** | `ndarray_bridge.rs` is standalone mirror, not a dep |
| lance-graph-planner | ndarray | **NOT WIRED** | JitCompile strategy references ndarray but no Cargo dep |
| lance-graph-planner | lance-graph | **NOT WIRED** | Planner is self-contained, doesn't call parser/DataFusion |
| n8n-rs | lance-graph-contract | **NOT WIRED** | n8n-rs still has its own ThinkingMode, not contract's |
| crewai-rust | lance-graph-contract | **NOT WIRED** | crewai-rust still has its own 36 styles, not contract's |
| bgz17 | lance-graph workspace | **EXCLUDED** | Still in `exclude = []` in workspace Cargo.toml |
| ladybug-rs | lance-graph | **INDIRECT** | "Stolen" parser copy, no dep |

## 5. Cross-Reference: Other Docs in This Series

| Document | What it maps |
|----------|-------------|
| [TYPE_DUPLICATION_MAP.md](TYPE_DUPLICATION_MAP.md) | Every duplicated type across repos with file:line precision |
| [CODEC_COMPRESSION_ATLAS.md](CODEC_COMPRESSION_ATLAS.md) | Full→ZeckBF17→BGZ17→CAM-PQ→Scent compression chain |
| [THINKING_ORCHESTRATION_WIRING.md](THINKING_ORCHESTRATION_WIRING.md) | End-to-end thinking pipeline from YAML to native kernel |
| [SEMIRING_ALGEBRA_SURFACE.md](SEMIRING_ALGEBRA_SURFACE.md) | All semirings across all repos with their algebra |
| [ADJACENCY_SYNERGY_MAP.md](ADJACENCY_SYNERGY_MAP.md) | CSR models, BLAS×PQ×BGZ17, KuzuDB synergies |
| [METADATA_SCHEMA_INVENTORY.md](METADATA_SCHEMA_INVENTORY.md) | 4096 surface, 16kbit fingerprints, Lance persistence, Arrow bridge |
| [INTEGRATION_DEBT_AND_PATHS.md](INTEGRATION_DEBT_AND_PATHS.md) | Weaknesses, strengths, epiphanies, strategic paths forward |
