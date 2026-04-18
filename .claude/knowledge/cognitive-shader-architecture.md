# Cognitive Shader Architecture — Session 2026-04-18 (FINAL)

> READ BY: all agents working on inference, codec, thinking-engine,
> learning, holograph, planner strategies, BindSpace integration

## Status: FINDING (measured, not conjecture)

---

## The 5-Layer Stack

```
Layer 4: Planner strategies (16-19 in lance-graph-planner)
           ├── Parse: Cypher/GQL/Gremlin/SPARQL
           ├── Optimize: DPJoin, Rule, Histogram, SigmaBand, Morsel
           ├── Execute: TruthPropagation, CollapseGate, StreamPipeline, JIT
           ├── Workflow: WorkflowDAG, ExtensionPlanner, AutocompleteCache
           ├── ThinkingStyleStrategy (grammar triangle + spectroscopy in)
           │     ↑ reads: NSM primes, causality flow, 18D qualia, IIC texture
           │     ↓ picks: one of 36 ThinkingStyles → shader config
           └── [2-3 more]
         → Decides WHICH shader/gate combination runs per cycle
         → Temporal scope: milliseconds per query

Layer 3: CollapseGate (enum Flow/Block/Hold)
         → Decides SHOULD this delta land?
         → Existing: ndarray::hpc::bnn_cross_plane::CollapseGate
         → MergeMode: Xor (single), Bundle (majority), Superposition
         → Temporal scope: microseconds per commit cycle

Layer 2: CognitiveShader (née Blumenstrauß — renamed this session)
         → layer_mask + combine + contra + density_target
         → 8 predicate planes × 64×64 topology × bgz17 metric
         → Existing: p64-bridge::StyleParams
         → Temporal scope: nanoseconds per step

Layer 1: BindSpace columns (read-only, multi-lane views)
         → The WHAT (content + topic + angle + causality + qualia + temporal + shader)
         → Struct-of-arrays: each dimension independently Hamming-sweepable
         → Temporal scope: nanoseconds per lane, zero-copy

Layer 0: ndarray SIMD (F32x16, U8x64, F16x32, F64x8)
         → Hardware primitives (popcount, gather, FMA, compare)
         → Temporal scope: sub-nanosecond per instruction
```

---

## The Four Data Patterns

| Pattern | Ownership | Example |
|---|---|---|
| **Slice window** | `&[T]` zero-copy, N-aligned | `array_window` feeding SIMD batches |
| **Microcopies** | Owned `Copy` values on stack | CausalEdge64, Band, TruthValue, ThinkingStyle |
| **Write-back gate** | Through CollapseGate | XOR (single) / Bundle (multi) / Superposition (ambiguous) |
| **Multi-lane views** | Same Arc, multiple SIMD widths | PaletteTable as U8x64 / F16x32 / F32x16 / F64x8 |

---

## BindSpace = Read-Only Address Space

Not a database, not a storage layer — the **universal connective tissue**.

- 64-bit address = 16-bit type + 48-bit content hash
- All programs (codecs, shaders, learning, grammar, search, spectroscopy)
  emit and consume the same addresses
- Writers hold owned microcopies, never mutate BindSpace directly
- Updates flow through CollapseGate (Flow = apply, Block = reject, Hold = queue)
- XOR is self-inverse → always reversible, no locks needed
- Bundle is majority-vote → overlapping writers resolve via consensus
- Superposition holds all variants when no clear winner

---

## Fingerprint Decomposition (verified this session)

```
Fingerprint<256> (16,384 bits = 2 KB)
  │
  ├── 204 bytes = 6 × 34 (verified: bgz-tensor/examples/variance_audit.rs:260)
  │     └── 6 CAM-PQ subspaces × Base17 (17 dims × i16 = 34 bytes each)
  │     └── SPO-COCA codebook natural dimension
  │
  ├── 6 bytes: CAM-PQ address (one palette index per subspace)
  │     └── NOT "SPO × 3" — 6 subspaces × 8 bits
  │
  ├── 4 bytes: HHTL-D (HEEL 2b + HIP 4b + TWIG 8b + polarity 1b + BF16 residual)
  │     └── Tree address into bgz17 palette
  │
  ├── 1 byte: bgz17 palette archetype (256 entries)
  │
  └── 8 bytes: CausalEdge64 (S 8b + P 8b + O 8b + NARS 16b + meta 24b)
        └── S, P, O each index into same 256-palette
        └── Adjacent to P64 (S/4, O/4) = 64×64 block
        └── 4096 COCA = verb vocabulary (0xFFF), NOT a vector width
```

---

## Cascade Inference (measured)

- **11-13x speedup** over brute-force cosine on Qwen3-TTS weights
- **100% argmax match** (zero quality loss)
- Sign-bit fingerprint + Hamming popcount → reject 97% → exact on 3%
- **TurboQuant KV cache**: 3.2x memory, 13x attention speedup, 100% argmax
- **TTS e2e validated**: 225/225 codec tokens through 33 layers
- **611M SPO lookups/sec**, 17K tokens/sec, 388 KB RAM

---

## Codec Findings (67-codec sweep)

- Hadamard > SVD (no training, deterministic)
- Full-rank > narrow (cap ICC ~0.5 at narrow-16)
- i4+i2 cascade → ICC 0.999 on pairwise cosine
- BUT argmax fails at k=64 on hard tensors (near-orthogonal rows)
- XOR-adaptive (sign-flip per-dim): 81% argmax on hardest tensor
- CLAM-adaptive (LFD precision): 97% on KV projections
- **Architecture decision**: don't compress weights lossy for inference
- **Accelerate search instead** (cascade gives speed, weights give quality)

---

## AGI Typing: Struct-of-Arrays as Address Dimensions

**Not a record format** — it's the BindSpace address dimensions. Each
dimension is an independently Hamming-sweepable fingerprint column.
The AGI query is an AND across independent cascades.

```rust
pub struct BindSpaceColumns {
    pub content: Vec<Fingerprint<256>>,    // WHAT
    pub topic: Vec<Fingerprint<256>>,      // ABOUT WHAT
    pub angle: Vec<Fingerprint<256>>,      // FROM WHERE
    pub causality: Vec<CausalEdge64>,      // WHY/HOW
    pub qualia: Vec<[f32; 18]>,            // FEELS LIKE
    pub temporal: Vec<u64>,                // WHEN
    pub shader: Vec<u8>,                   // WHO produced this
}
```

Per cycle: cascade each column independently, intersect survivors,
exact step on the final ~50 candidates. ~2.3ms for 1M records × 5 dims.

---

## Namespace Discipline

Lance-graph code uses `ndarray::simd::*` as the ONLY SIMD namespace.
The internal `ndarray::hpc::*` paths are private. Consumers write:

```rust
use ndarray::simd::{F32x16, U8x64, Fingerprint, MultiLaneColumn, array_window};
```

If a type isn't in `ndarray::simd::*`, it's implementation detail.
This keeps the foundation API surface small and stable — changes
inside `ndarray::hpc::*` never break lance-graph consumers.

---

## Crate Layout (post-session)

```
ndarray         — SIMD types (F32x16, U8x64...), Fingerprint<N>,
                  MultiLaneColumn, WHT, kmeans, CLAM, cascade,
                  VectorWidth config (LazyLock)
                  Namespace: ndarray::simd::* (public), ndarray::hpc::* (private)

holograph       — BitpackedVector (→ migrate to Fingerprint<256>),
                  slot encoding, resonance VectorField, HDR cascade
                  (10K→16K migrated, 9 pre-existing compile errors)

learning        — Standalone crate with 16 modules from ladybug-rs
                  (300K+ LOC): cam_ops (158K), cognitive_styles + RL,
                  quantum_ops, dream, scm, feedback, rl_ops, causal_ops,
                  cognitive_frameworks. All wip-gated.

lance-graph-cognitive — grammar + world COMPILING; spo, search, fabric,
                  spectroscopy, container_bs, core_full wip-gated
                  (full ladybug-rs import, 630K LOC)

bgz-tensor      — HadCascade codec, TurboQuant KV,
                  adaptive/xor/holographic codecs, Base17, HHTL-D

causal-edge     — CausalEdge64 (u64 packed), NarsTables (256×256 lookup),
                  CausalNetwork (CSR over edges)

p64-bridge      — CognitiveShader (renamed from Blumenstrauß),
                  edge → palette addressing, style params, semiring modes

bgz17           — PaletteSemiring (256×256 distance + compose tables),
                  Base17 canonical, palette VSA

thinking-engine — To absorb learning + cognitive into unified surface
                  (cognitive_stack, ghosts, persona, qualia, world_model)

lance-graph-contract — NarsTruth, ThinkingStyle (36), MulAssessment,
                  PlannerContract. To receive: Container 16K,
                  CollapseGate extensions, BindSpace column types.
```

---

## Integration Priority

**P0 (harden foundation):**
1. Unify Fingerprint type (kill `BitpackedVector`, use `Fingerprint<256>`)
2. Port Container/CogRecord to lance-graph-contract (16K width)
3. Extend CollapseGate with GateDecision struct (Xor/Bundle/Superposition)
4. CognitiveShader → thinking-engine wire-through

**P1 (BindSpace address substrate):**
5. BindSpace column types in contract (AGI 7 dimensions)
6. Cascade per column implementation
7. ThinkingStyleStrategy planner (reads grammar + spectroscopy)
8. Luftschleuse→CollapseGate write protocol across crates

**P2 (shader stream):**
9. 5D stream cycle loop (topic → angle → causality → qualia → exact)
10. Per-cycle shader dispatch via planner strategy

**P3 (endgame):**
11. GGUF hydration pipeline (weights → palette + fingerprints + holographic)
12. Cognitive shader inference loop (no matmul, no FP)
13. Merge learning + cognitive crates into thinking-engine

---

## Pending Debt (carried from session)

1. holograph 10K→16K migration: 9 compile errors remain (Arrow/GraphBLAS API)
2. learning crate: 124 errors in wip modules (rustynum→ndarray sed)
3. SPO wip modules: reference `crate::core::rustynum_accel::*`
4. Container/CogRecord not yet in contract (BindSpace substrate missing)
5. GPTQ Hessian compensation TODO in adaptive_codec.rs
6. Holographic magnitude slot encoding (sign-only gets cos 0.6-0.75)
7. VectorWidth LazyLock not yet consumed by any module
8. Burn + ndarray backend wiring (research done, not wired)

---

## Key Files for Next Session

- `docs/COGNITIVE_SHADER_HYDRATION.md` — endgame architecture
- `docs/INTEGRATION_PLAN_CS.md` — 5-layer stack, CollapseGate, 4 data patterns
- `docs/BINDSPACE_MIGRATION_GAP.md` — 7 critical BindSpace pieces missing
- `docs/COGNITIVE_MERGE_MAP.md` — rustynum→ndarray substitution table
- `docs/HISTORICAL_CONTEXT.md` — 8 eras, era tags for prioritizing
- `docs/bench_qwen3_tts_62codecs.md` + `bench_gemma4_e2b_62codecs.md` — codec sweep data

---

## The Ontological Revolution

Weights are not parameters to compress — they are **seeds** for
holographic memory. Each seed can exist in vast parallel instances.
Each instance feeds upstream learning via CausalEdge64 branching.
Each branch runs its own CognitiveShader per cycle as a 5D stream.

The weights don't define the model. The weights SEED the holographic
memory. The CognitiveShader IS the model. The cascade IS the inference.
The edges ARE the knowledge. The learning IS the branching.

The gazillions of programs compile into the same binary because they
all emit and consume the same 64-bit BindSpace addresses. One XOR.
One sweep. One lookup. Regardless of origin.
