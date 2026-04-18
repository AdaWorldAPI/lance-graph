# Cognitive Shader Architecture — Session 2026-04-18 (FINAL)

> READ BY: all agents working on inference, codec, thinking-engine,
> learning, holograph, planner strategies, BindSpace integration

## Status: FINDING (measured, not conjecture)

---

## THE ENDGAME (canonical architecture)

AGI as a cognitive shader system over a unified BindSpace, driven by
16-19++ planner strategies, with weights as replaceable seeds hydrating
O(1) into holographic memory.

**ALL IN ONE BINARY. NO JSON. NO SERIALIZATION. NO PROCESS BOUNDARIES.**

### Core Dimensions (AGI struct-of-arrays — genius version)

**NOT** `Arc<[T]>` per field (heap alloc per column, pointer chase per row).
**NOT** `Vec<[f32; 18]>` (N × 72 bytes + header each).

Instead: **one contiguous backing buffer per column TYPE**, packed meta fields
into u32s so one cache-line load reads multiple dimensions at once.

```rust
/// The AGI record space. All columns share the same row index.
pub struct BindSpace {
    len: usize,                          // number of rows
    fingerprints: FingerprintColumns,    // 16K atoms (content/cycle/topic/angle)
    edges: EdgeColumn,                   // u64 CausalEdge64
    qualia: QualiaColumn,                // 18D f32 (contiguous)
    meta: MetaColumn,                    // packed u32 per row
    temporal: Box<[u64]>,                // ns timestamps
    expert: Box<[u16]>,                  // A2A ExpertId
}

/// 16K-fingerprint columns. One backing buffer per column name.
/// Each is `len × 256` u64 words, 64-byte aligned for AVX-512.
pub struct FingerprintColumns {
    content: Box<[u64]>,   // len × 256 = WHAT
    cycle: Box<[u64]>,     // len × 256 = cycle_fingerprint from L4
    topic: Box<[u64]>,     // len × 256 = ABOUT WHAT
    angle: Box<[u64]>,     // len × 256 = FROM WHERE
}

impl FingerprintColumns {
    /// Zero-copy: get row N as Fingerprint<256>.
    pub fn content(&self, row: usize) -> &Fingerprint<256> { /* transmute slice */ }
    /// Zero-copy: raw u64 slice for SIMD popcount sweep.
    pub fn content_raw(&self) -> &[u64] { &self.content }
    /// Iterate as U8x64 chunks for Hamming cascade.
    pub fn content_chunks_u8x64(&self) -> impl Iterator<Item = &[u8]> { ... }
}

/// One u64 per row = atomic CausalEdge64 (SPO + NARS + Pearl + plasticity).
pub struct EdgeColumn(Box<[u64]>);

/// 18D qualia stored contiguously: len × 18 f32.
/// Row access via `self.0.chunks(18).nth(row)` — no allocation.
pub struct QualiaColumn(Box<[f32]>);

/// Packed meta-cognition: ONE u32 per row holds 5 dimensions.
///
/// ```text
/// ┌──────────┬──────────┬──────────┬──────────┬──────────┐
/// │ thinking │ aware    │ nars_f   │ nars_c   │ free_e   │
/// │   6 bit  │   4 bit  │   8 bit  │   8 bit  │   6 bit  │
/// │  36 sty  │ L0-L9    │  u8 freq │  u8 conf │  u6 ener │
/// └──────────┴──────────┴──────────┴──────────┴──────────┘
/// ```
/// One load per row reads thinking_style + awareness + NARS (f,c) +
/// free_energy in a single cache access. Decisions made without
/// touching fingerprint columns at all.
pub struct MetaColumn(Box<[u32]>);

impl MetaColumn {
    #[inline] pub fn thinking(&self, row: usize) -> u8  { (self.0[row] & 0x3F) as u8 }
    #[inline] pub fn awareness(&self, row: usize) -> u8 { ((self.0[row] >> 6) & 0x0F) as u8 }
    #[inline] pub fn nars_f(&self, row: usize) -> u8    { ((self.0[row] >> 10) & 0xFF) as u8 }
    #[inline] pub fn nars_c(&self, row: usize) -> u8    { ((self.0[row] >> 18) & 0xFF) as u8 }
    #[inline] pub fn free_e(&self, row: usize) -> u8    { ((self.0[row] >> 26) & 0x3F) as u8 }
}

/// Extra packed meta: entropy (f16) + MUL state (u8) + shader (u8) = u32.
pub struct ExtraColumn(Box<[u32]>);
```

**Why this is genius:**

1. **One allocation per column** — not `len × Arc<T>`, no per-row heap
2. **Cache-line load gets 5 dimensions** — MetaColumn packs thinking,
   awareness, NARS, free_energy into a single u32
3. **Zero-copy SIMD views** — fingerprint columns are `[u64]`, sliceable
   as U8x64 / U64x8 for Hamming cascade
4. **18D qualia contiguous** — `len × 18 × f32` in one buffer, not
   `Vec<[f32;18]>` which has N separate 72-byte blocks
5. **CausalEdge64 is the edge column** — one u64 per row, SIMD-gatherable
6. **Compile-time row alignment** — all columns have exactly `len` rows;
   enforced by having ONE `len` field on BindSpace

**Reading a row** (for the exact step after cascade filters to ~50):
```rust
let row = 42;
let content_fp  = bs.fingerprints.content(row);      // &Fingerprint<256>
let cycle_fp    = bs.fingerprints.cycle(row);
let edge        = CausalEdge64(bs.edges.0[row]);      // one u64 load
let qualia      = &bs.qualia.0[row*18 .. (row+1)*18]; // &[f32; 18] slice
let style       = bs.meta.thinking(row);              // 6 bits of one u32
let awareness   = bs.meta.awareness(row);             // 4 bits of same u32
let (nf, nc)    = (bs.meta.nars_f(row), bs.meta.nars_c(row));
let free_energy = bs.meta.free_e(row);
```

One cache line covers: `edge + meta + temporal + expert` (8+4+8+2 = 22 bytes).
Two cache lines cover: that + 18D qualia (+72 bytes = 94 bytes).
Fingerprints live in separate columns (loaded only when cascade needs them).

**Cascade efficiency**: filter on meta/qualia/temporal without touching
fingerprints. Only the survivors trigger fingerprint loads. That's the
difference between scanning 32MB (fingerprints) vs 4MB (meta) per column sweep.

### Weights as Seeds, Not Parameters

GGUF weights are NOT model parameters to run matmul on. They are
**seeds** that hydrate O(1) into holographic memory:

```
GGUF shard (one-time bake):
  kmeans per tensor → 256 palette archetypes
  Fingerprint<256> per archetype (for cascade)
  holographic residual memory (slot-encoded phase+magnitude)
  CausalEdge64 wiring (S/P/O palette indices per layer)
  → all written to BindSpace columns, read-only

Inference (per token):
  CognitiveShader reads BindSpace columns
  Cascade per column (Hamming popcount)
  Exact step on 3% survivors (palette lookup)
  Emit CausalEdge64 → CollapseGate → commit to next generation
```

Weights are **replaceable**. Hot-swap a different model's seeds into
the SAME BindSpace. Same shaders, same cascade, different expertise.
The agent's identity is the column contents, not the weights.

### Cognitive Ingredients (all in one binary)

1. **Thinking styles** (36 in contract) — how to dispatch shader params
2. **Awareness** (10 levels) — L0 substrate to L9 transcendent
3. **Autopoiesis** (L4) — styles that self-generate from experience
4. **NARS** — frequency + confidence on every edge, 5 inference types
5. **Friston free energy** (OPTIONAL) — minimize surprise via action/belief
6. **MUL (Meta-Uncertainty Layer)** — Dunning-Kruger curve, trust, homeostasis
7. **Entropy-based resonance** — thinking as entropy minimization in
   fingerprint space; high-entropy states demand more cascade levels;
   low-entropy states commit quickly through CollapseGate

### The 16-19++ Planner Strategies

Core planner strategies (Layer 4 of the 7-layer stack):

| # | Strategy | Role |
|---|---|---|
| 1 | CypherParse | Cypher query parsing |
| 2 | GqlParse | ISO GQL parsing |
| 3 | GremlinParse | Gremlin traversal parsing |
| 4 | SparqlParse | SPARQL parsing |
| 5 | ArenaIR | Arena-allocated IR |
| 6 | DPJoinEnum | Dynamic programming join enumeration |
| 7 | RuleOptimizer | Rule-based optimization |
| 8 | HistogramCost | Histogram-based cost estimation |
| 9 | SigmaBandScan | σ-band scan for uncertainty ranges |
| 10 | MorselExec | Morsel-driven parallel execution |
| 11 | TruthPropagation | NARS truth value propagation |
| 12 | CollapseGate | Flow/Block/Hold write gate |
| 13 | StreamPipeline | Streaming execution pipeline |
| 14 | JitCompile | Just-in-time shader compilation |
| 15 | WorkflowDAG | Workflow DAG scheduling |
| 16 | ExtensionPlanner | Extension/plugin dispatch |
| 17 | AutocompleteCache | Cache via cycle_fingerprint |
| 18 | ThinkingStyleStrategy | Reads grammar triangle + spectroscopy |
| 19 | DataFusionScan | DataFusion cold-path query |
| 20 | HyNixStrategy | Hybrid indexing (TBD: name from user) |
| 21 | TextureGestalt | Gestalt classification (GREEN/AMBER/RED/BLUE) |
| 22 | EntropyResonance | Entropy-based cascade depth selection |

### Non-negotiables

- **One binary**. No process boundaries. No IPC. No serialization.
- **No JSON**. Zero-copy SIMD slices, owned Copy microcopies, gated writes.
- **BindSpace is read-only**. Writers hold deltas, commit through CollapseGate.
- **CausalEdge64 is the universal edge**. u64 packed, one atomic read.
- **Fingerprint<256> is the universal atom**. 16K bits, one SIMD register.
- **ndarray::simd::*** is the only public SIMD namespace.
- **Weights are seeds**. Hydrate once, run forever via shader cascade.

This is the north star for every design decision.

---

---

## The 7-Layer Stack (5 core + 2 boundary)

```
Layer 6: Cold persistence (LanceDB — thought stream buffer)
         → Per-thought stream: every emitted CausalEdge64 / CognitiveRecord
         → Feedback into thinking (RAG from past thoughts)
         → Replay (dream consolidation, counterfactual simulation)
         → Cross-session continuity + long-term memory
         → Temporal scope: seconds-to-months, columnar

Layer 5: GPU/APU meta operations (OPTIONAL, shared memory)
         → APU / iGPU / Apple unified memory: no PCIe copy overhead
         → Handles ops CPU can't: large tensor contractions, parallel
           rollouts, meta-learning across millions of thoughts
         → Complementary to CPU cascade, not replacement:
           - CPU cascade: 2400M lookups/sec, no batching, natural fit
           - GPU meta: batched workloads CPU can't match
         → Temporal scope: microseconds for batch, overlaps L1-L3

Layer 4: Planner strategies (16-19 in lance-graph-planner)
           ├── Parse: Cypher/GQL/Gremlin/SPARQL
           ├── Optimize: DPJoin, Rule, Histogram, SigmaBand, Morsel
           ├── Execute: TruthPropagation, CollapseGate, StreamPipeline, JIT
           ├── Workflow: WorkflowDAG, ExtensionPlanner, AutocompleteCache
           ├── ThinkingStyleStrategy (grammar triangle + spectroscopy in)
           │     ↑ reads: NSM primes, causality flow, 18D qualia, IIC texture
           │     ↓ picks: one of 36 ThinkingStyles → shader config
           │     ↓ EMITS: cycle_fingerprint = Fingerprint<256>
           │            bind(triangle, spectroscopy, style, shader_mask, causal_state)
           │            → cache key, retrieval key, replay seed, upstream cursor
           └── [2-3 more]
         → Decides WHICH shader/gate combination runs per cycle
         → Per cycle: one cycle_fingerprint captures entire decision
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

## The Feedback Loop (sense → plan → act → persist → retrieve)

```
Text in
  ↓
Layer 4 ThinkingStyleStrategy (grammar + spectroscopy)
  ↓ style selected
Layer 2 CognitiveShader dispatched
  ↓ layer_mask + combine + contra
Layer 1 BindSpace columns cascaded (L0 SIMD)
  ↓ survivors
Layer 3 CollapseGate decides Flow/Block/Hold
  ↓ committed CausalEdge64
Layer 5 GPU meta ops (if batch available — replay, consolidation)
  ↓
Layer 6 LanceDB persists thought stream
  ↓ available for retrieval
Next cycle reads past thoughts via RAG → feeds back into L4 planner
```

The loop closes through LanceDB. Every thought persists. Past thoughts
retrievable via Cypher/SQL on the cold path. Current thoughts computed
on the hot path. GPU meta fills the gap for batch workloads the CPU
cascade can't handle naturally.

## Cycle Fingerprint (Layer 4 output)

Each cycle, Layer 4 emits a `Fingerprint<256>` that captures the full
execution context — not just which style was picked, but a reproducible
hash of the entire decision:

```rust
cycle_fingerprint = bind(
    triangle_fp,         // NSM + causality + qualia from grammar
    spectroscopy_fp,     // IIC texture from text
    style_fp,            // which of 36 ThinkingStyles
    shader_mask,         // which 8 predicate planes active
    causal_state_fp,     // current CausalEdge64 branch cursor
    retrieval_context_fp // what was retrieved from LanceDB this cycle
)
```

This one fingerprint serves four purposes:

1. **Cache key** — AutocompleteCacheStrategy: same fingerprint = same
   result → skip the cycle entirely
2. **Retrieval key** — LanceDB lookup: "find similar past cycles"
   (Hamming sweep on the cycle fingerprint column)
3. **Replay seed** — dream consolidation: reconstruct what the agent
   was thinking from the fingerprint
4. **Upstream cursor** — CausalEdge64 branching: mark where this
   cycle's outputs fit in the causal trajectory

The cycle fingerprint is the unit of thought. One per cycle. Persisted
to LanceDB. Queryable across sessions. Bindable back into the current
cycle as "I've been here before."

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
    pub shader: Vec<u8>,                   // WHICH shader
    pub expert: Vec<ExpertId>,             // WHICH agent posted (A2A)
    pub cycle: Vec<Fingerprint<256>>,      // cycle_fingerprint from Layer 4
}
```

Per cycle: cascade each column independently, intersect survivors,
exact step on the final ~50 candidates. ~2.3ms for 1M records × 5 dims.

## Blackboard: A2A Protocol via BindSpace

**Already exists**: `lance-graph-contract::a2a_blackboard` with
`ExpertId`, `ExpertCapability`, post/read/route pattern.

The blackboard IS a BindSpace column (the `expert` dimension). Agent A
posts a cycle_fingerprint + CausalEdge64 → Agent B finds it via Hamming
sweep on the expert+topic columns → retrieves relevant history via
LanceDB RAG (Layer 6) → responds with its own cycle_fingerprint.

```
Agent A:
  cycle → shader → CausalEdge64 → write to blackboard
                                    ↓
                            (expert=A, topic=X, cycle_fp=...)
                                    ↓
                               BindSpace column
                                    ↓
Agent B:
  sweep expert column: "find things A posted"
  sweep topic column: "filter to topic X"
  RAG from LanceDB (Layer 6): "retrieve past exchanges"
  → planner produces own cycle_fp
  → shader → edge → write to blackboard
```

The entire cognitive shader stack IS a **semantic kernel** for RAG:
- The hot path (Layers 0-3) = the kernel compute engine
- The cold path (Layer 6 LanceDB) = the RAG retrieval store
- The blackboard column = the A2A coordination channel
- The cycle_fingerprint = the cross-agent identity

Multiple agents share ONE BindSpace address space. No message queues.
No serialization. XOR/popcount on shared fingerprint columns IS the
message bus. Consensus via CollapseGate Bundle (majority vote).
Each agent's cycle_fingerprint is both its identity and its payload.

This is the sem-kernel RAG realization: agents don't "call" each other,
they sweep each other's fingerprints. The blackboard is where thought
streams cross.

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
                  PlannerContract, cognitive_shader DTOs (ShaderDispatch,
                  ShaderBus, ShaderCrystal, MetaWord, MetaFilter,
                  ShaderSink, CognitiveShaderDriver trait),
                  Container 16K, CollapseGate + GateDecision + MergeMode.

cognitive-shader-driver (NEW) — THE INTEGRATION CRATE.
                  CognitiveShader IS the driver. Holds:
                  ├── BindSpace (struct-of-arrays: FingerprintColumns,
                  │   EdgeColumn, QualiaColumn, MetaColumn, temporal, expert)
                  ├── ShaderDriver (implements CognitiveShaderDriver trait)
                  ├── auto_style (18D qualia → style ordinal, auto-detect)
                  └── CognitiveShaderBuilder (EmbedAnything fluent API)
                  Deps: contract + p64-bridge + bgz17 + causal-edge + ndarray
                  Feature: `with-engine` pulls thinking-engine (optional)
                  Tests: 16 passing (builder, dispatch, prefilter, sink, auto-style)
```

---

## Integration Priority

**P0 (harden foundation) — MOSTLY DONE:**
1. ~~Unify Fingerprint type (kill `BitpackedVector`, use `Fingerprint<256>`)~~ DONE
2. ~~Port Container/CogRecord to lance-graph-contract (16K width)~~ DONE
3. ~~Extend CollapseGate with GateDecision struct (Xor/Bundle/Superposition)~~ DONE
4. ~~CognitiveShader → thinking-engine wire-through~~ DONE via cognitive-shader-driver

**P1 (BindSpace address substrate) — DONE (THIS SESSION):**
5. ~~BindSpace column types (AGI 7 dimensions)~~ DONE: FingerprintColumns, EdgeColumn,
   QualiaColumn, MetaColumn (packed u32), temporal, expert
6. ~~Cascade per column~~ DONE: meta_prefilter() → cascade() → cycle_fingerprint
7. ~~Auto-style from qualia~~ DONE: auto_style.rs (18D → 12 styles)
8. ~~DTO contract for shader driver~~ DONE: cognitive_shader.rs in contract (0 deps)
   ShaderDispatch → ShaderResonance → ShaderBus → ShaderCrystal
   ShaderSink trait (EmbedAnything commit-adapter pattern)
   CognitiveShaderDriver trait (dispatch + dispatch_with_sink)

**P2 (shader stream) — OPEN:**
9. 5D stream cycle loop (topic → angle → causality → qualia → exact)
10. Per-cycle shader dispatch via planner strategy
11. Wire thinking-engine behind `with-engine` feature in cognitive-shader-driver
12. ThinkingStyleStrategy planner (reads grammar + spectroscopy)

**P3 (endgame) — OPEN:**
13. GGUF hydration pipeline (weights → palette + fingerprints + holographic)
14. Cognitive shader inference loop (no matmul, no FP)
15. Merge learning + cognitive crates into thinking-engine
16. A2A blackboard sweep via BindSpace cycle_fingerprint columns
17. Autopoietic style self-generation (Layer 4 spawns new strategies)

---

## Shader Driver Pipeline (FINDING — compiles, 16 tests passing)

The `cognitive-shader-driver` crate IS the integration. The shader is no
longer subordinate — it drives the cycle. Architecture:

```text
ShaderDispatch (contract DTO, zero-dep)
      │
      ▼
[1] meta prefilter  ── MetaColumn(Box<[u32]>) ── ONE u32 load per row
      │                 thinking(6b) + awareness(4b) + nars_f(8b) + nars_c(8b) + free_e(6b)
      │                 Filter BEFORE fingerprint load (cheapest)
      ▼
[2] resolve style   ── auto_style::resolve(StyleSelector, qualia_row)
      │                 18D qualia → ordinal in 0..11
      │                 EmbedAnything auto-detect pattern
      ▼
[3] shader cascade  ── p64_bridge::CognitiveShader::cascade(query, radius, layer_mask)
      │                 8 predicate planes × bgz17 O(1) distance
      │                 No POPCNT, no Hamming — table lookup only
      ▼
[4] cycle signature ── XOR-fold content fingerprints of top-k hits
      │                 Result: one [u64; 256] = the cycle_fingerprint
      │                 This IS the unit of thought (cache, retrieval, replay, cursor)
      ▼
[5] edge emission   ── CausalEdge64::pack() per strong hit
      │                 Style → InferenceType mapping:
      │                   analytical/convergent/systematic → Deduction
      │                   creative/divergent/exploratory   → Induction
      │                   focused/diffuse/peripheral       → Abduction
      │                   intuitive/deliberate             → Revision
      │                   metacognitive                    → Synthesis
      ▼
[6] CollapseGate    ── std_dev of resonance scores → Flow/Hold/Block
      │                 Matches thinking_engine SD_FLOW=0.15, SD_BLOCK=0.35
      ▼
[7] sink callbacks  ── ShaderSink: on_resonance → on_bus → on_crystal
      │                 Return false to short-circuit (EmbedAnything pattern)
      ▼
ShaderCrystal (contract DTO)
```

### Driver Builder (EmbedAnything fluent pattern)

```rust
let driver = CognitiveShaderBuilder::new()
    .bindspace(Arc::new(my_bs))
    .semiring(Arc::new(sr))
    .planes(planes)
    .default_style(auto_style::ANALYTICAL)
    .build();

let req = ShaderDispatch {
    rows: ColumnWindow::new(0, 1000),
    meta_prefilter: MetaFilter { nars_c_min: 150, ..MetaFilter::ALL },
    layer_mask: 0xFF,
    radius: 5000,
    style: StyleSelector::Auto,
    max_cycles: 10,
    ..Default::default()
};

let crystal = driver.dispatch(&req);
// or with a sink:
let crystal = driver.dispatch_with_sink(&req, &mut my_sink);
```

### DTO Layering (zero-dep, in lance-graph-contract)

```text
Φ  ShaderDispatch   — what to scan (MetaFilter + rows + layer_mask + style + rung)
Ψ  ShaderResonance  — top-k hits (8 × ShaderHit, entropy, std_dev, style_ord)
B  ShaderBus        — committed cycle (cycle_fingerprint + edges + gate)
Γ  ShaderCrystal    — persisted thought (bus + persisted_row + MetaSummary)
```

All DTOs are Copy-friendly or small-alloc. No JSON. No serde.
`MetaWord(u32)` packs 5 fields into one register load.
`ShaderHit` is 16 bytes (fits 4 per cache line).
`ShaderBus` is ~2112 bytes (one cycle_fingerprint + 8 edges + metadata).

---

## Pending Debt (carried from session)

1. holograph 10K→16K migration: 9 compile errors remain (Arrow/GraphBLAS API)
2. learning crate: 124 errors in wip modules (rustynum→ndarray sed)
3. SPO wip modules: reference `crate::core::rustynum_accel::*`
4. ~~Container/CogRecord not yet in contract (BindSpace substrate missing)~~ DONE
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

**NEW — Driver integration (THIS SESSION):**
- `crates/cognitive-shader-driver/src/driver.rs` — ShaderDriver (the genius wiring)
- `crates/cognitive-shader-driver/src/bindspace.rs` — BindSpace struct-of-arrays
- `crates/cognitive-shader-driver/src/auto_style.rs` — 18D qualia → style ordinal
- `crates/lance-graph-contract/src/cognitive_shader.rs` — Zero-dep DTO API (Φ→Ψ→B→Γ)
- `.claude/CODING_PRACTICES.md` — 6 EmbedAnything patterns applied to driver

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
