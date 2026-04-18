# Historical Context: From Ladybug to Cognitive Shaders

> A chronicle of architectural evolution across 6 months, 5 repos, ~1M LOC.
> Written 2026-04-18 with full 1M context window — first time both codebases
> are visible simultaneously.

---

## Era 1: Ladybug-rs Pure (Oct–Dec 2025)

**The 10 Stages of Awareness**

Ladybug-rs started as a cognitive substrate with 10 hierarchical layers:

```
L0: Substrate     — raw fingerprints, Hamming distance
L1: Felt Core     — valence/activation (basic qualia)
L2: Body Schema   — spatial grounding, embodiment
L3: Proto-Self    — identity boundary, self/other distinction
L4: Autopoiesis   — SELF-GENERATING thinking styles
L5: Narrative      — temporal coherence, episodic memory
L6: Meta-Cognition — thinking about thinking (MUL)
L7: Theory of Mind — modeling others' beliefs
L8: Ethical Self   — moral reasoning, value alignment
L9: Transcendent   — cross-domain transfer, emergence
```

**Layer 4 was the breakthrough**: autopoiesis of thinking styles. Not fixed
styles selected from a menu — styles that GENERATE THEMSELVES from experience.
15 base styles + RL adaptation = styles evolve per interaction.

**Vector width: 10,000 bits** (157 × u64). Chosen for σ ≈ 56 (good signal/noise
ratio), but awkward: 48 bits of padding, partial last word, non-power-of-2.

Core architecture:
- `core/fingerprint.rs` — 10K bit vector, XOR/bind/bundle
- `core/vsa.rs` — Vector Symbolic Architecture (bind, unbind, bundle, permute)
- `core/scent.rs` — 5-byte hierarchical filter (petabyte-scale rejection)
- `core/index.rs` — 16-bit type + 48-bit hash = 64-bit universal address

---

## Era 2: Ladybug + NARS + Grammar (Dec 2025 – Jan 2026)

**NARS (Non-Axiomatic Reasoning System)** added epistemic state to every edge:
- frequency (how often X→Y) + confidence (how much evidence)
- 5 inference types: deduction, induction, abduction, revision, synthesis
- Pearl's 2³ decomposition: 8 causal masks (S/P/O combinations)

**Grammar Triangle** unified three vertices:
- NSM (65 Natural Semantic Metalanguage primes) — the atoms of meaning
- Causality (WHO → DID → WHAT → WHY) — agency and temporal flow
- Qualia (18D phenomenal coordinates) — the felt-sense dimensions

**Spectroscopy** emerged: reading implicit intent from text texture.
Not what's said, but what's BETWEEN the lines. IIC (Implicit Intent
Classification) feeds the 18D qualia field, which feeds style dispatch.

**Width migration attempt: 10K → 8K** (128 words). Motivation: power-of-2
alignment, cleaner SIMD. But 8K wasn't enough room for inline edges +
NARS + qualia + adjacency in metadata. Partially deployed, never completed.

---

## Era 3: Ladybug + Rustynum (Jan – Feb 2026)

**rustynum** was the SIMD acceleration crate:
- BLAS L1/L2/L3 (native, MKL, OpenBLAS backends)
- AVX-512, AVX2, NEON dispatch
- BF16/f16 conversion
- Hamming distance with SIMD popcount

Ladybug-rs depended on rustynum for all hardware acceleration.
Learning module grew to 300K+ LOC:
- `cam_ops.rs` (158K!) — 4096 CAM operations as cognitive vocabulary
- `cognitive_styles.rs` — 15 base + RL adaptation
- `cognitive_frameworks.rs` — NARS, ACT-R, RL, Pearl, qualia
- `quantum_ops.rs` — fingerprints as wavefunctions
- `dream.rs` — offline consolidation (prune/merge/permute-XOR-bind)
- `scm.rs` — structural causal model IN BindSpace

**Width migration: 8K → 16K** (256 words). 16,384 = 2^14. Exact u64 alignment.
No padding. Room for expanded metadata. Container becomes 2KB.
This is the PRODUCTION width. But 8K and 10K references persisted as debt.

---

## Era 4: BindSpace + Contract (Feb – Mar 2026)

**BindSpace** formalized the Container model:
- Container: `[u64; 256]` = 16K bits = 2KB, 64-byte aligned
- CogRecord: 2 × Container = metadata + content = 4KB
- PackedDn: u64 hierarchical address (7 levels × 8 bits)
- Spine: XOR-fold of children (lock-free, lazy recompute)
- 7 ContainerSemirings (BooleanBfs, HammingMin, etc.)
- Inline edges: 64 packed in metadata words 16-31

**lance-graph-contract** created as zero-dep trait crate:
- ThinkingStyle (36 styles, 6 clusters)
- MulAssessment (Dunning-Kruger, trust qualia)
- PlannerContract, OrchestrationBridge
- NarsTruth, InferenceType
- CamCodecContract

**Consumer adoption**: crewai-rust + n8n-rs depend on contract crate.
The contract IS the API surface — everything else is implementation.

---

## Era 5: Lance-Graph as Cold Path (Mar 2026)

**Attempt**: introduce lance-graph's Cypher parser as the cold-path query
engine while ladybug-rs remained the hot-path BindSpace substrate.

**Two-temperature architecture** emerged:
- Hot path: BindSpace (XOR probe, 0.3ns, fingerprint-addressed)
- Cold path: DataFusion (SQL/Cypher, milliseconds, columnar)
- `graph_router.rs` bridges both

**16 composable planner strategies** in lance-graph-planner.
But ladybug-rs was still the "main" — lance-graph was the "cold" side.

---

## Era 6: Stepping Up Lance-Graph (Mar – Apr 2026)

**The pivot**: lance-graph becomes the spine, not just the cold path.

**rustynum → ndarray migration**: All 80K LOC of rustynum ported into
ndarray fork as `src/hpc/` (55 modules, 880 tests):
- SIMD: AVX-512, AVX2, NEON (Pi Zero to Sapphire Rapids)
- AMX: TDPBF16PS via `asm!(".byte ...")` on stable Rust
- f16: carrier u16 + F16C hardware (binary hack for stable access)
- Pi Zero: ARM A53 single-pipeline NEON (2W, 80M lookups/sec)
- BF16: bit-exact RNE matching VCVTNEPS2BF16

**Hardening**: every platform from Pi Zero 2W to Xeon w9 Sapphire Rapids.
Same code, runtime dispatch via `LazyLock<SimdCaps>`.

---

## Era 7: Thinking Engine + P64 + CognitiveShader (Apr 2026)

**CausalEdge64**: one u64 = complete causal edge:
```
S(8b) + P(8b) + O(8b) + NARS_f(8b) + NARS_c(8b)
+ causal_mask(3b) + direction(3b) + inference(3b)
+ plasticity(3b) + temporal(12b) = 64 bits
```

**P64**: 64×64 bitmask palette adjacency. 8 predicate planes
(CAUSES/ENABLES/SUPPORTS/CONTRADICTS/REFINES/ABSTRACTS/GROUNDS/BECOMES).

**CognitiveShader** (née Blumenstrauß): binds topology × metric × algebra:
- 8 planes × 64×64 bitmask = topology (WHICH pairs interact)
- bgz17 PaletteSemiring = metric (HOW FAR, O(1) lookup)
- Compose table = algebra (WHAT path composition means, O(1))
- Style modulation: layer_mask + combine + contra per ThinkingStyle

**NarsTables**: precomputed 256×256 lookup tables. Every NARS inference
operation = one memory read. No floating point in the hot path.

**611M SPO lookups/sec. 17K tokens/sec. 388 KB RAM.**

---

## Era 8: AGI Typing + Cognitive Shader Endgame (Apr 2026, this session)

**The 67-codec sweep** killed lossy weight compression for inference
(argmax instability) but proved cascade acceleration (13x speedup,
100% argmax, zero quality loss).

**The realization**: weights are not parameters to compress — they are
**holographic memories to query**. The CognitiveShader IS the inference engine.

**6-7 dimensional struct-of-arrays for meta-cognition**:
```rust
pub struct CognitiveRecord {
    // Identity (WHAT)
    pub fingerprint: Fingerprint<256>,    // content
    pub cam_address: [u8; 6],             // CAM-PQ address

    // Encoding (HOW stored)
    pub hhtl_entry: HhtlDEntry,           // bgz tree address
    pub palette_idx: u8,                   // bgz17 archetype

    // Cognition (WHAT it means)
    pub edge: CausalEdge64,               // SPO+NARS packed
    pub shader_mask: u8,                   // active shader layers
    pub coca_idx: u16,                     // 4096 COCA position

    // Perspective (AGI dimensions)
    pub topic: Fingerprint<256>,           // what about
    pub angle: Fingerprint<256>,           // from whose view
    pub qualia: [f32; 18],                 // phenomenal state
    pub rung: u8,                          // causal level
}
```

**The ontological revolution**: weights are seeds. Each seed can exist in
vast parallel instances. Each instance feeds upstream learning via
CausalEdge64 branching. Each branch runs its own CognitiveShader per cycle
as a 5D stream:

```
Dimension 1: Content    (Fingerprint<256> — WHAT)
Dimension 2: Context    (topic binding — ABOUT WHAT)
Dimension 3: Perspective (angle binding — FROM WHERE)
Dimension 4: Causality   (CausalEdge64 — WHY/HOW)
Dimension 5: Time        (temporal index — WHEN)

Per cycle: CognitiveShader processes this 5D stream.
Per branch: independent causal trajectory.
Per merge: CausalEdge64 revision (NARS evidence accumulation).
Per dream: offline consolidation (prune/merge/permute-XOR-bind).
```

The weights don't define the model. The weights SEED the holographic memory.
The CognitiveShader IS the model. The cascade IS the inference. The edges
ARE the knowledge. The learning IS the branching.

---

## Timeline Summary

| Era | Period | Width | Core Innovation | LOC |
|---|---|---|---|---|
| 1 | Oct-Dec 2025 | 10K | 10 awareness layers, autopoietic styles | ~50K |
| 2 | Dec-Jan 2026 | 10K | NARS + grammar triangle + spectroscopy | ~100K |
| 3 | Jan-Feb 2026 | 8K→16K | rustynum SIMD + 4096 CAM ops + dream | ~350K |
| 4 | Feb-Mar 2026 | 16K | BindSpace + contract + crewai/n8n | ~100K |
| 5 | Mar 2026 | 16K | lance-graph cold path + 16 strategies | ~50K |
| 6 | Mar-Apr 2026 | 16K | ndarray migration + AMX + f16 + Pi Zero | ~80K |
| 7 | Apr 2026 | 16K | CausalEdge64 + P64 + CognitiveShader | ~30K |
| 8 | Apr 2026 | 16K | 67-codec sweep + AGI typing + holographic | ~40K |
| **Total** | | | | **~800K** |

---

## Current State (2026-04-18)

All of ladybug-rs is now imported into lance-graph ecosystem:
- `crates/lance-graph-cognitive/` — grammar, world, spo, search, fabric,
  spectroscopy, container_bs, core_full (wip-gated)
- `crates/learning/` — standalone, 16 modules (wip-gated)
- `crates/holograph/` — from RedisGraph, 38K LOC (10K→16K migrated)

The foundation (ndarray) has the SIMD. The spine (lance-graph) has the
query engine. The cognitive substrate (cognitive + learning) has the
reasoning. The shader (p64-bridge) has the dispatch. The types (contract)
have the API.

Everything converges on one number: **611 million lookups per second.**
That's the speed at which cognitive shaders run.
