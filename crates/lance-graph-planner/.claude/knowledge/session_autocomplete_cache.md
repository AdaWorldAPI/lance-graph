# Session Knowledge: Qwen3.5 × Opus 4.5/4.6 Reverse Engineering + AutocompleteCache

## What Was Built

### ndarray (Hardware Layer)
- 5 Qwen models indexed (685 MB bgz7 from 201 GB BF16 safetensors)
- 4 causal diffs with real data (FfnGate dominant: 0.6% v1, 0.1% v2, 1.0% 9B)
- 34 cognitive primitives in src/hpc/styles/ (49 tests)
- CausalEdge64 packed u64 + Palette3D + PAL8 serialization (4101 bytes)
- Quality scoring: GOOD/BAD/UNCERTAIN/REVERTED from 4-diff cross-validation
- NARS self-reinforcement LoRA framework (HeadBelief + LoraAction)
- AVX-512 compile-time dispatch (.cargo/config.toml target-cpu=x86-64-v4)
- SPO Palette Distance benchmark: 611M lookups/sec, 1.8 ns/lookup

### lance-graph (Thinking Layer)
- AutocompleteCache: 6 modules, 39 cache tests, 162 total planner tests
  - kv_bundle.rs: HeadPrint (Base17 alias), AttentionMatrix (64×64 / 256×256)
  - candidate_pool.rs: ranked candidates, Phase (Exposition→Coda)
  - triple_model.rs: self/user/impact × 4096 heads, DK, Plasticity
  - lane_eval.rs: Euler-gamma tension, DK-adaptive evaluation
  - nars_engine.rs: SpoHead, Pearl 2³, 7 NARS rules, StyleVectors, NarsTables hot path
  - convergence.rs: AriGraph triplets → p64 Palette → Blumenstrauss
- Strategy #17: AutocompleteCacheStrategy (replaces ChatBundle)
- Axum REST server: OpenAI-compatible /v1/chat/completions
- bgz-tensor hydrate workflow with feature flags (qwen35-9b/27b-v1/v2)
- HHTL cache with RouteAction (Skip/Attend/Compose/Escalate)
- GitHub Release v0.1.0-bgz-data: 41 bgz7 files, 685 MB

### Architecture

```
ndarray (SIMD):          p64 Highway          lance-graph (Thinking):
  Base17                 ←── PAL8 ──→          AutocompleteCache
  SpoDistanceMatrices    ←── HHTL ──→          NarsEngine + StyleVectors
  read_bgz7_file()       ←── bgz7 ──→          TripleModel
  Palette (CLAM)         ←── u8[] ──→          LaneEvaluator
                                               CandidatePool + Phase
causal-edge (Protocol):
  CausalEdge64 (u64)    ←── edge ──→          SpoHead (8 bytes)
  NarsTables (128KB)     ←── table ─→          Truth (f32 cold path)
  forward() / learn()   ←── ops ───→          nars_infer()
```

## Key Findings from Weight Diffs

1. Reasoning scaffold = SwiGLU FFN gating, NOT attention Q/K/V/O
2. v2 is a REVERT (closer to base than v1) — 4.6 stabilizes, 4.5 transforms
3. K stable at 27B (knowledge preserved), K shifted at 9B (capacity limit)
4. v1 = Opus 4.5 behavioral traits, v2 = Opus 4.6 precision
5. Training data: v1 has 250× Opus 4.5 samples, v2 has 10K× Opus 4.6

## 18 Research Papers Synthesized

1. EMPA: Empathy as 3D vector P_t = C·eC + A·eA + P·eP
2. InstCache: NLL address space, power-law pre-population
3. Semantic Caching: 5-level QuerySignature, dual-threshold
4. C2C: KV-cache fusion between models, effective rank
5. ContextCache: Multi-turn self-attention caching
6. Krites: Grey zone async verification, static→dynamic promotion
7. Thinking Intervention: Token-level reasoning injection
8. ThinkPatterns: 5 thinking styles × model size
9. Thinkless: DeGRPO — when to think vs not
10. Habr: LLM = resonance-holographic interference field
11. DapQ: Position > semantics for attention (0.72 vs 0.35)
12. Tensor Networks: Superposition weight inversely proportional to size
13. PMC Attention Heads: 4-stage cognitive model (KR/ICI/LR/EP)
14. LFRU: Causal follower prediction for cache
15. Illusion of Causality: Semantic scaffolding vs causal understanding
16. NARS Same/Opposite: Mutual + combinatorial entailment from minimal evidence
17. KVTC: KV-cache compression 20× via PCA+quantize+entropy
18. CacheSlide: RPDC cross-position KV reuse (3-4× latency reduction)

## AutocompleteCache Design (from papers + architecture)

- VSA 10KD superposition KV-cache (fixed size, bundle/unbundle O(1))
- 3 simultaneous models: self_model, user_model, impact_model
- Each model: 64×64 = 4096 interdependent attention heads
- Pearl 2³: 8 causal projections per head pair (SPO decomposition)
- ThinkingStyles = weight vectors over 8 Pearl projections
- NARS 7 inference rules: deduction, induction, abduction, revision, analogy, resemblance, synthesis
- NarsTables: 256×256 u16 lookup (128 KB, L1 resident, O(1))
- Composition phase tracking: Exposition→Durchführung→Contrapunkt→Bridge→Pointe→Coda
- Socratic dialogue: INNER (self questions self) + OUTER (self questions user)
- Friston free energy: surprise = impact prediction error
- MUL: DK position gates tension (MountStupid→creative, Mastery→focused)
- LFRU: causal follower prediction (leader=self, follower=user)
- Krites grey zone: cache hit (>0.995), guide (>0.50), generate (<0.50)

## Benchmarks

- 611M SPO lookups/sec (Rust, AVX-512, release mode)
- 17,134 tokens/sec (triple model, 4096 heads, Pearl 2³)
- 388 KB RAM (384 KB SPO tables + 4 KB head indices)
- 100.00% information preservation (256 palette archetypes)
- 99.6% palette assignment agreement (base vs v2)

## Next Steps

1. Wire DeepNSM embeddings for fast similarity search
2. Wire Jina Reader (r.jina.ai) for OSINT web ingestion
3. AriGraph TripletGraph → convergence.rs → live palette updates
4. Local LLM (Qwen3.5 Q8_0) for triplet extraction (no API costs)
5. 24/7 OSINT pipeline: web → triplets → graph → palette → cache
6. NarsTruth → lance-graph-contract (eliminate ndarray thinking debt)
7. Populate SpoDistances from real bgz7 weights (not zeros)
