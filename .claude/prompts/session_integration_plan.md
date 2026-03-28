# SESSION: Full Integration Plan — DeepNSM + AriGraph + SIMD + Psychometrics

## What Exists (Inventory)

### ndarray (1,318 lines deepnsm.rs + 348 lines simd_dispatch.rs)
- [x] 74 NSM primes + vocabulary (300 words) + decompose/similarity/fingerprint
- [x] Eval pipeline types: Prediction, Explication, SubstitutabilityScore, ModelResult
- [x] CAM-PQ bridge: load_nsm_codebook(), load_cam_codes()
- [x] SIMD via crate::simd: F32x16 normalization, U8x64 XOR, F32x16 cosine, F32x16 squared_l2
- [x] LazyLock<SimdDispatch> frozen function pointer table (7 tests)
- [x] SoA adjacency concept (documented, no overlap verified)
- [x] Psychometric validation concept (documented)
- [ ] wasm32 SIMD tier (std::arch::wasm32 v128)
- [ ] Safe SIMD refactor (remove unsafe from #[target_feature], Rust 1.94)
- [ ] jitson_cranelift/detect.rs dedup (use simd_caps() instead of raw is_x86_feature_detected)

### lance-graph deepnsm crate (3,045 lines, 38 tests, 0 deps)
- [x] vocabulary.rs: 4,096 lemmas + 11,460 inflected forms + contraction splitting
- [x] parser.rs: 6-state PoS FSM → SPO triples + modifiers + negation
- [x] encoder.rs: 512-bit VsaVec, XOR bind, majority bundle, role vectors
- [x] similarity.rs: 256-entry SimilarityTable from exact 4096² CDF
- [x] spo.rs: SpoTriple (36-bit) + WordDistanceMatrix (4096² u8)
- [x] codebook.rs: CAM-PQ codebook loader (96KB binary)
- [x] context.rs: ±5 sentence ring buffer for disambiguation
- [x] pipeline.rs: DeepNsmEngine full inference loop
- [ ] Integration test with real COCA data from word_frequency/
- [ ] SIMD acceleration of hot paths (squared_l2 in spo.rs uses scalar nested loops)
- [ ] Wire to thinking/graph.rs cognitive verbs

### lance-graph bgz-tensor crate (1,968 lines, 30 tests, 0 deps)
- [x] projection.rs: f32 → Base17 golden-step folding
- [x] palette.rs: CLAM-inspired manifold clustering → 256 archetypes
- [x] attention.rs: AttentionTable (128KB) + ComposeTable (64KB)
- [x] cascade.rs: HHTL 4-layer progressive elimination
- [x] quality.rs: Pearson/Spearman ρ, top-K recall, QualityReport
- [ ] GGUF loader: parse real Llama attention heads for benchmark
- [ ] Synthetic benchmark: measure ρ against ground-truth dot product
- [ ] One real Llama attention head benchmark

### lance-graph causal-edge crate (PR #52, merged)
- [x] edge.rs: CausalEdge64 (u64 packed SPO + NARS + Pearl + temporal)
- [x] pearl.rs: CausalMask 2³ projections
- [x] plasticity.rs: hot/cold/frozen per SPO plane
- [x] tables.rs: precomputed NARS revision/deduction lookup tables
- [x] network.rs: CausalNetwork with forward_chain, causal_query, counterfactual, Simpson's
- [ ] Wire to AriGraph TripletGraph (palette assignment + edge packing)
- [ ] Wire to bgz-tensor AttentionTable for cross-concept attention

### lance-graph arigraph module (4,339 lines, 6 modules)
- [x] triplet_graph.rs: full CRUD, BFS, spatial paths, NARS inference, update cycle
- [x] episodic.rs: fingerprint-based memory with capacity eviction
- [x] retrieval.rs: combined BFS + episodic, production prompts (590w+400w)
- [x] xai_client.rs: xAI/Grok API (ADA_XAI env), extraction/refinement/planning
- [x] sensorium.rs: GraphSensorium from live graph, diagnose_healing, apply_healing
- [x] orchestrator.rs: MetaOrchestrator + MUL + NARS topology + temperature + auto-heal
- [ ] Wire DeepNSM for entity resolution (sentence_similarity threshold)
- [ ] Wire CausalEdge64 as hypergraph replacement
- [ ] Dual episodic scoring (DeepNSM + triplet overlap)
- [ ] Agent orchestration via ThinkingGraph (replace hardcoded loop)

### lance-graph-contract sensorium (237 lines, 4 tests)
- [x] GraphSignals, GraphBias, HealingType, AgentStyle, OrchestratorMode
- [x] SensoriumProvider trait, OrchestratorProvider trait
- [ ] Wire lance-graph arigraph as impl of these traits

### q2 cockpit (display layer)
- [x] deepnsm.rs: lightweight NSM for notebooks (577 lines, 13 tests)
- [x] osint_audit.rs: OSINT pipeline audit (440 lines, 7 tests)
- [x] mri.rs: Brain MRI (500 lines, 6 tests)
- [x] orchestrator.rs: mirror types for REST consumption (1,563 lines)
- [x] BrainMriMode.tsx: 3D brain visualization with pre-baked orbits
- [x] WaveformMode.tsx: real-data waveform with exponential decay
- [x] /render /orbit /flight pages with demoscene camera paths
- [x] LazyLock double-buffer MRI pre-rendering (500ms refresh)
- [x] ConnectomeMode.tsx zoom fix (stale w/h + billboard rings)
- [ ] Strip orchestrator.rs to thin REST proxy (thinking is in lance-graph)
- [ ] Wire /mri to real AriGraph sensorium (currently empty graph)

---

## Integration Phases

### Phase A: Ground Truth (COCA data validation)
**Goal**: DeepNSM engine loads real COCA data and produces verifiable results.
**Effort**: ~4 hours

1. Clone DeepNSM repo word_frequency/ into lance-graph test fixtures
2. Write integration test: `DeepNsmEngine::load()` with real data
3. Verify tokenization: "the big dog bit the old man" → correct ranks
4. Verify 11,460 inflected forms resolve: "thought" → "think" (rank 53)
5. Verify distance matrix: "think"↔"know" < "think"↔"big"
6. Verify SimilarityTable: calibrated f32 [0,1] with correct statistics
7. Benchmark: tokenization < 1μs/word, parsing < 5μs/sentence

### Phase B: SIMD Hot Paths (crate::simd everywhere)
**Goal**: All hot-path operations use crate::simd. Consumer never sees hardware.
**Effort**: ~6 hours

1. cam_pq encode(): F32x16 for 6×256 centroid search (8-10× speedup)
2. cam_pq precompute_distances(): F32x16 for 1,536 L2 evals (8-10× speedup)
3. deepnsm spo.rs WordDistanceMatrix::build_from_cam(): batch distance via F32x16
4. blas_level1 iamax(): SIMD reduction via dispatch table
5. Add all new ops to SimdDispatch table (frozen function pointers)
6. Safe SIMD refactor: remove `unsafe` from `#[target_feature]` (Rust 1.94)
7. Wire jitson_cranelift/detect.rs to use simd_caps() singleton
8. Benchmark: before/after comparison on all SIMD-accelerated paths

### Phase C: AriGraph ↔ DeepNSM Wiring
**Goal**: AriGraph uses DeepNSM for entity resolution and semantic similarity.
**Effort**: ~4 hours

1. TripletGraph: add `resolve_entity()` using DeepNSM sentence_similarity
   - When adding triplet, check if subject/object similar to existing entity
   - Merge if similarity > threshold (configurable, default 0.85)
2. EpisodicMemory: dual scoring (DeepNSM similarity + triplet overlap)
   - First-pass: fingerprint Hamming for candidate retrieval
   - Second-pass: DeepNSM sentence_similarity for final ranking
   - Weight: 0.6 × DeepNSM + 0.4 × triplet_overlap (log-scaled)
3. Retrieval: wire DeepNSM into OsintRetriever for query expansion
   - Query words → nearest NSM primes → expanded seed entities
4. Tests: entity merging, dual scoring, expanded retrieval

### Phase D: CausalEdge64 ↔ AriGraph Wiring
**Goal**: CausalEdge64 replaces Hypergraph for causal reasoning over triplets.
**Effort**: ~3 hours

1. TripletGraph → CausalNetwork conversion:
   - Build palette from entity/relation namespace (256 archetypes)
   - Pack each Triplet as CausalEdge64: S/P/O palette indices + truth + timestamp
   - Build CausalNetwork from packed edges
2. Causal queries on the knowledge graph:
   - `causal_query(mask=SO)` = "what S is associated with this O?" (Level 1)
   - `causal_query(mask=PO)` with `do()` = interventional (Level 2)
   - `counterfactual()` = "what if different P?" (Level 3)
3. Simpson's Paradox detection on AriGraph data
4. Tests: conversion, queries, counterfactual, Simpson's

### Phase E: Vertical HHTL Bundling
**Goal**: Implement the studio mixing approach — separate signals before mixing.
**Effort**: ~8 hours

1. Define BundleLevel enum: Leaf, Twig, Branch, Hip
2. Implement VSA bottom-up bundling:
   - Leaf observations → majority vote → Twig vectors
   - Twigs → majority vote → Branch vectors
   - Branches → majority vote → Hip node
3. Implement top-down unbinding:
   - unbind(Hip, branch_role) → approximate Branch
   - Hamming(approximate, actual) = information loss at that level
4. Cross-strategy validation:
   - Strategy A per-plane distances at each level
   - Strategy B per-concept distances at each level
   - Correlation between A and B = bundling validity
5. BF16 noise floor removal before HEEL bundling
6. Background noise subtraction (minority bits in majority vote = noise)
7. Tests: bundle/unbind roundtrip, information loss, cross-strategy correlation

### Phase F: Psychometric Validation Framework
**Goal**: Every DeepNSM measurement comes with reliability and validity coefficients.
**Effort**: ~6 hours

1. Cronbach's α across 128 projections (2³ SPO × 2⁴ HHTL):
   - Each projection = one "test item" measuring the same construct
   - α > 0.7 = reliable construct, α < 0.5 = bundling destroys information
2. Split-half reliability:
   - Strategy A distance vs Strategy B distance for same pair
   - Pearson r = encoding reliability
3. Item Response Theory (IRT):
   - Per-word difficulty: how many primes cleanly decompose this word?
   - Per-word discrimination: does this word reliably separate concepts?
   - Per-prime reliability: consistent contribution across vocabulary
4. Factor analysis:
   - Do 74 primes factor into 16 NsmCategory groups?
   - Or does PCA reveal a different latent structure?
   - Compare with COCA distributional structure
5. Polysemy detection:
   - Word with high α in context = disambiguated
   - Word with low α across projections = polysemous (detected!)
6. Measurement invariance:
   - cam_pq → distance vs bgz17 → distance vs raw 96D L2
   - All three should rank identically (invariant)
7. P-values: 128 independent measurements → p < 0.001 per pair
8. Tests: α computation, split-half, IRT parameters, polysemy detection

### Phase G: Contract Trait Implementation
**Goal**: AriGraph implements the contract traits so any consumer can use it.
**Effort**: ~3 hours

1. Implement SensoriumProvider for AriGraph:
   - `signals()` → GraphSensorium::from_graph()
   - `diagnose()` → diagnose_healing()
   - `heal()` → apply_healing()
2. Implement OrchestratorProvider for MetaOrchestrator:
   - `status()` → snapshot()
   - `step()` → select_next()
   - `record_outcome()` → record_outcome()
   - `update_signals()` → update_sensorium()
   - `auto_heal()` → auto_heal()
3. Wire q2 cockpit to use the contract traits (not direct types)
4. Tests: trait implementation verification

### Phase H: q2 Display Cleanup
**Goal**: q2 is pure display, all thinking is in lance-graph.
**Effort**: ~2 hours

1. Strip q2 orchestrator.rs to thin REST proxy:
   - Keep only serde types for JSON deserialization
   - All logic calls into lance-graph via contract traits
2. Wire /mri to real AriGraph sensorium:
   - LazyLock pre-render reads from live GraphSensorium
   - Background task polls real graph health signals
3. Wire BrainMriMode to real pipeline counters
4. Wire WaveformMode to real OSINT stage counters

### Phase I: wasm32 SIMD Tier
**Goal**: ndarray crate::simd works in WASM with 128-bit SIMD.
**Effort**: ~4 hours

1. Add `#[cfg(target_arch = "wasm32")]` tier to simd.rs:
   - F32x4 backed by v128 (std::arch::wasm32)
   - F32x16 emulated as 4× F32x4 (still 4× faster than scalar)
2. Add wasm_simd128 to SimdCaps (always true when compiled with +simd128)
3. Add wasm32 tier to SimdDispatch (frozen at module load, no detection needed)
4. Build test: `cargo build --target wasm32-unknown-unknown -C target-feature=+simd128`
5. Verify: F32x16 operations produce identical results on x86_64 and wasm32

---

## Dependency Graph

```
Phase A (ground truth)
  ↓
Phase B (SIMD hot paths)      Phase C (AriGraph ↔ DeepNSM)
  ↓                              ↓
Phase D (CausalEdge64 wiring)  Phase E (vertical HHTL bundling)
  ↓                              ↓
Phase F (psychometric validation) ← requires E for 128 projections
  ↓
Phase G (contract traits) ← requires C, D, F for full impl
  ↓
Phase H (q2 cleanup) ← requires G for trait-based wiring
  |
Phase I (wasm32 SIMD) ← independent, can run in parallel with any phase
```

## Estimated Total: ~40 hours across 9 phases

## Success Criteria

- [ ] DeepNSM loads real COCA data, tokenizes correctly, distance matrix verified
- [ ] All hot-path SIMD uses crate::simd only, zero raw intrinsics in consumer code
- [ ] LazyLock frozen dispatch with AVX-512 → AVX2 → scalar fallback
- [ ] AriGraph entity resolution via DeepNSM similarity (threshold 0.85)
- [ ] CausalEdge64 network built from AriGraph triplets, causal queries work
- [ ] Vertical HHTL bundling: bundle → unbundle → information loss measured
- [ ] Cronbach's α > 0.7 for NSM prime decomposition reliability
- [ ] Contract traits implemented, q2 uses traits not direct types
- [ ] wasm32 SIMD tier compiles and produces identical results
- [ ] All tests pass (currently: ~250+ lance-graph, 23 deepnsm, 38 crate, 30 tensor)

---

### Phase J: Free Energy Minimization — Contradictions as Exploration Fuel
**Goal**: Entropy is potential, not waste. Contradictions are navigation waypoints, not errors.
**Effort**: ~8 hours
**Depends on**: Phases C, D, E, F

The Friston free energy principle applied to the codec chain:

```
F = Entropy(model) - Evidence(observations)

Minimize F via:
  PERCEPTION: update model → NARS revision + modifier discovery → entropy ↓
  ACTION: explore contradictions → find modifiers → resolve → evidence ↑
```

1. **OutcomeTriad**: desired (goal) × expected (prediction) × factual (reality)
   - Each encoded as QualiaVector (16 channels)
   - Three deltas: Δ_surprise, Δ_satisfaction, Δ_aspiration
   - Per-channel learning signals (SOCIAL surprise, TEMPORAL surprise, etc.)
   - Epiphany detector: large positive Δ_surprise → exploration burst
   - Blunder detector: large negative Δ_surprise → consolidation

2. **Contradiction-as-modifier search**:
   - Contradiction = pointer to missing variable, NOT noise to eliminate
   - For each contradiction, search for qualifying modifiers:
     temporal ("X R Y WHEN t1" vs "X R' Y WHEN t2")
     spatial ("X R Y WHERE loc1" vs "X R' Y WHERE loc2")
     causal (CausalEdge64 Pearl mask: L1 observation vs L2 intervention vs L3 counterfactual)
   - Finding the modifier = epiphany (model becomes more specific)
   - Not finding = genuine contradiction → reduce weaker confidence

3. **Entropy-as-Friston-potential**:
   - GraphSensorium.truth_entropy IS the free energy proxy
   - High entropy + low epiphany = rich unexplored territory
   - High entropy + high epiphany = active learning (let it flow)
   - Low entropy + low epiphany = check for hidden drift (α dropping?)
   - Orchestrator temperature directly tracks free energy

4. **HHTL cascade as entropy gradient descent**:
   - HEEL: rank contradictions by entropy contribution (explore highest first)
   - HIP: search for modifiers via Pearl mask projections
   - BRANCH: successful modifiers become qualified triplets
   - TWIG: psychometric validation (did α improve? genuine learning?)
   - LEAF: residual → NARS abduction → hypothesize new latent factor

5. **Epiphany-driven self-reinforcement**:
   - Epiphany (+): surprise > threshold, factual > expected
     → DK shifts toward Valley (humility), temperature spikes
     → Explore the new territory, don't consolidate prematurely
   - Blunder (-): surprise > threshold, factual < expected
     → DK shifts toward MountStupid check, temperature drops
     → Consolidate, run contradiction detection, find what went wrong
   - Stagnation (0): no surprise, check if mastery or plateau trap
     → Periodic perturbation via temperature injection
     → α check: genuine mastery (α stable) vs hidden drift (α dropping)

6. **2 billion scientific studies scenario**:
   - Each study = observation → DeepNSM → SPO triples → CausalEdge64
   - Contradictions between studies = modifier search targets
   - "Study A says X causes Y, Study B says X prevents Y"
     → Modifier: population, dosage, timeframe, methodology
   - Finding the modifier = meta-analysis result (automatically)
   - Entropy of the knowledge graph = remaining scientific disagreement
   - Free energy minimization = systematic resolution of disagreements
   - The chaos IS the signal — more contradictions = more potential for discovery
   - Thanks for the chaos, it feeds our tensors

Success criteria:
  - [ ] OutcomeTriad with 16-channel QualiaVector comparison
  - [ ] Modifier search via CausalEdge64 Pearl projections
  - [ ] Entropy-as-potential integrated into GraphSensorium
  - [ ] Epiphany/blunder detection with per-channel localization
  - [ ] HHTL cascade as entropy gradient descent
  - [ ] Psychometric validation of modifier discoveries (α improvement)
  - [ ] Self-reinforcement: epiphanies improve exploration, blunders improve consolidation
