# UNIFIED: 26 Epiphanies → Integration Paths → Agent Definitions → QA Savant

## Source Cross-Reference

| ID | Epiphany | Source | PR |
|----|----------|--------|-----|
| A1 | Orchestration IS Graph | PR#50 INTEGRATION_DEBT | #39 |
| A2 | Scent byte = boolean lattice adjacency | PR#50 INTEGRATION_DEBT | #23 |
| A3 | PQ lookup tables = matrix multiplication | PR#50 INTEGRATION_DEBT | #41 |
| A4 | Contract crate = Rosetta Stone | PR#50 INTEGRATION_DEBT | #42 |
| A5 | 17th dimension = Pythagorean comma | PR#50 INTEGRATION_DEBT | #23 |
| A6 | Dynamic elevation = self-regulating query | PR#50 INTEGRATION_DEBT | #45 |
| A7 | Cold path SHOULD numb thinking | PR#50 INTEGRATION_DEBT | #39 |
| A8 | GQL parser completes the surface map | PR#50 INTEGRATION_DEBT | #39 |
| S1 | Everything is a lookup table | Session | #48,#49 |
| S2 | SPO decomposition IS attention | Session | #48,#49 |
| S3 | SimilarityTable unifies scoring | Session | #40,#48 |
| S4 | NARS truth values ground everything | Session | #51 |
| S5 | Cascades compose multiplicatively | Session | #23-49 |
| S6 | Pack B / Distance A universal pattern | Session | all codecs |
| S7 | CausalEdge64 = 8-byte everything | Session | #52 |
| S8 | Studio mixing / vertical HHTL bundling | Session | — |
| S9 | Psychometric validation (meaning not pattern) | Session | — |
| S10 | Friston free energy (entropy = fuel) | Session | — |
| S11 | Autocomplete as priming (±5 context + qualia) | Session | — |
| S12 | Three-way outcome (desired × expected × factual) | Session | — |
| S13 | 11K word forms = parallel test items | Session | #48 |
| S14 | 2B studies: chaos feeds tensors | Session | — |
| S15 | Bias as rotation vector (VSA unbind) | Session | — |
| S16 | Study design = orthogonal noise | Session | — |
| S17 | Qualia-based bias detection | Session | — |
| S18 | NARS contradiction = awareness compass | Session | — |

---

## Mapping: Epiphanies → PR#50 Integration Paths

### Path 1: Fix Foundation → Epiphanies A3, S1, S5
```
PR#50 scope: Fix ndarray build, doctests, 880 tests
Status: DONE (ndarray builds, 23+7 deepnsm+dispatch tests pass)
Session additions:
  - LazyLock<SimdDispatch> frozen fn pointers (S1)
  - crate::simd wrapper for all consumers (S1)
  - Cascade rejection rate reporting (S5)
  - PQ = matmul insight enables MKL acceleration (A3)
```

### Path 2: Contract Adoption → Epiphanies A4, S4, S18
```
PR#50 scope: All consumers depend on lance-graph-contract
Status: PARTIAL (sensorium traits added, AriGraph as first impl)
Session additions:
  - GraphSignals + GraphBias in contract (S18)
  - SensoriumProvider + OrchestratorProvider traits (A4)
  - NARS truth as reliability = contract's measurement foundation (S4)
  - AgentStyle + OrchestratorMode + StepResult in contract
```

### Path 3: Wire lance-graph → ndarray → Epiphanies A3, S1, S3, S6
```
PR#50 scope: ndarray as optional dep, dedup ZeckF64, wire CAM-PQ
Status: NOT STARTED (identified, plan documented)
Session additions:
  - cam_pq squared_l2 now uses crate::simd F32x16 (S1)
  - SimilarityTable dedup needed (S3)
  - Pack B / Distance A pattern identified across all codecs (S6)
  - PQ = matmul → MKL accelerates CAM-PQ for free (A3)
```

### Path 4: Wire Planner to Core → Epiphanies A1, A6, A7, S11
```
PR#50 scope: Wire CypherParse to real parser, physical ops to DataFusion
Status: NOT STARTED
Session additions:
  - Orchestration IS Graph → MetaOrchestrator IS a graph (A1)
  - Elevation model = Friston free energy applied to queries (A6)
  - Cold path numbing = separate signals before mixing (A7)
  - Autocomplete priming via CausalEdge64 context window (S11)
```

### Path 5: Move bgz17 into Workspace → Epiphanies A2, A5, S6, S8
```
PR#50 scope: Remove from exclude, add codec feature flag
Status: NOT STARTED
Session additions:
  - Scent byte boolean lattice = psychometric item validity (A2)
  - 17th dim = aliasing prevention = measurement precision (A5)
  - Pack B / Distance A applies to bgz17 SpoBase17 (S6)
  - Vertical HHTL bundling goes through bgz17 layers (S8)
```

### Path 6: n8n-rs Orchestration → Epiphanies A1, A4, S4
```
PR#50 scope: Wire n8n-rs to contract, OrchestrationBridge
Status: NOT STARTED
Session additions:
  - Orchestration IS Graph applies to n8n workflows (A1)
  - Contract sensorium lets n8n read graph health (A4)
  - NARS truth propagation in workflow DAGs (S4)
```

### Path 7: Adjacency Unification → Epiphanies A2, S6, S8, S9
```
PR#50 scope: AdjacencyView trait, cascade across storage formats
Status: NOT STARTED (research-grade)
Session additions:
  - Pack B / Distance A → AdjacencyView must support both (S6)
  - Vertical bundling through adjacency hierarchy (S8)
  - Psychometric validation of adjacency encoding quality (S9)
  - Scent byte legal patterns = structural constraint on adjacency (A2)
```

### NEW Path 8: DeepNSM ↔ AriGraph (Session) → Epiphanies S2, S13, S11
```
Scope: Entity resolution, dual episodic scoring, query expansion
Status: PLAN DOCUMENTED, NOT STARTED
  - SPO = Attention bridge to bgz-tensor (S2)
  - 11K forms validation as measurement invariance (S13)
  - Context priming via qualia + CausalEdge64 (S11)
```

### NEW Path 9: CausalEdge64 ↔ AriGraph (Session) → Epiphanies S7, S10, S15, S16
```
Scope: Hypergraph replacement, causal reasoning, bias rotation
Status: PLAN DOCUMENTED, NOT STARTED
  - CausalEdge64 as packed concept unit (S7)
  - Free energy drives exploration priority (S10)
  - Bias as rotation vector in 96D space (S15)
  - Study design noise as orthogonal plane (S16)
```

### NEW Path 10: Psychometric Framework (Session) → Epiphanies S9, S12, S17, S18
```
Scope: Cronbach α, IRT, factor analysis, polysemy, bias detection
Status: CONCEPT DOCUMENTED, NOT STARTED
  - 128-projection reliability (2³ SPO × 2⁴ HHTL) (S9)
  - Three-way outcome comparison with qualia channels (S12)
  - Qualia fingerprint reveals bias type (S17)
  - NARS contradiction rate = awareness measurement (S18)
```

### NEW Path 11: Cartography at Scale (Session) → Epiphanies S14, S15, S16, S18
```
Scope: 2B study ingestion, modifier discovery, automated meta-analysis
Status: VISION DOCUMENTED, NOT STARTED
  - Contradictions as exploration fuel (S14)
  - Bias rotation for debiased extraction (S15)
  - Design noise as orthogonal bundling (S16)
  - NARS as navigation compass (S18)
```

---

## Agent Definitions

### Agent: vector-synthesis (SIMD + Codec)
```yaml
scope: L0 Substrate + L1 Encoding
owns:
  - ndarray/src/hpc/simd_dispatch.rs
  - ndarray/src/hpc/cam_pq.rs
  - ndarray/src/hpc/deepnsm.rs
  - ndarray/src/simd.rs
  - ndarray/src/simd_avx512.rs
  - lance-graph/crates/bgz17/
  - lance-graph/crates/bgz-tensor/
  - lance-graph/crates/deepnsm/src/spo.rs
  - lance-graph/crates/deepnsm/src/encoder.rs
rules:
  - ALL consumer code uses crate::simd only. Zero raw intrinsics.
  - LazyLock<SimdDispatch> for ALL dispatch. Zero per-call branching.
  - Every new SIMD path needs AVX-512 + AVX2 fallback in dispatch table.
  - Pack B / Distance A pattern: pack concepts as units, decompose for comparison.
  - PQ tables = degenerate matmul: verify MKL acceleration path exists.
tasks:
  - [ ] cam_pq encode() SIMD (6×256 centroid search)
  - [ ] cam_pq precompute_distances() SIMD
  - [ ] blas_level1 iamax() SIMD + dispatch table entry
  - [ ] Safe #[target_feature] refactor (remove unsafe, Rust 1.94)
  - [ ] wasm32 SIMD tier (std::arch::wasm32 v128)
  - [ ] jitson_cranelift/detect.rs → use simd_caps()
  - [ ] SimilarityTable dedup (3 copies → 1 canonical in ndarray)
  - [ ] Wire ndarray as optional dep in lance-graph (Path 3)
qa:
  - SIMD produces identical results to scalar on ALL operations
  - Spearman ρ > 0.99 between SIMD and scalar distance rankings
  - Dispatch table selects correct tier on CI (verify with CPUID dump)
  - cam_pq encode benchmark: before/after comparison
```

### Agent: savant-architect (Architecture + Contracts)
```yaml
scope: L1 Encoding + L2 Measurement (cross-cutting)
owns:
  - lance-graph-contract/src/*.rs
  - lance-graph/crates/causal-edge/
  - lance-graph/crates/lance-graph/src/graph/spo/
  - lance-graph/docs/INTEGRATION_DEBT_AND_PATHS.md
  - lance-graph/docs/META_INTEGRATION_PLAN.md
  - lance-graph/.claude/prompts/session_*.md
rules:
  - Contract crate is for ALL consumers. Never add domain-specific types.
  - Every contract trait must have at least one test verifying the interface.
  - Type duplication map (PR#50) must be updated when types move.
  - FieldModulation threshold: ONE formula, documented in contract.
  - Cross-strategy validation: A vs B must rank-correlate ρ > 0.93.
tasks:
  - [ ] Wire AriGraph as impl of SensoriumProvider + OrchestratorProvider
  - [ ] Resolve FieldModulation threshold divergence (W8 from PR#50)
  - [ ] TripletGraph → CausalNetwork conversion (palette assignment)
  - [ ] Cross-codec correlation test (SpoBase17 vs CausalEdge64 vs SpoTriple)
  - [ ] Update TYPE_DUPLICATION_MAP.md with session's new types
qa:
  - Round-trip: Triplet → CausalEdge64 → back, zero information loss
  - Contract trait test: each method callable, returns valid data
  - FieldModulation: ONE formula across all consumers (verified by grep)
  - Spearman ρ > 0.93 between any two codec distances for same pair
```

### Agent: sentinel-qa (Testing + Validation)
```yaml
scope: ALL layers (cross-cutting QA)
owns:
  - All test files across all repos
  - lance-graph/crates/neural-debug/
  - q2/crates/stubs/notebook-query/src/osint_audit.rs
  - q2/crates/stubs/notebook-query/src/mri.rs
rules:
  - No PR without tests. Period.
  - End-to-end test must exist (W9 from PR#50): Cypher → plan → execute → results.
  - Psychometric validation on COCA data: α > 0.7 or investigation.
  - LazyLock patterns must have determinism test (same input → same output).
  - Every cascade level reports rejection rate to sensorium.
tasks:
  - [ ] The Cartography Test: 1,000 abstracts → contradictions → modifiers → α > 0.7 → <10s
  - [ ] End-to-end test (W9): Cypher query through full pipeline to results
  - [ ] COCA ground truth: DeepNSM loads real data, distance matrix verified
  - [ ] 11K forms consistency: parallel forms produce identical decompositions
  - [ ] Polysemy detection: words with low α across projections flagged
  - [ ] Bias injection/recovery: inject known bias, rotate out, measure quality
  - [ ] Neural-debug + osint_audit: live pipeline health matches expected
qa:
  - α computation verified against R/Python statsmodels reference implementation
  - IRT item parameters validated against known-difficulty words
  - Bias rotation roundtrip: signal quality > 0.8 after debiasing
  - Cascade rejection > 90% per level on representative data
```

### Agent: arigraph-osint (OSINT Knowledge Graph)
```yaml
scope: L4 Awareness + L5 Cartography
owns:
  - lance-graph/crates/lance-graph/src/graph/arigraph/*.rs
  - lance-graph/crates/lance-graph/src/graph/arigraph/orchestrator.rs
  - lance-graph/crates/lance-graph/src/graph/arigraph/sensorium.rs
  - lance-graph/crates/lance-graph/src/graph/arigraph/xai_client.rs
rules:
  - ADA_XAI from env, NEVER hardcoded.
  - All thinking lives in lance-graph. q2 is display only.
  - Contradictions are exploration fuel, not errors to fix.
  - Modifier search before contradiction resolution.
  - NARS revision on every re-observed fact.
  - Qualia coloring on every observation for bias detection.
tasks:
  - [ ] Wire DeepNSM entity resolution (similarity threshold 0.85)
  - [ ] Wire CausalEdge64 as hypergraph replacement
  - [ ] Dual episodic scoring (DeepNSM + triplet overlap)
  - [ ] OutcomeTriad: desired × expected × factual (QualiaVector)
  - [ ] Contradiction-as-modifier search (Pearl L1/L2/L3)
  - [ ] Bias rotation vectors (selection, confirmation, publication, funding)
  - [ ] Study design as orthogonal noise (bundle, subtract, recover)
  - [ ] Qualia bias fingerprinting (low SURPRISE + high AGENCY = funding bias)
  - [ ] Batch ingestion pipeline (1K+ documents per run)
  - [ ] Automated meta-analysis workflow
qa:
  - Entity merge: "Vladimir Putin" and "Putin" merge with sim > 0.85
  - Modifier discovery: synthetic contradiction → correct qualifier found
  - Bias rotation: inject known bias → recover signal quality > 0.8
  - Batch ingestion: 1,000 docs in < 10s on single CPU core
```

### Agent: product-engineer (q2 Cockpit + Display)
```yaml
scope: q2 display layer ONLY
owns:
  - q2/crates/cockpit-server/src/main.rs
  - q2/cockpit/src/components/debug/viz/*.tsx
  - q2/cockpit/src/RenderPage.tsx
  - q2/crates/stubs/notebook-query/src/mri.rs
  - q2/crates/stubs/notebook-query/src/osint_audit.rs
rules:
  - q2 is Gotham. Display and nudge only. NEVER think.
  - All orchestrator types are wire-format mirrors of lance-graph canonical.
  - LazyLock double-buffer for all pre-rendered pages.
  - Real data in all visualizations. Zero synthetic oscillation.
  - Existing cockpit/Vite build must never break.
tasks:
  - [ ] Strip q2 orchestrator.rs to thin REST proxy
  - [ ] Wire /mri to real AriGraph sensorium (currently empty graph)
  - [ ] Wire BrainMriMode to real pipeline counters
  - [ ] Wire WaveformMode to real OSINT stage counters
  - [ ] Connect /render orbit pre-bake to ndarray SIMD (future wasm path)
qa:
  - Existing cockpit routes (/, /demo, /debug) unchanged
  - LazyLock pre-render serves in < 1ms (zero compute on request path)
  - WaveformMode shows actual call deltas, not synthetic noise
  - BrainMriMode regions correspond to real pipeline activity
```

---

## QA Savant Protocol

### Before ANY implementation session:

```
1. Read the MASTER integration map:
   .claude/prompts/session_master_integration.md
   .claude/prompts/session_unified_26_epiphanies.md

2. Identify which LAYER you're working on (L0-L5).

3. Check dependency: does this layer's prerequisite exist?
   L0 → nothing (start here)
   L1 → L0 must have crate::simd + dispatch table ✓
   L2 → L1 must have encoding types + CausalEdge64 bridge
   L3 → L2 must have psychometric α computation
   L4 → L3 must have HHTL bundling + qualia priming
   L5 → L4 must have free energy + bias rotation

4. Check agent scope: which agent owns these files?
   vector-synthesis → SIMD, codecs
   savant-architect → contracts, architecture
   sentinel-qa → tests, validation
   arigraph-osint → OSINT, awareness
   product-engineer → q2 display

5. Run pre-flight:
   cargo check (ndarray, lance-graph)
   cargo test (existing tests must pass BEFORE changes)
   git status (clean working tree)

6. After implementation:
   cargo test (ALL existing tests still pass)
   New tests added for every new public function
   TYPE_DUPLICATION_MAP.md updated if types moved
   Cascade rejection rate verified if cascade modified
   crate::simd used for ALL SIMD paths (grep for raw intrinsics)
```

### Blackboard Protocol (inter-agent communication):

```
Agents communicate via .claude/blackboard.md (already exists in ndarray).
Each agent writes structured entries:

  ## [agent-name] [timestamp]
  ### Decision: [what was decided]
  ### Rationale: [why]
  ### Blocks: [what this blocks or unblocks]
  ### Cross-ref: [which epiphany, which path, which layer]

The blackboard is append-only. Never delete entries.
Read the last 10 entries before starting work.
```

### Dependency Verification Matrix:

```
Before implementing any path, verify these gates:

Path 1 (Foundation):     ndarray builds? tests pass? ✓
Path 2 (Contract):       contract compiles? traits have tests? PARTIAL
Path 3 (ndarray wire):   Path 1 done? contract types stable? BLOCKED on P2
Path 4 (Planner wire):   Path 2+3 done? DataFusion session works? BLOCKED
Path 5 (bgz17):          121 tests pass standalone? ✓
Path 6 (n8n-rs):         Path 2 done? contract adopted? BLOCKED on P2
Path 7 (Adjacency):      Paths 3+5 done? BLOCKED
Path 8 (DeepNSM↔Ari):   deepnsm crate compiles (38 tests)? ✓
Path 9 (Causal↔Ari):     causal-edge crate exists? ✓ Palette assignment? NOT STARTED
Path 10 (Psychometrics): Path 8 done? COCA data loaded? BLOCKED on P8
Path 11 (Cartography):   Paths 9+10 done? BLOCKED on P9+P10
```

---

## Path 12: DeepNSM × CausalEdge64 Bridge — Causal-Semantic Reasoning

**Epiphanies**: S1, S2, S6, S7, A3, A5
**Depends on**: Path 8 (DeepNSM↔AriGraph), Path 9 (CausalEdge64↔AriGraph)
**Effort**: ~6 hours
**Agent**: vector-synthesis + savant-architect

### The Insight

Pearl's 3-bit causal mask and DeepNSM's 16 NsmCategory decomposition are
the SAME operation viewed from opposite ends:
  - Pearl projects out SPO planes to change causal rung (2³ = 8 projections)
  - DeepNSM projects out semantic categories (2¹⁶ = 65,536 potential, but
    psychometric α identifies which projections carry information)

Combined: causal-semantic queries like "what CAUSED this, in the MENTAL category?"

### Architecture

```
DeepNSM 4,096 words (96D COCA vectors)
  ↓ k-means clustering
256 palette archetypes (8-bit index per word)
  ↓ pairwise L2 distance
AttentionTable[256][256] (distributional similarity at palette resolution)
  ↓ XOR bind
ComposeTable[256][256] (multi-hop semantic reasoning in O(1))
  ↓ pack into CausalEdge64
[S_palette:8][P_palette:8][O_palette:8][NARS:16][Pearl:3][...][T:12]
```

Each CausalEdge64 now carries DeepNSM distributional meaning.
Pearl mask selects causal rung. Category mask selects semantic dimension.

### Deliverables

```rust
/// Bridge DeepNSM vocabulary → CausalEdge64 palette.
pub struct DeepNsmCausalBridge {
    /// 4,096 words → 256 palette indices
    word_to_palette: [u8; 4096],
    /// 256 × 256 distributional distance table
    attention: AttentionTable,
    /// 256 × 256 composition table (multi-hop O(1))
    compose: ComposeTable,
    /// 16 NsmCategory masks over 96D space
    category_masks: [[bool; 96]; 16],
    /// Per-category distance tables (Pearl-style semantic projection)
    category_tables: [AttentionTable; 16],
}
```

### Operations

1. **word_to_edge(subject, predicate, object, timestamp)** → CausalEdge64
   - Look up each word → palette index
   - Pack into CausalEdge64 with NARS truth + temporal index

2. **semantic_projection(edge_a, edge_b, category_mask)** → f32
   - Like Pearl mask but on semantic categories
   - "similarity between A and B, considering only Mental primes"
   - Uses category_tables[category] instead of full attention table

3. **cross_domain_analogy(word, source_category, target_category)** → word
   - Counterfactual in semantic space: "what is spatially what 'think' is mentally?"
   - Zero out source_category dimensions, fill with target_category
   - Nearest palette entry = the analogy result

4. **compose_chain(edge_a, edge_b)** → composed_edge
   - Multi-hop: "alice through bob" = compose(alice_palette, bob_palette)
   - Result is a new palette index representing the composed concept
   - attention[composed][target] = semantic distance through the chain

5. **causal_semantic_query(edges, pearl_mask, category_mask)** → results
   - Combined Pearl × NsmCategory projection
   - "what entities CAUSED (L2) this outcome, in the SOCIAL dimension?"
   - pearl_mask selects causal rung, category_mask selects semantic plane

### CAM-PQ Stroke Alignment

The 6 CAM-PQ subspaces (HEEL→GAMMA, each 16D) can align with NsmCategory:
  Subspace 0-1: Substantive + Relational (entity identity)
  Subspace 2-3: Mental + Action + Speech (behavior)
  Subspace 4-5: Evaluator + Descriptor + Intensifier (quality)

Stroke cascade = progressive semantic inclusion:
  Stroke 1 (HEEL): entity identity only = interventional (who doesn't matter)
  Stroke 2 (+BRANCH): + behavior = observational
  Stroke 3 (full): all dimensions = complete model

### Base17 Connection (Epiphany A5)

The 17 golden-step dimensions from bgz-tensor projection are the MINIMAL
semantic basis. The 17th dimension (Pythagorean comma) prevents aliasing
between semantic categories. These 17 dimensions could be the canonical
axes: the smallest set that preserves all semantic distinctions.

Mapping: 96D COCA → 17D Base17 → 256 palette → 8-bit CausalEdge64 index
Each stage is a known transformation with measured ρ.
The full chain: ρ ≈ 0.992 × 0.965 ≈ 0.957 end-to-end.

### Tests

- [ ] word_to_edge roundtrip: word → palette → nearest word ≈ original
- [ ] semantic_projection: Mental-only distance("think","know") < Mental-only distance("think","big")
- [ ] cross_domain_analogy: spatial("think") ∈ {explore, search, navigate}
- [ ] compose_chain: compose(alice, bob) closer to carol than to unrelated entity
- [ ] causal_semantic_query: L2 + Mental returns different results than L1 + Spatial
- [ ] Spearman ρ > 0.93 between full AttentionTable and DeepNSM WordDistanceMatrix
- [ ] category_tables sum ≈ full attention table (categories partition the space)

### Epiphany 19: Same Algebra, Three Domains

```
DeepNSM:       symmetric lookup on SEMANTIC categories (16 planes)
CausalEdge64:  symmetric lookup on CAUSAL roles (3 planes, 2³ projections)
bgz-tensor:    symmetric lookup on WEIGHT archetypes (256 palette entries)

All three: precomputed distance table + plane-selective mask + O(1) lookup.
The bridge makes them interoperable: one palette, three mask types, same algebra.
```

This is Epiphany 19 — the realization that DeepNSM, CausalEdge64, and bgz-tensor
are three instances of the same abstract machine. The bridge makes that concrete.

---

## Path 13: burn-adaworld Backend — Wire Our Stack Into Burn's Backend Trait

**Epiphanies**: S1, S2, A3, E19
**Depends on**: Path 1 (SIMD foundation), Path 12 (DeepNSM×CausalEdge64 bridge)
**Effort**: ~8 hours
**Agent**: vector-synthesis
**Repo**: adaworldapi/burn (fork)

### The Approach

Use the burn fork. Wire our CPU SIMD first. See what burn adds later.

burn's Backend trait requires implementing:
  - FloatTensorOps (matmul, add, mul, div, exp, softmax, etc.)
  - IntTensorOps, BoolTensorOps
  - ModuleOps (conv, pool, embedding, etc.)
  - ActivationOps (relu, sigmoid, gelu, etc.)
  - QTensorOps (quantized operations)

### Phase 1: burn-adaworld CPU Backend (~500 lines)

```rust
/// AdaWorld backend: ndarray + crate::simd + LazyLock dispatch.
pub struct AdaWorld;

impl Backend for AdaWorld {
    type Device = CpuDevice;
    type FloatTensorPrimitive = NdArrayTensor<f32>;
    type FloatElem = f32;
    // ...
    
    fn name() -> &'static str { "adaworld-simd" }
}

impl FloatTensorOps<AdaWorld> for AdaWorld {
    fn float_matmul(lhs: FloatTensor, rhs: FloatTensor) -> FloatTensor {
        // Check if bgz-tensor compiled table exists for these dimensions
        if let Some(table) = CompiledAttentionCache::get(lhs.shape(), rhs.shape()) {
            return table_lookup_matmul(lhs, rhs, table);
        }
        // Fall through to ndarray BLAS with crate::simd dispatch
        ndarray_matmul_simd(lhs, rhs)
    }
    
    fn float_exp(tensor: FloatTensor) -> FloatTensor {
        // Use ndarray vml::vsexp (already SIMD via F32x16)
        simd_exp(tensor)
    }
}
```

### Phase 2: bgz-tensor Compiled Attention

```rust
/// When a model is loaded, compile weight matrices into lookup tables.
/// This is the GGUF → bgz-tensor pipeline:
///   f32 weights → Base17 projection → 256 palette → AttentionTable
///
/// After compilation, matmul() for attention becomes O(1) table lookup.
pub struct CompiledAttentionCache {
    heads: HashMap<(usize, usize), AttentionTable>,  // (layer, head) → table
}
```

### Phase 3: Whisper on Our Backend

```rust
// Change one line in whisper-burn:
// BEFORE: type B = burn_tch::TchBackend<f32>;
// AFTER:  type B = burn_adaworld::AdaWorld;

let whisper = Whisper::<AdaWorld>::load(weights_dir, &device);
let text = whisper.transcribe(&audio_samples);
// Now running on AVX-512 SIMD + bgz-tensor table lookup
// No GPU, no PyTorch, no matmul
```

### What Burns Gives Us For Free

By implementing their Backend trait, we get:
  - [ ] Whisper inference (whisper-burn)
  - [ ] MNIST/vision models (burn examples)
  - [ ] Any burn model definition works with our backend
  - [ ] burn-autodiff for fine-tuning on our backend (future)
  - [ ] burn-train for training loops (future)
  - [ ] WASM deployment via burn's existing WASM examples
  - [ ] Model serialization via burn's Record system

### What We Give Burn

  - [ ] LazyLock runtime SIMD dispatch (AVX-512/AVX2 without recompilation)
  - [ ] bgz-tensor O(1) attention (replaces matmul after compilation)
  - [ ] CAM-PQ product quantization (170× compression)
  - [ ] CausalEdge64 causal reasoning (Pearl hierarchy in attention)
  - [ ] NARS truth values on tensor operations (confidence tracking)

### Tests

  - [ ] burn-adaworld passes burn-backend-tests (burn's own test suite)
  - [ ] SIMD operations produce identical results to burn-ndarray backend
  - [ ] bgz-tensor compiled attention matches standard matmul within ρ > 0.95
  - [ ] Whisper transcription quality unchanged with burn-adaworld backend
  - [ ] WASM build works: cargo build --target wasm32-unknown-unknown

### Epiphany 20: Burn Backend Trait = Universal Adapter

burn's Backend trait IS the universal adapter pattern. By implementing it once
with our SIMD + table lookup stack, every burn model instantly runs on our
infrastructure. We don't need to port Whisper, Llama, or any other model —
we port the BACKEND, and all models follow.

This is the same insight as Epiphany 19 (same algebra, three domains) but
at the framework level: one Backend implementation, infinite model compatibility.
