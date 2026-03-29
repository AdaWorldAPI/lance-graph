# SESSION CAPSTONE: Complete Architecture Map + Expansion Plan

> This document is the SINGLE SOURCE OF TRUTH for all sessions.
> Read this FIRST. Everything else is reference material.

## Part 1: What Exists (Validated on Real Data)

### ndarray (adaworldapi/ndarray)
```
Core SIMD:
  src/simd.rs              → crate::simd wrapper (F32x16, U8x64, etc.)
  src/simd_avx512.rs       → AVX-512 intrinsics (stable Rust 1.94)
  src/hpc/simd_caps.rs     → LazyLock<SimdCaps> CPU detection (one-time)
  src/hpc/simd_dispatch.rs → LazyLock<SimdDispatch> frozen fn pointers (7 tests)

DeepNSM (1,318 lines, 23 tests):
  src/hpc/deepnsm.rs       → 74 primes, 300-word vocab, decompose, similarity
                              eval pipeline (Prediction, Explication, SubstitutabilityScore)
                              CAM-PQ bridge (load_nsm_codebook, load_cam_codes)
                              SIMD: F32x16 normalization, U8x64 XOR, F32x16 cosine

CAM-PQ (792 lines):
  src/hpc/cam_pq.rs        → 6-byte fingerprints, stroke cascade, AVX-512 batch distance
                              squared_l2 now uses F32x16 SIMD (16× speedup on hot path)

VML (vectorized math, +900 lines this session):
  src/hpc/vml.rs           → vsexp, vsln, vssqrt, vsabs, vssin, vscos, vspow
                              NEW: vstanh, vsfloor, vsceil, vsround, vstrunc, vsneg
                              BENCHMARKS: F64 hydration, golden-step vs random,
                              photography grid, centroid focus, hotspot bundling,
                              multi-scan NARS, full cascade resonance

GGUF (469 lines, 5 tests):
  src/hpc/gguf.rs          → Parse GGUF header + dequantize F32/F16/BF16/Q8_0

Burn backend (symlink overlay, 3 real files):
  crates/burn/             → upstream burn-ndarray + our SIMD overrides
                              12 ops wired: exp, log, sqrt, abs, sin, cos, sigmoid,
                              floor, ceil, round, trunc, tanh
                              AttentionTable intercept in matmul (O(1) when compiled)
                              30 tests passing

Toolchain: rust-toolchain.toml pins Rust 1.94.0
  → std::f64::consts::PHI and GAMMA available
  → safe #[target_feature] (no unsafe needed)
```

### lance-graph (adaworldapi/lance-graph)
```
DeepNSM crate (3,045 lines, 38 tests, 0 deps):
  crates/deepnsm/          → vocabulary (4,096 + 11K forms), parser (6-state FSM),
                              encoder (512-bit VSA), similarity (256-entry CDF),
                              spo (36-bit triple + 4096² matrix), codebook (CAM-PQ),
                              context (±5 ring buffer), pipeline (DeepNsmEngine)

bgz-tensor crate (1,968 lines, 30 tests, 0 deps):
  crates/bgz-tensor/       → projection (Base17 golden-step), palette (CLAM clustering),
                              attention (AttentionTable 128KB + ComposeTable 64KB),
                              cascade (HHTL 4-layer), quality (Pearson/Spearman ρ)

CausalEdge64 crate:
  crates/causal-edge/      → edge (u64 packed SPO+NARS+Pearl+plasticity+temporal),
                              pearl (2³ projections), plasticity (hot/cold/frozen),
                              tables (precomputed NARS revision/deduction),
                              network (forward_chain, causal_query, counterfactual, Simpson's)

AriGraph (4,339 lines, 6 modules):
  crates/lance-graph/src/graph/arigraph/
    triplet_graph.rs       → full CRUD, BFS, spatial paths, NARS inference, update cycle,
                              production prompts (590w extraction, 400w refinement),
                              normalization, direction mapping, spatial graph
    episodic.rs            → fingerprint-based memory, capacity eviction
    retrieval.rs           → combined BFS + episodic, system prompts for 4 agents
    xai_client.rs          → xAI/Grok API (ADA_XAI env), extraction/refinement/planning
    sensorium.rs           → GraphSensorium from live graph, diagnose_healing, apply_healing
    orchestrator.rs        → MetaOrchestrator + MUL + NARS topology + temperature + auto-heal

Contract (lance-graph-contract):
  src/sensorium.rs         → GraphSignals, GraphBias, HealingType, AgentStyle,
                              OrchestratorMode, SensoriumProvider, OrchestratorProvider traits

Planner (lance-graph-planner):
  src/mul/                 → DK position, trust texture, flow state, compass, gate
  src/thinking/            → ThinkingGraph, 36 styles, sigma chain, topology
  src/elevation/           → L0-L5 cost-sensitive execution
  src/prediction/          → scenario generation, news ingestion, temporal NARS simulation
```

### q2 (adaworldapi/q2)
```
Cockpit server:
  /mri                     → 3D brain MRI (LazyLock pre-render, 500ms refresh)
  /render, /orbit, /flight → demoscene camera paths, pre-baked orbits
  /api/debug/osint         → OSINT pipeline audit (12 stage counters)
  /api/debug/strategies    → live strategy diagnostics (16 × 12 styles)
  /api/orchestrator/*      → meta-orchestrator status + step + quality feedback

Visualization:
  BrainMriMode.tsx         → 5 brain regions, qualia color temperature, real pipeline data
  WaveformMode.tsx         → real OSINT counter deltas, exponential decay
  ConnectomeMode.tsx       → 3D neural connectome (zoom fix landed)

Modules:
  notebook-query/mri.rs         → brain scan computation
  notebook-query/osint_audit.rs → pipeline health
  notebook-query/orchestrator.rs → wire-format mirrors of lance-graph types
  q2-ndarray/deepnsm.rs         → lightweight NSM for notebooks
```


---

## Part 2: 25 Epiphanies (Dependency-Ordered)

### L0 Substrate
- **E1**: Everything = precomputed symmetric lookup table. O(1).
- **E5**: Cascades multiply. 90% × 90% × 90% = 99.9% rejected.

### L1 Encoding
- **E2**: SPO = Attention. Subject=Query, Predicate=Key, Object=Value.
- **E6**: Pack B, Distance A. Every codec packs together, decomposes to compare.
- **E7**: CausalEdge64 = 8 bytes = complete causal unit.
- **E21**: PHI/GAMMA constants = free storage (Rust 1.94).

### L2 Measurement
- **E3**: SimilarityTable (256-entry CDF) calibrates ALL distance types.
- **E4**: NARS confidence = measurement reliability.
- **E9**: Psychometrics validates MEANING. α across 128 projections.
- **E13**: 11K word forms = parallel test items for reliability.
- **E22**: Photography 1/3 grid = structured subsampling (ρ=0.924).

### L3 Composition
- **E8**: Studio mixing. Separate signals before remix. HHTL = frequency bands.
- **E11**: Context window = causal priming via CausalEdge64 + qualia.
- **E23**: Centroid focus = object detection without CNNs (50.5% accuracy).
- **E25**: Scent byte = visual grammar (19 compositions in 1 byte).

### L4 Awareness
- **E10**: Friston free energy. Entropy = fuel. Contradictions = gradient.
- **E12**: Three-way: desired × expected × factual → per-channel learning.
- **E15**: Bias = rotation vector. Unbind to debias.
- **E16**: Study design = orthogonal noise. Bundle, subtract, recover.
- **E18**: NARS contradiction = awareness compass.
- **E24**: Multi-scan = evidence accumulation. Training IS inference.

### L5 Cartography
- **E14**: 2B studies: chaos feeds tensors. Modifier search = meta-analysis.
- **E17**: Qualia detects bias without labels.

### L6 Convergence
- **E19**: Same algebra, three domains. One bridge makes them interoperable.
- **E20**: Burn Backend trait = universal adapter. Implement once → all models.

---

## Part 3: 15 Integration Paths (Status + Dependencies)

### DONE or PARTIAL
```
Path 1  (Foundation):     ndarray builds ✓, SIMD dispatch ✓, 12 burn ops ✓
Path 2  (Contract):       sensorium traits ✓, AriGraph first impl ✓
Path 5  (bgz17):          121 tests ✓ (still excluded from workspace)
Path 8  (DeepNSM↔Ari):   deepnsm 38 tests ✓, entity resolution concept ✓
Path 12 (DeepNSM×Causal): bridge concept ✓, pipeline described
Path 13 (burn backend):   12 SIMD ops ✓, AttentionTable intercept ✓, GGUF loader ✓
```

### READY (components exist, need wiring)
```
Path 3  (ndarray wire):   ndarray dep added in PR#58 ✓, ZeckF64 dedup pending
Path 4  (planner wire):   classify_query wired ✓, full pipeline pending
Path 9  (Causal↔Ari):     causal-edge exists ✓, palette assignment needed
Path 14 (image tensor):   all benchmarks done ✓, wiring needed
Path 15 (photo scans):    concept validated ✓, JIT kernels needed
```

### NOT STARTED
```
Path 6  (n8n-rs):         contract adoption needed
Path 7  (adjacency):      research-grade, AdjacencyView trait needed
Path 10 (psychometrics):  α function not written, concept documented
Path 11 (cartography):    needs Paths 9+10, vision documented
```

---

## Part 4: Expansion Potential

### A. DeepNSM + 36 Thinking Styles = Semantic Awareness Engine

Each thinking style applies a DIFFERENT MASK over the 74 NSM primes.
36 styles × 74 primes = 2,664 semantic perspectives.

```
Integration points:
  lance-graph-planner/src/thinking/style.rs → FieldModulation per style
  ndarray/src/hpc/deepnsm.rs → 74 prime weights per word
  
  WIRE: style.modulation.resonance_threshold → filter which primes pass
        style.modulation.fan_out → how many related concepts to explore
        style.modulation.depth_bias → follow chains deep or scan wide
  
  RESULT: every DeepNSM decomposition gets 36 different interpretations.
  Cronbach's α across 36 perspectives = measurement reliability.
  The style with highest α for this content = the RIGHT way to think about it.
```

### B. GGUF → Transformer + CausalEdge64 = Causality Learning

Extract the causal knowledge that transformers LEARNED during pre-training.

```
Pipeline:
  GGUF model → hpc::gguf::read_tensor_f32() → dequantized weights
  Per attention head:
    Q[4096×128] → Base17 → 256 Q archetypes
    K[4096×128] → Base17 → 256 K archetypes
    V[4096×128] → Base17 → 256 V archetypes
    → AttentionTable[256][256] per head (128KB)
    → ComposeTable[256][256] per head (64KB)
    → Pack as CausalEdge64: Q=Subject, K=Predicate, V=Object
  
  1,024 CausalEdge64 networks (32 layers × 32 heads)
  Each = a different causal perspective on language
  Combined = causal world model extracted from pretrained LLM
  
  MIT CHiLD requirement: 3 independent observations → S, P, O planes ✓
  MIT TRACE requirement: temporal trajectories → CausalEdge64 bits 52-63 ✓
  Pearl do-calculus: intervention → CausalMask bits 40-42 ✓

Total model size: 1,024 × (128KB + 64KB) ≈ 192MB
  vs original GGUF: ~4GB (Q4_K_M)
  vs original f16: ~14GB
  Compression: 20-73×
  AND: the compressed model is INTERPRETABLE + QUERYABLE + LEARNABLE
```

### C. Wikidata as World Knowledge Graph

5.5 billion SPO triples → CausalEdge64 network.

```
Encoding:
  Entity label → DeepNSM tokenize → NSM prime decomposition → palette index
  Property label → DeepNSM tokenize → predicate palette index
  Statement → CausalEdge64: [S_idx, P_idx, O_idx, truth, Pearl, temporal]
  
  Truth from Wikidata:
    rank=preferred → f=0.95, c=0.90
    rank=normal → f=0.70, c=0.60
    rank=deprecated → f=0.10, c=0.80 (high confidence it's WRONG)
    references_count → confidence boost
  
  Qualifiers → temporal/spatial modifiers on CausalEdge64:
    "capital of" + "since 1990" → temporal index in bits 52-63
    "capital of" + "end 1990" → separate edge with lower temporal index

Storage in Lance columns:
  scent:    1B × 5.5B = 5.5GB (HEEL screening)
  palette:  3B × 5.5B = 16.5GB (HIP matching)
  edge:     8B × 5.5B = 44GB (full CausalEdge64)
  Total:    ~66GB with all levels. Fits in server RAM.
  
  Query: "what caused X?" → 
    HEEL scan scent column → 99% rejected
    HIP scan palette → 90% of survivors rejected
    BRANCH read CausalEdge64 → Pearl L2/L3 causal query
    NARS deduction over result set → chain traversal
```

### D. Tensor × OSINT × Reasoning × Semantic Kernel

The synergy matrix:

```
              Tensor    OSINT     Reasoning   Semantic    Embedding
              (burn)    (Ari)     (NARS)      (DeepNSM)   (GGUF→CE64)
Tensor        matmul→   weight→   truth→      prime→      extract→
              table     archetype revision    decompose   causal net
              
OSINT         classify→ observe→  contra→     tokenize→   prior→
              image     triplet   diction     parse SPO   knowledge
              
Reasoning     compose→  deduce→   revise→     nearest→    chains→
              multi-hop new edges evidence    predicate   traverse
              
Semantic      distance→ entity→   α→          vocab→      embed→
              metric    resolve   validate    4096 words  96D COCA
              
Embedding     decode→   correct→  truth→      bridge→     store→
              weights   classify  from GGUF   NSM↔embed   CE64 net

Every cell = a specific integration. 25 cross-connections.
The architecture is a COMPLETE GRAPH — every component talks to every other.
```

### E. Consumer Applications

```
AriGraph (OSINT):
  Ingest news → extract SPO → accumulate in CausalEdge64 network
  36 thinking styles analyze each event from different perspectives
  Contradictions = intelligence gaps → exploration targets
  xAI/Grok for extraction → NARS for reasoning → no human in the loop

q2 (Cockpit):
  /mri shows brain activation per thinking style per query
  /render shows 3D causal graph navigation
  /api/orchestrator controls which style is active
  WaveformMode shows real-time OSINT pipeline activity

burn (ML Models):
  Any burn model runs on our SIMD backend
  Whisper: speech → text → DeepNSM → SPO → knowledge graph
  Vision: image → centroid focus → archetype → SPO → causal edge
  Llama: GGUF → extract causal patterns → CausalEdge64 networks

Wikidata (World Knowledge):
  5.5B triples as CausalEdge64 network
  Every DeepNSM query enriched with world knowledge priors
  NARS deduction expands the graph automatically
  Contradictions with observations = interesting findings
```

---

## Part 5: Benchmark Results (All on Real Data)

### bgz17 Golden-Step Projection
```
Tiny ImageNet (200 images, 12288D → 17D):
  Golden-step: ρ = 0.6476
  Random:      ρ = 0.0806
  Δ = 0.567 (8× better). bgz17 is NOT useless.
```

### Photography Grid Classification
```
4 grid lines 768D:          29.8% accuracy (3× random)
6 grid lines 1152D:         ρ = 0.9264
Centroid focus 432D:        50.5% accuracy (5× random)
Hotspot 8×8 bundle 768D:   43.5% accuracy (4.3× random)
Multi-scan NARS:            51.5% (beats best single scan)
```

### Full Cascade Resonance
```
LEAF   432D  864B   50.5%  ρ=1.000
BRANCH 17D   34B    27.5%  ρ=0.556
HIP    17D   34B    28.0%  ρ=0.556
HEEL   2D    2B     25.0%  ρ=0.180
HEEL rejects 68% of wrong classes at 2 bytes.
```

### F64 Hydration
```
Encode:  51μs (f64[4096] → i16[17])
Hydrate: 79μs (i16[17] → f64[4096])
Compression: 963× (32KB → 34 bytes)
f64 overhead vs f32: effectively zero
```

### Speed Estimates
```
DeepNSM: < 10μs per sentence (tokenize + parse + encode + similarity)
CausalEdge64: < 0.05μs per edge (pack + distance + compose + revise)
Combined: ~10μs per sentence → knowledge graph edge
Throughput: 100,000 sentences/second on one CPU core
```

---

## Part 6: Open Loose Ends (Prioritized)

### P0 (Next Session)
```
L1:  Cronbach's α computation function
L12: CausalEdge64 online learning loop (frame-by-frame)
L7:  Real GGUF benchmark (download TinyLlama, measure ρ on real weights)
```

### P1 (Ready to Wire)
```
L2:  NARS correction matrix as AttentionTable[256][256]
L3:  VSA hyperposition for scene encoding
L8:  ComposeTable for multi-hop visual reasoning
```

### P2 (Needs New Code)
```
L1:  Cronbach's α function in ndarray
L4:  JIT scan kernels from thinking styles
L5:  Diagonal, spiral, multi-scale scan strategies
L9:  Lance columnar storage for cascade columns
```

### P3 (Needs External Data)
```
L6:  CNN features (pretrained ResNet → Base17)
L7:  Real GGUF model (TinyLlama download)
L10: 10Kbit fingerprint for image LEAF level
L11: q2 cockpit visualization of image cascade
     Wikidata dump → CausalEdge64 network (44GB ingest)
```

---

## Part 7: How to Start Each Path

### Starting Path 9 (CausalEdge64 ↔ AriGraph)
```
READ:  EPIPHANIES_COMPRESSED.md (E7, E19, E24)
       session_tensor_codec_vision.md (CausalEdge64 learns while classifying)
DO:    1. Build palette from AriGraph entity names (k-means on DeepNSM embeddings)
       2. Pack each Triplet as CausalEdge64 (palette indices + truth + timestamp)
       3. Build CausalNetwork from packed edges
       4. Test: causal_query() + counterfactual() on AriGraph data
FILES: lance-graph/crates/lance-graph/src/graph/arigraph/triplet_graph.rs
       lance-graph/crates/causal-edge/src/network.rs
```

### Starting Path 10 (Psychometrics)
```
READ:  EPIPHANIES_COMPRESSED.md (E4, E9, E13)
       session_master_integration.md (Phase F)
       session_tensor_codec_vision.md (Cronbach's α section)
DO:    1. Implement cronbachs_alpha(scores: &[&[f64]]) in ndarray hpc/
       2. Compute α across 2³ SPO projections for DeepNSM similarity
       3. Compute α across multi-scan strategies for image classification
       4. Validate: α > 0.7 on COCA data
FILES: ndarray/src/hpc/ (new file: psychometrics.rs)
       ndarray/src/hpc/deepnsm.rs (add α calls to similarity)
```

### Starting Path 11 (Cartography)
```
READ:  EPIPHANIES_COMPRESSED.md (E10, E14, E15, E16, E18)
       session_integration_plan.md (Phase J)
DO:    1. Download 1,000 PubMed abstracts (XML via E-utilities API)
       2. DeepNSM: extract SPO triples per abstract
       3. NARS: detect contradictions between abstracts
       4. Pearl: search for modifiers (temporal, methodological)
       5. Report: modifier map + α + remaining entropy
FILES: lance-graph/crates/lance-graph/src/graph/arigraph/*.rs
       lance-graph/crates/deepnsm/src/pipeline.rs
```

### Starting Path 14 (Image Tensor Codec)
```
READ:  session_tensor_codec_vision.md (EVERYTHING)
       EPIPHANIES_COMPRESSED.md (E22, E23, E24, E25)
DO:    1. Wire centroid focus → DeepNSM SPO extraction
       2. Wire NARS correction matrix as AttentionTable
       3. Wire CausalEdge64 online learning from image frames
       4. Benchmark on full tiny-imagenet (10K images, 200 classes)
FILES: ndarray/src/hpc/vml.rs (benchmark tests)
       lance-graph/crates/deepnsm/src/pipeline.rs
       lance-graph/crates/causal-edge/src/network.rs
```

### Starting Wikidata Integration
```
READ:  This document Part 4C (Wikidata section)
       EPIPHANIES_COMPRESSED.md (E1, E5, E7, E18)
DO:    1. Download Wikidata JSON dump (latest-all.json.bz2, ~80GB compressed)
       2. Stream-parse: extract (subject_label, property_label, object_label) triples
       3. DeepNSM tokenize labels → palette indices
       4. Pack as CausalEdge64 (truth from rank + references)
       5. Store in Lance columns (scent, palette, edge)
       6. Test: "what is the capital of Germany?" → cascade query → Berlin
FILES: NEW: lance-graph/crates/lance-graph/src/wikidata/ (ingest module)
       lance-graph/crates/deepnsm/src/vocabulary.rs (tokenize labels)
       lance-graph/crates/causal-edge/src/edge.rs (pack triples)
```

---

## Part 8: Cross-Reference to Existing Docs

```
EPIPHANIES_COMPRESSED.md         → 25 epiphanies in 40 lines (read FIRST)
session_unified_26_epiphanies.md → full detail + 15 paths + 5 agents + QA protocol
session_master_integration.md    → 6 layers + 42 hours + cartography test
session_integration_plan.md      → Phases A-J technical tasks
session_tensor_codec_vision.md   → vision pipeline + benchmarks + loose ends
session_deepnsm_cam.md           → DeepNSM-CAM architecture (the original blueprint)
session_deepnsm_compile.md       → deepnsm crate compile/test handover
session_bgz_tensor.md            → bgz-tensor compile/test handover
session_arigraph_transcode.md    → AriGraph Python → Rust transcode spec
session_thinking_topology.md     → thinking pipeline + YAML + topology learning

docs/INTEGRATION_DEBT_AND_PATHS.md → PR#50 audit (8 strengths, 9 weaknesses)
docs/META_INTEGRATION_PLAN.md      → 8-layer stack, 3 memory layers
docs/CODEC_COMPRESSION_ATLAS.md    → full→zeckbf17→bgz17→cam-pq→scent chain
docs/TYPE_DUPLICATION_MAP.md       → 40+ duplicated types with file:line
docs/SEMIRING_ALGEBRA_SURFACE.md   → all 14 semirings across 4 repos
docs/THINKING_MICROCODE.md         → YAML→JIT→LazyLock→NARS RL
```

---

## Part 9: Path 16 — Vocabulary Expansion via Machine-Readable Wordlists

**Source**: github.com/lpmi-13/machine_readable_wordlists
**Depends on**: Path 8 (DeepNSM↔AriGraph), Path 11 (Cartography)
**Effort**: ~6 hours
**Agent**: arigraph-osint + vector-synthesis

### The Opportunity

DeepNSM covers 98.4% of running English text with 4,096 COCA words.
But OSINT, medical, cyber, and scientific domains use specialized vocabulary
that falls in the 1.6% gap. These wordlists fill that gap with
CURATED domain-specific terms — not random OOV words.

### Phase 1: BNC/COCA Extension (4,096 → 25,000)

```
Source:    BNC/COCA Lists (29 JSON files, 25 × 1,000-word frequency lists)
Format:   JSON/YML, already frequency-ranked
Effort:   2 hours
Approach: Same corpus family as our existing 4,096 — distributional vectors
          are COMPATIBLE. Just load additional words + compute prime weights.
          
  Current: word_rank_lookup.csv (5,050 entries, top 4,096 used)
  Extended: BNC/COCA 25K → 25,000 entries with frequency + PoS
  
  The 12-bit vocabulary index (4,096 max) becomes 15-bit (32,768 max).
  SpoTriple: [S:15][P:15][O:15] = 45 bits, fits in u64 (was 36 bits).
  CausalEdge64: S/P/O palette indices stay 8-bit (256 archetypes).
  The palette handles the compression: 25K words → 256 archetypes.
```

### Phase 2: Domain-Specific OSINT Vocabularies

```
Priority domain lists (all JSON, machine-readable):

  Newspaper Word List (NWL):     588 families
    → OSINT: deploy, sanction, treaty, alliance, regime, insurgent
    → Maps to NSM: [Do, Bad, Not, Can, Someone, Place, Time]
    
  Medical Academic (MAWL):       623 headwords
    → Health OSINT: pathogen, epidemic, vaccine, transmission, mortality
    → Maps to NSM: [Die, Live, Body, Bad, Many, Someone]
    
  Computer Science:              433 headwords + 23 multi-word
    → Cyber OSINT: vulnerability, encryption, breach, malware, protocol
    → Maps to NSM: [Bad, Thing, Inside, Not, Can, Do]
    
  Business English (BEAWL):      415 headwords
    → Financial OSINT: acquisition, compliance, dividend, leverage
    → Maps to NSM: [Mine, Much, Do, Want, More, Big]
    
  Science Jargon:                ~500 terms
    → Scientific OSINT: correlation, hypothesis, variable, significant
    → Maps to NSM: [Think, True, Maybe, Because, Like, Know]
    
  Engineering (EEWL):            729 families
    → Technical OSINT: specification, tolerance, calibration, throughput
    → Maps to NSM: [Do, Thing, Kind, Part, Good, Much]

  Total: ~3,288 domain terms not in COCA top 4K
  Combined with BNC/COCA 25K: covers ~99.5% of domain text
```

### Phase 3: NSM Prime Weight Computation for New Words

```
For each new word, compute 74 prime weights automatically:

  Method 1 (if COCA distributional vector available):
    Load 96D vector from subgenres_5k.csv or BNC/COCA frequency data
    Project through DeepNSM's existing decomposition
    → prime weights from distributional statistics
    
  Method 2 (if no distributional vector):
    Use DeepNSM's existing vocabulary to APPROXIMATE:
    "sanction" → nearest known words: "punish" (0.7), "law" (0.5), "stop" (0.6)
    → weighted average of their prime decompositions
    → sanction_primes ≈ 0.7 × punish_primes + 0.5 × law_primes + 0.6 × stop_primes
    
  Method 3 (via xAI/Grok):
    Ask: "decompose 'sanction' into NSM semantic primes"
    → LLM-assisted prime weight assignment
    → validate via Cronbach's α against Method 1/2
    
  All three methods produce the same shape: [f32; 74] per word.
  Cross-validation: methods that agree have high α → reliable decomposition.
```

### Phase 4: Wikidata Entity Resolution Enhancement

```
ICE-CORE (7 English varieties × ~1,000 words):
  "colour" (UK) = "color" (US) → same entity
  "lorry" (UK) = "truck" (US) → same entity
  
  For Wikidata ingestion: entity labels vary by English variety.
  The ICE-CORE wordlist provides cross-variety mapping.
  DeepNSM: dist(colour, color) should be ≈ 0 after variety normalization.

Academic Spoken (1,741 word families at 4 proficiency levels):
  Wikidata descriptions use academic vocabulary.
  "photosynthesis" is in ASWL but not COCA top 4K.
  With ASWL loaded: DeepNSM can parse Wikidata science descriptions.

Secondary Vocabulary List (8 subjects):
  biology, chemistry, economics, English, geology, history, math, physics
  Each subject's terms help classify Wikidata entities BY DOMAIN.
  "mitosis" → biology. "valence" → chemistry. "GDP" → economics.
  This IS the domain classifier for Wikidata entity typing.
```

### Phase 5: Integration with 36 Thinking Styles

```
Each domain wordlist aligns with specific thinking style clusters:

  NWL (newspaper)    → Analytical, Systematic (factual reporting)
  MAWL (medical)     → Systematic, Convergent (evidence-based)
  CS wordlist        → Analytical, Focused (technical precision)
  BEAWL (business)   → Pragmatic, Convergent (outcome-focused)
  Science Jargon     → Exploratory, Metacognitive (hypothesis testing)
  EEWL (engineering) → Systematic, Focused (specification-driven)

  When the MetaOrchestrator detects domain-specific vocabulary in the input
  (via DeepNSM tokenization), it ACTIVATES the corresponding thinking style
  cluster automatically. Medical text → Systematic. News → Analytical.
  
  This IS the MODULATE cognitive verb: content drives thinking mode.
```

### Implementation Steps

```
1. git clone https://github.com/lpmi-13/machine_readable_wordlists /tmp/wordlists
2. Parse JSON: extract (word, pos, frequency, domain) per list
3. Merge with existing COCA vocabulary (deduplicate by lemma)
4. Compute NSM prime weights for new words (Method 1/2/3)
5. Update DeepNSM vocabulary.rs to load extended vocabulary
6. Update SpoTriple to 15-bit indices (45-bit total, still fits u64)
7. Rebuild palette: k-means on expanded vocabulary → 256 archetypes
8. Test: domain-specific text → correct tokenization → correct SPO
9. Benchmark: coverage % on OSINT/medical/cyber/scientific text samples
```

### Expected Impact

```
Coverage improvement:
  Current: 98.4% of general English text
  With BNC/COCA 25K: ~99.2% of general English
  With domain lists: ~99.5% of domain-specific text
  
OSINT improvement:
  "Country X deployed Y" → "deploy" now in vocabulary (was OOV)
  → DeepNSM parses correctly → SPO(X, deploy, Y)
  → CausalEdge64 with proper predicate (not fallback)
  
Wikidata improvement:
  Scientific entity descriptions → parseable with academic vocabulary
  Cross-variety entity resolution → "colour"="color" normalized
  Domain classification → SVL subject lists → entity type detection
```

### Files to Modify

```
lance-graph/crates/deepnsm/src/vocabulary.rs  → extended loading
lance-graph/crates/deepnsm/src/spo.rs         → 15-bit indices
lance-graph/crates/deepnsm/src/pipeline.rs    → load domain lists
ndarray/src/hpc/deepnsm.rs                    → extended prime weights
NEW: lance-graph/crates/deepnsm/data/         → domain wordlist JSON files
```
