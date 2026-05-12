# MASTER INTEGRATION: 18 Epiphanies → 6 Layers → Implementation Map

## The Insight Dependency Tree

Epiphanies aren't independent — they form layers where each enables the next.
This ordering IS the implementation plan: build bottom-up, each layer
unlocks the one above.

```
LAYER 5: CARTOGRAPHY (2B studies, automated meta-analysis)
  E14: Chaos feeds tensors (contradictions = exploration targets)
  E17: Qualia-based bias detection (no human labeling needed)
  E18: NARS contradiction awareness = navigation compass for science
  ↑ requires L4

LAYER 4: AWARENESS (self-regulated thinking about thinking)
  E10: Friston free energy (entropy = potential, not waste)
  E12: Three-way outcome (desired × expected × factual → learning signal)
  E15: Bias as rotation vector (observer/study/funding = known transforms)
  E16: Study design as orthogonal noise (bundle, subtract, recover signal)
  ↑ requires L3

LAYER 3: COMPOSITION (building complex structures from primitives)
  E8:  Studio mixing / vertical HHTL bundling (separate before remix)
  E11: Autocomplete as priming (±5 context + qualia + CausalEdge64 chains)
  ↑ requires L2

LAYER 2: MEASUREMENT (validating that numbers mean something)
  E3:  SimilarityTable unifies scoring (one CDF → f32 [0,1])
  E4:  NARS truth = reliability (confidence IS measurement quality)
  E9:  Psychometric framework (α, IRT, factor analysis, p-values)
  E13: 11K word forms = parallel test items (inflection = measurement invariance)
  ↑ requires L1

LAYER 1: ENCODING (representing knowledge in computable form)
  E2:  SPO = Attention (Subject=Query, Predicate=Key, Object=Value)
  E6:  Pack B / Distance A (universal codec pattern across all components)
  E7:  CausalEdge64 = 8 bytes (SPO + NARS + Pearl + temporal + plasticity)
  ↑ requires L0

LAYER 0: SUBSTRATE (the physical computation layer)
  E1:  Everything is a lookup table (precompute, then serve)
  E5:  Cascades compose multiplicatively (< 0.001% survives full pipeline)
```

---

## Layer 0 → Implementation: SUBSTRATE

### What exists
- [x] LazyLock<SimdDispatch> frozen function pointer table
- [x] crate::simd wrapper (F32x16, U8x64) — consumer never sees hardware
- [x] cam_pq squared_l2 SIMD (1,536× per query, now F32x16)
- [x] HHTL cascade in bgz17, bgz-tensor, cam_pq, deepnsm

### What's missing
- [ ] cam_pq encode() SIMD (6×256 centroid search)
- [ ] cam_pq precompute_distances() SIMD
- [ ] blas_level1 iamax() SIMD
- [ ] Safe #[target_feature] (Rust 1.94, remove unsafe)
- [ ] wasm32 SIMD tier (std::arch::wasm32 v128)
- [ ] jitson_cranelift dedup (use simd_caps() not raw detection)
- [ ] Cascade rejection rate → GraphSensorium (discrimination metric)

### Tests needed
- [ ] SIMD produces identical results to scalar on all operations
- [ ] LazyLock dispatch selects correct tier on this CPU
- [ ] Cascade rejection > 90% per level on representative data

---

## Layer 1 → Implementation: ENCODING

### What exists
- [x] deepnsm SpoTriple [S:12][P:12][O:12] in u64
- [x] CausalEdge64 [S:8][P:8][O:8][NARS:16][Pearl:3][Dir:3][Inf:3][Plast:3][T:12]
- [x] SpoBase17 (3 × Base17, Strategy A pure)
- [x] VsaVec (512-bit, role-tagged bundle, Strategy B pure)
- [x] CausalMask as A↔B bridge

### What's missing
- [ ] Palette assignment: AriGraph entities → 256 archetypes for CausalEdge64
- [ ] TripletGraph → CausalNetwork conversion
- [ ] deepnsm SpoTriple → bgz-tensor CompiledHead bridge (same palette)
- [ ] Cross-codec distance correlation verification (A vs B should rank-correlate)

### Tests needed
- [ ] Round-trip: Triplet → CausalEdge64 → back, no information loss
- [ ] SPO distance matches across all encodings (SpoBase17, CausalEdge64, SpoTriple, VsaVec)
- [ ] Spearman ρ > 0.93 between any two encoding distances for same pair

---

## Layer 2 → Implementation: MEASUREMENT

### What exists
- [x] SimilarityTable (256-entry CDF) in deepnsm
- [x] NARS truth values on every AriGraph triplet
- [x] NARS 7 inference rules in ndarray (deduction, abduction, induction, etc.)
- [x] NARS inference wired into TripletGraph (infer_deductions, detect_contradictions, revise_with_evidence)
- [x] 11K word forms loaded in deepnsm vocabulary.rs

### What's missing
- [ ] Cronbach's α computation across 2³ SPO projections
- [ ] Split-half reliability: Strategy A distance vs Strategy B distance
- [ ] IRT item parameters: per-word difficulty + discrimination
- [ ] Factor analysis: do 74 primes factor into 16 NsmCategory?
- [ ] Polysemy detection via α drop across projections
- [ ] Measurement invariance: cam_pq vs bgz17 vs raw 96D L2
- [ ] 11K forms consistency check (parallel forms reliability)
- [ ] SimilarityTable dedup (currently 3 copies across repos)

### Tests needed
- [ ] α > 0.7 for NSM prime decomposition on COCA data
- [ ] Inflected forms produce consistent decompositions (test-retest)
- [ ] Same pair ranked identically by all 3 distance codecs

---

## Layer 3 → Implementation: COMPOSITION

### What exists
- [x] AriGraph ±5 context as VsaVec ring buffer
- [x] 16-channel qualia coloring (luminance through surprise)
- [x] Qualia-driven causality (warmth, social, sacredness → triune brain)
- [x] bgz-tensor ComposeTable (multi-hop O(1))

### What's missing
- [ ] Vertical HHTL bundling (leaf → twig → branch → hip)
- [ ] Bottom-up unbinding with reconstruction error measurement
- [ ] CausalEdge64 chains in ±5 context window (not just VsaVec)
- [ ] Qualia priming: running qualia accumulation biases disambiguation
- [ ] qualia_cam_key() as priming fingerprint in context window
- [ ] BF16 noise floor removal before HEEL bundling
- [ ] Studio mixing: separate per-level signals before remix

### Tests needed
- [ ] Bundle → unbundle roundtrip: Hamming error < 15% per level
- [ ] Context priming: "bank" disambiguated by qualia of surrounding sentences
- [ ] ComposeTable multi-hop matches BFS result with ρ > 0.95

---

## Layer 4 → Implementation: AWARENESS

### What exists
- [x] GraphSensorium with 6 signals (contradiction_rate, truth_entropy, etc.)
- [x] MetaOrchestrator with MUL (DK, trust, flow, compass)
- [x] NARS topology learning (4×4 style transitions)
- [x] Temperature / stagnation detection / auto-heal
- [x] Epiphany/blunder concept documented

### What's missing
- [ ] OutcomeTriad: desired × expected × factual (QualiaVector each)
- [ ] Three deltas: Δ_surprise, Δ_satisfaction, Δ_aspiration (per-channel)
- [ ] Epiphany detector: |Δ_surprise| > threshold → metacognition trigger
- [ ] Friston free energy proxy: F ≈ truth_entropy - evidence_rate
- [ ] Entropy-as-potential: high entropy + low epiphany = exploration target
- [ ] Contradiction-as-modifier search:
    - For each contradiction, search temporal/spatial/causal modifiers
    - Pearl L1→L2→L3 as progressive modifier application
    - Finding modifier = epiphany (entropy drop = F drop)
- [ ] Bias rotation vectors:
    - Selection bias, confirmation bias, publication bias, funding bias
    - Each as a known vector in 96D distributional space
    - Unbind to debias (XOR is self-inverse)
- [ ] Study design as orthogonal noise:
    - Design flaws consistent across studies → accumulate in DESIGN plane
    - Causal content varies → averages out unless real
    - Unbind DESIGN plane → recovered causal signal
- [ ] Progressive debiasing via Pearl hierarchy:
    - L1 → L2: unbind(study_bias) = intervention estimate
    - L2 → L3: unbind(observer_bias) = counterfactual (bias-free)
- [ ] Qualia-based bias detection:
    - Low SURPRISE + high AGENCY = funding bias fingerprint
    - High FAMILIARITY + low SURPRISE = confirmation bias fingerprint
    - Qualia coloring reveals bias without human labeling
- [ ] Self-reinforcement loop:
    - Observation → decomposition → prediction → action → comparison → learning
    - Per-channel learning: SOCIAL surprise → adjust social context weight
    - Epiphany → temperature spike → exploration burst
    - Blunder → consolidation → contradiction detection

### Tests needed
- [ ] Epiphany detection on synthetic contradiction resolution
- [ ] Bias rotation: inject known bias, recover signal, measure quality
- [ ] Free energy decreases monotonically during productive learning sessions
- [ ] Qualia fingerprint correctly identifies bias type (blind test)

---

## Layer 5 → Implementation: CARTOGRAPHY

### What exists
- [x] AriGraph xAI client for OSINT extraction
- [x] Production prompts (590w extraction, 400w refinement)
- [x] Full update cycle (extract → refine → delete → spatial)
- [x] Entity resolution concept (DeepNSM similarity threshold)

### What's missing
- [ ] Batch ingestion pipeline (1K+ documents per run)
- [ ] Cross-document contradiction mapping
- [ ] Modifier discovery at scale (temporal, spatial, methodological)
- [ ] Contradiction landscape visualization (entropy heat map)
- [ ] Research priority ranking (highest-entropy contradictions first)
- [ ] Automated meta-analysis workflow:
    1. Ingest N studies as SPO triples
    2. Detect contradictions (NARS)
    3. Search for modifiers (Pearl L1/L2/L3)
    4. Rotate out known biases
    5. Compute reliability of findings (α)
    6. Report: modifier map + reliability + remaining entropy
- [ ] The Cartography Test:
    1,000 abstracts → contradictions → modifiers → α > 0.7 → under 10 seconds

### Tests needed
- [ ] 100 synthetic abstracts: extract, detect, resolve, measure α
- [ ] Known contradiction pairs: modifier search finds the correct qualifier
- [ ] Bias injection/recovery roundtrip: signal quality measured

---

## Cross-Layer Integration Points

| From | To | Mechanism | Status |
|------|----|-----------|--------|
| L0 SIMD | L1 Encoding | crate::simd in all distance computations | PARTIAL |
| L0 Cascade | L2 Measurement | rejection rate = item discrimination | NOT STARTED |
| L1 CausalEdge64 | L3 Context | CausalEdge64 chains in ±5 window | NOT STARTED |
| L1 SpoTriple | L4 Awareness | contradiction detection on SPO pairs | DONE |
| L2 Psychometrics | L4 Awareness | α = reliability = NARS confidence | NOT STARTED |
| L2 SimilarityTable | L3 Composition | calibrated CDF for all comparisons | DONE |
| L3 Qualia | L4 Awareness | bias detection via qualia fingerprint | NOT STARTED |
| L3 HHTL bundling | L2 Psychometrics | per-level reliability = per-level α | NOT STARTED |
| L4 Free energy | L5 Cartography | highest-entropy contradictions = priority | NOT STARTED |
| L4 Bias rotation | L5 Cartography | debias before meta-analysis | NOT STARTED |
| L5 Modifier map | L4 Awareness | found modifiers reduce entropy → epiphany | NOT STARTED |
| L5 Scale | L0 Cascade | 2B studies × 99.9% rejection = tractable | NOT STARTED |

---

## Estimated Effort Per Layer

| Layer | Hours | Depends On | Key Deliverable |
|-------|-------|------------|-----------------|
| L0 Substrate | 6 | nothing | Full SIMD coverage via crate::simd |
| L1 Encoding | 4 | L0 | TripletGraph ↔ CausalNetwork bridge |
| L2 Measurement | 6 | L1 | Cronbach's α + IRT + factor analysis |
| L3 Composition | 8 | L2 | Vertical HHTL bundling + qualia priming |
| L4 Awareness | 10 | L3 | OutcomeTriad + bias rotation + free energy |
| L5 Cartography | 8 | L4 | 1,000 abstracts → modifier map → α > 0.7 |
| **TOTAL** | **42** | | |

---

## The Ultimate Test

Load 1,000 real PubMed abstracts on a controversial topic.
The system should:

1. Extract SPO triples (DeepNSM, < 10μs each) → L1
2. Detect contradictions (NARS, < 1ms) → L2
3. Measure reliability of each position (α across projections) → L2
4. Search for modifiers (Pearl L1/L2/L3) → L4
5. Detect and rotate out study biases (qualia + VSA unbind) → L4
6. Produce a modifier map (which conditions make each position true) → L5
7. Report: remaining entropy, top contradictions, research priorities → L5
8. All in under 10 seconds on a single CPU core
9. With measurement reliability α > 0.7

If this works for 1,000 abstracts, the 2 billion study cartography
is a matter of cascade engineering (L0) — not algorithmic breakthrough.

The chaos feeds the tensors. The contradictions are the compass.
NARS is the awareness that makes it navigable.
