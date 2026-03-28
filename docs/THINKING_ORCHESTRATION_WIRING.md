# Thinking Orchestration Wiring: From YAML to Native Kernel

> *The complete end-to-end path of a thought through the system.*

## 1. The Full Pipeline (Bird's Eye)

```
YAML Template (crewai-rust agent card)
  │
  ▼
23D Sparse Vector (JitProfile)
  │
  ▼
ThinkingStyle (36 enum variants, 6 clusters)
  │
  ├──▶ τ (tau) address (0x20-0xC5) ──▶ BindSpace prefix 0x0D (thinking templates)
  │
  ▼
FieldModulation (7D: resonance, fan_out, depth, breadth, noise, speed, explore)
  │
  ├──▶ ScanParams (threshold, top_k, prefetch_ahead, filter_mask)
  │       │
  │       ▼
  │    JitTemplate (JITSON format JSON)
  │       │
  │       ▼
  │    Cranelift IR → native x86-64 with AVX-512
  │       │
  │       ▼
  │    KernelHandle (fn ptr, param_hash, avx512 flag)
  │       │
  │       ▼
  │    ScanKernel (native function pointer, cached by τ address)
  │
  ▼
MUL Assessment (Dunning-Kruger + Trust + Flow + Compass)
  │
  ├──▶ GateDecision: Proceed / Sandbox / Compass
  │
  ▼
ThinkingContext (style + modulation + NARS type + semiring + sigma stage)
  │
  ▼
Strategy Selection (16 strategies score affinity against ThinkingContext)
  │
  ▼
ElevationLevel (L0:Point → L1:Scan → L2:Cascade → L3:Batch → L4:IVF → L5:Async)
  │
  ▼
Physical Plan (CamPqScanOp / ScanOp / CollapseOp / AccumulateOp)
  │
  ▼
Results (RecordBatch or Vec<(index, distance)>)
```

## 2. Layer-by-Layer Detail

### 2A. Layer 0: Agent Card → JitProfile (crewai-rust)

crewai-rust agents are defined in YAML cards. Each card specifies a `thinking_style` as a 23D sparse vector.

```yaml
# Example agent card
agent:
  name: "analyst"
  thinking_style:
    depth: 0.9          # dim 0
    breadth: 0.2        # dim 1
    noise_tolerance: 0.1 # dim 2
    speed: 0.3          # dim 3
    exploration: 0.1    # dim 4
    focus: 0.95         # dim 5
    # ... 18 more dimensions
  capabilities: [cypher, resonance, cam_pq]
```

**Location**: crewai-rust agent cards → `JitProfile` struct
**Contract**: `lance-graph-contract/src/thinking.rs:233` — `SparseVec = Vec<(usize, f32)>`

### 2B. Layer 1: JitProfile → ThinkingStyle (contract)

The 23D vector maps to one of **36 ThinkingStyles** in 6 clusters:

| Cluster | τ Range | Styles | Planner Cluster |
|---------|---------|--------|----------------|
| **Analytical** | 0x40-0x45 | Logical, Analytical, Critical, Systematic, Methodical, Precise | Convergent |
| **Creative** | 0xA0-0xA5 | Creative, Imaginative, Innovative, Artistic, Poetic, Playful | Divergent |
| **Empathic** | 0x80-0x85 | Empathetic, Compassionate, Supportive, Nurturing, Gentle, Warm | Attention |
| **Direct** | 0x60-0x65 | Direct, Concise, Efficient, Pragmatic, Blunt, Frank | Convergent |
| **Exploratory** | 0x20-0x25 | Curious, Exploratory, Questioning, Investigative, Speculative, Philosophical | Divergent |
| **Meta** | 0xC0-0xC5 | Reflective, Contemplative, Metacognitive, Wise, Transcendent, Sovereign | Speed |

**Source of truth**: `lance-graph-contract/src/thinking.rs:23` — `ThinkingStyle` enum (36 variants)
**Parallel copy**: `lance-graph-planner/src/thinking/style.rs:16` — 12 styles (NOT using contract)
**n8n-rs copy**: `n8n-contract/src/thinking_mode.rs` — ThinkingMode struct (different shape)

### 2C. Layer 2: ThinkingStyle → FieldModulation (7D Control Knobs)

Each style has a default 7D FieldModulation that controls planner behavior:

| Style | resonance | fan_out | depth | breadth | noise | speed | explore |
|-------|-----------|---------|-------|---------|-------|-------|---------|
| Analytical | 0.85 | 4 | 0.9 | 0.2 | 0.1 | 0.3 | 0.1 |
| Creative | 0.50 | 12 | 0.4 | 0.9 | 0.7 | 0.6 | 0.8 |
| Exploratory | 0.30 | 20 | 0.3 | 1.0 | 0.9 | 0.5 | 1.0 |
| Focused | 0.90 | 2 | 1.0 | 0.1 | 0.05 | 0.4 | 0.05 |
| Intuitive | 0.60 | 8 | 0.5 | 0.5 | 0.5 | 0.9 | 0.4 |
| Deliberate | 0.75 | 6 | 0.7 | 0.4 | 0.2 | 0.2 | 0.2 |
| Metacognitive | 0.70 | 8 | 0.6 | 0.6 | 0.3 | 0.3 | 0.5 |

**FieldModulation serializes to a 7-byte thermometer-coded fingerprint** stored at BindSpace prefix 0x0D.

### 2D. Layer 3: FieldModulation → ScanParams → JIT Kernel

```
FieldModulation.to_scan_params():
  threshold     = resonance × 2000 + 100    (100..2100)
  top_k         = fan_out                    (1..128)
  prefetch_ahead = (1 - speed) × 7 + 1      (1..8)
  filter_mask   = noise < 0.3 ? 0xFFFFFFFF : 0

ScanParams → JitTemplate:
  JSON format consumed by ndarray jitson engine
  Cranelift compiles threshold/top_k as IMMEDIATES (no memory fetch)
  focus_mask → VPANDQ bitmask
  prefetch → PREFETCHT0 offset

JitTemplate → KernelHandle:
  Cranelift IR → native x86-64
  Cached by τ address + param_hash
  avx512 flag from simd_caps() singleton
```

**Contract**: `lance-graph-contract/src/jit.rs:48` — `JitCompiler` trait
**Implementation**: `ndarray/src/hpc/jitson_cranelift/engine.rs` — `JitEngine`
**Cache**: `n8n-rs/n8n-contract/src/compiled_style.rs:38` — `CompiledStyle`

### 2E. Layer 4: MUL Assessment (Meta-Uncertainty Layer)

Before any planning, MUL assesses whether planning should proceed:

```
SituationInput (13 fields: felt_competence, demonstrated_competence, ...)
     │
     ├──▶ DK Detector → DkPosition
     │      MountStupid:          felt >> demonstrated → DANGEROUS
     │      ValleyOfDespair:      aware of gaps
     │      SlopeOfEnlightenment: building competence
     │      PlateauOfMastery:     expert
     │
     ├──▶ Trust Qualia → TrustTexture
     │      Crystalline: well-calibrated
     │      Solid: reliable
     │      Fuzzy: uncertain
     │      Murky: contradictory
     │      Dissonant: conflicting signals
     │
     ├──▶ Homeostasis → FlowState
     │      Flow:    challenge ≈ skill (optimal)
     │      Anxiety: challenge > skill
     │      Boredom: skill > challenge
     │      Apathy:  neither
     │
     └──▶ GATE CHECK
            Proceed:  DK not MountStupid, complexity mapped, not depleted
            Sandbox:  Need human assistance
            Compass:  Unknown territory → 5 compass needles:
                      KANT, ANALOGY, IDENTITY, REVERSIBILITY, CURIOSITY
```

**DK Position → ThinkingStyle selection**:
| DK Position | Selected Style | Rationale |
|-------------|---------------|-----------|
| MountStupid | Metacognitive | Force self-reflection |
| ValleyOfDespair | Systematic | Careful, methodical |
| SlopeOfEnlightenment + Flow | Analytical | Precise, productive |
| SlopeOfEnlightenment + Anxiety | Deliberate | Slow down |
| SlopeOfEnlightenment + Boredom | Creative | Engage novelty |
| PlateauOfMastery + Crystalline trust | Intuitive | Trust the gut |
| PlateauOfMastery + Solid trust | Analytical | Standard operation |
| PlateauOfMastery + Fuzzy trust | Exploratory | Investigate uncertainty |

### 2F. Layer 5: NARS Inference Type → Query Strategy → Semiring

```
Query text analysis → NarsInferenceType:
  "Find X"    → Deduction  → CamExact         → Boolean semiring
  "What is"   → Induction  → CamWide          → HammingMin semiring
  "Why did"   → Abduction  → DnTreeFull       → TruthPropagating semiring
  "Update"    → Revision   → BundleInto        → TruthPropagating semiring
  "Connect"   → Synthesis  → BundleAcross      → XorBundle semiring
```

**Semiring auto-selection** (`planner/thinking/semiring_selection.rs`):
1. RESONATE/HAMMING/SIMILARITY → XorBundle
2. NARS Revision/Synthesis → TruthPropagating
3. SHORTEST PATH/DISTANCE → Tropical
4. EXISTS/ANY/ALL → Boolean
5. FINGERPRINT WHERE → HammingMin
6. Creative/Exploratory style → XorBundle
7. Default → Boolean

### 2G. Layer 6: Sigma Chain (Epistemic Lifecycle)

Each thinking atom progresses through 5 stages:

```
Ω (OMEGA)    → Observation: grounded in direct experience (confidence 0.9)
Δ (DELTA)    → Insight/Hypothesis: abductive leap (confidence 0.5)
Φ (PHI)      → Belief: evaluated proposition with frequency + confidence
Θ (THETA)    → Integration: synthesizes multiple beliefs
Λ (LAMBDA)   → Trajectory: action tendency or prediction
```

**Location**: `lance-graph-planner/src/thinking/sigma_chain.rs`
**From**: bighorn/extensions/agi_stack thinking_atom.py

### 2H. Layer 7: Strategy Selection (16 Composable Strategies)

Three selection modes:
1. **Explicit**: User names strategies → `"cypher_parse, sigma_scan, jit_compile"`
2. **Auto**: Each strategy scores affinity → top-N per phase
3. **Resonance**: ThinkingStyle 23D vector → cosine similarity with strategy profiles

The 16 strategies, by phase:

| Phase | # | Strategy | Source | Affinity Signal |
|-------|---|----------|--------|----------------|
| **Parse** | 1 | CypherParse | lance-graph nom | `MATCH`, `RETURN` |
| | 2 | GremlinParse | TinkerPop | `g.V()`, `.hasLabel(` |
| | 3 | SparqlParse | W3C | `PREFIX`, `SELECT ?` |
| | 4 | GqlParse | ISO 39075 | `LEFT MATCH`, `ANY SHORTEST`, `TRAIL` |
| **Plan** | 5 | ArenaIR | Polars arena | Any parsed query |
| | 6 | DPJoinEnum | KuzuDB DP | Multi-hop patterns (2+ MATCH) |
| **Optimize** | 7 | RuleOptimizer | DataFusion | Any logical plan |
| | 8 | HistogramCost | Hyrise | Cost estimation needed |
| **Physicalize** | 9 | SigmaBandScan | lance-graph | RESONATE, FINGERPRINT |
| | 10 | MorselExec | KuzuDB/Polars | Large cardinality |
| | 11 | TruthPropagation | lance-graph semiring | TRUTH, CONFIDENCE |
| | 12 | CollapseGate | agi-chat | Resonance gating |
| **Execute** | 13 | StreamPipeline | Polars | Streaming results |
| | 14 | JitCompile | ndarray | Compiled kernel available |
| **Cross** | 15 | WorkflowDAG | LangGraph | WORKFLOW, TASK |
| | 16 | ExtensionPlanner | DataFusion | Custom extensions |

### 2I. Layer 8: Dynamic Elevation (Cost Model That Smells Resistance)

Instead of guessing cost upfront, start cheap and **elevate** on observed resistance:

| Level | Name | Typical Latency | What Triggers Elevation |
|-------|------|----------------|------------------------|
| L0 | Point | ~100ns | Single lookup in BindSpace |
| L1 | Scan | ~10μs | palette_distance over neighborhood |
| L2 | Cascade | ~100μs | CAM-PQ 3-stroke (99% rejection) |
| L3 | Batch | ~10ms | Morsel pipeline with backpressure |
| L4 | IVF+Batch | ~100ms | IVF partition → batch per partition |
| L5 | Async | seconds+ | Fire-and-forget, results via SSE |

**Elevation trigger**: `elapsed > patience_budget OR result_count > threshold OR memory > budget`
**Patience budget**: Derived from FieldModulation.speed_bias

## 3. The Wiring Gaps

### Gap 1: Contract Not Consumed

The `lance-graph-contract` crate defines canonical types, but **nobody depends on it**:

| Consumer | Current State | Needed |
|----------|--------------|--------|
| lance-graph-planner | Own ThinkingStyle (12 variants) | Import contract's (36 variants) |
| n8n-rs | Own ThinkingMode + CompiledStyle | Import contract's ThinkingStyle + JitCompiler |
| crewai-rust | Own 36 styles | Import contract's ThinkingStyle |
| ladybug-rs | No thinking types | Import contract's PlannerContract |

### Gap 2: Planner Not Connected to Core

lance-graph-planner is **self-contained** — it doesn't call lance-graph core's parser or DataFusion. The strategies are all feature-detection-based (regex on query text), not actual parsing.

### Gap 3: JIT Pipeline Not Wired

The full pipeline: `FieldModulation → ScanParams → JitTemplate → Cranelift → KernelHandle` exists across three repos but has never been executed end-to-end:
- n8n-rs has `CompiledStyle` that calls jitson
- ndarray has `JitEngine` with Cranelift
- contract has `JitCompiler` trait
- **Nobody calls the contract trait with a real implementation**

### Gap 4: Elevation Not Connected to Execution

ElevationLevel exists in the planner but:
- No actual timing measurement during execution
- No feedback loop to trigger elevation
- The `should_elevate()` function exists but is never called from a physical operator

## 4. Integration Paths

### Path A: Contract-First (Bottom-Up)
1. All consumers add `lance-graph-contract` as a dependency
2. Replace duplicated types with contract types
3. Implement contract traits in lance-graph-planner
4. Wire implementations to consumers

### Path B: Planner-First (Top-Down)
1. Wire lance-graph-planner to lance-graph core (actual parser, not regex)
2. Wire planner strategies to real physical operators
3. Wire elevation to execution with timing feedback
4. Then connect consumers via contract traits

### Path C: JIT-First (Cross-Cutting)
1. Wire ndarray JitEngine to contract JitCompiler trait
2. Wire n8n-rs CompiledStyleRegistry to contract StyleRegistry trait
3. Test end-to-end: YAML card → compiled kernel → scan execution
4. Then wire planner and consumers
