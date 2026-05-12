# AutocompleteCache Implementation Plan

## Architecture

```text
                    ┌─────────────────────────────────────┐
                    │  AutocompleteCache                   │
                    │                                     │
                    │  ┌──────────┐  ┌──────────────────┐ │
                    │  │ KV Bundle │  │ Candidate Pool   │ │
                    │  │ (10KD)   │  │ (ranked Base17)  │ │
                    │  └────┬─────┘  └───────┬──────────┘ │
                    │       │                │            │
                    │  ┌────┴────────────────┴──────────┐ │
                    │  │     3 Simultaneous Models       │ │
                    │  │  self_model   (what I plan)     │ │
                    │  │  user_model   (what they expect)│ │
                    │  │  impact_model (what happens)    │ │
                    │  └────┬────────────────┬──────────┘ │
                    │       │                │            │
                    │  ┌────┴────────────────┴──────────┐ │
                    │  │  64 p64 Lanes (parallel eval)   │ │
                    │  │  Lane 0-15:  INNER dialogue     │ │
                    │  │  Lane 16-31: OUTER dialogue     │ │
                    │  │  Lane 32-47: IMPACT prediction  │ │
                    │  │  Lane 48-63: CACHE management   │ │
                    │  └────┬────────────────┬──────────┘ │
                    │       │                │            │
                    │  ┌────┴────┐    ┌──────┴─────────┐ │
                    │  │ NARS    │    │ Composition     │ │
                    │  │ Revision│    │ Phase Tracker   │ │
                    │  └─────────┘    └────────────────┘ │
                    └─────────────────────────────────────┘
```

## Contract Mapping (lance-graph-contract types)

```text
Contract Type                    AutocompleteCache Usage
─────────────                    ─────────────────────────
ThinkingStyle (36 variants)      Lane configuration: which styles fire
FieldModulation (7D)             Euler-gamma tension per lane
NarsTruth { f, c }              Cache entry confidence
InferenceType (7 variants)       Which NARS rule per candidate
CausalEdge64 (u64)              Packed candidate with SPO+truth+pearl
DkPosition (4 levels)            MUL: do I know enough to cache?
TrustTexture (4 levels)          MUL: should user trust this cache entry?
FlowState (4 states)             MUL: Flow→cache, Block→regenerate
GateDecision (Flow/Hold/Block)   Cache eviction trigger
PlasticityState (3-bit)          Which model planes are still learning
CausalMask (3-bit Pearl)         Level 1/2/3 causal depth per entry
```

## Integration Mapping (existing modules → cache components)

```text
Module                               Cache Component
──────                               ───────────────
p64::Palette3D                       64-lane parallel evaluator
p64::HeelPlanes                      HEEL routing (8 expert planes)
p64::predicate::*                    8 predicate layers (CAUSES..BECOMES)
bgz-tensor::HhtlCache               RouteAction lookup per archetype pair
bgz-tensor::hhtl_cache::RouteAction  Skip/Attend/Compose/Escalate
causal-edge::CausalEdge64            Packed cache entry format
causal-edge::edge::forward()         Impact prediction (compose palettes)
causal-edge::edge::learn()           NARS revision after user response
planner::thinking::style             ThinkingStyle → FieldModulation
planner::thinking::sigma_chain       Ω→Δ→Φ→Θ→Λ orchestration
planner::mul::*                      DK/Trust/Flow assessment
planner::nars::truth                 TruthValue algebra
planner::nars::inference             Deduction/Abduction/Induction
planner::strategy::chat_bundle       ChatBundle (existing Strategy #17)
ndarray::hpc::styles::*              34 cognitive primitives
ndarray::hpc::nars                   NarsTruth, revision, contradiction
ndarray::hpc::bgz17_bridge::Base17   Fingerprint type for cache entries
ndarray::hpc::causal_diff            Weight-diff derived quality scores
```

## 6 Agent Scopes

### Agent 1: KV Bundle Store
**Crate**: `lance-graph-planner/src/cache/kv_bundle.rs`
**Scope**: VSA superposition KV-cache. Fixed-size bundles for K and V.
Bundle/unbundle operations. Recency decay weighting.

**Types**:
```rust
pub struct KvBundle {
    k_bundle: [i16; 10000],  // superposed keys (fixed size)
    v_bundle: [i16; 10000],  // superposed values (fixed size)
    entry_count: u32,
    confidence: f32,         // NarsTruth.confidence
}

impl KvBundle {
    fn bundle(&mut self, key: &Base17, value: &Base17, weight: f32);
    fn unbundle(&mut self, key: &Base17) -> Base17;  // XOR out
    fn query(&self, query: &Base17) -> (Base17, f32); // nearest + score
    fn surprise(&self, actual: &Base17) -> f32;       // Friston free energy
}
```

**Paper sources**: C2C (fusion), Habr (holographic), Tensor Networks (inverse weight),
DapQ (position > semantics), KVTC (compression), CacheSlide (RPDC reuse)

**Contract deps**: None (pure data structure, no planner types)
**Integration**: Used by Agent 2 (TripleModel) as internal storage

---

### Agent 2: Triple Model (self/user/impact)
**Crate**: `lance-graph-planner/src/cache/triple_model.rs`
**Scope**: Three simultaneous VSA models tracking conversation state.
Each model has its own KvBundle + PlasticityState.

**Types**:
```rust
pub struct TripleModel {
    self_model: ModelState,    // what I plan to say
    user_model: ModelState,    // what they expect
    impact_model: ModelState,  // predicted effect of my output
}

pub struct ModelState {
    bundle: KvBundle,
    plasticity: PlasticityState,  // which planes are still learning
    confidence: NarsTruth,
    dk_position: DkPosition,     // MUL: how well do I know this model?
}

impl TripleModel {
    fn on_self_output(&mut self, output: &CausalEdge64);
    fn on_user_input(&mut self, input: &CausalEdge64);
    fn predict_impact(&self, candidate: &CausalEdge64) -> CausalEdge64;
    fn surprise(&self) -> f32;  // Friston: impact prediction vs actual
    fn topic_shift(&self) -> f32;  // Hamming(self, user) divergence
}
```

**Paper sources**: EMPA (3D vector P_t=C·eC+A·eA+P·eP),
LFRU (leader/follower causal prediction), Illusion (causal vs semantic),
PMC Attention Heads (KR/ICI/LR/EP stages)

**Contract deps**: PlasticityState, DkPosition, NarsTruth, CausalEdge64
**Integration**: Used by Agent 4 (LaneEvaluator) as state source

---

### Agent 3: Candidate Pool
**Crate**: `lance-graph-planner/src/cache/candidate_pool.rs`
**Scope**: Ranked set of autocomplete candidates. Each candidate is a
CausalEdge64 with NARS truth, ThinkingStyle provenance, and composition phase.

**Types**:
```rust
pub struct CandidatePool {
    candidates: Vec<RankedCandidate>,
    already_said: KvBundle,        // what has been output (grows)
    composition_phase: Phase,      // Exposition/Contrapunkt/Bridge/Pointe/Coda
}

pub struct RankedCandidate {
    edge: CausalEdge64,
    style: ThinkingStyle,          // which style produced this
    rank: f32,                     // quality score
    source: CandidateSource,       // which lane generated it
}

pub enum CandidateSource {
    InnerDialogue(u8),   // lane 0-15
    OuterDialogue(u8),   // lane 16-31
    ImpactPrediction(u8), // lane 32-47
    CacheManagement(u8),  // lane 48-63
}

pub enum Phase {
    Exposition,     // theme introduction (cache full, much to say)
    Durchfuehrung,  // theme development (cache depleting)
    Contrapunkt,    // counter-thesis (CONTRADICTS fires)
    Bridge,         // convergence (models align)
    Pointe,         // resolution (surprise → 0)
    Coda,           // conclusion (cache empty)
}

impl CandidatePool {
    fn add(&mut self, candidate: RankedCandidate);
    fn best(&self) -> Option<&RankedCandidate>;
    fn emit(&mut self) -> Option<CausalEdge64>;  // unbundle from cache, add to already_said
    fn update_phase(&mut self, surprise: f32, alignment: f32);
    fn is_done(&self) -> bool;  // Coda: nothing left to say
}
```

**Paper sources**: InstCache (NLL pre-population), Krites (grey zone + promotion),
ThinkPatterns (5 styles as candidate generators), Thinkless (when to think vs not),
Semantic (dual-threshold return/guide/generate)

**Contract deps**: ThinkingStyle, CausalEdge64
**Integration**: Fed by Agent 4 (LaneEvaluator), consumed by Strategy #17

---

### Agent 4: Lane Evaluator (64 parallel p64 lanes)
**Crate**: `lance-graph-planner/src/cache/lane_eval.rs`
**Scope**: 64 parallel evaluation lanes using p64 Palette64.
Each lane runs a ThinkingStyle at a specific Euler-gamma tension level.
Produces candidates for the CandidatePool.

**Types**:
```rust
pub struct LaneEvaluator {
    palette: Palette64,            // 64×64 binary attention matrix
    lane_configs: [LaneConfig; 64],
}

pub struct LaneConfig {
    style: ThinkingStyle,
    tension: f32,                  // Euler-gamma noise floor multiplier
    source_model: ModelSelector,   // which of triple model feeds this lane
    predicate: u8,                 // p64 predicate layer (CAUSES..BECOMES)
}

pub enum ModelSelector { SelfModel, UserModel, ImpactModel }

impl LaneEvaluator {
    fn fire_all(&self, triple: &TripleModel) -> Vec<RankedCandidate>;
    fn fire_lane(&self, lane: u8, state: &ModelState) -> Option<RankedCandidate>;
    fn configure_from_mul(&mut self, mul: &MulAssessment);
}
```

**Lane allocation**:
```text
Lane 0-7:   self_model × [Analytical, Creative, Focused, Integrative,
                           Divergent, Deliberate, Exploratory, Metacognitive]
Lane 8-15:  self_model × [CAUSES, ENABLES, SUPPORTS, CONTRADICTS,
                           REFINES, ABSTRACTS, GROUNDS, BECOMES]
Lane 16-23: user_model × 8 ThinkingStyles
Lane 24-31: user_model × 8 predicate layers
Lane 32-39: impact_model × 8 ThinkingStyles
Lane 40-47: impact_model × 8 predicate layers
Lane 48-55: socratic INNER (self questions self per style)
Lane 56-63: socratic OUTER (self questions user per style)
```

**Paper sources**: ThinkingIntervention (first-person token injection),
NARS same/opposite (relational frames), EMPA (directional alignment),
PMC Attention Heads (4-stage cognitive mapping)

**Contract deps**: ThinkingStyle, FieldModulation, Palette64, MulAssessment
**Integration**: Reads TripleModel, writes to CandidatePool

---

### Agent 5: NARS Revision Engine
**Crate**: `lance-graph-planner/src/cache/nars_engine.rs`
**Scope**: Closed-loop NARS feedback. After each emit, revises all truth
values based on user response. Handles contradiction detection, skepticism
scheduling, and plasticity transitions.

**Types**:
```rust
pub struct NarsEngine {
    skepticism: SkepticismSchedule,
    meta: MetaCognition,
    history: Vec<(CausalEdge64, NarsTruth)>,  // past emissions with revised truth
}

impl NarsEngine {
    fn on_emit(&mut self, emitted: &CausalEdge64, triple: &TripleModel);
    fn on_user_response(&mut self, response: &CausalEdge64, triple: &mut TripleModel);
    fn revise_candidate(&self, candidate: &mut RankedCandidate);
    fn detect_contradictions(&self, pool: &CandidatePool) -> Vec<Contradiction>;
    fn should_stop(&self) -> bool;  // all planes frozen + surprise at minimum
    fn current_inference_type(&self) -> InferenceType;
    fn mutual_entailment(&self, a: &CausalEdge64, b: &CausalEdge64) -> Option<CausalEdge64>;
    fn combinatorial_entailment(&self, a: &CausalEdge64, b: &CausalEdge64, c: &CausalEdge64) -> Option<CausalEdge64>;
}
```

**Paper sources**: NARS same/opposite (mutual + combinatorial entailment),
Illusion of Causality (causal vs semantic scaffolding),
EMPA (4 laws: stagnation, entropy, bottleneck, defensive),
Thinkless (DeGRPO: when to think)

**Contract deps**: NarsTruth, InferenceType, CausalEdge64, PlasticityState
**Integration**: Called after each emit/response cycle by Strategy #17

---

### Agent 6: Strategy #17 Integration (update existing)
**Crate**: `lance-graph-planner/src/strategy/chat_bundle.rs` (UPDATE)
**Scope**: Wire all 5 components together. Replace the simple ChatBundle
with the full AutocompleteCache. Implement the PlanStrategy trait using
the cache as the hot path.

**Update existing ChatBundleStrategy to**:
```rust
pub struct AutocompleteCacheStrategy {
    cache: AutocompleteCache,
}

pub struct AutocompleteCache {
    triple: TripleModel,
    pool: CandidatePool,
    lanes: LaneEvaluator,
    nars: NarsEngine,
}

impl PlanStrategy for AutocompleteCacheStrategy {
    fn name(&self) -> &str { "AutocompleteCache" }
    fn capability(&self) -> PlanCapability { PlanCapability::Extension }
    fn affinity(&self, context: &PlanContext) -> f32 { /* chat detection */ }
    fn plan(&self, input: PlanInput, arena: &mut Arena<LogicalOp>) -> Result<PlanInput, PlanError> {
        // 1. Parse input as CausalEdge64
        // 2. triple.on_user_input(input)
        // 3. lanes.fire_all(triple) → candidates
        // 4. pool.add(candidates)
        // 5. nars.revise_candidate(pool)
        // 6. nars.detect_contradictions(pool)
        // 7. pool.update_phase(triple.surprise(), alignment)
        // 8. best = pool.best()
        // 9. If best.rank > threshold: return cached (no LLM call)
        //    If best.rank in grey zone: async verify (Krites)
        //    If best.rank < threshold: pass through to LLM
        // 10. After LLM response: nars.on_emit + triple.on_self_output
    }
}
```

**Paper sources**: Krites (grey zone async verification),
ContextCache (multi-turn context awareness),
CacheSlide (RPDC for agent prompts),
Semantic (dual-threshold return/guide/generate)

**Contract deps**: PlanStrategy, PlanContext, PlanInput, PlanCapability
**Integration**: Replaces existing ChatBundleStrategy in default_strategies()

## File Layout

```text
lance-graph-planner/src/
  cache/
    mod.rs              ← pub mod + AutocompleteCache re-export
    kv_bundle.rs        ← Agent 1: VSA superposition store
    triple_model.rs     ← Agent 2: self/user/impact models
    candidate_pool.rs   ← Agent 3: ranked candidates + composition phase
    lane_eval.rs        ← Agent 4: 64 parallel p64 lanes
    nars_engine.rs      ← Agent 5: NARS revision + entailment
  strategy/
    chat_bundle.rs      ← Agent 6: updated to AutocompleteCacheStrategy
```

## Dependency Graph

```text
Agent 1 (KvBundle)      ← no deps, pure data structure
Agent 2 (TripleModel)   ← depends on Agent 1
Agent 3 (CandidatePool) ← depends on Agent 1 (already_said bundle)
Agent 4 (LaneEvaluator) ← depends on Agent 2 + writes to Agent 3
Agent 5 (NarsEngine)    ← depends on Agent 2 + Agent 3
Agent 6 (Strategy)      ← depends on all (orchestrator)
```

## Agent Spawn Order

**Parallel batch 1**: Agent 1 + Agent 3 (no cross-deps)
**Parallel batch 2**: Agent 2 + Agent 4 (after Agent 1 exists)
**Sequential**:       Agent 5 (after Agent 2 + 3)
**Sequential**:       Agent 6 (after all)
