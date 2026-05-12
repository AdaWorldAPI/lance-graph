# P18 — Internal LLM Language Surface

## Status: READY (all dependencies complete)

**Depends on**: P1 (ndarray), P13 (burn/SIMD), P17 (Jina pipeline)
**Does NOT block or conflict with**: P4-P16 (data flow paths)
**Enables**: P8 (entity resolution), P9 (causal reasoning), P11 (scale cartography)

## Architecture

```
                    ┌────────────────────────────┐
                    │   User / External Input     │
                    └─────────────┬──────────────┘
                                  ↓
                    ┌────────────────────────────┐
                    │   MetaOrchestrator.step()   │
                    │   MUL → DK, Trust, Flow     │
                    │   Temperature, Style select  │
                    └─────────────┬──────────────┘
                                  ↓
              ┌───────────────────────────────────────┐
              │         trait LanguageBackend          │
              │  fn extract_triplets() → Vec<Triplet>  │
              │  fn classify_entities()               │
              │  fn plan() → PlanResult               │
              │  fn refine() → Vec<OutdatedTriplet>   │
              │  fn chat() → String                   │
              ├───────────────────────────────────────┤
              │                                       │
    ┌─────────┴─────────┐            ┌───────────────┴──────────┐
    │  InternalBackend   │            │   ExternalBackend        │
    │                    │            │                          │
    │  OpenChat 3.5      │            │  xAI/Grok API            │
    │  (7B, GGUF, local) │            │  (HTTP, rate-limited)    │
    │  + GPT-2 (124M,    │            │  + OpenAI API fallback   │
    │    fast tokenizer)  │            │  + Any A2A endpoint      │
    │                    │            │                          │
    │  Context:          │            │  Context:                │
    │  - KG blackboard   │            │  - Same KG blackboard    │
    │  - Episodic memory │            │  - OSINT web search      │
    │  - HHTL palette    │            │  - External knowledge    │
    │  - CausalEdge64    │            │  - Fresh training data   │
    └────────┬───────────┘            └────────────┬─────────────┘
             │                                     │
             └──────────────┬──────────────────────┘
                            ↓
              ┌────────────────────────────┐
              │  process_triplets()        │  ← UNCHANGED
              │  TripletGraph.add()        │  ← UNCHANGED
              │  CausalEdge64.pack()       │  ← UNCHANGED
              │  NARS revision             │  ← UNCHANGED
              └────────────────────────────┘
```

## Why Internal + External (not either/or)

| Scenario | Internal (OpenChat) | External (xAI/Grok) |
|----------|--------------------|--------------------|
| **Latency** | <100ms (in-process) | 200-2000ms (HTTP) |
| **Context** | KG blackboard (unlimited via graph) | Fixed context window |
| **OSINT** | ✗ No web access | ✓ Web search, live data |
| **Cost** | Free (CPU cycles) | $/token |
| **Offline** | ✓ Works without internet | ✗ Needs network |
| **Quality** | Good for structured extraction | Better for open-ended reasoning |
| **Safety** | DK-gated (MountStupid → read-only) | More conservative by default |

**Routing decision** lives in `MetaOrchestrator`:
- `DK == MountStupid` → external only (don't trust internal parser yet)
- `DK == SlopeOfEnlightenment` → internal first, external for OSINT
- `DK == PlateauOfMastery` → internal only (fast, confident)
- `temperature > 0.7` (stagnation) → swap backend (break loops)
- `contradiction_rate > 0.3` → external verify (don't trust internal)

## Unlimited Context Window via Knowledge Graph

The key insight: **the KG IS the context window**.

Traditional LLM: prompt = [system + history + user] → fixed 8K/32K tokens
Our approach:

```
1. User input → OpenChat tokenizes (32K vocab)
2. Entity extraction → BFS into TripletGraph (get_associated, depth=2)
3. Retrieved triplets → format as context (~20 most relevant)
4. OpenChat generates with KG-grounded context
5. New triplets → graph update → NARS revision
6. Episodic memory → fingerprint → retrieval for next turn
```

**Effective context = entire graph** (millions of triplets), accessed via BFS.
Each turn only loads the *relevant* 20-50 triplets into the LLM's actual window.

## Context Blackboard Architecture

```rust
/// Shared blackboard between internal LLM, external API, and orchestrator.
/// Lives in lance-graph (not q2 — q2 is display only).
pub struct ContextBlackboard {
    /// Current conversation turn.
    pub turn: u64,
    /// Active entity set (BFS frontier from last query).
    pub active_entities: HashSet<String>,
    /// Retrieved graph context (formatted triplets).
    pub graph_context: Vec<String>,
    /// Episodic memory hits (past observations).
    pub episodic_context: Vec<String>,
    /// Pending triplets to add (from internal LLM).
    pub pending_triplets: Vec<Triplet>,
    /// Pending refinements (outdated triplets to replace).
    pub pending_refinements: Vec<(String, String, String)>,
    /// CausalEdge64 attention log (from last inference pass).
    pub attention_edges: Vec<u64>,
    /// HHTL cascade stats (how deep did we go for each lookup).
    pub cascade_stats: CascadeStats,
    /// Current thinking style (from orchestrator).
    pub style: AgentStyle,
    /// Current DK position (from MUL).
    pub dk_position: DkPosition,
}
```

## A2A + External Routing

When internal LLM encounters a query it can't answer from the KG:

```
1. Internal LLM detects low-confidence answer (logprob < threshold)
2. Orchestrator routes to ExternalBackend:
   a. xAI/Grok for OSINT search
   b. Any A2A endpoint (other agents)
   c. Web search API (Tavily, SerpAPI)
3. External result → triplet extraction → KG update
4. Internal LLM re-answers with updated KG context
5. NARS records: "external lookup improved answer" → topology learns
```

## Chat Surface

The chat endpoint (`/v1/chat/completions`) becomes the **mixed-mode** surface:

```
User: "What connections does CRISPR have to agricultural policy?"

1. OpenChat tokenizes → entity extraction: ["CRISPR", "agricultural policy"]
2. BFS: TripletGraph.get_associated(["CRISPR", "agricultural policy"], depth=3)
3. Found: 47 triplets connecting CRISPR → gene editing → crop resistance → EU regulation → ...
4. Context blackboard: top-20 triplets + episodic memory
5. OpenChat generates answer grounded in KG facts
6. Low confidence on "EU regulation timeline" → route to xAI for OSINT
7. xAI returns fresh data → new triplets → KG update
8. Final answer: grounded in KG + verified by external source
9. CausalEdge64: pack attention patterns → causal reasoning layer
10. NARS revision: update confidence on all touched triplets
```

## Thinking Styles × LLM Backend

| Style | Internal LLM role | External LLM role |
|-------|------------------|--------------------|
| **Plan** | Generate plan from KG context | Verify plan against web knowledge |
| **Act** | Execute triplet extraction (fast) | OSINT data acquisition |
| **Explore** | Generate alternative entity connections | Discover new entities via search |
| **Reflex** | Self-critique last answer | External validation of reasoning |

## Implementation Order (non-blocking)

### Phase 1: LanguageBackend trait (lance-graph)
```rust
// arigraph/language.rs — NEW
pub trait LanguageBackend: Send + Sync {
    fn extract_triplets(&self, observation: &str, context: &[String]) -> Vec<Triplet>;
    fn classify_entities(&self, entities: &[String]) -> Vec<(String, EntityType)>;
    fn plan(&self, context: &ContextBlackboard) -> PlanResult;
    fn refine(&self, existing: &[String], observation: &str) -> Vec<(String,String,String)>;
    fn chat(&self, messages: &[ChatMessage], context: &ContextBlackboard) -> String;
    fn name(&self) -> &str;
    fn is_local(&self) -> bool;
}
```

### Phase 2: InternalBackend (lance-graph, depends on ndarray)
```rust
// arigraph/internal_llm.rs — NEW
pub struct InternalBackend {
    engine: ndarray::hpc::openchat::inference::OpenChatEngine,
    prompt_templates: CompactPrompts,
}

impl LanguageBackend for InternalBackend { ... }
```

### Phase 3: Refactor XaiClient to implement LanguageBackend
```rust
// arigraph/xai_client.rs — MODIFY
impl LanguageBackend for XaiClient { ... }
```

### Phase 4: Wire into MetaOrchestrator
```rust
// arigraph/orchestrator.rs — MODIFY
pub struct MetaOrchestrator {
    // ... existing fields ...
    internal: Option<Box<dyn LanguageBackend>>,   // OpenChat/GPT-2
    external: Option<Box<dyn LanguageBackend>>,   // xAI/Grok
    blackboard: ContextBlackboard,                // NEW
}
```

### Phase 5: Chat surface (q2 cockpit-server)
- `/v1/chat/completions` routes through `MetaOrchestrator`
- Internal LLM handles fast parsing
- External API handles OSINT
- KG provides unlimited context via BFS

## Non-contradiction with P4-P16

| Path | Relationship to P18 |
|------|-------------------|
| P4 (planner→DataFusion) | P18 provides language input TO the planner |
| P5 (bgz17 workspace) | Independent — workspace organization |
| P6 (n8n orchestration) | P18's LanguageBackend is an n8n node |
| P7 (adjacency unification) | Independent — data structure |
| P8 (DeepNSM↔AriGraph entity resolution) | **P18 ENABLES P8** — internal LLM does entity matching |
| P9 (CausalEdge64↔AriGraph) | **P18 FEEDS P9** — attention edges from inference |
| P10 (psychometric validation) | Independent — measurement, not parsing |
| P11 (2B-scale cartography) | **P18 ENABLES P11** — internal LLM parses papers at scale |
| P12 (DeepNSM×CausalEdge64 bridge) | P18 provides the language surface for the bridge |
| P14-P15 (image codec, photography) | Independent — visual pipeline |
| P16 (COCA vocabulary) | **P18 USES P16** — expanded vocabulary improves parsing |

## Metrics

- Internal LLM triplet extraction: target <200ms per observation
- External fallback: <2000ms per observation
- KG BFS context retrieval: <5ms for depth=2, top_k=20
- Context blackboard update: <1ms
- NARS topology learning: tracks internal vs external quality per-style
- Temperature-based switching: measured by contradiction_rate delta
