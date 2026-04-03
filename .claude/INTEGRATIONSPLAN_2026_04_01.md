# Integrationsplan: Quick Wins to AGI Stack

> **Date**: 2026-04-01
> **Scope**: lance-graph, ndarray, q2, AriGraph
> **Prerequisite reading**: `CROSS_REPO_AUDIT_2026_04_01.md` (same directory)
> **Branch**: `claude/review-neuronprint-handover-KEFcb`

---

## OVERVIEW

This plan is ordered by **impact x effort** — quick wins first, then high-impact
blockers, then the full stack wiring. Each phase unlocks the next.

```
Phase 0: Housekeeping (1h)          ← update stale docs, unblock Session C
Phase 1: Critical Bridges (4h)      ← bgz7→f32, LanguageBackend, tokenizers
Phase 2: OSINT Pipeline (4h)        ← reader-lm + bge-m3 → live SPO hydration
Phase 3: AriGraph Thinking (4h)     ← wire 6D SPO + NeuronPrint into episodic memory
Phase 4: Chat Loop (2h)             ← HTTP in → SPO → NARS → NL out
Phase 5: Testing (2h)               ← 20-step neural-debug validation
Phase 6: OpenClaw + AGI (4h)        ← persona layer, meta-reasoning, self-improvement
```

Total: ~21 hours of focused implementation to a testable AGI prototype.

---

## PHASE 0: HOUSEKEEPING (quick wins, 1 hour)

### 0.1 Update integration-lead agent (10 min)

**File**: `.claude/agents/integration-lead.md`

```
Session B: PARTIAL → DONE
  Evidence: palette_semiring, palette_matrix, palette_csr, simd,
  typed_palette_graph all on main (121 tests). Phase 2 checklist all [x].
```

### 0.2 Update session_B prompt (5 min)

**File**: `.claude/prompts/session_B_v3_bgz17_container_semiring.md`
Add "COMPLETED" header matching Session A's format.

### 0.3 Update blackboard.md (10 min)

**File**: `.claude/blackboard.md`
Current state is from Mar 23. Update with:
- NeuronPrint shipped (neuron.rs, 9 tests)
- reader-lm + bge-m3 crates created (forward passes, blocked on weight bridge)
- p64-bridge + Blumenstrauss shipped
- OSINT pipeline skeleton working
- Session B fully done, C ready to start

### 0.4 Remove dead field (5 min)

**File**: `crates/lance-graph-planner/src/cache/convergence.rs`
Remove `AutocompleteCache.palette_indices` (unused after SPO refactor).

### 0.5 Populate empty placeholder (5 min)

**File**: `.claude/prompts/SESSION_BGZ_TENSOR_HYDRATE.md` (0 bytes)
Either populate with bgz-tensor hydration workflow or delete the empty file.

### 0.6 Add bgz17 as optional dependency of lance-graph (15 min)

**File**: `crates/lance-graph/Cargo.toml`

```toml
[dependencies]
bgz17 = { path = "../bgz17", optional = true }

[features]
bgz17-codec = ["bgz17"]
```

This is the **single most important line change** in the entire plan.
It unblocks Session C, which unblocks Session D, which unblocks FalkorDB.

---

## PHASE 1: CRITICAL BRIDGES (high impact, 4 hours)

### 1.1 bgz7-to-f32 weight decompression bridge

**Why**: reader-lm and bge-m3 have complete forward passes but cannot load
weights because bgz7 files contain palette-indexed Base17 rows and the models
need f32 tensors.

**Where**: `crates/bgz-tensor/src/` (existing crate)

**What to build**:
```rust
/// Decompress a Base17 row back to approximate f32 values.
/// Inverse of golden-step octave averaging.
pub fn base17_to_f32(row: &[i16; 17], output_dim: usize) -> Vec<f32> {
    // Each i16 value represents mean * 256 for that golden-step position
    // Broadcast each of 17 dims across output_dim/17 octaves
    // This is lossy (964:1 expansion) but preserves the signal direction
}

/// Load a bgz7 shard and decompress all rows to f32 weight matrix.
pub fn load_weight_matrix(path: &Path, rows: usize, cols: usize) -> Vec<Vec<f32>>
```

**Tests**: Round-trip test — encode f32→Base17→f32, verify cosine similarity > 0.95.

### 1.2 Real tokenizers for reader-lm and bge-m3

**Why**: Both crates have byte-level stub tokenizers. Real inference needs real vocab.

**Options** (choose one):
1. **Wrap `tokenizers` crate** (HuggingFace Rust, production-grade, ~50KB binary)
   — Load `tokenizer.json` from HF hub, `encode()` / `decode()`. 10 lines of glue.
2. **Sentencepiece via `sentencepiece-rs`** — More control, larger dependency.
3. **Minimal BPE from vocab file** — Zero deps, ~200 lines, good enough for reader-lm.

**Recommendation**: Option 1. The `tokenizers` crate is battle-tested and handles
Qwen2's tiktoken-style vocab and XLM-RoBERTa's sentencepiece vocab.

### 1.3 LanguageBackend: InternalBackend implementation

**Why**: The trait exists with 0 implementations. This is the NL generation gap.

**File**: `crates/lance-graph/src/graph/arigraph/language.rs`

**Implementation**:
```rust
pub struct InternalBackend {
    router: ndarray::hpc::models::ModelRouter,
}

impl LanguageBackend for InternalBackend {
    fn extract_triplets(&self, observation: &str, context: &[String], timestamp: u64)
        -> Result<Vec<Triplet>, String>
    {
        // 1. Build prompt: "Extract (subject, predicate, object) from: {observation}"
        // 2. router.complete(prompt) → raw text
        // 3. Parse triplets from structured output
        // 4. Attach NARS truth (freq=1.0, conf=0.5 for new observations)
    }

    fn chat(&self, user_message: &str, blackboard: &ContextBlackboard)
        -> Result<String, String>
    {
        // 1. blackboard.load_from_graph() → top-K relevant triplets
        // 2. Build prompt with triplet context + user message
        // 3. router.chat_complete(messages) → response text
    }
}
```

**Depends on**: ndarray `ModelRouter` (already built, GPT-2 + OpenChat engines).
Requires ndarray as a dependency of lance-graph (add to Cargo.toml).

### 1.4 LanguageBackend: ExternalBackend implementation

**File**: Same as above.

```rust
pub struct ExternalBackend {
    client: xai_client::XaiClient,  // already exists in arigraph/xai_client.rs
}

impl LanguageBackend for ExternalBackend {
    fn extract_triplets(...) { /* POST to xAI/Grok API with structured prompt */ }
    fn chat(...) { /* POST to xAI/Grok with blackboard context */ }
}
```

**Already half-built**: `xai_client.rs` has HTTP POST to xAI. Just needs to implement
the trait methods using the existing client.

---

## PHASE 2: OSINT PIPELINE — LIVE 6D SPO HYDRATION (4 hours)

### 2.1 Wire reader-lm into OSINT pipeline

**Current**: `lance-graph-osint/reader.rs` strips HTML with regex.
**Target**: Use Reader LM (Qwen2-1.5B) to convert HTML → clean markdown.

```rust
// In reader.rs
fn fetch_and_clean(&self, url: &str) -> Result<String, Error> {
    let html = ureq::get(url).call()?.into_string()?;
    // OLD: hand_strip_html(&html)
    // NEW:
    let weights = ReaderLmWeights::load("reader_lm_1_5b.bgz7")?;
    let model = Qwen2Model::new(&weights)?;  // needs Phase 1.1 bridge
    model.html_to_markdown(&html)
}
```

**Fallback**: Keep rule-based HTML stripping as `feature = "lite"` for environments
without model weights.

### 2.2 Wire bge-m3 into OSINT pipeline

**Current**: `reader.rs` uses golden-step hash for embeddings.
**Target**: Use BGE-M3 (XLM-RoBERTa) for real multilingual semantic embeddings.

```rust
fn embed_paragraph(&self, text: &str) -> Base17 {
    // OLD: hash_to_base17(text)
    // NEW:
    let engine = BgeM3Engine::with_weights("bge_m3_f16.bgz7")?;
    engine.embed_to_base17(text)  // 1024-dim → Base17 via golden-step
}
```

### 2.3 Google/reader-lm for online 6D SPO hydration

**The pipeline**:
```
URL → reader-lm (HTML→markdown) → sentence split
  → bge-m3 (sentence→1024-dim→Base17 per sentence)
  → extract_triplets() via LanguageBackend (structured SPO)
  → triplet_to_headprint(S, P, O) → HeadPrint (17-dim per role)
  → For each triplet: construct NeuronPrint-style 6D encoding:
      Q = query_base17 (what triggers this knowledge)
      K = subject_base17 (what it matches against)
      V = object_base17 (what it retrieves)
      Gate = confidence_base17 (NARS freq → magnitude)
      Up = predicate_base17 (the relation type)
      Down = context_base17 (surrounding paragraph embedding)
  → 204-byte NeuronPrint per triplet
  → palette_compress → 6-byte palette edge
  → store in AriGraph TripletGraph + Lance dataset
```

**This is online 6D SPO hydration**: every URL ingested becomes a set of
6-dimensional cognitive edges, not flat text embeddings. The HHTL cascade
(scent → palette → Base17 → full) provides sub-microsecond retrieval.

### 2.4 OSINT harvesting loop

```rust
/// Continuous OSINT hydration
pub async fn harvest_loop(pipeline: &mut OsintPipeline, urls: &[String]) {
    for url in urls {
        // 1. Fetch + clean via reader-lm
        let markdown = pipeline.reader.fetch_and_clean(url)?;
        // 2. Extract triplets via LanguageBackend
        let triplets = pipeline.backend.extract_triplets(&markdown, &[], now())?;
        // 3. Embed each triplet component via bge-m3
        // 4. Construct 6D NeuronPrint per triplet
        // 5. NARS revision against existing knowledge
        // 6. Store in TripletGraph + Lance
        // 7. Update Blumenstrauss topology (p64 palette layers)
        pipeline.ingest_triplets(&triplets)?;
    }
}
```

---

## PHASE 3: ARIGRAPH THINKING WIRE-UP (4 hours)

### 3.1 Encoding decision: bgz17 vs 3x16kbit for AriGraph edges

**Answer: BOTH, layered.**

| Layer | Encoding | Size | When |
|-------|----------|------|------|
| **Learning** | 3x16kbit Plane | 48 KB/edge | During encounter (saturating i8 accumulation) |
| **Storage** | Base17 | 102 B/edge | After crystallization (golden-step compression) |
| **Traversal** | Palette index | 3 B/edge | Graph algebra, semiring mxm, bulk operations |
| **Inspection** | NeuronPrint 6D | 204 B/node | When asking "what does this neuron do?" |

AriGraph edges should be **palette-encoded (3 bytes)** for production graph
traversal (611M lookups/sec). The 16kbit Plane is the write-path accumulator
used during learning — it never travels over edges.

For **cognitive edges** (thinking-about-thinking), the 6D SPO encoding adds
the NeuronPrint dimension: each cognitive edge carries Q/K/V/Gate/Up/Down
roles from the 6D decomposition. This compresses to 6 palette indices (6 bytes)
via the existing palette infrastructure.

### 3.2 Wire AriGraph to new thinking

**Current**: AriGraph uses `Fingerprint` (512-bit Hamming) for similarity.
**Target**: Use NeuronPrint-informed palette edges.

**File**: `crates/lance-graph/src/graph/arigraph/episodic.rs`

```rust
// Current EpisodicMemory stores:
struct Episode {
    timestamp: u64,
    observation: String,
    triplets: Vec<(String, String, String)>,
    fingerprint: Fingerprint<8>,  // 512-bit, Hamming distance
}

// New: add NeuronPrint-based encoding
struct Episode {
    timestamp: u64,
    observation: String,
    triplets: Vec<(String, String, String)>,
    fingerprint: Fingerprint<8>,          // keep for backward compat
    neuron_prints: Vec<NeuronPrint>,      // 6D per triplet
    palette_edges: Vec<[u8; 6]>,          // 6-byte palette-compressed
}
```

Retrieval becomes:
```rust
fn retrieve(&self, query: &NeuronQuery) -> Vec<&Episode> {
    // NeuronQuery selects which roles to probe (6-bit mask)
    // e.g. NeuronQuery::attention(q) probes K store only
    // Score via palette distance table: O(1) per edge
    self.episodes.iter()
        .filter_map(|ep| {
            let score = ep.palette_edges.iter()
                .map(|e| palette.distance(query_idx, e))
                .min()?;
            (score < threshold).then_some(ep)
        })
        .collect()
}
```

### 3.3 Persona abstraction layer

**File**: New `crates/lance-graph/src/graph/arigraph/persona.rs`

```rust
pub struct Persona {
    pub name: String,
    pub gestalt: NeuronPrint,          // 204-byte behavioral fingerprint
    pub style: ThinkingStyle,          // from lance-graph-contract
    pub episodic: EpisodicMemory,      // personal episode store
    pub knowledge: TripletGraph,       // personal knowledge graph
    pub dk_position: DkPosition,       // Dunning-Kruger self-assessment
}

impl Persona {
    /// Merge another persona's knowledge via NARS revision
    pub fn absorb(&mut self, other: &Persona) { ... }

    /// Generate response using persona's style + knowledge
    pub fn respond(&self, input: &str, backend: &dyn LanguageBackend) -> String {
        let blackboard = ContextBlackboard::from_persona(self);
        backend.chat(input, &blackboard).unwrap_or_default()
    }
}
```

The persona's `gestalt` NeuronPrint is the 34-byte holographic bundle of all
their knowledge — it acts as a fast content-addressable key for "which persona
should handle this query?" via L1 distance.

### 3.4 p64 highway for episodic + persona

The p64-bridge `Blumenstrauss` already binds p64 topology with bgz17 distance.
Extend it to support persona routing:

```rust
/// Each persona gets a HeelPlane (8x u64 = 64 bytes).
/// Persona selection = attend(query, persona_heels) → best persona index.
/// Then route to that persona's EpisodicMemory + TripletGraph.
pub struct PersonaRouter {
    personas: Vec<Persona>,
    heels: Vec<HeelPlanes>,  // one per persona, derived from gestalt
}

impl PersonaRouter {
    pub fn route(&self, query: &Base17) -> &Persona {
        let heel_query = HeelPlanes::from_clam_seed(&query.to_i8());
        let best = self.heels.iter()
            .enumerate()
            .min_by_key(|(_, h)| p64::nearest_k(&heel_query, h).distance)
            .map(|(i, _)| i)
            .unwrap_or(0);
        &self.personas[best]
    }
}
```

---

## PHASE 4: CHAT LOOP — HTTP IN, SPO, NARS, NL OUT (2 hours)

### 4.1 Wire LanguageBackend into serve.rs

**Current**: `serve.rs` calls `extract_triplets()` (rule-based) and returns diagnostic strings.
**Target**: Use `LanguageBackend` for both extraction and response generation.

**File**: `crates/lance-graph-planner/src/serve.rs`

```rust
// At startup:
let internal = InternalBackend::new(ModelRouter::default());
let external = ExternalBackend::new(XaiClient::new(api_key));
let blackboard = ContextBlackboard::new();

// In /v1/chat/completions handler:
async fn chat(State(state): State<AppState>, Json(req): Json<ChatRequest>) -> Json<ChatResponse> {
    let user_msg = req.messages.last().unwrap();

    // 1. Select backend via DK-gated routing
    let backend = select_backend(&state.blackboard, &state.internal, &state.external);

    // 2. Extract triplets via LLM (not regex)
    let triplets = backend.extract_triplets(&user_msg.content, &[], now())?;

    // 3. For each triplet: headprint -> SpoHead -> NARS inference (existing pipeline)
    for triplet in &triplets {
        let hp = triplet_to_headprint(&triplet.subject, &triplet.relation, &triplet.object);
        let spo = headprint_to_spo(&hp, triplet.truth.freq, triplet.truth.conf);
        state.engine.learn(&spo);
        state.engine.infer_all(&spo);
    }

    // 4. Generate natural language response via LLM
    state.blackboard.load_from_graph(&state.graph);
    let response_text = backend.chat(&user_msg.content, &state.blackboard)?;

    // 5. Extract triplets from our own response (self-learning)
    let self_triplets = backend.extract_triplets(&response_text, &[], now())?;
    for t in &self_triplets { state.engine.learn(&triplet_to_spo(t)); }

    Json(ChatResponse { content: response_text, .. })
}
```

### 4.2 Wire q2 cockpit to lance-graph

**Current**: q2 cockpit `/v1/chat/completions` returns placeholder string.
**Target**: Proxy to lance-graph serve.rs (port 3000) or embed the planner directly.

**Option A** (quick, recommended): HTTP proxy — cockpit forwards to port 3000.
**Option B** (embedded): Add lance-graph-planner as a dep of cockpit-server.
More coupling but eliminates the HTTP hop and exposes live SPO/NARS diagnostics.

### 4.3 Chess / thinking-about-thinking application

The SPO + NARS pipeline naturally supports chess and meta-reasoning:

```
# Chess position as SPO triplets:
("e4-pawn", "controls", "d5-square")     -- spatial relation
("e4-pawn", "attacks", "d5-knight")      -- threat relation
("player", "plans", "kingside-attack")   -- strategic intention

# Meta-reasoning (thinking about thinking):
("my-analysis", "overlooked", "back-rank-mate")   -- self-correction
("pattern-X", "similar-to", "Sicilian-defense")   -- analogy
("confidence", "decreasing-for", "kingside-plan") -- NARS truth update
```

Each triplet gets 6D NeuronPrint encoding -> palette edge -> NARS inference.
The sensorium tracks temperature (stagnation), plasticity (learning rate),
and the orchestrator triggers backend swaps when the position is novel
(high contradiction rate -> switch to ExternalBackend for verification).

### 4.4 Chat input/output flow

```
User message --> LanguageBackend.extract_triplets()
                    |
                    v
              SPO triplets --> NeuronPrint 6D encoding
                    |              |
                    v              v
              NARS inference   Palette storage
                    |              |
                    v              v
              ContextBlackboard <-- BFS from TripletGraph
                    |
                    v
              LanguageBackend.chat(user_msg, blackboard)
                    |
                    v
              NL response --> extract_triplets(self-response)
                    |              |
                    v              v
              HTTP response   Self-learning (NARS revision)
```

Every conversation turn enriches the knowledge graph. The graph IS the memory.
Context window is unlimited because `blackboard.load_from_graph()` does BFS
to select only the most relevant triplets for the current query.

---

## PHASE 5: 20-STEP NEURAL-DEBUG VALIDATION (2 hours)

Uses `neural-debug` crate (lance-graph static scanner + q2 runtime instrument)
to validate the full stack end-to-end.

| Step | Test | Tool | Expected |
|------|------|------|----------|
| **1** | `cargo test -p bgz17` | CLI | 121 tests pass -- palette, semiring, SIMD |
| **2** | `cargo test -p lance-graph --lib -- graph::neuron` | CLI | 9 tests -- NeuronPrint/Query/Trace |
| **3** | `cargo test -p lance-graph --lib -- graph::hydrate` | CLI | 9 tests -- TensorRole, partition cols |
| **4** | `cargo test -p lance-graph --lib -- graph::arigraph` | CLI | TripletGraph BFS, EpisodicMemory retrieval |
| **5** | `cargo check -p lance-graph --features bgz17-codec` | CLI | Compiles -- bgz17 wired as optional dep |
| **6** | `cargo test -p lance-graph-contract` | CLI | 15 tests -- ThinkingStyle, modulation |
| **7** | `cargo test -p lance-graph-planner --lib -- cache` | CLI | 39 tests -- HeadPrint, NARS, convergence |
| **8** | `neural-scan --repo lance-graph` | neural-debug | 0 dead neurons in critical path |
| **9** | `neural-scan --repo ndarray` | neural-debug | Stubbed bridges flagged (Wire Synapses 1-3) |
| **10** | `cargo run -p lance-graph-planner --features serve` | Runtime | Server on :3000, `/health` returns "ok" |
| **11** | `curl localhost:3000/v1/models` | HTTP | 7 models listed |
| **12** | POST `/v1/chat/completions` "Alice knows Bob" | HTTP | SPO: (Alice, knows, Bob), NARS score |
| **13** | POST "Bob teaches Carol" | HTTP | NARS deduction chain: Alice->Bob->Carol |
| **14** | POST `/v1/embeddings` "test sentence" | HTTP | 17-dim HeadPrint vector |
| **15** | `cargo test -p lance-graph-osint -- --ignored` | CLI | OSINT fetch + triplet extraction |
| **16** | `curl localhost:2718/api/debug/strategies` | HTTP | 16 strategies with call counts |
| **17** | `curl localhost:2718/mri` | HTTP | Plasticity, activation, NARS chains |
| **18** | `cargo test -p lance-graph-planner --lib -- cache::convergence` | CLI | Episodes -> palette -> Blumenstrauss |
| **19** | Blumenstrauss cascade test | Unit | HEEL->HIP->TWIG->LEAF 4-stage completes |
| **20** | Full round-trip: chat -> triplets -> NeuronPrint -> palette -> store -> retrieve -> NL | Integration | End-to-end knowledge cycle |

### Consumer contract expansion via p64 highway

Expand `lance-graph-contract` for episodic + persona:

```rust
// New in lance-graph-contract/src/persona.rs:
pub trait PersonaContract: Send + Sync {
    fn gestalt(&self) -> &[i16; 17];          // 34-byte holographic identity
    fn style(&self) -> ThinkingStyle;
    fn dk_position(&self) -> DkPosition;
    fn route_query(&self, query: &[i16; 17]) -> RoutingDecision;
}

pub trait EpisodicContract: Send + Sync {
    fn store(&mut self, episode: EpisodeSummary);
    fn retrieve(&self, query: &[i16; 17], k: usize) -> Vec<EpisodeSummary>;
    fn forget(&mut self, below_confidence: f32);
}

pub struct EpisodeSummary {
    pub timestamp: u64,
    pub gestalt: [i16; 17],       // Base17 holographic summary
    pub palette_edge: [u8; 6],    // 6D palette-compressed
    pub nars_truth: (f32, f32),   // (frequency, confidence)
}
```

---

## PHASE 6: OPENCLAW + AGI WIRING (4 hours)

### 6.1 OpenClaw agent card import

```rust
// New crate: crates/openclaw-bridge/src/lib.rs
pub struct OpenClawCard {
    pub name: String,
    pub capabilities: Vec<String>,
    pub tools: Vec<ToolManifest>,
    pub memory: MemorySpec,
}

impl From<OpenClawCard> for Persona {
    fn from(card: OpenClawCard) -> Persona {
        Persona {
            name: card.name,
            gestalt: NeuronPrint::from_capabilities(&card.capabilities),
            style: ThinkingStyle::from_card(&card),
            episodic: EpisodicMemory::with_capacity(card.memory.max_episodes),
            knowledge: TripletGraph::new(),
            dk_position: DkPosition::MountStupid, // new agent starts ignorant
        }
    }
}
```

### 6.2 Testing with OpenClaw

Since OpenClaw uses OpenAI-compatible API, our serve.rs is already a valid backend:

```bash
# 1. Start lance-graph server
cargo run -p lance-graph-planner --features serve

# 2. Point OpenClaw at our server
export OPENAI_API_BASE=http://localhost:3000/v1
export OPENAI_API_KEY=dummy  # our server doesn't require auth

# 3. Run OpenClaw agent -- every message flows through SPO + NARS
openclaw run --card agent.yaml
```

**Rust transcode vs OpenAI API drop-in**: Both work simultaneously.
- **Rust transcode path**: Import OpenClaw card -> Persona struct -> in-process
  NeuronPrint routing, p64 attend, palette distance. Zero HTTP overhead.
- **OpenAI API drop-in path**: Any OpenAI-compatible client connects to port 3000.
  Gets SPO reasoning transparently. No client-side code changes needed.

### 6.3 Self-improving AGI loop

```
                    +------------------------------+
                    |     Meta-Orchestrator         |
                    |  (sensorium + DK-gating)      |
                    +------+----------+-------------+
                           |          |
                    +------v--+  +----v-------+
                    | Internal |  |  External   |
                    | Backend  |  |  Backend    |
                    |(OpenChat)|  |  (xAI/Grok) |
                    +------+--+  +----+--------+
                           |          |
                    +------v----------v-------------+
                    |      ContextBlackboard         |
                    |  (BFS into TripletGraph)       |
                    +------+----------+-------------+
                           |          |
                    +------v--+  +----v-------+
                    | Episodic |  |  Triplet   |
                    | Memory   |  |  Graph     |
                    |(NeurPrt) |  |  (NARS)    |
                    +------+--+  +----+--------+
                           |          |
                    +------v----------v-------------+
                    |   p64 Highway (Blumenstrauss)  |
                    |  attend / cascade / deduce     |
                    +------+----------+-------------+
                           |          |
                    +------v--+  +----v-------+
                    | Palette  |  |  Lance     |
                    | 3-byte   |  |  Dataset   |
                    | O(1)     |  |  (persist) |
                    +----------+  +------------+
```

**AGI properties from composition**:

1. **Unlimited memory**: TripletGraph + EpisodicMemory grow without bound.
   BFS context selection replaces fixed context windows.

2. **Self-correction**: NARS truth revision. New evidence revises old beliefs.
   Contradictions trigger ExternalBackend verification (DK-gated).

3. **Learning from every interaction**: Both user messages AND self-responses
   get triplet-extracted and stored. The graph grows with every turn.

4. **Multi-persona**: PersonaRouter selects best-matching persona per query.
   New personas spawn from OpenClaw cards or knowledge clustering.

5. **Meta-reasoning**: Sensorium tracks temperature, plasticity, contradiction rate.
   - Switch backends when stuck (temperature rising)
   - Reduce learning rate when confident (plasticity decrease)
   - Spawn new exploration when contradictions accumulate
   - Merge personas when gestalts converge

6. **Hardware-accelerated thinking**: p64 Palette64 + AVX-512 SIMD means
   persona routing, episodic retrieval, and graph traversal are sub-microsecond.
   611M SPO lookups/sec = thinking at hardware speed.

---

## PRIORITY MATRIX

| # | Item | Impact | Effort | Phase | Blocked by |
|---|------|--------|--------|-------|------------|
| 1 | Add bgz17 as lance-graph dep | **Critical** | 15 min | 0.6 | Nothing |
| 2 | Update integration-lead agent | Low | 10 min | 0.1 | Nothing |
| 3 | bgz7->f32 weight bridge | **Critical** | 2h | 1.1 | Nothing |
| 4 | Real tokenizers | **Critical** | 1h | 1.2 | Nothing |
| 5 | InternalBackend impl | **High** | 2h | 1.3 | ndarray as dep |
| 6 | ExternalBackend impl | Medium | 1h | 1.4 | xai_client (done) |
| 7 | Wire reader-lm into OSINT | **High** | 1h | 2.1 | Items 3, 4 |
| 8 | Wire bge-m3 into OSINT | **High** | 1h | 2.2 | Items 3, 4 |
| 9 | Online 6D SPO hydration | **High** | 2h | 2.3 | Items 7, 8 |
| 10 | AriGraph NeuronPrint encoding | Medium | 2h | 3.2 | Item 1 |
| 11 | Persona abstraction | Medium | 2h | 3.3 | Item 10 |
| 12 | Chat loop (serve.rs NL) | **High** | 1h | 4.1 | Item 5 |
| 13 | q2 cockpit wiring | Medium | 1h | 4.2 | Item 12 |
| 14 | 20-step validation | **High** | 2h | 5 | Items 1-12 |
| 15 | DataFusion UDFs | Medium | 2h | 6d.3 | Item 1 |
| 16 | OpenClaw bridge | Low | 2h | 6.1 | Item 11 |
| 17 | PersonaRouter | Medium | 1h | 3.4 | Item 11 |
| 18 | Contract expansion | Low | 1h | 5 | Item 11 |
| 19 | Session C (ndarray bgz17 dual-path) | Medium | 4h | -- | Item 1 |
| 20 | Session D (FalkorDB retrofit) | Low | 4h | -- | Item 19 |

**Critical path**: 1 -> 3 -> 4 -> 7 -> 9 -> 12 -> 14
**Quick wins (< 30 min each)**: Items 1, 2, 6
**Highest single-item impact**: Item 5 (InternalBackend -- enables NL generation everywhere)

---

## APPENDIX A: ENCODING DECISION MATRIX

### Should AriGraph use bgz17 or 3x16kbit bitpacked?

**Answer: Both, at different stages.**

| Layer | Encoding | Size | When |
|-------|----------|------|------|
| **Learning** | 3x16kbit Plane | 48 KB/edge | During encounter (saturating i8 accum) |
| **Storage** | Base17 | 102 B/edge | After crystallization (golden-step) |
| **Traversal** | Palette index | 3 B/edge | Graph algebra, semiring mxm, bulk |
| **Inspection** | NeuronPrint 6D | 204 B/node | "What does this neuron do?" |

For **cognitive edges** (6D SPO): 6 palette indices = 6 bytes per edge.
The palette distance table gives O(1) similarity per role independently.

The 3x16kbit Plane is **never stored as an edge**. It is the write-path
accumulator used during learning. Once crystallized to Base17, the Plane
can be GC'd. It only re-materializes for un-learn or RL reward signals.

---

## APPENDIX B: REPO FILE LOCATIONS

| Component | Path |
|-----------|------|
| NeuronPrint | `lance-graph/crates/lance-graph/src/graph/neuron.rs` |
| TripletGraph | `lance-graph/crates/lance-graph/src/graph/arigraph/triplet_graph.rs` |
| EpisodicMemory | `lance-graph/crates/lance-graph/src/graph/arigraph/episodic.rs` |
| LanguageBackend | `lance-graph/crates/lance-graph/src/graph/arigraph/language.rs` |
| Sensorium | `lance-graph/crates/lance-graph/src/graph/arigraph/sensorium.rs` |
| Orchestrator | `lance-graph/crates/lance-graph/src/graph/arigraph/orchestrator.rs` |
| serve.rs | `lance-graph/crates/lance-graph-planner/src/serve.rs` |
| Convergence | `lance-graph/crates/lance-graph-planner/src/cache/convergence.rs` |
| NARS engine | `lance-graph/crates/lance-graph-planner/src/cache/nars_engine.rs` |
| reader-lm | `lance-graph/crates/reader-lm/src/` |
| bge-m3 | `lance-graph/crates/bge-m3/src/` |
| OSINT pipeline | `lance-graph/crates/lance-graph-osint/src/` |
| p64 | `ndarray/crates/p64/src/lib.rs` |
| p64-bridge | `lance-graph/crates/p64-bridge/src/` |
| Palette | `lance-graph/crates/bgz17/src/palette.rs` |
| Base17 | `ndarray/src/hpc/bgz17_bridge.rs` |
| Plane | `ndarray/src/hpc/plane.rs` |
| ModelRouter | `ndarray/src/hpc/models/router.rs` |
| Contract | `lance-graph/crates/lance-graph-contract/src/` |
| neural-debug (scanner) | `lance-graph/crates/neural-debug/src/` |
| neural-debug (runtime) | `q2/crates/neural-debug/src/` |
| Cockpit server | `q2/crates/cockpit-server/src/` |
| Cockpit UI | `q2/cockpit/src/` |
