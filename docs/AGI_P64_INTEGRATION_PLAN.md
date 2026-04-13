# AGI::p64 — Integration Plan

> **Date**: 2026-04-03
> **Status**: Verified — all types checked against source, all tests pass
> **Branch**: `claude/qwen-claude-reverse-eng-vHuHv`

---

## Verified Type Map (source of truth)

Every mapping below has been verified against actual Rust signatures.

```
ndarray types (canonical):
  Base17            { dims: [i16; 17] }                    — bgz17_bridge.rs:33
  Base17::l1(&Base17) -> u32                               — bgz17_bridge.rs:518
  Base17::xor_bind(&Base17) -> Base17                      — bgz17_bridge.rs:552
  Base17::zero() -> Base17                                 — bgz17_bridge.rs:509
  NarsTruth         { frequency: f32, confidence: f32 }    — nars.rs:19
  NarsTruth::expectation() -> f32                          — nars.rs:114
  nars_revision(NarsTruth, NarsTruth) -> NarsTruth         — nars.rs:322 (free fn, NOT method)

p64 types:
  Palette64         { rows: [u64; 64] }                    — p64/lib.rs:39
  HeelPlanes        { planes: [u64; 8] }                   — p64/lib.rs:50
  attend(&self, query: u64, gamma: u8) -> AttentionResult  — p64/lib.rs:450
  moe_gate(&self, query: u64, threshold: u8) -> MoeGate    — p64/lib.rs:533
  hhtl_cascade_search(&Palette64, u8, &mut [f32;256])      — p64/lib.rs:1457

lance-graph types:
  NeuronPrint       { layer, feature, q, k, v, gate, up, down }  — neuron.rs:26
  NeuronQuery       { layer?, feature?, q?, k?, v?, gate?, up?, down? }  — neuron.rs:89
  NeuronTrace       { frequency, confidence, attention, coherence, expectation }  — neuron.rs:171
  TensorRole        enum { QProj=0..Other=9 }              — hydrate.rs:21
  SpoHead           { s_idx, p_idx, o_idx, freq, conf, pearl, inference, temporal }  — nars_engine.rs:28
  SpoDistances      { s_table, p_table, o_table: Vec<u16> } — nars_engine.rs:78
  StyleVector       { name: &'static str, weights: [f32; 8] }  — nars_engine.rs:188
  nars_infer(&SpoHead, &SpoHead, Inference) -> Truth       — nars_engine.rs:144
  NarsEngine::score(&self, &SpoHead, &SpoHead, &StyleVector) -> f32  — nars_engine.rs:375

bgz17 types:
  Palette           { entries: Vec<Base17> }                — palette.rs:14
  Palette::build(&[Base17], k: usize, max_iter: usize)     — palette.rs:90
  Palette::nearest(&self, &Base17) -> u8                    — palette.rs:55
  DistanceMatrix    { data: Vec<u16>, k: usize }            — distance_matrix.rs:15
  DistanceMatrix::build(&Palette) -> Self                   — distance_matrix.rs:24
  DistanceMatrix::distance(&self, u8, u8) -> u16            — distance_matrix.rs:45
  SimilarityTable   { table: [f32;256], bucket_width, max_distance }  — similarity.rs:12
  SimilarityTable::from_reservoir(&mut [u32]) -> Self       — similarity.rs:38
  SimilarityTable::similarity(&self, u32) -> f32            — similarity.rs:66

convergence bridge:
  triplet_to_headprint(&str, &str, &str) -> HeadPrint       — convergence.rs:28
  headprint_to_spo(&HeadPrint, f32, f32) -> SpoHead         — convergence.rs:49
  HeadPrint = ndarray::hpc::bgz17_bridge::Base17             — kv_bundle.rs:14
  Truth = ndarray::hpc::nars::NarsTruth                      — triple_model.rs:36
```

---

## Quick Wins → High Impact (ordered)

### TIER 1: Alive in 1 session (make it run end-to-end)

#### 1.1 OSINT Harvesting via reader-lm + DeepNSM
**What**: Fetch a URL → strip HTML (reader-lm) → extract SPO triplets (DeepNSM) → store in AriGraph.
**Wiring**:
```
lance-graph-osint/src/reader.rs::fetch_and_strip(url) -> String
lance-graph-osint/src/extractor.rs::extract_triplets(text, ts) -> Vec<Triplet>
convergence.rs::triplet_to_headprint(s, p, o) -> HeadPrint
convergence.rs::headprint_to_spo(fp, 0.9, 0.7) -> SpoHead
AriGraph::TripletGraph::add_triplets(&[Triplet])
```
**Status**: All functions exist. Just needs a CLI command or serve.rs endpoint to chain them.
**Test**: `curl http://localhost:3000/v1/ingest -d '{"url":"https://..."}'` → returns triplet count.

#### 1.2 Chat as Input/Output via serve.rs SPO pipeline
**What**: Every user message → SPO triplets → NARS score against knowledge → response.
**Status**: Already wired in serve.rs (commit 5f07f3a). Works but `message_to_base17()` is still
byte-hash for the embeddings endpoint — only the chat path uses `triplet_to_headprint()`.
**Test**: `curl /v1/chat/completions -d '{"messages":[{"role":"user","content":"Claude causes reasoning"}]}'`
→ should return `[SPO×1] (Claude —causes→ reasoning) S=42 P=17 O=88 score=0.XXX`

#### 1.3 BGE-M3 Embedding for Online 6D SPO Hydration
**What**: User text → BGE-M3 embed → Base17 projection → NeuronQuery probe against weight store.
**Wiring**:
```
bge-m3/src/embed.rs::embed_text(text) -> Vec<f32>     // 1024-dim
bge-m3/src/embed.rs::embed_to_base17(embed) -> Base17  // golden-step fold
NeuronQuery::attention(base17_query).at_layer(N)        // probe Q store
```
**Gap**: bge-m3 crate has the inference transcode but needs actual weight loading at runtime.
The bgz7 weights for bge-m3 exist (7.3 MB in GitHub Release). Need to wire `load_bgz7()` at startup.

### TIER 2: Thinking wired (make it reason)

#### 2.1 Wire AriGraph to Use NeuronPrint Episodic Memory
**What**: Replace string triplets in `EpisodicMemory` with `NeuronPrint`-enriched observations.
**Current**: `EpisodicMemory.add(observation: &str, triplets: &[String], step: u64)`
**Target**: Each observation carries a gestalt (34-byte bundle) and a `NeuronTrace`.
**Encoding decision**: AriGraph should use **Base17 (34 bytes)** for episodic memory, NOT 3×16Kbit.
Rationale: Base17 has ρ=0.993, is L1-searchable, bundles via addition, and fits the same space
as weight vectors. 3×16Kbit (6 KB per edge) is for the blasgraph SPO store where you need
exact Hamming distance on full accumulator planes. Episodic memory is search-oriented, not
exact — Base17 is the right trade-off.
**File**: `crates/lance-graph/src/graph/arigraph/episodic.rs`

#### 2.2 Cognitive Edge Encoding for 6D SPO
**What**: Each AriGraph edge gets a `NeuronTrace` as a property, derived from the weight vectors
that encode that relationship.
**Encoding**: Edges use `SpoHead` (8 bytes: s_idx/p_idx/o_idx + NARS truth + Pearl mask).
The 6D enrichment comes from looking up the matching NeuronPrint at the SPO indices
and deriving the NeuronTrace. The edge doesn't store 204 bytes — it stores 8 bytes of
SpoHead and derives the trace on demand from the weight store.
```
Edge stored: SpoHead { s_idx: 42, p_idx: 17, o_idx: 88, freq, conf, pearl, ... }
Edge derived: NeuronTrace::from_neuron(weight_store.lookup(layer, feature))
```

#### 2.3 MetaCognition64 — Community Blumenstrauss
**What**: 64 community experts in a `Palette64`, connected by 8-layer causal topology.
**Struct** (new, verified against p64 types):
```rust
struct MetaCognition64 {
    topology: Palette64,           // p64::Palette64 { rows: [u64; 64] }
    experts: Vec<Community>,       // up to 64 bundled gestalts
    truths: [NarsTruth; 4096],     // one per angle (64×64)
    staunen: [f32; 64],            // Up/Down ratio per expert
    heel: HeelPlanes,              // p64::HeelPlanes { planes: [u64; 8] }
}
```
**Depends on**: 2.1 (episodic memory produces the observations that get clustered into communities).

### TIER 3: Rosetta exploration (understand what it says)

#### 3.1 DataFusion UDFs (Phase 1 of query language)
**What**: Register `l1`, `magnitude`, `xor_bind`, `bundle`, `neuron_trace`, `nars_revision`
as DataFusion scalar UDFs. Makes the 6D store queryable via raw SQL.
**File**: New `crates/lance-graph/src/neuron_udf.rs`
**Dep**: `datafusion = "51"` (already in Cargo.toml)

#### 3.2 Hydrate Real Model with Partitions
**What**: Run `hydrate_bgz7()` on existing bgz7 shards, verify `tensor_role` and `layer_idx`
columns are populated, write to Lance dataset.
**Test**: `SELECT tensor_role, COUNT(*) FROM weights GROUP BY tensor_role`
→ should show ~equal counts for Q/K/V/Gate/Up/Down per layer.

#### 3.3 Per-Role Palette Exploration
**What**: Build 6 palettes (one per TensorRole). Compare archetype distributions.
```sql
-- After UDFs registered:
SELECT tensor_role, layer_idx, AVG(magnitude(vector)) AS avg_mag
FROM weights
GROUP BY tensor_role, layer_idx
ORDER BY tensor_role, layer_idx
```
**Question**: Do Q archetypes cluster semantically? Does Gate magnitude predict neuron importance?

### TIER 4: Scale and test (make it real)

#### 4.1 Test with neural-debug Crate (20 steps)
```
 1. cargo test -p lance-graph --lib -- graph::neuron        (9 tests)
 2. cargo test -p lance-graph --lib -- graph::hydrate        (8 tests)
 3. cargo test -p lance-graph-planner --lib -- chat_bundle   (6 tests)
 4. cargo test -p lance-graph-planner --lib -- convergence   (7 tests)
 5. cargo test -p lance-graph-planner --lib -- nars_engine   (15 tests)
 6. cargo test --manifest-path crates/bgz17/Cargo.toml      (121 tests)
 7. cargo test --manifest-path crates/deepnsm/Cargo.toml    (22 tests)
 8. cargo test --manifest-path crates/bgz-tensor/Cargo.toml (12 tests)
 9. Hydrate one bgz7 shard → verify partition columns
10. Build Q-only palette from hydrated data → inspect 256 centroids
11. Build DistanceMatrix from Q palette → verify symmetry + self-zero
12. NeuronPrint::from_partitioned_data(layer=15, feature=42) → verify 6 roles
13. NeuronTrace::from_neuron() → verify frequency/confidence/attention ranges
14. NeuronQuery::attention(q).score(neuron) → verify lower=closer
15. triplet_to_headprint → headprint_to_spo → nars_infer chain
16. serve.rs: POST /v1/chat/completions with SPO extraction → inspect response
17. serve.rs: POST /v1/embeddings → verify 17-dim output
18. OSINT: fetch URL → extract triplets → add to AriGraph
19. Diff: compare two model shards via NeuronPrint L1 per role
20. End-to-end: ingest URL → extract → store → query → NARS reason → respond
```

#### 4.2 Contract Bridge Expansion for Consumers
**What**: Extend `lance-graph-contract` with:
```rust
// New traits for consumers (ladybug-rs, crewai-rust, n8n-rs):
pub trait NeuronProbe {
    fn probe(&self, query: NeuronQuery) -> Vec<(NeuronPrint, NeuronTrace)>;
}
pub trait EpisodicStore {
    fn observe(&mut self, observation: Base17, triplets: &[SpoHead]);
    fn recall(&self, query: Base17, k: usize) -> Vec<EpisodicResult>;
}
pub trait MetaCognitionBridge {
    fn think(&mut self, topic: Base17, style: ThinkingStyle) -> Insight;
}
```
**p64 highway**: The convergence point stays p64. Consumers use contract traits.
lance-graph-planner implements them. p64 routes between ndarray hardware and
lance-graph thinking.

#### 4.3 Persona Abstraction in AriGraph
**What**: Each AriGraph agent persona is a bundled NeuronPrint gestalt.
Different personas = different bundles = different retrieval patterns.
"Analytical Claude" has high Q·K alignment, strict Gate. "Creative Claude" has
broad K matching, high Up. The persona IS a 34-byte vector that biases all queries.

#### 4.4 Test with OpenClaw (Rust transcode vs OpenAI API)
**What**: Run the same queries through both:
```
A) serve.rs → SPO extraction → NARS reasoning → palette route (Rust, ~1μs)
B) OpenAI API → GPT-4o → response (API, ~2s)
```
Compare: does the SPO extraction find the same entities? Does NARS reasoning
reach similar conclusions? Where do they diverge?
**Purpose**: Calibration. The Rust path is 10⁶× faster but we need to verify
it's not 10⁶× dumber.

### TIER 5: Reader-LM for online hydration

#### 5.1 Google → reader-lm → 6D SPO
```
User asks about topic → Google search (via reader-lm fetch) → HTML
→ reader-lm strips to Markdown → DeepNSM extracts SPO triplets
→ triplet_to_headprint → headprint_to_spo → store in AriGraph
→ NeuronQuery probes weight store for matching patterns
→ NARS combines web evidence + model knowledge → response
```
**Key insight**: reader-lm + DeepNSM is the OSINT ingestion pipeline.
It runs at ~10μs per sentence with zero API cost. The 6D NeuronPrint
probing happens AFTER ingestion, when the query hits the weight store.

---

## Decision Record

### Q: Should AriGraph use bgz17 (34B) or 3×16Kbit bitpacked?
**A: bgz17 (34 bytes = Base17).** Episodic memory needs search, not exact reconstruction.
Base17 has ρ=0.993 distance preservation, bundles via addition, fits Lance vector columns,
and is the same space as weight vectors. 3×16Kbit (6 KB per edge) is for the blasgraph
SPO store where you need exact Hamming on full accumulators — a different use case.

### Q: What encoding for cognitive edges with 6D SPO?
**A: SpoHead (8 bytes) + on-demand NeuronTrace derivation.** The edge stores s_idx/p_idx/o_idx
(3 bytes of palette indices) + NARS truth (2 bytes) + Pearl mask (1 byte) + metadata (2 bytes).
The full 6D NeuronPrint (204 bytes) is looked up from the weight store when needed, not stored
per-edge. This keeps edge size at 8 bytes while giving 204-byte richness on demand.

### Q: How to make the stack AGI?
**A: `AGI::p64(topic, angle)`.** The 4 KB MetaCognition64 struct IS the inference engine.
64 community experts. 4096 NARS-propagated angles. 8 causal layers. Style-driven method
dispatch. Gate-controlled hydration. The model's weights are the offline compilation
(5M × 34 bytes in Lance). At runtime, one function call routes through the topology.

---

## Code Review Fixes Applied This Commit

1. ~~`palette_indices` dead field on AutocompleteCache~~ → removed
2. ~~`NeuronPrint::BYTE_SIZE` misleading name~~ → renamed to `PAYLOAD_SIZE`
3. ~~`weight_schema()` unused function~~ → removed (schema inferred from `try_from_iter`)
4. ~~Unused imports in hydrate.rs~~ → cleaned (`DataType`, `Field`, `Schema`)
