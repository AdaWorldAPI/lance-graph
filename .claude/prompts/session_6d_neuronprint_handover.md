# SESSION HANDOVER: 6D NeuronPrint + Partitioned CAM + Cypher Query Language

> **Date**: 2026-03-31
> **Branch**: `claude/qwen-claude-reverse-eng-vHuHv` (both repos)
> **Last commit**: lance-graph 4e8b960, ndarray unchanged this session

---

## What Was Built This Session

### 1. serve.rs — SPO Extraction + NARS Reasoning (lance-graph-planner)

**File**: `crates/lance-graph-planner/src/serve.rs`

The OpenAI-compatible REST endpoint now decomposes messages into SPO triplets
instead of brute-force vector search:

```
message → extract_triplets() → (S, P, O) strings
→ triplet_to_headprint(S, P, O) → HeadPrint (S:6, P:6, O:5 dims)
→ headprint_to_spo() → SpoHead (palette indices + NARS truth)
→ nars_infer() deduction/abduction against knowledge base
```

Key insight: messages are decomposed at SPO level (like AriGraph does),
not hashed into flat fingerprints. The palette/DistanceMatrix/SimilarityTable
infrastructure is for the SPO triple store path, not for query-time search.

### 2. hydrate.rs — Partitioned CAM Index (lance-graph core)

**File**: `crates/lance-graph/src/graph/hydrate.rs`

Arrow RecordBatch schema now includes partition columns:

```
tensor_name: Utf8         — full tensor path
row_idx: UInt32           — row within tensor
layer_idx: UInt16         — parsed from tensor name (nullable for non-layer tensors)
tensor_role: UInt8        — TensorRole enum (Q=0, K=1, V=2, O=3, Gate=4, Up=5, Down=6, ...)
vector: FixedSizeList(f32, 17) — for Lance ANN/RaBitQ
base17: FixedSizeList(i16, 17) — for direct L1 / palette
palette_s/p/o: UInt8      — SPO palette indices (populated later)
```

`TensorRole::from_name()` parses HuggingFace + GGUF naming conventions.
`parse_layer_idx()` extracts layer number. No re-extraction from models needed.

**Tests**: 9 passing (tensor_role_parsing, layer_idx_parsing, partition_columns_populated, etc.)

### 3. neuron.rs — 6D Holographic Neuron Representation (lance-graph core)

**File**: `crates/lance-graph/src/graph/neuron.rs`

Three structs:

```rust
NeuronPrint  // 204 bytes: Q/K/V/Gate/Up/Down — what a neuron IS
NeuronQuery  // Selective role probing with Option<Base17> per role — how you ASK
NeuronTrace  // NARS truth from role ratios — how it REASONS
```

Key methods:
- `NeuronPrint::bundle()` → 34-byte holographic gestalt
- `NeuronPrint::attention()` → Q ⊕ K (retrieval fingerprint)
- `NeuronPrint::mlp()` → Gate ⊕ Up ⊕ Down (transform fingerprint)
- `NeuronQuery::attention(q)` → probes K store only
- `NeuronQuery::score(neuron)` → L1 on active roles only
- `NeuronQuery::role_mask()` → 6-bit mask (Q/K/V/Gate/Up/Down)
- `NeuronTrace::from_neuron()` → derives NARS f/c/attention/coherence/expectation

**Tests**: 9 passing

### 4. Documentation

- `docs/NEURONPRINT_ROSETTA.md` — Epiphanies, LLM architecture zoo, unanswered questions
- `docs/NEURON_QUERY_LANGUAGE.md` — Cypher/GQL extension design, DataFusion UDFs, 4-phase plan

---

## Key Epiphanies

### The 6 tensor roles ARE 6 dimensions of one neuron
Each neuron (layer i, feature j) has the same row index across Q/K/V/Gate/Up/Down.
204 bytes = complete behavioral fingerprint. Bundle all 6 → 34 bytes holographic.

### Two triads = 6D SPO
- Attention triad: Q=Subject, K=Predicate, V=Object (communication)
- MLP triad: Gate=Subject, Up=Predicate, Down=Object (computation)
- Each triad is an SPO decomposition → Pearl 2⁶ instead of Pearl 2³

### K+V = retrieval store, Q = query, Gate+Up+Down = NARS hydration
Retrieval and reasoning are separate operations on the same aligned data.
NeuronQuery selects which roles participate via a 6-bit role mask.

### The palette is a cleanup memory, not a search engine
For queries: direct L1 on 34-byte Base17 (17 subtractions, sub-μs).
For SPO triple store (millions of edges): palette → DistanceMatrix → O(1).
For cleanup after VSA unbind: palette.nearest() snaps noisy bundle to archetype.

### Base17 = the Lindenstrauss projection
Golden-step codec compresses BF16 d_model → 17 dims, ρ=0.993. No need for
a second random projection on top. The Hyperprobe paper's 55M-param neural
encoder does what golden-step gives for free deterministically.

### HHTL cascade over 5M vectors
- HEEL: model gestalt (34 bytes) → "is this query in this model's space?"
- HIP: per-layer or per-role bundles (136 KB) → "which region?"
- TWIG: palette 256×256 distance table (128 KB) → "which archetype?"
- LEAF: 5M vectors in Lance + RaBitQ → "which exact weight row?"

---

## Loose Ends

### Must Fix
1. **`message_to_base17()` in serve.rs is still a byte hash** — needs to use
   `triplet_to_headprint()` (which it now does for the SPO path) but the
   embedding endpoint still uses the old hash. Low priority since embeddings
   endpoint is secondary.

2. **`AutocompleteCache.palette_indices` field was added but is unused** after
   the refactor from palette pipeline to direct SPO. Can be removed or
   repurposed for NeuronPrint palette assignment.

### Should Do (Next Session)
3. **Register DataFusion UDFs** — `l1`, `magnitude`, `xor_bind`, `bundle`,
   `neuron_trace`, `nars_revision`. Pure scalar functions, no Cypher changes.
   This makes the 6D store queryable via raw SQL immediately.
   **File**: `crates/lance-graph/src/nsm/` or new `crates/lance-graph/src/neuron_udf.rs`

4. **Hydrate a real model with partition columns** — run hydrate on existing
   bgz7 files, verify tensor_role and layer_idx are correctly populated,
   write to Lance dataset, query with the UDFs.

5. **Build per-role palettes** — instead of one palette for all 5M vectors,
   build 6 palettes (one per tensor role). Compare archetype distributions.
   Do Q archetypes cluster semantically?

6. **NeuronPrint construction from partitioned Lance data** — given a (layer, feature)
   pair, load Q/K/V/Gate/Up/Down rows from the 6 partitions, assemble NeuronPrint.
   This is the hydration step that creates the 204-byte struct from stored data.

### Could Explore (Rosetta Stone)
7. **Q·K alignment per layer** — does attention sharpness increase with depth?
   `SELECT layer_idx, AVG(l1(q.vector, k.vector)) FROM weights GROUP BY layer_idx`

8. **Gate magnitude distribution** — which layers have the most active gates?
   Are early layers feature detectors (low gate, broad) and late layers
   concept composers (high gate, selective)?

9. **Up/Down ratio as polysemanticity detector** — monosemantic neurons should
   have low Up and low Down (clean pass-through). Polysemantic neurons should
   have high both (many features, aggressive compression).

10. **Cross-model NeuronPrint diff** — compare Opus 4.5 vs 4.6 per-role.
    Which roles diverge? Which layers? This localizes behavioral differences.

11. **AriGraph episodic memory with NeuronPrint** — replace string triplets
    with NeuronTriplet { q, k, v, gate, up, down }. Episodic retrieval
    becomes NeuronQuery::attention(q) instead of Hamming on fingerprints.
    The NARS truth comes from Gate/Up/Down ratio instead of heuristics.

12. **Cypher extension (Phase 2)** — add Neuron node type, role relationships,
    USING ROLES() clause, trace property access to the nom-based parser.

---

## Architecture Map

```
ndarray (unchanged this session)
├── src/hpc/bgz17_bridge.rs    — Base17 type, SIMD L1, xor_bind
├── src/hpc/palette_distance.rs — Palette::build(), DistanceMatrix, SimilarityTable
├── src/hpc/nars.rs             — NarsTruth type
├── crates/p64/src/lib.rs       — Palette64, HHTL cascade
└── src/hpc/gguf_indexer.rs     — read_bgz7_file(), CompressedTensor

lance-graph
├── crates/lance-graph/src/graph/
│   ├── neuron.rs          [NEW] — NeuronPrint, NeuronQuery, NeuronTrace (9 tests)
│   ├── hydrate.rs     [UPDATED] — TensorRole, parse_layer_idx, partition columns (9 tests)
│   ├── arigraph/                 — TripletGraph, EpisodicMemory (existing)
│   └── fingerprint.rs           — 512-bit Fingerprint, Hamming (existing)
├── crates/lance-graph-planner/src/
│   ├── serve.rs       [UPDATED] — SPO extraction + NARS reasoning endpoint
│   ├── cache/convergence.rs      — triplet_to_headprint, headprint_to_spo (existing)
│   └── cache/nars_engine.rs      — SpoHead, NarsEngine, Pearl 2³ (existing)
├── crates/bgz17/src/
│   ├── palette.rs                — Palette::build(), nearest() (existing)
│   ├── distance_matrix.rs        — DistanceMatrix, SpoDistanceMatrices (existing)
│   └── similarity.rs             — SimilarityTable, from_reservoir() (existing)
├── crates/bgz-tensor/src/
│   ├── palette.rs                — WeightPalette (CLAM-inspired, existing)
│   └── attention.rs              — AttentionTable, ComposeTable, CompiledHead (existing)
└── docs/
    ├── NEURONPRINT_ROSETTA.md  [NEW] — Epiphanies, architecture zoo, exploration plan
    └── NEURON_QUERY_LANGUAGE.md [NEW] — Cypher extension design, UDFs, 4-phase plan
```

---

## Commits This Session (lance-graph)

```
4e8b960 docs: 6D SPO query language design — Cypher/GQL extension for NeuronPrint
8650b4a docs: NeuronPrint Rosetta Stone — 6D holographic neuron representation
41f6b73 feat: NeuronPrint + NeuronQuery + NeuronTrace — 6D holographic neuron representation
6f59d5c feat: partitioned CAM index — TensorRole + layer_idx from tensor names
5f07f3a feat: wire SPO extraction + NARS reasoning into serve.rs endpoint
85d1c41 refactor: direct L1 search on raw Base17 vectors, keep palette infra
c680c02 feat: wire bgz17 Palette→DistanceMatrix→SimilarityTable into serve.rs + Lance write
```

---

## How to Continue

### Quick Start (15 min)
```bash
cd /home/user/lance-graph
git checkout claude/qwen-claude-reverse-eng-vHuHv
cargo test -p lance-graph --lib -- graph::neuron   # 9 tests
cargo test -p lance-graph --lib -- graph::hydrate   # 9 tests
cargo check -p lance-graph-planner --features serve # compiles clean
```

### Rosetta Exploration (needs bgz7 files)
```bash
# 1. Hydrate with partition columns
# (needs bgz7 files in /tmp/ from previous indexing session)
cargo test -p lance-graph --lib -- graph::hydrate::tests::test_hydrate_real

# 2. Register DataFusion UDFs (Phase 1 of query language)
# Create crates/lance-graph/src/neuron_udf.rs with l1, magnitude, etc.

# 3. Query the 6D store
# SELECT tensor_role, layer_idx, COUNT(*) FROM weights GROUP BY tensor_role, layer_idx
```

### Key Question for Next Session
**"What do the Q archetypes look like?"** — build a palette from only Q-role vectors,
inspect the 256 centroids, see if they cluster by semantic function. This is the
first Rosetta reading. Everything else follows from what you find there.

---

## External References

- **Hyperprobe paper**: arXiv 2509.25045 — validates residual→VSA→algebra approach.
  Their 55M-param encoder = our zero-param golden-step projection.
  GitHub: `Ipazia-AI/hyperprobe` (cloned to `/home/user/hyperprobe/`)

- **Anthropic Monosemanticity** (2024): individual neurons represent single concepts.
  NeuronPrint should capture this — monosemantic = tight fingerprint across all 6 roles.

- **SwiGLU analysis** (Shazeer 2020): Gate acts as learned binary mask.
  NeuronTrace.frequency is derived from Gate magnitude — validates the mapping.

- **Residual stream as communication bus** (Elhage et al. 2021): all layers read/write
  the same bus. NeuronPrint captures both read (Q/K) and write (V/Down) sides.

- **Original AriGraph**: AdaWorldAPI/AriGraph (Python), used 768D Contriever embeddings.
  Transcoded to lance-graph with DeepNSM (0 params, 16.5 MB, bit-exact).
