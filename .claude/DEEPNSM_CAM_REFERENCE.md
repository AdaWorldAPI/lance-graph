# DeepNSM-CAM: Quick Reference for CC Sessions

## What Is It

A semantic processing layer that replaces transformer inference with
precomputed distributional lookup. 4,096 words × 12 bits. 8MB distance
matrix. No GPU. No learned weights. Same decision boundaries as cosine.

## Where Things Live

### Data (AdaWorldAPI/DeepNSM)
```
word_frequency/
├── word_rank_lookup.csv     ← START HERE: rank → word → PoS → freq
├── word_forms.csv           ← lemma → inflected forms (11,460 entries)
├── cam_codes.bin            ← 5,050 × 6 bytes (CAM-PQ fingerprints)
├── codebook_pq.bin          ← 96KB (6×256×16 centroids)
├── nsm_primes.json          ← 63 NSM semantic primes → CAM codes
└── codebook_pq.json         ← codebook + normalization params
```

### Codec (AdaWorldAPI/ndarray)
```
src/hpc/cam_pq.rs            ← CamCodebook, DistanceTables, PackedDatabase, AVX-512
```

### Engine (AdaWorldAPI/lance-graph)
```
crates/bgz17/
├── src/base17.rs             ← Base17::l1(), SpoBase17
├── src/distance_matrix.rs    ← DistanceMatrix, SpoDistanceMatrices
├── src/bridge.rs             ← Bgz17Distance trait, Precision enum
└── src/similarity.rs         ← SimilarityTable (TO BUILD: session_bgz17_similarity.md)

crates/lance-graph/src/cam_pq/
├── storage.rs                ← Arrow schema: FixedSizeBinary(6) CAM columns
├── udf.rs                    ← DataFusion UDF: cam_distance()
├── ivf.rs                    ← IVF billion-scale partitioning
└── jitson_kernel.rs          ← JIT-compiled stroke cascade

crates/lance-graph-planner/src/thinking/
├── graph.rs                  ← ThinkingGraph: 36 styles as AdjacencyStore
├── process.rs                ← 13 cognitive verbs + composable programs
├── topology.rs               ← 12×12 NARS adjacency + learning
├── persistence.rs            ← binary serialize/deserialize topology
└── mod.rs                    ← orchestrate_with_topology()
```

### Documentation (this repo)
```
docs/deepnsm_cam_architecture.md   ← Full architecture (THIS file's companion)
.claude/prompts/session_deepnsm_cam.md  ← CC session prompt (7 deliverables)
.claude/DEEPNSM_CAM_REFERENCE.md   ← THIS FILE
```

## Key Numbers

```
Vocabulary:        4,096 words (98.4% text coverage)
Token size:        12 bits (16 bits with PoS)
SPO triple:        36 bits (in u64)
Distance matrix:   8 MB u8 (L2 cache)
SimilarityTable:   512 bytes (L1 cache)
Context window:    220 KB (±5 sentences)
Total runtime:     ~9 MB
NSM primes:        62/63 in vocabulary
Full pipeline:     < 10μs per sentence
```

## Key Insight Chain

1. COCA corpus has 96 subgenre frequency dimensions per word
2. These ARE distributional semantics (same thing transformers learn)
3. Pairwise distances on 96D vectors = semantic relationships
4. For 4,096 words, precompute ALL 8.4M pairwise distances
5. Store as 8MB u8 matrix → fits L2 cache
6. bgz17 SimilarityTable calibrates raw distance → f32 [0,1]
7. Same decision boundaries as cosine similarity
8. XOR role binding gives word-order sensitivity (SPO ≠ OPS)
9. ±5 sentence context window gives disambiguation
10. Graph structure provides entity-level disambiguation
11. No learned parameters anywhere in the pipeline

## Session Execution Order

```
PRIORITY  SESSION                          REPO        DEPENDS ON
────────  ───────                          ────        ──────────
P0        bgz17 SimilarityTable            ndarray     bgz17 Cascade
P0        DeepNSM D7 (build pipeline)      DeepNSM     word_frequency data
P1        DeepNSM D1 (tokenizer)           DeepNSM     D7 artifacts
P1        DeepNSM D4 (similarity)          DeepNSM     D7 + bgz17 SimilarityTable
P2        DeepNSM D2 (parser)              DeepNSM     D1
P2        DeepNSM D3 (encoder)             DeepNSM     D1 + D2 + D4
P3        DeepNSM D5 (ThinkingGraph)       lance-graph D3 + thinking/graph.rs
P3        DeepNSM D8 (context window)      DeepNSM     D3
P4        DeepNSM D6 (DataFusion UDFs)     lance-graph D3 + cam_pq/udf.rs
```

## Disambiguation Architecture (Three Layers)

```
Layer 1: Graph          Named entities = unique node IDs (zero ambiguity)
Layer 2: Distribution   96D COCA genre vectors (implicit disambiguation)
Layer 3: Context        ±5 sentence window (explicit disambiguation)

Transformer equivalent: all three layers are learned via attention
DeepNSM:                all three layers are structural (zero weights)
```

## SPO Distance = 3 Matrix Lookups

```rust
// The entire semantic comparison operation:
let d_subject   = matrix[a.subject()][b.subject()];     // 1 lookup
let d_predicate = matrix[a.predicate()][b.predicate()];  // 1 lookup
let d_object    = matrix[a.object()][b.object()];        // 1 lookup
let similarity  = table.similarity(d_subject + d_predicate + d_object); // 1 lookup
// Total: 4 memory accesses. < 15ns.
```

## Cognitive Verb Integration

```
Verb           Semantic Operation                    Uses
────           ──────────────────                    ────
EXPLORE        weight by distributional similarity   distance_matrix
HYPOTHESIS     ground in NSM decomposition           nsm_primes
COUNTERFACTUAL negate predicate plane                SPO per-role distance
SYNTHESIS      merge if pred_sim > 0.85              SimilarityTable
ABDUCTION      scan column for plausible subjects    distance_matrix column
INTERRELATE    bridge via shared predicates           per-role decomposition
DEEPEN         NSM prime decomposition               nsm_primes + distances
MODULATE       ThinkingStyle from content PoS         PoS tags
```

## Related Session Prompts

| Prompt | Repo | What |
|--------|------|------|
| `session_deepnsm_cam.md` | all three | 7+1 deliverables, 24 tests |
| `session_bgz17_similarity.md` | ndarray + lance-graph | SimilarityTable spec |
| `CAM_PQ_SPEC.md` | lance-graph | CAM-PQ codec integration |
| `session_unified_vector_search.md` | lance-graph | vector search wiring |
