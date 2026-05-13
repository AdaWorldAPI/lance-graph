# DeepNSM + Thinking Pipeline Architecture

## Overview

DeepNSM is a semantic processing layer that gives lance-graph the ability to
understand MEANING and GRAMMAR of natural language — without transformers,
without GPU, without learned weights.

```
4,096 words × 12 bits × 8MB distance matrix = complete semantic engine
O(1) per word, O(n) per sentence, deterministic, bit-reproducible
```

## The Three Replacements

### 1. Embedding Matrix → Distance Matrix (GPU replacement)

```
Transformer:  token → embedding_matrix[token] → 768 floats → GPU matmul → similarity
              ~3M learned parameters, 1.5GB model, non-deterministic

DeepNSM:      token → distance_matrix[a][b]   → 1 byte    → table lookup → similarity
              0 learned parameters, 8MB model, bit-exact
```

The distance matrix IS the learned knowledge. Precomputed from 1 billion words
of COCA corpus distributional statistics across 96 genre contexts. Same information
that transformers learn in the embedding layer, just derived from corpus statistics
instead of gradient descent.

### 2. Attention → SPO Role Binding (context replacement)

```
Transformer:  tokens → Q·K^T/√d → attention weights → context-weighted output
              O(n²), learned attention heads, non-deterministic

DeepNSM:      tokens → PoS parse → XOR role bind → majority bundle → SPO triple
              O(n), structural (not learned), bit-exact
```

Word order sensitivity comes from role binding: "dog bites man" ≠ "man bites dog"
because XOR(dog, ROLE_SUBJECT) ≠ XOR(dog, ROLE_OBJECT). The role vectors are
fixed random patterns, not learned weights. The PoS-driven parser is a 6-state FSM.

### 3. Contextual Disambiguation → Streaming Context Window

```
Transformer:  full self-attention across all tokens → contextual embeddings
              "bank" near "river" → river-bank embedding
              "bank" near "money" → financial-bank embedding

DeepNSM:      ±5 sentence sliding window → bundle context → XOR with word
              "bank" + financial context bundle → financial-colored vector
              O(1) per sentence update, no recomputation
```

The context window replaces attention for local disambiguation. Each sentence is
VSA-encoded and pushed into a ring buffer. The running bundle (majority vote) IS
the context. XOR-binding a word with the context shifts its representation toward
the contextually appropriate meaning.

## Data Model

### 12-Bit Vocabulary

4,096 words from COCA frequency ranking. 98.4% coverage of running English text.
62/63 Wierzbicka NSM semantic primes present.

```
Word entry:
  rank:    u12    (1 = "the", 4096 = "journalism")
  lemma:   &str   (canonical form)
  pos:     PoS    (13 tags: n, v, j, r, i, p, c, d, m, u, a, x, t, e)
  freq:    u32    (raw frequency in 1B-word corpus)
  disp:    f32    (dispersion 0-1: evenness across texts)
  forms:   [..]   (inflected forms: "bit"→"bite", "ran"→"run")
  vector:  [f32;96]  (96D subgenre frequency distribution)
```

### SPO Triple (36 bits)

```
[Subject:12][Predicate:12][Object:12] = 36 bits in u64

"dog bites man" → SPO(671, 2943, 95) = 0x029F_B7F_05F
```

Distance between triples = 3 matrix lookups (subject + predicate + object).
Per-role decomposition: WHO did WHAT to WHOM, each scored independently.

### Distance Matrix (8MB)

Precomputed from exact 96D distributional vectors. Symmetric, palette-quantized to u8.

```
4,096 × 4,096 × u8 / 2 = 8 MB (upper triangle)
Palette: 256 levels → actual distance values

distance(word_a, word_b) = palette[matrix[a][b]]  → ONE memory access
```

### SimilarityTable (512 bytes)

Built from the EXACT 4,096² = 8.4M pair distribution. Not sampled.

```
256 × f16 = 512 bytes
similarity(distance) = 1.0 - CDF(distance)  → O(1)

Calibrated so:
  similarity(think, know)  ≈ 0.92  (close, same NSM field)
  similarity(think, big)   ≈ 0.65  (far, different fields)
  similarity(random pair)  ≈ 0.50  (baseline)
```

## Streaming Context Window

±5 sentence sliding window replaces transformer attention for local disambiguation.

```rust
struct ContextWindow {
    buffer: [BitVec; 11],    // ring buffer: current ± 5 sentences
    counts: Vec<i16>,        // 10K entries for incremental majority
    context: BitVec,          // running bundle (majority vote)
}
```

**Operations:**
- `push(sentence)` — O(1): XOR-out oldest, XOR-in newest, recount
- `contextualize(word)` — O(1): word XOR context → disambiguated vector
- `disambiguation_strength(word)` — O(1): how much context shifted meaning

**How it disambiguates:**

"Bank" in a financial article (context = {Goldman, stock, earnings, lending}):
- Context bundle has high activation in financial-genre dimensions
- XOR("bank", context) → financial-colored bank vector
- Distance to "lend" DECREASES (more similar in context)
- Distance to "river" INCREASES (less similar in context)

"Bank" near a river description (context = {river, flow, water, fish}):
- Context bundle has high activation in nature-genre dimensions
- XOR("bank", context) → geographic-colored bank vector
- Distance to "shore" DECREASES
- Distance to "lend" INCREASES

Same 12-bit token. Different context bundle. Different effective meaning.
No weights. No attention. The context IS the data.

**Graph-level disambiguation:**

Named entities (Goldman Sachs, Thames) are graph node IDs, not vocabulary tokens.
They arrive pre-disambiguated by type:

```
(:Company {name: "Goldman Sachs"})-[:IS_A]->(:Bank)    ← financial
(:River {name: "Thames"})-[:HAS]->(:Bank)              ← geographic
```

The label IS the context. The edge IS the attention. Graph structure eliminates
the disambiguation problem that transformers solve with 400M parameters.

## Cognitive Pipeline Integration

The thinking pipeline (`crates/lance-graph-planner/src/thinking/`) uses
DeepNSM to reason about MEANING, not string patterns.

### Architecture

```
Raw text
  → nsm_tokenizer    12-bit tokens + PoS          O(n)
  → nsm_parser       SPO triples via 6-state FSM  O(n)
  → nsm_encoder      XOR bind + bundle → BitVec   O(n)
  → ContextWindow    ±5 sentence disambiguation   O(1) update
  → nsm_similarity   calibrated f32 [0,1]         O(1) lookup
  → ThinkingGraph    cognitive verbs on meaning    O(verbs)
```

### Verb Enhancements

- **EXPLORE**: weight neighbor activation by semantic similarity of typical queries
- **HYPOTHESIS**: ground in NSM decomposition ("journalism" → news + write + people)
- **COUNTERFACTUAL**: negate PREDICATE plane, find antonym by high S/O + low P similarity
- **SYNTHESIS**: merge triples by per-role similarity (pred_sim > 0.85 → merge)
- **ABDUCTION**: scan matrix column for plausible subjects given an object
- **INTERRELATE**: detect cross-domain analogies via shared predicates
- **DEEPEN**: NSM prime decomposition (word → constituent primes)
- **MODULATE**: shift ThinkingStyle based on content PoS (mental → Metacognitive, action → Pragmatic)

## Memory Budget

```
Component               Size       Cache     Access Pattern
────────────────        ────       ─────     ──────────────
Vocabulary hash         32 KB      L1        tokenize: per word
Forms hash              64 KB      L1/L2     inflection: per word
SimilarityTable         512 B      L1        calibrate: per distance
Role vectors            7.5 KB     L1        bind: per triple
NSM prime indices       2 KB       L1        decompose: per word
Context window          220 KB     L2        push: per sentence
Distance matrix (u8)    8 MB       L2        distance: per pair
Vectors (BF16)          768 KB     L2        VSA encode: per word
OOV buckets             4 KB       L1        OOV resolve: per OOV word
────────────────        ────       ─────     ──────────────
TOTAL                   ~9.1 MB    L2/L3     Everything in cache
```

## Performance Targets

```
Operation              Target         Method
─────────              ──────         ──────
tokenize(word)         < 100ns        hash lookup
tokenize(sentence)     < 1μs          O(n) scan
parse(sentence)        < 500ns        6-state FSM
triple_distance        < 10ns         3 matrix lookups
triple_similarity      < 15ns         3 lookups + table
context_push           < 1μs          XOR-in/XOR-out + recount
contextualize(word)    < 50ns         XOR with context bundle
sentence_similarity    < 5μs          VSA compose + hamming
decompose(word)        < 1μs          scan 63 primes
full pipeline          < 10μs         text → calibrated similarity
```

## Cross-Repository Map

```
AdaWorldAPI/DeepNSM
  word_frequency/              ← COCA data + CAM-PQ codebook
  .claude/prompts/session_deepnsm_cam.md

AdaWorldAPI/ndarray
  src/hpc/cam_pq.rs            ← CamCodebook codec (encode/decode/distance)
  .claude/prompts/session_bgz17_similarity.md
  .claude/prompts/session_deepnsm_cam.md

AdaWorldAPI/lance-graph
  crates/bgz17/                ← Base17 + SimilarityTable + DistanceMatrix
  crates/lance-graph/cam_pq/   ← DataFusion UDF + Arrow storage + IVF
  crates/lance-graph-planner/
    src/thinking/
      graph.rs                 ← ThinkingGraph (36 styles on AdjacencyStore)
      process.rs               ← 13 cognitive verbs + programs
      topology.rs              ← 12×12 NARS adjacency + learning
      persistence.rs           ← topology save/load
      mod.rs                   ← orchestrate_with_topology()
  docs/DEEPNSM_ARCHITECTURE.md ← this file
  .claude/prompts/
    session_thinking_topology.md
    session_deepnsm_cam.md
```

## Why This Works (Theoretical Grounding)

1. **Harris 1954** — distributional hypothesis: words in similar contexts have
   similar meanings. The 96D COCA vector IS this. Transformers rediscovered it
   with more compute.

2. **Wierzbicka 1972** — Natural Semantic Metalanguage: all concepts decompose
   into ~65 semantic primes. 62/63 primes are in the top 4,096 COCA words.
   The vocabulary IS semantically complete.

3. **Kanerva 2009** — hyperdimensional computing: XOR binding + majority bundling
   on high-dimensional binary vectors preserves compositional structure. This is
   the theoretical basis for VSA role binding.

4. **Zipf 1949** — word frequency follows power law. The top 4,096 words cover
   98.4% of text. The remaining 1.6% decomposes into combinations of the top 4,096.
   The vocabulary IS sufficient.

5. **Fibonacci non-commensurability** — Fibonacci-spaced basis vectors never alias
   because consecutive Fibonacci ratios approach φ (irrational). SPO planes in
   bgz17 use this property for exact bundling/unbundling without interpolation.
