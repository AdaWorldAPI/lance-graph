# DeepNSM-CAM: Architecture Reference

> 4,096 words × 12 bits. No transformer. No GPU. No learned weights.
> 8MB distance matrix replaces 400M parameters.
> O(1) per word, O(n) per sentence. Exact, deterministic, bit-reproducible.

## 1. The Core Claim

Distributional semantics from a 1-billion-word corpus (COCA) gives you the same
semantic relationships that transformers learn — but as a precomputed lookup table
instead of a neural network. The distance matrix IS the model.

```
Transformer:  token → embedding_matrix → 768D → attention → cosine → similarity
              ~400M learned parameters, GPU, non-deterministic

DeepNSM:      token → distance_matrix[a][b] → u8 → SimilarityTable → similarity
              0 learned parameters, CPU, bit-exact
```

## 2. Why 4,096 Words

| Top N  | Coverage | Boundary Word | Cache Level |
|--------|----------|---------------|-------------|
| 1,024  | 84.0%    | "adult"       | —           |
| 2,048  | 91.9%    | "invest"      | —           |
| **4,096** | **98.4%** | **"journalism"** | **L2 (8MB)** |
| 8,192  | 99.2%    | —             | L3 (64MB)   |

Key properties at 4,096:
- 62/63 NSM semantic primes present
- 99.0% have dispersion > 0.8 (well-distributed across genres)
- 12 bits per word = clean binary representation
- 8MB u8 distance matrix fits L2 cache
- Everything beyond 4,096 decomposes into core vocabulary combinations

## 3. Zero Weights

```
The transformer needs weights because it starts from arbitrary BPE token IDs
and must LEARN what they mean.

The 12-bit vocabulary doesn't need weights because the token ID IS the meaning:
- Rank IS frequency ordering from 1 billion words of real usage
- Relationships are precomputed from distributional co-occurrence across 96 genres
- Distance matrix IS the embedding: matrix[a][b] = semantic distance

The two things transformers use weights for:
1. "What does this word mean?" → DeepNSM: distance_matrix row
2. "How do words compose?"    → DeepNSM: XOR role binding (structural, not learned)
```

## 4. Three-Layer Disambiguation (No Attention)

```
Layer 1: Graph structure     Named entities = unique node IDs (zero ambiguity)
Layer 2: Genre distribution  96D COCA vectors separate word senses by usage
Layer 3: Context window      ±5 sentence bundle shifts word representation
```

## 5. Grammar via PoS FSM (No Regex)

```
6-state FSM: START → DET? → ADJ* → NOUN+ → VERB → DET? → ADJ* → NOUN+
Handles ~85% of English SVO. Secondary patterns for passive/relative/existential.
PoS tags from vocabulary. Word forms resolved via 11,460-entry hash table.
```

## 6. SPO Composition via XOR Binding

```
BIND:    word XOR role → bound representation (order-sensitive)
BUNDLE:  majority_vote(bound_S, bound_P, bound_O) → sentence vector
UNBIND:  sentence XOR role → recover component

Verified: "dog bites man" vs "man bites dog" = 0.75 similarity (different!)
          Without binding: 1.0 (identical, can't distinguish)
```

Maps directly to bgz17 SpoBase17 plane separation + Fibonacci spacing.

## 7. Streaming Context Window

```
±5 sentence ring buffer. O(1) push via incremental majority vote.
word XOR context → disambiguated representation. ~220KB working memory.
Replaces transformer attention for local (paragraph-level) disambiguation.
```

## 8. Memory Budget

```
Component               Size       Cache Level
────────────────        ────       ───────────
Vocabulary hash         32 KB      L1
Forms hash              64 KB      L1/L2
SimilarityTable         512 B      L1
Role vectors            7.5 KB     L1
NSM prime indices       2 KB       L1
Context window          220 KB     L2
Distance matrix (u8)    8 MB       L2
Vectors (BF16)          768 KB     L2
OOV buckets             4 KB       L1
TOTAL                   ~9.1 MB    L2/L3
```

## 9. Performance Targets

```
tokenize(word)         < 100ns     hash lookup
triple_distance        < 10ns      3 matrix lookups
triple_similarity      < 15ns      3 lookups + table
context_push           < 1μs       XOR-in/XOR-out
disambiguate(word)     < 100ns     word XOR context
full pipeline          < 10μs      text → calibrated similarity
```

## 10. What Transformers Do Better

1. Long-range dependencies (>±5 sentences)
2. Learned disambiguation from billions of examples
3. Text generation (DeepNSM analyzes, doesn't generate)
4. Rare/technical vocabulary precision
5. Multilingual (COCA is English-only; architecture generalizes, data doesn't)

None affect the primary use case: thinking pipeline reasoning about query semantics.

## 11. Cross-Repo Map

```
DeepNSM:     word_frequency/ (data) + .claude/prompts/ (session)
ndarray:     src/hpc/cam_pq.rs (codec) + .claude/prompts/ (session)
lance-graph: crates/bgz17/ (similarity) + crates/lance-graph/cam_pq/ (wiring)
             crates/lance-graph-planner/src/thinking/ (cognitive verbs)
             docs/deepnsm_cam_architecture.md (THIS DOCUMENT)
```
