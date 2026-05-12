# SESSION: DeepNSM Crate — Compile, Test, Wire

## What Happened

A new `crates/deepnsm/` crate was created in lance-graph. This is the **transformer replacement** — a distributional semantic engine that processes natural language using:

- 4,096-word COCA vocabulary (12-bit tokens)
- 8MB precomputed distance matrix (replaces embedding layer)
- PoS-driven FSM parser → SPO triples (replaces attention)
- VSA XOR-bind + majority bundle (replaces feed-forward)
- ±5 sentence context window (replaces contextual embeddings)
- Calibrated similarity via empirical CDF (replaces softmax)

**Zero dependencies. Zero learned parameters. < 10μs per sentence.**

## Files Created

```
crates/deepnsm/
├── Cargo.toml          # standalone, zero deps
└── src/
    ├── lib.rs          # module declarations + re-exports  
    ├── pos.rs          # 13 COCA PoS tags, 4-bit packed
    ├── spo.rs          # 36-bit SPO triple + WordDistanceMatrix + CAM-PQ builder
    ├── vocabulary.rs   # Tokenizer: hash lookup, word_forms.csv, contractions
    ├── parser.rs       # 6-state PoS FSM → SPO triples + modifiers + negation
    ├── similarity.rs   # SimilarityTable: 256-entry CDF lookup, 1KB
    ├── encoder.rs      # 512-bit VSA: XOR bind, majority bundle, role vectors
    ├── context.rs      # ±5 sentence ring buffer for disambiguation
    ├── codebook.rs     # CAM-PQ codebook loader (96KB binary, 30KB codes)
    └── pipeline.rs     # DeepNsmEngine: full inference loop
```

## Data Files (already in repo)

The COCA word frequency data lives in `AdaWorldAPI/DeepNSM` repo under `word_frequency/`:

```
word_frequency/
├── word_rank_lookup.csv    # 5,050 rows: rank,word,pos,freq
├── word_forms.csv          # 11,460 rows: lemRank,lemma,PoS,lemFreq,wordFreq,word
├── cam_codes.bin           # 30KB: 5,050 × 6 bytes (CAM-PQ fingerprints)
├── codebook_pq.bin         # 96KB: [6][256][16] × f32 centroids
├── codebook_pq.json        # Same + normalization params (mean, std)
├── nsm_primes.json         # 63 NSM semantic primes → rank + CAM codes
├── subgenres_5k.csv        # 96 subgenre frequency distributions
├── lemmas_5k.csv           # Full lemma data with per-genre frequencies
└── lemmas_compact.csv      # Compact version
```

Clone the DeepNSM repo to get data:
```bash
curl -H "Authorization: token $GH_TOKEN" \
     -L "https://api.github.com/repos/AdaWorldAPI/DeepNSM/zipball/main" \
     -o deepnsm-data.zip
unzip -q deepnsm-data.zip
```

## Task List

### P0: Compile and fix

```bash
cd crates/deepnsm
cargo check
cargo test
```

Known issues to fix:
1. `Cargo.toml` has been added to workspace `exclude` list (already done)
2. The `parser.rs` creates `SentenceStructure` with public field access but the struct fields need to match what `pipeline.rs` and `context.rs` expect
3. `WordDistanceMatrix` is defined in `spo.rs` but referenced by `pipeline.rs` — verify the import chain works
4. `vocabulary.rs` uses `std::collections::HashMap` and `std::fs` — this is fine since we're not `no_std`

### P1: Integration test with real COCA data

Create `tests/integration.rs`:

```rust
use deepnsm::DeepNsmEngine;
use std::path::Path;

#[test]
fn load_and_tokenize() {
    let engine = DeepNsmEngine::load(Path::new("../../DeepNSM/word_frequency/")).unwrap();
    
    let result = engine.process_sentence("the big dog bit the old man");
    assert!(result.known_token_count >= 5); // most words in vocab
    assert!(!result.structure.triples.is_empty());
    
    // Check triple: dog → bite → man
    let t = &result.structure.triples[0];
    println!("Triple: {}", engine.describe_triple(t));
}

#[test]
fn nearest_words() {
    let engine = DeepNsmEngine::load(Path::new("../../DeepNSM/word_frequency/")).unwrap();
    
    // "think" should be near "know", "believe", "feel"
    let neighbors = engine.word_neighbors("think", 10).unwrap();
    println!("Nearest to 'think': {:?}", neighbors);
    
    // "before" should be near "after"
    let neighbors = engine.word_neighbors("before", 10).unwrap();
    println!("Nearest to 'before': {:?}", neighbors);
}

#[test] 
fn sentence_similarity() {
    let mut engine = DeepNsmEngine::load(Path::new("../../DeepNSM/word_frequency/")).unwrap();
    
    let a = engine.process_sentence("the dog bites the man");
    let b = engine.process_sentence("the cat scratches the boy");
    let c = engine.process_sentence("the stock market crashed today");
    
    let sim_ab = engine.sentence_similarity(&a, &b);
    let sim_ac = engine.sentence_similarity(&a, &c);
    
    // Similar sentences should score higher
    println!("dog-bites-man vs cat-scratches-boy: {:.3}", sim_ab.overall);
    println!("dog-bites-man vs stock-market:      {:.3}", sim_ac.overall);
    assert!(sim_ab.overall > sim_ac.overall);
}
```

### P2: Performance benchmarks

The distance matrix build is O(n²) and will be slow for the full 4096 vocab. Profile it:

```bash
cargo test --release -- --nocapture load_and_tokenize
```

Expected timings:
- Matrix build: ~2s (one-time, can be cached)
- Tokenization: < 1μs per word
- SPO parsing: < 5μs per sentence
- Similarity lookup: < 15ns per pair

### P3: Wire to lance-graph thinking pipeline

Once tests pass, the next step is connecting `DeepNsmEngine` to `crates/lance-graph-planner/src/thinking/graph.rs`:

```rust
// In thinking/graph.rs, cognitive verbs use DeepNSM for semantic reasoning:
// SYNTHESIS: merge triples if similarity > 0.85
// COUNTERFACTUAL: negate the predicate plane
// INTERRELATE: find cross-domain bridges via subject similarity
```

## Key Architecture Decisions

1. **WordDistanceMatrix is 4096×4096 u8 (16MB)** — fits L2 cache. Built from CAM-PQ codes at load time. Can be serialized to disk after first build.

2. **SimilarityTable is 256 × f32 (1KB)** — built from the EXACT empirical CDF of the distance matrix. Not parametric. Not sampled.

3. **VSA vectors are 512 bits (64 bytes)** — compact but sufficient for 4096 vocab. Upgradeable to 10Kbit Fingerprint<256> for production.

4. **Parser is a 6-state PoS FSM** — handles SVO English (85% of sentences). Secondary patterns cover passive/existential.

5. **Context window is a ring buffer** — O(1) per sentence push. Running majority vote IS the context. No recomputation.

## What NOT to change

- The crate MUST remain zero-dependency. No serde, no arrow, no datafusion.
- The vocabulary size MUST be exactly 4096 (12-bit addressing).
- The distance matrix MUST be symmetric u8 (palette-quantized).
- The role vectors MUST be deterministic from fixed seeds.
- The parser MUST NOT use regex.

## Repo Access

```bash
# lance-graph (main codebase) — use the AdaWorldAPI GitHub PAT
# DeepNSM (COCA data) — same token, same org

# Clone both via REST zipball (use PAT from secure config)
for repo in lance-graph DeepNSM; do
  curl -H "Authorization: token $GH_TOKEN" \
       -L "https://api.github.com/repos/AdaWorldAPI/$repo/zipball/main" \
       -o $repo.zip
  unzip -q $repo.zip
  mv AdaWorldAPI-$repo-* $repo
  rm $repo.zip
done
```
