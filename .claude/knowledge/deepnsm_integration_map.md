# DeepNSM ↔ bgz17 ↔ cam_pq Integration Map

## The Pipeline

```
Text → DeepNSM tokenizer → 12-bit tokens → SPO parser → triples
  → bgz17 distance_matrix → raw distance → SimilarityTable → f32 [0,1]
  → cam_pq CamCodebook → 6-byte fingerprint (OOV resolution)
  → thinking/graph.rs → cognitive verbs reason about meaning
```

## What Connects Where

### DeepNSM → bgz17
- `DistanceMatrix` (4096×4096 u8): built from COCA 96D vectors
- `SimilarityTable` (256 × f16): calibrated from exact matrix
- `Base17` encoding: 96D → Fibonacci palette → L1 distance
- `SpoBase17`: SPO triple → 3 planes → per-role distance
- `Precision` bands: Foveal/Near/Good/Miss → decision thresholds

### DeepNSM → cam_pq
- `CamCodebook` (96KB): trained from COCA subgenre vectors
- `CamFingerprint` (6 bytes): word → compressed representation
- `PackedDatabase` + stroke cascade: batch OOV resolution
- `cam_distance()` UDF: DataFusion query interface
- Subspace assignment: HEEL=register, BRANCH=domain (semantic, not positional)

### DeepNSM → ThinkingGraph
- `NsmEncoder` attaches to `ThinkingGraph` via `.with_semantics()`
- Cognitive verbs use `triple_similarity()` for semantic decisions
- `ContextWindow` provides streaming disambiguation
- PoS tags drive `MODULATE` verb (content → thinking style)
- NSM prime decomposition feeds `DEEPEN` verb

## Key Invariants

1. Distance matrix is symmetric: `matrix[a][b] == matrix[b][a]`
2. Self-distance is zero: `matrix[a][a] == 0`
3. SimilarityTable is monotonic: lower distance → higher similarity
4. SPO binding is recoverable: `unbind(bundle(bind(w, role)), role) ≈ w`
5. Context window is incremental: O(1) push, not O(window_size)
6. All operations are deterministic: same input → same output, always

## Build Dependencies

```
word_frequency/word_rank_lookup.csv → Vocabulary (4,096 entries)
word_frequency/word_forms.csv → Forms hash (11,460 inflections)
word_frequency/subgenres_5k.csv → 96D vectors → distance_matrix
word_frequency/nsm_primes.json → 63 anchor indices
word_frequency/codebook_pq.bin → CamCodebook (OOV path)
word_frequency/cam_codes.bin → CamFingerprints (OOV path)
```

## Session Prompts (all synced across repos)

| File | Repo | Lines |
|------|------|-------|
| `.claude/prompts/session_deepnsm_cam.md` | DeepNSM, ndarray, lance-graph | ~800 |
| `.claude/prompts/session_bgz17_similarity.md` | ndarray, lance-graph | ~300 |
| `.claude/prompts/CAM_PQ_SPEC.md` | lance-graph | — |
| `docs/deepnsm_cam_architecture.md` | lance-graph | ~400 |
| `.claude/DEEPNSM_CAM_REFERENCE.md` | lance-graph | ~200 |
