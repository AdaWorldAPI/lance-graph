# WIKIDATA HHTL TILES
# S3-Backed Entity Search via bgz17 Compressor/De-esser Encoding

## The Encoding: Compressor + De-esser

bgz17's Fibonacci/Euler gamma rotation is an audio compressor + de-esser for data.

### Compressor: i8 Dynamic Range Normalization

```
Problem: Wikidata entities have insane dynamic range.
  Q30 (United States):     500 properties, millions of references
  Q98616473 (random fish): 8 properties, 2 references
  Raw values span 6 orders of magnitude → distance is meaningless

Solution: i8[10,000] encoding.
  i8 = [-128, +127]. 256 levels. No NaN. No Inf. No subnormals.
  
  Loud parts (common properties like instance_of:human) → attenuated.
  Quiet parts (rare properties like IUCN_conservation_status) → amplified.
  Character preserved. Dynamic range compressed.
  
  Both USA and random fish live in the same [-128, +127] range.
  L1 distance between them is now MEANINGFUL.
  Rare properties of fish matter as much as common properties of USA.
```

### De-esser: Fibonacci Rotation Prevents SPO Bleed

```
Problem: When you bundle S+P+O into one vector, the planes BLEED.
  Regular PQ: S in dims [0:3333], P in [3333:6666], O in [6666:10000]
  At boundaries 3333 and 6666: S leaks into P, P leaks into O
  Like audio sibilance: harsh peaks at specific frequencies
  → false similarities, can't unbind cleanly

Solution: Fibonacci/Euler gamma rotation.
  S at Fibonacci positions shifted by φ^0 = 1.000
  P at Fibonacci positions shifted by φ^1 = 1.618
  O at Fibonacci positions shifted by φ^2 = 2.618

  φ is irrational. No two planes EVER share a harmonic.
  No beating. No aliasing. No moiré. No sibilance.
  Like Fujifilm X-Trans vs Bayer sensor.
  Like a de-esser removing the "ssss" at 4-8 kHz.

  The shape IS weird. A Fibonacci spiral, not a grid.
  But weird = no NaN, no artifacts, exact reconstruction.
```

### Zeckendorf = Brick Wall Limiter

```
Every positive integer has exactly ONE Fibonacci decomposition.
No ambiguity. No rounding. No "which decomposition did we use?"
Like a brick wall limiter: absolute ceiling, nothing above.
Digital clipping impossible. Signal integrity guaranteed.
No matter how complex the entity.
```

## The Tile Architecture

```
Tile: 10,001 vectors × i8[10,000]
      = 10,000 entities + 1 orthogonal reference (90° rotation)
      = 100.01 MB per tile

Wikidata: 112M entities / 10K per tile = 11,200 tiles
S3 storage: 11,200 × 100 MB = ~1.12 TB
(with bgz17 palette compression: ~200-400 GB)
```

## The Cascade INSIDE the Tile

NOT 100M ops. 7M ops. The HEEL filters 93% first.

```
Query vector: i8[10,000]
Tile: 10,000 entities × i8[10,000]

HEEL pass (first 16 dimensions):
  query[0..16] vs entity[0..16] for all 10,000 entities
  16 bytes × 10,000 = 160 KB (fits L2)
  L1 > threshold → REJECT
  93% rejected. 700 survive.
  Cost: ~5μs

BRANCH pass (dims 16..128):
  query[16..128] vs 700 survivors
  112 bytes × 700 = 78 KB (fits L1)
  Tighter threshold. 50% of survivors gone.
  350 remain.
  Cost: ~3μs

FULL pass (all 10,000 dims):
  query[0..10000] vs 350 survivors
  10 KB × 350 = 3.5 MB (fits L3)
  Exact L1 distance. Ranked.
  Cost: ~30μs

TOTAL: ~38μs per tile (vs 500μs brute force)
       7M ops (vs 100M brute force)
       350 scored entities (vs 10,000 brute force)
```

## Two-Cycle Resonance with 90° Rotation

### Cycle 1: Horizontal — "Which entities match?"

```
query_vec L1 against all 10,000 entities (via cascade)
→ 350 scored → 27 resonate (top 0.27%)
27 entities ARE the answer to "what matches the query"
```

### Cycle 2: Vertical — "What does the match MEAN?"

```
The 10,001st vector is orthogonal to all entity vectors.
It's the predicate axis. Separates WHAT from HOW.

bundle(27 resonators) → meta_vec (majority vote)
rotate(meta_vec, orthogonal_ref) → gestalt_vec

gestalt_vec is NOT the query. It's the PATTERN of what resonated.
"what kind of entity clusters around this query?"
If query = "scientists at CERN":
  meta_vec encodes: [human, physics, European, institution]
  gestalt_vec (rotated): [research, collaboration, experiment]
  
gestalt_vec L1 against prefetched entities → next-hop filter
Finds entities that RELATE to the pattern, not repeat it.
```

## 32 MB Prefetch Pipeline

```
27 resonators × ~25 outbound edges = 675 cross-tile targets
deduplicate → ~200 unique target entities

Prefetch 25× lookahead:
  200 entities × 25 neighbors each = 5,000 candidates
  5,000 × palette (3 bytes) = 15 KB coarse pre-filter
  palette filter → ~1,000 worth full hydration
  1,000 × 10 KB = 10 MB entity vectors

Plus resonators' own tile fragments:
  ~3 tiles × partial hydration = ~20 MB

Total prefetch: ~32 MB
  = 0.27% of a full tile (10K entities)
  = exactly the resonating fraction
  = lands in L3 while cycle 1 is returning results
  = available for cycle 2 with zero wait
```

## Memory Layout

```
HOT (always in RAM, ~2.2 GB):
  CAM index:          672 MB    all 112M entities, 6 bytes each
  Popular tiles:      1 GB      top 100K entities, pre-hydrated
  DeepNSM runtime:    9 MB      4K vocab + distance matrix + similarity
  Tile cache (LRU):   500 MB    ~5 recently used tiles

WARM (S3, prefetched on demand):
  Entity tiles:       ~1.12 TB  11,200 tiles × 100 MB
  Cross-tile edges:   ~1 GB     inter-tile adjacency
  
QUERY COST:
  First query:  ~50ms   (S3 fetch dominates)
  Cached:       ~70μs   (cascade + resonance + ranking)
  Popular:      ~40μs   (pre-hydrated, no S3)
```

## Semiring on HHTL Vectors

The BlasGraph semiring operates directly on i8[10,000]:

```
multiply(entity, truth_value):
  entity[i] * confidence → weighted entity
  i8 × f16 → i8 (clamp, no NaN)
  "propagate belief along an edge"

add(path_a, path_b):
  majority_vote(a[i], b[i]) → merged entity
  two paths arriving at same node → bundle evidence
  "merge two reasoning chains"

zero: i8[10,000] all zeros (no evidence)
one:  the entity itself (full evidence)

These ARE the spo_bundle operations from ndarray:
  cyclic_shift (Fibonacci rotation) = SPO plane separation
  majority_vote = evidence bundling
  L1 distance = similarity measurement
```

## Elevation Hierarchy (same entity, increasing precision)

```
Level       Size/Entity   Total (112M)   Cache     Operation
─────       ───────────   ────────────   ─────     ─────────
CAM-6       6 bytes       672 MB         RAM       stroke cascade
Palette-3   3 bytes       336 MB         RAM/L3    palette lookup
Base17      34 bytes      3.8 GB         L3/S3     L1 on i16[17]
HHTL raw    10 KB         1.12 TB        S3        L1 on i8[10,000]

Same SimilarityTable calibrates ALL levels.
Elevation operator: try CAM → if insufficient → palette → Base17 → raw.
Each level subsumes the previous.
```

## Connection to ORCHESTRATION_IS_GRAPH

```
Layer 1 — Subject:    ThinkingStyle (36 nodes, local, LazyLock)
Layer 2 — Predicate:  CognitiveVerb (13 verbs, local, compiled)
Layer 3 — Object:     Wikidata entity (112M, S3-backed, HHTL encoded)
Layer 4 — Adjacent:   Entity property bag (decoded from HHTL vector)
                      The adjacency IS the gestalt.
                      Process entities have process-shaped adjacency.
                      Person entities have person-shaped adjacency.
                      No type field needed. The shape IS the type.

The bgz17 compressor normalizes all entities to the same dynamic range.
The Fibonacci de-esser keeps SPO planes separate.
The Zeckendorf limiter guarantees unique encoding.
The cascade filter does 7M ops instead of 100M.
The 90° rotation separates "what matches" from "what the match means."
The 32MB prefetch = exactly the resonating fraction.

Everything is one graph traversal.
```

## S3 Layout

```
s3://wikidata-hhtl/
├── tiles/
│   ├── tile_0000.bin       ← 10,001 × i8[10,000] = 100.01 MB
│   ├── tile_0001.bin
│   ├── ...
│   └── tile_1119.bin       ← 11,200 tiles total
│
├── index/
│   ├── entity_to_tile.bin  ← Q-number → tile_id + offset (448 MB)
│   ├── adjacency_cross.bin ← cross-tile edge list (~1 GB)
│   └── popular_100k.bin    ← pre-hydrated hot entities (1 GB)
│
├── cam/
│   └── entity_cam.bin      ← 112M × 6 bytes = 672 MB (RAM resident)
│
└── palette/
    └── entity_palette.bin  ← 112M × 3 bytes = 336 MB (optional RAM)
```

## Why No NaN

```
1. i8 is integer. Integer types have no NaN representation.
   Every bit pattern in i8 is a valid value in [-128, 127].

2. Fibonacci decomposition is unique. No ambiguity, no "undefined."
   Zeckendorf's theorem: every positive integer has exactly one
   representation as a sum of non-consecutive Fibonacci numbers.

3. Euler gamma rotation uses modular arithmetic, not floating point.
   cyclic_shift(bits, golden_shift(D)) is integer shift + OR.
   No division. No sqrt. No floating-point operation that could produce NaN.

4. Majority vote on i8 values: sum → sign → i8.
   Sum of i8s fits in i32. Sign is always defined. Cast to i8 clamps.
   No edge case produces NaN.

5. L1 distance on i8: |a - b| is always defined, always non-negative.
   No division by zero. No negative under sqrt. No undefined result.

The entire pipeline from raw Wikidata entity to L1 distance score
uses exactly ZERO floating-point operations that could produce NaN.
Float only enters at SimilarityTable lookup: u32 → f16 → f32.
And f16 values in the table are precomputed and validated (no NaN stored).
```
