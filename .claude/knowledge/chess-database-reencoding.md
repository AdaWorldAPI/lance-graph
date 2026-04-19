# Chess Database Re-Encoding — Substrate Play (#5)

> Complements `wikidata-spo-nars-at-scale.md` (knowledge-graph scale demo)
> and `chess-nars-vertical-slice.md` in q2 (precision demo). This doc is
> the **scale demo for chess**: the entire historical record of chess loaded
> as a first-class graph, queryable in Cypher at semantic similarity speed.

---

## The Corpus

| Source | Size | Games |
|--------|------|-------|
| Lichess Open Database (PGN monthly dumps) | ~7 TB compressed | ~5 billion |
| ChessBase Mega Database | ~10 GB | ~10 million master-level |
| Syzygy 7-piece endgame tablebase | ~18 TB | All ≤7-piece positions (exact) |
| Stockfish polyglot opening books | ~300 MB | Canonical opening theory |
| TCEC + ICGA engine-match archives | ~2 GB | ~1 million top-engine games |

## Compression Target

After transposition deduplication:

```
5B games × ~40 plies                  = 200B position-occurrences
unique canonical FENs (merged)         = ~10B
12 bytes per triple (CAM-PQ + NARS)    = 120 GB position graph
+ game metadata (40 B × 5B games)     = 200 GB
+ Syzygy endgame palette (encoded)    =   5 GB
──────────────────────────────────────────────
Total: ~325 GB — fits on a modern SSD
```

ChessBase with indexes: ~600 GB for 10M games.
We carry **500× more games in half the space**, with semantic similarity
as an O(1) palette lookup rather than BFS.

## Encoding per Position

```
FEN → Fingerprint<256>                   (16,384 bits, content-addressable)
Adjacent positions → CausalEdge64 per move  (packed SPO + NARS truth)
NARS frequency = historical win-rate at this position across corpus
NARS confidence = sample size (games through this position)
Palette entry = common position archetype (k=256 via CAM-PQ clustering)
```

## Queries That Become Possible (Impossible Elsewhere)

### Semantic position similarity at scale

```cypher
MATCH (p:Position)
WHERE SEMANTIC_DISTANCE(p, $ruy_lopez_main_line) < 0.3
  AND p.game.white_elo > 2500
  AND p.outcome = 'white_wins'
RETURN p.game.id, p.move_number
LIMIT 20
```

ChessBase: can't query by semantic distance. Lichess: can't.
Our answer in < 1 second over 5 billion games because palette distance
is O(1).

### Multi-hop NARS confidence paths

```cypher
MATCH path = (start:Position {pgn: $classical_french})-[*1..15]-(endgame)
WHERE NARS_CONFIDENCE(path) > 0.8
  AND endgame.eval > 2.0
RETURN path, endgame.game_count
```

NARS revision rule folds confidence at every hop. No chess database has
native truth-propagating multi-hop.

### Find human-intuition moves (engine-dispreferred but winning)

```cypher
MATCH (p:Position)-[:PLAYED]->(m:Move)
WHERE p.game.white = 'Carlsen, Magnus'
  AND m.engine_approval < 0.3
  AND p.game.outcome = 'white_wins'
RETURN p, m, p.game.id, p.game.black
ORDER BY m.eval_loss DESC
```

"Positions where Magnus played a computer-dispreferred move and won."
Queryable across his entire career in seconds.

## How This Fits the Five-Program Strategy

```
Week 0-2:  #5 ingest (lichess-db monthly dump → BindSpace)  ← THIS DOC
              │ produces the chess-scale graph
              ▼
Week 2-4:  #4 opening harvest + deep search USING the #5 corpus
           #2 style benchmark AGAINST the #5 corpus
              │ styles ranked on 5B games, not 500
              ▼
Week 4-12: #3 learning benchmark (NARS revision ON vs OFF)
              │ richer signal from real-game ground truth
              ▼
Week 12+:  #1 longitudinal study (27-week cognitive evolution)
              │ the bot's play cross-referenced against 5B-game baselines
              ▼
           the culminating demo
```

**Key insight:** #5 is the substrate. Every subsequent experiment is
measurably more credible with 5 billion games behind it. The difference
between "our bot's style is Analytical" and "our bot's style is
Analytical, and here's how its openings compare against the 2.3 million
master games that share this pattern."

## Why It's Uniquely Ours

Everyone has the PGN files. What they don't have:

- **Palette-compressed FENs** — 16K-bit `Fingerprint<256>`, Hamming
  similarity in 2 cache lines. No existing chess DB does this.
- **NARS truth on every edge** — win-rate as frequency, sample-size as
  confidence, automatic propagation across multi-hop paths.
- **Cypher over the whole lot** — ChessBase uses proprietary query; Lichess
  uses Elasticsearch-style text search. Neither speaks Cypher/GQL/SPARQL.
- **4-pillar contract** — every query returns a `WorldModelDto` with
  proprioception classification of the retrieval state. The query engine
  knows what it's "feeling" as it searches.

ChessBase wins on UI. Lichess wins on recency. **Neither can query by
semantic similarity**, because neither has the palette. That's the gap.

## Implementation

### What already exists

| Crate | Ready? | Role |
|-------|--------|------|
| `arigraph::sensorium` | Yes | Observation → triplets |
| `arigraph::triplet_graph` | Yes | SPO store with NARS |
| `arigraph::episodic` | Yes | Hamming retrieval |
| `lance-graph` Cypher parser | Yes | 44 tests |
| `causal-edge` | Yes | CausalEdge64 packing |
| `bgz17` PaletteSemiring | Yes | O(1) distance tables |
| `cognitive-shader-driver` | Yes | BindSpace + dispatch |
| `lance-graph-osint` | Yes | Pipeline template |

### What needs building

| Component | LOC estimate | Time |
|-----------|-------------|------|
| PGN streaming parser → FEN → triplets | ~400 | 2-3 days |
| FEN → Fingerprint<256> content encoder | ~100 | 1 day |
| Position palette bake (k=256 from 10B FENs) | ~200 | 1 day + compute |
| Syzygy tablebase encoder | ~300 | 2 days |
| Cypher UDF: `SEMANTIC_DISTANCE()` | ~100 | 1 day |
| Ingest orchestrator (streaming, resume-safe) | ~300 | 2 days |
| **Total** | **~1400** | **~10 days** |

### Ingest pipeline

```
[lichess-db .pgn.zst files, monthly dumps]
    │ streaming decompression
    ▼
[pgn-parse crate (existing)]
    │ game at a time
    ▼
[FEN extraction per ply]
    │ canonical FEN (strip move counters)
    ▼
[Fingerprint<256> from FEN]
    │ piece placement → bit pattern
    │ similar positions → similar fingerprints (Hamming)
    ▼
[CausalEdge64 per move]
    │ from-FEN → to-FEN, NARS f/c from game outcome
    ▼
[AriGraph triplet_graph.insert()]
    │ with game metadata (players, ELO, date, result)
    ▼
[BindSpace row per unique position]
    │ content_fp = FEN fingerprint
    │ edge = canonical strongest move
    │ meta = compressed game count + avg eval
    ▼
[palette bake: cluster top 256 positions from 10B FENs]
    │ one-time: CAM-PQ k-means over Fingerprint<256>
    ▼
[PaletteSemiring: 256×256 distance + compose tables]
    │ distance(pos_A, pos_B) = O(1)
    │ compose(pos_A, pos_B) = path composition
    ▼
[ready for Cypher queries]
```

## Strategic Value

Per the positioning doc (`positioning-quarto-4d.md` in q2):

> "Neo4j Browser alternative — Cypher-compatible, faster, one binary."

The killer demo for chess is:

1. Load 5 billion games on a workstation SSD.
2. Run the three Cypher queries above in the cockpit.
3. Note that ChessBase literally can't do query #1 (no semantic distance),
   and Neo4j can't fit the data (325 GB graph, no JVM can handle it).

That's the moment the positioning stops being marketing.

The Wikidata scale demo (14.4 GB) proves the encoding. The chess scale
demo (325 GB) proves the query engine at 10× more data. Together they
cover both "knowledge graph" (Wikidata) and "domain-specific analytics"
(chess) positioning lanes.

## Risks

- **Ingest wall time.** 5B games × ~40 plies = 200B insert ops. At
  1M inserts/sec = 200K seconds ≈ 2.3 days. Acceptable for one-time
  ingest. Parallelise across monthly dumps.
- **FEN fingerprint similarity preservation.** The bit-placement encoding
  must ensure that positions with similar piece configurations have low
  Hamming distance. Needs a quick calibration probe: take 1000 annotated
  "similar position" pairs, measure Hamming correlation.
- **Palette quality at k=256.** 10B unique positions is a LOT of diversity.
  k=256 may undertile. Could go to k=1024 (still O(1) with a 1M-entry
  distance table) or hierarchical palette.
- **Storage.** 325 GB is large for RAM, comfortable for SSD. Queries that
  need full-scan (no palette pre-filter) may take seconds, not milliseconds.
  The palette pre-filter is what makes it fast; ensure the Cypher UDF
  uses it.
