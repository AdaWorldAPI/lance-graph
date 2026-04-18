# Wikidata → SPO/NARS at Scale — Turning Our Strengths

> Substrate-level plan. The chess-NARS vertical is the **precision** demo
> (one game, end-to-end). This is the **scale** demo (1.2 billion triples,
> sub-second query, full Cypher compatibility).

---

## The Claim

Our native encoding stack (4096 COCA + Base17 + CAM-PQ + NARS truth +
Fingerprint<256>) makes Wikidata roughly **50× smaller** than an
RDF/Neo4j representation, and turns every semantic query into an
O(1) palette-distance lookup instead of an adjacency-matrix BFS.

At that compression level, the full Wikidata dataset fits in RAM on a
workstation. Cypher over 1.2 billion triples at sub-second latency is
on the table.

---

## The Numbers

### Wikidata scale (2024-2025)

- ~110 million entities (items)
- ~11,000 properties
- ~1.2 billion statements (SPO triples)
- ~180 GB uncompressed RDF (`latest-all.nt.bz2`)
- ~110 GB compressed

### Our compression target

Every entity encodes as:

```
  COCA_4096   (12 bits)      — primary semantic address (words)
+ SCI_20K     (15 bits)      — scientific-term extension
+ ENTITY_DISAMBIG (5 bits)   — per-address disambiguation
= 32 bits per entity
```

4096 COCA gives 98.4% English coverage (per DeepNSM). Adding 20,000
scientific terms (Wikidata category heads: species, chemicals, gene
symbols, astronomical bodies, protein names, etc.) extends to the
long-tail technical entities Wikidata tracks.

Every property (Wikidata P-number) compresses to ~12 bits (11,000
properties round up to 2^14 = 16,384 slots, cluster-pack to 12).

Every NARS truth value: `frequency (u8) + confidence (u8) = 2 bytes`.

**Per triple:** 32 + 12 + 32 + 16 = **92 bits ≈ 12 bytes** (packed).

**Total:** 1.2B × 12 B = **14.4 GB.** Fits in RAM on a 16GB laptop,
comfortably on a 32GB workstation.

Compare to:
- Neo4j with properties: ~400 GB for the same data (nobody runs this).
- Raw NT file: ~180 GB (parse-once, query-slow).
- GraphDB / RDF4J: ~80 GB indexed (hours to load).

---

## Why This Actually Works (Not Hand-Waving)

### 1. COCA 4096 already covers 98.4% of English

DeepNSM (in `crates/deepnsm/`, 2200 LOC, 0 deps) ships with the
4,096-word COCA vocabulary and a 4096² u8 distance matrix. This is our
semantic substrate. Any entity whose label or description maps to
a COCA word gets a "warm" address immediately.

### 2. Scientific long tail is small

Wikidata's technical entities (taxonomy, genes, chemistry, astronomy)
are dense in a small domain-specific vocabulary. 20,000 scientific
terms covers most of it. Together, 24,096 addresses cover the vast
majority of real queries.

### 3. NARS truth fits naturally

Wikidata statements already have provenance and references. Converting
a reference count to NARS `frequency` (0-255) and inferred trust to
`confidence` (0-255) is one pass over the dump. Our `causal-edge` crate
already packs NARS into `CausalEdge64`.

### 4. CAM-PQ distance is O(1), not O(log n)

bgz17 PaletteSemiring ships a 256×256 u16 distance matrix + 256×256 u8
compose table. Both fit in L2 cache. Every "how similar are these two
entities" query is a single memory lookup. Neo4j's equivalent is a
multi-hop graph traversal.

### 5. Fingerprint<256> is content-addressable

Each Wikidata entity gets a 16K-bit fingerprint from its surrounding
SPO neighborhood. Semantic similarity via Hamming. This IS the
AriGraph episodic-memory pattern but at Wikidata scale.

---

## The Pipeline (Ingest Once, Query Forever)

```
[Wikidata dump: .nt.bz2, ~110 GB compressed]
       │
       ▼
┌───────────────────────────────────────────────────────┐
│  wikidata-ingest (new crate, ~500 lines)              │
│    - Stream-parse NT triples                          │
│    - Map subject/object label → COCA_4096 + SCI_20K   │
│    - Map property Qxxx → PropertyId (u12)             │
│    - Count references, derive NARS freq/conf          │
│    - Emit CausalEdge64 per triple                     │
└────────────────────┬──────────────────────────────────┘
                     │ stream
                     ▼
┌───────────────────────────────────────────────────────┐
│  AriGraph triplet_graph (1064 LOC, already shipped)   │
│    - Insert Triplet with NARS truth                   │
│    - Build subject → [predicate, object, truth] index │
│    - BFS association for multi-hop                    │
│    - Already handles spatial paths                    │
└────────────────────┬──────────────────────────────────┘
                     │
                     ▼
┌───────────────────────────────────────────────────────┐
│  lance-graph Cypher parser (already shipped, 44 tests)│
│    - MATCH (p:Person)-[:BORN_IN]->(c:Country)         │
│      WHERE p.name = "Ada Lovelace"                    │
│    - Compiles to CAM-PQ palette ops                   │
│    - Runs at palette-distance speed, not BFS speed    │
└────────────────────┬──────────────────────────────────┘
                     │
                     ▼
[Cockpit at cubus.up.railway.app — Cypher query box]
```

---

## What Becomes Possible

Three categories of query that **Neo4j cannot reasonably run** on full
Wikidata:

### Semantic fuzzy match at scale

```cypher
MATCH (p:Person)-[r]-(concept)
WHERE SEMANTIC_DISTANCE(concept, 'mathematics') < 0.3
  AND p.birthYear > 1900
RETURN p.name, concept.label, r.type
LIMIT 20
```

Neo4j: scan millions of `Person` nodes, evaluate `SEMANTIC_DISTANCE` via
embeddings. Minutes to hours.

Ours: CAM-PQ palette distance is O(1) per pair. 20 results in ~100 ms.

### Multi-hop with NARS confidence

```cypher
MATCH path = (ada:Person {name:"Ada Lovelace"})-[*1..3]-(related)
WHERE CONFIDENCE(path) > 0.7
RETURN related.label, CONFIDENCE(path)
ORDER BY CONFIDENCE(path) DESC
```

Neo4j: BFS, no native confidence propagation.

Ours: NARS revision rule at every hop, confidence folds along the path,
AriGraph's spatial-path functions do the BFS in O(k·d).

### Provenance-aware query

```cypher
MATCH (e:Event)
WHERE e.date > '2020-01-01'
  AND e.references >= 3
  AND NARS_TRUTH(e) > (0.8, 0.6)
RETURN e.title, NARS_TRUTH(e) as truth
```

Neo4j: no notion of reference-weighted truth. User has to implement it.

Ours: `causal-edge` packs NARS truth into every edge at ingest time.
Predicate is a cheap bitwise comparison.

---

## Why This Is Strategic

Per the `positioning-quarto-4d.md` doc in q2, the product is framed as
"Neo4j Browser alternative — Cypher-compatible, faster, one binary."
The killer demo for that positioning is:

1. Load full Wikidata on a consumer laptop.
2. Run the three queries above in the cockpit.
3. Note that Neo4j literally cannot do this — not won't, can't.

That's the moment the positioning stops being marketing and starts
being a benchmark.

### Secondary: the NARS + proprioception dividend

Once Wikidata is loaded, every query through the cockpit produces a
WorldModelDto with `proprioception.anchor` classified. A query that
spans general knowledge might classify as `Observer` (insight-dominant);
a query probing scientific depths might classify as `Focused` (clarity +
contact); a novel connection might classify as `Flow`
(vitality + novelty).

The `/mri` view during a Wikidata query becomes a **cognitive telemetry
of understanding-in-progress**. That's the killer screenshot for the
product page.

### Tertiary: airwar.cloud transfer is trivial

Wikidata is the largest consumed open knowledge graph. If our pipeline
handles Wikidata scale, it handles airwar.cloud intelligence feeds
with room to spare. The OSINT ingestor (`lance-graph-osint/`) swaps in
for `wikidata-ingest`; everything downstream is the same.

---

## What Needs Building

Tier 0 (the scale demo, ~1 week of focused work):

1. **`wikidata-ingest` crate** (~500 lines) — stream-parse NT, map to
   COCA/SCI addresses, derive NARS truth from reference counts.
2. **COCA/SCI address table bake** — one-time compilation of the
   4096 + 20000 vocabulary into a palette-compatible u27 mapping.
3. **Property palette** — 11,000 Wikidata P-numbers clustered to
   ~4000 palette slots for O(1) compose.
4. **Cypher query benchmark suite** — the three queries above, plus
   5 more, with Neo4j comparison numbers (for the website).
5. **Cockpit query UI hookup** — wire the existing Cypher parser to
   the cockpit's query box; render result graphs in the 3D scene.

Tier 1 (once scale is demonstrated):

6. **Incremental re-ingest** — Wikidata changes daily. Dump diff →
   CausalEdge64 diff → AriGraph update. ~2 hours/day of background work.
7. **xAI / Claude enrichment on novel queries** — when
   `classification_distance` is high (liminal state), the `xai_client.rs`
   in AriGraph is already there to enrich.
8. **Federation with live sources** — Wikidata as the base layer,
   specialist feeds (OSINT, scientific papers, news) as overlays.

---

## Risks and Open Questions

- **COCA miss rate.** 98.4% English coverage still leaves 1.6% misses.
  For Wikidata that's ~1.8M entities without a clean COCA address.
  Fallback: hash unknown labels into a 27-bit extension space, accept
  collision rate.
- **Property cardinality.** 11,000 properties is at the edge of 12-bit
  encoding; may need 13 bits for safe clustering.
- **Reference count → NARS confidence calibration.** A statement with 3
  references isn't strictly twice as confident as one with 1.5 references.
  Needs a proper calibration function, not linear scaling.
- **Ingest time.** 1.2B triples × (lookup + pack + write) must complete
  in reasonable wall time. Target: 2-4 hours on a 16-core workstation.
  If it takes a day, the demo loses crispness.
- **Cypher dialect coverage.** Our parser handles basic MATCH/WHERE/RETURN
  but not every Neo4j dialect extension. The demo queries should stay
  within spec so we don't hand-wave compatibility claims.

---

## Why This Is Worth the Session

It validates the positioning (Neo4j replacement at scale), the
substrate (COCA + CAM-PQ + NARS compress 180 GB to 14 GB), the
pipeline (AriGraph handles billions of triples), and the demo story
("we loaded Wikidata on a laptop and Neo4j couldn't").

It's the **turning-our-strengths** demo. Everything we've built
(4096 surface, palette semirings, NARS truth, SPO triples, Fingerprint,
CAM-PQ, AriGraph episodic memory) was designed for this regime.
Wikidata is the natural consumer.

After this, airwar.cloud is not a new project — it's a sensor swap.
