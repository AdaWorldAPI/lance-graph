# BindSpace Migration Gap Analysis

> Compared AdaWorldAPI/ladybug-rs container model against lance-graph.
> Session: 2026-04-18. Context: 1M tokens allows full cross-repo view.

## The Core Issue

The ladybug-rs BindSpace model is built on a **1KB aligned Container** that
unifies metadata + content + edges + NARS + qualia into a single record.
Lance-graph has the computational infrastructure (semirings, cascade, CLAM,
CAM-PQ) but lacks the **record structure** that ties it all together.

## What Migrated Successfully

| Component | ladybug-rs | lance-graph | Status |
|---|---|---|---|
| Fingerprint | 16K fixed | Fingerprint\<N\> const-generic (ndarray) | ✅ Better |
| XOR bind/unbind | VsaOps trait | Fingerprint XOR (ndarray) + holograph | ✅ Equivalent |
| Bundle (majority) | VsaOps::bundle | holograph bitpack | ✅ Equivalent |
| NARS truth | NarsTruthValue | NarsTruth (contract) | ✅ Migrated |
| Thinking styles | CognitiveStyle | ThinkingStyle (contract, 36 styles) | ✅ Extended |
| Semirings | 7 ContainerSemirings | 7 HDR semirings (blasgraph) | ✅ Different types |
| Cascade search | HDR cascade | cascade.rs (ndarray) | ✅ Equivalent |
| CLAM tree | CLAM paths in SPO | ClamTree (ndarray, 46 tests) | ✅ Better |
| CAM-PQ | cam_ops (4096 ops) | cam_pq (encode/distance/cascade) | ⚠️ Subset |
| Base17 | Base17 in SPO | Base17 (bgz-tensor, bgz17) | ✅ Equivalent |
| MUL assessment | learning/feedback | MulAssessment (planner) | ⚠️ Partial |

## What Did NOT Migrate (7 Critical Gaps)

### 1. Container (1KB aligned record)
```
ladybug-rs: Container = [u64; 128] = 8,192 bits = 1 KB
  - 64-byte aligned for AVX-512
  - Self-describing (metadata in words 0-15)
  - SIMD-scannable (16 AVX-512 iterations for full Hamming)

lance-graph: NO EQUIVALENT
  - Fingerprints are separate from metadata
  - No fixed-size record format
  - Arrow RecordBatch for tabular data (flexible but not SIMD-aligned)
```

### 2. CogRecord (metadata + content)
```
ladybug-rs: CogRecord = 2 KB = Container[0] (meta) + Container[1] (content)
  - One Redis GET returns vector AND edges AND NARS AND qualia
  - Searchable by Hamming on content, filterable by metadata
  - The query surface IS the record

lance-graph: NO EQUIVALENT
  - MetadataStore (Arrow RecordBatch) separate from vectors
  - Graph structure separate from vector storage
  - Multiple lookups to assemble what CogRecord gives in one read
```

### 3. PackedDn (hierarchical addressing)
```
ladybug-rs: PackedDn = u64, 7 levels × 8 bits
  - /concepts/animals/cats = [3][0x01][0x03][0x42][0][0][0][0]
  - Lexicographic sort = tree order
  - Range scan = subtree enumeration

lance-graph: NO EQUIVALENT
  - Cypher paths are strings
  - No packed hierarchical addressing
  - ClamTree has paths but not packed into u64
```

### 4. Spine (XOR-fold lazy recompute)
```
ladybug-rs: Spine = XOR-fold of children containers
  - Lock-free: XOR is commutative + associative
  - Lazy: recompute on read, not on write
  - SpineCache: dirty bitmap tracks which spines need recompute

lance-graph: NO EQUIVALENT
  - No XOR-fold aggregation
  - No lazy recomputation model
  - TypedGraph does eager adjacency updates
```

### 5. Inline edges in metadata
```
ladybug-rs: Container 0 words 16-31 = 64 packed edges
  - verb:u8 + target_hint:u8 per edge, 4 per u64 word
  - CSR overflow in words 96-111 for 200+ edges
  - The graph IS the metadata — no separate edge table

lance-graph: SEPARATE STRUCTURES
  - SPO triple store (graph/spo/) — separate from vectors
  - BlasGraph CSR — separate adjacency matrix
  - Need multiple data structures for what Container encodes in one record
```

### 6. Scent index (petabyte-scale filtering)
```
ladybug-rs: Scent = 5 bytes (40 bits) per chunk
  - Hierarchical bucketing: 256 buckets per level
  - ChunkHeader with cognitive markers (plasticity, decision, last_access)
  - Eliminates 99.997% of corpus in ~100ns

lance-graph: NO EQUIVALENT
  - cascade.rs has 3-stroke Band filtering
  - But no hierarchical scent bucketing
  - No plasticity/attention tracking per chunk
```

### 7. Universal 64-bit address
```
ladybug-rs: 16-bit type + 48-bit fingerprint prefix = 64-bit address
  - Type namespaces: THOUGHT, CONCEPT, STYLE, EDGE_CAUSES, LAYER_SUBSTRATE...
  - All query types (SQL, Cypher, Hamming, Vector) resolve to same operation
  - Immutable after build, zero-copy mmap

lance-graph: NO EQUIVALENT
  - Different ID schemes per subsystem
  - No unified addressing across query languages
```

## Migration Priority

| Gap | Impact | Effort | Where |
|---|---|---|---|
| Container + CogRecord | Foundational | Large | lance-graph-contract |
| PackedDn | Addressing | Medium | lance-graph-contract |
| Universal address | Query unification | Medium | lance-graph-contract |
| Inline edges | Graph = metadata | Large | lance-graph-contract + core |
| Spine | Concurrency | Medium | lance-graph core |
| Scent index | Scale | Medium | ndarray or lance-graph |
| cam_ops (4096) | Vocabulary | Large | lance-graph-cognitive |

## Recommendation

Container and CogRecord should go in `lance-graph-contract` (the zero-dep
trait crate) as the canonical record type. PackedDn and universal addressing
go there too. These are not "cognitive extensions" — they're the fundamental
data unit that everything else builds on.

The 1M context window that's now available means we can do this migration
in one pass, seeing both codebases simultaneously, instead of the piecemeal
adaptation that lost these pieces originally.
