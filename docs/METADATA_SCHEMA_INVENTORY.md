# Metadata Schema Inventory: 4096 Surface, 16Kbit, Lance, Arrow

> *Every schema, every field, every byte — where data lives and how it flows.*

## 1. BindSpace (ladybug-rs) — The 4096 Surface

### 1A. Address Space Layout

```
16 prefixes × 256 slots = 4,096 surface addresses
Each address holds one BindNode with [u64; 256] = 16,384-bit fingerprint

PREFIX  RANGE       NAME            SLOT SUBDIVISION
──────────────────────────────────────────────────────
0x00    0x00-0xFF   Lance/Kuzu      Full (vector ops)
0x01    0x00-0x7F   SQL             Primary (columnar)
0x01    0x80-0xFF   CQL             Secondary (Cassandra)
0x02    0x00-0x7F   Cypher          Primary (property graph)
0x02    0x80-0xFF   GQL             Secondary (ISO 39075)
0x03    0x00-0xFF   GraphQL         Full (schema-first)
0x04    0x00-0x7F   NARS            Primary (cognitive arch)
0x04    0x80-0xFF   ACT-R           Secondary (cognitive arch)
0x05    0x00-0xFF   Causal          Full (Pearl reasoning)
0x06    0x00-0xFF   Meta            Full (meta-cognition)
0x07    0x00-0xFF   Verbs           Full (CAUSES, BECOMES, ...)
0x08    0x00-0xFF   Concepts        Full (core types)
0x09    0x00-0xFF   Qualia          Full (qualia operations)
0x0A    0x00-0xFF   Memory          Full (memory ops)
0x0B    0x00-0xFF   Learning        Full (learning ops)
0x0C    0x00-0xFF   Agents          Full (crewai agent registry) [overlay]
0x0D    0x00-0xFF   Thinking        Full (thinking style templates) [overlay]
0x0E    0x00-0xFF   Blackboard      Full (per-agent state) [overlay]
0x0F    0x00-0xFF   A2A             Full (agent-to-agent routing) [overlay]
```

### 1B. BindNode Structure

```rust
// ladybug-rs/src/storage/bind_space.rs
pub struct BindNode {
    fingerprint: [u64; FINGERPRINT_WORDS],  // 256 words = 2048 bytes = 16,384 bits
    label: Option<String>,
    qidx: u8,                               // qualia index
    access_count: u32,
    // ... zone info (surface/fluid/node)
}
```

### 1C. Addr Type

```rust
pub struct Addr(u16);  // 8-bit prefix : 8-bit slot

impl Addr {
    pub fn new(prefix: u8, slot: u8) -> Self { Addr((prefix as u16) << 8 | slot as u16) }
    pub fn prefix(&self) -> u8 { (self.0 >> 8) as u8 }
    pub fn slot(&self) -> u8 { self.0 as u8 }
}
```

### 1D. Sublanguage Resolution (Slot Subdivision)

```rust
// Slot 0x00-0x7F = primary language, 0x80-0xFF = secondary
pub const SLOT_SUBDIVISION: u8 = 0x80;

// PREFIX_SQL (0x01): slots 0x00-0x7F = SQL, 0x80-0xFF = CQL
// PREFIX_CYPHER (0x02): slots 0x00-0x7F = Cypher, 0x80-0xFF = GQL
// PREFIX_NARS (0x04): slots 0x00-0x7F = NARS, 0x80-0xFF = ACT-R
```

### 1E. τ (Tau) Address Mapping for Thinking Styles

Thinking style templates live at prefix 0x0D:

| τ Address | Style | Cluster |
|-----------|-------|---------|
| 0x0D:0x20 | Curious | Exploratory |
| 0x0D:0x25 | Philosophical | Exploratory |
| 0x0D:0x40 | Logical | Analytical |
| 0x0D:0x45 | Precise | Analytical |
| 0x0D:0x60 | Direct | Direct |
| 0x0D:0x65 | Frank | Direct |
| 0x0D:0x80 | Empathetic | Empathic |
| 0x0D:0x85 | Warm | Empathic |
| 0x0D:0xA0 | Creative | Creative |
| 0x0D:0xA5 | Playful | Creative |
| 0x0D:0xC0 | Reflective | Meta |
| 0x0D:0xC5 | Sovereign | Meta |

Each slot stores a 7-byte thermometer-coded FieldModulation fingerprint:
```
[resonance, fan_out, depth, breadth, noise, speed, exploration]
```

## 2. Lance Persistence Schemas (lance-graph)

### 2A. Upstream (Untouched)

```
graph_nodes.lance:
  node_id     Int64       (primary key)
  labels      List<Utf8>  (node labels)
  properties  Map<Utf8, Utf8>

graph_rels.lance:
  rel_id      Int64
  source_id   Int64
  target_id   Int64
  rel_type    Utf8
  properties  Map<Utf8, Utf8>
```

### 2B. Extension Tables (lance-graph/blasgraph/lance_neighborhood.rs)

```
scopes.lance:
  scope_id        Int64
  node_ids        FixedSizeBinary(80000)    -- [Int64; 10000] packed
  node_count      UInt16
  created_at      Timestamp(Nanosecond, None)

neighborhoods.lance:
  node_id         Int64
  scope_id        Int64
  scent           FixedSizeBinary(10000)    -- [u8; 10000] ZeckF64 byte 0 per neighbor
  resolution      FixedSizeBinary(10000)    -- [u8; 10000] ZeckF64 byte 1 per neighbor
  edge_count      UInt16
  updated_at      Timestamp(Nanosecond, None)
  NOTE: Lance column pruning means reading only scent never loads resolution.

cognitive_nodes.lance:
  node_id         Int64
  zeckf16_self    UInt8                     -- self-ZeckF16 (compressed scent)
  integrated_16k  FixedSizeBinary(2048)     -- 16Kbit cascade L1 fingerprint
  subject_plane   FixedSizeBinary(2048)     -- 16Kbit Subject plane
  predicate_plane FixedSizeBinary(2048)     -- 16Kbit Predicate plane
  object_plane    FixedSizeBinary(2048)     -- 16Kbit Object plane
  truth_freq      Float32                   -- NARS truth frequency
  truth_conf      Float32                   -- NARS truth confidence
  merkle_root     FixedSizeBinary(6)        -- Blake3 integrity hash (truncated)
```

**Storage per cognitive node**: 1 + 2048 + 2048 + 2048 + 2048 + 4 + 4 + 6 = **8,207 bytes**
**Storage per scope (10K nodes)**: 80,000 + 2 + 8 = **~80KB**
**Storage per neighborhood**: 10,000 + 10,000 + 2 + 8 = **~20KB**

## 3. Arrow Bridge Schemas (ndarray)

### 3A. ThreePlaneFingerprintBuffer

```rust
// ndarray/src/hpc/arrow_bridge.rs
pub struct ThreePlaneFingerprintBuffer {
    /// Subject planes: [u8; BYTES_PER_PLANE * N]
    pub subjects: Vec<u8>,
    /// Predicate planes: [u8; BYTES_PER_PLANE * N]
    pub predicates: Vec<u8>,
    /// Object planes: [u8; BYTES_PER_PLANE * N]
    pub objects: Vec<u8>,
    /// Node count
    pub count: usize,
}
// BYTES_PER_PLANE = 256 * 8 = 2048
```

### 3B. SoakingBuffer (Hot → Cold Transfer)

```rust
pub struct SoakingBuffer {
    pub fingerprints: ThreePlaneFingerprintBuffer,
    pub truth_frequencies: Vec<f32>,
    pub truth_confidences: Vec<f32>,
    pub gate_states: Vec<GateState>,
}

pub enum GateState {
    Open,
    Soaking { rounds_remaining: u16 },
    Closed,
}
```

### 3C. Arrow Schema for DataFusion Integration

```
bindspace table (ladybug-rs/query/fingerprint_table.rs):
  address      UInt16                      -- 8-bit prefix : 8-bit slot
  fingerprint  FixedSizeBinary(2048)       -- 16Kbit fingerprint
  label        Utf8 (nullable)
  qidx         UInt8                       -- qualia index
  access_count UInt32
  zone         Utf8                        -- 'surface', 'fluid', 'node'
```

## 4. MetadataStore Schema (lance-graph/graph/metadata.rs)

The "cold path skeleton" — no fingerprints, just structural metadata:

```
Node Schema:
  node_id     UInt32
  label       Utf8
  [dynamic property columns]

Edge Schema:
  source      UInt32
  target      UInt32
  edge_type   Utf8
  [dynamic property columns]
```

**This is intentionally sparse** — it's the skeleton for DataFusion columnar joins.
Fingerprints (the "flesh") live in the SPO store and cognitive_nodes.lance.

## 5. BGZ17 Container Schema (bgz17/container.rs)

The 256-word container packs multiple data channels:

```
Container = [u64; 256] = 2048 bytes

Word ranges:
  W[0..15]:     SPO fingerprint planes (3 × 5 words = 15 words, partial)
  W[16..31]:    Inline edge storage (up to 16 edges as packed u64)
  W[32..47]:    Base17 encoded planes (3 × i16[17] = 102 bytes ≈ 13 words)
  W[48..63]:    Palette indices + distance cache
  W[64..127]:   Reserved (future: truth values, temporal, provenance)
  W[128..255]:  Extended fingerprint (if full 16Kbit needed)

Pack operations:
  container.pack_spo(subject, predicate, object)
  container.pack_base17(spo_base17)
  container.pack_inline_edge(source_palette, verb_palette, target_palette)
  container.extract_scent() -> u8  (ZeckF64 byte 0 from inline edges)
```

## 6. DataFusion UDF Schema (lance-graph)

### 6A. Existing UDFs (lance-graph/datafusion_planner/udf.rs)

| UDF | Input | Output | Purpose |
|-----|-------|--------|---------|
| `vector_distance(a, b, metric)` | Binary, Binary, Utf8 | Float64 | L2/cosine distance |
| `vector_similarity(a, b)` | Binary, Binary | Float64 | Cosine similarity |
| `hamming_distance(a, b)` | Binary, Binary | Int64 | Hamming distance on fingerprints |

### 6B. CAM-PQ UDFs (lance-graph, post-rebase)

| UDF | Input | Output | Purpose |
|-----|-------|--------|---------|
| `cam_pq_distance(query, cam, codebook)` | Binary, Binary, Binary | Float32 | ADC distance |
| `cam_pq_encode(vector, codebook)` | Binary, Binary | Binary(6) | Encode to 6-byte CAM |
| `cam_pq_cascade(query, cam_col, threshold)` | Binary, Binary, Float32 | Boolean | 3-stroke filter |

### 6C. Cognitive UDFs (ladybug-rs/query/cognitive_udfs.rs)

| UDF | Input | Output | Purpose |
|-----|-------|--------|---------|
| `hamming(a, b)` | Binary, Binary | Int64 | Hamming distance (0-16384) |
| `similarity(a, b)` | Binary, Binary | Float64 | 1 - hamming/16384 |
| `resonate(fingerprint, query, threshold)` | Binary, Binary, Float64 | Boolean | Similarity filter |

## 7. Schema Flow: Query → Storage → Result

```
User query: "MATCH (n:Person) WHERE RESONATE(n.fingerprint, $q, 0.7) RETURN n.name"

1. PARSE (lance-graph parser.rs or planner CypherParse strategy)
   → AST: Match { pattern: [Node("n", "Person")], where: Resonate(...) }

2. PLAN (thinking orchestration)
   → Semiring: XorBundle (RESONATE detected)
   → Style: Analytical (default)
   → Elevation: L0:Point → try BindSpace first

3. BIND SPACE LOOKUP (ladybug-rs)
   → Schema: address=UInt16, fingerprint=FixedSizeBinary(2048)
   → Filter: zone='surface', prefix=0x02 (Cypher nodes)
   → Returns: Vec<(Addr, &[u64;256])>

4. ELEVATION → L2:Cascade (too many candidates)
   → CAM-PQ encode: [u64;256] → [u8;6] per candidate
   → Distance tables: 6×256 f32 = 6KB (L1 cache)
   → Cascade: HEEL → BRANCH → full ADC
   → Returns: Vec<(index, f32)> top-K

5. COLD PATH JOIN (DataFusion)
   → Schema: node_id=UInt32, label=Utf8, name=Utf8
   → Join: top-K indices → metadata.nodes RecordBatch
   → Returns: RecordBatch with name column

6. RESULT
   → Arrow RecordBatch: [name: Utf8] with K rows
```

## 8. Schema Migration Considerations

### Schemas that MUST NOT change (backward compat):
- `graph_nodes.lance` / `graph_rels.lance` — upstream Lance format
- Arrow Bridge schemas — consumed by DataFusion

### Schemas that CAN evolve (extension tables):
- `cognitive_nodes.lance` — add columns for CAM-PQ codes, palette indices
- `neighborhoods.lance` — add Base17 encoded scent
- Container word layout — reserved words W[64..127] are available

### Proposed additions:
```
cognitive_nodes.lance (v2):
  + cam_pq_code    FixedSizeBinary(6)    -- CAM-PQ 6-byte fingerprint
  + palette_spo    FixedSizeBinary(3)    -- PaletteEdge (s_idx, p_idx, o_idx)
  + base17_s       FixedSizeBinary(34)   -- Base17 subject plane
  + base17_p       FixedSizeBinary(34)   -- Base17 predicate plane
  + base17_o       FixedSizeBinary(34)   -- Base17 object plane
  + elevation_hint UInt8                 -- last successful ElevationLevel
```

This adds 6 + 3 + 102 + 1 = **112 bytes** per node for the compressed codec columns,
enabling the full HHTL cascade without decompressing the 6KB full planes.
