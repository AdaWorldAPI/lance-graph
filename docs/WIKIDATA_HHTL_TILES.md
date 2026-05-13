# WIKIDATA VIA LANCE + bgz17
# Zero-Copy Storage, Built-in RaBitQ, bgz17 Cascade

> The files stay in S3. LanceDB mmaps them zero-copy. RaBitQ is built-in.
> bgz17's precision cascade ranks results. No custom tile management.

## 1. Why Lance, Not Custom Tiles

LanceDB already provides everything the previous design tried to build by hand:

- Zero-copy reads via mmap (no deserialization, no memory copy)
- S3/object store as native backend (compute-storage separation)
- Built-in IVF_RQ (RaBitQ quantization, 1 bit/dim + corrections)
- Columnar format (read scent column without touching 6 KB planes)
- Automatic versioning (time travel on Wikidata snapshots)
- Fragment-level caching (LanceDB manages prefetch automatically)

## 2. Actual Numbers from Code

From `ndarray/src/hpc/spo_bundle.rs`:
- Per plane: 16,384 bits = 2,048 bytes = 2 KB
- SPO triple: 3 × 2 KB = 6 KB (Exact Hamming)
- spo_bundle Level A: 8,192 bits = 1 KB (metadata index)
- spo_bundle Level B: 16,384 bits = 2 KB (holographic)
- Golden shift A: 3,131 (odd, gcd=1, full orbit — de-esser)
- Golden shift B: 6,261 (odd, gcd=1, full orbit — de-esser)

From `ndarray/src/hpc/zeck.rs`:
- ZeckF64: 8 bytes progressive (scent + 7 resolution bytes)
- Scent (byte 0): 7 boolean lattice bits, 19/128 legal patterns
- Progressive reading: byte 0 alone gives ρ ≈ 0.94 rank correlation

From `bgz17/src/lib.rs` — precision cascade:

| Level | Size | ρ | Metric-safe | Use |
|-------|------|---|-------------|-----|
| Scent | 1 byte | 0.937 | ❌ NO | HEEL pre-filter only |
| Palette | 3 bytes | 0.965 | ✓ | CLAM/CAKES pruning |
| Base | 102 bytes | 0.992 | ✓ | fine ranking (i16[17]) |
| Exact | 6 KB | 1.000 | ✓ | ground truth (Hamming) |

From `bgz17/src/rabitq_compat.rs` — already bridges RaBitQ ↔ bgz17:

```rust
pub struct RaBitQEncoding {
    pub binary: Vec<u64>,       // RaBitQ sign bits (1 bit/dim)
    pub norm: f32,              // L2 norm correction
    pub dot_correction: f32,    // unbiased distance scalar
    pub palette: PaletteEdge,   // bgz17 palette (3 bytes)
}
```

## 3. Lance Table Schema

```sql
CREATE TABLE wikidata_entities (
    qid           BIGINT PRIMARY KEY,
    label         VARCHAR,
    -- bgz17 cascade columns (read bottom-up by precision level)
    scent         TINYINT UNSIGNED,           -- 1 byte, ρ=0.937
    palette       FIXED_SIZE_BINARY(3),       -- 3 bytes, ρ=0.965
    zeckf64       BIGINT,                     -- 8 bytes, ρ=0.94 progressive
    base17        FIXED_SIZE_BINARY(102),     -- 102 bytes, ρ=0.992
    plane_s       FIXED_SIZE_BINARY(2048),    -- 2 KB, exact
    plane_p       FIXED_SIZE_BINARY(2048),    -- 2 KB, exact
    plane_o       FIXED_SIZE_BINARY(2048),    -- 2 KB, exact
    -- RaBitQ (for LanceDB IVF_RQ built-in ANN)
    rabitq_vec    FIXED_SIZE_LIST(Float32, D),
    rabitq_norm   FLOAT,
    rabitq_dot    FLOAT,
    -- DeepNSM projection
    nsm_rank      SMALLINT UNSIGNED,          -- 12-bit concept mapping
    nsm_cam       FIXED_SIZE_BINARY(6),       -- CAM-PQ fingerprint
);
```

Zero-copy: Lance reads ONLY the columns you query. Scent filter
reads 112 MB (112M × 1 byte), not 672 GB (112M × 6 KB).

## 4. Query Pipeline

```
Query: "scientists at CERN"

STEP 1 — DeepNSM (local, ~1μs):
  tokenize + parse → SPO(scientist, work, CERN)

STEP 2 — LanceDB IVF_RQ (built-in RaBitQ, zero-copy from S3):
  table.search(query_vector).index_type("IVF_RQ").limit(1000)
  → RaBitQ binary dot product + corrections → 1,000 candidates
  → Cost: ~10ms (S3 fragment fetch + RaBitQ scan)

STEP 3 — bgz17 Scent (HEEL, 1 byte per candidate):
  1,000 × zeckf64_scent_hamming() = 1 KB scan
  → 85% reject → 150 survive. Cost: ~1μs.

STEP 4 — bgz17 ZeckF64 (8 bytes per survivor):
  150 × zeckf64_progressive_distance() → full ranking
  → top-27. Cost: ~2μs.

STEP 5 — Hydrate winners (zero-copy column access):
  Lance reads plane_s + plane_p + plane_o for 27 entities ONLY
  27 × 6 KB = 162 KB from S3. Cost: ~20ms.

STEP 6 — bgz17 Exact Hamming (6 KB per winner):
  27 × hamming_distance_raw() → per-plane (S, P, O) distances
  → SimilarityTable → calibrated f32. Cost: ~10μs.

TOTAL:  ~30ms first query (S3). ~15μs cached (Lance cache hit).
```

## 5. Compressor/De-esser Analogy

**Compressor** (bgz17 palette/scent): normalizes dynamic range.
USA (500 properties) and random fish (8 properties) both encode
to the same 1-byte scent / 3-byte palette / 102-byte Base17.
Rare properties amplified. Common properties attenuated.

**De-esser** (Fibonacci golden shift): prevents SPO bleed.
Shift 3,131 on 8,192 bits. gcd=1, full orbit.
S, P, O planes at φ-spaced rotations. No aliasing. No moiré.
Unbinding is exact: `cyclic_shift(bits, D - shift)`.

**Limiter** (ZeckF64 lattice): 19/128 legal scent patterns.
85% of random bit patterns are ILLEGAL → built-in error detection.
No NaN possible (integer encoding throughout).

## 6. RaBitQ + bgz17: Same Data, Two Views

RaBitQ (LanceDB built-in): O(N) coarse ANN across 112M entities.
bgz17 (lance-graph): O(K) precision cascade on K candidates.

Both stored in the SAME Lance table. Both operate on the SAME entity.
`rabitq_compat.rs` already bridges them — same struct carries both
RaBitQ binary code AND bgz17 palette index.

```
RaBitQ finds 1,000 candidates from 112M entities  → 10ms (LanceDB)
bgz17 cascade ranks 1,000 → 27 winners             → 3μs (lance-graph)
bgz17 exact scores 27 with SPO decomposition        → 10μs (lance-graph)
```

## 7. Elevation = bgz17 Precision

| ElevationLevel | bgz17 Precision | Size/Entity | Lance Column |
|----------------|----------------|-------------|-------------|
| Point | Scent | 1 byte | `scent` |
| Scan | Palette | 3 bytes | `palette` |
| Cascade | Base (i16[17]) | 102 bytes | `base17` |
| Batch | Exact (Hamming) | 6 KB | `plane_s/p/o` |

Each level reads MORE bytes from the Lance table.
Each level is MORE expensive but MORE accurate.
The elevation operator decides WHEN to pay the cost.
Zero-copy: each level reads only its column, nothing more.

## 8. Scale Check

```
112M entities:
  IVF_RQ index:    ~1 GB RAM (LanceDB manages)
  Scent column:    112 MB (zero-copy, fits RAM)
  ZeckF64 column:  896 MB (zero-copy, fits RAM)
  Exact planes:    672 GB (S3, loaded only for winners)

1.65B statements:
  Edge table:      ~30 GB on S3 (Lance columnar)

Total RAM:         ~1.5 GB (LanceDB manages the rest)
S3 storage:        ~700 GB (zero-copy, pay only for what you read)

Query latency:
  First:           ~30ms (S3 fragment fetch)
  Cached:          ~15μs (Lance cache hit)
  Popular:         <1ms (Lance fragment cache warm)
```

No custom tiles. No custom prefetch. No custom index.
LanceDB handles storage. bgz17 handles precision. DeepNSM handles meaning.
