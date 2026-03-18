# Hot Path vs Cold Path Architecture

> **Generated**: 2026-03-17
> **Source of truth**: `crates/lance-graph/src/graph/blasgraph/` (hot), `crates/lance-graph/src/graph/metadata.rs` (cold)

---

## Overview

Lance-graph separates operations into two paths with fundamentally different performance profiles:

| Property        | Hot Path                              | Cold Path                              |
|-----------------|---------------------------------------|----------------------------------------|
| **Purpose**     | Binary vector search & distance       | Graph structure CRUD & Cypher queries  |
| **Data types**  | `u32`, `u64`, `[u64; 256]` — integer only | `String`, `HashMap`, `RecordBatch`    |
| **Arithmetic**  | XOR + popcount                        | Arrow serialization, DataFusion plans  |
| **Latency**     | ~50 cycles (L1), 2ms (10K-in-1M)     | Milliseconds to seconds               |
| **SIMD**        | 4-tier dispatch (AVX-512 → scalar)    | None                                   |
| **Allocations** | Zero on critical path                 | Vec, HashMap, String per operation     |
| **Files**       | `blasgraph/`, `vector_ops.rs`         | `metadata.rs`, `versioned.rs`          |

---

## Hot Path: Hamming Distance

### Core Operation

All hot-path operations reduce to `popcount(a XOR b)` — counting the number of differing bits between two binary vectors. No floating point on the tight loop.

### Implementation Tiers

#### Tier 1: SIMD Dispatch (`blasgraph/ndarray_bridge.rs:123-143`)

```rust
pub fn dispatch_hamming(a: &[u8], b: &[u8]) -> u64
```

Four-tier fallback chain with runtime CPU feature detection:

| Tier | Instruction Set       | Register Width | Detection                                    |
|------|-----------------------|----------------|----------------------------------------------|
| 1    | AVX-512 VPOPCNTDQ    | 512-bit        | `avx512vpopcntdq` + `avx512f`               |
| 2    | AVX-512BW             | 512-bit        | `avx512bw` + `avx512f`                      |
| 3    | AVX2                  | 256-bit        | `avx2`                                      |
| 4    | Scalar                | 64-bit         | Always available                             |

VPOPCNTDQ performs 512-bit popcount in a single instruction — the ideal case.
AVX-512BW and AVX2 use byte-level shuffle LUT for popcount emulation.

#### Tier 2: Word-Level (`blasgraph/types.rs:193-199`)

```rust
// BitVec: [u64; 256] = 16,384 bits
pub fn hamming_distance(&self, other: &BitVec) -> u32 {
    let mut dist = 0u32;
    for i in 0..VECTOR_WORDS {          // VECTOR_WORDS = 256
        dist += (self.words[i] ^ other.words[i]).count_ones();
    }
    dist
}
```

Operates directly on the `BitVec` type (256 x u64 words = 16,384-bit hyperdimensional vectors). Pure integer, no conversion.

#### Tier 3: Byte-Level (`datafusion_planner/vector_ops.rs:213-231`)

```rust
pub fn hamming_distance(a: &[u8], b: &[u8]) -> u32
```

General-purpose implementation for arbitrary-length byte slices. Processes 8 bytes at a time via `u64::from_le_bytes()` + `count_ones()`, with byte-level remainder handling.

#### Tier 4: DataFusion UDFs (`datafusion_planner/udf.rs:468-645`)

- `hamming_distance_func` — Arrow ColumnarValue → Float32Array of distances
- `hamming_similarity_func` — `1.0 - distance / total_bits`

These wrap Tier 3 for SQL queries. Supports Array-Array, Array-Scalar, Scalar-Scalar broadcast patterns.

### HDR Cascade: Three-Stage Filter (`blasgraph/hdr.rs`)

The Cascade eliminates 97%+ of candidates before full comparison:

```text
Stage 1 (1/16 sample)    →  reject if projected distance > threshold
        ↓ survivors
Stage 2 (1/4 sample)     →  reject if projected distance > threshold
        ↓ survivors
Stage 3 (full comparison) →  classify into band, collect top-k
```

#### Band Classification (`hdr.rs:468-484`)

Distances map to sigma bands (integer comparison, constant time):

| Band    | Criterion     | Meaning                          |
|---------|---------------|----------------------------------|
| Foveal  | `d < μ - 3σ`  | Exceptional match                |
| Near    | `d < μ - 2σ`  | Strong match                     |
| Good    | `d < μ - 1σ`  | Acceptable match                 |
| Weak    | `d < μ`        | Below average, not collected     |
| Reject  | `d ≥ μ`        | Noise                            |

For N=16,384 bits: `μ = 8192, σ = 64`.

#### Query Execution (`hdr.rs:591-675`)

```rust
pub fn query(
    &self,
    query: &[u64],
    candidates: &[&[u64]],
    top_k: usize,
) -> Vec<RankedHit>
```

- Early termination when `foveal.len() >= top_k`
- Results bucketed by band (Foveal first), sorted by `u32` distance within each bucket
- **No float anywhere** in the query path

---

## Cold Path: Metadata & Calibration

### MetadataStore (`metadata.rs:1-47`)

```rust
//! Metadata graph store — the cold path skeleton.
```

Structural graph data: nodes, edges, labels, properties. All operations are O(n) linear scans — appropriate for planning, not per-row execution.

```rust
pub struct NodeRecord {
    pub node_id: u32,
    pub label: String,
    pub properties: HashMap<String, Value>,
}

pub struct EdgeRecord {
    pub source: u32,
    pub target: u32,
    pub edge_type: String,
    pub properties: HashMap<String, Value>,
}

pub struct MetadataStore {
    nodes: Vec<NodeRecord>,
    edges: Vec<EdgeRecord>,
    // ...
}
```

**CRUD**: `add_node()`, `add_edge()`, `get_node()`, `get_edges_from()`, `remove_node()`, `remove_edge()`

**Arrow serialization**: `nodes_batch(label)`, `edges_batch(type)`, `to_datasets()` — builds `RecordBatch` for Cypher execution.

**Cypher query**:
```rust
pub async fn query(&self, cypher: &str) -> Result<RecordBatch> {
    let datasets = self.to_datasets();
    CypherQuery::new(cypher)?.with_config(config).execute(datasets, None).await
}
```

### Cascade Calibration (`hdr.rs:378-461`)

Cold-path operations that set up hot-path thresholds:

| Method                  | Purpose                                          | When Called          |
|-------------------------|--------------------------------------------------|----------------------|
| `Cascade::for_width(N)` | Theoretical init: μ=N/2, σ=isqrt(N/4)           | Startup              |
| `Cascade::calibrate()`  | Empirical init from sample distances             | Warmup phase         |
| `observe(distance)`     | Welford's online mean/variance update            | Every 1000 queries   |
| `recalibrate()`         | Rebuild bands from reservoir when shift detected | On `ShiftAlert`      |

**Shift detection**: every 1000 observations, checks if running μ has drifted by more than σ/2 from calibrated values. Returns `ShiftAlert` with old/new μ and σ.

**Reservoir sampling** (Vitter's Algorithm R): maintains 1000-element reservoir for empirical quantile thresholds. Used when distribution is non-normal (skewed, bimodal).

### Versioned Graph (`versioned.rs`)

Lance-backed ACID versioning for graph state. Cold-path persistence with time-travel support.

---

## Architectural Invariants

1. **No float on hot path** — distances are `u32`, thresholds are `u32`, square roots use integer Newton's method (`isqrt`)
2. **Hot path never allocates** — operates on borrowed slices (`&[u64]`, `&[u8]`)
3. **Cold path never blocks hot path** — calibration runs asynchronously, shift detection is periodic
4. **SIMD dispatch is runtime** — no compile-time feature gates; works on any x86_64 CPU
5. **Cascade is self-calibrating** — theoretical thresholds at startup, empirical after warmup, auto-adjusts on drift

---

## File Map

```text
Hot Path
├── crates/lance-graph/src/graph/blasgraph/
│   ├── hdr.rs              # Cascade query (3-stage), band classification
│   ├── ndarray_bridge.rs   # SIMD dispatch (AVX-512/AVX2/scalar)
│   ├── types.rs            # BitVec ([u64;256]) hamming_distance
│   ├── semiring.rs         # 7 semiring implementations
│   ├── ops.rs              # GraphBLAS free functions
│   └── sparse.rs           # Sparse vector/matrix ops
├── crates/lance-graph/src/datafusion_planner/
│   ├── vector_ops.rs       # hamming_distance(), hamming_similarity()
│   └── udf.rs              # DataFusion UDF wrappers

Cold Path
├── crates/lance-graph/src/graph/
│   ├── metadata.rs         # MetadataStore (nodes, edges, Cypher)
│   └── versioned.rs        # Lance ACID versioning
├── crates/lance-graph/src/graph/blasgraph/
│   └── hdr.rs              # Cascade calibration, shift detection, reservoir
```
