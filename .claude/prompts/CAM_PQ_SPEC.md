# CAM-PQ: Content-Addressable Memory as Product Quantization

## The Insight

FAISS Product Quantization (PQ) and CLAM 48-bit archetypes are the same structure.

```
FAISS PQ6x8:                          CLAM 48-bit:
  Split D-dim vector into 6 parts      6 bytes: HEEL/BRANCH/TWIG_A/TWIG_B/LEAF/GAMMA
  Each part → nearest centroid (u8)     Each byte → archetype index (u8)
  6 × 256-entry codebooks              6 × 256-entry codebooks
  48 bits per vector                    48 bits per vector
  Distance = 6 table lookups + sum     Distance = 6 table lookups + sum

  Training: minimize reconstruction    Training: maximize semantic separation
  Loss: ||x - decode(encode(x))||²     Loss: intent distance (domain-specific)

  STORAGE FORMAT: IDENTICAL
  QUERY PROTOCOL: IDENTICAL
  SIMD PATH: IDENTICAL
```

CAM-PQ unifies them: train codebooks for your domain (semantic or geometric),
encode as 6-byte fingerprints, store in LanceDB, query through lance-graph
DataFusion, compile distance computation via jitson.

170× compression. 500M candidates/second. Semantic AND geometric search
through the same index.

## Architecture

```
                    TRAINING (offline, once per domain)
                    ═══════════════════════════════════
Raw vectors (1024D f32)
    │
    ├─ Geometric mode: k-means per subspace (standard FAISS PQ)
    │   → codebooks minimize reconstruction error
    │
    ├─ Semantic mode: CLAM archetype clustering
    │   → codebooks maximize intent separation
    │
    └─ Hybrid mode: geometric init + semantic fine-tune
        → codebooks balance reconstruction AND meaning
    │
    ▼
6 codebooks × 256 centroids × D/6 floats each
    │
    ▼
Saved as cam_codebook.lance (one table, 1536 rows)


                    ENCODING (per vector)
                    ═════════════════════
Input: 1024D f32 vector (4096 bytes)
    │
    ▼
Split into 6 subvectors of D/6 dimensions
    │
    ├─ subvec[0] → nearest centroid in codebook[0] → byte 0 (HEEL)
    ├─ subvec[1] → nearest centroid in codebook[1] → byte 1 (BRANCH)
    ├─ subvec[2] → nearest centroid in codebook[2] → byte 2 (TWIG_A)
    ├─ subvec[3] → nearest centroid in codebook[3] → byte 3 (TWIG_B)
    ├─ subvec[4] → nearest centroid in codebook[4] → byte 4 (LEAF)
    └─ subvec[5] → nearest centroid in codebook[5] → byte 5 (GAMMA)
    │
    ▼
Output: 6-byte CAM fingerprint (48 bits)
Compression: 4096 bytes → 6 bytes = 682× for 1024D
             (170× for standard 256D Jina embeddings)


                    QUERYING (per search)
                    ═════════════════════
Query: 1024D f32 vector (full precision, NOT compressed)
    │
    ▼
Precompute distance tables (once per query):
    For each subspace s ∈ [0..6):
        For each centroid c ∈ [0..256):
            dist_table[s][c] = ||query_sub[s] - codebook[s][c]||²
    │
    Total: 6 × 256 = 1536 floats = 6KB (fits in L1 cache)
    │
    ▼
Scan candidates (per candidate = 6 byte lookups + 5 adds):
    For candidate with CAM fingerprint [b0, b1, b2, b3, b4, b5]:
        distance = dist_table[0][b0]
                 + dist_table[1][b1]
                 + dist_table[2][b2]
                 + dist_table[3][b3]
                 + dist_table[4][b4]
                 + dist_table[5][b5]
    │
    ▼
SIMD: process 8 candidates per cycle (48 bytes = 8 × 6 bytes per zmm)
      VPGATHERDD loads 8 distances per table in one instruction
      VPADDD accumulates across 6 subspaces
      VCMPPS filters by threshold
```

## LanceDB Storage Schema

```sql
-- Main data table
CREATE TABLE vectors (
    id          BIGINT PRIMARY KEY,
    cam         FIXED_SIZE_BINARY(6),     -- 48-bit CAM fingerprint
    metadata    VARCHAR,                   -- arbitrary metadata
    timestamp   TIMESTAMP
);

-- Codebook table (trained once, immutable)
CREATE TABLE cam_codebook (
    subspace    TINYINT,                   -- 0-5 (HEEL through GAMMA)
    centroid_id TINYINT UNSIGNED,           -- 0-255
    vector      FIXED_SIZE_LIST(FLOAT, N),  -- centroid vector (D/6 dimensions)
    label       VARCHAR                     -- semantic label (for CLAM mode)
);

-- Distance table cache (precomputed per query, ephemeral)
CREATE TEMP TABLE dist_cache (
    subspace    TINYINT,
    centroid_id TINYINT UNSIGNED,
    distance    FLOAT
);
```

In Lance columnar format, the `cam` column is 6 bytes per row.
1 million vectors = 6MB. 1 billion = 6GB. Fits in RAM.

## lance-graph Integration

### DataFusion UDF

```rust
// crates/lance-graph/src/cam_pq.rs

use arrow::array::{FixedSizeBinaryArray, Float32Array, UInt64Array};
use arrow::datatypes::DataType;
use datafusion::logical_expr::{ScalarUDF, Volatility};

/// The 6 codebooks, each 256 centroids of D/6 dimensions.
pub struct CamCodebook {
    /// codebooks[subspace][centroid] = centroid vector
    pub codebooks: Vec<Vec<Vec<f32>>>,  // [6][256][D/6]
    pub subspace_dim: usize,            // D/6
    pub total_dim: usize,               // D
}

/// Precomputed distance tables for one query.
/// 6 × 256 = 1536 floats = 6KB. Fits in L1.
pub struct DistanceTables {
    pub tables: [[f32; 256]; 6],
}

impl CamCodebook {
    /// Load codebook from Lance table.
    pub async fn load(dataset: &Dataset) -> Result<Self> {
        // SELECT subspace, centroid_id, vector FROM cam_codebook
        // ORDER BY subspace, centroid_id
    }

    /// Encode a full vector to 6-byte CAM fingerprint.
    pub fn encode(&self, vector: &[f32]) -> [u8; 6] {
        let mut cam = [0u8; 6];
        for s in 0..6 {
            let sub = &vector[s * self.subspace_dim..(s + 1) * self.subspace_dim];
            let mut best_dist = f32::MAX;
            let mut best_id = 0u8;
            for (c, centroid) in self.codebooks[s].iter().enumerate() {
                let dist = squared_l2(sub, centroid);
                if dist < best_dist {
                    best_dist = dist;
                    best_id = c as u8;
                }
            }
            cam[s] = best_id;
        }
        cam
    }

    /// Decode a 6-byte CAM fingerprint to approximate vector.
    pub fn decode(&self, cam: &[u8; 6]) -> Vec<f32> {
        let mut vec = Vec::with_capacity(self.total_dim);
        for s in 0..6 {
            vec.extend_from_slice(&self.codebooks[s][cam[s] as usize]);
        }
        vec
    }

    /// Precompute distance tables for a query vector.
    /// This is the key to ADC speed: O(6 × 256) precompute,
    /// then O(6) per candidate instead of O(D).
    pub fn precompute_distances(&self, query: &[f32]) -> DistanceTables {
        let mut tables = [[0.0f32; 256]; 6];
        for s in 0..6 {
            let q_sub = &query[s * self.subspace_dim..(s + 1) * self.subspace_dim];
            for c in 0..256 {
                tables[s][c] = squared_l2(q_sub, &self.codebooks[s][c]);
            }
        }
        DistanceTables { tables }
    }
}

impl DistanceTables {
    /// Compute distance to a single CAM fingerprint.
    /// 6 table lookups + 5 adds.
    #[inline(always)]
    pub fn distance(&self, cam: &[u8; 6]) -> f32 {
        self.tables[0][cam[0] as usize]
            + self.tables[1][cam[1] as usize]
            + self.tables[2][cam[2] as usize]
            + self.tables[3][cam[3] as usize]
            + self.tables[4][cam[4] as usize]
            + self.tables[5][cam[5] as usize]
    }

    /// Batch distance for N candidates.
    /// SIMD: VPGATHERDD + VPADDD, 8 candidates per cycle.
    pub fn distance_batch(&self, cams: &[[u8; 6]]) -> Vec<f32> {
        #[cfg(target_arch = "x86_64")]
        if is_x86_feature_detected!("avx512f") {
            return unsafe { self.distance_batch_avx512(cams) };
        }
        cams.iter().map(|c| self.distance(c)).collect()
    }

    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx512f")]
    unsafe fn distance_batch_avx512(&self, cams: &[[u8; 6]]) -> Vec<f32> {
        use core::arch::x86_64::*;

        let n = cams.len();
        let mut result = vec![0.0f32; n];

        // Process 16 candidates at a time
        let chunks = n / 16;
        for chunk in 0..chunks {
            let base = chunk * 16;
            let mut acc = _mm512_setzero_ps();

            for s in 0..6 {
                // Gather 16 centroid indices for subspace s
                let indices = _mm512_set_epi32(
                    cams[base + 15][s] as i32, cams[base + 14][s] as i32,
                    cams[base + 13][s] as i32, cams[base + 12][s] as i32,
                    cams[base + 11][s] as i32, cams[base + 10][s] as i32,
                    cams[base + 9][s] as i32,  cams[base + 8][s] as i32,
                    cams[base + 7][s] as i32,  cams[base + 6][s] as i32,
                    cams[base + 5][s] as i32,  cams[base + 4][s] as i32,
                    cams[base + 3][s] as i32,  cams[base + 2][s] as i32,
                    cams[base + 1][s] as i32,  cams[base][s] as i32,
                );

                // Gather distances from precomputed table
                let distances = _mm512_i32gather_ps::<4>(
                    indices,
                    self.tables[s].as_ptr() as *const u8,
                );

                // Accumulate
                acc = _mm512_add_ps(acc, distances);
            }

            // Store results
            _mm512_storeu_ps(result[base..].as_mut_ptr(), acc);
        }

        // Scalar tail
        for i in (chunks * 16)..n {
            result[i] = self.distance(&cams[i]);
        }

        result
    }
}

/// DataFusion UDF: cam_distance(query_vector, cam_column) → distance
pub fn cam_distance_udf(codebook: Arc<CamCodebook>) -> ScalarUDF {
    // Register as DataFusion UDF for lance-graph queries:
    // SELECT id, cam_distance(ARRAY[0.1, 0.2, ...], cam) AS dist
    // FROM vectors
    // ORDER BY dist
    // LIMIT 10
}
```

### jitson Compilation of Distance Tables

The distance tables are precomputed per query. But the TABLE STRUCTURE
(6 subspaces × 256 entries) is fixed. jitson compiles the gather + add
loop with the subspace count and entry count as immediates:

```json
{
  "kernel": "cam_pq_adc",
  "scan": {
    "threshold": 100,
    "record_size": 6,
    "top_k": 32
  },
  "pipeline": [
    { "stage": "gather",   "subspaces": 6, "centroids": 256 },
    { "stage": "accumulate", "avx512": "vpaddd" },
    { "stage": "threshold",  "avx512": "vcmpps" }
  ],
  "cranelift": {
    "preset": "sapphire_rapids",
    "opt_level": "speed"
  }
}
```

The compiled kernel:
- Loop count (6 subspaces) baked as unrolled iterations
- Table base addresses baked as immediate pointers
- Gather stride (4 bytes per float) baked as immediate
- Threshold baked as VCMPPS immediate
- No loop overhead for subspaces — fully unrolled

### lance-graph Query Path

```rust
// Cypher-like query through DataFusion:

// "Find 10 nearest neighbors to this vector"
let query = "
    SELECT id, cam_distance($query, cam) AS dist
    FROM vectors
    ORDER BY dist ASC
    LIMIT 10
";

// Under the hood:
// 1. Precompute distance tables from $query (once, 6KB)
// 2. Scan cam column (6 bytes per row, sequential prefetch)
// 3. For each candidate: 6 table lookups + 5 adds (via jitson kernel)
// 4. Top-K heap with early termination
// 5. Return 10 nearest

// With IVF coarse filter (optional):
let query = "
    SELECT id, cam_distance($query, cam) AS dist
    FROM vectors
    WHERE ivf_partition = cam_nearest_partition($query)
    ORDER BY dist ASC
    LIMIT 10
";
```

## Codebook Training

### Geometric Mode (standard PQ)

```rust
/// Train codebooks via k-means on subvectors.
/// This is standard FAISS PQ training.
pub fn train_geometric(
    vectors: &[Vec<f32>],
    num_subspaces: usize,    // 6 for CAM-PQ
    num_centroids: usize,    // 256 for 8-bit codes
    iterations: usize,       // k-means iterations
) -> CamCodebook {
    let dim = vectors[0].len();
    let sub_dim = dim / num_subspaces;

    let mut codebooks = Vec::with_capacity(num_subspaces);
    for s in 0..num_subspaces {
        // Extract subvectors for this subspace
        let subs: Vec<&[f32]> = vectors.iter()
            .map(|v| &v[s * sub_dim..(s + 1) * sub_dim])
            .collect();

        // k-means clustering
        let centroids = kmeans(&subs, num_centroids, iterations);
        codebooks.push(centroids);
    }

    CamCodebook { codebooks, subspace_dim: sub_dim, total_dim: dim }
}
```

### Semantic Mode (CLAM archetype training)

```rust
/// Train codebooks using domain-specific semantic labels.
/// Centroids are pulled toward semantically similar vectors
/// and pushed away from semantically different ones.
pub fn train_semantic(
    vectors: &[Vec<f32>],
    labels: &[Vec<String>],   // semantic labels per vector
    num_subspaces: usize,
    num_centroids: usize,
    alpha: f32,               // semantic weight vs reconstruction
) -> CamCodebook {
    // 1. Initialize with geometric k-means
    let mut codebook = train_geometric(vectors, num_subspaces, num_centroids, 10);

    // 2. Fine-tune: for each pair (v_i, v_j):
    //    if same_label(i, j): pull centroids closer
    //    if diff_label(i, j): push centroids apart
    for _epoch in 0..50 {
        for i in 0..vectors.len() {
            let cam_i = codebook.encode(&vectors[i]);
            for j in (i + 1)..vectors.len().min(i + 100) {
                let cam_j = codebook.encode(&vectors[j]);
                let semantic_sim = jaccard(&labels[i], &labels[j]);
                let cam_dist = cam_distance(&cam_i, &cam_j);

                // Semantic loss: high similarity should mean low CAM distance
                // Pull or push centroids based on label agreement
                let grad = alpha * (semantic_sim - (1.0 / (1.0 + cam_dist)));
                codebook.adjust_centroids(&cam_i, &cam_j, grad);
            }
        }
    }

    codebook
}
```

### Hybrid Mode (best of both)

```rust
/// Geometric init + semantic fine-tune.
/// Reconstruction error stays low (geometric) while
/// semantic separation improves (CLAM labels).
pub fn train_hybrid(
    vectors: &[Vec<f32>],
    labels: &[Vec<String>],
    num_subspaces: usize,
    num_centroids: usize,
) -> CamCodebook {
    // Phase 1: geometric for reconstruction
    let mut codebook = train_geometric(vectors, num_subspaces, num_centroids, 50);

    // Phase 2: semantic fine-tune (smaller learning rate)
    train_semantic_finetune(&mut codebook, vectors, labels, 0.1);

    codebook
}
```

## The CLAM ↔ FAISS Bridge

### CLAM byte names map to PQ subspaces:

```
CLAM byte    PQ subspace    Semantic role          Geometric role
─────────    ───────────    ──────────────         ──────────────
HEEL         subspace[0]    Coarse category        First D/6 dims
BRANCH       subspace[1]    Archetype selection    Dims D/6..2D/6
TWIG_A       subspace[2]    Shape parameter A      Dims 2D/6..3D/6
TWIG_B       subspace[3]    Shape parameter B      Dims 3D/6..4D/6
LEAF         subspace[4]    Fine detail            Dims 4D/6..5D/6
GAMMA        subspace[5]    Euler tension/energy   Last D/6 dims
```

The naming doesn't change the math. But it gives MEANING to each byte.
When you see CAM fingerprint `[42, 109, 7, 200, 15, 88]`, you can read:
- HEEL=42: category "surveillance technology"
- BRANCH=109: archetype "drone system"
- TWIG_A=7: shape "fixed-wing"
- TWIG_B=200: shape "autonomous"
- LEAF=15: detail "thermal imaging"
- GAMMA=88: tension 0.34 (moderate concern)

In pure PQ mode, those same bytes are just subquantizer indices.
In CLAM mode, they're LABELS. Same bits, different semantics.

## PackedDatabase Integration

The 6-byte CAM fingerprints fit perfectly into PackedDatabase's stroke layout:

```
Stroke 1 (1 byte = HEEL):
  All HEEL bytes contiguous across candidates.
  Coarse rejection: if HEEL doesn't match any target category, skip.
  90% of candidates eliminated by one byte comparison.

Stroke 2 (2 bytes = HEEL + BRANCH):
  HEEL+BRANCH contiguous for survivors of Stroke 1.
  Medium filter: category + archetype match.
  90% of survivors eliminated.

Stroke 3 (6 bytes = full CAM):
  Full ADC distance on final survivors.
  Precise ranking.
```

This gives 99% rejection before full distance computation.
For 1M vectors: Stroke 1 scans 1M × 1 byte = 1MB.
Survivors (10K): Stroke 2 scans 10K × 2 bytes = 20KB.
Survivors (100): Stroke 3 computes 100 × 6 lookups + 5 adds.
Total: ~1MB read instead of 6MB. 6× bandwidth reduction.

```rust
impl PackedDatabase {
    /// Pack CAM fingerprints into stroke-aligned layout.
    pub fn pack_cam(fingerprints: &[[u8; 6]]) -> Self {
        let n = fingerprints.len();

        // Stroke 1: HEEL bytes only (1 byte per candidate)
        let stroke1: Vec<u8> = fingerprints.iter().map(|f| f[0]).collect();

        // Stroke 2: HEEL + BRANCH (2 bytes per candidate)
        let stroke2: Vec<u8> = fingerprints.iter()
            .flat_map(|f| [f[0], f[1]])
            .collect();

        // Stroke 3: full CAM (6 bytes per candidate)
        let stroke3: Vec<u8> = fingerprints.iter()
            .flat_map(|f| f.iter().copied())
            .collect();

        Self { stroke1, stroke2, stroke3, num_candidates: n, .. }
    }

    /// Cascade query with CAM-PQ distance tables.
    pub fn cam_query(
        &self,
        dist_tables: &DistanceTables,
        heel_threshold: f32,    // Stroke 1: reject if HEEL distance > this
        branch_threshold: f32,  // Stroke 2: reject if HEEL+BRANCH distance > this
        top_k: usize,
    ) -> Vec<(usize, f32)> {
        // Stroke 1: scan HEEL bytes
        let mut survivors: Vec<usize> = Vec::new();
        for i in 0..self.num_candidates {
            let heel_dist = dist_tables.tables[0][self.stroke1[i] as usize];
            if heel_dist < heel_threshold {
                survivors.push(i);
            }
        }

        // Stroke 2: scan HEEL+BRANCH for survivors
        let mut refined: Vec<usize> = Vec::new();
        for &i in &survivors {
            let base = i * 2;
            let dist = dist_tables.tables[0][self.stroke2[base] as usize]
                     + dist_tables.tables[1][self.stroke2[base + 1] as usize];
            if dist < branch_threshold {
                refined.push(i);
            }
        }

        // Stroke 3: full ADC on refined candidates
        let mut hits: Vec<(usize, f32)> = refined.iter().map(|&i| {
            let base = i * 6;
            let cam = &self.stroke3[base..base + 6];
            let dist = dist_tables.distance(cam.try_into().unwrap());
            (i, dist)
        }).collect();

        // Top-K
        hits.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        hits.truncate(top_k);
        hits
    }
}
```

## Lossless Guarantee

"Lossless" means: given the codebook, the CAM fingerprint reconstructs
EXACTLY to the centroid vectors. There's no further quantization error
beyond the initial PQ assignment.

```
encode(vector) → cam          (lossy: vector → nearest centroid)
decode(cam) → centroids       (lossless: cam → exact centroids)
distance(query, cam) = ADC    (exact: same result as distance(query, decode(cam)))

The ADC distance IS the distance to the reconstructed vector.
No approximation beyond the initial quantization.
If you train codebooks well, reconstruction error < 5%.
For CLAM semantic mode, "error" is redefined as semantic distance.
```

## IVF + CAM-PQ (coarse + fine)

For billion-scale: add IVF (Inverted File) coarse partitioning on top.

```rust
/// IVF partitioning: cluster CAM fingerprints into N lists.
/// Query probes P lists, then runs CAM-PQ cascade within each.
pub struct IvfCamIndex {
    /// Coarse centroids (full vectors, not compressed)
    coarse_centroids: Vec<Vec<f32>>,
    /// Per-list: PackedDatabase of CAM fingerprints
    lists: Vec<PackedDatabase>,
    /// Codebook for CAM encoding/decoding
    codebook: CamCodebook,
}

impl IvfCamIndex {
    pub fn search(&self, query: &[f32], top_k: usize, n_probe: usize) -> Vec<(usize, f32)> {
        // 1. Find n_probe nearest coarse centroids
        let partitions = self.nearest_partitions(query, n_probe);

        // 2. Precompute distance tables (once)
        let dist_tables = self.codebook.precompute_distances(query);

        // 3. Search each partition via CAM cascade
        let mut all_hits = Vec::new();
        for &p in &partitions {
            let hits = self.lists[p].cam_query(&dist_tables, 50.0, 25.0, top_k);
            all_hits.extend(hits);
        }

        // 4. Merge and return top-K
        all_hits.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        all_hits.truncate(top_k);
        all_hits
    }
}
```

## Implementation in lance-graph

### New files

```
lance-graph/src/cam_pq/
  mod.rs              — re-exports
  codebook.rs         — CamCodebook, encode, decode, train_*
  distance.rs         — DistanceTables, distance_batch, AVX-512 path
  packed.rs           — PackedDatabase::pack_cam, cam_query cascade
  udf.rs              — DataFusion UDF registration
  ivf.rs              — IvfCamIndex (optional, for billion-scale)
  jitson_kernel.rs    — jitson template for ADC scan (behind feature flag)
```

### Estimated lines

```
codebook.rs       ~300  (encode, decode, 3 training modes)
distance.rs       ~250  (precompute, scalar, AVX-512 gather)
packed.rs         ~200  (pack_cam, stroke layout, cascade query)
udf.rs            ~150  (DataFusion integration)
ivf.rs            ~200  (coarse partitioning, optional)
jitson_kernel.rs  ~100  (ADC kernel template)
─────────────────────
TOTAL:           ~1200 lines
```

### Feature flags

```toml
[features]
cam-pq = []                     # codebook + distance + packed (no jitson)
cam-pq-jit = ["cam-pq", "ndarray/jit-native"]  # + jitson compiled ADC kernel
cam-pq-ivf = ["cam-pq"]        # + IVF coarse partitioning
```

## Performance Targets

```
Operation              Target          Hardware
──────────────────     ──────────      ────────────
Encode 1 vector        < 10µs          any
Decode 1 vector        < 1µs           any
Precompute dist tables < 50µs          any
Distance (1 candidate) < 10ns          scalar
Distance (16 cands)    < 20ns          AVX-512 gather
Cascade query (1M)     < 2ms           AVX-512
IVF+CAM query (1B)     < 20ms          AVX-512, 10 probes
Codebook training (1M) < 60s           geometric
Codebook training (1M) < 300s          semantic

Storage: 6 bytes/vector + codebook overhead (1536 × D/6 × 4 bytes)
For D=1024: codebook = 1536 × 170 × 4 = ~1MB (negligible)
For 1M vectors: 6MB data + 1MB codebook = 7MB total
For 1B vectors: 6GB data + 1MB codebook = ~6GB total
```

## Applications

### aiwar knowledge graph (q2 cockpit)

Encode aiwar node embeddings as 6-byte CAM fingerprints.
Store in lance-graph. Query via DataFusion.
The thinking graph uses CAM distance for Foveal/Parafoveal classification:
  HEEL match → Foveal (familiar category, skip reasoning)
  HEEL mismatch → Parafoveal (novel category, engage Layer 5)

### Ada consciousness (Jina embeddings → CAM)

Current: 1024D Jina embeddings stored raw.
With CAM-PQ: 6 bytes per memory. 170× compression.
Semantic mode: CLAM archetypes trained on Ada's qualia space.
HEEL = emotional valence, BRANCH = topic, GAMMA = intensity.

### Pumpkin (entity/block search)

Entity positions encoded as CAM fingerprints.
Spatial hash + CAM cascade = O(1) entity lookup.
Block state palettes as CAM codebooks — property queries become
CAM distance computations compiled via jitson.

### C64 SID (audio archetype codebook)

SID register states encoded as CAM fingerprints.
256 archetypes per CLAM subspace = the instrument codebook.
CAM distance = "how similar is this SID voice to that violin patch?"
Nearest archetype → compile via jitson → play as physical model.
