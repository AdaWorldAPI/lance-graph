# Adjacency Synergy Map: CSR Models, BLAS×PQ×BGZ17, KuzuDB

> *The mathematical synergy: adjacency as OBJECT (KuzuDB), as ALGEBRA (BlasGraph), as TENSOR (ndarray).*

## 1. Complete Adjacency Model Inventory

### Model 1: AdjacencyStore (lance-graph-planner, Kuzu-inspired)
**File**: `lance-graph-planner/src/adjacency/csr.rs:12`

```rust
pub struct AdjacencyStore {
    pub csr_offsets: Vec<u64>,      // node → start offset
    pub csr_targets: Vec<u64>,      // packed sorted target IDs
    pub csr_edge_ids: Vec<u64>,     // edge ID per entry
    pub csc_offsets: Vec<u64>,      // transposed for backward traversal
    pub csc_sources: Vec<u64>,
    pub csc_edge_ids: Vec<u64>,
    pub edge_properties: EdgeProperties,  // columnar (not row-major!)
    pub rel_type: String,           // one store per rel type (Kuzu pattern)
}
```

**Primitives**:
- `adjacent(source) -> &[u64]` — core forward traversal
- `adjacent_incoming(target) -> &[u64]` — core backward traversal
- `batch_adjacent(sources) -> AdjacencyBatch` — vectorized multi-source
- `intersect_adjacent(a_adj, b_adj) -> Vec<u64>` — Kuzu WCO join

**Properties**: NARS truth values stored as columns: `truth_f`, `truth_c`, `truth_t`

**Vision** (from `adjacency/mod.rs`): *"Future: VSA Focus-of-Awareness — each node's adjacency
pattern encoded as a 10K Hamming vector. High similarity with query = this node's neighborhood
is relevant. Prefilter before actual traversal. Attention for graphs."*

### Model 2: GrBMatrix (lance-graph/blasgraph, GraphBLAS-inspired)
**File**: `lance-graph/blasgraph/matrix.rs` + `sparse.rs`

```rust
pub struct GrBMatrix<T> {
    nrows: usize,
    ncols: usize,
    storage: StorageFormat<T>,  // COO, CSR, CSC, or HyperCSR
    descriptor: Descriptor,
}
```

**Sparse formats** (842L in `sparse.rs`):
- `CooStorage<T>`: Coordinate list (best for construction)
- `CsrStorage<T>`: Compressed Sparse Row (best for row-wise access)
- `CscStorage<T>`: Compressed Sparse Column (best for column-wise access)
- `HyperCsrStorage<T>`: Hypersparse CSR (only stores non-empty rows)

**Operations** (`ops.rs`, 510L):
- `mxm(A, B, semiring)` — matrix-matrix multiply
- `mxv(A, v, semiring)` — matrix-vector multiply
- `ewisemult(A, B, op)` — element-wise multiply
- `extract_column(j)`, `extract_row(i)`
- `reduce(monoid)` — reduce all elements

**T = `HdrScalar`**: either `BitVec` (16,384-bit) or `f64` depending on semiring.

### Model 3: ScentCsr (lance-graph/blasgraph)
**File**: `lance-graph/blasgraph/neighborhood_csr.rs:17`

```rust
pub struct ScentCsr {
    pub nrows: usize,
    pub ncols: usize,
    pub row_ptrs: Vec<usize>,
    pub col_indices: Vec<usize>,
    pub values: Vec<u8>,         // scent byte (ZeckF64 byte 0)
}
```

**Purpose**: Secondary search path for standard graph algorithms (BFS, PageRank).
Primary search uses neighborhood scent vectors directly.
Edge weight = 1 byte scent (0-255, higher = closer).

### Model 4: PaletteCsr (bgz17)
**File**: `bgz17/palette_csr.rs:20`

```rust
pub struct PaletteCsr {
    pub distances: SpoDistanceMatrices,   // 256×256 per plane
    pub assignments: Vec<u8>,              // node → archetype (palette index)
    pub palette_indices: Vec<PaletteEdge>, // per-node (s_idx, p_idx, o_idx)
    pub archetype_members: Vec<Vec<usize>>,// archetype → member nodes
    pub edge_topology: Vec<Vec<(u8, u8)>>, // archetype-level edges
    pub k: usize,                          // number of archetypes (≤256)
}
```

**ArchetypeTree**: Ball-tree over archetypes for O(k²) search.
**Purpose**: Instead of O(N²) pairwise distances, map to k archetypes → O(k²) = O(65K).

### Model 5: VerbCodebook (ndarray/hpc/graph.rs)
**File**: `ndarray/src/hpc/graph.rs:22`

```rust
pub struct VerbCodebook {
    verbs: Vec<(String, usize)>,  // verb name → rotation offset
    base_dim: usize,              // 4096
}
```

**Edge encoding**: `edge = permute(src, verb_offset) XOR tgt`
**Decode**: `target = permute(src, verb_offset) XOR edge`
**Not CSR** — represents edges as XOR bindings in hyperdimensional space.

### Model 6: Binding Matrix (ndarray/hpc/binding_matrix.rs)
**File**: `ndarray/src/hpc/binding_matrix.rs`

Dense 256³ = 16.7M point spatial field:
`matrix[i][j][k] = popcount(permute(X, i) XOR permute(Y, j) XOR permute(Z, k))`

**Purpose**: Reveals holographic sweet spots, discriminative hot spots, optimal verb offsets.
**Performance**: ~0.5s on AVX-512 for full 256³ computation.

## 2. KuzuDB Reference Architecture

KuzuDB is empty on our branch (bare `.git`), but the upstream C++ engine defines the reference:

```
KuzuDB Adjacency Model:
  - CSR adjacency lists grouped by (src_label, edge_label, dst_label)
  - One AdjacencyStore per relationship type (we adopted this)
  - Column-based storage with dictionary compression on node/rel tables
  - Vectorized scan with SIMD filter pushdown
  - Morsel-driven parallelism (we adopted MorselExec strategy)
  - WCO (Worst-Case Optimal) join via sorted intersection
```

**What we took from KuzuDB**:
1. Per-relationship-type CSR → `AdjacencyStore.rel_type`
2. Columnar edge properties → `EdgeProperties` (not row-major)
3. Sorted adjacency lists for intersection → `intersect_adjacent()`
4. Morsel execution → `strategy/morsel_exec.rs`
5. DP join enumeration → `strategy/dp_join.rs`
6. Histogram cost model → `strategy/histogram_cost.rs`

**What KuzuDB doesn't have that we do**:
1. Semiring algebra on adjacency (graph algorithms AS matrix operations)
2. Hyperdimensional vector edge weights (16Kbit fingerprints)
3. Progressive codec stack (scent → palette → base17 → full)
4. Thinking-style-modulated query planning
5. CAM-PQ compressed vector search integrated with graph traversal

## 3. The BLAS × PQ_CAM × BGZ17 Synergy

### The Key Insight: PQ Lookup Tables ARE Matrix Multiplication

```
CAM-PQ distance computation:
  d(query, candidate) = Σ_{s=0}^{5} table[s][code[s]]

This is a DOT PRODUCT:
  table_row = [table[0][c0], table[1][c1], ..., table[5][c5]]
  ones = [1, 1, 1, 1, 1, 1]
  d = dot(table_row, ones) = Σ table[s][code_s]

For N candidates:
  D = [table[0][codes[0:N,0]], table[1][codes[0:N,1]], ...] × ones
  = N×6 matrix × 6×1 vector
  = GEMV operation
```

**Therefore**: MKL `cblas_sgemv` on the precomputed distance table against batch-encoded candidates = **vectorized batch ADC**.

### Concrete Implementation Path

```
Step 1: Precompute distance table (6 × 256 floats = 6KB)
  → ndarray/hpc/cam_pq.rs: CamCodebook::precompute_tables()
  → Fits entirely in L1 cache

Step 2: Gather CAM codes for N candidates into 6 columns
  codes_col_0 = [cam[0][0], cam[1][0], ..., cam[N][0]]  // HEEL column
  codes_col_1 = [cam[0][1], cam[1][1], ..., cam[N][1]]  // BRANCH column
  ...

Step 3: Lookup → N×6 float matrix via VPGATHERDD (AVX-512)
  for s in 0..6:
    distances[0:N, s] = table[s][codes_col_s[0:N]]

Step 4: Sum across 6 subspaces → N distances
  d[0:N] = Σ_{s=0}^{5} distances[0:N, s]
  → cblas_sgemv if using MKL
  → SIMD horizontal add if using native backend

Step 5: Top-K via partial sort
  → spatial_hash.rs ring expansion or simple partial_sort
```

### BGZ17 × BLAS: Palette Distance as GEMM

```
Palette distance computation:
  For k=256 archetypes, each with Base17 (i16[17]):
  Distance matrix D[i][j] = L1(base17[i], base17[j])
  = 256 × 256 × 17 operations
  = can be expressed as |A - B|₁ where A, B are 256×17 matrices

With MKL:
  1. Convert i16[17] to f32[17] (one-time, 256 × 17 × 4 = ~17KB)
  2. D = |A - B|₁ via BLAS Level 2 operations
  3. Cache the 256×256 distance table (262K entries)
```

### Neighborhood × BLAS: Adjacency as Sparse GEMM

```
Graph traversal as sparse matrix-vector multiply:
  y = A × x  where A = adjacency matrix, x = query fingerprint

For ScentCsr (u8 weights):
  y[i] = Σ scent[i,j] × x[j]  for j ∈ neighbors(i)
  = SpMV (sparse matrix-vector multiply)
  → MKL mkl_sparse_s_mv() or native CSR SpMV

For GrBMatrix (BitVec values):
  y[i] = ⊕_{j∈adj(i)} (A[i,j] ⊗ x[j])
  = Semiring SpMV
  → Cannot use BLAS directly (non-numeric semiring)
  → But the popcount/XOR operations ARE SIMD-vectorizable
```

## 4. Synergy Matrix

| Feature | AdjacencyStore | GrBMatrix | ScentCsr | PaletteCsr | KuzuDB |
|---------|---------------|-----------|----------|------------|--------|
| CSR forward | ✓ | ✓ | ✓ | ✓ (archetype-level) | ✓ |
| CSC backward | ✓ | ✓ | ✗ | ✗ | ✓ |
| HyperCSR | ✗ | ✓ | ✗ | ✗ | ✗ |
| Per-rel-type | ✓ | ✓ (TypedGraph) | ✗ | ✗ | ✓ |
| Edge properties | Columnar | None (weight in T) | u8 scent | Palette index | Columnar |
| NARS truth | ✓ (columns) | ✗ | ✗ | ✗ | ✗ |
| Semiring mxm | ✗ | ✓ (7 semirings) | ✗ | ✓ (compose) | ✗ |
| BLAS acceleration | MKL SpMV | SIMD popcount | SIMD u8 ops | MKL L1 distance | Vectorized scan |
| Progressive codec | ✗ | Full only | Scent only | Palette only | ✗ |
| Morsel parallel | ✓ (via strategy) | ✗ | ✗ | ✗ | ✓ |
| Sorted intersection | ✓ | ✗ | ✗ | ✗ | ✓ |

## 5. The Unification Vision

```
                    ┌─────────────────────────────────────┐
                    │     AdjacencyView trait (contract)   │
                    │                                      │
                    │  fn adjacent(&self, node) -> &[u64]  │
                    │  fn edge_weight(&self, edge) -> W    │
                    │  fn semiring_mxv(&self, x, sr) -> y  │
                    └──────────┬──────────────────────────┘
                               │
          ┌────────────────────┼────────────────────┐
          │                    │                     │
  ┌───────▼──────┐    ┌───────▼──────┐     ┌───────▼──────┐
  │ AdjacencyStore│    │ GrBMatrix    │     │ PaletteCsr   │
  │ W = EdgeProps │    │ W = BitVec   │     │ W = u8 index │
  │ KuzuDB-style  │    │ BlasGraph    │     │ BGZ17 codec  │
  └──────────────┘    └──────────────┘     └──────────────┘

Each implementation:
  - Same traversal interface
  - Different edge weight types
  - Different semiring algebras
  - But composable: PaletteCsr pre-filters → GrBMatrix verifies → AdjacencyStore joins

The cascade:
  L0: ScentCsr (1 byte/edge, 94% ρ) → reject 95%
  L1: PaletteCsr (3 bytes/node, O(k²)) → narrow to archetype cluster
  L2: GrBMatrix (2KB/edge, full semiring) → precise algebra
  L3: AdjacencyStore (columnar properties) → join with metadata
```

## 6. Mathematical Synergy: Zeckendorf × Adjacency

The Zeckendorf encoding connects to adjacency through the **boolean lattice** structure:

```
ZeckF64 byte 0 bits:     SPO   _PO   S_O   SP_   __O   _P_   S__
                          b6    b5    b4    b3    b2    b1    b0

This IS an adjacency structure:
  - 7 "mask nodes" in a boolean lattice
  - Edges: SP_ → S__ (subset), SP_ → _P_ (subset), etc.
  - The lattice has 19 legal states out of 128 possible

Each scent byte encodes a node's position in this 7D boolean lattice.
ScentCsr stores these positions as edge weights.
Neighborhood search = finding nodes at similar lattice positions.
```

This means **the scent byte IS a compressed adjacency pattern** — not just a scalar weight,
but a 7-bit structural descriptor of how two nodes relate across S/P/O dimensions.
