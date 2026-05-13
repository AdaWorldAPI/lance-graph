# FalkorDB Architecture Map

Mapping FalkorDB concepts to our lance-graph equivalents.

## Component Map

| FalkorDB Component | Our Equivalent | Status |
|--------------------|---------------|--------|
| `GrB_Matrix` (CSC, scalar values) | `GrBMatrix` (CSR+CSC, 16Kbit BitVec values) | Session A |
| `GrB_mxm` (SuiteSparse C FFI) | `GrBMatrix::mxm` (pure Rust) | Same algebra |
| ~960 scalar semirings | 7 HDR semirings (`HdrSemiring`) + `FalkorSemiring` mapping | Session A+B |
| Query parser (C) | `parser.rs` (Rust, nom) | Already have |
| Query planner | `blasgraph_planner.rs` | Session A |
| Per-reltype adjacency matrices | `TypedGraph.relations` | Session A |
| Label masks (boolean per-node) | `TypedGraph.labels` | Session A |
| Property storage (Redis hashes) | `MetadataStore` -> `RecordBatch` | Already have |
| Redis server protocol | Not needed (library mode) | N/A |

## Our Advantages (Not in FalkorDB)

| Feature | Description |
|---------|-------------|
| `TruthGate` | Edge confidence filtering via NARS-style truth values. FalkorDB has no concept of edge confidence. |
| Palette compression (bgz17) | Base17 palette encoding for 57x compression of BitVec edges. |
| HHTL progressive search | Heel-Hip-Twig-Leaf cascade for multi-resolution graph search. |
| Container 256-word structure | 2KB containers with reserved words for inline graph storage (W16-31 edges, W112-125 bgz17). |
| HDR XorBundle semiring | Semantic path composition via XOR bind + majority-vote bundle. No scalar equivalent exists. |

## Semiring Mapping

| FalkorDB Semiring | GrB Name | Our Equivalent | Notes |
|-------------------|----------|---------------|-------|
| Shortest path | `GrB_MIN_PLUS_INT64` | `HdrSemiring::HammingMin` | compose=XOR (distance accumulates), reduce=min-popcount |
| Reachability | `GrB_LOR_LAND_BOOL` | `HdrSemiring::Boolean` | compose=AND, reduce=OR. Identical algebra. |
| Path composition | N/A | `HdrSemiring::XorBundle` | Novel: XOR bind + majority bundle. Not expressible in FalkorDB. |

## Key Architectural Differences

1. **Element type:** FalkorDB stores scalars (int64, float64, bool) per matrix entry.
   We store 16384-bit hyperdimensional vectors, enabling semantic operations
   that scalar graphs cannot express.

2. **Storage:** FalkorDB uses SuiteSparse CSC with scalar values (~16 bytes per entry).
   We use CSR+CSC with BitVec values (2KB per entry, or 105 bytes with bgz17 palette).

3. **Truth filtering:** FalkorDB filters by property predicates after traversal.
   We filter by `TruthGate` thresholds on edge confidence during traversal,
   enabling probabilistic graph queries.

4. **Query routing:** FalkorDB has one execution path (GraphBLAS C FFI).
   We have three backends: DataFusion (cold SQL), blasgraph (BitVec hot path),
   palette (bgz17 compressed hot path), with automatic routing via `FalkorCompat`.

## FalkorCompat Shim Status

The `FalkorCompat` shim currently exists with the **blasgraph backend only** (8 tests passing).
The DataFusion and palette backends are **not yet wired**. Routing logic dispatches to
blasgraph unconditionally; the planned 3-backend routing (DataFusion for cold SQL queries,
blasgraph for BitVec hot-path, palette for bgz17 compressed hot-path) is not yet implemented.
