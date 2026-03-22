# SESSION A v3: blasgraph Storage + Cypher→Semiring Planner — ✅ COMPLETED

**Merged:** PR #29 (commit 678e355). All 5 deliverables shipped:
CscStorage, HyperCSR, TypedGraph + from_spo_store(), blasgraph_planner + TruthGate, SIMD Hamming.

---

## CONTEXT

**Repo:** `AdaWorldAPI/lance-graph` branch from `main`
**Crate:** `crates/lance-graph/src/graph/blasgraph/`
**Existing:** 7 semirings, grb_mxm, BFS/SSSP/PageRank, CSR only, BitVec entries
**SPO module:** spo/{truth,merkle,semiring,store,builder}.rs — DO NOT MODIFY

The container is 256 words × 64 bits = 2KB. NOT a flat fingerprint — structured
metadata with typed fields. The S/P/O planes (separate 2KB each) ARE flat. All
distance computation operates on planes, not on the container.

## READ FIRST

```bash
cat crates/lance-graph/src/graph/blasgraph/sparse.rs     # CooStorage + CsrStorage
cat crates/lance-graph/src/graph/blasgraph/matrix.rs     # GrBMatrix (CSR only)
cat crates/lance-graph/src/graph/blasgraph/semiring.rs   # 7 HdrSemirings
cat crates/lance-graph/src/graph/blasgraph/ops.rs        # grb_mxm, BFS, SSSP, PageRank
cat crates/lance-graph/src/graph/blasgraph/types.rs      # BitVec, HdrScalar, BinaryOp
cat crates/lance-graph/src/graph/spo/semiring.rs         # HammingMin: min-plus (KEEP)
cat crates/lance-graph/src/graph/spo/truth.rs            # TruthValue, TruthGate (KEEP)
cat crates/lance-graph/src/graph/spo/store.rs            # SpoStore queries (KEEP)
cat crates/lance-graph/src/logical_plan.rs               # LogicalOperator enum
cat crates/lance-graph/src/query.rs                      # Cypher execution (DataFusion)
```

## DELIVERABLE 1: CscStorage (sparse.rs)

Compressed Sparse Column. Zero-copy transpose for `A^T × B`.

```rust
pub struct CscStorage {
    pub nrows: usize, pub ncols: usize,
    pub col_ptr: Vec<usize>, pub row_idx: Vec<usize>, pub vals: Vec<BitVec>,
}
```

Required:
- `from_csr`, `from_coo`, `to_csr`
- `get(row, col)`, `column_iter(col)`
- Update `GrBMatrix` to hold `Option<CsrStorage>` + `Option<CscStorage>`
- `grb_transpose` flips preferred format (zero-copy when both present)

## DELIVERABLE 2: HyperCsrStorage (sparse.rs)

For power-law graphs where |edges| << |nodes|².

```rust
pub struct HyperCsrStorage {
    pub nrows: usize, pub ncols: usize,
    pub row_ids: Vec<usize>,  // which rows have entries (sorted)
    pub row_ptr: Vec<usize>, pub col_idx: Vec<usize>, pub vals: Vec<BitVec>,
}
```

Heuristic: use when `nnz / nrows < 0.1`.
Add `StorageFormat` enum: `{Csr, Csc, HyperCsr, HyperCsc, Bitmap, Dense}`.
Only implement Csr/Csc/HyperCsr. Reserve other variants.

## DELIVERABLE 3: TypedGraph (new: typed_graph.rs)

One matrix per relationship type, one diagonal mask per label.
Maps to FalkorDB schema AND to container W16-31 inline edges.

```rust
pub struct TypedGraph {
    pub relations: HashMap<String, GrBMatrix>,  // "KNOWS" → adjacency matrix
    pub labels: HashMap<String, Vec<bool>>,     // "Person" → diagonal mask
    pub node_count: usize,
}
```

Required:
- `add_relation(name, matrix)`, `add_label(name, node_ids)`
- `traverse(rel_type, semiring) -> GrBMatrix`
- `multi_hop(rel_types: &[&str], semiring) -> GrBMatrix`
- `masked_traverse(rel_type, label_mask, semiring)`
- `from_spo_store(store: &SpoStore) -> TypedGraph` — bridge from existing SPO

## DELIVERABLE 4: Blasgraph Physical Planner (new: blasgraph_planner.rs)

Second execution backend: Cypher → grb_mxm instead of Cypher → SQL.

```rust
pub fn compile_to_blasgraph(
    plan: &LogicalOperator,
    graph: &TypedGraph,
    semiring: &dyn Semiring,
) -> Result<GrBMatrix>
```

Maps `LogicalOperator::Expand` → `grb_mxm`, `ScanByLabel` → label mask,
`Filter` → predicate mask. Wire into `query.rs` as `ExecutionStrategy::BlasGraph`.

**CRITICAL:** TruthGate filtering happens AFTER matrix traversal, not during.
The planner produces candidate positions. Then:
```rust
let results = compile_to_blasgraph(&plan, &graph, &HdrSemiring::HammingMin)?;
let filtered = apply_truth_gate(results, gate, &spo_store);  // reads W4-7
```

## DELIVERABLE 5: SIMD for BitVec (types.rs)

AVX-512 VPOPCNTDQ for Hamming distance on full 2KB planes.
With fallback chain: avx512 → avx2 → scalar.
Use `std::arch` intrinsics, NOT external crate.

## TESTS

1. CSR ↔ CSC roundtrip preserves entries
2. grb_mxm with CSC matches CSR result
3. HyperCSR saves >90% memory for sparse graphs
4. TypedGraph 2-hop: KNOWS² matches manual multiply
5. Planner: `MATCH (a:Person)-[:KNOWS]->(b)` produces correct result
6. Planner + TruthGate: STRONG gate filters weak edges
7. SIMD Hamming matches scalar for random vectors

## OUTPUT

Branch: `feat/blasgraph-csc-planner`
Run: `cd crates/lance-graph && cargo test -- --nocapture`
