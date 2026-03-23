# Scope E Findings — Wiring lance-graph + ndarray + rs-graph-llm

> Generated: 2026-03-23
> Purpose: API surface inventory for workspace integration

---

## 1. lance-graph Query API

### Public Exports (lib.rs)

```rust
pub use query::{CypherQuery, ExecutionStrategy};
pub use config::{GraphConfig, NodeMapping, RelationshipMapping};
pub use error::{GraphError, Result};
pub use sql_query::SqlQuery;
pub use lance_vector_search::VectorSearch;
// Catalog re-exports: GraphSourceCatalog, InMemoryCatalog, DirNamespace, etc.
```

### CypherQuery Struct

```rust
pub struct CypherQuery {
    query_text: String,
    ast: CypherAST,
    config: Option<GraphConfig>,
    parameters: HashMap<String, serde_json::Value>,
}
```

### CypherQuery Methods (builder + execution)

| Method | Signature | Notes |
|--------|-----------|-------|
| `new` | `pub fn new(query: &str) -> Result<Self>` | Parses Cypher via nom |
| `with_config` | `pub fn with_config(self, config: GraphConfig) -> Self` | Required before execute |
| `with_parameter` | `pub fn with_parameter<K, V>(self, key: K, value: V) -> Self` | Lowercase-normalized |
| `with_parameters` | `pub fn with_parameters(self, params: HashMap<...>) -> Self` | Bulk params |
| `query_text` | `pub fn query_text(&self) -> &str` | |
| `ast` | `pub fn ast(&self) -> &CypherAST` | |
| `config` | `pub fn config() -> Option<&GraphConfig>` | |
| `parameters` | `pub fn parameters() -> &HashMap<...>` | |

### Execution Methods

```rust
// In-memory datasets (HashMap<String, RecordBatch>)
pub async fn execute(
    &self,
    datasets: HashMap<String, RecordBatch>,
    strategy: Option<ExecutionStrategy>,
) -> Result<RecordBatch>

// Namespace-backed (Lance directory)
pub async fn execute_with_namespace(
    &self,
    namespace: DirNamespace,
    strategy: Option<ExecutionStrategy>,
) -> Result<RecordBatch>

// Shared namespace (Arc)
pub async fn execute_with_namespace_arc(
    &self,
    namespace: Arc<DirNamespace>,
    strategy: Option<ExecutionStrategy>,
) -> Result<RecordBatch>

// Explain plan
pub async fn explain(
    &self,
    datasets: HashMap<String, RecordBatch>,
) -> Result<String>

// Convert to SQL
pub async fn to_sql(
    &self,
    datasets: HashMap<String, RecordBatch>,
) -> Result<String>
```

**All execute methods return `Result<RecordBatch>`** (Arrow RecordBatch).

### ExecutionStrategy

```rust
pub enum ExecutionStrategy {
    DataFusion,   // default — fully implemented
    LanceNative,  // not yet implemented (returns UnsupportedFeature)
    BlasGraph,    // not yet implemented (returns UnsupportedFeature)
}
```

### GraphConfig Builder

```rust
let config = GraphConfig::builder()
    .with_node_label("Person", "person_id")
    .with_relationship("KNOWS", "src_person_id", "dst_person_id")
    .build()?;
```

---

## 2. ndarray SIMD API

### Version
`ndarray = "0.17.2"` (forked, local at `/home/user/ndarray`)

### Backend Dispatch (backend/mod.rs)

Free functions — no struct, no trait needed by callers (unless generic):

```rust
// BLAS Level 1
pub fn dot_f32(x: &[f32], y: &[f32]) -> f32;
pub fn dot_f64(x: &[f64], y: &[f64]) -> f64;
pub fn axpy_f32(alpha: f32, x: &[f32], y: &mut [f32]);
pub fn nrm2_f32(x: &[f32]) -> f32;   // L2 norm
pub fn asum_f32(x: &[f32]) -> f32;   // L1 norm
pub fn scal_f32(alpha: f32, x: &mut [f32]);

// BLAS Level 3
pub fn gemm_f32(m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
pub fn gemv_f32(m, n, alpha, a, lda, x, beta, y);
// (same for _f64 variants)
```

Dispatch priority: `intel-mkl` > `openblas` > `native` (pure Rust SIMD, AVX-512/AVX2/scalar).

### BlasFloat Trait (for generic code)

```rust
pub trait BlasFloat: num_traits::Float + Default + Send + Sync + 'static {
    fn backend_dot(x: &[Self], y: &[Self]) -> Self;
    fn backend_axpy(alpha: Self, x: &[Self], y: &mut [Self]);
    fn backend_scal(alpha: Self, x: &mut [Self]);
    fn backend_nrm2(x: &[Self]) -> Self;
    fn backend_asum(x: &[Self]) -> Self;
    fn backend_gemm(m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
    fn backend_gemv(m, n, alpha, a, lda, x, beta, y);
}
impl BlasFloat for f32 { ... }
impl BlasFloat for f64 { ... }
```

### HPC Extensions (hpc/mod.rs)

55 modules including:
- `blas_level1`, `blas_level2`, `blas_level3` — extension traits on Array
- `statistics` — median, var, std, percentile, top_k
- `activations` — sigmoid, softmax, log_softmax
- `hdc` — hyperdimensional computing ops
- `arrow_bridge` — Arrow interop
- `fft`, `vml`, `lapack`, `quantized` — numerical kernels
- `clam`, `clam_search`, `clam_compress` — CLAM tree (used by lance-graph neighborhood)
- `fingerprint`, `plane`, `seal`, `node`, `cascade` — cognitive layer types

### What Query Engines Need from ndarray

1. **dot_f32 / dot_f64** — vector similarity (cosine sim = dot / (nrm2 * nrm2))
2. **nrm2_f32** — L2 norm for normalization
3. **gemv_f32** — matrix-vector multiply for graph algebra
4. **gemm_f32** — matrix-matrix multiply for BlasGraph semiring mxv/mxm
5. **arrow_bridge** — zero-copy Array <-> RecordBatch conversion
6. **hdc** ops — for hyperdimensional vector binding in BlasGraph

---

## 3. rs-graph-llm Status

### Workspace Structure

```toml
[workspace]
members = [
    "graph-flow",                  # Core framework (v0.4.0, edition 2024)
    "graph-flow-server",           # Axum HTTP server
    "insurance-claims-service",    # Example: insurance workflow
    "examples",                    # Example binaries
    "recommendation-service",      # Example: RAG recommendation
    "medical-document-service",    # Example: medical docs
]
```

### Build Status

**FAILS** — `ort-sys` SSL certificate error (environment issue, not code bug).

### graph-flow Dependencies (key ones)

| Dep | Version | Notes |
|-----|---------|-------|
| tokio | 1.40 | full features |
| lance | 2 | optional, behind `lance-store` feature |
| arrow | 57 | optional, behind `lance-store` feature |
| rig-core | 0.19.0 | optional, behind `rig` feature |
| sqlx | 0.8.6 | postgres + sqlite |
| serde_json | 1.0 | |

### Feature Flags

```toml
[features]
default = ["mcp"]
rig = ["dep:rig-core"]
mcp = ["dep:reqwest"]
lance-store = ["dep:lance", "dep:arrow"]
tiered = ["lance-store"]
full = ["rig", "mcp", "tiered"]
```

### Key Issue: AriGraph Integration

The `graph-flow-memory` crate (AriGraph schema port) is **planned but zero code written**. This is the integration point where lance-graph Cypher queries would be called from graph-flow.

---

## 4. Workspace Design

### Proposed Unified Workspace

There are two options:

**Option A: rs-graph-llm becomes the top-level workspace (recommended)**

```toml
[workspace]
members = [
    "graph-flow",
    "graph-flow-server",
    "graph-flow-memory",           # NEW: AriGraph integration
    "insurance-claims-service",
    "examples",
    "recommendation-service",
    "medical-document-service",
]
resolver = "2"

[workspace.dependencies]
# Aligned versions across all crates
arrow = { version = "57", features = ["prettyprint"] }
datafusion = { version = "51" }
lance = { version = "2" }
lance-graph = { path = "../lance-graph/crates/lance-graph" }
ndarray = { path = "../ndarray" }
tokio = { version = "1.40", features = ["full"] }
serde = { version = "1.0", features = ["derive"] }
serde_json = "1.0"
```

**Option B: Mono-workspace (all repos merged)**

Not recommended — lance-graph and ndarray have distinct release cycles.

### Version Alignment Requirements

| Dependency | lance-graph | rs-graph-llm | Must Align? |
|-----------|-------------|--------------|-------------|
| arrow | 57 | 57 | YES — same major required |
| datafusion | 51 | (not yet) | YES when added |
| lance | 2 | 2 | YES — same major |
| serde | 1.0 | 1.0 | OK |
| serde_json | 1.0 | 1.0 | OK |
| tokio | (dev only) | 1.40 | OK |
| nom | 7.1 | (not used) | N/A |
| ndarray | (not yet) | (not yet) | Future — path dep |

**Critical**: arrow 57 and datafusion 51 are tightly coupled (DataFusion 51 depends on Arrow 57). Both repos already use arrow 57 and lance 2, so they are aligned.

### Rust Edition Note

lance-graph uses `edition = "2021"`, rs-graph-llm/graph-flow uses `edition = "2024"`. This is fine — editions are per-crate, not per-workspace. But `edition = "2024"` requires Rust 1.85+.

---

## 5. lance-graph Local Cypher Path

### Minimal Integration Flow

```rust
use lance_graph::{CypherQuery, GraphConfig, Result};
use arrow_array::RecordBatch;
use std::collections::HashMap;

// Step 1: Configure the graph schema mapping
let config = GraphConfig::builder()
    .with_node_label("Person", "person_id")
    .with_relationship("KNOWS", "src_person_id", "dst_person_id")
    .build()?;

// Step 2: Parse the Cypher query
let query = CypherQuery::new("MATCH (p:Person)-[:KNOWS]->(f) RETURN p.name, f.name")?
    .with_config(config);

// Step 3a: Execute with in-memory RecordBatches
let mut datasets: HashMap<String, RecordBatch> = HashMap::new();
datasets.insert("Person".to_string(), person_batch);
datasets.insert("KNOWS".to_string(), knows_batch);
let result: RecordBatch = query.execute(datasets, None).await?;

// Step 3b: OR execute with Lance namespace (directory of .lance files)
let namespace = DirNamespace::new("/path/to/lance/data");
let result: RecordBatch = query.execute_with_namespace(namespace, None).await?;
```

### What a Local Executor Needs to Call

1. **`CypherQuery::new(cypher_str)`** — parse Cypher into AST
2. **`.with_config(config)`** — attach schema mapping (node labels, relationship types, key columns)
3. **`.execute(datasets, None)`** — run against in-memory RecordBatches; returns `Result<RecordBatch>`
4. **`.execute_with_namespace(ns, None)`** — run against Lance directory; returns `Result<RecordBatch>`

### Internal Pipeline

```
CypherQuery::new()          -> parse_cypher_query() -> CypherAST
  .execute()                -> match ExecutionStrategy
    DataFusion path:        -> build_catalog_and_context_from_datasets()
                            -> create_logical_plans() -> (GraphLogicalPlan, DataFusion LogicalPlan)
                            -> ctx.execute_logical_plan() -> DataFrame
                            -> df.collect() -> Vec<RecordBatch>
                            -> concat_batches() -> single RecordBatch
```

### For graph-flow-memory Integration

The `graph-flow-memory` crate would:
1. Hold a `GraphConfig` describing the AriGraph schema (entities, relations, observations)
2. Maintain a `DirNamespace` pointing to Lance storage
3. Expose methods like `query_graph(cypher: &str) -> Result<RecordBatch>` that wrap `CypherQuery`
4. Convert RecordBatch results back into graph-flow `Context` values

### BlasGraph Path (Future)

When `ExecutionStrategy::BlasGraph` is implemented, it will bypass DataFusion entirely and use:
- `GrBMatrix` / `GrBVector` for sparse graph representation
- Semiring operations (XOR_BUNDLE, HAMMING_MIN, etc.) for path queries
- ndarray `backend::gemm_f32` / `gemv_f32` for the underlying matrix multiply
- This is Phase 3/4 work in lance-graph — not available yet

---

## Summary: What's Needed to Wire Together

| Task | Blocking? | Notes |
|------|-----------|-------|
| Create `graph-flow-memory` crate in rs-graph-llm | YES | AriGraph schema + CypherQuery wrapper |
| Add `lance-graph = { path = "..." }` dep | YES | Path dep to local lance-graph |
| Align arrow = 57, lance = 2 | DONE | Already aligned |
| Add ndarray path dep to lance-graph | NO | Phase 3 — blasgraph/ndarray_bridge exists but not wired |
| Fix ort-sys SSL build error | YES | rs-graph-llm cannot build without this |
| Implement BlasGraph execution strategy | NO | Phase 3/4 — DataFusion path works now |
