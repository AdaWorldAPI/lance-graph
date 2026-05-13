# Blackboard — lance-graph

> Single-binary architecture: already Rust. Integrates directly as `crate::graph`.

## What Exists

A **Cypher-to-DataFusion transpiler** with three execution backends (DataFusion SQL, BlasGraph semiring, Lance native). Plus an SPO triple store and the bgz17 palette codec.

## Public API

```rust
// Main entry point
let query = CypherQuery::new("MATCH (a:Person)-[:KNOWS]->(b) RETURN a.name, b.name")?
    .with_config(config)
    .with_parameter("min_age", 30);

// Execute against in-memory data
let result: RecordBatch = query.execute(datasets, None).await?;

// Execute against Lance directory
let result = query.execute_with_namespace(namespace, None).await?;

// Explain plan
let plan: String = query.explain(datasets).await?;

// Get SQL translation
let sql: String = query.to_sql(datasets).await?;
```

## Execution Backends

| Backend | Status | Path |
|---------|--------|------|
| DataFusion | Default, full features | Cypher → LogicalPlan → DataFusion SQL |
| BlasGraph | Phase 3 TODO | Cypher → TypedGraph.mxm(semiring) → BitVec |
| LanceNative | Phase 3 TODO | Direct Lance dataset access |

## Key Subsystems

### Cypher Parser (`parser.rs`, 66KB, nom-based)
MATCH, UNWIND, WHERE, WITH, RETURN, ORDER BY, LIMIT/SKIP. Variable-length paths (*min..max).

### BlasGraph (`graph/blasgraph/`)
- CSR/CSC/HyperCSR sparse matrix storage over 16,384-bit BitVec
- 7 semirings: XOR_BUNDLE, BIND_FIRST, HAMMING_MIN, SIMILARITY_MAX, RESONANCE, BOOLEAN, XOR_FIELD
- GrBMatrix with semiring-parameterized mxm

### SPO Triple Store (`graph/spo/`)
- Fingerprint-indexed S-P-O triples with NARS truth values
- Forward/reverse/relation queries ordered by Hamming distance
- Merkle integrity verification

### bgz17 Codec (`crates/bgz17/`, standalone, 121 tests)
- 4-layer compression: Scent → Palette → Base17 → Full planes
- Palette semiring for compressed SPO operations
- Zero external dependencies

## Dependencies

- Arrow 57, DataFusion 51, Lance 2
- nom 7.1 (parser), snafu 0.8 (errors)

## Integration Points for Binary

- SCOPE B's local Cypher path calls `CypherQuery::execute()`
- SCOPE A's runtime wraps lance-graph as a cell executor for `%%cypher` cells
- Arrow RecordBatch is the universal data exchange format

## Key Files

| File | LOC | Purpose |
|---|---|---|
| `src/lib.rs` | 76 | Public API |
| `src/query.rs` | 2200+ | CypherQuery execution |
| `src/parser.rs` | 1900+ | Cypher parser |
| `src/semantic.rs` | 2000+ | Validation |
| `src/logical_plan.rs` | 1700+ | Logical operators |
| `src/datafusion_planner/` | ~3000 | Physical planning |
| `graph/blasgraph/` | ~5000 | Semiring algebra |
| `graph/spo/` | ~1500 | Triple store |
