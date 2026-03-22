# CLAUDE.md — lance-graph

> **Updated**: 2026-03-22
> **Role**: "The Face" in the four-repo Ada architecture
> **Status**: Phase 2 DONE, Phase 3-4 OPEN

---

## What This Is

Graph query engine for the Ada stack. Cypher parsing, semiring algebra (GraphBLAS-style),
SPO triple store with NARS truth values, DataFusion SQL planner, and Lance persistence.
This is the query surface that ladybug-rs (Brain) and rs-graph-llm (Orchestrator) talk through.

```
Four-repo architecture:
  ndarray        = The Foundation  (SIMD, GEMM, HPC compute)
  ladybug-rs     = The Brain      (BindSpace, SPO, server)
  lance-graph    = The Face       (Cypher/SQL query surface)  <-- THIS REPO
  rs-graph-llm   = The Orchestrator (graph-flow execution engine)

Dependency chain:  rs-graph-llm --> lance-graph --> ndarray
```

---

## Workspace Structure

```toml
[workspace]
members = [
    "crates/lance-graph",          # Core: Cypher parser, DataFusion planner, graph modules
    "crates/lance-graph-catalog",  # Catalog providers (Unity Catalog, in-memory, connectors)
    "crates/lance-graph-python",   # Python bindings (PyO3/maturin)
    "crates/lance-graph-benches",  # Benchmarks
]
exclude = [
    "crates/bgz17",                # Palette semiring codec (standalone, 0 deps, 121 tests)
    "crates/lance-graph-codec-research",  # ZeckBF17 codec research
]
```

### Crate Details

**lance-graph** (core) — `crates/lance-graph/`
- `lib.rs` — public API: CypherQuery, GraphConfig, SqlQuery, VectorSearch, catalog re-exports
- `parser.rs` + `ast.rs` — Cypher parser (nom-based) and AST representation
- `semantic.rs` — semantic analysis of parsed Cypher
- `logical_plan.rs` — logical plan from Cypher AST
- `datafusion_planner/` — Cypher-to-DataFusion SQL translation (scan, join, predicate pushdown, vector, UDF, cost estimation)
- `lance_native_planner.rs` — native Lance query planner
- `lance_vector_search.rs` — vector search integration
- `graph/spo/` — SPO triple store (store, builder, truth values, merkle, semirings)
- `graph/blasgraph/` — GraphBLAS-style sparse matrix ops (CSC/CSR, HHTL, cascade, typed graphs, semirings, SIMD Hamming)
- `graph/neighborhood/` — neighborhood search (CLAM tree, scope, zeckf64)
- `graph/metadata.rs` — metadata store (Cypher -> DataFusion metadata queries)
- `graph/falkor_compat.rs` — FalkorDB compatibility shim (blasgraph backend only currently)
- `graph/fingerprint.rs` — fingerprint encoding
- `sql_catalog.rs` + `sql_query.rs` — SQL query execution
- `table_readers.rs` — Parquet and Delta table readers

**bgz17** (standalone codec) — `crates/bgz17/`
- Zero external dependencies, compiles standalone
- Palette semiring + compose tables, PaletteMatrix mxm, PaletteCsr
- Base17 VSA ops (xor_bind, bundle, permute)
- SIMD batch_palette_distance
- TypedPaletteGraph, container pack/unpack (256-word layout)
- 121 tests passing

**lance-graph-catalog** — `crates/lance-graph-catalog/`
- CatalogProvider trait, Unity Catalog integration, connectors, type mapping

**lance-graph-python** — `crates/lance-graph-python/`
- PyO3 bindings: catalog, executor, graph, namespace

---

## Build Commands

```bash
# Check the main workspace (needs network for lance/arrow deps first time)
cargo check

# Run core lance-graph tests
cargo test -p lance-graph

# Run bgz17 tests (standalone, fast, no network)
cargo test --manifest-path crates/bgz17/Cargo.toml

# Run all workspace tests
cargo test

# Check without default features (skips unity-catalog, delta)
cargo check -p lance-graph --no-default-features

# Python bindings (requires maturin)
cd crates/lance-graph-python && maturin develop
```

---

## Current Status (2026-03-22)

### What's DONE
- **Phase 1** (blasgraph CSC/Planner): DONE — CscStorage, HyperCsrStorage, TypedGraph, blasgraph_planner, TruthGate, SIMD Hamming
- **Phase 2** (bgz17 container/semiring): DONE — 121 tests, PaletteSemiring, PaletteMatrix mxm, PaletteCsr, Base17 VSA, SIMD batch distance, TypedPaletteGraph, container pack/unpack
- **Core modules**: Cypher parser (44 tests), SPO store (30 tests), semirings (10 tests), plus blasgraph and neighborhood tests

### What's OPEN
- **Phase 3** (dual-path integration): NOT STARTED — bgz17-codec feature flag, plane_to_base17(), parallel_search() dual-path
- **Phase 4** (FalkorDB 3-backend routing): NOT STARTED — DataFusion + palette backends not wired into FalkorCompat
- bgz17 is still in workspace `exclude`, not wired into lance-graph

### Test Summary
- bgz17: **121 passing** (standalone)
- lance-graph core: **~84+ passing** (SPO 30 + Cypher 44 + semirings 10+)
- Total across workspace: **200+ passing**

---

## Key Dependencies

```toml
arrow = "57"
datafusion = "51"
lance = "2"
lance-linalg = "2"
nom = "7.1"           # Cypher parser
snafu = "0.8"         # Error handling
deltalake = "0.30"    # Optional, behind "delta" feature
```

Features: `default = ["unity-catalog", "delta"]`

---

## What NOT To Do

1. **DO NOT** move bgz17 into workspace members yet — Phase 3 gates are not met. It stays in `exclude` until the feature flag and bridge code are written.

2. **DO NOT** add direct ndarray dependency to lance-graph — the ndarray bridge (`graph/blasgraph/ndarray_bridge.rs`) exists but the actual ndarray crate integration is Phase 3 work.

3. **DO NOT** wire DataFusion or palette backends into FalkorCompat — only blasgraph is connected. Phase 4 adds the other two backends.

4. **DO NOT** duplicate semiring implementations — there are semirings in `graph/spo/semiring.rs`, `graph/blasgraph/semiring.rs`, AND `bgz17/palette_semiring.rs`. Each serves a different layer. Read all three before touching any.

5. **DO NOT** create new Cypher parser paths — there is ONE parser (`parser.rs` + `ast.rs`), one planner (`datafusion_planner/`), one blasgraph planner. Do not fork them.

6. **DO NOT** assume lance-graph compiles alone for all features — DataFusion integration tests may need network access or specific Lance dataset fixtures.

---

## Cross-Repo Dependencies

```
WHO DEPENDS ON US:
  ladybug-rs     — will wire SPO store + Cypher execution (Plateau 2, not started)
  rs-graph-llm   — uses lance-graph for graph persistence + query API

WHO WE DEPEND ON:
  ndarray        — future: SIMD compute, Fingerprint/Plane types (Phase 3 bridge)
  lance          — columnar storage, versioning, vector search
  datafusion     — SQL query engine
  arrow          — columnar memory format

SIBLING REPOS (same machine):
  /home/user/ndarray/       — The Foundation (HPC compute)
  /home/user/ladybug-rs/    — The Brain (BindSpace, server)
  /home/user/rs-graph-llm/  — The Orchestrator (graph-flow)
```

---

## Phase Roadmap Reference

```
Phase 1  blasgraph CSC + Planner          DONE
Phase 2  bgz17 container + semiring       DONE (121 tests)
Phase 3  dual-path integration            OPEN (bgz17 feature flag, plane_to_base17, parallel_search)
Phase 4  FalkorDB 3-backend routing       OPEN (DataFusion + palette backends)
```

See `.claude/phases/integration_phases.md` for gate criteria.
See `PROGRESS.md` for plateau-level tracking.
See `.claude/FINAL_STACK.md` for full ecosystem architecture.
