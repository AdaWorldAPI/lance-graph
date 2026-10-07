# lance-graph

[![Rust Tests](https://github.com/AdaWorldAPI/lance-graph/actions/workflows/rust-test.yml/badge.svg)](https://github.com/AdaWorldAPI/lance-graph/actions/workflows/rust-test.yml) [![Style Check](https://github.com/AdaWorldAPI/lance-graph/actions/workflows/style.yml/badge.svg)](https://github.com/AdaWorldAPI/lance-graph/actions/workflows/style.yml) [![Build](https://github.com/AdaWorldAPI/lance-graph/actions/workflows/build.yml/badge.svg)](https://github.com/AdaWorldAPI/lance-graph/actions/workflows/build.yml)

A Rust workspace for graph and columnar query execution over Arrow and Lance. It has:

- a Cypher and SQL engine on DataFusion;
- **Quack**, a typed columnar query surface that lowers to a single
  mask-program (`lance-graph-mask-risc`) over borrowed column lanes;
- a zero-dependency contract crate of shared types;
- palette/tensor codecs.

SIMD kernels come from the AdaWorldAPI
[`ndarray`](https://github.com/AdaWorldAPI/ndarray) fork, which is a required
sibling checkout.

This repository started as a fork of `lancedb/lance-graph`. The Cypher/SQL
engine, the Python bindings and the `knowledge_graph` package are that
inheritance; everything else listed below was built here.

## Query paths

Traced from merged source. These are separate paths with separate
intermediate representations; they do not share one IR.

```text
Cypher text ─ nom parser ─→ ast::CypherQuery ─ semantic (binds to GraphConfig)
              (parser.rs)    ─→ logical plan ─→ DataFusion LogicalPlan ─→ RecordBatch
SQL text ──── DataFusion SQL parser ────────────→ DataFusion plan ─────→ RecordBatch
Python ────── Arrow C data interface ──→ the two paths above

Rust builders: Filter / Agg / GroupAddr::{Local, Via, Pair}
  └→ quack::Query ─ quack::lower* ─→ mask_risc::Program { ops, terminal }
       ─ mask_risc::execute_into(program, borrowed lanes) ─→ scalar / group values
ReportPlan ─ Selection::lower ─→ quack Filter ─→ (same as above)
```

| Path | Entry point | Frontend | Intermediate | Executes on | Status |
|---|---|---|---|---|---|
| Cypher | `CypherQuery::execute` | `parser::parse_cypher_query` (nom) | owned AST → logical plan → DataFusion plan | DataFusion | default build |
| SQL | `SqlQuery`, `SqlEngine` | DataFusion SQL | DataFusion plan | DataFusion | default build |
| Python | `lance_graph.CypherQuery` / `SqlQuery` / `SqlEngine` | as above | as above | DataFusion | opt-in crate `lance-graph-python` |
| Quack | `quack::lower`, `lower_fused`, `lower_group_by*`; `bind::Draft::bind` for named columns | none (Rust builders) | `quack::Query` → `mask_risc::Program` | `mask_risc::execute_into` | workspace members |
| ReportPlan | `lance_graph_report::exec` | none | `Selection` → Quack `Filter` → `Program` | `mask_risc::execute_into` | workspace member |

What this means:

- **No production text frontend reaches Quack.** Quack is reached through its
  Rust builders (and through ReportPlan, SAP and dir-sim, which use them).
  The experimental crate `lance-graph-cypher-quack` lowers a small subset of
  bound Cypher (single-population aggregates, one-hop counts) to Quack and
  refuses everything else with a typed reason; nothing routes to it.
- **Cypher runs on DataFusion.** The `BlasGraph` and `LanceNative` execution
  strategies currently return an error rather than results.
- **No path serializes between parsing and execution.**
  - The Cypher path allocates an owned AST and copies its inputs into a
    DataFusion `MemTable`.
  - The Quack path builds a newly allocated `Program` and reads column lanes
    by borrow (`LaneRef<'a>`). It materializes row ids only when asked to
    (`materialize_rows`).
- **`lance-graph-planner` is planning-only.** It detects Cypher/GQL by keyword
  and parses Gremlin and SPARQL into its own `Arena<LogicalOp>`. It does not
  execute queries.

## Workspace

59 crate directories under `crates/`: 22 are workspace members and 37 are
explicitly excluded (built with `--manifest-path`). The workspace has a 23rd
member outside `crates/`, `tools/dto-class-check`. The parts needed to
understand the system:

| Crate | Role |
|---|---|
| `lance-graph` | Cypher parser, semantic analysis, DataFusion planner; BLASGraph semirings and SPO store as public modules |
| `lance-graph-catalog` | Catalog connectors (Unity Catalog); Parquet table reader. A table whose format has no reader (e.g. Delta) is rejected with `CatalogError::UnsupportedFormat` |
| `lance-graph-quack` | Typed filter / aggregate / group-by surface; `GroupAddr::{Local, Via, Pair}` foreign-key grouping |
| `lance-graph-mask-risc` | The mask-program IR (`Pred`, `MaskOp`, `Terminal`) and its evaluator, built on the `ndarray::simd` masking facade |
| `lance-graph-report` | `ReportPlan` → Quack lowering and execution |
| `lance-graph-contract` | Zero-dependency shared types and traits |
| `lance-graph-planner` | Multi-language planning IR (planning only, see above) |
| `bgz17`, `bgz-tensor` | Palette and tensor codecs (enabled in core by default features) |
| `causal-edge`, `deepnsm`, `deepnsm-v2` | Standalone excluded crates |

## Building

The workspace resolves path dependencies on a sibling checkout of the
AdaWorldAPI `ndarray` fork at `../ndarray`. Without it, `cargo` fails to load
the workspace. Some excluded crates (the OGAR-integrated ones) additionally
need `../OGAR`; CI checks out both.

```bash
git clone https://github.com/AdaWorldAPI/lance-graph
git clone https://github.com/AdaWorldAPI/ndarray      # sibling, required
cd lance-graph
cargo check -p lance-graph
cargo test  -p lance-graph-contract
```

- **Toolchain:** pinned by `rust-toolchain.toml`, currently Rust 1.98.1.
- **protoc:** the Lance dependency tree needs it (`protobuf-compiler`).
- **Excluded crates** are built through their own manifest, for example
  `cargo test --manifest-path crates/deepnsm-v2/Cargo.toml`.

## Python

The Python package (`python/`) wraps the Cypher and SQL paths. Build it with:

```bash
cd python
uv venv --python 3.11 .venv && source .venv/bin/activate
uv pip install 'maturin[patchelf]'
uv pip install -e '.[tests]'
maturin develop
pytest python/tests/ -v
```

```python
import pyarrow as pa
from lance_graph import CypherQuery, GraphConfig, SqlQuery

people = pa.table({"person_id": [1, 2, 3, 4],
                   "name": ["Alice", "Bob", "Carol", "David"],
                   "age": [28, 34, 29, 42]})

config = GraphConfig.builder().with_node_label("Person", "person_id").build()
q = CypherQuery("MATCH (p:Person) WHERE p.age > 30 RETURN p.name AS name").with_config(config)
print(q.execute({"Person": people}).to_pydict())   # {'name': ['Bob', 'David']}

print(SqlQuery("SELECT name FROM person WHERE age > 30")
      .execute({"person": people}).to_pydict())
```

**Unity Catalog.** `lance_graph.UnityCatalog(url)` browses catalogs, schemas
and tables. `create_sql_engine(catalog, schema)` registers the schema's
Parquet tables for SQL. If any table has a format with no reader, such as
Delta, it raises instead of registering an empty table.

**`knowledge_graph` CLI.**
- Run it with `uv run knowledge_graph --help`. Inherited from upstream, it
  initializes Lance-backed storage, runs Cypher, and extracts entities from
  text: with OpenAI by default, or `--extractor heuristic` offline.
- The FastAPI service runs with `python -m knowledge_graph.webservice`. It
  serves `/graph/health`, `/graph/query`, `/graph/datasets` and
  `/graph/schema`.
- See `python/README.md` for details.

## Benchmarks

`crates/lance-graph-benches/benches/graph_execution.rs` measures
`CypherQuery::execute` on in-memory input of 100, 10,000 and 1,000,000 rows
(node filter, one-hop, two-hop). The timed loop covers per-call planning,
table registration, DataFusion execution and result collection. Storage reads
happen once, during setup.

The setup asserts that every input holds exactly the requested number of rows,
and that each query's output row count is the one it must produce from all of
them. That is what makes the reported input-rows/second throughput
meaningful.

```bash
cargo bench -p lance-graph-benches --bench graph_execution
```

No reference numbers are published here yet. The table this README used to
carry came from a version of the benchmark that executed only the first
scanned batch at the 10K and 1M sizes, while reporting throughput for all N
rows.

## Relationship to ndarray

`ndarray` owns the hardware layer: SIMD, numerical kernels, conversion
exactness and microbenchmarks. Kernel performance figures belong in its
README, not here.

lance-graph consumes those kernels. Bit-vector Hamming distance is available
as the DataFusion UDF `hamming_distance`. It is not wired into Lance's ANN
search: requesting `DistanceMetric::Hamming` there returns an error.

## Upstream

DeepWiki's [`lancedb/lance-graph`](https://deepwiki.com/lancedb/lance-graph)
page documents the upstream project, i.e. the inherited Cypher/SQL engine
only, not this fork.
