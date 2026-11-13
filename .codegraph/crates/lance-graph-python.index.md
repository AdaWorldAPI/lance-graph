# lance-graph-python

crate: lance-graph-python
path: crates/lance-graph-python
workspace: excluded
digest: 09be59d43fefa7f5
evidence legend: ../index.md#evidence-legend

Generated; the code is the truth. See ../index.md.

## Depends on

- dep lance-graph normal — crates/lance-graph-python/Cargo.toml:20 CONFIRMED
- dep arrow normal — crates/lance-graph-python/Cargo.toml:14 CONFIRMED
- dep arrow-array normal — crates/lance-graph-python/Cargo.toml:15 CONFIRMED
- dep arrow-ipc normal — crates/lance-graph-python/Cargo.toml:17 CONFIRMED
- dep arrow-schema normal — crates/lance-graph-python/Cargo.toml:16 CONFIRMED
- dep datafusion normal — crates/lance-graph-python/Cargo.toml:18 CONFIRMED
- dep futures normal — crates/lance-graph-python/Cargo.toml:19 CONFIRMED
- dep pyo3 normal — crates/lance-graph-python/Cargo.toml:23 CONFIRMED
- dep serde normal — crates/lance-graph-python/Cargo.toml:21 CONFIRMED
- dep serde_json normal — crates/lance-graph-python/Cargo.toml:22 CONFIRMED
- dep tokio normal — crates/lance-graph-python/Cargo.toml:24 CONFIRMED

## Depended on by


## Public items

- struct PyCatalogInfo [pub] — crates/lance-graph-python/src/catalog.rs:25 CONFIRMED
- struct PySchemaInfo [pub] — crates/lance-graph-python/src/catalog.rs:50 CONFIRMED
- struct PyTableInfo [pub] — crates/lance-graph-python/src/catalog.rs:83 CONFIRMED
- struct PyUnityCatalog [pub] — crates/lance-graph-python/src/catalog.rs:180 CONFIRMED
- fn register_catalog_module [pub] — crates/lance-graph-python/src/catalog.rs:367 CONFIRMED
- const SIGNAL_CHECK_INTERVAL [pub] — crates/lance-graph-python/src/executor.rs:20 CONFIRMED
- struct BackgroundExecutor [pub] — crates/lance-graph-python/src/executor.rs:29 CONFIRMED
- fn BackgroundExecutor::get_runtime_handle [pub] — crates/lance-graph-python/src/executor.rs:35 CONFIRMED
- fn BackgroundExecutor::new [pub] — crates/lance-graph-python/src/executor.rs:39 CONFIRMED
- fn BackgroundExecutor::spawn [pub] — crates/lance-graph-python/src/executor.rs:55 CONFIRMED
- fn BackgroundExecutor::spawn_background [pub] — crates/lance-graph-python/src/executor.rs:110 CONFIRMED
- fn BackgroundExecutor::block_on [pub] — crates/lance-graph-python/src/executor.rs:136 CONFIRMED
- enum ExecutionStrategy [pub] — crates/lance-graph-python/src/graph.rs:46 CONFIRMED
- enum SqlDialect [pub] — crates/lance-graph-python/src/graph.rs:65 CONFIRMED
- enum DistanceMetric [pub] — crates/lance-graph-python/src/graph.rs:93 CONFIRMED
- struct VectorSearch [pub] — crates/lance-graph-python/src/graph.rs:133 CONFIRMED
- fn graph_error_to_pyerr [pub(crate)] — crates/lance-graph-python/src/graph.rs:310 CONFIRMED
- struct GraphConfig [pub] — crates/lance-graph-python/src/graph.rs:326 CONFIRMED
- struct GraphConfigBuilder [pub] — crates/lance-graph-python/src/graph.rs:362 CONFIRMED
- struct CypherQuery [pub] — crates/lance-graph-python/src/graph.rs:438 CONFIRMED
- struct CypherEngine [pub] — crates/lance-graph-python/src/graph.rs:1167 CONFIRMED
- struct SqlQuery [pub] — crates/lance-graph-python/src/graph.rs:1378 CONFIRMED
- struct SqlEngine [pub] — crates/lance-graph-python/src/graph.rs:1485 CONFIRMED
- fn SqlEngine::from_context [pub(crate)] — crates/lance-graph-python/src/graph.rs:1577 CONFIRMED
- fn register_graph_module [pub] — crates/lance-graph-python/src/graph.rs:1585 CONFIRMED
- static RT [pub(crate)] — crates/lance-graph-python/src/lib.rs:10 CONFIRMED
- struct PyDirNamespace [pub] — crates/lance-graph-python/src/namespace.rs:7 CONFIRMED

## Imports from in-scope crates

### lance-graph
- use lance_graph::sql_catalog::build_context_from_connector — crates/lance-graph-python/src/catalog.rs:9 CONFIRMED
- use lance_graph::table_readers::default_table_readers — crates/lance-graph-python/src/catalog.rs:10 CONFIRMED
- use lance_graph::CatalogInfo — crates/lance-graph-python/src/catalog.rs:12 CONFIRMED
- use lance_graph::Connector — crates/lance-graph-python/src/catalog.rs:12 CONFIRMED
- use lance_graph::SchemaInfo — crates/lance-graph-python/src/catalog.rs:12 CONFIRMED
- use lance_graph::TableInfo — crates/lance-graph-python/src/catalog.rs:12 CONFIRMED
- use lance_graph::UnityCatalogConfig — crates/lance-graph-python/src/catalog.rs:12 CONFIRMED
- use lance_graph::UnityCatalogProvider — crates/lance-graph-python/src/catalog.rs:12 CONFIRMED
- use lance_graph::ast::DistanceMetric as RustDistanceMetric — crates/lance-graph-python/src/graph.rs:27 CONFIRMED
- use lance_graph::ast::GraphPattern — crates/lance-graph-python/src/graph.rs:27 CONFIRMED
- use lance_graph::ast::ReadingClause — crates/lance-graph-python/src/graph.rs:27 CONFIRMED
- use lance_graph::CypherQuery as RustCypherQuery — crates/lance-graph-python/src/graph.rs:28 CONFIRMED
- use lance_graph::ExecutionStrategy as RustExecutionStrategy — crates/lance-graph-python/src/graph.rs:28 CONFIRMED
- use lance_graph::GraphConfig as RustGraphConfig — crates/lance-graph-python/src/graph.rs:29 CONFIRMED
- use lance_graph::GraphError as RustGraphError — crates/lance-graph-python/src/graph.rs:29 CONFIRMED
- use lance_graph::InMemoryCatalog — crates/lance-graph-python/src/graph.rs:29 CONFIRMED
- use lance_graph::SqlDialect as RustSqlDialect — crates/lance-graph-python/src/graph.rs:30 CONFIRMED
- use lance_graph::SqlQuery as RustSqlQuery — crates/lance-graph-python/src/graph.rs:30 CONFIRMED
- use lance_graph::VectorSearch as RustVectorSearch — crates/lance-graph-python/src/graph.rs:30 CONFIRMED
- use lance_graph::DirNamespace — crates/lance-graph-python/src/namespace.rs:3 CONFIRMED

## Calls

CALL_GRAPH=ARTIFACT_MISSING
- calls ARTIFACT_MISSING — no ruff_charon artifact for this crate UNKNOWN

## Architecture

- none

Generated from Cargo manifests — extracted relationships only.

```mermaid
flowchart LR
  n34["lance-graph"]
  n62["lance-graph-python"]
  n62 --> n34
```
