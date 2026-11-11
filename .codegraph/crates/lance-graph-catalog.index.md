# lance-graph-catalog

crate: lance-graph-catalog
path: crates/lance-graph-catalog
workspace: member
digest: 9d0c34f70c28a465
evidence legend: ../index.md#evidence-legend

Generated; the code is the truth. See ../index.md.

## Depends on

- dep arrow-schema normal — crates/lance-graph-catalog/Cargo.toml:14 CONFIRMED
- dep async-trait normal — crates/lance-graph-catalog/Cargo.toml:15 CONFIRMED
- dep datafusion normal — crates/lance-graph-catalog/Cargo.toml:16 CONFIRMED
- dep lance-namespace normal — crates/lance-graph-catalog/Cargo.toml:17 CONFIRMED
- dep reqwest normal optional — crates/lance-graph-catalog/Cargo.toml:18 CONFIRMED
- dep serde normal — crates/lance-graph-catalog/Cargo.toml:19 CONFIRMED
- dep serde_json normal — crates/lance-graph-catalog/Cargo.toml:20 CONFIRMED
- dep snafu normal — crates/lance-graph-catalog/Cargo.toml:21 CONFIRMED
- dep tokio dev — crates/lance-graph-catalog/Cargo.toml:28 CONFIRMED
- dep wiremock dev — crates/lance-graph-catalog/Cargo.toml:29 CONFIRMED

## Depended on by

- rdep lance-graph normal — crates/lance-graph/Cargo.toml:49 CONFIRMED

## Public items

- struct CatalogInfo [pub] — crates/lance-graph-catalog/src/catalog_provider.rs:17 CONFIRMED
- struct SchemaInfo [pub] — crates/lance-graph-catalog/src/catalog_provider.rs:27 CONFIRMED
- struct ColumnInfo [pub] — crates/lance-graph-catalog/src/catalog_provider.rs:38 CONFIRMED
- enum DataSourceFormat [pub] — crates/lance-graph-catalog/src/catalog_provider.rs:52 CONFIRMED
- enum TableType [pub] — crates/lance-graph-catalog/src/catalog_provider.rs:65 CONFIRMED
- struct TableInfo [pub] — crates/lance-graph-catalog/src/catalog_provider.rs:72 CONFIRMED
- enum CatalogError [pub] — crates/lance-graph-catalog/src/catalog_provider.rs:88 CONFIRMED
- type CatalogResult [pub] — crates/lance-graph-catalog/src/catalog_provider.rs:135 CONFIRMED
- trait CatalogProvider [pub] — crates/lance-graph-catalog/src/catalog_provider.rs:151 CONFIRMED
- struct Connector [pub] — crates/lance-graph-catalog/src/connector.rs:51 CONFIRMED
- fn Connector::new [pub] — crates/lance-graph-catalog/src/connector.rs:59 CONFIRMED
- fn Connector::with_storage_options [pub] — crates/lance-graph-catalog/src/connector.rs:73 CONFIRMED
- fn Connector::storage_options [pub] — crates/lance-graph-catalog/src/connector.rs:79 CONFIRMED
- fn Connector::catalog [pub] — crates/lance-graph-catalog/src/connector.rs:84 CONFIRMED
- fn Connector::reader_for [pub] — crates/lance-graph-catalog/src/connector.rs:89 CONFIRMED
- fn Connector::readers [pub] — crates/lance-graph-catalog/src/connector.rs:97 CONFIRMED
- fn Connector::list_catalogs [pub] — crates/lance-graph-catalog/src/connector.rs:103 CONFIRMED
- fn Connector::get_catalog [pub] — crates/lance-graph-catalog/src/connector.rs:107 CONFIRMED
- fn Connector::list_schemas [pub] — crates/lance-graph-catalog/src/connector.rs:111 CONFIRMED
- fn Connector::get_schema [pub] — crates/lance-graph-catalog/src/connector.rs:115 CONFIRMED
- fn Connector::list_tables [pub] — crates/lance-graph-catalog/src/connector.rs:123 CONFIRMED
- fn Connector::get_table [pub] — crates/lance-graph-catalog/src/connector.rs:131 CONFIRMED
- fn Connector::register_schema [pub] — crates/lance-graph-catalog/src/connector.rs:161 CONFIRMED
- mod catalog_provider [pub] — crates/lance-graph-catalog/src/lib.rs:13 CONFIRMED
- mod connector [pub] — crates/lance-graph-catalog/src/lib.rs:14 CONFIRMED
- mod namespace [pub] — crates/lance-graph-catalog/src/lib.rs:15 CONFIRMED
- mod source_catalog [pub] — crates/lance-graph-catalog/src/lib.rs:16 CONFIRMED
- mod table_reader [pub] — crates/lance-graph-catalog/src/lib.rs:17 CONFIRMED
- mod type_mapping [pub] — crates/lance-graph-catalog/src/lib.rs:18 CONFIRMED
- mod unity_catalog [pub] cfg=feature="unity-catalog" — crates/lance-graph-catalog/src/lib.rs:20 CONFIRMED
- struct DirNamespace [pub] — crates/lance-graph-catalog/src/namespace/directory.rs:9 CONFIRMED
- fn DirNamespace::new [pub] — crates/lance-graph-catalog/src/namespace/directory.rs:17 CONFIRMED
- fn DirNamespace::base_uri [pub] — crates/lance-graph-catalog/src/namespace/directory.rs:26 CONFIRMED
- mod directory [pub] — crates/lance-graph-catalog/src/namespace/mod.rs:1 CONFIRMED
- trait GraphSourceCatalog [pub] — crates/lance-graph-catalog/src/source_catalog.rs:13 CONFIRMED
- struct InMemoryCatalog [pub] — crates/lance-graph-catalog/src/source_catalog.rs:19 CONFIRMED
- fn InMemoryCatalog::new [pub] — crates/lance-graph-catalog/src/source_catalog.rs:25 CONFIRMED
- fn InMemoryCatalog::with_node_source [pub] — crates/lance-graph-catalog/src/source_catalog.rs:32 CONFIRMED
- fn InMemoryCatalog::with_relationship_source [pub] — crates/lance-graph-catalog/src/source_catalog.rs:43 CONFIRMED
- struct SimpleTableSource [pub] — crates/lance-graph-catalog/src/source_catalog.rs:78 CONFIRMED
- fn SimpleTableSource::new [pub] — crates/lance-graph-catalog/src/source_catalog.rs:83 CONFIRMED
- fn SimpleTableSource::empty [pub] — crates/lance-graph-catalog/src/source_catalog.rs:86 CONFIRMED
- trait TableReader [pub] — crates/lance-graph-catalog/src/table_reader.rs:35 CONFIRMED
- fn uc_type_to_arrow [pub] — crates/lance-graph-catalog/src/type_mapping.rs:15 CONFIRMED
- fn columns_to_arrow_schema [pub] — crates/lance-graph-catalog/src/type_mapping.rs:65 CONFIRMED
- struct UnityCatalogConfig [pub] cfg=feature="unity-catalog" — crates/lance-graph-catalog/src/unity_catalog.rs:19 CONFIRMED
- fn UnityCatalogConfig::new [pub] cfg=feature="unity-catalog" — crates/lance-graph-catalog/src/unity_catalog.rs:29 CONFIRMED
- fn UnityCatalogConfig::with_token [pub] cfg=feature="unity-catalog" — crates/lance-graph-catalog/src/unity_catalog.rs:37 CONFIRMED
- fn UnityCatalogConfig::with_timeout [pub] cfg=feature="unity-catalog" — crates/lance-graph-catalog/src/unity_catalog.rs:42 CONFIRMED
- struct UnityCatalogProvider [pub] cfg=feature="unity-catalog" — crates/lance-graph-catalog/src/unity_catalog.rs:49 CONFIRMED
- fn UnityCatalogProvider::new [pub] cfg=feature="unity-catalog" — crates/lance-graph-catalog/src/unity_catalog.rs:55 CONFIRMED

## Imports from in-scope crates

- none

## Calls

CALL_GRAPH=ARTIFACT_MISSING
- calls ARTIFACT_MISSING — no ruff_charon artifact for this crate UNKNOWN

## Architecture

- none

Generated from Cargo manifests — extracted relationships only.

```mermaid
flowchart LR
  n29["lance-graph"]
  n35["lance-graph-catalog"]
  n29 --> n35
```
