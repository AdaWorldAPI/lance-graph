// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Built-in [`TableReader`] implementations for common data formats.
//!
//! - [`ParquetTableReader`] — reads Parquet tables using DataFusion's built-in support.
//!
//! (A Delta Lake reader existed behind the removed `delta` feature — see
//! Cargo.toml's removal note, 2026-08-18.)

use std::collections::HashMap;
use std::sync::Arc;

use async_trait::async_trait;
use datafusion::execution::context::SessionContext;

use lance_graph_catalog::catalog_provider::{
    CatalogError, CatalogResult, DataSourceFormat, TableInfo,
};
use lance_graph_catalog::table_reader::TableReader;

/// Reads Parquet tables using DataFusion's built-in `register_parquet()`.
pub struct ParquetTableReader;

#[async_trait]
impl TableReader for ParquetTableReader {
    fn name(&self) -> &str {
        "parquet"
    }

    fn supported_formats(&self) -> &[DataSourceFormat] {
        &[DataSourceFormat::Parquet]
    }

    async fn register_table(
        &self,
        ctx: &SessionContext,
        table_name: &str,
        table_info: &TableInfo,
        _schema: arrow_schema::SchemaRef,
        _storage_options: &HashMap<String, String>,
    ) -> CatalogResult<()> {
        let location = table_info.storage_location.as_deref().ok_or_else(|| {
            CatalogError::Other(format!("Table '{}' has no storage_location", table_name))
        })?;

        ctx.register_parquet(
            table_name,
            location,
            datafusion::datasource::file_format::options::ParquetReadOptions::default(),
        )
        .await
        .map_err(|e| {
            CatalogError::Other(format!(
                "Failed to register Parquet table '{}' at '{}': {}",
                table_name, location, e
            ))
        })
    }
}

/// Returns the default set of table readers.
///
/// Includes Parquet support. (Delta Lake support was removed with the `delta`
/// feature, 2026-08-18 — see Cargo.toml's note; a `DataSourceFormat::Delta`
/// table simply has no registered reader, exactly as when the feature was off.)
pub fn default_table_readers() -> Vec<Arc<dyn TableReader>> {
    vec![Arc::new(ParquetTableReader)]
}
