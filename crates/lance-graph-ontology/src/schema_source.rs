//! `SchemaSource` trait — the abstract producer of `MappingProposal`s.
//!
//! Implementations:
//!
//! - [`crate::ttl_parse::TtlSource`] (this session) — parses OGIT-shaped
//!   TTL files.
//! - MySQL / MSSQL scanners (future session) — introspect a relational
//!   schema and emit one proposal per discovered table / column.
//! - Customer admin form (future UX layer) — emits one proposal per row
//!   when a customer extends their tenant ontology at runtime.
//!
//! Every implementation funnels through the same registry append path so
//! the audit story is uniform: every dictionary row carries a
//! `created_by` + `source_uri` + `confidence` and every change is
//! immortalised in the Lance time-travel history.

use crate::error::Result;
use crate::proposal::MappingProposal;
use crate::semantic_types::SemanticTypeMap;

/// A source of `MappingProposal`s.
pub trait SchemaSource {
    /// Produce all proposals this source has to offer. Called eagerly; the
    /// returned `Vec` is appended to the registry as a batch.
    fn proposals(&self, sem: &SemanticTypeMap) -> Result<Vec<MappingProposal>>;

    /// Stable identifier for this source. Used in the dictionary's
    /// `created_by` column for audit. Examples: `"ogit_hydrator_v1"`,
    /// `"mysql_scanner_v1"`, `"admin:user@example.com"`.
    fn created_by(&self) -> String;
}
