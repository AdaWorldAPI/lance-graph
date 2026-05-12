//! `lance-graph-ontology` — the OGIT-canonical ontology spine for lance-graph
//! tenants.
//!
//! This crate consolidates per-tenant bridge multiplication into one shared
//! registry. OGIT becomes the canonical TTL ontology source; Lance becomes
//! the runtime dictionary cache; tenant bridges (woa, medcare, ogit) become
//! thin scoped views over the shared registry. TTL is the only ontology
//! exchange format.
//!
//! ## Surface
//!
//! - [`OntologyRegistry`] is the single registry. It hydrates from a TTL root
//!   directory (typically the AdaWorldAPI/OGIT fork checked out next to the
//!   workspace), holds an in-memory dictionary keyed by `(bridge_id,
//!   public_name)` and by OGIT URI, and (under the `lance-cache` feature)
//!   persists rows append-only to a Lance dataset.
//! - [`NamespaceBridge`] is the trait every tenant bridge implements. Default
//!   methods do the heavy lifting: a typical tenant bridge is ~15-20 lines
//!   that lock to one namespace and route resolution through the shared
//!   registry. See [`bridges::WoaBridge`], [`bridges::MedcareBridge`],
//!   [`bridges::OgitBridge`].
//! - [`MappingProposal`] is the producer-side DTO. TTL hydration emits
//!   proposals; schema scanners (MySQL/MSSQL, future) and customer admin
//!   forms emit proposals; everything funnels through one append path.
//! - [`SchemaSource`] is the abstract producer trait. Implementations: TTL
//!   directory walker (in this crate), MySQL/MSSQL scanners (future),
//!   customer admin forms (future UX layer).
//!
//! ## What this crate is NOT
//!
//! It is not a new SPO store. It is not a quad store. It does not parse
//! Cypher / Gremlin / SPARQL / GQL — those parsers already exist in
//! `lance-graph-planner::strategy::*`. It does not introduce new
//! `CausalEdge64` variants or new `BindSpace` columns. It does not modify
//! the MUL gate logic. It is a parser + cache + scoping facade over the
//! existing `lance-graph-contract::ontology` surface.

pub mod bridge;
pub mod bridges;
pub mod error;
pub mod foundry_map;
pub mod namespace;
pub mod namespace_registry;
pub mod proposal;
pub mod registry;
pub mod schema_source;
pub mod semantic_types;
pub mod ttl_parse;

#[cfg(feature = "lance-cache")]
pub mod lance_cache;

pub use bridge::{BridgeError, NamespaceBridge};
pub use error::Error;
pub use namespace::{NamespaceId, OgitUri, SchemaPtr};
pub use proposal::{
    HydrationReport, MappingHandle, MappingProposal, MappingProposalKind, MappingRow,
};
pub use registry::OntologyRegistry;
pub use schema_source::SchemaSource;
