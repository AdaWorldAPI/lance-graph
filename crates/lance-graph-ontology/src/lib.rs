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
pub mod hydrators;
pub mod namespace;
pub mod namespace_registry;
pub mod odoo_blueprint;
pub mod proposal;
pub mod registry;
pub mod schema_source;
pub mod semantic_types;
pub mod ttl_parse;

#[cfg(feature = "lance-cache")]
pub mod lance_cache;

pub use bridge::{BridgeError, NamespaceBridge};
pub use error::Error;
pub use hydrators::{
    classify_odoo, collect_xsd_files, hydrate_dolce, hydrate_dolce_from, hydrate_dolce_from_many,
    hydrate_fibo_be, hydrate_fibo_be_from, hydrate_fibo_fnd, hydrate_fibo_fnd_from, hydrate_odoo,
    hydrate_odoo_from, hydrate_owltime, hydrate_owltime_from, hydrate_provo, hydrate_provo_from,
    hydrate_qudt, hydrate_qudt_from, hydrate_schemaorg, hydrate_schemaorg_from, hydrate_skos,
    hydrate_skos_from, hydrate_skr03, hydrate_skr03_bau, hydrate_skr03_bau_from, hydrate_skr03_from,
    hydrate_skr04, hydrate_skr04_from, hydrate_zugferd, hydrate_zugferd_from, hydrate_zugferd_rules,
    hydrate_zugferd_rules_from, ContextBundle, DolceCategory, EntityId, HydrateErr,
    MetaStructureHydrator, OntologySlot, OwlHydrator, SchematronHydrator, SkrHydrator, XsdHydrator,
    SKR03_BAU_IRI_PREFIX, SKR03_IRI_PREFIX, SKR04_IRI_PREFIX,
};
pub use namespace::{NamespaceId, OgitUri, SchemaPtr};
pub use proposal::{
    HydrationReport, MappingHandle, MappingProposal, MappingProposalKind, MappingRow,
};
pub use registry::OntologyRegistry;
pub use schema_source::SchemaSource;
pub use ttl_parse::{parse_family_registry, FamilyRegistryEntry};
