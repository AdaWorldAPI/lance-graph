//! schema.org (`https://schema.org/`) hydration glue.
//!
//! L3 cross-domain business ontology — the dominant ontology for
//! commercial-web modelling: `Thing`, `Person`, `Organization`,
//! `LegalPerson`, `Place`, `Event`, `Product`, `CreativeWork`, `Action`,
//! plus ~1400 named sub-classes.
//!
//! Declares `inherits_from: Some(OGIT::DOLCE_V1.0)`. Alignment with DUL:
//! `schema:Person ⊑ dul:NaturalPerson`, `schema:Organization ⊑
//! dul:Organization`, `schema:Event ⊑ dul:Event`, `schema:Place ⊑
//! dul:Place`. Downstream consumers assert these axioms when they need
//! cross-G traversal.

use std::path::{Path, PathBuf};

use lance_graph_contract::manifest::OGIT;

use super::owl::{HydrateErr, MetaStructureHydrator, OwlHydrator};
use crate::registry::OntologyRegistry;

const SCHEMAORG_TTL_RELATIVE_PATH: &str = "data/ontologies/schemaorg.ttl";

const SCHEMAORG_EDGE_WHITELIST: &[&str] = &[
    "http://www.w3.org/2000/01/rdf-schema#subClassOf",
    "http://www.w3.org/2000/01/rdf-schema#subPropertyOf",
    // Property domain/range (schema.org's flexible-typing surface)
    "https://schema.org/domainIncludes",
    "https://schema.org/rangeIncludes",
    // Versioning / deprecation
    "https://schema.org/supersededBy",
    "https://schema.org/inverseOf",
    // Core relational predicates
    "https://schema.org/about",
    "https://schema.org/mainEntity",
    "https://schema.org/mainEntityOfPage",
    "https://schema.org/sameAs",
    "https://schema.org/isPartOf",
    "https://schema.org/hasPart",
    "https://schema.org/member",
    "https://schema.org/memberOf",
    "https://schema.org/parentOrganization",
    "https://schema.org/subOrganization",
    "https://schema.org/location",
    "https://schema.org/containedInPlace",
    "https://schema.org/containsPlace",
    "https://schema.org/agent",
    "https://schema.org/participant",
    "https://schema.org/object",
];

/// Hydrate schema.org as `OGIT::SCHEMAORG_V1` (L3 commercial-web).
pub fn hydrate_schemaorg(registry: &OntologyRegistry) -> Result<u32, HydrateErr> {
    hydrate_schemaorg_from(&schemaorg_ttl_path(), registry)
}

/// Test-friendly variant: hydrate schema.org from an explicit path.
pub fn hydrate_schemaorg_from(
    ttl_path: &Path,
    registry: &OntologyRegistry,
) -> Result<u32, HydrateErr> {
    let hydrator = OwlHydrator {
        g: OGIT::SCHEMAORG_V1.0,
        version: OGIT::SCHEMAORG_V1.1,
        domain_name: "schemaorg".to_string(),
        inherits_from: Some(OGIT::DOLCE_V1.0),
        starting_entity_id: 100,
    };
    hydrator.hydrate(ttl_path, registry)?;
    registry
        .register_edge_types(OGIT::SCHEMAORG_V1.0, SCHEMAORG_EDGE_WHITELIST)
        .map_err(|reason| HydrateErr::Registry {
            g: OGIT::SCHEMAORG_V1.0,
            reason,
        })?;
    Ok(OGIT::SCHEMAORG_V1.0)
}

fn schemaorg_ttl_path() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
        .join(SCHEMAORG_TTL_RELATIVE_PATH)
}
