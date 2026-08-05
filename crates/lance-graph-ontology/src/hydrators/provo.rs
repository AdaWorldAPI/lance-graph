//! PROV-O (W3C `http://www.w3.org/ns/prov#`) hydration glue.
//!
//! L2 universal provenance ontology — defines `Entity`, `Activity`, `Agent`
//! and the eight load-bearing provenance relations (`wasGeneratedBy`,
//! `used`, `wasDerivedFrom`, `wasAttributedTo`, `wasAssociatedWith`,
//! `wasInformedBy`, `actedOnBehalfOf`, `wasInvalidatedBy`).
//!
//! Declares `inherits_from: Some(OGIT::DOLCE_V1.0)` and aligns with DUL via
//! `prov:Activity ⊑ dul:Event`, `prov:Agent ⊑ dul:Agent`, `prov:Entity ⊑
//! dul:Object` (axioms downstream consumers assert when they need
//! cross-G traversal).

use std::path::{Path, PathBuf};

use lance_graph_contract::manifest::OGIT;

use super::owl::{HydrateErr, MetaStructureHydrator, OwlHydrator};
use crate::registry::OntologyRegistry;

const PROVO_TTL_RELATIVE_PATH: &str = "data/ontologies/provo.ttl";

const PROVO_EDGE_WHITELIST: &[&str] = &[
    "http://www.w3.org/2000/01/rdf-schema#subClassOf",
    "http://www.w3.org/2000/01/rdf-schema#subPropertyOf",
    // Generation, derivation, attribution
    "http://www.w3.org/ns/prov#wasGeneratedBy",
    "http://www.w3.org/ns/prov#generated",
    "http://www.w3.org/ns/prov#used",
    "http://www.w3.org/ns/prov#wasUsedBy",
    "http://www.w3.org/ns/prov#wasDerivedFrom",
    "http://www.w3.org/ns/prov#wasAttributedTo",
    "http://www.w3.org/ns/prov#wasAssociatedWith",
    "http://www.w3.org/ns/prov#wasInformedBy",
    // Activity lifecycle
    "http://www.w3.org/ns/prov#wasStartedBy",
    "http://www.w3.org/ns/prov#wasEndedBy",
    "http://www.w3.org/ns/prov#wasInvalidatedBy",
    // Agent delegation
    "http://www.w3.org/ns/prov#actedOnBehalfOf",
    "http://www.w3.org/ns/prov#hadDelegate",
    // Plan / role
    "http://www.w3.org/ns/prov#hadPlan",
    "http://www.w3.org/ns/prov#hadActivity",
    "http://www.w3.org/ns/prov#hadRole",
    "http://www.w3.org/ns/prov#qualifiedAssociation",
    "http://www.w3.org/ns/prov#qualifiedAttribution",
    "http://www.w3.org/ns/prov#qualifiedDerivation",
    "http://www.w3.org/ns/prov#qualifiedGeneration",
];

/// Hydrate PROV-O as `OGIT::PROVO_V1` (L2 universal provenance ontology).
pub fn hydrate_provo(registry: &OntologyRegistry) -> Result<u32, HydrateErr> {
    hydrate_provo_from(&provo_ttl_path(), registry)
}

/// Test-friendly variant: hydrate PROV-O from an explicit path.
pub fn hydrate_provo_from(ttl_path: &Path, registry: &OntologyRegistry) -> Result<u32, HydrateErr> {
    let hydrator = OwlHydrator {
        g: OGIT::PROVO_V1.0,
        version: OGIT::PROVO_V1.1,
        domain_name: "provo".to_string(),
        inherits_from: Some(OGIT::DOLCE_V1.0),
        starting_entity_id: 100,
    };
    hydrator.hydrate(ttl_path, registry)?;
    registry
        .register_edge_types(OGIT::PROVO_V1.0, PROVO_EDGE_WHITELIST)
        .map_err(|reason| HydrateErr::Registry {
            g: OGIT::PROVO_V1.0,
            reason,
        })?;
    Ok(OGIT::PROVO_V1.0)
}

fn provo_ttl_path() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
        .join(PROVO_TTL_RELATIVE_PATH)
}
