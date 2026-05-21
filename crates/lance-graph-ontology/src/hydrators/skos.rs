//! SKOS (Simple Knowledge Organization System, `http://www.w3.org/2004/02/skos/core#`)
//! hydration glue.
//!
//! L2 universal ontology for thesauri, classification schemes, subject-heading
//! lists, taxonomies, and other controlled vocabularies. Defines `Concept`,
//! `ConceptScheme`, `Collection`, `OrderedCollection`, plus the canonical
//! semantic-relation surface (`broader`/`narrower`/`related` plus their
//! transitive variants and the `*Match` mapping siblings).
//!
//! Multi-file hydration: SKOS Core + SKOS-XL (eXtension for Labels — adds
//! `skosxl:Label` so labels can be first-class resources with their own
//! IRIs rather than literals) hydrate into one bundle keyed by
//! `OGIT::SKOS_V1.0`. SKOS-XL declares `owl:imports` against Core so the
//! merged bundle gives consumers both surfaces in a single G slot.
//!
//! Declares `inherits_from: Some(OGIT::DOLCE_V1.0)`. Alignment with DUL:
//! `skos:Concept ⊑ dul:Concept`, `skos:Collection ⊑ dul:Collection`.
//! Downstream consumers assert these axioms when they need cross-G
//! traversal between SKR-style classification trees and DOLCE
//! upper-category cascade.

use std::path::{Path, PathBuf};

use lance_graph_contract::manifest::OGIT;

use super::owl::{HydrateErr, OwlHydrator};
use crate::registry::OntologyRegistry;

const SKOS_CORE_RELATIVE_PATH: &str = "data/ontologies/skos/skos-core.rdf";
const SKOS_XL_RELATIVE_PATH: &str = "data/ontologies/skos/skos-xl.rdf";

const SKOS_EDGE_WHITELIST: &[&str] = &[
    "http://www.w3.org/2000/01/rdf-schema#subClassOf",
    "http://www.w3.org/2000/01/rdf-schema#subPropertyOf",
    "http://www.w3.org/2002/07/owl#equivalentClass",
    // Hierarchical (intra-scheme) relations
    "http://www.w3.org/2004/02/skos/core#broader",
    "http://www.w3.org/2004/02/skos/core#narrower",
    "http://www.w3.org/2004/02/skos/core#broaderTransitive",
    "http://www.w3.org/2004/02/skos/core#narrowerTransitive",
    "http://www.w3.org/2004/02/skos/core#related",
    "http://www.w3.org/2004/02/skos/core#semanticRelation",
    // Scheme membership
    "http://www.w3.org/2004/02/skos/core#inScheme",
    "http://www.w3.org/2004/02/skos/core#hasTopConcept",
    "http://www.w3.org/2004/02/skos/core#topConceptOf",
    "http://www.w3.org/2004/02/skos/core#member",
    "http://www.w3.org/2004/02/skos/core#memberList",
    // Cross-scheme mapping (load-bearing for SKR03/SKR04 alignment)
    "http://www.w3.org/2004/02/skos/core#mappingRelation",
    "http://www.w3.org/2004/02/skos/core#broadMatch",
    "http://www.w3.org/2004/02/skos/core#narrowMatch",
    "http://www.w3.org/2004/02/skos/core#relatedMatch",
    "http://www.w3.org/2004/02/skos/core#exactMatch",
    "http://www.w3.org/2004/02/skos/core#closeMatch",
    // Lexical labels (Core + XL)
    "http://www.w3.org/2004/02/skos/core#prefLabel",
    "http://www.w3.org/2004/02/skos/core#altLabel",
    "http://www.w3.org/2004/02/skos/core#hiddenLabel",
    "http://www.w3.org/2004/02/skos/core#notation",
    "http://www.w3.org/2008/05/skos-xl#literalForm",
    "http://www.w3.org/2008/05/skos-xl#prefLabel",
    "http://www.w3.org/2008/05/skos-xl#altLabel",
    "http://www.w3.org/2008/05/skos-xl#hiddenLabel",
    "http://www.w3.org/2008/05/skos-xl#labelRelation",
];

/// Hydrate SKOS as `OGIT::SKOS_V1` (L2 universal knowledge-organization).
///
/// Hydrates both Core and XL artifacts into a single bundle.
pub fn hydrate_skos(registry: &OntologyRegistry) -> Result<u32, HydrateErr> {
    let core = skos_core_path();
    let xl = skos_xl_path();
    hydrate_skos_from(&[&core, &xl], registry)
}

/// Test-friendly variant: hydrate SKOS from explicit paths. Accepts a slice
/// so callers can opt into Core-only, or substitute the `skos-owl1dl.rdf`
/// OWL-DL-conformant variant (also shipped under `data/ontologies/skos/`).
pub fn hydrate_skos_from(
    ttl_paths: &[&Path],
    registry: &OntologyRegistry,
) -> Result<u32, HydrateErr> {
    let hydrator = OwlHydrator {
        g: OGIT::SKOS_V1.0,
        version: OGIT::SKOS_V1.1,
        domain_name: "skos".to_string(),
        inherits_from: Some(OGIT::DOLCE_V1.0),
        starting_entity_id: 100,
    };
    hydrator.hydrate_many(ttl_paths, registry)?;
    registry
        .register_edge_types(OGIT::SKOS_V1.0, SKOS_EDGE_WHITELIST)
        .map_err(|reason| HydrateErr::Registry {
            g: OGIT::SKOS_V1.0,
            reason,
        })?;
    Ok(OGIT::SKOS_V1.0)
}

fn skos_core_path() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
        .join(SKOS_CORE_RELATIVE_PATH)
}

fn skos_xl_path() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
        .join(SKOS_XL_RELATIVE_PATH)
}
