// Skip under Miri — hydration reads `data/ontologies/dul.ttl` via std::fs.
#![cfg(not(miri))]

//! PR-bO-1: assert the four DOLCE+DUL upper categories resolve to stable
//! `u32` entity IDs from their canonical IRIs.
//!
//! The cognitive shader's 16-bit DOLCE slot's high 8 bits encode the
//! upper-category sub-classification. Per the canonical DOLCE+DUL ontology
//! header ("the names of classes and relations have been made more
//! intuitive"), DUL renames the original DOLCE-Lite-Plus upper categories:
//!
//!   DOLCE-Lite-Plus  →  DOLCE+DnS Ultralite (DUL)
//!   Endurant         →  Object
//!   Perdurant        →  Event
//!   Quality          →  Quality   (unchanged)
//!   Abstract         →  Abstract  (unchanged)
//!
//! These four classes are direct sub-classes of `dul:Entity` in canonical
//! DUL. If any fails to resolve the slot is unsynchronised.

use lance_graph_contract::manifest::OGIT;
use lance_graph_ontology::{hydrate_dolce, OntologyRegistry};

const ENTITY: &str = "http://www.ontologydesignpatterns.org/ont/dul/DUL.owl#Entity";
const OBJECT: &str = "http://www.ontologydesignpatterns.org/ont/dul/DUL.owl#Object";
const EVENT: &str = "http://www.ontologydesignpatterns.org/ont/dul/DUL.owl#Event";
const QUALITY: &str = "http://www.ontologydesignpatterns.org/ont/dul/DUL.owl#Quality";
const ABSTRACT: &str = "http://www.ontologydesignpatterns.org/ont/dul/DUL.owl#Abstract";

#[test]
fn upper_categories_resolve_to_stable_ids() {
    let registry = OntologyRegistry::new_in_memory();
    hydrate_dolce(&registry).expect("DOLCE+DUL hydrates");

    let g = OGIT::DOLCE_V1.0;
    let entity = registry
        .resolve_iri_in(g, ENTITY)
        .expect("dul:Entity resolves (the upper-category root)");
    let object = registry
        .resolve_iri_in(g, OBJECT)
        .expect("dul:Object resolves (replaces DOLCE-LP Endurant)");
    let event = registry
        .resolve_iri_in(g, EVENT)
        .expect("dul:Event resolves (replaces DOLCE-LP Perdurant)");
    let quality = registry
        .resolve_iri_in(g, QUALITY)
        .expect("dul:Quality resolves");
    let abstract_id = registry
        .resolve_iri_in(g, ABSTRACT)
        .expect("dul:Abstract resolves");

    // All five IDs must be distinct — Entity is the root and the four are
    // direct sub-classes.
    let ids = [entity, object, event, quality, abstract_id];
    for i in 0..ids.len() {
        for j in (i + 1)..ids.len() {
            assert_ne!(
                ids[i], ids[j],
                "upper-category IDs must be distinct: ids[{i}] == ids[{j}] == {}",
                ids[i]
            );
        }
    }

    // IDs must be stable across re-resolution within the same hydration.
    assert_eq!(registry.resolve_iri_in(g, OBJECT), Some(object));
    assert_eq!(registry.resolve_iri_in(g, EVENT), Some(event));
}
