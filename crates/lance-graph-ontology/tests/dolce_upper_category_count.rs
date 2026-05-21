// Skip under Miri — hydration reads `data/ontologies/dul.ttl` via std::fs.
#![cfg(not(miri))]

//! PR-bO-1: assert all four DOLCE upper categories resolve to stable `u32`
//! entity IDs from their canonical IRIs.
//!
//! The cognitive shader's 16-bit DOLCE slot's high 8 bits encode the
//! Endurant/Perdurant/Quality/Abstract sub-classification. If any of these
//! four roots fails to resolve, the slot is unsynchronised against the
//! canonical upper-category register.

use lance_graph_contract::manifest::OGIT;
use lance_graph_ontology::{hydrate_dolce, OntologyRegistry};

const ENDURANT: &str = "http://www.ontologydesignpatterns.org/ont/dul/DUL.owl#Endurant";
const PERDURANT: &str = "http://www.ontologydesignpatterns.org/ont/dul/DUL.owl#Perdurant";
const QUALITY: &str = "http://www.ontologydesignpatterns.org/ont/dul/DUL.owl#Quality";
const ABSTRACT: &str = "http://www.ontologydesignpatterns.org/ont/dul/DUL.owl#Abstract";

#[test]
fn upper_categories_resolve_to_stable_ids() {
    let registry = OntologyRegistry::new_in_memory();
    hydrate_dolce(&registry).expect("DOLCE+DUL hydrates");

    let g = OGIT::DOLCE_V1.0;
    let endurant = registry
        .resolve_iri_in(g, ENDURANT)
        .expect("dul:Endurant resolves");
    let perdurant = registry
        .resolve_iri_in(g, PERDURANT)
        .expect("dul:Perdurant resolves");
    let quality = registry
        .resolve_iri_in(g, QUALITY)
        .expect("dul:Quality resolves");
    let abstract_id = registry
        .resolve_iri_in(g, ABSTRACT)
        .expect("dul:Abstract resolves");

    // All four IDs must be distinct — they're the four DOLCE roots.
    let ids = [endurant, perdurant, quality, abstract_id];
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
    assert_eq!(registry.resolve_iri_in(g, ENDURANT), Some(endurant));
    assert_eq!(registry.resolve_iri_in(g, PERDURANT), Some(perdurant));
}
