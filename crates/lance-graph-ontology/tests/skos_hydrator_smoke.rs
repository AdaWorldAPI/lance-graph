// Skip under Miri — hydration reads `data/ontologies/skos/*.rdf` via std::fs.
#![cfg(not(miri))]

//! PR-bO-5 smoke test: hydrate SKOS Core + SKOS-XL and assert L2 invariants.

use lance_graph_contract::manifest::OGIT;
use lance_graph_ontology::{hydrate_skos, OntologyRegistry};

const SKOS_BASE: &str = "http://www.w3.org/2004/02/skos/core#";
const SKOSXL_BASE: &str = "http://www.w3.org/2008/05/skos-xl#";

fn skos(name: &str) -> String {
    format!("{SKOS_BASE}{name}")
}

fn skosxl(name: &str) -> String {
    format!("{SKOSXL_BASE}{name}")
}

#[test]
fn skos_hydrator_smoke() {
    let registry = OntologyRegistry::new_in_memory();
    let g = hydrate_skos(&registry).expect("SKOS hydrates");
    assert_eq!(g, OGIT::SKOS_V1.0);

    let bundle = registry
        .bundle_for(OGIT::SKOS_V1.0)
        .expect("ContextBundle registered at SKOS_V1");

    assert_eq!(bundle.g, OGIT::SKOS_V1.0);
    assert_eq!(bundle.domain_name, "skos");
    assert_eq!(
        bundle.inherits_from,
        Some(OGIT::DOLCE_V1.0),
        "SKOS inherits_from must be DOLCE"
    );

    // SKOS Core has ~32 named entities + XL adds 5 more + RDF/OWL/DC
    // primitives interned during hydration; expect comfortably > 30.
    let entity_count = bundle.entity_count();
    assert!(
        entity_count > 30,
        "expected >30 entities for SKOS Core+XL, got {entity_count}"
    );
}

#[test]
fn skos_core_classes_and_properties_resolve() {
    let registry = OntologyRegistry::new_in_memory();
    hydrate_skos(&registry).expect("SKOS hydrates");
    let g = OGIT::SKOS_V1.0;
    // SKOS Core classes
    for name in &[
        "Concept",
        "ConceptScheme",
        "Collection",
        "OrderedCollection",
    ] {
        let iri = skos(name);
        assert!(
            registry.resolve_iri_in(g, &iri).is_some(),
            "skos:{name} must resolve under G={g}"
        );
    }
    // SKOS Core semantic-relation properties (the load-bearing surface for
    // SKR03/SKR04 alignment and thesaurus traversal).
    for name in &[
        "broader",
        "narrower",
        "related",
        "broaderTransitive",
        "narrowerTransitive",
        "inScheme",
        "hasTopConcept",
        "member",
        "prefLabel",
        "altLabel",
        "hiddenLabel",
        "notation",
        "broadMatch",
        "narrowMatch",
        "exactMatch",
        "closeMatch",
    ] {
        let iri = skos(name);
        assert!(
            registry.resolve_iri_in(g, &iri).is_some(),
            "skos:{name} must resolve under G={g}"
        );
    }
}

#[test]
fn skos_xl_label_surface_resolves() {
    let registry = OntologyRegistry::new_in_memory();
    hydrate_skos(&registry).expect("SKOS hydrates");
    let g = OGIT::SKOS_V1.0;
    // SKOS-XL lifts labels to first-class IRIs.
    for name in &[
        "Label",
        "literalForm",
        "prefLabel",
        "altLabel",
        "hiddenLabel",
        "labelRelation",
    ] {
        let iri = skosxl(name);
        assert!(
            registry.resolve_iri_in(g, &iri).is_some(),
            "skosxl:{name} must resolve under G={g}"
        );
    }
}

#[test]
fn skos_edge_whitelist_has_match_predicates() {
    let registry = OntologyRegistry::new_in_memory();
    hydrate_skos(&registry).expect("SKOS hydrates");
    let edges = registry
        .edge_types_for(OGIT::SKOS_V1.0)
        .expect("whitelist registered");
    // Cross-scheme `*Match` predicates are the load-bearing surface for
    // bridging SKR03/SKR04 to FIBO and HGB; they MUST be in the whitelist.
    for relation in &["broadMatch", "narrowMatch", "exactMatch", "closeMatch"] {
        let iri = skos(relation);
        assert!(
            edges.iter().any(|e| e == &iri),
            "skos:{relation} must be in the cascade whitelist"
        );
    }
}
