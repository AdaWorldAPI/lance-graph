// Skip under Miri — hydration reads `data/ontologies/time.ttl` via std::fs.
#![cfg(not(miri))]

//! PR-bO-2 smoke test: hydrate OWL-Time and assert L2 invariants.
//!
//! - bundle exists at `OGIT::TIME_V1.0`
//! - `inherits_from == Some(OGIT::DOLCE_V1.0)`  (L2 root under DOLCE)
//! - `domain_name == "owltime"`
//! - canonical `Interval` / `Instant` / `ProperInterval` IRIs resolve
//! - cascade edge whitelist includes Allen interval relations

use lance_graph_contract::manifest::OGIT;
use lance_graph_ontology::{hydrate_owltime, OntologyRegistry};

const TIME_BASE: &str = "http://www.w3.org/2006/time#";

fn time(name: &str) -> String {
    format!("{TIME_BASE}{name}")
}

#[test]
fn owltime_hydrator_smoke() {
    let registry = OntologyRegistry::new_in_memory();
    let g = hydrate_owltime(&registry).expect("OWL-Time hydrates");
    assert_eq!(g, OGIT::TIME_V1.0);

    let bundle = registry
        .bundle_for(OGIT::TIME_V1.0)
        .expect("ContextBundle registered at TIME_V1");

    assert_eq!(bundle.g, OGIT::TIME_V1.0);
    assert_eq!(bundle.version, OGIT::TIME_V1.1);
    assert_eq!(bundle.domain_name, "owltime");
    assert_eq!(
        bundle.inherits_from,
        Some(OGIT::DOLCE_V1.0),
        "OWL-Time inherits_from must be DOLCE (L2 under L1)"
    );

    let entity_count = bundle.entity_count();
    assert!(
        entity_count > 80,
        "expected >80 entities for OWL-Time, got {entity_count}"
    );
}

#[test]
fn owltime_canonical_iris_resolve() {
    let registry = OntologyRegistry::new_in_memory();
    hydrate_owltime(&registry).expect("OWL-Time hydrates");
    let g = OGIT::TIME_V1.0;
    for name in &[
        "TemporalEntity",
        "Instant",
        "Interval",
        "ProperInterval",
        "DateTimeDescription",
        "Duration",
    ] {
        let iri = time(name);
        assert!(
            registry.resolve_iri_in(g, &iri).is_some(),
            "time:{name} must resolve under G={g}"
        );
    }
}

#[test]
fn owltime_edge_whitelist_has_allen_relations() {
    let registry = OntologyRegistry::new_in_memory();
    hydrate_owltime(&registry).expect("OWL-Time hydrates");
    let edges = registry
        .edge_types_for(OGIT::TIME_V1.0)
        .expect("whitelist registered");
    for relation in &[
        "intervalBefore",
        "intervalAfter",
        "intervalMeets",
        "intervalOverlaps",
        "intervalStarts",
        "intervalFinishes",
        "intervalDuring",
        "intervalEquals",
    ] {
        let iri = time(relation);
        assert!(
            edges.iter().any(|e| e == &iri),
            "Allen relation time:{relation} must be in the cascade whitelist"
        );
    }
}
