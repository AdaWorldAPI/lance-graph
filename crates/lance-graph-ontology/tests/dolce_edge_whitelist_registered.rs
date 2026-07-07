// Skip under Miri — hydration reads `data/ontologies/dul.ttl` via std::fs.
#![cfg(not(miri))]

//! PR-bO-1: assert the cascade edge-IRI whitelist is registered after
//! `hydrate_dolce()` returns.
//!
//! The cognitive shader's cascade traversal walks the whitelist registered
//! per `G`. The load-bearing minimum is 15 edges spanning upper-category
//! subsumption, DnS classification + role-binding, part-of / constitution,
//! and temporal anchoring.

use lance_graph_contract::manifest::OGIT;
use lance_graph_ontology::{hydrate_dolce, OntologyRegistry};

#[test]
fn edge_whitelist_registered() {
    let registry = OntologyRegistry::new_in_memory();
    hydrate_dolce(&registry).expect("DOLCE+DUL hydrates");

    let edges = registry
        .edge_types_for(OGIT::DOLCE_V1.0)
        .expect("edge whitelist registered for DOLCE_V1");
    assert!(
        edges.len() >= 15,
        "expected >= 15 cascade edge IRIs, got {} ({:?})",
        edges.len(),
        edges
    );

    // Load-bearing minimum: upper-category subsumption.
    assert!(
        edges
            .iter()
            .any(|e| e == "http://www.w3.org/2000/01/rdf-schema#subClassOf"),
        "rdfs:subClassOf must be in the whitelist"
    );

    // Load-bearing minimum: DnS classification + role-binding.
    let dul = |name: &str| {
        format!("http://www.ontologydesignpatterns.org/ont/dul/DUL.owl#{name}")
    };
    for name in &["isClassifiedBy", "classifies", "hasRole", "hasParticipant"] {
        let iri = dul(name);
        assert!(
            edges.iter().any(|e| e == &iri),
            "dul:{name} must be in the whitelist"
        );
    }

    // Load-bearing minimum: part-of / constitution + temporal anchoring.
    for name in &["hasPart", "isPartOf", "hasTimeInterval"] {
        let iri = dul(name);
        assert!(
            edges.iter().any(|e| e == &iri),
            "dul:{name} must be in the whitelist"
        );
    }
}

#[test]
fn edge_types_for_unknown_g_returns_none() {
    let registry = OntologyRegistry::new_in_memory();
    // No bundle registered yet — query should return None.
    assert!(registry.edge_types_for(OGIT::DOLCE_V1.0).is_none());
}

#[test]
fn re_register_edge_types_is_idempotent() {
    let registry = OntologyRegistry::new_in_memory();
    hydrate_dolce(&registry).expect("DOLCE+DUL hydrates");
    let initial = registry
        .edge_types_for(OGIT::DOLCE_V1.0)
        .expect("whitelist registered");
    // Re-register the same edges; whitelist size should not grow.
    registry
        .register_edge_types(
            OGIT::DOLCE_V1.0,
            &["http://www.w3.org/2000/01/rdf-schema#subClassOf"],
        )
        .expect("re-registration succeeds");
    let after = registry
        .edge_types_for(OGIT::DOLCE_V1.0)
        .expect("whitelist still registered");
    assert_eq!(
        initial.len(),
        after.len(),
        "re-registering a known edge IRI must be idempotent"
    );
}
