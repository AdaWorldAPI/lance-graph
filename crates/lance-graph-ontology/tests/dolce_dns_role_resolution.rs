// Skip under Miri — hydration reads `data/ontologies/dul.ttl` via std::fs.
#![cfg(not(miri))]

//! PR-bO-1: assert the load-bearing DnS role IRIs resolve.
//!
//! The cognitive shader's 16-bit DOLCE slot's low 8 bits encode the DnS role
//! per the Descriptions & Situations module. If any of `Agent` / `Patient` /
//! `Instrument` / `Location` / `TimeInterval` fails to resolve, role-binding
//! at cascade-step time is unanchored.

use lance_graph_contract::manifest::OGIT;
use lance_graph_ontology::{hydrate_dolce, OntologyRegistry};

const DUL_BASE: &str = "http://www.ontologydesignpatterns.org/ont/dul/DUL.owl#";

fn dul(name: &str) -> String {
    format!("{DUL_BASE}{name}")
}

#[test]
fn known_dns_role_iris_resolve() {
    let registry = OntologyRegistry::new_in_memory();
    hydrate_dolce(&registry).expect("DOLCE+DUL hydrates");

    let g = OGIT::DOLCE_V1.0;
    for name in &["Agent", "Patient", "Instrument", "Location", "TimeInterval"] {
        let iri = dul(name);
        assert!(
            registry.resolve_iri_in(g, &iri).is_some(),
            "dul:{name} ({iri}) must resolve under G={g}"
        );
    }
}

#[test]
fn dns_role_anchor_concept_is_present() {
    // The DnS role hierarchy is rooted at dul:Concept (Role/Task/Parameter
    // are sub-classes of Concept). Concept itself must be present so the
    // upper-category cascade can reach it via `rdfs:subClassOf*`.
    let registry = OntologyRegistry::new_in_memory();
    hydrate_dolce(&registry).expect("DOLCE+DUL hydrates");
    let g = OGIT::DOLCE_V1.0;
    assert!(
        registry.resolve_iri_in(g, &dul("Concept")).is_some(),
        "dul:Concept must be present — it anchors the DnS role hierarchy"
    );
    assert!(
        registry.resolve_iri_in(g, &dul("Role")).is_some(),
        "dul:Role must be present (sub-class of Concept)"
    );
}
