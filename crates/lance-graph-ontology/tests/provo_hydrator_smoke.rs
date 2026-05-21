// Skip under Miri — hydration reads `data/ontologies/provo.ttl` via std::fs.
#![cfg(not(miri))]

//! PR-bO-3 smoke test: hydrate PROV-O and assert L2 invariants.

use lance_graph_contract::manifest::OGIT;
use lance_graph_ontology::{hydrate_provo, OntologyRegistry};

const PROV_BASE: &str = "http://www.w3.org/ns/prov#";

fn prov(name: &str) -> String {
    format!("{PROV_BASE}{name}")
}

#[test]
fn provo_hydrator_smoke() {
    let registry = OntologyRegistry::new_in_memory();
    let g = hydrate_provo(&registry).expect("PROV-O hydrates");
    assert_eq!(g, OGIT::PROVO_V1.0);

    let bundle = registry
        .bundle_for(OGIT::PROVO_V1.0)
        .expect("ContextBundle registered at PROVO_V1");

    assert_eq!(bundle.g, OGIT::PROVO_V1.0);
    assert_eq!(bundle.domain_name, "provo");
    assert_eq!(
        bundle.inherits_from,
        Some(OGIT::DOLCE_V1.0),
        "PROV-O inherits_from must be DOLCE"
    );

    let entity_count = bundle.entity_count();
    assert!(
        entity_count > 80,
        "expected >80 entities for PROV-O, got {entity_count}"
    );
}

#[test]
fn provo_canonical_iris_resolve() {
    let registry = OntologyRegistry::new_in_memory();
    hydrate_provo(&registry).expect("PROV-O hydrates");
    let g = OGIT::PROVO_V1.0;
    // The three PROV-O top-level classes (Entity, Activity, Agent) plus
    // the standard provenance relations.
    for name in &[
        "Entity",
        "Activity",
        "Agent",
        "Bundle",
        "wasGeneratedBy",
        "used",
        "wasDerivedFrom",
        "wasAttributedTo",
        "wasAssociatedWith",
        "actedOnBehalfOf",
    ] {
        let iri = prov(name);
        assert!(
            registry.resolve_iri_in(g, &iri).is_some(),
            "prov:{name} must resolve under G={g}"
        );
    }
}

#[test]
fn provo_edge_whitelist_has_load_bearing_relations() {
    let registry = OntologyRegistry::new_in_memory();
    hydrate_provo(&registry).expect("PROV-O hydrates");
    let edges = registry
        .edge_types_for(OGIT::PROVO_V1.0)
        .expect("whitelist registered");
    for relation in &[
        "wasGeneratedBy",
        "used",
        "wasDerivedFrom",
        "wasAttributedTo",
        "actedOnBehalfOf",
    ] {
        let iri = prov(relation);
        assert!(
            edges.iter().any(|e| e == &iri),
            "prov:{relation} must be in the cascade whitelist"
        );
    }
}
