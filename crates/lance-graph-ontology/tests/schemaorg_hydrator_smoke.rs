// Skip under Miri — hydration reads `data/ontologies/schemaorg.ttl` via std::fs.
#![cfg(not(miri))]

//! PR-bO-8 smoke test: hydrate schema.org and assert L3 invariants.

use lance_graph_contract::manifest::OGIT;
use lance_graph_ontology::{hydrate_schemaorg, OntologyRegistry};

const SCHEMA_BASE: &str = "https://schema.org/";

fn schema(name: &str) -> String {
    format!("{SCHEMA_BASE}{name}")
}

#[test]
fn schemaorg_hydrator_smoke() {
    let registry = OntologyRegistry::new_in_memory();
    let g = hydrate_schemaorg(&registry).expect("schema.org hydrates");
    assert_eq!(g, OGIT::SCHEMAORG_V1.0);

    let bundle = registry
        .bundle_for(OGIT::SCHEMAORG_V1.0)
        .expect("ContextBundle registered at SCHEMAORG_V1");

    assert_eq!(bundle.g, OGIT::SCHEMAORG_V1.0);
    assert_eq!(bundle.domain_name, "schemaorg");
    assert_eq!(
        bundle.inherits_from,
        Some(OGIT::DOLCE_V1.0),
        "schema.org inherits_from must be DOLCE"
    );

    // schema.org has ~1400 named classes/properties
    let entity_count = bundle.entity_count();
    assert!(
        entity_count > 1000,
        "expected >1000 entities for schema.org, got {entity_count}"
    );
}

#[test]
fn schemaorg_canonical_iris_resolve() {
    let registry = OntologyRegistry::new_in_memory();
    hydrate_schemaorg(&registry).expect("schema.org hydrates");
    let g = OGIT::SCHEMAORG_V1.0;
    // Top-level schema.org classes that downstream business ontologies bind
    // to (Person/Organization for FIBO LegalPerson, Place/Event for AEC3PO,
    // Product/CreativeWork for invoicing/UBL, …).
    for name in &[
        "Thing",
        "Person",
        "Organization",
        "Place",
        "Event",
        "Product",
        "CreativeWork",
        "Action",
        "Intangible",
    ] {
        let iri = schema(name);
        assert!(
            registry.resolve_iri_in(g, &iri).is_some(),
            "schema:{name} must resolve under G={g}"
        );
    }
}

#[test]
fn schemaorg_edge_whitelist_has_domain_range_includes() {
    let registry = OntologyRegistry::new_in_memory();
    hydrate_schemaorg(&registry).expect("schema.org hydrates");
    let edges = registry
        .edge_types_for(OGIT::SCHEMAORG_V1.0)
        .expect("whitelist registered");
    // schema.org's flexible-typing surface uses domainIncludes / rangeIncludes
    // instead of strict rdfs:domain / rdfs:range — these MUST be in the
    // whitelist or property-typing cascades break.
    for relation in &["domainIncludes", "rangeIncludes"] {
        let iri = schema(relation);
        assert!(
            edges.iter().any(|e| e == &iri),
            "schema:{relation} must be in the cascade whitelist"
        );
    }
}
