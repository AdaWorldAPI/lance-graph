// Skip under Miri — hydration reads `data/ontologies/qudt-*.ttl` via std::fs.
#![cfg(not(miri))]

//! PR-bO-4 smoke test: hydrate QUDT and assert L2 invariants.
//!
//! Multi-file hydration: QUDT ships as separate core + units artifacts.
//! Both merge into a single bundle keyed by `OGIT::QUDT_V1.0`.

use lance_graph_contract::manifest::OGIT;
use lance_graph_ontology::{hydrate_qudt, OntologyRegistry};

const QUDT_BASE: &str = "http://qudt.org/schema/qudt/";

fn qudt(name: &str) -> String {
    format!("{QUDT_BASE}{name}")
}

#[test]
fn qudt_hydrator_smoke() {
    let registry = OntologyRegistry::new_in_memory();
    let g = hydrate_qudt(&registry).expect("QUDT hydrates");
    assert_eq!(g, OGIT::QUDT_V1.0);

    let bundle = registry
        .bundle_for(OGIT::QUDT_V1.0)
        .expect("ContextBundle registered at QUDT_V1");

    assert_eq!(bundle.g, OGIT::QUDT_V1.0);
    assert_eq!(bundle.domain_name, "qudt");
    assert_eq!(
        bundle.inherits_from,
        Some(OGIT::DOLCE_V1.0),
        "QUDT inherits_from must be DOLCE"
    );

    // Core schema + units catalogue → expect thousands of entities
    let entity_count = bundle.entity_count();
    assert!(
        entity_count > 2000,
        "expected >2000 entities for QUDT (core+units), got {entity_count}"
    );
}

#[test]
fn qudt_core_classes_resolve() {
    let registry = OntologyRegistry::new_in_memory();
    hydrate_qudt(&registry).expect("QUDT hydrates");
    let g = OGIT::QUDT_V1.0;
    // Canonical QUDT 2.1 core classes (dimension carrier is named
    // `QuantityKindDimensionVector`, not `Dimension`).
    for name in &[
        "Quantity",
        "QuantityKind",
        "Unit",
        "QuantityValue",
        "QuantityKindDimensionVector",
    ] {
        let iri = qudt(name);
        assert!(
            registry.resolve_iri_in(g, &iri).is_some(),
            "qudt:{name} must resolve under G={g}"
        );
    }
}

#[test]
fn qudt_unit_individuals_resolve() {
    // Spot-check the seven SI base units from the units catalogue.
    // QUDT 2.1 uses single-letter symbols for SI base units
    // (M / SEC / K / A / MOL / CD / KiloGM), not "METER" etc.
    let registry = OntologyRegistry::new_in_memory();
    hydrate_qudt(&registry).expect("QUDT hydrates");
    let g = OGIT::QUDT_V1.0;
    for unit in &["M", "SEC", "K", "A", "MOL", "CD", "KiloGM"] {
        let iri = format!("http://qudt.org/vocab/unit/{unit}");
        assert!(
            registry.resolve_iri_in(g, &iri).is_some(),
            "unit:{unit} must resolve under G={g}"
        );
    }
}

#[test]
fn qudt_edge_whitelist_has_quantity_kind_binding() {
    let registry = OntologyRegistry::new_in_memory();
    hydrate_qudt(&registry).expect("QUDT hydrates");
    let edges = registry
        .edge_types_for(OGIT::QUDT_V1.0)
        .expect("whitelist registered");
    for relation in &["hasQuantityKind", "hasUnit", "conversionMultiplier"] {
        let iri = qudt(relation);
        assert!(
            edges.iter().any(|e| e == &iri),
            "qudt:{relation} must be in the cascade whitelist"
        );
    }
}
