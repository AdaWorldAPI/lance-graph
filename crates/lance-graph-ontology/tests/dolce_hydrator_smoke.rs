// Skip under Miri — hydration reads `data/ontologies/dul.ttl` via std::fs,
// and Miri isolation blocks the syscalls.
#![cfg(not(miri))]

//! PR-bO-1 smoke test: hydrate DOLCE+DUL and assert the L1-root invariants.
//!
//! - bundle exists at `OGIT::DOLCE_V1.0`
//! - `inherits_from == None` (L1 root — DOLCE is the only ontology that
//!   declares no parent)
//! - `domain_name == "dolce"`
//! - `entity_count > 200` per the spec (DOLCE+DUL has ~250 named entities)

use lance_graph_contract::manifest::OGIT;
use lance_graph_ontology::{hydrate_dolce, OntologyRegistry};

#[test]
fn dolce_hydrator_smoke() {
    let registry = OntologyRegistry::new_in_memory();
    let g = hydrate_dolce(&registry).expect("DOLCE+DUL hydrates");
    assert_eq!(g, OGIT::DOLCE_V1.0, "G slot must be DOLCE_V1");

    let bundle = registry
        .bundle_for(OGIT::DOLCE_V1.0)
        .expect("ContextBundle registered at DOLCE_V1");

    assert_eq!(bundle.g, OGIT::DOLCE_V1.0);
    assert_eq!(bundle.version, OGIT::DOLCE_V1.1);
    assert_eq!(bundle.domain_name, "dolce");
    assert_eq!(
        bundle.inherits_from, None,
        "DOLCE is the L1 root — no parent"
    );

    let entity_count = bundle.entity_count();
    assert!(
        entity_count > 200,
        "expected >200 entities, got {entity_count} (DOLCE+DUL has ~250 named entities)"
    );
}
