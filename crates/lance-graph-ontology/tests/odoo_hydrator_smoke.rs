// Skip under Miri — hydration reads the seed TTL via std::fs, and Miri
// isolation blocks the syscalls.
#![cfg(not(miri))]

//! Four-way-seam odoo Layer-1 smoke test: hydrate the odoo core seed and assert
//! the bundle invariants.
//!
//! - bundle exists at `OGIT::ODOO_V1.0`
//! - `inherits_from == Some(OGIT::FIBOFND_V1.0)` (odoo reaches the financial
//!   ontology through FIBO Foundations — Seam decision 1 / Option B)
//! - `domain_name == "odoo"`
//! - `entity_count > 0` over the seed TTL
//! - the cascade whitelist carries the two load-bearing predicates
//!   (`rdfs:subClassOf` + `owl:equivalentClass`)

use std::path::Path;

use lance_graph_contract::manifest::OGIT;
use lance_graph_ontology::{hydrate_odoo, hydrate_odoo_from, OntologyRegistry};

fn odoo_core_ttl() -> std::path::PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
        .join("data/ontologies/odoo/odoo-core.ttl")
}

#[test]
fn odoo_hydrator_smoke_from_seed() {
    let registry = OntologyRegistry::new_in_memory();
    let core = odoo_core_ttl();
    let g = hydrate_odoo_from(&[core.as_path()], &registry).expect("odoo seed hydrates");
    assert_eq!(g, OGIT::ODOO_V1.0, "G slot must be ODOO_V1");

    let bundle = registry
        .bundle_for(OGIT::ODOO_V1.0)
        .expect("ContextBundle registered at ODOO_V1");

    assert_eq!(bundle.g, OGIT::ODOO_V1.0);
    assert_eq!(bundle.version, OGIT::ODOO_V1.1);
    assert_eq!(bundle.domain_name, "odoo");
    assert_eq!(
        bundle.inherits_from,
        Some(OGIT::FIBOFND_V1.0),
        "odoo inherits from FIBO Foundations (Option B)"
    );

    let entity_count = bundle.entity_count();
    assert!(
        entity_count > 0,
        "expected a non-zero entity count from the odoo seed, got {entity_count}"
    );

    // A couple of seed classes must be resolvable to interned u32 ids.
    assert!(
        bundle
            .resolve_iri("https://ada.world/onto/odoo#res.partner")
            .is_some(),
        "odoo:res.partner must be interned"
    );
    assert!(
        bundle
            .resolve_iri("https://ada.world/onto/odoo#account.move")
            .is_some(),
        "odoo:account.move must be interned"
    );
}

#[test]
fn odoo_edge_whitelist_registered() {
    let registry = OntologyRegistry::new_in_memory();
    let core = odoo_core_ttl();
    hydrate_odoo_from(&[core.as_path()], &registry).expect("odoo seed hydrates");

    let edges = registry
        .edge_types_for(OGIT::ODOO_V1.0)
        .expect("edge whitelist registered for ODOO_V1");
    assert!(
        edges
            .iter()
            .any(|e| e == "http://www.w3.org/2000/01/rdf-schema#subClassOf"),
        "rdfs:subClassOf must be whitelisted"
    );
    assert!(
        edges
            .iter()
            .any(|e| e == "http://www.w3.org/2002/07/owl#equivalentClass"),
        "owl:equivalentClass must be whitelisted"
    );
}

#[test]
fn odoo_hydrator_smoke_canonical_paths() {
    // The canonical entry point reads the seed + the alignment overlays from
    // the workspace data dir. Asserts the production path is wired correctly.
    let registry = OntologyRegistry::new_in_memory();
    let g = hydrate_odoo(&registry).expect("odoo hydrates from canonical paths");
    assert_eq!(g, OGIT::ODOO_V1.0);

    let bundle = registry.bundle_for(OGIT::ODOO_V1.0).expect("bundle registered");
    // Seed alone is ~20 classes; with the FIBO + SKR alignment overlays the
    // interned count is strictly larger (pivot IRIs get interned too).
    assert!(
        bundle.entity_count() > 0,
        "canonical hydration must intern a non-zero entity count"
    );
    // The FIBO pivot from the alignment overlay must be interned.
    assert!(
        bundle
            .resolve_iri(
                "https://spec.edmcouncil.org/fibo/ontology/BE/LegalEntities/LegalPersons/LegalEntity"
            )
            .is_some(),
        "the fibo:LegalEntity alignment pivot must be interned from odoo-to-fibo.ttl"
    );
}
