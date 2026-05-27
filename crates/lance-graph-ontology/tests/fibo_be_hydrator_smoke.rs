// Skip under Miri — hydration reads `data/ontologies/fibo-be/*.rdf` via std::fs.
#![cfg(not(miri))]

//! PR-bO-7 smoke test: hydrate FIBO Business Entities (BE) and assert
//! L3-business-entity invariants.

use lance_graph_contract::manifest::OGIT;
use lance_graph_ontology::{hydrate_fibo_be, OntologyRegistry};

#[test]
fn fibo_be_hydrator_smoke() {
    let registry = OntologyRegistry::new_in_memory();
    let g = hydrate_fibo_be(&registry).expect("FIBO-BE hydrates");
    assert_eq!(g, OGIT::FIBOBE_V1.0);

    let bundle = registry
        .bundle_for(OGIT::FIBOBE_V1.0)
        .expect("ContextBundle registered at FIBOBE_V1");

    assert_eq!(bundle.g, OGIT::FIBOBE_V1.0);
    assert_eq!(bundle.domain_name, "fibobe");
    assert_eq!(
        bundle.inherits_from,
        Some(OGIT::FIBOFND_V1.0),
        "FIBO-BE inherits_from must be FIBOFND (BE → FND → DOLCE, PR #416)"
    );

    let entity_count = bundle.entity_count();
    assert!(
        entity_count > 500,
        "expected >500 entities for FIBO-BE, got {entity_count}"
    );
}

#[test]
fn fibo_be_canonical_iris_resolve() {
    let registry = OntologyRegistry::new_in_memory();
    hydrate_fibo_be(&registry).expect("FIBO-BE hydrates");
    let g = OGIT::FIBOBE_V1.0;

    // Load-bearing FIBO-BE IRIs covering corporate structures (the bO-*
    // series binds against these for SKR03/SKR04 / HGB / UStG bridges).
    // `LegalEntity` / `LegalPerson` live in OMG Commons (re-imported by
    // FIBO-BE), not under BE/LegalEntities/LegalPersons/ directly — that
    // module defines `BusinessEntity` and other LegalPerson-specialised
    // classes on top.
    let load_bearing = [
        // OMG Commons re-imports
        "https://www.omg.org/spec/Commons/Organizations/LegalEntity",
        "https://www.omg.org/spec/Commons/Organizations/LegalPerson",
        // BE/LegalEntities native specialisations
        "https://spec.edmcouncil.org/fibo/ontology/BE/LegalEntities/LegalPersons/BusinessEntity",
        // Corporations
        "https://spec.edmcouncil.org/fibo/ontology/BE/Corporations/Corporations/JointStockCompany",
        "https://spec.edmcouncil.org/fibo/ontology/BE/Corporations/Corporations/PubliclyHeldCompany",
        // Partnerships
        "https://spec.edmcouncil.org/fibo/ontology/BE/Partnerships/Partnerships/Partnership",
        "https://spec.edmcouncil.org/fibo/ontology/BE/Partnerships/Partnerships/GeneralPartnership",
        "https://spec.edmcouncil.org/fibo/ontology/BE/Partnerships/Partnerships/LimitedPartnership",
        // Trusts
        "https://spec.edmcouncil.org/fibo/ontology/BE/Trusts/Trusts/Trust",
    ];
    for iri in &load_bearing {
        assert!(
            registry.resolve_iri_in(g, iri).is_some(),
            "{iri} must resolve under G={g}"
        );
    }
}

#[test]
fn fibo_be_inherits_from_fibofnd() {
    // BE now chains BE → FND → DOLCE (FND itself inherits DOLCE); re-parented
    // from dolce-direct in PR #416 (FIBU subtree under fibofnd).
    let registry = OntologyRegistry::new_in_memory();
    hydrate_fibo_be(&registry).expect("FIBO-BE hydrates");
    let bundle = registry
        .bundle_for(OGIT::FIBOBE_V1.0)
        .expect("bundle registered");
    assert_eq!(bundle.inherits_from, Some(OGIT::FIBOFND_V1.0));
}
