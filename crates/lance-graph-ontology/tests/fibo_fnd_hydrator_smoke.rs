// Skip under Miri — hydration reads `data/ontologies/fibo-fnd/*.rdf` via std::fs.
#![cfg(not(miri))]

//! PR-bO-6 smoke test: hydrate FIBO Foundations (FND) and assert
//! L3-financial-foundation invariants.
//!
//! FIBO ships as ~59 RDF/XML files under data/ontologies/fibo-fnd/.
//! `OwlHydrator::hydrate_many` walks the tree and dispatches via
//! `oxrdfxml` (RDF/XML) per file.

use lance_graph_contract::manifest::OGIT;
use lance_graph_ontology::{hydrate_fibo_fnd, OntologyRegistry};

#[test]
fn fibo_fnd_hydrator_smoke() {
    let registry = OntologyRegistry::new_in_memory();
    let g = hydrate_fibo_fnd(&registry).expect("FIBO-FND hydrates");
    assert_eq!(g, OGIT::FIBOFND_V1.0);

    let bundle = registry
        .bundle_for(OGIT::FIBOFND_V1.0)
        .expect("ContextBundle registered at FIBOFND_V1");

    assert_eq!(bundle.g, OGIT::FIBOFND_V1.0);
    assert_eq!(bundle.domain_name, "fibofnd");
    assert_eq!(
        bundle.inherits_from,
        Some(OGIT::DOLCE_V1.0),
        "FIBO-FND inherits_from must be DOLCE"
    );

    // FIBO-FND has hundreds of named classes/properties across its modules.
    let entity_count = bundle.entity_count();
    assert!(
        entity_count > 1000,
        "expected >1000 entities for FIBO-FND, got {entity_count}"
    );
}

#[test]
fn fibo_fnd_canonical_iris_resolve() {
    let registry = OntologyRegistry::new_in_memory();
    hydrate_fibo_fnd(&registry).expect("FIBO-FND hydrates");
    let g = OGIT::FIBOFND_V1.0;

    // Load-bearing FIBO-FND IRIs that downstream FIBO modules and bO-*
    // bridges declare against. Note: many "Party"-flavored concepts live
    // under OMG Commons (cmns-*) since FIBO re-imports the OMG foundation;
    // the load-bearing test asserts both the OMG Commons re-imports AND the
    // FIBO-native FND classes resolve.
    let load_bearing = [
        // OMG Commons (re-imported by FND, used across every FIBO module)
        "https://www.omg.org/spec/Commons/Organizations/LegalPerson",
        // Agents and people
        "https://spec.edmcouncil.org/fibo/ontology/FND/AgentsAndPeople/People/Person",
        // Currency / monetary amounts (FND/Accounting)
        "https://spec.edmcouncil.org/fibo/ontology/FND/Accounting/CurrencyAmount/MonetaryAmount",
        "https://spec.edmcouncil.org/fibo/ontology/FND/Accounting/CurrencyAmount/Currency",
        "https://spec.edmcouncil.org/fibo/ontology/FND/Accounting/CurrencyAmount/AmountOfMoney",
        "https://spec.edmcouncil.org/fibo/ontology/FND/Accounting/CurrencyAmount/ExchangeRate",
        "https://spec.edmcouncil.org/fibo/ontology/FND/Accounting/CurrencyAmount/InterestRate",
    ];
    for iri in &load_bearing {
        assert!(
            registry.resolve_iri_in(g, iri).is_some(),
            "{iri} must resolve under G={g}"
        );
    }
}

#[test]
fn fibo_fnd_edge_whitelist_registered() {
    let registry = OntologyRegistry::new_in_memory();
    hydrate_fibo_fnd(&registry).expect("FIBO-FND hydrates");
    let edges = registry
        .edge_types_for(OGIT::FIBOFND_V1.0)
        .expect("whitelist registered");
    assert!(
        edges.len() >= 15,
        "expected >= 15 cascade edge IRIs, got {} ({:?})",
        edges.len(),
        edges
    );
    assert!(
        edges
            .iter()
            .any(|e| e == "https://www.omg.org/spec/Commons/Identifiers/hasIdentifier"),
        "FIBO Commons hasIdentifier must be in the whitelist"
    );
}
