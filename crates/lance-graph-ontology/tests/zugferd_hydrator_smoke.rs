// Skip under Miri — hydration reads `data/ontologies/zugferd/*.xsd` via std::fs.
#![cfg(not(miri))]

//! PR-bO-16 smoke test: hydrate ZUGFeRD/Factur-X EN16931 and assert
//! L3 e-invoicing invariants.

use lance_graph_contract::manifest::OGIT;
use lance_graph_ontology::{hydrate_zugferd, OntologyRegistry};

const RSM: &str = "urn:un:unece:uncefact:data:standard:CrossIndustryInvoice:100";
const RAM: &str = "urn:un:unece:uncefact:data:standard:ReusableAggregateBusinessInformationEntity:100";
const UDT: &str = "urn:un:unece:uncefact:data:standard:UnqualifiedDataType:100";
const QDT: &str = "urn:un:unece:uncefact:data:standard:QualifiedDataType:100";

#[test]
fn zugferd_hydrator_smoke() {
    let registry = OntologyRegistry::new_in_memory();
    let g = hydrate_zugferd(&registry).expect("ZUGFeRD hydrates");
    assert_eq!(g, OGIT::ZUGFERD_V1.0);

    let bundle = registry
        .bundle_for(OGIT::ZUGFERD_V1.0)
        .expect("ContextBundle registered at ZUGFERD_V1");

    assert_eq!(bundle.g, OGIT::ZUGFERD_V1.0);
    assert_eq!(bundle.domain_name, "zugferd");
    assert_eq!(
        bundle.inherits_from,
        Some(OGIT::FIBOFND_V1.0),
        "ZUGFeRD inherits_from must be FIBOFND (PR #416)"
    );

    // EN16931 profile covers 4 XSD files: top-level CrossIndustryInvoice,
    // ReusableAggregateBusinessInformationEntity (~39 complex types),
    // QualifiedDataType, UnqualifiedDataType. Expect hundreds of entities.
    let entity_count = bundle.entity_count();
    assert!(
        entity_count > 200,
        "expected >200 entities for ZUGFeRD EN16931 (4 XSDs), got {entity_count}"
    );
}

#[test]
fn zugferd_cii_root_resolves() {
    let registry = OntologyRegistry::new_in_memory();
    hydrate_zugferd(&registry).expect("hydrate");
    let g = OGIT::ZUGFERD_V1.0;
    // The CII root element and its type — the entry point of every
    // ZUGFeRD invoice.
    assert!(
        registry
            .resolve_iri_in(g, &format!("{RSM}#CrossIndustryInvoice"))
            .is_some(),
        "CrossIndustryInvoice element must resolve"
    );
    assert!(
        registry
            .resolve_iri_in(g, &format!("{RSM}#CrossIndustryInvoiceType"))
            .is_some(),
        "CrossIndustryInvoiceType complexType must resolve"
    );
}

#[test]
fn zugferd_ram_load_bearing_types_resolve() {
    let registry = OntologyRegistry::new_in_memory();
    hydrate_zugferd(&registry).expect("hydrate");
    let g = OGIT::ZUGFERD_V1.0;
    // The load-bearing CII RAM complex types — every ZUGFeRD invoice's
    // structure binds against these.
    for type_name in &[
        "ExchangedDocumentType",
        "ExchangedDocumentContextType",
        "SupplyChainTradeTransactionType",
        "HeaderTradeAgreementType",
        "HeaderTradeDeliveryType",
        "HeaderTradeSettlementType",
        "TradePartyType",
        "TradeAddressType",
        "TradeTaxType",
        "DocumentLineDocumentType",
    ] {
        let iri = format!("{RAM}#{type_name}");
        assert!(
            registry.resolve_iri_in(g, &iri).is_some(),
            "ram:{type_name} must resolve under G={g}"
        );
    }
}

#[test]
fn zugferd_datatype_namespaces_present() {
    let registry = OntologyRegistry::new_in_memory();
    hydrate_zugferd(&registry).expect("hydrate");
    let g = OGIT::ZUGFERD_V1.0;
    // Spot-check that at least one IRI from each of the four CII
    // namespaces was interned (proves the multi-file walk picks up every
    // XSD in the directory).
    let probes = [
        format!("{RSM}#CrossIndustryInvoice"),
        format!("{RAM}#ExchangedDocumentType"),
    ];
    for iri in &probes {
        assert!(
            registry.resolve_iri_in(g, iri).is_some(),
            "{iri} must resolve"
        );
    }
    // The QDT/UDT XSDs declare types like `AmountType`, `IDType`. They
    // may use either targetNamespace; just assert at least one IRI from
    // each namespace exists by enumerating the bundle.
    let bundle = registry.bundle_for(g).expect("bundle");
    let saw_udt = bundle
        .ontology
        .as_ref()
        .map(|o| o.iri_to_id.keys().any(|k| k.starts_with(UDT)))
        .unwrap_or(false);
    let saw_qdt = bundle
        .ontology
        .as_ref()
        .map(|o| o.iri_to_id.keys().any(|k| k.starts_with(QDT)))
        .unwrap_or(false);
    assert!(saw_udt, "expected at least one IRI in UDT namespace");
    assert!(saw_qdt, "expected at least one IRI in QDT namespace");
}

#[test]
fn zugferd_edge_whitelist_registered() {
    let registry = OntologyRegistry::new_in_memory();
    hydrate_zugferd(&registry).expect("hydrate");
    let edges = registry
        .edge_types_for(OGIT::ZUGFERD_V1.0)
        .expect("whitelist registered");
    assert!(
        edges.len() >= 10,
        "expected >= 10 cascade edge IRIs, got {} ({:?})",
        edges.len(),
        edges
    );
    // The three CII top-level relational containers MUST be in the
    // whitelist or invoice projection breaks.
    for top in &[
        "ExchangedDocument",
        "ExchangedDocumentContext",
        "SupplyChainTradeTransaction",
    ] {
        let iri = format!("{RSM}#{top}");
        assert!(
            edges.iter().any(|e| e == &iri),
            "rsm:{top} must be in the cascade whitelist"
        );
    }
}
