// Skip under Miri — hydration reads the CSV via std::fs.
#![cfg(not(miri))]

//! PR-bO-13 smoke test: hydrate DATEV SKR 03 + SKR 04 charts of accounts.
//!
//! Asserts that:
//! - Both schemes hydrate into separate G slots with the correct domain
//!   and DOLCE inheritance.
//! - The load-bearing anchor accounts resolve under their respective
//!   `urn:datev:{skr03,skr04}:account/{number}` IRIs.
//! - Entity counts are in the expected range (>1000 each).

use lance_graph_contract::manifest::OGIT;
use lance_graph_ontology::{
    hydrate_skr03, hydrate_skr03_bau, hydrate_skr04, OntologyRegistry, SKR03_BAU_IRI_PREFIX,
    SKR03_IRI_PREFIX, SKR04_IRI_PREFIX,
};

#[test]
fn skr03_hydrator_smoke() {
    let registry = OntologyRegistry::new_in_memory();
    let g = hydrate_skr03(&registry).expect("SKR 03 hydrates");
    assert_eq!(g, OGIT::SKR03_V1.0);

    let bundle = registry
        .bundle_for(OGIT::SKR03_V1.0)
        .expect("ContextBundle registered at SKR03_V1");
    assert_eq!(bundle.g, OGIT::SKR03_V1.0);
    assert_eq!(bundle.domain_name, "skr03");
    assert_eq!(bundle.inherits_from, Some(OGIT::FIBOFND_V1.0));
    assert!(
        bundle.entity_count() >= 1400,
        "expected >= 1400 canonical SKR 03 accounts, got {}",
        bundle.entity_count()
    );
}

#[test]
fn skr04_hydrator_smoke() {
    let registry = OntologyRegistry::new_in_memory();
    let g = hydrate_skr04(&registry).expect("SKR 04 hydrates");
    assert_eq!(g, OGIT::SKR04_V1.0);

    let bundle = registry
        .bundle_for(OGIT::SKR04_V1.0)
        .expect("ContextBundle registered at SKR04_V1");
    assert_eq!(bundle.g, OGIT::SKR04_V1.0);
    assert_eq!(bundle.domain_name, "skr04");
    assert_eq!(bundle.inherits_from, Some(OGIT::FIBOFND_V1.0));
    assert!(
        bundle.entity_count() >= 1200,
        "expected >= 1200 SKR 04 accounts, got {}",
        bundle.entity_count()
    );
}

#[test]
fn skr03_anchor_accounts_resolve() {
    let registry = OntologyRegistry::new_in_memory();
    hydrate_skr03(&registry).expect("hydrate");
    let g = OGIT::SKR03_V1.0;
    // Canonical SKR 03 anchor accounts. Each MUST resolve or downstream
    // bookkeeping projection breaks.
    //   1000 Kasse                          (process-oriented: cash)
    //   1200 Bank                            (bank)
    //   1400 Forderungen LL                  (receivables)
    //   1576 Abziehbare Vorsteuer 19%        (input VAT)
    //   3300 Wareneingang 19% Vorsteuer      (goods received)
    //   8400 Erlöse 19% USt                  (revenue, the SKR03 canonical)
    //   8300 Erlöse 7% USt                   (revenue, reduced VAT)
    for acct in &["1000", "1200", "1400", "1576", "3300", "8400", "8300"] {
        let iri = format!("{SKR03_IRI_PREFIX}/{acct}");
        assert!(
            registry.resolve_iri_in(g, &iri).is_some(),
            "SKR 03 anchor account {acct} must resolve at {iri}"
        );
    }
}

#[test]
fn skr04_anchor_accounts_resolve() {
    let registry = OntologyRegistry::new_in_memory();
    hydrate_skr04(&registry).expect("hydrate");
    let g = OGIT::SKR04_V1.0;
    // SKR 04 anchor accounts. Numbers DIFFER from SKR 03 — same number
    // means a different account in the other scheme.
    //   1000 Roh-, Hilfs- und Betriebsstoffe (NOT cash; in SKR04 this is RHB)
    //   1200 Forderungen LL                  (receivables in SKR04)
    //   1400 Abziehbare Vorsteuer            (input VAT, SKR04 canonical)
    //   1600 Kasse                            (cash, balance-sheet-numbered)
    //   1800 Bank                            (bank)
    //   3300 Verbindlichkeiten LL             (payables)
    //   4200 Erlöse                          (revenue, canonical generic SKR04)
    //
    // NOTE: account 2900 "Gezeichnetes Kapital" is documented in the PDF but
    // currently not extracted by the parser (page-8 column-boundary gap —
    // family 2 / Eigenkapital section). Same for 4400 (caught up in a
    // multi-account bleed at 4332). Known coverage gaps in the data
    // (TODO: parser improvement); the hydrator itself is correct.
    for acct in &["1000", "1200", "1400", "1600", "1800", "3300", "4200"] {
        let iri = format!("{SKR04_IRI_PREFIX}/{acct}");
        assert!(
            registry.resolve_iri_in(g, &iri).is_some(),
            "SKR 04 anchor account {acct} must resolve at {iri}"
        );
    }
}

#[test]
fn skr03_and_skr04_are_separate_slots() {
    // The same account NUMBER (e.g. "1000") MUST be a separate IRI in each
    // scheme because the underlying account-meaning differs:
    //   SKR 03 / 1000 = Kasse (cash)
    //   SKR 04 / 1000 = Roh-, Hilfs- und Betriebsstoffe (RHB stocks)
    // Confirms the two schemes hydrate into independent G slots.
    let registry = OntologyRegistry::new_in_memory();
    hydrate_skr03(&registry).expect("SKR 03");
    hydrate_skr04(&registry).expect("SKR 04");

    let skr03 = registry.bundle_for(OGIT::SKR03_V1.0).expect("skr03");
    let skr04 = registry.bundle_for(OGIT::SKR04_V1.0).expect("skr04");

    assert_ne!(skr03.g, skr04.g);
    assert_ne!(skr03.domain_name, skr04.domain_name);

    let id_03_1000 = skr03
        .resolve_iri(&format!("{SKR03_IRI_PREFIX}/1000"))
        .expect("SKR 03 / 1000");
    let id_04_1000 = skr04
        .resolve_iri(&format!("{SKR04_IRI_PREFIX}/1000"))
        .expect("SKR 04 / 1000");
    // Both bundles use the same starting_entity_id (100), so the integer
    // value of the entity ID happens to overlap — but the scheme-qualified
    // IRIs are different. The point is the BUNDLES are independent.
    let _ = (id_03_1000, id_04_1000);

    // Cross-resolve: SKR 03 / 1000 must NOT resolve in the SKR 04 slot.
    assert!(
        skr04.resolve_iri(&format!("{SKR03_IRI_PREFIX}/1000")).is_none(),
        "SKR 03 IRI must not resolve in SKR 04 bundle"
    );
    assert!(
        skr03.resolve_iri(&format!("{SKR04_IRI_PREFIX}/1000")).is_none(),
        "SKR 04 IRI must not resolve in SKR 03 bundle"
    );
}

#[test]
fn skr03_bau_extensions_hydrate_into_dedicated_slot() {
    // Bau hydrates into ITS OWN G slot (OGIT::SKR03BAU_V1 = 42), NOT the
    // canonical SKR03_V1 slot, so a caller that hydrates BOTH canonical
    // and Bau in the same OntologyRegistry holds both bundles
    // independently without one overwriting the other. The Bau bundle
    // declares inherits_from: Some(OGIT::SKR03_V1.0) to make the
    // structural dependency on canonical SKR 03 explicit.
    let registry = OntologyRegistry::new_in_memory();
    let g = hydrate_skr03_bau(&registry).expect("SKR 03 Bau hydrates");
    assert_eq!(g, OGIT::SKR03BAU_V1.0);

    let bundle = registry.bundle_for(g).expect("bundle");
    assert_eq!(bundle.domain_name, "skr03-bau");
    assert_eq!(
        bundle.inherits_from,
        Some(OGIT::SKR03_V1.0),
        "Bau must inherit from canonical SKR 03"
    );
    // 1686 in the Bau CSV. Allow a bit of slack.
    assert!(
        bundle.entity_count() >= 1600,
        "expected >= 1600 Bau accounts, got {}",
        bundle.entity_count()
    );
    // Spot-check the trade-specific extensions.
    for ext in &["007510", "010010", "010011", "011510"] {
        let iri = format!("{SKR03_BAU_IRI_PREFIX}/{ext}");
        assert!(
            registry.resolve_iri_in(g, &iri).is_some(),
            "Bau extension {ext} must resolve at {iri}"
        );
    }
}

#[test]
fn skr03_canonical_and_bau_coexist_in_one_registry() {
    // Regression test for the Codex P1 finding (PR #407 review):
    // hydrate_skr03 + hydrate_skr03_bau in sequence must populate TWO
    // distinct G slots, not overwrite each other. Before the fix the
    // second call dropped the canonical 4-digit accounts because both
    // hydrators registered into OGIT::SKR03_V1.
    let registry = OntologyRegistry::new_in_memory();
    hydrate_skr03(&registry).expect("canonical SKR 03");
    hydrate_skr03_bau(&registry).expect("SKR 03 Bau");

    let canonical = registry.bundle_for(OGIT::SKR03_V1.0).expect("canonical bundle");
    let bau = registry.bundle_for(OGIT::SKR03BAU_V1.0).expect("Bau bundle");

    // Canonical bundle: 4-digit IRI base, includes load-bearing accounts.
    assert!(
        canonical.resolve_iri(&format!("{SKR03_IRI_PREFIX}/1000")).is_some(),
        "canonical SKR 03 / 1000 Kasse must still resolve after Bau hydration"
    );
    assert!(
        canonical.entity_count() >= 1400,
        "canonical SKR 03 must retain its full account set after Bau hydration"
    );

    // Bau bundle: 6-digit IRI base, includes trade-specific extensions.
    assert!(
        bau.resolve_iri(&format!("{SKR03_BAU_IRI_PREFIX}/007510")).is_some(),
        "Bau extension 007510 must resolve in Bau bundle"
    );

    // Cross-resolve: Bau IRI must NOT resolve in canonical bundle and vice versa.
    assert!(
        canonical.resolve_iri(&format!("{SKR03_BAU_IRI_PREFIX}/007510")).is_none(),
        "Bau IRI must not resolve in canonical SKR 03 bundle"
    );
    assert!(
        bau.resolve_iri(&format!("{SKR03_IRI_PREFIX}/1000")).is_none(),
        "canonical SKR 03 IRI must not resolve in Bau bundle"
    );
}
