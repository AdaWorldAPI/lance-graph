// Skip under Miri — hydration reads the .sch file via std::fs.
#![cfg(not(miri))]

//! PR-bO-15 smoke test: hydrate the ZUGFeRD/Factur-X EN16931 Schematron
//! business-rule namespace and assert load-bearing invariants.

use lance_graph_contract::manifest::OGIT;
use lance_graph_ontology::{hydrate_zugferd_rules, OntologyRegistry};

const BASE: &str = "urn:schematron:factur-x-1.08-en16931";

#[test]
fn zugferd_rules_hydrator_smoke() {
    let registry = OntologyRegistry::new_in_memory();
    let g = hydrate_zugferd_rules(&registry).expect("ZUGFeRD rules hydrate");
    assert_eq!(g, OGIT::ZUGFERDRULES_V1.0);

    let bundle = registry
        .bundle_for(OGIT::ZUGFERDRULES_V1.0)
        .expect("ContextBundle registered at ZUGFERDRULES_V1");

    assert_eq!(bundle.g, OGIT::ZUGFERDRULES_V1.0);
    assert_eq!(bundle.domain_name, "zugferd-rules");
    assert_eq!(
        bundle.inherits_from,
        Some(OGIT::ZUGFERD_V1.0),
        "rules must inherit from structural ZUGFeRD slot"
    );

    // Actual shape (per inspection of the .sch file):
    //   - 428 <assert> elements with @id, BUT only 301 distinct IDs
    //     (assertion IDs are reused across patterns).
    //   - 191 <report> elements, NONE with @id (Schematron-1 style where
    //     <report> declares the inverse of <assert>; the message body is
    //     still scanned for bracketed business-rule IDs).
    //   - 0 <pattern> elements with @id (most patterns are anonymous).
    //   - ~209 distinct bracketed business-rule IDs across all message
    //     bodies (BR-NN / BR-CO-NN / BR-DEC-NN / BR-S/Z/E/...-NN / PEPPOL-*).
    // Expected total: 301 + ~209 = ~510 IRIs.
    let entity_count = bundle.entity_count();
    assert!(
        entity_count >= 500,
        "expected >=500 rule IRIs for ZUGFeRD EN16931, got {entity_count}"
    );
}

#[test]
fn zugferd_rules_canonical_en16931_anchors_resolve() {
    let registry = OntologyRegistry::new_in_memory();
    hydrate_zugferd_rules(&registry).expect("hydrate");
    let g = OGIT::ZUGFERDRULES_V1.0;

    // Canonical EN16931 anchor rules — these MUST exist or invoice
    // validation alignment to the EU directive is broken. Probes drawn
    // from the load-bearing rules in the .sch file.
    for rule in &[
        "BR-52",     // BG-24 supporting document reference
        "BR-45",     // BG-23 VAT category taxable amount
        "BR-46",     // BG-23 VAT category tax amount
        "BR-47",     // BG-23 VAT category code
        "BR-CO-03",  // VAT point date / code mutually exclusive
        "BR-CO-17",  // VAT category tax amount = taxable × rate
        "BR-S-08",   // Standard rated VAT breakdown sum
        "BR-Z-08",   // Zero rated VAT breakdown sum
        "BR-DEC-19", // Max 2 decimals on taxable amount
    ] {
        let iri = format!("{BASE}/rule/{rule}");
        assert!(
            registry.resolve_iri_in(g, &iri).is_some(),
            "EN16931 anchor rule {rule} must resolve at {iri}"
        );
    }
}

#[test]
fn zugferd_rules_peppol_and_de_extensions_present() {
    let registry = OntologyRegistry::new_in_memory();
    hydrate_zugferd_rules(&registry).expect("hydrate");
    let g = OGIT::ZUGFERDRULES_V1.0;

    // The PEPPOL extension and DE (German) extension business-rule
    // namespaces both appear in the EN16931 Schematron. At minimum,
    // PEPPOL-EN16931-R008 (the empty-element warning, present in the
    // very first <assert> in the .sch file) must resolve.
    assert!(
        registry
            .resolve_iri_in(g, &format!("{BASE}/rule/PEPPOL-EN16931-R008"))
            .is_some(),
        "PEPPOL-EN16931-R008 must resolve"
    );
}

#[test]
fn zugferd_rules_schema_assertion_ids_resolve() {
    let registry = OntologyRegistry::new_in_memory();
    hydrate_zugferd_rules(&registry).expect("hydrate");
    let g = OGIT::ZUGFERDRULES_V1.0;

    // FX-SCH-A-* schema assertion IDs come from the actual Schematron
    // assert element IDs. KoSIT validator output references these
    // directly, so downstream code needs IRIs for them.
    //
    // Spot-checks drawn from the .sch file's first patterns:
    //   FX-SCH-A-000372 — the empty-element PEPPOL warning
    //   FX-SCH-A-000280 — BG-24 supporting document reference (BR-52)
    //   FX-SCH-A-000047 — BG-23 VAT taxable amount (BR-45)
    for id in &[
        "FX-SCH-A-000372",
        "FX-SCH-A-000280",
        "FX-SCH-A-000047",
        "FX-SCH-A-000048",
        "FX-SCH-A-000049",
    ] {
        let iri = format!("{BASE}/assert/{id}");
        assert!(
            registry.resolve_iri_in(g, &iri).is_some(),
            "schema assertion {id} must resolve at {iri}"
        );
    }
}

#[test]
fn zugferd_rules_distinct_business_rule_namespace() {
    // The hydrator should produce IRIs spread across multiple business-
    // rule families: BR-NN (EN16931 core), BR-CO-NN (calculation/codelist
    // co-occurrence), BR-DEC-NN (decimal restrictions), BR-S/Z/E/AE/G/IC/
    // K/L/M/O-NN (VAT category-specific), and PEPPOL-EN16931-*.
    let registry = OntologyRegistry::new_in_memory();
    hydrate_zugferd_rules(&registry).expect("hydrate");
    let bundle = registry
        .bundle_for(OGIT::ZUGFERDRULES_V1.0)
        .expect("bundle");
    let ontology = bundle.ontology.as_ref().expect("ontology slot present");

    let rule_iris: Vec<&String> = ontology
        .iri_to_id
        .keys()
        .filter(|k| k.starts_with(&format!("{BASE}/rule/")))
        .collect();

    let saw = |prefix: &str| -> bool {
        rule_iris.iter().any(|iri| {
            iri.strip_prefix(&format!("{BASE}/rule/"))
                .map(|s| s.starts_with(prefix))
                .unwrap_or(false)
        })
    };

    assert!(
        saw("BR-01") || saw("BR-02"),
        "EN16931 BR-NN core rules missing"
    );
    assert!(saw("BR-CO-"), "BR-CO-NN co-occurrence rules missing");
    assert!(
        saw("BR-DEC-"),
        "BR-DEC-NN decimal-restriction rules missing"
    );
    assert!(saw("BR-S-"), "BR-S-NN standard-rated VAT rules missing");
    assert!(saw("BR-Z-"), "BR-Z-NN zero-rated VAT rules missing");
    assert!(saw("PEPPOL-"), "PEPPOL extension rules missing");

    // At minimum 150 distinct business-rule IDs should appear (we measured
    // 209 distinct ones across the EN16931 profile).
    assert!(
        rule_iris.len() >= 150,
        "expected >= 150 distinct business-rule IRIs, got {}",
        rule_iris.len()
    );
}
