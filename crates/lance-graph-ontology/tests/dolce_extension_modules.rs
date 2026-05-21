// Skip under Miri — hydration reads the DUL extension files via std::fs.
#![cfg(not(miri))]

//! PR-bO-1+: assert that the DUL extension modules (Conceptualization,
//! LMM_L2) merge into the OGIT::DOLCE_V1 bundle and contribute their
//! canonical IRIs.

use lance_graph_contract::manifest::OGIT;
use lance_graph_ontology::{hydrate_dolce, OntologyRegistry};

#[test]
fn dul_extension_conceptualization_classes_resolve() {
    let registry = OntologyRegistry::new_in_memory();
    hydrate_dolce(&registry).expect("DOLCE+DUL+extensions hydrate");
    let g = OGIT::DOLCE_V1.0;

    // Conceptualization extension defines these load-bearing classes /
    // properties for agent-belief-knowledge cascades.
    for iri in &[
        // Class
        "http://www.ontologydesignpatterns.org/ont/dul/Conceptualization.owl#InternalRepresentation",
        // Object properties (cognitive-shader agency cascade)
        "http://www.ontologydesignpatterns.org/ont/dul/Conceptualization.owl#knows",
        "http://www.ontologydesignpatterns.org/ont/dul/Conceptualization.owl#isKnownBy",
        "http://www.ontologydesignpatterns.org/ont/dul/Conceptualization.owl#believes",
        "http://www.ontologydesignpatterns.org/ont/dul/Conceptualization.owl#isBelievedBy",
        "http://www.ontologydesignpatterns.org/ont/dul/Conceptualization.owl#assumes",
        "http://www.ontologydesignpatterns.org/ont/dul/Conceptualization.owl#isAssumedBy",
        "http://www.ontologydesignpatterns.org/ont/dul/Conceptualization.owl#adopts",
        "http://www.ontologydesignpatterns.org/ont/dul/Conceptualization.owl#isAdoptedBy",
    ] {
        assert!(
            registry.resolve_iri_in(g, iri).is_some(),
            "Conceptualization IRI must resolve under G={g}: {iri}"
        );
    }
}

#[test]
fn dul_extension_lmm_l2_classes_resolve() {
    let registry = OntologyRegistry::new_in_memory();
    hydrate_dolce(&registry).expect("DOLCE+DUL+extensions hydrate");
    let g = OGIT::DOLCE_V1.0;

    // LMM_L2 — Lexical MetaModel L2 — adds named-entity / concept-reference /
    // syntactic-context surface used by NER, sense-tagging, and grammar
    // cascades. Each of these is defined as a class in the extension file.
    for iri in &[
        "http://www.ontologydesignpatterns.org/ont/lmm/LMM_L2.owl#NamedEntity",
        "http://www.ontologydesignpatterns.org/ont/lmm/LMM_L2.owl#Name",
        "http://www.ontologydesignpatterns.org/ont/lmm/LMM_L2.owl#ConceptExpression",
        "http://www.ontologydesignpatterns.org/ont/lmm/LMM_L2.owl#ConceptReference",
        "http://www.ontologydesignpatterns.org/ont/lmm/LMM_L2.owl#ContextualExpression",
        "http://www.ontologydesignpatterns.org/ont/lmm/LMM_L2.owl#IndividualReference",
        "http://www.ontologydesignpatterns.org/ont/lmm/LMM_L2.owl#MultipleReference",
        "http://www.ontologydesignpatterns.org/ont/lmm/LMM_L2.owl#Gloss",
        "http://www.ontologydesignpatterns.org/ont/lmm/LMM_L2.owl#hasSyntacticFunction",
        "http://www.ontologydesignpatterns.org/ont/lmm/LMM_L2.owl#hasInstance",
        "http://www.ontologydesignpatterns.org/ont/lmm/LMM_L2.owl#isInstanceOf",
    ] {
        assert!(
            registry.resolve_iri_in(g, iri).is_some(),
            "LMM_L2 IRI must resolve under G={g}: {iri}"
        );
    }
}

#[test]
fn dolce_entity_count_grows_with_extensions() {
    // The base DOLCE+DUL has ~196 named entities. Conceptualization and
    // LMM_L2 add roughly 10 + 25 = 35 more on top, plus the LMM_L1 / IOLite
    // IRIs they reference. The merged bundle should comfortably exceed the
    // base smoke-test threshold of 200.
    let registry = OntologyRegistry::new_in_memory();
    hydrate_dolce(&registry).expect("DOLCE+DUL+extensions hydrate");
    let bundle = registry.bundle_for(OGIT::DOLCE_V1.0).expect("bundle");
    assert!(
        bundle.entity_count() >= 240,
        "expected >=240 entities with extensions, got {}",
        bundle.entity_count()
    );
}
