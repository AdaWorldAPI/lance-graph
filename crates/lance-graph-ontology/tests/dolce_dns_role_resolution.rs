// Skip under Miri — hydration reads `data/ontologies/dul.ttl` via std::fs.
#![cfg(not(miri))]

//! PR-bO-1: assert the load-bearing DnS role-binding IRIs resolve.
//!
//! The cognitive shader's 16-bit DOLCE slot's low 8 bits encode the DnS role
//! per the Descriptions & Situations module. The canonical DOLCE+DUL
//! ontology defines the role hierarchy via `dul:Concept` → `dul:Role` →
//! `dul:Task` / `dul:Parameter` / `dul:Goal` / `dul:Method`. Specific
//! semantic roles (`Patient`, `Instrument`, `Location`, …) are NOT defined
//! as classes in canonical DUL — they are individuals instantiated by users
//! against `dul:Concept` and `dul:Role` at modelling time, or by DUL
//! extension modules (CoreLegal, IOLite, SystemsLite).
//!
//! This test asserts the *anchor* IRIs are present so DnS role-binding can
//! be performed at cascade-step time:
//! - `dul:Agent` — the agent-bearer of an action
//! - `dul:Concept` — the root of the DnS classification hierarchy
//! - `dul:Role` — the sub-class of Concept that classifies Objects
//! - `dul:Task` — the sub-class of Concept that classifies Actions
//! - `dul:Parameter` — the sub-class of Concept that classifies Regions
//! - `dul:Goal` / `dul:Method` / `dul:Plan` — Description sub-classes
//! - `dul:TimeInterval` — the temporal anchor for ObservableAt /
//!   hasTimeInterval cascade traversal

use lance_graph_contract::manifest::OGIT;
use lance_graph_ontology::{hydrate_dolce, OntologyRegistry};

const DUL_BASE: &str = "http://www.ontologydesignpatterns.org/ont/dul/DUL.owl#";

fn dul(name: &str) -> String {
    format!("{DUL_BASE}{name}")
}

#[test]
fn dns_role_anchor_iris_resolve() {
    let registry = OntologyRegistry::new_in_memory();
    hydrate_dolce(&registry).expect("DOLCE+DUL hydrates");

    let g = OGIT::DOLCE_V1.0;
    // The five canonical DUL classes that anchor the DnS role hierarchy.
    // Every cognitive-shader role-binding traversal must reach at least one
    // of these.
    for name in &[
        "Agent",
        "Concept",
        "Role",
        "Task",
        "Parameter",
        "Goal",
        "Method",
        "Plan",
        "TimeInterval",
    ] {
        let iri = dul(name);
        assert!(
            registry.resolve_iri_in(g, &iri).is_some(),
            "dul:{name} ({iri}) must resolve under G={g}"
        );
    }
}

#[test]
fn dns_role_anchor_concept_is_present() {
    // The DnS role hierarchy is rooted at dul:Concept (Role/Task/Parameter
    // are sub-classes of Concept). Concept itself must be present so the
    // upper-category cascade can reach it via `rdfs:subClassOf*`.
    let registry = OntologyRegistry::new_in_memory();
    hydrate_dolce(&registry).expect("DOLCE+DUL hydrates");
    let g = OGIT::DOLCE_V1.0;
    assert!(
        registry.resolve_iri_in(g, &dul("Concept")).is_some(),
        "dul:Concept must be present — it anchors the DnS role hierarchy"
    );
    assert!(
        registry.resolve_iri_in(g, &dul("Role")).is_some(),
        "dul:Role must be present (sub-class of Concept)"
    );
}

#[test]
fn dul_extension_role_iris_do_not_resolve() {
    // Canonical DOLCE+DUL does NOT define `Patient`, `Instrument`,
    // `Location`, etc. as named classes — those live in extension modules
    // (CoreLegal, IOLite, SystemsLite) and are user-defined Concept
    // individuals at modelling time. This test pins that expectation: if
    // future PRs swap in a DUL bundle that includes these as named classes,
    // it's worth flagging (likely a non-canonical fork was used).
    let registry = OntologyRegistry::new_in_memory();
    hydrate_dolce(&registry).expect("DOLCE+DUL hydrates");
    let g = OGIT::DOLCE_V1.0;
    for not_in_canonical in &["Patient", "Instrument", "Location", "Beneficiary"] {
        assert!(
            registry.resolve_iri_in(g, &dul(not_in_canonical)).is_none(),
            "dul:{not_in_canonical} is NOT a named class in canonical DUL — \
             expected an extension-module lookup or runtime Concept individual"
        );
    }
}
