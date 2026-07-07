//! DOLCE+DnS Ultralite (DUL) hydration glue.
//!
//! DOLCE is the L1 upper ontology — the root of the L1-L4 business-logic DAG.
//! It declares `inherits_from: None`. Every downstream L2/L3/L4 hydrator
//! (`hydrate_owltime`, `hydrate_provo`, `hydrate_qudt`, `hydrate_fibo_fnd`, …)
//! declares `inherits_from: Some(OGIT::DOLCE_V1.0)` and reaches the DOLCE
//! upper categories via the `rdfs:subClassOf` / `owl:equivalentClass` chains
//! the hydrator preserves.
//!
//! The cognitive shader's 16-bit DOLCE slot is *defined* relative to DOLCE's
//! upper categories: high 8 bits encode the upper-category sub-classification
//! (Object/Event/Quality/Abstract — note: canonical DOLCE+DUL renames the
//! original DOLCE-Lite-Plus `Endurant` → `Object` and `Perdurant` → `Event`
//! per the ontology header), low 8 bits encode the DnS role per the
//! Descriptions & Situations module. Until DOLCE is hydrated, the slot is
//! undefined.
//!
//! ## Edge whitelist
//!
//! DOLCE contributes the upper-category subsumption chain plus the DnS
//! role-binding edges. These are exactly the predicates the cognitive
//! shader's cascade reads at type-gating time. All 17 IRIs in
//! [`DOLCE_EDGE_WHITELIST`] are verified present in canonical DUL
//! (`http://www.ontologydesignpatterns.org/ont/dul/DUL.owl`).

use std::path::{Path, PathBuf};

use lance_graph_contract::manifest::OGIT;

use super::owl::{HydrateErr, OwlHydrator};
use crate::registry::OntologyRegistry;

/// Where the DUL TTL artifact lives, relative to the workspace root.
const DUL_TTL_RELATIVE_PATH: &str = "data/ontologies/dul.ttl";

/// DUL extension modules: small sibling ontologies that refine the DUL
/// upper-category surface. Each shares the `dul:` namespace and is
/// hydrated into the same `OGIT::DOLCE_V1` bundle so downstream consumers
/// see one transitive L1 surface.
const DUL_EXTENSION_RELATIVE_PATHS: &[&str] = &[
    "data/ontologies/dul-extensions/conceptualization.owl",
    "data/ontologies/dul-extensions/lmm-l2.owl",
];

/// Edge-IRI whitelist for the DOLCE+DUL cascade.
///
/// These are the predicates the cognitive shader's cascade will follow at the
/// upper-category tier. Per the spec recommendation (Open question 3) the list
/// is curated rather than the full ~70-property DUL surface: it covers
/// upper-category subsumption, DnS classification + role-binding, structural
/// part-of / constitution, and the DUL temporal-anchoring properties used by
/// downstream OWL-Time alignment.
const DOLCE_EDGE_WHITELIST: &[&str] = &[
    // Upper-category subsumption (RDF / OWL primitives)
    "http://www.w3.org/2000/01/rdf-schema#subClassOf",
    "http://www.w3.org/2002/07/owl#equivalentClass",
    "http://www.w3.org/2002/07/owl#disjointWith",
    // DnS classification
    "http://www.ontologydesignpatterns.org/ont/dul/DUL.owl#isClassifiedBy",
    "http://www.ontologydesignpatterns.org/ont/dul/DUL.owl#classifies",
    // DnS role-binding
    "http://www.ontologydesignpatterns.org/ont/dul/DUL.owl#hasRole",
    "http://www.ontologydesignpatterns.org/ont/dul/DUL.owl#isRoleOf",
    "http://www.ontologydesignpatterns.org/ont/dul/DUL.owl#hasParticipant",
    "http://www.ontologydesignpatterns.org/ont/dul/DUL.owl#isParticipantIn",
    "http://www.ontologydesignpatterns.org/ont/dul/DUL.owl#satisfies",
    "http://www.ontologydesignpatterns.org/ont/dul/DUL.owl#isSatisfiedBy",
    // Part-of / constitution (cross-cutting structural)
    "http://www.ontologydesignpatterns.org/ont/dul/DUL.owl#hasPart",
    "http://www.ontologydesignpatterns.org/ont/dul/DUL.owl#isPartOf",
    "http://www.ontologydesignpatterns.org/ont/dul/DUL.owl#hasConstituent",
    "http://www.ontologydesignpatterns.org/ont/dul/DUL.owl#isConstituentOf",
    // Temporal anchoring (DUL surface used by OWL-Time alignment downstream)
    "http://www.ontologydesignpatterns.org/ont/dul/DUL.owl#hasTimeInterval",
    "http://www.ontologydesignpatterns.org/ont/dul/DUL.owl#isObservableAt",
];

/// Hydrate DOLCE+DnS Ultralite as the L1 upper ontology.
///
/// Registers a [`super::owl::ContextBundle`] at `OGIT::DOLCE_V1.0` with
/// `inherits_from: None` (the L1 root, the only hydrator that declares no
/// parent). Hydrates the canonical `data/ontologies/dul.ttl` plus the
/// available DUL extension modules under `data/ontologies/dul-extensions/`
/// (Conceptualization patterns, LMM_L2 lexical metamodel) into one bundle.
///
/// After hydration the cascade whitelist is registered via
/// [`OntologyRegistry::register_edge_types`].
pub fn hydrate_dolce(registry: &OntologyRegistry) -> Result<u32, HydrateErr> {
    let dul = dul_ttl_path();
    let extensions: Vec<PathBuf> = DUL_EXTENSION_RELATIVE_PATHS
        .iter()
        .map(extension_path)
        .collect();
    let mut paths: Vec<&Path> = Vec::with_capacity(1 + extensions.len());
    paths.push(&dul);
    for ext in &extensions {
        if ext.exists() {
            paths.push(ext.as_path());
        }
    }
    hydrate_dolce_from_many(&paths, registry)
}

/// Test-friendly variant: hydrate DOLCE from an explicit single path
/// (typically a test fixture). Use [`hydrate_dolce_from_many`] when you want
/// to mix the canonical DUL.ttl with extension modules.
pub fn hydrate_dolce_from(ttl_path: &Path, registry: &OntologyRegistry) -> Result<u32, HydrateErr> {
    hydrate_dolce_from_many(&[ttl_path], registry)
}

/// Multi-file variant: hydrate DOLCE+DUL+extensions from explicit paths.
/// Used by [`hydrate_dolce`] to merge the canonical DUL with sibling
/// extension modules, and by tests that need to compose a custom bundle.
pub fn hydrate_dolce_from_many(
    paths: &[&Path],
    registry: &OntologyRegistry,
) -> Result<u32, HydrateErr> {
    let hydrator = OwlHydrator {
        g: OGIT::DOLCE_V1.0,
        version: OGIT::DOLCE_V1.1,
        domain_name: "dolce".to_string(),
        inherits_from: None,
        starting_entity_id: 100,
    };
    hydrator.hydrate_many(paths, registry)?;
    registry
        .register_edge_types(OGIT::DOLCE_V1.0, DOLCE_EDGE_WHITELIST)
        .map_err(|reason| HydrateErr::Registry {
            g: OGIT::DOLCE_V1.0,
            reason,
        })?;
    Ok(OGIT::DOLCE_V1.0)
}

fn dul_ttl_path() -> PathBuf {
    // `CARGO_MANIFEST_DIR` for this crate is `crates/lance-graph-ontology`;
    // the data file lives at `<workspace>/data/ontologies/dul.ttl`.
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
        .join(DUL_TTL_RELATIVE_PATH)
}

fn extension_path(rel: &&str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
        .join(rel)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dul_ttl_path_resolves_to_workspace_data() {
        let p = dul_ttl_path();
        // We don't assert existence here (the actual file presence is checked
        // by integration tests); only that the path shape is right.
        assert!(
            p.ends_with("data/ontologies/dul.ttl"),
            "unexpected path tail: {}",
            p.display()
        );
    }

    #[test]
    fn dolce_edge_whitelist_has_load_bearing_minimum() {
        assert!(
            DOLCE_EDGE_WHITELIST.len() >= 15,
            "whitelist below the load-bearing minimum"
        );
    }
}
