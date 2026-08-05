//! Odoo business-model hydration glue — Layer 1 of the four-way alignment seam.
//!
//! Per the seam spec (`woa-rs/.claude/reference/four_way_alignment_seam.md`),
//! odoo is an *extraction source*, not a domain. The Rust odoo extractor parses
//! `odoo/addons/<module>/models/*.py` ASTs and emits TTL in the `odoo:`
//! namespace (`https://ada.world/onto/odoo#`); this hydrator interns those
//! classes into a [`super::owl::ContextBundle`] keyed by `OGIT::ODOO_V1`.
//!
//! ## Where odoo lands in the L1-L4 stack
//!
//! This hydrator declares `inherits_from: Some(OGIT::FIBOFND_V1.0)` — odoo
//! reaches the financial ontology through FIBO Foundations. It does **NOT**
//! get its own CAM codebook family (Seam decision 1 / Option B): every odoo
//! class with an alignment axiom is `owl:equivalentClass`-routed into an
//! existing FIBO/SKR slot, so odoo content lands in the right domain
//! compartment by virtue of its alignment, and the type lattice stays unified.
//! The alignment axioms live in `data/ontologies/odoo/alignment/`
//! (`odoo-to-fibo.ttl`, `odoo-to-skr.ttl`) and are hydrated into the same
//! bundle so the cascade sees one transitive odoo surface.
//!
//! ## Edge whitelist
//!
//! The cascade follows `rdfs:subClassOf` (odoo's own facet subsumption, e.g.
//! `odoo:res.partner.Company ⊑ odoo:res.partner`) and `owl:equivalentClass`
//! (the Layer-2 alignment pivots that carry CAM-codebook resolution). Those
//! two are load-bearing and MUST be present; `rdfs:subPropertyOf` /
//! `owl:equivalentProperty` cover the field-level alignments.
//!
//! ## DOLCE category
//!
//! The DOLCE upper-category marker for an odoo class is NOT stored here — it is
//! computed by the suffix classifier in [`super::dolce_odoo::classify_odoo`]
//! (Seam decision 2, kept in its own module per Open-question 3).

use std::path::{Path, PathBuf};

use lance_graph_contract::manifest::OGIT;

use super::owl::{HydrateErr, OwlHydrator};
use crate::registry::OntologyRegistry;

/// Core odoo class seed, relative to the workspace root.
const ODOO_CORE_RELATIVE_PATH: &str = "data/ontologies/odoo/odoo-core.ttl";

/// Layer-2 alignment overlays hydrated into the same `OGIT::ODOO_V1` bundle so
/// the cascade sees odoo classes and their `owl:equivalentClass` pivots as one
/// transitive surface. Missing files are skipped (the seed alone still
/// hydrates), mirroring `dolce.rs`'s extension-module handling.
const ODOO_ALIGNMENT_RELATIVE_PATHS: &[&str] = &[
    "data/ontologies/odoo/alignment/odoo-to-fibo.ttl",
    "data/ontologies/odoo/alignment/odoo-to-skr.ttl",
];

/// Cascade edge-IRI whitelist for the odoo surface.
///
/// `rdfs:subClassOf` carries odoo's own facet subsumption; `owl:equivalentClass`
/// carries the Layer-2 alignment pivots that resolve CAM-codebook placement
/// through the FIBO/SKR slot the odoo class is equivalent to. The property
/// variants cover the field-level alignments (`odoo:res.partner.name`
/// `owl:equivalentProperty` `foaf:name`, etc.).
const ODOO_EDGE_WHITELIST: &[&str] = &[
    // odoo facet subsumption (load-bearing — REQUIRED)
    "http://www.w3.org/2000/01/rdf-schema#subClassOf",
    // Layer-2 alignment pivots (load-bearing — REQUIRED)
    "http://www.w3.org/2002/07/owl#equivalentClass",
    // Field-level alignments
    "http://www.w3.org/2000/01/rdf-schema#subPropertyOf",
    "http://www.w3.org/2002/07/owl#equivalentProperty",
];

/// Hydrate the odoo business-model surface as `OGIT::ODOO_V1` (Layer 1).
///
/// Registers a [`super::owl::ContextBundle`] at `OGIT::ODOO_V1.0` with
/// `inherits_from: Some(OGIT::FIBOFND_V1.0)`. Hydrates the canonical
/// `data/ontologies/odoo/odoo-core.ttl` plus the available alignment overlays
/// under `data/ontologies/odoo/alignment/` into one bundle, then registers the
/// cascade edge whitelist.
pub fn hydrate_odoo(registry: &OntologyRegistry) -> Result<u32, HydrateErr> {
    let core = odoo_core_path();
    let alignments: Vec<PathBuf> = ODOO_ALIGNMENT_RELATIVE_PATHS
        .iter()
        .map(alignment_path)
        .collect();
    let mut paths: Vec<&Path> = Vec::with_capacity(1 + alignments.len());
    paths.push(&core);
    for a in &alignments {
        if a.exists() {
            paths.push(a.as_path());
        }
    }
    hydrate_odoo_from(&paths, registry)
}

/// Hydrate odoo from explicit paths (test-friendly + multi-file).
///
/// Interns named IRIs across every file in `paths` into a single
/// [`super::owl::ContextBundle`] keyed by `OGIT::ODOO_V1.0`, then registers the
/// cascade edge whitelist. Used by [`hydrate_odoo`] to merge the core seed with
/// the alignment overlays, and by tests that compose a custom bundle.
pub fn hydrate_odoo_from(paths: &[&Path], registry: &OntologyRegistry) -> Result<u32, HydrateErr> {
    let hydrator = OwlHydrator {
        g: OGIT::ODOO_V1.0,
        version: OGIT::ODOO_V1.1,
        domain_name: "odoo".to_string(),
        inherits_from: Some(OGIT::FIBOFND_V1.0),
        starting_entity_id: 100,
    };
    hydrator.hydrate_many(paths, registry)?;
    registry
        .register_edge_types(OGIT::ODOO_V1.0, ODOO_EDGE_WHITELIST)
        .map_err(|reason| HydrateErr::Registry {
            g: OGIT::ODOO_V1.0,
            reason,
        })?;
    Ok(OGIT::ODOO_V1.0)
}

fn odoo_core_path() -> PathBuf {
    // `CARGO_MANIFEST_DIR` for this crate is `crates/lance-graph-ontology`;
    // the data file lives at `<workspace>/data/ontologies/odoo/odoo-core.ttl`.
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
        .join(ODOO_CORE_RELATIVE_PATH)
}

fn alignment_path(rel: &&str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
        .join(rel)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn odoo_core_path_resolves_to_workspace_data() {
        let p = odoo_core_path();
        assert!(
            p.ends_with("data/ontologies/odoo/odoo-core.ttl"),
            "unexpected path tail: {}",
            p.display()
        );
    }

    #[test]
    fn odoo_edge_whitelist_has_the_two_load_bearing_iris() {
        assert!(
            ODOO_EDGE_WHITELIST.contains(&"http://www.w3.org/2000/01/rdf-schema#subClassOf"),
            "rdfs:subClassOf is load-bearing"
        );
        assert!(
            ODOO_EDGE_WHITELIST.contains(&"http://www.w3.org/2002/07/owl#equivalentClass"),
            "owl:equivalentClass is load-bearing"
        );
    }
}
