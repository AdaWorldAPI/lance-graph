//! Generic OWL/Turtle → `ContextBundle` hydrator.
//!
//! Walks an OWL/Turtle file via `oxttl`, interns every named IRI as a stable
//! `u32` entity id, packs the assertions into an [`OntologySlot`] keyed by IRI,
//! and registers a [`ContextBundle`] under a chosen `G` slot in the
//! [`crate::OntologyRegistry`]. The same `OwlHydrator` is reusable for every
//! L1/L2/L3/L4 ontology that ships as OWL — DOLCE+DUL (this PR), then OWL-Time,
//! PROV-O, QUDT, SKOS, FIBO, schema.org, etc.
//!
//! Pattern D, in one sentence: *ontologies are data, not crates*. The per-
//! ontology glue (`hydrate_dolce`, `hydrate_owltime`, …) picks the parser,
//! declares the `G` slot, names the parent, and whitelists the edge IRIs —
//! everything else is in this generic module.
//!
//! Carrier-method doctrine: methods live on [`OwlHydrator`], not free
//! functions on its state.

use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use oxttl::TurtleParser;

use crate::registry::OntologyRegistry;

/// One named IRI interned to a stable `u32`. Returned by
/// [`OntologySlot::resolve_iri`] for downstream cascade traversal.
pub type EntityId = u32;

/// Per-`G` IRI interning table built by an [`OwlHydrator`].
///
/// `entity_count` is the number of named IRIs the hydrator saw. `iri_to_id`
/// maps every named IRI to its `u32`. Blank nodes and literals are NOT
/// interned — they don't participate in the cognitive shader's u32-keyed
/// cascade.
#[derive(Debug, Default)]
pub struct OntologySlot {
    pub entity_count: u32,
    pub iri_to_id: HashMap<String, EntityId>,
}

impl OntologySlot {
    /// Resolve an IRI to its interned `u32`, or `None` if it was never
    /// seen during hydration.
    pub fn resolve_iri(&self, iri: &str) -> Option<EntityId> {
        self.iri_to_id.get(iri).copied()
    }

    /// Number of distinct named IRIs interned.
    pub fn len(&self) -> usize {
        self.iri_to_id.len()
    }

    pub fn is_empty(&self) -> bool {
        self.iri_to_id.is_empty()
    }
}

/// The typed `OGIT` per-`G` runtime bundle.
///
/// Registered by [`OwlHydrator::hydrate`] (via [`OntologyRegistry::register_bundle`]).
/// `inherits_from = None` marks the L1 upper-ontology root (only DOLCE
/// declares this). Every downstream L2/L3/L4 hydrator declares
/// `inherits_from: Some(OGIT::DOLCE_V1.0)` directly or transitively.
///
/// `edge_types` is the cascade whitelist — the predicates the cognitive
/// shader's traversal will follow. It's populated separately via
/// [`OntologyRegistry::register_edge_types`] so the OWL hydrator stays
/// ontology-agnostic.
#[derive(Debug)]
pub struct ContextBundle {
    pub g: u32,
    pub version: u32,
    pub domain_name: String,
    pub inherits_from: Option<u32>,
    pub ontology: Option<Arc<OntologySlot>>,
    pub edge_types: Vec<String>,
}

impl ContextBundle {
    /// Number of named IRIs interned under this `G`.
    pub fn entity_count(&self) -> u32 {
        self.ontology.as_ref().map(|o| o.entity_count).unwrap_or(0)
    }

    /// Resolve an IRI under this `G`.
    pub fn resolve_iri(&self, iri: &str) -> Option<EntityId> {
        self.ontology.as_ref().and_then(|o| o.resolve_iri(iri))
    }
}

/// Errors raised by [`OwlHydrator::hydrate`].
#[derive(Debug, thiserror::Error)]
pub enum HydrateErr {
    #[error("I/O reading {path}: {source}")]
    Io {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error("OWL/Turtle parse error in {path}: {message}")]
    Parse { path: PathBuf, message: String },
    #[error("registry rejected bundle at G={g}: {reason}")]
    Registry { g: u32, reason: String },
}

/// Contract every per-ontology glue function fulfils.
///
/// Implemented by [`OwlHydrator`]; future siblings (`XsdHydrator` for UBL /
/// ISO 20022, `XbrlHydrator` for XBRL GL / IFRS, `SkosCsvHydrator` for
/// SKR03/SKR04, `ShaclHydrator` for GoBD / XRechnung) will implement the same
/// trait so the glue layer stays one-liner.
pub trait MetaStructureHydrator {
    /// Read `source`, intern entities, register a [`ContextBundle`] in
    /// `registry`, and return the `G` slot the bundle was registered under.
    fn hydrate(&self, source: &Path, registry: &OntologyRegistry) -> Result<u32, HydrateErr>;
}

/// Generic OWL/Turtle hydrator. Reusable for every OWL-shaped ontology.
pub struct OwlHydrator {
    pub g: u32,
    pub version: u32,
    pub domain_name: String,
    pub inherits_from: Option<u32>,
    /// Lower IDs (0..starting_entity_id) are reserved for OWL builtins / RDF
    /// vocabulary; the hydrator hands out IDs from `starting_entity_id`
    /// upward. Defaults to 100 in callers.
    pub starting_entity_id: EntityId,
}

impl MetaStructureHydrator for OwlHydrator {
    fn hydrate(&self, ttl_path: &Path, registry: &OntologyRegistry) -> Result<u32, HydrateErr> {
        self.hydrate_many(&[ttl_path], registry)
    }
}

impl OwlHydrator {
    /// Multi-file variant. Interns named IRIs across every file in
    /// `ttl_paths` into a single [`ContextBundle`] keyed by `self.g`.
    ///
    /// Use this when an ontology ships as multiple TTL artifacts that
    /// logically constitute one ontology — e.g. QUDT (core schema + units
    /// catalogue + quantity-kinds catalogue), FIBO (FND + BE + … bundle),
    /// or schema.org full + extensions. Per-file order is preserved, and
    /// IRIs seen earlier keep their lower `EntityId` so the interning is
    /// deterministic given the input file order.
    pub fn hydrate_many(
        &self,
        ttl_paths: &[&Path],
        registry: &OntologyRegistry,
    ) -> Result<u32, HydrateErr> {
        let mut iri_to_id: HashMap<String, EntityId> = HashMap::new();
        let mut next_id: EntityId = self.starting_entity_id;

        let intern = |iri: &str,
                      map: &mut HashMap<String, EntityId>,
                      n: &mut EntityId|
         -> EntityId {
            if let Some(&id) = map.get(iri) {
                return id;
            }
            let id = *n;
            *n += 1;
            map.insert(iri.to_string(), id);
            id
        };

        for ttl_path in ttl_paths {
            let bytes = fs::read(ttl_path).map_err(|e| HydrateErr::Io {
                path: ttl_path.to_path_buf(),
                source: e,
            })?;
            let parser = TurtleParser::new().for_slice(&bytes);
            for triple in parser {
                let triple = triple.map_err(|e| HydrateErr::Parse {
                    path: ttl_path.to_path_buf(),
                    message: e.to_string(),
                })?;

                if let oxrdf::Subject::NamedNode(n) = &triple.subject {
                    intern(n.as_str(), &mut iri_to_id, &mut next_id);
                }
                intern(triple.predicate.as_str(), &mut iri_to_id, &mut next_id);
                if let oxrdf::Term::NamedNode(n) = &triple.object {
                    intern(n.as_str(), &mut iri_to_id, &mut next_id);
                }
            }
        }

        let entity_count = iri_to_id.len() as u32;
        let ontology = OntologySlot {
            entity_count,
            iri_to_id,
        };

        let bundle = ContextBundle {
            g: self.g,
            version: self.version,
            domain_name: self.domain_name.clone(),
            inherits_from: self.inherits_from,
            ontology: Some(Arc::new(ontology)),
            edge_types: Vec::new(),
        };

        registry.register_bundle(bundle);

        Ok(self.g)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::tempdir;

    const TINY_TTL: &str = r#"
@prefix rdf:  <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
@prefix owl:  <http://www.w3.org/2002/07/owl#> .
@prefix ex:   <http://example.org/owl#> .

ex:Animal a owl:Class .
ex:Cat a owl:Class ;
    rdfs:subClassOf ex:Animal .
ex:Dog a owl:Class ;
    rdfs:subClassOf ex:Animal .
"#;

    #[test]
    fn tiny_ttl_hydrates_and_registers_bundle() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("tiny.ttl");
        let mut f = std::fs::File::create(&path).unwrap();
        f.write_all(TINY_TTL.as_bytes()).unwrap();
        drop(f);

        let reg = OntologyRegistry::new_in_memory();
        let hydrator = OwlHydrator {
            g: 999,
            version: 1,
            domain_name: "tiny".to_string(),
            inherits_from: None,
            starting_entity_id: 100,
        };
        let g = hydrator.hydrate(&path, &reg).expect("hydrate");
        assert_eq!(g, 999);

        let bundle = reg.bundle_for(999).expect("bundle registered");
        assert_eq!(bundle.g, 999);
        assert_eq!(bundle.inherits_from, None);
        // 3 ex:* classes + rdf:type + rdfs:subClassOf + owl:Class = 6 IRIs
        assert!(bundle.entity_count() >= 6);
        assert!(bundle
            .resolve_iri("http://example.org/owl#Animal")
            .is_some());
    }

    #[test]
    fn parse_error_surfaces() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("bad.ttl");
        std::fs::write(&path, b"this is not turtle @@@").unwrap();

        let reg = OntologyRegistry::new_in_memory();
        let hydrator = OwlHydrator {
            g: 1,
            version: 1,
            domain_name: "bad".to_string(),
            inherits_from: None,
            starting_entity_id: 100,
        };
        let result = hydrator.hydrate(&path, &reg);
        assert!(matches!(result, Err(HydrateErr::Parse { .. })));
    }
}
