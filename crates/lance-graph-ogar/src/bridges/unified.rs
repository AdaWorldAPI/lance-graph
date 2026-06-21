//! `UnifiedBridge<P: PortSpec>` — one generic bridge harness driven by
//! OGAR class schema.
//!
//! Before this module landed, each port shipped a clone of the same
//! NamespaceBridge boilerplate (WoaBridge / MedcareBridge /
//! OpenProjectBridge / RedmineBridge) — same struct, same impl shape,
//! same codebook-aware `entity()` override, with a per-bridge constants
//! table baked into each file. Adding a port meant copy-pasting a
//! NamespaceBridge impl AND duplicating its alias table.
//!
//! This bridge collapses that into one generic harness parameterized
//! by [`ogar_vocab::ports::PortSpec`]:
//!
//! ```rust,ignore
//! pub type OpenProjectBridge = UnifiedBridge<ogar_vocab::ports::OpenProjectPort>;
//! pub type RedmineBridge     = UnifiedBridge<ogar_vocab::ports::RedminePort>;
//! ```
//!
//! Differences between bridges are now **inherited from the OGAR class
//! schema** (namespace, bridge_id, and the public-name → canonical
//! class_id alias table all live in `ogar_vocab::ports`), not declared
//! per-bridge in this crate. Adding a port is `impl PortSpec for FooPort {…}`
//! in OGAR — no boilerplate here.
//!
//! # Codebook convergence (canonical-layer-driven)
//!
//! [`UnifiedBridge::entity`] and [`UnifiedBridge::entity_by_uri`] first
//! check `P::class_id(public_name)`. A hit synthesizes an [`EntityRef`]
//! whose `entity_type_id()` is the OGAR canonical class_id from the
//! port spec. Two bridges over the same canonical concept (e.g.
//! `OpenProjectPort` resolving `WorkPackage` and `RedminePort`
//! resolving `Issue`) return EntityRefs with the SAME
//! `entity_type_id` (both `0x0102 project_work_item`) — the cross-fork
//! convergence the codebook was calcified for, sourced entirely from
//! the canonical layer.
//!
//! Public names NOT in the port's alias table fall through to the
//! registry-resolution path the default trait impl uses (back-compat
//! for tenant-specific URIs that have been hydrated).
//!
//! # Context_id stamping
//!
//! Synthesised SchemaPtrs stamp `ontology_context_id` from
//! [`NamespaceRegistry::seed_context_id`] (looked up by
//! `P::NAMESPACE`), so downstream context-based routing can
//! distinguish per-port data from the default (unbound) context.

use lance_graph_ontology::bridge::{BridgeError, BridgeFromRegistry, EntityRef, NamespaceBridge};
use lance_graph_ontology::error::{Error, Result};
use lance_graph_ontology::namespace::{NamespaceId, OgitUri, SchemaKind, SchemaPtr};
use lance_graph_ontology::namespace_registry::NamespaceRegistry;
use lance_graph_ontology::registry::OntologyRegistry;
use ogar_vocab::ports::PortSpec;
use std::marker::PhantomData;
use std::sync::Arc;

/// Generic NamespaceBridge harness parameterised by an OGAR-supplied
/// [`PortSpec`]. See module docs for the design rationale.
pub struct UnifiedBridge<P: PortSpec> {
    registry: Arc<OntologyRegistry>,
    g_lock: NamespaceId,
    _spec: PhantomData<fn() -> P>,
}

impl<P: PortSpec> UnifiedBridge<P> {
    /// New bridge over the given registry. Returns
    /// [`Error::UnknownNamespace`] if the port's namespace
    /// (`P::NAMESPACE`) is not registered yet — callers must hydrate
    /// before constructing the bridge.
    ///
    /// # Errors
    ///
    /// - [`Error::UnknownNamespace`] when the registry has no
    ///   `P::NAMESPACE` namespace registered.
    pub fn new(registry: Arc<OntologyRegistry>) -> Result<Self> {
        let g_lock = registry
            .namespace_id(P::NAMESPACE)
            .ok_or_else(|| Error::UnknownNamespace(P::NAMESPACE.to_string()))?;
        Ok(Self {
            registry,
            g_lock,
            _spec: PhantomData,
        })
    }

    /// Build a synthesised [`EntityRef`] for a codebook concept.
    /// Shared by [`NamespaceBridge::entity`] and
    /// [`NamespaceBridge::entity_by_uri`] so the two paths can't drift
    /// on the convergence contract.
    fn synthesize_codebook_entity(&self, class_id: u16) -> EntityRef {
        let ctx_id = NamespaceRegistry::seed_context_id(P::NAMESPACE).unwrap_or(0);
        let schema_ptr = SchemaPtr::new(self.g_lock, class_id, SchemaKind::Entity)
            .with_context_id(ctx_id);
        EntityRef { schema_ptr }
    }
}

impl<P: PortSpec> NamespaceBridge for UnifiedBridge<P> {
    fn bridge_id(&self) -> &'static str {
        P::BRIDGE_ID
    }
    fn registry(&self) -> &OntologyRegistry {
        &self.registry
    }
    fn g_lock(&self) -> NamespaceId {
        self.g_lock
    }

    /// Codebook-aware override of the default trait impl. Public
    /// names in `P::aliases()` synthesize EntityRefs with the
    /// canonical class_id; everything else falls through to the
    /// registry-resolution path the default impl uses.
    fn entity(&self, public_name: &str) -> std::result::Result<EntityRef, BridgeError> {
        if let Some(class_id) = P::class_id(public_name) {
            return Ok(self.synthesize_codebook_entity(class_id));
        }
        let ptr = self
            .registry()
            .resolve(self.bridge_id(), public_name)
            .ok_or(BridgeError::NotInScope {
                bridge_id: self.bridge_id_static(),
                public_name: public_name.to_string(),
            })?;
        if ptr.namespace_id() != self.g_lock() {
            return Err(BridgeError::CrossNamespaceLeak {
                bridge_id: self.bridge_id_static(),
                resolved_id: ptr.namespace_id(),
                locked_id: self.g_lock(),
            });
        }
        Ok(EntityRef { schema_ptr: ptr })
    }

    /// Codebook-aware override of [`NamespaceBridge::entity_by_uri`].
    /// `ogit.<P::NAMESPACE>:<Name>` where `<Name>` is a codebook entry
    /// goes through the same synthesis path as [`Self::entity`], so
    /// URI-based and public-name-based resolution converge on the
    /// same `entity_type_id`.
    fn entity_by_uri(&self, uri: &OgitUri) -> std::result::Result<EntityRef, BridgeError> {
        if uri.namespace() == Some(P::NAMESPACE) {
            if let Some(name) = uri.name() {
                if let Some(class_id) = P::class_id(name) {
                    return Ok(self.synthesize_codebook_entity(class_id));
                }
            }
        }
        let ptr = self
            .registry()
            .resolve_uri(uri.as_str())
            .ok_or_else(|| BridgeError::NotInScope {
                bridge_id: self.bridge_id_static(),
                public_name: uri.as_str().to_string(),
            })?;
        if ptr.namespace_id() != self.g_lock() {
            return Err(BridgeError::CrossNamespaceLeak {
                bridge_id: self.bridge_id_static(),
                resolved_id: ptr.namespace_id(),
                locked_id: self.g_lock(),
            });
        }
        Ok(EntityRef { schema_ptr: ptr })
    }
}

impl<P: PortSpec> BridgeFromRegistry for UnifiedBridge<P> {
    fn from_registry(registry: Arc<OntologyRegistry>) -> Result<Self> {
        Self::new(registry)
    }
}

#[cfg(test)]
mod tests {
    //! Generic-level tests — exercise the harness through a couple of
    //! built-in PortSpec impls. Per-port behaviour (specific class_ids
    //! per public name) is pinned in `tests/bridge_codebook_convergence.rs`.

    use super::*;
    use ogar_vocab::ports::{OpenProjectPort, RedminePort};
    use std::fs;

    fn registry_with_namespaces(names: &[&str]) -> Arc<OntologyRegistry> {
        let mut ttl = String::from(
            "@prefix ogit: <http://www.purl.org/ogit/> .\n\
             @prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .\n",
        );
        for n in names {
            ttl.push_str(&format!(
                "@prefix ogit.{n}: <http://www.purl.org/ogit/{n}/> .\n"
            ));
        }
        for n in names {
            ttl.push_str(&format!(
                "ogit.{n}:Seed a rdfs:Class; rdfs:subClassOf ogit:Entity; \
                 rdfs:label \"Seed\"; ogit:scope \"NTO\"; ogit:parent ogit:Node; \
                 ogit:mandatory-attributes ( ogit:id ); ogit:optional-attributes ( ) .\n"
            ));
        }
        let tmp = tempfile::tempdir().unwrap();
        for n in names {
            fs::create_dir_all(tmp.path().join(n)).unwrap();
            fs::write(tmp.path().join(n).join("ents.ttl"), &ttl).unwrap();
        }
        let registry = Arc::new(OntologyRegistry::new_in_memory());
        registry.hydrate_once_sync(tmp.path(), names).unwrap();
        std::mem::forget(tmp);
        registry
    }

    #[test]
    fn new_succeeds_for_each_port_when_namespace_registered() {
        let r = registry_with_namespaces(&["OpenProject", "Redmine"]);
        let op = UnifiedBridge::<OpenProjectPort>::new(Arc::clone(&r)).unwrap();
        let rm = UnifiedBridge::<RedminePort>::new(r).unwrap();
        assert_ne!(op.g_lock(), NamespaceId::UNKNOWN);
        assert_ne!(rm.g_lock(), NamespaceId::UNKNOWN);
        assert_ne!(op.g_lock(), rm.g_lock(), "namespaces must differ");
    }

    #[test]
    fn new_returns_unknown_namespace_when_port_namespace_missing() {
        // No hydration -> no OpenProject namespace -> UnknownNamespace.
        // `unwrap_err()` would require `UnifiedBridge<P>: Debug`; match
        // explicitly instead.
        let r = Arc::new(OntologyRegistry::new_in_memory());
        match UnifiedBridge::<OpenProjectPort>::new(r) {
            Ok(_) => panic!("expected UnknownNamespace, got Ok"),
            Err(Error::UnknownNamespace(name)) => assert_eq!(name, "OpenProject"),
            Err(other) => panic!("expected UnknownNamespace, got {other:?}"),
        }
    }

    #[test]
    fn bridge_id_comes_from_port_spec_constant() {
        let r = registry_with_namespaces(&["OpenProject", "Redmine"]);
        let op = UnifiedBridge::<OpenProjectPort>::new(Arc::clone(&r)).unwrap();
        let rm = UnifiedBridge::<RedminePort>::new(r).unwrap();
        assert_eq!(op.bridge_id(), "openproject");
        assert_eq!(rm.bridge_id(), "redmine");
    }

    #[test]
    fn entity_routes_codebook_concept_to_canonical_class_id() {
        let r = registry_with_namespaces(&["OpenProject"]);
        let bridge = UnifiedBridge::<OpenProjectPort>::new(r).unwrap();
        let wp = bridge.entity("WorkPackage").unwrap();
        // 0x0102 == ogar_vocab::class_ids::PROJECT_WORK_ITEM
        assert_eq!(wp.schema_ptr.entity_type_id(), 0x0102);
        assert_eq!(wp.schema_ptr.kind(), SchemaKind::Entity);
        assert_eq!(wp.schema_ptr.ontology_context_id(), 6); // seeded OpenProject ctx_id
    }

    #[test]
    fn entity_for_non_codebook_name_falls_back_to_registry() {
        // The fixture only hydrates `OpenProject:Seed`, which is NOT in
        // the OpenProjectPort alias table — so the codebook check misses
        // and the registry path runs. `Seed` resolves through the
        // bridge's registry lookup (by_uri-style); since the bridge's
        // public-name dictionary has no `Seed` entry filed under
        // `openproject`, NotInScope returns.
        let r = registry_with_namespaces(&["OpenProject"]);
        let bridge = UnifiedBridge::<OpenProjectPort>::new(r).unwrap();
        let err = bridge.entity("Seed").unwrap_err();
        match err {
            BridgeError::NotInScope { public_name, .. } => {
                assert_eq!(public_name, "Seed")
            }
            other => panic!("expected NotInScope, got {other:?}"),
        }
    }

    #[test]
    fn entity_by_uri_routes_codebook_concept_through_port_alias() {
        let r = registry_with_namespaces(&["Redmine"]);
        let bridge = UnifiedBridge::<RedminePort>::new(r).unwrap();
        let uri = OgitUri::parse("ogit.Redmine:Issue").unwrap();
        let issue = bridge.entity_by_uri(&uri).unwrap();
        assert_eq!(issue.schema_ptr.entity_type_id(), 0x0102);
        assert_eq!(issue.schema_ptr.namespace_id(), bridge.g_lock());
    }

    #[test]
    fn entity_and_entity_by_uri_produce_identical_entity_refs_for_codebook_hits() {
        // The two paths can't drift — both go through
        // `synthesize_codebook_entity`.
        let r = registry_with_namespaces(&["OpenProject"]);
        let bridge = UnifiedBridge::<OpenProjectPort>::new(r).unwrap();
        let by_name = bridge.entity("WorkPackage").unwrap();
        let by_uri = bridge
            .entity_by_uri(&OgitUri::parse("ogit.OpenProject:WorkPackage").unwrap())
            .unwrap();
        assert_eq!(by_name.schema_ptr.raw(), by_uri.schema_ptr.raw());
        assert_eq!(
            by_name.schema_ptr.ontology_context_id(),
            by_uri.schema_ptr.ontology_context_id(),
        );
    }
}
