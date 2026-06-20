//! OpenProject tenant bridge — locks to the `OpenProject` namespace.
//!
//! Northstar plan §3, C4. Sibling of [`crate::bridges::MedcareBridge`] and
//! [`crate::bridges::WoaBridge`]; supplies the OpenProject port
//! (`openproject-nexgen-rs` + `op-canon`) with the scoped registry view
//! every consumer that touches OpenProject data on the unified bridge
//! goes through.
//!
//! # Codebook convergence (codex P1 on PR #559)
//!
//! [`OpenProjectBridge::entity`] overrides the default trait impl with a
//! codebook-aware lookup: a public name in [`OPENPROJECT_CODEBOOK`]
//! synthesizes an [`EntityRef`] whose `entity_type_id()` is the OGAR
//! canonical class_id (e.g. `WorkPackage` → `0x0102 project_work_item`).
//! Redmine's bridge does the SAME synthesis for `Issue` → `0x0102`, so
//! `op_bridge.entity("WorkPackage")` and `rm_bridge.entity("Issue")`
//! return EntityRefs with the SAME `entity_type_id` — the cross-fork
//! convergence the codebook was calcified for.
//!
//! Public names NOT in the codebook fall through to the default
//! registry-resolution path (URI-keyed entity_type_id), preserving
//! back-compat for any tenant-specific URI a consumer has hydrated.
//!
//! # Context_id stamping (codex P2 on PR #558)
//!
//! Synthesized SchemaPtrs stamp `ontology_context_id` from
//! [`crate::namespace_registry::NamespaceRegistry::seed_context_id`]
//! ("OpenProject" → `6`, see
//! [`crate::namespace_registry::NamespaceRegistry::seed_defaults`]) so
//! downstream context-based routing can distinguish OpenProject data
//! from the default context (id 0).

use crate::bridge::{BridgeError, BridgeFromRegistry, EntityRef, NamespaceBridge};
use crate::bridges::codebook;
use crate::error::{Error, Result};
use crate::namespace::{NamespaceId, SchemaKind, SchemaPtr};
use crate::namespace_registry::NamespaceRegistry;
use crate::registry::OntologyRegistry;
use std::sync::Arc;

/// Canonical namespace name for OpenProject. Matches the
/// `ogit.OpenProject:` TTL prefix the corpus's per-entity files use.
pub const NAMESPACE: &str = "OpenProject";

/// OpenProject's port-specific **public name → OGAR canonical class_id**
/// mapping. Each entry is `(rails_model_name, class_id)`.
///
/// Used by [`OpenProjectBridge::entity`] to synthesize convergent
/// EntityRefs without going through the registry's URI-keyed minting
/// path. The class_ids on the right column live in
/// [`crate::bridges::codebook`] (single source of truth for codebook
/// constants); the strings on the left are the Rails class names the
/// OpenProject port uses.
///
/// Sister table: `RedmineBridge::REDMINE_CODEBOOK`. Both reference the
/// SAME `codebook::*` constants for converging concepts (e.g.
/// `WorkPackage` and `Issue` both → `codebook::PROJECT_WORK_ITEM`).
pub const OPENPROJECT_CODEBOOK: &[(&str, u16)] = &[
    ("Project", codebook::PROJECT),
    ("WorkPackage", codebook::PROJECT_WORK_ITEM),
    ("TimeEntry", codebook::BILLABLE_WORK_ENTRY),
    ("User", codebook::PROJECT_ACTOR),
    ("Status", codebook::PROJECT_STATUS),
    ("Type", codebook::PROJECT_TYPE),
    ("Priority", codebook::PRIORITY),
    ("Membership", codebook::PROJECT_MEMBERSHIP),
    ("Journal", codebook::PROJECT_JOURNAL),
    ("Repository", codebook::PROJECT_REPOSITORY),
    ("Version", codebook::PROJECT_VERSION),
    ("WikiPage", codebook::PROJECT_WIKI_PAGE),
    ("Query", codebook::PROJECT_QUERY),
    ("Attachment", codebook::PROJECT_ATTACHMENT),
    ("CustomField", codebook::PROJECT_CUSTOM_FIELD),
    ("Relation", codebook::PROJECT_RELATION),
    ("Changeset", codebook::PROJECT_CHANGESET),
    ("Watcher", codebook::PROJECT_WATCHER),
    ("News", codebook::PROJECT_NEWS),
    ("Message", codebook::PROJECT_MESSAGE),
    ("Forum", codebook::PROJECT_FORUM),
    ("Role", codebook::PROJECT_ROLE),
    ("MemberRole", codebook::PROJECT_MEMBER_ROLE),
    ("CustomValue", codebook::PROJECT_CUSTOM_VALUE),
    ("EnabledModule", codebook::PROJECT_ENABLED_MODULE),
];

/// Lookup an OpenProject public name in [`OPENPROJECT_CODEBOOK`].
/// Returns the canonical class_id or `None` if `public_name` is not a
/// codebook concept (caller falls back to registry resolution).
fn openproject_class_id(public_name: &str) -> Option<u16> {
    OPENPROJECT_CODEBOOK
        .iter()
        .find(|(name, _)| *name == public_name)
        .map(|(_, id)| *id)
}

/// Scoped registry view locked to the `OpenProject` namespace. Built
/// over an [`OntologyRegistry`] that has already hydrated the
/// OpenProject namespace (via [`OntologyRegistry::hydrate_once_sync`]
/// or a programmatic `register_*` call).
pub struct OpenProjectBridge {
    registry: Arc<OntologyRegistry>,
    g_lock: NamespaceId,
}

impl OpenProjectBridge {
    /// New bridge over the given registry. Returns
    /// [`Error::UnknownNamespace`] if the `OpenProject` namespace
    /// is not registered yet — callers must hydrate before
    /// constructing the bridge.
    ///
    /// # Errors
    ///
    /// - [`Error::UnknownNamespace`] when the registry has no
    ///   `OpenProject` namespace registered.
    pub fn new(registry: Arc<OntologyRegistry>) -> Result<Self> {
        let g_lock = registry
            .namespace_id(NAMESPACE)
            .ok_or_else(|| Error::UnknownNamespace(NAMESPACE.to_string()))?;
        Ok(Self { registry, g_lock })
    }
}

impl NamespaceBridge for OpenProjectBridge {
    fn bridge_id(&self) -> &'static str {
        "openproject"
    }
    fn registry(&self) -> &OntologyRegistry {
        &self.registry
    }
    fn g_lock(&self) -> NamespaceId {
        self.g_lock
    }

    /// Codebook-aware override of the default trait impl. See module
    /// docs (codex P1 on PR #559) for the convergence contract. Public
    /// names not in [`OPENPROJECT_CODEBOOK`] fall through to the
    /// registry-resolution path the default impl uses (back-compat for
    /// tenant-specific URIs).
    fn entity(&self, public_name: &str) -> std::result::Result<EntityRef, BridgeError> {
        if let Some(class_id) = openproject_class_id(public_name) {
            let ctx_id = NamespaceRegistry::seed_context_id(NAMESPACE).unwrap_or(0);
            let schema_ptr = SchemaPtr::new(self.g_lock, class_id, SchemaKind::Entity)
                .with_context_id(ctx_id);
            return Ok(EntityRef { schema_ptr });
        }
        // Fall through to the trait's default behaviour. Inlined here
        // because Rust default-method override can't delegate to the
        // default directly; the two arms stay small enough to keep.
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
}

impl BridgeFromRegistry for OpenProjectBridge {
    fn from_registry(registry: Arc<OntologyRegistry>) -> Result<Self> {
        Self::new(registry)
    }
}

#[cfg(test)]
mod tests {
    //! Co-located unit tests — CodeRabbit on PR #558 flagged that
    //! coverage was integration-only and asked for focused per-method
    //! scenarios. Integration tests live in
    //! `tests/openproject_bridge_scope_lock.rs` (hydration + scope-lock)
    //! and `tests/bridge_codebook_convergence.rs` (cross-bridge
    //! `entity()` convergence).

    use super::*;
    use crate::namespace::SchemaKind;
    use std::fs;

    /// Build a minimal hydrated registry holding only the OpenProject
    /// namespace. Used by the contract-method tests.
    fn registry_with_openproject() -> Arc<OntologyRegistry> {
        let ttl = r#"
@prefix ogit:                <http://www.purl.org/ogit/> .
@prefix ogit.OpenProject:    <http://www.purl.org/ogit/OpenProject/> .
@prefix rdfs:                <http://www.w3.org/2000/01/rdf-schema#> .

ogit.OpenProject:WorkPackage
    a rdfs:Class;
    rdfs:subClassOf ogit:Entity;
    rdfs:label "WorkPackage";
    ogit:scope "NTO";
    ogit:parent ogit:Node;
    ogit:mandatory-attributes ( ogit:id );
    ogit:optional-attributes ( ) ;
.
"#;
        let tmp = tempfile::tempdir().unwrap();
        fs::create_dir_all(tmp.path().join("OpenProject")).unwrap();
        fs::write(tmp.path().join("OpenProject").join("ents.ttl"), ttl).unwrap();
        let registry = Arc::new(OntologyRegistry::new_in_memory());
        registry
            .hydrate_once_sync(tmp.path(), &["OpenProject"])
            .unwrap();
        std::mem::forget(tmp);
        registry
    }

    #[test]
    fn new_succeeds_when_namespace_registered() {
        let registry = registry_with_openproject();
        let bridge = OpenProjectBridge::new(registry).unwrap();
        // Smoke: the bridge built without panicking and has a non-UNKNOWN
        // g_lock (UNKNOWN(0) is the unbound sentinel).
        assert_ne!(bridge.g_lock(), NamespaceId::UNKNOWN);
    }

    #[test]
    fn new_returns_unknown_namespace_when_not_registered() {
        let registry = Arc::new(OntologyRegistry::new_in_memory());
        let err = OpenProjectBridge::new(registry).unwrap_err();
        match err {
            Error::UnknownNamespace(name) => assert_eq!(name, "OpenProject"),
            other => panic!("expected UnknownNamespace, got {other:?}"),
        }
    }

    #[test]
    fn bridge_id_is_lowercase_openproject() {
        let bridge = OpenProjectBridge::new(registry_with_openproject()).unwrap();
        assert_eq!(bridge.bridge_id(), "openproject");
    }

    #[test]
    fn g_lock_matches_registry_namespace_id() {
        let registry = registry_with_openproject();
        let expected = registry.namespace_id(NAMESPACE).unwrap();
        let bridge = OpenProjectBridge::new(registry).unwrap();
        assert_eq!(bridge.g_lock(), expected);
    }

    #[test]
    fn entity_resolves_workpackage_to_project_work_item_class_id() {
        // The headline codex P1 fix: WorkPackage → 0x0102.
        let bridge = OpenProjectBridge::new(registry_with_openproject()).unwrap();
        let entity = bridge.entity("WorkPackage").unwrap();
        assert_eq!(
            entity.schema_ptr.entity_type_id(),
            codebook::PROJECT_WORK_ITEM
        );
        assert_eq!(entity.schema_ptr.entity_type_id(), 0x0102);
        // SchemaKind on synthesized entity is Entity (not Edge / Attribute).
        assert_eq!(entity.schema_ptr.kind(), SchemaKind::Entity);
    }

    #[test]
    fn entity_synthesised_schema_ptr_stamps_seeded_context_id() {
        // Codex P2 on PR #558: the bridge stamps the OpenProject ctx_id
        // (6) on the SchemaPtr so context-based dispatch can distinguish
        // OpenProject data from the default (id 0) context.
        let bridge = OpenProjectBridge::new(registry_with_openproject()).unwrap();
        let entity = bridge.entity("WorkPackage").unwrap();
        let expected = NamespaceRegistry::seed_context_id("OpenProject").unwrap();
        assert_eq!(entity.schema_ptr.ontology_context_id(), expected);
        assert_eq!(entity.schema_ptr.ontology_context_id(), 6);
    }

    #[test]
    fn entity_for_each_codebook_entry_returns_its_canonical_class_id() {
        // All 25 entries in OPENPROJECT_CODEBOOK resolve through the
        // override and return the table's class_id verbatim.
        let bridge = OpenProjectBridge::new(registry_with_openproject()).unwrap();
        for &(public_name, expected_id) in OPENPROJECT_CODEBOOK {
            let entity = bridge.entity(public_name).unwrap_or_else(|e| {
                panic!("codebook entry `{public_name}` failed to resolve: {e:?}")
            });
            assert_eq!(
                entity.schema_ptr.entity_type_id(),
                expected_id,
                "codebook entry `{public_name}` should resolve to 0x{expected_id:04X}",
            );
        }
    }

    #[test]
    fn entity_for_non_codebook_name_falls_back_to_registry_lookup() {
        // A public name NOT in the codebook is unregistered in the
        // registry (the fixture only hydrates WorkPackage), so the
        // fallback path returns NotInScope.
        let bridge = OpenProjectBridge::new(registry_with_openproject()).unwrap();
        let err = bridge.entity("NotAConcept").unwrap_err();
        match err {
            BridgeError::NotInScope { public_name, .. } => {
                assert_eq!(public_name, "NotAConcept")
            }
            other => panic!("expected NotInScope, got {other:?}"),
        }
    }
}
