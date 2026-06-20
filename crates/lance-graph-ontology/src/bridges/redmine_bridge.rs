//! Redmine tenant bridge — locks to the `Redmine` namespace.
//!
//! Northstar plan §3, C5. Sibling of [`crate::bridges::OpenProjectBridge`]
//! (C4); supplies the Redmine port (`redmine-rs` + `redmine-canon`) with
//! the scoped registry view every consumer that touches Redmine data on
//! the unified bridge goes through.
//!
//! # Codebook convergence (codex P1 on PR #559)
//!
//! [`RedmineBridge::entity`] overrides the default trait impl with a
//! codebook-aware lookup: a public name in [`REDMINE_CODEBOOK`]
//! synthesizes an [`EntityRef`] whose `entity_type_id()` is the OGAR
//! canonical class_id (e.g. `Issue` → `0x0102 project_work_item`).
//! OpenProject's bridge does the SAME synthesis for `WorkPackage` →
//! `0x0102`, so `op_bridge.entity("WorkPackage")` and
//! `rm_bridge.entity("Issue")` return EntityRefs with the SAME
//! `entity_type_id` — the cross-fork convergence the codebook was
//! calcified for.
//!
//! Public names NOT in the codebook fall through to the default
//! registry-resolution path (URI-keyed entity_type_id), preserving
//! back-compat for any tenant-specific URI a consumer has hydrated.
//!
//! # Apple meets apple — the convergence pin
//!
//! ```text
//!   op.entity("WorkPackage").schema_ptr.entity_type_id() == 0x0102
//!   rm.entity("Issue")      .schema_ptr.entity_type_id() == 0x0102
//!                              ↑          ↑
//!                       same codebook id → same OgarClassView arm
//! ```
//!
//! Pinned by `tests/bridge_codebook_convergence.rs`.

use crate::bridge::{BridgeError, BridgeFromRegistry, EntityRef, NamespaceBridge};
use crate::bridges::codebook;
use crate::error::{Error, Result};
use crate::namespace::{NamespaceId, SchemaKind, SchemaPtr};
use crate::namespace_registry::NamespaceRegistry;
use crate::registry::OntologyRegistry;
use std::sync::Arc;

/// Canonical namespace name for Redmine. Matches the
/// `ogit.Redmine:` TTL prefix the corpus's per-entity files use.
pub const NAMESPACE: &str = "Redmine";

/// Redmine's port-specific **public name → OGAR canonical class_id**
/// mapping. Each entry is `(rails_model_name, class_id)`.
///
/// Sister table: `OpenProjectBridge::OPENPROJECT_CODEBOOK`. Both
/// reference the SAME `codebook::*` constants for converging concepts
/// (e.g. `Issue` and `WorkPackage` both → `codebook::PROJECT_WORK_ITEM`).
pub const REDMINE_CODEBOOK: &[(&str, u16)] = &[
    ("Project", codebook::PROJECT),
    ("Issue", codebook::PROJECT_WORK_ITEM),
    ("TimeEntry", codebook::BILLABLE_WORK_ENTRY),
    ("User", codebook::PROJECT_ACTOR),
    ("IssueStatus", codebook::PROJECT_STATUS),
    ("Tracker", codebook::PROJECT_TYPE),
    ("Member", codebook::PROJECT_MEMBERSHIP),
    ("Journal", codebook::PROJECT_JOURNAL),
    ("Repository", codebook::PROJECT_REPOSITORY),
    ("Version", codebook::PROJECT_VERSION),
    ("WikiPage", codebook::PROJECT_WIKI_PAGE),
    ("Query", codebook::PROJECT_QUERY),
    ("Attachment", codebook::PROJECT_ATTACHMENT),
    ("Comment", codebook::PROJECT_COMMENT),
    ("CustomField", codebook::PROJECT_CUSTOM_FIELD),
    ("IssueRelation", codebook::PROJECT_RELATION),
    ("Changeset", codebook::PROJECT_CHANGESET),
    ("Watcher", codebook::PROJECT_WATCHER),
    ("News", codebook::PROJECT_NEWS),
    ("Message", codebook::PROJECT_MESSAGE),
    ("Board", codebook::PROJECT_FORUM),
    ("Role", codebook::PROJECT_ROLE),
    ("MemberRole", codebook::PROJECT_MEMBER_ROLE),
    ("CustomValue", codebook::PROJECT_CUSTOM_VALUE),
    ("EnabledModule", codebook::PROJECT_ENABLED_MODULE),
];

/// Lookup a Redmine public name in [`REDMINE_CODEBOOK`].
/// Returns the canonical class_id or `None` if `public_name` is not a
/// codebook concept (caller falls back to registry resolution).
fn redmine_class_id(public_name: &str) -> Option<u16> {
    REDMINE_CODEBOOK
        .iter()
        .find(|(name, _)| *name == public_name)
        .map(|(_, id)| *id)
}

/// Scoped registry view locked to the `Redmine` namespace. Built over
/// an [`OntologyRegistry`] that has already hydrated the Redmine
/// namespace (via [`OntologyRegistry::hydrate_once_sync`] or a
/// programmatic `register_*` call).
pub struct RedmineBridge {
    registry: Arc<OntologyRegistry>,
    g_lock: NamespaceId,
}

impl RedmineBridge {
    /// New bridge over the given registry. Returns
    /// [`Error::UnknownNamespace`] if the `Redmine` namespace
    /// is not registered yet — callers must hydrate before
    /// constructing the bridge.
    ///
    /// # Errors
    ///
    /// - [`Error::UnknownNamespace`] when the registry has no
    ///   `Redmine` namespace registered.
    pub fn new(registry: Arc<OntologyRegistry>) -> Result<Self> {
        let g_lock = registry
            .namespace_id(NAMESPACE)
            .ok_or_else(|| Error::UnknownNamespace(NAMESPACE.to_string()))?;
        Ok(Self { registry, g_lock })
    }
}

impl NamespaceBridge for RedmineBridge {
    fn bridge_id(&self) -> &'static str {
        "redmine"
    }
    fn registry(&self) -> &OntologyRegistry {
        &self.registry
    }
    fn g_lock(&self) -> NamespaceId {
        self.g_lock
    }

    /// Codebook-aware override of the default trait impl. See module
    /// docs (codex P1 on PR #559) for the convergence contract. Public
    /// names not in [`REDMINE_CODEBOOK`] fall through to the
    /// registry-resolution path the default impl uses (back-compat for
    /// tenant-specific URIs).
    fn entity(&self, public_name: &str) -> std::result::Result<EntityRef, BridgeError> {
        if let Some(class_id) = redmine_class_id(public_name) {
            let ctx_id = NamespaceRegistry::seed_context_id(NAMESPACE).unwrap_or(0);
            let schema_ptr = SchemaPtr::new(self.g_lock, class_id, SchemaKind::Entity)
                .with_context_id(ctx_id);
            return Ok(EntityRef { schema_ptr });
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
}

impl BridgeFromRegistry for RedmineBridge {
    fn from_registry(registry: Arc<OntologyRegistry>) -> Result<Self> {
        Self::new(registry)
    }
}

#[cfg(test)]
mod tests {
    //! Co-located unit tests — same shape CodeRabbit asked for on
    //! OpenProjectBridge, applied symmetrically here.

    use super::*;
    use std::fs;

    fn registry_with_redmine() -> Arc<OntologyRegistry> {
        let ttl = r#"
@prefix ogit:                <http://www.purl.org/ogit/> .
@prefix ogit.Redmine:        <http://www.purl.org/ogit/Redmine/> .
@prefix rdfs:                <http://www.w3.org/2000/01/rdf-schema#> .

ogit.Redmine:Issue
    a rdfs:Class;
    rdfs:subClassOf ogit:Entity;
    rdfs:label "Issue";
    ogit:scope "NTO";
    ogit:parent ogit:Node;
    ogit:mandatory-attributes ( ogit:id );
    ogit:optional-attributes ( ) ;
.
"#;
        let tmp = tempfile::tempdir().unwrap();
        fs::create_dir_all(tmp.path().join("Redmine")).unwrap();
        fs::write(tmp.path().join("Redmine").join("ents.ttl"), ttl).unwrap();
        let registry = Arc::new(OntologyRegistry::new_in_memory());
        registry
            .hydrate_once_sync(tmp.path(), &["Redmine"])
            .unwrap();
        std::mem::forget(tmp);
        registry
    }

    #[test]
    fn new_succeeds_when_namespace_registered() {
        let registry = registry_with_redmine();
        let bridge = RedmineBridge::new(registry).unwrap();
        assert_ne!(bridge.g_lock(), NamespaceId::UNKNOWN);
    }

    #[test]
    fn new_returns_unknown_namespace_when_not_registered() {
        let registry = Arc::new(OntologyRegistry::new_in_memory());
        let err = RedmineBridge::new(registry).unwrap_err();
        match err {
            Error::UnknownNamespace(name) => assert_eq!(name, "Redmine"),
            other => panic!("expected UnknownNamespace, got {other:?}"),
        }
    }

    #[test]
    fn bridge_id_is_lowercase_redmine() {
        let bridge = RedmineBridge::new(registry_with_redmine()).unwrap();
        assert_eq!(bridge.bridge_id(), "redmine");
    }

    #[test]
    fn g_lock_matches_registry_namespace_id() {
        let registry = registry_with_redmine();
        let expected = registry.namespace_id(NAMESPACE).unwrap();
        let bridge = RedmineBridge::new(registry).unwrap();
        assert_eq!(bridge.g_lock(), expected);
    }

    #[test]
    fn entity_resolves_issue_to_project_work_item_class_id() {
        // Headline codex P1 fix: Issue → 0x0102, the SAME id
        // OpenProject's WorkPackage resolves to.
        let bridge = RedmineBridge::new(registry_with_redmine()).unwrap();
        let entity = bridge.entity("Issue").unwrap();
        assert_eq!(
            entity.schema_ptr.entity_type_id(),
            codebook::PROJECT_WORK_ITEM
        );
        assert_eq!(entity.schema_ptr.entity_type_id(), 0x0102);
        assert_eq!(entity.schema_ptr.kind(), SchemaKind::Entity);
    }

    #[test]
    fn entity_synthesised_schema_ptr_stamps_seeded_context_id() {
        let bridge = RedmineBridge::new(registry_with_redmine()).unwrap();
        let entity = bridge.entity("Issue").unwrap();
        let expected = NamespaceRegistry::seed_context_id("Redmine").unwrap();
        assert_eq!(entity.schema_ptr.ontology_context_id(), expected);
        assert_eq!(entity.schema_ptr.ontology_context_id(), 7);
    }

    #[test]
    fn entity_for_each_codebook_entry_returns_its_canonical_class_id() {
        let bridge = RedmineBridge::new(registry_with_redmine()).unwrap();
        for &(public_name, expected_id) in REDMINE_CODEBOOK {
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
        let bridge = RedmineBridge::new(registry_with_redmine()).unwrap();
        let err = bridge.entity("NotAConcept").unwrap_err();
        match err {
            BridgeError::NotInScope { public_name, .. } => {
                assert_eq!(public_name, "NotAConcept")
            }
            other => panic!("expected NotInScope, got {other:?}"),
        }
    }
}
