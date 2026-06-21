//! Redmine tenant bridge — now a thin type alias over
//! [`crate::bridges::unified::UnifiedBridge`] parameterised by
//! [`ogar_vocab::ports::RedminePort`].
//!
//! The differences between bridges (namespace, bridge_id, public-name
//! → class_id alias table) all come from the OGAR class schema. This
//! file used to carry a copy of the `REDMINE_CODEBOOK` constant and
//! a hand-written `NamespaceBridge` impl; both moved to OGAR
//! (`ogar_vocab::ports::RedminePort`) so a single source of truth
//! covers every consumer.

use crate::bridges::unified::UnifiedBridge;
// `RedminePort::NAMESPACE` / `::aliases()` are `PortSpec` associated
// items — the trait must be in scope for the resolution to work
// (codex P1 on PR #570).
use ogar_vocab::ports::PortSpec;
pub use ogar_vocab::ports::RedminePort;

/// Redmine `NamespaceBridge` — alias over the generic harness.
pub type RedmineBridge = UnifiedBridge<RedminePort>;

/// Canonical namespace name for Redmine. Mirrors `RedminePort::NAMESPACE`
/// so existing consumers that imported the constant from this module
/// keep building.
pub const NAMESPACE: &str = RedminePort::NAMESPACE;

/// Compatibility shim — re-exports `ogar_vocab::ports::REDMINE_ALIASES`
/// under the pre-migration name (codex P2 on PR #570). New code should
/// reach for the OGAR constant directly.
#[deprecated(
    note = "use `ogar_vocab::ports::REDMINE_ALIASES` (or `RedminePort::aliases()`) — the constant moved to OGAR"
)]
pub const REDMINE_CODEBOOK: &[(&str, u16)] = ogar_vocab::ports::REDMINE_ALIASES;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bridge::{BridgeError, NamespaceBridge};
    use crate::error::Error;
    use crate::namespace::NamespaceId;
    use crate::namespace_registry::NamespaceRegistry;
    use crate::registry::OntologyRegistry;
    use ogar_vocab::class_ids;
    // PortSpec needed in scope for `RedminePort::aliases()` (the method
    // is a trait item — codex P1 on PR #570).
    use ogar_vocab::ports::PortSpec;
    use std::fs;
    use std::sync::Arc;

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
        match RedmineBridge::new(registry) {
            Ok(_) => panic!("expected UnknownNamespace, got Ok(_)"),
            Err(Error::UnknownNamespace(name)) => assert_eq!(name, "Redmine"),
            Err(other) => panic!("expected UnknownNamespace, got {other:?}"),
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
        let bridge = RedmineBridge::new(registry_with_redmine()).unwrap();
        let entity = bridge.entity("Issue").unwrap();
        assert_eq!(
            entity.schema_ptr.entity_type_id(),
            class_ids::PROJECT_WORK_ITEM
        );
        assert_eq!(entity.schema_ptr.entity_type_id(), 0x0102);
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
        for &(public_name, expected_id) in RedminePort::aliases() {
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
