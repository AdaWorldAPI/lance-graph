//! OpenProject tenant bridge — now a thin type alias over
//! [`crate::bridges::unified::UnifiedBridge`] parameterised by
//! [`ogar_vocab::ports::OpenProjectPort`].
//!
//! The differences between bridges (namespace, bridge_id, public-name
//! → class_id alias table) all come from the OGAR class schema. This
//! file used to carry a copy of the `OPENPROJECT_CODEBOOK` constant
//! and a hand-written `NamespaceBridge` impl; both moved to OGAR
//! (`ogar_vocab::ports::OpenProjectPort`) so a single source of truth
//! covers every consumer.
//!
//! All previous tests on the old `OpenProjectBridge` struct now exercise
//! the type alias — same names, same assertions. See
//! `crate::bridges::unified::tests` for the generic-level coverage and
//! `tests/bridge_codebook_convergence.rs` for the cross-port pin.

use crate::bridges::unified::UnifiedBridge;
// `OpenProjectPort::NAMESPACE` / `::aliases()` are `PortSpec`
// associated items — the trait must be in scope for the resolution to
// work (codex P1 on PR #570). Same import in the test module below.
pub use ogar_vocab::ports::OpenProjectPort;
use ogar_vocab::ports::PortSpec;

/// OpenProject `NamespaceBridge` — alias over the generic harness.
///
/// **Deprecated:** pull the classid via the OGAR PortSpec instead —
/// `ogar_vocab::ports::OpenProjectPort::class_id(name)`. The bridge will
/// be removed once all consumers migrate. See
/// `docs/CONSUMER-BRIDGE-DEPRECATION.md` + AdaWorldAPI/OGAR#95.
#[deprecated(
    note = "pull the classid via `OpenProjectPort::class_id(name)` — see AdaWorldAPI/OGAR#95 + docs/CONSUMER-BRIDGE-DEPRECATION.md"
)]
pub type OpenProjectBridge = UnifiedBridge<OpenProjectPort>;

/// Canonical namespace name for OpenProject. Mirrors
/// `OpenProjectPort::NAMESPACE` so existing consumers that imported
/// the constant from this module keep building.
pub const NAMESPACE: &str = OpenProjectPort::NAMESPACE;

/// Compatibility shim — re-exports `ogar_vocab::ports::OPENPROJECT_ALIASES`
/// under the pre-migration name so consumers that imported the constant
/// from this module still build (codex P2 on PR #570). New code should
/// reach for `ogar_vocab::ports::OPENPROJECT_ALIASES` (or
/// `OpenProjectPort::aliases()`) directly — going through the canonical
/// layer keeps lance-graph free of port-specific data.
#[deprecated(
    note = "use `ogar_vocab::ports::OPENPROJECT_ALIASES` (or `OpenProjectPort::aliases()`) — the constant moved to OGAR"
)]
pub const OPENPROJECT_CODEBOOK: &[(&str, u16)] = ogar_vocab::ports::OPENPROJECT_ALIASES;

#[cfg(test)]
#[allow(deprecated)] // exercises the deprecated bridge alias on purpose
mod tests {
    //! Co-located unit tests retained from the pre-migration shape —
    //! constructor success/failure, contract methods, codebook
    //! resolution, fallback to registry. The body of each test is
    //! unchanged; only the `OpenProjectBridge` ident now resolves to
    //! `UnifiedBridge<OpenProjectPort>` instead of a local struct.

    use super::*;
    use lance_graph_ontology::bridge::{BridgeError, NamespaceBridge};
    use lance_graph_ontology::error::Error;
    use lance_graph_ontology::namespace::NamespaceId;
    use lance_graph_ontology::namespace_registry::NamespaceRegistry;
    use lance_graph_ontology::registry::OntologyRegistry;
    use ogar_vocab::class_ids;
    // PortSpec needed in scope for `OpenProjectPort::aliases()` (the
    // method is a trait item — codex P1 on PR #570).
    use ogar_vocab::ports::PortSpec;
    use std::fs;
    use std::sync::Arc;

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
        assert_ne!(bridge.g_lock(), NamespaceId::UNKNOWN);
    }

    #[test]
    fn new_returns_unknown_namespace_when_not_registered() {
        let registry = Arc::new(OntologyRegistry::new_in_memory());
        // `unwrap_err()` would require `OpenProjectBridge: Debug`, which
        // the underlying `UnifiedBridge<P>` intentionally doesn't impl.
        match OpenProjectBridge::new(registry) {
            Ok(_) => panic!("expected UnknownNamespace, got Ok(_)"),
            Err(Error::UnknownNamespace(name)) => assert_eq!(name, "OpenProject"),
            Err(other) => panic!("expected UnknownNamespace, got {other:?}"),
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
        let bridge = OpenProjectBridge::new(registry_with_openproject()).unwrap();
        let entity = bridge.entity("WorkPackage").unwrap();
        assert_eq!(
            entity.schema_ptr.entity_type_id(),
            class_ids::PROJECT_WORK_ITEM
        );
        assert_eq!(entity.schema_ptr.entity_type_id(), 0x0102);
    }

    #[test]
    fn entity_synthesised_schema_ptr_stamps_seeded_context_id() {
        let bridge = OpenProjectBridge::new(registry_with_openproject()).unwrap();
        let entity = bridge.entity("WorkPackage").unwrap();
        let expected = NamespaceRegistry::seed_context_id("OpenProject").unwrap();
        assert_eq!(entity.schema_ptr.ontology_context_id(), expected);
        assert_eq!(entity.schema_ptr.ontology_context_id(), 6);
    }

    #[test]
    fn entity_for_each_codebook_entry_returns_its_canonical_class_id() {
        let bridge = OpenProjectBridge::new(registry_with_openproject()).unwrap();
        for &(public_name, expected_id) in OpenProjectPort::aliases() {
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
