//! MedCare (healthcare) tenant bridge — now a thin type alias over
//! [`crate::bridges::unified::UnifiedBridge`] parameterised by
//! [`ogar_vocab::ports::HealthcarePort`].
//!
//! Before the Healthcare codebook promotion (Northstar T9) this file
//! carried a bespoke `MedcareBridge` struct + hand-written
//! `NamespaceBridge` impl — the same boilerplate every per-tenant bridge
//! cloned. lance-graph#570 collapsed `OpenProjectBridge` / `RedmineBridge`
//! onto the generic [`UnifiedBridge<P>`] harness but explicitly deferred
//! `MedcareBridge` "until Healthcare gets promoted into the codebook".
//! That promotion has now landed: OGAR mints the `0x09XX` Health concepts
//! and `ports::HealthcarePort` carries the namespace / bridge_id / public-
//! name → class_id alias table, so this bridge becomes one line.
//!
//! The differences between bridges (namespace, bridge_id, alias table)
//! all come from the OGAR class schema. Codebook public names (`Patient`,
//! `Diagnosis`, …) synthesize an `EntityRef` whose `entity_type_id()` is
//! the canonical Health class_id; names outside the alias table fall
//! through to the registry-resolution path (so a hydrated TTL entity that
//! is not yet a codebook concept still resolves). The audit / authorization
//! path uses `row()` (registry-backed, not overridden here), so it is
//! unaffected by codebook synthesis on `entity()`.
//!
//! See `crate::bridges::unified::tests` for the generic-level coverage and
//! `tests/bridge_scope_lock.rs` for the Healthcare scope-lock pins.

use crate::bridges::unified::UnifiedBridge;
// `HealthcarePort::NAMESPACE` / `::aliases()` are `PortSpec` associated
// items — the trait must be in scope for the resolution to work (codex
// P1 on PR #570). Same import in the test module below.
pub use ogar_vocab::ports::HealthcarePort;
use ogar_vocab::ports::PortSpec;

/// MedCare `NamespaceBridge` — alias over the generic harness, locked to
/// the `Healthcare` namespace via [`HealthcarePort`].
pub type MedcareBridge = UnifiedBridge<HealthcarePort>;

/// Canonical namespace name for MedCare / Healthcare. Mirrors
/// `HealthcarePort::NAMESPACE` so existing consumers that imported the
/// constant from this module keep building.
pub const NAMESPACE: &str = HealthcarePort::NAMESPACE;

#[cfg(test)]
mod tests {
    //! Co-located unit tests for the migrated alias — constructor
    //! success/failure, contract methods, codebook resolution, fallback
    //! to registry. Mirrors `openproject_bridge::tests`; only the
    //! `MedcareBridge` ident (now `UnifiedBridge<HealthcarePort>`) and the
    //! Healthcare fixtures differ.

    use super::*;
    use lance_graph_ontology::bridge::{BridgeError, NamespaceBridge};
    use lance_graph_ontology::error::Error;
    use lance_graph_ontology::namespace::NamespaceId;
    use lance_graph_ontology::namespace_registry::NamespaceRegistry;
    use lance_graph_ontology::registry::OntologyRegistry;
    use ogar_vocab::class_ids;
    // PortSpec needed in scope for `HealthcarePort::aliases()`.
    use ogar_vocab::ports::PortSpec;
    use std::fs;
    use std::sync::Arc;

    fn registry_with_healthcare() -> Arc<OntologyRegistry> {
        let ttl = r#"
@prefix ogit:               <http://www.purl.org/ogit/> .
@prefix ogit.Healthcare:    <http://www.purl.org/ogit/Healthcare/> .
@prefix rdfs:               <http://www.w3.org/2000/01/rdf-schema#> .

ogit.Healthcare:Patient
    a rdfs:Class;
    rdfs:subClassOf ogit:Entity;
    rdfs:label "Patient";
    ogit:scope "NTO";
    ogit:parent ogit:Node;
    ogit:mandatory-attributes ( ogit:id );
    ogit:optional-attributes ( ) ;
.
"#;
        let tmp = tempfile::tempdir().unwrap();
        fs::create_dir_all(tmp.path().join("Healthcare")).unwrap();
        fs::write(tmp.path().join("Healthcare").join("ents.ttl"), ttl).unwrap();
        let registry = Arc::new(OntologyRegistry::new_in_memory());
        registry
            .hydrate_once_sync(tmp.path(), &["Healthcare"])
            .unwrap();
        // `hydrate_once_sync` parses the TTL into the in-memory registry, so
        // the temp dir is no longer needed — let it drop (no leak). The
        // earlier `std::mem::forget(tmp)` kept it alive unnecessarily
        // (CodeRabbit, PR #582).
        registry
    }

    #[test]
    fn new_succeeds_when_namespace_registered() {
        let registry = registry_with_healthcare();
        let bridge = MedcareBridge::new(registry).unwrap();
        assert_ne!(bridge.g_lock(), NamespaceId::UNKNOWN);
    }

    #[test]
    fn new_returns_unknown_namespace_when_not_registered() {
        let registry = Arc::new(OntologyRegistry::new_in_memory());
        // `unwrap_err()` would require `MedcareBridge: Debug`, which the
        // underlying `UnifiedBridge<P>` intentionally doesn't impl.
        match MedcareBridge::new(registry) {
            Ok(_) => panic!("expected UnknownNamespace, got Ok(_)"),
            Err(Error::UnknownNamespace(name)) => assert_eq!(name, "Healthcare"),
            Err(other) => panic!("expected UnknownNamespace, got {other:?}"),
        }
    }

    #[test]
    fn bridge_id_is_lowercase_medcare() {
        let bridge = MedcareBridge::new(registry_with_healthcare()).unwrap();
        assert_eq!(bridge.bridge_id(), "medcare");
    }

    #[test]
    fn g_lock_matches_registry_namespace_id() {
        let registry = registry_with_healthcare();
        let expected = registry.namespace_id(NAMESPACE).unwrap();
        let bridge = MedcareBridge::new(registry).unwrap();
        assert_eq!(bridge.g_lock(), expected);
    }

    #[test]
    fn entity_resolves_patient_to_canonical_class_id() {
        let bridge = MedcareBridge::new(registry_with_healthcare()).unwrap();
        let entity = bridge.entity("Patient").unwrap();
        assert_eq!(entity.schema_ptr.entity_type_id(), class_ids::PATIENT);
        assert_eq!(entity.schema_ptr.entity_type_id(), 0x0901);
    }

    #[test]
    fn entity_synthesised_schema_ptr_stamps_seeded_context_id() {
        let bridge = MedcareBridge::new(registry_with_healthcare()).unwrap();
        let entity = bridge.entity("Patient").unwrap();
        let expected = NamespaceRegistry::seed_context_id("Healthcare").unwrap();
        assert_eq!(entity.schema_ptr.ontology_context_id(), expected);
        assert_eq!(entity.schema_ptr.ontology_context_id(), 2);
    }

    #[test]
    fn entity_for_each_codebook_entry_returns_its_canonical_class_id() {
        let bridge = MedcareBridge::new(registry_with_healthcare()).unwrap();
        for &(public_name, expected_id) in HealthcarePort::aliases() {
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
        let bridge = MedcareBridge::new(registry_with_healthcare()).unwrap();
        let err = bridge.entity("NotAConcept").unwrap_err();
        match err {
            BridgeError::NotInScope { public_name, .. } => {
                assert_eq!(public_name, "NotAConcept")
            }
            other => panic!("expected NotInScope, got {other:?}"),
        }
    }
}
