//! Odoo (ERP) tenant bridge — thin type alias over
//! [`crate::bridges::unified::UnifiedBridge`] parameterised by
//! [`ogar_vocab::ports::OdooPort`].
//!
//! Odoo's OGAR-driven port surface (OGAR PR #94, 2026-06-21). Companion
//! to the [`lance_graph_supervisor::actors::OdooConsumerActor`]
//! skeleton (lance-graph activation profile, supervisor slot G=50) and
//! the four-way alignment seam in
//! `lance_graph_callcenter::odoo_alignment` (Seam decision 1 /
//! Option B — odoo inherits FIBO/SKR family slots via
//! `owl:equivalentClass` routing; no new CAM codebook family minted).
//!
//! # The convergence pin (operator value statement 2026-06-21)
//!
//! Odoo's `HrAttendance` and `account.move.line(qty=hours)` resolve to
//! the SAME canonical [`ogar_vocab::class_ids::BILLABLE_WORK_ENTRY`]
//! the planner consumers (OpenProject `TimeEntry`, Redmine `TimeEntry`)
//! and the German ERP consumers (WoA `Stundenzettel`, SMB
//! `Stundenzettel`) all resolve to. The planner→ERP→billing chain
//! collapses into one codebook lookup at every hop.

use crate::bridges::unified::UnifiedBridge;
pub use ogar_vocab::ports::OdooPort;
use ogar_vocab::ports::PortSpec;

/// Odoo `NamespaceBridge` — alias over the generic harness, locked to
/// the `Odoo` namespace via [`OdooPort`].
pub type OdooBridge = UnifiedBridge<OdooPort>;

/// Canonical namespace name for Odoo. Mirrors `OdooPort::NAMESPACE`.
pub const NAMESPACE: &str = OdooPort::NAMESPACE;

#[cfg(test)]
mod tests {
    use super::*;
    use lance_graph_ontology::bridge::{BridgeError, NamespaceBridge};
    use lance_graph_ontology::error::Error;
    use lance_graph_ontology::namespace::NamespaceId;
    use lance_graph_ontology::registry::OntologyRegistry;
    use ogar_vocab::ports::PortSpec;
    use std::fs;
    use std::sync::Arc;

    fn registry_with_odoo() -> Arc<OntologyRegistry> {
        let ttl = r#"
@prefix ogit:       <http://www.purl.org/ogit/> .
@prefix ogit.Odoo:  <http://www.purl.org/ogit/Odoo/> .
@prefix rdfs:       <http://www.w3.org/2000/01/rdf-schema#> .

ogit.Odoo:Partner
    a rdfs:Class;
    rdfs:subClassOf ogit:Entity;
    rdfs:label "Partner";
    ogit:scope "NTO";
    ogit:parent ogit:Node;
    ogit:mandatory-attributes ( ogit:id );
    ogit:optional-attributes ( ) ;
.
"#;
        let tmp = tempfile::tempdir().unwrap();
        fs::create_dir_all(tmp.path().join("Odoo")).unwrap();
        fs::write(tmp.path().join("Odoo").join("ents.ttl"), ttl).unwrap();
        let registry = Arc::new(OntologyRegistry::new_in_memory());
        registry.hydrate_once_sync(tmp.path(), &["Odoo"]).unwrap();
        registry
    }

    #[test]
    fn new_succeeds_when_namespace_registered() {
        let registry = registry_with_odoo();
        let bridge = OdooBridge::new(registry).unwrap();
        assert_ne!(bridge.g_lock(), NamespaceId::UNKNOWN);
    }

    #[test]
    fn new_returns_unknown_namespace_when_not_registered() {
        let registry = Arc::new(OntologyRegistry::new_in_memory());
        match OdooBridge::new(registry) {
            Ok(_) => panic!("expected UnknownNamespace, got Ok(_)"),
            Err(Error::UnknownNamespace(name)) => assert_eq!(name, "Odoo"),
            Err(other) => panic!("expected UnknownNamespace, got {other:?}"),
        }
    }

    #[test]
    fn bridge_id_is_lowercase_odoo() {
        let bridge = OdooBridge::new(registry_with_odoo()).unwrap();
        assert_eq!(bridge.bridge_id(), "odoo");
    }

    #[test]
    fn entity_for_each_codebook_entry_returns_its_canonical_class_id() {
        let bridge = OdooBridge::new(registry_with_odoo()).unwrap();
        for &(public_name, expected_id) in OdooPort::aliases() {
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
        let bridge = OdooBridge::new(registry_with_odoo()).unwrap();
        match bridge.entity("NotAConcept") {
            Err(BridgeError::NotInScope { public_name, .. }) => {
                assert_eq!(public_name, "NotAConcept")
            }
            other => panic!("expected NotInScope, got {other:?}"),
        }
    }
}
