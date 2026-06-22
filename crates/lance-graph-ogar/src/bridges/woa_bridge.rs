//! WoA (work-order management) tenant bridge — thin type alias over
//! [`crate::bridges::unified::UnifiedBridge`] parameterised by
//! [`ogar_vocab::ports::WoaPort`].
//!
//! WoA's OGAR-driven port surface (OGAR PR #93, 2026-06-21). The legacy
//! OGIT-only [`lance_graph_ontology::bridges::WoaBridge`] stays in place
//! for tools that still resolve raw `ogit.WorkOrder:*` URIs through the
//! registry without codebook synthesis; this OGAR-side bridge gives
//! consumers cross-fork-convergent class_ids per the doctrine
//! `docs/OGAR_AR_SHAPE_ENDGAME.md` (*"Curators teach. OGAR compiles.
//! LanceGraph thinks. Adapters obey."*).
//!
//! # The convergence pin (operator value statement 2026-06-21)
//!
//! Planner-side `TimeEntry` (OpenProject/Redmine) and ERP-side
//! `Stundenzettel` / `TimesheetActivity` / `Zeiterfassung` (WoA) all
//! resolve to [`ogar_vocab::class_ids::BILLABLE_WORK_ENTRY`] via this
//! bridge's `entity()` codebook synthesis path. That's the operator's
//! *"planner times align with billable hours"* statement realised as
//! data — the planner→ERP integration is a codebook lookup, not a
//! translation layer. See `ogar_vocab::ports::tests::
//! time_entry_converges_across_planner_and_erp_ports` for the
//! cross-port pin.

use crate::bridges::unified::UnifiedBridge;
// `WoaPort::NAMESPACE` / `::aliases()` are `PortSpec` associated items —
// the trait must be in scope for the resolution to work (same codex P1
// fix as MedcareBridge).
use ogar_vocab::ports::PortSpec;
pub use ogar_vocab::ports::WoaPort;

/// WoA `NamespaceBridge` — alias over the generic harness, locked to
/// the `WorkOrder` namespace via [`WoaPort`].
///
/// **Deprecated:** pull the classid via the OGAR PortSpec instead —
/// `ogar_vocab::ports::WoaPort::class_id(name)`. See
/// `docs/CONSUMER-BRIDGE-DEPRECATION.md` + AdaWorldAPI/OGAR#95.
#[deprecated(
    note = "pull the classid via `WoaPort::class_id(name)` — see AdaWorldAPI/OGAR#95 + docs/CONSUMER-BRIDGE-DEPRECATION.md"
)]
pub type WoaBridge = UnifiedBridge<WoaPort>;

/// Canonical namespace name for WoA. Mirrors `WoaPort::NAMESPACE`.
pub const NAMESPACE: &str = WoaPort::NAMESPACE;

#[cfg(test)]
#[allow(deprecated)] // exercises the deprecated bridge alias on purpose
mod tests {
    use super::*;
    use lance_graph_ontology::bridge::{BridgeError, NamespaceBridge};
    use lance_graph_ontology::error::Error;
    use lance_graph_ontology::namespace::NamespaceId;
    use lance_graph_ontology::registry::OntologyRegistry;
    use ogar_vocab::class_ids;
    use ogar_vocab::ports::PortSpec;
    use std::fs;
    use std::sync::Arc;

    fn registry_with_workorder() -> Arc<OntologyRegistry> {
        let ttl = r#"
@prefix ogit:            <http://www.purl.org/ogit/> .
@prefix ogit.WorkOrder:  <http://www.purl.org/ogit/WorkOrder/> .
@prefix rdfs:            <http://www.w3.org/2000/01/rdf-schema#> .

ogit.WorkOrder:Customer
    a rdfs:Class;
    rdfs:subClassOf ogit:Entity;
    rdfs:label "Customer";
    ogit:scope "NTO";
    ogit:parent ogit:Node;
    ogit:mandatory-attributes ( ogit:id );
    ogit:optional-attributes ( ) ;
.
"#;
        let tmp = tempfile::tempdir().unwrap();
        fs::create_dir_all(tmp.path().join("WorkOrder")).unwrap();
        fs::write(tmp.path().join("WorkOrder").join("ents.ttl"), ttl).unwrap();
        let registry = Arc::new(OntologyRegistry::new_in_memory());
        registry
            .hydrate_once_sync(tmp.path(), &["WorkOrder"])
            .unwrap();
        registry
    }

    #[test]
    fn new_succeeds_when_namespace_registered() {
        let registry = registry_with_workorder();
        let bridge = WoaBridge::new(registry).unwrap();
        assert_ne!(bridge.g_lock(), NamespaceId::UNKNOWN);
    }

    #[test]
    fn new_returns_unknown_namespace_when_not_registered() {
        let registry = Arc::new(OntologyRegistry::new_in_memory());
        match WoaBridge::new(registry) {
            Ok(_) => panic!("expected UnknownNamespace, got Ok(_)"),
            Err(Error::UnknownNamespace(name)) => assert_eq!(name, "WorkOrder"),
            Err(other) => panic!("expected UnknownNamespace, got {other:?}"),
        }
    }

    #[test]
    fn bridge_id_is_lowercase_woa() {
        let bridge = WoaBridge::new(registry_with_workorder()).unwrap();
        assert_eq!(bridge.bridge_id(), "woa");
    }

    #[test]
    fn entity_resolves_customer_to_canonical_billing_party_class_id() {
        let bridge = WoaBridge::new(registry_with_workorder()).unwrap();
        let entity = bridge.entity("Customer").unwrap();
        assert_eq!(entity.schema_ptr.entity_type_id(), class_ids::BILLING_PARTY);
        assert_eq!(entity.schema_ptr.entity_type_id(), 0x0204);
    }

    #[test]
    fn entity_resolves_stundenzettel_to_billable_work_entry_planner_convergence() {
        // The operator's value statement (2026-06-21): WoA's Stundenzettel
        // and OpenProject's TimeEntry resolve to the SAME canonical id.
        // Cross-port convergence is asserted in
        // `ogar_vocab::ports::tests::time_entry_converges_across_planner_and_erp_ports`;
        // this bridge-side test confirms the synthesis path on the entity
        // resolver actually returns the canonical id.
        let bridge = WoaBridge::new(registry_with_workorder()).unwrap();
        for public_name in [
            "Stundenzettel",
            "TimesheetActivity",
            "TimeEntry",
            "Zeiterfassung",
        ] {
            let entity = bridge
                .entity(public_name)
                .unwrap_or_else(|e| panic!("{public_name}: {e:?}"));
            assert_eq!(
                entity.schema_ptr.entity_type_id(),
                class_ids::BILLABLE_WORK_ENTRY,
                "WoA `{public_name}` must resolve to BILLABLE_WORK_ENTRY (planner-ERP convergence)",
            );
        }
    }

    #[test]
    fn entity_for_each_codebook_entry_returns_its_canonical_class_id() {
        let bridge = WoaBridge::new(registry_with_workorder()).unwrap();
        for &(public_name, expected_id) in WoaPort::aliases() {
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
        let bridge = WoaBridge::new(registry_with_workorder()).unwrap();
        match bridge.entity("NotAConcept") {
            Err(BridgeError::NotInScope { public_name, .. }) => {
                assert_eq!(public_name, "NotAConcept")
            }
            other => panic!("expected NotInScope, got {other:?}"),
        }
    }
}
