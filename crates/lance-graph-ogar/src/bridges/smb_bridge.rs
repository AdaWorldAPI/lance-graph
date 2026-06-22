//! SMB (small-and-medium-business German office ERP) tenant bridge —
//! thin type alias over [`crate::bridges::unified::UnifiedBridge`]
//! parameterised by [`ogar_vocab::ports::SmbPort`].
//!
//! SMB's OGAR-driven port surface (OGAR PR #93, 2026-06-21). The legacy
//! OGIT-side [`lance_graph_ontology::bridges::OgitBridge`] (pass-through
//! for raw OGIT URIs) stays in place for tools that don't need codebook
//! synthesis; this OGAR-side bridge gives smb-office-rs cross-fork
//! convergence with WoA + OpenProject + Odoo on the canonical class_ids.
//!
//! # The convergence pin (operator value statement 2026-06-21)
//!
//! SMB's `Stundenzettel` / `TimeEntry` / `Zeiterfassung` resolve to
//! [`ogar_vocab::class_ids::BILLABLE_WORK_ENTRY`] via this bridge's
//! `entity()` codebook synthesis path — the SAME id WoA's Stundenzettel,
//! OpenProject's TimeEntry, and Odoo's HrAttendance / account.move.line
//! (qty=hours) resolve to. *"Planner times align with billable hours"*
//! becomes a codebook lookup, not a translation layer. See
//! `ogar_vocab::ports::tests::time_entry_converges_across_planner_and_erp_ports`.

use crate::bridges::unified::UnifiedBridge;
use ogar_vocab::ports::PortSpec;
pub use ogar_vocab::ports::SmbPort;

/// SMB `NamespaceBridge` — alias over the generic harness, locked to
/// the `SMB` namespace via [`SmbPort`].
pub type SmbBridge = UnifiedBridge<SmbPort>;

/// Canonical namespace name for SMB. Mirrors `SmbPort::NAMESPACE`.
pub const NAMESPACE: &str = SmbPort::NAMESPACE;

#[cfg(test)]
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

    fn registry_with_smb() -> Arc<OntologyRegistry> {
        let ttl = r#"
@prefix ogit:      <http://www.purl.org/ogit/> .
@prefix ogit.SMB:  <http://www.purl.org/ogit/SMB/> .
@prefix rdfs:      <http://www.w3.org/2000/01/rdf-schema#> .

ogit.SMB:Kunde
    a rdfs:Class;
    rdfs:subClassOf ogit:Entity;
    rdfs:label "Kunde";
    ogit:scope "NTO";
    ogit:parent ogit:Node;
    ogit:mandatory-attributes ( ogit:id );
    ogit:optional-attributes ( ) ;
.
"#;
        let tmp = tempfile::tempdir().unwrap();
        fs::create_dir_all(tmp.path().join("SMB")).unwrap();
        fs::write(tmp.path().join("SMB").join("ents.ttl"), ttl).unwrap();
        let registry = Arc::new(OntologyRegistry::new_in_memory());
        registry.hydrate_once_sync(tmp.path(), &["SMB"]).unwrap();
        registry
    }

    #[test]
    fn new_succeeds_when_namespace_registered() {
        let registry = registry_with_smb();
        let bridge = SmbBridge::new(registry).unwrap();
        assert_ne!(bridge.g_lock(), NamespaceId::UNKNOWN);
    }

    #[test]
    fn new_returns_unknown_namespace_when_not_registered() {
        let registry = Arc::new(OntologyRegistry::new_in_memory());
        match SmbBridge::new(registry) {
            Ok(_) => panic!("expected UnknownNamespace, got Ok(_)"),
            Err(Error::UnknownNamespace(name)) => assert_eq!(name, "SMB"),
            Err(other) => panic!("expected UnknownNamespace, got {other:?}"),
        }
    }

    #[test]
    fn bridge_id_is_lowercase_smb() {
        let bridge = SmbBridge::new(registry_with_smb()).unwrap();
        assert_eq!(bridge.bridge_id(), "smb");
    }

    #[test]
    fn entity_resolves_kunde_to_canonical_billing_party_class_id() {
        let bridge = SmbBridge::new(registry_with_smb()).unwrap();
        let entity = bridge.entity("Kunde").unwrap();
        assert_eq!(entity.schema_ptr.entity_type_id(), class_ids::BILLING_PARTY);
        assert_eq!(entity.schema_ptr.entity_type_id(), 0x0204);
    }

    #[test]
    fn entity_resolves_stundenzettel_to_billable_work_entry_planner_convergence() {
        let bridge = SmbBridge::new(registry_with_smb()).unwrap();
        for public_name in ["Stundenzettel", "TimeEntry", "Zeiterfassung"] {
            let entity = bridge
                .entity(public_name)
                .unwrap_or_else(|e| panic!("{public_name}: {e:?}"));
            assert_eq!(
                entity.schema_ptr.entity_type_id(),
                class_ids::BILLABLE_WORK_ENTRY,
                "SMB `{public_name}` must resolve to BILLABLE_WORK_ENTRY (planner-ERP convergence)",
            );
        }
    }

    #[test]
    fn entity_for_each_codebook_entry_returns_its_canonical_class_id() {
        let bridge = SmbBridge::new(registry_with_smb()).unwrap();
        for &(public_name, expected_id) in SmbPort::aliases() {
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
        let bridge = SmbBridge::new(registry_with_smb()).unwrap();
        match bridge.entity("Artikel") {
            // Artikel/Product/SKU isn't in the codebook yet (intentional —
            // needs a `0x02XX` codebook extension). Until then it falls
            // through to the registry-resolution path which returns
            // NotInScope because the TTL fixture only hydrates `Kunde`.
            Err(BridgeError::NotInScope { public_name, .. }) => {
                assert_eq!(public_name, "Artikel")
            }
            other => panic!("expected NotInScope, got {other:?}"),
        }
    }
}
