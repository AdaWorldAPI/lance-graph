//! `lance-graph-consumer-conformance` — cross-crate registry conformance harness.
//!
//! This crate is the CI gate that prevents a consumer crate from shipping a
//! `NamespaceBridge` impl that compiles but violates the `UnifiedBridge`
//! contract semantics.
//!
//! ## How to add a new consumer
//!
//! 1. Add a `#[test]` function (or `#[test] #[ignore]` for scaffolds).
//! 2. Seed a registry with the consumer's `MappingProposal`.
//! 3. Build three `UnifiedBridge<B>` instances (allow / deny / blank).
//! 4. Call `harness::assert_consumer_conformance(...)`.
//!
//! ## Assertions (A1-A10)
//!
//! See [`harness`] module for the full contract description.

pub mod harness;

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use lance_graph_callcenter::super_domain::SuperDomain;
    use lance_graph_callcenter::unified_bridge::{TenantId, UnifiedBridge};
    use lance_graph_contract::property::{Marking, PrefetchDepth, Schema};
    use lance_graph_ontology::bridge::NamespaceBridge;
    use lance_graph_ontology::namespace::OgitUri;
    use lance_graph_ontology::proposal::{MappingProposal, MappingProposalKind};
    use lance_graph_ontology::OntologyRegistry;
    use lance_graph_rbac::permission::PermissionSpec;
    use lance_graph_rbac::policy::Policy;
    use lance_graph_rbac::role::Role;

    use crate::harness::{assert_consumer_conformance, ConformanceFixture, RecordingSink};

    // ═══════════════════════════════════════════════════════════════════════
    // Shared fixture helpers
    // ═══════════════════════════════════════════════════════════════════════

    /// Build a `Policy` that grants `role_name` read-only access to `canonical_entity`.
    fn canonical_allow_policy(role_name: &'static str, canonical_entity: &'static str) -> Policy {
        Policy::new("conformance-allow")
            .with_role(
                Role::new(role_name).with_permission(PermissionSpec::read_at(
                    canonical_entity,
                    PrefetchDepth::Identity,
                )),
            )
    }

    /// Build a `Policy` that grants `role_name` read-only access to
    /// `alias_entity` — note this is the ALIAS, NOT the canonical name. Used
    /// for the A5 deny-on-alias-keyed-policy test.
    fn alias_deny_policy(role_name: &'static str, alias_entity: &'static str) -> Policy {
        Policy::new("conformance-deny-alias")
            .with_role(
                Role::new(role_name).with_permission(PermissionSpec::read_at(
                    alias_entity,
                    PrefetchDepth::Identity,
                )),
            )
    }

    // ═══════════════════════════════════════════════════════════════════════
    // E1 — MedcareBridge / Patient / Healthcare
    // ═══════════════════════════════════════════════════════════════════════

    const MEDCARE_FIXTURE: ConformanceFixture = ConformanceFixture {
        public_name: "Patient",
        canonical_name: "Patient", // ogit.Healthcare:Patient -> local = "Patient"
        super_domain: SuperDomain::Healthcare,
        role_that_can_read: "physician",  // OQ-3 direct migration consumed by MedCare-rs#119
        is_active: true,
    };

    fn seed_medcare_registry() -> Arc<OntologyRegistry> {
        let registry = Arc::new(OntologyRegistry::new_in_memory());
        let uri = OgitUri::parse("ogit.Healthcare:Patient").unwrap();
        registry
            .append_mapping(MappingProposal {
                public_name: "Patient".to_string(),
                bridge_id: "medcare".to_string(),
                ogit_uri: uri,
                namespace: "Healthcare".to_string(),
                kind: MappingProposalKind::Entity {
                    schema: Schema::builder("Patient").required("patient_id").build(),
                },
                marking: Marking::Pii, // closest to "Confidential" in the Marking enum
                confidence: 1.0,
                source_uri: "test://medcare-fixture".to_string(),
                checksum: "checksum-medcare-patient".to_string(),
                created_by: "conformance-test".to_string(),
            })
            .unwrap();
        registry
    }

    #[test]
    fn medcare_bridge_conforms() {
        use lance_graph_ontology::bridges::MedcareBridge;

        let registry = seed_medcare_registry();
        let bridge = Arc::new(MedcareBridge::new(registry).unwrap());

        // Blank bridge: Healthcare namespace seeded with a dummy entity so
        // MedcareBridge::new() succeeds, but "Patient" is absent.
        // This exercises A4 (BridgeError on unknown entity emits no audit).
        let blank_registry = Arc::new(OntologyRegistry::new_in_memory());
        let dummy_uri = OgitUri::parse("ogit.Healthcare:DummyForBlank").unwrap();
        blank_registry
            .append_mapping(MappingProposal {
                public_name: "__dummy__".to_string(),
                bridge_id: "medcare".to_string(),
                ogit_uri: dummy_uri,
                namespace: "Healthcare".to_string(),
                kind: MappingProposalKind::Entity {
                    schema: Schema::builder("DummyForBlank").required("id").build(),
                },
                marking: Marking::Internal,
                confidence: 1.0,
                source_uri: "test://blank".to_string(),
                checksum: "checksum-blank".to_string(),
                created_by: "conformance-test".to_string(),
            })
            .unwrap();
        let blank_bridge = Arc::new(MedcareBridge::new(blank_registry).unwrap());

        let sink_allow: Arc<RecordingSink> = Arc::new(RecordingSink::default());
        let sink_blank: Arc<RecordingSink> = Arc::new(RecordingSink::default());
        let sink_deny: Arc<RecordingSink> = Arc::new(RecordingSink::default());

        let policy_allow = Arc::new(canonical_allow_policy(
            MEDCARE_FIXTURE.role_that_can_read,
            MEDCARE_FIXTURE.canonical_name,
        ));
        // For MedcareBridge: public_name == canonical_name so the "deny on alias"
        // test uses a completely wrong name to demonstrate the deny path.
        let policy_deny = Arc::new(alias_deny_policy(
            MEDCARE_FIXTURE.role_that_can_read,
            "WRONG_ALIAS",
        ));
        let policy_blank = Arc::new(canonical_allow_policy(
            MEDCARE_FIXTURE.role_that_can_read,
            MEDCARE_FIXTURE.canonical_name,
        ));

        let bridge_allow = UnifiedBridge::new(
            bridge.clone(),
            policy_allow,
            MEDCARE_FIXTURE.role_that_can_read,
            TenantId(1),
        )
        .with_audit_chain(MEDCARE_FIXTURE.super_domain, 0xC0FF_EE01_u64, sink_allow.clone());

        let bridge_deny = UnifiedBridge::new(
            bridge.clone(),
            policy_deny,
            MEDCARE_FIXTURE.role_that_can_read,
            TenantId(1),
        )
        .with_audit_chain(MEDCARE_FIXTURE.super_domain, 0xC0FF_EE01_u64, sink_deny.clone());

        let bridge_blank = UnifiedBridge::new(
            blank_bridge,
            policy_blank,
            MEDCARE_FIXTURE.role_that_can_read,
            TenantId(1),
        )
        .with_audit_chain(MEDCARE_FIXTURE.super_domain, 0xC0FF_EE01_u64, sink_blank.clone());

        assert_consumer_conformance(
            &bridge_allow,
            Some(&bridge_deny),
            &bridge_blank,
            &MEDCARE_FIXTURE,
            &sink_allow,
            &sink_blank,
        );
    }

    // ═══════════════════════════════════════════════════════════════════════
    // E2 — OgitBridge / Invoice / WorkOrderBilling (smb-office-rs)
    // ═══════════════════════════════════════════════════════════════════════

    const SMB_FIXTURE: ConformanceFixture = ConformanceFixture {
        public_name: "Invoice",
        canonical_name: "Invoice",
        super_domain: SuperDomain::WorkOrderBilling,
        role_that_can_read: "accountant",
        is_active: true,
    };

    fn seed_ogit_registry_smb() -> Arc<OntologyRegistry> {
        let registry = Arc::new(OntologyRegistry::new_in_memory());
        let uri = OgitUri::parse("ogit.SMB:Invoice").unwrap();
        registry
            .append_mapping(MappingProposal {
                public_name: "Invoice".to_string(),
                bridge_id: "ogit".to_string(),
                ogit_uri: uri,
                namespace: "SMB".to_string(),
                kind: MappingProposalKind::Entity {
                    schema: Schema::builder("Invoice").required("invoice_id").build(),
                },
                marking: Marking::Internal,
                confidence: 1.0,
                source_uri: "test://smb-fixture".to_string(),
                checksum: "checksum-smb-invoice".to_string(),
                created_by: "conformance-test".to_string(),
            })
            .unwrap();
        registry
    }

    #[test]
    fn smb_ogit_bridge_conforms() {
        use lance_graph_ontology::bridges::OgitBridge;

        let registry = seed_ogit_registry_smb();
        let bridge = Arc::new(OgitBridge::for_namespace(registry, "SMB").unwrap());

        // Blank bridge: SMB namespace exists but no "Invoice" row
        let blank_registry = Arc::new(OntologyRegistry::new_in_memory());
        let dummy_uri = OgitUri::parse("ogit.SMB:DummyForBlank").unwrap();
        blank_registry
            .append_mapping(MappingProposal {
                public_name: "__dummy__".to_string(),
                bridge_id: "ogit".to_string(),
                ogit_uri: dummy_uri,
                namespace: "SMB".to_string(),
                kind: MappingProposalKind::Entity {
                    schema: Schema::builder("DummyForBlank").required("id").build(),
                },
                marking: Marking::Internal,
                confidence: 1.0,
                source_uri: "test://blank".to_string(),
                checksum: "checksum-blank-smb".to_string(),
                created_by: "conformance-test".to_string(),
            })
            .unwrap();
        let blank_bridge = Arc::new(OgitBridge::for_namespace(blank_registry, "SMB").unwrap());

        let sink_allow: Arc<RecordingSink> = Arc::new(RecordingSink::default());
        let sink_blank: Arc<RecordingSink> = Arc::new(RecordingSink::default());

        let policy_allow = Arc::new(canonical_allow_policy(
            SMB_FIXTURE.role_that_can_read,
            SMB_FIXTURE.canonical_name,
        ));
        let policy_blank = Arc::new(canonical_allow_policy(
            SMB_FIXTURE.role_that_can_read,
            SMB_FIXTURE.canonical_name,
        ));

        let bridge_allow = UnifiedBridge::new(
            bridge.clone(),
            policy_allow,
            SMB_FIXTURE.role_that_can_read,
            TenantId(1),
        )
        .with_audit_chain(SMB_FIXTURE.super_domain, 0xC0FF_EE02_u64, sink_allow.clone());

        let bridge_blank = UnifiedBridge::new(
            blank_bridge,
            policy_blank,
            SMB_FIXTURE.role_that_can_read,
            TenantId(1),
        )
        .with_audit_chain(SMB_FIXTURE.super_domain, 0xC0FF_EE02_u64, sink_blank.clone());

        // OgitBridge: public_name == canonical_name (no alias gap), bridge_deny = None
        assert_consumer_conformance(
            &bridge_allow,
            None,
            &bridge_blank,
            &SMB_FIXTURE,
            &sink_allow,
            &sink_blank,
        );
    }

    // ═══════════════════════════════════════════════════════════════════════
    // E3 — WoaBridge / WorkOrder→Order / WorkOrderBilling
    // ═══════════════════════════════════════════════════════════════════════

    const WOA_FIXTURE: ConformanceFixture = ConformanceFixture {
        public_name: "WorkOrder",    // bridge alias
        canonical_name: "Order",     // OGIT canonical local name
        super_domain: SuperDomain::WorkOrderBilling,
        role_that_can_read: "dispatcher",
        is_active: true,
    };

    fn seed_woa_registry() -> Arc<OntologyRegistry> {
        let registry = Arc::new(OntologyRegistry::new_in_memory());
        let uri = OgitUri::parse("ogit.WorkOrder:Order").unwrap();
        registry
            .append_mapping(MappingProposal {
                public_name: "WorkOrder".to_string(), // bridge alias
                bridge_id: "woa".to_string(),
                ogit_uri: uri, // canonical = "Order"
                namespace: "WorkOrder".to_string(),
                kind: MappingProposalKind::Entity {
                    schema: Schema::builder("Order").required("order_id").build(),
                },
                marking: Marking::Internal,
                confidence: 1.0,
                source_uri: "test://woa-fixture".to_string(),
                checksum: "checksum-woa-order".to_string(),
                created_by: "conformance-test".to_string(),
            })
            .unwrap();
        registry
    }

    #[test]
    fn woa_bridge_conforms() {
        use lance_graph_ontology::bridges::WoaBridge;

        let registry = seed_woa_registry();
        let bridge = Arc::new(WoaBridge::new(registry).unwrap());

        // Blank bridge: WorkOrder namespace seeded with a dummy entity so
        // WoaBridge::new() succeeds, but "WorkOrder" alias is absent.
        let blank_registry = Arc::new(OntologyRegistry::new_in_memory());
        let dummy_uri = OgitUri::parse("ogit.WorkOrder:DummyForBlank").unwrap();
        blank_registry
            .append_mapping(MappingProposal {
                public_name: "__dummy__".to_string(),
                bridge_id: "woa".to_string(),
                ogit_uri: dummy_uri,
                namespace: "WorkOrder".to_string(),
                kind: MappingProposalKind::Entity {
                    schema: Schema::builder("DummyForBlank").required("id").build(),
                },
                marking: Marking::Internal,
                confidence: 1.0,
                source_uri: "test://blank".to_string(),
                checksum: "checksum-blank-woa".to_string(),
                created_by: "conformance-test".to_string(),
            })
            .unwrap();
        let blank_bridge = Arc::new(WoaBridge::new(blank_registry).unwrap());

        let sink_allow: Arc<RecordingSink> = Arc::new(RecordingSink::default());
        let sink_blank: Arc<RecordingSink> = Arc::new(RecordingSink::default());
        let sink_deny: Arc<RecordingSink> = Arc::new(RecordingSink::default());

        // A5 WoaBridge: policy keyed on canonical "Order" grants;
        // policy keyed on alias "WorkOrder" denies.
        let policy_allow =
            Arc::new(canonical_allow_policy(WOA_FIXTURE.role_that_can_read, "Order"));
        let policy_deny_on_alias =
            Arc::new(alias_deny_policy(WOA_FIXTURE.role_that_can_read, "WorkOrder"));
        let policy_blank =
            Arc::new(canonical_allow_policy(WOA_FIXTURE.role_that_can_read, "Order"));

        let bridge_allow = UnifiedBridge::new(
            bridge.clone(),
            policy_allow,
            WOA_FIXTURE.role_that_can_read,
            TenantId(1),
        )
        .with_audit_chain(WOA_FIXTURE.super_domain, 0xC0FF_EE03_u64, sink_allow.clone());

        let bridge_deny = UnifiedBridge::new(
            bridge.clone(),
            policy_deny_on_alias,
            WOA_FIXTURE.role_that_can_read,
            TenantId(1),
        )
        .with_audit_chain(WOA_FIXTURE.super_domain, 0xC0FF_EE03_u64, sink_deny.clone());

        let bridge_blank = UnifiedBridge::new(
            blank_bridge,
            policy_blank,
            WOA_FIXTURE.role_that_can_read,
            TenantId(1),
        )
        .with_audit_chain(WOA_FIXTURE.super_domain, 0xC0FF_EE03_u64, sink_blank.clone());

        assert_consumer_conformance(
            &bridge_allow,
            Some(&bridge_deny), // A5: alias != canonical — test the deny path
            &bridge_blank,
            &WOA_FIXTURE,
            &sink_allow,
            &sink_blank,
        );
    }

    // ═══════════════════════════════════════════════════════════════════════
    // E4 — hiro-rs scaffold (stub bridge, OWL file not yet seeded)
    // ═══════════════════════════════════════════════════════════════════════

    #[test]
    #[ignore = "hiro-rs: stub bridge, OWL file not yet seeded (E4 scaffold)"]
    fn hiro_bridge_conforms() {
        // FIXTURE reference only — body runs when #[ignore] is lifted (E4 OWL
        // file committed to hiro-rs and HiroBridge is added to this crate's
        // dev-dependencies).
        //
        // static HIRO_FIXTURE: ConformanceFixture = ConformanceFixture {
        //     public_name: "Ticket",
        //     canonical_name: "Ticket",
        //     super_domain: SuperDomain::TicketTool,  // discriminant = 5
        //     role_that_can_read: "agent",
        //     is_active: true,
        // };
        //
        // When HiroBridge is available:
        //   let registry = seed_hiro_registry();
        //   let bridge = Arc::new(HiroBridge::new(registry).unwrap());
        //   ... (same pattern as E1/E2/E3)
        unimplemented!("hiro-rs E4 scaffold: implement when HiroBridge is available")
    }

    // ═══════════════════════════════════════════════════════════════════════
    // E5 — hubspot-rs scaffold (stub bridge, OWL file not yet seeded)
    // ═══════════════════════════════════════════════════════════════════════

    #[test]
    #[ignore = "hubspot-rs: stub bridge, OWL file not yet seeded (E5 scaffold)"]
    fn hubspot_bridge_conforms() {
        // FIXTURE reference only — body runs when #[ignore] is lifted (E5 OWL
        // file committed to hubspot-rs and HubspotBridge is added to this
        // crate's dev-dependencies).
        //
        // static HUBSPOT_FIXTURE: ConformanceFixture = ConformanceFixture {
        //     public_name: "Contact",
        //     canonical_name: "Contact",
        //     super_domain: SuperDomain::Unknown,   // TBD discriminant
        //     role_that_can_read: "sales_rep",
        //     is_active: false,  // scaffold: Unknown is acceptable
        // };
        unimplemented!("hubspot-rs E5 scaffold: implement when HubspotBridge is available")
    }

    // ═══════════════════════════════════════════════════════════════════════
    // Self-test: mock bridge — all 10 assertions pass for a correct bridge
    // ═══════════════════════════════════════════════════════════════════════

    /// Minimal stub bridge for self-test and negative-test purposes.
    /// Locks to a synthetic NamespaceId; resolves any name it was seeded with.
    struct StubBridge {
        registry: Arc<OntologyRegistry>,
        g_lock: lance_graph_ontology::namespace::NamespaceId,
    }

    impl NamespaceBridge for StubBridge {
        fn bridge_id(&self) -> &'static str {
            "stub"
        }
        fn registry(&self) -> &OntologyRegistry {
            &self.registry
        }
        fn g_lock(&self) -> lance_graph_ontology::namespace::NamespaceId {
            self.g_lock
        }
    }

    /// Seed a registry with one entity and return a StubBridge locked to that namespace.
    fn make_stub_bridge(
        namespace: &'static str,
        entity_public_name: &'static str,
        entity_canonical_name: &'static str,
    ) -> StubBridge {
        let registry = Arc::new(OntologyRegistry::new_in_memory());
        let uri_str = format!("ogit.{}:{}", namespace, entity_canonical_name);
        let uri = OgitUri::parse(&uri_str).unwrap();
        registry
            .append_mapping(MappingProposal {
                public_name: entity_public_name.to_string(),
                bridge_id: "stub".to_string(),
                ogit_uri: uri,
                namespace: namespace.to_string(),
                kind: MappingProposalKind::Entity {
                    schema: Schema::builder(entity_canonical_name)
                        .required("id")
                        .build(),
                },
                marking: Marking::Internal,
                confidence: 1.0,
                source_uri: "test://stub".to_string(),
                checksum: format!("checksum-{entity_public_name}"),
                created_by: "self-test".to_string(),
            })
            .unwrap();
        let g_lock = registry.namespace_id(namespace).unwrap();
        StubBridge { registry, g_lock }
    }

    /// Seed a registry with a dummy entity only (so namespace_id succeeds but
    /// the test entity is absent — used for the blank/A4 path).
    fn make_blank_stub_bridge(namespace: &'static str) -> StubBridge {
        let registry = Arc::new(OntologyRegistry::new_in_memory());
        let uri_str = format!("ogit.{}:Dummy", namespace);
        let uri = OgitUri::parse(&uri_str).unwrap();
        registry
            .append_mapping(MappingProposal {
                public_name: "__dummy__".to_string(),
                bridge_id: "stub".to_string(),
                ogit_uri: uri,
                namespace: namespace.to_string(),
                kind: MappingProposalKind::Entity {
                    schema: Schema::builder("Dummy").required("id").build(),
                },
                marking: Marking::Internal,
                confidence: 1.0,
                source_uri: "test://blank".to_string(),
                checksum: "checksum-dummy".to_string(),
                created_by: "self-test".to_string(),
            })
            .unwrap();
        let g_lock = registry.namespace_id(namespace).unwrap();
        StubBridge { registry, g_lock }
    }

    #[test]
    fn self_test_mock_bridge_all_assertions_pass() {
        let bridge = Arc::new(make_stub_bridge("SelfTest", "Widget", "Widget"));
        let blank_bridge = Arc::new(make_blank_stub_bridge("SelfTest"));

        let sink_allow: Arc<RecordingSink> = Arc::new(RecordingSink::default());
        let sink_blank: Arc<RecordingSink> = Arc::new(RecordingSink::default());

        let fixture = ConformanceFixture {
            public_name: "Widget",
            canonical_name: "Widget",
            super_domain: SuperDomain::WorkOrderBilling,
            role_that_can_read: "tester",
            is_active: true,
        };

        let policy_allow = Arc::new(canonical_allow_policy("tester", "Widget"));
        let policy_blank = Arc::new(canonical_allow_policy("tester", "Widget"));

        let bridge_allow = UnifiedBridge::new(bridge.clone(), policy_allow, "tester", TenantId(1))
            .with_audit_chain(SuperDomain::WorkOrderBilling, 0xDEAD_BEEF, sink_allow.clone());
        let bridge_blank = UnifiedBridge::new(blank_bridge, policy_blank, "tester", TenantId(1))
            .with_audit_chain(SuperDomain::WorkOrderBilling, 0xDEAD_BEEF, sink_blank.clone());

        assert_consumer_conformance(
            &bridge_allow,
            None,
            &bridge_blank,
            &fixture,
            &sink_allow,
            &sink_blank,
        );
    }

    // ═══════════════════════════════════════════════════════════════════════
    // Negative tests — each assertion catches a deliberately-broken bridge
    // ═══════════════════════════════════════════════════════════════════════

    #[test]
    fn negative_a9_wrong_role_hash_is_caught() {
        // Construct a bridge with role "role_a"; verify its hash differs from "role_b".
        use lance_graph_contract::hash::fnv1a_str;

        let bridge = Arc::new(make_stub_bridge("NegTest", "Thing", "Thing"));
        let sink: Arc<RecordingSink> = Arc::new(RecordingSink::default());
        let policy = Arc::new(canonical_allow_policy("role_a", "Thing"));
        let unified = UnifiedBridge::new(bridge, policy, "role_a", TenantId(1))
            .with_audit_chain(SuperDomain::WorkOrderBilling, 0, sink.clone());

        let _ = unified
            .authorize_read("Thing", PrefetchDepth::Identity)
            .expect("allow");
        let events = sink.snapshot();
        assert_eq!(events.len(), 1);

        // role_a hash must match
        assert_eq!(events[0].actor_role_hash, fnv1a_str("role_a"),
            "A9 negative: actor_role_hash must equal fnv1a_str('role_a')");
        // role_b hash must NOT match
        assert_ne!(events[0].actor_role_hash, fnv1a_str("role_b"),
            "A9 negative: hash for 'role_a' must differ from hash for 'role_b'");
    }

    #[test]
    fn negative_a4_blank_bridge_emits_no_event_on_unknown_entity() {
        // A BridgeError on unknown entity must not emit an audit event.
        let blank_bridge = Arc::new(make_blank_stub_bridge("NegTest4"));
        let sink: Arc<RecordingSink> = Arc::new(RecordingSink::default());
        let policy = Arc::new(canonical_allow_policy("tester", "Thing"));
        let unified = UnifiedBridge::new(blank_bridge, policy, "tester", TenantId(1))
            .with_audit_chain(SuperDomain::WorkOrderBilling, 0, sink.clone());

        let result = unified.authorize_read("__nonexistent__", PrefetchDepth::Identity);
        assert!(result.is_err(), "A4 negative: expect BridgeError for unknown entity");
        assert!(
            sink.is_empty(),
            "A4 negative: no audit event must be emitted on BridgeError; got {} events",
            sink.len()
        );
    }

    #[test]
    fn negative_a8_tenant_isolation_verified() {
        // Two bridges with different TenantId values must emit distinct tenant fields.
        let bridge_1 = Arc::new(make_stub_bridge("TenantTest", "Item", "Item"));
        let bridge_42 = Arc::new(make_stub_bridge("TenantTest", "Item", "Item"));
        let sink_1: Arc<RecordingSink> = Arc::new(RecordingSink::default());
        let sink_42: Arc<RecordingSink> = Arc::new(RecordingSink::default());

        let policy1 = Arc::new(canonical_allow_policy("tester", "Item"));
        let policy2 = Arc::new(canonical_allow_policy("tester", "Item"));

        let unified_1 = UnifiedBridge::new(bridge_1, policy1, "tester", TenantId(1))
            .with_audit_chain(SuperDomain::WorkOrderBilling, 0, sink_1.clone());
        let unified_42 = UnifiedBridge::new(bridge_42, policy2, "tester", TenantId(42))
            .with_audit_chain(SuperDomain::WorkOrderBilling, 0, sink_42.clone());

        let _ = unified_1.authorize_read("Item", PrefetchDepth::Identity).unwrap();
        let _ = unified_42.authorize_read("Item", PrefetchDepth::Identity).unwrap();

        let events_1 = sink_1.snapshot();
        let events_42 = sink_42.snapshot();

        assert_eq!(events_1[0].tenant, TenantId(1),
            "A8 negative: TenantId(1) bridge must emit tenant=1");
        assert_eq!(events_42[0].tenant, TenantId(42),
            "A8 negative: TenantId(42) bridge must emit tenant=42");
        assert_ne!(events_1[0].tenant, events_42[0].tenant,
            "A8 negative: tenant fields must be distinct for different TenantIds");
    }

    #[test]
    fn negative_a3_merkle_chain_advances() {
        // Verify merkle chain property directly on a stub bridge.
        use lance_graph_callcenter::unified_audit::AuditMerkleRoot;

        let bridge = Arc::new(make_stub_bridge("MerkleTest", "Node", "Node"));
        let sink: Arc<RecordingSink> = Arc::new(RecordingSink::default());
        let policy = Arc::new(canonical_allow_policy("tester", "Node"));
        let unified = UnifiedBridge::new(bridge, policy, "tester", TenantId(1))
            .with_audit_chain(SuperDomain::WorkOrderBilling, 0xBEEF, sink.clone());

        let _ = unified.authorize_read("Node", PrefetchDepth::Identity).unwrap();
        let _ = unified.authorize_read("Node", PrefetchDepth::Identity).unwrap();
        let _ = unified.authorize_read("Node", PrefetchDepth::Identity).unwrap();

        let events = sink.snapshot();
        assert_eq!(events.len(), 3);
        assert_ne!(events[0].merkle_root, events[1].merkle_root,
            "A3 negative: consecutive roots must differ");
        assert_ne!(events[1].merkle_root, events[2].merkle_root,
            "A3 negative: consecutive roots must differ");
        for ev in &events {
            assert_ne!(ev.merkle_root, AuditMerkleRoot::GENESIS,
                "A3 negative: roots must not equal GENESIS after advance");
        }
    }
}
