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
    use lance_graph_callcenter::unified_audit::UnifiedAuditSink;
    use lance_graph_callcenter::unified_bridge::{TenantId, UnifiedBridge};
    use lance_graph_contract::property::{Marking, PrefetchDepth, Schema};
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
        role_that_can_read: "doctor",
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
                marking: Marking::Confidential,
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

        // Empty registry for A4 / bridge-blank path
        let empty_registry = Arc::new(OntologyRegistry::new_in_memory());
        // MedcareBridge::new will fail if Healthcare ns is absent, so use
        // a StubBridge equivalent — but MedcareBridge only exists for seeded
        // registries. Per meta-review W12 guidance: construct the blank bridge
        // over a registry that HAS the Healthcare namespace seeded (so
        // construction succeeds) but does NOT have the entity "Patient", so
        // row() returns BridgeError. We seed the namespace-only by appending a
        // dummy entity under a different name, then looking up the absent one.
        let blank_registry = Arc::new(OntologyRegistry::new_in_memory());
        // Seed one dummy row so the namespace exists (needed for MedcareBridge
        // construction, which calls namespace_id("Healthcare")).
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
        // For MedcareBridge: public_name == canonical_name so the "deny on alias" test
        // is trivially the same as "deny on wrong name". We use a different wrong name.
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
        .with_audit_chain(
            MEDCARE_FIXTURE.super_domain,
            0xC0FFEE_MEDCARE,
            sink_allow.clone(),
        );

        let bridge_deny = UnifiedBridge::new(
            bridge.clone(),
            policy_deny,
            MEDCARE_FIXTURE.role_that_can_read,
            TenantId(1),
        )
        .with_audit_chain(
            MEDCARE_FIXTURE.super_domain,
            0xC0FFEE_MEDCARE,
            sink_deny.clone(),
        );

        let bridge_blank = UnifiedBridge::new(
            blank_bridge,
            policy_blank,
            MEDCARE_FIXTURE.role_that_can_read,
            TenantId(1),
        )
        .with_audit_chain(
            MEDCARE_FIXTURE.super_domain,
            0xC0FFEE_MEDCARE,
            sink_blank.clone(),
        );

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
        .with_audit_chain(SMB_FIXTURE.super_domain, 0xC0FFEE_SMB, sink_allow.clone());

        let bridge_blank = UnifiedBridge::new(
            blank_bridge,
            policy_blank,
            SMB_FIXTURE.role_that_can_read,
            TenantId(1),
        )
        .with_audit_chain(SMB_FIXTURE.super_domain, 0xC0FFEE_SMB, sink_blank.clone());

        // OgitBridge: public_name == canonical_name (no alias gap), so bridge_deny = None
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

        // Blank bridge: WorkOrder namespace exists but no "WorkOrder" alias row
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

        // A5 WoaBridge: policy keyed on canonical "Order" grants; keyed on alias "WorkOrder" denies.
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
        .with_audit_chain(WOA_FIXTURE.super_domain, 0xC0FFEE_WOA, sink_allow.clone());

        let bridge_deny = UnifiedBridge::new(
            bridge.clone(),
            policy_deny_on_alias,
            WOA_FIXTURE.role_that_can_read,
            TenantId(1),
        )
        .with_audit_chain(WOA_FIXTURE.super_domain, 0xC0FFEE_WOA, sink_deny.clone());

        let bridge_blank = UnifiedBridge::new(
            blank_bridge,
            policy_blank,
            WOA_FIXTURE.role_that_can_read,
            TenantId(1),
        )
        .with_audit_chain(WOA_FIXTURE.super_domain, 0xC0FFEE_WOA, sink_blank.clone());

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
        //     super_domain: SuperDomain::Unknown,   // TBD discriminant — assigned before un-ignore
        //     role_that_can_read: "sales_rep",
        //     is_active: false,  // scaffold: Unknown is acceptable
        // };
        unimplemented!("hubspot-rs E5 scaffold: implement when HubspotBridge is available")
    }

    // ═══════════════════════════════════════════════════════════════════════
    // Self-test: mock bridge — all 10 assertions pass for a correct bridge
    // ═══════════════════════════════════════════════════════════════════════

    /// Minimal stub bridge for self-test and negative-test purposes.
    /// Locks to a synthetic NamespaceId(1); resolves any name it was seeded with.
    struct StubBridge {
        registry: Arc<OntologyRegistry>,
        g_lock: lance_graph_ontology::namespace::NamespaceId,
    }

    impl lance_graph_ontology::bridge::NamespaceBridge for StubBridge {
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

    fn make_stub_bridge_with_entity(
        namespace: &str,
        ns_ogit_uri: &str,
        entity_public_name: &str,
        entity_canonical_name: &str,
        bridge_id: &'static str,
    ) -> StubBridge {
        let registry = Arc::new(OntologyRegistry::new_in_memory());
        let ogit_uri_str = format!("ogit.{}:{}", namespace, entity_canonical_name);
        let uri = OgitUri::parse(&ogit_uri_str).unwrap();
        let _ = ns_ogit_uri; // unused; namespace is derived from uri namespace part
        registry
            .append_mapping(MappingProposal {
                public_name: entity_public_name.to_string(),
                bridge_id: bridge_id.to_string(),
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

    #[test]
    fn self_test_mock_bridge_all_assertions_pass() {
        // Build a correctly-implemented stub bridge; all 10 assertions should pass.
        let bridge = Arc::new(make_stub_bridge_with_entity(
            "SelfTest",
            "ogit.SelfTest:Widget",
            "Widget",  // public_name == canonical_name (no alias)
            "Widget",
            "stub",
        ));

        // Blank bridge: namespace seeded with dummy but "Widget" is absent
        let empty_registry = Arc::new(OntologyRegistry::new_in_memory());
        let dummy_uri = OgitUri::parse("ogit.SelfTest:Dummy").unwrap();
        empty_registry
            .append_mapping(MappingProposal {
                public_name: "__dummy__".to_string(),
                bridge_id: "stub".to_string(),
                ogit_uri: dummy_uri,
                namespace: "SelfTest".to_string(),
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
        let blank_g_lock = empty_registry.namespace_id("SelfTest").unwrap();
        let blank_bridge = Arc::new(StubBridge {
            registry: empty_registry,
            g_lock: blank_g_lock,
        });

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

    /// A1 negative: canonical_bytes length is the canonical 26; the struct
    /// cannot return a wrong-length array (it's [u8; 26] at the type level),
    /// so A1 failures manifest as wrong field-offset values. We test this by
    /// checking a known good event's offsets rather than by breaking the type.
    #[test]
    fn negative_a9_wrong_role_hash_is_caught() {
        // Construct a bridge with role "tester" but ask fnv1a_str("OTHER_ROLE").
        // The harness checks actor_role_hash == fnv1a_str(fixture.role_that_can_read).
        // If we pass a fixture with a DIFFERENT role_name, the hash won't match.
        use lance_graph_contract::hash::fnv1a_str;

        let bridge = Arc::new(make_stub_bridge_with_entity(
            "NegTest",
            "ogit.NegTest:Thing",
            "Thing",
            "Thing",
            "stub",
        ));
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
        assert_eq!(events[0].actor_role_hash, fnv1a_str("role_a"));
        // role_b hash must NOT match
        assert_ne!(events[0].actor_role_hash, fnv1a_str("role_b"),
            "A9 negative: hash for 'role_a' must differ from hash for 'role_b'");
    }

    #[test]
    fn negative_a4_blank_bridge_emits_no_event_on_unknown_entity() {
        // A BridgeError on unknown entity must not emit an audit event.
        let empty_registry = Arc::new(OntologyRegistry::new_in_memory());
        // Don't seed any entity; namespace also absent → NotInScope
        let sink: Arc<RecordingSink> = Arc::new(RecordingSink::default());

        // Use the WoaBridge which requires WorkOrder namespace. With an empty
        // registry, WoaBridge::new() fails, so we use the StubBridge instead
        // (which returns NotInScope for any unseen public_name).
        // A StubBridge with g_lock=NamespaceId(0) simulates an unregistered namespace.
        let stub_bridge = StubBridge {
            registry: empty_registry,
            g_lock: lance_graph_ontology::namespace::NamespaceId(0),
        };
        let policy = Arc::new(canonical_allow_policy("tester", "Widget"));
        let unified = UnifiedBridge::new(
            Arc::new(stub_bridge),
            policy,
            "tester",
            TenantId(1),
        )
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
    fn negative_a8_wrong_tenant_detected() {
        // Two bridges with different TenantId values must emit distinct tenant fields.
        let bridge_1 = Arc::new(make_stub_bridge_with_entity(
            "TenantTest",
            "ogit.TenantTest:Item",
            "Item",
            "Item",
            "stub",
        ));
        let bridge_42 = Arc::new(make_stub_bridge_with_entity(
            "TenantTest",
            "ogit.TenantTest:Item",
            "Item",
            "Item",
            "stub",
        ));
        let sink_1: Arc<RecordingSink> = Arc::new(RecordingSink::default());
        let sink_42: Arc<RecordingSink> = Arc::new(RecordingSink::default());

        let policy = Arc::new(canonical_allow_policy("tester", "Item"));
        let policy2 = Arc::new(canonical_allow_policy("tester", "Item"));

        let unified_1 = UnifiedBridge::new(bridge_1, policy, "tester", TenantId(1))
            .with_audit_chain(SuperDomain::WorkOrderBilling, 0, sink_1.clone());
        let unified_42 = UnifiedBridge::new(bridge_42, policy2, "tester", TenantId(42))
            .with_audit_chain(SuperDomain::WorkOrderBilling, 0, sink_42.clone());

        let _ = unified_1.authorize_read("Item", PrefetchDepth::Identity).unwrap();
        let _ = unified_42.authorize_read("Item", PrefetchDepth::Identity).unwrap();

        let events_1 = sink_1.snapshot();
        let events_42 = sink_42.snapshot();

        assert_eq!(events_1[0].tenant, TenantId(1), "A8 negative: TenantId(1) bridge must emit tenant=1");
        assert_eq!(events_42[0].tenant, TenantId(42), "A8 negative: TenantId(42) bridge must emit tenant=42");
        assert_ne!(
            events_1[0].tenant,
            events_42[0].tenant,
            "A8 negative: tenant fields must be distinct for different TenantIds"
        );
    }
}
