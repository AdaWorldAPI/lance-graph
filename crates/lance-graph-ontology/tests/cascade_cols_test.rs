//! D-CASCADE-V1-7 + D-PARITY-V2-12 — codec cascade columns + thinking_style
//! + AttributeProvenance threading on `MappingRow`.

use lance_graph_contract::property::{LinkSpec, Marking, Schema};
use lance_graph_contract::thinking::ThinkingStyle;
use lance_graph_ontology::namespace::OgitUri;
use lance_graph_ontology::proposal::{
    AttributeProvenance, IdentityCodec, MappingProposalKind, ProvenanceBundle, QualiaMeta,
};
use lance_graph_ontology::{MappingProposal, OntologyRegistry};

fn proposal(uri: &str, kind: MappingProposalKind) -> MappingProposal {
    let parsed = OgitUri::parse(uri).unwrap();
    let ns = parsed.namespace().unwrap().to_string();
    let name = parsed.name().unwrap().to_string();
    MappingProposal {
        public_name: name,
        bridge_id: "ogit".into(),
        ogit_uri: parsed,
        namespace: ns,
        kind,
        marking: Marking::Internal,
        confidence: 1.0,
        source_uri: format!("test://{uri}"),
        checksum: format!("ck-{uri}"),
        created_by: "test".into(),
    }
}

fn entity(name: &str) -> MappingProposalKind {
    MappingProposalKind::Entity {
        schema: Schema::builder(Box::leak(name.to_string().into_boxed_str())).build(),
    }
}

fn edge(pred: &str, s: &'static str, o: &'static str) -> MappingProposalKind {
    MappingProposalKind::Edge {
        link: LinkSpec::many_to_many(s, Box::leak(pred.to_string().into_boxed_str()), o),
    }
}

#[test]
fn columns_default_then_attach_round_trips() {
    // Gap 1: column-presence + AttributeProvenance round-trip + ThinkingStyle.
    let reg = OntologyRegistry::new_in_memory();
    reg.append_mapping(proposal("ogit.WorkOrder:Customer", entity("Customer")))
        .unwrap();
    let row = reg.row_for_uri("ogit.WorkOrder:Customer").unwrap();
    assert_eq!(row.identity_codec, IdentityCodec::default());
    assert_eq!(row.qualia_meta, QualiaMeta::default());
    assert!(row.thinking_style.is_none());
    assert_eq!(row.entity_type_ref, "Customer");

    let bundle = ProvenanceBundle {
        entity_uri: "ogit.WorkOrder:Customer".into(),
        entity_source_uri: "AdaWorldAPI/WoA/models.py:Customer".into(),
        attribute_sources: vec![AttributeProvenance {
            predicate_iri: "ogit.WorkOrder:fahrtKm".into(),
            source_uri: "AdaWorldAPI/WoA/models.py:Customer.fahrt_km".into(),
        }],
    };
    assert!(reg.attach_provenance(&bundle));
    assert!(reg.attach_thinking_style("ogit.WorkOrder:Customer", ThinkingStyle::Pragmatic));
    let row = reg.row_for_uri("ogit.WorkOrder:Customer").unwrap();
    assert_eq!(row.attribute_source_count(), 1);
    assert_eq!(row.thinking_style, Some(ThinkingStyle::Pragmatic));
}

#[test]
fn link_and_entity_type_id_resolution() {
    // Gap 2 (driver.rs:311) + Gap 3 (META-NUDGE-1).
    let reg = OntologyRegistry::new_in_memory();
    reg.append_mapping(proposal(
        "ogit.WorkOrder:assignedTo",
        edge("assignedTo", "Order", "User"),
    ))
    .unwrap();
    let row = reg.row_for_uri("ogit.WorkOrder:assignedTo").unwrap();
    assert_eq!(row.subject_type, "Order");
    assert_eq!(row.object_type, "User");

    let h = reg
        .append_mapping(proposal("ogit.Healthcare:Patient", entity("Patient")))
        .unwrap();
    let resolved = reg
        .enumerate_first_with_entity_type_id(h.schema_ptr.entity_type_id())
        .unwrap();
    assert_eq!(resolved.public_name, "Patient");
    // Healthcare is seeded to ontology_context_id = 2 in
    // NamespaceRegistry::seed_defaults() — the Codex P1 fix in PR #364
    // makes RegistryState::append stamp the seeded id onto SchemaPtr so
    // the MulThresholdProfile MEDICAL/CALLCENTER lookup at
    // driver.rs:303-321 actually fires for Healthcare rows. The
    // previous `== 0` was written before that fix landed.
    assert_eq!(resolved.ontology_context_id(), 2);
}
