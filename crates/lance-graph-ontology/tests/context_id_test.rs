//! D-CASCADE-V1-2 acceptance: ontology_context_id field on SchemaPtr +
//! NamespaceRegistry sidecar with v1 seed allocations.
//!
//! Per `.claude/plans/ogit-cascade-supabase-callcenter-v1.md` §Pillar 1
//! ("OGIT as the universal SPO-G lingua franca") and
//! `.claude/plans/lance-graph-rdf-fma-snomed-v1.md` §Core types
//! (`OntologyContextId(u32)` reference design).

use lance_graph_ontology::namespace::{NamespaceId, SchemaKind, SchemaPtr};
use lance_graph_ontology::namespace_registry::NamespaceRegistry;

#[test]
fn schema_ptr_defaults_context_id_to_zero_for_back_compat() {
    let ptr = SchemaPtr::new(NamespaceId(7), 42, SchemaKind::Entity);
    assert_eq!(
        ptr.ontology_context_id(),
        0,
        "default ontology_context_id must be 0 so existing registry.rs + \
         lance_cache.rs construction sites compile unchanged",
    );
}

#[test]
fn schema_ptr_with_context_id_preserves_packed_layout() {
    let base = SchemaPtr::new(NamespaceId(7), 42, SchemaKind::Entity);
    let with = base.with_context_id(13);
    assert_eq!(with.ontology_context_id(), 13);
    assert_eq!(with.namespace_id(), NamespaceId(7));
    assert_eq!(with.entity_type_id(), 42);
    assert_eq!(with.kind(), SchemaKind::Entity);
    // raw() returns ONLY the packed [ns|etid|kind] bits — context id rides
    // in a sibling field so it does not pollute the packed u32.
    assert_eq!(with.raw(), base.raw());
}

#[test]
fn namespace_registry_seed_defaults_assigns_canonical_v1_ids() {
    let r = NamespaceRegistry::seed_defaults();

    // Acceptance criterion 1 (per the bound prompt): Healthcare == 2.
    assert_eq!(r.get("Healthcare"), Some(2));

    // Acceptance criterion 2: Medical/ICD10CM == 10.
    assert_eq!(r.get("Medical/ICD10CM"), Some(10));

    // Live cognitive namespaces.
    assert_eq!(r.get("WorkOrder"), Some(1));
    assert_eq!(r.get("Network"), Some(3));
    // SMB seeded as 0 (export-only per v5 ratification).
    assert_eq!(r.get("SMB"), Some(0));
    // Mail orchestration namespace (spear / stalwart / SharePoint).
    assert_eq!(r.get("EmailCorrespondance"), Some(4));
    // SharePoint content orchestration namespace (Sharepoint→smb-office-rs).
    assert_eq!(r.get("SharePoint"), Some(5));

    // Medical reserved range 10..=19, dense, alphabetical-stable.
    assert_eq!(r.get("Medical/RxNorm"), Some(11));
    assert_eq!(r.get("Medical/LOINC"), Some(12));
    assert_eq!(r.get("Medical/FMA"), Some(13));
    assert_eq!(r.get("Medical/RadLex"), Some(14));
    assert_eq!(r.get("Medical/SNOMED"), Some(15));
    assert_eq!(r.get("Medical/MONDO"), Some(16));
    assert_eq!(r.get("Medical/HPO"), Some(17));
    assert_eq!(r.get("Medical/DRON"), Some(18));
    assert_eq!(r.get("Medical/CHEBI"), Some(19));

    // 16 seed mappings total (6 cognitive + 10 medical).
    assert_eq!(r.len(), 16);
}

#[test]
fn namespace_registry_get_returns_none_for_unregistered() {
    let r = NamespaceRegistry::seed_defaults();
    assert_eq!(r.get("NotARealNamespace"), None);
    assert_eq!(r.get(""), None);
}

#[test]
fn namespace_registry_allocate_is_idempotent_and_dense() {
    let mut r = NamespaceRegistry::seed_defaults();
    // Allocate a new namespace; gets the first free id (6 — between
    // SharePoint=5 and Medical/ICD10CM=10).
    let id1 = r.allocate("CallCenter");
    assert_eq!(id1, 6);
    // Idempotent.
    assert_eq!(r.allocate("CallCenter"), 6);
    // Next free id continues densely (7).
    let id2 = r.allocate("Splat");
    assert_eq!(id2, 7);
    assert_ne!(id1, id2);
    assert_eq!(r.len(), 18);
}

#[test]
fn namespace_registry_can_round_trip_through_schema_ptr() {
    let r = NamespaceRegistry::seed_defaults();
    let healthcare = r.get("Healthcare").expect("Healthcare seeded as 2");
    let icd10 = r
        .get("Medical/ICD10CM")
        .expect("Medical/ICD10CM seeded as 10");

    // Build a SchemaPtr in the Healthcare context and an attribute SchemaPtr
    // in the ICD10CM context — they MUST be distinguishable by context.
    let hc_ptr = SchemaPtr::new(NamespaceId(2), 42, SchemaKind::Entity).with_context_id(healthcare);
    let icd_ptr =
        SchemaPtr::new(NamespaceId(2), 42, SchemaKind::Entity).with_context_id(icd10);

    // Same packed bits, different ontology_context_id => different SchemaPtr.
    assert_eq!(hc_ptr.raw(), icd_ptr.raw());
    assert_ne!(hc_ptr, icd_ptr);
    assert_eq!(hc_ptr.ontology_context_id(), 2);
    assert_eq!(icd_ptr.ontology_context_id(), 10);
}
