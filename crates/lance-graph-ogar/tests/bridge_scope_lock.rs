// Skip under Miri — the whole file uses `tempfile::tempdir()` +
// `std::fs::create_dir_all/write` to stage TTL fixtures on disk, and Miri's
// isolation blocks `mkdir`/`open`. Stable and nightly without Miri run it
// normally.
#![cfg(not(miri))]

//! Bridge scope-lock test.
//!
//! Verifies that a `WoaBridge` cannot resolve a `Healthcare` entity, and
//! vice versa. The error returned must be `BridgeError::CrossNamespaceLeak`
//! or `BridgeError::NotInScope` (the latter when the namespace itself is
//! present but the public name was filed under a different bridge id).

use lance_graph_ogar::bridges::MedcareBridge;
use lance_graph_ontology::bridges::{OgitBridge, WoaBridge};
use lance_graph_ontology::{NamespaceBridge, OgitUri, OntologyRegistry};
use std::fs;
use std::sync::Arc;

const TTL: &str = r#"
@prefix ogit:                <http://www.purl.org/ogit/> .
@prefix ogit.WorkOrder:      <http://www.purl.org/ogit/WorkOrder/> .
@prefix ogit.Healthcare:     <http://www.purl.org/ogit/Healthcare/> .
@prefix rdfs:                <http://www.w3.org/2000/01/rdf-schema#> .

ogit.WorkOrder:Order
    a rdfs:Class;
    rdfs:subClassOf ogit:Entity;
    rdfs:label "Order";
    ogit:scope "NTO";
    ogit:parent ogit:Node;
    ogit:mandatory-attributes (
        ogit:id
    );
    ogit:optional-attributes ( ) ;
.

ogit.Healthcare:Patient
    a rdfs:Class;
    rdfs:subClassOf ogit:Entity;
    rdfs:label "Patient";
    ogit:scope "NTO";
    ogit:parent ogit:Node;
    ogit:mandatory-attributes (
        ogit:id
    );
    ogit:optional-attributes ( ) ;
.
"#;

fn make_registry() -> Arc<OntologyRegistry> {
    let tmp = tempfile::tempdir().unwrap();
    fs::create_dir_all(tmp.path().join("WorkOrder")).unwrap();
    fs::create_dir_all(tmp.path().join("Healthcare")).unwrap();
    fs::write(tmp.path().join("WorkOrder").join("ents.ttl"), TTL).unwrap();
    fs::write(tmp.path().join("Healthcare").join("ents.ttl"), TTL).unwrap();
    let registry = Arc::new(OntologyRegistry::new_in_memory());
    registry
        .hydrate_once_sync(tmp.path(), &["WorkOrder", "Healthcare"])
        .unwrap();
    // Keep the tempdir alive for the duration of the test by leaking; the
    // test process exits shortly after.
    std::mem::forget(tmp);
    registry
}

#[test]
fn woa_bridge_resolves_workorder_entity_by_uri() {
    let registry = make_registry();
    let bridge = WoaBridge::new(registry).unwrap();
    let uri = OgitUri::parse("ogit.WorkOrder:Order").unwrap();
    let entity = bridge.entity_by_uri(&uri).expect("scoped URI resolution");
    assert_eq!(entity.schema_ptr.namespace_id(), bridge.g_lock());
}

#[test]
fn woa_bridge_rejects_healthcare_entity_by_uri() {
    let registry = make_registry();
    let bridge = WoaBridge::new(registry).unwrap();
    let uri = OgitUri::parse("ogit.Healthcare:Patient").unwrap();
    let result = bridge.entity_by_uri(&uri);
    assert!(
        result.is_err(),
        "expected scope lock to refuse cross-namespace, got {result:?}",
    );
    let err = result.unwrap_err();
    let msg = format!("{err:?}");
    assert!(
        msg.contains("CrossNamespaceLeak") || msg.contains("NotInScope"),
        "expected CrossNamespaceLeak or NotInScope, got {msg}",
    );
}

#[test]
fn medcare_bridge_resolves_healthcare_entity_by_uri() {
    let registry = make_registry();
    let bridge = MedcareBridge::new(registry).unwrap();
    let uri = OgitUri::parse("ogit.Healthcare:Patient").unwrap();
    let entity = bridge.entity_by_uri(&uri).expect("scoped URI resolution");
    assert_eq!(entity.schema_ptr.namespace_id(), bridge.g_lock());
}

#[test]
fn medcare_bridge_rejects_workorder_entity_by_uri() {
    let registry = make_registry();
    let bridge = MedcareBridge::new(registry).unwrap();
    let uri = OgitUri::parse("ogit.WorkOrder:Order").unwrap();
    let result = bridge.entity_by_uri(&uri);
    assert!(
        result.is_err(),
        "expected scope lock to refuse, got {result:?}"
    );
}

#[test]
fn woa_bridge_public_name_aliases_via_append() {
    use lance_graph_contract::property::{Marking, Schema};
    use lance_graph_ontology::{MappingProposal, MappingProposalKind};
    let registry = make_registry();

    // A tenant adds a public-name alias for its locked namespace's
    // canonical URI by appending one mapping under its own bridge_id.
    let _ = registry.append_mapping(MappingProposal {
        public_name: "WorkOrder".to_string(),
        bridge_id: "woa".to_string(),
        ogit_uri: OgitUri::parse("ogit.WorkOrder:Order").unwrap(),
        namespace: "WorkOrder".to_string(),
        kind: MappingProposalKind::Entity {
            schema: Schema::builder("Order").required("id").build(),
        },
        marking: Marking::Internal,
        confidence: 1.0,
        source_uri: "test://woa-alias".to_string(),
        checksum: "alias-checksum".to_string(),
        created_by: "test".to_string(),
    });

    let bridge = WoaBridge::new(registry).unwrap();
    let entity = bridge.entity("WorkOrder").expect("public name resolves");
    assert_eq!(entity.schema_ptr.namespace_id(), bridge.g_lock());
}

#[test]
fn ogit_bridge_per_namespace_works() {
    let registry = make_registry();
    let work_order_bridge = OgitBridge::for_namespace(registry.clone(), "WorkOrder").unwrap();
    let healthcare_bridge = OgitBridge::for_namespace(registry, "Healthcare").unwrap();
    assert_ne!(work_order_bridge.g_lock(), healthcare_bridge.g_lock());
    let _ = work_order_bridge
        .entity_by_uri(&lance_graph_ontology::OgitUri::parse("ogit.WorkOrder:Order").unwrap())
        .expect("URI-based resolution within the same namespace");
}
