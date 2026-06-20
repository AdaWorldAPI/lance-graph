// Skip under Miri — TTL fixtures use `tempfile::tempdir()` +
// `std::fs::create_dir_all/write`, both blocked by Miri's isolation.
#![cfg(not(miri))]

//! `RedmineBridge` scope-lock test — Northstar plan §3 C5.
//!
//! Sibling of `openproject_bridge_scope_lock.rs` (C4). Verifies the
//! scope-lock contract for the Redmine ↔ OpenProject pair so the
//! cross-fork convergence (pinned in `bridge_codebook_convergence.rs`)
//! lands on a substrate that refuses cross-namespace leaks by default.

use lance_graph_ontology::bridges::{OpenProjectBridge, RedmineBridge};
use lance_graph_ontology::{NamespaceBridge, OgitUri, OntologyRegistry};
use std::fs;
use std::sync::Arc;

const TTL: &str = r#"
@prefix ogit:                <http://www.purl.org/ogit/> .
@prefix ogit.OpenProject:    <http://www.purl.org/ogit/OpenProject/> .
@prefix ogit.Redmine:        <http://www.purl.org/ogit/Redmine/> .
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

ogit.Redmine:Issue
    a rdfs:Class;
    rdfs:subClassOf ogit:Entity;
    rdfs:label "Issue";
    ogit:scope "NTO";
    ogit:parent ogit:Node;
    ogit:mandatory-attributes ( ogit:id );
    ogit:optional-attributes ( ) ;
.
"#;

fn make_registry() -> Arc<OntologyRegistry> {
    let tmp = tempfile::tempdir().unwrap();
    fs::create_dir_all(tmp.path().join("OpenProject")).unwrap();
    fs::create_dir_all(tmp.path().join("Redmine")).unwrap();
    fs::write(tmp.path().join("OpenProject").join("ents.ttl"), TTL).unwrap();
    fs::write(tmp.path().join("Redmine").join("ents.ttl"), TTL).unwrap();
    let registry = Arc::new(OntologyRegistry::new_in_memory());
    registry
        .hydrate_once_sync(tmp.path(), &["OpenProject", "Redmine"])
        .unwrap();
    std::mem::forget(tmp);
    registry
}

#[test]
fn redmine_bridge_resolves_redmine_entity_by_uri() {
    let registry = make_registry();
    let bridge = RedmineBridge::new(registry).unwrap();
    let uri = OgitUri::parse("ogit.Redmine:Issue").unwrap();
    let entity = bridge.entity_by_uri(&uri).expect("scoped URI resolution");
    assert_eq!(entity.schema_ptr.namespace_id(), bridge.g_lock());
}

#[test]
fn redmine_bridge_rejects_openproject_entity_by_uri() {
    let registry = make_registry();
    let bridge = RedmineBridge::new(registry).unwrap();
    let uri = OgitUri::parse("ogit.OpenProject:WorkPackage").unwrap();
    let result = bridge.entity_by_uri(&uri);
    assert!(
        result.is_err(),
        "expected scope lock to refuse OpenProject URI, got {result:?}",
    );
    let err = result.unwrap_err();
    let msg = format!("{err:?}");
    assert!(
        msg.contains("CrossNamespaceLeak") || msg.contains("NotInScope"),
        "expected CrossNamespaceLeak or NotInScope, got {msg}",
    );
}

#[test]
fn openproject_bridge_rejects_redmine_entity_by_uri() {
    // Symmetric pin: the OpenProject bridge refuses a Redmine entity.
    let registry = make_registry();
    let bridge = OpenProjectBridge::new(registry).unwrap();
    let uri = OgitUri::parse("ogit.Redmine:Issue").unwrap();
    let result = bridge.entity_by_uri(&uri);
    assert!(
        result.is_err(),
        "expected openproject scope lock to refuse Redmine URI, got {result:?}",
    );
    let err = result.unwrap_err();
    let msg = format!("{err:?}");
    assert!(
        msg.contains("CrossNamespaceLeak") || msg.contains("NotInScope"),
        "expected CrossNamespaceLeak or NotInScope, got {msg}",
    );
}

#[test]
fn openproject_and_redmine_bridges_have_distinct_g_locks() {
    // The convergence pin operates at the codebook class_id layer
    // (above the namespace), NOT by collapsing namespaces. Different
    // g_lock + same class_id is the architectural shape.
    let registry = make_registry();
    let op_bridge = OpenProjectBridge::new(Arc::clone(&registry)).unwrap();
    let rm_bridge = RedmineBridge::new(registry).unwrap();
    assert_ne!(
        op_bridge.g_lock(),
        rm_bridge.g_lock(),
        "OpenProject and Redmine must lock to different namespace ids"
    );
}
