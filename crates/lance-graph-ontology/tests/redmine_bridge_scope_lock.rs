// Skip under Miri — TTL fixtures use `tempfile::tempdir()` +
// `std::fs::create_dir_all/write`, both blocked by Miri's isolation.
#![cfg(not(miri))]

//! `RedmineBridge` scope-lock test — Northstar plan §3 C5.
//!
//! Sibling of `openproject_bridge_scope_lock.rs` (C4). Verifies the
//! scope-lock contract for the Redmine ↔ OpenProject pair so the
//! cross-fork convergence (`fork_convergence.json` schema
//! `fork-convergence/2`) lands on a substrate that refuses
//! cross-namespace leaks by default.

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
    ogit:mandatory-attributes (
        ogit:id
    );
    ogit:optional-attributes ( ) ;
.

ogit.Redmine:Issue
    a rdfs:Class;
    rdfs:subClassOf ogit:Entity;
    rdfs:label "Issue";
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
    fs::create_dir_all(tmp.path().join("OpenProject")).unwrap();
    fs::create_dir_all(tmp.path().join("Redmine")).unwrap();
    fs::write(tmp.path().join("OpenProject").join("ents.ttl"), TTL).unwrap();
    fs::write(tmp.path().join("Redmine").join("ents.ttl"), TTL).unwrap();
    let registry = Arc::new(OntologyRegistry::new_in_memory());
    registry
        .hydrate_once_sync(tmp.path(), &["OpenProject", "Redmine"])
        .unwrap();
    // Keep the tempdir alive for the duration of the test by leaking; the
    // test process exits shortly after.
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
    // Together with the previous test, the cross-namespace lock is
    // bidirectional across the two ports.
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
fn redmine_bridge_id_is_lowercase_redmine() {
    let registry = make_registry();
    let bridge = RedmineBridge::new(registry).unwrap();
    assert_eq!(bridge.bridge_id(), "redmine");
}

#[test]
fn redmine_bridge_g_lock_matches_redmine_namespace_id() {
    let registry = make_registry();
    let expected = registry.namespace_id("Redmine").unwrap();
    let bridge = RedmineBridge::new(registry).unwrap();
    assert_eq!(bridge.g_lock(), expected);
}

#[test]
fn redmine_bridge_construction_fails_when_namespace_missing() {
    // No hydration -> no Redmine namespace -> constructor returns
    // Error::UnknownNamespace.
    let registry = Arc::new(OntologyRegistry::new_in_memory());
    let result = RedmineBridge::new(registry);
    assert!(
        result.is_err(),
        "expected UnknownNamespace error, got {result:?}",
    );
}

#[test]
fn openproject_and_redmine_bridges_have_distinct_g_locks() {
    // Two ports, two namespaces, two distinct g_lock NamespaceIds.
    // The convergence pin operates at the codebook id layer (above the
    // namespace), not by collapsing namespaces.
    let registry = make_registry();
    let op_bridge = OpenProjectBridge::new(Arc::clone(&registry)).unwrap();
    let rm_bridge = RedmineBridge::new(registry).unwrap();
    assert_ne!(
        op_bridge.g_lock(),
        rm_bridge.g_lock(),
        "OpenProject and Redmine must lock to different namespace ids"
    );
}
