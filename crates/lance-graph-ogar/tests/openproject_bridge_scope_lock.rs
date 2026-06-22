// Skip under Miri — TTL fixtures use `tempfile::tempdir()` +
// `std::fs::create_dir_all/write`, both blocked by Miri's isolation.
#![cfg(not(miri))]
// Exercises the deprecated `OpenProjectBridge` alias on purpose — see
// `docs/CONSUMER-BRIDGE-DEPRECATION.md`.
#![allow(deprecated)]

//! `OpenProjectBridge` scope-lock test — Northstar plan §3 C4.
//!
//! Verifies that a bridge locked to the `OpenProject` namespace
//! resolves an OpenProject entity by URI, and that the same bridge
//! refuses a `Healthcare` entity (cross-namespace leak refused).
//! Mirrors the contract pinned by `bridge_scope_lock.rs` for the
//! Woa/Medcare pair — symmetric coverage so the addition can't
//! silently relax the scope-lock guarantee.

use lance_graph_ogar::bridges::{MedcareBridge, OpenProjectBridge};
use lance_graph_ontology::{NamespaceBridge, OgitUri, OntologyRegistry};
use std::fs;
use std::sync::Arc;

const TTL: &str = r#"
@prefix ogit:                <http://www.purl.org/ogit/> .
@prefix ogit.OpenProject:    <http://www.purl.org/ogit/OpenProject/> .
@prefix ogit.Healthcare:     <http://www.purl.org/ogit/Healthcare/> .
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
    fs::create_dir_all(tmp.path().join("OpenProject")).unwrap();
    fs::create_dir_all(tmp.path().join("Healthcare")).unwrap();
    fs::write(tmp.path().join("OpenProject").join("ents.ttl"), TTL).unwrap();
    fs::write(tmp.path().join("Healthcare").join("ents.ttl"), TTL).unwrap();
    let registry = Arc::new(OntologyRegistry::new_in_memory());
    registry
        .hydrate_once_sync(tmp.path(), &["OpenProject", "Healthcare"])
        .unwrap();
    // Keep the tempdir alive for the duration of the test by leaking; the
    // test process exits shortly after.
    std::mem::forget(tmp);
    registry
}

#[test]
fn openproject_bridge_resolves_openproject_entity_by_uri() {
    let registry = make_registry();
    let bridge = OpenProjectBridge::new(registry).unwrap();
    let uri = OgitUri::parse("ogit.OpenProject:WorkPackage").unwrap();
    let entity = bridge.entity_by_uri(&uri).expect("scoped URI resolution");
    assert_eq!(entity.schema_ptr.namespace_id(), bridge.g_lock());
}

#[test]
fn openproject_bridge_rejects_healthcare_entity_by_uri() {
    let registry = make_registry();
    let bridge = OpenProjectBridge::new(registry).unwrap();
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
fn medcare_bridge_rejects_openproject_entity_by_uri() {
    // Symmetry pin: the Healthcare bridge equally refuses an OpenProject
    // entity. Together with the previous test, the cross-namespace lock
    // is bidirectional.
    let registry = make_registry();
    let bridge = MedcareBridge::new(registry).unwrap();
    let uri = OgitUri::parse("ogit.OpenProject:WorkPackage").unwrap();
    let result = bridge.entity_by_uri(&uri);
    assert!(
        result.is_err(),
        "expected medcare scope lock to refuse OpenProject URI, got {result:?}",
    );
    let err = result.unwrap_err();
    let msg = format!("{err:?}");
    assert!(
        msg.contains("CrossNamespaceLeak") || msg.contains("NotInScope"),
        "expected CrossNamespaceLeak or NotInScope, got {msg}",
    );
}

#[test]
fn openproject_bridge_id_is_lowercase_openproject() {
    let registry = make_registry();
    let bridge = OpenProjectBridge::new(registry).unwrap();
    assert_eq!(bridge.bridge_id(), "openproject");
}

#[test]
fn openproject_bridge_g_lock_matches_openproject_namespace_id() {
    let registry = make_registry();
    let expected = registry.namespace_id("OpenProject").unwrap();
    let bridge = OpenProjectBridge::new(registry).unwrap();
    assert_eq!(bridge.g_lock(), expected);
}

#[test]
fn openproject_bridge_construction_fails_when_namespace_missing() {
    // No hydration -> no OpenProject namespace -> constructor returns
    // Error::UnknownNamespace.
    let registry = Arc::new(OntologyRegistry::new_in_memory());
    let result = OpenProjectBridge::new(registry);
    // NB: `OpenProjectBridge` is `UnifiedBridge<OpenProjectPort>`, which
    // intentionally does not implement `Debug`, so we assert on `is_err()`
    // without formatting the `Ok` value (pre-#570 the bridge was a Debug
    // struct and this message interpolated `{result:?}`).
    assert!(
        result.is_err(),
        "expected UnknownNamespace error from constructing over an empty registry",
    );
}
