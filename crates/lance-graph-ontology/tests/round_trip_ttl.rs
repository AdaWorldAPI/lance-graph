//! End-to-end TTL hydration test.
//!
//! Builds a tiny TTL fixture, writes it to a tempdir, hydrates the
//! registry, and asserts that resolution by `(bridge_id, public_name)` and
//! by OGIT URI both work.

use lance_graph_ontology::{NamespaceBridge, OntologyRegistry};
use std::fs;
use std::sync::Arc;

const FIXTURE: &str = r#"
@prefix ogit:                   <http://www.purl.org/ogit/> .
@prefix ogit.RoundTrip:         <http://www.purl.org/ogit/RoundTrip/> .
@prefix rdfs:                   <http://www.w3.org/2000/01/rdf-schema#> .
@prefix dcterms:                <http://purl.org/dc/terms/> .

ogit.RoundTrip:Widget
    a rdfs:Class;
    rdfs:subClassOf ogit:Entity;
    rdfs:label "Widget";
    dcterms:description "A test entity." ;
    ogit:scope "NTO";
    ogit:parent ogit:Node;
    ogit:mandatory-attributes (
        ogit:id
    );
    ogit:optional-attributes (
        ogit:name
    );
.

ogit.RoundTrip:Sprocket
    a rdfs:Class;
    rdfs:subClassOf ogit:Entity;
    rdfs:label "Sprocket";
    ogit:scope "NTO";
    ogit:parent ogit:Node;
    ogit:mandatory-attributes (
        ogit:id
    );
    ogit:optional-attributes ( ) ;
.
"#;

#[test]
fn ttl_round_trip_in_memory() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let ns_dir = tmp.path().join("RoundTrip").join("entities");
    fs::create_dir_all(&ns_dir).unwrap();
    fs::write(ns_dir.join("Widget.ttl"), FIXTURE).unwrap();

    let registry = Arc::new(OntologyRegistry::new_in_memory());
    let report = registry
        .hydrate_once_sync(tmp.path(), &["RoundTrip"])
        .expect("hydration");
    assert!(report.is_clean(), "failures: {:?}", report.failures);
    assert!(report.registered >= 2, "report: {report:?}");

    let widget = registry
        .resolve("ogit", "ogit.RoundTrip:Widget")
        .expect("widget by bridge_id+public_name");
    let sprocket = registry
        .resolve_uri("ogit.RoundTrip:Sprocket")
        .expect("sprocket by URI");
    assert_ne!(widget, sprocket);
    assert_eq!(widget.namespace_id(), sprocket.namespace_id());
}

#[test]
fn ttl_round_trip_idempotent() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let ns_dir = tmp.path().join("RoundTrip").join("entities");
    fs::create_dir_all(&ns_dir).unwrap();
    fs::write(ns_dir.join("Widget.ttl"), FIXTURE).unwrap();

    let registry = Arc::new(OntologyRegistry::new_in_memory());
    let r1 = registry
        .hydrate_once_sync(tmp.path(), &["RoundTrip"])
        .expect("first hydration");
    let r2 = registry
        .hydrate_once_sync(tmp.path(), &["RoundTrip"])
        .expect("second hydration");
    assert!(r1.registered >= 2);
    assert!(r2.from_cache, "second hydration must short-circuit");
}

#[test]
fn export_ttl_writes_file() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let ns_dir = tmp.path().join("RoundTrip").join("entities");
    fs::create_dir_all(&ns_dir).unwrap();
    fs::write(ns_dir.join("Widget.ttl"), FIXTURE).unwrap();

    let registry = Arc::new(OntologyRegistry::new_in_memory());
    registry
        .hydrate_once_sync(tmp.path(), &["RoundTrip"])
        .unwrap();

    let out = tmp.path().join("export.ttl");
    registry.export_ttl("RoundTrip", &out).unwrap();
    let body = fs::read_to_string(&out).unwrap();
    assert!(body.contains("ogit.RoundTrip:Widget"));
    assert!(body.contains("ogit.RoundTrip:Sprocket"));
}

#[test]
fn ogit_bridge_for_namespace_locks() {
    use lance_graph_ontology::bridges::OgitBridge;
    let tmp = tempfile::tempdir().expect("tempdir");
    let ns_dir = tmp.path().join("RoundTrip").join("entities");
    fs::create_dir_all(&ns_dir).unwrap();
    fs::write(ns_dir.join("Widget.ttl"), FIXTURE).unwrap();

    let registry = Arc::new(OntologyRegistry::new_in_memory());
    registry
        .hydrate_once_sync(tmp.path(), &["RoundTrip"])
        .unwrap();

    let bridge = OgitBridge::for_namespace(registry.clone(), "RoundTrip").unwrap();
    let entity = bridge.entity("ogit.RoundTrip:Widget").unwrap();
    assert_eq!(entity.schema_ptr.namespace_id(), bridge.g_lock());
}
