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

// -------------------------------------------------------------------------
// Probe TTL-PROBE-1: Malformed TTL must produce HydrationFailure (no panic).
// -------------------------------------------------------------------------
//
// The OGIT NTO tree is human-edited; truncated edits and missing prefixes are
// realistic failure modes. The contract from `parse_into_proposals` is that
// these surface as `HydrationFailure { source, reason }` on the
// `HydrationReport`, never as a panic that aborts the whole crawl.
const MALFORMED_TTL: &str = r#"
ogit.Bad:Thing
    a rdfs:Class ;
    rdfs:subClassOf ogit:Entity ;
    ogit:mandatory-attributes ( ogit:id
.
"#;

#[test]
fn malformed_ttl_yields_hydration_failure() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let ns_dir = tmp.path().join("Bad").join("entities");
    fs::create_dir_all(&ns_dir).unwrap();
    fs::write(ns_dir.join("Thing.ttl"), MALFORMED_TTL).unwrap();

    let registry = Arc::new(OntologyRegistry::new_in_memory());
    let report = registry
        .hydrate_once_sync(tmp.path(), &["Bad"])
        .expect("hydrate_once_sync must not return Err for parser-level failures");
    assert!(
        !report.is_clean(),
        "expected at least one HydrationFailure, got clean report"
    );
    assert!(
        report
            .failures
            .iter()
            .any(|f| f.source.contains("Thing.ttl")),
        "malformed file must appear in failures: {:?}",
        report.failures
    );
}

// -------------------------------------------------------------------------
// Probe TTL-PROBE-2: Entity with empty mandatory-attributes ( ) registers.
// -------------------------------------------------------------------------
//
// An OGIT entity may legitimately declare no mandatory attributes (e.g. an
// abstract base class, or a stub awaiting refinement). Empty `()` lists must
// not block registration; the entity should land in the registry with an
// empty Schema.
const EMPTY_MANDATORY_TTL: &str = r#"
@prefix ogit:                   <http://www.purl.org/ogit/> .
@prefix ogit.Empty:             <http://www.purl.org/ogit/Empty/> .
@prefix rdfs:                   <http://www.w3.org/2000/01/rdf-schema#> .

ogit.Empty:Stub
    a rdfs:Class;
    rdfs:subClassOf ogit:Entity;
    rdfs:label "Stub";
    ogit:scope "NTO";
    ogit:parent ogit:Node;
    ogit:mandatory-attributes ( ) ;
    ogit:optional-attributes ( ) ;
.
"#;

#[test]
fn entity_with_empty_attribute_lists_registers() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let ns_dir = tmp.path().join("Empty").join("entities");
    fs::create_dir_all(&ns_dir).unwrap();
    fs::write(ns_dir.join("Stub.ttl"), EMPTY_MANDATORY_TTL).unwrap();

    let registry = Arc::new(OntologyRegistry::new_in_memory());
    let report = registry
        .hydrate_once_sync(tmp.path(), &["Empty"])
        .expect("hydration");
    assert!(report.is_clean(), "failures: {:?}", report.failures);
    assert!(
        report.registered >= 1,
        "empty-attr entity must register: {report:?}"
    );
    let stub = registry
        .resolve_uri("ogit.Empty:Stub")
        .expect("stub resolves by URI");
    assert!(
        stub.namespace_id().is_known(),
        "namespace G must be assigned"
    );
    assert_eq!(
        registry.namespace_id("Empty"),
        Some(stub.namespace_id()),
        "registry must map 'Empty' to the same NamespaceId"
    );
}

// -------------------------------------------------------------------------
// Probe TTL-PROBE-3: Multiple entities in one TTL file.
// -------------------------------------------------------------------------
//
// Real OGIT (e.g. ogit.ttl at the root of the NTO repo) declares many
// classes in a single file. Verify our parser indexes by subject and emits
// one proposal per entity, not one per file.
const MULTI_ENTITY_TTL: &str = r#"
@prefix ogit:                   <http://www.purl.org/ogit/> .
@prefix ogit.Multi:             <http://www.purl.org/ogit/Multi/> .
@prefix rdfs:                   <http://www.w3.org/2000/01/rdf-schema#> .

ogit.Multi:Alpha
    a rdfs:Class;
    rdfs:subClassOf ogit:Entity;
    rdfs:label "Alpha";
    ogit:scope "NTO";
    ogit:parent ogit:Node;
    ogit:mandatory-attributes ( ogit:id ) ;
    ogit:optional-attributes ( ) ;
.

ogit.Multi:Beta
    a rdfs:Class;
    rdfs:subClassOf ogit:Entity;
    rdfs:label "Beta";
    ogit:scope "NTO";
    ogit:parent ogit:Node;
    ogit:mandatory-attributes ( ogit:id ) ;
    ogit:optional-attributes ( ogit:name ) ;
.

ogit.Multi:Gamma
    a rdfs:Class;
    rdfs:subClassOf ogit:Entity;
    rdfs:label "Gamma";
    ogit:scope "NTO";
    ogit:parent ogit:Node;
    ogit:mandatory-attributes ( ogit:id ) ;
    ogit:optional-attributes ( ) ;
.
"#;

#[test]
fn multi_entity_ttl_emits_one_proposal_per_subject() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let ns_dir = tmp.path().join("Multi").join("entities");
    fs::create_dir_all(&ns_dir).unwrap();
    fs::write(ns_dir.join("all.ttl"), MULTI_ENTITY_TTL).unwrap();

    let registry = Arc::new(OntologyRegistry::new_in_memory());
    let report = registry
        .hydrate_once_sync(tmp.path(), &["Multi"])
        .expect("hydration");
    assert!(report.is_clean(), "failures: {:?}", report.failures);
    assert!(
        report.registered >= 3,
        "all three entities from one TTL file must register: {report:?}"
    );
    for name in ["ogit.Multi:Alpha", "ogit.Multi:Beta", "ogit.Multi:Gamma"] {
        registry
            .resolve_uri(name)
            .unwrap_or_else(|| panic!("{name} must resolve"));
    }
}

// -------------------------------------------------------------------------
// Probe TTL-PROBE-4: TTL with @base declaration parses correctly.
// -------------------------------------------------------------------------
//
// The Turtle spec allows `@base <iri>` to set the base for relative IRIs.
// Our parser hard-codes a base of `http://www.purl.org/ogit/`; verify a
// document that declares an explicit @base matching the OGIT root still
// produces correct proposals.
const BASE_DECL_TTL: &str = r#"
@base <http://www.purl.org/ogit/> .
@prefix ogit:                   <http://www.purl.org/ogit/> .
@prefix ogit.BaseDecl:          <http://www.purl.org/ogit/BaseDecl/> .
@prefix rdfs:                   <http://www.w3.org/2000/01/rdf-schema#> .

ogit.BaseDecl:Item
    a rdfs:Class;
    rdfs:subClassOf ogit:Entity;
    rdfs:label "Item";
    ogit:scope "NTO";
    ogit:parent ogit:Node;
    ogit:mandatory-attributes ( ogit:id ) ;
    ogit:optional-attributes ( ) ;
.
"#;

#[test]
fn base_declaration_does_not_break_parser() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let ns_dir = tmp.path().join("BaseDecl").join("entities");
    fs::create_dir_all(&ns_dir).unwrap();
    fs::write(ns_dir.join("Item.ttl"), BASE_DECL_TTL).unwrap();

    let registry = Arc::new(OntologyRegistry::new_in_memory());
    let report = registry
        .hydrate_once_sync(tmp.path(), &["BaseDecl"])
        .expect("hydration");
    assert!(report.is_clean(), "failures: {:?}", report.failures);
    assert!(report.registered >= 1, "report: {report:?}");
    let item = registry
        .resolve_uri("ogit.BaseDecl:Item")
        .expect("item resolves by URI");
    assert!(
        item.namespace_id().is_known(),
        "namespace G must be assigned"
    );
    assert_eq!(
        registry.namespace_id("BaseDecl"),
        Some(item.namespace_id()),
        "registry must map 'BaseDecl' to the same NamespaceId"
    );
}

// -------------------------------------------------------------------------
// Probe TTL-PROBE-5: dcterms:source annotation should round-trip.
// -------------------------------------------------------------------------
//
// The TTL spec carries an optional `dcterms:source` per entity (provenance
// pointer to the upstream definition). Today the dictionary `source_uri`
// column is set to the local `file://...` path. If a TTL declares its own
// `dcterms:source`, that value is currently dropped — see TECH_DEBT entry.
// This probe documents the gap by asserting that it is in fact dropped.
const DCTERMS_SOURCE_TTL: &str = r#"
@prefix ogit:                   <http://www.purl.org/ogit/> .
@prefix ogit.Provenance:        <http://www.purl.org/ogit/Provenance/> .
@prefix rdfs:                   <http://www.w3.org/2000/01/rdf-schema#> .
@prefix dcterms:                <http://purl.org/dc/terms/> .

ogit.Provenance:Tracked
    a rdfs:Class;
    rdfs:subClassOf ogit:Entity;
    rdfs:label "Tracked";
    dcterms:source <https://github.com/arago/OGiT/blob/master/NTO/Provenance/entities/Tracked.ttl> ;
    ogit:scope "NTO";
    ogit:parent ogit:Node;
    ogit:mandatory-attributes ( ogit:id ) ;
    ogit:optional-attributes ( ) ;
.
"#;

#[test]
fn dcterms_source_is_currently_dropped() {
    use lance_graph_ontology::semantic_types::SemanticTypeMap;
    use lance_graph_ontology::ttl_parse::TtlSource;

    let path = std::path::PathBuf::from("dcterms_probe.ttl");
    let src = TtlSource::from_bytes(path.clone(), DCTERMS_SOURCE_TTL.as_bytes().to_vec());
    let sem = SemanticTypeMap::defaults();
    let proposals = src
        .parse_into_proposals("ogit", sem)
        .expect("dcterms TTL must parse");
    let entity = proposals
        .iter()
        .find(|p| p.public_name == "ogit.Provenance:Tracked")
        .expect("entity must register");
    // CURRENT behaviour: source_uri is the local file path, not the
    // dcterms:source IRI from the TTL. This assertion locks the gap so a
    // future fix flips this test (and TTL-PROBE-5 in TECH_DEBT.md closes).
    assert!(
        entity.source_uri.starts_with("file:"),
        "expected file:-prefixed source_uri, got {:?}",
        entity.source_uri
    );
    assert!(
        !entity.source_uri.contains("github.com"),
        "BUG: dcterms:source already round-trips, update test and close TTL-PROBE-5"
    );
}
