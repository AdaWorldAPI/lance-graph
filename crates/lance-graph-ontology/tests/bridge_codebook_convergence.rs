// Skip under Miri — TTL fixtures use `tempfile::tempdir()` +
// `std::fs::create_dir_all/write`, both blocked by Miri's isolation.
#![cfg(not(miri))]

//! Bridge codebook convergence test — codex P1 on PR #559.
//!
//! Pins the headline contract C4 + C5 promised: `OpenProjectBridge` and
//! `RedmineBridge`, when handed their respective port-public names for
//! the SAME canonical concept, return EntityRefs whose `entity_type_id()`
//! is identical (the OGAR codebook class_id, e.g. `0x0102` for
//! `project_work_item`).
//!
//! Before this PR shipped, the bridges minted distinct entity_type_ids
//! (the registry's URI-keyed allocation gave `ogit.Redmine:Issue` and
//! `ogit.OpenProject:WorkPackage` different ids), defeating the
//! convergence promise. The fix routed `entity()` through the
//! per-port codebook tables (`OPENPROJECT_CODEBOOK`,
//! `REDMINE_CODEBOOK`) keyed off the shared
//! `lance_graph_ontology::bridges::codebook::*` constants.

use lance_graph_ontology::bridges::{codebook, OpenProjectBridge, RedmineBridge};
use lance_graph_ontology::{NamespaceBridge, OntologyRegistry};
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
fn work_package_and_issue_resolve_to_same_class_id() {
    // The headline pin: WorkPackage (OpenProject) and Issue (Redmine)
    // both map to project_work_item (0x0102) per the OGAR codebook.
    let registry = make_registry();
    let op = OpenProjectBridge::new(Arc::clone(&registry)).unwrap();
    let rm = RedmineBridge::new(registry).unwrap();

    let wp = op.entity("WorkPackage").unwrap();
    let issue = rm.entity("Issue").unwrap();

    assert_eq!(
        wp.schema_ptr.entity_type_id(),
        issue.schema_ptr.entity_type_id(),
        "WorkPackage and Issue must converge on the same entity_type_id",
    );
    assert_eq!(
        wp.schema_ptr.entity_type_id(),
        codebook::PROJECT_WORK_ITEM,
        "shared class_id is 0x{:04X} (project_work_item)",
        codebook::PROJECT_WORK_ITEM,
    );
}

#[test]
fn project_and_time_entry_converge_on_canonical_class_ids() {
    // Project and TimeEntry use identical public names in both ports;
    // both routes through their respective codebooks land on the same
    // canonical class_id.
    let registry = make_registry();
    let op = OpenProjectBridge::new(Arc::clone(&registry)).unwrap();
    let rm = RedmineBridge::new(registry).unwrap();

    assert_eq!(
        op.entity("Project").unwrap().schema_ptr.entity_type_id(),
        rm.entity("Project").unwrap().schema_ptr.entity_type_id(),
    );
    assert_eq!(
        op.entity("Project").unwrap().schema_ptr.entity_type_id(),
        codebook::PROJECT,
    );

    assert_eq!(
        op.entity("TimeEntry").unwrap().schema_ptr.entity_type_id(),
        rm.entity("TimeEntry").unwrap().schema_ptr.entity_type_id(),
    );
    assert_eq!(
        op.entity("TimeEntry").unwrap().schema_ptr.entity_type_id(),
        codebook::BILLABLE_WORK_ENTRY,
    );
}

#[test]
fn status_and_type_alias_pairs_converge_through_each_ports_naming() {
    // Each port has port-specific public names for status / type:
    //   OpenProject: Status / Type
    //   Redmine:     IssueStatus / Tracker
    // Both pairs converge on the canonical class_ids.
    let registry = make_registry();
    let op = OpenProjectBridge::new(Arc::clone(&registry)).unwrap();
    let rm = RedmineBridge::new(registry).unwrap();

    assert_eq!(
        op.entity("Status").unwrap().schema_ptr.entity_type_id(),
        rm.entity("IssueStatus").unwrap().schema_ptr.entity_type_id(),
    );
    assert_eq!(
        op.entity("Status").unwrap().schema_ptr.entity_type_id(),
        codebook::PROJECT_STATUS,
    );

    assert_eq!(
        op.entity("Type").unwrap().schema_ptr.entity_type_id(),
        rm.entity("Tracker").unwrap().schema_ptr.entity_type_id(),
    );
    assert_eq!(
        op.entity("Type").unwrap().schema_ptr.entity_type_id(),
        codebook::PROJECT_TYPE,
    );
}

#[test]
fn synthesised_entity_refs_keep_distinct_namespace_ids() {
    // Convergence runs at the entity_type_id layer; namespace_id stays
    // port-specific (g_lock per bridge). This is the "different
    // namespace, same canonical class" architecture — anti-pattern
    // would be collapsing namespaces.
    let registry = make_registry();
    let op = OpenProjectBridge::new(Arc::clone(&registry)).unwrap();
    let rm = RedmineBridge::new(registry).unwrap();

    let wp = op.entity("WorkPackage").unwrap();
    let issue = rm.entity("Issue").unwrap();

    assert_ne!(
        wp.schema_ptr.namespace_id(),
        issue.schema_ptr.namespace_id(),
        "namespace_id must stay port-specific even when class_id converges",
    );
    assert_eq!(wp.schema_ptr.namespace_id(), op.g_lock());
    assert_eq!(issue.schema_ptr.namespace_id(), rm.g_lock());
}

#[test]
fn synthesised_entity_refs_stamp_seeded_context_ids() {
    // Codex P2 on PR #558: both bridges stamp the seeded ctx_id, NOT
    // the default `0`. OpenProject=6, Redmine=7 per
    // NamespaceRegistry::seed_defaults().
    let registry = make_registry();
    let op = OpenProjectBridge::new(Arc::clone(&registry)).unwrap();
    let rm = RedmineBridge::new(registry).unwrap();

    let wp = op.entity("WorkPackage").unwrap();
    let issue = rm.entity("Issue").unwrap();

    assert_eq!(wp.schema_ptr.ontology_context_id(), 6);
    assert_eq!(issue.schema_ptr.ontology_context_id(), 7);
    // Both are NON-zero — that's the test for "we're not falling back
    // to the default context".
    assert_ne!(wp.schema_ptr.ontology_context_id(), 0);
    assert_ne!(issue.schema_ptr.ontology_context_id(), 0);
}

#[test]
fn entity_by_uri_routes_through_codebook_for_canonical_concepts() {
    // Codex P2 on PR #562: URI resolution must converge on the
    // SAME entity_type_id as public-name resolution. Without this,
    // `bridge.entity_by_uri("ogit.OpenProject:WorkPackage")` would
    // return the registry's URI-keyed (= append-order) id, while
    // `bridge.entity("WorkPackage")` returns the codebook id —
    // splitting the convergence contract by lookup path.
    use lance_graph_ontology::OgitUri;
    let registry = make_registry();
    let op = OpenProjectBridge::new(Arc::clone(&registry)).unwrap();
    let rm = RedmineBridge::new(registry).unwrap();

    let wp_uri = OgitUri::parse("ogit.OpenProject:WorkPackage").unwrap();
    let issue_uri = OgitUri::parse("ogit.Redmine:Issue").unwrap();

    let wp_by_uri = op.entity_by_uri(&wp_uri).unwrap();
    let wp_by_name = op.entity("WorkPackage").unwrap();
    let issue_by_uri = rm.entity_by_uri(&issue_uri).unwrap();
    let issue_by_name = rm.entity("Issue").unwrap();

    // Same lookup path -> same EntityRef.
    assert_eq!(wp_by_uri.schema_ptr.raw(), wp_by_name.schema_ptr.raw());
    assert_eq!(issue_by_uri.schema_ptr.raw(), issue_by_name.schema_ptr.raw());

    // Across-port convergence on the codebook class_id holds for both
    // lookup paths.
    assert_eq!(
        wp_by_uri.schema_ptr.entity_type_id(),
        codebook::PROJECT_WORK_ITEM,
    );
    assert_eq!(
        issue_by_uri.schema_ptr.entity_type_id(),
        codebook::PROJECT_WORK_ITEM,
    );
    assert_eq!(
        wp_by_uri.schema_ptr.entity_type_id(),
        issue_by_uri.schema_ptr.entity_type_id(),
    );

    // Synthesized ctx_id stamps the seeded namespace id, same as the
    // public-name path.
    assert_eq!(wp_by_uri.schema_ptr.ontology_context_id(), 6);
    assert_eq!(issue_by_uri.schema_ptr.ontology_context_id(), 7);
}

#[test]
fn entity_by_uri_falls_through_for_non_codebook_uris() {
    // A URI in the bridge's namespace but NOT in the codebook reaches
    // the default registry path. The fixture hydrates only WorkPackage
    // / Issue, so an unregistered URI returns NotInScope.
    use lance_graph_ontology::OgitUri;
    let registry = make_registry();
    let op = OpenProjectBridge::new(registry).unwrap();
    let nonsense = OgitUri::parse("ogit.OpenProject:NotAConcept").unwrap();
    let result = op.entity_by_uri(&nonsense);
    assert!(result.is_err(), "non-codebook URI must not synthesize: {result:?}");
}
