//! Hydrate against the actual `AdaWorldAPI/OGIT` fork.
//!
//! Runs only when `OGIT_FORK_PATH` is set to a directory containing the
//! cloned fork. The CI matrix sets this to `/home/user/OGIT` (or the
//! workspace-relative `../OGIT`).
//!
//! Verifies that hydration of the `Network` namespace produces the
//! expected canonical entities (`IPAddress`, `MACAddress`, `VLAN`,
//! `Switch`, etc.), that resolution is fast, and that idempotent
//! re-hydration short-circuits.

use lance_graph_ontology::{NamespaceBridge, OntologyRegistry};
use std::path::Path;
use std::sync::Arc;

fn ogit_root() -> Option<std::path::PathBuf> {
    if let Ok(p) = std::env::var("OGIT_FORK_PATH") {
        let p = std::path::PathBuf::from(p);
        if p.exists() {
            return Some(p.join("NTO"));
        }
    }
    // Convention: AdaWorldAPI/OGIT is checked out next to lance-graph.
    let candidate = Path::new("/home/user/OGIT/NTO");
    if candidate.exists() {
        return Some(candidate.to_path_buf());
    }
    let from_workspace = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../../OGIT/NTO");
    if from_workspace.exists() {
        return Some(from_workspace);
    }
    None
}

#[test]
fn hydrate_network_namespace_from_real_ogit() {
    let Some(root) = ogit_root() else {
        eprintln!("SKIP: OGIT fork not found (set OGIT_FORK_PATH)");
        return;
    };
    let registry = Arc::new(OntologyRegistry::new_in_memory());
    let report = registry
        .hydrate_once_sync(&root, &["Network"])
        .expect("hydration");
    assert!(
        report.registered > 0,
        "expected at least one Network entity, report = {report:?}"
    );
    let ip = registry.resolve_uri("ogit.Network:IPAddress");
    assert!(ip.is_some(), "ogit.Network:IPAddress should be present");

    // Ogit-bridge for the Network namespace should resolve URIs.
    let bridge = lance_graph_ontology::bridges::OgitBridge::for_namespace(
        registry.clone(),
        "Network",
    )
    .unwrap();
    let entity = bridge
        .entity_by_uri(&lance_graph_ontology::OgitUri::parse("ogit.Network:IPAddress").unwrap())
        .expect("network bridge resolves IPAddress");
    assert_eq!(entity.schema_ptr.namespace_id(), bridge.g_lock());
}

/// Hydrate the WorkOrder namespace from the OGIT fork (the namespace
/// added in this session via `AdaWorldAPI/WoA` transcode). Closes the
/// "Phase 7 has no executed proof" gap surfaced by the integration-lead
/// review: this test exercises the full TTL → registry path against the
/// 15 entities + 12 verbs that landed in commit `3871d37` on the OGIT
/// fork branch `claude/create-graph-ontology-crate-gkuJG`.
#[test]
fn hydrate_workorder_namespace_from_real_ogit() {
    let Some(root) = ogit_root() else {
        eprintln!("SKIP: OGIT fork not found (set OGIT_FORK_PATH)");
        return;
    };
    if !root.join("WorkOrder").exists() {
        eprintln!(
            "SKIP: OGIT fork has no WorkOrder/ namespace at {}",
            root.display()
        );
        return;
    }
    let registry = Arc::new(OntologyRegistry::new_in_memory());
    let report = registry
        .hydrate_once_sync(&root, &["WorkOrder"])
        .expect("hydration");
    assert!(
        report.is_clean(),
        "WorkOrder hydration must report no failures: {:?}",
        report.failures
    );
    assert!(
        report.registered >= 15,
        "expected at least 15 WorkOrder entities + 12 verbs, got {report:?}"
    );

    // Spot-check the canonical entity URIs that map to WoA models.py.
    for entity in [
        "ogit.WorkOrder:Tenant",
        "ogit.WorkOrder:Customer",
        "ogit.WorkOrder:Order",
        "ogit.WorkOrder:Article",
        "ogit.WorkOrder:Position",
        "ogit.WorkOrder:Activity",
        "ogit.WorkOrder:Picture",
        "ogit.WorkOrder:HistoryEntry",
        "ogit.WorkOrder:User",
        "ogit.WorkOrder:LogbookEntry",
        "ogit.WorkOrder:NumberSequence",
        "ogit.WorkOrder:Setting",
        "ogit.WorkOrder:CustomerPortalUser",
        "ogit.WorkOrder:PasswordEntry",
        "ogit.WorkOrder:TimeSheet",
    ] {
        let ptr = registry
            .resolve_uri(entity)
            .unwrap_or_else(|| panic!("{entity} must resolve in WorkOrder namespace"));
        assert!(
            ptr.namespace_id().is_known(),
            "{entity} must carry an assigned NamespaceId"
        );
    }

    // The WoA bridge must lock all 15 to its single G partition.
    let bridge = lance_graph_ontology::bridges::WoaBridge::new(registry.clone())
        .expect("WoaBridge constructs once WorkOrder namespace is hydrated");
    let order_ptr = bridge
        .entity_by_uri(&lance_graph_ontology::OgitUri::parse("ogit.WorkOrder:Order").unwrap())
        .expect("Order resolves through the bridge scope-lock");
    assert_eq!(order_ptr.schema_ptr.namespace_id(), bridge.g_lock());
}

#[test]
fn idempotent_re_hydration_is_fast() {
    let Some(root) = ogit_root() else {
        eprintln!("SKIP: OGIT fork not found (set OGIT_FORK_PATH)");
        return;
    };
    let registry = Arc::new(OntologyRegistry::new_in_memory());
    let r1 = registry
        .hydrate_once_sync(&root, &["Network"])
        .expect("first hydration");
    let t = std::time::Instant::now();
    let r2 = registry
        .hydrate_once_sync(&root, &["Network"])
        .expect("second hydration");
    let elapsed = t.elapsed();
    assert!(r1.registered > 0);
    assert!(r2.from_cache, "second hydration must short-circuit");
    assert!(
        elapsed.as_millis() < 250,
        "idempotent re-hydration should be fast; got {elapsed:?}"
    );
}
