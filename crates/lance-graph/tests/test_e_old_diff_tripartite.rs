//! TEST E — can the OLD path (store B, `VersionedGraph`) produce the tripartite
//! change set `inserted / updated / removed`?
//!
//! Scenario: `V1 = {A, B, D}`, `V2 = {A, B', C}`. Desired:
//!
//! - `inserted = to - from = {C}`
//! - `updated  = seals differ on the intersection = {B}`
//! - `removed  = from - to = {D}`
//!
//! `GraphDiff` (`graph/versioned.rs:70`) carries `new_nodes` and
//! `modified_nodes` and NO removed-nodes field; `diff()` computes
//! `to_ids.difference(&from_ids)` (`:556`) and the seal-differing intersection
//! (`:560`) but never `from_ids.difference(&to_ids)` — even though both sets
//! are materialized side by side at `:553-554`.
//!
//! This probe asserts the two sets the API does expose, then DERIVES the third
//! from exactly the material `diff()` already reads (`extract_node_seals` over
//! both checked-out versions), showing the omission is API-only rather than an
//! information loss in the store.
//!
//! Modelled on `tests/lance_row_identity_probe.rs` (D-LNC-2) for the write
//! idioms; no `_rowaddr` here — this probe is about the change SETS.

use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

use arrow_array::builder::FixedSizeBinaryBuilder;
use arrow_array::{Array, FixedSizeBinaryArray, RecordBatch, UInt32Array};
use arrow_schema::{DataType, SchemaRef};
use futures::TryStreamExt;
use lance::dataset::Dataset;
use lance_graph::graph::blasgraph::columnar::{EdgeSchema, FingerprintSchema, NodeSchema};
use lance_graph::graph::versioned::{GraphSealStatus, VersionedGraph};

/// (node_id, seal tag): a changed tag on the same id is an UPDATE, not an insert.
type Round = Vec<(u32, u8)>;

const A: u32 = 1;
const B: u32 = 2;
const C: u32 = 3;
const D: u32 = 4;

const V1: &[(u32, u8)] = &[(A, 0x10), (B, 0x10), (D, 0x10)];
const V2: &[(u32, u8)] = &[(A, 0x10), (B, 0x20), (C, 0x10)];

fn fsb_width(schema: &SchemaRef, name: &str) -> i32 {
    match schema.field_with_name(name).expect(name).data_type() {
        DataType::FixedSizeBinary(w) => *w,
        other => panic!("{name}: expected FixedSizeBinary, got {other:?}"),
    }
}

fn seal_bytes(node_id: u32, tag: u8, width: usize) -> Vec<u8> {
    let mut v = vec![tag; width];
    v[..4].copy_from_slice(&node_id.to_le_bytes());
    v
}

fn node_batch(round: &Round) -> RecordBatch {
    let schema = NodeSchema::arrow_schema_ref();
    let plane_w = fsb_width(&schema, "plane_s");
    let seal_w = fsb_width(&schema, "seal_s");
    let n = round.len();
    let mut planes = (0..3)
        .map(|_| FixedSizeBinaryBuilder::with_capacity(n, plane_w))
        .collect::<Vec<_>>();
    let mut seals = (0..3)
        .map(|_| FixedSizeBinaryBuilder::with_capacity(n, seal_w))
        .collect::<Vec<_>>();
    let plane = vec![0u8; plane_w as usize];
    for &(id, tag) in round {
        for p in planes.iter_mut() {
            p.append_value(&plane).unwrap();
        }
        for (k, s) in seals.iter_mut().enumerate() {
            s.append_value(seal_bytes(id, tag.wrapping_add(k as u8), seal_w as usize))
                .unwrap();
        }
    }
    let ids: Vec<u32> = round.iter().map(|r| r.0).collect();
    let enc: Vec<u32> = round.iter().map(|_| 1).collect();
    let mut cols: Vec<Arc<dyn Array>> = vec![Arc::new(UInt32Array::from(ids))];
    for p in planes.iter_mut() {
        cols.push(Arc::new(p.finish()));
    }
    for s in seals.iter_mut() {
        cols.push(Arc::new(s.finish()));
    }
    cols.push(Arc::new(UInt32Array::from(enc)));
    RecordBatch::try_new(schema, cols).unwrap()
}

fn empty_batch(schema: SchemaRef) -> RecordBatch {
    let cols = schema
        .fields()
        .iter()
        .map(|f| arrow_array::new_empty_array(f.data_type()))
        .collect();
    RecordBatch::try_new(schema, cols).unwrap()
}

/// `node_id -> seal_s ++ seal_p ++ seal_o` — the same material
/// `VersionedGraph::extract_node_seals` builds (`graph/versioned.rs:660`),
/// reproduced here because that helper is private.
async fn seals_at(ds: &Dataset) -> BTreeMap<u32, Vec<u8>> {
    let batches: Vec<RecordBatch> = ds
        .scan()
        .try_into_stream()
        .await
        .expect("scan")
        .try_collect()
        .await
        .expect("collect");
    let mut out = BTreeMap::new();
    for b in &batches {
        let ids = b
            .column_by_name("node_id")
            .and_then(|c| c.as_any().downcast_ref::<UInt32Array>())
            .expect("node_id");
        let seals: Vec<&FixedSizeBinaryArray> = ["seal_s", "seal_p", "seal_o"]
            .iter()
            .map(|c| {
                b.column_by_name(c)
                    .and_then(|x| x.as_any().downcast_ref::<FixedSizeBinaryArray>())
                    .expect("seal column")
            })
            .collect();
        for i in 0..b.num_rows() {
            let mut all = Vec::new();
            for s in &seals {
                all.extend_from_slice(s.value(i));
            }
            out.insert(ids.value(i), all);
        }
    }
    out
}

#[tokio::test(flavor = "multi_thread")]
async fn old_diff_gives_inserted_and_updated_and_removed_is_derivable() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let graph = VersionedGraph::local(tmp.path().to_str().unwrap());

    let mut versions = Vec::new();
    for round in [V1.to_vec(), V2.to_vec()] {
        let v = graph
            .commit_encounter_round(
                node_batch(&round),
                empty_batch(EdgeSchema::arrow_schema_ref()),
                empty_batch(FingerprintSchema::arrow_schema_ref()),
            )
            .await
            .expect("commit");
        versions.push(v);
    }
    let (v1, v2) = (versions[0], versions[1]);

    // ---- what the API DOES report ----------------------------------------
    let diff = graph.diff(v1, v2).await.expect("diff V1->V2");

    let inserted: BTreeSet<u32> = diff.new_nodes.iter().copied().collect();
    assert_eq!(
        inserted,
        BTreeSet::from([C]),
        "inserted = to - from = {{C}} (versioned.rs:556)"
    );

    let updated: BTreeSet<u32> = diff.modified_nodes.iter().copied().collect();
    assert_eq!(
        updated,
        BTreeSet::from([B]),
        "updated = seal-differing intersection = {{B}} (versioned.rs:560)"
    );

    // Anti-vacuity: A is in both versions with an UNCHANGED seal and must
    // appear in neither set — otherwise "updated" would just mean "present".
    assert!(
        !updated.contains(&A) && !inserted.contains(&A),
        "A is quiet"
    );

    // ---- the omission: GraphDiff has no removed-nodes field ---------------
    // D is gone in V2 and appears NOWHERE in the diff. Pinned; if a
    // `removed_nodes` field lands, this assertion must be re-pinned deliberately.
    assert!(
        !inserted.contains(&D) && !updated.contains(&D),
        "PINNED BLIND SPOT: removed node D is absent from GraphDiff: {diff:?}"
    );

    // ---- removals ARE detected, just not reported -------------------------
    // graph_seal_check walks `from_seals.keys()` and returns Staunen on a
    // removal (versioned.rs:632-637) — so the store distinguishes the states.
    assert_eq!(
        graph.graph_seal_check(v1, v2).await.expect("seal check"),
        GraphSealStatus::Staunen,
        "a removal is Staunen"
    );
    // The silence half: same version against itself is Wisdom.
    assert_eq!(
        graph.graph_seal_check(v2, v2).await.expect("seal check"),
        GraphSealStatus::Wisdom,
        "no change is Wisdom"
    );

    // ---- DERIVATION: removed = from - to, from the same material ----------
    // `diff()` materializes exactly these two maps (`from_nodes` / `to_nodes`,
    // versioned.rs:550-551) and their key sets (`from_ids` / `to_ids`, :553-554)
    // before throwing the difference away. Recomputing it here proves the
    // information is present in the store, not lost.
    let from_seals = seals_at(&graph.at_version(v1).await.expect("checkout V1")).await;
    let to_seals = seals_at(&graph.at_version(v2).await.expect("checkout V2")).await;
    let from_ids: BTreeSet<u32> = from_seals.keys().copied().collect();
    let to_ids: BTreeSet<u32> = to_seals.keys().copied().collect();

    let removed: BTreeSet<u32> = from_ids.difference(&to_ids).copied().collect();
    assert_eq!(
        removed,
        BTreeSet::from([D]),
        "removed = from - to = {{D}} — derivable from the SAME read diff() does"
    );

    // And the derivation reproduces the two sets the API does expose, so the
    // three are one pass over one pair of maps.
    let derived_inserted: BTreeSet<u32> = to_ids.difference(&from_ids).copied().collect();
    assert_eq!(derived_inserted, inserted);
    let derived_updated: BTreeSet<u32> = to_ids
        .intersection(&from_ids)
        .copied()
        .filter(|id| from_seals.get(id) != to_seals.get(id))
        .collect();
    assert_eq!(derived_updated, updated);
}
