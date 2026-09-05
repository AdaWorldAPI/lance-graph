//! D-LNC-2 — the row-identity probe.
//!
//! Plan: `.claude/plans/lance-convergence-staged-migration-v1.md` §5. The lance
//! 11 breaking changes cluster on ROW IDENTITY (fragment-id reuse across an
//! overwrite, stable-row-id migration, the delete delta). This probe pins what
//! `VersionedGraph` actually relies on, so the 10→11 bump is measured rather
//! than assumed:
//!
//! 1. **time travel** — `at_version(v)` returns the exact `(node_id, seal)` set
//!    that was written at `v`, for every `v` in `versions()`, including the
//!    versions BEFORE an overwrite and BEFORE a delete;
//! 2. **row addresses** — the physical `_rowaddr` of every row is stable across
//!    a re-open (a row-address-keyed reader must not see rows move);
//! 3. **fragment-id reuse across an overwrite** — measured, and pinned by
//!    policy (see [`fragment_reuse_policy`]). `VersionedGraph::write_batch`
//!    uses `WriteMode::Overwrite` for every commit after the first
//!    (`graph/versioned.rs`), so lance-format/lance#8206 ("stop reusing
//!    fragment ids across an overwrite") is a change we are EXPOSED to;
//! 4. **deletes** — a real `Dataset::delete` is seen by `graph_seal_check`
//!    (Staunen) and is INVISIBLE to `diff()` (it has no removed-nodes field) —
//!    pinned so that adding one forces a deliberate re-pin;
//! 5. **tagged retention** — after `cleanup_old_versions`, a tagged old version
//!    is still checkout-able and an untagged old version is gone (the fire
//!    half; without it the retention assertion is vacuous).
//!
//! # Cross-version mode
//!
//! With `LNC2_FIXTURE_DIR=<dir>` the probe persists the dataset instead of
//! using a tempdir. On an EMPTY dir it writes the fixture + `reference.tsv`
//! (one line per `version \t node_id \t rowaddr \t seal_hex`, plus the
//! fragment ids per version) and stops before cleanup. On a NON-empty dir it
//! verifies every version against the reference and THEN runs the cleanup arm.
//! Build once under lance 10 (write), once under lance 11 (verify): that is
//! the "a real dataset written under 10, opened under 11" assertion of D-LNC-2.
//! The persisted dir is a probe artefact, never a committed fixture.

use chrono::TimeDelta;
use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};
use std::sync::Arc;

use arrow_array::builder::FixedSizeBinaryBuilder;
use arrow_array::{Array, FixedSizeBinaryArray, RecordBatch, UInt32Array, UInt64Array};
use arrow_schema::{DataType, SchemaRef};
use futures::TryStreamExt;
use lance::dataset::Dataset;
use lance_graph::graph::blasgraph::columnar::{EdgeSchema, FingerprintSchema, NodeSchema};
use lance_graph::graph::versioned::{GraphSealStatus, VersionedGraph};

const ROW_ADDR: &str = "_rowaddr";

/// (node_id, seal tag). The seal bytes are derived from both, so a changed
/// tag on the same node is a MODIFIED node, not a new one.
type Round = Vec<(u32, u8)>;

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

/// What the probe expects for a version: node_id -> concatenated seal bytes.
fn expected(round: &Round) -> BTreeMap<u32, Vec<u8>> {
    let seal_w = fsb_width(&NodeSchema::arrow_schema_ref(), "seal_s") as usize;
    round
        .iter()
        .map(|&(id, tag)| {
            let mut all = Vec::with_capacity(3 * seal_w);
            for k in 0..3u8 {
                all.extend(seal_bytes(id, tag.wrapping_add(k), seal_w));
            }
            (id, all)
        })
        .collect()
}

/// node_id -> (seal bytes, physical row address), read with `_rowaddr`.
async fn snapshot(ds: &Dataset) -> BTreeMap<u32, (Vec<u8>, u64)> {
    let mut scanner = ds.scan();
    scanner.with_row_address();
    let batches: Vec<RecordBatch> = scanner
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
            .unwrap()
            .as_any()
            .downcast_ref::<UInt32Array>()
            .unwrap();
        let addrs = b
            .column_by_name(ROW_ADDR)
            .expect("_rowaddr column")
            .as_any()
            .downcast_ref::<UInt64Array>()
            .unwrap();
        let seals: Vec<&FixedSizeBinaryArray> = ["seal_s", "seal_p", "seal_o"]
            .iter()
            .map(|c| {
                b.column_by_name(c)
                    .unwrap()
                    .as_any()
                    .downcast_ref::<FixedSizeBinaryArray>()
                    .unwrap()
            })
            .collect();
        for i in 0..b.num_rows() {
            let mut all = Vec::new();
            for s in &seals {
                all.extend_from_slice(s.value(i));
            }
            let prev = out.insert(ids.value(i), (all, addrs.value(i)));
            assert!(
                prev.is_none(),
                "duplicate node_id {} in one version",
                ids.value(i)
            );
        }
    }
    out
}

fn fragment_ids(ds: &Dataset) -> BTreeSet<usize> {
    ds.get_fragments().iter().map(|f| f.id()).collect()
}

fn seals_of(snap: &BTreeMap<u32, (Vec<u8>, u64)>) -> BTreeMap<u32, Vec<u8>> {
    snap.iter().map(|(k, v)| (*k, v.0.clone())).collect()
}

/// Whether consecutive overwrite versions may share fragment ids.
///
/// `LNC2_FRAGMENT_REUSE=forbidden` pins the lance-11 behaviour
/// (lance-format/lance#8206); `=expected` pins the lance-10 behaviour;
/// unset = measure and print only. The two-sided pin lives in the plan's
/// disable table: run the lance-11 build with `expected` → must go RED.
fn fragment_reuse_policy() -> Option<bool> {
    match std::env::var("LNC2_FRAGMENT_REUSE").as_deref() {
        Ok("forbidden") => Some(false),
        Ok("expected") => Some(true),
        _ => None,
    }
}

enum Mode {
    /// Everything in one process, in a tempdir.
    SelfContained(tempfile::TempDir),
    /// Persist to `dir`, write `reference.tsv`, stop before cleanup.
    FixtureWrite(PathBuf),
    /// Verify `dir` against `reference.tsv`, then run cleanup.
    FixtureVerify(PathBuf),
}

impl Mode {
    fn detect() -> Self {
        match std::env::var("LNC2_FIXTURE_DIR") {
            Ok(d) => {
                let d = PathBuf::from(d);
                if d.join("nodes.lance").exists() {
                    Mode::FixtureVerify(d)
                } else {
                    std::fs::create_dir_all(&d).expect("fixture dir");
                    Mode::FixtureWrite(d)
                }
            }
            Err(_) => Mode::SelfContained(tempfile::tempdir().expect("tempdir")),
        }
    }
    fn dir(&self) -> &Path {
        match self {
            Mode::SelfContained(t) => t.path(),
            Mode::FixtureWrite(d) | Mode::FixtureVerify(d) => d,
        }
    }
}

const ROUNDS: [&[(u32, u8)]; 3] = [
    &[(1, 0x10), (2, 0x10), (3, 0x10)],            // A: create
    &[(1, 0x10), (2, 0x20), (3, 0x10), (4, 0x10)], // B: 2 modified, 4 new (overwrite)
    &[(1, 0x10), (3, 0x10), (4, 0x10)],            // C: 2 removed (overwrite)
];

fn reference_lines(
    versions: &[u64],
    snaps: &[BTreeMap<u32, (Vec<u8>, u64)>],
    frags: &[BTreeSet<usize>],
) -> String {
    let mut s = String::new();
    for ((v, snap), fr) in versions.iter().zip(snaps).zip(frags) {
        let ids: Vec<String> = fr.iter().map(|f| f.to_string()).collect();
        s.push_str(&format!("frag\t{v}\t{}\n", ids.join(",")));
        for (id, (seal, addr)) in snap {
            let hex: String = seal.iter().map(|b| format!("{b:02x}")).collect();
            s.push_str(&format!("row\t{v}\t{id}\t{addr}\t{hex}\n"));
        }
    }
    s
}

#[tokio::test(flavor = "multi_thread")]
async fn row_identity_holds_across_overwrite_delete_and_time_travel() {
    let mode = Mode::detect();
    let base = mode.dir().to_str().unwrap().to_string();
    let graph = VersionedGraph::local(&base);
    let rounds: Vec<Round> = ROUNDS.iter().map(|r| r.to_vec()).collect();

    // ---- write phase (self-contained + fixture-write) -----------------------
    let mut versions: Vec<u64> = Vec::new();
    let mut expected_by_version: Vec<BTreeMap<u32, Vec<u8>>> = Vec::new();
    if !matches!(mode, Mode::FixtureVerify(_)) {
        for round in &rounds {
            let v = graph
                .commit_encounter_round(
                    node_batch(round),
                    empty_batch(EdgeSchema::arrow_schema_ref()),
                    empty_batch(FingerprintSchema::arrow_schema_ref()),
                )
                .await
                .expect("commit");
            versions.push(v);
            expected_by_version.push(expected(round));
        }
        // D: a REAL delete (not an overwrite) — node 3 leaves.
        let mut ds = graph.open_nodes().await.expect("open nodes");
        let del = ds.delete("node_id = 3").await.expect("delete");
        assert_eq!(del.num_deleted_rows, 1, "exactly one row deleted");
        versions.push(del.new_dataset.version().version);
        let mut after = expected_by_version.last().unwrap().clone();
        after.remove(&3);
        expected_by_version.push(after);
        // Tag B: the version cleanup must retain.
        graph.tag_version("keep", versions[1]).await.expect("tag");
    } else {
        // Reconstruct the expectations from the rounds; versions come from the store.
        let listed = graph.versions().await.expect("versions");
        versions = listed.iter().map(|v| v.version).collect();
        for round in &rounds {
            expected_by_version.push(expected(round));
        }
        let mut after = expected_by_version.last().unwrap().clone();
        after.remove(&3);
        expected_by_version.push(after);
    }
    assert_eq!(versions.len(), 4, "A, B, C (overwrites) + D (delete)");

    // ---- 1. time travel: every version is byte-exact on (node_id, seal) ----
    let listed: Vec<u64> = graph
        .versions()
        .await
        .expect("versions")
        .iter()
        .map(|v| v.version)
        .collect();
    assert_eq!(
        listed, versions,
        "versions() lists exactly what was committed"
    );
    let mut snaps = Vec::new();
    let mut frags = Vec::new();
    for (v, want) in versions.iter().zip(&expected_by_version) {
        let ds = graph.at_version(*v).await.expect("checkout");
        assert_eq!(ds.version().version, *v);
        let snap = snapshot(&ds).await;
        assert_eq!(&seals_of(&snap), want, "version {v}: (node_id, seal) set");
        frags.push(fragment_ids(&ds));
        snaps.push(snap);
    }
    // The delete is visible as an absent row, not as a tombstoned present one.
    assert!(!snaps[3].contains_key(&3), "deleted node 3 absent at D");
    assert!(snaps[2].contains_key(&3), "node 3 present at C");

    // ---- 2. row addresses are stable across a re-open -----------------------
    for (v, snap) in versions.iter().zip(&snaps) {
        let again = snapshot(&graph.at_version(*v).await.expect("re-open")).await;
        assert_eq!(
            &again, snap,
            "version {v}: re-open yields identical _rowaddr"
        );
    }
    // A delete does not move the surviving rows (address stability across D).
    for (id, (_, addr)) in &snaps[3] {
        assert_eq!(
            snaps[2][id].1, *addr,
            "node {id}: address unchanged by the delete"
        );
    }

    // ---- 3. fragment-id reuse across the two OVERWRITES — measured ---------
    let mut reused = 0usize;
    let mut aliased = 0usize;
    for k in 0..2 {
        let shared: BTreeSet<_> = frags[k].intersection(&frags[k + 1]).collect();
        reused += shared.len();
        let by_addr: BTreeMap<u64, u32> = snaps[k].iter().map(|(id, (_, a))| (*a, *id)).collect();
        for (id, (_, a)) in &snaps[k + 1] {
            if let Some(prev) = by_addr.get(a) {
                if prev != id {
                    aliased += 1;
                }
            }
        }
        eprintln!(
            "LNC2 overwrite v{}->v{}: fragments {:?} -> {:?}, shared {:?}",
            versions[k],
            versions[k + 1],
            frags[k],
            frags[k + 1],
            shared
        );
    }
    eprintln!("LNC2 fragment ids reused across overwrites: {reused}; _rowaddr aliased to a DIFFERENT node_id: {aliased}");
    match fragment_reuse_policy() {
        Some(false) => assert_eq!(
            reused, 0,
            "lance 11: no fragment-id reuse across an overwrite (lance#8206)"
        ),
        Some(true) => assert!(
            reused > 0,
            "lance 10: fragment ids ARE reused across an overwrite"
        ),
        None => {}
    }

    // ---- 4. the delete: seen by graph_seal_check; diff() has two blind spots
    let (b, c, d) = (versions[1], versions[2], versions[3]);
    assert_eq!(
        graph.graph_seal_check(c, d).await.expect("seal check"),
        GraphSealStatus::Staunen,
        "a removed node is Staunen"
    );
    assert_eq!(
        graph.graph_seal_check(c, c).await.expect("seal check"),
        GraphSealStatus::Wisdom,
        "the same version is Wisdom (the silence half)"
    );
    // Blind spot 1: GraphDiff has no removed-nodes field. B->C drops node 2
    // through the overwrite path, in lockstep on all three datasets, and the
    // diff is EMPTY. If a `removed_nodes` field lands, re-pin this deliberately.
    let bc = graph.diff(b, c).await.expect("diff B->C");
    assert!(
        bc.new_nodes.is_empty() && bc.modified_nodes.is_empty() && bc.new_edges.is_empty(),
        "PINNED BLIND SPOT: a removal is an empty GraphDiff: {bc:?}"
    );
    let ab = graph.diff(versions[0], b).await.expect("diff A->B");
    assert_eq!(ab.new_nodes, vec![4]);
    assert_eq!(ab.modified_nodes, vec![2]);
    // Blind spot 2: diff() checks out the EDGES dataset at the NODES version
    // number. A direct `Dataset::delete` on nodes (D) advances nodes to a
    // version edges never had, so diff(C, D) is an error, not a diff. The
    // lockstep assumption is only true while every write goes through
    // `commit_encounter_round`. Pinned as measured; a version-map fix re-pins.
    assert!(
        graph.diff(c, d).await.is_err(),
        "PINNED LOCKSTEP COUPLING: diff across a nodes-only version must fail today"
    );

    // ---- fixture-write mode stops here (cleanup would destroy A and C) ----
    if let Mode::FixtureWrite(dir) = &mode {
        std::fs::write(
            dir.join("reference.tsv"),
            reference_lines(&versions, &snaps, &frags),
        )
        .expect("reference.tsv");
        eprintln!("LNC2 fixture written to {}", dir.display());
        return;
    }
    if let Mode::FixtureVerify(dir) = &mode {
        let want = std::fs::read_to_string(dir.join("reference.tsv")).expect("reference.tsv");
        let got = reference_lines(&versions, &snaps, &frags);
        assert_eq!(
            got, want,
            "reference written under the previous lance major"
        );
        eprintln!(
            "LNC2 fixture verified against reference.tsv ({} lines)",
            want.lines().count()
        );
    }

    // ---- 5. tagged retention survives cleanup; untagged does not -----------
    let ds = graph.open_nodes().await.expect("open");
    let stats = ds
        .cleanup_old_versions(TimeDelta::zero(), Some(true), Some(false))
        .await
        .expect("cleanup");
    eprintln!("LNC2 cleanup: {stats:?}");
    assert!(
        stats.old_versions >= 1,
        "cleanup removed at least one old version"
    );
    let kept = snapshot(
        &graph
            .at_version(versions[1])
            .await
            .expect("tagged B survives"),
    )
    .await;
    assert_eq!(
        kept, snaps[1],
        "tagged version B is byte-identical after cleanup"
    );
    let latest = snapshot(
        &graph
            .at_version(versions[3])
            .await
            .expect("latest survives"),
    )
    .await;
    assert_eq!(latest, snaps[3]);
    assert!(
        graph.at_version(versions[0]).await.is_err(),
        "FIRE HALF: untagged old version A is gone after cleanup"
    );
}
