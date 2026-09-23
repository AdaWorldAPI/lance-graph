//! D-LNC-5b — does the STABLE row id switch hold, and can it replace the seal?
//!
//! ## The question (pre-registered, 2026-09-23)
//!
//! `enable_stable_row_ids` is still documented as experimental in lance 11 and
//! 12 alike (`write.rs`: *"Experimental … stable after compaction operations,
//! but not after updates"*). Tombstone records need two things from it:
//!
//! 1. **identity** — a row keeps its `_rowid` across compaction, both update
//!    paths (`merge_insert`, `UpdateBuilder`), index maintenance and
//!    concurrent commits;
//! 2. **exact tombstones** — `delta().get_deleted_row_ids()` over a version
//!    range reports exactly the rows actually deleted: no update read as a
//!    delete, no compaction read as a delete.
//!
//! D-LNC-5a (`delta_version_columns_probe.rs`) measured the OTHER half: the
//! insert/update delta arms are empty WITHOUT the switch. This file measures
//! what the switch gives when it is ON.
//!
//! ## Does it replace the seal? — what the concurrent arm measures
//!
//! The seal (`persist_sink::freeze`, see
//! `.claude/knowledge/seal-vs-temporal-ordering-information.md`) provides a
//! cross-owner TOTAL order, a per-row last-writer-wins FOLD decided by that
//! order, and ONE COHORT per version (one cycle → one `DatasetVersion`, stamped
//! with its `base_version` read horizon). Row identity is a different axis
//! from all three, so the concurrent arm pins the observable consequences of
//! letting writers commit directly instead of sealing:
//!
//! - **cohort**: how many versions W concurrent writers mint (the seal mints
//!   ONE per cycle);
//! - **fold**: which write to a contended row survives, and what decides it
//!   (the seal decides by `stream_position`, fixed before the append).
//!
//! Every property below is asserted, so a lance bump that changes any of them
//! turns this file red. Each arm carries a control that must hold first, or
//! its result says nothing about lance.
//!
//! Run: `cargo test -p lance-graph --test stable_row_id_probe -- --nocapture`

use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

use arrow_array::{Array, Int32Array, RecordBatch, StringArray, UInt64Array};
use arrow_schema::{DataType, Field, Schema};
use futures::TryStreamExt;
use lance::dataset::optimize::{compact_files, CompactionOptions};
use lance::dataset::write::DeleteBuilder;
use lance::dataset::{
    Dataset, MergeInsertBuilder, UpdateBuilder, WhenMatched, WhenNotMatched, WriteMode, WriteParams,
};
use lance::index::DatasetIndexExt;
use lance_index::optimize::OptimizeOptions;
use lance_index::scalar::ScalarIndexParams;
use lance_index::IndexType;

/// `id` (the business key) and `val` (the payload).
fn schema() -> Arc<Schema> {
    Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("val", DataType::Utf8, false),
    ]))
}

fn batch(ids: &[i32], val: &str) -> RecordBatch {
    let vals: Vec<&str> = ids.iter().map(|_| val).collect();
    RecordBatch::try_new(
        schema(),
        vec![
            Arc::new(Int32Array::from(ids.to_vec())),
            Arc::new(StringArray::from(vals)),
        ],
    )
    .expect("batch")
}

fn reader(
    b: RecordBatch,
) -> arrow_array::RecordBatchIterator<
    std::vec::IntoIter<Result<RecordBatch, arrow_schema::ArrowError>>,
> {
    arrow_array::RecordBatchIterator::new(vec![Ok(b)].into_iter(), schema())
}

/// Write with stable row ids ON — every arm in this file keeps the switch on.
async fn write(path: &str, b: RecordBatch, create: bool) -> Dataset {
    let params = WriteParams {
        mode: if create {
            WriteMode::Create
        } else {
            WriteMode::Append
        },
        enable_stable_row_ids: true,
        ..Default::default()
    };
    Dataset::write(reader(b), path, Some(params))
        .await
        .expect("write")
}

/// `id -> (_rowid, val)` at the dataset's current version, via a full scan.
async fn rows(ds: &Dataset) -> BTreeMap<i32, (u64, String)> {
    let mut sc = ds.scan();
    sc.with_row_id();
    let bs: Vec<RecordBatch> = sc
        .try_into_stream()
        .await
        .expect("scan")
        .try_collect()
        .await
        .expect("collect");
    let mut out = BTreeMap::new();
    for b in bs {
        let ids = b
            .column_by_name("id")
            .expect("id")
            .as_any()
            .downcast_ref::<Int32Array>()
            .expect("i32")
            .clone();
        let vals = b
            .column_by_name("val")
            .expect("val")
            .as_any()
            .downcast_ref::<StringArray>()
            .expect("utf8")
            .clone();
        let rid = b
            .column_by_name("_rowid")
            .expect("_rowid")
            .as_any()
            .downcast_ref::<UInt64Array>()
            .expect("u64")
            .clone();
        for i in 0..b.num_rows() {
            let prev = out.insert(ids.value(i), (rid.value(i), vals.value(i).to_string()));
            assert!(
                prev.is_none(),
                "key {} visible twice in one version",
                ids.value(i)
            );
        }
    }
    out
}

/// The tombstone read: row ids deleted between `a` and `b`, sorted.
async fn deleted(ds: &Dataset, a: u64, b: u64) -> Vec<u64> {
    let d = ds
        .delta()
        .with_begin_version(a)
        .with_end_version(b)
        .build()
        .expect("delta");
    let bs: Vec<RecordBatch> = d
        .get_deleted_row_ids()
        .await
        .expect("get_deleted_row_ids")
        .try_collect()
        .await
        .expect("collect");
    let mut out = Vec::new();
    for b in bs {
        let c = b
            .column(0)
            .as_any()
            .downcast_ref::<UInt64Array>()
            .expect("u64");
        out.extend((0..c.len()).map(|i| c.value(i)));
    }
    out.sort_unstable();
    out
}

/// Keys present in both maps whose `_rowid` differs.
fn moved(a: &BTreeMap<i32, (u64, String)>, b: &BTreeMap<i32, (u64, String)>) -> Vec<i32> {
    a.iter()
        .filter(|(k, (rid, _))| b.get(k).is_some_and(|(r2, _)| r2 != rid))
        .map(|(k, _)| *k)
        .collect()
}

/// `_rowid` of `id = key` read two ways — through the scalar index and by a
/// plain filtered scan — so index maintenance can be checked against truth.
async fn lookup(ds: &Dataset, key: i32, use_index: bool) -> Vec<u64> {
    let mut sc = ds.scan();
    sc.with_row_id()
        .use_scalar_index(use_index)
        .filter(&format!("id = {key}"))
        .expect("filter");
    let bs: Vec<RecordBatch> = sc
        .try_into_stream()
        .await
        .expect("scan")
        .try_collect()
        .await
        .expect("collect");
    let mut out = Vec::new();
    for b in bs {
        let c = b
            .column_by_name("_rowid")
            .expect("_rowid")
            .as_any()
            .downcast_ref::<UInt64Array>()
            .expect("u64");
        out.extend((0..c.len()).map(|i| c.value(i)));
    }
    out
}

/// A1 — single writer, every row-changing operation in turn.
#[tokio::test(flavor = "multi_thread")]
async fn a_single_writer_keeps_every_row_id_and_tombstones_only_real_deletes() {
    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path().join("ds").to_string_lossy().to_string();
    write(&path, batch(&[1, 2, 3, 4], "a"), true).await;
    let mut ds = write(&path, batch(&[5, 6, 7, 8], "a"), false).await;
    let base = rows(&ds).await;

    // Compaction — control: it must actually rewrite fragments.
    let m = compact_files(&mut ds, CompactionOptions::default(), None)
        .await
        .expect("compact");
    assert!(
        m.fragments_removed > 0,
        "CONTROL: compaction rewrote nothing"
    );
    let after = rows(&ds).await;
    eprintln!(
        "A1 compaction removed {} fragments; moved ids: {:?}",
        m.fragments_removed,
        moved(&base, &after)
    );
    assert_eq!(
        moved(&base, &after),
        Vec::<i32>::new(),
        "compaction moved a row id"
    );

    // Update via merge_insert.
    let v0 = ds.version().version;
    MergeInsertBuilder::try_new(Arc::new(ds.clone()), vec!["id".into()])
        .expect("merge")
        .when_matched(WhenMatched::UpdateAll)
        .when_not_matched(WhenNotMatched::InsertAll)
        .try_build()
        .expect("build")
        .execute_reader(reader(batch(&[3], "z")))
        .await
        .expect("upsert");
    let ds = Dataset::open(&path).await.expect("open");
    let upd = rows(&ds).await;
    assert_eq!(upd[&3].1, "z", "CONTROL: the upsert must have changed id 3");
    let tomb = deleted(&ds, v0, ds.version().version).await;
    eprintln!(
        "A1 merge_insert update: id 3 _rowid {} -> {}; moved: {:?}; tombstones: {tomb:?}",
        after[&3].0,
        upd[&3].0,
        moved(&after, &upd)
    );
    assert_eq!(
        moved(&after, &upd),
        Vec::<i32>::new(),
        "merge_insert moved a row id"
    );
    assert!(
        tomb.is_empty(),
        "an update was reported as a delete: {tomb:?}"
    );

    // Update via UpdateBuilder (the SQL-style path).
    let v0 = ds.version().version;
    let r = UpdateBuilder::new(Arc::new(ds.clone()))
        .update_where("id = 6")
        .expect("where")
        .set("val", "'y'")
        .expect("set")
        .build()
        .expect("build")
        .execute()
        .await
        .expect("update");
    assert_eq!(
        r.rows_updated, 1,
        "CONTROL: UpdateBuilder must update exactly id 6"
    );
    let mut ds = Dataset::open(&path).await.expect("open");
    let upd2 = rows(&ds).await;
    let tomb = deleted(&ds, v0, ds.version().version).await;
    eprintln!(
        "A1 UpdateBuilder update: moved: {:?}; tombstones: {tomb:?}",
        moved(&upd, &upd2)
    );
    assert_eq!(
        moved(&upd, &upd2),
        Vec::<i32>::new(),
        "UpdateBuilder moved a row id"
    );
    assert!(
        tomb.is_empty(),
        "an update was reported as a delete: {tomb:?}"
    );

    // A real delete — the tombstone must be exactly that row.
    let v0 = ds.version().version;
    let victim = upd2[&5].0;
    ds.delete("id = 5").await.expect("delete");
    let del = rows(&ds).await;
    let tomb = deleted(&ds, v0, ds.version().version).await;
    eprintln!("A1 delete id 5 (_rowid {victim}): tombstones: {tomb:?}");
    assert_eq!(
        tomb,
        vec![victim],
        "tombstones must be exactly the deleted row"
    );
    assert_eq!(
        moved(&upd2, &del),
        Vec::<i32>::new(),
        "a delete moved another row's id"
    );

    // Compaction again, now materializing the deletion — no new tombstones.
    let v0 = ds.version().version;
    compact_files(&mut ds, CompactionOptions::default(), None)
        .await
        .expect("compact");
    let fin = rows(&ds).await;
    let tomb = deleted(&ds, v0, ds.version().version).await;
    assert_eq!(
        moved(&del, &fin),
        Vec::<i32>::new(),
        "second compaction moved a row id"
    );
    assert!(
        tomb.is_empty(),
        "compaction was reported as a delete: {tomb:?}"
    );
}

const WRITERS: i32 = 4;
const ROUNDS: i32 = 6;
const CONTENDED: i32 = 0;

/// What one writer committed.
struct Commit {
    writer: i32,
    round: i32,
    version: u64,
}

/// A2 — W writers commit concurrently to one dataset carrying a BTree index.
///
/// Every round each writer upserts the contended key (`id = 0`) with its own
/// value and inserts one fresh key; once, each writer deletes one key it owns.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn concurrent_writers_keep_row_ids_but_mint_one_version_per_commit() {
    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path().join("ds").to_string_lossy().to_string();
    let seed: Vec<i32> = (0..64).collect();
    let mut ds = write(&path, batch(&seed, "seed"), true).await;
    ds.create_index_builder(&["id"], IndexType::BTree, &ScalarIndexParams::default())
        .name("id_idx".to_string())
        .await
        .expect("create index");
    let base_version = ds.version().version;
    let base = rows(&ds).await;

    let mut tasks = Vec::new();
    for w in 0..WRITERS {
        let path = path.clone();
        tasks.push(tokio::spawn(async move {
            let mut commits = Vec::new();
            let mut failures = 0u32;
            for r in 0..ROUNDS {
                let ds = Arc::new(Dataset::open(&path).await.expect("open"));
                let fresh = 1000 + w * 100 + r;
                let job = MergeInsertBuilder::try_new(ds, vec!["id".into()])
                    .expect("merge")
                    .when_matched(WhenMatched::UpdateAll)
                    .when_not_matched(WhenNotMatched::InsertAll)
                    .conflict_retries(50)
                    .try_build()
                    .expect("build");
                let b = RecordBatch::try_new(
                    schema(),
                    vec![
                        Arc::new(Int32Array::from(vec![CONTENDED, fresh])),
                        Arc::new(StringArray::from(vec![
                            format!("w{w}r{r}"),
                            format!("w{w}r{r}"),
                        ])),
                    ],
                )
                .expect("batch");
                match job.execute_reader(reader(b)).await {
                    Ok((new, _)) => commits.push(Commit {
                        writer: w,
                        round: r,
                        version: new.version().version,
                    }),
                    Err(_) => failures += 1,
                }
                if r == ROUNDS / 2 {
                    let ds = Arc::new(Dataset::open(&path).await.expect("open"));
                    let victim = 8 + w; // keys 8..12, one per writer
                    match DeleteBuilder::new(ds, format!("id = {victim}"))
                        .conflict_retries(50)
                        .execute()
                        .await
                    {
                        Ok(_) => {}
                        Err(_) => failures += 1,
                    }
                }
            }
            (commits, failures)
        }));
    }
    let mut commits = Vec::new();
    let mut failures = 0;
    for t in tasks {
        let (c, f) = t.await.expect("join");
        commits.extend(c);
        failures += f;
    }
    assert_eq!(
        failures, 0,
        "CONTROL: every concurrent commit must land (conflict retries)"
    );

    let mut ds = Dataset::open(&path).await.expect("open");
    let end_version = ds.version().version;
    let end = rows(&ds).await;
    let deletes = WRITERS as u64;
    let upserts = (WRITERS * ROUNDS) as u64;

    // ── cohort: how many versions did W concurrent writers mint? ────────────
    let minted = end_version - base_version;
    eprintln!("A2 {WRITERS} writers x {ROUNDS} rounds: {upserts} upserts + {deletes} deletes committed; versions minted = {minted}");
    assert_eq!(
        minted,
        upserts + deletes,
        "every commit is its own version — nothing groups a cycle into one cohort"
    );

    // ── identity: every surviving seed row kept its id; fresh ids unique ────
    let seed_moved = moved(&base, &end);
    let ids: BTreeSet<u64> = end.values().map(|(r, _)| *r).collect();
    eprintln!(
        "A2 seed rows whose _rowid moved: {seed_moved:?}; distinct _rowid = {} over {} live rows",
        ids.len(),
        end.len()
    );
    assert_eq!(
        seed_moved,
        Vec::<i32>::new(),
        "a concurrent commit moved a row id"
    );
    assert_eq!(ids.len(), end.len(), "two live rows share a _rowid");
    assert_eq!(
        end.len(),
        64 - WRITERS as usize + upserts as usize,
        "live row count"
    );

    // ── tombstones: exactly the deleted keys, nothing from 25 upserts ───────
    let want: Vec<u64> = {
        let mut v: Vec<u64> = (0..WRITERS).map(|w| base[&(8 + w)].0).collect();
        v.sort_unstable();
        v
    };
    let tomb = deleted(&ds, base_version, end_version).await;
    eprintln!("A2 tombstones over the whole run: {tomb:?} (expected {want:?})");
    assert_eq!(
        tomb, want,
        "tombstones must be exactly the four deleted keys"
    );

    // ── fold: which write to the contended row survived? ────────────────────
    let last = commits.iter().max_by_key(|c| c.version).expect("commits");
    eprintln!(
        "A2 contended key final val = {:?}; highest-version upsert = w{}r{} @v{}",
        end[&CONTENDED].1, last.writer, last.round, last.version
    );
    assert_eq!(
        end[&CONTENDED].1,
        format!("w{}r{}", last.writer, last.round),
        "the contended row is decided by commit order"
    );
    assert_eq!(
        end[&CONTENDED].0, base[&CONTENDED].0,
        "the contended row kept its id through every upsert"
    );

    // ── index: CONTROL first — the filter must actually run through the BTree,
    // or "index == scan" below compares a scan with itself.
    let plan = {
        let mut sc = ds.scan();
        sc.use_scalar_index(true).filter("id = 3").expect("filter");
        sc.explain_plan(true).await.expect("plan")
    };
    let plan_off = {
        let mut sc = ds.scan();
        sc.use_scalar_index(false).filter("id = 3").expect("filter");
        sc.explain_plan(true).await.expect("plan")
    };
    assert!(
        plan.contains("ScalarIndexQuery"),
        "CONTROL: the lookup must run through the BTree:\n{plan}"
    );
    assert!(
        !plan_off.contains("ScalarIndexQuery"),
        "CONTROL: the comparison scan must NOT use the index:\n{plan_off}"
    );
    // ── index: lookups through the BTree agree with a plain scan ────────────
    for k in end.keys().copied().chain(8..8 + WRITERS) {
        assert_eq!(
            lookup(&ds, k, true).await,
            lookup(&ds, k, false).await,
            "index disagrees with scan for id {k}"
        );
    }
    compact_files(&mut ds, CompactionOptions::default(), None)
        .await
        .expect("compact");
    ds.optimize_indices(&OptimizeOptions::default())
        .await
        .expect("optimize indices");
    let fin = rows(&ds).await;
    assert_eq!(
        moved(&end, &fin),
        Vec::<i32>::new(),
        "compaction + index optimize moved a row id"
    );
    for k in fin.keys().copied().chain(8..8 + WRITERS) {
        let via_index = lookup(&ds, k, true).await;
        assert_eq!(
            via_index,
            lookup(&ds, k, false).await,
            "index disagrees with scan for id {k} after compaction"
        );
        assert_eq!(
            via_index.first(),
            fin.get(&k).map(|(r, _)| r),
            "index returns the wrong row for id {k}"
        );
    }
    eprintln!(
        "A2 after compaction + optimize_indices: no id moved; index == scan for {} keys",
        fin.len() + WRITERS as usize
    );
}
