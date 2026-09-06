//! D-LNC-5a — are the delta version columns populated WITHOUT stable row ids?
//!
//! ## The question, pre-registered
//!
//! lance 11's `Dataset::delta()` exposes three arms. Reading the 11.0.0 source
//! showed that only ONE of them documents a stable-row-id requirement:
//!
//! | arm | mechanism | gates on `uses_stable_row_ids()` in its own source |
//! |---|---|---|
//! | `get_deleted_row_ids` | row-id set difference | YES ("Requires stable row ids at both endpoints") |
//! | `get_inserted_rows` | filter `_row_created_at_version > begin AND <= end` | no |
//! | `get_updated_rows` | filter on both version columns | no |
//!
//! `_row_created_at_version` / `_row_last_updated_at_version` are ordinary
//! schema columns (constants from `lance_core`), filtered through
//! `scanner.filter`. So the insert/update arms have no *explicit* gate. What
//! the source does NOT settle is whether those columns are POPULATED and
//! MEANINGFUL when the dataset is written in the default physical-row-address
//! mode (`enable_stable_row_ids: false`).
//!
//! That is what this probe measures, and nothing else.
//!
//! ## Why it matters
//!
//! Every production write path in this repo was surveyed (2026-09-06, nine
//! sites) and NONE configures a row-id field. If the insert/update arms work
//! on physical addresses, then `VersionedGraph::diff`'s two full
//! materializations can be replaced by two native delta reads with NO new
//! addressing regime — `GraphDiff`'s `new_nodes` maps onto `get_inserted_rows`
//! and `modified_nodes` onto `get_updated_rows`, one to one. If they do NOT,
//! the whole insert/update half is gated behind the same stable-row-id
//! decision as the delete arm, which is D-LNC-5's to make.
//!
//! ## Arms, each two-sided
//!
//! - **A1 (physical, the default)** — write v1, append v2, ask for the delta.
//!   PASS means `get_inserted_rows(1 -> 2)` returns exactly the v2 rows.
//!   FAIL (empty, or an error) means the columns are not populated without
//!   stable row ids.
//! - **A2 (stable row ids ON)** — identical writes with
//!   `enable_stable_row_ids: true`. This is the CONTROL: if A2 also comes back
//!   empty the probe is measuring its own wiring, not the feature, and the
//!   result must be discarded rather than reported. A null result is a claim
//!   about the apparatus until this arm proves otherwise.
//! - **A3 (update arm)** — overwrite an existing key's value and ask
//!   `get_updated_rows`. Same two-sided treatment.
//!
//! Run: `cargo test -p lance-graph --test delta_version_columns_probe -- --nocapture`

use std::sync::Arc;

use arrow_array::{Int32Array, RecordBatch, StringArray};
use arrow_schema::{DataType, Field, Schema};
use futures::TryStreamExt;
use lance::dataset::{
    Dataset, MergeInsertBuilder, WhenMatched, WhenNotMatched, WriteMode, WriteParams,
};

/// The two-column probe schema: an `id` key and a `val` payload.
fn schema() -> Arc<Schema> {
    Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("val", DataType::Utf8, false),
    ]))
}

/// One `RecordBatch` over [`schema`] from parallel id/val slices.
fn batch(ids: &[i32], vals: &[&str]) -> RecordBatch {
    RecordBatch::try_new(
        schema(),
        vec![
            Arc::new(Int32Array::from(ids.to_vec())),
            Arc::new(StringArray::from(vals.to_vec())),
        ],
    )
    .expect("batch")
}

/// Write `batch` at `path`, creating on the first call and appending after.
async fn write(path: &str, b: RecordBatch, stable: bool, create: bool) -> Dataset {
    let params = WriteParams {
        mode: if create {
            WriteMode::Create
        } else {
            WriteMode::Append
        },
        enable_stable_row_ids: stable,
        ..Default::default()
    };
    let reader = arrow_array::RecordBatchIterator::new(vec![Ok(b)].into_iter(), schema());
    Dataset::write(reader, path, Some(params))
        .await
        .expect("write")
}

/// How many rows a delta stream yields.
async fn count(stream: lance::dataset::scanner::DatasetRecordBatchStream) -> usize {
    let batches: Vec<RecordBatch> = stream.try_collect().await.expect("collect");
    batches.iter().map(RecordBatch::num_rows).sum()
}

/// One arm: build a two-version dataset and report the inserted-row count.
async fn inserted_between_v1_and_v2(dir: &tempfile::TempDir, stable: bool) -> usize {
    let path = dir.path().join("ds").to_string_lossy().to_string();
    let ds = write(&path, batch(&[1, 2], &["a", "b"]), stable, true).await;
    let v1 = ds.version().version;
    let ds = write(&path, batch(&[3, 4], &["c", "d"]), stable, false).await;
    let v2 = ds.version().version;
    assert!(v2 > v1, "the append must advance the version: {v1} -> {v2}");

    let delta = ds
        .delta()
        .with_begin_version(v1)
        .with_end_version(v2)
        .build()
        .expect("delta builder");
    count(delta.get_inserted_rows().await.expect("get_inserted_rows")).await
}

/// The insert arm. Runs [`inserted_between_v1_and_v2`] twice over two
/// independent datasets — stable row ids on, then off — and prints both. The
/// stable arm is the pre-registered CONTROL: it must report a non-zero count,
/// or the physical arm's zero says nothing about lance and only that this
/// probe failed to write anything worth counting.
#[tokio::test(flavor = "multi_thread")]
async fn the_insert_delta_arm_on_physical_addresses_vs_stable_row_ids() {
    let dir = tempfile::tempdir().expect("tempdir");
    let dir2 = tempfile::tempdir().expect("tempdir");

    // A2 first — the CONTROL. If this is 0 the probe measures its own wiring.
    let stable = inserted_between_v1_and_v2(&dir2, true).await;
    eprintln!("A2 stable_row_ids=true  inserted rows v1->v2 = {stable}");
    assert_eq!(
        stable, 2,
        "CONTROL FAILED: with stable row ids ON the insert arm must report the \
         2 appended rows. A zero here means this probe is measuring its own \
         wiring, so the A1 result below is uninterpretable and must NOT be \
         reported as a finding about lance."
    );

    // A1 — the question.
    let physical = inserted_between_v1_and_v2(&dir, false).await;
    eprintln!("A1 stable_row_ids=false inserted rows v1->v2 = {physical}");

    // Deliberately NOT an assert on a hoped-for value: both outcomes are real
    // findings and the point is to LEARN which. The gate is only that the
    // control held, which is asserted above.
    if physical == 2 {
        eprintln!(
            "RESULT: the insert delta arm WORKS on physical row addresses. \
             GraphDiff's new_nodes/modified_nodes can be served natively with \
             no new addressing regime."
        );
    } else {
        eprintln!(
            "RESULT: the insert delta arm returns {physical} (expected 2) \
             WITHOUT stable row ids. The version columns are not usable in \
             physical-address mode; the insert/update half is gated behind \
             the same D-LNC-5 decision as the delete arm."
        );
    }
}

/// The `val` stored for `id`, by scanning — the update arm's evidence that an
/// update actually committed, rather than that a delta arm read zero.
async fn val_of(ds: &Dataset, id: i32) -> Option<String> {
    use arrow_array::Array;
    let stream = ds.scan().try_into_stream().await.expect("scan");
    let batches: Vec<RecordBatch> = stream.try_collect().await.expect("collect");
    for b in batches {
        let ids = b
            .column_by_name("id")?
            .as_any()
            .downcast_ref::<Int32Array>()?;
        let vals = b
            .column_by_name("val")?
            .as_any()
            .downcast_ref::<StringArray>()?;
        for i in 0..b.num_rows() {
            if ids.value(i) == id {
                return Some(vals.value(i).to_string());
            }
        }
    }
    None
}

/// A3 — the update arm, with its own pre-registered control.
///
/// ⊘ The first version of this test asserted nothing about whether an update
/// had HAPPENED, so its zero was uninterpretable in exactly the way a null
/// result always is until the apparatus is ruled out: "the delta arm does not
/// report updates" and "no update was committed" produce the same zero. It now
/// establishes both preconditions before reading the delta at all — the
/// version advanced, AND the row's value actually changed — so a zero
/// afterwards is a statement about lance rather than about this file.
#[tokio::test(flavor = "multi_thread")]
async fn the_update_arm_reports_a_changed_row_under_both_modes() {
    for stable in [true, false] {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("ds").to_string_lossy().to_string();

        let ds = write(&path, batch(&[1, 2], &["a", "b"]), stable, true).await;
        let v1 = ds.version().version;
        assert_eq!(
            val_of(&ds, 1).await.as_deref(),
            Some("a"),
            "precondition: row 1 starts at 'a' (stable_row_ids={stable})"
        );

        // An UPDATE, not an append: same key, new value, through the same
        // `merge_insert` path `LanceCycleWriter` itself uses. `Dataset::update`
        // does not exist in lance 11 — the upsert IS merge-on-primary-key.
        let ds_arc = Arc::new(Dataset::open(&path).await.expect("open"));
        let reader = arrow_array::RecordBatchIterator::new(
            vec![Ok(batch(&[1], &["z"]))].into_iter(),
            schema(),
        );
        MergeInsertBuilder::try_new(ds_arc, vec!["id".to_string()])
            .expect("merge builder")
            // `when_matched` DEFAULTS to `DoNothing` — find-or-create. Without
            // this line the merge is a no-op on an existing key, which is
            // exactly what the control below caught on the first run.
            .when_matched(WhenMatched::UpdateAll)
            .when_not_matched(WhenNotMatched::InsertAll)
            .try_build()
            .expect("build")
            .execute_reader(reader)
            .await
            .expect("upsert");

        let ds = Dataset::open(&path).await.expect("reopen");
        let v2 = ds.version().version;

        // ── the control, both halves ────────────────────────────────────────
        assert!(
            v2 > v1,
            "CONTROL: the upsert must advance the version ({v1} -> {v2}, \
             stable_row_ids={stable}); without that there is no version range \
             for a delta to be read over and any count below is vacuous"
        );
        assert_eq!(
            val_of(&ds, 1).await.as_deref(),
            Some("z"),
            "CONTROL: the upsert must have CHANGED row 1 to 'z' \
             (stable_row_ids={stable}); a delta reading zero over an update \
             that never happened says nothing about lance"
        );

        let delta = ds
            .delta()
            .with_begin_version(v1)
            .with_end_version(v2)
            .build()
            .expect("delta builder");
        let n = count(delta.get_updated_rows().await.expect("get_updated_rows")).await;
        eprintln!(
            "A3 update arm stable_row_ids={stable} v{v1}->v{v2}: control HELD \
             (version advanced, value changed); updated rows = {n}"
        );
    }
}
