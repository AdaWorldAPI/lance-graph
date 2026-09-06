//! D-MW-P2 — is a MemWAL `put` atomic across a real process kill?
//!
//! ## The pre-registered question
//!
//! The `08-memwal-vs-batchwriter.md` audit returned **FOLD** (keep the
//! descriptor/cast layer, put durability on MemWAL) and named exactly one
//! measurement that would overturn it:
//!
//! > Can `ShardWriter::put` seal N landing rows plus the frame row atomically?
//! > Kill mid-flush on a 5,000-row cycle. Observing 0 or 5,000 ⇒ FOLD stands.
//! > Anything in between ⇒ KEEP, because the single-atomic-commit-per-cycle
//! > guarantee is the whole point of `commit_cycle`.
//!
//! ## This is a FALSIFICATION, not an exploration
//!
//! lance 11 documents the answer already, in `ShardWriterConfig`:
//!
//! - `max_wal_buffer_size` — *"This is a soft threshold - write batches are
//!   atomic and won't be split."*
//! - `durable_write: true` — *"Each write waits for WAL persistence before
//!   returning. Guarantees no data loss on crash."*
//!
//! So the probe exists to try to BREAK a documented claim, which is the only
//! reason to run it: a doc-comment claim is not a behaviour. A partial row
//! count would falsify lance's own documentation; matching it proves nothing
//! new about lance but does establish that the guarantee the FOLD verdict
//! leans on is real on this platform and this version.
//!
//! ## Arms
//!
//! - **A1 — clean drop, no `close()`.** Writer A puts, is dropped without
//!   closing, writer B reopens and replays. This is NOT a crash (it unwinds),
//!   so it is the WIRING CONTROL: if replay does not recover a cleanly-dropped
//!   write, the probe is measuring its own setup and A2 is uninterpretable.
//! - **A2 — SIGKILL mid-put.** A child process puts `ROWS` rows and is killed
//!   with SIGKILL at a sweep of delays. SIGKILL is chosen deliberately: it
//!   gives no unwinding, no `Drop`, no flush-on-exit, which is the only way to
//!   ask about crash atomicity rather than about destructors.
//!
//! Every recovered count is reported. The verdict is a property of the SET:
//! all counts in {0, ROWS} means atomic; ANY other value falsifies.
//!
//! Run: `cargo test -p lance-graph --test memwal_atomicity_probe -- --nocapture`

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

use arrow_array::{Int32Array, RecordBatch, StringArray};
use arrow_schema::{DataType, Field, Schema as ArrowSchema};
use lance::dataset::mem_wal::{ShardWriter, ShardWriterConfig};
use lance::io::ObjectStore;
use object_store::path::Path;
use uuid::Uuid;

/// Rows per put — the audit's "5,000-row cycle".
const ROWS: i32 = 5_000;

/// The env var that puts the test binary into child mode.
const CHILD_ENV: &str = "MW_P2_CHILD_DIR";

/// A fixed shard id so parent and child address the same shard without IPC.
fn shard_id() -> Uuid {
    Uuid::from_u128(0x4d57_5032_0000_0000_0000_0000_0000_0001)
}

/// `id` carries the unenforced-primary-key marker MemWAL requires.
fn schema() -> Arc<ArrowSchema> {
    let pk: HashMap<String, String> = [(
        "lance-schema:unenforced-primary-key".to_string(),
        "1".to_string(),
    )]
    .into_iter()
    .collect();
    Arc::new(ArrowSchema::new(vec![
        Field::new("id", DataType::Int32, false).with_metadata(pk),
        Field::new("val", DataType::Utf8, false),
    ]))
}

fn batch(n: i32) -> RecordBatch {
    let ids: Vec<i32> = (0..n).collect();
    let vals: Vec<String> = (0..n).map(|i| format!("v{i}")).collect();
    RecordBatch::try_new(
        schema(),
        vec![
            Arc::new(Int32Array::from(ids)),
            Arc::new(StringArray::from(vals)),
        ],
    )
    .expect("batch")
}

/// Durable writes with a live flush ticker — `durable_write` requires a
/// positive interval or `open` refuses (a put would otherwise have nothing to
/// drive its WAL append).
fn config() -> ShardWriterConfig {
    ShardWriterConfig {
        shard_id: shard_id(),
        shard_spec_id: 0,
        durable_write: true,
        max_wal_buffer_size: 1024 * 1024,
        max_wal_flush_interval: Some(Duration::from_millis(10)),
        max_memtable_size: 64 * 1024 * 1024,
        manifest_scan_batch_size: 2,
        ..Default::default()
    }
}

async fn store_at(dir: &str) -> (Arc<ObjectStore>, Path, String) {
    let uri = format!("file://{dir}");
    let (store, path) = ObjectStore::from_uri(&uri).await.expect("from_uri");
    (store, path, uri)
}

async fn open_writer(dir: &str) -> ShardWriter {
    let (store, path, uri) = store_at(dir).await;
    ShardWriter::open(store, path, uri, config(), schema(), vec![])
        .await
        .expect("ShardWriter::open")
}

/// Reopen the shard and report how many rows replay recovered.
async fn recovered_rows(dir: &str) -> u64 {
    let w = open_writer(dir).await;
    let n = w.memtable_stats().await.expect("memtable_stats").row_count as u64;
    w.close().await.expect("close");
    n
}

// ── A1: the wiring control ──────────────────────────────────────────────────

#[tokio::test(flavor = "multi_thread")]
async fn a1_control_a_cleanly_dropped_put_is_recovered_by_replay() {
    let dir = tempfile::tempdir().expect("tempdir");
    let d = dir.path().to_string_lossy().to_string();

    {
        let w = open_writer(&d).await;
        w.put(vec![batch(ROWS)]).await.expect("put");
        // Deliberately no close(): the WAL persists, the MemTable does not.
    }

    let n = recovered_rows(&d).await;
    eprintln!("A1 clean-drop recovered rows = {n} (expected {ROWS})");
    assert_eq!(
        n, ROWS as u64,
        "CONTROL FAILED: replay must recover a cleanly-dropped durable put. \
         A mismatch here means this probe is measuring its own setup, so the \
         A2 kill sweep below would be uninterpretable and must NOT be reported \
         as a finding about lance."
    );
}

// ── A2: the real question ───────────────────────────────────────────────────

/// Child mode. Runs only when `MW_P2_CHILD_DIR` is set; otherwise a no-op so
/// the ordinary test run does not hang.
#[tokio::test(flavor = "multi_thread")]
async fn mw_p2_child_body() {
    let Ok(dir) = std::env::var(CHILD_ENV) else {
        return;
    };
    let w = open_writer(&dir).await;
    // The parent kills us somewhere in here. With `durable_write` the put does
    // not return until its WAL append lands, so a kill before this line
    // returns is a kill mid-flush.
    let _ = w.put(vec![batch(ROWS)]).await;
    eprintln!("CHILD: put returned");
    // Stay alive so a late kill lands after durability rather than ending the
    // process for us — the sweep needs both sides of the boundary.
    std::thread::sleep(Duration::from_secs(30));
}

#[tokio::test(flavor = "multi_thread")]
async fn a2_sigkill_mid_put_leaves_all_or_nothing() {
    if std::env::var(CHILD_ENV).is_ok() {
        return; // we ARE a child; do not recurse.
    }

    let exe = std::env::current_exe().expect("current_exe");
    let mut observed: Vec<(u64, u64)> = Vec::new();

    // A sweep rather than one delay: the flush boundary cannot be hit on
    // purpose, so the honest instrument brackets it and reports every count.
    for delay_ms in [5u64, 15, 30, 60, 120, 250, 500] {
        let dir = tempfile::tempdir().expect("tempdir");
        let d = dir.path().to_string_lossy().to_string();

        let mut child = std::process::Command::new(&exe)
            .args(["--exact", "mw_p2_child_body", "--nocapture"])
            .env(CHILD_ENV, &d)
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .spawn()
            .expect("spawn child");

        std::thread::sleep(Duration::from_millis(delay_ms));
        // SIGKILL on unix: no unwinding, no Drop, no flush-on-exit.
        child.kill().expect("kill child");
        let _ = child.wait();

        let n = recovered_rows(&d).await;
        eprintln!("A2 kill@{delay_ms}ms recovered rows = {n}");
        observed.push((delay_ms, n));
    }

    let partial: Vec<(u64, u64)> = observed
        .iter()
        .copied()
        .filter(|(_, n)| *n != 0 && *n != ROWS as u64)
        .collect();

    // Anti-vacuity: a sweep in which the child never got far enough to write
    // anything at all measures process startup, not WAL atomicity.
    let any_landed = observed.iter().any(|(_, n)| *n > 0);

    if !any_landed {
        eprintln!(
            "A2 INCONCLUSIVE: every kill recovered 0 rows, so no arm reached a \
             durable write. This measures spawn latency, not atomicity — widen \
             the delay sweep before reading anything into it."
        );
    } else if partial.is_empty() {
        eprintln!(
            "A2 RESULT: every recovered count is 0 or {ROWS}. No partial batch \
             survived a SIGKILL, so the documented \"write batches are atomic \
             and won't be split\" claim held on every arm. FOLD stands."
        );
    } else {
        eprintln!(
            "A2 RESULT: PARTIAL BATCHES OBSERVED at {partial:?} — a put was \
             split across a crash. This FALSIFIES lance's documented batch \
             atomicity and overturns FOLD to KEEP."
        );
    }

    assert!(
        partial.is_empty(),
        "a MemWAL put was split by a crash: {partial:?}. The FOLD verdict in \
         .claude/temporal/08-memwal-vs-batchwriter.md depends on batch \
         atomicity and must be revised to KEEP."
    );
}
