//! `hydration_probe` — closes the **§4 verification gate** of
//! `.claude/plans/idle-flush-dataset-eviction-v1.md`, and measures the
//! hydration cost that plan's economics rest on.
//!
//! ```text
//! cargo run -p lance-graph --release --example hydration_probe
//! ```
//!
//! # What the plan owes, and what this measures
//!
//! The plan is a PROPOSAL with one named blocker (§9.1):
//!
//! > *"The §4 verification gate. Cheap local version read — **assumed,
//! > unchecked**. Closing this is the first task; if it fails, the plan needs a
//! > different dirty-detector and this document is wrong rather than
//! > incomplete."*
//!
//! Four columns, each answering one thing the plan states without evidence:
//!
//! 1. **§4 gate — is a version read cheap?** The dirty detector is
//!    `current_local_version != version_at_hydration`. That is only viable if
//!    reading the current version is much cheaper than opening the dataset. This
//!    times both, at several sizes, **locally** — the sweep runs against local
//!    copies, so a local measurement is the one that decides it.
//! 2. **§1 cost model — what does a hydration actually cost?** The plan grades
//!    its own economics as incomplete and names the omission: *request count*,
//!    since "a dataset is a multi-file directory". This counts the files and
//!    measures the wall time against the **real** endpoint.
//! 3. **§0 / §5 — the ~1.4 s rehydration figure.** Graded there as a *single
//!    observation, provider- and region-dependent, not re-run*. This re-runs it
//!    at three sizes so the shape is visible rather than one point.
//! 4. **T10 — is the round trip lossless?** Flush → rehydrate → read must equal
//!    the pre-flush read. The cheapest of the plan's acceptance criteria to
//!    settle, and the one whose failure would void the rest.
//!
//! # What this does NOT do
//!
//! It does not implement eviction, and it is **not** an authorisation to. It
//! measures four inputs the plan needs; the policy stays a proposal. Nothing
//! here evicts anything, and the only bytes it removes are the ones it wrote.
//!
//! # Error direction, stated once
//!
//! Every timing here **flatters the fast path**: the version read runs after
//! the dataset has been opened once, so the OS page cache and the object
//! store's own connection pool are warm. A cold version read can only be
//! slower. The gate therefore passes only if the *warm* ratio is already
//! decisive — a marginal warm result is a failed gate, not a close call.
//!
//! Hydration is measured cold in the only sense available in-process: a fresh
//! local directory per run. The remote side's caches are not ours to clear, so
//! a repeated run against the same key may report faster than a first-ever
//! fetch. Treat the numbers as a floor on cost, never a ceiling.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;

use arrow::array::{Float32Array, Int64Array, RecordBatch, RecordBatchIterator};
use arrow::datatypes::{DataType, Field, Schema};
use lance::dataset::{Dataset, WriteMode, WriteParams};

/// Row counts to probe. Chosen to bracket the plan's "tens of MB" reference
/// point from both sides, so the reported ~1.4 s can be placed on a curve
/// rather than taken as a constant.
const SIZES: &[usize] = &[10_000, 200_000, 1_000_000];

/// Columns per row: one `i64` + eight `f32` = 40 B of payload, so
/// 1,000,000 rows is ~40 MB before encoding — the plan's own scale.
const FLOAT_COLS: usize = 8;

fn env(k: &str) -> Option<String> {
    // The same strip the workspace's other S3 callers apply: these variables
    // arrive wrapped in literal quotes in this environment, and an unstripped
    // value fails authentication in a way that looks like a credential error
    // rather than a parsing one.
    std::env::var(k)
        .ok()
        .map(|v| v.trim().trim_matches('"').trim_matches('\'').to_string())
        .filter(|v| !v.is_empty())
}

/// The storage options for the configured endpoint.
///
/// Built explicitly rather than leaning on `from_env`, because `object_store`
/// reads **`AWS_ENDPOINT`** while this environment sets **`AWS_ENDPOINT_URL`**.
/// Relying on the implicit path would silently address AWS proper instead of
/// the configured endpoint — a failure that reads as a permissions problem.
fn storage_options() -> Option<HashMap<String, String>> {
    let mut o = HashMap::new();
    o.insert("aws_access_key_id".into(), env("AWS_ACCESS_KEY_ID")?);
    o.insert("aws_secret_access_key".into(), env("AWS_SECRET_ACCESS_KEY")?);
    o.insert("aws_endpoint".into(), env("AWS_ENDPOINT_URL")?);
    o.insert(
        "aws_region".into(),
        env("AWS_DEFAULT_REGION").unwrap_or_else(|| "auto".into()),
    );
    // Path-style keeps the request off a per-bucket virtual host, which is what
    // the workspace's other S3 caller already assumes for this endpoint.
    o.insert("aws_virtual_hosted_style_request".into(), "false".into());
    Some(o)
}

fn schema() -> Arc<Schema> {
    let mut fields = vec![Field::new("id", DataType::Int64, false)];
    for i in 0..FLOAT_COLS {
        fields.push(Field::new(format!("f{i}"), DataType::Float32, false));
    }
    Arc::new(Schema::new(fields))
}

fn batch(schema: &Arc<Schema>, rows: usize) -> RecordBatch {
    let ids: Int64Array = (0..rows as i64).collect::<Vec<_>>().into();
    let mut cols: Vec<arrow::array::ArrayRef> = vec![Arc::new(ids)];
    for c in 0..FLOAT_COLS {
        // Deterministic, and NOT constant per column — a constant column would
        // compress to nothing and make the transfer measurement meaningless.
        let v: Float32Array = (0..rows)
            .map(|i| ((i * 2_654_435_761usize).wrapping_add(c) % 100_003) as f32 * 0.001)
            .collect::<Vec<_>>()
            .into();
        cols.push(Arc::new(v));
    }
    RecordBatch::try_new(schema.clone(), cols).expect("batch")
}

/// Total bytes and file count under a directory — the request-count proxy the
/// plan's §1 says the first-draft cost model omitted.
fn dir_stats(p: &std::path::Path) -> (u64, usize) {
    let (mut bytes, mut files) = (0u64, 0usize);
    let mut stack = vec![p.to_path_buf()];
    while let Some(d) = stack.pop() {
        let Ok(rd) = std::fs::read_dir(&d) else {
            continue;
        };
        for e in rd.flatten() {
            let Ok(ft) = e.file_type() else { continue };
            if ft.is_dir() {
                stack.push(e.path());
            } else if let Ok(m) = e.metadata() {
                bytes += m.len();
                files += 1;
            }
        }
    }
    (bytes, files)
}

/// Read every row's `id` column, summed — a full scan, so a truncated or
/// partially-hydrated dataset cannot pass as equal.
async fn checksum(ds: &Dataset) -> (u64, i64) {
    use futures::TryStreamExt;
    let mut stream = ds.scan().try_into_stream().await.expect("scan");
    let (mut rows, mut sum) = (0u64, 0i64);
    while let Some(b) = stream.try_next().await.expect("next batch") {
        rows += b.num_rows() as u64;
        let ids = b
            .column_by_name("id")
            .expect("id column")
            .as_any()
            .downcast_ref::<Int64Array>()
            .expect("id is i64");
        for i in 0..ids.len() {
            sum = sum.wrapping_add(ids.value(i));
        }
    }
    (rows, sum)
}

#[tokio::main(flavor = "multi_thread")]
async fn main() {
    let tmp = std::env::temp_dir().join(format!("hydration_probe_{}", std::process::id()));
    std::fs::create_dir_all(&tmp).expect("scratch dir");
    let schema = schema();

    println!("scratch: {}", tmp.display());
    println!();
    println!("── (1) §4 GATE — is a version read cheap enough to run per sweep candidate?");
    println!(
        "{:>10}  {:>8}  {:>7}  {:>12}  {:>14}  {:>10}",
        "rows", "MB", "files", "open (ms)", "version (ms)", "ratio"
    );

    let mut gate_rows: Vec<(usize, f64, f64, u64, usize)> = Vec::new();
    for &rows in SIZES {
        let path = tmp.join(format!("local_{rows}.lance"));
        let b = batch(&schema, rows);
        let reader = RecordBatchIterator::new(vec![Ok(b)].into_iter(), schema.clone());
        Dataset::write(
            reader,
            path.to_str().unwrap(),
            Some(WriteParams {
                mode: WriteMode::Create,
                ..Default::default()
            }),
        )
        .await
        .expect("write local");

        let (bytes, files) = dir_stats(&path);

        // Warm both paths once so neither is charged for first-touch cost; the
        // doc comment states why this flatters the fast path.
        let warm = Dataset::open(path.to_str().unwrap()).await.expect("open");
        let _ = warm.latest_version_id().await.expect("version");

        const N: u32 = 10;
        let t = Instant::now();
        for _ in 0..N {
            let ds = Dataset::open(path.to_str().unwrap()).await.expect("open");
            std::hint::black_box(ds.version().version);
        }
        let open_ms = t.elapsed().as_secs_f64() * 1e3 / f64::from(N);

        let t = Instant::now();
        for _ in 0..N {
            let v = warm.latest_version_id().await.expect("version");
            std::hint::black_box(v);
        }
        let ver_ms = t.elapsed().as_secs_f64() * 1e3 / f64::from(N);

        println!(
            "{rows:>10}  {:>8.1}  {files:>7}  {open_ms:>12.3}  {ver_ms:>14.3}  {:>9.1}x",
            bytes as f64 / 1e6,
            open_ms / ver_ms.max(1e-9)
        );
        gate_rows.push((rows, open_ms, ver_ms, bytes, files));
    }

    println!();
    println!("  A version read is the sweep's PER-CANDIDATE cost; an open is what it avoids.");
    println!("  The gate passes only if the ratio is decisive while WARM (see module doc).");

    // ── (2)+(3) hydration against the configured endpoint ──
    println!();
    let Some(opts) = storage_options() else {
        println!("── (2) HYDRATION — SKIPPED: no S3 credentials in the environment.");
        println!("  Columns 1 and 4 above/below are local and still valid; the cost model");
        println!("  in the plan's §1 stays UNMEASURED. This is a skip, not a pass.");
        let _ = std::fs::remove_dir_all(&tmp);
        return;
    };
    let bucket = env("AWS_S3_BUCKET_NAME").expect("bucket");
    let prefix = format!("OSM/_hydration_probe_{}", std::process::id());

    println!("── (2)+(3) HYDRATION — the real endpoint, {} sizes", SIZES.len());
    println!(
        "{:>10}  {:>8}  {:>7}  {:>12}  {:>13}  {:>12}  {:>10}",
        "rows", "MB", "files", "upload (s)", "hydrate (s)", "MB/s", "roundtrip"
    );

    let mut any = false;
    for &(rows, _, _, bytes, files) in &gate_rows {
        let local = tmp.join(format!("local_{rows}.lance"));
        let remote = format!("s3://{bucket}/{prefix}/d_{rows}.lance");

        // Read the local truth BEFORE anything remote happens, so the
        // comparison is against the dataset as written, not as re-read.
        let before = {
            let ds = Dataset::open(local.to_str().unwrap()).await.expect("open");
            checksum(&ds).await
        };

        let b = batch(&schema, rows);
        let reader = RecordBatchIterator::new(vec![Ok(b)].into_iter(), schema.clone());
        let t = Instant::now();
        let wrote = Dataset::write(
            reader,
            &remote,
            Some(WriteParams {
                mode: WriteMode::Create,
                store_params: Some(lance::io::ObjectStoreParams {
                    storage_options: Some(opts.clone()),
                    ..Default::default()
                }),
                ..Default::default()
            }),
        )
        .await;
        let up_s = t.elapsed().as_secs_f64();
        if let Err(e) = wrote {
            println!("{rows:>10}  upload FAILED: {e}");
            println!("  Reporting the failure rather than the skip: the endpoint was configured,");
            println!("  so this is a real negative result for the hydration column.");
            continue;
        }
        any = true;

        // Hydrate into a FRESH directory — the `absent -> hydrated` edge.
        let hydrated = tmp.join(format!("hydrated_{rows}.lance"));
        let t = Instant::now();
        let remote_ds = lance::dataset::DatasetBuilder::from_uri(&remote)
            .with_storage_options(opts.clone())
            .load()
            .await
            .expect("open remote");
        let mut stream = {
            use futures::TryStreamExt;
            remote_ds.scan().try_into_stream().await.expect("scan remote")
        };
        let mut batches = Vec::new();
        {
            use futures::TryStreamExt;
            while let Some(b) = stream.try_next().await.expect("remote batch") {
                batches.push(Ok(b));
            }
        }
        let reader = RecordBatchIterator::new(batches.into_iter(), schema.clone());
        Dataset::write(
            reader,
            hydrated.to_str().unwrap(),
            Some(WriteParams {
                mode: WriteMode::Create,
                ..Default::default()
            }),
        )
        .await
        .expect("write hydrated");
        let hy_s = t.elapsed().as_secs_f64();

        // T10 — flush -> rehydrate -> read equals the pre-flush read.
        let after = {
            let ds = Dataset::open(hydrated.to_str().unwrap())
                .await
                .expect("open hydrated");
            checksum(&ds).await
        };

        println!(
            "{rows:>10}  {:>8.1}  {files:>7}  {up_s:>12.2}  {hy_s:>13.2}  {:>12.1}  {:>10}",
            bytes as f64 / 1e6,
            (bytes as f64 / 1e6) / hy_s.max(1e-9),
            if before == after { "EQUAL" } else { "DIFFERS" }
        );
        if before != after {
            println!("    T10 FAILED: {before:?} != {after:?} — the round trip is NOT lossless.");
        }
    }

    if any {
        println!();
        println!("  (3) The plan's ~1.4 s is ONE observation; the rows above are this endpoint");
        println!("      on this day. Read the MB/s column, not the seconds — the seconds are");
        println!("      only comparable at the same size.");
        println!("  (4) T10 is the cheapest acceptance criterion in the plan and the one whose");
        println!("      failure would void the rest. EQUAL is a full-scan id checksum, not a");
        println!("      row count — a truncated hydration cannot pass it.");
    }

    // Remove only what this probe wrote.
    let _ = std::fs::remove_dir_all(&tmp);
    println!();
    println!("local scratch removed: {}", tmp.display());
    println!("REMOTE OBJECTS LEFT IN PLACE at s3://{bucket}/{prefix}/ — delete when done.");
}
