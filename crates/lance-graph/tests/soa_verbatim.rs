//! Does a 512-byte SoA slab survive a Lance write BYTE-FOR-BYTE, contiguously,
//! at a known offset?
//!
//! This is the load-bearing claim for the disk-sink-in deployment pattern:
//! **if the row column lands verbatim, then `mmap(file)[off .. off + rows*512]`
//! IS the slab** — the process pages in only what it touches, RSS follows the
//! access pattern rather than the dataset size, and (because mmap is
//! page-aligned and `4096 % 64 == 0`) the `&[NodeRow]` cast is always valid.
//!
//! Measured once by hand on the Iceland bake (335,302,144 slab bytes →
//! 335,302,663 file bytes, 100 % identical from offset 0). A single manual run
//! is not a guarantee: Lance's encoder picks a layout per column, so a future
//! version or a different column shape could silently start chunking or
//! compressing. Nothing would fail — the round trip would still return the
//! right bytes — and the mmap premise would be gone.
//!
//! So this test asserts the PHYSICAL layout, not the round trip. The round trip
//! is checked too, but only as the weaker companion: it passes for a chunked or
//! compressed file, which is exactly the case that must fail here.
//!
//! # WHY it lands verbatim — measured, and NOT what was first claimed
//!
//! The first version of this file (and of `examples/soa_to_lance.rs`) said the
//! `lance-encoding:compression = "none"` field metadata was load-bearing. That
//! is **false**, and the falsifier is blunt: removing the key leaves the layout
//! byte-identical, and so does setting it to `"zstd"`. The key is spelled
//! correctly and IS parsed (`lance-encoding-9.0.0` `compression.rs:576`) — it
//! simply cannot reach this column.
//!
//! The real cause is the **stride**:
//!
//! - `is_narrow` (`encodings/logical/primitive.rs:3861`) calls a value narrow
//!   below `MINIBLOCK_MAX_BYTE_LENGTH_PER_VALUE = 256`. At 512 bytes a canonical
//!   node row is NOT narrow, so `prefers_miniblock` is false and the column
//!   takes the **full-zip** path.
//! - Full-zip calls `create_per_value`, whose `DataBlock::FixedWidth(_)` arm
//!   returns `ValueEncoder::default()` **unconditionally** (`compression.rs:753`)
//!   — it computes the merged field params one line earlier and then ignores
//!   them on that branch.
//! - The only branch that honours the metadata,
//!   `build_fixed_width_compressor` (`compression.rs:624`), is on the
//!   mini-block path, which a 512-byte value never reaches.
//!
//! So `NODE_ROW_STRIDE = 512 > 256` is what buys the verbatim run. The metadata
//! line is kept in the writer as a defensive pin — if that threshold ever moved
//! above 512, the mini-block path WOULD read it and WOULD keep the column
//! uncompressed — but it is a backstop, not the cause, and it is labelled that
//! way now rather than credited with work it never did.
//!
//! `the_narrow_column_falsifier` is the paired proof that this file can SEE
//! compression at all: at a 64-byte stride the same write becomes mini-block,
//! the metadata is honoured, and `zstd` makes the byte search fail.
//!
//! # What would make this test go red, and why that is correct
//!
//! - Lance's mini-block threshold rises above `NODE_ROW_STRIDE`
//! - Lance changes its default full-zip fixed-width layout
//! - the row column stops being `FixedSizeBinary(NODE_ROW_STRIDE)`
//!
//! Each of those breaks the deployment pattern while breaking nothing else. A
//! red here means *re-measure and re-decide*, never *loosen the assertion*.

use std::collections::HashMap;
use std::sync::Arc;

use arrow::array::{Array, FixedSizeBinaryArray, RecordBatchIterator};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use lance::dataset::{Dataset, WriteMode, WriteParams};
use lance::io::{ObjectStore, ObjectStoreParams, StorageOptionsAccessor};
use lance_graph_contract::canonical_node::NODE_ROW_STRIDE;
use lance_graph_contract::soa_envelope::ENVELOPE_LAYOUT_VERSION;

/// An environment value, with the wrapping quotes this sandbox's exporter adds
/// stripped. A quoted `"…"` credential authenticates as garbage, and the error
/// it produces points at the credential rather than at the quoting.
fn env(k: &str) -> Option<String> {
    std::env::var(k)
        .ok()
        .map(|v| v.trim().trim_matches('"').trim_matches('\'').to_string())
        .filter(|v| !v.is_empty())
}

/// The `object_store` option map, built from the deployment's own variables.
///
/// `aws_endpoint` ← `AWS_ENDPOINT_URL` is the load-bearing line: `object_store`
/// reads `AWS_ENDPOINT`, which this environment does not set, so its own env
/// discovery would address AWS proper instead of the configured endpoint.
fn s3_options() -> Option<HashMap<String, String>> {
    let mut o = HashMap::new();
    o.insert("aws_access_key_id".into(), env("AWS_ACCESS_KEY_ID")?);
    o.insert(
        "aws_secret_access_key".into(),
        env("AWS_SECRET_ACCESS_KEY")?,
    );
    o.insert("aws_endpoint".into(), env("AWS_ENDPOINT_URL")?);
    o.insert(
        "aws_region".into(),
        env("AWS_DEFAULT_REGION").unwrap_or_else(|| "auto".into()),
    );
    o.insert("aws_virtual_hosted_style_request".into(), "false".into());
    Some(o)
}

/// Lance's own key for per-field compression. Hard-coded rather than imported
/// because `lance-encoding` is not a direct dependency of this crate. Its
/// spelling is verified by `the_narrow_column_falsifier`, which only works if
/// the key actually reaches the encoder.
const COMPRESSION_META_KEY: &str = "lance-encoding:compression";

/// Lance's mini-block cutoff, restated from
/// `lance-encoding-9.0.0/src/encodings/logical/primitive.rs:3861`
/// (`MINIBLOCK_MAX_BYTE_LENGTH_PER_VALUE`). Below this a column is "narrow" and
/// takes the mini-block path, where compression metadata is honoured; at or
/// above it the column takes full-zip, where the metadata is ignored and the
/// values are written verbatim.
const MINIBLOCK_MAX_BYTE_LENGTH_PER_VALUE: usize = 256;

/// Rows that are DISTINGUISHABLE from one another and from any plausible
/// padding, so a shifted, truncated, reordered or partially-written slab cannot
/// pass by coincidence.
///
/// Each row's first 8 bytes are its index as LE `u64`; the remainder is a
/// deterministic byte pattern derived from the index. A run of zeroes, a
/// repeated row, or an off-by-one offset all fail the search.
fn slab_bytes_with_stride(rows: usize, stride: usize) -> Vec<u8> {
    let mut v = vec![0u8; rows * stride];
    for i in 0..rows {
        let base = i * stride;
        v[base..base + 8].copy_from_slice(&(i as u64).to_le_bytes());
        for (j, b) in v[base + 8..base + stride].iter_mut().enumerate() {
            // Never 0, so "all zero" can never be mistaken for real content.
            *b = ((i.wrapping_mul(31).wrapping_add(j)) % 251 + 1) as u8;
        }
    }
    v
}

fn slab_bytes(rows: usize) -> Vec<u8> {
    slab_bytes_with_stride(rows, NODE_ROW_STRIDE)
}

fn schema_for(stride: usize, compression: &str) -> Arc<Schema> {
    let mut field_meta = HashMap::new();
    field_meta.insert(COMPRESSION_META_KEY.to_string(), compression.to_string());

    let mut table_meta = HashMap::new();
    // The values are IMPORTED, never restated — a second spelling of the
    // stride is a second source of truth, and the one that drifts is always
    // the copy.
    table_meta.insert(
        "soa:envelope_layout_version".into(),
        ENVELOPE_LAYOUT_VERSION.to_string(),
    );
    table_meta.insert("soa:row_stride".into(), NODE_ROW_STRIDE.to_string());
    table_meta.insert("soa:endianness".into(), "le".into());

    Arc::new(
        Schema::new(vec![Field::new(
            "row",
            DataType::FixedSizeBinary(stride as i32),
            false,
        )
        .with_metadata(field_meta)])
        .with_metadata(table_meta),
    )
}

async fn write_slab(dir: &std::path::Path, bytes: Vec<u8>) -> Dataset {
    write_with(dir, bytes, NODE_ROW_STRIDE, "none").await
}

async fn write_with(
    dir: &std::path::Path,
    bytes: Vec<u8>,
    stride: usize,
    compression: &str,
) -> Dataset {
    let schema = schema_for(stride, compression);
    let array =
        FixedSizeBinaryArray::new(stride as i32, arrow::buffer::Buffer::from_vec(bytes), None);
    let batch = RecordBatch::try_new(schema.clone(), vec![Arc::new(array)]).expect("batch");
    let reader = RecordBatchIterator::new(vec![Ok(batch)].into_iter(), schema);
    Dataset::write(
        reader,
        dir.to_str().expect("utf8"),
        Some(WriteParams {
            mode: WriteMode::Create,
            ..Default::default()
        }),
    )
    .await
    .expect("write")
}

/// The single `.lance` data file under a dataset directory.
fn sole_data_file(dir: &std::path::Path) -> std::path::PathBuf {
    let data = dir.join("data");
    let mut files: Vec<_> = std::fs::read_dir(&data)
        .expect("data dir")
        .flatten()
        .map(|e| e.path())
        .filter(|p| p.extension().is_some_and(|x| x == "lance"))
        .collect();
    assert_eq!(
        files.len(),
        1,
        "expected exactly one data file; a split changes what an mmap offset means"
    );
    files.pop().expect("one")
}

#[tokio::test]
async fn a_slab_is_written_verbatim_and_contiguously() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let dir = tmp.path().join("verbatim.lance");
    const ROWS: usize = 2048;

    let bytes = slab_bytes(ROWS);
    write_slab(&dir, bytes.clone()).await;

    let file = std::fs::read(sole_data_file(&dir)).expect("read data file");

    // (1) The slab appears in the file, exactly once, as one unbroken run.
    let off = file
        .windows(bytes.len())
        .position(|w| w == bytes.as_slice())
        .unwrap_or_else(|| {
            panic!(
                "the {} slab bytes are NOT contiguous in the {}-byte data file — \
                 Lance chunked or transformed the column, so the mmap premise is gone",
                bytes.len(),
                file.len()
            )
        });

    // (2) Every row is at its computed address. This is what an mmap reader
    //     does, and it is a stronger claim than "the bytes are in there
    //     somewhere": a reversed or rotated slab would pass (1) only if it
    //     matched exactly, but this states the addressing rule the reader uses.
    for i in [0usize, 1, ROWS / 2, ROWS - 1] {
        let at = off + i * NODE_ROW_STRIDE;
        let row = &file[at..at + NODE_ROW_STRIDE];
        assert_eq!(
            u64::from_le_bytes(row[..8].try_into().unwrap()),
            i as u64,
            "row {i} is not at offset {at}"
        );
        assert_eq!(row, &bytes[i * NODE_ROW_STRIDE..(i + 1) * NODE_ROW_STRIDE]);
    }

    // (3) The file is the slab plus a bounded footer — not the slab plus a
    //     second encoded copy of it. Without this, a file that stored the data
    //     twice (raw + compressed) would satisfy (1) and (2) while doubling
    //     what an mmap deployment must fetch.
    let overhead = file.len() - bytes.len();
    assert!(
        overhead < 64 * 1024,
        "data file carries {overhead} bytes beyond the slab; expected a small footer"
    );
}

/// The **anti-vacuity companion**: prove the search in the test above can fail.
///
/// Without this, `a_slab_is_written_verbatim_and_contiguously` might be passing
/// because `windows().position()` finds *something* rather than because Lance
/// wrote the slab intact. Here the needle is genuinely absent, and the same
/// search must come back `None`.
#[tokio::test]
async fn the_verbatim_search_can_fail() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let dir = tmp.path().join("absent.lance");
    const ROWS: usize = 256;

    write_slab(&dir, slab_bytes(ROWS)).await;
    let file = std::fs::read(sole_data_file(&dir)).expect("read data file");

    // A slab the writer never saw: same shape, different content.
    let mut other = slab_bytes(ROWS);
    for b in &mut other {
        *b = b.wrapping_add(7).max(1);
    }
    assert!(
        file.windows(other.len()).position(|w| w == other).is_none(),
        "a slab that was never written must NOT be found — otherwise the \
         verbatim test proves nothing"
    );
}

/// The round trip, kept deliberately as the WEAKER check.
///
/// It passes for a chunked or compressed file, which is precisely the case the
/// physical-layout test must reject. It is here so that a red physical test can
/// be diagnosed: round trip green + layout red means "Lance changed its
/// encoding", while both red means "the write itself broke".
#[tokio::test]
async fn the_round_trip_is_lossless_but_proves_less() {
    use futures::TryStreamExt;

    let tmp = tempfile::tempdir().expect("tempdir");
    let dir = tmp.path().join("roundtrip.lance");
    const ROWS: usize = 512;

    let bytes = slab_bytes(ROWS);
    let ds = write_slab(&dir, bytes.clone()).await;

    let mut back = Vec::with_capacity(bytes.len());
    let mut stream = ds.scan().try_into_stream().await.expect("scan");
    while let Some(b) = stream.try_next().await.expect("batch") {
        let col = b
            .column_by_name("row")
            .expect("row column")
            .as_any()
            .downcast_ref::<FixedSizeBinaryArray>()
            .expect("FixedSizeBinary");
        for i in 0..col.len() {
            back.extend_from_slice(col.value(i));
        }
    }
    assert_eq!(back, bytes, "the round trip must be lossless");
}

/// **The S3 arm — the same assertion, 1:1, against the store the deployment
/// actually reads from.**
///
/// The local arms above prove the encoder's behaviour. They do NOT prove that
/// the object store round-trips it: a remote write goes through a different
/// path (multipart upload, a different `ObjectStore` impl, a different commit
/// handler), and "it worked locally" has never been evidence about S3.
///
/// **No new environment key is invented for this.** The arm runs off exactly
/// the variables the deployment already sets — `AWS_S3_BUCKET_NAME`,
/// `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_ENDPOINT_URL`,
/// `AWS_DEFAULT_REGION` — so a machine that can serve the maps can also run
/// this test, with no extra setup. KEY NAMES are universal and live here;
/// VALUES are deployment configuration and never enter the repository.
///
/// `AWS_ENDPOINT_URL` is mapped explicitly because `object_store` itself reads
/// `AWS_ENDPOINT`; relying on its own env discovery silently addresses the
/// wrong host.
///
/// The dataset is written under `TEST_PREFIX` with a unique suffix and removed
/// afterwards, so repeated and concurrent runs neither collide nor accumulate.
#[tokio::test]
async fn a_slab_is_written_verbatim_to_s3_too() {
    /// Scratch space, kept out of any data prefix so a failed run can never be
    /// mistaken for a bake.
    const TEST_PREFIX: &str = "_tests/soa-verbatim";

    let Some(bucket) = env("AWS_S3_BUCKET_NAME") else {
        eprintln!(
            "SKIP a_slab_is_written_verbatim_to_s3_too: AWS_S3_BUCKET_NAME unset — \
             set the same AWS_* variables the deployment uses to run the remote arm"
        );
        return;
    };
    let Some(opts) = s3_options() else {
        eprintln!("SKIP a_slab_is_written_verbatim_to_s3_too: AWS_* credentials incomplete");
        return;
    };

    const ROWS: usize = 2048;
    let bytes = slab_bytes(ROWS);
    let stamp = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .expect("clock")
        .as_nanos();
    let dest = format!("s3://{bucket}/{TEST_PREFIX}-{stamp}.lance");

    let schema = schema_for(NODE_ROW_STRIDE, "none");
    let array = FixedSizeBinaryArray::new(
        NODE_ROW_STRIDE as i32,
        arrow::buffer::Buffer::from_vec(bytes.clone()),
        None,
    );
    let batch = RecordBatch::try_new(schema.clone(), vec![Arc::new(array)]).expect("batch");
    Dataset::write(
        RecordBatchIterator::new(vec![Ok(batch)].into_iter(), schema),
        &dest,
        Some(WriteParams {
            mode: WriteMode::Create,
            store_params: Some(ObjectStoreParams {
                storage_options_accessor: Some(Arc::new(
                    StorageOptionsAccessor::with_static_options(opts.clone()),
                )),
                ..Default::default()
            }),
            ..Default::default()
        }),
    )
    .await
    .expect("write to s3");

    let (store, root) = ObjectStore::from_uri_and_params(
        Default::default(),
        &dest,
        &ObjectStoreParams {
            storage_options_accessor: Some(Arc::new(StorageOptionsAccessor::with_static_options(
                opts,
            ))),
            ..Default::default()
        },
    )
    .await
    .expect("object store");

    // The data objects, read back through the object store — the same bytes a
    // hydrating deployment would pull down.
    let listing = store
        .list_with_delimiter(Some(&root.clone().join("data")))
        .await
        .expect("list data/");
    let data_objects: Vec<_> = listing
        .objects
        .into_iter()
        .filter(|o| o.location.as_ref().ends_with(".lance"))
        .collect();
    assert_eq!(
        data_objects.len(),
        1,
        "expected exactly one remote data object; a split changes what an offset means"
    );

    let file = store
        .read_one_all(&data_objects[0].location)
        .await
        .expect("read remote data object");

    let off = file
        .windows(bytes.len())
        .position(|w| w == bytes.as_slice())
        .unwrap_or_else(|| {
            panic!(
                "the {} slab bytes are NOT contiguous in the {}-byte REMOTE data object — \
                 the object store path transformed the column, so hydrating to disk and \
                 mmapping it would not serve the slab",
                bytes.len(),
                file.len()
            )
        });
    for i in [0usize, 1, ROWS / 2, ROWS - 1] {
        let at = off + i * NODE_ROW_STRIDE;
        assert_eq!(
            &file[at..at + NODE_ROW_STRIDE],
            &bytes[i * NODE_ROW_STRIDE..(i + 1) * NODE_ROW_STRIDE],
            "remote row {i} is not at offset {at}"
        );
    }
    assert!(
        file.len() - bytes.len() < 64 * 1024,
        "remote object carries {} bytes beyond the slab",
        file.len() - bytes.len()
    );

    store.remove_dir_all(root).await.expect("clean up");
}

/// **The sensitivity proof.** Can this file's byte search see a compressed
/// column at all?
///
/// It has to be asked, because at `NODE_ROW_STRIDE` the compression metadata is
/// provably inert (`"none"`, `"zstd"` and *absent* all produce the same file —
/// measured). If the search were simply blind to compression, every assertion
/// above would be green for the wrong reason.
///
/// So: drop below `MINIBLOCK_MAX_BYTE_LENGTH_PER_VALUE`, where the mini-block
/// path DOES honour the metadata, and check both directions on the same shape.
/// `"none"` must land verbatim; `"zstd"` must not. If either half fails, this
/// file's central claim is unproven — a green `"zstd"` arm would mean the search
/// cannot detect compression, and a red `"none"` arm would mean the narrow path
/// is not a valid control.
///
/// This is also what verifies `COMPRESSION_META_KEY`'s spelling: a typo'd key
/// would be ignored on both arms and the `"zstd"` half would go green.
#[tokio::test]
async fn the_narrow_column_falsifier() {
    const NARROW: usize = 64;
    const ROWS: usize = 4096;
    // The control must be narrow and the real row must not be — otherwise this
    // proves nothing about either. A compile-time check: if either constant
    // above ever moves, the build itself fails rather than a possibly-skipped
    // test.
    const _: () = assert!(
        NARROW < MINIBLOCK_MAX_BYTE_LENGTH_PER_VALUE
            && NODE_ROW_STRIDE >= MINIBLOCK_MAX_BYTE_LENGTH_PER_VALUE
    );

    let tmp = tempfile::tempdir().expect("tempdir");
    let bytes = slab_bytes_with_stride(ROWS, NARROW);
    let sample = [0usize, 1, ROWS / 2, ROWS - 1];

    // PER-ROW needles, not the whole slab. The narrow path is mini-block, which
    // is CHUNKED by construction — a whole-slab search fails there even
    // uncompressed, for a reason that has nothing to do with compression. (That
    // is itself worth knowing: it is why the main test's contiguity assertion
    // is not free.) A single row's bytes survive chunking; they do not survive
    // compression.
    let mut rows_found = Vec::new();
    for compression in ["none", "zstd"] {
        let dir = tmp.path().join(format!("narrow-{compression}.lance"));
        write_with(&dir, bytes.clone(), NARROW, compression).await;
        let file = std::fs::read(sole_data_file(&dir)).expect("read data file");
        rows_found.push(
            sample
                .iter()
                .filter(|&&i| {
                    let needle = &bytes[i * NARROW..(i + 1) * NARROW];
                    file.windows(NARROW).any(|w| w == needle)
                })
                .count(),
        );
    }

    assert_eq!(
        rows_found[0],
        sample.len(),
        "a NARROW column with compression=none must still carry its rows \
         verbatim — without this the zstd arm below proves nothing, because the \
         difference could be chunking rather than the metadata"
    );
    assert_eq!(
        rows_found[1],
        0,
        "a NARROW column with compression=zstd still carries {} of {} rows \
         verbatim — either the metadata key is misspelled, or this file's byte \
         search cannot detect compression at all, in which case every assertion \
         above is green for the wrong reason",
        rows_found[1],
        sample.len()
    );
}

/// The header contract is persisted and readable — and carries the COMPILED
/// constants, not a copy of them.
///
/// A reader verifies this before casting anything; a table whose stride
/// disagrees with the reader's `NODE_ROW_STRIDE` must be refused, not cast.
#[tokio::test]
async fn the_soa_contract_survives_in_the_table_header() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let dir = tmp.path().join("header.lance");
    write_slab(&dir, slab_bytes(16)).await;

    let ds = Dataset::open(dir.to_str().expect("utf8"))
        .await
        .expect("open");
    let meta = &ds.schema().metadata;

    assert_eq!(
        meta.get("soa:row_stride").map(String::as_str),
        Some(NODE_ROW_STRIDE.to_string().as_str()),
        "the header must carry the stride a reader will verify against"
    );
    assert_eq!(
        meta.get("soa:envelope_layout_version").map(String::as_str),
        Some(ENVELOPE_LAYOUT_VERSION.to_string().as_str())
    );
    assert_eq!(meta.get("soa:endianness").map(String::as_str), Some("le"));
}
