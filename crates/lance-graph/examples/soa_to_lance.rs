//! `soa_to_lance` — the ONE-TIME write-back of a `.soa` bake into Lance's own
//! format, with the 512-byte SoA contract persisted in the table header.
//!
//! ```text
//! soa_to_lance <slab.soa> <uri> <table> <classid-hex> <slab-digest-hex>
//! ```
//!
//! `uri` may be a local directory or `s3://bucket/prefix` — the same call
//! either way; S3 credentials come from `AWS_*` env vars. There is no export
//! step: writing to the object store IS this write.
//!
//! # The header carries the contract, once
//!
//! The operator's requirement, verbatim: *"nur das 512-Byte-SoA-Schema muss
//! dann in den table headers vermutlich einmalig sauber persistiert werden."*
//! Done here as Arrow schema metadata (which Lance persists in its manifest),
//! with the load-bearing values IMPORTED from `lance-graph-contract` rather
//! than restated — a restated constant is a second source of truth that
//! drifts:
//!
//! | key | value | source |
//! |---|---|---|
//! | `soa:envelope_layout_version` | `2` | `soa_envelope::ENVELOPE_LAYOUT_VERSION` |
//! | `soa:row_stride` | `512` | `canonical_node::NODE_ROW_STRIDE` |
//! | `soa:row_carving` | `key:0..16\|edges:16..32\|value:32..512` | canon, locked 2026-06-13 |
//! | `soa:endianness` | `le` | the LE contract |
//! | `soa:classid` | per bake (arg) | the bake's own report |
//! | `soa:slab_digest` | per bake (arg) | the bake's own report — pairs the table with its `.books` sidecar |
//! | `soa:source` | filename + row count | provenance |
//!
//! A reader verifies `envelope_layout_version` against its COMPILED contract
//! before casting anything — the same shape as `SoaEnvelope::verify_layout`.
//! This binary performs that verification itself, by re-opening what it wrote.
//!
//! # The row column is written UNCOMPRESSED, deliberately
//!
//! Field metadata `lance-encoding:compression = "none"` (the key is
//! `lance_encoding::constants::COMPRESSION_META_KEY`; the value `"none"` is a
//! documented passthrough). The row is a content-blind register — compression
//! would buy little — and an uncompressed fixed-width column is written as a
//! verbatim byte run inside the `.lance` data file, which is what makes the
//! mmap serving path (pattern b) possible at all. **Claimed, then verified**:
//! for a local `uri`, this binary scans the written data file for the slab's
//! own bytes and reports the offset and whether a multi-row run is contiguous
//! (P-CACHE-4 in `.claude/knowledge/lance-cache-surface.md`).
//!
//! # Zero-copy import
//!
//! `Buffer::from_vec` ADOPTS the allocation that read the file;
//! `FixedSizeBinaryArray::try_from_iter` would copy it chunk by chunk
//! (measured on this branch, commit a27b06a's history). The adoption is
//! asserted at runtime by pointer identity, not trusted.

use std::collections::HashMap;
use std::sync::Arc;

use arrow::array::{Array, FixedSizeBinaryArray};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use lance::dataset::{Dataset, WriteMode, WriteParams};
use lance::io::{ObjectStoreParams, StorageOptionsAccessor};
use lance_graph_contract::canonical_node::NODE_ROW_STRIDE;
use lance_graph_contract::soa_envelope::ENVELOPE_LAYOUT_VERSION;

/// Schema-metadata keys. Namespaced `soa:` so they cannot collide with
/// Lance's own (`lance-encoding:*`) or Arrow's.
const K_LAYOUT: &str = "soa:envelope_layout_version";
const K_STRIDE: &str = "soa:row_stride";
const K_CARVING: &str = "soa:row_carving";
const K_ENDIAN: &str = "soa:endianness";
const K_CLASSID: &str = "soa:classid";
const K_DIGEST: &str = "soa:slab_digest";
const K_SOURCE: &str = "soa:source";

const ROW_COLUMN: &str = "row";

fn env(k: &str) -> Option<String> {
    std::env::var(k)
        .ok()
        .map(|v| v.trim().trim_matches('"').trim_matches('\'').to_string())
        .filter(|v| !v.is_empty())
}

fn s3_options() -> Option<HashMap<String, String>> {
    let mut o = HashMap::new();
    o.insert("aws_access_key_id".into(), env("AWS_ACCESS_KEY_ID")?);
    o.insert("aws_secret_access_key".into(), env("AWS_SECRET_ACCESS_KEY")?);
    o.insert("aws_endpoint".into(), env("AWS_ENDPOINT_URL")?);
    o.insert(
        "aws_region".into(),
        env("AWS_DEFAULT_REGION").unwrap_or_else(|| "auto".into()),
    );
    o.insert("aws_virtual_hosted_style_request".into(), "false".into());
    Some(o)
}

fn store_params() -> Option<ObjectStoreParams> {
    Some(ObjectStoreParams {
        storage_options_accessor: Some(Arc::new(StorageOptionsAccessor::with_static_options(
            s3_options()?,
        ))),
        ..Default::default()
    })
}

#[tokio::main(flavor = "multi_thread")]
async fn main() {
    let a: Vec<String> = std::env::args().collect();
    if a.len() != 6 {
        eprintln!("usage: soa_to_lance <slab.soa> <uri> <table> <classid-hex> <slab-digest-hex>");
        std::process::exit(2);
    }
    let (slab_path, uri, table, classid, digest) = (&a[1], &a[2], &a[3], &a[4], &a[5]);
    let is_remote = uri.contains("://");

    // ── read the slab; this Vec's allocation is the one Lance will serialize ──
    let bytes = std::fs::read(slab_path).expect("read slab");
    assert!(
        !bytes.is_empty() && bytes.len().is_multiple_of(NODE_ROW_STRIDE),
        "slab is {} bytes — not a whole number of {NODE_ROW_STRIDE}-byte rows; refusing a truncated bake",
        bytes.len()
    );
    let rows = bytes.len() / NODE_ROW_STRIDE;
    // First 4 KiB kept aside for the P-CACHE-4 contiguity scan AFTER the Vec
    // moves into arrow.
    let probe: Vec<u8> = bytes[..4096.min(bytes.len())].to_vec();
    let before = bytes.as_ptr();

    // ── schema: the contract, persisted once, imported not restated ──
    let field = Field::new(
        ROW_COLUMN,
        DataType::FixedSizeBinary(NODE_ROW_STRIDE as i32),
        false,
    )
    .with_metadata(HashMap::from([(
        // lance_encoding::constants::COMPRESSION_META_KEY — restated here only
        // because lance-encoding is not a direct dep of this crate; the value
        // "none" is the documented passthrough.
        "lance-encoding:compression".to_string(),
        "none".to_string(),
    )]));
    let schema_meta = HashMap::from([
        (K_LAYOUT.to_string(), ENVELOPE_LAYOUT_VERSION.to_string()),
        (K_STRIDE.to_string(), NODE_ROW_STRIDE.to_string()),
        (
            K_CARVING.to_string(),
            "key:0..16|edges:16..32|value:32..512".to_string(),
        ),
        (K_ENDIAN.to_string(), "le".to_string()),
        (K_CLASSID.to_string(), classid.clone()),
        (K_DIGEST.to_string(), digest.clone()),
        (K_SOURCE.to_string(), format!("{slab_path} rows={rows}")),
    ]);
    let schema = Arc::new(Schema::new_with_metadata(vec![field], schema_meta));

    // ── zero-copy import: the Vec's allocation becomes arrow's buffer ──
    let array = FixedSizeBinaryArray::new(
        NODE_ROW_STRIDE as i32,
        arrow::buffer::Buffer::from_vec(bytes),
        None,
    );
    assert_eq!(
        array.value_data().as_ptr(),
        before,
        "Buffer::from_vec must ADOPT the allocation — a moved address means a copy crept in"
    );
    let batch = RecordBatch::try_new(schema.clone(), vec![Arc::new(array)]).expect("batch");

    // ── the write: local and S3 are the same call ──
    let dest = format!("{}/{}.lance", uri.trim_end_matches('/'), table);
    let params = WriteParams {
        mode: WriteMode::Create,
        store_params: if is_remote { store_params() } else { None },
        ..Default::default()
    };
    let t = std::time::Instant::now();
    Dataset::write(
        arrow::array::RecordBatchIterator::new(vec![Ok(batch)].into_iter(), schema),
        &dest,
        Some(params),
    )
    .await
    .expect("write dataset");
    let wrote_s = t.elapsed().as_secs_f64();

    // ── verify: re-open what was written; the header must answer for itself ──
    let ds = {
        let mut b = lance::dataset::builder::DatasetBuilder::from_uri(&dest);
        if is_remote {
            b = b.with_storage_options(s3_options().expect("s3 opts"));
        }
        b.load().await.expect("re-open")
    };
    assert_eq!(ds.count_rows(None).await.expect("count"), rows);
    let meta = &ds.schema().metadata;
    let read_layout: u8 = meta
        .get(K_LAYOUT)
        .expect("header must carry soa:envelope_layout_version")
        .parse()
        .expect("numeric");
    assert_eq!(
        read_layout, ENVELOPE_LAYOUT_VERSION,
        "the persisted contract must match the COMPILED contract"
    );
    let read_stride: usize = meta.get(K_STRIDE).expect("stride").parse().expect("numeric");
    assert_eq!(read_stride, NODE_ROW_STRIDE);
    assert_eq!(meta.get(K_DIGEST).map(String::as_str), Some(digest.as_str()));

    println!("wrote  {dest}");
    println!("rows   {rows}  ({:.2} GiB)  in {wrote_s:.1}s", (rows * NODE_ROW_STRIDE) as f64 / (1u64 << 30) as f64);
    println!("header verified: layout v{read_layout}, stride {read_stride}, classid {classid}, digest {digest}");
    println!("fragments: {}", ds.get_fragments().len());

    // ── P-CACHE-4 (local only): is the row column a verbatim contiguous run? ──
    if !is_remote {
        let data_dir = std::path::Path::new(&dest).join("data");
        let mut checked = false;
        for entry in std::fs::read_dir(&data_dir).expect("data dir").flatten() {
            let p = entry.path();
            if p.extension().and_then(|e| e.to_str()) != Some("lance") {
                continue;
            }
            let file = std::fs::read(&p).expect("read data file");
            // find the slab's first 512 bytes
            if let Some(off) = file
                .windows(NODE_ROW_STRIDE)
                .position(|w| w == &probe[..NODE_ROW_STRIDE])
            {
                let run = probe.len().min(file.len() - off);
                let contiguous = &file[off..off + run] == &probe[..run];
                println!(
                    "P-CACHE-4: data file {} — row 0 found at offset {off}; first {run} bytes {}",
                    p.file_name().unwrap().to_string_lossy(),
                    if contiguous {
                        "CONTIGUOUS (verbatim run — mmap+offset can serve this)"
                    } else {
                        "NOT contiguous past row 0 — encoding reorders; mmap serving is NOT available"
                    }
                );
                checked = true;
                break;
            }
        }
        if !checked {
            println!(
                "P-CACHE-4: row 0's bytes NOT found verbatim in any data file — \
                 the column is encoded/compressed despite compression=none; mmap serving is NOT available"
            );
        }
    }
}
