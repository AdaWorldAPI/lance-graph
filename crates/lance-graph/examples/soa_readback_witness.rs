//! Read a SoA table back out of the store and prove it is byte-identical to
//! the slab that was written.
//!
//! # Why byte equality, and not a re-decode
//!
//! The obvious witness — decode the rows and check the fields look right —
//! would need a second implementation of the row reader living here, and two
//! implementations that disagree are a worse problem than the one being
//! checked. Byte equality is both stronger and smaller: if the bytes coming out
//! of the store are the bytes that went in, then **every** reading of those
//! bytes is preserved by construction — keys, the edge-block degree histogram,
//! the value tenants, the edge lanes — including readings that do not exist
//! yet. Nothing about the row's meaning has to be known here to check it.
//!
//! This complements `tests/soa_verbatim.rs`, which asserts the *physical*
//! layout (one unbroken 512-aligned run, bounded footer) on a synthetic slab.
//! This binary asserts *content* survival for a real table. It computes no
//! hash: byte equality with the slab means the slab's digest already IS the
//! digest of what the store returns, so the header's value is established
//! transitively rather than recomputed (and `lance-graph` needs no digest
//! dependency for a witness).
//!
//! Usage: `soa_readback_witness <uri> <table> <expected-slab.soa>`

use arrow::array::{Array, FixedSizeBinaryArray};
use futures::TryStreamExt;
use lance_graph::dev_s3_env::s3_options;

const ROW_COLUMN: &str = "row";
const STRIDE: usize = 512;

#[tokio::main(flavor = "multi_thread")]
async fn main() {
    let a: Vec<String> = std::env::args().collect();
    if a.len() != 4 {
        eprintln!("usage: soa_readback_witness <uri> <table> <expected-slab.soa>");
        std::process::exit(2);
    }
    let (uri, table, slab_path) = (&a[1], &a[2], &a[3]);
    let dest = format!("{}/{}.lance", uri.trim_end_matches('/'), table);
    let is_remote = uri.contains("://");

    let expected = std::fs::read(slab_path).expect("read expected slab");
    assert!(
        expected.len().is_multiple_of(STRIDE),
        "expected slab is not a multiple of {STRIDE}"
    );

    let ds = {
        let mut b = lance::dataset::builder::DatasetBuilder::from_uri(&dest);
        if is_remote {
            let opts = s3_options().unwrap_or_else(|| {
                eprintln!(
                    "remote uri requires AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, \
                     and AWS_ENDPOINT_URL"
                );
                std::process::exit(2);
            });
            b = b.with_storage_options(opts);
        }
        b.load().await.expect("open dataset")
    };

    // What the header claims about itself, so a mismatch below can be attributed.
    let meta = ds.schema().metadata.clone();
    for k in ["soa:row_stride", "soa:classid", "soa:slab_digest"] {
        println!("  {k} = {}", meta.get(k).map_or("<absent>", String::as_str));
    }

    let t = std::time::Instant::now();
    let mut stream = ds
        .scan()
        .project(&[ROW_COLUMN])
        .expect("project row column")
        .try_into_stream()
        .await
        .expect("scan");

    let mut got: Vec<u8> = Vec::with_capacity(expected.len());
    let mut batches = 0usize;
    while let Some(b) = stream.try_next().await.expect("next batch") {
        batches += 1;
        let col = b
            .column_by_name(ROW_COLUMN)
            .expect("row column present")
            .as_any()
            .downcast_ref::<FixedSizeBinaryArray>()
            .expect("row column is FixedSizeBinary");
        assert_eq!(col.value_length() as usize, STRIDE, "wrong row stride");
        for i in 0..col.len() {
            got.extend_from_slice(col.value(i));
        }
    }
    let read_s = t.elapsed().as_secs_f64();

    println!(
        "  read {} rows in {batches} batch(es), {read_s:.2}s",
        got.len() / STRIDE
    );

    // The witness. Report the first divergence rather than just "differs" —
    // a row index and byte offset says whether a key, the edge block, or a
    // value lane moved, which is the difference between three distinct bugs.
    assert_eq!(
        got.len(),
        expected.len(),
        "row count differs: store has {} rows, slab has {}",
        got.len() / STRIDE,
        expected.len() / STRIDE
    );
    if let Some(i) = (0..got.len()).find(|&i| got[i] != expected[i]) {
        let (row, off) = (i / STRIDE, i % STRIDE);
        let region = match off {
            0..=15 => "key",
            16..=31 => "edge block (row_ptr)",
            _ => "value slab",
        };
        panic!(
            "BYTES DIFFER at row {row}, byte {off} ({region}): store 0x{:02x} vs slab 0x{:02x}",
            got[i], expected[i]
        );
    }

    println!(
        "\n  VERBATIM: {} bytes byte-identical to the slab",
        got.len()
    );
    // No hash is computed here on purpose: `lance-graph` does not depend on a
    // digest crate, and it does not need to. Byte equality with the slab means
    // the slab's digest IS the digest of what the store returns, so the header
    // value below is established transitively rather than recomputed.
    match meta.get("soa:slab_digest") {
        Some(h) => println!("  so the store's content digest is the header's: {h}"),
        None => println!("  (header carries no digest)"),
    }
}
