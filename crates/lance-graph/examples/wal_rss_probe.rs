//! D-WAL-4 — where the L arm's resident memory goes.
//!
//! Hypothesis: every Lance commit inserts a full clone of the new manifest into
//! the session metadata cache (`record_successful_commit`, lance 13
//! `io/commit.rs`). An append-only store adds one fragment per version, so the
//! manifest of version v lists v fragments, and the cache holds
//! sum(v) = V^2/2 fragment entries until it reaches its 1 GiB default cap.
//!
//! Two arms append the same batches. `default` uses the default session;
//! `capped` passes a session whose metadata cache is limited to 8 MiB.
//! Prediction if the hypothesis holds: `default` cache bytes grow
//! super-linearly and RSS follows; `capped` cache stays at or under 8 MiB and
//! RSS stays flat.
//!
//! ```text
//! cargo run --release -p lance-graph --example wal_rss_probe -- <scratch-dir> <arm> <versions>
//! ```
//! Run each arm in its own process so RSS is not shared.

use std::sync::Arc;

use arrow_array::{Int64Array, RecordBatch, RecordBatchIterator};
use arrow_schema::{DataType, Field, Schema};
use lance::dataset::{Dataset, WriteMode, WriteParams};
use lance::session::Session;

fn rss_mib() -> f64 {
    std::fs::read_to_string("/proc/self/status")
        .ok()
        .and_then(|s| {
            s.lines()
                .find(|l| l.starts_with("VmRSS:"))
                .and_then(|l| l.split_whitespace().nth(1)?.parse::<u64>().ok())
        })
        .unwrap_or(0) as f64
        / 1024.0
}

#[tokio::main(flavor = "multi_thread")]
async fn main() {
    let args: Vec<String> = std::env::args().collect();
    let dir = &args[1];
    let arm = args[2].as_str();
    let versions: u64 = args[3].parse().expect("versions");
    let path = format!("{dir}/rss-{arm}.lance");
    let _ = std::fs::remove_dir_all(&path);

    let session = match arm {
        "default" => None,
        "capped" => Some(Arc::new(Session::new(
            0,
            8 << 20,
            Arc::new(Default::default()),
        ))),
        other => panic!("arm {other}"),
    };
    let schema = Arc::new(Schema::new(vec![
        Field::new("cycle", DataType::Int64, false),
        Field::new("v", DataType::Int64, false),
    ]));
    let batch = |c: u64| {
        RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(Int64Array::from(vec![c as i64; 256])),
                Arc::new(Int64Array::from_iter_values(0..256)),
            ],
        )
        .unwrap()
    };

    let params = |mode| WriteParams {
        mode,
        session: session.clone(),
        ..Default::default()
    };
    let reader = RecordBatchIterator::new(vec![Ok(batch(1))], schema.clone());
    let mut ds = Dataset::write(reader, &path, Some(params(WriteMode::Create)))
        .await
        .expect("create");

    println!("arm,version,fragments,cache_mib,cache_items,rss_mib");
    for c in 2..=versions {
        let reader = RecordBatchIterator::new(vec![Ok(batch(c))], schema.clone());
        ds.append(reader, None).await.expect("append");
        if c.is_power_of_two() || c % 200 == 0 || c == versions {
            let s = ds.session();
            println!(
                "{arm},{},{},{:.1},{},{:.1}",
                ds.version().version,
                ds.get_fragments().len(),
                s.size_bytes() as f64 / (1 << 20) as f64,
                s.approx_num_items(),
                rss_mib(),
            );
        }
    }
    let _ = std::fs::remove_dir_all(&path);
}
