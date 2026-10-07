// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Cypher query execution benchmark: `CypherQuery::execute` over in-memory
//! Arrow batches of N rows.
//!
//! WHAT THE TIMED LOOP MEASURES — query execution, end to end on in-memory
//! input: per-call Cypher planning (semantic analysis, logical plan,
//! DataFusion plan), table registration, DataFusion execution, and collection
//! of the result batch.
//!
//! WHAT IT DOES NOT MEASURE:
//! - storage: the inputs are written to Lance and fully scanned back ONCE,
//!   during setup, outside the timer;
//! - planning in isolation;
//! - first-batch latency.
//!
//! `Throughput::Elements(n)` reports INPUT rows per second. That is honest
//! only because setup asserts each input table, concatenated from every
//! scanned batch, holds exactly `n` rows (`processed_rows == requested_rows`) and
//! a pre-flight run asserts the output row count each query must produce
//! from all `n` rows. An earlier version kept only the FIRST scanned
//! `RecordBatch` for the 10K and 1M sizes while still crediting `n`, so its
//! "1M rows" cases executed on a single scan batch.
//!
//! Run from the repo root:
//! ```text
//! cargo bench -p lance-graph-benches --bench graph_execution
//! ```

use std::collections::HashMap;
use std::sync::Arc;

use arrow::compute::concat_batches;
use arrow_array::{Int32Array, RecordBatch, RecordBatchIterator, StringArray};
use arrow_schema::{DataType, Field, Schema as ArrowSchema};
use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use futures::TryStreamExt;
use lance::dataset::{Dataset, WriteMode, WriteParams};
use lance_graph::{CypherQuery, GraphConfig};
use tempfile::TempDir;

fn create_people_batch() -> RecordBatch {
    let schema = Arc::new(ArrowSchema::new(vec![
        Field::new("person_id", DataType::Int32, false),
        Field::new("name", DataType::Utf8, false),
        Field::new("age", DataType::Int32, false),
    ]));

    RecordBatch::try_new(
        schema,
        vec![
            Arc::new(Int32Array::from(vec![1, 2, 3, 4, 5])),
            Arc::new(StringArray::from(vec![
                "Alice", "Bob", "Carol", "David", "Eve",
            ])),
            Arc::new(Int32Array::from(vec![28, 34, 29, 42, 31])),
        ],
    )
    .unwrap()
}

fn create_friendship_batch() -> RecordBatch {
    let schema = Arc::new(ArrowSchema::new(vec![
        Field::new("person1_id", DataType::Int32, false),
        Field::new("person2_id", DataType::Int32, false),
        Field::new("friendship_type", DataType::Utf8, false),
    ]));

    RecordBatch::try_new(
        schema,
        vec![
            Arc::new(Int32Array::from(vec![1, 1, 2, 3, 4])),
            Arc::new(Int32Array::from(vec![2, 3, 4, 4, 5])),
            Arc::new(StringArray::from(vec![
                "close", "casual", "close", "casual", "close",
            ])),
        ],
    )
    .unwrap()
}

// Execute query using CypherQuery::execute against in-memory batches
fn execute_cypher_query(
    rt: &tokio::runtime::Runtime,
    q: &CypherQuery,
    datasets: HashMap<String, RecordBatch>,
) -> RecordBatch {
    rt.block_on(async move { q.execute(datasets, None).await.unwrap() })
}

fn make_people_batch(n: usize) -> RecordBatch {
    if n == 5 {
        return create_people_batch();
    }
    let schema = Arc::new(ArrowSchema::new(vec![
        Field::new("person_id", DataType::Int32, false),
        Field::new("name", DataType::Utf8, false),
        Field::new("age", DataType::Int32, false),
    ]));
    let ids: Vec<i32> = (0..n as i32).collect();
    let names: Vec<String> = (0..n).map(|i| format!("name_{}", i)).collect();
    let ages: Vec<i32> = (0..n as i32).map(|i| 20 + (i % 60)).collect();
    RecordBatch::try_new(
        schema,
        vec![
            Arc::new(Int32Array::from(ids)),
            Arc::new(StringArray::from(names)),
            Arc::new(Int32Array::from(ages)),
        ],
    )
    .unwrap()
}

fn make_friendship_batch(n: usize) -> RecordBatch {
    if n == 5 {
        return create_friendship_batch();
    }
    let schema = Arc::new(ArrowSchema::new(vec![
        Field::new("person1_id", DataType::Int32, false),
        Field::new("person2_id", DataType::Int32, false),
        Field::new("friendship_type", DataType::Utf8, false),
    ]));
    let src: Vec<i32> = (0..n as i32).collect();
    let dst: Vec<i32> = (0..n as i32).map(|i| (i + 1) % n as i32).collect();
    let ftype: Vec<&str> = std::iter::repeat_n("friend", n).collect();
    RecordBatch::try_new(
        schema,
        vec![
            Arc::new(Int32Array::from(src)),
            Arc::new(Int32Array::from(dst)),
            Arc::new(StringArray::from(ftype)),
        ],
    )
    .unwrap()
}

/// Write `n` people and `n` ring friendships to Lance, scan them back in
/// full, and return each table as ONE batch of exactly `n` rows.
fn load_via_lance(rt: &tokio::runtime::Runtime, n: usize) -> (TempDir, RecordBatch, RecordBatch) {
    let tmpdir = tempfile::tempdir().unwrap();
    let people = write_and_scan(rt, &tmpdir, "person.lance", make_people_batch(n), n);
    let friends = write_and_scan(rt, &tmpdir, "friendship.lance", make_friendship_batch(n), n);
    (tmpdir, people, friends)
}

fn write_and_scan(
    rt: &tokio::runtime::Runtime,
    dir: &TempDir,
    name: &str,
    batch: RecordBatch,
    requested_rows: usize,
) -> RecordBatch {
    let path = dir.path().join(name);
    rt.block_on(async {
        let schema = batch.schema();
        Dataset::write(
            RecordBatchIterator::new(vec![Ok(batch)].into_iter(), schema.clone()),
            path.to_str().unwrap(),
            Some(WriteParams {
                mode: WriteMode::Create,
                ..Default::default()
            }),
        )
        .await
        .unwrap();
        let ds = Dataset::open(path.to_str().unwrap()).await.unwrap();
        let batches = ds
            .scan()
            .try_into_stream()
            .await
            .unwrap()
            .try_collect::<Vec<_>>()
            .await
            .unwrap();
        let processed_rows: usize = batches.iter().map(|b| b.num_rows()).sum();
        assert_eq!(
            processed_rows,
            requested_rows,
            "{name}: scanned {processed_rows} rows in {} batches, wanted {requested_rows}",
            batches.len()
        );
        let one = concat_batches(&batches[0].schema(), &batches).unwrap();
        assert_eq!(one.num_rows(), requested_rows);
        one
    })
}

fn bench_cypher_execution(c: &mut Criterion) {
    let mut group = c.benchmark_group("cypher_execution");
    let sizes = [100usize, 10_000usize, 1_000_000usize];

    // Global runtime reused across iterations
    let rt = tokio::runtime::Runtime::new().unwrap();

    // Helper function to create graph config
    let make_config = || {
        GraphConfig::builder()
            .with_node_label("Person", "person_id")
            .with_relationship("FRIEND_OF", "person1_id", "person2_id")
            .build()
            .unwrap()
    };

    // Prebuild queries (reuse per iteration)
    let q_basic = CypherQuery::new("MATCH (n:Person) WHERE n.age > 50 RETURN n.name")
        .unwrap()
        .with_config(make_config());
    let q_single_hop = CypherQuery::new("MATCH (a:Person)-[:FRIEND_OF]->(b:Person) RETURN b.name")
        .unwrap()
        .with_config(make_config());
    let q_two_hop = CypherQuery::new(
        "MATCH (a:Person)-[:FRIEND_OF]->(b:Person)-[:FRIEND_OF]->(c:Person) RETURN c.name",
    )
    .unwrap()
    .with_config(make_config());

    // Load every size through the same path: write to Lance, scan back, and
    // concatenate ALL scanned batches. The TempDirs must outlive the loop.
    let (_tmp_small, person_small, friendship_small) = load_via_lance(&rt, 100);
    let (_tmp_medium, person_medium, friendship_medium) = load_via_lance(&rt, 10_000);
    let (_tmp_large, person_large, friendship_large) = load_via_lance(&rt, 1_000_000);

    // Pre-flight, untimed: each query must see every input row. The ages are
    // 20 + (i % 60), so `age > 50` keeps i % 60 in 31..=59; the friendship
    // table is a ring, so each hop yields exactly n rows.
    for (n, people, friends) in [
        (100usize, &person_small, &friendship_small),
        (10_000, &person_medium, &friendship_medium),
        (1_000_000, &person_large, &friendship_large),
    ] {
        let expected_filter = (0..n).filter(|i| 20 + (i % 60) > 50).count();
        let one = |q: &CypherQuery, with_edges: bool| {
            let mut ds = HashMap::new();
            ds.insert("Person".to_string(), people.clone());
            if with_edges {
                ds.insert("FRIEND_OF".to_string(), friends.clone());
            }
            execute_cypher_query(&rt, q, ds).num_rows()
        };
        assert_eq!(
            one(&q_basic, false),
            expected_filter,
            "basic_node_filter n={n}"
        );
        assert_eq!(one(&q_single_hop, true), n, "single_hop_expand n={n}");
        assert_eq!(one(&q_two_hop, true), n, "two_hop_expand n={n}");
    }

    // 1) Basic node filter + projection
    for &n in &sizes {
        group.throughput(Throughput::Elements(n as u64));
        group.bench_with_input(BenchmarkId::new("basic_node_filter", n), &n, |b, &n| {
            b.iter(|| {
                let people_batch = match n {
                    100 => person_small.clone(),
                    10_000 => person_medium.clone(),
                    _ => person_large.clone(),
                };
                let mut ds = HashMap::new();
                ds.insert("Person".to_string(), people_batch);
                let out = execute_cypher_query(&rt, &q_basic, ds);
                black_box(out.num_rows());
            })
        });
    }

    // 2) Single-hop relationship expansion
    for &n in &sizes {
        group.throughput(Throughput::Elements(n as u64));
        group.bench_with_input(BenchmarkId::new("single_hop_expand", n), &n, |b, &n| {
            b.iter(|| {
                let people_batch = match n {
                    100 => person_small.clone(),
                    10_000 => person_medium.clone(),
                    _ => person_large.clone(),
                };
                let friendship = match n {
                    100 => friendship_small.clone(),
                    10_000 => friendship_medium.clone(),
                    _ => friendship_large.clone(),
                };
                let mut ds = HashMap::new();
                ds.insert("Person".to_string(), people_batch);
                ds.insert("FRIEND_OF".to_string(), friendship);
                let out = execute_cypher_query(&rt, &q_single_hop, ds);
                black_box(out.num_rows());
            })
        });
    }

    // 3) Two-hop relationship expansion
    for &n in &sizes {
        group.throughput(Throughput::Elements(n as u64));
        group.bench_with_input(BenchmarkId::new("two_hop_expand", n), &n, |b, &n| {
            b.iter(|| {
                let people_batch = match n {
                    100 => person_small.clone(),
                    10_000 => person_medium.clone(),
                    _ => person_large.clone(),
                };
                let friendship = match n {
                    100 => friendship_small.clone(),
                    10_000 => friendship_medium.clone(),
                    _ => friendship_large.clone(),
                };
                let mut ds = HashMap::new();
                ds.insert("Person".to_string(), people_batch);
                ds.insert("FRIEND_OF".to_string(), friendship);
                let out = execute_cypher_query(&rt, &q_two_hop, ds);
                black_box(out.num_rows());
            })
        });
    }

    group.finish();
}

criterion_group!(benches, bench_cypher_execution);
criterion_main!(benches);
