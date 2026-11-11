// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! The same Cypher text answered two ways, over the inputs `graph_execution`
//! uses (100, 10,000 and 1,000,000 people; a ring of n `FRIEND_OF` edges;
//! `age = 20 + i % 60`; `person_id` = row ordinal):
//!
//! - `datafusion`: `CypherQuery::execute` (planning, table registration,
//!   DataFusion execution, result collection) over Arrow batches;
//! - `quack_compiled`: `lance_graph_cypher_quack::compile` (parse, semantic
//!   analysis, logical plan, lowering) plus `Compiled::execute`, per call;
//! - `quack_execute`: `Compiled::execute` only, the program compiled once
//!   before the timer.
//!
//! The queries are the count forms of `graph_execution`'s three, because the
//! Quack lowering answers terminals only (it has no row-returning carrier):
//!
//! | case | Cypher |
//! |---|---|
//! | `filter_count` | `MATCH (n:Person) WHERE n.age > 50 RETURN count(*)` |
//! | `one_hop_count` | `MATCH (a:Person)-[:FRIEND_OF]->(b:Person) RETURN count(*)` |
//! | `two_hop_count` | `MATCH (a:Person)-[:FRIEND_OF]->(b:Person)-[:FRIEND_OF]->(c:Person) RETURN count(*)` |
//!
//! `two_hop_count` has no Quack arm: the lowering refuses two hops with a
//! typed `Refusal::Gap` (the walk count needs a foreign-value sum). The
//! pre-flight asserts that refusal, so the arm appears here the day it lowers.
//!
//! Pre-flight, untimed: every arm's answer equals a row-at-a-time oracle at
//! every size, so a fast wrong answer cannot be reported.
//! `Throughput::Elements(n)` counts input rows (people for the filter, edges
//! for the hops).
//!
//! Inputs go through Lance once, during setup and outside the timer (as in
//! `graph_execution`); see [`Input`] for which columns each arm reads. The
//! `intake` group times those two reads separately, as a report: a full scan
//! of `person` against a scan of only the columns Quack's binding names.
//!
//! ```text
//! cargo bench -p lance-graph-benches --bench cypher_df_vs_quack
//! ```

use std::collections::HashMap;
use std::sync::Arc;

use arrow::compute::{cast, concat_batches};
use arrow_array::{
    Array, ArrayRef, Int32Array, Int64Array, RecordBatch, RecordBatchIterator, StringArray,
    UInt32Array,
};
use arrow_schema::{DataType, Field, Schema as ArrowSchema};
use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use lance_graph::{CypherQuery, GraphConfig};
use futures::TryStreamExt;
use lance::dataset::{Dataset, WriteMode, WriteParams};
use tempfile::TempDir;
use lance_graph_cypher_quack::{
    compile, Answer, Binding, Compiled, EdgeTable, Kind, NodeTable, On, Refusal, TableLanes,
};
use lance_graph_mask_risc::LaneRef;
use lance_graph_quack::Col;

const FILTER: &str = "MATCH (n:Person) WHERE n.age > 50 RETURN count(*)";
const ONE_HOP: &str = "MATCH (a:Person)-[:FRIEND_OF]->(b:Person) RETURN count(*)";
const TWO_HOP: &str =
    "MATCH (a:Person)-[:FRIEND_OF]->(b:Person)-[:FRIEND_OF]->(c:Person) RETURN count(*)";

/// One size of the `graph_execution` input, written to Lance once and read
/// back twice:
/// - DataFusion gets every column (`person_id`, `name`, `age`;
///   `person1_id`, `person2_id`), as `graph_execution` gives it;
/// - Quack gets only the columns its [`Binding`] names (`age`, `person_id`;
///   `person1_id`, `person2_id`), through a Lance projection, so `name` is
///   never decoded on its side.
///
/// The Quack lanes borrow the scanned Arrow buffers (`values()`), no copy.
/// The id columns are cast to `UInt32` once at intake, because Quack's fk
/// lanes are `u32`; that cast is the one intake copy.
struct Input {
    n: usize,
    _dir: TempDir,
    person: RecordBatch,
    friends: RecordBatch,
    age: Int32Array,
    pid: UInt32Array,
    src: UInt32Array,
    dst: UInt32Array,
}

fn people_batch(n: usize) -> RecordBatch {
    RecordBatch::try_new(
        Arc::new(ArrowSchema::new(vec![
            Field::new("person_id", DataType::Int32, false),
            Field::new("name", DataType::Utf8, false),
            Field::new("age", DataType::Int32, false),
        ])),
        vec![
            Arc::new(Int32Array::from_iter_values(0..n as i32)),
            Arc::new(StringArray::from_iter_values(
                (0..n).map(|i| format!("name_{i}")),
            )),
            Arc::new(Int32Array::from_iter_values(
                (0..n as i32).map(|i| 20 + (i % 60)),
            )),
        ],
    )
    .unwrap()
}

fn friends_batch(n: usize) -> RecordBatch {
    RecordBatch::try_new(
        Arc::new(ArrowSchema::new(vec![
            Field::new("person1_id", DataType::Int32, false),
            Field::new("person2_id", DataType::Int32, false),
        ])),
        vec![
            Arc::new(Int32Array::from_iter_values(0..n as i32)),
            Arc::new(Int32Array::from_iter_values(
                (0..n as i32).map(|i| (i + 1) % n as i32),
            )),
        ],
    )
    .unwrap()
}

fn write(rt: &tokio::runtime::Runtime, path: &str, batch: RecordBatch) {
    rt.block_on(async {
        let schema = batch.schema();
        Dataset::write(
            RecordBatchIterator::new(vec![Ok(batch)].into_iter(), schema),
            path,
            Some(WriteParams {
                mode: WriteMode::Create,
                ..Default::default()
            }),
        )
        .await
        .unwrap();
    })
}

/// Scan `path` in full, or only `columns`, and return ONE batch holding
/// every row (asserted).
fn scan(
    rt: &tokio::runtime::Runtime,
    path: &str,
    columns: Option<&[&str]>,
    rows: usize,
) -> RecordBatch {
    rt.block_on(async {
        let ds = Dataset::open(path).await.unwrap();
        let mut sc = ds.scan();
        if let Some(c) = columns {
            sc.project(c).unwrap();
        }
        let batches: Vec<RecordBatch> = sc
            .try_into_stream()
            .await
            .unwrap()
            .try_collect()
            .await
            .unwrap();
        let one = concat_batches(&batches[0].schema(), &batches).unwrap();
        assert_eq!(one.num_rows(), rows, "{path}");
        if let Some(c) = columns {
            assert_eq!(one.num_columns(), c.len(), "{path}: projection");
        }
        one
    })
}

fn col<'a>(b: &'a RecordBatch, name: &str) -> &'a ArrayRef {
    b.column_by_name(name).unwrap()
}

fn as_u32(a: &ArrayRef) -> UInt32Array {
    cast(a, &DataType::UInt32)
        .unwrap()
        .as_any()
        .downcast_ref::<UInt32Array>()
        .unwrap()
        .clone()
}

impl Input {
    fn new(rt: &tokio::runtime::Runtime, n: usize) -> Self {
        let dir = tempfile::tempdir().unwrap();
        let pp = dir.path().join("person.lance");
        let fp = dir.path().join("friend_of.lance");
        let (pp, fp) = (pp.to_str().unwrap(), fp.to_str().unwrap());
        write(rt, pp, people_batch(n));
        write(rt, fp, friends_batch(n));

        let person = scan(rt, pp, None, n);
        let friends = scan(rt, fp, None, n);
        let qp = scan(rt, pp, Some(&PERSON_LANES), n);
        let qf = scan(rt, fp, Some(&EDGE_LANES), n);
        let age = col(&qp, "age")
            .as_any()
            .downcast_ref::<Int32Array>()
            .unwrap()
            .clone();
        Input {
            n,
            person,
            friends,
            age,
            pid: as_u32(col(&qp, "person_id")),
            src: as_u32(col(&qf, "person1_id")),
            dst: as_u32(col(&qf, "person2_id")),
            _dir: dir,
        }
    }

    fn batches(&self, with_edges: bool) -> HashMap<String, RecordBatch> {
        let mut m = HashMap::from([("Person".to_string(), self.person.clone())]);
        if with_edges {
            m.insert("FRIEND_OF".to_string(), self.friends.clone());
        }
        m
    }

    /// Row-at-a-time answers, touching neither engine.
    fn oracle_filter(&self) -> i64 {
        self.age.values().iter().filter(|&&a| a > 50).count() as i64
    }
    fn oracle_one_hop(&self) -> i64 {
        self.src.len() as i64
    }
    fn oracle_two_hop(&self) -> i64 {
        let mut out_deg = vec![0i64; self.n];
        let mut in_deg = vec![0i64; self.n];
        for (&s, &t) in self.src.values().iter().zip(self.dst.values().iter()) {
            out_deg[s as usize] += 1;
            in_deg[t as usize] += 1;
        }
        (0..self.n).map(|b| in_deg[b] * out_deg[b]).sum()
    }
}

/// The columns Quack's binding reads; nothing else is decoded for it.
const PERSON_LANES: [&str; 2] = ["age", "person_id"];
const EDGE_LANES: [&str; 2] = ["person1_id", "person2_id"];

fn binding() -> Binding {
    Binding {
        nodes: vec![NodeTable {
            label: "Person".into(),
            id_property: "person_id".into(),
            properties: vec![
                ("age".into(), Col(0), Kind::I32),
                ("person_id".into(), Col(1), Kind::U32),
            ],
        }],
        edges: vec![EdgeTable {
            rel_type: "FRIEND_OF".into(),
            label: "Person".into(),
            src: Col(0),
            dst: Col(1),
            // The ring's last edge points back to row 0.
            dst_ordered: false,
        }],
    }
}

fn df_config() -> GraphConfig {
    GraphConfig::builder()
        .with_node_label("Person", "person_id")
        .with_relationship("FRIEND_OF", "person1_id", "person2_id")
        .build()
        .unwrap()
}

fn datafusion(rt: &tokio::runtime::Runtime, q: &CypherQuery, input: &Input, edges: bool) -> i64 {
    let b = rt
        .block_on(q.execute(input.batches(edges), None))
        .expect("datafusion executes");
    assert_eq!(b.num_rows(), 1);
    let c = b.column(0);
    if let Some(a) = c.as_any().downcast_ref::<Int64Array>() {
        return a.value(0);
    }
    if let Some(a) = c.as_any().downcast_ref::<Int32Array>() {
        return a.value(0) as i64;
    }
    panic!("count column type {:?}", c.data_type())
}

fn quack_execute(c: &Compiled, input: &Input) -> i64 {
    let person = [
        LaneRef::I32(input.age.values()),
        LaneRef::U32(input.pid.values()),
    ];
    let edges = [
        LaneRef::U32(input.src.values()),
        LaneRef::U32(input.dst.values()),
    ];
    let p = TableLanes {
        n_rows: input.n,
        lanes: &person,
    };
    let e = TableLanes {
        n_rows: input.src.len(),
        lanes: &edges,
    };
    let a = match c.on {
        On::Nodes(_) => c.execute(p, None),
        On::Edges { .. } => c.execute(e, Some(p)),
    }
    .expect("quack executes");
    match a {
        Answer::Int(v) => v,
        Answer::OptInt(v) => v.expect("count is never empty"),
    }
}

fn quack_compile(text: &str, bind: &Binding) -> Compiled {
    compile(text, &HashMap::new(), bind).expect("lowers")
}

fn bench(c: &mut Criterion) {
    let rt = tokio::runtime::Runtime::new().unwrap();
    let bind = binding();
    let inputs: Vec<Input> = [100usize, 10_000, 1_000_000]
        .into_iter()
        .map(|n| Input::new(&rt, n))
        .collect();

    let df = |t: &str| CypherQuery::new(t).unwrap().with_config(df_config());
    let df_filter = df(FILTER);
    let df_one = df(ONE_HOP);
    let df_two = df(TWO_HOP);
    let q_filter = quack_compile(FILTER, &bind);
    let q_one = quack_compile(ONE_HOP, &bind);

    // Pre-flight, untimed.
    match compile(TWO_HOP, &HashMap::new(), &bind) {
        Err(Refusal::Gap(_)) => {}
        Err(other) => panic!("two-hop refused for an unexpected reason: {other:?}"),
        Ok(_) => panic!("two-hop now lowers: add its Quack arm"),
    }
    for i in &inputs {
        let n = i.n;
        assert_eq!(datafusion(&rt, &df_filter, i, false), i.oracle_filter(), "df filter n={n}");
        assert_eq!(quack_execute(&q_filter, i), i.oracle_filter(), "quack filter n={n}");
        assert_eq!(datafusion(&rt, &df_one, i, true), i.oracle_one_hop(), "df one-hop n={n}");
        assert_eq!(quack_execute(&q_one, i), i.oracle_one_hop(), "quack one-hop n={n}");
        assert_eq!(datafusion(&rt, &df_two, i, true), i.oracle_two_hop(), "df two-hop n={n}");
    }

    let mut g = c.benchmark_group("cypher_df_vs_quack");
    for i in &inputs {
        let n = i.n;
        g.throughput(Throughput::Elements(n as u64));

        g.bench_with_input(BenchmarkId::new("filter_count/datafusion", n), i, |b, i| {
            b.iter(|| black_box(datafusion(&rt, &df_filter, i, false)))
        });
        g.bench_with_input(BenchmarkId::new("filter_count/quack_compiled", n), i, |b, i| {
            b.iter(|| black_box(quack_execute(&quack_compile(FILTER, &bind), i)))
        });
        g.bench_with_input(BenchmarkId::new("filter_count/quack_execute", n), i, |b, i| {
            b.iter(|| black_box(quack_execute(&q_filter, i)))
        });

        g.bench_with_input(BenchmarkId::new("one_hop_count/datafusion", n), i, |b, i| {
            b.iter(|| black_box(datafusion(&rt, &df_one, i, true)))
        });
        g.bench_with_input(BenchmarkId::new("one_hop_count/quack_compiled", n), i, |b, i| {
            b.iter(|| black_box(quack_execute(&quack_compile(ONE_HOP, &bind), i)))
        });
        g.bench_with_input(BenchmarkId::new("one_hop_count/quack_execute", n), i, |b, i| {
            b.iter(|| black_box(quack_execute(&q_one, i)))
        });

        g.bench_with_input(BenchmarkId::new("two_hop_count/datafusion", n), i, |b, i| {
            b.iter(|| black_box(datafusion(&rt, &df_two, i, true)))
        });
    }
    g.finish();

    let mut g = c.benchmark_group("intake");
    for i in &inputs {
        let n = i.n;
        let path = i._dir.path().join("person.lance");
        let path = path.to_str().unwrap().to_string();
        g.throughput(Throughput::Elements(n as u64));
        g.bench_with_input(BenchmarkId::new("person/all_columns", n), &path, |b, p| {
            b.iter(|| black_box(scan(&rt, p, None, n).num_rows()))
        });
        g.bench_with_input(BenchmarkId::new("person/bound_columns", n), &path, |b, p| {
            b.iter(|| black_box(scan(&rt, p, Some(&PERSON_LANES), n).num_rows()))
        });
    }
    g.finish();
}

criterion_group!(benches, bench);
criterion_main!(benches);
