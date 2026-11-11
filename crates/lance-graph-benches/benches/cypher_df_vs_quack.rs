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
//! Inputs are built in memory; `graph_execution` already measures the same
//! DataFusion path after a Lance round trip, and storage is not timed there
//! either.
//!
//! ```text
//! cargo bench -p lance-graph-benches --bench cypher_df_vs_quack
//! ```

use std::collections::HashMap;
use std::sync::Arc;

use arrow_array::{Array, Int32Array, Int64Array, RecordBatch, StringArray};
use arrow_schema::{DataType, Field, Schema as ArrowSchema};
use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use lance_graph::{CypherQuery, GraphConfig};
use lance_graph_cypher_quack::{
    compile, Answer, Binding, Compiled, EdgeTable, Kind, NodeTable, On, Refusal, TableLanes,
};
use lance_graph_mask_risc::LaneRef;
use lance_graph_quack::Col;

const FILTER: &str = "MATCH (n:Person) WHERE n.age > 50 RETURN count(*)";
const ONE_HOP: &str = "MATCH (a:Person)-[:FRIEND_OF]->(b:Person) RETURN count(*)";
const TWO_HOP: &str =
    "MATCH (a:Person)-[:FRIEND_OF]->(b:Person)-[:FRIEND_OF]->(c:Person) RETURN count(*)";

/// One size of the `graph_execution` input, as lanes and as Arrow batches.
struct Input {
    n: usize,
    age: Vec<i32>,
    pid: Vec<u32>,
    src: Vec<u32>,
    dst: Vec<u32>,
    person: RecordBatch,
    friends: RecordBatch,
}

impl Input {
    fn new(n: usize) -> Self {
        let age: Vec<i32> = (0..n as i32).map(|i| 20 + (i % 60)).collect();
        let pid: Vec<u32> = (0..n as u32).collect();
        let src: Vec<u32> = (0..n as u32).collect();
        let dst: Vec<u32> = (0..n as u32).map(|i| (i + 1) % n as u32).collect();
        let person = RecordBatch::try_new(
            Arc::new(ArrowSchema::new(vec![
                Field::new("person_id", DataType::Int32, false),
                Field::new("name", DataType::Utf8, false),
                Field::new("age", DataType::Int32, false),
            ])),
            vec![
                Arc::new(Int32Array::from_iter_values(pid.iter().map(|&v| v as i32))),
                Arc::new(StringArray::from_iter_values(
                    (0..n).map(|i| format!("name_{i}")),
                )),
                Arc::new(Int32Array::from(age.clone())),
            ],
        )
        .unwrap();
        let friends = RecordBatch::try_new(
            Arc::new(ArrowSchema::new(vec![
                Field::new("person1_id", DataType::Int32, false),
                Field::new("person2_id", DataType::Int32, false),
            ])),
            vec![
                Arc::new(Int32Array::from_iter_values(src.iter().map(|&v| v as i32))),
                Arc::new(Int32Array::from_iter_values(dst.iter().map(|&v| v as i32))),
            ],
        )
        .unwrap();
        Input {
            n,
            age,
            pid,
            src,
            dst,
            person,
            friends,
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
        self.age.iter().filter(|&&a| a > 50).count() as i64
    }
    fn oracle_one_hop(&self) -> i64 {
        self.src.len() as i64
    }
    fn oracle_two_hop(&self) -> i64 {
        let mut out_deg = vec![0i64; self.n];
        let mut in_deg = vec![0i64; self.n];
        for (&s, &t) in self.src.iter().zip(&self.dst) {
            out_deg[s as usize] += 1;
            in_deg[t as usize] += 1;
        }
        (0..self.n).map(|b| in_deg[b] * out_deg[b]).sum()
    }
}

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
    let person = [LaneRef::I32(&input.age), LaneRef::U32(&input.pid)];
    let edges = [LaneRef::U32(&input.src), LaneRef::U32(&input.dst)];
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
        .map(Input::new)
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
}

criterion_group!(benches, bench);
criterion_main!(benches);
