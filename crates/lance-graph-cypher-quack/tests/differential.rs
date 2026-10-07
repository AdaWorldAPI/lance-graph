//! Differential: every query runs three ways and must agree.
//!
//! 1. Cypher → Quack (this crate), over borrowed lanes;
//! 2. DataFusion (`CypherQuery::execute`, the upstream engine), over the same
//!    rows as Arrow batches;
//! 3. a row-at-a-time oracle that touches neither engine.
//!
//! The fixtures include #1305's two semantic witnesses, kept as oracles:
//! - DAG KNOWS = {1→2, 1→3, 2→3, 3→4, 4→5}: 2-hop `count(*)` = 4 paths,
//!   3 distinct end nodes. A lowering that answers a path count with node
//!   support says 3; this crate must REFUSE the 2-hop count instead.
//! - cycle {1→2, 2→1, 2→2}: 2-hop walks = 5 (DataFusion), trails = 4.
//!   Quack, like DataFusion, is WALK.
//!
//! Plus a parallel-edge + self-loop fixture where one-hop `count(*)` (walks)
//! differs from `count(DISTINCT b)` (support) — the one-hop form of the same
//! distinction — and a hashed graph with uneven degrees.
//!
//! Every edge table is stored in `dst` order (the precondition of
//! `count(DISTINCT b)`); `person_id` equals the row ordinal.

use std::collections::HashMap;
use std::sync::Arc;

use arrow_array::{Array, Int32Array, Int64Array, RecordBatch, StringArray};
use arrow_schema::{DataType, Field, Schema};
use lance_graph::{CypherQuery, GraphConfig};
use lance_graph_cypher_quack::{
    compile, Answer, Binding, Demand, EdgeTable, Kind, NodeTable, Refusal, TableLanes,
};
use lance_graph_mask_risc::LaneRef;
use lance_graph_quack::Col;

struct Graph {
    age: Vec<i32>,
    pid: Vec<u32>,
    src: Vec<u32>,
    dst: Vec<u32>,
}

impl Graph {
    fn new(age: Vec<i32>, mut edges: Vec<(u32, u32)>) -> Self {
        edges.sort_by_key(|&(s, t)| (t, s));
        let (src, dst) = edges.into_iter().unzip();
        let pid = (0..age.len() as u32).collect();
        Graph { age, pid, src, dst }
    }
    fn n(&self) -> usize {
        self.age.len()
    }
    fn batches(&self) -> HashMap<String, RecordBatch> {
        let person = RecordBatch::try_new(
            Arc::new(Schema::new(vec![
                Field::new("person_id", DataType::Int32, false),
                Field::new("name", DataType::Utf8, false),
                Field::new("age", DataType::Int32, false),
            ])),
            vec![
                Arc::new(Int32Array::from_iter_values(
                    self.pid.iter().map(|&v| v as i32),
                )),
                Arc::new(StringArray::from_iter_values(
                    (0..self.n()).map(|i| format!("p{i}")),
                )),
                Arc::new(Int32Array::from(self.age.clone())),
            ],
        )
        .unwrap();
        let knows = RecordBatch::try_new(
            Arc::new(Schema::new(vec![
                Field::new("src_id", DataType::Int32, false),
                Field::new("dst_id", DataType::Int32, false),
            ])),
            vec![
                Arc::new(Int32Array::from_iter_values(
                    self.src.iter().map(|&v| v as i32),
                )),
                Arc::new(Int32Array::from_iter_values(
                    self.dst.iter().map(|&v| v as i32),
                )),
            ],
        )
        .unwrap();
        HashMap::from([("Person".to_string(), person), ("KNOWS".to_string(), knows)])
    }
}

/// #1305's DAG, 0-based.
fn dag() -> Graph {
    Graph::new(
        vec![30, 40, 50, 60, 70],
        vec![(0, 1), (0, 2), (1, 2), (2, 3), (3, 4)],
    )
}
/// #1305's cycle {1→2, 2→1, 2→2}, 0-based.
fn cycle() -> Graph {
    Graph::new(vec![30, 60], vec![(0, 1), (1, 0), (1, 1)])
}
/// Parallel edges 0→1 twice, a self-loop on 2, and 2→1.
fn parallel() -> Graph {
    Graph::new(vec![25, 35, 45], vec![(0, 1), (0, 1), (2, 2), (2, 1)])
}
fn hashed(n: u64) -> Graph {
    let mix = |mut x: u64| {
        x = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
        x = (x ^ (x >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        x = (x ^ (x >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        x ^ (x >> 31)
    };
    let age = (0..n as i32).map(|i| 20 + (i % 60)).collect();
    let edges = (0..n)
        .map(|j| ((mix(2 * j) % n) as u32, (mix(2 * j + 1) % n) as u32))
        .collect();
    Graph::new(age, edges)
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
            rel_type: "KNOWS".into(),
            label: "Person".into(),
            src: Col(0),
            dst: Col(1),
            dst_ordered: true,
        }],
    }
}

fn df_config() -> GraphConfig {
    GraphConfig::builder()
        .with_node_label("Person", "person_id")
        .with_relationship("KNOWS", "src_id", "dst_id")
        .build()
        .unwrap()
}

/// The Cypher → Quack answer.
fn quack(g: &Graph, q: &str) -> Result<(Answer, Demand), Refusal> {
    let c = compile(q, &HashMap::new(), &binding())?;
    let person = [LaneRef::I32(&g.age), LaneRef::U32(&g.pid)];
    let edges = [LaneRef::U32(&g.src), LaneRef::U32(&g.dst)];
    let p = TableLanes {
        n_rows: g.n(),
        lanes: &person,
    };
    let e = TableLanes {
        n_rows: g.src.len(),
        lanes: &edges,
    };
    let a = match c.on {
        lance_graph_cypher_quack::On::Nodes(_) => c.execute(p, None),
        lance_graph_cypher_quack::On::Edges { .. } => c.execute(e, Some(p)),
    }
    .expect("executes");
    Ok((a, c.demand))
}

/// The DataFusion answer, read as an integer (`None` = SQL NULL).
fn datafusion(g: &Graph, q: &str) -> Result<Option<i64>, String> {
    let rt = tokio::runtime::Runtime::new().unwrap();
    let b = rt
        .block_on(
            CypherQuery::new(q)
                .unwrap()
                .with_config(df_config())
                .execute(g.batches(), None),
        )
        .map_err(|e| e.to_string())?;
    assert_eq!(b.num_rows(), 1, "{q}: one row");
    let c = b.column(0);
    if let Some(a) = c.as_any().downcast_ref::<Int64Array>() {
        return Ok((!a.is_null(0)).then(|| a.value(0)));
    }
    if let Some(a) = c.as_any().downcast_ref::<Int32Array>() {
        return Ok((!a.is_null(0)).then(|| a.value(0) as i64));
    }
    panic!("{q}: result type {:?}", c.data_type())
}

fn int(a: Answer) -> Option<i64> {
    match a {
        Answer::Int(v) => Some(v),
        Answer::OptInt(v) => v,
    }
}

/// Oracles, row at a time.
fn walks1(g: &Graph) -> i64 {
    g.src.len() as i64
}
fn distinct_dst(g: &Graph) -> i64 {
    let mut s = g.dst.clone();
    s.dedup();
    s.len() as i64
}
fn walks2(g: &Graph) -> i64 {
    let mut o = vec![0i64; g.n()];
    let mut i = vec![0i64; g.n()];
    for (&s, &t) in g.src.iter().zip(&g.dst) {
        o[s as usize] += 1;
        i[t as usize] += 1;
    }
    (0..g.n()).map(|b| i[b] * o[b]).sum()
}
fn ages(g: &Graph, f: impl Fn(i32) -> bool) -> Vec<i32> {
    g.age.iter().copied().filter(|&a| f(a)).collect()
}

fn fixtures() -> Vec<(&'static str, Graph)> {
    vec![
        ("dag", dag()),
        ("cycle", cycle()),
        ("parallel", parallel()),
        ("hashed1000", hashed(1000)),
    ]
}

/// Answered three ways, all equal.
#[test]
fn lowered_queries_agree_with_datafusion_and_the_oracle() {
    type Oracle = fn(&Graph) -> Option<i64>;
    let cases: &[(&str, Oracle, Demand)] = &[
        (
            "MATCH (n:Person) RETURN count(*)",
            |g| Some(g.n() as i64),
            Demand::TerminalSet,
        ),
        (
            "MATCH (n:Person) WHERE n.age > 50 RETURN count(*)",
            |g| Some(ages(g, |a| a > 50).len() as i64),
            Demand::TerminalSet,
        ),
        (
            "MATCH (n:Person) WHERE n.age >= 40 AND NOT n.age = 60 RETURN count(*)",
            |g| Some(ages(g, |a| a >= 40 && a != 60).len() as i64),
            Demand::TerminalSet,
        ),
        (
            "MATCH (n:Person) WHERE n.age < 31 OR n.age <= 25 RETURN count(*)",
            |g| Some(ages(g, |a| a < 31).len() as i64),
            Demand::TerminalSet,
        ),
        (
            "MATCH (n:Person) WHERE 50 < n.age RETURN count(*)",
            |g| Some(ages(g, |a| a > 50).len() as i64),
            Demand::TerminalSet,
        ),
        (
            "MATCH (n:Person) WHERE n.person_id <> 1 RETURN count(*)",
            |g| Some(g.n() as i64 - 1),
            Demand::TerminalSet,
        ),
        (
            "MATCH (n:Person) WHERE n.age > 50 RETURN sum(n.age)",
            |g| Some(ages(g, |a| a > 50).iter().map(|&a| a as i64).sum()),
            Demand::TerminalSet,
        ),
        (
            "MATCH (n:Person) WHERE n.age > 31 RETURN min(n.age)",
            |g| ages(g, |a| a > 31).into_iter().min().map(i64::from),
            Demand::TerminalSet,
        ),
        (
            "MATCH (n:Person) RETURN max(n.age)",
            |g| g.age.iter().copied().max().map(i64::from),
            Demand::TerminalSet,
        ),
        (
            "MATCH (n:Person) WHERE n.age > 1000 RETURN max(n.age)",
            |_| None,
            Demand::TerminalSet,
        ),
        // one hop: one edge row is one walk
        (
            "MATCH (a:Person)-[:KNOWS]->(b:Person) RETURN count(*)",
            |g| Some(walks1(g)),
            Demand::TerminalCount,
        ),
        (
            "MATCH (b:Person)<-[:KNOWS]-(a:Person) RETURN count(*)",
            |g| Some(walks1(g)),
            Demand::TerminalCount,
        ),
        (
            "MATCH (a:Person)-[:KNOWS]->(b:Person) RETURN count(DISTINCT b.person_id)",
            |g| Some(distinct_dst(g)),
            Demand::TerminalSet,
        ),
        (
            "MATCH (a:Person)-[:KNOWS]->(b:Person) RETURN count(*) LIMIT 5",
            |g| Some(walks1(g)),
            Demand::TerminalCount,
        ),
    ];
    for (name, g) in fixtures() {
        for &(q, oracle, demand) in cases {
            let want = oracle(&g);
            let (got, d) = quack(&g, q).unwrap_or_else(|r| panic!("{name}: {q} refused: {r:?}"));
            assert_eq!(int(got), want, "{name}: Quack vs oracle for {q}");
            assert_eq!(d, demand, "{name}: demand of {q}");
            let df =
                datafusion(&g, q).unwrap_or_else(|e| panic!("{name}: DataFusion failed {q}: {e}"));
            if q.contains("sum(") && want == Some(0) {
                // pinned separately: sum_over_no_rows_is_null_in_datafusion
                assert_eq!(df, None, "{name}: {q}");
            } else {
                assert_eq!(df, want, "{name}: DataFusion vs oracle for {q}");
            }
        }
    }
}

/// openCypher's `sum()` over no rows is 0; DataFusion returns SQL NULL.
/// Inherited from upstream (the planner maps `sum` to SQL `SUM`); Quack
/// answers 0. Pinned so a fix in either direction is noticed.
#[test]
fn sum_over_no_rows_is_null_in_datafusion() {
    let g = parallel(); // ages 25, 35, 45
    let q = "MATCH (n:Person) WHERE n.age > 50 RETURN sum(n.age)";
    assert_eq!(quack(&g, q).unwrap().0, Answer::Int(0));
    assert_eq!(datafusion(&g, q).unwrap(), None);
}

/// The one-hop bag/support distinction, measured on the fixture built to
/// expose it, so a lowering that answered `count(*)` with the target support
/// (or the reverse) fails here.
#[test]
fn one_hop_walks_differ_from_target_support() {
    let g = parallel();
    let walks = quack(&g, "MATCH (a:Person)-[:KNOWS]->(b:Person) RETURN count(*)")
        .unwrap()
        .0;
    let support = quack(
        &g,
        "MATCH (a:Person)-[:KNOWS]->(b:Person) RETURN count(DISTINCT b)",
    )
    .unwrap()
    .0;
    // 0→1, 0→1, 2→1, 2→2: four walks over two distinct targets {1, 2}
    assert_eq!(walks, Answer::Int(4));
    assert_eq!(support, Answer::Int(2));
}

/// `count(n)` / `count(DISTINCT b)` on a node variable: Quack answers;
/// DataFusion fails to plan because `COUNT(var)` reads a hard-coded `<var>__id` column
/// (`datafusion_planner/expression.rs`), not the label's id field. Inherited
/// from upstream; pinned so a fix there is noticed.
#[test]
fn count_of_a_node_variable_is_an_inherited_datafusion_defect() {
    type Oracle = fn(&Graph) -> i64;
    let cases: &[(&str, &str, Oracle)] = &[
        (
            "MATCH (a:Person)-[:KNOWS]->(b:Person) RETURN count(DISTINCT b)",
            "b__id",
            distinct_dst,
        ),
        ("MATCH (n:Person) RETURN count(n)", "n__id", |g| {
            g.n() as i64
        }),
    ];
    for (name, g) in fixtures() {
        for &(q, col, oracle) in cases {
            assert_eq!(
                quack(&g, q).unwrap().0,
                Answer::Int(oracle(&g)),
                "{name}: {q}"
            );
            let e = datafusion(&g, q).expect_err("DataFusion plans a node-variable count");
            assert!(e.contains(col), "{name}: {e}");
        }
    }
}

/// #1305's witnesses: DataFusion's path counts are preserved as oracles, and
/// the lowering refuses the 2-hop count instead of answering with support.
#[test]
fn two_hop_counts_are_refused_never_answered_with_support() {
    let q = "MATCH (a:Person)-[:KNOWS]->(b:Person)-[:KNOWS]->(c:Person) RETURN count(*)";
    // DAG: 4 paths (1→2→3, 1→3→4, 2→3→4, 3→4→5), 3 end nodes.
    assert_eq!(datafusion(&dag(), q).unwrap(), Some(4));
    assert_eq!(walks2(&dag()), 4);
    // cycle: 5 walks; trails (no edge twice) would be 4. DataFusion is WALK.
    assert_eq!(datafusion(&cycle(), q).unwrap(), Some(5));
    assert_eq!(walks2(&cycle()), 5);
    for (name, g) in fixtures() {
        match quack(&g, q) {
            Err(Refusal::Gap(why)) => assert!(why.contains("two or more hops"), "{name}: {why}"),
            other => panic!("{name}: a 2-hop count must be refused, got {other:?}"),
        }
    }
}

#[test]
fn every_other_shape_is_a_typed_refusal() {
    let g = dag();
    type Expect = fn(&Refusal) -> bool;
    let cases: &[(&str, Expect)] = &[
        (
            "MATCH (a:Person)-[:KNOWS]->(b:Person) RETURN a.age, b.age",
            |r| *r == Refusal::Bindings,
        ),
        (
            "MATCH (a:Person)-[:KNOWS]->(b:Person) WHERE a.age > 30 RETURN count(*)",
            |r| matches!(r, Refusal::Gap(w) if w.contains("WHERE with a hop")),
        ),
        (
            "MATCH (a:Person)-[:KNOWS*1..2]->(b:Person) RETURN count(*)",
            |r| matches!(r, Refusal::Gap(w) if w.contains("variable length")),
        ),
        (
            "MATCH (a:Person)-[:KNOWS]->(b:Person) RETURN count(DISTINCT a)",
            |r| matches!(r, Refusal::Layout(_)),
        ),
        ("MATCH (n:Person) RETURN count(DISTINCT n.age)", |r| {
            matches!(r, Refusal::Value(_))
        }),
        ("MATCH (n:Person) RETURN avg(n.age)", |r| {
            matches!(r, Refusal::Value(_))
        }),
        (
            "MATCH (n:Person) WHERE n.person_id > 1 RETURN count(*)",
            |r| matches!(r, Refusal::Gap(w) if w.contains("u32")),
        ),
        ("MATCH (n:Person) RETURN n.age", |r| {
            matches!(r, Refusal::Shape(_))
        }),
        ("MATCH (n:Person) RETURN count(*) SKIP 1", |r| {
            matches!(r, Refusal::Shape(_))
        }),
        ("MATCH (n:Person) WHERE n.height > 1 RETURN count(*)", |r| {
            matches!(r, Refusal::Unbound(_))
        }),
        ("MATCH (n:Ghost) RETURN count(*)", |r| {
            matches!(r, Refusal::Unplanned(_) | Refusal::Unbound(_))
        }),
        ("MATCH (n:Person RETURN count(*)", |r| {
            matches!(r, Refusal::Unparsed(_))
        }),
    ];
    for &(q, ok) in cases {
        match quack(&g, q) {
            Err(r) => assert!(ok(&r), "{q}: unexpected refusal {r:?}"),
            Ok(a) => panic!("{q}: must be refused, answered {a:?}"),
        }
    }
}

/// An edge table not declared `dst`-ordered refuses `count(DISTINCT b)`
/// rather than computing it some other way.
#[test]
fn distinct_target_needs_the_declared_order() {
    let mut b = binding();
    b.edges[0].dst_ordered = false;
    let r = compile(
        "MATCH (a:Person)-[:KNOWS]->(b:Person) RETURN count(DISTINCT b)",
        &HashMap::new(),
        &b,
    );
    assert!(matches!(r, Err(Refusal::Layout(_))), "{r:?}");
}
