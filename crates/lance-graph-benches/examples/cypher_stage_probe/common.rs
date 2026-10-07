//! Shared body of the Cypher stage probe. Compiled twice, verbatim:
//! - by `examples/cypher_stage_probe.rs` in this (fork) workspace;
//! - by a host crate outside the upstream checkout that path-depends on the
//!   upstream `lance-graph` at a pinned SHA.
//!
//! It only calls the public API both specimens share (parser, semantic,
//! logical_plan, datafusion_planner, InMemoryCatalog), so a difference between
//! the two runs is a difference between the specimens, not between harnesses.
//!
//! Stage boundaries (each timed and allocation-counted on its own):
//! - `parse`   `parser::parse_cypher_query`        (nom: tokenizing and AST
//!   construction are fused; they cannot be split from outside)
//! - `bind`    `SemanticAnalyzer::analyze`         (also substitutes `$params`)
//! - `lplan`   `LogicalPlanner::plan`              (graph logical plan)
//! - `dfplan`  `DataFusionPlanner::plan`           (DataFusion logical plan)
//! - `reg`     `SessionContext` + `MemTable` registration of the inputs
//! - `phys`    `execute_logical_plan` + `create_physical_plan` (DF optimizer)
//! - `exec`    `physical_plan::collect`            (execution)
//! - `mat`     `concat_batches` of the collected result
//!
//! Every answer is checked against a row-at-a-time oracle that touches no
//! engine. A query whose result disagrees with the oracle is reported WRONG,
//! never timed as if it had succeeded.

use std::alloc::{GlobalAlloc, Layout, System};
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering::Relaxed};
use std::sync::Arc;
use std::time::Instant;

use arrow_array::{Array, Int32Array, Int64Array, RecordBatch, StringArray};
use arrow_schema::{DataType, Field, Schema};

use datafusion::datasource::{DefaultTableSource, MemTable};
use datafusion::execution::context::SessionContext;
use lance_graph::datafusion_planner::{DataFusionPlanner, GraphPhysicalPlanner};
use lance_graph::logical_plan::LogicalPlanner;
use lance_graph::parser::parse_cypher_query;
use lance_graph::semantic::SemanticAnalyzer;
use lance_graph::{GraphConfig, InMemoryCatalog};

// ---------------------------------------------------------------- allocator

pub struct CountingAlloc;
pub static ALLOCS: AtomicU64 = AtomicU64::new(0);
pub static BYTES: AtomicU64 = AtomicU64::new(0);

unsafe impl GlobalAlloc for CountingAlloc {
    unsafe fn alloc(&self, l: Layout) -> *mut u8 {
        ALLOCS.fetch_add(1, Relaxed);
        BYTES.fetch_add(l.size() as u64, Relaxed);
        unsafe { System.alloc(l) }
    }
    unsafe fn dealloc(&self, p: *mut u8, l: Layout) {
        unsafe { System.dealloc(p, l) }
    }
    unsafe fn realloc(&self, p: *mut u8, l: Layout, n: usize) -> *mut u8 {
        ALLOCS.fetch_add(1, Relaxed);
        BYTES.fetch_add(n as u64, Relaxed);
        unsafe { System.realloc(p, l, n) }
    }
}

#[derive(Clone, Copy, Default)]
pub struct Cost {
    pub ns: u128,
    pub allocs: u64,
    pub bytes: u64,
}

pub fn measure<T>(f: impl FnOnce() -> T) -> (T, Cost) {
    let a0 = ALLOCS.load(Relaxed);
    let b0 = BYTES.load(Relaxed);
    let t = Instant::now();
    let v = f();
    let ns = t.elapsed().as_nanos();
    (
        v,
        Cost {
            ns,
            allocs: ALLOCS.load(Relaxed) - a0,
            bytes: BYTES.load(Relaxed) - b0,
        },
    )
}

pub fn median(mut v: Vec<u128>) -> u128 {
    v.sort_unstable();
    v[v.len() / 2]
}

// ---------------------------------------------------------------- data

/// The logical graph both engines and the oracle read. `person_id` equals the
/// row ordinal, so the same columns serve DataFusion (as ids) and Quack (as
/// row ordinals) with no remap.
pub struct Data {
    pub n_people: usize,
    pub age: Vec<i32>,
    pub name: Vec<String>,
    pub src: Vec<u32>,
    pub dst: Vec<u32>,
}

fn splitmix(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
    x = (x ^ (x >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    x ^ (x >> 31)
}

/// `n` people, `n` hashed edges: uneven in- and out-degree, self-loops and
/// parallel edges allowed. That is what makes bag counts (paths) differ from
/// terminal support (distinct nodes) — the upstream bench's ring (`i -> i+1`)
/// has in = out = 1 everywhere and cannot expose the difference.
pub fn make_data(n: usize) -> Data {
    let age = (0..n as i32).map(|i| 20 + (i % 60)).collect();
    let name = (0..n).map(|i| format!("name_{i}")).collect();
    let mut edges: Vec<(u32, u32)> = (0..n as u64)
        .map(|j| {
            (
                (splitmix(2 * j) % n as u64) as u32,
                (splitmix(2 * j + 1) % n as u64) as u32,
            )
        })
        .collect();
    // Stored in `dst` order: the physical precondition of Quack's only exact
    // `count(DISTINCT dst)` (`Agg::CountDistinctOrderedU32`). Every engine and
    // the oracle read this same row order.
    edges.sort_by_key(|&(s, t)| (t, s));
    let (src, dst) = edges.into_iter().unzip();
    Data {
        n_people: n,
        age,
        name,
        src,
        dst,
    }
}

/// The #1305 fixture: KNOWS = {1→2, 1→3, 2→3, 3→4, 4→5}, stored 0-based.
/// Measured there: 2-hop `count(*)` = 4, `count(DISTINCT c)` = 3.
pub fn small_dag() -> Data {
    Data {
        n_people: 5,
        age: vec![30, 40, 50, 60, 70],
        name: (0..5).map(|i| format!("p{i}")).collect(),
        src: vec![0, 0, 1, 2, 3],
        dst: vec![1, 2, 2, 3, 4],
    }
}

/// The #1305 cycle: {1→2, 2→1, 2→2}, stored 0-based. Two-hop walks = 5
/// (DataFusion's answer, pinned there), trails = 4 (Cypher's relationship
/// uniqueness drops 2→2→2).
pub fn small_cycle() -> Data {
    Data {
        n_people: 2,
        age: vec![30, 60],
        name: vec!["p0".into(), "p1".into()],
        // {2→1, 1→2, 2→2} in dst order
        src: vec![1, 0, 1],
        dst: vec![0, 1, 1],
    }
}

/// Size code → fixture: 5 is the #1305 DAG, 2 is the #1305 cycle, anything
/// else is `make_data(n)`.
pub fn fixture(n: usize) -> Data {
    match n {
        5 => small_dag(),
        2 => small_cycle(),
        _ => make_data(n),
    }
}

impl Data {
    pub fn batches(&self) -> (RecordBatch, RecordBatch) {
        let ps = Arc::new(Schema::new(vec![
            Field::new("person_id", DataType::Int32, false),
            Field::new("name", DataType::Utf8, false),
            Field::new("age", DataType::Int32, false),
        ]));
        let person = RecordBatch::try_new(
            ps,
            vec![
                Arc::new(Int32Array::from_iter_values(0..self.n_people as i32)),
                Arc::new(StringArray::from_iter_values(self.name.iter())),
                Arc::new(Int32Array::from(self.age.clone())),
            ],
        )
        .unwrap();
        let fs = Arc::new(Schema::new(vec![
            Field::new("person1_id", DataType::Int32, false),
            Field::new("person2_id", DataType::Int32, false),
        ]));
        let friend = RecordBatch::try_new(
            fs,
            vec![
                Arc::new(Int32Array::from_iter_values(self.src.iter().map(|&v| v as i32))),
                Arc::new(Int32Array::from_iter_values(self.dst.iter().map(|&v| v as i32))),
            ],
        )
        .unwrap();
        (person, friend)
    }
}

// ---------------------------------------------------------------- corpus

#[derive(Debug, Clone, PartialEq)]
pub enum Answer {
    Int(i64),
    Strings(Vec<String>),
    /// LIMIT 1 existence shape: `true` iff exactly one row came back.
    Exists(bool),
}

pub struct Q {
    pub id: &'static str,
    pub text: &'static str,
    pub params: Vec<(&'static str, serde_json::Value)>,
}

pub const PROBE_ID: i64 = 7;

pub fn corpus() -> Vec<Q> {
    let q = |id, text| Q {
        id,
        text,
        params: vec![],
    };
    vec![
        q("Q0", "MATCH (n:Person) RETURN count(*)"),
        q("Q1", "MATCH (n:Person) WHERE n.age > 50 RETURN count(*)"),
        q("Q2", "MATCH (n:Person) WHERE n.age > 50 RETURN sum(n.age)"),
        q("Q3", "MATCH (a:Person)-[:FRIEND_OF]->(b:Person) RETURN count(*)"),
        q(
            "Q4",
            "MATCH (a:Person)-[:FRIEND_OF]->(b:Person)-[:FRIEND_OF]->(c:Person) RETURN count(*)",
        ),
        Q {
            id: "Q5",
            text: "MATCH (n:Person) WHERE n.person_id = $id RETURN n.name",
            params: vec![("id", serde_json::json!(PROBE_ID))],
        },
        q(
            "Q6",
            "MATCH (a:Person)-[:FRIEND_OF]->(b:Person) RETURN count(DISTINCT b.person_id)",
        ),
        q(
            "Q6b",
            "MATCH (a:Person)-[:FRIEND_OF]->(b:Person) RETURN count(DISTINCT b)",
        ),
        q(
            "Q7",
            "MATCH (a:Person)-[:FRIEND_OF]->(b:Person) WHERE a.age > 77 RETURN count(*)",
        ),
        q(
            "Q8",
            "MATCH (a:Person)-[:FRIEND_OF]->(b:Person) WHERE a.age > 21 RETURN count(*)",
        ),
        q(
            "Q9",
            "MATCH (n:Person) WHERE n.age > 78 RETURN n.name LIMIT 1",
        ),
        q(
            "QV",
            "MATCH (a:Person)-[:FRIEND_OF*1..2]->(b:Person) WHERE a.person_id = 0 RETURN count(*)",
        ),
        q(
            "QVd",
            "MATCH (a:Person)-[:FRIEND_OF*1..2]->(b:Person) WHERE a.person_id = 0 RETURN count(DISTINCT b.person_id)",
        ),
        q(
            "Q10",
            "MATCH (a:Person)-[:FRIEND_OF]->(b:Person)-[:FRIEND_OF]->(c:Person) RETURN count(DISTINCT c.person_id)",
        ),
    ]
}

/// Row-at-a-time oracle. Touches no engine.
pub fn oracle(id: &str, d: &Data) -> Answer {
    let n = d.n_people;
    let mut out_deg = vec![0i64; n];
    let mut in_deg = vec![0i64; n];
    for (&s, &t) in d.src.iter().zip(&d.dst) {
        out_deg[s as usize] += 1;
        in_deg[t as usize] += 1;
    }
    let distinct = |v: &mut Vec<bool>| v.iter().filter(|&&b| b).count() as i64;
    match id {
        "Q0" => Answer::Int(n as i64),
        "Q1" => Answer::Int(d.age.iter().filter(|&&a| a > 50).count() as i64),
        "Q2" => Answer::Int(d.age.iter().filter(|&&a| a > 50).map(|&a| a as i64).sum()),
        "Q3" => Answer::Int(d.src.len() as i64),
        // walks a->b->c: for every middle node b, in(b) * out(b)
        "Q4" => Answer::Int((0..n).map(|b| in_deg[b] * out_deg[b]).sum()),
        "Q5" => Answer::Strings(if (PROBE_ID as usize) < n {
            vec![d.name[PROBE_ID as usize].clone()]
        } else {
            vec![]
        }),
        "Q6" | "Q6b" => {
            let mut seen = vec![false; n];
            for &t in &d.dst {
                seen[t as usize] = true;
            }
            Answer::Int(distinct(&mut seen))
        }
        "Q7" | "Q8" => {
            let k = if id == "Q7" { 77 } else { 21 };
            Answer::Int(d.src.iter().filter(|&&s| d.age[s as usize] > k).count() as i64)
        }
        "Q9" => Answer::Exists(d.age.iter().any(|&a| a > 78)),
        // walks of length 1..2 from node 0; WALK semantics (edges may repeat)
        "QV" => {
            let w1 = out_deg[0];
            let w2: i64 = d
                .src
                .iter()
                .zip(&d.dst)
                .filter(|(&s, _)| s == 0)
                .map(|(_, &t)| out_deg[t as usize])
                .sum();
            Answer::Int(w1 + w2)
        }
        "QVd" => {
            // hop 1 from node 0, then hop 2 from that frontier: two linear passes
            let mut f1 = vec![false; n];
            for (&s, &t) in d.src.iter().zip(&d.dst) {
                if s == 0 {
                    f1[t as usize] = true;
                }
            }
            let mut seen = f1.clone();
            for (&s, &t) in d.src.iter().zip(&d.dst) {
                if f1[s as usize] {
                    seen[t as usize] = true;
                }
            }
            Answer::Int(distinct(&mut seen))
        }
        "Q10" => {
            // c is reachable as the end of a 2-walk iff some edge b->c has in(b) > 0
            let mut seen = vec![false; n];
            for (&s, &t) in d.src.iter().zip(&d.dst) {
                if in_deg[s as usize] > 0 {
                    seen[t as usize] = true;
                }
            }
            Answer::Int(distinct(&mut seen))
        }
        _ => unreachable!(),
    }
}

fn read_answer(id: &str, b: &RecordBatch) -> Answer {
    if id == "Q9" {
        return Answer::Exists(b.num_rows() == 1);
    }
    if id == "Q5" {
        let c = b.column(0);
        let s = c.as_any().downcast_ref::<StringArray>().expect("Utf8 name");
        return Answer::Strings((0..s.len()).map(|i| s.value(i).to_string()).collect());
    }
    assert_eq!(b.num_rows(), 1, "{id}: aggregate returns one row");
    let c = b.column(0);
    if let Some(a) = c.as_any().downcast_ref::<Int64Array>() {
        return Answer::Int(if a.is_null(0) { 0 } else { a.value(0) });
    }
    if let Some(a) = c.as_any().downcast_ref::<Int32Array>() {
        return Answer::Int(a.value(0) as i64);
    }
    panic!("{id}: unexpected result type {:?}", c.data_type())
}

pub fn config() -> GraphConfig {
    GraphConfig::builder()
        .with_node_label("Person", "person_id")
        .with_relationship("FRIEND_OF", "person1_id", "person2_id")
        .build()
        .unwrap()
}

// ---------------------------------------------------------------- pipeline

#[derive(Default, Clone, Copy)]
pub struct Stages {
    pub parse: Cost,
    pub bind: Cost,
    pub lplan: Cost,
    pub dfplan: Cost,
    pub reg: Cost,
    pub phys: Cost,
    pub exec: Cost,
    pub mat: Cost,
}

pub enum Outcome {
    Ok(Answer, Stages),
    Err(&'static str, String),
}

fn catalog_and_ctx(person: &RecordBatch, friend: &RecordBatch) -> (InMemoryCatalog, SessionContext) {
    let ctx = SessionContext::new();
    let mut catalog = InMemoryCatalog::new();
    for (name, batch) in [("Person", person), ("FRIEND_OF", friend)] {
        let mem = Arc::new(MemTable::try_new(batch.schema(), vec![vec![batch.clone()]]).unwrap());
        ctx.register_table(name.to_lowercase(), mem.clone()).unwrap();
        let src = Arc::new(DefaultTableSource::new(mem));
        catalog = catalog
            .with_node_source(name, src.clone())
            .with_relationship_source(name, src);
    }
    (catalog, ctx)
}

/// One cold run of the whole pipeline, stage by stage.
pub async fn run_cold(q: &Q, person: &RecordBatch, friend: &RecordBatch) -> Outcome {
    let cfg = config();
    let params: HashMap<String, serde_json::Value> =
        q.params.iter().map(|(k, v)| (k.to_string(), v.clone())).collect();
    let mut st = Stages::default();

    let (ast, c) = measure(|| parse_cypher_query(q.text));
    st.parse = c;
    let ast = match ast {
        Ok(a) => a,
        Err(e) => return Outcome::Err("parse", e.to_string()),
    };
    let (sem, c) = measure(|| SemanticAnalyzer::new(cfg.clone()).analyze(&ast, &params));
    st.bind = c;
    let sem = match sem {
        Ok(s) if s.errors.is_empty() => s,
        Ok(s) => return Outcome::Err("bind", s.errors.join("; ")),
        Err(e) => return Outcome::Err("bind", e.to_string()),
    };
    let (lp, c) = measure(|| LogicalPlanner::new(&cfg).plan(&sem.ast));
    st.lplan = c;
    let lp = match lp {
        Ok(p) => p,
        Err(e) => return Outcome::Err("lplan", e.to_string()),
    };
    let ((catalog, ctx), c) = measure(|| catalog_and_ctx(person, friend));
    st.reg = c;
    let catalog = Arc::new(catalog);
    let (dfp, c) = measure(|| DataFusionPlanner::with_catalog(cfg.clone(), catalog).plan(&lp));
    st.dfplan = c;
    let dfp = match dfp {
        Ok(p) => p,
        Err(e) => return Outcome::Err("dfplan", e.to_string()),
    };
    let t = (ALLOCS.load(Relaxed), BYTES.load(Relaxed), Instant::now());
    let phys = match ctx.execute_logical_plan(dfp).await {
        Ok(df) => df.create_physical_plan().await,
        Err(e) => Err(e),
    };
    st.phys = Cost {
        ns: t.2.elapsed().as_nanos(),
        allocs: ALLOCS.load(Relaxed) - t.0,
        bytes: BYTES.load(Relaxed) - t.1,
    };
    let phys = match phys {
        Ok(p) => p,
        Err(e) => return Outcome::Err("phys", e.to_string()),
    };
    let t = (ALLOCS.load(Relaxed), BYTES.load(Relaxed), Instant::now());
    let batches = datafusion::physical_plan::collect(phys, ctx.task_ctx()).await;
    st.exec = Cost {
        ns: t.2.elapsed().as_nanos(),
        allocs: ALLOCS.load(Relaxed) - t.0,
        bytes: BYTES.load(Relaxed) - t.1,
    };
    let batches = match batches {
        Ok(b) => b,
        Err(e) => return Outcome::Err("exec", e.to_string()),
    };
    let (batch, c) = measure(|| {
        if batches.is_empty() {
            None
        } else {
            Some(arrow::compute::concat_batches(&batches[0].schema(), &batches).unwrap())
        }
    });
    st.mat = c;
    let ans = match batch {
        Some(b) => read_answer(q.id, &b),
        None if q.id == "Q9" => Answer::Exists(false),
        None if q.id == "Q5" => Answer::Strings(vec![]),
        None => return Outcome::Err("mat", "empty result".into()),
    };
    Outcome::Ok(ans, st)
}

/// Prepared/repeated execution. Parse, bind, graph planning and DataFusion
/// logical planning run ONCE; each repetition rebuilds only the physical plan
/// and executes it. Reusing the physical plan itself is not possible: see
/// [`reexec_probe`].
pub async fn run_prepared_exec(q: &Q, person: &RecordBatch, friend: &RecordBatch, reps: usize) -> Option<u128> {
    let cfg = config();
    let params: HashMap<String, serde_json::Value> =
        q.params.iter().map(|(k, v)| (k.to_string(), v.clone())).collect();
    let ast = parse_cypher_query(q.text).ok()?;
    let sem = SemanticAnalyzer::new(cfg.clone()).analyze(&ast, &params).ok()?;
    let lp = LogicalPlanner::new(&cfg).plan(&sem.ast).ok()?;
    let (catalog, ctx) = catalog_and_ctx(person, friend);
    let dfp = DataFusionPlanner::with_catalog(cfg, Arc::new(catalog)).plan(&lp).ok()?;
    let mut ts = Vec::with_capacity(reps);
    for _ in 0..reps {
        let t = Instant::now();
        let phys = ctx.execute_logical_plan(dfp.clone()).await.ok()?.create_physical_plan().await.ok()?;
        let b = datafusion::physical_plan::collect(phys, ctx.task_ctx()).await.ok()?;
        std::hint::black_box(&b);
        ts.push(t.elapsed().as_nanos());
    }
    Some(median(ts))
}

/// Can one physical plan be executed twice? Runs `collect` on the same
/// `Arc<dyn ExecutionPlan>` two times, each inside its own task so a panic is
/// observed rather than taking the probe down.
pub async fn reexec_probe() {
    let d = make_data(10_000);
    let (person, friend) = d.batches();
    println!("# physical-plan re-execution (size 10000)");
    for q in corpus() {
        let cfg = config();
        let params: HashMap<String, serde_json::Value> =
            q.params.iter().map(|(k, v)| (k.to_string(), v.clone())).collect();
        let ast = parse_cypher_query(q.text).unwrap();
        let sem = SemanticAnalyzer::new(cfg.clone()).analyze(&ast, &params).unwrap();
        let lp = LogicalPlanner::new(&cfg).plan(&sem.ast).unwrap();
        let (catalog, ctx) = catalog_and_ctx(&person, &friend);
        let dfp = match DataFusionPlanner::with_catalog(cfg, Arc::new(catalog)).plan(&lp) {
            Ok(p) => p,
            Err(_) => {
                println!("{}	no plan", q.id);
                continue;
            }
        };
        let phys = ctx.execute_logical_plan(dfp).await.unwrap().create_physical_plan().await.unwrap();
        let mut verdicts = Vec::new();
        for _ in 0..2 {
            let (p, tc) = (phys.clone(), ctx.task_ctx());
            let r = tokio::spawn(async move { datafusion::physical_plan::collect(p, tc).await }).await;
            verdicts.push(match r {
                Ok(Ok(b)) => format!("ok({} rows)", b.iter().map(|b| b.num_rows()).sum::<usize>()),
                Ok(Err(e)) => format!("err({})", e.to_string().chars().take(60).collect::<String>()),
                Err(_) => "PANIC".to_string(),
            });
        }
        println!("{}	{}	{}", q.id, verdicts[0], verdicts[1]);
    }
}

pub fn fmt_us(ns: u128) -> String {
    format!("{:.1}", ns as f64 / 1000.0)
}

/// Median of `reps` cold runs per stage, plus allocation counts of the last.
pub async fn suite(label: &str, sizes: &[usize], reps_for: impl Fn(usize) -> usize) {
    // PROBE_SIZES=1000000 restricts the sizes.
    let env_sizes: Option<Vec<usize>> =
        std::env::var("PROBE_SIZES").ok().map(|v| v.split(',').filter_map(|x| x.parse().ok()).collect());
    let sizes: &[usize] = env_sizes.as_deref().unwrap_or(sizes);
    println!("# specimen = {label}");
    println!("size\tq\tverdict\tparse_us\tbind_us\tlplan_us\tdfplan_us\treg_us\tphys_us\texec_us\tmat_us\ttotal_us\tprep_exec_us\tparse_allocs\tparse_bytes\tbind_allocs\tlplan_allocs\tdfplan_allocs\tphys_allocs\texec_allocs");
    for &n in sizes {
        let d = fixture(n);
        let (person, friend) = d.batches();
        // PROBE_REPS overrides the repetition count; PROBE_Q=Q0,Q7 filters.
        let reps = std::env::var("PROBE_REPS").ok().and_then(|v| v.parse().ok()).unwrap_or(reps_for(n));
        let only: Option<Vec<String>> = std::env::var("PROBE_Q").ok().map(|v| v.split(',').map(str::to_string).collect());
        for q in corpus() {
            if only.as_ref().is_some_and(|o| !o.iter().any(|x| x == q.id)) {
                continue;
            }
            let want = oracle(q.id, &d);
            let mut per: Vec<Stages> = Vec::new();
            let mut verdict = String::from("OK");
            for _ in 0..reps {
                match run_cold(&q, &person, &friend).await {
                    Outcome::Ok(got, st) => {
                        if got != want {
                            verdict = format!("WRONG(got={got:?},want={want:?})");
                        }
                        per.push(st);
                    }
                    Outcome::Err(stage, e) => {
                        let e: String = e.chars().filter(|c| *c != '\n' && *c != '\t').take(140).collect();
                        verdict = format!("FAIL@{stage}: {e}");
                        break;
                    }
                }
            }
            if per.is_empty() {
                println!("{n}\t{}\t{verdict}", q.id);
                continue;
            }
            let m = |f: fn(&Stages) -> u128| median(per.iter().map(f).collect());
            let total = m(|s| {
                s.parse.ns + s.bind.ns + s.lplan.ns + s.dfplan.ns + s.reg.ns + s.phys.ns + s.exec.ns + s.mat.ns
            });
            let prep = if verdict == "OK" {
                run_prepared_exec(&q, &person, &friend, reps).await.map(fmt_us).unwrap_or("-".into())
            } else {
                "-".into()
            };
            let l = per.last().unwrap();
            println!(
                "{n}\t{}\t{verdict}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{prep}\t{}\t{}\t{}\t{}\t{}\t{}\t{}",
                q.id,
                fmt_us(m(|s| s.parse.ns)),
                fmt_us(m(|s| s.bind.ns)),
                fmt_us(m(|s| s.lplan.ns)),
                fmt_us(m(|s| s.dfplan.ns)),
                fmt_us(m(|s| s.reg.ns)),
                fmt_us(m(|s| s.phys.ns)),
                fmt_us(m(|s| s.exec.ns)),
                fmt_us(m(|s| s.mat.ns)),
                fmt_us(total),
                l.parse.allocs,
                l.parse.bytes,
                l.bind.allocs,
                l.lplan.allocs,
                l.dfplan.allocs,
                l.phys.allocs,
                l.exec.allocs,
            );
        }
    }
}

/// Parser-only microbench: short and long queries, many reps, alloc counts.
pub fn parser_bench() {
    let long = "MATCH (a:Person)-[:FRIEND_OF]->(b:Person)-[:FRIEND_OF]->(c:Person) \
        WHERE a.age > 30 AND a.age < 60 AND b.age >= 18 AND c.age <> 40 AND a.name = 'x' \
        AND b.name <> 'y' OR c.person_id = $id AND NOT a.age = 42 \
        RETURN a.name AS an, b.name AS bn, count(*) AS n, sum(c.age) AS s ORDER BY n DESC LIMIT 10";
    println!("# parser-only (median of 20000)");
    println!("query\tlen\tparse_ns\tallocs\tbytes");
    for (name, text) in [
        ("Q0", "MATCH (n:Person) RETURN count(*)"),
        ("Q1", "MATCH (n:Person) WHERE n.age > 50 RETURN count(*)"),
        ("Q4", "MATCH (a:Person)-[:FRIEND_OF]->(b:Person)-[:FRIEND_OF]->(c:Person) RETURN count(*)"),
        ("LONG", long),
    ] {
        let mut ts = Vec::with_capacity(20000);
        let mut last = Cost::default();
        for _ in 0..20000 {
            let (r, c) = measure(|| parse_cypher_query(text));
            assert!(r.is_ok(), "{name} parses");
            std::hint::black_box(r.ok());
            ts.push(c.ns);
            last = c;
        }
        // Control: the SAME number of allocations of the same mean size,
        // allocated and freed with no parsing. This bounds how much of
        // `parse_ns` the allocator itself can account for.
        let (k, sz) = (last.allocs as usize, (last.bytes / last.allocs.max(1)) as usize);
        let mut cs = Vec::with_capacity(20000);
        for _ in 0..20000 {
            let t = Instant::now();
            let v: Vec<Vec<u8>> = (0..k).map(|_| Vec::with_capacity(sz)).collect();
            std::hint::black_box(&v);
            drop(v);
            cs.push(t.elapsed().as_nanos());
        }
        println!(
            "{name}\t{}\t{}\t{}\t{}\talloc-only control: {} ns",
            text.len(),
            median(ts),
            last.allocs,
            last.bytes,
            median(cs)
        );
    }
}

/// DataFusion's cost for a NEW parameter value. `$id` is substituted during
/// semantic analysis, so a new value re-runs bind, graph planning, DataFusion
/// planning, physical planning and execution; only parsing (and table
/// registration) is reused.
pub async fn df_param_rebind(n: usize, values: &[u32]) {
    let d = fixture(n);
    let (person, friend) = d.batches();
    let cfg = config();
    let text = "MATCH (n:Person) WHERE n.person_id = $id RETURN n.name";
    let ast = parse_cypher_query(text).unwrap();
    let (catalog, ctx) = catalog_and_ctx(&person, &friend);
    let catalog = Arc::new(catalog);
    let mut ts = Vec::with_capacity(values.len());
    for &v in values {
        let params: HashMap<String, serde_json::Value> = [("id".to_string(), serde_json::json!(v))].into();
        let t = Instant::now();
        let sem = SemanticAnalyzer::new(cfg.clone()).analyze(&ast, &params).unwrap();
        let lp = LogicalPlanner::new(&cfg).plan(&sem.ast).unwrap();
        let dfp = DataFusionPlanner::with_catalog(cfg.clone(), catalog.clone()).plan(&lp).unwrap();
        let phys = ctx.execute_logical_plan(dfp).await.unwrap().create_physical_plan().await.unwrap();
        let b = datafusion::physical_plan::collect(phys, ctx.task_ctx()).await.unwrap();
        ts.push(t.elapsed().as_nanos());
        let got: usize = b.iter().map(|b| b.num_rows()).sum();
        assert_eq!(got, usize::from((v as usize) < n), "DataFusion answer for id {v}");
    }
    println!(
        "datafusion Q5 n={n}: {} values, re-bind+plan+exec per value; median {} us",
        values.len(),
        fmt_us(median(ts))
    );
}
