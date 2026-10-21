//! D-QSP-0 remainder, slice 1: scalar aggregates under SQL NULL against
//! PostgreSQL.
//!
//! `COUNT(*)`, `COUNT(x)`, `SUM(x)`, `MIN(x)`, `MAX(x)` and `AVG(x)`, each over
//! a random three-valued `WHERE`, on random tables with two nullable `int`
//! columns. PostgreSQL is the oracle. An in-file Kleene evaluator answers the
//! same questions without PostgreSQL, so a Quack-vs-oracle disagreement is
//! visible even where PostgreSQL is missing a case.
//!
//! The data stays in column lanes (values plus validity bits) throughout; no
//! row structs are built, in the tables or in the oracle. The oracle folds the
//! TRUE positions straight into the aggregates.
//!
//! A negative control, the "naive SUM" (`Agg::SumI32` read as `Some(sum)`
//! always), is counted against PostgreSQL. It must disagree exactly where
//! no non-NULL `x` was selected (SQL answers NULL, the naive fold answers 0);
//! that count is reported and is not a failure.
//!
//! Needs `psql` and a server; connection from the usual `PG*` variables.
//! Without one it prints `SKIP` and exits 0, so nothing here gates CI.
//!
//! ```text
//! PGHOST=<socket dir> PGPORT=<port> PGUSER=<user> PGDATABASE=postgres \
//!   cargo run --release -p lance-graph-quack --example qsp_agg_pg_parity
//! ```

use lance_graph_mask_risc::{execute, words_for, LaneRef, Planes, Scratch, Value};
use lance_graph_quack::{
    avg_finish, lower, lower_avg, lower_sum, sum_finish, Agg, Cmp, Col, Filter, Mask, Query,
};
use std::io::Write;
use std::process::{Command, Stdio};

const X: Col = Col(0);
const Y: Col = Col(1);
const VX: Mask = Mask(0);
const VY: Mask = Mask(1);
const NULLABLE: [(Col, Mask); 2] = [(X, VX), (Y, VY)];

const TABLES: usize = 400;
const QUERIES_PER_TABLE: usize = 8;
const AGG_NAMES: [&str; 6] = [
    "count(*)", "count(x)", "sum(x)", "min(x)", "max(x)", "avg(x)",
];

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    fn below(&mut self, n: u64) -> u64 {
        self.next() % n
    }
}

/// Column lanes: values, plus validity bits (set = not NULL). A NULL row
/// carries a non-zero payload, so a path that reads it is caught.
struct Table {
    n: usize,
    xs: Vec<i32>,
    ys: Vec<i32>,
    vx: Vec<u64>,
    vy: Vec<u64>,
}

fn bit(w: &[u64], r: usize) -> bool {
    w[r / 64] >> (r % 64) & 1 == 1
}

fn table(rng: &mut Rng) -> Table {
    let n = 1 + rng.below(150) as usize;
    let px = [0, 30, 70, 100][rng.below(4) as usize];
    let py = [0, 30][rng.below(2) as usize];
    let mut t = Table {
        n,
        xs: Vec::with_capacity(n),
        ys: Vec::with_capacity(n),
        vx: vec![0; words_for(n)],
        vy: vec![0; words_for(n)],
    };
    for r in 0..n {
        let val = |rng: &mut Rng| rng.below(9) as i32 - 4;
        let payload = |rng: &mut Rng| 1 + rng.below(4) as i32;
        if rng.below(100) < px {
            t.xs.push(payload(rng));
        } else {
            t.xs.push(val(rng));
            t.vx[r / 64] |= 1 << (r % 64);
        }
        if rng.below(100) < py {
            t.ys.push(payload(rng));
        } else {
            t.ys.push(val(rng));
            t.vy[r / 64] |= 1 << (r % 64);
        }
    }
    t
}

/// The `WHERE` as written: the source of the SQL text, the Kleene oracle and
/// the Quack filter, so all three read one tree.
enum Node {
    Cmp(u8, u8, i32), // column (0 = x, 1 = y), operator index, constant
    IsNull(u8),
    IsNotNull(u8),
    Not(Box<Node>),
    And(Vec<Node>),
    Or(Vec<Node>),
}

const OPS_SQL: [&str; 6] = ["=", "<>", "<", "<=", ">", ">="];

fn col_name(c: u8) -> &'static str {
    if c == 0 {
        "x"
    } else {
        "y"
    }
}

fn leaf(rng: &mut Rng) -> Node {
    let c = rng.below(2) as u8;
    match rng.below(10) {
        0..=5 => Node::Cmp(c, rng.below(6) as u8, rng.below(11) as i32 - 5),
        6 | 7 => Node::IsNull(c),
        _ => Node::IsNotNull(c),
    }
}

fn node(rng: &mut Rng, depth: u32) -> Node {
    if depth == 0 || rng.below(4) == 0 {
        return leaf(rng);
    }
    match rng.below(3) {
        0 => Node::Not(Box::new(node(rng, depth - 1))),
        k => {
            let cs = (0..2 + rng.below(2))
                .map(|_| node(rng, depth - 1))
                .collect();
            if k == 1 {
                Node::And(cs)
            } else {
                Node::Or(cs)
            }
        }
    }
}

fn sql(n: &Node) -> String {
    match n {
        Node::Cmp(c, op, k) => format!("{} {} {k}", col_name(*c), OPS_SQL[*op as usize]),
        Node::IsNull(c) => format!("{} IS NULL", col_name(*c)),
        Node::IsNotNull(c) => format!("{} IS NOT NULL", col_name(*c)),
        Node::Not(i) => format!("(NOT {})", sql(i)),
        Node::And(cs) => format!("({})", cs.iter().map(sql).collect::<Vec<_>>().join(" AND ")),
        Node::Or(cs) => format!("({})", cs.iter().map(sql).collect::<Vec<_>>().join(" OR ")),
    }
}

fn mask_of(c: u8) -> Mask {
    if c == 0 {
        VX
    } else {
        VY
    }
}

fn to_filter(n: &Node) -> Filter {
    match n {
        Node::Cmp(c, op, k) => {
            let col = if *c == 0 { X } else { Y };
            Filter::cmp(
                col,
                match op {
                    0 => Cmp::EqI32(*k),
                    1 => Cmp::NeI32(*k),
                    2 => Cmp::LtI32(*k),
                    3 => Cmp::LeI32(*k),
                    4 => Cmp::GtI32(*k),
                    _ => Cmp::GeI32(*k),
                },
            )
        }
        Node::IsNull(c) => Filter::is_null(mask_of(*c)),
        Node::IsNotNull(c) => Filter::is_not_null(mask_of(*c)),
        Node::Not(i) => Filter::negate(to_filter(i)),
        Node::And(cs) => Filter::and(cs.iter().map(to_filter)),
        Node::Or(cs) => Filter::or(cs.iter().map(to_filter)),
    }
}

/// Kleene truth value.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Tv {
    T,
    F,
    U,
}

fn eval(n: &Node, t: &Table, r: usize) -> Tv {
    let tv = |b: bool| if b { Tv::T } else { Tv::F };
    match n {
        Node::Cmp(c, op, k) => {
            let (lane, valid) = if *c == 0 {
                (&t.xs, &t.vx)
            } else {
                (&t.ys, &t.vy)
            };
            if !bit(valid, r) {
                return Tv::U;
            }
            let v = lane[r];
            tv(match op {
                0 => v == *k,
                1 => v != *k,
                2 => v < *k,
                3 => v <= *k,
                4 => v > *k,
                _ => v >= *k,
            })
        }
        Node::IsNull(c) => tv(!bit(if *c == 0 { &t.vx } else { &t.vy }, r)),
        Node::IsNotNull(c) => tv(bit(if *c == 0 { &t.vx } else { &t.vy }, r)),
        Node::Not(i) => match eval(i, t, r) {
            Tv::T => Tv::F,
            Tv::F => Tv::T,
            Tv::U => Tv::U,
        },
        Node::And(cs) => {
            let mut unknown = false;
            for c in cs {
                match eval(c, t, r) {
                    Tv::F => return Tv::F,
                    Tv::U => unknown = true,
                    Tv::T => {}
                }
            }
            if unknown {
                Tv::U
            } else {
                Tv::T
            }
        }
        Node::Or(cs) => {
            let mut unknown = false;
            for c in cs {
                match eval(c, t, r) {
                    Tv::T => return Tv::T,
                    Tv::U => unknown = true,
                    Tv::F => {}
                }
            }
            if unknown {
                Tv::U
            } else {
                Tv::F
            }
        }
    }
}

/// The six answers for one query.
#[derive(Debug, Clone, Copy)]
struct Answers {
    cstar: i64,
    cx: i64,
    sum: Option<i64>,
    min: Option<i32>,
    max: Option<i32>,
    avg: Option<f64>,
}

impl Answers {
    /// Indices into `AGG_NAMES` where `self` and `other` differ.
    fn diff(&self, other: &Answers) -> Vec<usize> {
        // PostgreSQL's `avg(int)` is numeric, cast to float8 in the query;
        // Quack divides i64 by u64 in f64. Same value up to rounding, so AVG
        // is compared with a relative tolerance; everything else is exact.
        let avg_eq = match (self.avg, other.avg) {
            (None, None) => true,
            (Some(a), Some(b)) => (a - b).abs() <= 1e-12 * a.abs().max(b.abs()).max(1.0),
            _ => false,
        };
        let mut d = Vec::new();
        if self.cstar != other.cstar {
            d.push(0);
        }
        if self.cx != other.cx {
            d.push(1);
        }
        if self.sum != other.sum {
            d.push(2);
        }
        if self.min != other.min {
            d.push(3);
        }
        if self.max != other.max {
            d.push(4);
        }
        if !avg_eq {
            d.push(5);
        }
        d
    }
}

/// The Kleene oracle: fold the TRUE positions straight into the aggregates.
fn oracle(n: &Node, t: &Table) -> Answers {
    let (mut cstar, mut cx, mut sum) = (0i64, 0i64, 0i64);
    let (mut min, mut max): (Option<i32>, Option<i32>) = (None, None);
    for r in 0..t.n {
        if eval(n, t, r) != Tv::T {
            continue;
        }
        cstar += 1;
        if bit(&t.vx, r) {
            let v = t.xs[r];
            cx += 1;
            sum += i64::from(v);
            min = Some(min.map_or(v, |m| m.min(v)));
            max = Some(max.map_or(v, |m| m.max(v)));
        }
    }
    Answers {
        cstar,
        cx,
        sum: (cx != 0).then_some(sum),
        min,
        max,
        avg: (cx != 0).then(|| sum as f64 / cx as f64),
    }
}

/// The Quack side: every aggregate through `lower` + `execute`. The second
/// value is the naive `SUM` (`Agg::SumI32` read as `Some` always).
fn quack(n: &Node, t: &Table) -> (Answers, Option<i64>) {
    let masks: [&[u64]; 2] = [&t.vx, &t.vy];
    let lanes = [LaneRef::I32(&t.xs), LaneRef::I32(&t.ys)];
    let pl = Planes {
        n_rows: t.n,
        masks: &masks,
        lanes: &lanes,
    };
    let run = |p: &lance_graph_mask_risc::Program| -> Value {
        let mut s = Scratch::for_program(p, t.n).expect("carves");
        execute(p, &pl, &mut s, None).expect("runs")
    };
    let run_q = |filter: &Filter, agg: Agg| -> Value {
        run(&lower(&Query {
            filter: filter.clone(),
            agg,
        })
        .expect("lowers"))
    };
    let count = |v: Value| match v {
        Value::Count(c) => c,
        other => panic!("expected Count, got {other:?}"),
    };

    let w = to_filter(n).sql_where(&NULLABLE);
    let gated = Filter::and([w.clone(), Filter::is_not_null(VX)]);

    let cstar = count(run_q(&w, Agg::Count)) as i64;
    let cx = count(run_q(&gated, Agg::Count)) as i64;

    let plan = lower_sum(&w, X, Some(VX)).expect("lowers");
    let (Value::SumI64(s), Value::Count(c)) = (run(&plan.sum), run(&plan.count)) else {
        panic!("SUM plan shapes")
    };
    let sum = sum_finish(s, c as u64);

    let opt = |v: Value| match v {
        Value::OptI32(o) => o,
        other => panic!("expected OptI32, got {other:?}"),
    };
    let min = opt(run_q(&gated, Agg::MinI32(X)));
    let max = opt(run_q(&gated, Agg::MaxI32(X)));

    let plan = lower_avg(&gated, X).expect("lowers");
    let (Value::SumI64(s), Value::Count(c)) = (run(&plan.sum), run(&plan.count)) else {
        panic!("AVG plan shapes")
    };
    let avg = avg_finish(s, c as u64);

    let Value::SumI64(naive) = run_q(&gated, Agg::SumI32(X)) else {
        panic!("naive SUM shape")
    };
    (
        Answers {
            cstar,
            cx,
            sum,
            min,
            max,
            avg,
        },
        Some(naive),
    )
}

fn lit(v: Option<i32>) -> String {
    v.map_or("NULL".into(), |x| x.to_string())
}

fn field<T: std::str::FromStr>(s: &str) -> Option<T> {
    (!s.is_empty()).then(|| {
        s.parse()
            .unwrap_or_else(|_| panic!("unparsable field {s:?}"))
    })
}

fn parse_pg(line: &str) -> Answers {
    let f: Vec<&str> = line.split('|').collect();
    assert_eq!(f.len(), 6, "six fields per result row: {line:?}");
    Answers {
        cstar: field(f[0]).expect("count(*) is never NULL"),
        cx: field(f[1]).expect("count(x) is never NULL"),
        sum: field(f[2]),
        min: field(f[3]),
        max: field(f[4]),
        avg: field(f[5]),
    }
}

fn main() {
    let mut rng = Rng(0x00D1_5EED_0A66_0001);
    let tables: Vec<Table> = (0..TABLES).map(|_| table(&mut rng)).collect();

    // (table index, filter tree, SQL text)
    let mut queries: Vec<(usize, Node, String)> = Vec::new();
    for k in 0..TABLES {
        for _ in 0..QUERIES_PER_TABLE {
            let n = node(&mut rng, 3);
            let s = sql(&n);
            queries.push((k, n, s));
        }
        // Matches nothing, for sure.
        let n = Node::And(vec![Node::IsNull(0), Node::IsNotNull(0)]);
        queries.push((k, n, "(x IS NULL AND x IS NOT NULL)".to_string()));
    }

    let mut script = String::new();
    for (k, t) in tables.iter().enumerate() {
        script.push_str(&format!("CREATE TEMP TABLE t{k} (x int, y int);\n"));
        let rows: Vec<String> = (0..t.n)
            .map(|r| {
                format!(
                    "({}, {})",
                    lit(bit(&t.vx, r).then_some(t.xs[r])),
                    lit(bit(&t.vy, r).then_some(t.ys[r]))
                )
            })
            .collect();
        script.push_str(&format!("INSERT INTO t{k} VALUES {};\n", rows.join(",")));
    }
    for (k, _, s) in &queries {
        script.push_str(&format!(
            "SELECT count(*), count(x), sum(x), min(x), max(x), avg(x)::float8 \
             FROM t{k} WHERE {s};\n"
        ));
    }

    let out = Command::new("psql")
        .args(["-X", "-q", "-At", "-F|", "-v", "ON_ERROR_STOP=1", "-f", "-"])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .and_then(|mut child| {
            // Write on a separate thread: psql answers while it reads, so a
            // parent that wrote everything first would block once the stdout
            // pipe filled. A refused connection closes the pipe early; the
            // exit status and stderr below say why.
            let mut stdin = child.stdin.take().expect("piped stdin");
            let writer = std::thread::spawn(move || {
                let _ = stdin.write_all(script.as_bytes());
            });
            let out = child.wait_with_output();
            let _ = writer.join();
            out
        });
    let out = match out {
        Ok(o) if o.status.success() => o,
        Ok(o) => {
            println!(
                "SKIP: psql failed: {}",
                String::from_utf8_lossy(&o.stderr).trim()
            );
            return;
        }
        Err(e) => {
            println!("SKIP: could not run psql ({e})");
            return;
        }
    };
    let lines: Vec<&str> = std::str::from_utf8(&out.stdout)
        .expect("utf-8")
        .lines()
        .filter(|l| !l.is_empty())
        .collect();
    assert_eq!(
        lines.len(),
        queries.len(),
        "one result row per query (got {} lines for {} queries)",
        lines.len(),
        queries.len()
    );

    // Mismatch counts per aggregate, per arm.
    let mut quack_bad = [0usize; 6];
    let mut oracle_bad = [0usize; 6];
    let mut shown = 0usize;
    let (mut naive_bad, mut empty, mut all_null, mut real_zero) = (0, 0, 0, 0);

    for ((k, n, s), line) in queries.iter().zip(&lines) {
        let t = &tables[*k];
        let pg = parse_pg(line);
        let (q, naive) = quack(n, t);
        let o = oracle(n, t);

        let dq = pg.diff(&q);
        let dor = pg.diff(&o);
        for &i in &dq {
            quack_bad[i] += 1;
        }
        for &i in &dor {
            oracle_bad[i] += 1;
        }
        if (!dq.is_empty() || !dor.is_empty()) && shown < 5 {
            shown += 1;
            println!(
                "MISMATCH t{k} WHERE {s}\n  postgres {pg:?}\n  quack    {q:?} (differs: {:?})\n  oracle   {o:?} (differs: {:?})",
                dq.iter().map(|&i| AGG_NAMES[i]).collect::<Vec<_>>(),
                dor.iter().map(|&i| AGG_NAMES[i]).collect::<Vec<_>>(),
            );
        }

        // Negative control: the naive fold reads as Some(sum) always.
        naive_bad += usize::from(naive != pg.sum);
        empty += usize::from(pg.cstar == 0);
        all_null += usize::from(pg.cstar > 0 && pg.cx == 0);
        real_zero += usize::from(pg.sum == Some(0));
    }

    let qm: usize = quack_bad.iter().sum();
    let om: usize = oracle_bad.iter().sum();
    let version = Command::new("psql")
        .args(["-X", "-tA", "-c", "SHOW server_version"])
        .output()
        .map(|o| o.stdout)
        .unwrap_or_default();
    println!(
        "PostgreSQL {}",
        std::str::from_utf8(&version).unwrap_or("?").trim()
    );
    for (i, name) in AGG_NAMES.iter().enumerate() {
        println!(
            "  {name}: quack mismatches {}, oracle mismatches {}",
            quack_bad[i], oracle_bad[i]
        );
    }
    println!(
        "qsp_agg_pg_parity: {} queries, quack mismatches {qm}, oracle mismatches {om}, \
         naive SUM disagreements {naive_bad} (empty {empty}, all-NULL {all_null}, \
         real-zero sums {real_zero})",
        queries.len()
    );

    // Anti-vacuity: the corpus must contain the cases this slice is about.
    assert!(empty > 0, "no query selected zero rows");
    assert!(all_null > 0, "no query selected rows with only NULL x");
    assert!(real_zero > 0, "no query had a real SUM of zero");
    assert_eq!(
        naive_bad,
        empty + all_null,
        "the naive SUM must differ exactly where no non-NULL x was selected"
    );
    assert_eq!(qm, 0, "Quack disagrees with PostgreSQL");
    assert_eq!(om, 0, "the Kleene oracle disagrees with PostgreSQL");
}
