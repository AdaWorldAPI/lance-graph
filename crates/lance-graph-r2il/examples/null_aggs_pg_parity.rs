//! D-RFT-1 correction: `bool_or`, `bool_and` and `COUNT(DISTINCT)` over
//! nullable columns, r2il procedures against PostgreSQL.
//!
//! Random tables with a nullable boolean `b` and a nullable key-ordered key
//! `k` are aggregated under random selections `x < t`. The procedures in
//! `lance_graph_r2il::procs` run through `FoldDialect` and mask-risc; PostgreSQL
//! answers `bool_or(b)`, `bool_and(b)` and `count(DISTINCT k)` for the same
//! rows. Values and NULL states are compared, never a zero-coalesced scalar.
//!
//! At a NULL row the payload of `b` and `k` is random, so a procedure that
//! reads it is caught. A disable arm runs the same aggregates with the
//! validity plane replaced by every row (the defect #1681 shipped) and must
//! differ from PostgreSQL somewhere.
//!
//! Needs `psql` and a server, connection from the usual `PG*` variables.
//! Without one it prints `SKIP` and exits 0: a lab instrument, not CI parity.
//!
//! ```text
//! PGHOST=<socket dir> PGPORT=<port> PGUSER=<user> PGDATABASE=postgres \
//!   cargo run --release -p lance-graph-r2il --example null_aggs_pg_parity
//! ```

use lance_graph_mask_risc::{words_for, Foreign, LaneRef, Planes};
use lance_graph_r2il::procs::{bool_and, bool_or, count_distinct_sorted, select_lt, Select};
use lance_graph_r2il::{FoldDialect, Val};
use std::io::Write;
use std::process::{Command, Stdio};

const ROWS: usize = 120;
const TABLES: usize = 60;
const SELECTIONS: [i32; 5] = [0, 1, 5, 12, 21];

// Lanes: x (i32, selection), k (u32, key-ordered). Planes: b valid, b TRUE,
// k valid, every row.
const X: u16 = 0;
const K: u16 = 1;
const VB: u16 = 0;
const TB: u16 = 1;
const VK: u16 = 2;
const ALL: u16 = 3;

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

struct Table {
    x: Vec<i32>,
    k: Vec<u32>,
    planes: [Vec<u64>; 4],
}

fn set(p: &mut [u64], r: usize) {
    p[r / 64] |= 1 << (r % 64);
}

fn bit(p: &[u64], r: usize) -> bool {
    p[r / 64] >> (r % 64) & 1 == 1
}

/// `b` is NULL on about 40% of rows, more on some tables so all-NULL
/// selections occur. `k` is key-ordered: it never decreases, NULL rows keep
/// the running key as payload (so the lane stays ordered).
fn table(rng: &mut Rng) -> Table {
    let w = words_for(ROWS);
    let null_b = 2 + rng.below(6);
    let mut t = Table {
        x: (0..ROWS).map(|_| rng.below(25) as i32).collect(),
        k: vec![0; ROWS],
        planes: std::array::from_fn(|_| vec![0; w]),
    };
    let mut key = 0u32;
    for r in 0..ROWS {
        if rng.below(10) >= null_b {
            set(&mut t.planes[VB as usize], r);
        }
        if rng.below(2) == 0 {
            set(&mut t.planes[TB as usize], r);
        }
        if rng.below(3) == 0 {
            key += 1;
        }
        t.k[r] = key;
        if rng.below(4) != 0 {
            set(&mut t.planes[VK as usize], r);
        }
        set(&mut t.planes[ALL as usize], r);
    }
    t
}

/// (value, present) for a two-scalar procedure.
fn two(t: &Table, body: &Select<'_>) -> Option<i64> {
    let masks: Vec<&[u64]> = t.planes.iter().map(Vec::as_slice).collect();
    let lanes = [LaneRef::I32(&t.x), LaneRef::U32(&t.k)];
    let planes = Planes {
        n_rows: ROWS,
        masks: &masks,
        lanes: &lanes,
    };
    let mut d = FoldDialect::new(&planes, &Foreign::NONE);
    let mut stack = vec![];
    body(&mut d, &mut stack).expect("procedure");
    match stack.as_slice() {
        [Val::Scalar(v), Val::Scalar(1)] => Some(*v),
        [Val::Scalar(0), Val::Scalar(0)] => None,
        [Val::Scalar(n)] => Some(*n),
        other => panic!("unexpected result {other:?}"),
    }
}

/// The three aggregates for `x < sel`, with `vb`/`vk` as the validity planes.
fn aggs(t: &Table, sel: i32, vb: u16, vk: u16) -> [Option<i64>; 3] {
    let select = move |d: &mut FoldDialect<'_>, s: &mut Vec<Val>| select_lt(d, s, X, sel);
    [
        two(t, &|d, s| bool_or(d, s, &select, vb, TB)),
        two(t, &|d, s| bool_and(d, s, &select, vb, TB)),
        two(t, &|d, s| {
            select_lt(d, s, X, sel)?;
            count_distinct_sorted(d, s, K, vk)
        }),
    ]
}

fn pg_value(s: &str) -> Option<i64> {
    match s {
        "" => None,
        "t" => Some(1),
        "f" => Some(0),
        n => Some(n.parse().expect("number")),
    }
}

fn psql(script: String) -> Result<Vec<u8>, String> {
    let out = Command::new("psql")
        .args(["-X", "-q", "-tA", "-F,", "-v", "ON_ERROR_STOP=1", "-f", "-"])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .and_then(|mut child| {
            let mut stdin = child.stdin.take().expect("piped stdin");
            let writer = std::thread::spawn(move || {
                let _ = stdin.write_all(script.as_bytes());
            });
            let out = child.wait_with_output();
            let _ = writer.join();
            out
        })
        .map_err(|e| format!("could not run psql ({e})"))?;
    if out.status.success() {
        Ok(out.stdout)
    } else {
        Err(format!(
            "psql failed: {}",
            String::from_utf8_lossy(&out.stderr).trim()
        ))
    }
}

fn main() {
    let mut rng = Rng(0x00D1_5EED_0011_0001);
    let tables: Vec<Table> = (0..TABLES).map(|_| table(&mut rng)).collect();

    let mut script = String::new();
    for (i, t) in tables.iter().enumerate() {
        script.push_str(&format!(
            "CREATE TEMP TABLE t{i} (id int, x int, b boolean, k int);\nINSERT INTO t{i} VALUES\n"
        ));
        let values: Vec<String> = (0..ROWS)
            .map(|r| {
                let b = if bit(&t.planes[VB as usize], r) {
                    bit(&t.planes[TB as usize], r).to_string()
                } else {
                    "NULL".into()
                };
                let k = if bit(&t.planes[VK as usize], r) {
                    t.k[r].to_string()
                } else {
                    "NULL".into()
                };
                format!("({r}, {}, {b}, {k})", t.x[r])
            })
            .collect();
        script.push_str(&values.join(",\n"));
        script.push_str(";\n");
        for sel in SELECTIONS {
            script.push_str(&format!(
                "SELECT bool_or(b), bool_and(b), count(DISTINCT k) FROM t{i} WHERE x < {sel};\n"
            ));
        }
    }
    let stdout = match psql(script) {
        Ok(o) => o,
        Err(why) => {
            println!("SKIP: {why}");
            return;
        }
    };
    let lines: Vec<&str> = std::str::from_utf8(&stdout)
        .expect("utf-8")
        .lines()
        .filter(|l| !l.is_empty())
        .collect();
    assert_eq!(lines.len(), TABLES * SELECTIONS.len(), "one row per query");

    let (mut nulls, mut trues_with_null, mut naive_wrong) = (0usize, 0usize, 0usize);
    let mut line = lines.iter();
    for t in &tables {
        for sel in SELECTIONS {
            let cols: Vec<&str> = line.next().expect("row").split(',').collect();
            let pg = [pg_value(cols[0]), pg_value(cols[1]), pg_value(cols[2])];
            let ours = aggs(t, sel, VB, VK);
            assert_eq!(ours, pg, "x < {sel}: r2il {ours:?} vs PostgreSQL {pg:?}");
            nulls += usize::from(pg[0].is_none());
            let has_null_selected =
                (0..ROWS).any(|r| t.x[r] < sel && !bit(&t.planes[VB as usize], r));
            trues_with_null += usize::from(pg[1] == Some(1) && has_null_selected);
            naive_wrong += usize::from(aggs(t, sel, ALL, ALL) != pg);
        }
    }
    let version = Command::new("psql")
        .args(["-X", "-tA", "-c", "SHOW server_version"])
        .output()
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
        .unwrap_or_default();
    let total = TABLES * SELECTIONS.len();
    println!(
        "PostgreSQL {version}: {total} queries ({nulls} with NULL bool aggregates, \
         {trues_with_null} bool_and TRUE over a NULL row). r2il bool_or, bool_and and \
         count(DISTINCT) equal PostgreSQL on value and NULL state for all. Disabled \
         (validity = every row): differs on {naive_wrong}."
    );
    assert!(nulls > 0, "no selection produced a NULL aggregate");
    assert!(trues_with_null > 0, "no bool_and TRUE over a NULL row");
    assert!(naive_wrong > 0, "the disable arm was not detected");
}
