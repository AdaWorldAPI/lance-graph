//! D-RFT-0 / D-QSP-0: set normalization against PostgreSQL's three-valued
//! `WHERE`.
//!
//! Random [`SetExpr`]s over a table with nullable lanes are rendered to SQL.
//! PostgreSQL counts the rows each `WHERE` keeps for the expression as
//! written, after [`normalize_kleene`], and after the two-valued
//! [`normalize`]. A Kleene evaluator in this file counts the same rows
//! without PostgreSQL: it evaluates whole TRUE/FALSE bit planes over the
//! column lanes and counts by popcount, never per row. Expected: as written = Kleene = PostgreSQL for every
//! expression; the two-valued form differs on some.
//!
//! Needs `psql` and a server; connection from the usual `PG*` variables.
//! Without one it prints `SKIP` and exits 0, so nothing here gates CI.
//!
//! ```text
//! PGHOST=/var/tmp/k3pg PGPORT=54329 PGUSER=k3 PGDATABASE=postgres \
//!   cargo run --release -p lance-graph-contract --example k3_pg_parity
//! ```

use lance_graph_contract::set_algebra::{normalize, normalize_kleene, Cmp, Res, SetExpr};
use std::io::Write;
use std::process::{Command, Stdio};

const ROWS: usize = 200;
const EXPRS: usize = 2000;

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

const WORDS: usize = ROWS.div_ceil(64);

/// A column as lanes, never as rows: a value lane plus a validity plane per
/// nullable column (`a0`, `a1` int, `u0` int code) and a bit plane per
/// two-valued boolean column (`s0`, `s1`; a `Sym` is never NULL).
struct Table {
    a: [Vec<i32>; 2],
    a_valid: [[u64; WORDS]; 2],
    u: Vec<u32>,
    u_valid: [u64; WORDS],
    s: [[u64; WORDS]; 2],
}

fn set(plane: &mut [u64; WORDS], i: usize) {
    plane[i / 64] |= 1 << (i % 64);
}

fn has(plane: &[u64; WORDS], i: usize) -> bool {
    plane[i / 64] >> (i % 64) & 1 == 1
}

fn table(rng: &mut Rng) -> Table {
    let mut t = Table {
        a: [vec![0; ROWS], vec![0; ROWS]],
        a_valid: [[0; WORDS]; 2],
        u: vec![0; ROWS],
        u_valid: [0; WORDS],
        s: [[0; WORDS]; 2],
    };
    for i in 0..ROWS {
        for l in 0..2 {
            if rng.below(4) != 0 {
                t.a[l][i] = rng.below(9) as i32 - 4;
                set(&mut t.a_valid[l], i);
            }
        }
        if rng.below(4) != 0 {
            t.u[i] = rng.below(3) as u32;
            set(&mut t.u_valid, i);
        }
        for l in 0..2 {
            if rng.below(2) == 0 {
                set(&mut t.s[l], i);
            }
        }
    }
    t
}

fn leaf(rng: &mut Rng) -> SetExpr {
    const OPS: [Cmp; 6] = [Cmp::Lt, Cmp::Le, Cmp::Gt, Cmp::Ge, Cmp::Eq, Cmp::Ne];
    match rng.below(5) {
        0 => SetExpr::Sym(rng.below(2) as u32),
        1 | 2 => SetExpr::i32(
            rng.below(2) as u16,
            OPS[rng.below(6) as usize],
            rng.below(11) as i32 - 5,
        ),
        3 => SetExpr::Res(Res::U32 {
            lane: 0,
            eq: rng.below(2) == 0,
            v: rng.below(3) as u32,
        }),
        _ => SetExpr::i32(rng.below(2) as u16, Cmp::Ge, i32::MIN),
    }
}

fn expr(rng: &mut Rng, depth: u32) -> SetExpr {
    if depth == 0 || rng.below(4) == 0 {
        return leaf(rng);
    }
    match rng.below(3) {
        0 => SetExpr::negated(expr(rng, depth - 1)),
        k => {
            let cs = (0..1 + rng.below(4))
                .map(|_| expr(rng, depth - 1))
                .collect();
            if k == 1 {
                SetExpr::And(cs)
            } else {
                SetExpr::Or(cs)
            }
        }
    }
}

fn op_sql(op: Cmp) -> &'static str {
    match op {
        Cmp::Lt => "<",
        Cmp::Le => "<=",
        Cmp::Gt => ">",
        Cmp::Ge => ">=",
        Cmp::Eq => "=",
        Cmp::Ne => "<>",
    }
}

fn sql(e: &SetExpr) -> String {
    let junction = |cs: &[SetExpr], sep: &str, unit: &str| {
        if cs.is_empty() {
            return unit.to_string();
        }
        let parts: Vec<String> = cs.iter().map(sql).collect();
        format!("({})", parts.join(sep))
    };
    match e {
        SetExpr::All => "TRUE".into(),
        SetExpr::Empty => "FALSE".into(),
        SetExpr::Sym(i) => format!("s{i}"),
        // `t` as a bigint literal: `-2147483648` would otherwise parse as
        // the negation of an out-of-range int.
        SetExpr::Res(Res::I32 { lane, op, t }) => {
            format!("(a{lane} {} {}::bigint)", op_sql(*op), i64::from(*t))
        }
        SetExpr::Res(Res::U32 { lane, eq, v }) => {
            format!("(u{lane} {} {v})", if *eq { "=" } else { "<>" })
        }
        SetExpr::Not(c) => format!("(NOT {})", sql(c)),
        SetExpr::And(cs) => junction(cs, " AND ", "TRUE"),
        SetExpr::Or(cs) => junction(cs, " OR ", "FALSE"),
    }
}

/// Kleene evaluation as two planes: `(T, F)`, where a bit set in neither
/// plane is UNKNOWN. A leaf is one compare per lane slot into a mask; NOT
/// swaps the planes; AND / OR combine whole words. Nothing is decoded into
/// rows, and the oracle's answer is a popcount of the TRUE plane.
fn kleene(t: &Table, e: &SetExpr) -> ([u64; WORDS], [u64; WORDS]) {
    let all = {
        let mut p = [0u64; WORDS];
        (0..ROWS).for_each(|i| set(&mut p, i));
        p
    };
    match e {
        SetExpr::All => (all, [0; WORDS]),
        SetExpr::Empty => ([0; WORDS], all),
        SetExpr::Sym(i) => {
            let s = t.s[*i as usize];
            (s, std::array::from_fn(|w| all[w] & !s[w]))
        }
        SetExpr::Res(r) => {
            let (mut tp, mut fp) = ([0u64; WORDS], [0u64; WORDS]);
            for i in 0..ROWS {
                let hit = match r {
                    Res::I32 { lane, op, t: c } => {
                        if !has(&t.a_valid[*lane as usize], i) {
                            continue;
                        }
                        let x = t.a[*lane as usize][i];
                        match op {
                            Cmp::Lt => x < *c,
                            Cmp::Le => x <= *c,
                            Cmp::Gt => x > *c,
                            Cmp::Ge => x >= *c,
                            Cmp::Eq => x == *c,
                            Cmp::Ne => x != *c,
                        }
                    }
                    Res::U32 { eq, v, .. } => {
                        if !has(&t.u_valid, i) {
                            continue;
                        }
                        (t.u[i] == *v) == *eq
                    }
                };
                set(if hit { &mut tp } else { &mut fp }, i);
            }
            (tp, fp)
        }
        SetExpr::Not(c) => {
            let (tp, fp) = kleene(t, c);
            (fp, tp)
        }
        SetExpr::And(cs) => cs.iter().fold((all, [0; WORDS]), |(at, af), c| {
            let (ct, cf) = kleene(t, c);
            (
                std::array::from_fn(|w| at[w] & ct[w]),
                std::array::from_fn(|w| af[w] | cf[w]),
            )
        }),
        SetExpr::Or(cs) => cs.iter().fold(([0; WORDS], all), |(at, af), c| {
            let (ct, cf) = kleene(t, c);
            (
                std::array::from_fn(|w| at[w] | ct[w]),
                std::array::from_fn(|w| af[w] & cf[w]),
            )
        }),
    }
}

fn lit<T: std::fmt::Display>(valid: bool, v: T) -> String {
    if valid {
        v.to_string()
    } else {
        "NULL".into()
    }
}

fn main() {
    let mut rng = Rng(0x00D1_5EED_0003_0001);
    let table = table(&mut rng);
    let exprs: Vec<SetExpr> = (0..EXPRS).map(|_| expr(&mut rng, 4)).collect();

    let mut script = String::from(
        "CREATE TEMP TABLE t (a0 int, a1 int, u0 int, s0 bool, s1 bool);\nINSERT INTO t VALUES\n",
    );
    // Rendering the INSERT is the one place lanes are read side by side: it
    // is the input handed to PostgreSQL, not a comparison.
    let values: Vec<String> = (0..ROWS)
        .map(|i| {
            format!(
                "({}, {}, {}, {}, {})",
                lit(has(&table.a_valid[0], i), table.a[0][i]),
                lit(has(&table.a_valid[1], i), table.a[1][i]),
                lit(has(&table.u_valid, i), table.u[i]),
                has(&table.s[0], i),
                has(&table.s[1], i)
            )
        })
        .collect();
    script.push_str(&values.join(",\n"));
    script.push_str(";\n");
    for e in &exprs {
        script.push_str(&format!(
            "SELECT count(*) FILTER (WHERE {}), count(*) FILTER (WHERE {}), \
             count(*) FILTER (WHERE {}) FROM t;\n",
            sql(e),
            sql(&normalize_kleene(e)),
            sql(&normalize(e))
        ));
    }

    let out = Command::new("psql")
        .args(["-X", "-q", "-tA", "-F,", "-v", "ON_ERROR_STOP=1", "-f", "-"])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .and_then(|mut child| {
            // Write on a separate thread: psql answers while it reads, so a
            // parent that wrote everything before reading would block once
            // the stdout pipe filled. A server that refuses the connection
            // closes the pipe early; the exit status and stderr below say
            // why, so a write error is moot.
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
    assert_eq!(lines.len(), EXPRS, "one result row per expression");

    let (mut kleene_ok, mut two_valued_wrong, mut rewritten) = (0, 0, 0);
    for (e, line) in exprs.iter().zip(&lines) {
        let n: Vec<usize> = line.split(',').map(|x| x.parse().expect("count")).collect();
        let oracle: usize = kleene(&table, e)
            .0
            .iter()
            .map(|w| w.count_ones() as usize)
            .sum();
        assert_eq!(n[0], oracle, "PostgreSQL vs the Kleene oracle: {e:?}");
        assert_eq!(n[1], n[0], "normalize_kleene changed the count: {e:?}");
        kleene_ok += 1;
        two_valued_wrong += usize::from(n[2] != n[0]);
        rewritten += usize::from(normalize_kleene(e) != *e);
    }
    println!(
        "PostgreSQL {}: {EXPRS} expressions over {ROWS} rows; as written = Kleene oracle = \
         normalize_kleene on {kleene_ok}/{EXPRS} ({rewritten} rewritten); two-valued \
         normalize wrong on {two_valued_wrong}",
        std::str::from_utf8(
            &Command::new("psql")
                .args(["-X", "-tA", "-c", "SHOW server_version"])
                .output()
                .map(|o| o.stdout)
                .unwrap_or_default()
        )
        .unwrap_or("?")
        .trim()
    );
    assert!(two_valued_wrong > 0, "the run never reached a lane cover");
}
