//! D-RFT-0 / D-QSP-0: Quack's three-valued `WHERE` against PostgreSQL, row
//! for row.
//!
//! Random [`SetExpr`]s over a table with nullable lanes are mapped to Quack
//! [`Filter`]s and executed through the existing engine: [`Filter::sql_where`]
//! (TRUE / FALSE lowering over each column's validity plane), then [`lower`]
//! and [`lower_fused`], then mask-risc [`execute_into`] with the `Keep`
//! terminal. PostgreSQL is the independent semantic reference: for each
//! expression it returns the rows where the predicate is TRUE and the rows
//! where it is FALSE, as two bit strings in `id` order.
//!
//! Equality is checked on the masks, not on counts: two masks are equal when
//! mask-risc's `Xor` followed by the `Any` terminal says no bit differs. Both
//! the TRUE mask (`sql_where(e)`) and the FALSE mask (`sql_where(NOT e)`) are
//! compared, so an UNKNOWN row (in neither) cannot hide. `normalize_kleene(e)`
//! is held to its own contract, which is WHERE-terminal: its TRUE rows must
//! match exactly; how many expressions it moves FALSE rows on is reported,
//! because NOT over its result is refused, not promised. The two-valued `normalize(e)` must differ on some
//! expressions, which shows the generator reaches the rewrites `normalize_kleene`
//! exists to refuse.
//!
//! Needs `psql` and a server, connection from the usual `PG*` variables.
//! Without one it prints `SKIP` and exits 0, so nothing here gates CI.
//!
//! ```text
//! PGHOST=/var/tmp/k3pg PGPORT=54329 PGUSER=k3 PGDATABASE=postgres \
//!   cargo run --release -p lance-graph-quack --example k3_where_pg_parity
//! ```

use lance_graph_contract::set_algebra::{normalize, normalize_kleene, Cmp as SetCmp, Res, SetExpr};
use lance_graph_mask_risc::{
    execute_into, words_for, Foreign, LaneRef, MaskOp, Operand, Out, Planes, Program, Scratch,
    Terminal, Value,
};
use lance_graph_quack::{lower, lower_fused, Agg, Cmp, Col, Filter, Mask, Query};
use std::io::Write;
use std::process::{Command, Stdio};

const ROWS: usize = 200;
const EXPRS: usize = 2000;

// Value lanes: a0, a1 (i32), u0 (u32). Mask planes: validity of a0, a1, u0;
// the two symbol columns; the all-rows plane that `All` reads.
const A0: Col = Col(0);
const A1: Col = Col(1);
const U0: Col = Col(2);
const VA0: Mask = Mask(0);
const VA1: Mask = Mask(1);
const VU0: Mask = Mask(2);
const S0: u16 = 3;
const ALL: Mask = Mask(5);
const NULLABLE: [(Col, Mask); 3] = [(A0, VA0), (A1, VA1), (U0, VU0)];

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

/// Resident lanes and planes. At a NULL row the value lane holds a payload
/// some predicate tests for (`0`, `1`), so an engine that reads the payload of
/// a NULL row is caught.
struct Table {
    a: [Vec<i32>; 2],
    u: Vec<u32>,
    planes: [Vec<u64>; 6],
}

fn set(p: &mut [u64], r: usize) {
    p[r / 64] |= 1 << (r % 64);
}

fn table(rng: &mut Rng) -> Table {
    let w = words_for(ROWS);
    let mut t = Table {
        a: [vec![0; ROWS], vec![0; ROWS]],
        u: vec![0; ROWS],
        planes: std::array::from_fn(|_| vec![0; w]),
    };
    for r in 0..ROWS {
        for l in 0..2 {
            if rng.below(4) != 0 {
                t.a[l][r] = rng.below(9) as i32 - 4;
                set(&mut t.planes[l], r);
            }
        }
        if rng.below(4) != 0 {
            t.u[r] = rng.below(3) as u32;
            set(&mut t.planes[2], r);
        } else {
            t.u[r] = 1;
        }
        for s in 0..2 {
            if rng.below(2) == 0 {
                set(&mut t.planes[S0 as usize + s], r);
            }
        }
        set(&mut t.planes[ALL.0 as usize], r);
    }
    t
}

fn bit(p: &[u64], r: usize) -> bool {
    p[r / 64] >> (r % 64) & 1 == 1
}

fn leaf(rng: &mut Rng) -> SetExpr {
    const OPS: [SetCmp; 6] = [
        SetCmp::Lt,
        SetCmp::Le,
        SetCmp::Gt,
        SetCmp::Ge,
        SetCmp::Eq,
        SetCmp::Ne,
    ];
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
        _ => SetExpr::i32(rng.below(2) as u16, SetCmp::Ge, i32::MIN),
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

/// The contract expression as a Quack filter. A mapping only: every
/// operator below is one Quack already has.
fn filter(e: &SetExpr) -> Filter {
    let all = || Filter::Plane(ALL);
    match e {
        SetExpr::All => all(),
        SetExpr::Empty => Filter::Not(Box::new(all())),
        SetExpr::Sym(i) => Filter::Plane(Mask(S0 + *i as u16)),
        SetExpr::Res(Res::I32 { lane, op, t }) => Filter::Cmp(
            if *lane == 0 { A0 } else { A1 },
            match op {
                SetCmp::Lt => Cmp::LtI32(*t),
                SetCmp::Le => Cmp::LeI32(*t),
                SetCmp::Gt => Cmp::GtI32(*t),
                SetCmp::Ge => Cmp::GeI32(*t),
                SetCmp::Eq => Cmp::EqI32(*t),
                SetCmp::Ne => Cmp::NeI32(*t),
            },
        ),
        SetExpr::Res(Res::U32 { eq, v, .. }) => {
            Filter::Cmp(U0, if *eq { Cmp::EqU32(*v) } else { Cmp::NeU32(*v) })
        }
        SetExpr::Not(c) => Filter::Not(Box::new(filter(c))),
        // `AND()` is TRUE and `OR()` is FALSE, as in SQL's rendering below.
        SetExpr::And(cs) if cs.is_empty() => all(),
        SetExpr::Or(cs) if cs.is_empty() => Filter::Not(Box::new(all())),
        SetExpr::And(cs) => Filter::And(cs.iter().map(filter).collect()),
        SetExpr::Or(cs) => Filter::Or(cs.iter().map(filter).collect()),
    }
}

fn op_sql(op: SetCmp) -> &'static str {
    match op {
        SetCmp::Lt => "<",
        SetCmp::Le => "<=",
        SetCmp::Gt => ">",
        SetCmp::Ge => ">=",
        SetCmp::Eq => "=",
        SetCmp::Ne => "<>",
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

/// Which lowering of the engine to run.
#[derive(Clone, Copy)]
enum Mode {
    InPlace,
    Fused,
}

/// The rows the engine keeps for `f` read as SQL `WHERE`: Quack's
/// `sql_where`, lowered, executed by mask-risc with the `Keep` terminal.
fn keep(t: &Table, f: &Filter, mode: Mode) -> Vec<u64> {
    let q = Query {
        filter: f.sql_where(&NULLABLE),
        agg: Agg::Rows,
    };
    let p = match mode {
        Mode::InPlace => lower(&q),
        Mode::Fused => lower_fused(&q),
    }
    .expect("lowers");
    let masks: Vec<&[u64]> = t.planes.iter().map(Vec::as_slice).collect();
    let lanes = [
        LaneRef::I32(&t.a[0]),
        LaneRef::I32(&t.a[1]),
        LaneRef::U32(&t.u),
    ];
    let pl = Planes {
        n_rows: ROWS,
        masks: &masks,
        lanes: &lanes,
    };
    let mut out = vec![0u64; words_for(ROWS)];
    let mut s = Scratch::for_program(&p, ROWS).expect("carves");
    execute_into(&p, &pl, &Foreign::NONE, &mut s, Out::Mask(&mut out)).expect("runs");
    out
}

/// Whether two masks differ in any row: mask-risc `Xor`, then `Any`.
fn differ(a: &[u64], b: &[u64]) -> bool {
    let p = Program::new(
        vec![MaskOp::Xor {
            a: Operand::Plane(0),
            b: Operand::Plane(1),
            dst: 0,
        }],
        Terminal::Any {
            mask: Operand::Scratch(0),
        },
    );
    let pl = Planes {
        n_rows: ROWS,
        masks: &[a, b],
        lanes: &[],
    };
    let mut s = Scratch::for_program(&p, ROWS).expect("carves");
    match execute_into(&p, &pl, &Foreign::NONE, &mut s, Out::None).expect("runs") {
        Value::Bool(any) => any,
        v => panic!("Any returned {v:?}"),
    }
}

/// PostgreSQL's bit string (`'1'` per kept row, in `id` order) as a plane.
fn plane(bits: &str) -> Vec<u64> {
    assert_eq!(bits.len(), ROWS, "one character per row");
    let mut p = vec![0u64; words_for(ROWS)];
    for (r, c) in bits.bytes().enumerate() {
        if c == b'1' {
            set(&mut p, r);
        }
    }
    p
}

fn lit(valid: bool, v: impl std::fmt::Display) -> String {
    if valid {
        v.to_string()
    } else {
        "NULL".into()
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
            // psql answers while it reads: write on a separate thread so a
            // full stdout pipe cannot block the writer.
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
    let mut rng = Rng(0x00D1_5EED_0003_0001);
    let t = table(&mut rng);
    let exprs: Vec<SetExpr> = (0..EXPRS).map(|_| expr(&mut rng, 4)).collect();

    // The INSERT is PostgreSQL's input: the one place the lanes are read
    // side by side.
    let mut script = String::from(
        "CREATE TEMP TABLE t (id int, a0 int, a1 int, u0 int, s0 bool, s1 bool);\n\
         INSERT INTO t VALUES\n",
    );
    let values: Vec<String> = (0..ROWS)
        .map(|r| {
            format!(
                "({r}, {}, {}, {}, {}, {})",
                lit(bit(&t.planes[0], r), t.a[0][r]),
                lit(bit(&t.planes[1], r), t.a[1][r]),
                lit(bit(&t.planes[2], r), t.u[r]),
                bit(&t.planes[S0 as usize], r),
                bit(&t.planes[S0 as usize + 1], r),
            )
        })
        .collect();
    script.push_str(&values.join(",\n"));
    script.push_str(";\n");
    for e in &exprs {
        let w = sql(e);
        script.push_str(&format!(
            "SELECT string_agg(CASE WHEN {w} THEN '1' ELSE '0' END, '' ORDER BY id), \
             string_agg(CASE WHEN NOT {w} THEN '1' ELSE '0' END, '' ORDER BY id) FROM t;\n"
        ));
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
    assert_eq!(lines.len(), EXPRS, "one result row per expression");

    let (mut rewritten, mut two_valued_wrong, mut unknown_rows, mut nk_false_moved) =
        (0usize, 0usize, 0usize, 0usize);
    for (e, line) in exprs.iter().zip(&lines) {
        let (tbits, fbits) = line.split_once(',').expect("two columns");
        let (pg_t, pg_f) = (plane(tbits), plane(fbits));
        unknown_rows += ROWS - tbits.matches('1').count() - fbits.matches('1').count();
        let (f, nk) = (filter(e), filter(normalize_kleene(e).where_true()));
        for mode in [Mode::InPlace, Mode::Fused] {
            let neg = Filter::Not(Box::new(f.clone()));
            assert!(
                !differ(&keep(&t, &f, mode), &pg_t),
                "TRUE rows differ from PostgreSQL: {e:?}"
            );
            assert!(
                !differ(&keep(&t, &neg, mode), &pg_f),
                "FALSE rows differ from PostgreSQL: {e:?}"
            );
            // `normalize_kleene` is WHERE-terminal: it keeps the TRUE rows and
            // promises nothing about the FALSE ones (NOT over it is refused).
            assert!(
                !differ(&keep(&t, &nk, mode), &pg_t),
                "normalize_kleene changed the TRUE rows: {e:?}"
            );
        }
        nk_false_moved += usize::from(differ(
            &keep(&t, &Filter::Not(Box::new(nk.clone())), Mode::InPlace),
            &pg_f,
        ));
        rewritten += usize::from(normalize_kleene(e).where_true() != e);
        two_valued_wrong += usize::from(differ(
            &keep(&t, &filter(&normalize(e)), Mode::InPlace),
            &pg_t,
        ));
    }
    let version = Command::new("psql")
        .args(["-X", "-tA", "-c", "SHOW server_version"])
        .output()
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
        .unwrap_or_default();
    println!(
        "PostgreSQL {version}: {EXPRS} expressions over {ROWS} rows ({unknown_rows} UNKNOWN \
         row-verdicts). Quack sql_where -> lower / lower_fused -> mask-risc keeps PostgreSQL's \
         TRUE and FALSE rows exactly as written; normalize_kleene ({rewritten} rewritten) keeps \
         the TRUE rows on all and moves FALSE rows on {nk_false_moved} (WHERE-terminal). \
         Two-valued normalize differs on {two_valued_wrong}."
    );
    assert!(unknown_rows > 0, "the table never produced UNKNOWN");
    assert!(two_valued_wrong > 0, "the run never reached a lane cover");
}
