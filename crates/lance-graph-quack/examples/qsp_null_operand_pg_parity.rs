//! D-QSP: SQL NULL operands and boolean literals, Quack against PostgreSQL,
//! row for row.
//!
//! Random expressions with NULL operands (`x = NULL`, `NULL < c`, `NULL::boolean`,
//! `x IN (.., NULL)`, `x NOT IN (.., NULL)`) over nullable columns are mapped to
//! Quack [`Filter`]s. A NULL operand is a column `N` whose value lane borrows
//! another lane and whose validity plane is all clear, listed as nullable; the
//! TRUE and FALSE literals are all-ones and all-zero planes. Each expression
//! runs through [`Filter::sql_where`] -> [`lower`] / [`lower_fused`] -> mask-risc
//! [`execute_into`] with `Out::Mask`, once for the TRUE plane (`e`) and once for
//! the FALSE plane (`NOT e`).
//!
//! PostgreSQL is the reference: it returns the TRUE rows and the FALSE rows as
//! two bit strings in `id` order. Planes are compared by mask-risc `Xor` then the
//! `Any` terminal, so an UNKNOWN row (in neither plane) cannot hide.
//!
//! Three more arms:
//! - a payload scramble: every NULL payload of `x`, `y` and the whole `N` lane
//!   re-drawn, the planes must not move;
//! - disable arm 1, `unknown_as_false`: NULL operands mapped to the FALSE plane;
//! - disable arm 2, `null_operand_not_nullable`: `N` left out of the nullable
//!   list.
//!
//! Both disable arms must differ from PostgreSQL on some expressions, which
//! shows the differential can see each error.
//!
//! Needs `psql` and a server, connection from the usual `PG*` variables.
//! Without one it prints `SKIP` and exits 0, so nothing here gates CI.
//!
//! ```text
//! PGHOST=<socket dir> PGPORT=<port> PGUSER=<user> PGDATABASE=postgres \
//!   cargo run --release -p lance-graph-quack --example qsp_null_operand_pg_parity
//! ```

use lance_graph_mask_risc::{
    execute_into, words_for, Foreign, LaneRef, MaskOp, Operand, Out, Planes, Program, Scratch,
    Terminal, Value,
};
use lance_graph_quack::{lower, lower_fused, Agg, Cmp, Col, Filter, Mask, Query};
use std::io::Write;
use std::process::{Command, Stdio};

const ROWS: usize = 200;
const EXPRS: usize = 2000;

const X: Col = Col(0);
const Y: Col = Col(1);
const N: Col = Col(2);
const VX: Mask = Mask(0);
const VY: Mask = Mask(1);
const VN: Mask = Mask(2);
const TRUE_M: Mask = Mask(3);
const FALSE_M: Mask = Mask(4);
const NULLABLE: [(Col, Mask); 3] = [(X, VX), (Y, VY), (N, VN)];
const NULLABLE_NO_N: [(Col, Mask); 2] = [(X, VX), (Y, VY)];

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

/// `n` is `None` when N borrows X's lane, `Some` for an independent lane.
struct Table {
    x: Vec<i32>,
    y: Vec<i32>,
    n: Option<Vec<i32>>,
    planes: [Vec<u64>; 5],
}

fn set(p: &mut [u64], r: usize) {
    p[r / 64] |= 1 << (r % 64);
}

fn bit(p: &[u64], r: usize) -> bool {
    p[r / 64] >> (r % 64) & 1 == 1
}

/// Values in -4..4; the NULL payload is drawn from the same range so it
/// collides with the constants the predicates test.
fn table(rng: &mut Rng) -> Table {
    let w = words_for(ROWS);
    let mut t = Table {
        x: vec![0; ROWS],
        y: vec![0; ROWS],
        n: None,
        planes: std::array::from_fn(|_| vec![0; w]),
    };
    for r in 0..ROWS {
        for l in 0..2 {
            let v = rng.below(9) as i32 - 4;
            if l == 0 {
                t.x[r] = v;
            } else {
                t.y[r] = v;
            }
            if rng.below(4) != 0 {
                set(&mut t.planes[l], r);
            }
        }
        set(&mut t.planes[TRUE_M.0 as usize], r);
    }
    t
}

/// Same validity, every invalid payload of x and y re-drawn from a wide range,
/// and N an independent lane of arbitrary values.
fn scramble(t: &Table, rng: &mut Rng) -> Table {
    const WIDE: [i32; 6] = [i32::MIN, i32::MAX, -7, 0, 1, 2];
    let mut s = Table {
        x: t.x.clone(),
        y: t.y.clone(),
        n: Some((0..ROWS).map(|_| rng.next() as i32).collect()),
        planes: t.planes.clone(),
    };
    for r in 0..ROWS {
        if !bit(&t.planes[0], r) {
            s.x[r] = WIDE[rng.below(6) as usize];
        }
        if !bit(&t.planes[1], r) {
            s.y[r] = WIDE[rng.below(6) as usize];
        }
    }
    s
}

#[derive(Clone, Copy)]
enum Op {
    Eq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,
}

const OPS: [Op; 6] = [Op::Eq, Op::Ne, Op::Lt, Op::Le, Op::Gt, Op::Ge];

fn op_sql(op: Op) -> &'static str {
    match op {
        Op::Eq => "=",
        Op::Ne => "<>",
        Op::Lt => "<",
        Op::Le => "<=",
        Op::Gt => ">",
        Op::Ge => ">=",
    }
}

/// A local expression, used only to map to a Quack filter and to SQL.
#[derive(Debug, Clone)]
enum E {
    Cmp {
        col: u8,
        op: Op,
        c: i32,
    },
    CmpNull {
        col: u8,
        op: Op,
    },
    NullCmp {
        op: Op,
        c: i32,
    },
    True,
    False,
    Unknown,
    In {
        col: u8,
        set: Vec<i32>,
        has_null: bool,
    },
    NotIn {
        col: u8,
        set: Vec<i32>,
        has_null: bool,
    },
    Not(Box<E>),
    And(Vec<E>),
    Or(Vec<E>),
}

impl std::fmt::Debug for Op {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(op_sql(*self))
    }
}

fn col_of(c: u8) -> Col {
    if c == 0 {
        X
    } else {
        Y
    }
}

fn col_name(c: u8) -> &'static str {
    if c == 0 {
        "x"
    } else {
        "y"
    }
}

fn leaf(rng: &mut Rng) -> E {
    let col = rng.below(2) as u8;
    let c = rng.below(9) as i32 - 4;
    let op = OPS[rng.below(6) as usize];
    let members = |rng: &mut Rng| -> Vec<i32> {
        (0..1 + rng.below(3))
            .map(|_| rng.below(9) as i32 - 4)
            .collect()
    };
    match rng.below(10) {
        0..=2 | 8 | 9 => E::Cmp { col, op, c },
        3 => E::CmpNull { col, op },
        4 => E::NullCmp { op, c },
        5 => match rng.below(3) {
            0 => E::True,
            1 => E::False,
            _ => E::Unknown,
        },
        6 => E::In {
            col,
            set: members(rng),
            has_null: rng.below(2) == 0,
        },
        _ => E::NotIn {
            col,
            set: members(rng),
            has_null: rng.below(2) == 0,
        },
    }
}

fn expr(rng: &mut Rng, depth: u32) -> E {
    if depth == 0 || rng.below(4) == 0 {
        return leaf(rng);
    }
    match rng.below(3) {
        0 => E::Not(Box::new(expr(rng, depth - 1))),
        k => {
            let cs = (0..1 + rng.below(4))
                .map(|_| expr(rng, depth - 1))
                .collect();
            if k == 1 {
                E::And(cs)
            } else {
                E::Or(cs)
            }
        }
    }
}

fn cmp_i32(op: Op, c: i32) -> Cmp {
    match op {
        Op::Eq => Cmp::EqI32(c),
        Op::Ne => Cmp::NeI32(c),
        Op::Lt => Cmp::LtI32(c),
        Op::Le => Cmp::LeI32(c),
        Op::Gt => Cmp::GtI32(c),
        Op::Ge => Cmp::GeI32(c),
    }
}

/// The expression as a Quack filter; `null` is the leaf every NULL operand
/// maps to (the N column in the real mapping, the FALSE plane in the disable
/// arm). A mapping only: every operator below is one Quack already has.
fn to_filter(e: &E, null: &Filter) -> Filter {
    let in_list = |col: u8, set: &[i32], has_null: bool| {
        let mut parts: Vec<Filter> = set
            .iter()
            .map(|&c| Filter::Cmp(col_of(col), Cmp::EqI32(c)))
            .collect();
        if has_null {
            parts.push(null.clone());
        }
        Filter::Or(parts)
    };
    match e {
        E::Cmp { col, op, c } => Filter::Cmp(col_of(*col), cmp_i32(*op, *c)),
        E::CmpNull { .. } | E::NullCmp { .. } | E::Unknown => null.clone(),
        E::True => Filter::Plane(TRUE_M),
        E::False => Filter::Plane(FALSE_M),
        E::In { col, set, has_null } => in_list(*col, set, *has_null),
        E::NotIn { col, set, has_null } => Filter::Not(Box::new(in_list(*col, set, *has_null))),
        E::Not(c) => Filter::Not(Box::new(to_filter(c, null))),
        E::And(cs) => Filter::And(cs.iter().map(|c| to_filter(c, null)).collect()),
        E::Or(cs) => Filter::Or(cs.iter().map(|c| to_filter(c, null)).collect()),
    }
}

fn sql(e: &E) -> String {
    let list = |col: u8, set: &[i32], has_null: bool, kw: &str| {
        let mut items: Vec<String> = set.iter().map(|c| format!("({c})::int")).collect();
        if has_null {
            items.push("NULL::int".into());
        }
        format!("({} {kw} ({}))", col_name(col), items.join(", "))
    };
    match e {
        E::Cmp { col, op, c } => format!("({} {} ({c})::int)", col_name(*col), op_sql(*op)),
        E::CmpNull { col, op } => format!("({} {} NULL::int)", col_name(*col), op_sql(*op)),
        E::NullCmp { op, c } => format!("(NULL::int {} ({c})::int)", op_sql(*op)),
        E::True => "TRUE".into(),
        E::False => "FALSE".into(),
        E::Unknown => "NULL::boolean".into(),
        E::In { col, set, has_null } => list(*col, set, *has_null, "IN"),
        E::NotIn { col, set, has_null } => list(*col, set, *has_null, "NOT IN"),
        E::Not(c) => format!("(NOT {})", sql(c)),
        E::And(cs) => format!("({})", cs.iter().map(sql).collect::<Vec<_>>().join(" AND ")),
        E::Or(cs) => format!("({})", cs.iter().map(sql).collect::<Vec<_>>().join(" OR ")),
    }
}

fn has_null_operand(e: &E) -> bool {
    match e {
        E::CmpNull { .. } | E::NullCmp { .. } | E::Unknown => true,
        E::In { has_null, .. } | E::NotIn { has_null, .. } => *has_null,
        E::Not(c) => has_null_operand(c),
        E::And(cs) | E::Or(cs) => cs.iter().any(has_null_operand),
        _ => false,
    }
}

fn has_not_in_null(e: &E) -> bool {
    match e {
        E::NotIn { has_null, .. } => *has_null,
        E::Not(c) => has_not_in_null(c),
        E::And(cs) | E::Or(cs) => cs.iter().any(has_not_in_null),
        _ => false,
    }
}

#[derive(Clone, Copy)]
enum Mode {
    InPlace,
    Fused,
}

fn keep(t: &Table, f: &Filter, mode: Mode, nullable: &[(Col, Mask)]) -> Vec<u64> {
    let q = Query {
        filter: f.sql_where(nullable),
        agg: Agg::Rows,
    };
    let p = match mode {
        Mode::InPlace => lower(&q),
        Mode::Fused => lower_fused(&q),
    }
    .expect("lowers");
    let masks: Vec<&[u64]> = t.planes.iter().map(Vec::as_slice).collect();
    let n_lane: &[i32] = t.n.as_deref().unwrap_or(&t.x);
    let lanes = [LaneRef::I32(&t.x), LaneRef::I32(&t.y), LaneRef::I32(n_lane)];
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
    let mut rng = Rng(0x00D1_5EED_0007_0001);
    let t = table(&mut rng);
    let t2 = scramble(&t, &mut rng);
    let exprs: Vec<E> = (0..EXPRS).map(|_| expr(&mut rng, 4)).collect();

    let mut script =
        String::from("CREATE TEMP TABLE t (id int, x int, y int);\nINSERT INTO t VALUES\n");
    let values: Vec<String> = (0..ROWS)
        .map(|r| {
            format!(
                "({r}, {}, {})",
                lit(bit(&t.planes[0], r), t.x[r]),
                lit(bit(&t.planes[1], r), t.y[r]),
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

    let real_null = Filter::Cmp(N, Cmp::EqI32(0));
    let false_null = Filter::Plane(FALSE_M);
    let (mut unknown_rows, mut with_null, mut empty_not_in_null) = (0usize, 0usize, 0usize);
    let (mut dis_false, mut dis_nullable) = (0usize, 0usize);
    for (e, line) in exprs.iter().zip(&lines) {
        let (tbits, fbits) = line.split_once(',').expect("two columns");
        let (pg_t, pg_f) = (plane(tbits), plane(fbits));
        unknown_rows += ROWS - tbits.matches('1').count() - fbits.matches('1').count();
        with_null += usize::from(has_null_operand(e));
        empty_not_in_null += usize::from(has_not_in_null(e) && !tbits.contains('1'));

        let f = to_filter(e, &real_null);
        let neg = Filter::Not(Box::new(f.clone()));
        for mode in [Mode::InPlace, Mode::Fused] {
            assert!(
                !differ(&keep(&t, &f, mode, &NULLABLE), &pg_t),
                "TRUE rows differ from PostgreSQL: {e:?} = {}",
                sql(e)
            );
            assert!(
                !differ(&keep(&t, &neg, mode, &NULLABLE), &pg_f),
                "FALSE rows differ from PostgreSQL: {e:?} = {}",
                sql(e)
            );
        }
        // Payload scramble: invalid payloads and the N lane re-drawn.
        assert!(
            !differ(&keep(&t2, &f, Mode::InPlace, &NULLABLE), &pg_t),
            "TRUE rows depend on NULL payloads: {e:?}"
        );
        assert!(
            !differ(&keep(&t2, &neg, Mode::InPlace, &NULLABLE), &pg_f),
            "FALSE rows depend on NULL payloads: {e:?}"
        );

        // Disable arm 1: UNKNOWN read as FALSE.
        let fw = to_filter(e, &false_null);
        let nw = Filter::Not(Box::new(fw.clone()));
        dis_false += usize::from(
            differ(&keep(&t, &fw, Mode::InPlace, &NULLABLE), &pg_t)
                || differ(&keep(&t, &nw, Mode::InPlace, &NULLABLE), &pg_f),
        );
        // Disable arm 2: N not listed as nullable.
        dis_nullable += usize::from(
            differ(&keep(&t, &f, Mode::InPlace, &NULLABLE_NO_N), &pg_t)
                || differ(&keep(&t, &neg, Mode::InPlace, &NULLABLE_NO_N), &pg_f),
        );
    }
    let version = Command::new("psql")
        .args(["-X", "-tA", "-c", "SHOW server_version"])
        .output()
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
        .unwrap_or_default();
    println!(
        "PostgreSQL {version}: {EXPRS} expressions over {ROWS} rows ({unknown_rows} UNKNOWN \
         row-verdicts, {with_null} with a NULL operand, {empty_not_in_null} NOT IN-with-NULL \
         with an empty TRUE plane). Quack sql_where -> lower / lower_fused -> mask-risc matches \
         PostgreSQL's TRUE and FALSE rows exactly, also with every NULL payload scrambled. \
         Disabled: unknown_as_false differs on {dis_false}, null_operand_not_nullable on \
         {dis_nullable}."
    );
    assert!(unknown_rows > 0, "the table never produced UNKNOWN");
    assert!(
        with_null > EXPRS / 4,
        "too few expressions carry a NULL operand"
    );
    assert!(
        empty_not_in_null > 0,
        "no NOT IN with NULL produced an empty TRUE plane"
    );
    assert!(dis_false > 0, "unknown_as_false was not detected");
    assert!(
        dis_nullable > 0,
        "null_operand_not_nullable was not detected"
    );
}
