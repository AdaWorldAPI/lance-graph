//! SQL NULL operands and boolean literals through Quack -> mask-risc.
//!
//! The contract. A nullable value is a value lane plus a validity plane (bit 1
//! = present, 0 = NULL). For a nullable predicate `p`: TRUE = valid AND p,
//! FALSE = valid AND NOT p, UNKNOWN = NOT (TRUE OR FALSE). Quack computes TRUE
//! as `f.sql_where(&NULLABLE)` and FALSE as
//! `Filter::Not(f).sql_where(&NULLABLE)`.
//!
//! A SQL literal NULL operand is a column `N` whose value lane is a borrowed
//! alias of an existing lane and whose validity plane is all clear, listed in
//! `NULLABLE`. So `x = NULL`, `NULL = c` and the boolean literal `NULL::bool`
//! are all `Filter::Cmp(N, Cmp::EqI32(0))` (the constant is irrelevant: the
//! gate is clear on every row). The TRUE literal is a plane with every row set;
//! the FALSE literal is a present, all-zero plane, so FALSE and UNKNOWN stay
//! distinct.
//!
//! Every answer runs Quack `sql_where` -> `lower` / `lower_fused` -> mask-risc
//! `execute_into` with `Out::Mask`. Planes are compared by mask-risc `Xor`
//! followed by the `Any` terminal. The expected planes below are row lists read
//! off SQL semantics, not computed by an evaluator.

use lance_graph_mask_risc::{
    execute_into, words_for, Foreign, LaneRef, MaskOp, Operand, Out, Planes, Program, Scratch,
    Terminal, Value,
};
use lance_graph_quack::{lower, lower_fused, Agg, Cmp, Col, Filter, Mask, Query};

const ROWS: usize = 8;

const X: Col = Col(0);
const Y: Col = Col(1);
const N: Col = Col(2);
const VX: Mask = Mask(0);
const VY: Mask = Mask(1);
const VN: Mask = Mask(2);
const TRUE_M: Mask = Mask(3);
const FALSE_M: Mask = Mask(4);
const NULLABLE: [(Col, Mask); 3] = [(X, VX), (Y, VY), (N, VN)];

const ALL: [usize; ROWS] = [0, 1, 2, 3, 4, 5, 6, 7];
const NONE: [usize; 0] = [];

/// Resident lanes and planes. `n` is `None` when N borrows X's lane (the same
/// slice passed twice); `Some` when a test gives it arbitrary values.
struct Table {
    x: Vec<i32>,
    y: Vec<i32>,
    n: Option<Vec<i32>>,
    planes: [Vec<u64>; 5],
}

fn rows(rs: &[usize]) -> Vec<u64> {
    let mut p = vec![0u64; words_for(ROWS)];
    for &r in rs {
        p[r / 64] |= 1 << (r % 64);
    }
    p
}

/// x = [1, 2, NULL, 1, NULL, 3, 0, 1], y = [NULL, 1, 1, NULL, 0, 2, NULL, 5];
/// every NULL row's payload is 1 (the IN value, so a payload read is caught).
fn base() -> Table {
    Table {
        x: vec![1, 2, 1, 1, 1, 3, 0, 1],
        y: vec![1, 1, 1, 1, 0, 2, 1, 5],
        n: None,
        planes: [
            rows(&[0, 1, 3, 5, 6, 7]),
            rows(&[1, 2, 4, 5, 7]),
            rows(&NONE),
            rows(&ALL),
            rows(&NONE),
        ],
    }
}

/// The same validity, every NULL payload changed, N over arbitrary values.
fn scrambled() -> Table {
    let mut t = base();
    t.x = vec![1, 2, 0, 1, i32::MIN, 3, 0, 1];
    t.y = vec![2, 1, 1, -7, 0, 2, i32::MAX, 5];
    t.n = Some(vec![5, -1, 0, 1, 1, i32::MIN, 0, i32::MAX]);
    t
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

/// (TRUE plane, FALSE plane) of `f` on `t` under `nullable`, for both
/// lowerings, which must agree.
fn planes_with(t: &Table, f: &Filter, nullable: &[(Col, Mask)]) -> (Vec<u64>, Vec<u64>) {
    let neg = Filter::Not(Box::new(f.clone()));
    let (ti, fi) = (
        keep(t, f, Mode::InPlace, nullable),
        keep(t, &neg, Mode::InPlace, nullable),
    );
    let (tf_, ff) = (
        keep(t, f, Mode::Fused, nullable),
        keep(t, &neg, Mode::Fused, nullable),
    );
    assert!(
        !differ(&ti, &tf_),
        "TRUE plane: lower and lower_fused disagree for {f:?}"
    );
    assert!(
        !differ(&fi, &ff),
        "FALSE plane: lower and lower_fused disagree for {f:?}"
    );
    (ti, fi)
}

fn tf(f: &Filter) -> (Vec<u64>, Vec<u64>) {
    planes_with(&base(), f, &NULLABLE)
}

fn assert_k3(f: &Filter, expect_t: &[usize], expect_f: &[usize], label: &str) {
    let (t, fl) = tf(f);
    assert!(
        !differ(&t, &rows(expect_t)),
        "{label}: TRUE plane differs from the expected rows {expect_t:?}"
    );
    assert!(
        !differ(&fl, &rows(expect_f)),
        "{label}: FALSE plane differs from the expected rows {expect_f:?}"
    );
}

// ----- mappings (leaves Quack already has) -----

fn null_leaf() -> Filter {
    Filter::Cmp(N, Cmp::EqI32(0))
}
fn x_eq(c: i32) -> Filter {
    Filter::Cmp(X, Cmp::EqI32(c))
}
fn x_in(set: &[i32], null: Option<Filter>) -> Filter {
    let mut parts: Vec<Filter> = set.iter().map(|&c| x_eq(c)).collect();
    parts.extend(null);
    Filter::Or(parts)
}
fn not(f: Filter) -> Filter {
    Filter::Not(Box::new(f))
}
fn tru() -> Filter {
    Filter::Plane(TRUE_M)
}
fn fals() -> Filter {
    Filter::Plane(FALSE_M)
}

/// FAILS IF `x = NULL` is read as FALSE (F would be every row).
#[test]
fn x_eq_null_is_unknown_not_false() {
    assert_k3(&null_leaf(), &NONE, &NONE, "x = NULL");
}

/// FAILS IF NOT over UNKNOWN turns TRUE (T would be every row).
#[test]
fn not_x_eq_null_stays_unknown_not_true() {
    assert_k3(&not(null_leaf()), &NONE, &NONE, "NOT (x = NULL)");
}

/// FAILS IF a NULL IN-member does not move the non-matches from FALSE to
/// UNKNOWN, or a NULL row of x is treated as present.
#[test]
fn x_in_1_null_splits_match_nonmatch_and_null() {
    assert_k3(
        &x_in(&[1], Some(null_leaf())),
        &[0, 3, 7],
        &NONE,
        "x IN (1, NULL)",
    );
    assert_k3(&x_in(&[1, 2], None), &[0, 1, 3, 7], &[5, 6], "x IN (1, 2)");
}

/// FAILS IF `NOT IN` with a NULL member lets an UNKNOWN row through as TRUE.
#[test]
fn x_not_in_1_null_never_turns_unknown_true() {
    assert_k3(
        &not(x_in(&[1], Some(null_leaf()))),
        &NONE,
        &[0, 3, 7],
        "x NOT IN (1, NULL)",
    );
    assert_k3(
        &not(x_in(&[1, 2], None)),
        &[5, 6],
        &[0, 1, 3, 7],
        "x NOT IN (1, 2)",
    );
}

#[derive(Clone, Copy)]
enum K {
    T,
    F,
    U,
}

fn konst(k: K) -> Filter {
    match k {
        K::T => tru(),
        K::F => fals(),
        K::U => null_leaf(),
    }
}

fn assert_const(f: &Filter, expect: K, label: &str) {
    match expect {
        K::T => assert_k3(f, &ALL, &NONE, label),
        K::F => assert_k3(f, &NONE, &ALL, label),
        K::U => assert_k3(f, &NONE, &NONE, label),
    }
}

/// FAILS IF AND is two-valued: FALSE AND UNKNOWN must be FALSE (either order),
/// TRUE AND UNKNOWN must be UNKNOWN.
#[test]
fn k3_and_truth_table() {
    use K::{F, T, U};
    let table = [
        (T, T, T),
        (T, F, F),
        (T, U, U),
        (F, T, F),
        (F, F, F),
        (F, U, F),
        (U, T, U),
        (U, F, F),
        (U, U, U),
    ];
    for (a, b, r) in table {
        let f = Filter::And(vec![konst(a), konst(b)]);
        assert_const(&f, r, "AND");
    }
}

/// FAILS IF OR is two-valued: TRUE OR UNKNOWN must be TRUE (either order),
/// FALSE OR UNKNOWN must be UNKNOWN.
#[test]
fn k3_or_truth_table() {
    use K::{F, T, U};
    let table = [
        (T, T, T),
        (T, F, T),
        (T, U, T),
        (F, T, T),
        (F, F, F),
        (F, U, U),
        (U, T, T),
        (U, F, U),
        (U, U, U),
    ];
    for (a, b, r) in table {
        let f = Filter::Or(vec![konst(a), konst(b)]);
        assert_const(&f, r, "OR");
    }
}

/// FAILS IF NOT UNKNOWN is not UNKNOWN, or NOT flips a present constant wrongly.
#[test]
fn k3_not_truth_table() {
    use K::{F, T, U};
    for (a, r) in [(T, F), (F, T), (U, U)] {
        assert_const(&not(konst(a)), r, "NOT");
    }
}

/// FAILS IF the FALSE literal and UNKNOWN collapse into one verdict.
#[test]
fn false_literal_and_unknown_are_distinct() {
    let (tf_, ff) = tf(&fals());
    assert!(
        !differ(&tf_, &rows(&NONE)),
        "FALSE literal: TRUE plane must be empty"
    );
    assert!(
        !differ(&ff, &rows(&ALL)),
        "FALSE literal: FALSE plane must be every row"
    );
    let (tu, fu) = tf(&null_leaf());
    assert!(
        !differ(&tu, &rows(&NONE)),
        "UNKNOWN: TRUE plane must be empty"
    );
    assert!(
        !differ(&fu, &rows(&NONE)),
        "UNKNOWN: FALSE plane must be empty"
    );
    assert!(
        differ(&ff, &fu),
        "FALSE and UNKNOWN must differ on the FALSE plane"
    );
}

fn battery() -> Vec<(&'static str, Filter)> {
    let x_in_1_null = x_in(&[1], Some(null_leaf()));
    let y_in = Filter::Or(vec![Filter::Cmp(Y, Cmp::EqI32(1)), null_leaf()]);
    vec![
        ("x = NULL", null_leaf()),
        ("NOT x = NULL", not(null_leaf())),
        ("x IN (1, NULL)", x_in_1_null.clone()),
        ("x NOT IN (1, NULL)", not(x_in_1_null)),
        ("x IN (1, 2)", x_in(&[1, 2], None)),
        ("x NOT IN (1, 2)", not(x_in(&[1, 2], None))),
        ("x = 1", x_eq(1)),
        ("x <> 1", Filter::Cmp(X, Cmp::NeI32(1))),
        ("y > 0", Filter::Cmp(Y, Cmp::GtI32(0))),
        (
            "x = 1 AND y = NULL",
            Filter::And(vec![x_eq(1), null_leaf()]),
        ),
        (
            "(x = 1 OR y = NULL) AND NOT (y IN (1, NULL))",
            Filter::And(vec![Filter::Or(vec![x_eq(1), null_leaf()]), not(y_in)]),
        ),
        (
            "TRUE AND NOT (x IN (0, NULL))",
            Filter::And(vec![tru(), not(x_in(&[0], Some(null_leaf())))]),
        ),
    ]
}

/// FAILS IF any verdict reads the payload of a NULL row of x or y, or the
/// values of the N alias lane.
#[test]
fn invalid_payload_changes_nothing() {
    let (a, b) = (base(), scrambled());
    for (label, f) in battery() {
        let (ta, fa) = planes_with(&a, &f, &NULLABLE);
        let (tb, fb) = planes_with(&b, &f, &NULLABLE);
        assert!(
            !differ(&ta, &tb),
            "{label}: TRUE plane depends on NULL payloads"
        );
        assert!(
            !differ(&fa, &fb),
            "{label}: FALSE plane depends on NULL payloads"
        );
    }
}

/// FAILS IF the checker cannot tell UNKNOWN read as FALSE from UNKNOWN: with
/// every NULL operand mapped to the FALSE plane, NOT (x = NULL) becomes TRUE on
/// every row and x NOT IN (1, NULL) keeps rows 1, 5, 6.
#[test]
fn unknown_read_as_false_is_detected() {
    let wrong_not = not(fals());
    let (t, _) = tf(&wrong_not);
    assert!(
        !differ(&t, &rows(&ALL)),
        "the wrong reading is TRUE on every row"
    );
    assert!(
        differ(&t, &rows(&NONE)),
        "NOT (x = NULL): wrong reading must differ from []"
    );

    let wrong_not_in = not(x_in(&[1], Some(fals())));
    let (t, _) = tf(&wrong_not_in);
    assert!(
        !differ(&t, &rows(&[1, 5, 6])),
        "the wrong reading keeps rows 1, 5, 6"
    );
    assert!(
        differ(&t, &rows(&NONE)),
        "x NOT IN (1, NULL): wrong reading must differ from []"
    );
}

/// FAILS IF the checker cannot tell a NULL operand left out of the nullable
/// list (its alias payload read two-valued) from the correct three-valued one.
#[test]
fn null_operand_left_out_of_nullable_is_detected() {
    let without_n = [(X, VX), (Y, VY)];
    let (t, f) = planes_with(&base(), &null_leaf(), &without_n);
    assert!(
        differ(&t, &rows(&NONE)),
        "x = NULL: TRUE plane must differ without N listed"
    );
    assert!(
        differ(&f, &rows(&NONE)),
        "x = NULL: FALSE plane must differ without N listed"
    );

    let e = not(x_in(&[1], Some(null_leaf())));
    let (t, f) = planes_with(&base(), &e, &without_n);
    let diff_t = differ(&t, &rows(&NONE));
    let diff_f = differ(&f, &rows(&[0, 3, 7]));
    assert!(
        diff_t || diff_f,
        "x NOT IN (1, NULL): some plane must differ without N listed"
    );
}
