//! SQL three-valued `WHERE` over nullable columns, against an independent
//! row-at-a-time 3VL oracle.
//!
//! The model under test: a nullable column is an ordinary value lane PLUS its
//! resident validity plane. A row is NULL in `x` exactly when its validity bit
//! is clear; the value lane's payload there is irrelevant. To make that
//! visible, the fixture puts payloads at NULL rows that WOULD match the
//! predicates (`5`, `0`, `7`). Any implementation that reads the payload of a
//! NULL row gets caught.
//!
//! The oracle never calls `sql_where`. It walks the ORIGINAL filter with
//! Kleene logic (TRUE / FALSE / UNKNOWN) and keeps the rows whose verdict is
//! TRUE. Every lowered program — `lower` and `lower_fused` — must keep the same
//! rows, and its `COUNT(*)` must agree.

use lance_graph_mask_risc::{
    execute, execute_into, materialize_rows, words_for, Foreign, LaneRef, Out, Planes, Scratch,
    Value,
};
use lance_graph_quack::{
    avg_finish, lower, lower_avg, lower_fused, Agg, Cmp, Col, Filter, Mask, Query,
};

const X: Col = Col(0);
const Y: Col = Col(1);
const VX: Mask = Mask(0);
const VY: Mask = Mask(1);
const NULLABLE: [(Col, Mask); 2] = [(X, VX), (Y, VY)];

/// `None` is NULL. The payload stored at a NULL row is a separate field.
#[derive(Clone, Copy)]
struct Cell {
    v: Option<i32>,
    payload_if_null: i32,
}

struct Table {
    x: Vec<i32>,
    y: Vec<i32>,
    vx: Vec<u64>,
    vy: Vec<u64>,
    cells: Vec<(Cell, Cell)>,
}

fn bit(w: &[u64], r: usize) -> bool {
    w[r / 64] >> (r % 64) & 1 == 1
}

/// Every combination of `x ∈ {NULL, NULL, 0, 1, 5, 7, 10, 11}` and
/// `y ∈ {NULL, 7, 3}`. The two NULL x cells carry payloads `5` and `0`, the
/// NULL y cell carries `7`: each is the value some predicate below tests for.
/// The product (24 rows) is repeated with a stride so the table spans three
/// words and the tiles see both edge and interior words.
fn table() -> Table {
    let xs = [
        Cell {
            v: None,
            payload_if_null: 5,
        },
        Cell {
            v: None,
            payload_if_null: 0,
        },
        Cell {
            v: Some(0),
            payload_if_null: 0,
        },
        Cell {
            v: Some(1),
            payload_if_null: 0,
        },
        Cell {
            v: Some(5),
            payload_if_null: 0,
        },
        Cell {
            v: Some(7),
            payload_if_null: 0,
        },
        Cell {
            v: Some(10),
            payload_if_null: 0,
        },
        Cell {
            v: Some(11),
            payload_if_null: 0,
        },
    ];
    let ys = [
        Cell {
            v: None,
            payload_if_null: 7,
        },
        Cell {
            v: Some(7),
            payload_if_null: 0,
        },
        Cell {
            v: Some(3),
            payload_if_null: 0,
        },
    ];
    let mut cells = Vec::new();
    for rep in 0..7 {
        for i in 0..xs.len() {
            for j in 0..ys.len() {
                // Rotate the product per repetition so row order is not the
                // same pattern repeated (row bits land in different word slots).
                cells.push((xs[(i + rep) % xs.len()], ys[(j + 2 * rep) % ys.len()]));
            }
        }
    }
    let n = cells.len();
    let mut t = Table {
        x: Vec::with_capacity(n),
        y: Vec::with_capacity(n),
        vx: vec![0; words_for(n)],
        vy: vec![0; words_for(n)],
        cells,
    };
    for (r, (cx, cy)) in t.cells.iter().enumerate() {
        t.x.push(cx.v.unwrap_or(cx.payload_if_null));
        t.y.push(cy.v.unwrap_or(cy.payload_if_null));
        if cx.v.is_some() {
            t.vx[r / 64] |= 1 << (r % 64);
        }
        if cy.v.is_some() {
            t.vy[r / 64] |= 1 << (r % 64);
        }
    }
    t
}

// ── The oracle: Kleene logic over the ORIGINAL filter ─────────────────────

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Tv {
    T,
    F,
    U,
}

fn tv(b: bool) -> Tv {
    if b {
        Tv::T
    } else {
        Tv::F
    }
}

fn eval(f: &Filter, t: &Table, r: usize) -> Tv {
    match f {
        Filter::Cmp(col, cmp) => {
            let cell = if *col == X {
                t.cells[r].0
            } else {
                t.cells[r].1
            };
            let Some(v) = cell.v else { return Tv::U };
            tv(match *cmp {
                Cmp::EqI32(c) => v == c,
                Cmp::NeI32(c) => v != c,
                Cmp::LtI32(c) => v < c,
                Cmp::LeI32(c) => v <= c,
                Cmp::GtI32(c) => v > c,
                Cmp::GeI32(c) => v >= c,
                other => panic!("oracle: unsupported {other:?}"),
            })
        }
        Filter::Plane(m) => {
            let w = if *m == VX { &t.vx } else { &t.vy };
            tv(bit(w, r))
        }
        Filter::Not(inner) => match eval(inner, t, r) {
            Tv::T => Tv::F,
            Tv::F => Tv::T,
            Tv::U => Tv::U,
        },
        Filter::And(parts) => {
            let vs: Vec<Tv> = parts.iter().map(|p| eval(p, t, r)).collect();
            if vs.contains(&Tv::F) {
                Tv::F
            } else if vs.contains(&Tv::U) {
                Tv::U
            } else {
                Tv::T
            }
        }
        Filter::Or(parts) => {
            let vs: Vec<Tv> = parts.iter().map(|p| eval(p, t, r)).collect();
            if vs.contains(&Tv::T) {
                Tv::T
            } else if vs.contains(&Tv::U) {
                Tv::U
            } else {
                Tv::F
            }
        }
        other => panic!("oracle: unsupported {other:?}"),
    }
}

fn oracle(f: &Filter, t: &Table) -> Vec<usize> {
    (0..t.cells.len())
        .filter(|&r| eval(f, t, r) == Tv::T)
        .collect()
}

// ── Running a lowered program ────────────────────────────────────────────

fn planes(t: &Table) -> ([&[u64]; 2], [LaneRef<'_>; 2]) {
    ([&t.vx, &t.vy], [LaneRef::I32(&t.x), LaneRef::I32(&t.y)])
}

/// The rows a filter keeps, run through `lower` (`fused == false`) or
/// `lower_fused`, with `COUNT(*)` checked against the kept set.
fn kept(f: &Filter, t: &Table, fused: bool) -> Vec<usize> {
    let n = t.cells.len();
    let (masks, lanes) = planes(t);
    let pl = Planes {
        n_rows: n,
        masks: &masks,
        lanes: &lanes,
    };
    let lower_fn = if fused { lower_fused } else { lower };
    let p = lower_fn(&Query {
        filter: f.clone(),
        agg: Agg::Rows,
    })
    .expect("lowers");
    let mut s = Scratch::for_program(&p, n).expect("carves");
    let mut out = vec![0u64; words_for(n)];
    execute_into(&p, &pl, &Foreign::NONE, &mut s, Out::Mask(&mut out)).expect("runs");
    let rows = materialize_rows(&out, n);

    let c = lower_fn(&Query {
        filter: f.clone(),
        agg: Agg::Count,
    })
    .expect("lowers");
    let mut s = Scratch::for_program(&c, n).expect("carves");
    assert_eq!(
        execute(&c, &pl, &mut s, None).expect("runs"),
        Value::Count(rows.len()),
        "COUNT(*) disagrees with the kept rows"
    );
    rows
}

fn check(name: &str, f: &Filter, t: &Table) {
    let want = oracle(f, t);
    let w = f.sql_where(&NULLABLE);
    for fused in [false, true] {
        assert_eq!(kept(&w, t, fused), want, "{name}: fused={fused}");
    }
}

fn x(c: Cmp) -> Filter {
    Filter::cmp(X, c)
}
fn y(c: Cmp) -> Filter {
    Filter::cmp(Y, c)
}
fn not(f: Filter) -> Filter {
    Filter::negate(f)
}

/// The required cases. Each is checked on every (x valid/NULL) × (y
/// valid/NULL) combination, since the fixture is their full product.
#[test]
fn where_is_sql_three_valued_on_the_required_cases() {
    let t = table();
    let cases: Vec<(&str, Filter)> = vec![
        ("x = 5", x(Cmp::EqI32(5))),
        ("x <> 5", x(Cmp::NeI32(5))),
        ("NOT (x = 5)", not(x(Cmp::EqI32(5)))),
        (
            "x = 5 AND y = 7",
            Filter::and([x(Cmp::EqI32(5)), y(Cmp::EqI32(7))]),
        ),
        (
            "x = 5 OR y = 7",
            Filter::or([x(Cmp::EqI32(5)), y(Cmp::EqI32(7))]),
        ),
        (
            "NOT (x = 5 AND y = 7)",
            not(Filter::and([x(Cmp::EqI32(5)), y(Cmp::EqI32(7))])),
        ),
        (
            "NOT (x = 5 OR y = 7)",
            not(Filter::or([x(Cmp::EqI32(5)), y(Cmp::EqI32(7))])),
        ),
        (
            "x BETWEEN 1 AND 10",
            Filter::and([x(Cmp::GeI32(1)), x(Cmp::LeI32(10))]),
        ),
        (
            "x NOT BETWEEN 1 AND 10",
            not(Filter::and([x(Cmp::GeI32(1)), x(Cmp::LeI32(10))])),
        ),
        ("x IN (1,2,3)", Filter::in_i32(X, [1, 2, 3])),
        ("x NOT IN (1,2,3)", not(Filter::in_i32(X, [1, 2, 3]))),
        ("x IS NULL", Filter::is_null(VX)),
        ("x IS NOT NULL", Filter::is_not_null(VX)),
        ("NOT (x IS NULL)", not(Filter::is_null(VX))),
        (
            "x IS NULL OR x = 5",
            Filter::or([Filter::is_null(VX), x(Cmp::EqI32(5))]),
        ),
        ("NOT NOT (x = 5)", not(not(x(Cmp::EqI32(5))))),
    ];
    for (name, f) in &cases {
        check(name, f, &t);
    }

    // Anti-vacuity: the fixture must reach every Kleene outcome of the
    // composite cases, and each of the four validity combinations.
    let both = Filter::and([x(Cmp::EqI32(5)), y(Cmp::EqI32(7))]);
    let outcomes: Vec<Tv> = (0..t.cells.len()).map(|r| eval(&both, &t, r)).collect();
    for o in [Tv::T, Tv::F, Tv::U] {
        assert!(outcomes.contains(&o), "fixture never yields {o:?}");
    }
    for (vx, vy) in [(true, true), (false, true), (true, false), (false, false)] {
        assert!(
            t.cells
                .iter()
                .any(|(a, b)| a.v.is_some() == vx && b.v.is_some() == vy),
            "fixture lacks x valid={vx}, y valid={vy}"
        );
    }
}

/// FAILS IF: `NOT (x = 5)` is lowered as `NOT (valid(x) AND x = 5)`.
///
/// That form is the obvious one, and it is wrong: it holds on every NULL row,
/// where SQL's verdict is UNKNOWN and `WHERE` rejects the row. The raw,
/// unrewritten filter is wrong too, differently: it reads the payload of the
/// NULL rows. Both are shown to disagree with the oracle on this fixture, so
/// the oracle is not vacuous against them.
#[test]
fn gating_a_leaf_before_negation_is_not_three_valued() {
    let t = table();
    let f = not(x(Cmp::EqI32(5)));
    let want = oracle(&f, &t);
    let naive = not(Filter::and([Filter::is_not_null(VX), x(Cmp::EqI32(5))]));
    let nulls: Vec<usize> = (0..t.cells.len())
        .filter(|&r| t.cells[r].0.v.is_none())
        .collect();
    for fused in [false, true] {
        let got = kept(&naive, &t, fused);
        assert_ne!(got, want, "naive gating must be wrong (fused={fused})");
        assert!(
            nulls.iter().all(|r| got.contains(r)),
            "the naive form keeps every NULL row"
        );
        let raw = kept(&f, &t, fused);
        assert_ne!(raw, want, "the unrewritten filter reads NULL payloads");
        assert_eq!(kept(&f.sql_where(&NULLABLE), &t, fused), want);
    }
    assert!(
        nulls.iter().all(|r| !want.contains(r)),
        "WHERE rejects UNKNOWN"
    );
}

/// FAILS IF: NULL is treated as a value. `x = 0` keeps the rows whose x is a
/// real zero and none of the NULL rows whose payload is `0`; `x <> 0` keeps
/// neither kind of NULL.
#[test]
fn null_is_not_zero() {
    let t = table();
    let eq0 = x(Cmp::EqI32(0)).sql_where(&NULLABLE);
    let got = kept(&eq0, &t, false);
    assert!(!got.is_empty());
    assert!(got.iter().all(|&r| t.cells[r].0.v == Some(0)));
    let null_zero_payload: Vec<usize> = (0..t.cells.len())
        .filter(|&r| t.cells[r].0.v.is_none() && t.x[r] == 0)
        .collect();
    assert!(
        !null_zero_payload.is_empty(),
        "fixture needs a NULL with payload 0"
    );
    let ne0 = kept(&x(Cmp::NeI32(0)).sql_where(&NULLABLE), &t, false);
    assert!(null_zero_payload
        .iter()
        .all(|r| !got.contains(r) && !ne0.contains(r)));
}

fn lcg(s: &mut u64) -> u64 {
    *s = s
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    *s >> 33
}

fn random_filter(s: &mut u64, depth: u32) -> Filter {
    let leaf = |s: &mut u64| -> Filter {
        let col = if lcg(s).is_multiple_of(2) { X } else { Y };
        let v = [0, 3, 5, 7, 10][(lcg(s) % 5) as usize];
        match lcg(s) % 8 {
            0 => Filter::cmp(col, Cmp::EqI32(v)),
            1 => Filter::cmp(col, Cmp::NeI32(v)),
            2 => Filter::cmp(col, Cmp::LtI32(v)),
            3 => Filter::cmp(col, Cmp::GeI32(v)),
            4 => Filter::cmp(col, Cmp::GtI32(v)),
            5 => Filter::cmp(col, Cmp::LeI32(v)),
            6 => Filter::is_null(if col == X { VX } else { VY }),
            _ => Filter::is_not_null(if col == X { VX } else { VY }),
        }
    };
    if depth == 0 || lcg(s).is_multiple_of(4) {
        return leaf(s);
    }
    match lcg(s) % 3 {
        0 => Filter::negate(random_filter(s, depth - 1)),
        1 => Filter::and((0..2 + lcg(s) % 2).map(|_| random_filter(s, depth - 1))),
        _ => Filter::or((0..2 + lcg(s) % 2).map(|_| random_filter(s, depth - 1))),
    }
}

/// FAILS IF: any composition law is wrong — De Morgan duals, double
/// negation, or a leaf's validity on either polarity — over 300 random trees
/// of depth ≤ 3 mixing nullable comparisons, IS [NOT] NULL and NOT/AND/OR,
/// lowered both ways.
#[test]
fn random_trees_agree_with_the_oracle_both_lowerings() {
    let t = table();
    let mut s = 0x3_7a1_u64;
    let mut differs_from_raw = 0;
    for i in 0..300 {
        let f = random_filter(&mut s, 3);
        check(&format!("random #{i}: {f:?}"), &f, &t);
        if kept(&f, &t, false) != oracle(&f, &t) {
            differs_from_raw += 1;
        }
    }
    // Anti-vacuity: the raw two-valued lowering must be wrong on a real share
    // of these trees, or the corpus does not exercise NULL at all.
    assert!(
        differs_from_raw > 30,
        "only {differs_from_raw} trees exercised NULL"
    );
}

/// FAILS IF: a filter that reads no nullable column is rewritten. The
/// non-nullable fast path must lower to the identical program, both ways.
#[test]
fn non_nullable_filters_are_returned_unchanged() {
    let mut s = 0xfa57_u64;
    for _ in 0..50 {
        let f = random_filter(&mut s, 3);
        assert_eq!(f.sql_where(&[]), f);
        for fused in [false, true] {
            let lower_fn = if fused { lower_fused } else { lower };
            let q = |filter| Query {
                filter,
                agg: Agg::Count,
            };
            assert_eq!(
                lower_fn(&q(f.sql_where(&[]))).ok(),
                lower_fn(&q(f.clone())).ok(),
                "fused={fused}"
            );
        }
    }
    // Only y nullable: a subtree over x alone is reused as written.
    let f = Filter::and([x(Cmp::EqI32(5)), not(y(Cmp::EqI32(7)))]);
    let w = f.sql_where(&[(Y, VY)]);
    match &w {
        Filter::And(parts) => assert_eq!(parts[0], x(Cmp::EqI32(5))),
        other => panic!("expected an AND, got {other:?}"),
    }
}

/// FAILS IF: the aggregate conventions regress. `COUNT(*)` counts rows;
/// `COUNT(x)`, `SUM(x)`, `MIN(x)`, `MAX(x)` and `AVG(x)` see valid `x` only
/// (validity in their filter, as before this change); and an empty set of
/// contributions is distinguishable from a real zero.
#[test]
fn aggregates_keep_their_null_conventions() {
    let t = table();
    let n = t.cells.len();
    let (masks, lanes) = planes(&t);
    let pl = Planes {
        n_rows: n,
        masks: &masks,
        lanes: &lanes,
    };
    let run = |filter: Filter, agg: Agg| -> Value {
        let p = lower(&Query { filter, agg }).expect("lowers");
        let mut s = Scratch::for_program(&p, n).expect("carves");
        execute(&p, &pl, &mut s, None).expect("runs")
    };
    let w = y(Cmp::EqI32(7)).sql_where(&NULLABLE);
    let rows = oracle(&y(Cmp::EqI32(7)), &t);
    let xs: Vec<i32> = rows.iter().filter_map(|&r| t.cells[r].0.v).collect();
    assert!(xs.len() < rows.len(), "some selected rows must have NULL x");

    let with_x = Filter::and([w.clone(), Filter::is_not_null(VX)]);
    assert_eq!(
        run(w.clone(), Agg::Count),
        Value::Count(rows.len()),
        "COUNT(*)"
    );
    assert_eq!(
        run(with_x.clone(), Agg::Count),
        Value::Count(xs.len()),
        "COUNT(x)"
    );
    assert_eq!(
        run(with_x.clone(), Agg::SumI32(X)),
        Value::SumI64(xs.iter().map(|&v| i64::from(v)).sum()),
        "SUM(x)"
    );
    assert_eq!(
        run(with_x.clone(), Agg::MinI32(X)),
        Value::OptI32(xs.iter().copied().min())
    );
    assert_eq!(
        run(with_x.clone(), Agg::MaxI32(X)),
        Value::OptI32(xs.iter().copied().max())
    );

    let plan = lower_avg(&with_x, X).expect("lowers");
    let exec = |p: &lance_graph_mask_risc::Program| {
        let mut s = Scratch::for_program(p, n).expect("carves");
        execute(p, &pl, &mut s, None).expect("runs")
    };
    let (Value::SumI64(sum), Value::Count(cnt)) = (exec(&plan.sum), exec(&plan.count)) else {
        panic!("AVG plan shapes")
    };
    assert_eq!(
        avg_finish(sum, cnt as u64),
        Some(xs.iter().map(|&v| f64::from(v)).sum::<f64>() / xs.len() as f64)
    );

    // No contribution is not a real zero: the only-NULL-x group has COUNT(x)
    // = 0, MIN(x) = None and AVG(x) = None, while a valid x = 0 group has
    // COUNT(x) = 1 and MIN(x) = Some(0).
    let only_null = Filter::and([Filter::is_null(VX), Filter::is_not_null(VX)]);
    assert_eq!(run(only_null.clone(), Agg::Count), Value::Count(0));
    assert_eq!(run(only_null.clone(), Agg::MinI32(X)), Value::OptI32(None));
    let plan = lower_avg(&only_null, X).expect("lowers");
    let (Value::SumI64(sum), Value::Count(cnt)) = (exec(&plan.sum), exec(&plan.count)) else {
        panic!("AVG plan shapes")
    };
    assert_eq!(avg_finish(sum, cnt as u64), None);
    let zero = x(Cmp::EqI32(0)).sql_where(&NULLABLE);
    assert!(matches!(run(zero.clone(), Agg::Count), Value::Count(c) if c > 0));
    assert_eq!(run(zero, Agg::MinI32(X)), Value::OptI32(Some(0)));
}

/// FAILS IF: NULL is folded into the ordinal of `''` or into `false` on a
/// `u32` code lane. Both are ordinary values that happen to be encoded as
/// ordinal `0` (an empty string's codebook ordinal, a boolean's `false`);
/// NULL is the cleared validity bit, whatever ordinal the payload holds.
/// The same law as for `i32`, on the lane type text and booleans bind to.
#[test]
fn null_is_not_the_empty_string_or_false() {
    let n = 130;
    const Z: Col = Col(0);
    const VZ: Mask = Mask(0);
    // Every third row is NULL with payload 0; the rest alternate 0 / 1.
    let z: Vec<u32> = (0..n)
        .map(|r| if r % 3 == 0 { 0 } else { (r % 2) as u32 })
        .collect();
    let mut vz = vec![0u64; words_for(n)];
    for r in (0..n).filter(|r| r % 3 != 0) {
        vz[r / 64] |= 1 << (r % 64);
    }
    let masks: [&[u64]; 1] = [&vz];
    let lanes = [LaneRef::U32(&z)];
    let pl = Planes {
        n_rows: n,
        masks: &masks,
        lanes: &lanes,
    };
    let rows = |f: Filter| -> Vec<usize> {
        let p = lower(&Query {
            filter: f.sql_where(&[(Z, VZ)]),
            agg: Agg::Rows,
        })
        .expect("lowers");
        let mut s = Scratch::for_program(&p, n).expect("carves");
        let mut out = vec![0u64; words_for(n)];
        execute_into(&p, &pl, &Foreign::NONE, &mut s, Out::Mask(&mut out)).expect("runs");
        materialize_rows(&out, n)
    };
    let eq0 = rows(Filter::cmp(Z, Cmp::EqU32(0)));
    let ne0 = rows(Filter::cmp(Z, Cmp::NeU32(0)));
    let null = rows(Filter::is_null(VZ));
    let want_eq0: Vec<usize> = (0..n).filter(|&r| r % 3 != 0 && r % 2 == 0).collect();
    let want_ne0: Vec<usize> = (0..n).filter(|&r| r % 3 != 0 && r % 2 == 1).collect();
    let want_null: Vec<usize> = (0..n).filter(|&r| r % 3 == 0).collect();
    assert_eq!(eq0, want_eq0, "'' / false: real ordinal-0 rows only");
    assert_eq!(ne0, want_ne0, "a NULL row is neither = 0 nor <> 0");
    assert_eq!(null, want_null);
    assert_eq!(
        eq0.len() + ne0.len() + null.len(),
        n,
        "TRUE / FALSE / NULL partition the rows"
    );
}
