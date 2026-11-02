//! D-MPF-1 gates. Every answer is compared with `oracle`, a row loop that
//! shares no code with the procedures, the dialect or mask-risc.

use lance_graph_mask_risc::{words_for, Foreign, LaneRef, Planes};
use lance_graph_r2il::procs::{count, select_lt};
use lance_graph_r2il::{
    FoldDialect, FoldError, Val, INT_ADD, INT_LESS, INT_MULT, INT_SUB, INT_S_DIV, LOAD,
};
use lance_graph_relalg::{run, Answer, Cmp, Measure, Query, Rel};
use math_polyfill_probe::{eval, oracle, Expr, Layout};

const L: Layout = Layout {
    x: 0,
    y: 1,
    v: 2,
    valid: 0,
};

struct Fix {
    x: Vec<i32>,
    y: Vec<i32>,
    v: Vec<i32>,
    valid: Vec<u64>,
}

/// `n` rows; every 7th `v` is NULL. Values are signed so a sum can cancel.
fn fix(n: usize) -> Fix {
    let mut valid = vec![0u64; words_for(n)];
    for r in (0..n).filter(|r| r % 7 != 3) {
        valid[r / 64] |= 1 << (r % 64);
    }
    Fix {
        x: (0..n).map(|i| (i % 50) as i32 - 20).collect(),
        y: (0..n).map(|i| ((i * 7) % 31) as i32).collect(),
        v: (0..n).map(|i| ((i * 13) % 97) as i32 - 48).collect(),
        valid,
    }
}

fn with_planes<R>(f: &Fix, body: impl FnOnce(&Planes<'_>) -> R) -> R {
    let lanes = [LaneRef::I32(&f.x), LaneRef::I32(&f.y), LaneRef::I32(&f.v)];
    let masks: [&[u64]; 1] = [&f.valid];
    let planes = Planes {
        n_rows: f.v.len(),
        masks: &masks,
        lanes: &lanes,
    };
    body(&planes)
}

/// Empty, partial and full selections, and several multipliers.
const EXPRS: [Expr; 5] = [
    Expr {
        lo: 100,
        hi: 31,
        k: 3,
    }, // x ≥ 100 keeps nothing
    Expr {
        lo: -5,
        hi: 20,
        k: 3,
    }, // partial
    Expr {
        lo: -20,
        hi: 31,
        k: 0,
    }, // every present row
    Expr {
        lo: 0,
        hi: 10,
        k: -7,
    }, // negative multiplier
    Expr {
        lo: -20,
        hi: 31,
        k: 1,
    }, // every present row, k = 1
];

/// FAILS IF: any procedure, the scalar arithmetic, or the NULL handling
/// diverges from the row loop, at one word, a ragged tail, and many tiles.
#[test]
fn the_polyfill_expression_equals_the_oracle() {
    for n in [64, 1_000, 4_096, 65_537] {
        let f = fix(n);
        for e in EXPRS {
            let (_, want) = oracle(&f.x, &f.y, &f.v, &f.valid, e);
            let got = with_planes(&f, |p| {
                let mut d = FoldDialect::new(p, &Foreign::NONE);
                eval(&mut d, L, e)
            });
            assert_eq!(got.ok(), want, "n = {n}, {e:?}");
        }
    }
}

/// FAILS IF: the fixture stops exercising what it claims (anti-vacuity): one
/// case must select nothing, one must select a strict subset, and NULL rows
/// must exist inside the otherwise-selected range.
#[test]
fn the_fixture_covers_empty_partial_and_null() {
    let f = fix(4_096);
    let sel = |e| oracle(&f.x, &f.y, &f.v, &f.valid, e).0;
    let present = f
        .valid
        .iter()
        .map(|w| u64::from(w.count_ones()))
        .sum::<u64>();
    assert_eq!(sel(EXPRS[0]), 0);
    let partial = sel(EXPRS[1]);
    assert!(
        partial * 10 > present && partial < present,
        "partial = {partial} of {present}"
    );
    assert_eq!(sel(EXPRS[2]), present);
    assert!(present < 4_096, "no NULL rows");
}

/// FAILS IF: the NULL plane stops gating the sum. With every row present the
/// answer must change, so the plane is load-bearing (can-fire), and the
/// polyfill must still match the oracle on the all-present fixture.
#[test]
fn null_rows_change_the_answer() {
    let mut f = fix(1_000);
    let e = EXPRS[2];
    let with_nulls = oracle(&f.x, &f.y, &f.v, &f.valid, e).1;
    for w in f.valid.iter_mut() {
        *w = u64::MAX;
    }
    *f.valid.last_mut().unwrap() &= (1u64 << (1_000 % 64)) - 1;
    let all_present = oracle(&f.x, &f.y, &f.v, &f.valid, e).1;
    assert_ne!(with_nulls, all_present);
    let got = with_planes(&f, |p| eval(&mut FoldDialect::new(p, &Foreign::NONE), L, e));
    assert_eq!(got.ok(), all_present);
}

/// FAILS IF: the two frontends over the same substrate disagree: relalg's
/// `Query` (normalize + lower) and the procedures must count the same rows.
#[test]
fn relalg_and_the_procedures_count_the_same_rows() {
    let f = fix(4_096);
    for e in EXPRS {
        let q = Query {
            rel: Rel::And(vec![
                Rel::Plane(L.valid),
                Rel::i32(L.x, Cmp::Ge, e.lo),
                Rel::i32(L.y, Cmp::Lt, e.hi),
            ]),
            measure: Measure::Count,
        };
        let rel = with_planes(&f, |p| run(&q, p, None).expect("relalg").answer);
        let (want, _) = oracle(&f.x, &f.y, &f.v, &f.valid, e);
        assert_eq!(rel, Answer::Count(want), "{e:?}");
    }
}

// ── typed refusals: each is an existing FoldError, never a wrong answer ─────

fn stack_err(
    n: usize,
    body: impl FnOnce(&mut FoldDialect<'_>, &mut Vec<Val>) -> Result<(), FoldError>,
) -> FoldError {
    let f = fix(n);
    with_planes(&f, |p| {
        let mut d = FoldDialect::new(p, &Foreign::NONE);
        let mut s = Vec::new();
        body(&mut d, &mut s).expect_err("must refuse")
    })
}

fn load_i32(d: &mut FoldDialect<'_>, s: &mut Vec<Val>, lane: u16) -> Result<(), FoldError> {
    s.push(Val::Scalar(i64::from(lane)));
    d.call(LOAD, [1, 0, 0], s)
}

/// FAILS IF: lane + lane arithmetic is silently accepted. The fold path has
/// no element-wise value arithmetic; that is the measured Stage 1 gap.
#[test]
fn lane_plus_lane_is_refused() {
    let e = stack_err(64, |d, s| {
        load_i32(d, s, L.x)?;
        load_i32(d, s, L.y)?;
        d.call(INT_ADD, [0; 3], s)
    });
    assert_eq!(e, FoldError::LaneVersusLane(INT_ADD));
}

/// FAILS IF: an unsigned lane compare is coerced onto the signed one.
#[test]
fn unsigned_compare_is_refused() {
    let e = stack_err(64, |d, s| {
        load_i32(d, s, L.x)?;
        s.push(Val::Scalar(0));
        d.call(INT_LESS, [0; 3], s)
    });
    assert_eq!(e, FoldError::NoUnsignedLaneCompare(INT_LESS));
}

/// FAILS IF: scalar overflow wraps instead of refusing. The silent twin
/// (a product that fits) must succeed, so the guard is not always-on.
#[test]
fn scalar_overflow_is_refused_and_a_fitting_product_is_not() {
    let e = stack_err(64, |d, s| {
        s.push(Val::Scalar(i64::MAX / 2 + 1));
        s.push(Val::Scalar(2));
        d.call(INT_MULT, [0; 3], s)
    });
    assert_eq!(e, FoldError::Overflow(INT_MULT));
    let f = fix(64);
    let ok = with_planes(&f, |p| {
        let mut d = FoldDialect::new(p, &Foreign::NONE);
        let mut s = vec![Val::Scalar(i64::MAX / 2), Val::Scalar(2)];
        d.call(INT_MULT, [0; 3], &mut s).map(|()| s)
    });
    assert_eq!(ok.unwrap(), vec![Val::Scalar(i64::MAX / 2 * 2)]);
}

/// FAILS IF: an expression whose exact value leaves i64 is answered. The
/// oracle says it does not fit; the polyfill must refuse, not wrap.
#[test]
fn an_expression_that_leaves_i64_is_refused() {
    let f = fix(1_000);
    let e = Expr {
        lo: -20,
        hi: 31,
        k: i64::MAX,
    };
    assert_eq!(oracle(&f.x, &f.y, &f.v, &f.valid, e).1, None);
    let got = with_planes(&f, |p| eval(&mut FoldDialect::new(p, &Foreign::NONE), L, e));
    assert_eq!(got, Err(FoldError::Overflow(INT_MULT)));
}

/// FAILS IF: the final subtraction wraps. One selected row with `v = -2`
/// and `k = i64::MAX`: the product `k * 1` fits, `-2 - i64::MAX` does not.
/// Found by review on #1752; the oracle already said `None`.
#[test]
fn an_expression_that_overflows_at_the_subtraction_is_refused() {
    let f = Fix {
        x: vec![0],
        y: vec![0],
        v: vec![-2],
        valid: vec![1],
    };
    let e = Expr {
        lo: 0,
        hi: 1,
        k: i64::MAX,
    };
    assert_eq!(oracle(&f.x, &f.y, &f.v, &f.valid, e), (1, None));
    let got = with_planes(&f, |p| eval(&mut FoldDialect::new(p, &Foreign::NONE), L, e));
    assert_eq!(got, Err(FoldError::Overflow(INT_SUB)));
    // Silent twin: the same subtraction with room to spare answers.
    let fits = Expr {
        k: i64::MAX - 2,
        ..e
    };
    let got = with_planes(&f, |p| {
        eval(&mut FoldDialect::new(p, &Foreign::NONE), L, fits)
    });
    assert_eq!(got, Ok(-2 - (i64::MAX - 2)));
}

/// FAILS IF: division by zero answers instead of refusing.
#[test]
fn division_by_zero_is_refused() {
    let e = stack_err(64, |d, s| {
        select_lt(d, s, L.x, 0)?;
        count(d, s)?;
        s.push(Val::Scalar(0));
        d.call(INT_S_DIV, [0; 3], s)
    });
    assert_eq!(e, FoldError::DivisionByZero(INT_S_DIV));
}
