//! D-QSP-0 remainder, slice 1: scalar aggregates under SQL NULL.
//!
//! The SQL rules under test:
//! - `COUNT(*)` counts the rows `WHERE` keeps, NULLs included.
//! - `COUNT(x)` counts only the kept rows whose `x` is not NULL.
//! - `SUM(x)`, `MIN(x)`, `MAX(x)` and `AVG(x)` see non-NULL `x` only, and are
//!   NULL when no kept row has a non-NULL `x`. A sum of real values that
//!   happens to be `0` is `0`, not NULL.
//!
//! A nullable column is a value lane plus its validity plane. The fixtures put
//! a non-zero payload at every NULL row, so a path that reads a NULL payload
//! changes a sum or a minimum and is caught.
//!
//! The oracle never lowers anything. It evaluates the original filter in
//! Kleene logic position by position, reading the lanes and validity bits in
//! place, and folds the TRUE positions straight into the aggregates. No row
//! objects are built. PostgreSQL checks the same aggregates in
//! `examples/qsp_agg_pg_parity.rs`.

use lance_graph_mask_risc::{
    execute, execute_into, Foreign, LaneRef, Out, Planes, Program, Scratch, Value,
};
use lance_graph_quack::{
    avg_finish, lower, lower_avg, lower_sum, normalize_group_sink, sum_finish, Agg, Cmp, Col,
    Filter, GroupAddr, GroupAgg, Mask, Query,
};

const X: Col = Col(0);
const Y: Col = Col(1);
const VX: Mask = Mask(0);
const VY: Mask = Mask(1);
const NULLABLE: [(Col, Mask); 2] = [(X, VX), (Y, VY)];

/// Two nullable `i32` columns as lanes. A clear validity bit is NULL; the
/// value lane keeps a non-zero payload there.
struct Lanes {
    x: Vec<i32>,
    y: Vec<i32>,
    vx: Vec<u64>,
    vy: Vec<u64>,
}

impl Lanes {
    fn n(&self) -> usize {
        self.x.len()
    }

    /// `None` at a NULL position, read in place.
    fn x_at(&self, i: usize) -> Option<i32> {
        bit(&self.vx, i).then(|| self.x[i])
    }

    fn y_at(&self, i: usize) -> Option<i32> {
        bit(&self.vy, i).then(|| self.y[i])
    }
}

fn bit(w: &[u64], i: usize) -> bool {
    w[i / 64] >> (i % 64) & 1 == 1
}

/// Build lanes from `Option` cells; a NULL cell stores `payload` in the lane.
fn lanes(x: &[Option<i32>], y: &[Option<i32>], payload: i32) -> Lanes {
    let n = x.len();
    let words = n.div_ceil(64);
    let mut l = Lanes {
        x: vec![0; n],
        y: vec![0; n],
        vx: vec![0; words],
        vy: vec![0; words],
    };
    for i in 0..n {
        match x[i] {
            Some(v) => {
                l.x[i] = v;
                l.vx[i / 64] |= 1 << (i % 64);
            }
            None => l.x[i] = payload,
        }
        match y[i] {
            Some(v) => {
                l.y[i] = v;
                l.vy[i / 64] |= 1 << (i % 64);
            }
            None => l.y[i] = payload,
        }
    }
    l
}

// ── Kleene oracle ────────────────────────────────────────────────────────

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

fn cmp_holds(c: Cmp, v: i32) -> bool {
    match c {
        Cmp::EqI32(k) => v == k,
        Cmp::NeI32(k) => v != k,
        Cmp::LtI32(k) => v < k,
        Cmp::LeI32(k) => v <= k,
        Cmp::GtI32(k) => v > k,
        Cmp::GeI32(k) => v >= k,
        other => panic!("fixture uses only i32 comparisons, got {other:?}"),
    }
}

fn eval(f: &Filter, l: &Lanes, i: usize) -> Tv {
    match f {
        Filter::Cmp(col, c) => {
            let v = if *col == X { l.x_at(i) } else { l.y_at(i) };
            v.map_or(Tv::U, |v| tv(cmp_holds(*c, v)))
        }
        Filter::Plane(m) => tv(bit(if *m == VX { &l.vx } else { &l.vy }, i)),
        Filter::Not(g) => match eval(g, l, i) {
            Tv::T => Tv::F,
            Tv::F => Tv::T,
            Tv::U => Tv::U,
        },
        Filter::And(gs) => gs.iter().fold(Tv::T, |acc, g| match (acc, eval(g, l, i)) {
            (Tv::F, _) | (_, Tv::F) => Tv::F,
            (Tv::U, _) | (_, Tv::U) => Tv::U,
            _ => Tv::T,
        }),
        Filter::Or(gs) => gs.iter().fold(Tv::F, |acc, g| match (acc, eval(g, l, i)) {
            (Tv::T, _) | (_, Tv::T) => Tv::T,
            (Tv::U, _) | (_, Tv::U) => Tv::U,
            _ => Tv::F,
        }),
        other => panic!("fixture builds no {other:?}"),
    }
}

/// The SQL answers, folded over the positions `WHERE` keeps.
#[derive(Debug, PartialEq)]
struct Answers {
    count_star: usize,
    count_x: usize,
    sum: Option<i64>,
    min: Option<i32>,
    max: Option<i32>,
    avg: Option<f64>,
}

fn oracle(f: &Filter, l: &Lanes) -> Answers {
    let (mut count_star, mut count_x, mut sum) = (0usize, 0usize, 0i64);
    let (mut min, mut max) = (None::<i32>, None::<i32>);
    for i in 0..l.n() {
        if eval(f, l, i) != Tv::T {
            continue;
        }
        count_star += 1;
        if let Some(v) = l.x_at(i) {
            count_x += 1;
            sum += i64::from(v);
            min = Some(min.map_or(v, |m| m.min(v)));
            max = Some(max.map_or(v, |m| m.max(v)));
        }
    }
    Answers {
        count_star,
        count_x,
        sum: (count_x > 0).then_some(sum),
        min,
        max,
        avg: (count_x > 0).then(|| sum as f64 / count_x as f64),
    }
}

// ── Quack ────────────────────────────────────────────────────────────────

fn run(p: &Program, l: &Lanes) -> Value {
    let masks: [&[u64]; 2] = [&l.vx, &l.vy];
    let lanes = [LaneRef::I32(&l.x), LaneRef::I32(&l.y)];
    let pl = Planes {
        n_rows: l.n(),
        masks: &masks,
        lanes: &lanes,
    };
    let mut s = Scratch::for_program(p, l.n()).expect("carves");
    execute(p, &pl, &mut s, None).expect("runs")
}

fn count(filter: Filter, l: &Lanes) -> usize {
    let p = lower(&Query {
        filter,
        agg: Agg::Count,
    })
    .expect("lowers");
    match run(&p, l) {
        Value::Count(c) => c,
        v => panic!("COUNT shape {v:?}"),
    }
}

fn opt_i32(filter: Filter, agg: Agg, l: &Lanes) -> Option<i32> {
    let p = lower(&Query { filter, agg }).expect("lowers");
    match run(&p, l) {
        Value::OptI32(v) => v,
        v => panic!("MIN/MAX shape {v:?}"),
    }
}

/// `SUM(x)` through [`lower_sum`] and [`sum_finish`].
fn sql_sum(w: &Filter, l: &Lanes) -> Option<i64> {
    let plan = lower_sum(w, X, Some(VX)).expect("lowers");
    match (run(&plan.sum, l), run(&plan.count, l)) {
        (Value::SumI64(s), Value::Count(c)) => sum_finish(s, c as u64),
        v => panic!("SUM plan shapes {v:?}"),
    }
}

fn quack(f: &Filter, l: &Lanes) -> Answers {
    let w = f.sql_where(&NULLABLE);
    let with_x = Filter::and([w.clone(), Filter::is_not_null(VX)]);
    let plan = lower_avg(&with_x, X).expect("lowers");
    let avg = match (run(&plan.sum, l), run(&plan.count, l)) {
        (Value::SumI64(s), Value::Count(c)) => avg_finish(s, c as u64),
        v => panic!("AVG plan shapes {v:?}"),
    };
    Answers {
        count_star: count(w.clone(), l),
        count_x: count(with_x.clone(), l),
        sum: sql_sum(&w, l),
        min: opt_i32(with_x.clone(), Agg::MinI32(X), l),
        max: opt_i32(with_x, Agg::MaxI32(X), l),
        avg,
    }
}

// ── Tests ────────────────────────────────────────────────────────────────

/// FAILS IF: an empty or all-NULL `SUM(x)` answers `0`, a real zero sum
/// answers NULL, or a NULL payload reaches the sum. The fixture's NULL
/// payload is `9`.
#[test]
fn sum_of_no_contribution_is_null_and_a_real_zero_is_zero() {
    // rows: x = NULL, NULL, 0, 0, 2, -2;  y = 1, 1, 2, 2, 3, 3
    let l = lanes(
        &[None, None, Some(0), Some(0), Some(2), Some(-2)],
        &[Some(1), Some(1), Some(2), Some(2), Some(3), Some(3)],
        9,
    );
    let y_is = |k| Filter::cmp(Y, Cmp::EqI32(k)).sql_where(&NULLABLE);

    // WHERE keeps nothing.
    assert_eq!(sql_sum(&y_is(7), &l), None, "empty selection");
    // WHERE keeps two rows whose x are both NULL.
    assert_eq!(count(y_is(1), &l), 2, "the all-NULL selection is not empty");
    assert_eq!(sql_sum(&y_is(1), &l), None, "all-NULL selection");
    // Real values summing to zero.
    assert_eq!(sql_sum(&y_is(2), &l), Some(0), "0 + 0");
    assert_eq!(sql_sum(&y_is(3), &l), Some(0), "2 + -2");
    // Everything: the two NULL payloads (9 each) must not appear.
    let all = Filter::is_not_null(VY);
    assert_eq!(sql_sum(&all, &l), Some(0), "NULL payloads excluded");

    // Negative control: the lone scalar fold cannot tell the cases apart.
    let naive = |w: Filter| {
        let p = lower(&Query {
            filter: Filter::and([w, Filter::is_not_null(VX)]),
            agg: Agg::SumI32(X),
        })
        .expect("lowers");
        run(&p, &l)
    };
    assert_eq!(naive(y_is(1)), naive(y_is(2)), "both read as SumI64(0)");
}

/// FAILS IF: an empty `WHERE` stops answering ungrouped `SUM(x)` with one
/// NULL, or starts answering the grouped form with any group. PostgreSQL:
/// `SELECT sum(x) FROM t WHERE false` is one row holding NULL;
/// `SELECT k, sum(x) FROM t WHERE false GROUP BY k` is zero rows. A
/// non-empty selection must still produce groups, or the zero-groups check
/// proves nothing. (Map probe P-EMPTYSUM. A reached group whose `x` are all
/// NULL is a `GROUP BY` question and is not pinned here.)
#[test]
fn empty_where_is_one_null_ungrouped_and_no_group_grouped() {
    let l = lanes(
        &[None, Some(4), Some(0), Some(-1)],
        &[Some(1), Some(1), Some(2), Some(2)],
        9,
    );
    let keys: Vec<u32> = vec![0, 1, 1, 2];
    let masks: [&[u64]; 2] = [&l.vx, &l.vy];
    let lanes_ = [LaneRef::I32(&l.x), LaneRef::I32(&l.y), LaneRef::U32(&keys)];
    let pl = Planes {
        n_rows: l.n(),
        masks: &masks,
        lanes: &lanes_,
    };
    let present_groups = |filter: Filter| -> usize {
        let p = lower(&Query {
            filter: Filter::and([filter, Filter::is_not_null(VX)]),
            agg: Agg::GroupReduce {
                key: GroupAddr::Local(Col(2)),
                agg: GroupAgg::SumI32(X),
            },
        })
        .expect("lowers");
        let mut s = Scratch::for_program(&p, l.n()).expect("carves");
        let mut sink = vec![0i64; 3];
        execute_into(&p, &pl, &Foreign::NONE, &mut s, Out::I64(&mut sink)).expect("runs");
        normalize_group_sink(GroupAgg::SumI32(X).fold(), &mut sink)
            .iter()
            .map(|w| w.count_ones() as usize)
            .sum()
    };
    let nothing = Filter::cmp(Y, Cmp::EqI32(7)).sql_where(&NULLABLE);

    assert_eq!(sql_sum(&nothing, &l), None, "ungrouped: one NULL answer");
    assert_eq!(present_groups(nothing), 0, "grouped: no group at all");
    let everything = Filter::is_not_null(VY);
    assert_eq!(
        present_groups(everything),
        2,
        "groups 1 and 2 have a valid x"
    );
}

/// FAILS IF: any of `COUNT(*)`, `COUNT(x)`, `SUM(x)`, `MIN(x)`, `MAX(x)`,
/// `AVG(x)` disagrees with the Kleene oracle over random three-valued `WHERE`
/// clauses, or the corpus stops containing the cases this slice is about
/// (empty selections, all-NULL selections, real zero sums).
#[test]
fn random_where_clauses_agree_with_the_kleene_oracle() {
    let mut rng = Rng(0x9E37_79B9_7F4A_7C15);
    let (mut queries, mut empty, mut all_null, mut real_zero) = (0, 0, 0, 0);
    for t in 0..150 {
        let n = 1 + rng.below(130) as usize;
        // NULL rate for x cycles through 0 %, 30 %, 70 % and 100 %.
        let null_x = [0, 30, 70, 100][t % 4];
        let cell =
            |rng: &mut Rng, pct: u64| (rng.below(100) >= pct).then(|| rng.below(9) as i32 - 4);
        let x: Vec<_> = (0..n).map(|_| cell(&mut rng, null_x)).collect();
        let y: Vec<_> = (0..n).map(|_| cell(&mut rng, 30)).collect();
        let l = lanes(&x, &y, 3);
        for _ in 0..8 {
            let f = random_filter(&mut rng, 3);
            let want = oracle(&f, &l);
            assert_eq!(quack(&f, &l), want, "table {t}: {f:?}");
            queries += 1;
            empty += usize::from(want.count_star == 0);
            all_null += usize::from(want.count_star > 0 && want.count_x == 0);
            real_zero += usize::from(want.sum == Some(0));
        }
    }
    assert!(empty > 0, "no empty selection in {queries} queries");
    assert!(all_null > 0, "no all-NULL selection in {queries} queries");
    assert!(real_zero > 0, "no real zero sum in {queries} queries");
}

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

fn random_filter(rng: &mut Rng, depth: u32) -> Filter {
    if depth == 0 || rng.below(3) == 0 {
        let (col, valid) = if rng.below(2) == 0 { (X, VX) } else { (Y, VY) };
        let k = rng.below(11) as i32 - 5;
        return match rng.below(8) {
            0 => Filter::is_null(valid),
            1 => Filter::is_not_null(valid),
            2 => Filter::cmp(col, Cmp::EqI32(k)),
            3 => Filter::cmp(col, Cmp::NeI32(k)),
            4 => Filter::cmp(col, Cmp::LtI32(k)),
            5 => Filter::cmp(col, Cmp::LeI32(k)),
            6 => Filter::cmp(col, Cmp::GtI32(k)),
            _ => Filter::cmp(col, Cmp::GeI32(k)),
        };
    }
    let kids = |rng: &mut Rng| {
        (0..2 + rng.below(2))
            .map(|_| random_filter(rng, depth - 1))
            .collect::<Vec<_>>()
    };
    match rng.below(3) {
        0 => Filter::And(kids(rng)),
        1 => Filter::Or(kids(rng)),
        _ => Filter::negate(random_filter(rng, depth - 1)),
    }
}
