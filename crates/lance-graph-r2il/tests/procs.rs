//! D-SQLP-2: procedures, `IntOr`, and the two Boolean modes.
//!
//! Every answer is compared against a scalar reference computed row by row
//! from the same lanes. Fixtures have four groups and are run with an empty
//! selection, a partial one and a full one.

use lance_graph_mask_risc::{words_for, Foreign, LaneRef, MaskOp, Planes};
use lance_graph_r2il::procs::{
    all, and_between_i32, and_cmp_i32, avg_parts, bool_and, bool_or, count, count_and,
    count_distinct_sorted, count_union_all, count_valid, differ, except, exists, filter_lt,
    group_sum, in_u32, is_not_null, is_null, keep, margin, max, min, not_exists, select_lt,
    select_u32, sum, union, Order, Select,
};
use lance_graph_r2il::{
    BoolMode, FoldDialect, FoldError, Val, INT_ADD, INT_AND, INT_MULT, INT_NOT, INT_OR, INT_S_DIV,
    INT_XOR, POP_COUNT,
};
use ogar_r2il::KEEP;

const N: usize = 1_000;
const GROUPS: usize = 4;

// Lane indexes in every fixture below.
const X: u16 = 0; // i32, i % 50
const Y: u16 = 1; // i32, (i * 7) % 31
const REVENUE: u16 = 2; // i32
const COST: u16 = 3; // i32
const KEY: u16 = 4; // u32, i % GROUPS

struct Fixture {
    x: Vec<i32>,
    y: Vec<i32>,
    revenue: Vec<i32>,
    cost: Vec<i32>,
    key: Vec<u32>,
}

fn fixture() -> Fixture {
    Fixture {
        x: (0..N).map(|i| (i % 50) as i32).collect(),
        y: (0..N).map(|i| ((i * 7) % 31) as i32).collect(),
        revenue: (0..N).map(|i| (i % 13) as i32 + 10).collect(),
        cost: (0..N).map(|i| (i % 5) as i32).collect(),
        key: (0..N).map(|i| (i % GROUPS) as u32).collect(),
    }
}

fn lanes(f: &Fixture) -> [LaneRef<'_>; 5] {
    [
        LaneRef::I32(&f.x),
        LaneRef::I32(&f.y),
        LaneRef::I32(&f.revenue),
        LaneRef::I32(&f.cost),
        LaneRef::U32(&f.key),
    ]
}

/// `x < t` thresholds: empty, partial, full.
const THRESHOLDS: [i32; 3] = [0, 17, 50];

fn rows(m: &[u64]) -> Vec<usize> {
    (0..N).filter(|&r| m[r / 64] >> (r % 64) & 1 == 1).collect()
}

/// FAILS IF: `filter_lt` stops writing exactly the rows `x < t` (dropping its
/// `KEEP` leaves the sink empty and fails the partial and full cases).
#[test]
fn filter_lt_keeps_the_rows_the_reference_keeps() {
    let f = fixture();
    let lanes = lanes(&f);
    let planes = Planes {
        n_rows: N,
        masks: &[],
        lanes: &lanes,
    };
    for t in THRESHOLDS {
        let mut sink = vec![0u64; words_for(N)];
        {
            let mut d = FoldDialect::with_keep(&planes, &Foreign::NONE, &mut sink);
            let mut stack = vec![];
            filter_lt(&mut d, &mut stack, X, t).expect("filter_lt");
            assert!(stack.is_empty());
        }
        let want: Vec<usize> = (0..N).filter(|&r| f.x[r] < t).collect();
        assert_eq!(rows(&sink), want, "t = {t}");
    }
}

/// FAILS IF: `select_u32` stops keeping exactly the live rows that pass every
/// condition. The live plane drops every seventh row, so a selection that
/// ignores the plane counts too many; `!=` and `=` are both exercised.
#[test]
fn select_u32_counts_live_rows_passing_every_condition() {
    let f = fixture();
    let lanes = lanes(&f);
    let mut live = vec![0u64; words_for(N)];
    for r in (0..N).filter(|r| r % 7 != 0) {
        live[r / 64] |= 1 << (r % 64);
    }
    let planes = Planes {
        n_rows: N,
        masks: &[live.as_slice()],
        lanes: &lanes,
    };
    type Case = (&'static [(u16, u32, bool)], fn(usize) -> bool);
    let cases: [Case; 4] = [
        (&[], |_| true),
        (&[(KEY, 3, false)], |k| k == 3),
        (&[(KEY, 3, true)], |k| k != 3),
        (&[(KEY, 3, true), (KEY, 1, true)], |k| k != 3 && k != 1),
    ];
    for (conds, keep) in cases {
        let want = (0..N)
            .filter(|&r| r % 7 != 0 && keep(f.key[r] as usize))
            .count() as i64;
        let mut d = FoldDialect::new(&planes, &Foreign::NONE);
        let mut stack = vec![];
        select_u32(&mut d, &mut stack, 0, conds).expect("select");
        count(&mut d, &mut stack).expect("count");
        assert_eq!(stack, vec![Val::Scalar(want)], "{conds:?}");
        assert_eq!(d.programs_run(), 1, "one fold");
    }
    // Non-vacuity: the plane removes rows that pass the conditions.
    assert!((0..N).any(|r| r % 7 == 0 && f.key[r] == 3));
}

/// FAILS IF: `sum` stops matching `SUM(revenue | x < t)`.
#[test]
fn sum_matches_the_reference() {
    let f = fixture();
    let lanes = lanes(&f);
    let planes = Planes {
        n_rows: N,
        masks: &[],
        lanes: &lanes,
    };
    for t in THRESHOLDS {
        let want: i64 = (0..N)
            .filter(|&r| f.x[r] < t)
            .map(|r| i64::from(f.revenue[r]))
            .sum();
        let mut d = FoldDialect::new(&planes, &Foreign::NONE);
        let mut stack = vec![];
        select_lt(&mut d, &mut stack, X, t).expect("select");
        sum(&mut d, &mut stack, REVENUE).expect("sum");
        assert_eq!(stack, vec![Val::Scalar(want)], "t = {t}");
    }
}

/// FAILS IF: `group_sum` stops matching a per-group reference over four
/// groups.
#[test]
fn group_sum_matches_the_reference() {
    let f = fixture();
    let lanes = lanes(&f);
    let planes = Planes {
        n_rows: N,
        masks: &[],
        lanes: &lanes,
    };
    for t in THRESHOLDS {
        let mut want = [0i64; GROUPS];
        for r in (0..N).filter(|&r| f.x[r] < t) {
            want[f.key[r] as usize] += i64::from(f.revenue[r]);
        }
        let mut sink = [0i64; GROUPS];
        {
            let mut d = FoldDialect::with_groups(&planes, &Foreign::NONE, &mut sink);
            let mut stack = vec![];
            select_lt(&mut d, &mut stack, X, t).expect("select");
            group_sum(&mut d, &mut stack, KEY, REVENUE).expect("group_sum");
            assert!(stack.is_empty());
        }
        assert_eq!(sink, want, "t = {t}");
    }
}

/// FAILS IF: `margin` stops being `SUM(revenue) - SUM(cost)` over the
/// selection. The fixture keeps revenue and cost different everywhere, so
/// swapping the `IntSub` operands flips the sign and fails.
#[test]
fn margin_is_revenue_minus_cost() {
    let f = fixture();
    let lanes = lanes(&f);
    let planes = Planes {
        n_rows: N,
        masks: &[],
        lanes: &lanes,
    };
    for t in THRESHOLDS {
        let want: i64 = (0..N)
            .filter(|&r| f.x[r] < t)
            .map(|r| i64::from(f.revenue[r] - f.cost[r]))
            .sum();
        let mut d = FoldDialect::new(&planes, &Foreign::NONE);
        let mut stack = vec![];
        margin(
            &mut d,
            &mut stack,
            &|d, s| select_lt(d, s, X, t),
            REVENUE,
            COST,
        )
        .expect("margin");
        assert_eq!(stack, vec![Val::Scalar(want)], "t = {t}");
        assert_eq!(d.programs_run(), 2, "one fold per SUM");
    }
    // Non-vacuity: the partial case has a positive margin, so a sign flip
    // cannot pass it.
    assert!((0..N).any(|r| f.x[r] < 17 && f.revenue[r] != f.cost[r]));
}

/// `(x < a) OP (y < b)` counted, in `mode`. Returns the count and the ops of
/// the program that ran.
fn count_two(
    planes: &Planes<'_>,
    mode: BoolMode,
    op: ogar_loco::FnIndex,
    a: i32,
    b: i32,
) -> (i64, Vec<MaskOp>) {
    let mut d = FoldDialect::new(planes, &Foreign::NONE).with_mode(mode);
    let mut stack = vec![];
    select_lt(&mut d, &mut stack, X, a).expect("x < a");
    select_lt(&mut d, &mut stack, Y, b).expect("y < b");
    d.call(op, [0; 3], &mut stack).expect("combine");
    d.call(POP_COUNT, [0; 3], &mut stack).expect("count");
    let Some(Val::Scalar(n)) = stack.pop() else {
        panic!("count pushes a scalar");
    };
    (n, d.last_program().expect("ran").ops.clone())
}

/// FAILS IF: `IntOr` is lowered as anything but a union. The thresholds are
/// chosen so the union and the intersection differ.
#[test]
fn int_or_is_a_union_in_both_modes() {
    let f = fixture();
    let lanes = lanes(&f);
    let planes = Planes {
        n_rows: N,
        masks: &[],
        lanes: &lanes,
    };
    let union = (0..N).filter(|&r| f.x[r] < 10 || f.y[r] < 5).count() as i64;
    let inter = (0..N).filter(|&r| f.x[r] < 10 && f.y[r] < 5).count() as i64;
    assert_ne!(union, inter, "the fixture must separate OR from AND");
    for mode in [BoolMode::InPlace, BoolMode::Fused] {
        let (n, _) = count_two(&planes, mode, INT_OR, 10, 5);
        assert_eq!(n, union, "{mode:?}");
    }
}

/// FAILS IF: the two modes stop keeping the same rows, the in-place mode
/// stops emitting one op per Boolean call, or the fused mode stops lowering
/// the combination to ternlogs.
#[test]
fn both_modes_keep_the_same_rows_with_different_ops() {
    let f = fixture();
    let lanes = lanes(&f);
    let planes = Planes {
        n_rows: N,
        masks: &[],
        lanes: &lanes,
    };
    // `(x < 20 OR y < 7) AND NOT (x < 3)`, kept.
    let body = |d: &mut FoldDialect<'_>, s: &mut Vec<Val>| -> Result<(), FoldError> {
        select_lt(d, s, X, 20)?;
        select_lt(d, s, Y, 7)?;
        d.call(INT_OR, [0; 3], s)?;
        select_lt(d, s, X, 3)?;
        d.call(INT_NOT, [0; 3], s)?;
        d.call(INT_AND, [0; 3], s)?;
        d.call(KEEP, [0; 3], s)
    };
    let want: Vec<usize> = (0..N)
        .filter(|&r| (f.x[r] < 20 || f.y[r] < 7) && f.x[r] >= 3)
        .collect();
    assert!(!want.is_empty() && want.len() < N);

    let mut kept = Vec::new();
    for mode in [BoolMode::InPlace, BoolMode::Fused] {
        let mut sink = vec![0u64; words_for(N)];
        let ops = {
            let mut d = FoldDialect::with_keep(&planes, &Foreign::NONE, &mut sink).with_mode(mode);
            let mut stack = vec![];
            body(&mut d, &mut stack).expect("body");
            d.last_program().expect("ran").ops.clone()
        };
        assert_eq!(rows(&sink), want, "{mode:?}");
        let ternlogs = ops
            .iter()
            .filter(|o| matches!(o, MaskOp::Ternlog { .. }))
            .count();
        let per_call = ops
            .iter()
            .filter(|o| {
                matches!(
                    o,
                    MaskOp::Or { .. } | MaskOp::Not { .. } | MaskOp::And { .. }
                )
            })
            .count();
        match mode {
            BoolMode::InPlace => {
                assert_eq!(ternlogs, 0, "in-place emits no ternlog: {ops:?}");
                assert!(per_call >= 2, "in-place emits per-call ops: {ops:?}");
            }
            BoolMode::Fused => {
                assert!(ternlogs >= 1, "fused holds ternlogs: {ops:?}");
                assert_eq!(per_call, 0, "fused emits no per-call op: {ops:?}");
            }
        }
        kept.push(rows(&sink));
    }
    assert_eq!(kept[0], kept[1]);
}

/// FAILS IF: a fused expression consumed by a stale slot is folded instead
/// of refused.
#[test]
fn a_fused_slot_from_an_earlier_fold_is_refused() {
    let f = fixture();
    let lanes = lanes(&f);
    let planes = Planes {
        n_rows: N,
        masks: &[],
        lanes: &lanes,
    };
    let mut d = FoldDialect::new(&planes, &Foreign::NONE).with_mode(BoolMode::Fused);
    let mut stack = vec![];
    select_lt(&mut d, &mut stack, X, 20).unwrap();
    select_lt(&mut d, &mut stack, Y, 7).unwrap();
    d.call(INT_OR, [0; 3], &mut stack).unwrap();
    let stale = stack[0];
    d.call(POP_COUNT, [0; 3], &mut stack).unwrap();
    let mut stack = vec![stale];
    assert_eq!(
        d.call(POP_COUNT, [0; 3], &mut stack),
        Err(FoldError::StaleSlot)
    );
}

/// FAILS IF: an `IntAnd` of two resident planes stops being accepted in
/// either mode, or the two modes stop agreeing on it.
#[test]
fn plane_and_plane_is_accepted_in_both_modes() {
    let f = fixture();
    let lanes = lanes(&f);
    let pa: Vec<u64> = (0..words_for(N))
        .map(|w| 0xF0F0_F0F0_F0F0_F0F0 ^ w as u64)
        .collect();
    let pb: Vec<u64> = (0..words_for(N))
        .map(|w| 0x3333_3333_3333_3333u64.rotate_left(w as u32))
        .collect();
    // mask-risc refuses set bits past `n_rows` in a resident plane.
    let tail = |mut w: Vec<u64>| {
        let last = w.len() - 1;
        w[last] &= (1u64 << (N % 64)) - 1;
        w
    };
    let (pa, pb) = (tail(pa), tail(pb));
    let masks = [pa.as_slice(), pb.as_slice()];
    let planes = Planes {
        n_rows: N,
        masks: &masks,
        lanes: &lanes,
    };
    let want = (0..N)
        .filter(|&r| pa[r / 64] >> (r % 64) & 1 == 1 && pb[r / 64] >> (r % 64) & 1 == 1)
        .count() as i64;
    assert!(want > 0 && want < N as i64);
    for mode in [BoolMode::InPlace, BoolMode::Fused] {
        let mut d = FoldDialect::new(&planes, &Foreign::NONE).with_mode(mode);
        let mut stack = vec![
            Val::Address(lance_graph_r2il::Addr::Plane(0)),
            Val::Address(lance_graph_r2il::Addr::Plane(1)),
        ];
        d.call(INT_AND, [0; 3], &mut stack)
            .expect("plane AND plane");
        d.call(POP_COUNT, [0; 3], &mut stack).expect("count");
        assert_eq!(stack, vec![Val::Scalar(want)], "{mode:?}");
    }
}

/// FAILS IF: a fold refuses, or misreads, a resident plane as its mask.
/// A body with no condition hands the fold the live plane itself, not a
/// computed slot; SUM, GROUP_SUM and KEEP must read it where it stands, as
/// POP_COUNT does.
#[test]
fn every_fold_takes_a_resident_plane_as_its_mask() {
    let f = fixture();
    let lanes = lanes(&f);
    let mut live = vec![0u64; words_for(N)];
    for r in (0..N).filter(|r| r % 7 != 0) {
        live[r / 64] |= 1 << (r % 64);
    }
    let planes = Planes {
        n_rows: N,
        masks: &[live.as_slice()],
        lanes: &lanes,
    };
    let kept: Vec<usize> = (0..N).filter(|r| r % 7 != 0).collect();
    let plane = || vec![Val::Address(lance_graph_r2il::Addr::Plane(0))];

    let mut d = FoldDialect::new(&planes, &Foreign::NONE);
    let mut stack = plane();
    sum(&mut d, &mut stack, REVENUE).expect("sum over a plane");
    let want: i64 = kept.iter().map(|&r| i64::from(f.revenue[r])).sum();
    assert_eq!(stack, vec![Val::Scalar(want)]);

    let mut want = [0i64; GROUPS];
    for &r in &kept {
        want[f.key[r] as usize] += i64::from(f.revenue[r]);
    }
    let mut sink = [0i64; GROUPS];
    {
        let mut d = FoldDialect::with_groups(&planes, &Foreign::NONE, &mut sink);
        let mut stack = plane();
        group_sum(&mut d, &mut stack, KEY, REVENUE).expect("group_sum over a plane");
    }
    assert_eq!(sink, want);

    let mut out = vec![0u64; words_for(N)];
    {
        let mut d = FoldDialect::with_keep(&planes, &Foreign::NONE, &mut out);
        let mut stack = plane();
        d.call(KEEP, [0; 3], &mut stack).expect("keep a plane");
    }
    assert_eq!(rows(&out), kept);
    // Non-vacuity: the plane drops rows, so reading every row would differ.
    assert!(kept.len() < N);
}

/// FAILS IF: `min`/`max` stop matching `MIN`/`MAX(revenue | x < t)`, or an
/// empty selection stops answering the fold's identity (`i64::MAX` for
/// `MIN`, `i64::MIN` for `MAX`): the grouped terminals' seed, which no `I32`
/// value can equal, so a caller reads it as SQL NULL.
#[test]
fn min_max_match_the_reference() {
    let f = fixture();
    let lanes = lanes(&f);
    let planes = Planes {
        n_rows: N,
        masks: &[],
        lanes: &lanes,
    };
    // `(t, value lane)`: the thresholds, plus one where the mask decides
    // the answer (rows with `x < 1` all have cost 0; the table's max is 4).
    let cases = THRESHOLDS
        .iter()
        .map(|&t| (t, REVENUE, &f.revenue))
        .chain([(1, COST, &f.cost)]);
    for (t, lane, vals) in cases {
        let sel = || (0..N).filter(|&r| f.x[r] < t).map(|r| i64::from(vals[r]));
        let lo = sel().min().unwrap_or(i64::MAX);
        let hi = sel().max().unwrap_or(i64::MIN);
        for (fold, want) in [(min as Fold, lo), (max as Fold, hi)] {
            let mut d = FoldDialect::new(&planes, &Foreign::NONE);
            let mut stack = vec![];
            select_lt(&mut d, &mut stack, X, t).expect("select");
            fold(&mut d, &mut stack, lane).expect("fold");
            assert_eq!(stack, vec![Val::Scalar(want)], "t = {t}, lane {lane}");
        }
    }
    let masked_max = (0..N).filter(|&r| f.x[r] < 1).map(|r| f.cost[r]).max();
    assert_eq!(masked_max, Some(0));
    assert_eq!(f.cost.iter().copied().max(), Some(4));
}

/// A row-level reference for an ordered compare.
type Holds = fn(i32, i32) -> bool;

type Fold = fn(&mut FoldDialect<'_>, &mut Vec<Val>, u16) -> Result<(), FoldError>;

/// FAILS IF: an ordered compare ANDed onto a selection stops matching
/// `y op t` row by row, for each of `<`, `<=`, `>`, `>=`.
#[test]
fn ordered_compares_match_the_reference() {
    let f = fixture();
    let lanes = lanes(&f);
    let planes = Planes {
        n_rows: N,
        masks: &[],
        lanes: &lanes,
    };
    let ops: [(Order, Holds); 4] = [
        (Order::Lt, |a, b| a < b),
        (Order::Le, |a, b| a <= b),
        (Order::Gt, |a, b| a > b),
        (Order::Ge, |a, b| a >= b),
    ];
    for (op, holds) in ops {
        for t in [-1, 0, 15, 30, 31] {
            let want = (0..N).filter(|&r| f.x[r] < 40 && holds(f.y[r], t)).count() as i64;
            let mut d = FoldDialect::new(&planes, &Foreign::NONE);
            let mut stack = vec![];
            select_lt(&mut d, &mut stack, X, 40).expect("select");
            and_cmp_i32(&mut d, &mut stack, Y, op, t).expect("compare");
            count(&mut d, &mut stack).expect("count");
            assert_eq!(stack, vec![Val::Scalar(want)], "{op:?} {t}");
        }
    }
}

// ── D-RFT-1 polyfill: IntXor, ANY, ALL and the procedures over them ────────

/// A plane over `N` rows holding the rows `keep` selects; tail bits clear.
fn plane(keep: impl Fn(usize) -> bool) -> Vec<u64> {
    let mut m = vec![0u64; words_for(N)];
    for r in (0..N).filter(|&r| keep(r)) {
        m[r / 64] |= 1 << (r % 64);
    }
    m
}

/// The scalar a one-result call left on the stack.
fn top(stack: &[Val]) -> i64 {
    match stack {
        [Val::Scalar(n)] => *n,
        other => panic!("expected one scalar, got {other:?}"),
    }
}

/// FAILS IF: `count_and` stops counting the conjunction of its planes, in
/// either mode. Reference: popcount of the word-wise AND (a fold over words,
/// no rows decoded). Each added plane must remove rows, or the test could
/// not see a dropped `IntAnd`.
#[test]
fn count_and_counts_the_conjunction_in_both_modes() {
    let a = plane(|r| r % 2 == 0);
    let b = plane(|r| r % 3 == 0);
    let c = plane(|r| r % 5 != 0);
    let masks = [a.as_slice(), b.as_slice(), c.as_slice()];
    let planes = Planes {
        n_rows: N,
        masks: &masks,
        lanes: &[],
    };
    let reference = |ps: &[u16]| -> i64 {
        (0..words_for(N))
            .map(|w| {
                ps.iter()
                    .fold(u64::MAX, |acc, &p| acc & masks[usize::from(p)][w])
                    .count_ones() as i64
            })
            .sum()
    };
    let cases: [&[u16]; 3] = [&[0], &[0, 1], &[0, 1, 2]];
    let counts: Vec<i64> = cases.iter().map(|ps| reference(ps)).collect();
    assert!(
        counts.windows(2).all(|w| w[1] < w[0]),
        "every plane must remove rows: {counts:?}"
    );
    for mode in [BoolMode::InPlace, BoolMode::Fused] {
        for (ps, want) in cases.iter().zip(&counts) {
            let mut d = FoldDialect::new(&planes, &Foreign::NONE).with_mode(mode);
            let mut stack = vec![];
            count_and(&mut d, &mut stack, ps).expect("count_and");
            assert_eq!(top(&stack), *want, "{mode:?} {ps:?}");
            assert_eq!(d.programs_run(), 1, "one fold");
        }
    }
}

/// FAILS IF: `differ` stops answering mask inequality, in either mode. The
/// planes that differ do so in ONE row, the last, which sits in the tail
/// word; an `IntXor` lowered as `And` or `Or`, or an `ANY` that skips the
/// tail word, answers wrongly here.
#[test]
fn differ_is_mask_inequality_in_both_modes() {
    let a = plane(|r| r % 7 == 0);
    let same = plane(|r| r % 7 == 0);
    let last = N - 1;
    assert_ne!(last % 7, 0, "the flipped row must be clear in `a`");
    let b = plane(|r| r % 7 == 0 || r == last);
    let empty = plane(|_| false);
    let masks = [
        a.as_slice(),
        same.as_slice(),
        b.as_slice(),
        empty.as_slice(),
    ];
    let planes = Planes {
        n_rows: N,
        masks: &masks,
        lanes: &[],
    };
    for mode in [BoolMode::InPlace, BoolMode::Fused] {
        for (x, y, want) in [(0, 1, 0), (0, 2, 1), (2, 0, 1), (3, 3, 0), (0, 3, 1)] {
            let mut d = FoldDialect::new(&planes, &Foreign::NONE).with_mode(mode);
            let mut stack = vec![];
            differ(&mut d, &mut stack, x, y).expect("differ");
            assert_eq!(top(&stack), want, "{mode:?} planes {x} vs {y}");
        }
    }
}

/// FAILS IF: `exists` or `all` stop answering over exactly `n_rows` rows.
/// `full` holds all `N` rows and no tail bit; `all_but_one` misses only the
/// last row, so an `ALL` that ignores the tail word answers 1 for it.
#[test]
fn exists_and_all_answer_over_n_rows() {
    let empty = plane(|_| false);
    let one = plane(|r| r == N - 1);
    let full = plane(|_| true);
    let all_but_one = plane(|r| r != N - 1);
    let masks = [
        empty.as_slice(),
        one.as_slice(),
        full.as_slice(),
        all_but_one.as_slice(),
    ];
    let planes = Planes {
        n_rows: N,
        masks: &masks,
        lanes: &[],
    };
    let run = |p: u16, f: fn(&mut FoldDialect<'_>, &mut Vec<Val>) -> Result<(), FoldError>| {
        let mut d = FoldDialect::new(&planes, &Foreign::NONE);
        let mut stack = vec![Val::Address(lance_graph_r2il::Addr::Plane(p))];
        f(&mut d, &mut stack).expect("terminal");
        top(&stack)
    };
    assert_eq!([0, 1, 2, 3].map(|p| run(p, exists)), [0, 1, 1, 1], "exists");
    assert_eq!([0, 1, 2, 3].map(|p| run(p, all)), [0, 0, 1, 0], "all");
}

/// FAILS IF: scalar `IntXor` stops being bitwise exclusive or.
#[test]
fn int_xor_on_scalars_is_exclusive_or() {
    let planes = Planes {
        n_rows: N,
        masks: &[],
        lanes: &[],
    };
    let mut d = FoldDialect::new(&planes, &Foreign::NONE);
    let mut stack = vec![Val::Scalar(0b1100), Val::Scalar(0b1010)];
    d.call(INT_XOR, [0; 3], &mut stack).expect("xor");
    assert_eq!(stack, vec![Val::Scalar(0b0110)]);
}

/// FAILS IF: `union` stops writing the union of its planes through `keep`,
/// in either mode. Reference: the word-wise OR; equality by `differ`, not by
/// decoding rows. Each added plane must add rows.
#[test]
fn union_keeps_the_union_in_both_modes() {
    let a = plane(|r| r % 5 == 0);
    let b = plane(|r| r % 7 == 0);
    let c = plane(|r| r % 11 == 0);
    let want: Vec<u64> = (0..words_for(N)).map(|w| a[w] | b[w] | c[w]).collect();
    let partial: Vec<u64> = (0..words_for(N)).map(|w| a[w] | b[w]).collect();
    for mode in [BoolMode::InPlace, BoolMode::Fused] {
        let mut sink = vec![0u64; words_for(N)];
        {
            let masks = [a.as_slice(), b.as_slice(), c.as_slice()];
            let planes = Planes {
                n_rows: N,
                masks: &masks,
                lanes: &[],
            };
            let mut d = FoldDialect::with_keep(&planes, &Foreign::NONE, &mut sink).with_mode(mode);
            let mut stack = vec![];
            union(&mut d, &mut stack, &[0, 1, 2]).expect("union");
            keep(&mut d, &mut stack).expect("keep");
            assert!(stack.is_empty());
        }
        let masks = [sink.as_slice(), want.as_slice(), partial.as_slice()];
        let planes = Planes {
            n_rows: N,
            masks: &masks,
            lanes: &[],
        };
        for (other, expect) in [(1u16, 0i64), (2, 1)] {
            let mut d = FoldDialect::new(&planes, &Foreign::NONE);
            let mut stack = vec![];
            differ(&mut d, &mut stack, 0, other).expect("differ");
            assert_eq!(top(&stack), expect, "{mode:?} vs plane {other}");
        }
    }
}

/// FAILS IF: a union or intersection over more planes than the dialect has
/// scratch slots (`MAX_SLOTS`) refuses instead of folding in
/// `BoolMode::InPlace`. The chain consumes each intermediate once, so it
/// reuses slots; a chain that allocated a fresh slot per `IntOr` needs
/// `k - 1` and breaks at 9.
///
/// `BoolMode::Fused` is pinned as a refusal, not a panic: mask-risc's `fuse`
/// gives every ternlog group its own slot, so a 24-leaf chain overflows the
/// arena. That is the reference algebra's limit, recorded as open.
#[test]
fn wide_union_and_intersect_fit_the_scratch_arena() {
    const K: usize = 3 * lance_graph_r2il::MAX_SLOTS;
    let owned: Vec<Vec<u64>> = (0..K).map(|k| plane(|r| r % (k + 2) != 0)).collect();
    let masks: Vec<&[u64]> = owned.iter().map(Vec::as_slice).collect();
    let ix: Vec<u16> = (0..K as u16).collect();
    let words = words_for(N);
    let want_or: i64 = (0..words)
        .map(|w| masks.iter().fold(0u64, |a, m| a | m[w]).count_ones() as i64)
        .sum();
    let want_and: i64 = (0..words)
        .map(|w| masks.iter().fold(u64::MAX, |a, m| a & m[w]).count_ones() as i64)
        .sum();
    assert!(
        want_and > 0 && want_or > want_and,
        "anti-vacuity: {want_and} {want_or}"
    );
    let planes = Planes {
        n_rows: N,
        masks: &masks,
        lanes: &[],
    };
    let mut d = FoldDialect::new(&planes, &Foreign::NONE);
    let mut stack = vec![];
    union(&mut d, &mut stack, &ix).expect("wide union");
    count(&mut d, &mut stack).expect("count over the union");
    assert_eq!(top(&stack), want_or, "union");
    let mut d = FoldDialect::new(&planes, &Foreign::NONE);
    let mut stack = vec![];
    count_and(&mut d, &mut stack, &ix).expect("wide intersect");
    assert_eq!(top(&stack), want_and, "intersect");

    let mut d = FoldDialect::new(&planes, &Foreign::NONE).with_mode(BoolMode::Fused);
    let mut stack = vec![];
    union(&mut d, &mut stack, &ix).expect("fused union builds");
    assert!(
        matches!(count(&mut d, &mut stack), Err(FoldError::Exec(_))),
        "fused wide chain: refused, not folded and not a panic"
    );
}

// ── PostgreSQL primitives ────────────────────────────────────────────────
//
// Fixture: `x` (I32, `i % 50`) is NULL on every 9th row (presence plane
// `VALID`); `key` (U32, `i % GROUPS`); a flag plane `FLAG` = rows where
// `i % 3 == 0`; a key-ordered lane `SORTED` (`i / 10`). References are
// computed from the fixture's own rule per row, never from the dialect.

const VALID: u16 = 0;
const FLAG: u16 = 1;
const SORTED: u16 = 5;
const ALL_ROWS: u16 = 2;

fn null_row(r: usize) -> bool {
    r.is_multiple_of(9)
}

fn run_scalar(
    planes: &Planes<'_>,
    body: impl FnOnce(&mut FoldDialect<'_>, &mut Vec<Val>) -> Result<(), FoldError>,
) -> i64 {
    let mut d = FoldDialect::new(planes, &Foreign::NONE);
    let mut stack = vec![];
    body(&mut d, &mut stack).expect("body");
    top(&stack)
}

/// FAILS IF: any PostgreSQL primitive stops matching its row-rule reference:
/// `IS [NOT] NULL`, `BETWEEN` (with a NULL `x` dropped), `IN`, `IN ()`,
/// `COUNT(x)` vs `COUNT(*)`, `AVG` parts, `bool_or`/`bool_and`, `NOT EXISTS`,
/// `EXCEPT`, `COUNT(DISTINCT)` on a key-ordered lane, `UNION ALL` counts, and
/// `IntAdd`/`IntMult`/`IntSDiv` on scalars.
#[test]
fn postgresql_primitives_match_row_rule_references() {
    let f = fixture();
    let sorted: Vec<u32> = (0..N).map(|i| (i / 10) as u32).collect();
    let lanes = [
        LaneRef::I32(&f.x),
        LaneRef::I32(&f.y),
        LaneRef::I32(&f.revenue),
        LaneRef::I32(&f.cost),
        LaneRef::U32(&f.key),
        LaneRef::U32(&sorted),
    ];
    let valid = plane(|r| !null_row(r));
    let flag = plane(|r| r % 3 == 0);
    let all_rows = plane(|_| true);
    let masks = [valid.as_slice(), flag.as_slice(), all_rows.as_slice()];
    let planes = Planes {
        n_rows: N,
        masks: &masks,
        lanes: &lanes,
    };
    let count_rows = |keep: &dyn Fn(usize) -> bool| (0..N).filter(|&r| keep(r)).count() as i64;

    // IS NULL / IS NOT NULL.
    let nulls = count_rows(&|r| null_row(r));
    assert!(nulls > 0 && nulls < N as i64);
    assert_eq!(
        run_scalar(&planes, |d, s| {
            is_null(d, s, VALID)?;
            count(d, s)
        }),
        nulls
    );
    assert_eq!(
        run_scalar(&planes, |d, s| {
            is_not_null(d, s, VALID)?;
            count(d, s)
        }),
        N as i64 - nulls
    );

    // BETWEEN over the presence plane: a NULL x is dropped.
    let between = count_rows(&|r| !null_row(r) && (10..=20).contains(&f.x[r]));
    let between_ignoring_null = count_rows(&|r| (10..=20).contains(&f.x[r]));
    assert_ne!(
        between, between_ignoring_null,
        "the fixture must have NULL rows in range"
    );
    assert_eq!(
        run_scalar(&planes, |d, s| {
            is_not_null(d, s, VALID)?;
            and_between_i32(d, s, X, 10, 20)?;
            count(d, s)
        }),
        between
    );
    // An empty range keeps nothing.
    assert_eq!(
        run_scalar(&planes, |d, s| {
            is_not_null(d, s, VALID)?;
            and_between_i32(d, s, X, 20, 10)?;
            count(d, s)
        }),
        0
    );

    // IN (…) and IN ().
    assert_eq!(
        run_scalar(&planes, |d, s| {
            in_u32(d, s, KEY, &[1, 3])?;
            count(d, s)
        }),
        count_rows(&|r| f.key[r] == 1 || f.key[r] == 3)
    );
    assert_eq!(
        run_scalar(&planes, |d, s| {
            in_u32(d, s, KEY, &[])?;
            count(d, s)
        }),
        0
    );

    // COUNT(x) vs COUNT(*) under x < 17.
    let star = count_rows(&|r| f.x[r] < 17);
    let col = count_rows(&|r| f.x[r] < 17 && !null_row(r));
    assert!(col < star);
    assert_eq!(
        run_scalar(&planes, |d, s| {
            select_lt(d, s, X, 17)?;
            count(d, s)
        }),
        star
    );
    assert_eq!(
        run_scalar(&planes, |d, s| {
            select_lt(d, s, X, 17)?;
            count_valid(d, s, VALID)
        }),
        col
    );

    // AVG parts: SUM(x) and COUNT(x) over x < 17, NULL rows excluded.
    let mut d = FoldDialect::new(&planes, &Foreign::NONE);
    let mut stack = vec![];
    let sel = |d: &mut FoldDialect<'_>, s: &mut Vec<Val>| select_lt(d, s, X, 17);
    avg_parts(&mut d, &mut stack, &sel, X, VALID).expect("avg");
    let want_sum: i64 = (0..N)
        .filter(|&r| f.x[r] < 17 && !null_row(r))
        .map(|r| i64::from(f.x[r]))
        .sum();
    assert_eq!(stack, vec![Val::Scalar(want_sum), Val::Scalar(col)]);

    // bool_or / bool_and over a selection: (value, present); present 0 is NULL.
    let two = |body: &Select<'_>| {
        let mut d = FoldDialect::new(&planes, &Foreign::NONE);
        let mut stack = vec![];
        body(&mut d, &mut stack).expect("body");
        match stack.as_slice() {
            [Val::Scalar(v), Val::Scalar(p)] => (*v, *p),
            other => panic!("expected value and present, got {other:?}"),
        }
    };
    let bor = |t: i32| {
        let sel = move |d: &mut FoldDialect<'_>, s: &mut Vec<Val>| select_lt(d, s, X, t);
        two(&|d, s| bool_or(d, s, &sel, VALID, FLAG))
    };
    let band = |t: i32| {
        let sel = move |d: &mut FoldDialect<'_>, s: &mut Vec<Val>| select_lt(d, s, X, t);
        two(&|d, s| bool_and(d, s, &sel, VALID, FLAG))
    };
    // x < 1 selects rows 0, 50, 100, …; row 0 is NULL (0 % 9), 50 is present
    // and not flagged, 150 is present and flagged.
    assert_eq!((bor(1), band(1)), ((1, 1), (0, 1)));
    assert_eq!(
        (bor(0), band(0)),
        ((0, 0), (0, 0)),
        "empty selection is NULL"
    );

    // NOT EXISTS.
    assert_eq!(
        run_scalar(&planes, |d, s| {
            select_lt(d, s, X, 0)?;
            not_exists(d, s)
        }),
        1
    );
    assert_eq!(
        run_scalar(&planes, |d, s| {
            select_lt(d, s, X, 1)?;
            not_exists(d, s)
        }),
        0
    );

    // EXCEPT: valid ∧ ¬flag.
    assert_eq!(
        run_scalar(&planes, |d, s| {
            except(d, s, VALID, FLAG)?;
            count(d, s)
        }),
        count_rows(&|r| !null_row(r) && r % 3 != 0)
    );

    // COUNT(DISTINCT sorted) under x < 5: distinct i / 10 over rows with i % 50 < 5.
    let mut seen = [false; N / 10 + 1];
    for r in (0..N).filter(|&r| f.x[r] < 5) {
        seen[r / 10] = true;
    }
    let distinct = seen.iter().filter(|&&b| b).count() as i64;
    assert!(distinct < count_rows(&|r| f.x[r] < 5), "keys repeat");
    assert_eq!(
        run_scalar(&planes, |d, s| {
            select_lt(d, s, X, 5)?;
            count_distinct_sorted(d, s, SORTED, ALL_ROWS)
        }),
        distinct
    );

    // UNION ALL of two counts.
    let a = |d: &mut FoldDialect<'_>, s: &mut Vec<Val>| select_lt(d, s, X, 10);
    let b = |d: &mut FoldDialect<'_>, s: &mut Vec<Val>| select_lt(d, s, Y, 5);
    assert_eq!(
        run_scalar(&planes, |d, s| count_union_all(d, s, &a, &b)),
        count_rows(&|r| f.x[r] < 10) + count_rows(&|r| f.y[r] < 5)
    );

    // Scalar arithmetic.
    for (op, x, y, want) in [
        (INT_ADD, 7, 5, 12),
        (INT_MULT, 7, -5, -35),
        (INT_S_DIV, -7, 2, -3),
    ] {
        let mut d = FoldDialect::new(&planes, &Foreign::NONE);
        let mut stack = vec![Val::Scalar(x), Val::Scalar(y)];
        d.call(op, [0; 3], &mut stack).expect("arith");
        assert_eq!(stack, vec![Val::Scalar(want)], "{op:?}");
    }
}

/// FAILS IF: `COUNT(DISTINCT)` on an UNORDERED lane is answered instead of
/// refused (`LaneNotOrdered`).
#[test]
fn count_distinct_refuses_an_unordered_lane() {
    let f = fixture();
    let lanes = lanes(&f);
    let all_rows = plane(|_| true);
    let planes = Planes {
        n_rows: N,
        masks: &[all_rows.as_slice()],
        lanes: &lanes,
    };
    let mut d = FoldDialect::new(&planes, &Foreign::NONE);
    let mut stack = vec![];
    select_lt(&mut d, &mut stack, X, 17).expect("select");
    let err =
        count_distinct_sorted(&mut d, &mut stack, KEY, 0).expect_err("key = i % 4 is unordered");
    assert!(matches!(err, FoldError::Exec(_)), "{err:?}");
}

/// The nullable boolean of one aggregate case: `None` is NULL.
fn bool_case(vals: &[Option<bool>], scramble: bool) -> (Vec<u64>, Vec<u64>, Vec<u64>) {
    let n = vals.len();
    let sel = plane(|r| r < n);
    let valid = plane(|r| r < n && vals[r].is_some());
    // Under a NULL the flag bit is a payload: set it on every other row, or
    // flip that choice when scrambling.
    let flag = plane(|r| match vals.get(r) {
        Some(Some(b)) => *b,
        Some(None) => (r % 2 == 0) != scramble,
        None => false,
    });
    (sel, valid, flag)
}

/// `(bool_or, bool_and)` of one case; `None` is SQL NULL.
fn bool_aggs(vals: &[Option<bool>], scramble: bool) -> (Option<bool>, Option<bool>) {
    let (sel, valid, flag) = bool_case(vals, scramble);
    let masks = [sel.as_slice(), valid.as_slice(), flag.as_slice()];
    let planes = Planes {
        n_rows: N,
        masks: &masks,
        lanes: &[],
    };
    let select = |d: &mut FoldDialect<'_>, s: &mut Vec<Val>| is_not_null(d, s, 0);
    let read = |f: &Select<'_>| {
        let mut d = FoldDialect::new(&planes, &Foreign::NONE);
        let mut stack = vec![];
        f(&mut d, &mut stack).expect("agg");
        match stack.as_slice() {
            [Val::Scalar(v), Val::Scalar(1)] => Some(*v == 1),
            [Val::Scalar(0), Val::Scalar(0)] => None,
            other => panic!("unexpected result {other:?}"),
        }
    };
    (
        read(&|d, s| bool_or(d, s, &select, 1, 2)),
        read(&|d, s| bool_and(d, s, &select, 1, 2)),
    )
}

/// FAILS IF: `bool_or` / `bool_and` read a NULL row as FALSE (or its payload),
/// or answer FALSE where PostgreSQL answers NULL. Disable: drop the `valid`
/// gate from either procedure (the contributing rows become `select` alone).
#[test]
fn bool_aggregates_keep_null_apart_from_false() {
    const T: Option<bool> = Some(true);
    const F: Option<bool> = Some(false);
    const U: Option<bool> = None;
    type Case<'a> = (&'a [Option<bool>], Option<bool>, Option<bool>);
    let cases: [Case<'_>; 6] = [
        (&[T, U], T, T),
        (&[F, U], F, F),
        (&[T, F, U], T, F),
        (&[U, U], None, None),
        (&[], None, None),
        (&[F, F], F, F),
    ];
    for (vals, want_or, want_and) in cases {
        for scramble in [false, true] {
            assert_eq!(
                bool_aggs(vals, scramble),
                (want_or, want_and),
                "{vals:?} (scramble {scramble})"
            );
        }
    }
}

/// FAILS IF: `COUNT(DISTINCT)` counts a NULL key, i.e. a NULL row whose
/// payload equals a real key, or a run made only of NULL rows. Disable: drop
/// the `valid` gate from `count_distinct_sorted`.
#[test]
fn count_distinct_excludes_null_keys_whatever_their_payload() {
    // Key-ordered lane: runs of 10 rows, key = row / 10 (100 runs).
    let sorted: Vec<u32> = (0..N).map(|i| (i / 10) as u32).collect();
    let lanes = [LaneRef::U32(&sorted)];
    let count = |valid: &[u64], sel: &[u64]| {
        let masks = [sel, valid];
        let planes = Planes {
            n_rows: N,
            masks: &masks,
            lanes: &lanes,
        };
        run_scalar(&planes, |d, s| {
            is_not_null(d, s, 0)?;
            count_distinct_sorted(d, s, 0, 1)
        })
    };
    // Selected: the first, a middle and the last run, entirely.
    let in_runs = |r: usize, runs: &[usize]| runs.contains(&(r / 10));
    let sel = plane(|r| in_runs(r, &[0, 50, 99]));
    // Every key present: 3 distinct.
    assert_eq!(count(&plane(|_| true), &sel), 3);
    // The middle run all NULL (payload 50 still equals a real key): 2.
    assert_eq!(count(&plane(|r| r / 10 != 50), &sel), 2);
    // First and last runs NULL except one row each: still 3.
    let one_left = |r: usize| (r / 10 != 0 && r / 10 != 99) || r == 3 || r == 995;
    assert_eq!(count(&plane(one_left), &sel), 3);
    // All selected rows NULL: 0.
    assert_eq!(count(&plane(|r| !in_runs(r, &[0, 50, 99])), &sel), 0);
}
