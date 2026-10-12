//! D-SQLP-2: procedures, `IntOr`, and the two Boolean modes.
//!
//! Every answer is compared against a scalar reference computed row by row
//! from the same lanes. Fixtures have four groups and are run with an empty
//! selection, a partial one and a full one.

use lance_graph_mask_risc::{words_for, Foreign, LaneRef, MaskOp, Planes};
use lance_graph_r2il::procs::{
    and_cmp_i32, count, filter_lt, group_sum, margin, max, min, select_lt, select_u32, sum, Order,
};
use lance_graph_r2il::{
    BoolMode, FoldDialect, FoldError, Val, INT_AND, INT_NOT, INT_OR, POP_COUNT,
};
use ogar_loco::Dialect;
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
