//! D-SQLP-2: procedures, `IntOr`, and the two Boolean modes.
//!
//! Every answer is compared against a scalar reference computed row by row
//! from the same lanes. Fixtures have four groups and are run with an empty
//! selection, a partial one and a full one.

use lance_graph_mask_risc::{words_for, Foreign, LaneRef, MaskOp, Planes};
use lance_graph_r2il::procs::{filter_lt, group_sum, margin, select_lt, sum};
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
