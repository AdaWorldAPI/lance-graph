//! `MaskedArgMinI32` / `MaskedArgMaxI32`: the row holding the extreme value of
//! an `i32` lane over the selected rows, lowest row among ties.
//!
//! Every answer is checked three ways where possible: the executor, the
//! independent row-at-a-time oracle, and a scalar loop written in the test.
//! The tests use heavy ties on purpose: with few distinct values the tie rule
//! decides the answer, so a wrong tie rule cannot hide.

use lance_graph_mask_risc::{
    execute, execute_extent, execute_into, reference_execute, words_for, ExecError, Foreign,
    LaneKind, LaneRef, MaskOp, Operand, Out, Planes, Pred, Program, Scratch, Terminal, Value,
};

const SIZES: [usize; 12] = [
    1, 63, 64, 65, 255, 256, 257, 16_383, 16_384, 16_385, 65_536, 100_000,
];

fn lcg(seed: &mut u64) -> u64 {
    *seed = seed
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    *seed >> 11
}

fn bit(p: &[u64], r: usize) -> bool {
    p[r / 64] >> (r % 64) & 1 == 1
}

fn plane_from(n: usize, mut set: impl FnMut(usize) -> bool) -> Vec<u64> {
    let mut w = vec![0u64; words_for(n)];
    for r in (0..n).filter(|&r| set(r)) {
        w[r / 64] |= 1 << (r % 64);
    }
    w
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Which {
    ArgMin,
    ArgMax,
    Min,
    Max,
}

fn terminal(which: Which, mask: Operand, lane: u16) -> Terminal {
    match which {
        Which::ArgMin => Terminal::MaskedArgMinI32 { mask, lane },
        Which::ArgMax => Terminal::MaskedArgMaxI32 { mask, lane },
        Which::Min => Terminal::MaskedMinI32 { mask, lane },
        Which::Max => Terminal::MaskedMaxI32 { mask, lane },
    }
}

/// Region mask (plane 0) plus two `i32` lanes: lane 0 heavily tied, lane 1
/// random. About half the region rows are set.
struct Fix {
    n: usize,
    region: Vec<u64>,
    l0: Vec<i32>,
    l1: Vec<i32>,
}

impl Fix {
    fn new(n: usize) -> Self {
        let mut s = 0xA46E_u64 ^ (n as u64);
        let region = plane_from(n, |_| lcg(&mut s) & 1 == 1);
        let l0 = (0..n).map(|_| (lcg(&mut s) % 7) as i32 - 3).collect();
        let l1 = (0..n).map(|_| (lcg(&mut s) % 101) as i32 - 50).collect();
        Self { n, region, l0, l1 }
    }

    fn selected(&self, t: i32) -> impl Fn(usize) -> bool + '_ {
        move |r| bit(&self.region, r) && self.l1[r] > t
    }
}

/// `scratch0 = (lane1 > t) under plane0`, then `which` over lane 0 of scratch 0.
fn where_program(t: i32, which: Which) -> Program {
    Program::new(
        vec![MaskOp::Pred {
            pred: Pred::GtI32 { lane: 1, t },
            under: Some(Operand::Plane(0)),
            dst: 0,
        }],
        terminal(which, Operand::Scratch(0), 0),
    )
}

/// Executor (one tile as wide as the population), executor (3-word tiles) and
/// oracle. The two executor widths must agree before they are compared.
fn eval(
    n: usize,
    masks: &[&[u64]],
    lanes: &[LaneRef<'_>],
    p: &Program,
) -> (Result<Value, ExecError>, Result<Value, ExecError>) {
    let planes = Planes {
        n_rows: n,
        masks,
        lanes,
    };
    let slots = p.scratch_slots as usize;
    let mut wide = Scratch::new(words_for(n), slots);
    let got = execute(p, &planes, &mut wide, None);
    let mut narrow = Scratch::new(words_for(n).min(3), slots);
    let got_narrow = execute(p, &planes, &mut narrow, None);
    assert_eq!(got, got_narrow, "tile width changed the answer at n={n}");
    (got, reference_execute(p, &planes, None))
}

fn eval_fix(f: &Fix, p: &Program) -> (Result<Value, ExecError>, Result<Value, ExecError>) {
    let masks: [&[u64]; 1] = [&f.region];
    let lanes = [LaneRef::I32(&f.l0), LaneRef::I32(&f.l1)];
    eval(f.n, &masks, &lanes, p)
}

/// Scalar answer: strict comparison keeps the lowest row among ties.
fn scalar_arg(values: &[i32], selected: impl Fn(usize) -> bool, max: bool) -> Option<(usize, i32)> {
    let mut best: Option<(usize, i32)> = None;
    for (r, &v) in values.iter().enumerate().filter(|&(r, _)| selected(r)) {
        best = match best {
            None => Some((r, v)),
            Some((_, b)) if (max && v > b) || (!max && v < b) => Some((r, v)),
            keep => keep,
        };
    }
    best
}

/// FAILS IF: the executor, the oracle and a scalar loop disagree on the row or
/// the value, at any size straddling a word, a 256-row block or a 16 384-row
/// tile edge, or the tie rule picks any row but the lowest.
#[test]
fn executor_equals_reference_on_where_then_arg() {
    for n in SIZES {
        let f = Fix::new(n);
        for (which, max) in [(Which::ArgMin, false), (Which::ArgMax, true)] {
            let want = Value::OptRowI32(scalar_arg(&f.l0, f.selected(0), max));
            let (got, oracle) = eval_fix(&f, &where_program(0, which));
            assert_eq!(got, Ok(want), "executor {which:?} n={n}");
            assert_eq!(oracle, Ok(want), "oracle {which:?} n={n}");
        }
    }
}

/// FAILS IF: an empty selection returns anything but `OptRowI32(None)`, whether
/// the mask has no bit set or the predicate rejects every row.
#[test]
fn empty_selection_is_none() {
    for n in [1usize, 64, 65, 16_385] {
        let f = Fix::new(n);
        let zero = plane_from(n, |_| false);
        let masks: [&[u64]; 1] = [&zero];
        let lanes = [LaneRef::I32(&f.l0), LaneRef::I32(&f.l1)];
        for which in [Which::ArgMin, Which::ArgMax] {
            let (got, oracle) = eval(n, &masks, &lanes, &where_program(0, which));
            assert_eq!(
                got,
                Ok(Value::OptRowI32(None)),
                "empty mask {which:?} n={n}"
            );
            assert_eq!(oracle, got, "oracle, empty mask {which:?} n={n}");

            // lane 1 never exceeds 50, so `> 50` rejects every row
            let (got, oracle) = eval_fix(&f, &where_program(50, which));
            assert_eq!(
                got,
                Ok(Value::OptRowI32(None)),
                "empty pred {which:?} n={n}"
            );
            assert_eq!(oracle, got, "oracle, empty pred {which:?} n={n}");
        }
    }
}

fn arg_over_plane(n: usize, plane: &[u64], lane: &[i32], which: Which) -> Value {
    let masks: [&[u64]; 1] = [plane];
    let lanes = [LaneRef::I32(lane)];
    let p = Program::new(vec![], terminal(which, Operand::Plane(0), 0));
    let (got, oracle) = eval(n, &masks, &lanes, &p);
    assert_eq!(got, oracle, "executor vs oracle, {which:?}");
    got.expect("program runs")
}

/// FAILS IF: ties across a 16 384-row tile boundary resolve to anything but
/// the lowest row, or a row cleared from a RESIDENT mask still wins.
#[test]
fn lowest_row_wins_across_a_tile_boundary() {
    let n = 40_000;
    let mut lane = vec![5i32; n];
    for r in [16_383, 16_386, 30_000] {
        lane[r] = -9;
    }
    for r in [16_384, 16_385, 20] {
        lane[r] = 99;
    }
    let all = plane_from(n, |_| true);
    assert_eq!(
        arg_over_plane(n, &all, &lane, Which::ArgMin),
        Value::OptRowI32(Some((16_383, -9)))
    );
    assert_eq!(
        arg_over_plane(n, &all, &lane, Which::ArgMax),
        Value::OptRowI32(Some((20, 99)))
    );
    let without = plane_from(n, |r| r != 16_383);
    assert_eq!(
        arg_over_plane(n, &without, &lane, Which::ArgMin),
        Value::OptRowI32(Some((16_386, -9)))
    );
}

/// FAILS IF: a row the mask excludes can win. `i32::MIN` and `i32::MAX` are
/// planted at rows 100 and 200; excluded they must not appear, included they
/// must win at exactly those rows.
#[test]
fn masked_out_extreme_cannot_win() {
    let n = 1_000;
    let mut s = 77u64;
    let mut lane: Vec<i32> = (0..n).map(|_| (lcg(&mut s) % 1000) as i32 - 500).collect();
    lane[100] = i32::MIN;
    lane[200] = i32::MAX;
    let without = plane_from(n, |r| r != 100 && r != 200);
    let with = plane_from(n, |_| true);
    for (which, max) in [(Which::ArgMin, false), (Which::ArgMax, true)] {
        let out = arg_over_plane(n, &without, &lane, which);
        let want = scalar_arg(&lane, |r| r != 100 && r != 200, max);
        assert_eq!(out, Value::OptRowI32(want), "excluded, {which:?}");
        let (row, v) = want.expect("rows selected");
        assert!(v != i32::MIN && v != i32::MAX && row != 100 && row != 200);
    }
    assert_eq!(
        arg_over_plane(n, &with, &lane, Which::ArgMin),
        Value::OptRowI32(Some((100, i32::MIN)))
    );
    assert_eq!(
        arg_over_plane(n, &with, &lane, Which::ArgMax),
        Value::OptRowI32(Some((200, i32::MAX)))
    );
}

/// FAILS IF: the value inside the arg terminal differs from the value-only
/// terminal over the same selection.
#[test]
fn arg_agrees_with_the_value_only_terminals() {
    for n in SIZES {
        let f = Fix::new(n);
        for (arg, plain) in [(Which::ArgMin, Which::Min), (Which::ArgMax, Which::Max)] {
            let a = eval_fix(&f, &where_program(0, arg)).0.expect("arg runs");
            let v = eval_fix(&f, &where_program(0, plain))
                .0
                .expect("plain runs");
            let from_arg = match a {
                Value::OptRowI32(x) => x.map(|(_, v)| v),
                other => panic!("expected OptRowI32, got {other:?}"),
            };
            let from_plain = match v {
                Value::OptI32(x) => x,
                other => panic!("expected OptI32, got {other:?}"),
            };
            assert_eq!(from_arg, from_plain, "{arg:?} vs {plain:?} n={n}");
        }
    }
}

fn first_set_bit(words: &[u64]) -> Option<usize> {
    words
        .iter()
        .enumerate()
        .find(|(_, w)| **w != 0)
        .map(|(i, w)| i * 64 + w.trailing_zeros() as usize)
}

/// FAILS IF: the new terminal's row differs from what two EXISTING programs
/// find: the value from `MaskedMin/MaxI32`, then the first row of `Keep` over
/// `lane == value` under the same mask. The terminal adds only the coordinate.
#[test]
fn the_coordinate_equals_what_two_existing_programs_find() {
    for n in [65usize, 16_385, 65_536] {
        let f = Fix::new(n);
        let masks: [&[u64]; 1] = [&f.region];
        let lanes = [LaneRef::I32(&f.l0), LaneRef::I32(&f.l1)];
        let planes = Planes {
            n_rows: n,
            masks: &masks,
            lanes: &lanes,
        };
        for (arg, plain) in [(Which::ArgMin, Which::Min), (Which::ArgMax, Which::Max)] {
            let over_plane = |w: Which| Program::new(vec![], terminal(w, Operand::Plane(0), 0));
            let v = match eval(n, &masks, &lanes, &over_plane(plain)).0 {
                Ok(Value::OptI32(Some(v))) => v,
                other => panic!("region should select rows, got {other:?}"),
            };
            let b = Program::new(
                vec![MaskOp::Pred {
                    pred: Pred::EqI32 { lane: 0, v },
                    under: Some(Operand::Plane(0)),
                    dst: 0,
                }],
                Terminal::Keep {
                    mask: Operand::Scratch(0),
                },
            );
            let mut kept = vec![0u64; words_for(n)];
            let mut s = Scratch::for_program(&b, n).expect("scratch");
            execute_into(&b, &planes, &Foreign::NONE, &mut s, Out::Mask(&mut kept))
                .expect("keep runs");
            let row = first_set_bit(&kept).expect("the extreme value exists in the mask");
            let a = eval(n, &masks, &lanes, &over_plane(arg)).0;
            assert_eq!(a, Ok(Value::OptRowI32(Some((row, v)))), "{arg:?} n={n}");
        }
    }
}

/// Better value wins; on equal value the lower row wins.
fn merge(max: bool, a: Option<(usize, i32)>, b: Option<(usize, i32)>) -> Option<(usize, i32)> {
    match (a, b) {
        (None, x) | (x, None) => x,
        (Some((ra, va)), Some((rb, vb))) => {
            let b_better = if max { vb > va } else { vb < va };
            if b_better || (va == vb && rb < ra) {
                b
            } else {
                a
            }
        }
    }
}

/// FAILS IF: per-extent answers, merged by (better value, then lower row) in
/// any order, differ from the whole-range answer. The data is heavy-tie, and
/// the test first checks that two extents report the same extreme, so the tie
/// half of the merge law is exercised and not vacuous.
#[test]
fn partial_extents_merge_to_the_whole_answer() {
    let n = 50_000;
    let f = Fix::new(n);
    let masks: [&[u64]; 1] = [&f.region];
    let lanes = [LaneRef::I32(&f.l0), LaneRef::I32(&f.l1)];
    let planes = Planes {
        n_rows: n,
        masks: &masks,
        lanes: &lanes,
    };
    let bounds = [0usize, 1, 63, 64, 1000, 16_384, 16_385, 33_000, 50_000];
    for (which, max) in [(Which::ArgMin, false), (Which::ArgMax, true)] {
        let p = where_program(0, which);
        let parts: Vec<Option<(usize, i32)>> = bounds
            .windows(2)
            .map(|w| {
                let mut s = Scratch::for_program(&p, n).expect("scratch");
                match execute_extent(&p, &planes, &Foreign::NONE, &mut s, Out::None, w[0]..w[1]) {
                    Ok(Value::OptRowI32(x)) => x,
                    other => panic!("extent {w:?}: {other:?}"),
                }
            })
            .collect();
        let whole = eval_fix(&f, &p);
        let Ok(Value::OptRowI32(whole_arg)) = whole.0 else {
            panic!("whole run failed: {:?}", whole.0);
        };
        assert_eq!(whole.1, whole.0, "oracle, {which:?}");

        let values: Vec<i32> = parts.iter().flatten().map(|&(_, v)| v).collect();
        let tied = values
            .iter()
            .enumerate()
            .any(|(i, v)| values[i + 1..].contains(v));
        assert!(
            tied,
            "{which:?}: no two extents share an extreme; tie law unexercised"
        );

        let mut ltr = None;
        for &x in &parts {
            ltr = merge(max, ltr, x);
        }
        let mut rtl = None;
        for &x in parts.iter().rev() {
            rtl = merge(max, rtl, x);
        }
        let mut order: Vec<usize> = (0..parts.len()).collect();
        let mut seed = 0x5EED_u64;
        for i in (1..order.len()).rev() {
            order.swap(i, (lcg(&mut seed) as usize) % (i + 1));
        }
        let mut shuffled = None;
        for &i in &order {
            shuffled = merge(max, shuffled, parts[i]);
        }
        assert_eq!(ltr, whole_arg, "left to right, {which:?}");
        assert_eq!(rtl, whole_arg, "right to left, {which:?}");
        assert_eq!(shuffled, whole_arg, "shuffled {order:?}, {which:?}");
    }
}

/// FAILS IF: executor and oracle refuse a non-`i32` lane differently, or one
/// of them accepts it.
#[test]
fn a_wrong_lane_kind_is_refused_the_same_way_by_both() {
    let n = 130;
    let f = Fix::new(n);
    let u: Vec<u32> = (0..n as u32).collect();
    let masks: [&[u64]; 1] = [&f.region];
    let lanes = [LaneRef::I32(&f.l0), LaneRef::I32(&f.l1), LaneRef::U32(&u)];
    for which in [Which::ArgMin, Which::ArgMax] {
        let p = Program::new(vec![], terminal(which, Operand::Plane(0), 2));
        let (got, oracle) = eval(n, &masks, &lanes, &p);
        assert_eq!(got, oracle, "{which:?}");
        assert_eq!(
            got,
            Err(ExecError::LaneKind {
                lane: 2,
                expected: LaneKind::I32,
                found: LaneKind::U32,
            }),
            "{which:?}"
        );
    }
}

/// FAILS IF: a terminal reading a scratch slot nothing wrote is accepted, or
/// executor and oracle name different errors for it.
#[test]
fn an_unwritten_scratch_mask_is_refused_the_same_way_by_both() {
    let n = 130;
    let f = Fix::new(n);
    let masks: [&[u64]; 1] = [&f.region];
    let lanes = [LaneRef::I32(&f.l0)];
    for which in [Which::ArgMin, Which::ArgMax] {
        let p = Program::new(vec![], terminal(which, Operand::Scratch(0), 0));
        let (got, oracle) = eval(n, &masks, &lanes, &p);
        assert_eq!(got, oracle, "{which:?}");
        assert_eq!(got, Err(ExecError::ScratchReadBeforeWrite { slot: 0 }));
    }
}
