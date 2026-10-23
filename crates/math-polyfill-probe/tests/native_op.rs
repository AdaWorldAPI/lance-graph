//! D-MPF-1b: one native op (`SumWhere`) and its ONNX-semantics wrapper.
//! Every answer is compared with `sum_where_oracle` (an `i128` row loop) or
//! with ONNX's own definition written out by hand.

use lance_graph_mask_risc::{words_for, Foreign, LaneRef, Planes};
use lance_graph_r2il::{FoldDialect, FoldError};
use math_polyfill_probe::native::{sum_where_oracle, Pred, SumWhere};
use math_polyfill_probe::onnx_compat::{reduce_sum_where_greater, Output, Refusal};

fn lane(n: usize) -> Vec<i32> {
    (0..n).map(|i| ((i * 13) % 97) as i32 - 48).collect()
}

fn plane(n: usize) -> Vec<u64> {
    let mut w = vec![0u64; words_for(n)];
    for r in (0..n).filter(|r| r % 7 != 3) {
        w[r / 64] |= 1 << (r % 64);
    }
    w
}

const PREDS: [Pred; 8] = [
    Pred::Gt(0),
    Pred::Ge(0),
    Pred::Lt(0),
    Pred::Le(0),
    Pred::Gt(100),  // empty
    Pred::Lt(-100), // empty
    Pred::Ge(-48),  // everything
    Pred::Le(-48),  // one value class
];

/// FAILS IF: any predicate form, the NULL plane, or the sum diverges from the
/// row loop, at one word, a ragged tail and several tiles; on one reused
/// dialect, so a stale-slot bug across calls would show.
#[test]
fn sum_where_equals_the_oracle_on_a_reused_dialect() {
    for n in [64, 1_000, 65_537] {
        let x = lane(n);
        let p = plane(n);
        let lanes = [LaneRef::I32(&x)];
        let masks: [&[u64]; 1] = [&p];
        let planes = Planes {
            n_rows: n,
            masks: &masks,
            lanes: &lanes,
        };
        let mut d = FoldDialect::new(&planes, &Foreign::NONE);
        let mut s = Vec::new();
        for pred in PREDS {
            for present in [None, Some(0u16)] {
                let got = SumWhere {
                    lane: 0,
                    present,
                    pred,
                }
                .eval(&mut d, &mut s);
                let want = sum_where_oracle(&x, present.map(|_| p.as_slice()), pred);
                assert_eq!(got.map(i128::from), Ok(want), "n={n} {pred:?} {present:?}");
            }
        }
    }
}

/// FAILS IF: the fixture is vacuous: some predicate must select nothing, some
/// a strict subset, and the NULL plane must change at least one answer.
#[test]
fn the_fixture_is_not_vacuous() {
    let x = lane(1_000);
    let p = plane(1_000);
    assert_eq!(sum_where_oracle(&x, None, Pred::Gt(100)), 0);
    let all: i128 = x.iter().map(|&v| i128::from(v)).sum();
    let part = sum_where_oracle(&x, None, Pred::Gt(0));
    assert!(part != 0 && part != all);
    assert_ne!(
        sum_where_oracle(&x, None, Pred::Gt(0)),
        sum_where_oracle(&x, Some(&p), Pred::Gt(0))
    );
}

/// ONNX's definition, written out: Greater → bool, Where(c, X, 0), ReduceSum.
fn onnx_reference(x: &[i32], t: i32) -> i64 {
    let cond: Vec<bool> = x.iter().map(|&v| v > t).collect();
    let picked: Vec<i64> = x
        .iter()
        .zip(&cond)
        .map(|(&v, &c)| if c { i64::from(v) } else { 0 })
        .collect();
    picked.iter().sum()
}

/// FAILS IF: the wrapper's answer differs from ONNX's definition for any
/// admitted attribute combination.
#[test]
fn the_wrapper_matches_the_onnx_definition() {
    let x = lane(1_000);
    for t in [-100, -1, 0, 7, 100] {
        let want = i32::try_from(onnx_reference(&x, t)).unwrap();
        for (axes, kd) in [
            (None, 0),
            (Some(&[0i64][..]), 0),
            (Some(&[-1][..]), 1),
            (None, 1),
        ] {
            for t_shape in [&[][..], &[1][..]] {
                let got = reduce_sum_where_greater(&x, &[1_000], t, t_shape, axes, kd, 0);
                assert_eq!(
                    got,
                    Ok(Output {
                        value: want,
                        keepdims: kd == 1
                    }),
                    "t={t} {axes:?} {kd}"
                );
            }
        }
    }
}

/// FAILS IF: an out-of-subset call is answered instead of refused.
#[test]
fn out_of_subset_calls_are_refused() {
    let x = lane(8);
    let r = |shape: &[usize], ts: &[usize], axes: Option<&[i64]>, kd: i64, noop: i64| {
        reduce_sum_where_greater(&x, shape, 0, ts, axes, kd, noop)
    };
    assert_eq!(r(&[2, 4], &[], None, 0, 0), Err(Refusal::Rank(2)));
    assert_eq!(
        r(&[8], &[8], None, 0, 0),
        Err(Refusal::ThresholdNotScalar(vec![8]))
    );
    assert_eq!(r(&[8], &[], Some(&[1]), 0, 0), Err(Refusal::Axes(vec![1])));
    assert_eq!(r(&[8], &[], None, 2, 0), Err(Refusal::KeepDims(2)));
    assert_eq!(r(&[8], &[], None, 0, 1), Err(Refusal::NoopWithEmptyAxes));
    // The silent twin: the same call inside the subset answers.
    assert!(r(&[8], &[], None, 0, 0).is_ok());
}

/// FAILS IF: an int32 sum that leaves int32 wraps (as a naive i32 loop would)
/// instead of being refused; a sum that fits is still answered.
#[test]
fn int32_overflow_is_refused_not_wrapped() {
    let x = vec![i32::MAX, 1];
    let exact = i64::from(i32::MAX) + 1;
    assert_eq!(
        reduce_sum_where_greater(&x, &[2], 0, &[], None, 0, 0),
        Err(Refusal::OutOfDtypeRange(exact))
    );
    assert_eq!(
        x.iter().fold(0i32, |a, &b| a.wrapping_add(b)),
        i32::MIN,
        "what wrapping would answer"
    );
    assert_eq!(
        reduce_sum_where_greater(&x, &[2], 1, &[], None, 0, 0),
        Ok(Output {
            value: i32::MAX,
            keepdims: false
        })
    );
}

/// FAILS IF: the native op stops refusing an i64 overflow (2^32 rows of
/// i32::MAX would be needed for real data, so this pins the error type the
/// wrapper forwards).
#[test]
fn native_refusal_is_forwarded() {
    assert_eq!(
        format!(
            "{:?}",
            Refusal::Native(FoldError::Overflow(lance_graph_r2il::INT_ADD))
        ),
        format!(
            "Native({:?})",
            FoldError::Overflow(lance_graph_r2il::INT_ADD)
        )
    );
}
