//! Conformance: every admitted operator against an independent per-element
//! reference written from the ONNX operator definitions, every refusal, the
//! alias identities, and a cross-check of the composed reduction against
//! mask-risc, the r2il fold reading and relalg.

// The operator tables below pair a name, the facade function and its
// reference; spelling the tuple types out is the point.
#![allow(clippy::type_complexity)]

use lance_graph_onnx as onnx;
use lance_graph_onnx::{ReduceSum, Reduced, Refusal};
use ndarray::{arr0, Array1, Array2, ArrayD, IxDyn};

/// Values that separate an exact IEEE operator from a careless one.
const SPECIAL: [f32; 12] = [
    0.0,
    -0.0,
    1.0,
    -1.5,
    f32::NAN,
    f32::INFINITY,
    f32::NEG_INFINITY,
    f32::MIN_POSITIVE,
    1.0e-40, // subnormal
    -1.0e-40,
    f32::MAX,
    3.25,
];

fn samples(n: usize) -> Vec<f32> {
    (0..n)
        .map(|i| SPECIAL[i % SPECIAL.len()] * if i % 5 == 0 { 7.0 } else { 1.0 })
        .collect()
}

/// Bit equality, so `-0.0 != 0.0` and NaN payload classes count; any NaN
/// equals any NaN (ONNX does not fix NaN payloads).
fn same(a: f32, b: f32) -> bool {
    (a.is_nan() && b.is_nan()) || a.to_bits() == b.to_bits()
}

#[test]
fn unary_aliases_are_exact_on_special_values_and_every_tail_length() {
    for n in 0..70 {
        let x = Array1::from(samples(n));
        let mut out = Array1::<f32>::zeros(n);
        let cases: [(
            &str,
            fn(ndarray::ArrayView1<f32>, ndarray::ArrayViewMut1<f32>),
            fn(f32) -> f32,
        ); 2] = [
            ("Abs", onnx::abs_f32, f32::abs),
            ("Sqrt", onnx::sqrt_f32, f32::sqrt),
        ];
        for (name, op, reference) in cases {
            op(x.view(), out.view_mut());
            for i in 0..n {
                assert!(
                    same(out[i], reference(x[i])),
                    "{name}({}) = {} at n = {n}",
                    x[i],
                    out[i]
                );
            }
        }
        let y = onnx::neg(&x);
        for i in 0..n {
            assert!(same(y[i], -x[i]), "Neg({}) = {} at n = {n}", x[i], y[i]);
        }
    }
}

/// The aliases are the native items, not wrappers around them.
/// `vsneg` used to lose the sign of zero in its SIMD body, which is why `Neg`
/// is not an alias. ndarray #368 fixed it; this test now pins that `vsneg`
/// and `Neg` agree on +0 in the SIMD body and the tail. Making `Neg` an
/// alias is a separate change: `vsneg` is `f32`-only and writes into a view.
#[test]
fn vml_vsneg_matches_onnx_neg_on_positive_zero() {
    let x = Array1::<f32>::zeros(17);
    let mut out = Array1::<f32>::zeros(17);
    ndarray::hpc::vml::vsneg(x.view(), out.view_mut());
    let y = onnx::neg(&x);
    for i in [0, 16] {
        assert_eq!(out[i].to_bits(), (-0.0f32).to_bits(), "vsneg(+0) at {i}");
        assert_eq!(y[i].to_bits(), (-0.0f32).to_bits(), "Neg(+0) at {i}");
    }
}

#[test]
fn aliases_are_the_native_symbols() {
    type Cmp = fn(&[i32], i32, &mut [u64]);
    assert!(std::ptr::fn_addr_eq(
        onnx::greater_i32 as Cmp,
        ndarray::simd::gt_i32_to_mask as Cmp
    ));
    assert!(std::ptr::fn_addr_eq(
        onnx::less_i32 as Cmp,
        ndarray::simd::lt_i32_to_mask as Cmp
    ));
    assert!(std::ptr::fn_addr_eq(
        onnx::equal_i32 as Cmp,
        ndarray::simd::eq_i32_to_mask as Cmp
    ));
    type Bin = fn(&[u64], &[u64], &mut [u64]);
    assert!(std::ptr::fn_addr_eq(
        onnx::and as Bin,
        ndarray::simd::mask_and as Bin
    ));
    type Un = fn(ndarray::ArrayView1<f32>, ndarray::ArrayViewMut1<f32>);
    assert!(std::ptr::fn_addr_eq(
        onnx::abs_f32 as Un,
        ndarray::hpc::vml::vsabs as Un
    ));
    assert!(std::ptr::fn_addr_eq(
        onnx::bounded::tanh_f32 as Un,
        ndarray::hpc::vml::vstanh as Un
    ));
    assert!(std::ptr::fn_addr_eq(
        onnx::bounded::exp_f32 as Un,
        ndarray::hpc::vml::vsexp as Un
    ));
}

/// ULP distance on the total order of `f32`, with `-0` and `+0` one apart.
fn ulp_distance(a: f32, b: f32) -> u32 {
    let key = |x: f32| {
        let i = x.to_bits() as i32;
        if i < 0 {
            i32::MIN - i
        } else {
            i
        }
    };
    (key(a) as i64 - key(b) as i64).unsigned_abs() as u32
}

/// The ONNX reference for a bounded operator: `f64` evaluation rounded once.
/// Where that result is a zero, an infinity or NaN, or the input is a zero,
/// the operator must match it bit for bit; elsewhere within 2 ULP.
fn check_bounded(name: &str, x: f32, got: f32, want: f32) {
    if x == 0.0 || want == 0.0 || want.is_infinite() || want.is_nan() {
        assert!(
            same(got, want),
            "{name}({x:e}) = {got:e}, want exactly {want:e}"
        );
    } else {
        let u = ulp_distance(got, want);
        assert!(u <= 2, "{name}({x:e}) = {got:e}, want {want:e} ({u} ULP)");
    }
}

#[test]
fn bounded_aliases_hold_their_ulp_bound_on_special_values_and_every_tail_length() {
    let cases: [(
        &str,
        fn(ndarray::ArrayView1<f32>, ndarray::ArrayViewMut1<f32>),
        fn(f64) -> f64,
    ); 2] = [
        ("Tanh", onnx::bounded::tanh_f32, f64::tanh),
        ("Exp", onnx::bounded::exp_f32, f64::exp),
    ];
    for n in 0..70 {
        let x = Array1::from(samples(n));
        let mut out = Array1::<f32>::zeros(n);
        for (name, op, reference) in cases {
            op(x.view(), out.view_mut());
            for i in 0..n {
                check_bounded(name, x[i], out[i], reference(x[i] as f64) as f32);
            }
        }
    }
}

#[test]
fn bounded_aliases_hold_their_ulp_bound_across_magnitudes() {
    // Log-spaced magnitudes from subnormal to 1e4, both signs; the tanh small
    // branch, its 0.625 switch and saturation, and exp's subnormal and
    // overflow edges all fall inside.
    let n = 1 << 16;
    let x = Array1::from_iter((0..2 * n).map(|i| {
        let t = (i % n) as f64 / n as f64;
        let m = (10f64.powf(-44.0 + 48.0 * t)) as f32;
        if i < n {
            m
        } else {
            -m
        }
    }));
    let mut out = Array1::<f32>::zeros(x.len());
    onnx::bounded::tanh_f32(x.view(), out.view_mut());
    for i in 0..x.len() {
        check_bounded("Tanh", x[i], out[i], (x[i] as f64).tanh() as f32);
        assert!(
            out[i].abs() <= 1.0,
            "Tanh({:e}) = {:e} outside [-1, 1]",
            x[i],
            out[i]
        );
        if i >= n {
            assert_eq!(
                out[i].to_bits(),
                (-out[i - n]).to_bits(),
                "Tanh is odd at {:e}",
                x[i]
            );
        }
    }
    // Exp over its full f32 range, including subnormal results and overflow.
    let e = Array1::from_iter((0..n).map(|i| -104.0 + 194.0 * (i as f32 + 0.5) / n as f32));
    let mut out = Array1::<f32>::zeros(e.len());
    onnx::bounded::exp_f32(e.view(), out.view_mut());
    for i in 0..e.len() {
        check_bounded("Exp", e[i], out[i], (e[i] as f64).exp() as f32);
    }
}

/// Reference ONNX broadcasting: right-aligned, a 1 stretches.
fn reference_binary(a: &ArrayD<f32>, b: &ArrayD<f32>, f: fn(f32, f32) -> f32) -> ArrayD<f32> {
    let r = a.ndim().max(b.ndim());
    let pad = |s: &[usize]| {
        let mut v = vec![1; r - s.len()];
        v.extend_from_slice(s);
        v
    };
    let (sa, sb) = (pad(a.shape()), pad(b.shape()));
    let out: Vec<usize> = sa.iter().zip(&sb).map(|(&x, &y)| x.max(y)).collect();
    let a = a.view().into_shape_with_order(IxDyn(&sa)).unwrap();
    let b = b.view().into_shape_with_order(IxDyn(&sb)).unwrap();
    ArrayD::from_shape_fn(IxDyn(&out), |idx| {
        let ia: Vec<usize> = (0..r)
            .map(|k| if sa[k] == 1 { 0 } else { idx[k] })
            .collect();
        let ib: Vec<usize> = (0..r)
            .map(|k| if sb[k] == 1 { 0 } else { idx[k] })
            .collect();
        f(a[IxDyn(&ia)], b[IxDyn(&ib)])
    })
}

#[test]
fn binary_arithmetic_matches_onnx_broadcasting() {
    let shapes: [(&[usize], &[usize]); 7] = [
        (&[37], &[37]),
        (&[3, 1], &[4]),
        (&[2, 3], &[3]),
        (&[], &[5]),
        (&[5], &[]),
        (&[2, 1, 4], &[3, 1]),
        (&[1], &[6, 1]),
    ];
    for (sa, sb) in shapes {
        let na: usize = sa.iter().product();
        let nb: usize = sb.iter().product();
        let a = ArrayD::from_shape_vec(IxDyn(sa), samples(na)).unwrap();
        let b = ArrayD::from_shape_vec(IxDyn(sb), samples(nb + 3)[3..].to_vec()).unwrap();
        type Op = fn(
            &ndarray::ArrayRef<f32, IxDyn>,
            &ndarray::ArrayRef<f32, IxDyn>,
        ) -> Result<ArrayD<f32>, Refusal>;
        let ops: [(&str, Op, fn(f32, f32) -> f32); 4] = [
            ("Add", onnx::add, |x, y| x + y),
            ("Sub", onnx::sub, |x, y| x - y),
            ("Mul", onnx::mul, |x, y| x * y),
            ("Div", onnx::div, |x, y| x / y),
        ];
        for (name, op, f) in ops {
            let got = op(&a, &b).unwrap();
            let want = reference_binary(&a, &b, f);
            assert_eq!(got.shape(), want.shape(), "{name} {sa:?} {sb:?}");
            for (g, w) in got.iter().zip(want.iter()) {
                assert!(same(*g, *w), "{name} {sa:?} {sb:?}: {g} vs {w}");
            }
        }
    }
    // Static ranks broadcast too: a 0-d scalar against a matrix.
    let m = Array2::from_shape_vec((2, 2), vec![1.0f32, 2.0, 3.0, 4.0]).unwrap();
    assert_eq!(onnx::mul(&m, &arr0(2.0f32)).unwrap(), m.mapv(|v| v * 2.0));
}

#[test]
fn incompatible_shapes_are_refused_not_panicked() {
    let a = ArrayD::<f32>::zeros(IxDyn(&[3]));
    let b = ArrayD::<f32>::zeros(IxDyn(&[4]));
    assert_eq!(
        onnx::add(&a, &b),
        Err(Refusal::Broadcast {
            lhs: vec![3],
            rhs: vec![4]
        })
    );
    let c = ArrayD::<f32>::zeros(IxDyn(&[2, 3]));
    let d = ArrayD::<f32>::zeros(IxDyn(&[2]));
    assert!(matches!(onnx::div(&c, &d), Err(Refusal::Broadcast { .. })));
    // The check is silent on every admitted pair (it must not refuse what
    // ndarray would compute).
    assert_eq!(onnx::broadcastable(&[2, 1, 4], &[3, 1]), Ok(()));
    assert_eq!(onnx::broadcastable(&[], &[7, 7]), Ok(()));
}

fn pack(bits: impl Iterator<Item = bool>, n: usize) -> Vec<u64> {
    let mut w = vec![0u64; onnx::mask_words(n)];
    for (i, b) in bits.enumerate() {
        if b {
            w[i / 64] |= 1 << (i % 64);
        }
    }
    w
}

fn lane(n: usize) -> Vec<i32> {
    (0..n)
        .map(|i| match i % 9 {
            0 => i32::MIN,
            1 => i32::MAX,
            _ => (i as i32 * 37) % 23 - 11,
        })
        .collect()
}

#[test]
fn comparisons_against_a_scalar_match_the_reference() {
    type Cmp = fn(&[i32], i32, &mut [u64]);
    let ops: [(&str, Cmp, fn(i32, i32) -> bool); 5] = [
        ("Greater", onnx::greater_i32, |x, t| x > t),
        ("GreaterOrEqual", onnx::greater_or_equal_i32, |x, t| x >= t),
        ("Less", onnx::less_i32, |x, t| x < t),
        ("LessOrEqual", onnx::less_or_equal_i32, |x, t| x <= t),
        ("Equal", onnx::equal_i32, |x, t| x == t),
    ];
    for n in [0, 1, 15, 16, 17, 63, 64, 65, 130] {
        let x = lane(n);
        for t in [i32::MIN, -3, 0, 5, i32::MAX] {
            for (name, op, f) in ops {
                let mut got = vec![u64::MAX; onnx::mask_words(n)];
                op(&x, t, &mut got);
                assert_eq!(
                    got,
                    pack(x.iter().map(|&v| f(v, t)), n),
                    "{name} n={n} t={t}"
                );
            }
        }
    }
}

#[test]
fn boolean_ops_and_where_match_the_reference() {
    for n in [1, 63, 64, 65, 200] {
        let p: Vec<bool> = (0..n).map(|i| i % 3 == 0).collect();
        let q: Vec<bool> = (0..n).map(|i| i % 5 < 2).collect();
        let (a, b) = (pack(p.iter().copied(), n), pack(q.iter().copied(), n));
        let mut out = vec![0u64; a.len()];
        onnx::and(&a, &b, &mut out);
        assert_eq!(out, pack((0..n).map(|i| p[i] & q[i]), n));
        onnx::or(&a, &b, &mut out);
        assert_eq!(out, pack((0..n).map(|i| p[i] | q[i]), n));
        onnx::xor(&a, &b, &mut out);
        assert_eq!(out, pack((0..n).map(|i| p[i] ^ q[i]), n));
        onnx::not(&a, n, &mut out);
        assert_eq!(
            out,
            pack((0..n).map(|i| !p[i]), n),
            "Not keeps the tail zero"
        );

        let x = lane(n);
        let y: Vec<i32> = (0..n as i32).collect();
        let mut dst = vec![0i32; n];
        onnx::where_i32(&a, &x, &y, &mut dst);
        let want: Vec<i32> = (0..n).map(|i| if p[i] { x[i] } else { y[i] }).collect();
        assert_eq!(dst, want);
    }
}

#[test]
fn reduce_sum_attributes_are_validated_once() {
    assert!(ReduceSum::new(None, 0, 0).is_ok());
    assert!(ReduceSum::new(Some(&[0]), 1, 0).is_ok());
    assert!(ReduceSum::new(Some(&[-1]), 0, 1).is_ok());
    assert_eq!(
        ReduceSum::new(Some(&[1]), 0, 0),
        Err(Refusal::Axes(vec![1]))
    );
    assert_eq!(
        ReduceSum::new(Some(&[0, 0]), 0, 0),
        Err(Refusal::Axes(vec![0, 0]))
    );
    assert_eq!(ReduceSum::new(None, 2, 0), Err(Refusal::KeepDims(2)));
    assert_eq!(ReduceSum::new(None, 0, 1), Err(Refusal::NoopWithEmptyAxes));
}

#[test]
fn reduce_sum_is_exact_and_refuses_int32_overflow() {
    let rs = ReduceSum::new(None, 1, 0).unwrap();
    assert_eq!(
        rs.sum_i32(&[]),
        Ok(Reduced {
            value: 0,
            keepdims: true
        })
    );
    let x: Vec<i32> = (-50..70).collect();
    assert_eq!(rs.sum_i32(&x).unwrap().value, x.iter().sum::<i32>());
    assert_eq!(
        rs.sum_i32(&[i32::MAX, 1]),
        Err(Refusal::OutOfDtypeRange(i64::from(i32::MAX) + 1))
    );
    // An intermediate sum may leave int32 as long as the result fits.
    assert_eq!(rs.sum_i32(&[i32::MAX, 1, -1]).unwrap().value, i32::MAX);
}

fn oracle_sum_where_greater(x: &[i32], t: i32) -> i64 {
    x.iter().filter(|&&v| v > t).map(|&v| i64::from(v)).sum()
}

#[test]
fn composite_matches_the_oracle_mask_risc_r2il_and_relalg() {
    use lance_graph_mask_risc::exec::{execute_into, Scratch};
    use lance_graph_mask_risc::ir::{MaskOp, Operand, Pred, Program, Terminal};
    use lance_graph_mask_risc::{Foreign, LaneRef, Out, Planes, Value};
    use lance_graph_r2il::{FoldDialect, Val, INT_S_LESS, LOAD};
    use lance_graph_relalg::{run, Answer, Atom, Cmp, Measure, Query, Rel};

    let rs = ReduceSum::new(None, 0, 0).unwrap();
    for n in [0usize, 1, 64, 65, 1000, 4097] {
        let x: Vec<i32> = (0..n).map(|i| ((i * 13) % 97) as i32 - 48).collect();
        let lanes = [LaneRef::I32(&x)];
        let planes = Planes {
            n_rows: n,
            masks: &[],
            lanes: &lanes,
        };
        for t in [-49, -1, 0, 30, 48] {
            let want = oracle_sum_where_greater(&x, t);
            let mut scratch = vec![0u64; onnx::mask_words(n)];
            assert_eq!(
                rs.sum_where_greater_i32(&x, t, &mut scratch).unwrap().value as i64,
                want,
                "facade n={n} t={t}"
            );

            // mask-risc: Pred::GtI32 → Terminal::MaskedSumI32.
            let p = Program::new(
                vec![MaskOp::Pred {
                    pred: Pred::GtI32 { lane: 0, t },
                    under: None,
                    dst: 0,
                }],
                Terminal::MaskedSumI32 {
                    mask: Operand::Scratch(0),
                    lane: 0,
                },
            );
            let mut buf = vec![0u64; lance_graph_mask_risc::exec::tile_words_for(n).max(1) * 2];
            let mut s = Scratch::over_for_program(&mut buf, &p, n).unwrap();
            assert_eq!(
                execute_into(&p, &planes, &Foreign::NONE, &mut s, Out::None),
                Ok(Value::SumI64(want)),
                "mask-risc n={n} t={t}"
            );

            // The r2il fold reading: t < x, then SUM.
            let mut d = FoldDialect::new(&planes, &Foreign::NONE);
            let mut stack = vec![Val::Scalar(i64::from(t)), Val::Scalar(0)];
            d.call(LOAD, [1, 0, 0], &mut stack).unwrap();
            d.call(INT_S_LESS, [0; 3], &mut stack).unwrap();
            lance_graph_r2il::procs::sum(&mut d, &mut stack, 0).unwrap();
            assert_eq!(stack.pop(), Some(Val::Scalar(want)), "r2il n={n} t={t}");

            // relalg: σ(x > t), Σ x.
            let q = Query {
                rel: Rel::Atom(Atom::I32 {
                    lane: 0,
                    op: Cmp::Gt,
                    t,
                }),
                measure: Measure::Sum(0),
            };
            assert_eq!(
                run(&q, &planes, None).unwrap().answer,
                Answer::Sum(want),
                "relalg n={n} t={t}"
            );
        }
    }
}

#[test]
fn composite_refuses_int32_overflow_and_keeps_keepdims() {
    let rs = ReduceSum::new(Some(&[0]), 1, 0).unwrap();
    let x = [i32::MAX, i32::MAX, -5];
    let mut scratch = [0u64; 1];
    assert_eq!(
        rs.sum_where_greater_i32(&x, 0, &mut scratch),
        Err(Refusal::OutOfDtypeRange(2 * i64::from(i32::MAX)))
    );
    assert_eq!(
        rs.sum_where_greater_i32(&[1, 2, 3], 1, &mut scratch),
        Ok(Reduced {
            value: 5,
            keepdims: true
        })
    );
}
