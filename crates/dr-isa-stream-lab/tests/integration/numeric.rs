//! bf16 is a numeric realization, not the value's meaning: an exactness
//! claim is refused where bf16 cannot hold the value.

use dr_isa_stream_lab::fixture::{bf16_exact, bf16_round, bf16_value};

#[test]
fn small_integers_are_exact_and_257_is_not() {
    for v in -256..=256 {
        assert_eq!(bf16_value(bf16_exact(v).unwrap()) as i32, v, "{v}");
    }
    assert!(bf16_exact(257).is_none(), "bf16 has 8 significant bits");
    assert!(bf16_exact(1_000_001).is_none());
    assert_eq!(bf16_exact(1 << 20).map(bf16_value), Some((1 << 20) as f32));
}

/// At the edge of `i32`, `v as f32` rounds up to 2^31, which a cast back to
/// `i32` saturates to `i32::MAX`. The check must not be fooled by that
/// saturation. Decoded values are compared in f64, which holds both exactly.
#[test]
fn exactness_is_not_fooled_by_saturating_casts() {
    for v in [i32::MAX, i32::MAX - 1, i32::MIN + 1] {
        assert!(bf16_exact(v).is_none(), "{v} is not a bf16 value");
    }
    for v in [i32::MIN, 1 << 30, -(1 << 30), 3 << 28] {
        let b = bf16_exact(v).unwrap_or_else(|| panic!("{v} is a bf16 value"));
        assert_eq!(f64::from(bf16_value(b)), f64::from(v), "{v}");
    }
}

/// A decimal-scaled value (cents as a float) is approximate in bf16; the
/// error is measured, not hidden.
#[test]
fn approximate_values_report_their_error() {
    let cents = [1.99f32, 12.34, 99.99, 1234.56];
    let mut worst = 0f32;
    for &v in &cents {
        let back = bf16_value(bf16_round(v));
        worst = worst.max(((back - v) / v).abs());
    }
    assert!(worst > 0.0, "these values are not exact in bf16");
    assert!(worst <= 2f32.powi(-8), "round-to-nearest bound: {worst}");
}
