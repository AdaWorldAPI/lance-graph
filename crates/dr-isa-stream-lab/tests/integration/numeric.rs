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
