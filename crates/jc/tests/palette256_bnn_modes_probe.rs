//! BNN is ambiguous: Binarized Neural Network vs Bayesian Neural Network.
//! A bitwise inference accelerator does NOT imply posterior calibration.
fn bipolar_dot(x: u64, y: u64, active: u64) -> i32 {
    let same = (!(x ^ y) & active).count_ones() as i32;
    let dims = active.count_ones() as i32;
    2 * same - dims
}
fn scalar_oracle(x: u64, y: u64, active: u64) -> i32 {
    (0..64)
        .filter(|&bit| active & (1_u64 << bit) != 0)
        .map(|bit| {
            if (x & (1_u64 << bit) != 0) == (y & (1_u64 << bit) != 0) {
                1
            } else {
                -1
            }
        })
        .sum()
}
fn jeffreys_beta_stats(positive: f64, negative: f64) -> (f64, f64) {
    let a = positive + 0.5;
    let b = negative + 0.5;
    let mean = a / (a + b);
    let variance = a * b / ((a + b) * (a + b) * (a + b + 1.0));
    (mean, variance)
}
#[test]
fn binarized_net_dot_matches_independent_bitwise_scalar_oracle() {
    let pairs = [
        (0xAAAA_AAAA_AAAA_AAAA, 0x5555_5555_5555_5555, u64::MAX),
        (u64::MAX, 0, 0xFFFF_0000_FFFF_0000),
        (0x1357_9BDF_2468_ACE0, 0xDEAD_BEEF_1234_5678, u64::MAX),
        (0, 0, 0),
    ];
    for (x, y, mask) in pairs {
        assert_eq!(bipolar_dot(x, y, mask), scalar_oracle(x, y, mask));
    }
}
#[test]
fn bayesian_model_with_same_mean_can_have_different_uncertainty() {
    let (mean_small, var_small) = jeffreys_beta_stats(1.0, 1.0);
    let (mean_large, var_large) = jeffreys_beta_stats(100.0, 100.0);
    assert_eq!(mean_small, mean_large);
    assert!(var_large < var_small);
}
