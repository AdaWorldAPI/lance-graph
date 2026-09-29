//! Key monotonicity (F8) and the lowering oracle (V3) on fixed inputs.

use perturbationsfeld_probe::helpers::{key, permutation};
use perturbationsfeld_probe::{e_ids, e_m_ids, verdict, Outcome};

#[test]
fn key_is_strictly_monotone_over_finite_f32() {
    let xs = [
        f32::MIN,
        -1.0e30,
        -2.0,
        -1.0,
        -f32::MIN_POSITIVE,
        -1.0e-45, // negative subnormal
        -0.0,
        0.0,
        1.0e-45, // positive subnormal
        f32::MIN_POSITIVE,
        0.01,
        0.5,
        1.0,
        f32::MAX,
    ];
    for w in xs.windows(2) {
        assert!(key(w[0]) < key(w[1]), "{} vs {}", w[0], w[1]);
    }
}

#[test]
fn key_fails_on_a_wrong_map_can_it_fire() {
    // The naive bit cast is NOT monotone for negatives: the test above would
    // catch it. Prove the property actually discriminates.
    let naive = |e: f32| e.to_bits() as i32;
    assert!(naive(-2.0) > naive(-1.0));
    assert!(key(-2.0) < key(-1.0));
}

#[test]
fn e_arm_matches_scalar_and_is_not_trivial() {
    let mut e = vec![0.0f32; 256];
    for (i, v) in e.iter_mut().enumerate() {
        *v = if i % 7 == 0 {
            0.02
        } else {
            0.001 * (i % 5) as f32
        };
    }
    let (ids, agree) = e_ids(&e, 0.01);
    assert!(agree, "mask-risc E mask disagrees with scalar or oracle");
    let expected: Vec<u16> = (0..256u16).filter(|i| i % 7 == 0).collect();
    assert_eq!(ids, expected);
    // anti-vacuity: the filter excludes most rows
    assert!(ids.len() * 3 < 256);
}

#[test]
fn e_m_is_top_k_by_energy() {
    let e = [0.0, 0.3, 0.1, 0.3, 0.0, 0.2];
    assert_eq!(e_m_ids(&e, 3), vec![1, 3, 5]);
    assert_eq!(e_m_ids(&e, 10), vec![1, 2, 3, 5]);
}

#[test]
fn permutation_is_a_bijection_and_moves_rows() {
    let p = permutation(256, 0x5EED_0000_0000_0001);
    let mut s = p.clone();
    s.sort_unstable();
    assert_eq!(s, (0..256u16).collect::<Vec<_>>());
    assert!(
        p.iter()
            .enumerate()
            .filter(|&(i, &v)| i as u16 != v)
            .count()
            > 200
    );
}

#[test]
fn verdict_combinations() {
    use Outcome::*;
    assert_eq!(
        verdict(&HigherRetention, &HigherRetention),
        "HIGHER-RETENTION"
    );
    assert_eq!(
        verdict(&HigherRetention, &NoRetentionDifference),
        "HIGHER-RETENTION, LENS-SPECIFIC"
    );
    assert_eq!(
        verdict(&InputInsensitive, &Invalid(vec![])),
        "INPUT-INSENSITIVE, REPLICATION INVALID"
    );
    assert_eq!(verdict(&Invalid(vec![]), &HigherRetention), "INVALID");
}
