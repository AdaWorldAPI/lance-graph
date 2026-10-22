//! A palette coordinate is an address. Order *pairwise scores*, never IDs.
//! Within one family gamma, encoding is monotone (apart from ties).
use bgz_tensor::fisher_z::FamilyGamma;

/// Score intervals in the same declared family and transformed domain.
/// None means the winner is ambiguous at this precision.
fn certified_argmax(scores: &[(f64, f64)]) -> Option<usize> {
    scores.iter().enumerate().find_map(|(i, &(s, eps))| {
        if !s.is_finite() || !eps.is_finite() || eps < 0.0 { return None; }
        ((0..scores.len()).all(|j| i == j || s - eps > scores[j].0 + scores[j].1)).then_some(i)
    })
}
fn certified_argmin(scores: &[(f64, f64)]) -> Option<usize> {
    scores.iter().enumerate().find_map(|(i, &(s, eps))| {
        if !s.is_finite() || !eps.is_finite() || eps < 0.0 { return None; }
        ((0..scores.len()).all(|j| i == j || s + eps < scores[j].0 - scores[j].1)).then_some(i)
    })
}
#[test]
fn monotone_fisher_encoding_preserves_clear_order_within_one_family() {
    let data = [-0.65_f32, -0.3, 0.1, 0.65];
    let family = FamilyGamma::from_cosines(&data);
    let codes: Vec<i8> = data.iter().copied().map(|x| family.encode(x)).collect();
    assert!(codes.windows(2).all(|w| w[0] < w[1]), "chosen fixture avoids ties");
    assert_eq!(codes.iter().enumerate().max_by_key(|(_, x)| *x).unwrap().0, 3);
    assert_eq!(codes.iter().enumerate().min_by_key(|(_, x)| *x).unwrap().0, 0);
    let sortable_unsigned: Vec<u8> = codes.iter().map(|&i| (i as u8) ^ 0x80).collect();
    assert!(sortable_unsigned.windows(2).all(|w| w[0] < w[1]));
}
#[test]
fn certify_winner_or_abstain_when_error_intervals_overlap() {
    assert_eq!(certified_argmax(&[(0.20, 0.01), (0.50, 0.02), (0.30, 0.01)]), Some(1));
    assert_eq!(certified_argmin(&[(0.20, 0.01), (0.50, 0.02), (0.30, 0.01)]), Some(0));
    assert_eq!(certified_argmax(&[(0.500, 0.02), (0.505, 0.02)]), None);
    assert_eq!(certified_argmin(&[(0.500, 0.02), (0.505, 0.02)]), None);
}
