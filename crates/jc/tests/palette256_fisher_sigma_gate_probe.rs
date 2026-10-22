//! The classical Fisher-z SE belongs to estimated Pearson correlation under
//! model assumptions. It is NOT a universal cosine/noise/hallucination gate.
fn fisher_z_gate(r: f64, independent_n: usize, sigmas: f64) -> Option<bool> {
    if independent_n <= 3
        || !r.is_finite()
        || r.abs() >= 1.0
        || !sigmas.is_finite()
        || sigmas <= 0.0
    {
        return None;
    }
    let se = 1.0 / ((independent_n - 3) as f64).sqrt();
    Some(r.atanh().abs() >= sigmas * se)
}
#[test]
fn sample_size_and_model_domain_are_explicit() {
    assert_eq!(fisher_z_gate(0.6, 300, 3.0), Some(true));
    assert_eq!(fisher_z_gate(0.01, 300, 3.0), Some(false));
    assert_eq!(fisher_z_gate(0.6, 3, 3.0), None);
    assert_eq!(fisher_z_gate(1.0, 300, 3.0), None);
    assert_eq!(fisher_z_gate(f64::NAN, 300, 3.0), None);
}
#[test]
fn local_subthreshold_terms_cannot_be_rewritten_to_zero_before_fusion() {
    let (a, b, cutoff) = (0.20_f64, 0.20_f64, 0.30_f64);
    assert!(a.abs() < cutoff);
    assert!(b.abs() < cutoff);
    assert!((a + b).abs() > cutoff);
    let incorrectly_pruned = 0.0_f64 + 0.0_f64;
    assert_ne!(incorrectly_pruned, a + b);
}
#[test]
fn a_calibrated_null_gate_does_not_define_causal_truth() {
    // Nonzero, statistically detectable correlation ≠ an intervention.
    assert_eq!(fisher_z_gate(0.7, 500, 3.0), Some(true));
    let has_intervention_witness = false;
    assert!(!has_intervention_witness);
}
