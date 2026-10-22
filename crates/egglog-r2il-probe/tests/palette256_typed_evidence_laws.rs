//! Cold admission probes for mathematical candidate laws, *not* a new runtime
//! e-graph. Keep existing egglog -> R2IL -> mask-risc ownership boundaries.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Domain {
    OpenUnitReal,
    IndependentCounts,
    BayesLogLikelihood,
    FisherCorrelation,
    QuantizedPalette,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Law {
    TanhAtanhCancel,
    MergeIndependentCounts,
    AddLogLikelihoodRatios,
    UnconditionalThreeSigmaPrune,
}
fn admit(law: Law, domain: Domain) -> bool {
    matches!(
        (law, domain),
        (Law::TanhAtanhCancel, Domain::OpenUnitReal)
        | (Law::MergeIndependentCounts, Domain::IndependentCounts)
        | (Law::AddLogLikelihoodRatios, Domain::BayesLogLikelihood)
    )
}
#[test]
fn exact_identity_is_admitted_only_in_its_declared_numeric_domain() {
    for x in [-0.99_f64, -0.5, 0.0, 0.25, 0.99] {
        assert!((x.atanh().tanh() - x).abs() < 1e-12);
    }
    assert!(admit(Law::TanhAtanhCancel, Domain::OpenUnitReal));
    assert!(!admit(Law::TanhAtanhCancel, Domain::QuantizedPalette));
    assert!(!admit(Law::TanhAtanhCancel, Domain::FisherCorrelation));
    assert!(!admit(Law::UnconditionalThreeSigmaPrune, Domain::OpenUnitReal));
    assert!(1.0_f64.atanh().is_infinite());
}
#[test]
fn bayesian_log_likelihood_addition_is_not_nars_frequency_revision() {
    let f1 = 0.7_f64;
    let f2 = 0.7_f64;
    let nars_revision = (7.0 + 7.0) / (10.0 + 10.0);
    let signed1 = 2.0 * f1 - 1.0;
    let signed2 = 2.0 * f2 - 1.0;
    let hyperbolic = ((signed1.atanh() + signed2.atanh()).tanh() + 1.0) * 0.5;
    assert!((nars_revision - 0.7).abs() < 1e-12);
    assert!(hyperbolic > 0.8, "different evidence algebra, not a compiler optimization");
    assert!(admit(Law::MergeIndependentCounts, Domain::IndependentCounts));
    assert!(!admit(Law::MergeIndependentCounts, Domain::BayesLogLikelihood));
    assert!(admit(Law::AddLogLikelihoodRatios, Domain::BayesLogLikelihood));
    assert!(!admit(Law::AddLogLikelihoodRatios, Domain::IndependentCounts));
}
#[test]
fn tanh_addition_identity_is_a_distinct_true_mathematical_law() {
    let a: f64 = 0.4;
    let b: f64 = -0.2;
    let left = (a + b).tanh();
    let right = (a.tanh() + b.tanh()) / (1.0 + a.tanh() * b.tanh());
    assert!((left - right).abs() < 1e-12);
}
