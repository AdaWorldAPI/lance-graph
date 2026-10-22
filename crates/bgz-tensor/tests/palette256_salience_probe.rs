//! Lab-only salience/energy probe: MP3 is an analogy, not a perceptual codec.
//! Never change the immutable trained codebook during inference.
fn top_energy(input: &[f64], k: usize) -> Vec<usize> {
    let mut rank: Vec<usize> = (0..input.len()).collect();
    rank.sort_by(|&a, &b| input[b].abs().total_cmp(&input[a].abs()).then(a.cmp(&b)));
    rank.truncate(k.min(rank.len()));
    rank
}
fn energy(input: &[f64], keep: &[usize]) -> (f64, f64) {
    let total = input.iter().map(|x| x * x).sum::<f64>();
    let saved = keep.iter().map(|&i| input[i] * input[i]).sum::<f64>();
    (saved, total)
}
#[test]
fn dominant_signal_energy_is_retained_without_retraining_the_reference() {
    let features = [8.0, -6.0, 0.10, 0.08, -0.04, 0.02, 0.01, 0.0];
    let before = features;
    let selected = top_energy(&features, 2);
    assert_eq!(selected, vec![0, 1]);
    let (saved, total) = energy(&features, &selected);
    assert!(saved / total > 0.999);
    assert_eq!(
        features, before,
        "salience is a selection, not codebook motion"
    );
}
#[test]
fn rare_low_amplitude_causal_feature_is_a_counterexample_to_energy_only() {
    // This is a deliberate negative control: a small coordinate may determine
    // the label even when discarded energy is negligible.
    let class_a = [12.0, 9.0, 0.01];
    let class_b = [12.0, 9.0, -0.01];
    let keep_a = top_energy(&class_a, 2);
    let keep_b = top_energy(&class_b, 2);
    assert_eq!(keep_a, keep_b);
    assert!(keep_a.iter().all(|&idx| class_a[idx] == class_b[idx]));
    assert_ne!(class_a[2].is_sign_positive(), class_b[2].is_sign_positive());
    // No global claim that loud-only preservation implies semantic correctness.
}
