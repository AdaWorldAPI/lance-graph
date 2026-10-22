//! Distinguish four claims: frozen lookup (no re-encode), quantized contraction,
//! shared-source false precision, and semantic correctness. Only the first two
//! have numerical properties; bounded outputs do not certify factual truth.
fn q8(x: f64) -> f64 { (x.clamp(-1.0, 1.0) * 127.0).round() / 127.0 }
#[test]
fn stable_lookup_of_immutable_values_does_not_requantize_per_hop() {
    const PAIRS: [[i8; 3]; 3] = [[127, 64, -40], [33, 127, 10], [-55, 15, 127]];
    let original = PAIRS;
    let mut address = 0_usize;
    for _ in 0..64 {
        let next = (address + 1) % 3;
        let relation = PAIRS[address][next];
        assert_eq!(relation, original[address][next]);
        address = next; // traverse addresses; never feed a decoded score into bake
    }
    assert_eq!(PAIRS, original);
}
#[test]
fn contractive_quantized_recursion_has_a_measurable_64hop_bound() {
    let (mut exact, mut quantized) = (0.25_f64, 0.25_f64);
    let lipschitz = 0.7_f64;
    let one_step_rounding = 0.5 / 127.0 + 1e-12;
    for hop in 1..=64 {
        exact = lipschitz * exact + 0.08;
        quantized = q8(lipschitz * quantized + 0.08);
        let envelope = one_step_rounding * (1.0 - lipschitz.powi(hop)) / (1.0 - lipschitz);
        assert!((exact - quantized).abs() <= envelope + 1e-12);
    }
}
#[test]
fn saturation_and_duplicate_paths_can_preserve_an_unsupported_claim() {
    let mut spurious_score = 0.01_f64;
    for _ in 0..16 { spurious_score = (1.5 * spurious_score).clamp(-1.0, 1.0); }
    assert!(spurious_score > 0.5, "finite range alone does NOT prevent reinforcement");
    let shared_root = 1_u64 << 7;
    let paths = [shared_root; 32];
    let distinct = paths.iter().copied().fold(0u64, |a, b| a | b).count_ones();
    assert_eq!(distinct, 1, "32 derived paths are one independent observation");
}
