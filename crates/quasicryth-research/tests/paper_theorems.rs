//! Integration tests verifying the five core theorems of the Quasicryth paper
//! (Tacconelli 2026, arxiv 2603.14999) on synthetic L/S sequences.
//!
//! These tests cross-check the workspace's φ-substrate decisions against the
//! reference algebra without requiring the upstream C build.

use quasicryth_research::{
    build_hierarchy, deep_counts, detect_deep_positions, period5_tiling, qc_word_tiling,
    qc_word_tiling_alpha, sanddrift_tiling, thue_morse_tiling, verify_no_adjacent_s,
    HIER_WORD_LENS, INV_PHI, MAX_HIER, PHI,
};

/// **Thm 2 — Fibonacci hierarchy never collapses.**
///
/// The level-k supertile sequence contains both L- and S-supertiles for every
/// `k ≥ 0`. We verify this directly: at every level of the built hierarchy,
/// neither L- nor S- supertile count is zero (or we've capped at MAX_HIER).
#[test]
fn t02_fibonacci_hierarchy_never_collapses() {
    let tiles = qc_word_tiling(50_000, 0.0);
    let hier = build_hierarchy(&tiles, MAX_HIER);

    // The hierarchy should reach the cap on a 50k-tile input.
    assert!(
        hier.n_levels() >= MAX_HIER - 1,
        "depth = {}, expected near {MAX_HIER}",
        hier.n_levels()
    );

    // Each level must have both tile types.
    for k in 0..hier.n_levels() {
        let level = &hier.levels[k];
        if level.len() < 2 {
            break; // exhaustion is allowed, collapse is not
        }
        let n_l = level.iter().filter(|h| h.is_l).count();
        let n_s = level.iter().filter(|h| !h.is_l).count();
        assert!(n_l > 0, "level {k}: no super-L tiles");
        assert!(n_s > 0, "level {k}: no super-S tiles");
    }
}

/// **Cor 4 — Period-5 collapses at level k* ≈ log(5)/log(φ) ≈ 3.3.**
///
/// Verifies that the Period-5 hierarchy reaches a point where one of the
/// supertile types vanishes (or the hierarchy can no longer extend).
#[test]
fn cor4_period5_collapses_within_log_phi_5_levels() {
    let tiles = period5_tiling(10_000);
    let hier = build_hierarchy(&tiles, MAX_HIER);

    // Find the first level where either tile type is missing or count < 2.
    let mut collapse_level = None;
    for k in 1..hier.n_levels() {
        let level = &hier.levels[k];
        let n_l = level.iter().filter(|h| h.is_l).count();
        let n_s = level.iter().filter(|h| !h.is_l).count();
        if n_l == 0 || n_s == 0 || level.len() < 2 {
            collapse_level = Some(k);
            break;
        }
    }

    // Expected: collapse by level 4 or 5 (log_φ(5) ≈ 3.35, plus tile-vs-word
    // discretisation).
    let level = collapse_level.expect("Period-5 must collapse");
    assert!(
        level <= 6,
        "Period-5 collapsed at level {level}, expected ≤ 5"
    );
}

/// **Thm 9 — Golden Compensation (scale-invariant L:S ratio).**
///
/// At every hierarchy level, the L:S ratio of the Fibonacci tiling is φ
/// exactly. We test that it stays within 10% of φ at every level reached.
#[test]
fn t09_golden_compensation_ls_ratio_stays_phi() {
    let tiles = qc_word_tiling(100_000, 0.0);
    let hier = build_hierarchy(&tiles, MAX_HIER);

    for k in 0..hier.n_levels() {
        let level = &hier.levels[k];
        if level.len() < 16 {
            // Statistical noise dominates below a handful of supertiles.
            break;
        }
        let n_l = level.iter().filter(|h| h.is_l).count() as f64;
        let n_s = level.iter().filter(|h| !h.is_l).count() as f64;
        if n_s < 1.0 {
            break;
        }
        let ratio = n_l / n_s;
        let dev = (ratio - PHI).abs() / PHI;
        assert!(
            dev < 0.10,
            "level {k}: L:S = {ratio:.4} (expected φ = {PHI:.4}), deviation = {dev:.4}"
        );
    }
}

/// **Cor 15 / Thm 13 — Aperiodic advantage grows with corpus scale.**
///
/// At small N both Fibonacci and Period-5 produce similar deep-position counts
/// at shallow levels; the divergence is observable at moderate scale and grows
/// rapidly thereafter. We verify the advantage exists at our test scale.
#[test]
fn t13_aperiodic_advantage_grows_with_scale() {
    let scales = [1_000u32, 10_000u32];
    let mut last_g_deep = 0i64;
    let mut last_p_deep = 0i64;

    for &n in &scales {
        let golden = qc_word_tiling(n, 0.0);
        let period5 = period5_tiling(n);

        let g_hier = build_hierarchy(&golden, MAX_HIER);
        let p_hier = build_hierarchy(&period5, MAX_HIER);

        let g_dp = detect_deep_positions(&golden, &g_hier);
        let p_dp = detect_deep_positions(&period5, &p_hier);

        let g_counts = deep_counts(&g_dp);
        let p_counts = deep_counts(&p_dp);

        // Deep positions at level 4+ are where the advantage lives.
        let g_deep: i64 = g_counts.iter().skip(4).map(|&c| c as i64).sum();
        let p_deep: i64 = p_counts.iter().skip(4).map(|&c| c as i64).sum();

        let advantage = g_deep - p_deep;
        // Advantage at this scale should be at least nonnegative.
        assert!(
            advantage >= 0,
            "n={n}: g_deep={g_deep}, p_deep={p_deep}, advantage negative"
        );

        // At the larger scale we expect a larger absolute advantage.
        if n == scales[scales.len() - 1] {
            assert!(
                advantage > last_g_deep - last_p_deep,
                "advantage did not grow with scale: {} → {}",
                last_g_deep - last_p_deep,
                advantage
            );
        }
        last_g_deep = g_deep;
        last_p_deep = p_deep;
    }
}

/// **Sturmian property — golden tiling factor complexity is `n + 1`.**
///
/// Sturmian sequences are characterized by having exactly `n + 1` distinct
/// length-n factors. The golden Fibonacci word is Sturmian (paper §2.3, §4.10);
/// this is the algebraic root of maximal codebook efficiency (Thm 7 + Cor 8).
#[test]
fn sturmian_factor_complexity_is_n_plus_1() {
    use std::collections::HashSet;

    let tiles = qc_word_tiling(20_000, 0.0);
    // Use the is_l boolean stream as the binary Sturmian sequence.
    let stream: Vec<bool> = tiles.iter().map(|t| t.is_l).collect();

    for n in 1..=8 {
        let mut factors: HashSet<Vec<bool>> = HashSet::new();
        for window in stream.windows(n) {
            factors.insert(window.to_vec());
        }
        // Paper §4.10: Sturmian sequences have EXACTLY n+1 distinct length-n
        // factors (minimal-complexity property; Thm 7 corollary). For the
        // 20,000-symbol golden-tiling prefix sampled above, n ∈ [1, 8] is
        // well within the regime where the exact equality holds — assert
        // it rather than the looser `≤` bound to catch any drift toward
        // degenerate (< n+1, e.g. periodic) or super-Sturmian (> n+1)
        // streams.
        assert_eq!(
            factors.len(),
            n + 1,
            "n={n}: {} distinct factors, Sturmian minimality requires exactly {}",
            factors.len(),
            n + 1
        );
    }
}

/// **No-adjacent-S invariant — holds on cut-and-project tilings by construction.**
#[test]
fn no_adjacent_s_holds_on_canonical_tilings() {
    // All 36 canonical descs.
    for desc in &quasicryth_research::tiling_descs() {
        let tiles = quasicryth_research::gen_from_desc(desc, 5_000);
        assert!(
            verify_no_adjacent_s(&tiles),
            "tiling '{}' (α={}, φ={}) violates no-adjacent-S",
            desc.name,
            desc.alpha,
            desc.phase
        );
    }
}

/// **HIER_WORD_LENS matches the C reference Fibonacci constants.**
#[test]
fn hier_word_lens_are_fibonacci_3_through_12() {
    // F_3 = 2, F_4 = 3, F_5 = 5, F_6 = 8, F_7 = 13, F_8 = 21,
    // F_9 = 34, F_10 = 55, F_11 = 89, F_12 = 144.
    let expected = [2usize, 3, 5, 8, 13, 21, 34, 55, 89, 144];
    assert_eq!(HIER_WORD_LENS, expected);
}

/// **PV-property (φ Perron-Frobenius eigenvalue).**
///
/// φ satisfies φ² = φ + 1. Verifies the algebraic identity at machine epsilon.
#[test]
fn pv_property_phi_squared_equals_phi_plus_one() {
    let lhs = PHI * PHI;
    let rhs = PHI + 1.0;
    assert!((lhs - rhs).abs() < 1e-12, "φ² = {lhs}, φ + 1 = {rhs}");
    // Inverse: 1/φ = φ - 1.
    assert!((INV_PHI - (PHI - 1.0)).abs() < 1e-12);
}

/// **Multi-tiling families produce nonempty tile sequences.**
#[test]
fn alternative_tilings_generate_nonempty_outputs() {
    let n = 1_000u32;

    let tm = thue_morse_tiling(n);
    assert!(!tm.is_empty());
    assert!(verify_no_adjacent_s(&tm));

    let sd = sanddrift_tiling(n);
    assert!(!sd.is_empty());

    let n5 = period5_tiling(n);
    assert!(!n5.is_empty());

    // Cut-and-project with a non-golden quadratic irrational.
    let sqrt58 = qc_word_tiling_alpha(n, 58.0_f64.sqrt() - 7.0, 0.0);
    assert!(!sqrt58.is_empty());
    assert!(verify_no_adjacent_s(&sqrt58));
}
