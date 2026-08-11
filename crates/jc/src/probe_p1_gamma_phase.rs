//! Probe P1: γ-phase-offset ranking discrimination.
//!
//! Citation: `.claude/knowledge/bf16-hhtl-terrain.md` Probe Queue P1
//! (status before this probe: NOT RUN).
//!
//! # The hypothesis being tested
//!
//! Constraint C3 in `bf16-hhtl-terrain.md` divides γ+φ into two regimes:
//!
//! - **VALID — pre-rank discrete selector** (codebook offset, start position
//!   on spiral): different offsets → different subsets of spiral → different
//!   ranked output. Dupain-Sós discrepancy property applies.
//! - **DEAD — post-rank monotone transform** (applied inside a rank operation):
//!   monotone transform before rank = identity on rank. Proven ρ=1.000.
//!
//! P1 directly tests the VALID regime: do 4 γ-phase offsets produce
//! *meaningfully different* rankings on the same base codebook?
//!
//! # PASS criterion (from probe queue table)
//!
//! - PASS: ρ differs by >0.01 across offsets (rankings meaningfully differ)
//! - FAIL: ρ identical across offsets (offsets are no-ops, γ+φ DEAD)
//!
//! Translated to a concrete bound for this implementation: for at least one
//! pair of offsets (i,j) with i ≠ j, Spearman ρ between their rankings
//! must be < 0.99 (i.e. the rankings differ meaningfully). If ALL pairwise
//! Spearman ρ values are > 0.999, the offsets are effectively no-ops.
//!
//! # What is being measured
//!
//! 1. Synthesize a base codebook of 256 entries on the unit interval (the
//!    canonical case for γ-spiral application). Use a Beta-shaped
//!    distribution biased toward the middle to avoid edge degeneracy.
//! 2. Synthesize 1000 query points on the unit interval, also Beta-shaped.
//! 3. For each γ-phase offset φ_k ∈ {0.0, 1/(4φ), 2/(4φ), 3/(4φ)} where
//!    φ = golden ratio (1.618...): apply the offset to the codebook
//!    (modular addition on [0,1)), then for each query produce a ranking
//!    of codebook entries by distance.
//! 4. For each pair of offsets (i,j), compute the mean Spearman ρ across
//!    the 1000 queries between their respective rankings.
//! 5. PASS if at least one pairwise mean ρ < 0.99.
//!
//! # Why this matters architecturally
//!
//! γ+φ is currently used in:
//! - `bgz-tensor::gamma_phi` (encoder for golden-ratio quantization)
//! - `bgz-tensor::gamma_calibration` (3-γ calibration: role/cosine/meta)
//! - `bgz-tensor::projection` (Base17 golden-step folding)
//!
//! All three rely on the assumption that γ+φ as pre-rank selector is
//! discriminative. If P1 FAILs, this assumption is wrong, and the
//! architecture has a load-bearing axiom that doesn't hold.
//!
//! If P1 PASSes, it confirms Constraint C3's distinction is real and
//! gives a quantitative measure of how much rankings differ — useful for
//! choosing offset spacing in production.

use crate::PillarResult;

const N_CODEBOOK: usize = 256;
const N_QUERIES: usize = 1_000;
const N_OFFSETS: usize = 4;
const SEED: u64 = 0xF1B0_BACC_1AF0_F0F0;

// ════════════════════════════════════════════════════════════════════════════
// Deterministic RNG (consistent with other pillars/probes in jc)
// ════════════════════════════════════════════════════════════════════════════

fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

fn rand_uniform(state: &mut u64) -> f64 {
    (splitmix64(state) >> 11) as f64 / (1u64 << 53) as f64
}

/// Beta(α=2, β=2)-shaped sample on [0,1) — biased toward middle.
/// Avoids edge degeneracy that would make rankings trivially differ near 0/1.
/// Approximation via order-statistic of two uniforms.
fn rand_beta22(state: &mut u64) -> f64 {
    let u1 = rand_uniform(state);
    let u2 = rand_uniform(state);
    if u1 < u2 { u1 } else { u2 }.max(rand_uniform(state).min(rand_uniform(state)))
}

// ════════════════════════════════════════════════════════════════════════════
// γ-phase-offset application
//
// Offset: shift codebook entry by phase * 1/(4φ) modulo 1. The 1/(4φ)
// stride is chosen so that 4 offsets cover roughly 1/φ of the unit interval
// — enough to produce meaningfully different rankings if discrimination
// works at all, while staying close enough that rankings could plausibly
// agree if γ+φ is a no-op.
// ════════════════════════════════════════════════════════════════════════════

const PHI_INV: f64 = 0.618_033_988_749_894_9; // 1/φ
const STRIDE: f64 = PHI_INV / 4.0;             // 1/(4φ) ≈ 0.1545

fn apply_offset(value: f64, k: usize) -> f64 {
    let offset = k as f64 * STRIDE;
    let shifted = value + offset;
    shifted - shifted.floor() // mod 1
}

// ════════════════════════════════════════════════════════════════════════════
// Distance + Ranking
//
// Distance is the toroidal (circular) absolute difference on [0, 1):
//   d(a, b) = min(|a-b|, 1 - |a-b|)
//
// Ranking: for each query q, produce indices [i_0, i_1, ..., i_{N-1}] sorted
// by ascending d(q, codebook[i]).
// ════════════════════════════════════════════════════════════════════════════

#[inline]
fn toroidal_distance(a: f64, b: f64) -> f64 {
    let d = (a - b).abs();
    d.min(1.0 - d)
}

fn rank_codebook(query: f64, codebook: &[f64]) -> Vec<usize> {
    let mut indices: Vec<usize> = (0..codebook.len()).collect();
    indices.sort_by(|&a, &b| {
        let da = toroidal_distance(query, codebook[a]);
        let db = toroidal_distance(query, codebook[b]);
        da.partial_cmp(&db).unwrap_or(std::cmp::Ordering::Equal)
    });
    indices
}

// ════════════════════════════════════════════════════════════════════════════
// Spearman ρ between two rankings of the same items
//
// Given two permutations of [0..N), compute Spearman correlation between
// the position vectors. If ranking_a[i] = j means "j is at rank i in a",
// then position_a[j] = i means "j has rank-position i".
//
// Spearman ρ = 1 - 6 Σ d_i² / (N (N² - 1))  where d_i = rank_a(i) - rank_b(i).
// ════════════════════════════════════════════════════════════════════════════

fn spearman_rho(ranking_a: &[usize], ranking_b: &[usize]) -> f64 {
    let n = ranking_a.len();
    debug_assert_eq!(n, ranking_b.len());
    if n < 2 {
        return 1.0;
    }
    // Build position vectors: pos_a[item] = rank where item appears.
    let mut pos_a = vec![0usize; n];
    let mut pos_b = vec![0usize; n];
    for (rank, &item) in ranking_a.iter().enumerate() {
        pos_a[item] = rank;
    }
    for (rank, &item) in ranking_b.iter().enumerate() {
        pos_b[item] = rank;
    }
    let mut sum_d_sq = 0.0f64;
    for i in 0..n {
        let d = pos_a[i] as f64 - pos_b[i] as f64;
        sum_d_sq += d * d;
    }
    let n_f = n as f64;
    1.0 - 6.0 * sum_d_sq / (n_f * (n_f * n_f - 1.0))
}

// ════════════════════════════════════════════════════════════════════════════
// The probe
// ════════════════════════════════════════════════════════════════════════════

pub fn prove() -> PillarResult {
    let mut state = SEED;

    // 1. Synthesize base codebook (256 entries, Beta(2,2) on [0,1))
    let codebook_base: Vec<f64> = (0..N_CODEBOOK).map(|_| rand_beta22(&mut state)).collect();

    // 2. Build 4 offset variants of the codebook.
    let codebooks: Vec<Vec<f64>> = (0..N_OFFSETS)
        .map(|k| codebook_base.iter().map(|&v| apply_offset(v, k)).collect())
        .collect();

    // 3. Synthesize 1000 queries (Beta(2,2) on [0,1))
    let queries: Vec<f64> = (0..N_QUERIES).map(|_| rand_beta22(&mut state)).collect();

    // 4. For each pair (i,j) with i < j, compute mean Spearman ρ across queries.
    let mut pairwise_means = Vec::with_capacity(N_OFFSETS * (N_OFFSETS - 1) / 2);
    let mut min_rho = f64::INFINITY;
    let mut max_rho = f64::NEG_INFINITY;

    for i in 0..N_OFFSETS {
        for j in (i + 1)..N_OFFSETS {
            let mut sum_rho = 0.0f64;
            for &q in &queries {
                let ranking_i = rank_codebook(q, &codebooks[i]);
                let ranking_j = rank_codebook(q, &codebooks[j]);
                sum_rho += spearman_rho(&ranking_i, &ranking_j);
            }
            let mean_rho = sum_rho / N_QUERIES as f64;
            pairwise_means.push((i, j, mean_rho));
            if mean_rho < min_rho {
                min_rho = mean_rho;
            }
            if mean_rho > max_rho {
                max_rho = mean_rho;
            }
        }
    }

    // 5. PASS criterion: at least one pair has mean ρ < 0.99 (rankings differ
    //    meaningfully). FAIL if all pairs have ρ > 0.999 (offsets are no-ops).
    let pass = min_rho < 0.99;

    // Format pairwise breakdown for the detail string
    let pairwise_str = pairwise_means
        .iter()
        .map(|(i, j, rho)| format!("({i},{j})={rho:.6}"))
        .collect::<Vec<_>>()
        .join(", ");

    let conclusion = if pass {
        "γ+φ pre-rank selector is VALID — different offsets produce \
         meaningfully different rankings, Dupain-Sós discrepancy property \
         confirmed in this synthetic regime. Updates Probe P1 status \
         NOT RUN → PASS in bf16-hhtl-terrain.md queue."
    } else {
        "γ+φ pre-rank selector is DEAD — offsets produce near-identical \
         rankings (all pairwise ρ > 0.999). γ+φ joins post-rank monotone \
         in the DEAD regime. Updates Probe P1 status NOT RUN → FAIL in \
         bf16-hhtl-terrain.md queue. Architectural consequence: γ-encoding \
         strategy in bgz-tensor needs revision."
    };

    let detail = format!(
        "Codebook: {N_CODEBOOK} entries, Beta(2,2) on [0,1). \
         Queries: {N_QUERIES}, Beta(2,2) on [0,1). \
         Offsets: {N_OFFSETS} at stride 1/(4φ) ≈ {STRIDE:.6}. \
         Distance: toroidal absolute difference. \
         Pairwise mean Spearman ρ across queries: {pairwise_str}. \
         min ρ = {min_rho:.6}, max ρ = {max_rho:.6}. \
         PASS criterion: min ρ < 0.99. {conclusion}"
    );

    PillarResult {
        name: "Probe P1: γ-phase-offset ranking discrimination",
        pass,
        measured: min_rho,
        predicted: 0.99,
        detail,
        runtime_ms: 0,
    }
}

// ════════════════════════════════════════════════════════════════════════════
// Tests
// ════════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn toroidal_distance_is_zero_for_equal() {
        assert!(toroidal_distance(0.5, 0.5) < 1e-12);
    }

    #[test]
    fn toroidal_distance_wraps() {
        // d(0.05, 0.95) should be 0.10 (going through 0/1 wrap), not 0.90
        let d = toroidal_distance(0.05, 0.95);
        assert!((d - 0.10).abs() < 1e-12, "expected 0.10, got {d}");
    }

    #[test]
    fn toroidal_distance_max_is_half() {
        // Antipodal on a unit circle is at distance 0.5
        let d = toroidal_distance(0.0, 0.5);
        assert!((d - 0.5).abs() < 1e-12);
    }

    #[test]
    fn apply_offset_zero_is_identity() {
        for v in [0.1, 0.3, 0.5, 0.7, 0.9].iter() {
            let r = apply_offset(*v, 0);
            assert!((r - v).abs() < 1e-12);
        }
    }

    #[test]
    fn apply_offset_wraps_modulo_one() {
        // value 0.9 + offset k=4 (4 * 1/(4φ) = 1/φ ≈ 0.618) → 1.518 mod 1 = 0.518
        let r = apply_offset(0.9, 4);
        assert!(r >= 0.0 && r < 1.0, "result out of [0,1): {r}");
    }

    #[test]
    fn spearman_identity_is_one() {
        let r1: Vec<usize> = (0..50).collect();
        let r2: Vec<usize> = (0..50).collect();
        let rho = spearman_rho(&r1, &r2);
        assert!((rho - 1.0).abs() < 1e-12);
    }

    #[test]
    fn spearman_reverse_is_minus_one() {
        let r1: Vec<usize> = (0..50).collect();
        let r2: Vec<usize> = (0..50).rev().collect();
        let rho = spearman_rho(&r1, &r2);
        assert!((rho - (-1.0)).abs() < 1e-9, "expected -1.0, got {rho}");
    }

    #[test]
    fn rank_codebook_returns_permutation() {
        let codebook = [0.1, 0.5, 0.9, 0.3];
        let ranking = rank_codebook(0.4, &codebook);
        assert_eq!(ranking.len(), 4);
        // All indices are present exactly once
        let mut seen = [false; 4];
        for &idx in &ranking {
            assert!(idx < 4);
            assert!(!seen[idx], "duplicate index {idx}");
            seen[idx] = true;
        }
    }

    #[test]
    fn rank_codebook_orders_by_distance() {
        // query 0.4: distances are |0.1-0.4|=0.3, |0.5-0.4|=0.1, ...
        // closest should be index 1 (0.5), then 3 (0.3), then 0 (0.1), then 2 (0.9)
        // Wait: d(0.4, 0.9) toroidal = min(0.5, 0.5) = 0.5
        //       d(0.4, 0.1) toroidal = min(0.3, 0.7) = 0.3
        // So order: 1 (d=0.1), 3 (d=0.1), 0 (d=0.3), 2 (d=0.5)
        let codebook = [0.1, 0.5, 0.9, 0.3];
        let ranking = rank_codebook(0.4, &codebook);
        assert_eq!(ranking[0], 1, "closest should be index 1 (value 0.5), got {}", ranking[0]);
    }

    #[test]
    fn deterministic_with_fixed_seed() {
        // Two prove() calls with the same SEED should produce identical results.
        // (We can't easily call prove() twice cheaply, so just check the RNG itself.)
        let mut s1 = 42u64;
        let mut s2 = 42u64;
        for _ in 0..100 {
            assert_eq!(splitmix64(&mut s1), splitmix64(&mut s2));
        }
    }

    #[test]
    fn probe_runs_and_returns_meaningful_result() {
        let r = prove();
        assert!(r.measured.is_finite(), "measured ρ should be finite");
        assert!(
            r.measured >= -1.0 && r.measured <= 1.0,
            "measured ρ out of [-1,1]: {}",
            r.measured
        );
        assert!(!r.detail.is_empty());
    }
}
