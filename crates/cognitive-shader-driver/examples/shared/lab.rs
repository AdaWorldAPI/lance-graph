//! D-SELF-CALIBRATING-LAB-0 — the statistics every lab probe reports through.
//!
//! A task probe runs a strategy on matched blocks (seeds, puzzles) and hands
//! this module one outcome per (arm, block). Nothing here knows the task.
//!
//! - **Paired comparison** on matched blocks: the per-block difference, an
//!   exact sign-flip p-value (no distributional assumption; exhaustive up to
//!   20 blocks), and a t-interval (which does assume roughly normal block
//!   differences, and says so).
//! - **Holm** correction across a pre-registered family.
//! - **Verdict** per pre-registration: SUPPORTED needs the Holm rejection,
//!   the declared direction and a mean effect of at least the declared
//!   minimum. FALSIFIED needs the interval to rule the minimum effect out.
//!   Everything else is INCONCLUSIVE.
//! - **Reliability** of predicted probabilities: bins, Brier score, ECE.
//!
//! Every result carries the kind of certificate it is. A statistical verdict
//! is never a deterministic guarantee, and an empirical number is neither.
#![allow(dead_code)]

/// What kind of claim a result is.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CertKind {
    /// Proved bound; holds for every input within its assumptions.
    Deterministic,
    /// Probability statement under stated assumptions (test, interval).
    Statistical,
    /// A measured number, no guarantee.
    Empirical,
}

/// A hypothesis, fixed before the confirmatory run.
#[derive(Debug, Clone, Copy)]
pub struct Prereg {
    pub id: &'static str,
    pub hypothesis: &'static str,
    pub baseline: &'static str,
    pub treatment: &'static str,
    pub endpoint: &'static str,
    /// Larger endpoint values are better.
    pub higher_is_better: bool,
    /// Smallest effect worth acting on, in endpoint units.
    pub min_effect: f64,
    /// Number of matched blocks the run must have.
    pub blocks: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Verdict {
    Supported,
    Falsified,
    Inconclusive,
}

/// Paired comparison of treatment against baseline on matched blocks.
#[derive(Debug, Clone, Copy)]
pub struct Paired {
    pub n: usize,
    /// Mean of (treatment − baseline), signed so that positive = better.
    pub mean: f64,
    pub sd: f64,
    /// Exact two-sided sign-flip p-value.
    pub p: f64,
    /// 95 % t-interval for `mean`.
    pub lo: f64,
    pub hi: f64,
}

/// Two-sided 97.5 % t quantiles for df 1..=30; df > 30 uses 1.96.
const T975: [f64; 30] = [
    12.706, 4.303, 3.182, 2.776, 2.571, 2.447, 2.365, 2.306, 2.262, 2.228, 2.201, 2.179, 2.160,
    2.145, 2.131, 2.120, 2.110, 2.101, 2.093, 2.086, 2.080, 2.074, 2.069, 2.064, 2.060, 2.056,
    2.052, 2.048, 2.045, 2.042,
];

/// Exact two-sided sign-flip p-value of the mean of `d` (n ≤ 20).
pub fn sign_flip_p(d: &[f64]) -> f64 {
    assert!(
        d.len() <= 20,
        "exhaustive sign flip is limited to 20 blocks"
    );
    let obs = d.iter().sum::<f64>().abs();
    let n = d.len();
    let mut extreme = 0u64;
    for mask in 0u64..(1 << n) {
        let s: f64 = d
            .iter()
            .enumerate()
            .map(|(i, &x)| if mask >> i & 1 == 1 { -x } else { x })
            .sum();
        if s.abs() >= obs - 1e-9 {
            extreme += 1;
        }
    }
    extreme as f64 / (1u64 << n) as f64
}

/// Compare `treatment` with `baseline`, block for block.
pub fn paired(baseline: &[f64], treatment: &[f64], higher_is_better: bool) -> Paired {
    assert_eq!(baseline.len(), treatment.len(), "blocks must be matched");
    let sign = if higher_is_better { 1.0 } else { -1.0 };
    let d: Vec<f64> = baseline
        .iter()
        .zip(treatment)
        .map(|(b, t)| sign * (t - b))
        .collect();
    let n = d.len();
    let mean = d.iter().sum::<f64>() / n as f64;
    let var = if n > 1 {
        d.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (n - 1) as f64
    } else {
        0.0
    };
    let sd = var.sqrt();
    let t = if n >= 2 && n - 1 <= 30 {
        T975[n - 2]
    } else {
        1.96
    };
    let half = t * sd / (n as f64).sqrt();
    Paired {
        n,
        mean,
        sd,
        p: sign_flip_p(&d),
        lo: mean - half,
        hi: mean + half,
    }
}

/// Holm step-down over a family. Returns which hypotheses are rejected.
pub fn holm(p: &[f64], alpha: f64) -> Vec<bool> {
    let m = p.len();
    let mut order: Vec<usize> = (0..m).collect();
    order.sort_by(|&a, &b| p[a].total_cmp(&p[b]));
    let mut out = vec![false; m];
    for (rank, &i) in order.iter().enumerate() {
        if p[i] <= alpha / (m - rank) as f64 {
            out[i] = true;
        } else {
            break;
        }
    }
    out
}

/// The verdict a pre-registration earns from its paired result and its
/// Holm decision.
pub fn verdict(pr: &Prereg, r: &Paired, rejected: bool) -> Verdict {
    if r.n != pr.blocks {
        return Verdict::Inconclusive;
    }
    if rejected && r.mean >= pr.min_effect {
        Verdict::Supported
    } else if r.hi < pr.min_effect {
        Verdict::Falsified
    } else {
        Verdict::Inconclusive
    }
}

/// One reliability bin: predictions in `[lo, hi)`.
#[derive(Debug, Clone, Copy)]
pub struct Bin {
    pub lo: f64,
    pub hi: f64,
    pub n: u64,
    pub mean_pred: f64,
    pub observed: f64,
}

/// Reliability from pre-binned counts `(n, Σ pred, hits)` over equal bins
/// on [0, 1]. Returns the bins, the expected calibration error, and the
/// Brier score lower bound computable from bins (exact Brier needs the
/// per-prediction values; see `brier`).
pub fn reliability(binned: &[(u64, f64, u64)]) -> (Vec<Bin>, f64) {
    let k = binned.len();
    let total: u64 = binned.iter().map(|b| b.0).sum();
    let mut out = Vec::with_capacity(k);
    let mut ece = 0.0;
    for (i, &(n, sp, hits)) in binned.iter().enumerate() {
        let (mean_pred, observed) = if n == 0 {
            (0.0, 0.0)
        } else {
            (sp / n as f64, hits as f64 / n as f64)
        };
        if n > 0 && total > 0 {
            ece += n as f64 / total as f64 * (mean_pred - observed).abs();
        }
        out.push(Bin {
            lo: i as f64 / k as f64,
            hi: (i + 1) as f64 / k as f64,
            n,
            mean_pred,
            observed,
        });
    }
    (out, ece)
}

/// Brier score of `(prediction, outcome)` pairs.
pub fn brier(pairs: &[(f64, bool)]) -> f64 {
    pairs
        .iter()
        .map(|&(p, y)| (p - if y { 1.0 } else { 0.0 }).powi(2))
        .sum::<f64>()
        / pairs.len().max(1) as f64
}

/// One printed line per pre-registration.
pub fn report(pr: &Prereg, r: &Paired, rejected: bool) -> String {
    format!(
        "{:<6} {:<34} {} vs {}: mean {:+.4} [{:+.4}, {:+.4}]  p {:.4}  holm {}  -> {:?} (statistical)",
        pr.id,
        pr.endpoint,
        pr.treatment,
        pr.baseline,
        r.mean,
        r.lo,
        r.hi,
        r.p,
        if rejected { "reject" } else { "keep" },
        verdict(pr, r, rejected)
    )
}

#[cfg(test)]
mod lab_tests {
    use super::*;

    #[test]
    fn sign_flip_p_is_exact_on_a_small_case() {
        // All four diffs positive: only the all-plus and all-minus flips are
        // as extreme, 2 of 16.
        assert!((sign_flip_p(&[1.0, 2.0, 3.0, 4.0]) - 2.0 / 16.0).abs() < 1e-12);
        // A symmetric set has p = 1.
        assert!((sign_flip_p(&[1.0, -1.0]) - 1.0).abs() < 1e-12);
    }

    #[test]
    fn holm_matches_the_textbook_step_down() {
        // m = 4, alpha 0.05: thresholds 0.0125, 0.0167, 0.025, 0.05.
        assert_eq!(
            holm(&[0.010, 0.040, 0.030, 0.005], 0.05),
            vec![true, false, false, true]
        );
        // Both pass their step thresholds (0.025, then 0.05).
        assert_eq!(holm(&[0.03, 0.001], 0.05), vec![true, true]);
        // Stops at the first failure: 0.03 > 0.025 halts, so 0.04 is kept
        // even though it is below its own threshold 0.05.
        assert_eq!(holm(&[0.001, 0.03, 0.04], 0.05), vec![true, false, false]);
        assert_eq!(holm(&[0.06, 0.001], 0.05), vec![false, true]);
    }

    #[test]
    fn a_planted_effect_is_found_and_a_null_is_not() {
        let base: Vec<f64> = (0..12).map(|i| 0.5 + 0.01 * (i % 3) as f64).collect();
        let better: Vec<f64> = base.iter().map(|x| x + 0.05).collect();
        let r = paired(&base, &better, true);
        assert!(r.p < 0.001 && r.mean > 0.049);
        let pr = Prereg {
            id: "T",
            hypothesis: "",
            baseline: "b",
            treatment: "t",
            endpoint: "e",
            higher_is_better: true,
            min_effect: 0.02,
            blocks: 12,
        };
        assert_eq!(verdict(&pr, &r, true), Verdict::Supported);
        // Null: alternating ± noise, no effect.
        let noisy: Vec<f64> = base
            .iter()
            .enumerate()
            .map(|(i, x)| x + if i % 2 == 0 { 0.01 } else { -0.01 })
            .collect();
        let r0 = paired(&base, &noisy, true);
        assert!(r0.p > 0.5);
        assert_eq!(
            verdict(&pr, &r0, holm(&[r0.p], 0.05)[0]),
            Verdict::Falsified
        );
        // Direction matters: "lower is better" flips the sign.
        assert!(paired(&base, &better, false).mean < 0.0);
        // Significant but below the minimum effect: never SUPPORTED. A tiny
        // shift on every block is rejected by Holm, and its interval sits
        // under 0.02, so the verdict is FALSIFIED.
        let tiny: Vec<f64> = base.iter().map(|x| x + 0.005).collect();
        let rt = paired(&base, &tiny, true);
        assert!(rt.p < 0.001);
        assert_eq!(verdict(&pr, &rt, true), Verdict::Falsified);
        // A run with the wrong block count is never SUPPORTED.
        let short = paired(&base[..6], &better[..6], true);
        assert_eq!(verdict(&pr, &short, true), Verdict::Inconclusive);
    }

    #[test]
    fn reliability_sees_calibration_and_miscalibration() {
        // Perfectly calibrated: bin with mean 0.25 hits 25 %.
        let (_, ece) = reliability(&[(0, 0.0, 0), (0, 0.0, 0), (4, 1.0, 1), (0, 0.0, 0)]);
        assert!(ece < 1e-12);
        // Overconfident: predicts 0.9, hits 50 %.
        let mut bins = [(0u64, 0.0f64, 0u64); 10];
        bins[9] = (10, 9.0, 5);
        let (_, ece) = reliability(&bins);
        assert!((ece - 0.4).abs() < 1e-12);
        assert!((brier(&[(1.0, true), (0.0, false)])).abs() < 1e-12);
        assert!((brier(&[(1.0, false)]) - 1.0).abs() < 1e-12);
    }
}
