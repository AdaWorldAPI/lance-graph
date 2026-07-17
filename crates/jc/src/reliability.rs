//! Reliability & validity metric battery — the four measures the workspace
//! names for construct agreement: **Pearson r**, **Spearman ρ**, **Cronbach α**,
//! and the **intraclass correlation coefficient (ICC)**.
//!
//! # Why this lives in `jc`
//!
//! `jc` is otherwise a proof-in-code harness (the registered pillars). But the
//! workspace repeatedly asks the SAME empirical question — "do two
//! representations measure the same construct, reliably?" — and until now each
//! caller rolled its own (e.g. a private `spearman_rho` on rankings in
//! `probe_p1_gamma_phase.rs`, and a hand-rolled `spearman_rho` in the
//! stockfish-rs probe). The D-TRI-2/4/5 gates (12-family vs 12-step agreement;
//! chess↔thinking transfer; emulation vs resonance) require all four as
//! callable functions, so they are consolidated here once.
//!
//! # The four measures
//!
//! - **Pearson r** — linear product-moment correlation. Sensitive to the exact
//!   values; use when the two vectors are interval-scaled and a linear relation
//!   is the hypothesis.
//! - **Spearman ρ** — Pearson on the *ranks* (average-rank tie correction).
//!   Monotone-but-nonlinear relations and ordinal data; robust to outliers.
//! - **Cronbach α** — internal-consistency reliability of a `k`-item scale over
//!   `n` subjects. `α = (k/(k-1))·(1 − Σσ²_item / σ²_total)`. "Do these `k`
//!   columns behave as one coherent scale?"
//! - **ICC** — agreement/consistency of `k` raters over `n` subjects via the
//!   two-way ANOVA decomposition (Shrout & Fleiss 1979). Two single-measure
//!   forms are provided: `Icc2_1` (two-way random, **absolute agreement**) and
//!   `Icc3_1` (two-way mixed, **consistency**). Unlike α, ICC penalises
//!   systematic rater bias (absolute-agreement form).
//!
//! # Statistical-significance note (I-NOISE-FLOOR-JIRAK)
//!
//! Per the workspace iron rule, when any of these metrics is used to claim
//! "observed value is N σ above the noise floor" on the 16384-bit fingerprints,
//! the significance MUST be calibrated with **Jirak 2016** (weak dependence),
//! NOT classical IID Berry-Esseen — the bits are weakly dependent by
//! construction. These functions compute the point estimates; the significance
//! calibration is `crate::jirak`'s job. See `crate::jirak` and the iron rule in
//! `lance-graph/CLAUDE.md`.
//!
//! # Citations
//!
//! - C. Spearman, "The proof and measurement of association between two
//!   things", Am. J. Psychol. 15 (1904).
//! - L. J. Cronbach, "Coefficient alpha and the internal structure of tests",
//!   Psychometrika 16 (1951).
//! - P. E. Shrout & J. L. Fleiss, "Intraclass correlations: uses in assessing
//!   rater reliability", Psychological Bulletin 86(2) (1979), 420–428. The
//!   `icc` forms and the worked example in the tests are from this paper.
//!
//! All functions return `Option<f64>`, yielding `None` on degenerate input
//! (too few observations, zero variance, ragged matrices) rather than panicking
//! or returning `NaN` — the caller decides how to treat an undefined estimate.

/// Arithmetic mean of a slice, or `None` if empty.
#[inline]
fn mean(xs: &[f64]) -> Option<f64> {
    if xs.is_empty() {
        return None;
    }
    Some(xs.iter().sum::<f64>() / xs.len() as f64)
}

/// Whether every element is finite (no `NaN`, no `±∞`). The no-`NaN` API
/// contract is enforced by rejecting non-finite INPUTS up front: a `NaN` sorts
/// as "equal" in the rank step and would otherwise receive an ordinary rank,
/// silently producing a finite-but-garbage Spearman ρ (and `Some(NaN)` for the
/// other three). Every public metric guards on this before computing.
#[inline]
fn all_finite(xs: &[f64]) -> bool {
    xs.iter().all(|v| v.is_finite())
}

/// Pearson product-moment correlation coefficient of two equal-length series.
///
/// Returns `None` if the series differ in length, have fewer than 2 elements,
/// or either has zero variance (the coefficient is undefined — no linear
/// relation can be measured against a constant).
///
/// ```
/// use jc::reliability::pearson;
/// assert!((pearson(&[1.0, 2.0, 3.0], &[1.0, 2.0, 3.0]).unwrap() - 1.0).abs() < 1e-12);
/// assert!((pearson(&[1.0, 2.0, 3.0], &[3.0, 2.0, 1.0]).unwrap() + 1.0).abs() < 1e-12);
/// ```
pub fn pearson(x: &[f64], y: &[f64]) -> Option<f64> {
    if x.len() != y.len() || x.len() < 2 {
        return None;
    }
    if !all_finite(x) || !all_finite(y) {
        return None; // NaN / ±∞ input → undefined (no-NaN contract)
    }
    let mx = mean(x)?;
    let my = mean(y)?;
    let mut sxy = 0.0;
    let mut sxx = 0.0;
    let mut syy = 0.0;
    for (&xi, &yi) in x.iter().zip(y.iter()) {
        let dx = xi - mx;
        let dy = yi - my;
        sxy += dx * dy;
        sxx += dx * dx;
        syy += dy * dy;
    }
    let denom = (sxx * syy).sqrt();
    if denom == 0.0 || !denom.is_finite() {
        // `denom == 0` → at least one series is constant. `denom == ∞` → the
        // squared-deviation product overflowed on large finite input; then
        // `sxy / ∞ = 0.0` is FINITE-but-wrong, so the trailing `is_finite`
        // guard would miss it (e.g. `pearson(&[1e100,-1e100],&[1e100,-1e100])`
        // is perfectly correlated but the ratio collapses to 0.0). Reject here.
        return None;
    }
    let r = sxy / denom;
    // With a finite non-zero denom, `sxy` can still be ±∞/NaN on overflow —
    // reject a non-finite ratio too.
    r.is_finite().then_some(r)
}

/// Average (fractional) ranks of a series, tie-corrected: tied values receive
/// the mean of the ranks they span. Ranks are 1-based, but any affine shift
/// cancels in the subsequent Pearson step, so the base is irrelevant.
fn average_ranks(xs: &[f64]) -> Vec<f64> {
    let n = xs.len();
    let mut idx: Vec<usize> = (0..n).collect();
    // NaN-safe ordering: partial_cmp, treating equal/incomparable as Equal so
    // ties group together (NaN inputs are the caller's contract to avoid).
    idx.sort_by(|&a, &b| {
        xs[a]
            .partial_cmp(&xs[b])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let mut ranks = vec![0.0; n];
    let mut i = 0;
    while i < n {
        let mut j = i + 1;
        // Extend the tie group while values are equal.
        while j < n && xs[idx[j]] == xs[idx[i]] {
            j += 1;
        }
        // Ranks i..j (0-based) → average of the 1-based ranks (i+1 .. j).
        let avg = ((i + 1 + j) as f64) / 2.0; // mean of i+1 ..= j
        for &k in &idx[i..j] {
            ranks[k] = avg;
        }
        i = j;
    }
    ranks
}

/// Spearman rank correlation coefficient — Pearson `r` computed on the
/// average-rank transform of each series (so ties are handled correctly).
///
/// Returns `None` under the same degeneracy conditions as [`pearson`] (e.g. all
/// values in a series tied → zero rank variance → undefined).
///
/// ```
/// use jc::reliability::spearman;
/// // Monotone but nonlinear → ρ = 1 even though Pearson r < 1.
/// let x = [1.0, 2.0, 3.0, 4.0, 5.0];
/// let y = [1.0, 4.0, 9.0, 16.0, 25.0];
/// assert!((spearman(&x, &y).unwrap() - 1.0).abs() < 1e-12);
/// ```
pub fn spearman(x: &[f64], y: &[f64]) -> Option<f64> {
    if x.len() != y.len() || x.len() < 2 {
        return None;
    }
    if !all_finite(x) || !all_finite(y) {
        // Guard here, not just in the delegated `pearson`: `average_ranks`
        // maps a NaN to an ordinary finite rank (partial_cmp → Equal), so the
        // ranks handed to `pearson` are all finite and the NaN would slip
        // through as a finite-but-garbage ρ. Reject non-finite input up front.
        return None;
    }
    let rx = average_ranks(x);
    let ry = average_ranks(y);
    pearson(&rx, &ry)
}

/// Population variance (divisor `n`) of a slice, or `None` if empty.
fn pop_var(xs: &[f64]) -> Option<f64> {
    let m = mean(xs)?;
    let n = xs.len() as f64;
    Some(xs.iter().map(|&v| (v - m) * (v - m)).sum::<f64>() / n)
}

/// Cronbach's α — internal-consistency reliability of a `k`-item scale.
///
/// `items` is a `k`-length slice, each element a length-`n` vector of that
/// item's score across all `n` subjects (i.e. `items[i][s]` = item `i`, subject
/// `s`). All inner vectors must share length `n ≥ 1`, and `k ≥ 2`.
///
/// `α = (k / (k-1)) · (1 − Σ_i σ²_i / σ²_total)`, where `σ²_i` is the variance
/// of item `i` and `σ²_total` is the variance of the per-subject sums. Returns
/// `None` on ragged input, `k < 2`, or `σ²_total == 0` (no between-subject
/// variance → α undefined). α can be negative for anti-correlated items; that
/// is a real (if pathological) value and is returned as-is.
///
/// ```
/// use jc::reliability::cronbach_alpha;
/// // Three identical items → perfect internal consistency, α = 1.
/// let items = vec![
///     vec![1.0, 2.0, 3.0, 4.0, 5.0],
///     vec![1.0, 2.0, 3.0, 4.0, 5.0],
///     vec![1.0, 2.0, 3.0, 4.0, 5.0],
/// ];
/// assert!((cronbach_alpha(&items).unwrap() - 1.0).abs() < 1e-12);
/// ```
pub fn cronbach_alpha(items: &[Vec<f64>]) -> Option<f64> {
    let k = items.len();
    if k < 2 {
        return None;
    }
    let n = items[0].len();
    if n == 0 || items.iter().any(|it| it.len() != n) {
        return None;
    }
    if items.iter().any(|it| !all_finite(it)) {
        return None; // NaN / ±∞ input → undefined (no-NaN contract)
    }
    let sum_item_var: f64 = items.iter().filter_map(|it| pop_var(it)).sum();
    if items.iter().any(|it| pop_var(it).is_none()) {
        return None;
    }
    // Per-subject total across items.
    let totals: Vec<f64> = (0..n)
        .map(|s| items.iter().map(|it| it[s]).sum::<f64>())
        .collect();
    let total_var = pop_var(&totals)?;
    if total_var == 0.0 || !total_var.is_finite() {
        // 0 → no between-subject variance (α undefined). ∞ → the totals
        // overflowed on large finite input; `sum_item_var / ∞ = 0.0` would
        // yield a finite-but-wrong α, so reject the overflowed denominator.
        return None;
    }
    let kf = k as f64;
    let alpha = (kf / (kf - 1.0)) * (1.0 - sum_item_var / total_var);
    // Even with finite inputs, large magnitudes can overflow the variance
    // sums to ±∞, making α non-finite — reject that too.
    alpha.is_finite().then_some(alpha)
}

/// The single-measure ICC forms of Shrout & Fleiss (1979).
///
/// Both are for `n` subjects each rated by the SAME `k` raters (a two-way
/// design). They differ in what counts as error:
/// - [`IccForm::Icc2_1`] — two-way random effects, **absolute agreement**:
///   rater bias (systematic column differences) IS counted as disagreement.
/// - [`IccForm::Icc3_1`] — two-way mixed effects, **consistency**: rater bias is
///   NOT counted (the raters are the only raters of interest).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum IccForm {
    /// ICC(2,1): two-way random, single measures, absolute agreement.
    Icc2_1,
    /// ICC(3,1): two-way mixed, single measures, consistency.
    Icc3_1,
}

/// Intraclass correlation coefficient over a subjects×raters matrix.
///
/// `ratings` is an `n`-length slice, each element a length-`k` vector of one
/// subject's ratings across the `k` raters (`ratings[s][r]` = subject `s`, rater
/// `r`). Requires `n ≥ 2`, `k ≥ 2`, and a rectangular matrix.
///
/// Uses the two-way ANOVA mean squares (Shrout & Fleiss 1979, eqns for MSR/
/// MSC/MSE):
/// - `MSR = k·Σ_s (row_mean_s − grand)² / (n−1)` (between subjects),
/// - `MSC = n·Σ_r (col_mean_r − grand)² / (k−1)` (between raters),
/// - `MSE = [SS_total − (n−1)·MSR − (k−1)·MSC] / [(n−1)(k−1)]` (residual).
///
/// Then:
/// - `ICC(3,1) = (MSR − MSE) / (MSR + (k−1)·MSE)`,
/// - `ICC(2,1) = (MSR − MSE) / (MSR + (k−1)·MSE + (k/n)·(MSC − MSE))`.
///
/// Returns `None` on ragged input, `n < 2`, `k < 2`, or a zero denominator.
pub fn icc(ratings: &[Vec<f64>], form: IccForm) -> Option<f64> {
    let n = ratings.len();
    if n < 2 {
        return None;
    }
    let k = ratings[0].len();
    if k < 2 || ratings.iter().any(|r| r.len() != k) {
        return None;
    }
    if ratings.iter().any(|r| !all_finite(r)) {
        return None; // NaN / ±∞ input → undefined (no-NaN contract)
    }
    let nf = n as f64;
    let kf = k as f64;

    let grand = ratings.iter().flat_map(|r| r.iter()).sum::<f64>() / (nf * kf);

    // Between-subjects (row) sum of squares.
    let mut ss_rows = 0.0;
    for row in ratings {
        let rm = row.iter().sum::<f64>() / kf;
        ss_rows += (rm - grand) * (rm - grand);
    }
    ss_rows *= kf; // each row mean stands for k observations
    let ms_r = ss_rows / (nf - 1.0);

    // Between-raters (column) sum of squares.
    let mut ss_cols = 0.0;
    for c in 0..k {
        let cm = ratings.iter().map(|r| r[c]).sum::<f64>() / nf;
        ss_cols += (cm - grand) * (cm - grand);
    }
    ss_cols *= nf; // each column mean stands for n observations
    let ms_c = ss_cols / (kf - 1.0);

    // Total sum of squares → residual (error) mean square.
    let ss_total: f64 = ratings
        .iter()
        .flat_map(|r| r.iter())
        .map(|&v| (v - grand) * (v - grand))
        .sum();
    let ss_error = ss_total - ss_rows - ss_cols;
    let ms_e = ss_error / ((nf - 1.0) * (kf - 1.0));

    let denom = match form {
        IccForm::Icc3_1 => ms_r + (kf - 1.0) * ms_e,
        IccForm::Icc2_1 => ms_r + (kf - 1.0) * ms_e + (kf / nf) * (ms_c - ms_e),
    };
    if denom == 0.0 || !denom.is_finite() {
        // 0 → zero-variance degenerate. ∞ → the mean-square sums overflowed on
        // large finite input; `(ms_r - ms_e) / ∞ = 0.0` would be finite-but-
        // wrong, so reject the overflowed denominator here.
        return None;
    }
    let v = (ms_r - ms_e) / denom;
    // A finite non-zero denom can still pair with a non-finite numerator on
    // overflow — reject a non-finite ratio too.
    v.is_finite().then_some(v)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() < tol
    }

    #[test]
    fn pearson_perfect_and_anti() {
        assert!(approx(
            pearson(&[1.0, 2.0, 3.0], &[1.0, 2.0, 3.0]).unwrap(),
            1.0,
            1e-12
        ));
        assert!(approx(
            pearson(&[1.0, 2.0, 3.0], &[3.0, 2.0, 1.0]).unwrap(),
            -1.0,
            1e-12
        ));
    }

    #[test]
    fn pearson_textbook_value() {
        // x, y with a known r ≈ 0.774597 (mean x=3, mean y=4).
        let x = [1.0, 2.0, 3.0, 4.0, 5.0];
        let y = [2.0, 4.0, 5.0, 4.0, 5.0];
        assert!(approx(pearson(&x, &y).unwrap(), 0.774_597, 1e-5));
    }

    #[test]
    fn pearson_degenerate_returns_none() {
        assert_eq!(pearson(&[1.0, 1.0, 1.0], &[1.0, 2.0, 3.0]), None); // constant x
        assert_eq!(pearson(&[1.0], &[1.0]), None); // n < 2
        assert_eq!(pearson(&[1.0, 2.0], &[1.0]), None); // ragged
    }

    #[test]
    fn spearman_monotone_nonlinear_is_one() {
        let x = [1.0, 2.0, 3.0, 4.0, 5.0];
        let y = [1.0, 4.0, 9.0, 16.0, 25.0]; // strictly increasing → ρ = 1
        assert!(approx(spearman(&x, &y).unwrap(), 1.0, 1e-12));
    }

    #[test]
    fn spearman_tie_correction() {
        // With a tie in y, average-rank must be used. x=[1,2,3,4], y=[1,2,2,3].
        // ranks(x)=[1,2,3,4], ranks(y)=[1,2.5,2.5,4] → ρ = pearson of those.
        let x = [1.0, 2.0, 3.0, 4.0];
        let y = [1.0, 2.0, 2.0, 3.0];
        let rho = spearman(&x, &y).unwrap();
        // Hand-computed: ρ ≈ 0.948683.
        assert!(approx(rho, 0.948_683, 1e-5));
    }

    #[test]
    fn cronbach_identical_items_is_one() {
        let items = vec![
            vec![1.0, 2.0, 3.0, 4.0, 5.0],
            vec![1.0, 2.0, 3.0, 4.0, 5.0],
            vec![1.0, 2.0, 3.0, 4.0, 5.0],
        ];
        assert!(approx(cronbach_alpha(&items).unwrap(), 1.0, 1e-12));
    }

    #[test]
    fn cronbach_known_value() {
        // 3 items × 4 subjects; the items are near-affine (highly consistent).
        //   item1 = [2,4,3,5], item2 = [3,5,3,6], item3 = [1,3,2,4]
        // Hand calc: Σσ²_item = 1.25+1.6875+1.25 = 4.1875; subject totals
        // [6,12,8,15] have σ²_total = 12.1875; α = (3/2)(1 − 4.1875/12.1875)
        // = 1.5 · 0.656410… = 0.984615…
        let items = vec![
            vec![2.0, 4.0, 3.0, 5.0],
            vec![3.0, 5.0, 3.0, 6.0],
            vec![1.0, 3.0, 2.0, 4.0],
        ];
        let a = cronbach_alpha(&items).unwrap();
        assert!(approx(a, 0.984_615, 1e-5), "cronbach α was {a}");
    }

    #[test]
    fn cronbach_degenerate_returns_none() {
        assert_eq!(cronbach_alpha(&[vec![1.0, 2.0]]), None); // k < 2
                                                             // No between-subject variance (all subject totals equal) → None.
        let flat = vec![vec![1.0, 2.0], vec![2.0, 1.0]]; // totals both 3
        assert_eq!(cronbach_alpha(&flat), None);
    }

    /// The canonical worked example from Shrout & Fleiss (1979): 6 subjects ×
    /// 4 judges. Reference values (reproduced by R `psych::ICC`, Python
    /// `pingouin.intraclass_corr`): ICC(2,1) ≈ 0.290, ICC(3,1) ≈ 0.715.
    fn shrout_fleiss_data() -> Vec<Vec<f64>> {
        vec![
            vec![9.0, 2.0, 5.0, 8.0],
            vec![6.0, 1.0, 3.0, 2.0],
            vec![8.0, 4.0, 6.0, 8.0],
            vec![7.0, 1.0, 2.0, 6.0],
            vec![10.0, 5.0, 6.0, 9.0],
            vec![6.0, 2.0, 4.0, 7.0],
        ]
    }

    #[test]
    fn icc_2_1_absolute_agreement_matches_shrout_fleiss() {
        let data = shrout_fleiss_data();
        let v = icc(&data, IccForm::Icc2_1).unwrap();
        assert!(approx(v, 0.290, 2e-3), "ICC(2,1) was {v}, expected ≈ 0.290");
    }

    #[test]
    fn icc_3_1_consistency_matches_shrout_fleiss() {
        let data = shrout_fleiss_data();
        let v = icc(&data, IccForm::Icc3_1).unwrap();
        assert!(approx(v, 0.715, 2e-3), "ICC(3,1) was {v}, expected ≈ 0.715");
    }

    #[test]
    fn icc_absolute_penalises_rater_bias_more_than_consistency() {
        // Add a constant +5 bias to rater 0: consistency (3,1) is unchanged in
        // spirit but absolute-agreement (2,1) drops, because column means now
        // differ systematically. Assert 2_1 < 3_1 on biased data.
        let mut data = shrout_fleiss_data();
        for row in &mut data {
            row[0] += 5.0;
        }
        let a = icc(&data, IccForm::Icc2_1).unwrap();
        let c = icc(&data, IccForm::Icc3_1).unwrap();
        assert!(
            a < c,
            "absolute-agreement {a} should be < consistency {c} under rater bias"
        );
    }

    #[test]
    fn icc_degenerate_returns_none() {
        assert_eq!(icc(&[vec![1.0, 2.0]], IccForm::Icc2_1), None); // n < 2
        assert_eq!(icc(&[vec![1.0], vec![2.0]], IccForm::Icc2_1), None); // k < 2
        assert_eq!(icc(&[vec![1.0, 2.0], vec![1.0]], IccForm::Icc2_1), None); // ragged
    }

    #[test]
    fn non_finite_inputs_return_none() {
        // The no-NaN API contract: any NaN / ±∞ in the input yields None, never
        // a finite-but-garbage estimate or Some(NaN). The Spearman case is the
        // one two independent reviewers flagged: `average_ranks` maps NaN to a
        // finite rank, so without the up-front guard it would return Some(1.0).
        assert_eq!(spearman(&[1.0, f64::NAN, 2.0], &[1.0, 2.0, 3.0]), None);
        assert_eq!(spearman(&[1.0, 2.0, 3.0], &[1.0, f64::INFINITY, 3.0]), None);
        assert_eq!(pearson(&[1.0, f64::NAN, 3.0], &[1.0, 2.0, 3.0]), None);
        assert_eq!(pearson(&[1.0, 2.0, 3.0], &[f64::NEG_INFINITY, 2.0, 3.0]), None);

        let nan_items = vec![vec![1.0, 2.0, f64::NAN], vec![1.0, 2.0, 3.0]];
        assert_eq!(cronbach_alpha(&nan_items), None);
        let inf_items = vec![vec![1.0, 2.0, 3.0], vec![1.0, f64::INFINITY, 3.0]];
        assert_eq!(cronbach_alpha(&inf_items), None);

        let nan_ratings = vec![vec![1.0, 2.0], vec![f64::NAN, 4.0], vec![5.0, 6.0]];
        assert_eq!(icc(&nan_ratings, IccForm::Icc2_1), None);
        assert_eq!(icc(&nan_ratings, IccForm::Icc3_1), None);
    }

    #[test]
    fn overflowing_large_finite_inputs_return_none_not_nan() {
        // The Codex reproducer: perfectly-correlated large-magnitude data whose
        // squared-deviation product overflows the denominator to ∞, so `sxy/∞`
        // collapses to a FINITE-but-wrong 0.0 that a trailing `is_finite` guard
        // misses. Must be None (the denom-finiteness guard catches it).
        assert_eq!(pearson(&[1e100, -1e100], &[1e100, -1e100]), None);
        // (spearman is immune to this: it Pearson's the tiny 1..n ranks, which
        // never overflow — `spearman(&[1e100,-1e100],&[1e100,-1e100])` is a
        // correct `Some(1.0)`, so it is not asserted here.)
        // Finite but astronomically large magnitudes overflow the squared
        // deviations / mean-square sums to ±∞, whose ratio is NaN or a
        // finite-but-wrong 0.0. The denom + result guards must return None.
        let big = 1e308;
        for r in [pearson(&[big, -big, big], &[big, big, -big]), spearman(&[big, -big, big], &[big, big, -big])] {
            assert!(r.map(|v| v.is_finite()).unwrap_or(true), "expected finite or None, got {r:?}");
        }
        let big_items = vec![vec![big, -big, big], vec![-big, big, -big]];
        let a = cronbach_alpha(&big_items);
        assert!(a.map(|v| v.is_finite()).unwrap_or(true), "cronbach: expected finite or None, got {a:?}");
        let big_ratings = vec![vec![big, -big], vec![-big, big], vec![big, big]];
        let v = icc(&big_ratings, IccForm::Icc2_1);
        assert!(v.map(|x| x.is_finite()).unwrap_or(true), "icc: expected finite or None, got {v:?}");
    }
}
