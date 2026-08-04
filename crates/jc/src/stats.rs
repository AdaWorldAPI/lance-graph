//! Second statistics battery — **agreement**, **factor-model reliability**,
//! **r-family effect size**, and their **significance** companions.
//!
//! Additive companion to [`crate::reliability`], which ships Pearson r,
//! Spearman ρ, Cronbach α and ICC. Nothing here re-implements those: `phi`
//! delegates to [`crate::reliability::pearson`], and the two crate-private
//! helpers this module borrows (`mean`, `all_finite`) are shared, not copied.
//!
//! # Why these five, and why here (D-KIA-C1b)
//!
//! The [`crate::reliability`] audit found that two of the statistics the
//! workspace wanted under new names were the SAME computation already shipped
//! — **φ is Pearson r on two binary variables**, and **KR-20 is Cronbach α on
//! dichotomous items** — while **κ was genuinely absent**, blocking the D3
//! fusion falsifier. This module closes that gap and adds the effect-size and
//! significance surface the gates need:
//!
//! - **[`cohen_kappa`]** — chance-corrected agreement between two raters over
//!   nominal categories. NOT an ICC under another name: ICC decomposes
//!   *variance* for interval ratings; κ corrects *counts* for the agreement
//!   expected from the raters' marginals alone.
//! - **[`omega_total`]** — McDonald's ω_t, the congeneric single-factor
//!   reliability coefficient α is routinely mis-substituted for. α assumes
//!   *tau-equivalence* (equal loadings); ω does not, so **ω ≥ α** whenever
//!   loadings differ, and α understates reliability exactly there.
//! - **[`phi`]** — the r-family effect size for a 2×2 contrast, as a named
//!   wrapper over `pearson` (φ² = proportion of variance explained).
//! - **[`multiple_r`] / [`multiple_r_squared`] / [`eta_squared`]** — the
//!   r-family over many criteria (R, R²) and over group factors (η²,
//!   *erklärte Varianz*).
//! - **[`t_test_one_sample`] / [`t_test_paired`] / [`t_test_welch`] /
//!   [`t_test_student`] / [`anova_one_way`]** — the significance companions,
//!   each reporting the statistic, its df, and a p-value.
//!
//! # The effect-size family is r, deliberately
//!
//! Everything here is the **r family** — correlation and variance explained
//! (φ, R, R², η²). The **d family (Cohen's d) is out of scope by design**: if a
//! standardised mean difference is ever wanted it is computed separately. The
//! t-tests below are the *significance* companions to η²/R², not a d-family
//! back door — they report `t`, `df` and `p`, and the effect size is read off
//! η² or R².
//!
//! # Statistical-significance note (I-NOISE-FLOOR-JIRAK)
//!
//! The p-values here are the **classical** ones — they assume independent
//! observations. Per the workspace iron rule, a claim of the form "observed
//! value is N σ above the noise floor" **on the 16384-bit fingerprints** must
//! be calibrated with **Jirak 2016** (weak dependence), NOT with these p-values
//! and not with classical IID Berry-Esseen: those bits are weakly dependent by
//! construction. Use `p` here for ordinary independent samples; use
//! [`crate::jirak`] for substrate claims. Stating which one a result used is
//! part of the claim.
//!
//! # Citations
//!
//! - J. Cohen, "A coefficient of agreement for nominal scales", Educational and
//!   Psychological Measurement 20(1) (1960), 37–46.
//! - R. P. McDonald, *Test Theory: A Unified Treatment*, Erlbaum (1999), §6.5
//!   (ω_t from the congeneric single-factor model).
//! - C. Spearman, "'General intelligence', objectively determined and
//!   measured", Am. J. Psychol. 15 (1904) — the vanishing-tetrad / triad
//!   identity used to estimate the loadings.
//! - B. L. Welch, "The generalization of Student's problem when several
//!   different population variances are involved", Biometrika 34 (1947).
//! - W. H. Press et al., *Numerical Recipes*, 3rd ed., §6.4 — the continued
//!   fraction for the regularised incomplete beta function used for every
//!   p-value here.
//!
//! All estimators return `Option<...>`, yielding `None` on degenerate input
//! (too few observations, zero variance, ragged matrices, an unidentified
//! factor model) rather than panicking or returning `NaN` — matching the
//! [`crate::reliability`] contract exactly. Non-finite input is rejected up
//! front by the same `all_finite` guard.

use crate::reliability::{all_finite, mean, pearson};
use std::collections::BTreeSet;

// ─────────────────────────── local helpers ───────────────────────────
//
// `sample_var` / `sample_cov` use the UNBIASED divisor (n−1); `reliability`'s
// private `pop_var` uses the population divisor (n). They are different
// estimators, not a duplicate — and the divisor cancels in every ratio below,
// so mixing conventions across the two modules changes no returned value.

/// Unbiased (divisor `n−1`) sample variance. `None` if `n < 2`.
#[inline]
fn sample_var(xs: &[f64]) -> Option<f64> {
    if xs.len() < 2 {
        return None;
    }
    let m = mean(xs)?;
    let n = xs.len() as f64;
    Some(xs.iter().map(|&v| (v - m) * (v - m)).sum::<f64>() / (n - 1.0))
}

/// Unbiased (divisor `n−1`) sample covariance. `None` if lengths differ or
/// `n < 2`.
#[inline]
fn sample_cov(x: &[f64], y: &[f64]) -> Option<f64> {
    if x.len() != y.len() || x.len() < 2 {
        return None;
    }
    let mx = mean(x)?;
    let my = mean(y)?;
    let n = x.len() as f64;
    Some(
        x.iter()
            .zip(y.iter())
            .map(|(&a, &b)| (a - mx) * (b - my))
            .sum::<f64>()
            / (n - 1.0),
    )
}

// ───────────────────── incomplete beta (p-values) ─────────────────────

/// Lanczos approximation to `ln Γ(x)` for `x > 0` (g = 7, n = 9).
fn ln_gamma(x: f64) -> f64 {
    const C: [f64; 9] = [
        0.999_999_999_999_809_9,
        676.520_368_121_885_1,
        -1_259.139_216_722_402_8,
        771.323_428_777_653_1,
        -176.615_029_162_140_6,
        12.507_343_278_686_905,
        -0.138_571_095_265_720_12,
        9.984_369_578_019_572e-6,
        1.505_632_735_149_311_6e-7,
    ];
    if x < 0.5 {
        // Reflection: Γ(x)Γ(1−x) = π / sin(πx)
        return (std::f64::consts::PI / (std::f64::consts::PI * x).sin()).ln() - ln_gamma(1.0 - x);
    }
    let x = x - 1.0;
    let mut a = C[0];
    let t = x + 7.5;
    for (i, &c) in C.iter().enumerate().skip(1) {
        a += c / (x + i as f64);
    }
    0.5 * (2.0 * std::f64::consts::PI).ln() + (x + 0.5) * t.ln() - t + a.ln()
}

/// Continued fraction for the incomplete beta function (Numerical Recipes
/// §6.4, modified Lentz). Converges for `x < (a+1)/(a+b+2)`.
fn betacf(a: f64, b: f64, x: f64) -> f64 {
    const MAXIT: usize = 300;
    const EPS: f64 = 3.0e-16;
    const FPMIN: f64 = 1.0e-300;

    let qab = a + b;
    let qap = a + 1.0;
    let qam = a - 1.0;
    let mut c = 1.0;
    let mut d = 1.0 - qab * x / qap;
    if d.abs() < FPMIN {
        d = FPMIN;
    }
    d = 1.0 / d;
    let mut h = d;
    for m in 1..=MAXIT {
        let m_f = m as f64;
        let m2 = 2.0 * m_f;
        // Even step.
        let aa = m_f * (b - m_f) * x / ((qam + m2) * (a + m2));
        d = 1.0 + aa * d;
        if d.abs() < FPMIN {
            d = FPMIN;
        }
        c = 1.0 + aa / c;
        if c.abs() < FPMIN {
            c = FPMIN;
        }
        d = 1.0 / d;
        h *= d * c;
        // Odd step.
        let aa = -(a + m_f) * (qab + m_f) * x / ((a + m2) * (qap + m2));
        d = 1.0 + aa * d;
        if d.abs() < FPMIN {
            d = FPMIN;
        }
        c = 1.0 + aa / c;
        if c.abs() < FPMIN {
            c = FPMIN;
        }
        d = 1.0 / d;
        let del = d * c;
        h *= del;
        if (del - 1.0).abs() < EPS {
            break;
        }
    }
    h
}

/// Regularised incomplete beta `I_x(a, b)`, the CDF machinery behind every
/// p-value in this module. Returns `None` outside `x ∈ [0,1]` or on
/// non-positive shape parameters.
fn reg_inc_beta(a: f64, b: f64, x: f64) -> Option<f64> {
    if !(a > 0.0 && b > 0.0 && (0.0..=1.0).contains(&x) && x.is_finite()) {
        return None;
    }
    if x == 0.0 {
        return Some(0.0);
    }
    if x == 1.0 {
        return Some(1.0);
    }
    let front =
        (ln_gamma(a + b) - ln_gamma(a) - ln_gamma(b) + a * x.ln() + b * (1.0 - x).ln()).exp();
    let v = if x < (a + 1.0) / (a + b + 2.0) {
        front * betacf(a, b, x) / a
    } else {
        // Symmetry: I_x(a,b) = 1 − I_{1−x}(b,a)
        1.0 - front * betacf(b, a, 1.0 - x) / b
    };
    v.is_finite().then(|| v.clamp(0.0, 1.0))
}

/// Two-tailed p-value of Student's `t` with `df` degrees of freedom:
/// `P(|T| ≥ |t|) = I_{df/(df+t²)}(df/2, 1/2)`.
fn t_two_tailed_p(t: f64, df: f64) -> Option<f64> {
    if !t.is_finite() || !df.is_finite() || df <= 0.0 {
        return None;
    }
    reg_inc_beta(df / 2.0, 0.5, df / (df + t * t))
}

/// Upper-tail p-value of the F distribution:
/// `P(F ≥ f) = I_{df2/(df2+df1·f)}(df2/2, df1/2)`.
fn f_upper_p(f: f64, df1: f64, df2: f64) -> Option<f64> {
    if !f.is_finite() || f < 0.0 || df1 <= 0.0 || df2 <= 0.0 {
        return None;
    }
    reg_inc_beta(df2 / 2.0, df1 / 2.0, df2 / (df2 + df1 * f))
}

// ──────────────────────────── agreement: κ ────────────────────────────

/// Cohen's κ — chance-corrected agreement between two raters over nominal
/// categories.
///
/// `κ = (p_o − p_e) / (1 − p_e)`, where `p_o` is the observed proportion of
/// matching labels and `p_e` is the agreement expected from the two raters'
/// marginal distributions alone. κ = 1 is perfect agreement, κ = 0 is exactly
/// chance, and κ < 0 is worse than chance (a real, if pathological, value —
/// returned as-is).
///
/// Categories are `usize` labels, NOT `f64`: the estimator compares them for
/// exact equality, and float equality on measured values is a defect waiting
/// to happen. Any labelling works — only the partition matters.
///
/// **κ is not an ICC under another name.** ICC decomposes *variance* for
/// interval-scaled ratings; κ corrects *counts* for chance. On binary criteria
/// where a continuous workflow would reach for ICC, compute κ.
///
/// Returns `None` if the slices differ in length, are empty, or `p_e == 1`
/// (both raters used a single identical category throughout — every
/// assignment agrees by construction, so chance-corrected agreement is
/// undefined, `0/0`).
///
/// ```
/// use jc::stats::cohen_kappa;
/// // Perfect agreement over two used categories → κ = 1.
/// let a = [0usize, 1, 0, 1];
/// let b = [0usize, 1, 0, 1];
/// assert!((cohen_kappa(&a, &b).unwrap() - 1.0).abs() < 1e-12);
/// ```
pub fn cohen_kappa(a: &[usize], b: &[usize]) -> Option<f64> {
    if a.len() != b.len() || a.is_empty() {
        return None;
    }
    let n = a.len() as f64;
    let cats: BTreeSet<usize> = a.iter().chain(b.iter()).copied().collect();

    let agree = a.iter().zip(b.iter()).filter(|(x, y)| x == y).count() as f64;
    let p_o = agree / n;

    let mut p_e = 0.0;
    for c in &cats {
        let ma = a.iter().filter(|&v| v == c).count() as f64 / n;
        let mb = b.iter().filter(|&v| v == c).count() as f64 / n;
        p_e += ma * mb;
    }

    let denom = 1.0 - p_e;
    if denom == 0.0 || !denom.is_finite() {
        return None;
    }
    let k = (p_o - p_e) / denom;
    k.is_finite().then_some(k)
}

// ───────────────────── reliability: McDonald's ω ─────────────────────

/// McDonald's ω_t — congeneric single-factor reliability.
///
/// `items` has the same shape as [`crate::reliability::cronbach_alpha`]'s: a
/// `k`-length slice, each element the length-`n` vector of one item's scores
/// across subjects (`items[i][s]`).
///
/// Where α assumes **tau-equivalence** (all items load equally on the common
/// factor), ω only assumes a **congeneric** single factor, so it does not
/// understate reliability when loadings differ: **ω ≥ α**, with equality when
/// the loadings are equal. That inequality is the reason to prefer ω, and it
/// is asserted as a test rather than merely claimed.
///
/// `ω_t = (Σλ_i)² / [(Σλ_i)² + Σψ_i]`, with loadings estimated by Spearman's
/// triad identity — under a single factor, `σ_ij = λ_iλ_j`, so
/// `λ_i² = σ_ij·σ_ik / σ_jk` for any pair `j,k ≠ i`; the estimate averages
/// every admissible triad. Residual variances are `ψ_i = σ_ii − λ_i²`.
///
/// Requires `k ≥ 3` — with two items the single-factor model is **not
/// identified** (one covariance, two unknown loadings), so `None` is returned
/// rather than a fabricated estimate. Also returns `None` on ragged input,
/// `n < 2`, non-finite input, a triad set with no usable denominator, a
/// negative `λ_i²`, or a **Heywood case** (`ψ_i < 0`, i.e. estimated common
/// variance exceeding the item's total variance) — each of which means the
/// congeneric model does not fit, not that reliability is low.
///
/// ```
/// use jc::stats::omega_total;
/// // Three items that are exact multiples of one factor (no residual):
/// // perfectly congeneric, zero error → ω = 1.
/// let items = vec![
///     vec![1.0, 2.0, 3.0, 4.0],
///     vec![2.0, 4.0, 6.0, 8.0],
///     vec![3.0, 6.0, 9.0, 12.0],
/// ];
/// assert!((omega_total(&items).unwrap() - 1.0).abs() < 1e-9);
/// ```
pub fn omega_total(items: &[Vec<f64>]) -> Option<f64> {
    let k = items.len();
    if k < 3 {
        return None; // single-factor model unidentified below 3 items
    }
    let n = items[0].len();
    if n < 2 || items.iter().any(|it| it.len() != n) {
        return None;
    }
    if items.iter().any(|it| !all_finite(it)) {
        return None;
    }

    // Covariance matrix.
    let mut cov = vec![vec![0.0f64; k]; k];
    for i in 0..k {
        for j in 0..k {
            cov[i][j] = if i == j {
                sample_var(&items[i])?
            } else {
                sample_cov(&items[i], &items[j])?
            };
        }
    }

    // λ_i² averaged over every admissible triad (j,k ≠ i, j ≠ k, σ_jk ≠ 0).
    let mut lambda = vec![0.0f64; k];
    for i in 0..k {
        let mut acc = 0.0;
        let mut count = 0usize;
        for j in 0..k {
            if j == i {
                continue;
            }
            for l in (j + 1)..k {
                if l == i {
                    continue;
                }
                let denom = cov[j][l];
                if denom == 0.0 || !denom.is_finite() {
                    continue; // this triad carries no information
                }
                let est = cov[i][j] * cov[i][l] / denom;
                if est.is_finite() {
                    acc += est;
                    count += 1;
                }
            }
        }
        if count == 0 {
            return None; // no usable triad → loading not identified
        }
        let lam_sq = acc / count as f64;
        // Tolerance is RELATIVE to the item's own variance: a perfect-fit item
        // has λ² == σ_ii exactly in real arithmetic but lands a few ulps either
        // side of it in f64, and a bare `< 0.0` test would reject a valid model
        // as misfit. Only a violation LARGER than rounding is a real one.
        let tol = 1e-9 * cov[i][i].abs().max(1.0);
        if lam_sq < -tol || !lam_sq.is_finite() {
            return None; // single-factor model violated (negative common variance)
        }
        lambda[i] = lam_sq.max(0.0).sqrt();
    }

    let sum_lambda: f64 = lambda.iter().sum();
    let mut sum_psi = 0.0;
    for i in 0..k {
        let psi = cov[i][i] - lambda[i] * lambda[i];
        let tol = 1e-9 * cov[i][i].abs().max(1.0);
        if psi < -tol {
            // Heywood case: estimated common variance genuinely exceeds the
            // item's total variance → model misfit, not a reliability value.
            return None;
        }
        sum_psi += psi.max(0.0);
    }

    let common = sum_lambda * sum_lambda;
    let denom = common + sum_psi;
    if denom == 0.0 || !denom.is_finite() {
        return None;
    }
    let omega = common / denom;
    omega.is_finite().then_some(omega)
}

// ───────────────────── effect size: the r family ─────────────────────

/// The φ coefficient — the r-family effect size for a 2×2 contrast.
///
/// **φ IS Pearson r computed on two binary variables**, so this function
/// delegates to [`crate::reliability::pearson`] rather than re-deriving it;
/// `bool` inputs make the binary precondition unforgeable. `φ²` is the
/// proportion of variance explained.
///
/// **Ceiling caveat (report it with the value):** φ can only reach ±1 when the
/// two marginal distributions match. With unequal marginals the attainable
/// maximum is strictly below 1, so a "low" φ may be at its own ceiling — φ
/// without its marginals is not interpretable.
///
/// Returns `None` under [`pearson`]'s conditions: lengths differ, `n < 2`, or
/// either vector is constant (all-true or all-false → zero variance).
///
/// ```
/// use jc::stats::phi;
/// let a = [true, true, false, false];
/// let b = [true, true, false, false];
/// assert!((phi(&a, &b).unwrap() - 1.0).abs() < 1e-12);
/// ```
pub fn phi(x: &[bool], y: &[bool]) -> Option<f64> {
    let xf: Vec<f64> = x.iter().map(|&v| if v { 1.0 } else { 0.0 }).collect();
    let yf: Vec<f64> = y.iter().map(|&v| if v { 1.0 } else { 0.0 }).collect();
    pearson(&xf, &yf)
}

/// Solve `A·b = rhs` by Gaussian elimination with partial pivoting.
/// `None` if the system is singular to working precision.
fn solve(mut a: Vec<Vec<f64>>, mut rhs: Vec<f64>) -> Option<Vec<f64>> {
    let n = rhs.len();
    for col in 0..n {
        // Partial pivot.
        let (piv, &max) = a[col..]
            .iter()
            .enumerate()
            .map(|(i, row)| (i + col, &row[col]))
            .max_by(|(_, x), (_, y)| {
                x.abs()
                    .partial_cmp(&y.abs())
                    .unwrap_or(std::cmp::Ordering::Equal)
            })?;
        if max.abs() < 1e-12 {
            return None; // singular → collinear predictors
        }
        a.swap(col, piv);
        rhs.swap(col, piv);
        for row in (col + 1)..n {
            let f = a[row][col] / a[col][col];
            if !f.is_finite() {
                return None;
            }
            let (upper, lower) = a.split_at_mut(row);
            let pivot_row = &upper[col];
            for (c, cell) in lower[0].iter_mut().enumerate().skip(col) {
                *cell -= f * pivot_row[c];
            }
            rhs[row] -= f * rhs[col];
        }
    }
    // Back-substitution.
    let mut out = vec![0.0; n];
    for row in (0..n).rev() {
        let mut acc = rhs[row];
        for c in (row + 1)..n {
            acc -= a[row][c] * out[c];
        }
        let v = acc / a[row][row];
        if !v.is_finite() {
            return None;
        }
        out[row] = v;
    }
    Some(out)
}

/// Coefficient of multiple determination `R²` — the proportion of variance in
/// `y` explained by an ordinary least-squares fit on all `predictors`
/// (intercept included).
///
/// This is the **multi-criterion** member of the r family: with a single
/// predictor it equals `pearson(y, x)²` exactly (asserted in the tests), and
/// the genuinely new surface is `k > 1`.
///
/// `predictors` is a `k`-length slice of length-`n` columns. Requires `n ≥ k+2`
/// (one residual degree of freedom beyond the fitted coefficients), a
/// rectangular design, finite input, non-constant `y`, and linearly
/// independent predictors — `None` otherwise. Note `R²` never decreases when a
/// predictor is added, so it is **not** a model-selection criterion.
///
/// ```
/// use jc::stats::multiple_r_squared;
/// // y = 3 + 2a + b exactly → R² = 1. Note `b` must not be an affine function
/// // of `a`: the design includes an intercept, so `b = a + 1` would make the
/// // normal equations singular and yield `None`, not 1.0.
/// let a = vec![1.0, 2.0, 3.0, 4.0, 5.0];
/// let b = vec![1.0, 0.0, 2.0, 1.0, 3.0];
/// let y = vec![6.0, 7.0, 11.0, 12.0, 16.0];
/// assert!((multiple_r_squared(&y, &[a, b]).unwrap() - 1.0).abs() < 1e-9);
/// ```
pub fn multiple_r_squared(y: &[f64], predictors: &[Vec<f64>]) -> Option<f64> {
    let n = y.len();
    let k = predictors.len();
    if k == 0 || n < k + 2 {
        return None;
    }
    if predictors.iter().any(|p| p.len() != n) {
        return None;
    }
    if !all_finite(y) || predictors.iter().any(|p| !all_finite(p)) {
        return None;
    }

    // Design matrix with intercept: columns [1, p_0, .., p_{k-1}].
    let m = k + 1;
    let col = |c: usize, i: usize| -> f64 {
        if c == 0 {
            1.0
        } else {
            predictors[c - 1][i]
        }
    };

    // Normal equations XᵀX b = Xᵀy.
    let mut xtx = vec![vec![0.0f64; m]; m];
    let mut xty = vec![0.0f64; m];
    for (r, xtx_row) in xtx.iter_mut().enumerate() {
        for (c, cell) in xtx_row.iter_mut().enumerate() {
            *cell = (0..n).map(|i| col(r, i) * col(c, i)).sum();
        }
        xty[r] = (0..n).map(|i| col(r, i) * y[i]).sum();
    }
    if xtx.iter().any(|row| row.iter().any(|v| !v.is_finite()))
        || xty.iter().any(|v| !v.is_finite())
    {
        return None; // overflowed on large finite input
    }

    let beta = solve(xtx, xty)?;

    let my = mean(y)?;
    let ss_tot: f64 = y.iter().map(|&v| (v - my) * (v - my)).sum();
    if ss_tot == 0.0 || !ss_tot.is_finite() {
        return None; // constant y → R² undefined
    }
    let ss_res: f64 = (0..n)
        .map(|i| {
            let pred: f64 = (0..m).map(|c| beta[c] * col(c, i)).sum();
            let e = y[i] - pred;
            e * e
        })
        .sum();
    if !ss_res.is_finite() {
        return None;
    }
    // Clamp: with an exact fit `ss_res` can land a few ulps below zero.
    let r2 = (1.0 - ss_res / ss_tot).clamp(0.0, 1.0);
    r2.is_finite().then_some(r2)
}

/// Multiple correlation `R = √R²`. See [`multiple_r_squared`].
pub fn multiple_r(y: &[f64], predictors: &[Vec<f64>]) -> Option<f64> {
    multiple_r_squared(y, predictors).map(|v| v.sqrt())
}

/// Between-group and total sums of squares for a one-way layout.
/// `None` on fewer than 2 groups, any empty group, or non-finite input.
fn one_way_ss(groups: &[Vec<f64>]) -> Option<(f64, f64, usize, usize)> {
    if groups.len() < 2 || groups.iter().any(|g| g.is_empty()) {
        return None;
    }
    if groups.iter().any(|g| !all_finite(g)) {
        return None;
    }
    let n_total: usize = groups.iter().map(|g| g.len()).sum();
    let grand = groups.iter().flat_map(|g| g.iter()).sum::<f64>() / n_total as f64;
    if !grand.is_finite() {
        return None;
    }
    let mut ss_between = 0.0;
    for g in groups {
        let gm = mean(g)?;
        let d = gm - grand;
        ss_between += g.len() as f64 * d * d;
    }
    let ss_total: f64 = groups
        .iter()
        .flat_map(|g| g.iter())
        .map(|&v| (v - grand) * (v - grand))
        .sum();
    if !ss_between.is_finite() || !ss_total.is_finite() {
        return None;
    }
    Some((ss_between, ss_total, groups.len(), n_total))
}

/// η² — the proportion of total variance explained by group membership
/// (*erklärte Varianz*), `η² = SS_between / SS_total`.
///
/// The r-family effect size for a group factor: `η² ∈ [0,1]`, 0 when every
/// group mean equals the grand mean and 1 when all variance is between groups.
/// For two groups it coincides with `R²` on a 0/1 dummy predictor and with
/// `t²/(t²+df)` from the pooled two-sample t — both asserted in the tests.
///
/// Requires ≥ 2 non-empty groups, finite input, and non-zero total variance;
/// `None` otherwise. Note η² is the **sample** ratio and is upward-biased as an
/// estimate of the population value (ω²/ε² correct for that; neither is
/// provided here — say which one a claim uses).
///
/// ```
/// use jc::stats::eta_squared;
/// // Identical group means → nothing explained by group membership.
/// let g = vec![vec![1.0, 2.0, 3.0], vec![1.0, 2.0, 3.0]];
/// assert!(eta_squared(&g).unwrap().abs() < 1e-12);
/// ```
pub fn eta_squared(groups: &[Vec<f64>]) -> Option<f64> {
    let (ss_b, ss_t, _, _) = one_way_ss(groups)?;
    if ss_t == 0.0 || !ss_t.is_finite() {
        return None;
    }
    let v = (ss_b / ss_t).clamp(0.0, 1.0);
    v.is_finite().then_some(v)
}

// ──────────────────── significance: t and F ────────────────────

/// Outcome of a t-test: the statistic, its degrees of freedom, and the
/// two-tailed p-value.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TTest {
    /// The t statistic (signed).
    pub t: f64,
    /// Degrees of freedom — fractional for Welch's test.
    pub df: f64,
    /// Two-tailed p-value, `P(|T| ≥ |t|)`.
    pub p_two_tailed: f64,
}

/// Outcome of a one-way ANOVA: the F ratio, both degrees of freedom, the
/// upper-tail p-value, and the η² effect size alongside it.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Anova {
    /// The F ratio, `MS_between / MS_within`.
    pub f: f64,
    /// Between-groups degrees of freedom, `k − 1`.
    pub df_between: f64,
    /// Within-groups degrees of freedom, `N − k`.
    pub df_within: f64,
    /// Upper-tail p-value, `P(F ≥ f)`.
    pub p: f64,
    /// Proportion of variance explained — see [`eta_squared`].
    pub eta_squared: f64,
}

/// One-sample t-test of `H₀: mean(x) = mu0`.
///
/// `t = (x̄ − μ₀)/(s/√n)` with `df = n−1`, `s` the unbiased sample SD.
/// Returns `None` if `n < 2`, the input is non-finite, or the sample variance
/// is zero (every value identical → `t` undefined).
///
/// ```
/// use jc::stats::t_test_one_sample;
/// let x = [5.1, 4.9, 5.0, 5.2, 4.8];
/// let r = t_test_one_sample(&x, 5.0).unwrap();
/// assert!(r.p_two_tailed > 0.5); // sample mean is 5.0 → nothing to reject
/// ```
pub fn t_test_one_sample(x: &[f64], mu0: f64) -> Option<TTest> {
    if x.len() < 2 || !all_finite(x) || !mu0.is_finite() {
        return None;
    }
    let n = x.len() as f64;
    let m = mean(x)?;
    let var = sample_var(x)?;
    if var <= 0.0 || !var.is_finite() {
        return None;
    }
    let se = (var / n).sqrt();
    if se == 0.0 || !se.is_finite() {
        return None;
    }
    let t = (m - mu0) / se;
    let df = n - 1.0;
    let p = t_two_tailed_p(t, df)?;
    t.is_finite().then_some(TTest {
        t,
        df,
        p_two_tailed: p,
    })
}

/// Paired t-test — the one-sample test on the within-pair differences.
///
/// Returns `None` if lengths differ, `n < 2`, input is non-finite, or every
/// difference is identical (zero variance).
pub fn t_test_paired(x: &[f64], y: &[f64]) -> Option<TTest> {
    if x.len() != y.len() || x.len() < 2 {
        return None;
    }
    if !all_finite(x) || !all_finite(y) {
        return None;
    }
    let d: Vec<f64> = x.iter().zip(y.iter()).map(|(&a, &b)| a - b).collect();
    if !all_finite(&d) {
        return None; // difference overflowed on large finite input
    }
    t_test_one_sample(&d, 0.0)
}

/// Welch's two-sample t-test — unequal variances **not** assumed.
///
/// This is the default two-sample test: it does not require the two groups to
/// share a variance, and its cost when they do is negligible. `df` is the
/// Welch–Satterthwaite fractional value.
///
/// Returns `None` if either group has `n < 2`, input is non-finite, or both
/// variances are zero.
///
/// ```
/// use jc::stats::t_test_welch;
/// let a = [1.0, 2.0, 3.0, 4.0];
/// let b = [11.0, 12.0, 13.0, 14.0];
/// let r = t_test_welch(&a, &b).unwrap();
/// assert!(r.p_two_tailed < 0.01); // well-separated groups
/// ```
pub fn t_test_welch(a: &[f64], b: &[f64]) -> Option<TTest> {
    if a.len() < 2 || b.len() < 2 || !all_finite(a) || !all_finite(b) {
        return None;
    }
    let (na, nb) = (a.len() as f64, b.len() as f64);
    let (va, vb) = (sample_var(a)?, sample_var(b)?);
    let (ma, mb) = (mean(a)?, mean(b)?);
    let sa = va / na;
    let sb = vb / nb;
    let denom = sa + sb;
    if denom <= 0.0 || !denom.is_finite() {
        return None;
    }
    let t = (ma - mb) / denom.sqrt();
    let df_num = denom * denom;
    let df_den = sa * sa / (na - 1.0) + sb * sb / (nb - 1.0);
    if df_den <= 0.0 || !df_den.is_finite() {
        return None;
    }
    let df = df_num / df_den;
    if !df.is_finite() || df <= 0.0 {
        return None;
    }
    let p = t_two_tailed_p(t, df)?;
    t.is_finite().then_some(TTest {
        t,
        df,
        p_two_tailed: p,
    })
}

/// Student's pooled two-sample t-test — equal variances **assumed**.
///
/// Prefer [`t_test_welch`] unless the equal-variance assumption is itself
/// justified; this form is provided because `t² = F` of the two-group ANOVA
/// and `η² = t²/(t²+df)` hold exactly for the pooled statistic (both asserted
/// in the tests), making it the bridge between the t and r families.
///
/// `df = n_a + n_b − 2`. Returns `None` if either group has `n < 2`, input is
/// non-finite, or the pooled variance is zero.
pub fn t_test_student(a: &[f64], b: &[f64]) -> Option<TTest> {
    if a.len() < 2 || b.len() < 2 || !all_finite(a) || !all_finite(b) {
        return None;
    }
    let (na, nb) = (a.len() as f64, b.len() as f64);
    let (va, vb) = (sample_var(a)?, sample_var(b)?);
    let (ma, mb) = (mean(a)?, mean(b)?);
    let df = na + nb - 2.0;
    let pooled = ((na - 1.0) * va + (nb - 1.0) * vb) / df;
    if pooled <= 0.0 || !pooled.is_finite() {
        return None;
    }
    let se = (pooled * (1.0 / na + 1.0 / nb)).sqrt();
    if se == 0.0 || !se.is_finite() {
        return None;
    }
    let t = (ma - mb) / se;
    let p = t_two_tailed_p(t, df)?;
    t.is_finite().then_some(TTest {
        t,
        df,
        p_two_tailed: p,
    })
}

/// One-way ANOVA over `k ≥ 2` groups: `F = MS_between / MS_within`, with the
/// upper-tail p-value and the [`eta_squared`] effect size in the same result.
///
/// Returns `None` on fewer than 2 groups, any empty group, non-finite input,
/// `N ≤ k` (no within-group degrees of freedom), or zero within-group
/// variance (`F` undefined — every group is internally constant).
///
/// ```
/// use jc::stats::anova_one_way;
/// let g = vec![vec![1.0, 2.0, 3.0], vec![7.0, 8.0, 9.0]];
/// let r = anova_one_way(&g).unwrap();
/// assert!(r.p < 0.01 && r.eta_squared > 0.9);
/// ```
pub fn anova_one_way(groups: &[Vec<f64>]) -> Option<Anova> {
    let (ss_b, ss_t, k, n_total) = one_way_ss(groups)?;
    if n_total <= k {
        return None; // no within-group df
    }
    let df_b = (k - 1) as f64;
    let df_w = (n_total - k) as f64;
    let ss_w = ss_t - ss_b;
    if ss_w <= 0.0 || !ss_w.is_finite() {
        // ≤ 0 → no within-group variance (F undefined / degenerate).
        return None;
    }
    let ms_b = ss_b / df_b;
    let ms_w = ss_w / df_w;
    if ms_w <= 0.0 || !ms_w.is_finite() {
        return None;
    }
    let f = ms_b / ms_w;
    if !f.is_finite() {
        return None;
    }
    let p = f_upper_p(f, df_b, df_w)?;
    if ss_t == 0.0 || !ss_t.is_finite() {
        return None;
    }
    Some(Anova {
        f,
        df_between: df_b,
        df_within: df_w,
        p,
        eta_squared: (ss_b / ss_t).clamp(0.0, 1.0),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::reliability::cronbach_alpha;

    fn approx(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() < tol
    }

    // ───────────────────────────── κ ─────────────────────────────

    #[test]
    fn kappa_perfect_and_chance() {
        // Perfect agreement, both categories used → κ = 1.
        let a = [0usize, 1, 0, 1, 1, 0];
        assert!(approx(cohen_kappa(&a, &a).unwrap(), 1.0, 1e-12));

        // Constructed exact-chance case: rater A = [0,0,1,1], rater B = [0,1,0,1].
        // p_o = 2/4 = 0.5; marginals are both (0.5, 0.5) → p_e = 0.5 → κ = 0.
        let x = [0usize, 0, 1, 1];
        let y = [0usize, 1, 0, 1];
        assert!(approx(cohen_kappa(&x, &y).unwrap(), 0.0, 1e-12));
    }

    #[test]
    fn kappa_textbook_value() {
        // Classic 2×2: a=20 (both yes), b=5, c=10, d=15; N=50.
        // p_o = (20+15)/50 = 0.70
        // marginals A: yes 25/50 = .5, no .5 ; B: yes 30/50 = .6, no .4
        // p_e = .5*.6 + .5*.4 = 0.50  →  κ = (.70-.50)/(1-.50) = 0.40
        let mut x = Vec::new();
        let mut y = Vec::new();
        for _ in 0..20 {
            x.push(1usize);
            y.push(1usize);
        } // both yes
        for _ in 0..5 {
            x.push(1);
            y.push(0);
        } // A yes, B no
        for _ in 0..10 {
            x.push(0);
            y.push(1);
        } // A no, B yes
        for _ in 0..15 {
            x.push(0);
            y.push(0);
        } // both no
        let k = cohen_kappa(&x, &y).unwrap();
        assert!(approx(k, 0.40, 1e-12), "κ was {k}, expected 0.40");
    }

    #[test]
    fn kappa_below_chance_is_negative() {
        // Systematic disagreement → worse than chance.
        let x = [0usize, 0, 1, 1];
        let y = [1usize, 1, 0, 0];
        let k = cohen_kappa(&x, &y).unwrap();
        assert!(
            k < 0.0,
            "systematic disagreement should give κ < 0, got {k}"
        );
    }

    #[test]
    fn kappa_is_not_raw_agreement() {
        // Anti-vacuity: κ must DIFFER from p_o whenever chance agreement is
        // non-trivial. Here p_o = 0.8 but the marginals are lopsided, so κ is
        // substantially lower — a κ that merely echoed p_o would fail this.
        let x = [1usize, 1, 1, 1, 1, 1, 1, 1, 0, 0];
        let y = [1usize, 1, 1, 1, 1, 1, 1, 0, 1, 0];
        let p_o = 0.8;
        let k = cohen_kappa(&x, &y).unwrap();
        assert!(
            (k - p_o).abs() > 0.2,
            "κ={k} should be far from raw agreement {p_o}"
        );
    }

    #[test]
    fn kappa_degenerate_returns_none() {
        assert_eq!(cohen_kappa(&[], &[]), None); // empty
        assert_eq!(cohen_kappa(&[1, 2], &[1]), None); // ragged
                                                      // Both raters constant on ONE category → p_e = 1 → undefined.
        assert_eq!(cohen_kappa(&[3usize, 3, 3], &[3usize, 3, 3]), None);
    }

    // ───────────────────────────── ω ─────────────────────────────

    /// Exactly-congeneric fixture with UNEQUAL loadings, built so that the
    /// in-sample residuals are exactly orthogonal to the factor and to each
    /// other. Factor F=[1,−1,1,−1]; e1=[1,1,−1,−1]; e2=[1,−1,−1,1]; e3=0.
    /// item_i = λ_i·F + e_i with λ = (2, 3, 1).
    fn congeneric_items() -> Vec<Vec<f64>> {
        vec![
            vec![3.0, -1.0, 1.0, -3.0], // 2F + e1
            vec![4.0, -4.0, 2.0, -2.0], // 3F + e2
            vec![1.0, -1.0, 1.0, -1.0], // 1F
        ]
    }

    #[test]
    fn omega_hand_computed_value() {
        // Sample covariances (divisor n−1=3): σ12=8, σ13=8/3, σ23=4,
        // σ11=20/3, σ22=40/3, σ33=4/3.
        // λ1²=σ12σ13/σ23=16/3, λ2²=σ12σ23/σ13=12, λ3²=σ13σ23/σ12=4/3.
        // Σλ = 4/√3 + √12 + 2/√3 = 6.9282… → (Σλ)² = 48.
        // ψ = (20/3−16/3, 40/3−12, 4/3−4/3) = (4/3, 4/3, 0) → Σψ = 8/3.
        // ω = 48 / (48 + 8/3) = 0.9473684…
        let w = omega_total(&congeneric_items()).unwrap();
        assert!(approx(w, 0.947_368_42, 1e-8), "ω was {w}");
    }

    #[test]
    fn omega_exceeds_alpha_when_loadings_differ() {
        // THE reason ω exists. Same data, hand-computed α = 1.5·(1 − 16/38)
        // = 0.8684210…; ω = 0.9473684… → α understates reliability.
        let items = congeneric_items();
        let a = cronbach_alpha(&items).unwrap();
        let w = omega_total(&items).unwrap();
        assert!(approx(a, 0.868_421_05, 1e-8), "α was {a}");
        assert!(w > a, "ω={w} must exceed α={a} for unequal loadings");
    }

    #[test]
    fn omega_equals_alpha_under_tau_equivalence() {
        // Equal loadings (λ = 2 for all three items), residuals orthogonal:
        // items = 2F + e_i. Tau-equivalent → α is unbiased → ω ≈ α.
        // F=[1,−1,1,−1]; e1=[1,1,−1,−1]; e2=[1,−1,−1,1]; e3=[−2,0,2,0]…
        // use e3 = [0,0,0,0] scaled residual-free item plus matched noise:
        let f = [1.0, -1.0, 1.0, -1.0];
        let e = [
            [1.0, 1.0, -1.0, -1.0],
            [1.0, -1.0, -1.0, 1.0],
            [0.0, 0.0, 0.0, 0.0],
        ];
        let items: Vec<Vec<f64>> = (0..3)
            .map(|i| (0..4).map(|s| 2.0 * f[s] + e[i][s]).collect())
            .collect();
        let a = cronbach_alpha(&items).unwrap();
        let w = omega_total(&items).unwrap();
        assert!(
            approx(w, a, 1e-9),
            "under tau-equivalence ω={w} should equal α={a}"
        );
    }

    #[test]
    fn omega_zero_residual_model_is_one_not_none() {
        // Regression: items that are exact multiples of one factor have ψ = 0
        // in real arithmetic but land a few ulps NEGATIVE in f64. A bare
        // `psi < 0.0` Heywood test rejected this valid model as misfit — the
        // doctest caught it. The guard is now relative-tolerance based.
        let items = vec![
            vec![1.0, 2.0, 3.0, 4.0],
            vec![2.0, 4.0, 6.0, 8.0],
            vec![3.0, 6.0, 9.0, 12.0],
        ];
        let w = omega_total(&items).unwrap();
        assert!(approx(w, 1.0, 1e-9), "zero-residual ω should be 1, got {w}");
    }

    #[test]
    fn omega_heywood_case_still_returns_none() {
        // Can-it-fire twin of the tolerance relaxation above: a REAL Heywood
        // violation (larger than rounding) must still be rejected, or the
        // relaxation silently disabled the guard.
        // items (mean-zero): σ12 = 14/3, σ13 = 10/3, σ23 = 2, σ11 = 20/3.
        // λ1² = σ12·σ13/σ23 = 70/9 ≈ 7.778 > σ11 ≈ 6.667 → ψ1 ≈ −1.111.
        let items = vec![
            vec![3.0, 1.0, -1.0, -3.0],
            vec![2.0, 1.0, -1.0, -2.0],
            vec![2.0, -1.0, 1.0, -2.0],
        ];
        assert_eq!(omega_total(&items), None);
    }

    #[test]
    fn omega_degenerate_returns_none() {
        // k < 3 → single-factor model unidentified.
        assert_eq!(
            omega_total(&[vec![1.0, 2.0, 3.0], vec![1.0, 2.0, 3.0]]),
            None
        );
        // Ragged.
        assert_eq!(
            omega_total(&[vec![1.0, 2.0], vec![1.0, 2.0], vec![1.0]]),
            None
        );
        // n < 2.
        assert_eq!(omega_total(&[vec![1.0], vec![2.0], vec![3.0]]), None);
        // Non-finite.
        assert_eq!(
            omega_total(&[
                vec![1.0, 2.0, f64::NAN],
                vec![1.0, 2.0, 3.0],
                vec![2.0, 4.0, 6.0]
            ]),
            None
        );
    }

    // ───────────────────────────── φ ─────────────────────────────

    #[test]
    fn phi_is_pearson_on_binaries() {
        // The delegation is the point: φ must equal Pearson on the 0/1 coding,
        // computed independently here through the already-proven function.
        let x = [true, true, false, true, false, false, true, false];
        let y = [true, false, false, true, true, false, true, false];
        let xf: Vec<f64> = x.iter().map(|&v| if v { 1.0 } else { 0.0 }).collect();
        let yf: Vec<f64> = y.iter().map(|&v| if v { 1.0 } else { 0.0 }).collect();
        let expect = crate::reliability::pearson(&xf, &yf).unwrap();
        assert!(approx(phi(&x, &y).unwrap(), expect, 1e-15));
    }

    #[test]
    fn phi_textbook_2x2() {
        // a=4 (1,1), b=1 (1,0), c=1 (0,1), d=4 (0,0); N=10.
        // φ = (ad − bc)/√((a+b)(c+d)(a+c)(b+d)) = (16−1)/√(5·5·5·5) = 15/25 = 0.6
        let x = [
            true, true, true, true, true, false, false, false, false, false,
        ];
        let y = [
            true, true, true, true, false, true, false, false, false, false,
        ];
        let p = phi(&x, &y).unwrap();
        assert!(approx(p, 0.6, 1e-12), "φ was {p}, expected 0.6");
    }

    #[test]
    fn phi_degenerate_returns_none() {
        assert_eq!(phi(&[true, true], &[true, true]), None); // constant → no variance
        assert_eq!(phi(&[true, false], &[true]), None); // ragged
    }

    // ────────────────────────── R / R² ──────────────────────────

    #[test]
    fn r_squared_with_one_predictor_equals_pearson_squared() {
        // Cross-identity against already-proven code — the strongest available
        // check on the regression path.
        let y = vec![2.0, 4.0, 5.0, 4.0, 5.0, 7.0, 8.0];
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0];
        let r = crate::reliability::pearson(&y, &x).unwrap();
        let r2 = multiple_r_squared(&y, std::slice::from_ref(&x)).unwrap();
        assert!(approx(r2, r * r, 1e-12), "R²={r2} vs r²={}", r * r);
        assert!(approx(multiple_r(&y, &[x]).unwrap(), r.abs(), 1e-12));
    }

    #[test]
    fn r_squared_exact_fit_is_one_and_partial_fit_is_between() {
        // Exact linear combination → 1.
        let y = vec![3.0, 5.0, 7.0, 9.0, 11.0, 13.0];
        let p1 = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        assert!(approx(
            multiple_r_squared(&y, std::slice::from_ref(&p1)).unwrap(),
            1.0,
            1e-9
        ));

        // Anti-vacuity: a noisy second criterion must land STRICTLY inside
        // (0,1) — an R² that always returned 1 (or 0) would fail this.
        let noisy = vec![1.0, 3.0, 2.0, 6.0, 4.0, 9.0];
        let r2 = multiple_r_squared(&noisy, &[p1]).unwrap();
        assert!(r2 > 0.05 && r2 < 0.99, "expected partial fit, got R²={r2}");
    }

    #[test]
    fn r_squared_never_decreases_when_a_predictor_is_added() {
        let y = vec![1.0, 3.0, 2.0, 6.0, 4.0, 9.0, 7.0, 11.0];
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let b = vec![2.0, 1.0, 4.0, 3.0, 6.0, 5.0, 8.0, 7.0];
        let one = multiple_r_squared(&y, std::slice::from_ref(&a)).unwrap();
        let two = multiple_r_squared(&y, &[a, b]).unwrap();
        assert!(two >= one - 1e-12, "R² fell from {one} to {two}");
    }

    #[test]
    fn r_squared_degenerate_returns_none() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        assert_eq!(
            multiple_r_squared(&[1.0, 1.0, 1.0, 1.0], std::slice::from_ref(&x)),
            None
        ); // constant y
        assert_eq!(multiple_r_squared(&[1.0, 2.0], &[vec![1.0, 2.0]]), None); // n < k+2
        assert_eq!(multiple_r_squared(&[1.0, 2.0, 3.0, 4.0], &[]), None); // no predictors
        assert_eq!(
            multiple_r_squared(&[1.0, 2.0, 3.0, 4.0], &[vec![1.0, 2.0]]),
            None
        ); // ragged
           // Collinear predictors → singular normal equations.
        let dup = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        assert_eq!(
            multiple_r_squared(&[1.0, 3.0, 2.0, 5.0, 4.0], &[dup.clone(), dup]),
            None
        );
        // The subtler collinearity, and the one that broke this function's own
        // first doc example: `b = a + 1` is not a duplicate column, but the
        // design carries an INTERCEPT, so [1, a, a+1] is still rank-deficient.
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let b: Vec<f64> = a.iter().map(|v| v + 1.0).collect();
        assert_eq!(
            multiple_r_squared(&[3.0, 5.0, 7.0, 9.0, 11.0], &[a, b]),
            None
        );
        // Non-finite.
        assert_eq!(multiple_r_squared(&[1.0, 2.0, f64::NAN, 4.0], &[x]), None);
    }

    // ─────────────────────────── η² / F ───────────────────────────

    #[test]
    fn eta_squared_bounds_are_reached() {
        // 0: identical group means.
        let same = vec![vec![1.0, 2.0, 3.0], vec![3.0, 2.0, 1.0]];
        assert!(approx(eta_squared(&same).unwrap(), 0.0, 1e-12));
        // 1: zero within-group variance, different means → all variance between.
        let split = vec![vec![1.0, 1.0, 1.0], vec![5.0, 5.0, 5.0]];
        assert!(approx(eta_squared(&split).unwrap(), 1.0, 1e-12));
    }

    #[test]
    fn eta_squared_equals_r_squared_on_a_dummy() {
        // Two groups: η² over the grouping IS R² of the pooled values on a 0/1
        // dummy predictor. Two independent code paths, one quantity.
        let g0 = vec![1.0, 2.0, 4.0, 3.0];
        let g1 = vec![7.0, 6.0, 9.0, 8.0];
        let eta = eta_squared(&[g0.clone(), g1.clone()]).unwrap();

        let mut y = g0.clone();
        y.extend(g1.iter().copied());
        let dummy: Vec<f64> = (0..g0.len())
            .map(|_| 0.0)
            .chain((0..g1.len()).map(|_| 1.0))
            .collect();
        let r2 = multiple_r_squared(&y, &[dummy]).unwrap();
        assert!(approx(eta, r2, 1e-12), "η²={eta} vs R²={r2}");
    }

    #[test]
    fn two_group_anova_f_equals_pooled_t_squared_and_eta_matches() {
        // The bridge between the t and r families, asserted both ways:
        //   F = t²  and  η² = t²/(t² + df).
        let a = vec![1.0, 2.0, 4.0, 3.0, 5.0];
        let b = vec![7.0, 6.0, 9.0, 8.0, 11.0];
        let t = t_test_student(&a, &b).unwrap();
        let av = anova_one_way(&[a, b]).unwrap();
        assert!(
            approx(av.f, t.t * t.t, 1e-9),
            "F={} vs t²={}",
            av.f,
            t.t * t.t
        );
        let expect_eta = t.t * t.t / (t.t * t.t + t.df);
        assert!(
            approx(av.eta_squared, expect_eta, 1e-12),
            "η²={} vs t²/(t²+df)={expect_eta}",
            av.eta_squared
        );
        // …and the two p-values agree, since F(1,df) is exactly t(df)².
        assert!(approx(av.p, t.p_two_tailed, 1e-9));
    }

    #[test]
    fn anova_p_discriminates_and_stays_silent() {
        // Can-it-fire: well-separated groups → small p, large η².
        let sep = vec![vec![1.0, 2.0, 3.0], vec![20.0, 21.0, 22.0]];
        let fired = anova_one_way(&sep).unwrap();
        assert!(fired.p < 0.001 && fired.eta_squared > 0.9);
        // Can-it-stay-silent, on NON-TRIVIAL input: two interleaved samples
        // from the same distribution → large p, small η². (An empty or
        // constant input would prove nothing.)
        let same = vec![vec![4.0, 9.0, 2.0, 7.0, 5.0], vec![5.0, 3.0, 8.0, 6.0, 4.0]];
        let quiet = anova_one_way(&same).unwrap();
        assert!(
            quiet.p > 0.3 && quiet.eta_squared < 0.2,
            "expected silence, got p={} η²={}",
            quiet.p,
            quiet.eta_squared
        );
    }

    #[test]
    fn eta_and_anova_degenerate_return_none() {
        assert_eq!(eta_squared(&[vec![1.0, 2.0]]), None); // < 2 groups
        assert_eq!(eta_squared(&[vec![1.0, 2.0], vec![]]), None); // empty group
        assert_eq!(eta_squared(&[vec![2.0, 2.0], vec![2.0, 2.0]]), None); // zero total variance
        assert_eq!(anova_one_way(&[vec![1.0, 1.0], vec![5.0, 5.0]]), None); // no within variance
        assert_eq!(
            anova_one_way(&[vec![1.0, f64::INFINITY], vec![3.0, 4.0]]),
            None
        );
    }

    // ─────────────────────────── t-tests ───────────────────────────

    #[test]
    fn one_sample_t_textbook_value() {
        // x = [1..5]: mean 3, sample var 10/4 = 2.5, se = √(2.5/5) = 0.7071068.
        // Against μ₀ = 2 → t = 1/0.7071068 = √2 = 1.4142136, df = 4.
        //
        // The p reference is derived in CLOSED FORM, not read back from this
        // code: p = I_x(df/2, 1/2) with df = 4, x = df/(df+t²) = 4/6 = 2/3, so
        //   ∫₀^x t(1−t)^(−1/2) dt = 4/3 − 2√(1−x) + (2/3)(1−x)^(3/2)
        //                         = 4/3 − 2/√3 + (2/3)·3^(−3/2) = 0.30693287…
        //   B(2, ½) = Γ(2)Γ(½)/Γ(5/2) = 4/3
        //   p = 0.30693287… / (4/3) = 0.23019965…
        let x = [1.0, 2.0, 3.0, 4.0, 5.0];
        let r = t_test_one_sample(&x, 2.0).unwrap();
        assert!(
            approx(r.t, std::f64::consts::SQRT_2, 1e-12),
            "t was {}",
            r.t
        );
        assert!(approx(r.df, 4.0, 1e-12));
        assert!(
            approx(r.p_two_tailed, 0.230_199_65, 1e-7),
            "p was {}",
            r.p_two_tailed
        );
    }

    #[test]
    fn welch_t_reference_value() {
        // a = [27,20,32,28,25]: mean 26.4, Σd² = 77.2, var = 19.3
        // b = [22,15,25,17,20]: mean 19.8, Σd² = 62.8, var = 15.7
        // Welch: t = 6.6/√(19.3/5 + 15.7/5) = 6.6/√7 = 2.4945652…
        // df = 7²/((3.86²/4) + (3.14²/4)) = 49/(3.7249 + 2.4649)
        //    = 49/6.1898 = 7.9162493…
        //
        // `t` and `df` are pinned to those closed-form values. The p-value is
        // only BOUNDED here, deliberately: with fractional df there is no hand
        // closed form, and pinning a number read back from this code would be
        // an assertion implied by the code it tests. The tail function itself
        // is pinned against independent critical values in
        // `t_distribution_tail_matches_reference`.
        let a = [27.0, 20.0, 32.0, 28.0, 25.0];
        let b = [22.0, 15.0, 25.0, 17.0, 20.0];
        let r = t_test_welch(&a, &b).unwrap();
        assert!(approx(r.t, 6.6 / 7.0f64.sqrt(), 1e-12), "t was {}", r.t);
        assert!(approx(r.df, 49.0 / 6.1898, 1e-9), "df was {}", r.df);
        assert!(
            r.p_two_tailed > 0.03 && r.p_two_tailed < 0.05,
            "p was {}, expected in (0.03, 0.05)",
            r.p_two_tailed
        );
    }

    #[test]
    fn paired_t_is_one_sample_on_differences() {
        let x = [5.0, 6.0, 7.0, 8.0, 9.0];
        let y = [4.0, 6.0, 5.0, 9.0, 7.0];
        let d: Vec<f64> = x.iter().zip(y.iter()).map(|(&a, &b)| a - b).collect();
        let p = t_test_paired(&x, &y).unwrap();
        let o = t_test_one_sample(&d, 0.0).unwrap();
        assert!(approx(p.t, o.t, 1e-15) && approx(p.df, o.df, 1e-15));
        assert!(approx(p.p_two_tailed, o.p_two_tailed, 1e-15));
    }

    #[test]
    fn welch_and_student_differ_under_unequal_variances() {
        // Anti-vacuity for the Welch/Student distinction: with very unequal
        // variances and unequal n, the two must NOT coincide. A Welch that
        // silently pooled would fail here.
        let a = [10.0, 10.1, 9.9, 10.05, 9.95, 10.02];
        let b = [1.0, 20.0, -5.0, 25.0, 0.0, 18.0];
        let w = t_test_welch(&a, &b).unwrap();
        let s = t_test_student(&a, &b).unwrap();
        assert!(
            (w.df - s.df).abs() > 1.0,
            "Welch df {} should differ from pooled df {}",
            w.df,
            s.df
        );
    }

    #[test]
    fn t_test_p_is_monotone_in_separation() {
        // A p-value that ignored the data would fail this: increasing the
        // separation must strictly decrease p.
        let base = [1.0, 2.0, 3.0, 4.0, 5.0];
        let near: Vec<f64> = base.iter().map(|v| v + 1.0).collect();
        let far: Vec<f64> = base.iter().map(|v| v + 10.0).collect();
        let p_near = t_test_welch(&base, &near).unwrap().p_two_tailed;
        let p_far = t_test_welch(&base, &far).unwrap().p_two_tailed;
        assert!(p_far < p_near, "p_far={p_far} should be < p_near={p_near}");
        assert!(p_near > 0.05, "adjacent means should not be significant");
    }

    #[test]
    fn t_tests_degenerate_return_none() {
        assert_eq!(t_test_one_sample(&[1.0], 0.0), None); // n < 2
        assert_eq!(t_test_one_sample(&[2.0, 2.0, 2.0], 0.0), None); // zero variance
        assert_eq!(t_test_paired(&[1.0, 2.0], &[1.0]), None); // ragged
        assert_eq!(t_test_paired(&[1.0, 2.0], &[0.0, 1.0]), None); // constant difference
        assert_eq!(t_test_welch(&[1.0], &[1.0, 2.0]), None); // n < 2
        assert_eq!(t_test_student(&[2.0, 2.0], &[2.0, 2.0]), None); // zero pooled variance
        assert_eq!(t_test_one_sample(&[1.0, f64::NAN, 3.0], 0.0), None);
        assert_eq!(t_test_welch(&[1.0, 2.0], &[f64::INFINITY, 3.0]), None);
    }

    // ────────────────────── numerics under the hood ──────────────────────

    #[test]
    fn incomplete_beta_matches_reference_values() {
        // I_x(a,b) reference values (R: pbeta).
        assert!(approx(reg_inc_beta(2.0, 3.0, 0.5).unwrap(), 0.687_5, 1e-9));
        assert!(approx(reg_inc_beta(0.5, 0.5, 0.5).unwrap(), 0.5, 1e-9));
        assert!(approx(reg_inc_beta(5.0, 1.0, 0.5).unwrap(), 0.031_25, 1e-9));
        // Boundaries and invalid domains.
        assert_eq!(reg_inc_beta(2.0, 3.0, 0.0), Some(0.0));
        assert_eq!(reg_inc_beta(2.0, 3.0, 1.0), Some(1.0));
        assert_eq!(reg_inc_beta(2.0, 3.0, 1.5), None);
        assert_eq!(reg_inc_beta(0.0, 3.0, 0.5), None);
    }

    #[test]
    fn t_distribution_tail_matches_reference() {
        // Two-tailed p at the classic 5% critical values (R: qt).
        assert!(approx(t_two_tailed_p(2.776_445, 4.0).unwrap(), 0.05, 1e-6));
        assert!(approx(t_two_tailed_p(2.228_139, 10.0).unwrap(), 0.05, 1e-6));
        assert!(approx(t_two_tailed_p(0.0, 5.0).unwrap(), 1.0, 1e-12));
    }

    #[test]
    fn f_distribution_tail_matches_reference() {
        // F(2,10) upper 5% critical value is 4.102821 (R: qf).
        assert!(approx(f_upper_p(4.102_821, 2.0, 10.0).unwrap(), 0.05, 1e-6));
        // F(1,4) at 7.708647 → 0.05.
        assert!(approx(f_upper_p(7.708_647, 1.0, 4.0).unwrap(), 0.05, 1e-6));
    }

    #[test]
    fn overflowing_large_finite_inputs_return_none_or_finite() {
        // Matches the reliability module's contract: astronomically large but
        // finite input must never yield Some(NaN).
        let big = 1e308;
        assert!(t_test_one_sample(&[big, -big, big], 0.0)
            .map(|r| r.t.is_finite() && r.p_two_tailed.is_finite())
            .unwrap_or(true));
        assert!(t_test_welch(&[big, -big, big], &[-big, big, -big])
            .map(|r| r.t.is_finite() && r.p_two_tailed.is_finite())
            .unwrap_or(true));
        let big_groups = vec![vec![big, -big, big], vec![-big, big, -big]];
        assert!(eta_squared(&big_groups)
            .map(|v| v.is_finite())
            .unwrap_or(true));
        assert!(anova_one_way(&big_groups)
            .map(|r| r.f.is_finite() && r.p.is_finite())
            .unwrap_or(true));
        let big_items = vec![
            vec![big, -big, big],
            vec![-big, big, -big],
            vec![big, big, -big],
        ];
        assert!(omega_total(&big_items)
            .map(|v| v.is_finite())
            .unwrap_or(true));
        assert!(
            multiple_r_squared(&[big, -big, big, big], &[vec![big, big, -big, -big]])
                .map(|v| v.is_finite())
                .unwrap_or(true)
        );
    }
}
