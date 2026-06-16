//! Reliability & validity statistics (zero-dep mirror of
//! `ndarray::hpc::reliability`) so the validation harness is self-contained.
//! Pearson `r`, tie-aware Spearman `ρ`, Cronbach `α`, ICC(2,1). All `f64`.
//! Formulas as documented in `METHODS.md §5`.
//!
//! Significance of any correlation here must use the **Jirak** `n^(p/2−1)` rate
//! (weakly-dependent contingencies), not IID — this module computes the point
//! estimates only.

/// Pearson linear correlation; 0 on degenerate input.
pub fn pearson(a: &[f64], b: &[f64]) -> f64 {
    let n = a.len();
    if n < 2 || b.len() != n {
        return 0.0;
    }
    let (ma, mb) = (mean(a), mean(b));
    let (mut sab, mut saa, mut sbb) = (0.0, 0.0, 0.0);
    for i in 0..n {
        let (da, db) = (a[i] - ma, b[i] - mb);
        sab += da * db;
        saa += da * da;
        sbb += db * db;
    }
    if saa < 1e-12 || sbb < 1e-12 {
        0.0
    } else {
        sab / (saa * sbb).sqrt()
    }
}

/// Tie-aware Spearman rank correlation.
pub fn spearman(a: &[f64], b: &[f64]) -> f64 {
    if a.len() != b.len() {
        return 0.0;
    }
    pearson(&average_ranks(a), &average_ranks(b))
}

/// Cronbach's α over `items` (each item is a column: one variable across all
/// subjects). `0` if `< 2` items / subjects or zero total variance.
pub fn cronbach_alpha(items: &[Vec<f64>]) -> f64 {
    let k = items.len();
    if k < 2 {
        return 0.0;
    }
    let n = items[0].len();
    if n < 2 || items.iter().any(|c| c.len() != n) {
        return 0.0;
    }
    let item_var_sum: f64 = items.iter().map(|c| pop_var(c)).sum();
    let total: Vec<f64> = (0..n).map(|s| items.iter().map(|c| c[s]).sum()).collect();
    let tv = pop_var(&total);
    if tv < 1e-12 {
        return 0.0;
    }
    (k as f64 / (k as f64 - 1.0)) * (1.0 - item_var_sum / tv)
}

/// ICC(2,1): two-way random, single rater, absolute agreement.
/// `ratings[r]` is rater `r`'s scores over the `n` subjects; `k` raters.
pub fn icc_a1(ratings: &[Vec<f64>]) -> f64 {
    let k = ratings.len();
    if k < 2 {
        return 0.0;
    }
    let n = ratings[0].len();
    if n < 2 || ratings.iter().any(|r| r.len() != n) {
        return 0.0;
    }
    let (kf, nf) = (k as f64, n as f64);
    let grand = ratings.iter().flat_map(|r| r.iter()).sum::<f64>() / (kf * nf);
    let row_mean = |s: usize| ratings.iter().map(|r| r[s]).sum::<f64>() / kf; // per subject
    let col_mean = |r: usize| ratings[r].iter().sum::<f64>() / nf; // per rater

    let mut ss_total = 0.0;
    for r in ratings {
        for &x in r {
            ss_total += (x - grand).powi(2);
        }
    }
    let ss_subj = kf * (0..n).map(|s| (row_mean(s) - grand).powi(2)).sum::<f64>();
    let ss_rater = nf * (0..k).map(|r| (col_mean(r) - grand).powi(2)).sum::<f64>();
    let ss_err = ss_total - ss_subj - ss_rater;

    let msr = ss_subj / (nf - 1.0);
    let msc = ss_rater / (kf - 1.0);
    let mse = ss_err / ((nf - 1.0) * (kf - 1.0));
    let denom = msr + (kf - 1.0) * mse + (kf / nf) * (msc - mse);
    if denom.abs() < 1e-12 {
        0.0
    } else {
        (msr - mse) / denom
    }
}

/// Z-score a column (mean 0, unit variance); leaves a constant column at 0.
pub fn zscore(x: &[f64]) -> Vec<f64> {
    let m = mean(x);
    let sd = pop_var(x).sqrt();
    if sd < 1e-12 {
        vec![0.0; x.len()]
    } else {
        x.iter().map(|&v| (v - m) / sd).collect()
    }
}

fn mean(x: &[f64]) -> f64 {
    x.iter().sum::<f64>() / x.len() as f64
}

fn pop_var(x: &[f64]) -> f64 {
    let m = mean(x);
    x.iter().map(|&v| (v - m).powi(2)).sum::<f64>() / x.len() as f64
}

fn average_ranks(x: &[f64]) -> Vec<f64> {
    let n = x.len();
    let mut idx: Vec<usize> = (0..n).collect();
    idx.sort_by(|&i, &j| x[i].partial_cmp(&x[j]).unwrap_or(std::cmp::Ordering::Equal));
    let mut ranks = vec![0.0; n];
    let mut i = 0;
    while i < n {
        let mut j = i;
        while j + 1 < n && (x[idx[j + 1]] - x[idx[i]]).abs() < 1e-12 {
            j += 1;
        }
        let avg = ((i + j) as f64) / 2.0 + 1.0; // 1-based average rank
        for &id in &idx[i..=j] {
            ranks[id] = avg;
        }
        i = j + 1;
    }
    ranks
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pearson_extremes() {
        let a = vec![1.0, 2.0, 3.0, 4.0];
        assert!((pearson(&a, &a) - 1.0).abs() < 1e-12);
        let neg: Vec<f64> = a.iter().map(|x| -x).collect();
        assert!((pearson(&a, &neg) + 1.0).abs() < 1e-12);
    }

    #[test]
    fn spearman_is_monotone_invariant() {
        let a: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let b: Vec<f64> = a.iter().map(|&x| x.exp()).collect(); // monotone
        assert!((spearman(&a, &b) - 1.0).abs() < 1e-12);
    }

    #[test]
    fn cronbach_high_for_redundant_items() {
        let base = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let items = vec![base.clone(), base.clone(), base.clone()];
        assert!(cronbach_alpha(&items) > 0.99, "identical items → α≈1");
    }

    #[test]
    fn icc_catches_bias_that_pearson_misses() {
        // A constant offset: perfectly correlated (Pearson=1) but NOT in
        // absolute agreement (ICC < 1) — the bias-detection signature.
        let r1 = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let r2: Vec<f64> = r1.iter().map(|x| x + 10.0).collect();
        assert!((pearson(&r1, &r2) - 1.0).abs() < 1e-9);
        let icc = icc_a1(&[r1, r2]);
        assert!(icc < 0.5, "ICC must penalize the +10 offset, got {icc}");
    }

    #[test]
    fn icc_one_for_identical_raters() {
        let r = vec![2.0, 5.0, 1.0, 8.0, 3.0];
        let icc = icc_a1(&[r.clone(), r.clone()]);
        assert!(icc > 0.99, "identical raters → ICC≈1, got {icc}");
    }
}
