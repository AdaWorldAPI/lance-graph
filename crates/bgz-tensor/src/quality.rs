//! Quality metrics for bgz-tensor compiled attention.
//!
//! Measures how well the palette-compiled attention reproduces
//! the ground-truth full-precision attention scores. The key metric
//! is Pearson correlation (ρ): how well does distance-table attention
//! rank pairs in the same order as dot-product attention?
//!
//! Target: ρ > 0.95 for paper, ρ > 0.99 for product.

/// Pearson correlation between two f64 vectors.
///
/// Returns ρ ∈ [-1, 1]. +1 = perfect positive correlation.
pub fn pearson(x: &[f64], y: &[f64]) -> f64 {
    let n = x.len().min(y.len());
    if n < 2 {
        return 0.0;
    }

    let mean_x = x[..n].iter().sum::<f64>() / n as f64;
    let mean_y = y[..n].iter().sum::<f64>() / n as f64;

    let mut cov = 0.0f64;
    let mut var_x = 0.0f64;
    let mut var_y = 0.0f64;

    for i in 0..n {
        let dx = x[i] - mean_x;
        let dy = y[i] - mean_y;
        cov += dx * dy;
        var_x += dx * dx;
        var_y += dy * dy;
    }

    let denom = (var_x * var_y).sqrt();
    if denom < 1e-12 {
        0.0
    } else {
        cov / denom
    }
}

/// Spearman rank correlation.
///
/// Like Pearson but on ranks. Measures whether the ordering is preserved,
/// not the exact values. More relevant for attention (we care about which
/// pairs have HIGH attention, not the exact score).
pub fn spearman(x: &[f64], y: &[f64]) -> f64 {
    let n = x.len().min(y.len());
    if n < 2 {
        return 0.0;
    }

    let rank_x = ranks(&x[..n]);
    let rank_y = ranks(&y[..n]);
    pearson(&rank_x, &rank_y)
}

/// Convert values to ranks (1-based, average for ties).
fn ranks(values: &[f64]) -> Vec<f64> {
    let n = values.len();
    let mut indexed: Vec<(usize, f64)> = values.iter().copied().enumerate().collect();
    indexed.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

    let mut result = vec![0.0f64; n];
    let mut i = 0;
    while i < n {
        let mut j = i + 1;
        while j < n && (indexed[j].1 - indexed[i].1).abs() < 1e-12 {
            j += 1;
        }
        // Average rank for tied group
        let avg_rank = (i + j) as f64 / 2.0 + 0.5;
        for k in i..j {
            result[indexed[k].0] = avg_rank;
        }
        i = j;
    }
    result
}

/// Mean Absolute Error between two vectors.
pub fn mae(x: &[f64], y: &[f64]) -> f64 {
    let n = x.len().min(y.len());
    if n == 0 {
        return 0.0;
    }
    x[..n]
        .iter()
        .zip(y[..n].iter())
        .map(|(a, b)| (a - b).abs())
        .sum::<f64>()
        / n as f64
}

/// Root Mean Square Error.
pub fn rmse(x: &[f64], y: &[f64]) -> f64 {
    let n = x.len().min(y.len());
    if n == 0 {
        return 0.0;
    }
    let mse: f64 = x[..n]
        .iter()
        .zip(y[..n].iter())
        .map(|(a, b)| (a - b).powi(2))
        .sum::<f64>()
        / n as f64;
    mse.sqrt()
}

/// Top-K recall: of the K highest-attention pairs in ground truth,
/// how many are also in the top-K of the compiled model?
///
/// This is the most important metric — we need to get the TOP attention
/// right, not every pair. Missing a low-attention pair is fine.
pub fn top_k_recall(ground_truth: &[f64], compiled: &[f64], k: usize) -> f64 {
    let n = ground_truth.len().min(compiled.len());
    let k = k.min(n);
    if k == 0 {
        return 1.0;
    }

    // Find top-K indices in ground truth
    let mut gt_indexed: Vec<(usize, f64)> = ground_truth[..n]
        .iter()
        .copied()
        .enumerate()
        .collect();
    gt_indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    let gt_top: std::collections::HashSet<usize> =
        gt_indexed[..k].iter().map(|&(i, _)| i).collect();

    // Find top-K indices in compiled
    let mut comp_indexed: Vec<(usize, f64)> = compiled[..n]
        .iter()
        .copied()
        .enumerate()
        .collect();
    comp_indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    let comp_top: std::collections::HashSet<usize> =
        comp_indexed[..k].iter().map(|&(i, _)| i).collect();

    // Intersection / K
    gt_top.intersection(&comp_top).count() as f64 / k as f64
}

/// Complete quality report for a compiled attention head.
#[derive(Clone, Debug)]
pub struct QualityReport {
    /// Pearson correlation of raw distances vs dot products.
    pub pearson_rho: f64,
    /// Spearman rank correlation.
    pub spearman_rho: f64,
    /// Mean Absolute Error.
    pub mae: f64,
    /// Root Mean Square Error.
    pub rmse: f64,
    /// Top-K recall at various K values.
    pub top_k_recall: Vec<(usize, f64)>,
    /// Compression ratio (input bytes / compiled bytes).
    pub compression_ratio: f64,
    /// Number of pairs evaluated.
    pub n_pairs: usize,
}

impl QualityReport {
    /// Compute from ground-truth dot products and compiled distances.
    ///
    /// Note: distances are NEGATED for correlation (lower distance = higher attention).
    pub fn compute(
        ground_truth_dots: &[f64],
        compiled_distances: &[f64],
        input_bytes: usize,
        compiled_bytes: usize,
    ) -> Self {
        let n = ground_truth_dots.len().min(compiled_distances.len());

        // Negate distances for correlation (lower distance = higher similarity)
        let neg_distances: Vec<f64> = compiled_distances[..n]
            .iter()
            .map(|&d| -d)
            .collect();

        let gt = &ground_truth_dots[..n];

        let ks = [8, 16, 32, 64, 128];
        let top_k = ks
            .iter()
            .filter(|&&k| k <= n)
            .map(|&k| (k, top_k_recall(gt, &neg_distances, k)))
            .collect();

        QualityReport {
            pearson_rho: pearson(gt, &neg_distances),
            spearman_rho: spearman(gt, &neg_distances),
            mae: mae(gt, &neg_distances),
            rmse: rmse(gt, &neg_distances),
            top_k_recall: top_k,
            compression_ratio: input_bytes as f64 / compiled_bytes.max(1) as f64,
            n_pairs: n,
        }
    }

    /// Does this meet paper quality? (ρ > 0.95)
    pub fn is_paper_quality(&self) -> bool {
        self.pearson_rho > 0.95
    }

    /// Does this meet product quality? (ρ > 0.99)
    pub fn is_product_quality(&self) -> bool {
        self.pearson_rho > 0.99
    }

    /// Human-readable summary.
    pub fn summary(&self) -> String {
        let mut s = format!(
            "Quality Report ({} pairs, {:.0}× compression):\n\
             Pearson  ρ = {:.4} {}\n\
             Spearman ρ = {:.4}\n\
             MAE        = {:.4}\n\
             RMSE       = {:.4}",
            self.n_pairs,
            self.compression_ratio,
            self.pearson_rho,
            if self.is_product_quality() {
                "✓ PRODUCT"
            } else if self.is_paper_quality() {
                "✓ PAPER"
            } else {
                "✗ BELOW THRESHOLD"
            },
            self.spearman_rho,
            self.mae,
            self.rmse,
        );
        for &(k, recall) in &self.top_k_recall {
            s.push_str(&format!("\nTop-{:>3} recall = {:.3}", k, recall));
        }
        s
    }
}

/// Ground-truth dot product computation.
///
/// Computes Q[i] · K[j] for all pairs. This is what standard
/// attention does. We use it as the reference to measure bgz-tensor quality.
pub fn ground_truth_dots(
    q_weights: &[f32],
    k_weights: &[f32],
    n_rows: usize,
    n_cols: usize,
) -> Vec<f64> {
    let mut dots = Vec::with_capacity(n_rows * n_rows);
    for i in 0..n_rows {
        for j in 0..n_rows {
            let mut dot = 0.0f64;
            for c in 0..n_cols {
                dot += q_weights[i * n_cols + c] as f64 * k_weights[j * n_cols + c] as f64;
            }
            dots.push(dot);
        }
    }
    dots
}

// ═════════════════════════════════════════════════════════════════════
// Psychometric metrics for codec R&D bench (P8)
// ═════════════════════════════════════════════════════════════════════

/// Cronbach's alpha — internal consistency of k "items" (codec score vectors)
/// measured on the same n observations.
///
/// `items` is k columns × n rows: items[codec_idx][observation_idx].
/// Each "item" is one codec's pairwise score vector on the same row pairs.
///
/// α = (k / (k-1)) × (1 - Σ var(item_i) / var(total))
///
/// α ≥ 0.85 → codecs measure the same construct (redundant, keep cheapest)
/// α < 0.70 → codecs measure different constructs (both informative)
pub fn cronbach_alpha(items: &[Vec<f64>]) -> f64 {
    let k = items.len();
    if k < 2 { return 0.0; }
    let n = items[0].len();
    if n < 2 { return 0.0; }

    // Item variances
    let mut sum_item_var = 0.0f64;
    for item in items {
        let mean: f64 = item.iter().sum::<f64>() / n as f64;
        let var: f64 = item.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (n - 1) as f64;
        sum_item_var += var;
    }

    // Total score variance (sum across items per observation)
    let totals: Vec<f64> = (0..n).map(|obs| {
        items.iter().map(|item| item[obs]).sum::<f64>()
    }).collect();
    let total_mean: f64 = totals.iter().sum::<f64>() / n as f64;
    let total_var: f64 = totals.iter().map(|x| (x - total_mean).powi(2)).sum::<f64>() / (n - 1) as f64;

    if total_var < 1e-15 { return 0.0; }
    (k as f64 / (k - 1) as f64) * (1.0 - sum_item_var / total_var)
}

/// Cohen's kappa — agreement between two raters beyond chance.
///
/// `a` and `b` are categorical labels (e.g., argmax indices) for n items.
/// κ = (p_o - p_e) / (1 - p_e)
///   where p_o = observed agreement, p_e = expected agreement by chance.
///
/// κ ≥ 0.80 → almost perfect agreement
/// κ 0.60–0.80 → substantial
/// κ < 0.40 → fair or poor
pub fn cohens_kappa(a: &[usize], b: &[usize]) -> f64 {
    let n = a.len().min(b.len());
    if n == 0 { return 0.0; }

    // Count categories
    let max_cat = a.iter().chain(b.iter()).copied().max().unwrap_or(0) + 1;

    // Confusion matrix
    let mut conf = vec![vec![0usize; max_cat]; max_cat];
    for i in 0..n { conf[a[i].min(max_cat - 1)][b[i].min(max_cat - 1)] += 1; }

    // Observed agreement
    let p_o: f64 = (0..max_cat).map(|c| conf[c][c] as f64).sum::<f64>() / n as f64;

    // Expected agreement by chance (marginal products)
    let row_sums: Vec<f64> = (0..max_cat).map(|r| conf[r].iter().sum::<usize>() as f64).collect();
    let col_sums: Vec<f64> = (0..max_cat).map(|c| (0..max_cat).map(|r| conf[r][c]).sum::<usize>() as f64).collect();
    let p_e: f64 = (0..max_cat).map(|c| row_sums[c] * col_sums[c]).sum::<f64>() / (n as f64 * n as f64);

    if (1.0 - p_e).abs() < 1e-15 { return 1.0; } // perfect marginal agreement
    (p_o - p_e) / (1.0 - p_e)
}

/// Bias and variance decomposition of signed errors.
///
/// `errors` = predicted - ground_truth (signed).
/// Returns (bias, variance) where:
///   bias = mean(errors) — systematic over/under-estimation
///   variance = var(errors) — random scatter
///
/// If |bias| >> sqrt(variance): systematic problem (correction helps)
/// If |bias| << sqrt(variance): random noise (correction doesn't help, P5 finding)
pub fn bias_variance(errors: &[f64]) -> (f64, f64) {
    let n = errors.len();
    if n == 0 { return (0.0, 0.0); }
    let mean = errors.iter().sum::<f64>() / n as f64;
    let var = if n > 1 {
        errors.iter().map(|e| (e - mean).powi(2)).sum::<f64>() / (n - 1) as f64
    } else { 0.0 };
    (mean, var)
}

/// ICC(3,1) — intraclass correlation, consistency form.
///
/// Two "raters" (ground truth and codec) rate the same n targets.
/// ICC(3,1) = (MS_targets - MS_error) / (MS_targets + MS_error)
///
/// Equivalent to a two-way mixed model where raters are fixed.
/// Ranges [-1, 1]; ≥ 0.75 = good, ≥ 0.90 = excellent.
pub fn icc_3_1(ground_truth: &[f64], codec: &[f64]) -> f64 {
    let n = ground_truth.len().min(codec.len());
    if n < 2 { return 0.0; }

    // Two-way ANOVA decomposition (2 raters, n targets)
    let grand_mean = (ground_truth[..n].iter().sum::<f64>() + codec[..n].iter().sum::<f64>())
        / (2 * n) as f64;

    // Target means (average of GT and codec per target)
    let target_means: Vec<f64> = (0..n).map(|i| {
        (ground_truth[i] + codec[i]) / 2.0
    }).collect();

    // MS_targets = 2 × Σ (target_mean - grand_mean)² / (n-1)
    let ss_targets: f64 = target_means.iter()
        .map(|&m| (m - grand_mean).powi(2))
        .sum::<f64>();
    let ms_targets = 2.0 * ss_targets / (n - 1) as f64;

    // MS_error = Σ (x_ij - target_mean_i)² / n  (for 2 raters)
    let ss_error: f64 = (0..n).map(|i| {
        (ground_truth[i] - target_means[i]).powi(2)
        + (codec[i] - target_means[i]).powi(2)
    }).sum::<f64>();
    let ms_error = ss_error / n as f64;

    if (ms_targets + ms_error).abs() < 1e-15 { return 0.0; }
    (ms_targets - ms_error) / (ms_targets + ms_error)
}

/// Kendall's tau-b — concordance-based rank correlation.
///
/// Counts concordant vs discordant pairs. More robust than Spearman
/// for tied ranks (common with quantized scores).
pub fn kendall_tau(x: &[f64], y: &[f64]) -> f64 {
    let n = x.len().min(y.len());
    if n < 2 { return 0.0; }
    let mut concordant = 0i64;
    let mut discordant = 0i64;
    let mut ties_x = 0i64;
    let mut ties_y = 0i64;
    for i in 0..n {
        for j in (i + 1)..n {
            let dx = x[i] - x[j];
            let dy = y[i] - y[j];
            let product = dx * dy;
            if product > 0.0 {
                concordant += 1;
            } else if product < 0.0 {
                discordant += 1;
            } else {
                if dx.abs() < 1e-15 { ties_x += 1; }
                if dy.abs() < 1e-15 { ties_y += 1; }
            }
        }
    }
    let n0 = (n * (n - 1) / 2) as f64;
    let denom = ((n0 - ties_x as f64) * (n0 - ties_y as f64)).sqrt();
    if denom < 1e-15 { return 0.0; }
    (concordant - discordant) as f64 / denom
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pearson_perfect_positive() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = vec![2.0, 4.0, 6.0, 8.0, 10.0];
        let r = pearson(&x, &y);
        assert!((r - 1.0).abs() < 1e-10);
    }

    #[test]
    fn pearson_perfect_negative() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = vec![10.0, 8.0, 6.0, 4.0, 2.0];
        let r = pearson(&x, &y);
        assert!((r - (-1.0)).abs() < 1e-10);
    }

    #[test]
    fn spearman_rank_preserved() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = vec![10.0, 400.0, 600.0, 800.0, 10000.0]; // same order, different scale
        let r = spearman(&x, &y);
        assert!((r - 1.0).abs() < 1e-10);
    }

    #[test]
    fn top_k_recall_perfect() {
        let gt = vec![5.0, 4.0, 3.0, 2.0, 1.0];
        let comp = vec![50.0, 40.0, 30.0, 20.0, 10.0]; // same ordering
        assert!((top_k_recall(&gt, &comp, 3) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn top_k_recall_worst() {
        let gt = vec![5.0, 4.0, 3.0, 2.0, 1.0];
        let comp = vec![1.0, 2.0, 3.0, 4.0, 5.0]; // reversed
        // Top-1 of GT is index 0, top-1 of comp is index 4 → recall = 0
        assert_eq!(top_k_recall(&gt, &comp, 1), 0.0);
    }

    #[test]
    fn quality_report_basic() {
        let gt = vec![1.0, 0.5, 0.1, 0.8, 0.3];
        let distances = vec![-1.0, -0.5, -0.1, -0.8, -0.3]; // negative = closer = more similar
        let report = QualityReport::compute(&gt, &distances, 1000, 100);
        assert!(report.pearson_rho > 0.99);
        assert!(report.compression_ratio > 9.0);
    }

    // ═══════════════════════════════════════════════════════════════
    // Psychometric metrics tests
    // ═══════════════════════════════════════════════════════════════

    #[test]
    fn cronbach_identical_items() {
        // k identical items → α = 1
        let item = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let items = vec![item.clone(), item.clone(), item.clone()];
        let a = cronbach_alpha(&items);
        assert!((a - 1.0).abs() < 1e-6, "identical items → α≈1, got {}", a);
    }

    #[test]
    fn cronbach_uncorrelated_items() {
        // k items that are uncorrelated → α near 0
        let items = vec![
            vec![1.0, 2.0, 3.0, 4.0, 5.0],
            vec![5.0, 1.0, 4.0, 2.0, 3.0],
            vec![3.0, 5.0, 1.0, 3.0, 2.0],
        ];
        let a = cronbach_alpha(&items);
        assert!(a < 0.5, "uncorrelated items → α<0.5, got {}", a);
    }

    #[test]
    fn cohens_kappa_perfect_agreement() {
        let a = vec![0, 1, 2, 0, 1, 2];
        let b = vec![0, 1, 2, 0, 1, 2];
        let k = cohens_kappa(&a, &b);
        assert!((k - 1.0).abs() < 1e-6, "perfect agreement → κ=1, got {}", k);
    }

    #[test]
    fn cohens_kappa_low_agreement() {
        // Raters disagree more than agree → κ should be low (possibly negative)
        let a = vec![0, 1, 0, 1, 0, 1, 0, 1];
        let b = vec![1, 0, 1, 0, 0, 1, 1, 0];
        let k = cohens_kappa(&a, &b);
        // With balanced 50/50 marginals, chance agreement p_e = 0.5.
        // Observed agreement = 2/8 = 0.25, so κ = (0.25-0.5)/(1-0.5) = -0.5
        assert!(k < 0.0, "mostly disagreement → κ<0, got {}", k);
    }

    #[test]
    fn bias_variance_zero_errors() {
        let errors = vec![0.0; 10];
        let (bias, var) = bias_variance(&errors);
        assert!(bias.abs() < 1e-12);
        assert!(var.abs() < 1e-12);
    }

    #[test]
    fn bias_variance_systematic_overestimate() {
        let errors = vec![0.5, 0.5, 0.5, 0.5]; // constant bias
        let (bias, var) = bias_variance(&errors);
        assert!((bias - 0.5).abs() < 1e-6);
        assert!(var < 1e-6, "constant errors → var≈0, got {}", var);
    }

    #[test]
    fn icc_perfect_agreement() {
        let gt = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let codec = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let icc = icc_3_1(&gt, &codec);
        assert!((icc - 1.0).abs() < 1e-6, "perfect → ICC=1, got {}", icc);
    }

    #[test]
    fn icc_poor_agreement() {
        let gt = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let codec = vec![5.0, 4.0, 3.0, 2.0, 1.0]; // reversed
        let icc = icc_3_1(&gt, &codec);
        assert!(icc < 0.0, "reversed → ICC<0, got {}", icc);
    }

    #[test]
    fn kendall_tau_perfect_concordance() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = vec![10.0, 20.0, 30.0, 40.0, 50.0];
        let tau = kendall_tau(&x, &y);
        assert!((tau - 1.0).abs() < 1e-6, "perfect concordance → τ=1, got {}", tau);
    }

    #[test]
    fn kendall_tau_perfect_discordance() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = vec![50.0, 40.0, 30.0, 20.0, 10.0];
        let tau = kendall_tau(&x, &y);
        assert!((tau - (-1.0)).abs() < 1e-6, "perfect discordance → τ=-1, got {}", tau);
    }
}
