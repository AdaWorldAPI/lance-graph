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
}
