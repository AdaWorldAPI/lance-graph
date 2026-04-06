//! Online contrastive table learning (Stage 2A).
//!
//! The distance table starts static (computed once from codebook).
//! Each forward pass on a text pair gives us the REAL cosine similarity.
//! We compare it with the table's stored value and update toward truth.
//!
//! This is exponential moving average at cell granularity:
//!   table[i][j] += alpha * (real_cos - table[i][j])
//!
//! Over thousands of forward passes, the table converges from
//! "codebook approximation" to "ground truth distribution".

/// Learning statistics snapshot.
#[derive(Clone, Debug)]
pub struct LearnerStats {
    /// Total number of cell updates applied.
    pub update_count: u64,
    /// Mean absolute error across all updates so far.
    pub mean_absolute_error: f64,
    /// Current learning rate.
    pub alpha: f32,
}

/// Online contrastive learner for the distance table.
///
/// Holds a mutable f32 distance table (256x256 or 4096x4096)
/// and updates it toward ground truth cosine similarities
/// observed during forward passes.
pub struct ContrastiveLearner {
    /// Mutable f32 distance table (size x size, row-major).
    table: Vec<f32>,
    /// Table dimension (256 or 4096).
    size: usize,
    /// Learning rate (default 0.01).
    alpha: f32,
    /// Number of updates applied.
    update_count: u64,
    /// Running sum of absolute errors (for tracking improvement).
    total_error: f64,
}

impl ContrastiveLearner {
    /// Create from an existing f32 cosine table.
    ///
    /// `table` must have exactly `sqrt(len)` squared elements (i.e. N*N).
    /// `alpha` is the learning rate; 0.01 is a good default.
    ///
    /// # Panics
    /// Panics if `table.len()` is not a perfect square.
    pub fn new(table: Vec<f32>, alpha: f32) -> Self {
        let len = table.len();
        let size = (len as f64).sqrt() as usize;
        assert_eq!(size * size, len, "table must be square (got {} elements)", len);
        Self {
            table,
            size,
            alpha,
            update_count: 0,
            total_error: 0.0,
        }
    }

    /// Update one pair: table[i][j] += alpha * (real_cos - table[i][j]).
    ///
    /// Also updates the symmetric entry table[j][i] to maintain symmetry.
    /// Returns the absolute error before update.
    ///
    /// # Panics
    /// Panics if `i` or `j` >= `size`.
    pub fn update_pair(&mut self, i: usize, j: usize, real_cosine: f32) -> f32 {
        assert!(i < self.size && j < self.size, "indices out of bounds");

        let idx_ij = i * self.size + j;
        let error = (real_cosine - self.table[idx_ij]).abs();

        // Update table[i][j]
        self.table[idx_ij] += self.alpha * (real_cosine - self.table[idx_ij]);

        // Maintain symmetry: update table[j][i] too
        if i != j {
            let idx_ji = j * self.size + i;
            self.table[idx_ji] += self.alpha * (real_cosine - self.table[idx_ji]);
        }

        self.update_count += 1;
        self.total_error += error as f64;

        error
    }

    /// Fan-out update: given centroid `i` and observed cosine similarities
    /// with its neighbors, update table[i][j] for each neighbor j.
    ///
    /// `neighbors` is a slice of (centroid_index, real_cosine_with_i).
    /// Returns the mean absolute error across all updated pairs.
    pub fn update_fan_out(&mut self, i: usize, neighbors: &[(usize, f32)]) -> f32 {
        if neighbors.is_empty() {
            return 0.0;
        }
        let mut total_err = 0.0f32;
        for &(j, real_cos) in neighbors {
            total_err += self.update_pair(i, j, real_cos);
        }
        total_err / neighbors.len() as f32
    }

    /// Get the current table as a slice (for building an engine from it).
    pub fn table(&self) -> &[f32] {
        &self.table
    }

    /// Get learning statistics.
    pub fn stats(&self) -> LearnerStats {
        LearnerStats {
            update_count: self.update_count,
            mean_absolute_error: if self.update_count > 0 {
                self.total_error / self.update_count as f64
            } else {
                0.0
            },
            alpha: self.alpha,
        }
    }

    /// Decay learning rate: alpha *= decay_factor.
    pub fn decay_lr(&mut self, factor: f32) {
        self.alpha *= factor;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper: create a 4x4 table initialized to 0.5 everywhere.
    fn make_table(size: usize, value: f32) -> Vec<f32> {
        vec![value; size * size]
    }

    #[test]
    fn test_update_pair_moves_toward_real_cosine() {
        let mut learner = ContrastiveLearner::new(make_table(4, 0.5), 0.1);

        // Real cosine is 0.9; table starts at 0.5.
        let error = learner.update_pair(1, 2, 0.9);

        // Error should be |0.9 - 0.5| = 0.4
        assert!((error - 0.4).abs() < 1e-6, "error was {}", error);

        // After update: 0.5 + 0.1 * (0.9 - 0.5) = 0.54
        let idx = 1 * 4 + 2;
        assert!((learner.table[idx] - 0.54).abs() < 1e-6,
            "table[1][2] was {}", learner.table[idx]);
    }

    #[test]
    fn test_symmetry_maintained() {
        let mut learner = ContrastiveLearner::new(make_table(4, 0.5), 0.1);

        learner.update_pair(0, 3, 0.8);

        let val_03 = learner.table[0 * 4 + 3];
        let val_30 = learner.table[3 * 4 + 0];
        assert!((val_03 - val_30).abs() < 1e-6,
            "symmetry broken: [0][3]={} vs [3][0]={}", val_03, val_30);
    }

    #[test]
    fn test_diagonal_update_no_double() {
        // Updating (i, i) should not double-update.
        let mut learner = ContrastiveLearner::new(make_table(4, 0.5), 0.5);

        learner.update_pair(2, 2, 1.0);

        // 0.5 + 0.5 * (1.0 - 0.5) = 0.75  (NOT 1.0)
        let val = learner.table[2 * 4 + 2];
        assert!((val - 0.75).abs() < 1e-6, "diagonal was {}", val);
    }

    #[test]
    fn test_mae_decreases_over_repeated_updates() {
        // Repeatedly tell the learner that table[1][2] = 0.9.
        // The MAE of each batch of updates should decrease monotonically.
        let size = 8;
        let mut learner = ContrastiveLearner::new(make_table(size, 0.0), 0.3);
        let target = 0.9;

        let mut prev_error = f32::MAX;
        for _ in 0..30 {
            let err = learner.update_pair(1, 2, target);
            assert!(err <= prev_error + 1e-7,
                "MAE not decreasing: {} > {}", err, prev_error);
            prev_error = err;
        }

        // After 30 updates at alpha=0.3, error = 0.9 * (1-0.3)^30 ≈ 0.00002
        assert!(prev_error < 0.01,
            "error still large after 30 updates: {}", prev_error);
    }

    #[test]
    fn test_fan_out_k16_neighbors() {
        let size = 256;
        let mut learner = ContrastiveLearner::new(make_table(size, 0.5), 0.01);

        // 16 neighbors of centroid 42 with varying real cosines.
        let neighbors: Vec<(usize, f32)> = (0..16)
            .map(|k| (k + 100, 0.3 + 0.04 * k as f32)) // cos from 0.3 to 0.9
            .collect();

        let mean_err = learner.update_fan_out(42, &neighbors);

        // All started at 0.5; targets range 0.3..0.9.
        // Mean target = 0.6, so mean |target - 0.5| = mean of deviations.
        assert!(mean_err > 0.0, "mean error should be positive");
        assert!(mean_err < 0.5, "mean error unreasonably large: {}", mean_err);

        // Check symmetry for one neighbor.
        let val_42_100 = learner.table[42 * size + 100];
        let val_100_42 = learner.table[100 * size + 42];
        assert!((val_42_100 - val_100_42).abs() < 1e-6,
            "fan-out broke symmetry: {} vs {}", val_42_100, val_100_42);

        // Verify update count: 16 neighbors = 16 update_pair calls.
        assert_eq!(learner.stats().update_count, 16);
    }

    #[test]
    fn test_stats_tracking() {
        let mut learner = ContrastiveLearner::new(make_table(4, 0.0), 0.1);

        let s0 = learner.stats();
        assert_eq!(s0.update_count, 0);
        assert_eq!(s0.mean_absolute_error, 0.0);
        assert!((s0.alpha - 0.1).abs() < 1e-6);

        learner.update_pair(0, 1, 0.8); // error = 0.8
        learner.update_pair(2, 3, 0.4); // error = 0.4

        let s1 = learner.stats();
        assert_eq!(s1.update_count, 2);
        assert!((s1.mean_absolute_error - 0.6).abs() < 1e-6,
            "MAE was {}", s1.mean_absolute_error);
    }

    #[test]
    fn test_decay_lr() {
        let mut learner = ContrastiveLearner::new(make_table(4, 0.0), 0.1);
        learner.decay_lr(0.5);
        assert!((learner.stats().alpha - 0.05).abs() < 1e-6);
        learner.decay_lr(0.1);
        assert!((learner.stats().alpha - 0.005).abs() < 1e-6);
    }

    #[test]
    fn test_fan_out_empty_neighbors() {
        let mut learner = ContrastiveLearner::new(make_table(4, 0.5), 0.01);
        let mean_err = learner.update_fan_out(0, &[]);
        assert_eq!(mean_err, 0.0);
        assert_eq!(learner.stats().update_count, 0);
    }

    #[test]
    #[should_panic(expected = "table must be square")]
    fn test_non_square_table_panics() {
        ContrastiveLearner::new(vec![0.0; 5], 0.01);
    }
}
