//! Scalar-valued sparse matrix for palette graph operations.
//!
//! The blasgraph `CsrStorage` stores `Vec<BitVec>`. The palette distance
//! matrix needs scalar-valued edges (u16 distances). This module provides
//! a lightweight scalar CSR that can feed into semiring-style operations.

/// Scalar-valued CSR sparse matrix.
#[derive(Clone, Debug)]
pub struct ScalarCsr {
    pub nrows: usize,
    pub ncols: usize,
    /// Row pointers: `row_ptr[i]..row_ptr[i+1]` indexes into col_idx/vals.
    pub row_ptr: Vec<usize>,
    /// Column indices.
    pub col_idx: Vec<usize>,
    /// Scalar values (u16 distances or f32 weights).
    pub vals: Vec<f32>,
}

impl ScalarCsr {
    /// Create empty matrix.
    pub fn new(nrows: usize, ncols: usize) -> Self {
        ScalarCsr {
            nrows,
            ncols,
            row_ptr: vec![0; nrows + 1],
            col_idx: Vec::new(),
            vals: Vec::new(),
        }
    }

    /// Build from dense distance matrix (only stores non-zero entries below threshold).
    /// All entries stored if threshold = f32::MAX.
    pub fn from_dense(data: &[f32], nrows: usize, ncols: usize, threshold: f32) -> Self {
        let mut row_ptr = Vec::with_capacity(nrows + 1);
        let mut col_idx = Vec::new();
        let mut vals = Vec::new();

        row_ptr.push(0);
        for i in 0..nrows {
            for j in 0..ncols {
                let v = data[i * ncols + j];
                if v > 0.0 && v < threshold {
                    col_idx.push(j);
                    vals.push(v);
                }
            }
            row_ptr.push(col_idx.len());
        }

        ScalarCsr { nrows, ncols, row_ptr, col_idx, vals }
    }

    /// Number of stored entries.
    pub fn nnz(&self) -> usize {
        self.vals.len()
    }

    /// Get value at (row, col). Returns None if not stored.
    pub fn get(&self, row: usize, col: usize) -> Option<f32> {
        let start = self.row_ptr[row];
        let end = self.row_ptr[row + 1];
        for idx in start..end {
            if self.col_idx[idx] == col {
                return Some(self.vals[idx]);
            }
        }
        None
    }

    /// Sparse matrix-vector multiply: y = A * x under (multiply, add) semiring.
    /// Default: (float_mul, float_add) = standard matrix-vector product.
    pub fn spmv(&self, x: &[f32]) -> Vec<f32> {
        assert_eq!(x.len(), self.ncols);
        let mut y = vec![0.0f32; self.nrows];

        for (i, y_val) in y.iter_mut().enumerate() {
            let start = self.row_ptr[i];
            let end = self.row_ptr[i + 1];
            let mut acc = 0.0f32;
            for idx in start..end {
                acc += self.vals[idx] * x[self.col_idx[idx]];
            }
            *y_val = acc;
        }
        y
    }

    /// Sparse matrix-vector multiply under min-plus (tropical) semiring.
    /// y[i] = min_j (A[i,j] + x[j]).
    /// Used for shortest-path / nearest-neighbor in palette space.
    pub fn spmv_min_plus(&self, x: &[f32]) -> Vec<f32> {
        assert_eq!(x.len(), self.ncols);
        let mut y = vec![f32::INFINITY; self.nrows];

        for (i, y_val) in y.iter_mut().enumerate() {
            let start = self.row_ptr[i];
            let end = self.row_ptr[i + 1];
            for idx in start..end {
                let candidate = self.vals[idx] + x[self.col_idx[idx]];
                if candidate < *y_val {
                    *y_val = candidate;
                }
            }
        }
        y
    }

    /// Convert a DistanceMatrix to ScalarCsr (threshold-filtered).
    pub fn from_distance_matrix(dm: &crate::distance_matrix::DistanceMatrix, threshold: u16) -> Self {
        let k = dm.k;
        let dense: Vec<f32> = dm.data.iter().map(|&d| d as f32).collect();
        ScalarCsr::from_dense(&dense, k, k, threshold as f32)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_spmv_identity() {
        // 3×3 identity matrix
        let mut csr = ScalarCsr::new(3, 3);
        csr.row_ptr = vec![0, 1, 2, 3];
        csr.col_idx = vec![0, 1, 2];
        csr.vals = vec![1.0, 1.0, 1.0];

        let x = vec![10.0, 20.0, 30.0];
        let y = csr.spmv(&x);
        assert_eq!(y, x);
    }

    #[test]
    fn test_spmv_min_plus() {
        // Simple 2×2: [[1, 3], [2, 1]]
        let mut csr = ScalarCsr::new(2, 2);
        csr.row_ptr = vec![0, 2, 4];
        csr.col_idx = vec![0, 1, 0, 1];
        csr.vals = vec![1.0, 3.0, 2.0, 1.0];

        let x = vec![0.0, 5.0]; // distances from source
        let y = csr.spmv_min_plus(&x);
        // y[0] = min(1+0, 3+5) = 1
        // y[1] = min(2+0, 1+5) = 2
        assert_eq!(y[0], 1.0);
        assert_eq!(y[1], 2.0);
    }
}
