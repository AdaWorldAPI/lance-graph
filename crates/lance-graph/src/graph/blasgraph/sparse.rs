// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! # Sparse Storage Formats
//!
//! Provides Compressed Sparse Row (CSR) and Coordinate (COO) storage
//! for sparse matrices and vectors of [`BitVec`] elements.

use crate::graph::blasgraph::types::BitVec;

/// A single entry in a sparse structure: `(index, value)`.
#[derive(Clone, Debug)]
pub struct SparseEntry {
    /// Row or column index of this entry.
    pub index: usize,
    /// The hyperdimensional vector stored at this position.
    pub value: BitVec,
}

/// A sparse vector stored as sorted `(index, value)` pairs.
#[derive(Clone, Debug)]
pub struct SparseVec {
    /// Dimension (logical length) of the vector.
    pub dim: usize,
    /// Sorted entries.
    pub entries: Vec<SparseEntry>,
}

impl SparseVec {
    /// Create an empty sparse vector of the given dimension.
    pub fn new(dim: usize) -> Self {
        Self {
            dim,
            entries: Vec::new(),
        }
    }

    /// Number of stored (non-zero) entries.
    pub fn nnz(&self) -> usize {
        self.entries.len()
    }

    /// Insert or replace an entry. Maintains sorted order.
    pub fn set(&mut self, index: usize, value: BitVec) {
        assert!(
            index < self.dim,
            "index {} out of bounds (dim={})",
            index,
            self.dim
        );
        match self.entries.binary_search_by_key(&index, |e| e.index) {
            Ok(pos) => self.entries[pos].value = value,
            Err(pos) => self.entries.insert(pos, SparseEntry { index, value }),
        }
    }

    /// Get the value at the given index, if present.
    pub fn get(&self, index: usize) -> Option<&BitVec> {
        self.entries
            .binary_search_by_key(&index, |e| e.index)
            .ok()
            .map(|pos| &self.entries[pos].value)
    }

    /// Iterate over `(index, &BitVec)` pairs.
    pub fn iter(&self) -> impl Iterator<Item = (usize, &BitVec)> {
        self.entries.iter().map(|e| (e.index, &e.value))
    }
}

/// Coordinate (triplet) storage format for sparse matrices.
///
/// Stores `(row, col, value)` triplets. Suitable for incremental
/// construction before converting to CSR.
#[derive(Clone, Debug)]
pub struct CooStorage {
    /// Number of rows.
    pub nrows: usize,
    /// Number of columns.
    pub ncols: usize,
    /// Row indices.
    pub rows: Vec<usize>,
    /// Column indices.
    pub cols: Vec<usize>,
    /// Values.
    pub vals: Vec<BitVec>,
}

impl CooStorage {
    /// Create an empty COO storage with the given dimensions.
    pub fn new(nrows: usize, ncols: usize) -> Self {
        Self {
            nrows,
            ncols,
            rows: Vec::new(),
            cols: Vec::new(),
            vals: Vec::new(),
        }
    }

    /// Number of stored entries.
    pub fn nnz(&self) -> usize {
        self.vals.len()
    }

    /// Add a triplet `(row, col, value)`.
    pub fn push(&mut self, row: usize, col: usize, value: BitVec) {
        assert!(
            row < self.nrows,
            "row {} out of bounds (nrows={})",
            row,
            self.nrows
        );
        assert!(
            col < self.ncols,
            "col {} out of bounds (ncols={})",
            col,
            self.ncols
        );
        self.rows.push(row);
        self.cols.push(col);
        self.vals.push(value);
    }

    /// Convert to CSR format.
    pub fn to_csr(&self) -> CsrStorage {
        let mut csr = CsrStorage::new(self.nrows, self.ncols);

        // Sort triplets by (row, col)
        let mut indices: Vec<usize> = (0..self.nnz()).collect();
        indices.sort_by(|&a, &b| {
            self.rows[a]
                .cmp(&self.rows[b])
                .then(self.cols[a].cmp(&self.cols[b]))
        });

        let mut row_ptrs = vec![0usize; self.nrows + 1];
        let mut col_indices = Vec::with_capacity(self.nnz());
        let mut values = Vec::with_capacity(self.nnz());

        for &idx in &indices {
            let r = self.rows[idx];
            row_ptrs[r + 1] += 1;
            col_indices.push(self.cols[idx]);
            values.push(self.vals[idx].clone());
        }

        // Prefix sum
        for i in 1..=self.nrows {
            row_ptrs[i] += row_ptrs[i - 1];
        }

        csr.row_ptrs = row_ptrs;
        csr.col_indices = col_indices;
        csr.values = values;
        csr
    }

    /// Iterate over `(row, col, &BitVec)` triplets.
    pub fn iter(&self) -> impl Iterator<Item = (usize, usize, &BitVec)> {
        self.rows
            .iter()
            .zip(self.cols.iter())
            .zip(self.vals.iter())
            .map(|((&r, &c), v)| (r, c, v))
    }
}

/// Compressed Sparse Row (CSR) storage format.
///
/// The standard format for efficient row-wise access and matrix-vector
/// multiplication.
#[derive(Clone, Debug)]
pub struct CsrStorage {
    /// Number of rows.
    pub nrows: usize,
    /// Number of columns.
    pub ncols: usize,
    /// Row pointers: `row_ptrs[i]..row_ptrs[i+1]` are the entries for row `i`.
    pub row_ptrs: Vec<usize>,
    /// Column indices for each non-zero entry.
    pub col_indices: Vec<usize>,
    /// Values for each non-zero entry.
    pub values: Vec<BitVec>,
}

impl CsrStorage {
    /// Create an empty CSR storage with the given dimensions.
    pub fn new(nrows: usize, ncols: usize) -> Self {
        Self {
            nrows,
            ncols,
            row_ptrs: vec![0; nrows + 1],
            col_indices: Vec::new(),
            values: Vec::new(),
        }
    }

    /// Number of stored entries.
    pub fn nnz(&self) -> usize {
        self.values.len()
    }

    /// Get the entries for a specific row as `(col, &BitVec)` pairs.
    pub fn row(&self, i: usize) -> impl Iterator<Item = (usize, &BitVec)> {
        let start = self.row_ptrs[i];
        let end = self.row_ptrs[i + 1];
        self.col_indices[start..end]
            .iter()
            .zip(self.values[start..end].iter())
            .map(|(&c, v)| (c, v))
    }

    /// Get the value at `(row, col)` if present.
    pub fn get(&self, row: usize, col: usize) -> Option<&BitVec> {
        let start = self.row_ptrs[row];
        let end = self.row_ptrs[row + 1];
        let slice = &self.col_indices[start..end];
        slice
            .binary_search(&col)
            .ok()
            .map(|pos| &self.values[start + pos])
    }

    /// Convert back to COO format.
    pub fn to_coo(&self) -> CooStorage {
        let mut coo = CooStorage::new(self.nrows, self.ncols);
        for row in 0..self.nrows {
            let start = self.row_ptrs[row];
            let end = self.row_ptrs[row + 1];
            for idx in start..end {
                coo.push(row, self.col_indices[idx], self.values[idx].clone());
            }
        }
        coo
    }

    /// Transpose the CSR matrix, returning a new CSR.
    pub fn transpose(&self) -> CsrStorage {
        let coo = self.to_coo();
        let mut transposed = CooStorage::new(self.ncols, self.nrows);
        for (r, c, v) in coo.iter() {
            transposed.push(c, r, v.clone());
        }
        transposed.to_csr()
    }
}

/// Selector for the sparse storage format.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SparseFormat {
    /// Compressed Sparse Row.
    Csr,
    /// Compressed Sparse Column.
    Csc,
    /// Coordinate / triplet.
    Coo,
}

/// Storage format selector (includes future variants).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StorageFormat {
    /// Compressed Sparse Row — efficient row iteration.
    Csr,
    /// Compressed Sparse Column — efficient column iteration / zero-copy transpose.
    Csc,
    /// Hypersparse CSR — only stores non-empty rows.
    HyperCsr,
    /// Hypersparse CSC (reserved).
    HyperCsc,
    /// Dense bitmap (reserved).
    Bitmap,
    /// Dense (reserved).
    Dense,
}

// ─── CscStorage ──────────────────────────────────────────────────────

/// Compressed Sparse Column (CSC) storage format.
///
/// The column-oriented dual of CSR. Column `j` has entries at
/// `col_ptrs[j]..col_ptrs[j+1]`. Enables zero-cost transpose when paired
/// with CSR: CSR of A is CSC of A^T and vice versa.
#[derive(Clone, Debug)]
pub struct CscStorage {
    /// Number of rows.
    pub nrows: usize,
    /// Number of columns.
    pub ncols: usize,
    /// Column pointers: `col_ptrs[j]..col_ptrs[j+1]` are the entries for column `j`.
    pub col_ptrs: Vec<usize>,
    /// Row indices for each non-zero entry.
    pub row_indices: Vec<usize>,
    /// Values for each non-zero entry.
    pub values: Vec<BitVec>,
}

impl CscStorage {
    /// Create an empty CSC storage with the given dimensions.
    pub fn new(nrows: usize, ncols: usize) -> Self {
        Self {
            nrows,
            ncols,
            col_ptrs: vec![0; ncols + 1],
            row_indices: Vec::new(),
            values: Vec::new(),
        }
    }

    /// Number of stored entries.
    pub fn nnz(&self) -> usize {
        self.values.len()
    }

    /// Get the entries for a specific column as `(row, &BitVec)` pairs.
    pub fn column_iter(&self, j: usize) -> impl Iterator<Item = (usize, &BitVec)> {
        let start = self.col_ptrs[j];
        let end = self.col_ptrs[j + 1];
        self.row_indices[start..end]
            .iter()
            .zip(self.values[start..end].iter())
            .map(|(&r, v)| (r, v))
    }

    /// Get the value at `(row, col)` if present.
    pub fn get(&self, row: usize, col: usize) -> Option<&BitVec> {
        let start = self.col_ptrs[col];
        let end = self.col_ptrs[col + 1];
        let slice = &self.row_indices[start..end];
        slice
            .binary_search(&row)
            .ok()
            .map(|pos| &self.values[start + pos])
    }

    /// Build CSC from CSR (transpose of the index structure).
    pub fn from_csr(csr: &CsrStorage) -> Self {
        // CSC of A = CSR of A^T, but we build it directly for efficiency.
        let mut csc = CscStorage::new(csr.nrows, csr.ncols);

        // Count entries per column
        let mut col_counts = vec![0usize; csr.ncols];
        for &c in &csr.col_indices {
            col_counts[c] += 1;
        }

        // Build col_ptrs via prefix sum
        csc.col_ptrs = vec![0usize; csr.ncols + 1];
        for (j, &count) in col_counts.iter().enumerate() {
            csc.col_ptrs[j + 1] = csc.col_ptrs[j] + count;
        }

        let nnz = csr.nnz();
        csc.row_indices = vec![0usize; nnz];
        csc.values = vec![BitVec::zero(); nnz];

        // Place entries
        let mut write_pos = csc.col_ptrs.clone();
        for row in 0..csr.nrows {
            let start = csr.row_ptrs[row];
            let end = csr.row_ptrs[row + 1];
            for idx in start..end {
                let col = csr.col_indices[idx];
                let pos = write_pos[col];
                csc.row_indices[pos] = row;
                csc.values[pos] = csr.values[idx].clone();
                write_pos[col] += 1;
            }
        }

        csc
    }

    /// Build CSC from COO.
    pub fn from_coo(coo: &CooStorage) -> Self {
        Self::from_csr(&coo.to_csr())
    }

    /// Convert to CSR format.
    pub fn to_csr(&self) -> CsrStorage {
        // CSR of A = CSC of A treated as A^T's CSR, but keeping dims
        let mut csr = CsrStorage::new(self.nrows, self.ncols);

        // Count entries per row
        let mut row_counts = vec![0usize; self.nrows];
        for &r in &self.row_indices {
            row_counts[r] += 1;
        }

        csr.row_ptrs = vec![0usize; self.nrows + 1];
        for (i, &count) in row_counts.iter().enumerate() {
            csr.row_ptrs[i + 1] = csr.row_ptrs[i] + count;
        }

        let nnz = self.nnz();
        csr.col_indices = vec![0usize; nnz];
        csr.values = vec![BitVec::zero(); nnz];

        let mut write_pos = csr.row_ptrs.clone();
        for col in 0..self.ncols {
            let start = self.col_ptrs[col];
            let end = self.col_ptrs[col + 1];
            for idx in start..end {
                let row = self.row_indices[idx];
                let pos = write_pos[row];
                csr.col_indices[pos] = col;
                csr.values[pos] = self.values[idx].clone();
                write_pos[row] += 1;
            }
        }

        // Sort each row by column index for binary search
        for row in 0..self.nrows {
            let start = csr.row_ptrs[row];
            let end = csr.row_ptrs[row + 1];
            if end - start <= 1 {
                continue;
            }
            let mut pairs: Vec<(usize, BitVec)> = csr.col_indices[start..end]
                .iter()
                .zip(csr.values[start..end].iter())
                .map(|(&c, v)| (c, v.clone()))
                .collect();
            pairs.sort_by_key(|(c, _)| *c);
            for (i, (c, v)) in pairs.into_iter().enumerate() {
                csr.col_indices[start + i] = c;
                csr.values[start + i] = v;
            }
        }

        csr
    }
}

// ─── HyperCsrStorage ────────────────────────────────────────────────

/// Hypersparse CSR storage for power-law graphs.
///
/// When `nnz / nrows < 0.1`, most rows are empty. HyperCSR only stores
/// rows that actually have entries, saving O(nrows) in the row pointer array.
#[derive(Clone, Debug)]
pub struct HyperCsrStorage {
    /// Number of rows (logical).
    pub nrows: usize,
    /// Number of columns (logical).
    pub ncols: usize,
    /// Which rows have entries (sorted, no duplicates).
    pub row_ids: Vec<usize>,
    /// Row pointers: `row_ptrs[i]..row_ptrs[i+1]` are the entries for `row_ids[i]`.
    pub row_ptrs: Vec<usize>,
    /// Column indices for each non-zero entry.
    pub col_indices: Vec<usize>,
    /// Values for each non-zero entry.
    pub values: Vec<BitVec>,
}

impl HyperCsrStorage {
    /// Number of stored entries.
    pub fn nnz(&self) -> usize {
        self.values.len()
    }

    /// Number of non-empty rows.
    pub fn n_nonempty_rows(&self) -> usize {
        self.row_ids.len()
    }

    /// Heuristic: use HyperCSR when density is below this threshold.
    pub const DENSITY_THRESHOLD: f64 = 0.1;

    /// Check if HyperCSR is beneficial for the given sparsity.
    pub fn should_use(nnz: usize, nrows: usize) -> bool {
        nrows > 0 && (nnz as f64 / nrows as f64) < Self::DENSITY_THRESHOLD
    }

    /// Build from CSR storage.
    pub fn from_csr(csr: &CsrStorage) -> Self {
        let mut row_ids = Vec::new();
        let mut row_ptrs = vec![0usize];
        let mut col_indices = Vec::new();
        let mut values = Vec::new();

        for row in 0..csr.nrows {
            let start = csr.row_ptrs[row];
            let end = csr.row_ptrs[row + 1];
            if start < end {
                row_ids.push(row);
                for idx in start..end {
                    col_indices.push(csr.col_indices[idx]);
                    values.push(csr.values[idx].clone());
                }
                row_ptrs.push(col_indices.len());
            }
        }

        HyperCsrStorage {
            nrows: csr.nrows,
            ncols: csr.ncols,
            row_ids,
            row_ptrs,
            col_indices,
            values,
        }
    }

    /// Convert back to CSR.
    pub fn to_csr(&self) -> CsrStorage {
        let mut csr = CsrStorage::new(self.nrows, self.ncols);
        let mut all_row_ptrs = vec![0usize; self.nrows + 1];
        let mut col_indices = Vec::with_capacity(self.nnz());
        let mut values = Vec::with_capacity(self.nnz());

        for (ri, &row) in self.row_ids.iter().enumerate() {
            let start = self.row_ptrs[ri];
            let end = self.row_ptrs[ri + 1];
            for idx in start..end {
                col_indices.push(self.col_indices[idx]);
                values.push(self.values[idx].clone());
            }
            all_row_ptrs[row + 1] = end - start;
        }

        // Prefix sum
        for i in 1..=self.nrows {
            all_row_ptrs[i] += all_row_ptrs[i - 1];
        }

        csr.row_ptrs = all_row_ptrs;
        csr.col_indices = col_indices;
        csr.values = values;
        csr
    }

    /// Get the value at `(row, col)` if present.
    pub fn get(&self, row: usize, col: usize) -> Option<&BitVec> {
        let ri = self.row_ids.binary_search(&row).ok()?;
        let start = self.row_ptrs[ri];
        let end = self.row_ptrs[ri + 1];
        let slice = &self.col_indices[start..end];
        slice
            .binary_search(&col)
            .ok()
            .map(|pos| &self.values[start + pos])
    }

    /// Iterate over entries of a specific row as `(col, &BitVec)`.
    pub fn row(&self, row: usize) -> Option<impl Iterator<Item = (usize, &BitVec)>> {
        let ri = self.row_ids.binary_search(&row).ok()?;
        let start = self.row_ptrs[ri];
        let end = self.row_ptrs[ri + 1];
        Some(
            self.col_indices[start..end]
                .iter()
                .zip(self.values[start..end].iter())
                .map(|(&c, v)| (c, v)),
        )
    }

    /// Memory saved compared to full CSR (in row pointer words).
    pub fn row_ptr_savings(&self) -> usize {
        // Full CSR needs nrows+1 pointers; HyperCSR needs n_nonempty+1
        self.nrows.saturating_sub(self.n_nonempty_rows())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sparse_vec_basic() {
        let mut sv = SparseVec::new(10);
        assert_eq!(sv.nnz(), 0);
        assert!(sv.get(0).is_none());

        sv.set(3, BitVec::random(1));
        sv.set(7, BitVec::random(2));
        assert_eq!(sv.nnz(), 2);
        assert!(sv.get(3).is_some());
        assert!(sv.get(7).is_some());
        assert!(sv.get(5).is_none());
    }

    #[test]
    fn test_sparse_vec_overwrite() {
        let mut sv = SparseVec::new(5);
        sv.set(2, BitVec::random(10));
        let v2 = BitVec::random(20);
        sv.set(2, v2.clone());
        assert_eq!(sv.nnz(), 1);
        assert_eq!(sv.get(2).unwrap(), &v2);
    }

    #[test]
    fn test_sparse_vec_sorted_order() {
        let mut sv = SparseVec::new(10);
        sv.set(5, BitVec::random(1));
        sv.set(1, BitVec::random(2));
        sv.set(8, BitVec::random(3));
        let indices: Vec<usize> = sv.iter().map(|(i, _)| i).collect();
        assert_eq!(indices, vec![1, 5, 8]);
    }

    #[test]
    fn test_coo_basic() {
        let mut coo = CooStorage::new(3, 3);
        coo.push(0, 1, BitVec::random(1));
        coo.push(1, 2, BitVec::random(2));
        coo.push(2, 0, BitVec::random(3));
        assert_eq!(coo.nnz(), 3);
    }

    #[test]
    fn test_coo_to_csr_roundtrip() {
        let mut coo = CooStorage::new(3, 4);
        let v1 = BitVec::random(10);
        let v2 = BitVec::random(20);
        let v3 = BitVec::random(30);
        coo.push(0, 1, v1.clone());
        coo.push(0, 3, v2.clone());
        coo.push(2, 2, v3.clone());

        let csr = coo.to_csr();
        assert_eq!(csr.nrows, 3);
        assert_eq!(csr.ncols, 4);
        assert_eq!(csr.nnz(), 3);

        assert_eq!(csr.get(0, 1).unwrap(), &v1);
        assert_eq!(csr.get(0, 3).unwrap(), &v2);
        assert_eq!(csr.get(2, 2).unwrap(), &v3);
        assert!(csr.get(1, 0).is_none());
    }

    #[test]
    fn test_csr_row_iteration() {
        let mut coo = CooStorage::new(2, 3);
        coo.push(0, 0, BitVec::random(1));
        coo.push(0, 2, BitVec::random(2));
        coo.push(1, 1, BitVec::random(3));

        let csr = coo.to_csr();

        let row0: Vec<usize> = csr.row(0).map(|(c, _)| c).collect();
        assert_eq!(row0, vec![0, 2]);

        let row1: Vec<usize> = csr.row(1).map(|(c, _)| c).collect();
        assert_eq!(row1, vec![1]);
    }

    #[test]
    fn test_csr_transpose() {
        let mut coo = CooStorage::new(2, 3);
        let v1 = BitVec::random(1);
        coo.push(0, 2, v1.clone());
        coo.push(1, 0, BitVec::random(2));

        let csr = coo.to_csr();
        let trans = csr.transpose();
        assert_eq!(trans.nrows, 3);
        assert_eq!(trans.ncols, 2);
        assert_eq!(trans.get(2, 0).unwrap(), &v1);
    }

    #[test]
    fn test_csr_to_coo() {
        let mut coo = CooStorage::new(2, 2);
        coo.push(0, 0, BitVec::random(1));
        coo.push(1, 1, BitVec::random(2));

        let csr = coo.to_csr();
        let coo2 = csr.to_coo();
        assert_eq!(coo2.nnz(), 2);
    }

    #[test]
    fn test_empty_csr() {
        let csr = CsrStorage::new(5, 5);
        assert_eq!(csr.nnz(), 0);
        assert!(csr.get(0, 0).is_none());
        let row: Vec<_> = csr.row(0).collect();
        assert!(row.is_empty());
    }

    #[test]
    #[should_panic(expected = "out of bounds")]
    fn test_coo_out_of_bounds_row() {
        let mut coo = CooStorage::new(2, 2);
        coo.push(5, 0, BitVec::zero());
    }

    #[test]
    #[should_panic(expected = "out of bounds")]
    fn test_sparse_vec_out_of_bounds() {
        let mut sv = SparseVec::new(3);
        sv.set(10, BitVec::zero());
    }

    // ── CscStorage tests ────────────────────────────────────────────

    #[test]
    fn test_csc_from_csr_roundtrip() {
        let mut coo = CooStorage::new(3, 4);
        let v1 = BitVec::random(10);
        let v2 = BitVec::random(20);
        let v3 = BitVec::random(30);
        coo.push(0, 1, v1.clone());
        coo.push(0, 3, v2.clone());
        coo.push(2, 2, v3.clone());

        let csr = coo.to_csr();
        let csc = CscStorage::from_csr(&csr);

        assert_eq!(csc.nrows, 3);
        assert_eq!(csc.ncols, 4);
        assert_eq!(csc.nnz(), 3);

        // Verify entries
        assert_eq!(csc.get(0, 1).unwrap(), &v1);
        assert_eq!(csc.get(0, 3).unwrap(), &v2);
        assert_eq!(csc.get(2, 2).unwrap(), &v3);
        assert!(csc.get(1, 0).is_none());
    }

    #[test]
    fn test_csc_to_csr_roundtrip() {
        let mut coo = CooStorage::new(3, 3);
        let v1 = BitVec::random(1);
        let v2 = BitVec::random(2);
        coo.push(0, 2, v1.clone());
        coo.push(2, 0, v2.clone());

        let csr_orig = coo.to_csr();
        let csc = CscStorage::from_csr(&csr_orig);
        let csr_back = csc.to_csr();

        assert_eq!(csr_back.nnz(), 2);
        assert_eq!(csr_back.get(0, 2).unwrap(), &v1);
        assert_eq!(csr_back.get(2, 0).unwrap(), &v2);
    }

    #[test]
    fn test_csc_column_iter() {
        let mut coo = CooStorage::new(3, 3);
        coo.push(0, 1, BitVec::random(1));
        coo.push(2, 1, BitVec::random(2));
        coo.push(1, 0, BitVec::random(3));

        let csc = CscStorage::from_coo(&coo);
        let col1_rows: Vec<usize> = csc.column_iter(1).map(|(r, _)| r).collect();
        assert_eq!(col1_rows, vec![0, 2]);
    }

    #[test]
    fn test_csc_empty() {
        let csc = CscStorage::new(5, 5);
        assert_eq!(csc.nnz(), 0);
        assert!(csc.get(0, 0).is_none());
    }

    // ── HyperCsrStorage tests ───────────────────────────────────────

    #[test]
    fn test_hyper_csr_from_csr_roundtrip() {
        let mut coo = CooStorage::new(1000, 1000);
        // Very sparse: only 3 entries in a 1000x1000 matrix
        let v1 = BitVec::random(10);
        let v2 = BitVec::random(20);
        let v3 = BitVec::random(30);
        coo.push(5, 10, v1.clone());
        coo.push(100, 200, v2.clone());
        coo.push(999, 0, v3.clone());

        let csr = coo.to_csr();
        let hyper = HyperCsrStorage::from_csr(&csr);

        assert_eq!(hyper.nnz(), 3);
        assert_eq!(hyper.n_nonempty_rows(), 3);
        assert_eq!(hyper.row_ids, vec![5, 100, 999]);

        // Verify entries
        assert_eq!(hyper.get(5, 10).unwrap(), &v1);
        assert_eq!(hyper.get(100, 200).unwrap(), &v2);
        assert_eq!(hyper.get(999, 0).unwrap(), &v3);
        assert!(hyper.get(0, 0).is_none());
    }

    #[test]
    fn test_hyper_csr_to_csr_roundtrip() {
        let mut coo = CooStorage::new(100, 100);
        let v1 = BitVec::random(1);
        let v2 = BitVec::random(2);
        coo.push(50, 75, v1.clone());
        coo.push(75, 50, v2.clone());

        let csr_orig = coo.to_csr();
        let hyper = HyperCsrStorage::from_csr(&csr_orig);
        let csr_back = hyper.to_csr();

        assert_eq!(csr_back.get(50, 75).unwrap(), &v1);
        assert_eq!(csr_back.get(75, 50).unwrap(), &v2);
    }

    #[test]
    fn test_hyper_csr_memory_savings() {
        // 1000 nodes, only 3 have entries → saves 997 row pointers
        let mut coo = CooStorage::new(1000, 1000);
        coo.push(0, 1, BitVec::random(1));
        coo.push(500, 501, BitVec::random(2));
        coo.push(999, 0, BitVec::random(3));

        let csr = coo.to_csr();
        let hyper = HyperCsrStorage::from_csr(&csr);

        // Full CSR: 1001 row_ptrs. HyperCSR: 4 row_ptrs + 3 row_ids = 7
        let savings = hyper.row_ptr_savings();
        assert!(savings >= 997, "savings={}, expected ≥997", savings);
        // nnz/nrows = 3/1000 = 0.003 < 0.1 threshold
        assert!(HyperCsrStorage::should_use(3, 1000));
    }

    #[test]
    fn test_hyper_csr_row_iter() {
        let mut coo = CooStorage::new(10, 10);
        coo.push(3, 1, BitVec::random(1));
        coo.push(3, 5, BitVec::random(2));
        coo.push(7, 2, BitVec::random(3));

        let csr = coo.to_csr();
        let hyper = HyperCsrStorage::from_csr(&csr);

        let row3: Vec<usize> = hyper.row(3).unwrap().map(|(c, _)| c).collect();
        assert_eq!(row3, vec![1, 5]);
        assert!(hyper.row(0).is_none());
    }

    #[test]
    fn test_should_use_heuristic() {
        assert!(HyperCsrStorage::should_use(5, 1000));    // 0.005 < 0.1
        assert!(!HyperCsrStorage::should_use(200, 1000)); // 0.2 > 0.1
        assert!(!HyperCsrStorage::should_use(0, 0));      // empty
    }
}
