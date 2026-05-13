//! PaletteMatrix: sparse CSR matrix with PaletteEdge values.
//!
//! Unlike blasgraph's `CsrStorage` (which stores `Vec<BitVec>` = 2KB/entry),
//! PaletteMatrix stores 3-byte PaletteEdge entries. This enables:
//! - Matrix-matrix multiply (mxm) under palette semiring for multi-hop paths
//! - Conversion to ScalarCsr for distance-weighted operations
//!
//! The compose operation uses PaletteSemiring compose tables to combine
//! edges: path(a→b) + path(b→c) = compose(a→b, b→c) per S/P/O plane.

use crate::palette::PaletteEdge;
use crate::distance_matrix::SpoDistanceMatrices;
use crate::scalar_sparse::ScalarCsr;

/// Sparse CSR matrix with PaletteEdge values (3 bytes per entry).
#[derive(Clone, Debug)]
pub struct PaletteMatrix {
    pub nrows: usize,
    pub ncols: usize,
    /// Row pointers: `row_ptr[i]..row_ptr[i+1]` indexes into col_idx/vals.
    pub row_ptr: Vec<usize>,
    /// Column indices.
    pub col_idx: Vec<usize>,
    /// PaletteEdge values (3 bytes each: s_idx, p_idx, o_idx).
    pub vals: Vec<PaletteEdge>,
}

impl PaletteMatrix {
    /// Create an empty matrix.
    pub fn new(nrows: usize, ncols: usize) -> Self {
        PaletteMatrix {
            nrows,
            ncols,
            row_ptr: vec![0; nrows + 1],
            col_idx: Vec::new(),
            vals: Vec::new(),
        }
    }

    /// Number of stored entries.
    pub fn nnz(&self) -> usize {
        self.vals.len()
    }

    /// Get value at (row, col). Returns None if not stored.
    pub fn get(&self, row: usize, col: usize) -> Option<&PaletteEdge> {
        let start = self.row_ptr[row];
        let end = self.row_ptr[row + 1];
        for idx in start..end {
            if self.col_idx[idx] == col {
                return Some(&self.vals[idx]);
            }
        }
        None
    }

    /// Build from adjacency data: (source, target, PaletteEdge) triples.
    pub fn from_triples(
        nrows: usize,
        ncols: usize,
        triples: &[(usize, usize, PaletteEdge)],
    ) -> Self {
        // Count entries per row
        let mut row_counts = vec![0usize; nrows];
        for &(r, _, _) in triples {
            row_counts[r] += 1;
        }

        let mut row_ptr = Vec::with_capacity(nrows + 1);
        row_ptr.push(0);
        for &c in &row_counts {
            row_ptr.push(row_ptr.last().unwrap() + c);
        }

        let nnz = triples.len();
        let mut col_idx = vec![0usize; nnz];
        let mut vals = vec![PaletteEdge { s_idx: 0, p_idx: 0, o_idx: 0 }; nnz];
        let mut offsets = vec![0usize; nrows];

        for &(r, c, pe) in triples {
            let idx = row_ptr[r] + offsets[r];
            col_idx[idx] = c;
            vals[idx] = pe;
            offsets[r] += 1;
        }

        PaletteMatrix { nrows, ncols, row_ptr, col_idx, vals }
    }

    /// Matrix-matrix multiply under palette semiring.
    ///
    /// C[i,j] = "best" composition of A[i,k] and B[k,j] for any k.
    /// "Best" = minimum SPO distance of the composed path.
    ///
    /// compose_s/p/o: k×k compose tables per plane.
    pub fn mxm(
        a: &PaletteMatrix,
        b: &PaletteMatrix,
        compose_s: &[u8],
        compose_p: &[u8],
        compose_o: &[u8],
        k_pal: usize,
        dm: &SpoDistanceMatrices,
    ) -> PaletteMatrix {
        assert_eq!(a.ncols, b.nrows);
        let nrows = a.nrows;
        let ncols = b.ncols;

        let mut row_ptr = Vec::with_capacity(nrows + 1);
        let mut col_idx = Vec::new();
        let mut vals = Vec::new();
        row_ptr.push(0);

        for i in 0..nrows {
            // Accumulate best path to each column
            let mut best: Vec<Option<(PaletteEdge, u32)>> = vec![None; ncols];

            let a_start = a.row_ptr[i];
            let a_end = a.row_ptr[i + 1];
            for a_idx in a_start..a_end {
                let mid = a.col_idx[a_idx];
                let a_pe = &a.vals[a_idx];

                let b_start = b.row_ptr[mid];
                let b_end = b.row_ptr[mid + 1];
                for b_idx in b_start..b_end {
                    let j = b.col_idx[b_idx];
                    let b_pe = &b.vals[b_idx];

                    // Compose: path(i→mid) + path(mid→j)
                    let cs = compose_s[a_pe.s_idx as usize * k_pal + b_pe.s_idx as usize];
                    let cp = compose_p[a_pe.p_idx as usize * k_pal + b_pe.p_idx as usize];
                    let co = compose_o[a_pe.o_idx as usize * k_pal + b_pe.o_idx as usize];

                    // Distance of the composed path (lower = better path)
                    let dist = dm.spo_distance(cs, cp, co, 0, 0, 0);

                    match &best[j] {
                        None => {
                            best[j] = Some((PaletteEdge { s_idx: cs, p_idx: cp, o_idx: co }, dist));
                        }
                        Some((_, prev_dist)) if dist < *prev_dist => {
                            best[j] = Some((PaletteEdge { s_idx: cs, p_idx: cp, o_idx: co }, dist));
                        }
                        _ => {}
                    }
                }
            }

            // Emit row entries
            for (j, best_j) in best.iter().enumerate().take(ncols) {
                if let Some((pe, _)) = best_j {
                    col_idx.push(j);
                    vals.push(*pe);
                }
            }
            row_ptr.push(col_idx.len());
        }

        PaletteMatrix { nrows, ncols, row_ptr, col_idx, vals }
    }

    /// Convert to ScalarCsr using distance matrices.
    ///
    /// Each PaletteEdge becomes its combined S+P+O distance from origin (0,0,0).
    pub fn to_distance_csr(&self, dm: &SpoDistanceMatrices) -> ScalarCsr {
        let mut csr = ScalarCsr::new(self.nrows, self.ncols);
        let mut row_ptr = Vec::with_capacity(self.nrows + 1);
        let mut col_idx = Vec::new();
        let mut vals = Vec::new();
        row_ptr.push(0);

        for i in 0..self.nrows {
            let start = self.row_ptr[i];
            let end = self.row_ptr[i + 1];
            for idx in start..end {
                let pe = &self.vals[idx];
                let d = dm.spo_distance(pe.s_idx, pe.p_idx, pe.o_idx, 0, 0, 0);
                col_idx.push(self.col_idx[idx]);
                vals.push(d as f32);
            }
            row_ptr.push(col_idx.len());
        }

        csr.row_ptr = row_ptr;
        csr.col_idx = col_idx;
        csr.vals = vals;
        csr
    }

    /// Byte size of the matrix (approximate).
    pub fn byte_size(&self) -> usize {
        self.row_ptr.len() * 8 + self.col_idx.len() * 8 + self.vals.len() * 3
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::base17::Base17;
    use crate::palette::Palette;
    use crate::palette_semiring::PaletteSemiring;
    use crate::distance_matrix::SpoDistanceMatrices;
    use crate::BASE_DIM;

    fn make_palette(k: usize) -> Palette {
        let entries = (0..k).map(|i| {
            let mut dims = [0i16; BASE_DIM];
            for d in 0..BASE_DIM {
                dims[d] = ((i * 97 + d * 31) % 512) as i16 - 256;
            }
            Base17 { dims }
        }).collect();
        Palette { entries }
    }

    #[test]
    fn test_from_triples() {
        let triples = vec![
            (0, 1, PaletteEdge { s_idx: 1, p_idx: 2, o_idx: 3 }),
            (0, 2, PaletteEdge { s_idx: 4, p_idx: 5, o_idx: 6 }),
            (1, 0, PaletteEdge { s_idx: 7, p_idx: 8, o_idx: 9 }),
        ];
        let pm = PaletteMatrix::from_triples(3, 3, &triples);
        assert_eq!(pm.nnz(), 3);
        assert_eq!(pm.get(0, 1).unwrap().s_idx, 1);
        assert_eq!(pm.get(0, 2).unwrap().s_idx, 4);
        assert_eq!(pm.get(1, 0).unwrap().s_idx, 7);
        assert!(pm.get(2, 0).is_none());
    }

    #[test]
    fn test_mxm_2hop() {
        // 3 nodes: 0→1, 1→2. MxM should yield 0→2 (2-hop).
        let pal = make_palette(16);
        let sr = PaletteSemiring::build(&pal);
        let dm = SpoDistanceMatrices::build(&pal, &pal, &pal);

        let pe_01 = PaletteEdge { s_idx: 1, p_idx: 2, o_idx: 3 };
        let pe_12 = PaletteEdge { s_idx: 4, p_idx: 5, o_idx: 6 };

        let a = PaletteMatrix::from_triples(3, 3, &[
            (0, 1, pe_01),
        ]);
        let b = PaletteMatrix::from_triples(3, 3, &[
            (1, 2, pe_12),
        ]);

        let c = PaletteMatrix::mxm(
            &a, &b,
            &sr.compose_table, &sr.compose_table, &sr.compose_table,
            sr.k, &dm,
        );

        // Should have exactly one entry: 0→2
        assert_eq!(c.nnz(), 1);
        let result = c.get(0, 2);
        assert!(result.is_some(), "2-hop path 0→1→2 should produce entry at (0,2)");

        // Verify the composed edge matches manual computation
        let expected_s = sr.compose(pe_01.s_idx, pe_12.s_idx);
        let expected_p = sr.compose(pe_01.p_idx, pe_12.p_idx);
        let expected_o = sr.compose(pe_01.o_idx, pe_12.o_idx);
        let got = result.unwrap();
        assert_eq!(got.s_idx, expected_s);
        assert_eq!(got.p_idx, expected_p);
        assert_eq!(got.o_idx, expected_o);
    }

    #[test]
    fn test_to_distance_csr() {
        let pal = make_palette(16);
        let dm = SpoDistanceMatrices::build(&pal, &pal, &pal);

        let pm = PaletteMatrix::from_triples(3, 3, &[
            (0, 1, PaletteEdge { s_idx: 1, p_idx: 2, o_idx: 3 }),
            (1, 2, PaletteEdge { s_idx: 4, p_idx: 5, o_idx: 6 }),
        ]);

        let csr = pm.to_distance_csr(&dm);
        assert_eq!(csr.nnz(), 2);
        // Distance should be non-negative
        for v in &csr.vals {
            assert!(*v >= 0.0);
        }
    }
}
