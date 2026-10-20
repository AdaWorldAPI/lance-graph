//! D-RMAP-2 tile arm: Boolean relationship composition and triangle counting
//! over one 256 × 256 focus map, as a bit matrix and as a BF16 16 × 16 tile
//! GEMM.
//!
//! STATUS: PROBE. Results only.
//!
//! - **Bit arm.** `A` is 256 rows of 256 bits (8 KiB). Composition
//!   `C = A ∘ B` is the outer-product form: for every coordinate `k`, each row
//!   `i` whose bit `k` is set ORs in row `k` of `B`. Every row is addressed by
//!   its own coordinate (`i`, `k`); no position is taken from a stored value.
//! - **GEMM arm.** `A` staged as dense BF16 (0 or 1.0, 128 KiB), multiplied by
//!   `ndarray::simd::bf16_tile_gemm_16x16_packed` in 16 × 16 output blocks with
//!   `K = 256`, then thresholded. For 0/1 operands the products are exact, and
//!   every count is at most 256, far below 2^24, so `count > 0` equals Boolean
//!   composition (D-RMAP-2 §3). The counts themselves are walk counts, which
//!   the bit arm cannot give without popcounts.
//!
//! Triangles: `T = Σ_{i,j,k} A[i,j]·A[j,k]·A[k,i]`, the directed 3-cycles
//! counted once per rotation. Both arms compute it, against a triple-loop
//! oracle. Nothing here applies to weighted, negated or evidence-bearing
//! relations; that is the boundary D-RMAP-2 §3 states.

use ndarray::simd::{
    PackedBf16B, bf16_tile_gemm_16x16, bf16_tile_gemm_16x16_packed, bf16_tile_gemm_tier,
};

/// Focus side.
pub const N: usize = 256;
const W: usize = N / 64;
const BF16_ONE: u16 = 0x3F80;

/// A Boolean relation over one focus: row `i`, bit `j` set when `A[i, j]`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BitMat {
    rows: Vec<[u64; W]>,
}

impl BitMat {
    pub fn empty() -> Self {
        BitMat {
            rows: vec![[0; W]; N],
        }
    }

    pub fn from_edges(edges: impl IntoIterator<Item = (u8, u8)>) -> Self {
        let mut m = Self::empty();
        for (i, j) in edges {
            m.set(i as usize, j as usize);
        }
        m
    }

    pub fn set(&mut self, i: usize, j: usize) {
        self.rows[i][j >> 6] |= 1 << (j & 63);
    }

    pub fn get(&self, i: usize, j: usize) -> bool {
        self.rows[i][j >> 6] >> (j & 63) & 1 == 1
    }

    pub fn row(&self, i: usize) -> &[u64; W] {
        &self.rows[i]
    }

    pub fn edges(&self) -> usize {
        self.rows
            .iter()
            .flatten()
            .map(|w| w.count_ones() as usize)
            .sum()
    }

    pub fn transpose(&self) -> BitMat {
        let mut t = Self::empty();
        for i in 0..N {
            for j in 0..N {
                if self.get(i, j) {
                    t.set(j, i);
                }
            }
        }
        t
    }

    /// `C = A ∘ B` (Boolean), outer-product form over the transpose of `A`:
    /// for each `k`, the rows `i` with `A[i, k]` OR in `B[k]`.
    pub fn compose(&self, at: &BitMat, b: &BitMat) -> BitMat {
        let mut c = Self::empty();
        for k in 0..N {
            let bk = b.rows[k];
            for (wi, &word) in at.rows[k].iter().enumerate() {
                let mut w = word;
                while w != 0 {
                    let i = wi * 64 + w.trailing_zeros() as usize;
                    let ci = &mut c.rows[i];
                    for x in 0..W {
                        ci[x] |= bk[x];
                    }
                    w &= w - 1;
                }
            }
        }
        c
    }

    /// Walk counts `(A · B)[i, j] = popcount(row_i(A) ∧ row_j(Bᵀ))`, the bit
    /// arm's answer to the counts GEMM gives. Row-major 256 × 256.
    pub fn square_counts(&self, bt: &BitMat) -> Vec<u32> {
        let mut c = vec![0u32; N * N];
        for i in 0..N {
            let ri = self.rows[i];
            for j in 0..N {
                let cj = bt.rows[j];
                let mut n = 0u32;
                for x in 0..W {
                    n += (ri[x] & cj[x]).count_ones();
                }
                c[i * N + j] = n;
            }
        }
        c
    }

    /// Directed 3-cycles: for every edge `(i, j)`, the `k` with `A[j, k]` and
    /// `A[k, i]` is `row_j(A) ∧ row_i(Aᵀ)`; popcount and sum.
    pub fn triangles(&self, at: &BitMat) -> u64 {
        let mut t = 0u64;
        for i in 0..N {
            let ti = at.rows[i];
            for (wj, &word) in self.rows[i].iter().enumerate() {
                let mut w = word;
                while w != 0 {
                    let j = wj * 64 + w.trailing_zeros() as usize;
                    let rj = self.rows[j];
                    for x in 0..W {
                        t += u64::from((rj[x] & ti[x]).count_ones());
                    }
                    w &= w - 1;
                }
            }
        }
        t
    }
}

/// `A` staged for the GEMM: row-major BF16, plus each 16-column block of `A`
/// as a packed `B` operand (`K = 256`).
pub struct Bf16Staged {
    a: Vec<u16>,
    b_blocks: Vec<PackedBf16B>,
    /// The same blocks row-major, for the polyfill entry. On a host without
    /// AMX or `avx512bf16`, the packed entry unpacks (and allocates) per call
    /// before taking the polyfill, so the row-major entry is that tier's
    /// honest cost.
    b_rows: Vec<Vec<u16>>,
}

impl Bf16Staged {
    pub fn stage(m: &BitMat) -> Self {
        let mut a = vec![0u16; N * N];
        for i in 0..N {
            for j in 0..N {
                if m.get(i, j) {
                    a[i * N + j] = BF16_ONE;
                }
            }
        }
        let b_rows: Vec<Vec<u16>> = (0..N / 16)
            .map(|bj| {
                let mut blk = vec![0u16; N * 16];
                for k in 0..N {
                    blk[k * 16..k * 16 + 16]
                        .copy_from_slice(&a[k * N + bj * 16..k * N + bj * 16 + 16]);
                }
                blk
            })
            .collect();
        let b_blocks = b_rows.iter().map(|b| PackedBf16B::pack(b, N)).collect();
        Bf16Staged {
            a,
            b_blocks,
            b_rows,
        }
    }

    /// Bytes held: the BF16 matrix, the packed blocks and the row-major
    /// blocks kept for the polyfill entry.
    pub fn bytes(&self) -> usize {
        self.a.len() * 2
            + self
                .b_blocks
                .iter()
                .map(|b| b.data().len() * 2)
                .sum::<usize>()
            + self.b_rows.iter().map(|b| b.len() * 2).sum::<usize>()
    }

    /// Walk counts through the row-major polyfill entry (no pack, no unpack).
    pub fn square_rowmajor(&self, c: &mut [f32]) {
        assert_eq!(c.len(), N * N);
        let mut blk = [0f32; 256];
        for bi in 0..N / 16 {
            let a_rows = &self.a[bi * 16 * N..(bi + 1) * 16 * N];
            for (bj, b) in self.b_rows.iter().enumerate() {
                blk.fill(0.0);
                bf16_tile_gemm_16x16(a_rows, b, &mut blk, N);
                for r in 0..16 {
                    let at = (bi * 16 + r) * N + bj * 16;
                    c[at..at + 16].copy_from_slice(&blk[r * 16..r * 16 + 16]);
                }
            }
        }
    }

    /// Walk counts `C = A · A` in f32, row-major 256 × 256, through the
    /// tier-dispatched packed entry.
    pub fn square(&self, c: &mut [f32]) {
        assert_eq!(c.len(), N * N);
        let mut blk = [0f32; 256];
        for bi in 0..N / 16 {
            let a_rows = &self.a[bi * 16 * N..(bi + 1) * 16 * N];
            for (bj, b) in self.b_blocks.iter().enumerate() {
                blk.fill(0.0);
                bf16_tile_gemm_16x16_packed(a_rows, b, &mut blk);
                for r in 0..16 {
                    let at = (bi * 16 + r) * N + bj * 16;
                    c[at..at + 16].copy_from_slice(&blk[r * 16..r * 16 + 16]);
                }
            }
        }
    }
}

/// The tier ndarray will take on this host, for the report.
pub fn tier() -> &'static str {
    bf16_tile_gemm_tier()
}

/// Boolean composition from walk counts: `C[i, j] = counts[i, j] > 0`.
pub fn threshold(counts: &[f32]) -> BitMat {
    let mut m = BitMat::empty();
    for i in 0..N {
        for j in 0..N {
            if counts[i * N + j] > 0.0 {
                m.set(i, j);
            }
        }
    }
    m
}

/// Triangles from walk counts: `Σ_{i,j} (A²)[j, i] · A[i, j]`.
pub fn triangles_from_square(m: &BitMat, sq: &[f32]) -> u64 {
    let mut t = 0u64;
    for i in 0..N {
        for j in 0..N {
            if m.get(i, j) {
                t += sq[j * N + i] as u64;
            }
        }
    }
    t
}

/// Independent oracle: a triple loop over the bits.
pub fn oracle_square(m: &BitMat) -> Vec<u32> {
    let mut c = vec![0u32; N * N];
    for i in 0..N {
        for k in 0..N {
            if m.get(i, k) {
                for j in 0..N {
                    if m.get(k, j) {
                        c[i * N + j] += 1;
                    }
                }
            }
        }
    }
    c
}

pub fn oracle_triangles(m: &BitMat) -> u64 {
    let sq = oracle_square(m);
    let mut t = 0u64;
    for i in 0..N {
        for j in 0..N {
            if m.get(i, j) {
                t += u64::from(sq[j * N + i]);
            }
        }
    }
    t
}

/// A deterministic fixture: each edge with probability `p_permille / 1000`;
/// `hubs` rows (and columns) are dense, which gives a skewed degree
/// distribution.
pub fn fixture(p_permille: u32, hubs: usize, seed: u64) -> BitMat {
    let mut s = seed;
    let mut next = || {
        s = s.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = s;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    };
    let mut m = BitMat::empty();
    for i in 0..N {
        for j in 0..N {
            let hub = i < hubs || j < hubs;
            if hub || (next() % 1000) < u64::from(p_permille) {
                m.set(i, j);
            }
        }
    }
    m
}
