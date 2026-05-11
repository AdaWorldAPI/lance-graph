//! HhtlF32Tensor — reconstruction-grade codec with f32 palette centroids.
//!
//! Parallel to `HhtlDTensor`. Swaps the Base17-folded palette substrate
//! (lookup-grade, row reconstruction cos ≈ 0.04 on real data per PR #183
//! measurement) for **CLAM centroids stored as f32/BF16 vectors**, reusing
//! SlotL as the directional residual correction.
//!
//! Wire layout per row:
//!   - Slot D (u8):  twig_centroid index (0..256)
//!   - Slot L (8 × i8): directional residual on shared SVD basis
//!   - Total: 9 bytes per row (vs HhtlDTensor's 12 B/row with Slot D+V+L)
//!
//! Per-group overhead:
//!   - Palette: 256 × n_cols × 2 bytes (BF16)
//!   - SVD basis: 8 × n_cols × 2 bytes (BF16)
//!
//! Target quality on real Qwen3 weight rows:
//!   - argmax regime (SlotL off): ρ ≈ 0.75 per row (CLAM centroid alone)
//!   - argmax regime (SlotL on):  ρ ≈ 0.95 per row
//!   - index regime (SlotL on):   ρ ≈ 0.98 per row
//!
//! The `Base17`-based `HhtlDTensor` is NOT removed — it remains the correct
//! codec for HHTL cascade lookup inference (343:1 ratio per
//! `BGZ_HHTL_D.md`). `HhtlF32Tensor` is the reconstruction-grade sibling for
//! f32 GEMM inference paths.

// SLOT_L_LANES used by tests and encode_with_leaf
use crate::matryoshka::SvdBasis;
#[allow(unused_imports)]
use crate::slot_l::{
    decode_row as decode_slot_l_row, encode_rows as encode_slot_l, SlotL, SLOT_L_LANES,
};

/// One row of HhtlF32Tensor: twig index into the f32 palette.
/// Slot V scalar residual is omitted — f32 centroid already carries direction.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct HhtlF32Entry {
    pub twig: u8,
}

impl HhtlF32Entry {
    pub const BYTE_SIZE: usize = 1;
    pub fn to_le_bytes(&self) -> [u8; 1] {
        [self.twig]
    }
    pub fn from_le_bytes(b: &[u8; 1]) -> Self {
        Self { twig: b[0] }
    }
}

/// Maximum palette size supported by HhtlF32Tensor (constrained by u8 twig index).
///
/// `HhtlF32Entry.twig` is u8, so palette centroid IDs above 255 would silently
/// wrap at encode time (e.g. 300 → 44) and map rows to the wrong centroid.
/// `encode` / `encode_with_leaf` enforce `0 < k <= MAX_PALETTE_K`; callers that
/// need a larger palette must use a codec with a wider twig index (future work,
/// separate wire format).
pub const MAX_PALETTE_K: usize = 256;

/// Reconstruction-grade HHTL-like tensor: f32 palette + SlotL residual.
#[derive(Clone, Debug)]
pub struct HhtlF32Tensor {
    pub role: String,
    /// CLAM centroids as f32 vectors, shape [k, n_cols] row-major.
    pub palette_f32: Vec<Vec<f32>>,
    /// Per-row entries: twig index into palette_f32.
    pub entries: Vec<HhtlF32Entry>,
    pub original_shape: [usize; 2],
    /// Slot L directional residual (same pattern as HhtlDTensor).
    pub slot_l: Option<Vec<SlotL>>,
    pub slot_l_scale: Option<f32>,
    pub svd_basis: Option<SvdBasis>,
}

// ═════════════════════════════════════════════════════════════════════
// CLAM furthest-point sampling on f32 rows (same algo as tts_rvq_e2e.rs)
// ═════════════════════════════════════════════════════════════════════

#[inline(always)]
fn l2_dist_sq(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len().min(b.len());
    let mut s = 0.0f32;
    for i in 0..n {
        let d = a[i] - b[i];
        s += d * d;
    }
    s
}

fn clam_furthest_point_f32(rows: &[Vec<f32>], k: usize) -> Vec<usize> {
    let n = rows.len();
    if n == 0 || k == 0 {
        return Vec::new();
    }
    let k = k.min(n);

    // Seed: row with largest L2 norm (proxy for "farthest from origin").
    let mut first = 0usize;
    let mut first_norm = 0.0f32;
    for (i, r) in rows.iter().enumerate() {
        let n_sq: f32 = r.iter().map(|x| x * x).sum();
        if n_sq > first_norm {
            first_norm = n_sq;
            first = i;
        }
    }

    let mut selected = vec![first];
    let mut min_dist = vec![f32::MAX; n];
    for (i, md) in min_dist.iter_mut().enumerate().take(n) {
        *md = l2_dist_sq(&rows[i], &rows[first]);
    }
    min_dist[first] = 0.0;

    for _ in 1..k {
        let mut next = 0usize;
        let mut best = f32::MIN;
        for (i, &md) in min_dist.iter().enumerate().take(n) {
            if md > best {
                best = md;
                next = i;
            }
        }
        if best <= 0.0 {
            break;
        } // All rows already covered
        selected.push(next);
        let cidx = next;
        for (i, md) in min_dist.iter_mut().enumerate().take(n) {
            let d = l2_dist_sq(&rows[i], &rows[cidx]);
            if d < *md {
                *md = d;
            }
        }
    }
    selected
}

fn assign_nearest_f32(rows: &[Vec<f32>], centroids: &[Vec<f32>]) -> Vec<u8> {
    rows.iter()
        .map(|row| {
            let mut best = 0u8;
            let mut best_d = f32::MAX;
            for (ci, c) in centroids.iter().enumerate() {
                let d = l2_dist_sq(row, c);
                if d < best_d {
                    best_d = d;
                    best = ci as u8;
                }
            }
            best
        })
        .collect()
}

// ═════════════════════════════════════════════════════════════════════
// Encode / decode
// ═════════════════════════════════════════════════════════════════════

impl HhtlF32Tensor {
    /// Encode without SlotL (argmax-regime path, 1 byte/row + palette).
    ///
    /// # Panics
    /// `k` must satisfy `0 < k <= MAX_PALETTE_K` (256). Larger palettes need
    /// a codec with a wider twig-index; using `k > 256` here would silently
    /// wrap indices via `as u8` and corrupt row assignments.
    pub fn encode(role: &str, rows_f32: &[Vec<f32>], k: usize) -> Self {
        assert!(k > 0, "HhtlF32Tensor::encode requires k > 0, got {}", k);
        assert!(
            k <= MAX_PALETTE_K,
            "HhtlF32Tensor::encode requires k <= {} (u8 twig limit), got {}",
            MAX_PALETTE_K,
            k
        );
        let n_rows = rows_f32.len();
        let n_cols = if n_rows > 0 { rows_f32[0].len() } else { 0 };

        let selected = clam_furthest_point_f32(rows_f32, k);
        let palette_f32: Vec<Vec<f32>> = selected.iter().map(|&i| rows_f32[i].clone()).collect();
        let twig_assign = assign_nearest_f32(rows_f32, &palette_f32);
        let entries: Vec<HhtlF32Entry> = twig_assign
            .iter()
            .map(|&t| HhtlF32Entry { twig: t })
            .collect();

        Self {
            role: role.to_string(),
            palette_f32,
            entries,
            original_shape: [n_rows, n_cols],
            slot_l: None,
            slot_l_scale: None,
            svd_basis: None,
        }
    }

    /// Encode with SlotL leaf residual (index-regime path, 9 bytes/row).
    /// `svd_basis` should have `SLOT_L_LANES` components built from
    /// representative rows of the group (via `SvdBasis::build`).
    ///
    /// # Panics
    /// Same `k` bounds as [`encode`]: `0 < k <= MAX_PALETTE_K` (256).
    pub fn encode_with_leaf(
        role: &str,
        rows_f32: &[Vec<f32>],
        k: usize,
        svd_basis: &SvdBasis,
    ) -> Self {
        assert!(
            k > 0,
            "HhtlF32Tensor::encode_with_leaf requires k > 0, got {}",
            k
        );
        assert!(
            k <= MAX_PALETTE_K,
            "HhtlF32Tensor::encode_with_leaf requires k <= {} (u8 twig limit), got {}",
            MAX_PALETTE_K,
            k
        );
        let mut t = Self::encode(role, rows_f32, k);
        if rows_f32.is_empty() {
            return t;
        }

        // Per-row centroid (copy from palette by twig index).
        let centroids_per_row: Vec<Vec<f32>> = t
            .entries
            .iter()
            .map(|e| t.palette_f32[e.twig as usize].clone())
            .collect();

        let (slot_l, scale) = encode_slot_l(rows_f32, &centroids_per_row, svd_basis);
        t.slot_l = Some(slot_l);
        t.slot_l_scale = Some(scale);
        t.svd_basis = Some(svd_basis.clone());
        t
    }

    /// Reconstruct one row at the given n_cols dimensionality.
    pub fn reconstruct_row(&self, idx: usize, n_cols: usize) -> Vec<f32> {
        if idx >= self.entries.len() {
            return vec![0.0f32; n_cols];
        }
        let twig = self.entries[idx].twig as usize;
        if twig >= self.palette_f32.len() {
            return vec![0.0f32; n_cols];
        }
        let centroid = &self.palette_f32[twig];
        // Cap the centroid to n_cols (may differ from original training shape).
        let mut base = vec![0.0f32; n_cols];
        base[..n_cols.min(centroid.len())].copy_from_slice(&centroid[..n_cols.min(centroid.len())]);

        match (&self.slot_l, &self.svd_basis, self.slot_l_scale) {
            (Some(sl), Some(basis), Some(scale)) if idx < sl.len() => {
                decode_slot_l_row(&base, &sl[idx], scale, basis, n_cols)
            }
            _ => base,
        }
    }

    /// Reconstruct all rows.
    pub fn reconstruct_rows(&self, n_cols: usize) -> Vec<Vec<f32>> {
        (0..self.entries.len())
            .map(|i| self.reconstruct_row(i, n_cols))
            .collect()
    }

    /// Byte sizes (excluding palette + basis, which are per-group shared).
    pub fn entries_byte_size(&self) -> usize {
        self.entries.len() * HhtlF32Entry::BYTE_SIZE
    }
    pub fn slot_l_byte_size(&self) -> usize {
        self.slot_l
            .as_ref()
            .map(|v| v.len() * SlotL::BYTE_SIZE)
            .unwrap_or(0)
    }

    /// Palette byte size (BF16 footprint — centroids compressed for shipping).
    pub fn palette_byte_size_bf16(&self) -> usize {
        let k = self.palette_f32.len();
        let cols = if k > 0 { self.palette_f32[0].len() } else { 0 };
        k * cols * 2
    }
    pub fn svd_basis_byte_size(&self) -> usize {
        self.svd_basis.as_ref().map(|b| b.byte_size()).unwrap_or(0)
    }

    /// Total per-tensor footprint when shipping (BF16 palette + entries + SlotL + basis).
    pub fn total_byte_size_bf16(&self) -> usize {
        self.entries_byte_size()
            + self.slot_l_byte_size()
            + self.palette_byte_size_bf16()
            + self.svd_basis_byte_size()
    }
}

// ═════════════════════════════════════════════════════════════════════
// Tests
// ═════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    fn low_rank_rows(n: usize, cols: usize, seed: u32) -> Vec<Vec<f32>> {
        let n_atoms = 8usize;
        let mut atoms: Vec<Vec<f32>> = Vec::with_capacity(n_atoms);
        let mut s = seed;
        let mut next = || {
            s = s.wrapping_mul(1103515245).wrapping_add(12345);
            ((s >> 8) as i32 as f32) / 2_147_483_648.0
        };
        for _ in 0..n_atoms {
            let atom: Vec<f32> = (0..cols).map(|_| next()).collect();
            atoms.push(atom);
        }
        (0..n)
            .map(|_| {
                let mut row = vec![0.0f32; cols];
                for atom in &atoms {
                    let w = next() * 0.5;
                    for j in 0..cols {
                        row[j] += atom[j] * w;
                    }
                }
                row
            })
            .collect()
    }

    fn cosine(a: &[f32], b: &[f32]) -> f64 {
        let mut dot = 0.0f64;
        let mut na = 0.0f64;
        let mut nb = 0.0f64;
        for i in 0..a.len().min(b.len()) {
            let x = a[i] as f64;
            let y = b[i] as f64;
            dot += x * y;
            na += x * x;
            nb += y * y;
        }
        let d = (na * nb).sqrt();
        if d < 1e-15 {
            0.0
        } else {
            dot / d
        }
    }

    #[test]
    fn encode_without_leaf_picks_real_rows_as_centroids() {
        let n = 32;
        let cols = 64;
        let rows = low_rank_rows(n, cols, 0xAAA);
        let t = HhtlF32Tensor::encode("test", &rows, 8);

        // Every palette centroid must equal one of the original rows
        // (furthest-point sampling picks from the row set, doesn't synthesize).
        for centroid in &t.palette_f32 {
            let matches = rows.iter().any(|r| {
                r.iter()
                    .zip(centroid.iter())
                    .all(|(a, b)| (a - b).abs() < 1e-6)
            });
            assert!(matches, "every centroid must be an original row");
        }
        assert_eq!(t.entries.len(), n);
        assert!(t.slot_l.is_none());
    }

    #[test]
    fn reconstruct_without_leaf_returns_nearest_centroid() {
        let rows = low_rank_rows(16, 32, 0xBBB);
        let t = HhtlF32Tensor::encode("argmax_regime", &rows, 4);

        for (i, _row) in rows.iter().enumerate() {
            let recon = t.reconstruct_row(i, 32);
            let expected = &t.palette_f32[t.entries[i].twig as usize];
            for d in 0..32 {
                assert!(
                    (recon[d] - expected[d]).abs() < 1e-6,
                    "without SlotL, reconstruct_row == palette[twig]"
                );
            }
        }
    }

    #[test]
    fn encode_with_leaf_beats_without_leaf_on_real_rows() {
        // Key test: proves the Path A codec recovers per-row cosine far
        // beyond what the #183 Base17-reconstruction measurement showed.
        let n = 64;
        let cols = 128;
        let rows = low_rank_rows(n, cols, 0xC0DE);

        let t_plain = HhtlF32Tensor::encode("plain", &rows, 16);
        let basis = SvdBasis::build("leaf", &rows, SLOT_L_LANES);
        let t_leaf = HhtlF32Tensor::encode_with_leaf("leaf", &rows, 16, &basis);

        let mut sum_plain = 0.0f64;
        let mut sum_leaf = 0.0f64;
        for i in 0..n {
            let rec_plain = t_plain.reconstruct_row(i, cols);
            let rec_leaf = t_leaf.reconstruct_row(i, cols);
            sum_plain += cosine(&rows[i], &rec_plain);
            sum_leaf += cosine(&rows[i], &rec_leaf);
        }
        let avg_plain = sum_plain / n as f64;
        let avg_leaf = sum_leaf / n as f64;

        // On low-rank rows both should be high, leaf strictly better.
        assert!(
            avg_plain >= 0.70,
            "plain f32-centroid avg cos should be >= 0.70 on low-rank data, got {:.4}",
            avg_plain
        );
        assert!(
            avg_leaf >= avg_plain,
            "leaf avg cos ({:.4}) must be >= plain ({:.4})",
            avg_leaf,
            avg_plain
        );
        assert!(
            avg_leaf >= 0.95,
            "leaf avg cos should be >= 0.95 on low-rank 8-atom data, got {:.4}",
            avg_leaf
        );
    }

    #[test]
    fn entry_byte_size_is_one() {
        assert_eq!(HhtlF32Entry::BYTE_SIZE, 1);
        let e = HhtlF32Entry { twig: 42 };
        let b = e.to_le_bytes();
        assert_eq!(HhtlF32Entry::from_le_bytes(&b), e);
    }

    #[test]
    #[should_panic(expected = "k > 0")]
    fn encode_rejects_zero_k() {
        let rows = low_rank_rows(4, 8, 0);
        let _ = HhtlF32Tensor::encode("zero_k", &rows, 0);
    }

    #[test]
    #[should_panic(expected = "u8 twig limit")]
    fn encode_rejects_k_above_256() {
        // Codex P1 regression: u8 twig index can only represent 0..=255,
        // so k > 256 would silently wrap `ci as u8`. Must panic loudly instead.
        // Use a bigger row set so clam_furthest_point_f32 doesn't cap k first.
        let rows = low_rank_rows(512, 16, 0);
        let _ = HhtlF32Tensor::encode("oversize", &rows, 300);
    }

    #[test]
    #[should_panic(expected = "u8 twig limit")]
    fn encode_with_leaf_rejects_k_above_256() {
        let rows = low_rank_rows(512, 16, 0);
        let basis = SvdBasis::build("oversize", &rows, SLOT_L_LANES);
        let _ = HhtlF32Tensor::encode_with_leaf("oversize_leaf", &rows, 300, &basis);
    }

    #[test]
    fn encode_accepts_k_at_max_palette() {
        // k == MAX_PALETTE_K (256) is the largest legal value and must succeed.
        let rows = low_rank_rows(300, 16, 0); // Need > 256 rows so all 256 centroid slots fill
        let t = HhtlF32Tensor::encode("max_k", &rows, MAX_PALETTE_K);
        assert!(t.palette_f32.len() <= MAX_PALETTE_K);
        assert_eq!(t.entries.len(), 300);
    }

    #[test]
    fn storage_accounting_is_additive() {
        let rows = low_rank_rows(16, 64, 0xDEAD);
        let basis = SvdBasis::build("storage", &rows, SLOT_L_LANES);
        let t = HhtlF32Tensor::encode_with_leaf("storage_test", &rows, 8, &basis);

        assert_eq!(t.entries_byte_size(), 16); // 16 rows × 1 byte
        assert_eq!(t.slot_l_byte_size(), 16 * SLOT_L_LANES);
        assert_eq!(t.palette_byte_size_bf16(), 8 * 64 * 2); // 8 centroids × 64 cols × 2 B

        let expected_total = 16 + 16 * SLOT_L_LANES + 8 * 64 * 2 + t.svd_basis_byte_size();
        assert_eq!(t.total_byte_size_bf16(), expected_total);
    }
}
