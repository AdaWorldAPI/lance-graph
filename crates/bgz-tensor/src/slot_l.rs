//! Slot L — optional 8 × i8 leaf residual on a shared SVD basis.
//!
//! Extends BGZ-HHTL-D from 4 B/row (Slot D + Slot V scalar) to 12 B/row
//! (Slot D + Slot V scalar + Slot L 8×i8). Plugs the "directional residual"
//! gap: scalar residuals can only encode magnitude, not direction. With a
//! shared SVD basis (from `matryoshka::SvdBasis`) and 8 i8 coefficients
//! per row, index-lookup tensors (vocab embeddings, lm_heads) recover
//! per-row identity at ρ ≳ 0.98 — enough for correct row-indexing.
//!
//! Usage pattern (per palette group / per role):
//!
//! ```text
//! 1. Build SvdBasis::build(role, sample_rows, 8) once per group         (shared)
//! 2. For each row:
//!    a. centroid_f32 = palette[twig_centroid].to_f32(n_cols)
//!    b. residual_f32 = row_f32 - centroid_f32
//!    c. coeffs_f32   = SvdBasis::project(residual_f32)   (length 8)
//!    d. quantize coeffs to i8 using shared per-group scale
//!    e. store 8 i8 bytes as SlotL
//! 3. Reconstruct:
//!    a. coeffs_f32 = dequantize(slot_l.bytes, scale)
//!    b. residual_f32 = SvdBasis::reconstruct(coeffs_f32)
//!    c. row_f32 = centroid_f32 + residual_f32
//! ```
//!
//! Storage per row: 8 bytes (i8 × 8). Per-group overhead: one f32 scale +
//! one SvdBasis (8 × n_cols × 2 bytes BF16).

use crate::matryoshka::SvdBasis;

/// Number of SVD components kept per row (fits in 8 bytes as i8).
pub const SLOT_L_LANES: usize = 8;

/// 8 × i8 quantized SVD coefficients for one row's residual.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SlotL {
    pub bytes: [i8; SLOT_L_LANES],
}

impl SlotL {
    /// Byte size of a single Slot L entry.
    pub const BYTE_SIZE: usize = SLOT_L_LANES;

    pub fn zero() -> Self {
        SlotL { bytes: [0i8; SLOT_L_LANES] }
    }

    pub fn to_le_bytes(&self) -> [u8; SLOT_L_LANES] {
        let mut out = [0u8; SLOT_L_LANES];
        for (i, &b) in self.bytes.iter().enumerate() {
            out[i] = b as u8;
        }
        out
    }

    pub fn from_le_bytes(bytes: &[u8; SLOT_L_LANES]) -> Self {
        let mut arr = [0i8; SLOT_L_LANES];
        for (i, &b) in bytes.iter().enumerate() {
            arr[i] = b as i8;
        }
        SlotL { bytes: arr }
    }
}

// ═════════════════════════════════════════════════════════════════════
// Batch encode / decode helpers
// ═════════════════════════════════════════════════════════════════════

/// Encode all rows into Slot L entries using a shared SVD basis.
///
/// Pipeline per row: residual = row - centroid → SVD::project(residual) →
/// quantize top-`SLOT_L_LANES` coefficients to i8 with shared per-tensor scale.
///
/// Returns (entries, shared scale). The scale is computed from the maximum
/// absolute projected coefficient across all rows — so no coefficient
/// saturates during i8 quantization.
pub fn encode_rows(
    rows_f32: &[Vec<f32>],
    centroids_per_row: &[Vec<f32>],
    basis: &SvdBasis,
) -> (Vec<SlotL>, f32) {
    debug_assert_eq!(rows_f32.len(), centroids_per_row.len());
    debug_assert!(basis.n_components >= SLOT_L_LANES,
        "SVD basis needs at least {} components for SlotL", SLOT_L_LANES);

    // First pass: project all residuals, track max abs coefficient.
    let n = rows_f32.len();
    let mut all_coeffs: Vec<[f32; SLOT_L_LANES]> = Vec::with_capacity(n);
    let mut max_abs: f32 = 0.0;
    for (row, centroid) in rows_f32.iter().zip(centroids_per_row.iter()) {
        let mut residual = vec![0.0f32; row.len()];
        for i in 0..row.len().min(centroid.len()) {
            residual[i] = row[i] - centroid[i];
        }
        let coeffs = basis.project(&residual);
        let mut arr = [0.0f32; SLOT_L_LANES];
        for i in 0..SLOT_L_LANES.min(coeffs.len()) {
            arr[i] = coeffs[i];
            let a = arr[i].abs();
            if a > max_abs { max_abs = a; }
        }
        all_coeffs.push(arr);
    }

    // Scale so max-abs maps to ±127.
    let scale = if max_abs > 1e-12 { max_abs / 127.0 } else { 1.0 };
    let inv_scale = 1.0 / scale;

    // Second pass: quantize.
    let entries: Vec<SlotL> = all_coeffs.iter().map(|coeffs| {
        let mut bytes = [0i8; SLOT_L_LANES];
        for i in 0..SLOT_L_LANES {
            let q = (coeffs[i] * inv_scale).round();
            bytes[i] = q.clamp(-127.0, 127.0) as i8;
        }
        SlotL { bytes }
    }).collect();

    (entries, scale)
}

/// Decode one row from (centroid_f32, SlotL, shared scale, SVD basis).
/// Returns the reconstructed row in f32.
pub fn decode_row(
    centroid_f32: &[f32],
    slot_l: &SlotL,
    scale: f32,
    basis: &SvdBasis,
    n_cols: usize,
) -> Vec<f32> {
    let mut coeffs = [0.0f32; SLOT_L_LANES];
    for (i, coeff) in coeffs.iter_mut().enumerate().take(SLOT_L_LANES) {
        *coeff = slot_l.bytes[i] as f32 * scale;
    }
    let residual = basis.reconstruct(&coeffs);
    let mut row = vec![0.0f32; n_cols];
    for i in 0..n_cols {
        let c = if i < centroid_f32.len() { centroid_f32[i] } else { 0.0 };
        let r = if i < residual.len() { residual[i] } else { 0.0 };
        row[i] = c + r;
    }
    row
}

// ═════════════════════════════════════════════════════════════════════
// Tests
// ═════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn slot_l_byte_size() {
        assert_eq!(SlotL::BYTE_SIZE, 8);
        assert_eq!(std::mem::size_of::<SlotL>(), 8);
    }

    #[test]
    fn slot_l_roundtrip() {
        let entry = SlotL { bytes: [-128i8, -1, 0, 1, 42, 100, 127, -50] };
        let b = entry.to_le_bytes();
        let back = SlotL::from_le_bytes(&b);
        assert_eq!(entry, back);
    }

    /// Generate deterministic pseudo-random rows with a low-rank structure
    /// so SVD can actually capture them. Otherwise 8 components of SVD on
    /// fully-random rows would give poor reconstruction regardless of codec.
    fn synthetic_rows(n: usize, cols: usize, seed: u32) -> Vec<Vec<f32>> {
        // Build via 8 "atoms" mixed per row — reconstructable by SVD at 8 components.
        let n_atoms = 8usize;
        let mut atoms = Vec::with_capacity(n_atoms);
        let mut s = seed;
        let mut next = || {
            s = s.wrapping_mul(1103515245).wrapping_add(12345);
            ((s >> 8) as i32 as f32) / 2_147_483_648.0
        };
        for _ in 0..n_atoms {
            let atom: Vec<f32> = (0..cols).map(|_| next()).collect();
            atoms.push(atom);
        }
        (0..n).map(|_| {
            let mut row = vec![0.0f32; cols];
            for atom in &atoms {
                let w = next() * 0.5;
                for j in 0..cols {
                    row[j] += atom[j] * w;
                }
            }
            row
        }).collect()
    }

    fn cosine(a: &[f32], b: &[f32]) -> f64 {
        let mut dot = 0.0f64;
        let mut na = 0.0f64;
        let mut nb = 0.0f64;
        let n = a.len().min(b.len());
        for i in 0..n {
            dot += a[i] as f64 * b[i] as f64;
            na += a[i] as f64 * a[i] as f64;
            nb += b[i] as f64 * b[i] as f64;
        }
        let d = (na * nb).sqrt();
        if d < 1e-15 { 0.0 } else { dot / d }
    }

    #[test]
    fn encode_decode_roundtrip_with_zero_centroid_and_low_rank_rows() {
        let n = 64;
        let cols = 256;
        let rows = synthetic_rows(n, cols, 0xABCD);

        // Zero centroids so residual == row (tests Slot L alone).
        let centroids = vec![vec![0.0f32; cols]; n];

        let basis = SvdBasis::build("synth", &rows, SLOT_L_LANES);

        let (entries, scale) = encode_rows(&rows, &centroids, &basis);
        assert_eq!(entries.len(), n);
        assert!(scale > 0.0, "scale should be positive, got {}", scale);

        // Per-row cos >= 0.98 (the design target for SlotL on index-regime tensors).
        let mut min_cos: f64 = 1.0;
        let mut sum_cos = 0.0f64;
        for i in 0..n {
            let recon = decode_row(&centroids[i], &entries[i], scale, &basis, cols);
            let c = cosine(&rows[i], &recon);
            if c < min_cos { min_cos = c; }
            sum_cos += c;
        }
        let avg_cos = sum_cos / n as f64;
        assert!(avg_cos >= 0.98,
            "avg cos should be >= 0.98 on low-rank synthetic with 8 atoms, got {:.4}", avg_cos);
        assert!(min_cos >= 0.95,
            "min cos should be >= 0.95, got {:.4}", min_cos);
    }

    #[test]
    fn zero_residual_when_centroid_equals_row() {
        let cols = 128;
        let row: Vec<f32> = (0..cols).map(|i| (i as f32 * 0.01).sin()).collect();

        // Centroid equals row -> residual is zero -> SlotL should be all zeros.
        let rows = vec![row.clone()];
        let centroids = vec![row.clone()];
        // Basis must still have at least SLOT_L_LANES components; any basis works
        // since we quantize zero to zero regardless.
        let sample: Vec<Vec<f32>> = (0..8).map(|_| row.clone()).collect();
        let basis = SvdBasis::build("trivial", &sample, SLOT_L_LANES);

        let (entries, _scale) = encode_rows(&rows, &centroids, &basis);
        for &b in &entries[0].bytes {
            assert_eq!(b, 0, "zero residual should quantize to zero, got {}", b);
        }
    }
}
