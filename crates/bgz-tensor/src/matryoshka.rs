//! Matryoshka codec: SVD-ordered variable bit allocation for weight rows.
//!
//! Like Opus allocates more bits to speech fundamentals and fewer to air,
//! this codec allocates more bits to SVD components that carry signal and
//! fewer to components that carry noise. All dimensions preserved.
//!
//! ```text
//! Opus audio bands:                Weight SVD bands:
//!   Band 0-3   (fundamentals): 6b    Components 0-63:    i16  (128B)
//!   Band 4-7   (formants):     5b    Components 64-191:  i8   (128B)
//!   Band 8-13  (consonants):   3b    Components 192-383: i4   (96B)
//!   Band 14-20 (air):          1b    Components 384-D:   i2   (var)
//!                                    ─────────────────────────────
//!                                    Total: ~512 bytes/row (4:1)
//! ```
//!
//! Error budget through 33 transformer layers:
//!   Per-layer ε < 0.001 → (0.999)^33 = 0.967 output fidelity
//!   Band 0 (i16): ε = 0.00003/element × sqrt(64) = 0.00024/row
//!   Band 3 (i2):  ε = 0.25/element BUT <5% signal energy → weighted ε < 0.001
//!
//! The SVD basis is shared across same-role layers (28 talker layers share
//! one basis). Storage: basis [D, cols] as BF16, computed once during bake.

use crate::stacked_n::{bf16_to_f32, f32_to_bf16};

// ═══════════════════════════════════════════════════════════════════
// Band configuration
// ═══════════════════════════════════════════════════════════════════

/// Quantization precision for one SVD band.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum BandPrecision {
    /// 16-bit signed integer. For critical components (attention routing).
    I16 = 16,
    /// 8-bit signed integer. For important components (head discrimination).
    I8 = 8,
    /// 4-bit signed integer (packed 2 per byte). For detail.
    I4 = 4,
    /// 2-bit signed integer (packed 4 per byte). For noise floor.
    I2 = 2,
}

impl BandPrecision {
    /// Bytes per element at this precision.
    pub fn bytes_per_element(self) -> f64 {
        match self {
            BandPrecision::I16 => 2.0,
            BandPrecision::I8 => 1.0,
            BandPrecision::I4 => 0.5,
            BandPrecision::I2 => 0.25,
        }
    }

    /// Max representable value (symmetric range [-max, +max]).
    pub fn max_val(self) -> i32 {
        match self {
            BandPrecision::I16 => 32767,
            BandPrecision::I8 => 127,
            BandPrecision::I4 => 7,
            BandPrecision::I2 => 1,
        }
    }
}

/// One band in the matryoshka: a range of SVD components at a fixed precision.
#[derive(Clone, Debug)]
pub struct Band {
    /// Start component index (inclusive).
    pub start: usize,
    /// End component index (exclusive).
    pub end: usize,
    /// Quantization precision for this band.
    pub precision: BandPrecision,
}

impl Band {
    pub fn n_components(&self) -> usize {
        self.end - self.start
    }

    /// Bytes needed for this band's quantized data per row.
    pub fn bytes_per_row(&self) -> usize {
        let bits = self.n_components() * self.precision as usize;
        (bits + 7) / 8
    }
}

/// Band allocation profile — how to split D SVD components into bands.
#[derive(Clone, Debug)]
pub struct BandProfile {
    pub bands: Vec<Band>,
    /// Total SVD components covered.
    pub total_components: usize,
    /// Original dimensionality (columns in weight matrix).
    pub original_dim: usize,
}

impl BandProfile {
    /// Standard 4-band profile for TTS weight matrices.
    ///
    /// Tuned for Qwen3-TTS where head_dim=128, hidden=1024/2048.
    /// Band boundaries at 64/192/384 cover the energy decay curve
    /// measured in the SVD session (70.9% in top 256).
    pub fn standard(n_components: usize, original_dim: usize) -> Self {
        let d = n_components;
        let bands = vec![
            Band { start: 0, end: 64.min(d), precision: BandPrecision::I16 },
            Band { start: 64.min(d), end: 192.min(d), precision: BandPrecision::I8 },
            Band { start: 192.min(d), end: 384.min(d), precision: BandPrecision::I4 },
            Band { start: 384.min(d), end: d, precision: BandPrecision::I2 },
        ];
        BandProfile { bands, total_components: d, original_dim }
    }

    /// Aggressive profile for high compression (8:1+).
    /// Narrows the i16 band, widens i2.
    pub fn aggressive(n_components: usize, original_dim: usize) -> Self {
        let d = n_components;
        let bands = vec![
            Band { start: 0, end: 32.min(d), precision: BandPrecision::I16 },
            Band { start: 32.min(d), end: 128.min(d), precision: BandPrecision::I8 },
            Band { start: 128.min(d), end: 256.min(d), precision: BandPrecision::I4 },
            Band { start: 256.min(d), end: d, precision: BandPrecision::I2 },
        ];
        BandProfile { bands, total_components: d, original_dim }
    }

    /// Conservative profile for quality-critical roles (2:1).
    /// Wide i16 band, no i2.
    pub fn conservative(n_components: usize, original_dim: usize) -> Self {
        let d = n_components;
        let bands = vec![
            Band { start: 0, end: 128.min(d), precision: BandPrecision::I16 },
            Band { start: 128.min(d), end: 384.min(d), precision: BandPrecision::I8 },
            Band { start: 384.min(d), end: d, precision: BandPrecision::I4 },
        ];
        BandProfile { bands, total_components: d, original_dim }
    }

    /// Bytes per row for the quantized coefficients (excluding basis).
    pub fn bytes_per_row(&self) -> usize {
        // gain (2 bytes BF16) + band data
        2 + self.bands.iter().map(|b| b.bytes_per_row()).sum::<usize>()
    }

    /// Compression ratio vs BF16 original.
    pub fn compression_ratio(&self) -> f64 {
        let original = self.original_dim * 2; // BF16
        original as f64 / self.bytes_per_row() as f64
    }
}

// ═══════════════════════════════════════════════════════════════════
// SVD Basis — shared across same-role layers
// ═══════════════════════════════════════════════════════════════════

/// SVD basis for one role group. Computed once, shared across all layers.
///
/// The basis vectors are the right singular vectors of the weight matrix,
/// ordered by singular value (descending). The first vector captures the
/// most variance, the last the least — that's the matryoshka ordering.
#[derive(Clone, Debug)]
pub struct SvdBasis {
    /// Role identifier (e.g., "talker_q_proj").
    pub role: String,
    /// Number of SVD components kept (= number of basis vectors).
    pub n_components: usize,
    /// Original column count of the weight matrix.
    pub original_cols: usize,
    /// Basis vectors as flat BF16: [n_components × original_cols].
    /// Row-major: basis[i * original_cols + j] = Vt[i][j].
    pub basis_bf16: Vec<u16>,
    /// Singular values (f32) for energy analysis. Length = n_components.
    pub singular_values: Vec<f32>,
}

impl SvdBasis {
    /// Byte size of the basis (BF16 storage).
    pub fn byte_size(&self) -> usize {
        self.n_components * self.original_cols * 2
    }

    /// Fraction of total energy captured by the first `k` components.
    pub fn energy_fraction(&self, k: usize) -> f64 {
        let total: f64 = self.singular_values.iter().map(|&s| (s as f64).powi(2)).sum();
        if total < 1e-30 { return 0.0; }
        let partial: f64 = self.singular_values[..k.min(self.n_components)].iter()
            .map(|&s| (s as f64).powi(2)).sum();
        partial / total
    }

    /// Get basis vector `i` as f32 slice (hydrated from BF16).
    pub fn vector_f32(&self, i: usize) -> Vec<f32> {
        let start = i * self.original_cols;
        let end = start + self.original_cols;
        self.basis_bf16[start..end].iter()
            .map(|&bits| bf16_to_f32(bits))
            .collect()
    }

    /// Project a row onto the basis: row[cols] → coefficients[n_components].
    pub fn project(&self, row: &[f32]) -> Vec<f32> {
        let mut coeffs = Vec::with_capacity(self.n_components);
        for i in 0..self.n_components {
            let start = i * self.original_cols;
            let mut dot = 0.0f64;
            for j in 0..self.original_cols.min(row.len()) {
                dot += row[j] as f64 * bf16_to_f32(self.basis_bf16[start + j]) as f64;
            }
            coeffs.push(dot as f32);
        }
        coeffs
    }

    /// Reconstruct a row from coefficients: coefficients[n_components] → row[cols].
    pub fn reconstruct(&self, coeffs: &[f32]) -> Vec<f32> {
        let mut row = vec![0.0f32; self.original_cols];
        for i in 0..self.n_components.min(coeffs.len()) {
            let start = i * self.original_cols;
            let c = coeffs[i];
            for j in 0..self.original_cols {
                row[j] += c * bf16_to_f32(self.basis_bf16[start + j]);
            }
        }
        row
    }

    /// Build SVD basis from sampled weight rows.
    ///
    /// Uses power iteration (no LAPACK dependency) for the top `n_components`
    /// singular vectors. Sufficient for our use case where we need at most
    /// d=512 components from matrices up to 6144×2048.
    ///
    /// `sample_rows`: representative f32 rows (e.g., 4096 sampled from all layers).
    /// `n_components`: how many SVD components to keep.
    pub fn build(role: &str, sample_rows: &[Vec<f32>], n_components: usize) -> Self {
        let n = sample_rows.len();
        if n == 0 {
            return SvdBasis {
                role: role.to_string(),
                n_components: 0,
                original_cols: 0,
                basis_bf16: Vec::new(),
                singular_values: Vec::new(),
            };
        }
        let cols = sample_rows[0].len();
        let d = n_components.min(n).min(cols);

        // Power iteration for top-d right singular vectors.
        // For each component: iterate v ← A^T A v / ||A^T A v||
        let mut basis_f32 = Vec::with_capacity(d * cols);
        let mut singular_values = Vec::with_capacity(d);

        // Work with the matrix as-is (rows stored as Vec<Vec<f32>>)
        // Deflate after each component
        let mut deflated: Vec<Vec<f32>> = sample_rows.to_vec();

        for comp in 0..d {
            // Initialize v randomly (deterministic seed)
            let mut v = vec![0.0f64; cols];
            for j in 0..cols {
                // Simple hash-based initialization
                let seed = (comp * 7919 + j * 104729 + 31) as f64;
                v[j] = (seed.sin() * 43758.5453).fract() - 0.5;
            }
            normalize_f64(&mut v);

            // Power iteration: 20 iterations is enough for convergence
            for _iter in 0..20 {
                // u = A @ v
                let mut u = vec![0.0f64; n];
                for (i, row) in deflated.iter().enumerate() {
                    let mut dot = 0.0f64;
                    for j in 0..cols {
                        dot += row[j] as f64 * v[j];
                    }
                    u[i] = dot;
                }

                // v = A^T @ u
                let mut v_new = vec![0.0f64; cols];
                for (i, row) in deflated.iter().enumerate() {
                    let ui = u[i];
                    for j in 0..cols {
                        v_new[j] += row[j] as f64 * ui;
                    }
                }

                let sigma = norm_f64(&v_new);
                if sigma < 1e-15 { break; }

                for j in 0..cols {
                    v[j] = v_new[j] / sigma;
                }

                // Only store sigma on last iteration
                if _iter == 19 || sigma < 1e-10 {
                    singular_values.push(sigma.sqrt() as f32);
                }
            }

            // Store basis vector
            for j in 0..cols {
                basis_f32.push(v[j] as f32);
            }

            // Deflate: remove this component from the matrix
            for row in &mut deflated {
                let mut dot = 0.0f64;
                for j in 0..cols {
                    dot += row[j] as f64 * v[j];
                }
                for j in 0..cols {
                    row[j] -= (dot * v[j]) as f32;
                }
            }
        }

        // Pad singular_values if power iteration didn't fill all
        while singular_values.len() < d {
            singular_values.push(0.0);
        }

        // Convert basis to BF16
        let basis_bf16: Vec<u16> = basis_f32.iter().map(|&v| f32_to_bf16(v)).collect();

        SvdBasis {
            role: role.to_string(),
            n_components: d,
            original_cols: cols,
            basis_bf16,
            singular_values,
        }
    }
}

fn normalize_f64(v: &mut [f64]) {
    let norm = norm_f64(v);
    if norm > 1e-15 {
        for x in v.iter_mut() { *x /= norm; }
    }
}

fn norm_f64(v: &[f64]) -> f64 {
    v.iter().map(|x| x * x).sum::<f64>().sqrt()
}

// ═══════════════════════════════════════════════════════════════════
// Matryoshka encoded row
// ═══════════════════════════════════════════════════════════════════

/// One weight row encoded with the matryoshka codec.
///
/// Gain (BF16) + variable-precision SVD coefficients packed into bytes.
/// Total size determined by BandProfile.
#[derive(Clone, Debug)]
pub struct MatryoshkaRow {
    /// Per-row gain (L2 norm of original row). BF16.
    pub gain_bf16: u16,
    /// Packed coefficient data. Layout determined by BandProfile.
    pub data: Vec<u8>,
}

impl MatryoshkaRow {
    pub fn byte_size(&self) -> usize {
        2 + self.data.len()
    }
}

// ═══════════════════════════════════════════════════════════════════
// Encode / Decode
// ═══════════════════════════════════════════════════════════════════

/// Encode a single row: project onto SVD basis, quantize per band.
pub fn encode_row(row: &[f32], basis: &SvdBasis, profile: &BandProfile) -> MatryoshkaRow {
    // Compute gain (L2 norm)
    let gain = row.iter().map(|&x| (x as f64).powi(2)).sum::<f64>().sqrt() as f32;
    let gain_bf16 = f32_to_bf16(gain);
    let inv_gain = if gain > 1e-15 { 1.0 / gain } else { 0.0 };

    // Project normalized row onto basis
    let normalized: Vec<f32> = row.iter().map(|&x| x * inv_gain).collect();
    let coeffs = basis.project(&normalized);

    // Quantize per band
    let mut data = Vec::with_capacity(profile.bytes_per_row() - 2);

    for band in &profile.bands {
        let max_val = band.precision.max_val();
        let band_coeffs = &coeffs[band.start..band.end.min(coeffs.len())];

        // Find scale for this band
        let band_max = band_coeffs.iter()
            .map(|&c| c.abs())
            .fold(0.0f32, f32::max)
            .max(1e-15);
        let scale = max_val as f32 / band_max;

        match band.precision {
            BandPrecision::I16 => {
                // Scale as BF16 header for this band
                data.extend_from_slice(&f32_to_bf16(band_max).to_le_bytes());
                for &c in band_coeffs {
                    let q = (c * scale).round().clamp(-(max_val as f32), max_val as f32) as i16;
                    data.extend_from_slice(&q.to_le_bytes());
                }
            }
            BandPrecision::I8 => {
                data.extend_from_slice(&f32_to_bf16(band_max).to_le_bytes());
                for &c in band_coeffs {
                    let q = (c * scale).round().clamp(-127.0, 127.0) as i8;
                    data.push(q as u8);
                }
            }
            BandPrecision::I4 => {
                data.extend_from_slice(&f32_to_bf16(band_max).to_le_bytes());
                // Pack 2 nibbles per byte
                let mut i = 0;
                while i < band_coeffs.len() {
                    let a = (band_coeffs[i] * scale).round().clamp(-7.0, 7.0) as i8;
                    let b = if i + 1 < band_coeffs.len() {
                        (band_coeffs[i + 1] * scale).round().clamp(-7.0, 7.0) as i8
                    } else { 0 };
                    // Pack: low nibble = a+8, high nibble = b+8 (unsigned 0-15)
                    let byte = ((a + 8) as u8) | (((b + 8) as u8) << 4);
                    data.push(byte);
                    i += 2;
                }
            }
            BandPrecision::I2 => {
                data.extend_from_slice(&f32_to_bf16(band_max).to_le_bytes());
                // Pack 4 crumbs per byte
                let mut i = 0;
                while i < band_coeffs.len() {
                    let mut byte = 0u8;
                    for bit in 0..4 {
                        if i + bit < band_coeffs.len() {
                            let q = (band_coeffs[i + bit] * scale).round().clamp(-1.0, 1.0) as i8;
                            let u = (q + 1) as u8; // 0, 1, 2
                            byte |= (u & 0x03) << (bit * 2);
                        }
                    }
                    data.push(byte);
                    i += 4;
                }
            }
        }
    }

    MatryoshkaRow { gain_bf16, data }
}

/// Decode a single row: dequantize per band, reconstruct via basis.
pub fn decode_row(encoded: &MatryoshkaRow, basis: &SvdBasis, profile: &BandProfile) -> Vec<f32> {
    let gain = bf16_to_f32(encoded.gain_bf16);
    let mut coeffs = vec![0.0f32; basis.n_components];
    let mut offset = 0usize;

    for band in &profile.bands {
        let max_val = band.precision.max_val();

        // Read band scale
        if offset + 2 > encoded.data.len() { break; }
        let band_max = bf16_to_f32(u16::from_le_bytes([
            encoded.data[offset], encoded.data[offset + 1]
        ]));
        offset += 2;
        let inv_scale = band_max / max_val as f32;

        let n = band.n_components();

        match band.precision {
            BandPrecision::I16 => {
                for i in 0..n {
                    if offset + 2 > encoded.data.len() { break; }
                    let q = i16::from_le_bytes([
                        encoded.data[offset], encoded.data[offset + 1]
                    ]);
                    coeffs[band.start + i] = q as f32 * inv_scale;
                    offset += 2;
                }
            }
            BandPrecision::I8 => {
                for i in 0..n {
                    if offset >= encoded.data.len() { break; }
                    let q = encoded.data[offset] as i8;
                    coeffs[band.start + i] = q as f32 * inv_scale;
                    offset += 1;
                }
            }
            BandPrecision::I4 => {
                let mut i = 0;
                while i < n {
                    if offset >= encoded.data.len() { break; }
                    let byte = encoded.data[offset];
                    let a = (byte & 0x0F) as i8 - 8;
                    coeffs[band.start + i] = a as f32 * inv_scale;
                    if i + 1 < n {
                        let b = (byte >> 4) as i8 - 8;
                        coeffs[band.start + i + 1] = b as f32 * inv_scale;
                    }
                    offset += 1;
                    i += 2;
                }
            }
            BandPrecision::I2 => {
                let mut i = 0;
                while i < n {
                    if offset >= encoded.data.len() { break; }
                    let byte = encoded.data[offset];
                    for bit in 0..4 {
                        if i + bit < n {
                            let u = (byte >> (bit * 2)) & 0x03;
                            let q = u as i8 - 1; // -1, 0, 1
                            coeffs[band.start + i + bit] = q as f32 * inv_scale;
                        }
                    }
                    offset += 1;
                    i += 4;
                }
            }
        }
    }

    // Reconstruct: gain × (coeffs @ basis)
    let mut row = basis.reconstruct(&coeffs);
    for x in &mut row {
        *x *= gain;
    }
    row
}

// ═══════════════════════════════════════════════════════════════════
// Batch encode/decode
// ═══════════════════════════════════════════════════════════════════

/// Encode a full weight matrix.
pub fn encode_matrix(
    rows: &[Vec<f32>],
    basis: &SvdBasis,
    profile: &BandProfile,
) -> Vec<MatryoshkaRow> {
    rows.iter().map(|row| encode_row(row, basis, profile)).collect()
}

/// Decode a full weight matrix.
pub fn decode_matrix(
    encoded: &[MatryoshkaRow],
    basis: &SvdBasis,
    profile: &BandProfile,
) -> Vec<Vec<f32>> {
    encoded.iter().map(|row| decode_row(row, basis, profile)).collect()
}

/// Measure reconstruction quality: per-row cosine and pairwise rank.
pub fn measure_quality(
    original: &[Vec<f32>],
    reconstructed: &[Vec<f32>],
) -> (f64, f64) {
    let n = original.len().min(reconstructed.len());
    if n == 0 { return (0.0, 0.0); }

    // Per-row cosine (average)
    let mut cos_sum = 0.0f64;
    for i in 0..n {
        cos_sum += cosine_f32(&original[i], &reconstructed[i]);
    }
    let avg_row_cos = cos_sum / n as f64;

    // Pairwise rank preservation (Spearman on 100 random pairs)
    let n_pairs = 100.min(n * (n - 1) / 2);
    let mut gt = Vec::with_capacity(n_pairs);
    let mut rc = Vec::with_capacity(n_pairs);

    let mut seed = 0x9E3779B97F4A7C15u64;
    for _ in 0..n_pairs {
        seed = seed.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = seed;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z ^= z >> 31;
        let a = (z as usize) % n;

        seed = seed.wrapping_add(0x9E3779B97F4A7C15);
        z = seed;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z ^= z >> 31;
        let b = (z as usize) % n;

        if a == b { continue; }
        gt.push(cosine_f32(&original[a], &original[b]));
        rc.push(cosine_f32(&reconstructed[a], &reconstructed[b]));
    }

    let pairwise_rho = crate::quality::spearman(&gt, &rc);

    (avg_row_cos, pairwise_rho)
}

fn cosine_f32(a: &[f32], b: &[f32]) -> f64 {
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
    let denom = (na * nb).sqrt();
    if denom < 1e-15 { 0.0 } else { dot / denom }
}

// ═══════════════════════════════════════════════════════════════════
// Serialization (for safetensors output)
// ═══════════════════════════════════════════════════════════════════

impl SvdBasis {
    /// Serialize basis to flat bytes (BF16).
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut buf = Vec::with_capacity(self.byte_size() + 16);
        // Header: n_components (u32) + original_cols (u32)
        buf.extend_from_slice(&(self.n_components as u32).to_le_bytes());
        buf.extend_from_slice(&(self.original_cols as u32).to_le_bytes());
        // Singular values (f32)
        for &sv in &self.singular_values {
            buf.extend_from_slice(&sv.to_le_bytes());
        }
        // Basis vectors (BF16)
        for &bits in &self.basis_bf16 {
            buf.extend_from_slice(&bits.to_le_bytes());
        }
        buf
    }

    /// Deserialize from bytes.
    pub fn from_bytes(role: &str, data: &[u8]) -> Option<Self> {
        if data.len() < 8 { return None; }
        let n_components = u32::from_le_bytes([data[0], data[1], data[2], data[3]]) as usize;
        let original_cols = u32::from_le_bytes([data[4], data[5], data[6], data[7]]) as usize;

        let sv_start = 8;
        let sv_end = sv_start + n_components * 4;
        if data.len() < sv_end { return None; }
        let singular_values: Vec<f32> = (0..n_components).map(|i| {
            let off = sv_start + i * 4;
            f32::from_le_bytes([data[off], data[off+1], data[off+2], data[off+3]])
        }).collect();

        let basis_start = sv_end;
        let basis_len = n_components * original_cols;
        if data.len() < basis_start + basis_len * 2 { return None; }
        let basis_bf16: Vec<u16> = (0..basis_len).map(|i| {
            let off = basis_start + i * 2;
            u16::from_le_bytes([data[off], data[off+1]])
        }).collect();

        Some(SvdBasis {
            role: role.to_string(),
            n_components,
            original_cols,
            basis_bf16,
            singular_values,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_row(seed: usize, dim: usize) -> Vec<f32> {
        (0..dim).map(|d| {
            let x = ((d * 97 + seed * 31 + 17) as f64 * 0.618).sin() as f32;
            x * 0.01
        }).collect()
    }

    #[test]
    fn band_profile_sizes() {
        let p = BandProfile::standard(512, 2048);
        let bpr = p.bytes_per_row();
        // gain(2) + band0: scale(2)+64×2 + band1: scale(2)+128 + band2: scale(2)+96 + band3: scale(2)+32
        assert!(bpr > 350 && bpr < 600, "bytes_per_row = {} not in [350,600]", bpr);
        assert!(p.compression_ratio() > 3.0, "ratio should be >3:1");
    }

    #[test]
    fn encode_decode_roundtrip_nonzero() {
        let rows: Vec<Vec<f32>> = (0..50).map(|i| make_row(i, 256)).collect();
        let basis = SvdBasis::build("test", &rows, 128);
        let profile = BandProfile::standard(128, 256);

        let encoded = encode_matrix(&rows, &basis, &profile);
        let decoded = decode_matrix(&encoded, &basis, &profile);

        assert_eq!(decoded.len(), rows.len());
        assert_eq!(decoded[0].len(), 256);

        // Should be nonzero
        let mag: f64 = decoded[0].iter().map(|x| x.abs() as f64).sum();
        assert!(mag > 0.0);
    }

    #[test]
    fn roundtrip_quality_reasonable() {
        let rows: Vec<Vec<f32>> = (0..100).map(|i| make_row(i, 512)).collect();
        let basis = SvdBasis::build("test", &rows, 256);
        let profile = BandProfile::standard(256, 512);

        let encoded = encode_matrix(&rows, &basis, &profile);
        let decoded = decode_matrix(&encoded, &basis, &profile);

        let (avg_cos, pairwise_rho) = measure_quality(&rows, &decoded);
        assert!(avg_cos > 0.8, "row cosine {} should be >0.8", avg_cos);
        assert!(pairwise_rho > 0.7, "pairwise ρ {} should be >0.7", pairwise_rho);
    }

    #[test]
    fn basis_serialization_roundtrip() {
        let rows: Vec<Vec<f32>> = (0..30).map(|i| make_row(i, 128)).collect();
        let basis = SvdBasis::build("test_role", &rows, 64);

        let bytes = basis.to_bytes();
        let restored = SvdBasis::from_bytes("test_role", &bytes).unwrap();

        assert_eq!(restored.n_components, basis.n_components);
        assert_eq!(restored.original_cols, basis.original_cols);
        assert_eq!(restored.basis_bf16.len(), basis.basis_bf16.len());
    }

    #[test]
    fn aggressive_profile_smaller() {
        let std_p = BandProfile::standard(512, 2048);
        let agg_p = BandProfile::aggressive(512, 2048);
        assert!(agg_p.bytes_per_row() < std_p.bytes_per_row());
        assert!(agg_p.compression_ratio() > std_p.compression_ratio());
    }

    #[test]
    fn conservative_profile_larger() {
        let std_p = BandProfile::standard(512, 2048);
        let con_p = BandProfile::conservative(512, 2048);
        assert!(con_p.bytes_per_row() > std_p.bytes_per_row());
    }
}
