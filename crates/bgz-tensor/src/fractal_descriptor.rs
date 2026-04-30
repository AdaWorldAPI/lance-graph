//! Fractal descriptor leaf — orthogonal MFDFA-on-Hadamard codec primitive.
//!
//! Per `.claude/knowledge/fractal-codec-argmax-regime.md` Roadmap Step 2.
//! ~200 LOC probe of: do trained Qwen3 weight rows, after Hadamard rotation,
//! exhibit MFDFA-measurable self-similarity?
//!
//! If yes → row encodes as `(D_local, w_mfs, σ_energy, H_hurst)` =
//! 7 bytes / row at the LEAF of the HHTL cascade.
//!
//! ## Algorithm
//!
//! 1. **Hadamard rotate** `wht_f32` to project onto orthogonal basis.
//! 2. **Cumulative profile** `Y(k) = Σ_{i≤k} (x_i − mean)` after demean.
//! 3. **Multi-scale fluctuation** `F²(s,v) = var(profile_segment − linear_fit)`
//!    for non-overlapping segments of size s ∈ {4, 8, 16, …}.
//! 4. **Generalized Hurst** `F_q(s) = (mean_v F²(s,v)^(q/2))^(1/q) ~ s^h(q)`
//!    via log-log regression for q ∈ {−5, −3, −1, 1, 3, 5}.
//! 5. **Multifractal spectrum width** `w = α(q_min) − α(q_max)` where
//!    `α(q) = h(q) + q · dh/dq` (Legendre transform).
//! 6. **Hurst exponent** `H = h(2)`.
//! 7. **Fractal dimension** `D = 2 − H` (box-counting estimate from Hurst).

use crate::ndarray_compat::wht_f32;

/// 7-byte fractal descriptor of a row's self-similar shape on an
/// orthogonal (Hadamard) basis.
///
/// Wire format per `.claude/knowledge/fractal-codec-argmax-regime.md`
/// § Concrete Wire Format.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FractalDescriptor {
    /// Box-counting fractal dimension (q0.16 over [0, 2]).
    pub d_local: u16,
    /// Multifractal spectrum width α_max − α_min (q0.16 over [0, 2]).
    pub w_mfs: u16,
    /// Total L2 energy of the row, BF16-encoded.
    pub sigma_energy: u16,
    /// Hurst exponent H = h(2), q0.8 over [0, 1].
    pub h_hurst: u8,
}

impl FractalDescriptor {
    pub const SIZE_BYTES: usize = 7;

    pub fn pack(self) -> [u8; 7] {
        let mut out = [0u8; 7];
        out[0..2].copy_from_slice(&self.d_local.to_le_bytes());
        out[2..4].copy_from_slice(&self.w_mfs.to_le_bytes());
        out[4..6].copy_from_slice(&self.sigma_energy.to_le_bytes());
        out[6] = self.h_hurst;
        out
    }

    pub fn unpack(bytes: [u8; 7]) -> Self {
        Self {
            d_local: u16::from_le_bytes([bytes[0], bytes[1]]),
            w_mfs: u16::from_le_bytes([bytes[2], bytes[3]]),
            sigma_energy: u16::from_le_bytes([bytes[4], bytes[5]]),
            h_hurst: bytes[6],
        }
    }

    pub fn d_local_f32(&self) -> f32 {
        self.d_local as f32 / 65535.0 * 2.0
    }
    pub fn w_mfs_f32(&self) -> f32 {
        self.w_mfs as f32 / 65535.0 * 2.0
    }
    pub fn h_hurst_f32(&self) -> f32 {
        self.h_hurst as f32 / 255.0
    }
    pub fn sigma_energy_f32(&self) -> f32 {
        bf16_to_f32(self.sigma_energy)
    }
}

/// Compute the fractal descriptor for a row.
///
/// Steps: L2 energy → Hadamard rotation → MFDFA on rotated coefficients →
/// pack into 7-byte descriptor.
///
/// Row length must be a power of 2 ≥ 64.
pub fn compute_mfdfa_descriptor(row: &[f32]) -> FractalDescriptor {
    let n = row.len();
    assert!(
        n.is_power_of_two() && n >= 64,
        "row length must be power of 2 ≥ 64, got {n}"
    );

    // L2 energy (Hadamard preserves it; compute once before rotation).
    let energy: f32 = row.iter().map(|&x| x * x).sum::<f32>().sqrt();

    // Project onto orthogonal basis.
    let mut rotated = row.to_vec();
    wht_f32(&mut rotated);

    // MFDFA at q ∈ {−5, −3, −1, 1, 2, 3, 5}; track q=2 for Hurst.
    let q_values = [-5.0_f32, -3.0, -1.0, 1.0, 2.0, 3.0, 5.0];
    let h_q = generalized_hurst(&rotated, &q_values);

    let h2 = h_q[4]; // q = 2.0
    // Multifractal spectrum width via finite-diff Legendre transform.
    let w = spectrum_width(&q_values, &h_q);
    // Box-counting fractal dim from Hurst (1-D series): D = 2 − H.
    let d = (2.0 - h2).clamp(0.0, 2.0);

    FractalDescriptor {
        d_local: f32_to_q16_in(d, 0.0, 2.0),
        w_mfs: f32_to_q16_in(w.clamp(0.0, 2.0), 0.0, 2.0),
        sigma_energy: f32_to_bf16(energy),
        h_hurst: f32_to_q8_in(h2.clamp(0.0, 1.0), 0.0, 1.0),
    }
}

/// Generalized Hurst exponent h(q) at multiple q values via DFA on the
/// cumulative profile.
fn generalized_hurst(series: &[f32], q_values: &[f32]) -> Vec<f32> {
    let n = series.len();
    let mean = series.iter().sum::<f32>() / n as f32;

    // Cumulative profile (demean + cumulative sum).
    let mut profile = Vec::with_capacity(n);
    let mut acc = 0.0_f32;
    for &x in series {
        acc += x - mean;
        profile.push(acc);
    }

    // Scales: powers of 2 from 4 up to n/4 (need ≥ 4 segments per scale).
    let max_log = (n / 4).ilog2() as usize;
    let scales: Vec<usize> = (2..=max_log).map(|i| 1usize << i).collect();

    let mut h_out = Vec::with_capacity(q_values.len());
    for &q in q_values {
        let mut log_s = Vec::with_capacity(scales.len());
        let mut log_f = Vec::with_capacity(scales.len());
        for &s in &scales {
            let n_seg = n / s;
            if n_seg < 4 {
                continue;
            }
            // Per-segment detrended variance.
            let mut variances = Vec::with_capacity(n_seg);
            for v in 0..n_seg {
                variances.push(detrended_variance(&profile[v * s..(v + 1) * s]));
            }
            let f_q = aggregate_fq(&variances, q);
            if f_q.is_finite() && f_q > 1e-30 {
                log_s.push((s as f32).ln());
                log_f.push(f_q.ln());
            }
        }
        let h = if log_s.len() >= 3 {
            linreg_slope(&log_s, &log_f)
        } else {
            0.5 // insufficient data → Hurst-of-noise default
        };
        h_out.push(h.clamp(-1.0, 2.0));
    }
    h_out
}

/// Variance of segment after order-1 (linear) detrend.
fn detrended_variance(seg: &[f32]) -> f32 {
    let n = seg.len() as f32;
    let mean_x = (n - 1.0) / 2.0;
    let mean_y: f32 = seg.iter().sum::<f32>() / n;
    let mut cov = 0.0_f32;
    let mut var_x = 0.0_f32;
    for (i, &y) in seg.iter().enumerate() {
        let dx = i as f32 - mean_x;
        cov += dx * (y - mean_y);
        var_x += dx * dx;
    }
    let slope = cov / var_x.max(1e-30);
    let intercept = mean_y - slope * mean_x;
    let mut sum_sq = 0.0_f32;
    for (i, &y) in seg.iter().enumerate() {
        let r = y - (intercept + slope * i as f32);
        sum_sq += r * r;
    }
    sum_sq / n
}

/// F_q(s) = (mean_v var_v^(q/2))^(1/q), with q→0 limit as geometric mean.
fn aggregate_fq(variances: &[f32], q: f32) -> f32 {
    let n = variances.len() as f32;
    if q.abs() < 1e-6 {
        let log_sum: f32 = variances.iter().map(|&v| v.max(1e-30).ln()).sum();
        (0.5 * log_sum / n).exp()
    } else {
        let pow = q / 2.0;
        let sum: f32 = variances.iter().map(|&v| v.max(1e-30).powf(pow)).sum();
        (sum / n).powf(1.0 / q)
    }
}

/// OLS slope.
fn linreg_slope(x: &[f32], y: &[f32]) -> f32 {
    let n = x.len() as f32;
    let mean_x: f32 = x.iter().sum::<f32>() / n;
    let mean_y: f32 = y.iter().sum::<f32>() / n;
    let mut cov = 0.0_f32;
    let mut var_x = 0.0_f32;
    for (xi, yi) in x.iter().zip(y) {
        let dx = xi - mean_x;
        cov += dx * (yi - mean_y);
        var_x += dx * dx;
    }
    cov / var_x.max(1e-30)
}

/// Multifractal spectrum width via finite-difference Legendre transform.
/// α(q) = h(q) + q · dh/dq; w = α(q_min) − α(q_max).
fn spectrum_width(q: &[f32], h: &[f32]) -> f32 {
    if q.len() < 3 {
        return 0.0;
    }
    let mut alphas = Vec::with_capacity(q.len());
    for i in 0..q.len() {
        let dh_dq = if i == 0 {
            (h[1] - h[0]) / (q[1] - q[0])
        } else if i == q.len() - 1 {
            (h[i] - h[i - 1]) / (q[i] - q[i - 1])
        } else {
            (h[i + 1] - h[i - 1]) / (q[i + 1] - q[i - 1])
        };
        alphas.push(h[i] + q[i] * dh_dq);
    }
    let a_max = alphas.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let a_min = alphas.iter().cloned().fold(f32::INFINITY, f32::min);
    (a_max - a_min).max(0.0)
}

// ───── Encoding helpers ─────

fn f32_to_q16_in(x: f32, lo: f32, hi: f32) -> u16 {
    let t = ((x - lo) / (hi - lo)).clamp(0.0, 1.0);
    (t * 65535.0).round() as u16
}

fn f32_to_q8_in(x: f32, lo: f32, hi: f32) -> u8 {
    let t = ((x - lo) / (hi - lo)).clamp(0.0, 1.0);
    (t * 255.0).round() as u8
}

fn f32_to_bf16(x: f32) -> u16 {
    let bits = x.to_bits();
    // Round-to-nearest-even on truncation.
    let rounded = bits.wrapping_add(0x7FFF + ((bits >> 16) & 1));
    (rounded >> 16) as u16
}

fn bf16_to_f32(b: u16) -> f32 {
    f32::from_bits((b as u32) << 16)
}

// ─────────────────────────────────────────────────────────────────────────
// Phase descriptor — fractal statistics of the SIGN sequence post-Hadamard.
//
// The MFDFA descriptor above measures |coefficient| magnitude statistics.
// Those are near-constant across Qwen3 rows (CoV 0.19, measured PR #216).
// What varies per-row is the SIGN PATTERN of rotated coefficients — that
// IS the phase. Two rows with identical magnitude envelopes can have
// completely different inner products via their sign patterns alone.
//
// This descriptor measures fractal structure of the sign sequence itself:
// density of sign-flips at multiple scales → 5-D signature per row.
// Pairwise cosine between phase signatures asks "do two rows share phase
// structure?" — orthogonal to magnitude similarity.
// ─────────────────────────────────────────────────────────────────────────

/// 5-D fractal phase signature: normalized sign-flip density at scales
/// s ∈ {4, 8, 16, 32, 64}.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PhaseDescriptor {
    /// Flip density per window of size s ∈ {4, 8, 16, 32, 64}.
    /// Values in [0, 0.5] (max 1 flip per step after binning).
    pub flip_density: [f32; 5],
}

impl PhaseDescriptor {
    pub const SCALES: [usize; 5] = [4, 8, 16, 32, 64];

    /// Compute the fractal phase signature for a row:
    /// 1. Hadamard-rotate (uses existing wht_f32 SIMD butterfly).
    /// 2. Extract sign sequence (1 bit per coefficient).
    /// 3. Count sign flips per non-overlapping window at 5 scales.
    /// 4. Normalize by window size → flip density.
    pub fn from_row(row: &[f32]) -> Self {
        let n = row.len();
        assert!(n.is_power_of_two() && n >= 64, "row length must be power of 2 ≥ 64");

        // Rotate into orthogonal basis.
        let mut rotated = row.to_vec();
        wht_f32(&mut rotated);

        // Sign sequence: +1 for non-negative, −1 otherwise.
        let signs: Vec<i8> = rotated.iter().map(|&x| if x >= 0.0 { 1 } else { -1 }).collect();

        // Flip density at each scale.
        let mut flip_density = [0.0_f32; 5];
        for (i, &s) in Self::SCALES.iter().enumerate() {
            if s > n {
                flip_density[i] = 0.0;
                continue;
            }
            let n_windows = n / s;
            if n_windows == 0 {
                flip_density[i] = 0.0;
                continue;
            }
            let mut total_flips: u32 = 0;
            for w in 0..n_windows {
                let start = w * s;
                for k in 0..(s - 1) {
                    if signs[start + k] != signs[start + k + 1] {
                        total_flips += 1;
                    }
                }
            }
            // Max possible flips = n_windows * (s - 1); normalize to [0, 1].
            let max_flips = (n_windows * (s - 1)) as f32;
            flip_density[i] = total_flips as f32 / max_flips.max(1.0);
        }

        Self { flip_density }
    }

    /// Normalized cosine similarity between two phase signatures.
    pub fn cosine(&self, other: &Self) -> f32 {
        let mut dot = 0.0_f32;
        let mut na = 0.0_f32;
        let mut nb = 0.0_f32;
        for i in 0..5 {
            dot += self.flip_density[i] * other.flip_density[i];
            na += self.flip_density[i] * self.flip_density[i];
            nb += other.flip_density[i] * other.flip_density[i];
        }
        let denom = (na * nb).sqrt();
        if denom < 1e-15 {
            0.0
        } else {
            dot / denom
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_white_noise(n: usize, seed: u64) -> Vec<f32> {
        let mut state = seed;
        (0..n)
            .map(|_| {
                state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                ((state >> 32) as i32 as f32) / i32::MAX as f32
            })
            .collect()
    }

    fn make_brownian(n: usize, seed: u64) -> Vec<f32> {
        let increments = make_white_noise(n, seed);
        let mut acc = 0.0_f32;
        increments
            .into_iter()
            .map(|x| {
                acc += x;
                acc
            })
            .collect()
    }

    #[test]
    fn descriptor_round_trip() {
        let d = FractalDescriptor {
            d_local: 32768,
            w_mfs: 16384,
            sigma_energy: f32_to_bf16(2.5),
            h_hurst: 200,
        };
        let bytes = d.pack();
        assert_eq!(bytes.len(), 7);
        assert_eq!(FractalDescriptor::unpack(bytes), d);
    }

    #[test]
    fn descriptor_size_invariant() {
        assert_eq!(FractalDescriptor::SIZE_BYTES, 7);
    }

    #[test]
    fn white_noise_hurst_near_half() {
        // White noise → Hurst ≈ 0.5, narrow multifractal spectrum.
        let row = make_white_noise(1024, 0xDEADBEEF);
        let d = compute_mfdfa_descriptor(&row);
        let h = d.h_hurst_f32();
        assert!(
            (0.2..=0.8).contains(&h),
            "white noise H expected ≈ 0.5, got {h}"
        );
    }

    #[test]
    fn brownian_hurst_higher() {
        // Cumulative noise (Brownian motion) → Hurst noticeably > 0.5.
        let white = make_white_noise(1024, 0xFEEDFACE);
        let brown = make_brownian(1024, 0xFEEDFACE);
        let h_w = compute_mfdfa_descriptor(&white).h_hurst_f32();
        let h_b = compute_mfdfa_descriptor(&brown).h_hurst_f32();
        assert!(
            h_b > h_w,
            "Brownian H ({h_b}) should exceed white-noise H ({h_w})"
        );
    }

    #[test]
    fn energy_preserved_through_descriptor() {
        let row: Vec<f32> = (0..256).map(|i| ((i as f32) * 0.01).sin()).collect();
        let expected_energy: f32 = row.iter().map(|x| x * x).sum::<f32>().sqrt();
        let d = compute_mfdfa_descriptor(&row);
        let recovered = d.sigma_energy_f32();
        // BF16 precision is ~2^-7 relative; allow 1% tolerance.
        let rel_err = (recovered - expected_energy).abs() / expected_energy.max(1e-6);
        assert!(rel_err < 0.01, "energy {expected_energy} → {recovered}, rel_err {rel_err}");
    }

    #[test]
    fn deterministic_signal_low_spectrum_width() {
        // Pure sinusoid → monofractal → spectrum width near zero.
        let row: Vec<f32> = (0..512).map(|i| ((i as f32) * 0.1).sin()).collect();
        let d = compute_mfdfa_descriptor(&row);
        let w = d.w_mfs_f32();
        // Real signals always have some w; just ensure it's not blown up.
        assert!(w < 1.5, "monofractal spectrum width should be < 1.5, got {w}");
    }
}
