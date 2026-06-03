//! Optional ndarray-accelerated hot path.
//!
//! The transcendental in the residue pipeline is the Fisher-Z `arctanh` (an `ln`
//! pair); that is where SIMD pays. With `--features ndarray-hpc`,
//! [`batch_fisher_z`] runs the `ln` pair through `ndarray::simd::simd_ln_f32`
//! (16-wide `F32x16` lanes); without it, an **identical-result** scalar path is
//! used. Endpoint L1 ([`batch_l1_u8`]) is memory-bound and trivially
//! auto-vectorised, so its math stays scalar (the `U8x64`-aligned 256-stride
//! layout lives in [`crate::DistanceLut`]); under the feature it still issues a
//! 64-wide `U8x64` load to keep the alignment contract visible.
//!
//! Both functions are correctness-equivalent across the two builds — the feature
//! is a pure accelerator, never a behaviour change.

// f32 clamp guard. Must exceed the f32 ULP near 1.0 (≈1.19e-7) so that
// `1.0 - EPS` is strictly < 1.0 in f32 — otherwise `ln(1 - s) = ln(0) = -inf`
// leaks through for `s = ±1`. (The f64 `Similarity::fisher_z` can use 1e-9.)
const EPS: f32 = 1e-6;

/// Batch Fisher-Z `z = ½·(ln(1+s) − ln(1−s))` over `similarities` into `out`
/// (each input clamped to `±(1 − 1e-6)`). `out.len()` must equal
/// `similarities.len()`. (ndarray `simd_ln_f32` path.)
#[cfg(feature = "ndarray-hpc")]
pub fn batch_fisher_z(similarities: &[f32], out: &mut [f32]) {
    use ndarray::simd::{simd_ln_f32, F32x16};
    assert_eq!(
        similarities.len(),
        out.len(),
        "batch_fisher_z: length mismatch"
    );
    let n = similarities.len();
    let mut pos = [0f32; 16];
    let mut neg = [0f32; 16];
    let mut i = 0;
    while i + 16 <= n {
        for l in 0..16 {
            let s = similarities[i + l].clamp(-1.0 + EPS, 1.0 - EPS);
            pos[l] = 1.0 + s;
            neg[l] = 1.0 - s;
        }
        let lp = simd_ln_f32(F32x16::from_slice(&pos)).to_array();
        let ln = simd_ln_f32(F32x16::from_slice(&neg)).to_array();
        for l in 0..16 {
            out[i + l] = 0.5 * (lp[l] - ln[l]);
        }
        i += 16;
    }
    while i < n {
        let s = similarities[i].clamp(-1.0 + EPS, 1.0 - EPS);
        out[i] = 0.5 * ((1.0 + s).ln() - (1.0 - s).ln());
        i += 1;
    }
}

/// Batch Fisher-Z (scalar fallback — identical results to the ndarray path).
#[cfg(not(feature = "ndarray-hpc"))]
pub fn batch_fisher_z(similarities: &[f32], out: &mut [f32]) {
    assert_eq!(
        similarities.len(),
        out.len(),
        "batch_fisher_z: length mismatch"
    );
    for (o, &s) in out.iter_mut().zip(similarities) {
        let s = s.clamp(-1.0 + EPS, 1.0 - EPS);
        *o = 0.5 * ((1.0 + s).ln() - (1.0 - s).ln());
    }
}

/// Batch L1 distance `|a − b|` over equal-length u8 endpoint slices into `out`.
/// (ndarray `U8x64` 64-wide load path.)
#[cfg(feature = "ndarray-hpc")]
pub fn batch_l1_u8(a: &[u8], b: &[u8], out: &mut [u16]) {
    use ndarray::simd::U8x64;
    assert_eq!(a.len(), b.len(), "batch_l1_u8: length mismatch");
    assert_eq!(a.len(), out.len(), "batch_l1_u8: length mismatch");
    let n = a.len();
    let mut i = 0;
    while i + 64 <= n {
        let av = U8x64::from_slice(&a[i..i + 64]).to_array();
        let bv = U8x64::from_slice(&b[i..i + 64]).to_array();
        for l in 0..64 {
            out[i + l] = (av[l] as i32 - bv[l] as i32).unsigned_abs() as u16;
        }
        i += 64;
    }
    while i < n {
        out[i] = (a[i] as i32 - b[i] as i32).unsigned_abs() as u16;
        i += 1;
    }
}

/// Batch L1 distance (scalar fallback).
#[cfg(not(feature = "ndarray-hpc"))]
pub fn batch_l1_u8(a: &[u8], b: &[u8], out: &mut [u16]) {
    assert_eq!(a.len(), b.len(), "batch_l1_u8: length mismatch");
    assert_eq!(a.len(), out.len(), "batch_l1_u8: length mismatch");
    for ((o, &x), &y) in out.iter_mut().zip(a).zip(b) {
        *o = (x as i32 - y as i32).unsigned_abs() as u16;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fisher_z::Similarity;

    #[test]
    fn batch_fisher_z_matches_scalar_reference() {
        let s: Vec<f32> = (0..40).map(|k| -0.95 + k as f32 * 0.045).collect();
        let mut out = vec![0f32; s.len()];
        batch_fisher_z(&s, &mut out);
        for (i, &si) in s.iter().enumerate() {
            let want = Similarity(si as f64).fisher_z() as f32;
            assert!(
                (out[i] - want).abs() < 1e-3,
                "i={i}: got {}, want {}",
                out[i],
                want
            );
        }
    }

    #[test]
    fn batch_fisher_z_boundary_inputs_are_finite() {
        // ±1 and out-of-range inputs must clamp to a finite result — the f32 EPS
        // must exceed the f32 ULP near 1.0, else ln(0) = -inf leaks through.
        let s = [1.0f32, -1.0, 2.0, -2.0, 0.999_999, -0.999_999];
        let mut out = vec![0f32; s.len()];
        batch_fisher_z(&s, &mut out);
        for (i, o) in out.iter().enumerate() {
            assert!(o.is_finite(), "out[{i}] = {o} must be finite");
        }
    }

    #[test]
    fn batch_l1_u8_is_abs_diff() {
        let a: Vec<u8> = (0..130u16).map(|x| x as u8).collect();
        let b: Vec<u8> = (0..130u16).map(|x| (255 - x) as u8).collect();
        let mut out = vec![0u16; a.len()];
        batch_l1_u8(&a, &b, &mut out);
        for i in 0..a.len() {
            assert_eq!(out[i], (a[i] as i32 - b[i] as i32).unsigned_abs() as u16);
        }
    }
}
