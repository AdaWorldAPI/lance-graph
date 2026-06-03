//! The three constants (three disjoint jobs) and the fixed residue parameters.
//!
//! Per `KNOWLEDGE.md` § "The Three Constants": their values sit close enough to
//! invite confusion; their *roles* are disjoint.
//!
//! - [`GOLDEN_RATIO`] (φ) — **PLACES**. The stride/placement constant: lowest
//!   star-discrepancy among all irrationals (continued fraction `[1;1,1,…]`,
//!   Ostrowski bound `2/N`). φ decides *where* points fall, with minimal aliasing.
//! - [`EULER_GAMMA`] (γ) — **CORRECTS**. The harmonic-to-log bridge
//!   `γ = lim(Σ 1/k − ln n)`: exactly the difference between a discrete rank-sum
//!   and the continuous log — the term that reconciles a discrete index rank
//!   with the continuous φ scale.
//! - [`E`] (e) — **DESCRIBES GROWTH**. The natural base for the exponential
//!   growth of golden-lattice spacing, and the base of `ln`/`exp` when the
//!   Fisher-Z / arctanh step leaves the LUT.
//!
//! **NOTE (2026-06-03 correction).** `KNOWLEDGE.md` references these as
//! `const::simd::GOLDEN_RATIO` etc. That path does **not** exist — the canonical
//! source is `std::f64::consts::{GOLDEN_RATIO, EULER_GAMMA, E}` (Rust ≥ 1.94) and
//! the `ndarray` fork does not wrap them. helix defines local consts (mirroring
//! `std`, exactly as `jc::weyl` defines its own `PHI_INV`) to stay zero-dep and
//! robust across toolchains that have not yet stabilised those float constants.

/// φ — the golden ratio `(1 + √5) / 2`. The PLACES constant.
pub const GOLDEN_RATIO: f64 = 1.618_033_988_749_895;

/// φ⁻¹ = `(√5 − 1) / 2` = `φ − 1`. The golden stride of the low-discrepancy
/// sequence `{k·φ⁻¹ mod 1}` (mirrors `jc::weyl`'s `PHI_INV`).
pub const GOLDEN_RATIO_INV: f64 = 0.618_033_988_749_894_9;

/// The golden angle in radians = `2π·(1 − φ⁻¹)` = `2π / φ²` ≈ 137.5°. Used for
/// the equal-area sunflower azimuth when a true angle (not the raw `n·φ` stride)
/// is wanted for Cartesian projection.
pub const GOLDEN_ANGLE: f64 = 2.399_963_229_728_653;

/// γ — Euler–Mascheroni. The CORRECTS constant (harmonic-to-log bridge).
pub const EULER_GAMMA: f64 = 0.577_215_664_901_532_9;

/// e — Euler's number. The DESCRIBES-GROWTH constant. (Sourced from `std`, which
/// is stable for `E`; `GOLDEN_RATIO` / `EULER_GAMMA` stay literals because they
/// are not stable on every supported toolchain.)
pub const E: f64 = std::f64::consts::E;

/// Prime modulus M = 17 — Base17 / zeck17 alignment. Because 17 is prime,
/// *every* coprime stride is a full permutation of all 17 residues.
pub const MODULUS: u8 = 17;

/// Stride = 4 — coprime to 17 (`gcd(4, 17) = 1`), so `(start + 4·k) mod 17`
/// visits all 17 residues. (The banned pattern is **Fibonacci mod 17**, which
/// misses {6, 7, 10, 11}; a coprime stride over a prime modulus does not.)
pub const STRIDE: u8 = 4;

/// `ln(17)` — the constant γ shove (Base17 alignment), applied **per
/// calibration, not per rank**, keeping the Euler hand-off deterministic.
pub const LN_17: f64 = 2.833_213_344_056_216;

/// Transient skip: the first ~17 golden-spiral indices are the bad-discrepancy
/// warm-up (the Ostrowski `2/N` bound is large for small N). Starting at 17
/// aligns to Base17; 20 sits just below F₈ = 21. Both skip the transient.
pub const TRANSIENT_SKIP: usize = 17;

/// Palette resolution: 256 buckets (int8 / `U8x64`-aligned). One bucket
/// (`1/256` = 0.390625%) is wider than the ±3σ tail (0.27%), so the tail
/// saturates into the two rim buckets and all inner resolution sits in the
/// informative range.
pub const PALETTE_SIZE: usize = 256;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn golden_ratio_identities() {
        // φ² = φ + 1
        assert!((GOLDEN_RATIO * GOLDEN_RATIO - (GOLDEN_RATIO + 1.0)).abs() < 1e-12);
        // φ⁻¹ = φ − 1
        assert!((GOLDEN_RATIO_INV - (GOLDEN_RATIO - 1.0)).abs() < 1e-12);
        // φ · φ⁻¹ = 1
        assert!((GOLDEN_RATIO * GOLDEN_RATIO_INV - 1.0).abs() < 1e-12);
    }

    #[test]
    fn derived_constants_match_std() {
        let ga = 2.0 * std::f64::consts::PI * (1.0 - GOLDEN_RATIO_INV);
        assert!((GOLDEN_ANGLE - ga).abs() < 1e-12);
        assert!((LN_17 - 17.0_f64.ln()).abs() < 1e-12);
        assert!((E - std::f64::consts::E).abs() < 1e-15);
    }

    #[test]
    fn stride_is_coprime_to_modulus() {
        // gcd(4, 17) = 1 → the stride visits all 17 residues (full permutation).
        let mut seen = [false; MODULUS as usize];
        let mut idx = 0u8;
        for _ in 0..MODULUS {
            seen[idx as usize] = true;
            idx = (idx + STRIDE) % MODULUS;
        }
        assert!(
            seen.iter().all(|&s| s),
            "stride-4 must visit all 17 residues"
        );
    }
}
