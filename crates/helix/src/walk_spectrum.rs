//! PROBE-WALK-SPECTRUM — measures the stride-4-mod-17 walk's autocorrelation
//! spectrum against a seeded-LCG ±1 baseline.
//!
//! Grades the §10(g) anti-confabulation claim from the H.268 probe wave: is
//! the walk's dependence structure actually "known and concentrated" (energy
//! only at period-17 harmonics, none at the palette-lattice periods 16/32/
//! 64/128/256), rather than a PRNG-shaped baseline would show? Cross-ref
//! `E-WH-TWO-SIDES-SIG-CHECKSUM-1` and `I-NOISE-FLOOR-JIRAK` (this probe is
//! the measured basis for the "weakly dependent by construction, and the
//! dependence is exactly where predicted" half of that iron rule — `gcd(17,
//! 256) = 1` is necessary but not sufficient, hence the measurement).
//!
//! **Test-only probe.** This module is `#![cfg(test)]`-gated end to end and
//! is not part of the crate's default build surface, and is not registered
//! in any pillar/`prove` list — it is a one-shot measurement, not a proof
//! obligation.
//!
//! ## Derivation note (worked by hand, then verified by the printed table)
//!
//! `s[k] = ±1` from the parity of the walk residue
//! `r(k) = (start + STRIDE·k) mod MODULUS`. Because `gcd(STRIDE, MODULUS) =
//! gcd(4, 17) = 1`, `r(k)` is an exact permutation of `0..17` with period
//! **exactly 17 in `k`** (unconditionally — this periodicity does not depend
//! on the sequence length `N`). Walking the permutation for `start = 0`
//! gives the residue order `0,4,8,12,16,3,7,11,15,2,6,10,14,1,5,9,13`, whose
//! parities are 9 even / 8 odd — **not balanced**. So `s` is exactly
//! period-17 but its one-period mean is `1/17 ≠ 0` (a small DC bias, not
//! flagged by this probe since only the AC structure at specific lags is
//! measured).
//!
//! `N = 4096` is **not** a multiple of 17 (`4096 mod 17 = 16`), so the
//! *circular* autocorrelation (which wraps at `N`, not at 17) is not an
//! exact period-17 identity — the wraparound compares `s[k]` against a
//! residue shifted by one extra step for a `τ`-sized tail of indices. Working
//! it through exactly: for `τ = 17·m`, `R(τ) = [(N − 17m)·1 + m·Σ_adj] / N`
//! where `Σ_adj = 9` is the fixed sum of adjacent-residue parity products
//! around one full period (computed directly from the residue-parity
//! sequence above). That simplifies to `R(17·m) = 1 − 8m/N`, i.e. `R(17) ≈
//! 0.998047`, `R(34) ≈ 0.996094`, `R(51) ≈ 0.994141` — all comfortably above
//! the 0.9 floor asserted below, confirming the "known, concentrated"
//! period-17 structure (not exactly 1, because `N` is not a clean multiple
//! of 17, but the deviation is a bounded `O(τ/N)` boundary effect, not
//! aliasing). For `τ` at the palette-lattice periods (16/32/64/128/256),
//! `gcd(17, τ) = 1` for all of them (17 is prime and none is a multiple of
//! 17), so no such exact-alignment mechanism exists — the printed table is
//! the actual measurement, not re-derived by hand here.

#![cfg(test)]

use crate::constants::{MODULUS, STRIDE};

/// Sequence length: `16 × 256` — full periods of the palette lattice.
const N: usize = 4096;
/// Max lag examined for the LCG baseline's overall-max context figure.
const TAU_MAX: usize = 512;
/// Deterministic walk start offset. The derivation above does not depend on
/// this choice: any start is a cyclic relabelling of the same 9-even/8-odd
/// residue-parity pattern, so the period-17 structure is unaffected.
const WALK_START: u8 = 0;
/// Deterministic LCG seed (SplitMix64's golden-ratio odd constant — used
/// here only as a fixed seed literal, not as a dependency).
const LCG_SEED: u64 = 0x9E37_79B9_7F4A_7C15;

/// The stride-4-mod-17 walk residue at step `k`, per `constants::{MODULUS,
/// STRIDE}` — mirrors `CurveRuler::index` without depending on that type.
fn walk_residue(start: u8, k: usize) -> u8 {
    let step = (STRIDE as u64 * k as u64) % MODULUS as u64;
    ((start as u64 + step) % MODULUS as u64) as u8
}

/// `±1` sign sequence from the parity of the walk residue.
fn walk_signs(start: u8, n: usize) -> Vec<i32> {
    (0..n)
        .map(|k| if walk_residue(start, k) % 2 == 0 { 1 } else { -1 })
        .collect()
}

/// One step of a 64-bit LCG (the PCG multiplier/increment pair — a fixed,
/// well-known constant choice, not a new crate dependency).
fn lcg_next(state: &mut u64) -> u64 {
    *state = state
        .wrapping_mul(6_364_136_223_846_793_005)
        .wrapping_add(1_442_695_040_888_963_407);
    *state
}

/// `±1` baseline sequence from a seeded LCG (top bit of each draw — the low
/// bits of a 64-bit LCG have short periods, so the top bit is used).
fn lcg_signs(seed: u64, n: usize) -> Vec<i32> {
    let mut state = seed;
    (0..n)
        .map(|_| if (lcg_next(&mut state) >> 63) & 1 == 1 { 1 } else { -1 })
        .collect()
}

/// Plain circular autocorrelation `R(τ) = (1/N) · Σ_k s[k]·s[(k+τ) mod N]`.
/// `O(N)` per call; the probe calls this `O(τ_max)` times, `O(N·τ_max)`
/// total — fine at this size (no FFT dependency).
fn autocorrelation(s: &[i32], tau: usize) -> f64 {
    let n = s.len();
    let mut acc: i64 = 0;
    for (k, &sk) in s.iter().enumerate() {
        let j = (k + tau) % n;
        acc += (sk * s[j]) as i64;
    }
    acc as f64 / n as f64
}

#[test]
fn probe_walk_spectrum() {
    let walk = walk_signs(WALK_START, N);
    let lcg = lcg_signs(LCG_SEED, N);

    // Structural sanity only — the printed table below is what the reviewer
    // adjudicates against the PASS/NEUTRAL/KILL bands, not these asserts.
    assert_eq!(walk.len(), N);
    assert_eq!(lcg.len(), N);
    assert!(walk.iter().all(|&v| v == 1 || v == -1));
    assert!(lcg.iter().all(|&v| v == 1 || v == -1));

    let harmonics = [17usize, 34, 51];
    let lattice = [16usize, 32, 64, 128, 256];

    eprintln!(
        "=== PROBE-WALK-SPECTRUM (N={N}, MODULUS={MODULUS}, STRIDE={STRIDE}, start={WALK_START}) ==="
    );

    eprintln!("-- walk |R(tau)| at period-17 harmonics (expected ~= 1) --");
    let mut harmonic_rs = Vec::with_capacity(harmonics.len());
    for &tau in &harmonics {
        let r = autocorrelation(&walk, tau);
        harmonic_rs.push(r);
        eprintln!("  tau={tau:4}  R={r:+.6}  |R|={:.6}", r.abs());
    }

    eprintln!("-- walk |R(tau)| at palette-lattice periods (gcd(17,tau)=1, expected ~ noise floor) --");
    let mut walk_lattice_max: f64 = 0.0;
    for &tau in &lattice {
        let r = autocorrelation(&walk, tau);
        walk_lattice_max = walk_lattice_max.max(r.abs());
        eprintln!("  tau={tau:4}  R={r:+.6}  |R|={:.6}", r.abs());
    }

    eprintln!("-- LCG baseline |R(tau)| at the same palette-lattice periods --");
    let mut lcg_lattice_max: f64 = 0.0;
    for &tau in &lattice {
        let r = autocorrelation(&lcg, tau);
        lcg_lattice_max = lcg_lattice_max.max(r.abs());
        eprintln!("  tau={tau:4}  R={r:+.6}  |R|={:.6}", r.abs());
    }

    let mut lcg_overall_max: f64 = 0.0;
    for tau in 1..=TAU_MAX {
        lcg_overall_max = lcg_overall_max.max(autocorrelation(&lcg, tau).abs());
    }

    eprintln!(
        "-- context: LCG baseline max|R| over tau in 1..={TAU_MAX} = {lcg_overall_max:.6}"
    );
    eprintln!(
        "walk lattice max|R| = {walk_lattice_max:.6}, LCG lattice max|R| = {lcg_lattice_max:.6}, ratio = {:.3}",
        walk_lattice_max / lcg_lattice_max.max(1e-12)
    );

    // Derivation-confirmed structural assertion only (see module doc): the
    // hand-worked R(17*m) = 1 - 8m/N formula puts all three harmonics above
    // 0.994 for N=4096. This is NOT the PASS/NEUTRAL/KILL verdict from the
    // plan — the reviewer adjudicates the full printed table above,
    // including whether the lattice-period ratio clears the 2x/3x bands.
    for (tau, &r) in harmonics.iter().zip(harmonic_rs.iter()) {
        assert!(
            r.abs() > 0.9,
            "period-17 harmonic tau={tau} |R|={r} unexpectedly low (derivation predicts ~0.99+)"
        );
    }
}
