// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! A tiny deterministic PRNG (SplitMix64) — zero-dep, seedable.
//!
//! The whole reason Aerial+ can be a *reproducible* fan-in proposer despite
//! being a neural net is that every source of randomness (weight init,
//! denoising mask, epoch shuffle) draws from one seeded stream. Same seed ⇒
//! identical weights ⇒ identical mined rules.

/// SplitMix64 — fast, well-distributed, fully deterministic from a `u64` seed.
#[derive(Debug, Clone)]
pub struct Rng {
    state: u64,
}

impl Rng {
    /// Seed the generator. Any `u64` is a valid seed.
    #[must_use]
    pub fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    /// Next raw `u64`.
    pub fn next_u64(&mut self) -> u64 {
        // SplitMix64 (Steele, Lea, Flood 2014).
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Uniform `f32` in `[0, 1)`.
    pub fn next_f32(&mut self) -> f32 {
        // Top 24 bits → mantissa precision of f32.
        let bits = (self.next_u64() >> 40) as u32; // 24 bits
        bits as f32 / (1u32 << 24) as f32
    }

    /// Uniform `f32` in `[-range, range)`.
    pub fn uniform(&mut self, range: f32) -> f32 {
        (self.next_f32() * 2.0 - 1.0) * range
    }

    /// Standard-normal `f32` via Box–Muller.
    pub fn normal(&mut self) -> f32 {
        // Guard u1 away from 0 so ln is finite.
        let u1 = (self.next_f32()).max(1e-7);
        let u2 = self.next_f32();
        let r = (-2.0 * u1.ln()).sqrt();
        r * (std::f32::consts::TAU * u2).cos()
    }

    /// Fisher–Yates shuffle of an index slice in place.
    pub fn shuffle(&mut self, idx: &mut [usize]) {
        let len = idx.len();
        for i in (1..len).rev() {
            let j = (self.next_u64() % (i as u64 + 1)) as usize;
            idx.swap(i, j);
        }
    }

    /// Bernoulli draw: `true` with probability `p`.
    pub fn bernoulli(&mut self, p: f32) -> bool {
        self.next_f32() < p
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deterministic_from_seed() {
        let mut a = Rng::new(42);
        let mut b = Rng::new(42);
        for _ in 0..100 {
            assert_eq!(a.next_u64(), b.next_u64());
        }
    }

    #[test]
    fn different_seeds_diverge() {
        let mut a = Rng::new(1);
        let mut b = Rng::new(2);
        assert_ne!(a.next_u64(), b.next_u64());
    }

    #[test]
    fn uniform_in_range() {
        let mut r = Rng::new(7);
        for _ in 0..1000 {
            let x = r.next_f32();
            assert!((0.0..1.0).contains(&x));
        }
    }

    #[test]
    fn shuffle_is_a_permutation() {
        let mut r = Rng::new(99);
        let mut idx: Vec<usize> = (0..50).collect();
        r.shuffle(&mut idx);
        let mut sorted = idx.clone();
        sorted.sort_unstable();
        assert_eq!(sorted, (0..50).collect::<Vec<_>>());
    }

    #[test]
    fn normal_has_reasonable_spread() {
        let mut r = Rng::new(3);
        let n = 5000;
        let mut sum = 0.0f64;
        let mut sumsq = 0.0f64;
        for _ in 0..n {
            let x = r.normal() as f64;
            sum += x;
            sumsq += x * x;
        }
        let mean = sum / n as f64;
        let var = sumsq / n as f64 - mean * mean;
        assert!(mean.abs() < 0.1, "mean ~0, got {mean}");
        assert!((var - 1.0).abs() < 0.2, "var ~1, got {var}");
    }
}
