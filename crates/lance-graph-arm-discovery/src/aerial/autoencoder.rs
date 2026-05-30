// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! The Aerial+ under-complete **denoising autoencoder** (paper §3.2).
//!
//! Architecture (one hidden layer, the paper's minimal form):
//!
//! ```text
//!   x  (D one-hot slots) ──noise──► x̃
//!        │ encoder  h = tanh(We·x̃ + be)         (H < D, under-complete)
//!        ▼
//!   h  (H)
//!        │ decoder  z = Wd·h + bd
//!        ▼  softmax PER FEATURE BLOCK
//!   p  (D, each block a probability distribution over its categories)
//! ```
//!
//! Loss is **categorical cross-entropy per block** summed over features —
//! the multi-class form of the paper's BCE-per-feature objective. The
//! softmax+CE pairing gives the clean logit gradient `dz = p − target`,
//! which keeps the hand-written backprop short and correct.
//!
//! Denoising (random input masking) is **load-bearing, not decorative**: it
//! is what forces the bottleneck to predict a feature from the *other*
//! features, which is exactly what the reconstruction probe in
//! [`crate::aerial::extract`] reads back out as an association rule.

// Dense GEMV / backprop kernels below are clearest as explicit index loops
// over the row-major weight matrices (multiple parallel arrays indexed by the
// same `i`/`j`). Clippy's `enumerate().take().skip()` rewrite would obscure
// the linear-algebra structure, so the lint is allowed module-wide here.
#![allow(clippy::needless_range_loop)]

use crate::aerial::rng::Rng;
use crate::encode::{Dataset, FeatureSpec};

/// A trained (or freshly-initialised) Aerial+ autoencoder.
#[derive(Debug, Clone)]
pub struct AerialAutoencoder {
    dim: usize,
    hidden: usize,
    /// Per-feature `[start, end)` blocks for the softmax.
    blocks: Vec<(usize, usize)>,
    /// Encoder weights, row-major `[hidden][dim]`.
    we: Vec<f32>,
    /// Encoder bias `[hidden]`.
    be: Vec<f32>,
    /// Decoder weights, row-major `[dim][hidden]`.
    wd: Vec<f32>,
    /// Decoder bias `[dim]`.
    bd: Vec<f32>,
}

impl AerialAutoencoder {
    /// Initialise an autoencoder for `spec` with `hidden` latent units.
    /// Weights are Xavier-scaled draws from the seeded `rng`; biases zero.
    #[must_use]
    pub fn new(spec: &FeatureSpec, hidden: usize, rng: &mut Rng) -> Self {
        let dim = spec.dim();
        assert!(hidden > 0, "hidden dim must be ≥ 1");
        let blocks: Vec<(usize, usize)> =
            (0..spec.num_features()).map(|f| spec.block(f)).collect();

        let enc_scale = (1.0 / dim as f32).sqrt();
        let dec_scale = (1.0 / hidden as f32).sqrt();
        let we = (0..hidden * dim).map(|_| rng.normal() * enc_scale).collect();
        let wd = (0..dim * hidden).map(|_| rng.normal() * dec_scale).collect();

        Self {
            dim,
            hidden,
            blocks,
            we,
            be: vec![0.0; hidden],
            wd,
            bd: vec![0.0; dim],
        }
    }

    /// Input dimension `D`.
    #[must_use]
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// Hidden dimension `H`.
    #[must_use]
    pub fn hidden(&self) -> usize {
        self.hidden
    }

    /// Forward pass on an arbitrary input vector (one-hot row OR a probe
    /// vector with uniform blocks). Returns `(hidden, probs)` where `probs`
    /// is per-block softmax over the decoder logits.
    #[must_use]
    pub fn forward(&self, input: &[f32]) -> (Vec<f32>, Vec<f32>) {
        debug_assert_eq!(input.len(), self.dim);
        // Encoder: h = tanh(We·x + be)
        let mut h = vec![0.0f32; self.hidden];
        for j in 0..self.hidden {
            let mut acc = self.be[j];
            let base = j * self.dim;
            for i in 0..self.dim {
                acc += self.we[base + i] * input[i];
            }
            h[j] = acc.tanh();
        }
        // Decoder logits: z = Wd·h + bd
        let mut z = vec![0.0f32; self.dim];
        for i in 0..self.dim {
            let mut acc = self.bd[i];
            let base = i * self.hidden;
            for j in 0..self.hidden {
                acc += self.wd[base + j] * h[j];
            }
            z[i] = acc;
        }
        // Per-block softmax.
        let mut p = z;
        for &(s, e) in &self.blocks {
            softmax_in_place(&mut p[s..e]);
        }
        (h, p)
    }

    /// Reconstruct per-block probabilities for an input (forward, probs only).
    #[must_use]
    pub fn reconstruct(&self, input: &[f32]) -> Vec<f32> {
        self.forward(input).1
    }

    /// Mean cross-entropy loss over a dataset's clean one-hot encodings.
    /// Used as a training-progress signal in tests and diagnostics.
    #[must_use]
    pub fn mean_loss(&self, data: &Dataset) -> f32 {
        if data.is_empty() {
            return 0.0;
        }
        let mut total = 0.0f32;
        for row in &data.rows {
            let x = data.spec.encode(row);
            let p = self.reconstruct(&x);
            total += cross_entropy(&x, &p, &self.blocks);
        }
        total / data.len() as f32
    }

    /// Train in place with denoising SGD. Returns the final epoch's mean loss.
    ///
    /// One pass per epoch over a shuffled index set; per-sample weight update.
    /// Each epoch corrupts inputs afresh (mask each slot with probability
    /// `noise`) while the reconstruction target stays the clean one-hot row.
    pub fn train(
        &mut self,
        data: &Dataset,
        epochs: usize,
        learning_rate: f32,
        noise: f32,
        rng: &mut Rng,
    ) -> f32 {
        if data.is_empty() {
            return 0.0;
        }
        // Pre-encode clean targets once.
        let targets: Vec<Vec<f32>> = data.rows.iter().map(|r| data.spec.encode(r)).collect();
        let mut order: Vec<usize> = (0..targets.len()).collect();
        let mut last = 0.0f32;

        for _epoch in 0..epochs {
            rng.shuffle(&mut order);
            let mut epoch_loss = 0.0f32;
            for &idx in &order {
                let target = &targets[idx];
                // Corrupt: copy then mask slots to 0 with probability `noise`.
                let mut input = target.clone();
                if noise > 0.0 {
                    for v in &mut input {
                        if rng.bernoulli(noise) {
                            *v = 0.0;
                        }
                    }
                }
                epoch_loss += self.train_step(&input, target, learning_rate);
            }
            last = epoch_loss / targets.len() as f32;
        }
        last
    }

    /// One forward+backward+update on a single (corrupted input, clean
    /// target) pair. Returns the cross-entropy loss before the update.
    fn train_step(&mut self, input: &[f32], target: &[f32], lr: f32) -> f32 {
        // Forward (recompute hidden so we can backprop through it).
        let (h, p) = self.forward(input);
        let loss = cross_entropy(target, &p, &self.blocks);

        // Output gradient: dz = p − target  (softmax + CE).
        let mut dz = vec![0.0f32; self.dim];
        for i in 0..self.dim {
            dz[i] = p[i] - target[i];
        }

        // Backprop into hidden: dh[j] = Σ_i Wd[i][j]·dz[i]
        let mut dh = vec![0.0f32; self.hidden];
        for i in 0..self.dim {
            let base = i * self.hidden;
            let dzi = dz[i];
            for j in 0..self.hidden {
                dh[j] += self.wd[base + j] * dzi;
            }
        }

        // Decoder update: Wd[i][j] -= lr·dz[i]·h[j]; bd[i] -= lr·dz[i]
        for i in 0..self.dim {
            let base = i * self.hidden;
            let dzi = dz[i];
            for j in 0..self.hidden {
                self.wd[base + j] -= lr * dzi * h[j];
            }
            self.bd[i] -= lr * dzi;
        }

        // Through tanh: da[j] = dh[j]·(1 − h[j]²)
        // Encoder update: We[j][i] -= lr·da[j]·input[i]; be[j] -= lr·da[j]
        for j in 0..self.hidden {
            let da = dh[j] * (1.0 - h[j] * h[j]);
            let base = j * self.dim;
            for i in 0..self.dim {
                self.we[base + i] -= lr * da * input[i];
            }
            self.be[j] -= lr * da;
        }

        loss
    }
}

/// Numerically-stable softmax in place over a slice (one feature block).
fn softmax_in_place(block: &mut [f32]) {
    if block.is_empty() {
        return;
    }
    let max = block.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0f32;
    for v in block.iter_mut() {
        *v = (*v - max).exp();
        sum += *v;
    }
    if sum > 0.0 {
        for v in block.iter_mut() {
            *v /= sum;
        }
    }
}

/// Cross-entropy `−Σ target·ln(p)` summed over all blocks. `target` is the
/// clean one-hot vector; `p` is the per-block softmax output.
fn cross_entropy(target: &[f32], p: &[f32], blocks: &[(usize, usize)]) -> f32 {
    let mut loss = 0.0f32;
    for &(s, e) in blocks {
        for i in s..e {
            if target[i] > 0.0 {
                loss -= target[i] * (p[i].max(1e-9)).ln();
            }
        }
    }
    loss
}

#[cfg(test)]
mod tests {
    use super::*;

    fn corr_dataset(n: usize, seed: u64) -> Dataset {
        // feature1 == feature0 (deterministic dependency); feature2 random.
        let spec = FeatureSpec::new(vec![2, 2, 2]);
        let mut rng = Rng::new(seed);
        let rows = (0..n)
            .map(|_| {
                let a = (rng.next_u64() % 2) as u32;
                let c = (rng.next_u64() % 2) as u32;
                vec![a, a, c]
            })
            .collect();
        Dataset::new(spec, rows)
    }

    #[test]
    fn forward_blocks_are_probability_distributions() {
        let spec = FeatureSpec::new(vec![2, 3]);
        let mut rng = Rng::new(1);
        let ae = AerialAutoencoder::new(&spec, 4, &mut rng);
        let (_, p) = ae.forward(&spec.encode(&[0, 2]));
        // each block sums to ~1
        assert!((p[0..2].iter().sum::<f32>() - 1.0).abs() < 1e-5);
        assert!((p[2..5].iter().sum::<f32>() - 1.0).abs() < 1e-5);
    }

    #[test]
    fn training_reduces_loss() {
        let data = corr_dataset(300, 7);
        let mut rng = Rng::new(42);
        let mut ae = AerialAutoencoder::new(&data.spec, 4, &mut rng);
        let before = ae.mean_loss(&data);
        ae.train(&data, 400, 0.1, 0.3, &mut rng);
        let after = ae.mean_loss(&data);
        assert!(
            after < before * 0.7,
            "loss should drop substantially: before={before}, after={after}"
        );
    }

    #[test]
    fn denoising_ae_learns_cross_feature_dependency() {
        // After training, probing "feature0 = 0" (one-hot) with the rest of
        // the input left at uniform should reconstruct feature1 = 0 with
        // high probability — that IS the learned A→B dependency.
        let data = corr_dataset(400, 11);
        let mut rng = Rng::new(42);
        let mut ae = AerialAutoencoder::new(&data.spec, 4, &mut rng);
        ae.train(&data, 600, 0.1, 0.3, &mut rng);

        // Probe vector: feature0 block = one-hot(0), features 1 and 2 uniform.
        let mut probe = vec![0.5f32; data.spec.dim()]; // all uniform (cards=2)
        probe[0] = 1.0; // feature0 = cat0
        probe[1] = 0.0;
        let p = ae.reconstruct(&probe);
        // feature1 block is slots [2,4): expect cat0 (slot 2) dominant.
        assert!(
            p[2] > 0.6,
            "feature1=0 should be reconstructed from feature0=0; p={:?}",
            &p[2..4]
        );
        assert!(p[2] > p[3], "the correlated category must win");
    }

    #[test]
    fn training_is_reproducible_from_seed() {
        let data = corr_dataset(200, 5);
        let mut ae1 = AerialAutoencoder::new(&data.spec, 4, &mut Rng::new(42));
        let mut ae2 = AerialAutoencoder::new(&data.spec, 4, &mut Rng::new(42));
        let l1 = ae1.train(&data, 100, 0.1, 0.2, &mut Rng::new(123));
        let l2 = ae2.train(&data, 100, 0.1, 0.2, &mut Rng::new(123));
        assert_eq!(l1, l2, "same seeds ⇒ identical training");
        assert_eq!(ae1.we, ae2.we);
        assert_eq!(ae1.wd, ae2.wd);
    }
}
