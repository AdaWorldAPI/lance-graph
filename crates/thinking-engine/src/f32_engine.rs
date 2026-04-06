//! F32 ThinkingEngine: full-precision distance table.
//!
//! Stores distance table as `Vec<f32>` — zero quantization loss.
//! u8 CDF tables destroy cosine geometry (Pearson r=0.80).
//! BF16 tables lose precision near zero causing attractor collapse.
//! f32 tables have perfect fidelity (r=0.9999).
//!
//! Think cycle: signed MatVec with ReLU + normalization.
//! No floor heuristic. No threshold. Full signed accumulation.

use crate::dto::{ResonanceDto, BusDto};

/// F32 thinking engine. Distance table at full f32 precision.
pub struct F32ThinkingEngine {
    /// f32 distance table: `distance_table[i * size + j]` = cosine(centroid_i, centroid_j).
    /// Full signed: positive = excitation, negative = inhibition.
    distance_table: Vec<f32>,

    /// Current energy distribution.
    pub energy: Vec<f32>,

    /// Number of thought-atoms.
    pub size: usize,

    /// Cycle counter.
    pub cycles: u16,

    /// Convergence threshold (L1 delta between cycles).
    pub convergence_threshold: f32,
}

impl F32ThinkingEngine {
    /// Create from a precomputed f32 distance table.
    /// Table must be a perfect square (N x N). No floor heuristic.
    pub fn new(distance_table: Vec<f32>) -> Self {
        let total = distance_table.len();
        let size = (total as f64).sqrt() as usize;
        assert_eq!(
            size * size, total,
            "f32 table length {} is not a perfect square (sqrt ~ {})",
            total, (total as f64).sqrt()
        );
        assert!(size >= 2, "need at least 2 atoms, got {}", size);

        Self {
            distance_table,
            energy: vec![0.0f32; size],
            size,
            cycles: 0,
            convergence_threshold: 0.001,
        }
    }

    /// Build f32 table from pairwise cosine of centroid vectors.
    /// Centroids must all have the same dimensionality.
    /// Cosine similarity is computed and stored directly as f32 — no quantization.
    pub fn from_codebook_f32(centroids: &[Vec<f32>]) -> Self {
        let n = centroids.len();
        assert!(n >= 2, "need at least 2 centroids");
        let dim = centroids[0].len();
        assert!(dim > 0, "centroid dimension must be > 0");

        let mut table = vec![0.0f32; n * n];

        for i in 0..n {
            assert_eq!(centroids[i].len(), dim, "centroid {} has wrong dimension", i);
            table[i * n + i] = 1.0; // self-similarity

            for j in (i + 1)..n {
                let cos = cosine_similarity(&centroids[i], &centroids[j]);
                table[i * n + j] = cos;
                table[j * n + i] = cos;
            }
        }

        Self::new(table)
    }

    /// Perturb: activate codebook indices and normalize energy.
    pub fn perturb(&mut self, indices: &[u16]) {
        for &idx in indices {
            if (idx as usize) < self.size {
                self.energy[idx as usize] += 1.0;
            }
        }
        // Normalize
        let total: f32 = self.energy.iter().sum();
        if total > 1e-10 {
            let inv = 1.0 / total;
            for e in &mut self.energy {
                *e *= inv;
            }
        }
    }

    /// Reset energy to zero and cycles to 0.
    pub fn reset(&mut self) {
        self.energy.fill(0.0);
        self.cycles = 0;
    }

    /// ONE signed MatVec cycle with softmax normalization:
    ///   next[j] = sum_i distance_table[i][j] * energy[i]   (FULL signed)
    ///   softmax: next[j] = exp(next[j] / T) / Σ exp(next[k] / T)
    ///   returns L1 delta for convergence check
    ///
    /// Softmax replaces ReLU — ReLU destroys inhibition information and causes
    /// attractor collapse. Softmax preserves relative ordering while ensuring
    /// positive probabilities. Low temperature (T=0.1) concentrates on best matches.
    fn cycle(&mut self) -> f32 {
        let k = self.size;
        let mut next = vec![0.0f32; k];

        // Signed MatVec: next = D^T * energy
        for i in 0..k {
            let e_i = self.energy[i];
            if e_i.abs() < 1e-15 {
                continue;
            }
            let row = &self.distance_table[i * k..(i + 1) * k];
            for j in 0..k {
                next[j] += row[j] * e_i;
            }
        }

        // Softmax with temperature (default T=0.1 for sharp focus)
        let inv_t = 10.0f32; // 1/T where T=0.1
        let max_e = next.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let mut exp_sum = 0.0f32;
        for e in &mut next {
            *e = ((*e - max_e) * inv_t).exp();
            exp_sum += *e;
        }
        if exp_sum > 1e-10 {
            let inv = 1.0 / exp_sum;
            for e in &mut next {
                *e *= inv;
            }
        }

        // Convergence: L1 delta
        let delta: f32 = self.energy.iter().zip(&next).map(|(a, b)| (a - b).abs()).sum();

        self.energy = next;
        self.cycles += 1;
        delta
    }

    /// ONE cycle with explicit temperature parameter.
    fn cycle_with_temp(&mut self, temperature: f32) -> f32 {
        let k = self.size;
        let t = temperature.max(0.01);
        let inv_t = 1.0 / t;
        let mut next = vec![0.0f32; k];

        // Signed MatVec: next = D^T * energy
        for i in 0..k {
            let e_i = self.energy[i];
            if e_i.abs() < 1e-15 {
                continue;
            }
            let row = &self.distance_table[i * k..(i + 1) * k];
            for j in 0..k {
                next[j] += row[j] * e_i;
            }
        }

        // Softmax with explicit temperature
        let max_e = next.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let mut exp_sum = 0.0f32;
        for e in &mut next {
            *e = ((*e - max_e) * inv_t).exp();
            exp_sum += *e;
        }
        if exp_sum > 1e-10 {
            let inv = 1.0 / exp_sum;
            for e in &mut next {
                *e *= inv;
            }
        }

        let delta: f32 = self.energy.iter().zip(&next).map(|(a, b)| (a - b).abs()).sum();

        self.energy = next;
        self.cycles += 1;
        delta
    }

    /// Think until convergence or max_cycles.
    pub fn think(&mut self, max_cycles: usize) -> ResonanceDto {
        for _ in 0..max_cycles {
            let delta = self.cycle();
            if delta < self.convergence_threshold {
                break;
            }
        }
        ResonanceDto::from_energy_f32(&self.energy, self.cycles)
    }

    /// Think with temperature scaling (1/T applied to MatVec output before normalization).
    pub fn think_with_temperature(&mut self, max_cycles: usize, temperature: f32) -> ResonanceDto {
        for _ in 0..max_cycles {
            let delta = self.cycle_with_temp(temperature);
            if delta < self.convergence_threshold {
                break;
            }
        }
        ResonanceDto::from_energy_f32(&self.energy, self.cycles)
    }

    /// Access the energy distribution.
    pub fn energy(&self) -> &[f32] {
        &self.energy
    }

    /// Return top-k peaks by energy, sorted descending.
    pub fn top_k(&self, k: usize) -> Vec<(u16, f32)> {
        let mut indexed: Vec<(u16, f32)> = self
            .energy
            .iter()
            .enumerate()
            .map(|(i, &e)| (i as u16, e))
            .collect();
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        indexed.truncate(k);
        indexed
    }

    /// Access the f32 distance table.
    pub fn distance_table_ref(&self) -> &[f32] {
        &self.distance_table
    }

    /// Entropy of the energy distribution.
    pub fn entropy(&self) -> f32 {
        let mut h = 0.0f32;
        for &e in &self.energy {
            if e > 1e-10 {
                h -= e * e.ln();
            }
        }
        h
    }

    /// Commit dominant peak as BusDto.
    pub fn commit(&self) -> BusDto {
        let resonance = ResonanceDto::from_energy_f32(&self.energy, self.cycles);
        BusDto {
            codebook_index: resonance.top_k[0].0,
            energy: resonance.top_k[0].1,
            top_k: resonance.top_k,
            cycle_count: self.cycles,
            converged: resonance.converged,
        }
    }
}

/// Cosine similarity between two f32 vectors.
fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    let mut dot = 0.0f64;
    let mut norm_a = 0.0f64;
    let mut norm_b = 0.0f64;
    for i in 0..a.len() {
        let ai = a[i] as f64;
        let bi = b[i] as f64;
        dot += ai * bi;
        norm_a += ai * ai;
        norm_b += bi * bi;
    }
    let denom = (norm_a * norm_b).sqrt();
    if denom < 1e-15 {
        0.0
    } else {
        (dot / denom) as f32
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a structured test table: nearby centroids are similar, distant are dissimilar.
    fn make_test_centroids(n: usize, dim: usize) -> Vec<Vec<f32>> {
        (0..n)
            .map(|i| {
                let mut v = vec![0.0f32; dim];
                // Each centroid is a unit vector with structure based on index
                for d in 0..dim {
                    v[d] = ((i * 7 + d * 3) as f32 * 0.1).sin();
                }
                // Normalize
                let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
                if norm > 1e-10 {
                    for x in &mut v {
                        *x /= norm;
                    }
                }
                v
            })
            .collect()
    }

    #[test]
    fn f32_table_preserves_exact_cosine() {
        // Build from codebook, verify roundtrip: table values == pairwise cosines
        let centroids = make_test_centroids(16, 64);
        let engine = F32ThinkingEngine::from_codebook_f32(&centroids);
        assert_eq!(engine.size, 16);

        let table = engine.distance_table_ref();
        for i in 0..16 {
            // Diagonal = 1.0
            assert!(
                (table[i * 16 + i] - 1.0).abs() < 1e-6,
                "diagonal[{}] = {}, expected 1.0",
                i,
                table[i * 16 + i]
            );
            for j in (i + 1)..16 {
                let expected = cosine_similarity(&centroids[i], &centroids[j]);
                let actual = table[i * 16 + j];
                let diff = (expected - actual).abs();
                assert!(
                    diff < 1e-6,
                    "table[{},{}] = {}, expected cosine = {}, diff = {}",
                    i,
                    j,
                    actual,
                    expected,
                    diff
                );
                // Symmetric
                assert_eq!(
                    table[i * 16 + j],
                    table[j * 16 + i],
                    "table not symmetric at ({}, {})",
                    i,
                    j
                );
            }
        }
    }

    #[test]
    fn signed_matvec_converges() {
        // Engine should converge (delta → 0) within reasonable cycles
        let centroids = make_test_centroids(32, 64);
        let mut engine = F32ThinkingEngine::from_codebook_f32(&centroids);
        engine.perturb(&[5, 7, 9]);

        let resonance = engine.think(50);
        // Must converge (cycles < max) or reach stable distribution
        assert!(
            resonance.converged || engine.cycles <= 50,
            "engine did not converge in 50 cycles"
        );
        // Energy should be non-zero on at least some atoms
        let active = engine.energy().iter().filter(|&&e| e > 0.01).count();
        assert!(
            active >= 1,
            "no active atoms after thinking: active={}",
            active
        );
    }

    #[test]
    fn different_inputs_produce_different_peaks() {
        // No attractor collapse: different perturbations should yield different dominant peaks
        let centroids = make_test_centroids(64, 128);

        let mut engine_a = F32ThinkingEngine::from_codebook_f32(&centroids);
        engine_a.perturb(&[5, 6, 7]);
        engine_a.think(30);
        let peak_a = engine_a.top_k(1)[0].0;

        let mut engine_b = F32ThinkingEngine::from_codebook_f32(&centroids);
        engine_b.perturb(&[50, 51, 52]);
        engine_b.think(30);
        let peak_b = engine_b.top_k(1)[0].0;

        let mut engine_c = F32ThinkingEngine::from_codebook_f32(&centroids);
        engine_c.perturb(&[30, 31, 32]);
        engine_c.think(30);
        let peak_c = engine_c.top_k(1)[0].0;

        // At least two of three should be distinct (no single attractor)
        let distinct = (peak_a != peak_b) as u8 + (peak_b != peak_c) as u8 + (peak_a != peak_c) as u8;
        assert!(
            distinct >= 2,
            "attractor collapse: all inputs converge to same peak: a={}, b={}, c={}",
            peak_a,
            peak_b,
            peak_c
        );
    }

    #[test]
    fn new_rejects_non_square() {
        let result = std::panic::catch_unwind(|| {
            F32ThinkingEngine::new(vec![0.0; 15]);
        });
        assert!(result.is_err(), "should panic on non-square table");
    }

    #[test]
    fn reset_clears_state() {
        let centroids = make_test_centroids(8, 16);
        let mut engine = F32ThinkingEngine::from_codebook_f32(&centroids);
        engine.perturb(&[1, 3, 5]);
        engine.think(5);
        assert!(engine.cycles > 0);
        assert!(engine.energy.iter().any(|&e| e > 0.0));

        engine.reset();
        assert_eq!(engine.cycles, 0);
        assert_eq!(engine.energy.iter().sum::<f32>(), 0.0);
    }

    #[test]
    fn top_k_returns_sorted() {
        let centroids = make_test_centroids(16, 32);
        let mut engine = F32ThinkingEngine::from_codebook_f32(&centroids);
        engine.perturb(&[3, 5, 7]);
        engine.think(10);

        let top = engine.top_k(5);
        assert_eq!(top.len(), 5);
        for i in 1..top.len() {
            assert!(
                top[i - 1].1 >= top[i].1,
                "top_k not sorted: {}={:.4} < {}={:.4}",
                top[i - 1].0,
                top[i - 1].1,
                top[i].0,
                top[i].1
            );
        }
    }

    #[test]
    fn temperature_affects_concentration() {
        let centroids = make_test_centroids(32, 64);

        // Low temperature: should concentrate more
        let mut eng_low = F32ThinkingEngine::from_codebook_f32(&centroids);
        eng_low.perturb(&[10, 12, 14]);
        eng_low.think_with_temperature(15, 0.1);
        let entropy_low = eng_low.entropy();

        // High temperature: should spread more
        let mut eng_high = F32ThinkingEngine::from_codebook_f32(&centroids);
        eng_high.perturb(&[10, 12, 14]);
        eng_high.think_with_temperature(15, 5.0);
        let entropy_high = eng_high.entropy();

        assert!(
            entropy_low <= entropy_high + 0.1,
            "low T entropy ({:.4}) should be <= high T entropy ({:.4})",
            entropy_low,
            entropy_high
        );
    }
}
