//! SignedThinkingEngine: i8 distance table with excitation/inhibition.
//!
//! The signed path preserves gate polarity natively:
//!   +127 = maximally similar (strong excitation)
//!      0 = orthogonal (no influence)
//!   -128 = maximally opposed (strong inhibition)
//!
//! After each cycle, negative energy is clamped to zero (inhibited atoms die).
//! This creates COMPETITIVE dynamics: opposing atoms suppress each other.
//!
//! Biological analogy (structural, not metaphor):
//!   ~80% excitatory (glutamate) = positive i8 entries
//!   ~20% inhibitory (GABA)      = negative i8 entries
//!
//! L4 is already i8. With signed L1-L3, the entire stack is uniform.
//! VNNI hardware: i8x i8->i32 = VPDPBSSD = native instruction.

use crate::dto::{ResonanceDto, BusDto};
use ndarray::hpc::heel_f64x8::cosine_f64_simd;
use ndarray::simd::F32x16;

/// Signed thinking engine. Same as ThinkingEngine but with i8 distance table.
///
/// Positive table entries EXCITE (reinforce). Negative entries INHIBIT (suppress).
/// Zero entries have no influence (orthogonal).
///
/// The sign IS the gate decision. No SiLU-ONNX correction needed.
pub struct SignedThinkingEngine {
    /// Signed distance table. i8[-128, +127].
    distance_table: Vec<i8>,

    /// Current energy distribution. Non-negative after clamping.
    pub energy: Vec<f32>,

    /// Number of thought-atoms.
    pub size: usize,

    /// Cycle counter.
    pub cycles: u16,

    /// Convergence threshold.
    pub convergence_threshold: f32,

    /// How many atoms were inhibited (clamped from negative) last cycle.
    pub inhibited_last_cycle: usize,

    /// Cumulative inhibition count across all cycles.
    pub total_inhibitions: usize,

    /// Sign distribution: count of positive entries in the table.
    pub positive_count: usize,

    /// Sign distribution: count of negative entries in the table.
    pub negative_count: usize,

    /// Excitation/inhibition ratio (positive / total non-zero).
    pub ei_ratio: f32,
}

impl SignedThinkingEngine {
    /// Create from a precomputed signed distance table.
    pub fn new(distance_table: Vec<i8>) -> Self {
        let total = distance_table.len();
        let size = (total as f64).sqrt() as usize;
        assert_eq!(size * size, total,
            "distance table length {} is not a perfect square", total);
        assert!(size >= 4, "need at least 4 atoms");

        // Count sign distribution
        let mut pos = 0usize;
        let mut neg = 0usize;
        for &v in &distance_table {
            if v > 0 { pos += 1; }
            else if v < 0 { neg += 1; }
        }
        let non_zero = (pos + neg).max(1);
        let ei_ratio = pos as f32 / non_zero as f32;

        Self {
            distance_table,
            energy: vec![0.0f32; size],
            size,
            cycles: 0,
            convergence_threshold: 0.001,
            inhibited_last_cycle: 0,
            total_inhibitions: 0,
            positive_count: pos,
            negative_count: neg,
            ei_ratio,
        }
    }

    /// Convert an unsigned u8 table to signed i8.
    ///
    /// Maps u8[0,255] -> i8[-128,+127] by subtracting 128.
    ///   u8=0   (cos=-1) -> i8=-128 (strong inhibition)
    ///   u8=128 (cos= 0) -> i8=0    (no influence)
    ///   u8=255 (cos=+1) -> i8=+127 (strong excitation)
    pub fn from_unsigned(table: &[u8]) -> Self {
        let signed: Vec<i8> = table.iter()
            .map(|&v| (v as i16 - 128) as i8)
            .collect();
        Self::new(signed)
    }

    /// Build signed table directly from f32 cosine matrix.
    ///
    /// This is the CORRECT i8 path: takes the raw f32 cosines
    /// (e.g. from silu(gate) × up pairwise cosine) and quantizes
    /// directly to i8 WITHOUT CDF encoding. The signs are preserved.
    ///
    /// WARNING: from_unsigned() and from_gate_corrected() both go
    /// through CDF u8 → relabel → i8, which DESTROYS sign information.
    /// Only this method and build_signed_table() produce real signed tables.
    pub fn from_f32_cosines(cosines: &[f32], size: usize) -> Self {
        assert_eq!(cosines.len(), size * size,
            "cosine matrix must be {}×{} = {}, got {}",
            size, size, size * size, cosines.len());
        let signed: Vec<i8> = cosines.iter()
            .map(|&c| (c * 127.0).round().clamp(-128.0, 127.0) as i8)
            .collect();
        Self::new(signed)
    }

    /// Build signed distance table directly from centroid vectors.
    /// cosine [-1,+1] -> i8 [-128,+127]. No information lost vs unsigned.
    pub fn build_signed_table(centroids: &[Vec<f64>]) -> Vec<i8> {
        let k = centroids.len();
        let mut table = vec![0i8; k * k];
        for i in 0..k {
            table[i * k + i] = 127; // self = max excitation
            for j in (i + 1)..k {
                let cos = cosine_f64_simd(&centroids[i], &centroids[j]);
                let s = (cos * 127.0).round().clamp(-128.0, 127.0) as i8;
                table[i * k + j] = s;
                table[j * k + i] = s; // symmetric
            }
        }
        table
    }

    /// ONE signed thinking cycle. Excitation + inhibition.
    ///
    /// Positive table entries EXCITE (add energy).
    /// Negative table entries INHIBIT (subtract energy).
    /// After accumulation, negative energy is clamped to zero.
    /// Inhibited atoms are dead for this cycle.
    ///
    /// SIMD: 4x F32x16 per iteration = 64 elements.
    pub fn cycle(&mut self) {
        let k = self.size;
        let mut next = vec![0.0f32; k];
        let inv_127 = 1.0f32 / 127.0;

        for i in 0..k {
            let e_i = self.energy[i];
            if e_i < 1e-10 { continue; }

            let row = &self.distance_table[i * k..(i + 1) * k];
            let e_scaled = e_i * inv_127;

            // 4x F32x16 = 64 elements per iteration
            let mut j = 0;
            while j + 64 <= k {
                macro_rules! do_lane {
                    ($off:expr) => {{
                        let base = j + $off * 16;
                        let d = F32x16::from_array([
                            row[base] as f32,      row[base + 1] as f32,
                            row[base + 2] as f32,  row[base + 3] as f32,
                            row[base + 4] as f32,  row[base + 5] as f32,
                            row[base + 6] as f32,  row[base + 7] as f32,
                            row[base + 8] as f32,  row[base + 9] as f32,
                            row[base + 10] as f32, row[base + 11] as f32,
                            row[base + 12] as f32, row[base + 13] as f32,
                            row[base + 14] as f32, row[base + 15] as f32,
                        ]);
                        let acc = F32x16::from_slice(&next[base..base + 16]);
                        let ei = F32x16::splat(e_scaled);
                        d.mul_add(ei, acc).copy_to_slice(&mut next[base..base + 16]);
                    }};
                }
                do_lane!(0); do_lane!(1); do_lane!(2); do_lane!(3);
                j += 64;
            }
            // Scalar tail
            while j < k {
                next[j] += row[j] as f32 * e_scaled;
                j += 1;
            }
        }

        // CLAMP: inhibited atoms die. Count them.
        let mut inhibited = 0usize;
        for e in &mut next {
            if *e < 0.0 {
                *e = 0.0;
                inhibited += 1;
            }
        }
        self.inhibited_last_cycle = inhibited;
        self.total_inhibitions += inhibited;

        // Normalize: total energy = 1.0
        let total: f32 = next.iter().sum();
        if total > 1e-10 {
            let inv = 1.0 / total;
            for e in &mut next { *e *= inv; }
        }

        self.energy = next;
        self.cycles += 1;
    }

    /// Run until convergence. Returns the resonance state.
    pub fn think(&mut self, max_cycles: usize) -> ResonanceDto {
        for _ in 0..max_cycles {
            let prev = self.energy.clone();
            self.cycle();

            let delta: f32 = self.energy.iter().zip(&prev)
                .map(|(a, b)| (a - b).abs()).sum();
            if delta < self.convergence_threshold {
                break;
            }
        }
        ResonanceDto::from_energy_f32(&self.energy, self.cycles)
    }

    /// Inject perturbation from sensor output.
    pub fn perturb(&mut self, codebook_indices: &[u16]) {
        for &idx in codebook_indices {
            if (idx as usize) < self.size {
                self.energy[idx as usize] += 1.0;
            }
        }
        let total: f32 = self.energy.iter().sum();
        if total > 1e-10 {
            let inv = 1.0 / total;
            for e in &mut self.energy { *e *= inv; }
        }
    }

    /// Reset energy to zero.
    pub fn reset(&mut self) {
        self.energy.fill(0.0);
        self.cycles = 0;
        self.inhibited_last_cycle = 0;
        self.total_inhibitions = 0;
    }

    /// Commit: dominant peak -> BusDto.
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

    /// Access the signed distance table.
    pub fn distance_table_ref(&self) -> &[i8] { &self.distance_table }

    /// Entropy of current energy distribution.
    pub fn entropy(&self) -> f32 {
        let mut h = 0.0f32;
        for &e in &self.energy {
            if e > 1e-10 {
                h -= e * e.ln();
            }
        }
        h
    }

    /// Number of active thought-atoms (energy > threshold).
    pub fn active_count(&self, threshold: f32) -> usize {
        self.energy.iter().filter(|&&e| e > threshold).count()
    }

    /// Sign distribution statistics.
    pub fn sign_stats(&self) -> String {
        let total = self.size * self.size;
        let zero = total - self.positive_count - self.negative_count;
        format!(
            "E/I ratio: {:.1}% excitatory, {:.1}% inhibitory, {:.1}% zero ({} pos, {} neg, {} zero)",
            self.ei_ratio * 100.0,
            (1.0 - self.ei_ratio) * 100.0,
            zero as f32 / total as f32 * 100.0,
            self.positive_count,
            self.negative_count,
            zero,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_signed_test_table(k: usize) -> Vec<i8> {
        let mut table = vec![0i8; k * k];
        for i in 0..k {
            table[i * k + i] = 127; // self = max excitation
            for j in 0..k {
                if i == j { continue; }
                let dist = (i as i64 - j as i64).unsigned_abs() as usize;
                if dist < 30 {
                    // Near neighbors: excitatory (positive)
                    let val = 100 - (dist as i32) * 3;
                    table[i * k + j] = val.clamp(1, 127) as i8;
                } else if dist > k / 2 {
                    // Distant atoms: inhibitory (opposite side)
                    let strength = ((dist - k / 2) as i32) * 2;
                    table[i * k + j] = (-strength).clamp(-128, -1) as i8;
                }
            }
        }
        table
    }

    #[test]
    fn signed_engine_creates() {
        let table = make_signed_test_table(256);
        let engine = SignedThinkingEngine::new(table);
        assert_eq!(engine.size, 256);
        assert_eq!(engine.energy.iter().sum::<f32>(), 0.0);
        assert!(engine.positive_count > 0);
        assert!(engine.negative_count > 0);
    }

    #[test]
    fn from_unsigned_preserves_semantics() {
        // u8=0 -> i8=-128, u8=128 -> i8=0, u8=255 -> i8=127
        // Need at least 4×4 = 16 entries (min 4 atoms)
        let mut unsigned = vec![128u8; 16]; // 4×4, all orthogonal
        unsigned[0] = 0;    // [0][0] -> -128
        unsigned[1] = 128;  // [0][1] -> 0
        unsigned[2] = 255;  // [0][2] -> 127
        unsigned[3] = 64;   // [0][3] -> -64
        let engine = SignedThinkingEngine::from_unsigned(&unsigned);
        let table = engine.distance_table_ref();
        assert_eq!(table[0], -128);
        assert_eq!(table[1], 0);
        assert_eq!(table[2], 127);
        assert_eq!(table[3], -64);
    }

    #[test]
    fn signed_perturb_normalizes() {
        let table = make_signed_test_table(256);
        let mut engine = SignedThinkingEngine::new(table);
        engine.perturb(&[42, 100]);
        let total: f32 = engine.energy.iter().sum();
        assert!((total - 1.0).abs() < 1e-6);
        assert!(engine.energy[42] > 0.0);
        assert!(engine.energy[100] > 0.0);
    }

    #[test]
    fn signed_cycle_inhibits() {
        let table = make_signed_test_table(256);
        let mut engine = SignedThinkingEngine::new(table);

        // Perturb two distant atoms — their neighborhoods inhibit each other
        engine.perturb(&[10, 200]);
        engine.cycle();

        assert!(engine.inhibited_last_cycle > 0,
            "Expected inhibition with distant atoms, got 0 inhibitions");
    }

    #[test]
    fn signed_think_converges() {
        let table = make_signed_test_table(256);
        let mut engine = SignedThinkingEngine::new(table);

        engine.perturb(&[50, 55, 60]);
        let resonance = engine.think(20);

        assert!(resonance.cycle_count <= 20);
        assert!(resonance.top_k[0].1 > 0.0);
    }

    #[test]
    fn signed_commit_works() {
        let table = make_signed_test_table(256);
        let mut engine = SignedThinkingEngine::new(table);
        engine.perturb(&[42]);
        engine.think(10);

        let bus = engine.commit();
        assert!(bus.energy > 0.0);
    }

    #[test]
    fn signed_reset_clears() {
        let table = make_signed_test_table(256);
        let mut engine = SignedThinkingEngine::new(table);
        engine.perturb(&[42]);
        engine.think(5);
        engine.reset();
        assert_eq!(engine.energy.iter().sum::<f32>(), 0.0);
        assert_eq!(engine.cycles, 0);
        assert_eq!(engine.total_inhibitions, 0);
    }

    #[test]
    fn ei_ratio_computed() {
        let table = make_signed_test_table(256);
        let engine = SignedThinkingEngine::new(table);
        assert!(engine.ei_ratio > 0.0 && engine.ei_ratio < 1.0,
            "E/I ratio should be between 0 and 1, got {}", engine.ei_ratio);
    }

    #[test]
    fn build_signed_table_symmetric() {
        let centroids: Vec<Vec<f64>> = (0..64).map(|i| {
            (0..32).map(|d| ((i * 97 + d * 31) as f64 % 200.0 - 100.0) * 0.01).collect()
        }).collect();

        let table = SignedThinkingEngine::build_signed_table(&centroids);
        assert_eq!(table.len(), 64 * 64);

        for i in 0..64 {
            assert_eq!(table[i * 64 + i], 127, "diagonal should be 127");
            for j in 0..64 {
                assert_eq!(table[i * 64 + j], table[j * 64 + i],
                    "table[{},{}] != table[{},{}]", i, j, j, i);
            }
        }
    }
}
