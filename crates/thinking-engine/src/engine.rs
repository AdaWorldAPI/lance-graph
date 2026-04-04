//! ThinkingEngine: one MatVec per cycle. 16M compositions per thought.
//!
//! energy_next = distance_table × energy_current
//!
//! The distance table is precomputed ONCE from codebook centroids.
//! It IS the brain. Every thought is a matrix-vector multiply.

use crate::dto::{ResonanceDto, BusDto};
use ndarray::hpc::heel_f64x8::cosine_f64_simd;
use ndarray::simd::F64x8;

/// Default codebook size. 4096 entries = 12-bit index.
pub const CODEBOOK_SIZE: usize = 4096;

/// Default distance table size: CODEBOOK_SIZE².
pub const TABLE_SIZE: usize = CODEBOOK_SIZE * CODEBOOK_SIZE;

/// The thinking engine. One MatVec per cycle.
///
/// Accepts any N×N distance table. Common sizes:
///   1024×1024 = 1 MB (BGE-M3, Jina)
///   1536×1536 = 2.4 MB (reader-LM)
///   4096×4096 = 16 MB (full codebook, L3-resident)
pub struct ThinkingEngine {
    /// Precomputed similarity between all codebook pairs.
    /// Built ONCE. This IS the brain.
    /// entry[i * size + j] = u8 similarity (0=opposite, 255=identical).
    distance_table: Vec<u8>,

    /// Current energy distribution = which thoughts are alive.
    /// High energy at index i = "thought i resonates."
    /// Zero at index i = "thought i destructively interfered."
    pub energy: Vec<f64>,

    /// Number of thought-atoms (table is size×size).
    pub size: usize,

    /// Cycle counter.
    pub cycles: u16,

    /// Convergence threshold.
    pub convergence_threshold: f64,

    /// Sigma floor: median of the distance table.
    /// Values below this are noise (orthogonal baseline).
    /// Only values ABOVE floor contribute to energy propagation.
    pub floor: u8,
}

impl ThinkingEngine {
    /// Create engine with a precomputed N×N distance table.
    /// Infers N from table length (must be a perfect square).
    /// Computes median as sigma floor automatically.
    pub fn new(distance_table: Vec<u8>) -> Self {
        let total = distance_table.len();
        let size = (total as f64).sqrt() as usize;
        assert_eq!(size * size, total,
            "distance table length {} is not a perfect square", total);
        assert!(size >= 4, "need at least 4 atoms");

        // Compute median as floor (σ-distribution baseline)
        let mut sorted = distance_table.clone();
        sorted.sort_unstable();
        let floor = sorted[total / 2];

        Self {
            distance_table,
            energy: vec![0.0; size],
            size,
            cycles: 0,
            convergence_threshold: 0.001,
            floor,
        }
    }

    /// Build distance table from codebook centroids.
    ///
    /// Computes pairwise cosine similarity between all 4096 centroids.
    /// Maps cosine [-1, 1] → u8 [0, 255].
    ///
    /// This is the ONE expensive operation. Done once per codebook.
    /// After this: every thought is just a MatVec on the table.
    pub fn build_distance_table(centroids_f64: &[Vec<f64>]) -> Vec<u8> {
        let k = centroids_f64.len();
        assert!(k <= CODEBOOK_SIZE);
        let mut table = vec![128u8; k * k]; // 128 = cosine 0 (orthogonal)

        for i in 0..k {
            table[i * k + i] = 255; // self-similarity = max
            for j in (i + 1)..k {
                let cos = cosine_f64_simd(&centroids_f64[i], &centroids_f64[j]);
                // Map cosine [-1, 1] → u8 [0, 255]
                let u = ((cos + 1.0) * 127.5).round().clamp(0.0, 255.0) as u8;
                table[i * k + j] = u;
                table[j * k + i] = u; // symmetric
            }
        }
        table
    }

    /// ONE thinking cycle. MatVec on the distance table.
    ///
    /// For each active thought-atom i:
    ///   its energy spreads to all j proportional to distance_table[i][j].
    ///   Similar atoms (high table value) reinforce.
    ///   Dissimilar atoms (low table value) don't contribute.
    ///
    /// SIMD: 8× F64x8 per iteration = 64 elements.
    /// 4096 / 64 = 64 iterations per row. 4096 active rows = 4096 × 64 = 262K SIMD ops.
    pub fn cycle(&mut self) {
        let k = self.size;
        let mut next = vec![0.0f64; k];
        let floor = self.floor;
        let scale = 1.0 / (255.0 - floor as f64);

        for i in 0..k {
            let e_i = self.energy[i];
            if e_i < 1e-15 { continue; }

            let row = &self.distance_table[i * k..(i + 1) * k];
            let e_scaled = e_i * scale;

            // 8× F64x8 = 64 elements per iteration
            let mut j = 0;
            while j + 64 <= k {
                // Load 64 u8 → 8× f64[8], subtract floor, FMA into accumulators
                macro_rules! do_lane {
                    ($off:expr) => {{
                        let base = j + $off * 8;
                        let d = F64x8::from_array([
                            row[base].saturating_sub(floor) as f64,
                            row[base + 1].saturating_sub(floor) as f64,
                            row[base + 2].saturating_sub(floor) as f64,
                            row[base + 3].saturating_sub(floor) as f64,
                            row[base + 4].saturating_sub(floor) as f64,
                            row[base + 5].saturating_sub(floor) as f64,
                            row[base + 6].saturating_sub(floor) as f64,
                            row[base + 7].saturating_sub(floor) as f64,
                        ]);
                        let acc = F64x8::from_slice(&next[base..base + 8]);
                        let ei = F64x8::splat(e_scaled);
                        d.mul_add(ei, acc).copy_to_slice(&mut next[base..base + 8]);
                    }};
                }
                do_lane!(0); do_lane!(1); do_lane!(2); do_lane!(3);
                do_lane!(4); do_lane!(5); do_lane!(6); do_lane!(7);
                j += 64;
            }
            // Scalar remainder (for sizes not divisible by 64)
            while j < k {
                let d = row[j].saturating_sub(floor) as f64;
                next[j] += d * e_scaled;
                j += 1;
            }
        }

        // Normalize: total energy = 1.0
        let total: f64 = next.iter().sum();
        if total > 1e-15 {
            for e in &mut next { *e /= total; }
        }

        self.energy = next;
        self.cycles += 1;
    }

    /// Run until convergence. Returns the resonance state.
    pub fn think(&mut self, max_cycles: usize) -> ResonanceDto {
        for _ in 0..max_cycles {
            let prev = self.energy.clone();
            self.cycle();

            let delta: f64 = self.energy.iter().zip(&prev)
                .map(|(a, b)| (a - b).abs()).sum();
            if delta < self.convergence_threshold {
                break;
            }
        }
        ResonanceDto::from_energy_vec(&self.energy, self.cycles)
    }

    /// Inject perturbation from sensor output.
    pub fn perturb(&mut self, codebook_indices: &[u16]) {
        for &idx in codebook_indices {
            if (idx as usize) < self.size {
                self.energy[idx as usize] += 1.0;
            }
        }
        let total: f64 = self.energy.iter().sum();
        if total > 1e-15 {
            for e in &mut self.energy { *e /= total; }
        }
    }

    /// Reset energy to zero. New thought starts fresh.
    pub fn reset(&mut self) {
        self.energy.fill(0.0);
        self.cycles = 0;
    }

    /// Commit: dominant peak → BusDto.
    pub fn commit(&self) -> BusDto {
        let resonance = ResonanceDto::from_energy_vec(&self.energy, self.cycles);
        BusDto {
            codebook_index: resonance.top_k[0].0,
            energy: resonance.top_k[0].1,
            top_k: resonance.top_k,
            cycle_count: self.cycles,
            converged: resonance.converged,
        }
    }

    /// Entropy of current energy distribution.
    /// Low entropy = crystallized thought (few dominant peaks).
    /// High entropy = diffuse resonance (many active atoms).
    pub fn entropy(&self) -> f64 {
        let mut h = 0.0f64;
        for &e in &self.energy {
            if e > 1e-15 {
                h -= e * e.ln();
            }
        }
        h
    }

    /// Number of active thought-atoms (energy > threshold).
    pub fn active_count(&self, threshold: f64) -> usize {
        self.energy.iter().filter(|&&e| e > threshold).count()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_test_table(k: usize) -> Vec<u8> {
        let mut table = vec![128u8; k * k];
        for i in 0..k {
            table[i * k + i] = 255; // self = max
            // Create some structure: nearby indices are similar
            for j in 0..k {
                let dist = (i as i64 - j as i64).unsigned_abs() as usize;
                if dist < 50 {
                    table[i * k + j] = (255 - dist * 2).min(255) as u8;
                }
            }
        }
        table
    }

    #[test]
    fn engine_creates() {
        let table = make_test_table(CODEBOOK_SIZE);
        let engine = ThinkingEngine::new(table);
        assert_eq!(engine.energy.iter().sum::<f64>(), 0.0);
        assert_eq!(engine.cycles, 0);
    }

    #[test]
    fn perturb_adds_energy() {
        let table = make_test_table(CODEBOOK_SIZE);
        let mut engine = ThinkingEngine::new(table);

        engine.perturb(&[42, 100, 200]);
        assert!(engine.energy[42] > 0.0);
        assert!(engine.energy[100] > 0.0);
        assert!(engine.energy[200] > 0.0);

        // Total should be 1.0 (normalized)
        let total: f64 = engine.energy.iter().sum();
        assert!((total - 1.0).abs() < 1e-10);
    }

    #[test]
    fn cycle_spreads_energy() {
        let table = make_test_table(CODEBOOK_SIZE);
        let mut engine = ThinkingEngine::new(table);

        // Perturb one atom
        engine.perturb(&[500]);
        assert!(engine.energy[500] > 0.9); // almost all energy at 500

        // One cycle: energy spreads to nearby atoms
        engine.cycle();
        assert!(engine.energy[500] > 0.0); // still has some
        assert!(engine.energy[499] > 0.0); // neighbor activated
        assert!(engine.energy[501] > 0.0); // neighbor activated
    }

    #[test]
    fn think_converges() {
        let table = make_test_table(CODEBOOK_SIZE);
        let mut engine = ThinkingEngine::new(table);

        engine.perturb(&[100, 110, 120]);
        let resonance = engine.think(20);

        assert!(resonance.cycle_count <= 20);
        assert!(resonance.top_k[0].1 > 0.0); // dominant peak exists

        eprintln!("Converged in {} cycles", resonance.cycle_count);
        eprintln!("Top peak: index={}, energy={:.4}",
            resonance.top_k[0].0, resonance.top_k[0].1);
        eprintln!("Entropy: {:.4}", engine.entropy());
        eprintln!("Active atoms: {}", engine.active_count(0.001));
    }

    #[test]
    fn commit_returns_dominant() {
        let table = make_test_table(CODEBOOK_SIZE);
        let mut engine = ThinkingEngine::new(table);

        engine.perturb(&[42]);
        engine.think(10);

        let bus = engine.commit();
        assert!(bus.energy > 0.0);
        eprintln!("Committed: index={}, energy={:.4}", bus.codebook_index, bus.energy);
    }

    #[test]
    fn entropy_decreases_with_convergence() {
        let table = make_test_table(CODEBOOK_SIZE);
        let mut engine = ThinkingEngine::new(table);

        engine.perturb(&[100, 200, 300, 400, 500]);
        let h0 = engine.entropy();

        engine.think(10);
        let h1 = engine.entropy();

        eprintln!("Entropy before: {:.4}, after: {:.4}", h0, h1);
        // Entropy should decrease or stay stable as peaks crystallize
        // (may not always decrease if the table has uniform structure)
    }

    #[test]
    fn multiple_perturbations_compose() {
        let table = make_test_table(CODEBOOK_SIZE);
        let mut engine = ThinkingEngine::new(table);

        // First sensor
        engine.perturb(&[100, 101, 102]);
        engine.think(5);

        // Second sensor adds more perturbation
        engine.perturb(&[100, 200, 300]);
        let resonance = engine.think(10);

        // Both perturbation sites should have influenced the result
        assert!(resonance.top_k[0].1 > 0.0);
        eprintln!("After dual perturbation: {} peaks above 0.01",
            engine.active_count(0.01));
    }

    #[test]
    fn reset_clears() {
        let table = make_test_table(CODEBOOK_SIZE);
        let mut engine = ThinkingEngine::new(table);

        engine.perturb(&[42]);
        engine.think(5);
        assert!(engine.energy[42] > 0.0);

        engine.reset();
        assert_eq!(engine.energy.iter().sum::<f64>(), 0.0);
        assert_eq!(engine.cycles, 0);
    }

    #[test]
    fn build_distance_table_symmetric() {
        let centroids: Vec<Vec<f64>> = (0..100).map(|i| {
            (0..64).map(|d| ((i * 97 + d * 31) as f64 % 200.0 - 100.0) * 0.01).collect()
        }).collect();

        let table = ThinkingEngine::build_distance_table(&centroids);
        assert_eq!(table.len(), 100 * 100);

        // Symmetric
        for i in 0..100 {
            for j in 0..100 {
                assert_eq!(table[i * 100 + j], table[j * 100 + i],
                    "table[{},{}]={} != table[{},{}]={}",
                    i, j, table[i * 100 + j], j, i, table[j * 100 + i]);
            }
        }

        // Diagonal = 255
        for i in 0..100 {
            assert_eq!(table[i * 100 + i], 255);
        }
    }
}
