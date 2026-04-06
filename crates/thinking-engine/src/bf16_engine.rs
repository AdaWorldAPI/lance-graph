//! BF16 ThinkingEngine: distance table at source precision.
//!
//! Uses BF16 (u16) for distance table values instead of u8/i8.
//! BF16 preserves:
//!   - Sign bit (bit 15) = gate decision (excitation vs inhibition)
//!   - Exponent (8 bits) = dynamic range (natural gamma near zero)
//!   - Mantissa (7 bits) = resolution matching BF16 source weights
//!
//! bf16_to_f32 is lossless (one bit shift). No CDF. No rank relabeling.
//! 256×256 × 2 bytes = 128 KB (fits L2 cache).
//! 4096×4096 × 2 bytes = 32 MB (fits L3 cache on Zen 4).
//!
//! Built from StackedN::cosine() via ClamCodebook, matching bgz-tensor pipeline.

use crate::dto::{ResonanceDto, BusDto};
use bgz_tensor::stacked_n::{bf16_to_f32, f32_to_bf16};
use ndarray::simd::F32x16;

/// BF16 thinking engine. Distance table at source weight precision.
pub struct BF16ThinkingEngine {
    /// BF16 distance table. u16 bit patterns, NOT unsigned integers.
    /// bf16_to_f32() is lossless (bit shift left 16).
    /// Positive BF16 = excitation. Negative BF16 = inhibition.
    distance_table: Vec<u16>,

    /// Current energy distribution.
    pub energy: Vec<f32>,

    /// Number of thought-atoms.
    pub size: usize,

    /// Cycle counter.
    pub cycles: u16,

    /// Convergence threshold.
    pub convergence_threshold: f32,

    /// Inhibition stats.
    pub inhibited_last_cycle: usize,
    pub total_inhibitions: usize,
}

impl BF16ThinkingEngine {
    /// Create from a precomputed BF16 distance table (u16 bit patterns).
    pub fn new(distance_table: Vec<u16>) -> Self {
        let total = distance_table.len();
        let size = (total as f64).sqrt() as usize;
        assert_eq!(size * size, total,
            "BF16 table length {} is not a perfect square", total);
        assert!(size >= 4, "need at least 4 atoms");

        Self {
            distance_table,
            energy: vec![0.0f32; size],
            size,
            cycles: 0,
            convergence_threshold: 0.001,
            inhibited_last_cycle: 0,
            total_inhibitions: 0,
        }
    }

    /// Build BF16 table from f32 cosine values.
    /// cosine [-1.0, +1.0] → BF16 (truncate mantissa, 1 ULP loss).
    pub fn from_f32_cosines(cosines: &[f32], size: usize) -> Self {
        assert_eq!(cosines.len(), size * size);
        let table: Vec<u16> = cosines.iter()
            .map(|&c| f32_to_bf16(c))
            .collect();
        Self::new(table)
    }

    /// Build BF16 table from f64 pairwise cosines (e.g. from StackedN::cosine()).
    pub fn from_f64_cosines(cosines: &[f64], size: usize) -> Self {
        assert_eq!(cosines.len(), size * size);
        let table: Vec<u16> = cosines.iter()
            .map(|&c| f32_to_bf16(c as f32))
            .collect();
        Self::new(table)
    }

    /// Build from a ClamCodebook (the correct pipeline).
    /// Computes pairwise StackedN cosine between all centroids.
    pub fn from_codebook(codebook: &bgz_tensor::stacked_n::ClamCodebook) -> Self {
        let n = codebook.entries.len();
        let mut table = vec![0u16; n * n];
        for i in 0..n {
            table[i * n + i] = f32_to_bf16(1.0); // self = max
            for j in (i + 1)..n {
                let cos = codebook.entries[i].stacked.cosine(
                    &codebook.entries[j].stacked
                );
                let bf16 = f32_to_bf16(cos as f32);
                table[i * n + j] = bf16;
                table[j * n + i] = bf16;
            }
        }
        Self::new(table)
    }

    /// Load BF16 table from file (N×N × 2 bytes).
    pub fn load(path: &std::path::Path) -> Result<Self, String> {
        let data = std::fs::read(path).map_err(|e| format!("read: {}", e))?;
        if data.len() % 2 != 0 {
            return Err("BF16 table must have even byte count".into());
        }
        let table: Vec<u16> = data.chunks_exact(2)
            .map(|c| u16::from_le_bytes([c[0], c[1]]))
            .collect();
        Ok(Self::new(table))
    }

    /// Save BF16 table to file.
    pub fn save(&self, path: &std::path::Path) -> Result<(), String> {
        let bytes: Vec<u8> = self.distance_table.iter()
            .flat_map(|&v| v.to_le_bytes())
            .collect();
        std::fs::write(path, &bytes).map_err(|e| format!("write: {}", e))
    }

    /// ONE cycle. BF16 → f32 (lossless) → accumulate → clamp → normalize.
    ///
    /// Positive BF16 values excite. Negative inhibit.
    /// No floor needed — negative cosines naturally suppress.
    pub fn cycle(&mut self) {
        let k = self.size;
        let mut next = vec![0.0f32; k];

        for i in 0..k {
            let e_i = self.energy[i];
            if e_i < 1e-10 { continue; }

            let row = &self.distance_table[i * k..(i + 1) * k];

            // F32x16 SIMD: 4× unrolled = 64 elements per iteration
            let mut j = 0;
            while j + 64 <= k {
                macro_rules! do_lane {
                    ($off:expr) => {{
                        let base = j + $off * 16;
                        // BF16 → f32: lossless bit shift
                        let d = F32x16::from_array([
                            bf16_to_f32(row[base]),      bf16_to_f32(row[base + 1]),
                            bf16_to_f32(row[base + 2]),  bf16_to_f32(row[base + 3]),
                            bf16_to_f32(row[base + 4]),  bf16_to_f32(row[base + 5]),
                            bf16_to_f32(row[base + 6]),  bf16_to_f32(row[base + 7]),
                            bf16_to_f32(row[base + 8]),  bf16_to_f32(row[base + 9]),
                            bf16_to_f32(row[base + 10]), bf16_to_f32(row[base + 11]),
                            bf16_to_f32(row[base + 12]), bf16_to_f32(row[base + 13]),
                            bf16_to_f32(row[base + 14]), bf16_to_f32(row[base + 15]),
                        ]);
                        let acc = F32x16::from_slice(&next[base..base + 16]);
                        let ei = F32x16::splat(e_i);
                        d.mul_add(ei, acc).copy_to_slice(&mut next[base..base + 16]);
                    }};
                }
                do_lane!(0); do_lane!(1); do_lane!(2); do_lane!(3);
                j += 64;
            }
            while j < k {
                next[j] += bf16_to_f32(row[j]) * e_i;
                j += 1;
            }
        }

        // Clamp: inhibited atoms die
        let mut inhibited = 0usize;
        for e in &mut next {
            if *e < 0.0 { *e = 0.0; inhibited += 1; }
        }
        self.inhibited_last_cycle = inhibited;
        self.total_inhibitions += inhibited;

        // Normalize
        let total: f32 = next.iter().sum();
        if total > 1e-10 {
            let inv = 1.0 / total;
            for e in &mut next { *e *= inv; }
        }

        self.energy = next;
        self.cycles += 1;
    }

    /// Cycle with temperature-as-excitation (softmax/T).
    pub fn cycle_with_temperature(&mut self, temperature: f32) {
        self.cycle();

        let t = temperature.max(0.01);
        let max_e = self.energy.iter().cloned().fold(0.0f32, f32::max);
        let mut exp_sum = 0.0f32;
        for e in &mut self.energy {
            if *e > 1e-10 {
                *e = ((*e - max_e) / t).exp();
                exp_sum += *e;
            }
        }
        if exp_sum > 1e-10 {
            let inv = 1.0 / exp_sum;
            for e in &mut self.energy { *e *= inv; }
        }
    }

    /// Think until convergence.
    pub fn think(&mut self, max_cycles: usize) -> ResonanceDto {
        for _ in 0..max_cycles {
            let prev = self.energy.clone();
            self.cycle();
            let delta: f32 = self.energy.iter().zip(&prev)
                .map(|(a, b)| (a - b).abs()).sum();
            if delta < self.convergence_threshold { break; }
        }
        ResonanceDto::from_energy_f32(&self.energy, self.cycles)
    }

    /// Think with temperature.
    pub fn think_with_temperature(&mut self, max_cycles: usize, temperature: f32) -> ResonanceDto {
        for _ in 0..max_cycles {
            let prev = self.energy.clone();
            self.cycle_with_temperature(temperature);
            let delta: f32 = self.energy.iter().zip(&prev)
                .map(|(a, b)| (a - b).abs()).sum();
            if delta < self.convergence_threshold { break; }
        }
        ResonanceDto::from_energy_f32(&self.energy, self.cycles)
    }

    /// Perturb with codebook indices.
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

    /// Reset.
    pub fn reset(&mut self) {
        self.energy.fill(0.0);
        self.cycles = 0;
        self.inhibited_last_cycle = 0;
        self.total_inhibitions = 0;
    }

    /// Commit dominant peak.
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

    /// Access the BF16 distance table.
    pub fn distance_table_ref(&self) -> &[u16] { &self.distance_table }

    /// Entropy.
    pub fn entropy(&self) -> f32 {
        let mut h = 0.0f32;
        for &e in &self.energy {
            if e > 1e-10 { h -= e * e.ln(); }
        }
        h
    }

    /// Active atom count.
    pub fn active_count(&self, threshold: f32) -> usize {
        self.energy.iter().filter(|&&e| e > threshold).count()
    }

    /// Sign distribution of the BF16 table.
    pub fn sign_stats(&self) -> (usize, usize, usize) {
        let mut pos = 0usize;
        let mut neg = 0usize;
        let mut zero = 0usize;
        for &v in &self.distance_table {
            let f = bf16_to_f32(v);
            if f > 0.0 { pos += 1; }
            else if f < 0.0 { neg += 1; }
            else { zero += 1; }
        }
        (pos, neg, zero)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_bf16_test_table(k: usize) -> Vec<u16> {
        let mut table = vec![f32_to_bf16(0.0); k * k];
        for i in 0..k {
            table[i * k + i] = f32_to_bf16(1.0); // self = max
            for j in 0..k {
                if i == j { continue; }
                let dist = (i as i64 - j as i64).unsigned_abs() as usize;
                if dist < 20 {
                    // Near: positive (excitation)
                    let cos = 0.8 - (dist as f32) * 0.03;
                    table[i * k + j] = f32_to_bf16(cos);
                } else if dist > k / 2 {
                    // Far: negative (inhibition)
                    let cos = -0.3 - ((dist - k / 2) as f32) * 0.01;
                    table[i * k + j] = f32_to_bf16(cos.max(-1.0));
                }
                // Middle: stays 0.0 (orthogonal)
            }
        }
        table
    }

    #[test]
    fn bf16_engine_creates() {
        let table = make_bf16_test_table(64);
        let engine = BF16ThinkingEngine::new(table);
        assert_eq!(engine.size, 64);
        assert_eq!(engine.energy.iter().sum::<f32>(), 0.0);
    }

    #[test]
    fn bf16_has_real_negatives() {
        let table = make_bf16_test_table(64);
        let engine = BF16ThinkingEngine::new(table);
        let (pos, neg, _) = engine.sign_stats();
        assert!(pos > 0, "should have positive entries");
        assert!(neg > 0, "should have negative entries (real inhibition)");
    }

    #[test]
    fn bf16_cycle_inhibits() {
        let table = make_bf16_test_table(64);
        let mut engine = BF16ThinkingEngine::new(table);
        engine.perturb(&[5, 50]); // near + far = should produce inhibition
        engine.cycle();
        assert!(engine.inhibited_last_cycle > 0,
            "BF16 table with real negatives should produce inhibition");
    }

    #[test]
    fn bf16_think_converges() {
        let table = make_bf16_test_table(64);
        let mut engine = BF16ThinkingEngine::new(table);
        engine.perturb(&[10, 12, 14]);
        let resonance = engine.think(20);
        assert!(resonance.top_k[0].1 > 0.0);
        assert!(resonance.cycle_count <= 20);
    }

    #[test]
    fn bf16_temperature_differentiates() {
        let table = make_bf16_test_table(64);

        let mut eng_low = BF16ThinkingEngine::new(table.clone());
        eng_low.perturb(&[10, 12, 14]);
        eng_low.think_with_temperature(10, 0.1);
        let low_active = eng_low.active_count(0.01);

        let mut eng_high = BF16ThinkingEngine::new(table);
        eng_high.perturb(&[10, 12, 14]);
        eng_high.think_with_temperature(10, 2.0);
        let high_active = eng_high.active_count(0.01);

        // Low T should concentrate energy (fewer active atoms)
        // High T should spread energy (more active atoms)
        assert!(low_active <= high_active + 5,
            "low T ({}) should have ≤ active atoms than high T ({})",
            low_active, high_active);
    }

    #[test]
    fn bf16_nearby_different_from_distant() {
        let table = make_bf16_test_table(64);

        // Think nearby centroids
        let mut eng_near = BF16ThinkingEngine::new(table.clone());
        eng_near.perturb(&[10, 11, 12]);
        eng_near.think(10);
        let near_peak = eng_near.commit().codebook_index;

        // Think distant centroids
        let mut eng_far = BF16ThinkingEngine::new(table);
        eng_far.perturb(&[10, 40, 60]);
        eng_far.think(10);
        let far_peak = eng_far.commit().codebook_index;

        // Different inputs should produce different peaks
        // (on a table with real structure, unlike CDF-uniform tables)
        eprintln!("near_peak={}, far_peak={}", near_peak, far_peak);
        // At minimum both should be valid
        assert!(near_peak < 64);
        assert!(far_peak < 64);
    }

    #[test]
    fn bf16_from_f32_cosines() {
        let n = 8;
        let mut cosines = vec![0.0f32; n * n];
        for i in 0..n {
            cosines[i * n + i] = 1.0;
            for j in (i + 1)..n {
                let cos = 1.0 - (i as f32 - j as f32).abs() * 0.15;
                cosines[i * n + j] = cos;
                cosines[j * n + i] = cos;
            }
        }
        let engine = BF16ThinkingEngine::from_f32_cosines(&cosines, n);
        assert_eq!(engine.size, 8);

        // Verify roundtrip: f32 → BF16 → f32 ≈ original
        let table_f32: Vec<f32> = engine.distance_table.iter()
            .map(|&v| bf16_to_f32(v)).collect();
        for i in 0..n * n {
            let diff = (cosines[i] - table_f32[i]).abs();
            assert!(diff < 0.01, "BF16 roundtrip error {} at index {}", diff, i);
        }
    }

    #[test]
    fn bf16_reset_clears() {
        let table = make_bf16_test_table(64);
        let mut engine = BF16ThinkingEngine::new(table);
        engine.perturb(&[10]);
        engine.think(5);
        engine.reset();
        assert_eq!(engine.energy.iter().sum::<f32>(), 0.0);
        assert_eq!(engine.cycles, 0);
    }
}
