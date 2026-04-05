//! DualEngine: run unsigned (u8) and signed (i8) engines in parallel.
//!
//! Same input, same perturbation, different table encoding.
//! Compare: peak agreement, convergence speed, inhibition count.
//! The disagreement IS the experiment result.
//!
//! ```text
//! If agreement > 90%:  signed adds nothing. Keep unsigned + SiLU-ONNX.
//! If agreement 50-90%: signed finds different peaks. Run both (Path C).
//! If agreement < 50%:  fundamentally different topology. Investigate.
//! ```

use crate::engine::ThinkingEngine;
use crate::signed_engine::SignedThinkingEngine;

/// Results from running both engines on the same input.
pub struct DualResult {
    /// Top-8 peaks from unsigned engine.
    pub unsigned_peaks: [(u16, f32); 8],
    /// Top-8 peaks from signed engine.
    pub signed_peaks: [(u16, f32); 8],
    /// Fraction of top-8 peaks that appear in both (0.0 = none, 1.0 = identical).
    pub agreement: f32,
    /// Peak indices found by signed but not unsigned — the inhibition effect.
    pub signed_unique: Vec<u16>,
    /// Peak indices found by unsigned but not signed — killed by inhibition.
    pub unsigned_unique: Vec<u16>,
    /// Atoms inhibited (clamped to 0) in the last signed cycle.
    pub inhibition_count: usize,
    /// Total inhibitions across all signed cycles.
    pub total_inhibitions: usize,
    /// Convergence cycles: unsigned.
    pub convergence_unsigned: u16,
    /// Convergence cycles: signed.
    pub convergence_signed: u16,
    /// Shannon entropy after convergence: unsigned.
    pub entropy_unsigned: f32,
    /// Shannon entropy after convergence: signed.
    pub entropy_signed: f32,
    /// E/I ratio of the signed table.
    pub ei_ratio: f32,
}

impl DualResult {
    /// Summary string for quick comparison.
    pub fn summary(&self) -> String {
        let shared = (self.agreement * 8.0).round() as usize;
        format!(
            "Agreement: {:.0}% ({}/8 shared)\n\
             Convergence: unsigned={} signed={} cycles\n\
             Entropy: unsigned={:.3} signed={:.3}\n\
             Inhibition: {} last cycle, {} total\n\
             E/I ratio: {:.1}%\n\
             Signed-unique peaks: {:?}\n\
             Unsigned-unique (killed by inhibition): {:?}",
            self.agreement * 100.0,
            shared,
            self.convergence_unsigned,
            self.convergence_signed,
            self.entropy_unsigned,
            self.entropy_signed,
            self.inhibition_count,
            self.total_inhibitions,
            self.ei_ratio * 100.0,
            self.signed_unique,
            self.unsigned_unique,
        )
    }
}

/// Dual engine: unsigned + signed running on the same distance data.
pub struct DualEngine {
    pub unsigned: ThinkingEngine,
    pub signed: SignedThinkingEngine,
}

impl DualEngine {
    /// Create from an existing u8 table. Signed table derived by subtracting 128.
    pub fn from_unsigned_table(table: Vec<u8>) -> Self {
        let signed = SignedThinkingEngine::from_unsigned(&table);
        let unsigned = ThinkingEngine::new(table);
        Self { unsigned, signed }
    }

    /// Create from separate u8 and i8 tables.
    pub fn from_tables(unsigned_table: Vec<u8>, signed_table: Vec<i8>) -> Self {
        Self {
            unsigned: ThinkingEngine::new(unsigned_table),
            signed: SignedThinkingEngine::new(signed_table),
        }
    }

    /// Perturb both engines identically.
    pub fn perturb_both(&mut self, codebook_indices: &[u16]) {
        self.unsigned.perturb(codebook_indices);
        self.signed.perturb(codebook_indices);
    }

    /// Run both engines to convergence and compare results.
    pub fn think_both(&mut self, max_cycles: usize) -> DualResult {
        let u_res = self.unsigned.think(max_cycles);
        let s_res = self.signed.think(max_cycles);

        // Extract top-8 indices for comparison (filter zero-energy entries)
        let u_indices: Vec<u16> = u_res.top_k.iter()
            .filter(|&&(_, e)| e > 1e-10)
            .map(|&(idx, _)| idx)
            .collect();
        let s_indices: Vec<u16> = s_res.top_k.iter()
            .filter(|&&(_, e)| e > 1e-10)
            .map(|&(idx, _)| idx)
            .collect();

        let overlap = u_indices.iter().filter(|p| s_indices.contains(p)).count();
        let max_len = u_indices.len().max(s_indices.len()).max(1);
        let agreement = overlap as f32 / max_len as f32;

        let signed_unique = s_indices.iter()
            .filter(|p| !u_indices.contains(p)).cloned().collect();
        let unsigned_unique = u_indices.iter()
            .filter(|p| !s_indices.contains(p)).cloned().collect();

        DualResult {
            unsigned_peaks: u_res.top_k,
            signed_peaks: s_res.top_k,
            agreement,
            signed_unique,
            unsigned_unique,
            inhibition_count: self.signed.inhibited_last_cycle,
            total_inhibitions: self.signed.total_inhibitions,
            convergence_unsigned: u_res.cycle_count,
            convergence_signed: s_res.cycle_count,
            entropy_unsigned: self.unsigned.entropy(),
            entropy_signed: self.signed.entropy(),
            ei_ratio: self.signed.ei_ratio,
        }
    }

    /// Reset both engines.
    pub fn reset_both(&mut self) {
        self.unsigned.reset();
        self.signed.reset();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_test_table(k: usize) -> Vec<u8> {
        let mut table = vec![128u8; k * k];
        for i in 0..k {
            table[i * k + i] = 255;
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
    fn dual_engine_creates() {
        let table = make_test_table(256);
        let dual = DualEngine::from_unsigned_table(table);
        assert_eq!(dual.unsigned.size, 256);
        assert_eq!(dual.signed.size, 256);
    }

    #[test]
    fn dual_perturb_symmetric() {
        let table = make_test_table(256);
        let mut dual = DualEngine::from_unsigned_table(table);
        dual.perturb_both(&[42, 100]);

        assert!(dual.unsigned.energy[42] > 0.0);
        assert!(dual.signed.energy[42] > 0.0);
        assert!(dual.unsigned.energy[100] > 0.0);
        assert!(dual.signed.energy[100] > 0.0);
    }

    #[test]
    fn dual_think_produces_results() {
        let table = make_test_table(256);
        let mut dual = DualEngine::from_unsigned_table(table);

        dual.perturb_both(&[50, 55, 60]);
        let result = dual.think_both(20);

        assert!(result.unsigned_peaks[0].1 > 0.0);
        assert!(result.signed_peaks[0].1 > 0.0);
        assert!(result.agreement >= 0.0 && result.agreement <= 1.0);
    }

    #[test]
    fn dual_reset_clears_both() {
        let table = make_test_table(256);
        let mut dual = DualEngine::from_unsigned_table(table);

        dual.perturb_both(&[42]);
        dual.think_both(5);
        dual.reset_both();

        assert_eq!(dual.unsigned.energy.iter().sum::<f32>(), 0.0);
        assert_eq!(dual.signed.energy.iter().sum::<f32>(), 0.0);
    }
}
