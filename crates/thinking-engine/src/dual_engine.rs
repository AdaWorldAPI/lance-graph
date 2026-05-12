//! DualEngine: compare two table encodings on the same input.
//!
//! The real comparison is u8 CDF vs BF16 direct (or any two BuiltEngine types).
//! Same input, same perturbation, different encoding → measure disagreement.

use crate::builder::BuiltEngine;
use crate::dto::ResonanceDto;

/// Results from running both engines on the same input.
pub struct DualResult {
    /// Top-8 peaks from engine A.
    pub peaks_a: [(u16, f32); 8],
    /// Top-8 peaks from engine B.
    pub peaks_b: [(u16, f32); 8],
    /// Fraction of top-8 peaks shared (0.0 = none, 1.0 = identical).
    pub agreement: f32,
    /// Peaks in A but not B.
    pub unique_a: Vec<u16>,
    /// Peaks in B but not A.
    pub unique_b: Vec<u16>,
    /// Convergence cycles: A.
    pub convergence_a: u16,
    /// Convergence cycles: B.
    pub convergence_b: u16,
    /// Entropy: A.
    pub entropy_a: f32,
    /// Entropy: B.
    pub entropy_b: f32,
    /// Labels for the two engines.
    pub label_a: String,
    pub label_b: String,
}

impl DualResult {
    pub fn summary(&self) -> String {
        let shared = (self.agreement * 8.0).round() as usize;
        format!(
            "{} vs {}: {:.0}% agreement ({}/8 shared)\n\
             Convergence: {}={} {}={} cycles\n\
             Entropy: {}={:.3} {}={:.3}\n\
             {}-unique: {:?}\n\
             {}-unique: {:?}",
            self.label_a, self.label_b, self.agreement * 100.0, shared,
            self.label_a, self.convergence_a, self.label_b, self.convergence_b,
            self.label_a, self.entropy_a, self.label_b, self.entropy_b,
            self.label_a, self.unique_a,
            self.label_b, self.unique_b,
        )
    }
}

/// Compare two engines of any type on the same input.
pub struct DualEngine {
    pub engine_a: BuiltEngine,
    pub engine_b: BuiltEngine,
    pub label_a: String,
    pub label_b: String,
}

impl DualEngine {
    /// Create from any two BuiltEngines.
    pub fn new(
        label_a: &str, engine_a: BuiltEngine,
        label_b: &str, engine_b: BuiltEngine,
    ) -> Self {
        Self {
            engine_a, engine_b,
            label_a: label_a.into(), label_b: label_b.into(),
        }
    }

    /// Compare u8 CDF vs BF16 from the same source table.
    pub fn u8_vs_bf16(table: Vec<u8>) -> Self {
        let bf16_cosines: Vec<f32> = table.iter()
            .map(|&v| (v as f32 - 128.0) / 127.0)
            .collect();
        let size = (table.len() as f64).sqrt() as usize;
        Self {
            engine_a: BuiltEngine::Unsigned(crate::engine::ThinkingEngine::new(table)),
            engine_b: BuiltEngine::BF16(
                crate::bf16_engine::BF16ThinkingEngine::from_f32_cosines(&bf16_cosines, size)
            ),
            label_a: "u8-CDF".into(),
            label_b: "BF16".into(),
        }
    }

    /// Perturb both identically.
    pub fn perturb_both(&mut self, indices: &[u16]) {
        self.engine_a.perturb(indices);
        self.engine_b.perturb(indices);
    }

    /// Think both and compare.
    pub fn think_both(&mut self, max_cycles: usize) -> DualResult {
        self.engine_a.think(max_cycles);
        self.engine_b.think(max_cycles);

        let res_a = ResonanceDto::from_energy_f32(self.engine_a.energy(), self.engine_a.cycles());
        let res_b = ResonanceDto::from_energy_f32(self.engine_b.energy(), self.engine_b.cycles());

        let a_indices: Vec<u16> = res_a.top_k.iter()
            .filter(|&&(_, e)| e > 1e-10)
            .map(|&(idx, _)| idx).collect();
        let b_indices: Vec<u16> = res_b.top_k.iter()
            .filter(|&&(_, e)| e > 1e-10)
            .map(|&(idx, _)| idx).collect();

        let overlap = a_indices.iter().filter(|p| b_indices.contains(p)).count();
        let max_len = a_indices.len().max(b_indices.len()).max(1);

        DualResult {
            peaks_a: res_a.top_k,
            peaks_b: res_b.top_k,
            agreement: overlap as f32 / max_len as f32,
            unique_a: a_indices.iter().filter(|p| !b_indices.contains(p)).cloned().collect(),
            unique_b: b_indices.iter().filter(|p| !a_indices.contains(p)).cloned().collect(),
            convergence_a: res_a.cycle_count,
            convergence_b: res_b.cycle_count,
            entropy_a: {
                let e = self.engine_a.energy();
                let mut h = 0.0f32;
                for &v in e { if v > 1e-10 { h -= v * v.ln(); } }
                h
            },
            entropy_b: {
                let e = self.engine_b.energy();
                let mut h = 0.0f32;
                for &v in e { if v > 1e-10 { h -= v * v.ln(); } }
                h
            },
            label_a: self.label_a.clone(),
            label_b: self.label_b.clone(),
        }
    }

    /// Reset both.
    pub fn reset_both(&mut self) {
        self.engine_a.reset();
        self.engine_b.reset();
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
    fn dual_u8_vs_bf16() {
        let table = make_test_table(256);
        let mut dual = DualEngine::u8_vs_bf16(table);

        dual.perturb_both(&[50, 55, 60]);
        let result = dual.think_both(20);

        assert!(result.peaks_a[0].1 > 0.0);
        assert!(result.peaks_b[0].1 > 0.0);
        assert!(result.agreement >= 0.0 && result.agreement <= 1.0);
        assert_eq!(result.label_a, "u8-CDF");
        assert_eq!(result.label_b, "BF16");
    }

    #[test]
    fn dual_custom_engines() {
        let table = make_test_table(64);
        let dual = DualEngine::new(
            "unsigned", BuiltEngine::Unsigned(crate::engine::ThinkingEngine::new(table.clone())),
            "signed", BuiltEngine::Signed(crate::signed_engine::SignedThinkingEngine::from_unsigned(&table)),
        );
        assert_eq!(dual.label_a, "unsigned");
        assert_eq!(dual.label_b, "signed");
    }

    #[test]
    fn dual_reset() {
        let table = make_test_table(256);
        let mut dual = DualEngine::u8_vs_bf16(table);
        dual.perturb_both(&[42]);
        dual.think_both(5);
        dual.reset_both();
        assert_eq!(dual.engine_a.energy().iter().sum::<f32>(), 0.0);
        assert_eq!(dual.engine_b.energy().iter().sum::<f32>(), 0.0);
    }
}
