//! Multi-model composition: multiple lenses → one thought.
//!
//! Each lens runs its own engine (u8, i8, or BF16). Results are composed via
//! energy superposition: weighted sum of peak energies across lenses.
//!
//! Uses BuiltEngine from builder.rs — supports all three table types.
//!
//! ```text
//! Jina v3 (BF16)    →  engine_1  →  peaks_1
//! Reranker (BF16)    →  engine_2  →  peaks_2   →  superposition  →  CompositeResult
//! BGE-M3 (u8 legacy) →  engine_3  →  peaks_3
//! ```

use crate::builder::BuiltEngine;
use crate::dto::ResonanceDto;

/// Result of multi-model composition.
pub struct CompositeResult {
    /// Per-lens results: (lens_name, top_k peaks, cycle_count).
    pub per_lens: Vec<(String, [(u16, f32); 8], u16)>,
    /// Superposed peaks: atoms that appear across multiple lenses.
    /// Sorted by composite score (sum of energies across lenses).
    pub superposed: Vec<(u16, f32, usize)>,
    /// Agreement matrix: pairwise peak overlap between lenses.
    pub agreement: Vec<(String, String, f32)>,
}

impl CompositeResult {
    pub fn summary(&self) -> String {
        let mut s = format!("Composite: {} lenses\n", self.per_lens.len());
        for (name, peaks, cycles) in &self.per_lens {
            let top3: Vec<String> = peaks.iter().take(3)
                .map(|(idx, e)| format!("{}:{:.3}", idx, e))
                .collect();
            s.push_str(&format!("  {} ({} cycles): [{}]\n",
                name, cycles, top3.join(", ")));
        }
        s.push_str("Superposed top-5:\n");
        for (atom, energy, count) in self.superposed.iter().take(5) {
            s.push_str(&format!("  atom {} = {:.3} ({} lenses)\n", atom, energy, count));
        }
        for (a, b, agree) in &self.agreement {
            s.push_str(&format!("  {} × {} = {:.0}%\n", a, b, agree * 100.0));
        }
        s
    }
}

/// Multi-model composition engine using BuiltEngine (supports u8/i8/BF16).
pub struct CompositeEngine {
    lenses: Vec<(String, BuiltEngine)>,
}

impl CompositeEngine {
    pub fn new() -> Self {
        Self { lenses: Vec::new() }
    }

    /// Add a pre-built engine as a lens.
    pub fn add_engine(&mut self, name: &str, engine: BuiltEngine) {
        self.lenses.push((name.to_string(), engine));
    }

    /// Add a u8 table lens (legacy compatibility).
    pub fn add_u8_lens(&mut self, name: &str, table: Vec<u8>) {
        self.lenses.push((
            name.to_string(),
            BuiltEngine::Unsigned(crate::engine::ThinkingEngine::new(table)),
        ));
    }

    /// Add a BF16 table lens.
    pub fn add_bf16_lens(&mut self, name: &str, table: Vec<u16>) {
        self.lenses.push((
            name.to_string(),
            BuiltEngine::BF16(crate::bf16_engine::BF16ThinkingEngine::new(table)),
        ));
    }

    /// Perturb a specific lens by name.
    pub fn perturb_lens(&mut self, name: &str, indices: &[u16]) {
        for (n, engine) in &mut self.lenses {
            if n == name {
                engine.perturb(indices);
                return;
            }
        }
    }

    /// Perturb all lenses with the same indices.
    pub fn perturb_all(&mut self, indices: &[u16]) {
        for (_, engine) in &mut self.lenses {
            engine.perturb(indices);
        }
    }

    /// Think all lenses and compose results.
    pub fn think_all(&mut self, max_cycles: usize) -> CompositeResult {
        let mut per_lens = Vec::new();
        let mut all_peaks: std::collections::HashMap<u16, (f32, usize)> =
            std::collections::HashMap::new();

        for (name, engine) in &mut self.lenses {
            engine.think(max_cycles);
            let res = ResonanceDto::from_energy_f32(engine.energy(), engine.cycles());

            for &(idx, energy) in &res.top_k {
                if energy > 1e-10 {
                    let entry = all_peaks.entry(idx).or_insert((0.0, 0));
                    entry.0 += energy;
                    entry.1 += 1;
                }
            }

            per_lens.push((name.clone(), res.top_k, res.cycle_count));
        }

        let mut superposed: Vec<(u16, f32, usize)> = all_peaks.into_iter()
            .map(|(atom, (energy, count))| (atom, energy, count))
            .collect();
        superposed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        let mut agreement = Vec::new();
        for i in 0..per_lens.len() {
            for j in (i + 1)..per_lens.len() {
                let a_set: Vec<u16> = per_lens[i].1.iter()
                    .filter(|&&(_, e)| e > 1e-10)
                    .map(|&(idx, _)| idx).collect();
                let b_set: Vec<u16> = per_lens[j].1.iter()
                    .filter(|&&(_, e)| e > 1e-10)
                    .map(|&(idx, _)| idx).collect();
                let overlap = a_set.iter().filter(|p| b_set.contains(p)).count();
                let max_len = a_set.len().max(b_set.len()).max(1);
                agreement.push((
                    per_lens[i].0.clone(),
                    per_lens[j].0.clone(),
                    overlap as f32 / max_len as f32,
                ));
            }
        }

        CompositeResult { per_lens, superposed, agreement }
    }

    /// Reset all lenses.
    pub fn reset_all(&mut self) {
        for (_, engine) in &mut self.lenses {
            engine.reset();
        }
    }

    /// Number of lenses.
    pub fn lens_count(&self) -> usize {
        self.lenses.len()
    }
}

impl Default for CompositeEngine {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::jina_lens::JINA_HDR_TABLE;
    use crate::bge_m3_lens::BGE_M3_HDR_TABLE;
    use crate::reranker_lens::RERANKER_HDR_TABLE;

    #[test]
    fn composite_creates() {
        let mut comp = CompositeEngine::new();
        comp.add_u8_lens("jina", JINA_HDR_TABLE.to_vec());
        comp.add_u8_lens("bge", BGE_M3_HDR_TABLE.to_vec());
        assert_eq!(comp.lens_count(), 2);
    }

    #[test]
    fn composite_mixed_types() {
        let mut comp = CompositeEngine::new();
        comp.add_u8_lens("jina-u8", JINA_HDR_TABLE.to_vec());

        // BF16 lens from converted u8
        let bf16_table: Vec<u16> = RERANKER_HDR_TABLE.iter()
            .map(|&v| bgz_tensor::stacked_n::f32_to_bf16((v as f32 - 128.0) / 127.0))
            .collect();
        comp.add_bf16_lens("reranker-bf16", bf16_table);

        assert_eq!(comp.lens_count(), 2);

        comp.perturb_all(&[50, 52, 54]);
        let result = comp.think_all(20);
        assert_eq!(result.per_lens.len(), 2);
        assert!(!result.superposed.is_empty());
    }

    #[test]
    fn composite_perturb_and_think() {
        let mut comp = CompositeEngine::new();
        comp.add_u8_lens("jina", JINA_HDR_TABLE.to_vec());
        comp.add_u8_lens("bge", BGE_M3_HDR_TABLE.to_vec());

        comp.perturb_all(&[50, 52, 54]);
        let result = comp.think_all(20);

        assert_eq!(result.per_lens.len(), 2);
        assert!(!result.superposed.is_empty());
        assert!(!result.agreement.is_empty());
    }

    #[test]
    fn composite_reset() {
        let mut comp = CompositeEngine::new();
        comp.add_u8_lens("jina", JINA_HDR_TABLE.to_vec());
        comp.perturb_all(&[42]);
        comp.think_all(5);
        comp.reset_all();
        comp.perturb_all(&[100]);
        let result = comp.think_all(10);
        assert!(result.per_lens[0].1[0].1 > 0.0);
    }
}
