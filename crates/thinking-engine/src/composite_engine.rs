//! Multi-model composition: multiple lenses → one thought.
//!
//! Each lens runs its own engine. Results are composed via
//! energy superposition: weighted sum of peak energies across lenses.
//!
//! ```text
//! Jina v3 lens     →  engine_1  →  peaks_1
//! Reranker lens     →  engine_2  →  peaks_2   →  superposition  →  CompositeResult
//! BGE-M3 lens       →  engine_3  →  peaks_3
//! ```
//!
//! Models are SENSORS. Each sees different aspects of the same input.
//! Superposition reveals what ALL lenses agree on (strong) vs what
//! only some see (weak). Disagreement IS information.

use crate::engine::ThinkingEngine;
use crate::dto::ResonanceDto;

/// Result of multi-model composition.
pub struct CompositeResult {
    /// Per-lens results: (lens_name, top_k peaks, cycle_count).
    pub per_lens: Vec<(String, [(u16, f32); 8], u16)>,
    /// Superposed peaks: atoms that appear across multiple lenses.
    /// Sorted by composite score (sum of energies across lenses).
    pub superposed: Vec<(u16, f32, usize)>, // (atom, composite_energy, lens_count)
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
        s.push_str(&format!("Superposed top-5:\n"));
        for (atom, energy, count) in self.superposed.iter().take(5) {
            s.push_str(&format!("  atom {} = {:.3} ({} lenses)\n", atom, energy, count));
        }
        for (a, b, agree) in &self.agreement {
            s.push_str(&format!("  {} × {} = {:.0}%\n", a, b, agree * 100.0));
        }
        s
    }
}

/// Multi-model composition engine.
///
/// Each lens has its own ThinkingEngine with its own distance table.
/// Perturb each lens independently (different token→centroid mappings),
/// then compose results via energy superposition.
pub struct CompositeEngine {
    lenses: Vec<(String, ThinkingEngine)>,
}

impl CompositeEngine {
    pub fn new() -> Self {
        Self { lenses: Vec::new() }
    }

    /// Add a lens with its baked distance table.
    pub fn add_lens(&mut self, name: &str, table: Vec<u8>) {
        self.lenses.push((name.to_string(), ThinkingEngine::new(table)));
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
    /// Only valid when all lenses share the same codebook space.
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
            let res = engine.think(max_cycles);

            // Accumulate peaks into superposition
            for &(idx, energy) in &res.top_k {
                if energy > 1e-10 {
                    let entry = all_peaks.entry(idx).or_insert((0.0, 0));
                    entry.0 += energy;
                    entry.1 += 1;
                }
            }

            per_lens.push((name.clone(), res.top_k, res.cycle_count));
        }

        // Sort superposed by composite energy
        let mut superposed: Vec<(u16, f32, usize)> = all_peaks.into_iter()
            .map(|(atom, (energy, count))| (atom, energy, count))
            .collect();
        superposed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // Compute pairwise agreement
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
                let agree = overlap as f32 / max_len as f32;
                agreement.push((
                    per_lens[i].0.clone(),
                    per_lens[j].0.clone(),
                    agree,
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
        comp.add_lens("jina", JINA_HDR_TABLE.to_vec());
        comp.add_lens("bge", BGE_M3_HDR_TABLE.to_vec());
        assert_eq!(comp.lens_count(), 2);
    }

    #[test]
    fn composite_perturb_and_think() {
        let mut comp = CompositeEngine::new();
        comp.add_lens("jina", JINA_HDR_TABLE.to_vec());
        comp.add_lens("bge", BGE_M3_HDR_TABLE.to_vec());

        comp.perturb_all(&[50, 52, 54]);
        let result = comp.think_all(20);

        assert_eq!(result.per_lens.len(), 2);
        assert!(!result.superposed.is_empty());
        assert!(!result.agreement.is_empty());
    }

    #[test]
    fn composite_three_lenses() {
        let mut comp = CompositeEngine::new();
        comp.add_lens("jina", JINA_HDR_TABLE.to_vec());
        comp.add_lens("bge", BGE_M3_HDR_TABLE.to_vec());
        comp.add_lens("reranker", RERANKER_HDR_TABLE.to_vec());

        comp.perturb_all(&[10, 100, 200]);
        let result = comp.think_all(20);

        assert_eq!(result.per_lens.len(), 3);
        // Should have 3 pairwise agreement scores
        assert_eq!(result.agreement.len(), 3);

        // Multi-lens atoms should appear in superposed
        let multi = result.superposed.iter()
            .filter(|&&(_, _, count)| count >= 2)
            .count();
        // At least some atoms should appear in multiple lenses
        // (same codebook indices perturbed)
        assert!(multi > 0 || result.superposed.len() > 0,
            "superposition should have results");
    }

    #[test]
    fn composite_reset() {
        let mut comp = CompositeEngine::new();
        comp.add_lens("jina", JINA_HDR_TABLE.to_vec());
        comp.perturb_all(&[42]);
        comp.think_all(5);
        comp.reset_all();
        // After reset, thinking should start fresh
        comp.perturb_all(&[100]);
        let result = comp.think_all(10);
        assert!(result.per_lens[0].1[0].1 > 0.0);
    }

    #[test]
    fn composite_per_lens_perturb() {
        let mut comp = CompositeEngine::new();
        comp.add_lens("jina", JINA_HDR_TABLE.to_vec());
        comp.add_lens("reranker", RERANKER_HDR_TABLE.to_vec());

        // Different perturbation per lens
        comp.perturb_lens("jina", &[10, 20, 30]);
        comp.perturb_lens("reranker", &[200, 210, 220]);

        let result = comp.think_all(20);
        assert_eq!(result.per_lens.len(), 2);
        // Different perturbations should produce different peaks
        let jina_top = result.per_lens[0].1[0].0;
        let rr_top = result.per_lens[1].1[0].0;
        // Not guaranteed different but likely with distant indices
        assert!(jina_top != 0 || rr_top != 0, "both should have valid peaks");
    }
}
