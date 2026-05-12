//! Energy pooling strategies for ThinkingEngine output.
//!
//! Different pooling = different qualia:
//!   ArgMax   = steelwind (sharp, decisive, single peak)
//!   Mean     = woodwarm (broad, grounded, gestalt)
//!   TopK     = oceandrift (multiple currents, multi-thought)
//!   Weighted = nightshade (experience-modulated, ghost-biased)
//!
//! EmbedAnything has Mean/Cls/LastToken for transformer output.
//! We have the same concept on the ENERGY VECTOR after convergence.

use crate::dto::BusDto;

/// Pooling strategy for energy vector after convergence.
#[derive(Clone, Debug)]
pub enum Pooling {
    /// Strongest peak wins. Sharp, decisive. Current default.
    ArgMax,
    /// Average of all active atoms (energy > threshold). Broad gestalt.
    Mean { threshold: f32 },
    /// Top K peaks. Multiple simultaneous thoughts.
    TopK(usize),
    /// Ghost-weighted: multiply energy by experience weights before pooling.
    Weighted { weights: Vec<f32>, inner: Box<Pooling> },
    /// Nucleus sampling (top-p) with temperature. Stochastic, anti-collapse.
    /// Temperature scales logits before softmax. top_p truncates the nucleus.
    /// seed makes it reproducible (None = use entropy from energy).
    Nucleus { temperature: f32, top_p: f32, seed: Option<u64> },
}

/// Result of pooling the energy vector.
#[derive(Clone, Debug)]
pub struct PooledResult {
    /// Primary peak (always present).
    pub primary: (u16, f32),
    /// All selected atoms with their pooled energies.
    pub atoms: Vec<(u16, f32)>,
    /// Pooling strategy used.
    pub strategy: String,
    /// Entropy of the pooled selection.
    pub entropy: f32,
    /// Concentration: what fraction of total energy is in the selection.
    pub concentration: f32,
}

impl Pooling {
    /// Pool the energy vector. Returns the selected atoms.
    pub fn pool(&self, energy: &[f32]) -> PooledResult {
        match self {
            Pooling::ArgMax => pool_argmax(energy),
            Pooling::Mean { threshold } => pool_mean(energy, *threshold),
            Pooling::TopK(k) => pool_topk(energy, *k),
            Pooling::Nucleus { temperature, top_p, seed } => {
                pool_nucleus(energy, *temperature, *top_p, *seed)
            }
            Pooling::Weighted { weights, inner } => {
                let mut weighted = energy.to_vec();
                for (i, e) in weighted.iter_mut().enumerate() {
                    if i < weights.len() {
                        *e *= weights[i];
                    }
                }
                // Re-normalize
                let total: f32 = weighted.iter().sum();
                if total > 1e-10 {
                    for e in &mut weighted { *e /= total; }
                }
                let mut result = inner.pool(&weighted);
                result.strategy = format!("Weighted({})", result.strategy);
                result
            }
        }
    }

    /// Convert a PooledResult into a BusDto (backward compat).
    pub fn to_bus(&self, energy: &[f32], cycles: u16) -> BusDto {
        let pooled = self.pool(energy);
        let mut top_k = [(0u16, 0.0f32); 8];
        for (i, &(idx, e)) in pooled.atoms.iter().take(8).enumerate() {
            top_k[i] = (idx, e);
        }
        BusDto {
            codebook_index: pooled.primary.0,
            energy: pooled.primary.1,
            top_k,
            cycle_count: cycles,
            converged: cycles < 10,
        }
    }
}

fn pool_argmax(energy: &[f32]) -> PooledResult {
    let mut best_idx = 0u16;
    let mut best_val = 0.0f32;
    for (i, &e) in energy.iter().enumerate() {
        if e > best_val {
            best_val = e;
            best_idx = i as u16;
        }
    }
    let total: f32 = energy.iter().sum();
    PooledResult {
        primary: (best_idx, best_val),
        atoms: vec![(best_idx, best_val)],
        strategy: "ArgMax".into(),
        entropy: 0.0, // single atom = zero entropy
        concentration: if total > 1e-10 { best_val / total } else { 0.0 },
    }
}

fn pool_mean(energy: &[f32], threshold: f32) -> PooledResult {
    let mut active: Vec<(u16, f32)> = energy.iter().enumerate()
        .filter(|(_, &e)| e > threshold)
        .map(|(i, &e)| (i as u16, e))
        .collect();
    active.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    if active.is_empty() {
        return PooledResult {
            primary: (0, 0.0),
            atoms: vec![],
            strategy: "Mean".into(),
            entropy: 0.0,
            concentration: 0.0,
        };
    }

    let total: f32 = energy.iter().sum();
    let active_sum: f32 = active.iter().map(|(_, e)| e).sum();
    let _mean_energy = active_sum / active.len() as f32;

    // Compute entropy of active atoms
    let mut entropy = 0.0f32;
    for &(_, e) in &active {
        if e > 1e-10 {
            let p = e / active_sum;
            entropy -= p * p.ln();
        }
    }

    PooledResult {
        primary: active[0],
        atoms: active,
        strategy: "Mean".into(),
        entropy,
        concentration: if total > 1e-10 { active_sum / total } else { 0.0 },
    }
}

fn pool_topk(energy: &[f32], k: usize) -> PooledResult {
    let mut indexed: Vec<(u16, f32)> = energy.iter().enumerate()
        .map(|(i, &e)| (i as u16, e))
        .collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    let selected: Vec<(u16, f32)> = indexed.into_iter()
        .take(k)
        .filter(|&(_, e)| e > 1e-10)
        .collect();

    let total: f32 = energy.iter().sum();
    let selected_sum: f32 = selected.iter().map(|(_, e)| e).sum();

    let mut entropy = 0.0f32;
    for &(_, e) in &selected {
        if e > 1e-10 && selected_sum > 1e-10 {
            let p = e / selected_sum;
            entropy -= p * p.ln();
        }
    }

    let primary = selected.first().cloned().unwrap_or((0, 0.0));

    PooledResult {
        primary,
        atoms: selected,
        strategy: format!("TopK({})", k),
        entropy,
        concentration: if total > 1e-10 { selected_sum / total } else { 0.0 },
    }
}

fn pool_nucleus(energy: &[f32], temperature: f32, top_p: f32, seed: Option<u64>) -> PooledResult {
    // Sort by energy descending
    let mut indexed: Vec<(u16, f32)> = energy.iter().enumerate()
        .map(|(i, &e)| (i as u16, e))
        .filter(|(_, e)| *e > 1e-10)
        .collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    if indexed.is_empty() {
        return PooledResult {
            primary: (0, 0.0), atoms: vec![], strategy: "Nucleus".into(),
            entropy: 0.0, concentration: 0.0,
        };
    }

    // Apply temperature: scale logits then softmax
    let temp = temperature.max(0.01);
    let scaled: Vec<f32> = indexed.iter()
        .map(|&(_, e)| (e.ln().max(-20.0) / temp).exp())
        .collect();
    let total_scaled: f32 = scaled.iter().sum();
    let probs: Vec<f32> = scaled.iter().map(|s| s / total_scaled.max(1e-10)).collect();

    // Nucleus: accumulate until top_p reached
    let mut cumsum = 0.0f32;
    let mut nucleus = Vec::new();
    for (i, &p) in probs.iter().enumerate() {
        cumsum += p;
        nucleus.push((indexed[i].0, p));
        if cumsum >= top_p { break; }
    }

    // Sample from the nucleus using a simple deterministic hash
    // (for reproducibility; replace with real RNG if needed)
    let hash_seed = seed.unwrap_or_else(|| {
        // Derive seed from energy distribution
        let mut h = 0x9e3779b97f4a7c15u64;
        for &(idx, _) in &nucleus {
            h = h.wrapping_mul(31).wrapping_add(idx as u64);
        }
        h
    });
    let sample_idx = (hash_seed % nucleus.len() as u64) as usize;
    let primary = nucleus[sample_idx];

    let total: f32 = energy.iter().sum();
    let nucleus_sum: f32 = nucleus.iter().map(|(_, p)| p).sum();

    let mut entropy = 0.0f32;
    for &(_, p) in &nucleus {
        if p > 1e-10 { entropy -= p * p.ln(); }
    }

    // Map back to energy scale for the atoms
    let atoms: Vec<(u16, f32)> = nucleus.iter()
        .map(|&(idx, _)| {
            let orig_energy = energy[idx as usize];
            (idx, orig_energy)
        })
        .collect();

    PooledResult {
        primary: (primary.0, energy[primary.0 as usize]),
        atoms,
        strategy: format!("Nucleus(T={:.1},p={:.2})", temperature, top_p),
        entropy,
        concentration: if total > 1e-10 { nucleus_sum } else { 0.0 },
    }
}

impl Default for Pooling {
    fn default() -> Self {
        Pooling::ArgMax
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_energy() -> Vec<f32> {
        let mut e = vec![0.0f32; 256];
        e[50] = 0.30;
        e[52] = 0.25;
        e[54] = 0.20;
        e[100] = 0.10;
        e[130] = 0.08;
        e[200] = 0.05;
        e[10] = 0.01;
        e[1] = 0.01;
        e
    }

    #[test]
    fn argmax_finds_peak() {
        let energy = make_energy();
        let result = Pooling::ArgMax.pool(&energy);
        assert_eq!(result.primary.0, 50);
        assert_eq!(result.atoms.len(), 1);
        assert!(result.concentration > 0.0);
    }

    #[test]
    fn mean_collects_active() {
        let energy = make_energy();
        let result = Pooling::Mean { threshold: 0.05 }.pool(&energy);
        assert_eq!(result.primary.0, 50);
        // Should include atoms > 0.05: 50, 52, 54, 100, 130
        assert!(result.atoms.len() >= 4);
        assert!(result.entropy > 0.0);
        assert!(result.concentration > 0.5);
    }

    #[test]
    fn topk_selects_k() {
        let energy = make_energy();
        let result = Pooling::TopK(3).pool(&energy);
        assert_eq!(result.atoms.len(), 3);
        assert_eq!(result.primary.0, 50);
        assert_eq!(result.atoms[1].0, 52);
        assert_eq!(result.atoms[2].0, 54);
    }

    #[test]
    fn weighted_modulates() {
        let energy = make_energy();
        // Suppress atom 50, boost atom 100
        let mut weights = vec![1.0f32; 256];
        weights[50] = 0.01; // suppress
        weights[100] = 10.0; // boost

        let result = Pooling::Weighted {
            weights,
            inner: Box::new(Pooling::ArgMax),
        }.pool(&energy);

        // Atom 100 should now dominate (0.10 * 10 = 1.0 vs 0.30 * 0.01 = 0.003)
        assert_eq!(result.primary.0, 100);
        assert!(result.strategy.contains("Weighted"));
    }

    #[test]
    fn to_bus_compat() {
        let energy = make_energy();
        let bus = Pooling::TopK(5).to_bus(&energy, 7);
        assert_eq!(bus.codebook_index, 50);
        assert!(bus.energy > 0.0);
        assert_eq!(bus.cycle_count, 7);
    }

    #[test]
    fn nucleus_samples_from_top() {
        let energy = make_energy();
        let result = Pooling::Nucleus {
            temperature: 1.0,
            top_p: 0.9,
            seed: Some(42),
        }.pool(&energy);
        // Should select from the nucleus (top atoms by energy)
        assert!(!result.atoms.is_empty());
        assert!(result.primary.1 > 0.0);
        assert!(result.strategy.contains("Nucleus"));
    }

    #[test]
    fn nucleus_low_temp_concentrates() {
        let energy = make_energy();
        let r_low = Pooling::Nucleus {
            temperature: 0.1,
            top_p: 0.9,
            seed: Some(42),
        }.pool(&energy);
        let r_high = Pooling::Nucleus {
            temperature: 2.0,
            top_p: 0.9,
            seed: Some(42),
        }.pool(&energy);
        // Low temperature should have fewer atoms in nucleus (more concentrated)
        assert!(r_low.atoms.len() <= r_high.atoms.len() + 1,
            "low temp {} atoms vs high temp {} atoms",
            r_low.atoms.len(), r_high.atoms.len());
    }

    #[test]
    fn nucleus_deterministic_with_seed() {
        let energy = make_energy();
        let p = Pooling::Nucleus { temperature: 0.8, top_p: 0.9, seed: Some(123) };
        let r1 = p.pool(&energy);
        let r2 = p.pool(&energy);
        assert_eq!(r1.primary.0, r2.primary.0, "same seed should give same result");
    }

    #[test]
    fn empty_energy_safe() {
        let energy = vec![0.0f32; 256];
        let r1 = Pooling::ArgMax.pool(&energy);
        assert_eq!(r1.primary.1, 0.0);

        let r2 = Pooling::Mean { threshold: 0.01 }.pool(&energy);
        assert!(r2.atoms.is_empty());

        let r3 = Pooling::TopK(3).pool(&energy);
        assert!(r3.atoms.is_empty());
    }
}
