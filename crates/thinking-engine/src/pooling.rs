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
    let mean_energy = active_sum / active.len() as f32;

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
