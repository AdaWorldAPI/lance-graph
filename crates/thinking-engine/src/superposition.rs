//! Superposition Gate: multi-lens interference + thinking style detection.
//!
//! Two lens ripples multiply at each atom → superposition amplitude.
//! L4 threshold filters which amplitudes survive → thinking style emerges.
//!
//! ```text
//! gate(atom) = visits_lens_A(atom) × visits_lens_B(atom)
//!   Zero × anything = collapsed (destructive interference)
//!   Both positive = reinforced (constructive interference)
//!   The superposition lives in the product space.
//! ```

use std::collections::HashMap;

/// Superposition amplitude at each atom from multi-lens interference.
#[derive(Clone, Debug)]
pub struct SuperpositionField {
    /// Product of visit counts from all lenses. Zero = destructive.
    pub amplitudes: Vec<f32>,
    /// Which atoms have constructive interference (amplitude > 0).
    pub resonant_atoms: Vec<(u16, f32)>,
    /// Total constructive energy.
    pub total_energy: f32,
    /// Number of atoms with nonzero amplitude.
    pub n_resonant: usize,
}

/// Detected thinking style based on superposition pattern.
#[derive(Clone, Debug, PartialEq)]
pub enum ThinkingStyle {
    /// Gate opens early, few atoms, high confidence.
    /// The thought is focused and decisive.
    Analytical,
    /// Gate opens late, many atoms, initially low confidence.
    /// The thought explores widely before settling.
    Creative,
    /// Gate opens specifically on high-dissonance atoms.
    /// The thought engages with tension and contradiction.
    Emotional,
    /// Gate opens where L4 recognition is high.
    /// The thought trusts familiarity and pattern.
    Intuitive,
    /// No clear pattern — mixed or neutral.
    Diffuse,
}

impl std::fmt::Display for ThinkingStyle {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            Self::Analytical => write!(f, "analytical (focused, decisive)"),
            Self::Creative => write!(f, "creative (exploratory, divergent)"),
            Self::Emotional => write!(f, "emotional (tension-engaged)"),
            Self::Intuitive => write!(f, "intuitive (pattern-trusting)"),
            Self::Diffuse => write!(f, "diffuse (mixed, searching)"),
        }
    }
}

/// L4-style thresholds for thinking style modulation.
/// These are learned over time via RL reward.
#[derive(Clone, Debug)]
pub struct StyleThresholds {
    /// Minimum superposition amplitude to count as "resonant."
    pub gate_threshold: f32,
    /// If resonant atoms < this fraction of total, style = analytical.
    pub analytical_sparsity: f32,
    /// If resonant atoms > this fraction, style = creative.
    pub creative_density: f32,
    /// If avg dissonance of resonant atoms > this, style = emotional.
    pub emotional_dissonance: f32,
}

impl Default for StyleThresholds {
    fn default() -> Self {
        Self {
            gate_threshold: 0.01,
            analytical_sparsity: 0.05,
            creative_density: 0.25,
            emotional_dissonance: 0.2,
        }
    }
}

/// Compute the superposition field from multiple lens visit maps.
///
/// Each lens contributes a HashMap<atom, visit_count>.
/// The superposition amplitude at each atom is the PRODUCT of visit counts
/// across all lenses. Zero in any lens = zero amplitude (destructive).
pub fn compute_superposition(
    lens_visits: &[&HashMap<u16, u32>],
    n_atoms: usize,
) -> SuperpositionField {
    let mut amplitudes = vec![0.0f32; n_atoms];

    // For each atom, multiply visit counts across all lenses
    // Only nonzero if ALL lenses visited this atom
    let n_lenses = lens_visits.len() as f32;

    // Collect all atoms that appear in ANY lens
    let mut all_atoms: HashMap<u16, Vec<u32>> = HashMap::new();
    for visits in lens_visits {
        for (&atom, &count) in *visits {
            all_atoms.entry(atom).or_insert_with(|| vec![0; lens_visits.len()]);
        }
    }
    for (i, visits) in lens_visits.iter().enumerate() {
        for (&atom, &count) in *visits {
            if let Some(counts) = all_atoms.get_mut(&atom) {
                if i < counts.len() {
                    counts[i] = count;
                }
            }
        }
    }

    // Compute product amplitude
    let mut resonant_atoms = Vec::new();
    let mut total_energy = 0.0f32;

    for (&atom, counts) in &all_atoms {
        if (atom as usize) >= n_atoms { continue; }

        // Product of all lens visit counts (geometric mean for normalization)
        let product: f32 = counts.iter()
            .map(|&c| c as f32)
            .product();

        // Normalize by number of lenses (geometric mean)
        let amplitude = if product > 0.0 {
            product.powf(1.0 / n_lenses)
        } else { 0.0 };

        amplitudes[atom as usize] = amplitude;
        if amplitude > 0.0 {
            resonant_atoms.push((atom, amplitude));
            total_energy += amplitude;
        }
    }

    resonant_atoms.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    let n_resonant = resonant_atoms.len();

    SuperpositionField { amplitudes, resonant_atoms, total_energy, n_resonant }
}

/// Detect thinking style from the superposition pattern.
pub fn detect_style(
    field: &SuperpositionField,
    n_atoms: usize,
    avg_dissonance: f32,
    thresholds: &StyleThresholds,
) -> ThinkingStyle {
    let resonant_fraction = field.n_resonant as f32 / n_atoms.max(1) as f32;

    // Gate: only count atoms above threshold
    let gated_count = field.resonant_atoms.iter()
        .filter(|(_, amp)| *amp > thresholds.gate_threshold)
        .count();
    let gated_fraction = gated_count as f32 / n_atoms.max(1) as f32;

    if gated_fraction < thresholds.analytical_sparsity && gated_count > 0 {
        ThinkingStyle::Analytical
    } else if avg_dissonance > thresholds.emotional_dissonance {
        ThinkingStyle::Emotional
    } else if gated_fraction > thresholds.creative_density {
        ThinkingStyle::Creative
    } else if gated_count > 0 {
        ThinkingStyle::Intuitive
    } else {
        ThinkingStyle::Diffuse
    }
}

/// Run the full superposition pipeline:
/// 1. Collect visit maps from each lens cascade
/// 2. Compute superposition field (product of visits)
/// 3. Detect thinking style
/// 4. Return gated atoms for next cascade stage
pub fn superposition_cascade(
    lens_stages: &[&[crate::domino::StageResult]],
    n_atoms: usize,
    avg_dissonance: f32,
    thresholds: &StyleThresholds,
) -> (SuperpositionField, ThinkingStyle, Vec<u16>) {
    // Build visit maps from cascade stages
    let visit_maps: Vec<HashMap<u16, u32>> = lens_stages.iter().map(|stages| {
        let mut visits: HashMap<u16, u32> = HashMap::new();
        for stage in *stages {
            for atom in &stage.focus {
                *visits.entry(atom.index).or_insert(0) += 1;
            }
            for atom in &stage.promoted {
                *visits.entry(atom.index).or_insert(0) += 1;
            }
        }
        visits
    }).collect();

    let visit_refs: Vec<&HashMap<u16, u32>> = visit_maps.iter().collect();
    let field = compute_superposition(&visit_refs, n_atoms);
    let style = detect_style(&field, n_atoms, avg_dissonance, thresholds);

    // Gated atoms: only those above threshold, sorted by amplitude
    let gated: Vec<u16> = field.resonant_atoms.iter()
        .filter(|(_, amp)| *amp > thresholds.gate_threshold)
        .map(|(atom, _)| *atom)
        .collect();

    (field, style, gated)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn superposition_product() {
        let mut a = HashMap::new();
        a.insert(10u16, 3u32);
        a.insert(20, 5);
        a.insert(30, 1);

        let mut b = HashMap::new();
        b.insert(10, 2);
        b.insert(20, 4);
        // 30 not in b → destructive

        let field = compute_superposition(&[&a, &b], 256);

        // Atom 10: sqrt(3*2) = sqrt(6) ≈ 2.45
        assert!(field.amplitudes[10] > 2.0);
        // Atom 20: sqrt(5*4) = sqrt(20) ≈ 4.47
        assert!(field.amplitudes[20] > 4.0);
        // Atom 30: destructive (not in b)
        assert_eq!(field.amplitudes[30], 0.0);
    }

    #[test]
    fn style_analytical() {
        let mut field = SuperpositionField {
            amplitudes: vec![0.0; 256],
            resonant_atoms: vec![(10, 5.0), (20, 3.0)],
            total_energy: 8.0,
            n_resonant: 2,
        };
        let style = detect_style(&field, 256, 0.0, &StyleThresholds::default());
        assert_eq!(style, ThinkingStyle::Analytical); // 2/256 < 5% = analytical
    }

    #[test]
    fn style_emotional() {
        let field = SuperpositionField {
            amplitudes: vec![0.0; 256],
            resonant_atoms: (0..50).map(|i| (i, 1.0)).collect(),
            total_energy: 50.0,
            n_resonant: 50,
        };
        let style = detect_style(&field, 256, 0.5, &StyleThresholds::default());
        assert_eq!(style, ThinkingStyle::Emotional); // high dissonance
    }

    #[test]
    fn style_creative() {
        let field = SuperpositionField {
            amplitudes: vec![0.0; 256],
            resonant_atoms: (0..100).map(|i| (i, 0.5)).collect(),
            total_energy: 50.0,
            n_resonant: 100,
        };
        let style = detect_style(&field, 256, 0.05, &StyleThresholds::default());
        assert_eq!(style, ThinkingStyle::Creative); // 100/256 > 25%
    }

    #[test]
    fn style_display() {
        assert_eq!(format!("{}", ThinkingStyle::Analytical), "analytical (focused, decisive)");
        assert_eq!(format!("{}", ThinkingStyle::Emotional), "emotional (tension-engaged)");
    }
}
