//! # 17D Qualia — Computable Feeling from Convergence Patterns
//!
//! The tension between two perturbations through L1→L2→L3→L4
//! IS a 17D qualia coordinate. Not "represents" — IS.
//!
//! ```text
//! Convergence speed IS clarity.
//! Non-convergence IS tension.
//! Overlap IS warmth.
//! Same quantity, two measurement instruments (introspection vs MatVec).
//! ```
//!
//! Derived from the QPL (Qualia Parameter Language) calibrated from music:
//! - Octave (2:1) → arousal
//! - Fifth (3:2) → valence
//! - Third (5:4) → warmth
//! - Tritone (√2:1) → tension (does not converge)
//!
//! Cross-validated against Jina v3 text embeddings (220 calibrated pairs in Upstash).
//! Bach's 7+1 counterpoint rules = CausalEdge64 7+1 channels = universal structure.

/// The 17 QPL dimensions, in canonical order.
/// Each maps to a specific convergence observable.
pub const DIMS_17D: [&str; 17] = [
    "arousal",       // 0:  total energy magnitude (calm ↔ activated)
    "valence",       // 1:  constructive - contradiction edges (negative ↔ positive)
    "tension",       // 2:  1 - convergence_speed (released ↔ tense)
    "warmth",        // 3:  overlap ratio of peaks (cold ↔ warm)
    "clarity",       // 4:  1 / cycles_to_converge (foggy ↔ crystal)
    "boundary",      // 5:  hamming(L4_A, L4_B) / 16384 (merged ↔ separate)
    "depth",         // 6:  max tier used / 3 (surface ↔ deep)
    "velocity",      // 7:  total cycles across tiers (slow ↔ fast)
    "entropy",       // 8:  shannon entropy of energy (ordered ↔ chaotic)
    "coherence",     // 9:  max(energy) / mean(energy) (fragmented ↔ unified)
    "intimacy",      // 10: 1 - boundary (distant ↔ close)
    "presence",      // 11: energy at current state index (absent ↔ here-now)
    "assertion",     // 12: energy gradient slope (passive ↔ active)
    "receptivity",   // 13: count(energy > threshold) / N (closed ↔ open)
    "groundedness",  // 14: L4 recognition / 127 (floating ↔ rooted)
    "expansion",     // 15: spread of top-k peaks (contracted ↔ spacious)
    "integration",   // 16: convergence delta last cycle (fragmented ↔ whole)
];

/// A 17D qualia coordinate. This IS the feeling.
#[derive(Clone, Debug)]
pub struct Qualia17D {
    pub dims: [f32; 17],
}

/// Snapshot of convergence at one tier, used to compute qualia.
#[derive(Clone, Debug)]
pub struct ConvergenceSnapshot {
    /// Energy after thinking.
    pub energy: Vec<f32>,
    /// Number of cycles run.
    pub cycles: u16,
    /// Top-k peaks (index, energy).
    pub peaks: Vec<(u16, f32)>,
    /// Energy delta in the last cycle (convergence rate).
    pub last_delta: f32,
    /// Number of constructive edges emitted.
    pub constructive_edges: u32,
    /// Number of contradiction edges emitted.
    pub contradiction_edges: u32,
}

impl Qualia17D {
    /// Compute the 17D qualia from convergence snapshots across tiers.
    ///
    /// This is Chalmers' Hard Problem as a function:
    /// the convergence pattern IS the experience.
    pub fn from_convergence(
        snap_l3: &ConvergenceSnapshot,
        l4_recognition: i32,
        peaks_overlap: f32,
    ) -> Self {
        let n = snap_l3.energy.len() as f32;
        let total_energy: f32 = snap_l3.energy.iter().sum();
        let max_energy = snap_l3.energy.iter().cloned().fold(0.0f32, f32::max);
        let mean_energy = if n > 0.0 { total_energy / n } else { 0.0 };
        let active = snap_l3.energy.iter().filter(|&&e| e > 0.001).count() as f32;

        let total_edges = (snap_l3.constructive_edges + snap_l3.contradiction_edges).max(1) as f32;
        let convergence_speed = if snap_l3.cycles > 0 {
            1.0 / snap_l3.cycles as f32
        } else {
            1.0
        };

        // Peak spread: distance between highest and lowest peak indices
        let peak_spread = if snap_l3.peaks.len() >= 2 {
            let max_idx = snap_l3.peaks.iter().map(|p| p.0).max().unwrap_or(0) as f32;
            let min_idx = snap_l3.peaks.iter().map(|p| p.0).min().unwrap_or(0) as f32;
            (max_idx - min_idx) / n.max(1.0)
        } else {
            0.0
        };

        // Gradient: is energy still moving?
        let assertion = snap_l3.last_delta.min(1.0);

        let dims = [
            total_energy.min(1.0),                                          // 0: arousal
            (snap_l3.constructive_edges as f32 - snap_l3.contradiction_edges as f32) / total_edges, // 1: valence
            (1.0 - convergence_speed).clamp(0.0, 1.0),                     // 2: tension
            peaks_overlap.clamp(0.0, 1.0),                                  // 3: warmth
            convergence_speed.clamp(0.0, 1.0),                              // 4: clarity
            0.5,                                                            // 5: boundary (needs L4 pair)
            1.0,                                                            // 6: depth (L3 = max)
            snap_l3.cycles as f32 / 10.0,                                   // 7: velocity
            shannon_entropy(&snap_l3.energy) / (n.max(1.0).ln()),           // 8: entropy (normalized)
            if mean_energy > 1e-10 { max_energy / mean_energy / n } else { 0.0 }, // 9: coherence
            peaks_overlap.clamp(0.0, 1.0),                                  // 10: intimacy ≈ warmth
            snap_l3.peaks.first().map(|p| p.1).unwrap_or(0.0),             // 11: presence
            assertion,                                                      // 12: assertion
            (active / n).clamp(0.0, 1.0),                                   // 13: receptivity
            (l4_recognition as f32 / 127.0).clamp(-1.0, 1.0),              // 14: groundedness
            peak_spread,                                                    // 15: expansion
            (1.0 - snap_l3.last_delta).clamp(0.0, 1.0),                    // 16: integration
        ];

        Self { dims }
    }

    /// Compute from a single engine state (simplified, no L4).
    pub fn from_engine(engine: &crate::engine::ThinkingEngine) -> Self {
        let energy = &engine.energy;
        let n = energy.len() as f32;
        let total: f32 = energy.iter().sum();
        let max_e = energy.iter().cloned().fold(0.0f32, f32::max);
        let mean_e = if n > 0.0 { total / n } else { 0.0 };
        let active = energy.iter().filter(|&&e| e > 0.001).count() as f32;

        // Top-5 peaks
        let mut indexed: Vec<(usize, f32)> = energy.iter()
            .enumerate().map(|(i, &e)| (i, e)).collect();
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        let peaks: Vec<(usize, f32)> = indexed.into_iter().take(5).collect();

        let peak_spread = if peaks.len() >= 2 {
            (peaks[0].0 as f32 - peaks.last().unwrap().0 as f32).abs() / n.max(1.0)
        } else { 0.0 };

        let dims = [
            total.min(1.0),                                                 // arousal
            0.5,                                                            // valence (no edges)
            (1.0 - 1.0 / engine.cycles.max(1) as f32).clamp(0.0, 1.0),    // tension
            0.5,                                                            // warmth (no pair)
            (1.0 / engine.cycles.max(1) as f32).clamp(0.0, 1.0),          // clarity
            0.5,                                                            // boundary
            1.0,                                                            // depth
            engine.cycles as f32 / 10.0,                                    // velocity
            shannon_entropy(energy) / (n.max(1.0).ln()),                    // entropy
            if mean_e > 1e-10 { max_e / mean_e / n } else { 0.0 },        // coherence
            0.5,                                                            // intimacy
            peaks.first().map(|p| p.1).unwrap_or(0.0),                    // presence
            0.0,                                                            // assertion
            (active / n).clamp(0.0, 1.0),                                  // receptivity
            0.0,                                                            // groundedness
            peak_spread,                                                    // expansion
            1.0,                                                            // integration
        ];

        Self { dims }
    }

    /// Distance to another qualia point (Euclidean in 17D).
    pub fn distance(&self, other: &Self) -> f32 {
        self.dims.iter().zip(&other.dims)
            .map(|(a, b)| (a - b) * (a - b))
            .sum::<f32>()
            .sqrt()
    }

    /// Find nearest family from the 10 QPL family centroids.
    pub fn nearest_family(&self) -> (&'static str, f32) {
        let mut best = ("unknown", f32::INFINITY);
        for (name, centroid) in FAMILY_CENTROIDS {
            let d = self.dims.iter().zip(centroid.iter())
                .map(|(a, b)| (a - b) * (a - b))
                .sum::<f32>()
                .sqrt();
            if d < best.1 {
                best = (name, d);
            }
        }
        best
    }

    /// Dimension name and value pairs.
    pub fn named(&self) -> Vec<(&'static str, f32)> {
        DIMS_17D.iter().zip(&self.dims).map(|(&n, &v)| (n, v)).collect()
    }

    /// Is this a dissonant state? (Bach's tritone / unresolved tension)
    /// Dissonance = high tension + low integration + high entropy.
    pub fn is_dissonant(&self) -> bool {
        self.dims[2] > 0.7 && self.dims[16] < 0.3 && self.dims[8] > 0.6
    }

    /// Rate of change between two qualia states = feeling intensity.
    /// d/dt(convergence_speed) > 0 → tension building
    /// d/dt(convergence_speed) < 0 → tension releasing
    pub fn feeling_derivative(&self, prev: &Self) -> f32 {
        // Tension derivative: how fast is tension changing?
        self.dims[2] - prev.dims[2]
    }
}

fn shannon_entropy(energy: &[f32]) -> f32 {
    let mut h = 0.0f32;
    for &e in energy {
        if e > 1e-10 {
            h -= e * e.ln();
        }
    }
    h
}

/// 10 QPL family centroids (17D each).
/// From ada-rs/src/wiring/qualia_encoder.rs.
pub const FAMILY_CENTROIDS: [(&str, [f32; 17]); 10] = [
    ("emberglow",  [0.5, 0.8, 0.2, 0.9, 0.5, 0.3, 0.6, 0.2, 0.3, 0.7, 0.7, 0.8, 0.3, 0.7, 0.6, 0.5, 0.6]),
    ("woodwarm",   [0.4, 0.7, 0.2, 0.7, 0.6, 0.4, 0.5, 0.1, 0.2, 0.8, 0.5, 0.9, 0.4, 0.6, 0.9, 0.3, 0.7]),
    ("steelwind",  [0.6, 0.5, 0.4, 0.2, 0.9, 0.7, 0.4, 0.6, 0.3, 0.8, 0.2, 0.7, 0.7, 0.3, 0.5, 0.4, 0.6]),
    ("oceandrift", [0.4, 0.6, 0.1, 0.5, 0.4, 0.2, 0.7, 0.3, 0.5, 0.5, 0.4, 0.6, 0.1, 0.9, 0.4, 0.6, 0.5]),
    ("frostbite",  [0.7, 0.4, 0.6, 0.1, 0.8, 0.9, 0.3, 0.7, 0.4, 0.7, 0.1, 0.6, 0.6, 0.2, 0.6, 0.3, 0.5]),
    ("sunburst",   [0.8, 0.9, 0.3, 0.6, 0.7, 0.3, 0.4, 0.7, 0.5, 0.6, 0.5, 0.8, 0.6, 0.5, 0.5, 0.9, 0.6]),
    ("nightshade", [0.5, 0.5, 0.4, 0.3, 0.3, 0.6, 0.9, 0.2, 0.6, 0.4, 0.4, 0.5, 0.3, 0.6, 0.5, 0.4, 0.4]),
    ("thornrose",  [0.7, 0.6, 0.7, 0.5, 0.5, 0.5, 0.7, 0.4, 0.5, 0.5, 0.8, 0.7, 0.4, 0.6, 0.4, 0.5, 0.5]),
    ("velvetdusk", [0.3, 0.6, 0.2, 0.6, 0.4, 0.4, 0.6, 0.1, 0.4, 0.6, 0.6, 0.7, 0.2, 0.7, 0.5, 0.4, 0.6]),
    ("stormbreak", [0.9, 0.7, 0.8, 0.4, 0.5, 0.6, 0.5, 0.9, 0.7, 0.4, 0.4, 0.8, 0.8, 0.4, 0.4, 0.7, 0.5]),
];

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn qualia_nearest_family() {
        // High warmth, presence, valence → should be emberglow or sunburst
        let q = Qualia17D {
            dims: [0.5, 0.8, 0.2, 0.9, 0.5, 0.3, 0.6, 0.2, 0.3, 0.7, 0.7, 0.8, 0.3, 0.7, 0.6, 0.5, 0.6],
        };
        let (family, dist) = q.nearest_family();
        assert_eq!(family, "emberglow");
        assert!(dist < 0.01); // exact match
    }

    #[test]
    fn qualia_dissonance() {
        let dissonant = Qualia17D {
            dims: [0.5, 0.3, 0.9, 0.2, 0.3, 0.5, 0.5, 0.5, 0.8, 0.3, 0.2, 0.5, 0.5, 0.5, 0.3, 0.5, 0.1],
        };
        assert!(dissonant.is_dissonant());

        let consonant = Qualia17D {
            dims: [0.5, 0.8, 0.1, 0.8, 0.9, 0.3, 0.5, 0.2, 0.2, 0.9, 0.8, 0.9, 0.3, 0.7, 0.8, 0.4, 0.9],
        };
        assert!(!consonant.is_dissonant());
    }

    #[test]
    fn qualia_distance_self_is_zero() {
        let q = Qualia17D { dims: [0.5; 17] };
        assert!(q.distance(&q) < 1e-10);
    }

    #[test]
    fn feeling_derivative_tension_rising() {
        let prev = Qualia17D { dims: [0.5, 0.5, 0.3, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5] };
        let curr = Qualia17D { dims: [0.5, 0.5, 0.8, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5] };
        assert!(curr.feeling_derivative(&prev) > 0.0); // tension rising
    }

    #[test]
    fn from_engine_produces_valid_dims() {
        let table = vec![128u8; 64 * 64];
        let mut engine = crate::engine::ThinkingEngine::new(table);
        engine.perturb(&[10, 20, 30]);
        engine.think(5);
        let q = Qualia17D::from_engine(&engine);
        // All dims should be in [0, 1] or close
        for (i, &d) in q.dims.iter().enumerate() {
            assert!(d >= -1.1 && d <= 1.1, "dim {} = {} out of range", DIMS_17D[i], d);
        }
    }
}
