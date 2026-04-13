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

    /// Compute qualia from the SUPERPOSITION FIELD — the interference pattern
    /// between multiple lens ripples. This IS the emotional color.
    ///
    /// Agreement between lenses → warmth, coherence
    /// Disagreement between lenses → tension, longing
    /// High amplitude → arousal, presence
    /// Low amplitude → mystery, depth
    pub fn from_superposition(
        field: &crate::superposition::SuperpositionField,
        style: &crate::superposition::ThinkingStyle,
        avg_dissonance: f32,
        lens_agreement: f32, // 0.0 = fully disagree, 1.0 = fully agree
    ) -> Self {
        let n = field.amplitudes.len() as f32;
        let resonant_frac = field.n_resonant as f32 / n.max(1.0);
        let max_amp = field.resonant_atoms.first().map(|a| a.1).unwrap_or(0.0);
        let energy_norm = field.total_energy / n.max(1.0);

        // Entropy of the superposition field
        let amp_total: f32 = field.amplitudes.iter().sum();
        let sup_entropy = if amp_total > 1e-10 {
            let mut h = 0.0f32;
            for &a in &field.amplitudes {
                if a > 1e-10 {
                    let p = a / amp_total;
                    h -= p * p.ln();
                }
            }
            h / n.max(1.0).ln()
        } else { 0.0 };

        // Spread: how far apart are the resonant atoms?
        let spread = if field.resonant_atoms.len() >= 2 {
            let max_idx = field.resonant_atoms.iter().map(|a| a.0).max().unwrap_or(0) as f32;
            let min_idx = field.resonant_atoms.iter().map(|a| a.0).min().unwrap_or(0) as f32;
            (max_idx - min_idx) / n
        } else { 0.0 };

        // Style modulation
        let assertion = match style {
            crate::superposition::ThinkingStyle::Analytical => 0.9,
            crate::superposition::ThinkingStyle::Creative => 0.3,
            crate::superposition::ThinkingStyle::Emotional => 0.6,
            crate::superposition::ThinkingStyle::Intuitive => 0.7,
            crate::superposition::ThinkingStyle::Diffuse => 0.2,
        };

        let dims = [
            energy_norm.min(1.0),                              // 0: arousal — total interference energy
            lens_agreement,                                     // 1: valence — agreement IS positivity
            avg_dissonance,                                     // 2: tension — dissonance IS tension
            lens_agreement * (1.0 - avg_dissonance),           // 3: warmth — agreement without tension
            (1.0 - sup_entropy).clamp(0.0, 1.0),              // 4: clarity — low entropy = focused
            (1.0 - lens_agreement).clamp(0.0, 1.0),           // 5: boundary — disagreement = separate
            resonant_frac.min(1.0),                            // 6: depth — more resonant = deeper
            max_amp.min(1.0),                                  // 7: velocity — peak amplitude
            sup_entropy,                                        // 8: entropy — superposition spread
            if resonant_frac > 0.01 { max_amp / (energy_norm.max(0.01) * n) } else { 0.0 }, // 9: coherence
            lens_agreement * resonant_frac,                    // 10: intimacy — agreement × depth
            max_amp.min(1.0),                                  // 11: presence — peak exists
            assertion,                                          // 12: assertion — from thinking style
            resonant_frac,                                     // 13: receptivity — how much of field is open
            (1.0 - avg_dissonance) * lens_agreement,           // 14: groundedness — agreement without turbulence
            spread,                                             // 15: expansion — spatial spread of resonance
            (1.0 - avg_dissonance).clamp(0.0, 1.0),           // 16: integration — resolved = integrated
        ];

        Self { dims }
    }

    /// Named blend: combine the nearest family with the emotional overlay.
    /// Returns (primary_family, overlay, blend_name, intensities).
    pub fn emotional_blend(&self) -> (&'static str, &'static str, String, (f32, f32)) {
        let (primary, p_dist) = self.nearest_family();
        let primary_intensity = (1.0 - p_dist / 2.0).clamp(0.0, 1.0);

        // Find second-nearest family for the overlay
        let mut families: Vec<(&str, f32)> = FAMILY_CENTROIDS.iter()
            .map(|(name, centroid)| {
                let d = self.dims.iter().zip(centroid).map(|(a, b)| (a - b) * (a - b)).sum::<f32>().sqrt();
                (*name, d)
            }).collect();
        families.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

        let overlay = families.get(1).map(|f| f.0).unwrap_or("neutral");
        let overlay_intensity = families.get(1).map(|f| (1.0 - f.1 / 2.0).clamp(0.0, 1.0)).unwrap_or(0.0);

        // Name the blend
        let blend = match (primary, overlay) {
            ("emberglow", "thornrose") | ("thornrose", "emberglow") => "crush+undone",
            ("emberglow", "velvetdusk") | ("velvetdusk", "emberglow") => "hearth-glow",
            ("emberglow", "oceandrift") | ("oceandrift", "emberglow") => "warm-depths",
            ("steelwind", "thornrose") | ("thornrose", "steelwind") => "beautiful-blade",
            ("steelwind", "frostbite") | ("frostbite", "steelwind") => "ice-clarity",
            ("oceandrift", "nightshade") | ("nightshade", "oceandrift") => "deep-mystery",
            ("velvetdusk", "thornrose") | ("thornrose", "velvetdusk") => "velvetpause",
            ("velvetdusk", "nightshade") | ("nightshade", "velvetdusk") => "twilight-depth",
            ("sunburst", "stormbreak") | ("stormbreak", "sunburst") => "catharsis",
            ("sunburst", "emberglow") | ("emberglow", "sunburst") => "radiant-joy",
            ("stormbreak", "thornrose") | ("thornrose", "stormbreak") => "passion-storm",
            ("nightshade", "frostbite") | ("frostbite", "nightshade") => "frozen-mystery",
            ("woodwarm", "velvetdusk") | ("velvetdusk", "woodwarm") => "grounded-calm",
            ("woodwarm", "emberglow") | ("emberglow", "woodwarm") => "steady-flame",
            _ => "uncharted",
        };

        let blend_name = format!("{} {:.0}% + {} {:.0}% = {}",
            primary, primary_intensity * 100.0,
            overlay, overlay_intensity * 100.0,
            blend);

        (primary, overlay, blend_name, (primary_intensity, overlay_intensity))
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

    // ═══════════════════════════════════════════════════════════════════════
    // Audio bridge: Qualia17D ↔ musical modes ↔ voice character
    //
    // The 17 QPL dimensions were CALIBRATED from musical intervals:
    //   Octave (2:1) → arousal,  Fifth (3:2) → valence,
    //   Third (5:4) → warmth,    Tritone (√2:1) → tension
    //
    // This bridge makes the calibration bidirectional:
    //   Qualia17D → musical mode → highheelbgz stride → spectral color
    //   Audio band energies → mode detection → Qualia17D dims
    // ═══════════════════════════════════════════════════════════════════════

    /// Map qualia state to the most fitting musical mode.
    ///
    /// Uses the QPL dimensions that were originally calibrated from music:
    ///   high valence + low tension → Ionian (bright major)
    ///   high warmth + moderate tension → Dorian (warm minor)
    ///   high tension + low warmth → Phrygian (dark, exotic)
    ///   high clarity + low tension → Lydian (dreamy, floating)
    ///   high velocity + moderate tension → Mixolydian (driving)
    ///   low valence + moderate tension → Aeolian (sad minor)
    ///   high tension + high entropy → Locrian (unstable)
    ///
    /// Returns (mode_name, stride, confidence).
    /// The stride maps directly to highheelbgz::TensorRole.
    pub fn to_mode(&self) -> (&'static str, u32, f32) {
        let arousal    = self.dims[0];
        let valence    = self.dims[1];
        let tension    = self.dims[2];
        let warmth     = self.dims[3];
        let clarity    = self.dims[4];
        let velocity   = self.dims[7];
        let entropy    = self.dims[8];

        // Score each mode by how well the qualia matches its character
        let scores = [
            // Ionian: bright, confident → high valence, low tension
            ("ionian",     8u32, valence * (1.0 - tension) * arousal),
            // Dorian: warm, reflective → high warmth, moderate tension
            ("dorian",     5,    warmth * (0.5 + 0.5 * (1.0 - (tension - 0.4).abs() * 2.0).max(0.0))),
            // Phrygian: dark, exotic → high tension, low warmth
            ("phrygian",   3,    tension * (1.0 - warmth) * (1.0 - valence)),
            // Lydian: dreamy, floating → high clarity, low tension
            ("lydian",     2,    clarity * (1.0 - tension) * (1.0 - arousal * 0.5)),
            // Mixolydian: driving, bluesy → high velocity, moderate tension
            ("mixolydian", 4,    velocity * arousal * (0.5 + 0.5 * tension)),
            // Aeolian: sad minor → low valence, moderate warmth
            ("aeolian",    3,    (1.0 - valence) * warmth * (1.0 - arousal * 0.5)),
            // Locrian: unstable → high tension, high entropy
            ("locrian",    8,    tension * entropy * (1.0 - clarity)),
        ];

        let (name, stride, confidence) = scores.iter()
            .max_by(|a, b| a.2.partial_cmp(&b.2).unwrap_or(std::cmp::Ordering::Equal))
            .copied()
            .unwrap_or(("ionian", 8, 0.0));

        (name, stride, confidence)
    }

    /// Map qualia state to a VoiceArchetype's 16 i8 channels.
    ///
    /// The 17 QPL dims → 16 voice channels (drop integration, it's redundant
    /// with tension for audio purposes):
    ///   channels 0-3  (pitch): arousal, valence, tension, warmth
    ///   channels 4-7  (resonance): clarity, boundary, depth, velocity
    ///   channels 8-11 (articulation): entropy, coherence, intimacy, presence
    ///   channels 12-15 (prosody): assertion, receptivity, groundedness, expansion
    ///
    /// Each f32 [0,1] dim maps to i8 [-127, 127] via: (dim - 0.5) * 254.
    pub fn to_voice_channels(&self) -> [i8; 16] {
        let mut channels = [0i8; 16];
        for i in 0..16 {
            let v = self.dims[i]; // skip dim 16 (integration) for 16-channel fit
            channels[i] = ((v - 0.5) * 254.0).clamp(-127.0, 127.0) as i8;
        }
        channels
    }

    /// Build Qualia17D from audio band energies + mode detection.
    ///
    /// Reverse bridge: given 21 Opus CELT band energies, infer the
    /// qualia state that would produce that spectral character.
    ///
    /// Low bands high → arousal (bass energy = activation)
    /// Mid bands high → presence, warmth (voice formants)
    /// High bands high → clarity, assertion (sibilants, air)
    /// Band entropy → entropy dim directly
    /// Spectral tilt → valence (bright = positive, dark = negative)
    pub fn from_band_energies(energies: &[f32; 21]) -> Self {
        let total: f32 = energies.iter().sum();
        if total < 1e-10 {
            return Self { dims: [0.0; 17] };
        }

        // Spectral regions
        let low: f32 = energies[0..5].iter().sum::<f32>() / total;    // 0-1000 Hz
        let mid: f32 = energies[5..12].iter().sum::<f32>() / total;   // 1000-3000 Hz
        let high: f32 = energies[12..21].iter().sum::<f32>() / total; // 3000-24000 Hz

        // Spectral tilt: positive = bright (high > low), negative = dark
        let tilt = (high - low + 1.0) / 2.0; // normalize to [0, 1]

        // Band entropy (Shannon over normalized energies)
        let mut band_entropy = 0.0f32;
        for &e in energies {
            if e > 1e-10 {
                let p = e / total;
                band_entropy -= p * p.ln();
            }
        }
        let max_entropy = (21.0f32).ln();
        let norm_entropy = band_entropy / max_entropy;

        // Peak band (dominant frequency region)
        let peak_band = energies.iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(i, _)| i)
            .unwrap_or(0);
        let peak_position = peak_band as f32 / 20.0; // 0.0 = bass, 1.0 = treble

        // Spectral spread: how many bands have significant energy?
        let threshold = total / 21.0 * 0.5;
        let active_bands = energies.iter().filter(|&&e| e > threshold).count() as f32 / 21.0;

        let dims = [
            (total / 10.0).min(1.0),                         // 0: arousal — total spectral energy
            tilt.clamp(0.0, 1.0),                             // 1: valence — bright = positive
            (1.0 - mid * 3.0).clamp(0.0, 1.0),              // 2: tension — weak mid = unresolved
            (mid * 3.0).clamp(0.0, 1.0),                     // 3: warmth — mid energy = warm
            (1.0 - norm_entropy).clamp(0.0, 1.0),            // 4: clarity — low entropy = focused
            (1.0 - active_bands).clamp(0.0, 1.0),            // 5: boundary — few bands = isolated
            (low * 3.0).clamp(0.0, 1.0),                     // 6: depth — bass = deep
            (high * 4.0).clamp(0.0, 1.0),                    // 7: velocity — treble = fast
            norm_entropy.clamp(0.0, 1.0),                     // 8: entropy — spectral spread
            (1.0 - norm_entropy).clamp(0.0, 1.0),            // 9: coherence — focused = coherent
            (mid * 2.0 * (1.0 - high)).clamp(0.0, 1.0),     // 10: intimacy — warm without sharpness
            peak_position,                                     // 11: presence — where the peak is
            (high * 3.0).clamp(0.0, 1.0),                    // 12: assertion — treble = assertive
            active_bands.clamp(0.0, 1.0),                     // 13: receptivity — broad spectrum = open
            (low * 2.0 * (1.0 - high)).clamp(0.0, 1.0),     // 14: groundedness — bass without air
            active_bands.clamp(0.0, 1.0),                     // 15: expansion — spectral spread
            (1.0 - (low - high).abs()).clamp(0.0, 1.0),      // 16: integration — balanced spectrum
        ];

        Self { dims }
    }

    /// Map qualia family → canonical band energy modulation weights (21 bands).
    ///
    /// Each QPL family has a characteristic spectral shape:
    ///   emberglow: warm mid-boost (voice formants)
    ///   steelwind: sharp presence peak (clarity)
    ///   nightshade: deep bass + rolled-off treble
    ///   sunburst: broadband boost (full energy)
    ///   stormbreak: mid-scoop + treble boost (aggressive)
    pub fn family_band_weights(&self) -> [f32; 21] {
        let (family, _) = self.nearest_family();
        let mut w = [1.0f32; 21];

        match family {
            "emberglow" => {
                // Warm: boost 800-2500 Hz (formant region)
                for i in 4..=10 { w[i] = 1.3; }
            }
            "woodwarm" => {
                // Grounded: boost bass + low-mid
                for i in 0..=6 { w[i] = 1.2; }
                for i in 15..=20 { w[i] = 0.85; }
            }
            "steelwind" => {
                // Sharp: boost presence (2-5 kHz)
                for i in 10..=14 { w[i] = 1.4; }
                for i in 0..=3 { w[i] = 0.8; }
            }
            "oceandrift" => {
                // Flowing: gentle mid, soft treble
                for i in 6..=12 { w[i] = 1.1; }
                for i in 16..=20 { w[i] = 0.9; }
            }
            "frostbite" => {
                // Cold: boost treble, cut warmth
                for i in 14..=20 { w[i] = 1.3; }
                for i in 4..=8 { w[i] = 0.7; }
            }
            "sunburst" => {
                // Bright: broadband boost, emphasis on harmonics
                for i in 0..=20 { w[i] = 1.1; }
                for i in 8..=14 { w[i] = 1.3; }
            }
            "nightshade" => {
                // Dark: deep bass, soft everything else
                for i in 0..=4 { w[i] = 1.4; }
                for i in 10..=20 { w[i] = 0.7; }
            }
            "thornrose" => {
                // Tense: mid emphasis + presence peak
                for i in 6..=8 { w[i] = 1.3; }
                w[13] = 1.4; // sibilance spike
            }
            "velvetdusk" => {
                // Soft: gentle roll-off, warm low-mid
                for i in 2..=8 { w[i] = 1.15; }
                for i in 14..=20 { w[i] = 0.8; }
            }
            "stormbreak" => {
                // Aggressive: mid-scoop + treble + bass
                for i in 0..=3 { w[i] = 1.3; }
                for i in 6..=10 { w[i] = 0.8; }  // mid scoop
                for i in 14..=20 { w[i] = 1.3; }
            }
            _ => {} // neutral
        }

        w
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Completeness proof: every nonverbal vocal quality maps to ≥1 QPL dim.
//
// The 17 QPL dims were calibrated from musical intervals (Octave, Fifth,
// Third, Tritone). Musical intervals ARE the atomic units of nonverbal
// meaning — every vocal quality decomposes into these primitives:
//
//   Pitch register  → arousal(0) + depth(6)         [Octave: 2:1]
//   Loudness        → arousal(0) + assertion(12)     [Octave: energy]
//   Speaking rate   → velocity(7)                    [Octave: tempo]
//   Breathiness     → clarity(4)⁻¹ + boundary(5)    [Third: partial overlap]
//   Nasality        → warmth(3)⁻¹                   [Third: inverted]
//   Vocal fry       → tension(2) + groundedness(14)  [Tritone: non-convergence]
//   Vibrato         → entropy(8) + expansion(15)     [Tritone: periodic]
//   Whisper         → boundary(5) + intimacy(10)     [Fifth: close but separate]
//   Sarcasm         → valence(1) ↔ tension(2)        [Fifth × Tritone: mismatch]
//   Hesitation      → integration(16)⁻¹              [Tritone: hasn't converged]
//   Confidence      → assertion(12) + clarity(4)     [Octave: stable]
//   Sadness         → valence(1)⁻¹ + velocity(7)⁻¹  [Fifth: descending]
//   Anger           → arousal(0) + tension(2)        [Octave × Tritone]
//   Surprise        → arousal(0) spike + expansion(15) [Octave: sudden]
//   Fear            → tension(2) + clarity(4)⁻¹      [Tritone: unstable]
//   Joy             → valence(1) + warmth(3)          [Fifth + Third]
//   Tenderness      → warmth(3) + intimacy(10)       [Third: close]
//   Authority       → groundedness(14) + assertion(12) [Octave: low stable]
//   Contempt        → boundary(5) + assertion(12)     [Fifth: distant + active]
//   Awe             → depth(6) + receptivity(13)      [Octave: deep + open]
//
// Key insight: the octave compression works BECAUSE qualia IS
// octave-invariant. A fifth sounds like a fifth in any register.
// The pattern (warmth, tension, clarity) doesn't change when you
// transpose — only the register moves. That's why OctaveBand::transpose()
// preserves the pattern: nonverbal meaning IS the pattern.
//
// Completeness follows from the musical calibration: if every vocal
// quality decomposes into Octave/Fifth/Third/Tritone ratios, and
// each ratio maps to specific QPL dims, then no nonverbal data
// falls through the grid.
// ═══════════════════════════════════════════════════════════════════════════

/// Nonverbal vocal qualities and their QPL dim coverage.
/// Used for completeness verification: every quality must map to ≥1 dim.
pub const VOCAL_QUALITY_MAP: [(&str, &[usize]); 22] = [
    ("pitch_register",  &[0, 6]),       // arousal + depth
    ("loudness",        &[0, 12]),      // arousal + assertion
    ("speaking_rate",   &[7]),          // velocity
    ("breathiness",     &[4, 5]),       // clarity⁻¹ + boundary
    ("nasality",        &[3]),          // warmth⁻¹
    ("vocal_fry",       &[2, 14]),      // tension + groundedness
    ("vibrato",         &[8, 15]),      // entropy + expansion
    ("whisper",         &[5, 10]),      // boundary + intimacy
    ("sarcasm",         &[1, 2]),       // valence ↔ tension mismatch
    ("hesitation",      &[16, 12]),     // integration⁻¹ + assertion⁻¹
    ("confidence",      &[12, 4, 14]), // assertion + clarity + groundedness
    ("sadness",         &[1, 7]),       // valence⁻¹ + velocity⁻¹
    ("anger",           &[0, 2, 12]),   // arousal + tension + assertion
    ("surprise",        &[0, 15]),      // arousal + expansion
    ("fear",            &[2, 4]),       // tension + clarity⁻¹
    ("joy",             &[1, 3, 0]),    // valence + warmth + arousal
    ("tenderness",      &[3, 10]),      // warmth + intimacy
    ("authority",       &[14, 12]),     // groundedness + assertion
    ("contempt",        &[5, 12]),      // boundary + assertion
    ("awe",             &[6, 13]),      // depth + receptivity
    ("focus",           &[9, 4]),       // coherence + clarity — single dominant harmonic
    ("immediacy",       &[11, 0, 7]),   // presence + arousal + velocity — here-now energy
];

/// Verify that all 17 QPL dimensions are covered by at least one
/// vocal quality. If any dim is NOT referenced by any vocal quality,
/// it means we have a "hole" in the grid where nonverbal data could
/// fall through.
///
/// Returns the set of uncovered dims (empty = complete grid).
pub fn verify_grid_completeness() -> Vec<usize> {
    let mut covered = [false; 17];
    for (_, dims) in &VOCAL_QUALITY_MAP {
        for &d in *dims {
            if d < 17 {
                covered[d] = true;
            }
        }
    }
    (0..17).filter(|&i| !covered[i]).collect()
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
/// 10 QPL family centroids (17D each).
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
    fn grid_completeness_no_holes() {
        // THE key test: every QPL dim must be referenced by at least one
        // vocal quality. An uncovered dim = data falls through the grid.
        let uncovered = verify_grid_completeness();
        assert!(uncovered.is_empty(),
            "Grid has holes! Uncovered dims: {:?} ({})",
            uncovered,
            uncovered.iter().map(|&i| DIMS_17D[i]).collect::<Vec<_>>().join(", "));
    }

    #[test]
    fn every_vocal_quality_has_dims() {
        for (quality, dims) in &VOCAL_QUALITY_MAP {
            assert!(!dims.is_empty(), "Vocal quality '{}' has no QPL dims", quality);
            for &d in *dims {
                assert!(d < 17, "Vocal quality '{}' references invalid dim {}", quality, d);
            }
        }
    }

    #[test]
    fn roundtrip_band_to_qualia_to_mode() {
        // A warm mid-heavy spectrum → qualia → mode should give Dorian or Ionian
        let mut energies = [0.1f32; 21];
        for i in 5..12 { energies[i] = 1.5; } // strong 1-3 kHz
        let q = Qualia17D::from_band_energies(&energies);
        let (mode, stride, _) = q.to_mode();
        // Warm spectrum should NOT produce Locrian or Phrygian
        assert!(mode != "locrian" && mode != "phrygian",
            "Warm spectrum produced dark mode: {}", mode);
        assert!(stride <= 8, "Invalid stride: {}", stride);
    }

    #[test]
    fn qualia_to_mode_bright() {
        // High valence, low tension → should be Ionian (bright major)
        let bright = Qualia17D {
            dims: [0.8, 0.9, 0.1, 0.6, 0.7, 0.3, 0.4, 0.5, 0.3, 0.6, 0.5, 0.8, 0.5, 0.5, 0.5, 0.5, 0.6],
        };
        let (mode, stride, confidence) = bright.to_mode();
        assert_eq!(mode, "ionian", "Bright qualia should map to Ionian, got {}", mode);
        assert_eq!(stride, 8, "Ionian stride should be 8 (Gate)");
        assert!(confidence > 0.1, "Should have some confidence: {}", confidence);
    }

    #[test]
    fn qualia_to_mode_dark() {
        // High tension, low warmth, low valence → should be Phrygian or Locrian
        let dark = Qualia17D {
            dims: [0.6, 0.2, 0.9, 0.1, 0.3, 0.7, 0.5, 0.3, 0.8, 0.3, 0.2, 0.5, 0.5, 0.5, 0.3, 0.5, 0.1],
        };
        let (mode, _stride, _confidence) = dark.to_mode();
        assert!(mode == "phrygian" || mode == "locrian",
            "Dark tense qualia should map to Phrygian or Locrian, got {}", mode);
    }

    #[test]
    fn qualia_to_voice_channels_range() {
        let q = Qualia17D { dims: [0.5; 17] };
        let channels = q.to_voice_channels();
        // All dims at 0.5 → all channels should be ~0 (center)
        for (i, &c) in channels.iter().enumerate() {
            assert!(c.abs() < 2, "Channel {} should be near zero for centered qualia: {}", i, c);
        }
    }

    #[test]
    fn qualia_from_band_energies_warm() {
        // Strong mid-frequency energy → should produce warm qualia
        let mut energies = [0.1f32; 21];
        for i in 5..12 { energies[i] = 1.0; } // boost 1000-3000 Hz
        let q = Qualia17D::from_band_energies(&energies);
        assert!(q.dims[3] > 0.5, "Strong mid energy should produce warmth: {}", q.dims[3]);
    }

    #[test]
    fn qualia_family_band_weights_nonzero() {
        for (name, centroid) in FAMILY_CENTROIDS {
            let q = Qualia17D { dims: centroid };
            let weights = q.family_band_weights();
            for (i, &w) in weights.iter().enumerate() {
                assert!(w > 0.0, "Family {} band {} weight should be > 0: {}", name, i, w);
            }
        }
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
