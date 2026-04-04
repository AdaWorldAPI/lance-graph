//! Ghosts: Friston's Free Energy as persistent cognitive priors.
//!
//! Ghosts are lingering emotional/cognitive traces from past thoughts
//! that decay asymptotically but never fully vanish. They bias
//! future perception — not by changing the topology, but by
//! pre-weighting which atoms the cascade visits first.
//!
//! ```text
//! Friston: The brain minimizes surprise by maintaining predictions.
//! Ghost:   A prediction from a past thought that persists as a prior.
//!          The cascade visits ghost-weighted atoms FIRST, reducing
//!          surprise when familiar patterns recur.
//!
//! Autocomplete: Ghost field IS the prediction cache.
//!   Past: "love" activated atoms [29, 85, 42] with stormbreak qualia.
//!   Now:  "love" appears again → ghosts bias toward [29, 85, 42].
//!   Result: Faster convergence. Less surprise. Autocomplete.
//!   But if the context is DIFFERENT, ghosts create prediction ERROR.
//!   Error = surprise = FREE ENERGY = signal to update the ghost.
//! ```
//!
//! The 8 ghost types from ladybug-rs/src/qualia/felt_parse.rs:
//! ```text
//! Affinity   — lingering pull toward a concept/person/thing
//! Epiphany   — residual clarity from a past insight
//! Somatic    — body-felt echo (tension, warmth, chill)
//! Staunen    — persistent wonder/awe
//! Wisdom     — deep knowing that colors all future perception
//! Thought    — a thought that won't let go (rumination or focus)
//! Grief      — loss that reshapes the topology
//! Boundary   — a limit discovered, still felt
//! ```

/// Ghost types — lingering cognitive traces with asymptotic decay.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum GhostType {
    Affinity,    // pull toward connection
    Epiphany,    // residual clarity
    Somatic,     // body-felt echo
    Staunen,     // persistent wonder
    Wisdom,      // deep knowing
    Thought,     // won't let go
    Grief,       // loss reshapes topology
    Boundary,    // discovered limit
}

impl std::fmt::Display for GhostType {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            Self::Affinity => write!(f, "affinity"),
            Self::Epiphany => write!(f, "epiphany"),
            Self::Somatic => write!(f, "somatic"),
            Self::Staunen => write!(f, "staunen"),
            Self::Wisdom => write!(f, "wisdom"),
            Self::Thought => write!(f, "thought"),
            Self::Grief => write!(f, "grief"),
            Self::Boundary => write!(f, "boundary"),
        }
    }
}

/// A single ghost: a lingering trace at a specific atom.
#[derive(Clone, Debug)]
pub struct Ghost {
    pub atom: u16,
    pub ghost_type: GhostType,
    pub intensity: f32,    // 0.0 = fully decayed, 1.0 = just created
    pub created_at: u64,   // thought cycle when created
    pub source_text: String, // what created this ghost (for debug)
}

/// The ghost field: all active ghosts across the atom space.
/// This IS Friston's prediction / the autocomplete cache.
#[derive(Clone, Debug)]
pub struct GhostField {
    ghosts: Vec<Ghost>,
    /// Decay rate per thought cycle. 0.9 = slow decay, 0.5 = fast.
    pub decay_rate: f32,
    /// Current thought cycle counter.
    pub cycle: u64,
}

impl GhostField {
    pub fn new() -> Self {
        Self {
            ghosts: Vec::new(),
            decay_rate: 0.85, // slow decay — ghosts linger
            cycle: 0,
        }
    }

    /// Create ghosts from a completed thought.
    /// The cascade's resonant atoms + cognitive markers → ghost types.
    pub fn imprint(
        &mut self,
        resonant_atoms: &[(u16, f32)],
        style: &crate::superposition::ThinkingStyle,
        staunen: f32,
        wisdom: f32,
        dissonance: f32,
        source: &str,
    ) {
        self.cycle += 1;

        for &(atom, amplitude) in resonant_atoms.iter().take(10) {
            // Determine ghost type from cognitive context
            let ghost_type = if staunen > 0.5 {
                GhostType::Staunen
            } else if wisdom > 0.3 {
                GhostType::Wisdom
            } else if dissonance > 0.3 {
                GhostType::Grief // unresolved tension becomes grief-ghost
            } else {
                match style {
                    crate::superposition::ThinkingStyle::Emotional => GhostType::Somatic,
                    crate::superposition::ThinkingStyle::Intuitive => GhostType::Affinity,
                    crate::superposition::ThinkingStyle::Analytical => GhostType::Thought,
                    crate::superposition::ThinkingStyle::Creative => GhostType::Epiphany,
                    crate::superposition::ThinkingStyle::Diffuse => GhostType::Boundary,
                }
            };

            self.ghosts.push(Ghost {
                atom,
                ghost_type,
                intensity: amplitude.min(1.0),
                created_at: self.cycle,
                source_text: source.chars().take(50).collect(),
            });
        }
    }

    /// Get the ghost bias for an atom: how much past experience
    /// pre-weights this atom for the next cascade.
    /// Returns (total_bias, dominant_ghost_type).
    pub fn bias(&self, atom: u16) -> (f32, Option<GhostType>) {
        let mut total = 0.0f32;
        let mut dominant_type = None;
        let mut max_intensity = 0.0f32;

        for ghost in &self.ghosts {
            if ghost.atom != atom { continue; }

            // Asymptotic decay: intensity * decay_rate^(cycles_since_creation)
            let age = (self.cycle - ghost.created_at) as f32;
            let decayed = ghost.intensity * self.decay_rate.powf(age);

            if decayed < 0.001 { continue; } // effectively dead

            total += decayed;
            if decayed > max_intensity {
                max_intensity = decayed;
                dominant_type = Some(ghost.ghost_type);
            }
        }

        (total, dominant_type)
    }

    /// Get ghost-weighted energy vector: pre-bias for the next cascade.
    /// This IS the autocomplete prediction.
    pub fn prediction(&self, n_atoms: usize) -> Vec<f32> {
        let mut pred = vec![0.0f32; n_atoms];
        for ghost in &self.ghosts {
            if (ghost.atom as usize) >= n_atoms { continue; }
            let age = (self.cycle - ghost.created_at) as f32;
            let decayed = ghost.intensity * self.decay_rate.powf(age);
            if decayed > 0.001 {
                pred[ghost.atom as usize] += decayed;
            }
        }
        // Normalize to [0, 1] range
        let max_pred = pred.iter().cloned().fold(0.0f32, f32::max);
        if max_pred > 0.001 {
            for p in &mut pred { *p /= max_pred; }
        }
        pred
    }

    /// Free energy: the surprise between prediction and actual activation.
    /// High free energy = ghosts predicted wrong = UPDATE needed.
    /// Low free energy = ghosts predicted right = AUTOCOMPLETE works.
    pub fn free_energy(&self, actual_energy: &[f32]) -> f32 {
        let pred = self.prediction(actual_energy.len());
        // KL divergence approximation: sum of |pred - actual|
        let mut surprise = 0.0f32;
        for (p, a) in pred.iter().zip(actual_energy) {
            let diff = (p - a).abs();
            surprise += diff;
        }
        surprise / actual_energy.len().max(1) as f32
    }

    /// Prune dead ghosts (intensity below threshold after decay).
    pub fn prune(&mut self) {
        self.ghosts.retain(|g| {
            let age = (self.cycle - g.created_at) as f32;
            let decayed = g.intensity * self.decay_rate.powf(age);
            decayed > 0.001
        });
    }

    /// Number of active ghosts.
    pub fn active_count(&self) -> usize {
        self.ghosts.iter().filter(|g| {
            let age = (self.cycle - g.created_at) as f32;
            g.intensity * self.decay_rate.powf(age) > 0.001
        }).count()
    }

    /// Ghost summary for display.
    pub fn summary(&self) -> Vec<(u16, GhostType, f32)> {
        let mut active: Vec<(u16, GhostType, f32)> = self.ghosts.iter()
            .filter_map(|g| {
                let age = (self.cycle - g.created_at) as f32;
                let decayed = g.intensity * self.decay_rate.powf(age);
                if decayed > 0.01 { Some((g.atom, g.ghost_type, decayed)) } else { None }
            })
            .collect();
        active.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap());
        active.dedup_by_key(|a| a.0);
        active
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ghost_decays() {
        let mut field = GhostField::new();
        field.imprint(
            &[(42, 0.8)],
            &crate::superposition::ThinkingStyle::Intuitive,
            0.0, 0.0, 0.0, "test",
        );
        let (bias_0, _) = field.bias(42);
        assert!(bias_0 > 0.7);

        // Simulate 10 cycles
        field.cycle += 10;
        let (bias_10, _) = field.bias(42);
        assert!(bias_10 < bias_0); // decayed
        assert!(bias_10 > 0.1);   // but not dead
    }

    #[test]
    fn ghost_prediction_is_autocomplete() {
        let mut field = GhostField::new();
        field.imprint(
            &[(10, 0.9), (20, 0.7), (30, 0.5)],
            &crate::superposition::ThinkingStyle::Creative,
            0.0, 0.0, 0.0, "previous thought",
        );
        let pred = field.prediction(256);
        assert!(pred[10] > pred[20]); // 10 was strongest
        assert!(pred[20] > pred[30]); // 20 was next
        assert_eq!(pred[100], 0.0);   // 100 was never activated
    }

    #[test]
    fn free_energy_low_when_matching() {
        let mut field = GhostField::new();
        field.imprint(
            &[(10, 1.0)],
            &crate::superposition::ThinkingStyle::Analytical,
            0.0, 0.0, 0.0, "test",
        );
        // Actual matches prediction
        let mut actual = vec![0.0f32; 64];
        actual[10] = 1.0;
        let fe_match = field.free_energy(&actual);

        // Actual doesn't match
        let mut actual_diff = vec![0.0f32; 64];
        actual_diff[50] = 1.0;
        let fe_diff = field.free_energy(&actual_diff);

        assert!(fe_match < fe_diff); // matching = less surprise
    }

    #[test]
    fn staunen_creates_staunen_ghost() {
        let mut field = GhostField::new();
        field.imprint(
            &[(42, 0.8)],
            &crate::superposition::ThinkingStyle::Diffuse,
            0.8, 0.0, 0.0, "wonder",
        );
        let (_, ghost_type) = field.bias(42);
        assert_eq!(ghost_type, Some(GhostType::Staunen));
    }

    #[test]
    fn prune_removes_dead_ghosts() {
        let mut field = GhostField::new();
        field.imprint(
            &[(42, 0.01)], // very weak
            &crate::superposition::ThinkingStyle::Diffuse,
            0.0, 0.0, 0.0, "weak",
        );
        field.cycle += 100; // age a lot
        field.prune();
        assert_eq!(field.active_count(), 0);
    }
}
