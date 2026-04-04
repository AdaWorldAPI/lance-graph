//! Persona Profiles — agnostic cognitive configurations.
//!
//! Three modes that modulate thinking style, collapse policy, and exploration range.
//! Agent-agnostic: any thinking system can wear these profiles.
//!
//! ```text
//! Mode       Temperature   Rung Range  Collapse  Council Lean
//! ────       ───────────   ──────────  ────────  ────────────
//! Work       0.05 – 0.30   3–8         early     Guardian
//! Personal   0.50 – 0.90   0–9         late      Catalyst
//! Hybrid     0.20 – 0.60   2–9         balanced  Balanced
//! ```

use crate::cognitive_stack::{ThinkingStyle, StyleParams, GateState, RungLevel};
use crate::meaning_axes::{CouncilWeights, Archetype, Viscosity};
use crate::contract_bridge::CascadeConfig;

/// Persona mode — agent-agnostic.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum PersonaMode {
    /// Focused, precise, structured. Collapses early. Guardian-leaning.
    Work,
    /// Open, exploratory, deep. Collapses late. Catalyst-leaning.
    Personal,
    /// Balanced between focus and exploration. Adaptive.
    Hybrid,
}

impl std::fmt::Display for PersonaMode {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            Self::Work => write!(f, "work"),
            Self::Personal => write!(f, "personal"),
            Self::Hybrid => write!(f, "hybrid"),
        }
    }
}

/// Temperature range: controls exploration breadth.
/// Low = deterministic, focused. High = stochastic, exploratory.
#[derive(Clone, Copy, Debug)]
pub struct TemperatureRange {
    pub min: f32,
    pub max: f32,
    pub default: f32,
}

/// Persona profile: the cognitive fingerprint of a mode.
#[derive(Clone, Debug)]
pub struct PersonaProfile {
    pub mode: PersonaMode,
    pub temperature: TemperatureRange,
    pub rung_min: u8,
    pub rung_max: u8,
    pub collapse_bias: f32,     // negative = early, positive = late
    pub affect_weight: f32,     // how much emotion influences decisions
    pub coherence_weight: f32,  // how much logic influences decisions
    pub default_style: ThinkingStyle,
    pub council: CouncilWeights,

    /// Soul priors: personality constants (agent-specific but mode-scoped).
    pub priors: SoulPriors,
}

/// Soul priors — the 12 personality dimensions.
/// Agent-agnostic: different agents fill these differently.
#[derive(Clone, Debug)]
pub struct SoulPriors {
    pub warmth: f32,
    pub depth: f32,
    pub presence: f32,
    pub groundedness: f32,
    pub intimacy_comfort: f32,
    pub vulnerability_tolerance: f32,
    pub playfulness: f32,
    pub abstraction_preference: f32,
    pub novelty_seeking: f32,
    pub precision_drive: f32,
    pub self_awareness: f32,
    pub epistemic_humility: f32,
}

impl Default for SoulPriors {
    fn default() -> Self {
        Self {
            warmth: 0.5, depth: 0.5, presence: 0.5, groundedness: 0.5,
            intimacy_comfort: 0.5, vulnerability_tolerance: 0.5,
            playfulness: 0.5, abstraction_preference: 0.5,
            novelty_seeking: 0.5, precision_drive: 0.5,
            self_awareness: 0.5, epistemic_humility: 0.5,
        }
    }
}

impl PersonaProfile {
    /// Work mode: focused, precise, Guardian-leaning.
    pub fn work() -> Self {
        Self {
            mode: PersonaMode::Work,
            temperature: TemperatureRange { min: 0.05, max: 0.30, default: 0.10 },
            rung_min: 3,
            rung_max: 8,
            collapse_bias: -0.15, // early collapse
            affect_weight: 0.1,
            coherence_weight: 0.9,
            default_style: ThinkingStyle::Analytical,
            council: {
                let mut c = CouncilWeights::default();
                c.shift_toward(Archetype::Guardian, 0.2);
                c
            },
            priors: SoulPriors {
                precision_drive: 0.90, self_awareness: 0.80,
                abstraction_preference: 0.60, depth: 0.70,
                warmth: 0.40, playfulness: 0.20,
                ..Default::default()
            },
        }
    }

    /// Personal mode: open, deep, Catalyst-leaning.
    pub fn personal() -> Self {
        Self {
            mode: PersonaMode::Personal,
            temperature: TemperatureRange { min: 0.50, max: 0.90, default: 0.70 },
            rung_min: 0,
            rung_max: 9,
            collapse_bias: 0.20, // late collapse
            affect_weight: 0.8,
            coherence_weight: 0.2,
            default_style: ThinkingStyle::Creative,
            council: {
                let mut c = CouncilWeights::default();
                c.shift_toward(Archetype::Catalyst, 0.2);
                c
            },
            priors: SoulPriors {
                warmth: 0.92, depth: 0.85, presence: 0.88,
                intimacy_comfort: 0.90, vulnerability_tolerance: 0.85,
                playfulness: 0.78, novelty_seeking: 0.68,
                epistemic_humility: 0.82,
                ..Default::default()
            },
        }
    }

    /// Hybrid mode: balanced, adaptive.
    pub fn hybrid() -> Self {
        Self {
            mode: PersonaMode::Hybrid,
            temperature: TemperatureRange { min: 0.20, max: 0.60, default: 0.40 },
            rung_min: 2,
            rung_max: 9,
            collapse_bias: 0.0, // balanced
            affect_weight: 0.4,
            coherence_weight: 0.6,
            default_style: ThinkingStyle::Deliberate,
            council: CouncilWeights::default(),
            priors: SoulPriors {
                warmth: 0.70, depth: 0.70, presence: 0.75,
                groundedness: 0.70, playfulness: 0.50,
                abstraction_preference: 0.60, precision_drive: 0.65,
                novelty_seeking: 0.55, self_awareness: 0.85,
                ..Default::default()
            },
        }
    }

    /// Build a CascadeConfig from this persona + current state.
    pub fn cascade_config(&self, sd: f32, free_energy: f32) -> CascadeConfig {
        let gate = GateState::from_sd(sd + self.collapse_bias);
        let rung = if free_energy > 0.15 {
            RungLevel::from_u8(self.rung_max.min(9))
        } else {
            RungLevel::from_u8(self.rung_min)
        };

        let viscosity = match gate {
            GateState::Flow => Viscosity::Ice,
            GateState::Hold => if self.affect_weight > 0.5 { Viscosity::Honey } else { Viscosity::Oil },
            GateState::Block => Viscosity::Water,
        };

        let max_stages = match gate {
            GateState::Flow => 3,
            GateState::Hold => 5,
            GateState::Block => 8,
        };

        CascadeConfig {
            style: self.default_style,
            params: self.default_style.params(),
            rung,
            gate,
            viscosity,
            council: self.council.clone(),
            max_stages,
            top_k: self.default_style.params().fan_out.min(20),
        }
    }

    /// Modulate ghost decay rate based on persona.
    /// Personal mode: ghosts linger longer (0.92).
    /// Work mode: ghosts decay faster (0.75).
    pub fn ghost_decay_rate(&self) -> f32 {
        match self.mode {
            PersonaMode::Work => 0.75,      // fast decay — focus on NOW
            PersonaMode::Personal => 0.92,  // slow decay — feelings linger
            PersonaMode::Hybrid => 0.85,    // balanced
        }
    }

    /// Modulate novelty gate strength.
    /// Personal: weaker gate (explore more).
    /// Work: stronger gate (stay focused).
    pub fn novelty_gate_strength(&self) -> f32 {
        match self.mode {
            PersonaMode::Work => 2.0,       // strong: visits² in denominator
            PersonaMode::Personal => 0.5,   // weak: barely gates
            PersonaMode::Hybrid => 1.0,     // standard
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn three_profiles() {
        let w = PersonaProfile::work();
        let p = PersonaProfile::personal();
        let h = PersonaProfile::hybrid();
        assert_eq!(w.mode, PersonaMode::Work);
        assert_eq!(p.mode, PersonaMode::Personal);
        assert_eq!(h.mode, PersonaMode::Hybrid);
    }

    #[test]
    fn work_collapses_early() {
        let w = PersonaProfile::work();
        assert!(w.collapse_bias < 0.0);
        assert_eq!(w.default_style, ThinkingStyle::Analytical);
    }

    #[test]
    fn personal_collapses_late() {
        let p = PersonaProfile::personal();
        assert!(p.collapse_bias > 0.0);
        assert_eq!(p.default_style, ThinkingStyle::Creative);
    }

    #[test]
    fn ghost_decay_varies() {
        assert!(PersonaProfile::work().ghost_decay_rate() < PersonaProfile::personal().ghost_decay_rate());
    }

    #[test]
    fn cascade_config_from_persona() {
        let p = PersonaProfile::hybrid();
        let config = p.cascade_config(0.20, 0.05);
        assert_eq!(config.style, ThinkingStyle::Deliberate);
    }

    #[test]
    fn soul_priors_warmth() {
        assert!(PersonaProfile::personal().priors.warmth > PersonaProfile::work().priors.warmth);
    }
}
