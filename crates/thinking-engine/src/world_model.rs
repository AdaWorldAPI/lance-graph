//! WorldModelDto — the agent's complete situational awareness.
//!
//! One struct. Four sections. Every field explicit and typed.
//!
//! ```text
//! WorldModelDto {
//!   self_state:    how the agent sees itself
//!   user_state:    how the agent reads the other party (empathy)
//!   field_state:   the dynamics between self and user (gestalt)
//!   context_state: semantic profile of the current content
//! }
//! ```

use crate::cognitive_stack::{ThinkingStyle, GateState, RungLevel};
use crate::meaning_axes::{HdrResonance, Archetype, Viscosity};
use crate::ghosts::GhostType;

// ═══════════════════════════════════════════════════════════════════════════
// SELF STATE — the agent's internal awareness
// ═══════════════════════════════════════════════════════════════════════════

/// How the agent sees itself right now.
#[derive(Clone, Copy, Debug)]
pub struct SelfState {
    /// Current thinking style.
    pub style: ThinkingStyle,
    /// Semantic depth level (0–9).
    pub rung: u8,
    /// Collapse gate state.
    pub gate: GateState,
    /// Processing fluidity.
    pub viscosity: Viscosity,
    /// Confidence in current reasoning.
    pub confidence: f32,
    /// Calibration quality (Brier score, 0 = perfect).
    pub calibration_error: f32,
    /// Should the agent acknowledge uncertainty?
    pub should_acknowledge_limits: bool,
    /// Active persistent trace count.
    pub trace_count: u16,
    /// Surprise level from last processing cycle.
    pub free_energy: f32,
    /// Thoughts processed this session.
    pub thought_count: u64,
}

// ═══════════════════════════════════════════════════════════════════════════
// USER STATE — the agent's model of the other party
// ═══════════════════════════════════════════════════════════════════════════

/// How the agent reads the other party. Inferred, not measured.
/// Confidence reflects how reliable the inference is.
#[derive(Clone, Copy, Debug)]
pub struct UserState {
    /// Inferred cognitive style.
    pub style: ThinkingStyle,
    /// Engagement level (0.0 = disengaged, 1.0 = fully engaged).
    pub engagement: f32,
    /// Sentiment (-1.0 = negative, 1.0 = positive).
    pub valence: f32,
    /// Preferred processing depth (0–9).
    pub depth: u8,
    /// Confidence in this user model (0.0–1.0).
    pub model_confidence: f32,
}

// ═══════════════════════════════════════════════════════════════════════════
// FIELD STATE — the dynamics between self and user (gestalt)
// ═══════════════════════════════════════════════════════════════════════════

/// The relational dynamics between agent and context.
#[derive(Clone, Copy, Debug)]
pub struct FieldState {
    /// Agreement field evolution.
    pub gestalt: GestaltState,
    /// 3D resonance: subject / predicate / object perspectives.
    pub hdr: HdrResonance,
    /// Which perspective dominates.
    pub dominant: Archetype,
    /// Disagreement level (0.0 = consensus, 1.0 = full disagreement).
    pub dissonance: f32,
    /// Active atoms in superposition.
    pub n_resonant: u16,
    /// Total field energy.
    pub total_energy: f32,
    /// One perspective significantly stronger than others.
    pub is_divergent: bool,
    /// All perspectives agree.
    pub is_converged: bool,
}

/// How the agreement field is evolving.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GestaltState {
    /// Evidence accumulating, perspectives converging.
    Crystallizing,
    /// Perspectives disagree, needs clarification.
    Contested,
    /// Confidence dropping, counter-evidence arriving.
    Dissolving,
    /// New pattern detected, previously unseen connection.
    Epiphany,
}

// ═══════════════════════════════════════════════════════════════════════════
// CONTEXT STATE — semantic profile of the current content
// ═══════════════════════════════════════════════════════════════════════════

/// The semantic character of what's being processed.
#[derive(Clone, Debug)]
pub struct ContextState {
    /// Primary semantic classification.
    pub primary_family: String,
    /// Secondary classification.
    pub overlay_family: String,
    /// Named blend between primary and overlay.
    pub blend: String,
    /// Activation level (0.0–1.0).
    pub arousal: f32,
    /// Conflict level (0.0–1.0).
    pub tension: f32,
    /// Connection level (0.0–1.0).
    pub warmth: f32,
    /// Focus level (0.0–1.0).
    pub clarity: f32,
    /// Dominant persistent trace type (if any).
    pub dominant_trace: Option<GhostType>,
    /// SPO triples extracted this cycle.
    pub spo_count: u16,
    /// Unresolved conflict detected.
    pub has_conflict: bool,
}

// ═══════════════════════════════════════════════════════════════════════════
// WORLD MODEL — the complete picture
// ═══════════════════════════════════════════════════════════════════════════

/// The agent's complete situational awareness.
/// Self + User + Field + Context in one struct.
#[derive(Clone, Debug)]
pub struct WorldModelDto {
    /// How the agent sees itself.
    pub self_state: SelfState,
    /// How the agent reads the other party.
    pub user_state: UserState,
    /// The relational dynamics (gestalt).
    pub field_state: FieldState,
    /// The semantic profile of the current content.
    pub context_state: ContextState,
}

impl WorldModelDto {
    /// Build from the thinking engine's current state.
    pub fn from_engine_state(
        agent: &crate::persona::Agent,
        field: &crate::superposition::SuperpositionField,
        qualia: &crate::qualia::Qualia17D,
        dissonance: f32,
        free_energy: f32,
        lens_agreement: f32,
        spo_count: u16,
        calibration_error: f32,
    ) -> Self {
        let ghost_summary = agent.ghosts.summary();
        let dominant_trace = ghost_summary.first().map(|g| g.1);

        let hdr = HdrResonance::new(
            lens_agreement,
            1.0 - dissonance,
            field.total_energy / field.amplitudes.len().max(1) as f32,
        );

        let gestalt = if hdr.is_unanimous() && dissonance < 0.1 {
            GestaltState::Crystallizing
        } else if hdr.is_epiphany() {
            GestaltState::Epiphany
        } else if dissonance > 0.3 {
            GestaltState::Contested
        } else {
            GestaltState::Dissolving
        };

        let (primary, p_dist) = qualia.nearest_family();
        let (_, _, blend_name, _) = qualia.emotional_blend();
        let overlay = blend_name.split(" + ").nth(1)
            .and_then(|s| s.split(" = ").next())
            .unwrap_or("neutral");
        let blend = blend_name.split(" = ").last().unwrap_or("uncharted");

        Self {
            self_state: SelfState {
                style: agent.current_style,
                rung: agent.current_rung.as_u8(),
                gate: hdr.gate(),
                viscosity: if free_energy < 0.05 { Viscosity::Ice }
                    else if free_energy < 0.15 { Viscosity::Oil }
                    else { Viscosity::Water },
                confidence: lens_agreement,
                calibration_error,
                should_acknowledge_limits: calibration_error > 0.2 && lens_agreement < 0.4,
                trace_count: agent.ghosts.active_count() as u16,
                free_energy,
                thought_count: agent.thought_count,
            },
            user_state: UserState {
                style: if dissonance < 0.1 { ThinkingStyle::Analytical }
                    else if dissonance > 0.3 { ThinkingStyle::Creative }
                    else { ThinkingStyle::Deliberate },
                engagement: lens_agreement,
                valence: (1.0 - dissonance * 2.0).clamp(-1.0, 1.0),
                depth: if dissonance > 0.2 { 5 } else { 2 },
                model_confidence: lens_agreement * 0.8,
            },
            field_state: FieldState {
                gestalt,
                hdr,
                dominant: hdr.dominant(),
                dissonance,
                n_resonant: field.n_resonant as u16,
                total_energy: field.total_energy,
                is_divergent: hdr.is_epiphany(),
                is_converged: hdr.is_unanimous(),
            },
            context_state: ContextState {
                primary_family: primary.to_string(),
                overlay_family: overlay.to_string(),
                blend: blend.to_string(),
                arousal: qualia.dims[0],
                tension: qualia.dims[2],
                warmth: qualia.dims[3],
                clarity: qualia.dims[4],
                dominant_trace,
                spo_count,
                has_conflict: qualia.is_dissonant(),
            },
        }
    }
}

impl std::fmt::Display for WorldModelDto {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "Self[{} r{} {:?} FE={:.2}] User[{} e={:.1}] Field[{:?} d={:.2}] Ctx[{} {}]",
            self.self_state.style, self.self_state.rung, self.self_state.gate, self.self_state.free_energy,
            self.user_state.style, self.user_state.engagement,
            self.field_state.gestalt, self.field_state.dissonance,
            self.context_state.primary_family, self.context_state.blend)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gestalt_states_distinct() {
        assert_ne!(GestaltState::Crystallizing, GestaltState::Contested);
        assert_ne!(GestaltState::Epiphany, GestaltState::Dissolving);
    }

    #[test]
    fn self_state_copy() {
        let s = SelfState {
            style: ThinkingStyle::Analytical, rung: 3, gate: GateState::Flow,
            viscosity: Viscosity::Oil, confidence: 0.8, calibration_error: 0.1,
            should_acknowledge_limits: false, trace_count: 5, free_energy: 0.05,
            thought_count: 42,
        };
        let s2 = s; // Copy
        assert_eq!(s2.thought_count, 42);
    }

    #[test]
    fn user_state_copy() {
        let u = UserState {
            style: ThinkingStyle::Creative, engagement: 0.9, valence: 0.5,
            depth: 5, model_confidence: 0.7,
        };
        let u2 = u;
        assert!(u2.engagement > 0.8);
    }

    #[test]
    fn display_format() {
        let w = WorldModelDto {
            self_state: SelfState {
                style: ThinkingStyle::Deliberate, rung: 2, gate: GateState::Hold,
                viscosity: Viscosity::Honey, confidence: 0.6, calibration_error: 0.15,
                should_acknowledge_limits: false, trace_count: 10, free_energy: 0.12,
                thought_count: 100,
            },
            user_state: UserState {
                style: ThinkingStyle::Analytical, engagement: 0.8, valence: 0.3,
                depth: 4, model_confidence: 0.6,
            },
            field_state: FieldState {
                gestalt: GestaltState::Crystallizing,
                hdr: HdrResonance::new(0.8, 0.7, 0.6),
                dominant: Archetype::Guardian, dissonance: 0.1,
                n_resonant: 20, total_energy: 5.0,
                is_divergent: false, is_converged: false,
            },
            context_state: ContextState {
                primary_family: "emberglow".into(), overlay_family: "steelwind".into(),
                blend: "steady-flame".into(), arousal: 0.6, tension: 0.2,
                warmth: 0.8, clarity: 0.7, dominant_trace: None,
                spo_count: 15, has_conflict: false,
            },
        };
        let s = format!("{}", w);
        assert!(s.contains("Deliberate"));
        assert!(s.contains("Crystallizing"));
    }
}
