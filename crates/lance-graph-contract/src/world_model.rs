//! WorldModelDto — the agent's complete situational awareness.
//!
//! This is a CONSUMER CONTRACT: any downstream crate (ladybug-rs, crewai-rust,
//! n8n-rs, thinking-engine) can depend on these types without pulling in
//! implementation details.
//!
//! ```text
//! WorldModelDto {
//!   self_state:    agent's internal awareness
//!   user_state:    inferred state of the other party (empathy)
//!   field_state:   relational dynamics between self and user (gestalt)
//!   context_state: semantic profile of the current content
//! }
//! ```
//!
//! Zero dependencies. Pure data types.

// ═══════════════════════════════════════════════════════════════════════════
// SELF STATE
// ═══════════════════════════════════════════════════════════════════════════

/// How the agent sees itself right now.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SelfState {
    /// Current thinking style (0–35, from ThinkingStyle enum).
    pub style_id: u8,
    /// Semantic depth level (0–9).
    pub rung: u8,
    /// Collapse gate state: 0=Flow, 1=Hold, 2=Block.
    pub gate: u8,
    /// Qualia state: 0=Ice(committed), 1=Tar(stalled), 2=Honey(iterating), 3=Oil(deliberate), 4=Water(streaming).
    pub qualia_state: u8,
    /// Confidence in current reasoning (0.0–1.0).
    pub confidence: f32,
    /// Calibration quality (Brier score, 0.0 = perfect, 1.0 = worst).
    pub calibration_error: f32,
    /// Should the agent acknowledge uncertainty?
    pub should_acknowledge_limits: bool,
    /// Active persistent trace count.
    pub trace_count: u16,
    /// Surprise from last processing cycle (Friston free energy, 0.0–1.0).
    pub free_energy: f32,
    /// Thoughts processed this session.
    pub thought_count: u64,
}

// ═══════════════════════════════════════════════════════════════════════════
// USER STATE (empathy model)
// ═══════════════════════════════════════════════════════════════════════════

/// How the agent reads the other party. Inferred, not measured.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct UserState {
    /// Inferred cognitive style (0–35).
    pub style_id: u8,
    /// Engagement level (0.0–1.0).
    pub engagement: f32,
    /// Sentiment (-1.0 negative, 1.0 positive).
    pub valence: f32,
    /// Preferred processing depth (0–9).
    pub depth: u8,
    /// How reliable this inference is (0.0–1.0).
    pub model_confidence: f32,
}

// ═══════════════════════════════════════════════════════════════════════════
// FIELD STATE (gestalt)
// ═══════════════════════════════════════════════════════════════════════════

/// The relational dynamics between agent and context.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FieldState {
    /// Agreement field state.
    pub gestalt: GestaltState,
    /// Subject perspective resonance (0.0–1.0).
    pub resonance_subject: f32,
    /// Predicate perspective resonance (0.0–1.0).
    pub resonance_predicate: f32,
    /// Object perspective resonance (0.0–1.0).
    pub resonance_object: f32,
    /// Dominant perspective: 0=Subject, 1=Predicate, 2=Object.
    pub dominant: u8,
    /// Disagreement level (0.0–1.0).
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
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum GestaltState {
    /// Evidence accumulating, perspectives converging.
    Crystallizing = 0,
    /// Perspectives disagree, needs clarification.
    Contested = 1,
    /// Confidence dropping, counter-evidence arriving.
    Dissolving = 2,
    /// New pattern detected, previously unseen connection.
    Epiphany = 3,
}

// ═══════════════════════════════════════════════════════════════════════════
// CONTEXT STATE
// ═══════════════════════════════════════════════════════════════════════════

/// Semantic profile of the current content.
#[derive(Clone, Debug, PartialEq)]
pub struct ContextState {
    /// Primary semantic classification family.
    pub primary_family: u8,
    /// Secondary classification family.
    pub overlay_family: u8,
    /// Activation level (0.0–1.0).
    pub arousal: f32,
    /// Conflict level (0.0–1.0).
    pub tension: f32,
    /// Connection level (0.0–1.0).
    pub warmth: f32,
    /// Focus level (0.0–1.0).
    pub clarity: f32,
    /// SPO triples extracted this cycle.
    pub spo_count: u16,
    /// Unresolved conflict detected.
    pub has_conflict: bool,
}

// ═══════════════════════════════════════════════════════════════════════════
// WORLD MODEL DTO
// ═══════════════════════════════════════════════════════════════════════════

/// The agent's complete situational awareness.
///
/// This is the canonical output of the thinking engine.
/// Any consumer can use this DTO to understand the agent's state
/// without depending on the thinking engine implementation.
#[derive(Clone, Debug, PartialEq)]
pub struct WorldModelDto {
    /// How the agent sees itself.
    pub self_state: SelfState,
    /// How the agent reads the other party.
    pub user_state: UserState,
    /// The relational dynamics (gestalt).
    pub field_state: FieldState,
    /// Semantic profile of the current content.
    pub context_state: ContextState,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn world_model_is_clone() {
        let wm = WorldModelDto {
            self_state: SelfState {
                style_id: 1, rung: 3, gate: 0, qualia_state: 3,
                confidence: 0.8, calibration_error: 0.1,
                should_acknowledge_limits: false,
                trace_count: 5, free_energy: 0.05, thought_count: 42,
            },
            user_state: UserState {
                style_id: 6, engagement: 0.9, valence: 0.5,
                depth: 5, model_confidence: 0.7,
            },
            field_state: FieldState {
                gestalt: GestaltState::Crystallizing,
                resonance_subject: 0.8, resonance_predicate: 0.7, resonance_object: 0.6,
                dominant: 0, dissonance: 0.1, n_resonant: 20,
                total_energy: 5.0, is_divergent: false, is_converged: false,
            },
            context_state: ContextState {
                primary_family: 0, overlay_family: 2,
                arousal: 0.6, tension: 0.2, warmth: 0.8, clarity: 0.7,
                spo_count: 15, has_conflict: false,
            },
        };
        let wm2 = wm.clone();
        assert_eq!(wm, wm2);
    }

    #[test]
    fn gestalt_repr() {
        assert_eq!(GestaltState::Crystallizing as u8, 0);
        assert_eq!(GestaltState::Epiphany as u8, 3);
    }

    #[test]
    fn self_state_is_copy() {
        let s = SelfState {
            style_id: 0, rung: 0, gate: 0, qualia_state: 0,
            confidence: 0.0, calibration_error: 0.0,
            should_acknowledge_limits: false,
            trace_count: 0, free_energy: 0.0, thought_count: 0,
        };
        let _s2 = s; // Copy
        let _s3 = s; // Still valid
    }
}
