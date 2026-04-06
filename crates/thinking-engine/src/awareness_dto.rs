//! Awareness DTOs — structured cognitive state transfer objects.
//!
//! ```text
//! ResonanceDto  — multi-perspective agreement + inferred user state
//! QualiaDto     — 48-axis semantic profile + classification + texture
//! MomentDto     — complete snapshot: resonance + qualia + agent state
//! ```

use crate::meaning_axes::{HdrResonance, Archetype, AxisActivation, Viscosity};
use crate::cognitive_stack::{ThinkingStyle, GateState, RungLevel};
use crate::ghosts::GhostType;

// ═══════════════════════════════════════════════════════════════════════════
// RESONANCE DTO — the gestalt + user model
// ═══════════════════════════════════════════════════════════════════════════

/// Multi-perspective resonance field + inferred user model.
/// Three perspectives (subject/predicate/object) produce a 3D agreement profile.
/// The user model is inferred from conversation patterns — approximate, not authoritative.
#[derive(Clone, Debug)]
pub struct ResonanceDto {
    // ── Multi-perspective agreement ──

    /// 3D resonance: subject/predicate/object perspectives.
    pub hdr: HdrResonance,
    /// Which perspective dominates.
    pub dominant_perspective: Archetype,
    /// One perspective significantly stronger than others.
    pub is_divergent: bool,
    /// All perspectives agree.
    pub is_converged: bool,
    /// Collapse gate state from resonance variance.
    pub gate: GateState,

    // ── Field state ──

    /// How the agreement field is evolving.
    pub gestalt_state: GestaltState,
    /// Disagreement level (0.0 = full agreement, 1.0 = full disagreement).
    pub dissonance: f32,
    /// Number of active atoms in the superposition field.
    pub n_resonant: usize,
    /// Total field energy.
    pub total_energy: f32,

    // ── Inferred user state ──

    /// Inferred user cognitive style (from conversation pattern).
    pub user_style: ThinkingStyle,
    /// Inferred user engagement level (0.0–1.0).
    pub user_engagement: f32,
    /// Inferred user sentiment (-1.0 negative, 1.0 positive).
    pub user_valence: f32,
    /// Inferred user depth preference (0–9).
    pub user_depth: u8,
    /// Confidence in the user model (0.0–1.0).
    pub user_model_confidence: f32,
}

/// Agreement field state: how multi-lens consensus is evolving.
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

impl ResonanceDto {
    /// Build from superposition field + cascade results.
    pub fn from_superposition(
        field: &crate::superposition::SuperpositionField,
        dissonance: f32,
        lens_agreement: f32,
    ) -> Self {
        // Map multi-lens agreement to I/Thou/It resonance
        let hdr = HdrResonance::new(
            lens_agreement,                           // I: how aligned am I with this input
            1.0 - dissonance,                         // Thou: how harmonious is our exchange
            field.total_energy / field.amplitudes.len().max(1) as f32, // It: how much signal exists
        );

        let gestalt_state = if hdr.is_unanimous() && dissonance < 0.1 {
            GestaltState::Crystallizing
        } else if hdr.is_epiphany() {
            GestaltState::Epiphany
        } else if dissonance > 0.3 {
            GestaltState::Contested
        } else {
            GestaltState::Dissolving
        };

        // User model: infer from the resonance pattern
        let user_style = if dissonance < 0.1 { ThinkingStyle::Analytical }
            else if dissonance > 0.3 { ThinkingStyle::Creative }
            else { ThinkingStyle::Deliberate };
        let user_engagement = lens_agreement;
        let user_valence = 1.0 - dissonance * 2.0;

        Self {
            hdr,
            dominant_perspective: hdr.dominant(),
            is_divergent: hdr.is_epiphany(),
            is_converged: hdr.is_unanimous(),
            gate: hdr.gate(),
            gestalt_state,
            dissonance,
            n_resonant: field.n_resonant,
            total_energy: field.total_energy,
            user_style,
            user_engagement,
            user_valence: user_valence.clamp(-1.0, 1.0),
            user_depth: if dissonance > 0.2 { 5 } else { 2 },
            user_model_confidence: lens_agreement * 0.8,
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// QUALIA DTO — the felt sense
// ═══════════════════════════════════════════════════════════════════════════

/// Semantic profile of a cognitive moment.
/// 48-axis activation + classification + processing texture.
#[derive(Clone, Debug)]
pub struct QualiaDto {
    // ── Semantic axes ──

    /// Full 48D semantic axis activation.
    pub axes: AxisActivation,
    /// Dominant axis family (osgood/physical/emotional/...).
    pub dominant_family: String,
    /// 17D compact projection of the 48D space.
    pub compact_17d: [f32; 17],

    // ── Classification (10 families) ──

    /// Primary classification family.
    pub primary_family: String,
    /// Primary match strength (0.0–1.0).
    pub primary_intensity: f32,
    /// Secondary classification family.
    pub overlay_family: String,
    /// Secondary match strength.
    pub overlay_intensity: f32,
    /// Blend label (e.g. "catharsis", "ice-clarity").
    pub blend: String,

    // ── Processing texture ──

    /// Processing fluidity.
    pub viscosity: Viscosity,
    /// Sentiment (-1.0 to 1.0).
    pub valence: f32,
    /// Activation level (0.0 to 1.0).
    pub arousal: f32,
    /// Conflict level (0.0 to 1.0).
    pub tension: f32,
    /// Connection level (0.0 to 1.0).
    pub warmth: f32,
    /// Focus level (0.0 to 1.0).
    pub clarity: f32,

    // ── Persistent traces ──

    /// Active persistent trace types and their intensities.
    pub traces: Vec<(GhostType, f32)>,
    /// Unresolved conflict detected.
    pub is_dissonant: bool,
}

impl QualiaDto {
    /// Build from qualia 17D + superposition.
    pub fn from_qualia(
        qualia: &crate::qualia::Qualia17D,
        ghost_summary: &[(u16, GhostType, f32)],
    ) -> Self {
        let (primary, p_dist) = qualia.nearest_family();
        let p_intensity = (1.0 - p_dist / 2.0).clamp(0.0, 1.0);
        let (_, _, blend, (_, o_intensity)) = qualia.emotional_blend();

        // Build 48-axis activation from 17D (reverse ICC — approximate)
        let mut axes = AxisActivation::neutral();
        axes.values[0] = qualia.dims[1];   // good↔bad ← valence
        axes.values[1] = qualia.dims[12];  // strong↔weak ← assertion
        axes.values[2] = qualia.dims[0];   // active↔passive ← arousal
        axes.values[7] = qualia.dims[3];   // hot↔cold ← warmth
        axes.values[9] = qualia.dims[7];   // fast↔slow ← velocity
        axes.values[20] = qualia.dims[4];  // certain↔uncertain ← clarity
        axes.values[24] = -qualia.dims[2]; // happy↔sad ← inverse tension
        axes.values[25] = 1.0 - qualia.dims[2]; // calm↔anxious ← inverse tension
        axes.values[26] = qualia.dims[3];  // loving↔hateful ← warmth
        axes.values[37] = qualia.dims[16]; // whole↔partial ← integration

        let overlay = blend.split(" + ").nth(1)
            .and_then(|s| s.split(" = ").next())
            .unwrap_or("neutral");

        let traces: Vec<(GhostType, f32)> = ghost_summary.iter()
            .map(|(_, gt, intensity)| (*gt, *intensity))
            .collect();

        Self {
            dominant_family: axes.dominant_family().to_string(),
            compact_17d: qualia.dims,
            axes,
            primary_family: primary.to_string(),
            primary_intensity: p_intensity,
            overlay_family: overlay.to_string(),
            overlay_intensity: o_intensity,
            blend: blend.split(" = ").last().unwrap_or("uncharted").to_string(),
            viscosity: if qualia.dims[2] > 0.5 { Viscosity::Honey } else { Viscosity::Oil },
            valence: qualia.dims[1],
            arousal: qualia.dims[0],
            tension: qualia.dims[2],
            warmth: qualia.dims[3],
            clarity: qualia.dims[4],
            traces,
            is_dissonant: qualia.is_dissonant(),
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// MOMENT DTO — the complete cognitive snapshot
// ═══════════════════════════════════════════════════════════════════════════

/// Complete cognitive snapshot: resonance + qualia + agent identity.
/// A single moment in the agent's experience.
#[derive(Clone, Debug)]
pub struct MomentDto {
    /// Agent identity.
    pub agent_id: String,
    pub mode: super::persona::PersonaMode,

    /// The gestalt: I/Thou/It + user model.
    pub resonance: ResonanceDto,

    /// The felt sense: 48 axes + qualia family + texture.
    pub qualia: QualiaDto,

    /// Thinking style active in this moment.
    pub style: ThinkingStyle,
    /// Semantic depth.
    pub rung: RungLevel,
    /// Free energy (surprise level).
    pub free_energy: f32,
    /// Thought count in session.
    pub thought_count: u64,
    /// SPO triples extracted.
    pub spo_count: usize,
    /// Timestamp.
    pub timestamp: u64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gestalt_states() {
        assert_ne!(GestaltState::Crystallizing, GestaltState::Contested);
        assert_ne!(GestaltState::Epiphany, GestaltState::Dissolving);
    }

    #[test]
    fn resonance_from_superposition() {
        let field = crate::superposition::SuperpositionField {
            amplitudes: vec![0.0; 256],
            resonant_atoms: vec![(10, 1.0), (20, 0.5)],
            total_energy: 1.5,
            n_resonant: 2,
        };
        let res = ResonanceDto::from_superposition(&field, 0.1, 0.8);
        // Low dissonance + high agreement → should not be Contested or Dissolving
        assert_ne!(res.gestalt_state, GestaltState::Contested);
        assert!(res.user_engagement > 0.7);
    }

    #[test]
    fn qualia_from_17d() {
        let mut q = crate::qualia::Qualia17D { dims: [0.5; 17] };
        q.dims[0] = 0.8; // arousal
        q.dims[1] = 0.7; // valence
        q.dims[2] = 0.1; // tension
        q.dims[3] = 0.9; // warmth
        let dto = QualiaDto::from_qualia(&q, &[]);
        assert!(dto.warmth > 0.8);
        assert!(dto.arousal > 0.7);
        assert!(!dto.is_dissonant);
    }

    #[test]
    fn high_dissonance_not_crystallizing() {
        let field = crate::superposition::SuperpositionField {
            amplitudes: vec![0.0; 256],
            resonant_atoms: vec![(10, 1.0)],
            total_energy: 1.0,
            n_resonant: 1,
        };
        let res = ResonanceDto::from_superposition(&field, 0.5, 0.3);
        // High dissonance → should not be Crystallizing
        assert_ne!(res.gestalt_state, GestaltState::Crystallizing);
    }
}
