//! Bridge: lance-graph contract types ↔ thinking-engine.
//!
//! Maps the canonical 36 ThinkingStyles from lance-graph-contract
//! to the thinking-engine's 12 styles and cascade parameters.
//! Also bridges the KV bundle's HeadPrint (Base17 17D) to the
//! HDR distance table codebook centroids.
//!
//! The CollapseGate from the planner controls cascade termination.

use crate::cognitive_stack::{self, GateState, RungLevel};
use crate::meaning_axes::{Viscosity, CouncilWeights, Archetype};

/// Map contract's 36 styles to the engine's 12 styles.
/// The 36 are grouped into 6 clusters of 6. Each cluster maps to
/// 1-2 of our 12 styles, selecting the closest match.
pub fn contract_style_to_engine(contract_style_id: u8) -> cognitive_stack::ThinkingStyle {
    match contract_style_id {
        // Analytical cluster (0-5): Logical, Analytical, Critical, Systematic, Methodical, Precise
        0..=2 => cognitive_stack::ThinkingStyle::Analytical,
        3..=5 => cognitive_stack::ThinkingStyle::Systematic,
        // Creative cluster (6-11): Creative, Imaginative, Innovative, Artistic, Poetic, Playful
        6..=8 => cognitive_stack::ThinkingStyle::Creative,
        9..=11 => cognitive_stack::ThinkingStyle::Exploratory,
        // Empathic cluster (12-17): Empathetic, Compassionate, Supportive, Nurturing, Gentle, Warm
        12..=17 => cognitive_stack::ThinkingStyle::Intuitive,
        // Direct cluster (18-23): Direct, Concise, Efficient, Pragmatic, Blunt, Frank
        18..=20 => cognitive_stack::ThinkingStyle::Focused,
        21..=23 => cognitive_stack::ThinkingStyle::Convergent,
        // Exploratory cluster (24-29): Curious, Exploratory, Questioning, Investigative, Speculative, Philosophical
        24..=26 => cognitive_stack::ThinkingStyle::Divergent,
        27..=29 => cognitive_stack::ThinkingStyle::Peripheral,
        // Meta cluster (30-35): Reflective, Contemplative, Metacognitive, Wise, Transcendent, Sovereign
        30..=32 => cognitive_stack::ThinkingStyle::Metacognitive,
        33..=35 => cognitive_stack::ThinkingStyle::Deliberate, // Wise/Transcendent/Sovereign = deliberate
        _ => cognitive_stack::ThinkingStyle::Deliberate,
    }
}

/// Map contract's 6 clusters to council archetypes.
pub fn cluster_to_archetype(cluster_id: u8) -> Archetype {
    match cluster_id {
        0 => Archetype::Guardian,  // Analytical → cautious
        1 => Archetype::Catalyst,  // Creative → curious
        2 => Archetype::Balanced,  // Empathic → balanced
        3 => Archetype::Guardian,  // Direct → decisive (guardian-like)
        4 => Archetype::Catalyst,  // Exploratory → curious
        5 => Archetype::Balanced,  // Meta → balanced
        _ => Archetype::Balanced,
    }
}

/// Full cascade configuration derived from style + rung + gate.
#[derive(Clone, Debug)]
pub struct CascadeConfig {
    pub style: cognitive_stack::ThinkingStyle,
    pub params: cognitive_stack::StyleParams,
    pub rung: RungLevel,
    pub gate: GateState,
    pub viscosity: Viscosity,
    pub council: CouncilWeights,
    pub max_stages: usize,
    pub top_k: usize,
}

impl CascadeConfig {
    /// Build cascade config from a contract style ID + current state.
    pub fn from_contract(
        style_id: u8,
        rung: RungLevel,
        sd: f32,
        _free_energy: f32,
    ) -> Self {
        let style = contract_style_to_engine(style_id);
        let params = style.params();
        let gate = GateState::from_sd_styled(sd, &style);

        let viscosity = match gate {
            GateState::Flow => Viscosity::Ice,     // crystallized
            GateState::Hold => Viscosity::Honey,   // still moving
            GateState::Block => Viscosity::Water,  // need to explore
        };

        let mut council = CouncilWeights::default();
        let archetype = cluster_to_archetype(style_id / 6);
        council.shift_toward(archetype, 0.15);

        let max_stages = match gate {
            GateState::Flow => 3,   // quick commit
            GateState::Hold => 5,   // normal exploration
            GateState::Block => 8,  // deep search
        };

        let top_k = params.fan_out.min(20);

        Self { style, params, rung, gate, viscosity, council, max_stages, top_k }
    }

    /// Build from the superposition field state directly.
    pub fn from_superposition(
        field: &crate::superposition::SuperpositionField,
        style: &crate::superposition::ThinkingStyle,
        free_energy: f32,
    ) -> Self {
        // Map our 5 detected styles to the 12 cognitive_stack styles
        let engine_style = match style {
            crate::superposition::ThinkingStyle::Analytical => cognitive_stack::ThinkingStyle::Analytical,
            crate::superposition::ThinkingStyle::Creative => cognitive_stack::ThinkingStyle::Creative,
            crate::superposition::ThinkingStyle::Emotional => cognitive_stack::ThinkingStyle::Intuitive,
            crate::superposition::ThinkingStyle::Intuitive => cognitive_stack::ThinkingStyle::Intuitive,
            crate::superposition::ThinkingStyle::Diffuse => cognitive_stack::ThinkingStyle::Diffuse,
        };

        let params = engine_style.params();
        let resonant_frac = field.n_resonant as f32 / field.amplitudes.len().max(1) as f32;
        let sd = resonant_frac; // proxy for SD
        let gate = GateState::from_sd_styled(sd, &engine_style);

        let viscosity = if free_energy < 0.05 { Viscosity::Ice }
            else if free_energy < 0.1 { Viscosity::Honey }
            else if free_energy < 0.2 { Viscosity::Oil }
            else { Viscosity::Water };

        let mut council = CouncilWeights::default();
        if free_energy > 0.1 {
            council.shift_toward(Archetype::Catalyst, free_energy.min(0.2));
        }

        let rung = if free_energy > 0.15 { RungLevel::Analogical }
            else if field.n_resonant > 50 { RungLevel::Contextual }
            else { RungLevel::Surface };

        let max_stages = match gate {
            GateState::Flow => 3,
            GateState::Hold => 5,
            GateState::Block => 8,
        };

        Self {
            style: engine_style, params, rung, gate, viscosity,
            council, max_stages, top_k: params.fan_out.min(20),
        }
    }
}

/// BusDto: the fast struct for passing thoughts between components.
/// Minimal allocation, cache-friendly, no String/Vec.
#[derive(Clone, Copy, Debug)]
#[repr(C)]
pub struct FastBusDto {
    /// Dominant codebook centroid.
    pub dominant: u16,
    /// Energy at dominant.
    pub energy: f32,
    /// Thinking style (from 12-style enum).
    pub style: u8,
    /// Gate state: 0=Flow, 1=Hold, 2=Block.
    pub gate: u8,
    /// Rung level (0-9).
    pub rung: u8,
    /// Number of resonant atoms in superposition.
    pub n_resonant: u8,
    /// Dissonance (0-255 mapped from 0.0-1.0).
    pub dissonance: u8,
    /// Free energy (0-255 mapped from 0.0-1.0).
    pub free_energy: u8,
    /// Top-3 resonant atoms.
    pub top3: [u16; 3],
    /// Confidence (0-255 mapped from 0.0-1.0).
    pub confidence: u8,
    /// Ghost count at this point.
    pub ghost_count: u8,
}

impl FastBusDto {
    pub fn from_thought(
        dominant: u16,
        energy: f32,
        config: &CascadeConfig,
        n_resonant: usize,
        dissonance: f32,
        free_energy: f32,
        top3: &[u16],
        confidence: f32,
        ghost_count: usize,
    ) -> Self {
        Self {
            dominant,
            energy,
            style: config.style as u8,
            gate: match config.gate { GateState::Flow => 0, GateState::Hold => 1, GateState::Block => 2 },
            rung: config.rung.as_u8(),
            n_resonant: n_resonant.min(255) as u8,
            dissonance: (dissonance * 255.0).min(255.0) as u8,
            free_energy: (free_energy * 255.0).min(255.0) as u8,
            top3: [
                top3.first().copied().unwrap_or(0),
                top3.get(1).copied().unwrap_or(0),
                top3.get(2).copied().unwrap_or(0),
            ],
            confidence: (confidence * 255.0).min(255.0) as u8,
            ghost_count: ghost_count.min(255) as u8,
        }
    }

    /// Size in bytes (repr(C), no padding).
    pub const SIZE: usize = std::mem::size_of::<Self>();
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn contract_style_mapping() {
        assert_eq!(contract_style_to_engine(0), cognitive_stack::ThinkingStyle::Analytical);
        assert_eq!(contract_style_to_engine(6), cognitive_stack::ThinkingStyle::Creative);
        assert_eq!(contract_style_to_engine(12), cognitive_stack::ThinkingStyle::Intuitive);
        assert_eq!(contract_style_to_engine(18), cognitive_stack::ThinkingStyle::Focused);
        assert_eq!(contract_style_to_engine(24), cognitive_stack::ThinkingStyle::Divergent);
        assert_eq!(contract_style_to_engine(30), cognitive_stack::ThinkingStyle::Metacognitive);
    }

    #[test]
    fn fast_bus_size() {
        // Should be compact — no padding waste
        assert!(FastBusDto::SIZE <= 24, "FastBusDto is {} bytes, should be ≤24", FastBusDto::SIZE);
    }

    #[test]
    fn cascade_config_from_state() {
        let config = CascadeConfig::from_contract(6, RungLevel::Surface, 0.1, 0.05);
        assert_eq!(config.style, cognitive_stack::ThinkingStyle::Creative);
        assert_eq!(config.gate, GateState::Hold); // Creative biases toward Hold
    }
}
