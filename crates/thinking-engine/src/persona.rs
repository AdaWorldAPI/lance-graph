//! Agent + Persona + A2A Protocol.
//!
//! The Agent IS the thinking entity. Persona lives INSIDE it.
//! A2A messages carry the sender's identity so the receiver
//! knows WHO is speaking and HOW to interpret the thought.
//!
//! ```text
//! Agent {
//!   id, persona, ghosts, council, style, rung
//! }
//!
//! A2AMessage {
//!   from: AgentDto,       // sender's identity snapshot
//!   to: agent_id,         // receiver
//!   payload: thought/knowledge/query
//!   resonance_weight: f32 // how much this message matters
//! }
//! ```

use crate::cognitive_stack::{ThinkingStyle, GateState, RungLevel};
use crate::meaning_axes::{CouncilWeights, Archetype, Viscosity};
use crate::contract_bridge::{CascadeConfig, FastBusDto};
use crate::ghosts::GhostField;

// ═══════════════════════════════════════════════════════════════════════════
// PERSONA MODE
// ═══════════════════════════════════════════════════════════════════════════

/// Cognitive mode — agent-agnostic.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum PersonaMode {
    Work,      // focused, precise, early collapse, Guardian
    Personal,  // open, deep, late collapse, Catalyst
    Hybrid,    // balanced, adaptive
}

impl std::fmt::Display for PersonaMode {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self { Self::Work => write!(f, "work"), Self::Personal => write!(f, "personal"), Self::Hybrid => write!(f, "hybrid") }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// SOUL PRIORS (the 12 personality dimensions)
// ═══════════════════════════════════════════════════════════════════════════

/// 12 personality constants. The fingerprint of who this agent IS.
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

// ═══════════════════════════════════════════════════════════════════════════
// PERSONA PROFILE (inside the agent)
// ═══════════════════════════════════════════════════════════════════════════

/// The cognitive configuration of an agent. Lives INSIDE the agent.
#[derive(Clone, Debug)]
pub struct PersonaProfile {
    pub mode: PersonaMode,
    pub temperature_min: f32,
    pub temperature_max: f32,
    pub temperature_default: f32,
    pub rung_min: u8,
    pub rung_max: u8,
    pub collapse_bias: f32,
    pub affect_weight: f32,
    pub coherence_weight: f32,
    pub default_style: ThinkingStyle,
    pub priors: SoulPriors,
}

impl PersonaProfile {
    pub fn work() -> Self {
        Self {
            mode: PersonaMode::Work,
            temperature_min: 0.05, temperature_max: 0.30, temperature_default: 0.10,
            rung_min: 3, rung_max: 8, collapse_bias: -0.15,
            affect_weight: 0.1, coherence_weight: 0.9,
            default_style: ThinkingStyle::Analytical,
            priors: SoulPriors { precision_drive: 0.90, self_awareness: 0.80, warmth: 0.40, playfulness: 0.20, ..Default::default() },
        }
    }

    pub fn personal() -> Self {
        Self {
            mode: PersonaMode::Personal,
            temperature_min: 0.50, temperature_max: 0.90, temperature_default: 0.70,
            rung_min: 0, rung_max: 9, collapse_bias: 0.20,
            affect_weight: 0.8, coherence_weight: 0.2,
            default_style: ThinkingStyle::Creative,
            priors: SoulPriors { warmth: 0.92, depth: 0.85, presence: 0.88, intimacy_comfort: 0.90, vulnerability_tolerance: 0.85, playfulness: 0.78, novelty_seeking: 0.68, epistemic_humility: 0.82, ..Default::default() },
        }
    }

    pub fn hybrid() -> Self {
        Self {
            mode: PersonaMode::Hybrid,
            temperature_min: 0.20, temperature_max: 0.60, temperature_default: 0.40,
            rung_min: 2, rung_max: 9, collapse_bias: 0.0,
            affect_weight: 0.4, coherence_weight: 0.6,
            default_style: ThinkingStyle::Deliberate,
            priors: SoulPriors { warmth: 0.70, depth: 0.70, presence: 0.75, groundedness: 0.70, self_awareness: 0.85, ..Default::default() },
        }
    }

    pub fn ghost_decay_rate(&self) -> f32 {
        match self.mode { PersonaMode::Work => 0.75, PersonaMode::Personal => 0.92, PersonaMode::Hybrid => 0.85 }
    }

    pub fn novelty_gate_strength(&self) -> f32 {
        match self.mode { PersonaMode::Work => 2.0, PersonaMode::Personal => 0.5, PersonaMode::Hybrid => 1.0 }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// AGENT (the thinking entity)
// ═══════════════════════════════════════════════════════════════════════════

/// A thinking agent. Owns its persona, ghost field, council, and current state.
pub struct Agent {
    pub id: String,
    pub persona: PersonaProfile,
    pub ghosts: GhostField,
    pub council: CouncilWeights,
    pub current_style: ThinkingStyle,
    pub current_rung: RungLevel,
    pub thought_count: u64,
}

impl Agent {
    pub fn new(id: impl Into<String>, persona: PersonaProfile) -> Self {
        let style = persona.default_style;
        let mut ghosts = GhostField::new();
        ghosts.decay_rate = persona.ghost_decay_rate();
        let mut council = CouncilWeights::default();
        match persona.mode {
            PersonaMode::Work => council.shift_toward(Archetype::Guardian, 0.2),
            PersonaMode::Personal => council.shift_toward(Archetype::Catalyst, 0.2),
            PersonaMode::Hybrid => {}
        }
        Self {
            id: id.into(), persona, ghosts, council,
            current_style: style, current_rung: RungLevel::Surface, thought_count: 0,
        }
    }

    /// Snapshot the agent's identity for A2A messages.
    pub fn to_dto(&self) -> AgentDto {
        AgentDto {
            id: self.id.clone(),
            mode: self.persona.mode,
            style: self.current_style,
            rung: self.current_rung.as_u8(),
            warmth: self.persona.priors.warmth,
            depth: self.persona.priors.depth,
            presence: self.persona.priors.presence,
            ghost_count: self.ghosts.active_count() as u16,
            thought_count: self.thought_count,
            collapse_bias: self.persona.collapse_bias,
        }
    }

    /// Build cascade config from current state.
    pub fn cascade_config(&self, sd: f32, free_energy: f32) -> CascadeConfig {
        let gate = GateState::from_sd(sd + self.persona.collapse_bias);
        let rung = if free_energy > 0.15 {
            RungLevel::from_u8(self.persona.rung_max.min(9))
        } else {
            RungLevel::from_u8(self.persona.rung_min)
        };
        let viscosity = match gate {
            GateState::Flow => Viscosity::Ice,
            GateState::Hold => if self.persona.affect_weight > 0.5 { Viscosity::Honey } else { Viscosity::Oil },
            GateState::Block => Viscosity::Water,
        };
        CascadeConfig {
            style: self.current_style,
            params: self.current_style.params(),
            rung, gate, viscosity,
            council: self.council.clone(),
            max_stages: match gate { GateState::Flow => 3, GateState::Hold => 5, GateState::Block => 8 },
            top_k: self.current_style.params().fan_out.min(20),
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// AGENT DTO (identity snapshot for A2A)
// ═══════════════════════════════════════════════════════════════════════════

/// Lightweight identity snapshot. Sent in A2A messages.
/// The receiver knows WHO is speaking and adjusts interpretation.
#[derive(Clone, Debug)]
#[repr(C)]
pub struct AgentDto {
    pub id: String,
    pub mode: PersonaMode,
    pub style: ThinkingStyle,
    pub rung: u8,
    pub warmth: f32,
    pub depth: f32,
    pub presence: f32,
    pub ghost_count: u16,
    pub thought_count: u64,
    pub collapse_bias: f32,
}

// ═══════════════════════════════════════════════════════════════════════════
// A2A MESSAGE (agent-to-agent protocol)
// ═══════════════════════════════════════════════════════════════════════════

/// A2A message kinds.
#[derive(Clone, Debug)]
pub enum A2APayload {
    /// A thought to process.
    Thought(FastBusDto),
    /// Knowledge to integrate (SPO triples).
    Knowledge(Vec<crate::cognitive_trace::SpoTriple>),
    /// A query to answer.
    Query(String),
    /// Sync: request alignment of ghost fields.
    Sync,
    /// Persona exchange: share identity for calibration.
    PersonaExchange(AgentDto),
}

/// Agent-to-Agent message.
#[derive(Clone, Debug)]
pub struct A2AMessage {
    /// Sender's identity snapshot.
    pub from: AgentDto,
    /// Receiver agent ID.
    pub to: String,
    /// Message content.
    pub payload: A2APayload,
    /// How much this message should influence the receiver's cascade.
    /// Higher = more dominant in XOR superposition.
    pub resonance_weight: f32,
    /// Thinking style hint: suggests which style the receiver should use.
    pub style_hint: Option<ThinkingStyle>,
    /// Timestamp.
    pub timestamp: u64,
}

impl A2AMessage {
    pub fn thought(from: &Agent, to: &str, bus: FastBusDto, weight: f32) -> Self {
        Self {
            from: from.to_dto(), to: to.into(),
            payload: A2APayload::Thought(bus),
            resonance_weight: weight,
            style_hint: None, timestamp: 0,
        }
    }

    pub fn knowledge(from: &Agent, to: &str, triples: Vec<crate::cognitive_trace::SpoTriple>) -> Self {
        Self {
            from: from.to_dto(), to: to.into(),
            payload: A2APayload::Knowledge(triples),
            resonance_weight: 1.0,
            style_hint: None, timestamp: 0,
        }
    }

    pub fn persona_exchange(from: &Agent, to: &str) -> Self {
        Self {
            from: from.to_dto(), to: to.into(),
            payload: A2APayload::PersonaExchange(from.to_dto()),
            resonance_weight: 1.0,
            style_hint: None, timestamp: 0,
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// SELF-MODEL DTO (the agent's understanding of itself)
// ═══════════════════════════════════════════════════════════════════════════

/// The agent's self-model — what it knows about its own state.
/// Updated after each thought. The meta-cognitive mirror.
///
/// This is NOT the soul priors (those are fixed personality).
/// This is the CURRENT self-assessment: "how am I doing right now?"
#[derive(Clone, Debug)]
pub struct SelfModelDto {
    /// Agent identity.
    pub agent_id: String,
    pub mode: PersonaMode,

    /// Current cognitive state.
    pub style: ThinkingStyle,
    pub rung: RungLevel,
    pub gate: GateState,

    /// Calibration quality (Brier score, lower = better).
    pub calibration_error: f32,
    /// Should the agent admit it doesn't know?
    pub should_admit_ignorance: bool,

    /// Ghost field summary.
    pub ghost_count: u16,
    pub dominant_ghost_type: Option<crate::ghosts::GhostType>,
    /// How surprised was the agent by the last thought?
    pub last_free_energy: f32,

    /// Council balance.
    pub guardian_weight: f32,
    pub catalyst_weight: f32,
    pub balanced_weight: f32,

    /// Thinking trajectory (last N styles used).
    pub recent_styles: Vec<ThinkingStyle>,

    /// Emotional color of current state.
    pub qualia_family: String,
    pub dissonance: f32,

    /// Accumulated experience.
    pub thought_count: u64,
    pub spo_triple_count: u64,

    /// Viscosity: how freely is thought flowing?
    pub viscosity: Viscosity,
}

impl Agent {
    /// Build the self-model DTO — the agent's meta-cognitive snapshot.
    pub fn self_model(
        &self,
        calibration_error: f32,
        last_free_energy: f32,
        qualia_family: &str,
        dissonance: f32,
        spo_count: u64,
    ) -> SelfModelDto {
        let ghost_summary = self.ghosts.summary();
        let dominant_ghost = ghost_summary.first().map(|g| g.1);

        let gate = GateState::from_sd(dissonance + self.persona.collapse_bias);
        let viscosity = match gate {
            GateState::Flow => Viscosity::Ice,
            GateState::Hold => Viscosity::Honey,
            GateState::Block => Viscosity::Water,
        };

        SelfModelDto {
            agent_id: self.id.clone(),
            mode: self.persona.mode,
            style: self.current_style,
            rung: self.current_rung,
            gate,
            calibration_error,
            should_admit_ignorance: calibration_error > 0.2 && dissonance > 0.3,
            ghost_count: self.ghosts.active_count() as u16,
            dominant_ghost_type: dominant_ghost,
            last_free_energy,
            guardian_weight: self.council.guardian,
            catalyst_weight: self.council.catalyst,
            balanced_weight: self.council.balanced,
            recent_styles: vec![self.current_style], // TODO: track history
            qualia_family: qualia_family.to_string(),
            dissonance,
            thought_count: self.thought_count,
            spo_triple_count: spo_count,
            viscosity,
        }
    }
}

impl std::fmt::Display for SelfModelDto {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "[{}] {} | {} | rung:{} gate:{:?} | FE:{:.2} | {} ghosts | {} thoughts | {}",
            self.agent_id, self.mode, self.style, self.rung.as_u8(),
            self.gate, self.last_free_energy,
            self.ghost_count, self.thought_count, self.qualia_family)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn agent_creation() {
        let agent = Agent::new("test", PersonaProfile::hybrid());
        assert_eq!(agent.id, "test");
        assert_eq!(agent.persona.mode, PersonaMode::Hybrid);
        assert_eq!(agent.current_style, ThinkingStyle::Deliberate);
    }

    #[test]
    fn agent_dto_snapshot() {
        let agent = Agent::new("a1", PersonaProfile::personal());
        let dto = agent.to_dto();
        assert_eq!(dto.id, "a1");
        assert_eq!(dto.mode, PersonaMode::Personal);
        assert!(dto.warmth > 0.9);
    }

    #[test]
    fn a2a_thought_message() {
        let sender = Agent::new("sender", PersonaProfile::work());
        let bus = FastBusDto::from_thought(42, 0.8, &sender.cascade_config(0.1, 0.05), 10, 0.2, 0.1, &[42, 85, 29], 0.7, 5);
        let msg = A2AMessage::thought(&sender, "receiver", bus, 0.9);
        assert_eq!(msg.to, "receiver");
        assert_eq!(msg.resonance_weight, 0.9);
        assert_eq!(msg.from.mode, PersonaMode::Work);
    }

    #[test]
    fn a2a_persona_exchange() {
        let agent = Agent::new("a1", PersonaProfile::personal());
        let msg = A2AMessage::persona_exchange(&agent, "a2");
        match msg.payload {
            A2APayload::PersonaExchange(dto) => assert_eq!(dto.mode, PersonaMode::Personal),
            _ => panic!("wrong payload"),
        }
    }

    #[test]
    fn work_guardian_council() {
        let agent = Agent::new("w", PersonaProfile::work());
        assert!(agent.council.guardian > agent.council.catalyst);
    }

    #[test]
    fn personal_catalyst_council() {
        let agent = Agent::new("p", PersonaProfile::personal());
        assert!(agent.council.catalyst > agent.council.guardian);
    }
}
