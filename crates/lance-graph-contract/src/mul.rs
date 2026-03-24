//! MUL (Meta-Uncertainty Layer) assessment contract.
//!
//! Defines the types for Dunning-Kruger positioning, trust assessment,
//! flow state detection, and compass gating. lance-graph-planner
//! implements the assessment logic; consumers pass SituationInput
//! and receive MulAssessment.

/// Situation input: what the consumer knows about the current context.
///
/// All fields are 0.0–1.0 unless noted.
#[derive(Debug, Clone)]
pub struct SituationInput {
    pub felt_competence: f64,
    pub demonstrated_competence: f64,
    pub source_reliability: f64,
    pub environment_stability: f64,
    pub calibration_accuracy: f64,
    pub challenge_level: f64,
    pub skill_level: f64,
    pub allostatic_load: f64,
    pub max_acceptable_damage: f64,
    pub reversibility_requirement: f64,
    pub sandbox_available: bool,
    pub complexity_ratio: f64,
    pub interdependency_density: f64,
}

impl Default for SituationInput {
    fn default() -> Self {
        Self {
            felt_competence: 0.5,
            demonstrated_competence: 0.5,
            source_reliability: 0.7,
            environment_stability: 0.7,
            calibration_accuracy: 0.5,
            challenge_level: 0.5,
            skill_level: 0.5,
            allostatic_load: 0.3,
            max_acceptable_damage: 0.5,
            reversibility_requirement: 0.5,
            sandbox_available: false,
            complexity_ratio: 1.0,
            interdependency_density: 0.3,
        }
    }
}

/// MUL assessment result.
#[derive(Debug, Clone)]
pub struct MulAssessment {
    /// Trust quality assessment.
    pub trust: TrustQualia,
    /// Dunning-Kruger position.
    pub dk_position: DkPosition,
    /// Flow/homeostasis state.
    pub homeostasis: Homeostasis,
    /// Whether complexity was successfully mapped.
    pub complexity_mapped: bool,
    /// Free will modifier (0.0 = fully constrained, 1.0 = fully autonomous).
    pub free_will_modifier: f64,
}

/// Trust quality: how much to trust the current assessment.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TrustQualia {
    /// Raw trust value (0.0–1.0).
    pub value: f64,
    /// Texture: how the trust "feels" (calibrated, tentative, etc.).
    pub texture: TrustTexture,
}

/// Trust texture — qualitative assessment of trust.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TrustTexture {
    /// Well-calibrated: felt ≈ demonstrated competence.
    Calibrated,
    /// Overconfident: felt >> demonstrated.
    Overconfident,
    /// Underconfident: felt << demonstrated.
    Underconfident,
    /// Uncertain: not enough data to assess.
    Uncertain,
}

/// Dunning-Kruger position on the competence curve.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DkPosition {
    /// Peak of Mount Stupid (overconfident novice).
    MountStupid,
    /// Valley of Despair (aware of incompetence).
    ValleyOfDespair,
    /// Slope of Enlightenment (growing competence).
    SlopeOfEnlightenment,
    /// Plateau of Sustainability (expert).
    Plateau,
}

/// Flow/homeostasis state.
#[derive(Debug, Clone)]
pub struct Homeostasis {
    /// Flow state assessment.
    pub flow_state: FlowState,
    /// Allostatic load (stress accumulation).
    pub allostatic_load: f64,
}

/// Flow state (Csikszentmihalyi).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FlowState {
    /// Challenge >> Skill → anxiety.
    Anxiety,
    /// Challenge ≈ Skill → flow.
    Flow,
    /// Challenge << Skill → boredom.
    Boredom,
    /// Transitioning between states.
    Transition,
}

/// Gate decision: should the system proceed, pause, or block?
#[derive(Debug, Clone)]
pub enum GateDecision {
    /// Proceed with full autonomy.
    Flow,
    /// Proceed with caution (reduced autonomy).
    Hold { reason: String },
    /// Block execution (require human input).
    Block { reason: String },
}

/// Compass result: surface-to-meta transition detection.
#[derive(Debug, Clone)]
pub struct CompassResult {
    /// Compass score (0.0 = stay surface, 1.0 = go meta).
    pub score: f64,
    /// Decision.
    pub decision: CompassDecision,
}

/// Compass decision.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CompassDecision {
    /// Stay at surface level (normal execution).
    StaySurface,
    /// Transition to meta level (reflect, replan).
    GoMeta,
}

/// Trait for MUL assessment providers.
///
/// lance-graph-planner implements this. Consumers call it.
pub trait MulProvider: Send + Sync {
    /// Assess a situation and return MUL result.
    fn assess(&self, input: &SituationInput) -> MulAssessment;

    /// Gate check: should execution proceed?
    fn gate_check(&self, assessment: &MulAssessment) -> GateDecision;

    /// Compass check: should we go meta?
    fn compass(&self, assessment: &MulAssessment) -> CompassResult;
}
