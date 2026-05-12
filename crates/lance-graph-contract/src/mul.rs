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

// ═══════════════════════════════════════════════════════════════════════════
// Ontology-aware MUL thresholds (D-ONTO-V5-9)
//
// Per `lance-graph-ontology-v5.md` §D-9: medical contexts demand stricter
// trust / flow / compass thresholds than callcenter contexts. Today the
// driver uses fixed scalar thresholds; this profile makes them
// ontology-context-aware. The driver's GateDecision computation site
// (cognitive-shader-driver::driver.rs ~L271-320) consults
// `MulThresholdProfile::for_context(ontology_context_id)` to pick the
// active profile.
//
// **Zone classification**: Zone 1 (BindSpace SoA, inside the BBB).
// MUST NOT carry `serde::Serialize` — `crates/lance-graph-callcenter/build.rs`
// (D-CASCADE-V1-1) actively scans for and rejects Serialize on Zone 1 types.
// See `.claude/knowledge/soa-dto-dependency-ledger.md`.
//
// **Integration plumb-through (TODO)**: `for_context` accepts a `u32`
// `ontology_context_id` placeholder. The Wave-2 `agent-context-id`
// deliverable adds `ontology_context_id: u32` onto
// `lance_graph_ontology::SchemaPtr`; the Wave-3 `agent-cascade-cols`
// deliverable threads it through `MappingRow` so `BindSpace` can read
// it per-row. Until then, the driver passes `0` (default profile).
// ═══════════════════════════════════════════════════════════════════════════

/// Per-ontology-context MUL gate thresholds.
///
/// Three canonical profiles ship with the contract: `MEDICAL` (strict),
/// `CALLCENTER` (lenient), `DEFAULT` (everything else). Lookup happens
/// via `for_context(ontology_context_id)`.
///
/// The struct is `Copy` so it can sit on the BindSpace per-row carrier
/// without indirection. `Eq`/`Hash` are NOT derived because the `f32`
/// fields cannot satisfy them; `PartialEq` is sufficient for the gate's
/// equality checks and the test asserts.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MulThresholdProfile {
    /// `GateDecision` rejects when `TrustQualia.value` (texture-derived) < this.
    pub trust_min: f32,
    /// Homeostasis floor: flow_state must clear this before the gate emits Flow.
    pub flow_min: f32,
    /// Angular drift ceiling: the compass blocks when drift > this.
    pub compass_max: f32,
    /// Symbolic profile name (`"medical" | "callcenter" | "default"`).
    pub label: &'static str,
}

impl MulThresholdProfile {
    /// Strict medical/healthcare profile — trust ≥ 0.85, flow ≥ 0.70, drift ≤ 0.15.
    pub const MEDICAL: Self = Self {
        trust_min: 0.85,
        flow_min: 0.70,
        compass_max: 0.15,
        label: "medical",
    };

    /// Lenient callcenter / WorkOrder profile — trust ≥ 0.55, flow ≥ 0.40, drift ≤ 0.40.
    pub const CALLCENTER: Self = Self {
        trust_min: 0.55,
        flow_min: 0.40,
        compass_max: 0.40,
        label: "callcenter",
    };

    /// Default profile for unmapped contexts — trust ≥ 0.65, flow ≥ 0.50, drift ≤ 0.30.
    pub const DEFAULT: Self = Self {
        trust_min: 0.65,
        flow_min: 0.50,
        compass_max: 0.30,
        label: "default",
    };

    /// Look up the active profile for an ontology context id.
    ///
    /// Mapping (per `lance-graph-ontology-v5.md` §D-9):
    /// - `1` (WorkOrder) → `CALLCENTER`
    /// - `2` (Healthcare) → `MEDICAL`
    /// - `10..=19` (Medical/* subnamespaces) → `MEDICAL`
    /// - everything else → `DEFAULT`
    #[inline]
    pub const fn for_context(ontology_context_id: u32) -> Self {
        match ontology_context_id {
            1 => Self::CALLCENTER,
            2 => Self::MEDICAL,
            10..=19 => Self::MEDICAL,
            _ => Self::DEFAULT,
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Carrier-method MUL assessment (TD-INT-3 wiring)
//
// Per CLAUDE.md doctrine ("methods on the carrier, not free functions on
// state"), MulAssessment carries its own compute() call. This is the
// shader-driver entry point: dispatch hands a SituationInput, gets back
// a MulAssessment, and uses dk_position + flow_state + trust.texture to
// modulate the gate decision.
//
// The planner has its own richer MulAssessment in lance-graph-planner::mul;
// this contract method is the zero-dep version that shader-driver and any
// other consumer can call without reaching into the planner.
// ═══════════════════════════════════════════════════════════════════════════

impl MulAssessment {
    /// Compute a MUL assessment directly from a SituationInput.
    ///
    /// Mirrors the planner's `mul::assess()` shape but lives on the carrier
    /// per the carrier-method doctrine. Pure, deterministic, zero-dep.
    ///
    /// Use this from any consumer that has a `SituationInput` and needs
    /// dk_position / trust.texture / homeostasis.flow_state to refine a
    /// downstream decision (the shader-driver collapse_gate is the
    /// canonical first consumer — see TD-INT-3).
    pub fn compute(input: &SituationInput) -> Self {
        // Phase 1: Trust qualia (geometric mean of 4 dimensions).
        let composite_trust = (input.demonstrated_competence
            * input.source_reliability
            * input.environment_stability
            * input.calibration_accuracy)
            .max(0.0)
            .powf(0.25);
        let trust_texture = trust_texture_from(
            input.felt_competence,
            input.demonstrated_competence,
            composite_trust,
        );
        let trust = TrustQualia {
            value: composite_trust,
            texture: trust_texture,
        };

        // Phase 1: Dunning-Kruger position (felt vs demonstrated competence).
        let dk_position = dk_from(input.felt_competence, input.demonstrated_competence);

        // Phase 2: Complexity mapping (≥30% of dimensions known).
        let complexity_mapped = input.complexity_ratio > 0.3;

        // Phase 3: Homeostasis (flow state + allostatic load).
        let flow_state = flow_state_from(input.challenge_level, input.skill_level);
        let homeostasis = Homeostasis {
            flow_state,
            allostatic_load: input.allostatic_load,
        };

        // Phase 4: Free-will modifier (multiplicative humility chain).
        let dk_factor = match dk_position {
            DkPosition::MountStupid => 0.3,
            DkPosition::ValleyOfDespair => 0.7,
            DkPosition::SlopeOfEnlightenment => 0.85,
            DkPosition::Plateau => 1.0,
        };
        let trust_factor = composite_trust;
        let complexity_factor = if complexity_mapped {
            0.8 + 0.2 * input.complexity_ratio
        } else {
            0.4
        };
        let load_penalty = if input.allostatic_load > 0.7 {
            0.3
        } else {
            1.0
        };
        let flow_factor = match flow_state {
            FlowState::Flow => 1.0,
            FlowState::Anxiety => 0.6,
            FlowState::Boredom => 0.8,
            FlowState::Transition => 0.7,
        } * load_penalty;

        let free_will_modifier =
            (dk_factor * trust_factor * complexity_factor * flow_factor).clamp(0.0, 1.0);

        Self {
            trust,
            dk_position,
            homeostasis,
            complexity_mapped,
            free_will_modifier,
        }
    }

    /// Whether the meta-uncertainty layer is signalling unskilled-overconfident:
    /// the system "feels confident" while DK and trust both flag the gap.
    /// Used by the shader-driver gate as a veto hint.
    #[inline]
    pub fn is_unskilled_overconfident(&self) -> bool {
        self.dk_position == DkPosition::MountStupid
            || self.trust.texture == TrustTexture::Overconfident
    }
}

fn trust_texture_from(felt: f64, demonstrated: f64, composite: f64) -> TrustTexture {
    let gap = felt - demonstrated;
    if composite < 0.25 {
        TrustTexture::Uncertain
    } else if gap > 0.25 {
        TrustTexture::Overconfident
    } else if gap < -0.25 {
        TrustTexture::Underconfident
    } else {
        TrustTexture::Calibrated
    }
}

fn dk_from(felt: f64, demonstrated: f64) -> DkPosition {
    let gap = felt - demonstrated;
    if gap > 0.3 && demonstrated < 0.4 {
        DkPosition::MountStupid
    } else if felt < 0.4 && demonstrated < 0.5 {
        DkPosition::ValleyOfDespair
    } else if demonstrated > 0.7 && gap.abs() < 0.15 {
        DkPosition::Plateau
    } else {
        DkPosition::SlopeOfEnlightenment
    }
}

fn flow_state_from(challenge: f64, skill: f64) -> FlowState {
    let delta = challenge - skill;
    if delta.abs() < 0.15 && challenge > 0.3 {
        FlowState::Flow
    } else if delta > 0.2 {
        FlowState::Anxiety
    } else if delta < -0.2 {
        FlowState::Boredom
    } else {
        FlowState::Transition
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn compute_default_input_is_calibratedish() {
        let mul = MulAssessment::compute(&SituationInput::default());
        assert!(mul.free_will_modifier >= 0.0 && mul.free_will_modifier <= 1.0);
        // Default is moderate competence; should NOT be Mount Stupid.
        assert_ne!(mul.dk_position, DkPosition::MountStupid);
    }

    #[test]
    fn compute_detects_mount_stupid() {
        let input = SituationInput {
            felt_competence: 0.95,
            demonstrated_competence: 0.10,
            ..SituationInput::default()
        };
        let mul = MulAssessment::compute(&input);
        assert_eq!(mul.dk_position, DkPosition::MountStupid);
        assert!(mul.is_unskilled_overconfident());
    }

    #[test]
    fn compute_detects_plateau() {
        let input = SituationInput {
            felt_competence: 0.85,
            demonstrated_competence: 0.85,
            source_reliability: 0.9,
            environment_stability: 0.9,
            calibration_accuracy: 0.9,
            challenge_level: 0.6,
            skill_level: 0.6,
            ..SituationInput::default()
        };
        let mul = MulAssessment::compute(&input);
        assert_eq!(mul.dk_position, DkPosition::Plateau);
        assert!(!mul.is_unskilled_overconfident());
    }
}
