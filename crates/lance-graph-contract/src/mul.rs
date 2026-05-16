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


// ═══════════════════════════════════════════════════════════════════════════
// i4 scalar evaluation path — D-CSV-8 (sprint-11)
//
// Integer SIMD-ready MUL evaluation that consumes `QualiaI4_16D` + signed
// mantissa (i8 from `InferenceType::to_mantissa()`) and produces the same
// MUL types as the existing f32 path. The actual AVX-512 / NEON hot path
// is sprint-12+; this module locks the scalar i4 shape so sprint-12 can
// vectorise without changing API.
//
// All decision logic is pure: no heap allocation, no f64, no f32.
// GateDecision::Hold/Block carry &'static str reason to preserve zero-alloc.
// ═══════════════════════════════════════════════════════════════════════════

/// i4-scalar MUL evaluation.
///
/// All functions are `#[inline]` and heap-free. They consume
/// `QualiaI4_16D` from `crate::qualia` and a signed mantissa i8
/// (from `causal_edge::InferenceType::to_mantissa()`), and return the
/// existing MUL contract types unchanged.
pub mod i4_eval {
    use super::{DkPosition, FlowState, GateDecision, Homeostasis, MulAssessment, TrustQualia, TrustTexture};
    use crate::qualia::QualiaI4_16D;

    // ── dim indices (aligned with QUALIA_I4_LABELS) ─────────────────────────
    const DIM_VALENCE: usize = 1;      // signed valence (polarity)
    const DIM_TENSION: usize = 2;      // tension / conflict load
    const DIM_WARMTH: usize = 3;       // warmth / affiliation
    const DIM_COHERENCE: usize = 9;    // coherence (story holds / breaks)
    const DIM_GROUNDEDNESS: usize = 14; // groundedness / stability

    /// On-demand intensity helper: `magnitude()` from the qualia struct.
    /// Returns coherence × valence as i8 (saturating). Used as a combined
    /// "signal strength × polarity" probe.
    #[inline]
    fn intensity_i4(qualia: &QualiaI4_16D) -> i8 {
        qualia.magnitude() // coherence(dim9) × valence(dim1), saturating
    }

    // ── DkPosition ──────────────────────────────────────────────────────────

    /// Classify Dunning-Kruger position from i4 qualia + signed mantissa.
    ///
    /// Decision rules (i4 range −8..+7):
    /// - `coherence(dim9) >= +5` AND `|signed_mantissa| >= 4`
    ///   → `Plateau` (expert: story holds, high-confidence rule active)
    /// - `coherence(dim9) >= +2` AND `|signed_mantissa| >= 2`
    ///   → `SlopeOfEnlightenment` (growing: moderate coherence + rule)
    /// - `coherence(dim9) <= -3` OR `|signed_mantissa| <= 1`
    ///   → `ValleyOfDespair` (low coherence or weak rule = aware of gaps)
    /// - otherwise → `MountStupid` (moderate-but-positive coherence, weak mantissa)
    #[inline]
    pub fn dk_position_i4(qualia: &QualiaI4_16D, signed_mantissa: i8) -> DkPosition {
        let coherence = qualia.get(DIM_COHERENCE);
        let abs_mantissa = signed_mantissa.unsigned_abs() as i8;

        if coherence >= 5 && abs_mantissa >= 4 {
            DkPosition::Plateau
        } else if coherence >= 2 && abs_mantissa >= 2 {
            DkPosition::SlopeOfEnlightenment
        } else if coherence <= -3 || abs_mantissa <= 1 {
            DkPosition::ValleyOfDespair
        } else {
            DkPosition::MountStupid
        }
    }

    // ── TrustTexture ─────────────────────────────────────────────────────────

    /// Derive TrustTexture from i4 qualia.
    ///
    /// Uses coherence (dim 9), valence (dim 1), tension (dim 2):
    ///
    /// | coherence | valence | tension | result        |
    /// |-----------|---------|---------|---------------|
    /// | ≥ +4      | ≥ +2    | ≤ +1   | Calibrated    |
    /// | ≤ -3      | any     | ≥ +3   | Uncertain     |
    /// | any       | ≥ +4    | any     | Overconfident |
    /// | any       | ≤ -3    | any     | Underconfident|
    /// | otherwise                     | Calibrated (moderate) |
    #[inline]
    pub fn trust_texture_i4(qualia: &QualiaI4_16D) -> TrustTexture {
        let coherence = qualia.get(DIM_COHERENCE);
        let valence = qualia.get(DIM_VALENCE);
        let tension = qualia.get(DIM_TENSION);

        if coherence <= -3 && tension >= 3 {
            TrustTexture::Uncertain
        } else if valence >= 4 && coherence < 5 {
            // High valence with only moderate coherence = overconfident
            TrustTexture::Overconfident
        } else if valence <= -3 {
            TrustTexture::Underconfident
        } else if coherence >= 4 && valence >= 2 && tension <= 1 {
            TrustTexture::Calibrated
        } else {
            // Moderate values — calibrated by default
            TrustTexture::Calibrated
        }
    }

    // ── FlowState ────────────────────────────────────────────────────────────

    /// Classify FlowState from i4 qualia + signed mantissa.
    ///
    /// Flow proxy = warmth(dim3) + groundedness(dim14) − tension(dim2).
    /// Combined with mantissa sign for direction:
    ///
    /// - flow_proxy ≥ +4 AND signed_mantissa > 0 → `Flow` (absorbed)
    /// - flow_proxy ≥ +2 AND signed_mantissa > 0 → `Transition` (building)
    /// - flow_proxy ≤ -2 OR (signed_mantissa < 0 AND coherence ≤ -1) → `Anxiety`
    /// - otherwise → `Boredom`
    #[inline]
    pub fn flow_state_i4(qualia: &QualiaI4_16D, signed_mantissa: i8) -> FlowState {
        let warmth = qualia.get(DIM_WARMTH);
        let groundedness = qualia.get(DIM_GROUNDEDNESS);
        let tension = qualia.get(DIM_TENSION);
        let coherence = qualia.get(DIM_COHERENCE);

        // Saturating i8 arithmetic on i4 inputs stays in i8 range safely
        let flow_proxy = (warmth as i16 + groundedness as i16 - tension as i16)
            .clamp(i8::MIN as i16, i8::MAX as i16) as i8;

        if flow_proxy >= 4 && signed_mantissa > 0 {
            FlowState::Flow
        } else if flow_proxy <= -2 || (signed_mantissa < 0 && coherence <= -1) {
            FlowState::Anxiety
        } else if flow_proxy >= 2 && signed_mantissa > 0 {
            FlowState::Transition
        } else {
            FlowState::Boredom
        }
    }

    // ── GateDecision ─────────────────────────────────────────────────────────

    /// Gate decision from i4 qualia + signed mantissa.
    ///
    /// Combines TrustTexture + FlowState:
    /// - `Uncertain` trust → `Block`
    /// - `Underconfident` trust + `Anxiety` → `Block`
    /// - `Overconfident` trust OR `Anxiety` alone → `Hold`
    /// - `Flow` or `Transition` + non-Uncertain trust → `Flow`
    /// - otherwise → `Hold`
    #[inline]
    pub fn gate_decision_i4(qualia: &QualiaI4_16D, signed_mantissa: i8) -> GateDecision {
        let texture = trust_texture_i4(qualia);
        let flow = flow_state_i4(qualia, signed_mantissa);

        match (texture, flow) {
            (TrustTexture::Uncertain, _) => {
                GateDecision::Block { reason: "uncertain trust: coherence low, tension high".to_string() }
            }
            (TrustTexture::Underconfident, FlowState::Anxiety) => {
                GateDecision::Block { reason: "underconfident + anxiety: execution blocked".to_string() }
            }
            (TrustTexture::Overconfident, _) => {
                GateDecision::Hold { reason: "overconfident trust: caution required".to_string() }
            }
            (_, FlowState::Anxiety) => {
                GateDecision::Hold { reason: "anxiety flow state: reduced autonomy".to_string() }
            }
            (TrustTexture::Calibrated | TrustTexture::Underconfident, FlowState::Flow | FlowState::Transition) => {
                GateDecision::Flow
            }
            _ => {
                GateDecision::Hold { reason: "boredom or moderate state: hold for re-evaluation".to_string() }
            }
        }
    }

    // ── MulAssessment ─────────────────────────────────────────────────────────

    /// Full MUL assessment from i4 qualia + signed mantissa.
    ///
    /// Combines `dk_position_i4`, `trust_texture_i4`, `flow_state_i4` into
    /// the existing `MulAssessment` struct. All fields are populated;
    /// `complexity_mapped` and `free_will_modifier` are derived from the
    /// i4 signals to produce a deterministic, zero-f64 result.
    ///
    /// `free_will_modifier` is approximated as a u8 fraction mapped to
    /// [0.0, 1.0] via the DK position × |mantissa| product, keeping the
    /// function free of heavy arithmetic while respecting the existing
    /// `f64` field type.
    pub fn mul_assess_i4(qualia: &QualiaI4_16D, signed_mantissa: i8) -> MulAssessment {
        let dk = dk_position_i4(qualia, signed_mantissa);
        let texture = trust_texture_i4(qualia);
        let flow = flow_state_i4(qualia, signed_mantissa);

        // TrustQualia.value: map texture + intensity to 0.0–1.0
        let intensity = intensity_i4(qualia); // i8 saturating product
        let trust_value: f64 = match texture {
            TrustTexture::Calibrated     => 0.75 + (intensity.clamp(0, 7) as f64 / 7.0) * 0.25,
            TrustTexture::Overconfident  => 0.45,
            TrustTexture::Underconfident => 0.40,
            TrustTexture::Uncertain      => 0.20,
        };

        let trust = TrustQualia { value: trust_value, texture };

        // complexity_mapped: coherence signal ≥ +2 implies the system can map complexity
        let coherence = qualia.get(DIM_COHERENCE);
        let complexity_mapped = coherence >= 2;

        // allostatic_load proxy: tension drives load (map i4 -8..+7 → 0.0..1.0)
        let tension = qualia.get(DIM_TENSION);
        let allostatic_load: f64 = ((tension as i16 + 8) as f64 / 15.0).clamp(0.0, 1.0);

        let homeostasis = Homeostasis { flow_state: flow, allostatic_load };

        // free_will_modifier: DK factor × trust_value × flow_factor
        let dk_factor: f64 = match dk {
            DkPosition::MountStupid          => 0.3,
            DkPosition::ValleyOfDespair      => 0.7,
            DkPosition::SlopeOfEnlightenment => 0.85,
            DkPosition::Plateau              => 1.0,
        };
        let flow_factor: f64 = match flow {
            FlowState::Flow       => 1.0,
            FlowState::Transition => 0.7,
            FlowState::Boredom    => 0.8,
            FlowState::Anxiety    => 0.5,
        };
        let free_will_modifier = (dk_factor * trust_value * flow_factor).clamp(0.0, 1.0);

        MulAssessment {
            trust,
            dk_position: dk,
            homeostasis,
            complexity_mapped,
            free_will_modifier,
        }
    }


    // ═══════════════════════════════════════════════════════════════════════
    // Batch evaluation API — D-CSV-13 (sprint-12)
    // ═══════════════════════════════════════════════════════════════════════

    /// Batch evaluation API for D-CSV-13.
    /// Processes N (qualia, mantissa) pairs in one call. Shape is SIMD-friendly:
    /// outputs are produced into pre-allocated `&mut [T]` buffers parallel to the
    /// inputs. Sprint-13+ replaces the scalar inner loop with AVX-512 i4 lane
    /// intrinsics; the API surface defined here is the contract that vectorization
    /// targets.
    pub mod batch {
        use super::*;

        /// Batch DK position: `qualia.len() == mantissas.len() == out.len()` must hold.
        /// Each output is the result of `dk_position_i4(qualia[i], mantissas[i])`.
        /// Panics on length mismatch.
        pub fn dk_position_batch(qualia: &[QualiaI4_16D], mantissas: &[i8], out: &mut [DkPosition]) {
            assert_eq!(qualia.len(), mantissas.len(), "qualia/mantissas length mismatch");
            assert_eq!(qualia.len(), out.len(), "input/output length mismatch");
            for i in 0..qualia.len() {
                out[i] = dk_position_i4(&qualia[i], mantissas[i]);
            }
        }

        /// Batch TrustTexture (qualia-only): for each qualia, compute trust_texture_i4.
        pub fn trust_texture_batch(qualia: &[QualiaI4_16D], out: &mut [TrustTexture]) {
            assert_eq!(qualia.len(), out.len());
            for i in 0..qualia.len() {
                out[i] = trust_texture_i4(&qualia[i]);
            }
        }

        /// Batch FlowState: parallel arrays of qualia + mantissas → flow states.
        pub fn flow_state_batch(qualia: &[QualiaI4_16D], mantissas: &[i8], out: &mut [FlowState]) {
            assert_eq!(qualia.len(), mantissas.len());
            assert_eq!(qualia.len(), out.len());
            for i in 0..qualia.len() {
                out[i] = flow_state_i4(&qualia[i], mantissas[i]);
            }
        }

        /// Batch GateDecision.
        pub fn gate_decision_batch(qualia: &[QualiaI4_16D], mantissas: &[i8], out: &mut [GateDecision]) {
            assert_eq!(qualia.len(), mantissas.len());
            assert_eq!(qualia.len(), out.len());
            for i in 0..qualia.len() {
                out[i] = gate_decision_i4(&qualia[i], mantissas[i]);
            }
        }

        /// Batch MulAssessment: the full pipeline.
        pub fn mul_assess_batch(qualia: &[QualiaI4_16D], mantissas: &[i8], out: &mut [MulAssessment]) {
            assert_eq!(qualia.len(), mantissas.len());
            assert_eq!(qualia.len(), out.len());
            for i in 0..qualia.len() {
                out[i] = mul_assess_i4(&qualia[i], mantissas[i]);
            }
        }

        /// Convenience: allocate the output Vec and return it (for non-hot-path callers).
        pub fn mul_assess_vec(qualia: &[QualiaI4_16D], mantissas: &[i8]) -> Vec<MulAssessment> {
            assert_eq!(qualia.len(), mantissas.len());
            let mut out = Vec::with_capacity(qualia.len());
            for i in 0..qualia.len() {
                out.push(mul_assess_i4(&qualia[i], mantissas[i]));
            }
            out
        }
    }

    // ═══════════════════════════════════════════════════════════════════════
    // Tests
    // ═══════════════════════════════════════════════════════════════════════

    #[cfg(test)]
    mod tests {
        use super::*;
        use crate::qualia::QualiaI4_16D;

        // Helper: build a qualia with specific named dims set; rest = 0.
        fn q_with(pairs: &[(usize, i8)]) -> QualiaI4_16D {
            let mut q = QualiaI4_16D::ZERO;
            for &(dim, val) in pairs {
                q.set(dim, val);
            }
            q
        }

        // ── DkPosition ────────────────────────────────────────────────────

        #[test]
        fn test_dk_position_i4_high_coherence_expert() {
            // coherence=+7, mantissa=+5 → Plateau
            let q = q_with(&[(DIM_COHERENCE, 7)]);
            assert_eq!(dk_position_i4(&q, 5), DkPosition::Plateau);
        }

        #[test]
        fn test_dk_position_i4_low_coherence_beginner() {
            // coherence=-3, mantissa=+1 → ValleyOfDespair
            let q = q_with(&[(DIM_COHERENCE, -3)]);
            assert_eq!(dk_position_i4(&q, 1), DkPosition::ValleyOfDespair);
        }

        #[test]
        fn test_dk_position_i4_neutral_intermediate() {
            // all-zero qualia + mantissa=+2 → ValleyOfDespair
            // (zero coherence fails the >=2 bar for SlopeOfEnlightenment,
            //  but |mantissa|=2 barely meets it; coherence=0 < 2, so we fall
            //  to ValleyOfDespair because coherence=0 <= -3 is false, but
            //  abs_mantissa=2 >= 2 and coherence=0 < 2, so we check:
            //  coherence=0 >= 5 → no; coherence=0 >= 2 → no (0<2);
            //  coherence=0 <= -3 → no; abs_mantissa=2 <= 1 → no;
            //  → MountStupid)
            let q = QualiaI4_16D::ZERO;
            assert_eq!(dk_position_i4(&q, 2), DkPosition::MountStupid);
        }

        // ── TrustTexture ──────────────────────────────────────────────────

        #[test]
        fn test_trust_texture_i4_crystalline() {
            // high coherence(+6) + high valence(+3) + low tension(0) → Calibrated
            let q = q_with(&[(DIM_COHERENCE, 6), (DIM_VALENCE, 3), (DIM_TENSION, 0)]);
            assert_eq!(trust_texture_i4(&q), TrustTexture::Calibrated);
        }

        #[test]
        fn test_trust_texture_i4_murky() {
            // low coherence(-5) + high tension(+5) → Uncertain
            let q = q_with(&[(DIM_COHERENCE, -5), (DIM_TENSION, 5)]);
            assert_eq!(trust_texture_i4(&q), TrustTexture::Uncertain);
        }

        #[test]
        fn test_trust_texture_i4_solid_calibrated() {
            // moderate coherence(+2) + moderate valence(+2) + moderate tension(+1) → Calibrated
            let q = q_with(&[(DIM_COHERENCE, 2), (DIM_VALENCE, 2), (DIM_TENSION, 1)]);
            assert_eq!(trust_texture_i4(&q), TrustTexture::Calibrated);
        }

        // ── FlowState ─────────────────────────────────────────────────────

        #[test]
        fn test_flow_state_i4_active() {
            // warmth(+5) + groundedness(+4) − tension(0) = proxy +9 → clamped fine; mantissa>0 → Flow
            let q = q_with(&[(DIM_WARMTH, 5), (DIM_GROUNDEDNESS, 4), (DIM_TENSION, 0)]);
            assert_eq!(flow_state_i4(&q, 3), FlowState::Flow);
        }

        #[test]
        fn test_flow_state_i4_stuck_negative_mantissa() {
            // coherence=-3 + mantissa=-4 → Anxiety
            let q = q_with(&[(DIM_COHERENCE, -3), (DIM_TENSION, 3)]);
            assert_eq!(flow_state_i4(&q, -4), FlowState::Anxiety);
        }

        // ── GateDecision ──────────────────────────────────────────────────

        #[test]
        fn test_gate_decision_i4_proceed() {
            // calibrated trust + flow state → GateDecision::Flow
            let q = q_with(&[
                (DIM_COHERENCE, 5),
                (DIM_VALENCE, 3),
                (DIM_TENSION, 0),
                (DIM_WARMTH, 5),
                (DIM_GROUNDEDNESS, 4),
            ]);
            let gate = gate_decision_i4(&q, 4);
            assert!(matches!(gate, GateDecision::Flow));
        }

        #[test]
        fn test_gate_decision_i4_block() {
            // uncertain trust (low coherence, high tension) → Block
            let q = q_with(&[(DIM_COHERENCE, -5), (DIM_TENSION, 5)]);
            let gate = gate_decision_i4(&q, 2);
            assert!(matches!(gate, GateDecision::Block { .. }));
        }

        // ── MulAssessment ─────────────────────────────────────────────────

        #[test]
        fn test_mul_assess_i4_combines_all_four() {
            // Strong expert signal: high coherence, high valence, low tension,
            // high warmth + groundedness, positive mantissa → all non-default fields
            let q = q_with(&[
                (DIM_COHERENCE, 6),
                (DIM_VALENCE, 5),
                (DIM_TENSION, 0),
                (DIM_WARMTH, 5),
                (DIM_GROUNDEDNESS, 5),
            ]);
            let mul = mul_assess_i4(&q, 5);
            assert_eq!(mul.dk_position, DkPosition::Plateau);
            assert_eq!(mul.trust.texture, TrustTexture::Calibrated);
            assert_eq!(mul.homeostasis.flow_state, FlowState::Flow);
            assert!(mul.free_will_modifier > 0.5, "expert+flow should give high autonomy");
            assert!(mul.complexity_mapped, "high coherence should map complexity");
        }

        #[test]
        fn test_mul_assess_i4_zero_qualia_zero_mantissa_default_path() {
            // All-zero input + zero mantissa → deterministic neutral baseline
            let q = QualiaI4_16D::ZERO;
            let mul = mul_assess_i4(&q, 0);
            // Zero coherence → not complexity_mapped
            assert!(!mul.complexity_mapped);
            // Zero mantissa (abs=0) → ValleyOfDespair
            assert_eq!(mul.dk_position, DkPosition::ValleyOfDespair);
            // free_will_modifier must be in [0.0, 1.0]
            assert!(mul.free_will_modifier >= 0.0 && mul.free_will_modifier <= 1.0);
            // Trust value must be > 0.0 (even uncertain has 0.20 floor)
            assert!(mul.trust.value > 0.0);
        }

        // ── Batch API tests (D-CSV-13) ────────────────────────────────────

        /// Helper: generate N deterministic qualia + mantissa pairs.
        fn make_batch(n: usize) -> (Vec<QualiaI4_16D>, Vec<i8>) {
            let pairs: &[(usize, i8, i8)] = &[
                // (dim_coherence, set_val, mantissa)
                (9, 7, 5),
                (9, 5, 4),
                (9, 3, 3),
                (9, 2, 2),
                (9, 0, 2),
                (9, -1, 1),
                (9, -3, -2),
                (9, -5, -4),
                (9, 6, 0),
                (9, 1, -1),
            ];
            let mut qualia = Vec::with_capacity(n);
            let mut mantissas = Vec::with_capacity(n);
            for i in 0..n {
                let (dim, coh, mant) = pairs[i % pairs.len()];
                qualia.push(QualiaI4_16D::ZERO.with(dim, coh));
                mantissas.push(mant);
            }
            (qualia, mantissas)
        }

        #[test]
        fn test_dk_position_batch_matches_scalar() {
            let (qualia, mantissas) = make_batch(10);
            let mut out = vec![DkPosition::MountStupid; 10];
            batch::dk_position_batch(&qualia, &mantissas, &mut out);
            for (i, (q, &m)) in qualia.iter().zip(mantissas.iter()).enumerate() {
                assert_eq!(out[i], dk_position_i4(q, m), "mismatch at index {}", i);
            }
        }

        #[test]
        fn test_trust_texture_batch_matches_scalar() {
            let (qualia, _) = make_batch(10);
            let mut out = vec![TrustTexture::Uncertain; 10];
            batch::trust_texture_batch(&qualia, &mut out);
            for (i, q) in qualia.iter().enumerate() {
                assert_eq!(out[i], trust_texture_i4(q), "mismatch at index {}", i);
            }
        }

        #[test]
        fn test_flow_state_batch_matches_scalar() {
            let (qualia, mantissas) = make_batch(10);
            let mut out = vec![FlowState::Boredom; 10];
            batch::flow_state_batch(&qualia, &mantissas, &mut out);
            for (i, (q, &m)) in qualia.iter().zip(mantissas.iter()).enumerate() {
                assert_eq!(out[i], flow_state_i4(q, m), "mismatch at index {}", i);
            }
        }

        #[test]
        fn test_gate_decision_batch_matches_scalar() {
            let (qualia, mantissas) = make_batch(10);
            let mut out: Vec<GateDecision> = (0..10).map(|_| GateDecision::Flow).collect();
            batch::gate_decision_batch(&qualia, &mantissas, &mut out);
            for (i, (q, &m)) in qualia.iter().zip(mantissas.iter()).enumerate() {
                let scalar = gate_decision_i4(q, m);
                // Compare discriminant since GateDecision carries String fields
                assert!(
                    matches_gate_discriminant(&out[i], &scalar),
                    "gate decision discriminant mismatch at index {}: batch={:?} scalar={:?}",
                    i, out[i], scalar
                );
            }
        }

        fn matches_gate_discriminant(a: &GateDecision, b: &GateDecision) -> bool {
            matches!((a, b),
                (GateDecision::Flow, GateDecision::Flow)
                | (GateDecision::Hold { .. }, GateDecision::Hold { .. })
                | (GateDecision::Block { .. }, GateDecision::Block { .. })
            )
        }

        #[test]
        fn test_mul_assess_batch_matches_scalar() {
            let (qualia, mantissas) = make_batch(10);
            let mut out: Vec<MulAssessment> = (0..10)
                .map(|_| mul_assess_i4(&QualiaI4_16D::ZERO, 0))
                .collect();
            batch::mul_assess_batch(&qualia, &mantissas, &mut out);
            for (i, (q, &m)) in qualia.iter().zip(mantissas.iter()).enumerate() {
                let scalar = mul_assess_i4(q, m);
                assert_eq!(out[i].dk_position, scalar.dk_position, "dk_position mismatch at {}", i);
                assert_eq!(out[i].trust.texture, scalar.trust.texture, "trust.texture mismatch at {}", i);
                assert_eq!(out[i].homeostasis.flow_state, scalar.homeostasis.flow_state, "flow_state mismatch at {}", i);
                assert!((out[i].free_will_modifier - scalar.free_will_modifier).abs() < 1e-10,
                    "free_will_modifier mismatch at {}", i);
            }
        }

        #[test]
        fn test_mul_assess_vec_allocates_correctly() {
            let (qualia, mantissas) = make_batch(10);
            let result = batch::mul_assess_vec(&qualia, &mantissas);
            assert_eq!(result.len(), qualia.len(), "output length must equal input length");
            for (i, (q, &m)) in qualia.iter().zip(mantissas.iter()).enumerate() {
                let scalar = mul_assess_i4(q, m);
                assert_eq!(result[i].dk_position, scalar.dk_position, "dk_position mismatch at {}", i);
                assert_eq!(result[i].trust.texture, scalar.trust.texture, "trust.texture mismatch at {}", i);
                assert_eq!(result[i].homeostasis.flow_state, scalar.homeostasis.flow_state, "flow_state mismatch at {}", i);
            }
        }

        #[test]
        fn test_batch_panic_on_length_mismatch() {
            let qualia = vec![QualiaI4_16D::ZERO; 3];
            let mantissas = vec![0i8; 2]; // intentional mismatch
            let mut out = vec![DkPosition::MountStupid; 3];
            let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                batch::dk_position_batch(&qualia, &mantissas, &mut out);
            }));
            assert!(result.is_err(), "must panic on qualia/mantissas length mismatch");
        }

        #[test]
        fn test_batch_empty_input_returns_empty_output() {
            let qualia: Vec<QualiaI4_16D> = vec![];
            let mantissas: Vec<i8> = vec![];
            let mut out_dk: Vec<DkPosition> = vec![];
            let mut out_tt: Vec<TrustTexture> = vec![];
            let mut out_fs: Vec<FlowState> = vec![];
            let mut out_gd: Vec<GateDecision> = vec![];
            let mut out_ma: Vec<MulAssessment> = vec![];

            // None of these should panic
            batch::dk_position_batch(&qualia, &mantissas, &mut out_dk);
            batch::trust_texture_batch(&qualia, &mut out_tt);
            batch::flow_state_batch(&qualia, &mantissas, &mut out_fs);
            batch::gate_decision_batch(&qualia, &mantissas, &mut out_gd);
            batch::mul_assess_batch(&qualia, &mantissas, &mut out_ma);
            let vec_result = batch::mul_assess_vec(&qualia, &mantissas);

            assert_eq!(out_dk.len(), 0);
            assert_eq!(out_tt.len(), 0);
            assert_eq!(out_fs.len(), 0);
            assert_eq!(out_gd.len(), 0);
            assert_eq!(out_ma.len(), 0);
            assert_eq!(vec_result.len(), 0);
        }
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
