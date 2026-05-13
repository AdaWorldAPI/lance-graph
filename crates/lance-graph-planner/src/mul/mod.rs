//! # Meta-Uncertainty Layer (MUL) + Compass Function
//!
//! "I know that I don't know — and I know HOW MUCH I don't know."
//!
//! This is the outer loop of the unified planner. Before any query planning
//! happens, MUL assesses whether planning should proceed at all.
//!
//! ## Phases
//! 1. Trust Qualia + Dunning-Kruger Detector (parallel inputs)
//! 2. Complexity Mapper + 9-Dot Epiphany Layer (parallel processors)
//! 3. Chosen Inconfdence + Flow & Homeostasis (calibration)
//! 4. Impact Ceiling & Human Humility Factor
//! 5. GATE: Can proceed to compass?
//! 6. Compass Function (navigation when the map runs out)
//! 7. Post-Action Learning Loop

pub mod compass;
pub mod dk;
pub mod gate;
pub mod homeostasis;
pub mod trust;

pub use compass::{CompassDecision, CompassResult};
pub use dk::{DkDetector, DkPosition};
pub use gate::GateDecision;
pub use homeostasis::{FlowState, Homeostasis};
pub use trust::{TrustQualia, TrustTexture};

/// Input to the MUL assessment.
#[derive(Debug, Clone)]
pub struct SituationInput {
    /// Felt competence (self-reported, 0..1).
    pub felt_competence: f64,
    /// Demonstrated competence (measured, e.g. Brier score, 0..1).
    pub demonstrated_competence: f64,
    /// Source reliability (NARS confidence of data source, 0..1).
    pub source_reliability: f64,
    /// Environment stability (input entropy, 0..1 where 1=stable).
    pub environment_stability: f64,
    /// Calibration accuracy (historical Brier error, 0..1 where 1=perfect).
    pub calibration_accuracy: f64,
    /// Current challenge level (0..1).
    pub challenge_level: f64,
    /// Current skill level (0..1).
    pub skill_level: f64,
    /// Allostatic load (cumulative deviation from set-point, 0..1).
    pub allostatic_load: f64,
    /// Max acceptable damage (impact ceiling).
    pub max_acceptable_damage: f64,
    /// Reversibility requirement (0..1 where 1=must be fully reversible).
    pub reversibility_requirement: f64,
    /// Is a sandbox available?
    pub sandbox_available: bool,
    /// Domain complexity estimate: [known_dimensions] / [estimated_total].
    pub complexity_ratio: f64,
    /// Graph density of interdependencies (0..1).
    pub interdependency_density: f64,
}

impl Default for SituationInput {
    fn default() -> Self {
        Self {
            felt_competence: 0.5,
            demonstrated_competence: 0.5,
            source_reliability: 0.7,
            environment_stability: 0.8,
            calibration_accuracy: 0.6,
            challenge_level: 0.5,
            skill_level: 0.5,
            allostatic_load: 0.0,
            max_acceptable_damage: 0.5,
            reversibility_requirement: 0.5,
            sandbox_available: false,
            complexity_ratio: 0.5,
            interdependency_density: 0.3,
        }
    }
}

/// Complete MUL assessment result.
#[derive(Debug, Clone)]
pub struct MulAssessment {
    pub trust: TrustQualia,
    pub dk_position: DkPosition,
    pub homeostasis: Homeostasis,
    pub complexity_mapped: bool,
    pub free_will_modifier: f64,
}

/// Assess the current situation through the MUL.
pub fn assess(input: &SituationInput) -> MulAssessment {
    // Phase 1: Trust Qualia + DK Detector (parallel)
    let trust = trust::assess(input);
    let dk_position = dk::detect(input);

    // Phase 2: Complexity mapping
    let complexity_mapped = input.complexity_ratio > 0.3; // At least 30% of dimensions known

    // Phase 3: Homeostasis
    let homeostasis = homeostasis::assess(input);

    // Phase 4: Free Will Modifier
    // 1.0 × DK_factor × Trust_factor × Complexity_factor × Flow_factor
    let dk_factor = dk_position.humility_factor();
    let trust_factor = trust.composite_score();
    let complexity_factor = if complexity_mapped {
        0.8 + 0.2 * input.complexity_ratio
    } else {
        0.4
    };
    let flow_factor = homeostasis.flow_factor();

    let free_will_modifier = dk_factor * trust_factor * complexity_factor * flow_factor;

    MulAssessment {
        trust,
        dk_position,
        homeostasis,
        complexity_mapped,
        free_will_modifier,
    }
}

/// Gate check: determine whether to proceed, sandbox, or use compass.
pub fn gate_check(assessment: &MulAssessment) -> GateDecision {
    gate::check(assessment)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_assessment() {
        let input = SituationInput::default();
        let assessment = assess(&input);

        // Default should be moderate trust, valley of despair or slope
        assert!(assessment.free_will_modifier > 0.0);
        assert!(assessment.free_will_modifier <= 1.0);
    }

    #[test]
    fn test_mount_stupid_blocks() {
        let input = SituationInput {
            felt_competence: 0.95,
            demonstrated_competence: 0.1,
            ..Default::default()
        };
        let assessment = assess(&input);
        let gate = gate_check(&assessment);

        // Mount Stupid should block
        matches!(gate, GateDecision::Sandbox { .. });
    }

    #[test]
    fn test_high_trust_proceeds() {
        let input = SituationInput {
            felt_competence: 0.8,
            demonstrated_competence: 0.85,
            source_reliability: 0.9,
            environment_stability: 0.9,
            calibration_accuracy: 0.85,
            challenge_level: 0.6,
            skill_level: 0.7,
            complexity_ratio: 0.7,
            ..Default::default()
        };
        let assessment = assess(&input);
        let gate = gate_check(&assessment);

        matches!(gate, GateDecision::Proceed { .. });
    }
}
