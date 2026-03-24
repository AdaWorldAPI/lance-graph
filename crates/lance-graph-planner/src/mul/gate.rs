//! GATE: Can proceed to compass?
//!
//! Checklist:
//! - [ ] Not Mount Stupid
//! - [ ] Complexity mapped
//! - [ ] Not depleted
//! - [ ] Trust not murky/dissonant

use super::{MulAssessment, dk::DkPosition, trust::TrustTexture, homeostasis::FlowState};

/// Gate decision.
#[derive(Debug, Clone)]
pub enum GateDecision {
    /// All checks pass. Proceed with free will modifier applied.
    Proceed {
        free_will_modifier: f64,
    },
    /// Gate blocked. Need sandbox or human assistance.
    Sandbox {
        reason: String,
    },
    /// Need compass function for navigation in unknown territory.
    Compass,
}

/// Check the gate conditions.
pub fn check(assessment: &MulAssessment) -> GateDecision {
    // Check 1: Not Mount Stupid
    if assessment.dk_position == DkPosition::MountStupid {
        return GateDecision::Sandbox {
            reason: "Dunning-Kruger: Mount Stupid detected. Learn first.".into(),
        };
    }

    // Check 2: Complexity mapped
    if !assessment.complexity_mapped {
        return GateDecision::Sandbox {
            reason: "Complexity not mapped. Map dimensions before proceeding.".into(),
        };
    }

    // Check 3: Not depleted
    if assessment.homeostasis.needs_recovery {
        return GateDecision::Sandbox {
            reason: "Depleted. Recover homeostasis first.".into(),
        };
    }

    // Check 4: Trust not murky/dissonant
    match assessment.trust.texture {
        TrustTexture::Murky => {
            return GateDecision::Compass; // Can navigate with compass
        }
        TrustTexture::Dissonant => {
            return GateDecision::Sandbox {
                reason: "Trust dissonant. Resolve dissonance before proceeding.".into(),
            };
        }
        _ => {}
    }

    // Check 5: If trust is fuzzy, use compass for navigation
    if assessment.trust.texture == TrustTexture::Fuzzy {
        return GateDecision::Compass;
    }

    // All checks pass
    GateDecision::Proceed {
        free_will_modifier: assessment.free_will_modifier,
    }
}
