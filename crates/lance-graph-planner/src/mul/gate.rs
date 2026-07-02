//! GATE: Can proceed to compass?
//!
//! Checklist:
//! - [ ] Not Mount Stupid
//! - [ ] Complexity mapped
//! - [ ] Not depleted
//! - [ ] Trust not murky/dissonant

#[allow(unused_imports)] // FlowState intended for homeostasis gate wiring
use super::{dk::DkPosition, homeostasis::FlowState, trust::TrustTexture, MulAssessment};

/// Gate decision.
///
/// Renamed from `GateDecision` (M15) — this is the planner's
/// Meta-Uncertainty-Layer verdict (Proceed/Sandbox/Compass), NOT the
/// contract's kanban gate (`lance_graph_contract::mul::GateDecision
/// {Flow, Hold, Block}`) which `KanbanColumn::advance_on_gate` consumes.
#[derive(Debug, Clone)]
pub enum MulGateDecision {
    /// All checks pass. Proceed with free will modifier applied.
    Proceed { free_will_modifier: f64 },
    /// Gate blocked. Need sandbox or human assistance.
    Sandbox { reason: String },
    /// Need compass function for navigation in unknown territory.
    Compass,
}

/// Renamed to `MulGateDecision` (M15); the unqualified name collided with
/// `lance_graph_contract::mul::GateDecision`.
#[deprecated(
    note = "renamed to MulGateDecision (M15); the unqualified name collided with contract mul::GateDecision"
)]
pub type GateDecision = MulGateDecision;

/// Check the gate conditions.
pub fn check(assessment: &MulAssessment) -> MulGateDecision {
    // Check 1: Not Mount Stupid
    if assessment.dk_position == DkPosition::MountStupid {
        return MulGateDecision::Sandbox {
            reason: "Dunning-Kruger: Mount Stupid detected. Learn first.".into(),
        };
    }

    // Check 2: Complexity mapped
    if !assessment.complexity_mapped {
        return MulGateDecision::Sandbox {
            reason: "Complexity not mapped. Map dimensions before proceeding.".into(),
        };
    }

    // Check 3: Not depleted
    if assessment.homeostasis.needs_recovery {
        return MulGateDecision::Sandbox {
            reason: "Depleted. Recover homeostasis first.".into(),
        };
    }

    // Check 4: Trust not murky/dissonant
    match assessment.trust.texture {
        TrustTexture::Murky => {
            return MulGateDecision::Compass; // Can navigate with compass
        }
        TrustTexture::Dissonant => {
            return MulGateDecision::Sandbox {
                reason: "Trust dissonant. Resolve dissonance before proceeding.".into(),
            };
        }
        _ => {}
    }

    // Check 5: If trust is fuzzy, use compass for navigation
    if assessment.trust.texture == TrustTexture::Fuzzy {
        return MulGateDecision::Compass;
    }

    // All checks pass
    MulGateDecision::Proceed {
        free_will_modifier: assessment.free_will_modifier,
    }
}
