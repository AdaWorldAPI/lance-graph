//! Flow & Homeostasis — Qualia awareness of cognitive state.
//!
//! Based on Csikszentmihalyi's flow model:
//! - FLOW: challenge ≈ skill (optimal operation)
//! - ANXIETY: challenge > skill (overwhelmed)
//! - BOREDOM: skill > challenge (understimulated)
//! - APATHY: neither challenged nor skilled (disengaged)
//!
//! If depleted → RECOVER first.

use super::SituationInput;

/// Cognitive flow state.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FlowState {
    /// Challenge ≈ skill: optimal operation.
    Flow,
    /// Challenge > skill: overwhelmed.
    Anxiety,
    /// Skill > challenge: understimulated.
    Boredom,
    /// Neither challenged nor skilled: disengaged.
    Apathy,
}

/// Homeostasis assessment.
#[derive(Debug, Clone)]
pub struct Homeostasis {
    pub state: FlowState,
    /// Allostatic load (cumulative deviation, 0..1). High = depleted.
    pub allostatic_load: f64,
    /// Whether recovery is needed before proceeding.
    pub needs_recovery: bool,
    /// Recommended corrective action.
    pub corrective: CorrectiveAction,
}

/// Corrective actions for homeostasis.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CorrectiveAction {
    /// Stay the course.
    Maintain,
    /// Reduce complexity.
    Simplify,
    /// Increase difficulty.
    Challenge,
    /// Re-engage with the task.
    Reengage,
    /// Rest before continuing.
    Rest,
}

impl Homeostasis {
    /// Flow factor: how much homeostasis supports autonomous operation.
    pub fn flow_factor(&self) -> f64 {
        if self.needs_recovery {
            return 0.3; // Heavily penalized when depleted
        }

        match self.state {
            FlowState::Flow => 1.0,
            FlowState::Anxiety => 0.6,
            FlowState::Boredom => 0.8,
            FlowState::Apathy => 0.4,
        }
    }
}

/// Assess homeostasis from situation input.
pub fn assess(input: &SituationInput) -> Homeostasis {
    let challenge = input.challenge_level;
    let skill = input.skill_level;
    let load = input.allostatic_load;

    // Determine flow state from challenge/skill balance
    let state = if (challenge - skill).abs() < 0.15 && challenge > 0.3 {
        FlowState::Flow
    } else if challenge > skill + 0.2 {
        FlowState::Anxiety
    } else if skill > challenge + 0.2 {
        FlowState::Boredom
    } else {
        FlowState::Apathy
    };

    let needs_recovery = load > 0.7;

    let corrective = if needs_recovery {
        CorrectiveAction::Rest
    } else {
        match state {
            FlowState::Flow => CorrectiveAction::Maintain,
            FlowState::Anxiety => CorrectiveAction::Simplify,
            FlowState::Boredom => CorrectiveAction::Challenge,
            FlowState::Apathy => CorrectiveAction::Reengage,
        }
    };

    Homeostasis {
        state,
        allostatic_load: load,
        needs_recovery,
        corrective,
    }
}
