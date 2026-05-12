//! Homeostasis → Patience Budget modification.
//!
//! The MUL layer's Flow & Homeostasis state modifies the patience budget
//! at runtime. Anxious system → elevate faster. Bored system → go deeper.

use super::budget::PatienceBudget;
use super::ElevationLevel;
use crate::mul::homeostasis::FlowState;
use std::time::Duration;

/// Modify a patience budget based on current homeostasis flow state.
///
/// - Anxiety: system is overwhelmed → cut latency budget to 25%, cap ceiling at Cascade
/// - Flow: optimal → use default budget unchanged
/// - Boredom: system is idle → 4× latency budget, 4× result threshold (go deeper)
/// - Apathy: depleted → minimal budget, cap at Scan (do almost nothing)
pub fn modify_budget(state: FlowState, base: PatienceBudget) -> PatienceBudget {
    match state {
        FlowState::Anxiety => PatienceBudget {
            latency_budget: base.latency_budget / 4,
            result_threshold: base.result_threshold / 2,
            memory_budget: base.memory_budget / 2,
            ceiling: base.ceiling.min(ElevationLevel::Cascade),
        },

        FlowState::Flow => base,

        FlowState::Boredom => PatienceBudget {
            latency_budget: base.latency_budget.saturating_mul(4),
            result_threshold: base.result_threshold.saturating_mul(4),
            memory_budget: base.memory_budget.saturating_mul(2),
            ..base
        },

        FlowState::Apathy => PatienceBudget {
            latency_budget: Duration::from_millis(1),
            result_threshold: 10,
            memory_budget: 1024 * 1024, // 1MB
            ceiling: ElevationLevel::Scan,
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::elevation::budget::budget_for_cluster;
    use crate::thinking::style::ThinkingCluster;

    #[test]
    fn test_flow_preserves_budget() {
        let base = budget_for_cluster(ThinkingCluster::Convergent);
        let modified = modify_budget(FlowState::Flow, base.clone());
        assert_eq!(modified.latency_budget, base.latency_budget);
        assert_eq!(modified.ceiling, base.ceiling);
    }

    #[test]
    fn test_anxiety_reduces_latency() {
        let base = budget_for_cluster(ThinkingCluster::Convergent);
        let modified = modify_budget(FlowState::Anxiety, base.clone());
        assert!(modified.latency_budget < base.latency_budget);
        assert!(modified.ceiling <= ElevationLevel::Cascade);
    }

    #[test]
    fn test_boredom_increases_patience() {
        let base = budget_for_cluster(ThinkingCluster::Speed);
        let modified = modify_budget(FlowState::Boredom, base.clone());
        assert!(modified.latency_budget > base.latency_budget);
        assert!(modified.result_threshold > base.result_threshold);
    }

    #[test]
    fn test_apathy_caps_at_scan() {
        let base = budget_for_cluster(ThinkingCluster::Divergent);
        let modified = modify_budget(FlowState::Apathy, base);
        assert_eq!(modified.ceiling, ElevationLevel::Scan);
        assert!(modified.result_threshold <= 10);
    }
}
