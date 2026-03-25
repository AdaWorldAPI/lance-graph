//! Patience Budget — thinking style → elevation thresholds.
//!
//! Each thinking cluster has different tolerance for latency, fan-out,
//! and memory before elevating to a more expensive execution level.

use std::time::Duration;
use super::ElevationLevel;
use crate::thinking::style::ThinkingCluster;

/// How much patience the planner has before elevating.
#[derive(Debug, Clone)]
pub struct PatienceBudget {
    /// Max wall-clock before elevating (from thinking style).
    pub latency_budget: Duration,
    /// Max results before elevating (fan-out control).
    pub result_threshold: usize,
    /// Max memory before elevating (bytes).
    pub memory_budget: usize,
    /// Max level this query is allowed to reach.
    pub ceiling: ElevationLevel,
}

impl PatienceBudget {
    /// Scale latency budget by a multiplier.
    pub fn with_latency_scale(mut self, scale: f64) -> Self {
        let nanos = (self.latency_budget.as_nanos() as f64 * scale) as u64;
        self.latency_budget = Duration::from_nanos(nanos);
        self
    }

    /// Scale result threshold by a multiplier.
    pub fn with_result_scale(mut self, scale: f64) -> Self {
        self.result_threshold = (self.result_threshold as f64 * scale) as usize;
        self
    }

    /// Cap the ceiling at a given level.
    pub fn with_ceiling(mut self, ceiling: ElevationLevel) -> Self {
        self.ceiling = ceiling.min(self.ceiling);
        self
    }
}

impl Default for PatienceBudget {
    fn default() -> Self {
        Self {
            latency_budget: Duration::from_millis(100),
            result_threshold: 10_000,
            memory_budget: 64 * 1024 * 1024, // 64MB
            ceiling: ElevationLevel::Batch,
        }
    }
}

/// Map thinking cluster to patience budget.
///
/// The cluster determines the personality of elevation:
/// - Convergent: deep, precise — let it run, don't elevate prematurely
/// - Divergent: wide exploration — tolerate high fan-out
/// - Attention: relationship focus — low fan-out, precise matches
/// - Speed: System 1 fast path — elevate quickly, approximate answers
pub fn budget_for_cluster(cluster: ThinkingCluster) -> PatienceBudget {
    match cluster {
        // Deep, precise — high patience, high ceiling
        ThinkingCluster::Convergent => PatienceBudget {
            latency_budget: Duration::from_millis(500),
            result_threshold: 100_000,
            memory_budget: 256 * 1024 * 1024, // 256MB
            ceiling: ElevationLevel::IvfBatch,
        },
        // Wide exploration — moderate time, high fan-out, allows async
        ThinkingCluster::Divergent => PatienceBudget {
            latency_budget: Duration::from_millis(200),
            result_threshold: 50_000,
            memory_budget: 128 * 1024 * 1024, // 128MB
            ceiling: ElevationLevel::Async,
        },
        // Relationship/qualia focus — low fan-out, precise matches
        ThinkingCluster::Attention => PatienceBudget {
            latency_budget: Duration::from_millis(100),
            result_threshold: 500,
            memory_budget: 32 * 1024 * 1024, // 32MB
            ceiling: ElevationLevel::Cascade,
        },
        // System 1 fast path — elevate quickly, get approximate answers
        ThinkingCluster::Speed => PatienceBudget {
            latency_budget: Duration::from_millis(10),
            result_threshold: 1_000,
            memory_budget: 16 * 1024 * 1024, // 16MB
            ceiling: ElevationLevel::Cascade,
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_convergent_has_high_patience() {
        let budget = budget_for_cluster(ThinkingCluster::Convergent);
        assert!(budget.latency_budget >= Duration::from_millis(500));
        assert!(budget.result_threshold >= 100_000);
        assert!(budget.ceiling >= ElevationLevel::IvfBatch);
    }

    #[test]
    fn test_speed_has_low_patience() {
        let budget = budget_for_cluster(ThinkingCluster::Speed);
        assert!(budget.latency_budget <= Duration::from_millis(10));
        assert!(budget.result_threshold <= 1_000);
        assert!(budget.ceiling <= ElevationLevel::Cascade);
    }

    #[test]
    fn test_divergent_allows_async() {
        let budget = budget_for_cluster(ThinkingCluster::Divergent);
        assert_eq!(budget.ceiling, ElevationLevel::Async);
    }

    #[test]
    fn test_attention_low_fanout() {
        let budget = budget_for_cluster(ThinkingCluster::Attention);
        assert!(budget.result_threshold <= 500);
    }

    #[test]
    fn test_budget_with_ceiling() {
        let budget = budget_for_cluster(ThinkingCluster::Divergent)
            .with_ceiling(ElevationLevel::Batch);
        assert_eq!(budget.ceiling, ElevationLevel::Batch);
    }

    #[test]
    fn test_budget_latency_scale() {
        let budget = budget_for_cluster(ThinkingCluster::Convergent)
            .with_latency_scale(0.25);
        assert!(budget.latency_budget <= Duration::from_millis(130));
    }
}
