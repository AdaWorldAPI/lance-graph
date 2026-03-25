//! Dynamic Elevation — cost model that smells resistance.
//!
//! Instead of guessing execution cost upfront, the planner starts cheap
//! and ELEVATES when it observes resistance: latency exceeding the
//! thinking style's patience budget.
//!
//! ## Elevation Stack
//!
//! ```text
//! Level 0: POINT     — single lookup, ~100ns (BindSpace hot path)
//! Level 1: SCAN      — palette_distance over neighborhood, ~10μs
//! Level 2: CASCADE   — CAM-PQ 3-stroke, ~100μs per 1M candidates
//! Level 3: BATCH     — morsel pipeline with backpressure, ~10ms
//! Level 4: IVF+BATCH — IVF partition → batch per partition, ~100ms
//! Level 5: ASYNC     — fire-and-forget, results via SSE, seconds+
//! ```

pub mod budget;
pub mod homeostasis;
pub mod operator;
pub mod decompose;
pub mod learning;

use std::time::{Duration, Instant};

/// Execution level in the elevation stack.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(u8)]
pub enum ElevationLevel {
    /// Single lookup — ~100ns. BindSpace hot path or literal SPO match.
    Point = 0,
    /// palette_distance over local neighborhood — ~10μs.
    Scan = 1,
    /// CAM-PQ 3-stroke cascade — ~100μs per 1M candidates.
    Cascade = 2,
    /// Morsel pipeline with backpressure — ~10ms.
    Batch = 3,
    /// IVF coarse partition → batch per partition — ~100ms.
    IvfBatch = 4,
    /// Fire-and-forget, results via SSE/webhook — seconds to minutes.
    Async = 5,
}

impl ElevationLevel {
    /// Next level up. Saturates at Async.
    pub fn next(self) -> Self {
        match self {
            Self::Point => Self::Scan,
            Self::Scan => Self::Cascade,
            Self::Cascade => Self::Batch,
            Self::Batch => Self::IvfBatch,
            Self::IvfBatch => Self::Async,
            Self::Async => Self::Async,
        }
    }

    /// Previous level down. Saturates at Point.
    pub fn prev(self) -> Self {
        match self {
            Self::Point => Self::Point,
            Self::Scan => Self::Point,
            Self::Cascade => Self::Scan,
            Self::Batch => Self::Cascade,
            Self::IvfBatch => Self::Batch,
            Self::Async => Self::IvfBatch,
        }
    }

    /// Typical latency order of magnitude for this level.
    pub fn typical_latency(self) -> Duration {
        match self {
            Self::Point => Duration::from_nanos(100),
            Self::Scan => Duration::from_micros(10),
            Self::Cascade => Duration::from_micros(100),
            Self::Batch => Duration::from_millis(10),
            Self::IvfBatch => Duration::from_millis(100),
            Self::Async => Duration::from_secs(1),
        }
    }

    /// All levels in order.
    pub const ALL: [ElevationLevel; 6] = [
        Self::Point, Self::Scan, Self::Cascade,
        Self::Batch, Self::IvfBatch, Self::Async,
    ];
}

impl std::fmt::Display for ElevationLevel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Point => write!(f, "L0:Point"),
            Self::Scan => write!(f, "L1:Scan"),
            Self::Cascade => write!(f, "L2:Cascade"),
            Self::Batch => write!(f, "L3:Batch"),
            Self::IvfBatch => write!(f, "L4:IVF+Batch"),
            Self::Async => write!(f, "L5:Async"),
        }
    }
}

/// What triggered an elevation decision.
#[derive(Debug, Clone)]
pub struct ElevationTrigger {
    /// How many results came back from current level.
    pub result_count: usize,
    /// Wall-clock time spent at current level.
    pub elapsed: Duration,
    /// Memory pressure (bytes allocated for intermediate results).
    pub memory_bytes: usize,
}

/// A recorded elevation event (for the learning loop).
#[derive(Debug, Clone)]
pub struct ElevationEvent {
    /// Level we were at.
    pub from_level: ElevationLevel,
    /// Level we elevated to.
    pub to_level: ElevationLevel,
    /// What triggered the elevation.
    pub trigger: ElevationTrigger,
    /// When it happened.
    pub timestamp: Instant,
}

/// Should the system elevate to the next level?
pub fn should_elevate(trigger: &ElevationTrigger, budget: &budget::PatienceBudget) -> bool {
    trigger.elapsed > budget.latency_budget
        || trigger.result_count > budget.result_threshold
        || trigger.memory_bytes > budget.memory_budget
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_level_ordering() {
        assert!(ElevationLevel::Point < ElevationLevel::Scan);
        assert!(ElevationLevel::Scan < ElevationLevel::Cascade);
        assert!(ElevationLevel::Cascade < ElevationLevel::Batch);
        assert!(ElevationLevel::Batch < ElevationLevel::IvfBatch);
        assert!(ElevationLevel::IvfBatch < ElevationLevel::Async);
    }

    #[test]
    fn test_level_next_prev() {
        assert_eq!(ElevationLevel::Point.next(), ElevationLevel::Scan);
        assert_eq!(ElevationLevel::Async.next(), ElevationLevel::Async);
        assert_eq!(ElevationLevel::Point.prev(), ElevationLevel::Point);
        assert_eq!(ElevationLevel::Batch.prev(), ElevationLevel::Cascade);
    }

    #[test]
    fn test_should_elevate_latency() {
        let budget = budget::PatienceBudget {
            latency_budget: Duration::from_millis(10),
            result_threshold: 10_000,
            memory_budget: 64 * 1024 * 1024,
            ceiling: ElevationLevel::Batch,
        };
        // Under budget — don't elevate
        let trigger = ElevationTrigger {
            result_count: 100,
            elapsed: Duration::from_millis(5),
            memory_bytes: 1024,
        };
        assert!(!should_elevate(&trigger, &budget));

        // Over latency — elevate
        let trigger = ElevationTrigger {
            result_count: 100,
            elapsed: Duration::from_millis(50),
            memory_bytes: 1024,
        };
        assert!(should_elevate(&trigger, &budget));
    }

    #[test]
    fn test_should_elevate_result_count() {
        let budget = budget::PatienceBudget {
            latency_budget: Duration::from_secs(10),
            result_threshold: 1_000,
            memory_budget: 1_000_000_000,
            ceiling: ElevationLevel::Async,
        };
        let trigger = ElevationTrigger {
            result_count: 5_000,
            elapsed: Duration::from_millis(1),
            memory_bytes: 0,
        };
        assert!(should_elevate(&trigger, &budget));
    }
}
