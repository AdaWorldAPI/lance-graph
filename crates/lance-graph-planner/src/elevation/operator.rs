//! ElevatingOp — physical operator that wraps any scan/traverse with dynamic elevation.
//!
//! Starts at Level 0 and elevates when it observes resistance (latency, fan-out, memory).
//! Partial results transfer to the next level via `seed_results`.

use std::collections::BTreeMap;
use std::time::Instant;

use super::budget::PatienceBudget;
use super::learning::ElevationHistory;
use super::{ElevationEvent, ElevationLevel, ElevationTrigger, should_elevate};
use crate::physical::{Morsel, PhysicalOperator};

/// A level-specific operator that can be seeded with partial results.
pub trait LevelOperator: std::fmt::Debug + Send + Sync {
    /// Execute at this level, returning a morsel of results.
    fn execute_level(&mut self) -> Result<Morsel, ElevationError>;

    /// Seed this operator with partial results from a lower level.
    /// The higher-level operator can skip work already done.
    fn seed_results(&mut self, partial: &Morsel);

    /// Estimated cardinality at this level.
    fn cardinality(&self) -> f64;
}

/// Errors from elevation.
#[derive(Debug, Clone)]
pub enum ElevationError {
    /// No operator registered for this level.
    NoPlanForLevel(ElevationLevel),
    /// Execution timed out at ceiling level.
    Timeout { level: ElevationLevel, elapsed_ms: u64 },
    /// Execution error at a specific level.
    LevelError { level: ElevationLevel, message: String },
}

impl std::fmt::Display for ElevationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NoPlanForLevel(l) => write!(f, "No plan for elevation {l}"),
            Self::Timeout { level, elapsed_ms } => {
                write!(f, "Timeout at {level} after {elapsed_ms}ms")
            }
            Self::LevelError { level, message } => {
                write!(f, "Error at {level}: {message}")
            }
        }
    }
}

impl std::error::Error for ElevationError {}

/// Physical operator that dynamically elevates execution level.
///
/// Wraps a set of level-specific operators. Starts at `start_level`,
/// executes, and elevates when the patience budget is exceeded.
/// Partial results transfer between levels — work already done is not repeated.
#[derive(Debug)]
pub struct ElevatingOp {
    /// Current execution level.
    pub level: ElevationLevel,
    /// Starting level (for history recording).
    #[allow(dead_code)] // future wiring for elevation history tracking
    start_level: ElevationLevel,
    /// Patience budget (from thinking style + homeostasis).
    pub budget: PatienceBudget,
    /// Operators for each level (not all need to be populated).
    levels: BTreeMap<ElevationLevel, Box<dyn LevelOperator>>,
    /// Elevation history for this query.
    pub history: ElevationHistory,
}

impl ElevatingOp {
    /// Create a new elevating operator starting at the given level.
    pub fn new(start_level: ElevationLevel, budget: PatienceBudget) -> Self {
        Self {
            level: start_level,
            start_level,
            budget,
            levels: BTreeMap::new(),
            history: ElevationHistory::new(),
        }
    }

    /// Register an operator for a specific elevation level.
    pub fn with_level(mut self, level: ElevationLevel, op: Box<dyn LevelOperator>) -> Self {
        self.levels.insert(level, op);
        self
    }

    /// Execute with dynamic elevation.
    ///
    /// Tries current level, checks the trigger against the budget,
    /// and elevates if resistance is detected. Partial results seed
    /// the next level operator.
    pub fn execute(&mut self) -> Result<Morsel, ElevationError> {
        loop {
            let op = self.levels.get_mut(&self.level)
                .ok_or(ElevationError::NoPlanForLevel(self.level))?;

            let start = Instant::now();
            match op.execute_level() {
                Ok(morsel) => {
                    let trigger = ElevationTrigger {
                        result_count: morsel.num_rows,
                        elapsed: start.elapsed(),
                        memory_bytes: estimate_morsel_bytes(&morsel),
                    };

                    if should_elevate(&trigger, &self.budget)
                        && self.level < self.budget.ceiling
                    {
                        let next = self.level.next();
                        self.history.record(ElevationEvent {
                            from_level: self.level,
                            to_level: next,
                            trigger,
                            timestamp: Instant::now(),
                        });
                        self.level = next;

                        // Transfer partial results to next level
                        if let Some(next_op) = self.levels.get_mut(&self.level) {
                            next_op.seed_results(&morsel);
                        }
                        continue; // retry at higher level
                    }

                    return Ok(morsel);
                }
                Err(ElevationError::Timeout { level, elapsed_ms }) => {
                    if self.level < self.budget.ceiling {
                        let next = self.level.next();
                        self.history.record(ElevationEvent {
                            from_level: self.level,
                            to_level: next,
                            trigger: ElevationTrigger {
                                result_count: 0,
                                elapsed: start.elapsed(),
                                memory_bytes: 0,
                            },
                            timestamp: Instant::now(),
                        });
                        self.level = next;
                        continue;
                    }
                    return Err(ElevationError::Timeout { level, elapsed_ms });
                }
                Err(e) => return Err(e),
            }
        }
    }

    /// How many times this operator elevated during execution.
    pub fn elevation_count(&self) -> usize {
        self.history.events().len()
    }

    /// Final level after execution.
    pub fn final_level(&self) -> ElevationLevel {
        self.level
    }
}

impl PhysicalOperator for ElevatingOp {
    fn name(&self) -> &str { "elevating" }

    fn cardinality(&self) -> f64 {
        self.levels.get(&self.level)
            .map(|op| op.cardinality())
            .unwrap_or(0.0)
    }

    fn is_pipeline_breaker(&self) -> bool {
        // Elevation can break pipelines when transferring between levels
        self.level >= ElevationLevel::Batch
    }

    fn children(&self) -> Vec<&dyn PhysicalOperator> {
        vec![] // Level operators are internal
    }
}

/// Rough estimate of morsel memory footprint.
fn estimate_morsel_bytes(morsel: &Morsel) -> usize {
    use crate::physical::ColumnData;
    morsel.columns.iter().map(|col| match col {
        ColumnData::Int64(v) => v.len() * 8,
        ColumnData::Float64(v) => v.len() * 8,
        ColumnData::String(v) => v.iter().map(|s| s.len() + 24).sum(),
        ColumnData::Fingerprint(v) => v.iter().map(|fp| fp.len() * 8).sum(),
        ColumnData::TruthValue(v) => v.len() * 16,
    }).sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Dummy level operator for testing.
    #[derive(Debug)]
    struct DummyLevel {
        rows: usize,
        seeded: bool,
    }

    impl DummyLevel {
        fn new(rows: usize) -> Self { Self { rows, seeded: false } }
    }

    impl LevelOperator for DummyLevel {
        fn execute_level(&mut self) -> Result<Morsel, ElevationError> {
            Ok(Morsel { num_rows: self.rows, columns: vec![] })
        }
        fn seed_results(&mut self, _partial: &Morsel) {
            self.seeded = true;
        }
        fn cardinality(&self) -> f64 { self.rows as f64 }
    }

    #[test]
    fn test_point_query_stays_at_level_0() {
        let budget = PatienceBudget {
            latency_budget: std::time::Duration::from_secs(1),
            result_threshold: 10_000,
            memory_budget: 1_000_000,
            ceiling: ElevationLevel::Batch,
        };
        let mut op = ElevatingOp::new(ElevationLevel::Point, budget)
            .with_level(ElevationLevel::Point, Box::new(DummyLevel::new(1)));

        let result = op.execute().unwrap();
        assert_eq!(result.num_rows, 1);
        assert_eq!(op.final_level(), ElevationLevel::Point);
        assert_eq!(op.elevation_count(), 0);
    }

    #[test]
    fn test_large_result_triggers_elevation() {
        let budget = PatienceBudget {
            latency_budget: std::time::Duration::from_secs(1),
            result_threshold: 100, // Low threshold
            memory_budget: 1_000_000_000,
            ceiling: ElevationLevel::Batch,
        };
        let mut op = ElevatingOp::new(ElevationLevel::Scan, budget)
            .with_level(ElevationLevel::Scan, Box::new(DummyLevel::new(500)))
            .with_level(ElevationLevel::Cascade, Box::new(DummyLevel::new(50)))
            .with_level(ElevationLevel::Batch, Box::new(DummyLevel::new(10)));

        let result = op.execute().unwrap();
        // Should have elevated past Scan (500 > 100 threshold)
        assert!(op.final_level() > ElevationLevel::Scan);
        assert!(op.elevation_count() >= 1);
    }

    #[test]
    fn test_ceiling_prevents_over_elevation() {
        let budget = PatienceBudget {
            latency_budget: std::time::Duration::from_nanos(1), // Impossibly low
            result_threshold: 1,
            memory_budget: 1,
            ceiling: ElevationLevel::Cascade, // Cap here
        };
        let mut op = ElevatingOp::new(ElevationLevel::Scan, budget)
            .with_level(ElevationLevel::Scan, Box::new(DummyLevel::new(100)))
            .with_level(ElevationLevel::Cascade, Box::new(DummyLevel::new(100)));

        let result = op.execute();
        // Should stop at Cascade ceiling even though trigger says elevate
        assert!(op.final_level() <= ElevationLevel::Cascade);
    }

    #[test]
    fn test_elevation_history_records_events() {
        let budget = PatienceBudget {
            latency_budget: std::time::Duration::from_secs(10),
            result_threshold: 10, // Force elevation
            memory_budget: 1_000_000_000,
            ceiling: ElevationLevel::Batch,
        };
        let mut op = ElevatingOp::new(ElevationLevel::Point, budget)
            .with_level(ElevationLevel::Point, Box::new(DummyLevel::new(100)))
            .with_level(ElevationLevel::Scan, Box::new(DummyLevel::new(100)))
            .with_level(ElevationLevel::Cascade, Box::new(DummyLevel::new(100)))
            .with_level(ElevationLevel::Batch, Box::new(DummyLevel::new(5)));

        let _ = op.execute();
        // Should have at least one elevation event
        assert!(!op.history.events().is_empty());
        // Each event should have from < to
        for event in op.history.events() {
            assert!(event.from_level < event.to_level);
        }
    }
}
