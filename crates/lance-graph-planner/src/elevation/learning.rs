//! Elevation History → Cost Model Feedback Loop.
//!
//! Every `ElevationEvent` records what triggered elevation. Over time,
//! the planner learns which query shapes tend to elevate and can start
//! at a higher level preemptively.
//!
//! This is the RL signal: if a query pattern always elevates L1→L3,
//! the cost model should predict L3 upfront.

use super::{ElevationEvent, ElevationLevel};
use std::collections::HashMap;

/// Recorded elevation history for a query or session.
#[derive(Debug, Clone)]
pub struct ElevationHistory {
    events: Vec<ElevationEvent>,
}

impl ElevationHistory {
    pub fn new() -> Self {
        Self { events: Vec::new() }
    }

    pub fn record(&mut self, event: ElevationEvent) {
        self.events.push(event);
    }

    pub fn events(&self) -> &[ElevationEvent] {
        &self.events
    }

    /// Final level reached (highest elevation).
    pub fn final_level(&self) -> Option<ElevationLevel> {
        self.events.last().map(|e| e.to_level)
    }

    /// Total number of elevations.
    pub fn elevation_count(&self) -> usize {
        self.events.len()
    }

    /// Total time spent elevating (sum of all triggers' elapsed times).
    pub fn total_elevation_overhead(&self) -> std::time::Duration {
        self.events.iter().map(|e| e.trigger.elapsed).sum()
    }
}

impl Default for ElevationHistory {
    fn default() -> Self {
        Self::new()
    }
}

/// Accumulated learning from multiple query executions.
///
/// Tracks which elevation patterns recur and recommends starting levels
/// for future queries with similar features.
#[derive(Debug)]
pub struct ElevationLearner {
    /// Feature hash → (start_level, final_level) observations.
    observations: HashMap<u64, Vec<LevelObservation>>,
    /// Minimum observations before making a prediction.
    min_samples: usize,
}

#[derive(Debug, Clone)]
struct LevelObservation {
    start_level: ElevationLevel,
    final_level: ElevationLevel,
    elevations: usize,
}

impl ElevationLearner {
    pub fn new() -> Self {
        Self {
            observations: HashMap::new(),
            min_samples: 3,
        }
    }

    /// Record an observed execution pattern.
    pub fn observe(
        &mut self,
        feature_hash: u64,
        start_level: ElevationLevel,
        history: &ElevationHistory,
    ) {
        let final_level = history.final_level().unwrap_or(start_level);
        self.observations.entry(feature_hash).or_default().push(LevelObservation {
            start_level,
            final_level,
            elevations: history.elevation_count(),
        });
    }

    /// Predict the recommended starting level for a query with this feature hash.
    ///
    /// If we've seen this pattern enough times and it always elevates,
    /// recommend starting at the median final level.
    pub fn predict_start_level(&self, feature_hash: u64) -> Option<ElevationLevel> {
        let obs = self.observations.get(&feature_hash)?;
        if obs.len() < self.min_samples {
            return None; // Not enough data
        }

        // If most observations end at the same level, recommend starting there
        let mut level_counts: HashMap<ElevationLevel, usize> = HashMap::new();
        for o in obs {
            *level_counts.entry(o.final_level).or_default() += 1;
        }

        let (most_common_level, count) = level_counts.iter()
            .max_by_key(|(_, count)| *count)?;

        // Only recommend if >50% of observations agree
        if *count * 2 > obs.len() {
            Some(*most_common_level)
        } else {
            None
        }
    }

    /// Simple feature hash from query characteristics.
    /// In production this would be a proper locality-sensitive hash.
    pub fn hash_features(
        has_vlp: bool,
        num_match: usize,
        has_fingerprint: bool,
        has_aggregation: bool,
    ) -> u64 {
        let mut h: u64 = 0;
        if has_vlp { h |= 1; }
        h |= (num_match as u64 & 0xFF) << 8;
        if has_fingerprint { h |= 1 << 16; }
        if has_aggregation { h |= 1 << 17; }
        h
    }
}

impl Default for ElevationLearner {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::ElevationTrigger;
    use std::time::{Duration, Instant};

    fn make_history(from: ElevationLevel, to: ElevationLevel) -> ElevationHistory {
        let mut h = ElevationHistory::new();
        h.record(ElevationEvent {
            from_level: from,
            to_level: to,
            trigger: ElevationTrigger {
                result_count: 10_000,
                elapsed: Duration::from_millis(50),
                memory_bytes: 1024,
            },
            timestamp: Instant::now(),
        });
        h
    }

    #[test]
    fn test_learner_no_prediction_without_data() {
        let learner = ElevationLearner::new();
        assert_eq!(learner.predict_start_level(42), None);
    }

    #[test]
    fn test_learner_predicts_after_enough_samples() {
        let mut learner = ElevationLearner::new();
        let hash = ElevationLearner::hash_features(true, 3, false, false);

        // Observe the same pattern 5 times: always elevates to Batch
        for _ in 0..5 {
            let history = make_history(ElevationLevel::Scan, ElevationLevel::Batch);
            learner.observe(hash, ElevationLevel::Scan, &history);
        }

        let prediction = learner.predict_start_level(hash);
        assert_eq!(prediction, Some(ElevationLevel::Batch));
    }

    #[test]
    fn test_learner_no_prediction_with_mixed_data() {
        let mut learner = ElevationLearner::new();
        let hash = 99;

        // Mixed: some go to Cascade, some to Batch, some to IvfBatch
        learner.observe(hash, ElevationLevel::Scan, &make_history(ElevationLevel::Scan, ElevationLevel::Cascade));
        learner.observe(hash, ElevationLevel::Scan, &make_history(ElevationLevel::Scan, ElevationLevel::Batch));
        learner.observe(hash, ElevationLevel::Scan, &make_history(ElevationLevel::Scan, ElevationLevel::IvfBatch));

        // No majority — no prediction
        assert_eq!(learner.predict_start_level(hash), None);
    }

    #[test]
    fn test_feature_hash_deterministic() {
        let h1 = ElevationLearner::hash_features(true, 3, false, true);
        let h2 = ElevationLearner::hash_features(true, 3, false, true);
        assert_eq!(h1, h2);

        let h3 = ElevationLearner::hash_features(false, 3, false, true);
        assert_ne!(h1, h3);
    }
}
