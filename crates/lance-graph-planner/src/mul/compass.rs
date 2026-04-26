//! Compass Function — Navigation When the Map Runs Out.
//!
//! Five compass needles:
//! - KANT: Universalizability ("If everyone did this, would it still work?")
//! - ANALOGY: Structural transfer, not content matching
//! - IDENTITY: "Am I still recognizably continuous with who I was?"
//! - REVERSIBILITY: Prefer reversible actions in unknown territory
//! - CURIOSITY: Bias toward actions that teach
//!
//! compass_score = universalizable × identity_preserved
//!               × (0.5 + 0.5 × reversible)
//!               × (1 + epistemic_value)
//!               × MUL_free_will_modifier

use super::MulAssessment;

/// Result of compass navigation.
#[derive(Debug, Clone)]
pub struct CompassResult {
    pub needles: CompassNeedles,
    pub score: f64,
    pub modified_score: f64,
    pub decision: CompassDecision,
}

/// The five compass needles, each scored 0..1.
#[derive(Debug, Clone)]
pub struct CompassNeedles {
    /// Universalizability test: "If everyone did this, would it still work?"
    pub kant: f64,
    /// Structural similarity to known patterns (analogy, not content).
    pub analogy: f64,
    /// Identity preservation: "Am I still recognizably continuous?"
    pub identity: f64,
    /// Reversibility: "Can I undo this?" (0 = irreversible, 1 = fully reversible).
    pub reversibility: f64,
    /// Epistemic value: "How much does this teach?" (curiosity gradient).
    pub curiosity: f64,
}

/// Compass navigation decision.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CompassDecision {
    /// Score > threshold: execute with learning flag.
    ExecuteWithLearning,
    /// Reversible option exists: take exploratory action.
    Exploratory,
    /// Can't decide: surface to meta ("I don't know, stakes high").
    SurfaceToMeta,
}

/// Navigate unknown territory using compass needles.
pub fn navigate(query: &str, assessment: &MulAssessment) -> CompassResult {
    // In a full implementation, these would be computed from the query
    // and the system's self-model. For now, derive from MUL assessment.
    let needles = compute_needles(query, assessment);

    // compass_score = universalizable × identity_preserved
    //               × (0.5 + 0.5 × reversible)
    //               × (1 + epistemic_value)
    //               × MUL_free_will_modifier
    let score = needles.kant
        * needles.identity
        * (0.5 + 0.5 * needles.reversibility)
        * (1.0 + needles.curiosity)
        * assessment.free_will_modifier;

    let modified_score = score;

    // Decision thresholds
    let decision = if modified_score > 0.6 {
        CompassDecision::ExecuteWithLearning
    } else if needles.reversibility > 0.7 {
        CompassDecision::Exploratory
    } else {
        CompassDecision::SurfaceToMeta
    };

    CompassResult {
        needles,
        score,
        modified_score,
        decision,
    }
}

fn compute_needles(query: &str, assessment: &MulAssessment) -> CompassNeedles {
    // Kant: universalizability scales with trust and calibration
    let kant = assessment.trust.calibration * 0.7 + assessment.trust.source * 0.3;

    // Analogy: scales with demonstrated competence (have we seen similar?)
    let analogy = assessment.trust.competence;

    // Identity: always high for query planning (low-stakes identity risk)
    let identity = 0.9;

    // Reversibility: queries are read-only by default (highly reversible)
    // unless the query contains mutations
    let is_mutation = query.contains("CREATE") || query.contains("SET")
        || query.contains("DELETE") || query.contains("MERGE");
    let reversibility = if is_mutation { 0.3 } else { 0.95 };

    // Curiosity: higher for complex/novel queries
    let curiosity = (1.0 - assessment.trust.competence) * 0.5; // Less known = more curious

    CompassNeedles {
        kant,
        analogy,
        identity,
        reversibility,
        curiosity,
    }
}

/// Post-action learning loop.
/// After execution, validate analogy accuracy, update domain library,
/// update self-model, convert compass guidance → ACT-R knowledge.
pub struct LearningLoop {
    /// History of (compass_score, actual_outcome_quality) pairs.
    observations: Vec<(f64, f64)>,
}

impl Default for LearningLoop {
    fn default() -> Self {
        Self::new()
    }
}

impl LearningLoop {
    pub fn new() -> Self {
        Self { observations: Vec::new() }
    }

    /// Record an observation: predicted compass score vs actual outcome.
    pub fn observe(&mut self, compass_score: f64, outcome_quality: f64) {
        self.observations.push((compass_score, outcome_quality));
        if self.observations.len() > 1000 {
            self.observations.remove(0);
        }
    }

    /// Brier score: mean squared error between predictions and outcomes.
    /// Lower = better calibration.
    pub fn brier_score(&self) -> f64 {
        if self.observations.is_empty() {
            return 0.5; // Default: uncalibrated
        }
        let sum: f64 = self.observations.iter()
            .map(|(predicted, actual)| (predicted - actual).powi(2))
            .sum();
        sum / self.observations.len() as f64
    }

    /// Is the compass well-calibrated? (Brier < 0.15)
    pub fn is_calibrated(&self) -> bool {
        self.observations.len() >= 10 && self.brier_score() < 0.15
    }
}
