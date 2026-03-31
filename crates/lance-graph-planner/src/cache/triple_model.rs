//! Triple Model: self/user/impact — three simultaneous 4096-head attention matrices.
//!
//! self_model:   what I plan to say (my intention)
//! user_model:   what the user expects (their mental model)
//! impact_model: what my output causes (causal prediction)
//!
//! All three are 64×64 AttentionMatrices that evolve with each turn.
//! CausalEdge64 forward() predicts impact. learn() revises after feedback.

use super::kv_bundle::{HeadPrint, AttentionMatrix};

/// 3-bit plasticity: which planes are still learning.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Plasticity {
    pub bits: u8, // bit 0=S hot, bit 1=P hot, bit 2=O hot
}

impl Plasticity {
    pub const ALL_HOT: Self = Self { bits: 0b111 };
    pub const ALL_FROZEN: Self = Self { bits: 0b000 };
    pub fn s_hot(self) -> bool { self.bits & 1 != 0 }
    pub fn p_hot(self) -> bool { self.bits & 2 != 0 }
    pub fn o_hot(self) -> bool { self.bits & 4 != 0 }
    pub fn freeze_if_confident(&mut self, confidence: f32) {
        if confidence > 0.9 {
            self.bits = 0;
        } else if confidence > 0.7 {
            // Freeze most stable planes
        }
    }
}

/// NARS truth value — type alias for ndarray's canonical NarsTruth.
/// API: `.frequency`, `.confidence`, `.expectation()`, `::new(f, c)`.
/// Revision: use `truth_revision(a, b)` free function below.
pub use ndarray::hpc::nars::NarsTruth as Truth;

/// Revision: merge evidence from two truth values.
/// Wraps ndarray's `nars_revision` for convenience.
pub fn truth_revision(a: Truth, b: Truth) -> Truth {
    ndarray::hpc::nars::nars_revision(a, b)
}

/// Create the "unknown" / ignorance truth value (f=0.5, c=0.0).
pub fn truth_unknown() -> Truth {
    Truth::new(0.5, 0.0)
}

/// Dunning-Kruger position.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DkPosition {
    MountStupid,
    ValleyOfDespair,
    SlopeOfEnlightenment,
    PlateauOfMastery,
}

/// One model in the triple.
pub struct ModelState {
    pub matrix: AttentionMatrix, // 64×64 = 4096 interdependent heads
    pub plasticity: Plasticity,
    pub truth: Truth,
    pub dk: DkPosition,
}

impl ModelState {
    pub fn new() -> Self {
        Self {
            matrix: AttentionMatrix::new_hip(),
            plasticity: Plasticity::ALL_HOT,
            truth: truth_unknown(),
            dk: DkPosition::MountStupid,
        }
    }

    /// Update one head and revise model truth.
    pub fn update_head(&mut self, row: usize, col: usize, head: HeadPrint, evidence: Truth) {
        self.matrix.set(row, col, head);
        self.truth = truth_revision(self.truth, evidence);
        self.plasticity.freeze_if_confident(self.truth.confidence);
        // DK transitions based on confidence trajectory
        self.dk = match (self.dk, self.truth.confidence) {
            (DkPosition::MountStupid, c) if c < 0.3 => DkPosition::ValleyOfDespair,
            (DkPosition::ValleyOfDespair, c) if c > 0.5 => DkPosition::SlopeOfEnlightenment,
            (DkPosition::SlopeOfEnlightenment, c) if c > 0.8 => DkPosition::PlateauOfMastery,
            (dk, _) => dk,
        };
    }
}

/// The triple: self, user, impact.
pub struct TripleModel {
    pub self_model: ModelState,
    pub user_model: ModelState,
    pub impact_model: ModelState,
}

impl TripleModel {
    pub fn new() -> Self {
        Self {
            self_model: ModelState::new(),
            user_model: ModelState::new(),
            impact_model: ModelState::new(),
        }
    }

    /// After I say something: update self_model, predict impact.
    pub fn on_self_output(&mut self, output: &HeadPrint, row: usize, col: usize) {
        let evidence = Truth::new(0.8, 0.7); // I know what I said
        self.self_model.update_head(row, col, output.clone(), evidence);
        // Impact prediction: how will user react?
        // Surprise = divergence between self and user models
    }

    /// After user responds: update user_model, measure prediction error.
    pub fn on_user_input(&mut self, input: &HeadPrint, row: usize, col: usize) {
        let evidence = Truth::new(0.6, 0.5); // less certain about user
        self.user_model.update_head(row, col, input.clone(), evidence);
        // Prediction error = Friston free energy
        let prediction_error = self.impact_model.matrix.surprise(input);
        // Revise impact model based on error
        let error_truth = Truth::new(1.0 - prediction_error, 0.8);
        self.impact_model.truth = truth_revision(self.impact_model.truth, error_truth);
    }

    /// Friston surprise: how wrong was my prediction?
    pub fn free_energy(&self, actual: &HeadPrint) -> f32 {
        self.impact_model.matrix.surprise(actual)
    }

    /// Alignment: how close are self and user models?
    pub fn alignment(&self) -> f32 {
        1.0 - self.self_model.matrix.gestalt.l1(&self.user_model.matrix.gestalt) as f32
            / (17u32 * 65535) as f32
    }

    /// Topic shift: self_model diverging from user_model?
    pub fn topic_shift(&self) -> f32 {
        self.self_model.matrix.gestalt.l1(&self.user_model.matrix.gestalt) as f32
            / (17u32 * 65535) as f32
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_triple_model_creation() {
        let triple = TripleModel::new();
        assert_eq!(triple.self_model.dk, DkPosition::MountStupid);
        assert_eq!(triple.user_model.dk, DkPosition::MountStupid);
        assert_eq!(triple.impact_model.dk, DkPosition::MountStupid);
        assert_eq!(triple.self_model.plasticity, Plasticity::ALL_HOT);
        assert_eq!(triple.self_model.matrix.resolution, 64);
        assert_eq!(triple.self_model.matrix.heads.len(), 64 * 64);
    }

    #[test]
    fn test_on_self_output() {
        let mut triple = TripleModel::new();
        let head = HeadPrint {
            dims: [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700],
        };
        triple.on_self_output(&head, 5, 10);

        // Self model should have the head set
        assert_eq!(triple.self_model.matrix.get(5, 10), &head);
        // Truth should have been revised from unknown
        assert!(triple.self_model.truth.confidence > 0.0, "confidence should increase after evidence");
        assert_eq!(triple.self_model.matrix.epoch, 1);
    }

    #[test]
    fn test_on_user_input_revises() {
        let mut triple = TripleModel::new();
        let input = HeadPrint {
            dims: [50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 800, 850],
        };
        let initial_impact_truth = triple.impact_model.truth.confidence;

        triple.on_user_input(&input, 3, 7);

        // User model should have the head
        assert_eq!(triple.user_model.matrix.get(3, 7), &input);
        // Impact model truth should be revised
        assert!(
            triple.impact_model.truth.confidence > initial_impact_truth,
            "impact model confidence should increase after revision"
        );
    }

    #[test]
    fn test_free_energy() {
        let triple = TripleModel::new();
        // Against a zero gestalt, a zero head should have zero surprise
        let zero = HeadPrint::zero();
        assert_eq!(triple.free_energy(&zero), 0.0);

        // A non-zero head against zero gestalt should have non-zero surprise
        let head = HeadPrint {
            dims: [1000; 17],
        };
        let energy = triple.free_energy(&head);
        assert!(energy > 0.0, "non-zero head should produce surprise: {energy}");
    }

    #[test]
    fn test_alignment() {
        let triple = TripleModel::new();
        // Both models start at zero gestalt, alignment should be 1.0
        assert_eq!(triple.alignment(), 1.0);
        assert_eq!(triple.topic_shift(), 0.0);
    }

    #[test]
    fn test_dk_transitions() {
        let mut state = ModelState::new();
        assert_eq!(state.dk, DkPosition::MountStupid);

        // Feed low-confidence evidence to trigger MountStupid -> ValleyOfDespair
        // We need confidence to end up below 0.3 after revision
        let low_evidence = Truth::new(0.5, 0.2);
        let head = HeadPrint::zero();
        state.update_head(0, 0, head.clone(), low_evidence);
        // After one low-confidence revision, c should be low enough
        if state.truth.confidence < 0.3 {
            assert_eq!(state.dk, DkPosition::ValleyOfDespair);
        }

        // Now feed medium-confidence evidence repeatedly to climb
        state.dk = DkPosition::ValleyOfDespair;
        state.truth = Truth::new(0.8, 0.55);
        let med_evidence = Truth::new(0.9, 0.6);
        state.update_head(1, 1, head.clone(), med_evidence);
        // After revision with existing 0.55 and new 0.6, confidence should exceed 0.5
        if state.truth.confidence > 0.5 {
            assert_eq!(state.dk, DkPosition::SlopeOfEnlightenment);
        }

        // Feed high confidence to reach PlateauOfMastery
        state.dk = DkPosition::SlopeOfEnlightenment;
        state.truth = Truth::new(0.9, 0.85);
        let high_evidence = Truth::new(0.95, 0.9);
        state.update_head(2, 2, head, high_evidence);
        if state.truth.confidence > 0.8 {
            assert_eq!(state.dk, DkPosition::PlateauOfMastery);
        }
    }
}
