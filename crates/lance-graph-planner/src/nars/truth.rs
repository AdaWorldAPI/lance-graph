//! NARS Truth Values — the data that lives on edges.
//!
//! Every edge has: frequency (f), confidence (c), temporal_index (t).
//! f = proportion of positive evidence (0..1)
//! c = total evidence weight (0..1, where 1 = maximum certainty)
//! t = when this truth was last updated

/// NARS truth value.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TruthValue {
    /// Frequency: proportion of positive evidence (0..1).
    pub frequency: f32,
    /// Confidence: total evidence weight (0..1).
    pub confidence: f32,
}

impl Default for TruthValue {
    fn default() -> Self {
        Self {
            frequency: 0.5,
            confidence: 0.0,
        }
    }
}

impl TruthValue {
    pub fn new(frequency: f32, confidence: f32) -> Self {
        Self {
            frequency: frequency.clamp(0.0, 1.0),
            confidence: confidence.clamp(0.0, 1.0),
        }
    }

    /// Expectation: weighted average of frequency and 0.5.
    /// Higher confidence → expectation closer to frequency.
    pub fn expectation(&self) -> f32 {
        self.confidence * (self.frequency - 0.5) + 0.5
    }

    /// Surprise: how unexpected this truth value is given prior expectation.
    pub fn surprise(&self, prior: f32) -> f32 {
        (self.frequency - prior).abs() * self.confidence
    }

    /// Evidence weight (w = c / (1 - c)).
    pub fn evidence_weight(&self) -> f32 {
        if self.confidence >= 1.0 {
            f32::MAX
        } else {
            self.confidence / (1.0 - self.confidence)
        }
    }

    /// NARS Revision: merge two truth values for the same statement.
    /// f_revised = (f1*w1 + f2*w2) / (w1 + w2)
    /// c_revised = (w1 + w2) / (w1 + w2 + 1)
    pub fn revise(&self, other: &TruthValue) -> TruthValue {
        let w1 = self.evidence_weight();
        let w2 = other.evidence_weight();
        let w_sum = w1 + w2;

        if w_sum < f32::EPSILON {
            return TruthValue::default();
        }

        let f_revised = (self.frequency * w1 + other.frequency * w2) / w_sum;
        let c_revised = w_sum / (w_sum + 1.0);

        TruthValue::new(f_revised, c_revised)
    }

    /// NARS Deduction: A→B, B→C ⊢ A→C.
    /// f_conclusion = f1 * f2
    /// c_conclusion = f1 * f2 * c1 * c2
    pub fn deduction(&self, other: &TruthValue) -> TruthValue {
        let f = self.frequency * other.frequency;
        let c = f * self.confidence * other.confidence;
        TruthValue::new(f, c)
    }

    /// NARS Induction: A→B, A→C ⊢ B→C.
    /// f_conclusion = f2
    /// c_conclusion = f1 * c1 * c2 / (f1 * c1 * c2 + 1)
    pub fn induction(&self, other: &TruthValue) -> TruthValue {
        let f = other.frequency;
        let w = self.frequency * self.confidence * other.confidence;
        let c = w / (w + 1.0);
        TruthValue::new(f, c)
    }

    /// NARS Abduction: A→B, C→B ⊢ A→C.
    /// f_conclusion = f1
    /// c_conclusion = f2 * c1 * c2 / (f2 * c1 * c2 + 1)
    pub fn abduction(&self, other: &TruthValue) -> TruthValue {
        let f = self.frequency;
        let w = other.frequency * self.confidence * other.confidence;
        let c = w / (w + 1.0);
        TruthValue::new(f, c)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_revision_increases_confidence() {
        let a = TruthValue::new(0.8, 0.5);
        let b = TruthValue::new(0.8, 0.5);
        let revised = a.revise(&b);

        assert!(
            revised.confidence > a.confidence,
            "Revision with agreeing evidence should increase confidence"
        );
        assert!(
            (revised.frequency - 0.8).abs() < 0.01,
            "Revision of agreeing evidence should preserve frequency"
        );
    }

    #[test]
    fn test_deduction_attenuates() {
        let premise = TruthValue::new(0.9, 0.8);
        let edge = TruthValue::new(0.7, 0.9);
        let conclusion = premise.deduction(&edge);

        assert!(conclusion.frequency < premise.frequency);
        assert!(conclusion.confidence < premise.confidence);
    }

    #[test]
    fn test_expectation() {
        let high = TruthValue::new(0.9, 0.8);
        let low = TruthValue::new(0.9, 0.1);

        assert!(
            high.expectation() > low.expectation(),
            "Higher confidence should push expectation toward frequency"
        );
    }
}
