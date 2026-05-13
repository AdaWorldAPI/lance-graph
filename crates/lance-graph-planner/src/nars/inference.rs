//! NARS Inference Rules — thinking operations on adjacent truth values.
//!
//! Each inference type maps to a specific semiring for adjacent_truth_propagate().

use super::truth::TruthValue;
use crate::ir::logical_op::SemiringType;

/// NARS inference types — how we reason along edges.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum NarsInference {
    /// A→B, B→C ⊢ A→C. Follow the chain.
    Deduction,
    /// A→B, A→C ⊢ B→C. Generalize from shared cause.
    Induction,
    /// A→B, C→B ⊢ A→C. Infer from shared effect.
    Abduction,
    /// Merge two truth values for the same statement.
    Revision,
    /// Combine complementary evidence across domains.
    Synthesis,
}

impl NarsInference {
    /// Which semiring algebra this inference type uses.
    pub fn semiring_type(&self) -> SemiringType {
        match self {
            Self::Deduction => SemiringType::TruthPropagating,
            Self::Induction => SemiringType::TruthPropagating,
            Self::Abduction => SemiringType::TruthPropagating,
            Self::Revision => SemiringType::TruthPropagating,
            Self::Synthesis => SemiringType::XorBundle,
        }
    }

    /// Apply this inference rule to produce a conclusion truth value.
    pub fn apply(&self, premise: &TruthValue, edge: &TruthValue) -> TruthValue {
        match self {
            Self::Deduction => premise.deduction(edge),
            Self::Induction => premise.induction(edge),
            Self::Abduction => premise.abduction(edge),
            Self::Revision => premise.revise(edge),
            Self::Synthesis => {
                // Synthesis: weighted average (simplified)
                TruthValue::new(
                    (premise.frequency + edge.frequency) / 2.0,
                    (premise.confidence + edge.confidence) / 2.0,
                )
            }
        }
    }

    /// Maps from the thinking layer's NarsInferenceType to the data layer's NarsInference.
    pub fn from_thinking_type(t: crate::thinking::NarsInferenceType) -> Self {
        match t {
            crate::thinking::NarsInferenceType::Deduction => Self::Deduction,
            crate::thinking::NarsInferenceType::Induction => Self::Induction,
            crate::thinking::NarsInferenceType::Abduction => Self::Abduction,
            crate::thinking::NarsInferenceType::Revision => Self::Revision,
            crate::thinking::NarsInferenceType::Synthesis => Self::Synthesis,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_all_inference_types() {
        let premise = TruthValue::new(0.8, 0.7);
        let edge = TruthValue::new(0.9, 0.8);

        for inference in [
            NarsInference::Deduction,
            NarsInference::Induction,
            NarsInference::Abduction,
            NarsInference::Revision,
            NarsInference::Synthesis,
        ] {
            let result = inference.apply(&premise, &edge);
            assert!(result.frequency >= 0.0 && result.frequency <= 1.0);
            assert!(result.confidence >= 0.0 && result.confidence <= 1.0);
        }
    }
}
