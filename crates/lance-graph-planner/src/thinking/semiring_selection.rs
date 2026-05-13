//! Semiring Auto-Selection — choose semiring from query shape + thinking style.
//!
//! Nobody else does this: the semiring is inferred from the query rather
//! than being hardcoded per backend.

use super::nars_dispatch::NarsInferenceType;
use super::style::ThinkingStyle;
use crate::ir::logical_op::SemiringType;

/// Semiring choice with rationale.
#[derive(Debug, Clone)]
pub struct SemiringChoice {
    pub semiring: SemiringType,
    pub rationale: &'static str,
}

/// Select the optimal semiring based on query shape, thinking style, and NARS type.
pub fn select(query: &str, style: &ThinkingStyle, nars_type: &NarsInferenceType) -> SemiringChoice {
    let q = query.to_uppercase();

    // 1. RESONATE queries → XorBundle (superposition algebra)
    if q.contains("RESONATE") || q.contains("HAMMING") || q.contains("SIMILARITY") {
        return SemiringChoice {
            semiring: SemiringType::XorBundle,
            rationale: "Resonance query: XOR superposition for similarity search",
        };
    }

    // 2. Truth-value propagation queries (NARS revision/synthesis)
    if matches!(
        nars_type,
        NarsInferenceType::Revision | NarsInferenceType::Synthesis
    ) {
        return SemiringChoice {
            semiring: SemiringType::TruthPropagating,
            rationale: "NARS revision/synthesis: truth values accumulated during traversal",
        };
    }

    // 3. Shortest path / distance queries → Tropical
    if q.contains("SHORTESTPATH") || q.contains("DISTANCE") || q.contains("COST") {
        return SemiringChoice {
            semiring: SemiringType::Tropical,
            rationale: "Path/distance query: tropical (min, +) semiring",
        };
    }

    // 4. Existence / reachability queries → Boolean
    if q.contains("EXISTS") || q.contains("ANY(") || q.contains("ALL(") {
        return SemiringChoice {
            semiring: SemiringType::Boolean,
            rationale: "Existence query: boolean AND/OR semiring",
        };
    }

    // 5. Hamming-based scan queries → HammingMin
    if q.contains("WHERE") && (q.contains("FINGERPRINT") || q.contains("CONTAINER")) {
        return SemiringChoice {
            semiring: SemiringType::HammingMin,
            rationale: "Fingerprint filtering: Hamming distance minimum semiring",
        };
    }

    // 6. Creative/exploratory thinking styles → XorBundle (supports superposition)
    if matches!(
        style,
        ThinkingStyle::Creative | ThinkingStyle::Exploratory | ThinkingStyle::Divergent
    ) {
        return SemiringChoice {
            semiring: SemiringType::XorBundle,
            rationale: "Divergent thinking: XOR bundle for exploration",
        };
    }

    // 7. Default: Boolean (cheapest, correct for standard queries)
    SemiringChoice {
        semiring: SemiringType::Boolean,
        rationale: "Default: boolean semiring for standard graph queries",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_resonate_selects_xor() {
        let choice = select(
            "MATCH (n) WHERE RESONATE(n.fp, $query, 0.7) RETURN n",
            &ThinkingStyle::Analytical,
            &NarsInferenceType::Deduction,
        );
        assert_eq!(choice.semiring, SemiringType::XorBundle);
    }

    #[test]
    fn test_shortest_path_selects_tropical() {
        let choice = select(
            "MATCH p = shortestPath((a)-[*]->(b)) RETURN p",
            &ThinkingStyle::Analytical,
            &NarsInferenceType::Abduction,
        );
        assert_eq!(choice.semiring, SemiringType::Tropical);
    }

    #[test]
    fn test_revision_selects_truth() {
        let choice = select(
            "MATCH (n) SET n.belief = 0.7",
            &ThinkingStyle::Deliberate,
            &NarsInferenceType::Revision,
        );
        assert_eq!(choice.semiring, SemiringType::TruthPropagating);
    }
}
