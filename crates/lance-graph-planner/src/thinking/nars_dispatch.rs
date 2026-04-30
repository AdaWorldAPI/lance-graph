//! NARS Inference Type → Query Strategy Routing.
//!
//! From n8n-rs thinking_mode.rs: routes by NARS inference type
//! to different query execution strategies.

/// NARS inference types (5 canonical operations).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum NarsInferenceType {
    /// Direct lookup, high confidence. P(A→B) ∧ P(B→C) ⊢ P(A→C).
    Deduction,
    /// Pattern generalization. P(A→B) ∧ P(A→C) ⊢ P(B→C).
    Induction,
    /// Abductive leap: best explanation. P(A→B) ∧ P(B) ⊢ P(A).
    Abduction,
    /// Update existing belief with new evidence. B_old ⊕ B_new → B_revised.
    Revision,
    /// Multi-path cross-domain integration.
    Synthesis,
}

/// Query execution strategy (derived from NARS type).
#[derive(Debug, Clone)]
pub enum QueryStrategy {
    /// Direct CAM lookup — exact match, narrow beam.
    CamExact { top_k: usize, beam: usize },
    /// Wide CAM scan — pattern matching, broad window.
    CamWide { top_k: usize, window: usize },
    /// DN-tree full traversal — abductive search through graph.
    DnTreeFull { beam: usize, no_early_exit: bool },
    /// Bundle into existing — update/revise with learning rate.
    BundleInto {
        learning_rate: f64,
        btsp_gate_prob: f64,
    },
    /// Bundle across domains — multi-winner synthesis.
    BundleAcross { winner_k: usize },
}

/// Route NARS inference type to query strategy.
pub fn route(nars_type: NarsInferenceType) -> QueryStrategy {
    match nars_type {
        NarsInferenceType::Deduction => QueryStrategy::CamExact { top_k: 8, beam: 1 },
        NarsInferenceType::Induction => QueryStrategy::CamWide {
            top_k: 32,
            window: 64,
        },
        NarsInferenceType::Abduction => QueryStrategy::DnTreeFull {
            beam: 4,
            no_early_exit: true,
        },
        NarsInferenceType::Revision => QueryStrategy::BundleInto {
            learning_rate: 0.1,
            btsp_gate_prob: 0.05,
        },
        NarsInferenceType::Synthesis => QueryStrategy::BundleAcross { winner_k: 3 },
    }
}

/// Detect NARS inference type from Cypher query structure.
pub fn detect_from_query(query: &str) -> NarsInferenceType {
    let q = query.to_uppercase();

    // Revision: mutations with SET/MERGE
    if q.contains("SET ") || q.contains("MERGE ") {
        return NarsInferenceType::Revision;
    }

    // Deduction: exact MATCH with specific property lookups
    if q.contains("WHERE")
        && q.contains("=")
        && !q.contains("LIKE")
        && !q.contains("CONTAINS")
        && !q.contains("*")
    {
        return NarsInferenceType::Deduction;
    }

    // Abduction: OPTIONAL MATCH, variable-length paths, shortest path
    if q.contains("OPTIONAL") || q.contains("*..") || q.contains("SHORTESTPATH") {
        return NarsInferenceType::Abduction;
    }

    // Synthesis: UNION, multiple MATCH clauses, CALL subqueries
    if q.contains("UNION") || q.contains("CALL {") || q.matches("MATCH").count() > 2 {
        return NarsInferenceType::Synthesis;
    }

    // Induction: pattern matching, CONTAINS, LIKE, RESONATE
    if q.contains("CONTAINS") || q.contains("LIKE") || q.contains("RESONATE") || q.contains("=~") {
        return NarsInferenceType::Induction;
    }

    // Default: deduction for simple queries
    NarsInferenceType::Deduction
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_detect_deduction() {
        let q = "MATCH (n:Person) WHERE n.name = 'Ada' RETURN n";
        assert_eq!(detect_from_query(q), NarsInferenceType::Deduction);
    }

    #[test]
    fn test_detect_revision() {
        let q = "MATCH (n:Person) SET n.age = 30";
        assert_eq!(detect_from_query(q), NarsInferenceType::Revision);
    }

    #[test]
    fn test_detect_abduction() {
        let q = "MATCH p = shortestPath((a)-[*..5]->(b)) RETURN p";
        assert_eq!(detect_from_query(q), NarsInferenceType::Abduction);
    }

    #[test]
    fn test_detect_induction() {
        let q = "MATCH (n) WHERE n.name CONTAINS 'ada' RETURN n";
        assert_eq!(detect_from_query(q), NarsInferenceType::Induction);
    }

    #[test]
    fn test_detect_synthesis() {
        let q = "MATCH (a) RETURN a UNION MATCH (b) RETURN b";
        assert_eq!(detect_from_query(q), NarsInferenceType::Synthesis);
    }
}
