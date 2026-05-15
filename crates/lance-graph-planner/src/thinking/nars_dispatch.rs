//! NARS Inference Type → Query Strategy Routing.
//!
//! From n8n-rs thinking_mode.rs: routes by NARS inference type
//! to different query execution strategies.

/// NARS inference types (7 canonical operations, including Pearl 2³ rungs 2 and 3).
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
    /// Pearl rung 2 — do-calculus / interventional reasoning.
    ///
    /// Implements `do(X = x)`: surgically sever the causal mechanism that sets X
    /// and force X to value x, while holding all other Independent Causal
    /// Mechanisms (ICM) invariant. Per Schölkopf et al. (Causal de Finetti,
    /// arXiv 2203.15756), mechanisms that are invariant across environments
    /// (SPO-G grouping) must remain untouched; only the targeted mechanism is
    /// replaced. Produces an interventional distribution `P(Y | do(X = x))`
    /// distinct from the observational `P(Y | X = x)`.
    ///
    /// Confidence modifier: 0.85 — TUNED-LATER (starting calibration).
    Intervention,
    /// Pearl rung 3 — counterfactual reasoning via the 3-step abduce→intervene→predict chain.
    ///
    /// Implements the full counterfactual query `P(Y_x = y | X = x', Y = y')`:
    ///   1. **Abduce** — infer latent background context U from observed evidence
    ///      (uses `NarsInferenceType::Abduction` on the prior SPO-G state).
    ///   2. **Intervene** — apply `do(X = x)` on the abduced world while
    ///      respecting ICM invariance (uses `NarsInferenceType::Intervention`).
    ///   3. **Predict** — forward-propagate through remaining mechanisms to
    ///      obtain the counterfactual outcome (uses `NarsInferenceType::Deduction`).
    ///
    /// Per Vashishtha et al. (Executable Counterfactuals, arXiv 2510.01539),
    /// LLMs drop 25–40 % from interventional to counterfactual reasoning;
    /// explicit 3-step dispatch (rather than end-to-end generation) closes
    /// most of this gap. RL (GRPO) over this 3-step chain generalises OOD;
    /// SFT does not.
    ///
    /// Confidence modifier: 0.7 — TUNED-LATER (starting calibration; lower
    /// than Intervention to reflect the compounded uncertainty of the 3-step
    /// chain plus abduced latent context).
    Counterfactual,
}

impl NarsInferenceType {
    /// Confidence modifier for each inference type.
    ///
    /// Scales the base NARS truth value to reflect the epistemic cost of each
    /// rung. Values for `Intervention` (0.85) and `Counterfactual` (0.7) are
    /// **TUNED-LATER** starting calibrations; adjust once GRPO training data
    /// from PR-LL-4 provides empirical ground truth.
    pub fn confidence_modifier(self) -> f64 {
        match self {
            NarsInferenceType::Deduction => 0.99,
            NarsInferenceType::Induction => 0.90,
            NarsInferenceType::Abduction => 0.80,
            NarsInferenceType::Revision => 0.95,
            NarsInferenceType::Synthesis => 0.85,
            // Pearl rung 2: interventional do-calculus — high confidence but
            // lower than Deduction because mechanism surgery introduces
            // structural uncertainty.  TUNED-LATER.
            NarsInferenceType::Intervention => 0.85,
            // Pearl rung 3: 3-step abduce→intervene→predict — compounded
            // uncertainty from latent-context abduction plus mechanism
            // surgery plus forward prediction.  TUNED-LATER.
            NarsInferenceType::Counterfactual => 0.70,
        }
    }
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
        // Pearl rung 2: targeted mechanism surgery — use DN-tree traversal with
        // early-exit disabled so the planner reaches the targeted mechanism node
        // even in deep graphs; beam=8 (wider than Abduction) to cover sibling
        // mechanisms that must remain invariant.
        NarsInferenceType::Intervention => QueryStrategy::DnTreeFull {
            beam: 8,
            no_early_exit: true,
        },
        // Pearl rung 3: 3-step chain — dispatch as wide CAM scan to surface the
        // background-context evidence needed for the abduction step; the
        // intervene and predict sub-steps are handled by the caller composing
        // Intervention + Deduction queries.
        NarsInferenceType::Counterfactual => QueryStrategy::CamWide {
            top_k: 64,
            window: 128,
        },
    }
}

/// Detect NARS inference type from Cypher query structure.
pub fn detect_from_query(query: &str) -> NarsInferenceType {
    let q = query.to_uppercase();

    // Counterfactual (rung 3): explicit COUNTERFACTUAL / WHAT_IF keyword or 3-step marker.
    // Detected before Intervention so that a combined query is routed to the
    // outer 3-step dispatch rather than the inner do-calculus step.
    if q.contains("COUNTERFACTUAL") || q.contains("WHAT_IF") || q.contains("HAD_BEEN") {
        return NarsInferenceType::Counterfactual;
    }

    // Intervention (rung 2): do-calculus markers — DO( operator or INTERVENE keyword.
    if q.contains("DO(") || q.contains("INTERVENE") || q.contains("SET_MECHANISM") {
        return NarsInferenceType::Intervention;
    }

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
