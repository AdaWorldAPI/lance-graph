//! **LAB-ONLY consumer.** `OrchestrationBridge` impl for Cypher queries —
//! owns `StepDomain::LanceGraph` (`lg.cypher` step_type).
//!
//! Minimal viable Cypher path: recognizes the shape of the query via a
//! lightweight classifier and dispatches against the AriGraph SPO triple
//! store / BindSpace. This is the Phase 1 stub — the regex/prefix
//! classifier. Phase 2 (real `lance_graph::parser::parse_cypher_query` wiring)
//! is deferred because pulling the full lance-graph core dep (arrow +
//! datafusion + lance) into `cognitive-shader-driver` would balloon build
//! time for what is today a test-path transport.
//!
//! Contract with the route_handler:
//! - step_type must be `lg.cypher` (or any `lg.cypher.*` refinement).
//! - reasoning field carries the Cypher query string.
//! - Unknown domains bubble back as `DomainUnavailable` so the handler can
//!   fall through to the planner bridge.
//!
//! Recognized shapes:
//! - `CREATE (n:Label {k:v})` — reports cypher CREATE parsed (SPO commit
//!   pending real wiring).
//! - `MATCH (n:Label) RETURN n` — reports cypher MATCH parsed (BindSpace
//!   label search pending real wiring).
//! - Anything else — `StepStatus::Skipped` with "unsupported cypher
//!   construct" reasoning. No failure: the downstream can plan around it.

use lance_graph_contract::crystal::fingerprint::CrystalFingerprint;
use lance_graph_contract::grammar::context_chain::{ContextChain, DisambiguationResult};
use lance_graph_contract::nars::InferenceType;
use lance_graph_contract::orchestration::{
    OrchestrationBridge, OrchestrationError, StepDomain, StepStatus, UnifiedStep,
};
use lance_graph_contract::plan::ThinkingContext;
use lance_graph_contract::thinking::ThinkingStyle;

/// TD-INT-6 — disambiguation hook for multi-candidate Cypher parses.
///
/// When a real parser returns N candidate parse trees for an ambiguous
/// query, this helper consults the live `ContextChain` to pick the
/// candidate whose insertion-coherence at position `i` is highest.
/// Today's regex stub returns a single candidate, so this is a dormant
/// call site — wire in place; activation lives at parser commit time.
pub fn disambiguate_parse_candidates(
    chain: &ContextChain,
    position: usize,
    candidates: Vec<CrystalFingerprint>,
) -> Result<CrystalFingerprint, DisambiguationResult> {
    let result = chain.disambiguate(position, candidates);
    if result.escalate_to_llm {
        Err(result)
    } else {
        Ok(result.chosen.clone())
    }
}

/// Bridge for `lg.cypher` step_types. Stateless in Phase 1; an SPO store
/// handle slots in here when Phase 2 wires the real parser + BindSpace.
pub struct CypherBridge;

impl OrchestrationBridge for CypherBridge {
    fn route(&self, step: &mut UnifiedStep) -> Result<(), OrchestrationError> {
        // Only claim `lg.cypher` (and `lg.cypher.*` refinements). Other
        // `lg.*` step types (e.g. `lg.plan`, `lg.resonate`) still fall
        // through to the planner bridge.
        if !step.step_type.starts_with("lg.cypher") {
            // Signal domain mismatch so the route_handler falls through.
            let domain = StepDomain::from_step_type(&step.step_type)
                .unwrap_or(StepDomain::LanceGraph);
            return Err(OrchestrationError::DomainUnavailable(domain));
        }

        step.status = StepStatus::Running;

        let query = step
            .reasoning
            .as_deref()
            .ok_or_else(|| {
                OrchestrationError::RoutingFailed(
                    "missing cypher query in reasoning field".to_string(),
                )
            })?
            .trim()
            .to_string();

        if query.is_empty() {
            step.status = StepStatus::Failed;
            return Err(OrchestrationError::RoutingFailed(
                "empty cypher query".to_string(),
            ));
        }

        // Classify the query shape. Case-insensitive match on the leading
        // keyword; anything else is Skipped (stub-in-place, not Failed).
        let upper = query.to_uppercase();
        if upper.starts_with("CREATE") {
            step.status = StepStatus::Completed;
            step.reasoning = Some(
                "cypher CREATE parsed (stub — actual SPO commit pending)".to_string(),
            );
            step.confidence = Some(0.5);
            Ok(())
        } else if upper.starts_with("MATCH") {
            step.status = StepStatus::Completed;
            step.reasoning = Some(
                "cypher MATCH parsed (stub — actual BindSpace search pending)".to_string(),
            );
            step.confidence = Some(0.5);
            Ok(())
        } else {
            step.status = StepStatus::Skipped;
            let preview_len = 50.min(query.len());
            step.reasoning = Some(format!(
                "unsupported cypher construct, stub in place: {}",
                &query[..preview_len]
            ));
            Ok(())
        }
    }

    fn resolve_thinking(
        &self,
        _style: ThinkingStyle,
        _inference_type: InferenceType,
    ) -> ThinkingContext {
        ThinkingContext {
            style: ThinkingStyle::Systematic,
            modulation: Default::default(),
            inference_type: InferenceType::Deduction,
            strategy: lance_graph_contract::nars::QueryStrategy::CamExact,
            semiring: lance_graph_contract::nars::SemiringChoice::HammingMin,
            free_will_modifier: 1.0,
            exploratory: false,
        }
    }

    fn domain_available(&self, domain: StepDomain) -> bool {
        matches!(domain, StepDomain::LanceGraph)
    }
}

// ──────────────────────────────────────────────────────────────────────
// Tests
// ──────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_step(step_type: &str, reasoning: Option<&str>) -> UnifiedStep {
        UnifiedStep {
            step_id: "t-1".to_string(),
            step_type: step_type.to_string(),
            status: StepStatus::Pending,
            thinking: None,
            reasoning: reasoning.map(|s| s.to_string()),
            confidence: None,
            depends_on: vec![],
        }
    }

    #[test]
    fn create_cypher_parses() {
        let bridge = CypherBridge;
        let mut step = make_step("lg.cypher", Some("CREATE (c:Customer {id:1})"));
        let result = bridge.route(&mut step);
        assert!(result.is_ok(), "CREATE should be accepted, got {:?}", result);
        assert_eq!(step.status, StepStatus::Completed);
        assert_eq!(step.confidence, Some(0.5));
        assert!(step
            .reasoning
            .as_deref()
            .unwrap_or("")
            .contains("CREATE parsed"));
    }

    #[test]
    fn match_cypher_parses() {
        let bridge = CypherBridge;
        let mut step = make_step("lg.cypher", Some("MATCH (c:Customer) RETURN c"));
        let result = bridge.route(&mut step);
        assert!(result.is_ok(), "MATCH should be accepted, got {:?}", result);
        assert_eq!(step.status, StepStatus::Completed);
        assert_eq!(step.confidence, Some(0.5));
        assert!(step
            .reasoning
            .as_deref()
            .unwrap_or("")
            .contains("MATCH parsed"));
    }

    #[test]
    fn unsupported_cypher_skipped() {
        let bridge = CypherBridge;
        let mut step = make_step("lg.cypher", Some("DROP INDEX"));
        let result = bridge.route(&mut step);
        assert!(result.is_ok());
        assert_eq!(step.status, StepStatus::Skipped);
        assert!(step
            .reasoning
            .as_deref()
            .unwrap_or("")
            .contains("unsupported cypher construct"));
    }

    #[test]
    fn non_cypher_rejected() {
        let bridge = CypherBridge;
        let mut step = make_step("lg.plan", Some("anything"));
        let result = bridge.route(&mut step);
        match result {
            Err(OrchestrationError::DomainUnavailable(_)) => {}
            other => panic!("expected DomainUnavailable, got {:?}", other),
        }
    }

    #[test]
    fn missing_reasoning_errors() {
        let bridge = CypherBridge;
        let mut step = make_step("lg.cypher", None);
        let result = bridge.route(&mut step);
        assert!(matches!(result, Err(OrchestrationError::RoutingFailed(_))));
    }

    #[test]
    fn lowercase_cypher_parses() {
        // Case-insensitive keyword match — Cypher is not case-sensitive
        // on keywords.
        let bridge = CypherBridge;
        let mut step = make_step("lg.cypher", Some("match (n) return n"));
        let result = bridge.route(&mut step);
        assert!(result.is_ok());
        assert_eq!(step.status, StepStatus::Completed);
    }

    #[test]
    fn nd_prefix_rejected() {
        // Sanity: `nd.*` steps are not this bridge's business.
        let bridge = CypherBridge;
        let mut step = make_step("nd.tensors", Some("{}"));
        let result = bridge.route(&mut step);
        match result {
            Err(OrchestrationError::DomainUnavailable(_)) => {}
            other => panic!("expected DomainUnavailable, got {:?}", other),
        }
    }

    /// TD-INT-6 — empty candidate list escalates.
    #[test]
    fn disambiguate_empty_candidates_escalates() {
        let chain = ContextChain::new(8);
        let result = disambiguate_parse_candidates(&chain, 0, Vec::new());
        assert!(result.is_err(), "empty candidates must escalate");
    }

    /// TD-INT-6 — single candidate escalates (margin = 0).
    #[test]
    fn disambiguate_single_candidate_escalates() {
        let chain = ContextChain::new(8);
        let cand = CrystalFingerprint::Binary16K(Box::new([0u64; 256]));
        let result = disambiguate_parse_candidates(&chain, 0, vec![cand]);
        assert!(result.is_err(), "single candidate must escalate");
    }
}
