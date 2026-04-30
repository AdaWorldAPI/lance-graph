//! `OrchestrationBridge` implementation for `PlannerAwareness`.
//!
//! Fulfils the contract in `lance_graph_contract::orchestration` — the
//! single routing trait that replaces duplicated dispatch logic across
//! crewai-rust (StepRouter), n8n-rs (crew_router / ladybug_router),
//! ladybug-rs (HybridEngine), and per-crate `*_bridge.rs` modules.
//!
//! # Scope
//!
//! This impl owns the `StepDomain::LanceGraph` domain. Step types with
//! the `lg.` prefix route here:
//!
//! | step_type             | Action                                         |
//! |-----------------------|------------------------------------------------|
//! | `lg.plan_auto`        | Run `plan_auto` on the request query           |
//! | `lg.plan_full`        | Run `plan_full` (requires situation blackboard)|
//! | `lg.orchestrate`      | Resolve `ThinkingContext` only                 |
//! | `lg.health`           | Domain availability check                      |
//!
//! The bridge mutates the `UnifiedStep` in place: sets `status`,
//! populates `thinking` on success, writes `reasoning` to a short
//! human-readable summary. Rich results (strategies_used, PlanResult)
//! are surfaced via the inherent `PlannerAwareness::plan_*` methods —
//! the bridge is the routing envelope, not the result carrier, per the
//! contract module's BridgeSlot guidance.
//!
//! # Why this exists
//!
//! Before this impl, consumers had to depend on the concrete
//! `PlannerAwareness` (or per-domain bridge crates) to invoke planning.
//! With the impl landed, any consumer holding a
//! `Box<dyn OrchestrationBridge>` can route `lg.*` steps without
//! coupling to planner internals.

use lance_graph_contract::nars::InferenceType;
use lance_graph_contract::orchestration::{
    OrchestrationBridge, OrchestrationError, StepDomain, StepStatus, UnifiedStep,
};
use lance_graph_contract::plan::ThinkingContext as ContractThinkingContext;
use lance_graph_contract::thinking::{
    FieldModulation as ContractFieldModulation, ThinkingStyle as ContractThinkingStyle,
};

use crate::PlannerAwareness;

impl OrchestrationBridge for PlannerAwareness {
    fn route(&self, step: &mut UnifiedStep) -> Result<(), OrchestrationError> {
        let domain = StepDomain::from_step_type(&step.step_type)
            .ok_or_else(|| OrchestrationError::RoutingFailed(format!(
                "unknown step_type prefix: {}",
                step.step_type,
            )))?;
        if domain != StepDomain::LanceGraph {
            return Err(OrchestrationError::DomainUnavailable(domain));
        }

        step.status = StepStatus::Running;

        // The `step_type` carries the operation name after the `lg.` prefix.
        // The query string is read from `step.reasoning` (set by the caller
        // before route() fires). Richer inputs flow through BridgeSlots in
        // a future extension — this impl covers the minimal path.
        let op = step.step_type.strip_prefix("lg.").unwrap_or(&step.step_type);
        let query = step.reasoning.as_deref().unwrap_or("");

        match op {
            "plan_auto" => match self.plan_auto(query) {
                Ok(result) => {
                    step.status = StepStatus::Completed;
                    step.reasoning = Some(format!(
                        "plan_auto strategies=[{}] free_will={:.3}",
                        result.strategies_used.join(","),
                        result.free_will_modifier,
                    ));
                    step.confidence = Some(result.free_will_modifier);
                    step.thinking = result.thinking.as_ref().map(thinking_to_contract);
                    Ok(())
                }
                Err(e) => {
                    step.status = StepStatus::Failed;
                    Err(OrchestrationError::ExecutionFailed(e.to_string()))
                }
            },
            "orchestrate" => {
                // Resolve a thinking context with default situation → bridge
                // fabricates a minimal ContractThinkingContext. Callers wanting
                // a full orchestrate should use the inherent method.
                let ctx = resolve_thinking_minimal(
                    ContractThinkingStyle::Analytical,
                    InferenceType::Deduction,
                );
                step.thinking = Some(ctx);
                step.status = StepStatus::Completed;
                step.reasoning = Some("thinking resolved".to_string());
                Ok(())
            }
            "health" => {
                step.status = StepStatus::Completed;
                step.reasoning = Some(format!(
                    "lance-graph-planner: {} strategies registered",
                    self.strategies().len(),
                ));
                Ok(())
            }
            other => {
                step.status = StepStatus::Failed;
                Err(OrchestrationError::RoutingFailed(format!(
                    "unknown lg.* operation: {other}"
                )))
            }
        }
    }

    fn resolve_thinking(
        &self,
        style: ContractThinkingStyle,
        inference_type: InferenceType,
    ) -> ContractThinkingContext {
        resolve_thinking_minimal(style, inference_type)
    }

    fn domain_available(&self, domain: StepDomain) -> bool {
        matches!(domain, StepDomain::LanceGraph)
    }
}

fn resolve_thinking_minimal(
    style: ContractThinkingStyle,
    inference_type: InferenceType,
) -> ContractThinkingContext {
    use lance_graph_contract::nars::{QueryStrategy, SemiringChoice};
    ContractThinkingContext {
        style,
        modulation: ContractFieldModulation::default(),
        inference_type,
        strategy: match inference_type {
            InferenceType::Deduction => QueryStrategy::CamExact,
            InferenceType::Induction => QueryStrategy::CamWide,
            InferenceType::Abduction => QueryStrategy::DnTreeFull,
            _ => QueryStrategy::CamExact,
        },
        semiring: match inference_type {
            InferenceType::Deduction | InferenceType::Induction => SemiringChoice::HammingMin,
            InferenceType::Revision | InferenceType::Synthesis => SemiringChoice::NarsTruth,
            _ => SemiringChoice::Boolean,
        },
        free_will_modifier: 1.0,
        exploratory: false,
    }
}

/// Convert internal planner `ThinkingContext` to the contract's version.
///
/// Planner's type has fields the contract doesn't model (sigma_stage) and
/// per-crate enums (ThinkingStyle etc.). We lose some resolution here —
/// acceptable for the routing envelope; consumers needing the full planner
/// context should call `PlannerAwareness::plan_auto` directly.
fn thinking_to_contract(p: &crate::thinking::ThinkingContext) -> ContractThinkingContext {
    use lance_graph_contract::nars::{QueryStrategy, SemiringChoice};

    let style = planner_style_to_contract(p.style);
    let inference = planner_nars_to_contract(p.nars_type);
    ContractThinkingContext {
        style,
        modulation: ContractFieldModulation::default(),
        inference_type: inference,
        strategy: match inference {
            InferenceType::Deduction => QueryStrategy::CamExact,
            InferenceType::Induction => QueryStrategy::CamWide,
            InferenceType::Abduction => QueryStrategy::DnTreeFull,
            _ => QueryStrategy::CamExact,
        },
        semiring: SemiringChoice::HammingMin,
        free_will_modifier: p.free_will_modifier,
        exploratory: p.exploratory,
    }
}

/// Planner's 12-style enum → contract's 36-style enum.
///
/// Mapping from `THINKING_RECONCILIATION.md`.
fn planner_style_to_contract(s: crate::thinking::ThinkingStyle) -> ContractThinkingStyle {
    use crate::thinking::ThinkingStyle as P;
    match s {
        P::Analytical    => ContractThinkingStyle::Analytical,
        P::Convergent    => ContractThinkingStyle::Logical,
        P::Systematic    => ContractThinkingStyle::Systematic,
        P::Creative      => ContractThinkingStyle::Creative,
        P::Divergent     => ContractThinkingStyle::Imaginative,
        P::Exploratory   => ContractThinkingStyle::Exploratory,
        P::Focused       => ContractThinkingStyle::Precise,
        P::Diffuse       => ContractThinkingStyle::Gentle,
        P::Peripheral    => ContractThinkingStyle::Poetic,
        P::Intuitive     => ContractThinkingStyle::Curious,
        P::Deliberate    => ContractThinkingStyle::Methodical,
        P::Metacognitive => ContractThinkingStyle::Reflective,
    }
}

/// Planner's NARS inference type → contract's.
fn planner_nars_to_contract(n: crate::thinking::NarsInferenceType) -> InferenceType {
    use crate::thinking::NarsInferenceType as P;
    match n {
        P::Deduction => InferenceType::Deduction,
        P::Induction => InferenceType::Induction,
        P::Abduction => InferenceType::Abduction,
        P::Revision  => InferenceType::Revision,
        P::Synthesis => InferenceType::Synthesis,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn health_step_routes_successfully() {
        let planner = PlannerAwareness::new();
        let mut step = UnifiedStep {
            step_id:"s1".into(),
            step_type: "lg.health".into(),
            status: StepStatus::Pending,
            thinking: None,
            reasoning: None,
            confidence: None,
            depends_on: vec![],
        };
        planner.route(&mut step).unwrap();
        assert_eq!(step.status, StepStatus::Completed);
        assert!(step.reasoning.as_ref().unwrap().contains("strategies"));
    }

    #[test]
    fn plan_auto_step_routes_for_match_query() {
        let planner = PlannerAwareness::new();
        let mut step = UnifiedStep {
            step_id:"s2".into(),
            step_type: "lg.plan_auto".into(),
            status: StepStatus::Pending,
            thinking: None,
            reasoning: Some("MATCH (n) RETURN n".into()),
            confidence: None,
            depends_on: vec![],
        };
        planner.route(&mut step).unwrap();
        assert_eq!(step.status, StepStatus::Completed);
        let r = step.reasoning.as_ref().unwrap();
        assert!(r.contains("plan_auto"));
    }

    #[test]
    fn wrong_domain_step_returns_domain_unavailable() {
        let planner = PlannerAwareness::new();
        let mut step = UnifiedStep {
            step_id:"s3".into(),
            step_type: "crew.agent.think".into(),
            status: StepStatus::Pending,
            thinking: None,
            reasoning: None,
            confidence: None,
            depends_on: vec![],
        };
        match planner.route(&mut step) {
            Err(OrchestrationError::DomainUnavailable(StepDomain::Crew)) => {}
            other => panic!("expected DomainUnavailable(Crew), got {other:?}"),
        }
    }

    #[test]
    fn domain_available_only_for_lance_graph() {
        let planner = PlannerAwareness::new();
        assert!(planner.domain_available(StepDomain::LanceGraph));
        assert!(!planner.domain_available(StepDomain::Crew));
        assert!(!planner.domain_available(StepDomain::N8n));
        assert!(!planner.domain_available(StepDomain::Ladybug));
        assert!(!planner.domain_available(StepDomain::Ndarray));
    }

    #[test]
    fn resolve_thinking_returns_minimal_context() {
        let planner = PlannerAwareness::new();
        let ctx = planner.resolve_thinking(
            ContractThinkingStyle::Analytical,
            InferenceType::Deduction,
        );
        assert_eq!(ctx.style, ContractThinkingStyle::Analytical);
        assert_eq!(ctx.inference_type, InferenceType::Deduction);
    }
}
