//! **LAB-ONLY consumer.** `OrchestrationBridge` impl for codec research —
//! owns `StepDomain::Ndarray`.
//!
//! The canonical architecture is `OrchestrationBridge` + `UnifiedStep` in
//! the contract. Research is just one consumer plugged into that trait;
//! production consumers plug in the same way under different `StepDomain`
//! values (Crew / Ladybug / N8n / LanceGraph / Thinking / Query / Semantic
//! / Persistence / Inference / Learning). The architecture does not
//! revolve around this consumer.
//!
//! Dispatches `nd.calibrate`, `nd.probe`, `nd.tensors` step-types through
//! the codec_research module. Combined with the LanceGraph bridge on the
//! planner, together they cover `lg.*` + `nd.*` domains.
//!
//! Consumers combine bridges: `Vec<Box<dyn OrchestrationBridge>>` and route
//! each step to whichever bridge reports `domain_available() = true`.

use lance_graph_contract::nars::InferenceType;
use lance_graph_contract::orchestration::{
    OrchestrationBridge, OrchestrationError, StepDomain, StepStatus, UnifiedStep,
};
use lance_graph_contract::plan::ThinkingContext;
use lance_graph_contract::thinking::ThinkingStyle;

use crate::codec_research;
use crate::wire::{WireCalibrateRequest, WireProbeRequest, WireTensorsRequest};

pub struct CodecResearchBridge;

impl OrchestrationBridge for CodecResearchBridge {
    fn route(&self, step: &mut UnifiedStep) -> Result<(), OrchestrationError> {
        let domain = StepDomain::from_step_type(&step.step_type)
            .ok_or_else(|| OrchestrationError::RoutingFailed(format!(
                "unknown step_type prefix: {}", step.step_type
            )))?;
        if domain != StepDomain::Ndarray {
            return Err(OrchestrationError::DomainUnavailable(domain));
        }

        step.status = StepStatus::Running;
        let op = step.step_type.strip_prefix("nd.").unwrap_or(&step.step_type);
        let args = step.reasoning.as_deref().unwrap_or("{}");

        match op {
            "tensors" => {
                let req: WireTensorsRequest = serde_json::from_str(args)
                    .map_err(|e| OrchestrationError::ExecutionFailed(e.to_string()))?;
                let r = codec_research::list_tensors(&req)
                    .map_err(|e| OrchestrationError::ExecutionFailed(e))?;
                step.status = StepStatus::Completed;
                step.reasoning = Some(format!(
                    "tensors total={} cam_pq={} passthrough={} skip={}",
                    r.total, r.cam_pq, r.passthrough, r.skip
                ));
                Ok(())
            }
            "calibrate" => {
                let req: WireCalibrateRequest = serde_json::from_str(args)
                    .map_err(|e| OrchestrationError::ExecutionFailed(e.to_string()))?;
                let r = codec_research::calibrate_tensor(&req)
                    .map_err(|e| OrchestrationError::ExecutionFailed(e))?;
                step.status = StepStatus::Completed;
                step.confidence = Some(r.icc_3_1 as f64);
                step.reasoning = Some(format!(
                    "calibrate tensor={} icc={:.4} rel_l2={:.4} elapsed={}ms",
                    r.tensor_name, r.icc_3_1, r.relative_l2_error, r.elapsed_ms
                ));
                Ok(())
            }
            "probe" => {
                let req: WireProbeRequest = serde_json::from_str(args)
                    .map_err(|e| OrchestrationError::ExecutionFailed(e.to_string()))?;
                let r = codec_research::row_count_probe(&req)
                    .map_err(|e| OrchestrationError::ExecutionFailed(e))?;
                step.status = StepStatus::Completed;
                step.reasoning = Some(format!(
                    "probe tensor={} n_rows={} entries={}",
                    r.tensor_name, r.n_rows, r.entries.len()
                ));
                Ok(())
            }
            other => {
                step.status = StepStatus::Failed;
                Err(OrchestrationError::RoutingFailed(format!(
                    "unknown nd.* operation: {other}"
                )))
            }
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
        matches!(domain, StepDomain::Ndarray)
    }
}
