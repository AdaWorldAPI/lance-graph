//! Planner bridge — delegates `WirePlan*` DTOs to `lance-graph-planner`.
//!
//! Behind `--features with-planner`. Same pattern as `with-engine`:
//! heavy dep is optional; absence returns a clear runtime error.
//!
//! Per INTEGRATION_PLAN_CS.md §5-layer-stack, Layer 4 is the planner.
//! The shader-driver (Layer 2) exposes plan operations through the same
//! unified REST/gRPC endpoint and delegates execution here. Nothing
//! below Layer 4 reimplements planning.

use std::time::Instant;

use lance_graph_planner::mul::SituationInput as PlannerSituation;
use lance_graph_planner::selector::StrategySelector;
use lance_graph_planner::PlannerAwareness;

use crate::wire::{WirePlanRequest, WirePlanResponse, WireSituation};

/// Construct a planner. One per server. `with_explicit` when strategies
/// are named; `new` for default auto mode.
pub fn build_planner(strategy_names: &[String]) -> PlannerAwareness {
    if strategy_names.is_empty() {
        PlannerAwareness::new()
    } else {
        PlannerAwareness::with_explicit(strategy_names.to_vec())
    }
}

/// Execute a plan request against a pre-built planner instance.
///
/// `auto` mode: runs `plan_auto` (no MUL).
/// `full` mode: runs `plan_full` with the wire-supplied SituationInput
/// (or all-0.5 defaults when absent).
pub fn plan(planner: &PlannerAwareness, req: &WirePlanRequest) -> Result<WirePlanResponse, String> {
    let t0 = Instant::now();
    let mode = req.mode.as_str();

    // If explicit strategies were requested, build a fresh planner for them.
    // The shared planner doesn't get mutated — clients can request different
    // strategy sets per request without side effects.
    let fresh_planner: Option<PlannerAwareness> = if !req.strategies.is_empty() {
        Some(PlannerAwareness::with_explicit(req.strategies.clone()))
    } else {
        None
    };
    let p: &PlannerAwareness = fresh_planner.as_ref().unwrap_or(planner);

    let result = match mode {
        "auto" => p.plan_auto(&req.query).map_err(|e| e.to_string())?,
        "full" => {
            let s = req.situation.clone().unwrap_or_default();
            let situation = situation_from_wire(&s);
            p.plan_full(&req.query, &situation).map_err(|e| e.to_string())?
        }
        other => return Err(format!("unknown plan mode '{other}' (use 'auto' or 'full')")),
    };

    Ok(WirePlanResponse {
        mode: mode.to_string(),
        strategies_used: result.strategies_used,
        free_will_modifier: result.free_will_modifier,
        compass_score: result.compass_score,
        mul_gate: result.mul.as_ref().map(|m| format!("{:?}", m)),
        thinking_style_name: result.thinking.as_ref().map(|t| format!("{:?}", t.style)),
        nars_type: result.thinking.as_ref().map(|t| format!("{:?}", t.nars_type)),
        elapsed_ms: t0.elapsed().as_millis() as u64,
    })
}

fn situation_from_wire(w: &WireSituation) -> PlannerSituation {
    PlannerSituation {
        felt_competence: w.felt_competence,
        demonstrated_competence: w.demonstrated_competence,
        source_reliability: w.source_reliability,
        environment_stability: w.environment_stability,
        calibration_accuracy: w.calibration_accuracy,
        complexity_ratio: w.complexity_ratio,
        ..Default::default()
    }
}

/// Expose the strategy selector pattern — clients can inspect which
/// strategies a planner has registered.
pub fn registered_strategies(planner: &PlannerAwareness) -> Vec<String> {
    planner
        .strategies()
        .iter()
        .map(|s| s.name().to_string())
        .collect()
}

/// Default-mode selector, matching the planner's own default.
#[allow(dead_code)]
pub fn default_selector() -> StrategySelector {
    StrategySelector::default()
}
