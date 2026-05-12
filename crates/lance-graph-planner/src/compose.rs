//! Pipeline composition: run selected strategies in phase order.

#[allow(unused_imports)] // LogicalOp intended for plan composition wiring
use crate::ir::{Arena, LogicalOp, LogicalPlan};
use crate::traits::{PlanContext, PlanInput, PlanStrategy};
use crate::PlanError;

/// Compose selected strategies into a pipeline and execute.
pub fn compose_and_execute(
    strategies: &[&dyn PlanStrategy],
    context: PlanContext,
) -> Result<LogicalPlan, PlanError> {
    let mut arena = Arena::new();
    let mut current = PlanInput {
        plan: None,
        context,
    };

    tracing::info!(
        "Composing {} strategies: [{}]",
        strategies.len(),
        strategies
            .iter()
            .map(|s| s.name())
            .collect::<Vec<_>>()
            .join(" → "),
    );

    for strategy in strategies {
        tracing::debug!(
            "Running strategy: {} (capability: {:?})",
            strategy.name(),
            strategy.capability()
        );
        current = strategy.plan(current, &mut arena)?;
    }

    current.plan.ok_or_else(|| {
        PlanError::Plan("No strategy produced a plan. Check strategy selection.".into())
    })
}
