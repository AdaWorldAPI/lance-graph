//! Strategy #18: StyleStrategy — the thinking-style planning substrate.
//!
//! Thinking styles are THE planning substrate (not recipes in isolation): a style
//! carries both the *selection* (which way to think) and, via its τ (tau) address,
//! the *executable* JIT path. This strategy wires the shipped contract substrate
//! into the planner's default registry — mirroring the `mul::escalation` precedent
//! (a thin planner module that `pub use`s the zero-dep contract + one adapter).
//!
//! ## The pipeline this attaches to (all shipped in `lance-graph-contract`)
//!
//! ```text
//! ThinkingStyle ─cluster()─▶ StyleCluster ─▶ Mechanism ─▶ the recipes that fire
//!      │ tau()                                                (recipe_kernels::Tactic)
//!      ▼
//!   τ macro address ──▶ JitTemplate ──▶ KernelHandle   (ExecTarget::Jit; jit.rs)
//! ```
//!
//! The **style selects the recipe** (by cluster→mechanism affinity), runs the
//! selected `Tactic` kernels over a `ThoughtCtx` built from the `PlanContext`
//! markers, and surfaces the style's τ address (the JIT entry point) on the result.
//! `ExecTarget::Jit` = the τ→template→Cranelift→`KernelHandle` path; `ExecTarget::Elixir`
//! = the interpreted `recipe_kernels` layer this slice exercises.
//!
//! ## Slice scope (D-MBX-A6-P3a)
//!
//! First cut: resolve the style → select + run its cluster's recipe kernels over a
//! `ThoughtCtx` (the recipe substrate the planner did not consume before). The plan
//! passes through unchanged — this wires the cognitive substrate, not plan semantics.
//! Deferred: `Outcome`→`Candidate`/`KanbanMove` adapter, the JIT compile call, and the
//! membrane commit path (see the D-MBX-COMPLETION-MAP / board).

use lance_graph_contract::recipe_kernels::{kernel, ThoughtCtx};
use lance_graph_contract::recipes::{Mechanism, Recipe, RECIPES};
use lance_graph_contract::thinking::{StyleCluster, ThinkingStyle};

use crate::ir::{Arena, LogicalOp};
use crate::traits::{PlanCapability, PlanContext, PlanInput, PlanStrategy};
use crate::PlanError;

/// Default thinking style when the `PlanContext` carries no explicit style.
///
/// `Analytical` (the Analytical cluster) is the conservative convergent default —
/// it selects truth-aware/parallel recipes, never the divergent/randomizing ones.
pub const DEFAULT_STYLE: ThinkingStyle = ThinkingStyle::Analytical;

/// The thinking-style planning substrate strategy.
#[derive(Debug, Default)]
pub struct StyleStrategy;

impl StyleStrategy {
    /// Map a behavioural cluster to the recipe [`Mechanism`] it preferentially fires.
    ///
    /// This is the **style → recipe selector** (the load-bearing link): a style does
    /// not name recipe ids, it names a *way of thinking*, and the cluster's mechanism
    /// chooses which of the 34 recipes are in-character.
    fn cluster_mechanism(cluster: StyleCluster) -> Mechanism {
        match cluster {
            // Analytical / Direct = convergent, truth-aware (deduce, revise, critique).
            StyleCluster::Analytical | StyleCluster::Direct => Mechanism::TruthAwareInference,
            // Creative / Exploratory = divergent (randomize, reframe, analogize).
            StyleCluster::Creative | StyleCluster::Exploratory => Mechanism::StructuralDivergence,
            // Empathic = parallel-independent perspective taking.
            StyleCluster::Empathic => Mechanism::ParallelIndependence,
            // Meta = the cross-cutting infrastructure tactics (meta-cognition, framing).
            StyleCluster::Meta => Mechanism::Infrastructure,
        }
    }

    /// The recipes a given style fires: those whose mechanism matches the style's
    /// cluster mechanism. (`by_mechanism` is a contract lookup; inlined here to keep
    /// the borrow `'static`.)
    fn recipes_for(style: ThinkingStyle) -> impl Iterator<Item = &'static Recipe> {
        let want = Self::cluster_mechanism(style.cluster());
        RECIPES.iter().filter(move |r| r.mechanism == want)
    }

    /// Build the recipe substrate's [`ThoughtCtx`] from the available `PlanContext`
    /// markers. Today the planner exposes `free_will_modifier` (→ temperature) and the
    /// query feature richness (→ candidate seeds); richer markers (real sd / free-energy
    /// from the live cognitive cycle) wire in later.
    fn thought_ctx_from(ctx: &PlanContext) -> ThoughtCtx {
        // free_will_modifier ∈ ~[0,1+] biases explore↔exploit temperature.
        let mut tc = ThoughtCtx::new(vec![ctx.features.estimated_complexity as f32]);
        tc.temperature = (ctx.free_will_modifier as f32).clamp(0.0, 1.0);
        tc
    }

    /// Resolve the active thinking style from the context, or the default.
    ///
    /// `PlanContext.thinking_style` is an `Option<Vec<f64>>` style *vector* (the i4-32D
    /// style projection in f64 form); a present vector means "a style was selected
    /// upstream". First slice: presence → keep `DEFAULT_STYLE` (decoding the vector to a
    /// specific `ThinkingStyle` is the i4-32D argmax wiring, deferred). Absence → default.
    fn resolve_style(ctx: &PlanContext) -> ThinkingStyle {
        // DECISION (follow-up): decode ctx.thinking_style (i4-32D vec) → argmax ThinkingStyle.
        // First slice keeps DEFAULT_STYLE regardless; the selector machinery below is what
        // this slice proves out, not the vector decode.
        let _ = ctx.thinking_style.as_ref();
        DEFAULT_STYLE
    }
}

impl PlanStrategy for StyleStrategy {
    fn name(&self) -> &str {
        "style_strategy"
    }

    fn capability(&self) -> PlanCapability {
        // Physicalize-phase: selects the cognitive substrate, does not gate the scan.
        PlanCapability::Extension
    }

    fn affinity(&self, _ctx: &PlanContext) -> f32 {
        // Low, always-eligible: the style substrate is a default cross-cutting layer,
        // not a dialect that wins/loses on keyword match.
        0.3
    }

    fn plan(
        &self,
        input: PlanInput,
        _arena: &mut Arena<LogicalOp>,
    ) -> Result<PlanInput, PlanError> {
        let style = Self::resolve_style(&input.context);
        let mut tc = Self::thought_ctx_from(&input.context);

        // Run the style-selected recipe kernels over the ThoughtCtx (the substrate the
        // planner did not consume before). `run` = gate + apply + clamp (contract-tested).
        for recipe in Self::recipes_for(style) {
            if let Some(k) = kernel(recipe.id) {
                let _outcome = k.run(&mut tc);
            }
        }

        // First slice: the recipe pass refines the ThoughtCtx (confidence/temperature);
        // the LogicalPlan passes through unchanged. Outcome→Candidate/KanbanMove +
        // the τ→JIT compile + membrane commit are deferred (D-MBX-A6-P3 follow-ups).
        Ok(input)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn analytical_default_selects_truth_aware_recipes() {
        // DEFAULT_STYLE (Analytical) → TruthAwareInference mechanism.
        assert_eq!(DEFAULT_STYLE.cluster(), StyleCluster::Analytical);
        assert_eq!(
            StyleStrategy::cluster_mechanism(DEFAULT_STYLE.cluster()),
            Mechanism::TruthAwareInference
        );
        // It selects a non-empty, in-character recipe set, and every selected recipe
        // genuinely carries that mechanism.
        let fired: Vec<_> = StyleStrategy::recipes_for(DEFAULT_STYLE).collect();
        assert!(!fired.is_empty(), "Analytical must fire some recipes");
        assert!(fired
            .iter()
            .all(|r| r.mechanism == Mechanism::TruthAwareInference));
    }

    #[test]
    fn each_cluster_maps_to_a_mechanism_and_fires_recipes() {
        for style in ThinkingStyle::ALL {
            // tau() is the JIT address — every style has one (grounds ExecTarget::Jit).
            let _tau = style.tau();
            let mech = StyleStrategy::cluster_mechanism(style.cluster());
            // The selector is total: every cluster's mechanism exists in the catalogue.
            assert!(
                RECIPES.iter().any(|r| r.mechanism == mech),
                "cluster {:?} mechanism {:?} must match >=1 recipe",
                style.cluster(),
                mech
            );
        }
    }

    #[test]
    fn plan_runs_style_recipes_without_error() {
        let s = StyleStrategy;
        let mut arena = Arena::new();
        let input = PlanInput {
            plan: None,
            context: PlanContext {
                query: "MATCH (n:Person) RETURN n".into(),
                features: crate::traits::QueryFeatures::default(),
                free_will_modifier: 0.7,
                thinking_style: None,
                nars_hint: None,
            },
        };
        let out = s.plan(input, &mut arena);
        assert!(
            out.is_ok(),
            "style strategy plan() must pass through cleanly"
        );
    }
}
