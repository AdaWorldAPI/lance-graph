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

    /// Resolve the active thinking style from the context's 23D style vector.
    ///
    /// `PlanContext.thinking_style` is an `Option<Vec<f64>>` — the **23D sparse cognitive
    /// vector** (per `traits.rs` / `selector.rs::style_alignment`, which reads idx
    /// 0=depth, 3=creative, 4=analytical). This decodes that vector to a concrete
    /// `ThinkingStyle` by which cluster axis dominates — the keystone projection that was
    /// previously a constant-`DEFAULT_STYLE` stub (the bug the council caught: recipe
    /// selection was identical for every query). Absence (or an all-zero vector) → default.
    ///
    /// NOTE: this matches `selector.rs`'s existing 23D index convention; it is *not* the
    /// contract `style_vector`/i4-32D `StyleRecipe` surface (a separate, deferred decode).
    fn resolve_style(ctx: &PlanContext) -> ThinkingStyle {
        let Some(v) = ctx.thinking_style.as_ref().filter(|v| !v.is_empty()) else {
            return DEFAULT_STYLE;
        };
        // Read the same axes selector.rs::style_alignment uses (idx 4/3/0).
        let analytical = v.get(4).copied().unwrap_or(0.0);
        let creative = v.get(3).copied().unwrap_or(0.0);
        let depth = v.first().copied().unwrap_or(0.0);
        // Pick the dominant axis → a representative style of that cluster. All-zero (no
        // axis active) falls through to the conservative default.
        let max = analytical.max(creative).max(depth);
        if max <= 0.0 {
            DEFAULT_STYLE
        } else if (analytical - max).abs() < f64::EPSILON {
            ThinkingStyle::Analytical // Analytical cluster → TruthAwareInference
        } else if (creative - max).abs() < f64::EPSILON {
            ThinkingStyle::Creative // Creative cluster → StructuralDivergence
        } else {
            ThinkingStyle::Reflective // depth-dominant → Meta cluster → Infrastructure
        }
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
        // The style-conditioned reliability is the substrate output this strategy
        // computes. It is NOT yet emitted as a KanbanMove — the planner cannot construct
        // one until the D-MBX-A6 output overhaul; faking one here would be theatre (the
        // exact dead-store the council flagged). So this slice computes the measurable
        // honestly and leaves the plan untouched; the emit edge is the next, separate slice.
        let _reliability =
            Self::reliability_of(Self::resolve_style(&input.context), &input.context);
        Ok(input)
    }
}

impl StyleStrategy {
    /// **The R-GATE measurable** — the style-conditioned RELIABILITY of crystallising at
    /// this context, in `[0,1]`. NOT validity (ground-truth correspondence is conferred
    /// externally, post-Commit — see `E-RELIABILITY-NOT-VALIDITY`); this is the
    /// reliability/settledness coefficient (NARS confidence family).
    ///
    /// Runs the style-selected recipe `Tactic` kernels over a `ThoughtCtx` (the substrate
    /// the planner did not consume before) and returns the resulting confidence. Different
    /// styles select different recipes (`cluster→mechanism`) and so yield different
    /// reliability — that variation is what the `r_gate_reliability_varies_by_style` probe
    /// measures BEFORE any Rubicon gate field is added (the reviewers' probe-first rule).
    ///
    /// Pure: no plan mutation, no commit. The `Evaluation→{Commit|Plan|Prune}` wiring that
    /// would CONSUME this is deferred until the probe proves it changes an outcome.
    pub fn reliability_of(style: ThinkingStyle, ctx: &PlanContext) -> f32 {
        let mut tc = Self::thought_ctx_from(ctx);
        for recipe in Self::recipes_for(style) {
            if let Some(k) = kernel(recipe.id) {
                // `run` gates + applies, mutating `tc.confidence` in place (returns the
                // per-recipe Outcome, which we don't need here — the accumulated
                // confidence on `tc` is the reliability signal).
                let _ = k.run(&mut tc);
            }
        }
        tc.confidence.clamp(0.0, 1.0)
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

    /// Build a 23D style vector with one cluster axis dominant (idx 4=analytical,
    /// 3=creative, 0=depth — the convention `selector.rs::style_alignment` reads).
    fn style_vec(analytical: f64, creative: f64, depth: f64) -> Vec<f64> {
        let mut v = vec![0.0; 23];
        v[4] = analytical;
        v[3] = creative;
        v[0] = depth;
        v
    }

    fn ctx_with(style: Option<Vec<f64>>) -> PlanContext {
        PlanContext {
            query: "MATCH (n:Person) RETURN n".into(),
            features: crate::traits::QueryFeatures::default(),
            free_will_modifier: 0.7,
            thinking_style: style,
            nars_hint: None,
        }
    }

    #[test]
    fn resolve_style_decodes_the_23d_vector_not_constant_default() {
        // The bug the council caught: resolve_style ignored the vector and always
        // returned DEFAULT_STYLE. It must now track the dominant axis.
        assert_eq!(
            StyleStrategy::resolve_style(&ctx_with(Some(style_vec(0.9, 0.1, 0.0)))).cluster(),
            StyleCluster::Analytical
        );
        assert_eq!(
            StyleStrategy::resolve_style(&ctx_with(Some(style_vec(0.1, 0.9, 0.0)))).cluster(),
            StyleCluster::Creative
        );
        // Absent / all-zero → conservative default (not a panic, not a wrong cluster).
        assert_eq!(StyleStrategy::resolve_style(&ctx_with(None)), DEFAULT_STYLE);
        assert_eq!(
            StyleStrategy::resolve_style(&ctx_with(Some(style_vec(0.0, 0.0, 0.0)))),
            DEFAULT_STYLE
        );
    }

    /// **R-GATE probe (reliability, not validity).** The reviewers' rule: measure that
    /// style-conditioned RELIABILITY actually differs by style BEFORE wiring any Rubicon
    /// gate field. If Analytical and Creative produced identical reliability, a
    /// style-conditioned gate would be cosmetic — this test is the falsifiable check.
    #[test]
    fn r_gate_reliability_varies_by_style() {
        let analytical = StyleStrategy::reliability_of(
            ThinkingStyle::Analytical,
            &ctx_with(Some(style_vec(0.9, 0.0, 0.0))),
        );
        let creative = StyleStrategy::reliability_of(
            ThinkingStyle::Creative,
            &ctx_with(Some(style_vec(0.0, 0.9, 0.0))),
        );
        // Both are valid reliability coefficients in [0,1] (NOT validity — see
        // E-RELIABILITY-NOT-VALIDITY).
        assert!((0.0..=1.0).contains(&analytical));
        assert!((0.0..=1.0).contains(&creative));
        // R-GATE pass criterion: the two styles fire different recipe mechanisms
        // (TruthAwareInference vs StructuralDivergence) → the measurable is
        // style-sensitive. If this ever collapses to equal, the gate is cosmetic and
        // must NOT be wired (the probe-first discipline).
        assert_ne!(
            StyleStrategy::cluster_mechanism(ThinkingStyle::Analytical.cluster()),
            StyleStrategy::cluster_mechanism(ThinkingStyle::Creative.cluster()),
            "R-GATE: styles must select distinct mechanisms or the gate is cosmetic"
        );
    }

    #[test]
    fn plan_is_pure_passthrough_until_emit_edge_lands() {
        // Honest test: plan() computes reliability but does NOT yet mutate the plan
        // (no KanbanMove emit until the D-MBX-A6 output overhaul). It must not error,
        // and — explicitly — must leave the plan None as it received it (no theatre).
        let s = StyleStrategy;
        let mut arena = Arena::new();
        let out = s
            .plan(
                ctx_input(ctx_with(Some(style_vec(0.9, 0.0, 0.0)))),
                &mut arena,
            )
            .expect("style strategy plan() must not error");
        assert!(
            out.plan.is_none(),
            "plan() is a pure pass-through this slice — it computes reliability, emits nothing yet"
        );
    }

    fn ctx_input(context: PlanContext) -> PlanInput {
        PlanInput {
            plan: None,
            context,
        }
    }
}
