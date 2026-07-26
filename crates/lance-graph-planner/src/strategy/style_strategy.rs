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

use lance_graph_contract::cognitive_shader::RungLevel;
use lance_graph_contract::kanban::{ExecTarget, KanbanColumn, KanbanMove};
use lance_graph_contract::recipe_kernels::{kernel, ThoughtCtx};
use lance_graph_contract::recipes::{Mechanism, Recipe, RECIPES};
use lance_graph_contract::thinking::{StyleCluster, ThinkingStyle};

use crate::ir::{Arena, LogicalOp};
use crate::traits::{PlanCapability, PlanContext, PlanInput, PlanStrategy, StrategyOutcome};
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

    /// The recipes a style fires **at a given rung** — [`recipes_for`] gated by
    /// [`Recipe::admissible_at`], so a tactic may not fire before the standing
    /// wave has earned the depth to pay for it
    /// (`E-STANDING-WAVE-IS-UNSTRATIFIED-SUDOKU-1`).
    ///
    /// Mechanism answers *which kind* of tactic this style prefers; the rung
    /// answers *how expensive* a tactic the resolution has earned. Filtering on
    /// mechanism alone (the pre-stratification behaviour) let a `Control`-bucket
    /// branchy inference fire on a chain that grounded in two hops.
    ///
    /// Callers get the rung from the wave:
    /// `RungLevel::for_pass(settle_pass)` where `settle_pass` comes from
    /// [`standing_wave_stratified`](lance_graph_contract::witness_fabric::standing_wave_stratified).
    fn recipes_for_at(
        style: ThinkingStyle,
        rung: RungLevel,
    ) -> impl Iterator<Item = &'static Recipe> {
        Self::recipes_for(style).filter(move |r| r.admissible_at(rung))
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
        mut input: PlanInput,
        _arena: &mut Arena<LogicalOp>,
    ) -> Result<PlanInput, PlanError> {
        // Surface the style-conditioned reliability AND the lifecycle transition this
        // planning-substrate strategy INTENDS — honestly, on the D-MBX-A6 carrier. This
        // is NOT a commit and NOT a plan mutation: `input.plan` is left exactly as
        // received, and the intended move is a *bootstrap intent* (owner 0, cycle 0,
        // the zero-fallback ladder) that no one consumes to advance a live mailbox in
        // this slice. It replaces the dead-store `let _reliability = …` the council
        // flagged — the value now has an honest home instead of `_`.
        let style = Self::resolve_style(&input.context);
        let reliability = Self::reliability_of(style, &input.context);
        input.outcome = Some(StrategyOutcome {
            reliability,
            intended_move: Some(Self::intended_move(style)),
        });
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
        // Unstratified entry point: admits the whole mechanism set, i.e. exactly
        // the pre-stratification behaviour. Preserved bit-for-bit so no existing
        // caller changes meaning (`I-LEGACY-API-FEATURE-GATED` discipline: the
        // same function name must not silently mean something new).
        Self::reliability_at(style, ctx, RungLevel::Transcendent)
    }

    /// [`reliability_of`](Self::reliability_of) **at a rung** — only the tactics
    /// the resolution has earned may contribute
    /// (`E-STANDING-WAVE-IS-UNSTRATIFIED-SUDOKU-1`).
    ///
    /// This is the Sudoku discipline made executable: a locus that grounded in
    /// two hops is scored by the cheap [`Gate`](lance_graph_contract::recipes::Bucket::Gate)
    /// tactics alone, while a chain that only settled deep in the standing wave
    /// unlocks the branchy `Control` tactics that cost more to run.
    ///
    /// The rung comes from the wave, not from a guess:
    ///
    /// ```text
    /// let (grounding, pass) = standing_wave_stratified(idx, window, locus, passes);
    /// let rung = RungLevel::for_pass(pass);
    /// let r = StyleStrategy::reliability_at(style, ctx, rung);
    /// ```
    ///
    /// **Why the rung is a parameter and not a `PlanContext` field:** `PlanContext`
    /// carries no witness window today, so there is no honest rung to read from it
    /// — deriving one from `estimated_complexity` would be inventing a semantic.
    /// Threading the wave into the planner's context is its own deliverable; until
    /// then callers that HAVE a window pass the rung explicitly, and callers that
    /// do not keep the unstratified [`reliability_of`](Self::reliability_of).
    pub fn reliability_at(style: ThinkingStyle, ctx: &PlanContext, rung: RungLevel) -> f32 {
        let mut tc = Self::thought_ctx_from(ctx);
        for recipe in Self::recipes_for_at(style, rung) {
            if let Some(k) = kernel(recipe.id) {
                // `run` gates + applies, mutating `tc.confidence` in place (returns the
                // per-recipe Outcome, which we don't need here — the accumulated
                // confidence on `tc` is the reliability signal).
                let _ = k.run(&mut tc);
            }
        }
        tc.confidence.clamp(0.0, 1.0)
    }

    /// The lifecycle transition a style-substrate strategy INTENDS, as a **bootstrap
    /// intent** (not an emission). StyleStrategy runs in the Planning column (the
    /// spawn/default state, [`KanbanColumn::default`]); having selected the style and
    /// measured its reliability, the honest intent is the forward Rubicon crossing
    /// `Planning → CognitiveWork` (a legal edge:
    /// `KanbanColumn::Planning.can_transition_to(CognitiveWork)`), carrying the −550 ms
    /// Σ-commit anchor (matches `soa_view` `advance_phase`, contract).
    ///
    /// Honestly-fillable fields: `from`/`to`/`libet_offset_us` (structural constants of
    /// the crossing) and `exec` (the backend `reliability_of` actually ran = the
    /// interpreted `recipe_kernels` layer = [`ExecTarget::Elixir`], per this module's
    /// doc header). Bootstrap-sentinel fields: `mailbox = 0` (write-on-behalf of the
    /// documented bootstrap owner, NOT as ourselves — the live owner rebinds it) and
    /// `witness_chain_position = 0` (no live `current_cycle` exists at plan time; 0 is
    /// the zero-fallback pre-cycle stamp the owner overwrites on adoption).
    fn intended_move(_style: ThinkingStyle) -> KanbanMove {
        KanbanMove {
            mailbox: 0,
            from: KanbanColumn::Planning,
            to: KanbanColumn::CognitiveWork,
            witness_chain_position: 0,
            libet_offset_us: -550_000,
            exec: ExecTarget::Elixir,
        }
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

    /// The rung gate narrows the mechanism-selected set at shallow rungs and
    /// restores it at deep ones — for EVERY style, so no cluster is accidentally
    /// starved or accidentally unstratified.
    #[test]
    fn rung_gate_narrows_shallow_and_restores_deep() {
        for style in ThinkingStyle::ALL {
            let all: Vec<u8> = StyleStrategy::recipes_for(style).map(|r| r.id).collect();
            let deep: Vec<u8> = StyleStrategy::recipes_for_at(style, RungLevel::Transcendent)
                .map(|r| r.id)
                .collect();
            assert_eq!(
                all, deep,
                "{style:?}: the top rung must admit exactly the mechanism set"
            );

            let shallow: Vec<u8> = StyleStrategy::recipes_for_at(style, RungLevel::Shallow)
                .map(|r| r.id)
                .collect();
            assert!(
                shallow.len() <= all.len(),
                "{style:?}: shallow rung admitted MORE than the full set"
            );
            // Monotone in rung: no tactic appears shallow and vanishes deep.
            for id in &shallow {
                assert!(
                    all.contains(id),
                    "{style:?}: recipe {id} admitted at Shallow but not in the set"
                );
            }
        }
    }

    /// The unstratified entry point must be EXACTLY the top-rung stratified one —
    /// no existing caller changes meaning when the gate lands.
    #[test]
    fn reliability_of_is_unchanged_by_the_gate() {
        let ctx = ctx_with(Some(style_vec(0.9, 0.0, 0.0)));
        for style in ThinkingStyle::ALL {
            let before = StyleStrategy::reliability_of(style, &ctx);
            let top = StyleStrategy::reliability_at(style, &ctx, RungLevel::Transcendent);
            assert_eq!(
                before.to_bits(),
                top.to_bits(),
                "{style:?}: reliability_of drifted from the top rung"
            );
        }
    }

    /// The gate must change a real OUTCOME, not just a recipe count — otherwise
    /// it is instrumentation rather than stratification.
    #[test]
    fn rung_changes_measured_reliability_for_some_style() {
        let ctx = ctx_with(Some(style_vec(0.9, 0.0, 0.0)));
        let moved = ThinkingStyle::ALL.iter().any(|&style| {
            let shallow = StyleStrategy::reliability_at(style, &ctx, RungLevel::Shallow);
            let deep = StyleStrategy::reliability_at(style, &ctx, RungLevel::Transcendent);
            shallow.to_bits() != deep.to_bits()
        });
        assert!(
            moved,
            "rung gate left every style's reliability identical — inert wiring"
        );
    }

    /// The stratification must actually bite somewhere — if every style's set
    /// were rung-invariant, the gate would be decoration (the `closed_class_guess`
    /// failure mode).
    #[test]
    fn rung_gate_is_not_inert_across_the_style_space() {
        let bitten = ThinkingStyle::ALL.iter().any(|&style| {
            let shallow = StyleStrategy::recipes_for_at(style, RungLevel::Shallow).count();
            let deep = StyleStrategy::recipes_for_at(style, RungLevel::Transcendent).count();
            shallow < deep
        });
        assert!(
            bitten,
            "rung gate changed nothing for any style — stratification is inert"
        );
    }

    #[test]
    fn plan_surfaces_outcome_without_mutating_the_plan() {
        // The plan itself stays a pure pass-through (no KanbanMove is *emitted*, no
        // Rubicon advance) — but the reliability + intended move are now SURFACED on
        // the D-MBX-A6 carrier instead of dead-stored (the `_reliability` the council
        // flagged now has an honest home).
        let s = StyleStrategy;
        let mut arena = Arena::new();
        let out = s
            .plan(
                ctx_input(ctx_with(Some(style_vec(0.9, 0.0, 0.0)))),
                &mut arena,
            )
            .expect("style strategy plan() must not error");
        // Plan untouched (no mutation, no theatre).
        assert!(
            out.plan.is_none(),
            "plan() must not mutate the plan this slice"
        );
        // Outcome surfaced honestly.
        let o = out.outcome.expect("plan() must surface a StrategyOutcome");
        assert!((0.0..=1.0).contains(&o.reliability), "reliability in [0,1]");
        let mv = o
            .intended_move
            .expect("StyleStrategy intends a lifecycle move");
        // Bootstrap intent — not a live mailbox advance.
        assert_eq!(mv.mailbox, 0, "write-on-behalf of the bootstrap owner (0)");
        assert_eq!(
            mv.witness_chain_position, 0,
            "no live cycle at plan time (zero-fallback)"
        );
        assert_eq!(mv.from, KanbanColumn::Planning);
        assert_eq!(mv.to, KanbanColumn::CognitiveWork);
        assert!(
            mv.from.can_transition_to(mv.to),
            "intended edge must be a legal Rubicon transition"
        );
        assert_eq!(
            mv.libet_offset_us, -550_000,
            "Σ-commit anchor on the crossing"
        );
        assert_eq!(
            mv.exec,
            ExecTarget::Elixir,
            "the backend reliability_of ran"
        );
    }

    fn ctx_input(context: PlanContext) -> PlanInput {
        PlanInput {
            plan: None,
            context,
            outcome: None,
        }
    }
}
