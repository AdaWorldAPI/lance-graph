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

    /// **The peripheral-dissent watchdog** — the guard against optimizing the
    /// dominant mode into blindness.
    ///
    /// The rung gate is a hard prune, and the standing wave STOPS when it
    /// settles, so a locus that grounds cheaply is otherwise never examined by
    /// the tactics its rung excluded. "Settles fast" correlates with "looks
    /// obvious", which is precisely when a wrong answer is most expensive — the
    /// System-1 easy path this workspace's own doctrine warns about.
    ///
    /// So: run a deterministic spread sample of `k` EXCLUDED tactics as
    /// observers. They never contribute to the score. If a peripheral tactic
    /// moves reliability by more than `tol` relative to the admitted set, the
    /// cheap consensus is not trustworthy and this returns the rung to elevate
    /// to — the periphery gets to force a deeper look, never to decide.
    ///
    /// Returns `None` when the periphery agrees (or there is none — the top rung
    /// is blind to nothing).
    ///
    /// This mirrors [`WaveGrounding::Escalate`](lance_graph_contract::witness_fabric::WaveGrounding::Escalate):
    /// a *signal*, not a verdict. Cheap consensus that a watcher disputes is the
    /// same shape as a chain that leaves the ±8 horizon — both say "this needs a
    /// wider read", neither says what the answer is.
    pub fn peripheral_dissent(
        style: ThinkingStyle,
        ctx: &PlanContext,
        rung: RungLevel,
        k: usize,
        tol: f32,
    ) -> Option<RungLevel> {
        let admitted = Self::reliability_at(style, ctx, rung);
        let want = Self::cluster_mechanism(style.cluster());
        // Eligibility is applied BEFORE the stride: sampling globally and then
        // `continue`-ing on mechanism let ineligible watchers spend the whole
        // `k` budget, so the channel could report agreement without having run
        // a single relevant tactic — silence produced by the sampler, not by
        // the evidence.
        for watcher in rung.peripheral_sample_where(k, |r| r.mechanism == want) {
            let Some(kern) = kernel(watcher.id) else {
                continue;
            };
            let mut tc = Self::thought_ctx_from(ctx);
            for r in Self::recipes_for_at(style, rung) {
                if let Some(k2) = kernel(r.id) {
                    let _ = k2.run(&mut tc);
                }
            }
            let _ = kern.run(&mut tc);
            if (tc.confidence.clamp(0.0, 1.0) - admitted).abs() > tol {
                // Elevate to where this watcher would have been legal anyway.
                return Some(watcher.min_rung());
            }
        }
        None
    }

    /// **Cross-family dissent** — the independence channel, deliberately NOT
    /// the same measurement as [`peripheral_dissent`].
    ///
    /// `peripheral_dissent` only consults watchers sharing the style's own
    /// [`Mechanism`], because an off-character tactic is not evidence about how
    /// well this style reasoned. That restriction is right for CALIBRATION and
    /// wrong for INDEPENDENCE: every watcher it can hear from is a sibling of
    /// the admitted set, so the channel is a monoculture by construction and
    /// agreement inside it says nothing about whether the conclusion survives
    /// an instrument not derived from the same basin.
    ///
    /// The distinction is measured, not theoretical. This session's
    /// false-witness probe cloned one translation lane and watched naive
    /// cross-witness agreement rise 94 % while the clone sat at similarity
    /// 1.000000 — *n* agreeing witnesses are one witness counted *n* times
    /// unless their independence was established. `Mechanism` IS the
    /// independence partition for tactics (it names the structural capability
    /// the recipe relies on), so a watcher from another mechanism is the
    /// cheapest available orthogonal instrument.
    ///
    /// Returns the objecting watcher's rung AND its mechanism, so the caller
    /// learns *which family* objected. The two channels are never summed into
    /// one dissent number: "a sibling scored this differently" and "an
    /// independent instrument disagrees" demand different responses, and a
    /// merged score is exactly the proxy that hides the second inside the first.
    ///
    /// Suggestion-only, like every periphery channel here: it can force a
    /// wider read, never name the answer.
    pub fn cross_family_dissent(
        style: ThinkingStyle,
        ctx: &PlanContext,
        rung: RungLevel,
        k: usize,
        tol: f32,
    ) -> Option<(RungLevel, Mechanism)> {
        let admitted = Self::reliability_at(style, ctx, rung);
        let want = Self::cluster_mechanism(style.cluster());
        for watcher in rung.peripheral_sample_where(k, |r| r.mechanism != want) {
            let Some(kern) = kernel(watcher.id) else {
                continue;
            };
            let mut tc = Self::thought_ctx_from(ctx);
            for r in Self::recipes_for_at(style, rung) {
                if let Some(k2) = kernel(r.id) {
                    let _ = k2.run(&mut tc);
                }
            }
            let _ = kern.run(&mut tc);
            if (tc.confidence.clamp(0.0, 1.0) - admitted).abs() > tol {
                return Some((watcher.min_rung(), watcher.mechanism));
            }
        }
        None
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
        let reliability = Self::reliability_for(style, &input.context);
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

    /// **The context-honest entry point** — stratified when, and only when, the
    /// context carries witness evidence that EARNED a rung.
    ///
    /// This is the root of the audit gap `E-RUNG-STRATIFIED-WAVE-SHIPPED-1` left
    /// open ("threading the wave into planner context is its own deliverable"):
    /// the planner now runs [`standing_wave_stratified`](lance_graph_contract::witness_fabric::standing_wave_stratified)
    /// itself, via [`WitnessWindow::rung`](crate::traits::WitnessWindow::rung),
    /// instead of trusting a rung it was handed.
    ///
    /// No window — or a window whose wave escalated or was unbound — yields NO
    /// rung, and falls back to the unstratified [`reliability_of`](Self::reliability_of).
    /// **Absence must never be read as [`RungLevel::Surface`]**: rung 0 admits 4
    /// of 34 tactics, so treating "no evidence" as "shallowest evidence" would
    /// silently starve every caller that has never heard of the witness fabric.
    pub fn reliability_for(style: ThinkingStyle, ctx: &PlanContext) -> f32 {
        match ctx.witness.as_ref().and_then(|w| w.rung()) {
            Some(rung) => Self::reliability_at(style, ctx, rung),
            None => Self::reliability_of(style, ctx),
        }
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
    /// **Why the rung is still a parameter, now that `PlanContext` can carry a
    /// window:** a rung read from the context is only honest when the context
    /// HAS witness evidence — that path is [`reliability_for`](Self::reliability_for),
    /// which derives it from the wave. This entry point stays explicit for the
    /// callers that hold a window the `PlanContext` does not (the fabric's own
    /// consumers), and for the peripheral watchdog, which must score at rungs
    /// the context did not earn. The rule that has not changed: a rung is
    /// derived from a wave or passed by someone who ran one — never inferred
    /// from `estimated_complexity`, which would be inventing a semantic.
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
            witness: None,
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

    /// The watchdog must be able to FIRE — a guard that can never trigger is
    /// the same failure as the gate that never gates.
    #[test]
    fn peripheral_dissent_can_fire_and_never_decides() {
        let ctx = ctx_with(Some(style_vec(0.9, 0.0, 0.0)));
        // tol = 0 → any peripheral movement at all counts as dissent. If NO
        // style/rung pair can produce dissent even then, the watchdog is inert.
        let any_fires = ThinkingStyle::ALL.iter().any(|&s| {
            [
                RungLevel::Surface,
                RungLevel::Shallow,
                RungLevel::Contextual,
            ]
            .iter()
            .any(|&r| StyleStrategy::peripheral_dissent(s, &ctx, r, 8, 0.0).is_some())
        });
        assert!(
            any_fires,
            "peripheral watchdog can never fire — inert guard"
        );

        // It NEVER changes the score: dissent is a signal, not a vote.
        for style in ThinkingStyle::ALL {
            let before = StyleStrategy::reliability_at(style, &ctx, RungLevel::Shallow);
            let _ = StyleStrategy::peripheral_dissent(style, &ctx, RungLevel::Shallow, 8, 0.0);
            let after = StyleStrategy::reliability_at(style, &ctx, RungLevel::Shallow);
            assert_eq!(
                before.to_bits(),
                after.to_bits(),
                "{style:?}: watchdog mutated the score it was only meant to observe"
            );
        }
    }

    /// **The eligibility budget is not spent on ineligible watchers.**
    ///
    /// Falsifier for the sample-then-filter shape this replaced: at small `k`
    /// the pre-filtered sample must contain ONLY same-mechanism watchers and
    /// must be non-empty wherever eligible watchers exist. Under the old shape
    /// a global stride at `k = 2` returned watchers of other mechanisms, every
    /// one of which the loop skipped — so the channel observed nothing and
    /// reported agreement.
    #[test]
    fn eligible_watchers_are_not_starved_by_the_sampler() {
        let want = Mechanism::TruthAwareInference;
        for rung in [
            RungLevel::Surface,
            RungLevel::Shallow,
            RungLevel::Contextual,
        ] {
            let eligible_total = rung
                .peripheral_recipes()
                .filter(|r| r.mechanism == want)
                .count();
            for k in [1usize, 2, 3] {
                let picked: Vec<&Recipe> = rung
                    .peripheral_sample_where(k, |r| r.mechanism == want)
                    .collect();
                assert!(
                    picked.iter().all(|r| r.mechanism == want),
                    "{rung:?} k={k}: sampler returned an ineligible watcher"
                );
                assert_eq!(
                    picked.len(),
                    k.min(eligible_total),
                    "{rung:?} k={k}: budget spent on ineligible watchers \
                     ({eligible_total} eligible exist)"
                );
            }
            // The global sampler is genuinely different — otherwise this test
            // would pass against the shape it is meant to reject.
            if eligible_total > 0 && eligible_total < rung.peripheral_recipes().count() {
                let global: Vec<&Recipe> = rung.peripheral_sample(2).collect();
                assert!(
                    global.iter().any(|r| r.mechanism != want),
                    "{rung:?}: global sample is accidentally all-eligible, \
                     this fixture cannot distinguish the two samplers"
                );
            }
        }
    }

    /// **The two dissent channels must be able to DISAGREE.** Same-family
    /// dissent measures calibration; cross-family dissent measures
    /// independence. If no input can make one fire while the other stays
    /// silent, they are one channel wearing two names and the independence
    /// claim is decoration.
    #[test]
    fn cross_family_dissent_is_a_distinct_channel() {
        let ctx = ctx_with(Some(style_vec(0.9, 0.0, 0.0)));
        let rungs = [
            RungLevel::Surface,
            RungLevel::Shallow,
            RungLevel::Contextual,
        ];

        // It can fire at all.
        let cross_fires = ThinkingStyle::ALL.iter().any(|&s| {
            rungs
                .iter()
                .any(|&r| StyleStrategy::cross_family_dissent(s, &ctx, r, 8, 0.0).is_some())
        });
        assert!(cross_fires, "cross-family channel is inert");

        // And it reports a mechanism that is NOT the style's own — that is the
        // whole point of the channel.
        for style in ThinkingStyle::ALL {
            let own = StyleStrategy::cluster_mechanism(style.cluster());
            for rung in rungs {
                if let Some((_, m)) = StyleStrategy::cross_family_dissent(style, &ctx, rung, 8, 0.0)
                {
                    assert_ne!(
                        m, own,
                        "{style:?}: cross-family dissent reported the style's OWN mechanism"
                    );
                }
            }
        }

        // Somewhere in the matrix the two channels differ. A tolerance high
        // enough to silence one but not the other is the discriminating input.
        let differs = ThinkingStyle::ALL.iter().any(|&s| {
            rungs.iter().any(|&r| {
                let same = StyleStrategy::peripheral_dissent(s, &ctx, r, 8, 0.0).is_some();
                let cross = StyleStrategy::cross_family_dissent(s, &ctx, r, 8, 0.0).is_some();
                same != cross
            })
        });
        assert!(
            differs,
            "the two channels never disagree — one of them is decoration"
        );

        // Neither channel touches the score.
        for style in ThinkingStyle::ALL {
            let before = StyleStrategy::reliability_at(style, &ctx, RungLevel::Shallow);
            let _ = StyleStrategy::cross_family_dissent(style, &ctx, RungLevel::Shallow, 8, 0.0);
            let after = StyleStrategy::reliability_at(style, &ctx, RungLevel::Shallow);
            assert_eq!(
                before.to_bits(),
                after.to_bits(),
                "{style:?}: cross-family channel mutated the score"
            );
        }
    }

    /// At the top rung there is no periphery, so there is nothing to dissent —
    /// the guard must be silent rather than fabricate an elevation.
    #[test]
    fn no_periphery_no_dissent() {
        let ctx = ctx_with(Some(style_vec(0.9, 0.0, 0.0)));
        for style in ThinkingStyle::ALL {
            assert!(
                StyleStrategy::peripheral_dissent(style, &ctx, RungLevel::Transcendent, 8, 0.0)
                    .is_none(),
                "{style:?}: dissent reported where nothing is excluded"
            );
        }
    }

    /// A dissent must point UP — elevating to a rung that would not actually
    /// admit the dissenting watcher would be theatre.
    #[test]
    fn dissent_elevates_to_a_rung_that_admits_the_dissenter() {
        let ctx = ctx_with(Some(style_vec(0.9, 0.0, 0.0)));
        for style in ThinkingStyle::ALL {
            for rung in [
                RungLevel::Surface,
                RungLevel::Shallow,
                RungLevel::Contextual,
            ] {
                if let Some(up) = StyleStrategy::peripheral_dissent(style, &ctx, rung, 8, 0.0) {
                    assert!(
                        (up as u8) > (rung as u8),
                        "{style:?}: dissent at {rung:?} elevated to {up:?} (not deeper)"
                    );
                }
            }
        }
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

    // ── the wave threaded into planner context (task #29) ────────────────────

    use crate::traits::WitnessWindow;
    use lance_graph_contract::causal_witness::{CausalWitnessFacet, Locus};

    fn facet(edges: &[(Locus, i8)]) -> CausalWitnessFacet {
        let mut f = CausalWitnessFacet::ZERO;
        for &(l, o) in edges {
            f = f.with(l, o);
        }
        f
    }

    fn window(rows: Vec<(usize, CausalWitnessFacet)>, passes: u8) -> WitnessWindow {
        WitnessWindow {
            rows,
            focal_idx: 0,
            locus: Locus::Antecedent,
            passes,
        }
    }

    /// A single-hop chain that terminates: the cheapest ground the wave can
    /// observe (settles at pass 2 — two successive budgets agree).
    fn cheap_ground_window() -> WitnessWindow {
        window(
            vec![
                (0, facet(&[(Locus::Antecedent, 1)])),
                (1, CausalWitnessFacet::ZERO),
            ],
            8,
        )
    }

    /// A chain that leaves the `±8` reference horizon — the hard case.
    fn escalating_window() -> WitnessWindow {
        window(vec![(0, facet(&[(Locus::Antecedent, 7)]))], 8)
    }

    /// The three contexts `resolve_style` can actually produce, so a claim
    /// about "every style" is a claim about every reachable one.
    fn reachable_ctxs() -> Vec<PlanContext> {
        vec![
            ctx_with(Some(style_vec(0.9, 0.0, 0.0))),
            ctx_with(Some(style_vec(0.0, 0.9, 0.0))),
            ctx_with(Some(style_vec(0.0, 0.0, 0.9))),
        ]
    }

    /// **The non-breaking guarantee.** A context without witness evidence must
    /// score EXACTLY as it did before the window existed — bit-identical, not
    /// approximately.
    #[test]
    fn no_witness_window_is_bit_identical_to_the_unstratified_path() {
        for ctx in reachable_ctxs() {
            assert!(ctx.witness.is_none());
            let style = StyleStrategy::resolve_style(&ctx);
            assert_eq!(
                StyleStrategy::reliability_for(style, &ctx).to_bits(),
                StyleStrategy::reliability_of(style, &ctx).to_bits(),
                "{style:?}: no-window path drifted from the unstratified score"
            );
            // …including through `plan()`, the surface a consumer actually reads.
            let out = StyleStrategy
                .plan(ctx_input(ctx.clone()), &mut Arena::new())
                .expect("plan must not error");
            assert_eq!(
                out.outcome.unwrap().reliability.to_bits(),
                StyleStrategy::reliability_of(style, &ctx).to_bits()
            );
        }
    }

    /// **Absent ≠ zero.** No window means NO rung — never
    /// [`RungLevel::Surface`], which admits 4 of 34 tactics and would silently
    /// starve every caller that has no witness evidence.
    #[test]
    fn absent_window_is_not_read_as_rung_surface() {
        // The claim is only falsifiable where Surface and the unstratified set
        // actually score differently; assert such a style exists, then assert
        // the absent path lands on the unstratified side of that difference.
        let mut discriminating = 0;
        for ctx in reachable_ctxs() {
            let style = StyleStrategy::resolve_style(&ctx);
            let surface = StyleStrategy::reliability_at(style, &ctx, RungLevel::Surface);
            let absent = StyleStrategy::reliability_for(style, &ctx);
            if surface.to_bits() != StyleStrategy::reliability_of(style, &ctx).to_bits() {
                discriminating += 1;
                assert_ne!(
                    absent.to_bits(),
                    surface.to_bits(),
                    "{style:?}: a context with NO witness was scored as rung 0"
                );
            }
        }
        assert!(
            discriminating > 0,
            "no style distinguishes Surface from the full set — the test cannot falsify"
        );
    }

    /// The rung is DERIVED from the wave, not asserted: a window that grounds
    /// cheaply is normalized to a shallow rung, and that changes the measured
    /// reliability — not merely a recipe count.
    #[test]
    fn witness_window_derives_the_rung_from_the_wave() {
        let cheap = cheap_ground_window();
        assert_eq!(
            cheap.rung(),
            Some(RungLevel::Shallow),
            "a single-hop terminal chain settles at pass 2 → Shallow"
        );

        let mut moved = 0;
        for mut ctx in reachable_ctxs() {
            let style = StyleStrategy::resolve_style(&ctx);
            let unstratified = StyleStrategy::reliability_of(style, &ctx);
            ctx.witness = Some(cheap_ground_window());
            let stratified = StyleStrategy::reliability_for(style, &ctx);
            // The score is exactly the one the derived rung dictates.
            assert_eq!(
                stratified.to_bits(),
                StyleStrategy::reliability_at(style, &ctx, RungLevel::Shallow).to_bits(),
                "{style:?}: context rung did not come from the wave"
            );
            if stratified.to_bits() != unstratified.to_bits() {
                moved += 1;
            }
        }
        assert!(
            moved > 0,
            "threading the wave changed no OUTCOME for any style — inert wiring"
        );
    }

    /// A window whose chain escalates earns NO rung, so the caller falls back to
    /// the full set — never to the shallow rung its early settle pass would
    /// suggest. Starving the hardest case of the deepest tactics is exactly the
    /// blindness `E-PERIPHERAL-DISSENT-GUARDS-THE-STRATIFICATION-1` names.
    #[test]
    fn escalating_and_unbound_windows_earn_no_rung_and_never_starve() {
        assert_eq!(escalating_window().rung(), None, "escalation earns no rung");
        assert_eq!(
            window(vec![(0, CausalWitnessFacet::ZERO)], 8).rung(),
            None,
            "an unbound locus earns no rung"
        );

        for mut ctx in reachable_ctxs() {
            let style = StyleStrategy::resolve_style(&ctx);
            let unstratified = StyleStrategy::reliability_of(style, &ctx);
            ctx.witness = Some(escalating_window());
            assert_eq!(
                StyleStrategy::reliability_for(style, &ctx).to_bits(),
                unstratified.to_bits(),
                "{style:?}: an escalating window was scored at a rung it never earned"
            );
            // …and the cheap ground really is the STRICTER of the two: fewer
            // tactics admitted than the escalating (unstratified) fallback.
            let shallow = StyleStrategy::recipes_for_at(style, RungLevel::Shallow).count();
            let full = StyleStrategy::recipes_for(style).count();
            assert!(
                shallow <= full,
                "{style:?}: cheap ground admitted more than the fallback"
            );
        }
        // At least one style must show the inequality strictly, or "shallower
        // than escalate" is a claim about nothing.
        assert!(
            ThinkingStyle::ALL.iter().any(|&s| {
                StyleStrategy::recipes_for_at(s, RungLevel::Shallow).count()
                    < StyleStrategy::recipes_for(s).count()
            }),
            "cheap grounding never restricts anything relative to escalation"
        );
    }

    /// End-to-end: the wave reaches the D-MBX-A6 carrier `plan()` surfaces.
    #[test]
    fn plan_surfaces_the_wave_derived_reliability() {
        let mut ctx = ctx_with(Some(style_vec(0.9, 0.0, 0.0)));
        ctx.witness = Some(cheap_ground_window());
        let style = StyleStrategy::resolve_style(&ctx);
        let out = StyleStrategy
            .plan(ctx_input(ctx.clone()), &mut Arena::new())
            .expect("plan must not error");
        assert_eq!(
            out.outcome.unwrap().reliability.to_bits(),
            StyleStrategy::reliability_at(style, &ctx, RungLevel::Shallow).to_bits(),
            "plan() ignored the context's witness window"
        );
    }
}
