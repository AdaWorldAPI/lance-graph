//! The consumer reasoning facade — **concept-blind by construction**.
//!
//! Consumers (medcare-rs and any other) supply *opaque* interned `u16` concept
//! ids plus `(frequency, confidence)` premises; this module supplies the
//! deduction matrix and never learns what an id means. That asymmetry is the
//! point, and it is a hard fence, not a preference: `lance-graph` is public,
//! so a domain vocabulary living here would be a disclosure surface as well as
//! a reusability loss. **Do not add LOINC codes, disease tables, panel
//! specifics, or any other domain catalogue to this module.** The consumer owns
//! the meaning, privately; this crate owns the mechanism.
//!
//! # Why a curated facade rather than `pub use lance_graph_planner`
//!
//! A blanket re-export would hand every consumer the whole strategy surface
//! (16 planner strategies, MUL, elevation) to reach four reasoning types. This
//! module names exactly the clinical-reasoning entry points, so the consumer's
//! dependency tier stays narrow and auditable. Consumers must NOT path-dep
//! `lance-graph-planner` directly — that breaks the BBB tiering every consumer
//! repo in this workspace enforces.
//!
//! # The shape of an inference here
//!
//! ```text
//!   premises (one per independent evidence AXIS)
//!        │  each stamped with its own disjoint Stamp
//!        ▼
//!   BeliefArena  ── revision ──►  accumulated truth   (confidence RISES)
//!        │
//!        ├── resolve()      → the best-supported statement
//!        └── differential() → ranked rivals + what is MISSING to separate them
//! ```
//!
//! Guard rules ([`detect_violations`]) deliberately do **not** enter that
//! pipeline — see their own documentation.

pub use lance_graph_planner::nars::belief::{
    Belief, BeliefArena, CStmt, Copula, ReviseOutcome, Stamp,
};
pub use lance_graph_planner::nars::tactics::{
    asc_challenge, cas_abstract, cr_synthesize, rcr_abduce, tr_diverge, AscOutcome, Candidate,
    Frontier, GapKind, ReasoningGap, Tactic, Throttle,
};
pub use lance_graph_planner::nars::truth::TruthValue;

pub use lance_graph_cognitive::world::counterfactual::{
    multi_substitute_binding, substitute_binding, worlds_differ, BindingSubstitution,
    Counterfactual, SubstitutedWorld,
};

/// The evidential-stamp horizon: [`Stamp`] is a 64-bit source bitset.
///
/// Load-bearing, not decorative. `Stamp::source(id)` folds `id % 64`, so axis
/// 0 and axis 64 would produce the *same* bit — and two premises that look
/// independent would then be judged overlapping and resolved by CHOICE instead
/// of pooled by revision. The failure is silent: confidence simply stops
/// rising. [`Axis::new`] refuses out-of-range indices so the fold can never
/// happen through this facade.
pub const MAX_AXES: u8 = 64;

/// One **independent evidence source**, by index — the consumer names it.
///
/// This type exists to make the workspace's most expensive NARS trap
/// *structurally* unreachable. `BeliefArena::observe` pools evidence
/// ([`TruthValue::revise`], confidence rises) only when the incoming stamp is
/// non-empty AND disjoint from the belief's existing sources; otherwise it
/// falls back to CHOICE, which keeps the better-supported truth and moves
/// confidence **not at all**. Both paths return an `Ok`-shaped
/// [`ReviseOutcome`] and neither logs anything, so passing one stamp for four
/// genuinely independent axes produces a plausible answer carrying the
/// confidence of a single axis.
///
/// Taking an `Axis` rather than a `Stamp` means a consumer cannot express that
/// mistake: distinct axes yield distinct bits by construction.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Axis(u8);

impl Axis {
    /// An axis by index, or `None` if `index >= MAX_AXES` (see [`MAX_AXES`] for
    /// why an out-of-range index must be refused rather than folded).
    #[must_use]
    pub const fn new(index: u8) -> Option<Self> {
        if index < MAX_AXES {
            Some(Axis(index))
        } else {
            None
        }
    }

    /// The index this axis was built from.
    #[must_use]
    pub const fn index(self) -> u8 {
        self.0
    }

    /// The evidential stamp for this axis — one distinct bit per axis.
    #[must_use]
    pub fn stamp(self) -> Stamp {
        Stamp::source(u32::from(self.0))
    }
}

/// A patient's (or any subject's) premises, each tagged with the axis that
/// observed it.
///
/// The bundle owns stamp assignment precisely so the consumer never touches
/// [`Stamp`] directly — see [`Axis`].
#[derive(Debug, Clone, Default)]
pub struct PremiseBundle {
    premises: Vec<(CStmt, TruthValue, Axis)>,
}

impl PremiseBundle {
    /// An empty bundle.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Record one premise as observed by `axis`.
    pub fn observe(&mut self, stmt: CStmt, truth: TruthValue, axis: Axis) -> &mut Self {
        self.premises.push((stmt, truth, axis));
        self
    }

    /// The premises, in insertion order.
    #[must_use]
    pub fn premises(&self) -> &[(CStmt, TruthValue, Axis)] {
        &self.premises
    }

    /// How many DISTINCT axes contributed — the honest "how independent is
    /// this?" number. One axis observing four times is still one axis, and the
    /// arena will have pooled none of it.
    #[must_use]
    pub fn distinct_axes(&self) -> usize {
        let mut seen = 0u64;
        for (_, _, axis) in &self.premises {
            seen |= 1u64 << axis.index();
        }
        seen.count_ones() as usize
    }

    /// Fold the bundle into a [`BeliefArena`], stamping each premise with its
    /// own axis. Premises on the same statement from DIFFERENT axes pool
    /// (confidence rises); from the SAME axis they compete by CHOICE.
    #[must_use]
    pub fn arena(&self) -> BeliefArena {
        let mut arena = BeliefArena::new();
        for (stmt, truth, axis) in &self.premises {
            arena.observe(*stmt, *truth, axis.stamp());
        }
        arena
    }
}

/// The best-supported statement in a bundle, with the evidence that backs it.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Resolution {
    /// The statement.
    pub stmt: CStmt,
    /// Its accumulated truth after pooling across axes.
    pub truth: TruthValue,
    /// Preserved dialectic depth — `max |f₁−f₂|` across the revisions that
    /// produced it. **Non-zero means the axes disagreed**, even though they
    /// pooled to a confident answer; this is the cross-modal disagreement flag,
    /// and it is committed rather than erased.
    pub contradiction: f32,
    /// How many distinct axes contributed to the bundle overall.
    pub axes: usize,
}

/// Resolve a bundle to its best-supported statement, ranked by NARS
/// `expectation()` (which weighs frequency BY confidence, so a confident
/// moderate belief outranks a near-certain guess).
///
/// Returns `None` for an empty bundle — an absent answer, never a fabricated
/// low-confidence one.
#[must_use]
pub fn resolve(bundle: &PremiseBundle) -> Option<Resolution> {
    let arena = bundle.arena();
    let axes = bundle.distinct_axes();
    arena
        .entries()
        .iter()
        .enumerate()
        // `max_by` keeps the LAST maximum; reverse the index so ties resolve to
        // the earliest-observed belief and the result is insertion-deterministic.
        .max_by(|(ia, a), (ib, b)| {
            a.truth
                .expectation()
                .total_cmp(&b.truth.expectation())
                .then(ib.cmp(ia))
        })
        .map(|(_, b)| Resolution {
            stmt: b.stmt,
            truth: b.truth,
            contradiction: b.contradiction,
            axes,
        })
}

/// The ranked differential: rival explanations plus **what is missing to
/// separate them**.
///
/// The [`Frontier::gaps`] half is the clinically honest part and the reason
/// this returns a `Frontier` rather than a bare `Vec<Candidate>`: when two
/// rivals cannot be separated, a [`ReasoningGap`] names the absent premise
/// instead of forcing a ranking on evidence that does not support one.
/// Surface the gaps — do not discard them.
#[must_use]
pub fn differential(bundle: &PremiseBundle, throttle: &Throttle) -> Frontier {
    rcr_abduce(&bundle.arena(), throttle)
}

/// Comparison operator for a [`GuardRule`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Cmp {
    /// `observed < threshold`
    Lt,
    /// `observed <= threshold`
    Le,
    /// `observed > threshold`
    Gt,
    /// `observed >= threshold`
    Ge,
}

impl Cmp {
    #[must_use]
    fn holds(self, observed: f32, threshold: f32) -> bool {
        match self {
            Cmp::Lt => observed < threshold,
            Cmp::Le => observed <= threshold,
            Cmp::Gt => observed > threshold,
            Cmp::Ge => observed >= threshold,
        }
    }
}

/// A contradiction between something PRESENT and something OBSERVED — a rule
/// over two stored values.
///
/// Concept ids are opaque, exactly as elsewhere in this module.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GuardRule {
    /// The concept whose presence arms the rule.
    pub when_present: u16,
    /// The observable whose value is compared.
    pub observable: u16,
    /// How the observed value is compared to `threshold`.
    pub cmp: Cmp,
    /// The threshold.
    pub threshold: f32,
}

/// A fired [`GuardRule`], carrying **both operands** so the caller can report
/// what contradicts what without re-deriving it.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GuardViolation {
    /// The rule that fired.
    pub rule: GuardRule,
    /// The value that tripped it.
    pub observed: f32,
}

/// Detect guard violations — **deliberately NOT an inference**.
///
/// This returns no [`TruthValue`] and creates no [`Belief`], and that is a
/// correctness requirement rather than a stylistic one. Routing a stored-value
/// contradiction through the arena would mint a *derived belief with a
/// confidence* about a *fact*; being a belief, it would then be **revisable**,
/// so later evidence from another axis could pool against it and soften it.
/// A contradiction between two recorded values must never soften — it is a
/// lookup, and its truth is not a matter of degree.
///
/// Deterministic: rules are evaluated in the order given, observations scanned
/// in order, so the output order is a pure function of the inputs.
#[must_use]
pub fn detect_violations(
    rules: &[GuardRule],
    present: &[u16],
    observations: &[(u16, f32)],
) -> Vec<GuardViolation> {
    let mut out = Vec::new();
    for rule in rules {
        if !present.contains(&rule.when_present) {
            continue;
        }
        for &(concept, observed) in observations {
            if concept == rule.observable && rule.cmp.holds(observed, rule.threshold) {
                out.push(GuardViolation {
                    rule: *rule,
                    observed,
                });
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn axis(i: u8) -> Axis {
        Axis::new(i).expect("axis index within MAX_AXES")
    }

    fn stmt(s: u16, p: u16) -> CStmt {
        CStmt {
            s,
            cop: Copula::Inh,
            p,
        }
    }

    /// **THE gating falsifier** (medcare seam, both halves required).
    ///
    /// Four independent axes agreeing must POOL — confidence strictly rises
    /// above any single axis. The same four observations on ONE axis must NOT
    /// pool — they compete by CHOICE and confidence does not move.
    ///
    /// Both halves matter. The rise alone would pass even if the arena pooled
    /// indiscriminately (which would double-count correlated evidence); the
    /// no-rise alone would pass even if the arena never pooled at all.
    #[test]
    fn four_disjoint_axes_pool_but_one_axis_repeated_does_not() {
        let hypothesis = stmt(1, 2);
        let single = TruthValue::new(0.8, 0.5);

        let mut four = PremiseBundle::new();
        for i in 0..4u8 {
            four.observe(hypothesis, single, axis(i));
        }
        let pooled = resolve(&four).expect("non-empty bundle resolves");

        let mut one = PremiseBundle::new();
        for _ in 0..4 {
            one.observe(hypothesis, single, axis(0));
        }
        let chosen = resolve(&one).expect("non-empty bundle resolves");

        assert_eq!(four.distinct_axes(), 4, "four axes must read as four");
        assert_eq!(
            one.distinct_axes(),
            1,
            "one axis repeated is still one axis"
        );

        assert!(
            pooled.truth.confidence > single.confidence,
            "four DISJOINT axes must pool: {} should exceed the single-axis {}",
            pooled.truth.confidence,
            single.confidence
        );
        assert!(
            (chosen.truth.confidence - single.confidence).abs() < 1e-6,
            "one axis repeated must NOT pool (CHOICE, not revision): got {}, expected {}",
            chosen.truth.confidence,
            single.confidence
        );
        assert!(
            pooled.truth.confidence > chosen.truth.confidence,
            "the whole point: independence must be worth something"
        );
    }

    /// Disagreement across axes is preserved, not averaged away — the
    /// cross-modal disagreement flag.
    #[test]
    fn disagreeing_axes_pool_but_record_the_contradiction() {
        let hypothesis = stmt(1, 2);
        let mut b = PremiseBundle::new();
        b.observe(hypothesis, TruthValue::new(0.9, 0.5), axis(0));
        b.observe(hypothesis, TruthValue::new(0.1, 0.5), axis(1));
        let r = resolve(&b).expect("resolves");
        assert!(
            r.contradiction > 0.5,
            "a 0.9-vs-0.1 split must survive as committed contradiction depth, got {}",
            r.contradiction
        );

        let mut agree = PremiseBundle::new();
        agree.observe(hypothesis, TruthValue::new(0.9, 0.5), axis(0));
        agree.observe(hypothesis, TruthValue::new(0.9, 0.5), axis(1));
        assert!(
            resolve(&agree).expect("resolves").contradiction < 1e-6,
            "agreeing axes must record NO contradiction — else the flag is decoration"
        );
    }

    /// The stamp horizon is refused, not folded. Without this, axis 64 would
    /// alias axis 0 and two independent sources would silently stop pooling.
    #[test]
    fn axis_beyond_the_stamp_horizon_is_refused() {
        assert!(Axis::new(MAX_AXES - 1).is_some());
        assert!(
            Axis::new(MAX_AXES).is_none(),
            "an index that would fold onto axis 0 must be refused"
        );
        assert_ne!(
            axis(0).stamp(),
            axis(1).stamp(),
            "distinct axes must carry distinct evidence bits"
        );
    }

    /// An empty bundle resolves to `None` — no fabricated low-confidence answer.
    #[test]
    fn empty_bundle_resolves_to_nothing() {
        assert!(resolve(&PremiseBundle::new()).is_none());
    }

    /// Guard rules fire on the armed case and stay silent otherwise — and the
    /// silence must come from the RULE, not from an empty input.
    #[test]
    fn guard_rule_fires_armed_and_stays_silent_unarmed() {
        const DRUG: u16 = 10;
        const OTHER_DRUG: u16 = 11;
        const MARKER: u16 = 20;
        let rule = GuardRule {
            when_present: DRUG,
            observable: MARKER,
            cmp: Cmp::Lt,
            threshold: 45.0,
        };
        let rules = [rule];

        let fired = detect_violations(&rules, &[DRUG], &[(MARKER, 30.0)]);
        assert_eq!(fired.len(), 1, "armed + tripped must fire");
        assert_eq!(fired[0].observed, 30.0, "the operand must come back");
        assert_eq!(fired[0].rule.threshold, 45.0, "…and so must the rule");

        // Silent because the value is fine — NOT because the input was empty.
        assert!(
            detect_violations(&rules, &[DRUG], &[(MARKER, 60.0)]).is_empty(),
            "armed but within threshold must not fire"
        );
        // Silent because the rule is unarmed, on the SAME tripping value.
        assert!(
            detect_violations(&rules, &[OTHER_DRUG], &[(MARKER, 30.0)]).is_empty(),
            "unarmed must not fire even on a tripping value"
        );
    }

    /// A guard violation carries no truth value and mints no belief — the
    /// category fence, asserted rather than merely documented.
    #[test]
    fn guard_violations_never_enter_the_arena() {
        const DRUG: u16 = 10;
        const MARKER: u16 = 20;
        let rules = [GuardRule {
            when_present: DRUG,
            observable: MARKER,
            cmp: Cmp::Lt,
            threshold: 45.0,
        }];
        let violations = detect_violations(&rules, &[DRUG], &[(MARKER, 30.0)]);
        assert_eq!(violations.len(), 1);

        // The bundle that reasons about this patient is untouched by the
        // violation: nothing to observe, nothing to revise, nothing to soften.
        let bundle = PremiseBundle::new();
        assert!(
            resolve(&bundle).is_none(),
            "a contraindication must not have produced a belief"
        );
    }
}
