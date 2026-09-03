//! `insight` — the S10 insight-vs-mush detector over a `BeliefArena` reasoning
//! step (`.claude/plans/dialectic-engine-v1.md` §1 S10, D-DIA-V2-A).
//!
//! A reasoning STEP is a before→after pair of arena snapshots (e.g. bracketing a
//! `close_transitive` or a tactic-admit round). The detector scores whether the
//! step produced **insight** (free-energy descent — coherence and wonder rise
//! together while entropy falls, gated on a real derivation yield) or **mush**
//! (churn + stall — revision thrash with no yield, or entropy with no coherence
//! change). It **reuses the contract types** ([`GraphSignals`] for the signal
//! carrier, [`FlowState`] for the flow classification) — nothing invented.
//!
//! **This is a registered convention, NOT a proven detector.** Per the S10
//! discipline (`E-BASIN-WIDTH`), "detects insight" is only promotable once the
//! discriminator BEATS a size-preserving null — the shuffle control in the
//! [`insight_beats_size_preserving_null`](tests) test. If a coherent arena does
//! not out-score a size-matched random rewiring, the honest finding is that S10
//! measures SIZE, not insight (the D-SRS-3b "composite = size" collapse shape).

use super::belief::BeliefArena;
use crate::temporal::QueryReference;
use lance_graph_contract::mul::FlowState;
use lance_graph_contract::sensorium::GraphSignals;
use lance_graph_contract::thought_atoms::normalized_entropy;

/// A snapshot of the arena's cognitive signals at one instant — the contract
/// [`GraphSignals`] (reused) plus the two scalars S10 needs that it does not
/// carry: `coherence` (how much the graph has closed into strong conclusions)
/// and `wonder` (committed contradiction depth, the Staunen/novelty pole).
#[derive(Debug, Clone, Copy)]
pub struct Snapshot {
    /// The reused contract signal carrier (entropy, yield, revision-velocity, …).
    pub signals: GraphSignals,
    /// Coherence: **closure density** = `derived / total`. A step that closes
    /// the graph into many transitive conclusions scores high; before any
    /// derivation it is 0.
    ///
    /// **Why NOT `·mean_exp` (the null-falsifier correction, `E-S10-COHERENCE-
    /// CLOSURE-DENSITY-1`):** the first draft multiplied by mean expectation
    /// over the derived set. But NAL deduction *correctly* attenuates confidence
    /// with chain depth, so the deepest, most-coherent chains earn the LOWEST
    /// mean-expectation — the multiplier dragged coherence DOWN exactly when the
    /// graph closed MOST. The `insight_beats_size_preserving_null` falsifier
    /// scored real=null=0 under that proxy. Closure density is size-invariant
    /// (a ratio) and discriminates chain-structure (dense closure) from random
    /// rewiring (sparse closure) — which is what S10 must measure.
    pub coherence: f32,
    /// Wonder: mean committed contradiction depth (`|f₁−f₂|` preserved across
    /// revisions) — the Staunen pole.
    pub wonder: f32,
}

/// A [`Snapshot`] stamped with the epistemic view it was read under — an owned,
/// movable record of "these signals, as of this reader's reference".
///
/// **Owned identity, never a borrow.** An earlier draft held `arena: &BeliefArena`
/// alongside the snapshot; that couples a historical reading to a live mutable
/// arena (so the two can silently disagree), makes the record unmovable and
/// unpersistable, and leaves it ambiguous whether the field means *identity* or
/// *current content*. The arena is re-read separately through `arena_id` + the
/// reference when needed.
///
/// **`at` is a [`QueryReference`], not a bare version.** A version answers
/// "which dataset revision"; a `QueryReference` answers "which revision was this
/// observer permitted to see, under which epistemic mode, at which rung" — the
/// `Strict`/`Retro` distinction is the whole point of the temporal layer, and a
/// settlement or insight reading taken under `Retro` is not the same
/// observation as one taken under `Strict`.
///
/// No `branch_id` field: `QueryReference::server_id` already names the version
/// line, and duplicating it is how two carriers drift apart.
#[derive(Debug, Clone, Copy)]
pub struct VersionedSnapshot {
    /// The epistemic view this reading was taken under.
    pub at: QueryReference,
    /// Which arena was read.
    pub arena_id: u32,
    /// The signals themselves.
    pub snapshot: Snapshot,
}

impl VersionedSnapshot {
    /// Stamp a snapshot with the view it was read under.
    #[must_use]
    pub fn new(at: QueryReference, arena_id: u32, snapshot: Snapshot) -> Self {
        Self {
            at,
            arena_id,
            snapshot,
        }
    }

    /// Read `arena` and stamp the result in one step.
    #[must_use]
    pub fn of(
        arena: &BeliefArena,
        revision_velocity: f32,
        at: QueryReference,
        arena_id: u32,
    ) -> Self {
        Self::new(at, arena_id, Snapshot::of(arena, revision_velocity))
    }

    /// May these two readings be compared as a before→after step?
    ///
    /// Same arena, same epistemic MODE, same rung — differing only in version,
    /// which is what a step IS. Comparing a `Strict` reading against a `Retro`
    /// one measures the mode change, not the reasoning.
    #[must_use]
    pub fn steppable_to(&self, later: &Self) -> bool {
        self.arena_id == later.arena_id
            && self.at.mode == later.at.mode
            && self.at.rung == later.at.rung
            && self.at.server_id == later.at.server_id
    }
}

impl Snapshot {
    /// Read the arena into a snapshot. `revision_velocity` (revisions this step ÷
    /// steps) is supplied by the caller — it is a rate over the step, not a
    /// static arena property.
    #[must_use]
    pub fn of(arena: &BeliefArena, revision_velocity: f32) -> Self {
        Self {
            signals: arena_graph_signals(arena, revision_velocity),
            coherence: coherence(arena),
            wonder: wonder(arena),
        }
    }
}

/// Populate the contract [`GraphSignals`] from an arena read (reuse, not invent).
/// `truth_entropy` = normalized Shannon entropy of the confidence distribution;
/// `deduction_yield` = derived ÷ total; `contradiction_rate` = fraction with a
/// committed contradiction; `revision_velocity` is passed through.
#[must_use]
pub fn arena_graph_signals(arena: &BeliefArena, revision_velocity: f32) -> GraphSignals {
    let n = arena.entries().len();
    let (derived, contradicted) = arena.entries().iter().fold((0usize, 0usize), |(d, c), b| {
        (
            d + usize::from(b.rung >= 1),
            c + usize::from(b.contradiction > 0.0),
        )
    });
    GraphSignals {
        contradiction_rate: ratio(contradicted, n),
        truth_entropy: confidence_entropy(arena),
        revision_velocity,
        plasticity_flux: 0.0,
        deduction_yield: ratio(derived, n),
        episodic_saturation: 0.0,
    }
}

/// **Closure density** = `derived / total` (0 if none derived). Size-invariant:
/// coherent chains close densely (many transitive conclusions per observed
/// edge); random rewiring closes sparsely. See [`Snapshot::coherence`] for why
/// the `·mean_exp` multiplier was dropped (the null-falsifier confound).
#[must_use]
fn coherence(arena: &BeliefArena) -> f32 {
    let n = arena.entries().len();
    if n == 0 {
        return 0.0;
    }
    let derived = arena.entries().iter().filter(|b| b.rung >= 1).count();
    derived as f32 / n as f32
}

/// Mean committed contradiction depth (the Staunen/novelty pole).
#[must_use]
fn wonder(arena: &BeliefArena) -> f32 {
    let n = arena.entries().len();
    if n == 0 {
        return 0.0;
    }
    arena.entries().iter().map(|b| b.contradiction).sum::<f32>() / n as f32
}

/// Normalized Shannon entropy of the confidence distribution over 10 bins
/// (`[0, 1]`; 0 = a single peaked confidence, 1 = uniform spread).
///
/// Routed through the operator-ruled atom
/// [`lance_graph_contract::thought_atoms::normalized_entropy`] rather than
/// carrying its own Shannon loop. The census
/// (`examples/entropy_surface_census.rs`, PROBE-ENTROPY-SURFACE-CENSUS-1)
/// measured this substitution safe: the log base is inert under
/// normalization (`log2/log2(10)` and `ln/ln(10)` agreed to `0.00000000` on
/// every fixture), and the histogram counts are unnormalized weights, which
/// the atom divides by their own sum.
///
/// **The `n == 0` early return is load-bearing and must not be removed.**
/// An empty arena builds an all-zero histogram, and the atom's zero-mass
/// convention is `Some(1.0)` ("nothing prefers anything — indistinguishable
/// from uniform"). Dropping the guard therefore inverts an empty arena's
/// entropy from 0.0 to 1.0 — silently, since both are in range. That is the
/// one fixture where the census measured the two conventions OPPOSITE, and
/// it is pinned by `an_empty_arena_has_zero_truth_entropy_not_one`.
#[must_use]
fn confidence_entropy(arena: &BeliefArena) -> f32 {
    const BINS: usize = 10;
    if arena.entries().is_empty() {
        return 0.0;
    }
    // Counts stay INTEGRAL and are widened once, at the boundary. Accumulating
    // into `f32` instead loses counts above 2^24: `f32(2^24) + 1.0 == f32(2^24)`
    // (measured), so two bins holding 20M and 40M beliefs would both read
    // 16_777_216.0 and the atom would see them as equally populated. Nothing in
    // `BeliefArena` bounds its entry count, so this is unbounded in principle
    // and merely unobserved today — which is not a reason to accumulate in f32.
    let mut hist = [0usize; BINS];
    for b in arena.entries() {
        let idx = ((b.truth.confidence.clamp(0.0, 1.0) * BINS as f32) as usize).min(BINS - 1);
        hist[idx] += 1;
    }
    let weights: [f32; BINS] = core::array::from_fn(|i| hist[i] as f32);
    // BINS is a non-zero constant > 1, so the atom's empty/single arms are
    // unreachable here; the default is defensive, never taken.
    normalized_entropy(&weights).unwrap_or(0.0)
}

#[must_use]
fn ratio(num: usize, den: usize) -> f32 {
    if den == 0 {
        0.0
    } else {
        num as f32 / den as f32
    }
}

/// The S10 score of one reasoning step.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct InsightMush {
    /// `clamp(Δcoh + Δwonder, 0, 1) · [yield > θ]` — coherent closure with the
    /// committed-contradiction (wonder) pole. **The `−Δent` term the plan's
    /// draft carried is REMOVED** — see [`detect`] for the null-falsifier
    /// correction (`E-S10-COHERENCE-CLOSURE-DENSITY-1`): confidence-spread
    /// entropy RISES on every productive term-logic step, so subtracting it
    /// double-penalized the reasoning insight is meant to reward. Entropy's
    /// correct role is the mush `stall` term below (entropy WITHOUT coherence
    /// change = mush).
    pub insight: f32,
    /// `0.5·churn + 0.5·stall` — `churn = revision_velocity·(1−yield)`,
    /// `stall = entropy·(1−|Δcoh|)`. Entropy paired WITH a coherence change is
    /// attenuated (low stall); entropy WITHOUT coherence change is the stall
    /// pole of mush — the correct, asymmetric home for `truth_entropy`.
    pub mush: f32,
}

/// Score a reasoning step (S10). The `yield_theta` gate keeps insight from firing
/// on a step that produced no real derivation.
///
/// **Null-falsifier correction (`E-S10-COHERENCE-CLOSURE-DENSITY-1`).** The
/// plan's first-draft insight was `clamp(Δcoh + Δwonder − Δent, 0, 1)·gate`. The
/// `insight_beats_size_preserving_null` probe scored real=null=0 under it, for
/// two reasons the falsifier isolated:
///   1. coherence was `(derived/total)·mean_exp`, but NAL deduction attenuates
///      expectation with chain depth → the DEEPEST coherent chains earned the
///      LOWEST coherence. Fixed: coherence = closure density (`derived/total`).
///   2. `−Δent` (confidence-spread Shannon) subtracts a large positive on every
///      productive step (deduction spreads confidence across bins) — a term-logic
///      confound, NOT the free-energy surprise-descent it modeled in the
///      VSA-codebook world (where likelihood = cosine-vs-codebook). REMOVED from
///      insight; entropy stays in the mush `stall` term where its sign is correct.
///
/// The surviving discriminator is closure density: coherent chains close densely
/// (0.82 here), random rewiring sparsely — the size-invariant structural signal.
#[must_use]
pub fn detect(before: &Snapshot, after: &Snapshot, yield_theta: f32) -> InsightMush {
    let d_coh = after.coherence - before.coherence;
    let d_wonder = after.wonder - before.wonder;
    let gate = f32::from(after.signals.deduction_yield > yield_theta);
    let insight = (d_coh + d_wonder).clamp(0.0, 1.0) * gate;

    let churn = after.signals.revision_velocity * (1.0 - after.signals.deduction_yield);
    let stall = after.signals.truth_entropy * (1.0 - d_coh.abs());
    let mush = 0.5 * churn + 0.5 * stall;
    InsightMush { insight, mush }
}

/// Classify a step into the contract [`FlowState`] (reused) from its S10 score.
#[must_use]
pub fn flow_state(im: &InsightMush) -> FlowState {
    if im.insight > 0.5 && im.mush < 0.3 {
        FlowState::Flow
    } else if im.mush > 0.5 {
        FlowState::Anxiety
    } else if im.insight < 0.1 && im.mush < 0.2 {
        FlowState::Boredom
    } else {
        FlowState::Transition
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::nars::{CStmt, Copula, Stamp, TruthValue};

    /// A before→after step is two readings of the SAME arena under the SAME
    /// lens at different versions. Can-fire.
    #[test]
    fn same_lens_different_version_is_a_step() {
        let snap = Snapshot::of(&BeliefArena::new(), 0.0);
        let before = VersionedSnapshot::new(QueryReference::at(10, 2), 7, snap);
        let after = VersionedSnapshot::new(QueryReference::at(11, 2), 7, snap);
        assert!(before.steppable_to(&after));
    }

    /// Can-stay-silent, on NON-TRIVIAL differences — one per scope component.
    /// A reading taken under a different epistemic mode, rung, or arena is a
    /// different observation, and differencing it measures the lens change
    /// rather than the reasoning.
    #[test]
    fn a_changed_lens_is_not_a_step() {
        let snap = Snapshot::of(&BeliefArena::new(), 0.0);
        let base = VersionedSnapshot::new(QueryReference::at(10, 2), 7, snap);

        // Different rung — and, via `at`, a different derived mode.
        let other_rung = VersionedSnapshot::new(QueryReference::at(11, 5), 7, snap);
        assert!(!base.steppable_to(&other_rung));

        // Different arena entirely.
        let other_arena = VersionedSnapshot::new(QueryReference::at(11, 2), 9, snap);
        assert!(!base.steppable_to(&other_arena));

        // Different version line (server).
        let mut other_line = QueryReference::at(11, 2);
        other_line.server_id = 3;
        assert!(!base.steppable_to(&VersionedSnapshot::new(other_line, 7, snap)));
    }

    fn inh(s: u16, p: u16) -> CStmt {
        CStmt {
            s,
            cop: Copula::Inh,
            p,
        }
    }

    /// Build an arena whose beliefs carry exactly the given confidences.
    fn arena_with_confidences(cs: &[f32]) -> BeliefArena {
        let mut arena = BeliefArena::new();
        for (i, &c) in cs.iter().enumerate() {
            arena.observe(
                inh(i as u16, 900),
                TruthValue::new(0.9, c),
                Stamp::source(i as u32),
            );
        }
        arena
    }

    /// **The load-bearing guard of the atom routing.** `confidence_entropy`
    /// now delegates to `thought_atoms::normalized_entropy`, whose zero-mass
    /// convention is `Some(1.0)` — the OPPOSITE of this caller's, and the one
    /// fixture where PROBE-ENTROPY-SURFACE-CENSUS-1 measured the two
    /// conventions maximally apart. An empty arena builds an all-zero
    /// histogram, so without the caller's own early return the atom would
    /// report maximal uncertainty for an arena that holds no uncertainty at
    /// all — silently, since 1.0 is in range.
    ///
    /// Can-stay-silent. Disable by deleting the `is_empty` early return in
    /// `confidence_entropy`: this fails with 1.0.
    #[test]
    fn an_empty_arena_has_zero_truth_entropy_not_one() {
        let entropy = Snapshot::of(&BeliefArena::new(), 0.0).signals.truth_entropy;
        assert_eq!(
            entropy, 0.0,
            "an empty arena has no confidence distribution to be uncertain \
             about; the atom's zero-mass convention (1.0) must not reach here"
        );
        assert!(
            (entropy - 1.0).abs() > 0.5,
            "guard removed: the all-zero histogram reached the atom and came \
             back as uniform ({entropy})"
        );
    }

    /// Can-fire, on NON-TRIVIAL inputs — otherwise the assertion above would
    /// also hold for a `confidence_entropy` stubbed to always return 0.0.
    /// The routed atom must still span the full range: one occupied bin is
    /// zero uncertainty, ten evenly occupied bins is maximal, and the
    /// normalization is by the BIN COUNT (10), not by the occupied count.
    #[test]
    fn the_routed_atom_still_spans_the_confidence_range() {
        let peaked = Snapshot::of(&arena_with_confidences(&[0.42; 8]), 0.0)
            .signals
            .truth_entropy;
        assert!(
            peaked.abs() < 1e-6,
            "eight beliefs in one bin is a degenerate distribution: {peaked}"
        );

        let spread: Vec<f32> = (0..10).map(|k| 0.05 + 0.1 * k as f32).collect();
        let uniform = Snapshot::of(&arena_with_confidences(&spread), 0.0)
            .signals
            .truth_entropy;
        assert!(
            (uniform - 1.0).abs() < 1e-5,
            "ten beliefs, one per bin, is maximal spread: {uniform}"
        );

        // Half-occupied: five bins of two. H = log(5), normalized by log(10).
        let half: Vec<f32> = (0..10).map(|k| 0.05 + 0.2 * (k / 2) as f32).collect();
        let mid = Snapshot::of(&arena_with_confidences(&half), 0.0)
            .signals
            .truth_entropy;
        let expected = 5f32.ln() / 10f32.ln();
        assert!(
            (mid - expected).abs() < 1e-5,
            "normalization is by the bin count, not the occupied count: \
             {mid} vs {expected}"
        );
    }

    /// A deterministic SplitMix64 — the size-preserving null needs reproducible
    /// randomness (no clock/`rand`), like the certification-officer seed.
    struct SplitMix64(u64);
    impl SplitMix64 {
        fn next(&mut self) -> u64 {
            self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = self.0;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^ (z >> 31)
        }
        fn below(&mut self, n: u16) -> u16 {
            (self.next() % n as u64) as u16
        }
    }

    /// Snapshot a fresh arena over a set of `is_a` edges, run `close_transitive`,
    /// and score the step (before = observed-only, after = closed).
    fn score_step(edges: &[(u16, u16)]) -> InsightMush {
        let mut arena = BeliefArena::new();
        for (i, &(s, p)) in edges.iter().enumerate() {
            arena.observe(
                inh(s, p),
                TruthValue::new(0.95, 0.9),
                Stamp::source(i as u32),
            );
        }
        let before = Snapshot::of(&arena, 0.0);
        arena.close_transitive(64);
        // revision_velocity ~ 0 here (pure derivation, no observation revision).
        let after = Snapshot::of(&arena, 0.0);
        detect(&before, &after, 0.02)
    }

    /// The S10 MANDATORY gate (`E-BASIN-WIDTH`): a COHERENT arena (deep `is_a`
    /// chains that close into strong conclusions) must score higher insight than
    /// a SIZE-PRESERVING null (the same edge count, randomly rewired so chains
    /// are destroyed). If it does not, S10 measures size, not insight.
    #[test]
    fn insight_beats_size_preserving_null() {
        // Coherent: three long chains 0→1→…→10, 20→21→…→30, 40→41→…→50 — closure
        // derives many strong transitive conclusions.
        let mut coherent: Vec<(u16, u16)> = Vec::new();
        for base in [0u16, 20, 40] {
            for k in 0..10u16 {
                coherent.push((base + k, base + k + 1));
            }
        }
        let real = score_step(&coherent);

        // Size-preserving null: SAME number of edges, endpoints drawn from the
        // SAME concept id pool, but randomly rewired (chains destroyed).
        let n_nodes = 51u16;
        let mut rng = SplitMix64(0x1234_5678);
        let null_edges: Vec<(u16, u16)> = (0..coherent.len())
            .map(|_| (rng.below(n_nodes), rng.below(n_nodes)))
            .collect();
        let null = score_step(&null_edges);

        // The discriminator must BEAT the size-preserving null.
        assert!(
            real.insight > null.insight,
            "S10 must beat the size-preserving null: real insight {} vs null {} \
             (if this fails, S10 measures size, not insight)",
            real.insight,
            null.insight
        );
        assert!(real.insight > 0.0, "a coherent step registers insight");
    }

    /// A step that derives nothing (no shared middle terms) is gated to zero
    /// insight — the yield gate holds.
    #[test]
    fn no_yield_no_insight() {
        // Disjoint edges: no chain composes.
        let im = score_step(&[(1, 2), (3, 4), (5, 6)]);
        assert_eq!(im.insight, 0.0, "no derivation → yield gate → 0 insight");
        assert_eq!(
            flow_state(&im),
            FlowState::Boredom,
            "no insight, no mush → Boredom"
        );
    }

    /// `arena_graph_signals` reuses the contract carrier and reads a real yield.
    #[test]
    fn signals_reuse_contract_carrier() {
        let mut arena = BeliefArena::new();
        arena.observe(inh(1, 2), TruthValue::new(0.9, 0.9), Stamp::source(0));
        arena.observe(inh(2, 3), TruthValue::new(0.9, 0.9), Stamp::source(1));
        arena.close_transitive(8);
        let sig: GraphSignals = arena_graph_signals(&arena, 0.0);
        // 3 statements, 1 derived (1→3) → yield = 1/3.
        assert!((sig.deduction_yield - 1.0 / 3.0).abs() < 1e-6);
        assert!(sig.truth_entropy >= 0.0 && sig.truth_entropy <= 1.0);
    }
}
