//! D-GSO-6 (P6): deterministic, versioned recipe selector.
//!
//! Plan: `.claude/plans/2026-10-06-global-sudoku-replayable-orchestration-v1.md`
//! §11 (the loop), §13 (replay contract) and §18 P6.
//!
//! Claim under test: recipe selection is a pure function of a declared policy
//! version and a declared epistemic state. The same `(policy, state)` always
//! chooses the same recipe, a recorded selection replays under its recorded
//! policy even after the default policy changes, and a policy change is visible
//! as a different version rather than a silent change of behaviour.
//!
//! # The declared state
//!
//! Four facts an earlier step has already measured, nothing the selector
//! computes itself:
//!
//! | field | set by | meaning |
//! |---|---|---|
//! | `observations_pending` | evidence arrival | verdicts not yet folded (D-GSO-3) |
//! | `frontier_bounded` | a product interrogation | the candidate frontier is known (D-GSO-5 R1) |
//! | `local_disagreement` | a local fold | neighbourhood tension is present (D-GSO-2/3) |
//! | `revision_pending` | evidence arrival | revising would change the horizon (D-GSO-5 R3) |
//!
//! Policy V1 follows the §11 order DETECT → BOUND → PROPOSE/TEST → REVISE:
//! fold pending observations first, then bound the frontier, then interrogate
//! local disagreement, then revise. With nothing left it returns no recipe:
//! the cycle rests (a hole is never forced into a guess, §11).
//!
//! # What this probe does not decide
//!
//! - **The recipe ordinals and the V1 order are a scaffold, not canon.** They
//!   reuse the P5 scaffold names; `ProbeRecipe` is again its own type and never
//!   a shipped `recipes::Recipe` ID (plan §12).
//! - **The policy version covers selection only.** Plan §13 requires the
//!   recipe-policy version to cover every executable operation a recipe can
//!   select; tying the recipes' implementations to the version (a semantics
//!   digest) is not done here.
//! - **The state transitions in the replay test are a stand-in.** Each step
//!   clears the condition its recipe answers; real recipes would produce that
//!   change from their outputs. The wired cycle (`wired_cycle`) has none:
//!   all four facts are read from the world, see "Revision, wired",
//!   "Observation, wired" and "Frontier, wired" below. The stand-in `cycle`
//!   stays as the oracle for the bare selector.
//!
//! # Revision, wired
//!
//! `revision_pending` is read from the world, not supplied: it holds while
//! revising with the pending encounter would still change the horizon's
//! masks: the projected claims, a root not yet held, an inherited root not
//! yet held, or a contradiction not yet in the tension. Those are exactly the
//! fields `GadamerRevision` writes into `delta.resulting` (with the horizon as
//! ancestry), each one mask comparison (one fused fold, Round 6). It covers
//! rootless revisions too (`ContradictionPreserved`, `AssumptionExposed`,
//! `Reinterpretation`): they change the horizon without minting a root. The
//! `Revision` recipe runs `GadamerRevision::revise` and its only output that
//! reaches the next state is `delta.resulting`. So the cycle rests after
//! revision because the encounter is absorbed, not because a stand-in cleared
//! a flag. Bypass the write and the selector picks `Revision` again, forever.
//!
//! `local_disagreement` cannot be read off `unresolved_tension` alone:
//! revision preserves tension and never clears it, so a selector reading it
//! would interrogate forever. It is read as tension not yet interrogated:
//! `unresolved_tension \ interrogated`, where `interrogated` is the coverage
//! the `MooreInterrogation` recipe writes, the tension bits it has examined.
//! The tension itself is never cleared; only coverage grows. A revision that
//! adds a new contradiction reopens the fact, so a cycle can run `Revision`
//! and then interrogate the tension that revision introduced.
//!
//! Coverage is per tension bit, not per piece of evidence: new evidence on a
//! bit that is already covered does not reopen the interrogation. A coverage
//! that tracks receipts or generations would; it is not built here.
//!
//! What this does not decide: `interrogated` is a probe-local record. The
//! horizon carries no such field, and the Moore recipe's actual fold (the
//! Palette hop over `Register128`, D-GSO-5 R2) is not linked to claim bits
//! here; only the coverage it leaves is. Where coverage lives durably, and
//! what the interrogation concludes, stay open.
//!
//! # Observation, wired
//!
//! `observations_pending` is read from the world: more verdicts have arrived
//! than the quorum has counted. The `ObserveFold` recipe folds the unfolded
//! verdicts with `Quorum::observe` (D-GSO-3), and its only output is the
//! quorum. The count includes silent verdicts. `Quorum::speaking()` does not,
//! so a selector reading it would never rest on a silent source; that is
//! pinned as a test. Arrival itself is the "do": the test writes verdicts
//! into the world, the selector never creates one.
//!
//! What this does not decide: the quorum is not linked to the horizon. A
//! conflicting verdict does not become an encounter or a contradiction here.
//!
//! # Frontier, wired
//!
//! `frontier_bounded` is read from the world: a frontier has been recorded,
//! and it was recorded for the current candidate product and target. The
//! `ProductInterrogation` recipe folds the product with `Quad8::fold_product`
//! (D-GSO-5 R1) and records the frontier together with the space it bounded.
//! When the candidate space changes (a "do": the test writes a new product),
//! the recorded frontier is stale and the fact reopens. An empty frontier is
//! still a bounded one.
//!
//! What this does not decide: the frontier does not feed any other recipe
//! here, and nothing derives the candidate product from the horizon.
//!
//! Run: `cargo run -p cognitive-shader-driver --example recipe_selector_probe`
//! Tests: `cargo test -p cognitive-shader-driver --example recipe_selector_probe`

/// A probe recipe ordinal, as in the P5 scaffold.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum ProbeRecipe {
    ObserveFold,
    ProductInterrogation,
    MooreInterrogation,
    Revision,
}

/// The declared epistemic state the selector reads. Every field is supplied;
/// the selector derives nothing.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct EpistemicState {
    observations_pending: bool,
    frontier_bounded: bool,
    local_disagreement: bool,
    revision_pending: bool,
}

impl EpistemicState {
    /// All 16 states, in a fixed order.
    fn all() -> impl Iterator<Item = Self> {
        (0..16u8).map(|b| Self {
            observations_pending: b & 1 != 0,
            frontier_bounded: b & 2 != 0,
            local_disagreement: b & 4 != 0,
            revision_pending: b & 8 != 0,
        })
    }
}

/// A selection policy. The version is part of every selection it makes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum SelectorPolicy {
    /// §11 order: fold, bound, local, revise.
    V1,
    /// A later policy that interrogates local disagreement before bounding
    /// the frontier. It exists to show that a policy change is a new version.
    V2,
}

impl SelectorPolicy {
    /// The policy a new cycle uses. Recorded selections never consult this.
    const CURRENT: Self = Self::V1;

    /// Pure selection: no clock, no counter, no global, no randomness. `None`
    /// means nothing is left to do and the cycle rests.
    const fn select(self, s: EpistemicState) -> Option<ProbeRecipe> {
        if s.observations_pending {
            return Some(ProbeRecipe::ObserveFold);
        }
        match self {
            Self::V1 => {
                if !s.frontier_bounded {
                    return Some(ProbeRecipe::ProductInterrogation);
                }
                if s.local_disagreement {
                    return Some(ProbeRecipe::MooreInterrogation);
                }
            }
            Self::V2 => {
                if s.local_disagreement {
                    return Some(ProbeRecipe::MooreInterrogation);
                }
                if !s.frontier_bounded {
                    return Some(ProbeRecipe::ProductInterrogation);
                }
            }
        }
        if s.revision_pending {
            return Some(ProbeRecipe::Revision);
        }
        None
    }
}

/// What a replay needs to choose again: the policy version and the state.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct Selection {
    policy: SelectorPolicy,
    state: EpistemicState,
    recipe: Option<ProbeRecipe>,
}

impl Selection {
    /// Select under `policy` and record what was used.
    const fn make(policy: SelectorPolicy, state: EpistemicState) -> Self {
        Self {
            policy,
            state,
            recipe: policy.select(state),
        }
    }

    /// Choose again from the record alone.
    const fn replay(self) -> Option<ProbeRecipe> {
        self.policy.select(self.state)
    }
}

/// Stand-in transition: the recipe clears the condition it answers.
const fn after(recipe: ProbeRecipe, s: EpistemicState) -> EpistemicState {
    match recipe {
        ProbeRecipe::ObserveFold => EpistemicState {
            observations_pending: false,
            ..s
        },
        ProbeRecipe::ProductInterrogation => EpistemicState {
            frontier_bounded: true,
            ..s
        },
        ProbeRecipe::MooreInterrogation => EpistemicState {
            local_disagreement: false,
            ..s
        },
        ProbeRecipe::Revision => EpistemicState {
            revision_pending: false,
            ..s
        },
    }
}

/// The most steps any cycle can take: one per condition.
const MAX_STEPS: usize = 4;

/// Run a cycle from `start` under `policy` until it rests. Returns the
/// recipes chosen, in order, and how many there were. Fixed-size, no heap.
fn cycle(
    policy: SelectorPolicy,
    start: EpistemicState,
) -> ([Option<ProbeRecipe>; MAX_STEPS + 1], usize) {
    let mut trace = [None; MAX_STEPS + 1];
    let mut state = start;
    for (step, slot) in trace.iter_mut().enumerate() {
        match policy.select(state) {
            Some(recipe) => {
                *slot = Some(recipe);
                state = after(recipe, state);
            }
            None => return (trace, step),
        }
    }
    (trace, MAX_STEPS + 1)
}

// ── revision, wired ─────────────────────────────────────────────────────

use cognitive_shader_driver::quad8::Quad8;
use lance_graph_contract::ontology_warrant::{Quorum, SourceVerdict};
use lance_graph_contract::revision::{
    BasisView, CodebookId, EncounterEvidence, EvidenceMask, GadamerRevision, GrammarId, HorizonId,
    InterpretiveHorizon, LanguageId, LensId, QuestionId, RevisionPolicy,
};

type Horizon = InterpretiveHorizon<(), u64>;

/// The part of the world the wired fact reads: the current horizon and the
/// encounter waiting at it. The encounter is supplied by observation; the
/// selector never creates one.
#[derive(Debug, Clone, PartialEq, Eq)]
struct World {
    horizon: Horizon,
    encounter: EncounterEvidence<u64>,
    /// Tension bits `MooreInterrogation` has examined. Only that recipe
    /// writes it, and only by union.
    interrogated: u64,
    /// Verdicts that have arrived, in arrival order. Only arrival writes here.
    verdicts: [SourceVerdict; VERDICTS],
    /// How many of `verdicts` have arrived.
    arrived: usize,
    /// The `ObserveFold` recipe's output: the quorum over folded verdicts.
    quorum: Quorum,
    /// The candidate space: a finite product and the sum a candidate needs.
    /// Only the "do" changes it.
    space: (Quad8, u8),
    /// The `ProductInterrogation` recipe's output, if it has run.
    frontier: Option<Frontier>,
}

/// What `ProductInterrogation` records: the frontier of one candidate space.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Frontier {
    /// The space this frontier was measured over.
    of: (Quad8, u8),
    /// How many candidates meet the target.
    count: u16,
    /// The first one, as a raw product address.
    first: Option<u16>,
}

/// How many verdicts a world can hold.
const VERDICTS: usize = 8;

/// The horizon as ancestry, as in the Round 6 cycle.
fn ancestry(h: &Horizon) -> BasisView<u64> {
    BasisView {
        ancestry_independent_roots: h.independent_roots,
        ancestry_derived_roots: h.inherited_roots,
        ancestor_claims: h.projected_claims,
        closes_cycle: false,
    }
}

impl World {
    /// `revision_pending`, derived: revising would still change a mask of the
    /// horizon. One test per field `delta.resulting` writes.
    fn revision_pending(&self) -> bool {
        let (h, e) = (&self.horizon, &self.encounter);
        e.proposed_claims != h.projected_claims
            || !e
                .independent_roots
                .difference(&h.independent_roots)
                .is_empty()
            || !e.inherited_roots.difference(&h.inherited_roots).is_empty()
            || !e
                .contradictions
                .difference(&h.unresolved_tension)
                .is_empty()
    }

    /// `local_disagreement`, derived: tension not yet interrogated.
    fn local_disagreement(&self) -> bool {
        !self
            .horizon
            .unresolved_tension
            .difference(&self.interrogated)
            .is_empty()
    }

    /// The declared state, every fact read from the world.
    fn state(&self) -> EpistemicState {
        EpistemicState {
            observations_pending: self.observations_pending(),
            frontier_bounded: self.frontier_bounded(),
            revision_pending: self.revision_pending(),
            local_disagreement: self.local_disagreement(),
        }
    }

    /// `frontier_bounded`, derived: a frontier exists for the current space.
    fn frontier_bounded(&self) -> bool {
        self.frontier.is_some_and(|f| f.of == self.space)
    }

    /// The "do": the candidate space changes. Not a recipe.
    fn change_space(&mut self, quad: Quad8, target_sum: u8) {
        self.space = (quad, target_sum);
    }

    /// The `ProductInterrogation` recipe: fold the product, record the
    /// frontier and the space it covers.
    fn bound_frontier(&mut self) {
        let (quad, target) = self.space;
        self.frontier = Some(quad.fold_product(
            Frontier {
                of: self.space,
                count: 0,
                first: None,
            },
            |acc, addr| {
                let sum: u8 = addr.ordinals().iter().sum();
                if sum == target {
                    Frontier {
                        count: acc.count + 1,
                        first: acc.first.or(Some(addr.raw())),
                        ..acc
                    }
                } else {
                    acc
                }
            },
        ));
    }

    /// How many verdicts the quorum has counted, silence included.
    fn folded(&self) -> usize {
        let q = self.quorum;
        usize::from(q.corroborating) + usize::from(q.silent) + usize::from(q.conflicting)
    }

    /// `observations_pending`, derived: verdicts arrived but not yet counted.
    fn observations_pending(&self) -> bool {
        self.arrived > self.folded()
    }

    /// The "do": a verdict arrives. Not a recipe; the selector never calls it.
    fn arrive(&mut self, v: SourceVerdict) {
        self.verdicts[self.arrived] = v;
        self.arrived += 1;
    }

    /// The `ObserveFold` recipe: fold the verdicts not yet counted.
    fn fold_observations(&mut self) {
        let start = self.folded();
        self.quorum = self.verdicts[start..self.arrived]
            .iter()
            .fold(self.quorum, |q, &v| q.observe(v));
    }

    /// The `MooreInterrogation` recipe's write: the tension it examined.
    fn interrogate(&mut self) {
        self.interrogated |= self.horizon.unresolved_tension;
    }

    /// The `Revision` recipe: revise once; `delta.resulting` is the only
    /// output carried forward.
    fn revise(&mut self) {
        let delta =
            GadamerRevision.revise(&self.horizon, &self.encounter, &ancestry(&self.horizon));
        self.horizon = delta.resulting;
    }
}

/// The most steps a wired cycle may take: the two stand-in conditions, one
/// revision, and an interrogation before and after it (revision can reopen
/// local disagreement).
const WIRED_MAX_STEPS: usize = 5;

/// Run a cycle whose four facts are all read from `world` at every step.
/// `drop` names a recipe whose write is discarded (the bypass falsifiers):
/// the recipe runs on a copy and nothing lands.
fn wired_cycle(
    policy: SelectorPolicy,
    world: &mut World,
    drop: Option<ProbeRecipe>,
) -> ([Option<ProbeRecipe>; WIRED_MAX_STEPS + 1], usize) {
    let mut trace = [None; WIRED_MAX_STEPS + 1];
    for (step, slot) in trace.iter_mut().enumerate() {
        let Some(recipe) = policy.select(world.state()) else {
            return (trace, step);
        };
        *slot = Some(recipe);
        let target = if drop == Some(recipe) {
            &mut world.clone()
        } else {
            &mut *world
        };
        match recipe {
            ProbeRecipe::Revision => target.revise(),
            ProbeRecipe::MooreInterrogation => target.interrogate(),
            ProbeRecipe::ObserveFold => target.fold_observations(),
            ProbeRecipe::ProductInterrogation => target.bound_frontier(),
        }
    }
    (trace, WIRED_MAX_STEPS + 1)
}

/// A small world: the horizon holds claims {0, 2}; the encounter proposes
/// {1, 2}, rooted in {1, 2}, and contradicts claim 0.
/// The fusion world with its frontier already bounded, so only the facts a
/// test sets up are open.
#[cfg(test)]
fn fusion_world() -> World {
    let mut w = unbounded_fusion_world();
    w.bound_frontier();
    w
}

fn unbounded_fusion_world() -> World {
    World {
        horizon: InterpretiveHorizon {
            id: HorizonId(0),
            awareness: (),
            question: QuestionId(6),
            language: LanguageId(0),
            grammar: GrammarId(0),
            codebook: CodebookId(0),
            lens: LensId(0),
            projected_claims: 0b101,
            independent_roots: 0b101,
            inherited_roots: 0,
            unresolved_tension: 0,
            revision_index: 0,
        },
        encounter: EncounterEvidence {
            proposed_claims: 0b110,
            independent_roots: 0b110,
            inherited_roots: 0,
            resistance: 0b001,
            contradictions: 0b001,
            affected_parts: 0b111,
        },
        interrogated: 0,
        verdicts: [SourceVerdict::Silent; VERDICTS],
        arrived: 0,
        quorum: Quorum::default(),
        space: (
            Quad8::new(0b0000_0111, 0b0000_0110, 0b1000_0001, 0b0000_0011),
            4,
        ),
        frontier: None,
    }
}

fn main() {
    let start = EpistemicState {
        observations_pending: true,
        frontier_bounded: false,
        local_disagreement: true,
        revision_pending: true,
    };
    for policy in [SelectorPolicy::V1, SelectorPolicy::V2] {
        let (trace, n) = cycle(policy, start);
        println!("{policy:?}: {:?}", &trace[..n]);
    }
    let mut replayed = 0;
    for state in EpistemicState::all() {
        let recorded = Selection::make(SelectorPolicy::CURRENT, state);
        assert_eq!(recorded.replay(), recorded.recipe);
        replayed += 1;
    }
    println!(
        "{replayed} selections recorded under {:?} and replayed",
        SelectorPolicy::CURRENT
    );
    let mut world = unbounded_fusion_world();
    world.arrive(SourceVerdict::Corroborates);
    let (trace, n) = wired_cycle(SelectorPolicy::CURRENT, &mut world, None);
    println!(
        "wired: {:?}, horizon roots {:#b}",
        &trace[..n],
        world.horizon.independent_roots
    );
    world.change_space(Quad8::new(0b11, 0b11, 0b1, 0b1), 2);
    let (trace, n) = wired_cycle(SelectorPolicy::CURRENT, &mut world, None);
    println!(
        "after a space change: {:?}, frontier {:?}",
        &trace[..n],
        world.frontier.map(|f| f.count)
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use lance_graph_contract::revision::RevisionKind;
    use ProbeRecipe::*;

    fn state(o: bool, f: bool, l: bool, n: bool) -> EpistemicState {
        EpistemicState {
            observations_pending: o,
            frontier_bounded: f,
            local_disagreement: l,
            revision_pending: n,
        }
    }

    /// FAILS IF: V1 departs from the §11 order on any of the 16 states. The
    /// oracle is written out as a table, not derived from `select`.
    #[test]
    fn v1_follows_the_loop_order_on_every_state() {
        let mut checked = 0;
        for s in EpistemicState::all() {
            let expected = if s.observations_pending {
                Some(ObserveFold)
            } else if !s.frontier_bounded {
                Some(ProductInterrogation)
            } else if s.local_disagreement {
                Some(MooreInterrogation)
            } else if s.revision_pending {
                Some(Revision)
            } else {
                None
            };
            assert_eq!(SelectorPolicy::V1.select(s), expected, "{s:?}");
            checked += 1;
        }
        assert_eq!(checked, 16);
        // Spot checks that pin the order directly.
        assert_eq!(
            SelectorPolicy::V1.select(state(false, false, true, true)),
            Some(ProductInterrogation)
        );
        assert_eq!(
            SelectorPolicy::V1.select(state(false, true, false, false)),
            None
        );
    }

    /// FAILS IF: the same (policy, state) ever chooses differently, whether
    /// asked twice in a row or in a different order.
    #[test]
    fn selection_is_a_function_of_policy_and_state() {
        for policy in [SelectorPolicy::V1, SelectorPolicy::V2] {
            let forward: Vec<_> = EpistemicState::all().map(|s| policy.select(s)).collect();
            let again: Vec<_> = EpistemicState::all().map(|s| policy.select(s)).collect();
            let mut backward: Vec<_> = EpistemicState::all()
                .collect::<Vec<_>>()
                .into_iter()
                .rev()
                .map(|s| policy.select(s))
                .collect();
            backward.reverse();
            assert_eq!(forward, again, "{policy:?}");
            assert_eq!(forward, backward, "{policy:?}");
        }
    }

    /// FAILS IF: a recorded selection does not replay to the same recipe from
    /// the record alone, or a policy change is not visible as a version.
    #[test]
    fn a_recorded_selection_replays_under_its_own_policy() {
        // V1 and V2 differ on some states, so the version carries meaning.
        let differing: Vec<_> = EpistemicState::all()
            .filter(|&s| SelectorPolicy::V1.select(s) != SelectorPolicy::V2.select(s))
            .collect();
        assert_eq!(
            differing.len(),
            2,
            "pinned: the two states with local tension and no frontier"
        );
        let s = differing[0];

        // A selection recorded under V1 replays to V1's answer even though V2
        // would answer differently today.
        let recorded = Selection::make(SelectorPolicy::V1, s);
        assert_eq!(recorded.replay(), recorded.recipe);
        assert_ne!(recorded.replay(), SelectorPolicy::V2.select(s));

        // Every record replays, under both policies, on every state.
        for policy in [SelectorPolicy::V1, SelectorPolicy::V2] {
            for s in EpistemicState::all() {
                let r = Selection::make(policy, s);
                assert_eq!(r.replay(), r.recipe);
            }
        }
        assert_eq!(SelectorPolicy::CURRENT, SelectorPolicy::V1);
    }

    /// FAILS IF: a cycle does not rest, takes more than one step per
    /// condition, or replays to a different path.
    #[test]
    fn every_cycle_rests_and_replays() {
        for policy in [SelectorPolicy::V1, SelectorPolicy::V2] {
            for s in EpistemicState::all() {
                let (trace, n) = cycle(policy, s);
                assert!(n <= MAX_STEPS, "{policy:?} {s:?} did not rest");
                let conditions = usize::from(s.observations_pending)
                    + usize::from(!s.frontier_bounded)
                    + usize::from(s.local_disagreement)
                    + usize::from(s.revision_pending);
                assert_eq!(n, conditions, "{policy:?} {s:?}");
                assert_eq!(cycle(policy, s), (trace, n), "{policy:?} {s:?}");
            }
        }
        // The full cycle, pinned per policy.
        let full = state(true, false, true, true);
        assert_eq!(
            &cycle(SelectorPolicy::V1, full).0[..4],
            &[
                Some(ObserveFold),
                Some(ProductInterrogation),
                Some(MooreInterrogation),
                Some(Revision)
            ]
        );
        assert_eq!(
            &cycle(SelectorPolicy::V2, full).0[..4],
            &[
                Some(ObserveFold),
                Some(MooreInterrogation),
                Some(ProductInterrogation),
                Some(Revision)
            ]
        );
    }

    /// The settled stand-in facts, so only the wired fact is open.
    fn settled() -> EpistemicState {
        state(false, true, false, false)
    }

    /// The masks `delta.resulting` can change.
    fn masks(h: &Horizon) -> [u64; 4] {
        [
            h.projected_claims,
            h.independent_roots,
            h.inherited_roots,
            h.unresolved_tension,
        ]
    }

    /// FAILS IF: the derived fact disagrees with whether revising actually
    /// changes the horizon, on any world of a 2-bit universe.
    #[test]
    fn the_derived_fact_is_whether_revision_changes_the_horizon() {
        let (mut pending, mut settled_worlds) = (0, 0);
        for x in 0u32..1 << 16 {
            let f = |i: u32| u64::from(x >> (2 * i) & 3);
            let mut w = fusion_world();
            w.horizon.projected_claims = f(0);
            w.horizon.independent_roots = f(1);
            w.horizon.inherited_roots = f(2);
            w.horizon.unresolved_tension = f(3);
            w.encounter.proposed_claims = f(4);
            w.encounter.independent_roots = f(5);
            w.encounter.inherited_roots = f(6);
            w.encounter.contradictions = f(7);
            w.encounter.resistance = f(7);
            let before = masks(&w.horizon);
            let derived = w.revision_pending();
            w.revise();
            assert_eq!(derived, masks(&w.horizon) != before, "{x:#x}");
            if derived {
                pending += 1;
            } else {
                settled_worlds += 1;
            }
        }
        assert!(pending > 0 && settled_worlds > 0, "both outcomes occur");
    }

    /// FAILS IF: the derived fact disagrees with revision's own new-root test.
    #[test]
    fn the_derived_fact_is_revisions_new_root_test() {
        let w = fusion_world();
        let delta = GadamerRevision.revise(&w.horizon, &w.encounter, &ancestry(&w.horizon));
        assert!(w.revision_pending());
        assert_ne!(delta.new_independent_roots, 0);
        let after = World {
            horizon: delta.resulting,
            ..w
        };
        assert!(
            !after.revision_pending(),
            "the resulting horizon has the roots"
        );
    }

    /// FAILS IF: the cycle does not rest after one real revision, or rests
    /// without one. The rest comes from `delta.resulting`, not a stand-in.
    #[test]
    fn a_real_revision_makes_the_cycle_rest() {
        let mut world = fusion_world();
        let before = world.horizon.clone();
        let (trace, n) = wired_cycle(SelectorPolicy::V1, &mut world, None);
        // Revision adds contradiction 0 to the tension, which reopens local
        // disagreement; one interrogation covers it and the cycle rests.
        assert_eq!(&trace[..n], &[Some(Revision), Some(MooreInterrogation)]);
        assert_eq!(world.horizon.independent_roots, 0b111);
        assert_eq!(world.horizon.revision_index, before.revision_index + 1);

        // The full cycle under V1: one verdict has arrived, the frontier is
        // not yet bounded, and there is no tension yet.
        let mut world = unbounded_fusion_world();
        world.arrive(SourceVerdict::Corroborates);
        let (trace, n) = wired_cycle(SelectorPolicy::V1, &mut world, None);
        assert_eq!(
            &trace[..n],
            &[
                Some(ObserveFold),
                Some(ProductInterrogation),
                Some(Revision),
                Some(MooreInterrogation)
            ]
        );
    }

    /// FAILS IF: dropping revision's write still lets the cycle rest, i.e.
    /// the rest does not depend on revision's output.
    #[test]
    fn dropping_the_revision_write_never_rests() {
        let mut world = fusion_world();
        let (trace, n) = wired_cycle(SelectorPolicy::V1, &mut world, Some(Revision));
        assert_eq!(n, WIRED_MAX_STEPS + 1, "did not rest");
        assert!(trace.iter().all(|r| *r == Some(Revision)));
        assert_eq!(world, fusion_world(), "nothing was written");
    }

    /// FAILS IF: the derived fact reads claims instead of roots. Claim 1 is
    /// already projected but has no independent root; the encounter brings
    /// one. Revision calls that an independent confirmation, so the selector
    /// must pick it.
    #[test]
    fn a_new_root_for_a_held_claim_selects_revision() {
        let mut world = fusion_world();
        world.horizon.projected_claims = 0b011;
        world.horizon.independent_roots = 0b001;
        world.encounter = EncounterEvidence {
            proposed_claims: 0b011,
            independent_roots: 0b010,
            inherited_roots: 0,
            resistance: 0,
            contradictions: 0,
            affected_parts: 0b011,
        };
        let delta =
            GadamerRevision.revise(&world.horizon, &world.encounter, &ancestry(&world.horizon));
        assert_eq!(delta.kind, RevisionKind::IndependentConfirmation);
        assert!(world.revision_pending());
        let (trace, n) = wired_cycle(SelectorPolicy::V1, &mut world, None);
        assert_eq!(&trace[..n], &[Some(Revision)]);
        assert_eq!(world.horizon.independent_roots, 0b011);
    }

    /// An echo: the encounter proposes what the horizon already projects,
    /// with roots it already holds and no contradiction.
    fn echo_world() -> World {
        let mut w = fusion_world();
        w.encounter = EncounterEvidence {
            proposed_claims: 0b101,
            independent_roots: 0b001,
            inherited_roots: 0,
            resistance: 0,
            contradictions: 0,
            affected_parts: 0b101,
        };
        w
    }

    /// FAILS IF: an echo selects revision.
    #[test]
    fn an_echo_selects_nothing() {
        let mut world = echo_world();
        let delta =
            GadamerRevision.revise(&world.horizon, &world.encounter, &ancestry(&world.horizon));
        assert_eq!(delta.kind, RevisionKind::Echo);
        assert!(!world.revision_pending());
        let (_, n) = wired_cycle(SelectorPolicy::V1, &mut world, None);
        assert_eq!(n, 0);
        assert_eq!(world, echo_world());
    }

    /// FAILS IF: a revision that mints no root but changes the horizon is
    /// dropped. The encounter contradicts claim 0 and withdraws it, with no
    /// new root: `ContradictionPreserved`. It must be applied once, then rest.
    #[test]
    fn a_rootless_contradiction_is_still_revised() {
        let mut world = fusion_world();
        world.encounter.proposed_claims = 0b100;
        world.encounter.independent_roots = 0b100;
        let delta =
            GadamerRevision.revise(&world.horizon, &world.encounter, &ancestry(&world.horizon));
        assert_eq!(delta.kind, RevisionKind::ContradictionPreserved);
        assert_eq!(delta.new_independent_roots, 0);
        assert!(world.revision_pending());
        let (trace, n) = wired_cycle(SelectorPolicy::V1, &mut world, None);
        assert_eq!(&trace[..n], &[Some(Revision), Some(MooreInterrogation)]);
        assert_eq!(world.horizon.projected_claims, 0b100);
        assert_eq!(world.horizon.unresolved_tension, 0b001);
    }

    /// FAILS IF: the same starting world replays to a different path or a
    /// different final horizon.
    #[test]
    fn the_wired_cycle_replays() {
        for policy in [SelectorPolicy::V1, SelectorPolicy::V2] {
            for target in EpistemicState::all() {
                let (mut a, mut b) = (world_for(target), world_for(target));
                assert_eq!(a.state(), target, "the world reads as the state");
                let ra = wired_cycle(policy, &mut a, None);
                let rb = wired_cycle(policy, &mut b, None);
                assert_eq!(ra, rb, "{policy:?} {target:?}");
                assert_eq!(a, b, "{policy:?} {target:?}");
                assert!(
                    ra.1 <= WIRED_MAX_STEPS,
                    "{policy:?} {target:?} did not rest"
                );
                assert_eq!(a.state(), settled(), "{policy:?} {target:?}");
            }
        }
    }

    /// FAILS IF: interrogation's coverage is not what clears the fact, or
    /// clearing it touches the tension itself.
    #[test]
    fn interrogation_clears_the_fact_and_keeps_the_tension() {
        let mut world = fusion_world();
        world.horizon.unresolved_tension = 0b001;
        world.encounter = echo_world().encounter;
        assert!(world.local_disagreement());
        assert!(!world.revision_pending());
        let (trace, n) = wired_cycle(SelectorPolicy::V1, &mut world, None);
        assert_eq!(&trace[..n], &[Some(MooreInterrogation)]);
        assert!(!world.local_disagreement());
        assert_eq!(world.horizon.unresolved_tension, 0b001, "tension kept");
        assert_eq!(world.interrogated, 0b001);
    }

    /// FAILS IF: dropping the interrogation's write still lets the cycle
    /// rest, i.e. the rest does not depend on the interrogation's output.
    #[test]
    fn dropping_the_interrogation_write_never_rests() {
        let mut world = fusion_world();
        world.horizon.unresolved_tension = 0b001;
        world.encounter = echo_world().encounter;
        let before = world.clone();
        let (trace, n) = wired_cycle(SelectorPolicy::V1, &mut world, Some(MooreInterrogation));
        assert_eq!(n, WIRED_MAX_STEPS + 1, "did not rest");
        assert!(trace.iter().all(|r| *r == Some(MooreInterrogation)));
        assert_eq!(world, before, "nothing was written");
    }

    /// FAILS IF: tension already interrogated reopens the fact, or a new
    /// contradiction does not.
    #[test]
    fn only_new_tension_reopens_local_disagreement() {
        // Contradiction 0 is already in the tension and already covered: the
        // revision changes the projection, so it runs, but adds no new
        // tension, so nothing is interrogated afterwards.
        let mut world = fusion_world();
        world.horizon.unresolved_tension = 0b001;
        world.interrogated = 0b001;
        let (trace, n) = wired_cycle(SelectorPolicy::V1, &mut world, None);
        assert_eq!(&trace[..n], &[Some(Revision)]);

        // A contradiction on a new bit reopens it after the revision.
        let mut world = fusion_world();
        world.horizon.unresolved_tension = 0b001;
        world.interrogated = 0b001;
        world.encounter.contradictions = 0b011;
        world.encounter.resistance = 0b011;
        let (trace, n) = wired_cycle(SelectorPolicy::V1, &mut world, None);
        assert_eq!(&trace[..n], &[Some(Revision), Some(MooreInterrogation)]);
        assert_eq!(world.interrogated, 0b011);
    }

    /// An echo world (no revision pending, no tension) with `verdicts` arrived.
    fn arrivals(verdicts: &[SourceVerdict]) -> World {
        let mut w = echo_world();
        for &v in verdicts {
            w.arrive(v);
        }
        w
    }

    /// FAILS IF: the fold does not clear the fact, or counts a verdict twice
    /// or not at all.
    #[test]
    fn folding_clears_pending_observations() {
        use SourceVerdict::*;
        let mut world = arrivals(&[Corroborates, Silent, Conflicts]);
        assert!(world.observations_pending());
        let (trace, n) = wired_cycle(SelectorPolicy::V1, &mut world, None);
        assert_eq!(&trace[..n], &[Some(ObserveFold)]);
        assert_eq!(world.quorum, Quorum::new(1, 1, 1));
        assert!(!world.observations_pending());
    }

    /// FAILS IF: pending is read off the speaking count. A silent verdict is
    /// folded but never speaks, so that reading would never rest.
    #[test]
    fn a_silent_verdict_still_settles() {
        let mut world = arrivals(&[SourceVerdict::Silent]);
        let (trace, n) = wired_cycle(SelectorPolicy::V1, &mut world, None);
        assert_eq!(&trace[..n], &[Some(ObserveFold)]);
        assert_eq!(world.quorum.speaking(), 0, "silence does not speak");
        assert!(usize::from(world.quorum.speaking()) < world.arrived);
    }

    /// FAILS IF: dropping the fold's write still lets the cycle rest.
    #[test]
    fn dropping_the_fold_write_never_rests() {
        let mut world = arrivals(&[SourceVerdict::Corroborates]);
        let before = world.clone();
        let (trace, n) = wired_cycle(SelectorPolicy::V1, &mut world, Some(ObserveFold));
        assert_eq!(n, WIRED_MAX_STEPS + 1, "did not rest");
        assert!(trace.iter().all(|r| *r == Some(ObserveFold)));
        assert_eq!(world, before, "nothing was written");
    }

    /// FAILS IF: a late verdict does not reopen the fact, or the second fold
    /// recounts verdicts it already counted.
    #[test]
    fn a_late_verdict_is_folded_once() {
        use SourceVerdict::*;
        let mut world = arrivals(&[Corroborates]);
        wired_cycle(SelectorPolicy::V1, &mut world, None);
        assert_eq!(world.quorum, Quorum::new(1, 0, 0));
        world.arrive(Conflicts);
        assert!(world.observations_pending());
        let (trace, n) = wired_cycle(SelectorPolicy::V1, &mut world, None);
        assert_eq!(&trace[..n], &[Some(ObserveFold)]);
        assert_eq!(world.quorum, Quorum::new(1, 0, 1));
    }

    /// A world whose derived state is `target`: start from the settled echo
    /// world and open each fact with its own "do".
    fn world_for(target: EpistemicState) -> World {
        let mut w = echo_world();
        if target.observations_pending {
            w.arrive(SourceVerdict::Conflicts);
        }
        if !target.frontier_bounded {
            w.change_space(Quad8::new(0b11, 0b11, 0b1, 0b1), 2);
        }
        if target.local_disagreement {
            w.horizon.unresolved_tension |= 0b100;
        }
        if target.revision_pending {
            w.encounter = fusion_world().encounter;
        }
        w
    }

    /// How many candidates of `(quad, target)` meet the target, by brute
    /// force over every coordinate.
    fn frontier_oracle(quad: Quad8, target: u8) -> u16 {
        let b = quad.bytes();
        let on = |byte: u8, i: u8| byte >> i & 1 == 1;
        let mut count = 0;
        for a in 0..8u8 {
            for c in 0..8u8 {
                for e in 0..8u8 {
                    for d in 0..8u8 {
                        let occupied = on(b[0], a) && on(b[1], c) && on(b[2], e) && on(b[3], d);
                        if occupied && a + c + e + d == target {
                            count += 1;
                        }
                    }
                }
            }
        }
        count
    }

    /// FAILS IF: bounding does not clear the fact, or records a frontier
    /// different from a brute-force count over the same space.
    #[test]
    fn bounding_records_the_frontier_and_clears_the_fact() {
        let mut world = echo_world();
        world.frontier = None;
        assert!(!world.frontier_bounded());
        let (trace, n) = wired_cycle(SelectorPolicy::V1, &mut world, None);
        assert_eq!(&trace[..n], &[Some(ProductInterrogation)]);
        let f = world.frontier.expect("recorded");
        let (quad, target) = world.space;
        assert_eq!(f.of, world.space);
        assert_eq!(f.count, frontier_oracle(quad, target));
        assert!(f.count > 0, "the fixture space has candidates");
        assert!(f.first.is_some());
        assert!(world.frontier_bounded());
    }

    /// FAILS IF: a changed candidate space does not reopen the fact, or the
    /// new frontier is not measured over the new space.
    #[test]
    fn changing_the_space_reopens_the_frontier() {
        let mut world = echo_world();
        assert!(world.frontier_bounded());
        let new = (Quad8::new(0b11, 0b11, 0b1, 0b1), 2);
        world.change_space(new.0, new.1);
        assert!(!world.frontier_bounded(), "the old frontier is stale");
        let (trace, n) = wired_cycle(SelectorPolicy::V1, &mut world, None);
        assert_eq!(&trace[..n], &[Some(ProductInterrogation)]);
        let f = world.frontier.expect("recorded");
        assert_eq!(f.of, new);
        assert_eq!(f.count, frontier_oracle(new.0, new.1));
    }

    /// FAILS IF: an empty frontier is treated as unbounded. Knowing there are
    /// no candidates is a bound.
    #[test]
    fn an_empty_frontier_is_bounded() {
        let mut world = echo_world();
        world.change_space(Quad8::new(0b1, 0b1, 0b1, 0b1), 9);
        let (trace, n) = wired_cycle(SelectorPolicy::V1, &mut world, None);
        assert_eq!(&trace[..n], &[Some(ProductInterrogation)]);
        assert_eq!(world.frontier.map(|f| f.count), Some(0));
        assert!(world.frontier_bounded());
    }

    /// FAILS IF: dropping the product interrogation's write still rests.
    #[test]
    fn dropping_the_frontier_write_never_rests() {
        let mut world = echo_world();
        world.frontier = None;
        let before = world.clone();
        let (trace, n) = wired_cycle(SelectorPolicy::V1, &mut world, Some(ProductInterrogation));
        assert_eq!(n, WIRED_MAX_STEPS + 1, "did not rest");
        assert!(trace.iter().all(|r| *r == Some(ProductInterrogation)));
        assert_eq!(world, before, "nothing was written");
    }

    /// FAILS IF: revision starts clearing tension. Then `local_disagreement`
    /// could be read off the horizon; while it holds, it cannot, because the
    /// tension survives every revision and a reader of it would never rest.
    #[test]
    fn preserved_tension_cannot_drive_the_selector() {
        let mut world = fusion_world();
        world.revise();
        assert_eq!(world.horizon.unresolved_tension, 0b001);
        world.revise();
        assert_eq!(world.horizon.unresolved_tension, 0b001, "still held");
    }

    /// FAILS IF: the selector fires on a settled state or stays silent on an
    /// open one. Only the fully settled state rests.
    #[test]
    fn only_a_settled_state_rests() {
        for policy in [SelectorPolicy::V1, SelectorPolicy::V2] {
            let resting: Vec<_> = EpistemicState::all()
                .filter(|&s| policy.select(s).is_none())
                .collect();
            assert_eq!(
                resting,
                vec![state(false, true, false, false)],
                "{policy:?}"
            );
        }
    }
}
