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
//! | `new_encounter` | evidence arrival | an encounter awaits revision (D-GSO-5 R3) |
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
//!   change from their outputs. `new_encounter` is no longer one: see
//!   "Revision, wired" below. The other three still are.
//!
//! # Revision, wired
//!
//! `new_encounter` is read from the world, not supplied: it holds while the
//! pending encounter carries an independent root the horizon does not yet
//! have. That is `GadamerRevision`'s own `has_new_root` with the horizon as
//! ancestry, one mask difference (one fused `Any` fold, Round 6). The
//! `Revision` recipe runs `GadamerRevision::revise` and its only output that
//! reaches the next state is `delta.resulting`. So the cycle rests after
//! revision because revision absorbed the roots, not because a stand-in
//! cleared a flag. Bypass the write and the selector picks `Revision` again,
//! forever.
//!
//! `local_disagreement` cannot be read off the horizon the same way:
//! `unresolved_tension` is preserved by revision and never cleared, so a
//! selector reading it would interrogate forever. Pinned as a test; the
//! source for that fact stays open.
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
    new_encounter: bool,
}

impl EpistemicState {
    /// All 16 states, in a fixed order.
    fn all() -> impl Iterator<Item = Self> {
        (0..16u8).map(|b| Self {
            observations_pending: b & 1 != 0,
            frontier_bounded: b & 2 != 0,
            local_disagreement: b & 4 != 0,
            new_encounter: b & 8 != 0,
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
        if s.new_encounter {
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
            new_encounter: false,
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
}

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
    /// `new_encounter`, derived: the encounter carries a root the horizon
    /// does not have. Equal to `revise(...).new_independent_roots != 0`.
    fn new_encounter(&self) -> bool {
        !self
            .encounter
            .independent_roots
            .difference(&self.horizon.independent_roots)
            .is_empty()
    }

    /// The declared state: three stand-in facts plus the derived one.
    fn state(&self, rest: EpistemicState) -> EpistemicState {
        EpistemicState {
            new_encounter: self.new_encounter(),
            ..rest
        }
    }

    /// The `Revision` recipe: revise once; `delta.resulting` is the only
    /// output carried forward.
    fn revise(&mut self) {
        let delta =
            GadamerRevision.revise(&self.horizon, &self.encounter, &ancestry(&self.horizon));
        self.horizon = delta.resulting;
    }
}

/// Run a cycle whose `new_encounter` is derived from `world` at every step.
/// The other three facts use the stand-in transition. `write` is false only
/// in the bypass falsifier: the revision runs but its result is dropped.
fn wired_cycle(
    policy: SelectorPolicy,
    rest: EpistemicState,
    world: &mut World,
    write: bool,
) -> ([Option<ProbeRecipe>; MAX_STEPS + 1], usize) {
    let mut trace = [None; MAX_STEPS + 1];
    let mut rest = rest;
    for (step, slot) in trace.iter_mut().enumerate() {
        match policy.select(world.state(rest)) {
            Some(ProbeRecipe::Revision) => {
                *slot = Some(ProbeRecipe::Revision);
                if write {
                    world.revise();
                } else {
                    // Revise a copy and drop it: the work runs, nothing lands.
                    world.clone().revise();
                }
            }
            Some(recipe) => {
                *slot = Some(recipe);
                rest = after(recipe, rest);
            }
            None => return (trace, step),
        }
    }
    (trace, MAX_STEPS + 1)
}

/// A small world: the horizon holds claims {0, 2}; the encounter proposes
/// {1, 2}, rooted in {1, 2}, and contradicts claim 0.
fn fusion_world() -> World {
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
    }
}

fn main() {
    let start = EpistemicState {
        observations_pending: true,
        frontier_bounded: false,
        local_disagreement: true,
        new_encounter: true,
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
    let mut world = fusion_world();
    let (trace, n) = wired_cycle(SelectorPolicy::CURRENT, start, &mut world, true);
    println!(
        "wired: {:?}, horizon roots {:#b}",
        &trace[..n],
        world.horizon.independent_roots
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
            new_encounter: n,
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
            } else if s.new_encounter {
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
                    + usize::from(s.new_encounter);
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

    /// FAILS IF: the derived fact disagrees with revision's own new-root test.
    #[test]
    fn the_derived_fact_is_revisions_new_root_test() {
        let w = fusion_world();
        let delta = GadamerRevision.revise(&w.horizon, &w.encounter, &ancestry(&w.horizon));
        assert!(w.new_encounter());
        assert_ne!(delta.new_independent_roots, 0);
        let after = World {
            horizon: delta.resulting,
            ..w
        };
        assert!(
            !after.new_encounter(),
            "the resulting horizon has the roots"
        );
    }

    /// FAILS IF: the cycle does not rest after one real revision, or rests
    /// without one. The rest comes from `delta.resulting`, not a stand-in.
    #[test]
    fn a_real_revision_makes_the_cycle_rest() {
        let mut world = fusion_world();
        let before = world.horizon.clone();
        let (trace, n) = wired_cycle(SelectorPolicy::V1, settled(), &mut world, true);
        assert_eq!(&trace[..n], &[Some(Revision)]);
        assert_eq!(world.horizon.independent_roots, 0b111);
        assert_eq!(world.horizon.revision_index, before.revision_index + 1);

        // The full cycle under V1, with the wired fact last.
        let mut world = fusion_world();
        let (trace, n) = wired_cycle(
            SelectorPolicy::V1,
            state(true, false, true, false),
            &mut world,
            true,
        );
        assert_eq!(
            &trace[..n],
            &[
                Some(ObserveFold),
                Some(ProductInterrogation),
                Some(MooreInterrogation),
                Some(Revision)
            ]
        );
    }

    /// FAILS IF: dropping revision's write still lets the cycle rest, i.e.
    /// the rest does not depend on revision's output.
    #[test]
    fn dropping_the_revision_write_never_rests() {
        let mut world = fusion_world();
        let (trace, n) = wired_cycle(SelectorPolicy::V1, settled(), &mut world, false);
        assert_eq!(n, MAX_STEPS + 1, "did not rest");
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
        assert!(world.new_encounter());
        let (trace, n) = wired_cycle(SelectorPolicy::V1, settled(), &mut world, true);
        assert_eq!(&trace[..n], &[Some(Revision)]);
        assert_eq!(world.horizon.independent_roots, 0b011);
    }

    /// FAILS IF: an echo (every root already held) still selects revision.
    #[test]
    fn an_echo_selects_nothing() {
        let mut world = fusion_world();
        world.encounter.independent_roots = 0b001;
        assert!(!world.new_encounter());
        let (_, n) = wired_cycle(SelectorPolicy::V1, settled(), &mut world, true);
        assert_eq!(n, 0);
        assert_eq!(world, {
            let mut w = fusion_world();
            w.encounter.independent_roots = 0b001;
            w
        });
    }

    /// FAILS IF: the same starting world replays to a different path or a
    /// different final horizon.
    #[test]
    fn the_wired_cycle_replays() {
        for policy in [SelectorPolicy::V1, SelectorPolicy::V2] {
            for rest in EpistemicState::all() {
                let (mut a, mut b) = (fusion_world(), fusion_world());
                let ra = wired_cycle(policy, rest, &mut a, true);
                let rb = wired_cycle(policy, rest, &mut b, true);
                assert_eq!(ra, rb, "{policy:?} {rest:?}");
                assert_eq!(a, b, "{policy:?} {rest:?}");
                assert!(ra.1 <= MAX_STEPS, "{policy:?} {rest:?} did not rest");
            }
        }
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
