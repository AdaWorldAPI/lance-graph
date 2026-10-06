//! D-GSO-5 (P5): first recipe quartet over existing primitives.
//!
//! Plan: `.claude/plans/2026-10-06-global-sudoku-replayable-orchestration-v1.md`
//! §12 (non-materialized program law) and §18 P5.
//!
//! Claim under test: four qualitatively different recipes can orchestrate
//! existing primitives without building an instruction vector. A recipe is a
//! `match` arm that calls its primitive directly; nothing is compiled into a
//! `Vec<Op>` and interpreted.
//!
//! | ordinal | scaffold meaning | primitive (unchanged) |
//! |---|---|---|
//! | 0 | observe / fold | `ontology_warrant::Quorum::observe` |
//! | 1 | finite-product interrogation | `Quad8::fold_product` |
//! | 2 | local Palette / Moore interrogation | `Morton8x8::checked_offset` + `PalettePerturbation::hop` over `Register128` |
//! | 3 | counterfactual removal + revision | `GadamerRevision::revise`, run twice |
//!
//! # What this probe does not decide
//!
//! - **The ordinal meanings are a test scaffold, not canon** (plan §18 P5).
//! - **The ordinal is not a shipped recipe ID.** `recipes::RECIPES` already owns
//!   IDs `1..=34` (plan §12). [`ProbeRecipe`] is its own type with no conversion
//!   into that surface; `the_probe_ordinal_is_not_a_shipped_recipe_id` shows why
//!   mixing them would be wrong.
//! - **R3 has no Pearl 2³ projection.** Its causal leg is the counterfactual
//!   removal of the encounter's new roots. Projection and band permission are
//!   P7.
//! - **No selector.** Which recipe runs when is P6.
//!
//! # How "no instruction vector" is checked
//!
//! A counting global allocator records allocations per thread. Every recipe
//! runs inside a measured window and must allocate zero times. Building any
//! `Vec<Op>` before interpreting it would show up as at least one allocation.
//!
//! Run: `cargo run -p cognitive-shader-driver --example recipe_quartet_probe`
//! Tests: `cargo test -p cognitive-shader-driver --example recipe_quartet_probe`

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;

use cognitive_shader_driver::palette_perturbation::{
    PaletteLut, PalettePerturbation, PaletteState, PALETTE_LUT_LEN,
};
use cognitive_shader_driver::quad8::Quad8;
use lance_graph_contract::morton8x8::Morton8x8;
use lance_graph_contract::ontology_warrant::{Quorum, SourceVerdict};
use lance_graph_contract::register128::Register128;
use lance_graph_contract::revision::{
    BasisView, CounterfactualVerdict, EncounterEvidence, EvidenceMask, EvidentialEffect,
    GadamerRevision, InterpretiveHorizon, RevisionKind, RevisionPolicy, RevisionVerdict,
};

// ── allocation counter ─────────────────────────────────────────────────────

/// Counts allocations on the current thread. Test threads run in parallel, so a
/// process-wide counter would mix their allocations; a thread-local one does not.
struct CountingAlloc;

thread_local! {
    static ALLOCATIONS: Cell<usize> = const { Cell::new(0) };
}

// SAFETY: forwards every call to `System` unchanged; it only adds a counter.
unsafe impl GlobalAlloc for CountingAlloc {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let _ = ALLOCATIONS.try_with(|c| c.set(c.get() + 1));
        unsafe { System.alloc(layout) }
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) }
    }
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        let _ = ALLOCATIONS.try_with(|c| c.set(c.get() + 1));
        unsafe { System.realloc(ptr, layout, new_size) }
    }
}

#[global_allocator]
static GLOBAL: CountingAlloc = CountingAlloc;

/// Run `f` and return its result with the number of allocations it made on
/// this thread.
fn counting<T>(f: impl FnOnce() -> T) -> (T, usize) {
    let before = ALLOCATIONS.with(Cell::get);
    let out = f();
    (out, ALLOCATIONS.with(Cell::get) - before)
}

// ── the quartet ────────────────────────────────────────────────────────────

/// A probe recipe ordinal. Its own type: never converted into a shipped
/// `recipes::Recipe` ID (plan §12).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ProbeRecipe {
    ObserveFold,
    ProductInterrogation,
    MooreInterrogation,
    CounterfactualRevision,
}

impl ProbeRecipe {
    const ALL: [Self; 4] = [
        Self::ObserveFold,
        Self::ProductInterrogation,
        Self::MooreInterrogation,
        Self::CounterfactualRevision,
    ];

    /// `0..=3` only. The rest of `0..=63` is unassigned in this probe.
    const fn from_ordinal(ordinal: u8) -> Option<Self> {
        match ordinal {
            0 => Some(Self::ObserveFold),
            1 => Some(Self::ProductInterrogation),
            2 => Some(Self::MooreInterrogation),
            3 => Some(Self::CounterfactualRevision),
            _ => None,
        }
    }

    /// Dispatch. Each arm calls its primitive directly; no schedule is built.
    fn run(self, ctx: &Context<'_>) -> Outcome {
        match self {
            Self::ObserveFold => Outcome::Observed(observe_fold(ctx.evidence)),
            Self::ProductInterrogation => {
                Outcome::Frontier(product_frontier(ctx.quad, ctx.target_sum))
            }
            Self::MooreInterrogation => Outcome::Moore(moore_fold(ctx.field, ctx.lane, ctx.law)),
            Self::CounterfactualRevision => Outcome::Revised(counterfactual_revision(
                &ctx.prior,
                &ctx.encounter,
                &ctx.ancestry,
            )),
        }
    }
}

/// Everything a recipe may read. Borrowed or `Copy`; nothing is owned that a
/// recipe could grow.
struct Context<'a> {
    /// R0: one verdict per consulted source.
    evidence: &'a [SourceVerdict],
    /// R1: the four bit-lanes whose 8×8×8×8 product is interrogated.
    quad: Quad8,
    /// R1: the constraint, `a + b + c + d == target_sum`.
    target_sum: u8,
    /// R2: the resident 16-byte register, read as a 4 × 4 Morton grid.
    field: Register128,
    /// R2: the lane (= its Morton code) whose neighbourhood is folded.
    lane: Morton8x8,
    /// R2: the closed Palette law.
    law: PalettePerturbation<'a>,
    /// R3: the horizon, encounter and ancestry handed to revision.
    prior: InterpretiveHorizon<(), u64>,
    encounter: EncounterEvidence<u64>,
    ancestry: BasisView<u64>,
}

/// The result of one recipe. Every variant is `Copy`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Outcome {
    Observed(Quorum),
    Frontier(Frontier),
    Moore(PaletteState),
    Revised(Revised),
}

/// R1: how many occupied products satisfy the constraint, and the first one in
/// the product's lexicographic order.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Frontier {
    count: u16,
    first: Option<u16>,
}

/// R3: what revision did, whether the counterfactual found the new roots
/// necessary, and the claims the next replay starts from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Revised {
    kind: RevisionKind,
    verdict: RevisionVerdict,
    resulting_claims: u64,
}

/// R0: fold every verdict into one quorum as it is read.
fn observe_fold(evidence: &[SourceVerdict]) -> Quorum {
    evidence
        .iter()
        .fold(Quorum::default(), |q, &v| q.observe(v))
}

/// R1: fold the implicit product; only occupied coordinates are visited.
fn product_frontier(quad: Quad8, target_sum: u8) -> Frontier {
    quad.fold_product(
        Frontier {
            count: 0,
            first: None,
        },
        |acc, addr| {
            let sum: u8 = addr.ordinals().iter().sum();
            if sum == target_sum {
                Frontier {
                    count: acc.count + 1,
                    first: acc.first.or(Some(addr.raw())),
                }
            } else {
                acc
            }
        },
    )
}

/// The eight Moore offsets, in a fixed order.
const MOORE: [(i8, i8); 8] = [
    (-1, -1),
    (0, -1),
    (1, -1),
    (-1, 0),
    (1, 0),
    (-1, 1),
    (0, 1),
    (1, 1),
];

/// Codes below this lie on the 4 × 4 grid (the 2-bit prefix of the tile).
const GRID_CODES: u16 = 16;

/// R2: fold the lane's in-grid Moore neighbours through the Palette law,
/// starting from the lane's own byte. The register byte at index `code` is the
/// cell with that Morton code.
fn moore_fold(field: Register128, lane: Morton8x8, law: PalettePerturbation<'_>) -> PaletteState {
    let local = PaletteState(field.0[lane.code() as usize]);
    MOORE.iter().fold(local, |state, &(dx, dy)| {
        match lane.checked_offset(dx, dy) {
            Some(n) if n.code() < GRID_CODES => {
                law.hop(state, local, PaletteState(field.0[n.code() as usize]))
            }
            _ => state,
        }
    })
}

/// R3: revise, then attack: revise again with the encounter's new independent
/// roots removed. If the effect stops being `IncreaseEligible`, those roots
/// were necessary. The counterfactual only runs when there is something to
/// attack.
fn counterfactual_revision(
    prior: &InterpretiveHorizon<(), u64>,
    encounter: &EncounterEvidence<u64>,
    ancestry: &BasisView<u64>,
) -> Revised {
    let delta = GadamerRevision.revise(prior, encounter, ancestry);
    let counterfactual = if delta.evidential_effect == EvidentialEffect::IncreaseEligible {
        let without = EncounterEvidence {
            independent_roots: encounter
                .independent_roots
                .difference(&delta.new_independent_roots),
            ..encounter.clone()
        };
        let attacked = GadamerRevision.revise(prior, &without, ancestry);
        if attacked.evidential_effect == EvidentialEffect::IncreaseEligible {
            CounterfactualVerdict::Dispensable
        } else {
            CounterfactualVerdict::Necessary
        }
    } else {
        CounterfactualVerdict::NotRun
    };
    Revised {
        kind: delta.kind,
        verdict: RevisionVerdict {
            effect: delta.evidential_effect,
            counterfactual,
        },
        resulting_claims: delta.resulting.projected_claims,
    }
}

// ── fixture ────────────────────────────────────────────────────────────────

/// Two LUTs of the Palette law: relation = XOR, perturbation = wrapping add.
fn tables() -> (Box<[u8; PALETTE_LUT_LEN]>, Box<[u8; PALETTE_LUT_LEN]>) {
    let mut relation = vec![0u8; PALETTE_LUT_LEN];
    let mut perturb = vec![0u8; PALETTE_LUT_LEN];
    for l in 0..=255u8 {
        for r in 0..=255u8 {
            let i = (usize::from(l) << 8) | usize::from(r);
            relation[i] = l ^ r;
            perturb[i] = l.wrapping_add(r);
        }
    }
    (
        relation.into_boxed_slice().try_into().unwrap(),
        perturb.into_boxed_slice().try_into().unwrap(),
    )
}

const EVIDENCE: [SourceVerdict; 6] = [
    SourceVerdict::Corroborates,
    SourceVerdict::Silent,
    SourceVerdict::Corroborates,
    SourceVerdict::Conflicts,
    SourceVerdict::Corroborates,
    SourceVerdict::Silent,
];

fn horizon() -> InterpretiveHorizon<(), u64> {
    use lance_graph_contract::revision::{
        CodebookId, GrammarId, HorizonId, LanguageId, LensId, QuestionId,
    };
    InterpretiveHorizon {
        id: HorizonId(0),
        awareness: (),
        question: QuestionId(1),
        language: LanguageId(1),
        grammar: GrammarId(1),
        codebook: CodebookId(1),
        lens: LensId(1),
        projected_claims: 0b0011,
        independent_roots: 0b0001,
        inherited_roots: 0,
        unresolved_tension: 0,
        revision_index: 0,
    }
}

/// An encounter that keeps claim 0, adds claim 2 and contacts a new root
/// (bit 1) beside one already in ancestry (bit 0).
fn encounter() -> EncounterEvidence<u64> {
    EncounterEvidence {
        proposed_claims: 0b0111,
        independent_roots: 0b0011,
        inherited_roots: 0,
        resistance: 0,
        contradictions: 0,
        affected_parts: 0b0100,
    }
}

fn ancestry() -> BasisView<u64> {
    BasisView {
        ancestry_independent_roots: 0b0001,
        ancestry_derived_roots: 0,
        ancestor_claims: 0b0011,
        closes_cycle: false,
    }
}

fn context<'a>(evidence: &'a [SourceVerdict], law: PalettePerturbation<'a>) -> Context<'a> {
    Context {
        evidence,
        quad: Quad8::new(0b0000_0111, 0b0000_0110, 0b1000_0001, 0b0000_0011),
        target_sum: 4,
        field: Register128([
            3, 141, 59, 26, 53, 58, 97, 93, 238, 46, 26, 43, 38, 32, 79, 50,
        ]),
        lane: Morton8x8::from_xy(1, 2),
        law,
        prior: horizon(),
        encounter: encounter(),
        ancestry: ancestry(),
    }
}

fn main() {
    let (relation, perturb) = tables();
    let law = PalettePerturbation::new(PaletteLut::new(&relation), PaletteLut::new(&perturb));
    let ctx = context(&EVIDENCE, law);
    for (ordinal, recipe) in ProbeRecipe::ALL.iter().enumerate() {
        assert_eq!(ProbeRecipe::from_ordinal(ordinal as u8), Some(*recipe));
        let (out, allocations) = counting(|| recipe.run(&ctx));
        println!("R{ordinal} {recipe:?}: {out:?}  (allocations: {allocations})");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn with_ctx<T>(f: impl FnOnce(&Context<'_>) -> T) -> T {
        let (relation, perturb) = tables();
        let law = PalettePerturbation::new(PaletteLut::new(&relation), PaletteLut::new(&perturb));
        f(&context(&EVIDENCE, law))
    }

    /// FAILS IF: any recipe allocates while it runs, e.g. by collecting an
    /// operation list (or a coordinate or pair list) before executing it.
    #[test]
    fn no_recipe_allocates() {
        with_ctx(|ctx| {
            let (_, warm) = counting(|| Vec::<u8>::with_capacity(1));
            assert_eq!(warm, 1, "the counter must see a plain allocation");
            for recipe in ProbeRecipe::ALL {
                let (_, n) = counting(|| recipe.run(ctx));
                assert_eq!(n, 0, "{recipe:?} allocated {n} times");
            }
        });
    }

    /// FAILS IF: a recipe disagrees with an independent oracle for its own
    /// primitive.
    #[test]
    fn each_recipe_matches_its_oracle() {
        with_ctx(|ctx| {
            // R0: count by hand.
            assert_eq!(
                ProbeRecipe::ObserveFold.run(ctx),
                Outcome::Observed(Quorum::new(3, 2, 1))
            );

            // R1: brute force over all 4096 coordinates, testing membership.
            let b = ctx.quad.bytes();
            let mut count = 0u16;
            let mut first = None;
            for a in 0..8u8 {
                for bb in 0..8u8 {
                    for c in 0..8u8 {
                        for d in 0..8u8 {
                            let live = [a, bb, c, d]
                                .iter()
                                .zip(b)
                                .all(|(&o, byte)| byte & (1 << o) != 0);
                            if live && a + bb + c + d == ctx.target_sum {
                                count += 1;
                                let raw = (u16::from(a) << 9)
                                    | (u16::from(bb) << 6)
                                    | (u16::from(c) << 3)
                                    | u16::from(d);
                                first.get_or_insert(raw);
                            }
                        }
                    }
                }
            }
            assert_eq!(
                ProbeRecipe::ProductInterrogation.run(ctx),
                Outcome::Frontier(Frontier { count, first })
            );
            assert_eq!(count, 3, "pinned frontier size");

            // R2: for every lane, decode coordinates, test the 4 × 4 bounds,
            // build the pair list and run it through `hops`. Edge and corner
            // lanes are where the grid bound matters.
            let mut visits = 0;
            for x in 0..4u8 {
                for y in 0..4u8 {
                    let lane = Morton8x8::from_xy(x, y);
                    let local = PaletteState(ctx.field.0[lane.code() as usize]);
                    let pairs: Vec<_> = MOORE
                        .iter()
                        .filter_map(|&(dx, dy)| {
                            let (nx, ny) =
                                (i16::from(x) + i16::from(dx), i16::from(y) + i16::from(dy));
                            ((0..4).contains(&nx) && (0..4).contains(&ny)).then(|| {
                                let n = Morton8x8::from_xy(nx as u8, ny as u8);
                                (local, PaletteState(ctx.field.0[n.code() as usize]))
                            })
                        })
                        .collect();
                    visits += pairs.len();
                    let at_lane = Context {
                        lane,
                        ..context(ctx.evidence, ctx.law)
                    };
                    assert_eq!(
                        ProbeRecipe::MooreInterrogation.run(&at_lane),
                        Outcome::Moore(ctx.law.hops(local, &pairs)),
                        "lane ({x}, {y})"
                    );
                }
            }
            assert_eq!(visits, 84, "pinned 4 × 4 Moore visit count");
        });
    }

    /// FAILS IF: R3 reports a counterfactual it did not run, or reports the
    /// same verdict whether or not the new root carries the evidence.
    #[test]
    fn the_counterfactual_actually_runs() {
        with_ctx(|ctx| {
            let Outcome::Revised(r) = ProbeRecipe::CounterfactualRevision.run(ctx) else {
                unreachable!()
            };
            assert_eq!(r.kind, RevisionKind::HorizonExpansion);
            assert_eq!(r.verdict.counterfactual, CounterfactualVerdict::Necessary);
            assert!(r.verdict.is_acceptable());
            assert_eq!(r.resulting_claims, 0b0111);

            // An echo: no new root, no resistance, recycled claims. Nothing to
            // attack, so the counterfactual must not claim to have run.
            let echo = EncounterEvidence {
                proposed_claims: 0b0011,
                independent_roots: 0b0001,
                affected_parts: 0,
                ..encounter()
            };
            let r = counterfactual_revision(&ctx.prior, &echo, &ctx.ancestry);
            assert_eq!(r.kind, RevisionKind::Echo);
            assert_eq!(r.verdict.counterfactual, CounterfactualVerdict::NotRun);
            assert!(!r.verdict.is_acceptable());
        });
    }

    /// FAILS IF: one recipe reads another recipe's input. Changing one input
    /// changes exactly one outcome.
    #[test]
    fn each_recipe_reads_only_its_own_input() {
        let (relation, perturb) = tables();
        let law = PalettePerturbation::new(PaletteLut::new(&relation), PaletteLut::new(&perturb));
        let base = context(&EVIDENCE, law);
        let run_all = |c: &Context<'_>| ProbeRecipe::ALL.map(|r| r.run(c));
        let before = run_all(&base);

        let other_evidence = [SourceVerdict::Conflicts; 2];
        let variants: [(usize, Context<'_>); 4] = [
            (
                0,
                Context {
                    evidence: &other_evidence,
                    ..context(&EVIDENCE, law)
                },
            ),
            (
                1,
                Context {
                    target_sum: 5,
                    ..context(&EVIDENCE, law)
                },
            ),
            (
                2,
                Context {
                    field: Register128([7; 16]),
                    ..context(&EVIDENCE, law)
                },
            ),
            (
                3,
                Context {
                    ancestry: BasisView {
                        ancestry_independent_roots: 0b0011,
                        ..ancestry()
                    },
                    ..context(&EVIDENCE, law)
                },
            ),
        ];
        for (changed, ctx) in variants {
            let after = run_all(&ctx);
            for i in 0..4 {
                assert_eq!(
                    before[i] != after[i],
                    i == changed,
                    "changing R{changed}'s input moved R{i}"
                );
            }
        }
    }

    /// FAILS IF: the same context gives a different outcome on a second run.
    #[test]
    fn replay_is_deterministic() {
        with_ctx(|ctx| {
            for recipe in ProbeRecipe::ALL {
                assert_eq!(recipe.run(ctx), recipe.run(ctx), "{recipe:?}");
            }
        });
    }

    /// FAILS IF: the probe ordinal space and the shipped recipe IDs could be
    /// confused. Ordinal 0 has no shipped recipe and ordinal 1 names an
    /// unrelated one, so passing a probe ordinal into `recipes::recipe` would
    /// fail or silently run the wrong tactic.
    #[test]
    fn the_probe_ordinal_is_not_a_shipped_recipe_id() {
        use lance_graph_contract::recipes::recipe;
        assert!(recipe(0).is_none());
        assert!(recipe(1).is_some());
        assert_eq!(
            (0..=63u8)
                .filter(|&o| ProbeRecipe::from_ordinal(o).is_some())
                .count(),
            4
        );
        for (o, r) in ProbeRecipe::ALL.iter().enumerate() {
            assert_eq!(ProbeRecipe::from_ordinal(o as u8), Some(*r));
        }
    }
}
