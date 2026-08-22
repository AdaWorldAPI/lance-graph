// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! `recipe_vocab` — the 34 NARS recipes as `ogar-loco` ops above
//! [`DOMAIN_FLOOR`], with the kanban census as the awareness surface
//! (`D-ACR-9`).
//!
//! # Why this crate and not either of the two it joins
//!
//! [`ogar_loco`] is zero-dep by design (*"defined locally so this crate keeps
//! its plug-and-play posture and takes no substrate dependency"*), and
//! `lance_graph_contract` is zero-dep by charter. **Neither may import the
//! other** — the same rule that hosts `D-ACR-7`'s G10b in `causal-edge`. A
//! vocabulary needs both, so it lives in a consumer that already depends on
//! both, exactly as `medcare-cohorts::loco_vocab` does for the medcare domain.
//!
//! # The ladder IS a program
//!
//! [`recipe_dispatch::ladder`] returns an ordered `Vec<RecipeStep>`; loco
//! executes ordered `(function : value)` calls. That is mechanism, not
//! resemblance — so the lowering here is a mapping, not a translation:
//! recipe id `1..=34` ↔ op byte `0x90..=0xB1`, order preserved from
//! [`recipe_dispatch::dispatch_order`] (ascending rung, then id — the
//! escalation ladder itself).
//!
//! # `6×2×8bit↑n` — why the operand is NOT a flat index
//!
//! A recipe's operand is **where to think**, and "where" ranges over the whole
//! ontology. A 256-entry table cannot address that, and neither can a
//! 256-position `WideFieldMask` — the ergonomics run out long before the
//! population does. loco's own ABI already carries the answer:
//! [`ValueCodebook`] is *"a codebook scoped to the classid prefix the call's
//! own body lives under, **not a vocabulary-wide table**"*.
//!
//! So a recipe operand resolves against a **prefix-scoped** codebook: one byte
//! selects one level of refinement, and depth is bought by stacking levels
//! rather than by widening the byte. That is the canon's *"scale is the next
//! cascade level, never field-widening"* at the operand, and it is the same
//! `6×(u8:u8) = 12` carving [`LaneShape::Pairs`] already names — byte-identical
//! to `contract::attention_facet::AttentionFocusFacet` under `CascadeShape::G6D2`,
//! with neither side importing the other (the measured six-site convergence).
//!
//! One shallow prefix therefore covers an unbounded subtree, so "run recipe R
//! over focus F" is expressible for a whole basin in one call — which is the
//! property a flat index cannot have at any table size.
//!
//! # What this module does NOT claim
//!
//! It does not execute a recipe, does not write a `ReasoningBand` (the only
//! writer stays `with_reasoning_band`), and does not resolve the codebook it
//! declares — declaring one is advisory metadata by loco's own contract, and
//! the basin that owns the prefix owns the resolution. The revision window
//! (`−220..0` vs `−550..−200`) is an open question in
//! `alpha-channel-rung-overlay-v1.md` §3f and is **not** decided here; nothing
//! below depends on it.

use lance_graph_contract::kanban::KanbanColumn;
use lance_graph_contract::recipe_dispatch::{dispatch_order, inference, rung, RecipeInference};
use lance_graph_contract::recipes::recipe;
use ogar_loco::vocabulary::{ValueCodebook, Vocabulary};
use ogar_loco::{FnIndex, DOMAIN_FLOOR};

/// Extract `N` from `dispatch_order`'s own return type `[u8; N]`.
///
/// `dispatch_order()` is not `const`, so its VALUE cannot be read here — but
/// its TYPE can, and the type is where the count lives. This keeps the count
/// derived rather than restated: a 35th recipe changes the signature and this
/// const follows, with no literal to forget.
const fn count_of<const N: usize>(_: fn() -> [u8; N]) -> usize {
    N
}

/// How many recipes the ladder carries — derived from the shipped dispatch
/// order's arity, never a second literal.
pub const RECIPE_COUNT: usize = count_of(dispatch_order);

/// The first op byte a recipe occupies — the domain floor itself.
pub const RECIPE_OP_BASE: u8 = DOMAIN_FLOOR;

/// One past the last recipe op. `0x90 + 34 = 0xB2`, comfortably inside the
/// domain range, so the 34 need no second block.
pub const RECIPE_OP_END: u8 = RECIPE_OP_BASE + RECIPE_COUNT as u8;

/// The basin-local codebook a recipe's operand resolves against: an **attention
/// prefix**, not a flat id. See the module docs' `6×2×8bit↑n` section.
pub const CB_ATTENTION_FOCUS: u8 = 1;

/// Op byte for a recipe id (`1..=34`); `None` outside that range — a refusal,
/// never a clamp.
#[must_use]
pub fn op_of(recipe_id: u8) -> Option<FnIndex> {
    (1..=RECIPE_COUNT as u8)
        .contains(&recipe_id)
        .then(|| FnIndex(RECIPE_OP_BASE + recipe_id - 1))
}

/// Recipe id for an op byte; `None` for any byte outside the minted range —
/// including every shared-core byte below the floor.
#[must_use]
pub fn recipe_of(f: FnIndex) -> Option<u8> {
    (RECIPE_OP_BASE..RECIPE_OP_END)
        .contains(&f.0)
        .then(|| f.0 - RECIPE_OP_BASE + 1)
}

/// The escalation ladder lowered to a loco call sequence — the same order
/// [`dispatch_order`] returns (ascending rung, then id).
#[must_use]
pub fn ladder_program() -> Vec<FnIndex> {
    dispatch_order().into_iter().filter_map(op_of).collect()
}

/// The 34 recipes as a loco vocabulary.
///
/// Zero-sized: every answer is a property of the shipped recipe tables, so the
/// vocabulary is a TABLE OF SPECS and needs no resident data — the same posture
/// `medcare-cohorts::loco_vocab::MedcareVocabulary` documents.
#[derive(Debug, Clone, Copy, Default)]
pub struct RecipeVocabulary;

impl Vocabulary for RecipeVocabulary {
    /// One operand: **the focus to think over**. Not two — the recipe's own
    /// parameters are properties of the recipe (a table lookup by id), never
    /// stack operands, so a second operand would be a value with no source.
    fn domain_stack_arity(&self, f: FnIndex) -> Option<u8> {
        recipe_of(f).map(|_| 1)
    }

    /// Zero. A recipe is not control flow: the ladder is a SEQUENCE, and
    /// sequencing is the program's shape, not a branch inside one call. This
    /// keeps every recipe inside [`LaneShape::Pairs`](ogar_loco::LaneShape)
    /// — the same 12-byte carving the attention facet uses.
    fn domain_body_refs(&self, _f: FnIndex) -> u8 {
        0
    }

    /// A recipe yields a thought field, so it pushes. Declaring the column is
    /// what makes a recipe body statement-segmentable; leaving it `None` would
    /// make segmentation refuse every ladder.
    fn domain_pushes_result(&self, f: FnIndex) -> Option<bool> {
        recipe_of(f).map(|_| true)
    }

    /// The recipe's short code (`"RCR"`, `"SMAD"`, …) — read from the shipped
    /// table, never restated here.
    fn domain_name(&self, f: FnIndex) -> Option<&'static str> {
        recipe(recipe_of(f)?).map(|r| r.code)
    }

    /// Every recipe operand resolves against the prefix-scoped attention
    /// codebook. This is the declaration that the operand is an ADDRESS PREFIX
    /// rather than a literal or a body reference — the module docs' whole
    /// `6×2×8bit↑n` point, stated where a renderer/fuzzer/oracle reads it.
    fn domain_value_codebook(&self, f: FnIndex) -> Option<ValueCodebook> {
        recipe_of(f).map(|_| ValueCodebook {
            id: CB_ATTENTION_FOCUS,
            name: "attention-focus-prefix",
        })
    }
}

/// **The awareness surface** — how deep the ladder may go, read from the
/// board's own phase distribution.
///
/// The operator's composition names `kanban_actor`'s census as the awareness
/// surface. This takes it by VALUE (a count per [`KanbanColumn`]) rather than
/// importing `lance_graph_supervisor::PhaseCensus`: the census is one `&self`
/// pass over the fleet, and lifting six counts across the seam costs nothing,
/// while a dependency from this crate onto the runtime supervisor would invert
/// the layering (`lance-graph-ogar` is an activation crate, not a runtime).
///
/// # The policy, and it is labelled a policy
///
/// `recipe_loci`'s ruling is *Maslow-monotone climb, elevate on sustained
/// BLOCK*. A census has no history, so "sustained" is not observable from one
/// pass; what IS observable is **non-closure**: cards accumulated in
/// `CognitiveWork` that have not reached `Commit`. So:
///
/// ```text
/// max_rung = floor + (CognitiveWork − Commit)     clamped to floor..=9
/// ```
///
/// where `floor` is the ladder's own SHALLOWEST rung, measured from
/// [`dispatch_order`] — **2** today (base 1 + Deduction's +1), and derived
/// rather than restated. A first draft hardcoded `1` and the never-empty test
/// caught it immediately: no recipe fires at rung 1, so a closing board was
/// admitted nothing — "the system may not think", the exact state the census
/// must never be able to express.
///
/// A board that is closing work admits only shallow, cheap recipes; a board
/// that is stuck admits the deep ones. This is a **first policy**, not a canon
/// claim — the census makes it replaceable without touching the vocabulary,
/// and its inertness is pinned in both directions by the tests (raising the
/// stuck count admits strictly more; a committing board admits strictly less).
#[must_use]
pub fn max_rung_admitted<F>(census: F) -> u8
where
    F: Fn(KanbanColumn) -> usize,
{
    // The ladder's own shallowest rung — never a literal, so a re-tiering of
    // the recipes moves the floor with it.
    let floor = dispatch_order().into_iter().map(rung).min().unwrap_or(1);
    let stuck = census(KanbanColumn::CognitiveWork).saturating_sub(census(KanbanColumn::Commit));
    let depth = usize::from(floor).saturating_add(stuck);
    u8::try_from(depth.min(9)).unwrap_or(9)
}

/// The ladder the census admits, in dispatch order — the awareness surface
/// applied to the program.
///
/// Never empty for any census: rung 1 is always admitted, so a board in any
/// state still thinks. An empty return would mean "the system may not think",
/// which no phase distribution should be able to say.
#[must_use]
pub fn admitted_program<F>(census: F) -> Vec<FnIndex>
where
    F: Fn(KanbanColumn) -> usize,
{
    let ceiling = max_rung_admitted(census);
    dispatch_order()
        .into_iter()
        .filter(|&id| rung(id) <= ceiling)
        .filter_map(op_of)
        .collect()
}

/// The inference a minted op embodies — the bridge a caller needs to stamp a
/// `CausalEdge64` mantissa for the step it just ran. Exposed so a consumer
/// need not re-derive the id↔op mapping to reach it.
#[must_use]
pub fn inference_of(f: FnIndex) -> Option<RecipeInference> {
    recipe_of(f).map(inference)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ogar_loco::vocabulary::conformance;

    /// A census fixture: explicit per-column counts, so no test depends on a
    /// column ordering it did not state.
    fn census(work: usize, commit: usize) -> impl Fn(KanbanColumn) -> usize {
        move |c| match c {
            KanbanColumn::CognitiveWork => work,
            KanbanColumn::Commit => commit,
            _ => 0,
        }
    }

    /// The 34 map bijectively onto a contiguous domain block, and **nothing
    /// else does** — the can-fire and can-stay-silent halves of the mapping.
    #[test]
    fn the_thirty_four_recipes_map_bijectively_above_the_floor() {
        let mut seen = std::collections::BTreeSet::new();
        for id in 1..=RECIPE_COUNT as u8 {
            let f = op_of(id).expect("every recipe id mints an op");
            assert!(f.0 >= DOMAIN_FLOOR, "{f:?} must be above the shared core");
            assert_eq!(recipe_of(f), Some(id), "round-trip");
            assert!(seen.insert(f.0), "{f:?} minted twice");
        }
        assert_eq!(seen.len(), RECIPE_COUNT, "34 distinct ops");

        // Silence: outside the range, refused rather than clamped.
        assert_eq!(op_of(0), None, "there is no recipe 0");
        assert_eq!(op_of(RECIPE_COUNT as u8 + 1), None, "nor a 35th");
        assert_eq!(
            recipe_of(FnIndex(DOMAIN_FLOOR - 1)),
            None,
            "below the floor"
        );
        assert_eq!(recipe_of(FnIndex(RECIPE_OP_END)), None, "past the block");
        assert_eq!(
            recipe_of(FnIndex::ADD),
            None,
            "a shared-core byte is not a recipe"
        );
    }

    /// The vocabulary must pass loco's own conformance gate — the mechanical
    /// enforcement every vocabulary crate is required to run.
    #[test]
    fn the_vocabulary_conforms_to_the_shared_core() {
        let checked = conformance::validate(RecipeVocabulary).expect("conforms");
        // …and the shared half still answers as the core does, through the
        // validated table (the anti-drift property, sampled non-trivially).
        assert_eq!(checked.stack_arity(FnIndex::ADD), Some(2));
        assert_eq!(checked.body_refs(FnIndex::IF_ELSE), 2);
        // The domain half is ours.
        let f = op_of(1).unwrap();
        assert_eq!(checked.stack_arity(f), Some(1), "one operand: the focus");
        assert_eq!(checked.body_refs(f), 0, "a recipe is not control flow");
    }

    /// **The `6×2×8bit↑n` claim, as a test.** A recipe operand declares a
    /// PREFIX-SCOPED codebook; a shared-core operand declares none. If a
    /// recipe's operand were a flat vocabulary-wide index, this column would
    /// be `None` and the address space would be capped at 256.
    #[test]
    fn a_recipe_operand_is_a_prefix_codebook_and_a_core_operand_is_not() {
        let v = RecipeVocabulary;
        for id in 1..=RECIPE_COUNT as u8 {
            let cb = v
                .value_codebook(op_of(id).unwrap())
                .expect("every recipe operand resolves against a codebook");
            assert_eq!(cb.id, CB_ATTENTION_FOCUS);
        }
        // The silence half, over non-trivial core bytes: the computational core
        // has no basin-scoped operands, and claiming one there is a drift the
        // conformance check rejects outright.
        for f in [
            FnIndex::ADD,
            FnIndex::IF_ELSE,
            FnIndex::REPEAT,
            FnIndex::TEXT,
        ] {
            assert_eq!(
                v.value_codebook(f),
                None,
                "{f:?} carries a literal, not a prefix"
            );
        }
        // …and an unminted domain byte declares nothing either.
        assert_eq!(v.value_codebook(FnIndex(RECIPE_OP_END)), None);
    }

    /// The ladder lowers to a program in dispatch order — `ladder()`'s own
    /// order, recovered through the op mapping.
    #[test]
    fn the_ladder_lowers_to_a_program_in_dispatch_order() {
        let prog = ladder_program();
        assert_eq!(prog.len(), RECIPE_COUNT, "every recipe lowers");
        let recovered: Vec<u8> = prog.iter().copied().filter_map(recipe_of).collect();
        assert_eq!(
            recovered,
            dispatch_order().to_vec(),
            "the program IS the dispatch order, not a re-sort"
        );
        // Anti-vacuity: dispatch order is not merely ascending id, so the
        // assertion above is testing an ORDER and not an identity.
        let ascending: Vec<u8> = (1..=RECIPE_COUNT as u8).collect();
        assert_ne!(
            recovered, ascending,
            "dispatch order must differ from id order, or this proves nothing"
        );
        // …and it is genuinely rung-ordered.
        let rungs: Vec<u8> = recovered.iter().map(|&id| rung(id)).collect();
        assert!(rungs.windows(2).all(|w| w[0] <= w[1]), "ascending rung");
    }

    /// **The awareness surface fires and stays silent.** A stuck board admits
    /// strictly more of the ladder than a closing one, and no board admits
    /// nothing.
    #[test]
    fn the_census_admits_deeper_when_the_board_is_stuck() {
        // Closing: everything committed, nothing accumulating.
        let closing = admitted_program(census(2, 2));
        // Stuck: real work piled up in CognitiveWork with nothing committed.
        let stuck = admitted_program(census(9, 0));

        // The floor is the ladder's own minimum rung (2 today), derived — so
        // this asserts the RELATION, not the number.
        let floor = dispatch_order().into_iter().map(rung).min().unwrap();
        assert_eq!(
            max_rung_admitted(census(2, 2)),
            floor,
            "a closing board sits exactly on the ladder's floor"
        );
        assert!(
            max_rung_admitted(census(9, 0)) > floor,
            "a stuck board escalates"
        );

        assert!(
            stuck.len() > closing.len(),
            "escalation must admit strictly more: {} vs {}",
            stuck.len(),
            closing.len()
        );
        // Anti-vacuity on BOTH ends: the shallow board is not empty (the system
        // may always think) and the stuck board is not everything-by-default.
        assert!(
            !closing.is_empty(),
            "the floor band is always admitted — no census may say 'do not think'"
        );
        assert!(
            closing.len() * 3 < RECIPE_COUNT * 2,
            "a closing board must exclude a non-trivial share ({} of {})",
            closing.len(),
            RECIPE_COUNT
        );
        // The knob is not decoration: raising the ceiling admits more, and the
        // maximum admits the whole ladder.
        assert_eq!(admitted_program(census(usize::MAX, 0)).len(), RECIPE_COUNT);
    }

    /// The inference bridge resolves for minted ops and refuses elsewhere —
    /// so a caller cannot stamp a mantissa from a byte that names no recipe.
    #[test]
    fn the_inference_bridge_answers_only_for_minted_ops() {
        assert!(inference_of(op_of(31).unwrap()) == Some(RecipeInference::Counterfactual));
        assert_eq!(inference_of(FnIndex::ADD), None);
        assert_eq!(inference_of(FnIndex(RECIPE_OP_END)), None);
    }
}
