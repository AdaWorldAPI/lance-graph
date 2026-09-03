//! **D-DCR-2 (W2) — question-scoped candidate masking.** `dismech-causal-replay-v1` §3 W2.
//!
//! # This is ONE of three kinds of Mengenlehre — the third (operator, 2026-09-01)
//!
//! *"Fox mammal wombat whale should not simply eliminate."* A whale disagreeing
//! with the typical mammal features is **information about the field**, not
//! grounds to remove the whale from the mammals. So set difference is NOT the
//! substrate's primary operation, and this module deliberately claims only the
//! narrowest of the three roles:
//!
//! | # | kind | this module |
//! |---|---|---|
//! | 1 | **propagation / the field map** — propagate precision about a knowledge stage over the WHOLE field; agreement, disagreement, support chains and MISSING LINKS written into the HHTL nodes; the boring rails (`is_a`/`part_of`) lifted into a causality graph with propagated node edges. **This is what explains Mengenlehre.** | **NO** — unbuilt, `D-DCR-2b` |
//! | 2 | **threshold elimination** — a READING of that map at a measured threshold (Shannon proprioception, EWA sandwich, Hambly, Lyons) | **NO** — belongs with W4's Σ / entropy machinery |
//! | 3 | **question masking** — scope to ONE case, patient, question; logically distinct from any generalization | **YES — this module, and only this** |
//!
//! The ordering is the ruling: kind 1 writes the field, kind 2 reads a
//! threshold off it, kind 3 masks it to one question. A mask is the LAST step,
//! never the substrate.
//!
//! **What that forbids here, concretely.** Nothing in this module may be used
//! to conclude anything general. A candidate absent from the surviving set is
//! absent *for this question*; it has not been refuted, demoted, or learned
//! about. The masking is scoped, reversible by asking a different question,
//! and writes nothing back.
//!
//! Differential evaluation as SET ARITHMETIC over the recorded chains: which
//! recorded trajectories are still consistent with the evidence seen so far,
//! **for the question being asked**.
//! Supporting evidence intersects, refuting evidence subtracts. The carrier is
//! the shipped [`EvidenceMask`], so a candidate set is a bitmask over chain
//! ids and every operation is a word-wise AND / AND-NOT — no allocation, no
//! per-candidate branch.
//!
//! ```text
//! candidates := { recorded chain ids }
//! Support  ⇒  candidates ∩= item.candidates
//! Refute   ⇒  candidates ∖= item.candidates
//! ```
//!
//! # Only TWO of the four stances are set operations, and that is the design
//!
//! [`Supports`] has four values. Just `Support` and `Refute` move the set:
//!
//! | stance | set effect | why |
//! |---|---|---|
//! | `Support` | `∩` | the source asserts the relation holds; candidates that do not carry it are out |
//! | `Refute` | `∖` | the source asserts it does NOT hold; candidates that carry it are out |
//! | `Partial` | **none** | weaker than support by the source's own vocabulary. Intersecting on it would eliminate every candidate outside a partially-supported item — full-strength elimination bought with partial evidence |
//! | `NoEvidence` | **none** | the source asserts it has no evidence. An absence of evidence must never narrow anything |
//!
//! This is deliberate restraint, not an unfinished switch. Elimination is
//! irreversible within a run — a candidate removed is not reconsidered — so
//! the bar for removing one is that the source actually asserted something.
//! `Partial` remains useful as a RANKING signal (a later wave's business);
//! it is inert here because ranking and elimination are different operations
//! and conflating them is how a set quietly loses the right answer.
//!
//! Cf. `dismech_evidence`'s own fail-closed parsing rule: `UNKNOWN` is a value
//! the corpus asserts, so it may never be minted from a parse failure. Same
//! instinct, one layer up — `NoEvidence` is an assertion of absence, and an
//! assertion of absence is not a licence to cut.
//!
//! # Synthetic fixtures only (D-DCR-6)
//!
//! The corpus bake and live evidence stay consumer-side. Everything here is
//! exercised against masks built in-crate.

use lance_graph_contract::dismech_evidence::Supports;
use lance_graph_contract::revision::EvidenceMask;

/// One evidence item as the set algebra sees it: a stance, and the candidates
/// it names.
///
/// `candidates` is "the chains this item is ABOUT" — for a `Support` item the
/// chains consistent with it, for a `Refute` item the chains it rules out. The
/// stance decides which of those two readings applies; the mask itself is
/// stance-agnostic, which is what keeps the item constructible before the
/// stance is known.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EvidenceItem<M: EvidenceMask> {
    /// The source's own stance — never inferred from confidence or count.
    pub stance: Supports,
    /// The candidate chains this item names.
    pub candidates: M,
}

impl<M: EvidenceMask> EvidenceItem<M> {
    /// Construct an item.
    pub fn new(stance: Supports, candidates: M) -> Self {
        Self { stance, candidates }
    }

    /// Whether this item can move a candidate set at all.
    ///
    /// False for `Partial` and `NoEvidence` — see the module doc. Exposed so a
    /// caller can report "N items seen, K of them decisive" rather than
    /// silently discarding the rest.
    #[must_use]
    pub fn is_decisive(&self) -> bool {
        matches!(self.stance, Supports::Support | Supports::Refute)
    }
}

/// Apply one evidence item to a candidate set.
///
/// Returns the narrowed set. `Partial` and `NoEvidence` return it unchanged —
/// deliberately, per the module doc.
#[must_use]
pub fn apply<M: EvidenceMask>(candidates: &M, item: &EvidenceItem<M>) -> M {
    match item.stance {
        Supports::Support => candidates.intersection(&item.candidates),
        Supports::Refute => candidates.difference(&item.candidates),
        Supports::Partial | Supports::NoEvidence => candidates.clone(),
    }
}

/// The outcome of evaluating a run of evidence against a candidate set.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Evaluation<M: EvidenceMask> {
    /// The surviving candidates.
    pub candidates: M,
    /// How many items were applied in total.
    pub seen: usize,
    /// How many of them were `Support` or `Refute` — the only stances that can
    /// move the set. Reported rather than inferred, so "the set did not move"
    /// and "nothing decisive arrived" stay distinguishable.
    pub decisive: usize,
    /// How many items actually CHANGED the set.
    ///
    /// Strictly `<= decisive`: a decisive item can still be redundant (it
    /// names a superset of what already survives). That gap is the signal a
    /// frontier scheduler wants — an item that changes nothing taught nothing
    /// — and it is measured here rather than re-derived later.
    pub narrowing: usize,
}

/// Fold a run of evidence over a candidate set.
///
/// Order-independent for the set itself: `∩` and `∖` over the same universe
/// commute here because each item's mask is fixed, so the surviving set is the
/// same whatever order the items arrive in. `narrowing` is NOT order-
/// independent, and that is honest rather than a defect: whether an item
/// teaches anything genuinely depends on what was already known when it
/// arrived.
#[must_use]
pub fn evaluate<M: EvidenceMask>(initial: &M, items: &[EvidenceItem<M>]) -> Evaluation<M> {
    let mut candidates = initial.clone();
    let mut decisive = 0usize;
    let mut narrowing = 0usize;
    for item in items {
        if item.is_decisive() {
            decisive += 1;
        }
        let next = apply(&candidates, item);
        if next != candidates {
            narrowing += 1;
        }
        candidates = next;
    }
    Evaluation {
        candidates,
        seen: items.len(),
        decisive,
        narrowing,
    }
}

/// Whether `item` would change `candidates` — the "does this teach anything"
/// question, without paying for the narrowed set.
///
/// This is the primitive W5's frontier scheduling needs (which observation is
/// worth making next), surfaced now because it is one comparison and belongs
/// beside the algebra it reasons about rather than beside the scheduler.
#[must_use]
pub fn is_informative<M: EvidenceMask>(candidates: &M, item: &EvidenceItem<M>) -> bool {
    apply(candidates, item) != *candidates
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 256 candidate chains — wide enough that the `[u64; N]` carrier is
    /// genuinely multi-word, so a single-word implementation could not pass.
    type Mask = [u64; 4];

    fn mask_of(ids: &[usize]) -> Mask {
        let mut m = Mask::empty();
        for &i in ids {
            m[i / 64] |= 1u64 << (i % 64);
        }
        m
    }

    fn count(m: &Mask) -> usize {
        m.iter().map(|w| w.count_ones() as usize).sum()
    }

    /// Deterministic fixture PRNG. A set-arithmetic test seeded from the clock
    /// could not be re-run against a failure.
    struct Lcg(u64);
    impl Lcg {
        fn next(&mut self) -> u64 {
            self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1);
            self.0
        }
        fn below(&mut self, n: usize) -> usize {
            (self.next() >> 33) as usize % n
        }
    }

    /// A synthetic corpus (D-DCR-6: no bake, no live evidence): 256 chains,
    /// and one evidence item that names roughly a third of them.
    fn corpus(rng: &mut Lcg, universe: usize, share: usize) -> Mask {
        let mut ids = Vec::new();
        for i in 0..universe {
            if rng.below(share) == 0 {
                ids.push(i);
            }
        }
        mask_of(&ids)
    }

    /// GATE — anti-vacuity. A filter that "filters" without excluding anything
    /// carries no information. The plan pins `kept * 3 < total`.
    #[test]
    fn supporting_evidence_excludes_a_non_trivial_share_of_the_corpus() {
        let mut rng = Lcg(0x0B1E_5E7A_11C0_DE01);
        let all = mask_of(&(0..256).collect::<Vec<_>>());
        assert_eq!(count(&all), 256, "the universe must be full to start");

        let named = corpus(&mut rng, 256, 3);
        let out = evaluate(&all, &[EvidenceItem::new(Supports::Support, named)]);

        let kept = count(&out.candidates);
        assert!(
            kept * 3 < 256,
            "support kept {kept}/256 — not a non-trivial exclusion",
        );
        // ...and it must not have excluded EVERYTHING either, which would pass
        // the bound above while being just as uninformative.
        assert!(kept > 0, "support eliminated the entire universe");
        assert_eq!(out.decisive, 1);
        assert_eq!(out.narrowing, 1);
    }

    /// GATE — two-sided discrimination. A discriminating item must split the
    /// set; a REDUNDANT one must not shrink it further. Both halves on the
    /// same set, so the difference is the item and nothing else.
    #[test]
    fn a_discriminating_item_splits_the_set_and_a_redundant_one_does_not() {
        let all = mask_of(&(0..256).collect::<Vec<_>>());

        // Discriminating: names half.
        let half = mask_of(&(0..128).collect::<Vec<_>>());
        let after = evaluate(&all, &[EvidenceItem::new(Supports::Support, half)]);
        assert_eq!(count(&after.candidates), 128, "must split the universe");
        assert_eq!(after.narrowing, 1);

        // Redundant: a SUPERSET of what already survives. Decisive by stance,
        // but it teaches nothing — and the two must stay distinguishable.
        let superset = mask_of(&(0..200).collect::<Vec<_>>());
        let redundant = EvidenceItem::new(Supports::Support, superset);
        assert!(
            !is_informative(&after.candidates, &redundant),
            "an item naming a superset of the survivors cannot narrow",
        );
        let again = evaluate(&after.candidates, &[redundant]);
        assert_eq!(
            count(&again.candidates),
            128,
            "a redundant item must not shrink the set further",
        );
        assert_eq!(again.decisive, 1, "it IS decisive by stance");
        assert_eq!(again.narrowing, 0, "...and still taught nothing");
    }

    /// GATE — the refute path, can-fire. `∖` must remove exactly what the item
    /// names, and nothing else.
    #[test]
    fn refuting_evidence_removes_exactly_what_it_names() {
        let all = mask_of(&(0..256).collect::<Vec<_>>());
        let named = mask_of(&[3, 64, 65, 199, 255]);

        let out = evaluate(&all, &[EvidenceItem::new(Supports::Refute, named)]);
        assert_eq!(count(&out.candidates), 251);
        for id in [3usize, 64, 65, 199, 255] {
            assert_eq!(
                out.candidates[id / 64] & (1u64 << (id % 64)),
                0,
                "candidate {id} was refuted and must be gone",
            );
        }
        // The neighbours of each removed id must survive — an off-by-one in
        // the word/bit split would take one of these with it, and the ids
        // above deliberately straddle the 64-bit word boundary.
        for id in [2usize, 4, 63, 66, 198, 200, 254] {
            assert_ne!(
                out.candidates[id / 64] & (1u64 << (id % 64)),
                0,
                "candidate {id} was not named and must survive",
            );
        }
        assert_eq!(out.narrowing, 1);
    }

    /// GATE — the refute path, can-stay-SILENT. On a NON-TRIVIAL input: an
    /// item naming only candidates that are already gone must change nothing.
    /// (An empty-mask silence case would prove the code handles emptiness, not
    /// that it discriminates.)
    #[test]
    fn refuting_what_is_already_gone_changes_nothing() {
        let all = mask_of(&(0..256).collect::<Vec<_>>());
        let first = evaluate(
            &all,
            &[EvidenceItem::new(Supports::Refute, mask_of(&[10, 20, 30]))],
        );
        assert_eq!(
            count(&first.candidates),
            253,
            "the fixture must have removed something"
        );

        let already_gone = EvidenceItem::new(Supports::Refute, mask_of(&[10, 20]));
        assert!(!is_informative(&first.candidates, &already_gone));
        let second = evaluate(&first.candidates, &[already_gone]);
        assert_eq!(second.candidates, first.candidates);
        assert_eq!(second.decisive, 1);
        assert_eq!(second.narrowing, 0);
    }

    /// GATE — the two inert stances are inert, on inputs that WOULD move the
    /// set under `Support` / `Refute`. That pairing is the whole test: the
    /// same mask, four stances, two that move it and two that must not.
    #[test]
    fn partial_and_no_evidence_never_move_the_set_though_the_same_mask_would() {
        let all = mask_of(&(0..256).collect::<Vec<_>>());
        let named = mask_of(&(0..64).collect::<Vec<_>>());

        // Proof the mask is not inert: the decisive stances DO move it.
        assert_eq!(
            count(&apply(&all, &EvidenceItem::new(Supports::Support, named))),
            64,
        );
        assert_eq!(
            count(&apply(&all, &EvidenceItem::new(Supports::Refute, named))),
            192,
        );

        // ...and the restrained ones do not.
        for stance in [Supports::Partial, Supports::NoEvidence] {
            let item = EvidenceItem::new(stance, named);
            assert!(!item.is_decisive(), "{stance:?} must not be decisive");
            assert_eq!(apply(&all, &item), all, "{stance:?} moved the set");
            assert!(!is_informative(&all, &item));
        }

        // Reported, not silently dropped: a run of inert items is visible as
        // "seen but not decisive" rather than looking like nothing happened.
        let run = evaluate(
            &all,
            &[
                EvidenceItem::new(Supports::Partial, named),
                EvidenceItem::new(Supports::NoEvidence, named),
            ],
        );
        assert_eq!((run.seen, run.decisive, run.narrowing), (2, 0, 0));
        assert_eq!(run.candidates, all);
    }

    /// The surviving SET does not depend on the order evidence arrives in,
    /// while `narrowing` legitimately does. Both halves asserted, because the
    /// second is the one a reader would assume away.
    #[test]
    fn the_surviving_set_is_order_independent_but_narrowing_is_not() {
        let all = mask_of(&(0..256).collect::<Vec<_>>());
        let a = EvidenceItem::new(Supports::Support, mask_of(&(0..128).collect::<Vec<_>>()));
        let b = EvidenceItem::new(Supports::Support, mask_of(&(0..64).collect::<Vec<_>>()));

        let ab = evaluate(&all, &[a.clone(), b.clone()]);
        let ba = evaluate(&all, &[b, a]);
        assert_eq!(
            ab.candidates, ba.candidates,
            "the set must not depend on order"
        );
        assert_eq!(count(&ab.candidates), 64);

        // `b` narrows after `a`; `a` teaches nothing after `b` (it names a
        // superset of what already survives). Same two items, different counts.
        assert_eq!(ab.narrowing, 2);
        assert_eq!(ba.narrowing, 1);
    }
}
