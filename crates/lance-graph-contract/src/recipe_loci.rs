// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! `recipe_loci` — wire all 34 recipes to the **24-dimension causal-witness
//! organ**, closing the #780 Axis-B gap on the DISPATCH path.
//!
//! # What #780 left open
//!
//! [`recipe_dispatch::ladder`](crate::recipe_dispatch::ladder) already sweeps all
//! 34 recipes in rung order — but it gates each one on the **8-field scalar
//! proxy** ([`ThoughtField`](crate::recipe_kernels::ThoughtField): `sd`,
//! `free_energy`, `dissonance`, …). The #780 audit's Axis B measured exactly this:
//! *34/34 kernels read ONLY the scalar proxy, 0 read a real organ*
//! (`E-RECIPE-SUBSTRATE-WIRING-1`). #780 wired the ORGANS into the ctx
//! ([`recipe_substrate::SubstrateView`](crate::recipe_substrate)) but the
//! DISPATCH gate stayed scalar. This module supplies the missing half: a
//! **loci gate** keyed to the real [`CausalWitnessFacet`](crate::causal_witness)
//! 24-dimension reading.
//!
//! # The 24 dimensions × the rung-level walk
//!
//! Each recipe declares WHICH of the 16 named [`Locus`](crate::causal_witness::Locus)
//! dimensions it consumes ([`required_loci`]) — grounded in the recipe's own
//! documented [`substrate`](crate::recipes::Recipe::substrate) + inference. The
//! **rung-level walk** ([`loci_ladder`]) then visits the 34 in
//! [`dispatch_order`](crate::recipe_dispatch::dispatch_order) (ascending rung)
//! and fires a recipe only when **every required locus is BOUND** (nonzero offset
//! — the dimension is placed in the `±8` `temporal.rs` window). An UNBOUND
//! required locus ([`loci_disqualifier`]) is the 24-dimension analog of
//! [`nan_disqualifier`](crate::recipe_dispatch::nan_disqualifier): a recipe never
//! fires on a dimension the row could not ground.
//!
//! # Door C — the organ gate (complements the two shipped doors)
//!
//! `docs/NARS_RECIPES_DISPATCH.md` §7 already reaches all 34 by TWO doors:
//! **Door A** the style→mechanism fan (`recipes_for` over the 4 covering styles —
//! the wide door, all 34), **Door B** the surprise selector
//! ([`materialize::select_tactic`](crate::materialize::select_tactic) — the
//! narrow door, 8; `E-RECIPE-SELECTOR-REACHABILITY-1`). Both gate on
//! **style / mechanism / scalar surprise** — NEITHER reads the real
//! causal-witness organ. That is precisely #780's Axis-B gap.
//!
//! This module is **Door C — the organ gate**: a recipe is reached by
//! GROUNDING its 24-dimension loci in the row's live
//! [`CausalWitnessFacet`](crate::causal_witness), not by style coverage or a
//! surprise-band argmax. It is the dispatch path that actually READS the
//! substrate the #780 audit found unread. [`reachable`] returns the
//! loci-reachable set; the counterfactual ICR #31 fires unshadowed at the
//! deepest rung once its `Kausal` + `Contradiction` + S/P/O loci are all placed,
//! and a CrossTier recipe fires exactly when its basin/temporal/meaning loci are
//! bound — the organ decides, not the style label. (Door C does not *raise* the
//! reachable count above Door A's 34; it makes reachability CONTINGENT ON THE
//! REAL ORGAN — the honest #780-closing contribution, not a reachability claim.)
//!
//! # The Maslow pyramid of cognition (the shipped rung vocabulary)
//!
//! The rung a recipe fires at IS a level of the operator-ruled **Maslow pyramid
//! of cognition** — the shipped 10-name
//! [`RungLevel`](crate::cognitive_shader::RungLevel) `Surface..Transcendent`
//! (`docs/NARS_RECIPES_DISPATCH.md` §7 table, `E-FOVEATED-AWARENESS-1`), climbed
//! by the shipped [`RungElevator`](crate::cognitive_shader::RungElevator)
//! (elevate on sustained BLOCK, never below `base` — Maslow-monotone). [`rung_level`]
//! names a recipe's rung in that vocabulary; this module never re-invents the
//! ladder, it gates each pyramid level on the organ.
//!
//! # Carrying lower-rung awareness — the anti-rediscovery property
//!
//! The 24-loci register is **read, not consumed**, across the ENTIRE rung walk.
//! A dimension grounded for a shallow-rung recipe (e.g. `Kausal` placed for
//! RCR #4 at its rung) stays grounded for every deeper recipe that reads it
//! (ICR #31 at the deepest rung). So the substrate itself **carries the
//! lower-level rung awareness forward** — the cause is placed once, at its rung,
//! and READ by all higher rungs ([`carried_awareness`]), never re-derived. This
//! is the concrete mechanism behind #777's "orientation for free": observational
//! causal discovery pays super-exponentially to RE-search for the cause at each
//! step, whereas here the deeper rung reads the `Kausal`/`BasinAnchor`/
//! `Antecedent` locus the lower rung already placed. Escalation is **monotone
//! accumulation** (the elected schema binds MORE loci, never fewer — #778's
//! "more loci bound, shape unchanged"), so the cumulative grounded set is
//! non-decreasing in rung (the `carried_is_monotone` test proves it). The rung
//! level occupies ZERO stored bytes — it is carried by WHICH loci are bound, not
//! by a magnitude (#778 / le-contract §2 slot purity).
//!
//! # Discipline
//!
//! Additive, zero-dep, reading-only. `required_loci` is a documented
//! classification (like [`recipe_dispatch::inference`](crate::recipe_dispatch::inference)),
//! never a stored byte. The witness is READ (loci are identity pointers,
//! `I-VSA-IDENTITIES` clean); nothing here mutates a register or reserves a slot.

use crate::causal_witness::{CausalWitnessFacet, Locus};
use crate::cognitive_shader::RungLevel;
use crate::recipe_dispatch::{inference, RecipeInference};

/// The [`Locus`] dimensions recipe `id` consumes, grounded in the recipe's
/// documented [`substrate`](crate::recipes::Recipe::substrate) + its NARS
/// inference. Every recipe reads at least one dimension — the wiring is real (no
/// recipe trivially always-fires). The 8 reserved loci (`16..24`) are never
/// required; only the 16 named dimensions are consumable.
///
/// The classification (per `recipes.rs` `substrate` string):
/// - **causal / backward** recipes read [`Kausal`](Locus::Kausal) (their stored
///   cause) — RCR #4, TCA #12, ICR #31, IDR #29, SSAM #28.
/// - **S/P/O-grounding** recipes read the meaning loci — MCT #14, ARE #19,
///   ZCF #24, SDD #32, HKF #34, RCR #4, ICR #31.
/// - **consensus / revision** recipes read [`Quorum`](Locus::Quorum) and/or
///   [`Contradiction`](Locus::Contradiction) — SMAD #3, ASC #7, CR #11, CDI #17,
///   TCF #20, SSR #21, MPC #27, SPP #30, ICR #31.
/// - **hierarchy / basin** recipes read [`BasinAnchor`](Locus::BasinAnchor) and
///   the `hi/lo`-chain loci — HTD #2, CAS #8, LSI #15, ETD #22, HPM #25.
/// - **meta / template / rung** recipes read
///   [`MeaningLevel`](Locus::MeaningLevel) and
///   [`RunbookEvidence`](Locus::RunbookEvidence) — RTE #1, MCP #10, PSO #16,
///   AMP #23, CUR #26, DTMF #33.
/// - **texture / divergence** recipes read
///   [`QualiaReference`](Locus::QualiaReference) — TR #6, IRS #9, CDT #13.
/// - **episodic / temporal** recipes read [`Temporal`](Locus::Temporal) and
///   [`Antecedent`](Locus::Antecedent) — TCA #12, CWS #18.
#[must_use]
pub fn required_loci(id: u8) -> &'static [Locus] {
    use Locus::*;
    match id {
        1 => &[MeaningLevel],                   // RTE: rung-depth expand
        2 => &[BasinAnchor, Supports],          // HTD: hierarchical decompose
        3 => &[Quorum, Contradiction],          // SMAD: debate → consensus
        4 => &[Kausal, SMeaning, OMeaning],     // RCR: backward S_O + Granger
        5 => &[Supports],                       // TCP: prune the branch it supports
        6 => &[QualiaReference],                // TR: temperature perturb
        7 => &[Contradiction],                  // ASC: adversarial self-critique
        8 => &[MeaningLevel, BasinAnchor],      // CAS: abstraction scaling
        9 => &[Modal, QualiaReference],         // IRS: persona synthesis
        10 => &[MeaningLevel, QualiaReference], // MCP: meta-cognition calibrate
        11 => &[Contradiction, Quorum],         // CR: contradiction resolution
        12 => &[Temporal, Kausal],              // TCA: Granger temporal
        13 => &[QualiaReference, MeaningLevel], // CDT: converge/diverge
        14 => &[SMeaning, PMeaning, OMeaning],  // MCT: grammar-triangle fuse
        15 => &[BasinAnchor],                   // LSI: latent cluster read
        16 => &[RunbookEvidence],               // PSO: scaffold slots
        17 => &[Contradiction],                 // CDI: dissonance induction
        18 => &[Temporal, Antecedent],          // CWS: episodic memory
        19 => &[SMeaning, OMeaning],            // ARE: unbind component
        20 => &[Quorum],                        // TCF: cascade agreement
        21 => &[Contradiction, MeaningLevel],   // SSR: self-skepticism
        22 => &[BasinAnchor, Supports],         // ETD: emergent decompose
        23 => &[RunbookEvidence, MeaningLevel], // AMP: adaptive meta-prompt
        24 => &[SMeaning, OMeaning],            // ZCF: zero-shot fuse
        25 => &[BasinAnchor],                   // HPM: nearest pattern
        26 => &[MeaningLevel],                  // CUR: coarse→fine reduce
        27 => &[Quorum],                        // MPC: majority compress
        28 => &[SMeaning, PMeaning, OMeaning],  // SSAM: analogical mapping
        29 => &[SMeaning, PMeaning, Kausal],    // IDR: causality-flow reframe
        30 => &[Quorum, Contradiction],         // SPP: shadow parallel agree
        31 => &[Kausal, Contradiction, SMeaning, PMeaning, OMeaning], // ICR: counterfactual
        32 => &[SMeaning, OMeaning],            // SDD: reciprocal validate
        33 => &[RunbookEvidence, MeaningLevel], // DTMF: meta-frame switch
        34 => &[SMeaning, PMeaning, OMeaning],  // HKF: cross-domain fuse
        _ => &[],
    }
}

/// The first required [`Locus`] that is **unbound** (offset `0`) in `w` — so the
/// recipe is DISQUALIFIED — or `None` if every required dimension is grounded.
/// The 24-dimension analog of
/// [`nan_disqualifier`](crate::recipe_dispatch::nan_disqualifier): the loci-side
/// reliability gate (a recipe never fires on a dimension the row could not
/// ground). An unknown id (`required_loci == []`) is trivially grounded → `None`
/// (matches `nan_disqualifier`'s "no requirement → never disqualified").
#[must_use]
pub fn loci_disqualifier(w: &CausalWitnessFacet, id: u8) -> Option<Locus> {
    required_loci(id).iter().copied().find(|&l| !w.is_bound(l))
}

/// Whether every dimension recipe `id` requires is bound in `w`.
#[must_use]
pub fn is_grounded(w: &CausalWitnessFacet, id: u8) -> bool {
    loci_disqualifier(w, id).is_none()
}

/// The **Maslow rung a single [`Locus`] dimension sits at** (`0..=6`,
/// [`RungLevel`](crate::cognitive_shader::RungLevel)-aligned): raw observation +
/// meaning grounding shallow, basin/relate mid, meta/texture higher, consensus
/// structural, causal/counterfactual at the apex. This is the organ's own depth
/// scale — a dimension IS a cognitive need level.
///
/// - `Temporal` = 0 (Surface — the raw time signal)
/// - `SMeaning`/`PMeaning`/`OMeaning` = 1 (Shallow — match to known meaning)
/// - `Lokal`/`BasinAnchor`/`Antecedent` = 2 (Contextual — relate to neighbors, Pearl L1)
/// - `Supports`/`SupportedBy`/`Modal` = 3 (Analogical — map onto structure)
/// - `MeaningLevel`/`RunbookEvidence`/`QualiaReference` = 4 (Abstract — generalize)
/// - `Quorum` = 5 (Structural — consensus models the mechanism, Pearl L2)
/// - `Kausal`/`Contradiction` = 6 (Counterfactual — the cause / preserved
///   contradiction, Pearl L3)
#[must_use]
pub const fn locus_rung(l: Locus) -> u8 {
    match l {
        Locus::Temporal => 0,
        Locus::SMeaning | Locus::PMeaning | Locus::OMeaning => 1,
        Locus::Lokal | Locus::BasinAnchor | Locus::Antecedent => 2,
        Locus::Supports | Locus::SupportedBy | Locus::Modal => 3,
        Locus::MeaningLevel | Locus::RunbookEvidence | Locus::QualiaReference => 4,
        Locus::Quorum => 5,
        Locus::Kausal | Locus::Contradiction => 6,
    }
}

/// The **organ-derived rung**: a recipe is as deep as the DEEPEST dimension it
/// requires — `max(locus_rung(l))` over [`required_loci`]. This is the 24-based
/// dispatch order (the whole walk, order AND gate, keyed on the organ), replacing
/// the static `Tier`+`inference` classification of
/// [`recipe_dispatch::rung`](crate::recipe_dispatch::rung) (kept as a documented
/// cross-check, not the order). So ICR #31 (reads `Kausal`+`Contradiction`) lands
/// at the counterfactual apex BECAUSE those are apex dimensions, not because a
/// Tier table said so; HPM #25 (reads only `BasinAnchor`) sits shallow. A recipe
/// with no required loci is rung `0`.
#[must_use]
pub fn loci_rung(id: u8) -> u8 {
    required_loci(id)
        .iter()
        .map(|&l| locus_rung(l))
        .max()
        .unwrap_or(0)
}

/// The 34 recipe ids in **organ-derived dispatch order** — ascending [`loci_rung`],
/// then ascending id. The 24-based sibling of
/// [`recipe_dispatch::dispatch_order`](crate::recipe_dispatch::dispatch_order)
/// (which orders by the static rung): shallow-organ recipes first, the
/// causal/counterfactual apex last.
#[must_use]
pub fn loci_dispatch_order() -> [u8; 34] {
    let mut ids: [u8; 34] = core::array::from_fn(|i| i as u8 + 1);
    ids.sort_by_key(|&id| (loci_rung(id), id));
    ids
}

/// One rung-ordered loci-gated dispatch decision — the 24-dimension sibling of
/// [`RecipeStep`](crate::recipe_dispatch::RecipeStep). Records whether the recipe
/// fired, at what rung, on which inference, how many of its required loci were
/// bound (the grounded coverage that licensed it), and — if disqualified — the
/// first unbound dimension.
#[derive(Debug, Clone, Copy)]
pub struct LociStep {
    /// Recipe id (1..=34).
    pub id: u8,
    /// Rung this recipe fires at — the organ-derived [`loci_rung`] (the deepest
    /// dimension it requires), NOT the static `recipe_dispatch::rung`.
    pub rung: u8,
    /// Inference the recipe embodies.
    pub inference: RecipeInference,
    /// How many [`Locus`] dimensions the recipe requires.
    pub required: u8,
    /// How many of those required dimensions are bound in the witness.
    pub bound: u8,
    /// `None` = fired (all required loci bound); `Some(locus)` = disqualified
    /// because that dimension was unbound.
    pub disqualified_by: Option<Locus>,
}

impl LociStep {
    /// Did this recipe fire (every required dimension grounded)?
    #[must_use]
    pub const fn fired(&self) -> bool {
        self.disqualified_by.is_none()
    }
}

/// **The rung-level walk over the 24 dimensions.** Visits all 34 recipes in
/// [`loci_dispatch_order`] (ascending organ-derived [`loci_rung`], then id),
/// gating each on [`loci_disqualifier`]. The returned steps are the causal chain
/// of the loci-gated orchestration: shallow-organ recipes first, the
/// causal/counterfactual apex last — each fired iff its 24-dimension checklist is
/// grounded. Order AND gate are keyed on the organ, not on a static Tier table.
#[must_use]
pub fn loci_ladder(w: &CausalWitnessFacet) -> Vec<LociStep> {
    loci_dispatch_order()
        .into_iter()
        .map(|id| {
            let req = required_loci(id);
            let bound = req.iter().filter(|&&l| w.is_bound(l)).count() as u8;
            LociStep {
                id,
                rung: loci_rung(id),
                inference: inference(id),
                required: req.len() as u8,
                bound,
                disqualified_by: loci_disqualifier(w, id),
            }
        })
        .collect()
}

/// The recipes **reachable** under witness `w` — those whose every required
/// dimension is bound — in rung-dispatch order. With a fully-bound witness this
/// is all 34; the count grows monotonically as more loci are grounded.
#[must_use]
pub fn reachable(w: &CausalWitnessFacet) -> Vec<u8> {
    loci_ladder(w)
        .into_iter()
        .filter(LociStep::fired)
        .map(|s| s.id)
        .collect()
}

/// The **Maslow-pyramid** rung a recipe fires at, named in the shipped
/// [`RungLevel`](crate::cognitive_shader::RungLevel) vocabulary
/// (`Surface..Transcendent`). Bridges the organ-derived [`loci_rung`] into the
/// operator-ruled cognitive pyramid via
/// [`RungLevel::from_u8`](crate::cognitive_shader::RungLevel::from_u8) — the ONE
/// u8→rung mapping. A recipe's cognitive "need level" IS its deepest required
/// organ dimension; the [`RungElevator`](crate::cognitive_shader::RungElevator)
/// climbs the pyramid Maslow-monotone.
#[must_use]
pub fn rung_level(id: u8) -> RungLevel {
    RungLevel::from_u8(loci_rung(id))
}

/// **The carried lower-rung awareness at a pyramid level** — the set of witness
/// dimensions already grounded by every recipe at rung `<= up_to`, in canonical
/// [`Locus::ALL`](crate::causal_witness::Locus) order. This is what the substrate
/// carries UP the Maslow climb: a deeper recipe reads these already-placed loci
/// (the `Kausal` cause, the `BasinAnchor`, the `Antecedent`) instead of
/// re-deriving them — the anti-exponential-rediscovery property (#777 "orientation
/// for free"). Because the register is read (never consumed) and escalation only
/// binds MORE loci, [`carried_awareness`] is **monotone non-decreasing** in
/// `up_to` (a lower rung's grounding is never lost to a higher one).
#[must_use]
pub fn carried_awareness(w: &CausalWitnessFacet, up_to: u8) -> Vec<Locus> {
    let mut carried: Vec<Locus> = Vec::new();
    for &id in loci_dispatch_order().iter().filter(|&&id| loci_rung(id) <= up_to) {
        for &l in required_loci(id) {
            if w.is_bound(l) && !carried.contains(&l) {
                carried.push(l);
            }
        }
    }
    // canonical Locus::ALL order (deterministic, independent of dispatch order)
    Locus::ALL
        .into_iter()
        .filter(|l| carried.contains(l))
        .collect()
}

/// **Higher-rung thinking prunes lower-related.** The dual of
/// [`carried_awareness`]: carry accumulates lower→higher; prune subsumes
/// higher→lower. A reachable recipe is PRUNED when some reachable recipe at a
/// STRICTLY HIGHER rung requires a **strict superset** of its dimensions — the
/// higher, more-complete thinking already grounds everything the lower related
/// one would, so re-firing the lower is redundant. Returns the ACTIVE recipes
/// after prune, in rung-dispatch order. This keeps the Maslow climb from
/// re-doing subsumed lower work (the attention-economy complement to
/// anti-rediscovery): the substrate carries the lower groundings UP, and the
/// higher thinking prunes the lower related ones from active dispatch.
///
/// The prune is by dimension-subsumption only (a documented model, not the only
/// possible rule); a recipe reading dimensions no higher recipe covers always
/// stays active.
#[must_use]
pub fn active_after_prune(w: &CausalWitnessFacet) -> Vec<u8> {
    let reach = reachable(w);
    reach
        .iter()
        .copied()
        .filter(|&id| {
            let mine = required_loci(id);
            // pruned iff a strictly-higher-rung reachable recipe's loci ⊋ mine
            !reach.iter().any(|&other| {
                other != id && loci_rung(other) > loci_rung(id) && {
                    let theirs = required_loci(other);
                    mine.iter().all(|l| theirs.contains(l)) && theirs.len() > mine.len()
                }
            })
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::causal_witness::Locus;

    /// A witness with every named locus bound (offset −1 = one event back).
    fn fully_bound() -> CausalWitnessFacet {
        let mut w = CausalWitnessFacet::ZERO;
        for l in Locus::ALL {
            w = w.with(l, -1);
        }
        w
    }

    #[test]
    fn every_recipe_requires_at_least_one_named_locus() {
        for id in 1..=34u8 {
            let req = required_loci(id);
            assert!(!req.is_empty(), "recipe {id} declares no loci — wiring gap");
            // never requires a reserved slot (only the 16 named loci are consumable)
            for &l in req {
                assert!(
                    Locus::ALL.contains(&l),
                    "recipe {id} requires non-named locus {l:?}"
                );
            }
        }
    }

    #[test]
    fn fully_bound_witness_reaches_all_34() {
        let w = fully_bound();
        let r = reachable(&w);
        assert_eq!(r.len(), 34, "a fully-bound witness grounds every recipe");
        // a permutation of 1..=34
        let mut sorted = r.clone();
        sorted.sort_unstable();
        assert_eq!(sorted, (1..=34).collect::<Vec<u8>>());
    }

    #[test]
    fn unbound_witness_reaches_nothing() {
        // ZERO witness: no dimension placed → every recipe disqualified.
        let r = reachable(&CausalWitnessFacet::ZERO);
        assert!(
            r.is_empty(),
            "no dimension grounded → no recipe fires (all NaN-gated)"
        );
    }

    #[test]
    fn unbinding_a_dimension_disqualifies_exactly_its_consumers() {
        // Start fully bound, then unbind Kausal: the Kausal-consuming recipes
        // (RCR #4, TCA #12, IDR #29, ICR #31) must drop out; others stay.
        let mut w = fully_bound();
        w = w.with(Locus::Kausal, 0); // unbind
        let reached: std::collections::HashSet<u8> = reachable(&w).into_iter().collect();
        for id in [4u8, 12, 29, 31] {
            assert!(
                !reached.contains(&id),
                "recipe {id} requires Kausal — unbinding it must disqualify {id}"
            );
            assert_eq!(loci_disqualifier(&w, id), Some(Locus::Kausal));
        }
        // A recipe that does NOT read Kausal (HPM #25 reads BasinAnchor) still fires.
        assert!(
            reached.contains(&25),
            "HPM #25 does not read Kausal → still fires"
        );
    }

    #[test]
    fn ladder_is_rung_ascending_and_records_coverage() {
        let w = fully_bound();
        let steps = loci_ladder(&w);
        assert_eq!(steps.len(), 34);
        assert!(
            steps.windows(2).all(|s| s[0].rung <= s[1].rung),
            "the loci walk fires shallow → deep"
        );
        // every step's bound == required under a fully-bound witness.
        assert!(steps.iter().all(|s| s.bound == s.required && s.fired()));
        // ICR #31 (counterfactual) sits at the deepest ORGAN rung (Kausal=6) and
        // reads 5 loci — deepest because it reads apex dimensions, not by a table.
        let icr = steps.iter().find(|s| s.id == 31).unwrap();
        assert_eq!(icr.required, 5, "ICR reads Kausal+Contradiction+S/P/O");
        assert_eq!(icr.rung, 6, "counterfactual apex = Kausal/Contradiction rung");
        assert_eq!(icr.rung, steps.iter().map(|s| s.rung).max().unwrap());
    }

    #[test]
    fn rung_level_names_the_maslow_pyramid() {
        // ICR #31 (counterfactual) sits high on the pyramid; a shallow forward
        // deduction (TCP #5, CrossTier deduction) sits low.
        assert!(
            rung_level(31) >= RungLevel::Counterfactual,
            "ICR climbs to the counterfactual apex region"
        );
        assert!(
            rung_level(5) < rung_level(31),
            "shallow deduction is a lower cognitive need than counterfactual"
        );
        // the mapping is the shipped one — round-trips through RungLevel::from_u8.
        for id in 1..=34u8 {
            assert_eq!(rung_level(id), RungLevel::from_u8(loci_rung(id)));
        }
    }

    #[test]
    fn carried_is_monotone() {
        // The substrate carries lower-rung awareness UP: the cumulative grounded
        // dimension set never shrinks as the pyramid climbs (anti-rediscovery).
        let w = fully_bound();
        for r in 1..9u8 {
            let lo: std::collections::HashSet<Locus> =
                carried_awareness(&w, r).into_iter().collect();
            let hi: std::collections::HashSet<Locus> =
                carried_awareness(&w, r + 1).into_iter().collect();
            assert!(
                lo.is_subset(&hi),
                "rung {r} awareness must be carried into rung {}",
                r + 1
            );
        }
    }

    #[test]
    fn deep_recipe_stands_on_carried_lower_rung_loci() {
        // ICR #31 reads Kausal + Contradiction + S/P/O. Under a fully-bound
        // witness, every dimension it needs was ALREADY placed by recipes at or
        // below its rung — so reaching the counterfactual apex re-derives
        // nothing; it reads the carried substrate (the #777 "orientation for free").
        let w = fully_bound();
        let carried: std::collections::HashSet<Locus> =
            carried_awareness(&w, loci_rung(31)).into_iter().collect();
        for &l in required_loci(31) {
            assert!(
                carried.contains(&l),
                "ICR's dimension {l:?} must be carried, not rediscovered at the apex"
            );
        }
    }

    #[test]
    fn higher_rung_thinking_prunes_lower_related() {
        // Under a fully-bound witness, all 34 are reachable; prune removes the
        // lower-related ones a higher-rung recipe strictly subsumes.
        let w = fully_bound();
        let active: std::collections::HashSet<u8> = active_after_prune(&w).into_iter().collect();
        let reach: std::collections::HashSet<u8> = reachable(&w).into_iter().collect();
        assert!(active.is_subset(&reach), "prune only removes, never adds");
        // ICR #31 is the deepest rung — nothing above it, so it is never pruned.
        assert!(active.contains(&31), "the apex recipe is never pruned");
        // RTE #1 reads only [MeaningLevel]; SSR #21 (higher rung) reads
        // [Contradiction, MeaningLevel] ⊋ [MeaningLevel] → RTE is subsumed.
        assert!(loci_rung(21) > loci_rung(1) && required_loci(21).contains(&Locus::MeaningLevel));
        assert!(
            !active.contains(&1),
            "RTE #1 is pruned — a higher-rung recipe already grounds MeaningLevel + more"
        );
        assert!(active.len() < 34, "some lower-related thinking is pruned");
    }

    #[test]
    fn loci_rung_is_organ_derived_not_the_static_tier_table() {
        // ICR #31: organ rung = its deepest dimension (Kausal/Contradiction = 6,
        // Counterfactual). The static Tier+inference table put it at 9 — so the
        // ORDER now comes from the organ, not the classification.
        assert_eq!(loci_rung(31), 6);
        assert_ne!(
            crate::recipe_dispatch::rung(31),
            loci_rung(31),
            "organ-derived rung must differ from the static Tier table"
        );
        // loci_dispatch_order: a permutation of 1..=34, ascending loci_rung.
        let order = loci_dispatch_order();
        let mut sorted = order;
        sorted.sort_unstable();
        assert_eq!(sorted, core::array::from_fn::<u8, 34, _>(|i| i as u8 + 1));
        let rungs: Vec<u8> = order.iter().map(|&id| loci_rung(id)).collect();
        assert!(
            rungs.windows(2).all(|w| w[0] <= w[1]),
            "organ-ascending order: {rungs:?}"
        );
        // every named dimension's rung is a Pearl-aligned 0..=6.
        for l in Locus::ALL {
            assert!(locus_rung(l) <= 6, "{l:?} rung out of range");
        }
    }

    #[test]
    fn partial_witness_reaches_a_monotone_subset() {
        // Bind only the meaning loci → S/P/O-grounding recipes fire; consensus
        // and causal recipes do not.
        let w = CausalWitnessFacet::ZERO
            .with(Locus::SMeaning, -1)
            .with(Locus::PMeaning, -1)
            .with(Locus::OMeaning, -1);
        let reached: std::collections::HashSet<u8> = reachable(&w).into_iter().collect();
        assert!(reached.contains(&14), "MCT #14 (S/P/O) fires");
        assert!(reached.contains(&34), "HKF #34 (S/P/O) fires");
        assert!(
            !reached.contains(&3),
            "SMAD #3 needs Quorum+Contradiction → no"
        );
        assert!(!reached.contains(&31), "ICR #31 needs Kausal too → no");
    }
}
