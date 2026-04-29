//! 144-cell verb-role lookup table â 12 semantic families Ã 12 tense/aspect/mood.
//!
//! Each cell holds a TEKAMOLO slot prior: which slots a verb of this family
//! and tense expects to be filled. Parsing reduces to (family, tense) â
//! row â fill slots from morphology â NARS-revise truth.
//!
//! Slot priors seeded from grammar-landscape.md Â§3 TEKAMOLO semantics.
//! Starter values â tune empirically with corpus statistics.
//!
//! See PR #279 outlook E3 + grammar-landscape.md Â§9.
//!
//! META-AGENT: `pub mod verb_table;` to mod.rs.

use crate::grammar::role_keys::Tense;

/// Twelve top-level semantic families. The naming is deliberately
/// process-oriented (verbs as transformations on configurations of
/// the world) rather than syntax-oriented â these are the "roles a
/// predicate plays" that disambiguate which TEKAMOLO slots get filled.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum VerbFamily {
    Becomes, Causes, Supports, Contradicts, Refines, Grounds,
    Abstracts, Enables, Prevents, Transforms, Mirrors, Dissolves,
}

impl VerbFamily {
    pub const ALL: [Self; 12] = [
        Self::Becomes, Self::Causes, Self::Supports, Self::Contradicts,
        Self::Refines, Self::Grounds, Self::Abstracts, Self::Enables,
        Self::Prevents, Self::Transforms, Self::Mirrors, Self::Dissolves,
    ];
}

/// Slot prior per TEKAMOLO axis. Cells in [0.0, 1.0]: 0 = slot rarely filled,
/// 1 = slot always filled.
#[derive(Debug, Clone, Copy)]
pub struct SlotPrior {
    pub temporal: f32,
    pub kausal: f32,
    pub modal: f32,
    pub lokal: f32,
    pub instrument: f32,
}

impl SlotPrior {
    pub const fn uniform() -> Self {
        Self { temporal: 0.5, kausal: 0.5, modal: 0.5, lokal: 0.5, instrument: 0.5 }
    }
}

/// 144-cell lookup: rows = `VerbFamily`, columns = `Tense`. Indexing is
/// by enum discriminant (`as usize`), so any future reordering of either
/// enum must keep `#[repr(u8)]` (or equivalent) and contiguous indices.
pub struct VerbRoleTable {
    cells: [[SlotPrior; 12]; 12],
}

impl VerbRoleTable {
    pub fn new_uniform() -> Self {
        Self { cells: [[SlotPrior::uniform(); 12]; 12] }
    }
    pub fn lookup(&self, family: VerbFamily, tense: Tense) -> SlotPrior {
        self.cells[family as usize][tense as usize]
    }
    pub fn set(&mut self, family: VerbFamily, tense: Tense, prior: SlotPrior) {
        self.cells[family as usize][tense as usize] = prior;
    }
}

/// Default table with hand-set families per the plan's table and
/// grammar-landscape.md Â§3 TEKAMOLO slot semantics.
///
/// Semantic profiles â starter â tune empirically:
///   BECOMES    â Change verb: high Temporal + Modal
///   CAUSES     â Action verb: high Kausal + Instrument
///   SUPPORTS   â State verb:  high Modal, low Temporal
///   CONTRADICTS â State verb: high Modal + Kausal
///   REFINES    â State verb:  high Modal, moderate Kausal
///   GROUNDS    â State verb:  high Lokal + Modal
///   ABSTRACTS  â Change verb: high Modal + Temporal
///   ENABLES    â Discovery verb: high Kausal + Lokal
///   PREVENTS   â Action verb: high Kausal + Temporal
///   TRANSFORMS â Action verb: high Kausal + Temporal + Instrument
///   MIRRORS    â Change verb: high Temporal + Modal + Lokal
///   DISSOLVES  â Change verb: high Temporal + Modal
///
/// The numbers are *priors*, not facts: a future PR replaces them
/// with corpus-derived statistics. Mark this `// starter â tune empirically`
/// in any consumer that depends on specific values.
pub fn default_table() -> VerbRoleTable {
    let mut t = VerbRoleTable::new_uniform();

    // --- Change verbs: high Temporal + Modal ---

    // BECOMES: state-change, strongly temporal + modal
    let becomes = SlotPrior { temporal: 0.9, kausal: 0.2, modal: 0.7, lokal: 0.3, instrument: 0.2 };
    for tense in Tense::ALL {
        t.set(VerbFamily::Becomes, tense, becomes);
    }

    // DISSOLVES: destruction-as-change, high temporal + modal
    let dissolves = SlotPrior { temporal: 0.85, kausal: 0.3, modal: 0.7, lokal: 0.25, instrument: 0.2 };
    for tense in Tense::ALL {
        t.set(VerbFamily::Dissolves, tense, dissolves);
    }

    // ABSTRACTS: conceptual transformation, high modal + temporal
    let abstracts = SlotPrior { temporal: 0.7, kausal: 0.25, modal: 0.85, lokal: 0.15, instrument: 0.2 };
    for tense in Tense::ALL {
        t.set(VerbFamily::Abstracts, tense, abstracts);
    }

    // MIRRORS: reflection/symmetry, temporal + modal + lokal
    let mirrors = SlotPrior { temporal: 0.75, kausal: 0.2, modal: 0.7, lokal: 0.6, instrument: 0.15 };
    for tense in Tense::ALL {
        t.set(VerbFamily::Mirrors, tense, mirrors);
    }

    // --- Action verbs: high Kausal + Temporal ---

    // CAUSES: strong causal agency, high kausal + instrument
    let causes = SlotPrior { temporal: 0.4, kausal: 0.95, modal: 0.4, lokal: 0.3, instrument: 0.5 };
    for tense in Tense::ALL {
        t.set(VerbFamily::Causes, tense, causes);
    }

    // PREVENTS: blocking action, high kausal + temporal
    let prevents = SlotPrior { temporal: 0.7, kausal: 0.9, modal: 0.4, lokal: 0.25, instrument: 0.35 };
    for tense in Tense::ALL {
        t.set(VerbFamily::Prevents, tense, prevents);
    }

    // TRANSFORMS: active change, high kausal + temporal + instrument
    let transforms = SlotPrior { temporal: 0.8, kausal: 0.85, modal: 0.35, lokal: 0.3, instrument: 0.6 };
    for tense in Tense::ALL {
        t.set(VerbFamily::Transforms, tense, transforms);
    }

    // --- State verbs: high Modal, low Temporal ---

    // SUPPORTS: epistemic backing, high modal
    let supports = SlotPrior { temporal: 0.2, kausal: 0.35, modal: 0.85, lokal: 0.2, instrument: 0.3 };
    for tense in Tense::ALL {
        t.set(VerbFamily::Supports, tense, supports);
    }

    // CONTRADICTS: logical opposition, high modal + kausal
    let contradicts = SlotPrior { temporal: 0.15, kausal: 0.7, modal: 0.9, lokal: 0.15, instrument: 0.1 };
    for tense in Tense::ALL {
        t.set(VerbFamily::Contradicts, tense, contradicts);
    }

    // REFINES: iterative improvement, high modal, moderate kausal
    let refines = SlotPrior { temporal: 0.3, kausal: 0.4, modal: 0.8, lokal: 0.2, instrument: 0.35 };
    for tense in Tense::ALL {
        t.set(VerbFamily::Refines, tense, refines);
    }

    // GROUNDS: anchoring to context, high lokal + modal
    let grounds = SlotPrior { temporal: 0.25, kausal: 0.3, modal: 0.75, lokal: 0.85, instrument: 0.2 };
    for tense in Tense::ALL {
        t.set(VerbFamily::Grounds, tense, grounds);
    }

    // --- Discovery/enablement verbs: high Kausal + Lokal ---

    // ENABLES: facilitation, high kausal + lokal
    let enables = SlotPrior { temporal: 0.35, kausal: 0.8, modal: 0.4, lokal: 0.7, instrument: 0.45 };
    for tense in Tense::ALL {
        t.set(VerbFamily::Enables, tense, enables);
    }

    t
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn table_has_144_cells() {
        let t = VerbRoleTable::new_uniform();
        let mut count = 0;
        for f in VerbFamily::ALL.iter() {
            for tense_idx in 0..12 {
                let _ = t.cells[*f as usize][tense_idx];
                count += 1;
            }
        }
        assert_eq!(count, 144);
    }

    #[test]
    fn lookup_returns_uniform_for_unset_cell() {
        let t = VerbRoleTable::new_uniform();
        let p = t.lookup(VerbFamily::Mirrors, Tense::Pluperfect);
        assert!((p.temporal - 0.5).abs() < 1e-6);
    }

    #[test]
    fn default_table_overrides_some_cells() {
        let t = default_table();
        let p = t.lookup(VerbFamily::Causes, Tense::Present);
        assert!(p.kausal > 0.8);
    }

    // --- Per-family tests: verify priors are non-zero for at least 2 TEKAMOLO slots ---

    /// Helper: count slots that are non-uniform (differ from 0.5 by > 0.05).
    fn count_non_uniform(p: &SlotPrior) -> usize {
        let slots = [p.temporal, p.kausal, p.modal, p.lokal, p.instrument];
        slots.iter().filter(|&&v| (v - 0.5).abs() > 0.05).count()
    }

    #[test]
    fn becomes_change_verb_temporal_modal() {
        let t = default_table();
        let p = t.lookup(VerbFamily::Becomes, Tense::Present);
        assert!(p.temporal > 0.7, "Becomes should have high temporal");
        assert!(p.modal > 0.6, "Becomes should have high modal");
        assert!(count_non_uniform(&p) >= 2);
    }

    #[test]
    fn causes_action_verb_kausal_instrument() {
        let t = default_table();
        let p = t.lookup(VerbFamily::Causes, Tense::Past);
        assert!(p.kausal > 0.8, "Causes should have high kausal");
        assert!(p.instrument > 0.4, "Causes should have elevated instrument");
        assert!(count_non_uniform(&p) >= 2);
    }

    #[test]
    fn supports_state_verb_modal_high() {
        let t = default_table();
        let p = t.lookup(VerbFamily::Supports, Tense::Present);
        assert!(p.modal > 0.7, "Supports should have high modal");
        assert!(p.temporal < 0.4, "Supports should have low temporal");
        assert!(count_non_uniform(&p) >= 2);
    }

    #[test]
    fn contradicts_state_verb_modal_kausal() {
        let t = default_table();
        let p = t.lookup(VerbFamily::Contradicts, Tense::Future);
        assert!(p.modal > 0.8, "Contradicts should have high modal");
        assert!(p.kausal > 0.6, "Contradicts should have elevated kausal");
        assert!(count_non_uniform(&p) >= 2);
    }

    #[test]
    fn refines_state_verb_modal() {
        let t = default_table();
        let p = t.lookup(VerbFamily::Refines, Tense::Perfect);
        assert!(p.modal > 0.7, "Refines should have high modal");
        assert!(p.temporal < 0.4, "Refines should have low temporal");
        assert!(count_non_uniform(&p) >= 2);
    }

    #[test]
    fn grounds_state_verb_lokal_modal() {
        let t = default_table();
        let p = t.lookup(VerbFamily::Grounds, Tense::Habitual);
        assert!(p.lokal > 0.7, "Grounds should have high lokal");
        assert!(p.modal > 0.6, "Grounds should have elevated modal");
        assert!(count_non_uniform(&p) >= 2);
    }

    #[test]
    fn abstracts_change_verb_modal_temporal() {
        let t = default_table();
        let p = t.lookup(VerbFamily::Abstracts, Tense::PresentContinuous);
        assert!(p.modal > 0.7, "Abstracts should have high modal");
        assert!(p.temporal > 0.6, "Abstracts should have elevated temporal");
        assert!(count_non_uniform(&p) >= 2);
    }

    #[test]
    fn enables_discovery_verb_kausal_lokal() {
        let t = default_table();
        let p = t.lookup(VerbFamily::Enables, Tense::Potential);
        assert!(p.kausal > 0.7, "Enables should have high kausal");
        assert!(p.lokal > 0.6, "Enables should have elevated lokal");
        assert!(count_non_uniform(&p) >= 2);
    }

    #[test]
    fn prevents_action_verb_kausal_temporal() {
        let t = default_table();
        let p = t.lookup(VerbFamily::Prevents, Tense::Past);
        assert!(p.kausal > 0.8, "Prevents should have high kausal");
        assert!(p.temporal > 0.6, "Prevents should have elevated temporal");
        assert!(count_non_uniform(&p) >= 2);
    }

    #[test]
    fn transforms_action_verb_kausal_temporal_instrument() {
        let t = default_table();
        let p = t.lookup(VerbFamily::Transforms, Tense::FuturePerfect);
        assert!(p.kausal > 0.7, "Transforms should have high kausal");
        assert!(p.temporal > 0.7, "Transforms should have high temporal");
        assert!(p.instrument > 0.5, "Transforms should have elevated instrument");
        assert!(count_non_uniform(&p) >= 3);
    }

    #[test]
    fn mirrors_change_verb_temporal_modal_lokal() {
        let t = default_table();
        let p = t.lookup(VerbFamily::Mirrors, Tense::Pluperfect);
        assert!(p.temporal > 0.6, "Mirrors should have elevated temporal");
        assert!(p.modal > 0.6, "Mirrors should have elevated modal");
        assert!(p.lokal > 0.5, "Mirrors should have elevated lokal");
        assert!(count_non_uniform(&p) >= 2);
    }

    #[test]
    fn dissolves_change_verb_temporal_modal() {
        let t = default_table();
        let p = t.lookup(VerbFamily::Dissolves, Tense::Imperative);
        assert!(p.temporal > 0.7, "Dissolves should have high temporal");
        assert!(p.modal > 0.6, "Dissolves should have elevated modal");
        assert!(count_non_uniform(&p) >= 2);
    }

    #[test]
    fn all_families_have_non_uniform_priors() {
        let t = default_table();
        for family in VerbFamily::ALL {
            let p = t.lookup(family, Tense::Present);
            assert!(
                count_non_uniform(&p) >= 2,
                "{:?} should have at least 2 non-uniform TEKAMOLO slots",
                family
            );
        }
    }
}
