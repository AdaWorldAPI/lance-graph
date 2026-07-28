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
//!
//! ## Tense modulation (G4 loose end)
//!
//! Earlier seed broadcast 12 family priors across all 12 tenses, producing a
//! degenerate 12-unique-value table with zero tense x family interaction. The
//! refactor introduces `SlotPriorDelta` + `SlotPrior::combine` and a
//! `tense_modifier(Tense)` function so each cell becomes
//! `final = base.combine(tense_modifier(tense))`.
//!
//! Modifiers are linguistically grounded in standard English grammar
//! (Quirk, Greenbaum, Leech & Svartvik, *A Comprehensive Grammar of the
//! English Language*, Longman 1985, sections 4.21-4.27 on tense / aspect /
//! mood):
//!
//! - Perfect aspects (Perfect, Pluperfect, FuturePerfect) emphasise
//!   completion and therefore temporal anchoring -> `temporal +0.15`.
//! - Continuous (progressive) aspects emphasise an ongoing process ->
//!   `temporal +0.10`, `modal -0.05` (less anchored, less modal weight).
//! - Imperative is a timeless directive command -> `temporal -0.20`,
//!   `modal +0.20`.
//! - Potential (irrealis / possibility mood; this enum's stand-in for the
//!   Subjunctive) emphasises possibility -> `temporal -0.10`, `modal +0.25`,
//!   `kausal -0.05` (cause is hypothetical).
//! - Habitual is recurring-as-timeless -> `temporal -0.10`, `modal +0.05`.
//! - Default (Present, Past, Future) leaves the base prior untouched.
//!
//! All resulting axes are clamped to [0.0, 1.0] in `SlotPrior::combine`.

use crate::grammar::role_keys::Tense;

/// Twelve top-level semantic families. The naming is deliberately
/// process-oriented (verbs as transformations on configurations of
/// the world) rather than syntax-oriented â these are the "roles a
/// predicate plays" that disambiguate which TEKAMOLO slots get filled.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum VerbFamily {
    Becomes,
    Causes,
    Supports,
    Contradicts,
    Refines,
    Grounds,
    Abstracts,
    Enables,
    Prevents,
    Transforms,
    Mirrors,
    Dissolves,
}

impl VerbFamily {
    pub const ALL: [Self; 12] = [
        Self::Becomes,
        Self::Causes,
        Self::Supports,
        Self::Contradicts,
        Self::Refines,
        Self::Grounds,
        Self::Abstracts,
        Self::Enables,
        Self::Prevents,
        Self::Transforms,
        Self::Mirrors,
        Self::Dissolves,
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
        Self {
            temporal: 0.5,
            kausal: 0.5,
            modal: 0.5,
            lokal: 0.5,
            instrument: 0.5,
        }
    }

    /// Apply a tense-driven delta to each axis and clamp the result to
    /// `[0.0, 1.0]`. This is how the broadcast-flat 12 priors per family
    /// gain tense x family interaction (G4 loose end).
    pub fn combine(self, delta: SlotPriorDelta) -> Self {
        fn clamp(x: f32) -> f32 {
            x.clamp(0.0, 1.0)
        }
        Self {
            temporal: clamp(self.temporal + delta.temporal),
            kausal: clamp(self.kausal + delta.kausal),
            modal: clamp(self.modal + delta.modal),
            lokal: clamp(self.lokal + delta.lokal),
            instrument: clamp(self.instrument + delta.instrument),
        }
    }
}

/// Additive delta applied to a `SlotPrior` per tense. Each axis is summed
/// with the base prior and clamped via `SlotPrior::combine`. Default = no
/// change (all zeros).
#[derive(Debug, Clone, Copy, Default)]
pub struct SlotPriorDelta {
    pub temporal: f32,
    pub kausal: f32,
    pub modal: f32,
    pub lokal: f32,
    pub instrument: f32,
}

/// Tense-driven modifier table. Linguistic grounding: Quirk et al.
/// *Comprehensive Grammar of the English Language* sections 4.21-4.27.
/// See module-level doc comment for the per-tense rationale.
pub fn tense_modifier(tense: Tense) -> SlotPriorDelta {
    use Tense::*;
    match tense {
        // Perfect aspects emphasise completion -> temporal anchoring.
        Perfect | Pluperfect | FuturePerfect => SlotPriorDelta {
            temporal: 0.15,
            kausal: 0.0,
            modal: 0.0,
            lokal: 0.0,
            instrument: 0.0,
        },
        // Continuous (progressive) aspects emphasise ongoing process.
        PresentContinuous | PastContinuous | FutureContinuous => SlotPriorDelta {
            temporal: 0.10,
            kausal: 0.0,
            modal: -0.05,
            lokal: 0.0,
            instrument: 0.0,
        },
        // Imperative: timeless directive -> suppresses temporal, amplifies modal.
        Imperative => SlotPriorDelta {
            temporal: -0.20,
            kausal: 0.0,
            modal: 0.20,
            lokal: 0.0,
            instrument: 0.0,
        },
        // Potential (irrealis / subjunctive role): possibility -> modal up,
        // kausal slightly down (cause is hypothetical), temporal slightly down.
        Potential => SlotPriorDelta {
            temporal: -0.10,
            kausal: -0.05,
            modal: 0.25,
            lokal: 0.0,
            instrument: 0.0,
        },
        // Habitual: recurring-as-timeless.
        Habitual => SlotPriorDelta {
            temporal: -0.10,
            kausal: 0.0,
            modal: 0.05,
            lokal: 0.0,
            instrument: 0.0,
        },
        // Present, Past, Future: unmarked tense, no modifier.
        Present | Past | Future => SlotPriorDelta::default(),
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
        Self {
            cells: [[SlotPrior::uniform(); 12]; 12],
        }
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
/// Base prior for a `VerbFamily` (pre-tense-modulation). The full per-cell
/// prior is `base_prior(family).combine(tense_modifier(tense))`.
pub fn base_prior(family: VerbFamily) -> SlotPrior {
    match family {
        // --- Change verbs: high Temporal + Modal ---
        VerbFamily::Becomes => SlotPrior {
            temporal: 0.9,
            kausal: 0.2,
            modal: 0.7,
            lokal: 0.3,
            instrument: 0.2,
        },
        VerbFamily::Dissolves => SlotPrior {
            temporal: 0.85,
            kausal: 0.3,
            modal: 0.7,
            lokal: 0.25,
            instrument: 0.2,
        },
        VerbFamily::Abstracts => SlotPrior {
            temporal: 0.7,
            kausal: 0.25,
            modal: 0.85,
            lokal: 0.15,
            instrument: 0.2,
        },
        VerbFamily::Mirrors => SlotPrior {
            temporal: 0.75,
            kausal: 0.2,
            modal: 0.7,
            lokal: 0.6,
            instrument: 0.15,
        },
        // --- Action verbs: high Kausal + Temporal ---
        VerbFamily::Causes => SlotPrior {
            temporal: 0.4,
            kausal: 0.95,
            modal: 0.4,
            lokal: 0.3,
            instrument: 0.5,
        },
        VerbFamily::Prevents => SlotPrior {
            temporal: 0.7,
            kausal: 0.9,
            modal: 0.4,
            lokal: 0.25,
            instrument: 0.35,
        },
        VerbFamily::Transforms => SlotPrior {
            temporal: 0.8,
            kausal: 0.85,
            modal: 0.35,
            lokal: 0.3,
            instrument: 0.6,
        },
        // --- State verbs: high Modal, low Temporal ---
        VerbFamily::Supports => SlotPrior {
            temporal: 0.2,
            kausal: 0.35,
            modal: 0.85,
            lokal: 0.2,
            instrument: 0.3,
        },
        VerbFamily::Contradicts => SlotPrior {
            temporal: 0.15,
            kausal: 0.7,
            modal: 0.9,
            lokal: 0.15,
            instrument: 0.1,
        },
        VerbFamily::Refines => SlotPrior {
            temporal: 0.3,
            kausal: 0.4,
            modal: 0.8,
            lokal: 0.2,
            instrument: 0.35,
        },
        VerbFamily::Grounds => SlotPrior {
            temporal: 0.25,
            kausal: 0.3,
            modal: 0.75,
            lokal: 0.85,
            instrument: 0.2,
        },
        // --- Discovery / enablement: high Kausal + Lokal ---
        VerbFamily::Enables => SlotPrior {
            temporal: 0.35,
            kausal: 0.8,
            modal: 0.4,
            lokal: 0.7,
            instrument: 0.45,
        },
    }
}

pub fn default_table() -> VerbRoleTable {
    let mut t = VerbRoleTable::new_uniform();
    for family in VerbFamily::ALL {
        let base = base_prior(family);
        for tense in Tense::ALL {
            t.set(family, tense, base.combine(tense_modifier(tense)));
        }
    }
    t
}

// ═══════════ The 4×4 Morton cascade reading (operator, 2026-07-28) ═══════════
//
// The 144 table addressed in the canonical 16×16 = 256 cascade space (one
// palette256 page): each cell is ONE BYTE `[fq:2|tq:2|fm:2|tm:2]` whose HIGH
// NIBBLE is the coarse 4×4 quadrant pair — nibble = ancestry, the D-TILE256
// rigor condition. The 12×12 stays AS-IS (occupied); the 4 spare members per
// quadrant axis are RESERVE-DON'T-RECLAIM — minted later with zero layout
// change, never compacted away.
//
// The INVERSE-PYRAMID perturbation reading: apex (uniform prior) → quadrant
// centroid (the coarse semantic signal) → member cell (a small residual
// perturbation on its centroid). The table was already secretly this shape —
// `base_prior` groups the 12 families into exactly these 4 superclasses (its
// own comment headers) and `tense_modifier` is class-shaped over the 4 tense
// quadrants — the cascade only makes the pyramid ADDRESSABLE. Deterministic
// throughout (verb actionability is table reads; no stochastic scoring).
//
// Compartment grounding (starter, wordnet-tunable): WordNet's verb inventory
// is compartmentalized into 15 supersenses (verb.change / verb.cognition /
// verb.perception / verb.motion / verb.stative / …) — a near-fill of the 16
// coarse cells — with troponymy as the pyramid below; Levin's alternation
// classes make the same coarse cuts. A families↔supersenses alignment probe
// over the in-house wordnet rail is the queued corpus tune (D-RCC-5 adjacent).
//
// O7 fence: this addresses verb_table's 144 ONLY. The divergent sigma_rosetta
// 144 (E-RUNG2-TWO-144S-1) is NOT bridged here.

/// The four family superclasses — the coarse family axis of the 4×4 quadrant
/// grid, lifted from `base_prior`'s own grouping.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum FamilyQuadrant {
    /// Change verbs — high Temporal + Modal (Becomes, Dissolves, Abstracts, Mirrors).
    Change = 0,
    /// Action verbs — high Kausal + Temporal (Causes, Prevents, Transforms; 1 reserve).
    Action = 1,
    /// State verbs — high Modal, low Temporal (Supports, Contradicts, Refines, Grounds).
    State = 2,
    /// Discovery / enablement — high Kausal + Lokal (Enables; 3 reserve).
    Discovery = 3,
}

/// The four tense superclasses — the coarse tense axis (`tense as usize / 3`,
/// matching `Tense`'s declaration order and `tense_modifier`'s class shape).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum TenseQuadrant {
    /// Present / Past / Future — unmarked (zero modifier).
    Simple = 0,
    /// The three continuous aspects — ongoing process.
    Continuous = 1,
    /// Perfect / Pluperfect / FuturePerfect — completion, temporal anchoring.
    Perfect = 2,
    /// Habitual / Potential / Imperative — marked mood.
    Mood = 3,
}

impl VerbFamily {
    /// This family's (quadrant, member-index) coordinate in the coarse grid.
    /// Member indices are dense within each quadrant; unassigned indices are
    /// RESERVED (Action 3; Discovery 1..=3).
    #[must_use]
    pub fn quadrant(self) -> (FamilyQuadrant, u8) {
        match self {
            Self::Becomes => (FamilyQuadrant::Change, 0),
            Self::Dissolves => (FamilyQuadrant::Change, 1),
            Self::Abstracts => (FamilyQuadrant::Change, 2),
            Self::Mirrors => (FamilyQuadrant::Change, 3),
            Self::Causes => (FamilyQuadrant::Action, 0),
            Self::Prevents => (FamilyQuadrant::Action, 1),
            Self::Transforms => (FamilyQuadrant::Action, 2),
            Self::Supports => (FamilyQuadrant::State, 0),
            Self::Contradicts => (FamilyQuadrant::State, 1),
            Self::Refines => (FamilyQuadrant::State, 2),
            Self::Grounds => (FamilyQuadrant::State, 3),
            Self::Enables => (FamilyQuadrant::Discovery, 0),
        }
    }
}

/// A tense's (quadrant, member-index): `(t/3, t%3)` by declaration order.
/// Member 3 of every tense quadrant is RESERVED.
#[must_use]
pub fn tense_quadrant(tense: Tense) -> (TenseQuadrant, u8) {
    let t = tense as u8;
    let q = match t / 3 {
        0 => TenseQuadrant::Simple,
        1 => TenseQuadrant::Continuous,
        2 => TenseQuadrant::Perfect,
        _ => TenseQuadrant::Mood,
    };
    (q, t % 3)
}

/// The cell's one-byte Morton address: `[fq:2|tq:2|fm:2|tm:2]` MSB→LSB. The
/// high nibble `(fq,tq)` is the coarse quadrant pair — two addresses share a
/// coarse cell iff `a >> 4 == b >> 4` (ancestry by nibble).
#[must_use]
pub fn morton_cell(family: VerbFamily, tense: Tense) -> u8 {
    let (fq, fm) = family.quadrant();
    let (tq, tm) = tense_quadrant(tense);
    ((fq as u8) << 6) | ((tq as u8) << 4) | (fm << 2) | tm
}

/// Do two cell addresses share a coarse quadrant (high-nibble ancestry)?
#[must_use]
pub fn same_quadrant(a: u8, b: u8) -> bool {
    a >> 4 == b >> 4
}

/// The coarse-cell centroid: mean of the OCCUPIED member cells' full priors
/// in the `(fq, tq)` quadrant. This is the pyramid's middle level — what an
/// unknown verb resolvable only to a quadrant reads (graceful degradation;
/// Moore's cheap-check-first at the representation level). Reserved members
/// contribute nothing (occupied-mean, never zero-padded).
#[must_use]
pub fn quadrant_prior(fq: FamilyQuadrant, tq: TenseQuadrant) -> SlotPrior {
    let mut sum = [0.0f32; 5];
    let mut n = 0.0f32;
    for family in VerbFamily::ALL {
        if family.quadrant().0 != fq {
            continue;
        }
        for tense in Tense::ALL {
            if tense_quadrant(tense).0 != tq {
                continue;
            }
            let p = base_prior(family).combine(tense_modifier(tense));
            sum[0] += p.temporal;
            sum[1] += p.kausal;
            sum[2] += p.modal;
            sum[3] += p.lokal;
            sum[4] += p.instrument;
            n += 1.0;
        }
    }
    SlotPrior {
        temporal: sum[0] / n,
        kausal: sum[1] / n,
        modal: sum[2] / n,
        lokal: sum[3] / n,
        instrument: sum[4] / n,
    }
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
        // Tense::Present is unmarked (no modifier) so the family-level base
        // prior is preserved. (Under tense modulation, Perfect adds +0.15 to
        // temporal, which would push Refines.temporal from 0.3 to 0.45.)
        let t = default_table();
        let p = t.lookup(VerbFamily::Refines, Tense::Present);
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
        assert!(
            p.instrument > 0.5,
            "Transforms should have elevated instrument"
        );
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
        // Use Tense::Present (unmarked) so the family base prior is preserved.
        // Imperative would suppress temporal by 0.20 (0.85 -> 0.65 < 0.7) and
        // amplify modal — those are tested in `test_imperative_suppresses_temporal`.
        let t = default_table();
        let p = t.lookup(VerbFamily::Dissolves, Tense::Present);
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

    // --- Tense modulation tests (G4 loose end: priors must vary across tenses
    // within a family; broadcast-flat 12-priors-across-12-tenses produces
    // zero tense×family interaction). ---

    /// Failing-test-first: Perfect aspect (completion → temporal anchoring
    /// per Quirk et al. CGEL §4.21–4.27) must yield strictly higher temporal
    /// prior than the unmarked Past for the same family.
    #[test]
    fn test_perfect_amplifies_temporal_within_family() {
        let t = default_table();
        let perfect = t.lookup(VerbFamily::Causes, Tense::Perfect);
        let past = t.lookup(VerbFamily::Causes, Tense::Past);
        assert!(
            perfect.temporal > past.temporal,
            "Perfect should amplify temporal over Past for Causes; got perfect={} past={}",
            perfect.temporal,
            past.temporal
        );
    }

    /// Imperative (timeless command) suppresses temporal in favour of modal.
    #[test]
    fn test_imperative_suppresses_temporal() {
        let t = default_table();
        let imperative = t.lookup(VerbFamily::Causes, Tense::Imperative);
        let present = t.lookup(VerbFamily::Causes, Tense::Present);
        assert!(
            imperative.temporal < present.temporal,
            "Imperative should suppress temporal vs Present for Causes; got imp={} pres={}",
            imperative.temporal,
            present.temporal
        );
        assert!(
            imperative.modal > present.modal,
            "Imperative should amplify modal vs Present for Causes; got imp={} pres={}",
            imperative.modal,
            present.modal
        );
    }

    /// Subjunctive equivalent — this enum has Potential (irrealis/possibility
    /// mood), which fills the Subjunctive role. Potential should amplify modal
    /// over Present.
    #[test]
    fn test_subjunctive_amplifies_modal() {
        let t = default_table();
        let potential = t.lookup(VerbFamily::Supports, Tense::Potential);
        let present = t.lookup(VerbFamily::Supports, Tense::Present);
        assert!(
            potential.modal > present.modal,
            "Potential (subjunctive role) should amplify modal vs Present for Supports; \
             got pot={} pres={}",
            potential.modal,
            present.modal
        );
    }

    /// Sanity: continuous aspects amplify temporal but less than perfect.
    /// Use `Causes` (temporal base 0.4) so neither modifier saturates at 1.0.
    #[test]
    fn test_continuous_amplifies_temporal_less_than_perfect() {
        let t = default_table();
        let cont = t.lookup(VerbFamily::Causes, Tense::PresentContinuous);
        let perf = t.lookup(VerbFamily::Causes, Tense::Perfect);
        let pres = t.lookup(VerbFamily::Causes, Tense::Present);
        assert!(
            cont.temporal > pres.temporal,
            "Continuous > Present temporal"
        );
        assert!(
            perf.temporal > cont.temporal,
            "Perfect > Continuous temporal"
        );
    }

    /// Sanity: clamp to [0,1] holds even when base prior is near saturation.
    #[test]
    fn test_combine_clamps_to_unit_interval() {
        let t = default_table();
        // Causes has kausal=0.95 base; no tense modifier touches kausal,
        // but Perfect adds +0.15 to temporal where Causes.temporal=0.4 → 0.55.
        let p = t.lookup(VerbFamily::Causes, Tense::Perfect);
        assert!(p.temporal >= 0.0 && p.temporal <= 1.0);
        assert!(p.kausal >= 0.0 && p.kausal <= 1.0);
        assert!(p.modal >= 0.0 && p.modal <= 1.0);
        assert!(p.lokal >= 0.0 && p.lokal <= 1.0);
        assert!(p.instrument >= 0.0 && p.instrument <= 1.0);
    }

    // ── The 4×4 Morton cascade layer ──

    /// Every occupied (family, tense) cell gets a UNIQUE byte address, and the
    /// high nibble is exactly the quadrant pair (ancestry by nibble).
    #[test]
    fn morton_addresses_are_unique_and_nibble_ancestored() {
        let mut seen = std::collections::HashSet::new();
        for family in VerbFamily::ALL {
            for tense in Tense::ALL {
                let cell = morton_cell(family, tense);
                assert!(seen.insert(cell), "duplicate address {cell:#04x}");
                let (fq, _) = family.quadrant();
                let (tq, _) = tense_quadrant(tense);
                assert_eq!(cell >> 4, ((fq as u8) << 2) | (tq as u8));
            }
        }
        assert_eq!(seen.len(), 144, "12×12 occupied cells in the 16×16 space");
    }

    /// Ancestry discriminates: same-quadrant pairs share the high nibble,
    /// cross-quadrant pairs do not (fire + stay-silent on non-trivial input).
    #[test]
    fn same_quadrant_fires_and_stays_silent() {
        // Becomes/Dissolves are both Change; Present/Past both Simple.
        let a = morton_cell(VerbFamily::Becomes, Tense::Present);
        let b = morton_cell(VerbFamily::Dissolves, Tense::Past);
        assert!(
            same_quadrant(a, b),
            "Change×Simple pair shares the coarse cell"
        );
        // Causes is Action — different family quadrant.
        let c = morton_cell(VerbFamily::Causes, Tense::Present);
        assert!(!same_quadrant(a, c), "Change vs Action must not share");
        // Same family, mood tense — different tense quadrant.
        let d = morton_cell(VerbFamily::Becomes, Tense::Potential);
        assert!(!same_quadrant(a, d), "Simple vs Mood must not share");
    }

    /// The inverse-pyramid residual probe — MEASURED, then pinned. The claim:
    /// the 144's information lives mostly at the coarse level — each occupied
    /// cell is a small perturbation on its quadrant centroid. Measured on the
    /// shipped starter priors: **mean residual 0.0774** (the pyramid claim
    /// holds on the mean), **max 0.500 = Grounds.lokal** vs the State
    /// quadrant's lokal centroid 0.35 — the largest of a named outlier
    /// catalogue (Grounds.L 0.500, Causes.T 0.300, Contradicts.K 0.279,
    /// Mirrors.L 0.275): the per-family axis SIGNATURES the coarse level
    /// deliberately does not carry. Lokal is the axis the 4-class carve
    /// compresses worst — a real input to the queued wordnet-supersense
    /// alignment tune. Inertness: max must sit AT the Grounds.lokal value
    /// (a smaller max means the priors changed and the pins are stale);
    /// mean far under the ~0.30 spread a shuffled family→quadrant
    /// assignment would produce.
    #[test]
    fn quadrant_centroids_reconstruct_cells_within_measured_residuals() {
        let mut max_res = 0.0f32;
        let mut sum_res = 0.0f32;
        let mut n = 0.0f32;
        for family in VerbFamily::ALL {
            let (fq, _) = family.quadrant();
            for tense in Tense::ALL {
                let (tq, _) = tense_quadrant(tense);
                let cell = base_prior(family).combine(tense_modifier(tense));
                let cen = quadrant_prior(fq, tq);
                for (c, q) in [
                    (cell.temporal, cen.temporal),
                    (cell.kausal, cen.kausal),
                    (cell.modal, cen.modal),
                    (cell.lokal, cen.lokal),
                    (cell.instrument, cen.instrument),
                ] {
                    let r = (c - q).abs();
                    max_res = max_res.max(r);
                    sum_res += r;
                    n += 1.0;
                }
            }
        }
        let mean_res = sum_res / n;
        assert!(
            (0.49..=0.51).contains(&max_res),
            "max residual {max_res:.3} — pinned at Grounds.lokal 0.500; a different \
             value means the priors changed and these pins need re-measuring"
        );
        assert!(
            (0.07..=0.09).contains(&mean_res),
            "mean residual {mean_res:.4} — measured 0.0774: cells are small \
             perturbations on centroids (the pyramid claim, on the mean)"
        );
    }

    /// Graceful degradation: a quadrant centroid preserves its class's
    /// DOMINANT semantic axis — the read an unknown-but-quadrant-resolvable
    /// verb receives is still class-correct.
    #[test]
    fn quadrant_centroids_keep_the_class_signal() {
        // Action×Simple: kausal dominates (Causes/Prevents/Transforms class).
        let action = quadrant_prior(FamilyQuadrant::Action, TenseQuadrant::Simple);
        assert!(action.kausal > action.temporal && action.kausal > action.modal);
        // State×Simple: modal dominates.
        let state = quadrant_prior(FamilyQuadrant::State, TenseQuadrant::Simple);
        assert!(state.modal > state.temporal && state.modal > state.kausal);
        // Change×Simple: temporal+modal high, kausal low.
        let change = quadrant_prior(FamilyQuadrant::Change, TenseQuadrant::Simple);
        assert!(change.temporal > change.kausal && change.modal > change.kausal);
    }
}
