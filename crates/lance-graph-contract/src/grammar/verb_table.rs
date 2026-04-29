//! 144-cell verb-role lookup table — 12 semantic families × 12 tense/aspect/mood.
//!
//! Each cell holds a TEKAMOLO slot prior: which slots a verb of this family
//! and tense expects to be filled. Parsing reduces to (family, tense) →
//! row → fill slots from morphology → NARS-revise truth.
//!
//! Currently uniform priors; future PR populates from corpus statistics.
//!
//! See PR #279 outlook E3 + grammar-landscape.md §9.
//!
//! META-AGENT: `pub mod verb_table;` to mod.rs.

use crate::grammar::role_keys::Tense;

/// Twelve top-level semantic families. The naming is deliberately
/// process-oriented (verbs as transformations on configurations of
/// the world) rather than syntax-oriented — these are the "roles a
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

/// Default table with hand-set families per the plan's table:
///   BECOMES → Temporal + Modal high
///   CAUSES → Subject + Object + Kausal high
///   ...
///
/// Currently a starter — only a few cells are seeded; uniform fills the
/// rest. The numbers are *priors*, not facts: a future PR replaces them
/// with corpus-derived statistics. Mark this `// starter — tune empirically`
/// in any consumer that depends on specific values.
pub fn default_table() -> VerbRoleTable {
    let mut t = VerbRoleTable::new_uniform();
    let high = SlotPrior { temporal: 0.9, kausal: 0.2, modal: 0.7, lokal: 0.3, instrument: 0.2 };
    let causes = SlotPrior { temporal: 0.4, kausal: 0.95, modal: 0.4, lokal: 0.3, instrument: 0.5 };
    // populate per the plan's family table; mark "starter — tune empirically"
    // ... apply for each (family, tense) combination ...
    t.set(VerbFamily::Becomes, Tense::Present, high);
    t.set(VerbFamily::Causes, Tense::Present, causes);
    // (etc — only seed a few; uniform fills the rest)
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
}
