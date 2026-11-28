//! The from-scratch oracle: the next version computed from the givens, the
//! previous version and the thoughts that land, with no incremental state.
//!
//! The rules are the ones `thought-git-basics-v1.md` reads out of the existing
//! code (correction item 5) and the capstone's givens row:
//!
//! 1. Only a thought whose outcome is `Commit` lands. `Plan`, `Prune` and the
//!    pre-Rubicon veto leave the version unchanged.
//! 2. A cell with a given (layer 0) is never overwritten. An offered value
//!    equal to the given is agreement and is not stored; a different value is
//!    refused and recorded as a [`Contradiction`].
//! 3. A cell without a given takes the landed value. If thoughts landing in
//!    the same version offer different values for it, nothing is stored and a
//!    [`Divergence`] is recorded; the cell keeps its previous state.
//! 4. A cell nobody wrote reads as absent (`None`), never as zero.

use std::collections::{BTreeMap, BTreeSet};

use crate::fixtures::{Cell, Outcome, Thought};

/// Layer 0: hydrated values that no thought may overwrite.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Givens(pub BTreeMap<Cell, u64>);

/// A refused write to a given.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct Contradiction {
    pub cell: Cell,
    pub given: u64,
    pub offered: u64,
    pub thought: u32,
}

/// Two or more thoughts landing in one version offered different values for
/// one cell without a given.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Divergence {
    pub cell: Cell,
    /// `(value, thought)`, sorted.
    pub offers: Vec<(u64, u32)>,
}

/// One version: the cells written above layer 0, plus the records of what was
/// refused while producing it.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Version {
    pub cells: BTreeMap<Cell, u64>,
    pub contradictions: Vec<Contradiction>,
    pub divergences: Vec<Divergence>,
}

impl Version {
    /// What a reader sees at `cell`: the given if there is one, else the
    /// landed value, else absent.
    #[must_use]
    pub fn read(&self, givens: &Givens, cell: Cell) -> Option<u64> {
        givens
            .0
            .get(&cell)
            .copied()
            .or_else(|| self.cells.get(&cell).copied())
    }
}

/// The next version, computed from scratch.
#[must_use]
pub fn next_version(givens: &Givens, prior: &Version, landing: &[Thought]) -> Version {
    let mut cells = prior.cells.clone();
    let mut contradictions = Vec::new();
    let mut offers: BTreeMap<Cell, BTreeSet<(u64, u32)>> = BTreeMap::new();

    for t in landing.iter().filter(|t| t.outcome == Outcome::Commit) {
        for &(cell, value) in &t.writes {
            match givens.0.get(&cell) {
                Some(&given) if given == value => {}
                Some(&given) => contradictions.push(Contradiction {
                    cell,
                    given,
                    offered: value,
                    thought: t.id,
                }),
                None => {
                    offers.entry(cell).or_default().insert((value, t.id));
                }
            }
        }
    }

    let mut divergences = Vec::new();
    for (cell, set) in offers {
        let values: BTreeSet<u64> = set.iter().map(|&(v, _)| v).collect();
        if values.len() == 1 {
            cells.insert(cell, *values.iter().next().expect("one value"));
        } else {
            divergences.push(Divergence {
                cell,
                offers: set.into_iter().collect(),
            });
        }
    }

    contradictions.sort();
    Version {
        cells,
        contradictions,
        divergences,
    }
}

/// Squash a stack into one thought: the union of its writes, later writes
/// replacing earlier ones at the same cell. This is the naive squash; the
/// tests show why it is only safe on a stack without anomalies.
#[must_use]
pub fn squash(id: u32, stack: &[Thought]) -> Thought {
    let mut merged: BTreeMap<Cell, u64> = BTreeMap::new();
    for t in stack {
        for &(cell, value) in &t.writes {
            merged.insert(cell, value);
        }
    }
    Thought::new(id, merged.into_iter().collect(), Outcome::Commit)
}
