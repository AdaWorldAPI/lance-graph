//! The D-TGB-0 fixtures. Pure data; nothing here decides anything.
//!
//! The address space is twelve rows of four lanes. Rows 0..4 carry givens on
//! lane 0; every other cell starts absent. Each fixture names the plan's
//! wording it stands for, so a later item can say which one it runs on.

use std::collections::{BTreeMap, BTreeSet};

use crate::oracle::{Givens, Version};

/// Rows in the fixture address space.
pub const ROWS: u32 = 12;
/// Lanes per row.
pub const LANES: u8 = 4;

/// One addressable value: a row and a lane within it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Cell {
    pub row: u32,
    pub lane: u8,
}

/// Shorthand for a [`Cell`].
#[must_use]
pub const fn cell(row: u32, lane: u8) -> Cell {
    Cell { row, lane }
}

/// How a thought's write intention ends on the Rubicon lifecycle.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Outcome {
    /// `Planning → CognitiveWork → Evaluation → Commit`: the result lands.
    Commit,
    /// `… → Evaluation → Plan`: back to the in-memory work, nothing lands.
    Plan,
    /// `… → Evaluation → Prune`: the intention is dropped after evaluation.
    Prune,
    /// `Planning → Prune`: the pre-Rubicon veto, before any work.
    Veto,
}

/// A thought: its writes (one per cell) and how its write intention ends.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Thought {
    pub id: u32,
    pub writes: Vec<(Cell, u64)>,
    pub outcome: Outcome,
}

impl Thought {
    /// A thought; panics if it writes one cell twice or leaves the address
    /// space, because every fixture below relies on one write per cell.
    #[must_use]
    pub fn new(id: u32, writes: Vec<(Cell, u64)>, outcome: Outcome) -> Self {
        let mut seen = BTreeSet::new();
        for &(c, _) in &writes {
            assert!(
                c.row < ROWS && c.lane < LANES,
                "cell {c:?} outside the fixture space"
            );
            assert!(seen.insert(c), "thought {id} writes {c:?} twice");
        }
        Self {
            id,
            writes,
            outcome,
        }
    }
}

/// Layer 0: row r (r < 4) has value `100 + r` on lane 0.
#[must_use]
pub fn givens() -> Givens {
    Givens((0..4).map(|r| (cell(r, 0), 100 + u64::from(r))).collect())
}

/// The empty previous version (main before any thought landed).
#[must_use]
pub fn main_version() -> Version {
    Version::default()
}

/// One write intention per Rubicon outcome, each writing one absent cell.
#[must_use]
pub fn rubicon_outcomes() -> [Thought; 4] {
    [
        Thought::new(10, vec![(cell(4, 1), 41)], Outcome::Commit),
        Thought::new(11, vec![(cell(5, 1), 51)], Outcome::Plan),
        Thought::new(12, vec![(cell(6, 1), 61)], Outcome::Prune),
        Thought::new(13, vec![(cell(7, 1), 71)], Outcome::Veto),
    ]
}

/// A stack of unremarkable thoughts: disjoint absent cells, one agreeing
/// write to a given (agreement is not an anomaly), no conflicts.
#[must_use]
pub fn unremarkable_stack() -> Vec<Thought> {
    vec![
        Thought::new(
            20,
            vec![(cell(8, 1), 81), (cell(8, 2), 82)],
            Outcome::Commit,
        ),
        Thought::new(21, vec![(cell(9, 1), 91)], Outcome::Commit),
        Thought::new(
            22,
            vec![(cell(10, 1), 101), (cell(0, 0), 100)],
            Outcome::Commit,
        ),
        Thought::new(23, vec![(cell(11, 3), 113)], Outcome::Commit),
    ]
}

/// Contradiction: offers a value for a given that differs from it.
#[must_use]
pub fn contradiction_thought() -> Thought {
    Thought::new(
        30,
        vec![(cell(1, 0), 999), (cell(4, 2), 42)],
        Outcome::Commit,
    )
}

/// Divergence: two thoughts landing together disagree on an absent cell.
#[must_use]
pub fn divergent_pair() -> [Thought; 2] {
    [
        Thought::new(31, vec![(cell(5, 2), 7), (cell(6, 2), 62)], Outcome::Commit),
        Thought::new(32, vec![(cell(5, 2), 8)], Outcome::Commit),
    ]
}

/// Regression: the row sequence attention visits; row 3 is returned to.
pub const REGRESSION_SCANPATH: [u32; 5] = [1, 3, 2, 3, 4];
/// The same shape without a return.
pub const SMOOTH_SCANPATH: [u32; 4] = [1, 2, 3, 4];

/// Cold/hot surprise: rows an independent route expected, and rows attended.
pub const SURPRISE_EXPECTED: [u32; 3] = [1, 2, 3];
pub const SURPRISE_ATTENDED: [u32; 3] = [1, 2, 4];

/// The lane no fixture writes: every cell on lane 3 except (11, 3).
#[must_use]
pub fn unwritten_cells() -> Vec<Cell> {
    (0..ROWS).filter(|&r| r != 11).map(|r| cell(r, 3)).collect()
}

/// Every write of every fixture thought, for the absence check.
#[must_use]
pub fn all_writes() -> BTreeMap<Cell, Vec<u64>> {
    let mut m: BTreeMap<Cell, Vec<u64>> = BTreeMap::new();
    let mut add = |t: &Thought| {
        for &(c, v) in &t.writes {
            m.entry(c).or_default().push(v);
        }
    };
    rubicon_outcomes().iter().for_each(&mut add);
    unremarkable_stack().iter().for_each(&mut add);
    add(&contradiction_thought());
    divergent_pair().iter().for_each(&mut add);
    m
}
