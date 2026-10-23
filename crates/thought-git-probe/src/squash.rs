//! D-TGA-5: the squash gate, moved here from `tests/squash_gate.rs` so that
//! D-TGA-6 (`tests/protected_readers.rs`) can use it. The semantics are those
//! pre-registered in
//! `.claude/board/entries/2026-10-23-d-tga-5-squash-gate-preregistration.md`:
//! strict R1/R2 over landed final states, Commit-only (P8), blame as a lane
//! bitset, the ordered history and its seal. Lab only.

use std::collections::{BTreeMap, BTreeSet};

use crate::fixtures::{Cell, Outcome, Thought};
use crate::oracle::Givens;

/// A thought lane: its id (1..=63) and its chain of origin, oldest first.
pub struct Lane {
    pub id: u8,
    pub chain: Vec<Thought>,
}

impl Lane {
    #[must_use]
    pub fn new(id: u8, chain: Vec<Thought>) -> Self {
        assert!((1..=63).contains(&id), "lane {id} outside 1..=63");
        Self { id, chain }
    }

    /// The thoughts whose writes land: only `Outcome::Commit`, as in
    /// `next_version`. Plan, Prune and Veto stay in the history only.
    pub fn landed(&self) -> impl Iterator<Item = &Thought> {
        self.chain.iter().filter(|t| t.outcome == Outcome::Commit)
    }

    /// Total landed writes in the chain.
    #[must_use]
    pub fn ops(&self) -> usize {
        self.landed().map(|t| t.writes.len()).sum()
    }

    /// The landed chain folded in order: later writes replace earlier ones.
    #[must_use]
    pub fn final_state(&self) -> BTreeMap<Cell, u64> {
        let mut f = BTreeMap::new();
        for t in self.landed() {
            for &(c, v) in &t.writes {
                f.insert(c, v);
            }
        }
        f
    }
}

#[derive(Debug, PartialEq, Eq)]
pub enum Refusal {
    Contradiction {
        cell: Cell,
        given: u64,
        offered: u64,
        lane: u8,
    },
    Divergence {
        cell: Cell,
        offers: Vec<(u64, u8)>,
    },
}

#[derive(Debug)]
pub struct Squashed {
    pub target: u8,
    pub cells: BTreeMap<Cell, u64>,
    pub order: Vec<(u8, u32)>,
    pub blame: BTreeMap<Cell, u64>,
    pub seal: u64,
}

impl Squashed {
    /// The squash as one committed thought, for the oracle.
    #[must_use]
    pub fn as_thought(&self, id: u32) -> Thought {
        Thought::new(
            id,
            self.cells.iter().map(|(&c, &v)| (c, v)).collect(),
            Outcome::Commit,
        )
    }
}

fn fnv1a(seal: &mut u64, x: u64) {
    for b in x.to_le_bytes() {
        *seal ^= u64::from(b);
        *seal = seal.wrapping_mul(0x0000_0100_0000_01b3);
    }
}

/// The gate. Lanes are taken in ascending id order whatever order they are
/// passed in.
/// # Errors
///
/// [`Refusal::Contradiction`] when a lane's final state overwrites a given;
/// [`Refusal::Divergence`] when two lanes' final states disagree on a cell.
pub fn squash_gate(g: &Givens, lanes: &[Lane]) -> Result<Squashed, Refusal> {
    let mut lanes: Vec<&Lane> = lanes.iter().collect();
    lanes.sort_by_key(|l| l.id);

    // Target: most ops; on a tie the lowest id (max_by_key keeps the last
    // maximum, so compare on (ops, Reverse(id))).
    let target = lanes
        .iter()
        .max_by_key(|l| (l.ops(), std::cmp::Reverse(l.id)))
        .map_or(0, |l| l.id);

    // R1, then R2, over final states only.
    let mut offers: BTreeMap<Cell, BTreeSet<(u64, u8)>> = BTreeMap::new();
    for l in &lanes {
        for (c, v) in l.final_state() {
            if let Some(&given) = g.0.get(&c) {
                if given != v {
                    return Err(Refusal::Contradiction {
                        cell: c,
                        given,
                        offered: v,
                        lane: l.id,
                    });
                }
                continue; // agreement with a given is not stored
            }
            offers.entry(c).or_default().insert((v, l.id));
        }
    }
    let mut cells = BTreeMap::new();
    for (c, set) in offers {
        let values: BTreeSet<u64> = set.iter().map(|&(v, _)| v).collect();
        if values.len() > 1 {
            return Err(Refusal::Divergence {
                cell: c,
                offers: set.into_iter().collect(),
            });
        }
        cells.insert(c, *values.iter().next().expect("one value"));
    }

    let mut order = Vec::new();
    let mut blame: BTreeMap<Cell, u64> = BTreeMap::new();
    let mut seal = 0xcbf2_9ce4_8422_2325_u64;
    for l in &lanes {
        for t in &l.chain {
            order.push((l.id, t.id));
            fnv1a(&mut seal, u64::from(l.id));
            fnv1a(&mut seal, u64::from(t.id));
            for &(c, v) in &t.writes {
                if t.outcome == Outcome::Commit {
                    *blame.entry(c).or_default() |= 1u64 << l.id;
                }
                fnv1a(&mut seal, u64::from(c.row));
                fnv1a(&mut seal, u64::from(c.lane));
                fnv1a(&mut seal, v);
            }
        }
    }

    Ok(Squashed {
        target,
        cells,
        order,
        blame,
        seal,
    })
}
