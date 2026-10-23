//! D-TGA-5: the squash gate, moved here from `tests/squash_gate.rs` so that
//! D-TGA-5D (`tests/protected_readers.rs`) can use it. The semantics are those
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

// ── D-TGA-5D: protected readers ─────────────────────────────────────────────
//
// Pre-registered in
// `.claude/board/entries/2026-10-23-d-tga-5d-protected-readers-preregistration.md`.
// A squash replaces a history with one delta. A reader is anything that may
// later ask what happened. The gate below answers every protected reader from
// the RETAINED data only (the squash, its witness, the prior version and the
// givens), compares that with the answer the full history gives, and refuses
// when a protected reader cannot be answered. Refusal is a valid result.

use crate::oracle::{Contradiction, Divergence, Version, next_version};

/// What may later be asked of a squashed family.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Reader {
    /// The read at every cell after the last step.
    Final,
    /// The read at every cell after each step `1 <= k < T`.
    Intermediate,
    /// Every contradiction any step recorded.
    Contradictions,
    /// Every divergence any step recorded.
    Divergences,
    /// Committed writes per cell, counted from the thought log.
    Multiplicity,
}

/// What a protected squash retains beyond the [`Squashed`] value. Only the
/// fields a protected reader needs are filled.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Witness {
    pub contradictions: Vec<Contradiction>,
    pub divergences: Vec<Divergence>,
    pub multiplicity: BTreeMap<Cell, u32>,
}

/// Why a protected squash was refused.
#[derive(Debug, PartialEq, Eq)]
pub enum ProtectedRefusal {
    /// R1 or R2, unchanged from [`squash_gate`].
    Gate(Refusal),
    /// The retained data cannot answer this protected reader the way the
    /// history does.
    Unretainable(Reader),
}

/// The history: step `i` lands every lane's `i`-th thought together in one
/// version, starting from `prior`. Returns `v_1 .. v_T`.
#[must_use]
pub fn land_interleaved(g: &Givens, prior: &Version, lanes: &[Lane]) -> Vec<Version> {
    let steps = lanes.iter().map(|l| l.chain.len()).max().unwrap_or(0);
    let mut out = Vec::with_capacity(steps);
    let mut v = prior.clone();
    for i in 0..steps {
        let landing: Vec<Thought> = lanes
            .iter()
            .filter_map(|l| l.chain.get(i).cloned())
            .collect();
        v = next_version(g, &v, &landing);
        out.push(v.clone());
    }
    out
}

/// Every cell a reader compares: the givens, the prior's cells and every cell
/// any thought offered.
fn touched(g: &Givens, prior: &Version, lanes: &[Lane]) -> BTreeSet<Cell> {
    let mut s: BTreeSet<Cell> = g.0.keys().copied().collect();
    s.extend(prior.cells.keys().copied());
    for l in lanes {
        for t in &l.chain {
            s.extend(t.writes.iter().map(|&(c, _)| c));
        }
    }
    s
}

/// The reads of one version over `cells`.
fn reads(g: &Givens, v: &Version, cells: &BTreeSet<Cell>) -> Vec<Option<u64>> {
    cells.iter().map(|&c| v.read(g, c)).collect()
}

/// History answer: every contradiction, in step order.
#[must_use]
pub fn history_contradictions(history: &[Version]) -> Vec<Contradiction> {
    history
        .iter()
        .flat_map(|v| v.contradictions.iter().copied())
        .collect()
}

/// History answer: every divergence, in step order.
#[must_use]
pub fn history_divergences(history: &[Version]) -> Vec<Divergence> {
    history
        .iter()
        .flat_map(|v| v.divergences.iter().cloned())
        .collect()
}

/// History answer: committed writes per cell, from the thought log.
#[must_use]
pub fn history_multiplicity(lanes: &[Lane]) -> BTreeMap<Cell, u32> {
    let mut m = BTreeMap::new();
    for l in lanes {
        for t in l.landed() {
            for &(c, _) in &t.writes {
                *m.entry(c).or_default() += 1;
            }
        }
    }
    m
}

/// The best multiplicity a bare [`Squashed`] can give: blame's popcount,
/// which counts lanes, not writes.
#[must_use]
pub fn squashed_multiplicity(s: &Squashed) -> BTreeMap<Cell, u32> {
    s.blame.iter().map(|(&c, &b)| (c, b.count_ones())).collect()
}

/// Squash `lanes` on top of `prior`, keeping every reader in `protect`.
///
/// # Errors
///
/// [`ProtectedRefusal::Gate`] when R1 or R2 refuses;
/// [`ProtectedRefusal::Unretainable`] when a protected reader's retained
/// answer would differ from the history's.
pub fn protected_squash(
    g: &Givens,
    prior: &Version,
    lanes: &[Lane],
    protect: &[Reader],
) -> Result<(Squashed, Witness), ProtectedRefusal> {
    let s = squash_gate(g, lanes).map_err(ProtectedRefusal::Gate)?;
    let history = land_interleaved(g, prior, lanes);
    let wants = |r: Reader| protect.contains(&r);

    let mut w = Witness::default();
    if wants(Reader::Contradictions) {
        w.contradictions = history_contradictions(&history);
    }
    if wants(Reader::Divergences) {
        w.divergences = history_divergences(&history);
    }
    if wants(Reader::Multiplicity) {
        w.multiplicity = history_multiplicity(lanes);
    }

    // Verify every protected reader from the retained data alone.
    let cells = touched(g, prior, lanes);
    let landed = next_version(g, prior, &[s.as_thought(u32::MAX)]);
    let last = history.last().unwrap_or(prior);
    if wants(Reader::Final) && reads(g, &landed, &cells) != reads(g, last, &cells) {
        return Err(ProtectedRefusal::Unretainable(Reader::Final));
    }
    if wants(Reader::Intermediate) {
        // The retained data keeps no per-step state: the only intermediate
        // answer it can give is the final one.
        let fin = reads(g, &landed, &cells);
        if history[..history.len().saturating_sub(1)]
            .iter()
            .any(|v| reads(g, v, &cells) != fin)
        {
            return Err(ProtectedRefusal::Unretainable(Reader::Intermediate));
        }
    }
    if wants(Reader::Contradictions) && w.contradictions != history_contradictions(&history) {
        return Err(ProtectedRefusal::Unretainable(Reader::Contradictions));
    }
    if wants(Reader::Divergences) && w.divergences != history_divergences(&history) {
        return Err(ProtectedRefusal::Unretainable(Reader::Divergences));
    }
    if wants(Reader::Multiplicity) && w.multiplicity != history_multiplicity(lanes) {
        return Err(ProtectedRefusal::Unretainable(Reader::Multiplicity));
    }
    Ok((s, w))
}

/// Bytes, by a fixed encoding: a cell is 5 bytes (row u32 + lane u8), a value
/// 8, a lane id 1, a thought id 4, a count 4.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Bytes {
    /// Every write of every thought, landed or not: `(cell, value)`.
    pub transient: usize,
    /// delta + blame + order + seal.
    pub squash: usize,
    /// The witness alone.
    pub witness: usize,
}

const CELL: usize = 5;

/// The byte accounting of one protected squash.
#[must_use]
pub fn bytes(lanes: &[Lane], s: &Squashed, w: &Witness) -> Bytes {
    let transient = lanes
        .iter()
        .flat_map(|l| l.chain.iter())
        .map(|t| t.writes.len() * (CELL + 8))
        .sum();
    let squash =
        s.cells.len() * (CELL + 8) + s.blame.len() * (CELL + 8) + s.order.len() * (1 + 4) + 8;
    let witness = w.contradictions.len() * (CELL + 8 + 8 + 4)
        + w.divergences
            .iter()
            .map(|d| CELL + d.offers.len() * (8 + 4))
            .sum::<usize>()
        + w.multiplicity.len() * (CELL + 4);
    Bytes {
        transient,
        squash,
        witness,
    }
}
