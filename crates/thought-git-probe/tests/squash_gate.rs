//! D-TGA-5 lab precursor: squash every lane's chain of origin into the
//! most-updated lane, gated on contradiction, with blame preserved.
//!
//! Pre-registered in
//! `.claude/board/entries/2026-10-23-d-tga-5-squash-gate-preregistration.md`
//! before this file was written. Predictions P1..P7 are the tests below; the
//! disables D1..D5 are named on the test each must turn red.
//!
//! The gate lives here, in the test, because it is a lab question: it does
//! not change `oracle.rs`, `slots.rs` or `receipt.rs`.

use std::collections::{BTreeMap, BTreeSet};

use thought_git_probe::fixtures::{Cell, Outcome, Thought, cell, givens, main_version};
use thought_git_probe::oracle::{Givens, Version, next_version, squash};

/// A thought lane: its id (1..=63) and its chain of origin, oldest first.
struct Lane {
    id: u8,
    chain: Vec<Thought>,
}

impl Lane {
    fn new(id: u8, chain: Vec<Thought>) -> Self {
        assert!((1..=63).contains(&id), "lane {id} outside 1..=63");
        Self { id, chain }
    }

    /// Total writes in the chain.
    fn ops(&self) -> usize {
        self.chain.iter().map(|t| t.writes.len()).sum()
    }

    /// The chain folded in order: later writes replace earlier ones.
    fn final_state(&self) -> BTreeMap<Cell, u64> {
        let mut f = BTreeMap::new();
        for t in &self.chain {
            for &(c, v) in &t.writes {
                f.insert(c, v);
            }
        }
        f
    }
}

#[derive(Debug, PartialEq, Eq)]
enum Refusal {
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
struct Squashed {
    target: u8,
    cells: BTreeMap<Cell, u64>,
    order: Vec<(u8, u32)>,
    blame: BTreeMap<Cell, u64>,
    seal: u64,
}

impl Squashed {
    /// The squash as one committed thought, for the oracle.
    fn as_thought(&self, id: u32) -> Thought {
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
fn squash_gate(g: &Givens, lanes: &[Lane]) -> Result<Squashed, Refusal> {
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
                *blame.entry(c).or_default() |= 1u64 << l.id;
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

fn commit(id: u32, writes: Vec<(Cell, u64)>) -> Thought {
    Thought::new(id, writes, Outcome::Commit)
}

/// Three lanes, no anomaly: lane 7 rewrites its own cell twice, lanes 3 and 7
/// agree on (6, 2), lane 12 has the most ops and agrees with a given.
fn clean_lanes() -> Vec<Lane> {
    vec![
        Lane::new(
            3,
            vec![commit(1, vec![(cell(4, 1), 41)]), commit(2, vec![(cell(6, 2), 62)])],
        ),
        Lane::new(
            7,
            vec![
                commit(3, vec![(cell(5, 1), 50)]),
                commit(4, vec![(cell(5, 1), 51)]),
                commit(5, vec![(cell(5, 1), 52), (cell(6, 2), 62)]),
            ],
        ),
        Lane::new(
            12,
            vec![
                commit(6, vec![(cell(8, 1), 81), (cell(8, 2), 82), (cell(0, 0), 100)]),
                commit(7, vec![(cell(9, 1), 91), (cell(9, 2), 92), (cell(10, 1), 101)]),
            ],
        ),
    ]
}

/// The cells every fixture here touches, for the read comparison.
fn probe_cells() -> Vec<Cell> {
    (0..12)
        .flat_map(|r| (0..4).map(move |l| cell(r, l)))
        .collect()
}

/// P1. Kill condition if it fails: squash-then-land is not land-the-finals.
#[test]
fn p1_landing_the_squash_equals_landing_every_final_state() {
    let g = givens();
    let lanes = clean_lanes();
    let s = squash_gate(&g, &lanes).expect("clean lanes squash");

    let finals: Vec<Thought> = lanes
        .iter()
        .map(|l| commit(1000 + u32::from(l.id), l.final_state().into_iter().collect()))
        .collect();
    let via_finals = next_version(&g, &main_version(), &finals);
    let via_squash = next_version(&g, &main_version(), &[s.as_thought(999)]);

    assert_eq!(via_squash, via_finals, "squash differs from landing the finals");
    for c in probe_cells() {
        assert_eq!(via_squash.read(&g, c), via_finals.read(&g, c), "read at {c:?}");
    }
    // Anti-vacuity: the comparison covers a rewritten cell and a shared cell.
    assert_eq!(via_squash.read(&g, cell(5, 1)), Some(52));
    assert_eq!(via_squash.read(&g, cell(6, 2)), Some(62));
}

/// P2. Disable D1 (no divergence check) turns this red.
#[test]
fn p2_two_lanes_that_disagree_are_refused() {
    let g = givens();
    let lanes = vec![
        Lane::new(4, vec![commit(1, vec![(cell(5, 2), 7)])]),
        Lane::new(9, vec![commit(2, vec![(cell(5, 2), 8)])]),
    ];
    assert_eq!(
        squash_gate(&g, &lanes).err(),
        Some(Refusal::Divergence {
            cell: cell(5, 2),
            offers: vec![(7, 4), (8, 9)],
        })
    );
    // The naive squash keeps the later value on the same input: the gate is
    // what stops a silent overwrite.
    let naive = squash(1, &[lanes[0].chain[0].clone(), lanes[1].chain[0].clone()]);
    assert_eq!(naive.writes, vec![(cell(5, 2), 8)]);
}

/// P3. Disable D2 (no contradiction check) turns this red.
#[test]
fn p3_a_lane_that_overwrites_a_given_is_refused() {
    let g = givens();
    let lanes = vec![
        Lane::new(2, vec![commit(1, vec![(cell(4, 2), 42)])]),
        Lane::new(5, vec![commit(2, vec![(cell(1, 0), 999)])]),
    ];
    assert_eq!(
        squash_gate(&g, &lanes).err(),
        Some(Refusal::Contradiction {
            cell: cell(1, 0),
            given: 101,
            offered: 999,
            lane: 5,
        })
    );
}

/// P4, the silence twin. Disable D3 (divergence over every chain entry)
/// turns this red: lane 7's own rewrites of (5, 1) would read as a
/// disagreement.
#[test]
fn p4_rewrites_in_one_lane_and_agreement_across_lanes_are_accepted() {
    let g = givens();
    let lanes = clean_lanes();
    // Non-trivial: the input really has an in-lane rewrite and a shared cell.
    let rewrites: BTreeSet<u64> = lanes[1]
        .chain
        .iter()
        .flat_map(|t| t.writes.iter())
        .filter(|(c, _)| *c == cell(5, 1))
        .map(|&(_, v)| v)
        .collect();
    assert_eq!(rewrites.len(), 3, "fixture lost its in-lane rewrite");
    assert!(squash_gate(&g, &lanes).is_ok());
}

/// P5. Disable D4 (target = lowest lane id) turns this red.
#[test]
fn p5_the_target_is_the_most_updated_lane() {
    let lanes = clean_lanes();
    let ops: Vec<(u8, usize)> = lanes.iter().map(|l| (l.id, l.ops())).collect();
    assert_eq!(ops, vec![(3, 2), (7, 4), (12, 6)]);
    let s = squash_gate(&givens(), &lanes).expect("clean lanes squash");
    assert_eq!(s.target, 12);

    // A tie goes to the lowest id, deterministically.
    let tie = vec![
        Lane::new(20, vec![commit(1, vec![(cell(4, 1), 1)])]),
        Lane::new(11, vec![commit(2, vec![(cell(4, 2), 2)])]),
    ];
    assert_eq!(squash_gate(&givens(), &tie).expect("tie squash").target, 11);
}

/// P6. Disable D5 (blame keeps only the target's bit) turns this red.
#[test]
fn p6_blame_names_every_lane_that_wrote_a_cell() {
    let lanes = clean_lanes();
    let s = squash_gate(&givens(), &lanes).expect("clean lanes squash");

    // Independent reference from the raw chains.
    let mut expected: BTreeMap<Cell, u64> = BTreeMap::new();
    for l in &lanes {
        for t in &l.chain {
            for &(c, _) in &t.writes {
                *expected.entry(c).or_default() |= 1u64 << l.id;
            }
        }
    }
    assert_eq!(s.blame, expected);
    assert_eq!(
        s.blame[&cell(6, 2)],
        (1 << 3) | (1 << 7),
        "a cell two lanes agreed on keeps both lanes"
    );
    assert_eq!(s.blame[&cell(0, 0)], 1 << 12, "agreement with a given is still blamed");
}

/// P7. The stored delta does not grow with the history.
#[test]
fn p7_the_delta_stays_n_cells_while_the_history_grows() {
    const N: u32 = 8;
    let mut rows = Vec::new();
    for k in [1u32, 2, 4, 8, 16, 32, 64] {
        let chain: Vec<Thought> = (0..k)
            .map(|round| {
                commit(
                    round,
                    (0..N)
                        .map(|i| (cell(4 + i, 1), u64::from(round * 100 + i)))
                        .collect(),
                )
            })
            .collect();
        let lane = Lane::new(1, chain);
        let history = lane.ops();
        let s = squash_gate(&givens(), &[lane]).expect("one lane squash");
        rows.push((k, history, s.cells.len(), s.order.len()));
        assert_eq!(s.cells.len(), N as usize, "delta at k = {k}");
        assert_eq!(history, (k * N) as usize, "history at k = {k}");
        assert_eq!(s.order.len(), k as usize, "order is one entry per thought");
        // The delta holds the LAST round's values.
        assert_eq!(s.cells[&cell(4, 1)], u64::from((k - 1) * 100));
    }
    println!("k, history writes, delta cells, order entries: {rows:?}");
}

/// The seal changes when the history changes, and only then.
#[test]
fn the_seal_depends_on_the_ordered_history() {
    let g = givens();
    let a = squash_gate(&g, &clean_lanes()).expect("squash").seal;
    let b = squash_gate(&g, &clean_lanes()).expect("squash").seal;
    assert_eq!(a, b, "same history, same seal");

    let mut changed = clean_lanes();
    changed[1].chain.swap(0, 1); // same writes, other order
    let c = squash_gate(&g, &changed).expect("squash").seal;
    assert_ne!(a, c, "reordering the chain must change the seal");
}

/// The squash agrees with the oracle's own divergence record: where the
/// gate refuses, landing the finals records a `Divergence` on the same cell.
#[test]
fn a_refused_squash_is_a_divergence_the_oracle_also_records() {
    let g = givens();
    let lanes = vec![
        Lane::new(4, vec![commit(1, vec![(cell(5, 2), 7)])]),
        Lane::new(9, vec![commit(2, vec![(cell(5, 2), 8)])]),
    ];
    let finals: Vec<Thought> = lanes
        .iter()
        .map(|l| commit(u32::from(l.id), l.final_state().into_iter().collect()))
        .collect();
    let v: Version = next_version(&g, &main_version(), &finals);
    assert_eq!(v.divergences.len(), 1);
    assert_eq!(v.divergences[0].cell, cell(5, 2));
    assert!(squash_gate(&g, &lanes).is_err());
}
