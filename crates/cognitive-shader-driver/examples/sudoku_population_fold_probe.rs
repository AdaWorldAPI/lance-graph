//! D-PUZZLE-0, step 1: Sudoku at scale on the `EpistemicState5` population
//! algebra.
//!
//! The domain-free half (the propagation reading, the five questions, the
//! folds, the oracle) is `shared/population_fold.rs`, shared with the
//! crossword probe. This file holds only Sudoku: its claims, its law (naked
//! singles) and its own counters.
//!
//! # The corpus
//!
//! The Sudoku Wikipedia article's example puzzle (the same literal as
//! `lance-graph-ogar/examples/sudoku_cognitive_corpus_probe.rs`, which solves
//! it completely by naked singles), re-labelled and re-ordered by Sudoku's own
//! symmetries: digit permutation, band and stack permutation, row and column
//! permutation within a band or stack, transposition. Each copy is a valid
//! puzzle with the transformed solution, so the base solution checks every
//! copy's solve independently of the propagation order. Each copy is
//! snapshotted after `k` naked-single placements, `k` varying across copies.
//!
//! # The claims
//!
//! Claim = "cell c holds digit d", one edge per claim not yet eliminated, in
//! the shared propagation reading: a given clue is `GIVEN`, a naked-single
//! placement `FORCED`, the true digit of a cell not yet placed `ENTAILED`
//! (the puzzle has one solution, checked), every other surviving candidate
//! `CANDIDATE`. Class `0x0906` is this probe's own.
//!
//! # Not decided here
//!
//! - No solver claim beyond naked singles; the base puzzle needs nothing else.
//! - No SIMD: the folds are plain loops over `&[CausalEdge64]` (all SIMD comes
//!   from `ndarray::simd`; whether the compiler vectorizes these loops is not
//!   claimed). Timings are this machine's, reported with the build's flags.
//!
//! Run: `cargo run --release -p cognitive-shader-driver --example sudoku_population_fold_probe`
//! Tests: `cargo test -p cognitive-shader-driver --example sudoku_population_fold_probe`

use std::time::Instant;

use lance_graph_contract::class_view::ClassId;

#[path = "shared/population_fold.rs"]
mod population_fold;
use population_fold::{
    admit, check_three_ways, declarations, per_group, report, Lane, Rng, CANDIDATE, ENTAILED,
    FORCED, GIVEN,
};

/// The probe's own class (no production class declares a Sudoku reading).
const SUDOKU_CLASS: ClassId = 0x0906;

type Grid = [u8; 81];

/// The Sudoku Wikipedia article's example puzzle, `0` = empty (30 givens).
const PUZZLE: Grid = [
    5, 3, 0, 0, 7, 0, 0, 0, 0, //
    6, 0, 0, 1, 9, 5, 0, 0, 0, //
    0, 9, 8, 0, 0, 0, 0, 6, 0, //
    8, 0, 0, 0, 6, 0, 0, 0, 3, //
    4, 0, 0, 8, 0, 3, 0, 0, 1, //
    7, 0, 0, 0, 2, 0, 0, 0, 6, //
    0, 6, 0, 0, 0, 0, 2, 8, 0, //
    0, 0, 0, 4, 1, 9, 0, 0, 5, //
    0, 0, 0, 0, 8, 0, 0, 7, 9, //
];

fn peers() -> Vec<[u8; 20]> {
    (0..81)
        .map(|i| {
            let (r, c) = (i / 9, i % 9);
            let mut out = [0u8; 20];
            let mut n = 0;
            for j in 0..81 {
                let (r2, c2) = (j / 9, j % 9);
                let same_box = r / 3 == r2 / 3 && c / 3 == c2 / 3;
                if j != i && (r == r2 || c == c2 || same_box) {
                    out[n] = j as u8;
                    n += 1;
                }
            }
            assert_eq!(n, 20);
            out
        })
        .collect()
}

/// The state after the givens and at most `depth` naked-single placements
/// (lowest cell first): the grid, each cell's surviving candidates (bit `d`
/// for digit `d`), and the cells placed by propagation, in order.
struct Snapshot {
    grid: Grid,
    candidates: [u16; 81],
    derived: Vec<u8>,
}

fn propagate(puzzle: &Grid, depth: usize, peers: &[[u8; 20]]) -> Snapshot {
    let mut grid = *puzzle;
    let mut candidates = [0u16; 81];
    for i in 0..81 {
        if grid[i] == 0 {
            let seen = peers[i]
                .iter()
                .fold(0u16, |m, &p| m | (1 << grid[p as usize]));
            candidates[i] = 0b11_1111_1110 & !seen;
        }
    }
    let mut derived = Vec::new();
    while derived.len() < depth {
        let Some(i) = (0..81).find(|&i| grid[i] == 0 && candidates[i].count_ones() == 1) else {
            break;
        };
        let d = candidates[i].trailing_zeros() as u8;
        grid[i] = d;
        candidates[i] = 0;
        for &p in &peers[i] {
            candidates[p as usize] &= !(1 << d);
        }
        derived.push(i as u8);
    }
    Snapshot {
        grid,
        candidates,
        derived,
    }
}

/// Every row, column and box holds 1..9 once.
fn is_solution(g: &Grid) -> bool {
    let full = 0b11_1111_1110u16;
    (0..9).all(|k| {
        let row = (0..9).fold(0u16, |m, c| m | 1 << g[k * 9 + c]);
        let col = (0..9).fold(0u16, |m, r| m | 1 << g[r * 9 + k]);
        let bx = (0..9).fold(0u16, |m, j| {
            m | 1 << g[(k / 3 * 3 + j / 3) * 9 + k % 3 * 3 + j % 3]
        });
        row == full && col == full && bx == full
    })
}

/// One Sudoku symmetry: digit relabelling, then rows and columns through
/// band/stack and within-band/stack permutations, then optional transpose.
#[derive(Clone, Copy)]
struct Symmetry {
    digit: [u8; 10],
    row: [usize; 9],
    col: [usize; 9],
    transpose: bool,
}

impl Symmetry {
    fn draw(rng: &mut Rng) -> Self {
        let mut digits = [1u8, 2, 3, 4, 5, 6, 7, 8, 9];
        rng.shuffle(&mut digits);
        let mut digit = [0u8; 10];
        for (i, d) in digits.iter().enumerate() {
            digit[i + 1] = *d;
        }
        let lines = |rng: &mut Rng| {
            let mut bands = [0usize, 1, 2];
            rng.shuffle(&mut bands);
            let mut out = [0usize; 9];
            for (b, &band) in bands.iter().enumerate() {
                let mut within = [0usize, 1, 2];
                rng.shuffle(&mut within);
                for (w, &line) in within.iter().enumerate() {
                    out[b * 3 + w] = band * 3 + line;
                }
            }
            out
        };
        let row = lines(rng);
        let col = lines(rng);
        Self {
            digit,
            row,
            col,
            transpose: rng.next() & 1 == 1,
        }
    }

    fn apply(&self, g: &Grid) -> Grid {
        let mut out = [0u8; 81];
        for r in 0..9 {
            for c in 0..9 {
                let (sr, sc) = if self.transpose { (c, r) } else { (r, c) };
                out[r * 9 + c] = self.digit[g[self.row[sr] * 9 + self.col[sc]] as usize];
            }
        }
        out
    }
}

/// About `min_edges` claim edges; each copy of the puzzle is one group.
fn build_lane(min_edges: usize, seed: u64) -> Lane {
    let peers = peers();
    let base = propagate(&PUZZLE, usize::MAX, &peers);
    assert!(
        is_solution(&base.grid),
        "the base puzzle must solve by naked singles"
    );
    let base_givens = PUZZLE.iter().filter(|&&d| d != 0).count();

    let mut rng = Rng(seed);
    let mut lane = Lane::default();
    while lane.edges.len() < min_edges {
        let sym = Symmetry::draw(&mut rng);
        let puzzle = sym.apply(&PUZZLE);
        let solution = sym.apply(&base.grid);
        // The copy solves to the transformed base solution, whatever order
        // propagation takes.
        let full = propagate(&puzzle, usize::MAX, &peers);
        assert_eq!(
            full.grid, solution,
            "copy {} solves differently",
            lane.groups
        );
        let depth = rng.below(81 - base_givens as u64 + 1) as usize;
        let snap = propagate(&puzzle, depth, &peers);

        for i in 0..81 {
            if puzzle[i] != 0 {
                lane.push(GIVEN);
                lane.expected[0] += 1;
            } else if snap.grid[i] != 0 {
                lane.push(FORCED);
                lane.expected[1] += 1;
            } else {
                // Naked singles never eliminate the true digit.
                assert_ne!(snap.candidates[i] & 1 << solution[i], 0);
                lane.push(ENTAILED);
                lane.expected[2] += 1;
                let others = snap.candidates[i] & !(1 << solution[i]);
                for _ in 0..others.count_ones() {
                    lane.push(CANDIDATE);
                    lane.expected[3] += 1;
                }
            }
        }
        assert_eq!(
            snap.derived.len(),
            (0..81)
                .filter(|&i| puzzle[i] == 0 && snap.grid[i] != 0)
                .count()
        );
        lane.close_group();
    }
    lane
}

fn main() {
    let decl = declarations(SUDOKU_CLASS);
    admit(&decl, SUDOKU_CLASS).expect("the Sudoku class declares the canonical reading");
    let t = Instant::now();
    let lane = build_lane(1_000_000, 0x5D0C_u64);
    println!("D-PUZZLE-0 / Sudoku at scale");
    println!(
        "  lane: {} edges from {} puzzle copies, built in {:.2?}",
        lane.edges.len(),
        lane.groups,
        t.elapsed()
    );
    let counts = check_three_ways(&lane, &decl, SUDOKU_CLASS);
    assert!(
        per_group(&lane, GIVEN.bit() | FORCED.bit() | ENTAILED.bit())
            .iter()
            .all(|&n| n == 81)
    );
    println!("  every copy: exactly 81 asserted claims (group fold)");
    report(&lane, &decl, SUDOKU_CLASS, &counts);
}

#[cfg(test)]
mod tests {
    use super::population_fold::{count_in, queries, UNKNOWN_CAUSES};
    use super::*;
    use lance_graph_contract::epistemic_state5::fact::{CAUSES, DIRECT, IND_KNOWN};
    use lance_graph_contract::epistemic_state5::facts_population;

    fn small_lane() -> Lane {
        build_lane(60_000, 0x7E57)
    }

    /// Every count three ways (filter, histogram, decode through the
    /// declaration), each equal to the solver's own counters, and every
    /// question actually exercised.
    #[test]
    fn every_question_counts_the_same_three_ways() {
        let lane = small_lane();
        let counts = check_three_ways(&lane, &declarations(SUDOKU_CLASS), SUDOKU_CLASS);
        assert!(counts.iter().all(|&n| n > 0), "a question went unexercised");
    }

    /// The four claim states partition the lane; as populations the three
    /// asserted states miss exactly `Unknown × Causes`, which this reading never
    /// emits, so the lane holds none of it.
    #[test]
    fn the_four_states_partition_the_lane() {
        let lane = small_lane();
        let qs = queries();
        let parts = &qs[1..];
        for (i, a) in parts.iter().enumerate() {
            for b in &parts[i + 1..] {
                assert_eq!(a.population & b.population, 0, "{} / {}", a.name, b.name);
            }
        }
        let union = parts.iter().fold(0, |u, q| u | q.population);
        assert_eq!(count_in(&lane.edges, union), lane.edges.len());
        let placed = parts[..3].iter().fold(0, |u, q| u | q.population);
        assert_eq!(qs[0].population & !placed, UNKNOWN_CAUSES.bit());
        assert_eq!(count_in(&lane.edges, UNKNOWN_CAUSES.bit()), 0);
    }

    /// Every copy keeps exactly 81 placed-digit claims and 30 givens, whatever
    /// its depth; the depths actually vary.
    #[test]
    fn the_group_fold_sees_every_copy_whole() {
        let lane = small_lane();
        assert!(per_group(&lane, facts_population(CAUSES))
            .iter()
            .all(|&n| n == 81));
        assert!(per_group(&lane, facts_population(DIRECT | CAUSES))
            .iter()
            .all(|&n| n == 30));
        let derived = per_group(&lane, facts_population(IND_KNOWN | CAUSES));
        let (lo, hi) = (derived.iter().min().unwrap(), derived.iter().max().unwrap());
        assert!(*lo < 5 && *hi > 45, "depths do not vary: {lo}..{hi}");
    }

    /// The histogram fold agrees with the filter fold for EVERY requirement
    /// over the declared facts, not only the five questions.
    #[test]
    fn the_histogram_answers_every_requirement() {
        use super::population_fold::{count_from, histogram};
        use lance_graph_contract::epistemic_state5::fact::{CERTIFICATION_MASK, TOPOLOGY_MASK};
        let lane = small_lane();
        let h = histogram(&lane.edges);
        let all = TOPOLOGY_MASK | CERTIFICATION_MASK;
        let mut sub = all;
        loop {
            let p = facts_population(sub);
            assert_eq!(count_from(&h, p), count_in(&lane.edges, p), "{sub:#x}");
            if sub == 0 {
                break;
            }
            sub = (sub - 1) & all;
        }
    }

    /// The gate is the declaration: an undeclared class is refused once,
    /// before any edge is read.
    #[test]
    fn an_undeclared_lane_is_refused_before_any_edge() {
        let decl = declarations(SUDOKU_CLASS);
        assert!(admit(&decl, SUDOKU_CLASS).is_some());
        assert!(admit(&decl, 0x0904).is_none());
    }

    /// The symmetries produce distinct puzzles, each a valid Sudoku.
    #[test]
    fn the_copies_are_distinct_valid_puzzles() {
        let peers = peers();
        let base = propagate(&PUZZLE, usize::MAX, &peers).grid;
        let mut rng = Rng(1);
        let mut seen = std::collections::HashSet::new();
        for _ in 0..200 {
            let sym = Symmetry::draw(&mut rng);
            let solution = sym.apply(&base);
            assert!(is_solution(&solution));
            seen.insert(sym.apply(&PUZZLE));
        }
        assert!(seen.len() > 190, "only {} distinct copies", seen.len());
    }
}
