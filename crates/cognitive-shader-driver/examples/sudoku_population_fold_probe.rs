//! D-PUZZLE-0, step 1: Sudoku at scale on the `EpistemicState5` population
//! algebra.
//!
//! One edge carries a coordinate (`raw5`, bits 59..63). One semantic question
//! is a population: a `u32` over all 32 codes
//! (`epistemic_state5::facts_population`). Many edges are answered by a fold
//! over the edge words — the class declaration is checked ONCE per lane, never
//! per edge. This probe measures that shape on about a million edges and
//! checks every count three ways.
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
//! # The reading (a probe-local declaration, not canon)
//!
//! Claim = "cell c holds digit d". One edge per claim not yet eliminated:
//!
//! | claim at the snapshot | `Topology2 × Certification3` | raw5 |
//! |---|---|---|
//! | a given | `Direct × Causes` | 20 |
//! | placed by a naked single | `IndirectKnown × Causes` | 21 |
//! | the true digit of a cell not yet placed | `IndirectUnknown × Causes` | 22 |
//! | any other surviving candidate | `Direct × Associated` | 4 |
//!
//! The third row is the case the Cartesian layout was built to keep: the value
//! is entailed (the puzzle has one solution, checked) but the chain that
//! derives it has not been produced yet. Eliminated claims are not edges: the
//! 5-bit state has no "refuted" coordinate, and inventing one would be a
//! semantic choice this probe does not make. The policy pins (which state each
//! row gets) belong to this probe's class only.
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

use std::hint::black_box;
use std::time::Instant;

use causal_edge::edge::CausalEdge64;
use causal_edge::layout::EPISTEMIC_SHIFT;
use lance_graph_contract::band_reading::EdgeProvenance;
use lance_graph_contract::class_view::ClassId;
use lance_graph_contract::epistemic_state5::fact::{
    ASSOCIATED, CAUSES, DIRECT, IND_KNOWN, IND_UNKNOWN, RELATED,
};
use lance_graph_contract::epistemic_state5::{
    facts_population, Certification3, Epi5Declarations, Epi5Gen, Epi5Reading, EpistemicState5,
    Population, Topology2,
};
use lance_graph_contract::rail_geometry::RailAxis;

/// The probe's own class (no production class declares a Sudoku reading).
const SUDOKU_CLASS: ClassId = 0x0906;
const RAIL: RailAxis = RailAxis::Taxonomy;

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

// ── the four claim states ────────────────────────────────────────────────

const fn state(t: Topology2, c: Certification3) -> EpistemicState5 {
    EpistemicState5::new(Epi5Gen::V1, t, c)
}
const GIVEN: EpistemicState5 = state(Topology2::Direct, Certification3::Causes);
const DERIVED: EpistemicState5 = state(Topology2::IndirectKnown, Certification3::Causes);
const ENTAILED: EpistemicState5 = state(Topology2::IndirectUnknown, Certification3::Causes);
const CANDIDATE: EpistemicState5 = state(Topology2::Direct, Certification3::Associated);

// ── the questions, each one population ───────────────────────────────────

/// A question: its population (the fast path) and the same question asked of
/// one decoded state (the slow path). Both are written from the facts, never
/// from code numbers.
struct Query {
    name: &'static str,
    population: Population,
    asks: fn(EpistemicState5) -> bool,
}

fn queries() -> [Query; 5] {
    [
        Query {
            name: "every placed digit (any topology, Causes)",
            population: facts_population(CAUSES),
            asks: |s| s.asserts(CAUSES),
        },
        Query {
            name: "givens (Direct x Causes)",
            population: facts_population(DIRECT | CAUSES),
            asks: |s| s.asserts(DIRECT | CAUSES),
        },
        Query {
            name: "derived, chain known (IndirectKnown x Causes)",
            population: facts_population(IND_KNOWN | CAUSES),
            asks: |s| s.asserts(IND_KNOWN | CAUSES),
        },
        Query {
            name: "entailed, chain not yet derived (IndirectUnknown x Causes)",
            population: facts_population(IND_UNKNOWN | CAUSES),
            asks: |s| s.asserts(IND_UNKNOWN | CAUSES),
        },
        Query {
            name: "live candidates (Direct, Associated, not Related)",
            population: facts_population(DIRECT | ASSOCIATED) & !facts_population(RELATED),
            asks: |s| s.asserts(DIRECT | ASSOCIATED) && !s.asserts(RELATED),
        },
    ]
}

// ── Sudoku: peers, naked singles, symmetries ─────────────────────────────

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

/// SplitMix64: deterministic, seedable.
struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    fn shuffle<T>(&mut self, xs: &mut [T]) {
        for i in (1..xs.len()).rev() {
            xs.swap(i, (self.next() % (i as u64 + 1)) as usize);
        }
    }
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

// ── the lane ─────────────────────────────────────────────────────────────

/// About a million claim edges, with the solver's own counts kept apart from
/// the edges so the folds have something independent to agree with.
struct Lane {
    edges: Vec<CausalEdge64>,
    puzzle_of: Vec<u32>,
    puzzles: u32,
    expected: [usize; 5],
}

fn build_lane(min_edges: usize, seed: u64) -> Lane {
    let peers = peers();
    let base = propagate(&PUZZLE, usize::MAX, &peers);
    assert!(
        is_solution(&base.grid),
        "the base puzzle must solve by naked singles"
    );
    let base_givens = PUZZLE.iter().filter(|&&d| d != 0).count();

    let mut rng = Rng(seed);
    let (mut edges, mut puzzle_of) = (Vec::new(), Vec::new());
    let mut expected = [0usize; 5];
    let mut puzzles = 0u32;
    while edges.len() < min_edges {
        let sym = Symmetry::draw(&mut rng);
        let puzzle = sym.apply(&PUZZLE);
        let solution = sym.apply(&base.grid);
        // The copy solves to the transformed base solution, whatever order
        // propagation takes.
        let full = propagate(&puzzle, usize::MAX, &peers);
        assert_eq!(full.grid, solution, "copy {puzzles} solves differently");
        let depth = (rng.next() % (81 - base_givens as u64 + 1)) as usize;
        let snap = propagate(&puzzle, depth, &peers);

        let mut push = |s: EpistemicState5| {
            edges.push(CausalEdge64::ZERO.with_epistemic_raw5(s.raw()));
            puzzle_of.push(puzzles);
        };
        for i in 0..81 {
            if puzzle[i] != 0 {
                push(GIVEN);
                expected[1] += 1;
            } else if snap.grid[i] != 0 {
                push(DERIVED);
                expected[2] += 1;
            } else {
                // Naked singles never eliminate the true digit.
                assert_ne!(snap.candidates[i] & 1 << solution[i], 0);
                push(ENTAILED);
                expected[3] += 1;
                let others = snap.candidates[i] & !(1 << solution[i]);
                for _ in 0..others.count_ones() {
                    push(CANDIDATE);
                    expected[4] += 1;
                }
            }
        }
        assert_eq!(snap.derived.len(), expected_derived_in(&puzzle, &snap.grid));
        expected[0] += 81;
        puzzles += 1;
    }
    Lane {
        edges,
        puzzle_of,
        puzzles,
        expected,
    }
}

fn expected_derived_in(puzzle: &Grid, grid: &Grid) -> usize {
    (0..81).filter(|&i| puzzle[i] == 0 && grid[i] != 0).count()
}

/// The declaration gate, checked once per lane: a lane whose class does not
/// declare the canonical reading is refused before any edge is read.
fn admit(decl: &Epi5Declarations, class: ClassId) -> Option<Epi5Reading> {
    decl.get(class, RAIL)
}

fn declarations() -> Epi5Declarations {
    let mut d = Epi5Declarations::new();
    d.declare(SUDOKU_CLASS, RAIL, Epi5Reading::default());
    d
}

// ── the folds ────────────────────────────────────────────────────────────

fn raw5(e: CausalEdge64) -> u32 {
    (e.0 >> EPISTEMIC_SHIFT) as u32
}

/// Fast path, one question: a shift and a mask per edge.
fn count_in(edges: &[CausalEdge64], population: Population) -> usize {
    edges
        .iter()
        .filter(|&&e| population >> raw5(e) & 1 == 1)
        .count()
}

/// Fast path, every question at once: one pass builds a 32-bin histogram; a
/// population's count is then the sum of its bins.
fn histogram(edges: &[CausalEdge64]) -> [usize; 32] {
    let mut h = [0usize; 32];
    for &e in edges {
        h[raw5(e) as usize] += 1;
    }
    h
}

fn count_from(h: &[usize; 32], population: Population) -> usize {
    (0..32)
        .filter(|&r| population >> r & 1 == 1)
        .map(|r| h[r])
        .sum()
}

/// Slow path, the oracle: decode every edge through the declaration, then
/// ask the decoded state.
fn count_decoded(
    edges: &[CausalEdge64],
    decl: &Epi5Declarations,
    asks: fn(EpistemicState5) -> bool,
) -> usize {
    edges
        .iter()
        .filter(|e| {
            let s = decl
                .project_state5(
                    SUDOKU_CLASS,
                    RAIL,
                    Epi5Gen::V1,
                    e.epistemic_raw5(),
                    EdgeProvenance::V2Stamped,
                )
                .expect("every edge in a declared lane decodes");
            asks(s)
        })
        .count()
}

/// Group fold: one question per puzzle.
fn per_puzzle(lane: &Lane, population: Population) -> Vec<u32> {
    let mut out = vec![0u32; lane.puzzles as usize];
    for (&e, &p) in lane.edges.iter().zip(&lane.puzzle_of) {
        out[p as usize] += population >> raw5(e) & 1;
    }
    out
}

fn median_ns_per_edge(edges: usize, mut f: impl FnMut() -> usize) -> f64 {
    let mut times: Vec<f64> = (0..7)
        .map(|_| {
            let t = Instant::now();
            black_box(f());
            t.elapsed().as_nanos() as f64 / edges as f64
        })
        .collect();
    times.sort_by(f64::total_cmp);
    times[times.len() / 2]
}

fn main() {
    let decl = declarations();
    admit(&decl, SUDOKU_CLASS).expect("the Sudoku class declares the canonical reading");
    let t = Instant::now();
    let lane = build_lane(1_000_000, 0x5D0C_u64);
    let built = t.elapsed();
    println!("D-PUZZLE-0 / Sudoku at scale");
    println!(
        "  lane: {} edges from {} puzzle copies, built in {:.2?}",
        lane.edges.len(),
        lane.puzzles,
        built
    );
    println!(
        "  build: debug_assertions={} avx2={} avx512f={}",
        cfg!(debug_assertions),
        cfg!(target_feature = "avx2"),
        cfg!(target_feature = "avx512f")
    );

    let h = histogram(&lane.edges);
    for (q, want) in queries().iter().zip(lane.expected) {
        let fast = count_in(&lane.edges, q.population);
        let hist = count_from(&h, q.population);
        let slow = count_decoded(&lane.edges, &decl, q.asks);
        assert!(fast == want && hist == want && slow == want, "{}", q.name);
        println!(
            "  {:<60} pop {:#010x}  {:>8} edges",
            q.name, q.population, want
        );
    }
    assert!(per_puzzle(&lane, facts_population(CAUSES))
        .iter()
        .all(|&n| n == 81));
    println!("  every copy: exactly 81 placed-digit claims (group fold)");

    let n = lane.edges.len();
    let pop = facts_population(IND_UNKNOWN | CAUSES);
    let filter = median_ns_per_edge(n, || count_in(black_box(&lane.edges), pop));
    let hist = median_ns_per_edge(n, || histogram(black_box(&lane.edges))[22]);
    let decoded = median_ns_per_edge(n, || {
        count_decoded(black_box(&lane.edges), &decl, |s| {
            s.asserts(IND_UNKNOWN | CAUSES)
        })
    });
    println!("  median of 7, ns per edge:");
    println!("    one population, filter fold      {filter:.3}");
    println!("    32-bin histogram (all questions)  {hist:.3}");
    println!("    per-edge decode + asserts (oracle) {decoded:.3}");
}

#[cfg(test)]
mod tests {
    use super::*;

    fn small_lane() -> Lane {
        build_lane(60_000, 0x7E57)
    }

    /// The four claim states are the coordinates the table in the module doc
    /// names, and each query's population is exactly the expected code set.
    #[test]
    fn the_reading_and_the_populations_are_the_stated_codes() {
        assert_eq!(
            [GIVEN.raw(), DERIVED.raw(), ENTAILED.raw(), CANDIDATE.raw()],
            [20, 21, 22, 4]
        );
        let pops: Vec<Population> = queries().iter().map(|q| q.population).collect();
        assert_eq!(pops, [0xF0_0000, 1 << 20, 1 << 21, 1 << 22, 1 << 4]);
    }

    /// Every count three ways: the population filter, the histogram, and the
    /// per-edge decode through the declaration — all equal to the solver's own
    /// independent counters.
    #[test]
    fn every_question_counts_the_same_three_ways() {
        let (lane, decl) = (small_lane(), declarations());
        let h = histogram(&lane.edges);
        for (q, want) in queries().iter().zip(lane.expected) {
            assert!(
                want > 0,
                "{}: the corpus never exercised this question",
                q.name
            );
            assert_eq!(count_in(&lane.edges, q.population), want, "{}", q.name);
            assert_eq!(count_from(&h, q.population), want, "{}", q.name);
            assert_eq!(
                count_decoded(&lane.edges, &decl, q.asks),
                want,
                "{}",
                q.name
            );
        }
    }

    /// The four claim states partition the lane: their populations are
    /// disjoint and together cover every edge.
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
        // Every placed digit is a given, a derivation or an entailment. As
        // POPULATIONS the three miss exactly `Unknown × Causes` (23), which is
        // meaningful in general but which this reading never emits, so the
        // lane holds none of it and the counts agree.
        let placed = parts[..3].iter().fold(0, |u, q| u | q.population);
        let unknown_causes = state(Topology2::Unknown, Certification3::Causes).bit();
        assert_eq!(qs[0].population & !placed, unknown_causes);
        assert_eq!(count_in(&lane.edges, unknown_causes), 0);
        assert_eq!(
            count_in(&lane.edges, placed),
            count_in(&lane.edges, qs[0].population)
        );
    }

    /// Every copy keeps exactly 81 placed-digit claims and 30 givens, whatever
    /// its depth; the depths actually vary.
    #[test]
    fn the_group_fold_sees_every_copy_whole() {
        let lane = small_lane();
        assert!(per_puzzle(&lane, facts_population(CAUSES))
            .iter()
            .all(|&n| n == 81));
        assert!(per_puzzle(&lane, facts_population(DIRECT | CAUSES))
            .iter()
            .all(|&n| n == 30));
        let derived = per_puzzle(&lane, facts_population(IND_KNOWN | CAUSES));
        let (lo, hi) = (derived.iter().min().unwrap(), derived.iter().max().unwrap());
        assert!(*lo < 5 && *hi > 45, "depths do not vary: {lo}..{hi}");
    }

    /// The histogram fold agrees with the filter fold for EVERY requirement
    /// over the declared facts, not only the five questions.
    #[test]
    fn the_histogram_answers_every_requirement() {
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
        let decl = declarations();
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
