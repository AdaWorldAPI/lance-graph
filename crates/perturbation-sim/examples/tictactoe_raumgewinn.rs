//! D-HXP-8, arm 1 — tic-tac-toe as the falsifier of "popcount Raumgewinn".
//!
//! Operator (2026-09-15): *"Popcount Raumgewinn — und Mississippi Queen im
//! Anschluss."* Pre-registration: `EPIPHANIES.md`
//! `E-POPCOUNTS-UPPER-RANGE-SIMILARITY-IS-THE-HEXAGONS-RAUMGEWINN-AND-BOARD-GAMES-MAKE-IT-FALSIFIABLE-1`,
//! `hexagon-plasticity-v1.md` §12. The claim under test: a position evaluated by
//! popcount STACKING over a cell's rails (its neighbours), ring by ring, through
//! the rolling-floor Belichtungsmesser (`TierFloors::stack_early_exit`), ranks
//! moves in agreement with the solved game — territory, not lines.
//!
//! # What is fixed before the first number
//!
//! - **Ground truth** — full negamax over every position reachable from the
//!   empty board; a move is *optimal* when it preserves the position's value.
//! - **Encoding** — each cell a unit; its rails are its neighbours at Chebyshev
//!   ring 1 and ring 2 (tic-tac-toe is the 8-direction lattice; on 3×3 rings 3
//!   and 4 are empty, so tiers 2 and 3 of the stack contribute zero — the API is
//!   the four-tier one so a 15×15 Gobang board runs unchanged).
//! - **The stack** — for a candidate move by P, tier r = popcount of P's stones
//!   among the ring-r rails (the pre-registered AGREEMENT arm); a second,
//!   exploratory arm uses the signed Zobrist influence own − opponent (NET).
//!   Floors are preheated on the population of every (position, empty cell)
//!   pair with `k = 2`, then `stack_early_exit` runs per candidate on a CLONE of
//!   the preheated floors, so the ranking inside a position is order-independent.
//!   Rank = descending, ties to the lowest index — but the RANKED QUANTITY is
//!   arm-dependent: AGREEMENT's per-tier terms are non-negative popcounts, so
//!   the early-exit *partial* stacked sum is a sound monotone bound and F1
//!   ranks on it directly; NET's per-tier terms are SIGNED (own − opp), so a
//!   partial sum is NOT a bound on the full stack (a later tier can be
//!   negative and pull the total below what an earlier positive partial
//!   suggested — see the "meter note" printed at the verdict site), and F1
//!   ranks NET candidates on the FULL stack instead. F2 (below) is unaffected
//!   by this split either way — it always compares the EARLY-EXIT top move
//!   against the FULL-STACK top move, regardless of which one F1 used.
//! - **F0 fixture validity, read FIRST** — the rails' horizon must be smaller
//!   than the board. If every cell's rings reach every other cell, the FULL stack
//!   of any ring-additive intensity (both arms sum a per-cell term over ring
//!   members) is the board census — identical for every candidate — and the
//!   meter can separate candidates only by early-exit PARTIAL sums. Such a
//!   fixture cannot read F1 or F2, and F1 landing exactly on the random-move
//!   baseline is the census signature, not a KILL. The gate is computed from the
//!   rails alone, before any position is scored; the degree-1 rails (reach 1 of
//!   8) are its can-it-stay-silent twin.
//! - **F1 correctness** — top-ranked move ∈ optimal set. Reported with the
//!   deterministic tie-break AND tie-aware (share of the tied top set that is
//!   optimal). Chance is MEASURED twice: the random-move baseline (mean share of
//!   optimal moves per position) and the shuffled-rail null (every cell's rails
//!   rewired to random cells of the same count, 20 seeds).
//! - **F2 economy** — the early-exited top move must equal the full-stack top
//!   move (an equality), and the mean exposed tiers is reported. This
//!   comparison is independent of which value F1 ranks by (see "The stack"
//!   above): it always pits the early-exit top against the full-stack top.
//! - **F3 degree ablation, mandatory** — every cell keeps ONE rail (its first
//!   ring-1 neighbour); F1 must DROP, or the task never exercised the six (E-Q8).
//!
//! Run: `cargo run --manifest-path crates/perturbation-sim/Cargo.toml --example tictactoe_raumgewinn --release`

use perturbation_sim::rolling_floor::TierFloors;
use std::collections::{HashMap, HashSet};

const N: usize = 9;
const K_SIGMA: f64 = 2.0;
const NULL_SEEDS: u64 = 20;
const F1_BAR: f64 = 0.95;

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
enum Cell {
    E,
    X,
    O,
}

type Board = [Cell; N];

const LINES: [[usize; 3]; 8] = [
    [0, 1, 2],
    [3, 4, 5],
    [6, 7, 8],
    [0, 3, 6],
    [1, 4, 7],
    [2, 5, 8],
    [0, 4, 8],
    [2, 4, 6],
];

fn other(c: Cell) -> Cell {
    match c {
        Cell::X => Cell::O,
        Cell::O => Cell::X,
        Cell::E => Cell::E,
    }
}

fn winner(b: &Board) -> Option<Cell> {
    LINES
        .iter()
        .find(|l| b[l[0]] != Cell::E && b[l[0]] == b[l[1]] && b[l[1]] == b[l[2]])
        .map(|l| b[l[0]])
}

fn terminal(b: &Board) -> bool {
    winner(b).is_some() || b.iter().all(|c| *c != Cell::E)
}

/// Negamax value for the side to move and the set of value-preserving moves.
fn solve(
    b: &Board,
    to_move: Cell,
    memo: &mut HashMap<(Board, Cell), (i8, Vec<usize>)>,
) -> (i8, Vec<usize>) {
    if let Some(v) = memo.get(&(*b, to_move)) {
        return v.clone();
    }
    let result = if winner(b).is_some() {
        // The previous mover completed a line: the side to move has lost.
        (-1, vec![])
    } else if b.iter().all(|c| *c != Cell::E) {
        (0, vec![])
    } else {
        let mut best = -2i8;
        let mut moves = Vec::new();
        for m in 0..N {
            if b[m] != Cell::E {
                continue;
            }
            let mut nb = *b;
            nb[m] = to_move;
            let v = -solve(&nb, other(to_move), memo).0;
            if v > best {
                best = v;
                moves = vec![m];
            } else if v == best {
                moves.push(m);
            }
        }
        (best, moves)
    };
    memo.insert((*b, to_move), result.clone());
    result
}

/// Every non-terminal position reachable from the empty board, X to move first.
fn reachable() -> Vec<(Board, Cell)> {
    let mut seen: HashSet<(Board, Cell)> = HashSet::new();
    let mut out = Vec::new();
    let mut stack = vec![([Cell::E; N], Cell::X)];
    while let Some((b, p)) = stack.pop() {
        if !seen.insert((b, p)) || terminal(&b) {
            continue;
        }
        out.push((b, p));
        for m in 0..N {
            if b[m] == Cell::E {
                let mut nb = b;
                nb[m] = p;
                stack.push((nb, other(p)));
            }
        }
    }
    out.sort_by_key(|(b, p)| (encode(b), *p == Cell::O));
    out
}

fn encode(b: &Board) -> u32 {
    b.iter().fold(0u32, |acc, c| {
        acc * 3
            + match c {
                Cell::E => 0,
                Cell::X => 1,
                Cell::O => 2,
            }
    })
}

/// The eight symmetries of the square, as index maps.
fn symmetries() -> Vec<[usize; N]> {
    let rot = |b: [usize; N]| -> [usize; N] {
        let mut o = [0; N];
        for (i, v) in b.iter().enumerate() {
            let (r, c) = (i / 3, i % 3);
            o[c * 3 + (2 - r)] = *v;
        }
        o
    };
    let flip = |b: [usize; N]| -> [usize; N] {
        let mut o = [0; N];
        for (i, v) in b.iter().enumerate() {
            let (r, c) = (i / 3, i % 3);
            o[r * 3 + (2 - c)] = *v;
        }
        o
    };
    let id: [usize; N] = std::array::from_fn(|i| i);
    let mut out = vec![id];
    let mut cur = id;
    for _ in 0..3 {
        cur = rot(cur);
        out.push(cur);
    }
    let f = flip(id);
    out.push(f);
    let mut cur = f;
    for _ in 0..3 {
        cur = rot(cur);
        out.push(cur);
    }
    out
}

fn canonical(b: &Board, syms: &[[usize; N]]) -> u32 {
    syms.iter()
        .map(|s| {
            let mut o = [Cell::E; N];
            for (i, &j) in s.iter().enumerate() {
                o[j] = b[i];
            }
            encode(&o)
        })
        .min()
        .expect("eight symmetries")
}

/// A cell's rails: neighbours by Chebyshev ring (index 0 = ring 1). Four rings
/// so the stack is four tiers; on 3×3 rings 3 and 4 are empty.
type Rails = Vec<[Vec<usize>; 4]>;

fn lattice_rails() -> Rails {
    (0..N)
        .map(|i| {
            let (r, c) = ((i / 3) as i32, (i % 3) as i32);
            let mut rings: [Vec<usize>; 4] = Default::default();
            for j in 0..N {
                if j == i {
                    continue;
                }
                let (rj, cj) = ((j / 3) as i32, (j % 3) as i32);
                let d = (r - rj).abs().max((c - cj).abs()) as usize;
                if (1..=4).contains(&d) {
                    rings[d - 1].push(j);
                }
            }
            rings
        })
        .collect()
}

/// F0 — the rails' horizon: how many of the other `N − 1` cells each cell's
/// rings reach in total. When every cell reaches every other cell the full stack
/// is the board census (see the module doc) and F1/F2 are not readable.
struct Horizon {
    min_reach: usize,
    max_reach: usize,
    exhausted_cells: usize,
}

impl Horizon {
    fn of(rails: &Rails) -> Self {
        let mut min_reach = usize::MAX;
        let mut max_reach = 0;
        let mut exhausted_cells = 0;
        for rings in rails {
            let reach: HashSet<usize> = rings.iter().flatten().copied().collect();
            min_reach = min_reach.min(reach.len());
            max_reach = max_reach.max(reach.len());
            if reach.len() == N - 1 {
                exhausted_cells += 1;
            }
        }
        Self {
            min_reach,
            max_reach,
            exhausted_cells,
        }
    }

    /// **Any** cell whose rings reach every other cell makes the ranked
    /// population contaminated — not only the all-cells case.
    ///
    /// The first version of this gate tested `exhausted_cells == N`, which is
    /// right for a uniform board like 3×3 and wrong for a board where only the
    /// CENTRE is covered by all rings (Hex 5×5 is the intended next fixture).
    /// There `exhausted_cells` is non-zero but less than `N`, the old
    /// predicate declared the fixture readable, and census-valued candidates —
    /// ones whose stacked score carries no ranking information at all — went
    /// straight into F1/F2 alongside real ones. Reported by codex on PR #1239.
    ///
    /// Strictly-any is deliberately conservative: it reports contamination
    /// rather than trying to weigh it. The probe already computes
    /// `f1_tie_nondegenerate` (F1 over the uncontaminated positions only), so
    /// a partially-exhausted board is not lost — it is read from that field
    /// instead of from `f1_tie`, and the verdict says so.
    fn exhausts_board(&self) -> bool {
        self.exhausted_cells > 0
    }
}

/// F3 — degree 1: every cell keeps only its first ring-1 rail.
fn degree_one(rails: &Rails) -> Rails {
    rails
        .iter()
        .map(|rings| {
            let mut one: [Vec<usize>; 4] = Default::default();
            if let Some(&first) = rings[0].first() {
                one[0].push(first);
            }
            one
        })
        .collect()
}

fn lcg(seed: &mut u64) -> u64 {
    *seed = seed
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    *seed >> 11
}

/// The shuffled-rail null: same rail COUNTS per cell and ring, targets drawn at
/// random from the other cells — the topology is scrambled, nothing else.
fn shuffled_rails(rails: &Rails, seed: u64) -> Rails {
    let mut s = seed;
    rails
        .iter()
        .enumerate()
        .map(|(i, rings)| {
            let mut out: [Vec<usize>; 4] = Default::default();
            let mut pool: Vec<usize> = (0..N).filter(|&j| j != i).collect();
            // Fisher–Yates over the pool, then deal it out ring by ring.
            for k in (1..pool.len()).rev() {
                let j = (lcg(&mut s) % (k as u64 + 1)) as usize;
                pool.swap(k, j);
            }
            let mut cursor = 0;
            for (r, ring) in rings.iter().enumerate() {
                out[r] = pool[cursor..cursor + ring.len()].to_vec();
                cursor += ring.len();
            }
            out
        })
        .collect()
}

#[derive(Clone, Copy)]
enum Arm {
    /// Popcount of P's stones among the ring's rails — the pre-registered arm.
    Agreement,
    /// Signed Zobrist influence: own − opponent per ring — exploratory.
    Net,
}

fn intensity(b: &Board, p: Cell, m: usize, rails: &Rails, arm: Arm) -> [f64; 4] {
    let mut out = [0.0; 4];
    for (r, ring) in rails[m].iter().enumerate() {
        let own = ring.iter().filter(|&&j| b[j] == p).count() as f64;
        let opp = ring.iter().filter(|&&j| b[j] == other(p)).count() as f64;
        out[r] = match arm {
            Arm::Agreement => own,
            Arm::Net => own - opp,
        };
    }
    out
}

/// Preheat one floor per tier on the cumulative stacked value through that tier,
/// over every (position, empty cell) pair — the population the meter will read.
fn preheat(positions: &[(Board, Cell)], rails: &Rails, arm: Arm) -> TierFloors {
    let mut samples: Vec<Vec<f64>> = vec![Vec::new(); 4];
    for (b, p) in positions {
        for m in 0..N {
            if b[m] != Cell::E {
                continue;
            }
            let inc = intensity(b, *p, m, rails, arm);
            let mut acc = 0.0;
            for (t, x) in inc.iter().enumerate() {
                acc += x;
                samples[t].push(acc);
            }
        }
    }
    let mut floors = TierFloors::new(K_SIGMA);
    floors.preheat(&samples);
    floors
}

struct Scored {
    f1_det: f64,
    f1_tie: f64,
    f2_equal: f64,
    exposed_tiers: f64,
    early_frac: f64,
    /// Anti-vacuity: positions where EVERY candidate carries the same stacked
    /// value — there the ranking is empty and tie-aware F1 is the random-move
    /// baseline by construction.
    all_tied_share: f64,
    /// Mean number of distinct stacked values among a position's candidates.
    distinct_mean: f64,
    /// Mean distinct FULL-stack values per position; 1.000 = the census (F0).
    full_distinct_mean: f64,
    /// Tie-aware F1 over the NON-degenerate positions only (None if there are none).
    f1_tie_nondegenerate: Option<f64>,
}

/// Rank every position's candidates by the stacked reading and score against
/// the solved optimal set.
fn score(
    positions: &[(Board, Cell)],
    optimal: &HashMap<(Board, Cell), Vec<usize>>,
    rails: &Rails,
    floors: &TierFloors,
    arm: Arm,
) -> Scored {
    let (mut hit_det, mut hit_tie, mut equal, mut tiers, mut early) = (0.0, 0.0, 0.0, 0.0, 0.0);
    let mut candidates_total = 0usize;
    let (mut all_tied, mut distinct_sum, mut nondeg_hit, mut nondeg_n) =
        (0usize, 0usize, 0.0, 0usize);
    let mut full_distinct_sum = 0usize;
    for (b, p) in positions {
        let opt = &optimal[&(*b, *p)];
        // (cell, stacked-with-early-exit, full stack, exit tier, early)
        let mut rows: Vec<(usize, f64, f64, usize, bool)> = Vec::new();
        for m in 0..N {
            if b[m] != Cell::E {
                continue;
            }
            let inc = intensity(b, *p, m, rails, arm);
            let mut mine = floors.clone();
            let res = mine.stack_early_exit(inc);
            rows.push((m, res.stacked, inc.iter().sum(), res.exit_tier, res.early));
        }
        candidates_total += rows.len();
        // Deterministic ranking: descending, then lowest index. `top_early`
        // always ranks by the early-exit stacked value (index 1); `top_full`
        // always ranks by the full stack (index 2) — F2 below compares these
        // two UNCONDITIONALLY, regardless of arm. F1's own ranking (`top`) is
        // arm-dependent: AGREEMENT's per-tier terms are non-negative, so the
        // early-exit partial is a sound bound and F1 ranks on it directly
        // (== `top_early`, unchanged from before this split existed); NET's
        // per-tier terms are signed, so a partial is NOT a bound on the full
        // stack (see the module doc's "The stack" + the "meter note" below),
        // and F1 must rank NET candidates on the full stack (== `top_full`).
        let top_early = rows
            .iter()
            .copied()
            .max_by(|a, b| a.1.partial_cmp(&b.1).expect("finite").then(b.0.cmp(&a.0)))
            .expect("non-terminal has a move");
        let top_full = rows
            .iter()
            .copied()
            .max_by(|a, b| a.2.partial_cmp(&b.2).expect("finite").then(b.0.cmp(&a.0)))
            .expect("non-terminal has a move");
        let top = match arm {
            Arm::Agreement => top_early,
            Arm::Net => top_full,
        };
        if opt.contains(&top.0) {
            hit_det += 1.0;
        }
        let tied: Vec<usize> = match arm {
            Arm::Agreement => rows.iter().filter(|r| r.1 == top.1).map(|r| r.0).collect(),
            Arm::Net => rows.iter().filter(|r| r.2 == top.2).map(|r| r.0).collect(),
        };
        let tied_opt = tied.iter().filter(|m| opt.contains(m)).count();
        let tie_score = tied_opt as f64 / tied.len() as f64;
        hit_tie += tie_score;
        let mut distinct: Vec<u64> = match arm {
            Arm::Agreement => rows.iter().map(|r| r.1.to_bits()).collect(),
            Arm::Net => rows.iter().map(|r| r.2.to_bits()).collect(),
        };
        distinct.sort_unstable();
        distinct.dedup();
        distinct_sum += distinct.len();
        let mut distinct_full: Vec<u64> = rows.iter().map(|r| r.2.to_bits()).collect();
        distinct_full.sort_unstable();
        distinct_full.dedup();
        full_distinct_sum += distinct_full.len();
        if tied.len() == rows.len() {
            all_tied += 1;
        } else {
            nondeg_hit += tie_score;
            nondeg_n += 1;
        }
        // F2's own question — does early exit change the top move — is
        // asked the SAME way regardless of which quantity F1 ranked by.
        if top_early.0 == top_full.0 {
            equal += 1.0;
        }
        for r in &rows {
            tiers += (r.3 + 1) as f64;
            if r.4 {
                early += 1.0;
            }
        }
    }
    let n = positions.len() as f64;
    Scored {
        f1_det: hit_det / n,
        f1_tie: hit_tie / n,
        f2_equal: equal / n,
        exposed_tiers: tiers / candidates_total as f64,
        early_frac: early / candidates_total as f64,
        all_tied_share: all_tied as f64 / n,
        distinct_mean: distinct_sum as f64 / n,
        full_distinct_mean: full_distinct_sum as f64 / n,
        f1_tie_nondegenerate: (nondeg_n > 0).then(|| nondeg_hit / nondeg_n as f64),
    }
}

fn main() {
    let positions = reachable();
    let mut memo = HashMap::new();
    let mut optimal: HashMap<(Board, Cell), Vec<usize>> = HashMap::new();
    for (b, p) in &positions {
        let (_, moves) = solve(b, *p, &mut memo);
        assert!(
            !moves.is_empty(),
            "a non-terminal position has an optimal move"
        );
        optimal.insert((*b, *p), moves);
    }
    let syms = symmetries();
    let classes: HashSet<(u32, Cell)> = positions
        .iter()
        .map(|(b, p)| (canonical(b, &syms), *p))
        .collect();
    let empty_value = solve(&[Cell::E; N], Cell::X, &mut memo).0;
    assert_eq!(empty_value, 0, "tic-tac-toe is a draw");

    // The random-move baseline: what an uninformed pick scores, per position.
    let random_move: f64 = positions
        .iter()
        .map(|(b, p)| {
            let empties = b.iter().filter(|c| **c == Cell::E).count() as f64;
            optimal[&(*b, *p)].len() as f64 / empties
        })
        .sum::<f64>()
        / positions.len() as f64;

    println!(
        "tic-tac-toe: {} reachable non-terminal positions, {} classes up to symmetry, value of the empty board {empty_value}",
        positions.len(),
        classes.len()
    );
    println!(
        "random-move baseline (mean share of value-preserving moves per position): {:.4}\n",
        random_move
    );

    let lattice = lattice_rails();
    let horizon = Horizon::of(&lattice);
    let horizon_d1 = Horizon::of(&degree_one(&lattice));
    // The gate's own two sides, on non-trivial rails: it must fire on the 3×3
    // lattice (rings 1 ∪ 2 = every other cell) and stay silent at degree 1.
    assert!(
        !horizon_d1.exhausts_board(),
        "F0 must stay silent on degree-1 rails (reach {} of {})",
        horizon_d1.max_reach,
        N - 1
    );
    // F0 has TWO independent ways to be unreadable, and the topology is only
    // the first. A board can be locally bounded — every cell's horizon smaller
    // than the board — and STILL give every candidate the same full-stack
    // value under the chosen intensity, in which case there is nothing to rank
    // and an F1 PASS/KILL would be meaningless. The second half is measured,
    // not derived, so it can only be applied after `score` runs; see
    // `score_is_degenerate` at the verdict site. Reported by codex on #1239.
    let topology_degenerate = horizon.exhausts_board();
    println!(
        "F0  fixture validity — ring coverage: each cell's rails reach {}..={} of {} other cells; horizon exhausts the board on {}/{N} cells (degree-1 rails reach {} — the gate stays silent there)",
        horizon.min_reach,
        horizon.max_reach,
        N - 1,
        horizon.exhausted_cells,
        horizon_d1.max_reach
    );
    if topology_degenerate {
        println!(
            "    => FIXTURE DEGENERATE (topology): {}/{N} cells reach every other cell, so their full stack is the board census and carries no ranking information. F1/F2/F3 below are DATA, not verdicts; the readable arm needs a board larger than the rails' horizon.\n",
            horizon.exhausted_cells
        );
    } else {
        println!("    => topology readable: every cell's horizon is smaller than the board (the SCORE is checked separately, per arm, below).\n");
    }
    for (arm, name) in [
        (Arm::Agreement, "AGREEMENT (pre-registered)"),
        (Arm::Net, "NET own-opp (exploratory)"),
    ] {
        println!("=== arm: {name}");
        let floors = preheat(&positions, &lattice, arm);
        for (t, f) in floors.floors.iter().enumerate() {
            println!(
                "      tier {t} floor: mu {:.3} sigma {:.3} threshold {:.3}",
                f.mu(),
                f.sigma(),
                f.threshold()
            );
        }
        let s = score(&positions, &optimal, &lattice, &floors, arm);
        println!(
            "      F1  top move optimal: {:.4} (deterministic tie-break)  {:.4} (tie-aware)   bar {F1_BAR}",
            s.f1_det, s.f1_tie
        );
        println!(
            "      F2  early-exit top == full-stack top: {:.4}   exposed tiers {:.3} of 4   early-exit share {:.4}",
            s.f2_equal, s.exposed_tiers, s.early_frac
        );
        println!(
            "      anti-vacuity: all candidates tied in {:.4} of positions; mean distinct stacked values {:.3} (FULL stack {:.3}; 1.000 = census); F1 tie-aware on non-degenerate positions {}",
            s.all_tied_share,
            s.distinct_mean,
            s.full_distinct_mean,
            s.f1_tie_nondegenerate
                .map(|v| format!("{v:.4}"))
                .unwrap_or_else(|| "n/a".into())
        );
        let d1 = degree_one(&lattice);
        let f1 = preheat(&positions, &d1, arm);
        let s1 = score(&positions, &optimal, &d1, &f1, arm);
        println!(
            "      F3  degree-1 ablation F1: {:.4} (det)  {:.4} (tie-aware)   drop {:+.4}",
            s1.f1_det,
            s1.f1_tie,
            s1.f1_tie - s.f1_tie
        );
        let mut nulls = Vec::new();
        for seed in 1..=NULL_SEEDS {
            let sr = shuffled_rails(&lattice, seed);
            let fr = preheat(&positions, &sr, arm);
            nulls.push(score(&positions, &optimal, &sr, &fr, arm).f1_tie);
        }
        let mean = nulls.iter().sum::<f64>() / nulls.len() as f64;
        let (lo, hi) = nulls
            .iter()
            .fold((f64::MAX, f64::MIN), |(l, h), &x| (l.min(x), h.max(x)));
        println!(
            "      null shuffled rails, {NULL_SEEDS} seeds, F1 tie-aware: mean {:.4}  range [{:.4}, {:.4}]",
            mean, lo, hi
        );
        // The measured half of F0: if the ranked quantity has one distinct
        // value per position, the ranking is empty whatever the topology says.
        // `full_distinct_mean` is the mean number of distinct FULL-stack values
        // per position; 1.000 means every candidate tied everywhere.
        let score_degenerate = s.full_distinct_mean <= 1.0 + 1e-9;
        let degenerate = topology_degenerate || score_degenerate;
        if score_degenerate && !topology_degenerate {
            println!(
                "      F0 (score): every candidate carries the SAME full-stack value (mean distinct {:.3}) although the topology is bounded — the ranking is empty and F1 below is not a verdict",
                s.full_distinct_mean
            );
        }
        let verdict = if degenerate {
            "F0 DEGENERATE — F1 not read (F1 at the baseline is the census signature, not a KILL)"
        } else if s.f1_tie >= F1_BAR {
            "F1 PASS"
        } else if s.f1_tie <= mean + (hi - lo) {
            "F1 KILL — at the null"
        } else {
            "F1 BETWEEN — above the null, below the bar"
        };
        let f3 = if degenerate {
            "F3 not read (a degree-1 F1 against a census F1 compares nothing)"
        } else if s1.f1_tie < s.f1_tie - 1e-9 {
            "F3 real (degree 1 drops)"
        } else {
            "F3 FLAT — the task did not exercise the neighbours"
        };
        let f2 = if degenerate {
            "not read (the full stack is constant per position; any spread is early-exit partial sums)"
        } else if s.f2_equal >= 1.0 - 1e-12 {
            "equality holds"
        } else {
            "verdict CHANGES under early exit"
        };
        println!("      verdict: {verdict}; {f3}; F2 {f2}");
        if s.f2_equal < 1.0 - 1e-12 {
            println!(
                "      meter note: early exit changed the top move in {:.4} of positions — `stack_early_exit` reports the PARTIAL sum at the exit tier, and with signed tiers (NET) a partial is not a bound on the full stack; the early-reject premise is monotone non-negative stacking",
                1.0 - s.f2_equal
            );
        }
        println!();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ---- negamax ---------------------------------------------------------

    #[test]
    fn negamax_forced_win_returns_exact_value_one() {
        // X to move, top row two-thirds full: cell 2 completes a line right
        // now. Whatever else is true of the position, having an immediately
        // winning move pins the negamax value at the domain's maximum, 1 —
        // no deeper search can ever beat "win this move."
        #[rustfmt::skip]
        let b: Board = [
            Cell::X, Cell::X, Cell::E,
            Cell::O, Cell::O, Cell::E,
            Cell::E, Cell::E, Cell::E,
        ];
        assert!(winner(&b).is_none(), "fixture must not already be decided");
        let mut memo = HashMap::new();
        let (value, moves) = solve(&b, Cell::X, &mut memo);
        assert_eq!(
            value, 1,
            "an immediate winning move must yield the maximum negamax value"
        );
        assert!(
            moves.contains(&2),
            "cell 2 completes the top row and must be among the optimal moves"
        );
    }

    #[test]
    fn negamax_forced_draw_returns_exact_value_zero() {
        // The empty board, X to move: the single best-established fact about
        // tic-tac-toe — perfect play from the start draws. `main()` leans on
        // this exact same fact (`assert_eq!(empty_value, 0, ...)`); pinning
        // it here as a dedicated unit test means a `solve`/`winner`/
        // `terminal` regression is caught by `cargo test`, not only by a
        // one-off assertion buried inside `main`.
        let mut memo = HashMap::new();
        let (value, moves) = solve(&[Cell::E; N], Cell::X, &mut memo);
        assert_eq!(
            value, 0,
            "tic-tac-toe from the empty board is a draw under perfect play"
        );
        assert!(
            !moves.is_empty(),
            "a non-terminal position must report at least one optimal move"
        );
    }

    // ---- reachable-position filtering -------------------------------------

    #[test]
    fn reachable_and_symmetry_classes_stay_linked() {
        // Deliberately NOT pinned against the classic external combinatorics
        // trivia for tic-tac-toe (the well-known ~5478/~765 figures) — those
        // numbers were not independently re-derived here and hard-coding a
        // misremembered one would be worse than no test at all. Instead this
        // pins a relationship between `reachable()` and the up-to-symmetry
        // class count that is TRUE BY CONSTRUCTION whenever the underlying
        // code is correct, so the two can never drift apart silently even
        // without trusting an external number.
        let positions = reachable();
        assert!(
            positions.contains(&([Cell::E; N], Cell::X)),
            "the empty board is the BFS seed and is itself non-terminal"
        );
        for (b, _) in &positions {
            assert!(
                !terminal(b),
                "reachable() must never report an already-decided board"
            );
        }
        let syms = symmetries();
        let classes: HashSet<(u32, Cell)> = positions
            .iter()
            .map(|(b, p)| (canonical(b, &syms), *p))
            .collect();
        // Canonicalization can only ever coarsen: `classes` is the image of
        // `positions` under `canonical`, so its cardinality cannot exceed the
        // domain's.
        assert!(
            classes.len() <= positions.len(),
            "canonicalization must not manufacture positions: {} classes over {} positions",
            classes.len(),
            positions.len()
        );
        // Every orbit under the eight square symmetries has size at most 8,
        // so no canonical class can be the image of more than 8 positions —
        // a bug that OVER-merges distinct positions (e.g. an `encode`
        // collision, or a broken `canonical` that isn't actually a per-
        // symmetry minimum) would drive `classes.len()` below this floor.
        assert!(
            classes.len() * 8 >= positions.len(),
            "an orbit of size <= 8 cannot make {} classes cover {} positions",
            classes.len(),
            positions.len()
        );
        // Anti-vacuity: the two bounds above both hold even if `classes.len()
        // == 1` (total collapse) or `classes.len() == positions.len()` (no
        // reduction at all) — neither of which is what symmetry reduction on
        // a real board should do. Assert the reduction is genuine and partial.
        assert!(
            classes.len() > 1 && classes.len() < positions.len(),
            "canonicalization must produce a real, non-trivial reduction: {} classes from {} positions",
            classes.len(),
            positions.len()
        );
    }

    // ---- horizon detection -------------------------------------------------

    #[test]
    fn horizon_detects_the_full_lattice_as_exhausted() {
        let lattice = lattice_rails();
        let h = Horizon::of(&lattice);
        assert!(h.exhausts_board());
        assert_eq!(
            h.exhausted_cells, N,
            "every cell's ring1+ring2 rails reach all 8 of the other cells on 3x3"
        );
        assert_eq!(h.max_reach, N - 1);
    }

    #[test]
    fn horizon_stays_silent_on_degree_one_rails() {
        let d1 = degree_one(&lattice_rails());
        let h = Horizon::of(&d1);
        assert!(!h.exhausts_board());
        assert_eq!(h.exhausted_cells, 0);
        assert_eq!(h.max_reach, 1);
    }

    #[test]
    fn horizon_detects_partial_exhaustion_from_a_single_cell() {
        // Built directly as a `Rails` value (not derived from a board or from
        // `lattice_rails`): cell 0's single ring reaches every one of the
        // other 8 cells (exhausted), while every other cell keeps exactly one
        // harmless ring-1 rail back to cell 0 (nowhere near exhausted). This
        // is exactly the case the `exhausts_board` doc comment's fix
        // addresses: the OLD predicate (`exhausted_cells == N`) would have
        // read this as fine (1 != 9), while the fixed, strictly-any predicate
        // (`exhausted_cells > 0`) must flag it.
        let mut rails: Rails = vec![Default::default(); N];
        rails[0][0] = (1..N).collect();
        for i in 1..N {
            rails[i][0] = vec![0];
        }
        let h = Horizon::of(&rails);
        assert_eq!(h.exhausted_cells, 1, "exactly cell 0 is exhausted");
        assert_eq!(h.max_reach, N - 1);
        assert_eq!(h.min_reach, 1);
        assert!(
            h.exhausts_board(),
            "ANY exhausted cell must contaminate the board, not only ALL cells"
        );
    }

    // ---- rail shuffling ------------------------------------------------------

    #[test]
    fn shuffled_rails_permutes_targets_and_preserves_per_ring_counts() {
        // A hand-built `Rails` with distinct, easily-checked ring sizes per
        // cell (3/3/2/0), independent of `lattice_rails`'s own correctness,
        // so the shuffle's ring-by-ring "deal it out" step has real structure
        // to preserve.
        let mut rails: Rails = vec![Default::default(); N];
        for i in 0..N {
            let others: Vec<usize> = (0..N).filter(|&j| j != i).collect();
            rails[i][0] = others[0..3].to_vec();
            rails[i][1] = others[3..6].to_vec();
            rails[i][2] = others[6..8].to_vec();
        }
        let shuffled = shuffled_rails(&rails, 7);
        for i in 0..N {
            for r in 0..4 {
                assert_eq!(
                    rails[i][r].len(),
                    shuffled[i][r].len(),
                    "cell {i} ring {r}'s rail COUNT must survive the shuffle"
                );
            }
            let mut before: Vec<usize> = rails[i].iter().flatten().copied().collect();
            let mut after: Vec<usize> = shuffled[i].iter().flatten().copied().collect();
            before.sort_unstable();
            after.sort_unstable();
            assert_eq!(
                before, after,
                "cell {i}'s shuffled targets must be a PERMUTATION of the originals, \
                 never an invention or a drop"
            );
            assert!(
                !after.contains(&i),
                "a cell must never rail to itself after shuffling"
            );
        }
        // Anti-vacuity: every assertion above would also pass for an
        // implementation that returns `rails` unchanged. Assert the shuffle
        // actually moved something.
        let moved = (0..N).any(|i| rails[i] != shuffled[i]);
        assert!(
            moved,
            "shuffled_rails(seed=7) must not be the identity permutation"
        );
    }

    // ---- signed-arm ranking --------------------------------------------------

    #[test]
    fn net_arm_ranks_by_the_full_stack_not_the_early_exit_partial() {
        // A hand-built board + rails where cell 0's tier-0 term alone (+3,
        // all-own) trips the Alarm on a virgin floor and stops the cascade
        // before its own negative tiers 1..3 ever get added, while cell 1
        // sails through all four tiers on net-zero terms and only shows its
        // value at the leaf. Every number below is hand-traced against
        // `intensity`'s own own-opp formula and `stack_early_exit`'s own
        // preheat/observe mechanics — nothing here is asserted "because the
        // compiler said so."
        #[rustfmt::skip]
        let b: Board = [
            Cell::E, Cell::E, Cell::X,
            Cell::X, Cell::O, Cell::O,
            Cell::X, Cell::O, Cell::X,
        ];
        let p = Cell::X;
        let mut rails: Rails = vec![Default::default(); N];
        // Candidate A (cell 0): tier 0 alone is +3 (own-dominant, all-X
        // ring), tiers 1..3 are each a single opposing stone (-1 apiece).
        rails[0][0] = vec![2, 6, 8];
        rails[0][1] = vec![4];
        rails[0][2] = vec![5];
        rails[0][3] = vec![7];
        // Candidate B (cell 1): every non-leaf tier nets to zero (one X, one
        // O each); only the leaf tier carries a net +1.
        rails[1][0] = vec![2, 4];
        rails[1][1] = vec![3, 5];
        rails[1][2] = vec![6, 7];
        rails[1][3] = vec![8];

        let floors = TierFloors::new(K_SIGMA);
        let inc_a = intensity(&b, p, 0, &rails, Arm::Net);
        let inc_b = intensity(&b, p, 1, &rails, Arm::Net);
        assert_eq!(inc_a, [3.0, -1.0, -1.0, -1.0]);
        assert_eq!(inc_b, [0.0, 0.0, 0.0, 1.0]);
        let res_a = floors.clone().stack_early_exit(inc_a);
        let res_b = floors.clone().stack_early_exit(inc_b);
        let full_a: f64 = inc_a.iter().sum();
        let full_b: f64 = inc_b.iter().sum();
        // The scenario's whole point: the two readings DISAGREE on which
        // candidate is best.
        assert!(
            res_a.stacked > res_b.stacked,
            "A's early-exit partial ({}) must beat B's ({}) — that is the OLD \
             ranking this test would otherwise vindicate",
            res_a.stacked,
            res_b.stacked
        );
        assert!(
            full_b > full_a,
            "B's full stack ({full_b}) must beat A's ({full_a}) — that is the \
             signal a signed arm's F1 ranking must use"
        );
        assert!(
            res_a.early,
            "A must exit early, before its own negative tiers land"
        );
        assert!(
            !res_b.early,
            "B must run to the leaf, where stacked == full"
        );

        let mut optimal = HashMap::new();
        optimal.insert((b, p), vec![1usize]);
        let positions = vec![(b, p)];
        let s = score(&positions, &optimal, &rails, &floors, Arm::Net);
        assert_eq!(
            s.f1_det, 1.0,
            "Arm::Net must rank by the full stack and pick cell 1, not the \
             early-exit-favoured cell 0 (reverting the fix flips this to 0.0)"
        );
    }
}
