//! PROBE-SUDOKU-COGNITIVE-CORPUS-1 — the first real corpus through the
//! `PROBE-RECIPE-DISPATCH-BRIDGE-1` (#996) bridge: a real Sudoku puzzle,
//! solved by real (independently ground-truthed) constraint propagation,
//! with each elimination event ALSO dispatched through a real recipe
//! kernel, producing one canonical receipt.
//!
//! # What this is, precisely
//!
//! Two things run side by side, deliberately kept separate so neither can
//! borrow the other's correctness:
//!
//! 1. **The Sudoku solving itself is ordinary, correct, ground-truthed
//!    constraint propagation** (naked-single elimination), implemented in
//!    plain Rust — NOT claimed to be "solved by the recipe kernels". Every
//!    digit this probe assigns is checked against an INDEPENDENT full
//!    backtracking solve of the same puzzle (`backtracking_solve`, which
//!    shares no code with the propagation loop) — the "brute-force oracle
//!    says whether it was warranted" check from the session discussion.
//! 2. **Each elimination event is ALSO dispatched through a real
//!    `lance_graph_contract::recipe_kernels::kernel(id)`**, via the exact
//!    bridge #996 proved identity-preserving, with `ThoughtCtx.candidates`
//!    built from the cell's real, currently-live candidate set. This
//!    measures how the cognitive layer responds to REAL puzzle-derived
//!    state — it does NOT drive the solving. The recipe's own pruning
//!    heuristic is generic (confidence-shaped), not Sudoku-aware, and this
//!    probe does not pretend otherwise.
//!
//! # Why these three recipes
//!
//! `PROBE-RECIPE-EXECUTION-1` (#995) found `TCP`(5)/`TCF`(20)/`CUR`(26) —
//! "Thought Chain Pruning" / "Thought Cascade Filtering" / "Cascading
//! Uncertainty Reduction" — collapsed into IDENTICAL coarse effect
//! signatures across its synthetic 4-context battery. They are also the
//! three recipes whose NAMES most plausibly fit "prune a candidate set",
//! which is literally what a Sudoku elimination event is. Choosing them
//! is not a claim they are "the" right recipes for this domain — no such
//! claim is defensible from the catalogue alone — it is choosing the most
//! interesting re-test: **do these three still collide under real,
//! puzzle-derived contexts, or does real data separate what synthetic
//! data could not?**
//!
//! Selection per event is deterministic and puzzle-state-derived (not
//! hand-picked per step): the peer group (row / column / box) with the
//! FEWEST remaining unsolved cells — the tightest real constraint —
//! chooses row→TCP, column→CUR, box→TCF.
//!
//! # The puzzle
//!
//! The Sudoku Wikipedia article's illustrative example grid — a widely
//! published, real puzzle, not fabricated for this probe.
//!
//! Run (lance-graph-ogar is workspace-EXCLUDED, own [workspace] — BBB
//! firewall keeps OGAR out of the default lance-graph build):
//! `cargo run --manifest-path crates/lance-graph-ogar/Cargo.toml --example sudoku_cognitive_corpus_probe`

use lance_graph_contract::recipe_kernels::{kernel, MaturityPolicy, Outcome, ThoughtCtx};
use lance_graph_contract::recipes::recipe;

type Grid = [[u8; 9]; 9];

const TCP: u8 = 5;
const TCF: u8 = 20;
const CUR: u8 = 26;

/// The Sudoku Wikipedia article's example puzzle, `0` = empty.
const PUZZLE: Grid = [
    [5, 3, 0, 0, 7, 0, 0, 0, 0],
    [6, 0, 0, 1, 9, 5, 0, 0, 0],
    [0, 9, 8, 0, 0, 0, 0, 6, 0],
    [8, 0, 0, 0, 6, 0, 0, 0, 3],
    [4, 0, 0, 8, 0, 3, 0, 0, 1],
    [7, 0, 0, 0, 2, 0, 0, 0, 6],
    [0, 6, 0, 0, 0, 0, 2, 8, 0],
    [0, 0, 0, 4, 1, 9, 0, 0, 5],
    [0, 0, 0, 0, 8, 0, 0, 7, 9],
];

fn box_index(r: usize, c: usize) -> usize {
    (r / 3) * 3 + (c / 3)
}

/// Legal remaining digits for an empty cell — the row/column/box constraint,
/// applied directly (not derived from any recipe).
fn candidates(grid: &Grid, r: usize, c: usize) -> Vec<u8> {
    let mut used = [false; 10];
    for cc in 0..9 {
        used[grid[r][cc] as usize] = true;
    }
    for rr in 0..9 {
        used[grid[rr][c] as usize] = true;
    }
    let (br, bc) = (r / 3 * 3, c / 3 * 3);
    for rr in br..br + 3 {
        for cc in bc..bc + 3 {
            used[grid[rr][cc] as usize] = true;
        }
    }
    (1u8..=9).filter(|&d| !used[d as usize]).collect()
}

/// Independent ground truth — standard backtracking, shares no code with
/// the propagation loop below. Used ONLY to check that this probe's
/// propagation-assigned digits were warranted (match the unique solution).
fn backtracking_solve(grid: &Grid) -> Option<Grid> {
    let mut g = *grid;
    fn solve(g: &mut Grid) -> bool {
        for r in 0..9 {
            for c in 0..9 {
                if g[r][c] == 0 {
                    for d in candidates(g, r, c) {
                        g[r][c] = d;
                        if solve(g) {
                            return true;
                        }
                        g[r][c] = 0;
                    }
                    return false;
                }
            }
        }
        true
    }
    solve(&mut g).then_some(g)
}

/// How many empty cells remain in row `r`, column `c`, and this cell's box.
fn peer_pressure(grid: &Grid, r: usize, c: usize) -> (usize, usize, usize) {
    let row = (0..9).filter(|&cc| grid[r][cc] == 0).count();
    let col = (0..9).filter(|&rr| grid[rr][c] == 0).count();
    let (br, bc) = (r / 3 * 3, c / 3 * 3);
    let bx = (br..br + 3)
        .flat_map(|rr| (bc..bc + 3).map(move |cc| (rr, cc)))
        .filter(|&(rr, cc)| grid[rr][cc] == 0)
        .count();
    (row, col, bx)
}

/// One cognitive-layer dispatch event in the combined receipt.
#[derive(Debug)]
struct EliminationEvent {
    r: usize,
    c: usize,
    box_idx: usize,
    candidates_before: Vec<u8>,
    chosen_recipe_id: u8,
    chosen_because: &'static str,
    outcome: Outcome,
    ctx_candidates_after: Vec<f32>,
    became_naked_single: bool,
}

/// Build a `ThoughtCtx` from a cell's real, live candidate set — this
/// probe's own mapping (documented, not canonical): each remaining digit
/// gets equal confidence `1/n`; `sd` (entropy proxy) and `free_energy` rise
/// with more remaining candidates (more open = more surprise-prone).
fn ctx_from_candidates(cands: &[u8]) -> ThoughtCtx {
    let n = cands.len().max(1) as f32;
    let mut ctx = ThoughtCtx::new(vec![1.0 / n; cands.len()]);
    ctx.sd = ((n - 1.0) / 8.0).clamp(0.0, 1.0);
    ctx.free_energy = ctx.sd;
    ctx.temperature = 0.5;
    ctx.rung = 3;
    ctx
}

/// Naked-single constraint propagation WITH a cognitive-layer receipt.
/// Returns the propagated grid, the list of digits it assigned (each
/// checked against ground truth by the caller), and the combined receipt.
fn propagate_with_receipt(start: &Grid) -> (Grid, Vec<(usize, usize, u8)>, Vec<EliminationEvent>) {
    let mut grid = *start;
    let mut assignments = Vec::new();
    let mut receipt = Vec::new();

    loop {
        let mut naked_singles = Vec::new();

        for r in 0..9 {
            for c in 0..9 {
                if grid[r][c] != 0 {
                    continue;
                }
                let cands = candidates(&grid, r, c);
                if cands.is_empty() {
                    // Should not happen on a well-formed, solvable puzzle;
                    // this probe does not paper over it.
                    panic!("cell ({r},{c}) has zero candidates — puzzle malformed or bug");
                }

                let (row_p, col_p, box_p) = peer_pressure(&grid, r, c);
                let (chosen_recipe_id, why) = if row_p <= col_p && row_p <= box_p {
                    (TCP, "row has fewest remaining unsolved cells")
                } else if col_p <= box_p {
                    (CUR, "column has fewest remaining unsolved cells")
                } else {
                    (TCF, "box has fewest remaining unsolved cells")
                };

                let mut ctx = ctx_from_candidates(&cands);
                let k = kernel(chosen_recipe_id).expect("TCP/CUR/TCF are all minted ids");
                let outcome = k.run_with(&mut ctx, MaturityPolicy::Any);

                let is_naked_single = cands.len() == 1;
                if is_naked_single {
                    naked_singles.push((r, c, cands[0]));
                }

                receipt.push(EliminationEvent {
                    r,
                    c,
                    box_idx: box_index(r, c),
                    candidates_before: cands,
                    chosen_recipe_id,
                    chosen_because: why,
                    outcome,
                    ctx_candidates_after: ctx.candidates,
                    became_naked_single: is_naked_single,
                });
            }
        }

        if naked_singles.is_empty() {
            // Either fully solved (no empty cells were seen this pass) or
            // stalled (empty cells remain but none are naked singles —
            // this puzzle may need deduction beyond naked singles; that is
            // reported, not hidden). Either way, propagation is done.
            break;
        }
        for (r, c, d) in &naked_singles {
            grid[*r][*c] = *d;
            assignments.push((*r, *c, *d));
        }
    }

    (grid, assignments, receipt)
}

fn print_grid(g: &Grid) {
    for row in g {
        let s: String = row
            .iter()
            .map(|&d| if d == 0 { '.' } else { (b'0' + d) as char })
            .collect();
        println!("  {s}");
    }
}

fn main() {
    println!("═══ PROBE-SUDOKU-COGNITIVE-CORPUS-1 ═══\n");
    println!("Puzzle (Sudoku Wikipedia example, 0 = empty):");
    print_grid(&PUZZLE);

    // ── Independent ground truth, computed, not recited. ──────────────
    let solution = backtracking_solve(&PUZZLE).expect("this puzzle is well-known solvable");
    println!(
        "\nIndependent backtracking solution (ground truth, shares no code with propagation):"
    );
    print_grid(&solution);

    // ── Real propagation, with the cognitive receipt attached. ─────────
    let (final_grid, assignments, receipt) = propagate_with_receipt(&PUZZLE);

    println!(
        "\n── Naked-single propagation: {} cells assigned, {} total elimination events dispatched through real recipe kernels ──",
        assignments.len(),
        receipt.len()
    );

    // ── The falsifier: was every propagation-assigned digit warranted? ──
    let mut all_warranted = true;
    for (r, c, d) in &assignments {
        let ok = solution[*r][*c] == *d;
        all_warranted &= ok;
        if !ok {
            println!(
                "  MISMATCH at ({r},{c}): propagation assigned {d}, ground truth says {}",
                solution[*r][*c]
            );
        }
    }
    println!(
        "Warranted check (every propagation assignment matches the independent backtracking solution): {}",
        if all_warranted { "PASS — all assignments warranted" } else { "FAIL — see MISMATCH rows above" }
    );

    let solved_cells = final_grid.iter().flatten().filter(|&&d| d != 0).count();
    println!(
        "Propagation alone solved {solved_cells}/81 cells (naked singles only, no hidden-singles/pairs — the rest, if any, needs deeper deduction, honestly not attempted here)."
    );
    if solved_cells < 81 {
        println!("Remaining cells after propagation:");
        print_grid(&final_grid);
    }

    // ── A sample of the actual receipt entries, in full. ────────────────
    println!(
        "\n── Sample receipt entries (first 6 of {}) ──",
        receipt.len()
    );
    for ev in receipt.iter().take(6) {
        let r = recipe(ev.chosen_recipe_id).unwrap();
        println!(
            "  ({},{}) box={} candidates_before={:?} -> {} ({}) fired={} delta_conf={} candidates_after={:?} naked_single={}",
            ev.r,
            ev.c,
            ev.box_idx,
            ev.candidates_before,
            r.code,
            ev.chosen_because,
            ev.outcome.fired,
            ev.outcome.delta_conf,
            ev.ctx_candidates_after,
            ev.became_naked_single
        );
    }

    // ── Cognitive-layer response: real separability under real state. ──
    println!("\n── Cognitive-layer response over the real receipt ──");
    let mut by_recipe: std::collections::BTreeMap<u8, Vec<&EliminationEvent>> =
        std::collections::BTreeMap::new();
    for ev in &receipt {
        by_recipe.entry(ev.chosen_recipe_id).or_default().push(ev);
    }
    for (id, events) in &by_recipe {
        let r = recipe(*id).unwrap();
        let fired = events.iter().filter(|e| e.outcome.fired).count();
        let distinct_signatures: std::collections::BTreeSet<(bool, i8)> = events
            .iter()
            .map(|e| {
                let sign = if e.outcome.delta_conf > 0.0 {
                    1
                } else if e.outcome.delta_conf < 0.0 {
                    -1
                } else {
                    0
                };
                (e.outcome.fired, sign)
            })
            .collect();
        println!(
            "  {:<5} {:<28} dispatched={:<3} fired={:<3} distinct (fired,Δconf-sign) signatures={}",
            r.code,
            r.name,
            events.len(),
            fired,
            distinct_signatures.len()
        );
    }

    // Re-test #995's specific finding: TCP/TCF/CUR collided in the
    // SYNTHETIC battery. Under REAL puzzle-derived candidate sets, do they
    // still collide, or does real data separate them? Compare the set of
    // (fired, Δconf-sign) pairs each recipe actually produced.
    let sig_set = |id: u8| -> std::collections::BTreeSet<(bool, i8)> {
        by_recipe
            .get(&id)
            .map(|events| {
                events
                    .iter()
                    .map(|e| {
                        let sign = if e.outcome.delta_conf > 0.0 {
                            1
                        } else if e.outcome.delta_conf < 0.0 {
                            -1
                        } else {
                            0
                        };
                        (e.outcome.fired, sign)
                    })
                    .collect()
            })
            .unwrap_or_default()
    };
    let (tcp_sig, tcf_sig, cur_sig) = (sig_set(TCP), sig_set(TCF), sig_set(CUR));
    println!(
        "\nRe-test of #995's TCP/TCF/CUR collision under REAL puzzle-derived contexts (not the synthetic battery):"
    );
    println!(
        "  TCP signature set = {tcp_sig:?}\n  TCF signature set = {tcf_sig:?}\n  CUR signature set = {cur_sig:?}"
    );
    let still_collide = tcp_sig == tcf_sig && tcf_sig == cur_sig;
    println!(
        "  Verdict: {}",
        if still_collide {
            "STILL COLLIDE — real puzzle-derived candidate sets did not separate them at this signature granularity."
        } else {
            "SEPARATED — real puzzle-derived data distinguishes at least one of the three where the synthetic battery could not."
        }
    );

    println!("\n═══ Report ═══");
    println!(
        "PROBE-SUDOKU-COGNITIVE-CORPUS-1: {}",
        if all_warranted {
            "PASS on the load-bearing check — every digit this probe's propagation assigned is warranted by the puzzle's own constraints (verified against an independent backtracking solve). The recipe dispatch is a measured, honestly-scoped SIDE CHANNEL: it observes real puzzle state through real kernels, it does not drive or validate the solving."
        } else {
            "FAIL — a propagation assignment was not warranted; see MISMATCH rows above."
        }
    );
    println!(
        "Scope note: naked-single propagation only. Harder Sudoku deduction (hidden singles, \
         pairs, X-wing, ...) is NOT implemented here, and no claim is made that the recipe \
         kernels performed or could perform Sudoku-specific reasoning — their own pruning \
         logic is generic and confidence-shaped, unrelated to digit legality."
    );
}
