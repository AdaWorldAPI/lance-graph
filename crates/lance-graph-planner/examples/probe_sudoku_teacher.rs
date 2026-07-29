//! `probe_sudoku_teacher` — PROBE-SUDOKU-TEACHER: literal Sudoku as the first
//! teacher for §§4a-4c of
//! `.claude/plans/epistemic-quadrant-materialization-v1.md` (§4d).
//!
//! An 81-cell grid with a known solution is a free, never-wrong oracle. Every
//! mechanism claim in the plan becomes a checkable assertion against it:
//!
//! * **G1 — horizon.** The witness lane (`CausalWitnessFacet`) carries ONLY
//!   backward box-peer displacements (verified arithmetic: box peers are all
//!   within `±8`, cross-band column peers are always out of window, cross-stack
//!   row peers are mixed — so both ride the sweep). A box-forced hidden single
//!   must be findable from the lane alone (sweep stays silent); a
//!   column-forced one must NOT be (the sweep must fire and change the
//!   answer).
//! * **G2 — hidden single.** A puzzle seeded with a hidden-single-that-is-not-
//!   naked must find it; an all-naked puzzle must report zero.
//! * **G3 — fork-return.** Bifurcation clones the slab as a counterfactual
//!   world, propagates to contradiction, and ONLY the elimination returns —
//!   the main slab changes at exactly the sanctioned cell, and the fork's
//!   positive (wrong) guess never appears in it.
//! * **G4 — quadrant census.** `Quadrant::classify` (Staunen/Confusion/
//!   Boredom/Wisdom) migrates toward Wisdom across passes for a solvable
//!   puzzle; a fork-refusing policy on a bifurcation-required puzzle does
//!   NOT fully migrate.
//! * **G5 — triangle motion.** Two style-family policies (A = elections-first,
//!   B = bifurcate-early), graded by (solved, cost, path-Levenshtein vs a
//!   teacher path), drive `ExploreStyle → LearnedStyle → FrozenStyle` for the
//!   first time: a promote case (train and held-out agree) AND a refuse case
//!   (train favors B, held-out favors A) — write-isolation asserted on the
//!   triangle lanes both times.
//! * **G6 — Hamming monotone.** Grid-vs-solution Hamming distance never
//!   increases across passes.
//!
//! Content placement (digit + given/derived flag) is an EXPERIMENTAL reading
//! of the existing `EntityType` `u16` lane — no new tenant, no layout change,
//! every offset derived via `ValueTenant::value_offset()` (the Tekamolo
//! honest-catalogue idiom: label the reading, touch no bytes it doesn't own).
//! Candidate sets are local pure compute, never stored in a row.
//!
//! Puzzle construction is deterministic (base pattern `(i*3 + i/3 + j) % 9 +
//! 1`, fixed digit-permutation tables, fixed blanking index lists — no RNG,
//! D-QUANTGATE replay).
//!
//! Usage: `cargo run -p lance-graph-planner --example probe_sudoku_teacher`

// Box-major position arithmetic (`box_pos`/`row_col_of`/`cell_in_box`) is
// threaded through most loops below as an index used for MULTIPLE purposes
// at once (peer-position comparison, row/col derivation, AND the grid read)
// — not the single-purpose element access `needless_range_loop` targets.
#![allow(clippy::needless_range_loop)]

use lance_graph_contract::canonical_node::{EdgeBlock, NodeGuid, NodeRow, ValueTenant};
use lance_graph_contract::causal_witness::CausalWitnessFacet;
use lance_graph_contract::witness_fabric::WitnessLens;
use ndarray::hpc::entropy_ladder::Quadrant;

// ─────────────────────────────────────────────────────────────────────────
// Box-major addressing (verified arithmetic — §4d of the plan)
// ─────────────────────────────────────────────────────────────────────────

/// `pos = box*9 + cell_in_box`, `box = (r/3)*3 + c/3`, `cell_in_box =
/// (r%3)*3 + c%3`.
fn box_pos(r: usize, c: usize) -> usize {
    let b = (r / 3) * 3 + c / 3;
    let k = (r % 3) * 3 + c % 3;
    b * 9 + k
}

/// Inverse of [`box_pos`].
fn row_col_of(pos: usize) -> (usize, usize) {
    let b = pos / 9;
    let k = pos % 9;
    let br = b / 3;
    let bc = b % 3;
    let r = br * 3 + k / 3;
    let c = bc * 3 + k % 3;
    (r, c)
}

/// `cell_in_box` index (`0..9`) of a box-major position — the number of
/// backward box-peer predecessors this cell has.
fn cell_in_box(pos: usize) -> usize {
    pos % 9
}

fn box_of(pos: usize) -> usize {
    pos / 9
}

#[test]
fn box_pos_round_trips() {
    for r in 0..9 {
        for c in 0..9 {
            let p = box_pos(r, c);
            assert_eq!(row_col_of(p), (r, c));
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────
// EntityType lane — EXPERIMENTAL reading: digit(1..9) + given/derived flag
// ─────────────────────────────────────────────────────────────────────────
//
// Honest-catalogue idiom: this is a READING of the existing `EntityType`
// `u16` lane (a cell-class discriminator by design), not a new tenant. Byte
// layout (low byte of the u16, high byte reserved 0): bits 0..3 = digit
// (0 = empty), bit 4 = given(1)/derived(0).

fn entity_offset() -> usize {
    ValueTenant::EntityType.value_offset()
}

/// `Some((digit, is_given))`, or `None` if the cell is empty.
fn read_cell(row: &NodeRow) -> Option<(u8, bool)> {
    let b = row.value[entity_offset()];
    if b == 0 {
        None
    } else {
        Some((b & 0x0F, b & 0x10 != 0))
    }
}

fn write_cell(row: &mut NodeRow, digit: u8, given: bool) {
    let o = entity_offset();
    let mut v = digit & 0x0F;
    if given {
        v |= 0x10;
    }
    row.value[o] = v;
    row.value[o + 1] = 0;
}

fn clear_cell(row: &mut NodeRow) {
    let o = entity_offset();
    row.value[o] = 0;
    row.value[o + 1] = 0;
}

fn blank_row() -> NodeRow {
    NodeRow {
        key: NodeGuid::local(0),
        edges: EdgeBlock::default(),
        value: [0u8; 480],
    }
}

// ─────────────────────────────────────────────────────────────────────────
// Witness lane — backward box-peer displacements ONLY, in the 8 reserved
// slots (16..24) of the CausalWitnessFacet register (the 16 named loci are
// untouched — this experimental use claims none of them).
// ─────────────────────────────────────────────────────────────────────────

/// Write every row's box-peer witness: cell `k` (0-indexed within its box)
/// gets exactly `k` backward slots (16..16+k), one per predecessor
/// `j < k` in the same box, holding the signed displacement `j - k` (always
/// in `-1..=-8`, so it fits the i4 range with room to spare).
fn write_box_witness(grid: &mut [NodeRow; 81]) {
    for (pos, row) in grid.iter_mut().enumerate() {
        let k = cell_in_box(pos);
        let mut facet = CausalWitnessFacet::ZERO;
        for j in 0..k {
            let slot = 16 + j;
            let off = j as isize - k as isize;
            facet.set(slot, off as i8);
        }
        WitnessLens::write_register(row, &facet);
    }
}

/// Candidates for `pos` computed from the witness LANE alone (box peers
/// only, backward displacements read through `WitnessLens`) — zero-copy,
/// never gathers. Empty if `pos` is already filled.
fn candidates_from_box_lane(grid: &[NodeRow; 81], pos: usize) -> Vec<u8> {
    if read_cell(&grid[pos]).is_some() {
        return Vec::new();
    }
    let lens = WitnessLens::new(grid);
    let Some(&facet) = lens.at(pos) else {
        return (1..=9).collect();
    };
    let mut candidates: Vec<u8> = (1..=9).collect();
    for slot in 16..24 {
        let off = facet.get(slot);
        if off == 0 {
            continue;
        }
        let peer_signed = pos as isize + off as isize;
        let Ok(peer_pos) = usize::try_from(peer_signed) else {
            continue;
        };
        if let Some((d, _)) = grid.get(peer_pos).and_then(read_cell) {
            candidates.retain(|&x| x != d);
        }
    }
    candidates
}

/// Candidates for `pos` computed from a FULL predicate sweep (box + row +
/// column peers) — the sweep the horizon claim says must fire when the lane
/// alone cannot resolve a cell. Zero-copy: a predicate over positions, never
/// a gathered `Vec` of rows.
fn candidates_from_full_sweep(grid: &[NodeRow; 81], pos: usize) -> Vec<u8> {
    if read_cell(&grid[pos]).is_some() {
        return Vec::new();
    }
    let (r, c) = row_col_of(pos);
    let b = box_of(pos);
    let mut candidates: Vec<u8> = (1..=9).collect();
    for p in 0..81 {
        if p == pos {
            continue;
        }
        let (pr, pc) = row_col_of(p);
        if box_of(p) == b || pr == r || pc == c {
            if let Some((d, _)) = read_cell(&grid[p]) {
                candidates.retain(|&x| x != d);
            }
        }
    }
    candidates
}

// ─────────────────────────────────────────────────────────────────────────
// Hidden singles (unit-wise: row / column / box)
// ─────────────────────────────────────────────────────────────────────────

/// `(pos, digit)` pairs where `digit` can go in exactly one empty cell of
/// some unit (row/col/box), AND that cell's own full-sweep candidate set has
/// more than one member (i.e. it is hidden, not merely naked).
fn hidden_singles(grid: &[NodeRow; 81]) -> Vec<(usize, u8)> {
    // The grid is immutable for the whole function — compute every cell's
    // full-sweep candidate set exactly ONCE instead of recomputing it up to
    // 27 times (3 unit kinds × 9 digits, plus once more for the len() > 1
    // check) per cell.
    let cands: Vec<Vec<u8>> = (0..81)
        .map(|p| candidates_from_full_sweep(grid, p))
        .collect();
    let mut result = Vec::new();
    for unit_kind in 0..3u8 {
        for u in 0..9usize {
            let cells: Vec<usize> = (0..81)
                .filter(|&p| {
                    let (r, c) = row_col_of(p);
                    match unit_kind {
                        0 => r == u,
                        1 => c == u,
                        _ => box_of(p) == u,
                    }
                })
                .collect();
            for d in 1..=9u8 {
                let mut holder = None;
                let mut count = 0u8;
                for &p in &cells {
                    if read_cell(&grid[p]).is_some() {
                        continue;
                    }
                    if cands[p].contains(&d) {
                        count += 1;
                        holder = Some(p);
                    }
                }
                if count == 1 {
                    let p = holder.unwrap();
                    if cands[p].len() > 1 {
                        result.push((p, d));
                    }
                }
            }
        }
    }
    result.sort_unstable();
    result.dedup();
    result
}

// ─────────────────────────────────────────────────────────────────────────
// Deterministic puzzle construction — no RNG (D-QUANTGATE replay)
// ─────────────────────────────────────────────────────────────────────────

/// The base valid solved grid, `(i*3 + i/3 + j) % 9 + 1`, stored box-major.
fn base_solution_boxmajor() -> [u8; 81] {
    let mut sol = [0u8; 81];
    for pos in 0..81 {
        let (r, c) = row_col_of(pos);
        sol[pos] = ((r * 3 + r / 3 + c) % 9 + 1) as u8;
    }
    sol
}

/// Fixed digit-permutation tables (any bijection on 1..9 is a symmetry of a
/// valid Sudoku solution) — deterministic puzzle variants, never RNG.
const PERM_A: [u8; 9] = [1, 2, 3, 4, 5, 6, 7, 8, 9];
const PERM_B: [u8; 9] = [4, 5, 6, 7, 8, 9, 1, 2, 3];
const PERM_C: [u8; 9] = [7, 8, 9, 1, 2, 3, 4, 5, 6];

fn permute_solution(sol: &[u8; 81], perm: &[u8; 9]) -> [u8; 81] {
    let mut out = [0u8; 81];
    for i in 0..81 {
        out[i] = perm[(sol[i] - 1) as usize];
    }
    out
}

/// Build a fully-given grid (every cell GIVEN) from a solution array.
fn grid_from_solution(sol: &[u8; 81]) -> [NodeRow; 81] {
    let mut grid: [NodeRow; 81] = [blank_row(); 81];
    for pos in 0..81 {
        write_cell(&mut grid[pos], sol[pos], true);
    }
    write_box_witness(&mut grid);
    grid
}

/// Blank a fixed, deterministic list of positions (given → empty).
fn blank_positions(grid: &mut [NodeRow; 81], positions: &[usize]) {
    for &p in positions {
        clear_cell(&mut grid[p]);
    }
}

/// Build the "ambiguous pair" fixture: the `ambiguous` box is left with a
/// genuine 2-way tie `{perm[7], perm[8]}` at the TARGET cell (k=7) and a
/// PARKED companion cell (k=3, the box's MIDDLE row — a different row band
/// from the shared row — deliberately, so it never becomes a peer of the
/// witness Z and can't silently collide with it), with the other 7 cells
/// given `perm[0..7]`.
/// The `witness` box — in the SAME ROW BAND as the target, so they end up
/// peers via a shared row — is independently box-forced to `perm[7]` alone
/// (its 8 predecessors use every digit except `perm[7]`). Guessing `perm[7]`
/// at the target then removes Z's only candidate via the shared row: a
/// genuine propagated contradiction, not a static given.
///
/// **Why the companion cell is parked off the shared row.** A 3×3 box with
/// 7 givens always leaves TWO cells holding the 2 leftover digits — there
/// is no way to leave exactly one. If BOTH leftover cells sat on the shared
/// row, Z's resolution would strip the same digit from both simultaneously,
/// and whichever elects first would leave the other with an unsatisfiable
/// (empty) candidate set — a self-inflicted contradiction that has nothing
/// to do with the fork-return mechanism this fixture exists to exercise.
/// Parking the companion on a different row keeps it independently solvable
/// (it resolves once the TARGET is filled, via ordinary box exclusion) and
/// keeps the target/Z pair the only real ambiguity.
///
/// Returns `(grid, z_pos, target_pos)`.
fn build_ambiguity_fixture(
    ambiguous_origin: (usize, usize),
    witness_origin: (usize, usize),
    perm: &[u8; 9],
) -> ([NodeRow; 81], usize, usize) {
    assert_eq!(
        ambiguous_origin.0, witness_origin.0,
        "both boxes must share a row band so target and Z end up in the same row"
    );
    let mut grid: [NodeRow; 81] = [blank_row(); 81];
    let (ar, ac) = ambiguous_origin;
    let mut k = 0usize;
    for r in ar..ar + 3 {
        for c in ac..ac + 3 {
            let pos = box_pos(r, c);
            let cib = cell_in_box(pos);
            // Target (k=7) and the parked companion (k=3, the box's MIDDLE
            // row) stay empty. The companion is deliberately parked on the
            // middle row, not the box's own top row (k=0..2) — that row
            // coincides with the WITNESS box's top row (same row band), and
            // the witness box's k=0 predecessor carries `perm[8]` (see
            // below), which would strip `perm[8]` from the companion's row-
            // based candidates before the target ever resolves, same class
            // of bug as the G1 column-forced write-order lesson.
            if cib == 7 || cib == 3 {
                continue;
            }
            write_cell(&mut grid[pos], perm[k], true);
            k += 1;
        }
    }
    let target = box_pos(ar + 2, ac + 1); // k=7 within the ambiguous box
    let (wr, wc) = witness_origin;
    // `perm[8]` is deliberately placed at the FIRST predecessor (k=0), never
    // at k=6/k=7 (the witness box's own shared-row cells) — landing it there
    // would eliminate `perm[8]` from the target's row-based candidates as a
    // side effect, before the fork ever runs (see the G1 column-forced
    // fixture's write-order lesson, same mechanism).
    let z_digits: [u8; 8] = [
        perm[8], perm[0], perm[1], perm[2], perm[3], perm[4], perm[5], perm[6],
    ];
    let mut zk = 0usize;
    for r in wr..wr + 3 {
        for c in wc..wc + 3 {
            let pos = box_pos(r, c);
            if cell_in_box(pos) == 8 {
                continue; // Z itself, left empty
            }
            write_cell(&mut grid[pos], z_digits[zk], true);
            zk += 1;
        }
    }
    let z = box_pos(wr + 2, wc + 2); // k=8 within the witness box
    write_box_witness(&mut grid);
    (grid, z, target)
}

// ─────────────────────────────────────────────────────────────────────────
// Fork-return (G3): bifurcation as an explicit counterfactual WORLD
// ─────────────────────────────────────────────────────────────────────────

fn rows_equal(a: &NodeRow, b: &NodeRow) -> bool {
    a.key == b.key && a.edges == b.edges && a.value == b.value
}

/// `true` iff some empty cell's full-sweep candidate set is empty — a
/// contradiction reached by propagation.
fn has_contradiction(grid: &[NodeRow; 81]) -> bool {
    (0..81).any(|p| read_cell(&grid[p]).is_none() && candidates_from_full_sweep(grid, p).is_empty())
}

/// Bifurcate on `pos` with the guessed `digit`: clone the slab as a
/// counterfactual world, assign the guess there, and report whether
/// propagation (a full board-wide candidate re-derivation — every other
/// cell's constraints recomputed against the guess in one shot) reaches a
/// contradiction. The clone is NEVER written back — only the caller decides
/// what (if anything) to write to the real slab from the verdict (the
/// fork-return rule, §4c).
fn try_world(grid: &[NodeRow; 81], pos: usize, digit: u8) -> bool {
    let mut world = *grid; // NodeRow is Copy — an explicit, deliberate clone
    write_cell(&mut world[pos], digit, false);
    has_contradiction(&world)
}

// ─────────────────────────────────────────────────────────────────────────
// One election pass: naked singles + hidden singles, zero-copy over the row
// slice via the predicate-sweep helpers above.
// ─────────────────────────────────────────────────────────────────────────

/// One (cell, digit) election, in the order it was made — the unit the
/// solve PATH is built from (Levenshtein-comparable to a teacher path).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Election {
    pos: usize,
    digit: u8,
}

/// Apply naked singles (candidate set size 1) — deterministic, position-
/// ascending so replay is bit-identical.
fn apply_naked_singles(grid: &mut [NodeRow; 81]) -> Vec<Election> {
    let mut made = Vec::new();
    for pos in 0..81 {
        if read_cell(&grid[pos]).is_some() {
            continue;
        }
        let cands = candidates_from_full_sweep(grid, pos);
        if cands.len() == 1 {
            write_cell(&mut grid[pos], cands[0], false);
            made.push(Election {
                pos,
                digit: cands[0],
            });
        }
    }
    made
}

/// Try a bifurcation-resolvable cell: pick the first empty cell with exactly
/// 2 candidates, try the FIRST candidate as a counterfactual world; if it
/// contradicts, the SECOND candidate is forced and lands as a real election
/// on the main slab (only the elimination returns — the fork's world is
/// discarded either way).
fn try_bifurcate(grid: &mut [NodeRow; 81]) -> Option<Election> {
    for pos in 0..81 {
        if read_cell(&grid[pos]).is_some() {
            continue;
        }
        let cands = candidates_from_full_sweep(grid, pos);
        if cands.len() == 2 {
            let (a, b) = (cands[0], cands[1]);
            if try_world(grid, pos, a) {
                write_cell(&mut grid[pos], b, false);
                return Some(Election { pos, digit: b });
            }
            if try_world(grid, pos, b) {
                write_cell(&mut grid[pos], a, false);
                return Some(Election { pos, digit: a });
            }
        }
    }
    None
}

/// Policy identity — the two style atoms this probe grades.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Policy {
    /// Elections-first: exhaust naked+hidden singles every pass; bifurcate
    /// only when genuinely stuck (both count toward passes exercised, but
    /// bifurcation is the LAST resort).
    ElectionsFirst,
    /// Bifurcate-early: try one bifurcation BEFORE exhausting singles each
    /// pass, if a 2-candidate cell exists.
    BifurcateEarly,
    /// Fork-refusing: elections only, NEVER bifurcates — the G4 second-half
    /// policy that demonstrates a bifurcation-required puzzle does not fully
    /// migrate.
    ForkRefusing,
}

/// Run one policy to a fixed point (or `max_passes`), returning the full
/// election path (for Levenshtein) and the pass-by-pass Hamming distance to
/// `solution` (for the monotone gate).
fn run_policy(
    grid: &mut [NodeRow; 81],
    policy: Policy,
    solution: &[u8; 81],
    max_passes: usize,
) -> (Vec<Election>, Vec<usize>) {
    let mut path = Vec::new();
    let mut hamming_series = vec![hamming(grid, solution)];
    for _ in 0..max_passes {
        let mut made_any = false;
        // Bifurcate-early: try a fork BEFORE running elections this pass —
        // the policy that reaches for a guess before exhausting the cheap
        // constraints (§4c's "elections-first vs bifurcate-early" split).
        if matches!(policy, Policy::BifurcateEarly) {
            if let Some(e) = try_bifurcate(grid) {
                path.push(e);
                made_any = true;
            }
        }
        // NAKED singles only here — hidden singles are exercised separately
        // and exhaustively by G2. Threading them through the policy loop
        // too would let a hidden-single shortcut resolve an engineered
        // 2-candidate tie BEFORE either policy ever gets to act on it
        // (hidden-single detection subsumes exactly this shape of
        // ambiguity), erasing the elections-first/bifurcate-early split
        // this loop exists to demonstrate (G5).
        let naked = apply_naked_singles(grid);
        made_any |= !naked.is_empty();
        path.extend(naked);
        // Elections-first: bifurcation is the LAST resort, only when a pass
        // made zero progress via singles. Fork-refusing never bifurcates.
        if !made_any && matches!(policy, Policy::ElectionsFirst) {
            if let Some(e) = try_bifurcate(grid) {
                path.push(e);
                made_any = true;
            }
        }
        hamming_series.push(hamming(grid, solution));
        if !made_any {
            break;
        }
    }
    (path, hamming_series)
}

fn hamming(grid: &[NodeRow; 81], solution: &[u8; 81]) -> usize {
    (0..81)
        .filter(|&p| match read_cell(&grid[p]) {
            Some((d, _)) => d != solution[p],
            None => true, // an unfilled cell counts as a mismatch
        })
        .count()
}

/// The canonical teacher path over a set of `(pos, digit)` resolutions:
/// row-major order of the positions (the ground truth the solve path is
/// judged against).
fn teacher_path(blanks_digits: &[(usize, u8)]) -> Vec<Election> {
    let mut ordered = blanks_digits.to_vec();
    ordered.sort_by_key(|&(p, _)| {
        let (r, c) = row_col_of(p);
        r * 9 + c
    });
    ordered
        .into_iter()
        .map(|(pos, digit)| Election { pos, digit })
        .collect()
}

/// Levenshtein edit distance over `(pos, digit)` election tokens.
fn levenshtein(a: &[Election], b: &[Election]) -> usize {
    let (n, m) = (a.len(), b.len());
    let mut prev: Vec<usize> = (0..=m).collect();
    for i in 1..=n {
        let mut cur = vec![0usize; m + 1];
        cur[0] = i;
        for j in 1..=m {
            let cost = if a[i - 1] == b[j - 1] { 0 } else { 1 };
            cur[j] = (prev[j] + 1).min(cur[j - 1] + 1).min(prev[j - 1] + cost);
        }
        prev = cur;
    }
    prev[m]
}

// ─────────────────────────────────────────────────────────────────────────
// Grading + the triangle motion (G5)
// ─────────────────────────────────────────────────────────────────────────

#[derive(Clone, Copy, Debug)]
struct Grade {
    solved: bool,
    cost: usize,
    path_lev: usize,
}

/// Lower is better; unsolved is always worse than solved, whatever the cost.
fn grade_score(g: Grade) -> i64 {
    if !g.solved {
        i64::MAX
    } else {
        g.cost as i64 * 100 + g.path_lev as i64
    }
}

/// Grade one policy over a pre-built fixture. `base_grid` already carries
/// its givens with `blanks_digits`' positions left empty; only those
/// positions' correct resolutions are graded ("solved" = every blanked cell
/// reached its correct digit — the OTHER 79 cells are context, not part of
/// the puzzle this fixture poses). `run_policy` still needs a full-board
/// solution array for its Hamming series (unused here beyond that call), so
/// unblanked positions borrow whatever digit `base_grid` already gives them.
fn run_and_grade(
    base_grid: &[NodeRow; 81],
    blanks_digits: &[(usize, u8)],
    policy: Policy,
) -> Grade {
    let mut solution = [0u8; 81];
    for (p, s) in solution.iter_mut().enumerate() {
        if let Some((d, _)) = read_cell(&base_grid[p]) {
            *s = d;
        }
    }
    for &(p, d) in blanks_digits {
        solution[p] = d;
    }
    let mut grid = *base_grid;
    let (path, _) = run_policy(&mut grid, policy, &solution, 40);
    let solved = blanks_digits
        .iter()
        .all(|&(p, d)| matches!(read_cell(&grid[p]), Some((gd, _)) if gd == d));
    let teacher = teacher_path(blanks_digits);
    Grade {
        solved,
        cost: path.len(),
        path_lev: levenshtein(&path, &teacher),
    }
}

/// Designated policy row: family ordinal 0 carries the A/B winner marker.
/// `0xAA` = A, `0xBB` = B, `0x00` = null (zero-fallback).
const FAMILY_ORDINAL: u8 = 0;
const ATOM_A: u8 = 0xAA;
const ATOM_B: u8 = 0xBB;

fn atom_of(policy: Policy) -> u8 {
    match policy {
        Policy::ElectionsFirst => ATOM_A,
        Policy::BifurcateEarly => ATOM_B,
        Policy::ForkRefusing => 0,
    }
}

fn winner(grade_a: Grade, grade_b: Grade) -> Policy {
    if grade_score(grade_a) <= grade_score(grade_b) {
        Policy::ElectionsFirst
    } else {
        Policy::BifurcateEarly
    }
}

fn learned_lane_atom(row: &NodeRow) -> u8 {
    row.style_lane(ValueTenant::LearnedStyle)[FAMILY_ORDINAL as usize]
}

fn frozen_lane_atom(row: &NodeRow) -> u8 {
    row.style_lane(ValueTenant::FrozenStyle)[FAMILY_ORDINAL as usize]
}

fn set_learned(row: &mut NodeRow, atom: u8) {
    let mut lane = row.style_lane(ValueTenant::LearnedStyle);
    lane[FAMILY_ORDINAL as usize] = atom;
    row.set_style_lane(ValueTenant::LearnedStyle, lane);
}

fn set_frozen(row: &mut NodeRow, atom: u8) {
    let mut lane = row.style_lane(ValueTenant::FrozenStyle);
    lane[FAMILY_ORDINAL as usize] = atom;
    row.set_style_lane(ValueTenant::FrozenStyle, lane);
}

fn main() {
    println!("════════ PROBE-SUDOKU-TEACHER ════════\n");

    let base = base_solution_boxmajor();
    let solved_variants = [
        permute_solution(&base, &PERM_A),
        permute_solution(&base, &PERM_B),
        permute_solution(&base, &PERM_C),
    ];

    let mut gates: Vec<(&str, bool, String)> = Vec::new();

    // ═══════════════════ G1 — horizon ═══════════════════
    // Box-forced fixture: box 0, target k=8, predecessors 0..7 GIVEN 1..8.
    let mut g1_box = grid_from_solution(&solved_variants[0]);
    // Clear the whole grid to isolate the box-forced scenario.
    for p in 0..81 {
        clear_cell(&mut g1_box[p]);
    }
    for k in 0..8 {
        write_cell(&mut g1_box[k], (k + 1) as u8, true);
    }
    write_box_witness(&mut g1_box);
    let box_target = 8usize; // box 0, k=8
    let box_lane_cands = candidates_from_box_lane(&g1_box, box_target);
    let box_sweep_cands = candidates_from_full_sweep(&g1_box, box_target);

    // Column-forced fixture: box 4 (center), target (r=5,c=3) k=6.
    let mut g1_col = grid_from_solution(&solved_variants[0]);
    for p in 0..81 {
        clear_cell(&mut g1_col[p]);
    }
    // predecessors k=0..5 in box4: rows 3,4 cols 3,4,5 -> digits 1..6
    write_cell(&mut g1_col[box_pos(3, 3)], 1, true);
    write_cell(&mut g1_col[box_pos(3, 4)], 2, true);
    write_cell(&mut g1_col[box_pos(3, 5)], 3, true);
    write_cell(&mut g1_col[box_pos(4, 3)], 4, true);
    write_cell(&mut g1_col[box_pos(4, 4)], 5, true);
    write_cell(&mut g1_col[box_pos(4, 5)], 6, true);
    // column givens outside box4, same column (c=3), eliminate 7 and 8
    write_cell(&mut g1_col[box_pos(0, 3)], 7, true);
    write_cell(&mut g1_col[box_pos(1, 3)], 8, true);
    write_box_witness(&mut g1_col);
    let col_target = box_pos(5, 3); // k=6 in box4
    let col_lane_cands = candidates_from_box_lane(&g1_col, col_target);
    let col_sweep_cands = candidates_from_full_sweep(&g1_col, col_target);

    println!("── G1: horizon ──");
    println!(
        "  box-forced   lane {box_lane_cands:?}  sweep {box_sweep_cands:?}  (lane alone sufficient)"
    );
    println!(
        "  column-forced lane {col_lane_cands:?}  sweep {col_sweep_cands:?}  (sweep must fire)"
    );
    let g1_box_ok =
        box_lane_cands.len() == 1 && box_lane_cands == vec![9] && box_sweep_cands == box_lane_cands; // sweep AGREES but adds nothing — stays silent
    let g1_col_ok = col_lane_cands.len() > 1 // lane alone is NOT enough
        && col_sweep_cands.len() == 1 // sweep resolves it
        && col_sweep_cands != col_lane_cands; // sweep genuinely changed the answer — fired
    let g1_pass = g1_box_ok && g1_col_ok;
    gates.push((
        "G1",
        g1_pass,
        format!(
            "box lane {:?}=={:?} sweep-silent={}  |  col lane {:?} sweep {:?} sweep-fired={}",
            box_lane_cands,
            box_sweep_cands,
            box_sweep_cands == box_lane_cands,
            col_lane_cands,
            col_sweep_cands,
            col_sweep_cands != col_lane_cands
        ),
    ));

    // ═══════════════════ G2 — hidden single ═══════════════════
    // Hidden-single-not-naked fixture: row 0, target (r=0,c=4).
    let mut g2_hidden = grid_from_solution(&solved_variants[0]);
    for p in 0..81 {
        clear_cell(&mut g2_hidden[p]);
    }
    write_cell(&mut g2_hidden[box_pos(1, 1)], 6, true); // box0: kills 6 for row0 c0..2
    write_cell(&mut g2_hidden[box_pos(1, 7)], 6, true); // box2: kills 6 for row0 c6..8
    write_cell(&mut g2_hidden[box_pos(4, 3)], 6, true); // col3: kills 6 for row0 c3
    write_cell(&mut g2_hidden[box_pos(4, 5)], 6, true); // col5: kills 6 for row0 c5
    write_box_witness(&mut g2_hidden);
    let hidden_found = hidden_singles(&g2_hidden);
    let target_r0c4 = box_pos(0, 4);
    let g2_hidden_ok = hidden_found.contains(&(target_r0c4, 6))
        && candidates_from_full_sweep(&g2_hidden, target_r0c4).len() > 1;

    // All-naked fixture: fully solved grid, blank 2 non-interacting cells.
    let mut g2_naked = grid_from_solution(&solved_variants[1]);
    let naked_targets = [box_pos(0, 0), box_pos(5, 6)];
    blank_positions(&mut g2_naked, &naked_targets);
    write_box_witness(&mut g2_naked);
    let naked_hidden_found = hidden_singles(&g2_naked);
    let g2_naked_ok = naked_hidden_found.is_empty()
        && naked_targets
            .iter()
            .all(|&p| candidates_from_full_sweep(&g2_naked, p).len() == 1);

    println!("\n── G2: hidden single ──");
    println!(
        "  seeded hidden (not naked): found {hidden_found:?}  target=(pos {target_r0c4}, digit 6)"
    );
    println!(
        "  all-naked puzzle: hidden singles found = {}",
        naked_hidden_found.len()
    );
    let g2_pass = g2_hidden_ok && g2_naked_ok;
    gates.push((
        "G2",
        g2_pass,
        format!(
            "seeded fires (contains target)={} all-naked silent (count==0)={}",
            hidden_found.contains(&(target_r0c4, 6)),
            naked_hidden_found.is_empty()
        ),
    ));

    // ═══════════════════ G3 — fork-return ═══════════════════
    // A genuine 2-candidate tie (box4's target) whose witness peer (box3's
    // Z, same row) is independently box-forced to one of the two candidates
    // — guessing that value at the target propagates to a contradiction at
    // Z (the fork-return machinery, `build_ambiguity_fixture` below).
    let (mut g3_main, z_target, fork_target) = build_ambiguity_fixture((3, 3), (3, 0), &PERM_A);
    assert_eq!(
        candidates_from_full_sweep(&g3_main, z_target),
        vec![8],
        "Z must be box-forced to {{8}} independent of the target"
    );

    let pre_fork_snapshot: Vec<NodeRow> = g3_main.to_vec();
    let target_cands = candidates_from_full_sweep(&g3_main, fork_target);
    // Bifurcate explicitly and capture the verdict.
    let world_a_contradicts = try_world(&g3_main, fork_target, target_cands[0]);
    let world_b_contradicts =
        target_cands.len() > 1 && try_world(&g3_main, fork_target, target_cands[1]);
    let resolved_digit = if world_a_contradicts && target_cands.len() > 1 {
        Some(target_cands[1])
    } else if world_b_contradicts {
        Some(target_cands[0])
    } else {
        None
    };
    if let Some(d) = resolved_digit {
        write_cell(&mut g3_main[fork_target], d, false);
    }
    // Assert: main slab unchanged everywhere except `fork_target`.
    let mut only_target_changed = true;
    for p in 0..81 {
        if p == fork_target {
            continue;
        }
        if !rows_equal(&g3_main[p], &pre_fork_snapshot[p]) {
            only_target_changed = false;
        }
    }
    // Assert: the fork's WRONG guess (whichever contradicted) never landed.
    let wrong_guess = if world_a_contradicts {
        Some(target_cands[0])
    } else if world_b_contradicts && target_cands.len() > 1 {
        Some(target_cands[1])
    } else {
        None
    };
    let wrong_guess_absent = match (wrong_guess, read_cell(&g3_main[fork_target])) {
        (Some(w), Some((d, _))) => d != w,
        _ => true,
    };

    println!("\n── G3: fork-return ──");
    println!(
        "  target {fork_target} candidates {target_cands:?}  world(a) contradicts={world_a_contradicts}  world(b) contradicts={world_b_contradicts}"
    );
    println!("  resolved digit = {resolved_digit:?}  only-target-changed = {only_target_changed}  wrong-guess-absent = {wrong_guess_absent}");
    let g3_pass = resolved_digit.is_some()
        && only_target_changed
        && wrong_guess_absent
        && (world_a_contradicts ^ world_b_contradicts); // exactly one branch failed
    gates.push((
        "G3",
        g3_pass,
        format!(
            "resolved={:?} only_target_changed={only_target_changed} wrong_guess_absent={wrong_guess_absent} exactly_one_branch_failed={}",
            resolved_digit,
            world_a_contradicts ^ world_b_contradicts
        ),
    ));

    // ═══════════════════ G4 — quadrant census ═══════════════════
    // Easy puzzle: many givens, solvable via singles alone.
    let easy_blanks: Vec<usize> = (0..81).filter(|&p| p % 5 == 0).collect(); // ~16 blanks
    let mut g4_easy = grid_from_solution(&solved_variants[0]);
    blank_positions(&mut g4_easy, &easy_blanks);
    write_box_witness(&mut g4_easy);
    let (_, easy_hamming) = run_policy(
        &mut g4_easy,
        Policy::ElectionsFirst,
        &solved_variants[0],
        40,
    );
    let easy_census = quadrant_census(&g4_easy);

    // Bifurcation-required puzzle: the G3 fixture, run to completion under a
    // fork-refusing policy AND under bifurcate-early, from a fresh copy.
    let mut g4_hard_refuse: [NodeRow; 81] = pre_fork_snapshot
        .clone()
        .try_into()
        .unwrap_or_else(|_| panic!("pre_fork_snapshot must be exactly 81 rows"));
    let (_, refuse_hamming) = run_policy(
        &mut g4_hard_refuse,
        Policy::ForkRefusing,
        &solved_variants[0],
        40,
    );
    let refuse_census = quadrant_census(&g4_hard_refuse);

    let mut g4_hard_bifurcate: [NodeRow; 81] = pre_fork_snapshot
        .clone()
        .try_into()
        .unwrap_or_else(|_| panic!("pre_fork_snapshot must be exactly 81 rows"));
    let (_, bifurcate_hamming) = run_policy(
        &mut g4_hard_bifurcate,
        Policy::BifurcateEarly,
        &solved_variants[0],
        40,
    );
    let bifurcate_census = quadrant_census(&g4_hard_bifurcate);

    println!("\n── G4: quadrant census ──");
    println!("  easy puzzle    hamming series: {easy_hamming:?}");
    println!("  easy census    {easy_census:?}");
    println!("  hard/refuse    hamming series: {refuse_hamming:?}  census {refuse_census:?}");
    println!("  hard/bifurcate hamming series: {bifurcate_hamming:?}  census {bifurcate_census:?}");
    // Fire half (spec): a solvable puzzle's census migrates toward Wisdom.
    let easy_migrates = easy_census.wisdom > easy_census.staunen + easy_census.confusion;
    // Silent half (spec, verbatim): "a fork-refusing policy on a
    // bifurcation-required puzzle does NOT fully migrate." This puzzle is
    // the sparse box3/box4 fixture (~15 of 81 givens) — most of the board
    // has too little local constraint for singles to ever narrow past a
    // handful of candidates, so ForkRefusing (which never guesses) leaves
    // the majority of cells unresolved. `bifurcate_census` is printed for
    // comparison but NOT required to differ — G5 grades that difference by
    // solve PATH, not by final census (see below).
    let refuse_does_not_fully_migrate =
        refuse_census.wisdom < 81 && refuse_census.staunen + refuse_census.confusion > 0;
    let g4_pass = easy_migrates && refuse_does_not_fully_migrate;
    gates.push((
        "G4",
        g4_pass,
        format!("easy_migrates={easy_migrates} refuse_does_not_fully_migrate={refuse_does_not_fully_migrate} (bifurcate census printed above for comparison)"),
    ));

    // ═══════════════════ G5 — triangle motion ═══════════════════
    //
    // Both policies always resolve this fixture's two blanks — the split is
    // in the ORDER, not the outcome, and the order is a structural property
    // of the two policies, not of the digits used:
    //   * ElectionsFirst ALWAYS resolves Z (independently box-forced) before
    //     the target (which depends on Z), because elections only fire on
    //     already-singleton cells.
    //   * BifurcateEarly ALWAYS resolves the target FIRST (it forks on any
    //     2-candidate cell before running elections that pass), then Z
    //     follows via an ordinary naked single.
    // The teacher path is row-major order of the two blanked cells, so which
    // policy WINS (lower path-Levenshitein) is decided by the fixture's
    // GEOMETRY: swapping which box holds the ambiguous cell vs the witness
    // flips whether the teacher's row-major order matches [target, Z]
    // (favors B) or [Z, target] (favors A) — the mechanism `build_
    // ambiguity_fixture`'s geometry knob controls directly.

    // TRAIN: ambiguous box3, witness box4 — target's row-major position
    // (row 5, col 1) precedes Z's (row 5, col 5), matching BifurcateEarly's
    // natural [target, Z] order. Favors B. Z resolves to its box's own
    // missing digit (`perm[7]`); the target's OTHER candidate — `perm[8]` —
    // is the one that survives once Z's row exclusion removes `perm[7]`.
    let (train_grid, train_z, train_target) = build_ambiguity_fixture((3, 0), (3, 3), &PERM_A);
    let train_blanks = vec![(train_z, PERM_A[7]), (train_target, PERM_A[8])];
    let grade_a_hard = run_and_grade(&train_grid, &train_blanks, Policy::ElectionsFirst);
    let grade_b_hard = run_and_grade(&train_grid, &train_blanks, Policy::BifurcateEarly);
    let train_winner = winner(grade_a_hard, grade_b_hard);

    println!("\n── G5: triangle motion ──");
    println!("  train (hard) grade A {grade_a_hard:?} grade B {grade_b_hard:?} → winner {train_winner:?}");

    // PROMOTE case: held-out uses the SAME favor-B geometry (ambiguous
    // box3 / witness box4) with a DIFFERENT digit permutation — B should
    // also win there, since the win is structural, not digit-dependent.
    let (promote_grid, promote_z, promote_target) =
        build_ambiguity_fixture((3, 0), (3, 3), &PERM_C);
    let heldout_promote_blanks = vec![(promote_z, PERM_C[7]), (promote_target, PERM_C[8])];
    let grade_a_ho1 = run_and_grade(
        &promote_grid,
        &heldout_promote_blanks,
        Policy::ElectionsFirst,
    );
    let grade_b_ho1 = run_and_grade(
        &promote_grid,
        &heldout_promote_blanks,
        Policy::BifurcateEarly,
    );
    let heldout_promote_winner = winner(grade_a_ho1, grade_b_ho1);
    println!(
        "  held-out (promote case) grade A {grade_a_ho1:?} grade B {grade_b_ho1:?} → winner {heldout_promote_winner:?}"
    );

    // REFUSE case: held-out uses the OPPOSITE geometry (ambiguous box4,
    // witness box3 — the same layout G3 uses), where Z's row-major position
    // precedes the target's, matching ElectionsFirst's natural [Z, target]
    // order instead. Favors A — a genuine train/held-out disagreement.
    let (refuse_grid, refuse_z, refuse_target) = build_ambiguity_fixture((3, 3), (3, 0), &PERM_B);
    let heldout_refuse_blanks = vec![(refuse_z, PERM_B[7]), (refuse_target, PERM_B[8])];
    let grade_a_ho2 = run_and_grade(&refuse_grid, &heldout_refuse_blanks, Policy::ElectionsFirst);
    let grade_b_ho2 = run_and_grade(&refuse_grid, &heldout_refuse_blanks, Policy::BifurcateEarly);
    let heldout_refuse_winner = winner(grade_a_ho2, grade_b_ho2);
    println!(
        "  held-out (refuse case)  grade A {grade_a_ho2:?} grade B {grade_b_ho2:?} → winner {heldout_refuse_winner:?}"
    );

    // Two DISTINCT designated policy rows so the promote/refuse write
    // histories don't alias each other.
    let mut policy_row_promote = blank_row();
    let mut policy_row_refuse = blank_row();

    let promote_snapshot_before = policy_row_promote.value;
    set_learned(&mut policy_row_promote, atom_of(train_winner));
    let promote_learned = learned_lane_atom(&policy_row_promote);
    let promote_should_promote = heldout_promote_winner == train_winner;
    if promote_should_promote {
        set_frozen(&mut policy_row_promote, promote_learned);
    }
    let promote_frozen = frozen_lane_atom(&policy_row_promote);

    let refuse_snapshot_before = policy_row_refuse.value;
    set_learned(&mut policy_row_refuse, atom_of(train_winner));
    let refuse_learned = learned_lane_atom(&policy_row_refuse);
    let refuse_should_promote = heldout_refuse_winner == train_winner;
    if refuse_should_promote {
        set_frozen(&mut policy_row_refuse, refuse_learned);
    }
    let refuse_frozen = frozen_lane_atom(&policy_row_refuse);

    // Write-isolation: only the LearnedStyle (and, for the promote row, the
    // FrozenStyle) byte at FAMILY_ORDINAL changed — everything else in the
    // 480-byte value slab is untouched.
    let learned_offset = ValueTenant::LearnedStyle.value_offset() + FAMILY_ORDINAL as usize;
    let frozen_offset = ValueTenant::FrozenStyle.value_offset() + FAMILY_ORDINAL as usize;
    let promote_isolated = (0..480).all(|i| {
        if i == learned_offset || i == frozen_offset {
            true
        } else {
            policy_row_promote.value[i] == promote_snapshot_before[i]
        }
    });
    let refuse_isolated = (0..480).all(|i| {
        if i == learned_offset {
            true
        } else {
            policy_row_refuse.value[i] == refuse_snapshot_before[i]
        }
    });

    println!(
        "  promote row: learned={promote_learned:#04x} frozen={promote_frozen:#04x} should_promote={promote_should_promote} isolated={promote_isolated}"
    );
    println!(
        "  refuse row:  learned={refuse_learned:#04x} frozen={refuse_frozen:#04x} should_promote={refuse_should_promote} isolated={refuse_isolated}"
    );

    let g5_promote_ok = promote_should_promote
        && promote_frozen == promote_learned
        && promote_frozen != 0
        && promote_isolated;
    let g5_refuse_ok = !refuse_should_promote && refuse_frozen == 0 && refuse_isolated;
    let g5_pass = g5_promote_ok && g5_refuse_ok;
    gates.push((
        "G5",
        g5_pass,
        format!(
            "promote(learned==frozen, nonzero, isolated)={g5_promote_ok} refuse(frozen stays 0, isolated)={g5_refuse_ok}"
        ),
    ));

    // ═══════════════════ G6 — Hamming monotone ═══════════════════
    println!("\n── G6: Hamming monotone ──");
    let easy_monotone = easy_hamming.windows(2).all(|w| w[1] <= w[0]);
    let bifurcate_monotone = bifurcate_hamming.windows(2).all(|w| w[1] <= w[0]);
    let refuse_monotone = refuse_hamming.windows(2).all(|w| w[1] <= w[0]);
    // Anti-vacuity: at least one series must show a STRICT decrease
    // somewhere, or the "monotone" claim is checking an all-flat sequence.
    let easy_strictly_decreases = easy_hamming.windows(2).any(|w| w[1] < w[0]);
    println!(
        "  easy monotone={easy_monotone} (strict decrease somewhere={easy_strictly_decreases})  bifurcate monotone={bifurcate_monotone}  refuse monotone={refuse_monotone}"
    );
    let g6_pass = easy_monotone && bifurcate_monotone && refuse_monotone && easy_strictly_decreases;
    gates.push((
        "G6",
        g6_pass,
        format!(
            "easy={easy_monotone} bifurcate={bifurcate_monotone} refuse={refuse_monotone} strict_decrease={easy_strictly_decreases}"
        ),
    ));

    // ═══════════════════ Report ═══════════════════
    println!("\n════════ GATES ════════");
    let mut all_pass = true;
    for (name, pass, detail) in &gates {
        println!(
            "  [{}] {name} — {detail}",
            if *pass { "PASS" } else { "FAIL" }
        );
        all_pass &= *pass;
    }
    if all_pass {
        println!("\nALL GATES GREEN");
    } else {
        panic!("PROBE-SUDOKU-TEACHER: one or more gates FAILED — see report above");
    }
}

// ─────────────────────────────────────────────────────────────────────────
// Quadrant census (G4)
// ─────────────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, Default)]
struct Census {
    staunen: usize,
    confusion: usize,
    boredom: usize,
    wisdom: usize,
}

/// Per-cell entropy = normalized candidate-set size (`len/9`, `0` for a
/// filled cell); energy = solved-peer fraction over the box witness lane
/// (fraction of box peers already filled). Filled cells are always Wisdom
/// (entropy 0, energy 1 — the `Quadrant::classify` split boundary, `>= 0.5`
/// on the energy axis, lands a fully-solved-peer cell in Wisdom).
fn quadrant_census(grid: &[NodeRow; 81]) -> Census {
    let mut c = Census::default();
    for pos in 0..81 {
        let (entropy, energy) = if read_cell(&grid[pos]).is_some() {
            (0.0, 1.0)
        } else {
            let cand_len = candidates_from_full_sweep(grid, pos).len();
            // A contradicted cell (empty, ZERO candidates) is maximal
            // entropy, NOT zero — `cand_len == 0` means "unsatisfiable",
            // the opposite of "resolved". Reading it as 0.0/9.0 = 0.0 would
            // classify a contradiction as Wisdom whenever its box-witness
            // energy happens to be >= 0.5 (`Quadrant::classify`), silently
            // counting an unsatisfiable board as resolved.
            let entropy = if cand_len == 0 {
                1.0
            } else {
                cand_len as f64 / 9.0
            };
            let k = cell_in_box(pos);
            let energy = if k == 0 {
                0.0
            } else {
                let filled = (0..k)
                    .filter(|&j| read_cell(&grid[box_of(pos) * 9 + j]).is_some())
                    .count();
                filled as f64 / k as f64
            };
            (entropy, energy)
        };
        match Quadrant::classify(entropy, energy) {
            Quadrant::Staunen => c.staunen += 1,
            Quadrant::Confusion => c.confusion += 1,
            Quadrant::Boredom => c.boredom += 1,
            Quadrant::Wisdom => c.wisdom += 1,
        }
    }
    c
}

/// **F1 falsifier — the contradicted cell must NOT score as Wisdom.**
///
/// `quadrant_census` maps an empty cell's entropy to `cand_len / 9`. A
/// *contradicted* cell (empty, but every digit eliminated by its peers) has
/// `cand_len == 0`, which naively gives entropy `0.0` — and `0.0` entropy with
/// high box energy classifies as **Wisdom**, i.e. an UNSATISFIABLE cell counted
/// as a resolved one. That inflates `wisdom` and can let a migration claim pass
/// on a board that cannot be solved at all.
///
/// The fix scores a zero-candidate empty cell at entropy `1.0` (maximal —
/// a contradiction is the opposite of resolved). This test exists because the
/// fix was otherwise **unfalsifiable**: no existing fixture reaches a
/// zero-candidate cell during census, so the probe's printed output is
/// byte-identical with and without the fix. Without this test the correction
/// would be indistinguishable from a no-op.
#[test]
fn contradicted_cell_is_not_counted_as_wisdom() {
    // Box-major layout: box 0 owns positions 0..=8; `cell_in_box(8) == 8`, so
    // position 8 has all 8 of its box predecessors before it (energy = 8/8).
    let mut grid: [NodeRow; 81] = [blank_row(); 81];
    for (i, pos) in (0..8).enumerate() {
        write_cell(&mut grid[pos], (i + 1) as u8, true); // digits 1..=8
    }
    // Eliminate the last remaining digit (9) via a row/column peer OUTSIDE
    // box 0, so position 8 is empty with an EMPTY candidate set.
    let (r8, c8) = row_col_of(8);
    let peer = (0..81)
        .find(|&p| {
            if box_of(p) == box_of(8) || read_cell(&grid[p]).is_some() {
                return false;
            }
            let (pr, pc) = row_col_of(p);
            pr == r8 || pc == c8
        })
        .expect("position 8 must have a row/column peer outside its own box");
    write_cell(&mut grid[peer], 9, true);

    // ── preconditions, asserted so the test cannot pass vacuously ──
    assert!(
        read_cell(&grid[8]).is_none(),
        "position 8 must be EMPTY for this to be a contradiction rather than a fill"
    );
    assert!(
        candidates_from_full_sweep(&grid, 8).is_empty(),
        "fixture failed to contradict position 8: candidates = {:?}",
        candidates_from_full_sweep(&grid, 8)
    );

    // ── the claim ──
    // Every FILLED cell scores (entropy 0, energy 1) = Wisdom. Position 8 is
    // empty and contradicted, so Wisdom must equal exactly the filled count.
    // Under the old `cand_len / 9` mapping it would be filled + 1.
    let filled = (0..81).filter(|&p| read_cell(&grid[p]).is_some()).count();
    let census = quadrant_census(&grid);
    assert_eq!(
        census.wisdom, filled,
        "a contradicted cell was counted as Wisdom (wisdom={} vs filled={}) — \
         the zero-candidate entropy inversion has regressed",
        census.wisdom, filled
    );
}
