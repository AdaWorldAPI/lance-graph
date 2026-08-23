//! PROBE-METACOGNITIVE-TRIANGLE-1 — the missing arrow: a lower rung that
//! READS its policy from `StyleLane::Frozen`, a higher rung that reads the
//! lower rung's RECEIPT (never the puzzle) and actuates the triangle —
//! including the first production-path call of `MailboxSoA::promote_family`.
//!
//! # What the audit found (2026-08-23, main @ f5e27c9d)
//!
//! The autopoiesis triangle's STORAGE and MECHANICS are complete and tested:
//! `StyleLane` (`contract::soa_view`), the three `ValueTenant` lanes at
//! offsets 152/164/176, `MailboxSoA::{set_style_lane, set_style_atom,
//! promote_family}`, `MailboxSoaView::{style_lane_at, triangle_at}`. The
//! teacher probe (`lance-graph-planner/examples/probe_sudoku_teacher.rs`,
//! G1–G7 green) demonstrated explore→learned→frozen motion. But two
//! half-arrows were missing, and `persona-vs-rung-ladder.md` O6 says it
//! outright ("triangle structure unbuilt"):
//!
//! - **Read side:** no code anywhere read `StyleLane::Frozen` to CHOOSE how
//!   to reason — "dispatch reads Frozen" was doc-intent only.
//! - **Decide side:** no code consumed a receipt of a reasoning run to
//!   decide keep/explore/promote. The teacher probe computed its decision
//!   from fresh grades and merely RECORDED it into lanes; and
//!   `promote_family` — the real owner-mechanics actuator — had ZERO
//!   callers outside its own unit tests.
//!
//! This probe closes exactly that loop, once, falsifier-first:
//!
//! ```text
//! frozen lane ──READ──▶ lower rung reasons about the puzzle
//!                              │
//!                              ▼
//!                    RungReceipt (warranted assignments, fixed point,
//!                    unresolved count, kernel side-channel events)
//!                              │
//!            meta pass (higher rung) — object = the receipt, NEVER the grid
//!                              │
//!          ┌─────────┬─────────┼──────────────┐
//!          ▼         ▼         ▼              ▼
//!      KeepFrozen TryExplore RecordLearned promote_family
//!          │                                  │
//!          └────────── next run READS the lane it chose ◀┘
//! ```
//!
//! # The warrant boundary, carried in the receipt (not a new trait)
//!
//! Selection may be heuristic; TRUTH-BEARING destructive mutation must be
//! warranted. Here that is concrete: every digit this probe commits carries
//! ≥1 recorded EXCLUSION WARRANT (a named constraint fact — "digit d is
//! excluded from cell e by a peer holding d"), and is additionally checked
//! against an independent backtracking oracle. The recipe kernels
//! (TCP/TCF/CUR — the #995/#997 collision trio) run as SIDE-CHANNEL
//! observers on CLONED candidate sets: their output can never write back
//! into puzzle truth (F15 by construction), and when TCF manufactures a
//! singleton from an unresolved equal-1/n set with zero warrants, the meta
//! layer classifies it `UnwarrantedCertainty` and the cell stays unresolved
//! (F3). Nothing edits `delta_conf` to break the #997 tie — the meta layer
//! instead measures that the EXACT transitions already separate TCF
//! (→singleton) from TCP/CUR (→identity) where the coarse `(fired, Δconf
//! sign)` signature collides, and returns `ObserverInsufficient` (F8).
//!
//! # Scope fences (slice 1 — what this probe does NOT do)
//!
//! - **No CausalEdge64.** This file never imports `causal_edge`; bits
//!   59..63 are untouched by construction (F12 trivially holds; F10/F11 are
//!   the Kanban-hinge slice, not this one).
//! - **No Kanban wiring.** `InnerCouncil`/`CollapseHint`'s production edge
//!   (`verdict_from → select_tactic`) stays "designed, not wired" — the
//!   board already names it; consuming this probe's receipt there is the
//!   NEXT slice, deliberately.
//! - **No new types in src/.** All scaffolding here is probe-local; the
//!   only load-bearing calls are the existing triangle surface.
//! - **The Revision hinge uses the SHIPPED counterfactual surface, not a
//!   local pseudo-Revision.** The A-vs-B comparison is a
//!   [`FreeEnergyComparison`] (residual free energy = unresolved/81), its
//!   verdict is a [`RevisionOutcome`], and the Explore run executes in a
//!   real counterfactual lane: [`deposit_counterfactual`] stamps a
//!   [`RawEdge`] with the `−6` mantissa when the meta pass declares the
//!   split (frozen commitment vs explore hypothesis), and the mantissa is
//!   cleared to `0` once the revision commits — the exact step-5 protocol
//!   `revise_if_minority_wins` documents. What is NOT called: the two
//!   `todo!()` bodies (`CounterfactualMailbox::*`,
//!   `revise_if_minority_wins` itself), which stay blocked on D-PERSONA-5's
//!   ractor outer-swarm; this probe exercises the same decision shape
//!   through the shipped pure pieces and does not fake the actor arm.
//!   The invariant this buys for free: *exploration may be destructive
//!   inside the counterfactual lane; commitment may not be destructive
//!   without warrant* — the Explore run's grid never becomes observed
//!   truth, only its RECEIPT crosses back, and only `promote_family`
//!   (post-held-out) moves policy.
//! - **The Explore lane's atom is set by the probe arms** (including a
//!   deliberate degenerate explore for the can-it-stay-silent falsifier),
//!   standing in for the P64 perturbation ladder, which is not wired here.
//!
//! Falsifiers covered: F1 F2 F3 F4 F5 F6 F7 F8 F13 F14 F15.
//! Deferred with scope notes: F9 (reason-context equivalence), F10/F11
//! (Kanban hinge), F12 (held by construction, no causal-edge import).
//!
//! Run: `cargo run -p cognitive-shader-driver --example probe_metacognitive_triangle`

use cognitive_shader_driver::mailbox_soa::MailboxSoA;
use lance_graph_contract::counterfactual::{
    deposit_counterfactual, EpisodicEdge, FreeEnergyComparison, RawEdge, RevisionOutcome,
};
use lance_graph_contract::escalation::{CollapseHint, CouncilVerdict};
use lance_graph_contract::recipe_kernels::{kernel, MaturityPolicy, ThoughtCtx};
use lance_graph_contract::soa_view::{MailboxSoaView, StyleLane};

// ── Sudoku substrate (independent, ordinary, oracle-checked) ──────────────

type Grid = [[u8; 9]; 9];

/// The Sudoku Wikipedia article's example puzzle (same fixture as #997).
const BASE_PUZZLE: Grid = [
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

/// Count solutions up to `cap` — the independent oracle (shares no code with
/// the policy propagation below beyond `candidates`, which is the puzzle's
/// own legality definition, not a solving strategy).
fn count_solutions(grid: &Grid, cap: usize) -> usize {
    fn go(g: &mut Grid, cap: usize, found: &mut usize) {
        if *found >= cap {
            return;
        }
        for r in 0..9 {
            for c in 0..9 {
                if g[r][c] == 0 {
                    for d in candidates(g, r, c) {
                        g[r][c] = d;
                        go(g, cap, found);
                        g[r][c] = 0;
                        if *found >= cap {
                            return;
                        }
                    }
                    return;
                }
            }
        }
        *found += 1;
    }
    let mut g = *grid;
    let mut found = 0;
    go(&mut g, cap, &mut found);
    found
}

fn solve_unique(grid: &Grid) -> Option<Grid> {
    fn go(g: &mut Grid) -> bool {
        for r in 0..9 {
            for c in 0..9 {
                if g[r][c] == 0 {
                    for d in candidates(g, r, c) {
                        g[r][c] = d;
                        if go(g) {
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
    (count_solutions(grid, 2) == 1).then(|| {
        let mut g = *grid;
        assert!(go(&mut g));
        g
    })
}

// ── Policies, warrants, receipts ──────────────────────────────────────────

/// Policy atoms in the probe's tiny policy codebook. Atom 0 stays the null
/// default per the triangle canon; unknown atoms REFUSE (None), never clamp.
const ATOM_NAKED_ONLY: u8 = 1;
const ATOM_NAKED_PLUS_HIDDEN: u8 = 2;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Policy {
    NakedOnly,
    NakedPlusHidden,
}

fn policy_of(atom: u8) -> Option<Policy> {
    match atom {
        ATOM_NAKED_ONLY => Some(Policy::NakedOnly),
        ATOM_NAKED_PLUS_HIDDEN => Some(Policy::NakedPlusHidden),
        _ => None,
    }
}

/// One committed, WARRANTED assignment: the digit, and how many recorded
/// exclusion facts license it. Naked single: the 8 other digits are each
/// excluded by a peer (warrants = 8). Hidden single in a unit with k empty
/// cells: the k-1 other empty cells each exclude the digit (warrants = k-1,
/// ≥ 1 whenever the single is genuinely hidden).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct WarrantedAssignment {
    r: usize,
    c: usize,
    d: u8,
    warrants: usize,
}

/// One side-channel kernel observation on a CLONED candidate set — the
/// #997 receipt shape, plus the exact before/after lengths the coarse
/// signature throws away.
#[derive(Debug, Clone, PartialEq)]
struct KernelEvent {
    recipe_id: u8,
    fired: bool,
    delta_conf_sign: i8,
    len_before: usize,
    len_after: usize,
}

/// What the higher rung is allowed to see. Deliberately does NOT carry the
/// grid: the meta pass's object is the lower rung's reasoning process, not
/// the puzzle (enforced by `meta_decide`'s signature below).
#[derive(Debug, Clone, PartialEq)]
struct RungReceipt {
    policy_atom: u8,
    assignments: Vec<WarrantedAssignment>,
    fixed_point: bool,
    unresolved: usize,
    kernel_events: Vec<KernelEvent>,
}

/// Recipe ids for the #995/#997 collision trio (verified against the
/// catalogue by `kernel(id).meta().code` at runtime below).
const TCP: u8 = 5;
const TCF: u8 = 20;
const CUR: u8 = 26;

fn observe_kernel(id: u8, cands: &[u8]) -> KernelEvent {
    // The #997 mapping: equal 1/n confidence per remaining legal digit —
    // "these possibilities remain legal", NOT calibrated support scores.
    let n = cands.len().max(1) as f32;
    let mut ctx = ThoughtCtx::new(vec![1.0 / n; cands.len()]);
    ctx.sd = ((n - 1.0) / 8.0).clamp(0.0, 1.0);
    ctx.free_energy = ctx.sd;
    let len_before = ctx.candidates.len();
    let outcome = kernel(id)
        .expect("collision-trio ids are minted")
        .run_with(&mut ctx, MaturityPolicy::Any);
    KernelEvent {
        recipe_id: id,
        fired: outcome.fired,
        delta_conf_sign: match outcome.delta_conf {
            x if x > 0.0 => 1,
            x if x < 0.0 => -1,
            _ => 0,
        },
        len_before,
        len_after: ctx.candidates.len(),
    }
}

/// The lower rung: run `policy` to its fixed point, committing ONLY
/// warranted assignments (each oracle-checked), observing the kernel trio
/// per examined cell on cloned candidate sets. Returns the receipt and the
/// final grid (the grid stays with the lower rung; the receipt travels up).
fn run_lower_rung(policy_atom: u8, start: &Grid, solution: &Grid) -> (RungReceipt, Grid) {
    let policy = policy_of(policy_atom).expect("frozen lane must hold a known policy atom");
    let mut grid = *start;
    let mut assignments = Vec::new();
    let mut kernel_events = Vec::new();

    loop {
        // One warranted step per iteration (recompute after each commit).
        let mut step: Option<WarrantedAssignment> = None;

        'scan: for r in 0..9 {
            for c in 0..9 {
                if grid[r][c] != 0 {
                    continue;
                }
                let cands = candidates(&grid, r, c);
                assert!(!cands.is_empty(), "well-formed puzzle");
                // Side channel: the collision trio observes the REAL
                // candidate set — clones only, never written back (F15).
                for id in [TCP, TCF, CUR] {
                    kernel_events.push(observe_kernel(id, &cands));
                }
                if cands.len() == 1 {
                    step = Some(WarrantedAssignment {
                        r,
                        c,
                        d: cands[0],
                        warrants: 8, // the 8 excluded digits, each by a peer
                    });
                    break 'scan;
                }
            }
        }

        if step.is_none() && policy == Policy::NakedPlusHidden {
            step = find_hidden_single(&grid);
        }

        match step {
            Some(a) => {
                assert!(a.warrants >= 1, "F15: no commit without a warrant");
                assert_eq!(
                    solution[a.r][a.c], a.d,
                    "oracle: warranted assignment must match the unique solution"
                );
                grid[a.r][a.c] = a.d;
                assignments.push(a);
            }
            None => break,
        }
    }

    let unresolved = grid.iter().flatten().filter(|&&d| d == 0).count();
    (
        RungReceipt {
            policy_atom,
            assignments,
            fixed_point: true, // the loop above always runs to its fixed point
            unresolved,
            kernel_events,
        },
        grid,
    )
}

fn find_hidden_single(grid: &Grid) -> Option<WarrantedAssignment> {
    let units: Vec<Vec<(usize, usize)>> = (0..9)
        .map(|r| (0..9).map(|c| (r, c)).collect())
        .chain((0..9).map(|c| (0..9).map(|r| (r, c)).collect()))
        .chain((0..3).flat_map(|br| {
            (0..3).map(move |bc| {
                (0..3)
                    .flat_map(|dr| (0..3).map(move |dc| (br * 3 + dr, bc * 3 + dc)))
                    .collect()
            })
        }))
        .collect();
    for unit in &units {
        let empty: Vec<(usize, usize)> = unit
            .iter()
            .copied()
            .filter(|&(r, c)| grid[r][c] == 0)
            .collect();
        if empty.len() < 2 {
            continue; // a lone empty cell is a naked single, not hidden
        }
        for d in 1u8..=9 {
            if unit.iter().any(|&(r, c)| grid[r][c] == d) {
                continue;
            }
            let holders: Vec<(usize, usize)> = empty
                .iter()
                .copied()
                .filter(|&(r, c)| candidates(grid, r, c).contains(&d))
                .collect();
            if let [(r, c)] = holders[..] {
                // Every OTHER empty cell in the unit excludes d — each an
                // exclusion fact backed by a peer holding d.
                return Some(WarrantedAssignment {
                    r,
                    c,
                    d,
                    warrants: empty.len() - 1,
                });
            }
        }
    }
    None
}

/// Derive a fixture on which NakedOnly stalls but NakedPlusHidden still
/// makes warranted progress — deterministically, from the base puzzle, by
/// removing givens in scan order (forward or reverse) while the solution
/// stays unique. Self-falsifying: panics if the scan cannot produce one.
fn derive_stall_fixture(base: &Grid, reverse: bool) -> (Grid, Grid) {
    let mut grid = *base;
    let mut order: Vec<(usize, usize)> = (0..9)
        .flat_map(|r| (0..9).map(move |c| (r, c)))
        .filter(|&(r, c)| base[r][c] != 0)
        .collect();
    if reverse {
        order.reverse();
    }
    for (r, c) in order {
        let kept = grid[r][c];
        grid[r][c] = 0;
        if count_solutions(&grid, 2) != 1 {
            grid[r][c] = kept;
            continue;
        }
        let solution = solve_unique(&grid).expect("uniqueness just checked");
        let (naked, _) = run_lower_rung(ATOM_NAKED_ONLY, &grid, &solution);
        if naked.unresolved > 0 {
            let (hidden, _) = run_lower_rung(ATOM_NAKED_PLUS_HIDDEN, &grid, &solution);
            if hidden.assignments.len() > naked.assignments.len() {
                return (grid, solution);
            }
        }
    }
    panic!("derivation failed: no stall fixture found from this base/scan — the probe's premise is falsified");
}

// ── The higher rung ───────────────────────────────────────────────────────

/// Probe-local meta-decision vocabulary. NOT a minted type: the audit found
/// `InnerCouncil`'s Flow/Fanout/RungElevate speaks collapse breadth, not
/// lane choice, so forcing it here would be vocabulary theft; wiring the
/// council is the Kanban-hinge slice.
#[derive(Debug, Clone, PartialEq, Eq)]
enum MetaDecision {
    KeepFrozen,
    TryExplore,
    RecordLearned {
        atom: u8,
    },
    Promote {
        family: u8,
    },
    RefusePromotion,
    UnwarrantedCertainty {
        recipe_id: u8,
    },
    ObserverInsufficient {
        colliding: Vec<u8>,
        exact_separates: Vec<u8>,
    },
}

/// The single StyleFamily slot this probe exercises (ordinal 0 of 12).
const FAMILY: u8 = 0;

/// Residual free energy of a receipt: the unexplained fraction of the
/// board. This is the `f_majority`/`f_minority` fed into the SHIPPED
/// [`FreeEnergyComparison`] — the probe does not invent a comparison
/// operator; it supplies the two F values and lets `minority_wins()` rule.
fn residual_free_energy(receipt: &RungReceipt) -> f32 {
    receipt.unresolved as f32 / 81.0
}

/// Assess one lower-rung receipt. SIGNATURE IS THE FENCE: no `Grid`
/// parameter — the higher rung's object is the reasoning process only.
fn assess(receipt: &RungReceipt) -> MetaDecision {
    if receipt.unresolved == 0 || !receipt.fixed_point {
        MetaDecision::KeepFrozen
    } else {
        MetaDecision::TryExplore
    }
}

/// Assess the kernel side channel: coarse signatures vs exact transitions
/// over ONE unresolved (n≥3) observation triple. Returns the F8 verdict.
fn assess_observer(events: &[KernelEvent]) -> Option<MetaDecision> {
    // Find one examined cell where all three kernels saw the same n>=3 set.
    for w in events.chunks(3) {
        let [a, b, c] = w else { continue };
        if a.len_before < 3 || a.len_before != b.len_before || b.len_before != c.len_before {
            continue;
        }
        let coarse = |e: &KernelEvent| (e.fired, e.delta_conf_sign);
        if coarse(a) == coarse(b) && coarse(b) == coarse(c) {
            let exact = |e: &KernelEvent| (e.len_before, e.len_after);
            let mut separates = Vec::new();
            if exact(a) != exact(b) || exact(a) != exact(c) {
                for e in [a, b, c] {
                    if [a, b, c].iter().filter(|x| exact(x) == exact(e)).count() == 1 {
                        separates.push(e.recipe_id);
                    }
                }
            }
            return Some(MetaDecision::ObserverInsufficient {
                colliding: vec![a.recipe_id, b.recipe_id, c.recipe_id],
                exact_separates: separates,
            });
        }
    }
    None
}

// ── Gate runner ───────────────────────────────────────────────────────────

fn main() {
    println!("═══ PROBE-METACOGNITIVE-TRIANGLE-1 ═══\n");
    let mut gates: Vec<(&str, bool, String)> = Vec::new();

    // Verify the collision-trio ids against the catalogue, not by memory.
    for (id, code) in [(TCP, "TCP"), (TCF, "TCF"), (CUR, "CUR")] {
        assert_eq!(kernel(id).unwrap().meta().code, code);
    }

    // Fixtures. Base solves fully under NakedOnly (proven in #997); the two
    // stall fixtures are derived deterministically (forward + reverse scan).
    let base_solution = solve_unique(&BASE_PUZZLE).expect("base is unique");
    let (stall_a, stall_a_solution) = derive_stall_fixture(&BASE_PUZZLE, false);
    let (stall_b, stall_b_solution) = derive_stall_fixture(&BASE_PUZZLE, true);
    assert_ne!(stall_a, stall_b, "held-out fixture must differ from train");

    let run_loop = |log: &mut Vec<String>| -> (MailboxSoA<4>, Vec<MetaDecision>) {
        let mut mb = MailboxSoA::<4>::new(1, 0, 0.5);
        mb.set_populated(2);
        // Row 0 = the reasoning owner under test. Row 1 = bystander (F13).
        mb.set_style_atom(0, StyleLane::Frozen, FAMILY, ATOM_NAKED_ONLY);
        mb.set_style_lane(1, StyleLane::Frozen, [7u8; 12]);
        mb.set_style_lane(1, StyleLane::Learned, [8u8; 12]);
        mb.set_style_lane(1, StyleLane::Explore, [9u8; 12]);
        let mut decisions = Vec::new();

        // ── F1: Frozen earns warranted progress → leave it alone. ─────
        let frozen_atom = mb.style_lane_at(0, StyleLane::Frozen).unwrap()[FAMILY as usize];
        let (receipt, _) = run_lower_rung(frozen_atom, &BASE_PUZZLE, &base_solution);
        let d = assess(&receipt);
        log.push(format!("base: {d:?}"));
        decisions.push(d);

        // ── F2: Frozen stalls on the train fixture → TryExplore. ──────
        let (stalled, _) = run_lower_rung(frozen_atom, &stall_a, &stall_a_solution);
        let d = assess(&stalled);
        log.push(format!("stall_a under frozen: {d:?}"));
        decisions.push(d.clone());

        if d == MetaDecision::TryExplore {
            // The TryExplore verdict IS a split: majority pole = keep the
            // frozen commitment (incomplete), minority pole = the explore
            // hypothesis. Open the SHIPPED counterfactual lane: the −6
            // mantissa marks everything the Explore arm does as
            // hypothetical — never observed truth (counterfactual.rs's
            // invariant, enforced mechanically).
            let split = CouncilVerdict {
                hint: CollapseHint::Flow,
                confidence: 0.5,
                split: true,
            };
            let mut cf_edge = RawEdge::default();
            let deposited = deposit_counterfactual(&split, &mut cf_edge);
            assert!(deposited, "TryExplore split must deposit");
            assert_eq!(cf_edge.mantissa(), -6, "Counterfactual nibble = -6");
            log.push(
                "explore lane opened: counterfactual mantissa = -6 (never observed truth)".into(),
            );

            // ── F4 (can-it-stay-silent): degenerate explore = frozen. ──
            mb.set_style_atom(0, StyleLane::Explore, FAMILY, frozen_atom);
            let ex = mb.style_lane_at(0, StyleLane::Explore).unwrap()[FAMILY as usize];
            let (deg, _) = run_lower_rung(ex, &stall_a, &stall_a_solution);
            let cmp = FreeEnergyComparison {
                f_majority: residual_free_energy(&stalled),
                f_minority: residual_free_energy(&deg),
            };
            if !cmp.minority_wins() {
                log.push("degenerate explore: minority does not win, Learned NOT written".into());
            } else {
                log.push("degenerate explore unexpectedly improved".into());
            }
            let learned_before = mb.style_lane_at(0, StyleLane::Learned).unwrap();

            // ── F5: real explore's minority pole wins → RecordLearned. ──
            mb.set_style_atom(0, StyleLane::Explore, FAMILY, ATOM_NAKED_PLUS_HIDDEN);
            let ex = mb.style_lane_at(0, StyleLane::Explore).unwrap()[FAMILY as usize];
            let (explored, _) = run_lower_rung(ex, &stall_a, &stall_a_solution);
            let cmp = FreeEnergyComparison {
                f_majority: residual_free_energy(&stalled),
                f_minority: residual_free_energy(&explored),
            };
            if cmp.minority_wins() {
                mb.set_style_atom(0, StyleLane::Learned, FAMILY, ex);
                decisions.push(MetaDecision::RecordLearned { atom: ex });
                log.push(format!(
                    "explore minority wins (F {:.3} < {:.3}); Learned := {ex}",
                    cmp.f_minority, cmp.f_majority
                ));
            }
            assert_eq!(
                learned_before[FAMILY as usize], 0,
                "F4 held before F5 wrote"
            );

            // ── F6: held-out where the minority canNOT win → MajorityHolds. ──
            // (The base puzzle: NakedOnly already solves it fully.)
            let learned = mb.style_lane_at(0, StyleLane::Learned).unwrap()[FAMILY as usize];
            let (f_base, _) = run_lower_rung(frozen_atom, &BASE_PUZZLE, &base_solution);
            let (l_base, _) = run_lower_rung(learned, &BASE_PUZZLE, &base_solution);
            let cmp_base = FreeEnergyComparison {
                f_majority: residual_free_energy(&f_base),
                f_minority: residual_free_energy(&l_base),
            };
            let outcome_base = if cmp_base.minority_wins() {
                RevisionOutcome::Revised
            } else {
                RevisionOutcome::MajorityHolds
            };
            if outcome_base == RevisionOutcome::MajorityHolds {
                decisions.push(MetaDecision::RefusePromotion);
                log.push(
                    "held-out (base): RevisionOutcome::MajorityHolds — promotion refused".into(),
                );
            }

            // ── F7: held-out where the minority DOES win → Revised → promote. ──
            let (f_b, _) = run_lower_rung(frozen_atom, &stall_b, &stall_b_solution);
            let (l_b, _) = run_lower_rung(learned, &stall_b, &stall_b_solution);
            let cmp_b = FreeEnergyComparison {
                f_majority: residual_free_energy(&f_b),
                f_minority: residual_free_energy(&l_b),
            };
            let outcome_b = if cmp_b.minority_wins() {
                RevisionOutcome::Revised
            } else {
                RevisionOutcome::MajorityHolds
            };
            if outcome_b == RevisionOutcome::Revised {
                let promoted = mb.promote_family(0, FAMILY);
                decisions.push(MetaDecision::Promote { family: FAMILY });
                log.push(format!(
                    "held-out (stall_b): RevisionOutcome::Revised — promote_family = {promoted}"
                ));
                assert!(promoted, "promotion must report a real byte change");
                // revise_if_minority_wins's documented step 5: once the
                // revision is committed, clear the counterfactual mantissa
                // (Deduction = 0) so the nibble is not double-counted.
                cf_edge.set_inference_mantissa(0);
                assert_eq!(cf_edge.mantissa(), 0);
                log.push(
                    "counterfactual mantissa cleared to 0 after Revised (step-5 protocol)".into(),
                );
            }
        }

        // ── The read-side arrow, end to end: the NEXT run reads Frozen. ──
        let now_frozen = mb.style_lane_at(0, StyleLane::Frozen).unwrap()[FAMILY as usize];
        let (rerun, _) = run_lower_rung(now_frozen, &stall_a, &stall_a_solution);
        log.push(format!(
            "post-promotion frozen atom = {now_frozen}; rerun on stall_a leaves {} unresolved",
            rerun.unresolved
        ));

        // ── F3/F8/F15: the kernel side channel, assessed. ─────────────
        if let Some(v) = assess_observer(&stalled.kernel_events) {
            log.push(format!("observer verdict: {v:?}"));
            decisions.push(v);
        }
        // The TCF singleton event: manufactured certainty with zero warrants.
        if let Some(tcf) = stalled
            .kernel_events
            .iter()
            .find(|e| e.recipe_id == TCF && e.len_before >= 2 && e.len_after == 1)
        {
            decisions.push(MetaDecision::UnwarrantedCertainty {
                recipe_id: tcf.recipe_id,
            });
            log.push(format!(
                "TCF manufactured a singleton from n={} with 0 exclusion warrants — refused as truth",
                tcf.len_before
            ));
        }

        (mb, decisions)
    };

    let mut log1 = Vec::new();
    let (mb1, decisions1) = run_loop(&mut log1);
    for line in &log1 {
        println!("  {line}");
    }

    // ── Gate assembly ─────────────────────────────────────────────────
    gates.push((
        "F1 keep-frozen on warranted progress",
        decisions1.first() == Some(&MetaDecision::KeepFrozen),
        format!("{:?}", decisions1.first()),
    ));
    gates.push((
        "F2 fixed point + unresolved → TryExplore",
        decisions1.get(1) == Some(&MetaDecision::TryExplore),
        format!("{:?}", decisions1.get(1)),
    ));
    gates.push((
        "F4 degenerate explore stayed silent",
        log1.iter().any(|l| l.contains("Learned NOT written")),
        "no Learned write without improvement".into(),
    ));
    gates.push((
        "F5 warranted improvement → Learned recorded",
        decisions1.contains(&MetaDecision::RecordLearned {
            atom: ATOM_NAKED_PLUS_HIDDEN,
        }),
        "Learned := NakedPlusHidden".into(),
    ));
    gates.push((
        "F6 held-out non-reproduction → promotion refused",
        decisions1.contains(&MetaDecision::RefusePromotion),
        "base fixture: no improvement, frozen unchanged there".into(),
    ));
    gates.push((
        "F7 held-out reproduction → promote_family (first production-path call)",
        decisions1.contains(&MetaDecision::Promote { family: FAMILY })
            && mb1.style_lane_at(0, StyleLane::Frozen).unwrap()[FAMILY as usize]
                == ATOM_NAKED_PLUS_HIDDEN,
        "frozen[0] == NakedPlusHidden after promotion".into(),
    ));
    gates.push((
        "read-side arrow: next run reads the promoted Frozen lane",
        log1.iter()
            .any(|l| l.contains("rerun on stall_a leaves 0 unresolved")),
        "dispatch literally read StyleLane::Frozen".into(),
    ));
    gates.push((
        "counterfactual lane: Explore ran under -6 mantissa, cleared to 0 on Revised",
        log1.iter()
            .any(|l| l.contains("counterfactual mantissa = -6"))
            && log1
                .iter()
                .any(|l| l.contains("cleared to 0 after Revised")),
        "deposit_counterfactual + step-5 clear (shipped v2 surface)".into(),
    ));
    let observer_ok = decisions1.iter().any(|d| {
        matches!(d, MetaDecision::ObserverInsufficient { colliding, exact_separates }
            if colliding == &vec![TCP, TCF, CUR] && exact_separates == &vec![TCF])
    });
    gates.push((
        "F8 coarse collides, exact separates TCF; verdict = ObserverInsufficient, no delta_conf edit",
        observer_ok,
        "TCP/CUR identity vs TCF singleton on equal sets".into(),
    ));
    gates.push((
        "F3/F15 kernel singleton refused as truth; every commit carried ≥1 warrant",
        decisions1.contains(&MetaDecision::UnwarrantedCertainty { recipe_id: TCF }),
        "unresolved cell stayed unresolved; warrants asserted per commit".into(),
    ));
    gates.push((
        "F13 bystander row untouched",
        mb1.style_lane_at(1, StyleLane::Frozen).unwrap() == [7u8; 12]
            && mb1.style_lane_at(1, StyleLane::Learned).unwrap() == [8u8; 12]
            && mb1.style_lane_at(1, StyleLane::Explore).unwrap() == [9u8; 12],
        "row 1 lanes byte-identical".into(),
    ));

    // ── F14: determinism — full rerun, identical decision log. ────────
    let mut log2 = Vec::new();
    let (_, decisions2) = run_loop(&mut log2);
    gates.push((
        "F14 same receipts + same triangle → same meta-decisions",
        decisions1 == decisions2 && log1 == log2,
        format!("{} decisions, byte-identical logs", decisions1.len()),
    ));

    // ── Report ────────────────────────────────────────────────────────
    println!("\n═══ Gates ═══");
    let mut all = true;
    for (name, pass, detail) in &gates {
        all &= *pass;
        println!(
            "  [{}] {name} — {detail}",
            if *pass { "PASS" } else { "FAIL" }
        );
    }
    println!(
        "\nScope: F9 (reason-context equivalence) and F10/F11 (Revision→Kanban hinge, the \
         InnerCouncil 'designed, not wired' edge) are the NEXT slice. F12 holds by \
         construction: this probe never imports causal-edge; CE64 bits 59..63 untouched."
    );
    assert!(
        all,
        "PROBE-METACOGNITIVE-TRIANGLE-1: gate failure — see above"
    );
    println!("\nPROBE-METACOGNITIVE-TRIANGLE-1: ALL GATES GREEN");
}
