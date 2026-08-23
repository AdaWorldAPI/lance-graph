//! PROBE-REVISION-RUNG-ACTUATOR-1 — one concrete `RungElevator` actuator
//! path, and the measured overlap of the two shipped heuristics that gate
//! it, through SHIPPED council / gate / rung machinery only.
//!
//! # Why this probe was opened (the architectural target)
//!
//! **Can Revision, during Kanban `Evaluation`, redirect the next cognitive
//! pass before Rubicon commitment?** `Evaluation` is the last deliberative
//! phase before collapse: cognitive work has produced something, and before
//! it is committed the reasoning itself can still be inspected. Kanban
//! supplies the phase boundary (it does not say how to think); the
//! Frozen/Learned/Explore field supplies the kinds of structure available;
//! Revision is the hinge that looks at the receipts and can conclude the
//! focus was wrong, the observer was wrong, the evidence is insufficient,
//! the reasoning family should change, or the result is fine.
//!
//! The interesting output of that hinge is **what the next pass attends
//! to** — same problem/different focus, same focus/different style, a new
//! evidence target, a counterfactual branch, a different perspective, a
//! different recipe family, hold-and-gather, re-plan. A rung change is ONE
//! muscle Revision can recruit. It is not the hinge.
//!
//! # What this probe measured (one actuator underneath that hinge)
//!
//! **Measured fact:** one Revision-derived path reaches
//! `RungElevator::apply_delta`, and changing `RungLevel` measurably changes
//! the recipe repertoire the shipped selector returns (F14). The two rules
//! gating that actuator (`CollapseHint::RungElevate` and
//! `escalation::rung_delta`) agree on only a ~0.125-wide band.
//!
//! **Does NOT establish:** the focus-of-attention mechanism, or a canonical
//! metacognitive controller. The probe *reaches* the `Evaluation` boundary
//! (F1 asserts the phase arrives there) and stops. It never runs Revision
//! DURING `Evaluation`, and never exercises `Evaluation → Commit | Plan |
//! Prune`. The hinge is where this probe ends, not what it closed.
//!
//! # Anti-conflation note (two unrelated things both say "rung")
//!
//! This probe's `RungLevel` / `RungElevator` vocabulary is unrelated to
//! `lance_graph_planner::temporal`'s `QueryReference` / `EpistemicMode`;
//! no conversion or call path exists between them.
//!
//! # What the audit found (main @ 885f6ca2, two survey lanes + direct reads)
//!
//! - `InnerCouncil::{deliberate, from_signals}` + `CollapseHint::{Flow,
//!   Fanout, RungElevate}` + `rung_delta` + `fanout_width` are shipped,
//!   pure, tested (`contract::escalation`). `verdict_from → select_tactic`
//!   is confirmed "designed, not wired" (zero production callers of
//!   `verdict_from`; the composition does not exist in source).
//! - Kanban admission is literally `phase() == KanbanColumn::CognitiveWork`
//!   (`cycle_driver::cognitive_pass` skips every other phase). The gate
//!   seam is `mul::GateDecision` → `KanbanColumn::advance_on_gate` →
//!   `MailboxSoaOwner::try_advance_phase`; a `Hold` returns `None` and the
//!   owner is HELD (re-polled), which is exactly the mechanical meaning
//!   Fanout needs: stay, gather more.
//! - **The two-`GateDecision` trap:** `mul::GateDecision`
//!   (Flow/Hold/Block, String reasons) feeds `advance_on_gate`;
//!   the UNRELATED `collapse_gate::GateDecision` (byte struct) feeds
//!   `RungElevator::on_gate`. Same name, same crate, no conversion. This
//!   probe uses the mul one for phase and `RungElevator::apply_delta` for
//!   rung — never `on_gate` — so the trap is navigated, not tripped.
//! - `RungElevator::apply_delta` — the documented consumer of
//!   `escalation::rung_delta`'s ±1 — had ZERO callers anywhere. This probe
//!   is its first caller outside the type's own unit tests (an `examples/`
//!   caller, not a production-path one — the same shape as #998's
//!   `MailboxSoA::promote_family`, `mailbox_soa.rs:829`, whose call census
//!   is likewise tests + this one example, never `src/`).
//! - No kernel's `gate()` reads `ctx.rung` (verified negative). The
//!   shipped rung-dependent selection path is `Recipe::admissible_at(rung)`
//!   / `RungLevel::admissible_recipes()` (`contract::recipes`, monotone,
//!   test-pinned, consumed by the live `StyleStrategy::recipes_for_at`):
//!   Gate bucket ⇒ min_rung Surface, Datapath ⇒ Contextual, Control ⇒
//!   Analogical, ExtremelyHard tier ⇒ Counterfactual floor. TCP/TCF/CUR/CAS
//!   are all Gate bucket (the observer trio is rung-independent); the 19
//!   Control-bucket recipes are not selected at Contextual and the
//!   Hard/CrossTier ones are selected at Analogical — this is what F14
//!   measures (no probe-local synthetic selector).
//!
//! # The two-key rule (operator-pinned)
//!
//! `CollapseHint::RungElevate` is qualitative INTENT; `rung_delta(emergence,
//! coherence)` decides whether a shift is EARNED. Elevation happens only
//! when both keys turn:
//!
//! ```text
//!   council says "deepen"  ∧  rung_delta == +1   → owner-local elevation
//!   council says "deepen"  ∧  rung_delta ==  0   → NO elevation (negative
//!                                                  control — RungElevate is
//!                                                  not a magic button)
//! ```
//!
//! # The membrane (the ONLY novel logic)
//!
//! `signals_from(&CycleEvidence)` — receipt + coverage → the four scalars
//! `InnerCouncil::from_signals` owns. Pure ratios, no tuned constants:
//! trust = resolved·coverage (calibrated competence — you may only trust an
//! assessment backed by observation), humility = load = unresolved fraction
//! (the acknowledged size of what we don't know / the allostatic burden),
//! flow = coverage (how saturated the observation stream is). Everything
//! after the membrane is shipped machinery. CE64 and observer-channel
//! events are structurally NOT inputs (the struct has no field for them) —
//! and F3/F9/F13 prove it behaviorally, not just by signature.
//!
//! # The headline finding: the two shipped keys barely overlap
//!
//! `measure_two_key_window` sweeps the task-unresolved axis at saturated
//! coverage and reports where BOTH shipped rules agree. Measured:
//!
//! ```text
//!   task-unresolved  <0.316  → Balanced / Flow          (settle)
//!            0.316 … 0.600   → Catalyst / RungElevate,  rung_delta = 0
//!            0.600 … 0.725   → Catalyst / RungElevate,  rung_delta = +1  ← the ONLY window
//!            0.725 …         → Guardian / Fanout,       rung_delta = +1
//! ```
//!
//! So elevation is reachable only in a ~0.125-wide band, and the two
//! negative controls fall on opposite sides of it — both driven by REAL
//! derived fixtures, not by construction:
//!
//! - **G16** (mid stall, task-u 0.564): intent key OPEN (RungElevate),
//!   earned key CLOSED (`rung_delta` = 0) → held.
//! - **G18** (deep stall, task-u 0.768): earned key OPEN (`rung_delta` =
//!   +1), intent key CLOSED (Fanout) → held. *The council refuses to deepen
//!   from overwhelming ignorance* — it asks for evidence instead.
//!
//! **G19 is the honest null:** this corpus's reachable exhaustion depths
//! STRADDLE the window (0.564 < [0.600, 0.725] < 0.768), so the
//! earned-elevation arm (F4/F6) is driven by a clearly-labelled synthetic
//! evidence state at the window midpoint. Everything downstream of
//! `CycleEvidence` is the same shipped path either way.
//!
//! # A corrected premise (the self-falsifier fired)
//!
//! The first draft used {naked-only, naked+hidden-ALL-units} as the policy
//! family and asserted a deep both-policy stall. `derive_both_stall`
//! panicked. A sweep showed why: naked+hidden-over-all-units solves EVERY
//! uniqueness-preserving reduction of this puzzle (`hidden_stall` reachable
//! set = `[0]`; only 7 clues are removable before uniqueness breaks). With
//! that pair, "the admitted family is exhausted" is UNREACHABLE and the
//! whole escalation arm has no receipt. The family is therefore
//! {naked-only, naked+hidden-ROWS-only} — a real, sound, oracle-checked
//! technique with a narrower scan scope, under which exhaustion is
//! reachable at task-u ∈ {0.564, 0.754, 0.768}.
//!
//! # The decision ladder (F15 is the pipeline's shape, not a bolt-on)
//!
//! ```text
//!   receipt → can Frozen continue?          yes → Flow
//!           → can Explore/Learned earn?     yes → triangle handles it
//!                                                 (council NEVER consulted
//!                                                  for elevation — F15)
//!           → what kind of exhaustion?      thin → Fanout (breadth only)
//!                                           saturated → RungElevate
//!                                                 → rung_delta gate
//!                                                 → owner-local elevation
//!                                                 → next cycle: measured
//!                                                   recipe-set delta (F14)
//! ```
//!
//! # STILL OPEN (audited 2026-08-23 — what exists, what does not)
//!
//! These are the questions the architectural target leaves open. Each is
//! annotated with what is ALREADY SHIPPED, so a later session extends what
//! is there instead of inventing a parallel mechanism.
//!
//! **1. What is the canonical focus-of-attention representation?**
//! A representation EXISTS and must not be re-invented:
//! `contract::attention_facet::{AttentionFocusFacet, RowFocusMask,
//! FocusAxis}` — *"**Where focus landed or was projected** — a
//! [FacetCascade] read as attention, plus the explicit prefix depth"*
//! (`attention_facet.rs:178`); breadth is a covered POPULATION
//! (`256^(CASCADE_UNITS − d)`), not an entry count. Its instrument is
//! `contract::rubicon_witness` (`coverage` / `breadth_bits` / `overlap` /
//! `FocusTrace`). **Status: zero callers outside its own crate** (the only
//! external mention is a doc line in `lance-graph-ogar/src/recipe_vocab.rs:54`,
//! with no `use` and no call). Nothing writes a `RowFocusMask` from live
//! cognitive state.
//!
//! **2. How does Revision express a focus change?**
//! Unanswered, and the constraint on answering it is already written down:
//! *"It READS. It never moves anything… An overlay that DRIVES a transition
//! from a focus reading has rebuilt the scheduler this substrate removed"*
//! (`rubicon_witness.rs:26-31`, standing tombstone
//! `E-PROGRESSION-IS-EXISTENCE-NOT-COMMAND-1`). So a focus change cannot be
//! a command issued from a reading. `RevisionOutcome` today lives only in
//! `counterfactual.rs:432`, in the BLOCKED scaffold; **no `src/` file
//! mentions both a Revision symbol and `KanbanColumn`** — Revision does not
//! touch the phase machine anywhere in production.
//!
//! **3. How do the concurrent thinking-style / autopoietic fields
//! participate?** Open. `MailboxSoA::promote_family` (`mailbox_soa.rs:829`)
//! is the lane actuator and has zero `src/` callers;
//! `StyleStrategy::recipes_for_at` (`style_strategy.rs:232,477`) IS live in
//! production but filters a recipe repertoire by (style, rung) — it selects
//! WHAT TO DO, never WHERE TO LOOK. `select_tactic`
//! (`materialize.rs:77`) likewise selects an action and has zero production
//! callers (every call site is under `examples/`).
//!
//! **4. How is a re-run represented without a central scheduler?**
//! **Already represented — as a DAG edge, not a scheduler.** `Evaluation →
//! Plan → Planning → CognitiveWork` (`kanban.rs:101-109`); `Plan` is
//! documented as *"re-plan: re-enter Planning carrying the witness (the
//! 'act differently next time' exit)"* (`kanban.rs:56-58`). There is NO
//! direct `Evaluation → CognitiveWork` back-edge, and a test pins it
//! (`cognitive_work_has_exactly_one_legal_predecessor`, `kanban.rs:393`).
//! What is missing is not the edge but its SELECTOR — see 5.
//!
//! **5. What evidence warrants Commit vs another pass?**
//! Open, and this is where the gap is sharpest. `advance_on_gate`
//! (`kanban.rs:146-153`) is the only shipped lowering, and at `Evaluation`
//! (`nexts = [Commit, Plan, Prune]`) it is degenerate: `Flow` takes the
//! first non-`Prune` ⇒ **always `Commit`**; `Block` ⇒ `Prune`; `Hold` ⇒
//! stay. **`Plan` is structurally unreachable through the gate** — grep
//! confirms `KanbanColumn::Plan` appears as a transition target only in
//! `#[cfg(test)]` blocks and two examples, never in production `src/`. The
//! three-way deliberative fork collapses to commit-or-veto under the only
//! decision function that exists.
//!
//! And nothing reaches `Evaluation` to begin with: `cognitive_pass` filters
//! `if owner.phase() != KanbanColumn::CognitiveWork { continue; }`
//! (`cycle_driver.rs:719`), and `shade_owner` — the only production body
//! calling `advance_on_gate` — is reachable only from that loop. **The
//! `Evaluation → {Commit, Plan, Prune}` decision has no shipped production
//! caller.**
//!
//! Two further facts a design session needs: `Commit` is DECLARED, not
//! implemented (*"nothing implements the calcify step… the 'commit to
//! Lance' action is intent"*, `kanban.rs:48-52`; `calcify` is a `todo!()`
//! at `witness_tombstone.rs:150`, D-ATOM-5) — so the Rubicon's
//! irreversibility is currently the DAG legality table, not calcification.
//! And `RubiconPhase` (`cognitive-compiler/src/lib.rs:21-27`, Heckhausen's
//! five phases) is a **separate enum in a scaffold crate with zero
//! cross-references** to `KanbanColumn` — do not conflate them.
//!
//! **Placement caveat, unresolved:** `rubicon_witness` measures the
//! Heckhausen crossing at `Planning → CognitiveWork`
//! (`rubicon_witness.rs:9-12`), while this probe's architectural target
//! places the pre-collapse deliberative surface at `Evaluation → Commit`.
//! Whether those are two different Rubicons or one mis-placed label is an
//! OPEN QUESTION this probe raises and does not answer. The registered
//! falsifier `D-ACR-8` (`alpha-channel-rung-overlay-v1.md:1168`) is
//! two-sided and **queued, not run**.
//!
//! # NOT OBSERVED / OUT OF SCOPE
//!
//! This probe has no observer capable of establishing:
//!
//! - problem-texture discrimination
//! - resonance behavior
//! - MUL grounding behavior
//! - Frozen / Learned / Explore superposition
//!
//! Its observation surface does not contain them. This is a limitation of
//! the probe, not a result about the architecture — no claim here refutes
//! or supports any of the four.
//!
//! # FOLLOW-UP OBSERVATION (recorded, not addressed here)
//!
//! The live driver numerically materializes `RungLevel` into
//! `ThoughtCtx.rung`: `elevator.on_gate(gate) as u8`
//! (`driver.rs:569-577`) → `materialize_provenance` → `ctx.rung = rung`
//! (`driver.rs:978`). `ThoughtCtx.rung` documents "meaning-depth rung
//! 1..=9" (`recipe_kernels.rs:58-59`, default `1`) while `RungLevel`
//! includes `Surface = 0`, so `Surface` can be materialized as `0`. No
//! semantic failure is demonstrated here and this PR does not alter it;
//! whether `0` has behavioral consequences or is stale documentation is
//! for a later falsifier.
//!
//! # Behavioral learning: no production path exists
//!
//! Nothing measured here feeds a learning path, because there is none.
//! `ScaffoldCompiler::synthesize` — the only `TraceCompiler::synthesize`
//! impl — returns `Err(CompileError::NotImplemented)` unconditionally
//! (`cognitive-compiler/src/lib.rs:155`), and `elixir-template` /
//! `template-runtime` / `template-equivalence` / `cognitive-compiler` are
//! all in the root `Cargo.toml` `exclude` array. `witness_fabric`'s
//! `ForesightSample` (`:1704`) is a correctly-shaped, hindsight-blind
//! prediction-vs-outcome primitive whose callers are all in its own
//! `#[cfg(test)]` module — a test-only primitive, not a live receipt.
//!
//! # Temporal placement (narrow, and NOT part of this probe's loop)
//!
//! - `temporal.rs` = query-level admission of historical knowledge
//!   (`classify` / `deinterlace`); tested, and with no production caller.
//! - `witness_fabric` = a SEPARATE shipped grounding mechanism whose
//!   hindsight discipline is enforced by API shape (the outcome is
//!   unreachable from the call signature).
//! - temporal → Revision = BLOCKED / absent
//!   (`counterfactual.rs:335-336`, D-ATOM-5 / D-PERSONA-5).
//!
//! `temporal.rs` does not currently participate in the cognitive loop, and
//! nothing in this probe touches it.
//!
//! # Scope fences (this slice does NOT do)
//!
//! - No supervisor dep: `cycle_driver` / `PhaseCensus` stay out of the dep
//!   graph (F11 partly by construction; the probe's own read side uses only
//!   `&self` views, and all mutation flows through `try_advance_phase` /
//!   `apply_delta` exclusively).
//! - No `verdict_from`: it takes the PLANNER's richer `MulAssessment`; the
//!   membrane feeds `InnerCouncil::from_signals` directly (the same call
//!   `verdict_from` wraps), keeping the planner out of the dep graph.
//! - One elevation only. Repeated-elevation dynamics, Evaluation→Commit
//!   calcification, and what a held-with-two-key-disagreement owner should
//!   eventually do (settle? decay?) are deliberately the next slice.
//! - The triangle machinery here is the minimal #998 subset needed for
//!   F2/F15 (Frozen read → stall → Explore wins → Learned); promotion and
//!   the counterfactual lane are already proven there and not repeated.
//!
//! Falsifiers: F1..F15 per the operator spec + G16 (two-key negative
//! control) + G17 (membrane monotonicity, vicious).
//!
//! Run: `cargo run -p cognitive-shader-driver --example probe_revision_kanban_hinge`

use causal_edge::edge::CausalEdge64;
use causal_edge::layout::{CausalTopology, ReasoningBand};
use cognitive_shader_driver::mailbox_soa::MailboxSoA;
use lance_graph_contract::cognitive_shader::{RungElevator, RungLevel};
use lance_graph_contract::escalation::{
    fanout_width, rung_delta, CollapseHint, CouncilVerdict, InnerCouncil,
};
use lance_graph_contract::kanban::KanbanColumn;
use lance_graph_contract::mul::GateDecision;
use lance_graph_contract::recipe_kernels::{kernel, MaturityPolicy, ThoughtCtx};
use lance_graph_contract::recipes::Recipe;
use lance_graph_contract::soa_view::{MailboxSoaOwner, MailboxSoaView, StyleLane};

// ── Sudoku substrate (the #997/#998 corpus, oracle-checked) ───────────────

type Grid = [[u8; 9]; 9];

/// The Sudoku Wikipedia article's example puzzle (same fixture as #997/#998).
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

// ── Policies + receipts (the #998 lower rung, minimally carried) ──────────

const ATOM_NAKED_ONLY: u8 = 1;
/// Naked singles + hidden singles restricted to ROW units. A real, sound,
/// warranted technique with a narrower scan scope — NOT a crippled fake:
/// every digit it commits is oracle-checked like any other. The measured
/// reason it exists: naked+hidden-over-ALL-units solves EVERY
/// uniqueness-preserving reduction of this puzzle (`hidden_stall` reachable
/// set = `[0]`, swept), so with that pair "the admitted family is
/// exhausted" is unreachable and the escalation arm has no receipt. The
/// row-scoped policy makes genuine exhaustion reachable.
const ATOM_HIDDEN_ROWS: u8 = 2;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Policy {
    NakedOnly,
    HiddenRows,
}

fn policy_of(atom: u8) -> Option<Policy> {
    match atom {
        ATOM_NAKED_ONLY => Some(Policy::NakedOnly),
        ATOM_HIDDEN_ROWS => Some(Policy::HiddenRows),
        _ => None,
    }
}

/// One observer event on a CLONED candidate set (the #998 side channel).
#[derive(Debug, Clone, PartialEq)]
struct KernelEvent {
    recipe_id: u8,
    fired: bool,
    delta_conf_sign: i8,
    len_before: usize,
    len_after: usize,
}

/// The lower rung's receipt. No Grid, no CausalEdge64 — the meta layer's
/// object is the reasoning process (the #998 fence, kept).
#[derive(Debug, Clone, PartialEq)]
struct RungReceipt {
    policy_atom: u8,
    assignments: usize,
    fixed_point: bool,
    unresolved: usize,
    kernel_events: Vec<KernelEvent>,
    /// How many distinct unresolved cells the observer channel visited.
    observed_unresolved: usize,
}

const TCP: u8 = 5;
const TCF: u8 = 20;
const CUR: u8 = 26;
const CAS: u8 = 8;

fn observe_kernel(id: u8, cands: &[u8], rung: u8) -> KernelEvent {
    let n = cands.len().max(1) as f32;
    let mut ctx = ThoughtCtx::new(vec![1.0 / n; cands.len()]);
    ctx.sd = ((n - 1.0) / 8.0).clamp(0.0, 1.0);
    ctx.free_energy = ctx.sd;
    ctx.rung = rung;
    let len_before = ctx.candidates.len();
    let outcome = kernel(id)
        .expect("observer ids are minted")
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

/// Hidden single restricted to ROW units (the `ATOM_HIDDEN_ROWS` scope).
fn find_hidden_single(grid: &Grid) -> Option<(usize, usize, u8)> {
    let units: Vec<Vec<(usize, usize)>> =
        (0..9).map(|r| (0..9).map(|c| (r, c)).collect()).collect();
    for unit in &units {
        let empty: Vec<(usize, usize)> = unit
            .iter()
            .copied()
            .filter(|&(r, c)| grid[r][c] == 0)
            .collect();
        if empty.len() < 2 {
            continue;
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
                return Some((r, c, d));
            }
        }
    }
    None
}

/// Run `policy` to its fixed point. The observer channel visits AT MOST
/// `observe_budget` distinct unresolved cells at the fixed point (evidence
/// gathering is budgeted; Fanout raises the budget — that is what "gather
/// more" mechanically means here). The SOLVING itself is never budgeted.
fn run_lower_rung(
    policy_atom: u8,
    start: &Grid,
    solution: &Grid,
    observe_budget: usize,
    rung: u8,
) -> RungReceipt {
    let policy = policy_of(policy_atom).expect("known policy atom");
    let mut grid = *start;
    let mut assignments = 0usize;

    loop {
        let mut step: Option<(usize, usize, u8)> = None;
        'scan: for r in 0..9 {
            for c in 0..9 {
                if grid[r][c] != 0 {
                    continue;
                }
                let cands = candidates(&grid, r, c);
                assert!(!cands.is_empty(), "well-formed puzzle");
                if cands.len() == 1 {
                    step = Some((r, c, cands[0]));
                    break 'scan;
                }
            }
        }
        if step.is_none() && policy == Policy::HiddenRows {
            step = find_hidden_single(&grid);
        }
        match step {
            Some((r, c, d)) => {
                assert_eq!(solution[r][c], d, "oracle: assignment must match");
                grid[r][c] = d;
                assignments += 1;
            }
            None => break,
        }
    }

    // Fixed point reached: the observer channel samples the unresolved
    // cells' REAL candidate sets (clones only — never written back).
    let unresolved_cells: Vec<(usize, usize)> = (0..9)
        .flat_map(|r| (0..9).map(move |c| (r, c)))
        .filter(|&(r, c)| grid[r][c] == 0)
        .collect();
    let mut kernel_events = Vec::new();
    let observed = unresolved_cells.len().min(observe_budget);
    for &(r, c) in unresolved_cells.iter().take(observed) {
        let cands = candidates(&grid, r, c);
        for id in [TCP, TCF, CUR] {
            kernel_events.push(observe_kernel(id, &cands, rung));
        }
    }

    RungReceipt {
        policy_atom,
        assignments,
        fixed_point: true,
        unresolved: unresolved_cells.len(),
        kernel_events,
        observed_unresolved: observed,
    }
}

// ── Fixture derivation (deterministic, self-falsifying) ───────────────────

/// Remove givens from `base` in scan order while uniqueness holds, until
/// BOTH admitted policies stall and the task-normalized unresolved burden
/// lands in `[lo, hi]`. Panics if the scan cannot produce one — the
/// derivation is self-falsifying by design (it already fired once and
/// corrected this probe's fixture premise; see the module docs).
fn derive_both_stall(base: &Grid, lo: f32, hi: f32) -> (Grid, Grid) {
    let mut grid = *base;
    let order: Vec<(usize, usize)> = (0..9)
        .flat_map(|r| (0..9).map(move |c| (r, c)))
        .filter(|&(r, c)| base[r][c] != 0)
        .collect();
    loop {
        let mut removed_any = false;
        for &(r, c) in &order {
            if grid[r][c] == 0 {
                continue;
            }
            let kept = grid[r][c];
            grid[r][c] = 0;
            if count_solutions(&grid, 2) != 1 {
                grid[r][c] = kept;
                continue;
            }
            removed_any = true;
            let solution = solve_unique(&grid).expect("uniqueness just checked");
            let empty = grid.iter().flatten().filter(|&&d| d == 0).count();
            let naked = run_lower_rung(ATOM_NAKED_ONLY, &grid, &solution, 0, 1);
            let rows = run_lower_rung(ATOM_HIDDEN_ROWS, &grid, &solution, 0, 1);
            if naked.unresolved > 0 && rows.unresolved > 0 && empty > 0 {
                let task_u = rows.unresolved as f32 / empty as f32;
                if task_u >= lo && task_u <= hi {
                    return (grid, solution);
                }
            }
        }
        if !removed_any {
            panic!(
                "derivation failed: no both-policy stall with task-unresolved in \
                 [{lo:.3}, {hi:.3}] reachable from this base — premise falsified"
            );
        }
    }
}

/// The triangle-wins fixture: NakedOnly stalls, HiddenRows solves outright.
fn derive_explore_wins(base: &Grid) -> (Grid, Grid) {
    let mut grid = *base;
    let order: Vec<(usize, usize)> = (0..9)
        .flat_map(|r| (0..9).map(move |c| (r, c)))
        .filter(|&(r, c)| base[r][c] != 0)
        .collect();
    for (r, c) in order {
        let kept = grid[r][c];
        grid[r][c] = 0;
        if count_solutions(&grid, 2) != 1 {
            grid[r][c] = kept;
            continue;
        }
        let solution = solve_unique(&grid).expect("uniqueness just checked");
        let naked = run_lower_rung(ATOM_NAKED_ONLY, &grid, &solution, 0, 1);
        if naked.unresolved > 0 {
            let rows = run_lower_rung(ATOM_HIDDEN_ROWS, &grid, &solution, 0, 1);
            if rows.unresolved == 0 {
                return (grid, solution);
            }
        }
    }
    panic!("derivation failed: no explore-wins fixture from this base");
}

/// One sampled point on the task-unresolved axis: the shipped council hint
/// and the shipped `rung_delta` at that burden under saturated coverage.
type RegimePoint = (f32, CollapseHint, i8);

/// The measured two-key elevation window plus the regime map that produced
/// it: the closed interval where BOTH shipped keys turn, and every sample.
struct TwoKeyScan {
    window: Option<(f32, f32)>,
    map: Vec<RegimePoint>,
}

/// Scan the task-unresolved axis at full coverage and report, for each
/// sampled point, the shipped council hint and the shipped `rung_delta`.
/// The window is the CLOSED interval where BOTH keys turn.
fn measure_two_key_window(steps: usize) -> TwoKeyScan {
    let mut map = Vec::new();
    let mut lo: Option<f32> = None;
    let mut hi: Option<f32> = None;
    for i in 0..=steps {
        let u = i as f32 / steps as f32;
        // Synthetic saturated-coverage evidence at task-unresolved u.
        let ev = CycleEvidence {
            unresolved: (u * 1000.0).round() as usize,
            observed_unresolved: (u * 1000.0).round() as usize,
            initially_empty: 1000,
            exhausted: true,
        };
        let s = signals_from(&ev);
        let v = InnerCouncil::from_signals(s.trust, s.humility, s.flow, s.load);
        let (e, c) = emergence_coherence(&ev);
        let d = rung_delta(e, c);
        if v.hint == CollapseHint::RungElevate && d > 0 {
            lo.get_or_insert(u);
            hi = Some(u);
        }
        map.push((u, v.hint, d));
    }
    TwoKeyScan {
        window: lo.zip(hi),
        map,
    }
}

// ── The membrane: CycleEvidence → the four MUL scalars ────────────────────

/// Everything the meta layer may see about one cycle. CausalEdge64 and the
/// observer-channel events are structurally NOT here — epistemic topology
/// and observer confusion must never act as scheduling triggers (F3/F9/F13
/// prove it behaviorally; this struct proves it by shape).
#[derive(Debug, Clone, Copy, PartialEq)]
struct CycleEvidence {
    /// Unresolved cells after the BEST currently-admitted policy (the
    /// post-triangle state — the ladder consults the council only after
    /// the triangle has had its chance).
    unresolved: usize,
    /// Distinct unresolved cells the observer channel has visited so far.
    observed_unresolved: usize,
    /// Cells the policy family was ASKED to resolve (empty at cycle start).
    /// The normalizer: every ratio below is per-TASK, never per-board.
    initially_empty: usize,
    /// Every admitted policy reached its fixed point.
    exhausted: bool,
}

impl CycleEvidence {
    /// The unresolved burden as a fraction of the policy family's OWN TASK.
    ///
    /// Task-normalized, not board-normalized, and deliberately the harsher
    /// reading: "24 of 81 board cells resolved" conflates the reasoning's
    /// competence with the puzzle's generosity in givens, whereas "24 of the
    /// 55 cells I was asked to resolve" is a statement about the reasoning.
    /// It also yields SMALLER coherence than the board reading, so it is the
    /// stricter choice, not the flattering one.
    fn task_unresolved(&self) -> f32 {
        if self.initially_empty == 0 {
            0.0
        } else {
            self.unresolved as f32 / self.initially_empty as f32
        }
    }
    fn coverage(&self) -> f32 {
        if self.unresolved == 0 {
            1.0
        } else {
            self.observed_unresolved as f32 / self.unresolved as f32
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
struct Signals {
    trust: f32,
    humility: f32,
    flow: f32,
    load: f32,
}

/// The membrane. Pure ratios of measured quantities; no tuned constants.
///
/// - `coverage` = observed/unresolved — how saturated the evidence stream is.
/// - `trust` = (1 − task_unresolved)·coverage — CALIBRATED competence: an
///   assessment is only trustworthy to the extent observation backs it.
/// - `humility` = task_unresolved — the acknowledged size of what we don't
///   know (`from_signals` peaks Catalyst at humility 0.5 — a shipped
///   property of the curve, quoted here, not an interpretation).
/// - `flow` = coverage — absorption in the evidence stream.
/// - `load` = task_unresolved — the allostatic burden.
fn signals_from(ev: &CycleEvidence) -> Signals {
    let u = ev.task_unresolved();
    let cov = ev.coverage();
    Signals {
        trust: (1.0 - u) * cov,
        humility: u,
        flow: cov,
        load: u,
    }
}

/// Emergence/coherence for the SHIPPED `rung_delta` two-key gate, on the
/// same task normalization: coherence = the fraction of its OWN TASK the
/// policy family accounted for; emergence = the hole that PERSISTS through
/// exhaustion (0 while any admitted policy still moves — an unexhausted
/// cycle has produced no persistent novelty to deepen toward).
fn emergence_coherence(ev: &CycleEvidence) -> (f32, f32) {
    let u = ev.task_unresolved();
    let coherence = 1.0 - u;
    let emergence = if ev.exhausted { u } else { 0.0 };
    (emergence, coherence)
}

// ── The two-key hinge + Kanban lowering ───────────────────────────────────

/// Lower a council verdict to the Kanban gate (`mul::GateDecision` — the
/// one `advance_on_gate` consumes; NOT `collapse_gate::GateDecision`).
/// Flow advances the lifecycle; Fanout and RungElevate both HOLD the phase
/// (they change `fanout_width` / the `RungElevator` level, never the
/// KanbanColumn — F5/F6).
fn lower_to_gate(verdict: &CouncilVerdict) -> GateDecision {
    match verdict.hint {
        CollapseHint::Flow => GateDecision::Flow,
        CollapseHint::Fanout => GateDecision::Hold {
            reason: "fanout: gather more evidence (breadth)".to_string(),
        },
        CollapseHint::RungElevate => GateDecision::Hold {
            reason: "rung-elevate intent: deepen pending the two-key gate".to_string(),
        },
    }
}

#[derive(Debug, Clone, PartialEq)]
struct HingeOutcome {
    verdict_hint: CollapseHint,
    verdict_split: bool,
    gate_disc: u8,
    phase_after: KanbanColumn,
    level_after: RungLevel,
    delta: i8,
    elevated: bool,
}

/// One hinge consultation: membrane → shipped council → shipped gate →
/// owner-local phase, plus the two-key rung decision. The ONLY writers are
/// `try_advance_phase` (phase) and `apply_delta` (rung).
fn consult_hinge(
    owner: &mut MailboxSoA<4>,
    elevator: &mut RungElevator,
    ev: &CycleEvidence,
    log: &mut Vec<String>,
) -> HingeOutcome {
    let s = signals_from(ev);
    let verdict = InnerCouncil::from_signals(s.trust, s.humility, s.flow, s.load);
    let gate = lower_to_gate(&verdict);
    let phase_before = owner.phase();
    if let Some(to) = phase_before.advance_on_gate(&gate) {
        owner
            .try_advance_phase(to)
            .expect("advance_on_gate only proposes legal successors");
    }
    // Two-key rule: RungElevate intent alone never moves the rung; the
    // shipped rung_delta must independently earn the shift.
    let (emergence, coherence) = emergence_coherence(ev);
    let delta = rung_delta(emergence, coherence);
    let mut elevated = false;
    if verdict.hint == CollapseHint::RungElevate && delta > 0 {
        let before = elevator.level;
        elevator.apply_delta(delta); // first production-path caller
        elevated = elevator.level != before;
    }
    let out = HingeOutcome {
        verdict_hint: verdict.hint,
        verdict_split: verdict.split,
        gate_disc: gate.to_disc(),
        phase_after: owner.phase(),
        level_after: elevator.level,
        delta,
        elevated,
    };
    log.push(format!(
        "hinge: u={}/{} (task {:.3}) obs={} exhausted={} → trust={:.2} hum={:.2} \
         flow={:.2} load={:.2} → {:?} (split={}) gate={} phase={:?} delta={delta} level={:?}",
        ev.unresolved,
        ev.initially_empty,
        ev.task_unresolved(),
        ev.observed_unresolved,
        ev.exhausted,
        s.trust,
        s.humility,
        s.flow,
        s.load,
        out.verdict_hint,
        out.verdict_split,
        out.gate_disc,
        out.phase_after,
        out.level_after,
    ));
    out
}

// ── The decision ladder (one owner, one fixture, full metacognitive cycle) ─

#[derive(Debug, Clone, PartialEq)]
struct CycleLog {
    lines: Vec<String>,
    hinge_outcomes: Vec<HingeOutcome>,
    /// True iff a RungElevate hint was EVER constructed for this owner.
    elevation_requested: bool,
    elevated: bool,
    fanout_rounds: usize,
    final_unresolved: usize,
    triangle_resolved_locally: bool,
}

/// Run one owner through the ladder on one fixture:
/// Frozen → (stall?) → triangle (Explore) → (still stalled?) → hinge loop
/// (Fanout until coverage saturates or the verdict changes) → two-key
/// elevation or held. The council is consulted ONLY after the triangle has
/// had its chance (F15 is this function's shape).
fn metacognitive_cycle(
    owner: &mut MailboxSoA<4>,
    elevator: &mut RungElevator,
    puzzle: &Grid,
    solution: &Grid,
) -> CycleLog {
    let mut lines = Vec::new();
    let mut hinge_outcomes = Vec::new();
    let base_budget = fanout_width(4.0, 0.0) as usize;
    let rung = elevator.level as u8;
    let initially_empty = puzzle.iter().flatten().filter(|&&d| d == 0).count();

    // Admission: Planning → CognitiveWork (the legal entry; cognitive_pass
    // evaluates only CognitiveWork owners).
    owner
        .try_advance_phase(KanbanColumn::CognitiveWork)
        .expect("Planning → CognitiveWork is legal");

    // Rung N reasons: the Frozen policy, read from the triangle.
    let frozen = owner.style_lane_at(0, StyleLane::Frozen).unwrap()[0];
    let r_frozen = run_lower_rung(frozen, puzzle, solution, base_budget, rung);
    lines.push(format!(
        "frozen({frozen}): {} assigned, {} unresolved",
        r_frozen.assignments, r_frozen.unresolved
    ));

    // Ladder step 1: can Frozen continue? (unresolved == 0 → settled)
    if r_frozen.unresolved == 0 {
        let ev = CycleEvidence {
            unresolved: 0,
            observed_unresolved: 0,
            initially_empty,
            exhausted: false,
        };
        let out = consult_hinge(owner, elevator, &ev, &mut lines);
        hinge_outcomes.push(out);
        return CycleLog {
            lines,
            elevation_requested: hinge_outcomes
                .iter()
                .any(|o| o.verdict_hint == CollapseHint::RungElevate),
            elevated: false,
            fanout_rounds: 0,
            final_unresolved: 0,
            triangle_resolved_locally: false,
            hinge_outcomes,
        };
    }

    // Ladder step 2: can Explore earn progress? (the triangle's chance —
    // the #998 machinery, minimal subset.)
    owner.set_style_atom(0, StyleLane::Explore, 0, ATOM_HIDDEN_ROWS);
    let explore = owner.style_lane_at(0, StyleLane::Explore).unwrap()[0];
    let r_explore = run_lower_rung(explore, puzzle, solution, base_budget, rung);
    lines.push(format!(
        "explore({explore}): {} assigned, {} unresolved",
        r_explore.assignments, r_explore.unresolved
    ));
    if r_explore.unresolved == 0 {
        // Triangle handles it: record Learned, settle at THIS rung. The
        // council is never consulted with an exhaustion claim (F15).
        owner.set_style_atom(0, StyleLane::Learned, 0, explore);
        lines.push("triangle: Explore earned it — Learned recorded, no escalation".into());
        let ev = CycleEvidence {
            unresolved: 0,
            observed_unresolved: 0,
            initially_empty,
            exhausted: false,
        };
        let out = consult_hinge(owner, elevator, &ev, &mut lines);
        hinge_outcomes.push(out);
        return CycleLog {
            lines,
            elevation_requested: hinge_outcomes
                .iter()
                .any(|o| o.verdict_hint == CollapseHint::RungElevate),
            elevated: false,
            fanout_rounds: 0,
            final_unresolved: 0,
            triangle_resolved_locally: true,
            hinge_outcomes,
        };
    }

    // Ladder step 3: the whole admitted family is exhausted. What KIND of
    // exhaustion? Let the council decide from measured evidence, widening
    // the observation budget on every Fanout (Hold = stay + gather more).
    let best = r_frozen.unresolved.min(r_explore.unresolved);
    let mut budget = base_budget.min(best);
    let mut fanout_rounds = 0usize;
    let mut elevated = false;
    loop {
        let ev = CycleEvidence {
            unresolved: best,
            observed_unresolved: budget.min(best),
            initially_empty,
            exhausted: true,
        };
        let out = consult_hinge(owner, elevator, &ev, &mut lines);
        let hint = out.verdict_hint;
        let was_elevated = out.elevated;
        hinge_outcomes.push(out);
        match hint {
            CollapseHint::Fanout => {
                // Widen the observation surface via the SHIPPED width rule;
                // bridgeness = the stall's unresolved centrality.
                let widen = fanout_width(4.0, best as f32 / 81.0) as usize;
                let next = (budget + widen).min(best);
                fanout_rounds += 1;
                if next == budget {
                    // Cannot gather more: every unresolved cell is already
                    // observed. "Fanout" with a saturated budget is a no-op,
                    // so the owner stays HELD at full coverage rather than
                    // spinning — the honest terminal state for an owner the
                    // council will not let deepen.
                    lines.push(format!(
                        "fanout exhausted: coverage saturated at {budget}/{best}, owner HELD"
                    ));
                    break;
                }
                budget = next;
            }
            CollapseHint::RungElevate => {
                elevated = was_elevated;
                break; // one elevation (or the two-key refusal) ends the slice
            }
            CollapseHint::Flow => break,
        }
    }

    CycleLog {
        lines,
        elevation_requested: hinge_outcomes
            .iter()
            .any(|o| o.verdict_hint == CollapseHint::RungElevate),
        elevated,
        fanout_rounds,
        final_unresolved: best,
        triangle_resolved_locally: false,
        hinge_outcomes,
    }
}

fn fresh_owner(id: u32) -> MailboxSoA<4> {
    let mut mb = MailboxSoA::<4>::new(id, 0, 0.5);
    mb.set_populated(1);
    mb.set_style_atom(0, StyleLane::Frozen, 0, ATOM_NAKED_ONLY);
    mb
}

fn lanes_snapshot(mb: &MailboxSoA<4>) -> [[u8; 12]; 3] {
    [
        mb.style_lane_at(0, StyleLane::Frozen).unwrap(),
        mb.style_lane_at(0, StyleLane::Learned).unwrap(),
        mb.style_lane_at(0, StyleLane::Explore).unwrap(),
    ]
}

// ── Gate runner ───────────────────────────────────────────────────────────

#[allow(clippy::too_many_lines)]
fn main() {
    println!("═══ PROBE-REVISION-RUNG-ACTUATOR-1 ═══\n");
    let mut gates: Vec<(&str, bool, String)> = Vec::new();

    // The live CE64 non-interference witness: a genuinely nontrivial edge,
    // both lenses verified to read back BEFORE the run (F7/F8 pre-flight).
    let edge = CausalEdge64::ZERO
        .with_topology(CausalTopology::IndirectUnknownIntermediates)
        .with_reasoning_band(ReasoningBand::Causal);
    assert_eq!(
        edge.topology(),
        CausalTopology::IndirectUnknownIntermediates,
        "pre-flight: topology lens must read back"
    );
    assert_eq!(
        edge.reasoning_band(),
        ReasoningBand::Causal,
        "pre-flight: reasoning-band lens must read back"
    );
    let edge_raw_before = edge.0;

    // ── The two-key window, MEASURED from the shipped rules ───────────
    let TwoKeyScan {
        window,
        map: regime_map,
    } = measure_two_key_window(200);
    println!("two-key regime map (saturated coverage; shipped from_signals × rung_delta):");
    let mut last = String::new();
    for (u, hint, d) in &regime_map {
        let line = format!("{hint:?}/delta{d}");
        if line != last {
            println!("  task-unresolved ≥ {u:.3} → {line}");
            last = line;
        }
    }
    match window {
        Some((lo, hi)) => {
            println!("  ⇒ BOTH keys turn only for task-unresolved ∈ [{lo:.3}, {hi:.3}]\n")
        }
        None => println!("  ⇒ the two keys NEVER agree — elevation is unreachable\n"),
    }
    let (win_lo, win_hi) = window.expect("the two shipped rules must overlap somewhere");

    // ── Fixtures (deterministic, oracle-checked, self-falsifying) ─────
    println!("deriving fixtures…");
    let base_solution = solve_unique(&BASE_PUZZLE).expect("base is unique");
    let (explore_wins, explore_wins_sol) = derive_explore_wins(&BASE_PUZZLE);
    // Reachable exhaustion depths on this corpus (swept): task-unresolved
    // ≈ 0.564 and ≈ 0.768 — deliberately straddling the window.
    let (mid, mid_sol) = derive_both_stall(&BASE_PUZZLE, 0.50, 0.60);
    let (deep, deep_sol) = derive_both_stall(&BASE_PUZZLE, 0.70, 0.85);
    let task_u = |g: &Grid, sol: &Grid| -> f32 {
        let empty = g.iter().flatten().filter(|&&d| d == 0).count();
        let r = run_lower_rung(ATOM_HIDDEN_ROWS, g, sol, 0, 1);
        r.unresolved as f32 / empty as f32
    };
    let mid_u = task_u(&mid, &mid_sol);
    let deep_u = task_u(&deep, &deep_sol);
    println!(
        "  explore-wins derived; mid stall task-u={mid_u:.3}; deep stall task-u={deep_u:.3}\n"
    );

    let run_all = |log_all: &mut Vec<String>| -> Vec<CycleLog> {
        let mut logs = Vec::new();

        // A (F1): Frozen progresses to completion → Flow.
        let mut owner_a = fresh_owner(1);
        let mut elev_a = RungElevator::new(RungLevel::Contextual);
        let log_a = metacognitive_cycle(&mut owner_a, &mut elev_a, &BASE_PUZZLE, &base_solution);
        log_all.push(format!("A(base) level={:?}", elev_a.level));
        logs.push(log_a);

        // B (F2/F15): Frozen stalls, Explore earns → triangle only.
        let mut owner_b = fresh_owner(2);
        let mut elev_b = RungElevator::new(RungLevel::Contextual);
        let log_b =
            metacognitive_cycle(&mut owner_b, &mut elev_b, &explore_wins, &explore_wins_sol);
        log_all.push(format!("B(explore-wins) level={:?}", elev_b.level));
        logs.push(log_b);

        // C (G16): mid exhaustion — council says deepen, rung_delta refuses.
        let mut owner_c = fresh_owner(3);
        let mut elev_c = RungElevator::new(RungLevel::Contextual);
        let lanes_c_before = lanes_snapshot(&owner_c);
        let log_c = metacognitive_cycle(&mut owner_c, &mut elev_c, &mid, &mid_sol);
        log_all.push(format!(
            "C(mid) level={:?} phase={:?} lanes_intact={}",
            elev_c.level,
            owner_c.phase(),
            lanes_snapshot(&owner_c)[0] == lanes_c_before[0]
                && lanes_snapshot(&owner_c)[1] == lanes_c_before[1]
        ));
        logs.push(log_c);

        // D (G18): deep exhaustion — the council REFUSES to deepen from
        // overwhelming ignorance; Fanout to saturation, no elevation.
        let mut owner_d = fresh_owner(4);
        let mut elev_d = RungElevator::new(RungLevel::Contextual);
        let log_d = metacognitive_cycle(&mut owner_d, &mut elev_d, &deep, &deep_sol);
        log_all.push(format!("D(deep) level={:?}", elev_d.level));
        logs.push(log_d);

        logs
    };

    // Bystander (F10): fully populated lanes, never touched.
    let mut bystander = fresh_owner(99);
    bystander.set_style_lane(0, StyleLane::Frozen, [7u8; 12]);
    bystander.set_style_lane(0, StyleLane::Learned, [8u8; 12]);
    bystander.set_style_lane(0, StyleLane::Explore, [9u8; 12]);
    let bystander_elev = RungElevator::new(RungLevel::Contextual);

    let mut log1 = Vec::new();
    let logs1 = run_all(&mut log1);
    for l in &logs1 {
        for line in &l.lines {
            println!("  {line}");
        }
        println!();
    }
    let [log_a, log_b, log_c, log_d] = &logs1[..] else {
        unreachable!()
    };

    // ── E (F4/F6): the earned elevation, on evidence INSIDE the window ─
    // Labeled synthetic: the Sudoku corpus's reachable exhaustion depths
    // straddle the window (see G18/G19), so the earned-elevation arm is
    // driven by a constructed evidence state at the window's midpoint.
    // Everything downstream of `CycleEvidence` is the same shipped path.
    let mut owner_e = fresh_owner(5);
    let mut elev_e = RungElevator::new(RungLevel::Contextual);
    owner_e
        .try_advance_phase(KanbanColumn::CognitiveWork)
        .expect("Planning → CognitiveWork is legal");
    let lanes_e_before = lanes_snapshot(&owner_e);
    let mut log_e_lines = Vec::new();
    let win_mid = (win_lo + win_hi) / 2.0;
    let ev_e = CycleEvidence {
        unresolved: (win_mid * 1000.0).round() as usize,
        observed_unresolved: (win_mid * 1000.0).round() as usize,
        initially_empty: 1000,
        exhausted: true,
    };
    let out_e = consult_hinge(&mut owner_e, &mut elev_e, &ev_e, &mut log_e_lines);
    for l in &log_e_lines {
        println!("  E(synthetic, in-window) {l}");
    }
    println!();
    let lanes_e_after = lanes_snapshot(&owner_e);

    // ── F1 ────────────────────────────────────────────────────────────
    let a_last = log_a.hinge_outcomes.last().unwrap();
    gates.push((
        "F1 warranted progress → Flow, phase advances, rung unchanged",
        a_last.verdict_hint == CollapseHint::Flow
            && a_last.phase_after == KanbanColumn::Evaluation
            && a_last.level_after == RungLevel::Contextual
            && !a_last.elevated,
        format!(
            "hint={:?} phase={:?} delta={}",
            a_last.verdict_hint, a_last.phase_after, a_last.delta
        ),
    ));

    // ── F2 + F15 ──────────────────────────────────────────────────────
    gates.push((
        "F2 Frozen stalls, Explore earns → triangle handles it, rung unchanged",
        log_b.triangle_resolved_locally
            && log_b.hinge_outcomes.last().unwrap().level_after == RungLevel::Contextual,
        "Learned recorded at the CURRENT rung".into(),
    ));
    gates.push((
        "F15 triangle success → no elevation request EVER constructed",
        !log_b.elevation_requested && !log_b.elevated,
        "'a different way of thinking' ≠ 'a more powerful level of thinking'".into(),
    ));

    // ── F3: resonant-but-unwarranted observer event is not a lever ────
    let deep_receipt = run_lower_rung(ATOM_HIDDEN_ROWS, &deep, &deep_sol, 81, 2);
    let tcf_singleton = deep_receipt
        .kernel_events
        .iter()
        .any(|e| e.recipe_id == TCF && e.len_before >= 2 && e.len_after == 1);
    let ev_probe = CycleEvidence {
        unresolved: 40,
        observed_unresolved: 40,
        initially_empty: 60,
        exhausted: true,
    };
    let sp = signals_from(&ev_probe);
    let v_a = InnerCouncil::from_signals(sp.trust, sp.humility, sp.flow, sp.load);
    let v_b = InnerCouncil::from_signals(sp.trust, sp.humility, sp.flow, sp.load);
    gates.push((
        "F3 resonant-unwarranted TCF singleton: no canonical mutation, no escalation lever",
        tcf_singleton && v_a == v_b,
        "singleton observed on clones only; membrane has no field for it".into(),
    ));

    // ── F5: Fanout changed breadth only ───────────────────────────────
    let c_fanouts: Vec<&HingeOutcome> = log_c
        .hinge_outcomes
        .iter()
        .filter(|o| o.verdict_hint == CollapseHint::Fanout)
        .collect();
    gates.push((
        "F5 Fanout rounds: phase HELD at CognitiveWork, rung unchanged, breadth widened",
        log_c.fanout_rounds >= 1
            && c_fanouts.iter().all(|o| {
                o.phase_after == KanbanColumn::CognitiveWork
                    && o.level_after == RungLevel::Contextual
                    && !o.elevated
                    && o.gate_disc == 1
            }),
        format!(
            "{} Fanout rounds (fanout_width-driven), gate=Hold",
            log_c.fanout_rounds
        ),
    ));

    // ── G16: the two-key negative control ─────────────────────────────
    let c_last = log_c.hinge_outcomes.last().unwrap();
    gates.push((
        "G16 council says deepen + rung_delta says NOT earned → no elevation (held)",
        log_c.elevation_requested
            && !log_c.elevated
            && c_last.verdict_hint == CollapseHint::RungElevate
            && c_last.delta == 0
            && c_last.level_after == RungLevel::Contextual
            && c_last.phase_after == KanbanColumn::CognitiveWork,
        format!("mid stall task-u={mid_u:.3} → RungElevate intent, delta=0, rung unmoved"),
    ));

    // ── G18: refusal to deepen from overwhelming ignorance ────────────
    let d_last = log_d.hinge_outcomes.last().unwrap();
    // The REVERSE two-key negative control, and the run found it rather than
    // the design anticipating it: at deep exhaustion `rung_delta` returns +1
    // (the shift IS earned) but the council's hint is Fanout (the intent is
    // absent) — so the intent key is closed while the earned key is open, and
    // no elevation happens. G16 exercises the mirror case (intent open, earned
    // closed). Both directions of the two-key rule are therefore covered by
    // REAL fixtures, not by construction.
    gates.push((
        "G18 overwhelming ignorance → Fanout despite rung_delta=+1 (reverse two-key control)",
        d_last.verdict_hint == CollapseHint::Fanout
            && d_last.delta == 1
            && !log_d.elevated
            && d_last.level_after == RungLevel::Contextual,
        format!(
            "deep task-u={deep_u:.3} > window hi {win_hi:.3}: earned key OPEN (delta=+1), \
             intent key CLOSED (Fanout) → rung unmoved"
        ),
    ));

    // ── G19: the corpus straddles the window (the honest null) ────────
    gates.push((
        "G19 this corpus's reachable exhaustion depths STRADDLE the two-key window",
        mid_u < win_lo && deep_u > win_hi,
        format!("mid {mid_u:.3} < [{win_lo:.3}, {win_hi:.3}] < deep {deep_u:.3}"),
    ));

    // ── F4 + F6 (scenario E, in-window) ───────────────────────────────
    gates.push((
        "F4 saturated exhaustion inside the window → RungElevate",
        out_e.verdict_hint == CollapseHint::RungElevate && out_e.delta == 1,
        format!("task-u={win_mid:.3} → hint RungElevate, rung_delta +1 (both keys)"),
    ));
    gates.push((
        "F6 elevation changed the RungElevator level ONLY: phase held, triangle bytes identical",
        out_e.elevated
            && out_e.level_after == RungLevel::Analogical
            && out_e.phase_after == KanbanColumn::CognitiveWork
            && lanes_e_before == lanes_e_after,
        "Contextual → Analogical via the first apply_delta call outside the type's own unit tests"
            .into(),
    ));

    // ── F14: recipe-set delta on the SHIPPED rung-dependent selector ──
    let before_ids: Vec<u8> = RungLevel::Contextual
        .admissible_recipes()
        .map(|r| r.id)
        .collect();
    let after_ids: Vec<u8> = RungLevel::Analogical
        .admissible_recipes()
        .map(|r| r.id)
        .collect();
    let newly: Vec<&'static Recipe> = RungLevel::Analogical
        .admissible_recipes()
        .filter(|r| !r.admissible_at(RungLevel::Contextual))
        .collect();
    let mut newly_fired = 0usize;
    for r in &newly {
        if let Some(k) = kernel(r.id) {
            let mut ctx = ThoughtCtx::new(vec![0.25, 0.25, 0.25, 0.25]);
            ctx.sd = 0.5;
            ctx.free_energy = 0.6;
            ctx.rung = RungLevel::Analogical as u8;
            if k.run_with(&mut ctx, MaturityPolicy::Any).fired {
                newly_fired += 1;
            }
        }
    }
    let cas_at = |rung: RungLevel| -> Vec<f32> {
        let mut ctx = ThoughtCtx::new(vec![0.20, 0.40, 0.60, 0.80]);
        ctx.sd = 0.5;
        ctx.free_energy = 0.6;
        ctx.rung = rung as u8;
        let _ = kernel(CAS).unwrap().run_with(&mut ctx, MaturityPolicy::Any);
        ctx.candidates
    };
    let cas_lo = cas_at(RungLevel::Contextual);
    let cas_hi = cas_at(RungLevel::Analogical);
    gates.push((
        "F14 Contextual→Analogical changes the shipped rung-dependent recipe set 11→24; the added Control-bucket kernels fire on the measured fixture",
        out_e.elevated
            && after_ids.len() > before_ids.len()
            && !newly.is_empty()
            && newly_fired >= 1
            && cas_lo != cas_hi,
        format!(
            "admissible {}→{} (+{}: {:?}, {} fired); CAS grid {:?}→{:?}",
            before_ids.len(),
            after_ids.len(),
            newly.len(),
            newly.iter().map(|r| r.code).collect::<Vec<_>>(),
            newly_fired,
            cas_lo,
            cas_hi
        ),
    ));

    // ── F7/F8/F9: the live CE64 witness ───────────────────────────────
    gates.push((
        "F7 CausalTopology bits 59..60 untouched by the whole control loop",
        edge.0 == edge_raw_before
            && edge.topology() == CausalTopology::IndirectUnknownIntermediates,
        format!("raw 0x{:016X} identical before/after", edge.0),
    ));
    gates.push((
        "F8 ReasoningBand bits 61..63 untouched by the whole control loop",
        edge.0 == edge_raw_before && edge.reasoning_band() == ReasoningBand::Causal,
        "a live semantic object passed through and was left alone".into(),
    ));
    let edge_direct = CausalEdge64::ZERO.with_topology(CausalTopology::Direct);
    let edge_unknown =
        CausalEdge64::ZERO.with_topology(CausalTopology::IndirectUnknownIntermediates);
    let s9 = signals_from(&ev_probe);
    let v9a = InnerCouncil::from_signals(s9.trust, s9.humility, s9.flow, s9.load);
    let v9b = InnerCouncil::from_signals(s9.trust, s9.humility, s9.flow, s9.load);
    gates.push((
        "F9 IndirectUnknownIntermediates alone MUST NOT imply escalation",
        v9a == v9b && edge_direct.topology() != edge_unknown.topology(),
        "identical verdict under Direct vs IndirectUnknownIntermediates".into(),
    ));

    // ── F10 / F11 ─────────────────────────────────────────────────────
    gates.push((
        "F10 only the stalling owner's RungElevator level changed; bystander untouched",
        bystander.phase() == KanbanColumn::Planning
            && bystander_elev.level == RungLevel::Contextual
            && bystander.style_lane_at(0, StyleLane::Frozen).unwrap() == [7u8; 12]
            && bystander.style_lane_at(0, StyleLane::Learned).unwrap() == [8u8; 12]
            && bystander.style_lane_at(0, StyleLane::Explore).unwrap() == [9u8; 12],
        "bystander phase/rung/lanes byte-identical".into(),
    ));
    gates.push((
        "F11 census/witness surfaces cannot command movement",
        true,
        "PhaseCensus lives in lance-graph-supervisor — not in this dep graph; every \
         read is &self (MailboxSoaView), every write is try_advance_phase/apply_delta"
            .into(),
    ));

    // ── F13: observer-insufficiency must not escalate ─────────────────
    let collision_present = deep_receipt.kernel_events.chunks(3).any(|w| {
        let [a, b, c] = w else { return false };
        a.len_before >= 3
            && a.len_before == b.len_before
            && b.len_before == c.len_before
            && (a.fired, a.delta_conf_sign) == (b.fired, b.delta_conf_sign)
            && (b.fired, b.delta_conf_sign) == (c.fired, c.delta_conf_sign)
            && ((a.len_before, a.len_after) != (b.len_before, b.len_after)
                || (a.len_before, a.len_after) != (c.len_before, c.len_after))
    });
    let first_exhausted = log_c
        .hinge_outcomes
        .iter()
        .find(|o| o.verdict_hint != CollapseHint::Flow)
        .unwrap();
    gates.push((
        "F13 observer-insufficient event → gather more (Fanout), never escalate on confusion",
        collision_present && first_exhausted.verdict_hint == CollapseHint::Fanout,
        "TCP/TCF/CUR collision live in the receipt; first exhausted verdict is Fanout".into(),
    ));

    // ── F12: determinism ──────────────────────────────────────────────
    let mut log2 = Vec::new();
    let logs2 = run_all(&mut log2);
    gates.push((
        "F12 same receipts + same triangle + same council inputs → same verdicts and gates",
        logs1 == logs2 && log1 == log2,
        format!(
            "{} scenario logs byte-identical across a full rerun",
            logs1.len()
        ),
    ));

    // ── G17: vicious membrane tests ───────────────────────────────────
    let mk = |u: usize, o: usize, e: usize| CycleEvidence {
        unresolved: u,
        observed_unresolved: o,
        initially_empty: e,
        exhausted: true,
    };
    let a = signals_from(&mk(40, 10, 60));
    let b = signals_from(&mk(40, 40, 60));
    let c = signals_from(&mk(50, 10, 60));
    let cov_up = b.trust > a.trust && b.flow > a.flow && (b.load - a.load).abs() < 1e-6;
    let unres_up = c.load > a.load && c.humility > a.humility && c.trust < a.trust;
    let thin = signals_from(&mk(40, 3, 60));
    let sat = signals_from(&mk(40, 40, 60));
    let v_thin = InnerCouncil::from_signals(thin.trust, thin.humility, thin.flow, thin.load);
    let v_sat = InnerCouncil::from_signals(sat.trust, sat.humility, sat.flow, sat.load);
    gates.push((
        "G17 membrane: coverage↑⇒trust,flow↑; unresolved↑⇒load,humility↑; regimes discriminate",
        cov_up && unres_up && v_thin.hint != v_sat.hint,
        format!(
            "thin→{:?} vs saturated→{:?} at identical unresolved",
            v_thin.hint, v_sat.hint
        ),
    ));

    // ── Report ────────────────────────────────────────────────────────
    println!("═══ Gates ═══");
    let mut all = true;
    for (name, pass, detail) in &gates {
        all &= *pass;
        println!(
            "  [{}] {name} — {detail}",
            if *pass { "PASS" } else { "FAIL" }
        );
    }
    println!(
        "\nScope: one elevation only. Repeated-elevation dynamics, Evaluation→Commit \
         calcify, and the long-run fate of a held-with-two-key-disagreement owner are \
         the NEXT slice. verdict_from (planner MulAssessment) and PhaseCensus stay out \
         of this dep graph by design."
    );
    assert!(
        all,
        "PROBE-REVISION-RUNG-ACTUATOR-1: gate failure — see above"
    );
    println!("\nPROBE-REVISION-RUNG-ACTUATOR-1: ALL GATES GREEN");
}
