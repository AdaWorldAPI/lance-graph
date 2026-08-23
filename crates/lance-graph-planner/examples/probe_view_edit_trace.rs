//! `PROBE-WARRANTED-VIEW-TRACE-1` — make a cognitive trajectory reconstructible
//! BEFORE learning it. The seam one step short of the learner.
//!
//! # The three vicious invariants
//!
//! ```text
//!   RECONSTRUCTION  replay(A, [e1..en]) == final, AND at every prefix
//!   GROUNDING       each ei NAMES evidence available BEFORE ei
//!   ANTI-HINDSIGHT  no future outcome can manufacture a warrant
//! ```
//!
//! # Why GROUNDING needed a second pass
//!
//! The first draft's warrant carried `visible_before` / `rungs_before` — a
//! DESCRIPTION of the situation, which every possible edit satisfies. That is
//! the house anti-pattern: *"a guard that fires on everything carries exactly
//! as much information as one that never fires"* (`CLAUDE.md`
//! §falsifiability). A warrant must instead NAME evidence — `Evidence::
//! BeliefsAtRung { rung, count }` / `RowsBoundAt { locus, count }` — objects
//! drawn from the sealed state, and **G2 proves the channel can say NO**: an
//! off-field band (`RungBand{50,60}`, above the R6 ceiling) and an unbound
//! locus both come back UNGROUNDED. **G3** re-counts every named number
//! against the sealed state, so a warrant cannot pass by inventing plausible
//! figures.
//!
//! This is the difference between acquiring reasoning skill and acquiring
//! superstition: without G2, a later learner could compress a recurrent bad
//! habit exactly as efficiently as a good one.
//!
//! # What this adds over the single-edit particle
//!
//! ```text
//!   View A --e1--> View B --e2--> View C --e3--> View D
//!            w1            w2            w3
//! ```
//!
//! One edit proved a transformation can be typed and inverted. A TRAJECTORY
//! must additionally be: replayable exactly; replayable at every PREFIX (not
//! merely end-to-end); invertible where its edits are; carrying the warrant
//! that justified each step; and — the load-bearing one — **free of outcome
//! leakage backward**.
//!
//! # The outcome-leakage law, enforced structurally, not by convention
//!
//! `warrant_at` takes `&[ViewEdit]` — a PREFIX — and nothing else. It cannot
//! see later edits, the final view, or any outcome, because those are not
//! parameters. This is `witness_fabric`'s shipped discipline transplanted:
//! *"a run set computed as of revision v passes `upto = v + 1` and cannot
//! observe anything later. Retrospective judgement of a past state must not
//! be able to see what came after it, or it is hindsight wearing an audit's
//! clothes"* (`witness_fabric.rs:1476-1479`). Same argument, same shape: the
//! future is not reachable from the call, so no future edit can quietly
//! introduce backward leakage without changing the signature and failing
//! review.
//!
//! T4 checks the law behaviourally too — every recorded warrant must be
//! reproducible from its prefix alone.
//!
//! # Scope — what a green run does NOT mean
//!
//! No learning, no compression, no promotion, no recurrence detection. A
//! trace is an OBJECT that a later BPE-style learner could operate over; this
//! probe builds the object and stops. `BehaviorTrace` is probe-local and is
//! NOT proposed as a production type. F-REVISION-FOCUS-1 remains ABSENT —
//! nothing in production emits a `ViewEdit`, so the trace here is authored by
//! the probe, not harvested from a running cognition.

use lance_graph_contract::canonical_node::{EdgeBlock, NodeGuid, NodeRow};
use lance_graph_contract::causal_witness::{CausalWitnessFacet, Locus};
use lance_graph_contract::witness_fabric::WitnessLens;
use lance_graph_planner::nars::belief::{BeliefArena, CStmt, Copula, Stamp};
use lance_graph_planner::nars::truth::TruthValue;

const CHAIN_NODES: u16 = 64;
const BAND_RUNGS: [u32; 5] = [1, 2, 3, 4, 6];
const ROWS_PER_BAND: usize = 4;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Selector {
    BoundAt(Locus),
    RungBand { lo: u32, hi: u32 },
}

struct Ctx<'a> {
    lens: WitnessLens<'a>,
    rung_of_row: &'a [u32],
}

impl Selector {
    fn admits(self, pos: usize, ctx: &Ctx<'_>) -> bool {
        match self {
            Selector::BoundAt(l) => ctx.lens.at(pos).is_some_and(|f| f.is_bound(l)),
            Selector::RungBand { lo, hi } => ctx
                .rung_of_row
                .get(pos)
                .is_some_and(|r| *r >= lo && *r <= hi),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Default)]
struct ViewPlan {
    selectors: Vec<Selector>,
}

impl ViewPlan {
    fn of(s: &[Selector]) -> Self {
        Self {
            selectors: s.to_vec(),
        }
    }
    fn lower<'c, 'a>(&'c self, ctx: &'c Ctx<'a>) -> impl Fn(usize) -> bool + use<'c, 'a> {
        move |pos| self.selectors.iter().all(|s| s.admits(pos, ctx))
    }
    fn visible(&self, ctx: &Ctx<'_>) -> Vec<usize> {
        let f = self.lower(ctx);
        (0..ctx.lens.len()).filter(|p| f(*p)).collect()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ViewEdit {
    Push(Selector),
    RemoveAt(usize),
}

impl ViewEdit {
    fn apply(self, p: &ViewPlan) -> ViewPlan {
        let mut n = p.clone();
        match self {
            ViewEdit::Push(s) => n.selectors.push(s),
            ViewEdit::RemoveAt(i) => {
                if i < n.selectors.len() {
                    n.selectors.remove(i);
                }
            }
        }
        n
    }
    /// The inverse, when one exists. `Push` is always invertible (drop the
    /// tail). `RemoveAt` is invertible only against the plan it acted on —
    /// hence the `before` parameter, and `None` when the index was a no-op.
    fn inverse(self, before: &ViewPlan) -> Option<ViewEdit> {
        match self {
            ViewEdit::Push(_) => Some(ViewEdit::RemoveAt(before.selectors.len())),
            ViewEdit::RemoveAt(i) => before.selectors.get(i).map(|s| ViewEdit::Push(*s)),
        }
    }
}

/// A concrete piece of evidence drawn from the sealed state — an OBJECT the
/// edit can point at, never a description of the situation.
#[derive(Debug, Clone, PartialEq, Eq)]
enum Evidence {
    /// The arena actually holds `count` beliefs at this rung.
    BeliefsAtRung { rung: u32, count: usize },
    /// `count` rows in the population are actually bound at this locus.
    RowsBoundAt { locus: Locus, count: usize },
}

/// What EARNED one edit, computed from the PREFIX only.
///
/// **`support` is the whole point.** An earlier draft carried only
/// `visible_before` / `rungs_before` — a description of the situation, which
/// EVERY edit satisfies. A channel that fires on everything carries exactly as
/// much information as one that never fires (`CLAUDE.md` §falsifiability). So
/// the warrant must NAME evidence, and an edit that can name none is
/// `is_grounded() == false` — which G2 proves is reachable.
#[derive(Debug, Clone, PartialEq, Eq)]
struct Warrant {
    step: usize,
    visible_before: usize,
    /// Evidence in the sealed state that supports THIS edit. Empty = ungrounded.
    support: Vec<Evidence>,
}

impl Warrant {
    fn is_grounded(&self) -> bool {
        !self.support.is_empty()
    }
}

/// **The hindsight gate.** `prefix` is `edits[..k]`; `edit` is the one being
/// justified. There is no parameter through which a LATER edit, the final
/// view, or any outcome could enter — so no future edit can quietly introduce
/// backward leakage without changing this signature and failing review.
fn warrant_at(
    step: usize,
    initial: &ViewPlan,
    prefix: &[ViewEdit],
    edit: ViewEdit,
    arena: &BeliefArena,
    ctx: &Ctx<'_>,
) -> Warrant {
    let mut plan = initial.clone();
    for e in prefix {
        plan = e.apply(&plan);
    }
    let visible_before = plan.visible(ctx).len();

    // What in the sealed state SUPPORTS this particular edit?
    let support = match edit {
        ViewEdit::Push(Selector::RungBand { lo, hi }) => (lo..=hi)
            .filter_map(|rung| {
                let count = arena.entries().iter().filter(|b| b.rung == rung).count();
                (count > 0).then_some(Evidence::BeliefsAtRung { rung, count })
            })
            .collect(),
        ViewEdit::Push(Selector::BoundAt(locus)) => {
            let count = (0..ctx.lens.len())
                .filter(|p| ctx.lens.at(*p).is_some_and(|f| f.is_bound(locus)))
                .count();
            if count > 0 {
                vec![Evidence::RowsBoundAt { locus, count }]
            } else {
                vec![]
            }
        }
        // Removing a selector is warranted by what the CURRENT plan still
        // shows: the rungs on screen right now are the evidence that the
        // narrowing being dropped was ever doing work.
        ViewEdit::RemoveAt(i) => match plan.selectors.get(i) {
            Some(Selector::RungBand { lo, hi }) => (*lo..=*hi)
                .filter_map(|rung| {
                    let count = arena.entries().iter().filter(|b| b.rung == rung).count();
                    (count > 0).then_some(Evidence::BeliefsAtRung { rung, count })
                })
                .collect(),
            Some(Selector::BoundAt(locus)) => {
                let l = *locus;
                let count = (0..ctx.lens.len())
                    .filter(|p| ctx.lens.at(*p).is_some_and(|f| f.is_bound(l)))
                    .count();
                if count > 0 {
                    vec![Evidence::RowsBoundAt { locus: l, count }]
                } else {
                    vec![]
                }
            }
            None => vec![],
        },
    };

    Warrant {
        step,
        visible_before,
        support,
    }
}

/// One grounded trajectory. Probe-local; NOT a proposed production type.
#[derive(Debug, Clone)]
struct BehaviorTrace {
    initial: ViewPlan,
    steps: Vec<(ViewEdit, Warrant)>,
    final_view: ViewPlan,
    /// The view observed after each step, recorded as it happened.
    observed: Vec<Vec<usize>>,
}

/// Replay a prefix of length `k` from the initial plan.
fn replay(initial: &ViewPlan, edits: &[ViewEdit], k: usize) -> ViewPlan {
    let mut p = initial.clone();
    for e in edits.iter().take(k) {
        p = e.apply(&p);
    }
    p
}

fn inh(s: u16, p: u16) -> CStmt {
    CStmt {
        s,
        cop: Copula::Inh,
        p,
    }
}

fn main() {
    let mut gates: Vec<(&str, bool, String)> = Vec::new();

    // ═══ Same sealed field as the particle: 63 links, closed once ══════
    let mut arena = BeliefArena::new();
    for i in 1..CHAIN_NODES {
        arena.observe(
            inh(i, i + 1),
            TruthValue::new(1.0, 0.99),
            Stamp::source(u32::from(i) - 1),
        );
    }
    arena.close_transitive(16);

    let mut picks: Vec<u32> = Vec::new();
    for &band in &BAND_RUNGS {
        picks.extend(
            arena
                .entries()
                .iter()
                .filter(|b| b.rung == band)
                .take(ROWS_PER_BAND)
                .map(|b| b.rung),
        );
    }
    let n_rows = picks.len();
    let mut rows: Vec<NodeRow> = (0..n_rows)
        .map(|i| NodeRow {
            key: NodeGuid::local(i as u32),
            edges: EdgeBlock::default(),
            value: [0u8; 480],
        })
        .collect();
    for (i, row) in rows.iter_mut().enumerate() {
        let mut f = CausalWitnessFacet::ZERO;
        if i % 2 == 0 {
            f = f.with(Locus::SupportedBy, 1);
        }
        WitnessLens::write_register(row, &f);
    }
    let ctx = Ctx {
        lens: WitnessLens::new(&rows),
        rung_of_row: &picks,
    };

    // ═══ Author the trajectory, recording each warrant from its PREFIX ══
    let initial = ViewPlan::of(&[Selector::BoundAt(Locus::SupportedBy)]);
    let edits = [
        ViewEdit::Push(Selector::RungBand { lo: 0, hi: 2 }),
        ViewEdit::RemoveAt(1),
        ViewEdit::Push(Selector::RungBand { lo: 4, hi: 6 }),
    ];

    let mut plan = initial.clone();
    let mut steps: Vec<(ViewEdit, Warrant)> = Vec::new();
    let mut observed: Vec<Vec<usize>> = Vec::new();
    for (k, e) in edits.iter().enumerate() {
        // Warrant BEFORE applying — sees edits[..k] and nothing later.
        let w = warrant_at(k, &initial, &edits[..k], *e, &arena, &ctx);
        plan = e.apply(&plan);
        observed.push(plan.visible(&ctx));
        steps.push((*e, w));
    }
    let trace = BehaviorTrace {
        initial: initial.clone(),
        steps,
        final_view: plan.clone(),
        observed,
    };

    println!("═══ PROBE-WARRANTED-VIEW-TRACE-1 ═══\n");
    println!(
        "  sealed field: {} beliefs, {n_rows} rows sampled across rungs {:?}\n",
        arena.entries().len(),
        BAND_RUNGS
    );
    println!("  A {:?}", trace.initial.selectors);
    for (k, (e, w)) in trace.steps.iter().enumerate() {
        println!(
            "   ├─ e{} {e:?}\n   │   warrant [{}]: visible_before={} support={:?}\n   ├─ view -> {:?}",
            k + 1,
            if w.is_grounded() { "GROUNDED" } else { "UNGROUNDED" },
            w.visible_before,
            w.support,
            trace.observed[k]
        );
    }
    println!();
    // ═══ T1 — end-to-end replay is exact, on both layers ═══════════════
    let replayed = replay(&trace.initial, &edits, edits.len());
    gates.push((
        "T1 replay(initial, edits) == final_view — descriptor stack AND lowered view",
        replayed == trace.final_view && replayed.visible(&ctx) == trace.final_view.visible(&ctx),
        format!(
            "|final| = {} rows, plans equal = {}",
            trace.final_view.visible(&ctx).len(),
            replayed == trace.final_view
        ),
    ));

    // ═══ T2 — EVERY prefix replays to the view observed at that step ═══
    // Stronger than T1: an end-to-end match can hide two errors that cancel.
    let prefix_ok = (1..=edits.len())
        .all(|k| replay(&trace.initial, &edits, k).visible(&ctx) == trace.observed[k - 1]);
    gates.push((
        "T2 replay(prefix[0..k]) == the view actually observed at step k, for EVERY k",
        prefix_ok,
        format!(
            "k=1..{} all match; observed sizes {:?}",
            edits.len(),
            trace.observed.iter().map(Vec::len).collect::<Vec<_>>()
        ),
    ));

    // ═══ T3 — invert the whole trajectory, step by step, back to A ═════
    let mut back = trace.final_view.clone();
    let mut invertible = 0usize;
    let mut ok = true;
    for k in (0..edits.len()).rev() {
        let before = replay(&trace.initial, &edits, k);
        match edits[k].inverse(&before) {
            Some(inv) => {
                invertible += 1;
                back = inv.apply(&back);
                if back != before {
                    ok = false;
                }
            }
            None => ok = false,
        }
    }
    gates.push((
        "T3 inverting every edit walks the trajectory back to A exactly",
        ok && back == trace.initial && invertible == edits.len(),
        format!(
            "{invertible}/{} invertible; back == A: {}",
            edits.len(),
            back == trace.initial
        ),
    ));

    // ═══ T4 — NO OUTCOME LEAKAGE (the load-bearing law) ════════════════
    // Structural: `warrant_at` has no parameter through which the future
    // could enter. Behavioural: every recorded warrant must be reproducible
    // from its prefix alone, with the rest of the trace unavailable.
    let recomputed: Vec<Warrant> = (0..edits.len())
        .map(|k| warrant_at(k, &trace.initial, &edits[..k], edits[k], &arena, &ctx))
        .collect();
    let recorded: Vec<Warrant> = trace.steps.iter().map(|(_, w)| w.clone()).collect();
    gates.push((
        "T4 every warrant is reproducible from its PREFIX alone — no backward leakage",
        recomputed == recorded,
        format!(
            "{}/{} warrants byte-equal from prefix; warrant_at sees no later edit, no final view, no outcome",
            recomputed.iter().zip(&recorded).filter(|(a, b)| a == b).count(),
            edits.len()
        ),
    ));

    // ═══ T5 — control: replay is LOAD-BEARING, not vacuous ═════════════
    // Swapping two edits must change the outcome. If any permutation
    // reproduced the final view, T1/T2 would be measuring nothing.
    let mut scrambled = edits;
    scrambled.swap(0, 2);
    let scrambled_final = replay(&trace.initial, &scrambled, scrambled.len());
    // And a truncated trace must not reach the final view either.
    let truncated = replay(&trace.initial, &edits, edits.len() - 1);
    gates.push((
        "T5 control: a reordered trace and a truncated trace BOTH miss the final view",
        scrambled_final.visible(&ctx) != trace.final_view.visible(&ctx)
            && truncated.visible(&ctx) != trace.final_view.visible(&ctx),
        format!(
            "reordered -> {:?}, truncated -> {:?}, real final -> {:?} \
             (reordered happens to be the same SIZE as the real final and a \
             different SET — which is why this gate compares sets)",
            scrambled_final.visible(&ctx),
            truncated.visible(&ctx),
            trace.final_view.visible(&ctx)
        ),
    ));

    // ═══ T6 — control: the trajectory is non-trivial ═══════════════════
    // Each step must actually move the view; a trace of no-ops would satisfy
    // T1-T4 while proving nothing.
    let all_distinct = trace.observed.windows(2).all(|w| w[0] != w[1])
        && trace.observed[0] != trace.initial.visible(&ctx);
    gates.push((
        "T6 control: every step changes the visible set (no no-op trajectory)",
        all_distinct && edits.len() >= 3,
        format!(
            "A={:?} then {:?}",
            trace.initial.visible(&ctx).len(),
            trace.observed.iter().map(Vec::len).collect::<Vec<_>>()
        ),
    ));

    // ═══ G1 — every edit in the trajectory NAMES its evidence ═════════
    let all_grounded = trace.steps.iter().all(|(_, w)| w.is_grounded());
    gates.push((
        "G1 GROUNDING: every edit names evidence available BEFORE it (support non-empty)",
        all_grounded,
        format!(
            "support sizes {:?}",
            trace
                .steps
                .iter()
                .map(|(_, w)| w.support.len())
                .collect::<Vec<_>>()
        ),
    ));

    // ═══ G2 — the channel DISCRIMINATES (can-it-stay-silent) ══════════
    // An edit selecting a band the sealed state does not populate must come
    // back UNGROUNDED. Without this, G1 is true of any edit and measures
    // nothing. `RungBand{50,60}` is off-field: the arena's ceiling is R6.
    let ungrounded_edit = ViewEdit::Push(Selector::RungBand { lo: 50, hi: 60 });
    let w_bad = warrant_at(0, &initial, &[], ungrounded_edit, &arena, &ctx);
    // ...and a locus no row is bound at.
    let w_bad2 = warrant_at(
        0,
        &initial,
        &[],
        ViewEdit::Push(Selector::BoundAt(Locus::Contradiction)),
        &arena,
        &ctx,
    );
    gates.push((
        "G2 the warrant channel DISCRIMINATES: an off-field band and an unbound locus are UNGROUNDED",
        !w_bad.is_grounded() && !w_bad2.is_grounded() && all_grounded,
        format!(
            "off-field band support={} unbound locus support={} vs real trace all grounded={}",
            w_bad.support.len(),
            w_bad2.support.len(),
            all_grounded
        ),
    ));

    // ═══ G3 — evidence counts are REAL, not decorative ════════════════
    // Every BeliefsAtRung count must equal a recount of the sealed arena, and
    // every RowsBoundAt count a recount of the population. A warrant that
    // named plausible-but-wrong numbers would pass G1 and G2.
    let counts_true = trace.steps.iter().all(|(_, w)| {
        w.support.iter().all(|ev| match ev {
            Evidence::BeliefsAtRung { rung, count } => {
                arena.entries().iter().filter(|b| b.rung == *rung).count() == *count
            }
            Evidence::RowsBoundAt { locus, count } => {
                (0..ctx.lens.len())
                    .filter(|p| ctx.lens.at(*p).is_some_and(|f| f.is_bound(*locus)))
                    .count()
                    == *count
            }
        })
    });
    gates.push((
        "G3 every named evidence count re-verifies against the sealed state",
        counts_true,
        format!(
            "{} evidence items re-counted",
            trace
                .steps
                .iter()
                .map(|(_, w)| w.support.len())
                .sum::<usize>()
        ),
    ));

    let mut all_green = true;
    for (name, pass, detail) in &gates {
        println!(
            "  [{}] {name} — {detail}",
            if *pass { "PASS" } else { "FAIL" }
        );
        all_green &= *pass;
    }
    println!(
        "\n── SCOPE ──\n\
         A trace is an OBJECT a later BPE-style learner could operate over.\n\
         Nothing is learned, compressed, promoted, or recurrence-detected here.\n\
         `BehaviorTrace` is probe-local and is NOT proposed as a production type.\n\
         F-REVISION-FOCUS-1 remains ABSENT: no production surface emits a\n\
         `ViewEdit`, so this trajectory is AUTHORED by the probe, not harvested\n\
         from a running cognition. A trace OBJECT exists; a trace PRODUCER does\n\
         not. That arrow -- Evaluation -> Revision -> warranted ViewEdit -- is\n\
         the honest remaining gap, and it is one arrow, not a subsystem."
    );
    assert!(all_green, "PROBE-WARRANTED-VIEW-TRACE-1: a gate failed");
    println!("\nPROBE-WARRANTED-VIEW-TRACE-1: ALL GATES GREEN");
}
