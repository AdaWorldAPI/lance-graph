//! `foveated_awareness` — three operator directives, one mechanism:
//!
//!  1. **KanbanStep = self-driven foveated rendering of awareness.** A board of
//!     cards (mailboxes) is rendered at two resolutions: the ONE focal card (the
//!     fovea) renders cognition at full detail — it climbs the rung ladder and
//!     dispatches a specific recipe; every peripheral card renders coarse (no
//!     dispatch), exactly like a foveated display spends its pixel budget on the
//!     gaze point. (The cascade already calls its closest band `Foveal`; this is
//!     that idea at the awareness layer.)
//!  2. **Free energy = the meta-awareness exploration drive (the saccade).** Each
//!     cycle the fovea jumps to the card of highest `free_energy` — attention is
//!     spent where surprise is, and a card whose surprise is resolved drops back
//!     to the periphery. Nothing schedules this; surprise does.
//!  3. **RungLevel 0-9 = a Maslow pyramid of cognition.** A foveated card tries
//!     the cheapest rung first and ESCALATES only while surprise remains (an
//!     unmet need drives the climb); it rests the moment a rung resolves it. The
//!     ten `RungLevel` names ARE the pyramid (Surface…Transcendent), and they
//!     align with Pearl (rung 2 = association, 5 = intervention, 6 = counterfactual).
//!
//! And the load-bearing engineering ask — **all 34 recipes carved out AND wired**:
//! the 34 form a `(mechanism × depth)` address space (bijection, proven from the
//! live registry). The card's STYLE picks the mechanism column; its earned RUNG
//! picks the row. §1 proves 34/34 reachability across the two live doors (the
//! surprise selector + the style→mechanism fan); §2 renders the model.
//!
//! Honest frame (operator: "keep it grounded … don't drift into unscientific"):
//! foveation / saccade / Maslow here are RESOURCE-ALLOCATION models made
//! mechanical — a finite dispatch budget spent by surprise — not a consciousness
//! claim. Every number below is measured from `RECIPES`; KILLs gate the claims.
//!
//! ```sh
//! cargo run -p lance-graph-contract --example foveated_awareness
//! ```

use std::collections::BTreeSet;

use lance_graph_contract::cognitive_shader::RungLevel;
use lance_graph_contract::kanban::{KanbanColumn, KanbanMove};
use lance_graph_contract::materialize::select_tactic;
use lance_graph_contract::mul::GateDecision;
use lance_graph_contract::recipe_kernels::ThoughtCtx;
use lance_graph_contract::recipes::{Mechanism, Recipe, RECIPES};

/// Tier depth (Maslow-monotone: cheap CrossTier work before Hard before XHard).
fn tier_depth(r: &Recipe) -> u8 {
    use lance_graph_contract::recipes::Tier::*;
    match r.tier {
        CrossTier => 0,
        Hard => 1,
        ExtremelyHard => 2,
    }
}

/// The `(mechanism → column)` carve: the recipes of one mechanism, ordered by
/// tier depth then id — shallow rungs first. The union over the four mechanisms
/// is a BIJECTION onto the 34 (proven in §1), so `(mechanism, position)` is a
/// complete address space for the catalogue.
fn column(m: Mechanism) -> Vec<&'static Recipe> {
    let mut v: Vec<&'static Recipe> = RECIPES.iter().filter(|r| r.mechanism == m).collect();
    v.sort_by_key(|r| (tier_depth(r), r.id));
    v
}

/// A card's cluster→mechanism (mirrors planner `style_strategy::cluster_mechanism`
/// — replicated here so the contract example is self-contained; the 5 clusters
/// cover all 4 mechanisms, which is WHY the style fan reaches every recipe).
#[derive(Clone, Copy)]
enum Style {
    Analytical, // → TruthAwareInference
    Creative,   // → StructuralDivergence
    Empathic,   // → ParallelIndependence
    Reflective, // → Infrastructure (the Meta cluster — the ONLY door to the 14)
}
impl Style {
    fn mechanism(self) -> Mechanism {
        match self {
            Style::Analytical => Mechanism::TruthAwareInference,
            Style::Creative => Mechanism::StructuralDivergence,
            Style::Empathic => Mechanism::ParallelIndependence,
            Style::Reflective => Mechanism::Infrastructure,
        }
    }
    fn label(self) -> &'static str {
        match self {
            Style::Analytical => "Analytical",
            Style::Creative => "Creative  ",
            Style::Empathic => "Empathic  ",
            Style::Reflective => "Reflective",
        }
    }
}

/// The Maslow pyramid of cognition: each `RungLevel` → (tier name, the cognitive
/// need that drives the climb to it). Pearl anchors noted where they land.
const MASLOW: [(RungLevel, &str, &str); 10] = [
    (
        RungLevel::Surface,
        "Perception    ",
        "take in the raw signal",
    ),
    (
        RungLevel::Shallow,
        "Recognition   ",
        "match it to something known",
    ),
    (
        RungLevel::Contextual,
        "Association   ",
        "relate it to its neighbors (Pearl L1)",
    ),
    (
        RungLevel::Analogical,
        "Analogy       ",
        "map it onto another domain",
    ),
    (
        RungLevel::Abstract,
        "Abstraction   ",
        "generalize past the instance",
    ),
    (
        RungLevel::Structural,
        "Structure     ",
        "model the mechanism (Pearl L2, do)",
    ),
    (
        RungLevel::Counterfactual,
        "Counterfactual",
        "imagine it otherwise (Pearl L3)",
    ),
    (
        RungLevel::Meta,
        "Metacognition ",
        "reason about the reasoning",
    ),
    (
        RungLevel::Recursive,
        "Recursion     ",
        "reason about that, in turn",
    ),
    (
        RungLevel::Transcendent,
        "Self-actualize",
        "the whole reasons about itself",
    ),
];

/// Render depth: a rung (0-9) samples a position in its mechanism column.
fn position_at(rung: u8, col_len: usize) -> usize {
    if col_len <= 1 {
        0
    } else {
        (rung as usize * (col_len - 1) / 9).min(col_len - 1)
    }
}

/// A card on the board = a mailbox with a style, a surprise level, and the rung
/// it has climbed to. `Copy`, tiny — it rides as owned microcopy.
#[derive(Clone, Copy)]
struct Card {
    name: &'static str,
    style: Style,
    free_energy: f32,
    rung: u8,
    col: KanbanColumn,
}

const FLOOR: f32 = 0.2; // homeostasis: below this, the need is met — defoveate

fn main() {
    let mut kills: Vec<String> = Vec::new();

    // ══ §1 — ALL 34 CARVED OUT AND WIRED (measured coverage proof) ══
    println!("── §1 all 34 recipes carved into a (mechanism × depth) address space, wired via 2 doors ──\n");
    let mechs = [
        (Mechanism::ParallelIndependence, Style::Empathic),
        (Mechanism::TruthAwareInference, Style::Analytical),
        (Mechanism::StructuralDivergence, Style::Creative),
        (Mechanism::Infrastructure, Style::Reflective),
    ];
    // Door A — the style→mechanism fan reaches an entire column at a time.
    let mut wired: BTreeSet<u8> = BTreeSet::new();
    for (m, style) in mechs {
        let col = column(m);
        let ids: Vec<String> = col
            .iter()
            .enumerate()
            .map(|(p, r)| format!("[{p}]#{}{}", r.id, r.code))
            .collect();
        for r in &col {
            wired.insert(r.id);
        }
        let mname = format!("{m:?}");
        println!(
            "  {} → {:<20}  {} recipes: {}",
            style.label(),
            mname,
            col.len(),
            ids.join(" ")
        );
    }
    // Door B — the surprise selector (front door). Sweep its 54-cell band space.
    let mut front: BTreeSet<u8> = BTreeSet::new();
    for &sd in &[0.10f32, 0.25, 0.45] {
        for &rung in &[1u8, 5, 8] {
            for &diss in &[0.1f32, 0.7] {
                for &fe in &[0.15f32, 0.50, 0.80] {
                    let mut c = ThoughtCtx::new(vec![0.9, 0.6, 0.3]);
                    c.sd = sd;
                    c.rung = rung;
                    c.dissonance = diss;
                    c.free_energy = fe;
                    front.insert(select_tactic(&c));
                }
            }
        }
    }
    println!(
        "\n  door A (style→mechanism fan): {} recipes   door B (surprise selector): {} recipes",
        wired.len(),
        front.len()
    );
    let union: BTreeSet<u8> = wired.union(&front).copied().collect();
    let missing: Vec<u8> = (1..=34u8).filter(|id| !union.contains(id)).collect();
    println!(
        "  UNION of the two live doors: {}/34 reachable — {}",
        union.len(),
        if missing.is_empty() {
            "every recipe is carved out AND wired ✓".to_string()
        } else {
            format!("MISSING {missing:?}")
        }
    );
    if union.len() != 34 || !missing.is_empty() {
        kills.push(format!(
            "only {}/34 recipes reachable; missing {missing:?}",
            union.len()
        ));
    }
    // The carve is a bijection: the four columns partition the 34 with no overlap.
    let carved: usize = mechs.iter().map(|(m, _)| column(*m).len()).sum();
    println!(
        "  carve is a partition: {} + {} + {} + {} = {carved} (= 34, no recipe in two columns)",
        column(Mechanism::ParallelIndependence).len(),
        column(Mechanism::TruthAwareInference).len(),
        column(Mechanism::StructuralDivergence).len(),
        column(Mechanism::Infrastructure).len(),
    );
    if carved != 34 {
        kills.push(format!("address space carves {carved} recipes, not 34"));
    }

    // ══ §2 — KANBAN FOVEATION: surprise drives the saccade; the fovea climbs Maslow ══
    println!("\n── §2 the board renders foveated: the fovea jumps to max free-energy, climbs, dispatches ──\n");
    let mut board = [
        Card {
            name: "invoice-anomaly ",
            style: Style::Analytical,
            free_energy: 0.55,
            rung: 0,
            col: KanbanColumn::Planning,
        },
        Card {
            name: "novel-request   ",
            style: Style::Creative,
            free_energy: 0.82,
            rung: 0,
            col: KanbanColumn::Planning,
        },
        Card {
            name: "customer-empathy",
            style: Style::Empathic,
            free_energy: 0.34,
            rung: 0,
            col: KanbanColumn::Planning,
        },
        Card {
            name: "self-audit      ",
            style: Style::Reflective,
            free_energy: 0.68,
            rung: 0,
            col: KanbanColumn::Planning,
        },
    ];
    let mut trail: Vec<KanbanMove> = Vec::new();
    let mut saccade_fs: Vec<f32> = Vec::new();
    let mut prev_focus: Option<usize> = None;
    let mut cycle = 0u32;

    // Loop until every card's surprise is below the homeostasis floor (all rested).
    while board.iter().any(|c| c.free_energy >= FLOOR) && cycle < 12 {
        cycle += 1;
        // SACCADE: the fovea goes to the highest-surprise card that is not yet rested.
        let focus = board
            .iter()
            .enumerate()
            .filter(|(_, c)| c.free_energy >= FLOOR)
            .max_by(|a, b| a.1.free_energy.partial_cmp(&b.1.free_energy).unwrap())
            .map(|(i, _)| i)
            .unwrap();
        saccade_fs.push(board[focus].free_energy);
        // Re-foveating a card that already rested would violate the model.
        if let Some(p) = prev_focus {
            if p != focus && board[p].free_energy < FLOOR && board[p].col != KanbanColumn::Commit {
                kills.push(format!(
                    "card {} rested but was not committed",
                    board[p].name
                ));
            }
        }

        // FOVEATE: full-resolution render on the focal card — climb the Maslow
        // rungs while surprise remains, dispatching the carved recipe at each.
        let col = column(board[focus].style.mechanism());
        let start_rung = board[focus].rung;
        board[focus].col = advance(&mut trail, board[focus], GateDecision::Flow, cycle); // Planning→CognitiveWork
        print!(
            "  cycle {cycle}: FOVEA → {} (F={:.2})  climb:",
            board[focus].name, board[focus].free_energy
        );
        while board[focus].free_energy >= FLOOR && (board[focus].rung as usize) < 9 {
            let r = board[focus].rung;
            let rec = col[position_at(r, col.len())];
            print!(" R{r}·{}", rec.code);
            board[focus].free_energy *= 0.5; // this rung resolves half the surprise
            board[focus].rung += 1; // unmet need → escalate one rung
        }
        // The rung it rested at renders the final recipe.
        let rested_rec = col[position_at(board[focus].rung, col.len())];
        println!(
            "  → rest at R{} ({}) dispatching #{}{} (F={:.2})",
            board[focus].rung,
            MASLOW[board[focus].rung.min(9) as usize].1.trim(),
            rested_rec.id,
            rested_rec.code,
            board[focus].free_energy
        );
        // Monotone climb check (Maslow: you don't descend a rung mid-render).
        if board[focus].rung < start_rung {
            kills.push(format!(
                "card {} climbed DOWN the ladder",
                board[focus].name
            ));
        }
        // DEFOVEATE: surprise resolved → drive the card to Commit and drop to periphery.
        board[focus].col = advance(&mut trail, board[focus], GateDecision::Flow, cycle); // →Evaluation
        board[focus].col = advance(&mut trail, board[focus], GateDecision::Flow, cycle); // →Commit
        prev_focus = Some(focus);
    }

    // The saccade order must be non-increasing in surprise (attention follows F).
    let saccade_monotone = saccade_fs.windows(2).all(|w| w[1] <= w[0] + 1e-6);
    println!(
        "\n  saccade surprise sequence: {}  (attention follows surprise, non-increasing: {})",
        saccade_fs
            .iter()
            .map(|f| format!("{f:.2}"))
            .collect::<Vec<_>>()
            .join(" → "),
        yn(saccade_monotone)
    );
    if !saccade_monotone {
        kills.push("saccade did not follow surprise (F non-monotone)".into());
    }
    let all_rested = board
        .iter()
        .all(|c| c.free_energy < FLOOR && c.col == KanbanColumn::Commit);
    println!(
        "  all cards rested & committed (the board reaches homeostasis): {}",
        yn(all_rested)
    );
    if !all_rested {
        kills.push("not every card reached rest+Commit".into());
    }

    // ══ §3 — the Maslow pyramid of cognition (the rung ladder, named) ══
    println!("\n── §3 RungLevel 0-9 as the Maslow pyramid of cognition (cheap need first, escalate on surprise) ──\n");
    for (rung, tier, need) in MASLOW {
        let apex = if rung == RungLevel::Transcendent {
            "  ◀ apex"
        } else {
            ""
        };
        let rname = format!("{rung:?}");
        println!("  R{}  {tier}  {rname} — {need}{apex}", rung as u8);
    }

    // ── verdict ──
    println!("\n── verdict ──");
    if kills.is_empty() {
        println!(
            "  34/34 recipes carved (bijective address space) and wired (union of both doors);"
        );
        println!(
            "  the fovea followed surprise; each focal card climbed the Maslow ladder and rested."
        );
        println!("  Foveation/saccade/Maslow are resource-allocation models made mechanical — no");
        println!("  phenomenal claim. Every figure is measured from the live RECIPES registry. ✓");
    } else {
        for k in &kills {
            println!("  ✗ KILL: {k}");
        }
        std::process::exit(1);
    }
}

/// Record a legal Rubicon transition for the focal card (the kanban-view update).
fn advance(
    trail: &mut Vec<KanbanMove>,
    card: Card,
    gate: GateDecision,
    cycle: u32,
) -> KanbanColumn {
    match card.col.advance_on_gate(&gate) {
        Some(to) => {
            trail.push(KanbanMove {
                mailbox: 1,
                from: card.col,
                to,
                witness_chain_position: cycle,
                libet_offset_us: if card.col == KanbanColumn::Planning
                    && to == KanbanColumn::CognitiveWork
                {
                    -550_000
                } else {
                    0
                },
                exec: lance_graph_contract::kanban::ExecTarget::Native,
            });
            to
        }
        None => card.col,
    }
}

fn yn(b: bool) -> &'static str {
    if b {
        "YES ✓"
    } else {
        "NO ✗"
    }
}
