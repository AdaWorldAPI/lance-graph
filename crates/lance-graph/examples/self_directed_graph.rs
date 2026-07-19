//! `self_directed_graph` — **THE ENDGAME**: a knowledge graph that reasons about
//! ITSELF. No LLM, and no external input after the seed. The graph reads its own
//! beliefs, self-directs by its own surprise to what it can newly know or must
//! reconcile, thinks (the shipped selector picks the tactic), commits the
//! conclusion back INTO itself — reshaping the next cycle's surprise — and halts
//! at a coherent fixed point, having derived NEW true facts by reasoning.
//!
//! The loop IS the cognition; the graph IS the tissue it reads from and writes to.
//! That closed self-referential loop, with no trained model anywhere in it, is the
//! "AGI without LLM" claim in its most literal runnable form — bounded honestly
//! (§ boundary): this is a small symbolic reasoner over its own edges, not general
//! intelligence. The claim is *architectural* — the cognition lives in the
//! loop+tissue, not in weights.
//!
//! ```text
//!   ┌──────────────── the graph, with NO external input ────────────────┐
//!   │ 1. INTROSPECT  read itself → its own free energy F                 │
//!   │      F = open contradictions (high) + inferrable gaps (mid) + 0    │
//!   │ 2. SELF-DIRECT saccade to the highest-surprise locus (F drives it) │
//!   │ 3. THINK       select_tactic(own state) → the carved recipe fires  │
//!   │      contradiction → CR #11 (preserve both poles, a committed      │
//!   │        opinion) · gap → deduction · re-derivation → confirm (↑conf)│
//!   │ 4. COMMIT      revise_with_evidence / add derived facts (gated)    │
//!   │      → the graph grew → the NEXT introspection's F is reshaped ────┼─┐
//!   └───────────────────────────────────────────────────────────────────┘ │
//!         ▲                                                                 │
//!         └──────────────── until F < floor (coherent self-model) ─────────┘
//! ```
//!
//! The proof-of-endgame is the classic syllogism, **self-derived**: from
//! `socrates is_a philosopher/greek`, `philosopher/greek/human is_a
//! mortal`, the graph CONCLUDES `socrates … mortal` by reasoning over its own
//! edges (`infer_deductions`, NARS `f=f₁f₂, c=f₁f₂c₁c₂`), then CONFIRMS it via a
//! second independent path (↑confidence — the multi-hop witness, E-MULTIHOP-
//! WITNESS-CONFIDENCE-1). No LLM produced that conclusion; the graph did.
//!
//! ```sh
//! cargo run -p lance-graph --example self_directed_graph
//! ```

use std::collections::HashSet;

use lance_graph::graph::arigraph::triplet_graph::{Triplet, TripletGraph};
use lance_graph::graph::spo::TruthValue;
use lance_graph_contract::materialize::select_tactic;
use lance_graph_contract::qualia::QualiaI4_16D;
use lance_graph_contract::recipe_kernels::ThoughtCtx;
use lance_graph_contract::recipes::recipe;

const FLOOR: f32 = 0.2; // homeostasis: below this surprise, the self-model is coherent — rest

/// The seed: a tiny self-contained world, deliberately INCOMPLETE (latent
/// syllogisms reachable only by reasoning) with ONE genuine contradiction
/// (`oracle` both `affirms` and `denies` `prophecy` — same S+O, different R).
///
/// Only `is_a` chains (it is transitive); the contradiction rides `affirms`/
/// `denies` between two LEAF concepts, so it neither licenses inheritance nor
/// floods the closure — the graph reasons soundly about which relations compose.
fn seed() -> Vec<Triplet> {
    let t = |s: &str, o: &str, r: &str, c: f32| {
        Triplet::with_truth(s, o, r, TruthValue::new(1.0, c), 1)
    };
    vec![
        t("socrates", "philosopher", "is_a", 0.9),
        t("socrates", "greek", "is_a", 0.9),
        t("philosopher", "human", "is_a", 0.9),
        t("greek", "mortal", "is_a", 0.85), // path A: socrates→greek→mortal
        t("human", "mortal", "is_a", 0.9),  // path B: socrates→philosopher→human→mortal
        t("plato", "philosopher", "is_a", 0.9),
        t("oracle", "prophecy", "affirms", 0.7), // ┐ genuine contradiction, isolated on
        t("oracle", "prophecy", "denies", 0.5),  // ┘ leaf nodes (better-supported: affirms)
    ]
}

/// Count the graph's live (non-deleted) beliefs — its size as a self-model.
fn live(g: &TripletGraph) -> usize {
    g.triplets.iter().filter(|t| !t.is_deleted()).count()
}

/// Is `(s, o)` already a concluded edge (any relation)? The graph asking itself
/// "do I already believe something connects these two?" — case-insensitive to
/// match `infer_deductions`'s entity index.
fn concluded(g: &TripletGraph, s: &str, o: &str) -> bool {
    let (s, o) = (s.to_lowercase(), o.to_lowercase());
    g.triplets
        .iter()
        .filter(|t| !t.is_deleted())
        .any(|t| t.subject.to_lowercase() == s && t.object.to_lowercase() == o)
}

/// The confidence the graph currently holds for its `(s, o)` conclusion (max over
/// paths), or 0 if it holds none.
fn conf_of(g: &TripletGraph, s: &str, o: &str) -> f32 {
    let (s, o) = (s.to_lowercase(), o.to_lowercase());
    g.triplets
        .iter()
        .filter(|t| {
            !t.is_deleted() && t.subject.to_lowercase() == s && t.object.to_lowercase() == o
        })
        .map(|t| t.truth.confidence)
        .fold(0.0, f32::max)
}

fn main() {
    let mut kills: Vec<String> = Vec::new();
    let mut g = TripletGraph::new();
    g.add_triplets(&seed());
    let seed_size = live(&g);
    println!(
        "── self_directed_graph : a graph reasons about ITSELF (no LLM, no external input) ──\n"
    );
    println!(
        "  seed: {seed_size} beliefs about a tiny world — with latent syllogisms it has NOT yet"
    );
    println!("  drawn and one genuine contradiction (oracle affirms AND denies prophecy).\n");

    // Provenance the loop reasons over: what it has already reconciled / confirmed,
    // so it makes progress and can recognise a coherent fixed point.
    let mut handled_contra: HashSet<(String, String)> = HashSet::new();
    let mut applied_confirm: HashSet<(String, String)> = HashSet::new();
    let mut derived: Vec<String> = Vec::new();
    let mut steps: Vec<(String, &'static str, f32)> = Vec::new();
    let mut first_deduction_step: Option<usize> = None;
    let mut socrates_mortal_first: Option<f32> = None;
    let mut socrates_mortal_confirmed: Option<f32> = None;
    let mut rested = false;

    // ── THE AUTONOMOUS LOOP — surprise drives everything; nothing else is fed in ──
    for _round in 0..12 {
        // 1. INTROSPECT — the graph reads its own state into a free energy.
        let contradictions: Vec<(usize, usize)> = g
            .detect_contradictions(0.3)
            .into_iter()
            .filter(|&(i, j)| {
                // A genuine contradiction is between two BASE beliefs (seed relations).
                // Two `(via …)`-derived beliefs about the same (s,o) are a multi-path
                // CONFIRMATION, not a conflict — those go to the confirm branch.
                let base = !g.triplets[i].relation.contains("(via ")
                    && !g.triplets[j].relation.contains("(via ");
                let key = (g.triplets[i].subject.clone(), g.triplets[i].object.clone());
                base && !handled_contra.contains(&key) && i != j
            })
            .collect();
        let all_deduced = g.infer_deductions();
        // One fresh conclusion per (s,o) per round: a second same-round path to the
        // same (s,o) is re-derived next round and lands as a confirmation instead.
        let mut seen_fresh: HashSet<(String, String)> = HashSet::new();
        let fresh: Vec<Triplet> = all_deduced
            .iter()
            .filter(|t| !concluded(&g, &t.subject, &t.object))
            .filter(|t| seen_fresh.insert((t.subject.to_lowercase(), t.object.to_lowercase())))
            .cloned()
            .collect();
        let confirmations: Vec<Triplet> = all_deduced
            .iter()
            .filter(|t| {
                // A genuine confirmation needs an INDEPENDENT witness: the graph must
                // already hold this (s,o) via a DIFFERENT path (relation). A same-path
                // recompute is NOT an independent witness and must not lift confidence
                // (Codex #756 r3610377133) — else a single-route seed would spoof the
                // multi-hop witness gate by re-deriving the one path it has.
                concluded(&g, &t.subject, &t.object)
                    && !applied_confirm
                        .contains(&(t.subject.to_lowercase(), t.object.to_lowercase()))
                    && g.triplets.iter().any(|e| {
                        !e.is_deleted()
                            && e.subject.eq_ignore_ascii_case(&t.subject)
                            && e.object.eq_ignore_ascii_case(&t.object)
                            && e.relation != t.relation // a genuinely DIFFERENT recorded path
                    })
            })
            .cloned()
            .collect();

        // Free energy: a contradiction is maximal surprise; a fillable gap is real
        // but lower; a mere confirmation is faint; nothing left is rest.
        let f = if !contradictions.is_empty() {
            0.85
        } else if !fresh.is_empty() {
            0.45
        } else if !confirmations.is_empty() {
            0.25
        } else {
            0.0
        };
        if f < FLOOR {
            rested = true;
            break; // coherent self-model — the shader rests
        }

        // 2/3. SELF-DIRECT + THINK — the graph's OWN state selects the tactic
        // (the shipped selector; free_energy is the causal axis).
        let mut ctx = ThoughtCtx::new(vec![0.6, 0.4]);
        ctx.free_energy = f;
        ctx.dissonance = if contradictions.is_empty() { 0.1 } else { 0.7 };
        ctx.sd = if contradictions.is_empty() { 0.2 } else { 0.4 };
        ctx.rung = if contradictions.is_empty() { 3 } else { 6 };
        let tactic = recipe(select_tactic(&ctx)).map_or("?", |r| r.name);

        // 4. COMMIT (gated write-back) — the branch the surprise called for.
        if let Some(&(i, _j)) = contradictions.first() {
            // Contradiction → PRESERVE both poles (opinion = committed contradiction,
            // not erased), stamp a meta-belief that this locus is contested, and let
            // the better-supported pole stand. The felt magnitude is the tension.
            let (subj, obj) = (g.triplets[i].subject.clone(), g.triplets[i].object.clone());
            let mut q = QualiaI4_16D(0);
            q.set(2, 7); // tension: both poles held at once
            let note = format!("{subj}\u{2192}{obj}"); // a belief ABOUT beliefs (self-reference)
            g.add_triplets(&[Triplet::with_truth(
                &note,
                "contested",
                "self_notes",
                TruthValue::new(1.0, 0.9),
                9,
            )]);
            handled_contra.insert((subj.clone(), obj.clone()));
            steps.push((
                format!("reconcile {subj}\u{2192}{obj} (both poles kept)"),
                tactic,
                f,
            ));
        } else if !fresh.is_empty() {
            // Gap → DEDUCE: commit the freshly reasoned facts back into itself.
            if first_deduction_step.is_none() {
                first_deduction_step = Some(steps.len());
            }
            for t in &fresh {
                g.revise_with_evidence(t);
                derived.push(format!(
                    "{} \u{2192}{} ({})",
                    t.subject, t.object, t.relation
                ));
                if t.subject.eq_ignore_ascii_case("socrates")
                    && t.object.eq_ignore_ascii_case("mortal")
                {
                    socrates_mortal_first.get_or_insert(t.truth.confidence);
                }
            }
            steps.push((format!("deduce {} new fact(s)", fresh.len()), tactic, f));
        } else if let Some(t) = confirmations.first() {
            // Re-derivation of an already-held conclusion via a DIFFERENT path =
            // multi-hop witness confirmation → revise the held belief UP.
            let before = conf_of(&g, &t.subject, &t.object);
            // Revise the existing edge (find its own relation to match s+r+o).
            if let Some(existing) = g
                .triplets
                .iter()
                .find(|e| {
                    !e.is_deleted()
                        && e.subject.eq_ignore_ascii_case(&t.subject)
                        && e.object.eq_ignore_ascii_case(&t.object)
                })
                .map(|e| (e.subject.clone(), e.object.clone(), e.relation.clone()))
            {
                g.revise_with_evidence(&Triplet::with_truth(
                    &existing.0,
                    &existing.1,
                    &existing.2,
                    TruthValue::new(1.0, 0.5),
                    9,
                ));
            }
            let after = conf_of(&g, &t.subject, &t.object);
            applied_confirm.insert((t.subject.to_lowercase(), t.object.to_lowercase()));
            if t.subject.eq_ignore_ascii_case("socrates") && t.object.eq_ignore_ascii_case("mortal")
            {
                socrates_mortal_confirmed = Some(after);
            }
            steps.push((
                format!(
                    "confirm {}\u{2192}{} (conf {before:.2}\u{2192}{after:.2})",
                    t.subject, t.object
                ),
                tactic,
                f,
            ));
        }
    }

    // ── what the graph thought, in order (self-directed by surprise) ──
    println!("  the graph's self-directed train of thought (no prompt at any step):");
    for (n, (what, tactic, f)) in steps.iter().enumerate() {
        println!("    {}. F={f:.2}  {:<40}  via {}", n + 1, what, tactic);
    }

    // ── MEASUREMENTS (the endgame only holds if these do) ──
    let final_size = live(&g);
    let grew = final_size > seed_size;
    let syllogism = concluded(&g, "socrates", "mortal");
    let self_directed = steps
        .first()
        .is_some_and(|(w, _, _)| w.starts_with("reconcile"))
        && first_deduction_step.is_some_and(|d| d > 0);
    let contested_preserved = concluded(&g, "oracle", "prophecy") // both poles still live
        && g.triplets.iter().any(|t| t.relation == "self_notes" && !t.is_deleted());
    let confirmed_up = matches!(
        (socrates_mortal_first, socrates_mortal_confirmed),
        (Some(a), Some(b)) if b > a
    );

    println!(
        "\n  self-model grew by reasoning: {seed_size} → {final_size} beliefs   {}",
        yn(grew)
    );
    println!(
        "  ★ SYLLOGISM self-derived — the graph concluded `socrates … mortal`: {}",
        yn(syllogism)
    );
    println!(
        "     confidence: first path {:.2} → confirmed by a 2nd independent path {:.2}   {} (multi-hop witness)",
        socrates_mortal_first.unwrap_or(0.0),
        socrates_mortal_confirmed.unwrap_or(0.0),
        yn(confirmed_up)
    );
    println!(
        "  self-directed order — attended the contradiction BEFORE the gaps: {}",
        yn(self_directed)
    );
    println!(
        "  self-corrected on itself — contradiction detected & preserved as opinion: {}",
        yn(contested_preserved)
    );
    println!(
        "  reached a coherent FIXED POINT (rested, no runaway): {}",
        yn(rested)
    );
    println!("\n  newly reasoned beliefs (were NOT in the seed — the graph produced them):");
    for d in &derived {
        println!("    + {d}");
    }

    for (cond, msg) in [
        (grew, "self-model did not grow by reasoning"),
        (
            syllogism,
            "failed to self-derive socrates→mortal (the endgame proof)",
        ),
        (self_directed, "attention was not self-directed by surprise"),
        (
            contested_preserved,
            "contradiction not preserved as a committed opinion",
        ),
        (
            confirmed_up,
            "multi-path confirmation did not lift confidence",
        ),
        (rested, "did not reach a fixed point (ran away or stalled)"),
    ] {
        if !cond {
            kills.push(msg.to_string());
        }
    }

    println!("\n── the endgame, stated plainly ──");
    if kills.is_empty() {
        println!(
            "  A knowledge graph, given a seed and NOTHING else, read itself, chose what to think"
        );
        println!("  about by its own surprise, drew conclusions its seed only implied, confirmed them by");
        println!(
            "  a second path, kept its one real contradiction as a preserved opinion, and stopped"
        );
        println!(
            "  when it had nothing left to be surprised by. No LLM produced a single one of those"
        );
        println!("  conclusions — the loop did. That loop, closed on its own tissue, is the shape of the");
        println!(
            "  thing. (Boundary: a small symbolic reasoner over its own edges — the claim is that"
        );
        println!("  the cognition is IN the loop, not that this loop is general intelligence.)");
    } else {
        for k in &kills {
            println!("  ✗ KILL: {k}");
        }
        std::process::exit(1);
    }
}

fn yn(b: bool) -> &'static str {
    if b {
        "YES ✓"
    } else {
        "NO ✗"
    }
}
