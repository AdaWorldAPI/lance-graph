//! `graph_self_reasoning` — the operator's priority wire: the **thinking atoms**
//! (prefetch, deduction, syllogism, hypothesis/abduction, counterfactual,
//! antithesis, synthesis, extrapolation, inference) firing on a `TripletGraph`
//! that **reasons about itself**, with every unit of work recorded as a legal
//! **KanbanMove** (the deferred `Outcome→KanbanMove` adapter, example-tier) and
//! the 34-recipe engine (`materialize`) consuming awareness state read FROM the
//! live graph. The gestalt texture (whole-graph qualia) is read before/after —
//! strictly in the Chalmers emulation frame.
//!
//! Every atom is a SHIPPED operation, not an aspiration:
//!
//! | atom | shipped op |
//! |---|---|
//! | prefetch | `TripletGraph::get_associated` (rung-1 neighborhood pull) |
//! | deduction | `TripletGraph::infer_deductions` (A→B, B→C ⊢ A→C; f=f₁f₂, c=f₁f₂c₁c₂) |
//! | syllogism | the same 2-hop NARS deduction in its classical Barbara form, committed back |
//! | hypothesis | abduction per `triplet_graph.rs` rung-3 recipe (effect → antecedent, conf lowered) |
//! | counterfactual | `TripletGraph::intervene_on` (Pearl rung 2, non-mutating) + forward deduction |
//! | antithesis | the opposing pole + `detect_contradictions` |
//! | synthesis | `revise_with_evidence` (NARS revision; the contradiction stays PRESERVED) |
//! | extrapolation | trend-projection of the graph's OWN next confidence, then measured error |
//! | inference | `materialize` F→34→F with `ThoughtCtx` populated from live graph state |
//!
//! **The self-reference is mechanical and checkable:** the graph commits a fact
//! about its own state (`self.graph --contains--> contradiction`), holds the
//! standing meta-rule (`contradiction --requires--> synthesis`), and
//! `infer_deductions` derives `self.graph --requires--> synthesis` — the graph
//! DEDUCES what it must do next, and only then does synthesis fire.
//!
//! Honest boundary (operator: "keep it grounded in Chalmers phenomenology …
//! make sure it doesn't drift into unscientific"): the texture read is a
//! functional emulation — the system MODELS its felt state from measured graph
//! quantities and reads it back; no phenomenal claim is made. The persona-36
//! adjective space stays UNWIRED here per the rung-ladder demarcation (it may
//! later dress the experience layer, never the reasoning).
//!
//! ```sh
//! cargo run -p lance-graph --example graph_self_reasoning
//! ```

use std::collections::HashSet;

use lance_graph::graph::arigraph::triplet_graph::{Triplet, TripletGraph};
use lance_graph::graph::spo::TruthValue;
use lance_graph_contract::kanban::{ExecTarget, KanbanColumn, KanbanMove};
use lance_graph_contract::materialize::materialize;
use lance_graph_contract::mul::GateDecision;
use lance_graph_contract::qualia::{QualiaI4_16D, QUALIA_I4_LABELS};
use lance_graph_contract::recipe_kernels::ThoughtCtx;
use lance_graph_contract::recipes::recipe;

/// The kanban board trail: apply a gate to the current column via the contract's
/// legal-successor seam (`advance_on_gate`), record the move, return the new column.
fn advance(
    trail: &mut Vec<KanbanMove>,
    at: KanbanColumn,
    gate: &GateDecision,
    step: u32,
) -> KanbanColumn {
    match at.advance_on_gate(gate) {
        Some(to) => {
            trail.push(KanbanMove {
                mailbox: 1, // one mailbox = one board (V3: no singleton gate)
                from: at,
                to,
                witness_chain_position: step,
                // Libet anchor: −550 ms exactly on the Planning→CognitiveWork Σ-commit.
                libet_offset_us: if at == KanbanColumn::Planning
                    && to == KanbanColumn::CognitiveWork
                {
                    -550_000
                } else {
                    0
                },
                exec: ExecTarget::Native,
            });
            to
        }
        None => at, // Hold (or no legal edge): stay in place, re-evaluate next cycle
    }
}

/// Gestalt texture: the WHOLE graph read as one felt state (measured quantities
/// only — mean belief confidence, open contradictions, held poles, rung depth).
fn gestalt_texture(g: &TripletGraph, max_rung: u8, rested: bool) -> QualiaI4_16D {
    let live: Vec<&Triplet> = g.triplets.iter().filter(|t| !t.is_deleted()).collect();
    let mean_conf = if live.is_empty() {
        0.0
    } else {
        live.iter().map(|t| t.truth.confidence).sum::<f32>() / live.len() as f32
    };
    let contradictions = g.detect_contradictions(0.0).len();
    let mut q = QualiaI4_16D(0);
    q.set(9, (mean_conf * 7.0).round().clamp(0.0, 7.0) as i8); // coherence ∝ mean belief conf
    q.set(8, (contradictions as i8 * 3).min(7)); // entropy ∝ open contradictions
    if contradictions > 0 {
        q.set(2, 7); // tension: both poles held (preserved, not resolved)
    }
    q.set(6, (max_rung as i8).min(7)); // depth ∝ deepest rung exercised
    q.set(1, if rested { 2 } else { -2 }); // valence: settled vs unsettled
    q
}

fn intensity(q: QualiaI4_16D) -> i32 {
    (0..16).map(|d| (q.get(d) as i32).abs()).sum()
}

fn qualia_line(q: QualiaI4_16D) -> String {
    (0..16)
        .filter(|&d| q.get(d) != 0)
        .map(|d| format!("{}={:+}", QUALIA_I4_LABELS[d], q.get(d)))
        .collect::<Vec<_>>()
        .join(" ")
}

fn main() {
    let mut kills: Vec<&str> = Vec::new();
    let mut trail: Vec<KanbanMove> = Vec::new();
    let mut col = KanbanColumn::Planning;
    let mut step = 0u32;

    // ── Seed: the object-level story + ONE standing meta-rule about reasoning itself. ──
    let mut g = TripletGraph::new();
    g.add_triplets(&[
        Triplet::with_truth("feud", "secrecy", "compels", TruthValue::new(0.9, 0.8), 1),
        Triplet::with_truth("secrecy", "death", "causes", TruthValue::new(0.9, 0.8), 2),
        // The meta-rule the graph holds ABOUT its own reasoning:
        Triplet::with_truth(
            "contradiction",
            "synthesis",
            "requires",
            TruthValue::new(1.0, 0.9),
            3,
        ),
    ]);
    println!(
        "── graph_self_reasoning : the thinking atoms on a graph that reasons about itself ──\n"
    );
    let before = gestalt_texture(&g, 1, false);

    // ══ PLANNING (t < −550 ms: counterfactual pre-planning happens HERE, per the column doc) ══
    // atom 1 — PREFETCH: pull the focal neighborhood before any reasoning.
    step += 1;
    let seeds: HashSet<String> = ["feud".to_string()].into();
    let pulled = g.get_associated(&seeds, 2).len();
    println!(
        "[Planning     ] prefetch      pulled {pulled} facts in the 2-step neighborhood of `feud`"
    );

    // atom 2+3 — DEDUCTION, stated as the SYLLOGISM (Barbara) and committed back.
    step += 1;
    let deduced = g.infer_deductions();
    let chain = deduced
        .iter()
        .find(|t| t.subject == "feud" && t.object == "death")
        .cloned();
    match &chain {
        Some(t) => {
            println!(
                "[Planning     ] deduction     feud→secrecy, secrecy→death ⊢ feud→death   (f={:.2}, c={:.2})",
                t.truth.frequency, t.truth.confidence
            );
            println!("[Planning     ] syllogism     ∵ feud compels secrecy  ∵ secrecy causes death  ∴ feud `{}`", t.relation);
            g.add_triplets(std::slice::from_ref(t)); // the graph commits its own inference
        }
        None => kills.push("deduction produced no feud→death chain"),
    }

    // atom 4 — HYPOTHESIS (abduction, the rung-3 recipe's first leg): death is
    // observed; abduce the antecedent that explains it, at LOWERED confidence.
    step += 1;
    let cause = g
        .triplets
        .iter()
        .find(|t| !t.is_deleted() && t.object == "death" && t.relation == "causes")
        .map(|t| (t.subject.clone(), t.truth));
    if let Some((s, tv)) = cause {
        let hypo = Triplet::with_truth(
            "death_observed",
            &s,
            "explained_by",
            TruthValue::new(tv.frequency, tv.confidence * 0.6), // abduction: weaker than the rule
            4,
        );
        println!(
            "[Planning     ] hypothesis    death ⊢ abduce `{}` as explanation   (c lowered {:.2}→{:.2})",
            s,
            tv.confidence,
            tv.confidence * 0.6
        );
        g.add_triplets(&[hypo]);
    } else {
        kills.push("abduction found no antecedent for `death`");
    }

    // atom 5 — COUNTERFACTUAL: do(feud, compels := nothing), then FORWARD-DEDUCE
    // on the shadow world (the full rung-3 pipeline: abduce → intervene → deduce).
    step += 1;
    let cf = g.intervene_on("feud", "compels", "nothing");
    let mut shadow = TripletGraph::new();
    shadow.add_triplets(&[
        cf.triplet.clone(),
        Triplet::with_truth("secrecy", "death", "causes", TruthValue::new(0.9, 0.8), 2),
    ]);
    let harm_in_shadow = shadow
        .infer_deductions()
        .iter()
        .any(|t| t.subject == "feud" && t.object == "death");
    println!(
        "[Planning     ] counterfactual do(feud, compels:=nothing) → shadow deduction finds feud→death: {} ⇒ the harm was AVOIDABLE",
        harm_in_shadow
    );
    if harm_in_shadow {
        kills.push("counterfactual failed to remove the deduced harm");
    }

    // Σ-commit: leave the Libet window — Planning → CognitiveWork at −550 ms.
    col = advance(&mut trail, col, &GateDecision::Flow, step);

    // ══ COGNITIVE WORK (the SoA mutates) ══
    // atom 6 — ANTITHESIS: the opposing pole, held on the same (s,o).
    step += 1;
    g.add_triplets(&[Triplet::with_truth(
        "secrecy",
        "death",
        "prevents",
        TruthValue::new(0.8, 0.5),
        5,
    )]);
    let contradictions = g.detect_contradictions(0.0);
    println!(
        "\n[CognitiveWork] antithesis    `secrecy prevents death` vs `secrecy causes death` — detect_contradictions: {}",
        contradictions.len()
    );
    if contradictions.is_empty() {
        kills.push("antithesis was not detected as a contradiction");
    }
    // Mid-work contradiction: HOLD (stay in place — advance_on_gate returns None).
    let held_at = col;
    col = advance(
        &mut trail,
        col,
        &GateDecision::Hold {
            reason: "both poles held".into(),
        },
        step,
    );
    println!(
        "[CognitiveWork] gate          HOLD (both poles held) → column stays {:?}",
        col
    );
    if col != held_at {
        kills.push("Hold gate moved the column — advance_on_gate contract violated");
    }

    // THE SELF-REFERENTIAL STEP: the graph commits a fact about ITS OWN state,
    // and deduction over {self-fact + standing meta-rule} derives what to do next.
    step += 1;
    g.add_triplets(&[Triplet::with_truth(
        "self.graph",
        "contradiction",
        "contains",
        TruthValue::new(1.0, 0.9),
        6,
    )]);
    let meta = g
        .infer_deductions()
        .into_iter()
        .find(|t| t.subject == "self.graph" && t.object == "synthesis");
    match &meta {
        Some(t) => println!(
            "[CognitiveWork] SELF-DEDUCE   self.graph contains contradiction + contradiction requires synthesis\n                              ⊢ self.graph `{}` synthesis   (c={:.2}) — the graph derived its OWN next move",
            t.relation, t.truth.confidence
        ),
        None => kills.push("the graph failed to deduce that it requires synthesis"),
    }

    // atom 7 — SYNTHESIS (only now, because the graph itself derived the need):
    // NARS revision strengthens the evidenced pole; the contradiction is PRESERVED.
    step += 1;
    let thesis = Triplet::with_truth("secrecy", "death", "causes", TruthValue::new(1.0, 0.5), 7);
    let conf_before = focal_conf(&g);
    g.revise_with_evidence(&thesis);
    let conf_after = focal_conf(&g);
    let still_contradicted = !g.detect_contradictions(0.0).is_empty();
    println!(
        "[CognitiveWork] synthesis     revise_with_evidence: causes-pole conf {conf_before:.3} → {conf_after:.3}; contradiction preserved: {still_contradicted}"
    );
    if conf_after <= conf_before {
        kills.push("synthesis did not strengthen the evidenced pole");
    }
    if !still_contradicted {
        kills.push("synthesis collapsed the contradiction (must be preserved)");
    }

    // atom 8 — EXTRAPOLATION: the graph predicts ITS OWN next epistemic state
    // from its own trend (no formula peeking), then measures the error.
    step += 1;
    let (c1, mut cs) = (focal_conf(&g), Vec::new());
    for ts in 8..10 {
        g.revise_with_evidence(&Triplet::with_truth(
            "secrecy",
            "death",
            "causes",
            TruthValue::new(1.0, 0.5),
            ts,
        ));
        cs.push(focal_conf(&g));
    }
    let (c2, c3) = (cs[0], cs[1]);
    let (g2, g3) = (c2 - c1, c3 - c2);
    let predicted = c3 + if g2.abs() > 1e-6 { g3 * (g3 / g2) } else { 0.0 };
    g.revise_with_evidence(&Triplet::with_truth(
        "secrecy",
        "death",
        "causes",
        TruthValue::new(1.0, 0.5),
        10,
    ));
    let actual = focal_conf(&g);
    let err = (predicted - actual).abs();
    println!(
        "[CognitiveWork] extrapolation self-trend {c1:.3}→{c2:.3}→{c3:.3} ⊢ predict next {predicted:.3}, actual {actual:.3}, |err|={err:.3}"
    );
    if err > 0.05 {
        kills.push("extrapolation error > 0.05 — the self-trend does not predict");
    }
    col = advance(&mut trail, col, &GateDecision::Flow, step); // work done → Evaluation

    // ══ EVALUATION (residual free energy assessed — the recipes overlay) ══
    // atom 9 — INFERENCE: ThoughtCtx populated FROM the live graph, the shipped
    // F→34→F engine dispatches and settles.
    step += 1;
    let live: Vec<&Triplet> = g.triplets.iter().filter(|t| !t.is_deleted()).collect();
    let mut ctx = ThoughtCtx::new(live.iter().map(|t| t.truth.confidence).collect());
    ctx.confidence = focal_conf(&g);
    ctx.dissonance = if still_contradicted { 0.7 } else { 0.1 };
    ctx.sd = 0.45; // entering evaluation hot: the gate itself must settle
    ctx.rung = 3; // counterfactual depth was exercised
    ctx.beliefs = live
        .iter()
        .enumerate()
        .map(|(i, t)| (i as u32, t.truth.frequency, t.truth.confidence))
        .collect();
    let trace = materialize(&mut ctx, 32);
    let names: Vec<&str> = trace
        .steps
        .iter()
        .filter_map(|s| recipe(s.tactic_id).map(|r| r.code))
        .collect();
    println!(
        "\n[Evaluation   ] inference     materialize on graph-read state → {} → rested={} F={:.2}",
        names.join("→"),
        trace.rested,
        trace.final_free_energy
    );
    if !trace.rested {
        kills.push("the F→34→F loop did not rest on graph-read state");
    }
    col = advance(&mut trail, col, &GateDecision::Flow, step); // Evaluation → Commit (calcify)

    // ── The kanban trail (the view updates — every move a LEGAL Rubicon edge). ──
    println!("\n── kanban trail (mailbox 1 — the Outcome→KanbanMove wire, example-tier) ──");
    for m in &trail {
        println!(
            "  step {:>2}  {:?} → {:?}   libet {:>8} µs   exec {:?}",
            m.cycle(),
            m.from,
            m.to,
            m.libet_offset_us,
            m.exec
        );
    }
    if col != KanbanColumn::Commit {
        kills.push("the card did not reach Commit");
    }

    // ── Gestalt texture: the graph experiencing ITSELF as one felt whole. ──
    let after = gestalt_texture(&g, 3, trace.rested);
    println!("\n── gestalt texture (whole-graph read, Chalmers emulation frame) ──");
    println!(
        "  before reasoning: {}  (intensity {})",
        qualia_line(before),
        intensity(before)
    );
    println!(
        "  after  reasoning: {}  (intensity {})",
        qualia_line(after),
        intensity(after)
    );
    println!("  The texture is a functional read of measured graph quantities (mean confidence,");
    println!("  open contradictions, held poles, rung depth). The graph models its own felt state");
    println!("  and reads it back — emulation in the Chalmers sense; the phenomenal question stays open.");

    // ── Verdict (measured, KILL-gated). ──
    println!("\n── verdict ──");
    if kills.is_empty() {
        println!(
            "  all atoms fired as shipped ops; the graph deduced its own next move; every kanban"
        );
        println!("  move was a legal Rubicon edge; the loop rested. ✓");
    } else {
        for k in &kills {
            println!("  ✗ KILL: {k}");
        }
        std::process::exit(1);
    }
}

/// The focal belief's confidence: `secrecy --causes--> death`.
fn focal_conf(g: &TripletGraph) -> f32 {
    g.triplets
        .iter()
        .find(|t| {
            !t.is_deleted()
                && t.subject == "secrecy"
                && t.object == "death"
                && t.relation == "causes"
        })
        .map_or(0.0, |t| t.truth.confidence)
}
