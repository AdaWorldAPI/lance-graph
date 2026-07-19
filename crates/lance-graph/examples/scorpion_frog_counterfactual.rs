//! `scorpion_frog_counterfactual` — Pearl's three rungs on Aesop's scorpion &
//! frog, showing why the **counterfactual synthesis is a *preserved
//! contradiction*** (avoidable ∧ necessary), not a scalar truth.
//!
//! Run: `cargo run -p lance-graph --example scorpion_frog_counterfactual`
//!
//! The fable is built to break scalar truth: the scorpion stings the frog
//! mid-river; both drown; "it is my nature." The tragedy is that the death is
//! simultaneously **avoidable** (had it not stung, they cross — Rung 2 `do(¬sting)`)
//! and **necessary** (its nature compels the sting — Rung 3 modal). A scalar
//! `TruthValue` collapses that; the contradictory rung ladder preserves it.
//!
//! Grounded in the real APIs: `intervene_on` is the shipped Rung-2 do-operator
//! (`triplet_graph.rs:789`, non-mutating — the road-not-taken lives in a separate
//! lane, cf. `contract::counterfactual`: "costs 4 bits, not a replay buffer");
//! `detect_contradictions` surfaces the avoidable/necessary tension. The Rung-3
//! abduce→intervene→predict *composition* is a documented gap (`#[ignore]`-d,
//! PR-LL-4), so here Rung 3 is read as the modal path, not computed.

use lance_graph::graph::arigraph::triplet_graph::{Triplet, TripletGraph};

fn main() {
    // ── The fable as SPO. `Triplet::new` is (subject, OBJECT, relation, ts). ──
    let mut g = TripletGraph::new();
    g.add_triplets(&[
        Triplet::new("scorpion", "frog", "stings", 1), // the deed
        Triplet::new("sting", "death", "causes", 2),   // sting → death
        Triplet::new("frog", "scorpion", "carries", 3), // the ride
        Triplet::new("scorpion", "nature", "has", 4),  // the modal premise
        Triplet::new("nature", "sting", "compels", 5), // nature ⊨ sting
    ]);

    println!("── Aesop: the scorpion & the frog — counterfactual synthesis ──\n");

    // ── Rung 1 (observe): the factual association. ──
    println!("Rung 1  observe   : scorpion --stings--> frog ;  sting --causes--> death");
    println!("        factual   : both drown.  (association  sting ⇒ death)\n");

    // ── Rung 2 (intervene, do(¬sting)): the road not taken. Non-mutating. ──
    let cfact = g.intervene_on("scorpion", "stings", "nothing");
    println!(
        "Rung 2  intervene : do(scorpion, stings := {})  [{:?}]",
        cfact.triplet.object, cfact.context
    );
    println!("        counterfx : the scorpion stings NOTHING → the frog carries → both CROSS");
    println!("        ⇒ the death was AVOIDABLE.");
    println!(
        "        (the original graph is UNCHANGED: {} triples — the counterfactual is a separate lane)\n",
        g.triplets.len()
    );

    // ── Rung 3 (modal): nature makes do(¬sting) unreachable. ──
    // (The abduce→intervene→predict chain is a gap; we read the modal path.)
    println!("Rung 3  modal     : scorpion --has--> nature --compels--> sting");
    println!("        ⇒ do(¬sting) is unreachable given nature. the death was NECESSARY.\n");

    // ── Synthesis: retain the survival pole, let NARS surface the contradiction. ──
    // The counterfactual "spares" pole shares (subject, object) with the factual
    // "stings" pole but carries the opposite relation → detect_contradictions flags it.
    g.add_triplets(&[Triplet::new("scorpion", "frog", "spares", 6)]);
    let contradictions = g.detect_contradictions(0.0);

    println!("── synthesis (the moral) ──");
    println!(
        "NARS detects {} contradiction(s) on the (scorpion, frog) pair:",
        contradictions.len()
    );
    for &(i, j) in &contradictions {
        let a = &g.triplets[i];
        let b = &g.triplets[j];
        println!(
            "  \"{} {} {}\"   ⟂   \"{} {} {}\"",
            a.subject, a.relation, a.object, b.subject, b.relation, b.object
        );
    }
    println!("\n  The outcome is AVOIDABLE (Rung 2) and NECESSARY (Rung 3) at once.");
    println!("  That contradiction is not a defect to resolve — it IS the moral.");
    println!("  A scalar truth would collapse the tragedy; the contradictory rung ladder");
    println!("  preserves it — the survival pole retained (the -6 counterfactual mantissa,");
    println!("  contract::counterfactual) beside the factual death, addressable by (S,O).");
}
