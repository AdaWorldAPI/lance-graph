//! `discovery_to_reasoning` — the **ingestion leg of the endgame**: read a graph's
//! premises OUT OF DATA with no LLM (Aerial+ association-rule discovery), route
//! them into the graph as NARS-truth SPO edges, then let the graph reason about
//! itself to a conclusion the data only IMPLIED.
//!
//! This wires the two halves the workspace had kept apart:
//!   - **discovery** (this crate) — mines `(X → Y)` rules from tabular co-occurrence
//!     via the integer codebook probe (float-free, deterministic), with NARS truth
//!     `(frequency, confidence)` earned from the evidence counts;
//!   - **self-directed reasoning** (the loop below, std-only) — the same shape as
//!     lance-graph's shipped `examples/self_directed_graph.rs` (`TripletGraph` +
//!     `infer_deductions`): transitive NARS deduction to a fixed point. Kept to
//!     ~30 std-only lines here so this crate stays zero-dep; the deduction is
//!     identical (`f = f₁f₂, c = f₁f₂c₁c₂`).
//!
//! ```text
//!   tabular rows (co-occurrence)
//!     ─► extract_rules   (codebook probe picks the nearest consequent per feature,
//!                         then confirms support/confidence on the data)   [REAL discovery]
//!     ─► CandidateTriple::from_rule  →  (s, is_a, o, f, c)  NARS-truth SPO
//!     ─► the graph reasons about itself: transitive deduction to a fixed point
//!     ─► CONCLUSION derived from DISCOVERED premises: `socrates … mortal`
//! ```
//!
//! Nothing is hand-asserted: every `is_a` edge the reasoner chains was MINED from
//! the data by Aerial+ logic, its confidence the discovery's own evidence count.
//! (The multi-path witness — a belief corroborated by an independent path — needs
//! a fork, which splits a subject's rule-confidence below the mining floor; it is
//! demonstrated in the shipped `self_directed_graph.rs`, not re-proven here.)
//!
//! ```sh
//! cargo run --manifest-path crates/lance-graph-arm-discovery/Cargo.toml \
//!           --example discovery_to_reasoning
//! ```

use lance_graph_arm_discovery::{
    extract_rules, CandidateTriple, Dataset, ExtractParams, FeatureSpec, FeedProjector, Item,
    TopKDistance, NARS_PERSONALITY_K,
};

/// A shared entity vocabulary: the SAME string can be a subject (feature 0) in one
/// row and an object (feature 1) in another, so the mined `is_a` edges chain.
const VOCAB: [&str; 5] = ["socrates", "plato", "philosopher", "human", "mortal"];

/// Names a mined rule's items back to entity strings → `(subject, is_a, object)`.
/// This is the "functional routing of ideas": a discovered `(feat0=cat_i → feat1=cat_j)`
/// association becomes the graph edge `VOCAB[i] --is_a--> VOCAB[j]`.
struct EntityProjector;
impl FeedProjector for EntityProjector {
    fn subject(&self, antecedent: &[Item]) -> String {
        VOCAB[antecedent[0].category as usize].to_string()
    }
    fn predicate(&self) -> String {
        "is_a".to_string()
    }
    fn object(&self, consequent: &[Item]) -> String {
        VOCAB[consequent[0].category as usize].to_string()
    }
}

/// A NARS-truth `is_a` fact the graph holds (owned microcopy).
#[derive(Clone)]
struct Fact {
    s: String,
    o: String,
    f: f32,
    c: f32,
    path: String, // "" = mined base fact; else the intermediate it was derived through
}

fn concluded(facts: &[Fact], s: &str, o: &str) -> bool {
    facts.iter().any(|x| x.s == s && x.o == o)
}

fn main() {
    // ── 1. DATA — co-occurrence rows (each subject has ONE parent ⇒ clean 100%
    //    rule confidence; the reasoner supplies the chaining the single-hop mining
    //    cannot). Each pair ×50 = the evidence the mining counts. ──
    let spec = FeatureSpec::new(vec![VOCAB.len() as u32, VOCAB.len() as u32]);
    let pairs = [
        (0u32, 2u32), // socrates    is_a philosopher
        (1, 2),       // plato       is_a philosopher
        (2, 3),       // philosopher is_a human
        (3, 4),       // human       is_a mortal
    ];
    let mut rows = Vec::new();
    for &(s, o) in &pairs {
        for _ in 0..50 {
            rows.push(vec![s, o]);
        }
    }
    let data = Dataset::new(spec.clone(), rows);

    // ── 2. DISCOVER — the REAL Aerial+ codebook probe (float-free, deterministic).
    //    The splat oracle places each subject next to the object it co-occurs with;
    //    extract_rules confirms support/confidence on the data. ──
    let edges: Vec<(Item, Item, u32)> = pairs
        .iter()
        .map(|&(s, o)| (Item::new(0, s), Item::new(1, o), 1))
        .collect();
    let oracle = TopKDistance::new(spec, u32::MAX, &edges);
    let rules = extract_rules(
        &oracle,
        &data,
        &ExtractParams {
            theta: 2,
            max_antecedent: 1,
            min_support_ppm: 100_000,    // ≥10% of the window
            min_confidence_ppm: 700_000, // ≥70% rule confidence
        },
    );

    println!("── discovery_to_reasoning : read the premises from DATA (no LLM), then reason ──\n");
    println!(
        "  Aerial+ mined association rules from {} co-occurrence rows (nothing hand-asserted):",
        pairs.len() * 50
    );

    // ── 3. ROUTE — mined subject→object rules → NARS-truth `is_a` edges (the seed).
    //    Keep only the s→o direction (feature 0 → feature 1); the miner also emits
    //    reversed o→s rules, which are not the `is_a` reading. ──
    let proj = EntityProjector;
    let mut facts: Vec<Fact> = Vec::new();
    for r in &rules {
        let forward = r.antecedent.len() == 1
            && r.consequent.len() == 1
            && r.antecedent[0].feature == 0
            && r.consequent[0].feature == 1;
        if !forward {
            continue;
        }
        let t: CandidateTriple = CandidateTriple::from_rule(r, &proj, NARS_PERSONALITY_K);
        println!(
            "    {:>11} is_a {:<12} f={:.2} c={:.2}  (mined from {} co-occurrences)",
            t.s, t.o, t.f, t.c, r.cooccur
        );
        facts.push(Fact {
            s: t.s,
            o: t.o,
            f: t.f,
            c: t.c,
            path: String::new(),
        });
    }
    let mined = facts.len();

    // ── 4. THE GRAPH REASONS ABOUT ITSELF over the DISCOVERED facts: transitive
    //    NARS deduction (A is_a B, B is_a C ⇒ A is_a C) to a fixed point. ──
    let mut derived: Vec<String> = Vec::new();
    let mut socrates_mortal_conf: Option<f32> = None;
    let mut rested = false;
    for _round in 0..16 {
        let mut fresh: Option<Fact> = None;
        'scan: for a in &facts {
            for b in &facts {
                if a.o != b.s || a.s == b.o {
                    continue; // need A→B, B→C, no self-loop
                }
                if !concluded(&facts, &a.s, &b.o) {
                    fresh = Some(Fact {
                        s: a.s.clone(),
                        o: b.o.clone(),
                        f: a.f * b.f,
                        c: a.f * b.f * a.c * b.c,
                        path: a.o.clone(),
                    });
                    break 'scan;
                }
            }
        }
        match fresh {
            Some(nf) => {
                if nf.s == "socrates" && nf.o == "mortal" {
                    socrates_mortal_conf = Some(nf.c);
                }
                derived.push(format!(
                    "{} is_a {} (via {}) f={:.2} c={:.2}",
                    nf.s, nf.o, nf.path, nf.f, nf.c
                ));
                facts.push(nf);
            }
            None => {
                rested = true;
                break;
            }
        }
    }

    println!("\n  the graph then reasoned about itself over those mined facts:");
    for d in &derived {
        println!("    + {d}");
    }

    // ── MEASUREMENTS (the bridge is load-bearing only if these hold) ──
    let mined_ok = mined >= 4;
    let grew = facts.len() > mined && concluded(&facts, "socrates", "human");
    let syllogism = concluded(&facts, "socrates", "mortal");

    println!(
        "\n  mined ≥4 is_a premises from data (none hand-asserted): {}",
        yn(mined_ok)
    );
    println!(
        "  the graph grew by reasoning ({mined} mined → {} total beliefs): {}",
        facts.len(),
        yn(grew)
    );
    println!(
        "  ★ SYLLOGISM from DISCOVERED premises — `socrates … mortal` (c={:.2}), no LLM: {}",
        socrates_mortal_conf.unwrap_or(0.0),
        yn(syllogism)
    );
    println!("  reached a coherent fixed point (rested): {}", yn(rested));

    let kills: Vec<&str> = [
        (
            mined_ok,
            "fewer than 4 forward rules mined — discovery under-fired",
        ),
        (
            grew,
            "reasoning did not grow the graph past the mined facts",
        ),
        (
            syllogism,
            "failed to derive socrates→mortal from mined premises",
        ),
        (rested, "did not reach a fixed point"),
    ]
    .iter()
    .filter_map(|&(ok, m)| (!ok).then_some(m))
    .collect();

    println!("\n── the bridge, stated plainly ──");
    if kills.is_empty() {
        println!(
            "  The premises were not written by a human or an LLM — Aerial+ MINED them from data"
        );
        println!(
            "  as truth-valued rules, routed them into the graph as is_a edges, and the graph"
        );
        println!(
            "  reasoned over its own mined tissue to a conclusion the data only implied. That is"
        );
        println!(
            "  the ingestion leg of the endgame: reading-without-an-LLM feeding reasoning-about-"
        );
        println!("  itself — discovery and deduction as one float-free, deterministic pipeline.");
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
