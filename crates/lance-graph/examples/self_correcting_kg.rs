//! `self_correcting_kg` — W-C of `persistent-nars-kg`: the knowledge graph does
//! NOT start from zero every pass. Re-reading the same text REVISES existing
//! beliefs (NARS confidence accumulates) instead of rebuilding from scratch —
//! the **self-correcting** core of the endgame (a KG, read without an LLM, that
//! reasons about itself). The only difference between the two runs below is a
//! single `TripletGraph::new()`: the naive pass rebuilds from zero and forgets;
//! the self-correcting pass remembers and grows *surer*.
//!
//! ```sh
//! cargo run -p lance-graph --example self_correcting_kg
//! cargo run -p lance-graph --example self_correcting_kg -- /path/to/book.txt
//! ```
//!
//! Scope: this proves the **revise-not-recompute** logic in-process. Durable
//! cross-process persistence (NodeRow → Lance hydrate-on-entry, per the
//! trajectory map) is W-C.2; the mechanism it demonstrates is the same.

use std::path::{Path, PathBuf};

use deepnsm::parser;
use deepnsm::vocabulary::Vocabulary;

use lance_graph::graph::arigraph::triplet_graph::{Triplet, TripletGraph};
use lance_graph::graph::spo::TruthValue;

/// Extract SPO observations from text (the shipped no-LLM pipeline). Each mention
/// is ONE evidence unit at confidence 0.5 — a mention is evidence, not certainty
/// — so NARS revision visibly accumulates confidence on re-observation. (With the
/// default `certain()` at confidence 1.0 the value is already saturated and the
/// accumulation is invisible.)
fn extract(text: &str, vocab: &Vocabulary) -> Vec<Triplet> {
    let mut out = Vec::new();
    for (ts, sentence) in text.split(['.', '!', '?']).enumerate() {
        let s = sentence.trim();
        if s.is_empty() {
            continue;
        }
        let toks = vocab.tokenize(s);
        let structure = parser::parse(&toks);
        for t in &structure.triples {
            let subj = vocab.word(t.subject()).to_string();
            let pred = vocab.word(t.predicate()).to_string();
            let obj = if t.has_object() {
                vocab.word(t.object()).to_string()
            } else {
                String::new()
            };
            if pred == "free" || pred.is_empty() {
                continue;
            }
            out.push(Triplet::with_truth(
                &subj,
                &obj,
                &pred,
                TruthValue::new(1.0, 0.5),
                ts as u64,
            ));
        }
    }
    out
}

/// Confidence of a specific (s, r, o) triplet in the graph, if present.
fn confidence_of(g: &TripletGraph, s: &str, r: &str, o: &str) -> Option<f32> {
    g.triplets
        .iter()
        .find(|t| t.subject == s && t.relation == r && t.object == o)
        .map(|t| t.truth.confidence)
}

fn main() {
    let manifest = env!("CARGO_MANIFEST_DIR");
    let input = std::env::args()
        .nth(1)
        .map(PathBuf::from)
        .unwrap_or_else(|| Path::new(manifest).join("examples/data/aesop_fables.txt"));
    let text =
        std::fs::read_to_string(&input).unwrap_or_else(|e| panic!("read {}: {e}", input.display()));
    let vocab = Vocabulary::load(&Path::new(manifest).join("../deepnsm/word_frequency"))
        .expect("load COCA vocabulary");

    let obs = extract(&text, &vocab);
    // Witness = the first transitive observation (non-empty object); we watch its
    // confidence rise across passes.
    let witness = obs
        .iter()
        .find(|t| !t.object.is_empty())
        .map(|t| (t.subject.clone(), t.relation.clone(), t.object.clone()));

    const PASSES: usize = 3;
    println!("── self_correcting_kg : {} ──", input.display());
    println!(
        "input        : {} SPO observations (each = 0.5-confidence evidence, no LLM)",
        obs.len()
    );

    // ── NAIVE: every pass `new()`s the graph — from zero, no memory. ──
    let mut naive_len = 0usize;
    let mut naive_conf = None;
    for _pass in 0..PASSES {
        let mut g = TripletGraph::new(); // ← from zero, every pass
        for t in &obs {
            g.revise_with_evidence(t);
        }
        naive_len = g.triplets.len();
        naive_conf = witness
            .as_ref()
            .and_then(|(s, r, o)| confidence_of(&g, s, r, o));
    }
    println!(
        "naive        : {PASSES}× TripletGraph::new() → {naive_len} triplets each pass; witness confidence = {} EVERY pass (from zero, no memory)",
        naive_conf.map_or("-".into(), |c| format!("{c:.3}"))
    );

    // ── SELF-CORRECTING: ONE graph, revised across passes — it remembers. ──
    let mut g = TripletGraph::new();
    println!("self-correct : one graph, re-reading the same text {PASSES}×:");
    for pass in 1..=PASSES {
        let before = g.triplets.len();
        for t in &obs {
            g.revise_with_evidence(t);
        }
        let added = g.triplets.len() - before;
        let wconf = witness
            .as_ref()
            .and_then(|(s, r, o)| confidence_of(&g, s, r, o));
        println!(
            "  pass {pass}: {} triplets ({added} NEW), witness confidence = {}",
            g.triplets.len(),
            wconf.map_or("-".into(), |c| format!("{c:.3}"))
        );
    }
    if let Some((s, r, o)) = &witness {
        println!("witness      : \"{s} {r} {o}\" — confidence rises each re-reading (NARS revision); graph size constant after pass 1");
    }
    println!(
        "nars         : {} deductions, {} contradictions on the accumulated graph",
        g.infer_deductions().len(),
        g.detect_contradictions(0.5).len()
    );

    println!("── the point (endgame: self-correcting, no LLM) ──");
    println!("The naive graph rebuilds {naive_len} triplets FROM ZERO every pass and its confidence never moves.");
    println!("The self-correcting graph adds 0 after pass 1, revises in place, and its confidence ACCUMULATES —");
    println!("re-reading the same text makes it *surer*, not busier. Contradictions are retained, not collapsed,");
    println!("so the graph reasons about (and revises) itself. Durable NodeRow→Lance hydration is W-C.2.");
}
