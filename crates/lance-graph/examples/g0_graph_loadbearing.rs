// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! G0 — the **P-GRAPH-LOADBEARING** measurement harness (the graphrag plan's gate).
//!
//! The plan (`.claude/plans/graphrag-doc-retrieval-soa-integration-v1.md` §6, §9)
//! makes this probe the *first* deliverable and the gate on all retrieval wiring:
//!
//! > On a document corpus, measure retrieval quality **with vs without** graph
//! > traversal (vector-only vs vector+SPO-G+PPR). KILL condition: if the graph
//! > does not beat vector-only on multi-hop/global questions, do **not** wire
//! > Leiden/PPR into retrieval — the graph would be decorative.
//!
//! This harness exercises the three landed, *reversible* capabilities
//! ([`Bm25Index`], [`TripletGraph::personalized_pagerank`],
//! [`TripletGraph::communities`]) end-to-end so their composition is visible and
//! reproducible. It **prints** the with-vs-without delta; it deliberately does
//! **not** assert a PASS/FAIL, because the real gate is a measurement over a
//! *labeled multi-hop corpus with realistic distractors*, which this synthetic
//! fixture is not. Run it:
//!
//! ```text
//! cargo run -p lance-graph --example g0_graph_loadbearing
//! ```
//!
//! ## The fixture (a controlled multi-hop instance)
//!
//! A 4-fact chain `alice → acme → beta → turbines` plus a disconnected lexical
//! distractor. The gold answer entity (`turbines`) is **lexically absent** from
//! the query yet **graph-reachable** from the lexical seed (`alice`) only by a
//! 3-hop walk. That is precisely the regime where a graph should beat a bag of
//! words — and precisely the regime the KILL condition targets.

use std::collections::BTreeMap;

use lance_graph::graph::arigraph::triplet_graph::{Triplet, TripletGraph};
use lance_graph::graph::arigraph::Bm25Index;

/// `(document text, [(subject, relation, object), ...])` — each document yields
/// the SPO facts the graph is built from, so the entity→document map is exact.
type Corpus = [(
    &'static str,
    &'static [(&'static str, &'static str, &'static str)],
)];

fn rank_of(ranked: &[(String, f64)], entity: &str) -> Option<usize> {
    ranked.iter().position(|(e, _)| e == entity).map(|i| i + 1)
}

fn main() {
    // ── 1. Synthetic corpus + the fact graph each document yields ────────────
    let corpus: &Corpus = &[
        (
            "Alice founded Acme in Berlin.",
            &[
                ("alice", "founded", "acme"),
                ("acme", "located_in", "berlin"),
            ],
        ),
        (
            "Acme acquired Beta Corp last year.",
            &[("acme", "acquired", "beta")],
        ),
        (
            "Beta Corp builds wind turbines.",
            &[("beta", "builds", "turbines")],
        ),
        (
            "Berlin is the capital of Germany.",
            &[("berlin", "capital_of", "germany")],
        ),
        (
            "Carol paints alpine landscapes.", // disconnected distractor
            &[("carol", "paints", "landscapes")],
        ),
    ];

    let docs: Vec<&str> = corpus.iter().map(|(t, _)| *t).collect();
    let mut graph = TripletGraph::new();
    let mut ts = 0u64;
    for (_txt, facts) in corpus {
        let trips: Vec<Triplet> = facts
            .iter()
            .map(|(s, r, o)| {
                let t = Triplet::new(s, o, r, ts);
                ts += 1;
                t
            })
            .collect();
        graph.add_triplets(&trips);
    }

    // entity → documents that mention it (exact, from the fact provenance).
    let mut entity_docs: BTreeMap<&str, Vec<usize>> = BTreeMap::new();
    for (doc_id, (_txt, facts)) in corpus.iter().enumerate() {
        for (s, _r, o) in *facts {
            entity_docs.entry(s).or_default().push(doc_id);
            entity_docs.entry(o).or_default().push(doc_id);
        }
    }

    // ── 2. The multi-hop query + the gold answer entity ──────────────────────
    // "turbines" never appears in the query; it is reachable from "alice" only
    // through alice→acme→beta→turbines.
    let query = "the company Alice founded";
    let gold = "turbines";

    let bm25 = Bm25Index::build(&docs);

    // ── 3a. VECTOR-ONLY (no traversal): score each entity by the best BM25 doc
    //        score among the documents that mention it. Pure lexical. ─────────
    let mut vector_only: Vec<(String, f64)> = entity_docs
        .iter()
        .map(|(&e, doc_ids)| {
            let best = doc_ids
                .iter()
                .map(|&d| bm25.score(query, d))
                .fold(0.0f64, f64::max);
            (e.to_string(), best)
        })
        .collect();
    vector_only.sort_by(|a, b| {
        b.1.partial_cmp(&a.1)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.0.cmp(&b.0))
    });

    // ── 3b. GRAPH-AUGMENTED: BM25 picks the top document; its entities seed a
    //        personalized-PageRank spread over the fact graph. ───────────────
    let top_doc = bm25.rank(query).first().map(|&(d, _)| d).unwrap_or(0);
    let mut seeds: Vec<&str> = corpus[top_doc]
        .1
        .iter()
        .flat_map(|(s, _r, o)| [*s, *o])
        .collect();
    seeds.sort_unstable();
    seeds.dedup();
    let ppr = graph.personalized_pagerank(&seeds, 0.85, 50);
    let graph_ranked: Vec<(String, f64)> = ppr
        .ranked()
        .into_iter()
        .map(|(e, s)| (e.to_string(), s))
        .collect();

    // ── 4. Report the with-vs-without delta ──────────────────────────────────
    println!("== G0 · P-GRAPH-LOADBEARING (synthetic multi-hop scaffold) ==\n");
    println!("query : {query:?}");
    println!("seeds : {seeds:?}  (entities of BM25 top document #{top_doc})");
    println!("gold  : {gold:?}  (lexically absent from the query; 3 hops from \"alice\")\n");

    let print_top = |label: &str, ranked: &[(String, f64)]| {
        println!("{label}");
        for (i, (e, s)) in ranked.iter().take(5).enumerate() {
            println!("  {:>2}. {:<12} {:.4}", i + 1, e, s);
        }
    };
    print_top("vector-only (BM25 lexical, no traversal):", &vector_only);
    print_top("graph-augmented (BM25 seed → PPR spread):", &graph_ranked);

    let r_vec = rank_of(&vector_only, gold);
    let r_graph = rank_of(&graph_ranked, gold);
    println!("\nrank of gold {gold:?}:  vector-only = {r_vec:?}   graph = {r_graph:?}");
    match (r_vec, r_graph) {
        (Some(v), Some(g)) if g < v => println!(
            "  → graph surfaces the multi-hop answer {} place(s) higher (mechanism is load-bearing here).",
            v - g
        ),
        (Some(v), Some(g)) if g == v => println!("  → no change on this fixture."),
        (Some(_), Some(_)) => println!("  → vector-only ranked it higher on this fixture."),
        _ => println!("  → gold missing from one ranking."),
    }

    // Community focus (the "documents in this community" / D-GR-4 direction).
    let comms = graph.communities();
    if let Some(c) = comms.community_of(gold) {
        println!(
            "\ncommunity of {gold:?}: #{c}  members={:?}  (Q={:.3}, {} communities)",
            comms.members(c),
            comms.modularity,
            comms.num_communities
        );
    }

    println!(
        "\nNOTE: synthetic mechanism scaffold, NOT the gate verdict. The real KILL/PASS\n\
         needs a labeled multi-hop corpus with realistic distractors + jc::reliability\n\
         (Spearman/ICC) on the with-vs-without delta. D-GR-2 retrieval wiring stays\n\
         gated on that run (plan §9)."
    );
}
