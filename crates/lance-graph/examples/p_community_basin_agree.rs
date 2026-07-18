// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! P-COMMUNITY-BASIN-AGREE — the **empirical** probe for the operator's identity
//! ruling: *`part_of:is_a` category ≡ Leiden community ≡ episodic-witness basin —
//! one concept* (graphrag plan §3b.1).
//!
//! # Empirical, not scientific (the layer boundary)
//!
//! This harness is **empirical**: it sets up a measurement over real graph
//! structure and reports an observation. The **statistic** it reports is
//! `jc`'s — the certified scientific crate ([`jc::reliability::pearson`] /
//! [`spearman`]). The empirical construction (the two partitions, their pairwise
//! co-membership vectors) is this probe's; the science is consumed, never
//! re-implemented or extended here. jc has no partition-agreement metric
//! (Rand/ARI/NMI); if the empirical reading turns out to need one, that is an
//! **observation reported to the scientific track**, not a metric bolted on from
//! the empirical side.
//!
//! # What it measures
//!
//! Two partitions of the same entity set:
//! - **Community** — [`TripletGraph::communities`] (multi-level Louvain +
//!   Leiden refinement) over the *whole* relational field (all edges).
//! - **Basin** — the connected components under **only** the `is_a`/`part_of`
//!   edges (the inherited/taxonomic grouping the plan names as the basin rail).
//!
//! Agreement = Pearson (φ) of the two partitions' pairwise co-membership vectors
//! (`same-community?` vs `same-basin?` over every entity pair). φ = 1.0 iff the
//! partitions are identical up to relabeling — the **identity** the ruling
//! predicts. φ < 1.0 localises the disagreements: those entities are exactly the
//! *discovered bridges / revision candidates* (a community that crosses a basin).
//!
//! # The gate it feeds
//!
//! On a real `is_a`-annotated corpus this number is the S1 gate the other
//! session's D-TRI-1 classid-half mint waits on: **identity (φ→1) ⇒ community-id
//! collapses into the `is_a` rail, no new tenant minted; distinct (φ<1) ⇒
//! community-id earns its place in the batched mint.** This synthetic fixture
//! exercises the *mechanism*; the verdict needs the real corpus.
//!
//! Run: `cargo run -p lance-graph --example p_community_basin_agree`

use std::collections::HashMap;

use jc::reliability::{pearson, spearman};
use lance_graph::graph::arigraph::triplet_graph::{Triplet, TripletGraph};

/// A `(subject, relation, object)` fact.
type Fact = (&'static str, &'static str, &'static str);

/// Relations that define the **basin** rail (the inherited/taxonomic grouping).
fn is_taxonomic(relation: &str) -> bool {
    let r = relation.to_lowercase();
    r.contains("is_a") || r.contains("is a") || r.contains("part_of") || r.contains("part of")
}

fn build_graph(facts: &[Fact]) -> TripletGraph {
    let mut g = TripletGraph::new();
    let ts: Vec<Triplet> = facts
        .iter()
        .enumerate()
        .map(|(i, (s, r, o))| Triplet::new(s, o, r, i as u64))
        .collect();
    g.add_triplets(&ts);
    g
}

/// Basin partition: dense connected-component ids over ONLY the `is_a`/`part_of`
/// edges, aligned to `entities` order. Entities in no taxonomic edge are their
/// own basin. A deterministic union-find (find with path compression, union by
/// lower root so ids are stable given the sorted `entities`).
fn basin_labels(graph: &TripletGraph, entities: &[String]) -> Vec<u32> {
    let index: HashMap<&str, usize> = entities
        .iter()
        .enumerate()
        .map(|(i, e)| (e.as_str(), i))
        .collect();
    let mut parent: Vec<usize> = (0..entities.len()).collect();

    fn find(parent: &mut [usize], mut x: usize) -> usize {
        while parent[x] != x {
            parent[x] = parent[parent[x]];
            x = parent[x];
        }
        x
    }

    for t in &graph.triplets {
        if t.is_deleted() || !is_taxonomic(&t.relation) {
            continue;
        }
        let (Some(&a), Some(&b)) = (index.get(t.subject.as_str()), index.get(t.object.as_str()))
        else {
            continue;
        };
        let (ra, rb) = (find(&mut parent, a), find(&mut parent, b));
        if ra != rb {
            // union by lower root → deterministic given sorted entities.
            let (lo, hi) = (ra.min(rb), ra.max(rb));
            parent[hi] = lo;
        }
    }

    // Densify roots to 0..k by first appearance (deterministic).
    let mut root_to_dense: HashMap<usize, u32> = HashMap::new();
    let mut next = 0u32;
    (0..entities.len())
        .map(|i| {
            let r = find(&mut parent, i);
            *root_to_dense.entry(r).or_insert_with(|| {
                let d = next;
                next += 1;
                d
            })
        })
        .collect()
}

/// Pairwise co-membership over all `i<j` pairs: `1.0` iff `label[i]==label[j]`.
/// This binary vector is the empirical bridge to jc's continuous correlation.
fn comembership(labels: &[u32]) -> Vec<f64> {
    let n = labels.len();
    let mut v = Vec::with_capacity(n * (n - 1) / 2);
    for i in 0..n {
        for j in (i + 1)..n {
            v.push(if labels[i] == labels[j] { 1.0 } else { 0.0 });
        }
    }
    v
}

fn distinct(labels: &[u32]) -> usize {
    let mut s = labels.to_vec();
    s.sort_unstable();
    s.dedup();
    s.len()
}

fn measure(title: &str, facts: &[Fact]) {
    let graph = build_graph(facts);
    let comms = graph.communities();
    let community = comms.labels.clone();
    let basin = basin_labels(&graph, &comms.entities);

    let comm_co = comembership(&community);
    let basin_co = comembership(&basin);

    // Scientific statistic — jc, consumed as-is.
    let phi = pearson(&comm_co, &basin_co);
    let rho = spearman(&comm_co, &basin_co);
    // Empirical descriptive — raw fraction of pairs that agree (Rand-like).
    let agree_frac = if comm_co.is_empty() {
        f64::NAN
    } else {
        comm_co
            .iter()
            .zip(&basin_co)
            .filter(|(a, b)| a == b)
            .count() as f64
            / comm_co.len() as f64
    };

    // Bridges = entities whose basin differs from the MAJORITY basin of their
    // community — the minority member a community pulled across a basin line
    // (the discovered bridge / revision candidate, plan §3b.1).
    let mut comm_basin_counts: HashMap<u32, HashMap<u32, usize>> = HashMap::new();
    for i in 0..community.len() {
        *comm_basin_counts
            .entry(community[i])
            .or_default()
            .entry(basin[i])
            .or_insert(0) += 1;
    }
    let majority_basin: HashMap<u32, u32> = comm_basin_counts
        .iter()
        .map(|(&c, counts)| {
            let maj = counts
                .iter()
                .max_by_key(|(_, &n)| n)
                .map(|(&b, _)| b)
                .unwrap_or(0);
            (c, maj)
        })
        .collect();
    let bridges: Vec<&str> = comms
        .entities
        .iter()
        .enumerate()
        .filter(|&(i, _)| basin[i] != majority_basin[&community[i]])
        .map(|(_, e)| e.as_str())
        .collect();

    println!("── {title} ──");
    println!(
        "  entities={}  communities={}  basins={}",
        comms.entities.len(),
        distinct(&community),
        distinct(&basin)
    );
    println!(
        "  φ (jc::pearson) = {}   ρ (jc::spearman) = {}   pair-agreement = {:.3}",
        phi.map(|x| format!("{x:.4}"))
            .unwrap_or_else(|| "n/a (constant)".into()),
        rho.map(|x| format!("{x:.4}"))
            .unwrap_or_else(|| "n/a (constant)".into()),
        agree_frac,
    );
    match phi {
        Some(p) if p > 0.999 => {
            println!("  → IDENTITY on this fixture (φ≈1): community ≡ basin, no bridges.")
        }
        Some(_) => println!(
            "  → DISTINCT-with-bridges: {:?} cross a basin (the revision candidates).",
            bridges
        ),
        None => println!("  → degenerate (a partition is trivial); not informative."),
    }
    println!();
}

fn main() {
    println!("== P-COMMUNITY-BASIN-AGREE (empirical probe; statistic = jc science) ==\n");

    // Scenario A — clean: two taxonomic basins, structure matches exactly.
    // Expect φ≈1.0 (the identity the ruling predicts, cleanly demonstrated).
    let clean: &[Fact] = &[
        ("cat", "is_a", "pet"),
        ("dog", "is_a", "pet"),
        ("hamster", "is_a", "pet"),
        ("cat", "relates_to", "dog"),
        ("dog", "relates_to", "hamster"),
        ("hamster", "relates_to", "cat"),
        ("oak", "is_a", "tree"),
        ("pine", "is_a", "tree"),
        ("birch", "is_a", "tree"),
        ("oak", "relates_to", "pine"),
        ("pine", "relates_to", "birch"),
        ("birch", "relates_to", "oak"),
    ];
    measure("A · aligned (identity expected)", clean);

    // Scenario B — one deliberate cross: `robot` is_a pet (basin=pet) but is
    // structurally pulled entirely into the tree cluster (community=tree).
    // Expect φ<1.0 with `robot` named as the single bridge / revision candidate.
    let mut bridged: Vec<Fact> = clean.to_vec();
    bridged.extend_from_slice(&[
        ("robot", "is_a", "pet"),
        ("robot", "relates_to", "oak"),
        ("robot", "relates_to", "pine"),
        ("robot", "relates_to", "birch"),
    ]);
    measure("B · one cross-basin bridge (distinct expected)", &bridged);

    println!(
        "NOTE: synthetic mechanism, NOT the verdict. The S1 gate reads φ on a REAL\n\
         is_a-annotated corpus: identity (φ→1) ⇒ community-id collapses into the\n\
         is_a rail, NO new tenant minted; distinct (φ<1) ⇒ community-id earns the\n\
         batched D-TRI-1 mint. jc supplies the statistic; if a Rand/ARI/NMI metric\n\
         is wanted, that is an observation for the scientific (jc) track."
    );
}
