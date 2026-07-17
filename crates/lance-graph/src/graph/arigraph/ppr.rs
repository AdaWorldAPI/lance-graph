// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! `ppr` — Personalized PageRank (HippoRAG-style spread activation) over the
//! AriGraph [`TripletGraph`](super::triplet_graph::TripletGraph).
//!
//! # Where this sits (doctrine)
//!
//! [`communities`](TripletGraph::communities) partitions the fact graph
//! *structurally* — which entities cluster together. PPR answers a
//! complementary question: *given a seed set, how related is every other
//! entity to it?* Where community detection yields a discrete partition, PPR
//! yields a continuous relevance ranking over the whole graph — the rung-3
//! multi-hop retrieval primitive HippoRAG (Gutiérrez et al. 2024,
//! "HippoRAG: Neurobiologically Inspired Long-Term Memory for LLMs")
//! popularised as a single-step alternative to iterative multi-hop RAG:
//! spread restart mass from seed entities (produced by rung-0/1 similarity
//! ranking) across the SPO fact graph and rank every entity by the mass it
//! accumulates. This is the rung-3 layer of the ascent (observation/ranking
//! → SPO hop → community/PPR), the same rung `community.rs` occupies from
//! the orthogonal, structural side.
//!
//! Per the workspace litmus (*method on the carrier, not a free function on
//! its state*), the entry point is [`TripletGraph::personalized_pagerank`],
//! never an external service over the graph. It reads the same graph the
//! mailbox already owns — no copy, no service call.
//!
//! # Algorithm
//!
//! Classic personalized PageRank via power iteration over a confidence-
//! weighted, row-normalized, undirected transition matrix (Page et al. 1999
//! — the "random surfer" restarting at the seed set instead of uniformly).
//! Deterministic: dense sorted entity index, `BTreeMap`-ordered adjacency
//! accumulation — the identical adjacency discipline
//! [`communities`](TripletGraph::communities) uses — so the same graph and
//! seeds always yield the same scores. A property an LLM-based retriever
//! cannot offer, and the precondition for certifying it with the jc
//! reliability battery. Dangling (degree-0) nodes redistribute their mass
//! into the personalization vector each step rather than leaking it off the
//! graph, so the result is always a proper probability distribution
//! (`scores.iter().sum() ≈ 1.0`).

use std::collections::{BTreeMap, HashMap};

use super::triplet_graph::TripletGraph;

/// Result of a personalized-PageRank spread over the fact graph.
#[derive(Debug, Clone)]
pub struct PersonalizedPageRank {
    /// Dense node id → entity name (parallel to [`Self::scores`]).
    pub entities: Vec<String>,
    /// Restart-mass share per entity, in the same order as [`Self::entities`].
    /// Forms a probability distribution: `scores.iter().sum() ≈ 1.0`.
    pub scores: Vec<f64>,
}

impl PersonalizedPageRank {
    /// The PPR score of `entity`, by exact-name lookup. `None` if `entity`
    /// is not in the graph.
    pub fn score_of(&self, entity: &str) -> Option<f64> {
        self.entities
            .iter()
            .position(|e| e == entity)
            .map(|i| self.scores[i])
    }

    /// All `(entity, score)` pairs, ranked descending by score; ties are
    /// broken by entity name ascending, so the order is fully deterministic.
    pub fn ranked(&self) -> Vec<(&str, f64)> {
        let mut pairs: Vec<(&str, f64)> = self
            .entities
            .iter()
            .map(String::as_str)
            .zip(self.scores.iter().copied())
            .collect();
        pairs.sort_by(|a, b| {
            b.1.partial_cmp(&a.1)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.0.cmp(b.0))
        });
        pairs
    }

    /// The top `k` `(entity, score)` pairs, per [`Self::ranked`] order.
    pub fn top_k(&self, k: usize) -> Vec<(&str, f64)> {
        let mut top = self.ranked();
        top.truncate(k);
        top
    }
}

impl TripletGraph {
    /// Personalized PageRank (HippoRAG spread-activation) over the fact
    /// graph.
    ///
    /// `seeds` are entity names that receive the restart mass — matched
    /// first by exact name and, failing that, case-insensitively against
    /// the entity index. `damping` is the follow-edge probability (`0.85`
    /// is the canonical PageRank value; the teleport-back probability to
    /// the personalization vector is `1.0 - damping`). `iters` is the
    /// power-iteration step count.
    ///
    /// Edges are the non-deleted triplets (subject↔object), weighted by
    /// NARS **confidence**; self-loops (`subject == object`) are dropped —
    /// the identical adjacency discipline
    /// [`communities`](Self::communities) uses. If none of `seeds` match an
    /// entity in the graph, the personalization vector falls back to
    /// uniform over every entity rather than producing an empty spread.
    ///
    /// Deterministic and dependency-free (reads the graph the mailbox
    /// already owns; never a global ranking singleton). The returned scores
    /// always sum to approximately `1.0`; an empty graph returns an empty
    /// result rather than panicking.
    pub fn personalized_pagerank(
        &self,
        seeds: &[&str],
        damping: f64,
        iters: usize,
    ) -> PersonalizedPageRank {
        let (entities, index) = dense_entity_index(self);
        let n = entities.len();
        if n == 0 {
            return PersonalizedPageRank {
                entities,
                scores: Vec::new(),
            };
        }

        let adj = weighted_adjacency(self, &index, n);
        let degree: Vec<f64> = adj
            .iter()
            .map(|row| row.iter().map(|&(_, w)| w).sum())
            .collect();
        let restart = personalization_vector(&entities, &index, seeds, n);

        let mut r = restart.clone();
        for _ in 0..iters {
            r = power_step(&adj, &degree, &restart, &r, damping, n);
        }
        normalize_to_unit_sum(&mut r);

        PersonalizedPageRank {
            entities,
            scores: r,
        }
    }
}

// ── pure PPR core (std only; verified standalone) ───────────────────────────
//
// Free functions rather than `TripletGraph` methods: inherent-impl method
// names are unique per type across the whole crate regardless of privacy, so
// duplicating `community.rs`'s private `entity_index_dense` /
// `build_adjacency` method names here would collide. These operate on
// `TripletGraph`'s public `triplets` / `entity_index` fields directly.

/// Dense entity index: a stable `Vec<String>` (sorted for determinism) and
/// an entity → dense-id map — the same shape
/// [`communities`](TripletGraph::communities) builds internally.
fn dense_entity_index(g: &TripletGraph) -> (Vec<String>, HashMap<String, usize>) {
    let mut names: Vec<String> = g.entity_index.keys().cloned().collect();
    names.sort_unstable();
    let index: HashMap<String, usize> = names
        .iter()
        .enumerate()
        .map(|(i, e)| (e.clone(), i))
        .collect();
    (names, index)
}

/// Weighted undirected adjacency (both endpoints), weight = summed NARS
/// confidence over the triplets joining two entities. Deleted triplets and
/// self-loops are skipped.
fn weighted_adjacency(
    g: &TripletGraph,
    index: &HashMap<String, usize>,
    n: usize,
) -> Vec<Vec<(usize, f64)>> {
    // Accumulate undirected weights in a per-node BTreeMap for determinism.
    let mut acc: Vec<BTreeMap<usize, f64>> = vec![BTreeMap::new(); n];
    for t in &g.triplets {
        if t.is_deleted() || t.subject == t.object {
            continue;
        }
        let (Some(&u), Some(&v)) = (index.get(&t.subject), index.get(&t.object)) else {
            continue;
        };
        let w = t.truth.confidence.max(0.0) as f64;
        if w <= 0.0 {
            continue;
        }
        *acc[u].entry(v).or_insert(0.0) += w;
        *acc[v].entry(u).or_insert(0.0) += w;
    }
    acc.into_iter().map(|m| m.into_iter().collect()).collect()
}

/// The restart / personalization vector: uniform mass over `seeds` matched
/// against `index` (exact name first, then a case-insensitive scan of
/// `entities`); uniform over every entity when nothing matches (including
/// an empty `seeds`).
fn personalization_vector(
    entities: &[String],
    index: &HashMap<String, usize>,
    seeds: &[&str],
    n: usize,
) -> Vec<f64> {
    let mut matched: Vec<usize> = Vec::new();
    for &seed in seeds {
        if let Some(&id) = index.get(seed) {
            matched.push(id);
            continue;
        }
        let seed_lower = seed.to_lowercase();
        if let Some(id) = entities.iter().position(|e| e.to_lowercase() == seed_lower) {
            matched.push(id);
        }
    }
    matched.sort_unstable();
    matched.dedup();

    let mut p = vec![0.0f64; n];
    if matched.is_empty() {
        p.fill(1.0 / n as f64);
    } else {
        let share = 1.0 / matched.len() as f64;
        for id in matched {
            p[id] = share;
        }
    }
    p
}

/// One power-iteration step:
/// `r' = (1 - damping)·p + damping·(Pᵀr + dangling_mass·p)`, where `P` is
/// the row-normalized transition matrix (`P(u→v) = w(u,v) / degree(u)`) and
/// dangling (degree-0) nodes redistribute their mass into `p` each step
/// instead of leaking it off the graph.
fn power_step(
    adj: &[Vec<(usize, f64)>],
    degree: &[f64],
    p: &[f64],
    r: &[f64],
    damping: f64,
    n: usize,
) -> Vec<f64> {
    let mut r_next = vec![0.0f64; n];
    let mut dangling_mass = 0.0f64;
    for u in 0..n {
        if degree[u] <= 0.0 {
            dangling_mass += r[u];
            continue;
        }
        let share = r[u] / degree[u];
        for &(v, w) in &adj[u] {
            r_next[v] += share * w;
        }
    }
    for x in 0..n {
        r_next[x] = (1.0 - damping) * p[x] + damping * (r_next[x] + dangling_mass * p[x]);
    }
    r_next
}

/// Rescale `r` to sum to `1.0`. A no-op on an all-zero vector, which cannot
/// occur here: `r` always carries the redistributed restart mass, and `p`
/// (the fallback when `r` sums to zero) always sums to `1.0` by
/// construction.
fn normalize_to_unit_sum(r: &mut [f64]) {
    let sum: f64 = r.iter().sum();
    if sum > 0.0 {
        for x in r.iter_mut() {
            *x /= sum;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::arigraph::triplet_graph::Triplet;

    fn tg(edges: &[(&str, &str)]) -> TripletGraph {
        let mut g = TripletGraph::new();
        let ts: Vec<Triplet> = edges
            .iter()
            .enumerate()
            .map(|(i, (s, o))| Triplet::new(s, o, "rel", i as u64))
            .collect();
        g.add_triplets(&ts);
        g
    }

    /// {a,b,c} triangle, {d,e,f} triangle, single a-d bridge — the same
    /// shape `community.rs`'s fixture uses, so the two rung-3 primitives are
    /// exercised on a directly comparable graph.
    fn two_triangles_bridge() -> TripletGraph {
        tg(&[
            ("a", "b"),
            ("b", "c"),
            ("c", "a"),
            ("d", "e"),
            ("e", "f"),
            ("f", "d"),
            ("a", "d"),
        ])
    }

    #[test]
    fn seed_favors_its_own_triangle_over_the_far_one() {
        let g = two_triangles_bridge();
        let ppr = g.personalized_pagerank(&["a"], 0.85, 50);
        let near_min = ["a", "b", "c"]
            .iter()
            .map(|e| ppr.score_of(e).expect("entity present"))
            .fold(f64::INFINITY, f64::min);
        let far_max = ["d", "e", "f"]
            .iter()
            .map(|e| ppr.score_of(e).expect("entity present"))
            .fold(f64::NEG_INFINITY, f64::max);
        assert!(
            near_min > far_max,
            "near_min={near_min} far_max={far_max} ranked={:?}",
            ppr.ranked()
        );
    }

    #[test]
    fn scores_sum_to_one() {
        let g = two_triangles_bridge();
        let ppr = g.personalized_pagerank(&["a"], 0.85, 50);
        let sum: f64 = ppr.scores.iter().sum();
        assert!((sum - 1.0).abs() < 1e-9, "sum={sum}");
    }

    #[test]
    fn deterministic() {
        let g = two_triangles_bridge();
        let a = g.personalized_pagerank(&["a"], 0.85, 50);
        let b = g.personalized_pagerank(&["a"], 0.85, 50);
        assert_eq!(a.entities, b.entities);
        assert_eq!(a.scores, b.scores);
    }

    #[test]
    fn empty_graph_is_safe() {
        let g = TripletGraph::new();
        let ppr = g.personalized_pagerank(&["anything"], 0.85, 50);
        assert!(ppr.entities.is_empty());
        assert!(ppr.scores.is_empty());
    }

    #[test]
    fn unmatched_seed_falls_back_to_uniform_without_panicking() {
        let g = two_triangles_bridge();
        let ppr = g.personalized_pagerank(&["nonexistent"], 0.85, 50);
        assert_eq!(ppr.scores.len(), 6);
        let sum: f64 = ppr.scores.iter().sum();
        assert!((sum - 1.0).abs() < 1e-9, "sum={sum}");
    }

    #[test]
    fn seeds_own_node_is_top_ranked() {
        let g = two_triangles_bridge();
        let ppr = g.personalized_pagerank(&["a"], 0.85, 50);
        assert_eq!(ppr.ranked()[0].0, "a");
    }
}
