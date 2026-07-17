// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! `community` — structural community detection over the AriGraph
//! [`TripletGraph`](super::triplet_graph::TripletGraph).
//!
//! # Where this sits (doctrine)
//!
//! A **community** is the *structural* partition of the fact graph — the
//! densely-connected regions of the relational field. It **complements**, and
//! does not duplicate, the *episodic / family* basins AriGraph already carries
//! (the `part_of:is_a` rail concept, realised by
//! [`EpisodicEdges64`](lance_graph_contract::episodic_edges) and the
//! `witness_corpus`). Two orthogonal partitions that cross-validate:
//!
//! - a community that **coincides** with a family basin = high-confidence structure;
//! - a community that **crosses** basins = a discovered bridge the episodic
//!   history has not yet captured;
//! - a basin with **no** community = experientially grouped but not structurally
//!   coherent (a revision candidate).
//!
//! Community detection reads the **distributional-meaning field**, not the 1-D
//! similarity ranking: modularity is optimised over the *relational* graph
//! (SPO edges, NARS-truth-weighted), so a community is a **mode of the
//! distribution** — exactly what a scalar rank cannot express. This is the
//! rung-3 layer of the ascent (observation/ranking → SPO hop → community/PPR).
//!
//! Per the workspace litmus (*method on the carrier, not a free function on its
//! state*), the entry point is [`TripletGraph::communities`], not an external
//! service over the graph.
//!
//! # Algorithm
//!
//! Multi-level **Louvain** modularity (Blondel et al. 2008): local-moving to a
//! modularity fixed point, then community aggregation, repeated — yielding a
//! hierarchy. The ΔQ gain uses the canonical `2·m²` denominator (matching
//! `jc/examples/splat_louvain_modularity.rs`). Deterministic: nodes iterate in
//! index order and candidate communities in `BTreeMap` order, so the same graph
//! always yields the same partition — a property an LLM-based clustering cannot
//! offer, and the precondition for certifying it with the jc reliability
//! battery. A **Leiden connectivity refinement** pass (`refine_connected`)
//! follows: any coarsest-level community whose induced subgraph is internally
//! disconnected is split into its connected components, guaranteeing every
//! returned community is internally connected — Leiden's defining property
//! over plain Louvain. Louvain remains the core the refinement operates on.

use std::collections::{BTreeMap, HashMap, VecDeque};

use super::triplet_graph::TripletGraph;

/// A hierarchical community assignment over a [`TripletGraph`]'s entities.
#[derive(Debug, Clone)]
pub struct Communities {
    /// Dense node id → entity name (the inverse of the internal index).
    pub entities: Vec<String>,
    /// Community label per entity at the **coarsest** (final) level.
    pub labels: Vec<u32>,
    /// Hierarchy: `levels[l][node]` = the node's community at level `l`
    /// (level 0 = finest). `levels.last()` equals [`Self::labels`].
    pub levels: Vec<Vec<u32>>,
    /// Modularity `Q` of the coarsest partition (in `[-0.5, 1.0]`).
    pub modularity: f64,
    /// Number of communities at the coarsest level.
    pub num_communities: usize,
}

impl Communities {
    /// The community id of `entity`, if it is in the graph.
    pub fn community_of(&self, entity: &str) -> Option<u32> {
        self.entities
            .iter()
            .position(|e| e == entity)
            .map(|i| self.labels[i])
    }

    /// The entity names in community `community` (coarsest level).
    pub fn members(&self, community: u32) -> Vec<&str> {
        self.entities
            .iter()
            .zip(self.labels.iter())
            .filter_map(|(e, &c)| {
                if c == community {
                    Some(e.as_str())
                } else {
                    None
                }
            })
            .collect()
    }
}

impl TripletGraph {
    /// Detect structural communities over this graph's entities via multi-level
    /// Louvain modularity, followed by a Leiden connectivity refinement pass.
    /// Edges are the non-deleted triplets (subject↔object), weighted by NARS
    /// **confidence**; self-loops (`subject == object`) are dropped.
    /// Deterministic and dependency-free (reads the graph the mailbox already
    /// owns; never a global partition singleton).
    ///
    /// `labels` on the returned [`Communities`] is the **connectivity-refined**
    /// coarsest partition: any community whose induced subgraph came out of
    /// Louvain internally disconnected is split into its connected components,
    /// so every returned community is guaranteed internally connected (Leiden's
    /// defining property over plain Louvain). `levels.last()` is the
    /// **pre-refinement** Louvain partition — the raw coarsest level from
    /// local-moving + aggregation, which may still contain an
    /// internally-disconnected community.
    pub fn communities(&self) -> Communities {
        let (entities, index) = self.entity_index_dense();
        let n = entities.len();
        if n == 0 {
            return Communities {
                entities,
                labels: Vec::new(),
                levels: Vec::new(),
                modularity: 0.0,
                num_communities: 0,
            };
        }
        let adj = self.build_adjacency(&index, n);
        let (levels_u, labels_u, q) = detect(adj.clone(), vec![0.0; n]);
        let louvain_labels: Vec<u32> = labels_u.iter().map(|&x| x as u32).collect();
        let labels = refine_connected(&adj, &louvain_labels);
        let levels: Vec<Vec<u32>> = levels_u
            .iter()
            .map(|lv| lv.iter().map(|&x| x as u32).collect())
            .collect();
        let num_communities = {
            let mut s = labels.clone();
            s.sort_unstable();
            s.dedup();
            s.len()
        };
        Communities {
            entities,
            labels,
            levels,
            modularity: q,
            num_communities,
        }
    }

    /// Dense entity index: a stable `Vec<String>` (sorted for determinism) and
    /// an entity → dense-id map.
    fn entity_index_dense(&self) -> (Vec<String>, HashMap<String, usize>) {
        let mut names: Vec<String> = self.entity_index.keys().cloned().collect();
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
    fn build_adjacency(&self, index: &HashMap<String, usize>, n: usize) -> Vec<Vec<(usize, f64)>> {
        // Accumulate undirected weights in a per-node BTreeMap for determinism.
        let mut acc: Vec<BTreeMap<usize, f64>> = vec![BTreeMap::new(); n];
        for t in &self.triplets {
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
}

// ── pure multi-level Louvain (std only; verified standalone) ────────────────

/// Weighted undirected graph: `adj[u]` = neighbours (`v != u`, weight `w`) on
/// both endpoints; `self_loop[u]` = self weight (aggregation introduces it).
struct WGraph {
    adj: Vec<Vec<(usize, f64)>>,
    self_loop: Vec<f64>,
    degree: Vec<f64>,
    two_m: f64,
}

fn build(adj: Vec<Vec<(usize, f64)>>, self_loop: Vec<f64>) -> WGraph {
    let n = adj.len();
    let mut degree = vec![0.0f64; n];
    for u in 0..n {
        let mut d = 0.0;
        for &(v, w) in &adj[u] {
            if v != u {
                d += w;
            }
        }
        degree[u] = d + 2.0 * self_loop[u];
    }
    let two_m: f64 = degree.iter().sum();
    WGraph {
        adj,
        self_loop,
        degree,
        two_m,
    }
}

/// First-appearance dense relabel — deterministic given node order.
fn relabel(label: &mut [usize]) {
    let mut map: HashMap<usize, usize> = HashMap::new();
    let mut next = 0usize;
    for l in label.iter_mut() {
        let nl = match map.get(l) {
            Some(&x) => x,
            None => {
                let x = next;
                next += 1;
                map.insert(*l, x);
                x
            }
        };
        *l = nl;
    }
}

/// Louvain local-moving to a modularity fixed point. Deterministic.
fn local_move(g: &WGraph) -> Vec<usize> {
    let n = g.adj.len();
    let mut label: Vec<usize> = (0..n).collect();
    if g.two_m <= 0.0 {
        return label;
    }
    let m = g.two_m / 2.0;
    let two_m_sq = 2.0 * m * m;
    let mut comm_deg: Vec<f64> = g.degree.clone();
    let mut improved = true;
    let mut passes = 0;
    while improved && passes < 100 {
        improved = false;
        passes += 1;
        for u in 0..n {
            let from = label[u];
            let k_u = g.degree[u];
            let mut ncw: BTreeMap<usize, f64> = BTreeMap::new();
            for &(v, w) in &g.adj[u] {
                if v == u {
                    continue;
                }
                *ncw.entry(label[v]).or_insert(0.0) += w;
            }
            let k_u_in_from = *ncw.get(&from).unwrap_or(&0.0);
            comm_deg[from] -= k_u; // a_from - k_u
            let gain = |to: usize, k_in: f64, comm_deg: &[f64]| -> f64 {
                k_in / m - k_u * comm_deg[to] / two_m_sq
            };
            let mut best = from;
            let mut best_gain = gain(from, k_u_in_from, &comm_deg);
            for (&c, &k_in) in &ncw {
                let g_c = gain(c, k_in, &comm_deg);
                if g_c > best_gain + 1e-12 {
                    best_gain = g_c;
                    best = c;
                }
            }
            comm_deg[best] += k_u;
            if best != from {
                label[u] = best;
                improved = true;
            }
        }
    }
    relabel(&mut label);
    label
}

/// Aggregate each community into a super-node. Preserves `two_m`.
fn aggregate(g: &WGraph, label: &[usize]) -> WGraph {
    let k = label.iter().copied().max().map(|x| x + 1).unwrap_or(0);
    let mut super_adj: Vec<BTreeMap<usize, f64>> = vec![BTreeMap::new(); k];
    let mut self_w = vec![0.0f64; k];
    for u in 0..g.adj.len() {
        let cu = label[u];
        self_w[cu] += g.self_loop[u];
        for &(v, w) in &g.adj[u] {
            if v == u {
                continue;
            }
            let cv = label[v];
            if cu == cv {
                self_w[cu] += w / 2.0;
            } else {
                *super_adj[cu].entry(cv).or_insert(0.0) += w;
            }
        }
    }
    let adj: Vec<Vec<(usize, f64)>> = super_adj
        .into_iter()
        .map(|m| m.into_iter().collect())
        .collect();
    build(adj, self_w)
}

/// Modularity `Q` of a partition on graph `g`.
fn modularity(g: &WGraph, label: &[usize]) -> f64 {
    if g.two_m <= 0.0 {
        return 0.0;
    }
    let k = label.iter().copied().max().map(|x| x + 1).unwrap_or(0);
    let mut sigma_in = vec![0.0f64; k];
    let mut sigma_tot = vec![0.0f64; k];
    for u in 0..g.adj.len() {
        let c = label[u];
        sigma_tot[c] += g.degree[u];
        sigma_in[c] += 2.0 * g.self_loop[u];
        for &(v, w) in &g.adj[u] {
            if v != u && label[v] == c {
                sigma_in[c] += w;
            }
        }
    }
    let two_m = g.two_m;
    (0..k)
        .map(|c| sigma_in[c] / two_m - (sigma_tot[c] / two_m).powi(2))
        .sum()
}

/// Multi-level Louvain. Returns (hierarchy over ORIGINAL nodes, coarsest
/// labels, `Q` of the coarsest partition on the original graph).
fn detect(adj0: Vec<Vec<(usize, f64)>>, self0: Vec<f64>) -> (Vec<Vec<usize>>, Vec<usize>, f64) {
    let n0 = adj0.len();
    let g0 = build(adj0.clone(), self0.clone());
    let mut g = build(adj0, self0);
    let mut levels: Vec<Vec<usize>> = Vec::new();
    let mut orig_to_super: Vec<usize> = (0..n0).collect();
    for _ in 0..64 {
        let labels = local_move(&g);
        let k = labels.iter().copied().max().map(|x| x + 1).unwrap_or(0);
        let level: Vec<usize> = (0..n0).map(|i| labels[orig_to_super[i]]).collect();
        levels.push(level);
        if k == g.adj.len() {
            break;
        }
        for s in orig_to_super.iter_mut() {
            *s = labels[*s];
        }
        g = aggregate(&g, &labels);
    }
    let coarsest = levels.last().cloned().unwrap_or_else(|| (0..n0).collect());
    let q = modularity(&g0, &coarsest);
    (levels, coarsest, q)
}

/// Split any internally-disconnected community into its connected components
/// (the Leiden connectivity guarantee over Louvain). Deterministic: nodes are
/// visited in dense-index order and components are relabeled by first appearance.
fn refine_connected(adj: &[Vec<(usize, f64)>], labels: &[u32]) -> Vec<u32> {
    let n = adj.len();
    let mut component: Vec<Option<u32>> = vec![None; n];
    let mut next_id: u32 = 0;
    for start in 0..n {
        if component[start].is_some() {
            continue;
        }
        let mut queue: VecDeque<usize> = VecDeque::new();
        queue.push_back(start);
        component[start] = Some(next_id);
        while let Some(u) = queue.pop_front() {
            for &(v, _w) in &adj[u] {
                if v == u || component[v].is_some() {
                    continue;
                }
                if labels[v] == labels[u] {
                    component[v] = Some(next_id);
                    queue.push_back(v);
                }
            }
        }
        next_id += 1;
    }
    component
        .into_iter()
        .map(|c| c.expect("every node visited exactly once"))
        .collect()
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

    #[test]
    fn two_triangles_bridge_yields_two_communities() {
        // {a,b,c} triangle, {d,e,f} triangle, single a-d bridge.
        let g = tg(&[
            ("a", "b"),
            ("b", "c"),
            ("c", "a"),
            ("d", "e"),
            ("e", "f"),
            ("f", "d"),
            ("a", "d"),
        ]);
        let c = g.communities();
        assert_eq!(c.num_communities, 2, "labels: {:?}", c.labels);
        assert!(c.modularity > 0.3, "Q too low: {}", c.modularity);
        assert_eq!(c.community_of("a"), c.community_of("b"));
        assert_eq!(c.community_of("b"), c.community_of("c"));
        assert_ne!(c.community_of("a"), c.community_of("d"));
        assert_eq!(c.members(c.community_of("a").unwrap()).len(), 3);
    }

    #[test]
    fn deterministic() {
        let g = tg(&[
            ("a", "b"),
            ("b", "c"),
            ("c", "a"),
            ("d", "e"),
            ("e", "f"),
            ("f", "d"),
            ("a", "d"),
        ]);
        assert_eq!(g.communities().labels, g.communities().labels);
    }

    #[test]
    fn clique_is_one_community() {
        let g = tg(&[
            ("a", "b"),
            ("a", "c"),
            ("a", "d"),
            ("b", "c"),
            ("b", "d"),
            ("c", "d"),
        ]);
        assert_eq!(g.communities().num_communities, 1);
    }

    #[test]
    fn empty_graph_is_safe() {
        let c = TripletGraph::new().communities();
        assert_eq!(c.num_communities, 0);
        assert!(c.labels.is_empty());
    }

    #[test]
    fn confidence_weight_keeps_strong_edges_together() {
        // Strong a-b-c triangle (conf 1.0) + one weak c-d edge (conf 0.05).
        let mut g = TripletGraph::new();
        let ts = vec![
            Triplet::new("a", "b", "rel", 0),
            Triplet::new("b", "c", "rel", 1),
            Triplet::new("c", "a", "rel", 2),
            Triplet::with_truth(
                "c",
                "d",
                "rel",
                crate::graph::spo::truth::TruthValue::new(1.0, 0.05),
                3,
            ),
        ];
        g.add_triplets(&ts);
        let c = g.communities();
        assert_eq!(c.community_of("a"), c.community_of("b"));
        assert_eq!(c.community_of("b"), c.community_of("c"));
    }

    #[test]
    fn refine_connected_splits_disconnected_community() {
        // Two disjoint edges 0-1 and 2-3, no edge between the pairs, but a
        // (deliberately wrong) Louvain-style label lumps all four into
        // community 0 — the case a real aggregation step can produce.
        let adj: Vec<Vec<(usize, f64)>> = vec![
            vec![(1, 1.0)],
            vec![(0, 1.0)],
            vec![(3, 1.0)],
            vec![(2, 1.0)],
        ];
        let labels = vec![0u32, 0, 0, 0];
        let refined = refine_connected(&adj, &labels);
        assert_eq!(refined[0], refined[1], "0 and 1 share an edge: {refined:?}");
        assert_eq!(refined[2], refined[3], "2 and 3 share an edge: {refined:?}");
        assert_ne!(
            refined[0], refined[2],
            "0-1 and 2-3 are disconnected: {refined:?}"
        );
        let mut distinct = refined.clone();
        distinct.sort_unstable();
        distinct.dedup();
        assert_eq!(
            distinct.len(),
            2,
            "expected exactly 2 components: {refined:?}"
        );
    }

    #[test]
    fn refine_connected_is_idempotent_on_connected_community() {
        // A single connected triangle, already one label - refinement must
        // not introduce a spurious split.
        let adj: Vec<Vec<(usize, f64)>> = vec![
            vec![(1, 1.0), (2, 1.0)],
            vec![(0, 1.0), (2, 1.0)],
            vec![(0, 1.0), (1, 1.0)],
        ];
        let labels = vec![0u32, 0, 0];
        let refined = refine_connected(&adj, &labels);
        let mut distinct = refined.clone();
        distinct.sort_unstable();
        distinct.dedup();
        assert_eq!(
            distinct.len(),
            1,
            "connected community must stay one component: {refined:?}"
        );
        assert_eq!(refined[0], refined[1]);
        assert_eq!(refined[1], refined[2]);
    }
}
