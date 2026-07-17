// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! `doc_graph` — the zero-dep **read surface** over the calcified
//! document / fact graph (D-GR-1).
//!
//! # Role — one query surface, two readers
//!
//! Per the graphrag integration plan (§0/§4a), "feeds both ogar-doc and
//! graphrag" is honestly a **single query/view surface** over the calcified
//! document-graph, consumed by two readers:
//!
//! - **graphrag retrieval** — the AriGraph carrier (`TripletGraph` +
//!   `OsintRetriever`) *implements* this trait; and
//! - **OGAR `ogar-doc`** — `reconstruct_document` / "documents in this
//!   community" *calls* this trait.
//!
//! Because `ogar-doc` reaches the graph **through this contract trait**, OGAR
//! keeps depending only on `lance-graph-contract` (never on the AriGraph impl):
//! the dependency direction stays consumer → contract, never the reverse.
//! **Neither reader ingests; both read.**
//!
//! # Litmus — a carrier method, not a service
//!
//! Every method takes `&self` and returns owned results — no `&mut self`, no
//! lifecycle, no config injection, no borrow into any graph. This is a trait an
//! AriGraph carrier *implements* (`impl DocGraphQuery for TripletGraph`), not a
//! service wrapped around the graph. It carries the **stateless** rung→walk
//! *mapping*; the **stateful** rung *elevator* (the #708 BLOCK/FLOW streak
//! machine) lives in the impl (D-GR-2, spec below), never in the contract.
//!
//! # Zero-dep
//!
//! std only. Ids are `String` (a `DocumentID` or an AriGraph entity name),
//! scores `f32`, community ids `u32`. **No lance-graph / arigraph types cross
//! this surface** — the contract is dependency-free. The one in-crate type used
//! is [`RungLevel`](crate::cognitive_shader::RungLevel) (also zero-dep, #708).
//!
//! # Usage (both readers)
//!
//! ```rust,ignore
//! use lance_graph_contract::cognitive_shader::RungLevel;
//! use lance_graph_contract::doc_graph::DocGraphQuery;
//!
//! // graphrag: rung-aware retrieval (the elevator picks the rung; see D-GR-2).
//! let seeds = vec!["acme_corp".to_string()];
//! let hits = graph.retrieve(&seeds, RungLevel::Contextual, 20);
//!
//! // ogar-doc: "documents in this topic community" for reconstruct_document.
//! if let Some(topic) = graph.community_of("invoice_2026_07") {
//!     let siblings = graph.community_members(topic);
//! }
//! ```
//!
//! ════════════════════════════════════════════════════════════════════════
//! # D-GR-2 design spec — binding `OsintRetriever` to the #708 `RungElevator`
//! ════════════════════════════════════════════════════════════════════════
//!
//! **This is a SPEC. D-GR-2 code is a follow-up in
//! `crates/lance-graph/src/graph/arigraph/retrieval.rs`.** It records exactly
//! how the retriever composes *existing* methods under the #708 elevator so the
//! follow-up PR has no design latitude.
//!
//! ## Owner & elevator
//!
//! `OsintRetriever` (retrieval.rs) holds one
//! [`RungElevator`](crate::cognitive_shader::RungElevator) (#708, merged
//! `8d3209c`, `E-RUNG-ASCENT-WIRED-1`), constructed `RungElevator::new(base)` at
//! the dispatched base rung. graphrag **never re-decides the level** — it reads
//! the elevator's `RungLevel` and supplies the wider graph walk. This is the
//! anti-decorative-graph guarantee: the graph is load-bearing *because* BLOCK
//! ascends the elevator.
//!
//! **Carrier.** The full `DocGraphQuery` surface is implemented on the
//! **retriever** (`OsintRetriever`, extended to hold the graph + a CAM-PQ handle
//! + the small out-of-graph BM25 leg), NOT on `TripletGraph` alone:
//! `similar_by_ranking` needs the ranking leg `TripletGraph` does not own. The
//! graph-side methods (`community_*`, `neighbours`) delegate to `self.graph.*`.
//!
//! ## Per-cycle loop (advance-before-sinks, review fix `17368ea`)
//!
//! 1. **Compute the cycle `GateDecision`** (`crate::collapse_gate::GateDecision`):
//!    - `TripletGraph::detect_contradictions(conf_threshold)` **non-empty** →
//!      `GateDecision::BLOCK` — the natural BLOCK trigger (retrieval surprise /
//!      conflicting facts, `triplet_graph.rs`).
//!    - Low mean NARS `truth.expectation()` over the walk, or an empty /
//!      low-cohesion result → also `BLOCK` (measurement ran out).
//!    - Contradictions empty **and** mean expectation high **and** the result
//!      is stable across the cycle → `GateDecision::FLOW_BUNDLE` (converged).
//!    - else `GateDecision::HOLD`.
//! 2. **Advance FIRST:** `let rung = elevator.on_gate(gate);` — the `17368ea`
//!    ordering lock: advance the elevator *before* the sinks, then gate the walk
//!    on the **post-advance** rung.
//! 3. **Gate the walk on `rung`** via [`DocGraphQuery::retrieve`] (implemented
//!    for the carrier), which dispatches breadth by rung — the Maslow ascent
//!    (`triangle-tenants-gestalt-separation-v1` §3a):
//!
//!    | Rung | Retrieval action | Existing method composed | Predicate mask |
//!    |---|---|---|---|
//!    | 0–1 (base, FLOW) | CAM-PQ vector + BM25 surface **ranking** | [`similar_by_ranking`](DocGraphQuery::similar_by_ranking) (impl = CAM-PQ ADC entry + the small out-of-graph BM25 leg) | identity |
//!    | 2 | **SPO-G edge hop** | `TripletGraph::get_associated(seeds, 2)` → [`neighbours`](DocGraphQuery::neighbours) | widen (`elevator.causal_mask_bits()` = `0b011`, Pearl L2) |
//!    | 3 | **community-scoped PPR** | `TripletGraph::communities()` (`Communities::{community_of, members}`) picks the subgraph, then the forthcoming `ppr.rs` (reset-distribution atop `blasgraph::hdr_pagerank` + `ScentCsr::spmv`) ranks within it | wider CAUSES..BECOMES union |
//!    | 4 (apex) | **Pearl rung-2 intervention** | `TripletGraph::intervene_on(subject, predicate, new_object)` → `CounterfactualSpoG` (`ContextTag::Intervention` G-slot) | — |
//!
//! 4. **Revise:** after the outcome, `TripletGraph::revise_with_evidence(&obs)`
//!    (NARS revision) updates truth and feeds the NEXT cycle's gate;
//!    `infer_deductions()` is an optional mid-rung 2-hop signal.
//! 5. **Relax:** a FLOW streak (≥ `threshold`) relaxes `elevator.level` one rung
//!    toward `base` (never below) — the walk narrows back to ranking; a sustained
//!    BLOCK streak elevates → a wider walk. `RetrievalConfig.max_depth` becomes
//!    **rung-derived** (rung 2 → depth 2, rung 3 → depth 3), replacing the fixed
//!    `max_depth: 2`.
//!
//! ## Predicate-plane widen (SECONDARY leg)
//!
//! `elevator.causal_mask_bits()` (the P2/P3-certified S/P/O mask) feeds
//! `cognitive-shader-driver::driver::rung_widened_layer_mask(base, level,
//! req_mask) -> u8` (`driver.rs:701`, currently **private** → promote `pub` /
//! move beside `RungElevator`, or replicate the pure `(base, level, mask)->u8`
//! — §10 caveat 2). The rung ASCENT works without it; the mask only sharpens
//! which of the 8 predicate planes the walk reads.
//!
//! ## Settlement probe (P-RUNG-RETRIEVAL, §6, mirrors #708 D-TRI-6)
//!
//! Hard / contradictory query → `detect_contradictions` non-empty → BLOCK →
//! elevator ascends → wider mask + wider walk → higher recall. Easy query →
//! FLOW → stays at base (identity mask, cheap ranking). BLOCK ascends, FLOW
//! relaxes to base.

use crate::cognitive_shader::RungLevel;

/// A retrieval hit: an entity / document id, a relevance score, and the hop
/// depth it surfaced at.
///
/// Fully owned (the `String` id) so the contract stays zero-dep — no borrow
/// into any graph, no lance-graph / arigraph type.
#[derive(Clone, Debug, PartialEq)]
pub struct ScoredId {
    /// Entity or document id — a stable string handle (e.g. a `DocumentID` or
    /// an AriGraph entity name). Never a lance-graph / arigraph type.
    pub id: String,
    /// Relevance score, **higher = more relevant** (a CAM-PQ similarity, a NARS
    /// truth expectation, or a PPR mass, per the impl). Not normalised across
    /// methods — compare only *within* one result list.
    pub score: f32,
    /// Hops from the nearest seed the hit surfaced at: `0` = a direct /
    /// base-rank hit ([`DocGraphQuery::similar_by_ranking`]), `1..` = a
    /// multi-hop graph inference. Provenance for the rung story — a deeper hit
    /// is a weaker, more inferential match.
    pub depth: u8,
}

impl ScoredId {
    /// A hit at an explicit hop `depth`.
    pub fn new(id: impl Into<String>, score: f32, depth: u8) -> Self {
        Self {
            id: id.into(),
            score,
            depth,
        }
    }

    /// A base / direct hit (`depth == 0`) — the ranking leg's shape.
    pub fn base(id: impl Into<String>, score: f32) -> Self {
        Self::new(id, score, 0)
    }
}

/// A zero-dep read surface over the document / fact graph.
///
/// Implemented by the AriGraph carrier (graphrag retrieval); consumed by OGAR
/// `ogar-doc` **through this contract** (never the impl). See the module docs
/// for the doctrine and the D-GR-2 elevator-binding spec.
///
/// The five required methods are the retrieval **primitives**; the provided
/// [`retrieve`](DocGraphQuery::retrieve) composes them into the stateless
/// rung→walk *mapping*. Impls SHOULD override `retrieve` for the real
/// CAM-PQ / PPR path — the default is a transparent reference composition over
/// the primitives.
///
/// The carrier is the **retriever** (`OsintRetriever`, which holds the graph +
/// the ranking leg), not `TripletGraph` alone — `similar_by_ranking` needs the
/// CAM-PQ / BM25 ranking `TripletGraph` does not own; the graph-side methods
/// delegate to `self.graph.*`.
///
/// ```rust,ignore
/// use lance_graph_contract::doc_graph::{DocGraphQuery, ScoredId};
///
/// impl DocGraphQuery for OsintRetriever<'_> {
///     fn community_of(&self, id: &str) -> Option<u32> {
///         self.graph.communities().community_of(id)
///     }
///     fn neighbours(&self, seeds: &[String], hops: u8) -> Vec<ScoredId> {
///         let set = seeds.iter().cloned().collect();
///         self.graph.get_associated(&set, hops as usize).iter()
///             .map(|t| ScoredId::new(t.object.clone(), t.truth.expectation(), 1))
///             .collect()
///     }
///     fn similar_by_ranking(&self, id: &str, top_k: usize) -> Vec<ScoredId> {
///         // the CAM-PQ ADC / BM25 leg the retriever owns (not TripletGraph)
///         self.rank_similar(id, top_k)
///     }
///     // community_ids / community_members ...
/// }
/// ```
pub trait DocGraphQuery {
    /// The community id `id` belongs to (structural Louvain partition), or
    /// `None` when the id is not in the graph.
    fn community_of(&self, id: &str) -> Option<u32>;

    /// The distinct community ids at the coarsest partition level, ascending.
    /// Lets a consumer (`ogar-doc`) enumerate topics — "for each community,
    /// reconstruct" — without a separate handle.
    fn community_ids(&self) -> Vec<u32>;

    /// The member ids of `community` (coarsest level). The "documents in this
    /// community" seam `ogar-doc`'s related-docs / reconstruct path consumes.
    fn community_members(&self, community: u32) -> Vec<String>;

    /// Multi-hop structural neighbours reachable within `hops` from `seeds` —
    /// the SPO-G walk (the impl composes `TripletGraph::get_associated`). Scored
    /// by the impl (NARS truth expectation, hop proximity, …); `hops` is small
    /// (1–4 typically).
    fn neighbours(&self, seeds: &[String], hops: u8) -> Vec<ScoredId>;

    /// Similar-by-ranking: the **256-ranking** leg (CAM-PQ / distributional
    /// similarity) — no graph walk, the base rung. Best-first, at most `top_k`.
    fn similar_by_ranking(&self, id: &str, top_k: usize) -> Vec<ScoredId>;

    /// Rung-aware retrieve: the #708 `RungLevel` selects the walk **breadth**,
    /// best-first, at most `top_k`.
    ///
    /// The default is the stateless rung→walk mapping (D-GR-2 minus the
    /// elevator), composed from the primitives:
    ///
    /// - **rung 0–1** — ranking: the union of each seed's
    ///   [`similar_by_ranking`](DocGraphQuery::similar_by_ranking) (no walk).
    /// - **rung 2** — the 2-hop [`neighbours`](DocGraphQuery::neighbours) walk
    ///   (SPO-G hop).
    /// - **rung 3+** — community-scoped: a wider (`rung`-hop, capped) walk,
    ///   focused to the hits sharing a seed's community (community picks the
    ///   subgraph; the real PPR ranks within it — impls override for `ppr.rs`).
    ///
    /// Widening the underlying walk with the rung is superset-monotone; the
    /// rung-3 community focus is a relevance refinement over that wider set
    /// (cheaper + more relevant, HippoRAG passage-community scoping), not a
    /// regression of it.
    fn retrieve(&self, seeds: &[String], rung: RungLevel, top_k: usize) -> Vec<ScoredId> {
        let mut hits: Vec<ScoredId> = match rung.as_u8() {
            // Rung 0–1 (base, FLOW) — 256-ranking: union each seed's similar set.
            0..=1 => {
                let mut acc: Vec<ScoredId> = Vec::new();
                for s in seeds {
                    acc.extend(self.similar_by_ranking(s, top_k));
                }
                acc
            }
            // Rung 2 — SPO-G hop: the 2-hop reachable set.
            2 => self.neighbours(seeds, 2),
            // Rung 3+ — community-scoped expansion: a wider walk, then keep the
            // hits that share a community with a seed (community picks the
            // subgraph; the real PPR ranks within it — override for ppr.rs).
            _ => {
                let hops = rung.as_u8().min(6);
                let wide = self.neighbours(seeds, hops);
                let seed_comms: std::collections::BTreeSet<u32> =
                    seeds.iter().filter_map(|s| self.community_of(s)).collect();
                if seed_comms.is_empty() {
                    wide
                } else {
                    wide.into_iter()
                        .filter(|h| {
                            self.community_of(&h.id)
                                .is_some_and(|c| seed_comms.contains(&c))
                        })
                        .collect()
                }
            }
        };
        dedup_best(&mut hits);
        hits.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(core::cmp::Ordering::Equal)
        });
        hits.truncate(top_k);
        hits
    }
}

/// Keep the highest-scoring entry per id (in place). Sorts by `(id asc, score
/// desc)` so the first of each id-run is its best, then drops the rest.
fn dedup_best(hits: &mut Vec<ScoredId>) {
    hits.sort_by(|a, b| {
        a.id.cmp(&b.id).then(
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(core::cmp::Ordering::Equal),
        )
    });
    hits.dedup_by(|a, b| a.id == b.id);
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::{HashMap, HashSet};

    /// A tiny in-memory graph implementing the read surface — the "AriGraph
    /// carrier" stand-in: adjacency + Louvain-style community labels + a
    /// similarity (ranking) table.
    struct MockDocGraph {
        adj: HashMap<String, Vec<String>>,
        community: HashMap<String, u32>,
        similar: HashMap<String, Vec<(String, f32)>>,
    }

    impl DocGraphQuery for MockDocGraph {
        fn community_of(&self, id: &str) -> Option<u32> {
            self.community.get(id).copied()
        }

        fn community_ids(&self) -> Vec<u32> {
            let mut ids: Vec<u32> = self.community.values().copied().collect();
            ids.sort_unstable();
            ids.dedup();
            ids
        }

        fn community_members(&self, community: u32) -> Vec<String> {
            let mut m: Vec<String> = self
                .community
                .iter()
                .filter_map(|(k, &c)| {
                    if c == community {
                        Some(k.clone())
                    } else {
                        None
                    }
                })
                .collect();
            m.sort();
            m
        }

        fn neighbours(&self, seeds: &[String], hops: u8) -> Vec<ScoredId> {
            // Deterministic BFS mirror of get_associated; score = 1/(1+depth).
            let mut seen: HashSet<String> = seeds.iter().cloned().collect();
            let mut frontier: Vec<String> = seeds.to_vec();
            let mut out: Vec<ScoredId> = Vec::new();
            for d in 1..=hops {
                let mut next: Vec<String> = Vec::new();
                for f in &frontier {
                    if let Some(ns) = self.adj.get(f) {
                        for n in ns {
                            if seen.insert(n.clone()) {
                                out.push(ScoredId::new(n.clone(), 1.0 / (1.0 + d as f32), d));
                                next.push(n.clone());
                            }
                        }
                    }
                }
                frontier = next;
                if frontier.is_empty() {
                    break;
                }
            }
            out
        }

        fn similar_by_ranking(&self, id: &str, top_k: usize) -> Vec<ScoredId> {
            let mut v: Vec<ScoredId> = self
                .similar
                .get(id)
                .map(|xs| {
                    xs.iter()
                        .map(|(o, s)| ScoredId::base(o.clone(), *s))
                        .collect()
                })
                .unwrap_or_default();
            v.sort_by(|a, b| {
                b.score
                    .partial_cmp(&a.score)
                    .unwrap_or(core::cmp::Ordering::Equal)
            });
            v.truncate(top_k);
            v
        }
    }

    /// Two triangle communities {a,b,c}=0 and {d,e,f}=1 joined by a single a–d
    /// bridge; a's similarity table lists a non-adjacent node `z` (the ranking
    /// vs walk discriminator).
    fn fixture() -> MockDocGraph {
        let mut adj: HashMap<String, Vec<String>> = HashMap::new();
        adj.insert("a".into(), vec!["b".into(), "c".into(), "d".into()]);
        adj.insert("b".into(), vec!["a".into(), "c".into()]);
        adj.insert("c".into(), vec!["a".into(), "b".into()]);
        adj.insert("d".into(), vec!["e".into(), "f".into(), "a".into()]);
        adj.insert("e".into(), vec!["d".into(), "f".into()]);
        adj.insert("f".into(), vec!["d".into(), "e".into()]);

        let mut community: HashMap<String, u32> = HashMap::new();
        for e in ["a", "b", "c"] {
            community.insert(e.into(), 0);
        }
        for e in ["d", "e", "f"] {
            community.insert(e.into(), 1);
        }

        let mut similar: HashMap<String, Vec<(String, f32)>> = HashMap::new();
        // `z` is similar to a but NOT a graph neighbour of a.
        similar.insert(
            "a".into(),
            vec![("b".into(), 0.9), ("c".into(), 0.8), ("z".into(), 0.5)],
        );
        similar.insert("b".into(), vec![("c".into(), 0.7), ("z".into(), 0.95)]);

        MockDocGraph {
            adj,
            community,
            similar,
        }
    }

    fn ids(hits: &[ScoredId]) -> HashSet<String> {
        hits.iter().map(|h| h.id.clone()).collect()
    }

    #[test]
    fn scored_id_constructors() {
        let d = ScoredId::new("x", 0.4, 2);
        assert_eq!(d.id, "x");
        assert_eq!(d.depth, 2);
        let b = ScoredId::base("y", 0.9);
        assert_eq!(b.depth, 0);
    }

    #[test]
    fn community_surface_round_trips() {
        let g = fixture();
        assert_eq!(g.community_of("a"), Some(0));
        assert_eq!(g.community_of("f"), Some(1));
        assert_eq!(g.community_of("nobody"), None);
        assert_eq!(g.community_ids(), vec![0, 1]);
        assert_eq!(g.community_members(0), vec!["a", "b", "c"]);
        assert_eq!(g.community_members(1), vec!["d", "e", "f"]);
    }

    #[test]
    fn neighbours_breadth_grows_with_hops() {
        let g = fixture();
        let seeds = vec!["a".to_string()];
        let one = ids(&g.neighbours(&seeds, 1));
        // 1 hop from a: b, c, d.
        assert_eq!(one, ["b", "c", "d"].iter().map(|s| s.to_string()).collect());
        let two = ids(&g.neighbours(&seeds, 2));
        // 2 hops also reaches e, f (via d).
        assert!(two.is_superset(&one));
        assert!(two.contains("e") && two.contains("f"));
    }

    #[test]
    fn similar_is_sorted_and_truncated() {
        let g = fixture();
        let top = g.similar_by_ranking("a", 2);
        assert_eq!(top.len(), 2);
        // Best-first: b (0.9) then c (0.8), z (0.5) truncated.
        assert_eq!(top[0].id, "b");
        assert_eq!(top[1].id, "c");
        assert!(top[0].score >= top[1].score);
    }

    #[test]
    fn retrieve_rung_base_is_ranking_not_walk() {
        // Rung 0–1 uses similar_by_ranking → surfaces `z` (similar but NOT a
        // graph neighbour). That `z` presence proves the base is the ranking
        // leg, not the SPO-G walk.
        let g = fixture();
        let seeds = vec!["a".to_string()];
        let base = ids(&g.retrieve(&seeds, RungLevel::Surface, 10));
        assert!(base.contains("z"), "base rung must use the ranking leg");
        assert!(!base.contains("e"), "base rung must NOT walk to 2-hop e");
    }

    #[test]
    fn retrieve_rung2_is_two_hop_walk() {
        // Rung 2 = SPO-G hop: reaches the cross-community d, e, f (the wide
        // walk), and never the ranking-only `z`.
        let g = fixture();
        let seeds = vec!["a".to_string()];
        let r2 = ids(&g.retrieve(&seeds, RungLevel::Contextual, 10));
        assert!(r2.contains("d") && r2.contains("e") && r2.contains("f"));
        assert!(!r2.contains("z"), "the walk is not the ranking leg");
    }

    #[test]
    fn retrieve_rung3_is_community_scoped() {
        // Rung 3 widens the walk THEN focuses to a's community {a,b,c}: the
        // cross-bridge d/e/f (community 1) are dropped. This is the community
        // focus that rung 2's raw walk does not apply.
        let g = fixture();
        let seeds = vec!["a".to_string()];
        let r3 = ids(&g.retrieve(&seeds, RungLevel::Analogical, 10));
        assert!(r3.contains("b") && r3.contains("c"));
        assert!(
            !r3.contains("d") && !r3.contains("e") && !r3.contains("f"),
            "rung 3 focuses to the seed community"
        );
    }

    #[test]
    fn retrieve_dedups_best_score_and_truncates() {
        // Seeds a and b both list `z` in their similar tables (a:0.5, b:0.95).
        // The union must keep ONE z at the best score (0.95), sorted best-first.
        let g = fixture();
        let seeds = vec!["a".to_string(), "b".to_string()];
        let hits = g.retrieve(&seeds, RungLevel::Surface, 10);
        let zs: Vec<&ScoredId> = hits.iter().filter(|h| h.id == "z").collect();
        assert_eq!(zs.len(), 1, "z must be deduped");
        assert!((zs[0].score - 0.95).abs() < 1e-6, "keeps the best score");
        // Sorted best-first.
        for w in hits.windows(2) {
            assert!(w[0].score >= w[1].score);
        }
        // top_k respected.
        let capped = g.retrieve(&seeds, RungLevel::Surface, 2);
        assert!(capped.len() <= 2);
    }

    #[test]
    fn retrieve_empty_seeds_is_safe() {
        let g = fixture();
        assert!(g.retrieve(&[], RungLevel::Contextual, 10).is_empty());
    }
}
