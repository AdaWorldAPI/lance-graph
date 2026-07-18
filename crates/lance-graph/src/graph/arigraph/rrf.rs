// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! `rrf` — Reciprocal Rank Fusion (Cormack, Clarke & Büttcher, SIGIR 2009).
//!
//! Fuses N independently-ranked result lists into one ranking, scoring each id
//! by `Σ_lists 1 / (k + rank)` where `rank` is the **1-based position** of the
//! id in that list. RRF fuses by *rank position*, never by the source scores —
//! which is exactly why it combines lists whose scores are **not
//! commensurable**: [`Bm25Index::rank`](super::bm25::Bm25Index::rank) (tf-idf
//! `f64`), [`PersonalizedPageRank::ranked`](super::ppr::PersonalizedPageRank)
//! (unit-sum probability), and CAM-PQ (`i8` distance) share no scale, yet their
//! rank orders fuse cleanly. `k` (default [`DEFAULT_RRF_K`] = 60, the paper's
//! constant) damps deep-ranked items so a strong agreement near the top of
//! several lists dominates a lone top-1 in one list.
//!
//! This is the fusion primitive named as the D-GR-2 retrieval **keystone** in
//! `.claude/knowledge/graphrag-representations-inventory.md` (the SAP "Practical
//! GraphRAG" reader's headline gap: every ranked leg exists —
//! `Bm25Index`/`PersonalizedPageRank`/CAM-PQ — but nothing fused them). It is a
//! **pure, reversible** capability: it computes a fused ranking and reads no
//! carrier state. Wiring it into `OsintRetriever::retrieve` (so the retriever
//! actually fuses its legs) stays gated on the G0 load-bearing verdict — this
//! module only lands the algorithm, ahead of that gate, exactly as
//! `Bm25Index`/`PersonalizedPageRank`/`Communities` landed as pure capabilities.

use std::collections::BTreeMap;

use lance_graph_contract::doc_graph::ScoredId;

/// The paper's default rank-fusion constant (`k = 60`).
pub const DEFAULT_RRF_K: f64 = 60.0;

/// Fuse `ranked_lists` — each already ordered **best-first** — into one ranking
/// by Reciprocal Rank Fusion.
///
/// Each id's fused score is `Σ_lists 1 / (k + rank)` with `rank` 1-based. The
/// returned [`ScoredId`]s are sorted by fused score **descending**, ties broken
/// by id **ascending** (deterministic). `depth` carries the *shallowest* depth
/// the id appeared at across the inputs (strongest provenance wins); an id
/// absent from every list simply does not appear.
///
/// Because fusion is by RANK, the per-list [`ScoredId::score`] values need not
/// be commensurable — the reason RRF can combine the BM25 / PPR / CAM-PQ legs,
/// whose scores live on unrelated scales.
///
/// `k` should be positive; the paper uses `60` ([`DEFAULT_RRF_K`]). An empty
/// `ranked_lists` (or all-empty lists) yields an empty result.
///
/// # Examples
/// ```
/// use lance_graph::graph::arigraph::rrf::{reciprocal_rank_fusion, DEFAULT_RRF_K};
/// use lance_graph_contract::doc_graph::ScoredId;
///
/// // `x` is rank-1 in BOTH lists; `y`/`z` are rank-2 in only one each.
/// let list_a = [ScoredId::new("x", 1.0, 0), ScoredId::new("y", 1.0, 0)];
/// let list_b = [ScoredId::new("x", 1.0, 0), ScoredId::new("z", 1.0, 0)];
/// let fused = reciprocal_rank_fusion(&[&list_a, &list_b], DEFAULT_RRF_K);
///
/// assert_eq!(fused.len(), 3);
/// assert_eq!(fused[0].id, "x"); // consensus at the top wins
/// ```
#[must_use]
pub fn reciprocal_rank_fusion(ranked_lists: &[&[ScoredId]], k: f64) -> Vec<ScoredId> {
    // id -> (accumulated RRF score, shallowest depth seen). BTreeMap gives a
    // deterministic id-ascending iteration order, which the stable sort below
    // preserves within equal fused scores.
    let mut acc: BTreeMap<&str, (f64, u8)> = BTreeMap::new();
    for list in ranked_lists {
        for (pos, item) in list.iter().enumerate() {
            let rank = pos as f64 + 1.0; // 1-based
            let entry = acc.entry(item.id.as_str()).or_insert((0.0, u8::MAX));
            entry.0 += 1.0 / (k + rank);
            entry.1 = entry.1.min(item.depth);
        }
    }
    let mut fused: Vec<ScoredId> = acc
        .into_iter()
        .map(|(id, (score, depth))| {
            ScoredId::new(id, score as f32, if depth == u8::MAX { 0 } else { depth })
        })
        .collect();
    // Fused score descending; `sort_by` is stable, so the BTreeMap's id-ascending
    // order breaks ties deterministically.
    fused.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    fused
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ids(v: &[ScoredId]) -> Vec<&str> {
        v.iter().map(|s| s.id.as_str()).collect()
    }

    #[test]
    fn consensus_near_top_beats_lone_top_one() {
        // `x` rank-1 in both; `y`,`z` rank-2 in one each.
        let a = [ScoredId::new("x", 1.0, 0), ScoredId::new("y", 1.0, 0)];
        let b = [ScoredId::new("x", 1.0, 0), ScoredId::new("z", 1.0, 0)];
        let fused = reciprocal_rank_fusion(&[&a, &b], DEFAULT_RRF_K);
        assert_eq!(fused.len(), 3);
        assert_eq!(fused[0].id, "x");
    }

    #[test]
    fn fuses_incommensurable_scores_by_rank_only() {
        // Wildly different score scales; only the RANK order matters.
        let bm25 = [ScoredId::new("a", 9000.0, 0), ScoredId::new("b", 4000.0, 0)];
        let ppr = [ScoredId::new("b", 0.51, 1), ScoredId::new("a", 0.49, 2)];
        let fused = reciprocal_rank_fusion(&[&bm25, &ppr], DEFAULT_RRF_K);
        // a: 1/61 + 1/62 ; b: 1/62 + 1/61 — equal → deterministic id-asc order.
        assert_eq!(ids(&fused), ["a", "b"]);
        // score is symmetric, not dominated by bm25's huge raw magnitudes.
        assert!((fused[0].score - fused[1].score).abs() < 1e-6);
    }

    #[test]
    fn shallowest_depth_wins() {
        let a = [ScoredId::new("a", 1.0, 3)];
        let b = [ScoredId::new("a", 1.0, 1)];
        let fused = reciprocal_rank_fusion(&[&a, &b], DEFAULT_RRF_K);
        assert_eq!(fused.len(), 1);
        assert_eq!(fused[0].depth, 1); // min(3, 1)
    }

    #[test]
    fn rank_position_dominates_within_one_list() {
        let only = [
            ScoredId::new("first", 0.1, 0),
            ScoredId::new("second", 0.1, 0),
            ScoredId::new("third", 0.1, 0),
        ];
        let fused = reciprocal_rank_fusion(&[&only], DEFAULT_RRF_K);
        assert_eq!(ids(&fused), ["first", "second", "third"]);
        // strictly decreasing: 1/61 > 1/62 > 1/63
        assert!(fused[0].score > fused[1].score && fused[1].score > fused[2].score);
    }

    #[test]
    fn smaller_k_sharpens_top_rank_advantage() {
        // A rank-1 hit vs a rank-10 hit: smaller k widens their score ratio.
        let mk = |k: f64| {
            let a = [ScoredId::new("top", 1.0, 0)];
            let mut deep: Vec<ScoredId> =
                (0..10).map(|i| ScoredId::new(format!("d{i}"), 1.0, 0)).collect();
            deep[9] = ScoredId::new("low", 1.0, 0); // "low" at rank 10
            let f = reciprocal_rank_fusion(&[&a, &deep], k);
            let top = f.iter().find(|s| s.id == "top").unwrap().score;
            let low = f.iter().find(|s| s.id == "low").unwrap().score;
            top / low
        };
        assert!(mk(10.0) > mk(60.0)); // smaller k → bigger top-vs-deep ratio
    }

    #[test]
    fn empty_inputs_are_safe() {
        assert!(reciprocal_rank_fusion(&[], DEFAULT_RRF_K).is_empty());
        let empty: [ScoredId; 0] = [];
        assert!(reciprocal_rank_fusion(&[&empty, &empty], DEFAULT_RRF_K).is_empty());
    }

    #[test]
    fn deterministic() {
        let a = [ScoredId::new("a", 1.0, 0), ScoredId::new("b", 1.0, 0)];
        let b = [ScoredId::new("c", 1.0, 0), ScoredId::new("a", 1.0, 0)];
        let x = reciprocal_rank_fusion(&[&a, &b], DEFAULT_RRF_K);
        let y = reciprocal_rank_fusion(&[&a, &b], DEFAULT_RRF_K);
        assert_eq!(ids(&x), ids(&y));
        assert_eq!(
            x.iter().map(|s| s.score).collect::<Vec<_>>(),
            y.iter().map(|s| s.score).collect::<Vec<_>>()
        );
    }
}
