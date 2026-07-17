// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! `bm25` — classic Okapi BM25 lexical ranking over a small in-memory
//! document corpus.
//!
//! # Where this sits (doctrine)
//!
//! BM25 is the rung-0/1 **lexical** baseline: exact-term relevance scoring,
//! computed independently of the embedding/codec stack. It complements
//! rather than competes with CAM-PQ vector ranking — the retrieval
//! capstone (D-GR-2) is expected to consume both signals side by side
//! (lexical recall for exact-term queries, CAM-PQ for semantic recall), the
//! same way [`community`](super::community) complements episodic basins:
//! two independent signals that cross-validate rather than duplicate.
//!
//! # Algorithm
//!
//! Classic Okapi BM25 (Robertson & Walker 1994). Tokenization is
//! deliberately minimal: lowercase, split on any non-alphanumeric
//! character. Per-term weight is `idf(term) * saturated_tf(term, doc)`:
//!
//! - `idf = ln(1 + (N - df + 0.5) / (df + 0.5))` — standard (non-negative)
//!   Okapi idf, `N` = corpus size, `df` = document frequency of the term.
//! - `saturated_tf = tf * (k1 + 1) / (tf + k1 * (1 - b + b * dl / avgdl))`
//!   — term-frequency saturation with document-length normalization.
//! - `k1 = 1.2`, `b = 0.75` — the standard defaults from the original paper
//!   and most production BM25 implementations.
//!
//! Deterministic and dependency-free: per-document term frequencies are
//! accumulated via `BTreeMap` (stable iteration order) before landing in
//! the postings list, so two [`Bm25Index::build`] calls over the same
//! corpus produce identical indices, and [`Bm25Index::rank`] output is
//! stable across runs.

use std::collections::{BTreeMap, HashMap};

/// BM25 free parameter: term-frequency saturation rate.
const K1: f64 = 1.2;
/// BM25 free parameter: document-length normalization strength.
const B: f64 = 0.75;

/// A BM25-indexed corpus of small in-memory documents.
#[derive(Debug, Clone)]
pub struct Bm25Index {
    /// Number of documents in the corpus.
    n_docs: usize,
    /// Average document length in tokens (`0.0` for an empty corpus).
    avgdl: f64,
    /// Token count per document, indexed by `doc_id`.
    doc_len: Vec<usize>,
    /// Document frequency: term → number of documents containing it.
    df: HashMap<String, usize>,
    /// Inverted index: term → `(doc_id, term_frequency)`, `doc_id` ascending.
    postings: HashMap<String, Vec<(usize, u32)>>,
}

impl Bm25Index {
    /// Build a BM25 index from `docs`. Tokenization: lowercase, split on any
    /// non-alphanumeric character (Unicode-aware), empty tokens dropped.
    pub fn build(docs: &[&str]) -> Self {
        let n_docs = docs.len();
        let mut doc_len = Vec::with_capacity(n_docs);
        let mut df: HashMap<String, usize> = HashMap::new();
        let mut postings: HashMap<String, Vec<(usize, u32)>> = HashMap::new();

        for (doc_id, doc) in docs.iter().enumerate() {
            let tokens = tokenize(doc);
            doc_len.push(tokens.len());

            // BTreeMap: deterministic per-document term iteration order, so
            // the postings list for every term is built in doc_id order.
            let mut tf: BTreeMap<String, u32> = BTreeMap::new();
            for tok in tokens {
                *tf.entry(tok).or_insert(0) += 1;
            }
            for (term, count) in tf {
                *df.entry(term.clone()).or_insert(0) += 1;
                postings.entry(term).or_default().push((doc_id, count));
            }
        }

        let total_len: usize = doc_len.iter().sum();
        let avgdl = if n_docs == 0 {
            0.0
        } else {
            total_len as f64 / n_docs as f64
        };

        Self {
            n_docs,
            avgdl,
            doc_len,
            df,
            postings,
        }
    }

    /// Okapi BM25 score of `doc_id` for `query`. Query terms are tokenized
    /// the same way as documents and deduplicated (a repeated query term
    /// does not double-count). Returns `0.0` for an out-of-range `doc_id`,
    /// an empty query, or a query whose terms never occur in the corpus.
    pub fn score(&self, query: &str, doc_id: usize) -> f64 {
        if doc_id >= self.n_docs {
            return 0.0;
        }
        let dl = self.doc_len[doc_id] as f64;
        let mut query_terms = tokenize(query);
        query_terms.sort_unstable();
        query_terms.dedup();

        let mut total = 0.0;
        for term in &query_terms {
            let Some(&df) = self.df.get(term) else {
                continue;
            };
            let Some(postings) = self.postings.get(term) else {
                continue;
            };
            let tf = postings
                .binary_search_by_key(&doc_id, |&(d, _)| d)
                .ok()
                .map(|idx| f64::from(postings[idx].1))
                .unwrap_or(0.0);
            if tf == 0.0 {
                continue;
            }
            let denom = tf + K1 * (1.0 - B + B * dl / self.avgdl);
            let saturated_tf = tf * (K1 + 1.0) / denom;
            total += idf(self.n_docs, df) * saturated_tf;
        }
        total
    }

    /// All documents ranked by [`Self::score`] against `query`, descending;
    /// ties (equal score, including all-zero for an empty query or a query
    /// with no corpus matches) are broken by `doc_id` ascending.
    pub fn rank(&self, query: &str) -> Vec<(usize, f64)> {
        let mut scores: Vec<(usize, f64)> = (0..self.n_docs)
            .map(|doc_id| (doc_id, self.score(query, doc_id)))
            .collect();
        scores.sort_by(|a, b| {
            b.1.partial_cmp(&a.1)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.0.cmp(&b.0))
        });
        scores
    }
}

/// Standard (non-negative) Okapi idf: `ln(1 + (N - df + 0.5) / (df + 0.5))`.
fn idf(n_docs: usize, df: usize) -> f64 {
    (((n_docs as f64 - df as f64 + 0.5) / (df as f64 + 0.5)) + 1.0).ln()
}

/// Tokenize: lowercase, split on non-alphanumeric, drop empty pieces.
fn tokenize(text: &str) -> Vec<String> {
    let lower = text.to_lowercase();
    lower
        .split(|c: char| !c.is_alphanumeric())
        .filter(|s| !s.is_empty())
        .map(String::from)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn doc_with_query_terms_outranks_doc_without() {
        let docs = ["the quick brown fox", "lorem ipsum dolor sit amet"];
        let idx = Bm25Index::build(&docs);
        let ranked = idx.rank("quick fox");
        assert_eq!(
            ranked[0].0, 0,
            "doc 0 contains both query terms: {ranked:?}"
        );
        assert!(ranked[0].1 > ranked[1].1);
        assert_eq!(ranked[1].1, 0.0, "doc 1 contains neither query term");
    }

    #[test]
    fn rarer_term_has_higher_idf_than_common_term() {
        // 10-document corpus; a term found in 1 doc is rarer than one found
        // in 8, so it must carry more idf weight.
        assert!(
            idf(10, 1) > idf(10, 8),
            "idf(10,1)={} should exceed idf(10,8)={}",
            idf(10, 1),
            idf(10, 8)
        );
    }

    #[test]
    fn empty_query_yields_all_zero_ranking() {
        let docs = ["alpha beta", "gamma delta"];
        let idx = Bm25Index::build(&docs);
        let ranked = idx.rank("");
        assert_eq!(ranked.len(), 2);
        assert!(ranked.iter().all(|&(_, s)| s == 0.0));
        assert_eq!(
            ranked[0].0, 0,
            "ties broken by doc_id ascending: {ranked:?}"
        );
        assert_eq!(ranked[1].0, 1);
    }

    #[test]
    fn rank_is_deterministic() {
        let docs = [
            "alpha beta gamma",
            "beta gamma delta",
            "gamma delta epsilon",
        ];
        let idx = Bm25Index::build(&docs);
        assert_eq!(idx.rank("beta gamma"), idx.rank("beta gamma"));
    }

    #[test]
    fn empty_corpus_is_safe() {
        let idx = Bm25Index::build(&[]);
        assert!(idx.rank("anything").is_empty());
        assert_eq!(idx.score("anything", 0), 0.0);
    }
}
